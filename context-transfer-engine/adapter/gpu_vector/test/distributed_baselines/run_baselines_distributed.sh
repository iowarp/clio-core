#!/bin/bash
# The MPI baselines as REAL cross-container jobs, for every workload.
#
#   ./run_baselines_distributed.sh {kmeans|weights|gmx|grayscott|lbann|lammps_md|all}
#
# WHAT THIS ADDS OVER `ctest -L distributed`. Those entries run the baselines
# at 2 ranks on ONE host: that exercises the parallel path and catches rot, but
# both ranks share a kernel, a filesystem and a loopback. Here the ranks are
# separate containers on a real network, so the transport is actually a
# transport. It is the same thing distributed_md/ does for eternia-MD, applied
# to the other five workloads.
#
# WHY ONLY MPI. NVSHMEM cross-container needs a GPU per node plus an RDMA
# fabric -- with one GPU the only valid placement is both PEs inside one
# container, which distributed_md/ already documents and does. NCCL needs a GPU
# per rank and cannot share a device at all. Neither is a software gap this
# script could close, so neither is pretended at.
#
# THE CLUSTER IS distributed_md/'s. It already has sshd, key exchange and an
# mpirun that works across containers; duplicating that would mean maintaining
# two copies of the fiddliest part.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MD_DIR="$SCRIPT_DIR/../distributed_md"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../../" && pwd)"
export HOST_WORKSPACE="${HOST_WORKSPACE:-$REPO_ROOT}"
export HOST_UID=$(id -u) HOST_GID=$(id -g)
export MD_SSH_DIR="${MD_SSH_DIR:-$(mktemp -d -t bl-ssh-XXXXXX)}"
chmod 777 "$MD_SSH_DIR" 2>/dev/null || true
# The md compose mounts an NVSHMEM prefix even though this script only builds
# the MPI baselines -- it is a required interpolation, so supply a default
# rather than fail at `compose up` with a message about a variable the
# caller has no reason to think is relevant.
export NVSHMEM_HOME="${NVSHMEM_HOME:-$HOME/opt/nvshmem-pkg/nvidia/nvshmem}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
ARCH="${BL_ARCH:-sm_89}"
BENCH_DIR=context-transfer-engine/adapter/gpu_vector/benchmark
# PORT 2222, not 22. The md compose starts sshd with -p 2222, so a default
# ssh invocation gets "Connection refused" from a container whose sshd is
# demonstrably up and ready -- which reads like a startup race and is not.
SSH_OPTS="-p 2222 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i /shared-ssh/id_rsa"

# Per-workload deck. Kept small: this gates the transport, not throughput.
deck() {
  case "$1" in
    kmeans)    ARGS="--data-mb 64 --iters 3" ;;
    weights)   ARGS="" ;;
    gmx)       ARGS="--k 32 --atoms 20000" ;;
    grayscott) ARGS="" ;;
    lbann)     ARGS="" ;;
    lammps_md) ARGS="--lattice 20 --steps 20" ;;
    *) echo "unknown workload: $1" >&2; return 2 ;;
  esac
}

cleanup() {
  ( cd "$MD_DIR" && docker compose down -v --remove-orphans >/dev/null 2>&1 ) || true
  [ -n "${MD_SSH_DIR:-}" ] && rm -rf "$MD_SSH_DIR"
}
trap cleanup EXIT

echo "== bringing up the 2-container cluster (distributed_md/)"
( cd "$MD_DIR" && docker compose down -v --remove-orphans >/dev/null 2>&1 ) || true
( cd "$MD_DIR" && docker compose up -d md-node1 md-node2 >/dev/null )

# sshd is up when node1 can actually open a connection to node2 -- a `ready`
# file only says the process started.
for _ in $(seq 1 60); do
  docker exec md-node1 ssh $SSH_OPTS -o ConnectTimeout=2 md-node2 true >/dev/null 2>&1 && break
  sleep 1
done
if ! docker exec md-node1 ssh $SSH_OPTS -o ConnectTimeout=2 md-node2 true >/dev/null 2>&1; then
  echo "!! node1 cannot ssh to node2; aborting"; exit 1
fi
echo "== cluster up"

rc=0
run_one() {
  local wl="$1"; deck "$wl" || return 2
  local src="clio_${wl}_mpi_bench.cc" bin="clio_${wl}_mpi_bench"
  echo
  echo "== $wl : MPI, 2 ranks across md-node1,md-node2"

  # EVERY node compiles its own copy into container-local /tmp: the workspace
  # is mounted READ-ONLY, and per-node compilation is what makes each rank's
  # binary match its own MPI by construction.
  local compile='
    set -e
    mkdir -p /tmp/blbin
    cd /workspace/'"$BENCH_DIR"'/'"$wl"'
    /usr/local/cuda/bin/nvcc -x cu -std=c++17 -O2 -arch='"$ARCH"' \
        -I/usr/lib/x86_64-linux-gnu/openmpi/include \
        '"$src"' -o /tmp/blbin/'"$bin"' \
        -Xlinker /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so
  '
  docker exec md-node1 bash -c "$compile" >/tmp/bl_${wl}_c1.log 2>&1 & local p1=$!
  docker exec md-node2 bash -c "$compile" >/tmp/bl_${wl}_c2.log 2>&1 & local p2=$!
  if ! wait $p1 || ! wait $p2; then
    echo "   !! compile failed"; tail -5 /tmp/bl_${wl}_c1.log /tmp/bl_${wl}_c2.log
    rc=1; return 1
  fi

  local log="/tmp/bl_${wl}.log"
  docker exec md-node1 bash -c "
    mpirun --allow-run-as-root -n 2 --host md-node1:1,md-node2:1 \
        --mca plm_rsh_agent 'ssh $SSH_OPTS' \
        --mca btl tcp,self --mca btl_tcp_if_include eth0 \
        -x LD_LIBRARY_PATH \
        /tmp/blbin/$bin $ARGS 2>&1
  " > "$log" 2>&1
  local code=$?
  sed 's/^/   | /' "$log" | tail -8
  if [ $code -ne 0 ]; then echo "   !! $wl exited $code"; rc=1; return 1; fi
  # The bench gates itself; require the PASS and require it to have actually
  # seen two ranks. Exit 0 alone would also be produced by a silent fallback
  # to one rank, which is the failure worth guarding against here.
  if ! grep -qE 'ALL GATES PASS|GATE: PASS' "$log"; then
    echo "   !! $wl: no passing gate line"; rc=1; return 1
  fi
  # BOTH SPELLINGS. These benches are not consistent: kmeans and lammps_md
  # print "ranks=2", the other four print "2 ranks". Matching only the first
  # form reported four passing cross-container runs as failures.
  if ! grep -qE '(ranks|PEs)=2|\b2 (ranks|PEs)\b' "$log"; then
    echo "   !! $wl: ran, but not as 2 ranks"; rc=1; return 1
  fi
  echo "   $wl OK (2 ranks, separate containers)"
}

TARGET="${1:-all}"
if [ "$TARGET" = all ]; then
  for wl in kmeans weights gmx grayscott lbann lammps_md; do run_one "$wl" || true; done
else
  run_one "$TARGET" || true
fi
echo
echo "== cross-container MPI baselines: $([ $rc -eq 0 ] && echo ALL PASS || echo FAILURES)"
exit $rc
