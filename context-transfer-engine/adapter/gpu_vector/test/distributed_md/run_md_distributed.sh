#!/bin/bash
# 2-node distributed eternia-MD test (docker): the MPI and NVSHMEM benches
# run as real multi-rank jobs. MPI always runs across the two containers; the
# NVSHMEM placement follows the GPU count (see the note above the run calls --
# on a single-GPU host, cross-container NVSHMEM is not a configuration that
# exists, so its two PEs run inside one container instead).
#
# What it gates -- and deliberately does NOT gate:
#   * GATED: that both transports still produce the SAME physics when the
#     ranks are separate containers on a real network -- step-0 statics
#     against a host double reference, resort continuity, NVE drift -- and
#     that the distributed paths are actually exercised (a nonzero halo /
#     staging count and a nonzero migrant count, asserted below; without
#     those the run could pass while doing nothing distributed at all).
#   * NOT GATED: speed. Two ranks time-slice one GPU here, and NVSHMEM
#     additionally runs in "limited MPG" mode with no peer mapping, so wall
#     time from this harness is meaningless. Timing lives in the benchmarks.
#
# Both benches are COMPILED INSIDE the container rather than mounted from the
# host build, which is what makes the MPI ABI match by construction and makes
# this script double as the compile gate for the two files.
#
# Requires: nvidia container toolkit, an NVSHMEM prefix (NVSHMEM_HOME), a CUDA
# toolkit on the host (CUDA_HOME), and the iowarp/deps-cpu image.
#
#   NVSHMEM_HOME=$HOME/opt/nvshmem-pkg/nvidia/nvshmem ./run_md_distributed.sh
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../../" && pwd)"
export HOST_WORKSPACE="${HOST_WORKSPACE:-$REPO_ROOT}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
# The ssh key exchange needs a directory the containers' (host) UID can write.
export MD_SSH_DIR="${MD_SSH_DIR:-$(mktemp -d -t md-ssh-XXXXXX)}"
mkdir -p "$MD_SSH_DIR"
: "${NVSHMEM_HOME:?set NVSHMEM_HOME to an NVSHMEM prefix (include/ and lib/)}"
export NVSHMEM_HOME

# Small deck: two ranks need at least three z-planes each, and --lattice 20
# gives 11 planes (5/6 per rank) at 32k atoms. Hot enough (T=3) and rebinned
# often enough that atoms really do cross the slab boundary.
#
# 50 STEPS, NOT 20, and the step count is a correctness parameter here rather
# than a runtime knob. The melt starts from a perfect FCC lattice, so the
# first steps are a violent transient in which the NVE drift RATIO is at its
# worst: the same deck measures 7.46e-3 over 20 steps and 3.97e-3 over 50, at
# one rank and at two alike. Gating on the transient would mean loosening the
# tolerance to hide it; gating on the settled window keeps the documented
# hot-melt bound (5e-3, itself larger than stock LAMMPS's own 1.9e-3 in
# double) meaningful.
MD_ARCH="${MD_ARCH:-sm_89}"
MD_RANKS=2
MD_ARGS="${MD_ARGS:---md --lattice 20 --steps 50 --rebin 10 --temp 3.0 --cap 48 --blocks 128 --threads 64 --rowchunk 1 --drift-tol 5e-3}"
BENCH_DIR=context-transfer-engine/adapter/gpu_vector/benchmark

SSH_OPTS="-p 2222 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

cd "$SCRIPT_DIR"
cleanup() {
  docker compose down -v --remove-orphans >/dev/null 2>&1 || true
  rm -rf "$MD_SSH_DIR" 2>/dev/null || true
}
trap cleanup EXIT
# Pre-clean the CLUSTER ONLY. Calling cleanup() here would delete the ssh
# directory that was just created, and Docker then recreates the missing bind
# path as ROOT -- after which the unprivileged containers cannot write the key
# and every login falls back to password auth.
docker compose down -v --remove-orphans >/dev/null 2>&1 || true

echo "== bringing up $MD_RANKS nodes"
docker compose up -d || exit 1

# sshd is up when the node touches /tmp/ready AND accepts a connection.
for n in md-node1 md-node2; do
  for _ in $(seq 1 60); do
    docker exec "$n" test -f /tmp/ready >/dev/null 2>&1 && break
    sleep 1
  done
done
for _ in $(seq 1 60); do
  docker exec md-node1 ssh $SSH_OPTS -o ConnectTimeout=2 md-node2 true >/dev/null 2>&1 && break
  sleep 1
done
if ! docker exec md-node1 ssh $SSH_OPTS -o ConnectTimeout=2 md-node2 true; then
  echo "!! node1 cannot ssh to node2; aborting"
  docker compose logs --no-color | tail -40
  exit 1
fi

echo "== compiling both benches inside each container (parallel)"
# EVERY node compiles its own copy into container-local /tmp. The workspace
# is mounted read-only (root containers must not litter the repo), so there
# is nowhere shared to put a binary -- and compiling per node is also what
# guarantees each rank runs a binary built against its own MPI.
compile_on() {
  docker exec "$1" bash -c '
    set -e
    mkdir -p /tmp/mdbin
    cd /workspace/'"$BENCH_DIR"'
    /usr/local/cuda/bin/nvcc -x cu -std=c++17 -O2 -arch='"$MD_ARCH"' \
        -DMD_NVSHMEM_USE_MPI -I/opt/nvshmem/include \
        -I/usr/lib/x86_64-linux-gnu/openmpi/include \
        -rdc=true clio_lammps_md_nvshmem_bench.cc -o /tmp/mdbin/clio_lammps_md_nvshmem_bench \
        -L/opt/nvshmem/lib -lnvshmem_device -lnvshmem_host \
        -Xlinker /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so
    /usr/local/cuda/bin/nvcc -x cu -std=c++17 -O2 -arch='"$MD_ARCH"' \
        -I/usr/lib/x86_64-linux-gnu/openmpi/include \
        clio_lammps_md_mpi_bench.cc -o /tmp/mdbin/clio_lammps_md_mpi_bench \
        -Xlinker /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so
  '
}
compile_on md-node1 & p1=$!
compile_on md-node2 & p2=$!
wait $p1 || { echo "!! compile failed on md-node1"; exit 1; }
wait $p2 || { echo "!! compile failed on md-node2"; exit 1; }
echo "== compile OK"

rc=0
run_one() {
  local name="$1" bin="$2" want_dist="$3"
  echo
  echo "== $name : $MD_RANKS ranks across md-node1,md-node2"
  local log="/tmp/md_${name}.log"
  docker exec md-node1 bash -c "
    mpirun --allow-run-as-root -n $MD_RANKS --host md-node1:1,md-node2:1 \
        --mca plm_rsh_agent 'ssh $SSH_OPTS' \
        --mca btl tcp,self --mca btl_tcp_if_include eth0 \
        -x LD_LIBRARY_PATH -x NVSHMEM_REMOTE_TRANSPORT \
        /tmp/mdbin/$bin $MD_ARGS 2>&1
  " > "$log" 2>&1
  local code=$?
  sed 's/^/   | /' "$log"
  if [ $code -ne 0 ]; then
    echo "   !! $name exited $code"
    rc=1
    return
  fi
  # Every gate must be present AND passing. Grepping for PASS alone would let
  # a run that silently skipped a gate through.
  local g
  for g in "STATICS GATE: PASS" "RESORT GATE: PASS" "NVE GATE: PASS"; do
    if ! grep -q "$g" "$log"; then
      echo "   !! missing or failed: $g"
      rc=1
    fi
  done
  # And the distributed path must actually have run. A 2-rank job that
  # migrated nothing and staged nothing would pass every physics gate while
  # proving nothing about the decomposition.
  if ! grep -qE "$want_dist" "$log"; then
    echo "   !! the distributed path did not run (no match for: $want_dist)"
    rc=1
  fi
  local mig
  mig=$(grep -oE "migrants [0-9]+" "$log" | head -1 | awk '{print $2}')
  [ -z "$mig" ] && mig=$(grep -oE "migrants ([0-9.]+) atoms" "$log" | head -1 | awk '{print $2}')
  if [ -z "$mig" ] || [ "${mig%.*}" -eq 0 ] 2>/dev/null; then
    echo "   !! no atoms migrated between ranks -- migration path untested"
    rc=1
  else
    echo "   .. migrants across ranks: $mig"
  fi
}

# HOW EACH TRANSPORT IS PLACED, and why they are not placed the same way.
#
# NVSHMEM decides node identity by HOSTNAME. Two containers therefore look
# like two NODES to it, and two nodes each holding a GPU can only be joined
# by a remote transport -- IB, UCX or libfabric. On a host with ONE GPU and
# no RDMA fabric that configuration is impossible, and NVSHMEM says so
# outright: "[GPU 0] Peer GPU 1 is not accessible ... building transport map
# failed". That is not something to work around by lying about hostnames; it
# is the correct answer for the hardware.
#
# So placement follows the hardware:
#   >= 2 GPUs : both transports run across the two containers, one GPU each.
#               This is the real distributed configuration and the one a
#               proper CI host should provide.
#   == 1 GPU  : MPI still runs across both containers (two-sided messaging
#               over TCP does not care that the ranks share a GPU), while
#               NVSHMEM runs its two PEs inside ONE container -- the same
#               single-node, shared-GPU configuration that works on a
#               workstation. Every distributed code path is still exercised:
#               slab decomposition, peer staging, remote atomics, remote puts
#               and cross-rank migration. Only the container boundary moves.
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
echo
echo "== host GPUs visible: ${NGPU:-0}"

run_one mpi clio_lammps_md_mpi_bench 'halo .* in [1-9][0-9]* exchanges'

if [ "${NGPU:-0}" -ge 2 ]; then
  run_one nvshmem clio_lammps_md_nvshmem_bench 'spans [1-9][0-9]* staged'
else
  echo
  echo "== nvshmem : $MD_RANKS PEs inside md-node1 (one GPU on this host)"
  echo "   .. cross-container NVSHMEM needs a GPU per node. It does NOT"
  echo "      need RDMA -- ucx/libfabric run over TCP -- but this image"
  echo "      ships UCX 1.16 and NVSHMEM 3.7 wants UCP >= 1.19, so the"
  echo "      ucx transport falls back to shm and cannot cross a"
  echo "      container. With one GPU the single-node placement is the only"
  echo "      valid one. Run this harness on a multi-GPU host for the"
  echo "      across-containers configuration."
  log=/tmp/md_nvshmem.log
  docker exec md-node1 bash -c "
    mpirun --allow-run-as-root -n $MD_RANKS \
        -x LD_LIBRARY_PATH -x NVSHMEM_REMOTE_TRANSPORT \
        /tmp/mdbin/clio_lammps_md_nvshmem_bench $MD_ARGS 2>&1
  " > "$log" 2>&1
  code=$?
  sed 's/^/   | /' "$log"
  if [ $code -ne 0 ]; then
    echo "   !! nvshmem exited $code"
    rc=1
  else
    for g in "STATICS GATE: PASS" "RESORT GATE: PASS" "NVE GATE: PASS"; do
      grep -q "$g" "$log" || { echo "   !! missing or failed: $g"; rc=1; }
    done
    grep -qE 'spans [1-9][0-9]* staged' "$log" || {
      echo "   !! nothing was staged from a peer -- remote path untested"; rc=1; }
    mig=$(grep -oE "migrants [0-9]+" "$log" | head -1 | awk '{print $2}')
    if [ -z "$mig" ] || [ "$mig" -eq 0 ] 2>/dev/null; then
      echo "   !! no atoms migrated between PEs -- migration path untested"
      rc=1
    else
      echo "   .. migrants across PEs: $mig"
    fi
  fi
fi

echo
if [ $rc -eq 0 ]; then
  if [ "${NGPU:-0}" -ge 2 ]; then
    echo "== DISTRIBUTED MD: PASS (both transports, $MD_RANKS ranks, 2 containers)"
  else
    echo "== DISTRIBUTED MD: PASS (mpi across 2 containers; nvshmem $MD_RANKS PEs in one -- single-GPU host)"
  fi
else
  echo "== DISTRIBUTED MD: FAIL"
fi
exit $rc
