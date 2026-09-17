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
# WHY ONLY MPI AND KOKKOS. For NVSHMEM the honest answer is: cross-container
# does not work here and FOUR plausible causes have each been TESTED AND
# RULED OUT. Recorded so the next person does not re-run them.
#
#   RDMA?           No. NVSHMEM ships ucx and libfabric transports beside the
#                   ib* ones. libfabric gets past transport selection over
#                   plain TCP, so a fabric is not required.
#   GPU per node?   Not by itself. The famous error
#                     [GPU 0] Peer GPU 1 is not accessible
#                   is CLEARED by NVSHMEM_DISABLE_P2P=1, with one physical
#                   GPU behind both containers.
#   UCX too old?    No. The image ships 1.16 and NVSHMEM 3.7 warns it wants
#                   >= 1.19 -- so UCX 1.19.0 was BUILT and mounted at
#                   /opt/ucx. The version warning goes away and the failure
#                   is bit-for-bit identical.
#   ipc: host?      No. Sharing an IPC namespace looked like it would make
#                   UCX believe a peer was shm-reachable; ipc: private
#                   changes nothing.
#
# WHAT ACTUALLY HAPPENS, unchanged through all four:
#     no active messages transport to <no debug data>:
#         posix/memory - Destination is unreachable
#     Failed to connect endpoint in UCX transport / connect EPS failed
#
# NVSHMEM's ucx module selects posix/memory and ignores UCX_TLS entirely, and
# the empty "<no debug data>" peer suggests the worker address it exchanged
# through its bootstrap is not usable. That is inside NVSHMEM, past what an
# image change reaches. NOT SOLVED, and no longer guessed at.
#
# NCCL is separate and simple: one GPU PER RANK, cannot share a device, says
# so itself, exits 77 for ctest to skip.
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
export KOKKOS_HOME="${KOKKOS_HOME:-$HOME/opt/kokkos-cuda}"
# distributed_md/ supplies the cluster; the overlay adds the Kokkos mount.
BL_RANKS="${BL_RANKS:-2}"
NODES="md-node1 md-node2"
COMPOSE=(docker compose -f "$MD_DIR/docker-compose.yml"
                        -f "$SCRIPT_DIR/docker-compose.kokkos.yml")
if [ "$BL_RANKS" -ge 3 ]; then
  COMPOSE+=(-f "$SCRIPT_DIR/docker-compose.four.yml")
  NODES="$NODES md-node3"
fi
[ "$BL_RANKS" -ge 4 ] && NODES="$NODES md-node4"
# --host wants one slot per container.
HOSTARG="$(echo $NODES | tr ' ' '\n' | sed 's/$/:1/' | paste -sd,)"
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
    lammps_md)
      # The z-slab decomposition needs >= 3 planes per rank, and --lattice 20
      # gives 11 -- enough for 2 ranks, one short for 4. The bench says so
      # rather than guessing, so scale the deck instead of loosening it.
      if [ "$BL_RANKS" -ge 4 ]; then ARGS="--lattice 32 --steps 20"
      else ARGS="--lattice 20 --steps 20"; fi ;;
    *) echo "unknown workload: $1" >&2; return 2 ;;
  esac
}

cleanup() {
  ( cd "$MD_DIR" && "${COMPOSE[@]}" down -v --remove-orphans >/dev/null 2>&1 ) || true
  [ -n "${MD_SSH_DIR:-}" ] && rm -rf "$MD_SSH_DIR"
}
trap cleanup EXIT

echo "== bringing up the 2-container cluster (distributed_md/)"
( cd "$MD_DIR" && "${COMPOSE[@]}" down -v --remove-orphans >/dev/null 2>&1 ) || true
( cd "$MD_DIR" && "${COMPOSE[@]}" up -d $NODES >/dev/null )

# sshd is up when node1 can actually open a connection to node2 -- a `ready`
# file only says the process started.
for peer in $(echo $NODES | tr " " "\n" | grep -v md-node1); do
  for _ in $(seq 1 60); do
    docker exec md-node1 ssh $SSH_OPTS -o ConnectTimeout=2 "$peer" true >/dev/null 2>&1 && break
    sleep 1
  done
  if ! docker exec md-node1 ssh $SSH_OPTS -o ConnectTimeout=2 "$peer" true >/dev/null 2>&1; then
    echo "!! node1 cannot ssh to $peer; aborting"; exit 1
  fi
done
echo "== cluster up"

rc=0
run_one() {
  local wl="$1"; deck "$wl" || return 2
  local src="clio_${wl}_mpi_bench.cc" bin="clio_${wl}_mpi_bench"
  echo
  echo "== $wl : MPI, $BL_RANKS ranks across $NODES"

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
    local pids=""
  for nd in $NODES; do
    docker exec "$nd" bash -c "$compile" >/tmp/bl_${wl}_${nd}.log 2>&1 &
    pids="$pids $!"
  done
  local ok=1; for p in $pids; do wait $p || ok=0; done
  if [ $ok -eq 0 ]; then
    echo "   !! compile failed"; tail -5 /tmp/bl_${wl}_md-node*.log
    rc=1; return 1
  fi

  local log="/tmp/bl_${wl}.log"
  docker exec md-node1 bash -c "
    mpirun --allow-run-as-root -n $BL_RANKS --host $HOSTARG \
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
  if ! grep -qE "(ranks|PEs)=$BL_RANKS|\b$BL_RANKS (ranks|PEs)\b" "$log"; then
    echo "   !! $wl: ran, but not as $BL_RANKS ranks"; rc=1; return 1
  fi
  echo "   $wl OK ($BL_RANKS ranks, separate containers)"
}

# The Kokkos baselines run cross-container too, but from the HOST BUILD rather
# than compiled in-container: building Kokkos inside each node would mean
# shipping a toolchain and a multi-minute build per node. Safe only because the
# container's Open MPI is the same 4.1.6 as the host's -- checked below, because
# an ABI mismatch here would surface as a corrupt result rather than a crash.
run_kokkos() {
  local wl="$1"; deck "$wl" || return 2
  local bin="/workspace/${BUILD_DIR:-build-gv}/bin/clio_${wl}_kokkos_bench"
  echo
  echo "== $wl : Kokkos, $BL_RANKS ranks across $NODES"
  if ! docker exec md-node1 test -x "$bin"; then
    echo "   .. not built ($bin); skipping"; return 0
  fi
  local log="/tmp/blk_${wl}.log"
  docker exec md-node1 bash -c "
    mpirun --allow-run-as-root -n $BL_RANKS --host $HOSTARG \
        --mca plm_rsh_agent 'ssh $SSH_OPTS' \
        --mca btl tcp,self --mca btl_tcp_if_include eth0 \
        -x LD_LIBRARY_PATH=/opt/kokkos/lib:/opt/kokkos/lib64:/usr/local/cuda/lib64 \
        $bin $ARGS 2>&1
  " > "$log" 2>&1
  local code=$?
  sed 's/^/   | /' "$log" | tail -6
  if [ $code -ne 0 ]; then echo "   !! $wl kokkos exited $code"; rc=1; return 1; fi
  if ! grep -qE 'ALL GATES PASS|GATE: PASS|PASS' "$log"; then
    echo "   !! $wl kokkos: no passing gate line"; rc=1; return 1
  fi
  echo "   $wl kokkos OK ($BL_RANKS ranks, separate containers)"
}

# An Open MPI mismatch would corrupt results rather than crash, so it is
# checked rather than assumed.
HOST_MPI="$(mpirun --version 2>/dev/null | head -1)"
CONT_MPI="$(docker exec md-node1 mpirun --version 2>/dev/null | head -1)"
if [ "$HOST_MPI" != "$CONT_MPI" ]; then
  echo "!! Open MPI differs (host: $HOST_MPI / container: $CONT_MPI)"
  echo "   the prebuilt Kokkos binaries cannot be trusted across that; skipping them"
  KOKKOS_OK=0
else
  KOKKOS_OK=1
fi

TARGET="${1:-all}"
if [ "$TARGET" = all ]; then
  for wl in kmeans weights gmx grayscott lbann lammps_md; do run_one "$wl" || true; done
  [ "$KOKKOS_OK" = 1 ] && for wl in kmeans weights gmx grayscott lbann lammps_md; do run_kokkos "$wl" || true; done
else
  run_one "$TARGET" || true
  [ "$KOKKOS_OK" = 1 ] && { run_kokkos "$TARGET" || true; }
fi
echo
echo "== cross-container MPI baselines: $([ $rc -eq 0 ] && echo ALL PASS || echo FAILURES)"
exit $rc
