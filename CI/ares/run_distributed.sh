#!/bin/bash
# Bare-metal multi-node suites for the nightly Ares run. Called INSIDE a Slurm
# allocation (salloc -N4 ... bash CI/ares/run_distributed.sh), on the head
# node; everything on the compute nodes is launched with `srun --overlap`.
#
# Ports of the docker-compose suites, using the same test binaries and configs
# with the container paths rewritten to real ones:
#   * runtime distributed: clio_run_bdev_chimod_tests bdev_file_explicit_backend
#     with CLIO_IPC_MODE = SHM, TCP, IPC (context-runtime/test/integration/distributed)
#   * CTE distributed:     test_core_functionality
#     (context-transfer-engine/test/integration/distributed)
#
# Each suite gets fresh runtimes on every node and is torn down afterwards, so
# one failing suite cannot poison the next. Exit status is non-zero if any
# suite failed. Logs land in $GITHUB_WORKSPACE/ci-logs (uploaded as an artifact).
set -uo pipefail

SRC=${GITHUB_WORKSPACE:-$(cd "$(dirname "$0")/../.." && pwd)}
BIN=${BUILD_DIR:-$SRC/build}/bin
LOGS=$SRC/ci-logs
# Node-local scratch for runtime logs and CTE file targets (never /tmp: shared
# by other users and small on Ares compute nodes).
SCRATCH=${ARES_SCRATCH:-/mnt/ssd/$USER}/gh-ci-${SLURM_JOB_ID:?must run inside a Slurm allocation}
READY_TIMEOUT=${READY_TIMEOUT:-120}
SUITE_TIMEOUT=${SUITE_TIMEOUT:-1200}

mkdir -p "$LOGS"
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
HEAD=${NODES[0]}
NN=${#NODES[@]}
WORK=$LOGS/cluster; mkdir -p "$WORK"
HOSTFILE=$WORK/hostfile
for h in "${NODES[@]}"; do getent hosts "$h" | awk '{print $1; exit}'; done > "$HOSTFILE"
echo "nodes: ${NODES[*]} (head $HEAD)"; cat "$HOSTFILE"

export PATH="$BIN:$PATH" LD_LIBRARY_PATH="$BIN:${LD_LIBRARY_PATH:-}" CLIO_REPO_PATH="$BIN"
ALL=(srun --overlap -N"$NN" --ntasks-per-node=1 --kill-on-bad-exit=0)
ONHEAD=(srun --overlap -N1 -n1 -w "$HEAD")
on_all() { "${ALL[@]}" "$@"; }

# Kill anything a previous (crashed) run left on the nodes.
reap() {
  on_all bash -c '
    pkill -9 -u "$USER" -f "[c]lio_run runtime" 2>/dev/null || true
    rm -f /dev/shm/chi_* /dev/shm/clio_* 2>/dev/null || true
    rm -rf "'"$SCRATCH"'" 2>/dev/null || true' || true
}

# $1 = source config, $2 = suite name. Rewrites container paths to real ones.
make_conf() {
  local out=$WORK/$2.yaml
  sed -E \
    -e "s#^([[:space:]]*hostfile:).*#\1 \"$HOSTFILE\"#" \
    -e "s#^([[:space:]]*file:)[[:space:]]*\"?/tmp/clio.log\"?#\1 $SCRATCH/clio.log#" \
    -e "s#^([[:space:]]*- path:)[[:space:]]*/mnt/(hdd[0-9]+)#\1 $SCRATCH/\2#" \
    "$1" > "$out"
  echo "$out"
}

# $1 = config, $2 = suite name. Starts one runtime per node, waits for all.
start_cluster() {
  local conf=$1 name=$2
  on_all bash -c "mkdir -p $SCRATCH/hdd1 $SCRATCH/hdd2 $SCRATCH/hdd3 $SCRATCH/hdd4"
  CLIO_SERVER_CONF=$conf CLIO_WITH_RUNTIME=0 \
    srun --overlap -N"$NN" --ntasks-per-node=1 --kill-on-bad-exit=0 \
    --output="$LOGS/$name-runtime-%N.log" \
    bash -c 'exec clio_run runtime start --fresh' &
  RUNTIME_STEP=$!
  local deadline=$((SECONDS + READY_TIMEOUT))
  until CLIO_SERVER_CONF=$conf on_all bash -c 'clio_run status >/dev/null 2>&1' 2>/dev/null; do
    if (( SECONDS > deadline )) || ! kill -0 "$RUNTIME_STEP" 2>/dev/null; then
      echo "ERROR: $name: runtimes not ready after ${READY_TIMEOUT}s (or a runtime exited)"
      return 1
    fi
    sleep 3
  done
  echo "$name: all $NN runtimes ready"
}

stop_cluster() {
  local conf=$1
  CLIO_SERVER_CONF=$conf timeout 60 "${ALL[@]}" bash -c 'clio_run runtime stop >/dev/null 2>&1' || true
  kill "${RUNTIME_STEP:-}" 2>/dev/null; wait "${RUNTIME_STEP:-}" 2>/dev/null
  reap
}

FAILED=()
run_suite() {   # $1 = name, $2 = source config, rest = command on the head node
  local name=$1 src=$2; shift 2
  echo "::group::$name"
  local conf rc=0; conf=$(make_conf "$src" "$name")
  if start_cluster "$conf" "$name"; then
    CLIO_SERVER_CONF=$conf CLIO_WITH_RUNTIME=0 \
      timeout "$SUITE_TIMEOUT" "${ONHEAD[@]}" "$@" 2>&1 | tee "$LOGS/$name.log"
    rc=${PIPESTATUS[0]}
  else
    rc=1
  fi
  stop_cluster "$conf"
  echo "::endgroup::"
  if (( rc == 0 )); then echo "PASS $name"; else echo "::error::FAIL $name (rc=$rc)"; FAILED+=("$name"); fi
}

reap
RT=$SRC/context-runtime/test/integration/distributed/clio_conf.yaml
for mode in SHM TCP IPC; do
  run_suite "runtime-distributed-${mode,,}" "$RT" \
    env CLIO_IPC_MODE=$mode clio_run_bdev_chimod_tests bdev_file_explicit_backend
done
run_suite cte-distributed \
  "$SRC/context-transfer-engine/test/integration/distributed/clio_config.yaml" \
  test_core_functionality

echo "=================="
if (( ${#FAILED[@]} )); then echo "FAILED suites: ${FAILED[*]}"; exit 1; fi
echo "ALL SUITES PASSED"
