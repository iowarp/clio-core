#!/bin/bash
# 4-node distributed CTE test on Slurm + Apptainer (the docker-compose
# `test/integration/distributed` suite, ported to a real cluster).
#
# Why this exists: the docker suite runs all 4 "nodes" on one host over a bridge
# network, and its cross-node distribution assertions are disabled under #503
# (every blob op resolved to the client's local container). This harness runs the
# same binaries on 4 REAL nodes with real hostnames and real TCP, so we can see
# whether #503 is a docker-networking artifact or a genuine routing bug.
#
# Apptainer uses host networking, so each daemon binds 0.0.0.0:<port> and peers
# reach it at <hostname>:<port> -- no bridge network needed (and docker is not
# available on Delta).
#
# Usage:
#   sbatch --nodes=4 run_slurm.sbatch ["<test filter>"]
#SBATCH --job-name=cte_dist
#SBATCH --account=bekn-delta-cpu
#SBATCH --partition=cpu-interactive
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32g
#SBATCH --time=00:25:00
#SBATCH --output=%x-%j.out
set -euo pipefail

REPO="${CLIO_REPO:-/u/rpawar/clio-core}"
SIF="${CLIO_SIF:-/u/rpawar/containers/deps-nvidia.sif}"
BIN="$REPO/build/bin"
NNODES="${SLURM_JOB_NUM_NODES:-4}"
# The stock distributed test needs a filter (its binary holds many unrelated
# cases); the coherency binary holds only coherency cases, so it runs unfiltered.
# Use ${1-...} (no colon) so an explicitly-empty $1 means "no filter" rather than
# silently falling back to a filter that matches nothing.
_default_filter=""
if [ "${CTE_TEST_BIN:-test_core_functionality}" = "test_core_functionality" ]; then
  _default_filter="FUNCTIONAL - Distributed Execution Validation"
fi
TEST_FILTER="${1-$_default_filter}"

# Shared (Lustre) rundir: visible to every node, used for hostfile + barriers.
RUNDIR="${CLIO_RUNDIR_BASE:-/u/rpawar/gsbench}/dist_${SLURM_JOB_ID}"
mkdir -p "$RUNDIR"
# Node-local storage for CTE file targets.
LOCAL_STORE="/tmp/clio_dist_${SLURM_JOB_ID}"

# Port must be identical on every node (peers dial hostname:PORT). Derive from
# the job id to avoid colliding with another user's run on a shared node.
PORT=$(( 9000 + (SLURM_JOB_ID % 900) ))

# ---- hostfile: one hostname per line; node id == line offset ----
scontrol show hostnames "$SLURM_JOB_NODELIST" > "$RUNDIR/hostfile"
echo "=== hostfile (port $PORT) ==="; cat -n "$RUNDIR/hostfile"

# ---- config: mirrors integration/distributed/clio_config.yaml, but with the
#      generated hostfile and node-local storage paths ----
cat > "$RUNDIR/clio_config.yaml" <<EOF
---
memory:
  main_segment_size: auto
  client_data_segment_size: 2GB
  runtime_data_segment_size: 2GB

networking:
  port: $PORT
  neighborhood_size: 32
  hostfile: $RUNDIR/hostfile

logging:
  level: info
  file: $RUNDIR/clio.log

runtime:
  num_threads: 16
  queue_depth: 1024
  local_sched: "default"
  heartbeat_interval: 1000

compose:
  - mod_name: clio_cte_core
    pool_name: cte_main
    pool_query: local
    pool_id: "512.0"
    storage:
      - path: $LOCAL_STORE/hdd1
        bdev_type: file
        capacity_limit: 10GB
        score: 0.25
      - path: $LOCAL_STORE/hdd2
        bdev_type: file
        capacity_limit: 10GB
        score: 0.25
      - path: $LOCAL_STORE/hdd3
        bdev_type: file
        capacity_limit: 10GB
        score: 0.25
      - path: $LOCAL_STORE/hdd4
        bdev_type: file
        capacity_limit: 10GB
        score: 0.25
    dpe:
      dpe_type: max_bw
    targets:
      neighborhood: $NNODES
      default_target_timeout_ms: 30000
      poll_period_ms: 5000
    performance:
      target_stat_interval_ms: 5000
      blob_cache_size_mb: 4096
      max_concurrent_operations: 64
      score_threshold: 0.7
      score_difference_threshold: 0.05
EOF
echo "=== config ==="; cat "$RUNDIR/clio_config.yaml"

export RUNDIR LOCAL_STORE CTE_NUM_NODES="$NNODES" CLIO_BIN_DIR="$BIN"
export TEST_FILTER CTE_TEST_BIN="${CTE_TEST_BIN:-test_core_functionality}"
export CTE_CLIENT_MODE="${CTE_CLIENT_MODE:-rank0}"
export CLIO_REPO_PATH="$BIN"

echo "=== launching $NNODES daemons (bin=$CTE_TEST_BIN mode=$CTE_CLIENT_MODE filter='$TEST_FILTER') ==="
srun --ntasks="$NNODES" --ntasks-per-node=1 --export=ALL \
  apptainer exec --bind /u/rpawar --bind /tmp "$SIF" \
    bash "$REPO/context-transfer-engine/test/integration/distributed_slurm/node_launch.sh" \
  || true

# Aggregate: in `all` mode every rank is a client, so the job fails if ANY does.
rc=0
for r in $(seq 0 $((NNODES-1))); do
  e=$(cat "$RUNDIR/EXIT.$r" 2>/dev/null || echo "MISSING")
  echo "rank $r exit: $e"
  [ "$e" = "0" ] || rc=1
done
echo "=== TEST EXIT: $rc  (rundir=$RUNDIR) ==="

# Did blob data actually land on more than one node? This is the direct evidence
# `completer_` cannot give (it is 0 = "unset" for CTE blob ops -- see #503).
echo "=== per-node storage (did data distribute?) ==="
grep -h "^rank=" "$RUNDIR"/storage.* 2>/dev/null || echo "no storage reports"
nodes_with_data=$(grep -h "^rank=" "$RUNDIR"/storage.* 2>/dev/null \
                  | awk -F'node_alloc_bytes=' '$2+0 > 0' | wc -l)
echo "nodes holding blob bytes: $nodes_with_data / $NNODES"

echo "=== per-rank test output ==="
for r in $(seq 0 $((NNODES-1))); do
  echo "--- rank $r ---"
  grep -E "^\[COH\]|PASS|FAIL|Passed:|Failed:" "$RUNDIR/test.$r.log" 2>/dev/null \
    | grep -v "INFO " | head -14 || echo "(no log)"
done
exit "$rc"
