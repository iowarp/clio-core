#!/bin/bash
# Distributed COMPRESSED GPU vector on Slurm + Apptainer.
#
# One GPU node (rank 0) compresses pages in HBM and stores them through the
# compressor into a cte_core pool whose storage spans ALL nodes; reads them back
# and verifies. Requires the compressor's hash-routed forward
# (CLIO_CTE_COMPRESS_DISTRIBUTE=1) or every compressed page would stay local.
#
# All nodes are GPU nodes running identical compose (compressor 600 -> cte_core
# 512 -> ram bdev). Only rank 0 runs the test client (CTE_CLIENT_MODE=rank0);
# the other nodes' daemons just host cte_core storage that compressed pages fan
# out to.
#
# Usage: sbatch --nodes=4 run_slurm_gpu.sbatch
#SBATCH --job-name=cte_gpu_dist
#SBATCH --account=bekn-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32g
#SBATCH --time=00:25:00
#SBATCH --output=%x-%j.out
set -euo pipefail

REPO="${CLIO_REPO:-/u/rpawar/clio-core}"
SIF="${CLIO_SIF:-/u/rpawar/containers/deps-nvidia.sif}"
BIN="$REPO/build/bin"
CUSZP="${CUSZP_LIB:-/u/rpawar/cuSZp/install-ctr/lib}"
NNODES="${SLURM_JOB_NUM_NODES:-4}"

RUNDIR="${CLIO_RUNDIR_BASE:-/u/rpawar/gsbench}/gdist_${SLURM_JOB_ID}"
mkdir -p "$RUNDIR"
LOCAL_STORE="/tmp/clio_gdist_${SLURM_JOB_ID}"
PORT=$(( 9000 + (SLURM_JOB_ID % 900) ))

scontrol show hostnames "$SLURM_JOB_NODELIST" > "$RUNDIR/hostfile"
echo "=== hostfile (port $PORT) ==="; cat -n "$RUNDIR/hostfile"

# Compose: compressor(600) -> cte_core(512, storage spans nodes) -> ram bdev.
# Identical on every node; the compressor exists everywhere but only rank 0
# invokes it (compression runs on rank 0's GPU).
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
  - mod_name: clio_bdev
    pool_name: "ram::chi_default_bdev"
    pool_query: local
    pool_id: "301.0"
    bdev_type: ram
    capacity: "4096MB"

  - mod_name: clio_cte_compressor
    pool_name: cte_compressor
    pool_query: local
    pool_id: "600.0"
    next_pool_id: "512.0"

  - mod_name: clio_cte_core
    pool_name: cte_core
    pool_query: local
    pool_id: "512.0"
    storage:
      - path: $LOCAL_STORE/hdd1
        bdev_type: file
        capacity_limit: 8GB
        score: 1.0
    dpe:
      dpe_type: max_bw
EOF
echo "=== config ==="; cat "$RUNDIR/clio_config.yaml"

export RUNDIR LOCAL_STORE CTE_NUM_NODES="$NNODES" CLIO_BIN_DIR="$BIN"
export CTE_TEST_BIN="test_gpu_vector_distributed" CTE_CLIENT_MODE="server_rank0"
export TEST_FILTER=""
export CLIO_REPO_PATH="$BIN"
# GPU-specific: cuSZp on the lib path, pin cuszp, enable distributed forwarding.
export CLIO_EXTRA_LDPATH="$CUSZP"
export CLIO_CTE_COMPRESS_LIB="cuszp"
export CLIO_CTE_COMPRESS_DISTRIBUTE="1"
export CLIO_APPTAINER_NV="1"          # node_launch adds --nv
export CLIO_DAEMON_WARMUP="16"        # compressor + cuSZp warmup

echo "=== launching $NNODES GPU daemons (distribute=1, compressor=cuszp) ==="
srun --ntasks="$NNODES" --ntasks-per-node=1 --export=ALL \
  apptainer exec --nv --bind /u/rpawar --bind /tmp "$SIF" \
    bash "$REPO/context-transfer-engine/test/integration/distributed_slurm/node_launch.sh" \
  || true

rc=0
for r in $(seq 0 $((NNODES-1))); do
  e=$(cat "$RUNDIR/EXIT.$r" 2>/dev/null || echo "MISSING")
  [ "$r" = "0" ] && { echo "rank $r (client) exit: $e"; [ "$e" = "0" ] || rc=1; } \
                 || echo "rank $r (server) exit: $e"
done
echo "=== TEST EXIT: $rc  (rundir=$RUNDIR) ==="

echo "=== per-node storage (did compressed pages distribute?) ==="
grep -h "^rank=" "$RUNDIR"/storage.* 2>/dev/null || echo "no storage reports"
nodes_with_data=$(grep -h "^rank=" "$RUNDIR"/storage.* 2>/dev/null \
                  | awk -F'node_alloc_bytes=' '$2+0 > 0' | wc -l)
echo "nodes holding compressed pages: $nodes_with_data / $NNODES"

echo "=== rank 0 client output ==="
grep -E "^\[DIST\]|PASS|FAIL|Passed:|Failed:" "$RUNDIR/test.0.log" 2>/dev/null \
  | grep -v "INFO " | head -20 || echo "(no log)"
exit "$rc"
