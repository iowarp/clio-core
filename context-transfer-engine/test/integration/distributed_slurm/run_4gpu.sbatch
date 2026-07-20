#!/bin/bash
# Single node, ALL 4 GPUs: one compressed Gray-Scott checkpoint per GPU, run
# concurrently. Each GPU independently evolves a GS field, compresses its pages
# IN its own HBM, stores them, reads them back and verifies -- so all four A100s
# on the node do zero-copy HBM compression at once.
#
# Why 4 processes (not one process, 4 GPUs): the compressor's dedicated CUDA
# context is created on device 0 (cuszp.h CompressorContext). Pinning each
# process to a different physical GPU with CUDA_VISIBLE_DEVICES makes that GPU
# the process's "device 0", so all four compress on their own GPU with no change
# to the compressor. Each process is a standalone server (no hostfile), distinct
# port + storage dir.
#
# Usage: sbatch run_4gpu.sbatch
#SBATCH --job-name=cte_4gpu
#SBATCH --account=bekn-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --mem=64g
#SBATCH --time=00:20:00
#SBATCH --output=%x-%j.out
set -euo pipefail

REPO="${CLIO_REPO:-/u/rpawar/clio-core}"
SIF="${CLIO_SIF:-/u/rpawar/containers/deps-nvidia.sif}"
BIN="$REPO/build/bin"
CUSZP="${CUSZP_LIB:-/u/rpawar/cuSZp/install-ctr/lib}"
NGPU="${NGPU:-4}"

RUNDIR="${CLIO_RUNDIR_BASE:-/u/rpawar/gsbench}/4gpu_${SLURM_JOB_ID}"
mkdir -p "$RUNDIR"

# One standalone compose per GPU: compressor(600) -> cte_core(512, file storage) ->
# ram bdev. No hostfile => standalone single-GPU. Distinct port + storage per GPU.
write_conf() {
  local g="$1" port="$2" store="$3"
  mkdir -p "$store/hdd1"
  cat > "$RUNDIR/conf_g$g.yaml" <<EOF
---
memory:
  main_segment_size: auto
  client_data_segment_size: 2GB
  runtime_data_segment_size: 2GB
networking:
  port: $port
  neighborhood_size: 32
logging:
  level: info
  file: $RUNDIR/clio_g$g.log
runtime:
  num_threads: 8
  queue_depth: 1024
  local_sched: "default"
compose:
  - mod_name: clio_bdev
    pool_name: "ram::chi_default_bdev"
    pool_query: local
    pool_id: "301.0"
    bdev_type: ram
    capacity: "2048MB"
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
      - path: $store/hdd1
        bdev_type: file
        capacity_limit: 4GB
        score: 1.0
    dpe:
      dpe_type: max_bw
EOF
}

declare -a PIDS=()
for g in $(seq 0 $((NGPU-1))); do
  # The runtime binds BOTH config_port and config_port+1, so space GPUs by 20 to
  # avoid any overlap between concurrent instances on this node.
  PORT=$(( 9000 + (SLURM_JOB_ID % 300) + g * 20 ))
  STORE="/tmp/clio_4gpu_${SLURM_JOB_ID}_g$g"
  write_conf "$g" "$PORT" "$STORE"
  echo "=== launching GPU $g (port $PORT, store $STORE) ==="
  (
    CUDA_VISIBLE_DEVICES="$g" \
    apptainer exec --nv --bind /u/rpawar --bind /tmp "$SIF" bash -lc "
      export LD_LIBRARY_PATH='$CUSZP:$BIN:\${LD_LIBRARY_PATH:-}'
      export CLIO_REPO_PATH='$BIN' CLIO_SERVER_CONF='$RUNDIR/conf_g$g.yaml'
      export CLIO_CTE_COMPRESS_LIB=cuszp
      export CTE_NUM_NODES=1 CTE_RANK=0
      '$BIN/test_gpu_vector_distributed'
    " >"$RUNDIR/gpu_$g.log" 2>&1
    echo "$?" >"$RUNDIR/EXIT_g$g"
  ) &
  PIDS+=($!)
done

echo "=== waiting for $NGPU concurrent GPU checkpoints ==="
for p in "${PIDS[@]}"; do wait "$p" || true; done

rc=0
echo "=== per-GPU result ==="
for g in $(seq 0 $((NGPU-1))); do
  # `|| true` on every grep: a no-match exits 1, which would abort under set -e.
  e=$(cat "$RUNDIR/EXIT_g$g" 2>/dev/null || echo MISSING)
  gpu=$(grep -oE "\[DIST\] GPU: .*" "$RUNDIR/gpu_$g.log" 2>/dev/null | head -1 || true)
  err=$(grep -oE "max_abs_err=[0-9.e+-]+" "$RUNDIR/gpu_$g.log" 2>/dev/null | head -1 || true)
  pass=$(grep -c "\[PASS\]" "$RUNDIR/gpu_$g.log" 2>/dev/null || true); pass=${pass:-0}
  echo "GPU $g: exit=$e pass=$pass $err | $gpu"
  { [ "$e" = "0" ] && [ "$pass" -ge 1 ]; } || rc=1
done
echo "=== 4-GPU EXIT: $rc  (rundir=$RUNDIR) ==="
[ "$rc" = 0 ] && echo "ALL $NGPU GPUs checkpointed a Gray-Scott field with in-HBM compression concurrently."
exit "$rc"
