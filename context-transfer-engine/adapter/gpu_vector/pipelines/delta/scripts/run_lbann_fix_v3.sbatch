#!/bin/bash
#SBATCH --job-name=lbfix3
#SBATCH --account=bekn-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus-per-node=1 --mem=120g
#SBATCH --time=00:30:00
#SBATCH --output=/u/rpawar/gsbench/%x-%j.out
set -uo pipefail
# Validate pulled lbann fix (6655b729e, in clio_lbann_PAGED_bench.cc): each node
# verifies only its resident band -> no cross-node refetch -> no stale replica.
# Direct apptainer calls (mirrors the working page-cache probe). Full output kept.
SIF=/u/rpawar/containers/deps-nvidia.sif; SRC=/u/rpawar/eternia-4node; CU=/usr/local/cuda-12.6
LOGD=/u/rpawar/eternia_logs
arm(){  # name  extra-bench-args...
  local name="$1"; shift
  local log="$LOGD/lbfix3_${name}.log"
  echo "===== ARM $name: clio_lbann_paged_bench --hidden 8192 --steps 5 $* ====="
  apptainer exec --nv --bind /u/rpawar "$SIF" bash -lc "
    export PATH=$SRC/build-clang/bin:$CU/bin:\$PATH LD_LIBRARY_PATH=$SRC/build-clang/bin:$CU/lib64:\$LD_LIBRARY_PATH CLIO_REPO_PATH=$SRC/build-clang/bin
    timeout 500 clio_lbann_paged_bench --in 256 --hidden 8192 --out 64 --batch 64 --steps 5 --lr 0.00001 $*
  " >"$log" 2>&1
  local rc=$?
  echo "  rc=$rc  (log: $log)"
  grep -aiE "GATE|ALL GATES|ms/step|band|owner|resident|error|assert|abort|stale" "$log" | grep -aivE "INFO|WARNING|AdminClient" | tail -12
  echo "  --- raw tail ---"; grep -aivE "INFO|WARNING|AdminClient" "$log" | tail -5
}
arm resident_1node
arm resident_2node --nodes 2 --node 0
arm ooc_2node --nodes 2 --node 0 --cap 64 --page-kb 64
echo "=== LBFIX3_DONE ==="
