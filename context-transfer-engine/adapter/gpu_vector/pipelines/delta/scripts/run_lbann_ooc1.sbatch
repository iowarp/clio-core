#!/bin/bash
#SBATCH --job-name=lbooc1
#SBATCH --account=bekn-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --gpus-per-node=1 --mem=120g
#SBATCH --time=00:20:00
#SBATCH --output=/u/rpawar/gsbench/%x-%j.out
set -uo pipefail
SIF=/u/rpawar/containers/deps-nvidia.sif; SRC=/u/rpawar/eternia-4node; CU=/usr/local/cuda-12.6
LOG=/u/rpawar/eternia_logs/lbann_ooc1.log
# OUT-OF-CORE single node (cap>0 forces paging) -- the OOC arm the fix commit validated.
echo "===== lbann OOC nodes=1 cap=64 (paging) ====="
apptainer exec --nv --bind /u/rpawar "$SIF" bash -lc "
  export PATH=$SRC/build-clang/bin:$CU/bin:\$PATH LD_LIBRARY_PATH=$SRC/build-clang/bin:$CU/lib64:\$LD_LIBRARY_PATH CLIO_REPO_PATH=$SRC/build-clang/bin
  timeout 400 clio_lbann_paged_bench --in 256 --hidden 8192 --out 64 --batch 64 --steps 5 --lr 0.00001 --cap 64 --page-kb 64
" >"$LOG" 2>&1
echo "  rc=$?"
grep -aiE "GATE|ALL GATES|ms/step|cap|pages|error" "$LOG" | grep -aivE "INFO|WARNING|AdminClient|HANGWATCH" | tail -10
echo "=== LBOOC1_DONE ==="
