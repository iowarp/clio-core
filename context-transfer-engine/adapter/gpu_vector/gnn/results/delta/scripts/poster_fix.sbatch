#!/bin/bash
#SBATCH -A bekn-delta-gpu
#SBATCH -p gpuA100x4
#SBATCH -J gnnpost
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus=1
#SBATCH --mem=224g
#SBATCH -t 04:00:00
#SBATCH -o /u/rpawar/gnnbench/logs/poster-fix-%j.out
#
# Closes the two gaps in the SC26 poster:
#
#   1. Result 1 (zstd vs cuSZp) was measured under FULL-BATCH GD, so its loss
#      curve (5.13 -> 4.55, accuracy at the majority-class floor) is inconsistent
#      with a headline that quotes mini-batch accuracy. Re-run BOTH codecs under
#      mini-batch so the figure and the headline describe the same configuration.
#
#   2. "Peak GPU ~2 GiB (being measured)" was never measured. The binary now
#      tracks (free_at_start - free_now) across the whole run via cudaMemGetInfo,
#      giving the process's true device footprint -- feature window PLUS labels,
#      page cache and runtime buffers -- not just the 64 MiB window.
#
# 10 epochs = ~8470 updates, already ~7x what arxiv needed to converge, and it
# keeps both codec arms inside one 4 h job.
set -uo pipefail
echo "HOST $(hostname)  START $(date)"

SIF=/u/rpawar/containers/deps-nvidia.sif
BIN=/u/rpawar/clio-core/build/bin
PAPERS=/tmp/gnn/data/papers100M
RES=/u/rpawar/gnnbench/results
CSV="$RES/gnn_poster_minibatch.csv"
mkdir -p "$RES"
APP="apptainer exec --nv -B /work,/tmp,/u,/projects $SIF"

echo "########## 1) PREP (~45 min) ##########"
t0=$SECONDS
apptainer exec -B /work,/tmp,/u,/projects "$SIF" \
  bash /u/rpawar/gnnbench/prep_papers.sh 2>&1 | tail -12
echo "[time] prep $((SECONDS-t0)) s"
[ -s "$PAPERS/agg_features.f32" ] || { echo "!! PREP FAILED"; exit 1; }

run() {   # $1=lib $2=preset $3=port $4=label
  echo "########## $4 ##########"
  t=$SECONDS
  $APP bash -c "
    cd $BIN || exit 1
    unset CC CXX
    export LD_LIBRARY_PATH=. CLIO_REPO_PATH=. CLIO_BIND_ADDR=127.0.0.1 CLIO_PORT=$3
    CLIO_CTE_COMPRESS_LIB=$1 CLIO_CTE_COMPRESS_PRESET=$2 \
    CLIO_GNN_DATA=$PAPERS CLIO_GNN_TRAIN_NODES=0 \
    CLIO_GNN_EPOCHS=10 CLIO_GNN_LR=0.2 CLIO_GNN_HIDDEN=64 \
    CLIO_GNN_MINIBATCH=1 CLIO_GNN_PAGE_ROWS=65536 CLIO_GNN_WINDOW=2 \
    CLIO_GNN_DRAM_MIB=140000 CLIO_GNN_HBM_MIB=2048 \
    CLIO_GNN_CSV=$CSV \
      ./test_gpu_vector_gnn_train 2>&1 \
      | grep -aE '^\[TRAIN\] (A=|IN-CORE|ETERNIA:|PEAK GPU|in-core OOM)|eternia e|Compressor pinned|Passed:|Failed:'
  "
  echo "[time] $4 $((SECONDS-t)) s"
  echo
}

run zstd  balanced 10921 "papers100M MINI-BATCH 10ep -- zstd (lossless)"
run cuszp balanced 10922 "papers100M MINI-BATCH 10ep -- cuSZp (lossy, balanced)"

echo "===== CSV ====="; cat "$CSV" 2>/dev/null
echo "DONE $(date)"
