#!/bin/bash
#SBATCH -A bekn-delta-gpu
#SBATCH -p gpuA100x4
#SBATCH -J gnnpeak
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus=1
#SBATCH --mem=224g
#SBATCH -t 02:00:00
#SBATCH -o /u/rpawar/gnnbench/logs/peakgpu-%j.out
#
# Measure the TRUE peak GPU footprint of the papers100M training run, so the
# poster can stop saying "~2 GiB (being measured)".
#
# The binary now samples cudaMemGetInfo every window and reports
# (free_at_start - free_now), i.e. the whole-process device footprint: the
# bounded 64 MiB feature window PLUS the O(N) labels (847 MiB at N=111M), the
# per-window activation buffers, the CTE HBM tier and the CUDA context.
#
# Analytic estimate for reference: ~1.3 GiB if the CTE HBM tier stays empty
# (every capacity run logged "split HBM=0MiB"), ~3.3 GiB if it fills its
# 2048 MiB cap. This job settles which.
#
# Peak is reached during epoch 0, so ONE epoch suffices -- no need to retrain.
# Both codecs are run because the tier layout (and therefore the peak) differs:
# zstd stores 50,257 MiB vs cuSZp 17,342 MiB.
set -uo pipefail
echo "HOST $(hostname)  START $(date)"

SIF=/u/rpawar/containers/deps-nvidia.sif
BIN=/u/rpawar/clio-core/build/bin
PAPERS=/tmp/gnn/data/papers100M
RES=/u/rpawar/gnnbench/results
CSV="$RES/gnn_peakgpu.csv"
mkdir -p "$RES"
APP="apptainer exec --nv -B /work,/tmp,/u,/projects $SIF"

echo "########## 1) PREP (~45 min) ##########"
t0=$SECONDS
apptainer exec -B /work,/tmp,/u,/projects "$SIF" \
  bash /u/rpawar/gnnbench/prep_papers.sh 2>&1 | tail -10
echo "[time] prep $((SECONDS-t0)) s"
[ -s "$PAPERS/agg_features.f32" ] || { echo "!! PREP FAILED"; exit 1; }

run() {   # $1=lib $2=port $3=label
  echo "########## $3 ##########"
  t=$SECONDS
  $APP bash -c "
    cd $BIN || exit 1
    unset CC CXX
    export LD_LIBRARY_PATH=. CLIO_REPO_PATH=. CLIO_BIND_ADDR=127.0.0.1 CLIO_PORT=$2
    CLIO_CTE_COMPRESS_LIB=$1 CLIO_CTE_COMPRESS_PRESET=balanced \
    CLIO_GNN_DATA=$PAPERS CLIO_GNN_TRAIN_NODES=0 \
    CLIO_GNN_EPOCHS=1 CLIO_GNN_LR=0.2 CLIO_GNN_HIDDEN=64 \
    CLIO_GNN_MINIBATCH=1 CLIO_GNN_PAGE_ROWS=65536 CLIO_GNN_WINDOW=2 \
    CLIO_GNN_DRAM_MIB=140000 CLIO_GNN_HBM_MIB=2048 \
    CLIO_GNN_CSV=$CSV \
      ./test_gpu_vector_gnn_train 2>&1 \
      | grep -aE '^\[TRAIN\] (A=|IN-CORE|ETERNIA:|PEAK GPU|in-core OOM)|eternia e|Compressor pinned|Passed:|Failed:'
  "
  echo "[time] $3 $((SECONDS-t)) s"
  echo
}

run zstd  10931 "papers100M PEAK GPU -- zstd (tier 50,257 MiB)"
run cuszp 10932 "papers100M PEAK GPU -- cuSZp (tier 17,342 MiB)"

echo "===== CSV ====="; cat "$CSV" 2>/dev/null
echo "DONE $(date)"
