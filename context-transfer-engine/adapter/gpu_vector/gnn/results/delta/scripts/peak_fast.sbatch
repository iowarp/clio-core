#!/bin/bash
#SBATCH -A bekn-delta-gpu
#SBATCH -p gpuA100x4
#SBATCH -J gnnfast
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gpus=1
#SBATCH --mem=32g
#SBATCH -t 00:15:00
#SBATCH -o /u/rpawar/gnnbench/logs/peak-fast-%j.out
#
# DEADLINE-DRIVEN: 15 min walltime so backfill schedules it immediately.
#
# Measures the PEAK GPU instrumentation on arxiv and products, which live on
# persistent /u and need NO prep. This pins down the two terms that cannot be
# computed on paper -- the CUDA context size and whether the CTE HBM tier
# actually occupies its configured cap -- so the papers100M peak can be stated
# with only the analytically-exact terms extrapolated (labels = N*8 bytes,
# dz2_buf = window*page_rows*C*4, both fully determined by N and C).
set -uo pipefail
echo "HOST $(hostname)  START $(date)"
SIF=/u/rpawar/containers/deps-nvidia.sif
BIN=/u/rpawar/clio-core/build/bin
APP="apptainer exec --nv -B /work,/tmp,/u,/projects $SIF"

run() {   # $1=data $2=port $3=label
  echo "########## $3 ##########"
  $APP bash -c "
    cd $BIN || exit 1
    unset CC CXX
    export LD_LIBRARY_PATH=. CLIO_REPO_PATH=. CLIO_BIND_ADDR=127.0.0.1 CLIO_PORT=$2
    CLIO_CTE_COMPRESS_LIB=zstd CLIO_CTE_COMPRESS_PRESET=balanced \
    CLIO_GNN_DATA=$1 CLIO_GNN_TRAIN_NODES=0 \
    CLIO_GNN_EPOCHS=1 CLIO_GNN_LR=0.2 CLIO_GNN_HIDDEN=64 \
    CLIO_GNN_MINIBATCH=1 CLIO_GNN_PAGE_ROWS=65536 CLIO_GNN_WINDOW=2 \
    CLIO_GNN_HBM_MIB=2048 \
      ./test_gpu_vector_gnn_train 2>&1 \
      | grep -aE '^\[TRAIN\] (A=|IN-CORE|ETERNIA:|PEAK GPU)|Passed:|Failed:'
  "
  echo
}
run /u/rpawar/gnnbench/data/arxiv    10941 "arxiv  (N=169343, C=40)"
run /u/rpawar/gnnbench/data/products 10942 "products (N=2449029, C=47)"
echo "DONE $(date)"
