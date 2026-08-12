#!/bin/bash
#SBATCH -A bekn-delta-gpu
#SBATCH -p gpuA100x4
#SBATCH -J gnnmb
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --gpus=1
#SBATCH --mem=64g
#SBATCH -t 01:30:00
#SBATCH -o /u/rpawar/gnnbench/logs/minibatch-%j.out
#
# Mini-batch SGD vs full-batch GD, on ogbn-arxiv (persistent on /u, so this needs
# no papers100M re-prep). Answers: does mini-batching escape the majority-class
# floor that 30 full-batch updates could not?
#
# Batch size = CLIO_GNN_PAGE_ROWS * CLIO_GNN_WINDOW. arxiv has 167,936 tiled nodes,
# so page_rows=4096/window=1 gives 41 updates per epoch -- matching the batch size
# HMGraph (ICPP'25) uses, for a like-for-like comparison.
set -uo pipefail
echo "HOST $(hostname)  $(date)"

SIF=/u/rpawar/containers/deps-nvidia.sif
BIN=/u/rpawar/clio-core/build/bin
ARXIV=/u/rpawar/gnnbench/data/arxiv
RES=/u/rpawar/gnnbench/results
CSV="$RES/gnn_minibatch.csv"
mkdir -p "$RES"

run() {   # $1=mb  $2=epochs  $3=page_rows  $4=port  $5=label
  echo "########## $5 ##########"
  apptainer exec --nv -B /work,/tmp,/u,/projects "$SIF" bash -c "
    cd $BIN || exit 1
    unset CC CXX
    export LD_LIBRARY_PATH=. CLIO_REPO_PATH=. CLIO_BIND_ADDR=127.0.0.1 CLIO_PORT=$4
    CLIO_CTE_COMPRESS_LIB=zstd CLIO_CTE_COMPRESS_PRESET=balanced \
    CLIO_GNN_DATA=$ARXIV CLIO_GNN_TRAIN_NODES=0 \
    CLIO_GNN_EPOCHS=$2 CLIO_GNN_LR=0.2 CLIO_GNN_HIDDEN=64 \
    CLIO_GNN_MINIBATCH=$1 CLIO_GNN_PAGE_ROWS=$3 CLIO_GNN_WINDOW=1 \
    CLIO_GNN_CSV=$CSV \
      ./test_gpu_vector_gnn_train 2>&1 \
      | grep -aE '^\[TRAIN\] (A=|IN-CORE|ETERNIA: (stored|epoch0))|VERIFY|Passed:|Failed:'
  "
  echo
}

# control: full-batch, exactly the headline setting -> 30 updates total
run 0  30  65536 10901 "FULL-BATCH  30 epochs (= 30 updates)"
# mini-batch, batch 4096 -> ~41 updates/epoch
run 1  30   4096 10902 "MINI-BATCH  30 epochs, batch 4096 (~1230 updates)"
run 1   5   4096 10903 "MINI-BATCH   5 epochs, batch 4096 (~205 updates)"
# larger batch -> fewer, less noisy updates
run 1  30  16384 10904 "MINI-BATCH  30 epochs, batch 16384 (~310 updates)"

echo "===== CSV ====="; cat "$CSV" 2>/dev/null
echo "DONE $(date)"
