#!/bin/bash
BENCH_ROOT="${BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# GNN/Eternia experiment driver. Run INSIDE the container on the A100 node.
#   apptainer exec --nv -B /work,/tmp,/u <sif> bash run_exp.sh <exp>
# where <exp> is one of: a | b | c | d_bal | d_best | e | all
set -uo pipefail

BIN=/u/rpawar/clio-core/build/bin
RES=${BENCH_ROOT}/results
LOGS=${BENCH_ROOT}/logs
ARXIV=${BENCH_ROOT}/data/arxiv
PAPERS=/tmp/gnn/data/papers100M
GNN=/u/rpawar/clio-core/context-transfer-engine/adapter/gpu_vector/gnn
mkdir -p "$RES" "$LOGS"

# Host RAM on the node is 251 GB and the job reserves 224 GB. papers100M holds a
# 57 GB host copy of A plus the compressed DRAM tier (~53 GB), so cap the tier
# well below what would push the job over its cgroup limit.
DRAM_MIB="${DRAM_MIB:-140000}"
HBM_MIB="${HBM_MIB:-2048}"
EPOCHS_P="${EPOCHS_P:-30}"

common_env() {   # $1 = port
  export LD_LIBRARY_PATH=.
  export CLIO_REPO_PATH=.
  export CLIO_BIND_ADDR=127.0.0.1
  export CLIO_PORT="$1"
}

run_a() {
  echo "########## (a) TRAINING CORRECTNESS - arxiv NATIVE, lossless zstd ##########"
  cd "$BIN" || exit 1
  common_env 10701
  CLIO_CTE_COMPRESS_LIB=zstd CLIO_CTE_COMPRESS_PRESET=balanced \
  CLIO_GNN_DATA="$ARXIV" CLIO_GNN_TRAIN_NODES=0 CLIO_GNN_EPOCHS=30 \
  CLIO_GNN_LR=0.2 CLIO_GNN_HIDDEN=64 \
  CLIO_GNN_CSV="$RES/gnn_train_results.csv" \
    ./test_gpu_vector_gnn_train 2>&1 | tee "$LOGS/exp_a_arxiv_zstd.log"
  echo "(a) rc=${PIPESTATUS[0]}"
}

# $1 = lib, $2 = preset, $3 = port, $4 = tag
run_papers_train() {
  local lib="$1" preset="$2" port="$3" tag="$4"
  echo "########## (b) PAPERS100M TRAINING - lib=$lib preset=$preset ##########"
  echo "     A = 111M x 128 f32 = ~57 GB  >>  40 GB HBM  => in-core MUST OOM"
  cd "$BIN" || exit 1
  common_env "$port"
  CLIO_CTE_COMPRESS_LIB="$lib" CLIO_CTE_COMPRESS_PRESET="$preset" \
  CLIO_GNN_DATA="$PAPERS" CLIO_GNN_TRAIN_NODES=0 \
  CLIO_GNN_EPOCHS="$EPOCHS_P" CLIO_GNN_LR=0.2 CLIO_GNN_HIDDEN=64 \
  CLIO_GNN_PAGE_ROWS=65536 CLIO_GNN_WINDOW=2 \
  CLIO_GNN_DRAM_MIB="$DRAM_MIB" CLIO_GNN_HBM_MIB="$HBM_MIB" \
  CLIO_GNN_CSV="$RES/gnn_train_results.csv" \
    ./test_gpu_vector_gnn_train 2>&1 | tee "$LOGS/exp_${tag}.log"
  echo "rc=${PIPESTATUS[0]}"
}

# $1 = lib, $2 = preset, $3 = port base, $4 = tag
run_papers_capacity() {
  local lib="$1" preset="$2" port="$3" tag="$4"
  echo "########## (c) PAPERS100M FORWARD CAPACITY SWEEP - lib=$lib preset=$preset ##########"
  cd "$BIN" || exit 1
  local p="$port"
  # sweep from comfortably-in-HBM (bit-exact checkable) up to the full 111M nodes
  for nodes in 2000000 8000000 20000000 60000000 111000000; do
    echo "---- CAP_NODES=$nodes (~$((nodes*128*4/1024/1024)) MiB) ----"
    common_env "$p"
    CLIO_CTE_COMPRESS_LIB="$lib" CLIO_CTE_COMPRESS_PRESET="$preset" \
    CLIO_GNN_DATA="$PAPERS" CLIO_GNN_CAP_NODES="$nodes" \
    CLIO_GNN_PAGE_ROWS=65536 CLIO_GNN_WINDOW=2 \
    CLIO_GNN_DRAM_MIB="$DRAM_MIB" CLIO_GNN_HBM_MIB="$HBM_MIB" \
    CLIO_GNN_CSV="$RES/gnn_cap_results.csv" \
      ./test_gpu_vector_gnn_capacity 2>&1 | tee -a "$LOGS/exp_${tag}.log"
    echo "rc=${PIPESTATUS[0]}"
    p=$((p+1))
  done
}

run_e() {
  echo "########## (e) PAGERANK CACHE PREDICTION - papers100M ##########"
  python3 "$GNN/gnn_pagerank_cache.py" --data "$PAPERS" \
     --edges /tmp/gnn/data/papers100M-bin/unpacked/edge_index.npy \
     --max-batches 400 --sweep-sample 40000000 --lru-limit 20000000 \
     --md-out "$RES/pagerank_papers100M.md" 2>&1 | tee "$LOGS/exp_e_pagerank_papers.log"
  echo "rc=${PIPESTATUS[0]}"
}

case "${1:-all}" in
  a)      run_a ;;
  b)      run_papers_train zstd balanced 10702 b_papers_zstd ;;
  c)      run_papers_capacity zstd balanced 10710 c_papers_cap_zstd ;;
  d_bal)  run_papers_train cuszp balanced 10720 d_papers_cuszp_balanced
          run_papers_capacity cuszp balanced 10730 d_papers_cap_cuszp_balanced ;;
  d_best) run_papers_train cuszp best 10740 d_papers_cuszp_best ;;
  e)      run_e ;;
  all)    run_a
          run_papers_train zstd balanced 10702 b_papers_zstd
          run_papers_capacity zstd balanced 10710 c_papers_cap_zstd
          run_papers_train cuszp balanced 10720 d_papers_cuszp_balanced
          run_papers_train cuszp best 10740 d_papers_cuszp_best
          run_e ;;
  *) echo "unknown exp '$1'"; exit 2 ;;
esac
