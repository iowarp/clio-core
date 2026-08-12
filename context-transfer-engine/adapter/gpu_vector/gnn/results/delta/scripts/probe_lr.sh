#!/bin/bash
# Underfitting probe.
#
# Hypothesis: the papers100M accuracy plateau at 0.0626 is the majority-class
# floor because full-batch GD gives ONE weight update per epoch, so 30 epochs =
# 30 updates -- far too few to leave the class prior. (Measured: epoch-29 loss
# 4.5522 is still 0.215 nats ABOVE the label entropy H(y)=4.3369, i.e. the model
# has not even finished learning the prior, let alone anything input-dependent.)
#
# Test it cheaply on a 2M-node tile (960 MiB, fits HBM so the in-core baseline
# runs too and epochs cost ~2 s instead of ~278 s), varying ONLY epochs and lr.
# If accuracy climbs with more updates, underfitting is the whole story.
set -uo pipefail

BIN=/u/rpawar/clio-core/build/bin
RES=/u/rpawar/gnnbench/results
LOGS=/u/rpawar/gnnbench/logs
PAPERS=/tmp/gnn/data/papers100M
NODES="${NODES:-2000000}"
CSV="$RES/gnn_lr_probe.csv"
mkdir -p "$RES" "$LOGS"

cd "$BIN" || exit 1
export LD_LIBRARY_PATH=. CLIO_REPO_PATH=. CLIO_BIND_ADDR=127.0.0.1

port=10800
run() {   # $1=epochs  $2=lr
  local ep="$1" lr="$2"
  port=$((port+1))
  echo "===== PROBE nodes=$NODES epochs=$ep lr=$lr ====="
  CLIO_PORT=$port \
  CLIO_CTE_COMPRESS_LIB=zstd CLIO_CTE_COMPRESS_PRESET=balanced \
  CLIO_GNN_DATA="$PAPERS" CLIO_GNN_TRAIN_NODES="$NODES" \
  CLIO_GNN_EPOCHS="$ep" CLIO_GNN_LR="$lr" CLIO_GNN_HIDDEN=64 \
  CLIO_GNN_PAGE_ROWS=65536 CLIO_GNN_WINDOW=2 \
  CLIO_GNN_CSV="$CSV" \
    ./test_gpu_vector_gnn_train 2>&1 \
    | grep -aE '^\[TRAIN\] (A=|IN-CORE|ETERNIA: (stored|epoch0))|VERIFY|Passed:|Failed:'
  echo
}

# control: the exact setting used for the headline runs
run 30   0.2
# same lr, many more updates -> tests "too few updates"
run 300  0.2
# same 30 updates, much larger steps -> tests "lr too small"
run 30   5.0
run 30   50.0
# both
run 300  5.0

echo "===== probe CSV ====="
cat "$CSV"
