#!/usr/bin/env bash
#===============================================================================
# run_v2_adapt.sh -- stream measured chunks of one dataset through Clio's
# compressor with NeuroPress v2 and record what it predicted and chose.
#
#   run_v2_adapt.sh DATASET ARM [LR]
#     DATASET  vpic | nyx   (chunks staged in /mnt/nvme0/v2-adapt/DATASET/fields)
#     ARM      static | learn | explore
#     LR       neuropress_learning_rate (default 0.01, the config default)
#
# static : no online learning, no exploration (the trained weights as shipped)
# learn  : online learning from each primary's measured result
# explore: learning + exploration of the next K=3 settings when the cost
#          error exceeds 0.5 (both the config defaults), adopting a winner
# Cost model: 1 GB/s (CLIO_NEUROPRESS_COST_BW=1e6 B/ms); decompress times are
# measured at write. Outputs in /mnt/nvme0/v2-adapt/runs/DATASET_ARM[_lrLR]/:
# v2_pred.csv (per-chunk predictions for all 45 settings before learning),
# selection.csv, explore.csv, blobs.csv (driver report), logs.
#===============================================================================
set -euo pipefail
DS=$1; ARM=$2; LR=${3:-0.01}
ROOT=/mnt/nvme0/v2-adapt
FIELDS=$ROOT/$DS/fields
WEIGHTS=/home/cc/clio-core/context-transport-primitives/src/compress/model/weights/v2
BIN=/home/cc/clio-core/build/bin/neuropress_field_replay
STORE=$ROOT/runs/${DS}_${ARM}_lr${LR}
rm -rf "$STORE"; mkdir -p "$STORE"
case $ARM in
  static)  LEARN=false; EXPLORE=false ;;
  learn)   LEARN=true;  EXPLORE=false ;;
  explore) LEARN=true;  EXPLORE=true ;;
  *) echo "ARM must be static|learn|explore" >&2; exit 1 ;;
esac
PORT=$(python3 -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1])')
MB=$(( $(du -smL "$FIELDS" | cut -f1) + 512 ))
cat > "$STORE/compose.yaml" <<YAML
networking:
  port: $PORT
runtime:
  num_threads: 4
  queue_depth: 1024
compose:
  - mod_name: clio_bdev
    pool_name: "$STORE/chi_bdev.dat"
    pool_query: local
    pool_id: "301.0"
    bdev_type: file
    path: "$STORE/chi_bdev.dat"
    capacity: "$(( MB * 2 ))MB"
  - mod_name: clio_cte_compressor
    pool_name: cte_compressor
    pool_query: local
    pool_id: "512.0"
    next_pool_id: "513.0"
    neuropress_model_path: "$WEIGHTS"
    neuropress_online_learning_enabled: $LEARN
    neuropress_exploration_enabled: $EXPLORE
    neuropress_exploration_k: 3
    neuropress_exploration_threshold: 0.5
    neuropress_learning_rate: $LR
  - mod_name: clio_cte_core
    pool_name: cte_core
    pool_query: local
    pool_id: "513.0"
    storage:
      - path: "$STORE/cte_tier.dat"
        bdev_type: "file"
        capacity_limit: "${MB}MB"
        score: 1.0
        persistence_level: "temporary"
    performance:
      metadata_log_path: "$STORE/cte_metadata_log"
      transaction_log_capacity: "32MB"
    dpe:
      dpe_type: "max_bw"
YAML
export LD_LIBRARY_PATH="/home/cc/clio-core/build/bin:/usr/local/lib:/usr/local/cuda/lib64:/home/cc/np-env/cusz/lib:/home/cc/np-env/cusz/lib64:${LD_LIBRARY_PATH:-}"
START=$(date +%s.%N)
set +e  # the replay process currently segfaults in CUDA teardown at exit (v1 too), after all output is written
env CLIO_SERVER_CONF="$STORE/compose.yaml" CLIO_WITH_RUNTIME=1 \
    CLIO_REPLAY_COMPRESSOR_POOL=512.0 \
    CLIO_NEUROPRESS_STAGE_H2D=1 \
    CLIO_NEUROPRESS_COST_BW=1e6 \
    CLIO_NEUROPRESS_EXPLORE_MEASURE_DT=1 \
    CLIO_NEUROPRESS_V2_PRED_LOG="$STORE/v2_pred.csv" \
    CLIO_NEUROPRESS_SELECTION_LOG="$STORE/selection.csv" \
    CLIO_NEUROPRESS_EXPLORE_LOG="$STORE/explore.csv" \
    CTP_LOG_LEVEL=warning \
    "$BIN" --dir "$FIELDS" --ext .f32 --chunk 0 --tag "v2_${DS}_${ARM}" \
    --report "$STORE/blobs.csv" --verify > "$STORE/stdout.log" 2> "$STORE/runtime.log"
RC=$?; set -e
grep -q "VERIFIED: \([0-9]*\) of \1 " "$STORE/stdout.log" || { echo "run failed (rc=$RC), see $STORE" >&2; exit 1; }
echo "$DS $ARM lr=$LR: $(awk -v a="$START" -v b="$(date +%s.%N)" 'BEGIN{printf "%.1f", b-a}') s, $(($(wc -l < "$STORE/v2_pred.csv")-1)) chunks predicted; $(tail -3 "$STORE/stdout.log" | tr '\n' ' ')"
