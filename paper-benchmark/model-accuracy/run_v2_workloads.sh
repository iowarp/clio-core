#!/usr/bin/env bash
#===============================================================================
# run_v2_workloads.sh -- stream one new-workloads dataset through Clio's
# compressor with NeuroPress v2 on the 4-tier cost model.
#
#   run_v2_workloads.sh DATASET MODE
#     DATASET  a directory under /mnt/nvme0/v2-work (stage_v2_workloads.py)
#     MODE     static     shipped v2 weights, no learning, no exploration
#              learn      online learning only (15 % gate, learning rate 0.5),
#                         no exploration: v2 learns from its own pick alone
#              learnexp   online learning + exploration, both gated at 15 %
#                         cost error; exploration measures the next K = 4
#                         ranked settings (with the pick: the top 5 of 45,
#                         10 % rounded up); learning rate 0.5
#              exhaustive no learning; all 44 alternatives measured on every
#                         chunk (Clio's own per-chunk ground truth)
#              fixed      fixed-single-codec baseline: every chunk uses the
#                         setting FIXED_SETTING (index or spec); no v2 work,
#                         no learning, no decompress-for-labels
#              oracle     each chunk stored with its own best setting from the
#                         map ORACLE_MAP (oracle_map.py); no v2 work
#              raw        no compressor: raw PutBlob/GetBlob through Clio, the
#                         end-to-end baseline (no v2 logs, no phases.csv)
# Tiers: chunks dealt round-robin in arrival order to 12 / 1 / 0.5 / 0.25 GB/s
# in 1:3:3:3 (CLIO_NEUROPRESS_TIERS); cost = compress + decompress + bytes /
# tier bandwidth. Every chunk is read back (--verify) and checked bit-exact.
# Each chunk's element type rides in its name (--dtype-from-name); v2 turns
# it into float32 for its features only, timed and kept out of every time.
# Output: /mnt/nvme0/v2-work/runs/DATASET_MODE/{v2_pred,v2_measured,phases,
# selection,blobs}.csv and logs.
#===============================================================================
set -euo pipefail
DS=$1; MODE=$2
ROOT=/mnt/nvme0/v2-work
FIELDS=$ROOT/$DS/fields
WEIGHTS=/home/cc/clio-core/context-transport-primitives/src/compress/model/weights/v2
BIN=/home/cc/clio-core/build/bin/neuropress_field_replay
# RUN_TAG (optional) keeps a run apart from the untagged one: <ds>_<mode>_<tag>.
STORE=$ROOT/runs/${DS}_${MODE}${RUN_TAG:+_$RUN_TAG}
rm -rf "$STORE"; mkdir -p "$STORE"
EXTRA=()
case $MODE in
  static)     LEARN=false; EXPLORE=false; K=0;  THRESH=0.15 ;;
  learn)      LEARN=true;  EXPLORE=false; K=0;  THRESH=0.15 ;;
  learnexp)   LEARN=true;  EXPLORE=true;  K=4;  THRESH=0.15 ;;
  exhaustive) LEARN=false; EXPLORE=true;  K=44; THRESH=0.0 ;;
  raw)        LEARN=false; EXPLORE=false; K=0;  THRESH=0.15; EXTRA=(--no-compress) ;;
  oracle)     LEARN=false; EXPLORE=false; K=0;  THRESH=0.15
              [ -f "${ORACLE_MAP:-}" ] || { echo "oracle needs ORACLE_MAP" >&2; exit 1; } ;;
  fixed)      LEARN=false; EXPLORE=false; K=0;  THRESH=0.15
              [ -n "${FIXED_SETTING:-}" ] || { echo "fixed needs FIXED_SETTING" >&2; exit 1; } ;;
  *) echo "MODE must be static|learn|learnexp|exhaustive|fixed|oracle|raw" >&2; exit 1 ;;
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
    neuropress_exploration_k: $K
    neuropress_exploration_threshold: $THRESH
    neuropress_mape_threshold: 0.15
    neuropress_learning_rate: 0.5
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
# NO_SELECTION_LOG=1 turns the selection log off (an empty path disables it):
# it checksums every chunk on the host inside the timed write path.
SELECTION_LOG="$STORE/selection.csv"
[ -n "${NO_SELECTION_LOG:-}" ] && SELECTION_LOG=""
# The decompress that measures learning labels serves learning only.
MEASURE_DT=1; FIXED=""; SETTING_MAP=""
[ "$MODE" = fixed ] && { MEASURE_DT=0; FIXED="$FIXED_SETTING"; }
[ "$MODE" = oracle ] && { MEASURE_DT=0; SETTING_MAP="$ORACLE_MAP"; }
START=$(date +%s.%N)
set +e  # the replay process segfaults in CUDA teardown at exit (v1 too), after all output is written
env CLIO_SERVER_CONF="$STORE/compose.yaml" CLIO_WITH_RUNTIME=1 \
    CLIO_REPLAY_COMPRESSOR_POOL=512.0 \
    CLIO_NEUROPRESS_STAGE_H2D=1 \
    CLIO_NEUROPRESS_TIERS="12e6:1,1e6:3,0.5e6:3,0.25e6:3" \
    CLIO_NEUROPRESS_EXPLORE_MEASURE_DT="$MEASURE_DT" \
    CLIO_NEUROPRESS_FIXED_SETTING="$FIXED" \
    CLIO_NEUROPRESS_SETTING_MAP="$SETTING_MAP" \
    CLIO_NEUROPRESS_V2_PRED_LOG="$STORE/v2_pred.csv" \
    CLIO_NEUROPRESS_V2_EXPLORE_LOG="$STORE/v2_measured.csv" \
    CLIO_NEUROPRESS_PHASE_LOG="$STORE/phases.csv" \
    CLIO_NEUROPRESS_SELECTION_LOG="$SELECTION_LOG" \
    CTP_LOG_LEVEL=warning \
    "$BIN" --dir "$FIELDS" --ext .chunk --chunk 0 --dtype-from-name \
    --tag "v2_${DS}_${MODE}" --report "$STORE/blobs.csv" --verify "${EXTRA[@]}" \
    > "$STORE/stdout.log" 2> "$STORE/runtime.log"
RC=$?; set -e
grep -q "VERIFIED: \([0-9]*\) of \1 " "$STORE/stdout.log" || { echo "run failed (rc=$RC), see $STORE" >&2; exit 1; }
echo "$DS $MODE: $(awk -v a="$START" -v b="$(date +%s.%N)" 'BEGIN{printf "%.1f", b-a}') s, $(($(wc -l < "$STORE/blobs.csv")-1)) chunks; $(grep -m1 VERIFIED "$STORE/stdout.log")"
