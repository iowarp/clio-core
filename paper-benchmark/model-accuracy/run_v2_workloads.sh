#!/usr/bin/env bash
#===============================================================================
# run_v2_workloads.sh -- stream one new-workloads dataset through Clio's
# compressor with NeuroPress v2 on the 4-tier cost model.
#
#   run_v2_workloads.sh DATASET MODE
#     DATASET  a directory under /mnt/nvme0/v2-work (stage_v2_workloads.py)
#     MODE     static     shipped v2 weights, no learning, no exploration
#              learn      online learning only (30 % cost-error gate, learning
#                         rate 0.5), no exploration: v2 learns from its own
#                         pick alone, on the write only (no NeuroPress work on
#                         reads); NeuroPress never explores (user, 2026-10-05)
#              exhaustive no learning; all 44 alternatives measured on every
#                         chunk (Clio's own per-chunk ground truth); codec
#                         times and ratio only, one wall clock for all
#                         settings (run_exhaustive_e2e.sh: one full run per
#                         setting, each with its own application time)
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
# Optional (environment), for the producer + analysis model (1 write, R reads):
#   TIERS="0.52e6:1"  one tier instead of the 4 (bandwidth in bytes per ms)
#   COST_W="1,4,5"    cost weights W_CT,W_DT,W_IO (default 1,1,1)
#   COST_BW=520000    bandwidth of the learning gate's cost, bytes per ms
#                     (Clio's default 5e6); set it to the one tier's
#   READS=4           timed read-backs of every chunk (default 1)
#   KMEANS=8          a k-means consumer with 8 clusters on the GPU: each
#                     timed read-back is one iteration (needs no other flag)
#   READ_INFLIGHT=8   keep 8 read-backs in flight (default 1); with KMEANS a
#                     pass's time then holds the consumer too
#   CLIO_REPLAY_INFLIGHT=8  keep 8 chunk writes in flight (default: one at a
#                     time); read by the driver from the environment
# Measurement (all modes the same; user, 2026-10-05):
#   PRELOAD=1         input files read into memory before the timed window
#                     (CLIO_REPLAY_PRELOAD; default 1): the write window holds
#                     compress and store only, as for a producer whose data
#                     are in memory
#   PREWARM=K         K codec objects of every setting built and warmed at
#                     start (CLIO_NEUROPRESS_PREWARM; default: the chunks in
#                     flight + 1, as many as GPU memory allows; the driver
#                     submits a chunk before it waits for the oldest, so one
#                     more than in flight can run), so no chunk pays a codec's
#                     first-use setup inside the timed work; CUDA kernels are
#                     loaded at start too (CUDA_MODULE_LOADING=EAGER)
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
# The learning gate: train when the picked setting's weighted-cost error is
# above MAPE (Clio's default 0.30; user, 2026-10-05).
MAPE=${MAPE_THRESHOLD:-0.30}
case $MODE in
  static)     LEARN=false; EXPLORE=false; K=0;  THRESH=$MAPE ;;
  learn)      LEARN=true;  EXPLORE=false; K=0;  THRESH=$MAPE ;;
  exhaustive) LEARN=false; EXPLORE=true;  K=44; THRESH=0.0 ;;
  raw)        LEARN=false; EXPLORE=false; K=0;  THRESH=$MAPE; EXTRA=(--no-compress) ;;
  oracle)     LEARN=false; EXPLORE=false; K=0;  THRESH=$MAPE
              [ -f "${ORACLE_MAP:-}" ] || { echo "oracle needs ORACLE_MAP" >&2; exit 1; } ;;
  fixed)      LEARN=false; EXPLORE=false; K=0;  THRESH=$MAPE
              [ -n "${FIXED_SETTING:-}" ] || { echo "fixed needs FIXED_SETTING" >&2; exit 1; } ;;
  *) echo "MODE must be static|learn|exhaustive|fixed|oracle|raw" >&2; exit 1 ;;
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
    neuropress_mape_threshold: $MAPE
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
# No decompress on the write path in the learning modes (user decision,
# 2026-10-05), and NeuroPress does no work on reads (user, 2026-10-05): it
# learns compress time and ratio when it writes a chunk and never learns the
# decompress time online. LEARN_MEASURE_DT=1 measures a decompress on the
# write path again (a label for learning, inside the write time).
MEASURE_DT=${LEARN_MEASURE_DT:-0}; FIXED=""; SETTING_MAP=""
TIERS=${TIERS:-12e6:1,1e6:3,0.5e6:3,0.25e6:3}
# The weights and the gate bandwidth go to Clio only when given: setting
# any CLIO_NEUROPRESS_COST_* also overrides the v1 bridge's shipped weights.
COST_ENV=()
if [ -n "${COST_W:-}" ]; then
  IFS=, read -r W_CT W_DT W_IO <<< "$COST_W"
  COST_ENV+=(CLIO_NEUROPRESS_COST_W_CT="$W_CT" CLIO_NEUROPRESS_COST_W_DT="$W_DT"
             CLIO_NEUROPRESS_COST_W_IO="$W_IO")
fi
[ -n "${COST_BW:-}" ] && COST_ENV+=(CLIO_NEUROPRESS_COST_BW="$COST_BW")
READ_OPTS=(--read-repeat "${READS:-1}")
[ "${READ_INFLIGHT:-1}" -gt 1 ] && READ_OPTS+=(--read-inflight "$READ_INFLIGHT")
[ "${KMEANS:-0}" -gt 0 ] && READ_OPTS+=(--kmeans "$KMEANS" --read-to-gpu)
[ "$MODE" = exhaustive ] && MEASURE_DT=1   # the search measures every setting's decompress time
[ "$MODE" = fixed ] && { MEASURE_DT=0; FIXED="$FIXED_SETTING"; }
[ "$MODE" = oracle ] && { MEASURE_DT=0; SETTING_MAP="$ORACLE_MAP"; }
INF=$(( ${CLIO_REPLAY_INFLIGHT:-1} > ${READ_INFLIGHT:-1} ? ${CLIO_REPLAY_INFLIGHT:-1} : ${READ_INFLIGHT:-1} ))
MEASURE_ENV=(CLIO_REPLAY_PRELOAD="${PRELOAD:-1}" CUDA_MODULE_LOADING=EAGER
             CLIO_NEUROPRESS_PREWARM="${PREWARM:-$(( INF + 1 ))}")
START=$(date +%s.%N)
set +e  # the replay process segfaults in CUDA teardown at exit (v1 too), after all output is written
env CLIO_SERVER_CONF="$STORE/compose.yaml" CLIO_WITH_RUNTIME=1 \
    CLIO_REPLAY_COMPRESSOR_POOL=512.0 \
    CLIO_NEUROPRESS_STAGE_H2D=1 \
    CLIO_NEUROPRESS_TIERS="$TIERS" \
    "${COST_ENV[@]}" "${MEASURE_ENV[@]}" \
    CLIO_NEUROPRESS_EXPLORE_MEASURE_DT="$MEASURE_DT" \
    CLIO_NEUROPRESS_FIXED_SETTING="$FIXED" \
    CLIO_NEUROPRESS_SETTING_MAP="$SETTING_MAP" \
    CLIO_NEUROPRESS_V2_PRED_LOG="$STORE/v2_pred.csv" \
    CLIO_NEUROPRESS_V2_EXPLORE_LOG="$STORE/v2_measured.csv" \
    CLIO_NEUROPRESS_PHASE_LOG="$STORE/phases.csv" \
    CLIO_NEUROPRESS_SELECTION_LOG="$SELECTION_LOG" \
    CTP_LOG_LEVEL=warning \
    "$BIN" --dir "$FIELDS" --ext .chunk --chunk 0 --dtype-from-name \
    --tag "v2_${DS}_${MODE}" --report "$STORE/blobs.csv" --verify "${READ_OPTS[@]}" "${EXTRA[@]}" \
    > "$STORE/stdout.log" 2> "$STORE/runtime.log"
RC=$?; set -e
grep -q "VERIFIED: \([0-9]*\) of \1 " "$STORE/stdout.log" || { echo "run failed (rc=$RC), see $STORE" >&2; exit 1; }
echo "$DS $MODE: $(awk -v a="$START" -v b="$(date +%s.%N)" 'BEGIN{printf "%.1f", b-a}') s, $(($(wc -l < "$STORE/blobs.csv")-1)) chunks; $(grep -m1 VERIFIED "$STORE/stdout.log")"
