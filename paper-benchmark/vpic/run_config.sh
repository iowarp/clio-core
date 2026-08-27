#!/usr/bin/env bash
# Run ONE named configuration of the VPIC paper benchmark.
#
#   ./run_config.sh <config> [--fields DIR] [--chunk B] [--max-files N]
#                            [--f64] [--no-verify] [--results DIR] [--tag NAME]
#
# --f64 replays .f64 dumps (a double Nyx build with NYX_DUMP_NATIVE=1) instead
# of .f32. The element width is the point of that control: the byte shuffle has
# to match it, so the same physics at both widths is what shows the stride
# actually matters rather than the dataset happening to prefer one.
#
# Replays the field dumps gen_fields.sh produced. Configurations differ only in
# how a codec is chosen per chunk -- identical to the LAMMPS benchmark's, so the
# two workloads are directly comparable:
#
#   dynamic        NeuroPress inference, default balanced cost model
#   dynamic-ratio  same, latency weights zeroed (ratio-only objective)
#   learn          dynamic + online SGD from measured outcomes
#   explore        ratio-only ranking, top-K alternatives measured, winner kept
#   best           best mode: exhaustive, ratio-only
#   static-zstd    fixed nvcomp-zstd, no shuffle
#   static-zstd-s4 fixed nvcomp-zstd + 4-byte shuffle -- the stride matching
#                  VPIC's float32 fields
#   static-zstd-s8 fixed nvcomp-zstd + 8-byte shuffle -- the mismatched stride,
#                  included because a shuffle claim that only holds on
#                  compressible data is not a claim about the element width
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

case "${1:-}" in
  -h|--help) sed -n '2,24p' "$0"; exit 0;;
esac
CONFIG=${1:-dynamic}; shift || true

FIELDS=${FIELDS:-$HERE/fields}
CHUNK=4194304 MAX_FILES=0 VERIFY=1 RESULTS="$HERE/results" TAG="" F64=0
# --- Cost-model bandwidth, error bound, exploration policy ----------------
# BW is CLIO_NEUROPRESS_COST_BW in BYTES PER MILLISECOND -- the unit of
# RankingWeights::bandwidth_bytes_per_ms (predictor.h), NOT bytes/s or GB/s.
# 1 GB/s = 1e6 B/ms; the shipped default 5e6 is 5 GB/s. It enters the cost as
# bytes/(ratio*bw), so it only changes a DECISION when some other term is
# non-zero -- under the ratio-only weights it is a positive scalar on the sole
# term and cannot reorder candidates at all. See ../BENCHMARK.md.
#
# EB is CLIO_NEUROPRESS_ERROR_BOUND, an ABSOLUTE bound: |orig - decoded| <= EB.
# 0 (empty) is lossless and masks NeuroPress's 16 quantize actions.
#
# THRESH_OPT=0 makes exploration UNCONDITIONAL. That is deliberate for this
# benchmark and not upstream's 0.5: at 0.5 a chunk explores only when the cost
# prediction was already off by >50%, so (a) the number of explored chunks
# varies with the cost model and bandwidth being compared, and (b) it varies
# run to run on a non-bit-reproducible workload -- measured 16/45 then 33/45
# on the same LAMMPS settings. Both make the 2x2 matrix less comparable.
# At 0 nearly every chunk explores, so nearly every compressed chunk yields a
# measured decompression time and the only variable left is the one under test.
#
# NEARLY, not every: the gate is `error_pct > threshold`, STRICT, and it is
# upstream's own (gpucompress_compress.cpp:733). A chunk whose cost the model
# predicted EXACTLY right has error_pct == 0 and does not explore at any
# threshold, 0 included. That is rare under the balance weights and common
# under the ratio ones, because ratio-only cost is bytes/(min(ratio,100)*bw)
# and nothing else: once the predicted AND actual ratio both clear the 100x
# cap the two costs are the same number, so the model is "exactly right" by
# arithmetic rather than by accuracy. Measured on Nyx 128^3, explore-ratio:
# 56 of 120 chunks lossless and 97 of 120 lossy never explored. Read
# `explored` in summary.csv rather than assuming full coverage.
# K stays at 3, NeuroPress's own ranked window.
BW="" EB="" EXPLORE_K_OPT=3 THRESH_OPT=0 CHECK_BOUND=0
while [ $# -gt 0 ]; do
  case "$1" in
    --fields) FIELDS=$2; shift 2;;
    --chunk) CHUNK=$2; shift 2;;
    --max-files) MAX_FILES=$2; shift 2;;
    --f64) F64=1; shift;;
    --no-verify) VERIFY=0; shift;;
    --results) RESULTS=$2; shift 2;;
    --tag) TAG=$2; shift 2;;
    --bw) BW=$2; shift 2;;
    --eb) EB=$2; shift 2;;
    --explore-k) EXPLORE_K_OPT=$2; shift 2;;
    --explore-thresh) THRESH_OPT=$2; shift 2;;
    --check-bound) CHECK_BOUND=1; shift;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
export FIELDS

[ -d "$FIELDS" ] || { echo "no field dumps at $FIELDS -- run ./gen_fields.sh first" >&2; exit 1; }

NP_LEARN=false NP_EXPLORE=false EXPLORE_K=0 THRESH=0.5 BEST=false
STATIC_LIB="" STATIC_SHUF=0
COST_ENV=()
# Which of the two cost models this config asks for, recorded in meta.json so
# a run is self-describing. The pre-existing configs keep their historical
# behaviour: `dynamic`/`learn` rank under the default balanced weights, the
# rest under the ratio-only ones.
case "$CONFIG" in
  dynamic|learn|explore-balance) COSTMODEL=balance ;;
  static-*)                      COSTMODEL=none ;;
  *)                             COSTMODEL=ratio ;;
esac
RATIO_ONLY=(CLIO_NEUROPRESS_COST_W_CT=0 CLIO_NEUROPRESS_COST_W_DT=0 CLIO_NEUROPRESS_COST_W_IO=1)
case "$CONFIG" in
  dynamic)        ;;
  dynamic-ratio)  COST_ENV=("${RATIO_ONLY[@]}") ;;
  learn)          NP_LEARN=true ;;
  explore)        NP_LEARN=true; NP_EXPLORE=true; EXPLORE_K=31; THRESH=0
                  COST_ENV=("${RATIO_ONLY[@]}") ;;
  best)           BEST=true ;;
  # EXPLORATION MODE, the two cost models. NP_LEARN=true is NOT optional and
  # not an extra experimental axis: the per-chunk features that gate the
  # exploration block are computed only when online learning or best mode is
  # on (compressor_runtime.cc:1145-1152), so exploration with it off silently
  # returns a plain inference result. `explore` above has always set it for
  # the same reason.
  #
  # K and the threshold come from --explore-k / --explore-thresh. K defaults
  # to 3, NeuroPress's own ranked window, rather than the exhaustive 31 that
  # `explore` above pins for action-space studies; the threshold defaults to
  # 0 for the comparability reason given where it is declared.
  explore-balance) NP_LEARN=true; NP_EXPLORE=true
                   EXPLORE_K=$EXPLORE_K_OPT; THRESH=$THRESH_OPT ;;
  explore-ratio)   NP_LEARN=true; NP_EXPLORE=true
                   EXPLORE_K=$EXPLORE_K_OPT; THRESH=$THRESH_OPT
                   COST_ENV=("${RATIO_ONLY[@]}") ;;
  static-zstd)    STATIC_LIB=nvcomp-zstd; STATIC_SHUF=0 ;;
  static-zstd-s4) STATIC_LIB=nvcomp-zstd; STATIC_SHUF=4 ;;
  static-zstd-s8) STATIC_LIB=nvcomp-zstd; STATIC_SHUF=8 ;;
  *) echo "unknown config: $CONFIG" >&2; sed -n '2,24p' "$0" >&2; exit 2;;
esac
export NP_LEARN NP_EXPLORE EXPLORE_K THRESH BEST STATIC_LIB STATIC_SHUF

# A positive error bound means the decompressed bytes are NOT the bytes that
# went in, by design. Every verify path here is an FNV-1a digest comparison, so
# under lossy compression it would report FAILED on a run that is behaving
# exactly as asked. Turn it off and say so; the quality number for a lossy run
# is PSNR in selection.csv, not a digest.
MODE=lossless
if [ -n "$EB" ] && awk -v e="$EB" 'BEGIN{exit !(e+0>0)}'; then
  MODE=lossy
  if [ "$CHECK_BOUND" = 1 ]; then
    # The replay driver can do the check that DOES apply to lossy data:
    # re-read each chunk from its source file and compare element-wise
    # against the error bound. Verification stays on for that.
    echo "   (lossy eb=$EB: verifying |orig-decoded| <= $EB instead of a digest)"
  elif [ "${VERIFY:-0}" = 1 ]; then
    echo "   (lossy eb=$EB: bit-exact verification disabled; pass --check-bound"
    echo "    to verify the error bound instead -- see run_config.sh)"
    VERIFY=0
  fi
fi

# shellcheck source=common.sh
. "$HERE/common.sh"
bench_setup || exit 1

NAME=${TAG:-$CONFIG}
STORE=$RESULTS/$NAME
rm -rf "$STORE"; mkdir -p "$STORE"
bench_compose "$STORE"

if [ "$F64" = 1 ]; then EXT=.f64; PREC=float64; else EXT=.f32; PREC=float32; fi
# Timesteps behind these dumps, from the manifest gen_fields.sh leaves beside
# them; see ../nyx/run_config.sh. Dump directories made before the manifest
# existed have none, and still report 0.
GEN_STEPS=0
if [ -f "$FIELDS/gen.json" ]; then
  GEN_STEPS=$(python3 -c "import json;print(json.load(open('$FIELDS/gen.json')).get('steps',0))" 2>/dev/null || echo 0)
fi

NFILES=$(find "$FIELDS" -name "*$EXT" | wc -l)
[ "$NFILES" -gt 0 ] || { echo "no *$EXT files under $FIELDS" >&2; exit 1; }
PAYLOAD=$(du -sb "$FIELDS" | cut -f1)
echo "== $NAME: $NFILES field file(s), $(awk -v p="$PAYLOAD" 'BEGIN{printf "%.1f", p/1048576}') MiB $PREC, chunk $CHUNK, port $PORT"

ARGS=(--dir "$FIELDS" --ext "$EXT" --chunk "$CHUNK"
      --tag "vpic_$NAME" --report "$STORE/blobs.csv")
[ "$F64" = 1 ] && ARGS+=(--f64)
[ "$MAX_FILES" -gt 0 ] && ARGS+=(--max-files "$MAX_FILES")
[ "$VERIFY" = 1 ]      && ARGS+=(--verify)
[ "$CHECK_BOUND" = 1 ] && [ "$MODE" = lossy ] && ARGS+=(--check-bound)

export LD_LIBRARY_PATH="$BUILD/bin:/usr/local/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
START=$(date +%s.%N)
set +e
env CLIO_SERVER_CONF="$STORE/compose.yaml" \
    CLIO_WITH_RUNTIME=1 \
    CLIO_REPLAY_COMPRESSOR_POOL=512.0 \
    CLIO_NEUROPRESS_SELECTION_LOG="$STORE/selection.csv" \
    ${NP_EXPLORE:+$([ "$NP_EXPLORE" = true ] && echo CLIO_NEUROPRESS_EXPLORE_LOG="$STORE/explore.csv")} \
    CTP_LOG_LEVEL="${CTP_LOG_LEVEL:-warning}" \
    ${EB:+CLIO_NEUROPRESS_ERROR_BOUND=$EB} \
    ${BW:+CLIO_NEUROPRESS_COST_BW=$BW} \
    CLIO_NEUROPRESS_EXPLORE_MEASURE_DT=${MEASURE_DT:-1} \
    "${COST_ENV[@]}" \
    "$BIN" "${ARGS[@]}" > "$STORE/stdout.log" 2> "$STORE/runtime.log"
RC=$?
set -e
WALL=$(awk -v a="$START" -v b="$(date +%s.%N)" 'BEGIN{printf "%.2f", b-a}')

if [ $RC -ne 0 ]; then
  echo "   FAILED rc=$RC"
  grep -vE "DEBUG|INFO" "$STORE/runtime.log" | tail -8
  grep -q "Address already in use" "$STORE/runtime.log" && \
    echo "   (port $PORT was taken; rerun -- a free port is picked per run)"
fi

cat > "$STORE/meta.json" <<JSON
{"config":"$CONFIG","tag":"$NAME","rc":$RC,"mode":"$MODE",
 "cost_model":"$COSTMODEL","bw_bytes_per_ms":${BW:-5e6},
 "error_bound":${EB:-0},"explore_k":$EXPLORE_K,"explore_thresh":$THRESH,"device":"gpu","workload":"vpic-weibel",
 "atoms":0,"steps":$GEN_STEPS,"gap":0,"frames":$(ls "$FIELDS" | grep -c '^plt' || echo 0),
 "chunk":$CHUNK,"files":$NFILES,"payload_bytes":$PAYLOAD,"wall_s":$WALL,
 "verified":$VERIFY,"port":$PORT,
 "physics":{"problem":"weibel","precision":"$PREC"}}
JSON
grep -E "^stored|VERIFIED|FAILED:|  time:" "$STORE/stdout.log" | sed 's/^/   /' || true
exit $RC
