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
while [ $# -gt 0 ]; do
  case "$1" in
    --fields) FIELDS=$2; shift 2;;
    --chunk) CHUNK=$2; shift 2;;
    --max-files) MAX_FILES=$2; shift 2;;
    --f64) F64=1; shift;;
    --no-verify) VERIFY=0; shift;;
    --results) RESULTS=$2; shift 2;;
    --tag) TAG=$2; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
export FIELDS

[ -d "$FIELDS" ] || { echo "no field dumps at $FIELDS -- run ./gen_fields.sh first" >&2; exit 1; }

NP_LEARN=false NP_EXPLORE=false EXPLORE_K=0 THRESH=0.5 BEST=false
STATIC_LIB="" STATIC_SHUF=0
COST_ENV=()
RATIO_ONLY=(CLIO_NEUROPRESS_COST_W_CT=0 CLIO_NEUROPRESS_COST_W_DT=0 CLIO_NEUROPRESS_COST_W_IO=1)
case "$CONFIG" in
  dynamic)        ;;
  dynamic-ratio)  COST_ENV=("${RATIO_ONLY[@]}") ;;
  learn)          NP_LEARN=true ;;
  explore)        NP_LEARN=true; NP_EXPLORE=true; EXPLORE_K=31; THRESH=0
                  COST_ENV=("${RATIO_ONLY[@]}") ;;
  best)           BEST=true ;;
  static-zstd)    STATIC_LIB=nvcomp-zstd; STATIC_SHUF=0 ;;
  static-zstd-s4) STATIC_LIB=nvcomp-zstd; STATIC_SHUF=4 ;;
  static-zstd-s8) STATIC_LIB=nvcomp-zstd; STATIC_SHUF=8 ;;
  *) echo "unknown config: $CONFIG" >&2; sed -n '2,24p' "$0" >&2; exit 2;;
esac
export NP_LEARN NP_EXPLORE EXPLORE_K THRESH BEST STATIC_LIB STATIC_SHUF

# shellcheck source=common.sh
. "$HERE/common.sh"
bench_setup || exit 1

NAME=${TAG:-$CONFIG}
STORE=$RESULTS/$NAME
rm -rf "$STORE"; mkdir -p "$STORE"
bench_compose "$STORE"

if [ "$F64" = 1 ]; then EXT=.f64; PREC=float64; else EXT=.f32; PREC=float32; fi
NFILES=$(find "$FIELDS" -name "*$EXT" | wc -l)
[ "$NFILES" -gt 0 ] || { echo "no *$EXT files under $FIELDS" >&2; exit 1; }
PAYLOAD=$(du -sb "$FIELDS" | cut -f1)
echo "== $NAME: $NFILES field file(s), $(awk -v p="$PAYLOAD" 'BEGIN{printf "%.1f", p/1048576}') MiB $PREC, chunk $CHUNK, port $PORT"

ARGS=(--dir "$FIELDS" --ext "$EXT" --chunk "$CHUNK"
      --tag "vpic_$NAME" --report "$STORE/blobs.csv")
[ "$F64" = 1 ] && ARGS+=(--f64)
[ "$MAX_FILES" -gt 0 ] && ARGS+=(--max-files "$MAX_FILES")
[ "$VERIFY" = 1 ]      && ARGS+=(--verify)

export LD_LIBRARY_PATH="$BUILD/bin:/usr/local/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
START=$(date +%s.%N)
set +e
env CLIO_SERVER_CONF="$STORE/compose.yaml" \
    CLIO_WITH_RUNTIME=1 \
    CLIO_REPLAY_COMPRESSOR_POOL=512.0 \
    CLIO_NEUROPRESS_SELECTION_LOG="$STORE/selection.csv" \
    ${NP_EXPLORE:+$([ "$NP_EXPLORE" = true ] && echo CLIO_NEUROPRESS_EXPLORE_LOG="$STORE/explore.csv")} \
    CTP_LOG_LEVEL="${CTP_LOG_LEVEL:-warn}" \
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
{"config":"$CONFIG","tag":"$NAME","rc":$RC,"device":"gpu","workload":"vpic-weibel",
 "atoms":0,"steps":0,"gap":0,"frames":$(ls "$FIELDS" | grep -c '^plt' || echo 0),
 "chunk":$CHUNK,"files":$NFILES,"payload_bytes":$PAYLOAD,"wall_s":$WALL,
 "verified":$VERIFY,"port":$PORT,
 "physics":{"problem":"weibel","precision":"$PREC"}}
JSON
grep -E "^stored|VERIFIED|FAILED:|  time:" "$STORE/stdout.log" | sed 's/^/   /' || true
exit $RC
