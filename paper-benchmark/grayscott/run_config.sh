#!/usr/bin/env bash
# Run ONE named configuration of the Gray-Scott paper benchmark.
#
#   ./run_config.sh <config> [--L N] [--steps N] [--gap N] [--chunk B]
#                            [--regime spots|stripes|chaos] [--results DIR]
#                            [--tag NAME] [--verify|--no-verify]
#
# Configurations differ only in how a codec is chosen per chunk:
#
#   dynamic        NeuroPress inference, default balanced cost model
#   dynamic-ratio  same, latency weights zeroed (ratio-only objective)
#   learn          dynamic + online SGD from measured outcomes
#   explore        ratio-only ranking, top-K alternatives measured, winner kept
#   best           best mode: exhaustive, ratio-only
#   static-zstd    fixed nvcomp-zstd, no shuffle
#   static-zstd-s4 fixed nvcomp-zstd + 4-byte shuffle -- the RIGHT stride here,
#                  because Gray-Scott is float32
#   static-zstd-s8 fixed nvcomp-zstd + 8-byte shuffle -- the mismatched stride
#
# The application writes HDF5 and links nothing from Clio; the VOL connector
# compresses on the way past, so this is in situ with no replay phase.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONFIG=${1:-}; shift || true
[ -n "$CONFIG" ] || { sed -n '2,20p' "$0"; exit 2; }

L=128 STEPS=200 GAP=25 CHUNK=1048576 REGIME=spots
RESULTS="$HERE/results" TAG="" VERIFY=1
while [ $# -gt 0 ]; do
  case "$1" in
    --L) L=$2; shift 2;;
    --steps) STEPS=$2; shift 2;;
    --gap) GAP=$2; shift 2;;
    --chunk) CHUNK=$2; shift 2;;
    --regime) REGIME=$2; shift 2;;
    --results) RESULTS=$2; shift 2;;
    --tag) TAG=$2; shift 2;;
    --verify) VERIFY=1; shift;;
    --no-verify) VERIFY=0; shift;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
NAME=${TAG:-$CONFIG}

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
  *) echo "unknown config: $CONFIG" >&2; sed -n '2,20p' "$0" >&2; exit 2;;
esac
export NP_LEARN NP_EXPLORE EXPLORE_K THRESH BEST STATIC_LIB STATIC_SHUF

# shellcheck source=common.sh
. "$HERE/common.sh"
bench_setup || exit 1

STORE="$RESULTS/$NAME"
rm -rf "$STORE"; mkdir -p "$STORE"
# HDF5 finds the connector by name under HDF5_PLUGIN_PATH.
ln -sf "$VOL" "$STORE/libclio_hdf5_vol.so"

NSNAP=$(( STEPS / GAP ))
SNAP_MB=$(( L * L * L * 4 / 1048576 )); [ "$SNAP_MB" -lt 1 ] && SNAP_MB=1
TIER_MB=$(( SNAP_MB * NSNAP + 512 ))
PORT=$(bench_port)
bench_compose "$STORE" "$PORT" "$TIER_MB"

echo "== $NAME: Gray-Scott ${L}^3 $REGIME, $STEPS steps, snapshot every $GAP, chunk $CHUNK, port $PORT"
export LD_LIBRARY_PATH="$BUILD/bin:/usr/local/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
START=$(date +%s.%N)
set +e
env CLIO_SERVER_CONF="$STORE/compose.yaml" \
    CLIO_WITH_RUNTIME=1 \
    HDF5_VOL_CONNECTOR=clio HDF5_PLUGIN_PATH="$STORE" \
    CLIO_VOL_COMPRESSOR_POOL=512.0 \
    CLIO_VOL_CHUNK_SIZE="$CHUNK" \
    `# The VOL stamps blob names; the reader must derive the SAME names, so`\
    `# pin the granularity in both phases rather than take a default.`\
    CLIO_VOL_STAMP_GRANULARITY_NS=0 \
    GS_REGIME="$REGIME" \
    CTP_LOG_LEVEL="${CTP_LOG_LEVEL:-warning}" \
    "${COST_ENV[@]}" \
    "$BIN" "$STORE/gs.h5" "$L" "$STEPS" "$GAP" write \
    > "$STORE/write.log" 2> "$STORE/runtime.log"
RC=$?
set -e
WALL=$(awk -v a="$START" -v b="$(date +%s.%N)" 'BEGIN{printf "%.2f", b-a}')
if [ $RC -ne 0 ]; then
  echo "   FAILED rc=$RC"; grep -vE "DEBUG|INFO" "$STORE/runtime.log" | tail -6
fi

# Per-chunk CSV from the VOL path trace -- the same converter the WarpX sweep
# uses, because the staging happens inside the VOL in both and there is no
# Clio driver recording chunks itself.
"$HERE/../warpx/trace_to_csv.py" "$STORE/runtime.log" "$STORE/blobs.csv" \
    ${NP_EXPLORE:+--explore-log "$STORE/explore.csv"} > "$STORE/convert.log" 2>&1 || true
NCHUNKS=$(( $(wc -l < "$STORE/blobs.csv" 2>/dev/null || echo 1) - 1 ))
PAYLOAD=$(awk -F, 'NR>1{s+=$2} END{printf "%d", s+0}' "$STORE/blobs.csv" 2>/dev/null || echo 0)

VERIFY_RESULT="n/a"
if [ "$VERIFY" = 1 ] && [ $RC -eq 0 ]; then
  # A separate process, its own runtime, CLIO_RESTART=1: the tier and the
  # metadata log are all it shares with the writer.
  #
  # CLIO_VOL_CHUNK_SIZE must MATCH THE WRITE. The VOL names blobs by chunk
  # index, so reading at a different chunk size asks for names nothing stored,
  # every lookup misses, and HDF5 serves the native file instead -- bit-exact
  # and completely uninformative.
  set +e
  env CLIO_SERVER_CONF="$STORE/compose.yaml" \
      CLIO_WITH_RUNTIME=1 CLIO_RESTART=1 \
      HDF5_VOL_CONNECTOR=clio HDF5_PLUGIN_PATH="$STORE" \
      CLIO_VOL_COMPRESSOR_POOL=512.0 \
      CLIO_VOL_CHUNK_SIZE="$CHUNK" \
      CLIO_VOL_STAMP_GRANULARITY_NS=0 \
      CTP_LOG_LEVEL="${CTP_LOG_LEVEL:-warning}" \
      "$BIN" "$STORE/gs.h5" "$L" "$STEPS" "$GAP" read \
      > "$STORE/verify.log" 2>&1
  VRC=$?
  set -e
  # Two conditions, and both must hold. Bit-exactness alone proves nothing
  # here: a miss falls through to native HDF5, which returns perfect bytes
  # having never touched the compressor. The path trace saying a codec was
  # inverted is what makes the check real.
  INV=$(grep -c 'inverting codec' "$STORE/verify.log" || true)
  if [ $VRC -eq 0 ] && grep -qE "VERIFY OK" "$STORE/verify.log" && [ "${INV:-0}" -gt 0 ]; then
    VERIFY_RESULT="pass"
  elif [ $VRC -eq 0 ] && [ "${INV:-0}" -eq 0 ]; then
    VERIFY_RESULT="INCONCLUSIVE"
  else
    VERIFY_RESULT="FAIL"
  fi
  echo "   verify: $VERIFY_RESULT ($INV chunk(s) inverted by a codec)"
fi

cat > "$STORE/meta.json" <<JSON
{"config":"$CONFIG","tag":"$NAME","rc":$RC,"device":"gpu","workload":"grayscott",
 "atoms":0,"steps":$STEPS,"gap":$GAP,"frames":$NSNAP,"chunk":$CHUNK,
 "files":$NCHUNKS,"payload_bytes":$PAYLOAD,"wall_s":$WALL,
 "verified":$VERIFY,"verify_result":"$VERIFY_RESULT","port":$PORT,
 "physics":{"problem":"gray_scott_$REGIME","precision":"float32","insitu":"yes"}}
JSON
echo "   $NCHUNKS chunk(s), $(awk -F, 'NR>1{i+=$2;s+=$7} END{printf "%.3fx", (s>0)? i/s : 0}' "$STORE/blobs.csv" 2>/dev/null) , wall ${WALL}s"
exit $RC
