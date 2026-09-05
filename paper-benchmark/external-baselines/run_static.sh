#!/usr/bin/env bash
# Run ONE fixed-codec arm. Generalizes run_cusz.sh to any codec in the
# registry, so the external compressors (cusz, cuszp, ndzip) can be measured
# on exactly the chunks Phase 1 swept.
#
# The codec is reached through neuropress_static_lib, a PRE-EXISTING compose
# key: when it is set the runtime compresses every chunk with that library and
# NeuroPress's selection is never consulted -- no inference, no learning, no
# exploration. paper-benchmark's own common.sh is sourced so pool layout,
# weights path, tier sizing and port selection are byte-for-byte what every
# other arm gets; only the codec differs.
#
# FIDELITY: cusz and cuszp take their error bound from the PRESET, and the
# static path pins preset 2 (BALANCED) = 1e-3 absolute -- the same bound
# Phase 1 ran at, so stored bytes are comparable. ndzip is LOSSLESS and has no
# preset, so it is verified bit-exact instead of against a bound.
#
#   run_static.sh <codec> <fields-dir> <max-files> <chunk-bytes> <eb> <store-dir>
set -uo pipefail
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
export BUILD=${BUILD:-$REPO/build}
CODEC=$1; FIELDS=$2; MAXF=$3; CHUNK=$4; EB=$5; STORE=$6
export FIELDS

# Where each external codec's shared library lives. A codec whose .so is not
# on the path loads as a missing symbol at runtime, not a clean error.
case "$CODEC" in
  cusz)  EXTRA_LIB=$NPENV/cusz/lib64  ; LOSSLESS=0 ;;
  cuszp) EXTRA_LIB=$NPENV/cuszp/lib64 ; LOSSLESS=0 ;;
  ndzip) EXTRA_LIB=$NPENV/ndzip/lib   ; LOSSLESS=1 ;;
  *)     EXTRA_LIB=""                             ; LOSSLESS=0 ;;
esac

# Named so the compose file is not built from unset variables. The runtime
# forces selection/learning/exploration off whenever a static library is set.
export NP_LEARN=false NP_EXPLORE=false EXPLORE_K=0 THRESH=0.5 BEST=false
export STATIC_LIB="$CODEC" STATIC_SHUF=0

# shellcheck source=/dev/null
. "$REPO/paper-benchmark/nyx/common.sh"
bench_setup || exit 1
rm -rf "$STORE"; mkdir -p "$STORE"
bench_compose "$STORE"
# common.sh's ${STATIC_LIB:+...} drops the inner quotes; accept either form.
grep -qE "neuropress_static_lib:[[:space:]]*\"?${CODEC}\"?" "$STORE/compose.yaml" || {
  echo "compose does not carry the $CODEC static lib -- refusing to run a" >&2
  echo "silent NeuroPress arm mislabelled as $CODEC" >&2; exit 3; }

export LD_LIBRARY_PATH="$BUILD/bin${EXTRA_LIB:+:$EXTRA_LIB}:${LD_LIBRARY_PATH:-}"
NFILES=$(find "$FIELDS" -name '*.f32' | wc -l)
echo "== $CODEC: $NFILES field file(s), chunk $CHUNK, eb $EB, port $PORT"

# --verify is what RUNS the check; --check-bound only changes what it asserts.
# Lossy codecs can never satisfy a digest, so they are checked element-wise
# against eb; ndzip is lossless and gets the stronger bit-exact digest.
VERIFY=(--verify); [ "$LOSSLESS" = 0 ] && VERIFY+=(--check-bound)

START=$(date +%s.%N)
env CLIO_SERVER_CONF="$STORE/compose.yaml" \
    CLIO_WITH_RUNTIME=1 \
    CLIO_REPLAY_COMPRESSOR_POOL=512.0 \
    CLIO_NEUROPRESS_SELECTION_LOG="$STORE/selection.csv" \
    CTP_LOG_LEVEL="${CTP_LOG_LEVEL:-warning}" \
    CLIO_NEUROPRESS_ERROR_BOUND="$EB" \
    CLIO_NEUROPRESS_COST_BW=5e6 \
    CLIO_NEUROPRESS_EXPLORE_MEASURE_DT=1 \
    CLIO_NEUROPRESS_MEASURE_QUALITY=1 \
    CLIO_NEUROPRESS_STAGE_H2D=1 \
    "$BIN" --dir "$FIELDS" --ext .f32 --chunk "$CHUNK" --max-files "$MAXF" \
           --tag "static_$CODEC" --report "$STORE/blobs.csv" "${VERIFY[@]}" \
    > "$STORE/stdout.log" 2> "$STORE/runtime.log"
RC=$?
WALL=$(awk -v a="$START" -v b="$(date +%s.%N)" 'BEGIN{printf "%.2f", b-a}')
cat > "$STORE/meta.json" <<JSON
{"config":"static-$CODEC","tag":"$CODEC","rc":$RC,
 "mode":$([ "$LOSSLESS" = 1 ] && echo '"lossless"' || echo '"lossy"'),
 "cost_model":"none","error_bound":$EB,"device":"gpu",
 "chunk":$CHUNK,"files":$NFILES,"wall_s":$WALL,"preset":"BALANCED(2)"}
JSON
[ $RC -ne 0 ] && { echo "   FAILED rc=$RC"; grep -vE "DEBUG|INFO" "$STORE/runtime.log" | tail -8; }
grep -E "^stored|VERIFIED|FAILED:|BOUND|  time:" "$STORE/stdout.log" | sed 's/^/   /'
# WireIdForName falls back to zstd for an unknown name, so a build lacking this
# codec would produce a plausible-looking result that is really zstd. The codec
# census is the only thing that rules that out.
echo "   --- codec census (must be $CODEC, or this is not a $CODEC result) ---"
grep -E "^  codec" "$STORE/stdout.log" | sed 's/^/   /'
