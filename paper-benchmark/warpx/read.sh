#!/usr/bin/env bash
# Phase 2: read the openPMD output back THROUGH the VOL and prove the tier
# answered.
#
#   ./run_config.sh static-zstd-s4
#   ./read.sh --run static-zstd-s4
#
# Verification differs from the other workloads. There a Clio driver staged the
# bytes and could digest them on the way past. Here the application is a stock
# WarpX and the staging happens inside the VOL, so the check is: read each field
# dataset twice -- once through Clio's VOL, once through the NATIVE HDF5 VOL --
# and compare, while reading the path trace to confirm the tier was used.
#
# WHY THIS RUNS FROM INSIDE THE RUN DIRECTORY. The VOL keys its tag on the file
# name AS GIVEN, so a reader opening an absolute path does not find what a
# writer stored under a relative one. WarpX writes "diags/diag1/openpmd_*.h5"
# relative to its working directory, so this script cds there and opens the same
# relative path. Opening the identical file by absolute path instead produced
# `populated=0`, `chunk_0_size=0` and `MISS (native + stage)` on every read --
# and still returned correct bytes, because the native file answered. That is
# the failure this script exists to distinguish.
#
# WHAT MATCHING BYTES DO AND DO NOT PROVE. The native .h5 is authoritative and
# holds the same data uncompressed, so a read that misses the tier entirely
# still returns correct bytes. Matching data is necessary, not sufficient. The
# trace is what says a codec ran, so this reports both and fails the run if the
# tier was not used.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(cd "$HERE/../.." && pwd)}
BUILD=${BUILD:-$REPO/build}
RESULTS="$HERE/results"; RUN=""; NDSET=${NDSET:-3}
while [ $# -gt 0 ]; do
  case "$1" in
    --run) RUN=$2; shift 2;;
    --results) RESULTS=$2; shift 2;;
    --datasets) NDSET=$2; shift 2;;
    -h|--help) sed -n '2,26p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[ -n "$RUN" ] || { echo "need --run NAME" >&2; exit 2; }
STORE=$RESULTS/$RUN
RUNDIR=$STORE/run
[ -d "$RUNDIR/diags" ] || { echo "no $RUNDIR/diags -- run_sweep.sh deletes them to save space; rerun ./run_config.sh $RUN" >&2; exit 1; }
[ -f "$STORE/compose.yaml" ] || { echo "no $STORE/compose.yaml" >&2; exit 1; }

# Relative, because that is how WarpX wrote it (see the note above).
REL=$(cd "$RUNDIR" && ls diags/diag1/*.h5 2>/dev/null | tail -1)
[ -n "$REL" ] || { echo "no openPMD .h5 under $RUNDIR/diags" >&2; exit 1; }
CHUNK=$(python3 -c "import json;print(json.load(open('$STORE/meta.json'))['chunk'])" 2>/dev/null || echo 1048576)

DSETS=$(cd "$RUNDIR" && env -u HDF5_VOL_CONNECTOR -u HDF5_PLUGIN_PATH h5ls -r "$REL" 2>/dev/null \
        | awk '/\/fields\/.*Dataset/ {print $1}' | head -"$NDSET")
[ -n "$DSETS" ] || { echo "no field datasets in $REL" >&2; exit 1; }

export LD_LIBRARY_PATH="$BUILD/bin:/usr/local/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
echo "== cold read of $REL (writer process is long gone)"

FAIL=0; TOTAL_INV=0
: > "$STORE/read.log"
for D in $DSETS; do
  ( cd "$RUNDIR" && env CLIO_SERVER_CONF="$STORE/compose.yaml" \
      CLIO_WITH_RUNTIME=1 CLIO_RESTART=1 \
      HDF5_VOL_CONNECTOR=clio HDF5_PLUGIN_PATH="$STORE" \
      CLIO_VOL_COMPRESSOR_POOL=512.0 CLIO_VOL_CHUNK_SIZE="$CHUNK" \
      CLIO_NEUROPRESS_PATH_TRACE=1 \
      CTP_LOG_LEVEL="${CTP_LOG_LEVEL:-warning}" \
      h5dump -d "$D" -b LE -o "$STORE/via_clio.bin" "$REL" ) > /dev/null 2>> "$STORE/read.log"
  RC=$?
  ( cd "$RUNDIR" && env -u HDF5_VOL_CONNECTOR -u HDF5_PLUGIN_PATH \
      h5dump -d "$D" -b LE -o "$STORE/via_native.bin" "$REL" ) > /dev/null 2>&1

  # "inverting codec" is a [np-path] trace line, so the trace above is not
  # optional here: without CLIO_NEUROPRESS_PATH_TRACE the count is always 0 and
  # every dataset reports "the tier was not used" while its bytes are perfect.
  INV=$(grep -c "inverting codec" "$STORE/read.log" 2>/dev/null | head -1)
  INV=$(( INV - TOTAL_INV )); TOTAL_INV=$(( TOTAL_INV + INV ))
  MISS=$(grep -c "MISS (native + stage)" "$STORE/read.log" 2>/dev/null | head -1)

  if [ $RC -ne 0 ] || [ ! -s "$STORE/via_clio.bin" ]; then
    echo "   $D: VOL READ FAILED (rc=$RC)"; FAIL=$((FAIL+1)); continue
  fi
  if ! cmp -s "$STORE/via_clio.bin" "$STORE/via_native.bin"; then
    echo "   $D: MISMATCH against the native read"; FAIL=$((FAIL+1)); continue
  fi
  if [ "${INV:-0}" -gt 0 ]; then
    echo "   $D: $(stat -c%s "$STORE/via_clio.bin") B, identical to native, $INV chunk(s) inverted by a codec"
  else
    echo "   $D: bytes correct BUT the tier was not used (no codec inversion)"
    FAIL=$((FAIL+1))
  fi
done
rm -f "$STORE/via_clio.bin" "$STORE/via_native.bin"

MISS=$(grep -c "MISS (native + stage)" "$STORE/read.log" 2>/dev/null | head -1)
echo
if [ "$FAIL" -eq 0 ]; then
  echo "PASS: every dataset came back byte-identical AND was served from the compressed tier"
  echo "      ($TOTAL_INV chunk inversions, $MISS cache miss(es))"
else
  echo "FAIL: $FAIL dataset(s) did not verify -- see $STORE/read.log"
fi
exit $(( FAIL > 0 ))
