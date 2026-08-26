#!/usr/bin/env bash
# Phase 2: read the HDF5 datasets back THROUGH the VOL, from a process that
# shares only the store directory with the writer.
#
#   ./run_config.sh static-zstd-s4
#   ./read.sh --run static-zstd-s4
#
# The application is the same binary in "read" mode: it opens the file through
# Clio's VOL and compares each dataset against the reference it wrote beside
# it. CLIO_RESTART=1 replays the metadata log so a fresh runtime finds the
# blobs; the compressed tier is the only copy of the field data.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN="" RESULTS="$HERE/results" L=128 STEPS=200 GAP=25 CHUNK=
while [ $# -gt 0 ]; do
  case "$1" in
    --run) RUN=$2; shift 2;;
    --results) RESULTS=$2; shift 2;;
    --L) L=$2; shift 2;;
    --steps) STEPS=$2; shift 2;;
    --gap) GAP=$2; shift 2;;
    --chunk) CHUNK=$2; shift 2;;
    -h|--help) sed -n '2,12p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[ -n "$RUN" ] || { echo "--run <config> is required" >&2; exit 2; }
. "$HERE/common.sh"; bench_setup || exit 1
STORE="$RESULTS/$RUN"
[ -f "$STORE/compose.yaml" ] || { echo "no store at $STORE -- run ./run_config.sh $RUN first" >&2; exit 1; }
export LD_LIBRARY_PATH="$BUILD/bin:/usr/local/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
# The chunk size has to match the write, or every blob lookup misses and HDF5
# quietly serves the native file. Taken from the run's own meta.json unless
# given.
if [ -z "$CHUNK" ] && [ -f "$STORE/meta.json" ]; then
  CHUNK=$(python3 -c "import json,sys;print(json.load(open('$STORE/meta.json'))['chunk'])" 2>/dev/null || echo 1048576)
fi
CHUNK=${CHUNK:-1048576}
echo "== cold read of $STORE/gs.h5 (the writer process is long gone), chunk $CHUNK"
env CLIO_SERVER_CONF="$STORE/compose.yaml" \
    CLIO_WITH_RUNTIME=1 CLIO_RESTART=1 \
    HDF5_VOL_CONNECTOR=clio HDF5_PLUGIN_PATH="$STORE" \
    CLIO_VOL_COMPRESSOR_POOL=512.0 \
    CLIO_VOL_CHUNK_SIZE="$CHUNK" \
    CLIO_VOL_STAMP_GRANULARITY_NS=0 \
    CTP_LOG_LEVEL="${CTP_LOG_LEVEL:-warning}" \
    "$BIN" "$STORE/gs.h5" "$L" "$STEPS" "$GAP" read > "$STORE/read.log" 2>&1
rc=$?
grep -E '^reading|^  verify |^VERIFY' "$STORE/read.log" || true
INV=$(grep -c 'inverting codec' "$STORE/read.log" || true)
echo
echo "  decompress : ${INV:-0} chunk(s) inverted by a codec"
[ $rc -eq 0 ] || { echo "READ VERIFY FAILED (rc=$rc)"; exit 4; }
if [ "${INV:-0}" -eq 0 ]; then
  echo "INCONCLUSIVE: bit-exact, but no codec was inverted -- these bytes came"
  echo "from the native HDF5 file, not from decompressing the stored data."
  exit 3
fi
echo "PASS: bit-exact AND served by inverting the stored codec." 
