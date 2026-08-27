#!/usr/bin/env bash
# Original vs decompressed for WarpX, WITHIN a single run.
#
# read.sh proves the tier answered and the bytes are right. This one WRITES THE
# BYTES OUT so something else can look at them -- the equivalent of
# --dump-decompressed on the other three workloads, which have a Clio driver to
# put the flag on and this one does not.
#
# Each run writes its own native .h5 -- uncompressed and authoritative -- so the
# comparison is that file read twice: once through the native HDF5 VOL (the
# originals) and once through Clio's VOL (the decompressed copies). Comparing
# across runs would measure WarpX's own 0.6% run-to-run spread instead.
#
# THE NATIVE FILE ANSWERS EVEN WHEN THE TIER IS MISSED, so a VOL read that never
# reached Clio returns correct bytes and looks like perfect lossless
# compression. Every read records its "inverting codec" count for that reason;
# a zero means the pair must be discarded, not published.
#   ./read_fields.sh [--results DIR] [--runs "eb001 eb01 eb10"] [--out DIR]
#                    [--datasets "B/y E/z"]
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(cd "$HERE/../.." && pwd)}
BUILD=${BUILD:-$REPO/build}
RESULTS=${RESULTS:-/tmp/wx2}; RUNS=${RUNS:-"eb001 eb01 eb10"}
OUTDIR=${OUTDIR:-/tmp/wx-cmp}; DSETS=${DSETS:-"B/y E/z"}
while [ $# -gt 0 ]; do
  case "$1" in
    --results) RESULTS=$2; shift 2;;
    --runs) RUNS=$2; shift 2;;
    --out) OUTDIR=$2; shift 2;;
    --datasets) DSETS=$2; shift 2;;
    -h|--help) sed -n '2,20p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
export LD_LIBRARY_PATH="$BUILD/bin:/usr/local/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
OUT=$OUTDIR; mkdir -p "$OUT"
echo "store,step,dset,inversions" > "$OUT/inversions.csv"
for TAG in $RUNS; do
  STORE=$RESULTS/$TAG; RUNDIR=$STORE/run
  [ -d "$RUNDIR/diags" ] || { echo "skip $TAG"; continue; }
  for H in $(cd "$RUNDIR" && ls diags/diag1/*.h5); do
    STEP=$(echo "$H" | sed -E 's/.*openpmd_0*([0-9]+)\.h5/\1/'); [ -z "$STEP" ] && STEP=0
    for D in $DSETS; do
      N=$(echo "$D" | tr -d '/')
      ( cd "$RUNDIR" && env -u HDF5_VOL_CONNECTOR -u HDF5_PLUGIN_PATH \
          h5dump -d "/data/$STEP/fields/$D" -b LE -o "$OUT/nat_${TAG}_${STEP}_${N}.bin" "$H" ) >/dev/null 2>&1
      LOG=$OUT/t.log
      ( cd "$RUNDIR" && env CLIO_SERVER_CONF="$STORE/compose.yaml" \
          CLIO_WITH_RUNTIME=1 CLIO_RESTART=1 HDF5_VOL_CONNECTOR=clio \
          HDF5_PLUGIN_PATH="$STORE" CLIO_VOL_COMPRESSOR_POOL=512.0 \
          CLIO_VOL_CHUNK_SIZE=1048576 CLIO_NEUROPRESS_PATH_TRACE=1 \
          CTP_LOG_LEVEL=warning \
          h5dump -d "/data/$STEP/fields/$D" -b LE -o "$OUT/dec_${TAG}_${STEP}_${N}.bin" "$H" \
        ) >/dev/null 2>"$LOG"
      echo "$TAG,$STEP,$D,$(grep -c 'inverting codec' "$LOG" 2>/dev/null || echo 0)" >> "$OUT/inversions.csv"
    done
  done
  echo "done $TAG"
done
rm -f "$OUT/t.log"; echo ALLDONE
