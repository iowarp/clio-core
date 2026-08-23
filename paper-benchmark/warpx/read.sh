#!/usr/bin/env bash
# Phase 2: read the openPMD file back THROUGH the VOL and check the bytes.
#
#   ./run_config.sh static-zstd-s4
#   ./read.sh --run static-zstd-s4
#
# Verification differs from the other workloads. There, a Clio driver staged
# the bytes and could digest them on the way past. Here the application is a
# stock WarpX and the staging happens inside the VOL, so the check is: read
# each field dataset twice -- once through Clio's VOL (which fetches from the
# tier and inverts the codec) and once through the NATIVE HDF5 VOL -- and
# compare.
#
# NOTE ON WHAT THIS PROVES. The native .h5 written alongside is authoritative
# and holds the same data uncompressed, so a VOL read that MISSES the tier
# entirely still returns correct bytes. Matching data is therefore necessary
# but not sufficient; the path trace is what says the tier was actually used.
# This script reports both, and calls the run INCONCLUSIVE when the trace shows
# no tier reads. Same caveat the LAMMPS HDF5 example documents.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(cd "$HERE/../.." && pwd)}
BUILD=${BUILD:-$REPO/build}
RESULTS="$HERE/results"; RUN=""
while [ $# -gt 0 ]; do
  case "$1" in
    --run) RUN=$2; shift 2;;
    --results) RESULTS=$2; shift 2;;
    -h|--help) sed -n '2,20p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[ -n "$RUN" ] || { echo "need --run NAME" >&2; exit 2; }
STORE=$RESULTS/$RUN
H5=$(ls "$STORE"/run/diags/diag1/*.h5 2>/dev/null | tail -1)
[ -n "$H5" ] || { echo "no openPMD .h5 under $STORE/run/diags -- the sweep deletes them to save space; rerun ./run_config.sh $RUN" >&2; exit 1; }
[ -f "$STORE/compose.yaml" ] || { echo "no $STORE/compose.yaml" >&2; exit 1; }
export LD_LIBRARY_PATH="$BUILD/bin:/usr/local/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

echo "== reading $(basename "$H5") back through the VOL"
# Discover a field dataset rather than assuming one: an openPMD file is named
# for its timestep and contains /data/<step>/..., so a hardcoded /data/0/...
# only works for the first dump.
DSET=${DSET:-$(env -u HDF5_VOL_CONNECTOR -u HDF5_PLUGIN_PATH h5ls -r "$H5" 2>/dev/null \
      | awk '/\/fields\/.*Dataset/ {print $1; exit}')}
[ -n "$DSET" ] || { echo "no field dataset found in $H5" >&2; exit 1; }
echo "   dataset: $DSET"
env CLIO_SERVER_CONF="$STORE/compose.yaml" CLIO_WITH_RUNTIME=1 CLIO_RESTART=1 \
    HDF5_VOL_CONNECTOR=clio HDF5_PLUGIN_PATH="$STORE" \
    CLIO_VOL_COMPRESSOR_POOL=512.0 CTP_LOG_LEVEL="${CTP_LOG_LEVEL:-warn}" \
    h5dump -d "$DSET" -b LE -o "$STORE/via_clio.bin" "$H5" > /dev/null 2> "$STORE/read.log"
RC=$?
env -u HDF5_VOL_CONNECTOR -u HDF5_PLUGIN_PATH \
    h5dump -d "$DSET" -b LE -o "$STORE/via_native.bin" "$H5" > /dev/null 2>&1

if [ $RC -ne 0 ] || [ ! -s "$STORE/via_clio.bin" ]; then
  echo "   VOL read FAILED (rc=$RC)"; grep -vE "DEBUG|INFO" "$STORE/read.log" | tail -6; exit 1
fi
if cmp -s "$STORE/via_clio.bin" "$STORE/via_native.bin"; then
  echo "   bytes match the native read ($(stat -c%s "$STORE/via_clio.bin") B)"
else
  echo "   MISMATCH between the VOL read and the native read"; exit 1
fi

TIER=$(grep -c "READ .*inverting codec" "$STORE/read.log" 2>/dev/null || echo 0)
MISS=$(grep -c "READ .*passthrough" "$STORE/read.log" 2>/dev/null || echo 0)
if [ "${TIER:-0}" -gt 0 ]; then
  echo "   PASS: $TIER chunk(s) fetched from the tier and inverted by a codec"
elif grep -q "np-path" "$STORE/read.log" 2>/dev/null; then
  echo "   INCONCLUSIVE: bytes correct, but the trace shows no codec inversion"
  echo "   (passthrough=$MISS) -- the native file may have answered the read"
else
  echo "   INCONCLUSIVE: no path trace in this build (-DCLIO_NEUROPRESS_PATH_TRACE=ON)"
fi
