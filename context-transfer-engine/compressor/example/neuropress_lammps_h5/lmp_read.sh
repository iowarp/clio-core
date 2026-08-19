#!/usr/bin/env bash
# Phase 2 of 2: a SEPARATE process with its own runtime reads the data back.
#
# The reader is h5dump -- a stock HDF5 tool with no Clio awareness whatsoever.
# It is handed the same file twice: once with the connector out of the
# environment (so HDF5 reads the native file directly) and once through Clio
# with CLIO_RESTART=1, which replays the metadata log so a fresh process can
# find what the writer stored. The two dumps must be byte-identical.
#
# WHY THE BYTE COMPARISON IS NOT ENOUGH, and the mistake this script exists to
# stop anyone repeating: the native HDF5 file is authoritative and holds the
# same data uncompressed. A read that misses the tier entirely falls through to
# it and returns perfectly correct bytes. So `cmp` passes just as happily when
# Clio is doing nothing at all -- during this example's development a whole
# series of runs "verified" that way while the compressed tier was dead weight.
# What distinguishes the two is the READ PATH: chunks fetched from the tier and
# inverted by a codec, and no native fallback. Both are asserted below.
#
# An earlier version of this script compared against a SECOND LAMMPS run with
# h5diff. That is invalid and was removed: GPU pair-force summation is not
# bit-reproducible, so two runs of the same deck diverge immediately and the
# diff reports millions of differences that say nothing about compression.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

STORE=${STORE:-$HERE/store}
CHUNK=${CHUNK:-4194304}
DSET=${DSET:-/particles/all/position/value}
BOX=${BOX:-80} STEPS=${STEPS:-300} GAP=${GAP:-50}
while [ $# -gt 0 ]; do
  case "$1" in
    --store) STORE=$2; shift 2;;
    --chunk) CHUNK=$2; shift 2;;
    --dataset) DSET=$2; shift 2;;
    -h|--help)
      echo "usage: $0 [--store DIR] [--chunk BYTES] [--dataset PATH]"; exit 0;;
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
done
export STORE CHUNK BOX STEPS GAP
[ -f "$STORE/melt.h5" ] || { echo "no $STORE/melt.h5 -- run lmp_write.sh first" >&2; exit 1; }

source "$HERE/lmp_common.sh"
lmp_setup >/dev/null || exit 1

NAT=$STORE/native.bin
CLIO=$STORE/clio.bin
LOG=$STORE/read.log
rm -f "$NAT" "$CLIO"

echo "reading $DSET"
echo "  [1/2] native HDF5, connector out of the environment"
env -u HDF5_VOL_CONNECTOR -u HDF5_PLUGIN_PATH \
  h5dump -d "$DSET" -b LE -o "$NAT" "$STORE/melt.h5" >/dev/null 2>&1
[ -s "$NAT" ] || { echo "native dump produced nothing" >&2; exit 1; }

echo "  [2/2] through Clio, fresh runtime, CLIO_RESTART=1 (replays the log)"
"${NP_ENV_RESTART[@]}" h5dump -d "$DSET" -b LE -o "$CLIO" "$STORE/melt.h5" \
  > "$LOG" 2>&1

echo
if ! cmp -s "$NAT" "$CLIO"; then
  echo "FAIL: the two dumps differ -- Clio returned different bytes than the file holds"
  exit 1
fi
SZ=$(stat -c %s "$NAT")
echo "  bytes           : $SZ, identical to the native read"

# Did the tier actually serve it? Counted, not assumed.
DEC=$(grep -c 'Decompress' "$LOG" 2>/dev/null || true)
REFILL=$(grep -c 'READ   refill' "$LOG" 2>/dev/null || true)
NATIVE=$(grep -c 'clio_native_read' "$LOG" 2>/dev/null || true)
POP=$(grep -oE 'populated=[01]' "$LOG" 2>/dev/null | sort -u | tr '\n' ' ')

# The trace is compile-time (-DCLIO_NEUROPRESS_PATH_TRACE=ON). Without it there
# is no evidence either way, and reporting PASS on no evidence is exactly the
# false positive described at the top of this file.
if [ "${DEC:-0}" -eq 0 ] && [ "${REFILL:-0}" -eq 0 ] && [ -z "$POP" ]; then
  echo
  echo "INCONCLUSIVE: the bytes match, but this connector was built without the"
  echo "path trace, so nothing here shows whether the tier served the read or it"
  echo "fell through to the native file. Rebuild with"
  echo "    cmake -DCLIO_NEUROPRESS_PATH_TRACE=ON"
  echo "to get a real answer."
  exit 2
fi

echo "  chunks fetched  : $REFILL   from the CTE tier"
echo "  codec inversions: $DEC"
echo "  native fallback : $NATIVE"
[ -n "$POP" ] && echo "  cache probe     : $POP"

grep -oE 'physical=[0-9]+ logical=[0-9]+' "$LOG" 2>/dev/null | \
  awk -F'[= ]' '{p+=$2; l+=$4; n++} END {
    if (n) printf "  stored          : %.1f MiB physical <- %.1f MiB logical = %.4fx\n",
                  p/1048576, l/1048576, l/p }'

echo
if [ "${NATIVE:-0}" -ne 0 ]; then
  echo "FAIL: $NATIVE read(s) fell through to the native file"
  exit 1
fi
if [ "${DEC:-0}" -eq 0 ]; then
  echo "FAIL: nothing was decompressed -- the bytes came from the native file,"
  echo "      so this run proves nothing about the compressed tier"
  exit 1
fi
echo "PASS: a separate process, its own runtime, reconstructed $DSET from"
echo "      Clio's compressed tier -- byte-identical to the native read, with"
echo "      no read falling back to the file."
