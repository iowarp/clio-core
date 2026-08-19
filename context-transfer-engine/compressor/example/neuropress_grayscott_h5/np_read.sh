#!/usr/bin/env bash
# Phase 2 of 2: a SEPARATE process, with its own runtime, reads what
# np_write.sh stored. It never simulates and never writes -- everything it
# produces has to come out of the persistent tier.
#
# Two checks, and both must hold:
#   1. bit-exact against a reference np_write.sh dumped with plain POSIX I/O,
#      outside HDF5 and so outside Clio;
#   2. the trace shows the codec being inverted. Without check 2 a pass proves
#      nothing: a cache miss falls through to native HDF5 and returns perfectly
#      correct bytes having never touched the compressor.
#
# Usage: ./np_read.sh [-b BUILD] [-L N] [-s STEPS] [-g GAP] [--store DIR]
set -euo pipefail
BUILD=""; L=128; STEPS=50; GAP=25; STORE=/tmp/np_store; CHUNK=1048576; EXPLORE_K=0
while [ $# -gt 0 ]; do
  case "$1" in
    -b|--build) BUILD=$2; shift 2 ;;
    -L)         L=$2; shift 2 ;;
    -s|--steps) STEPS=$2; shift 2 ;;
    -g|--gap)   GAP=$2; shift 2 ;;
    --store)    STORE=$2; shift 2 ;;
    --chunk)    CHUNK=$2; shift 2 ;;
    --explore)  EXPLORE_K=$2; shift 2 ;;
    -h|--help)  sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/np_common.sh"
np_setup || exit 1
[ -e "$STORE/gs.h5" ] || { echo "no store at $STORE -- run np_write.sh first" >&2; exit 1; }

TRACE=1; np_trace_on || TRACE=0
echo "reading from $STORE  (separate process, own runtime)"
[ "$TRACE" = 1 ] || echo "NOTE: path trace is OFF; check 2 cannot run."

"${NP_ENV_RESTART[@]}" "$BIN" "$STORE/gs.h5" "$L" "$STEPS" "$GAP" read \
  > "$STORE/read.log" 2>&1 && rc=0 || rc=$?
grep -E '^reading [0-9]+ dataset|^  verify |^VERIFY' "$STORE/read.log" || true

hit=$(grep -c 'HIT (serve from CTE)' "$STORE/read.log" || true)
mis=$(grep -c 'MISS (native + stage)' "$STORE/read.log" || true)
inv=$(grep -c 'inverting codec' "$STORE/read.log" || true)
echo
echo "=== evidence ==="
echo "  cache      : ${hit:-0} hit / ${mis:-0} miss"
echo "  decompress : ${inv:-0} chunk(s) inverted"
if [ "${inv:-0}" -gt 0 ]; then
  grep -oE 'physical=[0-9]+ logical=[0-9]+' "$STORE/read.log" \
    | awk -F'[= ]' '{p+=$2; l+=$4} END{if(p>0)printf("  stored     : %.2f MiB -> %.1f MiB (%.1fx inverted)\n", p/1048576, l/1048576, l/p)}'
fi

[ "$rc" -eq 0 ] || { echo; echo "READ VERIFY FAILED (rc=$rc)"; exit 4; }
if [ "$TRACE" != 1 ]; then
  echo
  echo "INCONCLUSIVE: bit-exact, but this build cannot report whether a codec"
  echo "was inverted, and native HDF5 returns correct bytes on a cache miss."
  echo "Rebuild with -DCLIO_NEUROPRESS_PATH_TRACE=ON to make this check real."
  exit 3
fi
if [ "${inv:-0}" -eq 0 ]; then
  echo
  echo "INCONCLUSIVE: bit-exact, but no codec was inverted -- these bytes came"
  echo "from the native HDF5 file, not from decompressing the stored data."
  exit 3
fi
echo
echo "PASS: a separate process, own runtime, reconstructed the data by"
echo "      inverting ${inv:-?} codec chunk(s) -- bit-exact against a reference"
echo "      Clio never touched."
