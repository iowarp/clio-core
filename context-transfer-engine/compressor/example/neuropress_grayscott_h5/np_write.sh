#!/usr/bin/env bash
# Phase 1 of 2: write Gray-Scott through Clio + NeuroPress into a PERSISTENT
# store, then exit. Run np_read.sh afterwards, as a separate process with its
# own runtime, to read it back.
#
# Usage: ./np_write.sh [-b BUILD] [-L N] [-s STEPS] [-g GAP] [--store DIR]
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
    -h|--help)  sed -n '2,7p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/np_common.sh"
np_setup || exit 1

# A fresh store every write: a stale tier from an earlier run would let the
# reader "pass" on data this run never produced.
rm -f "$STORE"/gs.h5 "$STORE"/gs.h5.ref "$STORE"/cte_tier.dat \
      "$STORE"/chi_bdev.dat "$STORE"/cte_metadata_log* "$STORE"/write.log
echo "Gray-Scott ${L}^3, $STEPS steps, snapshot every $GAP"
echo "store: $STORE  (persistent file tier + metadata log)"

"${NP_ENV[@]}" "$BIN" "$STORE/gs.h5" "$L" "$STEPS" "$GAP" write \
  > "$STORE/write.log" 2>&1 \
  || { echo "write FAILED -- tail:"; tail -25 "$STORE/write.log"; exit 1; }

grep -E 'Gray-Scott|step |done:' "$STORE/write.log" || true
n=$(grep -c 'Compression: ' "$STORE/write.log" || true)
echo
echo "chunks compressed: $n"
if [ "${n:-0}" -gt 0 ]; then
  grep -oE 'Compression: [0-9]+ bytes -> [0-9]+ bytes' "$STORE/write.log" \
    | awk '{i+=$2; o+=$5} END{printf("  stored: %.2f MiB from %.1f MiB (%.1fx)\n", o/1048576, i/1048576, i/o)}'
else
  echo "  NO COMPRESSION -- the reader cannot prove anything. Check that"
  echo "  CLIO_VOL_COMPRESSOR_POOL matched the compressor pool_id."
  exit 1
fi
echo
echo "now run:  ./np_read.sh --store $STORE -L $L -s $STEPS -g $GAP"
