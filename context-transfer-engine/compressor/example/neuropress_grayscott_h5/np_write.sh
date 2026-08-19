#!/usr/bin/env bash
# Phase 1 of 2: write Gray-Scott through Clio + NeuroPress into a PERSISTENT
# store, then exit. Run np_read.sh afterwards, as a separate process with its
# own runtime, to read it back.
#
# Usage: ./np_write.sh [-b BUILD] [-L N] [-s STEPS] [-g GAP] [--store DIR]
set -euo pipefail
BUILD=""; L=128; STEPS=50; GAP=25; STORE=/tmp/np_store; CHUNK=1048576; EXPLORE_K=0; BEST=false; LEARN=false; THRESH=0.5
while [ $# -gt 0 ]; do
  case "$1" in
    -b|--build) BUILD=$2; shift 2 ;;
    -L)         L=$2; shift 2 ;;
    -s|--steps) STEPS=$2; shift 2 ;;
    -g|--gap)   GAP=$2; shift 2 ;;
    --store)    STORE=$2; shift 2 ;;
    --chunk)    CHUNK=$2; shift 2 ;;
    --learn)    LEARN=true; shift ;;
    --explore)  EXPLORE_K=$2; shift 2 ;;
    --threshold) THRESH=$2; shift 2 ;;
    --best)     BEST=true; shift ;;
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
# Expected chunks: each snapshot is L^3*4 bytes cut into CHUNK-sized pieces.
BYTES_PER_SNAP=$(( L * L * L * 4 ))
CHUNKS_PER_SNAP=$(( (BYTES_PER_SNAP + CHUNK - 1) / CHUNK ))
EXPECT=$(( CHUNKS_PER_SNAP * NSNAP ))
echo
echo "chunks expected: $EXPECT ($CHUNKS_PER_SNAP per snapshot x $NSNAP)"
np_stored_stats "$STORE" "$EXPECT" || {
  echo "  Nothing was compressed -- the reader cannot prove anything. Check"
  echo "  that CLIO_VOL_COMPRESSOR_POOL matched the compressor pool_id."
  exit 1
}
echo
echo "now run:  ./np_read.sh --store $STORE -L $L -s $STEPS -g $GAP"
