#!/usr/bin/env bash
# Phase 1 of 2: run VPIC and dump its field array as flat float32 files.
#
#   ./gen_fields.sh [--ncell N] [--nppc N] [--steps N] [--dump-int N] [--out DIR]
#
# VPIC has no library interface either, and upstream NeuroPress's own VPIC
# benchmark deck is tightly coupled to it -- it includes gpucompress.h and runs
# the whole selection benchmark inside the deck. So, as with Nyx, the
# simulation is patched to dump raw fields and the compression sweep replays
# them. weibel_clio.cxx links nothing from Clio or NeuroPress.
#
# The Weibel instability grows magnetic filaments out of a smooth
# counter-streaming equilibrium, so the run spans a range of structure -- but
# PIC field arrays stay close to noise throughout, which is the point of having
# this workload alongside the other two.
#
# --ncell defaults to 126, NOT a round number: VPIC's field array carries a
# ghost layer, so the dumped extent is (N+2)^3. 126 gives exactly 128^3 =
# 2,097,152 voxels = 8 MiB per variable, which chunks at 4 MiB into exactly two
# whole chunks -- the same shape the Nyx dumps have, so chunk counts compare.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

NCELL=126 NPPC=8 STEPS=200 DUMP_INT=25
OUT=${OUT:-$HERE/fields}
BIN=${BIN:-$HERE/weibel_clio.Linux}
while [ $# -gt 0 ]; do
  case "$1" in
    --ncell) NCELL=$2; shift 2;;
    --nppc) NPPC=$2; shift 2;;
    --steps) STEPS=$2; shift 2;;
    --dump-int) DUMP_INT=$2; shift 2;;
    --out) OUT=$2; shift 2;;
    --bin) BIN=$2; shift 2;;
    -h|--help) sed -n '2,22p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

[ -x "$BIN" ] || { echo "missing deck binary: $BIN -- run ./build_deck.sh first" >&2; exit 1; }

rm -rf "$OUT"; mkdir -p "$OUT"
VOX=$(( (NCELL+2) * (NCELL+2) * (NCELL+2) ))
FRAMES=$(( STEPS / DUMP_INT ))
echo "== VPIC Weibel: ${NCELL}^3 cells (${VOX} voxels with ghosts), nppc=$NPPC, $STEPS steps"
echo "   dumping 16 field vars every $DUMP_INT steps -> ~$FRAMES frames"
echo "   out=$OUT"

# VPIC writes rundata/ and its own diagnostics into the working directory.
WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
cd "$WORK"
VPIC_NX=$NCELL VPIC_NY=$NCELL VPIC_NZ=$NCELL VPIC_NPPC=$NPPC \
VPIC_STEPS=$STEPS VPIC_DUMP_INT=$DUMP_INT \
VPIC_DUMP_FIELDS=1 VPIC_DUMP_DIR="$OUT" \
    "$BIN" > "$OUT/vpic.log" 2>&1
RC=$?
if [ $RC -ne 0 ]; then echo "VPIC failed (rc=$RC); see $OUT/vpic.log" >&2; tail -20 "$OUT/vpic.log" >&2; exit $RC; fi

N=$(find "$OUT" -name '*.f32' | wc -l)
# Manifest, so the replay side can report what simulation produced these dumps
# -- see the same block in ../nyx/gen_fields.sh. Without it a VPIC run reports
# "0 timesteps", because the replay phase only ever sees files.
#
# frames counts the dumps actually written, which is NOT steps/dump_int + 1
# here: this deck skips step 0 deliberately (the field array is identically
# zero before the first solve), so deriving it would overcount by one.
cat > "$OUT/gen.json" <<JSON
{"ncell":$NCELL,"nppc":$NPPC,"steps":$STEPS,"dump_int":$DUMP_INT,
 "frames":$(find "$OUT" -mindepth 1 -maxdepth 1 -type d | wc -l),"files":$N}
JSON
echo "   $N field files, $(du -sh "$OUT" | cut -f1)"
echo "   vars: $(find "$OUT" -name '*.f32' -printf '%f\n' | sed -E 's/fab[0-9]+_comp[0-9]+_//; s/\.f32//' | sort -u | tr '\n' ' ')"
echo
echo "now sweep it:  ./run_sweep.sh --fields $OUT"
