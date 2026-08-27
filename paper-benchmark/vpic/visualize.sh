#!/usr/bin/env bash
# Render every VPIC field variable at the benchmark's evolving default, into
# ./viz.
#
#   ./visualize.sh                       # ~60 s -> ./viz
#   ./visualize.sh --ncell 126           # the benchmark's own resolution (slow)
#   ./visualize.sh --keep-dumps          # also leave the .f32 dumps behind
#   ./visualize.sh --out DIR
#
# ONLY THE FIGURES ARE KEPT. Sixteen variables at 25 frames is gigabytes of
# .f32 that this script regenerates in a minute, so the dumps go to a scratch
# directory and are deleted when the render finishes. `--keep-dumps` puts them
# in ./fields-viz and leaves them.
#
# NOT `fields/`, which gen_fields.sh defaults to and run_sweep.sh reads: that
# directory is WIPED on entry to gen_fields.sh, so rendering into it would
# silently destroy a sweep's input.
#
# --ncell DEFAULTS TO 64 HERE, NOT THE BENCHMARK'S 126. The benchmark needs 126
# because the ghost layer makes the dumped extent (N+2)^3 and 126 gives exactly
# 128^3 voxels = 8 MiB per variable, which is a chunk-count property and means
# nothing to a picture. 126 costs ~130 s of simulation and 3.2 GB of dumps
# against 64's few seconds; pass --ncell 126 when the montage has to be of the
# benchmarked run itself.
#
# CLEAN DIVERGENCE CLEANING IS WHAT MAKES FOUR OF THESE PANELS MOVE. At
# upstream's clean_div_e_interval = 0, div_e_err, div_b_err, rhob and rhof are
# never recomputed, so a quarter of every montage is a still image. gen_fields.sh
# now defaults the interval to 10; see "Default Evolving Benchmark
# Configuration" in README.md. rhob stays constant at any setting -- this deck
# is vacuum and accumulates no bound charge -- so its panel is a still image on
# purpose.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

OUT=$HERE/viz
NCELL=64 NPPC=8 STEPS=1000 INT=40 KEEP=0
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT=$2; shift 2;;
    --ncell) NCELL=$2; shift 2;;
    --nppc) NPPC=$2; shift 2;;
    --steps) STEPS=$2; shift 2;;
    --int|--dump-int) INT=$2; shift 2;;
    --keep-dumps) KEEP=1; shift;;
    # Anything else reaches gen_fields.sh: --clean-div, --bin, ...
    *) EXTRA+=("$1"); shift;;
  esac
done

if [ "$KEEP" = 1 ]; then
  FIELDS=$HERE/fields-viz
else
  SCRATCH=$(mktemp -d); FIELDS=$SCRATCH/fields
  trap 'rm -rf "$SCRATCH"' EXIT
fi

echo "== generating $([ "$KEEP" = 1 ] && echo "$FIELDS" || echo "dumps (scratch, deleted after rendering)")"
"$HERE/gen_fields.sh" --ncell "$NCELL" --nppc "$NPPC" --steps "$STEPS" \
    --dump-int "$INT" --out "$FIELDS" ${EXTRA[@]+"${EXTRA[@]}"}

# The whole field array, in VPIC's own order. viz_fields.py turns its two
# blast-wave panels off automatically here -- shock radius and "% off ambient"
# are defined against a quiet background and a Weibel run has none.
FIELD_ARGS=()
for f in ex ey ez cbx cby cbz tcax tcay tcaz jfx jfy jfz rhof rhob \
         div_e_err div_b_err; do
  FIELD_ARGS+=(--field "$f")
done

echo "== rendering $OUT"
"$HERE/../viz_fields.py" --fields "$FIELDS" --out "$OUT" \
    "${FIELD_ARGS[@]}" --evolve-field cby

echo
echo "   $(ls "$OUT"/*.png "$OUT"/*.gif 2>/dev/null | wc -l) files in $OUT"
echo "   16 montages, 16 GIFs, and evolution.png"
[ "$KEEP" = 1 ] && echo "   dumps kept in $FIELDS"
exit 0
