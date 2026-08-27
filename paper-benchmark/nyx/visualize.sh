#!/usr/bin/env bash
# Render every Nyx field at the benchmark's evolving default, into ./viz.
#
#   ./visualize.sh                      # ~48 s -> ./viz
#   ./visualize.sh --steps 400 --int 16 # a quicker look
#   ./visualize.sh --keep-dumps         # also leave the .f32 dumps behind
#   ./visualize.sh --out DIR
#
# ONLY THE FIGURES ARE KEPT. The dumps behind them are ~4 GB at the default
# settings -- 1.3 GB of .f32 and 2.8 GB of AMReX plotfiles -- and they are
# regenerable from this script in under a minute, so they go to a scratch
# directory and are deleted when the render finishes. `--keep-dumps` puts them
# in ./fields-viz instead and leaves them, for when you want to sweep or
# re-slice the same bytes.
#
# NOT `fields/`, which gen_fields.sh defaults to and run_sweep.sh reads: that
# directory is WIPED on entry to gen_fields.sh, so rendering into it would
# silently destroy a sweep's input. A sweep and a montage also want different
# frame counts -- a sweep wants few large frames because its chunk count is the
# point, a montage is unreadable much past ~26.
#
# ZMOM IS SLICED ON x, NOT z, AND THAT IS NOT COSMETIC. A vector component is
# antisymmetric about the mid-plane normal to its own axis, so z-momentum is ~0
# across the whole z mid-plane by symmetry and the montage comes out blank while
# the field is perfectly healthy. Measured at 128^3, frame 20: zmom reaches 4.24
# globally and 0.048 on the z mid-plane, against xmom's 4.23 on that same plane.
# Same trap as the Fortran-ordering one in ../viz_fields.py, same silence.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

OUT=$HERE/viz
NCELL=128 STEPS=1000 INT=40 KEEP=0
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT=$2; shift 2;;
    --ncell) NCELL=$2; shift 2;;
    --steps) STEPS=$2; shift 2;;
    --int|--plot-int) INT=$2; shift 2;;
    --keep-dumps) KEEP=1; shift;;
    # Anything else reaches gen_fields.sh: --cfl, --exp-energy, --bin, ...
    *) EXTRA+=("$1"); shift;;
  esac
done

if [ "$KEEP" = 1 ]; then
  FIELDS=$HERE/fields-viz
else
  SCRATCH=$(mktemp -d)
  FIELDS=$SCRATCH/fields
  # Both the dumps and the -plotfiles sibling gen_fields.sh writes next to them.
  trap 'rm -rf "$SCRATCH"' EXIT
fi

echo "== generating $([ "$KEEP" = 1 ] && echo "$FIELDS" || echo "dumps (scratch, deleted after rendering)")"
"$HERE/gen_fields.sh" --ncell "$NCELL" --steps "$STEPS" --plot-int "$INT" \
    --keep-plt --out "$FIELDS" ${EXTRA[@]+"${EXTRA[@]}"}

echo "== rendering $OUT"
"$HERE/../viz_fields.py" --fields "$FIELDS" --out "$OUT" \
    ${FIELDS:+$([ -d "$FIELDS-plotfiles" ] && echo --plt "$FIELDS-plotfiles")} \
    --field density --field xmom --field ymom --field zmom \
    --field rho_E --field rho_e \
    --axis-for x:zmom

echo
echo "   $(ls "$OUT"/*.png "$OUT"/*.gif 2>/dev/null | wc -l) files in $OUT"
echo "   6 montages, 6 GIFs, and evolution.png (shock radius, domain disturbed,"
echo "   temporal redundancy, and a zlib stand-in for the ratio)"
[ "$KEEP" = 1 ] && echo "   dumps kept in $FIELDS and $FIELDS-plotfiles"
exit 0
