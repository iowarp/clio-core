#!/usr/bin/env bash
# Render the WarpX fields at the benchmark's evolving default, into ./viz.
#
#   ./visualize.sh                       # ~90 s -> ./viz
#   ./visualize.sh --steps 200 --int 20  # a quicker look
#   ./visualize.sh --keep-dumps          # also leave the openPMD output behind
#
# ONLY THE FIGURES ARE KEPT. A 1,000-step run at the default diagnostic
# interval writes ~2 GB of openPMD plus the compressed tier, all of it
# regenerable by this script, so it goes to a scratch directory and is deleted
# when the render finishes. `--keep-dumps` puts it in ./run-viz and leaves it.
#
# NOTHING NEEDS DUMPING HERE, unlike the other three workloads: WarpX writes
# openPMD-HDF5 exactly as it always does and the VOL compresses on the way past,
# so the native .h5 the viewer reads are the same bytes the compressor saw.
#
# --stage-h2d IS NOT OPTIONAL. A stock WarpX hands HDF5 host memory and
# CLIO_NEUROPRESS_REQUIRE_DEVICE defaults on, so without it every chunk is
# refused and run_config.sh fails the run. See BENCHMARK.md section 3b.
#
# THE GRID CANNOT BE SHRUNK TO MAKE THIS QUICK. At 64x64x128 a field is 512 KB
# against the 1 MiB chunk, so no chunk completes and ZERO field bytes reach the
# tier -- while the run exits 0 and the native .h5 is perfect. 64x64x512 is the
# smallest grid that stages anything, so --steps is the only knob that makes a
# run shorter.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

OUT=$HERE/viz
NCELL="64 64 512" STEPS=1000 INT=40 KEEP=0
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT=$2; shift 2;;
    --ncell) NCELL=$2; shift 2;;
    --steps) STEPS=$2; shift 2;;
    --int|--interval) INT=$2; shift 2;;
    --keep-dumps) KEEP=1; shift;;
    # Anything else reaches run_config.sh: --e-max, --density, --eb, ...
    *) EXTRA+=("$1"); shift;;
  esac
done

if [ "$KEEP" = 1 ]; then
  STORE=$HERE/run-viz
else
  SCRATCH=$(mktemp -d); STORE=$SCRATCH/run
  trap 'rm -rf "$SCRATCH"' EXIT
fi

echo "== running WarpX ${NCELL// /x}, $STEPS steps, diag every $INT"
"$HERE/run_config.sh" dynamic --ncell "$NCELL" --steps "$STEPS" \
    --interval "$INT" --stage-h2d --results "$STORE" --tag viz \
    ${EXTRA[@]+"${EXTRA[@]}"} 2>&1 | sed 's/^/   /'

echo "== rendering $OUT"
"$HERE/viz_openpmd.py" --run "$STORE/viz" --out "$OUT"

echo
echo "   $(ls "$OUT"/*.png "$OUT"/*.gif 2>/dev/null | wc -l) files in $OUT"
echo "   one montage per field, plus evolution.png (viz_openpmd.py"
echo "   writes no GIFs -- the other three viewers do)"
[ "$KEEP" = 1 ] && echo "   openPMD kept in $STORE/viz/run/diags/diag1"
exit 0
