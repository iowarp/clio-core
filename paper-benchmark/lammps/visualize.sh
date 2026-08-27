#!/usr/bin/env bash
# Render the LAMMPS atom state at the benchmark's evolving default, into ./viz.
#
#   ./visualize.sh                       # ~40 s -> ./viz
#   ./visualize.sh --box 40 --steps 2000 # bigger, slower
#   ./visualize.sh --ramp                # the temperature-ramp deck instead
#   ./visualize.sh --keep-dumps          # also leave the staged bytes behind
#
# ONLY THE FIGURES ARE KEPT. The staged bytes go to a scratch directory and are
# deleted when the render finishes; `--keep-dumps` puts them in ./raw-viz.
#
# THIS WORKLOAD HAS NOTHING ON DISK TO LOOK AT UNLESS THE RUN ASKS FOR IT.
# LAMMPS runs as a library inside the benchmark process and no file is written,
# so `--raw` is the only way to see it: the driver writes each staged blob's
# bytes, which are exactly what NeuroPress compressed.
#
# --chunk IS COMPUTED, NOT CONFIGURABLE. viz_atoms.py assumes ONE CHUNK PER
# FIELD PER FRAME, so the chunk has to be natoms*3*8 exactly or the .bin files
# are fragments that will not reshape. natoms = 4*box^3, so the arithmetic is
# done here rather than left to whoever calls it.
#
# The defaults are run_config.sh's, which the 1,000-timestep evolution study
# chose: --temp 6.0 with --skin 0.8 --every 5. The neighbour settings are a
# correctness condition at that temperature, not tuning -- on upstream's 0.3/20
# an atom crosses the skin between rebuilds and NVE leaks 3.5% of its total
# energy. See "Default Evolving Benchmark Configuration" in README.md.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

OUT=$HERE/viz
BOX=20 STEPS=1000 GAP=40 KEEP=0 RAMP=0
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT=$2; shift 2;;
    --box) BOX=$2; shift 2;;
    --steps) STEPS=$2; shift 2;;
    --gap) GAP=$2; shift 2;;
    --keep-dumps) KEEP=1; shift;;
    # in.melt_ramp: every frame a different state point rather than a
    # stationary hot liquid. Higher mean evolution, lower sustained score --
    # the README says which question each answers.
    --ramp) RAMP=1; shift;;
    # Anything else reaches run_config.sh: --temp, --density, --dt, --var, ...
    *) EXTRA+=("$1"); shift;;
  esac
done

NATOMS=$(( 4 * BOX * BOX * BOX ))
CHUNK=$(( NATOMS * 3 * 8 ))

if [ "$KEEP" = 1 ]; then
  RAW=$HERE/raw-viz; STORE=$HERE/raw-viz-run
else
  SCRATCH=$(mktemp -d); RAW=$SCRATCH/raw; STORE=$SCRATCH/run
  trap 'rm -rf "$SCRATCH"' EXIT
fi
rm -rf "$RAW"; mkdir -p "$RAW"

DECK=()
[ "$RAMP" = 1 ] && DECK=(--deck "$HERE/in.melt_ramp" --var "NSTEPS=$STEPS" \
                         --var T0=0.05 --var T1=12.0)

echo "== running $NATOMS atoms, $STEPS steps, a frame every $GAP, chunk $CHUNK"
"$HERE/run_config.sh" static-zstd --box "$BOX" --steps "$STEPS" --gap "$GAP" \
    --chunk "$CHUNK" --require-device --no-verify \
    --raw "$RAW" --results "$STORE" --tag viz \
    "${DECK[@]}" ${EXTRA[@]+"${EXTRA[@]}"} > "$RAW/../run.log" 2>&1 \
  || { echo "   FAILED"; tail -12 "$RAW/../run.log"; exit 1; }
grep -E "stored [0-9]+ blob" "$RAW/../run.log" | sed 's/^/   /' || true

echo "== rendering $OUT"
"$HERE/viz_atoms.py" --raw "$RAW" --out "$OUT"

echo
echo "   $(ls "$OUT"/*.png "$OUT"/*.gif 2>/dev/null | wc -l) files in $OUT"
echo "   atoms_montage.png, atoms.gif, and evolution.png (MSD, g(r),"
echo "   temperature, temporal redundancy, and a zlib stand-in for the ratio)"
[ "$KEEP" = 1 ] && echo "   staged bytes kept in $RAW"
exit 0
