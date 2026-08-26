#!/usr/bin/env bash
# Phase 1 of 2: run Nyx and dump its raw fields as flat float32 files.
#
#   ./gen_fields.sh [--ncell N] [--steps N] [--plot-int N] [--stop-time T]
#                   [--exp-energy E] [--out DIR]
#
# Nyx is an AMReX application with no library interface to embed, so -- exactly
# as upstream NeuroPress's own Nyx benchmark does -- the simulation is patched
# to dump raw fields, and the compression sweep replays those files offline.
# See patches/nyx-raw-field-dump.patch; it adds one self-contained function and
# links nothing.
#
# The Sedov blast wave is the workload upstream recommends: a point explosion
# into uniform background, so the data starts almost perfectly compressible and
# gets progressively less so as the shock expands. That spread is the point --
# it gives the selector genuinely different chunks within one run.
#
# HOW FAR THE SHOCK GETS is what decides whether the data evolves at all, and
# the default (200 steps, reaching t=0.005) leaves most of the box untouched.
# Sedov-Taylor gives R(t) = 1.033 (E t^2/rho)^(1/5) for this deck's rho=1, and
# a 61440 B chunk of a 128^3 field is essentially one z-slab, so a chunk is
# touched only once the shock reaches its z:
#
#     steps   t        E    R      z swept   blocks bit-identical between dumps
#      200    0.005   1.0  0.124     25%       91% -> 62%      (the default)
#     2600    0.113   3.0  0.533    100%       66% ->  0%
#
# Measured on density at 128^3: the DEFAULT deck leaves 62-91% of blocks
# BIT-IDENTICAL between consecutive dumps for the whole run -- E is exactly 0
# for more than half the domain at every dump, and the median block never
# changes at all. Keep that when you want a mix of frozen and active chunks,
# which is what makes the selector see genuinely different work; raise --steps
# (and --exp-energy, since R ~ E^(1/5)) when you want the whole domain live.
#
# stop_time DEFAULTS HIGH SO --steps IS THE CONTROL. Both bound the run; only
# one of them should be doing the work, and step count is what --plot-int is
# expressed in. Pass --stop-time only if you want a time-terminated run.
#
# NOTE: reaching these times at all needs the second patch,
# patches/nyx-comoving-a-single-precision-eps.patch. Without it Nyx aborts with
# "get_comoving_a: invalid time" partway in -- an eps in get_comoving_a() that
# underflows a float ULP in the AMReX_PRECISION=SINGLE build this benchmark
# uses. It aborts AFTER writing each dump, so the symptom is a nonzero exit
# with seemingly complete output. See the patch header for the full analysis.
#
# Phase 2 is run_sweep.sh, which replays whatever this produced.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

NCELL=128 STEPS=200 PLOT_INT=10 EXP_ENERGY=
# Parked above any time these step counts can reach; see the note above.
STOP_TIME=1.0
OUT=${OUT:-$HERE/fields}
NYX_BIN=${NYX_BIN:-$HOME/src/Nyx/build-clio/Exec/HydroTests/nyx_HydroTests}
while [ $# -gt 0 ]; do
  case "$1" in
    --ncell) NCELL=$2; shift 2;;
    --steps) STEPS=$2; shift 2;;
    --plot-int) PLOT_INT=$2; shift 2;;
    --stop-time) STOP_TIME=$2; shift 2;;
    --exp-energy) EXP_ENERGY=$2; shift 2;;
    --out) OUT=$2; shift 2;;
    --bin) NYX_BIN=$2; shift 2;;
    -h|--help) sed -n '2,42p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

if [ ! -x "$NYX_BIN" ]; then
  cat >&2 <<MSG
missing Nyx: $NYX_BIN

Build it (float32, CUDA, single rank -- see README.md):
  git clone --depth 1 --recursive https://github.com/AMReX-Astro/Nyx.git ~/src/Nyx
  cd ~/src/Nyx && git apply $HERE/patches/nyx-raw-field-dump.patch \\
                            $HERE/patches/nyx-comoving-a-single-precision-eps.patch
  cmake -S . -B build-clio -DCMAKE_BUILD_TYPE=Release \\
        -DNyx_MPI=NO -DNyx_OMP=NO -DNyx_HYDRO=YES -DNyx_HEATCOOL=NO \\
        -DNyx_GPU_BACKEND=CUDA -DAMReX_CUDA_ARCH=Ampere \\
        -DAMReX_PRECISION=SINGLE -DAMReX_PARTICLES_PRECISION=SINGLE
  cmake --build build-clio --target nyx_HydroTests -j
MSG
  exit 1
fi

DECK=$(dirname "$NYX_BIN")/inputs.3d.sph.sedov
[ -f "$DECK" ] || { echo "missing deck: $DECK" >&2; exit 1; }

rm -rf "$OUT"; mkdir -p "$OUT"
FRAMES=$(( STEPS / PLOT_INT + 1 ))
echo "== Nyx Sedov: ${NCELL}^3, $STEPS steps, dump every $PLOT_INT -> ~$FRAMES frames"
[ -n "$EXP_ENERGY" ] && echo "   exp_energy=$EXP_ENERGY (deck default is 1.0)"
echo "   out=$OUT"

# Nyx writes its own plotfiles into the working directory; keep them out of the
# source tree and out of the dump directory.
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
cd "$WORK"
NYX_DUMP_FIELDS=1 NYX_DUMP_DIR="$OUT" "$NYX_BIN" "$DECK" \
    amr.n_cell="$NCELL $NCELL $NCELL" \
    amr.max_grid_size="$NCELL" \
    max_step="$STEPS" amr.plot_int="$PLOT_INT" amr.check_int=0 \
    stop_time="$STOP_TIME" \
    ${EXP_ENERGY:+prob.exp_energy="$EXP_ENERGY"} \
    nyx.v=0 amr.v=0 > "$OUT/nyx.log" 2>&1
RC=$?
if [ $RC -ne 0 ]; then echo "Nyx failed (rc=$RC); see $OUT/nyx.log" >&2; tail -20 "$OUT/nyx.log" >&2; exit $RC; fi

N=$(find "$OUT" -name '*.f32' | wc -l)
echo "   $N field files, $(du -sh "$OUT" | cut -f1)"
echo "   fields: $(find "$OUT" -name '*.f32' -printf '%f\n' | sed -E 's/fab[0-9]+_comp[0-9]+_//; s/\.f32//' | sort -u | tr '\n' ' ')"
echo
echo "now sweep it:  ./run_sweep.sh --fields $OUT"
