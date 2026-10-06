#!/usr/bin/env bash
#===============================================================================
# gen_nyx_multiphase_50g.sh -- make the nyx-multiphase-50g workload: the 50 GiB
# Nyx Sedov run in a layered multiphase medium (nyx_probe.sh final50g,
# 2026-10-05), with the exact settings, and optionally cut it into the 4 MiB
# chunks the benchmark replays.
#
#   [NYX=binary] [DECK=inputs] gen_nyx_multiphase_50g.sh FIELDS_DIR [STAGE_ROOT]
#
# FIELDS_DIR  where Nyx writes its dump (plt00000 ... plt00266, three
#             fields each: density, rho_E, logden; ~51 GB). Any file system.
# STAGE_ROOT  optional: also write STAGE_ROOT/nyx-multiphase-50g/fields/*.chunk
#             (stage_full_workloads.py; 12816 chunks of 4 MiB, ~51 GB more).
#
# Needs Nyx 4ecfea2a (AMReX 6e875b7cc) with the four patches of ../nyx/patches
# applied and built as ../nyx/README.md "Building Nyx" says (CUDA, single
# precision), and one GPU. ~520 s on an A100. Expected result (Chameleon,
# 2026-10-05):
#   801 files of 67108864 B (256^3 float32), 267 frames x 3 fields
#   sha256 plt00000/fab0000_comp02_logden.f32
#     70d929e60053584f7bcccec4da3fc6e2bc957ec5d7440d809b2ea3341da67eaa
#   sha256 plt00266/fab0000_comp02_logden.f32
#     b3fd13b2ffdba7243f3444fa13f892af4de0d68c5b36e54ca69ba4fc01da96f6
# Nyx also writes its normal AMReX plotfiles every 10 steps (the raw dump is
# an extra output); they go to a work directory next to FIELDS_DIR, which needs
# that extra space during the run and is deleted at the end.
#===============================================================================
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
FIELDS=$(realpath -m "${1:?usage: gen_nyx_multiphase_50g.sh FIELDS_DIR [STAGE_ROOT]}")
STAGE=${2:-}
NYX=${NYX:-$HOME/src/Nyx/build-clio/Exec/HydroTests/nyx_HydroTests}
DECK=${DECK:-$HOME/src/Nyx/Exec/HydroTests/inputs.3d.sph.sedov}

# 256^3, 2660 steps, a dump every 10 steps (267 frames); state fields density
# and rho_E plus the derived logden (nyx-dump-select-derived.patch); the
# medium (nyx-sedov-multiphase.patch): 4 z-layers alternating clumpy (8
# power-of-two densities, pressure equilibrium) and smooth exp(wave k=1), one
# blast of energy 0.05. See ../sim-tuning/README.md "Nyx".
ARGS=(amr.n_cell="256 256 256" amr.max_grid_size=256 max_step=2660 amr.plot_int=10
      amr.check_int=0 stop_time=1.0 nyx.cfl=0.8 nyx.v=0 amr.v=0
      prob.nphase=8 prob.phase_pow2=1 prob.phase_p=1 prob.nlayer=4
      prob.smooth_exp=1 prob.smooth_amp=1.0 prob.smooth_k=1
      prob.nblast=1 prob.exp_energy=0.05 prob.p_ambient=7.62939453125e-06)

mkdir -p "$FIELDS"
WORK=$(mktemp -d "${FIELDS%/}.work.XXXX")
(cd "$WORK" && NYX_DUMP_FIELDS=1 NYX_DUMP_DIR="$FIELDS" NYX_DUMP_STATE="density rho_E" \
   NYX_DUMP_DERIVED="logden" "$NYX" "$DECK" "${ARGS[@]}" > "$FIELDS/nyx.log" 2>&1)
rm -rf "$WORK"
printf '%s\n' "NYX_DUMP_STATE=density rho_E" "NYX_DUMP_DERIVED=logden" "${ARGS[@]}" \
  > "$FIELDS/nyx_args.txt"

n=$(find "$FIELDS" -name '*.f32' | wc -l)
echo "Nyx wrote $n .f32 files (expected 801)"
sha256sum "$FIELDS/plt00000/fab0000_comp02_logden.f32" "$FIELDS/plt00266/fab0000_comp02_logden.f32" || true

if [ -n "$STAGE" ]; then
  python3 "$HERE/../model-accuracy/stage_full_workloads.py" nyx-multiphase-50g \
    --src "$FIELDS" --out "$STAGE"
fi
