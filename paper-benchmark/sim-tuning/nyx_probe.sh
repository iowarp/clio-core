#!/usr/bin/env bash
#===============================================================================
# nyx_probe.sh -- one Nyx tuning probe: run the Sedov deck with the given
# settings, stage it, search it exhaustively through Clio and score it
# (probe_eval.py: opportunity per field, NeuroPress v2 learning vs the best
# single codec).
#
#   [DERIVED="divu magvort ..."] [NCELL=128] [STEPS=1000] [PLOT_INT=40] \
#     nyx_probe.sh NAME [NOTE] [nyx key=value ...]
#
# DERIVED is written next to the state fields (NYX_DUMP_DERIVED, our Nyx
# patch); nyx key=value settings go to the Nyx command line, e.g.
# prob.noise_dens=0.05 prob.noise_vel=0.01 prob.nblast=8 (our Sedov patch).
# Data: /mnt/nvme0/tune/nyx-tune-NAME, workload ref-nyx-tune-NAME.
# Log: /mnt/nvme0/v2-work/runs/nyx_tuning.log
#===============================================================================
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
MA=$HERE/../model-accuracy
NYX=$HOME/src/Nyx/build-clio/Exec/HydroTests/nyx_HydroTests
DECK=$HOME/src/Nyx/Exec/HydroTests/inputs.3d.sph.sedov
LOG=/mnt/nvme0/v2-work/runs/nyx_tuning.log
NAME=nyx-tune-$1; NOTE=${2:-}; shift; [ $# -gt 0 ] && shift
NCELL=${NCELL:-128} STEPS=${STEPS:-1000} PLOT_INT=${PLOT_INT:-40}
D=/mnt/nvme0/tune/$NAME
rm -rf "$D" "/mnt/nvme0/v2-work/ref-$NAME" "/mnt/nvme0/v2-work/baselines/ref-$NAME"
mkdir -p "$D/fields" "$D/work"
ln -sfn "$D" "$HOME/np-data/$NAME"
t0=$(date +%s)
(cd "$D/work" && NYX_DUMP_FIELDS=1 NYX_DUMP_DIR="$D/fields" NYX_DUMP_DERIVED="${DERIVED:-}" NYX_DUMP_STATE="${STATE:-}" \
  "$NYX" "$DECK" amr.n_cell="$NCELL $NCELL $NCELL" amr.max_grid_size="$NCELL" \
  max_step="$STEPS" amr.plot_int="$PLOT_INT" amr.check_int=0 stop_time=1.0 nyx.cfl=0.8 \
  nyx.v=0 amr.v=0 "$@" > "$D/nyx.log" 2>&1) || { echo "$NAME: Nyx failed" | tee -a "$LOG"; exit 1; }
rm -rf "$D/work"
echo "$(date +%T) $NAME generated $(find "$D/fields" -name '*.f32' | wc -l) files, $(du -sh "$D/fields" | cut -f1), $(( $(date +%s) - t0 )) s | $NOTE | ${DERIVED:-} | $*" >> "$LOG"
cd "$MA"
python3 stage_full_workloads.py "ref-$NAME" >> "$LOG" 2>&1
NO_SELECTION_LOG=1 RUN_TAG=nolog MODES=exhaustive ./run_v2_all_workloads.sh "ref-$NAME"
python3 baseline_store.py save "ref-$NAME" exhaustive >> "$LOG" 2>&1
OPENBLAS_NUM_THREADS=4 ~/np-venv/bin/python "$HERE/probe_eval.py" "ref-$NAME" --note "$NOTE" 2>/dev/null | tee -a "$LOG"
