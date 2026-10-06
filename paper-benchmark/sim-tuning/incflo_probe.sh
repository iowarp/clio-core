#!/usr/bin/env bash
#===============================================================================
# incflo_probe.sh -- one incflo tuning probe: run incflo (AMReX-Fluids, CUDA)
# with an inputs file and overrides, convert its plotfiles to flat float32
# files (../incflo/plt_to_raw.py), stage them, search them exhaustively
# through Clio and score them (probe_eval.py: opportunity per field,
# NeuroPress v2 learning vs the best single codec).
#
#   [INPUTS=../incflo/inputs.rt] [NCELL="128 128 256"] [STOP=20] [PER=1.0] \
#   [PLOT="velx vely velz gpx gpy gpz density tracer"] \
#     incflo_probe.sh NAME [NOTE] [incflo key=value ...]
#
# Extra key=value arguments go to incflo after the defaults, so they override
# them. Frames are written as fields/plt<step>/<field>.f32 (time first in the
# staging order); the plotfiles are deleted after conversion.
# INCFLO_BIN default ~/src/incflo/build-clio/incflo.ex (~/builds/build_incflo.sh).
# Data: /mnt/nvme0/tune/incflo-tune-NAME, workload ref-incflo-tune-NAME.
# Log: /mnt/nvme0/v2-work/runs/incflo_tuning.log
#===============================================================================
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
MA=$HERE/../model-accuracy
LOG=/mnt/nvme0/v2-work/runs/incflo_tuning.log
INCFLO_BIN=${INCFLO_BIN:-$HOME/src/incflo/build-clio/incflo.ex}
INPUTS=$(realpath "${INPUTS:-$HERE/../incflo/inputs.rt}")
NAME=incflo-tune-$1; NOTE=${2:-}; shift; [ $# -gt 0 ] && shift
NCELL=${NCELL:-128 128 256} STOP=${STOP:-20} PER=${PER:-1.0}
PLOT=${PLOT:-velx vely velz gpx gpy gpz density tracer}
D=/mnt/nvme0/tune/$NAME
rm -rf "$D" "/mnt/nvme0/v2-work/ref-$NAME" "/mnt/nvme0/v2-work/baselines/ref-$NAME"
mkdir -p "$D/fields" "$D/native"
ln -sfn "$D" "$HOME/np-data/$NAME"
t0=$(date +%s)
( cd "$D/native" && "$INCFLO_BIN" "$INPUTS" amr.n_cell="$NCELL" stop_time="$STOP" \
    amr.plot_per_approx="$PER" amr.plotVariables="$PLOT" "$@" ) \
  > "$D/incflo.log" 2>&1 || { echo "$NAME: incflo failed, see $D/incflo.log" | tee -a "$LOG"; exit 1; }
"${PYTHON:-$HOME/np-venv/bin/python}" "$HERE/../incflo/plt_to_raw.py" "$D/fields" \
    $(ls -d "$D"/native/plt[0-9]* | sort) --delete > "$D/convert.log" 2>&1 \
  || { echo "$NAME: conversion failed, see $D/convert.log" | tee -a "$LOG"; exit 1; }
n=$(find "$D/fields" -name '*.f32' | wc -l)
echo "$(date +%T) $NAME generated $n files, $(du -sh "$D/fields" | cut -f1), $(( $(date +%s) - t0 )) s | $NOTE | INPUTS=$(basename "$INPUTS") NCELL=$NCELL STOP=$STOP PER=$PER PLOT=$PLOT $*" >> "$LOG"
cd "$MA"
python3 stage_full_workloads.py "ref-$NAME" >> "$LOG" 2>&1
NO_SELECTION_LOG=1 RUN_TAG=nolog MODES=exhaustive ./run_v2_all_workloads.sh "ref-$NAME"
python3 baseline_store.py save "ref-$NAME" exhaustive >> "$LOG" 2>&1
# the stored (compressed) data is not needed after the search: delete it
rm -f "/mnt/nvme0/v2-work/runs/ref-${NAME}_exhaustive_nolog/chi_bdev.dat" "/mnt/nvme0/v2-work/runs/ref-${NAME}_exhaustive_nolog"/cte_tier.dat*
OPENBLAS_NUM_THREADS=4 ~/np-venv/bin/python "$HERE/probe_eval.py" "ref-$NAME" --note "$NOTE" 2>/dev/null | tee -a "$LOG"
