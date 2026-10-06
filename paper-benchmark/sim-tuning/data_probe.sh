#!/usr/bin/env bash
#===============================================================================
# data_probe.sh -- one tuning probe of a dataset already on disk (no
# simulation run): stage the files of SRC_DIR in name order, search them
# exhaustively through Clio and score them (probe_eval.py: opportunity under
# the tuning model and the benchmark model, per field, NeuroPress v2 learning
# vs the best single codec).
#
#   data_probe.sh NAME SRC_DIR [NOTE]
#
# Workload ref-data-tune-NAME; staged chunks /mnt/nvme0/v2-work/ref-data-tune-NAME.
# File extensions give the element type (.f32, .f64, ...; anything else is
# read as f32, see stage_v2_workloads.dtype_of).
# Log: /mnt/nvme0/v2-work/runs/data_tuning.log
#===============================================================================
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
MA=$HERE/../model-accuracy
LOG=/mnt/nvme0/v2-work/runs/data_tuning.log
NAME=ref-data-tune-$1; SRC=$2; NOTE=${3:-}
rm -rf "/mnt/nvme0/v2-work/$NAME" "/mnt/nvme0/v2-work/baselines/$NAME"
echo "$(date +%T) $NAME from $SRC: $(find -L "$SRC" -type f | wc -l) files, $(du -shL "$SRC" | cut -f1) | $NOTE" >> "$LOG"
cd "$MA"
python3 stage_full_workloads.py "$NAME" --src "$SRC" >> "$LOG" 2>&1
NO_SELECTION_LOG=1 RUN_TAG=nolog MODES=exhaustive ./run_v2_all_workloads.sh "$NAME"
python3 baseline_store.py save "$NAME" exhaustive >> "$LOG" 2>&1
# the stored (compressed) data is not needed after the search: delete it
rm -f "/mnt/nvme0/v2-work/runs/${NAME}_exhaustive_nolog/chi_bdev.dat" "/mnt/nvme0/v2-work/runs/${NAME}_exhaustive_nolog"/cte_tier.dat*
OPENBLAS_NUM_THREADS=4 ~/np-venv/bin/python "$HERE/probe_eval.py" "$NAME" --note "$NOTE" 2>/dev/null | tee -a "$LOG"
