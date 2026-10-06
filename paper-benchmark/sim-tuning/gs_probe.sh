#!/usr/bin/env bash
#===============================================================================
# gs_probe.sh -- one Gray-Scott tuning probe: run the GPU Gray-Scott model
# (the NeuroPress / ADIOS2 3D model, ../grayscott/gs_dump.cu) with the given
# settings, stage it, search it exhaustively through Clio and score it
# (probe_eval.py: opportunity per field, NeuroPress v2 learning vs the best
# single codec).
#
#   [L=128] [F=0.04] [K=0.06075] [NOISE=0] [SEED=42] [STEPS=5000] \
#     [DUMP_INT=100] [FIELDS=u,v] gs_probe.sh NAME [NOTE]
#
# (F, K) select the pattern regime: 0.04/0.06075 spots, 0.035/0.065 stripes,
# 0.014/0.045 chaos, 0.04/0.065 sparse spots (grayscott_sim.h). The binary is
# built by ../grayscott/build.sh into ~/np-build/grayscott/gs_dump.
# Data: /mnt/nvme0/tune/gs-tune-NAME, workload ref-gs-tune-NAME.
# Log: /mnt/nvme0/v2-work/runs/gs_tuning.log
#===============================================================================
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
MA=$HERE/../model-accuracy
BIN=${BIN:-$HOME/np-build/grayscott/gs_dump}
LOG=/mnt/nvme0/v2-work/runs/gs_tuning.log
NAME=gs-tune-$1; NOTE=${2:-}
L=${L:-128} F=${F:-0.04} K=${K:-0.06075} NOISE=${NOISE:-0} SEED=${SEED:-42}
STEPS=${STEPS:-5000} DUMP_INT=${DUMP_INT:-100} FIELDS=${FIELDS:-u,v}
D=/mnt/nvme0/tune/$NAME
rm -rf "$D" "/mnt/nvme0/v2-work/ref-$NAME" "/mnt/nvme0/v2-work/baselines/ref-$NAME"
mkdir -p "$D"
ln -sfn "$D" "$HOME/np-data/$NAME"
t0=$(date +%s)
"$BIN" --out "$D/fields" --L "$L" --F "$F" --k "$K" --noise "$NOISE" --seed "$SEED" \
    --steps "$STEPS" --dump-int "$DUMP_INT" --fields "$FIELDS" > "$D/gen.log" 2>&1 ||
  { echo "$NAME: Gray-Scott failed, see $D/gen.log" | tee -a "$LOG"; exit 1; }
echo "$(date +%T) $NAME generated $(find "$D/fields" -name '*.f32' | wc -l) files, $(du -sh "$D/fields" | cut -f1), $(( $(date +%s) - t0 )) s | $NOTE | L=$L F=$F K=$K NOISE=$NOISE SEED=$SEED STEPS=$STEPS DUMP_INT=$DUMP_INT FIELDS=$FIELDS" >> "$LOG"
cd "$MA"
python3 stage_full_workloads.py "ref-$NAME" >> "$LOG" 2>&1
NO_SELECTION_LOG=1 RUN_TAG=nolog MODES=exhaustive ./run_v2_all_workloads.sh "ref-$NAME"
python3 baseline_store.py save "ref-$NAME" exhaustive >> "$LOG" 2>&1
# the stored (compressed) data is not needed after the search: delete it
rm -f "/mnt/nvme0/v2-work/runs/ref-${NAME}_exhaustive_nolog/chi_bdev.dat" "/mnt/nvme0/v2-work/runs/ref-${NAME}_exhaustive_nolog"/cte_tier.dat*
OPENBLAS_NUM_THREADS=4 ~/np-venv/bin/python "$HERE/probe_eval.py" "ref-$NAME" --note "$NOTE" 2>/dev/null | tee -a "$LOG"
