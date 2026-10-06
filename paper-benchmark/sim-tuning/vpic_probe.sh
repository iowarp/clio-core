#!/usr/bin/env bash
#===============================================================================
# vpic_probe.sh -- one VPIC tuning probe: run the Weibel deck with the given
# settings, stage it, search it exhaustively through Clio and score it
# (probe_eval.py: opportunity per field, NeuroPress v2 learning vs the best
# single codec).
#
#   [NCELL=126] [NZ=NCELL] [NPPC=8] [STEPS=400] [DUMP_INT=10] [VPIC_* deck settings] \
#     vpic_probe.sh NAME [NOTE]
#
# Deck settings are environment variables the deck reads itself, e.g.
# VPIC_SLABS=CSCSCSCPPCSCSCSC VPIC_VTHE=0.03 VPIC_VTHEX=0.006
# VPIC_DUMP_VARS=ex,ey,ez (see ../vpic/weibel_clio.cxx, LAYERED SLABS).
# Data: /mnt/nvme0/tune/vpic-tune-NAME, workload ref-vpic-tune-NAME.
# Log: /mnt/nvme0/v2-work/runs/vpic_tuning.log
#===============================================================================
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
MA=$HERE/../model-accuracy
LOG=/mnt/nvme0/v2-work/runs/vpic_tuning.log
NAME=vpic-tune-$1; NOTE=${2:-}
NCELL=${NCELL:-126} NPPC=${NPPC:-8} STEPS=${STEPS:-400} DUMP_INT=${DUMP_INT:-10}
D=/mnt/nvme0/tune/$NAME
rm -rf "$D" "/mnt/nvme0/v2-work/ref-$NAME" "/mnt/nvme0/v2-work/baselines/ref-$NAME"
mkdir -p "$D"
ln -sfn "$D" "$HOME/np-data/$NAME"
t0=$(date +%s)
"$HERE/../vpic/gen_fields.sh" --ncell "$NCELL" --nz "${NZ:-$NCELL}" --nppc "$NPPC" --steps "$STEPS" \
    --dump-int "$DUMP_INT" --clean-div 0 --out "$D/fields" > "$D/gen.log" 2>&1 ||
  { echo "$NAME: VPIC failed, see $D/gen.log" | tee -a "$LOG"; exit 1; }
echo "$(date +%T) $NAME generated $(find "$D/fields" -name '*.f32' | wc -l) files, $(du -sh "$D/fields" | cut -f1), $(( $(date +%s) - t0 )) s | $NOTE | $(env | grep '^VPIC_' | sort | tr '\n' ' ')" >> "$LOG"
cd "$MA"
python3 stage_full_workloads.py "ref-$NAME" >> "$LOG" 2>&1
NO_SELECTION_LOG=1 RUN_TAG=nolog MODES=exhaustive ./run_v2_all_workloads.sh "ref-$NAME"
python3 baseline_store.py save "ref-$NAME" exhaustive >> "$LOG" 2>&1
# the stored (compressed) data is not needed after the search: delete it
rm -f "/mnt/nvme0/v2-work/runs/ref-${NAME}_exhaustive_nolog/chi_bdev.dat" "/mnt/nvme0/v2-work/runs/ref-${NAME}_exhaustive_nolog"/cte_tier.dat*
OPENBLAS_NUM_THREADS=4 ~/np-venv/bin/python "$HERE/probe_eval.py" "ref-$NAME" --note "$NOTE" 2>/dev/null | tee -a "$LOG"
