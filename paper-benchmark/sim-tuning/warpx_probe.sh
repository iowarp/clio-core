#!/usr/bin/env bash
#===============================================================================
# warpx_probe.sh -- one WarpX tuning probe: run the laser-wakefield deck with
# the given settings, convert its openPMD dump to flat float32 files, stage
# it, search it exhaustively through Clio and score it (probe_eval.py:
# opportunity per field, NeuroPress v2 learning vs the best single codec).
#
#   [NCELL="64 64 512"] [STEPS=2000] [INTERVAL=40] \
#   [PLOT="Ex Ey Ez Bx By Bz jx jy jz rho"] [KEEP_NATIVE=0] \
#     warpx_probe.sh NAME [NOTE] [warpx key=value ...]
#
# Defaults reproduce the earlier evolving configuration (dump_warpx.sh):
# laser1.e_max=32.e12, electrons.density=2.e23, moving window on. Extra
# key=value arguments go to the WarpX command line after them, so they
# override them. Frames are written as fields/plt<step>/<field>.f32 so the
# staging keeps the order the simulation writes them in (time first).
# WARPX_BIN and DECK come from ../site.sh.
# Data: /mnt/nvme0/tune/warpx-tune-NAME, workload ref-warpx-tune-NAME.
# Log: /mnt/nvme0/v2-work/runs/warpx_tuning.log
#===============================================================================
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
MA=$HERE/../model-accuracy
source "$HERE/../site.sh"
LOG=/mnt/nvme0/v2-work/runs/warpx_tuning.log
NAME=warpx-tune-$1; NOTE=${2:-}; shift; [ $# -gt 0 ] && shift
NCELL=${NCELL:-64 64 512} STEPS=${STEPS:-2000} INTERVAL=${INTERVAL:-40}
PLOT=${PLOT:-Ex Ey Ez Bx By Bz jx jy jz rho}
D=/mnt/nvme0/tune/$NAME
rm -rf "$D" "/mnt/nvme0/v2-work/ref-$NAME" "/mnt/nvme0/v2-work/baselines/ref-$NAME"
mkdir -p "$D/fields" "$D/native"
ln -sfn "$D" "$HOME/np-data/$NAME"
t0=$(date +%s)
( cd "$D/native" && "$WARPX_BIN" "$DECK" \
    max_step="$STEPS" diag1.intervals="$INTERVAL" amr.n_cell="$NCELL" \
    diag1.openpmd_backend=h5 diag1.fields_to_plot="$PLOT" \
    laser1.e_max=32.e12 electrons.density=2.e23 warpx.do_moving_window=1 "$@" ) \
  > "$D/warpx.log" 2>&1 || { echo "$NAME: WarpX failed, see $D/warpx.log" | tee -a "$LOG"; exit 1; }
# openPMD field names -> datasets (Ex -> E/x, rho -> rho)
n=0
for h in $(find "$D/native" -name '*.h5' | sort); do
  step=$(h5ls "$h/data" 2>/dev/null | awk '{print $1}' | head -1)
  [ -n "$step" ] || continue
  out=$(printf '%s/fields/plt%05d' "$D" "$step"); mkdir -p "$out"
  for f in $PLOT; do
    ds=$f; [ "${#f}" -eq 2 ] && ds="${f:0:1}/${f:1:1}"
    if h5dump -d "/data/$step/fields/$ds" -b LE -o "$out/${ds//\//_}.f32" "$h" > /dev/null 2>&1 \
       && [ -s "$out/${ds//\//_}.f32" ]; then n=$((n + 1)); else rm -f "$out/${ds//\//_}.f32"; fi
  done
done
[ "${KEEP_NATIVE:-0}" = 1 ] || rm -rf "$D/native"
echo "$(date +%T) $NAME generated $n files, $(du -sh "$D/fields" | cut -f1), $(( $(date +%s) - t0 )) s | $NOTE | NCELL=$NCELL STEPS=$STEPS INTERVAL=$INTERVAL PLOT=$PLOT $*" >> "$LOG"
cd "$MA"
python3 stage_full_workloads.py "ref-$NAME" >> "$LOG" 2>&1
NO_SELECTION_LOG=1 RUN_TAG=nolog MODES=exhaustive ./run_v2_all_workloads.sh "ref-$NAME"
python3 baseline_store.py save "ref-$NAME" exhaustive >> "$LOG" 2>&1
# the stored (compressed) data is not needed after the search: delete it
rm -f "/mnt/nvme0/v2-work/runs/ref-${NAME}_exhaustive_nolog/chi_bdev.dat" "/mnt/nvme0/v2-work/runs/ref-${NAME}_exhaustive_nolog"/cte_tier.dat*
OPENBLAS_NUM_THREADS=4 ~/np-venv/bin/python "$HERE/probe_eval.py" "ref-$NAME" --note "$NOTE" 2>/dev/null | tee -a "$LOG"
