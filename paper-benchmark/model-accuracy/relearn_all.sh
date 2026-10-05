#!/usr/bin/env bash
#===============================================================================
# relearn_all.sh -- run the NeuroPress learning run again for every finished
# full-size workload (and nyx-multiphase-25g), with the current default of
# run_v2_workloads.sh: no decompress on the write path. The previous learning
# runs move to runs/_learn_with_write_decompress/ (kept, not deleted). Then
# compare_fixed_vs_learn.py and plot_fixed_vs_learn.py for each workload, and
# replot_all.sh at the end. The exhaustive, fixed and oracle runs are reused.
#
#   relearn_all.sh [DATASET ...]
#
# Progress: /mnt/nvme0/v2-work/runs/relearn_all.log, ending RELEARN_ALL_DONE.
#===============================================================================
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
R=/mnt/nvme0/v2-work/runs
LOG=$R/relearn_all.log
cd "$HERE"
export NO_SELECTION_LOG=1 RUN_TAG=nolog
DATASETS=("$@")
if [ ${#DATASETS[@]} -eq 0 ]; then
  mapfile -t DATASETS < <(python3 -c "
import os, eval_v2_workloads as ev
for d in ev.FULL_WORKLOADS + ['nyx-multiphase-25g']:
    if ev.run_finished(os.path.join(ev.RUNS, f'{d}_learn_nolog')):
        print(d)")
fi
mkdir -p "$R/_learn_with_write_decompress"
echo "$(date +%F_%T) start ${#DATASETS[@]} workloads" >> "$LOG"
for ds in "${DATASETS[@]}"; do
  old="$R/${ds}_learn_nolog"
  if [ -d "$old" ] && [ ! -d "$R/_learn_with_write_decompress/${ds}_learn_nolog" ]; then
    mv "$old" "$R/_learn_with_write_decompress/"
    [ -f "$R/${ds}_compare.txt" ] && cp "$R/${ds}_compare.txt" "$R/_learn_with_write_decompress/"
  fi
  MODES=learn ./run_v2_all_workloads.sh "$ds"
  best=$(python3 baseline_store.py best "$ds")
  python3 compare_fixed_vs_learn.py "$ds" --tag nolog > "$R/${ds}_compare.txt" 2>&1
  python3 plot_fixed_vs_learn.py "$ds" --codec "$best" >> "$LOG" 2>&1
  echo "$(date +%T) $ds relearned" >> "$LOG"
done
./replot_all.sh >> "$LOG" 2>&1
echo "$(date +%T) RELEARN_ALL_DONE" >> "$LOG"
