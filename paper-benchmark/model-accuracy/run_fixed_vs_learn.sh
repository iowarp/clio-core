#!/usr/bin/env bash
#===============================================================================
# run_fixed_vs_learn.sh -- best single codec vs NeuroPress v2 learning, per
# workload, measured through Clio.
#
#   run_fixed_vs_learn.sh DATASET [DATASET ...]
#
# For each staged DATASET (/mnt/nvme0/v2-work/<DATASET>/fields):
#   1. exhaustive  every chunk with all 45 settings (the per-chunk truth);
#                  steps 1 and 3 are reused from /mnt/nvme0/v2-work/baselines
#                  when stored for this exact input (baseline_store.py), and
#                  stored after running otherwise; FORCE_BASELINES=1 reruns
#   2. the best single codec = lowest total balanced 4-tier cost over the
#      whole workload (compare_fixed_vs_learn.candidates)
#   3. fixed       the whole workload with only that codec, no NeuroPress
#   4. learn       NeuroPress v2, learning on, exploration off
#   5. compare_fixed_vs_learn.py + plot_fixed_vs_learn.py
# All runs: selection log off (NO_SELECTION_LOG=1), RUN_TAG=nolog. The GPU
# clock should be locked. Progress: /mnt/nvme0/v2-work/runs/fixed_vs_learn.log
#===============================================================================
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
LOG=/mnt/nvme0/v2-work/runs/fixed_vs_learn.log
export NO_SELECTION_LOG=1 RUN_TAG=nolog
cd "$HERE"
for ds in "$@"; do
  # Exhaustive search and fixed run: reused from the baseline store when one
  # matches this input (baseline_store.py), run and stored otherwise;
  # FORCE_BASELINES=1 always reruns them.
  if [ -z "${FORCE_BASELINES:-}" ] && python3 baseline_store.py has "$ds" exhaustive; then
    python3 baseline_store.py restore "$ds" exhaustive >> "$LOG"
    echo "$(date +%T) $ds exhaustive reused from the store" >> "$LOG"
  else
    echo "$(date +%T) $ds exhaustive" >> "$LOG"
    MODES=exhaustive ./run_v2_all_workloads.sh "$ds"
    python3 baseline_store.py save "$ds" exhaustive >> "$LOG"
  fi
  best=$(python3 baseline_store.py best "$ds")
  echo "$(date +%T) $ds best single codec: $best" >> "$LOG"
  if [ "$best" = "store (raw)" ]; then
    echo "$(date +%T) $ds: storing raw is cheapest; fixed run skipped" >> "$LOG"
  elif [ -z "${FORCE_BASELINES:-}" ] && python3 baseline_store.py has "$ds" fixed --codec "$best"; then
    python3 baseline_store.py restore "$ds" fixed --codec "$best" >> "$LOG"
    echo "$(date +%T) $ds fixed ($best) reused from the store" >> "$LOG"
  else
    FIXED_SETTING="$best" MODES=fixed ./run_v2_all_workloads.sh "$ds"
    python3 baseline_store.py save "$ds" fixed >> "$LOG"
  fi
  MODES=learn ./run_v2_all_workloads.sh "$ds"
  python3 compare_fixed_vs_learn.py "$ds" --tag nolog \
      > "/mnt/nvme0/v2-work/runs/${ds}_compare.txt" 2>&1
  python3 plot_fixed_vs_learn.py "$ds" --codec "$best" >> "$LOG" 2>&1
  echo "$(date +%T) $ds done" >> "$LOG"
done
echo "$(date +%T) FIXED_VS_LEARN_DONE $*" >> "$LOG"
