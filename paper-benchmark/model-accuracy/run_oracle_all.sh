#!/usr/bin/env bash
#===============================================================================
# run_oracle_all.sh -- the oracle baseline through Clio for every workload with
# a stored exhaustive search: each chunk stored with its own cheapest setting
# (oracle_map.py, balanced 4-tier cost model), no NeuroPress work. One workload
# at a time; a stored oracle run that matches the input is reused
# (baseline_store.py), FORCE_BASELINES=1 reruns it.
#
#   run_oracle_all.sh [DATASET ...]     (default: every stored workload)
#
# Progress: /mnt/nvme0/v2-work/runs/oracle_all.log
#===============================================================================
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
LOG=/mnt/nvme0/v2-work/runs/oracle_all.log
export NO_SELECTION_LOG=1 RUN_TAG=nolog
cd "$HERE"
DATASETS=("$@")
if [ ${#DATASETS[@]} -eq 0 ]; then
  mapfile -t DATASETS < <(ls -d /mnt/nvme0/v2-work/baselines/*/exhaustive | xargs -n1 dirname | xargs -n1 basename)
fi
for ds in "${DATASETS[@]}"; do
  if [ -z "${FORCE_BASELINES:-}" ] && python3 baseline_store.py has "$ds" oracle; then
    python3 baseline_store.py restore "$ds" oracle >> "$LOG"
    echo "$(date +%T) $ds oracle reused from the store" >> "$LOG"
    continue
  fi
  python3 oracle_map.py "$ds" >> "$LOG"
  ORACLE_MAP="/mnt/nvme0/v2-work/runs/${ds}_oracle_map.csv" MODES=oracle \
      ./run_v2_all_workloads.sh "$ds"
  python3 baseline_store.py save "$ds" oracle >> "$LOG"
  echo "$(date +%T) $ds oracle done" >> "$LOG"
done
echo "$(date +%T) ORACLE_ALL_DONE" >> "$LOG"
