#!/usr/bin/env bash
#===============================================================================
# run_full_all.sh -- every workload in full (no sampling) through Clio, one at
# a time: stage it if not staged (stage_full_workloads.py), then the
# exhaustive search, the best single codec's fixed run and NeuroPress v2
# learning with exploration off (run_fixed_vs_learn.sh), then the oracle
# (run_oracle_all.sh). Stored baselines that match the input are reused.
#
#   run_full_all.sh [DATASET ...]       (default: the 19 below, smallest first)
#
# Progress: /mnt/nvme0/v2-work/runs/full_all.log, ending FULL_ALL_DONE.
#===============================================================================
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
LOG=/mnt/nvme0/v2-work/runs/full_all.log
cd "$HERE"
DATASETS=("$@")
if [ ${#DATASETS[@]} -eq 0 ]; then
  DATASETS=(graph-livejournal-full dl-opt-relu-act dl-resnet18-train
            graph-orkut-full genomics-reads detector-frames sparse-fem dl-gpt2-kv
            dl-qwen-bf16 consumer-nyx consumer-vpic-full astro-camels hep-nanoaod
            dl-pythia-ckpt gnn-igbh consumer-lammps-full ref-lammps-b70-2000
            ref-vpic-126-2000 ref-warpx-64x64x512-2000)
fi
echo "$(date +%F_%T) start ${#DATASETS[@]} workloads" >> "$LOG"
for ds in "${DATASETS[@]}"; do
  if [ ! -d "/mnt/nvme0/v2-work/$ds/fields" ]; then
    python3 stage_full_workloads.py "$ds" >> "$LOG" 2>&1
  fi
  echo "$(date +%T) $ds start ($(ls "/mnt/nvme0/v2-work/$ds/fields" | wc -l) chunks)" >> "$LOG"
  ./run_fixed_vs_learn.sh "$ds"
  ./run_oracle_all.sh "$ds"
  echo "$(date +%T) $ds all runs done" >> "$LOG"
done
echo "$(date +%T) FULL_ALL_DONE" >> "$LOG"
