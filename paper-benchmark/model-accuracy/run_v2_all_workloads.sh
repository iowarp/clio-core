#!/usr/bin/env bash
#===============================================================================
# run_v2_all_workloads.sh -- every new-workloads dataset through Clio with
# NeuroPress v2 in all three modes (static, learnexp, exhaustive), one run at
# a time on the GPU so runs do not perturb each other's timings.
#
#   [MODES="static learnexp exhaustive"] run_v2_all_workloads.sh [DATASET ...]
#   (default: the 19 datasets, all three modes)
#
# Each run is run_v2_workloads.sh DATASET MODE; a failed run is logged and
# the queue moves on. Progress: /mnt/nvme0/v2-work/runs/all_workloads.log
#===============================================================================
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
LOG=/mnt/nvme0/v2-work/runs/all_workloads.log
DATASETS=("$@")
if [ ${#DATASETS[@]} -eq 0 ]; then
  DATASETS=(detector-frames dl-gpt2-kv dl-opt-relu-act consumer-nyx
            consumer-vpic-full consumer-lammps-full dl-pythia-ckpt
            genomics-reads hep-nanoaod dl-resnet18-train graph-orkut-full
            sparse-fem graph-livejournal-full dl-qwen-bf16 ref-lammps-b70-2000
            ref-nyx-256-2000 ref-vpic-126-2000 ref-warpx-64x64x512-2000
            astro-camels)
fi
echo "start $(date +%F_%T) ${#DATASETS[@]} datasets" >> "$LOG"
for ds in "${DATASETS[@]}"; do
  for mode in ${MODES:-static learnexp exhaustive}; do
    t0=$(date +%s)
    # A failed run is retried (up to 3 attempts): the runtime's port is picked
    # free and then bound, so it can be taken in between, and a rare CUDA
    # illegal-address error has ended one long exhaustive run.
    for attempt in 1 2 3; do
      if out=$("$HERE/run_v2_workloads.sh" "$ds" "$mode" 2>&1 | tail -1); then
        echo "$(date +%T) OK   $ds $mode $(( $(date +%s) - t0 )) s | $out" >> "$LOG"
        break
      fi
      if [ "$attempt" -eq 3 ]; then
        echo "$(date +%T) FAIL $ds $mode $(( $(date +%s) - t0 )) s | $out" >> "$LOG"
        break
      fi
      echo "$(date +%T) RETRY $ds $mode ($(grep -m1 -o \
        'Address already in use\|CUDA Error [0-9]*: [a-z ]*' \
        "/mnt/nvme0/v2-work/runs/${ds}_${mode}/runtime.log" \
        "/mnt/nvme0/v2-work/runs/${ds}_${mode}/stdout.log" 2>/dev/null |
        head -1 | sed 's/.*://'))" >> "$LOG"
      sleep 2
    done
  done
done
echo "done $(date +%F_%T)" >> "$LOG"
