#!/usr/bin/env bash
#===============================================================================
# run_exhaustive_e2e.sh -- the end-to-end exhaustive search: each candidate
# setting is the single codec of the whole workload in its own full
# application run (1 write + READS timed k-means reads), so every setting has
# its own measured wall-clock time.
#
#   run_exhaustive_e2e.sh DATASET [SETTINGS]
#     SETTINGS     comma list of setting indices (default: the 33 candidate
#                  settings of eval_v2_workloads.candidate_settings)
#   DRY_RUN=1      print the runs and start nothing
#   KEEP_DATA=1    keep each run's stored data; by default a verified run's
#                  chi_bdev.dat and cte_tier.dat* are deleted (one run stores
#                  up to the dataset size); its CSV files and logs stay
#
# The exhaustive mode of run_v2_workloads.sh measures the codec only: it
# compresses each chunk with all 45 settings in one pass, so it has one
# wall-clock time for all of them, no timed reads and no k-means. Here each
# setting is one run_v2_workloads.sh fixed run in the environment of the
# k-means runs (TIERS=0.52e6:1 COST_BW=520000 READS=4 KMEANS=8
# NO_SELECTION_LOG=1; each can be overridden), tag km4s<setting>:
# runs/DATASET_fixed_km4s<setting>. A setting with a complete, verified run
# is skipped, so the queue can stop and start again. The queue runs one run
# at a time and does not start when another replay runs on the GPU.
# At the end, collect_single_codecs.py writes the table
# runs/DATASET_exhaustive_e2e.csv.
# Log: runs/DATASET_exhaustive_e2e.log.
#===============================================================================
set -u
DS=$1
HERE=$(cd "$(dirname "$0")" && pwd)
PY_=${PYTHON:-$HOME/np-venv/bin/python}
# Default: the candidate settings (eval_v2_workloads.candidate_settings: all
# 45 but eval_v2_workloads.DROPPED_SETTINGS).
SETTINGS=${2:-$(cd "$HERE" && "$PY_" -c 'import eval_v2_workloads as ev; print(",".join(map(str, ev.candidate_settings())))')}
RUNS=/mnt/nvme0/v2-work/runs
LOG=$RUNS/${DS}_exhaustive_e2e.log
PY=${PYTHON:-$HOME/np-venv/bin/python}   # has numpy, pandas and torch
[ -n "${DRY_RUN:-}" ] && LOG=/dev/null
export TIERS=${TIERS:-0.52e6:1} COST_BW=${COST_BW:-520000} READS=${READS:-4} \
       KMEANS=${KMEANS:-8} NO_SELECTION_LOG=${NO_SELECTION_LOG:-1}

# complete RUN: stdout.log has the untimed read check and every chunk verified
complete() {
  [ -f "$1/stdout.log" ] && grep -q "READ check" "$1/stdout.log" &&
    grep -q "VERIFIED: \([0-9]*\) of \1 " "$1/stdout.log"
}

if [ -z "${DRY_RUN:-}" ] && pgrep -f bin/neuropress_field_replay > /dev/null; then
  echo "another neuropress_field_replay runs; no concurrent GPU work" >&2
  exit 1
fi
echo "$(date +%T) start $DS settings $SETTINGS (TIERS=$TIERS COST_BW=$COST_BW" \
     "READS=$READS KMEANS=$KMEANS)" | tee -a "$LOG"
for s in ${SETTINGS//,/ }; do
  RUN=$RUNS/${DS}_fixed_km4s$s
  if complete "$RUN"; then
    echo "$(date +%T) setting $s: complete run $(basename "$(realpath "$RUN")"), skipped" | tee -a "$LOG"
    continue
  fi
  if [ -n "${DRY_RUN:-}" ]; then
    echo "would run: RUN_TAG=km4s$s FIXED_SETTING=$s run_v2_workloads.sh $DS fixed"
    continue
  fi
  echo "$(date +%T) start setting $s" >> "$LOG"
  RUN_TAG=km4s$s FIXED_SETTING=$s "$HERE/run_v2_workloads.sh" "$DS" fixed >> "$LOG" 2>&1
  rc=$?
  echo "$(date +%T) setting $s done (rc $rc)" >> "$LOG"
  if [ $rc -eq 0 ] && [ -z "${KEEP_DATA:-}" ]; then
    rm -f "$RUN/chi_bdev.dat" "$RUN"/cte_tier.dat*
  fi
done
[ -n "${DRY_RUN:-}" ] && exit 0
"$PY" "$HERE/collect_single_codecs.py" "$DS" --settings "$SETTINGS" >> "$LOG" 2>&1
echo "$(date +%T) EXHAUSTIVE_E2E_DONE" >> "$LOG"
