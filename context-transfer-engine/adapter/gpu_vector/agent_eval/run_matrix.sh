#!/usr/bin/env bash
# Run a grid of trials, PARALLEL at a time, then grade them one at a time.
#
#   TASKS="kmeans" MODELS="claude-opus-5-5 claude-sonnet-5" ARMS="none skill" \
#   REPS="1 2 3" PARALLEL=2 ./run_matrix.sh
#
# Trials share the one GPU while the agents work, which is fine for the
# agents; grading happens afterwards, serially, so performance is measured on
# an idle GPU. The script can be re-run to fill in a grid. A trial directory that exists (finished or
# still running) is skipped; delete it to rerun that trial.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/env.sh"

TASKS="${TASKS:-kmeans}"
MODELS="${MODELS:-claude-opus-5-5 claude-sonnet-5}"
ARMS="${ARMS:-none skill}"
REPS="${REPS:-1}"
PARALLEL="${PARALLEL:-3}"   # launch concurrency; MAX_CONCURRENT caps containers host-wide

jobs=()
for rep in ${REPS}; do for task in ${TASKS}; do for arm in ${ARMS}; do for model in ${MODELS}; do
  [ -e "${RUNS_HOST_VIEW}/trials/${task}/${arm}/${model}/r${rep}" ] && continue
  jobs+=("${model} ${task} ${arm} ${rep}")
done; done; done; done

echo "running ${#jobs[@]} trials, ${PARALLEL} at a time"
printf '%s\n' "${jobs[@]}" | xargs -P "${PARALLEL}" -L 1 "${HERE}/run_trial.sh"

# Grade only on an idle GPU: wait for every trial container, whoever
# launched it, to finish.
while docker ps --format '{{.Names}}' | grep -q '^agent_eval_'; do sleep 60; done
echo "grading"
for t in $(ls -d "${RUNS_HOST_VIEW}"/trials/*/*/*/r* 2>/dev/null); do
  [ -e "$t/meta.json" ] || { echo "skip (no meta.json): $t"; continue; }
  [ -e "$t/grade.json" ] || "${HERE}/grade.py" trial "$t"
done
"${HERE}/analyze.py" --csv "${RUNS_HOST_VIEW}/results.csv"
