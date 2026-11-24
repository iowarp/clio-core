#!/usr/bin/env bash
# Submit ONE two-node benchmark job, retrying while the queue's single Q slot
# is taken, and wait for it to finish.
#
#   submit_2n_aurora.sh <name> <tag> [args...]
#
# JOB_SCRIPT selects another job script in this directory (default
# pbs_newcoro_aurora_2n.sh); BENCH_EXTRA adds comma-separated qsub -v vars.
#
# Writes build-spike/pbs/<tag>.log. Waits (rather than returns) because the
# debug queue admits one queued job per user and these are run in sequence.
set -u
NAME=${1:?name}; TAG=${2:?tag}; shift 2
ARGS="$*"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
id=""
for i in $(seq 1 360); do
  out=$(qsub -N "c2n_${TAG}" -o "$ROOT/build-spike/pbs/${TAG}.log" \
        -v "BENCH_NAME=${NAME},BENCH_ARGS=${ARGS},ROOT=${ROOT}${BENCH_EXTRA:+,${BENCH_EXTRA}}" \
        "$HERE/${JOB_SCRIPT:-pbs_newcoro_aurora_2n.sh}" 2>&1)
  case "$out" in
    *"limit of jobs"*) sleep 20 ;;
    *qsub:*) echo "SUBMIT-FAILED ${TAG}: ${out}"; exit 1 ;;
    *) id=$out; echo "SUBMITTED ${TAG} ${id} args=[${ARGS}]"; break ;;
  esac
done
[ -n "$id" ] || { echo "GAVE UP submitting ${TAG}"; exit 1; }
while true; do
  st=$(qstat -f "${id}" 2>/dev/null | awk -F= '/job_state/{gsub(/[ \t]/,"",$2); print $2}')
  [ -z "${st}" ] && break
  [ "${st}" = "F" ] && break
  sleep 20
done
grep -h "^RESULT" "$ROOT/build-spike/pbs/${TAG}.log" 2>/dev/null | tail -1
