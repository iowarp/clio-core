#!/usr/bin/env bash
# Submit the E7 ladder: the same 480 GB on fewer and fewer nodes.
#
#   submit_e7_fewer_aurora.sh [rung ...]     default: 32 16 12 8 6 4
#
# Descending on purpose. The high rungs are the cheap ones and the ones the
# baselines can still run, so a mistake shows up there first, and the 4-node
# rung -- the one that answers "can this run at all on almost nothing" --
# goes last when the script has already been exercised.
#
# One job per rung, submitted in sequence because debug-scaling admits one
# queued job per user. Each waits for the previous to leave the queue.
set -u
RUNGS=${*:-"32 16 12 8 6 4"}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for n in ${RUNGS}; do
  id=""
  for i in $(seq 1 360); do
    out=$(qsub -N "e7_${n}" -l select="${n}" \
          -o "${ROOT}/build-spike/pbs/e7_fewer_${n}.log" \
          -v "ROOT=${ROOT},BENCH_TOTAL_MB=${BENCH_TOTAL_MB:-491520},BENCH_STEPS=${BENCH_STEPS:-2},BENCH_CAP=${BENCH_CAP:-900},BENCH_CACHE_MB=${BENCH_CACHE_MB:-8192}" \
          "${HERE}/pbs_e7_fewer_aurora.sh" 2>&1)
    case "${out}" in
      *"limit of jobs"*) sleep 20 ;;
      *qsub:*) echo "SUBMIT-FAILED ${n}: ${out}"; break ;;
      *) id="${out}"; echo "SUBMITTED rung ${n}: ${id}"; break ;;
    esac
  done
  [ -n "${id}" ] || { echo "GAVE UP on rung ${n}"; continue; }
  while true; do
    st=$(qstat -f "${id}" 2>/dev/null | awk -F= '/job_state/{gsub(/[ \t]/,"",$2); print $2}')
    [ -z "${st}" ] && break
    [ "${st}" = "F" ] && break
    sleep 20
  done
  grep -h "^RESULT e7" "${ROOT}/build-spike/pbs/e7_fewer_${n}.log" 2>/dev/null | tail -4
done
echo "E7 LADDER DONE"
