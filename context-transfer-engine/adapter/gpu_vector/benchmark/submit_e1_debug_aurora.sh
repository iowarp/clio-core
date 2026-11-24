#!/usr/bin/env bash
# Submit the plan's E1 ladder in debug-scaling, ascending:
#
#   submit_e1_debug_aurora.sh [rung ...]     default: 4 8 16 32 64
#
# Weak scaling at the plan's anchor, 32000 MB per node, kmeans and grayscott
# on all four substrates, Eternia resident (see pbs_e1_scale_aurora.sh).
# Ascending on purpose, as the plan says: a night that only clears 4/8/16
# still yields a three-point curve.
#
# One job per rung, in sequence: debug-scaling admits one queued job per
# user, so each rung waits for the previous one to leave the queue.
set -u
RUNGS=${*:-"4 8 16 32 64"}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for n in ${RUNGS}; do
  log="${ROOT}/build-spike/pbs/e1dbg_${n}.log"
  : > "${log}"   # PBS appends to -o; never read a previous run's RESULT lines
  id=""
  for i in $(seq 1 360); do
    out=$(qsub -q debug-scaling -l walltime=01:00:00 -N "e1d_${n}" \
          -l select="${n}" -o "${log}" \
          -v "ROOT=${ROOT},BENCH_PERNODE_MB=${BENCH_PERNODE_MB:-32000},BENCH_ITERS=${BENCH_ITERS:-8},BENCH_CAP=${BENCH_CAP:-600},BENCH_BLOCKS=${BENCH_BLOCKS:-1024}" \
          "${HERE}/pbs_e1_scale_aurora.sh" 2>&1)
    case "${out}" in
      *"limit of jobs"*|*"per-user limit"*) sleep 20 ;;
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
  grep -h "^RESULT e1" "${log}" 2>/dev/null
done
echo "E1 DEBUG LADDER DONE"
