#!/usr/bin/env bash
# Submit ONE wide job that runs many E4 cells concurrently, and wait for it.
#
#   submit_e4_batch_aurora.sh <tag> <workload:composition>...
#
# debug-scaling allows one running and one queued job per user, so cells
# submitted one per job are serialised by the queue: 8-10 minutes of waiting
# for 20-60 seconds of work. This asks for NODES nodes once and lets
# pbs_e4_batch_aurora.sh run NODES/4 cells at a time inside it.
#
#   NODES      allocation size, a multiple of 4 (default 16 -> 4 at a time)
#   BENCH_GROUP_N  nodes per cell (default 4). At 64 a 256-node allocation
#              runs four 64-node cells at once, which is how the scaled
#              E5 stresses DAOS and Lustre.
#   DATA_MB, TIER_BUDGET_MB, HBM_MB  passed through to the job; E3 needs a
#              smaller deck than the default because a checkpoint is a
#              SECOND full copy of the vector and has to fit the tier too
#   BENCH_CAP  per-rank cap in seconds (default 600)
set -u
TAG=${1:?tag}; shift
CELLS="$*"
[ -n "${CELLS}" ] || { echo "no cells given"; exit 2; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NODES=${NODES:-16}
id=""
for i in $(seq 1 360); do
  out=$(qsub -N "e4b_${TAG}" -l select="${NODES}" \
        -o "$ROOT/build-spike/pbs/e4batch_${TAG}.log" \
        -v "BENCH_CELLS=${CELLS},ROOT=${ROOT},BENCH_CAP=${BENCH_CAP:-600},DATA_MB=${DATA_MB:-32768},TIER_BUDGET_MB=${TIER_BUDGET_MB:-10240},HBM_MB=${HBM_MB:-4096},BENCH_GROUP_N=${BENCH_GROUP_N:-4},BENCH_STEPS=${BENCH_STEPS:-2},BENCH_CKPT_EVERY=${BENCH_CKPT_EVERY:-1}" \
        "$HERE/pbs_e4_batch_aurora.sh" 2>&1)
  case "$out" in
    *"limit of jobs"*) sleep 20 ;;
    *qsub:*) echo "SUBMIT-FAILED ${TAG}: ${out}"; exit 1 ;;
    *) id=$out; echo "SUBMITTED ${TAG} ${id} cells=[${CELLS}] nodes=${NODES}"; break ;;
  esac
done
[ -n "$id" ] || { echo "GAVE UP submitting ${TAG}"; exit 1; }
while true; do
  st=$(qstat -f "${id}" 2>/dev/null | awk -F= '/job_state/{gsub(/[ \t]/,"",$2); print $2}')
  [ -z "${st}" ] && break
  [ "${st}" = "F" ] && break
  sleep 20
done
grep -h "^CELL DONE\|^E4 BATCH DONE" "$ROOT/build-spike/pbs/e4batch_${TAG}.log" 2>/dev/null
