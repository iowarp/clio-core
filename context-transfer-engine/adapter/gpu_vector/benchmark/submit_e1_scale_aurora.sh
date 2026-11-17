#!/usr/bin/env bash
# Submit the E1 scaling ladder to the production queue.
#
#   submit_e1_scale_aurora.sh [rung ...]      default: 256 320 384 448 512
#
# WHY PRODUCTION. debug-scaling admits one running and one queued job per
# user, so a five-rung ladder there is five serial queue waits. `prod` routes
# to small/medium/large, caps queued jobs per PROJECT rather than per user
# (10 in small), and takes up to 12 hours -- so the whole ladder can sit in
# the queue at once. Its floor is 256 nodes, which is why the ladder starts
# there; anything smaller stays in debug-scaling.
#
# One job per rung, each running both workloads over all four editions.
# Submitted and left; this does not wait, because the rungs are independent
# and the point of production is that they queue together.
set -u
RUNGS=${*:-"256 320 384 448 512"}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for n in ${RUNGS}; do
  if [ "${n}" -lt 256 ]; then
    echo "SKIP ${n}: below the production floor of 256 nodes"
    continue
  fi
  out=$(qsub -N "e1_${n}" -l select="${n}" \
        -o "${ROOT}/build-spike/pbs/e1_scale_${n}.log" \
        -v "ROOT=${ROOT},BENCH_ITERS=${BENCH_ITERS:-20},BENCH_CAP=${BENCH_CAP:-900},BENCH_PERNODE_MB=${BENCH_PERNODE_MB:-4096}" \
        "${HERE}/pbs_e1_scale_aurora.sh" 2>&1)
  case "${out}" in
    *qsub:*) echo "SUBMIT-FAILED ${n}: ${out}" ;;
    *)       echo "SUBMITTED rung ${n}: ${out}" ;;
  esac
done
echo "E1 LADDER SUBMITTED"
