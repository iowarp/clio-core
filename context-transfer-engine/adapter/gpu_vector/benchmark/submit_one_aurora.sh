#!/usr/bin/env bash
# Submit ONE benchmark job, retrying while the queue's single Q slot is taken.
#
#   submit_one_aurora.sh <name> <tag> [exe-name] [args...]
#
# Writes build-spike/pbs/<tag>.log. Uses the _aot binary when no exe-name is
# given and one exists. The debug queue admits one queued job per user, so a
# refused qsub is retried every 20s rather than reported as a failure.
set -u
NAME=${1:?name}; TAG=${2:?tag}; shift 2
EXE=${1:-}; [ $# -gt 0 ] && shift
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "$EXE" ]; then
  EXE="clio_${NAME}_paged_newcoro"
  [ -x "$ROOT/build-spike/${EXE}_aot" ] && EXE="${EXE}_aot"
fi
ARGS="$*"
for i in $(seq 1 360); do   # 2 hours at 20s: the debug queue has held a single job for 30-40 min tonight
  out=$(qsub -N "coro_${TAG}" -o "$ROOT/build-spike/pbs/${TAG}.log" \
        -v "BENCH_NAME=${NAME},BENCH_EXE=${EXE},BENCH_ARGS=${ARGS},ROOT=${ROOT}${BENCH_EXTRA:+,${BENCH_EXTRA}}" \
        "$HERE/pbs_newcoro_aurora.sh" 2>&1)
  case "$out" in
    *"limit of jobs"*) sleep 20 ;;
    *) echo "SUBMITTED ${TAG} ${out} exe=${EXE} args=[${ARGS}]"; exit 0 ;;
  esac
done
echo "GAVE UP submitting ${TAG}"; exit 1
