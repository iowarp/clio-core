#!/usr/bin/env bash
# Submit and run one single-node Aurora job per ported benchmark, in sequence.
#
# ONE JOB EACH, deliberately: a hang or a crash in one benchmark then says
# nothing about the others, each gets its own log, and a benchmark that needs
# re-running does not drag the other five with it.
#
# SEQUENTIAL, not parallel, and not a dependency chain either. The debug queue
# allows one job per user in the Q state; six independent submissions get one
# job and five refusals, and `-W depend=afterany` is refused the same way --
# the limit is applied at submission, before the hold would take effect. So
# this submits one, waits for it to leave the queue, and submits the next.
#
# Five minute walltime each. Arguments are the benchmarks' own defaults except
# where a default is large enough to threaten that -- lammps_md's 100 steps.
# The job script caps the run itself at 240s regardless, so a config that turns
# out to be too big reports TIMEOUT with its output intact rather than being
# cut off mid-sentence when PBS reaps the job.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../../.." && pwd)"
JOB="${HERE}/pbs_newcoro_aurora.sh"
LOGS="${ROOT}/build-spike/pbs"
mkdir -p "${LOGS}"

# name:args
# SIZED FOR A 90s CAP, not for the numbers in RESULTS.md. The container ran
# these under a 20-minute timeout; the defaults are benchmark-scale (kmeans:
# 2 GB through a 512 MB cache x4 iterations; grayscott: 16 GB through 4 GB x3
# repeats). Every size here still exceeds its cache so the paging path -- the
# whole point -- is exercised on every run. Scale up only once they pass.
BENCHES=(
  "gmx:--page-kb 128"   # its default, 64, is not a square u64 plane; the benchmark refuses it
  "lbann:"              # 5 MB of weights, 5 steps: already small
  "weights:--repeat 1"
  "grayscott:--data-mb 512 --hbm-mb 128 --repeat 1 --steps 2"
  "kmeans:--data-mb 256 --hbm-mb 128"
  "lammps_md:--steps 20"
)

# Benchmarks already submitted by hand: skip the submit, still report.
SKIP=${SKIP:-}
# An already-queued job to wait on before the first submission.
PREV=${PREV:-}

wait_for() {  # wait_for <jobid>
  local id=$1 st
  while true; do
    st=$(qstat -f "${id}" 2>/dev/null |
         awk -F= '/job_state/{gsub(/[ \t]/,"",$2); print $2}')
    # Gone from qstat, or Finished: either way it is over.
    [ -z "${st}" ] && break
    [ "${st}" = "F" ] && break
    sleep 20
  done
}

# The queue admits ONE job per user in Q. Anything already queued -- a probe,
# a hand-submitted job -- makes every qsub here fail instantly, and the loop
# would then walk all six benchmarks in a second reporting SUBMIT-FAILED. So
# wait for the slot before each submission rather than assume it.
wait_slot() {
  while [ -n "$(qstat -u "$USER" 2>/dev/null | awk '$10=="Q"')" ]; do sleep 15; done
}

report() {  # report <name>
  local name=$1 line newest
  # The NEWEST log for this benchmark, not the one the driver would have
  # written: a hand-submitted run (kmeans_small.log, weights_lock.log) is the
  # current state, and a SKIP'd benchmark reported from a stale ${name}.log
  # once printed "kmeans: FAILED" an hour after kmeans had passed.
  newest=$(ls -t "${LOGS}/${name}"*.log 2>/dev/null | head -1)
  line=$(grep -h "^RESULT" "${newest:-/dev/null}" 2>/dev/null | tail -1)
  echo "${line:-RESULT ${name}: no result line -- see ${LOGS}/}"
}

[ -n "${PREV}" ] && wait_for "${PREV}"

for b in "${BENCHES[@]}"; do
  name=${b%%:*}
  args=${b#*:}
  case " ${SKIP} " in
    *" ${name} "*) report "${name}"; continue ;;
  esac
  # Prefer the _aot binary: PVC code pre-compiled through IGC on the login
  # node, so the 90s cap is spent on the workload rather than on JIT.
  exe_name="clio_${name}_paged_newcoro"
  [ -x "${ROOT}/build-spike/${exe_name}_aot" ] && exe_name="${exe_name}_aot"
  exe="${ROOT}/build-spike/${exe_name}"
  if [ ! -x "${exe}" ]; then
    echo "SKIP ${name} -- not built"
    continue
  fi

  wait_slot
  id=$(qsub -N "coro_${name}" \
            -o "${LOGS}/${name}.log" \
            -v "BENCH_NAME=${name},BENCH_EXE=${exe_name},BENCH_ARGS=${args},ROOT=${ROOT}" \
            "${JOB}" 2>&1)
  if [ "${id#*qsub:}" != "${id}" ]; then
    echo "SUBMIT-FAILED ${name}: ${id}"
    continue
  fi
  echo "SUBMITTED ${name} ${id}"
  wait_for "${id}"
  report "${name}"
done

echo "ALL DONE"
