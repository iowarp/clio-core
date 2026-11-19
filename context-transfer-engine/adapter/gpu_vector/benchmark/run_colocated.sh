#!/usr/bin/env bash
# Run an N-"node" gpu_vector benchmark on ONE host: N runtimes, one per rank,
# each on its own port and its own GPU tile. This is the smallest scale at
# which the distributed code paths (remote puts, generations, tags, probes)
# run at all, so it is where defects are reproduced first.
#
#   run_colocated.sh <n> <workdir> <exe> [benchmark args...]
#
# Each rank r gets CLIO_PORT = BENCH_BASE_PORT + 8*r, ZE_AFFINITY_MASK for
# tile (r/2).(r%2), the same generated yaml (the env port overrides the
# yaml's), and "--nodes n --node r". The hostfile lists "<host>:<port>" per
# rank, which the runtime resolves to its own entry by port.
#
# Environment: BENCH_BASE_PORT (9460), BENCH_CAP seconds per rank (600),
# BENCH_RANK_DELAY_<r> seconds to delay rank r's start, BENCH_RANK_KILL_<r>
# seconds after which rank r is SIGKILLed, BENCH_RANK_ENV_<r> extra
# "VAR=value ..." exported to rank r only (all three are fault injection),
# plus everything bench_config.sh reads. Exit status: the worst rank's.
set -u
n=${1:?usage: run_colocated.sh <n> <workdir> <exe> [args]}
workdir=${2:?workdir}
exe=${3:?exe}
shift 3
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=bench_config.sh
source "${here}/bench_config.sh"

bench_check_binary "${exe}" || exit 2
mkdir -p "${workdir}"
base=${BENCH_BASE_PORT:-9460}
host=$(hostname)
: > "${workdir}/hostfile"
for (( r = 0; r < n; ++r )); do
  echo "${host}:$(( base + 8 * r ))" >> "${workdir}/hostfile"
done
bench_clio_yaml "${workdir}/clio.yaml" "${workdir}/hostfile" "${base}"

# Each rank runs in its own process group (setsid), so stopping a rank stops
# the timeout wrapper AND the benchmark under it. Killing only the subshell
# left benchmarks (and their embedded runtimes) alive on the node, and every
# later run attached to those stale runtimes as a client and crashed.
pids=()
for (( r = 0; r < n; ++r )); do
  setsid bash -c '
    r=$1; n=$2; workdir=$3; exe=$4; base=$5; host=$6; shift 6
    cd "${workdir}" || exit 99
    # BENCH_CORES=1 keeps core dumps (in the workdir) for a backtrace.
    if [ "${BENCH_CORES:-0}" = 1 ]; then ulimit -c unlimited; else ulimit -c 0; fi
    export CLIO_PORT=$(( base + 8 * r ))
    export CLIO_SERVER_CONF="${workdir}/clio.yaml"
    export ZE_AFFINITY_MASK="$(( r / 2 )).$(( r % 2 ))"
    evar="BENCH_RANK_ENV_${r}"
    for kv in ${!evar:-}; do export "${kv?}"; done
    dvar="BENCH_RANK_DELAY_${r}"
    [ -n "${!dvar:-}" ] && sleep "${!dvar}"
    kvar="BENCH_RANK_KILL_${r}"
    cap=${BENCH_CAP:-600}; sig=TERM
    if [ -n "${!kvar:-}" ]; then cap=${!kvar}; sig=KILL; fi
    # --foreground keeps the benchmark in this process group (GNU timeout
    # otherwise moves its child to a new one), so a group kill reaches it.
    timeout --foreground --signal="${sig}" --kill-after=10s "${cap}" \
      stdbuf -oL -eL "${exe}" "$@" --nodes "${n}" --node "${r}" \
      > "rank${r}.log" 2>&1
    rc=$?
    echo "rank ${r} on ${host} port ${CLIO_PORT} exit=${rc}" >> "rank${r}.log"
    exit "${rc}"
  ' rank "${r}" "${n}" "${workdir}" "${exe}" "${base}" "${host}" "$@" &
  pids+=($!)
done
# FAIL FAST: the first rank to exit non-zero stops the rest (they would
# otherwise park at ROUND CAP until the cap). BENCH_NO_FAILFAST=1 disables.
worst=0
alive=${n}
while [ "${alive}" -gt 0 ]; do
  alive=0
  for (( r = 0; r < n; ++r )); do
    kill -0 "${pids[$r]}" 2>/dev/null && alive=$(( alive + 1 ))
  done
  bad=$(grep -lE '^rank [0-9]+ on .* exit=[1-9][0-9]*$' "${workdir}"/rank*.log 2>/dev/null | head -1)
  if [ -n "${bad}" ] && [ "${BENCH_NO_FAILFAST:-0}" != 1 ]; then
    echo "FAILFAST: $(basename "${bad}" .log) failed; stopping the other ranks"
    for (( r = 0; r < n; ++r )); do kill -TERM -- "-${pids[$r]}" 2>/dev/null; done
    sleep 5
    for (( r = 0; r < n; ++r )); do kill -KILL -- "-${pids[$r]}" 2>/dev/null; done
    break
  fi
  sleep 2
done
for (( r = 0; r < n; ++r )); do
  wait "${pids[$r]}"; rc=$?
  [ "${rc}" -gt "${worst}" ] && worst=${rc}
  echo "rank ${r}: exit=${rc}"
done
echo "RESULT colocated n=${n}: worst exit=${worst}"
exit "${worst}"
