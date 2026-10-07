#!/usr/bin/env bash
# E10 load-step injector (EVAL_PLAN.md section 2, E10).
#
# Timeline, relative to the moment this script is started (t0 = the
# workflow's start):
#   t=120 s  consumer node: stress-ng --cpu on every core, 120 s
#   t=240 s  (released by the timeout above)
#   t=300 s  producer node: stress-ng --cpu on every core, 120 s
#
# Usage:
#   bash load_steps.sh <producer-host> <consumer-host>   # explicit hosts
#   bash load_steps.sh                                   # in a SLURM job:
#                       producer = 1st, consumer = 2nd host of the job
#
# Without --fg the script re-launches itself in the background and
# returns immediately, so it can be hooked into a pipeline through
# builtin.my_shell without blocking the packages that follow. The log is
# ${HOME}/jarvis-runs/dtschedule_e10/load_steps.log. stress-ng is
# bounded by --timeout, so nothing is left running after t=420 s; to
# abort early: ssh <host> pkill stress-ng.

set -u

if [ "${1:-}" != "--fg" ]; then
  log_dir="${HOME}/jarvis-runs/dtschedule_e10"
  mkdir -p "${log_dir}"
  nohup bash "$0" --fg "$@" >"${log_dir}/load_steps.log" 2>&1 &
  echo "load_steps: timeline started in the background (pid $!)"
  exit 0
fi
shift

producer="${1:-}"
consumer="${2:-}"
if [ -z "${producer}" ] || [ -z "${consumer}" ]; then
  if [ -n "${SLURM_JOB_NODELIST:-}" ]; then
    mapfile -t hosts < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
    producer="${producer:-${hosts[0]}}"
    consumer="${consumer:-${hosts[1]:-${hosts[0]}}}"
  else
    echo "load_steps: need <producer-host> <consumer-host> outside SLURM" >&2
    exit 1
  fi
fi

ssh_cmd="env -u LD_LIBRARY_PATH ssh -o BatchMode=yes"
t0=$(date +%s)

# Sleep until t0 + $1 seconds.
wait_until() {
  local target=$(( t0 + $1 ))
  local now
  now=$(date +%s)
  if [ "${now}" -lt "${target}" ]; then
    sleep $(( target - now ))
  fi
}

# Run stress-ng --cpu 0 (all cores) on $1 for $2 seconds (non-blocking).
inject() {
  echo "$(date +%H:%M:%S) t+$(( $(date +%s) - t0 ))s stress-ng --cpu 0 on $1 for $2 s"
  ${ssh_cmd} "$1" "stress-ng --cpu 0 --timeout $2s --quiet" &
}

echo "load_steps: t0=$(date +%H:%M:%S) producer=${producer} consumer=${consumer}"
wait_until 120
inject "${consumer}" 120
wait_until 300
inject "${producer}" 120
wait
echo "load_steps: done at t+$(( $(date +%s) - t0 ))s"
