#!/usr/bin/env bash
# Submit a probe job script (one node, five minutes) and wait for it.
#   submit_probe_aurora.sh <script> <tag> [extra qsub -v vars]
set -u
SCRIPT=${1:?script}; TAG=${2:?tag}; EXTRA=${3:-}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
mkdir -p "$ROOT/build-spike/pbs"
id=""
for i in $(seq 1 360); do
  out=$(qsub -N "probe_${TAG}" -o "$ROOT/build-spike/pbs/${TAG}.log" \
        ${EXTRA:+-v "$EXTRA"} "$SCRIPT" 2>&1)
  case "$out" in
    *"limit of jobs"*) sleep 20 ;;
    *qsub:*) echo "SUBMIT-FAILED ${TAG}: ${out}"; exit 1 ;;
    *) id=$out; echo "SUBMITTED ${TAG} ${id}"; break ;;
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
