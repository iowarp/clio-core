#!/usr/bin/env bash
#
# CI gate for the SCRATCH-requiring clio FUSE xfstests.
#
# Runs ONLY the scratch tests known to pass (ci_scratch_baseline_pass.txt) via
# run_clio_scratch_xfstests.sh (two isolated runtimes + xfstests native fuse
# mount-helper support). Every listed test must pass; any non-pass fails the job.
#
# The many scratch tests that 'notrun' (fuse feature gaps: shutdown, dm targets,
# symlinks-on-fuse, ...) and the ones that currently FAIL (tracked in issue
# #680) are intentionally NOT listed here.
#
# Usage: scripts/xfstests/run_ci_scratch_xfstests.sh
# Honors the same env as run_clio_scratch_xfstests.sh.
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_FILE="${SCRIPT_DIR}/ci_scratch_baseline_pass.txt"
RUNNER="${SCRIPT_DIR}/run_clio_scratch_xfstests.sh"

[ -r "${BASELINE_FILE}" ] || { echo "ERROR: missing ${BASELINE_FILE}" >&2; exit 2; }
[ -r "${RUNNER}" ]        || { echo "ERROR: missing ${RUNNER}" >&2; exit 2; }

mapfile -t BASELINE < <(grep -oE '^generic/[0-9]+' "${BASELINE_FILE}" | sort -u)
[ "${#BASELINE[@]}" -gt 0 ] || { echo "ERROR: scratch baseline is empty" >&2; exit 2; }

echo "[ci-scratch] running ${#BASELINE[@]} scratch baseline test(s) (all must pass)"

RUN_LOG="$(mktemp)"
trap 'rm -f "${RUN_LOG}"' EXIT
TESTS="${BASELINE[*]}" bash "${RUNNER}" 2>&1 | tee "${RUN_LOG}"

# run_clio_scratch_xfstests.sh prints one "<test>: <status>" line per test.
status_of() {  # $1 = test id -> pass|FAIL|notrun|HANG|MISSING
  local s
  # Tolerate an optional space before the colon ("<id>: x" or "<id> : x").
  s="$(grep -oE "^$1 ?: [A-Za-z]+" "${RUN_LOG}" | tail -1 | awk '{print $NF}')"
  [ -n "${s}" ] && echo "${s}" || echo "MISSING"
}

failures=()
for t in "${BASELINE[@]}"; do
  st="$(status_of "${t}")"
  [ "${st}" = "pass" ] || failures+=("${t}(${st})")
done

echo "===================================================================="
echo "[ci-scratch] SUMMARY: baseline=${#BASELINE[@]} not-passing=${#failures[@]}"
if [ "${#failures[@]}" -gt 0 ]; then
  echo "[ci-scratch] FAIL: ${#failures[@]} scratch baseline test(s) did not pass:"
  printf '[ci-scratch]   - %s\n' "${failures[@]}"
  exit 1
fi
echo "[ci-scratch] OK: all ${#BASELINE[@]} scratch baseline tests passed."
exit 0
