#!/usr/bin/env bash
#
# CI gate for the clio FUSE filesystem xfstests conformance suite.
#
# Runs ONLY the tests known to pass (scripts/xfstests/ci_baseline_pass.txt --
# the maximum passing set discovered across the whole generic/* group) against
# the clio FUSE fs via run_clio_xfstests.sh. Every listed test must pass; any
# non-pass (FAIL / notrun / HANG / missing from the run) fails the job.
#
# The currently-failing tests are intentionally NOT run here -- they are
# tracked, with root causes, in issue #680.
#
# Usage:
#   scripts/xfstests/run_ci_xfstests.sh
# Honors the same env as run_clio_xfstests.sh (CLIO_BUILD_DIR, XFSTESTS_DIR,
# CLIO_XFS_PERTEST_TIMEOUT, ...).
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASELINE_FILE="${SCRIPT_DIR}/ci_baseline_pass.txt"
RUNNER="${SCRIPT_DIR}/run_clio_xfstests.sh"

[ -r "${BASELINE_FILE}" ] || { echo "ERROR: missing ${BASELINE_FILE}" >&2; exit 2; }
[ -r "${RUNNER}" ]        || { echo "ERROR: missing ${RUNNER}" >&2; exit 2; }

# --- read the expected-pass list (strip comments / blanks) ------------------
mapfile -t BASELINE < <(grep -oE '^generic/[0-9]+' "${BASELINE_FILE}" | sort -u)
[ "${#BASELINE[@]}" -gt 0 ] || { echo "ERROR: baseline is empty" >&2; exit 2; }

echo "[ci-xfs] running ${#BASELINE[@]} baseline test(s) (all must pass)"

# --- run the conformance driver, capture per-test outcomes ------------------
RUN_LOG="$(mktemp)"
trap 'rm -f "${RUN_LOG}"' EXIT
TESTS="${BASELINE[*]}" bash "${RUNNER}" 2>&1 | tee "${RUN_LOG}"

status_of() {  # $1 = test id -> prints pass|FAIL|notrun|HANG|MISSING
  local s
  s="$(grep -oE "^$1 : [a-zA-Z]+" "${RUN_LOG}" | tail -1 | awk '{print $NF}')"
  [ -n "${s}" ] && echo "${s}" || echo "MISSING"
}

# --- every baseline test must pass ------------------------------------------
failures=()
for t in "${BASELINE[@]}"; do
  st="$(status_of "${t}")"
  [ "${st}" = "pass" ] || failures+=("${t}(${st})")
done

echo "===================================================================="
echo "[ci-xfs] SUMMARY: baseline=${#BASELINE[@]} not-passing=${#failures[@]}"

if [ "${#failures[@]}" -gt 0 ]; then
  echo "[ci-xfs] FAIL: ${#failures[@]} baseline test(s) did not pass:"
  printf '[ci-xfs]   - %s\n' "${failures[@]}"
  exit 1
fi

echo "[ci-xfs] OK: all ${#BASELINE[@]} baseline tests passed."
exit 0
