#!/usr/bin/env bash
#
# CI gate for the clio FUSE filesystem xfstests conformance suite.
#
# Two lanes:
#
#   ENFORCED  ci_baseline_pass.txt MINUS ci_flaky_quarantine.txt. Every test
#             must pass; any non-pass (FAIL / notrun / HANG / missing from the
#             run) fails the job.
#   REPORTED  ci_flaky_quarantine.txt. Run, reported as a ::warning:: and in the
#             step summary, NEVER gating.
#
# WHY THE SECOND LANE EXISTS (issue #1022): between 2026-08-05 and 2026-08-25,
# 5 of 134 otherwise-clean CI runs (3.7%) went red on one or two of five
# xfstests that flake under the runner's constrained scheduling -- on branches
# that touched nothing those tests exercise. A red check that a PR author cannot
# act on is worse than no check: it trains everyone to re-run and stop reading.
# The evidence, per test, is in the header of ci_flaky_quarantine.txt.
#
# Quarantine is not deletion. These tests still run on every job; they just
# report instead of gate, so a quarantined test that starts failing 100% of the
# time is still visible in the log and can be promoted back.
#
# The currently-failing tests (never passing) are a separate matter -- they are
# not run here at all and are tracked, with root causes, in issue #680.
#
# Usage:
#   scripts/xfstests/run_ci_xfstests.sh
# Honors the same env as run_clio_xfstests.sh (CLIO_BUILD_DIR, XFSTESTS_DIR,
# CLIO_XFS_PERTEST_TIMEOUT, ...).
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASELINE_FILE="${SCRIPT_DIR}/ci_baseline_pass.txt"
QUARANTINE_FILE="${SCRIPT_DIR}/ci_flaky_quarantine.txt"
RUNNER="${SCRIPT_DIR}/run_clio_xfstests.sh"
SUMMARY="${GITHUB_STEP_SUMMARY:-/dev/null}"

[ -r "${BASELINE_FILE}" ] || { echo "ERROR: missing ${BASELINE_FILE}" >&2; exit 2; }
[ -r "${RUNNER}" ]        || { echo "ERROR: missing ${RUNNER}" >&2; exit 2; }

# --- read the two lists (strip comments / blanks) ---------------------------
mapfile -t ALL_LISTED < <(grep -oE '^generic/[0-9]+' "${BASELINE_FILE}" | sort -u)
[ "${#ALL_LISTED[@]}" -gt 0 ] || { echo "ERROR: baseline is empty" >&2; exit 2; }

QUARANTINED=()
if [ -r "${QUARANTINE_FILE}" ]; then
  mapfile -t QUARANTINED < <(grep -oE '^generic/[0-9]+' "${QUARANTINE_FILE}" | sort -u)
fi

# A quarantined test that is not in the baseline is a list-drift bug: it would
# silently be run by nobody and look, from the file, like it were covered.
for q in "${QUARANTINED[@]:-}"; do
  [ -z "${q}" ] && continue
  printf '%s\n' "${ALL_LISTED[@]}" | grep -qx "${q}" || {
    echo "::error::${q} is in ci_flaky_quarantine.txt but not in ci_baseline_pass.txt" >&2
    exit 2
  }
done

# ENFORCED = baseline - quarantine
mapfile -t ENFORCED < <(comm -23 \
  <(printf '%s\n' "${ALL_LISTED[@]}") \
  <(printf '%s\n' "${QUARANTINED[@]:-}" | sed '/^$/d' | sort -u))
[ "${#ENFORCED[@]}" -gt 0 ] || { echo "ERROR: every baseline test is quarantined" >&2; exit 2; }

echo "[ci-xfs] enforced: ${#ENFORCED[@]} test(s) (all must pass)"
echo "[ci-xfs] reported (flaky quarantine, non-gating): ${#QUARANTINED[@]} test(s)"

# --- run BOTH lanes in one driver invocation --------------------------------
# One invocation, not two: the driver remounts a fresh FUSE fs + runtime per
# test anyway, so splitting it would only pay the xfstests startup twice.
RUN_LOG="$(mktemp)"
trap 'rm -f "${RUN_LOG}"' EXIT
TESTS="${ALL_LISTED[*]}" bash "${RUNNER}" 2>&1 | tee "${RUN_LOG}"

status_of() {  # $1 = test id -> prints pass|FAIL|notrun|HANG|MISSING
  local s
  s="$(grep -oE "^$1 : [a-zA-Z]+" "${RUN_LOG}" | tail -1 | awk '{print $NF}')"
  [ -n "${s}" ] && echo "${s}" || echo "MISSING"
}

# --- REPORTED lane: never gates, always visible -----------------------------
q_bad=()
for t in "${QUARANTINED[@]:-}"; do
  [ -z "${t}" ] && continue
  st="$(status_of "${t}")"
  [ "${st}" = "pass" ] || q_bad+=("${t}(${st})")
done

{
  echo "## xfstests conformance (clio FUSE)"
  echo ""
  echo "- enforced: ${#ENFORCED[@]} test(s)"
  echo "- flaky quarantine (reported, non-gating): ${#QUARANTINED[@]} test(s)"
} >> "${SUMMARY}"

if [ "${#q_bad[@]}" -gt 0 ]; then
  echo "[ci-xfs] QUARANTINE (non-gating): ${#q_bad[@]} of ${#QUARANTINED[@]} did not pass:"
  printf '[ci-xfs]   ~ %s\n' "${q_bad[@]}"
  echo "::warning title=Flaky xfstests (quarantined, non-gating)::${q_bad[*]} -- see scripts/xfstests/ci_flaky_quarantine.txt"
  {
    echo ""
    echo "Quarantined tests that did not pass this run (did NOT fail the job):"
    echo '```'
    printf '  %s\n' "${q_bad[@]}"
    echo '```'
  } >> "${SUMMARY}"
else
  echo "[ci-xfs] QUARANTINE: all ${#QUARANTINED[@]} quarantined test(s) passed this run."
  {
    echo ""
    echo "All quarantined tests passed this run."
  } >> "${SUMMARY}"
fi

# --- ENFORCED lane: every test must pass ------------------------------------
failures=()
for t in "${ENFORCED[@]}"; do
  st="$(status_of "${t}")"
  [ "${st}" = "pass" ] || failures+=("${t}(${st})")
done

echo "===================================================================="
echo "[ci-xfs] SUMMARY: enforced=${#ENFORCED[@]} not-passing=${#failures[@]}"

if [ "${#failures[@]}" -gt 0 ]; then
  echo "[ci-xfs] FAIL: ${#failures[@]} enforced test(s) did not pass:"
  printf '[ci-xfs]   - %s\n' "${failures[@]}"
  {
    echo ""
    echo "**FAILED** — enforced tests that did not pass:"
    echo '```'
    printf '  %s\n' "${failures[@]}"
    echo '```'
  } >> "${SUMMARY}"
  exit 1
fi

echo "[ci-xfs] OK: all ${#ENFORCED[@]} enforced tests passed."
exit 0
