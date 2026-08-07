#!/bin/bash
# Local driver for the Distributed docker CI (mirrors .github/workflows/distributed-test.yml).
# Runs every suite regardless of earlier failures, then prints a summary.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export HOST_UID="$(id -u)" HOST_GID="$(id -g)"
LOGDIR=/tmp/distci
mkdir -p "$LOGDIR"

declare -a NAMES RESULTS
run_suite() {
  local name="$1"; shift
  echo "=================================================================="
  echo ">>> SUITE: $name"
  echo "=================================================================="
  local log="$LOGDIR/suite_${name}.log"
  ( "$@" ) >"$log" 2>&1
  local rc=$?
  NAMES+=("$name"); RESULTS+=("$rc")
  if [ $rc -eq 0 ]; then echo "<<< $name: PASS"; else echo "<<< $name: FAIL (rc=$rc)"; fi
}

run_suite runtime_distributed   ./context-runtime/test/integration/distributed/run_tests.sh all
run_suite runtime_expand        ./context-runtime/test/integration/expand/run_tests.sh all
run_suite runtime_migrate       ./context-runtime/test/integration/migrate/run_tests.sh all
run_suite runtime_reconnect     ./context-runtime/test/integration/reconnect/run_tests.sh all
run_suite runtime_leader_elect  ./context-runtime/test/integration/leader_elect/run_tests.sh all
run_suite runtime_recovery      ./context-runtime/test/integration/recovery/run_tests.sh all
run_suite cte_distributed       ./context-transfer-engine/test/integration/distributed/run_tests.sh
IOWARP_DOCKER_IMAGE=clio-fuse-test:local \
run_suite cte_fuse              env IOWARP_DOCKER_IMAGE=clio-fuse-test:local ./context-transfer-engine/test/integration/fuse/run_tests.sh

echo
echo "================= DISTRIBUTED CI SUMMARY ================="
fail=0
for i in "${!NAMES[@]}"; do
  if [ "${RESULTS[$i]}" -eq 0 ]; then s=PASS; else s="FAIL(${RESULTS[$i]})"; ((fail++)); fi
  printf "  %-22s %s\n" "${NAMES[$i]}" "$s"
done
echo "========================================================="
echo "TOTAL: ${#NAMES[@]} suites, $fail failed"
exit $fail
