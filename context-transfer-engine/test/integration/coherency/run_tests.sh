#!/bin/bash
# ---------------------------------------------------------------------------
# 4-node CTE memory-coherency integration test runner.
#
# Brings up 4 containers, each running clio_run + test_cte_coherency as a
# client, and PASSES ONLY IF all four nodes report 3/3 coherency cases.
#
# Guards (hard-won):
#  * Wipes the shared barrier volume every run (down -v) so stale READY/TESTDONE
#    /bar_* files can't let a node skip a barrier and fake a pass.
#  * A vacuous run is NOT a pass: the test binary exits 0 if a filter matches no
#    cases (prints "Passed: 0"). We assert "Total tests: 3 / Passed: 3 / Failed:
#    0" per node, not merely exit 0.
#  * Aggregates ALL four nodes; any node failing fails the suite (one
#    --abort-on-container-exit code is not enough).
# ---------------------------------------------------------------------------
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../" && pwd)"

if [ -n "${HOST_WORKSPACE:-}" ]; then
    export IOWARP_CORE_ROOT="${HOST_WORKSPACE}"
elif [ -z "${IOWARP_CORE_ROOT:-}" ]; then
    export IOWARP_CORE_ROOT="${REPO_ROOT}"
fi

cd "$SCRIPT_DIR"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
msg()     { local c=$1; shift; echo -e "${c}$*${NC}"; }
header()  { echo ""; msg "$BLUE" "=================================="; msg "$BLUE" "$*"; msg "$BLUE" "=================================="; }
ok()      { msg "$GREEN" "✓ $*"; }
err()     { msg "$RED" "✗ $*"; }
warn()    { msg "$YELLOW" "⚠ $*"; }

NODES="1 2 3 4"
cname()   { echo "cte-coherency-node$1"; }

keep=false
do_cleanup=true
cleanup_only=false
while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help) echo "Usage: $0 [-k|--keep] [-n|--no-cleanup] [-c|--cleanup-only]"; exit 0 ;;
    -k|--keep) keep=true; shift ;;
    -n|--no-cleanup) do_cleanup=false; shift ;;
    -c|--cleanup-only) cleanup_only=true; shift ;;
    *) err "Unknown option: $1"; exit 1 ;;
  esac
done

check_prereqs() {
  header "Checking Prerequisites"
  command -v docker >/dev/null 2>&1 || { err "Docker not installed"; exit 1; }
  docker compose version >/dev/null 2>&1 || { err "Docker Compose not installed"; exit 1; }
  docker ps >/dev/null 2>&1 || { err "Docker daemon not running"; exit 1; }
  [ -f docker-compose.yaml ] || { err "docker-compose.yaml missing"; exit 1; }
  [ -f clio_config.yaml ]    || { err "clio_config.yaml missing"; exit 1; }
  [ -f node_entrypoint.sh ]  || { err "node_entrypoint.sh missing"; exit 1; }
  # The prerequisite build must exist. Test for existence (-f), not the host
  # exec bit (-x): the binaries execute inside the Linux node containers, not on
  # this host, and some host filesystems (e.g. a Windows/NTFS bind mount) drop
  # the exec bit even though it is set inside the container.
  local bin="${IOWARP_CORE_ROOT}/build/bin"
  if [ ! -f "$bin/clio_run" ] || [ ! -f "$bin/test_cte_coherency" ]; then
    err "Missing binaries in $bin (need clio_run and test_cte_coherency)."
    err "Build first: cmake --build build --target clio_run test_cte_coherency"
    exit 1
  fi
  ok "Prerequisites OK"
}

# Wipe containers AND the named barrier volume (-v) so nothing stale survives.
cleanup() {
  header "Cleaning up previous run (containers + shared volume)"
  docker compose down -v --remove-orphans >/dev/null 2>&1 || true
  ok "Clean"
}

detect_image() {
  if [ -z "${IOWARP_DOCKER_IMAGE:-}" ]; then
    local clio_bin="${IOWARP_CORE_ROOT}/build/bin/clio_run"
    if [ -f "$clio_bin" ] && ldd "$clio_bin" 2>/dev/null | grep -q "libcudart"; then
      export IOWARP_DOCKER_IMAGE="iowarp/deps-nvidia:latest"
      local bindir; bindir="$(dirname "$clio_bin")"
      ldd "$clio_bin" 2>/dev/null | awk '/libcudart/{print $3}' | while read -r lib; do
        { [ -n "$lib" ] && [ -f "$lib" ] && cp -Lu "$lib" "$bindir/" 2>/dev/null; } || true
      done
    else
      export IOWARP_DOCKER_IMAGE="iowarp/deps-cpu:latest"
    fi
  fi
  msg "$BLUE" "Using image: ${IOWARP_DOCKER_IMAGE}"
}

wait_all_exit() {
  local timeout_s=${1:-480} start; start=$(date +%s)
  msg "$BLUE" "Waiting for all 4 nodes to finish (timeout ${timeout_s}s)..."
  while :; do
    local running=0
    for n in $NODES; do
      local st; st=$(docker inspect -f '{{.State.Running}}' "$(cname "$n")" 2>/dev/null || echo false)
      [ "$st" = "true" ] && running=$(( running + 1 ))
    done
    [ "$running" -eq 0 ] && return 0
    if [ $(( $(date +%s) - start )) -ge "$timeout_s" ]; then
      err "Timeout: $running node(s) still running"
      return 1
    fi
    sleep 2
  done
}

# The core assertion: each node must report exactly 3 passing coherency cases.
aggregate() {
  header "Per-node coherency results"
  local overall=0
  for n in $NODES; do
    local rank=$(( n - 1 )) c; c="$(cname "$n")"
    local code total passed failed log
    code=$(docker inspect -f '{{.State.ExitCode}}' "$c" 2>/dev/null || echo NA)
    log=$(docker logs "$c" 2>&1)
    total=$(echo "$log"  | grep -aoE "Total tests: [0-9]+" | tail -1 | grep -oE "[0-9]+")
    passed=$(echo "$log" | grep -aoE "Passed: [0-9]+"      | tail -1 | grep -oE "[0-9]+")
    failed=$(echo "$log" | grep -aoE "Failed: [0-9]+"      | tail -1 | grep -oE "[0-9]+")
    total=${total:-0}; passed=${passed:-0}; failed=${failed:-0}
    if [ "$code" = "0" ] && [ "$total" = "3" ] && [ "$passed" = "3" ] && [ "$failed" = "0" ]; then
      ok "node$n (rank $rank): 3/3 coherency cases PASS"
    else
      err "node$n (rank $rank): FAIL — exit=$code total=$total passed=$passed failed=$failed"
      echo "----- last 40 lines of node$n log -----"
      echo "$log" | tail -40
      echo "----------------------------------------"
      overall=1
    fi
  done
  return $overall
}

main() {
  header "CTE Coherency 4-Node Integration Test"
  check_prereqs
  [ "$do_cleanup" = true ] || [ "$cleanup_only" = true ] && cleanup
  if [ "$cleanup_only" = true ]; then ok "Cleanup complete"; exit 0; fi

  # Match host file ownership by default; allow override (e.g. a host whose uid
  # has no matching entry in the container's /etc/passwd).
  export HOST_UID="${HOST_UID:-$(id -u)}"
  export HOST_GID="${HOST_GID:-$(id -g)}"
  detect_image

  header "Starting 4-node cluster"
  if ! docker compose up -d; then
    err "docker compose up failed"; docker compose logs | tail -50; cleanup; exit 1
  fi

  local rc=0
  wait_all_exit 480 || rc=1
  aggregate || rc=1

  header "Result"
  if [ "$rc" = 0 ]; then ok "ALL 4 NODES PASSED 3/3 COHERENCY CASES"; else err "COHERENCY TEST FAILED"; fi

  if [ "$keep" = true ]; then
    warn "Keeping containers (use 'docker compose down -v' to clean)"
  else
    docker compose down -v --remove-orphans >/dev/null 2>&1 || true
  fi
  exit $rc
}

main "$@"
