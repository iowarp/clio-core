#!/bin/bash
# Safe-BDev integration test runner.
#
#   ./run_tests.sh              interactive: bring the stack up and leave it
#                               running; open http://localhost:5000/safe_bdev,
#                               remove/replace a bdev, watch recovery.
#   MODE=smoke ./run_tests.sh   automated: bring up, scripted remove+replace,
#                               assert recovery advanced, tear down (CI).
#   ./run_tests.sh -c           cleanup only.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../" && pwd)"
if [ -n "${HOST_WORKSPACE:-}" ]; then
  export IOWARP_CORE_ROOT="${HOST_WORKSPACE}"
elif [ -z "${IOWARP_CORE_ROOT:-}" ]; then
  export IOWARP_CORE_ROOT="${REPO_ROOT}"
fi
cd "$SCRIPT_DIR"
MODE="${MODE:-interactive}"
export MODE

RED='\033[0;31m'; GREEN='\033[0;32m'; BLUE='\033[0;34m'; YELLOW='\033[1;33m'; NC='\033[0m'
msg() { echo -e "${1}${2}${NC}"; }

cleanup() { docker compose down -v >/dev/null 2>&1 || true; }

check_prereqs() {
  command -v docker >/dev/null || { msg "$RED" "docker not found"; exit 1; }
  docker compose version >/dev/null 2>&1 || { msg "$RED" "docker compose not found"; exit 1; }
  docker ps >/dev/null 2>&1 || { msg "$RED" "docker daemon not running"; exit 1; }
  [ -c /dev/fuse ] || { msg "$RED" "/dev/fuse not available"; exit 1; }
  # Verify built artifacts against the CONTAINER-LOCAL repo (REPO_ROOT), NOT
  # IOWARP_CORE_ROOT. IOWARP_CORE_ROOT is the DOCKER-DAEMON-visible path used for
  # the compose bind mount (e.g. the HOST path when running inside a devcontainer
  # against the host's docker socket); it need not exist on the local filesystem
  # where this script runs. The files themselves live under REPO_ROOT here.
  local fuse_bin="${REPO_ROOT}/build/bin/clio_cte_fuse"
  [ -x "$fuse_bin" ] || { msg "$RED" "clio_cte_fuse missing -- build with -DCLIO_CTE_ENABLE_FUSE_ADAPTER=ON (looked in ${REPO_ROOT}/build/bin)"; exit 1; }
  [ -f "${REPO_ROOT}/build/bin/clio_runtime_ext"*.so ] 2>/dev/null || \
    ls "${REPO_ROOT}/build/bin/clio_runtime_ext"*.so >/dev/null 2>&1 || \
    { msg "$YELLOW" "warning: clio_runtime_ext python module not found (dashboard controls need it)"; }
  msg "$GREEN" "prerequisites OK"
}

[ "${1:-}" = "-c" ] && { cleanup; msg "$GREEN" "cleaned up"; exit 0; }

msg "$BLUE" "=== Safe-BDev integration test (MODE=$MODE) ==="
check_prereqs
cleanup
export HOST_UID=$(id -u); export HOST_GID=$(id -g)

if [ "$MODE" = "smoke" ]; then
  trap cleanup EXIT
  docker compose up -d
  msg "$BLUE" "smoke test running (compose up); waiting for completion..."
  rc=$(docker wait safe-bdev-test 2>/dev/null || echo 1)
  msg "$BLUE" "=== container logs (tail) ==="
  docker logs safe-bdev-test 2>&1 | tail -50
  if [ "$rc" = "0" ]; then msg "$GREEN" "✓ SMOKE PASSED"; else msg "$RED" "✗ smoke failed (rc=$rc)"; fi
  exit "$rc"
fi

# interactive
docker compose up -d
msg "$GREEN" "Stack starting. Streaming logs (Ctrl-C detaches; stack keeps running)."
msg "$BLUE"  "Dashboard: http://localhost:5000/safe_bdev"
msg "$YELLOW" "Tear down with: ./run_tests.sh -c"
docker logs -f safe-bdev-test
