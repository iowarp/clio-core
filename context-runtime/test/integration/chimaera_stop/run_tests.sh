#!/bin/bash
# Host driver for the distributed clio_run-stop CI test (issue #710).
#
# Brings up a 2-node docker-compose cluster, waits for node1 (which runs the
# Jarvis stop gate) to finish, and propagates its exit code. Node1's exit code
# is the test result: 0 iff `clio_run stop` killed the whole runtime on every
# node with zero stale artifacts.
#
# Usage:
#   ./run_tests.sh                                   # run the test
#   HOST_WORKSPACE=/host/path/to/workspace ./run_tests.sh   # devcontainers
#   ./run_tests.sh --keep                            # keep containers on exit
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../" && pwd)"

# Export workspace path for docker-compose (HOST_WORKSPACE > IOWARP_CORE_ROOT > repo root)
if [ -n "${HOST_WORKSPACE:-}" ]; then
    export IOWARP_CORE_ROOT="${HOST_WORKSPACE}"
elif [ -z "${IOWARP_CORE_ROOT:-}" ]; then
    export IOWARP_CORE_ROOT="${REPO_ROOT}"
fi

cd "$SCRIPT_DIR"

KEEP=0
[ "${1:-}" = "--keep" ] && KEEP=1

RED='\033[0;31m'; GREEN='\033[0;32m'; BLUE='\033[0;34m'; NC='\033[0m'
log() { echo -e "${BLUE}[stop-ci]${NC} $*"; }
ok()  { echo -e "${GREEN}[stop-ci]${NC} $*"; }
err() { echo -e "${RED}[stop-ci]${NC} $*"; }

cleanup() {
    if [ "$KEEP" = 1 ]; then
        log "keeping containers (--keep); 'docker compose down -v' to clean up"
        return
    fi
    log "tearing down cluster"
    docker compose down -v 2>/dev/null || true
    # Jarvis/SSH artifacts left in the mounted workspace, possibly root-owned.
    sudo rm -rf "$SCRIPT_DIR/.chimaera_ssh" "$SCRIPT_DIR/.jarvis-shared" 2>/dev/null \
        || rm -rf "$SCRIPT_DIR/.chimaera_ssh" "$SCRIPT_DIR/.jarvis-shared" 2>/dev/null || true
}
trap cleanup EXIT

# Auto-detect docker image: nvidia if the built clio_run links CUDA, else cpu.
if [ -z "${IOWARP_DOCKER_IMAGE:-}" ]; then
    CLIO_BIN="${IOWARP_CORE_ROOT:-/workspace}/build/bin/clio_run"
    if [ -f "$CLIO_BIN" ] && ldd "$CLIO_BIN" 2>/dev/null | grep -q "libcudart"; then
        export IOWARP_DOCKER_IMAGE="iowarp/deps-nvidia:latest"
    else
        export IOWARP_DOCKER_IMAGE="iowarp/deps-cpu:latest"
    fi
fi

export HOST_UID="$(id -u)" HOST_GID="$(id -g)"

log "starting 2-node cluster (image: $IOWARP_DOCKER_IMAGE)"
docker compose down -v 2>/dev/null || true
docker compose up -d

log "waiting for node1 to run the stop gate (timeout 300s)"
EXIT_CODE="$(timeout 300 docker wait chimaera-stop-node1 2>/dev/null || echo 1)"

echo "==================== node1 (gate) log ===================="
docker logs chimaera-stop-node1 2>&1 | tail -120
echo "==================== node2 (sshd) log ===================="
docker logs chimaera-stop-node2 2>&1 | tail -20
echo "=========================================================="

if [ "$EXIT_CODE" = "0" ]; then
    ok "PASS: clio_run stop consistently killed the whole runtime (no stale artifacts)"
else
    err "FAIL: distributed stop test exited $EXIT_CODE"
fi
exit "$EXIT_CODE"
