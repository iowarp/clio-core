#!/bin/bash
# "Clio as a cache" multi-node test runner (issue #1015).
#
# Brings up a 2-node Docker cluster with NO clio daemon on either node, and
# launches `mpirun -np 8` across both containers (4 ranks per node). Every rank
# runs CLIO_INIT with CLIO_WITH_RUNTIME=1: one rank per node must start the
# runtime, the other three must attach to it, and all eight must be able to
# PutBlob/GetBlob through the CTE afterwards.
#
# The authoritative result is node 1's exit code (the mpirun launcher): the MPI
# job max-reduces every rank's result onto rank 0.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../" && pwd)"

if [ -n "${HOST_WORKSPACE:-}" ]; then
    export IOWARP_CORE_ROOT="${HOST_WORKSPACE}"
elif [ -z "${IOWARP_CORE_ROOT:-}" ]; then
    export IOWARP_CORE_ROOT="${REPO_ROOT}"
fi

cd "$SCRIPT_DIR"

RED='\033[0;31m'; GREEN='\033[0;32m'; BLUE='\033[0;34m'; NC='\033[0m'
say()  { echo -e "${BLUE}$*${NC}"; }
ok()   { echo -e "${GREEN}✓ $*${NC}"; }
err()  { echo -e "${RED}✗ $*${NC}"; }

command -v docker >/dev/null 2>&1 || { err "Docker not installed"; exit 1; }
docker compose version >/dev/null 2>&1 || { err "docker compose not available"; exit 1; }
docker ps >/dev/null 2>&1 || { err "Docker daemon not running"; exit 1; }

# Match the docker image to the built binary (CPU vs CUDA), mirroring the
# sibling distributed tests.
if [ -z "${IOWARP_DOCKER_IMAGE:-}" ]; then
    CLIO_BIN="${IOWARP_CORE_ROOT:-/workspace}/build/bin/clio_run"
    if [ -f "$CLIO_BIN" ] && ldd "$CLIO_BIN" 2>/dev/null | grep -q "libcudart"; then
        export IOWARP_DOCKER_IMAGE="iowarp/deps-nvidia:latest"
        _bindir="$(dirname "$CLIO_BIN")"
        ldd "$CLIO_BIN" 2>/dev/null | awk '/libcudart/{print $3}' | while read -r _lib; do
            { [ -n "$_lib" ] && [ -f "$_lib" ] && cp -Lu "$_lib" "$_bindir/" 2>/dev/null; } || true
        done
    else
        export IOWARP_DOCKER_IMAGE="iowarp/deps-cpu:latest"
    fi
fi
say "Using image: ${IOWARP_DOCKER_IMAGE}"

# Clear any stale SSH-exchange state from a previous run.
rm -rf "${SCRIPT_DIR}/.mpi_ssh" 2>/dev/null || sudo rm -rf "${SCRIPT_DIR}/.mpi_ssh" 2>/dev/null || true

cleanup() {
    docker compose down -v --remove-orphans >/dev/null 2>&1 || true
    rm -rf "${SCRIPT_DIR}/.mpi_ssh" 2>/dev/null || sudo rm -rf "${SCRIPT_DIR}/.mpi_ssh" 2>/dev/null || true
}
trap cleanup EXIT

say "=== Cleaning any previous run ==="
docker compose down -v --remove-orphans >/dev/null 2>&1 || true

say "=== Starting 2-node daemonless cluster (8 MPI ranks) ==="
if ! docker compose up -d; then
    err "docker compose up failed"
    docker compose logs || true
    exit 1
fi

say "=== Waiting for node 1 (mpirun launcher) to finish ==="
node1_rc=$(docker wait wr-node1 2>/dev/null || echo 1)

# The ranks' own lines first: a debug build's runtime logging is voluminous
# enough to push them out of any reasonable tail.
say "=== Rank results (per-rank init / owner / PASS-FAIL) ==="
docker logs wr-node1 2>&1 | grep -a 'with_runtime rank' || echo "(no rank output captured)"
say "=== Node 1 (launcher) log tail ==="
docker logs wr-node1 2>&1 | grep -av ' DEBUG ' | tail -40
say "=== Node 2 log (tail) ==="
docker logs wr-node2 2>&1 | grep -av ' DEBUG ' | tail -20

echo ""
say "=== Result ==="
echo "node1(launcher) exit=${node1_rc}"

if [ "$node1_rc" = "0" ]; then
    ok "CLIO_WITH_RUNTIME=1 attach-or-start PASSED (#1015)"
    exit 0
else
    err "CLIO_WITH_RUNTIME=1 attach-or-start FAILED (node1=${node1_rc})."
    err "  rc=2 a rank's CLIO_INIT failed; rc=3 a node had != 1 runtime owner;"
    err "  rc=6..9 a PutBlob/GetBlob round trip through the shared runtime failed."
    exit 1
fi
