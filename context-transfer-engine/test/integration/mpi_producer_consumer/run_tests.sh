#!/bin/bash
# Multi-node MPI producer/consumer test runner (issue #714).
#
# Brings up a 2-node clio cluster in Docker and launches `mpirun -np 2` across
# both containers (rank 0 producer on node 1, rank 1 consumer on node 2) for
# both the CTE-core and CFS producer/consumer binaries. The authoritative result
# is node 1's exit code (the mpirun launcher): the MPI job reduces every rank's
# result, so a cross-node data failure (#714) surfaces there as non-zero.
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

# Match the docker image to the built binary (CPU vs CUDA), mirroring the sibling
# distributed tests.
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
    docker compose down -v >/dev/null 2>&1 || true
    rm -rf "${SCRIPT_DIR}/.mpi_ssh" 2>/dev/null || sudo rm -rf "${SCRIPT_DIR}/.mpi_ssh" 2>/dev/null || true
}
trap cleanup EXIT

say "=== Cleaning any previous run ==="
docker compose down -v >/dev/null 2>&1 || true

say "=== Starting 2-node MPI producer/consumer cluster ==="
if ! docker compose up -d; then
    err "docker compose up failed"
    docker compose logs || true
    exit 1
fi

say "=== Waiting for node 1 (mpirun launcher) to finish ==="
node1_rc=$(docker wait pc-node1 2>/dev/null || echo 1)

say "=== Node 1 (launcher) log ==="
docker logs pc-node1 2>&1 | tail -60
say "=== Node 2 (consumer host) log (tail) ==="
docker logs pc-node2 2>&1 | tail -30

echo ""
say "=== Result ==="
echo "node1(launcher) exit=${node1_rc}"

if [ "$node1_rc" = "0" ]; then
    ok "Multi-node producer/consumer PASSED (cross-node blob data works, #714)"
    exit 0
else
    err "Multi-node producer/consumer FAILED (node1=${node1_rc})."
    err "  A cross-node GetBlob/Read returning 0 bytes means blob payload is node-local (#714)."
    exit 1
fi
