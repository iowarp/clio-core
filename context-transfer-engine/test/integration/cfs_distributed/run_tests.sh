#!/bin/bash
# Distributed CTE filesystem (CFS) correctness test runner (issue #685).
#
# Brings up a 2-node clio cluster in Docker, where node 1 writes a CFS file and
# node 2 must read it back cross-node. The authoritative result is the READER
# (node 2) exit code: a shared CFS namespace => reader exits 0; the pre-#685
# node-local bug => reader gets ENOENT and exits non-zero.
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

export HOST_UID=$(id -u)
export HOST_GID=$(id -g)

# Match the docker image to the built binary (CPU vs CUDA), mirroring the
# sibling distributed test's auto-detect.
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

cleanup() { docker compose down -v >/dev/null 2>&1 || true; }
trap cleanup EXIT

say "=== Cleaning any previous run ==="
docker compose down -v >/dev/null 2>&1 || true

say "=== Starting 2-node CFS cluster (writer + reader) ==="
if ! docker compose up -d; then
    err "docker compose up failed"
    docker compose logs || true
    exit 1
fi

# Both node containers run to completion (writer waits for the reader's ack,
# reader exits after it reads). Block on each and capture exit codes.
say "=== Waiting for reader (node 2) and writer (node 1) to finish ==="
reader_rc=$(docker wait cfs-dist-node2 2>/dev/null || echo 1)
writer_rc=$(docker wait cfs-dist-node1 2>/dev/null || echo 1)

say "=== Reader (node 2) log ==="
docker logs cfs-dist-node2 2>&1 | tail -40
say "=== Writer (node 1) log (tail) ==="
docker logs cfs-dist-node1 2>&1 | tail -20

echo ""
say "=== Result ==="
echo "writer(node1) exit=${writer_rc}  reader(node2) exit=${reader_rc}"

# The reader is the cross-node correctness assertion; the writer must also have
# succeeded at its own (same-node) operations for the reader result to be valid.
if [ "$reader_rc" = "0" ] && [ "$writer_rc" = "0" ]; then
    ok "CFS namespace is shared across nodes (issue #685 correctness holds)"
    exit 0
else
    err "Distributed CFS test FAILED (writer=${writer_rc}, reader=${reader_rc})."
    [ "$reader_rc" != "0" ] && err "  Reader could not read the writer's file cross-node — CFS namespace is node-local (#685)."
    exit 1
fi
