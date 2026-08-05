#!/bin/bash
# 4-node cache-coherence suite runner (issue #886 distributed coherence).
#
# Brings up a 4-node clio cluster; EVERY node runs test_cache_coherence
# (put-once-read-many + segmented + fragmented scenarios, blob-based
# barriers). Success requires exit 0 from all four nodes.
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

say "=== Starting 4-node coherence cluster ==="
if ! docker compose up -d; then
    err "docker compose up failed"
    docker compose logs || true
    exit 1
fi

say "=== Waiting for all four nodes to finish ==="
overall=0
for n in 1 2 3 4; do
    rc=$(docker wait "coh-node$n" 2>/dev/null || echo 1)
    if [ "$rc" = "0" ]; then
        ok "node $n exit 0"
    else
        err "node $n exit $rc"
        overall=1
    fi
done

if [ "$overall" != "0" ]; then
    for n in 1 2 3 4; do
        say "=== node $n log (tail) ==="
        docker logs "coh-node$n" 2>&1 | tail -60
    done
    if [ -n "${COH_LOG_DIR:-}" ]; then
        # Full per-node logs for post-mortem (the tails above routinely miss
        # the interesting window; the EXIT trap destroys the containers).
        mkdir -p "$COH_LOG_DIR"
        for n in 1 2 3 4; do
            docker logs "coh-node$n" > "$COH_LOG_DIR/coh-node$n.log" 2>&1 || true
        done
        say "full node logs saved to $COH_LOG_DIR"
    fi
    err "cache coherence suite FAILED"
    exit 1
fi

say "=== node 1 log (tail) ==="
docker logs coh-node1 2>&1 | tail -25
ok "cache coherence suite PASSED on all 4 nodes"
exit 0
