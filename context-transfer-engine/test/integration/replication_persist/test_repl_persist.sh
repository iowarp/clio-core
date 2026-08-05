#!/bin/bash
# Replication persistence stress (issue #886): 50 CachedPuts (DRAM primary +
# disk replica), reboot the runtime, verify every disk replica survived and
# the cache heals itself. Modeled on ../restart/test_restart.sh.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BIN_DIR="${BIN_DIR:-/workspace/build/bin}"
COMPOSE_CONFIG="${SCRIPT_DIR}/test_repl_persist_compose.yaml"
CONF_DIR="/tmp/clio_repl_persist_test"

echo "=== CTE Replication Persistence Test ==="
echo "BIN_DIR: $BIN_DIR"
echo "COMPOSE_CONFIG: $COMPOSE_CONFIG"

stop_runtime() {
    if [ -n "$RUNTIME_PID" ] && kill -0 $RUNTIME_PID 2>/dev/null; then
        $BIN_DIR/clio_run runtime stop 2>/dev/null || true
        sleep 2
        if kill -0 $RUNTIME_PID 2>/dev/null; then
            kill -9 $RUNTIME_PID 2>/dev/null || true
        fi
        wait $RUNTIME_PID 2>/dev/null || true
    fi
    RUNTIME_PID=""
}

cleanup() {
    stop_runtime
    sleep 1
    rm -f /dev/shm/clio_*
    rm -rf "$CONF_DIR"
}
trap cleanup EXIT

# Clean slate
rm -f /dev/shm/clio_*
rm -rf "$CONF_DIR"
mkdir -p "$CONF_DIR"

# === Phase 1: start runtime, compose, CachedPut 50 blobs, flush ===
echo ""
echo "=== Phase 1: Start runtime and cache blobs (DRAM + disk) ==="

export CLIO_SERVER_CONF="$COMPOSE_CONFIG"
$BIN_DIR/clio_run runtime start &
RUNTIME_PID=$!
sleep 3

echo "Runtime started (PID=$RUNTIME_PID), composing pools..."
$BIN_DIR/clio_run compose "$COMPOSE_CONFIG"

echo "CachedPut x50 through the replication chimod..."
$BIN_DIR/test_repl_persist --put-blobs

echo "Rebooting runtime..."
stop_runtime
sleep 1
rm -f /dev/shm/clio_*   # volatile state dies with the machine

# === Phase 2: reboot runtime, verify disk replicas ===
echo ""
echo "=== Phase 2: Reboot runtime and verify disk replicas ==="

$BIN_DIR/clio_run runtime start &
RUNTIME_PID=$!
sleep 3

echo "Runtime rebooted (PID=$RUNTIME_PID), verifying..."
$BIN_DIR/test_repl_persist --verify-blobs

echo ""
echo "=== Replication persistence test PASSED ==="
