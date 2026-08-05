#!/bin/bash
# Indexer restart integration test (issue #905): prove the SemanticSearch
# index is rebuilt from storage after a daemon reboot.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Use BIN_DIR from environment, or fall back to /workspace/build/bin
BIN_DIR="${BIN_DIR:-/workspace/build/bin}"
COMPOSE_CONFIG="${SCRIPT_DIR}/test_indexer_restart_compose.yaml"
CONF_DIR="/tmp/clio_indexer_restart_test"

echo "=== Indexer Restart Integration Test (issue #905) ==="
echo "BIN_DIR: $BIN_DIR"
echo "COMPOSE_CONFIG: $COMPOSE_CONFIG"

stop_runtime() {
    if [ -n "$RUNTIME_PID" ] && kill -0 $RUNTIME_PID 2>/dev/null; then
        $BIN_DIR/clio_run stop --grace-period 2000 2>/dev/null || true
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

export CLIO_SERVER_CONF="$COMPOSE_CONFIG"
# Bind the test client to the indexer pool (the interposition entrypoint).
export CLIO_CTE_POOL="564.0"
# Isolate the restart-log WAL: compose registration must not pollute the
# shared ~/.clio/restart_log.bin (same discipline as the CTE WAL test).
export CLIO_RESTART_LOG="$CONF_DIR/restart_log.bin"

# === Phase 1: start runtime, compose, put docs, verify live index, flush ===
echo ""
echo "=== Phase 1: store documents, verify inline index ==="

$BIN_DIR/clio_run runtime start &
RUNTIME_PID=$!
sleep 3

echo "Runtime started (PID=$RUNTIME_PID), composing pools..."
$BIN_DIR/clio_run compose "$COMPOSE_CONFIG"

$BIN_DIR/test_indexer_restart --put

echo "Stopping runtime..."
stop_runtime
sleep 1
rm -f /dev/shm/clio_*

echo "Phase 1 complete. Persistent state in $CONF_DIR:"
ls -la "$CONF_DIR" 2>/dev/null || true

# === Phase 2: restart runtime, re-compose (Restart path), verify searches ===
echo ""
echo "=== Phase 2: restart, rebuild index from storage, verify ==="

$BIN_DIR/clio_run runtime start &
RUNTIME_PID=$!
sleep 3

echo "Runtime restarted (PID=$RUNTIME_PID), re-composing (restart path)..."
$BIN_DIR/clio_run compose "$COMPOSE_CONFIG"

$BIN_DIR/test_indexer_restart --verify

echo "Stopping runtime..."
stop_runtime

echo ""
echo "=== INDEXER RESTART TEST PASSED ==="
