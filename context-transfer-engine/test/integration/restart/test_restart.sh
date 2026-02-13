#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Use BIN_DIR from environment, or fall back to /workspace/build/bin
BIN_DIR="${BIN_DIR:-/workspace/build/bin}"
COMPOSE_CONFIG="${SCRIPT_DIR}/test_restart_compose.yaml"
CONF_DIR="/tmp/chimaera_restart_test"

echo "=== CTE Restart Integration Test ==="
echo "BIN_DIR: $BIN_DIR"
echo "COMPOSE_CONFIG: $COMPOSE_CONFIG"

# Stop runtime helper: try chimaera_stop_runtime, fall back to kill
stop_runtime() {
    if [ -n "$RUNTIME_PID" ] && kill -0 $RUNTIME_PID 2>/dev/null; then
        $BIN_DIR/chimaera_stop_runtime 2>/dev/null || true
        # Give graceful shutdown a chance
        sleep 2
        # Force kill if still running
        if kill -0 $RUNTIME_PID 2>/dev/null; then
            kill -9 $RUNTIME_PID 2>/dev/null || true
        fi
        wait $RUNTIME_PID 2>/dev/null || true
    fi
    RUNTIME_PID=""
}

# Cleanup function
cleanup() {
    stop_runtime
    sleep 1
    rm -f /dev/shm/chimaera_*
    rm -rf "$CONF_DIR"
    rm -rf /tmp/cte_restart_ram
}
trap cleanup EXIT

# Clean slate
rm -f /dev/shm/chimaera_*
rm -rf "$CONF_DIR"
rm -rf /tmp/cte_restart_ram

# === Phase 1: Start runtime, compose, put blobs, flush ===
echo ""
echo "=== Phase 1: Start runtime and store blobs ==="

export CHI_SERVER_CONF="$COMPOSE_CONFIG"
$BIN_DIR/chimaera_start_runtime &
RUNTIME_PID=$!
sleep 3

echo "Runtime started (PID=$RUNTIME_PID), composing pools..."
$BIN_DIR/chimaera_compose "$COMPOSE_CONFIG"

echo "Putting blobs and flushing..."
$BIN_DIR/test_restart --put-blobs

echo "Stopping runtime..."
stop_runtime
sleep 1

# Clear SHM but keep persistent files
rm -f /dev/shm/chimaera_*

echo "Phase 1 complete. Persistent files in $CONF_DIR:"
ls -la "$CONF_DIR"/restart/ 2>/dev/null || echo "  (no restart dir yet)"

# === Phase 2: Restart runtime (no compose), verify blobs ===
echo ""
echo "=== Phase 2: Restart runtime and verify blobs ==="

# Start runtime again with same conf_dir so it can find restart configs
$BIN_DIR/chimaera_start_runtime &
RUNTIME_PID=$!
sleep 3

echo "Runtime restarted (PID=$RUNTIME_PID), calling RestartContainers + verify..."
$BIN_DIR/test_restart --verify-blobs

echo "Stopping runtime..."
stop_runtime

echo ""
echo "=== RESTART TEST PASSED ==="
