#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Use BIN_DIR from environment, or fall back to /workspace/build/bin
BIN_DIR="${BIN_DIR:-/workspace/build/bin}"
COMPOSE_CONFIG="${SCRIPT_DIR}/test_restart_compose.yaml"
CONF_DIR="/tmp/clio_run_restart_test"

echo "=== CTE Restart Integration Test ==="
echo "BIN_DIR: $BIN_DIR"
echo "COMPOSE_CONFIG: $COMPOSE_CONFIG"

# Stop the runtime and WAIT FOR IT TO ACTUALLY EXIT.
#
# Same fix as the replication_persist driver: a flat `sleep 2` then SIGKILL
# cuts the runtime off 3s BEFORE its own 5000 ms graceful-teardown budget, and
# teardown is where persisted state is flushed. This test has not been failing,
# but it restarts the runtime and reads back what survived -- exactly the shape
# that lost blobs in cte_replication_persist_integration on macOS, where the
# metadata segment is file-backed (no memfd) and teardown does real disk I/O.
STOP_TIMEOUT_S=${STOP_TIMEOUT_S:-30}
stop_runtime() {
    if [ -n "$RUNTIME_PID" ] && kill -0 $RUNTIME_PID 2>/dev/null; then
        $BIN_DIR/clio_run runtime stop 2>/dev/null || true
        waited=0
        while kill -0 $RUNTIME_PID 2>/dev/null; do
            if [ "$waited" -ge "$STOP_TIMEOUT_S" ]; then
                echo "WARNING: runtime did not exit within ${STOP_TIMEOUT_S}s; sending SIGKILL" >&2
                kill -9 $RUNTIME_PID 2>/dev/null || true
                break
            fi
            sleep 1
            waited=$((waited + 1))
        done
        wait $RUNTIME_PID 2>/dev/null || true
    fi
    RUNTIME_PID=""
}

# Cleanup function
cleanup() {
    stop_runtime
    sleep 1
    rm -f /dev/shm/clio_*
    rm -rf "$CONF_DIR"
    rm -rf /tmp/cte_restart_ram
}
trap cleanup EXIT

# Clean slate
rm -f /dev/shm/clio_*
rm -rf "$CONF_DIR"
rm -rf /tmp/cte_restart_ram

# === Phase 1: Start runtime, compose, put blobs, flush ===
echo ""
echo "=== Phase 1: Start runtime and store blobs ==="

export CLIO_SERVER_CONF="$COMPOSE_CONFIG"
$BIN_DIR/clio_run runtime start &
RUNTIME_PID=$!
sleep 3

echo "Runtime started (PID=$RUNTIME_PID), composing pools..."
$BIN_DIR/clio_run compose "$COMPOSE_CONFIG"

echo "Putting blobs and flushing..."
$BIN_DIR/test_restart --put-blobs

echo "Stopping runtime..."
stop_runtime
sleep 1

# Clear SHM but keep persistent files
rm -f /dev/shm/clio_*

echo "Phase 1 complete. Persistent files in $CONF_DIR:"
ls -la "$CONF_DIR"/restart/ 2>/dev/null || echo "  (no restart dir yet)"

# === Phase 2: Restart runtime (no compose), verify blobs ===
echo ""
echo "=== Phase 2: Restart runtime and verify blobs ==="

# Start runtime again with same conf_dir so it can find restart configs
$BIN_DIR/clio_run runtime start &
RUNTIME_PID=$!
sleep 3

echo "Runtime restarted (PID=$RUNTIME_PID), calling RestartContainers + verify..."
$BIN_DIR/test_restart --verify-blobs

echo "Stopping runtime..."
stop_runtime

echo ""
echo "=== RESTART TEST PASSED ==="
