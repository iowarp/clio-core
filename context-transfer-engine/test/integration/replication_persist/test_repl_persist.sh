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

# Stop the runtime and WAIT FOR IT TO ACTUALLY EXIT.
#
# The old version slept 2s and then SIGKILLed, but the runtime asks for a
# 5000 ms grace period ("RequestStop: graceful stop requested (grace period:
# 5000 ms); main loop will run the full teardown") -- so the harness killed it
# 3s BEFORE its own teardown budget expired. Teardown is where cached blobs'
# disk replicas are flushed, so a SIGKILL mid-flush can lose whatever had not
# landed yet. Waiting is correct on its own terms: a harness deadline shorter
# than the runtime's own grace period is a bug in the harness whatever else is
# going on.
#
# CORRECTION: this was originally committed as the cause of macOS's
# "blob N replica lost", with the varying N read as a truncated flush. It was
# NOT. CI still failed with the wait in place, and the real cause turned out to
# be WAL replay order -- blob_txn_logs_ is sharded by WORKER and replayed in
# shard-INDEX order, so a blob's kExtendReplica could be replayed before the
# kCreateNewBlob that made the blob exist and was then dropped in silence. The
# varying N was which worker served which blob, not how far a flush got. Kept
# because the deadline mismatch is real; do not read it as the fix.
#
# Poll for exit up to STOP_TIMEOUT_S (comfortably above the grace period)
# before escalating; SIGKILL stays as a true last resort so a genuinely wedged
# runtime still cannot hang the suite.
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
