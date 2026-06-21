#!/bin/bash
# Stop IOWarp runtime + FUSE filesystem

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_BIN="$REPO_ROOT/build/bin"
PID_DIR="$SCRIPT_DIR/.pids"

export PATH="$BUILD_BIN:$PATH"
export LD_LIBRARY_PATH="$BUILD_BIN:${LD_LIBRARY_PATH:-}"

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC} $1"; }
ok()    { echo -e "${GREEN}[OK]${NC}   $1"; }

# --- Unmount FUSE ------------------------------------------------------------

if [ -f "$PID_DIR/mount_point" ]; then
    MOUNT_POINT="$(cat "$PID_DIR/mount_point")"
    if mountpoint -q "$MOUNT_POINT" 2>/dev/null; then
        info "Unmounting $MOUNT_POINT ..."
        fusermount3 -u "$MOUNT_POINT" 2>/dev/null || true
        sleep 1
    fi
fi

# --- Kill FUSE daemon --------------------------------------------------------

if [ -f "$PID_DIR/fuse.pid" ]; then
    FUSE_PID="$(cat "$PID_DIR/fuse.pid")"
    if kill -0 "$FUSE_PID" 2>/dev/null; then
        info "Stopping FUSE daemon (PID $FUSE_PID)..."
        kill "$FUSE_PID" 2>/dev/null || true
        wait "$FUSE_PID" 2>/dev/null || true
    fi
    rm -f "$PID_DIR/fuse.pid"
    ok "FUSE daemon stopped"
fi

# --- Stop CLIO Runtime runtime ---------------------------------------------------

if [ -f "$PID_DIR/runtime.pid" ]; then
    RUNTIME_PID="$(cat "$PID_DIR/runtime.pid")"
    if kill -0 "$RUNTIME_PID" 2>/dev/null; then
        info "Stopping Clio runtime (PID $RUNTIME_PID)..."
        kill "$RUNTIME_PID" 2>/dev/null || true
        wait "$RUNTIME_PID" 2>/dev/null || true
    fi
    rm -f "$PID_DIR/runtime.pid"
    ok "Clio runtime stopped"
fi

# --- Cleanup -----------------------------------------------------------------

rm -f "$PID_DIR/mount_point"
rmdir "$PID_DIR" 2>/dev/null || true

if [ -n "$MOUNT_POINT" ] && [ -d "$MOUNT_POINT" ]; then
    rmdir "$MOUNT_POINT" 2>/dev/null || true
fi

ok "All cleaned up"
