#!/usr/bin/env bash
#
# stop.sh — tear the stack down: unmount the CTE FUSE filesystem, then stop the
# clio runtime. On-disk state (bdev files, alloc-logs, superblocks) is left
# intact so resume.sh can bring it back.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${BUILD_DIR:-$REPO_DIR/build}"
BDEV_DIR="${BDEV_DIR:-$HOME/bdevs}"
MOUNT="${CTE_MOUNT:-$HOME/cte_mnt}"

BIN="$BUILD_DIR/bin"
export CLIO_SERVER_CONF="${CLIO_SERVER_CONF:-$SCRIPT_DIR/safe.yaml}"
export CHI_REPO_PATH="$BIN"
export LD_LIBRARY_PATH="$BIN:${LD_LIBRARY_PATH:-}"

# 1) Unmount FUSE.
echo "[stop] unmounting CTE FUSE at $MOUNT"
fusermount3 -u "$MOUNT" 2>/dev/null \
  || fusermount -u "$MOUNT" 2>/dev/null \
  || umount "$MOUNT" 2>/dev/null \
  || true
if [[ -f "$BDEV_DIR/fuse.pid" ]]; then
  kill "$(cat "$BDEV_DIR/fuse.pid")" 2>/dev/null || true
  rm -f "$BDEV_DIR/fuse.pid"
fi
pkill -f clio_cte_fuse 2>/dev/null || true

# 2) Stop the runtime (graceful, then fall back to the recorded pid).
echo "[stop] stopping clio runtime"
"$BIN/clio_run" stop 2>/dev/null || true
if [[ -f "$BDEV_DIR/runtime.pid" ]]; then
  pid="$(cat "$BDEV_DIR/runtime.pid")"
  for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
  kill "$pid" 2>/dev/null || true
  rm -f "$BDEV_DIR/runtime.pid"
fi

echo "[stop] done."
