#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Cross-platform FUSE mount smoke test (Linux libfuse3 / macOS macFUSE).
#
# Brings up the Clio runtime, composes the CTE pool, mounts the clio_cte_fuse
# daemon, writes+reads+verifies a file through the mount, then unmounts and
# tears the runtime down. Exercises the real FUSE backend end-to-end -- the
# unit tests (test_fuse_adapter) only cover the CTE helper functions.
#
# Usage:  CI/fuse_mount_smoke.sh <build-dir>
#   <build-dir>  CMake binary dir whose bin/ holds clio_run + clio_cte_fuse.
#
# Requires the FUSE adapter to have actually built (clio_cte_fuse present);
# if the platform's FUSE backend was not found at configure time the adapter
# is skipped and this script fails fast with a clear message.
# ---------------------------------------------------------------------------
set -euo pipefail

BUILD_DIR="${1:?usage: fuse_mount_smoke.sh <build-dir>}"
BIN="$(cd "$BUILD_DIR" && pwd)/bin"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG_DIR="$SRC_DIR/context-transfer-engine/test/integration/fuse-manual"

case "$(uname -s)" in
  Darwin) IS_MAC=1 ;;
  *)      IS_MAC=0 ;;
esac

MOUNT_POINT="$(mktemp -d 2>/dev/null || mktemp -d -t cte_fuse)"
RUNTIME_PID=""
FUSE_PID=""

export PATH="$BIN:$PATH"
export LD_LIBRARY_PATH="$BIN:${LD_LIBRARY_PATH:-}"
export DYLD_LIBRARY_PATH="$BIN:${DYLD_LIBRARY_PATH:-}"
export CLIO_SERVER_CONF="$CFG_DIR/cte_config.yaml"
export CLIO_BIND_ADDR=127.0.0.1

info() { echo "[smoke] $*"; }

is_mounted() {
  if [ "$IS_MAC" = 1 ]; then
    mount | grep -q " ${MOUNT_POINT} "
  else
    mountpoint -q "$MOUNT_POINT" 2>/dev/null
  fi
}

cleanup() {
  local rc=$?
  if is_mounted; then
    info "unmounting $MOUNT_POINT"
    if [ "$IS_MAC" = 1 ]; then
      umount "$MOUNT_POINT" 2>/dev/null || diskutil unmount force "$MOUNT_POINT" 2>/dev/null || true
    else
      fusermount3 -u "$MOUNT_POINT" 2>/dev/null || fusermount -u "$MOUNT_POINT" 2>/dev/null || true
    fi
    sleep 1
  fi
  [ -n "$FUSE_PID" ]    && kill "$FUSE_PID"    2>/dev/null || true
  [ -n "$RUNTIME_PID" ] && kill "$RUNTIME_PID" 2>/dev/null || true
  wait 2>/dev/null || true
  rmdir "$MOUNT_POINT" 2>/dev/null || true
  exit $rc
}
trap cleanup EXIT

# --- Preflight --------------------------------------------------------------
[ -x "$BIN/clio_run" ] || { echo "[smoke] ERROR: $BIN/clio_run not found"; exit 1; }
if [ ! -x "$BIN/clio_cte_fuse" ]; then
  echo "[smoke] ERROR: $BIN/clio_cte_fuse not found."
  echo "[smoke] The FUSE backend (libfuse3 / macFUSE / WinFsp) was not detected"
  echo "[smoke] at configure time, so the adapter binary was skipped."
  exit 1
fi

# --- Ensure a clean runtime slot --------------------------------------------
# A prior adapter test in this CI job can leak a Clio runtime (embedded in a
# test binary) that stays bound to the runtime port; our `runtime start` below
# would then die with "Address already in use". Best-effort clear any leftover
# and wait for the port to free -- bounded so we never hang.
RUNTIME_PORT="${CLIO_PORT:-9413}"
port_busy() {
  if command -v ss >/dev/null 2>&1; then
    ss -ltn 2>/dev/null | grep -q ":${RUNTIME_PORT}[[:space:]]"
  elif command -v lsof >/dev/null 2>&1; then
    lsof -iTCP:"${RUNTIME_PORT}" -sTCP:LISTEN >/dev/null 2>&1
  else
    return 1  # no probe available -> assume free
  fi
}
if port_busy; then
  info "runtime port ${RUNTIME_PORT} busy (leftover runtime); clearing"
  clio_run runtime stop >/dev/null 2>&1 || true
  for _ in $(seq 1 15); do port_busy || break; sleep 1; done
  if port_busy; then
    # Graceful stop did not free it: force-kill whatever still holds the port.
    if command -v lsof >/dev/null 2>&1; then
      lsof -ti tcp:"${RUNTIME_PORT}" 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    fi
    command -v fuser >/dev/null 2>&1 && fuser -k "${RUNTIME_PORT}/tcp" >/dev/null 2>&1 || true
    pkill -9 -f clio_run >/dev/null 2>&1 || true
    for _ in $(seq 1 10); do port_busy || break; sleep 1; done
  fi
fi

# --- Start runtime ----------------------------------------------------------
info "starting Clio runtime"
clio_run runtime start &
RUNTIME_PID=$!
sleep 3
kill -0 "$RUNTIME_PID" 2>/dev/null || { echo "[smoke] ERROR: runtime died on startup"; exit 1; }

# --- Compose the CTE pool ---------------------------------------------------
info "composing CTE pool"
clio_run compose "$CFG_DIR/cte_compose.yaml"

# --- Mount ------------------------------------------------------------------
info "mounting clio_cte_fuse at $MOUNT_POINT"
CLIO_WITH_RUNTIME=0 CLIO_IPC_MODE=SHM clio_cte_fuse "$MOUNT_POINT" -f &
FUSE_PID=$!
for _ in $(seq 1 20); do
  is_mounted && break
  kill -0 "$FUSE_PID" 2>/dev/null || { echo "[smoke] ERROR: FUSE daemon exited before mount"; exit 1; }
  sleep 1
done
is_mounted || { echo "[smoke] ERROR: mount did not appear within 20s"; exit 1; }
info "mount is live"

# --- I/O round-trip ---------------------------------------------------------
PAYLOAD="hello context-transfer-engine fuse $(uname -s)"
echo "$PAYLOAD" > "$MOUNT_POINT/smoke.txt"
GOT="$(cat "$MOUNT_POINT/smoke.txt")"
if [ "$GOT" != "$PAYLOAD" ]; then
  echo "[smoke] ERROR: readback mismatch"
  echo "[smoke]   wrote: $PAYLOAD"
  echo "[smoke]   read:  $GOT"
  exit 1
fi
info "PASS: wrote and read back '$PAYLOAD' through the FUSE mount"
