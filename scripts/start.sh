#!/usr/bin/env bash
#
# start.sh — bring up the IOWarp stack described by safe.yaml.
#
#   * starts the clio runtime, which composes the pools from safe.yaml
#       (4 file bdevs -> safe0 erasure-coded bdev -> CAE + CTE), then
#   * mounts the CTE FUSE filesystem.
#
# By default this is a FRESH session (clio_run start --ephemeral). resume.sh
# re-invokes this script with EPHEMERAL=0 to recover the persisted state
# instead (the bdev/safe-bdev alloc-logs + member superblocks).
#
# Prereqs: build with the FUSE adapter once:
#   cmake -B /workspace/build -DCLIO_CTE_ENABLE_FUSE_ADAPTER=ON
#   cmake --build /workspace/build -j"$(nproc)"
#
# Override any of these with environment variables.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"   # scripts/ lives directly under the repo root
BUILD_DIR="${BUILD_DIR:-$REPO_DIR/build}"
CONF="${CLIO_SERVER_CONF:-$SCRIPT_DIR/safe.yaml}"
BDEV_DIR="${BDEV_DIR:-$HOME/bdevs}"
MOUNT="${CTE_MOUNT:-$HOME/cte_mnt}"
EPHEMERAL="${EPHEMERAL:-1}"          # 1 = fresh start, 0 = resume/recover

BIN="$BUILD_DIR/bin"
export CLIO_SERVER_CONF="$CONF"
export CHI_REPO_PATH="$BIN"          # where the daemon finds module *_runtime.so
export LD_LIBRARY_PATH="$BIN:${LD_LIBRARY_PATH:-}"

# The bdev module itself opens each file bdev with O_CREAT and truncates it to
# the configured `capacity`, so we don't pre-create the .dat files here — we
# only ensure the directory (which also holds the alloc-log WALs) exists.
mkdir -p "$BDEV_DIR"

# 1) Runtime (composes safe.yaml's `compose:` section at startup).
START_FLAGS=()
[[ "$EPHEMERAL" == "1" ]] && START_FLAGS+=(--ephemeral)
echo "[start] launching clio runtime (${START_FLAGS[*]:-persistent}) with $CONF"
nohup "$BIN/clio_run" start "${START_FLAGS[@]}" \
  > "$BDEV_DIR/runtime.log" 2>&1 &
echo $! > "$BDEV_DIR/runtime.pid"

# Wait until compose finishes (or time out).
echo -n "[start] waiting for pools to compose"
for _ in $(seq 1 60); do
  if grep -qiE "pools created successfully|All .* pools created" \
       "$BDEV_DIR/runtime.log" 2>/dev/null; then
    echo " ... ok"; break
  fi
  if ! kill -0 "$(cat "$BDEV_DIR/runtime.pid")" 2>/dev/null; then
    echo; echo "[start] ERROR: runtime exited early — see $BDEV_DIR/runtime.log"; exit 1
  fi
  echo -n "."; sleep 0.5
done

# 2) CTE FUSE mount.
mkdir -p "$MOUNT"
echo "[start] mounting CTE FUSE at $MOUNT"
nohup "$BIN/clio_cte_fuse" "$MOUNT" -f \
  > "$BDEV_DIR/fuse.log" 2>&1 &
echo $! > "$BDEV_DIR/fuse.pid"
sleep 2

if mountpoint -q "$MOUNT" 2>/dev/null; then
  echo "[start] up. mount=$MOUNT  logs=$BDEV_DIR/{runtime,fuse}.log"
else
  echo "[start] runtime up; FUSE may not have mounted (needs /dev/fuse + privileges)."
  echo "        see $BDEV_DIR/fuse.log"
fi
