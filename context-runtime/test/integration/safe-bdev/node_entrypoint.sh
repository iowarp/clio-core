#!/bin/bash
# Safe-bdev interactive test entrypoint (runs INSIDE the container).
#
# Brings up the full stack on one node:
#   1. clio runtime (composes 3 data bdevs + safe0 + CTE + clio-fs)
#   2. adds a 4th bdev as PARITY (safe_bdev_add_bdev) -> tolerates 1 failure
#   3. mounts the clio-fs via FUSE at $MOUNT_POINT
#   4. launches the context-visualizer dashboard on :5000
#
# MODE=interactive (default): keep everything running so the operator can
#   remove/replace a bdev from the dashboard and watch recovery.
# MODE=smoke: verify the stack came up, do a small FUSE write/read, run a
#   scripted remove+replace and assert recovery progressed, then exit (for CI).

set -u
BIN=/workspace/build/bin
CONF=/workspace/context-runtime/test/integration/safe-bdev/clio_conf.yaml
MOUNT_POINT=/mnt/clio_fs
MODE="${MODE:-interactive}"
export CLIO_SERVER_CONF="$CONF"
export PATH="$BIN:$PATH"
export LD_LIBRARY_PATH="$BIN:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/workspace/context-visualizer:$BIN:${PYTHONPATH:-}"

log() { echo "[safe-bdev-test] $*"; }
fail() { echo "[safe-bdev-test] FAIL: $*" >&2; exit 1; }

# 1. Start the runtime.
log "starting clio runtime..."
CLIO_WITH_RUNTIME=1 "$BIN/clio_run" start &
RUNTIME_PID=$!
sleep 5
kill -0 "$RUNTIME_PID" 2>/dev/null || fail "runtime did not start"
log "runtime up (pid $RUNTIME_PID)"

# 2. Add the 4th device as parity so safe0 tolerates 1 failure (3 data + 1 par).
log "adding parity bdev /mnt/bdev3.dat ..."
python3 - <<'PY' || fail "parity add failed"
import sys
sys.path.insert(0, "/workspace/build/bin")
import clio_runtime_ext as m
assert m.clio_init(0), "client init failed"
pid = m.safe_bdev_add_bdev("350.0", "/mnt/bdev3.dat", "256MB", 0, 1)
m.clio_finalize()
print("[safe-bdev-test] parity member pool:", pid or "(FAILED)")
sys.exit(0 if pid else 1)
PY
log "safe0 now 3 data + 1 parity (tolerates 1 failure)"

# 3. Mount clio-fs via FUSE.
mkdir -p "$MOUNT_POINT"
log "mounting clio-fs at $MOUNT_POINT ..."
CLIO_WITH_RUNTIME=0 "$BIN/clio_cte_fuse" "$MOUNT_POINT" -f &
FUSE_PID=$!
sleep 3
mountpoint -q "$MOUNT_POINT" || fail "FUSE mount failed"
log "clio-fs mounted"

# 4. Launch the context-visualizer dashboard (install its deps if the image
#    lacks them).
python3 -c "import flask, yaml, msgpack" 2>/dev/null || {
  log "installing dashboard deps (flask/pyyaml/msgpack) ..."
  pip install --quiet --disable-pip-version-check flask pyyaml msgpack 2>&1 | tail -2 || true
}
log "launching context-visualizer on :5000 ..."
python3 -m context_visualizer --host 0.0.0.0 --port 5000 >/tmp/visualizer.log 2>&1 &
DASH_PID=$!
sleep 3
kill -0 "$DASH_PID" 2>/dev/null || { cat /tmp/visualizer.log; fail "visualizer did not start"; }
log "dashboard up: http://localhost:5000/safe_bdev"

if [ "$MODE" = "interactive" ]; then
  cat <<EOF

============================================================
  Safe-BDev interactive test is READY.

  Dashboard:  http://localhost:5000/safe_bdev
  FUSE mount: $MOUNT_POINT   (write files here through clio-fs)

  Try it:
    1. Write some data:  dd if=/dev/urandom of=$MOUNT_POINT/f bs=1M count=64
    2. In the dashboard, click "Remove" on a data member (e.g. 302.0),
       then "Replace + recover" and give a path like /mnt/bdev_new.dat.
    3. Watch the Recovery panel: ops in flight vs remaining.

  Ctrl-C (or 'docker compose down -v') to tear down.
============================================================
EOF
  # Idle until signalled so the operator can interact.
  trap 'kill $DASH_PID $FUSE_PID $RUNTIME_PID 2>/dev/null; exit 0' TERM INT
  while kill -0 "$RUNTIME_PID" 2>/dev/null; do sleep 5; done
  exit 0
fi

# MODE=smoke : automated end-to-end check.
log "smoke: writing 32 MB through clio-fs ..."
dd if=/dev/urandom of="$MOUNT_POINT/smoke.bin" bs=1M count=32 2>/dev/null || fail "fs write failed"
sync
log "smoke: querying recovery stats endpoint ..."
curl -sf "http://localhost:5000/api/safe_bdev/350.0/stats" >/tmp/stats0.json \
  || fail "stats endpoint unreachable"
grep -q "recovery_ops_total" /tmp/stats0.json || fail "stats missing recovery fields"

log "smoke: remove member 302.0 then replace + recover ..."
curl -sf -X POST "http://localhost:5000/api/safe_bdev/350.0/replace_member" \
  -H 'Content-Type: application/json' \
  -d '{"failed_pool_id":"302.0","member_path":"/mnt/bdev_repl.dat","capacity":"256MB","node_id":0}' \
  >/tmp/replace.json || fail "replace call failed"
grep -q '"success": true' /tmp/replace.json || { cat /tmp/replace.json; fail "replace not successful"; }

log "smoke: confirm recovery counters advanced ..."
curl -sf "http://localhost:5000/api/safe_bdev/350.0/stats" >/tmp/stats1.json
python3 - <<'PY' || fail "recovery counters did not advance"
import json
s = json.load(open("/tmp/stats1.json")).get("stats", {})
total = int(s.get("recovery_ops_total", 0))
done = int(s.get("recovery_ops_completed", 0))
print("[safe-bdev-test] recovery total=%d completed=%d" % (total, done))
raise SystemExit(0 if total > 0 and done >= total else 1)
PY

log "smoke: verifying data still readable after recovery ..."
dd if="$MOUNT_POINT/smoke.bin" of=/dev/null bs=1M 2>/dev/null || fail "fs read failed post-recovery"

log "SMOKE PASSED"
fusermount3 -u "$MOUNT_POINT" 2>/dev/null || true
kill "$DASH_PID" "$FUSE_PID" "$RUNTIME_PID" 2>/dev/null || true
exit 0
