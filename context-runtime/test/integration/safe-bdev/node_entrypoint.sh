#!/bin/bash
# Safe-bdev interactive test entrypoint (runs INSIDE the container).
#
# Brings up the full stack on one node:
#   1. clio runtime (composes 3 data bdevs + safe0 + CTE + clio-fs)
#   2. adds a 4th bdev as PARITY (safe_bdev_add_bdev) -> tolerates 1 failure
#   3. mounts the clio-fs via FUSE at $MOUNT_POINT
#   4. serves the runtime's built-in dashboard on :5000
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
export PYTHONPATH="$BIN:${PYTHONPATH:-}"

log() { echo "[safe-bdev-test] $*"; }
fail() { echo "[safe-bdev-test] FAIL: $*" >&2; exit 1; }

# 1. Start the runtime (its built-in dashboard serves :5000 off-container).
export CLIO_VIZ_ENABLE=1 CLIO_VIZ_PORT=5000 CLIO_VIZ_BIND=0.0.0.0
log "starting clio runtime..."
CLIO_WITH_RUNTIME=1 "$BIN/clio_run" start &
RUNTIME_PID=$!
sleep 5
kill -0 "$RUNTIME_PID" 2>/dev/null || fail "runtime did not start"
log "runtime up (pid $RUNTIME_PID)"

# 2. Add the 4th device as parity so safe0 tolerates 1 failure (3 data + 1
#    par), through the dashboard's own add_member action (issue #990).
log "adding parity bdev /mnt/bdev3.dat via the dashboard ..."
curl -sf --max-time 60 -X POST \
  "http://127.0.0.1:5000/api/mod/clio_safe_bdev/350.0/add_member" \
  -d "member_name=/mnt/bdev3.dat&capacity=256MB&bdev_type=file&as_parity=1" \
  >/tmp/parity.json || { cat /tmp/parity.json 2>/dev/null; fail "parity add failed"; }
grep -q '"ok":true' /tmp/parity.json || { cat /tmp/parity.json; fail "parity add not ok"; }
log "safe0 now 3 data + 1 parity (tolerates 1 failure)"

# 3. Mount clio-fs via FUSE.
mkdir -p "$MOUNT_POINT"
log "mounting clio-fs at $MOUNT_POINT ..."
CLIO_WITH_RUNTIME=0 "$BIN/clio_cte_fuse" "$MOUNT_POINT" -f &
FUSE_PID=$!
sleep 3
mountpoint -q "$MOUNT_POINT" || fail "FUSE mount failed"
log "clio-fs mounted"

# 4. The runtime serves its own dashboard (issue #990); confirm it answers.
curl -sf --max-time 10 "http://127.0.0.1:5000/api/health" >/dev/null \
  || fail "built-in dashboard did not answer on :5000"
log "dashboard up: http://localhost:5000/viz/clio_safe_bdev/index.html"

if [ "$MODE" = "interactive" ]; then
  cat <<EOF

============================================================
  Safe-BDev interactive test is READY.

  Dashboard:  http://localhost:5000/viz/clio_safe_bdev/index.html
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
  trap 'kill $FUSE_PID $RUNTIME_PID 2>/dev/null; exit 0' TERM INT
  while kill -0 "$RUNTIME_PID" 2>/dev/null; do sleep 5; done
  exit 0
fi

# MODE=smoke : automated end-to-end check.
log "smoke: writing 32 MB through clio-fs ..."
dd if=/dev/urandom of="$MOUNT_POINT/smoke.bin" bs=1M count=32 2>/dev/null || fail "fs write failed"
sync
log "smoke: querying recovery stats through the dashboard ..."
curl -sf "http://localhost:5000/api/pools/350.0/monitor?query=stats&routing=local" \
  >/tmp/stats0.json || fail "stats endpoint unreachable"
grep -q "recovery_ops_total" /tmp/stats0.json || fail "stats missing recovery fields"

log "smoke: replace member 302.0 + recover via the dashboard ..."
curl -sf --max-time 60 -X POST \
  "http://localhost:5000/api/mod/clio_safe_bdev/350.0/replace_member" \
  -d "failed_pool_id=302.0&member_name=/mnt/bdev_repl.dat&capacity=256MB&bdev_type=file" \
  >/tmp/replace.json || { cat /tmp/replace.json 2>/dev/null; fail "replace call failed"; }
grep -q '"ok":true' /tmp/replace.json || { cat /tmp/replace.json; fail "replace not successful"; }

log "smoke: confirm recovery counters advanced ..."
sleep 3
curl -sf "http://localhost:5000/api/pools/350.0/monitor?query=stats&routing=local" >/tmp/stats1.json
python3 - <<'PY' || fail "recovery counters did not advance"
import json
results = json.load(open("/tmp/stats1.json")).get("results", {})
s = next((v for v in results.values() if v), {})
total = int(s.get("recovery_ops_total", 0))
done = int(s.get("recovery_ops_completed", 0))
print("[safe-bdev-test] recovery total=%d completed=%d" % (total, done))
raise SystemExit(0 if total > 0 and done >= total else 1)
PY

log "smoke: verifying data still readable after recovery ..."
dd if="$MOUNT_POINT/smoke.bin" of=/dev/null bs=1M 2>/dev/null || fail "fs read failed post-recovery"

log "SMOKE PASSED"
fusermount3 -u "$MOUNT_POINT" 2>/dev/null || true
kill "$FUSE_PID" "$RUNTIME_PID" 2>/dev/null || true
exit 0
