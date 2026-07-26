#!/bin/bash
# ---------------------------------------------------------------------------
# Per-node orchestration for the 4-node CTE coherency integration test.
#
# Unlike the `distributed` suite (test client on node1 only), EVERY node here
# runs test_cte_coherency as a client against its LOCAL daemon -- coherency
# means different nodes touching the same blobs, and blobs hash-route by
# (tag_id, blob_name) so a blob written on one node generally lives on another.
#
# Sequence per node:
#   1. start clio_run daemon (background)
#   2. wait for the daemon's ChiMod SCAN to finish (not just "server started")
#   3. READY barrier   -- no client runs until all 4 daemons are up + scanned
#   3b. rank 0 creates pool 512.0 (clio_run compose start) now that every peer
#       has loaded clio_bdev; COMPOSED barrier so all wait for the pool
#   4. run test_cte_coherency as a client (CLIO_WITH_RUNTIME=0)
#   5. TESTDONE barrier -- CRITICAL: do not tear down this node's daemon while
#      peers may still be reading blobs that hash-route to this node
#   6. shut the daemon down; exit with the test's own return code
#
# Env (from docker-compose): CTE_RANK, CTE_NUM_NODES, CTE_RUNDIR (shared volume),
# NODE_ID, CLIO_SERVER_CONF, CLIO_WITH_RUNTIME=0.
# ---------------------------------------------------------------------------
set -u
RANK="${CTE_RANK:?CTE_RANK must be set}"
N="${CTE_NUM_NODES:?CTE_NUM_NODES must be set}"
RUNDIR="${CTE_RUNDIR:?CTE_RUNDIR must be set}"
BIN="/workspace/build/bin"
tag() { echo "[rank $RANK]" "$@"; }

mkdir -p "$RUNDIR"

# File-based cross-node barrier over the shared volume. Distinct filename prefix
# ($1) from the test's own bar_* files, so they never collide.
barrier() {
  local name="$1" timeout_s="${2:-180}"
  echo "1" > "$RUNDIR/coh_${name}.$RANK"
  local deadline=$(( $(date +%s) + timeout_s ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    local seen=0 r=0
    while [ "$r" -lt "$N" ]; do
      [ -f "$RUNDIR/coh_${name}.$r" ] && seen=$(( seen + 1 ))
      r=$(( r + 1 ))
    done
    [ "$seen" -ge "$N" ] && return 0
    sleep 0.5
  done
  tag "BARRIER '$name' TIMEOUT (saw $seen/$N)"
  return 1
}

# --- 1. start the daemon ---------------------------------------------------
tag "starting clio_run daemon"
"$BIN/clio_run" runtime start > /tmp/daemon.log 2>&1 &
RUNTIME_PID=$!

# --- 2. wait for the daemon to be ready ------------------------------------
# CRITICAL: wait for the ChiMod SCAN, not just "Main server started". The scan
# (which registers clio_bdev / clio_cte_core) runs AFTER the server-started
# marker, so gating on that marker alone lets clients -- and the cross-node
# pool-512 creation -- begin before peers have loaded clio_bdev, which surfaces
# as "ChiMod 'clio_bdev' not found". "Loaded ChiMod: clio_cte_core" is the last
# module loaded and reliably marks the scan complete on THIS node.
ready=0
for _ in $(seq 1 120); do
  if grep -qa "Loaded ChiMod: clio_cte_core" /tmp/clio.log /tmp/daemon.log 2>/dev/null; then
    ready=1; break
  fi
  if ! kill -0 "$RUNTIME_PID" 2>/dev/null; then
    tag "daemon exited during startup"; cat /tmp/daemon.log; exit 3
  fi
  sleep 1
done
[ "$ready" = 1 ] && tag "daemon ready (ChiMods scanned)" || tag "ChiMod-scan marker not seen (continuing)"

# --- 3. READY barrier: all 4 daemons have scanned their ChiMods ------------
if ! barrier READY 120; then
  tag "READY barrier failed"; kill "$RUNTIME_PID" 2>/dev/null; exit 4
fi
export CLIO_WITH_RUNTIME=0

# --- 3b. create the CTE pool (512.0) explicitly, now that every daemon is up
# AND has scanned its ChiMods. The daemon-startup compose races cross-node
# bdev-create broadcasts against peers' still-in-progress ChiMod scans and can
# fail ("ChiMod 'clio_bdev' not found"), leaving pool 512.0 uncreated. The
# distributed suite hides this because test_core_functionality creates the pool
# in its own body; test_cte_coherency instead ASSUMES 512.0 exists. So rank 0
# composes it here, after the READY barrier guarantees all peers loaded
# clio_bdev. Idempotent: returns the existing pool if the startup attempt (or a
# single-node run) already created it. ---
if [ "$RANK" = "0" ]; then
  tag "composing CTE pool (clio_run compose start)"
  if "$BIN/clio_run" compose start "$CLIO_SERVER_CONF" > /tmp/compose.log 2>&1; then
    tag "compose start OK"
  else
    tag "compose start returned nonzero (see log)"
    sed "s/\x1b\[[0-9;]*m//g" /tmp/compose.log | tail -15
  fi
fi
if ! barrier COMPOSED 120; then
  tag "COMPOSED barrier failed"; kill "$RUNTIME_PID" 2>/dev/null; exit 5
fi

# --- 4. run the coherency test as a client ---------------------------------
tag "running test_cte_coherency (client)"
"$BIN/test_cte_coherency" 2>&1 | tee "$RUNDIR/result.$RANK.log"
TEST_RC=${PIPESTATUS[0]}
echo "$TEST_RC" > "$RUNDIR/exit.$RANK"
tag "test_cte_coherency exited rc=$TEST_RC"

# --- 5. TESTDONE barrier: keep serving until every client has finished ------
barrier TESTDONE 180 || tag "TESTDONE barrier timed out (proceeding to shutdown)"

# --- 6. shut down the daemon ------------------------------------------------
tag "stopping daemon"
kill "$RUNTIME_PID" 2>/dev/null
wait "$RUNTIME_PID" 2>/dev/null
exit "$TEST_RC"
