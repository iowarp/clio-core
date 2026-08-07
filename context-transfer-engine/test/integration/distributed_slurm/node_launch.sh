#!/bin/bash
# Per-node launcher for the 4-node Slurm CTE distributed tests.
#
# Run once per node by `srun --ntasks=N --ntasks-per-node=1`. Every rank starts a
# clio_run daemon. Two client modes:
#
#   CTE_CLIENT_MODE=rank0  (default) - only rank 0 runs the test client, the way
#       node1 does in the docker-compose suite. Peers just serve.
#   CTE_CLIENT_MODE=all              - EVERY rank runs the test client. Required
#       for coherency tests, where different NODES must touch the same blobs.
#
# Ranks synchronize through files on the shared filesystem ($RUNDIR):
#   READY.<rank>     - daemon on that rank is up
#   TESTDONE.<rank>  - that rank's client finished
#   EXIT.<rank>      - that rank's client exit code
#   DONE             - (rank0 mode) rank 0 finished; peers may stop serving
set -u

RANK="${SLURM_PROCID:?}"
NNODES="${CTE_NUM_NODES:?}"
RUNDIR="${RUNDIR:?}"
BIN="${CLIO_BIN_DIR:?}"
MODE="${CTE_CLIENT_MODE:-rank0}"
HOST="$(hostname -s)"

export PATH="$BIN:$PATH"
# CLIO_EXTRA_LDPATH carries extra runtime libs (e.g. cuSZp) the daemon must
# dlopen for the compressor; apptainer special-cases LD_LIBRARY_PATH, so set it
# explicitly here rather than relying on host inheritance.
export LD_LIBRARY_PATH="${CLIO_EXTRA_LDPATH:+$CLIO_EXTRA_LDPATH:}$BIN:${LD_LIBRARY_PATH:-}"
export CLIO_SERVER_CONF="$RUNDIR/clio_config.yaml"
export CTE_NUM_NODES="$NNODES"

for i in 1 2 3 4; do mkdir -p "$LOCAL_STORE/hdd$i"; done

log() { echo "[rank $RANK $HOST] $*"; }

# Direct evidence of whether blob data actually distributed.
# NOTE: the file bdev PRE-ALLOCATES its backing file to the target's capacity as
# a SPARSE file, so apparent size (find -printf '%s' / du --apparent-size) reads
# the same on every node whether or not any blob landed there. Count ALLOCATED
# blocks (%b, 512B units) -- that is what tracks bytes actually written.
storage_report() {
  local files alloc
  files=$(find "$LOCAL_STORE" -type f 2>/dev/null | wc -l)
  alloc=$(find "$LOCAL_STORE" -type f -printf '%b\n' 2>/dev/null \
          | awk '{s+=$1} END{printf "%d", s*512}')
  echo "rank=$RANK host=$HOST node_files=$files node_alloc_bytes=$alloc" \
      >"$RUNDIR/storage.$RANK"
  find "$LOCAL_STORE" -type f -printf '%b blocks\t%s apparent\t%p\n' 2>/dev/null \
      | sort -rn | head -12 >>"$RUNDIR/storage.$RANK"
  log "storage: files=$files allocated_bytes=$alloc"
}

# Wait until `count` files matching $RUNDIR/<prefix>.<0..N-1> exist.
wait_for_all() {
  local prefix="$1" timeout="${2:-180}" waited=0 n
  while true; do
    n=0
    for r in $(seq 0 $((NNODES-1))); do
      [ -f "$RUNDIR/$prefix.$r" ] && n=$((n+1))
    done
    [ "$n" -ge "$NNODES" ] && return 0
    ls "$RUNDIR"/FAILED.* >/dev/null 2>&1 && { log "peer FAILED"; return 1; }
    sleep 2; waited=$((waited+2))
    if [ "$waited" -ge "$timeout" ]; then
      log "TIMEOUT waiting for $prefix ($n/$NNODES)"; return 1
    fi
  done
}

run_client() {
  export CLIO_WITH_RUNTIME=0          # attach to the local daemon, don't spawn one
  export CTE_RANK="$RANK" CTE_RUNDIR="$RUNDIR"
  "$BIN/${CTE_TEST_BIN:-test_core_functionality}" ${TEST_FILTER:+"$TEST_FILTER"} \
      >"$RUNDIR/test.$RANK.log" 2>&1
  local rc=$?
  # A run where NOTHING executed is not a pass. The test binary exits 0 when a
  # filter matches no cases, which would otherwise report a green 4-node run
  # that proved nothing.
  local passed
  passed=$(grep -oE "^Passed: [0-9]+" "$RUNDIR/test.$RANK.log" 2>/dev/null \
           | grep -oE "[0-9]+" | tail -1)
  if [ "$rc" = "0" ] && [ "${passed:-0}" -eq 0 ]; then
    log "NO TESTS RAN (filter='${TEST_FILTER:-<none>}') -- treating as failure"
    rc=2
  fi
  echo "$rc" >"$RUNDIR/EXIT.$RANK"
  log "client finished rc=$rc (passed=${passed:-0})"
  return $rc
}

# Run the test binary itself in SERVER mode (it self-hosts the runtime and IS
# node 0 of the cluster). Used for GPU tests where the compressor must run
# in-process so device pointers are valid. Do NOT set CLIO_WITH_RUNTIME=0.
run_server_test() {
  unset CLIO_WITH_RUNTIME
  export CTE_RANK="$RANK" CTE_RUNDIR="$RUNDIR"
  "$BIN/${CTE_TEST_BIN}" ${TEST_FILTER:+"$TEST_FILTER"} \
      >"$RUNDIR/test.$RANK.log" 2>&1
  local rc=$?
  local passed
  passed=$(grep -oE "^Passed: [0-9]+" "$RUNDIR/test.$RANK.log" 2>/dev/null \
           | grep -oE "[0-9]+" | tail -1)
  if [ "$rc" = "0" ] && [ "${passed:-0}" -eq 0 ]; then
    log "NO TESTS RAN -- treating as failure"; rc=2
  fi
  echo "$rc" >"$RUNDIR/EXIT.$RANK"
  log "server-test finished rc=$rc (passed=${passed:-0})"
  return $rc
}

# server_rank0: rank 0 does NOT run a separate daemon -- the test binary is the
# node-0 server. Peers run daemons; rank 0 waits for them, runs the test, then
# releases them. (Peers must be up first so the compressor's hash-routed forward
# reaches their cte_core containers.)
if [ "$MODE" = "server_rank0" ] && [ "$RANK" = "0" ]; then
  log "server_rank0: waiting for $((NNODES-1)) peer daemon(s) before starting node-0 test-server"
  waited=0
  while true; do
    n=0
    for r in $(seq 1 $((NNODES-1))); do [ -f "$RUNDIR/READY.$r" ] && n=$((n+1)); done
    [ "$n" -ge "$((NNODES-1))" ] && break
    ls "$RUNDIR"/FAILED.* >/dev/null 2>&1 && { log "peer FAILED"; echo 1 >"$RUNDIR/EXIT.0"; touch "$RUNDIR/DONE"; exit 1; }
    sleep 2; waited=$((waited+2))
    [ "$waited" -ge "${CLIO_READY_TIMEOUT:-180}" ] && { log "TIMEOUT waiting peers"; echo 1 >"$RUNDIR/EXIT.0"; touch "$RUNDIR/DONE"; exit 1; }
  done
  log "peers ready -- launching node-0 test-server"
  rc=0; run_server_test || rc=$?
  sync; storage_report
  touch "$RUNDIR/DONE"
  exit "$rc"
fi

log "starting clio_run daemon (conf=$CLIO_SERVER_CONF, mode=$MODE)"
"$BIN/clio_run" runtime start >"$RUNDIR/daemon.$RANK.log" 2>&1 &
DPID=$!

sleep "${CLIO_DAEMON_WARMUP:-12}"
if ! kill -0 "$DPID" 2>/dev/null; then
  log "DAEMON DIED during warmup -- see daemon.$RANK.log"
  tail -30 "$RUNDIR/daemon.$RANK.log" || true
  touch "$RUNDIR/FAILED.$RANK"; exit 1
fi
touch "$RUNDIR/READY.$RANK"
log "daemon up (pid=$DPID)"

rc=0
if [ "$MODE" = "server_rank0" ]; then
  # Peer (rank>0): rank 0 has no daemon, so don't wait for READY.0 -- just serve
  # until rank 0's test-server signals DONE.
  waited=0
  while [ ! -f "$RUNDIR/DONE" ]; do
    sleep 2; waited=$((waited+2))
    [ "$waited" -ge "${CLIO_SERVE_TIMEOUT:-900}" ] && { log "TIMEOUT for DONE"; break; }
    kill -0 "$DPID" 2>/dev/null || { log "daemon exited early"; break; }
  done
  sync; storage_report
  log "shutting down daemon"
  kill "$DPID" 2>/dev/null; wait "$DPID" 2>/dev/null
  exit 0
fi

if ! wait_for_all READY "${CLIO_READY_TIMEOUT:-180}"; then
  touch "$RUNDIR/FAILED.$RANK"; echo 1 >"$RUNDIR/EXIT.$RANK"
  touch "$RUNDIR/TESTDONE.$RANK" "$RUNDIR/DONE"
  kill "$DPID" 2>/dev/null; exit 1
fi
log "all $NNODES daemons ready"

if [ "$MODE" = "all" ]; then
  # Every rank is a client -- required for coherency.
  run_client || rc=$?
  touch "$RUNDIR/TESTDONE.$RANK"
  # A rank MUST NOT kill its daemon while a peer is still reading: blobs
  # hash-route across nodes, so node N's daemon serves blobs other ranks need.
  wait_for_all TESTDONE "${CLIO_SERVE_TIMEOUT:-900}" || true
  touch "$RUNDIR/DONE"
else
  if [ "$RANK" = "0" ]; then
    run_client || rc=$?
    touch "$RUNDIR/TESTDONE.$RANK"
    touch "$RUNDIR/DONE"
  else
    waited=0
    while [ ! -f "$RUNDIR/DONE" ]; do
      sleep 2; waited=$((waited+2))
      [ "$waited" -ge "${CLIO_SERVE_TIMEOUT:-900}" ] && { log "TIMEOUT for DONE"; break; }
      kill -0 "$DPID" 2>/dev/null || { log "daemon exited early"; break; }
    done
  fi
fi

sync; storage_report
log "shutting down daemon"
kill "$DPID" 2>/dev/null; wait "$DPID" 2>/dev/null
exit "$rc"
