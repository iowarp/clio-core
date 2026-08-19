#!/bin/bash
# Collective-restart reproducer (issue #628).
#
# Demonstrates that a cross-node collective must not hang forever when a
# participant is restarted mid-flight and never declared dead by SWIM.
#
#   1. Start a 4-node cluster (SWIM deliberately slow -- see clio_conf.yaml).
#   2. [setup] create a MOD_NAME pool (containers on all nodes).
#   3. [hang]  broadcast a long-hold CoMutexTest from node1 (background).
#   4. When node1 reports the broadcast is dispatched, restart node4.
#   5. Wait up to HANG_TIMEOUT for the broadcast to return.
#        returns  -> PASS (fix present: QueryTaskProgress completed it)
#        no return -> FAIL, hang reproduced (pre-fix behaviour)
#
# Requires the `clio-bv` docker volume to hold a build (bin/clio_run and
# bin/clio_run_collective_restart_tests). On a non-Linux host, pass the repo's
# host path via HOST_WORKSPACE so Docker can bind-mount it.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../" && pwd)"

export IOWARP_DOCKER_IMAGE="${IOWARP_DOCKER_IMAGE:-iowarp/deps-nvidia:latest}"
export HOST_WORKSPACE="${HOST_WORKSPACE:-$REPO_ROOT}"
export IOWARP_CORE_ROOT="${IOWARP_CORE_ROOT:-$REPO_ROOT}"

TEST_BIN="clio_run_collective_restart_tests"
NODE1="iowarp-crestart-node1"
NODE4="iowarp-crestart-node4"
HANG_TIMEOUT="${HANG_TIMEOUT:-45}"
REPRO_HOLD_MS="${REPRO_HOLD_MS:-15000}"
LOG="$(mktemp)"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${YELLOW}[INFO]${NC} $*"; }
pass()  { echo -e "${GREEN}[PASS]${NC} $*"; }
fail()  { echo -e "${RED}[FAIL]${NC} $*"; }

cd "$SCRIPT_DIR"

cleanup() { info "tearing down cluster"; docker compose down -v >/dev/null 2>&1; rm -f "$LOG"; }
trap cleanup EXIT

info "starting 4-node cluster (image=$IOWARP_DOCKER_IMAGE)"
docker compose up -d || { fail "docker compose up failed"; exit 2; }

info "waiting for runtimes to come up"
sleep 8

info "[setup] creating pool"
docker exec -e REPRO_HOLD_MS="$REPRO_HOLD_MS" "$NODE1" \
  bash -c "$TEST_BIN '[collective_restart][setup]'" || {
  fail "pool setup failed"; exit 2; }

info "[hang] launching broadcast (hold=${REPRO_HOLD_MS}ms) in background"
docker exec -e REPRO_HOLD_MS="$REPRO_HOLD_MS" "$NODE1" \
  bash -c "$TEST_BIN '[collective_restart][hang]'" >"$LOG" 2>&1 &
EXEC_PID=$!

info "waiting for broadcast dispatch marker"
for _ in $(seq 1 30); do
  grep -q "broadcast dispatched" "$LOG" && break
  sleep 0.5
done
grep -q "broadcast dispatched" "$LOG" || { fail "broadcast never dispatched"; cat "$LOG"; exit 2; }

info "restarting $NODE4 mid-collective"
docker restart "$NODE4" >/dev/null

info "waiting up to ${HANG_TIMEOUT}s for the broadcast to return"
RETURNED=0
for _ in $(seq 1 "$HANG_TIMEOUT"); do
  if grep -q "broadcast returned" "$LOG"; then RETURNED=1; break; fi
  if ! kill -0 "$EXEC_PID" 2>/dev/null; then break; fi
  sleep 1
done

echo "----- broadcast log -----"; cat "$LOG"; echo "-------------------------"

if [ "$RETURNED" -eq 1 ]; then
  pass "collective returned after mid-flight restart (no hang)"
  wait "$EXEC_PID" 2>/dev/null
  exit 0
else
  fail "collective HUNG after mid-flight restart (reproduced #628)"
  kill "$EXEC_PID" 2>/dev/null
  docker exec "$NODE1" bash -c "pkill -f $TEST_BIN" 2>/dev/null
  exit 1
fi
