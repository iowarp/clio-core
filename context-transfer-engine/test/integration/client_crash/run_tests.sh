#!/bin/bash
# Client-crash integration test runner (issues #722 + "client dies mid-op").
#
# Brings up a daemon + a separate-container external client that crashes
# mid-PutBlob, then asserts the RUNTIME is robust:
#   (A) the daemon survives the client crash (still running afterwards);
#   (B) it does not spin re-queueing the undeliverable response forever
#       (#722: bounded, not 22.9M retries);
#   (C) it stays responsive — a follow-up client (driver Phase 2) succeeds.
#
# (A)+(C) are proven by the driver's exit code (Phase 2 verify) plus a daemon
# liveness check; (B) by the daemon log: every response that entered the
# re-queue path must reach the bounded drop within the watch window.
#
# Why (B) changed. It used to be `grep -c "re-queueing client response" <
# 2000`, but SendOut logs that line once per struggling RESPONSE (later
# attempts are kDebug), so an 800-request burst can never reach 2000: the
# check could not fail. The bounded-drop marker is now polled for (cap
# CC_DROP_WAIT_SEC) and compared against the re-queue count instead.
#
# Caveat, from the code rather than a run: the dial-back DEALER has
# ZMQ_IMMEDIATE=0 and SNDHWM=100000, so responses to a dead client usually
# queue inside ZMQ and Send() returns 0 -- the drop path may not fire at all.
# That case is reported as "not exercised" and passes unless
# CC_REQUIRE_DROP=1, which makes it a hard failure.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../" && pwd)"

if [ -n "${HOST_WORKSPACE:-}" ]; then
    export IOWARP_CORE_ROOT="${HOST_WORKSPACE}"
elif [ -z "${IOWARP_CORE_ROOT:-}" ]; then
    export IOWARP_CORE_ROOT="${REPO_ROOT}"
fi

cd "$SCRIPT_DIR"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
say() { echo -e "${1}${2}${NC}"; }

DAEMON=cc-crash-daemon
DRIVER=cc-crash-driver
# A crash burst of ~800 undeliverable responses is bounded; the #722 runaway was
# 22.9M. Anything approaching thousands of re-queue lines is the bug.
REQUEUE_MAX=2000
SYNC_DIR="$REPO_ROOT/build/cluster_sync/client_crash"
REQUEUE_RE="re-queueing client response"
DROP_RE="dropping undeliverable client response"

# shellcheck source=../../../../CI/compose_wait.sh
source "$REPO_ROOT/CI/compose_wait.sh"

# Count daemon log lines matching a fixed string.
# @param $1 fixed string
daemon_count() {
    docker logs "$DAEMON" 2>&1 | grep -acF "$1" || true
}

# Watch the daemon log after the crash phase: return once every re-queued
# response has been dropped (drops >= re-queues > 0), or after the cap.
# Sets REQUEUES and DROPS.
# @param $1 cap in seconds
watch_drops() {
    local cap="$1" start
    start=$(date +%s)
    while :; do
        REQUEUES=$(daemon_count "$REQUEUE_RE")
        DROPS=$(daemon_count "$DROP_RE")
        if [ "$REQUEUES" -gt 0 ] && [ "$DROPS" -ge "$REQUEUES" ]; then
            say "$GREEN" "drop marker seen after $(( $(date +%s) - start ))s (re-queues=$REQUEUES drops=$DROPS)"
            return 0
        fi
        if [ $(( $(date +%s) - start )) -ge "$cap" ]; then
            say "$YELLOW" "drop watch ended after ${cap}s (re-queues=$REQUEUES drops=$DROPS)"
            return 0
        fi
        sleep 2
    done
}

cleanup() {
    docker compose down -v >/dev/null 2>&1 || true
}

main() {
    say "$BLUE" "===== Client-crash integration test (#722 / client-dies-mid-op) ====="

    command -v docker >/dev/null 2>&1 || { say "$RED" "docker not found"; exit 1; }
    docker compose version >/dev/null 2>&1 || { say "$RED" "docker compose not found"; exit 1; }

    export HOST_UID=$(id -u)
    export HOST_GID=$(id -g)
    if [ -z "${IOWARP_DOCKER_IMAGE:-}" ]; then
        CLIO_BIN="${IOWARP_CORE_ROOT:-/workspace}/build/bin/clio_run"
        if [ -f "$CLIO_BIN" ] && ldd "$CLIO_BIN" 2>/dev/null | grep -q "libcudart"; then
            export IOWARP_DOCKER_IMAGE="iowarp/deps-nvidia:latest"
        else
            export IOWARP_DOCKER_IMAGE="iowarp/deps-cpu:latest"
        fi
    fi

    say "$YELLOW" "Cleaning up any previous run..."
    cleanup

    # One deadline below the workflow step's 15 min (ctest overrides it via
    # CW_BUDGET_SEC to stay inside its own TIMEOUT).
    cw_init "${CC_BUDGET_SEC:-780}" "$DAEMON" "$DRIVER"
    cw_reset_sync "$SYNC_DIR"

    say "$BLUE" "Bringing up daemon + crash driver..."
    docker compose up -d

    # The daemon binds only CLIO_BIND_ADDR (not loopback). An in-container
    # `clio_run status` never completes its handshake there, even pointed at
    # that address (CI: UNRESPONSIVE for 180 s while the daemon was serving),
    # so wait for the runtime's own ready line instead.
    if ! cw_wait_log "$DAEMON" "Compose: All [0-9]+ pools created successfully"; then
        cleanup; exit 1
    fi
    touch "$SYNC_DIR/daemon_ready"

    say "$BLUE" "Waiting for the crash phase to finish..."
    local crash_cap
    crash_cap=$(cw_cap 300)
    for _ in $(seq 1 "$crash_cap"); do
        [ -e "$SYNC_DIR/crash_done" ] && break
        cw_is_running "$DRIVER" || break
        sleep 1
    done
    if [ ! -e "$SYNC_DIR/crash_done" ]; then
        cw_fail "driver never finished the crash phase" || true
        cleanup; exit 1
    fi

    say "$BLUE" "Watching the daemon log for the bounded-drop marker (cap ${CC_DROP_WAIT_SEC:-60}s)..."
    REQUEUES=0; DROPS=0
    watch_drops "$(cw_cap "${CC_DROP_WAIT_SEC:-60}")"
    touch "$SYNC_DIR/verify_go"

    say "$BLUE" "Waiting for the driver (verify phase) to finish..."
    local verify_exit=0
    cw_wait_containers "$DRIVER" || true
    verify_exit="${CW_RC[$DRIVER]:-timeout}"

    say "$BLUE" "----- driver log -----"
    docker logs "$DRIVER" 2>&1 | grep -aviE "jarvis|repo add|positional|optional argument|--force|repo_path|\(required|\(default|path to repo" | tail -30

    # (A) daemon must still be running after the client crash.
    local daemon_running
    daemon_running=$(docker inspect -f '{{.State.Running}}' "$DAEMON" 2>/dev/null || echo "false")

    # (B) re-queue must be bounded (no #722 runaway). Re-read: the verify
    # phase may have added lines.
    local requeues drops
    requeues=$(daemon_count "$REQUEUE_RE")
    drops=$(daemon_count "$DROP_RE")

    say "$BLUE" "----- daemon leak/drop markers -----"
    docker logs "$DAEMON" 2>&1 | grep -aE "dropping undeliverable|re-queueing client response|total SHM bytes leaked" | tail -10 || true

    echo ""
    say "$BLUE" "Results: verify_exit=$verify_exit daemon_running=$daemon_running re-queues=$requeues drops=$drops"

    local rc=0
    if [ "$verify_exit" != "0" ]; then
        say "$RED" "FAIL (C): daemon not responsive after client crash (verify exit=$verify_exit)"; rc=1
    else
        say "$GREEN" "PASS (C): daemon responsive after client crash"
    fi
    if [ "$daemon_running" != "true" ]; then
        say "$RED" "FAIL (A): daemon did not survive the client crash"; rc=1
    else
        say "$GREEN" "PASS (A): daemon survived the client crash"
    fi
    if [ "$requeues" -ge "$REQUEUE_MAX" ]; then
        say "$RED" "FAIL (B): unbounded re-queue ($requeues >= $REQUEUE_MAX) — #722 regression"; rc=1
    elif [ "$REQUEUES" -gt 0 ] && [ "$DROPS" -eq 0 ]; then
        say "$RED" "FAIL (B): $REQUEUES responses entered the re-queue path but none was dropped within ${CC_DROP_WAIT_SEC:-60}s — the bounded drop (#722) did not fire"; rc=1
    elif [ "$drops" -gt 0 ]; then
        say "$GREEN" "PASS (B): bounded drop fired (re-queues=$requeues drops=$drops)"
    elif [ "${CC_REQUIRE_DROP:-0}" = "1" ]; then
        say "$RED" "FAIL (B): drop path not exercised — no '$DROP_RE' line within ${CC_DROP_WAIT_SEC:-60}s (CC_REQUIRE_DROP=1)"; rc=1
    else
        say "$YELLOW" "NOTE (B): drop path not exercised (no re-queue, no drop: ZMQ buffered the responses); set CC_REQUIRE_DROP=1 to make this fatal"
    fi

    # A verify failure already dumped both containers; dump for (A)/(B) too.
    if [ "$rc" != "0" ] && [ "$verify_exit" = "0" ]; then cw_dump; fi
    if [ "${CC_KEEP:-0}" != "1" ]; then cleanup; fi

    if [ "$rc" = "0" ]; then say "$GREEN" "ALL CLIENT-CRASH CHECKS PASSED"; else say "$RED" "CLIENT-CRASH TEST FAILED"; fi
    exit $rc
}

main "$@"
