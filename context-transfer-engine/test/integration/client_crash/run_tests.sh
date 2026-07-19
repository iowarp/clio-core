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
# liveness check; (B) by scanning the daemon log for a bounded re-queue count.

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

    say "$BLUE" "Bringing up daemon + crash driver..."
    docker compose up -d

    say "$BLUE" "Waiting for the driver (crash + verify phases) to finish..."
    local verify_exit
    verify_exit=$(docker wait "$DRIVER" 2>/dev/null || echo "1")

    say "$BLUE" "----- driver log -----"
    docker logs "$DRIVER" 2>&1 | grep -aviE "jarvis|repo add|positional|optional argument|--force|repo_path|\(required|\(default|path to repo" | tail -30

    # (A) daemon must still be running after the client crash.
    local daemon_running
    daemon_running=$(docker inspect -f '{{.State.Running}}' "$DAEMON" 2>/dev/null || echo "false")

    # (B) re-queue must be bounded (no #722 runaway).
    local requeues
    requeues=$(docker logs "$DAEMON" 2>&1 | grep -c "re-queueing client response" || true)
    local drops
    drops=$(docker logs "$DAEMON" 2>&1 | grep -c "dropping undeliverable client response" || true)

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
    else
        say "$GREEN" "PASS (B): re-queue bounded ($requeues < $REQUEUE_MAX)"
    fi

    if [ "${CC_KEEP:-0}" != "1" ]; then cleanup; fi

    if [ "$rc" = "0" ]; then say "$GREEN" "ALL CLIENT-CRASH CHECKS PASSED"; else say "$RED" "CLIENT-CRASH TEST FAILED"; fi
    exit $rc
}

main "$@"
