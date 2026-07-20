#!/bin/bash
# Gray-Scott L=512 congestion-collapse repro driver (issue #774).
#
# Runs the 2-node cluster twice:
#   Phase 1  "L=256 analog"  (GS_BLOB_KB=512):  MUST complete all outputs on
#            both nodes — this validates the harness itself.
#   Phase 2  "L=512 analog"  (GS_BLOB_KB=4096): 8x the bytes and 8x the 64KB
#            block transactions per output. REPRODUCED = a node reports STALL
#            (exit 42) or the phase times out with outputs incomplete.
#
# Exit code: 0 if phase 1 passes (harness sane), regardless of phase-2 verdict
# — phase 2's outcome is the REPRO REPORT, printed at the end. Use
#   ./run_tests.sh small   | ./run_tests.sh large
# to run a single phase.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../" && pwd)"

if [ -n "${HOST_WORKSPACE:-}" ]; then
    export IOWARP_CORE_ROOT="${HOST_WORKSPACE}"
elif [ -z "${IOWARP_CORE_ROOT:-}" ]; then
    export IOWARP_CORE_ROOT="${REPO_ROOT}"
fi
cd "$SCRIPT_DIR"

RED='\033[0;31m'; GREEN='\033[0;32m'; BLUE='\033[0;34m'; YEL='\033[1;33m'; NC='\033[0m'
say()  { echo -e "${BLUE}$*${NC}"; }
ok()   { echo -e "${GREEN}✓ $*${NC}"; }
warn() { echo -e "${YEL}$*${NC}"; }
err()  { echo -e "${RED}✗ $*${NC}"; }

command -v docker >/dev/null 2>&1 || { err "Docker not installed"; exit 1; }
docker compose version >/dev/null 2>&1 || { err "docker compose not available"; exit 1; }
docker ps >/dev/null 2>&1 || { err "Docker daemon not running"; exit 1; }

export HOST_UID=$(id -u) HOST_GID=$(id -g)
export IOWARP_DOCKER_IMAGE="${IOWARP_DOCKER_IMAGE:-iowarp/deps-cpu:latest}"
say "Using image: ${IOWARP_DOCKER_IMAGE}"

cleanup() { docker compose down -v >/dev/null 2>&1 || true; }
trap cleanup EXIT

# run_phase <label> <blob_kb> <hard_timeout_sec>
# Sets: PHASE_RC1 PHASE_RC2 (node exits; 124 = phase hard-timeout).
run_phase() {
    local label="$1" blob_kb="$2" hard_to="$3"
    export GS_BLOB_KB="$blob_kb"
    export GS_BLOBS="${GS_BLOBS:-64}"
    export GS_OUTPUTS="${GS_OUTPUTS:-3}"
    export GS_OUTPUT_TIMEOUT_SEC="${GS_OUTPUT_TIMEOUT_SEC:-240}"

    say "=== Phase '$label': GS_BLOB_KB=$blob_kb GS_BLOBS=$GS_BLOBS GS_OUTPUTS=$GS_OUTPUTS (cap ${hard_to}s) ==="
    docker compose down -v >/dev/null 2>&1 || true
    if ! docker compose up -d; then
        err "docker compose up failed"; docker compose logs || true
        PHASE_RC1=1; PHASE_RC2=1; return
    fi

    # Bounded wait on both producers; 124 marks a hard phase timeout (a node
    # that never even reached its own per-output stall cutoff).
    PHASE_RC1=$(timeout "$hard_to" docker wait gs-congest-node1 2>/dev/null || echo 124)
    PHASE_RC2=$(timeout "$hard_to" docker wait gs-congest-node2 2>/dev/null || echo 124)

    say "--- node1 output lines ---"
    docker logs gs-congest-node1 2>&1 | grep -aE "gs-congest|OUTPUT|RESULT|STALL" | tail -12
    say "--- node2 output lines ---"
    docker logs gs-congest-node2 2>&1 | grep -aE "gs-congest|OUTPUT|RESULT|STALL" | tail -12
    echo "phase '$label': node1 exit=$PHASE_RC1 node2 exit=$PHASE_RC2"
    docker compose down -v >/dev/null 2>&1 || true
}

MODE="${1:-both}"
small_ok=-1; large_verdict="not-run"

# --- tiny: HARD regression gate for cross-node SHM clients (#774) -----------
# 4 x 64KB blobs, 2 outputs. Before the client_net_key_ echo fix, ANY blob
# whose name hash-routed to the other node's container left the SHM client's
# future incomplete forever (run2run repurposes task_id_.net_key_; the
# response came back bearing the daemon's key and the client demux parked
# it). This phase must PASS on both nodes; it is wired into cluster-tests.yml
# so the cross-node SHM client path can never silently regress again.
if [ "$MODE" = "tiny" ]; then
    GS_BLOBS=4 GS_OUTPUTS=2 run_phase "tiny cross-node SHM gate" 64 420
    if [ "$PHASE_RC1" = "0" ] && [ "$PHASE_RC2" = "0" ]; then
        ok "Cross-node SHM client gate PASSED (issue #774)"
        exit 0
    fi
    err "Cross-node SHM client gate FAILED (node1=$PHASE_RC1 node2=$PHASE_RC2)"
    err "  A cross-node-routed SHM-client task did not complete — see #774."
    exit 1
fi

if [ "$MODE" = "small" ] || [ "$MODE" = "both" ]; then
    run_phase "L=256 analog (small)" 512 600
    if [ "$PHASE_RC1" = "0" ] && [ "$PHASE_RC2" = "0" ]; then
        ok "Small analog completed on both nodes — harness is sane."
        small_ok=1
    else
        err "Small analog FAILED (node1=$PHASE_RC1 node2=$PHASE_RC2) — harness problem, phase-2 verdict would be meaningless."
        small_ok=0
    fi
fi

if { [ "$MODE" = "large" ] || { [ "$MODE" = "both" ] && [ "$small_ok" = "1" ]; }; }; then
    run_phase "L=512 analog (large)" 4096 900
    if [ "$PHASE_RC1" = "42" ] || [ "$PHASE_RC2" = "42" ] || \
       [ "$PHASE_RC1" = "124" ] || [ "$PHASE_RC2" = "124" ]; then
        large_verdict="REPRODUCED"
    elif [ "$PHASE_RC1" = "0" ] && [ "$PHASE_RC2" = "0" ]; then
        large_verdict="NOT-REPRODUCED (completed — compare per-output times above for degradation)"
    else
        large_verdict="INCONCLUSIVE (node1=$PHASE_RC1 node2=$PHASE_RC2)"
    fi
fi

echo ""
say "================= GRAY-SCOTT CONGESTION REPRO REPORT (issue #774) ================="
[ "$small_ok" != "-1" ] && echo "  small (L=256 analog): $( [ "$small_ok" = "1" ] && echo PASS || echo FAIL )"
echo "  large (L=512 analog): $large_verdict"
say "==================================================================================="

# The harness's own pass/fail is the small phase; the large phase is a report.
if [ "$MODE" = "large" ]; then exit 0; fi
[ "$small_ok" = "1" ] && exit 0 || exit 1
