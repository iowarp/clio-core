#!/bin/bash
# Host driver for the distributed clio_run-stop CI test (issue #710).
#
# Brings up a 2-node docker-compose cluster, waits for node1 (which runs the
# Jarvis stop gate) to finish, and propagates its exit code. Node1's exit code
# is the test result: 0 iff `clio_run stop` killed the whole runtime on every
# node with zero stale artifacts.
#
# Usage:
#   ./run_tests.sh                                   # run the test
#   HOST_WORKSPACE=/host/path/to/workspace ./run_tests.sh   # devcontainers
#   ./run_tests.sh --keep                            # keep containers on exit
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../" && pwd)"

# Export workspace path for docker-compose (HOST_WORKSPACE > IOWARP_CORE_ROOT > repo root)
if [ -n "${HOST_WORKSPACE:-}" ]; then
    export IOWARP_CORE_ROOT="${HOST_WORKSPACE}"
elif [ -z "${IOWARP_CORE_ROOT:-}" ]; then
    export IOWARP_CORE_ROOT="${REPO_ROOT}"
fi

cd "$SCRIPT_DIR"

KEEP=0
[ "${1:-}" = "--keep" ] && KEEP=1

RED='\033[0;31m'; GREEN='\033[0;32m'; BLUE='\033[0;34m'; NC='\033[0m'
log() { echo -e "${BLUE}[stop-ci]${NC} $*"; }
ok()  { echo -e "${GREEN}[stop-ci]${NC} $*"; }
err() { echo -e "${RED}[stop-ci]${NC} $*"; }

cleanup() {
    if [ "$KEEP" = 1 ]; then
        log "keeping containers (--keep); 'docker compose down -v' to clean up"
        return
    fi
    log "tearing down cluster"
    docker compose down -v 2>/dev/null || true
    # Jarvis/SSH artifacts left in the mounted workspace, possibly root-owned.
    sudo rm -rf "$SCRIPT_DIR/.chimaera_ssh" "$SCRIPT_DIR/.jarvis-shared" 2>/dev/null \
        || rm -rf "$SCRIPT_DIR/.chimaera_ssh" "$SCRIPT_DIR/.jarvis-shared" 2>/dev/null || true
}
trap cleanup EXIT

# Auto-detect docker image: nvidia if the built clio_run links CUDA, else cpu.
if [ -z "${IOWARP_DOCKER_IMAGE:-}" ]; then
    CLIO_BIN="${IOWARP_CORE_ROOT:-/workspace}/build/bin/clio_run"
    if [ -f "$CLIO_BIN" ] && ldd "$CLIO_BIN" 2>/dev/null | grep -q "libcudart"; then
        export IOWARP_DOCKER_IMAGE="iowarp/deps-nvidia:latest"
    else
        export IOWARP_DOCKER_IMAGE="iowarp/deps-cpu:latest"
    fi
fi

export HOST_UID="$(id -u)" HOST_GID="$(id -g)"

# jarvis_clio_core's packages import jarvis_cd.util.container_utils, which only
# exists on jarvis-cd's `dev` branch; the jarvis-cd baked into iowarp/deps-*
# images is an older `main` checkout that lacks it ("No module named
# 'jarvis_cd.util.container_utils'" at `jarvis pkg append`). The compose file
# already puts /workspace/external/jarvis-cd first on PYTHONPATH, so provision
# a jarvis-cd checkout there when the workspace doesn't have one. Override the
# ref with JARVIS_CD_REF, or point JARVIS_CD_REPO at a fork.
JARVIS_CD_DIR="$REPO_ROOT/external/jarvis-cd"
if [ ! -d "$JARVIS_CD_DIR/jarvis_cd" ]; then
    JARVIS_CD_REPO="${JARVIS_CD_REPO:-https://github.com/grc-iit/jarvis-cd.git}"
    # PINNED to a commit, not a branch name. Tracking dev meant an upstream
    # push could break this job with no commit of ours -- and did: jarvis-cd
    # moved jarvis_cd.util.container_utils and every run went red with
    #   ModuleNotFoundError: No module named 'jarvis_cd.util.container_utils'
    # for 13 of 31 runs across five days. A SHA turns upstream churn into a
    # deliberate bump we review instead of an overnight outage. Set
    # JARVIS_CD_REF=dev to probe a newer upstream before moving the pin.
    JARVIS_CD_REF="${JARVIS_CD_REF:-1d2cf925d36da3a1d62730fbc6c65b5001efe43b}"
    log "cloning jarvis-cd ($JARVIS_CD_REF) into external/jarvis-cd"
    # git clone --branch takes a ref and never a SHA, so fetch the commit
    # directly. The clone fallback keeps JARVIS_CD_REF=dev working.
    if ! ( git init --quiet "$JARVIS_CD_DIR" \
           && git -C "$JARVIS_CD_DIR" remote add origin "$JARVIS_CD_REPO" \
           && git -C "$JARVIS_CD_DIR" fetch --quiet --depth 1 origin "$JARVIS_CD_REF" \
           && git -C "$JARVIS_CD_DIR" checkout --quiet FETCH_HEAD ); then
        rm -rf "$JARVIS_CD_DIR"
        if ! git clone --quiet --depth 1 --branch "$JARVIS_CD_REF" \
                "$JARVIS_CD_REPO" "$JARVIS_CD_DIR"; then
            err "FAIL: could not clone jarvis-cd ($JARVIS_CD_REPO @ $JARVIS_CD_REF)"
            exit 1
        fi
    fi
else
    log "using existing external/jarvis-cd checkout"
fi

log "starting 2-node cluster (image: $IOWARP_DOCKER_IMAGE)"
docker compose down -v 2>/dev/null || true
docker compose up -d

log "waiting for node1 to run the stop gate (timeout 300s)"
EXIT_CODE="$(timeout 300 docker wait chimaera-stop-node1 2>/dev/null || echo 1)"

echo "==================== node1 (gate) log ===================="
docker logs chimaera-stop-node1 2>&1 | tail -120
echo "==================== node2 (sshd) log ===================="
docker logs chimaera-stop-node2 2>&1 | tail -20
echo "=========================================================="

if [ "$EXIT_CODE" = "0" ]; then
    ok "PASS: clio_run stop consistently killed the whole runtime (no stale artifacts)"
else
    err "FAIL: distributed stop test exited $EXIT_CODE"
fi
exit "$EXIT_CODE"
