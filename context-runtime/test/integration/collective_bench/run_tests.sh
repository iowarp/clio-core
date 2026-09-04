#!/bin/bash
# Collective latency benchmark runner: Clio PoolQuery::AllToOne vs MPI.
#
# Brings up a 4-node clio cluster in Docker and launches `mpirun -np 4` across
# all four containers (one rank per node, each attached to its own local
# daemon). The authoritative result is node 1's exit code (the mpirun launcher):
# the MPI job reduces every rank's result there, and the benchmark returns
# non-zero if any arm failed or the allreduce results did not verify.
#
# Commands: setup | run | clean | all (default)
# Env: NUM_NODES (4), COLL_BENCH_ITERS (1000), COLL_BENCH_WARMUP (100)
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../" && pwd)"

if [ -n "${HOST_WORKSPACE:-}" ]; then
    export IOWARP_CORE_ROOT="${HOST_WORKSPACE}"
elif [ -z "${IOWARP_CORE_ROOT:-}" ]; then
    export IOWARP_CORE_ROOT="${REPO_ROOT}"
fi

export NUM_NODES="${NUM_NODES:-4}"
export COLL_BENCH_ITERS="${COLL_BENCH_ITERS:-1000}"
export COLL_BENCH_WARMUP="${COLL_BENCH_WARMUP:-100}"
export COLL_BENCH_VERIFY_ROUNDS="${COLL_BENCH_VERIFY_ROUNDS:-4}"
export COLL_BENCH_STAGGER_MS="${COLL_BENCH_STAGGER_MS:-20}"

cd "$SCRIPT_DIR"

RED='\033[0;31m'; GREEN='\033[0;32m'; BLUE='\033[0;34m'; NC='\033[0m'
say()  { echo -e "${BLUE}$*${NC}"; }
ok()   { echo -e "${GREEN}✓ $*${NC}"; }
err()  { echo -e "${RED}✗ $*${NC}"; }

command -v docker >/dev/null 2>&1 || { err "Docker not installed"; exit 1; }
docker compose version >/dev/null 2>&1 || { err "docker compose not available"; exit 1; }
docker ps >/dev/null 2>&1 || { err "Docker daemon not running"; exit 1; }

BENCH_BIN="${IOWARP_CORE_ROOT}/build/bin/clio_collective_bench"
if [ ! -f "$BENCH_BIN" ]; then
    err "clio_collective_bench not built at ${BENCH_BIN}"
    err "  It requires MPI: configure with -DCLIO_CORE_ENABLE_MPI=ON -DCLIO_CORE_ENABLE_DOCKER_CI=ON"
    exit 1
fi

# Match the docker image to the built binary (CPU vs CUDA), as the sibling
# distributed tests do.
if [ -z "${IOWARP_DOCKER_IMAGE:-}" ]; then
    CLIO_BIN="${IOWARP_CORE_ROOT}/build/bin/clio_run"
    if [ -f "$CLIO_BIN" ] && ldd "$CLIO_BIN" 2>/dev/null | grep -q "libcudart"; then
        export IOWARP_DOCKER_IMAGE="iowarp/deps-nvidia:latest"
    else
        export IOWARP_DOCKER_IMAGE="iowarp/deps-cpu:latest"
    fi
fi

clear_share() {
    rm -rf "${SCRIPT_DIR}/.mpi_ssh" 2>/dev/null \
        || sudo rm -rf "${SCRIPT_DIR}/.mpi_ssh" 2>/dev/null || true
}

start_cluster() {
    say "=== Cleaning any previous run ==="
    docker compose down -v >/dev/null 2>&1 || true
    clear_share
    say "=== Starting ${NUM_NODES}-node cluster (image: ${IOWARP_DOCKER_IMAGE}) ==="
    say "    iters=${COLL_BENCH_ITERS} warmup=${COLL_BENCH_WARMUP}"
    if ! docker compose up -d; then
        err "docker compose up failed"
        docker compose logs || true
        return 1
    fi
    return 0
}

stop_cluster() {
    say "=== Stopping cluster ==="
    docker compose down -v >/dev/null 2>&1 || true
    clear_share
}

wait_and_report() {
    say "=== Waiting for node 1 (mpirun launcher) to finish ==="
    local rc
    rc=$(docker wait cb-node1 2>/dev/null || echo 1)

    # Pull artifacts out of the containers before they are torn down. The
    # daemon logs carry the [COLLPROF]/[NETQPROF] stage timings, which are the
    # only way to attribute a slow collective to a stage -- and they are gone
    # the moment `docker compose down` runs.
    docker cp cb-node1:/tmp/coll_bench_results.csv "${SCRIPT_DIR}/results.csv" \
        >/dev/null 2>&1 || true
    mkdir -p "${SCRIPT_DIR}/logs"
    for n in $(seq 1 "${NUM_NODES}"); do
        docker cp "cb-node${n}:/tmp/clio_daemon.log" \
            "${SCRIPT_DIR}/logs/daemon-node${n}.log" >/dev/null 2>&1 || true
    done

    say "=== Node 1 (launcher) log ==="
    docker logs cb-node1 2>&1 | tail -80
    for n in $(seq 2 "${NUM_NODES}"); do
        say "=== Node ${n} log (tail) ==="
        docker logs "cb-node${n}" 2>&1 | tail -15
    done

    echo ""
    say "=== Result ==="
    echo "node1(launcher) exit=${rc}"
    if [ -f "${SCRIPT_DIR}/results.csv" ]; then
        say "=== results.csv ==="
        cat "${SCRIPT_DIR}/results.csv"
    fi
    if grep -qha "COLLPROF\|NETQPROF" "${SCRIPT_DIR}"/logs/*.log 2>/dev/null; then
        say "=== stage profiles (last sample per node) ==="
        for n in $(seq 1 "${NUM_NODES}"); do
            local f="${SCRIPT_DIR}/logs/daemon-node${n}.log"
            [ -f "$f" ] || continue
            grep -ha "COLLPROF" "$f" | tail -1 | sed "s/^/node${n} /"
            grep -ha "NETQPROF" "$f" | tail -6 | sed "s/^/node${n} /"
        done
    fi

    if [ "$rc" = "0" ]; then
        ok "Collective benchmark completed (allreduce results verified)"
        return 0
    fi
    err "Collective benchmark FAILED (node1=${rc})"
    err "  exit 2=CLIO_INIT, 3=pool create, 4=allreduce mismatch, 5=failed iterations"
    return 1
}

COMMAND="${1:-all}"
case "$COMMAND" in
    setup)
        start_cluster
        ;;
    run)
        wait_and_report
        ;;
    clean)
        stop_cluster
        ok "Cleanup complete"
        ;;
    all)
        trap stop_cluster EXIT
        start_cluster || exit 1
        wait_and_report
        exit $?
        ;;
    *)
        err "Unknown command: $COMMAND (use setup|run|clean|all)"
        exit 1
        ;;
esac
