#!/bin/bash
# Dashboard Integration Test for CLIO Runtime
#
# Spins up a 4-node cluster whose node1 daemon serves the BUILT-IN web
# dashboard (issue #990) and validates the cluster-facing read paths:
#   - /api/topology lists every node
#   - /api/nodes/{n}/workers and /api/nodes/{n}/system_stats answer for the
#     local node AND for a remote node (the cross-node Monitor forward)
#   - /api/pools and /api/chimods answer
#
# Node shutdown/restart via the dashboard was a feature of the retired Python
# visualizer and was deliberately not carried into the built-in dashboard
# (unauthenticated destructive endpoints); those flows now live with the CLI.
#
# Usage:
#   bash run_tests.sh all      # setup, run tests, teardown
#   bash run_tests.sh setup    # start cluster only
#   bash run_tests.sh run      # run tests on existing cluster
#   bash run_tests.sh clean    # stop cluster

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../" && pwd)"

# Export workspace path for docker-compose
if [ -n "${HOST_WORKSPACE:-}" ]; then
    export IOWARP_CORE_ROOT="${HOST_WORKSPACE}"
elif [ -z "${IOWARP_CORE_ROOT:-}" ]; then
    export IOWARP_CORE_ROOT="${REPO_ROOT}"
fi

NUM_NODES=4
PASSED=0
FAILED=0
DOCKER_NETWORK="dashboard_dashboard-cluster"
IN_CONTAINER=false
NODE1_IP="172.26.0.10"

# Detect if running inside a container (devcontainer / CI)
if [ -f /.dockerenv ] || grep -qsm1 'docker\|containerd' /proc/1/cgroup 2>/dev/null ||
   [ "$(cat /proc/1/sched 2>/dev/null | head -1 | awk '{print $1}')" != "systemd" ] 2>/dev/null; then
    IN_CONTAINER=true
fi

# When inside a container, connect to the Docker network to reach the dashboard
# directly; otherwise use localhost via the published port.
if [ "$IN_CONTAINER" = true ]; then
    DASHBOARD_URL="http://${NODE1_IP}:5000"
else
    DASHBOARD_URL="http://localhost:5000"
fi

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[PASS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()   { echo -e "${RED}[FAIL]${NC} $1"; }

assert_curl() {
    local description="$1"
    local url="$2"
    local method="${3:-GET}"
    local expected_field="$4"
    local min_count="${5:-}"

    log_info "Test: $description"

    local response http_code
    if [ "$method" = "POST" ]; then
        response=$(curl -s --max-time 30 -w '\n%{http_code}' -X POST "$url" 2>&1)
    else
        response=$(curl -s --max-time 30 -w '\n%{http_code}' "$url" 2>&1)
    fi
    http_code=$(echo "$response" | tail -1)
    response=$(echo "$response" | sed '$d')
    if [ "$http_code" -lt 200 ] 2>/dev/null || [ "$http_code" -ge 400 ] 2>/dev/null; then
        log_error "$description -- HTTP $http_code"
        echo "  Response: $response" | head -3
        FAILED=$((FAILED + 1))
        return 1
    fi
    if [ -z "$http_code" ] || [ "$http_code" = "000" ]; then
        log_error "$description -- curl failed (no response)"
        FAILED=$((FAILED + 1))
        return 1
    fi

    if [ -n "$expected_field" ]; then
        local count
        count=$(echo "$response" | python3 -c "
import sys, json
data = json.load(sys.stdin)
field = '$expected_field'
parts = field.split('.')
val = data
for p in parts:
    val = val[p]
if isinstance(val, list):
    print(len(val))
elif isinstance(val, bool):
    print('true' if val else 'false')
else:
    print(val)
" 2>/dev/null) || {
            log_error "$description -- field '$expected_field' not found in response"
            echo "  Response: $response"
            FAILED=$((FAILED + 1))
            return 1
        }

        if [ -n "$min_count" ]; then
            if [ "$min_count" = "true" ]; then
                if [ "$count" != "true" ]; then
                    log_error "$description -- expected true, got '$count'"
                    FAILED=$((FAILED + 1))
                    return 1
                fi
            elif [ "$count" -lt "$min_count" ] 2>/dev/null; then
                log_error "$description -- expected >= $min_count, got $count"
                FAILED=$((FAILED + 1))
                return 1
            fi
        fi
    fi

    log_success "$description"
    PASSED=$((PASSED + 1))
    return 0
}

# --- Commands ---

start_docker_cluster() {
    log_info "Starting Docker cluster with $NUM_NODES nodes + dashboard..."
    cd "$SCRIPT_DIR"

    # Auto-detect Docker image: use nvidia image if binary requires CUDA
    if [ -z "${IOWARP_DOCKER_IMAGE:-}" ]; then
        CLIO_BIN="/workspace/build/bin/clio_run"
        [ ! -f "$CLIO_BIN" ] && CLIO_BIN="${IOWARP_CORE_ROOT:-/workspace}/build/bin/clio_run"
        if [ -f "$CLIO_BIN" ] && ldd "$CLIO_BIN" 2>/dev/null | grep -q "libcudart"; then
            export IOWARP_DOCKER_IMAGE="iowarp/deps-nvidia:latest"
        else
            export IOWARP_DOCKER_IMAGE="iowarp/deps-cpu:latest"
        fi
    fi

    docker compose up -d

    # When running inside a container, join the Docker network so we can
    # reach the dashboard node directly (localhost port-mapping won't work).
    if [ "$IN_CONTAINER" = true ]; then
        log_info "Detected containerised environment -- joining Docker network..."
        docker network connect "$DOCKER_NETWORK" "$(hostname)" 2>/dev/null || true
    fi

    log_info "Waiting 25s for cluster + dashboard to initialize..."
    sleep 25

    docker compose ps
    log_success "Docker cluster started"
}

stop_docker_cluster() {
    log_info "Stopping Docker cluster..."
    cd "$SCRIPT_DIR"
    if [ "$IN_CONTAINER" = true ]; then
        docker network disconnect "$DOCKER_NETWORK" "$(hostname)" 2>/dev/null || true
    fi
    docker compose down -v
    log_success "Docker cluster stopped"
}

run_tests() {
    log_info "Running dashboard integration tests against $DASHBOARD_URL"
    log_info ""

    # --- Liveness ---
    assert_curl \
        "GET /api/health answers ok" \
        "$DASHBOARD_URL/api/health" \
        "GET" \
        "ok" \
        "true"

    # --- Topology lists all 4 nodes ---
    assert_curl \
        "GET /api/topology returns $NUM_NODES nodes" \
        "$DASHBOARD_URL/api/topology" \
        "GET" \
        "nodes" \
        "$NUM_NODES"

    # --- Local node reads ---
    assert_curl \
        "GET /api/nodes/local/workers returns worker data" \
        "$DASHBOARD_URL/api/nodes/local/workers" \
        "GET" \
        "workers" \
        "1"

    assert_curl \
        "GET /api/nodes/local/system_stats returns entries" \
        "$DASHBOARD_URL/api/nodes/local/system_stats" \
        "GET" \
        "entries" \
        "1"

    # --- Cross-node forward: pick a node OTHER than the dashboard's own ---
    # This is the physical:N Monitor path a single-node ctest cannot exercise.
    local self_id remote_id
    self_id=$(curl -sf --max-time 10 "$DASHBOARD_URL/api/health" | python3 -c "
import sys, json
print(json.load(sys.stdin)['node_id'])
" 2>/dev/null) || self_id=0
    remote_id=$(curl -sf --max-time 15 "$DASHBOARD_URL/api/topology" | python3 -c "
import sys, json
nodes = json.load(sys.stdin)['nodes']
alive = [n['node_id'] for n in nodes if n.get('alive') and n['node_id'] != $self_id]
print(alive[0] if alive else '')
" 2>/dev/null) || remote_id=""

    if [ -n "$remote_id" ]; then
        assert_curl \
            "GET /api/nodes/$remote_id/workers (cross-node forward)" \
            "$DASHBOARD_URL/api/nodes/$remote_id/workers" \
            "GET" \
            "workers" \
            "1"
        assert_curl \
            "GET /api/nodes/$remote_id/system_stats (cross-node forward)" \
            "$DASHBOARD_URL/api/nodes/$remote_id/system_stats" \
            "GET" \
            "entries" \
            "1"
    else
        log_error "No alive remote node found for the cross-node forward test"
        FAILED=$((FAILED + 1))
    fi

    # --- Pool and module inventories ---
    assert_curl \
        "GET /api/pools lists composed pools" \
        "$DASHBOARD_URL/api/pools" \
        "GET" \
        "pools" \
        "1"

    assert_curl \
        "GET /api/chimods lists loaded modules" \
        "$DASHBOARD_URL/api/chimods" \
        "GET" \
        "chimods" \
        "1"

    # --- The dashboard shell itself is served ---
    log_info "Test: GET /viz/clio_admin/index.html serves the dashboard shell"
    local page_code
    page_code=$(curl -s --max-time 15 -o /dev/null -w '%{http_code}' \
        "$DASHBOARD_URL/viz/clio_admin/index.html" 2>/dev/null) || page_code=000
    if [ "$page_code" = "200" ]; then
        log_success "dashboard shell served (HTTP 200)"
        PASSED=$((PASSED + 1))
    else
        log_error "dashboard shell -- HTTP $page_code"
        FAILED=$((FAILED + 1))
    fi

    # --- Summary ---
    log_info ""
    log_info "========================================="
    log_info "  Results: $PASSED passed, $FAILED failed"
    log_info "========================================="

    if [ "$FAILED" -gt 0 ]; then
        log_error "Some tests failed"
        return 1
    fi
    log_success "All tests passed"
    return 0
}

usage() {
    cat << EOF
Usage: $0 COMMAND

Commands:
    setup    Start the 4-node Docker cluster with dashboard
    run      Run integration tests against running cluster
    clean    Stop the Docker cluster
    all      Setup, run tests, and clean up (default)

Environment Variables:
    HOST_WORKSPACE    Host path to workspace (for devcontainers)

Examples:
    $0 all       # Full test cycle
    $0 setup     # Just start the cluster
    $0 run       # Run tests on existing cluster
    $0 clean     # Tear down
EOF
}

# --- Parse args ---
COMMAND=""
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        setup|run|clean|all)
            COMMAND="$1"
            shift
            ;;
        *)
            log_error "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

COMMAND=${COMMAND:-all}

log_info "Dashboard Integration Test"
log_info "  Workspace: $IOWARP_CORE_ROOT"
log_info "  Command:   $COMMAND"
log_info ""

case $COMMAND in
    setup)
        start_docker_cluster
        ;;
    run)
        run_tests
        ;;
    clean)
        stop_docker_cluster
        ;;
    all)
        EXIT_CODE=0
        start_docker_cluster
        run_tests || EXIT_CODE=$?
        stop_docker_cluster
        if [ $EXIT_CODE -ne 0 ]; then
            log_error "Dashboard integration test FAILED"
            exit $EXIT_CODE
        fi
        log_success "Dashboard integration test PASSED"
        ;;
    *)
        log_error "Unknown command: $COMMAND"
        usage
        exit 1
        ;;
esac
