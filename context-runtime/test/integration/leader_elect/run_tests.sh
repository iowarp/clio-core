#!/bin/bash
# Run IOWarp Leader Election Integration Test (Chimaera Runtime)
#
# Tests leader shutdown, client failover, and leader restart:
# Phase 1: Client on node 1 kills local runtime, fails over to another node
# Phase 2: Node 1 runtime restarted, fresh client verifies system health

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../" && pwd)"

# Export workspace path for docker-compose
if [ -n "${HOST_WORKSPACE:-}" ]; then
    export IOWARP_CORE_ROOT="${HOST_WORKSPACE}"
elif [ -z "${IOWARP_CORE_ROOT:-}" ]; then
    export IOWARP_CORE_ROOT="${REPO_ROOT}"
fi

# Configuration
NUM_NODES=${NUM_NODES:-4}
TEST_FILTER=${TEST_FILTER:-[leader_elect]}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Start Docker cluster
start_docker_cluster() {
    log_info "Starting Docker cluster with $NUM_NODES nodes..."
    cd "$SCRIPT_DIR"

    # Pass host UID/GID so container processes match host file ownership
    export HOST_UID=$(id -u)
    export HOST_GID=$(id -g)

    # Tear down any leftover containers from a previous run
    docker compose down 2>/dev/null || true

    docker compose up -d

    log_info "Waiting for containers to initialize..."
    sleep 10

    docker compose ps

    log_success "Docker cluster started"
}

# Stop Docker cluster
stop_docker_cluster() {
    log_info "Stopping Docker cluster..."
    cd "$SCRIPT_DIR"
    docker compose down
    log_success "Docker cluster stopped"
}

# Run a single test phase inside the Docker cluster
run_single_test() {
    local filter="$1"
    docker exec iowarp-leader-elect-node1 bash -c "
        export CHI_WITH_RUNTIME=0
        chimaera_leader_elect_tests '$filter'
    "
}

# Check container and runtime status across the cluster
check_cluster_status() {
    local label="$1"
    log_info "=== Cluster status: $label ==="
    cd "$SCRIPT_DIR"
    docker compose ps --format "table {{.Name}}\t{{.Status}}" 2>/dev/null || true
    for n in 1 2 3 4; do
        local cname="iowarp-leader-elect-node${n}"
        local running
        running=$(docker inspect -f '{{.State.Running}}' "$cname" 2>/dev/null || echo "gone")
        if [ "$running" = "true" ]; then
            local procs
            procs=$(docker exec "$cname" pgrep -a chimaera 2>/dev/null || echo "(none)")
            log_info "  $cname: UP  runtime=$procs"
        else
            log_warning "  $cname: DOWN ($running)"
        fi
    done
    log_info "=== end cluster status ==="
}

# Run the full leader election test sequence
# Returns non-zero on any phase failure.
run_test_docker_direct() {
    local rc=0

    log_info "Running leader election test sequence"
    cd "$SCRIPT_DIR"

    # Wait for all runtimes to be ready
    log_info "Waiting for runtimes to initialize across all nodes..."
    sleep 5
    check_cluster_status "after init"

    # Phase 1: Leader shutdown and failover
    log_info "Phase 1: Running leader shutdown and failover test..."
    if ! run_single_test "[leader_elect][failover]"; then
        log_error "Phase 1: Leader failover test FAILED"
        check_cluster_status "after Phase 1 FAILURE"
        return 1
    fi
    log_success "Phase 1: Leader failover test passed"
    check_cluster_status "after Phase 1 (leader killed)"

    # Restart node 1's runtime (the leader) in detached mode
    log_info "Restarting node 1's runtime..."
    docker exec -d iowarp-leader-elect-node1 \
        /workspace/build/bin/chimaera runtime restart

    # Wait for the restarted runtime to initialize and rejoin the cluster
    log_info "Waiting 10s for restarted runtime to initialize..."
    sleep 10
    check_cluster_status "after restart + 10s"

    # Phase 2: Verify system health after leader restart
    log_info "Phase 2: Running post-restart health check..."
    if ! run_single_test "[leader_elect][post_restart]"; then
        log_error "Phase 2: Post-restart test FAILED"
        check_cluster_status "after Phase 2 FAILURE"
        return 1
    fi
    log_success "Phase 2: Post-restart test passed"

    log_success "All leader election tests completed"
}

# Usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS] COMMAND

Commands:
    setup       Start the Docker cluster
    run         Run the leader election test
    clean       Stop the Docker cluster and clean up
    all         Setup, run, and clean (default)

Options:
    -n, --num-nodes NUM     Number of nodes (default: 4)
    -f, --filter FILTER     Test name filter (default: [leader_elect])
    -h, --help              Show this help message

Examples:
    $0 all
    $0 -f "[leader_elect][failover]" run
    $0 setup
    $0 run
    $0 clean
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--num-nodes)
            NUM_NODES="$2"
            shift 2
            ;;
        -f|--filter)
            TEST_FILTER="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        setup|run|clean|all)
            COMMAND="$1"
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

COMMAND=${COMMAND:-all}

# Main execution
log_info "IOWarp Leader Election Test Runner"
log_info "Configuration:"
log_info "  Nodes: $NUM_NODES"
log_info "  Test filter: $TEST_FILTER"
log_info "  Workspace path: $IOWARP_CORE_ROOT"
log_info ""

case $COMMAND in
    setup)
        start_docker_cluster
        ;;

    run)
        run_test_docker_direct
        ;;

    clean)
        stop_docker_cluster
        log_success "Cleanup complete"
        ;;

    all)
        EXIT_CODE=0
        start_docker_cluster
        run_test_docker_direct || EXIT_CODE=$?
        stop_docker_cluster
        if [ $EXIT_CODE -ne 0 ]; then
            log_error "Leader election test FAILED"
            exit $EXIT_CODE
        fi
        log_success "Leader election test PASSED"
        ;;

    *)
        log_error "Unknown command: $COMMAND"
        usage
        exit 1
        ;;
esac
