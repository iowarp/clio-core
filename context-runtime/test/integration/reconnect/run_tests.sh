#!/bin/bash
# Run IOWarp Reconnect Integration Test (Chimaera Runtime)
#
# Tests client failover to a new host when the connected server dies:
# 1. Starts 4-node Docker cluster
# 2. Client on node 1 shuts down local runtime via AsyncStopRuntime
# 3. Client sends a new task — WaitForServerAndReconnect fails over to another host
# 4. Task completes on the new host

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
TEST_FILTER=${TEST_FILTER:-[reconnect]}

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

# Run the reconnect test inside node 1
run_single_test() {
    local filter="$1"
    docker exec iowarp-reconnect-node1 bash -c "
        export CHI_WITH_RUNTIME=0
        chimaera_reconnect_tests '$filter'
    "
}

# Run the full reconnect test sequence
run_test_docker_direct() {
    log_info "Running reconnect test sequence"
    cd "$SCRIPT_DIR"

    # Wait for all runtimes to be ready
    log_info "Waiting for runtimes to initialize across all nodes..."
    sleep 5

    # Single phase: the test itself shuts down node 1 and reconnects
    log_info "Running reconnect failover test..."
    local rc=0
    run_single_test "[reconnect]" || rc=$?
    if [ $rc -ne 0 ]; then
        log_error "Reconnect failover test FAILED (exit code $rc)"
        return $rc
    fi
    log_success "Reconnect failover test passed"

    log_success "All reconnect tests completed"
}

# Usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS] COMMAND

Commands:
    setup       Start the Docker cluster
    run         Run the reconnect test
    clean       Stop the Docker cluster and clean up
    all         Setup, run, and clean (default)

Options:
    -n, --num-nodes NUM     Number of nodes (default: 4)
    -f, --filter FILTER     Test name filter (default: [reconnect])
    -h, --help              Show this help message

Examples:
    $0 all
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
log_info "IOWarp Reconnect Test Runner"
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
            log_error "Reconnect test FAILED"
            exit $EXIT_CODE
        fi
        log_success "Reconnect test PASSED"
        ;;

    *)
        log_error "Unknown command: $COMMAND"
        usage
        exit 1
        ;;
esac
