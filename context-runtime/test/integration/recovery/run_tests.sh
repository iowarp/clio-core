#!/bin/bash
# Run IOWarp Recovery Integration Test (Chimaera Runtime)
#
# Tests node failure detection and container recovery:
# 1. Starts 4-node Docker cluster
# 2. Runs pre-failure tests (verify all 4 nodes work)
# 3. Kills node 4 (simulates node failure)
# 4. Waits 30s for SWIM detection + automatic recovery
# 5. Runs post-failure tests (verify tasks re-route to recovered containers)

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
TEST_FILTER=${TEST_FILTER:-[recovery]}

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

# Run a single test case inside the Docker cluster
run_single_test() {
    local filter="$1"
    docker exec iowarp-recovery-node1 bash -c "
        export CHI_WITH_RUNTIME=0
        chimaera_recovery_tests '$filter'
    "
}

# Run the full recovery test sequence
run_test_docker_direct() {
    log_info "Running recovery test sequence"
    cd "$SCRIPT_DIR"

    # Wait for all runtimes to be ready
    log_info "Waiting for runtimes to initialize across all nodes..."
    sleep 5

    # Phase 1: Pre-failure tests
    log_info "Phase 1: Running pre-failure tests..."
    run_single_test "[recovery][pre_failure]"
    log_success "Phase 1: Pre-failure tests passed"

    # Kill node 4 to simulate node failure
    log_info "Killing node 4 to simulate node failure..."
    docker stop iowarp-recovery-node4
    log_info "Node 4 stopped"

    # Wait for SWIM detection + automatic recovery
    # SWIM worst case: 5s (direct) + 3s (indirect) + 10s (suspicion) = 18s
    # With heartbeat_interval=500ms, 30s gives good margin
    log_info "Waiting 30s for SWIM detection and automatic recovery..."
    sleep 30

    # Phase 2: Post-failure tests
    log_info "Phase 2: Running post-failure tests..."
    run_single_test "[recovery][post_failure]"
    log_success "Phase 2: Post-failure tests passed"

    log_success "All recovery tests completed"
}

# Usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS] COMMAND

Commands:
    setup       Start the Docker cluster
    run         Run the recovery test
    clean       Stop the Docker cluster and clean up
    all         Setup, run, and clean (default)

Options:
    -n, --num-nodes NUM     Number of nodes (default: 4)
    -f, --filter FILTER     Test name filter (default: [recovery])
    -h, --help              Show this help message

Examples:
    $0 all
    $0 -f "[recovery][pre_failure]" run
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
log_info "IOWarp Recovery Test Runner"
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
            log_error "Recovery test FAILED"
            exit $EXIT_CODE
        fi
        log_success "Recovery test PASSED"
        ;;

    *)
        log_error "Unknown command: $COMMAND"
        usage
        exit 1
        ;;
esac
