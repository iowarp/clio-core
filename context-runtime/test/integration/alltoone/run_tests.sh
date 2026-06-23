#!/bin/bash
# Run IOWarp AllToOne Collective Integration Test (CLIO Runtime)
#
# This script manages the distributed test environment, including:
# - 4-node docker cluster; runs the AllToOne barrier test as a client
# - Test execution with coverage support
# - Cleanup
#
# Coverage: Uses deps-cpu container with mounted workspace, allowing
# gcda files to be written directly to the build directory for coverage.

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../" && pwd)"

# Export workspace path for docker-compose
# Priority: HOST_WORKSPACE > existing IOWARP_CORE_ROOT > computed REPO_ROOT
if [ -n "${HOST_WORKSPACE:-}" ]; then
    # Explicitly set by user
    export IOWARP_CORE_ROOT="${HOST_WORKSPACE}"
elif [ -z "${IOWARP_CORE_ROOT:-}" ]; then
    # IOWARP_CORE_ROOT not set, use computed path
    export IOWARP_CORE_ROOT="${REPO_ROOT}"
fi
# Otherwise keep existing IOWARP_CORE_ROOT (e.g., from devcontainer.json)

# Configuration
NUM_NODES=${NUM_NODES:-4}
TEST_FILTER=${TEST_FILTER:-"[alltoone]"}  # Catch2 tag of the AllToOne barrier test

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
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

    # Start containers in detached mode
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

    # Wait for containers to be ready
    log_info "Waiting for containers to initialize..."
    sleep 10

    # Check container status
    docker compose ps

    log_success "Docker cluster started"
    log_info "View live logs with: docker compose logs -f"
}

# Stop Docker cluster
stop_docker_cluster() {
    log_info "Stopping Docker cluster..."
    cd "$SCRIPT_DIR"
    docker compose down
    log_success "Docker cluster stopped"
}



# Check if a test name matches the filter
matches_filter() {
    local name="$1"
    local filter="$2"
    if [ -z "$filter" ]; then
        return 0
    fi
    case "$name" in
        *"$filter"*) return 0 ;;
        *) return 1 ;;
    esac
}

# Run the AllToOne collective test inside the cluster as a client of node 1.
# CLIO_WITH_RUNTIME=0 attaches to the already-running 4-node cluster instead of
# spawning a local runtime; CLIO_NUM_CONTAINERS (from the compose env) tells the
# test how many AllToOne requests to issue (one per container = one per node),
# which is exactly the pool's container count, so the barrier releases when all
# arrive. Returns the docker exec's exit code (also catches a crashed daemon's
# "container is not running").
run_single_test() {
    local filter="$1"
    if ! docker exec iowarp-alltoone-node1 bash -c "
        export CLIO_WITH_RUNTIME=0
        clio_run_bdev_chimod_tests '$filter'
    "; then
        return 1
    fi
    return 0
}

# Run the AllToOne barrier test against the cluster.
run_test_docker_direct() {
    log_info "Running AllToOne barrier test with filter: $TEST_FILTER"
    cd "$SCRIPT_DIR"

    # Wait for all runtimes to be ready (give them time to initialize).
    log_info "Waiting for runtimes to initialize across all nodes..."
    sleep 5

    if run_single_test "$TEST_FILTER"; then
        log_success "AllToOne barrier test passed"
        log_success "All tests completed"
        return 0
    fi
    log_error "AllToOne barrier test FAILED"
    return 1
}


# Usage information
usage() {
    cat << EOF
Usage: $0 [OPTIONS] COMMAND

Commands:
    setup       Start the Docker cluster
    run         Run the AllToOne test
    clean       Stop the Docker cluster and clean up
    all         Setup and run (default)

Options:
    -n, --num-nodes NUM     Number of nodes (default: 4)
    -f, --filter FILTER     Test name filter (default: bdev_file_explicit_backend)
    -h, --help              Show this help message

Environment Variables:
    NUM_NODES       Number of nodes
    TEST_FILTER     Test name filter

Examples:
    # Start cluster and run tests (default)
    $0 all

    # Run specific test
    $0 -f "bdev_file_explicit_backend" run

    # Use different number of nodes
    $0 -n 8 all

    # Just setup cluster
    $0 setup

    # Run tests on existing cluster
    $0 run
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

# Default command
COMMAND=${COMMAND:-all}

# Main execution
log_info "IOWarp AllToOne Test Runner"
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
            log_error "AllToOne test FAILED"
            exit $EXIT_CODE
        fi
        log_success "AllToOne test PASSED"
        ;;

    *)
        log_error "Unknown command: $COMMAND"
        usage
        exit 1
        ;;
esac
