#!/bin/bash
# Run IOWarp Distributed Unit Test
#
# This script manages the distributed test environment, including:
# - Docker cluster setup
# - Hostfile generation
# - Test execution
# - Cleanup

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Configuration
NUM_NODES=${NUM_NODES:-4}
TEST_FILTER=${TEST_FILTER:-bdev_file_explicit_backend}  # Test filter

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

    # Start containers in detached mode
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



# Run test directly in Docker
run_test_docker_direct() {
    log_info "Running distributed test with filter: $TEST_FILTER"
    cd "$SCRIPT_DIR"

    # Wait for all runtimes to be ready (give them time to initialize)
    log_info "Waiting for runtimes to initialize across all nodes..."
    sleep 5

    # Execute test on node1 using installed binary
    docker exec iowarp-distributed-node1 bash -c "
        export CHIMAERA_WITH_RUNTIME=0
        chimaera_bdev_chimod_tests $TEST_FILTER
    "

    log_success "Test completed"
}


# Usage information
usage() {
    cat << EOF
Usage: $0 [OPTIONS] COMMAND

Commands:
    setup       Start the Docker cluster
    run         Run the distributed test
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
log_info "IOWarp Distributed Test Runner"
log_info "Configuration:"
log_info "  Nodes: $NUM_NODES"
log_info "  Test filter: $TEST_FILTER"
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
        start_docker_cluster
        run_test_docker_direct
        log_success "All operations completed successfully"
        ;;

    *)
        log_error "Unknown command: $COMMAND"
        usage
        exit 1
        ;;
esac
