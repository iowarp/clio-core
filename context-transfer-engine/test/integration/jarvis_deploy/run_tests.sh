#!/bin/bash
# Jarvis Deployment CTE Integration Test Runner
# Deploys IOWarp CTE across a 2-node Docker cluster using Jarvis pipeline runner.
#
# Flow:
#   Node 1 (primary):
#     - Sets up SSH key pair (shared via workspace volume)
#     - Initialises Jarvis and adds the jarvis_iowarp package repository
#     - Runs: jarvis ppl run yaml cte_integration_test.yaml
#       which starts the IOWarp runtime on both nodes (via pssh/SSH),
#       deploys CTE pools (via chimaera compose on all nodes),
#       and runs the PutGet benchmark.
#   Node 2 (secondary):
#     - Waits for node1's SSH public key, then starts sshd in foreground.
#     - Does nothing else; the IOWarp runtime is started here remotely by Jarvis.
#
# Coverage: Uses deps-cpu container with mounted workspace, allowing
# gcda files to be written directly to the build directory for coverage.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../" && pwd)"

# Export workspace path for docker-compose
# Priority: HOST_WORKSPACE > existing IOWARP_CORE_ROOT > computed REPO_ROOT
if [ -n "${HOST_WORKSPACE:-}" ]; then
    export IOWARP_CORE_ROOT="${HOST_WORKSPACE}"
elif [ -z "${IOWARP_CORE_ROOT:-}" ]; then
    export IOWARP_CORE_ROOT="${REPO_ROOT}"
fi
# Otherwise keep existing IOWARP_CORE_ROOT (e.g., from devcontainer.json)

cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored message
print_msg() {
    local color=$1
    shift
    echo -e "${color}$@${NC}"
}

print_header() {
    echo ""
    print_msg "$BLUE" "=================================="
    print_msg "$BLUE" "$@"
    print_msg "$BLUE" "=================================="
}

print_success() {
    print_msg "$GREEN" "✓ $@"
}

print_error() {
    print_msg "$RED" "✗ $@"
}

print_warning() {
    print_msg "$YELLOW" "⚠ $@"
}

# Function to check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"

    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    print_success "Docker found"

    if ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not installed"
        exit 1
    fi
    print_success "Docker Compose found"

    # Check if docker daemon is running
    if ! docker ps &> /dev/null; then
        print_error "Docker daemon is not running"
        exit 1
    fi
    print_success "Docker daemon is running"

    # Check for required files
    if [ ! -f "docker-compose.yaml" ]; then
        print_error "docker-compose.yaml not found"
        exit 1
    fi
    print_success "docker-compose.yaml found"

    if [ ! -f "hostfile" ]; then
        print_error "hostfile not found"
        exit 1
    fi
    print_success "hostfile found"

    if [ ! -f "cte_integration_test.yaml" ]; then
        print_error "cte_integration_test.yaml (Jarvis pipeline) not found"
        exit 1
    fi
    print_success "cte_integration_test.yaml (Jarvis pipeline) found"
}

# Function to clean up previous runs
cleanup() {
    print_header "Cleaning Up Previous Test Environment"

    if docker compose ps -q 2>/dev/null | grep -q .; then
        print_msg "$YELLOW" "Stopping running containers..."
        docker compose down -v 2>/dev/null || true
        print_success "Containers stopped"
    else
        print_success "No running containers to clean up"
    fi

    # Clean up Jarvis runtime artifacts left in the workspace volume.
    # Use sudo because files may be owned by the container's root user.
    if [ -d ".jarvis_ssh" ]; then
        sudo rm -rf ".jarvis_ssh" || rm -rf ".jarvis_ssh" || true
        print_success "Removed .jarvis_ssh directory"
    fi
    if [ -d ".jarvis-shared" ]; then
        sudo rm -rf ".jarvis-shared" || rm -rf ".jarvis-shared" || true
        print_success "Removed .jarvis-shared directory"
    fi
}

# Function to start the test environment
start_environment() {
    print_header "Starting 2-Node Jarvis Deployment Test Environment"

    print_msg "$BLUE" "Starting containers..."
    docker compose up -d

    # Wait for containers to be ready with retry logic
    print_msg "$BLUE" "Waiting for containers to start..."
    local max_attempts=30
    local attempt=0
    while [ $attempt -lt $max_attempts ]; do
        # Count all containers (running or exited)
        local total=$(docker compose ps -a -q 2>/dev/null | wc -l)
        if [ "$total" -ge 2 ]; then
            print_success "All 2 containers started successfully"
            return 0
        fi
        sleep 1
        attempt=$((attempt + 1))
    done

    print_error "Not all containers started within ${max_attempts}s (expected 2, got $total)"
    docker compose logs
    exit 1
}

# Function to show usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Run CTE integration test using Jarvis pipeline runner in a 2-node Docker environment.
Node 1 runs 'jarvis ppl run yaml cte_integration_test.yaml'; Node 2 serves SSH only.

OPTIONS:
    -h, --help              Show this help message
    -c, --cleanup-only      Only cleanup previous runs, don't start new tests
    -k, --keep              Keep containers running after tests (for debugging)
    -n, --no-cleanup        Don't cleanup before starting tests

EXAMPLES:
    # Run the test (default)
    $0

    # Run and keep containers for debugging
    $0 --keep

    # Only cleanup previous test runs
    $0 --cleanup-only
EOF
}

# Main execution
main() {
    local cleanup_only=false
    local keep_containers=false
    local do_cleanup=true

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                usage
                exit 0
                ;;
            -c|--cleanup-only)
                cleanup_only=true
                shift
                ;;
            -k|--keep)
                keep_containers=true
                shift
                ;;
            -n|--no-cleanup)
                do_cleanup=false
                shift
                ;;
            -*)
                print_error "Unknown option: $1"
                usage
                exit 1
                ;;
            *)
                print_error "Unexpected argument: $1"
                usage
                exit 1
                ;;
        esac
    done

    print_header "CTE Jarvis Deployment Test Runner"

    # Check prerequisites
    check_prerequisites

    # Cleanup if requested
    if [ "$do_cleanup" = true ] || [ "$cleanup_only" = true ]; then
        cleanup
    fi

    # Exit if cleanup-only mode
    if [ "$cleanup_only" = true ]; then
        print_success "Cleanup complete"
        exit 0
    fi

    # Start test environment
    start_environment

    print_success "Test environment started. Containers are running."
    print_msg "$BLUE" "Jarvis deployment flow executing. Waiting for completion..."

    # Wait for test container (node1) to complete and capture exit code.
    # 2-minute timeout: kill containers and fail if test takes too long.
    local exit_code=0
    exit_code=$(timeout 120 docker wait cte-jarvis-node1 2>/dev/null || echo "1")
    if [ "$exit_code" = "1" ] && ! docker inspect cte-jarvis-node1 --format '{{.State.Status}}' 2>/dev/null | grep -q exited; then
        print_error "Test timed out after 2 minutes"
    fi

    # Show test logs after completion
    print_header "Node 1 Output (Primary)"
    docker logs cte-jarvis-node1 2>&1 | tail -50

    print_header "Node 2 Output (Secondary)"
    docker logs cte-jarvis-node2 2>&1 | tail -20

    print_header "Test Results"
    if [ "$exit_code" = "0" ]; then
        print_success "Jarvis deployment test passed!"
        print_success "  - SSH established between node1 and node2"
        print_success "  - Jarvis pipeline 'cte_integration_test' ran via 'jarvis ppl run yaml'"
        print_success "  - IOWarp runtime started on both nodes (via pssh)"
        print_success "  - CTE pools deployed via 'chimaera compose' (via pssh)"
        print_success "  - CTE PutGet benchmark completed successfully"
    else
        print_error "Jarvis deployment test failed with exit code: $exit_code"
    fi

    # Cleanup unless keep flag is set
    if [ "$keep_containers" = true ]; then
        print_warning "Containers kept running for debugging"
        print_msg "$BLUE" "To view logs: docker compose logs"
        print_msg "$BLUE" "To exec into node1: docker exec -it cte-jarvis-node1 bash"
        print_msg "$BLUE" "To exec into node2: docker exec -it cte-jarvis-node2 bash"
        print_msg "$BLUE" "To cleanup: docker compose down -v"
    else
        print_msg "$BLUE" "Cleaning up containers..."
        docker compose down -v 2>/dev/null || true
        print_success "Cleanup complete"
    fi

    exit $exit_code
}

# Run main function
main "$@"
