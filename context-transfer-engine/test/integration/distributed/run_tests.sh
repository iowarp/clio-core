#!/bin/bash
# Distributed CTE Integration Test Runner
# This script sets up and runs the distributed CTE tests in Docker containers
#
# Coverage: Uses deps-cpu container with mounted workspace, allowing
# gcda files to be written directly to the build directory for coverage.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../" && pwd)"

# Export workspace path for docker-compose
export IOWARP_CORE_ROOT="${REPO_ROOT}"

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

    if [ ! -f "wrp_config.yaml" ]; then
        print_error "wrp_config.yaml not found"
        exit 1
    fi
    print_success "wrp_config.yaml found"
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
}

# Function to start the test environment
start_environment() {
    print_header "Starting 4-Node Distributed Test Environment"

    print_msg "$BLUE" "Starting containers..."
    docker compose up -d

    # Wait for containers to be ready with retry logic
    print_msg "$BLUE" "Waiting for containers to start..."
    local max_attempts=30
    local attempt=0
    while [ $attempt -lt $max_attempts ]; do
        # Count all containers (running or exited)
        local total=$(docker compose ps -a -q 2>/dev/null | wc -l)
        if [ "$total" -ge 4 ]; then
            print_success "All 4 containers started successfully"
            return 0
        fi
        sleep 1
        attempt=$((attempt + 1))
    done

    print_error "Not all containers started within ${max_attempts}s (expected 4, got $total)"
    docker compose logs
    exit 1
}



# Function to show usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS] [TEST_FILTER]

Run CTE distributed unit tests in a 4-node Docker environment.

OPTIONS:
    -h, --help              Show this help message
    -c, --cleanup-only      Only cleanup previous runs, don't start new tests
    -k, --keep              Keep containers running after tests (for debugging)
    -n, --no-cleanup        Don't cleanup before starting tests
    -t, --test <filter>     Filter tests by name (Catch2 test name pattern)
    -s, --section <filter>  Filter by section name (Catch2 section pattern)

ARGUMENTS:
    TEST_FILTER             Optional test filter pattern (alternative to -t)
                           By default, all tests run. Use Catch2 patterns to filter.

EXAMPLES:
    # Run all tests (default)
    $0

    # Run tests matching a specific pattern
    $0 -t "PutBlob"
    $0 "GetBlob*"

    # Run tests and keep containers for debugging
    $0 --keep -t "neighborhood"

    # Run a specific section of a test
    $0 -s "PutBlob operations"

    # Only cleanup previous test runs
    $0 --cleanup-only


CATCH2 TEST FILTER PATTERNS:
    - Exact match: "PutBlob Basic Test"
    - Wildcard: "PutBlob*" or "*neighborhood*"
    - Multiple: "PutBlob,GetBlob"
    - Exclude: "~[slow]" (exclude slow tests)
    - Tags: "[core]" (run tests with core tag)
EOF
}

# Main execution
main() {
    local cleanup_only=false
    local keep_containers=false
    local do_cleanup=true
    local test_filter=""
    local section_filter=""

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
            -t|--test)
                if [[ -z "$2" ]] || [[ "$2" == -* ]]; then
                    print_error "Option -t|--test requires a test filter argument"
                    usage
                    exit 1
                fi
                test_filter="$2"
                shift 2
                ;;
            -s|--section)
                if [[ -z "$2" ]] || [[ "$2" == -* ]]; then
                    print_error "Option -s|--section requires a section filter argument"
                    usage
                    exit 1
                fi
                section_filter="$2"
                shift 2
                ;;
            -*)
                print_error "Unknown option: $1"
                usage
                exit 1
                ;;
            *)
                # Positional argument - treat as test filter
                if [[ -z "$test_filter" ]]; then
                    test_filter="$1"
                    shift
                else
                    print_error "Multiple test filters specified: '$test_filter' and '$1'"
                    usage
                    exit 1
                fi
                ;;
        esac
    done

    # Export test filter as environment variable for docker-compose
    if [[ -n "$test_filter" ]]; then
        export TEST_FILTER="$test_filter"
        print_msg "$BLUE" "Test filter: $test_filter"
    else
        export TEST_FILTER=""
    fi

    # Export section filter as environment variable for docker-compose
    if [[ -n "$section_filter" ]]; then
        export SECTION_FILTER="$section_filter"
        print_msg "$BLUE" "Section filter: $section_filter"
    else
        export SECTION_FILTER=""
    fi

    if [[ -z "$test_filter" ]] && [[ -z "$section_filter" ]]; then
        print_msg "$BLUE" "Running all tests (no filter)"
    fi

    print_header "CTE Distributed Test Runner"

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
    print_msg "$BLUE" "Tests are executing. Waiting for completion..."

    # Wait for test container (node1) to complete and capture exit code
    # Use docker wait which blocks until container exits and returns exit code
    local exit_code=0
    exit_code=$(docker wait cte-distributed-node1 2>/dev/null || echo "1")

    # Show test logs after completion
    print_header "Test Output"
    docker logs cte-distributed-node1 2>&1 | tail -50

    print_header "Test Results"
    if [ "$exit_code" = "0" ]; then
        print_success "All tests passed!"
    else
        print_error "Tests failed with exit code: $exit_code"
    fi

    # Cleanup unless keep flag is set
    if [ "$keep_containers" = true ]; then
        print_warning "Containers kept running for debugging"
        print_msg "$BLUE" "To view logs: docker compose logs"
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
