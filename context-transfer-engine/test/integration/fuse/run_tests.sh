#!/bin/bash
# FUSE Integration Test Runner
# Launches a Docker container with FUSE support, mounts a CTE-backed
# filesystem, and runs POSIX I/O tests against it.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../" && pwd)"

# Export workspace path for docker-compose
if [ -n "${HOST_WORKSPACE:-}" ]; then
    export IOWARP_CORE_ROOT="${HOST_WORKSPACE}"
elif [ -z "${IOWARP_CORE_ROOT:-}" ]; then
    export IOWARP_CORE_ROOT="${REPO_ROOT}"
fi

cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_msg() { echo -e "${1}${2}${NC}"; }
print_header() {
    echo ""
    print_msg "$BLUE" "=================================="
    print_msg "$BLUE" "$@"
    print_msg "$BLUE" "=================================="
}

check_prerequisites() {
    print_header "Checking Prerequisites"

    if ! command -v docker &>/dev/null; then
        print_msg "$RED" "✗ Docker is not installed"
        exit 1
    fi
    print_msg "$GREEN" "✓ Docker found"

    if ! docker compose version &>/dev/null; then
        print_msg "$RED" "✗ Docker Compose is not installed"
        exit 1
    fi
    print_msg "$GREEN" "✓ Docker Compose found"

    if ! docker ps &>/dev/null; then
        print_msg "$RED" "✗ Docker daemon is not running"
        exit 1
    fi
    print_msg "$GREEN" "✓ Docker daemon is running"

    # Check that /dev/fuse exists on host
    if [ ! -c /dev/fuse ]; then
        print_msg "$RED" "✗ /dev/fuse not available on host"
        exit 1
    fi
    print_msg "$GREEN" "✓ /dev/fuse available"

    # Check that clio_cte_fuse binary exists
    local fuse_bin="${IOWARP_CORE_ROOT}/build/bin/clio_cte_fuse"
    if [ ! -x "$fuse_bin" ]; then
        print_msg "$RED" "✗ clio_cte_fuse not found at $fuse_bin"
        print_msg "$YELLOW" "  Build with: cmake -DCLIO_CTE_ENABLE_FUSE_ADAPTER=ON .. && make clio_cte_fuse"
        exit 1
    fi
    print_msg "$GREEN" "✓ clio_cte_fuse binary found"
}

cleanup() {
    print_header "Cleaning Up"
    if docker compose ps -q 2>/dev/null | grep -q .; then
        docker compose down -v 2>/dev/null || true
        print_msg "$GREEN" "✓ Containers stopped"
    else
        print_msg "$GREEN" "✓ No containers to clean up"
    fi
}

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Run CTE FUSE filesystem integration tests in a Docker container.

OPTIONS:
    -h, --help          Show this help message
    -c, --cleanup-only  Only cleanup previous runs
    -k, --keep          Keep container running after tests
EOF
}

main() {
    local cleanup_only=false
    local keep_containers=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help) usage; exit 0 ;;
            -c|--cleanup-only) cleanup_only=true; shift ;;
            -k|--keep) keep_containers=true; shift ;;
            *) print_msg "$RED" "Unknown option: $1"; usage; exit 1 ;;
        esac
    done

    print_header "CTE FUSE Integration Test Runner"

    check_prerequisites
    cleanup

    if [ "$cleanup_only" = true ]; then
        print_msg "$GREEN" "✓ Cleanup complete"
        exit 0
    fi

    # Start test container
    print_header "Starting FUSE Test Container"
    export HOST_UID=$(id -u)
    export HOST_GID=$(id -g)

    # A CUDA-enabled clio_cte_fuse links libcudart.so.N, which the container
    # image may not carry. The container runs with no GPU driver, so the binary
    # only needs the runtime lib to LOAD (IsDevicePointer falls back to the CPU
    # path). Stage the exact libcudart into build/bin (mounted + first on the
    # container LD_LIBRARY_PATH). No-op for a CPU build.
    local _cte_bin="${IOWARP_CORE_ROOT}/build/bin/clio_cte_fuse"
    if [ -f "$_cte_bin" ]; then
        local _bindir="$(dirname "$_cte_bin")"
        ldd "$_cte_bin" 2>/dev/null | awk '/libcudart/{print $3}' | while read -r _lib; do
            { [ -n "$_lib" ] && [ -f "$_lib" ] && cp -Lu "$_lib" "$_bindir/" 2>/dev/null; } || true
        done
    fi

    docker compose up -d

    # Wait for container to start
    local max_attempts=15
    local attempt=0
    while [ $attempt -lt $max_attempts ]; do
        local total=$(docker compose ps -a -q 2>/dev/null | wc -l)
        if [ "$total" -ge 1 ]; then
            print_msg "$GREEN" "✓ Container started"
            break
        fi
        sleep 1
        attempt=$((attempt + 1))
    done

    print_msg "$BLUE" "Tests executing. Waiting for completion..."

    # Wait for test container to complete
    local exit_code=0
    exit_code=$(docker wait cte-fuse-test 2>/dev/null || echo "1")

    # Show test output
    print_header "Test Output"
    docker logs cte-fuse-test 2>&1 | tail -60

    print_header "Test Results"
    if [ "$exit_code" = "0" ]; then
        print_msg "$GREEN" "✓ All FUSE integration tests passed!"
    else
        print_msg "$RED" "✗ Tests failed with exit code: $exit_code"
    fi

    if [ "$keep_containers" = true ]; then
        print_msg "$YELLOW" "⚠ Container kept running for debugging"
        print_msg "$BLUE" "  Inspect: docker exec -it cte-fuse-test bash"
        print_msg "$BLUE" "  Cleanup: docker compose down -v"
    else
        docker compose down -v 2>/dev/null || true
        print_msg "$GREEN" "✓ Cleanup complete"
    fi

    exit $exit_code
}

main "$@"
