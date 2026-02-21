#!/bin/bash
#
# WRP CTE Benchmark Script
#
# This benchmark measures the performance of Put, Get, and combined operations
# in the Content Transfer Engine (CTE) with RAM-only storage configuration.
#
# Usage:
#   ./wrp_cte_bench.sh <test_case> <num_procs> <depth> <io_size> <io_count>
#
# Parameters:
#   test_case: Benchmark to conduct (Put, Get, PutGet)
#   num_procs: Number of MPI processes
#   depth: Number of async requests to generate
#   io_size: Size of I/O operations in bytes (supports k/K, m/M, g/G suffixes)
#   io_count: Number of I/O operations per process
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Script directory and paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${WRP_RUNTIME_CONF:-$SCRIPT_DIR/cte_config_ram.yaml}"
BENCHMARK_EXE="wrp_cte_bench"

# Parse size string with k/K, m/M, g/G suffixes
parse_size() {
    local size_str="$1"
    local num=""
    local suffix=""

    # Extract number and suffix
    if [[ $size_str =~ ^([0-9]+)([kKmMgG]?)$ ]]; then
        num="${BASH_REMATCH[1]}"
        suffix="${BASH_REMATCH[2]}"
    else
        echo "Error: Invalid size format: $size_str" >&2
        return 1
    fi

    local multiplier=1
    case "${suffix,,}" in
        k) multiplier=1024 ;;
        m) multiplier=$((1024 * 1024)) ;;
        g) multiplier=$((1024 * 1024 * 1024)) ;;
        *) multiplier=1 ;;
    esac

    echo $((num * multiplier))
}

# Format size to human-readable string
format_size() {
    local bytes=$1
    if [ $bytes -ge $((1024 * 1024 * 1024)) ]; then
        echo "$((bytes / (1024 * 1024 * 1024))) GB"
    elif [ $bytes -ge $((1024 * 1024)) ]; then
        echo "$((bytes / (1024 * 1024))) MB"
    elif [ $bytes -ge 1024 ]; then
        echo "$((bytes / 1024)) KB"
    else
        echo "$bytes B"
    fi
}

# Print usage
print_usage() {
    echo "Usage: $0 <test_case> <num_procs> <depth> <io_size> <io_count>"
    echo ""
    echo "Parameters:"
    echo "  test_case: Benchmark to conduct (Put, Get, PutGet)"
    echo "  num_procs: Number of MPI processes (e.g., 1, 4, 8)"
    echo "  depth: Number of async requests to generate (e.g., 4)"
    echo "  io_size: Size of I/O operations (e.g., 1m, 4k, 1g)"
    echo "  io_count: Number of I/O operations per process (e.g., 100)"
    echo ""
    echo "Examples:"
    echo "  $0 Put 1 4 1m 100"
    echo "  $0 Get 4 8 4k 1000"
    echo "  $0 PutGet 2 4 1m 100"
}

# Main script
main() {
    # Check arguments
    if [ $# -ne 5 ]; then
        print_usage
        exit 1
    fi

    local test_case="$1"
    local num_procs="$2"
    local depth="$3"
    local io_size_str="$4"
    local io_count="$5"

    # Parse I/O size
    local io_size=$(parse_size "$io_size_str")
    if [ $? -ne 0 ]; then
        exit 1
    fi

    # Validate parameters
    if [ $num_procs -le 0 ] || [ $depth -le 0 ] || [ $io_size -le 0 ] || [ $io_count -le 0 ]; then
        echo -e "${RED}Error: Invalid parameters${NC}" >&2
        echo "  num_procs must be > 0" >&2
        echo "  depth must be > 0" >&2
        echo "  io_size must be > 0" >&2
        echo "  io_count must be > 0" >&2
        exit 1
    fi

    # Validate test case
    case "${test_case,,}" in
        put|get|putget)
            ;;
        *)
            echo -e "${RED}Error: Unknown test case: $test_case${NC}" >&2
            echo "Valid options: Put, Get, PutGet" >&2
            exit 1
            ;;
    esac

    # Check if configuration file exists
    if [ ! -f "$CONFIG_FILE" ]; then
        echo -e "${RED}Error: Configuration file not found: $CONFIG_FILE${NC}" >&2
        exit 1
    fi

    # Check if mpirun is available
    if ! command -v mpirun &> /dev/null; then
        echo -e "${RED}Error: mpirun not found${NC}" >&2
        echo "Please install MPI (e.g., OpenMPI or MPICH)" >&2
        exit 1
    fi

    # Print benchmark configuration
    echo "=== WRP CTE Benchmark ==="
    echo "Test case: $test_case"
    echo "MPI processes: $num_procs"
    echo "Async depth: $depth"
    echo "I/O size: $(format_size $io_size)"
    echo "Operations per process: $io_count"
    echo "Total I/O per process: $(format_size $((io_size * io_count)))"
    echo "Total I/O across all processes: $(format_size $((io_size * io_count * num_procs)))"
    echo "Configuration: $CONFIG_FILE (RAM-only storage)"
    echo "============================"
    echo ""

    # Set environment variables
    export CHI_WITH_RUNTIME=1
    export WRP_RUNTIME_CONF="$CONFIG_FILE"

    # Run benchmark with mpirun
    echo -e "${GREEN}Starting benchmark...${NC}"
    echo ""

    mpirun -x WRP_RUNTIME_CONF -x CHI_WITH_RUNTIME -n $num_procs "$BENCHMARK_EXE" "$test_case" "$depth" "$io_size_str" "$io_count"

    local exit_code=$?

    echo ""
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}Benchmark completed successfully${NC}"
    else
        echo -e "${RED}Benchmark failed with exit code: $exit_code${NC}" >&2
        exit $exit_code
    fi
}

# Run main function
main "$@"
