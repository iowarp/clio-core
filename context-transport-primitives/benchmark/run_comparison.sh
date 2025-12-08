#!/bin/bash
# Allocator Benchmark Comparison Script
#
# This script runs the same workload across all three allocators
# to compare their performance.

if [ $# -lt 4 ]; then
    echo "Usage: $0 <num_threads> <min_size> <max_size> <duration_sec>"
    echo ""
    echo "Example: $0 8 4K 1M 10"
    echo "  Runs all allocators with 8 threads, 4KB-1MB size range, for 10 seconds each"
    exit 1
fi

THREADS=$1
MIN_SIZE=$2
MAX_SIZE=$3
DURATION=$4

# Find the benchmark executable
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_BIN="${SCRIPT_DIR}/../build/bin/allocator_benchmark"

if [ ! -x "$BENCHMARK_BIN" ]; then
    # Try alternative path
    BENCHMARK_BIN="$(pwd)/build/bin/allocator_benchmark"
    if [ ! -x "$BENCHMARK_BIN" ]; then
        echo "Error: Cannot find allocator_benchmark executable"
        echo "Tried: ${SCRIPT_DIR}/../build/bin/allocator_benchmark"
        echo "Tried: $(pwd)/build/bin/allocator_benchmark"
        exit 1
    fi
fi

echo "========================================"
echo "Allocator Benchmark Comparison"
echo "========================================"
echo "Configuration:"
echo "  Threads:    $THREADS"
echo "  Size Range: $MIN_SIZE - $MAX_SIZE"
echo "  Duration:   $DURATION seconds"
echo ""

echo "Running MultiProcessAllocator..."
"$BENCHMARK_BIN" mp "$THREADS" "$MIN_SIZE" "$MAX_SIZE" "$DURATION"
echo ""

echo "Running BuddyAllocator..."
"$BENCHMARK_BIN" buddy "$THREADS" "$MIN_SIZE" "$MAX_SIZE" "$DURATION"
echo ""

echo "Running Standard malloc..."
"$BENCHMARK_BIN" malloc "$THREADS" "$MIN_SIZE" "$MAX_SIZE" "$DURATION"
echo ""

echo "========================================"
echo "Comparison complete!"
echo "========================================"
