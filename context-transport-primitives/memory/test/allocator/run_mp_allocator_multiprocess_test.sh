#!/bin/bash

# Multi-process test script for MultiProcessAllocator
# This script orchestrates multiple processes accessing shared memory concurrently

set -e  # Exit on any error

# Configuration
TEST_BINARY="${1:-./test_mp_allocator_multiprocess}"
DURATION=5
NTHREADS=2

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Multi-Process Allocator Test"
echo "=========================================="
echo "Test binary: $TEST_BINARY"
echo "Duration: ${DURATION}s"
echo "Threads per process: $NTHREADS"
echo ""

# Check if test binary exists
if [ ! -f "$TEST_BINARY" ]; then
    echo -e "${RED}Error: Test binary not found: $TEST_BINARY${NC}"
    echo "Please build the test first or provide the correct path as argument"
    exit 1
fi

# Cleanup function
cleanup() {
    echo ""
    echo "Cleaning up background processes..."
    # Kill any remaining background jobs
    jobs -p | xargs -r kill 2>/dev/null || true
    wait 2>/dev/null || true
}

# Set trap to cleanup on exit
trap cleanup EXIT INT TERM

# Use PID-unique log files to avoid conflicts when tests run in parallel
LOG0="/tmp/mp_test_rank0_$$.log"
LOG1="/tmp/mp_test_rank1_$$.log"
LOG2="/tmp/mp_test_rank2_$$.log"

# Start all three ranks in background
echo -e "${YELLOW}[Step 1]${NC} Starting rank 0 (${DURATION}s, ${NTHREADS} threads) in background..."
"$TEST_BINARY" 0 $DURATION $NTHREADS > "$LOG0" 2>&1 &
RANK0_PID=$!
echo "Rank 0 PID: $RANK0_PID"

# Delay to ensure rank 0 initializes shared memory before other ranks attach
sleep 2

echo -e "${YELLOW}[Step 2]${NC} Starting rank 1 (${DURATION}s, ${NTHREADS} threads) in background..."
"$TEST_BINARY" 1 $DURATION $NTHREADS > "$LOG1" 2>&1 &
RANK1_PID=$!
echo "Rank 1 PID: $RANK1_PID"

echo -e "${YELLOW}[Step 3]${NC} Starting rank 2 (${DURATION}s, ${NTHREADS} threads) in background..."
"$TEST_BINARY" 2 $DURATION $NTHREADS > "$LOG2" 2>&1 &
RANK2_PID=$!
echo "Rank 2 PID: $RANK2_PID"
echo ""

# Wait for all processes and capture exit codes
echo "Waiting for tests to complete..."
echo ""

RANK0_EXIT=0
RANK1_EXIT=0
RANK2_EXIT=0

# Wait for rank 0
if wait $RANK0_PID; then
    echo -e "${GREEN}✓ Rank 0 completed successfully${NC}"
else
    RANK0_EXIT=$?
    echo -e "${RED}✗ Rank 0 failed with exit code $RANK0_EXIT${NC}"
fi

# Wait for rank 1
if wait $RANK1_PID; then
    echo -e "${GREEN}✓ Rank 1 completed successfully${NC}"
else
    RANK1_EXIT=$?
    echo -e "${RED}✗ Rank 1 failed with exit code $RANK1_EXIT${NC}"
fi

# Wait for rank 2
if wait $RANK2_PID; then
    echo -e "${GREEN}✓ Rank 2 completed successfully${NC}"
else
    RANK2_EXIT=$?
    echo -e "${RED}✗ Rank 2 failed with exit code $RANK2_EXIT${NC}"
fi

echo ""
echo "=========================================="
echo "Test Results"
echo "=========================================="

# Display logs
echo ""
echo "--- Rank 0 Output ---"
cat "$LOG0"
echo ""
echo "--- Rank 1 Output ---"
cat "$LOG1"
echo ""
echo "--- Rank 2 Output ---"
cat "$LOG2"
echo ""

# Final result
if [ $RANK0_EXIT -eq 0 ] && [ $RANK1_EXIT -eq 0 ] && [ $RANK2_EXIT -eq 0 ]; then
    echo -e "${GREEN}=========================================="
    echo -e "ALL TESTS PASSED"
    echo -e "==========================================${NC}"

    # Cleanup log files
    rm -f "$LOG0" "$LOG1" "$LOG2"

    exit 0
else
    echo -e "${RED}=========================================="
    echo -e "TESTS FAILED"
    echo -e "==========================================${NC}"
    echo "Rank 0 exit code: $RANK0_EXIT"
    echo "Rank 1 exit code: $RANK1_EXIT"
    echo "Rank 2 exit code: $RANK2_EXIT"
    echo ""
    echo "Log files preserved:"
    echo "  - $LOG0"
    echo "  - $LOG1"
    echo "  - $LOG2"

    exit 1
fi
