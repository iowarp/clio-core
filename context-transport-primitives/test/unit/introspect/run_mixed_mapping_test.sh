#!/bin/bash

# Test script for MapMixedMemory functionality
# Runs rank 0 in background, then rank 1

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Find test binary relative to script location (script is in source tree, binary in build tree)
# The binary is at: ${BUILD_DIR}/bin/test_mixed_mapping
# We determine BUILD_DIR from CMAKE_BINARY_DIR which is passed via environment or derived
if [ -n "$CMAKE_BINARY_DIR" ]; then
    TEST_BINARY="$CMAKE_BINARY_DIR/bin/test_mixed_mapping"
else
    # Default: look for build directory relative to source root
    # Script is at: core/context-transport-primitives/test/unit/introspect/
    # So source root is 4 directories up
    SOURCE_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
    TEST_BINARY="$SOURCE_ROOT/build/bin/test_mixed_mapping"
fi

echo "========================================"
echo "Mixed Mapping Test Script"
echo "========================================"
echo ""

# Check if test binary exists
if [ ! -f "$TEST_BINARY" ]; then
    echo "ERROR: Test binary not found at $TEST_BINARY"
    echo "Please build the test first with: cmake --build build --target test_mixed_mapping"
    exit 1
fi

# Clean up any existing shared memory
echo "Cleaning up any existing shared memory..."
rm -f /dev/shm/test_mixed_mapping 2>/dev/null || true

echo ""
echo "========================================"
echo "Starting Rank 0 (background)"
echo "========================================"
echo ""

# Start rank 0 in background with auto-input
(sleep 5 && echo "") | "$TEST_BINARY" 0 &
RANK0_PID=$!

echo "Rank 0 PID: $RANK0_PID"
echo "Waiting for rank 0 to initialize..."
sleep 2

echo ""
echo "========================================"
echo "Starting Rank 1"
echo "========================================"
echo ""

# Run rank 1
"$TEST_BINARY" 1
RANK1_EXIT=$?

echo ""
echo "========================================"
echo "Waiting for Rank 0 to complete..."
echo "========================================"
echo ""

# Wait for rank 0 to finish
wait $RANK0_PID
RANK0_EXIT=$?

echo ""
echo "========================================"
echo "Test Results"
echo "========================================"
echo "Rank 0 exit code: $RANK0_EXIT"
echo "Rank 1 exit code: $RANK1_EXIT"

if [ $RANK0_EXIT -eq 0 ] && [ $RANK1_EXIT -eq 0 ]; then
    echo ""
    echo "✓ ALL TESTS PASSED"
    echo ""
    exit 0
else
    echo ""
    echo "✗ TESTS FAILED"
    echo ""
    exit 1
fi
