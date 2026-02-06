#!/bin/bash
# Test script for chimaera_compose utility
# Tests that the compose utility can successfully create pools from a YAML configuration

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/../../build"
BIN_DIR="${BUILD_DIR}/bin"
TEST_CONFIG="/tmp/test_chimaera_compose_config.yaml"

# Executables
CHIMAERA_START_RUNTIME="${BIN_DIR}/chimaera_start_runtime"
CHIMAERA_COMPOSE="${BIN_DIR}/chimaera_compose"

echo -e "${YELLOW}=== Chimaera Compose Utility Test ===${NC}"

# Check if executables exist
if [ ! -f "${CHIMAERA_START_RUNTIME}" ]; then
    echo -e "${RED}Error: chimaera_start_runtime not found at ${CHIMAERA_START_RUNTIME}${NC}"
    exit 1
fi

if [ ! -f "${CHIMAERA_COMPOSE}" ]; then
    echo -e "${RED}Error: chimaera_compose not found at ${CHIMAERA_COMPOSE}${NC}"
    exit 1
fi

# Create test configuration file
echo -e "${YELLOW}Creating test configuration file...${NC}"
cat > "${TEST_CONFIG}" << 'EOF'
# Test compose configuration for chimaera_compose utility
workers:
  sched_threads: 2
  slow_threads: 2

memory:
  main_segment_size: 1GB
  client_data_segment_size: 256MB

networking:
  port: 5555

compose:
- mod_name: chimaera_bdev
  pool_name: /tmp/test_compose_util_bdev.dat
  pool_query: dynamic
  pool_id: 300.0
  capacity: 10MB
  bdev_type: file
  io_depth: 16
  alignment: 4096
EOF

echo -e "${GREEN}Test configuration created at ${TEST_CONFIG}${NC}"

# Set environment variable
export CHI_REPO_PATH="${BIN_DIR}"
export CHI_SERVER_CONF="${TEST_CONFIG}"

# Cleanup function
cleanup() {
    echo -e "${YELLOW}Cleaning up...${NC}"
    # Kill all chimaera processes
    pkill -9 -f "chimaera_start_runtime" 2>/dev/null || true
    pkill -9 -f "chimaera_compose" 2>/dev/null || true

    # Clean up test files
    rm -f "${TEST_CONFIG}" 2>/dev/null || true
    rm -f /tmp/test_compose_util_bdev.dat 2>/dev/null || true

    # Clean up shared memory
    rm -f /dev/shm/chi_* 2>/dev/null || true

    sleep 1
    echo -e "${GREEN}Cleanup complete${NC}"
}

# Set trap to cleanup on exit
trap cleanup EXIT INT TERM

# Start chimaera runtime in background
echo -e "${YELLOW}Starting Chimaera runtime...${NC}"
"${CHIMAERA_START_RUNTIME}" &
RUNTIME_PID=$!

# Wait for runtime to initialize
echo -e "${YELLOW}Waiting for runtime to initialize...${NC}"
sleep 2

# Check if runtime is still running
if ! ps -p ${RUNTIME_PID} > /dev/null; then
    echo -e "${RED}Error: Runtime failed to start${NC}"
    exit 1
fi

echo -e "${GREEN}Runtime started successfully (PID: ${RUNTIME_PID})${NC}"

# Run chimaera_compose utility
echo -e "${YELLOW}Running chimaera_compose utility...${NC}"
if "${CHIMAERA_COMPOSE}" "${TEST_CONFIG}"; then
    echo -e "${GREEN}chimaera_compose completed successfully${NC}"
else
    echo -e "${RED}Error: chimaera_compose failed${NC}"
    exit 1
fi

# Verify that the BDev file was created
if [ -f "/tmp/test_compose_util_bdev.dat" ]; then
    echo -e "${GREEN}BDev file created successfully${NC}"
else
    echo -e "${RED}Error: BDev file was not created${NC}"
    exit 1
fi

echo -e "${GREEN}=== All tests passed ===${NC}"
exit 0
