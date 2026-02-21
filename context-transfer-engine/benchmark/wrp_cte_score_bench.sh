#!/bin/bash
# CTE Score/Demotion Benchmark Runner Script
#
# This script runs the MPI-based score/demotion benchmark that compares
# different percentages of blob demotion (0%, 10%, 50%, 70%, 100%).
#
# Usage:
#   ./wrp_cte_score_bench.sh <num_procs> <data_per_rank_step> <busy_wait_sec> <num_steps> [config_file]
#
# Parameters:
#   num_procs:          Number of MPI processes
#   data_per_rank_step: Data per rank per step (e.g., 100m, 1g)
#   busy_wait_sec:      Busy wait time per step in seconds
#   num_steps:          Number of steps to run
#   config_file:        Optional path to configuration file (default: cte_score_bench_config.yaml)
#
# Example:
#   ./wrp_cte_score_bench.sh 4 100m 0.5 10

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check arguments
if [ $# -lt 4 ]; then
    echo -e "${RED}Error: Missing arguments${NC}"
    echo "Usage: $0 <num_procs> <data_per_rank_step> <busy_wait_sec> <num_steps> [config_file]"
    echo ""
    echo "Parameters:"
    echo "  num_procs:          Number of MPI processes"
    echo "  data_per_rank_step: Data per rank per step (e.g., 100m, 1g)"
    echo "  busy_wait_sec:      Busy wait time per step in seconds"
    echo "  num_steps:          Number of steps to run"
    echo "  config_file:        Optional path to config file"
    echo ""
    echo "Example:"
    echo "  $0 4 100m 0.5 10"
    exit 1
fi

NUM_PROCS=$1
DATA_PER_RANK_STEP=$2
BUSY_WAIT_SEC=$3
NUM_STEPS=$4
CONFIG_FILE="${5:-${SCRIPT_DIR}/cte_score_bench_config.yaml}"

# Find the benchmark executable
BENCHMARK_EXE=""
if [ -x "${SCRIPT_DIR}/../../build/bin/wrp_cte_score_bench" ]; then
    BENCHMARK_EXE="${SCRIPT_DIR}/../../build/bin/wrp_cte_score_bench"
elif [ -x "$(which wrp_cte_score_bench 2>/dev/null)" ]; then
    BENCHMARK_EXE="$(which wrp_cte_score_bench)"
else
    echo -e "${RED}Error: Cannot find wrp_cte_score_bench executable${NC}"
    echo "Please ensure the benchmark is built and installed."
    exit 1
fi

# Validate config file
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Configuration file not found: ${CONFIG_FILE}${NC}"
    exit 1
fi

# Create temporary directory for NVMe storage
NVME_DIR="/tmp/cte_nvme_tier"
mkdir -p "$NVME_DIR"

echo -e "${GREEN}=== CTE Score/Demotion Benchmark ===${NC}"
echo -e "Configuration:"
echo -e "  MPI processes:      ${YELLOW}${NUM_PROCS}${NC}"
echo -e "  Data/rank/step:     ${YELLOW}${DATA_PER_RANK_STEP}${NC}"
echo -e "  Busy wait:          ${YELLOW}${BUSY_WAIT_SEC} seconds${NC}"
echo -e "  Steps:              ${YELLOW}${NUM_STEPS}${NC}"
echo -e "  Config file:        ${YELLOW}${CONFIG_FILE}${NC}"
echo -e "  Benchmark exe:      ${YELLOW}${BENCHMARK_EXE}${NC}"
echo ""

# Set environment variables
export WRP_RUNTIME_CONF="$CONFIG_FILE"
export CHI_WITH_RUNTIME=1

# Run the benchmark
echo -e "${GREEN}Running benchmark...${NC}"
echo ""

mpirun --allow-run-as-root \
    -x WRP_RUNTIME_CONF \
    -x CHI_WITH_RUNTIME \
    -n "$NUM_PROCS" \
    "$BENCHMARK_EXE" "$DATA_PER_RANK_STEP" "$BUSY_WAIT_SEC" "$NUM_STEPS"

# Cleanup
echo ""
echo -e "${GREEN}Cleaning up temporary storage...${NC}"
rm -rf "$NVME_DIR"

echo -e "${GREEN}Benchmark complete!${NC}"
