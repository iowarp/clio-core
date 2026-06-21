#!/bin/bash
#
# run_test.sh - Integration test for Globus data assimilation
#
# This script:
# 1. Starts the CLIO Runtime runtime (with CTE + CAE compose) in the background
# 2. Runs clio_cae to process the OMNI file
#
# Prerequisites:
# - GLOBUS_ACCESS_TOKEN environment variable must be set
# - Globus endpoint must be accessible
# - clio_run and clio_cae must be installed and in PATH
# - Built with -DCAE_ENABLE_GLOBUS=ON
#
# Usage:
#   export GLOBUS_ACCESS_TOKEN="your_token_here"
#   ./run_test.sh

set -e  # Exit on error

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Locate build/bin directory (walk up from repo root)
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
BIN_DIR="${BIN_DIR:-${REPO_ROOT}/build/bin}"

# Runtime configuration (includes compose section for CTE + CAE pools)
RUNTIME_CONF="${RUNTIME_CONF:-${SCRIPT_DIR}/clio_runtime_conf.yaml}"

# OMNI file (override with OMNI_FILE env var)
OMNI_FILE="${OMNI_FILE:-${SCRIPT_DIR}/matsci_globus_omni.yaml}"

# Output directory for transferred files (override with OUTPUT_DIR env var)
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/globus_matsci}"

# Ensure build/bin is on PATH and LD_LIBRARY_PATH so the runtime can
# discover libclio_cae_core_runtime.so / libclio_cte_core_runtime.so.
# Conda iowarp lib must come before base miniconda3/lib so that libcurl's
# OPENSSL_3.2.0 dependency is satisfied by the conda-provided libssl.so.3.
CONDA_IOWARP_LIB="${HOME}/miniconda3/envs/iowarp/lib"
export PATH="${BIN_DIR}:${PATH}"
export LD_LIBRARY_PATH="${BIN_DIR}:${CONDA_IOWARP_LIB}:${LD_LIBRARY_PATH:-}"

echo "========================================="
echo "Globus Materials Science Integration Test"
echo "========================================="
echo ""

# Source saved tokens if not already set
if [ -z "${GLOBUS_ACCESS_TOKEN}" ] && [ -f /tmp/globus_tokens.sh ]; then
    source /tmp/globus_tokens.sh
fi

# Check for Globus access token
if [ -z "${GLOBUS_ACCESS_TOKEN}" ]; then
    echo "ERROR: GLOBUS_ACCESS_TOKEN environment variable is not set"
    echo ""
    echo "To obtain a Globus access token:"
    echo "1. Install globus-sdk: pip install globus-sdk"
    echo "2. Run: python3 ${SCRIPT_DIR}/get_oauth_token.py --client-id YOUR_CLIENT_ID COLLECTION_ID"
    echo "3. Load tokens: source /tmp/globus_tokens.sh"
    echo ""
    exit 1
fi

echo "Configuration:"
echo "  Runtime Config: ${RUNTIME_CONF}"
echo "  OMNI File:      ${OMNI_FILE}"
echo "  Output Dir:     ${OUTPUT_DIR}"
echo ""

# Create output directory
mkdir -p "${OUTPUT_DIR}"
echo "Created output directory: ${OUTPUT_DIR}"
echo ""

# Start CLIO Runtime runtime in the background
# The runtime config contains a compose section that creates both
# CTE (pool 512.0) and CAE (pool 400.0) automatically on startup.
echo "Starting Clio runtime..."
export CLIO_SERVER_CONF="${RUNTIME_CONF}"
clio_run runtime start &
CLIO_PID=$!
echo "Clio runtime started (PID: ${CLIO_PID})"
echo ""

# Wait for runtime to initialize and create compose pools (CTE + CAE).
# The IPC socket appears once the runtime is up; then we wait a further
# grace period for the compose pools to finish registering.
echo "Waiting for runtime to initialize..."
IPC_SOCKET="/tmp/clio_${USER}/clio_9413.ipc"
MAX_WAIT=60
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if [ -e "${IPC_SOCKET}" ]; then
        echo "Runtime socket ready (${WAITED}s)"
        break
    fi
    sleep 1
    WAITED=$((WAITED + 1))
done
if [ $WAITED -ge $MAX_WAIT ]; then
    echo "WARNING: runtime socket did not appear after ${MAX_WAIT}s"
fi
# Grace period for compose pools (CTE + CAE) to finish registering
sleep 5
echo ""

# Process OMNI file
echo "Processing OMNI file..."
clio_cae "${OMNI_FILE}"
OMNI_STATUS=$?

echo ""
if [ ${OMNI_STATUS} -eq 0 ]; then
    echo "========================================="
    echo "Test PASSED"
    echo "========================================="
    echo ""
    echo "Transferred files should be in: ${OUTPUT_DIR}"
    ls -lh "${OUTPUT_DIR}" 2>/dev/null || echo "No files found (transfer may have failed)"
else
    echo "========================================="
    echo "Test FAILED"
    echo "========================================="
    echo ""
    echo "OMNI processing failed with exit code: ${OMNI_STATUS}"
fi

# Cleanup: Stop CLIO Runtime runtime
echo ""
echo "Stopping Clio runtime..."
clio_run runtime stop 2>/dev/null || kill ${CLIO_PID} 2>/dev/null || true
wait ${CLIO_PID} 2>/dev/null || true
echo "Clio runtime stopped"

exit ${OMNI_STATUS}
