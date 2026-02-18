#!/bin/bash
#
# run_test.sh - Integration test for Globus data assimilation
#
# This script:
# 1. Starts the Chimaera runtime in the background
# 2. Launches the CTE (Content Transfer Engine)
# 3. Launches the CAE (Content Assimilation Engine)
# 4. Runs wrp_cae_omni to process the OMNI file
#
# Prerequisites:
# - GLOBUS_ACCESS_TOKEN environment variable must be set
# - Globus endpoint must be accessible
# - Required executables must be installed and in PATH
#
# Usage:
#   export GLOBUS_ACCESS_TOKEN="your_token_here"
#   ./run_test.sh

set -e  # Exit on error

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration file
export WRP_CTE_CONF="${SCRIPT_DIR}/wrp_conf.yaml"

# OMNI file
OMNI_FILE="${SCRIPT_DIR}/matsci_globus_omni.yaml"

# Output directory for transferred files
OUTPUT_DIR="/tmp/globus_matsci"

echo "========================================="
echo "Globus Materials Science Integration Test"
echo "========================================="
echo ""

# Check for Globus access token
if [ -z "${GLOBUS_ACCESS_TOKEN}" ]; then
    echo "ERROR: GLOBUS_ACCESS_TOKEN environment variable is not set"
    echo ""
    echo "To obtain a Globus access token:"
    echo "1. Visit: https://app.globus.org/settings/developers"
    echo "2. Create a new app or use an existing one"
    echo "3. Generate a new access token"
    echo "4. Export it: export GLOBUS_ACCESS_TOKEN='your_token_here'"
    echo ""
    exit 1
fi

echo "Configuration:"
echo "  CTE Config:  ${WRP_CTE_CONF}"
echo "  OMNI File:   ${OMNI_FILE}"
echo "  Output Dir:  ${OUTPUT_DIR}"
echo ""

# Create output directory
mkdir -p "${OUTPUT_DIR}"
echo "Created output directory: ${OUTPUT_DIR}"
echo ""

# Start Chimaera runtime in the background
echo "Starting Chimaera runtime..."
chimaera runtime start &
CHIMAERA_PID=$!
echo "Chimaera runtime started (PID: ${CHIMAERA_PID})"
echo ""

# Wait for runtime to initialize
echo "Waiting for runtime to initialize..."
sleep 1
echo ""

# Launch CTE
echo "Launching CTE (Content Transfer Engine)..."
launch_cte
CTE_STATUS=$?
if [ ${CTE_STATUS} -ne 0 ]; then
    echo "ERROR: Failed to launch CTE (exit code: ${CTE_STATUS})"
    kill ${CHIMAERA_PID} 2>/dev/null || true
    exit 1
fi
echo "CTE launched successfully"
echo ""

# Launch CAE
echo "Launching CAE (Content Assimilation Engine)..."
wrp_cae_launch local
echo "CAE launched successfully"
echo ""

# Process OMNI file
echo "Processing OMNI file..."
wrp_cae_omni "${OMNI_FILE}"
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

# Cleanup: Stop Chimaera runtime
echo ""
echo "Stopping Chimaera runtime..."
kill ${CHIMAERA_PID} 2>/dev/null || true
wait ${CHIMAERA_PID} 2>/dev/null || true
echo "Chimaera runtime stopped"

exit ${OMNI_STATUS}
