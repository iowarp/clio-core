#!/bin/bash
################################################################################
# IOWarp Core - Code Coverage Calculation Script
#
# This script collects coverage data and generates comprehensive code coverage
# reports. By default, it assumes the project has already been built with
# coverage enabled and tests have been run.
#
# Usage:
#   ./CI/calculate_coverage.sh [options]
#
# Options:
#   --build               Build the project with coverage instrumentation
#   --run-ctest           Run CTest unit tests
#   --run-distributed     Run distributed tests (requires Docker)
#   --all                 Build and run all tests (equivalent to --build --run-ctest --run-distributed)
#   --clean               Clean build directory before starting
#   --help                Show this help message
#
# Examples:
#   ./CI/calculate_coverage.sh              # Just generate reports from existing data
#   ./CI/calculate_coverage.sh --all        # Full build + test + report generation
#   ./CI/calculate_coverage.sh --run-ctest  # Run CTest and generate reports
#
################################################################################

set -e  # Exit on error

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build"

# Default options - skip build and tests by default
DO_BUILD=false
DO_CTEST=false
DO_DISTRIBUTED=false
CLEAN_BUILD=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo -e "${BLUE}======================================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}======================================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

show_help() {
    grep "^#" "$0" | grep -v "#!/bin/bash" | sed 's/^# \?//'
    exit 0
}

################################################################################
# Parse Command Line Arguments
################################################################################

while [[ $# -gt 0 ]]; do
    case $1 in
        --build)
            DO_BUILD=true
            shift
            ;;
        --run-ctest)
            DO_CTEST=true
            shift
            ;;
        --run-distributed)
            DO_DISTRIBUTED=true
            shift
            ;;
        --all)
            DO_BUILD=true
            DO_CTEST=true
            DO_DISTRIBUTED=true
            shift
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --help|-h)
            show_help
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            ;;
    esac
done

################################################################################
# Main Script
################################################################################

print_header "IOWarp Core - Code Coverage Calculation"

# Navigate to repository root
cd "${REPO_ROOT}"

################################################################################
# Step 1: Clean build directory (if requested)
################################################################################

if [ "$CLEAN_BUILD" = true ]; then
    print_info "Cleaning build directory..."
    rm -rf "${BUILD_DIR}"
    print_success "Build directory cleaned"
fi

################################################################################
# Step 2: Configure and build with coverage enabled (optional)
################################################################################

if [ "$DO_BUILD" = true ]; then
    print_header "Step 1: Building with Coverage Instrumentation"

    print_info "Configuring build with coverage enabled..."
    cmake --preset=debug -DWRP_CORE_ENABLE_COVERAGE=ON

    print_info "Building project..."
    NUM_CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    cmake --build "${BUILD_DIR}" -- -j"${NUM_CORES}"

    print_success "Build completed successfully"
else
    print_info "Skipping build step (use --build to enable)"

    if [ ! -d "${BUILD_DIR}" ]; then
        print_error "Build directory does not exist. Use --build option or build manually first."
        exit 1
    fi
fi

################################################################################
# Step 3: Run CTest unit tests (optional)
################################################################################

if [ "$DO_CTEST" = true ]; then
    print_header "Step 2: Running CTest Unit Tests"

    cd "${BUILD_DIR}"

    print_info "Running all CTest tests..."
    ctest --output-on-failure

    CTEST_EXIT_CODE=$?
    if [ $CTEST_EXIT_CODE -eq 0 ]; then
        print_success "All CTest tests passed"
    else
        print_error "Some CTest tests failed (exit code: $CTEST_EXIT_CODE)"
        print_warning "Continuing with coverage generation..."
    fi

    cd "${REPO_ROOT}"
else
    print_info "Skipping CTest tests (use --run-ctest to enable)"
fi

################################################################################
# Step 4: Run distributed tests (optional)
################################################################################

if [ "$DO_DISTRIBUTED" = true ]; then
    print_header "Step 3: Running Distributed Tests"

    # Check if Docker is available
    DOCKER_AVAILABLE=false
    if command -v docker &> /dev/null; then
        if docker ps &> /dev/null 2>&1; then
            DOCKER_AVAILABLE=true
            print_info "Docker is available and running"
        else
            print_warning "Docker is installed but not running"
        fi
    else
        print_warning "Docker is not installed"
    fi

    if [ "$DOCKER_AVAILABLE" = false ]; then
        print_warning "Skipping distributed tests (Docker not available)"
        print_info "Install Docker to enable distributed tests"
    else
        # Find all distributed test directories (in integration folders)
        DISTRIBUTED_DIRS=$(find "${REPO_ROOT}" -type d -path "*/test/integration/distributed" 2>/dev/null)

        if [ -z "$DISTRIBUTED_DIRS" ]; then
            print_warning "No distributed test directories found"
        else
            DIST_TEST_COUNT=0
            DIST_TEST_SUCCESS=0
            DIST_TEST_FAILED=0

            for TEST_DIR in $DISTRIBUTED_DIRS; do
                if [ -f "${TEST_DIR}/run_tests.sh" ]; then
                    COMPONENT_NAME=$(echo "$TEST_DIR" | sed 's|.*/\([^/]*\)/test/integration/distributed|\1|')
                    print_info "Running distributed tests for: ${COMPONENT_NAME}"

                    DIST_TEST_COUNT=$((DIST_TEST_COUNT + 1))

                    cd "${TEST_DIR}"

                    # Check if run_tests.sh is executable
                    if [ ! -x "run_tests.sh" ]; then
                        chmod +x run_tests.sh
                    fi

                    # Set environment variables for Docker volumes
                    export IOWARP_CORE_ROOT="${REPO_ROOT}"
                    export IOWARP_BUILD_DIR="${BUILD_DIR}"

                    # Cleanup any previous containers
                    print_info "Cleaning up any previous test containers..."
                    ./run_tests.sh clean 2>/dev/null || docker compose down -v 2>/dev/null || true

                    # Run the distributed tests
                    print_info "Starting distributed test for ${COMPONENT_NAME}..."
                    print_info "Build directory mounted to containers: ${BUILD_DIR}"
                    if ./run_tests.sh all 2>&1 | tee "${BUILD_DIR}/distributed_test_${COMPONENT_NAME}.log"; then
                        DIST_TEST_SUCCESS=$((DIST_TEST_SUCCESS + 1))
                        print_success "Distributed tests for ${COMPONENT_NAME} completed successfully"
                    else
                        DIST_EXIT_CODE=$?
                        DIST_TEST_FAILED=$((DIST_TEST_FAILED + 1))
                        print_warning "Distributed tests for ${COMPONENT_NAME} failed with exit code: ${DIST_EXIT_CODE}"
                        print_warning "Log saved to: ${BUILD_DIR}/distributed_test_${COMPONENT_NAME}.log"
                        print_warning "Continuing with coverage generation..."
                    fi

                    # Cleanup containers after test
                    print_info "Cleaning up test containers..."
                    ./run_tests.sh clean 2>/dev/null || docker compose down -v 2>/dev/null || true

                    cd "${REPO_ROOT}"
                else
                    print_warning "No run_tests.sh found in ${TEST_DIR}"
                fi
            done

            echo ""
            print_info "Distributed test summary:"
            echo "  Total:   ${DIST_TEST_COUNT}"
            echo "  Success: ${DIST_TEST_SUCCESS}"
            echo "  Failed:  ${DIST_TEST_FAILED}"

            if [ $DIST_TEST_COUNT -gt 0 ]; then
                print_success "Distributed tests completed"
            fi
        fi
    fi
else
    print_info "Skipping distributed tests (use --run-distributed to enable)"
fi

################################################################################
# Step 5: Collect and merge coverage data
################################################################################

print_header "Step 4: Collecting and Merging Coverage Data"

cd "${BUILD_DIR}"

print_info "Capturing final coverage data with lcov (using RC file for comprehensive error handling)..."
# Create temporary lcovrc to handle all known coverage data issues
cat > lcovrc_temp << 'EOFRC'
geninfo_unexecuted_blocks = 1
geninfo_compat_libtool = 1
lcov_branch_coverage = 0
EOFRC

# Use geninfo directly with comprehensive error ignoring for all components at once
# This approach gives more accurate results than capturing from root directory
lcov --capture \
     --directory . \
     --output-file coverage_combined.info \
     --rc lcov_branch_coverage=0 \
     --rc geninfo_unexecuted_blocks=1 \
     --ignore-errors graph,mismatch,negative,inconsistent,unused,empty,gcov,source \
     2>&1 | grep -E "Found [0-9]+ data files|Finished" || true

rm -f lcovrc_temp

if [ ! -f coverage_combined.info ] || [ ! -s coverage_combined.info ]; then
    print_error "Failed to generate coverage data"
    exit 1
fi

# For backward compatibility, also create coverage_all.info
cp coverage_combined.info coverage_all.info

print_success "Coverage data captured and merged"

################################################################################
# Step 6: Filter coverage data
################################################################################

print_header "Step 5: Filtering Coverage Data"

print_info "Filtering out system headers, conda headers, test files, and external dependencies..."
lcov --remove coverage_all.info \
     '/usr/*' \
     '*/test/*' \
     '*/miniconda3/*' \
     '*/conda/*' \
     '*/external/*' \
     '*/catch2/*' \
     '*/nanobind/*' \
     --output-file coverage_filtered.info \
     --ignore-errors mismatch,negative,unused \
     2>&1 | grep -E "Removed|Summary|lines|functions" | tail -5 || true

if [ ! -f coverage_filtered.info ] || [ ! -s coverage_filtered.info ]; then
    print_error "Failed to filter coverage data"
    exit 1
fi

print_success "Coverage data filtered"

################################################################################
# Step 7: Generate HTML report
################################################################################

print_header "Step 6: Generating HTML Coverage Report"

print_info "Generating HTML report..."
genhtml coverage_filtered.info \
        --output-directory coverage_report \
        --ignore-errors mismatch,negative \
        --title "IOWarp Core Coverage Report" \
        --legend \
        2>&1 | grep -E "Overall|Processing|Writing" | tail -10 || true

if [ ! -d coverage_report ]; then
    print_error "Failed to generate HTML report"
    exit 1
fi

print_success "HTML report generated at: ${BUILD_DIR}/coverage_report/index.html"

################################################################################
# Step 8: Generate coverage summary by component
################################################################################

print_header "Step 7: Generating Component Coverage Summary"

# Create temporary files for component coverage
TMP_DIR=$(mktemp -d)

# Extract coverage for each component
print_info "Extracting component coverage..."

# CTP is mostly header-only, so include both src/ and include/
lcov --extract coverage_filtered.info \
     '/workspace/context-transport-primitives/src/*' \
     '/workspace/context-transport-primitives/include/*' \
     --output-file "${TMP_DIR}/ctp.info" \
     --ignore-errors mismatch,negative,unused >/dev/null 2>&1 || true

lcov --extract coverage_filtered.info \
     '/workspace/context-runtime/src/*' \
     '/workspace/context-runtime/include/*' \
     '/workspace/context-runtime/modules/*/src/*' \
     '/workspace/context-runtime/modules/*/include/*' \
     --output-file "${TMP_DIR}/runtime.info" \
     --ignore-errors mismatch,negative,unused >/dev/null 2>&1 || true

lcov --extract coverage_filtered.info \
     '/workspace/context-transfer-engine/core/src/*' \
     '/workspace/context-transfer-engine/core/include/*' \
     --output-file "${TMP_DIR}/cte.info" \
     --ignore-errors mismatch,negative,unused >/dev/null 2>&1 || true

lcov --extract coverage_filtered.info \
     '/workspace/context-assimilation-engine/core/src/*' \
     '/workspace/context-assimilation-engine/core/include/*' \
     --output-file "${TMP_DIR}/cae.info" \
     --ignore-errors mismatch,negative,unused >/dev/null 2>&1 || true

lcov --extract coverage_filtered.info \
     '/workspace/context-exploration-engine/api/src/*' \
     '/workspace/context-exploration-engine/api/include/*' \
     --output-file "${TMP_DIR}/cee.info" \
     --ignore-errors mismatch,negative,unused >/dev/null 2>&1 || true

################################################################################
# Step 9: Generate summary report
################################################################################

print_header "Step 8: Generating Coverage Summary Report"

cat > "${BUILD_DIR}/COVERAGE_SUMMARY.txt" << 'EOFSUM'
################################################################################
# IOWarp Core - Code Coverage Summary
################################################################################

Generated: $(date)
Build Directory: ${BUILD_DIR}

EOFSUM

echo "" >> "${BUILD_DIR}/COVERAGE_SUMMARY.txt"
echo "=== Overall Coverage ===" >> "${BUILD_DIR}/COVERAGE_SUMMARY.txt"
lcov --summary coverage_filtered.info --ignore-errors mismatch,negative 2>&1 | \
    grep -E "lines|functions" >> "${BUILD_DIR}/COVERAGE_SUMMARY.txt"

echo "" >> "${BUILD_DIR}/COVERAGE_SUMMARY.txt"
echo "=== Component Coverage ===" >> "${BUILD_DIR}/COVERAGE_SUMMARY.txt"

for COMPONENT in ctp runtime cte cae cee; do
    case $COMPONENT in
        ctp)
            NAME="Context Transport Primitives (HSHM)"
            ;;
        runtime)
            NAME="Context Runtime (Chimaera)"
            ;;
        cte)
            NAME="Context Transfer Engine (CTE)"
            ;;
        cae)
            NAME="Context Assimilation Engine (CAE)"
            ;;
        cee)
            NAME="Context Exploration Engine (CEE)"
            ;;
    esac

    if [ -f "${TMP_DIR}/${COMPONENT}.info" ] && [ -s "${TMP_DIR}/${COMPONENT}.info" ]; then
        echo "" >> "${BUILD_DIR}/COVERAGE_SUMMARY.txt"
        echo "${NAME}:" >> "${BUILD_DIR}/COVERAGE_SUMMARY.txt"
        lcov --summary "${TMP_DIR}/${COMPONENT}.info" --ignore-errors mismatch,negative 2>&1 | \
            grep -E "lines|functions" | sed 's/^/  /' >> "${BUILD_DIR}/COVERAGE_SUMMARY.txt"
    fi
done

# Cleanup temp files
rm -rf "${TMP_DIR}"

# Display summary
cat "${BUILD_DIR}/COVERAGE_SUMMARY.txt"

print_success "Coverage summary saved to: ${BUILD_DIR}/COVERAGE_SUMMARY.txt"

################################################################################
# Final Summary
################################################################################

print_header "Coverage Calculation Complete"

echo ""
print_info "Coverage Reports Generated:"
echo "  - HTML Report:     ${BUILD_DIR}/coverage_report/index.html"
echo "  - Summary Report:  ${BUILD_DIR}/COVERAGE_SUMMARY.txt"
echo "  - Raw Data (all):  ${BUILD_DIR}/coverage_all.info"
echo "  - Raw Data (src):  ${BUILD_DIR}/coverage_filtered.info"
echo ""

print_info "To view the HTML report:"
echo "  firefox ${BUILD_DIR}/coverage_report/index.html"
echo "  # or"
echo "  google-chrome ${BUILD_DIR}/coverage_report/index.html"
echo ""

print_success "All coverage analysis complete!"

exit 0
