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
#   --site SITE_NAME      Submit results to CDash with the given site name
#   --help                Show this help message
#
# Examples:
#   ./CI/calculate_coverage.sh              # Just generate reports from existing data
#   ./CI/calculate_coverage.sh --all        # Full build + test + report generation
#   ./CI/calculate_coverage.sh --run-ctest  # Run CTest and generate reports
#   ./CI/calculate_coverage.sh --all --site ubu-24.amd64  # Full run + CDash submit
#
################################################################################

set -e  # Exit on error

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build"
echo $BUILD_DIR
# Default options - skip build and tests by default
DO_BUILD=false
DO_CTEST=false
DO_DISTRIBUTED=false
CLEAN_BUILD=false
SITE_NAME=""

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
        --site)
            SITE_NAME="$2"
            shift 2
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

# Set up conda environment paths if available (needed for pkg-config, cmake find)
if [ -n "$CONDA_PREFIX" ]; then
    export CMAKE_PREFIX_PATH="${CONDA_PREFIX}${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
    export PKG_CONFIG_PATH="${CONDA_PREFIX}/lib/pkgconfig:${CONDA_PREFIX}/share/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
    print_info "Using conda prefix: $CONDA_PREFIX"
fi

# Tests excluded from coverage: daemon-spawning and stress tests are too slow
# with coverage instrumentation. Label-based filtering catches CTE functional/query
# and integration tests. Name-based filtering catches bdev (which has no labels).
COVERAGE_EXCLUDE_LABELS="integration|restart|functional|query|stress|docker"
COVERAGE_EXCLUDE_NAMES="cr_bdev_|cr_mpi_|cr_per_process_shm_stress"

# ---------------------------------------------------------------------------
# Phased coverage. When PHASE_CTEST_INCLUDE_LABELS is set, this run is one
# phase of the build-and-test pipeline: it builds with PHASE_CMAKE_ARGS, runs
# ONLY the ctests carrying those labels, and scopes the emitted coverage to
# PHASE_EXTRACT_PATHS (space-separated lcov path globs). Left unset, the script
# keeps its original behavior: the "unit" phase that runs everything except the
# slow/daemon/integration labels above. Each phase uploads under its own
# Codecov flag so Codecov merges them into one report (see codecov.yml).
# ---------------------------------------------------------------------------
PHASE_CMAKE_ARGS="${PHASE_CMAKE_ARGS:-}"
PHASE_CTEST_INCLUDE_LABELS="${PHASE_CTEST_INCLUDE_LABELS:-}"
PHASE_CTEST_EXCLUDE_LABELS="${PHASE_CTEST_EXCLUDE_LABELS:-}"
PHASE_EXTRACT_PATHS="${PHASE_EXTRACT_PATHS:-}"
if [ -n "${PHASE_CTEST_INCLUDE_LABELS}" ]; then
    print_info "Phased run: labels='${PHASE_CTEST_INCLUDE_LABELS}' extra_cmake='${PHASE_CMAKE_ARGS}'"
fi

# Detect lcov version for RC option compatibility.
# lcov 1.x uses 'lcov_branch_coverage' and 'geninfo_unexecuted_blocks';
# lcov 2.x renamed them and treats unknown keys as fatal errors.
LCOV_MAJOR=$(lcov --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1 | cut -d. -f1)
LCOV_MAJOR=${LCOV_MAJOR:-1}
if [ "${LCOV_MAJOR}" -ge 2 ] 2>/dev/null; then
    LCOV_RC_OPTS=(--rc branch_coverage=0)
else
    LCOV_RC_OPTS=(--rc lcov_branch_coverage=0 --rc geninfo_unexecuted_blocks=1)
fi
print_info "Detected lcov version: ${LCOV_MAJOR}.x (using ${LCOV_RC_OPTS[*]})"

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

    # Ensure conda environment paths are visible to cmake/pkg-config
    if [ -n "$CONDA_PREFIX" ]; then
        export CMAKE_PREFIX_PATH="${CONDA_PREFIX}${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
        export PKG_CONFIG_PATH="${CONDA_PREFIX}/lib/pkgconfig:${CONDA_PREFIX}/share/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
        print_info "Using conda prefix: $CONDA_PREFIX"
    fi

    # Ensure build dependencies are available in conda env (coverage does a
    # fresh from-source build which needs development headers / cmake configs)
    if [ -n "$CONDA_PREFIX" ]; then
        print_info "Installing build dependencies into conda env..."
        conda install -y -c conda-forge \
            cmake make pkg-config cereal yaml-cpp zeromq \
            msgpack-c hdf5 catch2 libaio liburing poco 2>&1 | tail -3
    fi

    print_info "Configuring build with coverage enabled..."
    # CMake variables are CLIO_*; the legacy WRP_* names were silently
    # ignored (no_op overrides) after the rebrand, so the debug preset's
    # CLIO_CTE_ENABLE_COMPRESS=ON default leaked through and broke the
    # pkg_check_modules(liblz4 REQUIRED) call when lz4 wasn't in the
    # conda env.
    cmake --preset=debug \
        -DCLIO_CORE_ENABLE_COVERAGE=ON \
        -DCLIO_CORE_ENABLE_CONDA=ON \
        -DCLIO_CORE_ENABLE_DOCKER_CI=OFF \
        -DCLIO_CTE_ENABLE_ADIOS2_ADAPTER=OFF \
        -DCLIO_CTE_ENABLE_COMPRESS=OFF \
        -DCLIO_CORE_ENABLE_GRAY_SCOTT=OFF \
        ${PHASE_CMAKE_ARGS}

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

    # Phase-aware test selection: a phase runs ONLY its labels; the default
    # (unit) run keeps the historical slow/daemon/integration exclusions.
    if [ -n "${PHASE_CTEST_INCLUDE_LABELS}" ]; then
        CTEST_TEST_SELECT="INCLUDE_LABEL \"${PHASE_CTEST_INCLUDE_LABELS}\""
        if [ -n "${PHASE_CTEST_EXCLUDE_LABELS}" ]; then
            CTEST_TEST_SELECT="${CTEST_TEST_SELECT} EXCLUDE_LABEL \"${PHASE_CTEST_EXCLUDE_LABELS}\""
        fi
    else
        CTEST_TEST_SELECT="EXCLUDE_LABEL \"${COVERAGE_EXCLUDE_LABELS}\" EXCLUDE \"${COVERAGE_EXCLUDE_NAMES}\""
    fi

    if [ -n "${SITE_NAME}" ]; then
        print_info "Running tests and submitting to CDash (site: ${SITE_NAME})..."

        # Scope the CDash coverage submission to the subtree this phase owns
        # (PHASE_CDASH_KEEP, e.g. "context-transfer-engine/adapter" for the fuse
        # phase). ctest_coverage() has no include filter and otherwise reports
        # WHOLE-BUILD coverage measured under a partial (labelled) test run, so
        # the fuse adapter's ~97% got diluted into a ~36% whole-repo headline.
        # We exclude every OTHER source component so the CDash phase build
        # reports the phase's own code, mirroring the Codecov extract scoping.
        CDASH_SCOPE_BLOCK=""
        if [ -n "${PHASE_CDASH_KEEP:-}" ]; then
            print_info "Scoping CDash coverage to: ${PHASE_CDASH_KEEP}"
            _excludes=""
            while IFS= read -r comp; do
                [ -z "${comp}" ] && continue
                rel="${comp#"${REPO_ROOT}/"}"
                keep_it=0
                for k in ${PHASE_CDASH_KEEP}; do
                    k="${k%/}"
                    # keep if the component is under a kept path, OR is an
                    # ancestor of one (so parents like context-transfer-engine
                    # are not excluded wholesale).
                    case "${rel}/" in "${k}"/*) keep_it=1;; esac
                    case "${k}/" in "${rel}"/*) keep_it=1;; esac
                done
                [ "${keep_it}" = 0 ] && _excludes="${_excludes}    \".*/${rel}/.*\"\n"
            done < <(
                { find "${REPO_ROOT}" -mindepth 1 -maxdepth 1 -type d \
                       \( -name 'context-*' -o -name 'prediction_server' \);
                  find "${REPO_ROOT}/context-transfer-engine" -mindepth 1 -maxdepth 1 \
                       -type d 2>/dev/null; } | sort -u
            )
            if [ -n "${_excludes}" ]; then
                # Seed CTEST_CUSTOM_COVERAGE_EXCLUDE with the complement of the
                # kept subtree. The build's generated CTestCustom.cmake (read by
                # ctest_coverage()) then appends its own entries (e.g.
                # fuse_cte_main.cc) to this, so both are honoured.
                CDASH_SCOPE_BLOCK=$(printf 'set(CTEST_CUSTOM_COVERAGE_EXCLUDE\n%b)' "${_excludes}")
            fi
        fi

        # Generate CTest dashboard script for CDash submission
        cat > "${BUILD_DIR}/cdash_coverage.cmake" << EOFCMAKE
set(CTEST_SITE "${SITE_NAME}")
set(CTEST_BUILD_NAME "coverage")
set(CTEST_SOURCE_DIRECTORY "${REPO_ROOT}")
set(CTEST_BINARY_DIRECTORY "${BUILD_DIR}")
set(CTEST_DROP_METHOD "https")
set(CTEST_DROP_SITE "my.cdash.org")
set(CTEST_DROP_LOCATION "/submit.php?project=HERMES")
set(CTEST_DROP_SITE_CDASH TRUE)
set(CTEST_COVERAGE_COMMAND "gcov")
${CDASH_SCOPE_BLOCK}
ctest_start("Experimental")
ctest_test(RETURN_VALUE test_result ${CTEST_TEST_SELECT})
ctest_coverage()
ctest_submit()
if(NOT test_result EQUAL 0)
  message("Some tests failed (exit code: \${test_result})")
endif()
EOFCMAKE
        ctest -S "${BUILD_DIR}/cdash_coverage.cmake" -VV || true
        print_success "CDash submission complete"
    else
        CTEST_EXIT_CODE=0
        if [ -n "${PHASE_CTEST_INCLUDE_LABELS}" ]; then
            print_info "Running phase tests (labels: ${PHASE_CTEST_INCLUDE_LABELS}, exclude: ${PHASE_CTEST_EXCLUDE_LABELS:-none})..."
            PHASE_LE_ARGS=()
            [ -n "${PHASE_CTEST_EXCLUDE_LABELS}" ] && PHASE_LE_ARGS=(-LE "${PHASE_CTEST_EXCLUDE_LABELS}")
            ctest --output-on-failure --timeout 120 \
                -L "${PHASE_CTEST_INCLUDE_LABELS}" "${PHASE_LE_ARGS[@]}" || CTEST_EXIT_CODE=$?
        else
            print_info "Running unit tests (excluding slow/daemon tests)..."
            ctest --output-on-failure --timeout 120 \
                -LE "${COVERAGE_EXCLUDE_LABELS}" \
                -E "${COVERAGE_EXCLUDE_NAMES}" || CTEST_EXIT_CODE=$?
        fi
        if [ $CTEST_EXIT_CODE -eq 0 ]; then
            print_success "All CTest tests passed"
        else
            print_error "Some CTest tests failed (exit code: $CTEST_EXIT_CODE)"
            print_warning "Continuing with coverage generation..."
        fi
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

print_info "Capturing final coverage data with lcov..."

# lcov 2.x is stricter about errors than 1.x. Build a comprehensive
# --ignore-errors list so that stale .gcda files, missing sources, or
# version mismatches do not abort the capture.
if [ "${LCOV_MAJOR}" -ge 2 ] 2>/dev/null; then
    # 'gcov' covers the case where a single .gcda is corrupted (partial
    # flush after a crashed/timeout'd test invocation, or a race between
    # two writers on the same .gcda). Without it, lcov fails the entire
    # capture run on one bad file. We'd rather drop that file from the
    # report than fail the whole Build and Test job.
    LCOV_IGNORE_OPTS=(--ignore-errors source,graph,mismatch,empty,unused,negative,count,inconsistent,gcov)
else
    LCOV_IGNORE_OPTS=(--ignore-errors source,graph,gcov)
fi

lcov --capture \
     --directory . \
     --output-file coverage_combined.info \
     "${LCOV_RC_OPTS[@]}" \
     "${LCOV_IGNORE_OPTS[@]}" \
     2>&1 | tee /tmp/lcov_capture.log | grep -E "Found [0-9]+ data files|Finished|Reading" || true

if [ ! -f coverage_combined.info ] || [ ! -s coverage_combined.info ]; then
    print_error "Failed to generate coverage data"
    print_info "lcov capture log (last 20 lines):"
    tail -20 /tmp/lcov_capture.log 2>/dev/null || true
fi

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
     '*/_deps/*' \
     '*/benchmark/*' \
     '*/local_sched.cc' \
     '*/globus_file_assimilator.cc' \
     '*/fuse_cte_main.cc' \
     --output-file coverage_filtered.info \
     "${LCOV_IGNORE_OPTS[@]}" \
     2>&1 | grep -E "Removed|Summary|lines|functions" | tail -5 || true

if [ ! -f coverage_filtered.info ] || [ ! -s coverage_filtered.info ]; then
    print_error "Failed to filter coverage data"
    exit 1
fi

print_success "Coverage data filtered"

# Phased coverage: scope this phase's report to only the code it exercises, so
# each Codecov flag reports coverage for its own component (not the whole tree).
if [ -n "${PHASE_EXTRACT_PATHS}" ]; then
    print_info "Scoping coverage to phase paths: ${PHASE_EXTRACT_PATHS}"
    # shellcheck disable=SC2086  # intentional word-splitting of the glob list
    lcov --extract coverage_filtered.info ${PHASE_EXTRACT_PATHS} \
         --output-file coverage_phase.info \
         "${LCOV_IGNORE_OPTS[@]}" \
         2>&1 | grep -E "Removed|Summary|lines|functions" | tail -3 || true
    if [ -f coverage_phase.info ] && [ -s coverage_phase.info ]; then
        mv coverage_phase.info coverage_filtered.info
        print_success "Coverage scoped to phase paths"
    else
        print_warning "Phase extract produced no data; keeping full filtered set"
    fi
fi

################################################################################
# Step 7: Generate HTML report
################################################################################

print_header "Step 6: Generating HTML Coverage Report"

print_info "Generating HTML report..."
genhtml coverage_filtered.info \
        --output-directory coverage_report \
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
     "${REPO_ROOT}/context-transport-primitives/src/*" \
     "${REPO_ROOT}/context-transport-primitives/include/*" \
     --output-file "${TMP_DIR}/ctp.info" \
     >/dev/null 2>&1 || true

lcov --extract coverage_filtered.info \
     "${REPO_ROOT}/context-runtime/src/*" \
     "${REPO_ROOT}/context-runtime/include/*" \
     "${REPO_ROOT}/context-runtime/modules/*/src/*" \
     "${REPO_ROOT}/context-runtime/modules/*/include/*" \
     --output-file "${TMP_DIR}/runtime.info" \
     >/dev/null 2>&1 || true

lcov --extract coverage_filtered.info \
     "${REPO_ROOT}/context-transfer-engine/core/src/*" \
     "${REPO_ROOT}/context-transfer-engine/core/include/*" \
     --output-file "${TMP_DIR}/cte.info" \
     >/dev/null 2>&1 || true

lcov --extract coverage_filtered.info \
     "${REPO_ROOT}/context-assimilation-engine/core/src/*" \
     "${REPO_ROOT}/context-assimilation-engine/core/include/*" \
     --output-file "${TMP_DIR}/cae.info" \
     >/dev/null 2>&1 || true

lcov --extract coverage_filtered.info \
     "${REPO_ROOT}/context-exploration-engine/api/src/*" \
     "${REPO_ROOT}/context-exploration-engine/api/include/*" \
     --output-file "${TMP_DIR}/cee.info" \
     >/dev/null 2>&1 || true

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
lcov --summary coverage_filtered.info 2>&1 | \
    grep -E "lines|functions" >> "${BUILD_DIR}/COVERAGE_SUMMARY.txt"

echo "" >> "${BUILD_DIR}/COVERAGE_SUMMARY.txt"
echo "=== Component Coverage ===" >> "${BUILD_DIR}/COVERAGE_SUMMARY.txt"

for COMPONENT in ctp runtime cte cae cee; do
    case $COMPONENT in
        ctp)
            NAME="Context Transport Primitives (CTP)"
            ;;
        runtime)
            NAME="Context Runtime (Clio)"
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
        lcov --summary "${TMP_DIR}/${COMPONENT}.info" 2>&1 | \
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
