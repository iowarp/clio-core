#!/bin/bash
################################################################################
# IOWarp Core - Sanitizer Test Runner
#
# Builds the project with sanitizer instrumentation, runs tests via
# CTest's memcheck mode, and submits Dynamic Analysis results to CDash.
#
# Supported sanitizers:
#   ASan  - AddressSanitizer + LeakSanitizer (detect memory corruption & leaks)
#   UBSan - UndefinedBehaviorSanitizer (detect integer overflow, bad casts, etc.)
#   MSan  - MemorySanitizer (detect uninitialized reads; needs instrumented deps)
#   sanitize - ASan + LSan + UBSan combined
#
# Usage:
#   ./CI/run_sanitizers.sh [options]
#
# Options:
#   --asan                Run AddressSanitizer + LeakSanitizer tests
#   --ubsan               Run UndefinedBehaviorSanitizer tests
#   --msan                Run MemorySanitizer tests (requires instrumented deps)
#   --sanitize            Run combined ASan + UBSan tests
#   --all                 Run all sanitizer modes
#   --build               Configure and build before running tests
#   --clean               Clean build directories before building
#   --site SITE_NAME      Submit results to CDash with the given site name
#   --help                Show this help message
#
# Examples:
#   ./CI/run_sanitizers.sh --asan --build --site ubu-24.amd64-asan
#   ./CI/run_sanitizers.sh --all --build --clean --site ci-runner
#   ./CI/run_sanitizers.sh --ubsan --site local   # test only, already built
#
# Note on MSan:
#   MemorySanitizer requires ALL linked libraries (including the C++ standard
#   library) to be compiled with -fsanitize=memory. Without instrumented deps,
#   expect false-positive reports. See the msan preset in CMakePresets.json.
#
################################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Default options
DO_ASAN=false
DO_UBSAN=false
DO_MSAN=false
DO_SANITIZE=false
DO_BUILD=false
CLEAN_BUILD=false
SITE_NAME=""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo -e "${BLUE}======================================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}======================================================================${NC}"
}

print_success() { echo -e "${GREEN}✓ $1${NC}"; }
print_error()   { echo -e "${RED}✗ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠ $1${NC}"; }
print_info()    { echo -e "${BLUE}ℹ $1${NC}"; }

show_help() {
    grep "^#" "$0" | grep -v "#!/bin/bash" | sed 's/^# \?//'
    exit 0
}

################################################################################
# Parse Command Line Arguments
################################################################################

while [[ $# -gt 0 ]]; do
    case $1 in
        --asan)      DO_ASAN=true;     shift ;;
        --ubsan)     DO_UBSAN=true;    shift ;;
        --msan)      DO_MSAN=true;     shift ;;
        --sanitize)  DO_SANITIZE=true; shift ;;
        --all)
            DO_ASAN=true
            DO_UBSAN=true
            DO_MSAN=true
            DO_SANITIZE=true
            shift
            ;;
        --build)     DO_BUILD=true;    shift ;;
        --clean)     CLEAN_BUILD=true; shift ;;
        --site)      SITE_NAME="$2";   shift 2 ;;
        --help|-h)   show_help ;;
        *)
            print_error "Unknown option: $1"
            show_help
            ;;
    esac
done

if [ "$DO_ASAN" = false ] && [ "$DO_UBSAN" = false ] && \
   [ "$DO_MSAN" = false ] && [ "$DO_SANITIZE" = false ]; then
    print_error "No sanitizer mode selected. Use --asan, --ubsan, --msan, --sanitize, or --all."
    show_help
fi

################################################################################
# Core function: build and/or run memcheck for one sanitizer mode
################################################################################

run_sanitizer_mode() {
    local MODE="$1"          # asan | ubsan | msan | sanitize
    local MEMCHECK_TYPE="$2" # AddressSanitizer | UndefinedBehaviorSanitizer | MemorySanitizer
    local SANITIZER_OPTS="$3"
    local BUILD_DIR="${REPO_ROOT}/build-${MODE}"

    print_header "Sanitizer mode: ${MODE} (${MEMCHECK_TYPE})"

    # --- Clean ---
    if [ "$CLEAN_BUILD" = true ] && [ -d "${BUILD_DIR}" ]; then
        print_info "Cleaning ${BUILD_DIR}..."
        rm -rf "${BUILD_DIR}"
        print_success "Cleaned"
    fi

    # --- Build ---
    if [ "$DO_BUILD" = true ]; then
        print_info "Configuring with preset '${MODE}'..."
        if ! cmake --preset="${MODE}" -S "${REPO_ROOT}"; then
            print_error "CMake configure failed for mode '${MODE}'. Check that all dependencies are installed."
            return 1
        fi

        NUM_CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
        print_info "Building (${NUM_CORES} cores)..."
        if ! cmake --build "${BUILD_DIR}" -- -j"${NUM_CORES}"; then
            print_error "CMake build failed for mode '${MODE}'."
            return 1
        fi
        print_success "Build complete"
    else
        if [ ! -d "${BUILD_DIR}" ]; then
            print_error "Build directory ${BUILD_DIR} does not exist. Use --build."
            return 1
        fi
        print_info "Skipping build (using existing ${BUILD_DIR})"
    fi

    # --- Generate CTest dashboard script ---
    local DASHBOARD_SCRIPT="${BUILD_DIR}/cdash_${MODE}.cmake"
    local EFFECTIVE_SITE="${SITE_NAME:-local}"
    local CDASH_BUILD_NAME="${MODE}"

    print_info "Generating CTest dashboard script: ${DASHBOARD_SCRIPT}"

    # Build the submit block only when --site was provided
    local SUBMIT_BLOCK=""
    if [ -n "${SITE_NAME}" ]; then
        SUBMIT_BLOCK="ctest_submit(PARTS MemCheck)"
    fi

    cat > "${DASHBOARD_SCRIPT}" << EOFCMAKE
# Auto-generated by CI/run_sanitizers.sh — do not edit manually
set(CTEST_SITE "${EFFECTIVE_SITE}")
set(CTEST_BUILD_NAME "${CDASH_BUILD_NAME}")
set(CTEST_SOURCE_DIRECTORY "${REPO_ROOT}")
set(CTEST_BINARY_DIRECTORY "${BUILD_DIR}")

set(CTEST_DROP_METHOD "https")
set(CTEST_DROP_SITE "my.cdash.org")
set(CTEST_DROP_LOCATION "/submit.php?project=HERMES")
set(CTEST_DROP_SITE_CDASH TRUE)

set(CTEST_MEMORYCHECK_TYPE "${MEMCHECK_TYPE}")
set(CTEST_MEMORYCHECK_SANITIZER_OPTIONS "${SANITIZER_OPTS}")

ctest_start("Experimental")
ctest_memcheck(
  EXCLUDE_LABEL "manual"
  RETURN_VALUE memcheck_result
)
${SUBMIT_BLOCK}
if(NOT memcheck_result EQUAL 0)
  message(STATUS "Sanitizer defects detected (exit code: \${memcheck_result})")
endif()
EOFCMAKE

    # --- Run ---
    if [ -n "${SITE_NAME}" ]; then
        print_info "Running memcheck and submitting to CDash (site: ${EFFECTIVE_SITE}, build: ${CDASH_BUILD_NAME})..."
    else
        print_info "Running memcheck locally (use --site to submit to CDash)..."
    fi

    cd "${BUILD_DIR}"
    ctest -S "${DASHBOARD_SCRIPT}" -VV || true
    cd "${REPO_ROOT}"

    print_success "Done: ${MODE}"
    echo ""
}

################################################################################
# Main
################################################################################

print_header "IOWarp Core - Sanitizer Test Runner"

if [ -n "${SITE_NAME}" ]; then
    print_info "CDash site: ${SITE_NAME}"
    print_info "CDash server: my.cdash.org (project: HERMES)"
else
    print_warning "No --site specified; CDash submission is disabled"
fi
echo ""

OVERALL_STATUS=0

if [ "$DO_ASAN" = true ]; then
    run_sanitizer_mode \
        "asan" \
        "AddressSanitizer" \
        "detect_leaks=1:halt_on_error=0:print_stacktrace=1:symbolize=1" \
        || OVERALL_STATUS=$?
fi

if [ "$DO_UBSAN" = true ]; then
    run_sanitizer_mode \
        "ubsan" \
        "UndefinedBehaviorSanitizer" \
        "print_stacktrace=1:halt_on_error=0" \
        || OVERALL_STATUS=$?
fi

if [ "$DO_MSAN" = true ]; then
    print_warning "MSan note: false-positives are expected from uninstrumented third-party libraries."
    run_sanitizer_mode \
        "msan" \
        "MemorySanitizer" \
        "print_stacktrace=1:halt_on_error=0" \
        || OVERALL_STATUS=$?
fi

if [ "$DO_SANITIZE" = true ]; then
    run_sanitizer_mode \
        "sanitize" \
        "AddressSanitizer" \
        "detect_leaks=1:halt_on_error=0:print_stacktrace=1:symbolize=1" \
        || OVERALL_STATUS=$?
fi

print_header "Sanitizer Run Complete"
if [ $OVERALL_STATUS -eq 0 ]; then
    print_success "All requested sanitizer modes completed"
else
    print_warning "One or more sanitizer modes reported defects (exit: ${OVERALL_STATUS})"
fi

exit 0
