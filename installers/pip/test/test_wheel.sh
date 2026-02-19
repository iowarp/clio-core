#!/bin/bash
# =============================================================================
# test_wheel.sh - Build and test pip wheel locally via Docker
# =============================================================================
# Replicates the full .github/workflows/build-pip.yml pipeline:
#
#   Phase 1 (build-wheels): Build the wheel in manylinux_2_34, same as
#       cibuildwheel. Installs deps from source, builds with scikit-build-core,
#       fixes RPATHs, and verifies static linking.
#
#   Phase 2 (test-wheels): Install the wheel in clean distro containers
#       and verify import, CLI, and symbol resolution — identical to CI.
#
# Note: The local build produces a single wheel for cp312. Distros whose
# system Python doesn't match are skipped. The CI builds wheels for
# cp310-cp313 via cibuildwheel, which covers all distros.
#
# Prerequisites:
#   - Docker
#
# Usage (from any directory):
#   bash installers/pip/test/test_wheel.sh [--skip-build] [--distro <image>]
#
# Options:
#   --skip-build    Skip the build phase, use existing wheel in wheelhouse/
#   --distro IMAGE  Test only this distro (e.g. ubuntu:24.04)
#   --build-only    Only build the wheel, skip testing in distro containers
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
WHEELHOUSE="$SCRIPT_DIR/wheelhouse"

SKIP_BUILD=false
BUILD_ONLY=false
# Default distros matching CI matrix. ubuntu:24.04 and fedora:40 have cp312;
# ubuntu:22.04 (cp310) and debian:12 (cp311) need their own wheels.
DISTROS=("ubuntu:22.04" "ubuntu:24.04" "debian:12" "fedora:40")
CUSTOM_DISTRO=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --build-only)
            BUILD_ONLY=true
            shift
            ;;
        --distro)
            CUSTOM_DISTRO="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--skip-build] [--build-only] [--distro <image>]"
            exit 1
            ;;
    esac
done

if [[ -n "$CUSTOM_DISTRO" ]]; then
    DISTROS=("$CUSTOM_DISTRO")
fi

echo "======================================================================"
echo "IOWarp Pip Wheel Test"
echo "======================================================================"
echo "  Project root: $PROJECT_ROOT"
echo "  Wheelhouse:   $WHEELHOUSE"
echo ""

# ==================================================================
# Phase 1: Build the wheel in manylinux_2_34
# ==================================================================
if [[ "$SKIP_BUILD" == "false" ]]; then
    echo ">>> Phase 1: Building wheel in manylinux_2_34..."
    echo ""

    mkdir -p "$WHEELHOUSE"
    rm -f "$WHEELHOUSE"/*.whl

    cd "$PROJECT_ROOT"
    docker build --progress=plain \
        -t iowarp/pip-build:latest \
        -f installers/pip/test/Dockerfile.build .

    echo ""
    echo ">>> Extracting wheel..."
    CONTAINER_ID=$(docker create iowarp/pip-build:latest)
    docker cp "$CONTAINER_ID:/tmp/wheelhouse/." "$WHEELHOUSE/"
    docker rm "$CONTAINER_ID" >/dev/null

    echo ""
    echo "Built wheel(s):"
    ls -lh "$WHEELHOUSE"/*.whl
    echo ""
else
    echo ">>> Phase 1: Skipped (--skip-build)"
    if ! ls "$WHEELHOUSE"/*.whl &>/dev/null; then
        echo "ERROR: No wheels found in $WHEELHOUSE"
        exit 1
    fi
    echo "Using existing wheel(s):"
    ls -lh "$WHEELHOUSE"/*.whl
    echo ""
fi

if [[ "$BUILD_ONLY" == "true" ]]; then
    echo ">>> Phase 2: Skipped (--build-only)"
    echo ""
    echo "======================================================================"
    echo "Build complete!"
    echo "======================================================================"
    exit 0
fi

# ==================================================================
# Phase 2: Test in clean distro containers
#
# Identical to the CI test-wheels matrix job: install wheel with only
# Python + pip, verify import/CLI/symbol resolution.
# ==================================================================
echo ">>> Phase 2: Testing wheel in clean distro containers..."
echo ""

PASSED=()
FAILED=()
SKIPPED=()
for DISTRO in "${DISTROS[@]}"; do
    echo "----------------------------------------------------------------------"
    echo "Testing in: $DISTRO"
    echo "----------------------------------------------------------------------"

    # Inline test script — mirrors .github/workflows/build-pip.yml test-wheels
    RESULT=$(docker run --rm \
        -v "$WHEELHOUSE:/wheels:ro" \
        "$DISTRO" \
        bash -c '
            set -e

            # Install Python
            if command -v apt-get &>/dev/null; then
                apt-get update -qq
                apt-get install -y -qq python3 python3-pip python3-venv binutils >/dev/null 2>&1
            elif command -v dnf &>/dev/null; then
                dnf install -y -q python3 python3-pip binutils >/dev/null 2>&1
            elif command -v yum &>/dev/null; then
                yum install -y -q python3 python3-pip binutils >/dev/null 2>&1
            fi

            # Create clean venv
            python3 -m venv /tmp/test-venv
            source /tmp/test-venv/bin/activate

            # Install the wheel matching this Python version
            if ! pip install --quiet --no-index --find-links /wheels iowarp-core 2>&1; then
                echo "SKIP: No compatible wheel for this Python version"
                exit 99
            fi

            echo "=== Import test ==="
            python -c "import iowarp_core; print(\"Version:\", iowarp_core.get_version())"
            python -c "import iowarp_core; print(\"Lib dir:\", iowarp_core.get_lib_dir())"
            python -c "import iowarp_core; print(\"CTE available:\", iowarp_core.cte_available())"

            echo "=== CLI test ==="
            chimaera --help

            echo "=== Symbol resolution check ==="
            LIB_DIR=$(python -c "import iowarp_core; print(iowarp_core.get_lib_dir())")
            export LD_LIBRARY_PATH="$LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            MISSING=0
            for so in "$LIB_DIR"/*.so "$LIB_DIR"/*.so.*; do
                [ -f "$so" ] || continue
                if ldd "$so" 2>/dev/null | grep -q "not found"; then
                    echo "FAIL: $(basename $so) has missing symbols:"
                    ldd "$so" | grep "not found"
                    MISSING=1
                fi
            done
            if [ "$MISSING" = "1" ]; then
                echo "ERROR: Some libraries have unresolved symbols"
                exit 1
            fi
            echo "All shared libraries resolve correctly"

            echo "=== Dependency audit ==="
            for so in "$LIB_DIR"/*.so; do
                [ -f "$so" ] || continue
                echo "--- $(basename $so) ---"
                ldd "$so" 2>/dev/null | grep -v "linux-vdso\|ld-linux\|libc.so\|libm.so\|libstdc++\|libgcc_s\|libpthread\|librt\|libdl" || true
            done

            echo "=== All tests passed ==="
        ' 2>&1) && EXIT_CODE=0 || EXIT_CODE=$?

    echo "$RESULT"

    if [[ $EXIT_CODE -eq 0 ]]; then
        echo ""
        echo "  $DISTRO: PASSED"
        echo ""
        PASSED+=("$DISTRO")
    elif echo "$RESULT" | grep -q "SKIP: No compatible wheel"; then
        echo ""
        echo "  $DISTRO: SKIPPED (no compatible wheel)"
        echo ""
        SKIPPED+=("$DISTRO")
    else
        echo ""
        echo "  $DISTRO: FAILED"
        echo ""
        FAILED+=("$DISTRO")
    fi
done

# ==================================================================
# Summary
# ==================================================================
echo "======================================================================"
echo "Results: ${#PASSED[@]} passed, ${#FAILED[@]} failed, ${#SKIPPED[@]} skipped"
if [[ ${#PASSED[@]} -gt 0 ]]; then
    echo "  Passed:  ${PASSED[*]}"
fi
if [[ ${#SKIPPED[@]} -gt 0 ]]; then
    echo "  Skipped: ${SKIPPED[*]}"
fi
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "  Failed:  ${FAILED[*]}"
    echo "======================================================================"
    exit 1
fi
if [[ ${#PASSED[@]} -eq 0 ]]; then
    echo "  WARNING: No distros were tested (all skipped)"
    echo "======================================================================"
    exit 1
fi
echo "======================================================================"
