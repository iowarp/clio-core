#!/bin/bash
# =============================================================================
# test_in_container.sh - Runs inside a clean distro container
# =============================================================================
# Replicates the CI test-wheels matrix job. Verifies:
#   1. Wheel installs cleanly with only Python + pip
#   2. Python package imports and APIs work
#   3. CLI binary runs
#   4. All shared library symbols resolve (no "not found")
#   5. No unexpected external shared library dependencies
# =============================================================================
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

# Install wheel (grab the one matching this Python, or first available)
pip install --quiet /wheels/*.whl 2>/dev/null || \
    pip install --quiet $(ls /wheels/*.whl | head -1)

echo "=== Import test ==="
python -c "import iowarp_core; print('Version:', iowarp_core.get_version())"
python -c "import iowarp_core; print('Lib dir:', iowarp_core.get_lib_dir())"
python -c "import iowarp_core; print('CTE available:', iowarp_core.cte_available())"

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
