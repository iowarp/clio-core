#!/bin/bash
# Install dependencies for qtable_data_collect.py.
#
# Lossless compressors come from pip (and stdlib for bz2/zlib/lzma).
# Lossy compressors (ZFP / SZ3 / FPZIP) require libpressio, which is
# best installed via spack — instructions at the bottom.
#
# Usage:
#   ./install_qtable_deps.sh [venv_path]
# defaults to $HOME/venv (matches the existing IOWarp setup on ares).

set -euo pipefail

VENV="${1:-$HOME/venv}"
if [ ! -x "$VENV/bin/pip" ]; then
    echo "FATAL: no pip at $VENV/bin/pip — pass a valid venv path" >&2
    exit 2
fi

echo "=== Installing lossless Python compressors into $VENV ==="
"$VENV/bin/pip" install --upgrade \
    zstandard \
    lz4 \
    Brotli \
    python-snappy \
    blosc2

echo
echo "=== Installing ZFP Python binding (lossy, float-only) ==="
"$VENV/bin/pip" install --upgrade zfpy numpy

echo
echo "=== Optional: SZ3 / FPZIP via libpressio ==="
echo "These are not pip-installable. Use spack:"
echo
echo "  spack install libpressio +sz3 +fpzip +zfp +zstd"
echo "  spack load    libpressio"
echo "  $VENV/bin/pip install pylibpressio"
echo
echo "If libpressio isn't built, the script will simply skip sz3/fpzip"
echo "and still run all lossless + zfp."
echo
echo "=== Done. Check coverage with: ==="
echo "  $VENV/bin/python3 -c \"from qtable_data_collect import COMPRESSORS; \\"
echo "    print(sorted({c[0] for c in COMPRESSORS}))\""
