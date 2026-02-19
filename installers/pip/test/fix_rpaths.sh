#!/bin/bash
# fix_rpaths.sh - Set $ORIGIN-relative RPATHs on all IOWarp binaries
#
# Usage:
#   bash fix_rpaths.sh <install_dir>
#
# Where <install_dir> contains the iowarp_core/ package directory.

set -euo pipefail

INSTALL_DIR="${1:?Usage: fix_rpaths.sh <install_dir>}"

if ! command -v patchelf &>/dev/null; then
    echo "Error: patchelf not found. Install with: apt-get install patchelf (or yum install patchelf)" >&2
    exit 1
fi

echo "Fixing RPATHs in $INSTALL_DIR..."

# Shared libraries: find each other via $ORIGIN
for so in "$INSTALL_DIR"/iowarp_core/lib/*.so*; do
    [ -f "$so" ] || continue
    patchelf --set-rpath '$ORIGIN' "$so" 2>/dev/null || true
    echo "  lib/$(basename "$so"): \$ORIGIN"
done

# Python extensions: find libs via $ORIGIN/../lib
for so in "$INSTALL_DIR"/iowarp_core/ext/*.so*; do
    [ -f "$so" ] || continue
    patchelf --set-rpath '$ORIGIN/../lib' "$so" 2>/dev/null || true
    echo "  ext/$(basename "$so"): \$ORIGIN/../lib"
done

# CLI binaries: find libs via $ORIGIN/../lib
for bin in "$INSTALL_DIR"/iowarp_core/bin/*; do
    [ -f "$bin" ] || continue
    patchelf --set-rpath '$ORIGIN/../lib' "$bin" 2>/dev/null || true
    echo "  bin/$(basename "$bin"): \$ORIGIN/../lib"
done

echo "Done."
