#!/bin/bash
# repair_wheel.sh - Fix RPATHs and repack wheel for cibuildwheel
#
# Called by CIBW_REPAIR_WHEEL_COMMAND_LINUX in place of auditwheel repair.
# auditwheel doesn't work well with our bundled shared libraries, so we
# fix RPATHs to use $ORIGIN and repack preserving Unix permissions.
# We then use `wheel tags` to apply the correct manylinux platform tag.
#
# Usage (called by cibuildwheel):
#   bash repair_wheel.sh {dest_dir} {wheel}

set -euo pipefail

DEST_DIR="${1:?Usage: repair_wheel.sh <dest_dir> <wheel>}"
WHEEL="${2:?Usage: repair_wheel.sh <dest_dir> <wheel>}"

echo "=== Repairing wheel: $(basename "$WHEEL") ==="

WORK_DIR=$(mktemp -d)
trap "rm -rf $WORK_DIR" EXIT

# Unpack
python3 -m zipfile -e "$WHEEL" "$WORK_DIR/unpack/"

# Make binaries executable (zip loses permission bits)
chmod +x "$WORK_DIR"/unpack/iowarp_core/bin/* 2>/dev/null || true

# Fix RPATHs
echo "Fixing RPATHs..."

# Shared libraries: find each other via $ORIGIN
for so in "$WORK_DIR"/unpack/iowarp_core/lib/*.so*; do
    [ -f "$so" ] || continue
    patchelf --set-rpath '$ORIGIN' "$so" 2>/dev/null || true
done

# Python extensions: find libs via $ORIGIN/../lib
for so in "$WORK_DIR"/unpack/iowarp_core/ext/*.so*; do
    [ -f "$so" ] || continue
    patchelf --set-rpath '$ORIGIN/../lib' "$so" 2>/dev/null || true
done

# CLI binaries: find libs via $ORIGIN/../lib
for bin in "$WORK_DIR"/unpack/iowarp_core/bin/*; do
    [ -f "$bin" ] || continue
    patchelf --set-rpath '$ORIGIN/../lib' "$bin" 2>/dev/null || true
done

# Repack preserving permissions into a temp location
WHEEL_NAME=$(basename "$WHEEL")
REPACK_DIR="$WORK_DIR/repacked"
mkdir -p "$REPACK_DIR"

python3 -c "
import os, sys, zipfile
base = sys.argv[1]
out  = sys.argv[2]
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(base):
        for f in files:
            full = os.path.join(root, f)
            arc = os.path.relpath(full, base)
            info = zipfile.ZipInfo(arc)
            st = os.stat(full)
            info.external_attr = (st.st_mode & 0xFFFF) << 16
            with open(full, 'rb') as fh:
                zf.writestr(info, fh.read())
" "$WORK_DIR/unpack" "$REPACK_DIR/$WHEEL_NAME"

# Detect the architecture and apply the manylinux platform tag.
# PyPI rejects raw linux_* platform tags; we need manylinux_2_34_*.
# `wheel tags` properly updates the filename, WHEEL metadata, and RECORD.
ARCH=$(echo "$WHEEL_NAME" | grep -oP 'linux_\K(x86_64|aarch64|i686|ppc64le|s390x)')
if [ -n "$ARCH" ]; then
    echo "Retagging wheel with manylinux_2_34_${ARCH}..."
    pip install -q wheel
    python3 -m wheel tags \
        --platform-tag "manylinux_2_34_${ARCH}" \
        --remove \
        "$REPACK_DIR/$WHEEL_NAME"
fi

# Move the final wheel (retagged or original) to the destination
mv "$REPACK_DIR"/*.whl "$DEST_DIR/"
echo "=== Repaired wheel placed in: $DEST_DIR ==="
ls -1 "$DEST_DIR"/*.whl | tail -1
