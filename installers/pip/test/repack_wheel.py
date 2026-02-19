#!/usr/bin/env python3
"""Repack a wheel directory into a .whl file, preserving Unix permissions."""
import os
import sys
import zipfile


def repack(whl_path, base_dir):
    with zipfile.ZipFile(whl_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(base_dir):
            for f in files:
                full = os.path.join(root, f)
                arc = os.path.relpath(full, base_dir)
                info = zipfile.ZipInfo(arc)
                st = os.stat(full)
                # Preserve Unix permissions (especially execute bit)
                info.external_attr = (st.st_mode & 0xFFFF) << 16
                with open(full, "rb") as fh:
                    zf.writestr(info, fh.read())


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <output.whl> <unpacked_dir>")
        sys.exit(1)
    repack(sys.argv[1], sys.argv[2])
