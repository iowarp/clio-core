#!/usr/bin/env python3
"""Value range (min, max) of every field dump file, and of every chunk.

PSNR needs each file's (and each chunk's) own value range, and the traces'
measured PSNR cannot be used for it: the runtime reports 120 dB whenever the
absolute RMSE is below 1e-10, whatever the range, which hides chunks whose
values are that small. One read-only pass over the dumps:

  field_ranges.py <fields dir> <out prefix> [--chunk BYTES]

writes <out prefix>_field_ranges.csv (file, elements, min, max) and
<out prefix>_chunk_ranges.csv (blob, elements, min, max), blob names spelled
as the replay driver spells them ("<frame>/<stem>/chunk_<i>"). NaN elements
are ignored, as the quantizer ignores them.
"""
import argparse
import csv
import glob
import os

import numpy as np


def main():
    """One row per float32 dump under the fields dir, and one per chunk."""
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("prefix")
    ap.add_argument("--chunk", type=int, default=8388608, help="chunk size in bytes")
    a = ap.parse_args()
    per = a.chunk // 4
    files, chunks = [], []
    for path in sorted(glob.glob(os.path.join(a.src, "*", "*.f32"))):
        x = np.memmap(path, dtype=np.float32, mode="r")
        stem = os.path.relpath(path, a.src)[:-4]
        files.append((stem, x.size, float(np.nanmin(x)), float(np.nanmax(x))))
        for i in range(0, x.size, per):
            c = x[i:i + per]
            chunks.append((f"{stem}/chunk_{i // per}", c.size, float(np.nanmin(c)), float(np.nanmax(c))))
    for rows, kind, head in ((files, "field", "file"), (chunks, "chunk", "blob")):
        with open(f"{a.prefix}_{kind}_ranges.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([head, "elements", "min", "max"])
            w.writerows(rows)
    print(f"{len(files)} files, {len(chunks)} chunks -> {a.prefix}_{{field,chunk}}_ranges.csv")


if __name__ == "__main__":
    main()
