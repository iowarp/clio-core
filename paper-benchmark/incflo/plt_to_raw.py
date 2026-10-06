#!/usr/bin/env python3
"""Convert AMReX plotfiles (level 0) to raw float32 arrays, one file per field.

    plt_to_raw.py OUT_DIR PLOTFILE [PLOTFILE ...] [--fields a,b] [--delete]

For each plotfile pltNNNNN writes OUT_DIR/pltNNNNN/<field>.f32: the level-0
domain as float32, x fastest and z slowest (the order of the FABs), so one
4 MiB chunk holds whole z-planes. Doubles are rounded to float32 once, here;
the benchmark then stores these float32 bytes losslessly. --fields keeps only
those plot variables (default: all); --delete removes each plotfile after it
is converted (the raw files are the dump that is kept).
"""
import argparse
import os
import re
import shutil

import numpy as np


def read_header(plt):
    """@return (variable names, level-0 domain lo, hi) from plt/Header."""
    with open(os.path.join(plt, "Header")) as f:
        lines = f.read().split("\n")
    ncomp = int(lines[1])
    names = lines[2:2 + ncomp]
    i = 2 + ncomp
    dim = int(lines[i])
    # time, finest level, prob_lo, prob_hi, ref ratios (empty for 1 level), domains
    dom = lines[i + 6]
    m = re.search(r"\(\(([-\d,\s]+)\)\s*\(([-\d,\s]+)\)", dom)
    lo = [int(v) for v in m.group(1).replace(",", " ").split()]
    hi = [int(v) for v in m.group(2).replace(",", " ").split()]
    assert len(lo) == dim == 3, f"{plt}: only 3D plotfiles are supported"
    return names, lo, hi


def read_level0(plt, names, lo, hi, keep):
    """@return {field: float64 array [z, y, x]} for the fields in keep."""
    shape = tuple(h - l + 1 for l, h in zip(lo, hi))[::-1]
    out = {n: np.full(shape, np.nan) for n in names if n in keep}
    lev = os.path.join(plt, "Level_0")
    with open(os.path.join(lev, "Cell_H")) as f:
        cell_h = f.read()
    fabs = re.findall(r"FabOnDisk:\s*(\S+)\s+(\d+)", cell_h)
    for fname, off in fabs:
        with open(os.path.join(lev, fname), "rb") as f:
            f.seek(int(off))
            head = f.readline().decode()
            # FAB ((8, (64 11 52 0 1 12 0 1023)),(8, (1 2 3 4 5 6 7 8)))((lo) (hi) (type)) ncomp
            m = re.match(r"FAB \(\((\d+), \(([\d\s]+)\)\),\((\d+), \(([\d\s]+)\)\)\)"
                         r"\(\(([-\d,]+)\) \(([-\d,]+)\) \(([-\d,]+)\)\) (\d+)", head)
            nbytes = int(m.group(1))
            order = [int(v) for v in m.group(4).split()]
            # AMReX byte order: (1 2 ... n) = big-endian, (n ... 2 1) = little-endian
            endian = ">" if order[0] == 1 else "<"
            dt = np.dtype(f"{endian}f{nbytes}")
            blo = [int(v) for v in m.group(5).split(",")]
            bhi = [int(v) for v in m.group(6).split(",")]
            nc = int(m.group(8))
            bshape = tuple(h - l + 1 for l, h in zip(blo, bhi))[::-1]
            data = np.fromfile(f, dtype=dt, count=nc * int(np.prod(bshape)))
        data = data.reshape((nc,) + bshape)
        sl = tuple(slice(b - l, b - l + n) for b, l, n in
                   zip(blo[::-1], lo[::-1], bshape))
        for c, n in enumerate(names[:nc]):
            if n in out:
                out[n][sl] = data[c]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("plotfiles", nargs="+")
    ap.add_argument("--fields", default="")
    ap.add_argument("--delete", action="store_true")
    a = ap.parse_args()
    for plt in a.plotfiles:
        names, lo, hi = read_header(plt)
        keep = set(a.fields.split(",")) if a.fields else set(names)
        missing = keep - set(names)
        if missing:
            raise SystemExit(f"{plt}: no plot variable(s) {sorted(missing)}; has {names}")
        d = os.path.join(a.out, os.path.basename(os.path.normpath(plt)))
        os.makedirs(d, exist_ok=True)
        for n, arr in read_level0(plt, names, lo, hi, keep).items():
            if np.isnan(arr).any():
                raise SystemExit(f"{plt}: field {n} has cells that no FAB covers (or NaN data)")
            arr.astype(np.float32).tofile(os.path.join(d, f"{n}.f32"))
        if a.delete:
            shutil.rmtree(plt)
        print(f"{plt}: {len(keep)} field(s) -> {d}", flush=True)


if __name__ == "__main__":
    main()
