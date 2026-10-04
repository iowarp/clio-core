#!/usr/bin/env python3
"""Group list for temporal_sweep: the same chunk of one field at T
consecutive dumps, one group per line.

    temporal_groups.py --dir FIELDS --window T [--chunk BYTES] [--stride S]
                       [--max-groups K] [--fields a,b] > groups.txt

Reads the three dump layouts the workloads produce:
    Nyx, VPIC   <dir>/plt%05d/fab0000_comp%02d_<field>.f32
    WarpX       <dir>/step%05d/<field>.f32
    LAMMPS      <dir>/<field>_step_<N>_chunk_<c>.bin   (c = 0: whole field)

A window starts every --stride dumps (default: T, non-overlapping). Every
4 MiB chunk of every field of every window is a candidate; --max-groups K
keeps K of them evenly spaced over the run, so a sample covers early, middle
and late frames alike. Each line is
    <field>@<dump>#<k>  <k>  <path_0> ... <path_{T-1}>
with paths relative to --dir and k the chunk index.
"""
import argparse
import glob
import os
import re
import sys


def frames(d):
    """@return (ordered dump names, {dump: {field: relative path}})."""
    plt = sorted(glob.glob(os.path.join(d, "plt*")))
    step = sorted(glob.glob(os.path.join(d, "step*")))
    by = {}
    if plt or step:
        for p in plt or step:
            dump = os.path.basename(p)
            for f in sorted(glob.glob(os.path.join(p, "*.f32"))):
                name = os.path.basename(f)[:-4]
                m = re.match(r"fab\d+_comp\d+_(.+)$", name)
                field = m.group(1) if m else name
                by.setdefault(dump, {})[field] = os.path.relpath(f, d)
        return sorted(by), by
    for f in glob.glob(os.path.join(d, "*_step_*_chunk_*.bin")):
        m = re.match(r"(.+)_step_(\d+)_chunk_(\d+)\.bin$", os.path.basename(f))
        if not m or int(m.group(3)) != 0:
            continue
        dump = "step%07d" % int(m.group(2))
        by.setdefault(dump, {})[m.group(1)] = os.path.relpath(f, d)
    return sorted(by), by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=4 << 20)
    ap.add_argument("--stride", type=int, default=0)
    ap.add_argument("--max-groups", type=int, default=0)
    ap.add_argument("--fields", default="")
    a = ap.parse_args()
    dumps, by = frames(a.dir)
    if len(dumps) < a.window:
        sys.exit("only %d dumps, window %d" % (len(dumps), a.window))
    want = set(a.fields.split(",")) if a.fields else None
    stride = a.stride or a.window
    groups = []
    for s in range(0, len(dumps) - a.window + 1, stride):
        win = dumps[s:s + a.window]
        fields = sorted(set.intersection(*(set(by[d]) for d in win)))
        for f in fields:
            if want and f not in want:
                continue
            paths = [by[d][f] for d in win]
            size = os.path.getsize(os.path.join(a.dir, paths[0]))
            n = -(-size // a.chunk)
            for k in range(n):
                groups.append(("%s@%s#%d" % (f, win[0], k), k, paths))
    if a.max_groups and a.max_groups < len(groups):
        k = a.max_groups
        groups = [groups[j * (len(groups) - 1) // max(k - 1, 1)] for j in range(k)]
    for label, k, paths in groups:
        print(label, k, *paths)


if __name__ == "__main__":
    main()
