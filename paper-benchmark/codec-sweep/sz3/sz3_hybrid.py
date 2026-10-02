#!/usr/bin/env python3
"""SZ3 2-D/3-D vs 4-D (space x time) on every 4 MiB chunk of a dump series.

The dumps are cut into windows of T consecutive dumps. For every window,
field and 4 MiB chunk this records

  - how much the chunk changes inside the window:
      score_rms   max over the window's transitions of RMS(x_t - x_{t-1})
                  divided by the whole field's RMS at t (one scale per field)
      ident_frac  mean fraction of values bit-identical to the previous dump
  - SZ3 in every configuration and layout (sz3_window: 3d per dump, 4d with
    time slowest, 4d with time fastest), at an absolute bound
    eb = REL x (the field's value range over the window's dumps):
    compressed bytes, compress / decompress ms (one thread, in memory) and
    max |error| / eb.

The lossless side of a hybrid comes from the codec sweep's per-chunk CSVs for
the same (file, chunk); sz3_hybrid_eval.py joins the two.

  sz3_hybrid.py --wl nyx --root FIELDS --files FILES.txt --out CSV
                [--T 4] [--windows 0] [--rel 1e-3] [--workers 32]
"""
import argparse
import csv
import os
import subprocess
import sys
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

CHUNK = 4 << 20
DRIVER = os.path.expanduser("~/np-build/sz3-hybrid/sz3_window")
COLUMNS = ["wl", "window", "dump0", "field", "chunk", "files", "score_rms",
           "ident_frac", "eb_abs", "config", "layout", "bytes_in",
           "bytes_out", "comp_ms", "decomp_ms", "max_err_over_eb", "ok"]


def series(files):
    """Group a file list into fields and their dumps.

    @param files paths "dump/name.f32", relative to the root
    @return (sorted dump names, {field: {dump: path}})
    """
    by_field = defaultdict(dict)
    dumps = set()
    for f in files:
        dump, name = os.path.split(f)
        by_field[name][dump] = f
        dumps.add(dump)
    return sorted(dumps), by_field


def chunk_shape(field_bytes, dims):
    """Shape (Z, Y, X) of one 4 MiB chunk of a field with dims (Z, Y, X).

    A chunk must be whole Y*X planes, as it is for the 128^3 dumps.
    """
    z, y, x = dims
    plane = y * x * 4
    if CHUNK % plane or field_bytes % CHUNK:
        sys.exit(f"4 MiB chunks are not whole planes of {dims}")
    return CHUNK // plane, y, x


def change(frames, k, zc):
    """Change score and identical fraction of chunk k across the frames.

    @param frames list of (Z, Y, X) arrays, one per dump
    @param k      chunk index (planes k*zc .. k*zc+zc-1)
    @param zc     planes per chunk
    """
    score, ident = 0.0, []
    for a, b in zip(frames, frames[1:]):
        ca, cb = a[k * zc:(k + 1) * zc], b[k * zc:(k + 1) * zc]
        field_rms = np.sqrt(0.5 * (np.mean(a.astype(np.float64) ** 2) +
                                   np.mean(b.astype(np.float64) ** 2)))
        d = np.sqrt(np.mean((cb.astype(np.float64) - ca) ** 2))
        score = max(score, d / field_rms if field_rms > 0 else 0.0)
        ident.append(np.mean(ca.view(np.uint32) == cb.view(np.uint32)))
    return score, float(np.mean(ident))


def task(args):
    """All chunks of one (window, field): change scores and SZ3 rows."""
    wl, root, w, dumps, field, paths, dims, rel, configs, layouts = args
    frames = [np.fromfile(os.path.join(root, p), np.float32).reshape(dims)
              for p in paths]
    lo = min(float(f.min()) for f in frames)
    hi = max(float(f.max()) for f in frames)
    eb = rel * (hi - lo)
    zc, y, x = chunk_shape(frames[0].nbytes, dims)
    rows = []
    for k in range(dims[0] // zc):
        score, ident = change(frames, k, zc)
        base = {"wl": wl, "window": w, "dump0": dumps[0], "field": field,
                "chunk": k, "files": ";".join(paths), "score_rms": score,
                "ident_frac": ident, "eb_abs": eb}
        if not eb > 0:  # a constant field: nothing to bound, store it 1:1
            rows.append({**base, "config": "constant", "ok": 0})
            continue
        cmd = [DRIVER, "--files", ",".join(os.path.join(root, p) for p in paths),
               "--offset", str(k * CHUNK), "--shape", f"{zc},{y},{x}",
               "--eb", repr(eb), "--configs", configs, "--layouts", layouts]
        res = subprocess.run(cmd, capture_output=True, text=True)
        for line in res.stdout.splitlines():
            v = line.split(",")
            rows.append({**base, "config": v[0], "layout": v[1],
                         "bytes_in": v[2], "bytes_out": v[3], "comp_ms": v[4],
                         "decomp_ms": v[5], "max_err_over_eb": v[6],
                         "ok": int(len(v) == 7 and float(v[6]) <= 1 + 1e-6)})
        if res.returncode:
            rows.append({**base, "config": "driver-error", "ok": 0})
    return rows


def main():
    """Build the (window, field) tasks, run them in parallel, write the CSV."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--wl", required=True)
    ap.add_argument("--root", required=True, help="directory of the dumps")
    ap.add_argument("--files", required=True, help="file list (dump/name)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dims", default="128,128,128", help="Z,Y,X of a field")
    ap.add_argument("--T", type=int, default=4, help="dumps per window")
    ap.add_argument("--windows", type=int, default=0,
                    help="evenly spaced windows to run; 0 = all")
    ap.add_argument("--rel", type=float, default=1e-3)
    ap.add_argument("--configs", default="interp_lorenzo,interp_cubic,"
                    "interp_linear,lorenzo_reg,lorenzo2_reg,lorenzo_only,nopred")
    ap.add_argument("--layouts", default="3d,4d-tslow,4d-tfast")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    dims = tuple(int(v) for v in a.dims.split(","))
    dumps, by_field = series(open(a.files).read().split())
    nwin = len(dumps) // a.T
    wins = range(nwin) if a.windows <= 0 else sorted(
        {j * (nwin - 1) // max(a.windows - 1, 1) for j in range(a.windows)})
    tasks = []
    for w in wins:
        ds = dumps[w * a.T:(w + 1) * a.T]
        for field, paths in sorted(by_field.items()):
            if all(d in paths for d in ds):
                tasks.append((a.wl, a.root, w, ds, field, [paths[d] for d in ds],
                              dims, a.rel, a.configs, a.layouts))
    print(f"{a.wl}: {len(dumps)} dumps, {len(wins)} of {nwin} windows of "
          f"{a.T}, {len(tasks)} (window, field) tasks, {a.workers} workers",
          flush=True)
    with open(a.out, "w", newline="") as f, Pool(a.workers) as pool:
        wr = csv.DictWriter(f, COLUMNS)
        wr.writeheader()
        for i, rows in enumerate(pool.imap_unordered(task, tasks), 1):
            wr.writerows(rows)
            f.flush()
            if i % 50 == 0 or i == len(tasks):
                print(f"{i}/{len(tasks)} tasks", flush=True)


if __name__ == "__main__":
    main()
