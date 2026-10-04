#!/usr/bin/env python3
"""Lossy-pipeline intermediates: cuSZ-style dual-quantization codes of the
simulation fields, the uint16 stream a lossy compressor hands its lossless
back end (and the data GPULZ, ICS'23, was designed around).

    gen_quant.py --dump DIR --out DIR [--fields a,b] [--every N] [--rel 1e-3]
                 [--radius 512]

Per field and frame: eb = rel x value range; q = rint(x / 2eb) (int64);
d = q - Lorenzo3D(q) with zero padding; code = d + radius when |d| < radius,
else 0 (an outlier, stored separately by cuSZ). Written as
<out>/qcode_<field>_f<NNNN>.u16, plus the outlier share in a log line.
"""
import argparse
import glob
import os
import re

import numpy as np


def lorenzo_residual(q):
    p = np.pad(q, ((1, 0), (1, 0), (1, 0)))
    pred = (p[:-1, 1:, 1:] + p[1:, :-1, 1:] + p[1:, 1:, :-1]
            - p[:-1, :-1, 1:] - p[:-1, 1:, :-1] - p[1:, :-1, :-1]
            + p[:-1, :-1, :-1])
    return q - pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fields", default="")
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--rel", type=float, default=1e-3)
    ap.add_argument("--radius", type=int, default=512)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    want = set(a.fields.split(",")) if a.fields else None
    for i, d in enumerate(sorted(glob.glob(os.path.join(a.dump, "plt*")))):
        if i % a.every:
            continue
        for f in sorted(glob.glob(os.path.join(d, "*.f32"))):
            m = re.match(r"fab\d+_comp\d+_(.+)\.f32$", os.path.basename(f))
            if not m or (want and m.group(1) not in want):
                continue
            x = np.fromfile(f, dtype=np.float32).astype(np.float64)
            side = round(len(x) ** (1 / 3))
            x = x.reshape(side, side, side)
            rng = x.max() - x.min()
            if rng == 0:
                continue
            q = np.rint(x / (2 * a.rel * rng)).astype(np.int64)
            dlt = lorenzo_residual(q)
            ok = np.abs(dlt) < a.radius
            code = np.where(ok, dlt + a.radius, 0).astype(np.uint16)
            code.tofile(os.path.join(a.out, f"qcode_{m.group(1)}_f{i:04d}.u16"))
            print(f"frame {i} {m.group(1)}: outliers {100 * (1 - ok.mean()):.3f}%", flush=True)


if __name__ == "__main__":
    main()
