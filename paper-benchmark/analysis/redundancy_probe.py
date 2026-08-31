#!/usr/bin/env python3
"""Properties the runtime does NOT log, tested against what it achieved.

    ./redundancy_probe.py NAME FIELDS_DIR RUN_CSV [N_SAMPLE]

The three logged features -- byte entropy, MAD, lag-1 second derivative -- are
either permutation-invariant or see one axis, so none of them can account for
the EXCESS over the 8/H bound (the part of a lossless ratio contributed by
repeated byte SEQUENCES). This measures four candidates that are not logged,
straight from a run's own .f32 dumps, and reports whether each earns a place:

  delta entropy       H of the byte histogram of the FIRST DIFFERENCES.
  lag-1 autocorr      Pearson r between x[i] and x[i+1].
  temporal delta H    H of (x_t - x_{t-1}) for the same field, consecutive
                      dumps -- what a coder referencing the previous frame
                      would have to encode.
  headroom            spatial_H / temporal_dH. > 1 means such a coder would
                      beat the current one at the histogram level. Nothing
                      here references the previous frame, so this is gain
                      LEFT ON THE TABLE, not an explanation of what was got.

A candidate earns its place only if its PARTIAL correlation given byte entropy
is non-trivial AND the same sign across workloads. Measured on nyx/vpic/warpx:

  delta entropy   rho with byte entropy 0.88-0.96; partial vs ratio +-0.05.
                  Redundant on all three -- it is byte entropy in disguise.
  lag-1 autocorr  partial vs excess +0.170 / +0.007 / -0.179. A sign flip
                  with a zero in the middle: not a mechanism.
  headroom        nyx 1.148 (99.4% of pairs > 1), warpx 1.040 (97.9%),
                  vpic 1.001 (nothing to gain -- its dumps are 25 steps apart
                  in a turbulent plasma and decorrelate completely).

Reads a contiguous PREFIX of each chunk, not a stride: every statistic here is
about NEIGHBOURING values and striding would destroy the adjacency it measures.
"""
import csv, glob, os, sys
import numpy as np
import pandas as pd
from scipy import stats as sps

PREFIX = 262144          # 1 MiB of float32


def H(b: np.ndarray) -> float:
    c = np.bincount(b, minlength=256).astype(np.float64)
    p = c[c > 0] / c.sum()
    return float(-(p * np.log2(p)).sum())


def _resolve(fields_dir: str, blob: str):
    """Blob layouts differ: nyx/warpx are <frame>/<field>, vpic is
    <field>/<frame>, and a dump file may or may not carry a fab/comp prefix."""
    p = blob.split("/")
    for a, b in ((p[0], p[1]), (p[1], p[0])):
        for pat in (f"{fields_dir}/{a}/{b}.f32",
                    f"{fields_dir}/{a}/*_{b}.f32"):
            hit = glob.glob(pat)
            if hit:
                return hit[0], a, b
    return None, None, None


def spear(x, y):
    return float(sps.spearmanr(x, y).correlation)


def partial(x, y, z):
    """Spearman partial correlation of x,y controlling for z."""
    rx, ry, rz = (sps.rankdata(v) for v in (x, y, z))
    a = np.corrcoef(rx, rz)[0, 1]
    b = np.corrcoef(ry, rz)[0, 1]
    c = np.corrcoef(rx, ry)[0, 1]
    return (c - a * b) / np.sqrt((1 - a ** 2) * (1 - b ** 2))


def main(name, fields_dir, run_csv, n_sample=200):
    n_sample = int(n_sample)
    best = {}
    with open(run_csv) as fh:
        for r in csv.DictReader(fh):
            if r["quantize"] != "0":
                continue
            try:
                v = float(r["ratio"])
            except ValueError:
                continue
            if v > 0:
                best[r["blob"]] = max(best.get(r["blob"], 0.0), v)
    blobs = sorted(best)
    step = max(1, len(blobs) // n_sample)
    sample = blobs[::step][:n_sample]
    print(f"{name}: {len(sample)} of {len(blobs)} chunks sampled")

    rows, prev = [], {}
    for b in sample:
        path, frame, field = _resolve(fields_dir, b)
        if not path:
            continue
        a = np.fromfile(path, dtype=np.float32, count=PREFIX)
        if a.size < 1024:
            continue
        ent = H(a.view(np.uint8))
        if ent <= 0:                       # constant chunk: 8/H undefined
            continue
        x0, x1 = a[:-1].astype(np.float64), a[1:].astype(np.float64)
        ac = (float(np.corrcoef(x0, x1)[0, 1])
              if x0.std() > 0 and x1.std() > 0 else np.nan)
        if not np.isfinite(ac):
            continue
        rec = {"blob": b, "field": field, "ratio": best[b], "entropy": ent,
               "delta_entropy": H(np.diff(a).view(np.uint8)),
               "lag1_autocorr": ac, "excess": best[b] / (8.0 / ent),
               "temporal_dH": np.nan, "headroom": np.nan}
        p = prev.get(field)
        if p is not None and p.size == a.size:
            th = H((a - p).view(np.uint8))
            if th > 0:
                rec["temporal_dH"] = th
                rec["headroom"] = ent / th
        prev[field] = a
        rows.append(rec)

    D = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    out = f"{name}_redundancy.csv"
    D.to_csv(out, index=False)
    d = D.dropna(subset=["entropy", "ratio", "excess"])
    print(f"  {len(d)} chunks measured -> {out}\n")
    print("  candidate         rho(ratio)  rho(excess)  partial|H ratio  "
          "partial|H excess   verdict")
    for c in ("delta_entropy", "lag1_autocorr"):
        pr = partial(d[c], d.ratio, d.entropy)
        pe = partial(d[c], d.excess, d.entropy)
        v = "earns a place" if max(abs(pr), abs(pe)) > 0.15 else \
            "redundant with entropy"
        print(f"  {c:<18}{spear(d[c], d.ratio):>+10.3f}"
              f"{spear(d[c], d.excess):>+13.3f}{pr:>+17.3f}{pe:>+18.3f}"
              f"   {v}")
    t = D.dropna(subset=["headroom"])
    if len(t):
        print(f"\n  temporal headroom (spatial H / temporal delta H) over "
              f"{len(t)} frame pairs")
        print(f"    median {t.headroom.median():.3f}   "
              f"{100 * (t.headroom > 1).mean():.1f}% above 1.0"
              f"   -- gain a previous-frame coder would get, unexploited here")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__); sys.exit(2)
    sys.exit(main(*sys.argv[1:]))
