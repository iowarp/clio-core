#!/usr/bin/env python3
"""Motivation experiment: does fidelity that follows temporal evolution pay?

    region_adaptive.py --results DIR --window plt150-153 --root FIELDS --out CSV

Uses the per-block, per-transition states of classify_temporal.py (EXACT /
NEAR / COLD / HOT against the previous timestep) and stores T1..T3 of every
field under each policy. Blocks of one class are packed into one stream per
field and timestep (what an implementation would write), so no policy pays a
small-frame penalty the others do not.

  lossless        every block: byte shuffle + zstd-3
  uniform@eb      every block: error-bounded quantization at eb, then
                  integer delta along x, shuffle, zstd-3 (an SZ-like pipeline)
  adaptive@eb     EXACT -> reference; COLD -> quantized at eb; HOT and NEAR ->
                  lossless
  adaptive+reuse@eb  as adaptive, but a COLD block within eb of its LAST STORED
                  version (not the previous truth, so error cannot accumulate)
                  is a reference too
  inverse@eb      the control: HOT quantized, COLD lossless

eb is relative to the field's value range at T0. T0 is stored losslessly by
every policy and reported separately. Quality per field and timestep: max
|error| / range, PSNR, and the max error inside HOT blocks.
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
import zstandard

import classify_temporal as C

ZC = zstandard.ZstdCompressor(level=3)
REF = 16
EBS = [1e-4, 1e-3, 1e-2]


def zsize(a):
    """Shuffled zstd size of a numeric array (0 for an empty one)."""
    if a.size == 0:
        return 0
    b = np.ascontiguousarray(a).view(np.uint8).reshape(-1, a.itemsize).T
    return len(ZC.compress(b.tobytes()))


def quantize(x, eb):
    """Error-bounded quantization: integers q with |x - 2*eb*q| <= eb, delta
    coded along the block's fastest axis (exact on the integers).
    @return (bytes when compressed, reconstruction)"""
    q = np.rint(x.astype(np.float64) / (2 * eb)).astype(np.int64)
    recon = (q * (2 * eb)).astype(np.float32)
    # float32 rounding of the reconstruction can add an ulp; fall back to the
    # exact value there so the bound is a guarantee, not an approximation.
    bad = np.abs(recon.astype(np.float64) - x) > eb
    recon[bad] = x[bad]
    d = np.diff(q, axis=-1, prepend=0).astype(np.int32)
    return zsize(d) + 4 * int(bad.sum()), recon


def states_for(d, field):
    """(nblocks, 3) states of T1..T3 against the previous timestep."""
    g = d[d["field"] == field].sort_values("block_index")
    return np.stack([g[f"state_{p}"].map({s: i for i, s in enumerate(C.STATE_NAMES)})
                     .to_numpy() for p in C.CONSEC], 1)


def store(policy, eb, rows, st, rng):
    """Bytes and reconstructions of T1..T3 of one field under one policy.

    @param rows [T0..T3] block rows (float32)
    @param st   (nblocks, 3) states
    @param rng  the field's value range at T0 (eb is relative to it)
    @return (bytes per timestep, reconstructions per timestep)
    """
    e = eb * rng
    prev = rows[0].copy()           # T0 stored losslessly: the first stored version
    nbytes, recons = [], []
    for t in (1, 2, 3):
        x, s = rows[t], st[:, t - 1]
        recon = x.copy()
        if policy == "lossless":
            b = zsize(x)
        elif policy == "uniform":
            b, recon = quantize(x, e)
        else:
            lossy = (s == C.COLD) if policy != "inverse" else (s == C.HOT)
            ref = (s == C.EXACT) if policy != "inverse" else np.zeros_like(lossy)
            if policy == "adaptive+reuse":
                close = np.abs(x.astype(np.float64) - prev).max(1) <= e
                reuse = lossy & close
                ref = ref | reuse
                lossy = lossy & ~reuse
                recon[reuse] = prev[reuse]
            keep = ~lossy & ~ref
            b = REF * int(ref.sum()) + zsize(x[keep])
            if lossy.any():
                qb, qr = quantize(x[lossy], e)
                b += qb
                recon[lossy] = qr
            recon[ref & ~(s != C.EXACT)] = x[ref & ~(s != C.EXACT)]
        nbytes.append(b)
        recons.append(recon)
        prev = recon.astype(np.float64)
    return nbytes, recons


def quality(x, r, hot, rng):
    """max |err|/range, PSNR (dB) and the max |err|/range inside HOT blocks."""
    err = np.abs(r.astype(np.float64) - x)
    rmse = np.sqrt(np.mean(err ** 2))
    psnr = float("inf") if rmse == 0 else 20 * np.log10(rng / rmse)
    hot_err = float(err[hot].max()) / rng if hot.any() else 0.0
    return float(err.max()) / rng, psnr, hot_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--window", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lo, hi = map(int, re.findall(r"\d+", a.window))
    src = C.RawF32Source(a.root, [f"plt{k:05d}" for k in range(lo, hi + 1)])
    d = pd.read_parquet(os.path.join(a.results, a.window, "blocks.parquet"))
    bs = (16, 16, 16)
    recs = []
    for f in src.fields():
        grids = [src.load(f, t) for t in range(4)]
        rows = [C.to_blocks(g, bs)[0] for g in grids]
        st = states_for(d, f)
        rng = float(grids[0].max() - grids[0].min()) or 1.0
        raw = rows[1].nbytes
        cls_bytes = {c: [] for c in ("EXACT", "COLD", "HOT", "NEAR")}
        for t in (1, 2, 3):
            for k, name in enumerate(C.STATE_NAMES):
                m = st[:, t - 1] == k
                cls_bytes[name].append(zsize(rows[t][m]))
        recs.append(dict(field=f, policy="_lossless_bytes_by_class", eb=0, **{
            f"class_{k}": sum(v) for k, v in cls_bytes.items()}))
        for policy, ebs in (("lossless", [0.0]), ("uniform", EBS), ("adaptive", EBS),
                            ("adaptive+reuse", EBS), ("inverse", EBS)):
            for eb in ebs:
                nb, rc = store(policy, eb, rows, st, rng)
                for t in (1, 2, 3):
                    hot = np.repeat(st[:, t - 1] == C.HOT, rows[t].shape[1]).reshape(rows[t].shape)
                    mx, ps, he = quality(rows[t].astype(np.float64), rc[t - 1], hot, rng)
                    recs.append(dict(field=f, policy=policy, eb=eb, t=t, raw=raw,
                                     stored=nb[t - 1], max_err=mx, psnr=ps, hot_max_err=he))
        print(f"  {f} done", flush=True)
    df = pd.DataFrame(recs)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)
    by_class = df[df.policy == "_lossless_bytes_by_class"][[c for c in df if c.startswith("class_")]].sum()
    print("lossless compressed bytes by class (T1..T3):",
          {k[6:]: f"{100 * v / by_class.sum():.1f}%" for k, v in by_class.items()})
    p = df[df.policy != "_lossless_bytes_by_class"]
    s = p.groupby(["policy", "eb"]).agg(raw=("raw", "sum"), stored=("stored", "sum"),
                                         max_err=("max_err", "max"), psnr=("psnr", "min"),
                                         hot_max_err=("hot_max_err", "max"))
    s["ratio"] = s.raw / s.stored
    print(s[["ratio", "max_err", "psnr", "hot_max_err"]].round(4).to_string())


if __name__ == "__main__":
    main()
