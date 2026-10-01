#!/usr/bin/env python3
"""Per-block coding-mode decision on quantized data: is there real headroom?

    mode_decision.py --root FIELDS --window plt150-153 --out CSV

After error-bounded quantization (q = rint(x / 2eb), |x - 2eb q| <= eb) the
reconstruction is fixed; how the integers are CODED is a free, lossless
choice per block, exactly like a video codec's per-macroblock mode:

  SKIP      q_t == q_{t-1} for the whole block: a 16-byte reference
  SPATIAL   delta of q_t along x (the SZ-like default)
  TEMPORAL  q_t - q_{t-1}
  BOTH      x-delta of (q_t - q_{t-1})

Every mode reconstructs the same values, so the comparison is pure size. For
T1..T3 of every field: bytes with ONE mode for all blocks, and with the best
mode per block (blocks then packed into one stream per mode, as an
implementation would write them; the per-block choice is made from per-block
frame sizes, so it is realizable, not a hindsight on the packed streams).
"""
import argparse
import re

import numpy as np
import pandas as pd
import zstandard

import classify_temporal as C

ZC = zstandard.ZstdCompressor(level=3)
REF = 16
MODES = ["SPATIAL", "TEMPORAL", "BOTH", "LORENZO", "T+LORENZO"]
BS = 16


def lorenzo(v):
    """3-D Lorenzo residual of a WHOLE (nz, ny, nx) integer grid, exact: each
    value minus its prediction from the 7 already-coded neighbours (zero only
    outside the field), the predictor SZ / cuSZ use. Computed on the field,
    not per block, so blocks are not charged artificial edges."""
    p = np.pad(v, ((1, 0), (1, 0), (1, 0)))
    pred = (p[1:, 1:, :-1] + p[1:, :-1, 1:] + p[:-1, 1:, 1:]
            - p[1:, :-1, :-1] - p[:-1, 1:, :-1] - p[:-1, :-1, 1:]
            + p[:-1, :-1, :-1])
    return v - pred


def zsize(a):
    if a.size == 0:
        return 0
    b = np.ascontiguousarray(a, dtype=np.int32).view(np.uint8).reshape(-1, 4).T
    return len(ZC.compress(b.tobytes()))


def encodings(g, gp):
    """The non-skip encodings of a quantized grid g given the previous gp,
    each computed on the whole field, then cut into block rows."""
    td = g - gp
    full = {"SPATIAL": np.diff(g, axis=2, prepend=0),
            "TEMPORAL": td,
            "BOTH": np.diff(td, axis=2, prepend=0),
            "LORENZO": lorenzo(g),
            "T+LORENZO": lorenzo(td)}
    return {m: C.to_blocks(v, (BS, BS, BS))[0] for m, v in full.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--window", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ebs", nargs="+", type=float, default=[1e-4, 1e-3, 1e-2])
    a = ap.parse_args()
    lo, hi = map(int, re.findall(r"\d+", a.window))
    src = C.RawF32Source(a.root, [f"plt{k:05d}" for k in range(lo, hi + 1)])
    recs = []
    for f in src.fields():
        grids = [src.load(f, t).astype(np.float64) for t in range(4)]
        rng = float(grids[0].max() - grids[0].min()) or 1.0
        rows = [C.to_blocks(g, (16, 16, 16))[0] for g in grids]
        for eb in a.ebs:
            qg = [np.rint(g / (2 * eb * rng)).astype(np.int64) for g in grids]
            for t in (1, 2, 3):
                q, qp = C.to_blocks(qg[t], (BS, BS, BS))[0], C.to_blocks(qg[t - 1], (BS, BS, BS))[0]
                enc = encodings(qg[t], qg[t - 1])
                skip = np.all(q == qp, axis=1)
                per_block = {m: np.array([zsize(e[i]) for i in range(len(q))])
                             for m, e in enc.items()}
                choice = np.argmin(np.stack([per_block[m] for m in MODES], 1), 1)
                rec = dict(field=f, eb=eb, t=t, raw=rows[t].size * 4,
                           skip_blocks=int(skip.sum()), blocks=len(q))
                for m in MODES:                       # one mode for every block
                    rec[f"all_{m}"] = zsize(enc[m])
                    rec[f"all_{m}+skip"] = REF * int(skip.sum()) + zsize(enc[m][~skip])
                best = REF * int(skip.sum())          # per-block mode, packed by mode
                for k, m in enumerate(MODES):
                    sel = (choice == k) & ~skip
                    best += zsize(enc[m][sel])
                    rec[f"chosen_{m}"] = int(sel.sum())
                rec["per_block_best"] = best
                recs.append(rec)
        print(f"  {f} done", flush=True)
    df = pd.DataFrame(recs)
    df.to_csv(a.out, index=False)
    g = df.groupby("eb").sum(numeric_only=True)
    cols = [f"all_{m}" for m in MODES] + [f"all_{m}+skip" for m in MODES] + ["per_block_best"]
    out = pd.DataFrame({c: g["raw"] / g[c] for c in cols})
    out["skip_%"] = 100 * g["skip_blocks"] / g["blocks"]
    for m in MODES:
        out[f"pick_{m}_%"] = 100 * g[f"chosen_{m}"] / g["blocks"]
    print("ratio (T1..T3, all fields):")
    print(out.round(2).to_string())


if __name__ == "__main__":
    main()
