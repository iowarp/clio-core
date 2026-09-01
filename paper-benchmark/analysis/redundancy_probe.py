#!/usr/bin/env python3
"""Temporal redundancy a run leaves on the table, from its own .f32 dumps.

    ./redundancy_probe.py NAME FIELDS_DIR RUN_CSV [N_SAMPLE]

Every codec in this benchmark compresses each chunk INDEPENDENTLY -- nothing
is coded against the same field at the previous dump. So temporal redundancy,
where it exists, is not merely unmeasured: it is unexploited. This measures
how much, per consecutive frame pair:

  spatial_H     byte entropy of the frame -- what the codec is handed today.
  temporal_dH   byte entropy of (x_t - x_{t-1}) -- what a coder referencing
                the previous frame would encode instead.
  headroom      spatial_H / temporal_dH. Above 1, that coder wins at the
                histogram level, and by that factor.

Measured: nyx 1.044-1.148 (97-99% of pairs above 1), warpx 1.040 (97.9%),
vpic 1.001 (71%, and its x-range is 0.997..1.005 -- nothing to gain). VPIC's
dumps are 25 steps apart in a turbulent plasma and decorrelate completely,
which is why its poor compression is not a missed opportunity and Nyx's is.

DELTA ENTROPY WAS TESTED HERE AND REMOVED, and the reason is worth keeping so
it is not re-added. H of the byte histogram of the FIRST DIFFERENCES is a
standard feature and it correlated with the lossless ratio at rho -0.907 --
stronger than byte entropy's own raw correlation. It is nonetheless useless
on this data: it tracks byte entropy at rho 0.99, and its partial correlation
given byte entropy is -0.05. The mechanism is float32. Per byte position on a
Nyx density chunk:

    byte        values   deltas
    0 mantissa   7.999    3.615
    1 mantissa   7.993    7.989
    2 mantissa   7.962    7.965
    3 sign+exp   1.554    2.982

Subtracting two nearby floats shrinks the MAGNITUDE and leaves the BIT
PATTERN just as random -- a small float32 fills its 23 mantissa bits as
thoroughly as a large one -- so three of four bytes are unchanged and the
exponent byte gets worse, because deltas span more orders of magnitude than
values do. Quantize the same chunk to int32 first and the picture inverts
(H 2.277 -> 1.432): on integer-like data delta coding genuinely wins. It is
raw IEEE floats that defeat it, which is the same reason lossless ratios here
sit at 1.07-1.30x and element width predicts nothing.

Reads a contiguous PREFIX of each frame, not a stride: these statistics are
about NEIGHBOURING values and striding would destroy the adjacency they
measure.
"""
import csv
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
        for pat in (f"{fields_dir}/{a}/{b}.f32", f"{fields_dir}/{a}/*_{b}.f32"):
            hit = glob.glob(pat)
            if hit:
                return hit[0], a, b
    return None, None, None


def spear(x, y) -> float:
    return float(sps.spearmanr(x, y).correlation)


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
        if ent <= 0:                       # a constant chunk has no histogram
            continue
        p = prev.get(field)
        prev[field] = a
        if p is None or p.size != a.size:
            continue
        th = H((a - p).view(np.uint8))
        if th <= 0:
            continue
        pf, af = p.astype(np.float64), a.astype(np.float64)
        rows.append({
            "blob": b, "field": field, "ratio": best[b],
            "spatial_H": ent, "temporal_dH": th, "headroom": ent / th,
            "pct_cells_same": float((a == p).mean()),
            "temporal_autocorr": (float(np.corrcoef(pf, af)[0, 1])
                                  if pf.std() > 0 and af.std() > 0
                                  else np.nan)})

    D = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).dropna(
        subset=["headroom"])
    out = f"{name}_redundancy.csv"
    D.to_csv(out, index=False)
    if D.empty:
        print("  no consecutive frame pairs found -- nothing to measure")
        return 1
    print(f"  {len(D)} frame pairs -> {out}\n")
    print(f"  temporal autocorrelation  median {D.temporal_autocorr.median():.4f}"
          f"   range {D.temporal_autocorr.min():.3f}..{D.temporal_autocorr.max():.4f}")
    print(f"  cells identical to previous dump  median "
          f"{100 * D.pct_cells_same.median():.2f}%")
    print(f"  spatial entropy   median {D.spatial_H.median():.3f} bits/byte")
    print(f"  temporal delta H  median {D.temporal_dH.median():.3f} bits/byte")
    print(f"\n  HEADROOM  median {D.headroom.median():.3f}"
          f"   {100 * (D.headroom > 1).mean():.1f}% of pairs above 1.0")
    print(f"  -- gain a previous-frame coder would get; nothing here "
          f"references the previous frame, so it is unclaimed")
    print(f"\n  headroom vs the achieved ratio: rho "
          f"{spear(D.headroom, D.ratio):+.3f}"
          f"   (expected to be weak: no codec exploits it)")
    _figure(name, D)
    return 0


def _figure(name, D):
    """Two panels: is the delta cheaper than the frame, and by how much.

    The scatter carries the mechanism -- every point below the diagonal is a
    frame whose delta codes cheaper than the frame itself -- and the histogram
    carries the size of the prize. Deliberately NOT plotted against the
    achieved ratio: no codec here references the previous frame, so a
    correlation with what was achieved would be a confound, not a mechanism.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from analysis import plotting as P
    P._style()

    fig, ax = plt.subplots(1, 2, figsize=(10.2, 4.2))
    c = P.SERIES[0]

    ax[0].scatter(D.spatial_H, D.temporal_dH, s=13, alpha=0.5, color=c,
                  edgecolor="none")
    lo = float(min(D.spatial_H.min(), D.temporal_dH.min()))
    hi = float(max(D.spatial_H.max(), D.temporal_dH.max()))
    ax[0].plot([lo, hi], [lo, hi], color=P.INK_2, lw=1.2, ls="--")
    ax[0].set_xlabel("entropy of the frame (bits/byte)", fontsize=9)
    ax[0].set_ylabel("entropy of the TEMPORAL delta (bits/byte)", fontsize=9)
    ax[0].set_title("is the delta cheaper than the frame?", fontsize=10)
    ax[0].annotate("below the line =\ndelta codes cheaper", xy=(0.05, 0.93),
                   xycoords="axes fraction", va="top", fontsize=8,
                   color=P.INK_2)
    P._finish(ax[0])

    ax[1].hist(D.headroom, bins=40, color=c, edgecolor=P.SURFACE, linewidth=0.5)
    ax[1].axvline(1.0, color=P.INK_2, lw=1.2, ls="--")
    ax[1].set_xlabel("headroom  =  frame entropy / delta entropy", fontsize=9)
    ax[1].set_ylabel("frame pairs", fontsize=9)
    ax[1].set_title("how much is unexploited", fontsize=10)
    ax[1].annotate(f"median {D.headroom.median():.3f}\n"
                   f"{100 * (D.headroom > 1).mean():.1f}% above 1.0\n"
                   f"{len(D)} frame pairs",
                   xy=(0.97, 0.93), xycoords="axes fraction", ha="right",
                   va="top", fontsize=8, color=P.INK)
    P._finish(ax[1])

    fig.suptitle(f"{name}: temporal redundancy no codec here is using\n"
                 f"every chunk is compressed independently -- nothing "
                 f"references the previous frame", fontsize=11, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = f"{name}_redundancy.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  figure -> {out}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(*sys.argv[1:]))
