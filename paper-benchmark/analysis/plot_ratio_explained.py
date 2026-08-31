#!/usr/bin/env python3
"""Why a chunk gets the compression ratio it gets.

    ./plot_ratio_explained.py OUT.png nyx=DIR vpic=DIR warpx=DIR ...

Not "which configuration won" -- every (chunk, configuration) row is plotted.
The question is what NUMBER comes out and why, and the answer is that there
are two regimes with two different causes, and each property owns one:

  LOSSLESS.  Shannon's source coding theorem: a coder that assigns one
  codeword per byte symbol and reads no order cannot beat

      ratio  <=  8 / H          H = byte entropy in bits/byte

  This is not a fit. Column 1 plots the achieved ratio against that bound with
  the diagonal drawn: points ON the line are chunks whose ratio entropy
  predicts exactly, points ABOVE it are the extra a codec wins by reading
  repeated byte SEQUENCES -- which entropy cannot see, because a histogram is
  invariant to any permutation of the buffer.

  QUANTIZED.  Quantization REPLACES the byte stream, so H of the original
  says nothing about what the codec then sees. Column 2 plots the same rows
  against the same 8/H and the relationship is gone. What replaces it is the
  grid: the quantizer snaps values onto steps of 2*0.95*eb, so

      L_bulk  =  MAD / (2 * 0.95 * eb)

  is how many steps the bulk of the values spans -- the alphabet the codec is
  actually handed. Column 3 plots the ratio against it.

So "why this ratio" has two answers, and which one applies is decided by
whether quantization ran, not by the data. Entropy is an exact predictor in
one regime and irrelevant in the other; MAD is the reverse.

Points are hexbinned: 50-120k rows per workload would be a solid blob as a
scatter, and the density is itself the finding (where the mass sits relative
to the bound).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as sps  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis import plotting  # noqa: E402


def load(d: str) -> pd.DataFrame:
    r = pd.read_csv(os.path.join(d, "processed", "exploration_results.csv"),
                    low_memory=False).replace([np.inf, -np.inf], np.nan)
    return r[(r["ratio"] > 0) & r["entropy"].notna() & (r["entropy"] > 0)]


def _hex(ax, x, y, xlabel, ylabel, title, note):
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[ok], y[ok]
    if len(x) < 50:
        ax.text(0.5, 0.5, "too few rows", transform=ax.transAxes,
                ha="center", color=plotting.MUTED)
        return
    ax.hexbin(x, y, xscale="log", yscale="log", gridsize=42, mincnt=1,
              cmap=plotting.SEQ, linewidths=0)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.annotate(note, xy=(0.03, 0.96), xycoords="axes fraction", va="top",
                fontsize=7.5, color=plotting.INK)
    plotting._finish(ax)


def main(out: str, *specs: str) -> int:
    plotting._style()
    runs = [(s.split("=", 1)[0], s.split("=", 1)[1]) for s in specs]
    fig, axes = plt.subplots(len(runs), 3, figsize=(13.2, 3.5 * len(runs)),
                             squeeze=False)
    for i, (name, d) in enumerate(runs):
        r = load(d)
        ll = r[r["quantize"] == 0]
        qz = r[(r["quantize"] == 1) & r["error_bound"].notna()
               & (r["error_bound"] > 0) & r["mad"].notna()]
        bound_ll = 8.0 / ll["entropy"]
        rho = sps.spearmanr(bound_ll, ll["ratio"]).correlation
        med = np.median(ll["ratio"] / bound_ll)
        _hex(axes[i][0], bound_ll.to_numpy(), ll["ratio"].to_numpy(),
             "entropy bound  8 / H", "achieved ratio",
             f"{name} -- LOSSLESS: entropy predicts it",
             f"rho {rho:+.2f}\nmedian ratio / (8/H) = {med:.2f}\n"
             f"n = {len(ll):,} rows")
        lim = [min(bound_ll.min(), ll["ratio"].min()),
               max(bound_ll.max(), ll["ratio"].max())]
        axes[i][0].plot(lim, lim, color=plotting.INK_2, lw=1.2, ls="--")
        axes[i][0].annotate("ratio = 8/H\n(all a histogram can give)",
                            xy=(0.62, 0.06), xycoords="axes fraction",
                            fontsize=7, color=plotting.INK_2)

        bound_qz = 8.0 / qz["entropy"]
        rho2 = sps.spearmanr(bound_qz, qz["ratio"]).correlation
        _hex(axes[i][1], bound_qz.to_numpy(), qz["ratio"].to_numpy(),
             "entropy bound  8 / H  (of the ORIGINAL bytes)", "achieved ratio",
             f"{name} -- QUANTIZED: entropy no longer applies",
             f"rho {rho2:+.2f}\nquantization replaced the bytes\n"
             f"n = {len(qz):,} rows")

        lb = qz["mad"] / (2 * 0.95 * qz["error_bound"])
        rho3 = sps.spearmanr(lb, qz["ratio"]).correlation
        _hex(axes[i][2], lb.to_numpy(), qz["ratio"].to_numpy(),
             "levels the bulk spans   L = MAD / (2 x 0.95 x eb)",
             "achieved ratio",
             f"{name} -- QUANTIZED: the grid predicts it",
             f"rho {rho3:+.2f}\nfewer levels left -> higher ratio\n"
             f"n = {len(qz):,} rows")
        print(f"{name:<6} lossless: rho(8/H) {rho:+.2f}, median ratio/(8/H) "
              f"{med:.3f}   |   quantized: rho(8/H) {rho2:+.2f}, "
              f"rho(L_bulk) {rho3:+.2f}")
    fig.suptitle("Why a chunk gets the ratio it gets -- every (chunk, "
                 "configuration) row, colour = density of rows\n"
                 "entropy is exact for lossless and irrelevant once "
                 "quantized; the quantization grid takes over",
                 fontsize=11, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(2)
    sys.exit(main(sys.argv[1], *sys.argv[2:]))
