#!/usr/bin/env python3
"""What each of the three properties does to the compression ratio.

    ./plot_three_properties.py OUT.png nyx=DIR vpic=DIR warpx=DIR ...

Every (chunk, configuration) row, not the winner. Rows are the three
properties the model is given; columns split the ratio into the three things
it can be:

  1. LOSSLESS ratio. Bounded by ratio <= 8/H (source coding theorem), so
     entropy is not correlated with this -- it IS this, up to the excess in
     column 2. rho(entropy, 8/H) is -1.000 by definition, which is why the
     interesting question is column 2.

  2. EXCESS over the bound, ratio / (8/H): the part of the lossless ratio that
     a byte histogram provably cannot account for, contributed entirely by
     repeated byte SEQUENCES. If MAD or the second derivative earn their place
     in the feature vector for lossless data, this is where it shows.

  3. QUANTIZED ratio. Quantization replaces the byte stream, so the original
     H no longer bounds anything; what remains is the grid, and the grid is
     set by spread over the error bound.

Reading the grid tells you which property owns which regime, and the answer
is not symmetric: one property owns column 1 by theorem, none owns column 2
strongly, and the other two own column 3.
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

PROPS = [("entropy", "byte entropy"),
         ("mad", "MAD"),
         ("second_deriv", "2nd derivative")]
NBINS = 14


def load(d):
    r = pd.read_csv(os.path.join(d, "processed", "exploration_results.csv"),
                    low_memory=False).replace([np.inf, -np.inf], np.nan)
    r = r[(r["ratio"] > 0) & (r["entropy"] > 0)].dropna(
        subset=["mad", "second_deriv"])
    ll = r[r["quantize"] == 0].copy()
    ll["bound"] = 8.0 / ll["entropy"]
    ll["excess"] = ll["ratio"] / ll["bound"]
    qz = r[(r["quantize"] == 1) & (r["error_bound"] > 0)].copy()
    return ll, qz


def panel(ax, frames, pcol, ycol, logy, ylabel, title):
    """One line per workload: median y binned by the property's OWN VALUE.

    x is the property in its real units, not a rank. The bins are per
    workload quantiles -- so each line has equal support per point -- but they
    are PLOTTED at the bin's median property value, so a reader can see both
    the trend and the number. That puts each workload wherever its data
    actually lives, which is the honest picture: VPIC's MAD spans 1e-9..0.2
    and WarpX's reaches 1e13, and that separation is itself a finding.
    """
    logx = False
    for k, (name, d) in enumerate(frames):
        d = d[[pcol, ycol]].replace([np.inf, -np.inf], np.nan).dropna()
        if logy:
            d = d[d[ycol] > 0]
        if len(d) < 100 or d[pcol].nunique() < 4:
            continue
        rho = float(sps.spearmanr(d[pcol], d[ycol]).correlation)
        # Quantile bins for equal support; plotted at the bin's own value.
        try:
            bins = pd.qcut(d[pcol], NBINS, duplicates="drop")
        except ValueError:
            continue
        g = d.groupby(bins, observed=True)
        q = g[ycol].quantile([0.25, 0.5, 0.75]).unstack().dropna()
        x = g[pcol].median().reindex(q.index)
        pos = d[pcol][d[pcol] > 0]
        if len(pos) and pos.max() / pos.min() > 100:
            logx = True
        c = plotting.SERIES[k % len(plotting.SERIES)]
        ax.fill_between(x, q[0.25], q[0.75], color=c, alpha=0.12, linewidth=0)
        ax.plot(x, q[0.5], color=c, lw=2.0, marker="o", ms=3.5,
                mec=plotting.SURFACE, mew=0.7)
        ax.annotate(f"{name}  {rho:+.2f}", xy=(0.97, 0.93 - 0.11 * k),
                    xycoords="axes fraction", ha="right", fontsize=7.5,
                    color=c)
    if logy:
        ax.set_yscale("log")
    if logx:
        ax.set_xscale("log")
    if ycol == "excess":
        ax.axhline(1.0, color=plotting.INK_2, lw=1.0, ls="--")
    ax.set_ylabel(ylabel, fontsize=8)
    if title:
        ax.set_title(title, fontsize=9.5)
    plotting._finish(ax)


def main(out, *specs):
    plotting._style()
    runs = [(s.split("=", 1)[0], s.split("=", 1)[1]) for s in specs]
    data = {n: load(d) for n, d in runs}
    ll = [(n, data[n][0]) for n, _ in runs]
    qz = [(n, data[n][1]) for n, _ in runs]

    cols = [("LOSSLESS ratio\n(bounded by 8/H)", ll, "ratio", True,
             "achieved ratio"),
            ("EXCESS over the bound\nratio / (8/H) -- the ORDER part", ll,
             "excess", True, "excess (1.0 = at the bound)"),
            ("QUANTIZED ratio\n(H no longer applies)", qz, "ratio", True,
             "achieved ratio")]
    fig, axes = plt.subplots(len(PROPS), 3, figsize=(13.4, 3.2 * len(PROPS)),
                             squeeze=False)
    for i, (pcol, plabel) in enumerate(PROPS):
        for j, (title, frames, ycol, logy, ylab) in enumerate(cols):
            panel(axes[i][j], frames, pcol, ycol, logy,
                  ylab if j == 0 else "", title if i == 0 else "")
            if j == 0:
                axes[i][j].set_ylabel(f"{plabel}\n\n{ylab}", fontsize=8.5)
            axes[i][j].set_xlabel(
                {"entropy": "byte entropy (bits/byte)",
                 "mad": "MAD (raw data units, log)",
                 "second_deriv": "mean |2nd difference| (raw units, log)"
                 }[pcol], fontsize=7.5)
            if i == len(PROPS) - 1:
                unit = {"entropy": "byte entropy (bits/byte)",
                        "mad": "MAD (raw data units, log)",
                        "second_deriv":
                            "mean |2nd difference| (raw units, log)"}[pcol]
                axes[i][j].set_xlabel(unit, fontsize=8)
    fig.suptitle("What each property does to the ratio -- every (chunk, "
                 "configuration) row, median per percentile bin\n"
                 "row = the property (x is its REAL VALUE); column = which part "
                 "of the ratio; number = Spearman rho", fontsize=11, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}\n")

    hdr = f"{'':<14} {'lossless ratio':>16} {'excess over bound':>19} {'quantized ratio':>17}"
    for name, _ in runs:
        print(f"{name}\n{hdr}")
        for pcol, plabel in PROPS:
            cells = []
            for _, frames, ycol, _, _ in cols:
                d = dict(frames)[name][[pcol, ycol]].replace(
                    [np.inf, -np.inf], np.nan).dropna()
                r = float(sps.spearmanr(d[pcol], d[ycol]).correlation) \
                    if len(d) > 8 else float("nan")
                cells.append(f"{r:+.3f}")
            print(f"  {plabel:<12} {cells[0]:>16} {cells[1]:>19} "
                  f"{cells[2]:>17}")
        print()
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(2)
    sys.exit(main(sys.argv[1], *sys.argv[2:]))
