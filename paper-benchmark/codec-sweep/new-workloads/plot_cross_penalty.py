#!/usr/bin/env python3
"""Why codec choice must be automatic: one workload's best codec on others.

    plot_cross_penalty.py [--out PNG]

(a) Heatmap: extra cost (%) of using one fixed setting for a whole dataset
    instead of that dataset's own best single setting. Columns are every
    setting that is some dataset's best, plus Zstd and LZ4 as common
    defaults; the bold-outlined cell in each row is that row's own best.
(b) Inside one workflow: the best setting of one array type used on another
    array type of the same dataset -- compression ratio of the borrowed
    setting vs the array's own best, and the extra cost.
(c) Measured end to end (pipeline runs): producer-data and analysis-output
    ratio, best single codec vs per-chunk choice.
Cost = compress + decompress + compressed bytes / bandwidth on the 10/30/30/30
hierarchy (12 / 1 / 0.5 / 0.25 GB/s), volume-weighted over the sweep's
sampled chunks -- as hierarchy_balanced.png. Per-setting totals are cached in
results/cross-penalty/totals.csv.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cross_penalty as cp  # noqa: E402
import plot_new_workloads as pw  # noqa: E402
cm = pw.cm
RES = os.path.join(HERE, "..", "results", "cross-penalty")
PIPE = os.path.expanduser("~/np-pipeline")

# (dataset, best-of array, used-on array, short title) -- from products.csv
INSIDE = [("consumer-vpic-full", "kmeans.i32", "pdf2d.f64", "VPIC analysis:\nk-means labels' best -> 2-D PDF"),
          ("graph-orkut-full", "cc_label.i32", "edges_src.i32", "Orkut graph:\ncomponent labels' best -> edge list"),
          ("sparse-fem", "Flan_1565_row_ptr.i64", "Flan_1565_col_idx.i32", "FEM:\nrow pointers' best -> column indices"),
          ("ref-vpic-126-2000", "ex.f32", "div_e_err.f32", "VPIC sim:\nE-field's best -> div(E) error")]


def totals():
    """dataset, config -> cost, ratio over the whole dataset (cached)."""
    path = os.path.join(RES, "totals.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    parts = []
    for k in pw.DATASETS:
        t = cp.totals(cp.load(k)).reset_index()
        parts.append(t.assign(dataset=k))
    t = pd.concat(parts)
    t.to_csv(path, index=False)
    return t


def heatmap(ax, t):
    pen = t.pivot(index="dataset", columns="config", values="cost")
    ratio = t.pivot(index="dataset", columns="config", values="ratio")
    best = pen.idxmin(axis=1)
    rel = 100 * (pen.div(pen.min(axis=1), axis=0) - 1)
    cols = list(dict.fromkeys(best.value_counts().index.tolist() + ["zstd", "lz4"]))
    rows = [k for k in pw.DATASETS if k in rel.index]
    rows.sort(key=lambda k: -rel.loc[k, "spratio"])
    m = rel.loc[rows, cols]
    im = ax.imshow(m.clip(lower=0.5).to_numpy(), cmap="YlOrRd", norm=LogNorm(1, 1500), aspect="auto")
    for i, k in enumerate(rows):
        for j, c in enumerate(cols):
            v = m.iloc[i, j]
            ax.text(j, i, "best" if c == best[k] else f"+{v:.0f}%", ha="center", va="center",
                    fontsize=7.4, color="white" if v > 200 else cm.INK,
                    weight="bold" if c == best[k] else "normal")
            if c == best[k]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=cm.INK, lw=1.6))
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([pw.short_config(c) for c in cols], rotation=30, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{pw.DATASETS[k]}  ({pw.short_config(best[k])})" for k in rows], fontsize=8.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    mean = m.mean()
    ax.set_title("(a) Extra cost of one fixed codec for a whole dataset vs that dataset's own best "
                 f"(column mean: SPratio +{mean['spratio']:.0f}%, "
                 f"ANS +byte4 +{mean['ans shuffle=byte']:.0f}%)",
                 loc="left", fontsize=10.5, color=cm.INK, pad=8)
    cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("extra cost, % (log)", fontsize=8.5, color=cm.INK2)
    cb.ax.tick_params(labelsize=7.5)
    del ratio


def inside(ax):
    p = pd.read_csv(os.path.join(RES, "products.csv"))
    labels, own, borrowed, pen, names = [], [], [], [], []
    for ds, a, b, title in INSIDE:
        r = p[(p.dataset == ds) & (p.best_of == a) & (p.used_on == b)].iloc[0]
        labels.append(title)
        own.append(r.own_ratio)
        borrowed.append(r.ratio)
        pen.append(r.cost_penalty_pct)
        names.append((pw.short_config(r.own_setting), pw.short_config(r.setting)))
    y = np.arange(len(labels))
    ax.barh(y - 0.2, own, height=0.38, color=pw.cpc.ORACLE, label="the array's own best setting")
    ax.barh(y + 0.2, borrowed, height=0.38, color=cm.AXIS, label="best setting of the other array")
    for i in range(len(labels)):
        ax.text(own[i] * 1.08, y[i] - 0.2, f"{own[i]:.1f}x  {names[i][0]}", va="center", fontsize=8,
                color=cm.INK)
        ax.text(borrowed[i] * 1.08, y[i] + 0.2,
                f"{borrowed[i]:.2f}x  {names[i][1]}   cost +{pen[i]:.0f}%", va="center",
                fontsize=8, color=cm.INK, weight="bold")
    ax.set_xscale("log")
    ax.set_xlim(0.9, max(own) * 30)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("compression ratio of the array (log)", fontsize=8.5, color=cm.INK2)
    cm.style_axes(ax)
    ax.legend(loc="lower right", fontsize=8, frameon=False, labelcolor=cm.INK2)
    ax.set_title("(b) Inside one workflow: the best codec for one array, used on another array",
                 loc="left", fontsize=10.5, color=cm.INK, pad=8)


def measured(ax):
    rows = []
    for w, name in (("vpic", "VPIC"), ("lammps", "LAMMPS"), ("nanoaod", "NanoAOD"), ("fem", "FEM")):
        f = os.path.join(PIPE, w, "runs.csv")
        if not os.path.exists(f):
            continue
        g = pd.read_csv(f).groupby("policy").mean(numeric_only=True)
        e2e = 100 * (g.loc["perchunk", "end_to_end_s"] / g.loc["best", "end_to_end_s"] - 1)
        rows.append((f"{name} data", g.prod_in / g.prod_stored, e2e))
        if g.loc["none", "out_in"] > 0.1e9:
            rows.append((f"{name} outputs", g.out_in / g.out_stored, None))
    y = np.arange(len(rows))
    best = [r[1]["best"] for r in rows]
    pc = [r[1]["perchunk"] for r in rows]
    ax.barh(y - 0.2, best, height=0.38, color=cm.AXIS, label="best single codec")
    ax.barh(y + 0.2, pc, height=0.38, color=pw.cpc.ORACLE, label="per-chunk choice")
    for i, (lab, _, e2e) in enumerate(rows):
        txt = f"{pc[i]:.2f}x  ({100 * (pc[i] / best[i] - 1):+.0f}% ratio"
        txt += f", end-to-end {e2e:+.1f}%)" if e2e is not None else ")"
        ax.text(max(pc[i], best[i]) * 1.05, y[i] + 0.2, txt, va="center", fontsize=8, color=cm.INK)
    ax.set_xscale("log")
    ax.set_xlim(0.9, max(pc) * 12)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("compression ratio (log)", fontsize=8.5, color=cm.INK2)
    cm.style_axes(ax)
    ax.legend(loc="lower right", fontsize=8, frameon=False, labelcolor=cm.INK2)
    ax.set_title("(c) Measured end to end: higher ratio for a modest time gain",
                 loc="left", fontsize=10.5, color=cm.INK, pad=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "..", "figures", "new-workloads",
                                                  "cross_penalty.png"))
    a = ap.parse_args()
    t = totals()
    fig = plt.figure(figsize=(15, 17))
    fig.patch.set_facecolor(cm.SURFACE)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.45, 1], left=0.2, right=0.97, top=0.925,
                          bottom=0.04, hspace=0.3, wspace=0.75)
    heatmap(fig.add_subplot(gs[0, :]), t)
    inside(fig.add_subplot(gs[1, 0]))
    measured(fig.add_subplot(gs[1, 1]))
    fig.suptitle("No single lossless GPU codec is safe: the best for one dataset (or one array) is "
                 "often far from best for another",
                 x=0.01, ha="left", fontsize=13, color=cm.INK, y=0.985)
    fig.text(0.01, 0.951, "(a), (b): modelled cost = compress + decompress + bytes / bandwidth on the "
             "10/30/30/30 hierarchy (12 / 1 / 0.5 / 0.25 GB/s), volume-weighted, 139 lossless settings.\n"
             "(c): measured producer -> NVMe -> consumer runs, mean of 3 reps.",
             fontsize=9, color=cm.INK2)
    fig.savefig(a.out, dpi=150, facecolor=cm.SURFACE)
    print("wrote", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
