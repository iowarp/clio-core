#!/usr/bin/env python3
"""Paper figures: why the lossless codec must be chosen automatically.

    plot_paper_mismatch.py [--out DIR]

Writes three figures, each as PDF (vector) and PNG, sized for a two-column
paper (7 in wide):

  mismatch_heatmap      one fixed codec for a whole dataset, against that
                        dataset's own best codec: extra cost (%)
  mismatch_within       inside one workflow, the best codec of one array
                        used on another array: compression ratio and cost
  pipeline_ratio_time   measured producer -> storage -> consumer runs: how
                        much more per-chunk selection compresses, and the
                        end-to-end time

Inputs: results/cross-penalty/{totals,products}.csv (plot_cross_penalty.py,
cross_penalty.py) and ~/np-pipeline/<workload>/runs.csv (pipeline.py).
Cost = compress + decompress time + compressed bytes / bandwidth, chunks
spread 10/30/30/30 over tiers at 12 / 1 / 0.5 / 0.25 GB/s, volume-weighted.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "cross-penalty")
PIPE = os.path.expanduser("~/np-pipeline")

INK, INK2, GRID = "#1a1a1a", "#555555", "#dddddd"
OURS, BASE, NONE = "#d9531e", "#8c8c8c", "#cfcfcf"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5,
                     "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "axes.edgecolor": INK2, "axes.linewidth": 0.6, "pdf.fonttype": 42})

# Rows, grouped by domain: (sweep key, label)
GROUPS = [
    ("Simulation", [("ref-nyx-256-2000", "Nyx (cosmology)"),
                           ("ref-vpic-126-2000", "VPIC (plasma)"),
                           ("ref-warpx-64x64x512-2000", "WarpX (plasma)"),
                           ("ref-lammps-b70-2000", "LAMMPS (molecular)")]),
    ("Graphs", [("graph-orkut-full", "Orkut")]),
]   # analysis outputs, experiments, LiveJournal and deep learning removed (2026-10-05)
# Readable codec names: setting string -> label
NAMES = {"spratio": "SPratio", "ans shuffle=byte": "ANS\n+4-byte shuffle",
         "bitcomp algo=1 type=int": "Bitcomp\nsparse int32", "ans type=float16": "ANS\nfloat16",
         "ans elem=2 shuffle=byte": "ANS\n+2-byte shuffle", "spratio shuffle=bit": "SPratio\n+bit shuffle",
         "spspeed elem=2 shuffle=byte": "SPspeed\n+2-byte shuffle", "zstd": "Zstd", "lz4": "LZ4",
         "spspeed shuffle=bit": "SPspeed\n+bit shuffle", "spspeed": "SPspeed",
         "cascaded bp=1 delta=1 rle=0 type=longlong": "Cascaded\n(delta, int64)"}


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=300, bbox_inches="tight", pad_inches=0.03)
    print("wrote", os.path.join(out, name) + ".{pdf,png}")


def heatmap(out):
    """Extra cost of one fixed codec per dataset, rows grouped by domain."""
    t = pd.read_csv(os.path.join(RES, "totals.csv"))
    cost = t.pivot(index="dataset", columns="config", values="cost")
    rel = 100 * (cost.div(cost.min(axis=1), axis=0) - 1)
    best = cost.idxmin(axis=1)
    winners = best.value_counts().index.tolist()
    cols = winners + ["zstd", "lz4"]
    rows, labels, seps, group_pos = [], [], [], []
    for g, items in GROUPS:
        group_pos.append((g, len(rows), len(rows) + len(items) - 1))
        for k, lab in items:
            rows.append(k)
            labels.append(lab)
        seps.append(len(rows))
    m = rel.loc[rows, cols].to_numpy()
    # gap column between the winners and the common defaults
    gap = len(winners)
    x = np.array([j if j < gap else j + 0.35 for j in range(len(cols))])
    bounds = [0, 0.5, 10, 25, 50, 100, 300, 1e6]
    colors = ["#ffffff", "#fde9c9", "#fcc98b", "#f79a55", "#e8622c", "#c02d1a", "#7f1610"]
    cmap, norm = ListedColormap(colors), BoundaryNorm(bounds, len(colors))
    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = m[i, j]
            is_best = cols[j] == best[rows[i]]
            ax.add_patch(plt.Rectangle((x[j] - 0.5, i - 0.5), 1, 1, facecolor=cmap(norm(v)),
                                       edgecolor="white", lw=0.8))
            if is_best:
                ax.add_patch(plt.Rectangle((x[j] - 0.46, i - 0.46), 0.92, 0.92, fill=False,
                                           ec=INK, lw=1.2))
            txt = "best" if is_best else (f"{v:.0f}" if v < 1000 else f"{v / 1000:.1f}k")
            ax.text(x[j], i, txt, ha="center", va="center", fontsize=6.8,
                    color="white" if v >= 100 else INK, weight="bold" if is_best else "normal")
    ax.set_xlim(-0.5, x[-1] + 0.5)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([NAMES.get(c, c).replace("\n", " ") for c in cols], fontsize=7.2,
                       rotation=35, ha="left", rotation_mode="anchor")
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for s in seps[:-1]:
        ax.axhline(s - 0.5, color=INK, lw=0.8)
    for g, a, b in group_pos:  # domain labels in the left margin
        ax.annotate(g, xy=(0, (a + b) / 2), xycoords=("axes fraction", "data"),
                    xytext=(-104, 0), textcoords="offset points", rotation=90, ha="center",
                    va="center", fontsize=7, color=INK2, weight="bold")
    nwin = best.value_counts()
    ax.annotate(f"codecs that are the best for at least one dataset ({len(winners)} different)",
                xy=((gap - 1) / 2, -0.5), xytext=(0, 62), textcoords="offset points",
                ha="center", fontsize=7.5, color=INK2)
    ax.annotate("common defaults", xy=(x[gap] + 0.5, -0.5), xytext=(0, 62),
                textcoords="offset points", ha="center", fontsize=7.5, color=INK2)
    # mean row under the table
    mean = rel.loc[rows, cols].mean().to_numpy()
    for j in range(len(cols)):
        ax.text(x[j], len(rows) + 0.15, f"{mean[j]:.0f}", ha="center", va="center",
                fontsize=7, color=INK, weight="bold")
    ax.text(-0.62, len(rows) + 0.15, "mean extra cost (%)", ha="right", va="center",
            fontsize=7.5, color=INK, weight="bold")
    from matplotlib.patches import Patch
    bins = ["< 10", "10-25", "25-50", "50-100", "100-300", "> 300"]
    handles = [Patch(facecolor="white", edgecolor=INK, lw=1.2, label="the dataset's own best")]
    handles += [Patch(facecolor=c, edgecolor=INK2, lw=0.3, label=l) for l, c in zip(bins, colors[1:])]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.05), ncol=7, frameon=False,
               fontsize=7.2, handlelength=1.1, columnspacing=0.9,
               title="cell = extra cost (%) of using the column's codec for the whole row's dataset, "
                     "vs that dataset's own best codec", title_fontsize=7.5)
    del nwin
    save(fig, out, "mismatch_heatmap")
    plt.close(fig)


INSIDE = [  # dataset, borrowed-from array, used-on array, workflow, borrowed-from label, used-on label
    ("graph-orkut-full", "cc_label.i32", "edges_src.i32", "Orkut graph", "component labels", "edge list"),
    ("ref-vpic-126-2000", "ex.f32", "div_e_err.f32", "VPIC simulation", "electric field",
     "div(E) error"),
]


def within(out):
    """Best codec of one array of a workflow, used on another of its arrays."""
    p = pd.read_csv(os.path.join(RES, "products.csv"))
    fig, ax = plt.subplots(figsize=(7.0, 2.7))
    y = np.arange(len(INSIDE))[::-1] * 1.0
    for yi, (ds, a, b, wf, la, lb) in zip(y, INSIDE):
        r = p[(p.dataset == ds) & (p.best_of == a) & (p.used_on == b)].iloc[0]
        own, bor = r.own_ratio, r.ratio
        ax.plot([bor, own], [yi, yi], color=GRID, lw=3, zorder=1, solid_capstyle="round")
        ax.scatter([own], [yi], s=46, color=OURS, zorder=3)
        ax.scatter([bor], [yi], s=46, color=BASE, zorder=3)
        ax.text(own * 1.15, yi, f"{own:.1f}x  {NAMES.get(r.own_setting, r.own_setting).replace(chr(10), ' ')}",
                va="center", fontsize=7.2, color=OURS)
        ax.text(bor, yi + 0.22, f"{NAMES.get(r.setting, r.setting).replace(chr(10), ' ')}  {bor:.2f}x",
                va="bottom", ha="center", fontsize=7.2, color=BASE)
        ax.text(1.0, yi, f"+{r.cost_penalty_pct:.0f}%", transform=ax.get_yaxis_transform(),
                va="center", ha="left", fontsize=8, color=INK, weight="bold")
        ax.text(0, yi, f"{wf}: {lb}\n(codec chosen for its {la})",
                transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=7.4,
                color=INK)
    ax.set_xscale("log")
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(["1x", "10x", "100x", "1000x"])
    ax.set_xticks([m * 10 ** e for e in range(3) for m in range(2, 10)], minor=True)
    ax.set_xlim(0.7, 1000)
    ax.set_ylim(y.min() - 0.5, y.max() + 0.75)
    ax.set_yticks([])
    ax.set_xlabel("compression ratio of the array (log scale)")
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.text(1.0, y.max() + 0.75, "extra\ncost", transform=ax.get_yaxis_transform(), ha="left",
            va="bottom", fontsize=7.4, color=INK2)
    ax.scatter([], [], s=30, color=OURS, label="the array's own best codec")
    ax.scatter([], [], s=30, color=BASE, label="best codec of another array in the same workflow")
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, frameon=False, fontsize=7.4,
              handletextpad=0.3)
    save(fig, out, "mismatch_within")
    plt.close(fig)


def pipeline(out):
    """Measured runs: extra compression and end-to-end time."""
    names = {"vpic": "VPIC", "lammps": "LAMMPS"}   # nanoaod, fem removed
    rows = []
    for w, n in names.items():
        f = os.path.join(PIPE, w, "runs.csv")
        if not os.path.exists(f):
            continue
        g = pd.read_csv(f).groupby("policy").mean(numeric_only=True)
        rows.append({"name": n, "data": 100 * (g.prod_stored["best"] / g.prod_stored["perchunk"] - 1),
                     "out": (100 * (g.out_stored["best"] / g.out_stored["perchunk"] - 1)
                             if g.out_in["none"] > 0.1e9 else np.nan),
                     **{p: g.end_to_end_s[p] / g.end_to_end_s["none"] for p in ("none", "best", "perchunk")}})
    d = pd.DataFrame(rows)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"wspace": 0.38})
    x = np.arange(len(d))
    w = 0.36
    a1.bar(x - w / 2, d["data"], w, color=OURS, label="producer data")
    a1.bar(x + w / 2, d["out"].fillna(0), w, color="#f2a582", label="analysis outputs")
    for i, r in d.iterrows():
        a1.text(i - w / 2, r["data"] + 2, f"{r['data']:.0f}%", ha="center", fontsize=7)
        if not np.isnan(r["out"]):
            a1.text(i + w / 2, r["out"] + 2, f"{r['out']:.0f}%", ha="center", fontsize=7)
        else:
            a1.text(i + w / 2, 2, "n/a", ha="center", fontsize=6.5, color=INK2)
    a1.set_xticks(x)
    a1.set_xticklabels(d["name"])
    a1.set_ylabel("more compression than the best\nsingle codec (%, higher is better)")
    a1.set_ylim(0, max(d["data"].max(), d["out"].max()) * 1.18)
    a1.legend(frameon=False, fontsize=7, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.12),
              handlelength=1.2)
    a1.set_title("(a) Extra compression from per-chunk selection", loc="left")
    w = 0.26
    for k, (p, col, lab) in enumerate((("none", NONE, "uncompressed"), ("best", BASE, "best single codec"),
                                       ("perchunk", OURS, "per-chunk selection"))):
        a2.bar(x + (k - 1) * w, d[p], w, color=col, label=lab)
        if p != "none":
            for i, v in enumerate(d[p]):
                a2.text(i + (k - 1) * w, v + 0.02, f"{v:.2f}", ha="center", fontsize=6,
                        rotation=90, va="bottom")
    a2.set_xticks(x)
    a2.set_xticklabels(d["name"])
    a2.set_ylabel("end-to-end time, relative to\nuncompressed (lower is better)")
    a2.set_ylim(0, 1.15)
    a2.legend(frameon=False, fontsize=7, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.12),
              handlelength=1.2, columnspacing=0.8)
    a2.set_title("(b) End-to-end time: produce, store, read, analyse", loc="left")
    for ax in (a1, a2):
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(length=2)
    save(fig, out, "pipeline_ratio_time")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "..", "figures", "new-workloads", "paper"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    heatmap(a.out)
    within(a.out)
    pipeline(a.out)


if __name__ == "__main__":
    main()
