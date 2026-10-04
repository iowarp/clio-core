#!/usr/bin/env python3
"""Which codecs stay within 20% of the best single codec on every workload.

Uses the same four-tier weighted placement and cost model as
cost_model_hierarchy.py. One figure per engine (GPU, CPU): a heatmap of
(cost / best - 1) per codec x workload. A codec "always" qualifies when
every cell in its row is <= 20%.

  always_within.py --float32 DIR [--cpu DIR] [--model balanced]
                   [--margin 0.20] [--out DIR]
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import cost_model_hierarchy as hier
import cost_model_per_chunk as pc
import cost_model_tiers as cm


def gap_table(res):
    """Gap of each codec vs the best single codec, per workload.

    @param res workload -> evaluate() dict (has "fixed" and "best")
    @return DataFrame index=codec key, columns=workload, values=cost/best - 1
    """
    wls = [w for w in cm.WORKLOADS if w in res]
    keys = sorted({c for r in res.values() for c in r["fixed"].index},
                  key=lambda c: (cm.CODECS.get(c.partition("|")[0], c), c))
    rows = {}
    for c in keys:
        rows[c] = {w: (res[w]["fixed"][c] / res[w]["fixed"][res[w]["best"]] - 1
                       if c in res[w]["fixed"] else np.nan)
                   for w in wls}
    return pd.DataFrame(rows, index=wls).T


def always_keys(gaps, margin):
    """Codec keys whose gap is <= margin on every workload (no NaN).

    @param gaps   output of gap_table
    @param margin fraction above the best (0.20 = 20%)
    @return list of keys, worst-case gap ascending
    """
    ok = gaps.notna().all(axis=1) & (gaps.max(axis=1) <= margin)
    return list(gaps.loc[ok].max(axis=1).sort_values().index)


def sort_rows(gaps):
    """Always-within codecs first, then by worst-case gap, then median.

    @param gaps output of gap_table
    @return gaps with rows reordered for the heatmap
    """
    return gaps.assign(_max=gaps.max(axis=1), _med=gaps.median(axis=1)
                       ).sort_values(["_max", "_med"]).drop(
                           columns=["_max", "_med"])


def cell_text(v):
    """@return a compact percent for one heatmap cell."""
    if np.isnan(v):
        return ""
    if v <= 0.0005:
        return "best"
    return f"+{100 * v:.0f}%" if v >= 0.1 else f"+{100 * v:.1f}%"


def plot_heatmap(gaps, always, margin, title, note, out):
    """Draw one engine's within-20% heatmap.

    @param gaps   codec x workload gaps
    @param always keys that stay within margin on every workload
    @param margin fraction above the best
    @param title  figure title
    @param note   subtitle
    @param out    PNG path
    """
    gaps = sort_rows(gaps)
    n, m = gaps.shape
    fig, ax = plt.subplots(figsize=(max(10.5, 0.72 * m + 4.2),
                                    max(5.4, 0.34 * n + 2.4)))
    fig.patch.set_facecolor(cm.SURFACE)
    fig.subplots_adjust(left=0.22, right=0.985, bottom=0.08, top=0.82)
    # Two bins: within the margin (blue) vs outside (warm). Best cells are
    # the same blue, darker via the text label.
    inside = np.where(gaps.values <= margin, 0.0, 1.0)
    inside[np.isnan(gaps.values)] = np.nan
    cmap = ListedColormap(["#d7e8f8", "#f4d4c4"])
    ax.imshow(inside, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i, key in enumerate(gaps.index):
        for j, wl in enumerate(gaps.columns):
            v = gaps.iat[i, j]
            if np.isnan(v):
                continue
            weight = 500 if v <= 0.0005 or key in always else 400
            color = cm.INK if v <= margin else cm.INK2
            ax.text(j, i, cell_text(v), ha="center", va="center", fontsize=7.2,
                    color=color, fontweight=weight)
    ax.set_xticks(range(m))
    ax.set_xticklabels([cm.WORKLOADS[w] for w in gaps.columns], rotation=35,
                       ha="right", fontsize=8.5, color=cm.INK)
    ax.set_yticks(range(n))
    names = [pc.label(k) + ("  ★" if k in always else "")
             for k in gaps.index]
    ax.set_yticklabels(names, fontsize=8.5, color=cm.INK)
    for i, k in enumerate(gaps.index):
        if k in always:
            ax.get_yticklabels()[i].set_fontweight("medium")
            ax.get_yticklabels()[i].set_color(cm.BLUE)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(m) - 0.5, minor=True)
    ax.set_yticks(np.arange(n) - 0.5, minor=True)
    ax.grid(which="minor", color=cm.SURFACE, linewidth=1.1)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_facecolor(cm.SURFACE)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=cm.INK, y=0.97)
    fig.text(0.01, 0.905, note, fontsize=9, color=cm.INK2, wrap=True,
             va="top")
    fig.savefig(out, dpi=170, facecolor=cm.SURFACE)
    plt.close(fig)


def score(data, model):
    """@return workload -> evaluate() under weighted four-tier placement."""
    bw = {"dram": cm.FIXED_TIERS["dram"][1], "nvme": cm.FIXED_TIERS["nvme"][1],
          "burst_buffer": cm.FIXED_TIERS["ssd"][1],
          "lustre": cm.FIXED_TIERS["hdd"][1]}
    res = {}
    for w, df in data.items():
        placed, tier = hier.place_weighted(df)
        res[w] = pc.evaluate(placed, tier.map(bw), model)
    return res


def run_engine(data, engine, model, margin, out_dir):
    """Score one engine, write its heatmap, print which codecs always qualify.

    @param data    workload -> measurements
    @param engine  "gpu" or "cpu", for the file name and title
    @param model   cost-model key
    @param margin  fraction above the best
    @param out_dir output directory
    @return path of the PNG
    """
    res = score(data, model)
    gaps = gap_table(res)
    always = always_keys(gaps, margin)
    pct = f"{100 * margin:.0f}%"
    if always:
        names = ", ".join(pc.label(c) for c in always)
        answer = (f"{len(always)} codec{'s' if len(always) != 1 else ''} "
                  f"always within {pct} of the best: {names}.")
    else:
        answer = f"No codec stays within {pct} of the best on every workload."
    where = ("GPU codecs, A100, data in GPU memory"
             if engine == "gpu" else
             "CPU codecs, 2x EPYC 7763, data in DRAM, best of 1 and 128 threads")
    what = pc.OBJECTIVES[model][1]
    note = (f"{answer} Blue cell = within {pct} of that workload's best single "
            f"codec; peach = outside. Cost = {what}, four-tier hierarchy "
            f"(10/30/30/30% at 12 / 1 / 0.512 / 0.25 GB/s). {where}.")
    path = os.path.join(out_dir, f"always_within_{engine}.png")
    kind = "GPU" if engine == "gpu" else "CPU"
    plot_heatmap(gaps, always, margin,
                 f"Does any {kind} codec stay within {pct} of the best "
                 f"on every workload?",
                 note, path)
    print(f"{engine}: {answer}")
    for c in always:
        row = gaps.loc[c]
        print(f"  {pc.label(c)}: worst +{100 * row.max():.1f}% "
              f"({cm.WORKLOADS[row.idxmax()]})")
    print(f"wrote {path}")
    return path


def main():
    """Load GPU and/or CPU sweeps and write the always-within heatmaps."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--float32", help="GPU run_float32_sweep.sh results dir")
    ap.add_argument("--cpu", help="CPU results dir")
    ap.add_argument("--model", default="balanced",
                    choices=list(pc.OBJECTIVES))
    ap.add_argument("--margin", type=float, default=0.20,
                    help="fraction above the best (default 0.20)")
    ap.add_argument("--out", default=pc.DEFAULT_OUT.replace("/per-chunk",
                                                            "/float32-smoke"))
    a = ap.parse_args()
    if not a.float32 and not a.cpu:
        sys.exit("need --float32 and/or --cpu")
    os.makedirs(a.out, exist_ok=True)

    class Args:
        pass
    ns = Args()
    ns.float32, ns.cpu = a.float32, a.cpu
    if a.float32:
        ns.engines = "gpu"
        run_engine(pc.load_data(ns), "gpu", a.model, a.margin, a.out)
    if a.cpu:
        ns.engines = "cpu"
        run_engine(pc.load_data(ns), "cpu", a.model, a.margin, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
