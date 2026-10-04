#!/usr/bin/env python3
"""Which codecs stay close to the best one on every workload?

For each workload, every codec (at its best setting, shuffle variants apart)
is scored on the four-tier hierarchy exactly as in cost_model_hierarchy.py
(weighted placement, 10/30/30/30, FIXED_TIERS bandwidths), and its gap is

    gap = cost(codec) / cost(best single codec on that workload) - 1

The figure is a codec x workload heatmap of the gaps, one panel for the GPU
sweep and one for the CPU sweep, rows sorted by the worst gap over all
workloads. Codecs whose worst gap is at most --threshold (20%) sit above the
dashed line: they were within that margin of the top pick everywhere.

  cost_model_robust.py --float32 GPU_DIR --cpu CPU_DIR
                       [--model balanced|compression] [--threshold 0.2]
                       [--out DIR]

Writes <out>/robust_<model>.png and prints each panel's table.
"""
import argparse
import os
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import cost_model_hierarchy as hi  # noqa: E402
import cost_model_per_chunk as pc  # noqa: E402
import cost_model_tiers as cm  # noqa: E402

# Within the threshold: green; beyond it: from sand to red at CAP.
NEAR = LinearSegmentedColormap.from_list("near", ["#1f8a4c", "#bfe3c6"])
FAR = LinearSegmentedColormap.from_list("far", ["#f3e3c3", "#c8463d"])
CAP = 1.0  # gaps beyond 100% share the darkest red


def hierarchy_bw():
    """@return tier -> GB/s for the HIERARCHY tiers, from FIXED_TIERS."""
    fixed = {t: g for t, (_, g) in cm.FIXED_TIERS.items()}
    return {"dram": fixed["dram"], "nvme": fixed["nvme"],
            "burst_buffer": fixed["ssd"], "lustre": fixed["hdd"]}


def gap_table(gpu_dir, cpu_dir, engine, model):
    """Gap of every codec to the best one, per workload.

    @param gpu_dir run_float32_sweep.sh GPU results dir
    @param cpu_dir run_float32_sweep.sh ENGINE=cpu results dir
    @param engine  "gpu" or "cpu": whose codecs compete
    @param model   key of cost_model_per_chunk.OBJECTIVES
    @return (DataFrame codec label x workload of gaps, Series workload ->
            label of its best codec)
    """
    a = SimpleNamespace(float32=gpu_dir, cpu=cpu_dir, engines=engine,
                        sweep=None)
    bw = hierarchy_bw()
    gaps, best = {}, {}
    for w, df in pc.load_data(a).items():
        df, tier = hi.place_weighted(df)
        r = pc.evaluate(df, tier.map(bw), model)
        f = r["fixed"]
        gaps[cm.WORKLOADS.get(w, w)] = f / f.min() - 1.0
        best[cm.WORKLOADS.get(w, w)] = pc.label(f.idxmin())
    t = pd.DataFrame(gaps)
    t.index = [pc.label(k) for k in t.index]
    t = t.assign(worst=t.max(axis=1)).sort_values("worst")
    return t, pd.Series(best)


def cell_color(g, threshold):
    """@return the heatmap colour of one gap (fraction)."""
    if g <= threshold:
        return NEAR(g / threshold if threshold else 0.0)
    return FAR(min(1.0, (g - threshold) / (CAP - threshold)))


def plot_panel(ax, t, title, threshold):
    """Draw one engine's heatmap: workloads, then the worst-case column.

    @param ax        axes to draw into
    @param t         gap_table output (rows sorted by worst gap)
    @param title     panel title
    @param threshold gap that counts as "close to the best"
    """
    cols = list(t.columns)
    nr, nc = t.shape
    for i, (_, row) in enumerate(t.iterrows()):
        for j, c in enumerate(cols):
            g = row[c]
            x = j + (0.25 if c == "worst" else 0)  # gap before "worst"
            ax.add_patch(plt.Rectangle((x, i), 1, 1, color=cell_color(
                g, threshold), ec=cm.SURFACE, lw=1))
            txt = "best" if g == 0 else f"{100 * g:.0f}" if g < 9.995 else ">999"
            dark = g <= threshold * 0.5 or g >= 0.75 * CAP
            ax.text(x + 0.5, i + 0.5, txt, ha="center", va="center",
                    fontsize=6.5, color="white" if dark else cm.INK,
                    fontweight="bold" if c == "worst" else "normal")
    n_ok = int((t["worst"] <= threshold).sum())
    if 0 < n_ok < nr:
        ax.axhline(n_ok, color=cm.INK, lw=1.2, ls=(0, (4, 2)))
    ax.set_xlim(0, nc + 0.25)
    # Headroom in data coords for column names and the two-line title, so
    # neither can sit on the figure heading or the other panel.
    ax.set_ylim(nr, -6.8)
    ax.set_xticks([])
    ax.set_yticks(np.arange(nr) + 0.5)
    ax.set_yticklabels(t.index, fontsize=7.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for j, c in enumerate(cols):
        x = j + 0.5 + (0.25 if c == "worst" else 0)
        ax.text(x, -0.2, c if c != "worst" else "Worst", rotation=90,
                ha="center", va="bottom", fontsize=7.5, color=cm.INK)
    names = ", ".join(t.index[:n_ok]) if n_ok else "none"
    ax.text(0, -6.6, title, ha="left", va="top", fontsize=9.5, color=cm.INK)
    ax.text(0, -5.7, f"{n_ok} within {100 * threshold:.0f}% everywhere"
            + (f": {names}" if n_ok else " (none)"),
            ha="left", va="top", fontsize=8, color=cm.INK2)


def main():
    """Score both engines, draw the two heatmaps, print the tables."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--float32", required=True,
                    help="run_float32_sweep.sh GPU results dir")
    ap.add_argument("--cpu", required=True,
                    help="run_float32_sweep.sh ENGINE=cpu results dir")
    ap.add_argument("--model", default="balanced",
                    choices=["balanced", "compression"])
    ap.add_argument("--threshold", type=float, default=0.2,
                    help="gap to the best codec that counts as close (0.2)")
    ap.add_argument("--out", default=pc.DEFAULT_OUT)
    a = ap.parse_args()
    panels = []
    for eng, name in (("gpu", "GPU codecs (A100)"),
                      ("cpu", "CPU codecs (2x EPYC 7763)")):
        t, best = gap_table(a.float32, a.cpu, eng, a.model)
        panels.append((t, name))
        with pd.option_context("display.width", 250, "display.precision", 3):
            print(f"== {name}\n{(100 * t).round(1).to_string()}")
    rows = [len(t) for t, _ in panels]
    height = 0.24 * (sum(rows) + 14) + 3.4
    fig, axes = plt.subplots(2, 1, figsize=(10, height),
                             gridspec_kw={"height_ratios": [r + 7.0 for r in rows],
                                          "hspace": 0.16})
    fig.patch.set_facecolor(cm.SURFACE)
    fig.subplots_adjust(left=0.18, right=0.985, bottom=0.02,
                        top=1 - 1.55 / height)
    for ax, (t, name) in zip(axes, panels):
        plot_panel(ax, t, name, a.threshold)
    bw = hierarchy_bw()
    tiers = ", ".join(f"{10 * n}% {lab} {bw[k]:g} GB/s"
                      for k, (lab, n) in hi.HIERARCHY.items())
    fig.suptitle(f"{pc.OBJECTIVES[a.model][0]} cost: % above the best single "
                 f"codec of each workload", x=0.01, ha="left", fontsize=13,
                 color=cm.INK, y=1 - 0.16 / height)
    fig.text(0.01, 1 - 0.55 / height,
             "Cost = " + pc.OBJECTIVES[a.model][1]
             + " on the four-tier hierarchy\n"
             + f"({tiers}; each chunk on every tier, weighted).",
             fontsize=8.5, color=cm.INK2, va="top")
    fig.text(0.01, 1 - 1.05 / height,
             "Each codec at its best setting; 'best' = that workload's 1st. "
             f"Green = within {100 * a.threshold:.0f}%.\n"
             "Rows sorted by worst gap; those above the dashed line stay "
             "within it everywhere. GPU and CPU ranked separately.",
             fontsize=8.5, color=cm.INK2, va="top")
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, f"robust_{a.model}.png")
    fig.savefig(out, dpi=200, facecolor=fig.get_facecolor())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
