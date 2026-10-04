#!/usr/bin/env python3
"""Per-chunk regret of the setting NeuroPress v2 stored inside Clio.

    plot_v2_regret.py [--out PNG]

For every chunk, regret = cost of the stored setting / cost of the cheapest
of the 45 settings - 1, both from the codec sweep's measurements of that
chunk (cost = compress ms + decompress ms + bytes at 1 GB/s). 0 % means v2
stored the best setting. One bar per chunk in stream order; green ticks mark
chunks where it stored the best one. Columns: VPIC, Nyx; rows: static v2,
online learning, learning plus exploration. Each panel states how often the
best was stored, the median and the mean regret.
Data: eval_v2_adapt.py's *_chunks.csv for the run_v2_adapt.sh runs.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = "/mnt/nvme0/v2-adapt/runs"
DATASETS = [  # title, chunks csv, {row label: run}
    ("VPIC", "vpic_eval_chunks.csv",
     {"static v2": "vpic_static_lr0.5", "learn": "vpic_learn_lr0.5",
      "learn + explore": "vpic_explore_lr0.5"}),
    ("Nyx", "nyx_eval3_chunks.csv",
     {"static v2": "nyx_static_lr0.01", "learn": "nyx_learn_lr0.5",
      "learn + explore": "nyx_explore_lr0.5"}),
]
COLOURS = {"static v2": "#6c757d", "learn": "#2e86ab", "learn + explore": "#d1495b"}
INK, INK2 = "#1f2328", "#57606a"
TICKS = [0, 1, 10, 100, 1000]


def pct(v, _pos=None):
    """A percent as a plain number."""
    return f"{v:,.0f}%"


def panel(ax, regret, label, title):
    """One bar per chunk on a symlog axis; green ticks where regret is 0."""
    x = np.arange(len(regret))
    best = regret <= 1e-9
    ax.bar(x, regret, width=1.0, color=COLOURS[label], lw=0)
    ax.scatter(x[best], np.zeros(best.sum()), marker="|", s=40,
               color="#2ca02c", linewidths=1.2, zorder=3)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylim(0, 2000)
    ax.yaxis.set_major_locator(FixedLocator(TICKS))
    ax.yaxis.set_major_formatter(FuncFormatter(pct))
    ax.set_xlim(-1, len(regret))
    ax.grid(axis="y", color="#d0d7de", lw=0.6)
    ax.tick_params(labelsize=8, colors=INK2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.text(0.99, 0.93,
            f"best stored on {100 * best.mean():.0f}% of chunks   "
            f"median {np.median(regret):.1f}%   mean {np.mean(regret):.1f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            color=INK, bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                 ec="#d0d7de"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(
        HERE, "..", "figures", "new-workloads", "nn-v2", "v2_regret_per_chunk.png"))
    a = ap.parse_args()
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(3, 2, figsize=(14, 8.4), sharey=True)
    fig.patch.set_facecolor("white")
    for col, (name, csv, runs) in enumerate(DATASETS):
        chunks = pd.read_csv(os.path.join(RUNS, csv))
        for row, (label, run) in enumerate(runs.items()):
            regret = 100 * chunks[chunks.run == run].regret_stored.to_numpy()
            panel(axes[row, col], regret, label, f"{name} -- {label}")
            if row == 2:
                axes[row, col].set_xlabel("chunk, in stream order", fontsize=9,
                                          color=INK2)
            if col == 0:
                axes[row, col].set_ylabel("regret", fontsize=9, color=INK2)
    fig.suptitle("NeuroPress v2 inside Clio: how far each chunk's stored setting "
                 "is from that chunk's best", x=0.02, ha="left", fontsize=12.5,
                 color=INK, y=0.99)
    fig.text(0.02, 0.928, "Regret = cost of the stored setting / cost of the "
             "chunk's cheapest setting - 1, from the sweep's measurements of "
             "all 45 settings on that chunk (cost = compress + decompress + "
             "bytes at 1 GB/s).\n0% = the best setting was stored (green "
             "ticks). The sweep's times are ~1.8x faster than Clio's, so this "
             "best is the sweep's best, not necessarily Clio's.",
             fontsize=8.8, color=INK2)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.07,
                        hspace=0.45, wspace=0.08)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=160)
    fig.savefig(os.path.splitext(a.out)[0] + ".pdf")
    print("wrote", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
