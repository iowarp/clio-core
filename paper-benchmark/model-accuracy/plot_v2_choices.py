#!/usr/bin/env python3
"""Which setting NeuroPress v2 chose for each chunk inside Clio.

    plot_v2_choices.py [--out PNG]

One panel per dataset (VPIC, Nyx). Rows, top to bottom: the cheapest
setting for that chunk according to the codec sweep's measurements (cost =
compress + decompress + bytes at 1 GB/s), then the setting v2 stored with
the shipped weights (static), with online learning, and with learning plus
exploration. Each column is one chunk in stream order, coloured by setting;
the right margin gives how often a row matches the sweep's cheapest.
Data: eval_v2_adapt.py's *_chunks.csv for the run_v2_adapt.sh runs.
Note: the sweep's times are not Clio's (Clio's run about 1.8x slower), so
"matches the sweep's cheapest" is a reference, not the deployment optimum.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

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
INK, INK2 = "#1f2328", "#57606a"
# A fixed colour per codec family, shades for its shuffle forms, so the
# strips read by codec first and by shuffle second.
FAMILY = {"spratio": "#1f77b4", "spspeed": "#17becf", "ndzip": "#2ca02c",
          "ans": "#d62728", "bitcomp": "#9467bd", "cascaded": "#8c564b",
          "lz4": "#ff7f0e", "zstd": "#e377c2", "deflate": "#bcbd22",
          "gdeflate": "#7f7f7f", "snappy": "#aec7e8", "gpulz": "#98df8a",
          "store": "#000000"}


def colour(setting):
    """Family colour, lighter for byte shuffle, darker for bit shuffle."""
    base = np.array(matplotlib.colors.to_rgb(FAMILY[setting.split()[0]]))
    if "shuffle=byte" in setting:
        return tuple(base + (1 - base) * 0.45)
    if "shuffle=bit" in setting:
        return tuple(base * 0.6)
    return tuple(base)


def strips(ax, chunks, runs, title):
    """Rows of per-chunk choices, the sweep's cheapest on top."""
    first = chunks[chunks.run == list(runs.values())[0]]
    rows = [("cheapest (sweep)", first.best.tolist())]
    rows += [(lab, chunks[chunks.run == r].stored.tolist())
             for lab, r in runs.items()]
    best = rows[0][1]
    used = []
    for k, (lab, seq) in enumerate(rows):
        for i, s in enumerate(seq):
            ax.add_patch(plt.Rectangle((i, len(rows) - 1 - k), 1, 0.86,
                                       color=colour(s), lw=0))
            if s not in used:
                used.append(s)
        if k > 0:
            match = np.mean([a == b for a, b in zip(seq, best)])
            ax.text(len(seq) + 3, len(rows) - 1 - k + 0.43,
                    f"{100 * match:.0f}% = cheapest", va="center",
                    fontsize=8.5, color=INK2)
    ax.set_xlim(0, len(best) + 40)
    ax.set_ylim(-0.1, len(rows))
    ax.set_yticks([len(rows) - 1 - k + 0.43 for k in range(len(rows))])
    ax.set_yticklabels([lab for lab, _ in rows], fontsize=9, color=INK)
    ax.set_xlabel("chunk, in stream order", fontsize=9, color=INK2)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.tick_params(length=0, labelsize=8, colors=INK2)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    return used


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(
        HERE, "..", "figures", "new-workloads", "nn-v2", "v2_choices_per_chunk.png"))
    a = ap.parse_args()
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(2, 1, figsize=(13, 9.2))
    fig.patch.set_facecolor("white")
    used = []
    for ax, (title, csv, runs) in zip(axes, DATASETS):
        chunks = pd.read_csv(os.path.join(RUNS, csv))
        for s in strips(ax, chunks, runs, f"{title}: setting stored for each chunk"):
            if s not in used:
                used.append(s)
    used.sort(key=lambda s: (list(FAMILY).index(s.split()[0]), s))
    fig.legend(handles=[Patch(color=colour(s), label=s) for s in used],
               loc="lower center", ncol=4, frameon=False, fontsize=8.5,
               labelcolor=INK2, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("NeuroPress v2 inside Clio: the codec setting chosen for each chunk",
                 x=0.02, ha="left", fontsize=12.5, color=INK, y=0.99)
    fig.text(0.02, 0.928, "Top row: the cheapest setting by the sweep's measurements "
             "(compress + decompress + bytes at 1 GB/s). Colour = codec; lighter = "
             "byte shuffle, darker = bit shuffle.\nClio's own timings are ~1.8x the "
             "sweep's, so a learning run that adapts to them can differ from the "
             "top row and still be right for Clio.", fontsize=8.8, color=INK2)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.87, bottom=0.33, hspace=0.45)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=160)
    fig.savefig(os.path.splitext(a.out)[0] + ".pdf")
    print("wrote", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
