#!/usr/bin/env python3
"""Per-chunk prediction error of NeuroPress v2 inside Clio, over the stream.

    plot_v2_adapt.py [--out PNG]

For every chunk the run compressed with a v2 setting (selection.csv,
role=primary), the error of v2's prediction for the setting it picked,
against what Clio then measured for that same chunk: |pred - measured| /
measured, for compress time and for compression ratio. Points are chunks in
stream order; lines are the MAPE (mean error) over a rolling window of 25
chunks, labelled with the value at the start and the end of the stream. Rows: Nyx (where the
shipped weights are far off) and VPIC (where they are already close).
Runs: run_v2_adapt.sh, learning rate 0.5 (chosen on Nyx).
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = "/mnt/nvme0/v2-adapt/runs"
DATASETS = [  # label, {arm: run dir}
    ("Nyx", {"static": "nyx_static_lr0.01", "learn": "nyx_learn_lr0.5",
             "learn + explore": "nyx_explore_lr0.5"}),
    ("VPIC", {"static": "vpic_static_lr0.5", "learn": "vpic_learn_lr0.5",
              "learn + explore": "vpic_explore_lr0.5"}),
]
COLOURS = {"static": "#6c757d", "learn": "#2e86ab", "learn + explore": "#d1495b"}
INK, INK2 = "#1f2328", "#57606a"


def chunk_errors(run):
    """Per-chunk APE of the picked setting's ct and ratio, in stream order."""
    s = pd.read_csv(os.path.join(RUNS, run, "selection.csv"))
    s = s[(s.role == "primary") & (s.wire_lib == 24)].reset_index(drop=True)
    ct = (s.pred_ct_ms - s.actual_ct_ms).abs() / s.actual_ct_ms
    ratio = (s.pred_ratio - s.actual_ratio).abs() / s.actual_ratio
    return ct.to_numpy(), ratio.to_numpy()


def Pct(v, _pos):
    """A percent as a plain number: 0.5%, 12%, 1,000%, 2,000,000%."""
    if not np.isfinite(v) or v <= 0:
        return ""
    return f"{v:,.1f}%" if v < 10 else f"{v:,.0f}%"


def panel(ax, series, title, show_legend):
    """Scatter of per-chunk errors and their rolling MAPE, log y."""
    ends = []
    for arm, y in series.items():
        x = np.arange(len(y))
        ax.scatter(x, 100 * y, s=6, color=COLOURS[arm], alpha=0.25,
                   linewidths=0)
        mape = pd.Series(100 * y).rolling(25, center=True, min_periods=8).mean()
        ax.plot(x, mape, color=COLOURS[arm], lw=2, label=arm)
        ends.append((mape.iloc[-1], arm, x[-1]))
    # End-of-stream MAPE labels, nudged apart so close values stay readable.
    ends.sort()
    allv = 100 * np.concatenate([np.asarray(y) for y in series.values()])
    allv = allv[np.isfinite(allv) & (allv > 0)]
    sep = 0.05 * (np.log10(allv.max()) - np.log10(allv.min()))
    last_log = -np.inf
    for v, arm, xe in ends:
        lv = np.log10(v)
        shown = max(lv, last_log + sep)
        last_log = shown
        ax.annotate(Pct(v, None), (xe, v), xytext=(xe + 6, 10 ** shown),
                    textcoords="data", ha="left", va="center", fontsize=8.5,
                    color=COLOURS[arm], weight="bold")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(Pct))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, p: ""))
    ax.set_xlim(-5, len(y) + 45)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    ax.set_xlabel("chunk, in stream order", fontsize=9, color=INK2)
    ax.set_ylabel("prediction error (MAPE)", fontsize=9, color=INK2)
    ax.grid(axis="y", which="major", color="#d0d7de", lw=0.6)
    ax.tick_params(labelsize=8, colors=INK2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if show_legend:
        ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper right")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(
        HERE, "..", "figures", "new-workloads", "nn-v2", "v2_adaptation_per_chunk.png"))
    a = ap.parse_args()
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.patch.set_facecolor("white")
    for r, (name, runs) in enumerate(DATASETS):
        errs = {arm: chunk_errors(run) for arm, run in runs.items()}
        panel(axes[r, 0], {k: v[0] for k, v in errs.items()},
              f"{name}: compress time of the picked setting", r == 0)
        panel(axes[r, 1], {k: v[1] for k, v in errs.items()},
              f"{name}: compression ratio of the picked setting", False)
    fig.suptitle("NeuroPress v2 inside Clio: per-chunk prediction error vs what "
                 "Clio measured, as the stream goes on",
                 x=0.02, ha="left", fontsize=12.5, color=INK, y=0.985)
    fig.text(0.02, 0.918, "Each point is one chunk (the setting v2 picked, "
             "predicted before learning from that chunk), against what Clio measured.\n"
             "Lines: MAPE over a rolling window of 25 chunks; the number at the right "
             "is the MAPE at the end of the stream. Learning rate 0.5; exploration K = 3.",
             fontsize=9, color=INK2)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.875, bottom=0.07,
                        hspace=0.35, wspace=0.18)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=160)
    fig.savefig(os.path.splitext(a.out)[0] + ".pdf")
    print("wrote", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
