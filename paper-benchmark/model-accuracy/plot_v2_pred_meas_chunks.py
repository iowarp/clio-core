#!/usr/bin/env python3
"""Per-chunk predicted vs measured, NeuroPress v2 inside Clio (VPIC, Nyx).

    plot_v2_pred_meas_chunks.py [--out PNG]

Rows: compress time, decompress time, compression ratio of the setting v2
picked for each chunk. Columns: VPIC static, VPIC learn + explore, Nyx static,
Nyx learn + explore. Each chunk (stream order on x) has its measured value
(black, Clio's own measurement) and v2's prediction made before learning
from it (coloured), joined by a thin grey line. The y-range follows the
measured values; a prediction beyond it is drawn at the edge as a triangle
pointing out of the panel. Chunks with no measurement (stored raw) are
skipped. Data: /mnt/nvme0/v2-adapt/per_chunk/*.csv.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "/mnt/nvme0/v2-adapt/per_chunk"
COLUMNS = [("VPIC", "vpic_static", "static v2", "#6c757d"),
           ("VPIC", "vpic_learnexplore", "learn + explore", "#d1495b"),
           ("Nyx", "nyx_static", "static v2", "#6c757d"),
           ("Nyx", "nyx_learnexplore", "learn + explore", "#d1495b")]
METRICS = [("comp_ms", "compress time (ms)"),
           ("decomp_ms", "decompress time (ms)"),
           ("ratio", "compression ratio")]
INK, INK2 = "#1f2328", "#57606a"


def plain(v, _pos=None):
    """A tick value as a plain number."""
    if v <= 0:
        return ""
    if v >= 100:
        return f"{v:,.0f}"
    if v >= 1:
        return f"{v:g}"
    return f"{v:.3g}"


def panel(ax, d, key, colour):
    """Measured vs predicted for one metric of one run."""
    pred = d[f"pred_{key}"].to_numpy(float)
    meas = d[f"meas_{key}"].to_numpy(float)
    ok = np.isfinite(meas) & (meas > 0)
    x = d["i"].to_numpy()[ok]
    pred, meas = pred[ok], meas[ok]
    lo, hi = meas.min() / 3, meas.max() * 3
    shown = np.clip(pred, lo, hi)
    ax.vlines(x, np.minimum(shown, meas), np.maximum(shown, meas),
              color="#c9ced6", lw=0.6, zorder=1)
    ax.scatter(x, meas, s=7, color=INK, zorder=3, lw=0)
    inside = (pred >= lo) & (pred <= hi)
    ax.scatter(x[inside], pred[inside], s=7, color=colour, zorder=2, lw=0)
    ax.scatter(x[pred > hi], np.full((pred > hi).sum(), hi), marker="^", s=18,
               color=colour, zorder=2, lw=0)
    ax.scatter(x[pred < lo], np.full((pred < lo).sum(), lo), marker="v", s=18,
               color=colour, zorder=2, lw=0)
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(FuncFormatter(plain))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, p: ""))
    ape = np.abs(pred - meas) / meas
    ax.text(0.99, 0.96, f"MAPE {100 * ape.mean():,.1f}%  median {100 * np.median(ape):,.1f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            color=INK, bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                 ec="#d0d7de"))
    ax.grid(axis="y", color="#e6e9ed", lw=0.6)
    ax.tick_params(labelsize=7.5, colors=INK2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(
        HERE, "..", "figures", "new-workloads", "nn-v2", "v2_pred_vs_meas_per_chunk.png"))
    a = ap.parse_args()
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(3, 4, figsize=(17, 9.5))
    fig.patch.set_facecolor("white")
    for c, (ds, name, label, colour) in enumerate(COLUMNS):
        d = pd.read_csv(os.path.join(DATA, name + ".csv"))
        for r, (key, ylabel) in enumerate(METRICS):
            ax = axes[r, c]
            panel(ax, d, key, colour)
            if r == 0:
                ax.set_title(f"{ds} -- {label}", loc="left", fontsize=11, color=INK)
            if c == 0:
                ax.set_ylabel(ylabel, fontsize=9, color=INK2)
            if r == 2:
                ax.set_xlabel("chunk, in stream order", fontsize=9, color=INK2)
    handles = [Line2D([], [], marker="o", ls="", color=INK, label="measured by Clio"),
               Line2D([], [], marker="o", ls="", color="#6c757d", label="predicted, static v2"),
               Line2D([], [], marker="o", ls="", color="#d1495b", label="predicted, learn + explore"),
               Line2D([], [], marker="^", ls="", color=INK2,
                      label="prediction beyond the panel (drawn at its edge)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=9, labelcolor=INK2, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("NeuroPress v2 inside Clio: predicted vs measured for the setting "
                 "it picked, chunk by chunk", x=0.01, ha="left", fontsize=13,
                 color=INK, y=0.99)
    fig.text(0.01, 0.945, "Each chunk: Clio's measurement (black) and v2's prediction "
             "made before learning from that chunk (colour), joined by a grey line. "
             "Cost model: 1 GB/s. Chunks stored raw have no measurement and are skipped.",
             fontsize=9, color=INK2)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.9, bottom=0.1, hspace=0.3,
                        wspace=0.2)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=150)
    fig.savefig(os.path.splitext(a.out)[0] + ".pdf")
    print("wrote", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
