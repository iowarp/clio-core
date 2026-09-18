#!/usr/bin/env python3
"""Prediction accuracy against HCompress's cost model, as bars.

  plot_accuracy.py --out figures/hcompress-accuracy [--results DIR] [--tag _cap100]

Reads accuracy_long<tag>.csv and correctness<tag>.csv (accuracy_table.py's
output) and draws, per setting, one bar per model:

  (a) compression-time MAPE      (b) decompression-time MAPE
  (c) compression-ratio MAPE     (d) cost-model MAPE
  (e) prediction correctness, top-1 %   (f) median regret of the pick, %

One figure, accuracy_overview<tag>.png. The per-metric single-column panels
this used to emit beside it were dropped: six files saying what six panels
already say is six things to keep in step with the data.

WHY THE TIME PANELS CARRY A MARK. The corpus every static model was trained on
measures a per-CALL time: 256x more bytes moves its label by 1.58x, so it is
dominated by fixed overhead. The campaign measures CUDA-event KERNEL time. The
two are different quantities, so on the five real workloads the time panels
show that mismatch as much as they show model quality, and the panels say so
rather than leaving the reader to assume otherwise. Ratio and PSNR are the same
quantity on both sides, and correctness is scale-free; those four are the
comparable ones.

Panel (d) is the deployed cost model itself --
max(1ms,ct) + max(1ms,dt) + bytes/(min(ratio,100) x 5e6 B/ms) -- the scalar
selection minimises, scored on chunks where all three components were measured.
It is the one MAPE that answers "is this predictor good enough to choose with",
which is why it replaced PSNR here; PSNR can be restored by putting it back in
METRICS in accuracy_table.py.

The four error panels report sMAPE, 100*mean(|p-a|/(|p|+|a|)), which is bounded
to [0,100] by construction. A per-row cap at 100 was tried first and put every
model past 1x error on exactly 100 -- flat, indistinguishable bars for two very
different models. sMAPE keeps them separated (p=2a -> 33, p=10a -> 82,
p=100a -> 98) while still fitting a 0..100 axis, and it is monotone in the
error, so it does not reorder the models. Regret stays on a log axis.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ----------------------------------------------------------------------------
# STYLE CONSTANTS -- everything tweakable lives here
# ----------------------------------------------------------------------------
FIG_W_2COL = 7.16          # IEEE double-column width, inches
PANEL_H = 1.95
FIG_W_1COL = 3.5
FS_AXIS, FS_TICK, FS_LEG, FS_TAG = 9, 7.5, 7, 6.5
FONT_SERIF = ["Times New Roman", "STIXGeneral", "DejaVu Serif"]
GRID = dict(axis="y", linestyle=":", linewidth=0.5, color="0.75")
#: Bars fill 0.8 of a slot, whatever the number of series in the panel -- the
#: PSNR panel draws only the two models that HAVE a quality output.
GROUP_W = 0.8
EDGE = dict(edgecolor="0.25", linewidth=0.4)
XTICK_ROT = 30
PREVIEW_DPI = 300
#: A log axis cannot render 0, and 0% regret is a real and common outcome (the
#: model picked an optimum). True zeros are drawn at the axis bottom and
#: labelled; the bottom itself is DERIVED FROM THE DATA, never fixed -- a fixed
#: floor silently hides any bar below it, which is exactly what a small
#: favourable value is.
ZERO_HEADROOM = 3.0

# Fixed order and colours: legend order, draw order. Never sorted by value.
MODELS = [
    ("NeuroPress NN", "NeuroPress NN", "#1f77b4"),
    ("NeuroPress NN + online learning (as deployed)", "NeuroPress NN + online", "#17becf"),
    ("XGBoost", "XGBoost", "#ff7f0e"),
    ("HCompress CCP (seed only)", "HCompress CCP (seed)", "#8c564b"),
    ("HCompress CCP (+ feedback)", "HCompress CCP (+fb)", "#d62728"),
]
#: The five real workloads. The synthetic corpus is the models' own training
#: set and HCompress's profiler seed, not a test setting -- see accuracy_table.py.
SETTINGS = ["VPIC", "Nyx", "LAMMPS", "WarpX", "AI"]
SETTING_SHORT = {}


#: key, source, y label, log y, footnote under the panel
PANELS = [
    ("comp_time", "mape", "Comp. time sMAPE (%)", True,
     "labels: per-call vs kernel time"),
    ("decomp_time", "mape", "Decomp. time sMAPE (%)", True,
     "labels: per-call vs kernel time"),
    ("ratio", "mape", "Ratio sMAPE (%)", True, ""),
    ("cost", "mape", "Cost-model sMAPE (%)", True,
     "cost = ct + dt + bytes/(ratio x BW)"),
    ("top1_pct", "correct", "Correct pick, top-1 (%)", False,
     "hollow stub = 0 (never optimal)"),
    ("median_regret_pct", "correct", "Median regret of pick (%)", True,
     "hollow stub = 0 (typically optimal)"),
]


def style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": FONT_SERIF,
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    })


def value(long: pd.DataFrame, corr: pd.DataFrame, setting: str, model: str,
          key: str, source: str) -> float:
    if source == "mape":
        # sMAPE when the results carry it: bounded to [0,100] intrinsically, so
        # no ceiling has to be drawn and models past 1x error stay separated.
        r = long[(long.setting == setting) & (long.model == model)
                 & (long.metric == key)]
        if not r.empty and "smape" in r and np.isfinite(r.iloc[0]["smape"]):
            return float(r.iloc[0]["smape"])
    if source == "correct":
        r = corr[(corr.setting == setting) & (corr.model == model)]
        return float(r.iloc[0][key]) if not r.empty else np.nan
    r = long[(long.setting == setting) & (long.model == model)
             & (long.metric == key)]
    return float(r.iloc[0]["mape"]) if not r.empty else np.nan


def draw_panel(ax, long, corr, key, source, ylabel, logy, footnote, mape_cap=0.0):
    series = list(MODELS)
    bar_w = GROUP_W / len(series)
    x = np.arange(len(SETTINGS))
    marks = []   # annotated AFTER the scale is fixed, never in data coordinates
    zeros = []   # true zeros on a log axis: drawn once the bottom is known
    finite = []
    for i, (model, label, colour) in enumerate(series):
        vals = np.array([value(long, corr, s, model, key, source) for s in SETTINGS])
        off = (i - (len(series) - 1) / 2) * bar_w
        ax.bar(x + off, np.nan_to_num(vals, nan=0.0), bar_w, color=colour,
               label=label, zorder=3, **EDGE)
        finite += [v for v in vals if np.isfinite(v) and v > 0]
        for xi, v in enumerate(vals):
            if not np.isfinite(v):
                marks.append((xi + off, "n/a"))
            elif v <= 0:
                zeros.append((xi + off, colour))
    if source == "mape":
        # 100 is sMAPE's own limit, not a clip, so no ceiling line is drawn.
        ax.set_ylim(bottom=0, top=100)
    elif logy:
        ax.set_yscale("log")
        if finite:
            ax.set_ylim(bottom=min(finite) / ZERO_HEADROOM, top=max(finite) * 2.5)
    else:
        ax.set_ylim(bottom=0, top=(max(finite) * 1.18 if finite else 1))
    # A true zero becomes an OUTLINED stub at the axis bottom, so it reads as
    # "measured, and it is zero" rather than as a gap or as a small value.
    lo, hi = ax.get_ylim()
    for xi, colour in zeros:
        h = (lo * 1.7 - lo) if ax.get_yscale() == "log" else (hi - lo) * 0.011
        ax.bar([xi], [h], bar_w, bottom=lo, facecolor="none",
               edgecolor=colour, linewidth=0.7, zorder=4)
    # y in AXES coordinates: a data-coordinate placement on a log axis whose
    # limits are still autoscaling can land arbitrarily far from the plot, and
    # bbox_inches="tight" then grows the canvas to contain it.
    for xi, kind in marks:
        ax.text(xi, 0.02, kind, rotation=90 if kind == "n/a" else 0,
                ha="center", va="bottom", fontsize=FS_TAG, color="0.35",
                transform=ax.get_xaxis_transform())
    ax.set_ylabel(ylabel, fontsize=FS_AXIS)
    ax.set_xticks(x)
    ax.set_xticklabels([SETTING_SHORT.get(s, s) for s in SETTINGS],
                       fontsize=FS_TICK, rotation=XTICK_ROT, ha="right",
                       rotation_mode="anchor")
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.grid(zorder=0, **GRID)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if footnote:
        # Under the axis, not over the bars.
        ax.set_xlabel(footnote, fontsize=FS_TAG, color="0.4", style="italic",
                      labelpad=2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="/projects/bekn/imuradli/np-hcompress/out")
    ap.add_argument("--tag", default="", help="which variant to plot, e.g. _cap100")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "figures", "hcompress-accuracy"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    style()

    long = pd.read_csv(os.path.join(a.results, f"accuracy_long{a.tag}.csv"))
    corr = pd.read_csv(os.path.join(a.results, f"correctness{a.tag}.csv"))
    settings = [s for s in SETTINGS if s in set(long.setting)]
    if not settings:
        print("no settings found in the results", file=sys.stderr)
        return 1
    globals()["SETTINGS"] = settings
    # The ceiling travels in the results, so the plot cannot disagree with the
    # table about what its own bars mean.
    mape_cap = float(long["mape_cap"].iloc[0]) if "mape_cap" in long else 0.0


    # ---- the 2x3 overview --------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(FIG_W_2COL, 2 * PANEL_H + 0.75))
    for ax, (key, source, ylabel, logy, note) in zip(axes.ravel(), PANELS):
        draw_panel(ax, long, corr, key, source, ylabel, logy, note, mape_cap)
    handles = [Patch(facecolor=c, label=l, **EDGE) for _, l, c in MODELS]
    fig.legend(handles=handles, loc="lower center", ncol=len(MODELS),
               fontsize=FS_LEG, frameon=False, bbox_to_anchor=(0.5, 0.005))
    if mape_cap > 0:
        # No pipe characters: in the serif face they render as backslashes.
        fig.text(0.5, -0.025, "error panels: sMAPE, the absolute error over "
                 "(predicted + measured), bounded to [0,100]; raw MAPE in "
                 "accuracy_long.csv",
                 ha="center", va="bottom", fontsize=FS_TAG, color="0.4",
                 style="italic")
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    for name in ("accuracy_overview",):
        p = os.path.join(a.out, f"{name}{a.tag}.png")
        fig.savefig(p, dpi=PREVIEW_DPI, bbox_inches="tight")
        print(f"wrote {p}")
    plt.close(fig)

    return 0


if __name__ == "__main__":
    sys.exit(main())
