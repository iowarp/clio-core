#!/usr/bin/env python3
"""Prediction error along a 2,000-chunk stream through Clio: static v2 vs learn+explore.

    plot_v2_nyx2k.py [--dataset nyx2k|vpic2k] [--out PNG] [--chunks-out PNG]

2,000 chunks spread evenly over a simulation's timesteps, in output order
(run_v2_workloads.sh DATASET static|learnexp; 4-tier cost model, learning and
exploration gated at 15 % cost error, K = 4, learning rate 0.5). For each
chunk, v2's prediction for the setting it picked (made before learning from
that chunk) against what Clio then measured.

Figure 1 (--out): each chunk's absolute percent error as a dot, and the MAPE
over a rolling window of 100 chunks as a line. The ratio panel excludes the
constant-valued chunks (MAD = 0, early timesteps), whose huge ratios are
reported separately.
Figure 2 (--chunks-out): each chunk's predicted and measured value
(compress ms, decompress ms, ratio), static v2 on the left, learn + explore
on the right. Constant-valued chunks' measurements are drawn in light grey.
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
RUNS = "/mnt/nvme0/v2-work/runs"
FIGS = os.path.join(HERE, "..", "figures", "new-workloads", "nn-v2")
ARMS = [("static v2", "static", "#6c757d"),
        ("learn + explore", "learnexp", "#d1495b")]
TITLES = {"nyx2k": ("Nyx", 51), "vpic2k": ("VPIC", 153)}
DS = "nyx2k"  # set from --dataset in main()


def load(mode):
    """Per-chunk table of one run: eval_v2_workloads.py's DS_chunks.csv when
    present (it also carries regret), else the older DS_MODE_chunks.csv."""
    path = os.path.join(RUNS, f"{DS}_chunks.csv")
    if os.path.exists(path):
        d = pd.read_csv(path)
        return d[d["mode"] == mode].reset_index(drop=True)
    return pd.read_csv(os.path.join(RUNS, f"{DS}_{mode}_chunks.csv"))
PANELS = [("ape_ct", "compress time", False), ("ape_dt", "decompress time", False),
          ("ape_r", "compression ratio (non-constant chunks)", True)]
METRICS = [("ct", "compress time (ms)"), ("dt", "decompress time (ms)"),
           ("r", "compression ratio")]
INK, INK2, CONST = "#1f2328", "#57606a", "#b8bec6"


def pct(v, _p=None):
    """Plain percent tick."""
    if not np.isfinite(v) or v <= 0:
        return ""
    return f"{v:,.0f}%" if v >= 1 else f"{v:g}%"


def plain(v, _p=None):
    """A tick value as a plain number."""
    if v <= 0:
        return ""
    if v >= 100:
        return f"{v:,.0f}"
    return f"{v:g}" if v >= 1 else f"{v:.3g}"


def style(ax):
    """Shared axis styling."""
    ax.grid(axis="y", color="#e6e9ed", lw=0.6)
    ax.tick_params(labelsize=8, colors=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def error_panel(ax, col, title, nonconst):
    """Per-chunk APE dots and the rolling-100 MAPE line for both arms."""
    ends = []
    for label, mode, colour in ARMS:
        d = load(mode)
        if nonconst:
            d = d[~d["const"]]
        ok = np.isfinite(d[col]) & (d[col] > 0)
        ax.scatter(d["i"][ok], d[col][ok], s=4, color=colour, alpha=0.22, lw=0)
        y = d[col].rolling(100, min_periods=30).mean()
        ax.plot(d["i"], y, color=colour, lw=2.2, label=label)
        ends.append((y.iloc[-1], d["i"].iloc[-1], colour))
    # End-of-stream labels; the upper one moves up when the two are close.
    ends.sort()
    close = len(ends) == 2 and ends[1][0] / ends[0][0] < 1.25
    for k, (v, x, colour) in enumerate(ends):
        dy = (-5 if k == 0 else 5) if close else 0
        ax.annotate(pct(v), (x, v), xytext=(5, dy), textcoords="offset points",
                    va="center", fontsize=8.5, color=colour, weight="bold")
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.01)
    ax.yaxis.set_major_formatter(FuncFormatter(pct))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, p: ""))
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(f"chunk, in stream order ({TITLES[DS][1]} timesteps)", fontsize=9, color=INK2)
    ax.set_xlim(0, 2150)
    style(ax)


def error_figure(out):
    """Figure 1: per-chunk error and rolling MAPE, one panel per metric."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("white")
    for ax, (col, title, nonconst) in zip(axes, PANELS):
        error_panel(ax, col, title, nonconst)
    axes[0].set_ylabel("prediction error per chunk (dots), MAPE over the last "
                       "100 chunks (line)", fontsize=8.5, color=INK2)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower left")
    fig.suptitle(f"NeuroPress v2 on 2,000 {TITLES[DS][0]} chunks inside Clio: prediction error "
                 "as the stream goes on", x=0.01, ha="left", fontsize=12.5,
                 color=INK, y=0.99)
    fig.text(0.01, 0.9, "Dots: |predicted - measured| / measured for each chunk "
             "(the setting v2 picked, predicted before learning from that chunk). "
             "Lines: rolling 100-chunk MAPE. Learn + explore: 15 % gates, K = 4, "
             "learning rate 0.5, 4-tier cost model." + (
                 "\nThe row of grey dots at 100 % in the time panels: static v2 "
                 "predicts ~0 ms for constant-valued chunks." if DS == "nyx2k" else ""),
             fontsize=9, color=INK2)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.8, bottom=0.13, wspace=0.22)
    save(fig, out)


def value_panel(ax, d, key, colour):
    """Per-chunk measured vs predicted for one metric of one arm, log y."""
    pred = d[f"{key}_pred"].to_numpy(float)
    meas = d[f"{key}_meas"].to_numpy(float)
    const = d["const"].to_numpy(bool)
    ok = np.isfinite(meas) & (meas > 0)
    x, pred, meas, const = d["i"].to_numpy()[ok], pred[ok], meas[ok], const[ok]
    lo, hi = meas.min() / 3, meas.max() * 3
    shown = np.clip(pred, lo, hi)
    # Joining lines for non-constant chunks only; constant chunks' far-off
    # predictions are still shown as edge triangles.
    nc = ~const
    ax.vlines(x[nc], np.minimum(shown, meas)[nc], np.maximum(shown, meas)[nc],
              color="#dde1e6", lw=0.4, zorder=1)
    ax.scatter(x[~const], meas[~const], s=4, color=INK, zorder=3, lw=0)
    ax.scatter(x[const], meas[const], s=4, color=CONST, zorder=3, lw=0)
    inside = (pred >= lo) & (pred <= hi)
    ax.scatter(x[inside], pred[inside], s=4, color=colour, zorder=2, lw=0)
    for mask, edge, mk in ((pred > hi, hi, "^"), (pred < lo, lo, "v")):
        ax.scatter(x[mask], np.full(mask.sum(), edge), marker=mk, s=12,
                   color=colour, zorder=2, lw=0)
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(FuncFormatter(plain))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, p: ""))
    ape = np.abs(pred - meas) / meas
    ax.text(0.99, 0.97, f"MAPE {100 * ape.mean():,.1f}%   non-constant chunks "
            f"{100 * ape[nc].mean():,.1f}% (median {100 * np.median(ape[nc]):,.1f}%)",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            color=INK, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#d0d7de"))
    ax.set_xlim(-10, 2010)
    style(ax)


def value_figure(out):
    """Figure 2: per-chunk predicted vs measured, metrics x arms."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 10.5), sharex=True)
    fig.patch.set_facecolor("white")
    for c, (label, mode, colour) in enumerate(ARMS):
        d = load(mode)
        for r, (key, ylabel) in enumerate(METRICS):
            ax = axes[r, c]
            value_panel(ax, d, key, colour)
            if r == 0:
                ax.set_title(f"{TITLES[DS][0]} -- {label}", loc="left", fontsize=11.5, color=INK)
            if c == 0:
                ax.set_ylabel(ylabel, fontsize=9.5, color=INK2)
            if r == 2:
                ax.set_xlabel(f"chunk, in stream order ({TITLES[DS][1]} timesteps)", fontsize=9,
                              color=INK2)
    handles = [Line2D([], [], marker="o", ls="", color=INK, label="measured by Clio"),
               Line2D([], [], marker="o", ls="", color=CONST,
                      label="measured, constant-valued chunk"),
               Line2D([], [], marker="o", ls="", color=ARMS[0][2],
                      label="predicted, static v2"),
               Line2D([], [], marker="o", ls="", color=ARMS[1][2],
                      label="predicted, learn + explore"),
               Line2D([], [], marker="^", ls="", color=INK2,
                      label="prediction beyond the panel (drawn at its edge)")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=9, labelcolor=INK2, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"NeuroPress v2 on 2,000 {TITLES[DS][0]} chunks inside Clio: predicted vs "
                 "measured, chunk by chunk", x=0.01, ha="left", fontsize=13,
                 color=INK, y=0.995)
    fig.text(0.01, 0.955, "Each chunk: Clio's measurement of the setting v2 picked "
             "(black) and v2's prediction for it, made before learning from that "
             "chunk (colour), joined by a grey line (non-constant chunks).", fontsize=9,
             color=INK2)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.92, bottom=0.08, hspace=0.18,
                        wspace=0.14)
    save(fig, out)


LONG = {"static": "NeuroPress v2, shipped model, no learning",
        "learnexp": "NeuroPress v2, learning + exploration"}


def single_name():
    """Name of the best single setting, from eval_v2_workloads.py's summary."""
    path = os.path.join(RUNS, f"{DS}_summary.csv")
    if not os.path.exists(path):
        return "best single setting"
    return pd.read_csv(path).best_single.iloc[0]


def regret_figure(out):
    """Figure 3: learn + explore's per-chunk regret and its running cost
    against the per-chunk best and the best single setting, from the
    exhaustive run's measurements of all 45 settings."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 5))
    fig.patch.set_facecolor("white")
    single, ends, n = None, [], 0
    # Learning v2 against the best single setting only (the shipped model
    # without learning is left out of this comparison).
    for label, mode, colour in [arm for arm in ARMS if arm[1] == "learnexp"]:
        d = load(mode)
        x = d["i"].to_numpy()
        n = max(n, len(d))
        a1.scatter(x, d.regret_stored, s=4, color=colour, alpha=0.25, lw=0)
        a1.plot(x, d.regret_stored.rolling(100, min_periods=30).mean(),
                color=colour, lw=2.2, label=f"{LONG[mode]}: average {d.regret_stored.mean():.1f}%; "
                f"stored the cheapest setting on {100 * (d.regret_stored <= 1e-9).mean():.0f}% "
                "of chunks")
        run = 100 * (d.cost_stored.cumsum() / d.cost_best.cumsum() - 1)
        a2.plot(x, run, color=colour, lw=2.2, label=LONG[mode])
        ends.append((run.iloc[-1], x[-1], colour, run.iloc[50:].max()))
        single = d
    run = 100 * (single.cost_single.cumsum() / single.cost_best.cumsum() - 1)
    a2.plot(single["i"], run, color=INK, lw=1.6, ls="--",
            label=f"one codec for the whole stream: {single_name()}\n"
                  "(the best one, picked in hindsight)")
    ends.append((run.iloc[-1], single["i"].iloc[-1], INK, run.iloc[50:].max()))
    # The first chunks' running totals swing wildly; scale to the rest, and
    # keep the end labels at least 4 % of the axis apart.
    top = 1.3 * max(e[3] for e in ends)
    a2.set_ylim(0, top)
    last = -np.inf
    for v, x, colour, _ in sorted(ends):
        y = max(v, last + 0.04 * top)
        last = y
        a2.annotate(f"{v:.1f}%", (x, v), xytext=(x + 0.01 * n, y),
                    textcoords="data", va="center", fontsize=8.5, color=colour,
                    weight="bold")
    a1.set_yscale("symlog", linthresh=1.0)
    a1.set_ylim(0, None)
    a1.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:,.0f}%"))
    a1.set_title("Each chunk: how much more the stored setting cost than that "
                 "chunk's cheapest\n(dots = chunks, line = average of the last 100; "
                 "0% = the cheapest was stored)", loc="left", fontsize=10.5, color=INK)
    a1.set_ylabel("extra cost vs the chunk's cheapest setting", fontsize=9,
                  color=INK2)
    a2.set_title("Running total: extra cost so far vs the perfect choice on every "
                 "chunk\n(where each strategy stands after N chunks; lower = cheaper)",
                 loc="left", fontsize=10.5, color=INK)
    a2.set_ylabel("extra total cost vs perfect per-chunk choice", fontsize=9,
                  color=INK2)
    a2.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:,.0f}%"))
    for ax in (a1, a2):
        ax.set_xlabel(f"chunk, in stream order ({TITLES[DS][1]} timesteps)",
                      fontsize=9, color=INK2)
        ax.set_xlim(0, 1.08 * n)
        ax.legend(frameon=False, fontsize=8.3, labelcolor=INK, loc="upper right")
        style(ax)
    fig.suptitle(f"Does NeuroPress v2 store the cheapest setting? 2,000 "
                 f"{TITLES[DS][0]} chunks stored through Clio", x=0.01, ha="left",
                 fontsize=13.5, color=INK, y=0.995)
    fig.text(0.01, 0.925, "How to read: every chunk was also compressed with all 45 "
             "lossless settings, so its cheapest setting is known; storing that one on "
             "every chunk is perfect (0%). Lower is better.\nCost of a chunk = compress "
             "time + decompress time + time to write it to its storage tier (12 / 1 / "
             "0.5 / 0.25 GB/s); a setting that does not shrink the chunk is stored "
             "uncompressed.", fontsize=9, color=INK2, va="top", linespacing=1.45)
    fig.subplots_adjust(left=0.06, right=0.95, top=0.75, bottom=0.12, wspace=0.2)
    save(fig, out)


PDF = False  # also write a PDF next to each PNG (--pdf)


def save(fig, out):
    """Write the PNG, and the PDF when --pdf was given."""
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=160)
    if PDF:
        fig.savefig(os.path.splitext(out)[0] + ".pdf")
    print("wrote", os.path.abspath(out))


def main():
    global DS, PDF
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default="nyx2k", choices=sorted(TITLES))
    ap.add_argument("--out", default=None)
    ap.add_argument("--pdf", action="store_true",
                    help="also write PDFs next to the PNGs")
    ap.add_argument("--chunks-out", default=None)
    a = ap.parse_args()
    DS = a.dataset
    PDF = a.pdf
    a.out = a.out or os.path.join(FIGS, f"v2_{DS}_mape_stream.png")
    a.chunks_out = a.chunks_out or os.path.join(
        FIGS, f"v2_{DS}_pred_vs_meas_per_chunk.png")
    plt.rcParams["font.family"] = "DejaVu Sans"
    error_figure(a.out)
    value_figure(a.chunks_out)
    if "regret_stored" in load("static"):
        regret_figure(os.path.join(os.path.dirname(a.out), f"v2_{DS}_regret_stream.png"))


if __name__ == "__main__":
    main()
