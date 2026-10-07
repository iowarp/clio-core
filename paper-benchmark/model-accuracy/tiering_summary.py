#!/usr/bin/env python3
"""Tiering summary of all workloads: NeuroPress against the best static codec of the same storage
setup, in the measured run and in the tiering model's single-tier, tiered and tiered + async
scenarios, encoded as in the comparison summary (plot_workload_summary.py --final): NeuroPress's
bar on the time oracle's light track, the cost oracle as a marker.

    tiering_summary.py [--fig-root DIR] [--summary CSV] [--out PNG]

Reads FIG_ROOT/<workload>/tiering_<workload>.csv (tiering_final.py) and the measured summary
(--summary, default FIG_ROOT/final_summary.csv of final_check.py). Writes PNG (default
FIG_ROOT/tiering_summary.png) and the numbers to the PNG's name with .csv. The title, subtitle and
row labels state the cost model of each workload, the tier speeds, the composition and the setup.
"""
import argparse
import os
import textwrap

import matplotlib
import pandas as pd

import final_config as fc
import tiering_final as tf
from plot_workload_summary import COST_ORACLE, GRID, INK, INK_2, NP_COLOR, TRACK_COLOR, TRACK_INK

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SCENARIO = {"measured run (1 tier, real disk)": "measured, 1 tier", "model: single tier (burst buffer)": "single tier",
            "model: tiered": "tiered", "model: tiered + async": "tiered + async"}


def load(fig_root, summary):
    """@return one row per workload and scenario: NeuroPress's, the cost oracle's and the time
    oracle's time saved and ratio gain against the best static of the same scenario."""
    out = []
    for wl, (ds, w, bw) in fc.WORKLOADS.items():
        f = os.path.join(fig_root, wl.lower(), f"tiering_{wl.lower()}.csv")
        if not os.path.exists(f):
            continue
        meas, runs = tf.measured(summary, wl)
        for label, t, r, co, to in tf.gain_rows(pd.read_csv(f).set_index("config"), meas):
            out.append({"workload": wl, "cost_model": fc.model_name(w), "cost_bw_GBs": bw / 1e6,
                        "scenario": SCENARIO[label], "measured_runs": runs if label.startswith("measured") else 0,
                        "np_time_saved_pct": t, "np_ratio_gain_pct": r,
                        "cost_oracle_time_saved_pct": co[0] if co else float("nan"),
                        "cost_oracle_ratio_gain_pct": co[1] if co else float("nan"),
                        "time_oracle_time_saved_pct": to[0] if to else float("nan"),
                        "time_oracle_ratio_gain_pct": to[1] if to else float("nan")})
    return pd.DataFrame(out)


def panel(ax, t, ys, what, title):
    """One panel: per workload and scenario, NeuroPress's bar on the time oracle's track and the
    cost oracle's marker. @param what 'time_saved' or 'ratio_gain'"""
    np_v, co_v, to_v = (t[f"{k}_{what}_pct"] for k in ("np", "cost_oracle", "time_oracle"))
    hi = max(np_v.max(), co_v.max(), to_v.max())
    lo = min(0, np_v.min(), co_v.min())
    span = hi - lo + 1e-9
    pad = 0.015 * span
    for y, v, co, to in zip(ys, np_v, co_v, to_v):
        if to == to:
            ax.barh(y, to, 0.7, color=TRACK_COLOR, zorder=1)
        ax.barh(y, v, 0.48, color=NP_COLOR, zorder=2)
        if co == co:
            ax.plot(co, y + 0.42, "v", ms=6, color=COST_ORACLE, mec="white", mew=0.7, zorder=5)
        inside = abs(v) >= 0.14 * span
        ax.text(v - pad if inside else max(v, 0) + pad, y, f"{v:+.0f}%", va="center", ha="right" if inside else "left",
                fontsize=9, fontweight="bold", color="white" if inside else INK, zorder=3)
        if to == to:
            end = max(to, v if inside else max(v, 0) + 0.11 * span)
            ax.text(end + pad, y, f"{to:+.0f}%", va="center", ha="left", fontsize=8, color=TRACK_INK, zorder=3)
    ax.axvline(0, color=INK, lw=1.1)
    ax.set_xlim(lo - (0.04 * span if lo < 0 else 0), hi + 0.16 * span)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK, pad=10)
    ax.set_xlabel("% vs the best static codec of the same storage setup (positive = better)", color=INK_2, fontsize=9)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sd in ("top", "right", "left"):
        ax.spines[sd].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=INK_2, labelsize=8.5)


def plot(t, out):
    """The two panels (time saved, ratio gain), one block of scenario rows per workload."""
    block, pos, y = {}, [], 0.0   # workload: (lowest, highest row position), bottom block first
    for wl in reversed(list(dict.fromkeys(t.workload))):
        n = int((t.workload == wl).sum())
        block[wl] = (y, y + n - 1)
        y += n + 0.9
    for wl in dict.fromkeys(t.workload):   # the table's order, top row first in each block
        pos += [block[wl][1] - k for k in range(int((t.workload == wl).sum()))]
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 8.6), sharey=True)
    panel(ax[0], t, pos, "time_saved", "End-to-end time saved")
    panel(ax[1], t, pos, "ratio_gain", "Compression ratio gain")
    ax[0].set_yticks(pos, t.scenario, fontsize=9, color=INK_2)
    for wl, (a, b) in block.items():
        w = t[t.workload == wl].iloc[0]
        ax[0].text(-0.36, (a + b) / 2, f"{wl}\ncost model\n{w.cost_model}\n(measured at\n{w.cost_bw_GBs:g} GB/s)",
                   transform=ax[0].get_yaxis_transform(),
                   ha="center", va="center", fontsize=10.5, fontweight="bold", color=INK)
        if b < max(pos):   # a line between two workloads
            for a_ in ax:
                a_.axhline(b + 0.95, color=GRID, lw=1)
    ax[0].set_ylim(min(pos) - 0.6, max(pos) + 0.8)
    tb = fc.TIER_BW
    fig.text(0.012, 0.985, "NeuroPress with tiering: it saves end-to-end time over the best static codec in every storage setup",
             fontsize=15, fontweight="bold", color=INK, va="top")
    sub = (f"Measured: Clio on the test disk (1 tier), 1 process, 1 chunk in flight, page cache dropped before each read, "
           + ("1 run per option" if int(t.measured_runs.max()) == 1 else f"mean of {int(t.measured_runs.max())} runs per option")
           + ", cost bandwidth per workload as labelled.",
           f"Model (offline, from the exhaustive search): tier speeds DRAM {tb['DRAM'] / 1e6:g}, NVMe {tb['NVMe'] / 1e6:g}, "
           f"burst buffer {tb['burst buffer'] / 1e6:g}, PFS {tb['PFS'] / 1e6:g} GB/s; single tier = every chunk on the burst "
           f"buffer; tiered = write-through to the burst buffer (100 %) with copies of {fc.TIERED_COPIES['DRAM']} % of the "
           f"chunks on DRAM and {fc.TIERED_COPIES['NVMe']} % on NVMe, round-robin; async = perfect overlap of codec work and "
           f"transfers. Every scenario: 1 write + {fc.READS} reads; every option selects with its storage speed.")
    fig.text(0.012, 0.935, "\n".join(textwrap.fill(x, 165) for x in sub), fontsize=9.5, color=INK_2, va="top")
    fig.legend(handles=[matplotlib.patches.Patch(color=NP_COLOR, label="NeuroPress (learning)"),
                        matplotlib.patches.Patch(color=TRACK_COLOR, label="time oracle (best per chunk by end-to-end time)"),
                        matplotlib.lines.Line2D([], [], ls="", marker="v", ms=7, color=COST_ORACLE,
                                                label="cost oracle (best per chunk by the cost model)")],
               loc="upper left", bbox_to_anchor=(0.012, 0.845), ncol=3, frameon=False, fontsize=10)
    fig.text(0.012, 0.01, textwrap.fill(
        "Cost model: cost = w_ct × compress ms + w_dt × decompress ms + w_io × stored bytes / bandwidth, weights "
        "w_ct/w_dt/w_io per workload as labelled. Best static (the 0 line) = the one setting with the lowest total cost in "
        "that storage setup; NeuroPress = online learning, no exploration (model rows: replay of Clio's selection and "
        "learning, paying its measured selection and training time); cost oracle = each chunk's lowest-cost setting; time "
        "oracle = each chunk's lowest end-to-end time setting. Lossless; every measured chunk verified bit-exact.", 230),
        fontsize=8.5, color=INK_2, va="bottom")
    fig.tight_layout(rect=(0.07, 0.07, 1, 0.81))
    fig.savefig(out, dpi=160, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fig-root", default=os.path.join(HERE, "..", "figures", "new-workloads", "sim-tuning", "final"))
    ap.add_argument("--summary", default=None, help="measured summary CSV (default FIG_ROOT/final_summary.csv)")
    ap.add_argument("--out", default=None, help="PNG (default FIG_ROOT/tiering_summary.png)")
    a = ap.parse_args()
    out = a.out or os.path.join(a.fig_root, "tiering_summary.png")
    t = load(a.fig_root, a.summary or os.path.join(a.fig_root, "final_summary.csv"))
    if t.empty:
        raise SystemExit("no tiering CSV found (run tiering_final.py first)")
    t.to_csv(os.path.splitext(out)[0] + ".csv", index=False)
    plot(t, out)
    print("wrote", os.path.abspath(out))
    with pd.option_context("display.width", 200):
        print(t.drop(columns=["cost_bw_GBs", "measured_runs"]).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
