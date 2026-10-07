#!/usr/bin/env python3
"""Tiering summary of all workloads: NeuroPress against the best static codec of the same storage
setup, in the measured run and in the tiering model's single-tier, tiered and tiered + async
scenarios, encoded as in the comparison summary (plot_workload_summary.py --final): NeuroPress's
bar on the time oracle's light track, the cost oracle as a marker.

    tiering_summary.py [--fig-root DIR] [--summary CSV] [--out PNG]

Reads FIG_ROOT/<workload>/tiering_<workload>.csv (tiering_final.py) and the measured summary
(--summary, default FIG_ROOT/final_summary.csv of final_check.py). Writes PNG (default
FIG_ROOT/tiering_summary.png) and the numbers to the PNG's name with .csv. The figure explains
itself: a key of the four storage setups (where the chunks are written and read, the tier speeds,
what the time includes), a worked example of one row with every mark explained, the cost model of
each workload beside its rows and the definitions in the footnote.
"""
import argparse
import os
import textwrap

import matplotlib
import pandas as pd

import final_config as fc
import tiering_final as tf
from plot_workload_summary import GRID, HC_COLOR, INK, INK_2, NP_COLOR, TRACK_COLOR, TRACK_INK, XGB_COLOR

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SCENARIO = {"measured run (1 tier, real disk)": "measured, 1 tier", "model: single tier (burst buffer)": "single tier",
            "model: tiered": "tiered", "model: tiered + async": "tiered + async"}


def load(fig_root, summary):
    """@return one row per workload and scenario: NeuroPress's, the cost oracle's, the time
    oracle's and HCompress's time saved and ratio gain against the best static of the same scenario."""
    out = []
    for wl, (ds, w, bw) in fc.WORKLOADS.items():
        f = os.path.join(fig_root, wl.lower(), f"tiering_{wl.lower()}.csv")
        if not os.path.exists(f):
            continue
        meas, runs = tf.measured(summary, wl)
        for label, t, r, co, to, hc, xg in tf.gain_rows(pd.read_csv(f).set_index("config"), meas):
            out.append({"workload": wl, "cost_model": fc.model_name(w), "cost_bw_GBs": bw / 1e6,
                        "scenario": SCENARIO[label], "measured_runs": runs if label.startswith("measured") else 0,
                        "np_time_saved_pct": t, "np_ratio_gain_pct": r,
                        "cost_oracle_time_saved_pct": co[0] if co else float("nan"),
                        "cost_oracle_ratio_gain_pct": co[1] if co else float("nan"),
                        "time_oracle_time_saved_pct": to[0] if to else float("nan"),
                        "time_oracle_ratio_gain_pct": to[1] if to else float("nan"),
                        "hcompress_time_saved_pct": hc[0] if hc else float("nan"),
                        "hcompress_ratio_gain_pct": hc[1] if hc else float("nan"),
                        "xgboost_time_saved_pct": xg[0] if xg else float("nan"),
                        "xgboost_ratio_gain_pct": xg[1] if xg else float("nan")})
    return pd.DataFrame(out)


def panel(ax, t, ys, what, title):
    """One panel: per workload and scenario, NeuroPress's bar on the time oracle's track, and
    HCompress's and XGBoost's thin bars under them (the cost oracle is not drawn).
    @param what 'time_saved' or 'ratio_gain'"""
    np_v, to_v, hc_v, xg_v = (t[f"{k}_{what}_pct"] for k in ("np", "time_oracle", "hcompress", "xgboost"))
    hi = max(np_v.max(), to_v.max(), hc_v.max(), xg_v.max())
    lo = min(0, np_v.min(), hc_v.min(), xg_v.min())
    span = hi - lo + 1e-9
    pad = 0.015 * span
    for y0, v, to, hc, xg in zip(ys, np_v, to_v, hc_v, xg_v):
        bases = [(b, c) for b, c in ((hc, HC_COLOR), (xg, XGB_COLOR)) if b == b]
        y = y0 + (0.2 if len(bases) == 2 else 0.1 if bases else 0.0)
        if to == to:
            ax.barh(y, to, 0.44, color=TRACK_COLOR, zorder=1)
        ax.barh(y, v, 0.3, color=NP_COLOR, zorder=2)
        for (b, c), yb in zip(bases, (y0 - 0.12, y0 - 0.36) if len(bases) == 2 else (y0 - 0.32,)):
            ax.barh(yb, b, 0.16, color=c, zorder=2)
            ax.text(b + (pad if b >= 0 else -pad), yb, f"{b:+.0f}%", va="center",
                    ha="left" if b >= 0 else "right", fontsize=7, color=INK_2, zorder=3)
        inside = abs(v) >= 0.14 * span
        ax.text(v - pad if inside else max(v, 0) + pad, y, f"{v:+.0f}%", va="center", ha="right" if inside else "left",
                fontsize=9, fontweight="bold", color="white" if inside else INK, zorder=3)
        if to == to:
            end = max(to, v if inside else max(v, 0) + 0.11 * span)
            ax.text(end + pad, y, f"{to:+.0f}%", va="center", ha="left", fontsize=8, color=TRACK_INK, zorder=3)
    ax.axvline(0, color=INK, lw=1.1)
    ax.set_xlim(lo - (0.08 * span if lo < 0 else 0), hi + 0.16 * span)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK, pad=10)
    ax.set_xlabel("% vs the best static codec of the same storage setup (positive = better)", color=INK_2, fontsize=9)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sd in ("top", "right", "left"):
        ax.spines[sd].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=INK_2, labelsize=8.5)


TIER_COLOR = {"DRAM": "#1f6f54", "NVMe": "#5fae8f", "burst buffer": "#c3e2d4", "test disk": "#bdbcb6"}


def setups():
    """@return the four storage setups of the rows, as (row name, kind, write tiers, read tiers,
    description); tiers as [(tier, % of the chunks)]."""
    bb = [("burst buffer", 100)]
    rd = [("DRAM", fc.TIERED_COPIES["DRAM"]), ("NVMe", fc.TIERED_COPIES["NVMe"]),
          ("burst buffer", 100 - sum(fc.TIERED_COPIES.values()))]
    return [("measured, 1 tier", "real Clio run", [("test disk", 100)], [("test disk", 100)],
             "Clio runs the workload on the test NVMe disk at its real speed, with a k-means iteration "
             "after each read and the runtime's own work. Selects with the cost bandwidth under the "
             "workload's name. Codec work and I/O in series."),
            ("single tier", "offline model", bb, bb,
             "Every chunk is written to and read from the burst buffer; selects with its speed. "
             "Codec work and I/O in series. No k-means, no runtime work."),
            ("tiered", "offline model", bb, rd,
             "Every chunk is written to the burst buffer. DRAM and NVMe copies (dealt round-robin) "
             "serve the reads from the fastest copy; selects with each chunk's own speed. PFS flush "
             "in the background. Codec work and I/O in series."),
            ("tiered + async", "offline model", bb, rd,
             "As tiered, but codec work and I/O overlap perfectly (the best case): each write or "
             "read takes max(codec time, I/O time).")]


def tier_bar(ax, x0, x1, y, h, tiers):
    """Draw one stacked bar of tiers [(tier, %)] from x0 to x1 at height y."""
    x = x0
    for tier, pct in tiers:
        w = (x1 - x0) * pct / 100
        ax.add_patch(matplotlib.patches.Rectangle((x, y - h / 2), w, h, color=TIER_COLOR[tier], lw=0))
        dark = tier in ("DRAM",)
        ax.text(x + w / 2, y, f"{tier} {pct}%" if pct >= 50 else f"{pct}%", ha="center", va="center",
                fontsize=7.5 if pct >= 20 else 7, color="white" if dark else INK)
        x += w


def setup_key(fig, rect):
    """The key of the four storage setups (one box per row name): where the chunks are written,
    where they are read from, and what the time includes."""
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1.18)
    ax.axis("off")
    tb = fc.TIER_BW
    speed = {"DRAM": tb["DRAM"], "NVMe": tb["NVMe"], "burst buffer": tb["burst buffer"]}
    items = [f"{k} {v / 1e6:g} GB/s" for k, v in speed.items()] + ["test disk (real NVMe)"]
    ax.text(0.03, 1.13, "The four storage setups (the four rows of every workload)", fontsize=11.5,
            fontweight="bold", color=INK, va="center")
    for k, (name, label) in enumerate(zip(list(speed) + ["test disk"], items)):
        x = 2.0 + 0.5 * k
        ax.add_patch(matplotlib.patches.Rectangle((x, 1.105), 0.06, 0.05, color=TIER_COLOR[name], lw=0))
        ax.text(x + 0.08, 1.13, label, fontsize=8.5, color=INK_2, va="center")
    for i, (name, kind, wr, rd, text) in enumerate(setups()):
        ax.add_patch(matplotlib.patches.FancyBboxPatch((i + 0.03, 0.02), 0.94, 0.98, boxstyle="round,pad=0,rounding_size=0.03",
                                                       fc="#faf9f6", ec=GRID, lw=1))
        ax.text(i + 0.07, 0.91, name, fontsize=10.5, fontweight="bold", color=INK, va="center")
        ax.text(i + 0.93, 0.91, kind, fontsize=8.5, color=INK_2, va="center", ha="right", style="italic")
        for y, what, tiers in ((0.76, "write", wr), (0.62, f"{fc.READS} reads", rd)):
            ax.text(i + 0.07, y, what, fontsize=8.5, color=INK_2, va="center")
            tier_bar(ax, i + 0.3, i + 0.93, y, 0.11, tiers)
        ax.text(i + 0.07, 0.5, textwrap.fill(text, 52), fontsize=8.2, color=INK_2, va="top", linespacing=1.3)


def how_to_read(fig, rect, row):
    """A worked example of one row (row: a line of the table): each mark numbered on the bar and
    explained in a list beside it."""
    x0, y0, w, h = rect
    ax = fig.add_axes((x0, y0, 0.24, h))
    v, to = row.np_time_saved_pct, row.time_oracle_time_saved_pct
    hc, xg = row.hcompress_time_saved_pct, row.xgboost_time_saved_pct
    ax.barh(0.1, to, 0.6, color=TRACK_COLOR)
    ax.barh(0.1, v, 0.4, color=NP_COLOR)
    for b, c, yb in ((hc, HC_COLOR, -0.45), (xg, XGB_COLOR, -0.8)):
        ax.barh(yb, b, 0.24, color=c)
        ax.text(b + (0.5 if b >= 0 else -0.5), yb, f"{b:+.0f}%", ha="left" if b >= 0 else "right", va="center",
                fontsize=8, color=INK_2)
    ax.axvline(0, color=INK, lw=1.1)
    ax.set_xlim(min(0, hc, xg) * 1.6 - 1, to * 1.15)
    ax.set_ylim(-1.0, 0.6)
    ax.axis("off")
    ax.text(v - 0.4, 0.1, f"{v:+.0f}%", ha="right", va="center", fontsize=9, fontweight="bold", color="white")
    ax.text(to + 0.5, 0.1, f"{to:+.0f}%", ha="left", va="center", fontsize=8.5, color=TRACK_INK)
    ax.text(ax.get_xlim()[0], 2.7, f"How to read a row (example: {row.workload}, {row.scenario}; the ratio panel reads the same way)", fontsize=11.5,
            fontweight="bold", color=INK, va="center")
    num = dict(ha="center", va="center", fontsize=8, fontweight="bold", color="white",
               bbox=dict(boxstyle="circle,pad=0.25", fc=INK_2, ec="none"))
    for k, (x, y) in enumerate(((0, 1.05), (v / 2, 1.05), (to, 1.05), (hc / 2, -1.45), (xg / 2, -1.45)), 1):
        ax.text(x, y, str(k), **num)
    items = ("0 line = the best static codec of the same setup: one setting for all chunks, the lowest total cost.",
             f"Dark bar = NeuroPress (online learning): {v:+.0f}% means {v:.0f}% less end-to-end time than the best "
             "static codec.",
             "Light track = time oracle, the upper limit: each chunk's fastest setting, known only after the run.",
             f"Thin grey bar = HCompress (library + size model, same settings and cost model): {hc:+.0f}%.",
             f"Thin green bar = XGBoost (trees on NeuroPress's data and features, no online learning, selection "
             f"time included): {xg:+.0f}%.")
    lx = fig.add_axes((x0 + 0.29, y0 - 0.047, w - 0.29, h + 0.06))
    lx.axis("off")
    for k, text in enumerate(items):
        cx, cy = (k // 3) * 0.5, 0.88 - (k % 3) * 0.38
        lx.text(cx, cy, str(k + 1), transform=lx.transAxes, **num)
        lx.text(cx + 0.015, cy, textwrap.fill(text, 70), transform=lx.transAxes, ha="left", va="center",
                fontsize=8.8, color=INK, linespacing=1.25)


def plot(t, out):
    """The tiering summary: the storage-setup key, a worked example, then the two panels (time
    saved, ratio gain) with one block of setup rows per workload."""
    block, pos, y = {}, [], 0.0   # workload: (lowest, highest row position), bottom block first
    for wl in reversed(list(dict.fromkeys(t.workload))):
        n = int((t.workload == wl).sum())
        block[wl] = (y, y + n - 1)
        y += n + 0.9
    for wl in dict.fromkeys(t.workload):   # the table's order, top row first in each block
        pos += [block[wl][1] - k for k in range(int((t.workload == wl).sum()))]
    fig = plt.figure(figsize=(15, 15.8))
    setup_key(fig, (0.02, 0.775, 0.96, 0.15))
    ex = t[t.scenario == "single tier"].iloc[0]
    how_to_read(fig, (0.03, 0.695, 0.95, 0.022), ex)
    ax = [fig.add_axes((0.2, 0.115, 0.37, 0.5)), fig.add_axes((0.61, 0.115, 0.37, 0.5))]
    panel(ax[0], t, pos, "time_saved", f"End-to-end time saved (1 write + {fc.READS} reads)")
    panel(ax[1], t, pos, "ratio_gain", "Compression ratio gain")
    ax[0].set_xlabel("% less end-to-end time than the best static codec of the same setup", color=INK_2, fontsize=9)
    ax[1].set_xlabel("% higher compression ratio than the best static codec of the same setup", color=INK_2, fontsize=9)
    ax[0].set_yticks(pos, t.scenario, fontsize=9.5, color=INK)
    ax[1].set_yticks(pos, [])
    for a_ in ax:
        a_.set_ylim(min(pos) - 0.6, max(pos) + 0.8)
    for wl, (a, b) in block.items():
        w = t[t.workload == wl].iloc[0]
        ax[0].text(-0.3, (a + b) / 2, f"{wl}", transform=ax[0].get_yaxis_transform(), ha="center", va="bottom",
                   fontsize=13, fontweight="bold", color=INK)
        ax[0].text(-0.3, (a + b) / 2 - 0.1, f"cost model {w.cost_model}\ncost bandwidth of the\nreal run: {w.cost_bw_GBs:g} GB/s",
                   transform=ax[0].get_yaxis_transform(), ha="center", va="top", fontsize=9, color=INK_2)
        if b < max(pos):   # a line between two workloads
            for a_ in ax:
                a_.axhline(b + 0.95, color=GRID, lw=1)
    lo, hi = t.np_time_saved_pct.min(), t.np_time_saved_pct.max()
    hlo, hhi = t.hcompress_time_saved_pct.min(), t.hcompress_time_saved_pct.max()
    xlo, xhi = t.xgboost_time_saved_pct.min(), t.xgboost_time_saved_pct.max()
    beats = bool((t.np_time_saved_pct > t.hcompress_time_saved_pct).all()
                 and (t.np_time_saved_pct > t.xgboost_time_saved_pct).all())
    fig.text(0.02, 0.985, "NeuroPress with tiering: faster than the best static codec"
             + (", HCompress and XGBoost" if beats else "") + " in every storage setup",
             fontsize=16, fontweight="bold", color=INK, va="top")
    fig.text(0.02, 0.957, textwrap.fill(
        f"In all {len(t)} rows (4 workloads × 4 storage setups) NeuroPress saves {lo:.0f} to {hi:.0f} % of the end-to-end "
        f"time of the best static codec of the same setup; HCompress {hlo:+.0f} to {hhi:+.0f} %, XGBoost {xlo:+.0f} to "
        f"{xhi:+.0f} %. The time oracle shows the "
        f"upper limit. Lossless; "
        f"every chunk of the real runs verified bit-exact; "
        + ("1 run per option." if int(t.measured_runs.max()) == 1 else f"mean of {int(t.measured_runs.max())} runs per option."), 175),
        fontsize=10, color=INK_2, va="top")
    fig.text(0.02, 0.005, textwrap.fill(
        "Cost model: cost = w_ct × compress ms + w_dt × decompress ms + w_io × stored bytes / bandwidth, with the "
        "weights w_ct/w_dt/w_io of each workload. The cost bandwidth is the storage speed that the real run's cost model "
        "uses; the model rows use each tier's speed (a chunk with a DRAM or NVMe copy is selected with its faster reads). "
        "Best static = the one setting with the lowest total cost in that setup. NeuroPress = online learning without "
        "exploration (model rows: a replay of Clio's selection and learning, with its measured selection and training "
        "time). HCompress = Clio's HCompress (library + size model with feedback after every chunk; model rows: a replay "
        "of it at each chunk's speed, with its measured selection time). XGBoost = boosted trees on NeuroPress's "
        "training data and features, no online learning (model rows: its lowest predicted cost at each chunk's speed, "
        "with its measured selection time). Real runs: 1 process, 1 chunk in flight, page cache dropped before each read. Model rows: computed from "
        "the measured exhaustive search (codec times and ratio of every chunk and setting).", 215),
        fontsize=8.5, color=INK_2, va="bottom")
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
