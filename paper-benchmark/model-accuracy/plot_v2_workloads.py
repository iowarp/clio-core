#!/usr/bin/env python3
"""NeuroPress v2 on the new workloads inside Clio: cost against the per-chunk best.

    plot_v2_workloads.py [--summary CSV] [--out PNG]

One row per dataset (eval_v2_workloads.py's summary of the static, learnexp
and exhaustive runs on the 4-tier cost model): total cost of what was stored,
as a percentage above the per-chunk best (the cheapest of all 45 settings on
each chunk, from the exhaustive run), for the best single setting for that
dataset (chosen in hindsight), learn only (no exploration) and learn +
explore. The right columns state each v2 variant's cost relative to the
best single setting, with v2's own time (prediction for every chunk,
plus exploration where it ran) added to v2's bars and verdicts.
Labels name the codec behind each bar: the best single setting, and for
each v2 variant (which chooses per chunk) the two settings it stored most
often with their share of chunks and how many others it used.
Rows are split into the datasets where learn + explore is cheaper than the
best single setting and those where it is not.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY = "/mnt/nvme0/v2-work/runs/v2_workloads_summary.csv"
CHUNKS = "/mnt/nvme0/v2-work/runs/v2_workloads_chunks.csv"
FIGS = os.path.join(HERE, "..", "figures", "new-workloads", "nn-v2")
INK, INK2, GOOD, BAD = "#1f2328", "#57606a", "#1a7f37", "#cf222e"
SINGLE = ("One codec for the whole dataset (the best one, picked in hindsight)",
          "#1f2328")
# NeuroPress v2 variants drawn: (run mode, legend, colour, verdict heading).
VARIANTS = [("learn", "NeuroPress v2, learning only (no exploration)", "#2e86ab",
             "learning only"),
            ("learnexp", "NeuroPress v2, learning + exploration", "#d1495b",
             "learning +\nexploration")]
XMAX = 50.0  # x-axis end, set in main() from the data
CUT = 90.0   # bars longer than this are cut at it and labelled
H = 0.25     # bar height


def name(setting):
    """A setting's display name: 'store' is raw (uncompressed), and cascaded
    drops the 'bp=1 delta=0' and 'type=int' that all its variants share."""
    if setting == "store":
        return "raw"
    if setting.startswith("cascaded"):
        return setting.replace("bp=1 delta=0 ", "").replace(" type=int", "")
    return setting


def mix(chunks, mode):
    """Per dataset: one run's stored settings as a short label."""
    c = chunks[chunks["mode"] == mode]
    out = {}
    for ds, x in c.groupby("dataset"):
        share = x.stored.value_counts(normalize=True)
        top = ", ".join(f"{name(k)} {100 * p:.0f}%" for k, p in share.head(2).items())
        more = len(share) - 2
        out[ds] = f"({top}" + (f", +{more} more" if more > 0 else "") + ")"
    return pd.Series(out)


def table(path, chunks_path):
    """Per dataset: % above the per-chunk best for the best single setting
    and each v2 variant, and each variant relative to the best single."""
    s = pd.read_csv(path)
    st = s[s["mode"] == "static"].set_index("dataset")
    chunks = pd.read_csv(chunks_path, usecols=["dataset", "mode", "stored"])
    t = pd.DataFrame({"single": st.best_single_over_oracle_pct,
                      "chunks": st.chunks, "codec": st.best_single,
                      "oracle_ms": st.oracle_cost_ms})
    for mode, *_ in VARIANTS:
        run = s[s["mode"] == mode].set_index("dataset")
        t[mode] = run.over_oracle_pct
        # v2's own working time, charged on top of what it stored: prediction
        # for every chunk, and exploration where it ran.
        t[f"oh_{mode}"] = 100 * (run.select_ms + run.explore_ms) / t.oracle_ms
        t[f"tot_{mode}"] = t[mode] + t[f"oh_{mode}"]
        t[f"vs_{mode}"] = 100 * ((1 + t[f"tot_{mode}"] / 100) / (1 + t.single / 100) - 1)
        t[f"mix_{mode}"] = mix(chunks, mode)
    return t.sort_values("vs_learnexp")


def bar(ax, y, value, colour, text=None, overhead=0.0):
    """One horizontal bar with its value printed after it; overhead (v2's
    own working time) is drawn as a hatched extension and counted in the
    printed total."""
    total = value + overhead
    solid = min(value, CUT)
    ax.barh(y, solid, height=H, color=colour)
    if overhead > 0 and solid < CUT:
        ax.barh(y, min(total, CUT) - solid, left=solid, height=H, color="white",
                edgecolor=colour, hatch="////", lw=0.6)
    shown = min(total, CUT)
    if total > CUT:  # an arrow end marks the cut
        ax.plot([shown], [y], marker=">", color="white", markersize=5)
    value = total
    label = f"{value:,.1f}%" + (" (bar cut)" if value > CUT else "")
    if overhead > 0:
        label += f" [{overhead:.1f}% v2 time]"
    label += f"  {text}" if text else ""
    if shown > XMAX * 0.35:  # long bar: label inside it, clear of the verdicts
        ax.text(shown - XMAX * 0.01, y, label, va="center", ha="right",
                fontsize=7.4, color=colour, weight="bold",
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none",
                          alpha=0.9))
    else:
        ax.text(shown + XMAX * 0.006, y, label, va="center", fontsize=7.6,
                color=colour)


def verdict(ax, x, y, vs):
    """'N% cheaper' / 'N% costlier' than the best single codec, at axes x."""
    v = f"{abs(vs):.1f}" if abs(vs) < 10 else f"{abs(vs):.0f}"
    good = vs < 0
    ax.text(x, y, f"{v}% " + ("cheaper" if good else "costlier"),
            transform=ax.get_yaxis_transform(), va="center", fontsize=8.8,
            weight="bold", color=GOOD if good else BAD)


def rows(ax, t):
    """Bars per dataset (best single, then each v2 variant), a verdict column
    per variant, and a divider between the datasets where learning +
    exploration beats the best single codec and those where it does not."""
    n = len(t)
    wins = int((t.vs_learnexp < 0).sum())
    gap = 1.2  # extra space between the two groups
    ys = [n - 1 - i + (gap if i < wins else 0) for i in range(n)]
    cols = [1.01 + 0.13 * k for k in range(len(VARIANTS))]
    for y, (_ds, r) in zip(ys, t.iterrows()):
        bar(ax, y + H * 1.1, r.single, SINGLE[1], f"({name(r.codec)})")
        for k, (mode, _, colour, _) in enumerate(VARIANTS):
            bar(ax, y - H * 1.1 * k, r[mode], colour, r[f"mix_{mode}"],
                r[f"oh_{mode}"])
            verdict(ax, cols[k], y, r[f"vs_{mode}"])
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{d}\n{int(c)} chunks" for d, c in zip(t.index, t.chunks)],
                       fontsize=8.5, color=INK, linespacing=1.1)
    split = ys[wins - 1] - 0.5 - gap / 2 if 0 < wins < n else None
    if split is not None:
        ax.axhline(split, color=INK2, lw=0.8, ls=(0, (4, 3)))
        ax.text(XMAX * 0.98, ys[0] + 0.75, "NeuroPress v2 (learning + exploration) "
                "beats the best single codec", ha="right", fontsize=9.5, color=GOOD,
                weight="bold")
        ax.text(XMAX * 0.98, split - 0.45, "one codec is enough: the best single "
                "codec beats NeuroPress v2", ha="right", va="top", fontsize=9.5,
                color=BAD, weight="bold")
    ax.text(cols[0], ys[0] + 1.5, "vs the best single codec:", fontsize=8.5,
            color=INK2, transform=ax.get_yaxis_transform(), va="bottom")
    for k, (_, _, colour, head) in enumerate(VARIANTS):
        ax.text(cols[k], ys[0] + 0.75, head, transform=ax.get_yaxis_transform(),
                fontsize=8.5, color=colour, va="bottom", weight="bold")
    ax.set_ylim(-1.2, ys[0] + 2.3)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--summary", default=SUMMARY)
    ap.add_argument("--chunks", default=CHUNKS)
    ap.add_argument("--out", default=os.path.join(FIGS, "v2_workloads_cost_vs_best.png"))
    ap.add_argument("--pdf", action="store_true",
                    help="also write a PDF next to the PNG")
    a = ap.parse_args()
    t = table(a.summary, a.chunks)
    global XMAX
    XMAX = 1.45 * min(CUT, max(t.single.max(),
                               *(t[f"tot_{m}"].max() for m, *_ in VARIANTS)))
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(15, 14))
    fig.patch.set_facecolor("white")
    rows(ax, t)
    ax.set_xlim(0, XMAX)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.0f}%"))
    ax.set_xlabel("extra cost compared with a perfect choice on every chunk "
                  "(0% = perfect; shorter bar = cheaper)", fontsize=9.5, color=INK2)
    ax.grid(axis="x", color="#e6e9ed", lw=0.6)
    ax.tick_params(axis="x", labelsize=8, colors=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    entries = [SINGLE] + [(lab, c) for _, lab, c, _ in VARIANTS]
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in entries]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="white",
                                 edgecolor=INK2, hatch="////"))
    fig.legend(handles, [lab for lab, _ in entries] +
               ["hatched = NeuroPress v2's own time (prediction + exploration)"],
               loc="upper left", bbox_to_anchor=(0.01, 0.9), ncol=2,
               frameon=False, fontsize=9.5, labelcolor=INK)
    fig.suptitle("Does NeuroPress v2 beat simply using the best codec? "
                 "19 workloads stored through Clio", x=0.01, ha="left",
                 fontsize=14, color=INK, y=0.995)
    oracle = t.oracle_ms.sum()
    totals = {k: (t.oracle_ms * (1 + t[k] / 100)).sum() / oracle * 100 - 100
              for k in ["single"] + [f"tot_{m}" for m, *_ in VARIANTS]}
    fig.text(0.01, 0.958,
             "How to read: for each chunk, all 45 lossless settings were measured, so "
             "the cheapest one per chunk is known. Using it on every chunk is the "
             "perfect choice (0%).\nEach bar is how much MORE a strategy paid in total "
             "for the dataset. Cost of a chunk = compress time + decompress time + "
             "time to write it to its storage tier (12 / 1 / 0.5 / 0.25 GB/s).\n"
             "NeuroPress v2 bars also include its own time (hatched): predicting a "
             "setting for every chunk, and exploration (trying other codecs and "
             "writing a better one); the verdicts on the right include it too.\n"
             "Names after a bar: the codec it used. NeuroPress v2 chooses per chunk, so "
             "its label lists the two settings it stored most often (share of chunks) "
             "and how many others it used; raw = stored uncompressed.\n"
             f"All 19 datasets together, above perfect: best single codec "
             f"+{totals['single']:.1f}%, NeuroPress v2 learning only "
             f"+{totals['tot_learn']:.1f}%, learning + exploration "
             f"+{totals['tot_learnexp']:.1f}% (v2's own time included).", fontsize=9.2,
             color=INK2,
             va="top", linespacing=1.45)
    fig.subplots_adjust(left=0.15, right=0.78, top=0.86, bottom=0.04)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=160)
    if a.pdf:
        fig.savefig(os.path.splitext(a.out)[0] + ".pdf")
    print("wrote", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
