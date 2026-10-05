#!/usr/bin/env python3
"""Best single codec vs NeuroPress v2 learning on one workload: total measured
application wall-clock time and compression ratio, through Clio.

    plot_fixed_vs_learn.py DATASET --codec NAME [--tag TAG] [--out PNG] [--pdf]

Reads runs/<DATASET>_{fixed,learn}_<TAG>. Wall-clock time is measured by the
replay over the whole workload: writing every chunk through Clio (choose,
compress, learn, store) and reading every chunk back (load, decompress), with
Clio's start-up, the reading of the input files and the bit-exact check left
out (the same for both methods). Compression ratio is original bytes / stored
bytes over the whole workload (blobs.csv). The third panel shows which
settings NeuroPress stored, as shares of the chunks.
"""
import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import eval_v2_workloads as ev

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = "/mnt/nvme0/v2-work/runs"
FIGS = os.path.join(HERE, "..", "figures", "new-workloads", "nn-v2")
INK, INK2, FIX, NP, NPX = "#1f2328", "#57606a", "#1f2328", "#d1495b", "#e08e0b"


def wall(run):
    """(write s, read s): whole-workload wall clock from stdout.log."""
    text = open(os.path.join(run, "stdout.log")).read()
    w = re.search(r"stage\+compress ([0-9.]+) s", text)
    r = re.search(r"READ 1/1.*?get\+decompress: ([0-9.]+) ms", text, re.S)
    return float(w.group(1)), float(r.group(1)) / 1e3


def ratio(run):
    """Whole-workload compression ratio: original / stored bytes."""
    b = pd.read_csv(os.path.join(run, "blobs.csv")).drop_duplicates("blob")
    return b.bytes.sum() / b.stored.sum()


def bars(ax, groups, values, fmt, colours):
    """Side-by-side bars (one per method) per group, labelled."""
    n = len(colours)
    w = 0.8 / n
    for k, vals in enumerate(values):
        for j, (v, c) in enumerate(zip(vals, colours)):
            x = k - 0.4 + w * (j + 0.5)
            ax.bar(x, v, w * 0.95, color=c)
            ax.text(x, v, fmt(v), ha="center", va="bottom", fontsize=8.5, color=c)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=9.5, color=INK)
    ax.grid(axis="y", color="#e6e9ed", lw=0.6)
    ax.tick_params(labelsize=8.5, colors=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def short(setting):
    """A setting's display name: 'store' is raw, and cascaded drops the
    'bp=1 delta=0' and 'type=int' that all its variants share."""
    if setting == "store":
        return "raw"
    if setting.startswith("cascaded"):
        return setting.replace("bp=1 delta=0 ", "").replace(" type=int", "")
    return setting


def choices(ax, run, best, top=10, colour=NP, label="NeuroPress"):
    """Share of chunks per setting NeuroPress stored; the best single codec's
    bar outlined. 'raw' = stored uncompressed."""
    names, store = ev.settings_list()
    d = ev.run_chunks(run, store)
    share = pd.Series([names[k] for k in d.stored]).value_counts(normalize=True)
    shown = share.head(top)
    rest = share.iloc[top:]
    labels = ["raw (uncompressed)" if k == "store" else short(k) for k in shown.index]
    vals = list(100 * shown.values)
    if len(rest):
        labels.append(f"{len(rest)} other settings")
        vals.append(100 * rest.sum())
    y = range(len(vals))[::-1]
    for yy, lab, v in zip(y, labels, vals):
        is_best = lab == short(best)
        ax.barh(yy, v, 0.7, color=colour, edgecolor=INK if is_best else colour,
                lw=1.6 if is_best else 0)
        pct = f"{v:.0f}%" if v >= 1 else f"{v:.1f}%"
        ax.text(v + 0.5, yy, pct + ("  (= best single codec)" if is_best
                                             else ""), va="center", fontsize=8.5,
                color=INK)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8.5, color=INK)
    ax.set_xlabel("share of chunks", fontsize=9.5, color=INK2)
    ax.set_xlim(0, max(vals) * 1.45)
    ax.set_title(f"What {label} stored ({len(share)} settings)",
                 loc="left", fontsize=11, color=INK)
    ax.grid(axis="x", color="#e6e9ed", lw=0.6)
    ax.tick_params(axis="x", labelsize=8.5, colors=INK2)
    for sd in ("top", "right"):
        ax.spines[sd].set_visible(False)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dataset")
    ap.add_argument("--codec", required=True, help="the best single codec's name")
    ap.add_argument("--tag", default="nolog")
    ap.add_argument("--out", default=None)
    ap.add_argument("--pdf", action="store_true", help="also write a PDF")
    ap.add_argument("--with-explore", action="store_true",
                    help="also show the learn + explore run (off by default)")
    a = ap.parse_args()
    fx = os.path.join(RUNS, f"{a.dataset}_fixed_{a.tag}")
    le = os.path.join(RUNS, f"{a.dataset}_learn_{a.tag}")
    lx = os.path.join(RUNS, f"{a.dataset}_learnexp_{a.tag}")
    arms = [(fx, FIX, f"best single codec: {short(a.codec)}"),
            (le, NP, "NeuroPress v2 learning (exploration off)")]
    if a.with_explore and os.path.exists(os.path.join(lx, "stdout.log")):
        arms.append((lx, NPX, "NeuroPress v2 learning + exploration"))
    walls = [wall(r) for r, _, _ in arms]
    colours = [c for _, c, _ in arms]
    plt.rcParams["font.family"] = "DejaVu Sans"
    np_arms = arms[1:]
    fig, axs = plt.subplots(1, 2 + len(np_arms), figsize=(13 + 6.5 * len(np_arms), 5.4),
                            gridspec_kw={"width_ratios": [2.0, 0.9] + [1.6] * len(np_arms)})
    at, ar = axs[0], axs[1]
    fig.patch.set_facecolor("white")
    bars(at, ["write the workload", "read it back", "total (write + read)"],
         [[w for w, _ in walls], [r for _, r in walls], [w + r for w, r in walls]],
         lambda v: f"{v:.2f} s", colours)
    at.set_ylabel("seconds, measured wall clock", fontsize=9.5, color=INK2)
    at.set_title("Total application wall-clock time (shorter = faster)",
                 loc="left", fontsize=11, color=INK)
    bars(ar, ["whole workload"], [[ratio(r) for r, _, _ in arms]],
         lambda v: f"{v:.2f}x", colours)
    ar.set_ylabel("original bytes / stored bytes", fontsize=9.5, color=INK2)
    ar.set_title("Compression ratio (higher = smaller)", loc="left", fontsize=11,
                 color=INK)
    for ax, (r, c, lab) in zip(axs[2:], np_arms):
        choices(ax, r, a.codec, colour=c,
                label="learn + explore" if r == lx else "learning only")
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in colours],
               [lab for _, _, lab in arms], loc="upper left", ncol=3, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.01, 0.86), labelcolor=INK)
    fig.suptitle(f"{a.dataset}: best single codec vs NeuroPress learning, the "
                 "whole workload through Clio", x=0.01, ha="left", fontsize=13,
                 color=INK, y=0.99)
    fig.text(0.01, 0.94, "Best single codec = cheapest of all 45 settings by the "
             "balanced 4-tier cost model (exhaustive run).\nSame chunks and order, "
             "GPU clock locked, benchmark logging off; Clio start-up, input-file "
             "reading and the bit-exact check are not timed.", fontsize=8.8,
             color=INK2, va="top", linespacing=1.4)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.74, bottom=0.1, wspace=0.62)
    out = a.out or os.path.join(FIGS, f"v2_{a.dataset}_fixed_vs_learn.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    if a.pdf:
        fig.savefig(os.path.splitext(out)[0] + ".pdf")
    print("wrote", os.path.abspath(out))


if __name__ == "__main__":
    main()
