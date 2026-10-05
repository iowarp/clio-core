#!/usr/bin/env python3
"""Does NeuroPress v2's online learning converge if the workload runs longer?

    plot_learning_over_time.py [--summary CSV] [--out PNG] [--pdf]

Reads replay_learning.py's per-pass summary (runs/replay_summary_20pass.csv:
each workload replayed 20 times in a row through the exact Clio selection and
learning loop, labels from the exhaustive search) and draws, per workload,
the balanced 4-tier cost of NeuroPress's picks against the best single codec
pass by pass: the model as trained, with Clio's online learning, and the
per-chunk best (oracle).
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import eval_v2_workloads as ev

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = "/mnt/nvme0/v2-work/runs"
FIGS = os.path.join(HERE, "..", "figures", "new-workloads", "nn-v2")
ORDER = ev.FULL_WORKLOADS
COLS = 4
INK, INK2 = "#1f2328", "#57606a"
LINES = [("frozen", "#8c959f", "as trained (no learning)"),
         ("nlms lr 0.5 (Clio)", "#d1495b", "with online learning (Clio's rule)")]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--summary", default=os.path.join(RUNS, "replay_summary_20pass.csv"))
    ap.add_argument("--out", default=os.path.join(FIGS, "v2_learning_over_time.png"))
    ap.add_argument("--pdf", action="store_true", help="also write a PDF")
    a = ap.parse_args()
    t = pd.read_csv(a.summary)
    dss = [d for d in ORDER if d in set(t.workload)]
    plt.rcParams["font.family"] = "DejaVu Sans"
    cols = min(COLS, len(dss))
    rows = -(-len(dss) // cols)
    H = 1.6 + 3.6 * rows
    fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, H), squeeze=False)
    fig.patch.set_facecolor("white")
    axes = axes.ravel()
    for ax in axes[len(dss):]:
        ax.set_visible(False)
    for ax, ds in zip(axes, dss):
        d = t[t.workload == ds]
        n = int(d.chunks.iloc[0])
        orc = d.oracle_vs_best_single_pct.iloc[0]
        learn = d[d.rule == LINES[1][0]].cost_vs_best_single_pct
        # The y range follows the learning curve, the oracle and 0; a line far
        # outside it (the model as trained, up to +800%) is not drawn to scale
        # but written at the top edge with its value.
        lo = min(learn.min(), orc, 0.0)
        hi = max(learn.max(), orc, 0.0)
        pad = 0.08 * (hi - lo) + 0.5
        top = hi + 3 * (hi - lo) + 1
        for rule, colour, lab in LINES:
            r = d[d.rule == rule].sort_values("pass")
            v = r.cost_vs_best_single_pct
            if v.max() > top:
                ax.plot([], [], color=colour, lw=2, marker="o", ms=3, label=lab)
                ax.annotate(f"{lab.split(' (')[0]}: {v.iloc[-1]:+.0f}% (off scale) \u2191",
                            xy=(0.98, 0.97), xycoords="axes fraction", ha="right",
                            va="top", fontsize=8.5, color=colour)
                continue
            hi = max(hi, v.max())
            ax.plot(r["pass"], v, color=colour, lw=2, marker="o", ms=3, label=lab)
            ax.text(r["pass"].iloc[-1] + 0.3, v.iloc[-1], f"{v.iloc[-1]:+.0f}%",
                    color=colour, fontsize=8.5, va="center")
        ax.set_ylim(lo - pad, hi + pad + (0.12 * (hi - lo + 1)))
        ax.axhline(orc, color="#2e86ab", lw=1.2, ls="--", label="each chunk's best setting")
        ax.axhline(0, color=INK2, lw=0.8)
        ax.text(20.3, orc, f"{orc:+.0f}%", color="#2e86ab", fontsize=8.5, va="top")
        n_c, gib = ev.workload_size(ds)
        ax.set_title(f"{ds}\n({n_c} chunks, {gib:.1f} GiB per pass)", loc="left",
                     fontsize=10.5, color=INK)
        ax.set_xlabel("pass over the workload (1 = the real run)", fontsize=8.5,
                      color=INK2)
        ax.set_xlim(0.5, 22.5)
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.grid(color="#e6e9ed", lw=0.6)
        ax.tick_params(labelsize=8, colors=INK2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes[::cols]:
        ax.set_ylabel("cost vs best single codec\n(%; < 0 = cheaper)", fontsize=9, color=INK)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=9, loc="upper right", ncol=3,
               bbox_to_anchor=(0.99, 1 - 0.02 / H))
    fig.suptitle("Does online learning converge if the workload runs longer? "
                 f"({ev.total_label(dss)} per pass)",
                 x=0.01, ha="left", fontsize=14, color=INK, y=1 - 0.03 / H)
    fig.text(0.01, 1 - 0.45 / H, "Offline replay of Clio's exact selection and learning loop "
             "(checked: same picks as the Clio runs on 100% of chunks), each workload "
             "repeated 20 times in a row as if the application kept writing; each "
             "chunk's outcome taken from\nthe exhaustive search. Y: balanced 4-tier cost "
             "(compress + decompress + I/O at the chunk's tier) of NeuroPress's picks "
             "in that pass vs storing every chunk with the best single codec.",
             fontsize=9, color=INK2, va="top", linespacing=1.45)
    fig.subplots_adjust(left=0.07, right=0.96, top=1 - 1.45 / H, bottom=0.5 / H, wspace=0.3,
                        hspace=0.6)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=150)
    if a.pdf:
        fig.savefig(os.path.splitext(a.out)[0] + ".pdf")
    print("wrote", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
