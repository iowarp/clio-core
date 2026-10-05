#!/usr/bin/env python3
"""End-to-end producer-consumer results, one panel per workload.

    plot_pipeline.py [--runs ~/np-pipeline] [--out PNG]

Each panel has three bars -- store uncompressed, best single codec (one
setting for every chunk), per-chunk (each chunk its own setting) -- stacked
by phase, mean over the reps: compress (producer data + analysis outputs),
write, read, decompress, CPU<->GPU copies, the consumer's analysis. The
whisker is the std of end-to-end time over reps. Right of each bar: total
seconds, change against uncompressed and against the best single codec,
compression ratio of the producer data and bytes written.
"""
import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import plot_new_workloads as pw  # noqa: E402
cm = pw.cm

PHASES = [  # label, columns summed, colour
    ("compress", ["prod_compress_s", "cons_out_compress_s"], cm.ORANGE),
    ("write", ["prod_write_s", "cons_out_write_s"], cm.BLUE),
    ("read", ["cons_read_s"], "#8cb8ea"),
    ("decompress", ["cons_decompress_s"], "#f4a882"),
    ("CPU-GPU copies", ["prod_d2h_s", "cons_h2d_s"], cm.AXIS),
    ("analysis", ["cons_analysis_s"], cm.INK2),
]
POLICIES = [("none", "uncompressed"), ("best", "best single codec"), ("perchunk", "per-chunk")]
TITLES = {"vpic": "VPIC  sim -> k-means / structures / PDF",
          "lammps": "LAMMPS  sim -> coordination / RDF / k-means",
          "nanoaod": "CMS NanoAOD -> Z->ee selection / mass / PDFs",
          "fem": "FEM matrices -> 100 CG iterations"}
ORDER = ["vpic", "lammps"]   # nanoaod, fem removed (user, 2026-10-05)


def panel(ax, d, meta):
    """Stacked phase bars for one workload's runs.csv."""
    g = d.groupby("policy")
    mean, std = g.mean(numeric_only=True), g.std(numeric_only=True)
    ylab = []
    for i, (pol, name) in enumerate(POLICIES):
        r = mean.loc[pol]
        left = 0.0
        for _, cols, col in PHASES:
            w = sum(r[c] for c in cols)
            ax.barh(i, w, left=left, color=col, height=0.62, edgecolor=cm.SURFACE, linewidth=0.5)
            left += w
        tot = r["end_to_end_s"]
        ax.errorbar(tot, i, xerr=std.loc[pol, "end_to_end_s"], color=cm.INK, capsize=3, lw=1)
        vs = [f"{100 * (tot / mean.loc[ref, 'end_to_end_s'] - 1):+.1f}% vs {lab}"
              for ref, lab in (("none", "uncompr."), ("best", "best")) if ref != pol]
        ratio = r["prod_in"] / r["prod_stored"]
        gb = (r["prod_stored"] + r["out_stored"]) / 1e9
        ax.text(tot * 1.02 + 0.02 * mean["end_to_end_s"].max(), i,
                f"{tot:.2f} s   " + ("  ".join(vs) if pol != "none" else "") +
                f"\nratio {ratio:.2f}x, {gb:.1f} GB written",
                va="center", fontsize=8.3, color=cm.INK,
                weight="bold" if pol == "perchunk" else "normal")
        lab = name if pol != "best" else f"{name}\n{pw.short_config(meta['best_setting'])}"
        ylab.append(lab)
    ax.set_yticks(range(len(POLICIES)))
    ax.set_yticklabels(ylab, fontsize=8.8, color=cm.INK)
    ax.invert_yaxis()
    ax.set_xlim(0, mean["end_to_end_s"].max() * 1.5)
    ax.set_xlabel("seconds (mean of reps)", fontsize=8.5, color=cm.INK2)
    cm.style_axes(ax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.expanduser("~/np-pipeline"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "..", "..", "figures", "new-workloads",
                                                  "pipeline_end_to_end.png"))
    a = ap.parse_args()
    found = {os.path.basename(os.path.dirname(f)): f
             for f in glob.glob(os.path.join(a.runs, "*", "runs.csv"))}
    keys = [k for k in ORDER if k in found]
    fig, axes = plt.subplots(len(keys), 1, figsize=(13, 2.55 * len(keys) + 1.7))
    fig.patch.set_facecolor(cm.SURFACE)
    for ax, k in zip(axes, keys):
        d = pd.read_csv(found[k])
        meta = json.load(open(os.path.join(os.path.dirname(found[k]), "plan.json")))["meta"]
        panel(ax, d, meta)
        ax.set_title(f"{TITLES.get(k, k)}   ({d.frames.iloc[0]} frames, "
                     f"{d.prod_in.iloc[0] / 1e9:.1f} GB produced, {d.rep.max()} reps)",
                     loc="left", fontsize=10.5, color=cm.INK, pad=5)
    bw = json.load(open(os.path.join(os.path.dirname(found[keys[0]]), "plan.json")))["meta"]
    fig.legend(handles=[Patch(color=c, label=l) for l, _, c in PHASES], loc="upper left", ncol=6,
               frameon=False, fontsize=9, bbox_to_anchor=(0.01, 1 - 0.78 / fig.get_figheight()),
               labelcolor=cm.INK2)
    fig.suptitle("Producer -> storage -> consumer, end to end: uncompressed vs best single codec vs "
                 "per-chunk codec choice\n"
                 f"NVMe {bw['bw_write_gbs']:.2f} GB/s write / {bw['bw_read_gbs']:.2f} GB/s read; "
                 "every frame fsynced, page cache dropped before the consumer reads",
                 x=0.01, ha="left", fontsize=11.5, color=cm.INK, y=1 - 0.2 / fig.get_figheight())
    fig.subplots_adjust(left=0.155, right=0.985, top=1 - 1.5 / fig.get_figheight(), bottom=0.05,
                        hspace=0.62)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=150, facecolor=cm.SURFACE)
    print("wrote", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
