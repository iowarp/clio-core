#!/usr/bin/env python3
"""Per-chunk compress time vs NeuroPress v2's prediction time inside Clio.

    plot_v2_overhead.py [--dataset vpic2k|nyx2k] [--out PNG]

For each chunk in stream order: the measured compress time of the setting v2
picked (v2_measured.csv, role=primary; codec kernel time incl. shuffle), and
v2's prediction time for that chunk (v2_pred.csv select_ms: features +
network + ranking; the element-type conversion is timed separately and
excluded). Learn + explore also shows the time spent measuring alternatives
(phases.csv explore_ms) on the chunks where exploration ran. Columns: static
v2, learn + explore.
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
RUNS = "/mnt/nvme0/v2-work/runs"
FIGS = os.path.join(HERE, "..", "figures", "new-workloads", "nn-v2")
ARMS = [("static v2", "static"), ("learn + explore", "learnexp")]
TITLES = {"nyx2k": "Nyx", "vpic2k": "VPIC"}
INK, INK2 = "#1f2328", "#57606a"
COMP, PRED, EXPL = "#1f2328", "#2e86ab", "#e08e0b"


def load(ds, mode):
    """Per-chunk compress ms, prediction ms and exploration ms, stream order.

    @param ds    dataset name (runs/<ds>_<mode>)
    @param mode  run mode
    @return DataFrame with i, comp_ms, pred_ms, explore_ms
    """
    run = os.path.join(RUNS, f"{ds}_{mode}")
    p = pd.read_csv(os.path.join(run, "v2_pred.csv"), usecols=["blob", "select_ms"])
    m = pd.read_csv(os.path.join(run, "v2_measured.csv"))
    prim = m[(m.role == "primary") & (m.comp_ms > 0)].drop_duplicates("blob")
    ph = pd.read_csv(os.path.join(run, "phases.csv"))
    ph = ph[ph.path == "write"].drop_duplicates("chunk_id").set_index("chunk_id")
    return pd.DataFrame({
        "i": np.arange(len(p)),
        "comp_ms": prim.set_index("blob").comp_ms.reindex(p.blob).to_numpy(),
        "pred_ms": p.select_ms.to_numpy(),
        "explore_ms": ph.explore_ms.reindex(p.blob).to_numpy()})


def ms(v, _p=None):
    """Plain millisecond tick."""
    return f"{v:g}" if v >= 0.01 else f"{v:.3f}"


def panel(ax, d, title):
    """Dots per chunk for compress, prediction and exploration time."""
    ax.scatter(d.i, d.comp_ms, s=4, color=COMP, lw=0, alpha=0.7,
               label="compress time of the picked setting")
    ax.scatter(d.i, d.pred_ms, s=4, color=PRED, lw=0, alpha=0.7,
               label="v2 prediction time (features + network + ranking)")
    e = d[d.explore_ms > 0]
    if len(e):
        ax.scatter(e.i, e.explore_ms, s=4, color=EXPL, lw=0, alpha=0.6,
                   label="exploration time (measuring alternatives)")
    c, p = d.comp_ms.dropna(), d.pred_ms
    text = (f"median: compress {c.median():.3f} ms, prediction {p.median():.3f} ms"
            f"\nsum: compress {c.sum():,.0f} ms, prediction {p.sum():,.0f} ms")
    if len(e):
        text += f", exploration {e.explore_ms.sum():,.0f} ms on {len(e)} chunks"
    ax.text(0.99, 0.97, text, transform=ax.transAxes, ha="right", va="top",
            fontsize=8.5, color=INK, bbox=dict(boxstyle="round,pad=0.3",
                                               fc="white", ec="#d0d7de"))
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(ms))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, p: ""))
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_ylabel("ms per chunk", fontsize=9, color=INK2)
    ax.grid(axis="y", color="#e6e9ed", lw=0.6)
    ax.tick_params(labelsize=8, colors=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default="vpic2k", choices=sorted(TITLES))
    ap.add_argument("--out", default=None)
    ap.add_argument("--pdf", action="store_true",
                    help="also write a PDF next to the PNG")
    a = ap.parse_args()
    out = a.out or os.path.join(FIGS, f"v2_{a.dataset}_compress_vs_prediction_time.png")
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(2, 1, figsize=(15, 8.5), sharex=True, sharey=True)
    fig.patch.set_facecolor("white")
    for ax, (label, mode) in zip(axes, ARMS):
        panel(ax, load(a.dataset, mode), f"{TITLES[a.dataset]} -- {label}")
    axes[-1].set_xlabel("chunk, in stream order", fontsize=9, color=INK2)
    h, l = axes[-1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=9,
               labelcolor=INK2, markerscale=3)
    fig.suptitle(f"NeuroPress v2 on 2,000 {TITLES[a.dataset]} chunks inside Clio: "
                 "compress time vs prediction time, per chunk", x=0.01, ha="left",
                 fontsize=12.5, color=INK, y=0.99)
    fig.text(0.01, 0.935, "Compress time: codec kernel time of the setting v2 picked, "
             "shuffle included (chunks stored raw by a store pick have none). "
             "Prediction time: excludes the element-type conversion. "
             "GPU clock locked at 1,410 MHz.", fontsize=9, color=INK2)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.11, hspace=0.18)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=160)
    if a.pdf:
        fig.savefig(os.path.splitext(out)[0] + ".pdf")
    print("wrote", os.path.abspath(out))


if __name__ == "__main__":
    main()
