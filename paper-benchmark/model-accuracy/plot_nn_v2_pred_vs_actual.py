#!/usr/bin/env python3
"""Predicted vs measured for the NeuroPress v2 model: synthetic CV and real VPIC.

    plot_nn_v2_pred_vs_actual.py [--out PNG]

Top row: held-out predictions of model B from the 5-fold CV on the synthetic
corpus (cv_predictions.npz; a seeded sample of 15,000 of the 441,000 points).
Bottom row: the saved model_v2.nnwt on the real VPIC chunks the sweep measured
(~/np-newsweep/ref-vpic-126-2000), every point. Columns: compress time,
decompress time, compression ratio, log-log, one point per (chunk, setting),
coloured by codec family. The diagonal is a perfect prediction; the band is
+-25 %. Each panel gives MAPE and the median absolute percentage error.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "codec-sweep"))
import eval_nn_v2_real as ev  # noqa: E402
import train_nn_v2 as tr  # noqa: E402

FAMILIES = [  # label, codecs, colour
    ("LZ / Deflate family (LZ4, Snappy, Zstd, GDeflate, Deflate, GPULZ)",
     {"lz4", "snappy", "zstd", "gdeflate", "deflate", "gpulz"}, "#d1495b"),
    ("Float codecs (ndzip, SPspeed, SPratio)", {"ndzip", "spspeed", "spratio"}, "#2e86ab"),
    ("Other nvcomp (ANS, Bitcomp, Cascaded)", {"ans", "bitcomp", "cascaded"}, "#edae49"),
    ("Store (uncompressed)", {"store"}, "#6c757d"),
]
TITLES = ["Compress time (ms)", "Decompress time (ms)", "Compression ratio"]
INK, INK2 = "#1f2328", "#57606a"


def family_colour(settings):
    """@return one colour per setting, by codec family."""
    out = []
    for s in settings:
        c = s.split()[0]
        out.append(next(col for _, codecs, col in FAMILIES if c in codecs))
    return np.array(out)


def synthetic(n_sample, seed=0):
    """Held-out CV predictions of model B (true, pred: points x 3; colours)."""
    z = np.load("/mnt/nvme0/corpus-sweep/nn-v2/cv_predictions.npz", allow_pickle=True)
    true, pred = np.exp(z["true"]), np.exp(z["B"])
    col = np.broadcast_to(family_colour(z["settings"])[None, :], true.shape[:2])
    t, p, c = true.reshape(-1, 3), pred.reshape(-1, 3), col.reshape(-1)
    full = (t, p)
    i = np.random.default_rng(seed).choice(len(t), size=n_sample, replace=False)
    return t[i], p[i], c[i], full


def vpic():
    """Saved model on the VPIC chunks (true, pred: points x 3; colours)."""
    meta = json.load(open(ev.WEIGHTS[:-5] + ".json"))
    settings = [ev.canon(s) for s in meta["outputs"]["settings"]]
    names, _, true = ev.load_measured(os.path.expanduser("~/np-newsweep/ref-vpic-126-2000"),
                                      settings)
    x = np.array([ev.chunk_features(os.path.expanduser("~/np-data/vpic-126-2000/fields"),
                                    n, 4 << 20)[0] for n in names])
    pred = tr.numpy_forward(ev.WEIGHTS, x).astype(float).reshape(len(names), len(settings), 3)
    col = np.broadcast_to(family_colour(settings)[None, :], true.shape[:2])
    t, p = np.exp(true).reshape(-1, 3), np.exp(pred).reshape(-1, 3)
    return t, p, col.reshape(-1), (t, p), len(names)


def panel(ax, t, p, col, full, k, row_label):
    """One log-log scatter with the diagonal, a +-25 % band and the errors."""
    order = np.random.default_rng(1).permutation(len(t))  # no family drawn on top
    ax.scatter(t[order], p[order], s=4, c=col[order], alpha=0.35, linewidths=0, rasterized=True)
    lo = min(t.min(), p.min()) * 0.8
    hi = max(t.max(), p.max()) * 1.25
    xs = np.array([lo, hi])
    ax.plot(xs, xs, color=INK, lw=1)
    ax.fill_between(xs, xs / 1.25, xs * 1.25, color=INK, alpha=0.07, lw=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ape = np.abs(full[1] - full[0]) / full[0]
    ax.text(0.04, 0.96, f"MAPE {100 * ape.mean():.1f}%\nmedian {100 * np.median(ape):.1f}%",
            transform=ax.transAxes, va="top", fontsize=9, color=INK,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#d0d7de"))
    ax.set_xlabel(f"measured {TITLES[k].lower()}", fontsize=9, color=INK2)
    if k == 0:
        ax.set_ylabel(f"{row_label}\n\npredicted", fontsize=9, color=INK2)
    ax.tick_params(labelsize=8, colors=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(HERE, "..", "figures", "new-workloads", "nn-v2",
                                                  "pred_vs_actual.png"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    st, sp, sc, sfull = synthetic(15000)
    vt, vp, vc, vfull, n_vpic = vpic()

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(2, 3, figsize=(12, 8.6))
    fig.patch.set_facecolor("white")
    rows = [(st, sp, sc, sfull, "Synthetic float32\n(5-fold CV, unseen patterns)"),
            (vt, vp, vc, vfull, f"Real VPIC\n({n_vpic} chunks x 45 settings)")]
    for r, (t, p, c, full, label) in enumerate(rows):
        for k in range(3):
            panel(axes[r, k], t[:, k], p[:, k], c, (full[0][:, k], full[1][:, k]), k, label)
            if r == 0:
                axes[r, k].set_title(TITLES[k], fontsize=11, color=INK, pad=8)
    handles = [Line2D([], [], marker="o", ls="", color=col, markersize=6, label=lab)
               for lab, _, col in FAMILIES]
    handles.append(Line2D([], [], color=INK, lw=1, label="perfect prediction (band: +-25 %)"))
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9,
               labelcolor=INK2, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("NeuroPress v2 (model B): predicted vs measured, one point per (chunk, setting)",
                 fontsize=12.5, color=INK, x=0.02, ha="left", y=0.985)
    fig.text(0.02, 0.937, "Top: held-out predictions from cross-validation on the synthetic "
             "training corpus (15,000-point sample shown; errors over all 441,000).\n"
             "Bottom: the saved model on real VPIC fields, never trained on.",
             fontsize=8.8, color=INK2)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.885, bottom=0.14, hspace=0.32, wspace=0.28)
    fig.savefig(a.out, dpi=170)
    fig.savefig(os.path.splitext(a.out)[0] + ".pdf")
    print("wrote", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
