#!/usr/bin/env python3
"""Where does per-chunk codec choice pay? The opportunity per workload, and
what NeuroPress v2 learning captured of it, against the best single codec.

    plot_spectrum.py [--out PNG] [--pdf]

Per workload (the baseline store, /mnt/nvme0/v2-work/baselines):
  opportunity  oracle gain over the best fixed codec (oracle_analysis.py,
               4-tier mix as run): (best fixed - per-chunk best) / best fixed
  NeuroPress   its learning run (runs/<ds>_learn_nolog; exploration off),
               scored with the same exhaustive measurements: its cost against
               the best fixed codec's; and measured end to end through Clio
               (write + read wall clock) and compression ratio against the
               fixed run of the best codec (runs/<ds>_fixed_nolog).
Writes runs/spectrum.csv.
"""
import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import eval_v2_workloads as ev
import oracle_analysis as oa

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = "/mnt/nvme0/v2-work/runs"
STORE = "/mnt/nvme0/v2-work/baselines"
FIGS = os.path.join(HERE, "..", "figures", "new-workloads", "nn-v2")
# Workloads run in full (not sampled); nn-v2/ shows only these by default.
FULL = ["nyx-full", "omics-pbmc", "gnn-igbh", "analytics-tpch", "climate-era5"]
INK, INK2, GOOD, BAD, OPP = "#1f2328", "#57606a", "#1a7f37", "#cf222e", "#2e86ab"


def wall(run):
    """Measured write + read seconds."""
    t = open(os.path.join(run, "stdout.log")).read()
    w = float(re.search(r"stage\+compress ([0-9.]+) s", t).group(1))
    r = float(re.search(r"READ 1/1.*?get\+decompress: ([0-9.]+) ms", t, re.S).group(1))
    return w + r / 1e3


def ratio(run):
    """Whole-workload original / stored bytes."""
    b = pd.read_csv(os.path.join(run, "blobs.csv")).drop_duplicates("blob")
    return b.bytes.sum() / b.stored.sum()


def row(ds):
    """Opportunity and NeuroPress results for one workload."""
    ex = os.path.join(STORE, ds, "exhaustive")
    gain = [g for g in oa.analyse(ex)[0] if g["tier"].startswith("4-tier")][0]
    names, store = ev.settings_list()
    blobs, truth = ev.load_truth(ex, len(names), store)
    ok = ~np.isnan(truth).any(axis=1)
    best = names.index(gain["best_fixed"]) if gain["best_fixed"] in names else store
    le, fx = os.path.join(RUNS, f"{ds}_learn_nolog"), os.path.join(RUNS, f"{ds}_fixed_nolog")
    d = ev.run_chunks(le, store)
    idx = {b: i for i, b in enumerate(blobs)}
    rows = d.blob.map(idx).to_numpy()
    keep = ok[rows]
    np_cost = truth[rows[keep], d.stored.to_numpy()[keep]].sum()
    fixed_cost = truth[rows[keep], best].sum()
    oracle = np.nanmin(truth[rows[keep]], axis=1).sum()
    orc = os.path.join(RUNS, f"{ds}_oracle_nolog")
    has_orc = os.path.exists(os.path.join(orc, "stdout.log"))
    return {"workload": ds, "chunks": gain["chunks"], "best_fixed": gain["best_fixed"],
            "e2e_oracle_s": wall(orc) if has_orc else np.nan,
            "ratio_oracle": ratio(orc) if has_orc else np.nan,
            "opportunity_pct": gain["oracle_gain_pct"],
            "np_cost_vs_fixed_pct": 100 * (np_cost / fixed_cost - 1),
            "np_captured_pct": 100 * (fixed_cost - np_cost) / max(fixed_cost - oracle, 1e-9),
            "e2e_fixed_s": wall(fx), "e2e_np_s": wall(le),
            "ratio_fixed": ratio(fx), "ratio_np": ratio(le)}


def panel(ax, t, col, title, fmt, colour_fn):
    """Horizontal bars of one column, coloured per value."""
    y = np.arange(len(t))[::-1]
    v = t[col].to_numpy()
    ax.barh(y, v, 0.7, color=[colour_fn(x) for x in v])
    for yy, x in zip(y, v):
        ax.text(x + (0.4 if x >= 0 else -0.4), yy, fmt(x), va="center",
                ha="left" if x >= 0 else "right", fontsize=8, color=INK)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    ax.grid(axis="x", color="#e6e9ed", lw=0.6)
    ax.tick_params(labelsize=8, colors=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    lo, hi = min(0, v.min()), max(0, v.max())
    ax.set_xlim(lo - 0.25 * (hi - lo) - 1, hi + 0.25 * (hi - lo) + 1)


def pair(ax, t, cols, title, fmt, good_low):
    """Two bars per workload (oracle, NeuroPress) against the best single codec."""
    y = np.arange(len(t))[::-1]
    for k, (col, colour, lab) in enumerate(cols):
        v = t[col].to_numpy()
        ax.barh(y + (0.2 if k == 0 else -0.2), v, 0.38, color=colour, label=lab)
        for yy, x in zip(y + (0.2 if k == 0 else -0.2), v):
            if np.isfinite(x):
                ax.text(x + (0.6 if x >= 0 else -0.6), yy, fmt(x), va="center",
                        ha="left" if x >= 0 else "right", fontsize=7, color=colour)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    ax.grid(axis="x", color="#e6e9ed", lw=0.6)
    ax.tick_params(labelsize=8, colors=INK2)
    for sd in ("top", "right"):
        ax.spines[sd].set_visible(False)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right")
    vals = np.concatenate([t[c].to_numpy() for c, _, _ in cols])
    vals = vals[np.isfinite(vals)]
    lo, hi = min(0, vals.min()), max(0, vals.max())
    ax.set_xlim(lo - 0.3 * (hi - lo) - 1, hi + 0.3 * (hi - lo) + 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=None)
    ap.add_argument("--all", action="store_true",
                    help="every stored workload, sampled ones included "
                         "(written to nn-v2/archive-sampled/)")
    ap.add_argument("--pdf", action="store_true", help="also write a PDF")
    a = ap.parse_args()
    dss = sorted(os.path.basename(os.path.dirname(p))
                 for p in glob.glob(os.path.join(STORE, "*", "exhaustive")))
    dss = [d for d in dss if os.path.exists(os.path.join(RUNS, f"{d}_learn_nolog", "stdout.log"))]
    if not a.all:
        dss = [d for d in dss if d in FULL]
    a.out = a.out or (os.path.join(FIGS, "archive-sampled", "v2_spectrum.png") if a.all
                      else os.path.join(FIGS, "v2_spectrum.png"))
    t = pd.DataFrame([row(d) for d in dss]).sort_values("opportunity_pct", ascending=False)
    t["e2e_vs_fixed_pct"] = 100 * (t.e2e_np_s / t.e2e_fixed_s - 1)
    t["ratio_vs_fixed_pct"] = 100 * (t.ratio_np / t.ratio_fixed - 1)
    t["e2e_oracle_vs_fixed_pct"] = 100 * (t.e2e_oracle_s / t.e2e_fixed_s - 1)
    t["ratio_oracle_vs_fixed_pct"] = 100 * (t.ratio_oracle / t.ratio_fixed - 1)
    t.to_csv(os.path.join(RUNS, "spectrum.csv"), index=False)
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(1, 4, figsize=(20, 10), sharey=True)
    fig.patch.set_facecolor("white")
    panel(axes[0], t, "opportunity_pct",
          "Opportunity: per-chunk best vs\nbest single codec (cost, %)",
          lambda x: f"{x:.1f}%", lambda x: OPP)
    sign = lambda x: GOOD if x < 0 else BAD
    panel(axes[1], t, "np_cost_vs_fixed_pct",
          "NeuroPress learning: cost vs best\nsingle codec (%; < 0 = cheaper)",
          lambda x: f"{x:+.1f}%", sign)
    cols = [("e2e_oracle_vs_fixed_pct", OPP, "oracle (each chunk's best setting)"),
            ("e2e_vs_fixed_pct", "#d1495b", "NeuroPress learning")]
    pair(axes[2], t, cols, "Measured write + read time vs best\nsingle codec (%; < 0 = faster)",
         lambda x: f"{x:+.0f}%", True)
    cols = [("ratio_oracle_vs_fixed_pct", OPP, "oracle"),
            ("ratio_vs_fixed_pct", "#d1495b", "NeuroPress learning")]
    pair(axes[3], t, cols, "Compression ratio vs best single\ncodec (%; > 0 = smaller)",
         lambda x: f"{x:+.0f}%", False)
    axes[0].set_yticks(np.arange(len(t))[::-1])
    axes[0].set_yticklabels([f"{d}  ({n} chunks)" for d, n in zip(t.workload, t.chunks)],
                            fontsize=8.5, color=INK)
    fig.suptitle(f"Where does choosing a codec per chunk pay? {len(t)} "
                 f"{'' if a.all else 'full '}workloads through Clio",
                 x=0.01, ha="left", fontsize=14, color=INK, y=0.995)
    fig.text(0.01, 0.955, "Opportunity = how much cheaper the best setting per chunk "
             "would be than the best single codec for the whole workload (balanced "
             "4-tier cost model, from the exhaustive search: the most any per-chunk "
             "selector could gain).\nThe other panels compare NeuroPress v2 learning "
             "(exploration off), and the oracle run through Clio (each chunk stored "
             "with its own best setting, no NeuroPress work), with that best single "
             "codec: by cost, by measured end-to-end time and by compression ratio.",
             fontsize=9, color=INK2, va="top",
             linespacing=1.45)
    fig.subplots_adjust(left=0.15, right=0.99, top=0.88, bottom=0.04, wspace=0.12)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=150)
    if a.pdf:
        fig.savefig(os.path.splitext(a.out)[0] + ".pdf")
    print("wrote", os.path.abspath(a.out))
    pd.set_option("display.width", 220)
    print(t[["workload", "opportunity_pct", "np_cost_vs_fixed_pct", "e2e_oracle_vs_fixed_pct",
             "e2e_vs_fixed_pct", "ratio_oracle_vs_fixed_pct", "ratio_vs_fixed_pct"]]
          .round(1).to_string(index=False))


if __name__ == "__main__":
    main()
