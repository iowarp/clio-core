#!/usr/bin/env python3
"""Ratio and data quality at one error bound, every workload in one figure:
uniform lossy vs NeuroPress vs NeuroPress with its quality floor, all under
the RATIO cost model.

Two stacked panels, one group of bars per workload, the value on each:
  top     stored compression ratio (median of the measured runs)
  bottom  lowest PSNR of any field at any timestep: per dump file, MSE against
          the file's own value range (lossless files infinite), minimum over all files

  Uniform lossy     every chunk quantized with the workload's highest-ratio
                    fixed nvCOMP configuration (from its trace)
  relative bound    the same, each chunk bounded by 1% of its own value range
  NN + ... + floor  NeuroPress, learn-ratio, with CLIO_NEUROPRESS_TARGET_PSNR=40
  NN                NeuroPress, dynamic-ratio: the model's predictions, frozen
  NN + online learning  NeuroPress, learn-ratio (objective: bytes saved)

A workload appears once its runs exist. Data: results/selective/
measured_runs.csv (consumer/selective_analyze.py).

Usage:
  ./plot_simple.py [--workloads vpic,nyx,lammps,warpx] [--eb 1e-3] [--out DIR]
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# PAPER STYLE -- figures/fig9/plot_choices.py's EuroSys look: serif, >= 10 pt
# at the width drawn, greyscale-safe (hatch + legend carry identity).
SERIF = ["Nimbus Roman", "STIXGeneral", "DejaVu Serif", "serif"]
FS = 10
INK, INK_MUTED, GRID = "#2b2b2b", "#6b6a66", "#dcdbd7"
BLUE, GREEN, DARK_GREEN = "#1c5cab", "#31aa76", "#006435"   # figure 9's arm colours

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(os.path.dirname(HERE))
#: bar -> (measured arm, legend label, colour, hatch)
BARS = [("uniform_zstd", "Uniform lossy", BLUE, ""),
        ("np_lossless", "NeuroPress, lossless only", GREEN, ""),
        ("np_ratiofloor40", "NeuroPress, lossy accepted", DARK_GREEN, "//")]
NAMES = {"vpic": "VPIC", "nyx": "Nyx", "lammps": "LAMMPS", "warpx": "WarpX"}
PSNR_CAP = 120.0
REL_EPS = 1e-2     # relative baseline: each chunk bounded by 1% of its value range   # a field stored losslessly everywhere has no finite PSNR


def table(runs: pd.DataFrame, wls: list, eb: float) -> pd.DataFrame:
    """Median ratio and worst-field PSNR per workload and bar."""
    rel = runs.get("rel", pd.Series(0.0, index=runs.index)).fillna(0)
    at_eb = ((runs.eb - eb).abs() < eb * 1e-6) & (rel == 0)
    at_rel = (runs.arm == "uniform_rel") & ((rel - REL_EPS).abs() < REL_EPS * 1e-6)
    r = runs[(at_eb | at_rel) & runs.arm.isin([b[0] for b in BARS])]
    g = r.groupby(["workload", "arm"]).agg(ratio=("ratio", "median"), ssim=("ssim", "median"),
                                           psnr=("worst_psnr", "median"))
    g["psnr"] = g.psnr.fillna(PSNR_CAP).clip(upper=PSNR_CAP)
    return g.reindex([(w, b[0]) for w in wls for b in BARS]).dropna(how="all")


def panel(ax, t: pd.DataFrame, wls: list, col: str, fmt, ylabel: str) -> None:
    """One metric: a group of bars per workload, the value above each bar."""
    w = 0.8 / len(BARS)
    top = np.nanmax(t[col].to_numpy())
    for i, (arm, _, color, hatch) in enumerate(BARS):
        for j, wl in enumerate(wls):
            if (wl, arm) not in t.index or np.isnan(t.loc[(wl, arm), col]):
                continue
            v = t.loc[(wl, arm), col]
            x = j + (i - (len(BARS) - 1) / 2) * w
            ax.bar(x, v, width=w * 0.92, color=color, hatch=hatch, edgecolor="white",
                   linewidth=0, zorder=3)
            ax.text(x, v + top * 0.03, fmt(v), ha="center", va="bottom", fontsize=FS, color=INK,
                    rotation=90)
    ax.set_xlim(-0.5, max(len(wls), 4) - 0.5)   # bars keep their width while workloads fill in
    ax.set_xticks(range(len(wls)))
    ax.set_xticklabels([NAMES.get(w, w) for w in wls], fontsize=FS, color=INK)
    ax.set_ylim(0, top * 1.6)   # room for the vertical value labels
    ax.set_ylabel(ylabel, fontsize=FS, color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.tick_params(axis="y", labelsize=FS, colors=INK_MUTED, width=0.6)
    ax.tick_params(axis="x", length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(BENCH, "results", "selective", "measured_runs.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--workloads", default="vpic,nyx,lammps,warpx")
    ap.add_argument("--eb", type=float, default=1e-3)
    ap.add_argument("--width", type=float, default=7.0)
    ap.add_argument("--height", type=float, default=3.8)
    a = ap.parse_args()
    mpl.rcParams.update({"font.family": "serif", "font.serif": SERIF, "font.size": FS,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.8,
                         "hatch.color": "white", "hatch.linewidth": 1.0})
    runs = pd.read_csv(a.runs)
    t = table(runs, a.workloads.split(","), a.eb)
    wls = [w for w in a.workloads.split(",") if w in t.index.get_level_values(0)]
    print(t.to_string(float_format=lambda v: f"{v:.2f}"))
    fig, (top, bot) = plt.subplots(2, 1, figsize=(a.width, a.height), sharex=True)
    panel(top, t, wls, "ratio", lambda v: f"{v:.1f}×", "Compression\nratio")
    panel(bot, t, wls, "psnr", lambda v: "lossless" if v >= PSNR_CAP else f"{v:.0f}",
          "Lowest block\nPSNR (dB)")
    fig.legend([Patch(facecolor=c, hatch=h, edgecolor="white", linewidth=0) for _, _, c, h in BARS],
               [b[1] for b in BARS], loc="upper center", ncol=3, fontsize=FS, frameon=False,
               bbox_to_anchor=(0.5, 1.0), handlelength=1.6, columnspacing=1.6)
    fig.tight_layout(pad=0.3, h_pad=0.6, rect=(0, 0, 1, 0.93))
    small = [t_.get_fontsize() for t_ in fig.findobj(mpl.text.Text)
             if t_.get_text().strip() and t_.get_fontsize() < FS]
    if small or a.width > 7.001:
        raise SystemExit("figure check failed: text below 10 pt or wider than 7 in")
    os.makedirs(a.out, exist_ok=True)
    stem = os.path.join(a.out, f"ratio_psnr_eb{a.eb:g}")
    fig.savefig(stem + ".pdf", facecolor="white")
    fig.savefig(stem + ".png", dpi=300, facecolor="white")
    print(f"-> {stem}.pdf/.png ({a.width} x {a.height} in; place at \\textwidth)")


if __name__ == "__main__":
    main()
