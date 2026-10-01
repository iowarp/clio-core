#!/usr/bin/env python3
# Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
# All rights reserved. BSD 3-Clause license (see the repository's LICENSE).
"""Prediction error of the NeuroPress network, from upstream's cross-validation.

Reads the summary of NeuroPress/neural_net/training/cross_validate.py (5 folds
split by FILE, a fresh network per fold, MAPE in original units on the held-out
fold) and draws one horizontal bar per output: mean MAPE across folds, with the
fold-to-fold standard deviation as the whisker. Companion to shap_nn.py: same
rows (lossless, >= 64 KiB, 600k corpus), same labels, same size -- one IEEE
conference column (IPDPS: 252 pt = 3.49 in), all text 10 pt.

usage: plot_cv_mape.py --log <cv log> --out PNG
       plot_cv_mape.py --values CT DT RATIO --out PNG   (MAPE %, given, no whiskers)

--values draws numbers supplied by the caller, not computed here: the paper's
reference values 12.9 / 5.6 / 22.8 were transcribed from an earlier figure
(user, 2026-09-29) and have no reproducing run in this repository.
"""
import argparse
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

#: CV report name -> label on the figure (shap_nn.py's row names)
OUTPUTS = [("compression_time_ms", "comp. time"),
           ("decompression_time_ms", "decomp. time"),
           ("compression_ratio", "ratio")]
BLUE, INK, GRID = "#1c5cab", "#2b2b2b", "#dcdbd7"


def read_summary(path):
    """MAPE mean and std per output from the CV log's summary block.

    @param path cross_validate.py's stdout.
    @return {report name: (mean %, std %)}.
    """
    text = open(path).read()
    if "CROSS-VALIDATION SUMMARY" not in text:
        return read_folds(text, path)
    block = text.split("CROSS-VALIDATION SUMMARY", 1)[1]
    out = {}
    for name, _ in OUTPUTS:
        m = re.search(rf"^\s*{name}\s+\S+\s+\S+\s+([\d.]+)±([\d.]+)%", block, re.M)
        if not m:
            raise SystemExit(f"{path}: no summary row for {name}")
        out[name] = (float(m.group(1)), float(m.group(2)))
    return out


def read_folds(text, path):
    """MAPE mean and std over the folds that finished, for a run stopped
    before its summary (e.g. at the job's time limit). Says how many folds.

    @param text cross_validate.py's stdout.
    @param path Its file name, for messages.
    @return {report name: (mean %, std %)}.
    """
    out = {}
    for name, _ in OUTPUTS:
        v = [float(x) for x in re.findall(rf"^\s*{name}\s+MAE=.*MAPE=\s*([\d.]+)%", text, re.M)]
        if not v:
            raise SystemExit(f"{path}: no finished fold reports {name}")
        mean = sum(v) / len(v)
        std = (sum((x - mean) ** 2 for x in v) / len(v)) ** 0.5
        out[name] = (mean, std)
    print(f"note: {path} has no summary; using the {len(v)} finished fold(s)")
    return out


def main():
    """Take the MAPEs from a CV log or from --values, and draw the bars."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--log", help="cross_validate.py stdout")
    src.add_argument("--values", nargs=3, type=float, metavar=("CT", "DT", "RATIO"),
                     help="MAPE %% per output, supplied rather than computed")
    ap.add_argument("--out", required=True, help="output PNG")
    a = ap.parse_args()
    png = a.out
    s = (read_summary(a.log) if a.log else
         {n: (v, 0.0) for (n, _), v in zip(OUTPUTS, a.values)})
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Nimbus Roman", "STIXGeneral", "DejaVu Serif"]})
    fs = 10
    W, H = 3.49, 1.45
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0.90 / W, 0.42 / H, (W - 0.90 - 0.12) / W, (H - 0.42 - 0.06) / H])
    ys = range(len(OUTPUTS))[::-1]                   # first output on top
    means = [s[n][0] for n, _ in OUTPUTS]
    stds = [s[n][1] for n, _ in OUTPUTS]
    whisk = any(stds)
    ax.barh(list(ys), means, height=0.6, color=BLUE, edgecolor="none", zorder=3,
            xerr=stds if whisk else None,
            error_kw=dict(ecolor=INK, elinewidth=0.8, capsize=2.5, capthick=0.8))
    xmax = max(m + d for m, d in zip(means, stds))
    for y, m, d in zip(ys, means, stds):
        ax.text(m + d + xmax * 0.03, y, f"{m:.1f}", va="center", ha="left",
                fontsize=fs, color=INK)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([lab for _, lab in OUTPUTS], fontsize=fs, color=INK)
    ax.set_xlim(0, xmax * 1.25)
    ax.set_xlabel("MAPE (%)", fontsize=fs, color=INK, labelpad=2)
    ax.tick_params(labelsize=fs, colors=INK, length=2)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.savefig(png, dpi=300, facecolor="white")
    print({lab: f"{m:.1f} ± {d:.1f} %" for (_, lab), m, d in zip(OUTPUTS, means, stds)}, "->", png)


if __name__ == "__main__":
    main()
