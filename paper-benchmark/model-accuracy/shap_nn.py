#!/usr/bin/env python3
# Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
# All rights reserved. BSD 3-Clause license (see the repository's LICENSE).
"""Exact Shapley feature importance of the shipped NeuroPress network.

Which inputs drive the network's predictions -- the configuration it is asked
about (codec, quantization, shuffle) or the statistics of the data (size,
entropy, MAD, second derivative)? Answered with exact interventional Shapley
values on the benchmark_results_600k.csv corpus, no sampling approximation:

  scope     LOSSLESS candidates only -- the paper's comparison keeps the
            lossless codecs and byte-shuffle preprocessing, no quantization
            (user, 2026-09-28). Every row is quantize = 0 with the 1e-7
            lossless error-bound sentinel, so those two inputs are constant
            and are not players.
  players   6: the codec and the shuffle (configuration); chunk size,
            entropy, MAD and second derivative (data statistics).
  value     v(S) = mean over a background set of f(x_S, b_rest): the explained
            row's values for the players in S, a background row's for the rest.
  exact     all 2^6 = 64 coalitions per row; phi_i is the weighted sum of
            v(S + i) - v(S). Additivity (sum phi = f(x) - E_b f(b)) is asserted.
  output    the network's own output space, log1p of compression time (ms),
            decompression time (ms) and ratio, BEFORE the policy clamps (1 ms
            time floor, ratio cap 100) -- the clamps are the runtime's, not the
            model's.
  rows      explained: rows of the held-out 20% of files (upstream's split,
            train_xgb.split_files); background: rows of the training files.
            16 KiB files are dropped -- outside the network's trained input box.

usage: shap_nn.py [--corpus CSV] [--nnwt PATH] [--n 2000] [--bg 100] --out DIR --fig PNG
"""
import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from models_offline import ALGORITHMS, NeuroPressNN, nn_inputs  # noqa: E402
from prepare_inputs import MIN_TRAINED_SIZE  # noqa: E402
from train_xgb import split_files  # noqa: E402

#: player -> the network input columns it owns (nn_inputs order:
#: algo, quant, shuffle, eb, size, entropy, mad, second_deriv)
PLAYERS = [("algorithm", [0]), ("shuffle", [2]),
           ("data_size", [4]), ("entropy", [5]), ("mad", [6]), ("2nd_deriv", [7])]
N_CONFIG = 2          # the first N_CONFIG players are configuration
OUTPUTS = ["comp_ms", "decomp_ms", "comp_ratio"]
DEFAULT_NNWT = os.path.normpath(os.path.join(
    HERE, "..", "..", "context-transport-primitives", "src", "compress", "model",
    "weights", "model.nnwt"))


def load_corpus(path):
    """Lossless corpus rows inside the network's training box, with inputs.

    @param path benchmark_results_600k.csv.
    @return DataFrame of successful rows >= MIN_TRAINED_SIZE with an `x`
            column holding each row's 8 raw network inputs.
    """
    d = pd.read_csv(path, usecols=["file", "algorithm", "shuffle", "quantization",
                                   "error_bound", "original_size", "entropy", "mad",
                                   "second_derivative", "success"])
    d = d[d.success & (d.original_size >= MIN_TRAINED_SIZE)
          & (d.quantization == "none")].reset_index(drop=True)
    X = nn_inputs(d.algorithm.map(ALGORITHMS.index), (d.quantization != "none").astype(int),
                  d.shuffle, d.error_bound, d.original_size, d.entropy, d.mad,
                  d.second_derivative)
    return d, X


def model_out(nn, X):
    """The network's outputs in its own space (log1p ct, dt, ratio), unclamped.

    @param nn NeuroPressNN.
    @param X Raw inputs, rows x 8.
    @return rows x 3.
    """
    return (nn.raw_outputs(X) * nn.y_stds + nn.y_means)[:, :3]


def exact_shapley(nn, Xe, Xb):
    """Exact interventional Shapley values over the PLAYERS groups.

    @param nn NeuroPressNN.
    @param Xe Explained rows, N x 8.
    @param Xb Background rows, B x 8.
    @return (phi N x players x outputs, f(x) N x outputs, E_b f(b) outputs).
    """
    n, N, B = len(PLAYERS), len(Xe), len(Xb)
    v = {}
    for mask in range(1 << n):
        cols = [c for i, (_, cs) in enumerate(PLAYERS) if mask >> i & 1 for c in cs]
        X = np.repeat(Xb[None, :, :], N, axis=0)            # N x B x 8
        if cols:
            X[:, :, cols] = Xe[:, None, cols]
        v[mask] = model_out(nn, X.reshape(N * B, -1)).reshape(N, B, -1).mean(axis=1)
    phi = np.zeros((N, n, len(OUTPUTS)))
    for i in range(n):
        for mask in range(1 << n):
            if mask >> i & 1:
                continue
            s = bin(mask).count("1")
            w = math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)
            phi[:, i] += w * (v[mask | 1 << i] - v[mask])
    return phi, v[(1 << n) - 1], v[0].mean(axis=0)


def draw(mean_abs, path):
    """The heatmap, horizontal: outputs as rows, the 6 inputs as columns.

    One IEEE conference column (IPDPS: IEEEtran, 252 pt = 3.49 in), all text
    10 pt. "config." and "data statistics" label the column groups above the
    cells, split by a rule; the colour bar sits at the right.

    @param mean_abs players x outputs array.
    @param path Output PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix"})
    fs, ink = 10, "#2b2b2b"
    m = mean_abs.T                                   # outputs x players
    names = ["algorithm", "shuffle", "size", "entropy", "MAD", "2nd deriv."]
    rows = ["comp. time", "decomp. time", "ratio"]
    W, H = 3.49, 1.95
    # inches: row labels 0.90 | cells | gap 0.06 | bar 0.08 | ticks+label 0.42
    x0, bw = 0.90, 0.08
    cw = W - x0 - 0.06 - bw - 0.42
    y0, top = 0.62, 0.24                             # rotated names below, groups above
    ch = H - y0 - top
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([x0 / W, y0 / H, cw / W, ch / H])
    cax = fig.add_axes([(x0 + cw + 0.06) / W, y0 / H, bw / W, ch / H])
    vmax = math.ceil(m.max() * 10) / 10
    im = ax.imshow(m, cmap="YlOrRd", vmin=0, vmax=vmax, aspect="auto")
    for (r, c), val in np.ndenumerate(m):
        ax.text(c, r, f"{val:.2f}", ha="center", va="center", fontsize=fs,
                color="white" if val > 0.6 * vmax else ink)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=fs, color=ink, rotation=35, ha="right",
                       rotation_mode="anchor")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=fs, color=ink)
    ax.tick_params(length=0, pad=3)
    ax.set_xticks(np.arange(-0.5, len(names)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    # configuration left of the rule, data statistics right of it
    ax.axvline(N_CONFIG - 0.5, color=ink, linewidth=1.2)
    for lo, hi, txt in ((0, N_CONFIG - 1, "config."),
                        (N_CONFIG, len(PLAYERS) - 1, "data statistics")):
        ax.annotate(txt, xy=((lo + hi) / 2, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(0, 3), textcoords="offset points", ha="center",
                    va="bottom", fontsize=fs, color=ink)
    cb = fig.colorbar(im, cax=cax, ticks=[0, vmax / 2, vmax])
    cb.set_label("Mean |SHAP|", fontsize=fs, color=ink, labelpad=2)
    cb.ax.tick_params(labelsize=fs, length=2, pad=2)
    cb.outline.set_linewidth(0.6)
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)


def main():
    """Load, split, explain, check, save, draw."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", default=os.path.join(HERE, "..", "benchmark_results_600k.csv"))
    ap.add_argument("--nnwt", default=DEFAULT_NNWT)
    ap.add_argument("--n", type=int, default=2000, help="explained rows")
    ap.add_argument("--bg", type=int, default=100, help="background rows")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="directory for the SHAP CSVs")
    ap.add_argument("--fig", required=True, help="output PNG")
    a = ap.parse_args()
    nn = NeuroPressNN(a.nnwt)
    d, X = load_corpus(a.corpus)
    train, val = split_files(d)
    rng = np.random.default_rng(a.seed)
    ie = rng.choice(np.flatnonzero(d.file.isin(val).to_numpy()), a.n, replace=False)
    ib = rng.choice(np.flatnonzero(d.file.isin(train).to_numpy()), a.bg, replace=False)
    phi, fx, base = exact_shapley(nn, X[ie], X[ib])
    gap = np.abs(phi.sum(axis=1) - (fx - base)).max()
    assert gap < 1e-8, f"additivity violated by {gap}"
    mean_abs = np.abs(phi).mean(axis=0)
    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(mean_abs, index=[p for p, _ in PLAYERS], columns=OUTPUTS).to_csv(
        os.path.join(a.out, "mean_abs_shap.csv"), float_format="%.6f")
    rows = [dict(row=int(r), player=PLAYERS[i][0], output=OUTPUTS[k], shap=phi[j, i, k])
            for j, r in enumerate(ie) for i in range(len(PLAYERS)) for k in range(len(OUTPUTS))]
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "shap_values.csv.gz"), index=False)
    draw(mean_abs, a.fig)
    print(f"{a.n} explained x {a.bg} background lossless rows, {1 << len(PLAYERS)} "
          f"coalitions; additivity gap {gap:.1e}")
    print(pd.DataFrame(mean_abs, index=[p for p, _ in PLAYERS], columns=OUTPUTS).round(3))
    print(f"-> {a.fig}")


if __name__ == "__main__":
    main()
