#!/usr/bin/env python3
# Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
# BSD 3-Clause License.
#
# Plot the two headline GNN-on-Eternia results:
#   fig_capacity.png  -- Eternia runs GNN feature matrices that OOM an in-core GPU
#                        (peak GPU memory: traditional = dataset, Eternia = window)
#   fig_pagerank.png  -- reverse-PageRank predicts feature-page access far better
#                        than simple caching (cache hit rate vs budget)
#
# Usage: python3 plot_results.py [--csv /tmp/gnn_cap.csv] [--out figures]

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FREE_HBM_GIB = 6.86      # measured free HBM on the RTX 5060 (8 GB) at run time
GPU_TOTAL_GIB = 8.0


def load_capacity(path):
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            for r in csv.DictReader(f):
                rows.append(r)
    if not rows:  # fallback to the observed run (so the figure is reproducible)
        cols = ("features_mib nodes traditional_status eternia_store_s "
                "eternia_readout_s eternia_stored_mib eternia_ratio "
                "eternia_peak_gpu_mib bit_exact").split()
        data = [
            (4101, 10752000, "OK",  19.525, 14.640, 3795, 1.081, 39, "BIT-EXACT"),
            (6132, 16076800, "OK",  23.619, 20.027, 5675, 1.081, 39, "BIT-EXACT"),
            (8203, 21504000, "OOM", 32.376, 27.469, 7591, 1.081, 39, "n/a(OOM)"),
            (10234, 26828800, "OOM", 62.054, 66.148, 9470, 1.081, 39, "n/a(OOM)"),
        ]
        rows = [dict(zip(cols, d)) for d in data]
    rows.sort(key=lambda r: float(r["features_mib"]))
    return rows


def plot_capacity(rows, out):
    sizes = [float(r["features_mib"]) / 1024.0 for r in rows]          # GiB
    trad_ok = [r["traditional_status"] == "OK" for r in rows]
    eternia_peak = [float(r["eternia_peak_gpu_mib"]) / 1024.0 for r in rows]  # GiB
    x = range(len(sizes))
    w = 0.38

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    # Traditional in-core peak GPU need == the whole feature matrix.
    trad_bars = ax.bar([i - w / 2 for i in x], sizes, w,
                       color=["#2e7d32" if ok else "#c62828" for ok in trad_ok],
                       edgecolor="black", linewidth=0.6,
                       hatch=[None if ok else "///" for ok in trad_ok],
                       label="Traditional in-core (peak GPU = dataset)")
    # Eternia peak GPU == the streaming window (constant).
    ax.bar([i + w / 2 for i in x], eternia_peak, w, color="#1565c0",
           edgecolor="black", linewidth=0.6,
           label="Eternia (peak GPU = stream window)")

    ax.axhline(FREE_HBM_GIB, ls="--", color="black", lw=1.3)
    ax.text(len(sizes) - 0.5, FREE_HBM_GIB + 0.15,
            f"free HBM = {FREE_HBM_GIB} GiB  (OOM wall)",
            ha="right", va="bottom", fontsize=9)

    for i, (ok, s) in enumerate(zip(trad_ok, sizes)):
        if not ok:
            ax.text(i - w / 2, FREE_HBM_GIB + 0.05, "OOM✗", ha="center",
                    va="bottom", color="#c62828", fontweight="bold", fontsize=10)
        else:
            ax.text(i - w / 2, s + 0.1, "OK", ha="center", va="bottom",
                    color="#2e7d32", fontweight="bold", fontsize=9)
    for i, e in enumerate(eternia_peak):
        ax.text(i + w / 2, e + 0.1, "runs\n39 MiB", ha="center", va="bottom",
                color="#1565c0", fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{s:.1f} GiB\n({int(float(r['nodes'])/1e6)}M nodes)"
                        for s, r in zip(sizes, rows)])
    ax.set_ylabel("Peak GPU memory (GiB)")
    ax.set_xlabel("GNN node-feature matrix size (real ogbn-products, tiled)")
    ax.set_title("Eternia runs GNN feature matrices that OOM an in-core GPU\n"
                 "(lossless zstd + HBM→DRAM tiering; bounded 39 MiB stream "
                 "window; bit-exact where both run)", fontsize=11)
    ax.set_ylim(0, max(sizes) * 1.18)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    p = os.path.join(out, "fig_capacity.png")
    fig.savefig(p, dpi=150)
    print("wrote", p)


def plot_pagerank(out):
    # Observed hit rates @ node granularity, minibatch (sampled) trace, ogbn-arxiv.
    budgets = [1, 2, 5, 10, 20, 50]
    series = [
        ("reverse-PageRank (pin top-C)", [5.3, 9.1, 18.4, 30.2, 47.9, 80.7],
         "#1565c0", "o", "-"),
        ("degree (pin top-C)",           [1.6, 3.6, 10.6, 22.3, 41.7, 77.4],
         "#6a1b9a", "s", "--"),
        ("LRU",                          [0.0, 0.0, 10.3, 18.6, 36.1, 70.5],
         "#ef6c00", "^", "-."),
        ("first-C / random (naive)",     [1.0, 2.0, 5.0, 10.0, 20.0, 50.0],
         "#757575", "x", ":"),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for name, ys, c, m, ls in series:
        ax.plot(budgets, ys, ls, marker=m, color=c, lw=2, ms=6, label=name)
    ax.axhline(50, ls=":", color="black", alpha=0.5)
    ax.text(1, 51, "paper's ~50% regime", fontsize=9, va="bottom")
    ax.annotate("3× naive @ 10% budget", xy=(10, 30.2), xytext=(13, 22),
                arrowprops=dict(arrowstyle="->", color="#1565c0"),
                fontsize=9, color="#1565c0")
    ax.set_xlabel("GPU cache budget (% of feature pages resident)")
    ax.set_ylabel("Cache hit rate (%)")
    ax.set_title("reverse-PageRank predicts GNN feature-page access\n"
                 "(ogbn-arxiv, GraphSAGE minibatch sampling; each miss = one "
                 "zstd decompress + refetch)", fontsize=11)
    ax.set_xticks(budgets)
    ax.set_ylim(0, 90)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(ls=":", alpha=0.4)
    fig.tight_layout()
    p = os.path.join(out, "fig_pagerank.png")
    fig.savefig(p, dpi=150)
    print("wrote", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/tmp/gnn_cap.csv")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "figures"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    plot_capacity(load_capacity(args.csv), args.out)
    plot_pagerank(args.out)


if __name__ == "__main__":
    main()
