#!/usr/bin/env python3
"""One figure comparing classify_temporal.py windows of one simulation.

    plot_overview.py --results DIR --windows plt050-053 plt150-153 plt450-453 \
        --names Early Middle Late --out PNG [--field density]

(a) window class shares at the primary block size; (b) exact-dedup share vs
block edge, with the element-level ceiling; (c) cumulative share of change
magnitude vs share of blocks, hottest first; (d) the window class of one field
on its mid-z block slice, per window.
"""
import argparse
import os

import numpy as np
import pandas as pd

import classify_temporal as C

WINDOW_BLUES = ["#86b6ef", "#2a78d6", "#104281"]   # one hue, early -> late
CONSEC = C.CONSEC


def load(results, w):
    """blocks.parquet and granularity.csv of one window."""
    d = pd.read_parquet(os.path.join(results, w, "blocks.parquet"))
    xyz = d["coordinates"].str.strip("()").str.split(",", expand=True).astype(int)
    d["bz"], d["by"], d["bx"] = xyz[0], xyz[1], xyz[2]
    g = pd.read_csv(os.path.join(results, w, "granularity.csv"))
    return d, g


def lorenz(d):
    """Cumulative share of per-field-normalized change magnitude, hottest first."""
    mass = d[[f"mass_{p}" for p in CONSEC]].sum(1)
    norm = (mass / mass.groupby(d["field"]).transform("sum").replace(0, np.nan)).fillna(0)
    s = np.sort(norm.to_numpy())[::-1]
    return np.cumsum(s) / d["field"].nunique()


def panel_shares(ax, data, names, plt):
    rows = []
    for d, _ in data:
        tot = d["size_bytes"].sum()
        rows.append([100 * d.loc[d["classification"] == c, "size_bytes"].sum() / tot
                     for c in C.CLASSES])
    C._stacked(ax, rows, names)
    ax.set_xlabel("% of bytes (16³ blocks)")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=C.COLORS[c]) for c in C.CLASSES],
              labels=[C.LABEL[c] for c in C.CLASSES], frameon=False, ncol=4,
              loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8)
    ax.set_title("(a) Window class", loc="left", fontsize=10, color=C.INK, pad=22)


def panel_granularity(ax, data, names):
    for (d, g), n, col in zip(data, names, WINDOW_BLUES):
        g = g[g["mode"] == "re-derived"].sort_values("block")
        ceil = 100 * (1 - np.mean([np.average(d[f"bitfrac_{p}"], weights=d["size_bytes"])
                                   for p in CONSEC]))
        x = [1] + g["block"].tolist()
        y = [ceil] + g["exact"].tolist()
        ax.plot(x, y, color=col, lw=2, marker="o", ms=5, label=n)
        ax.annotate(f"{y[0]:.0f}%", (1, y[0]), textcoords="offset points",
                    xytext=(6, 2), fontsize=8, color=C.MUTED)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 8, 16, 32, 64], ["single\nvalue", "8³", "16³", "32³", "64³"])
    ax.set_ylim(-3, 100)
    ax.set_ylabel("% of bytes byte-identical")
    ax.set_xlabel("Granularity")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("(b) Exact dedup vs block size", loc="left", fontsize=10, color=C.INK)
    C._style(ax, "both")


def panel_concentration(ax, data, names):
    for (d, _), n, col in zip(data, names, WINDOW_BLUES):
        y = lorenz(d)
        x = 100 * np.arange(1, len(y) + 1) / len(y)
        k = int(np.ceil(0.1 * len(y))) - 1
        ax.plot(x, 100 * y, color=col, lw=2,
                label=f"{n}: top 10% hold {100 * y[k]:.0f}%")
    ax.plot([0, 100], [0, 100], color=C.MUTED, lw=0.8, ls=":")
    ax.axvline(10, color=C.GRID, lw=1)
    ax.set_xlabel("% of blocks (largest change first)")
    ax.set_ylabel("% of change magnitude")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("(c) Where the change is", loc="left", fontsize=10, color=C.INK)
    C._style(ax, "both")


def panel_maps(axs, data, names, field):
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([C.COLORS[c] for c in C.CLASSES])
    for ax, (d, _), n in zip(axs, data, names):
        g = d[d["field"] == field]
        s = g[g["bz"] == (g["bz"].max() + 1) // 2]
        lab = np.full((g["by"].max() + 1, g["bx"].max() + 1), np.nan)
        lab[s["by"], s["bx"]] = [C.CLASSES.index(c) for c in s["classification"]]
        ax.imshow(lab, cmap=cmap, vmin=-0.5, vmax=3.5, origin="lower",
                  interpolation="nearest")
        ax.set_title(n, fontsize=9, color=C.INK)
        ax.set_xticks([]), ax.set_yticks([])
    axs[0].text(0, 1.28, f"(d) {field}, mid-z slice of 16³ blocks", fontsize=10,
                color=C.INK, transform=axs[0].transAxes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--windows", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--field", default="density")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    names = [f"{n} ({w.replace('plt', 'dumps ')})" for n, w in zip(a.names, a.windows)]
    data = [load(a.results, w) for w in a.windows]
    plt = C._plt()
    fig = plt.figure(figsize=(11, 7.4))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.05], hspace=0.55, wspace=0.9)
    panel_shares(fig.add_subplot(gs[0, 0:3]), data, names, plt)
    panel_granularity(fig.add_subplot(gs[0, 3:6]), data, a.names)
    panel_concentration(fig.add_subplot(gs[1, 0:3]), data, a.names)
    sub = gs[1, 3:6].subgridspec(1, 3, wspace=0.08)
    panel_maps([fig.add_subplot(sub[0, i]) for i in range(3)], data, a.names, a.field)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=200, facecolor="white", bbox_inches="tight")
    print(a.out)


if __name__ == "__main__":
    main()
