#!/usr/bin/env python3
"""Compression ratio of each temporal storage strategy, per window and block
size, for one codec (temporal_compression.py output).

    plot_temporal_compression.py --results DIR --windows plt050-053 ... \
        --names Early ... --codec zstd-3+shuffle --out PNG
"""
import argparse
import os

import pandas as pd

import classify_temporal as C

STRATS = [("independent", "Each chunk alone"),
          ("dedup_index", "+ dedup, same index"),
          ("dedup_content", "+ dedup, any index"),
          ("group_time", "T0..T3 compressed together"),
          ("xor_delta", "T0 + XOR deltas")]
COLS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--windows", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--codec", default="zstd-3+shuffle")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    plt = C._plt()
    fig, axs = plt.subplots(1, len(a.windows), figsize=(4.2 * len(a.windows), 3.9),
                            sharey=False)
    w = 0.16
    for ax, win, name in zip(axs, a.windows, a.names):
        t = pd.read_csv(os.path.join(a.results, win, "compression.csv"))
        t = t[t["codec"] == a.codec]
        blocks = sorted(t["block"].unique())
        for k, ((s, lab), col) in enumerate(zip(STRATS, COLS)):
            ys = [float(t[(t["block"] == b) & (t["strategy"] == s)]["ratio"].iloc[0])
                  for b in blocks]
            xs = [i + (k - 2) * w for i in range(len(blocks))]
            ax.bar(xs, ys, w * 0.92, color=col, label=lab, edgecolor="white", linewidth=0.8)
            for x, y in zip(xs, ys):
                ax.text(x, y, f" {y:.2f}", ha="center", va="bottom", rotation=90,
                        fontsize=6.5, color=C.INK)
        ax.set_xticks(range(len(blocks)),
                      [f"{b}³\n({b ** 3 * 4 // 1024} KiB)" if b < 128 else f"{b}³\n(whole field, 8 MiB)"
                       for b in blocks])
        ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
        ax.set_title(f"{name} ({win.replace('plt', 'dumps ')})", fontsize=10, color=C.INK)
        ax.set_xlabel("Chunk size")
        C._style(ax)
    axs[0].set_ylabel(f"Compression ratio, 4 timesteps ({a.codec})")
    h, l = axs[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(h, l, frameon=False, fontsize=9, ncol=5, loc="lower center")
    fig.savefig(a.out, dpi=200, facecolor="white", bbox_inches="tight")
    print(a.out)


if __name__ == "__main__":
    main()
