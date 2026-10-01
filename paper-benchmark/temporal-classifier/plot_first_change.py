#!/usr/bin/env python3
"""When does each part of the simulation first change?

    plot_first_change.py --npz first_change.npz --out PNG [--block 16]
        [--windows 50 150 450]

The npz holds, per field, the first dump at which each value differs from
dump 0 (-1 = never). Left: share of values and of block regions that have
changed at least once, against dump (a region changes when ANY value of ANY
field in it does); the analysis windows are shaded. Right: the dump of first
change on the mid-z slice (min over fields), i.e. when the blast reached it.
"""
import argparse

import numpy as np

import classify_temporal as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--windows", nargs="*", type=int, default=[50, 150, 450])
    a = ap.parse_args()
    z = np.load(a.npz)
    big = 10 ** 6
    allf = np.stack([np.where(z[f] < 0, big, z[f]) for f in z.files])
    cell = allf.min(0)
    blk = C.to_blocks(cell, (a.block,) * 3)[0].min(1)
    last = int(max(cell[cell < big].max(), 1))
    n = max(int(allf[allf < big].max()) + 60, max(a.windows) + 20)
    dumps = np.arange(n + 1)
    vals = np.searchsorted(np.sort(allf.ravel()), dumps, side="right") / allf.size * 100
    regs = np.searchsorted(np.sort(blk), dumps, side="right") / blk.size * 100

    plt = C._plt()
    fig, (ax, axm) = plt.subplots(1, 2, figsize=(10.5, 4.0),
                                  gridspec_kw={"width_ratios": [1.5, 1]})
    for w in a.windows:
        ax.axvspan(w, w + 3, color=C.GRID, lw=0)
        ax.text(w + 1.5, 103, f"dumps {w}-{w + 3}", ha="center", fontsize=7.5, color=C.MUTED)
    ax.plot(dumps, regs, color="#2a78d6", lw=2, label=f"{a.block}³ regions (any value, any field)")
    ax.plot(dumps, vals, color="#eb6834", lw=2, label="individual values")
    k_r = int(blk.max())
    ax.annotate(f"every region changed\nby dump {k_r}", (k_r, 100), xytext=(k_r + 25, 72),
                fontsize=8, color=C.INK, arrowprops=dict(arrowstyle="-", color=C.MUTED, lw=0.8))
    last_v = int(allf[allf < big].max())
    ax.annotate(f"every value changed\nby dump {last_v}", (last_v, 100), xytext=(last_v + 25, 45),
                fontsize=8, color=C.INK, arrowprops=dict(arrowstyle="-", color=C.MUTED, lw=0.8))
    ax.set_xlim(0, n)
    ax.set_ylim(0, 108)
    ax.set_xlabel("Dump (≈ 10 simulation steps each)")
    ax.set_ylabel("% changed at least once since dump 0")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    C._style(ax, "both")

    mid = cell.shape[0] // 2
    sl = np.where(cell[mid] >= big, np.nan, cell[mid])
    im = axm.imshow(sl, cmap="Blues_r", origin="lower", vmin=0, vmax=last)
    axm.set_xticks([]), axm.set_yticks([])
    axm.set_title("Dump at which each cell first changed (mid-z slice)",
                  fontsize=9, color=C.INK)
    cb = fig.colorbar(im, ax=axm, fraction=0.046, pad=0.03)
    cb.set_label("first-change dump", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(a.out, dpi=200, facecolor="white", bbox_inches="tight")
    print(a.out)


if __name__ == "__main__":
    main()
