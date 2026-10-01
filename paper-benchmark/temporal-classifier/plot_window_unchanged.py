#!/usr/bin/env python3
"""Share of chunks byte-identical across every window of 4 consecutive dumps.

    plot_window_unchanged.py --npz changed_per_dump.npz --out PNG
        [--windows 50 150 450] [--len 4]

The npz holds b<size>: (fields, transitions, chunks) booleans, True where the
chunk changed against the previous dump. A window of `len` dumps starting at s
is unchanged for a chunk when none of its len-1 transitions changed it.
"""
import argparse

import numpy as np

import classify_temporal as C

RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]   # one hue: small -> large chunk
# Where each curve is labeled: (curve height to point at, text position in data
# coordinates). Tuned for the Nyx run; other sizes fall back to ~45% + offset.
LABEL_AT = {8: (25, (175, 40)), 16: (40, (125, 68)), 32: (50, (95, 88))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--windows", nargs="*", type=int, default=[50, 150, 450])
    ap.add_argument("--len", type=int, default=4)
    a = ap.parse_args()
    z = np.load(a.npz)
    sizes = sorted(int(k[1:]) for k in z.files)
    plt = C._plt()
    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    for w in a.windows:
        ax.axvspan(w, w + a.len - 1, color=C.GRID, lw=0)
        ax.text(w + 1.5, 103, f"dumps {w}-{w + a.len - 1}", ha="center", fontsize=7.5,
                color=C.MUTED)
    curves = {}
    for b in sizes:
        c = z[f"b{b}"]                                         # (fields, nT, chunks)
        starts = np.arange(c.shape[1] - (a.len - 2))
        curves[b] = np.array([100 * np.mean(~c[:, s:s + a.len - 1].any(1)) for s in starts])
    zero = [b for b in sizes if not curves[b].any()]
    live = [b for b in sizes if b not in zero]
    cols = ["#86b6ef", "#2a78d6", "#104281"][-len(live):] if len(live) <= 3 else RAMP
    for b, col in zip(live, cols):
        y = curves[b]
        kib = b ** 3 * 4 // 1024
        ax.plot(starts, y, color=col, lw=2.2)
        # Each label points at a different height of its curve, so labels of
        # neighbouring curves never share a spot.
        target, text_xy = LABEL_AT.get(b, (45, None))
        k = int(np.argmin(np.abs(y - target)))
        xy = (starts[k], y[k])
        txt = text_xy or (starts[k] + 25, y[k] + 10)
        ax.annotate(f"{b}³ ({kib} KiB): mean {y.mean():.1f}%", xy, xytext=txt,
                    fontsize=8, color=C.INK, va="center",
                    arrowprops=dict(arrowstyle="-", color=C.MUTED, lw=0.8))
    if zero:
        ax.plot(starts, curves[zero[0]], color=C.MUTED, lw=1.6, ls="--")
        names = " and ".join(f"{b}³" for b in zero)
        ax.text(starts[-1], 3, f"{names} (1 MiB, 8 MiB = pipeline chunk): 0% in every window",
                ha="right", va="bottom", fontsize=8, color=C.INK)
    ax.set_xlim(0, starts[-1])
    ax.set_ylim(-2, 108)
    ax.set_xlabel(f"First dump of the {a.len}-timestep window (≈ 10 simulation steps per dump)")
    ax.set_ylabel(f"% of chunks identical\nin all {a.len} timesteps")
    C._style(ax, "both")
    fig.tight_layout()
    fig.savefig(a.out, dpi=200, facecolor="white", bbox_inches="tight")
    print(a.out)


if __name__ == "__main__":
    main()
