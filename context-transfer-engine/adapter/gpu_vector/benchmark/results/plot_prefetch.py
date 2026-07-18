#!/usr/bin/env python3
"""
Reader prefetch (Transaction API, issue #700) -- IO-bound benchmark, A100.

Grouped bars: NO-PREFETCH vs SequentialTransaction wall time, per modeled
slow-tier read latency, annotated with the speedup. Shows prefetch-ahead
winning once a slow tier is in play and peaking where io ~= compute.

Palette validated with the dataviz validator (blue #2a78d6 / green #008300).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"; INK = "#1a1a1a"; SECOND = "#52514e"; MUTED = "#8a8a85"
GRID = "#e2e2e2"
NOPF = "#2a78d6"   # categorical slot 1 -- no prefetch
PF   = "#008300"   # categorical slot 4 -- prefetch (compressed GPU, matches figs)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": SECOND, "axes.edgecolor": GRID,
    "xtick.color": SECOND, "ytick.color": SECOND,
    "font.size": 8, "svg.fonttype": "none",
})

# 32 windows, ~12 ms compute/window, A100 (from test_gpu_vector_prefetch).
labels  = ["0 µs\n(real Lustre)\ncompute-bound", "3000 µs\n(io ≈ compute)", "6000 µs\nio-bound"]
no_pf   = [457.8, 817.2, 1319.8]
pf      = [435.5, 704.9, 1154.1]
speed   = [n / p for n, p in zip(no_pf, pf)]

x = np.arange(len(labels)); w = 0.38
fig, ax = plt.subplots(figsize=(6.4, 3.6))
b1 = ax.bar(x - w/2, no_pf, w, color=NOPF, ec=SURFACE, lw=1.5, zorder=3,
            label="NO-PREFETCH (read → compute, serial)")
b2 = ax.bar(x + w/2, pf, w, color=PF, ec=SURFACE, lw=1.5, zorder=3,
            label="SequentialTransaction (prefetch window w+1 while computing w)")

ax.set_ylabel("wall time to sweep 32 MiB dataset (ms)", fontsize=8, color=SECOND)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylim(0, max(no_pf) * 1.18)
ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)

for xi, n, p, s in zip(x, no_pf, pf, speed):
    ax.text(xi - w/2, n + 12, f"{n:.0f}", ha="center", va="bottom", fontsize=7, color=SECOND)
    ax.text(xi + w/2, p + 12, f"{p:.0f}", ha="center", va="bottom", fontsize=7, color=PF)
    ax.text(xi, max(n, p) * 1.0 + 55, f"{s:.2f}×", ha="center", va="bottom",
            fontsize=9, color=INK, fontweight="bold")

ax.legend(loc="upper left", fontsize=7, frameon=False)
ax.set_title("Reader prefetch (#700 Transaction API) on the compressed GPU vector — A100",
             fontsize=9.5, color=INK, pad=26, loc="left")
ax.text(0, 1.02,
        "Prefetch-ahead wins once a slow tier is modeled, peaking where io ≈ compute; "
        "at slow=0 the reader is compute-bound (compression already removed the I/O).",
        transform=ax.transAxes, fontsize=7, color=MUTED, va="bottom")

fig.tight_layout()
fig.savefig("fig_prefetch.png", dpi=150, bbox_inches="tight")
fig.savefig("fig_prefetch.pdf", bbox_inches="tight")
print("wrote fig_prefetch.png / .pdf")
