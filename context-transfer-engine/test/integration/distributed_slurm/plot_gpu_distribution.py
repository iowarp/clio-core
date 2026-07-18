#!/usr/bin/env python3
"""
Distributed COMPRESSED GPU vector -- per-node compressed-page footprint, real
Gray-Scott checkpoint.

One GPU node (rank 0) evolves a 2048x2048 Gray-Scott field, compresses its 64 x
256 KiB pages IN HBM (16 MiB logical) and stores them through the compressor into
a cte_core spanning 4 GPU nodes; the compressed pages fan out across all four.
Bars are allocated bytes per node.

Palette validated with the dataviz validator (green #008300 = compressed GPU).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"; INK = "#1a1a1a"; SECOND = "#52514e"; MUTED = "#8a8a85"
GRID = "#e2e2e2"; BAR = "#008300"; REF = "#2a78d6"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": SECOND, "axes.edgecolor": GRID,
    "xtick.color": SECOND, "ytick.color": SECOND,
    "font.size": 8, "svg.fonttype": "none",
})

# 4-node GPU GS run (job 20270928): allocated bytes of compressed pages per node.
nodes = ["gpua073\n(node 0,\ncompressor)", "gpua075\n(node 1)",
         "gpua076\n(node 2)", "gpua077\n(node 3)"]
kb = [868352/1024, 815104/1024, 503808/1024, 729088/1024]
total_kb = sum(kb)
uniform = total_kb / len(kb)

fig, ax = plt.subplots(figsize=(6.2, 3.7))
x = np.arange(len(nodes))
ax.bar(x, kb, 0.62, color=BAR, ec=SURFACE, lw=1.5, zorder=3)
ax.axhline(uniform, color=REF, lw=1.6, ls=(0, (4, 3)), zorder=4,
           label=f"even share ({total_kb:.0f} KiB ÷ 4 = {uniform:.0f})")
for xi, v in zip(x, kb):
    ax.text(xi, v + 4, f"{v:.0f}", ha="center", va="bottom", fontsize=9,
            color=INK, fontweight="bold")

ax.set_ylabel("compressed page bytes stored on node (KiB)", fontsize=8, color=SECOND)
ax.set_xticks(x); ax.set_xticklabels(nodes, fontsize=7.5)
ax.set_ylim(0, max(kb) * 1.28)
ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0); ax.set_axisbelow(True)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.legend(loc="upper right", fontsize=7, frameon=False)
ax.set_title("Compressed GPU vector: Gray-Scott checkpoint across 4 GPU nodes",
             fontsize=9.5, color=INK, pad=24, loc="left")
ax.text(0, 1.02,
        "A 2048x2048 Gray-Scott field (16 MiB) compressed in HBM on node 0 fans "
        "out across all 4 nodes (~5.7:1 on real sim data); read back cross-node + "
        "decompressed within the 1e-3 error bound.",
        transform=ax.transAxes, fontsize=7, color=MUTED, va="bottom")

fig.tight_layout()
fig.savefig("fig_gpu_distribution.png", dpi=150, bbox_inches="tight")
fig.savefig("fig_gpu_distribution.pdf", bbox_inches="tight")
print("wrote fig_gpu_distribution.png / .pdf")
