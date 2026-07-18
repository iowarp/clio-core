#!/usr/bin/env python3
"""
Distributed CTE coherency test -- per-node data distribution.

This is NOT a with/without performance comparison (coherency is a correctness
property, not a speedup). It visualizes the ONE quantitative result: blob data
hash-routes across all 4 nodes rather than collapsing to the client's local
container -- the direct evidence that overturns issue #503's premise.

Palette validated with the dataviz validator (blue #2a78d6 / green #008300).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"; INK = "#1a1a1a"; SECOND = "#52514e"; MUTED = "#8a8a85"
GRID = "#e2e2e2"; BAR = "#2a78d6"; REF = "#008300"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": SECOND, "axes.edgecolor": GRID,
    "xtick.color": SECOND, "ytick.color": SECOND,
    "font.size": 8, "svg.fonttype": "none",
})

# Single-client run (job 20253774): 16 x 4 KB blobs, allocated-block accounting.
nodes  = ["cn008\n(node 0)", "cn010\n(node 1)", "cn012\n(node 2)", "cn014\n(node 3)"]
blobs  = [1, 7, 3, 5]                       # blobs landed per node (sums to 16)
uniform = sum(blobs) / len(blobs)           # = 4.0 expected

fig, ax = plt.subplots(figsize=(6.0, 3.6))
x = np.arange(len(nodes))
b = ax.bar(x, blobs, 0.62, color=BAR, ec=SURFACE, lw=1.5, zorder=3)
ax.axhline(uniform, color=REF, lw=1.6, ls=(0, (4, 3)), zorder=4,
           label=f"uniform expectation (16 ÷ 4 = {uniform:.0f})")

for xi, v in zip(x, blobs):
    ax.text(xi, v + 0.12, str(v), ha="center", va="bottom", fontsize=9,
            color=INK, fontweight="bold")

ax.set_ylabel("4 KB blobs stored on node", fontsize=8, color=SECOND)
ax.set_xticks(x); ax.set_xticklabels(nodes, fontsize=7.5)
ax.set_ylim(0, max(blobs) * 1.3)
ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0); ax.set_axisbelow(True)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.legend(loc="upper right", fontsize=7, frameon=False)
ax.set_title("CTE blob data distributes across 4 real nodes (hash routing)",
             fontsize=9.5, color=INK, pad=24, loc="left")
ax.text(0, 1.02,
        "16 blobs written from one client fan out 1/7/3/5 across nodes — proving "
        "cross-node routing works. #503's assertion metric (completer_) is broken, "
        "not the routing.",
        transform=ax.transAxes, fontsize=7, color=MUTED, va="bottom", wrap=True)

fig.tight_layout()
fig.savefig("fig_distribution.png", dpi=150, bbox_inches="tight")
fig.savefig("fig_distribution.pdf", bbox_inches="tight")
print("wrote fig_distribution.png / .pdf")
