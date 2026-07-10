#!/usr/bin/env python3
"""Charts for the compressed GPU vector results:
   fig_checkpoint_compare  -- traditional vs compressed checkpoint (latency, footprint)
   fig_capacity            -- dataset vs GPU-memory budget
Palette validated with the dataviz validator (blue #2a78d6 / green #008300,
CVD deltaE 104 deutan, all checks PASS)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
HERE = os.path.dirname(os.path.abspath(__file__))

SURFACE = "#fcfcfb"
INK = "#1a1a1a"
SECOND = "#52514e"
MUTED = "#8a8a85"
GRID = "#e2e2e2"

TRAD = "#2a78d6"        # categorical slot 1 -- traditional path
TRAD_LT = "#a8c9ee"     # lighter tint of the SAME hue (phase breakdown = sequential)
COMP = "#008300"        # categorical slot 4 -- compressed GPU (matches earlier figs)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": SECOND, "axes.edgecolor": GRID,
    "xtick.color": SECOND, "ytick.color": SECOND,
})


def style(ax, ylabel):
    ax.set_ylabel(ylabel, fontsize=8, color=SECOND)
    ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)


# measured (clio_gs_checkpoint_bench, A100, 4096x4096 = 64 MiB field, 8 ckpts)
D2H_MS, PFS_MS, COMP_MS = 3.06, 93.66, 14.62
TRAD_MS = D2H_MS + PFS_MS
TRAD_MIB, COMP_MIB = 512.0, 68.2

BW = 0.34  # thin marks

# ---------------------------------------------------------------- figure 1
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.15))
fig.subplots_adjust(wspace=0.34, top=0.76, bottom=0.19, left=0.085, right=0.985)

# panel A: per-checkpoint latency; traditional split by phase (one hue, light->dark)
x = [0, 1]
a1.bar(x[0], D2H_MS, BW, color=TRAD_LT, ec=SURFACE, lw=2, zorder=3,
       label="D2H copy")
a1.bar(x[0], PFS_MS, BW, bottom=D2H_MS, color=TRAD, ec=SURFACE, lw=2, zorder=3,
       label="PFS file write")
a1.bar(x[1], COMP_MS, BW, color=COMP, ec=SURFACE, lw=2, zorder=3,
       label="GPU compress → HBM")
a1.text(x[0], TRAD_MS + 3, f"{TRAD_MS:.1f} ms", ha="center", fontsize=8.5,
        color=INK, fontweight="bold")
a1.text(x[1], COMP_MS + 3, f"{COMP_MS:.1f} ms", ha="center", fontsize=8.5,
        color=INK, fontweight="bold")
a1.text(1, TRAD_MS * 0.55, "6.6× faster", ha="center", fontsize=8.5, color=COMP,
        fontweight="bold")
a1.set_xticks(x)
a1.set_xticklabels(["Traditional\n(D2H + PFS file)", "Compressed\nGPU vector"])
a1.set_xlim(-0.55, 1.55)
a1.set_ylim(0, TRAD_MS * 1.42)
a1.set_title("Checkpoint latency — lower is better", fontsize=8.8, color=INK,
             pad=6, loc="left")
style(a1, "ms per checkpoint")
a1.legend(frameon=False, fontsize=7.0, loc="upper center", ncol=1,
          handlelength=1.0, labelspacing=0.25, borderpad=0.1,
          bbox_to_anchor=(0.62, 1.02))

# panel B: footprint written, and where it lives (2 bars, identity via labels)
a2.bar(x[0], TRAD_MIB, BW, color=TRAD, ec=SURFACE, lw=2, zorder=3)
a2.bar(x[1], COMP_MIB, BW, color=COMP, ec=SURFACE, lw=2, zorder=3)
a2.text(x[0], TRAD_MIB + 14, "512 MiB\non disk", ha="center", fontsize=8.2,
        color=INK, fontweight="bold")
a2.text(x[1], COMP_MIB + 14, "68 MiB\nin HBM", ha="center", fontsize=8.2,
        color=INK, fontweight="bold")
a2.text(1, TRAD_MIB * 0.52, "7.5× smaller", ha="center", fontsize=8.5, color=COMP,
        fontweight="bold")
a2.set_xticks(x)
a2.set_xticklabels(["Traditional\n(off-GPU)", "Compressed\n(on-GPU)"])
a2.set_xlim(-0.55, 1.55)
a2.set_ylim(0, TRAD_MIB * 1.34)
a2.set_title("Checkpoint footprint (8 checkpoints)", fontsize=8.8, color=INK,
             pad=6, loc="left")
style(a2, "MiB written")

fig.text(0.085, 0.955, "Gray-Scott checkpointing: traditional storage path vs. "
                       "compressed GPU vector",
         fontsize=9.4, color=INK, fontweight="bold")
fig.text(0.085, 0.902, "A100 · cuSZp · 4096×4096 field (64 MiB) · 200 steps · "
                       "checkpoint every 25 · clio_gs_checkpoint_bench",
         fontsize=6.9, color=MUTED)
for ext in ("png", "pdf"):
    fig.savefig(f"{HERE}/fig_checkpoint_compare.{ext}", dpi=200,
                facecolor=SURFACE)
plt.close(fig)

# ---------------------------------------------------------------- figure 2
fig, ax = plt.subplots(figsize=(5.4, 3.15))
fig.subplots_adjust(top=0.83, bottom=0.19, left=0.145, right=0.975)
LOGICAL, BUDGET, USED = 256.0, 128.0, 16.0
ax.bar(0, LOGICAL, BW, color=TRAD, ec=SURFACE, lw=2, zorder=3)
ax.bar(1, USED, BW, color=COMP, ec=SURFACE, lw=2, zorder=3)
ax.axhline(BUDGET, color=MUTED, lw=1.4, ls=(0, (5, 3)), zorder=4)
# Threshold label sits over the empty right half (the green bar is far below it),
# so it never crosses a mark.
ax.text(1.5, BUDGET + 7, "GPU memory budget (128 MiB)", ha="right", fontsize=7.3,
        color=SECOND)
ax.text(0, LOGICAL + 9, "256 MiB\ndoes NOT fit", ha="center", fontsize=8.2,
        color=INK, fontweight="bold")
ax.text(1, USED + 9, "16 MiB\nfits on GPU", ha="center", fontsize=8.2,
        color=INK, fontweight="bold")
ax.set_xticks([0, 1])
ax.set_xticklabels(["Uncompressed", "Compressed (15.9×)"])
ax.set_xlim(-0.55, 1.55)
ax.set_ylim(0, LOGICAL * 1.30)
style(ax, "HBM footprint (MiB)")
fig.text(0.145, 0.965, "Capacity: a 2×-budget dataset fits on the GPU",
         fontsize=9.4, color=INK, fontweight="bold")
fig.text(0.145, 0.915, "256 MiB Gray-Scott dataset · 128 MiB HBM budget · "
                       "cte_gpu_vector_capacity_cuda",
         fontsize=6.9, color=MUTED)
for ext in ("png", "pdf"):
    fig.savefig(f"{HERE}/fig_capacity.{ext}", dpi=200, facecolor=SURFACE)
plt.close(fig)
print("wrote fig_checkpoint_compare.{png,pdf} fig_capacity.{png,pdf}")
