#!/usr/bin/env python3
"""Figures for the GS transparent-compressed PutBlob evaluation (NVIDIA A100).

Fig 1: compression ratio + storage footprint per compressor (200 MB workload).
Fig 2: wall-time vs no-compression baseline (3 reps) -> shows no slowdown.

Palette: validated categorical hues (dataviz skill reference theme) —
  lossless=blue #2a78d6, lossy-CPU=yellow #eda100, lossy-GPU=green #008300;
  no-compression baseline = neutral gray #9aa0a6.
Text stays in ink tokens; color carries category only.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

INK   = "#0b0b0b"; INK2 = "#52514e"; GRID = "#e6e6e3"
LOSSLESS = "#2a78d6"; LOSSY_CPU = "#eda100"; LOSSY_GPU = "#008300"; NONE = "#9aa0a6"
CAT = {"Lossless (CPU)": LOSSLESS, "Lossy (CPU)": LOSSY_CPU,
       "Lossy (GPU)": LOSSY_GPU, "No compression": NONE}

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
})

def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.9, zorder=0)
    ax.set_axisbelow(True)

# ---------------------------------------------------------------- Fig 1
# (name, ratio, category) — A100, 1024 blocks x 50 steps, 200 MB
d1 = [("lz4",1.00,"Lossless (CPU)"), ("zstd",1.19,"Lossless (CPU)"),
      ("zfp",2.00,"Lossy (CPU)"), ("cuszp",5.69,"Lossy (GPU)"), ("sz3",26.94,"Lossy (CPU)")]
n1  = [x[0] for x in d1]; r1 = [x[1] for x in d1]; c1 = [CAT[x[2]] for x in d1]

fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.3))
bars = a.bar(n1, r1, color=c1, width=0.68, edgecolor="white", lw=0.8, zorder=3)
a.set_yscale("log"); a.set_ylim(0.8, 40)
a.axhline(1.0, color=INK2, ls="--", lw=0.9, zorder=2)
a.text(4.4, 1.03, "no compression", ha="right", va="bottom", fontsize=8.5, color=INK2)
a.set_ylabel("Compression ratio  (log)"); a.set_title("Compression ratio on Gray-Scott float data", color=INK)
style(a)
for bar, r in zip(bars, r1):
    a.text(bar.get_x()+bar.get_width()/2, r*1.07, f"{r:.2f}x", ha="center", va="bottom",
           fontsize=9.5, fontweight="bold", color=INK)

stored = [200.0/r for r in r1]
bars2 = b.bar(n1, stored, color=c1, width=0.68, edgecolor="white", lw=0.8, zorder=3)
b.set_ylabel("Stored size (MB) for 200 MB logical"); b.set_title("Storage footprint after compression", color=INK)
b.set_ylim(0, 215); style(b)
for bar, s in zip(bars2, stored):
    b.text(bar.get_x()+bar.get_width()/2, s+4, f"{s:.0f}", ha="center", va="bottom", fontsize=9.5, color=INK)

fig.legend(handles=[Patch(facecolor=CAT[k], edgecolor="white", label=k)
                    for k in ["Lossless (CPU)","Lossy (CPU)","Lossy (GPU)"]],
           loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03))
fig.suptitle("CLIO transparent compressed PutBlob (A100) — same pipeline, compressor set by CLIO_CTE_COMPRESS_LIB",
             y=-0.02, fontsize=9, color=INK2)
fig.tight_layout(rect=[0, 0.02, 1, 0.93])
for e in ("png","pdf"): fig.savefig(f"/u/rpawar/gsbench/fig_compression_ratio.{e}", dpi=160, bbox_inches="tight")

# ---------------------------------------------------------------- Fig 2 (slowdown)
# per-compressor total wall-time (ms), 3 reps; (name, reps, category, ratio)
d2 = [
    ("none",  [93.821,93.756,93.110], "No compression", 1.00),
    ("lz4",   [92.364,93.649,94.675], "Lossless (CPU)", 1.00),
    ("zstd",  [102.484,94.317,94.309],"Lossless (CPU)", 1.19),
    ("zfp",   [94.624,96.929,93.261], "Lossy (CPU)",    2.00),
    ("cuszp", [85.918,98.694,94.712], "Lossy (GPU)",    5.69),
    ("sz3",   [95.741,93.173,96.558], "Lossy (CPU)",   26.94),
]
def med(x): s=sorted(x); return s[len(s)//2]
n2  = [x[0] for x in d2]
meds= [med(x[1]) for x in d2]
lo  = [med(x[1])-min(x[1]) for x in d2]
hi  = [max(x[1])-med(x[1]) for x in d2]
c2  = [CAT[x[2]] for x in d2]
rat = [x[3] for x in d2]
base = meds[0]  # none median

figS, ax = plt.subplots(figsize=(8.2, 4.6))
# noise band around the baseline (min..max of the 'none' reps)
ax.axhspan(min(d2[0][1]), max(d2[0][1]), color=NONE, alpha=0.22, zorder=0)
ax.axhline(base, color=INK2, ls="--", lw=1.0, zorder=2)
ax.text(5.42, base+0.4, "no-compression baseline", ha="right", va="bottom", fontsize=8.6, color=INK2)
bars3 = ax.bar(n2, meds, yerr=[lo, hi], color=c2, width=0.66, edgecolor="white", lw=0.8,
               error_kw=dict(ecolor=INK2, elinewidth=1.1, capsize=4), zorder=3)
ax.set_ylim(0, 112)
ax.set_ylabel("Wall-clock time (ms)  — lower is better")
ax.set_title("Compression adds no measurable slowdown", color=INK)
style(ax)
# annotate compression ratio above each bar (the payoff, at ~same time)
for bar, m, h, r in zip(bars3, meds, hi, rat):
    lbl = "raw" if r == 1.00 and bar is bars3[0] else f"{r:.2g}x smaller" if r>1.01 else "1.0x"
    ax.text(bar.get_x()+bar.get_width()/2, m+h+2.0, lbl, ha="center", va="bottom",
            fontsize=8.8, color=INK, fontweight="bold" if r>2 else "normal")
ax.text(bar.get_x()+bar.get_width()/2, 0, "", )  # noop
figS.legend(handles=[Patch(facecolor=CAT[k], edgecolor="white", label=k)
                     for k in ["No compression","Lossless (CPU)","Lossy (CPU)","Lossy (GPU)"]],
            loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.04))
figS.suptitle("A100, 200 MB workload (1024 blocks x 50 steps), median of 3 reps; label = compression achieved",
              y=-0.02, fontsize=8.6, color=INK2)
figS.tight_layout(rect=[0, 0.02, 1, 0.93])
for e in ("png","pdf"): figS.savefig(f"/u/rpawar/gsbench/fig_slowdown.{e}", dpi=160, bbox_inches="tight")
print("wrote fig_compression_ratio.{png,pdf} and fig_slowdown.{png,pdf}")
