#!/usr/bin/env python3
"""Plot the GS transparent-compressed PutBlob results (A100, 200 MB workload)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# (name, ratio, throughput MiB/s, category)  -- A100, 1024 blocks x 50 steps, 200 MB
data = [
    ("lz4",   1.00, 2085, "Lossless (CPU)"),
    ("zstd",  1.19, 2141, "Lossless (CPU)"),
    ("zfp",   2.00, 2130, "Lossy (CPU)"),
    ("cuszp", 5.69, 1994, "Lossy (GPU)"),
    ("sz3",  26.94, 2108, "Lossy (CPU)"),
]
colors = {"Lossless (CPU)": "#9aa0a6", "Lossy (CPU)": "#4285f4", "Lossy (GPU)": "#34a853"}

names   = [d[0] for d in data]
ratios  = [d[1] for d in data]
thr     = [d[2] for d in data]
cats    = [d[3] for d in data]
bar_c   = [colors[c] for c in cats]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))

# --- Panel A: compression ratio (log scale) ---
bars = ax1.bar(names, ratios, color=bar_c, edgecolor="black", linewidth=0.6, zorder=3)
ax1.set_yscale("log")
ax1.set_ylim(0.8, 40)
ax1.axhline(1.0, color="black", ls="--", lw=0.9, zorder=2)
ax1.text(4.35, 1.03, "no compression", ha="right", va="bottom", fontsize=8, color="black")
ax1.set_ylabel("Compression ratio (x)  [log]")
ax1.set_title("Compression ratio on Gray-Scott float data")
ax1.grid(axis="y", ls=":", alpha=0.5, zorder=0)
for b, r in zip(bars, ratios):
    ax1.text(b.get_x()+b.get_width()/2, r*1.06, f"{r:.2f}x", ha="center", va="bottom",
             fontsize=9, fontweight="bold")

# --- Panel B: data actually stored for a 200 MB workload ---
stored = [200.0 / r for r in ratios]
bars2 = ax2.bar(names, stored, color=bar_c, edgecolor="black", linewidth=0.6, zorder=3)
ax2.set_ylabel("Stored size (MB) for 200 MB logical")
ax2.set_title("Storage footprint after compression")
ax2.grid(axis="y", ls=":", alpha=0.5, zorder=0)
for b, s in zip(bars2, stored):
    ax2.text(b.get_x()+b.get_width()/2, s+3, f"{s:.0f}", ha="center", va="bottom", fontsize=9)

legend = [Patch(facecolor=colors[k], edgecolor="black", label=k) for k in
          ["Lossless (CPU)", "Lossy (CPU)", "Lossy (GPU)"]]
fig.legend(handles=legend, loc="upper center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, 1.02))
fig.suptitle("CLIO transparent compressed PutBlob — NVIDIA A100, same pipeline, "
             "compressor chosen by CLIO_CTE_COMPRESS_LIB", y=-0.02, fontsize=9, color="#555")
fig.tight_layout(rect=[0, 0.02, 1, 0.94])
for ext in ("png", "pdf"):
    fig.savefig(f"/u/rpawar/gsbench/fig_compression_ratio.{ext}", dpi=160, bbox_inches="tight")
print("wrote fig_compression_ratio.png / .pdf")
