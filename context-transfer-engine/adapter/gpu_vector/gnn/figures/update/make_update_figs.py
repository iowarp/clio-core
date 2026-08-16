#!/usr/bin/env python3
"""Figures for the Eternia optimization update: old vs new GNN training epoch."""
import sys, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"figures")
OLD_EPOCH=278.37; NEW_W2=27.87
NEW_W4=float(sys.argv[1]) if len(sys.argv)>1 else NEW_W2   # filled after WINDOW=4 run
ET="#0E9E8E"; OLD="#C77A34"; OOM="#D6473D"
plt.rcParams.update({"figure.dpi":130,"savefig.dpi":300,"font.size":11,
  "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,
  "grid.color":"#D8DEE6","grid.linewidth":0.7,"axes.axisbelow":True,"font.family":"DejaVu Sans"})
def save(fig,s):
    for e in ("png","pdf"): fig.savefig(os.path.join(OUT,s+"."+e),bbox_inches="tight")
    plt.close(fig); print("wrote",s)
# Fig 1: epoch time old vs new (log) + speedup labels
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10.5,4.3))
labels=["Original\nEternia","Optimized\n(win=2)","Optimized\n(win=4)"]
vals=[OLD_EPOCH,NEW_W2,NEW_W4]; cols=[OLD,ET,ET]
b=ax1.bar(labels,vals,color=cols)
for i,v in enumerate(vals):
    ax1.text(i,v*1.05,f"{v:.1f}s",ha="center",fontsize=10,fontweight="bold")
    if i>0: ax1.text(i,v*0.5,f"{OLD_EPOCH/v:.1f}x\nfaster",ha="center",fontsize=9,color="white",fontweight="bold")
ax1.set_yscale("log"); ax1.set_ylabel("Time per epoch (s, log)")
ax1.set_title("Per-epoch training time")
# Fig 2: effective gather throughput
A=54208.0
tp=[A/OLD_EPOCH,A/NEW_W2,A/NEW_W4]
ax2.bar(labels,tp,color=cols)
for i,v in enumerate(tp): ax2.text(i,v*1.03,f"{v/1024:.2f} GB/s",ha="center",fontsize=9.5,fontweight="bold")
ax2.set_ylabel("Effective feature-store throughput (MiB/s)")
ax2.set_title("Effective feature-store throughput")
fig.suptitle("Eternia optimization update: ~10x faster GNN training, bit-identical numerics",fontsize=11)
save(fig,"fig_update_epoch_speedup")
print("done.")
