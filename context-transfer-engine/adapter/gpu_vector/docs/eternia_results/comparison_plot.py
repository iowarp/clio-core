#!/usr/bin/env python3
"""Eternia vs GPU-memory baselines — k-means runtime per iteration on real SIFT.
Reads comparison_baselines.csv, writes figures/fig_comparison_baselines.{png,pdf}.
NOTE: UVM/NVSHMEM numbers are WSL2-distorted (see doc); trustworthy baseline
numbers require a native-Linux GPU (Delta). This figure annotates that."""
import csv, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); OUT=os.path.join(HERE,"figures"); os.makedirs(OUT,exist_ok=True)
plt.rcParams.update({"figure.dpi":130,"savefig.dpi":300,"font.size":11,"axes.spines.top":False,
  "axes.spines.right":False,"axes.grid":True,"grid.color":"#D8DEE6","grid.linewidth":0.7,
  "axes.axisbelow":True,"font.family":"DejaVu Sans"})
COL={"eternia":"#0E9E8E","traditional":"#C77A34","uvm":"#D6473D","nvshmem":"#7C6FD0"}
LAB={"eternia":"Eternia (compress+tier+stream)","traditional":"Traditional in-core (cudaMalloc)",
     "uvm":"CUDA UVM (managed, oversubscribe)","nvshmem":"NVSHMEM (1 GPU, symmetric heap)"}
rows=list(csv.DictReader(open(os.path.join(HERE,"comparison_baselines.csv"))))
series={}
for r in rows:
    t=r["technology"]
    if t not in COL: continue
    if r["status"] in ("runs",) and r["km_per_iter_s"]:
        series.setdefault(t,[]).append((float(r["dataset_gib"]),float(r["km_per_iter_s"])))
fig,ax=plt.subplots(figsize=(7.0,4.6))
HBM=7.0
ax.axvspan(HBM,13,color="#C77A34",alpha=0.06)
ax.text(12.9,2.35,"> GPU HBM",color="#9A6a2a",ha="right",va="bottom",fontsize=8.5)
for t,pts in series.items():
    pts.sort(); xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
    ax.plot(xs,ys,"-o",color=COL[t],lw=2.2,ms=6,label=LAB[t],zorder=3)
# OOM markers for traditional at 9,12
for g in (9,12): ax.plot(g,2.9*(g/3),"x",color=COL["traditional"],ms=9,mew=2,zorder=4)
ax.annotate("OOM (in-core)",(9,8.7),textcoords="offset points",xytext=(0,10),color=COL["traditional"],fontsize=8.5,ha="center")
# NVSHMEM capped marker
ax.annotate("IPC-capped ~2-3 GiB (WSL2)",(1.5,4.21),textcoords="offset points",xytext=(8,-20),
            color=COL["nvshmem"],fontsize=8.5)
# UVM slow callout
ax.annotate("~60x slower than Eternia\n(WSL2 host-resident — not\nrepresentative; needs Delta)",(3,649),
            textcoords="offset points",xytext=(28,-4),color=COL["uvm"],fontsize=8.5,va="center",
            arrowprops=dict(arrowstyle="->",color=COL["uvm"],lw=1))
ax.set_yscale("log"); ax.set_ylim(2,1100)
ax.set_xlabel("dataset size (GiB)  —  real SIFT, tiled")
ax.set_ylabel("k-means time / iteration (s, log)")
ax.set_xlim(0,13.4); ax.set_xticks([1.5,3,6,9,12])
ax.legend(loc="center right",frameon=True,framealpha=0.9,edgecolor="none",fontsize=8.6)
ax.set_title("Eternia vs GPU-memory baselines on ONE 8 GiB GPU (real SIFT k-means)",fontsize=11.5,loc="left",pad=10)
fig.subplots_adjust(bottom=0.2)
fig.text(0.02,0.015,"Only Eternia and UVM run past HBM on one GPU. Eternia is ~60x faster than UVM here "
         "(3.4x less data moved + controlled streaming vs page-fault paging).\nUVM/NVSHMEM/MPI are "
         "WSL2-distorted or IPC-blocked — trustworthy baseline numbers need a native-Linux GPU (Delta).",
         fontsize=7.8,color="#666")
fig.savefig(os.path.join(OUT,"fig_comparison_baselines.png"),bbox_inches="tight")
fig.savefig(os.path.join(OUT,"fig_comparison_baselines.pdf"),bbox_inches="tight")
print("wrote fig_comparison_baselines.png/.pdf")
