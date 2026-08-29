#!/usr/bin/env python3
"""GNN multi-GPU pooling figure: NCCL/MPI/NVSHMEM aggregate throughput on
papers100M (2 GPUs). They reach high pooled bandwidth but need 2+ GPUs (1 GPU
OOMs) and hold the matrix uncompressed; Eternia holds it on ONE GPU compressed."""
import os, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); FIG=os.path.join(HERE,"figures")
os.makedirs(FIG,exist_ok=True); LOGS="/u/rpawar/eternia_logs"
TEAL="#0E9E8E"; GREY="#6C7A89"; RED="#C0392B"; DARKE="#0B6E63"
d={}
with open(os.path.join(LOGS,"gnn_real_measured.csv")) as f:
    for r in csv.DictReader(f):
        if r["method"] in ("nccl","mpi","nvshmem") and int(r["gpus"])==2:
            d[r["method"]]=float(r["throughput_mibps"])/1024.0  # GB/s aggregate
labels=["NCCL\n(2 GPU)","CUDA-MPI\n(2 GPU)","NVSHMEM\n(2 GPU)"]
vals=[d["nccl"],d["mpi"],d["nvshmem"]]
fig,ax=plt.subplots(figsize=(7.0,4.1))
bars=ax.bar(range(len(vals)),vals,color=[GREY,GREY,TEAL],edgecolor="#222",linewidth=0.6,width=0.6)
for i,v in enumerate(vals): ax.text(i,v+0.4,f"{v:.1f}",ha="center",va="bottom",fontsize=10,fontweight="bold")
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels,fontsize=9)
ax.set_ylabel("aggregate feature throughput (GB/s)",fontsize=10)
ax.set_title("papers100M multi-GPU pooling: high bandwidth, but needs 2+ GPUs",fontsize=10.5,fontweight="bold")
ax.set_ylim(0,max(vals)*1.25)
ax.text(0.5,0.90,"1 GPU: NCCL / MPI / NVSHMEM all OOM (54 GB matrix > 40 GiB HBM); pooling holds it UNCOMPRESSED across "
        "2x40 GiB.\nEternia holds the same papers100M matrix on ONE GPU (compressed, bounded ~1.4 GiB peak).",
        transform=ax.transAxes,ha="center",va="top",fontsize=8.0,color=DARKE,
        bbox=dict(boxstyle="round,pad=0.4",fc="#F2F7F6",ec="#0E9E8E",lw=0.8))
ax.spines[["top","right"]].set_visible(False); ax.grid(axis="y",alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(FIG,"fig_update_gnn_multigpu.png"),dpi=150); plt.close()
print("wrote fig_update_gnn_multigpu.png",[round(v,1) for v in vals])
