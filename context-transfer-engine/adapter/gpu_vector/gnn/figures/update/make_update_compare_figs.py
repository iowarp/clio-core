#!/usr/bin/env python3
"""Comparison figures for the update report: Eternia (updated) vs GPU-memory-
expansion competitors on the papers100M GNN feature-store, single A100.
Reads measured CSVs so a fresh competitor re-run refreshes the plots."""
import os, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE=os.path.dirname(os.path.abspath(__file__)); FIG=os.path.join(HERE,"figures")
os.makedirs(FIG,exist_ok=True)
LOGS="/u/rpawar/eternia_logs"
FEAT_MIB=54208.0  # papers100M aggregated features (111M x 128 x f32)

TEAL="#0E9E8E"; GREY="#9AA3A2"; DARKE="#0B6E63"; RED="#C0392B"; AMBER="#E0A800"

def load_competitors():
    d={}
    with open(os.path.join(LOGS,"gnn_real_measured.csv")) as f:
        for r in csv.DictReader(f):
            d[r["method"]]={"ran":r["ran"],"peak":float(r["peak_gpu_mib"]),
                            "tput":float(r["throughput_mibps"]),"gpus":int(r["gpus"])}
    return d

def load_eternia():
    # updated training epoch (window2 = bit-identical trajectory)
    new_s=old_s=peak=None
    with open(os.path.join(LOGS,"gnn_fast_measured.csv")) as f:
        for r in csv.DictReader(f):
            if r["config"].startswith("window2"):
                new_s=float(r["eternia_epoch_s"]); old_s=float(r["old_epoch_s"])
                peak=float(r["peak_gpu_mib"])
    return {"new_tput":FEAT_MIB/new_s,"old_tput":FEAT_MIB/old_s,"peak":peak,
            "new_s":new_s,"old_s":old_s}

C=load_competitors(); E=load_eternia()

# ---- Figure 1: single-GPU feature-store throughput (GB/s) ----
# bytes/sec of the 54.2 GB feature matrix streamed through GPU aggregation.
# Eternia bars = full training gather (does MORE work: neighbor-agg + backward);
# competitor bars = sum-pool readout. Common axis = effective feature bandwidth.
rows=[
 ("in-core\n(cudaMalloc)", 0.0, "OOM", GREY),
 ("Eternia\n(original)",   E["old_tput"]/1024.0, "", GREY),
 ("staged\nH2D window",    C["staged"]["tput"]/1024.0, "", "#6C7A89"),
 ("Eternia\n(updated)",    E["new_tput"]/1024.0, "", TEAL),
 ("zero-copy\nmapped",     C["zerocopy"]["tput"]/1024.0, "", "#6C7A89"),
 ("GDS-compat\nfile stream",C["gds_compat"]["tput"]/1024.0, "", "#6C7A89"),
]
fig,ax=plt.subplots(figsize=(8.2,4.2))
xs=range(len(rows)); vals=[r[1] for r in rows]; cols=[r[3] for r in rows]
bars=ax.bar(xs,vals,color=cols,edgecolor="#222",linewidth=0.6,width=0.66)
for i,(lab,v,tag,_) in enumerate(rows):
    if tag=="OOM":
        ax.text(i,0.06,"OOM",ha="center",va="bottom",fontsize=10,color=RED,fontweight="bold",rotation=90)
    else:
        ax.text(i,v+0.05,f"{v:.2f}",ha="center",va="bottom",fontsize=9.5,fontweight="bold")
ax.set_xticks(list(xs)); ax.set_xticklabels([r[0] for r in rows],fontsize=8.6)
ax.set_ylabel("effective feature-store throughput (GB/s)",fontsize=10)
ax.set_title("papers100M GNN, one A100-40GB: feature-store throughput (54.2 GB > HBM)",fontsize=11,fontweight="bold")
ax.set_ylim(0,max(vals)*1.18)
# arrow annotating the rewrite jump
ax.annotate("", xy=(3,E["new_tput"]/1024.0+0.06), xytext=(1,E["old_tput"]/1024.0+0.04),
            arrowprops=dict(arrowstyle="->",color=DARKE,lw=1.8,connectionstyle="arc3,rad=-0.32"))
ax.text(1.62,E["new_tput"]/1024.0*1.28,f"{E['new_tput']/E['old_tput']:.1f}x faster\n(Luke's kernel rewrite)",
        ha="center",va="center",fontsize=8.8,color=DARKE,fontweight="bold")
ax.spines[["top","right"]].set_visible(False); ax.grid(axis="y",alpha=0.25)
ax.text(0.5,-0.215,"Eternia bars = full training gather (neighbor-agg + backward, harder); competitor bars = sum-pool\n"
        "readout. Common axis = feature bytes/s delivered to GPU compute.",
        transform=ax.transAxes,ha="center",fontsize=7.6,color="#555")
plt.tight_layout(rect=[0,0.03,1,1])
plt.savefig(os.path.join(FIG,"fig_update_gnn_throughput.png"),dpi=150); plt.close()
print("wrote fig_update_gnn_throughput.png",[round(v,2) for v in vals])

# ---- Figure 2: single-GPU peak GPU memory (bounded footprint) ----
mrows=[
 ("staged",    C["staged"]["peak"], "#6C7A89"),
 ("GDS-compat",C["gds_compat"]["peak"], "#6C7A89"),
 ("zero-copy", C["zerocopy"]["peak"], "#6C7A89"),
 ("Eternia\n(updated)", E["peak"], TEAL),
]
fig,ax=plt.subplots(figsize=(6.8,4.2))
xs=range(len(mrows)); vals=[r[1] for r in mrows]; cols=[r[2] for r in mrows]
ax.bar(xs,vals,color=cols,edgecolor="#222",linewidth=0.6,width=0.6)
for i,(lab,v,_) in enumerate(mrows):
    ax.text(i,v+30,f"{int(v)}",ha="center",va="bottom",fontsize=9.5,fontweight="bold")
ax.set_xticks(list(xs)); ax.set_xticklabels([r[0] for r in mrows],fontsize=8.8)
ax.set_ylabel("whole-process peak GPU (MiB)",fontsize=10)
ax.set_title("Bounded GPU footprint: all << 40 GiB HBM (feature matrix is 54.2 GB)",fontsize=10.5,fontweight="bold")
ax.set_ylim(0,max(vals)*1.28)
ax.spines[["top","right"]].set_visible(False); ax.grid(axis="y",alpha=0.25)
ax.text(0.5,-0.205,"Eternia's peak carries the compression/HBM cache; still <4% of HBM and flat vs data size.\n"
        "Only Eternia + GDS scale past host RAM.",transform=ax.transAxes,ha="center",fontsize=7.6,color="#555")
plt.tight_layout(rect=[0,0.05,1,1])
plt.savefig(os.path.join(FIG,"fig_update_gnn_peakgpu.png"),dpi=150); plt.close()
print("wrote fig_update_gnn_peakgpu.png",[int(v) for v in vals])
