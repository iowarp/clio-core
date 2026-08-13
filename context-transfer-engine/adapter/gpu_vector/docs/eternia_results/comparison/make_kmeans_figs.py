#!/usr/bin/env python3
"""k-means comparison figures with ALL measured tech (Delta A100-40GB).
Capacity ceiling (necessity), peak GPU vs size (6 methods), Track B pooling."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"figures")
os.makedirs(OUT,exist_ok=True)
ET="#0E9E8E"; OOM="#D6473D"; INC="#C77A34"; UVM="#B05C8E"; STG="#4C8FBF"; ZC="#5BA36A"; GDS="#8A6FB0"
plt.rcParams.update({"figure.dpi":130,"savefig.dpi":300,"font.size":11,
  "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,
  "grid.color":"#D8DEE6","grid.linewidth":0.7,"axes.axisbelow":True,"font.family":"DejaVu Sans"})
def save(fig,s):
    for e in ("png","pdf"): fig.savefig(os.path.join(OUT,s+"."+e),bbox_inches="tight")
    plt.close(fig); print("wrote",s)

# ---------- Fig 1: CAPACITY CEILING (max dataset sustained on ONE A100) ----------
# measured: last success / first OOM
methods=["in-core\n(cudaMalloc)","CUDA UVM","zero-copy\n(mapped host)","staged\n(host stream)","GDS-compat\n(file stream)","Eternia\n(compress+tier)"]
maxgib=[32,32,96,256,320,320]          # measured max that ran
ceil_note=["OOM >40 (HBM)","impractical >40","OOM by 256 (host RAM)","OOM 320 (host RAM)","disk (≥320, tested)","PFS (≥320, tested)"]
cols=[INC,UVM,ZC,STG,GDS,ET]
fig,ax=plt.subplots(figsize=(9.0,4.4)); y=np.arange(len(methods))
ax.barh(y,maxgib,color=cols)
ax.axvline(40,color=OOM,ls=":",lw=1.6); ax.text(41,5.2,"40 GiB HBM",color=OOM,fontsize=8,rotation=90,va="top")
ax.axvline(257,color="#888",ls="--",lw=1.4); ax.text(258,5.2,"257 GiB host RAM",color="#555",fontsize=8,rotation=90,va="top")
for i,(v,n) in enumerate(zip(maxgib,ceil_note)):
    ax.text(v+6,i,n,va="center",fontsize=8,color="#333")
ax.set_yticks(y); ax.set_yticklabels(methods,fontsize=9); ax.invert_yaxis()
ax.set_xlabel("Max k-means dataset sustained on one A100 (GiB)"); ax.set_xlim(0,420)
ax.set_title("Capacity ceiling: only Eternia & file-streaming (GDS) pass host RAM;\nonly Eternia does it transparently + compressed (3.4x)")
save(fig,"fig_km_capacity_ceiling")

# ---------- Fig 2: PEAK GPU vs dataset size (whole-process) ----------
sizes=[8,16,32,48,64,96,256,320]
# eternia flat 3614 (all); staged 3618 (to 256, OOM 320); incore=dataset (to 32); uvm impractical>32; zerocopy/gds small
et=[3614]*8
inc=[8622,16814,33198,None,None,None,None,None]
uvmv=[11806,20000,36386,None,None,None,None,None]
zc=[None,None,536,None,556,620,None,None]        # expand-bench measured (32/64/96)
gdsv=[None,None,432,None,None,432,432,432]        # gds-compat (32/96/256/320)
stg=[3618]*7+[None]                               # staged CTE peak; OOM 320
fig,ax=plt.subplots(figsize=(7.4,4.6))
ax.plot(sizes,[s*1024 for s in sizes],"--",color="#999",lw=1.4,label="= dataset (in-core needs)")
def pl(v,c,l,m="o",lw=2,ms=6):
    xs=[s for s,val in zip(sizes,v) if val]; ys=[val for val in v if val]
    if xs: ax.plot(xs,ys,m+"-",color=c,label=l,lw=lw,ms=ms)
pl(inc,INC,"in-core"); pl(uvmv,UVM,"UVM")
pl(zc,ZC,"zero-copy"); pl(gdsv,GDS,"GDS-compat")
pl(stg,STG,"staged"); pl(et,ET,"Eternia",lw=2.8,ms=8)
ax.axhline(40*1024,color=OOM,ls=":",lw=1.6); ax.axhspan(40*1024,1e7,color=OOM,alpha=0.06)
ax.text(8,40*1024*1.15,"40 GiB HBM — in-core/UVM OOM above",color=OOM,fontsize=8)
ax.set_yscale("log"); ax.set_xlabel("Dataset size (GiB)"); ax.set_ylabel("Peak GPU memory (MiB, whole-process, log)")
ax.set_title("Peak GPU vs size: Eternia bounded/flat ~3.6 GiB; in-core = dataset")
ax.legend(fontsize=8,ncol=2,loc="lower right"); save(fig,"fig_km_peakgpu_allmethods")

# ---------- Fig 3: Track B multi-GPU pooling ----------
ng=[1,2,4]
nccl=[2168.9,4317.9,8494.0]; mpi=[2170.5,4314.9,8655.8]; nvs=[2157.0,4314.2,8583.4]
etd=[289.1,574.5,1021.7]   # eternia distributed 8GiB/GPU
fig,ax=plt.subplots(figsize=(7.2,4.4))
base=nccl[0]
ax.plot(ng,[base*n for n in ng],"--",color="#999",lw=1.3,label="ideal linear")
ax.plot(ng,nccl,"o-",color=STG,label="NCCL (uncompressed)")
ax.plot(ng,mpi,"s--",color=UVM,label="CUDA-aware MPI")
ax.plot(ng,nvs,"^:",color=GDS,label="NVSHMEM")
ax.plot(ng,etd,"o-",color=ET,lw=2.6,label="Eternia distributed (compressed)")
ax.set_xticks(ng); ax.set_xlabel("GPUs (8 GiB/GPU)"); ax.set_ylabel("Aggregate throughput (MiB/s)")
ax.set_title("Track B pooling: NCCL/MPI/NVSHMEM ~identical (comm-negligible),\nhold data 1.0x uncompressed; Eternia compresses 3.4x")
ax.legend(fontsize=8.5); save(fig,"fig_km_trackB_pooling")
print("done.")
