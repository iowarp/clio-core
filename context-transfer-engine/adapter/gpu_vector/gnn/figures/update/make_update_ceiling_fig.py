#!/usr/bin/env python3
"""k-means capacity-ceiling + tier-cascade figure (fast gather)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); FIG=os.path.join(HERE,"figures")
os.makedirs(FIG,exist_ok=True)
HBMc="#0B6E63"; DRAMc="#0E9E8E"; NVMEc="#5BA36A"; PFSc="#B05C8E"
TEAL="#0E9E8E"; GREY="#6C7A89"; RED="#C0392B"
sizes=["128 GiB","256 GiB","320 GiB"]; x=np.arange(len(sizes))
fig,(axA,axB)=plt.subplots(1,2,figsize=(9.8,4.2))
hbm=np.array([2047,2047,2047])/1024; dram=np.array([30719,30719,30719])/1024
nvme=np.array([6181,45316,51199])/1024; pfs=np.array([0,0,13684])/1024
axA.bar(x,hbm,width=0.55,color=HBMc,label="HBM (2 GiB cap)")
axA.bar(x,dram,width=0.55,bottom=hbm,color=DRAMc,label="DRAM (30 GiB)")
axA.bar(x,nvme,width=0.55,bottom=hbm+dram,color=NVMEc,label="NVMe (50 GiB)")
axA.bar(x,pfs,width=0.55,bottom=hbm+dram+nvme,color=PFSc,label="PFS (Lustre)")
tot=hbm+dram+nvme+pfs
for i,t in enumerate(tot): axA.text(i,t+1.5,f"{t:.0f}",ha="center",va="bottom",fontsize=9,fontweight="bold")
axA.set_xticks(x); axA.set_xticklabels(sizes,fontsize=9)
axA.set_ylabel("compressed residency (GiB)",fontsize=10)
axA.set_title("Compressed matrix cascades HBM->DRAM->NVMe->PFS",fontsize=10,fontweight="bold")
axA.legend(fontsize=7.4,loc="upper left",framealpha=0.9); axA.set_ylim(0,tot.max()*1.2)
axA.spines[["top","right"]].set_visible(False); axA.grid(axis="y",alpha=0.2)
axA.text(0.5,-0.16,"3.36x compression (zstd): a 320 GiB matrix stored in 95 GiB across four tiers.",transform=axA.transAxes,ha="center",fontsize=7.4,color="#555")
et=[809.3,837.4,302.6]; stg=[1881.4,1883.8,None]; w=0.38
axB.bar(x-w/2,et,width=w,color=TEAL,edgecolor="#222",linewidth=0.5,label="Eternia (fast gather)")
axB.bar(x[:2]+w/2,stg[:2],width=w,color=GREY,edgecolor="#222",linewidth=0.5,label="staged (uncompressed)")
for i,v in enumerate(et): axB.text(i-w/2,v+25,f"{v:.0f}",ha="center",va="bottom",fontsize=8.5,fontweight="bold")
for i in range(2): axB.text(i+w/2,stg[i]+25,f"{stg[i]:.0f}",ha="center",va="bottom",fontsize=8.5)
axB.text(2+w/2,60,"staged\nOOM",ha="center",va="bottom",fontsize=8,color=RED,fontweight="bold")
axB.set_xticks(x); axB.set_xticklabels(sizes,fontsize=9); axB.set_ylabel("throughput (MiB/s)",fontsize=10)
axB.set_title("At 320 GiB (> host RAM) only Eternia survives",fontsize=10,fontweight="bold")
axB.set_ylim(0,2200); axB.legend(fontsize=7.8,loc="upper right",framealpha=0.9)
axB.spines[["top","right"]].set_visible(False); axB.grid(axis="y",alpha=0.2)
axB.text(0.5,-0.16,"in-core OOMs and UVM is impractical at every size; staged OOMs past host RAM (320 > 257 GiB). Eternia peak stays ~2.9 GiB (111x smaller than 320 GiB).",transform=axB.transAxes,ha="center",fontsize=7.0,color="#555")
plt.tight_layout(rect=[0,0.04,1,1])
plt.savefig(os.path.join(FIG,"fig_update_kmeans_ceiling.png"),dpi=150); plt.close()
print("wrote fig_update_kmeans_ceiling.png")
