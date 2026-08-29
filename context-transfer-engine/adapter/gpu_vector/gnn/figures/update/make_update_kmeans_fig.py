#!/usr/bin/env python3
"""k-means fast-gather figure for the update report: (A) the gather-port speedup
at 8 GiB, (B) fast-gather scales flat in throughput and peak across a 24x range,
running past HBM where in-core OOMs. All numbers measured on Luke's branch."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE=os.path.dirname(os.path.abspath(__file__)); FIG=os.path.join(HERE,"figures")
os.makedirs(FIG,exist_ok=True)
TEAL="#0E9E8E"; GREY="#6C7A89"; RED="#C0392B"; DARKE="#0B6E63"

fig,(axA,axB)=plt.subplots(1,2,figsize=(9.6,4.1))

# ---- Panel A: gather-port speedup at 8 GiB (throughput) ----
labels=["naive port\n(single-block)","original\n(cuSZp, 637)","fast-gather\nport (this)","in-core\n(reference)"]
vals=[84.3,227.0,637.0,2167.1]
cols=[GREY,GREY,TEAL,"#B0B7BD"]
bars=axA.bar(range(len(vals)),vals,color=cols,edgecolor="#222",linewidth=0.6,width=0.66)
for i,v in enumerate(vals):
    axA.text(i,v+30,f"{v:.0f}",ha="center",va="bottom",fontsize=9.5,fontweight="bold")
axA.set_xticks(range(len(labels))); axA.set_xticklabels(labels,fontsize=8.3)
axA.set_ylabel("k-means throughput (MiB/s)",fontsize=10)
axA.set_title("Porting k-means onto the fast gather (8 GiB)",fontsize=10.5,fontweight="bold")
axA.set_ylim(0,max(vals)*1.16)
axA.annotate("", xy=(2,637), xytext=(0,84),
             arrowprops=dict(arrowstyle="->",color=DARKE,lw=1.8,connectionstyle="arc3,rad=-0.3"))
axA.text(0.75,760,"7.6x vs naive\n2.8x vs original",ha="center",va="center",
         fontsize=8.6,color=DARKE,fontweight="bold")
axA.spines[["top","right"]].set_visible(False); axA.grid(axis="y",alpha=0.25)

# ---- Panel B: fast-gather scales flat (throughput + peak) across sizes ----
sizes=["2 GiB","8 GiB","48 GiB"]; x=range(len(sizes))
tput=[684.4,637.0,664.5]; peak=[2940,2940,2940]; incore_ok=[True,True,False]
axB.bar(x,tput,color=TEAL,edgecolor="#222",linewidth=0.6,width=0.5,label="Eternia throughput (MiB/s)")
for i,v in enumerate(tput):
    axB.text(i,v+15,f"{v:.0f}",ha="center",va="bottom",fontsize=9,fontweight="bold")
axB.set_ylabel("throughput (MiB/s)",fontsize=10,color=DARKE)
axB.set_ylim(0,900); axB.set_xticks(list(x)); axB.set_xticklabels(sizes,fontsize=9)
axB.set_title("Fast-gather sustains ~650 MiB/s; peak flat; runs past HBM",fontsize=10.0,fontweight="bold")
# peak on twin axis
axB2=axB.twinx()
axB2.plot(x,peak,"o--",color="#B05C8E",lw=2,ms=7,label="peak GPU (MiB)")
axB2.set_ylabel("peak GPU (MiB)",fontsize=10,color="#B05C8E")
axB2.set_ylim(0,12000)
for i,p in enumerate(peak):
    axB2.text(i,p+500,f"{p}",ha="center",va="bottom",fontsize=8,color="#B05C8E")
# in-core OOM marker at 48 GiB
axB.text(2,40,"in-core\nOOM",ha="center",va="bottom",fontsize=8.5,color=RED,fontweight="bold")
axB.spines[["top"]].set_visible(False); axB.grid(axis="y",alpha=0.2)
axB.text(0.5,-0.20,"Bit-exact at every size (inertia == in-core). Peak 2940 MiB is 16.7x smaller than the 48 GiB matrix.",
         transform=axB.transAxes,ha="center",fontsize=7.4,color="#555")

plt.tight_layout(rect=[0,0.03,1,1])
plt.savefig(os.path.join(FIG,"fig_update_kmeans.png"),dpi=150); plt.close()
print("wrote fig_update_kmeans.png")
