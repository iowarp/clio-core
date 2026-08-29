#!/usr/bin/env python3
"""Gray-Scott checkpoint I/O figure: three-way (+variants) comparison on the fast
branch, 1875 MB, identical checksum across arms. CLIO GPU async is fastest."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE=os.path.dirname(os.path.abspath(__file__)); FIG=os.path.join(HERE,"figures")
os.makedirs(FIG,exist_ok=True)
TEAL="#0E9E8E"; GREY="#6C7A89"; DARKE="#0B6E63"

# measured MB/s (job 21446861), most→least
rows=[
 ("CLIO GPU async\n(gpuh5, pinned)", 2177.3, TEAL),
 ("raw disk\n(threaded)",            1713.7, GREY),
 ("HDF5\n(threaded)",                1365.9, GREY),
 ("HDF5\n(inline)",                  1179.1, GREY),
 ("host CLIO",                        839.5, GREY),
]
fig,ax=plt.subplots(figsize=(7.2,4.0))
xs=range(len(rows)); vals=[r[1] for r in rows]; cols=[r[2] for r in rows]
ax.bar(xs,vals,color=cols,edgecolor="#222",linewidth=0.6,width=0.62)
for i,(lab,v,_) in enumerate(rows):
    ax.text(i,v+30,f"{v:.0f}",ha="center",va="bottom",fontsize=9.5,fontweight="bold")
ax.set_xticks(list(xs)); ax.set_xticklabels([r[0] for r in rows],fontsize=8.4)
ax.set_ylabel("checkpoint throughput (MB/s)",fontsize=10)
ax.set_title("Gray-Scott checkpoint I/O (1875 MB, identical bytes across arms)",fontsize=10.5,fontweight="bold")
ax.set_ylim(0,max(vals)*1.16)
ax.annotate("1.27x raw disk\n2.6x host CLIO",xy=(0,2177),xytext=(1.4,2050),
            fontsize=8.6,color=DARKE,fontweight="bold",ha="center",
            arrowprops=dict(arrowstyle="->",color=DARKE,lw=1.4))
ax.spines[["top","right"]].set_visible(False); ax.grid(axis="y",alpha=0.25)
ax.text(0.5,-0.19,"Same GPU computation, three storage sinks; shared checksum proves identical bytes. CLIO's "
        "async snapshot overlaps I/O with compute.",transform=ax.transAxes,ha="center",fontsize=7.4,color="#555")
plt.tight_layout(rect=[0,0.03,1,1])
plt.savefig(os.path.join(FIG,"fig_update_grayscott.png"),dpi=150); plt.close()
print("wrote fig_update_grayscott.png",[round(v) for v in vals])
