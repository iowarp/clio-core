#!/usr/bin/env python3
"""Consolidate the "Eternia vs all" Track A / Track B runs into two CSVs and the
three comparison figures (A1 throughput-vs-size, A2 peak-GPU-vs-size, B1
aggregate-throughput-vs-GPUs), plus a max-dataset table.

Reads per-size Track-A CSVs from LOGDIR (eternia_logs/trackA_<gib>gb_<jid>.csv,
schema: method,size_gib,ran,peak_gpu_mib,throughput_mibps,time_per_iter_s,inertia,iters)
and the Track-B Eternia scaling CSV, writes consolidated CSVs + figures HERE.

Run on host python3 (has matplotlib):  python3 plot_eternia_vs_all.py
"""
import csv, os, glob, collections
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LOGDIR = "/u/rpawar/eternia_logs"
OUT = os.path.join(HERE, "figures"); os.makedirs(OUT, exist_ok=True)
ETERNIA="#0E9E8E"; INCORE="#C77A34"; UVM="#B05C8E"; STAGED="#4C8FBF"; OOM="#D6473D"
plt.rcParams.update({"figure.dpi":130,"savefig.dpi":300,"font.size":11,
    "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,
    "grid.color":"#D8DEE6","grid.linewidth":0.7,"axes.axisbelow":True,
    "font.family":"DejaVu Sans"})
def save(fig, stem):
    for e in ("png","pdf"): fig.savefig(os.path.join(OUT,stem+"."+e),bbox_inches="tight")
    plt.close(fig); print("wrote", stem)

# ---------- consolidate Track A ----------
rowsA = {}   # (method,size) -> row (latest wins by filename sort)
cand = sorted(glob.glob(os.path.join(LOGDIR, "trackA_*gb_*.csv")) +
              glob.glob(os.path.join(LOGDIR, "trackA_multi_*.csv")))
for f in cand:
    with open(f) as fh:
        rd = csv.DictReader(fh)
        if not rd.fieldnames or "method" not in rd.fieldnames or "size_gib" not in rd.fieldnames:
            continue   # skip the wide-schema CSVs
        for r in rd:
            if not r.get("method") or not r.get("size_gib"): continue
            rowsA[(r["method"], int(r["size_gib"]))] = r
if rowsA:
    order = ["eternia","traditional_incore","staged","uvm"]
    outp = os.path.join(HERE, "eternia_vs_all_trackA.csv")
    with open(outp,"w",newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method","size_gib","ran","peak_gpu_mib","throughput_mibps",
                    "time_per_iter_s","inertia","iters"])
        for m in order:
            for s in sorted({k[1] for k in rowsA}):
                if (m,s) in rowsA:
                    r = rowsA[(m,s)]
                    w.writerow([r["method"],r["size_gib"],r["ran"],r["peak_gpu_mib"],
                                r["throughput_mibps"],r["time_per_iter_s"],r["inertia"],r["iters"]])
    print("wrote eternia_vs_all_trackA.csv", len(rowsA), "rows")

    sizes = sorted({k[1] for k in rowsA})
    def series(m, field):
        return [(s, float(rowsA[(m,s)][field])) for s in sizes
                if (m,s) in rowsA and rowsA[(m,s)]["ran"]=="ran"]
    METHODS=[("eternia",ETERNIA,"Eternia (compress+tier)"),
             ("traditional_incore",INCORE,"Traditional in-core"),
             ("staged",STAGED,"Traditional staged"),
             ("uvm",UVM,"CUDA UVM oversubscription")]

    # Fig A1: throughput vs size
    fig,ax=plt.subplots(figsize=(6.2,4.0))
    for m,c,lab in METHODS:
        pts=series(m,"throughput_mibps")
        if pts: ax.plot([p[0] for p in pts],[p[1] for p in pts],"o-",color=c,label=lab)
    ax.axvline(40,color=OOM,ls=":",lw=1.5); ax.text(41,ax.get_ylim()[1]*0.05,
        "40 GiB HBM",color=OOM,fontsize=8,rotation=90,va="bottom")
    ax.set_xlabel("Dataset size (GiB)"); ax.set_ylabel("k-means throughput (MiB/s)")
    ax.set_title("Track A: same SIFT k-means, throughput vs dataset (one A100)")
    ax.legend(fontsize=8.5); save(fig,"fig_A1_throughput_vs_size")

    # Fig A2: peak GPU vs size (log), with in-core=dataset dashed + OOM line
    fig,ax=plt.subplots(figsize=(6.2,4.0))
    ax.plot(sizes,[s*1024 for s in sizes],"--",color="#999",label="= dataset (in-core requirement)")
    # draw growing baselines first, then the flat streaming methods on top (wider)
    for m,c,lab in METHODS:
        if m in ("eternia","staged"): continue
        pts=series(m,"peak_gpu_mib")
        if pts: ax.plot([p[0] for p in pts],[p[1] for p in pts],"o-",color=c,label=lab)
    for m,c,lab,lw,ms in [("staged",STAGED,"Traditional staged",2.2,7),
                          ("eternia",ETERNIA,"Eternia (compress+tier)",2.8,9)]:
        pts=series(m,"peak_gpu_mib")
        if pts: ax.plot([p[0] for p in pts],[p[1] for p in pts],"o-",color=c,label=lab,lw=lw,ms=ms)
    ax.axhline(40*1024,color=OOM,ls=":",lw=1.6); ax.axhspan(40*1024,1e7,color=OOM,alpha=0.07)
    ax.text(sizes[0],40*1024*1.15,"A100 40 GiB HBM — in-core OOMs above",color=OOM,fontsize=8)
    ax.annotate("Eternia + staged: flat ~3.6 GiB\n(peak = window+cache+context)",
                (64,3616),textcoords="offset points",xytext=(-6,26),fontsize=8,color=ETERNIA,
                ha="center",arrowprops=dict(arrowstyle="->",color=ETERNIA,lw=1))
    ax.set_yscale("log"); ax.set_xlabel("Dataset size (GiB)")
    ax.set_ylabel("Peak GPU memory (MiB, whole-process, log)")
    ax.set_title("Track A: peak GPU vs size — Eternia bounded/flat, in-core = dataset")
    ax.legend(fontsize=8.5,loc="upper left"); save(fig,"fig_A2_peakgpu_vs_size")

    # Inertia parity: Eternia (lossy) vs the lossless baseline (in-core where it
    # ran, else staged -- both bit-identical lossless) at each size.
    pp=os.path.join(HERE,"correctness_inertia_parity_a100.csv")
    with open(pp,"w",newline="") as fh:
        w=csv.writer(fh); w.writerow(["size_gib","baseline_method","baseline_inertia",
                                      "eternia_inertia","abs_delta","pct_delta"])
        worst=0.0
        for s in sizes:
            if ("eternia",s) not in rowsA: continue
            ei=float(rowsA[("eternia",s)]["inertia"])
            base=None
            for bm in ("traditional_incore","staged"):
                if (bm,s) in rowsA and rowsA[(bm,s)]["ran"]=="ran":
                    base=(bm,float(rowsA[(bm,s)]["inertia"])); break
            if not base: continue
            d=abs(ei-base[1]); pct=100.0*d/base[1] if base[1] else 0.0
            worst=max(worst,pct)
            w.writerow([s,base[0],f"{base[1]:.1f}",f"{ei:.1f}",f"{d:.1f}",f"{pct:.4f}"])
    print(f"wrote correctness_inertia_parity_a100.csv (max |dinertia| = {worst:.4f}%)")

    # Max-dataset table (stdout)
    print("\n=== MAX DATASET SUSTAINED (Track A, one A100-40GB) ===")
    for m,_,lab in METHODS:
        ran=[s for s in sizes if (m,s) in rowsA and rowsA[(m,s)]["ran"]=="ran"]
        print(f"  {lab:32s}: {max(ran) if ran else 0} GiB")

# ---------- Track B ----------
tb = os.path.join(LOGDIR,"trackB_eternia.csv")
if os.path.exists(tb):
    with open(tb) as fh: brows=[r for r in csv.DictReader(fh)]
    brows=[r for r in brows if r.get("ngpu")]
    if brows:
        outp=os.path.join(HERE,"eternia_vs_all_trackB.csv")
        with open(outp,"w",newline="") as fh:
            w=csv.writer(fh)
            w.writerow(["method","gpus","pooled_gib","agg_throughput_mibps","efficiency",
                        "peak_gpu_mib_per_gpu","max_total_gib","notes"])
            base=None
            for r in sorted(brows,key=lambda x:int(x["ngpu"])):
                n=int(r["ngpu"]); store=float(r["agg_store_mibps"])
                gib_gpu=int(r["pages_per_gpu"])//4096   # 256KiB pages -> GiB
                if base is None: base=store/n
                eff=store/(n*base) if base else 0
                pooled=n*gib_gpu
                w.writerow(["eternia_distributed",n,pooled,f"{store:.1f}",f"{eff:.3f}",
                            "~2500 (8GiB field x4 + HBM compressed cache)",pooled,
                            f"per-GPU cuSZp {gib_gpu}GiB/GPU; compressed 3.4x on-GPU"])
            # NVSHMEM measured (bare-metal, 32 GiB/GPU); NCCL/MPI same capacity model.
            w.writerow(["nvshmem",2,64,"interconnect-bound","n/a","32768 (uncompressed 1.0x, measured)",
                        80,"symmetric heap holds UNCOMPRESSED; ceiling = sum of physical HBM"])
            w.writerow(["nvshmem",4,128,"interconnect-bound","n/a","32768 (uncompressed 1.0x, measured)",
                        160,"measured on Delta bare-metal (job 20990679)"])
            w.writerow(["nccl","N","N*40","interconnect-bound","n/a","~40960 (uncompressed 1.0x)",
                        "N*40","collectives; shard uncompressed across GPUs; no per-GPU expansion"])
            w.writerow(["mpi_cuda_aware","N","N*40","interconnect-bound","n/a","~40960 (uncompressed 1.0x)",
                        "N*40","GPU-pointer send/recv; uncompressed shards"])
        print("\nwrote eternia_vs_all_trackB.csv")

        # Fig B1: aggregate throughput vs GPUs (Eternia distributed) + ideal
        ng=[int(r["ngpu"]) for r in sorted(brows,key=lambda x:int(x["ngpu"]))]
        st=[float(r["agg_store_mibps"]) for r in sorted(brows,key=lambda x:int(x["ngpu"]))]
        rd=[float(r["agg_read_mibps"]) for r in sorted(brows,key=lambda x:int(x["ngpu"]))]
        base=st[0]/ng[0]
        fig,ax=plt.subplots(figsize=(6.0,4.0))
        ax.plot(ng,[base*n for n in ng],"--",color="#999",label="ideal linear")
        ax.plot(ng,st,"o-",color=ETERNIA,lw=2,label="Eternia store (compressed)")
        ax.plot(ng,rd,"s-",color=STAGED,lw=2,label="Eternia read (decompressed)")
        for n,s in zip(ng,st): ax.annotate(f"{s:.0f}",(n,s),textcoords="offset points",xytext=(0,7),fontsize=8,ha="center")
        ax.set_xticks(ng); ax.set_xlabel("GPUs (32 GiB/GPU compressed checkpoint)")
        ax.set_ylabel("Aggregate throughput (MiB/s)")
        ax.set_title("Track B: Eternia distributed weak-scaling (per-GPU cuSZp)")
        ax.legend(fontsize=8.5); save(fig,"fig_B1_scaling_vs_gpus")
print("\ndone.")
