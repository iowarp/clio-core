#!/usr/bin/env python3
"""Generate Eternia DELTA A100 evaluation figures (PNG + PDF) from the b1..b4
result CSVs in this directory. Pure matplotlib (Agg). Run inside the container:
    python3 plot_results_a100.py

Kept separate from plot_results.py (which draws the laptop/8GiB a1..a4 figures).
"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

ETERNIA = "#0E9E8E"; TRAD = "#C77A34"; OOM = "#D6473D"; TIER = "#7C6FD0"
HBM = "#0E9E8E"; DRAM = "#4C8FBF"; NVME = "#C77A34"; PFS = "#8A6FB0"; UVM = "#B05C8E"
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 300, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#D8DEE6", "grid.linewidth": 0.7,
    "axes.axisbelow": True, "font.family": "DejaVu Sans",
})

def rows(name):
    p = os.path.join(HERE, name)
    if not os.path.exists(p):
        print("skip (missing):", name); return []
    with open(p) as f:
        return list(csv.DictReader(f))

def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, stem + "." + ext), bbox_inches="tight")
    plt.close(fig); print("wrote", stem + ".png/.pdf")

# ---- B1a: capacity crossover — peak GPU memory decoupling ----
d = rows("b1_capacity_crossover_a100.csv")
if d:
    sizes = [float(r["dataset_gib"]) for r in d]
    peak = [float(r["eternia_peak_gpu_mib"]) for r in d]      # MiB, ~4
    HBM_GIB = 40.0
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    # Traditional in-core would need the whole dataset resident = dataset GiB.
    ax.plot(sizes, [s * 1024 for s in sizes], "o--", color=TRAD,
            label="Traditional in-core (peak = dataset)")
    ax.plot(sizes, peak, "o-", color=ETERNIA, lw=2.2,
            label="Eternia (peak = one window)")
    ax.axhline(HBM_GIB * 1024, color=OOM, ls=":", lw=1.6)
    ax.axhspan(HBM_GIB * 1024, 1e6, color=OOM, alpha=0.08)
    ax.text(sizes[0], HBM_GIB * 1024 * 1.15, "A100 40 GiB HBM — traditional OOMs above",
            color=OOM, fontsize=8.5)
    ax.set_yscale("log")
    ax.set_xlabel("Dataset size (GiB)"); ax.set_ylabel("Peak GPU memory (MiB, log)")
    ax.set_title("Capacity decoupling: Eternia peak GPU is O(window), not O(dataset)")
    ax.legend(fontsize=8.5, loc="center right")
    save(fig, "fig_b1_capacity_crossover_a100")

    # ---- B1b: tiering cascade — stacked per-tier bytes ----
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    x = range(len(sizes))
    hbm = [float(r["hbm_used_mib"]) / 1024 for r in d]
    dram = [float(r["dram_used_mib"]) / 1024 for r in d]
    nvme = [float(r["nvme_used_mib"]) / 1024 for r in d]
    pfs = [float(r["pfs_used_mib"]) / 1024 for r in d]
    b = [0.0] * len(sizes)
    for vals, c, lab in [(hbm, HBM, "HBM"), (dram, DRAM, "DRAM"),
                         (nvme, NVME, "NVMe (local)"), (pfs, PFS, "PFS (Lustre)")]:
        ax.bar(x, vals, bottom=b, color=c, label=lab, width=0.62)
        b = [bb + vv for bb, vv in zip(b, vals)]
    ax.set_xticks(list(x)); ax.set_xticklabels([f"{int(s)}" for s in sizes])
    ax.set_xlabel("Dataset size (GiB)")
    ax.set_ylabel("Compressed bytes per tier (GiB)")
    ax.set_title("Deep tiering cascade: HBM → DRAM → NVMe → PFS (3.41× compressed)")
    ax.legend(fontsize=8.5, ncol=4, loc="upper left")
    save(fig, "fig_b1_tiering_cascade_a100")

# ---- B2: multi-GPU weak scaling ----
d = rows("b2_scaling_a100.csv")
if d:
    ng = [int(r["ngpu"]) for r in d]
    store = [float(r["agg_store_mibps"]) for r in d]
    base = store[0] / ng[0]
    ideal = [base * n for n in ng]
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.plot(ng, ideal, "--", color="#999", label="ideal linear")
    ax.plot(ng, store, "o-", color=ETERNIA, lw=2.2, label="measured aggregate")
    for n, s in zip(ng, store):
        ax.annotate(f"{s:.0f}", (n, s), textcoords="offset points",
                    xytext=(0, 7), fontsize=8.5, ha="center")
    ax.set_xticks(ng)
    ax.set_xlabel("GPUs (4 GiB compressed checkpoint each)")
    ax.set_ylabel("Aggregate store throughput (MiB/s)")
    ax.set_title("Multi-GPU weak scaling (per-GPU cuSZp compression)")
    ax.legend(fontsize=9, loc="upper left")
    save(fig, "fig_b2_scaling_a100")

# ---- B3: baseline throughput vs dataset ----
d = rows("b3_baselines_a100.csv")
if d:
    def series(method):
        pts = [(float(r["dataset_gib"]), float(r["end_to_end_mibps"]))
               for r in d if r["method"] == method and r["dataset_gib"]
               and r["end_to_end_mibps"]]
        return sorted(pts)
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for m, c, lab in [("uvm_oversub", UVM, "UVM oversubscription"),
                      ("traditional_staged_putget", TRAD, "Traditional staged put/get")]:
        pts = series(m)
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=c, label=lab)
    ax.axvline(40, color=OOM, ls=":", lw=1.6)
    ax.text(41, ax.get_ylim()[1] * 0.9 if False else 500,
            "40 GiB HBM\n(in-core OOM)", color=OOM, fontsize=8)
    ax.set_xlabel("Dataset size (GiB)")
    ax.set_ylabel("End-to-end throughput (MiB/s)")
    ax.set_title("Baselines on one A100: throughput vs dataset size")
    ax.legend(fontsize=8.5, loc="upper right")
    save(fig, "fig_b3_baselines_a100")

# ---- B4: Gray-Scott checkpoint — traditional vs eternia ----
d = rows("b4_grayscott_checkpoint_a100.csv")
if d:
    grids, trad, eter, spd = [], [], [], []
    seen = []
    for r in d:
        g = r["grid"]
        if g not in seen:
            seen.append(g)
    for g in seen:
        gr = [r for r in d if r["grid"] == g]
        t = next((float(r["ms_per_checkpoint"]) for r in gr if "traditional" in r["method"]), None)
        e = next((float(r["ms_per_checkpoint"]) for r in gr if "eternia" in r["method"]), None)
        s = next((float(r["speedup_vs_traditional"]) for r in gr if "eternia" in r["method"]), None)
        grids.append(g.replace("x", "×")); trad.append(t); eter.append(e); spd.append(s)
    import numpy as np
    x = np.arange(len(grids)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.bar(x - w/2, trad, w, color=TRAD, label="Traditional (D2H → PFS)")
    ax.bar(x + w/2, eter, w, color=ETERNIA, label="Eternia (compress in HBM)")
    for i, s in enumerate(spd):
        ax.annotate(f"{s:.1f}× faster", (x[i] + w/2, eter[i]),
                    textcoords="offset points", xytext=(0, 6), fontsize=8, ha="center",
                    color=ETERNIA)
    ax.set_xticks(x); ax.set_xticklabels(grids, fontsize=8.5)
    ax.set_xlabel("Gray-Scott grid"); ax.set_ylabel("ms / checkpoint")
    ax.set_title("Checkpoint transfer: compressed-in-HBM vs uncompressed-to-PFS")
    ax.legend(fontsize=9, loc="upper left")
    save(fig, "fig_b4_grayscott_checkpoint_a100")

print("done.")
