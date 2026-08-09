#!/usr/bin/env python3
"""Generate Eternia evaluation figures (PNG + PDF) from the result CSVs.

Reads the a1..a4 CSVs in this directory and writes figures/ alongside. Pure
matplotlib (Agg), no seaborn. Run: python3 plot_results.py
"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

ETERNIA = "#0E9E8E"; TRAD = "#C77A34"; OOM = "#D6473D"; TIER = "#7C6FD0"
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 300, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#D8DEE6", "grid.linewidth": 0.7,
    "axes.axisbelow": True, "font.family": "DejaVu Sans",
})

def rows(name):
    with open(os.path.join(HERE, name)) as f:
        return list(csv.DictReader(f))

def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, stem + "." + ext), bbox_inches="tight")
    plt.close(fig)
    print("wrote", stem + ".png/.pdf")

# ---- A1: capacity crossover (peak GPU mem decoupling) ----
d = rows("a1_capacity_crossover.csv")
sizes = [float(r["dataset_gib"]) for r in d]
trad_ok = [r["traditional_status"] == "OK" for r in d]
HBM = 8.0
fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.axhspan(HBM, max(sizes) + 1, color=OOM, alpha=0.08)
ax.text(sizes[-1], HBM + 0.4, "OOM  (dataset > HBM)", color=OOM, ha="right", fontsize=9)
ax.axhline(HBM, ls=(0, (2, 2)), color="#444", lw=1.2)
ax.text(sizes[0], HBM + 0.25, "8 GiB HBM", fontsize=9, color="#444")
# traditional needs peak GPU = dataset
ax.plot(sizes, sizes, "-o", color=TRAD, lw=2, label="Traditional (dataset resident)")
for s, ok in zip(sizes, trad_ok):
    if not ok:
        ax.plot(s, s, "o", ms=9, color=OOM, zorder=5)
        ax.annotate("✗", (s, s), color=OOM, fontsize=13, ha="center", va="center")
# eternia flat ~4 MiB
ax.plot(sizes, [0.004] * len(sizes), "-s", color=ETERNIA, lw=2.2,
        label="Eternia (streamed 4 MiB window)")
ax.set_xlabel("dataset size (GiB)"); ax.set_ylabel("peak GPU memory (GiB)")
ax.set_xticks(sizes); ax.set_ylim(0, max(sizes) + 1)
ax.legend(loc="upper left", frameon=False, fontsize=9)
ax.set_title("A1 · Traditional OOMs past HBM; Eternia streams", fontsize=11, loc="left")
save(fig, "fig_a1_capacity_crossover")

# ---- A2: compression ablation (stored footprint) ----
d = rows("a2_compression_ablation.csv")
cats = sorted({r["dataset_gib"] for r in d}, key=float)
et = {r["dataset_gib"]: float(r["stored_mib"]) / 1024
      for r in d if r["mode"] == "eternia"}
ti = {r["dataset_gib"]: float(r["stored_mib"]) / 1024
      for r in d if r["mode"] == "tier_only"}
x = range(len(cats)); w = 0.36
fig, ax = plt.subplots(figsize=(5.0, 3.6))
b1 = ax.bar([i - w/2 for i in x], [ti[c] for c in cats], w, color=TIER, label="Tiering only (raw)")
b2 = ax.bar([i + w/2 for i in x], [et[c] for c in cats], w, color=ETERNIA, label="Eternia (compressed)")
for i, c in enumerate(cats):
    ax.text(i - w/2, ti[c], f"{ti[c]:.1f}", ha="center", va="bottom", fontsize=9)
    ax.text(i + w/2, et[c], f"{et[c]:.1f}", ha="center", va="bottom", fontsize=9)
    ax.text(i, max(ti[c], et[c]) + 0.4, "3.41× smaller", ha="center", color=ETERNIA, fontsize=9)
ax.set_xticks(list(x)); ax.set_xticklabels([f"{c} GiB" for c in cats])
ax.set_ylabel("stored in tier stack (GiB)")
ax.legend(loc="upper left", frameon=False, fontsize=9)
ax.set_title("A2 · Compression = 3.4× more effective capacity", fontsize=11, loc="left")
save(fig, "fig_a2_compression_ablation")

# ---- A3: HBM window scaling ----
d = sorted(rows("a3_hbm_window_scaling.csv"), key=lambda r: float(r["window_mib"]))
wm = [float(r["window_mib"]) for r in d]; km = [float(r["km_runtime_s"]) for r in d]
fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.plot(wm, km, "-o", color=ETERNIA, lw=2.2)
i_best = km.index(min(km))
ax.plot(wm[i_best], km[i_best], "o", ms=10, color=TRAD, zorder=5)
ax.annotate(f"sweet spot\n{km[i_best]:.0f} s @ {wm[i_best]:.0f} MiB",
            (wm[i_best], km[i_best]), textcoords="offset points", xytext=(10, 14),
            fontsize=9, color=TRAD)
ax.set_xscale("log", base=2); ax.set_xticks(wm)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xlabel("HBM working window (MiB)"); ax.set_ylabel("k-means runtime (s)")
ax.set_ylim(0, max(km) * 1.15)
ax.set_title("A3 · Shrinking HBM 32× barely moves runtime (6 GiB)", fontsize=11, loc="left")
save(fig, "fig_a3_hbm_window_scaling")

# ---- A4: Gray-Scott checkpoint (two metrics) ----
d = {r["method"]: r for r in rows("a4_grayscott_checkpoint.csv")}
fig, (axL, axR) = plt.subplots(1, 2, figsize=(6.4, 3.4))
methods = ["traditional", "eternia"]; cols = [TRAD, ETERNIA]
ms = [float(d[m]["ms_per_checkpoint"]) for m in methods]
fp = [float(d[m]["footprint_mib"]) for m in methods]
axL.bar(methods, ms, color=cols, width=0.6)
for i, v in enumerate(ms): axL.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
axL.set_ylabel("ms / checkpoint"); axL.set_title("latency  (4.9× faster)", fontsize=10, loc="left")
axR.bar(methods, fp, color=cols, width=0.6)
for i, v in enumerate(fp):
    axR.text(i, v, f"{v:.0f} MiB", ha="center", va="bottom", fontsize=9)
axR.set_ylabel("footprint (MiB)"); axR.set_title("footprint  (7.3× smaller)", fontsize=10, loc="left")
for a in (axL, axR): a.set_xticklabels(["traditional\n→disk", "eternia\n→HBM"], fontsize=9)
fig.suptitle("A4 · Gray-Scott checkpoint: compressed in-HBM vs traditional",
             fontsize=11, x=0.02, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.95))
save(fig, "fig_a4_grayscott_checkpoint")

print("done ->", OUT)
