#!/usr/bin/env python3
"""How much does choosing a codec per chunk pay on each simulation probe?

    tune_report.py [--out PNG]

For every probe tune_sims.sh ran (workloads ref-*-tune-* with a stored
exhaustive search) and, for reference, the full simulation workloads: the best
single codec by total balanced 4-tier cost, the opportunity
(best single - per-chunk best) / best single, the share of chunks whose
cheapest setting is another one, and the opportunity per field and per
quarter of the run (early -> late). Writes runs/tune_report.csv and
figures/new-workloads/sim-tuning/tuning_probes.png.
"""
import argparse
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "model-accuracy"))
import eval_v2_workloads as ev  # noqa: E402

STORE = "/mnt/nvme0/v2-work/baselines"
FIGS = os.path.join(HERE, "..", "figures", "new-workloads", "sim-tuning")
REFERENCE = ["nyx-full", "ref-lammps-b70-2000"]
INK, INK2, OPP = "#1f2328", "#57606a", "#2e86ab"


def field(blob):
    """Field name of a chunk (Nyx / AMReX component or LAMMPS array)."""
    m = (re.search(r"comp\d+_([A-Za-z_]+)\.f32", blob)
         or re.search(r"__(force|position|velocity)_step", blob))
    return m.group(1) if m else "?"


def step(blob):
    """Simulation step of a chunk."""
    m = re.search(r"plt(\d+)", blob) or re.search(r"_step_(\d+)_", blob)
    return int(m.group(1)) if m else -1


def gain(cost, sel, bf):
    """Opportunity (%) of the per-chunk best over setting bf on the selected chunks."""
    c = cost[sel]
    return 100 * (c[:, bf].sum() - c.min(axis=1).sum()) / c[:, bf].sum()


def analyse(ds, names, store):
    """One probe's row: best single, opportunity, per field and per quarter."""
    blobs, cost = ev.load_truth(os.path.join(STORE, ds, "exhaustive"), len(names), store)
    ok = ~np.isnan(cost).any(axis=1)
    blobs, cost = np.array(blobs)[ok], cost[ok]
    cost = ev.for_selection(cost)   # best single and oracle: candidates only
    bf = int(np.argmin(cost.sum(axis=0)))
    f = np.array([field(b) for b in blobs])
    t = np.array([step(b) for b in blobs])
    q = np.digitize(t, np.quantile(t, [0.25, 0.5, 0.75]))
    n, gib = ev.workload_size(ds)
    row = {"probe": ds.replace("ref-", ""), "chunks": n, "GiB": gib,
           "best_single": names[bf], "opportunity_pct": gain(cost, slice(None), bf),
           "chunks_other_best_pct": 100 * np.mean(cost.argmin(axis=1) != bf)}
    for k in range(4):
        row[f"q{k + 1}_pct"] = gain(cost, q == k, bf)
    for fv in sorted(set(f)):
        row[f"field_{fv}_pct"] = gain(cost, f == fv, bf)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(FIGS, "tuning_probes.png"))
    a = ap.parse_args()
    names, store = ev.settings_list()
    probes = sorted(os.path.basename(os.path.dirname(os.path.dirname(p)))
                    for p in glob.glob(os.path.join(STORE, "ref-*-tune-*", "exhaustive", "meta.json")))
    refs = [d for d in REFERENCE if os.path.exists(os.path.join(STORE, d, "exhaustive", "meta.json"))]
    t = pd.DataFrame([analyse(d, names, store) for d in refs + probes])
    t.to_csv(os.path.join(ev.RUNS, "tune_report.csv"), index=False)
    pd.set_option("display.width", 250)
    print(t[["probe", "chunks", "GiB", "best_single", "opportunity_pct", "chunks_other_best_pct",
             "q1_pct", "q2_pct", "q3_pct", "q4_pct"]].round(1).to_string(index=False))
    plot(t, a.out)


def plot(t, out):
    """Opportunity per probe, and per quarter of the run."""
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 1.8 + 0.55 * len(t)), sharey=True)
    fig.patch.set_facecolor("white")
    y = np.arange(len(t))[::-1]
    a1.barh(y, t.opportunity_pct, 0.65, color=OPP)
    for yy, v, b in zip(y, t.opportunity_pct, t.best_single):
        a1.text(v, yy, f" {v:.1f}%  (best single: {b})", va="center", fontsize=8, color=INK)
    a1.set_yticks(y)
    a1.set_yticklabels([f"{p}\n({n} chunks, {g:.1f} GiB)" for p, n, g in
                        zip(t.probe, t.chunks, t.GiB)], fontsize=8.5, color=INK)
    a1.set_title("Opportunity: per-chunk best vs best single codec\n(balanced 4-tier cost, %)",
                 loc="left", fontsize=10.5, color=INK)
    a1.set_xlim(0, max(t.opportunity_pct.max() * 2.2, 1))
    colours = ["#9ecae1", "#4292c6", "#2171b5", "#08306b"]
    for k in range(4):
        a2.barh(y + 0.3 - 0.2 * k, t[f"q{k + 1}_pct"], 0.2, color=colours[k],
                label=["first quarter of the run", "second", "third", "last quarter"][k])
    a2.set_title("Opportunity by quarter of the run (early -> late, %)", loc="left",
                 fontsize=10.5, color=INK)
    a2.legend(frameon=False, fontsize=8, loc="lower right")
    for ax in (a1, a2):
        ax.grid(axis="x", color="#e6e9ed", lw=0.6)
        ax.tick_params(labelsize=8, colors=INK2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    H = 1.8 + 0.55 * len(t)
    fig.suptitle("Which simulation settings make one codec not the best?", x=0.01,
                 ha="left", fontsize=13, color=INK, y=1 - 0.05 / H)
    fig.text(0.01, 1 - 0.45 / H, "Small probe runs of LAMMPS and Nyx with different "
             "physics settings (tune_sims.sh), each searched exhaustively through Clio "
             "(all 45 settings on every 4 MiB chunk); the full-size workloads at the top "
             "for reference.", fontsize=8.5, color=INK2, va="top")
    fig.subplots_adjust(left=0.17, right=0.99, top=1 - 1.25 / H, bottom=0.3 / H, wspace=0.08)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print("wrote", os.path.abspath(out))


if __name__ == "__main__":
    main()
