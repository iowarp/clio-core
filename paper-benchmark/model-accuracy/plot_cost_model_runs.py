#!/usr/bin/env python3
"""Measured runs under several cost models, side by side: NeuroPress learning
and the oracle against the best single codec of the same cost model.

    plot_cost_model_runs.py DATASET [--out PNG]

Reads every runs/DATASET_km<R>b<BW>gp<P>i<I>w<W>_compare.csv of
run_kmeans_parallel.sh (the 2-process tables hold the 1-process reference
rows too) and the modelled values of np_cost_sweep.py
(runs/np_cost_sweep_DATASET.csv: NeuroPress learning replayed for one pass and
the oracle, modelled runtime at the cost model's bandwidth). Per cost model:
application time and ratio of NeuroPress learning and the oracle, % against
the best single codec, for each configuration measured, and the modelled
value as a marker.
"""
import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import compare_kmeans_runs as ck

TAG = re.compile(r"_km(\d+)b([\d.]+)gp(\d+)i(\d+)w([\d.\-]+)_compare\.csv$")


def load(ds):
    """One row per (cost model, configuration, option) with % vs best single."""
    rows = []
    for f in sorted(glob.glob(os.path.join(ck.RUNS, f"{ds}_km*b*gp*i*w*_compare.csv"))):
        m = TAG.search(f)
        if not m:
            continue
        reads, bw = int(m.group(1)), float(m.group(2))
        t = pd.read_csv(f)
        for _, r in t[t["mode"] != "fixed"].iterrows():
            rows.append({"w": m.group(5).replace("-", "/"), "bw": bw, "reads": reads,
                         "config": r.config, "mode": r["mode"],
                         "app_pct": r.app_s_vs_best_single_pct,
                         "ratio_pct": r.ratio_vs_best_single_pct})
    # a reference configuration is in several tables: keep it once
    return pd.DataFrame(rows).drop_duplicates(["w", "bw", "reads", "config", "mode"])


def modelled(ds, t):
    """{(w, bw, reads, mode): (runtime %, ratio %)} from np_cost_sweep.py."""
    f = os.path.join(ck.RUNS, f"np_cost_sweep_{ds}.csv")
    if not os.path.exists(f):
        return {}
    s = pd.read_csv(f)
    out = {}
    for _, r in s.iterrows():
        w = "/".join(f"{x:g}" for x in (r.w_ct, r.w_dt, r.w_io))
        for mode, key in (("learn", "np_learn1"), ("oracle", "oracle")):
            out[(w, float(r.bw_GBs), int(r.reads), mode)] = (r[f"{key}_runtime_pct"],
                                                              r[f"{key}_ratio_pct"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dataset")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    t = load(a.dataset)
    if t.empty:
        raise SystemExit("no runs")
    model = modelled(a.dataset, t)
    cms = t[["w", "bw", "reads"]].drop_duplicates().sort_values(["reads", "bw", "w"]).values.tolist()
    configs = sorted(t.config.unique(), key=lambda c: (int(c.split()[0]), "16" in c, c))
    fig, ax = plt.subplots(2, 1, figsize=(max(10, 2.3 * len(cms) + 3), 10))
    shades = {"learn": ("#c44e52", "#e8a0a2", "#8c1c22", "#f3cfd0"),
              "oracle": ("#4c72b0", "#a1b9de", "#1f3f73", "#d3def0")}
    colors = {(m, i): shades[m][i % 4] for m in shades for i in range(len(configs))}
    width = 0.8 / (2 * len(configs))
    x = np.arange(len(cms))
    for k, (col, title) in enumerate((("app_pct", "Measured application time vs the best single "
                                                  "codec (%; < 0 = faster)"),
                                      ("ratio_pct", "Compression ratio vs the best single codec "
                                                    "(%; > 0 = smaller)"))):
        for j, (mode, name) in enumerate((("learn", "NeuroPress learning"), ("oracle", "oracle"))):
            for c, cfg in enumerate(configs):
                pos = x + (j * len(configs) + c - (2 * len(configs) - 1) / 2) * width
                vals = [t[(t.w == w) & (t.bw == bw) & (t.reads == r) & (t["mode"] == mode)
                          & (t.config == cfg)][col] for w, bw, r in cms]
                v = np.array([s.iat[0] if len(s) else np.nan for s in vals])
                ax[k].bar(pos, v, width, color=colors[(mode, c)], label=f"{name}, {cfg} measured")
                for p, y in zip(pos, v):
                    if not np.isnan(y):
                        ax[k].text(p, y, f"{y:+.0f}", ha="center",
                                   va="bottom" if y >= 0 else "top", fontsize=7.5)
            mk = [model.get((w, bw, r, mode), (np.nan, np.nan))[k] for w, bw, r in cms]
            ax[k].scatter(x + (j * len(configs) + (len(configs) - 1) / 2
                               - (2 * len(configs) - 1) / 2) * width, mk, marker="D", s=28,
                          color="black" if mode == "learn" else "#555555", zorder=5,
                          label=f"{name}, modelled from the CSV (PFS at the cost model's bandwidth)")
        ax[k].axhline(0, color="#333333", linewidth=0.8)
        ax[k].set_title(title, fontsize=11)
        ax[k].set_xticks(x, [f"weights {w}\n{bw:g} GB/s, {r} reads" for w, bw, r in cms], fontsize=9)
        ax[k].set_ylabel("% vs best single codec")
    ax[1].legend(fontsize=8, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.16))
    fig.suptitle(f"{a.dataset}: NeuroPress learning and the oracle against the best single codec "
                 f"of the same cost model (all three select by it).\nBars: measured runs (local NVMe, "
                 f"data in the file cache); diamonds: modelled runtime and ratio from the stored "
                 f"exhaustive search.", fontsize=10.5)
    fig.tight_layout()
    out = a.out or os.path.join(ck.app_dir(a.dataset), f"v2_{a.dataset}_cost_models.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
