#!/usr/bin/env python3
"""What-if over many cost models: NeuroPress learning and the oracle against
the best single codec, from np_cost_sweep.py's table (no run).

    plot_cost_sweep.py DATASET [--reads 4,10] [--out PNG]

One row of panels per read count R. In each panel, x = the cost weights
w_ct / w_dt / w_io, y = the bandwidth of the cost model; each cell is one cost
model, and its colour and number give the change against the best single
codec of that cost model (all options select by the same cost):
  NeuroPress learning: modelled runtime %, ratio %
  oracle: modelled runtime %
Modelled runtime = compress + R x decompress + (1 + R) x stored bytes /
bandwidth (+ NeuroPress's prediction time), from the stored exhaustive search;
NeuroPress = an offline replay of Clio's selection with one learning pass.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUNS = "/mnt/nvme0/v2-work/runs"
PANELS = (("np_learn1_runtime_pct", "NeuroPress learning: runtime vs best single (%)", "RdYlGn_r"),
          ("np_learn1_ratio_pct", "NeuroPress learning: ratio vs best single (%)", "RdYlGn"),
          ("oracle_runtime_pct", "Oracle: runtime vs best single (%)", "RdYlGn_r"))


def wlabel(r):
    """@return the weights of a sweep row, e.g. 1/10/11."""
    return "/".join(f"{v:g}" for v in (r.w_ct, r.w_dt, r.w_io))


def panel(ax, t, col, title, cmap, lim):
    """One heat map: bandwidth x weights, annotated with the values."""
    t = t.assign(w=[wlabel(r) for r in t.itertuples()])
    order = list(dict.fromkeys(t.sort_values(["w_dt", "w_io", "w_ct"]).w))
    grid = t.pivot_table(index="bw_GBs", columns="w", values=col).reindex(columns=order)
    im = ax.imshow(grid.values, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    for (i, j), v in np.ndenumerate(grid.values):
        if np.isfinite(v):
            ax.text(j, i, f"{v:+.0f}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(order)), order, rotation=60, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(grid.index)), [f"{b:g} GB/s" for b in grid.index], fontsize=8)
    ax.set_title(title, fontsize=9.5)
    return im


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dataset")
    ap.add_argument("--reads", default="4,10")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    t = pd.read_csv(os.path.join(RUNS, f"np_cost_sweep_{a.dataset}.csv"))
    reads = [int(x) for x in a.reads.split(",")]
    fig, axes = plt.subplots(len(reads), len(PANELS), figsize=(7.2 * len(PANELS), 5.2 * len(reads)),
                             squeeze=False)
    for i, r in enumerate(reads):
        s = t[t.reads == r]
        for j, (col, title, cmap) in enumerate(PANELS):
            lim = 40 if "ratio" in col else 30
            im = panel(axes[i, j], s, col, f"{r} reads -- {title}", cmap, lim)
            fig.colorbar(im, ax=axes[i, j], fraction=0.035, pad=0.02)
            axes[i, j].set_xlabel("cost weights w_ct / w_dt / w_io", fontsize=8)
    fig.suptitle(
        f"{a.dataset}: what-if over cost models (modelled from the stored exhaustive search; "
        f"no run). Every option selects by the cell's cost model; % = change against that "
        f"cost model's best single codec.\nRuntime = compress + R x decompress + (1 + R) x "
        f"stored bytes / bandwidth (+ NeuroPress's prediction); NeuroPress = Clio's selection "
        f"replayed with one learning pass. Green = better.", fontsize=10.5)
    fig.tight_layout()
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures",
                                "new-workloads", "sim-tuning", "nyx",
                                f"whatif_cost_models_{a.dataset}.png")
    fig.savefig(out, dpi=120)
    print("wrote", os.path.abspath(out))


if __name__ == "__main__":
    main()
