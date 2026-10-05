#!/usr/bin/env python3
"""What-if on a storage hierarchy: the best single codec, NeuroPress and the
oracle when each chunk is stored on the tier it is dealt to, from the stored
CSV files (no run).

    whatif_tiers.py DATASET [--w 1,1,1:1 --w 1,4,5:4 ...] [--out PNG]

The stored exhaustive search was made with the four tiers of
CLIO_NEUROPRESS_TIERS=12e6:1,1e6:3,0.5e6:3,0.25e6:3: chunks dealt in arrival
order 1:3:3:3 to DRAM 12 GB/s, NVMe 1 GB/s, burst buffer 0.5 GB/s and PFS
0.25 GB/s; each chunk's tier bandwidth is in its v2_pred.csv (tier_bw).
For each --w (weights w_ct,w_dt,w_io and the read count R after ':') every
option selects by the cost w_ct x ct + w_dt x dt + w_io x stored / tier_bw:
  best single  the one setting with the lowest total cost
  NeuroPress   Clio's selection replayed at each chunk's tier bandwidth with
               one learning pass (+ its prediction time)
  oracle       each chunk's lowest-cost setting
and is scored by its modelled runtime of 1 write + R reads (compress + R x
decompress + (1 + R) x stored bytes / tier bandwidth), its ratio and its cost.
Output: the figure and runs/whatif_tiers_DATASET.csv.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import costmodel_whatif as cw  # noqa: E402

rl, ev = cw.rl, cw.ev
OPTIONS = (("best single", "#7f7f7f"), ("NeuroPress learning", "#c44e52"),
           ("oracle", "#4c72b0"))


def score(data, w, reads, predict_s):
    """Rows of the four options for weights w and R reads."""
    names, store, order, meas, _ = data
    nb = order.bytes.to_numpy(float)
    bw = order.tier_bw.to_numpy(float)
    ct, dt, st = cw.outcomes(meas, nb, store)
    cost = cw.truth_cost(meas, nb, store, w, bw)
    ok = ~np.isnan(cost).any(axis=1)
    sel = ev.for_selection(np.where(np.isnan(cost), np.inf, cost))
    best = int(np.argmin(np.where(ok[:, None], sel, 0).sum(axis=0)))
    picks = {"best single": np.full(len(nb), best), "oracle": np.argmin(sel, axis=1)}
    for m, p, _ in cw.replays(DS,
                              (names, store, order, meas, cost), {"w": w, "bw": None}, 1):
        if m == cw.LEARN1:
            picks["NeuroPress learning"] = p
    r = np.arange(ok.sum())
    rows = []
    for name, _ in OPTIONS:
        p = picks[name][ok].astype(int)
        rt = (ct[ok][r, p].sum() + reads * dt[ok][r, p].sum()
              + (1 + reads) * (st[ok][r, p] / bw[ok]).sum()) / 1e3
        if name.startswith("NeuroPress"):
            rt += predict_s
        rows.append({"option": name, "best_single": names[best], "runtime_s": rt,
                     "ratio": nb[ok].sum() / st[ok][r, p].sum(),
                     "cost_s": cost[ok][r, p].sum() / 1e3})
    t = pd.DataFrame(rows)
    for c in ("runtime_s", "ratio", "cost_s"):
        t[f"{c}_vs_best_pct"] = 100 * (t[c] / t[c].iat[0] - 1)
    return t


def plot(t, ds, out):
    """Runtime and ratio per weight set, the four options side by side."""
    sets = list(dict.fromkeys(t.case))
    fig, ax = plt.subplots(1, 2, figsize=(16, 5.8))
    x = np.arange(len(sets))
    width = 0.26
    for k, (name, color) in enumerate(OPTIONS):
        s = t[t.option == name].set_index("case").reindex(sets)
        for a, col, fmt in ((ax[0], "runtime_s", "{:.0f} s"), (ax[1], "ratio", "{:.2f}x")):
            pos = x + (k - (len(OPTIONS) - 1) / 2) * width
            a.bar(pos, s[col], width, color=color, label=name)
            for p, v, pct in zip(pos, s[col], s[f"{col}_vs_best_pct"]):
                lab = fmt.format(v) if k == 0 else f"{pct:+.1f}%"
                a.text(p, v, lab, ha="center", va="bottom", fontsize=8)
    best = t[t.option == "best single"].set_index("case").reindex(sets).best_single
    same = best.nunique() == 1
    for a in ax:
        a.set_xticks(x, [c if same else f"{c}\n(best single: {b})" for c, b in zip(sets, best)],
                     fontsize=9)
        a.set_ylim(0, a.get_ylim()[1] * 1.25)
    ax[0].set_ylabel("seconds")
    ax[0].set_title("Modelled runtime: 1 write + R reads on the chunk's tier (lower is better)",
                    fontsize=10.5)
    ax[1].set_title("Compression ratio (higher is better)", fontsize=10.5)
    ax[1].legend(fontsize=8.5, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.2))
    fig.suptitle(f"{ds}: four-tier hierarchy (chunks dealt 1:3:3:3 to DRAM 12, NVMe 1, burst "
                 f"buffer 0.5, PFS 0.25 GB/s), from the stored exhaustive search (no run).\n"
                 f"All options select by the same weights (w_ct / w_dt / w_io) at each chunk's "
                 f"tier bandwidth; % = change against the best single codec"
                 + (f" ({best.iat[0]} in every case)." if same else "."), fontsize=10.5)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("wrote", os.path.abspath(out))


DS = None


def main():
    global DS
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dataset")
    ap.add_argument("--w", action="append",
                    default=None, help="w_ct,w_dt,w_io:R (default 1,1,1:1 1,4,5:4 1,10,11:10 1,10,10:10)")
    ap.add_argument("--predict-s", type=float, default=3.3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    DS = a.dataset
    data = rl.load(a.dataset)
    tiers = data[2].tier_bw.value_counts().sort_index(ascending=False)
    print("chunks per tier (B/ms):", tiers.to_dict())
    parts = []
    for spec in a.w or ["1,1,1:1", "1,4,5:4", "1,10,11:10", "1,10,10:10"]:
        wv, r = spec.split(":")
        w = tuple(float(x) for x in wv.split(","))
        t = score(data, w, int(r), a.predict_s)
        t.insert(0, "case", f"{'/'.join(f'{v:g}' for v in w)}, {r} read{'s' if int(r) > 1 else ''}")
        parts.append(t)
    t = pd.concat(parts, ignore_index=True)
    t.to_csv(os.path.join(rl.RUNS, f"whatif_tiers_{a.dataset}.csv"), index=False)
    with pd.option_context("display.width", 200):
        print(t.round(3).to_string(index=False))
    out = a.out or os.path.join(HERE, "..", "figures", "new-workloads", "sim-tuning", "nyx",
                                f"whatif_tiers_{a.dataset}.png")
    plot(t, a.dataset, out)


if __name__ == "__main__":
    main()
