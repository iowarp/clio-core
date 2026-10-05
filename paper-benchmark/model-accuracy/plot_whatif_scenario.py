#!/usr/bin/env python3
"""One what-if scenario of costmodel_whatif.py as a figure.

    plot_whatif_scenario.py DATASET --scenario E [--time-ref B] [--png PNG]

Input: runs/whatif_<DATASET>.csv. Panels:
  1  cost under the scenario's weights, per method
  2  modelled time with every part counted (compress + transfer on the
     write, transfer + decompress on each read, NeuroPress's prediction),
     whatever the weights; with --time-ref, a dashed line marks the
     fastest single codec by time: the best single codec of a scenario on
     the same tier and reads whose cost IS the modelled time (weights
     1 / R / 1 + R, e.g. B = 1/1/2 for 1 read)
  3  compression ratio
Percent = change against the scenario's best single codec (and, in panel 2,
also against the fastest single codec by time).
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUNS = "/mnt/nvme0/v2-work/runs"
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "../figures/new-workloads/nn-v2")
METHODS = (("best single setting", "best single\ncodec\n({})"),
           ("NeuroPress as trained", "NeuroPress\nas trained\n(no learning)"),
           ("NeuroPress learning, pass 1", "NeuroPress\nlearning"),
           ("NeuroPress learning, pass 5", "NeuroPress\nlearning,\n5th pass"),
           ("oracle (per-chunk best)", "oracle\n(each chunk's\nbest)"))
COLORS = ("#4c72b0", "#bbbbbb", "#dd8452", "#f2b48a", "#55a868")


def label_bars(ax, x, vals, base, fmt, ref=None):
    """Value and change against `base` (and `ref`) above each bar."""
    for xi, v in zip(x, vals):
        pct = 100 * (v / base - 1)
        text = fmt.format(v) + ("" if abs(pct) < 1e-9 else f"\n{pct:+.1f}%")
        if ref is not None:
            text += f"\n({100 * (v / ref - 1):+.1f}% vs fastest)"
        ax.text(xi, v, text, ha="center", va="bottom", fontsize=8.5,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dataset")
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--time-ref", default=None,
                    help="scenario whose cost is the modelled time (same tier and reads)")
    ap.add_argument("--png", default=None)
    a = ap.parse_args()
    t = pd.read_csv(os.path.join(RUNS, f"whatif_{a.dataset}.csv"))
    w = t[t.scenario == a.scenario]
    sc = w.iloc[0]
    rows = [w[w.method == m].iloc[0] for m, _ in METHODS]
    names = [n.format(sc.best_single.replace(" ", "\n")) for _, n in METHODS]
    reads = int(sc.reads)
    ref = None
    if a.time_ref:
        r = t[(t.scenario == a.time_ref) & (t.method == "best single setting")].iloc[0]
        if int(r.reads) != reads or r.bw_bytes_per_ms != sc.bw_bytes_per_ms:
            raise SystemExit(f"{a.time_ref} has another tier or read count than {a.scenario}")
        ref = (r.best_single, r.time_s)
    x = np.arange(len(rows))
    fig, ax = plt.subplots(1, 3, figsize=(20, 6.2), gridspec_kw={"width_ratios": [1, 1.25, 1]})

    cost = np.array([r.cost_ms / 1e3 for r in rows])
    ax[0].bar(x, cost, color=COLORS, width=0.62)
    label_bars(ax[0], x, cost, cost[0], "{:.1f}")
    ax[0].set_title(f"1. Cost under weights {sc.w_ct:g} / {sc.w_dt:g} / {sc.w_io:g}\n"
                    f"(compress / decompress / transfer), what NeuroPress minimizes",
                    fontsize=10)
    ax[0].set_ylabel("cost (weighted seconds)")

    wr = np.array([r.write_s for r in rows])
    rd = np.array([reads * r.read_s for r in rows])
    pr = np.array([r.predict_s for r in rows])
    ax[1].bar(x, wr, color="#9db4d6", width=0.62, label="write (compress + transfer)")
    ax[1].bar(x, rd, bottom=wr, color="#f2b48a", width=0.62,
              label=f"{reads} read(s) (transfer + decompress)")
    ax[1].bar(x, pr, bottom=wr + rd, color="#555555", width=0.62, label="NeuroPress prediction")
    time_s = np.array([r.time_s for r in rows])
    label_bars(ax[1], x, time_s, time_s[0], "{:.1f} s", ref[1] if ref else None)
    if ref:
        ax[1].axhline(ref[1], color="black", ls="--", lw=1, zorder=0,
                      label=f"fastest single codec by time: {ref[0]}, {ref[1]:.1f} s")
    ax[1].set_title("2. Modelled time on the PFS, every part counted\n"
                    "(the compress time too, which these weights ignore)", fontsize=10)
    ax[1].set_ylabel("seconds")
    ax[1].legend(fontsize=8, loc="upper right", ncol=1)

    ratio = np.array([r.ratio for r in rows])
    ax[2].bar(x, ratio, color=COLORS, width=0.62)
    label_bars(ax[2], x, ratio, ratio[0], "{:.3f}x")
    ax[2].set_title("3. Compression ratio (higher is better)", fontsize=10)
    ax[2].set_ylabel("ratio")
    for p in ax:
        p.set_xticks(x, names, fontsize=8.5)
        p.set_ylim(0, p.get_ylim()[1] * 1.35)
    gib = sc.bytes_scored / 2**30
    fig.suptitle(
        f"{a.dataset} ({gib:.2f} GiB), what-if scenario {a.scenario} (offline replay of Clio's "
        f"selection and learning): cost = {sc.w_ct:g} x compress + {sc.w_dt:g} x decompress + "
        f"{sc.w_io:g} x stored bytes / {float(sc.bw_bytes_per_ms) / 1e6:.2f} GB/s (one PFS tier), "
        f"1 write + {reads} read(s).\nBest single codec = lowest total cost under these weights: "
        f"{sc.best_single}. Percent = change against it"
        + (f"; '(vs fastest)' = change against the fastest single codec by time ({ref[0]})."
           if ref else "."), fontsize=10.5)
    fig.tight_layout()
    png = a.png or os.path.join(FIG, f"v2_{a.dataset}_whatif{a.scenario}.png")
    fig.savefig(png, dpi=130)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
