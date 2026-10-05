#!/usr/bin/env python3
"""One what-if scenario next to the measured producer + k-means runs.

    plot_whatif_vs_measured.py DATASET [--scenario A] [--tag km4] [--png PNG]

Inputs:
  runs/whatif_<DATASET>.csv        costmodel_whatif.py: the offline replay of
                                   the scenario (modelled times on one tier)
  runs/<DATASET>_<tag>_compare.csv compare_kmeans_runs.py: the measured Clio
                                   runs (best single codec, NeuroPress
                                   learning, oracle) with the same weights
Panels:
  1  what-if: modelled time per method (write + R reads + prediction)
  2  measured: application time on the local NVMe (write + R reads + k-means)
  3  modelled time on the PFS: the what-if against the same model fed with
     the measured runs' own compress / decompress times and stored bytes
  4  compression ratio: what-if against measured
Each bar is labelled with its value and its change against the best single
codec of the same panel and source.
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
# What-if method -> the measured run's mode (None: no measured counterpart).
WHATIF = (("best single setting", "best single codec\n(lowest cost)", "fixed"),
          ("NeuroPress as trained", "NeuroPress as trained\n(no learning)", None),
          ("NeuroPress learning, pass 1", "NeuroPress learning", "learn"),
          ("NeuroPress learning, pass 5", "NeuroPress learning,\n5th pass", None),
          ("oracle (per-chunk best)", "oracle\n(each chunk's best\nby cost)", "oracle"))
COLORS = {"fixed": "#4c72b0", "learn": "#dd8452", "oracle": "#55a868", None: "#bbbbbb"}

def app_dir(ds):
    """@return the figure directory of a workload's simulation:
    sim-tuning/vpic, sim-tuning/nyx, ... (created), or nn-v2 for other
    workloads."""
    for app in ("vpic", "nyx", "lammps", "warpx"):
        if app in ds:
            d = os.path.join(FIG, "..", "sim-tuning", app)
            os.makedirs(d, exist_ok=True)
            return d
    return FIG



def label_bars(ax, xs, vals, base, fmt):
    """Writes value and change against `base` above each bar."""
    for x, v in zip(xs, vals):
        pct = 100 * (v / base - 1)
        extra = "" if abs(pct) < 1e-9 else f"\n{pct:+.1f}%"
        ax.text(x, v, fmt.format(v) + extra, ha="center", va="bottom", fontsize=8.5)


def panel_whatif(ax, w, reads):
    """Panel 1: the what-if's modelled time, split into write / reads / prediction."""
    rows = [w[w.method == m].iloc[0] for m, _, _ in WHATIF]
    x = np.arange(len(rows))
    wr = np.array([r.write_s for r in rows])
    rd = np.array([reads * r.read_s for r in rows])
    pr = np.array([r.predict_s for r in rows])
    ax.bar(x, wr, color="#9db4d6", width=0.62, label="write (compress + transfer)")
    ax.bar(x, rd, bottom=wr, color="#f2b48a", width=0.62,
           label=f"{reads} reads (transfer + decompress)")
    ax.bar(x, pr, bottom=wr + rd, color="#555555", width=0.62, label="NeuroPress prediction")
    label_bars(ax, x, [r.time_s for r in rows], rows[0].time_s, "{:.1f} s")
    ax.set_xticks(x, [n.replace("NeuroPress ", "NeuroPress\n").replace("best single ",
                                                                        "best single\n")
                      for _, n, _ in WHATIF], fontsize=8.5)
    ax.set_ylabel("seconds")
    ax.set_title("1. What-if (offline replay): modelled time on the PFS", fontsize=10)
    ax.legend(fontsize=8, loc="upper center", ncol=3)


def panel_measured(ax, m, reads):
    """Panel 2: measured application time on the local NVMe."""
    x = np.arange(len(m))
    parts = (("write_s", "producer write (compress + store)", "#4c72b0"),
             ("read_s", f"consumer: {reads} reads (get + decompress)", "#dd8452"),
             ("kmeans_s", f"consumer: k-means ({reads} iterations)", "#55a868"))
    bottom = np.zeros(len(m))
    for col, lab, c in parts:
        ax.bar(x, m[col], bottom=bottom, color=c, width=0.55, label=lab)
        bottom += m[col].to_numpy()
    label_bars(ax, x, m.e2e_s, m.e2e_s.iat[0], "{:.1f} s")
    ax.set_xticks(x, [n for _, n, mode in WHATIF if mode in set(m["mode"])], fontsize=8.5)
    ax.set_ylabel("seconds")
    ax.set_title("2. Measured through Clio: application time (local NVMe)", fontsize=10)
    ax.legend(fontsize=8, loc="upper center", ncol=3)


def panel_pairs(ax, w, m, col_w, col_m, title, fmt, ylabel):
    """Panels 3 and 4: what-if against measured, per measured method."""
    modes = [(wm, name, mode) for wm, name, mode in WHATIF if mode in set(m["mode"])]
    x = np.arange(len(modes))
    wv = np.array([w[w.method == wm].iloc[0][col_w] for wm, _, _ in modes])
    mv = np.array([m[m["mode"] == mode].iloc[0][col_m] for _, _, mode in modes])
    ax.bar(x - 0.2, wv, width=0.38, color="#bbbbbb", label="what-if (offline replay)")
    ax.bar(x + 0.2, mv, width=0.38, color=[COLORS[md] for _, _, md in modes],
           label="measured runs")
    label_bars(ax, x - 0.2, wv, wv[0], fmt)
    label_bars(ax, x + 0.2, mv, mv[0], fmt)
    ax.set_xticks(x, [n for _, n, _ in modes], fontsize=8.5)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="upper center", ncol=3)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dataset")
    ap.add_argument("--scenario", default="A")
    ap.add_argument("--tag", default="km4")
    ap.add_argument("--png", default=None)
    ap.add_argument("--best-scenario", default=None,
                    help="take the best single setting (= fastest by runtime) from this "
                         "scenario, e.g. A for 1 write + 4 reads, when the main scenario's "
                         "cost is not the runtime (same tier and reads required)")
    ap.add_argument("--whatif-csv", default=None,
                    help="what-if CSV [runs/whatif_<DATASET>.csv]")
    a = ap.parse_args()
    w = pd.read_csv(a.whatif_csv or os.path.join(RUNS, f"whatif_{a.dataset}.csv"))
    if a.best_scenario:
        b = w[(w.scenario == a.best_scenario) & (w.method == "best single setting")]
        main = w[w.scenario == a.scenario]
        if (int(b.reads.iat[0]) != int(main.reads.iat[0])
                or str(b.bw_bytes_per_ms.iat[0]) != str(main.bw_bytes_per_ms.iat[0])):
            raise SystemExit(f"{a.best_scenario} has another tier or read count")
        w = pd.concat([b.assign(scenario=a.scenario, best_single=b.best_single.iat[0]),
                       main[main.method != "best single setting"]])
    w = w[w.scenario == a.scenario]
    m = pd.read_csv(os.path.join(RUNS, f"{a.dataset}_{a.tag}_compare.csv"))
    sc = w[w.method != "best single setting"].iloc[0]   # the scenario's own weights
    best_name = w[w.method == "best single setting"].best_single.iat[0]
    reads = int(sc.reads)
    if int(m.reads.iat[0]) != reads:
        raise SystemExit(f"what-if {a.scenario} has {reads} reads, the runs {m.reads.iat[0]}")
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    panel_whatif(ax[0, 0], w, reads)
    panel_measured(ax[0, 1], m, reads)
    bw = float(sc.bw_bytes_per_ms)
    panel_pairs(ax[1, 0], w, m, "time_s", "model_pfs_s",
                f"3. Modelled time on the PFS ({bw / 1e6:.2f} GB/s): what-if against "
                f"the same model\nfed with the measured runs' compress / decompress "
                f"times and stored bytes", "{:.1f} s", "seconds")
    panel_pairs(ax[1, 1], w, m, "ratio", "ratio",
                "4. Compression ratio (higher is better)", "{:.3f}x", "ratio")
    for x in ax.flat:
        x.set_ylim(0, x.get_ylim()[1] * 1.3)
    gib = sc.bytes_scored / 2**30
    fig.suptitle(
        f"{a.dataset} ({gib:.2f} GiB), scenario {a.scenario}: the producer writes each chunk "
        f"once, a k-means consumer reads it {reads} times.\nCost = {sc.w_ct:g} x compress + "
        f"{sc.w_dt:g} x decompress + {sc.w_io:g} x stored bytes / {bw / 1e6:.2f} GB/s "
        f"(one PFS tier); every option selects by this cost.\nBest single codec = the "
        f"setting with the lowest total cost: {best_name}. Oracle = each chunk's lowest-cost "
        f"setting. Percent = change against the best single codec.",
        fontsize=10.5)
    fig.tight_layout()
    png = a.png or os.path.join(app_dir(a.dataset), f"v2_{a.dataset}_whatif{a.scenario}_vs_measured.png")
    fig.savefig(png, dpi=130)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
