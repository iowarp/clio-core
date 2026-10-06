#!/usr/bin/env python3
"""All parallel k-means configurations of one dataset in one figure, from the
compare CSVs of run_kmeans_parallel.sh (no run).

    plot_parallel_configs.py DATASET --out PNG [--w 1-10-10] [--exclude 8x12,...]

Reads runs/DATASET_<TAG_PREFIX>p<P>i<I>w<W>_compare.csv for every P x I that
exists, and draws per configuration the best single codec, NeuroPress
learning and the oracle:
  top     application time (s); NeuroPress and oracle as % against the best
          single of the same configuration
  bottom  compression ratio, with the same %
The configs CSV also holds each option's model cost (the cost every option
selects by, of the settings each run stored, from the exhaustive search).
Under each configuration: the codec copies per setting built before the timed
work and the codec builds inside the timed work (best single / NeuroPress /
oracle). A bar whose bit-exact check failed is marked in red.
Output: the figure and runs/DATASET_configs_w<W>.csv.
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

import compare_parallel_runs as cpr
import plot_style as style
import replay_learning as rl

RUNS = "/mnt/nvme0/v2-work/runs"
OPTIONS = (("fixed", "best single codec"), ("learn", "NeuroPress learning"),
           ("oracle", "oracle (each chunk's best)"), ("hcompress", "HCompress"))


def present(t):
    """@return the options of OPTIONS that have rows in t, in OPTIONS order."""
    have = set(t["mode"])
    return [(m, n) for m, n in OPTIONS if m in have]


def load(ds, w, prefix):
    """@return one row per (configuration, option) from all compare CSVs."""
    pat = os.path.join(RUNS, f"{ds}_{prefix}p*i*w{w}_compare.csv")
    parts = []
    for f in sorted(glob.glob(pat)):
        m = re.search(r"p(\d+)i(\d+)w", os.path.basename(f)[len(ds):])
        t = pd.read_csv(f)
        procs, infl = int(m.group(1)), int(m.group(2))
        # a compare CSV also holds its reference configurations: keep its own
        own = t.config.str.match(rf"^{procs} process(es)?, {infl} chunks? in flight")
        parts.append(t[own].assign(procs=procs, inflight=infl))
    if not parts:
        raise SystemExit(f"no compare CSV matches {pat}")
    t = pd.concat(parts, ignore_index=True).sort_values(["procs", "inflight"])
    return t.drop_duplicates(["procs", "inflight", "mode"], keep="last")


def model_costs(t, ds, prefix, w, bw):
    """Add model_cost and model_cost_vs_best_single_pct to t.

    The model cost of one option is the cost model's total (w_ct x compress +
    w_dt x decompress + w_io x stored bytes / bw) over the settings its run
    stored, every value from the stored exhaustive search: the quantity the
    oracle minimizes. A chunk stored raw costs its raw I/O; a setting that did
    not shrink the chunk costs its compress time plus raw I/O.

    @param t      one row per (configuration, option)
    @param prefix tag prefix of the runs (km10b1g, ...)
    @param w      weights as in the tag, e.g. "1-40-2.8"
    @param bw     cost-model bandwidth, B/ms
    @return t with the two columns (NaN where a run's settings are missing)
    """
    names, store, order, meas, _ = rl.load(ds)
    idx = {b: i for i, b in enumerate(order.blob)}
    nb = order.bytes.to_numpy(float)
    w_ct, w_dt, w_io = (float(v) for v in w.split("-"))
    cost = []
    for p, i, mode in zip(t.procs, t.inflight, t["mode"]):
        pick = np.full(len(order), -1)
        for run in glob.glob(os.path.join(RUNS, f"{ds}-p*_{mode}_{prefix}p{p}i{i}w{w}")):
            m = pd.read_csv(os.path.join(run, "v2_measured.csv"), usecols=["blob", "setting", "role"])
            m = m[m.role == "primary"].drop_duplicates("blob").set_index("blob").setting
            b = pd.read_csv(os.path.join(run, "blobs.csv"), usecols=["blob", "lib"])
            for blob, lib in zip(b.blob, b.lib):
                if blob in idx:
                    pick[idx[blob]] = store if lib == 0 else m.get(blob, -1)
        if (pick < 0).any():
            cost.append(np.nan)
            continue
        k = np.arange(len(pick))
        ct, dt, r = meas[k, pick].T
        raw = (pick == store) | ~(r > 1)
        stored = np.where(raw, nb, nb / np.where(raw, 1.0, r))
        c = (w_ct * np.where(pick == store, 0.0, ct) + w_dt * np.where(raw, 0.0, dt)
             + w_io * stored / bw)
        cost.append(c.sum() / 1e3)   # weighted seconds
    t = t.assign(model_cost=cost)
    best = t[t["mode"] == "fixed"].set_index(["procs", "inflight"]).model_cost
    ref = [best.get((p, i), np.nan) for p, i in zip(t.procs, t.inflight)]
    return t.assign(model_cost_vs_best_single_pct=100.0 * (t.model_cost / ref - 1.0))


def xlabels(t, configs):
    """@return the tick text per configuration: P x I, copies, timed builds."""
    out = []
    for p, i in configs:
        s = t[(t.procs == p) & (t.inflight == i)].set_index("mode")
        builds = " / ".join(str(int(s.codec_builds_in_timed.get(m, 0))) for m, _ in present(t))
        copies = int(s.prewarm_per_setting.iat[0])
        out.append(f"{p} process{'es' if p > 1 else ''} \u00d7 {i} in flight\n"
                   f"{copies} codec cop{'y' if copies == 1 else 'ies'} per setting\n"
                   f"timed builds {builds}")
    return out


def plot(t, ds, w, out, prefix="km10b1g", reads=10, bw=1e6):
    """Application time and ratio per configuration, three options each."""
    style.apply()
    configs = list(dict.fromkeys(zip(t.procs, t.inflight)))
    keys = [t[(t.procs == p) & (t.inflight == i)].config.iat[0] for p, i in configs]
    fig, ax = plt.subplots(2, 1, figsize=(max(12, 2.6 * len(configs)), 10), sharex=True,
                           gridspec_kw={"height_ratios": [1, 0.8]})
    names = dict(OPTIONS)
    cpr.bar_options(ax[:2], t, keys, names, reads)
    ax[0].set_ylabel("application time (s)")
    ax[0].set_title("Application time: first timed write to last k-means iteration "
                    "(lower is better)")
    ax[1].set_ylabel("compression ratio")
    ax[1].set_title("Compression ratio (higher is better)")
    wt = w.replace("-", "/")
    ax[1].set_xticks(np.arange(len(configs)), xlabels(t, configs), fontsize=9)
    style.titles(fig, f"{ds}: processes \u00d7 chunks in flight",
                 f"Cost model {wt} at {bw / 1e6:g} GB/s; 1 write + {reads} reads, each read followed by "
                 f"one k-means iteration. % = change against the best single codec of the same "
                 f"configuration.\nTimed builds = codec objects built inside the timed work "
                 f"({' / '.join(n.split(' (')[0] for _, n in present(t))}).")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("wrote", os.path.abspath(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dataset")
    ap.add_argument("--w", default="1-10-10")
    ap.add_argument("--prefix", default="km10b1g")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reads", type=int, default=10, help="timed reads per run")
    ap.add_argument("--bw", type=float, default=1e6, help="cost-model bandwidth, B/ms")
    ap.add_argument("--exclude", default="",
                    help="configurations to leave out, e.g. 8x12,2x16 (processes x in flight)")
    a = ap.parse_args()
    t = load(a.dataset, a.w, a.prefix)
    for c in filter(None, a.exclude.split(",")):
        p, i = (int(v) for v in c.lower().split("x"))
        t = t[~((t.procs == p) & (t.inflight == i))]
    t = model_costs(t, a.dataset, a.prefix, a.w, a.bw)
    t.to_csv(os.path.join(RUNS, f"{a.dataset}_configs_w{a.w}.csv"), index=False)
    cols = ["procs", "inflight", "mode", "app_s", "ratio", "prewarm_per_setting",
            "codec_builds_in_timed", "digest_ok", "app_s_vs_best_single_pct",
            "ratio_vs_best_single_pct", "model_cost", "model_cost_vs_best_single_pct"]
    with pd.option_context("display.width", 200):
        print(t[cols].round(2).to_string(index=False))
    plot(t, a.dataset, a.w, a.out, a.prefix, a.reads, a.bw)


if __name__ == "__main__":
    main()
