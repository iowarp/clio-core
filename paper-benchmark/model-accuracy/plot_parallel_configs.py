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

RUNS = "/mnt/nvme0/v2-work/runs"
OPTIONS = (("fixed", "best single codec"), ("learn", "NeuroPress learning"),
           ("oracle", "oracle (each chunk's best)"))


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


def selection(ds, procs, inflight, mode, prefix, w):
    """@return {setting spec or "stored raw": chunks} of one option, all processes.

    The setting is each chunk's primary row of v2_measured.csv; a chunk the
    codec did not shrink was stored raw (blobs.csv lib 0) and counts as such.
    """
    out = {}
    for run in glob.glob(os.path.join(RUNS, f"{ds}-p*_{mode}_{prefix}p{procs}i{inflight}w{w}")):
        m = pd.read_csv(os.path.join(run, "v2_measured.csv"), usecols=["blob", "spec", "role"])
        m = m[m.role == "primary"].drop_duplicates("blob").set_index("blob").spec
        b = pd.read_csv(os.path.join(run, "blobs.csv"), usecols=["blob", "lib"]).set_index("blob")
        spec = m.reindex(b.index).fillna("unknown")
        spec[b.lib == 0] = "stored raw"
        for k, n in spec.value_counts().items():
            out[k] = out.get(k, 0) + int(n)
    return out


def panel_selection(a, sel, keys, top=6):
    """Bottom panel: each bar's chunks by codec setting, as shares (100%).

    @param sel  {(config, mode): {setting: chunks}}
    @param keys the configurations, in x order
    @param top  settings shown by name (most chunks over all bars); the rest
                are "other settings"
    """
    total = {}
    for d in sel.values():
        for k, n in d.items():
            if k != "stored raw":
                total[k] = total.get(k, 0) + n
    named = sorted(total, key=total.get, reverse=True)[:top]
    cats = named + ["other settings", "stored raw"]
    palette = ["#4e79a7", "#f28e2b", "#59a14f", "#b07aa1", "#76b7b2", "#edc948",
               "#d4d4d4", "#2b2b2b"]
    x = np.arange(len(keys))
    width = 0.26
    for k, (mode, _) in enumerate(cpr.MODES):
        pos = x + (k - 1) * width
        bottom = np.zeros(len(keys))
        for c, color in zip(cats, palette):
            v = []
            for key in keys:
                d = sel[(key, mode)]
                n = sum(d.values()) or 1
                cnt = (sum(m for s2, m in d.items() if s2 not in named and s2 != "stored raw")
                       if c == "other settings" else d.get(c, 0))
                v.append(100.0 * cnt / n)
            v = np.array(v)
            a.bar(pos, v, width, bottom=bottom, color=color, edgecolor="white", linewidth=0.6,
                  label=c if k == 0 else None)
            for p, y0, h in zip(pos, bottom, v):
                if h >= 8:
                    a.text(p, y0 + h / 2, f"{h:.0f}%", ha="center", va="center", fontsize=7.5,
                           color="white" if color in ("#4e79a7", "#2b2b2b", "#b07aa1") else "#222222")
            bottom += v
        for p in pos:
            a.text(p, 101.5, {"fixed": "B", "learn": "N", "oracle": "O"}[mode], ha="center",
                   va="bottom", fontsize=8, fontweight="bold", color=style.OPTION_COLORS[mode])
    a.set_ylim(0, 108)
    a.set_ylabel("chunks (%)")
    a.set_title("Codec setting of each chunk (B = best single, N = NeuroPress learning, "
                "O = oracle)")
    a.legend(loc="upper left", bbox_to_anchor=(1.005, 1.0), fontsize=8.5, title="setting",
             title_fontsize=8.5, alignment="left")


def xlabels(t, configs):
    """@return the tick text per configuration: P x I, copies, timed builds."""
    out = []
    for p, i in configs:
        s = t[(t.procs == p) & (t.inflight == i)].set_index("mode")
        builds = " / ".join(str(int(s.codec_builds_in_timed.get(m, 0))) for m, _ in OPTIONS)
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
    fig, ax = plt.subplots(3, 1, figsize=(max(12, 2.6 * len(configs)), 14), sharex=True,
                           gridspec_kw={"height_ratios": [1, 0.8, 0.9]})
    names = dict(OPTIONS)
    cpr.bar_options(ax[:2], t, keys, names, reads)
    ax[0].set_ylabel("application time (s)")
    ax[0].set_title("Application time: first timed write to last k-means iteration "
                    "(lower is better)")
    ax[1].set_ylabel("compression ratio")
    ax[1].set_title("Compression ratio (higher is better)")
    sel = {(key, mode): selection(ds, p, i, mode, prefix, w)
           for key, (p, i) in zip(keys, configs) for mode, _ in OPTIONS}
    panel_selection(ax[2], sel, keys)
    ax[2].set_xticks(np.arange(len(configs)), xlabels(t, configs), fontsize=9)
    wt = w.replace("-", "/")
    style.titles(fig, f"{ds}: processes \u00d7 chunks in flight",
                 f"Cost model {wt} at {bw / 1e6:g} GB/s; 1 write + {reads} reads, each read followed by "
                 f"one k-means iteration. % = change against the best single codec of the same "
                 f"configuration.\nTimed builds = codec objects built inside the timed work "
                 f"(best single / NeuroPress / oracle).")
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
    t.to_csv(os.path.join(RUNS, f"{a.dataset}_configs_w{a.w}.csv"), index=False)
    cols = ["procs", "inflight", "mode", "app_s", "ratio", "prewarm_per_setting",
            "codec_builds_in_timed", "digest_ok", "app_s_vs_best_single_pct",
            "ratio_vs_best_single_pct"]
    with pd.option_context("display.width", 200):
        print(t[cols].round(2).to_string(index=False))
    plot(t, a.dataset, a.w, a.out, a.prefix, a.reads, a.bw)


if __name__ == "__main__":
    main()
