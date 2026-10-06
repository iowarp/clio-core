#!/usr/bin/env python3
"""Why the cost-model oracle is not always the fastest measured option: the
cost model against the measured runs, from the stored CSVs (no run).

    plot_model_vs_measured.py DATASET --out PNG [--w 1,10,10] [--bw 1000000]

A  the model (stored exhaustive search: one process, no load): seconds per
   option, compress + 10 x decompress + 11 x stored bytes / bandwidth
   (1 write + 10 reads), NeuroPress = Clio's selection replayed offline
B  per configuration: the measured application time against the best single
   codec (bars) and the change the model predicts (dashed lines)
C  per configuration, oracle and NeuroPress learning: the measured sums over
   every timed read of every chunk (runs/<DATASET>-p<i>_<mode>_<tag>/
   phases.csv; the untimed last check read is left out) against the best
   single codec: bytes read, I/O time, decompress time and the whole read
   time per chunk. The processes and the chunks in flight overlap, so these
   sums say where the time goes, not seconds of application time.
Output: the figure and runs/DATASET_model_vs_measured.csv.
"""
import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sim-tuning"))
import costmodel_whatif as cw  # noqa: E402
import plot_style as style  # noqa: E402

RUNS = "/mnt/nvme0/v2-work/runs"
READS = 10
COLORS = style.OPTION_COLORS
NAMES = {"fixed": "best single", "learn": "NeuroPress learning", "oracle": "oracle"}
METRICS = (("bytes", "bytes read"), ("io", "I/O time"), ("dec", "decompress time"),
           ("wall", "whole read time"))


def model(ds, w, bw):
    """@return {mode: (compress s, 10x decompress s, 11x I/O s)} from the
    stored exhaustive search, and the best single setting's name."""
    names, store, order, meas, _ = cw.rl.load(ds)
    nb = order.bytes.to_numpy(float)
    bwv = np.full(len(nb), bw)
    ct, dt, st = cw.outcomes(meas, nb, store)
    cost = cw.truth_cost(meas, nb, store, w, bwv)
    ok = ~np.isnan(cost).any(axis=1)
    sel = cw.ev.for_selection(np.where(np.isnan(cost), np.inf, cost))
    best = int(np.argmin(np.where(ok[:, None], sel, 0).sum(axis=0)))
    picks = {"fixed": np.full(len(nb), best), "oracle": np.argmin(sel, axis=1)}
    for m, p, _ in cw.replays(ds, (names, store, order, meas, cost), {"w": w, "bw": None}, 1):
        if m == cw.LEARN1:
            picks["learn"] = p
    r = np.arange(ok.sum())
    out = {}
    for mode, p in picks.items():
        p = p[ok].astype(int)
        out[mode] = (ct[ok][r, p].sum() / 1e3, READS * dt[ok][r, p].sum() / 1e3,
                     (READS + 1) * (st[ok][r, p] / bwv[ok]).sum() / 1e3)
    return out, names[best]


def read_sums(ds, procs, inflight, mode):
    """@return the sums over every timed read of every chunk of one option
    (bytes, I/O, decompress and whole read time, s), or None."""
    files = glob.glob(os.path.join(RUNS, f"{ds}-p*_{mode}_km10b1gp{procs}i{inflight}w*/phases.csv"))
    if len(files) != procs:
        return None
    parts = []
    for f in files:
        t = pd.read_csv(f, usecols=["chunk_id", "path", "decompress_ms", "io_ms", "wall_ms",
                                    "stored_bytes"])
        t = t[t.path == "read"]
        # the last read of each chunk is the untimed bit-exact check
        t = t[t.groupby("chunk_id").cumcount() < READS]
        parts.append(t)
    t = pd.concat(parts)
    return {"bytes": t.stored_bytes.sum(), "io": t.io_ms.sum() / 1e3,
            "dec": t.decompress_ms.sum() / 1e3, "wall": t.wall_ms.sum() / 1e3}


def label(a, p, y, text, color="#333333", bold=False):
    """A value label at the end of a bar (above it, or below a negative one)."""
    a.annotate(text, (p, y), xytext=(0, -2 if y < 0 else 2), textcoords="offset points",
               ha="center", va="top" if y < 0 else "bottom", fontsize=8, color=color,
               fontweight="bold" if bold else "normal",
               bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none"))


def panel_model(a, m):
    """Panel A: the modelled seconds per option, split in its three parts."""
    modes = ["fixed", "learn", "oracle"]
    x = np.arange(len(modes))
    bottom = np.zeros(len(modes))
    parts = (("compress", 0.45), (f"{READS} \u00d7 decompress", 0.7),
             (f"{READS + 1} \u00d7 I/O at the model's bandwidth", 1.0))
    for k, (_, shade) in enumerate(parts):
        v = np.array([m[mode][k] for mode in modes])
        a.bar(x, v, 0.62, bottom=bottom, edgecolor="white", linewidth=0.8,
              color=[style.darker(COLORS[mode], shade) if shade < 1 else COLORS[mode]
                     for mode in modes])
        bottom += v
    for xi, mode, tot in zip(x, modes, bottom):
        if mode == "fixed":
            label(a, xi, tot, f"{tot:.0f} s")
        else:
            label(a, xi, tot, style.pct(100 * (tot / bottom[0] - 1)), COLORS[mode], True)
    a.set_xticks(x, [NAMES[mode] for mode in modes])
    a.set_ylim(0, bottom.max() * 1.5)
    a.set_ylabel("modelled seconds")
    a.set_title("A. Cost model (one process, no load)")
    handles = [matplotlib.patches.Patch(facecolor=style.darker("#9aa0a6", sh) if sh < 1
                                        else "#9aa0a6", label=lab) for lab, sh in parts]
    a.legend(handles=handles, loc="upper right", fontsize=8.5)


def panel_measured(a, cfg, mpct, labels):
    """Panel B: measured % vs best single per configuration; model as lines."""
    x = np.arange(len(labels))
    width = 0.34
    for k, mode in enumerate(("learn", "oracle")):
        v = cfg[mode].to_numpy()
        pos = x + (k - 0.5) * width
        a.bar(pos, v, width, color=COLORS[mode], edgecolor="white", linewidth=0.8,
              label=f"{NAMES[mode]}, measured")
        a.axhline(mpct[mode], color=COLORS[mode], ls=(0, (5, 3)), lw=1.6,
                  label=f"{NAMES[mode]}, model: {style.pct(mpct[mode])}")
        for p, y in zip(pos, v):
            label(a, p, y, style.pct(y), COLORS[mode], True)
    a.axhline(0, color="#555555", lw=0.9)
    lo = min(cfg[["learn", "oracle"]].min().min(), min(mpct.values()))
    a.set_ylim(lo * 1.4, max(3, -lo * 0.12))
    a.set_xticks(x, labels)
    a.set_ylabel("application time vs best single (%)")
    a.set_title("B. Measured application time vs best single (lower is better)")
    a.legend(loc="lower right", ncol=2, fontsize=8.5)


def panel_reads(a, sums, mode, labels):
    """Panel C: one option's read-side sums vs best single, per configuration."""
    x = np.arange(len(labels))
    width = 0.2
    vals = []
    for k, (key, lab) in enumerate(METRICS):
        v = np.array([100 * (s[mode][key] / s["fixed"][key] - 1) for s in sums])
        vals += list(v)
        pos = x + (k - 1.5) * width
        a.bar(pos, v, width, color=style.METRIC_COLORS[key], edgecolor="white",
              linewidth=0.8, label=lab)
        for p, y in zip(pos, v):
            label(a, p, y, f"{y:+.0f}%")
    a.axhline(0, color="#555555", lw=0.9)
    a.set_ylim(min(min(vals) * 1.3, -5), max(max(vals) * 1.3, 4))
    a.set_xticks(x, labels)
    a.set_ylabel("vs best single (%)")
    a.set_title(f"C{'1' if mode == 'oracle' else '2'}. {NAMES[mode]}: measured sums over "
                f"all timed reads of all chunks, vs best single")
    a.legend(loc="lower right", ncol=4, fontsize=8.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dataset")
    ap.add_argument("--w", default="1,10,10")
    ap.add_argument("--bw", type=float, default=1e6, help="bytes per ms")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    w = tuple(float(v) for v in a.w.split(","))
    wl = "w" + a.w.replace(",", "-")
    m, best = model(a.dataset, w, a.bw)
    mtot = {mode: sum(v) for mode, v in m.items()}
    mpct = {mode: 100 * (mtot[mode] / mtot["fixed"] - 1) for mode in ("learn", "oracle")}
    cfg = pd.read_csv(os.path.join(RUNS, f"{a.dataset}_configs_{wl}.csv"))
    configs, labels, sums, rows = [], [], [], []
    for (p, i), g in cfg.groupby(["procs", "inflight"], sort=True):
        s = {mode: read_sums(a.dataset, p, i, mode) for mode in ("fixed", "learn", "oracle")}
        if any(v is None for v in s.values()):
            continue
        configs.append((p, i))
        labels.append(f"{p} \u00d7 {i}")
        sums.append(s)
        pct = g.set_index("mode").app_s_vs_best_single_pct
        for mode in ("fixed", "learn", "oracle"):
            rows.append({"procs": p, "inflight": i, "mode": mode,
                         "app_s_vs_best_pct": pct[mode], "model_pct": 0 if mode == "fixed"
                         else mpct[mode], **{f"read_{k}": v for k, v in s[mode].items()}})
    meas = pd.DataFrame(rows)
    meas.to_csv(os.path.join(RUNS, f"{a.dataset}_model_vs_measured.csv"), index=False)
    piv = meas.pivot_table(index=["procs", "inflight"], columns="mode",
                           values="app_s_vs_best_pct").loc[configs]
    style.apply()
    fig = plt.figure(figsize=(16, 15))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 2.2], hspace=0.32, wspace=0.16,
                          left=0.06, right=0.99, top=0.95, bottom=0.04)
    panel_model(fig.add_subplot(gs[0, 0]), m)
    panel_measured(fig.add_subplot(gs[0, 1]), piv, mpct, labels)
    panel_reads(fig.add_subplot(gs[1, :]), sums, "oracle", labels)
    panel_reads(fig.add_subplot(gs[2, :]), sums, "learn", labels)
    style.titles(fig, f"{a.dataset}: cost model against measured runs",
                 f"Cost model {a.w.replace(',', '/')} at {a.bw / 1e6:g} GB/s (best single = "
                 f"{best}); runs: 1 write + {READS} reads + k-means, processes \u00d7 chunks in "
                 f"flight on the x axes. The options select by the model; the runs read from "
                 f"this machine's page cache,\nso fewer stored bytes save less time than the "
                 f"model's I/O term says. Panel C sums overlap in time (processes and chunks "
                 f"in flight run together): they show where the time goes.")
    fig.savefig(a.out, dpi=150)
    print("wrote", os.path.abspath(a.out))
    with pd.option_context("display.width", 200):
        print(meas.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
