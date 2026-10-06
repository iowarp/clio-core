#!/usr/bin/env python3
"""Compare tuning probes in one figure, from their stored exhaustive searches
(no run).

    plot_probes.py DATASET[=LABEL] ... --out PNG [--title TEXT]

Top: possible gain (per-chunk best against the best single codec, the same
cost model for both) under the tuning model (1/1/1, each chunk's tier), the
benchmark model (1/10/10 at 1 GB/s) and the best realistic model of opp_grid.py
(w_io >= w_dt, the best single is a codec); dashed: the 15% target;
dotted: the 10% mark to keep a workload.
Bottom: possible gain per field under each probe's best realistic model.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "model-accuracy"))
import opp_grid as og  # noqa: E402
import plot_style as style  # noqa: E402

MODEL_COLORS = ("#8d99ae", "#e09f3e", "#2e86ab")


def best_realistic(ct, dt, st, tier, names):
    """@return (gain %, best single name, (w_dt, w_io, bw)) of the best realistic model."""
    best = None
    for w_dt in og.W_DT:
        for w_io in og.W_IO:
            if w_io < w_dt or w_io == 0:
                continue
            for bw in og.BW:
                g, bf = og.gain(ct, dt, st, tier, w_dt, w_io, bw)
                if names[bf] != "store" and (best is None or g > best[0]):
                    best = (g, names[bf], (w_dt, w_io, bw))
    return best


def probe_numbers(ds):
    """@return dict with the three model gains and the per-field gains of one probe."""
    fields, ct, dt, st, raw, tier, names, _ = og.matrices(ds)
    tun, tb = og.gain(ct, dt, st, tier, 1.0, 1.0, "tiers")
    ben, bb = og.gain(ct, dt, st, tier, 10.0, 10.0, 1.0)
    real, rname, model = best_realistic(ct, dt, st, tier, names)
    per = {}
    for f in sorted(set(fields)):
        k = fields == f
        per[f] = og.gain(ct[k], dt[k], st[k], tier[k], *model)[0]
    return {"tuning": (tun, names[tb]), "bench": (ben, names[bb]), "real": (real, rname),
            "model": model, "fields": per, "GB": raw.sum() / 1e9, "chunks": len(raw)}


def bar_label(ax, x, y, text, color):
    """A small label above one bar."""
    ax.text(x, y + 0.4, text, ha="center", va="bottom", fontsize=8, color=color,
            fontweight="bold")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("probes", nargs="+", help="DATASET or DATASET=LABEL")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Tuning probes: possible gain of per-chunk codec choice")
    a = ap.parse_args()
    probes = [p.split("=", 1) if "=" in p else (p, p) for p in a.probes]
    nums = [probe_numbers(ds) for ds, _ in probes]
    style.apply()
    fig, ax = plt.subplots(2, 1, figsize=(max(10, 3.2 * len(probes)), 9.5),
                           gridspec_kw={"height_ratios": [1, 1.1]})
    x = np.arange(len(probes))
    w = 0.26
    keys = (("tuning", "tuning model 1/1/1, chunk tiers"), ("bench", "benchmark model 1/10/10 at 1 GB/s"),
            ("real", "best realistic model"))
    for j, (key, lab) in enumerate(keys):
        ys = [n[key][0] for n in nums]
        ax[0].bar(x + (j - 1) * w, ys, w, color=MODEL_COLORS[j], label=lab)
        for xi, y in zip(x + (j - 1) * w, ys):
            bar_label(ax[0], xi, y, f"{y:.1f}%", style.darker(MODEL_COLORS[j], 0.75))
    ax[0].axhline(15, color="#d1495b", ls="--", lw=1.2, label="15% target")
    ax[0].axhline(10, color="#d1495b", ls=":", lw=1.2, label="10% keep mark")
    ticks = []
    for (ds, lab), n in zip(probes, nums):
        wd, wi, bw = n["model"]
        bwt = "chunk tiers" if bw == "tiers" else f"{bw:g} GB/s"
        ticks.append(f"{lab}\n{n['chunks']} chunks, {n['GB']:.1f} GB\n"
                     f"best realistic: 1/{wd:g}/{wi:g} at {bwt}\nbest single {n['real'][1]}")
    ax[0].set_xticks(x, ticks, fontsize=8.5)
    ax[0].set_ylabel("possible gain (%)")
    ax[0].set_ylim(0, max(20, max(n[k][0] for n in nums for k, _ in keys) + 4))
    ax[0].set_title("Possible gain: per-chunk best setting against the best single codec")
    ax[0].legend(loc="upper left", bbox_to_anchor=(1.005, 1.0))
    fields = sorted({f for n in nums for f in n["fields"]})
    fx = np.arange(len(fields))
    fw = 0.8 / len(probes)
    colors = plt.get_cmap("tab10").colors
    for j, ((ds, lab), n) in enumerate(zip(probes, nums)):
        ys = [n["fields"].get(f, np.nan) for f in fields]
        ax[1].bar(fx + (j - (len(probes) - 1) / 2) * fw, ys, fw, color=colors[j % 10], label=lab)
    ax[1].axhline(15, color="#d1495b", ls="--", lw=1.2, label="15% target")
    ax[1].axhline(10, color="#d1495b", ls=":", lw=1.2, label="10% keep mark")
    ax[1].set_xticks(fx, fields, fontsize=9)
    ax[1].set_ylabel("possible gain (%)")
    ax[1].set_title("Possible gain per field, under each probe's best realistic model")
    ax[1].legend(loc="upper left", bbox_to_anchor=(1.005, 1.0))
    style.titles(fig, a.title, "Possible gain = (best single cost - sum of each chunk's cheapest "
                 "setting) / best single cost; lossless; from the exhaustive search of every chunk.")
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print("wrote", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
