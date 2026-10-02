#!/usr/bin/env python3
"""Hybrid storage: SZ3 4-D for slow-moving chunks, GPU lossless for the rest.

Joins sz3_hybrid.py's per-(window, field, chunk) SZ3 results with the
lossless codec sweep's per-chunk results for the same files and chunks. A
unit is one 4 MiB chunk across a window of T dumps.

  lossless       the unit's T chunks with one GPU lossless codec (bytes and
                 CUDA-event times averaged over the sweep's repetitions)
  SZ3 <layout>   the unit with SZ3, using the configuration with the best
                 total ratio for that layout (3d, 4d-tslow, 4d-tfast)
  hybrid(f)      the fraction f of units that change least (lowest
                 score_rms) stored with SZ3, the rest lossless

Units with no valid SZ3 result (a constant field: eb = 0) stay lossless.
Times: lossless = GPU, SZ3 = one CPU thread per unit; throughput = input
bytes / summed time, so SZ3 parts are what ONE core sustains.

Writes figures/sz3-hybrid/<wl>_hybrid.png and prints the tables.

  sz3_hybrid_eval.py --wl nyx --sz3 SZ3.csv [--codec spratio] [--out DIR]
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import cost_model_tiers as cm  # noqa: E402

DEFAULT_OUT = os.path.join(HERE, "..", "..", "figures", "sz3-hybrid")
LAYOUTS = {"4d-tslow": ("4-D, time slowest", cm.BLUE),
           "4d-tfast": ("4-D, time fastest", "#1baf7a"),
           "3d": ("3-D, per dump", cm.ORANGE)}


def units(sz3):
    """One row per unit with its change score and file list.

    @param sz3 sz3_hybrid.py rows
    @return DataFrame indexed by (window, field, chunk)
    """
    key = ["window", "field", "chunk"]
    return sz3.groupby(key)[["files", "score_rms", "ident_frac"]].first()


def lossless_units(u, ll, codec):
    """Bytes and GPU times of the unit's T chunks under one lossless codec.

    @param u     output of units
    @param ll    cost_model_tiers.load_workload rows
    @param codec lossless codec name
    """
    c = ll[ll.codec == codec].set_index(["file", "chunk"])
    rows = []
    for (w, f, k), r in u.iterrows():
        keys = [(p, k) for p in r.files.split(";")]
        sub = c.loc[keys]
        rows.append({"window": w, "field": f, "chunk": k,
                     "bytes_in": sub.bytes.sum(), "ll_bytes": sub.comp_bytes.sum(),
                     "ll_comp": sub.comp_ms.sum(), "ll_dec": sub.decomp_ms.sum()})
    return pd.DataFrame(rows).set_index(["window", "field", "chunk"])


def best_configs(sz3):
    """Per layout, the SZ3 configuration with the best total ratio.

    Only configurations valid on every unit that has SZ3 results compete.
    @return {layout: config}, and the per-(config, layout) totals table
    """
    v = sz3[sz3.layout.notna()]
    n_units = v.groupby(["window", "field", "chunk"]).ngroups
    t = v.groupby(["config", "layout"]).agg(
        units=("ok", "size"), ok=("ok", "sum"), bytes_in=("bytes_in", "sum"),
        bytes_out=("bytes_out", "sum"), comp=("comp_ms", "sum"),
        dec=("decomp_ms", "sum"), max_err=("max_err_over_eb", "max"))
    t["ratio"] = t.bytes_in / t.bytes_out
    t["comp_MBps"] = t.bytes_in / 1e3 / t.comp
    t["dec_MBps"] = t.bytes_in / 1e3 / t.dec
    valid = t[(t.ok == n_units) & (t.units == n_units)]
    best = {lay: valid.xs(lay, level="layout").ratio.idxmax()
            for lay in LAYOUTS if lay in valid.index.get_level_values("layout")}
    return best, t


def curve(u, ll_u, sz3, layout, config, fractions):
    """Overall ratio and throughput as the slowest units move to SZ3.

    @return DataFrame per fraction: lossy share, ratio, comp/dec GB/s
    """
    s = sz3[(sz3.layout == layout) & (sz3.config == config)].set_index(
        ["window", "field", "chunk"])
    d = ll_u.join(u[["score_rms"]]).join(
        s[["bytes_out", "comp_ms", "decomp_ms"]], how="left")
    d = d.sort_values("score_rms")
    can = d.bytes_out.notna().to_numpy()
    out = []
    for f in fractions:
        n = int(round(f * len(d)))
        lossy = np.zeros(len(d), bool)
        lossy[:n] = True
        lossy &= can
        b = np.where(lossy, d.bytes_out, d.ll_bytes).sum()
        tc = np.where(lossy, d.comp_ms, d.ll_comp).sum()
        td = np.where(lossy, d.decomp_ms, d.ll_dec).sum()
        tot = d.bytes_in.sum()
        out.append({"fraction": f, "lossy_share": d.bytes_in[lossy].sum() / tot,
                    "ratio": tot / b, "comp_GBps": tot / 1e6 / tc,
                    "dec_GBps": tot / 1e6 / td,
                    "score_cut": d.score_rms.iloc[n - 1] if n else 0.0})
    return pd.DataFrame(out)


def gain_by_score(u, sz3, best, bins=10):
    """4-D over 3-D ratio gain per change-score decile (best configs)."""
    k = ["window", "field", "chunk"]
    b3 = sz3[(sz3.layout == "3d") & (sz3.config == best["3d"])].set_index(k)
    out = {}
    for lay in ("4d-tslow", "4d-tfast"):
        b4 = sz3[(sz3.layout == lay) & (sz3.config == best[lay])].set_index(k)
        d = u[["score_rms"]].join(b3.bytes_out.rename("b3")).join(
            b4.bytes_out.rename("b4")).dropna()
        d["bin"] = pd.qcut(d.score_rms.rank(method="first"), bins, labels=False)
        g = d.groupby("bin").agg(b3=("b3", "sum"), b4=("b4", "sum"),
                                 score=("score_rms", "median"))
        g["gain_pct"] = 100 * (g.b3 / g.b4 - 1)
        out[lay] = g
    return out


def plain_log(ax):
    """Log y axis labelled with plain numbers, 1-3 or 1-2-5 steps by span."""
    lo, hi = ax.get_ylim()
    subs = (1, 3) if np.log10(hi / lo) > 1.5 else (1, 2, 5)
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=subs))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def plot(wl, curves, gains, ll_codec, out):
    """Three panels: 4-D gain by score, ratio and throughput vs lossy share."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.9))
    fig.patch.set_facecolor(cm.SURFACE)
    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.14, top=0.78, wspace=0.27)
    for ax in axes:
        cm.style_axes(ax)
    a0, a1, a2 = axes
    x = np.arange(10)
    for i, (lay, g) in enumerate(gains.items()):
        a0.bar(x + (i - 0.5) * 0.38, g.gain_pct, 0.38, color=LAYOUTS[lay][1],
               edgecolor=cm.SURFACE, label=LAYOUTS[lay][0])
    a0.axhline(0, color=cm.AXIS, linewidth=0.8)
    a0.set_xticks(x)
    a0.set_xticklabels([f"{i + 1}" for i in x], fontsize=8.5)
    a0.set_xlabel("Change-score decile (1 = slowest-moving tenth)", color=cm.INK2)
    a0.set_ylabel("4-D size saving over 3-D (%)", color=cm.INK2)
    a0.set_title("Does 4-D help where data moves slowly?", loc="left",
                 fontsize=10, color=cm.INK)
    a0.legend(frameon=False, fontsize=8.5, labelcolor=cm.INK2)
    for lay, c in curves.items():
        pct = 100 * c.lossy_share
        a1.plot(pct, c.ratio, color=LAYOUTS[lay][1], linewidth=2,
                label=LAYOUTS[lay][0])
        a2.plot(pct, c.comp_GBps, color=LAYOUTS[lay][1], linewidth=2)
        a2.plot(pct, c.dec_GBps, color=LAYOUTS[lay][1], linewidth=1.5,
                linestyle=(0, (4, 2)))
    a1.set_yscale("log")
    a1.set_xlabel("Share of data stored lossy (slowest-moving first, %)",
                  color=cm.INK2)
    a1.set_ylabel("Overall compression ratio (log)", color=cm.INK2)
    a1.set_title("Ratio as slow chunks move to SZ3", loc="left", fontsize=10,
                 color=cm.INK)
    a1.legend(frameon=False, fontsize=8.5, labelcolor=cm.INK2)
    a2.set_yscale("log")
    a2.set_xlabel("Share of data stored lossy (slowest-moving first, %)",
                  color=cm.INK2)
    a2.set_ylabel("Throughput (GB/s, log)", color=cm.INK2)
    a2.set_title("Speed: compress (solid), decompress (dashed)", loc="left",
                 fontsize=10, color=cm.INK)
    for ax in (a1, a2):
        ax.set_xlim(0, 100)
        plain_log(ax)
    name = cm.WORKLOADS.get(wl, wl)
    fig.suptitle(f"{name}: SZ3 4-D (REL 1e-3) on slow-moving chunks, "
                 f"{cm.CODECS.get(ll_codec, ll_codec)} lossless on the rest",
                 x=0.01, ha="left", fontsize=12.5, color=cm.INK, y=0.985)
    fig.text(0.01, 0.885, "Unit = one 4 MiB chunk over a window of 4 dumps; "
             "change score = largest RMS change between consecutive dumps / "
             "field RMS. Lossless times are GPU (A100), SZ3 times one CPU core; "
             "each SZ3 layout uses its best-ratio configuration.",
             fontsize=8.8, color=cm.INK2)
    fig.savefig(out, dpi=180, facecolor=cm.SURFACE)
    plt.close(fig)


def main():
    """Load both sides, print the tables, draw the figure."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--wl", required=True)
    ap.add_argument("--sz3", required=True, help="sz3_hybrid.py CSV")
    ap.add_argument("--codec", default="spratio", help="lossless codec")
    ap.add_argument("--sweep", default=cm.DEFAULT_SWEEP)
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()
    sz3 = pd.read_csv(a.sz3)
    u = units(sz3)
    ll_u = lossless_units(u, cm.load_workload(a.sweep, a.wl), a.codec)
    best, table = best_configs(sz3)
    fr = np.linspace(0, 1, 41)
    curves = {lay: curve(u, ll_u, sz3, lay, cfg, fr) for lay, cfg in best.items()}
    gains = gain_by_score(u, sz3, best)
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(f"{a.wl}: {len(u)} units, lossless = {a.codec}")
        print(table.sort_values("ratio", ascending=False)[
            ["ok", "units", "ratio", "comp_MBps", "dec_MBps", "max_err"]].to_string())
        print("best per layout:", best)
        for lay, c in curves.items():
            print(lay)
            print(c.iloc[::4].to_string(index=False))
        for lay, g in gains.items():
            print(lay, "gain % by decile:", g.gain_pct.round(1).tolist())
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, f"{a.wl}_hybrid.png")
    plot(a.wl, curves, gains, a.codec, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
