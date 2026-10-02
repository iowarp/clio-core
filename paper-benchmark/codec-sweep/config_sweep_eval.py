#!/usr/bin/env python3
"""Every lossless codec setting, with and without byte / bit shuffle.

Reads run_config_sweep.sh's outputs (np-codec-sweep/config/<wl>-<grid>-
<shuffle>-<stamp>/configs.csv), totals each setting over the sampled 4 MiB
chunks (ratio = input / compressed bytes; GB/s = input / summed CUDA-event
time), and draws

  config_shuffle_effect.png  per dataset and codec: the best ratio it reaches
                             with no shuffle, byte shuffle and bit shuffle
  config_frontier.png        per dataset: every setting's ratio vs compress
                             throughput, coloured by shuffle, with each
                             shuffle mode's ratio-speed frontier

A setting counts only if every sampled chunk round-trips bit-exactly.

  config_sweep_eval.py [--root DIR] [--stamp S] [--wl a,b] [--out DIR]
"""
import argparse
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import cost_model_tiers as cm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = "/projects/bekn/imuradli/np-codec-sweep/config"
DEFAULT_OUT = os.path.join(HERE, "..", "figures", "config-sweep")
SHUFFLE = {"none": ("No shuffle", cm.BLUE), "byte": ("Byte shuffle", cm.ORANGE),
           "bit": ("Bit shuffle", "#1baf7a")}
BASES = ["lz4", "snappy", "zstd", "gdeflate", "deflate", "ans", "cascaded",
         "bitcomp", "ndzip", "gpulz", "spspeed", "spratio"]
LABEL = {"lz4": "LZ4", "snappy": "Snappy", "zstd": "Zstd", "gdeflate": "GDeflate",
         "deflate": "Deflate", "ans": "ANS", "cascaded": "Cascaded",
         "bitcomp": "Bitcomp", "ndzip": "ndzip", "gpulz": "GPULZ",
         "spspeed": "SPspeed", "spratio": "SPratio"}
SH_RE = re.compile(r"(^| )shuffle=(\w+)")


def load(root, stamp):
    """Every row of every workload's config runs.

    @return DataFrame with wl, algorithm, setting (shuffle removed), shuffle
    """
    frames = []
    for f in glob.glob(os.path.join(root, f"*-{stamp}", "configs.csv")):
        m = re.match(r"(.+)-(main|slow)-(none|byte|bit)-", os.path.basename(
            os.path.dirname(f)))
        d = pd.read_csv(f)
        d["wl"] = m.group(1)
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    s = d.settings.fillna("")
    d["shuffle"] = s.str.extract(SH_RE)[1].fillna("none")
    d["setting"] = s.str.replace(SH_RE, "", regex=True).str.strip()
    return d


def totals(d):
    """Per (wl, algorithm, setting, shuffle): ratio and GB/s over all chunks."""
    g = d.groupby(["wl", "algorithm", "setting", "shuffle"]).agg(
        n=("ok", "size"), ok=("ok", "sum"), bytes=("bytes", "sum"),
        comp_bytes=("comp_bytes", "sum"), comp_ms=("comp_ms", "sum"),
        decomp_ms=("decomp_ms", "sum")).reset_index()
    g = g[g.ok == g.n].copy()
    g["ratio"] = g.bytes / g.comp_bytes
    g["comp_GBps"] = g.bytes / 1e6 / g.comp_ms
    g["decomp_GBps"] = g.bytes / 1e6 / g.decomp_ms
    return g


def frontier(g):
    """Settings no other setting beats on both ratio and compress speed."""
    g = g.sort_values("comp_GBps", ascending=False)
    best, keep = 0.0, []
    for i, r in g.iterrows():
        if r.ratio > best:
            keep.append(i)
            best = r.ratio
    return g.loc[keep].sort_values("comp_GBps")


def plot_shuffle_effect(t, wls, out):
    """Grid of datasets: per codec, best ratio under each shuffle mode."""
    h = 2.3 * len(wls) + 1.2  # inches: panels plus header
    fig, axes = plt.subplots(len(wls), 1, figsize=(13, h), sharex=True)
    fig.patch.set_facecolor(cm.SURFACE)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.55 / h,
                        top=1 - 0.95 / h, hspace=0.35)
    x = np.arange(len(BASES))
    for ax, wl in zip(np.atleast_1d(axes), wls):
        cm.style_axes(ax)
        b = t[t.wl == wl].groupby(["algorithm", "shuffle"]).ratio.max()
        for i, (sh, (lab, col)) in enumerate(SHUFFLE.items()):
            v = [b.get((base, sh), np.nan) for base in BASES]
            ax.bar(x + (i - 1) * 0.27, v, 0.27, color=col, edgecolor=cm.SURFACE,
                   label=lab)
        ax.set_ylabel("Best ratio", color=cm.INK2, fontsize=9)
        ax.set_title(cm.WORKLOADS.get(wl, wl), loc="left", fontsize=10,
                     color=cm.INK)
    axes = np.atleast_1d(axes)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([LABEL[b] for b in BASES], fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, fontsize=9,
               labelcolor=cm.INK2, loc="upper left",
               bbox_to_anchor=(0.06, 1 - 0.45 / h))
    fig.suptitle("Best ratio each codec reaches over all its settings, with and "
                 "without shuffle (32 sampled 4 MiB chunks per dataset)",
                 x=0.01, ha="left", fontsize=12, color=cm.INK, y=1 - 0.1 / h)
    fig.savefig(out, dpi=170, facecolor=cm.SURFACE)
    plt.close(fig)


def plot_frontier(t, wls, out):
    """Per dataset: all settings, ratio vs compress GB/s, frontier per mode."""
    cols = min(4, len(wls))
    rows = -(-len(wls) // cols)
    h = 4.0 * rows + 1.5  # inches: panels plus a fixed header and footer
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols + 1, h))
    fig.patch.set_facecolor(cm.SURFACE)
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.7 / h, top=1 - 1.15 / h,
                        wspace=0.25, hspace=0.45)
    axes = np.atleast_1d(axes).ravel()
    for ax, wl in zip(axes, wls):
        cm.style_axes(ax)
        tw = t[t.wl == wl]
        for sh, (lab, col) in SHUFFLE.items():
            s = tw[tw.shuffle == sh]
            ax.scatter(s.comp_GBps, s.ratio, s=7, color=col, alpha=0.35,
                       linewidths=0)
            f = frontier(s)
            ax.step(f.comp_GBps, f.ratio, where="post", color=col,
                    linewidth=1.8, label=lab)
        best = tw.loc[tw.ratio.idxmax()]
        ax.annotate(f"{LABEL.get(best.algorithm, best.algorithm)} "
                    f"({best.shuffle}) {best.ratio:.2f}x",
                    (best.comp_GBps, best.ratio), xytext=(-4, 4),
                    textcoords="offset points", fontsize=7.5, color=cm.INK,
                    ha="right")
        ax.set_xscale("log")
        ax.set_title(cm.WORKLOADS.get(wl, wl), loc="left", fontsize=10,
                     color=cm.INK)
        ax.set_xlabel("Compress throughput (GB/s, log)", color=cm.INK2,
                      fontsize=8.5)
        ax.set_ylabel("Ratio", color=cm.INK2, fontsize=8.5)
    for ax in axes[len(wls):]:
        ax.set_visible(False)
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=cm.INK2)
    fig.suptitle("Every lossless setting (nvCOMP options x 3 internal chunk "
                 "sizes, ndzip 1-D/2-D/3-D, GPULZ, SPspeed, SPratio) x shuffle",
                 x=0.01, ha="left", fontsize=12, color=cm.INK, y=1 - 0.12 / h)
    fig.text(0.01, 1 - 0.55 / h, "Lines: the ratio-speed frontier of each shuffle mode "
             "(no setting of that mode is both faster and stronger). Totals over "
             "32 evenly spaced 4 MiB chunks; A100 CUDA-event times.",
             fontsize=9, color=cm.INK2)
    fig.savefig(out, dpi=170, facecolor=cm.SURFACE)
    plt.close(fig)


def main():
    """Load, total, print the per-codec shuffle table, draw both figures."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--stamp", default="10021052")
    ap.add_argument("--wl", default="", help="workloads to include (all)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()
    t = totals(load(a.root, a.stamp))
    if a.wl:
        t = t[t.wl.isin(a.wl.split(","))]
    wls = [w for w in cm.WORKLOADS if w in set(t.wl)]
    best = t.groupby(["wl", "algorithm", "shuffle"]).ratio.max().unstack()
    with pd.option_context("display.width", 200, "display.max_rows", None):
        print(best[["none", "byte", "bit"]].round(3).to_string())
        top = t.sort_values("ratio", ascending=False).groupby("wl").head(3)
        print(top[["wl", "algorithm", "setting", "shuffle", "ratio",
                   "comp_GBps", "decomp_GBps"]].round(3).to_string(index=False))
    os.makedirs(a.out, exist_ok=True)
    plot_shuffle_effect(t, wls, os.path.join(a.out, "config_shuffle_effect.png"))
    plot_frontier(t, wls, os.path.join(a.out, "config_frontier.png"))
    print(f"wrote {a.out}/config_shuffle_effect.png, config_frontier.png")


if __name__ == "__main__":
    main()
