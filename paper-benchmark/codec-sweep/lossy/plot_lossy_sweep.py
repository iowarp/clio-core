#!/usr/bin/env python3
"""Plot a run_lossy_sweep.sh result set.

Reads ROOT/<workload>/<codec>.csv and draws

  lossy_overview.png         compression ratio, compression and decompression
                             throughput against the relative error bound, one
                             column per workload
  lossy_rate_distortion.png  PSNR against bit rate (32 / ratio)

A row counts only when the round trip succeeded and its maximum error is
within the bound up to TOL (100 ppm). The harness's own bound_ok is stricter
(eb + one float32 ulp of the field's largest value); Lorenzo-style codecs
accumulate float32 rounding past that by 3-85 ppm on a handful of fields
(full Nyx run), which is precision, not a broken bound. Real violations (e.g.
cuSZ-Hi "cr" on constant fields, 3072x) stay excluded. A point is drawn only
when EVERY field of that workload is valid for it (no partial averages).
Codecs with modes are drawn at the mode with the highest ratio on that
workload. Aggregates: ratio and
throughput over total bytes; PSNR is the mean over non-constant fields.

  plot_lossy_sweep.py ROOT [ROOT...] [--out DIR] [--title TEXT] [--exclude a,b]
"""
import argparse
import glob
import os
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullLocator  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "..", "..", "figures", "lossy-sweep", "full")
TOL = 1e-4  # float32 rounding slack on the bound; see the module docstring

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
# Fixed codec -> (label, colour, marker). Eight categorical slots in their
# validated order, then a ninth codec on the neutral ink with its own marker
# (hue x marker composite encoding, never a generated hue).
CODECS = {
    "fsz":    ("FSZ", "#2a78d6", "o"),
    "cuszp3": ("cuSZp3/VGC", "#eb6834", "s"),
    "cuszhi": ("cuSZ-Hi", "#1baf7a", "^"),
    "cuszi":  ("cuSZ-I", "#eda100", "D"),
    "pfpl":   ("PFPL", "#e87ba4", "v"),
    "fzgpu":  ("FZ-GPU", "#008300", "P"),
    "cuzfp":  ("cuZFP", "#4a3aa7", "X"),
    "cuszp2": ("cuSZp2", "#e34948", "h"),
    "cuszx":  ("cuSZx", INK2, "*"),
}
WORKLOADS = ["nyx", "vpic", "hurricane", "cesm-atm", "cesm-atm-3d",
             "scale-letkf", "qmcpack"]


def load(roots):
    """Every row of every <workload>/<codec>.csv under the roots. A later root
    replaces an earlier one's (workload, codec) CSV, so a rerun of one codec
    can stand in for a failed run without touching the original results.

    @param roots result-set directories, in increasing precedence
    @return the rows, with a 'valid' column
    """
    picked = {}
    for root in roots:
        for f in glob.glob(os.path.join(root, "*", "*.csv")):
            if os.path.getsize(f) > 200:
                picked[(os.path.basename(os.path.dirname(f)),
                        os.path.basename(f))] = f
    d = pd.concat([pd.read_csv(f) for f in picked.values()], ignore_index=True)
    d["valid"] = (d["ok"] == 1) & (d["err_over_eb"] <= 1 + TOL)
    return d


def aggregate(d):
    """Per (workload, codec, variant, eb) totals over valid rows.

    @return DataFrame with ratio, comp_GBs, decomp_GBs, psnr, bits
    """
    key = ["workload", "codec", "variant", "eb_rel"]
    total = d.groupby("workload")["file"].nunique().rename("total")
    good = d[d["valid"]].groupby(key)["file"].nunique().rename("good")
    full = good.reset_index().merge(total.reset_index())
    full = full[full["good"] == full["total"]][key]
    v = d[d["valid"]].merge(full)
    g = v.groupby(key)
    a = g.agg(inb=("bytes_in", "sum"), outb=("bytes_out", "sum"),
              ct=("comp_ms", "sum"), dt=("decomp_ms", "sum"),
              fields=("valid", "size")).reset_index()
    nc = v[v["constant"] == 0].groupby(
        ["workload", "codec", "variant", "eb_rel"])["psnr_db"].mean()
    a = a.merge(nc.rename("psnr").reset_index(), how="left")
    a["ratio"] = a["inb"] / a["outb"]
    a["comp_GBs"] = a["inb"] / 1e6 / a["ct"]
    a["decomp_GBs"] = a["inb"] / 1e6 / a["dt"]
    a["bits"] = 32 / a["ratio"]
    # keep each codec's highest-ratio mode per workload
    best = (a.groupby(["workload", "codec", "variant"])["ratio"].mean()
            .reset_index().sort_values("ratio")
            .drop_duplicates(["workload", "codec"], keep="last"))
    return a.merge(best[["workload", "codec", "variant"]])


def style(ax):
    """Recessive grid and axes on the chart surface."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=8.5)


def plain_log(ax):
    """Log y with plain-number ticks (1, 2, 5, 10, ...)."""
    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    ticks = [m * 10 ** e for e in range(-2, 5) for m in (1, 2, 5)
             if lo <= m * 10 ** e <= hi]
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(NullLocator())


def layout(n, codecs, sub, body_h):
    """Inch-based figure geometry, so few workloads still fit the header.

    @param n      panels per row (workloads)
    @param codecs codecs in the legend
    @param sub    subtitle text (wrapped to the width)
    @param body_h height of the panel area in inches
    @return (width, height, wrapped subtitle, legend columns, header inches)
    """
    w = max(3.0 * n + 1.0, 11.0)
    lines = textwrap.wrap(sub, int(w * 15))  # ~15 chars per inch at 9 pt
    ncol = max(1, min(len(codecs), int((w - 0.6) / 1.55)))
    rows = -(-len(codecs) // ncol)
    head = 0.45 + 0.19 * len(lines) + 0.2 + 0.27 * rows + 0.3
    return w, head + body_h, "\n".join(lines), ncol, head


def header(fig, h, title, sub, ncol, codecs):
    """Title, wrapped subtitle and the shared legend, top-down in inches."""
    y = 1 - 0.12 / h
    fig.text(0.01, y, title, fontsize=12.5, color=INK, va="top")
    y -= 0.38 / h
    fig.text(0.01, y, sub, fontsize=9, color=INK2, va="top")
    y -= (0.19 * (sub.count("\n") + 1) + 0.15) / h
    hd = [plt.Line2D([], [], color=CODECS[c][1], marker=CODECS[c][2], ms=6.5,
                     lw=2, mec=SURFACE, mew=0.8, label=CODECS[c][0])
          for c in codecs]
    fig.legend(handles=hd, loc="upper left", ncol=ncol, frameon=False,
               fontsize=9, labelcolor=INK, bbox_to_anchor=(0.005, y),
               handlelength=2.2, columnspacing=1.3)


def plot_overview(a, wls, codecs, title, sub, out):
    """Ratio and throughputs against the bound, one column per workload."""
    rows = [("ratio", "Compression ratio"),
            ("comp_GBs", "Compression GB/s"),
            ("decomp_GBs", "Decompression GB/s")]
    w, h, subw, ncol, head = layout(len(wls), codecs, sub, 7.6)
    fig, axes = plt.subplots(3, len(wls), figsize=(w, h), sharex=True,
                             squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.75 / w, right=1 - 0.15 / w,
                        top=1 - (head + 0.05) / h, bottom=0.6 / h,
                        hspace=0.22, wspace=0.28)
    ebs = sorted(a["eb_rel"].unique())
    for j, wl in enumerate(wls):
        for i, (m, label) in enumerate(rows):
            ax = axes[i, j]
            style(ax)
            for c in codecs:
                s = a[(a.workload == wl) & (a.codec == c)].sort_values("eb_rel")
                if s.empty:
                    continue
                ax.plot(s["eb_rel"], s[m], color=CODECS[c][1],
                        marker=CODECS[c][2], ms=6, lw=1.8, mec=SURFACE,
                        mew=0.8)
            plain_log(ax)
            if i == 0:
                ax.set_title(wl, loc="left", fontsize=10.5, color=INK)
            if j == 0:
                ax.set_ylabel(label, color=INK2, fontsize=9.5)
            if i == len(rows) - 1:
                ax.set_xticks(ebs)
                ax.set_xticklabels([f"{e:g}" for e in ebs])
                ax.set_xlabel("relative error bound", color=INK2, fontsize=9)
    header(fig, h, title, subw, ncol, codecs)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_rd(a, wls, codecs, title, sub, out):
    """PSNR against bit rate, one panel per workload."""
    w, h, subw, ncol, head = layout(len(wls), codecs,
                                    "Up and to the left is better. " + sub, 3.0)
    fig, axes = plt.subplots(1, len(wls), figsize=(w, h), squeeze=False)
    axes = axes[0]
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.75 / w, right=1 - 0.15 / w,
                        top=1 - (head + 0.05) / h, bottom=0.6 / h, wspace=0.3)
    for ax, wl in zip(axes, wls):
        style(ax)
        for c in codecs:
            s = a[(a.workload == wl) & (a.codec == c)].dropna(subset=["psnr"])
            s = s.sort_values("bits")
            if s.empty:
                continue
            ax.plot(s["bits"], s["psnr"], color=CODECS[c][1],
                    marker=CODECS[c][2], ms=6, lw=1.8, mec=SURFACE, mew=0.8)
        ax.set_xscale("log")
        lo, hi = ax.get_xlim()
        ax.set_xticks([m * 10 ** e for e in range(-2, 3) for m in (1, 2, 5)
                       if lo <= m * 10 ** e <= hi])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_title(wl, loc="left", fontsize=10.5, color=INK)
        ax.set_xlabel("bits per value (log)", color=INK2, fontsize=9)
    axes[0].set_ylabel("PSNR (dB)", color=INK2, fontsize=9.5)
    header(fig, h, title + ": rate-distortion", subw, ncol, codecs)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    """Load, aggregate, draw both figures, print what was excluded."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("roots", nargs="+",
                    help="result sets; later ones replace a (workload, codec)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--title", default="GPU lossy codecs")
    ap.add_argument("--note", default="", help="appended to the subtitle")
    ap.add_argument("--exclude", default="", help="codecs to leave out")
    a_ = ap.parse_args()
    d = load(a_.roots)
    d = d[~d["codec"].isin([c for c in a_.exclude.split(",") if c])]
    a = aggregate(d)
    wls = [w for w in WORKLOADS if w in set(a.workload)]
    codecs = [c for c in CODECS if c in set(a.codec)]
    n = d.groupby("workload")["file"].nunique()
    sub = ("Bound = eb x field value range (held within 100 ppm); fields per workload: " +
           ", ".join(f"{w} {n.get(w, 0)}" for w in wls) + ". A100." +
           (" " + a_.note if a_.note else ""))
    os.makedirs(a_.out, exist_ok=True)
    plot_overview(a, wls, codecs, a_.title, sub,
                  os.path.join(a_.out, "lossy_overview.png"))
    plot_rd(a, wls, codecs, a_.title, sub,
            os.path.join(a_.out, "lossy_rate_distortion.png"))
    bad = d[~d["valid"]].groupby(["workload", "codec"]).size()
    if len(bad):
        print("excluded rows (failed or over the bound):")
        print(bad.to_string())
    print(f"wrote {a_.out}/lossy_overview.png and lossy_rate_distortion.png")


if __name__ == "__main__":
    main()
