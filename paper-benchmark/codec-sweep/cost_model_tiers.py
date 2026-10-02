#!/usr/bin/env python3
"""NeuroPress's cost model on four storage tiers: balanced, compression, speed.

NeuroPress ranks codecs by

    cost = w_ct * compress_ms + w_dt * decompress_ms
           + w_io * compressed_bytes / bandwidth

and picks the cheapest. Three settings are compared:
  balanced              w_ct = w_dt = w_io = 1 (the shipped default)
  compression-focused   w_ct = w_dt = 0 (NeuroPress's "best" mode): pure ratio
  speed-focused         w_io = 0: the fastest codec, whatever its ratio

Each is applied per 4 MiB chunk to the MEASURED ratio and CUDA-event times of
the lossless codec sweep (perfect prediction), with no time floor and no ratio
cap. Each pick is then charged its full cost, compress + decompress + I/O, so
the settings are judged on the same clock. The bandwidth of each tier comes
from probe_tiers.sh (median over its repetitions).

Draws one figure per tier into figures/cost-model-tiers/:
  <tier>_cost_model.png  top: compression ratio per workload; bottom: cost in
                         ms per GiB of input, with "no compression" for scale.

  cost_model_tiers.py PROBE_CSV [--sweep DIR] [--out DIR]
"""
import argparse
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "..", "figures", "cost-model-tiers")
DEFAULT_SWEEP = "/projects/bekn/imuradli/np-codec-sweep/full"

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"

# workload -> display name, in plot order
WORKLOADS = {"nyx": "Nyx", "vpic": "VPIC", "hurricane": "Hurricane",
             "cesm-atm": "CESM-ATM", "cesm-atm-3d": "CESM-ATM-3D",
             "scale-letkf": "SCALE-LETKF", "qmcpack": "QMCPACK"}
# codec -> label; the order is the candidate order that breaks exact ties
CODECS = {"nvcomp-lz4": "LZ4", "nvcomp-snappy": "Snappy", "nvcomp-zstd": "Zstd",
          "nvcomp-gdeflate": "GDeflate", "nvcomp-deflate": "Deflate",
          "nvcomp-ans": "ANS", "nvcomp-cascaded": "Cascaded",
          "nvcomp-bitcomp": "Bitcomp", "ndzip": "ndzip", "spspeed": "SPspeed",
          "spratio": "SPratio", "gpulz": "GPULZ"}
# tier -> (title, probe method that sets its bandwidth, how it was measured)
_DURABLE = "durable 4 GiB write, dd bs=4M conv=fdatasync"
TIERS = {"dram": ("DRAM (GPU to host memory)", "d2h_pinned",
                  "cudaMemcpy to pinned host memory in 4 MiB calls"),
         "nvme": ("node-local NVMe", "write_fdatasync", _DURABLE),
         "burst_buffer": ("the burst buffer (/work/nvme, flash Lustre)",
                          "write_fdatasync", _DURABLE),
         "lustre": ("the Lustre PFS (/work/hdd)", "write_fdatasync", _DURABLE)}
AQUA = "#1baf7a"
MODELS = {"balanced": ("Balanced  (w_ct = w_dt = w_io = 1)", BLUE),
          "compression": ("Compression-focused  (w_ct = w_dt = 0)", ORANGE),
          "speed": ("Speed-focused  (w_io = 0)", AQUA)}


def tier_bandwidths(probe_csv, method=None):
    """Median bandwidth of each tier from probe_tiers.sh or ior_tiers.sh.

    @param probe_csv tier_bw.csv (probe_tiers.sh) or ior_bw.csv (ior_tiers.sh)
    @param method    one probe method for every tier (e.g. ior_write_n1);
                     None = each tier's default in TIERS
    @return dict tier -> GB/s
    """
    p = pd.read_csv(probe_csv)
    bw = {}
    for tier, (_, default, _) in TIERS.items():
        method_t = method or default
        s = p[(p.tier == tier) & (p.method == method_t)]["GBps"]
        if s.empty:
            sys.exit(f"{probe_csv}: no {tier}/{method_t} rows")
        bw[tier] = float(s.median())
    return bw


def load_workload(sweep, wl):
    """Per-chunk measurements of one workload, times averaged over its runs.

    @param sweep directory holding <wl>-full-*/rep*/<wl>_chunk4m.csv
    @param wl    workload name
    @return DataFrame: file, chunk, codec, bytes, comp_bytes, comp_ms, decomp_ms
    """
    dirs = sorted(glob.glob(os.path.join(sweep, f"{wl}-full-*")))
    if not dirs:
        sys.exit(f"no {wl}-full-* under {sweep}")
    reps = [pd.read_csv(f) for f in
            sorted(glob.glob(os.path.join(dirs[-1], "rep*", f"{wl}_chunk4m.csv")))]
    df = pd.concat(reps, ignore_index=True)
    if (df["ok"] != 1).any():
        sys.exit(f"{wl}: failed round trips in the sweep")
    keys = ["file", "chunk", "codec"]
    return df.groupby(keys, as_index=False).agg(
        bytes=("bytes", "first"), comp_bytes=("comp_bytes", "first"),
        comp_ms=("comp_ms", "mean"), decomp_ms=("decomp_ms", "mean"),
        n_reps=("ok", "size"))


def select(df, bw_gbs):
    """Apply both cost-model settings per chunk and total what they pick.

    @param df     output of load_workload
    @param bw_gbs tier bandwidth, GB/s
    @return dict model -> totals (ratio, compress/decompress/I/O ms per GiB,
            the most-picked codec and its share of chunks)
    """
    d = df.copy()
    d["io_ms"] = d["comp_bytes"] / (bw_gbs * 1e6)  # GB/s = 1e6 bytes/ms
    d["order"] = d["codec"].map({c: i for i, c in enumerate(CODECS)})
    score = {"balanced": d["comp_ms"] + d["decomp_ms"] + d["io_ms"],
             "compression": d["io_ms"],
             "speed": d["comp_ms"] + d["decomp_ms"]}
    gib = df.groupby(["file", "chunk"])["bytes"].first().sum() / 2**30
    out = {}
    for model, s in score.items():
        pick = (d.assign(score=s).sort_values(["file", "chunk", "score", "order"])
                .groupby(["file", "chunk"]).head(1))
        share = pick["codec"].value_counts(normalize=True)
        out[model] = {
            "ratio": pick["bytes"].sum() / pick["comp_bytes"].sum(),
            "comp": pick["comp_ms"].sum() / gib,
            "decomp": pick["decomp_ms"].sum() / gib,
            "io": pick["io_ms"].sum() / gib,
            "top": CODECS[share.index[0]], "top_share": share.iloc[0]}
        out[model]["cost"] = sum(out[model][k] for k in ("comp", "decomp", "io"))
    out["raw_cost"] = 2**30 / (bw_gbs * 1e6)  # ms per GiB, no compression
    return out


def style_axes(ax):
    """Recessive grid and spines on the chart surface."""
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def pick_label(r):
    """Codec label for a bar: the most-picked codec, with its share if mixed."""
    if r["top_share"] >= 0.9:
        return r["top"]
    return f"{r['top']}\n{100 * r['top_share']:.0f}%"


def plot_tier(tier, bw_gbs, res, out):
    """One tier's figure: ratio (top) and cost per GiB (bottom), per workload.

    @param tier   key of TIERS
    @param bw_gbs measured bandwidth of the tier, GB/s
    @param res    workload -> output of select
    @param out    PNG path
    """
    wls = [w for w in WORKLOADS if w in res]
    x = np.arange(len(wls))
    width, off = 0.27, {"balanced": -0.28, "compression": 0.0, "speed": 0.28}
    fig, (top, bot) = plt.subplots(2, 1, figsize=(14, 7.6), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1.15]})
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.07, top=0.835,
                        hspace=0.12)
    for ax in (top, bot):
        style_axes(ax)
    for model, (_, color) in MODELS.items():
        r = [res[w][model] for w in wls]
        top.bar(x + off[model], [v["ratio"] for v in r], width, color=color,
                edgecolor=SURFACE, linewidth=2)
        bot.bar(x + off[model], [v["cost"] for v in r], width, color=color,
                edgecolor=SURFACE, linewidth=2)
        for xi, v in zip(x, r):
            top.annotate(pick_label(v), (xi + off[model], v["ratio"]),
                         xytext=(0, 3), textcoords="offset points",
                         ha="center", va="bottom", fontsize=7.5, color=INK2)
    for xi, w in zip(x, wls):
        b = res[w]["balanced"]["cost"]
        for m in ("compression", "speed"):
            c = res[w][m]["cost"]
            bot.annotate(f"×{c / b:.2f}", (xi + off[m], c), xytext=(0, 3),
                         textcoords="offset points", ha="center", va="bottom",
                         fontsize=8, color=INK, zorder=5,
                         bbox=dict(boxstyle="square,pad=0.15", fc=SURFACE,
                                   ec="none"))
        bot.plot([xi - 0.44, xi + 0.44], [res[w]["raw_cost"]] * 2, color=MUTED,
                 linewidth=1.5, linestyle=(0, (4, 2)), zorder=4)
    top.set_ylim(0, max(res[w][m]["ratio"] for w in wls for m in MODELS) * 1.22)
    top.set_ylabel("Compression ratio", color=INK2)
    bot.set_yscale("log")
    lo = min(min(res[w][m]["cost"] for m in MODELS) for w in wls)
    hi = max(max(max(res[w][m]["cost"] for m in MODELS), res[w]["raw_cost"])
             for w in wls)
    bot.set_ylim(lo / 1.6, hi * 1.9)
    bot.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    bot.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    bot.yaxis.set_minor_formatter(NullFormatter())
    bot.set_ylabel("Cost per GiB of input (ms, log)", color=INK2)
    bot.set_xticks(x)
    bot.set_xticklabels([WORKLOADS[w] for w in wls], fontsize=9.5, color=INK)
    top.set_title("Codec each setting picks (share of chunks when mixed)",
                  loc="left", fontsize=10, color=INK2)
    bot.set_title("What the pick costs: compress + decompress + I/O "
                  "(× = cost relative to balanced)",
                  loc="left", fontsize=10, color=INK2)
    handles = [Patch(color=c, label=l) for l, c in MODELS.values()]
    handles.append(Line2D([], [], color=MUTED, linewidth=1.5,
                          linestyle=(0, (4, 2)), label="No compression"))
    fig.legend(handles=handles, loc="upper left", ncol=4, frameon=False,
               fontsize=9, labelcolor=INK2, bbox_to_anchor=(0.065, 0.915))
    fig.suptitle(f"Cost model on {TIERS[tier][0]}: {bw_gbs:.2f} GB/s measured "
                 f"({TIERS[tier][2]}, median)",
                 x=0.01, ha="left", fontsize=12.5, color=INK, y=0.985)
    fig.text(0.01, 0.935, "cost = w_ct·compress + w_dt·decompress + "
             "w_io·compressed bytes / bandwidth, picked per 4 MiB chunk from measured "
             "A100 times and ratios (12 lossless codecs; no time floor, no ratio "
             "cap).", fontsize=9, color=INK2)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    """Load the sweep and the tier probe, print the table, draw four figures."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("probe_csv", help="tier_bw.csv from probe_tiers.sh")
    ap.add_argument("--sweep", default=DEFAULT_SWEEP)
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()
    bw = tier_bandwidths(a.probe_csv)
    data = {w: load_workload(a.sweep, w) for w in WORKLOADS}
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for tier, gbs in bw.items():
        res = {w: select(d, gbs) for w, d in data.items()}
        plot_tier(tier, gbs, res, os.path.join(a.out, f"{tier}_cost_model.png"))
        for w, r in res.items():
            for m in MODELS:
                rows.append({"tier": tier, "GBps": gbs, "workload": w,
                             "model": m, "pick": r[m]["top"],
                             "share": r[m]["top_share"], "ratio": r[m]["ratio"],
                             "ms_per_GiB": r[m]["cost"],
                             "raw_ms_per_GiB": r["raw_cost"]})
    with pd.option_context("display.width", 200, "display.precision", 3,
                           "display.max_rows", None):
        print(pd.DataFrame(rows).to_string(index=False))
    print(f"wrote {a.out}/<tier>_cost_model.png for {', '.join(bw)}")


if __name__ == "__main__":
    main()
