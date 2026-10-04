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
the settings are judged on the same clock. Tier bandwidths are fixed
(FIXED_TIERS: DRAM 12, NVMe 1, SSD 0.512, HDD 0.25 GB/s) unless --probe
names a probe_tiers.sh CSV (median over its repetitions).

Draws one figure per tier into figures/cost-model-tiers/:
  <tier>_cost_model.png  top: compression ratio per workload; bottom: cost in
                         ms per GiB of input, with "no compression" for scale.

  cost_model_tiers.py [--probe CSV] [--sweep DIR] [--out DIR]
  cost_model_tiers.py [--probe CSV] --float32 DIR [--out DIR]

With --float32, the candidates are every (codec, setting, shuffle) row of a
run_float32_sweep.sh results dir instead of one default setting per codec,
and only workloads whose sweep is complete are drawn. Bars are labeled by
codec family.
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
WORKLOADS = {"lammps": "LAMMPS", "nyx": "Nyx", "vpic": "VPIC",
             "hurricane": "Hurricane", "cesm-atm": "CESM-ATM",
             "cesm-atm-3d": "CESM-ATM-3D", "cesm": "CESM-ATM",
             "scale-letkf": "SCALE-LETKF", "scale": "SCALE-LETKF",
             "qmcpack": "QMCPACK", "exaalt": "EXAALT", "hacc": "HACC",
             "miranda": "Miranda", "sdr-nyx": "Nyx 512³ (SDR)", "s3d": "S3D"}
# the sweep's legacy default: one setting per codec, every workload
LEGACY_WORKLOADS = ["nyx", "vpic", "hurricane", "cesm-atm", "cesm-atm-3d",
                    "scale-letkf", "qmcpack"]
# codec -> label; the order is the candidate order that breaks exact ties
CODECS = {"nvcomp-lz4": "LZ4", "nvcomp-snappy": "Snappy", "nvcomp-zstd": "Zstd",
          "nvcomp-gdeflate": "GDeflate", "nvcomp-deflate": "Deflate",
          "nvcomp-ans": "ANS", "nvcomp-cascaded": "Cascaded",
          "nvcomp-bitcomp": "Bitcomp", "ndzip": "ndzip", "spspeed": "SPspeed",
          "spratio": "SPratio", "gpulz": "GPULZ", "store": "Uncompressed",
          # cpu_corpus_sweep (cpu_codecs.h); after the GPU codecs in tie order
          "cpu-zstd": "Zstd (CPU)", "cpu-lz4": "LZ4 (CPU)",
          "cpu-zlib": "zlib (CPU)", "cpu-bzip2": "bzip2 (CPU)",
          "cpu-xz": "xz (CPU)", "cpu-brotli": "Brotli (CPU)",
          "cpu-snappy": "Snappy (CPU)", "cpu-lzo": "LZO (CPU)",
          "cpu-store": "Uncompressed (CPU)",
          "cpu-blosc2": "Blosc2 (CPU)", "cpu-fpzip": "fpzip (CPU)",
          "cpu-zfp": "ZFP (CPU)", "cpu-ndzip": "ndzip (CPU)"}
# tier -> (title, probe method that sets its bandwidth, how it was measured)
_DURABLE = "durable 4 GiB write, dd bs=4M conv=fdatasync"
TIERS = {"dram": ("DRAM (GPU to host memory)", "d2h_pinned",
                  "cudaMemcpy to pinned host memory in 4 MiB calls"),
         "nvme": ("node-local NVMe", "write_fdatasync", _DURABLE),
         "burst_buffer": ("the burst buffer (/work/nvme, flash Lustre)",
                          "write_fdatasync", _DURABLE),
         "lustre": ("the Lustre PFS (/work/hdd)", "write_fdatasync", _DURABLE)}
# tier -> (title, assumed bandwidth in GB/s): used when no probe CSV is given
FIXED_TIERS = {"dram": ("DRAM", 12.0), "nvme": ("NVMe", 1.0),
               "ssd": ("SSD", 0.512), "hdd": ("HDD", 0.25)}
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


def load_float32(results_dir, engine="gpu"):
    """Per-chunk rows of every complete workload of a float32 sweep.

    A workload is complete when its results.csv holds one row per
    (chunk, setting) of its chunks.txt x settings.txt. Each (codec, setting,
    shuffle) is one candidate; failed round trips are dropped.

    @param results_dir run_float32_sweep.sh OUT directory
    @param engine      "gpu" (corpus_sweep) or "cpu" (cpu_corpus_sweep, whose
                       codecs get a cpu- family key and config prefix)
    @return dict workload -> DataFrame: file, chunk, codec (family key of
            CODECS), config, bytes, comp_bytes, comp_ms, decomp_ms
    """
    out = {}
    for wl in WORKLOADS:
        d = os.path.join(results_dir, wl)
        csv = os.path.join(d, "results.csv")
        if not os.path.isfile(csv):
            continue
        n_chunks = sum(1 for _ in open(os.path.join(d, "chunks.txt")))
        n_set = sum(1 for _ in open(os.path.join(d, "settings.txt")))
        df = pd.read_csv(csv, keep_default_na=False)
        if len(df) != n_chunks * n_set:
            print(f"skip {wl}: {len(df)}/{n_chunks * n_set} rows (running)")
            continue
        df = df[df["ok"] == 1]
        if engine == "cpu":
            codec = "cpu-" + df["algorithm"]
            config = "cpu " + df["algorithm"] + " " + df["settings"]
        else:
            nv = ~df["algorithm"].isin(["ndzip", "gpulz", "spspeed",
                                        "spratio", "store"])
            codec = np.where(nv, "nvcomp-" + df["algorithm"], df["algorithm"])
            config = df["algorithm"] + " " + df["settings"]
        out[wl] = pd.DataFrame({
            "file": df["file"], "chunk": 0, "codec": codec, "config": config,
            "bytes": df["bytes"], "comp_bytes": df["comp_bytes"],
            "comp_ms": df["comp_ms"], "decomp_ms": df["decomp_ms"]})
    return out


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
            "top": CODECS[share.index[0]], "top_share": share.iloc[0],
            "config": (pick["config"].value_counts().index[0]
                       if "config" in pick else "")}
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


def plot_tier(tier, bw_gbs, res, out, candidates="12 lossless codecs",
              probed=True):
    """One tier's figure: ratio (top) and cost per GiB (bottom), per workload.

    @param tier       key of TIERS (probed) or FIXED_TIERS
    @param bw_gbs     bandwidth of the tier, GB/s
    @param res        workload -> output of select
    @param out        PNG path
    @param candidates what the model picks from, for the subtitle
    @param probed     True when bw_gbs came from a probe CSV
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
    ratios = [res[w][m]["ratio"] for w in wls for m in MODELS]
    if max(ratios) / min(ratios) > 20:
        top.set_yscale("log")
        top.set_ylim(min(ratios) / 1.3, max(ratios) * 2.2)
        top.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        top.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        top.yaxis.set_minor_formatter(NullFormatter())
        top.set_ylabel("Compression ratio (log)", color=INK2)
    else:
        top.set_ylim(0, max(ratios) * 1.22)
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
    if probed:
        title = (f"Cost model on {TIERS[tier][0]}: {bw_gbs:.2f} GB/s measured "
                 f"({TIERS[tier][2]}, median)")
    else:
        title = f"Cost model on {FIXED_TIERS[tier][0]}: {bw_gbs:g} GB/s"
    fig.suptitle(title, x=0.01, ha="left", fontsize=12.5, color=INK, y=0.985)
    fig.text(0.01, 0.935, "cost = w_ct·compress + w_dt·decompress + "
             "w_io·compressed bytes / bandwidth, picked per 4 MiB chunk from measured "
             f"A100 times and ratios ({candidates}; no time floor, no ratio "
             "cap).", fontsize=9, color=INK2)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    """Load the sweep and the tier bandwidths, print the table, draw figures."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--probe", help="tier_bw.csv from probe_tiers.sh; "
                    "default: the fixed bandwidths of FIXED_TIERS")
    ap.add_argument("--sweep", default=DEFAULT_SWEEP)
    ap.add_argument("--float32", help="run_float32_sweep.sh results dir")
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()
    if a.probe:
        bw = tier_bandwidths(a.probe)
    else:
        bw = {t: gbs for t, (_, gbs) in FIXED_TIERS.items()}
    if a.float32:
        data = load_float32(a.float32)
        if not data:
            sys.exit(f"no complete workload under {a.float32}")
        n = max(d["config"].nunique() for d in data.values())
        candidates = f"up to {n} lossless codec settings incl. shuffle"
    else:
        data = {w: load_workload(a.sweep, w) for w in LEGACY_WORKLOADS}
        candidates = "12 lossless codecs"
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for tier, gbs in bw.items():
        res = {w: select(d, gbs) for w, d in data.items()}
        plot_tier(tier, gbs, res, os.path.join(a.out, f"{tier}_cost_model.png"),
                  candidates, probed=bool(a.probe))
        for w, r in res.items():
            for m in MODELS:
                rows.append({"tier": tier, "GBps": gbs, "workload": w,
                             "model": m, "pick": r[m]["top"],
                             "share": r[m]["top_share"],
                             "top_config": r[m]["config"],
                             "ratio": r[m]["ratio"],
                             "ms_per_GiB": r[m]["cost"],
                             "raw_ms_per_GiB": r["raw_cost"]})
    with pd.option_context("display.width", 200, "display.precision", 3,
                           "display.max_rows", None):
        print(pd.DataFrame(rows).to_string(index=False))
    print(f"wrote {a.out}/<tier>_cost_model.png for {', '.join(bw)}")


if __name__ == "__main__":
    main()
