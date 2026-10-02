#!/usr/bin/env python3
"""Is one lossless codec always best, or does the choice depend on the data?

Reads codec_sweep run sets (one per workload) and answers three questions:

  1. Per workload, which codec wins each single metric (ratio, compression
     GB/s, decompression GB/s)?
  2. Per chunk, how often is the workload's best-ratio codec also the chunk's
     best-ratio codec, and what would a per-chunk oracle gain in ratio?
  3. Under a write-cost model -- per chunk, compression time plus the time to
     write the compressed bytes at bandwidth B, with "raw" (no compression) as
     a choice -- which FIXED choice is cheapest per workload at each B, and how
     much cheaper is the per-chunk oracle than that best fixed choice?

Times are the CUDA-event compress times, averaged over the repetitions of a
run set; compressed sizes are identical across repetitions (checked). The
cost is serial (compress, then write), not pipelined.

Figures: metric_lines_by_workload.png (each metric per codec across the
workloads), regret_by_workload.png (how much worse every fixed choice is than
the best one on each workload), codec_rank_by_workload.png (every codec ranked
per workload and metric) and choice_vs_bandwidth.png (question 3).

  selection_headroom.py nyx=DIR vpic=DIR ... [--out DIR]
DIR holds rep*/<wl>_chunk4m.csv.
"""
import argparse
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullLocator  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "..", "figures", "codec-sweep")
BW_GBS = np.logspace(-1, 2, 61)  # 0.1 .. 100 GB/s

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
# Categorical slots in their validated order. A codec keeps one colour: slots
# go to codecs in this fixed priority, never by order of appearance. "raw" is
# not a codec, so it takes the neutral.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
         "#4a3aa7", "#e34948"]
SLOT_ORDER = ["spratio", "spspeed", "nvcomp-bitcomp", "ndzip", "nvcomp-ans",
              "nvcomp-zstd", "nvcomp-cascaded", "nvcomp-deflate"]
# Second channel for the line figure, so identity never rests on hue alone.
CODEC_MARKER = dict(zip(SLOT_ORDER, ["o", "s", "^", "D", "v", "P", "X", "h"]))
CODEC_COLOR = dict(zip(SLOT_ORDER, SLOTS))
RAW_COLOR = "#c3c2b7"
# Workload lines are told apart by dash and a direct label, not by hue, so
# they never reuse a codec's colour in the same figure.
DASHES = ["-", (0, (6, 2)), (0, (2, 2)), (0, (6, 2, 2, 2)), (0, (1, 1))]
# Ordinal blue ramp, darkest = rank 1; the light end stops at step 250 so the
# last rank still clears 2:1 on the surface.
BLUE_RAMP = ["#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf",
             "#2a78d6", "#3987e5", "#5598e7", "#6da7ec", "#86b6ef"]

LABEL = {"nvcomp-lz4": "LZ4", "nvcomp-snappy": "Snappy", "nvcomp-zstd": "Zstd",
         "nvcomp-gdeflate": "GDeflate", "nvcomp-deflate": "Deflate",
         "nvcomp-ans": "ANS", "nvcomp-cascaded": "Cascaded",
         "nvcomp-bitcomp": "Bitcomp", "ndzip": "ndzip", "gpulz": "GPULZ",
         "spspeed": "SPspeed", "spratio": "SPratio", "raw": "raw"}


def load(run_dir, wl):
    """Per-chunk table for one workload, times averaged over repetitions.

    @param run_dir run set directory holding rep*/
    @param wl      workload name (CSV prefix)
    @return (DataFrame with file, chunk, bytes, codec, comp_bytes, comp_ms,
             decomp_ms; number of repetitions)
    """
    frames = []
    for csv in sorted(glob.glob(os.path.join(run_dir, "rep*", f"{wl}_chunk4m.csv"))):
        summary = os.path.join(os.path.dirname(csv), "summary.txt")
        if not (os.path.exists(summary) and
                any(l.startswith("codec ") for l in open(summary))):
            print(f"skipping unfinished {csv}", file=sys.stderr)
            continue
        df = pd.read_csv(csv)
        if (df["ok"] != 1).any():
            sys.exit(f"{csv}: failed round trips")
        frames.append(df)
    if not frames:
        sys.exit(f"no finished runs under {run_dir}")
    key = ["file", "chunk", "bytes", "codec"]
    allr = pd.concat(frames)
    if allr.groupby(key)["comp_bytes"].nunique().max() != 1:
        sys.exit(f"{wl}: compressed sizes differ between repetitions")
    t = allr.groupby(key, as_index=False).agg(
        comp_bytes=("comp_bytes", "first"), comp_ms=("comp_ms", "mean"),
        decomp_ms=("decomp_ms", "mean"))
    return t, len(frames)


def metric_table(t):
    """Aggregate ratio and throughputs of every codec.

    @return DataFrame indexed by codec: ratio, comp_GBs, decomp_GBs
    """
    g = t.groupby("codec").agg(b=("bytes", "sum"), c=("comp_bytes", "sum"),
                               ct=("comp_ms", "sum"), dt=("decomp_ms", "sum"))
    return pd.DataFrame({"ratio": g["b"] / g["c"],
                         "comp_GBs": g["b"] / g["ct"] / 1e6,
                         "decomp_GBs": g["b"] / g["dt"] / 1e6})


def plot_ranks(tables, out):
    """Rank of every codec per workload, one panel per metric.

    @param tables dict workload -> metric_table() frame
    @param out    PNG path
    """
    metrics = [("ratio", "Compression ratio"),
               ("comp_GBs", "Compression throughput"),
               ("decomp_GBs", "Decompression throughput")]
    wls = list(tables)
    ranks = {m: pd.DataFrame({wl: tables[wl][m].rank(ascending=False)
                              for wl in wls}) for m, _ in metrics}
    order = (sum(r for r in ranks.values()).mean(axis=1)
             .sort_values().index.tolist())
    n = len(order)
    cmap = ListedColormap([BLUE_RAMP[min(i * len(BLUE_RAMP) // n,
                                         len(BLUE_RAMP) - 1)]
                           for i in range(n)])
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.6), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.80, bottom=0.12,
                        wspace=0.06)
    for ax, (m, title) in zip(axes, metrics):
        r = ranks[m].loc[order, wls].to_numpy()
        ax.imshow(r, cmap=cmap, vmin=0.5, vmax=n + 0.5, aspect="auto")
        for i in range(n):
            for j in range(len(wls)):
                k = int(r[i, j])
                ax.text(j, i, str(k), ha="center", va="center", fontsize=8.5,
                        fontweight="bold" if k == 1 else "normal",
                        color="#ffffff" if k <= n * 0.6 else INK)
        ax.set_xticks(range(len(wls)))
        ax.set_xticklabels(wls, rotation=30, ha="right")
        ax.set_yticks(range(n))
        ax.set_yticklabels([LABEL[c] for c in order])
        ax.set_xticks(np.arange(-0.5, len(wls)), minor=True)
        ax.set_yticks(np.arange(-0.5, n), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2)
        ax.tick_params(which="both", length=0, labelsize=9, labelcolor=INK2)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    fig.suptitle("Rank of each codec per workload (1 = best)", x=0.01,
                 ha="left", fontsize=12.5, color=INK, y=0.975)
    fig.text(0.01, 0.895, "Aggregate over the workload, 4 MiB chunks, no "
             "preprocessing, A100. Rows ordered by mean rank over all three "
             "metrics.", fontsize=9, color=INK2)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


TOP_N = 5  # codecs in the line figure


def label_ends(ax, ends, min_gap_px=11):
    """Label line ends at the right edge, pushed apart vertically so no two
    labels overlap; a short connector joins a moved label to its line.

    @param ax         axes holding the lines
    @param ends       list of (x, y, text, colour) in data coordinates
    @param min_gap_px smallest vertical distance between two labels
    """
    fig = ax.figure
    fig.canvas.draw()
    pts = sorted(((ax.transData.transform((x, y)), t, c)
                  for x, y, t, c in ends), key=lambda e: e[0][1])
    ys = [p[1] for p, _, _ in pts]
    for i in range(1, len(ys)):  # push up
        ys[i] = max(ys[i], ys[i - 1] + min_gap_px)
    top = ax.get_window_extent().y1
    if ys and ys[-1] > top:  # then shift the stack back down if it overflows
        shift = ys[-1] - top
        ys = [y - shift for y in ys]
        for i in range(len(ys) - 2, -1, -1):
            ys[i] = min(ys[i], ys[i + 1] - min_gap_px)
    inv = ax.transData.inverted()
    for (p, text, c), y in zip(pts, ys):
        lx = p[0] + 14
        _, ly = inv.transform((lx, y))
        x0, y0 = inv.transform(p)
        x1, _ = inv.transform((lx - 3, y))
        ax.plot([x0, x1], [y0, ly], color=c, linewidth=0.6,
                clip_on=False, zorder=1)
        ax.text(inv.transform((lx, y))[0], ly, text, va="center",
                fontsize=8.5, color=INK if c != RAW_COLOR else INK2,
                clip_on=False)


def plot_metric_lines(tables, out):
    """One line chart per metric for the TOP_N codecs, across the workloads.

    The codecs are the TOP_N by mean rank over all metrics and workloads, the
    same set in every panel; colour and marker come from the fixed codec map
    and are named once, in a shared legend.

    @param tables dict workload -> metric_table() frame
    @param out    PNG path
    """
    wls = list(tables)
    metrics = [("ratio", "Compression ratio", False),
               ("comp_GBs", "Compression throughput (GB/s, log scale)", True),
               ("decomp_GBs", "Decompression throughput (GB/s, log scale)",
                True)]
    mean_rank = sum(tables[wl][m].rank(ascending=False)
                    for wl in wls for m, _, _ in metrics)
    top = [c for c in SLOT_ORDER if c in
           set(mean_rank.sort_values().index[:TOP_N])]
    fig, axes = plt.subplots(3, 1, figsize=(8, 10.5), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.865, bottom=0.05,
                        hspace=0.25)
    x = np.arange(len(wls))
    for ax, (m, title, logy) in zip(axes, metrics):
        style_axes(ax)
        ax.grid(True, axis="y", color=GRID, linewidth=0.6)
        for c in top:
            ax.plot(x, [tables[wl].loc[c, m] for wl in wls],
                    color=CODEC_COLOR[c], linewidth=2.2,
                    marker=CODEC_MARKER[c], ms=7.5, mec=SURFACE, mew=1.2,
                    label=LABEL[c], zorder=3)
        if logy:
            ax.set_yscale("log")
            # plain numbers, not 6x10^1: pick round ticks inside the range
            lo, hi = ax.get_ylim()
            ax.set_yticks([v for v in (10, 15, 20, 30, 40, 60, 80, 100, 150,
                                       200) if lo <= v <= hi])
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_locator(NullLocator())
        ax.set_xlim(-0.3, len(wls) - 0.7)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(wls, fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", ncol=len(top),
               frameon=False, fontsize=10, labelcolor=INK,
               bbox_to_anchor=(0.085, 0.925), handlelength=2.6,
               columnspacing=1.6)
    fig.suptitle(f"Top {len(top)} lossless GPU codecs, per metric and "
                 "workload", x=0.01, ha="left", fontsize=12.5, color=INK,
                 y=0.985)
    fig.text(0.01, 0.945, f"Higher is better. Top {len(top)} by mean rank "
             f"over all three metrics and {len(wls)} workloads. 4 MiB chunks, A100.",
             fontsize=9, color=INK2)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


# Regret bins (percent worse than the workload's best choice) and their fills:
# 0 is the winner; then one blue ramp, light to dark, as the loss grows.
REGRET_BINS = [0.05, 5, 20, 50, 100]
REGRET_FILL = ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
REGRET_TEXT = [INK, INK, INK, "#ffffff", "#ffffff", "#ffffff"]
REGRET_KEY = ["best", "< 5%", "5-20%", "20-50%", "50-100%", "> 100%"]


def regret_tables(data, bw_gbs):
    """Percent by which every fixed choice is worse than each workload's best.

    @param data   dict workload -> per-chunk table
    @param bw_gbs bandwidth for the write-cost panel
    @return (storage regret, write-time regret): DataFrames indexed by choice
            (12 codecs and raw), one column per workload plus "worst"
    """
    store, write = {}, {}
    for wl, t in data.items():
        ratio = metric_table(t)["ratio"]
        ratio["raw"] = 1.0
        store[wl] = 100 * (ratio.max() / ratio - 1)
        cost = write_cost(t, bw_gbs).sum()
        write[wl] = 100 * (cost / cost.min() - 1)
    out = []
    for d in (store, write):
        df = pd.DataFrame(d)
        df["worst"] = df.max(axis=1)
        out.append(df)
    return out


def plot_regret(tables, bw_gbs, out):
    """Regret matrix: one panel per objective, rows = fixed choices.

    @param tables (storage, write) from regret_tables
    @param bw_gbs bandwidth the write panel assumes
    @param out    PNG path
    """
    store, write = tables
    order = (store["worst"] + write["worst"]).sort_values().index.tolist()
    cols = list(store.columns)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.4), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.80, bottom=0.12,
                        wspace=0.05)
    titles = ["Storage: extra bytes vs the best codec",
              f"Write time at {bw_gbs:g} GB/s: compress + write, vs the best"]
    for ax, df, title in zip(axes, (store, write), titles):
        ax.set_facecolor(SURFACE)
        for i, row in enumerate(order):
            for j, col in enumerate(cols):
                v = df.loc[row, col]
                b = int(np.searchsorted(REGRET_BINS, v, side="right"))
                x = j + (0.25 if col == "worst" else 0)  # gap before "worst"
                ax.add_patch(plt.Rectangle((x - 0.5, i - 0.5), 1, 1,
                                           facecolor=REGRET_FILL[b],
                                           edgecolor=SURFACE, linewidth=2))
                if b == 0:
                    ax.add_patch(plt.Rectangle((x - 0.42, i - 0.42), 0.84,
                                               0.84, fill=False,
                                               edgecolor=INK, linewidth=1.2))
                # a non-winner that rounds to 0 must not read as the winner
                txt = ("0" if b == 0 else "<1" if v < 1 else
                       f"{v:.0f}" if v < 1000 else ">999")
                ax.text(x, i, txt, ha="center", va="center", fontsize=8.5,
                        color=REGRET_TEXT[b],
                        fontweight="bold" if b == 0 or col == "worst"
                        else "normal")
        ax.set_xlim(-0.5, len(cols) - 0.25)
        ax.set_ylim(len(order) - 0.5, -0.5)
        ax.set_xticks([j + (0.25 if c == "worst" else 0)
                       for j, c in enumerate(cols)])
        ax.set_xticklabels(cols, rotation=30, ha="right")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([LABEL[r] for r in order])
        ax.tick_params(length=0, labelsize=9, labelcolor=INK2)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=f,
                             edgecolor=INK if k == 0 else f, linewidth=1)
               for k, f in enumerate(REGRET_FILL)]
    fig.legend(handles, REGRET_KEY, loc="upper right", ncol=6, frameon=False,
               fontsize=9, labelcolor=INK2, bbox_to_anchor=(0.99, 0.905),
               handlelength=1.4, columnspacing=1.2)
    fig.suptitle("No single codec is best on every workload", x=0.01,
                 ha="left", fontsize=12.5, color=INK, y=0.975)
    fig.text(0.01, 0.905, "% worse than the best fixed choice for that "
             "workload (0 = it is the best). raw = store uncompressed. "
             "4 MiB chunks, A100.", fontsize=9, color=INK2)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def metric_winners(t):
    """Winner of each single aggregate metric.

    @return dict metric -> (codec, value)
    """
    g = t.groupby("codec").agg(b=("bytes", "sum"), c=("comp_bytes", "sum"),
                               ct=("comp_ms", "sum"), dt=("decomp_ms", "sum"))
    ratio = g["b"] / g["c"]
    comp = g["b"] / g["ct"] / 1e6
    decomp = g["b"] / g["dt"] / 1e6
    return {"ratio": (ratio.idxmax(), ratio.max()),
            "comp_GBs": (comp.idxmax(), comp.max()),
            "decomp_GBs": (decomp.idxmax(), decomp.max())}


def ratio_oracle(t):
    """Per-chunk best-ratio codec vs the workload's best fixed one.

    @return (best fixed codec, share of chunks it also wins, oracle ratio gain
             in percent, per-codec share of chunk wins)
    """
    wide = t.pivot_table(index=["file", "chunk"], columns="codec",
                         values="comp_bytes")
    nbytes = t.groupby(["file", "chunk"])["bytes"].first().reindex(wide.index)
    fixed = wide.sum().idxmin()
    win = wide.idxmin(axis=1)
    oracle = nbytes.sum() / wide.min(axis=1).sum()
    best = nbytes.sum() / wide[fixed].sum()
    return (fixed, (win == fixed).mean(), 100 * (oracle / best - 1),
            win.value_counts(normalize=True))


def write_cost(t, bw_gbs):
    """Per-chunk write cost (ms) for every codec and raw at one bandwidth.

    @param bw_gbs bandwidth in GB/s
    @return DataFrame indexed by chunk, one column per choice
    """
    bpm = bw_gbs * 1e6  # bytes per ms
    cost = t.assign(cost=t["comp_ms"] + t["comp_bytes"] / bpm).pivot_table(
        index=["file", "chunk"], columns="codec", values="cost")
    nbytes = t.groupby(["file", "chunk"])["bytes"].first().reindex(cost.index)
    cost["raw"] = nbytes / bpm
    return cost


def bandwidth_sweep(t):
    """Best fixed choice and oracle headroom across BW_GBS.

    @return DataFrame: bw, fixed, fixed_ms, oracle_ms, headroom_pct,
            oracle_choices (dict of per-chunk winner shares)
    """
    rows = []
    for bw in BW_GBS:
        c = write_cost(t, bw)
        tot = c.sum()
        fixed = tot.idxmin()
        oracle = c.min(axis=1).sum()
        rows.append({"bw": bw, "fixed": fixed, "fixed_ms": tot.min(),
                     "oracle_ms": oracle,
                     "headroom_pct": 100 * (tot.min() / oracle - 1)})
    return pd.DataFrame(rows)


def style_axes(ax):
    """Recessive grid and axes on the chart surface."""
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=9)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def plot(sweeps, out):
    """Best fixed choice per workload vs bandwidth, and the oracle headroom.

    @param sweeps dict workload -> bandwidth_sweep() frame
    @param out    PNG path
    """
    wls = list(sweeps)
    winners = {w for s in sweeps.values() for w in s["fixed"]}
    color = {c: CODEC_COLOR.get(c, MUTED) for c in winners}
    color["raw"] = RAW_COLOR

    fig, (top, bot) = plt.subplots(
        2, 1, figsize=(9.5, 7.2), sharex=True,
        gridspec_kw={"height_ratios": [len(wls) * 0.55, 3.2]})
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.12, right=0.83, top=0.86, bottom=0.09,
                        hspace=0.28)
    edges = np.sqrt(BW_GBS[:-1] * BW_GBS[1:])
    lo = np.r_[BW_GBS[0], edges]
    hi = np.r_[edges, BW_GBS[-1]]
    style_axes(top)
    for row, wl in enumerate(wls):
        s = sweeps[wl]
        run_start = 0
        for i in range(1, len(s) + 1):
            if i == len(s) or s["fixed"].iloc[i] != s["fixed"].iloc[run_start]:
                w = s["fixed"].iloc[run_start]
                x0, x1 = lo[run_start], hi[i - 1]
                top.barh(row, x1 - x0, left=x0, height=0.62,
                         color=color[w], edgecolor=SURFACE, linewidth=2)
                if np.log10(x1 / x0) > 0.28:
                    top.text(np.sqrt(x0 * x1), row, LABEL[w], ha="center",
                             va="center", fontsize=8.5,
                             color="#ffffff" if w != "raw" else INK)
                run_start = i
    top.set_yticks(range(len(wls)))
    top.set_yticklabels(wls)
    top.invert_yaxis()
    top.set_xscale("log")
    top.set_xlim(BW_GBS[0], BW_GBS[-1])
    top.tick_params(axis="y", length=0)
    top.spines["left"].set_visible(False)
    top.set_title("Cheapest FIXED choice for the whole workload",
                  loc="left", fontsize=10.5, color=INK)
    keys = [c for c in SLOT_ORDER if c in winners] + ["raw"]
    top.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=color[c],
                                      label=LABEL[c]) for c in keys],
               loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False,
               fontsize=9, labelcolor=INK2)

    style_axes(bot)
    bot.grid(True, color=GRID, linewidth=0.6)
    for i, wl in enumerate(wls):
        s = sweeps[wl]
        bot.plot(s["bw"], s["headroom_pct"], color=INK2, linewidth=1.8,
                 linestyle=DASHES[i % len(DASHES)], label=wl)
        k = s["headroom_pct"].idxmax()
        if s["headroom_pct"][k] >= 1:  # label the peak; the legend has the rest
            bot.annotate(f"{wl} {s['headroom_pct'][k]:.1f}%",
                         (s["bw"][k], s["headroom_pct"][k]), xytext=(0, 5),
                         textcoords="offset points", fontsize=9, color=INK,
                         ha="center", va="bottom", path_effects=[
                             patheffects.withStroke(linewidth=3,
                                                    foreground=SURFACE)])
    bot.set_xscale("log")
    bot.set_ylim(0, max(s["headroom_pct"].max() for s in sweeps.values()) * 1.2)
    bot.set_xlabel("Write bandwidth after compression (GB/s, log scale)")
    bot.set_ylabel("Oracle gain over best fixed (%)")
    bot.set_title("What a perfect per-chunk choice saves over that fixed "
                  "choice", loc="left", fontsize=10.5, color=INK)
    bot.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False,
               fontsize=9, labelcolor=INK2, handlelength=3)

    fig.suptitle("Does one lossless codec win everywhere?", x=0.01, ha="left",
                 fontsize=12.5, color=INK, y=0.975)
    fig.text(0.01, 0.915, "Write cost per chunk = GPU compression time + "
             "compressed bytes / bandwidth; raw = no compression. "
             "4 MiB chunks, 12 codecs, A100.", fontsize=9, color=INK2)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    """Load each workload, print the three answers, draw the figure."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", help="WL=DIR per workload")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--regret-bw", type=float, default=1.0,
                    help="GB/s for the regret figure's write-time panel")
    a = ap.parse_args()
    data = {}
    for spec in a.runs:
        wl, d = spec.split("=", 1)
        data[wl] = load(d, wl)

    print("1) single-metric winners (aggregate over the workload)")
    rows = []
    for wl, (t, nrep) in data.items():
        w = metric_winners(t)
        rows.append({"workload": wl, "reps": nrep, "chunks": t[["file", "chunk"]]
                     .drop_duplicates().shape[0],
                     **{k: f"{LABEL[c]} ({v:.3g})" for k, (c, v) in w.items()}})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n2) per-chunk ratio: best fixed codec vs per-chunk oracle")
    for wl, (t, _) in data.items():
        fixed, share, gain, wins = ratio_oracle(t)
        top = ", ".join(f"{LABEL[c]} {100 * s:.1f}%" for c, s in wins.head(4).items())
        print(f"  {wl:10s} fixed={LABEL[fixed]:8s} wins {100 * share:5.1f}% of "
              f"chunks; oracle +{gain:.2f}% ratio; chunk winners: {top}")

    print("\n3) write cost: best fixed choice and oracle headroom by bandwidth")
    sweeps = {wl: bandwidth_sweep(t) for wl, (t, _) in data.items()}
    for wl, s in sweeps.items():
        print(f"  {wl}")
        for bw in (0.5, 1, 2, 5, 10, 25, 50, 100):
            r = s.iloc[(s["bw"] - bw).abs().argmin()]
            print(f"    {r['bw']:7.2f} GB/s  fixed={LABEL[r['fixed']]:8s} "
                  f"oracle saves {r['headroom_pct']:5.2f}%")
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, "choice_vs_bandwidth.png")
    plot(sweeps, out)
    lines_out = os.path.join(a.out, "metric_lines_by_workload.png")
    plot_metric_lines({wl: metric_table(t) for wl, (t, _) in data.items()},
                      lines_out)
    print(f"wrote {lines_out}")
    regret_out = os.path.join(a.out, "regret_by_workload.png")
    plot_regret(regret_tables({wl: t for wl, (t, _) in data.items()},
                              a.regret_bw), a.regret_bw, regret_out)
    print(f"wrote {regret_out}")
    ranks_out = os.path.join(a.out, "codec_rank_by_workload.png")
    plot_ranks({wl: metric_table(t) for wl, (t, _) in data.items()}, ranks_out)
    print(f"\nwrote {out}\nwrote {ranks_out}")


if __name__ == "__main__":
    main()
