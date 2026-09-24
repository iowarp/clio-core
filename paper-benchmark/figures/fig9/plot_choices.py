#!/usr/bin/env python3
"""What NeuroPress actually chose, per workload -- the motivation histogram.

One 100%-stacked bar per workload, segmented by the codec NeuroPress selected
for each chunk, most-chosen first. The claim the figure carries is the first
segment of each bar: the winning codec is DIFFERENT on four of the five
workloads, so a system that commits to one fixed codec is wrong on the others.
That is the same claim figure 9 panel (b) measures as the Best/Worst fixed
nvCOMP spread; this draws the reason for it.

THE DATA IS blobs.csv, which every figure-9 arm writes at no cost -- no rerun
and no extra instrumentation. Each row is one chunk: the codec that ran, its
ratio, and the CUDA-event bracket around the launch.

TWO THINGS THIS FIGURE CANNOT SHOW, both worth knowing before citing it:

  PREPROCESSING IS NOT IN blobs.csv. The task context the replay driver reads
  (core_tasks.h) carries compress_lib_ and a SUMMED actual_preproc_time_ms_,
  never the chosen action's quantize/byte_shuffle booleans. On a LOSSLESS arm
  the share is recoverable exactly -- quantize is masked to -INFINITY without a
  positive bound, so preproc_ms > 0 means byte-shuffle -- and --shuffle-from
  prints it under each bar. On a lossy arm it is ambiguous and is left off.
  It is not a small effect: on Nyx, `ans` goes from ratio 1.09 unshuffled to
  3.43 shuffled.

  THE SHARES DRIFT RUN TO RUN. The chooser learns online. Two runs of the same
  Nyx arm gave 44.1% and 75.6% for the same codec. Ratios reproduce to 1-3%;
  choice shares do not. Cite the SPREAD (how many codecs cover the bar), not a
  single codec's percentage, unless the weights were frozen.

Usage:
  ./plot_choices.py --camp Nyx=/work/hdd/.../fig9-full-nyx-.../nyx \
                    --camp VPIC=... --out live
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# STYLE -- matches plot_fig9.py beside this file
# ---------------------------------------------------------------------------
FIG_W = 7.16                       # IEEE two-column figure* width, inches
PLOT_H = 2.05                      # inches of plot area, axis to axis
FS_TITLE, FS_TICK, FS_LEG, FS_VAL = 8.5, 7, 7, 6.5   # nothing below 6.5 pt
INK, INK_MUTED, GRID = "#2b2b2b", "#6b6a66", "#e7e6e2"
BAR_W = 0.62
LABEL_FLOOR = 4.0                  # percent: below this a segment goes unlabelled
NAME_FLOOR = 11.0                  # percent: above this the segment carries the
                                   # codec NAME too, not just its share. Direct
                                   # labelling is what lets a reader take the
                                   # figure in without decoding the legend.
WORKLOAD_ORDER = ["VPIC", "Nyx", "LAMMPS", "WarpX", "AI"]

# CODEC COLOURS. Hue = codec, held fixed across every workload so a reader
# compares bars by colour rather than by reading five legends. "raw" is the
# neutral grey that Baseline carries in figure 9, because it is the same
# outcome: bytes stored untouched.
CODEC_COLORS = {
    "nvcomp-bitcomp":     "#1c5cab",
    "nvcomp-ans":         "#008856",
    "nvcomp-zstd":        "#c98500",
    "nvcomp-lz4":         "#872985",
    "nvcomp-snappy":      "#d9403f",
    "nvcomp-cascaded":    "#5089cc",
    "nvcomp-gdeflate":    "#e87ba4",
    "nvcomp-deflate":     "#eda100",
    "raw(not-beneficial)": "#8c8b87",
}
FALLBACK_COLORS = ["#e87ba4", "#006435", "#4b4a47"]
OTHER_COLOR = "#d6d5d1"

#: Axis title when the caller does not override it.
DEFAULT_TITLE = "Codec chosen per chunk by NeuroPress, by workload"

#: Printed instead of the wire name; "raw" is an outcome, not a library.
PRETTY = {"raw(not-beneficial)": "stored raw"}


def pretty(codec: str) -> str:
    """Legend text for a codec, with the nvcomp- prefix dropped."""
    if codec in PRETTY:
        return PRETTY[codec]
    return codec[len("nvcomp-"):] if codec.startswith("nvcomp-") else codec


#: How each workload's blob names carry a timestep. Tried in order; the first
#: that matches wins. Nyx/VPIC name AMReX plotfiles (plt00000/...), WarpX openPMD
#: steps (step00000/...), LAMMPS embeds the MD step in the chunk name
#: (force_step_21800_chunk_3), AI names the training epoch (epoch01/...).
STEP_PATTERNS = [r"plt(\d+)", r"step_(\d+)", r"step(\d+)", r"epoch(\d+)"]


def blob_step(name: str):
    """Pull the simulation timestep out of a blob name.

    :param name: the blobs.csv `blob` column, e.g. "plt00042/fab0000_comp00_density/chunk_0"
    :return: the step as an int, or None when no pattern matches
    """
    for pat in STEP_PATTERNS:
        m = re.search(pat, name)
        if m:
            return int(m.group(1))
    return None


def bin_steps(rows: list, nbins: int):
    """Group chunks into equal-width timestep bins.

    Workloads run different numbers of steps (AI 11, Nyx 559), so a shared bin
    count is what lets their columns line up in one chart. Bins are over the
    step RANGE, not over chunk order, so an uneven dump does not distort time.

    :param rows: blobs.csv rows
    :param nbins: how many bins to spread the run across
    :return: (list of Counter per bin, lo_step, hi_step); empty bins are empty
    """
    steps = [(blob_step(r["blob"]), r) for r in rows]
    steps = [(s, r) for s, r in steps if s is not None]
    if not steps:
        return [], None, None
    lo = min(s for s, _ in steps)
    hi = max(s for s, _ in steps)
    span = max(1, hi - lo)
    # BINNED BY RANK AMONG THE DUMPS, not by step value. Value-binning leaves a
    # blank row wherever the dumps are unevenly spaced -- AI writes 11 epochs
    # between 1 and 50 -- and a blank row reads as "nothing was written", which
    # is wrong. Each row is therefore "the Nth dump of the run", and the axis
    # ends carry the real step numbers. Also caps at the dump count, so a run
    # with fewer dumps than bins does not get empty rows either.
    order = sorted(set(st for st, _ in steps))
    rank = {st: i for i, st in enumerate(order)}
    k = min(nbins, len(order))
    bins = [collections.Counter() for _ in range(k)]
    for st, r in steps:
        idx = min(k - 1, rank[st] * k // len(order))
        bins[idx][r["codec"]] += 1
    return bins, lo, hi


def find_blobs(root: str, arm: str) -> str | None:
    """Locate an arm's per-chunk log under a campaign's workload directory."""
    for dirpath, _, names in os.walk(os.path.join(root, arm)):
        if "blobs.csv" in names:
            return os.path.join(dirpath, "blobs.csv")
    return None


def read_choices(path: str, top: int):
    """Codec shares for one arm.

    :param path: a blobs.csv
    :param top: how many codecs get their own segment; the rest merge into one
    :return: (rows_total, [(codec, count, pct)] with an "(other)" tail)
    """
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    cnt = collections.Counter(r["codec"] for r in rows)
    n = max(1, len(rows))
    out = [(c, v, 100.0 * v / n) for c, v in cnt.most_common(top)]
    rest = n - sum(v for _, v, _ in out)
    if rest:
        out.append(("(other)", rest, 100.0 * rest / n))
    return n, out


def shuffle_pct(path: str) -> float:
    """Percent of chunks that ran a preprocessing kernel.

    EXACT byte-shuffle only on a lossless arm, where quantize cannot run. The
    caller is responsible for pointing this at one.
    """
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    n = max(1, len(rows))
    return 100.0 * sum(1 for r in rows if float(r["preproc_ms"]) > 0) / n


def colour_for(codec: str, taken: dict) -> str:
    """Stable colour per codec, with a fallback for one the palette lacks."""
    if codec == "(other)":
        return OTHER_COLOR
    if codec in CODEC_COLORS:
        return CODEC_COLORS[codec]
    if codec not in taken:
        taken[codec] = FALLBACK_COLORS[len(taken) % len(FALLBACK_COLORS)]
    return taken[codec]


def draw(data, shuffles, out_path: str, title: str) -> None:
    """Draw the stacked chart.

    :param data: [(workload, chunks, [(codec, count, pct)])] in display order
    :param shuffles: {workload: pct} or {} to omit the shuffle line
    :param out_path: PNG to write
    :param title: axis title
    """
    key_h = 0.62 if shuffles else 0.40
    legend_h = 0.50
    fig_h = PLOT_H + key_h + legend_h + 0.22
    fig = plt.figure(figsize=(FIG_W, fig_h))
    ax = fig.add_axes([0.055, (key_h + 0.10) / fig_h, 0.925, PLOT_H / fig_h])

    taken: dict = {}
    # LEGEND BY TOTAL SHARE, not by first appearance: appearance order put the
    # pale "(other)" tail between two real codecs and scattered the rare ones.
    weight: dict = collections.defaultdict(float)
    for _wl, _n, segs in data:
        for codec, _c, pct in segs:
            weight[codec] += pct
    for i, (wl, _n, segs) in enumerate(data):
        bottom = 0.0
        for codec, _c, pct in segs:
            col = colour_for(codec, taken)
            ax.bar(i, pct, BAR_W, bottom=bottom, color=col,
                   edgecolor="white", linewidth=0.6, zorder=3)
            if pct >= LABEL_FLOOR:
                # Dark fills take white text; the pale "(other)" tail takes ink.
                fg = INK if codec == "(other)" else "white"
                txt = (f"{pretty(codec)}\n{pct:.0f}%" if pct >= NAME_FLOOR
                       else f"{pct:.0f}%")
                ax.text(i, bottom + pct / 2.0, txt, ha="center", va="center",
                        fontsize=FS_VAL, color=fg, zorder=4, linespacing=1.25)
            bottom += pct

    ax.set_xticks(range(len(data)))
    ax.set_xticklabels([wl for wl, _, _ in data], fontsize=FS_TICK, color=INK)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "25", "50", "75", "100%"], fontsize=FS_TICK,
                       color=INK_MUTED)
    ax.set_ylabel("chunks", fontsize=FS_TICK, color=INK)
    ax.set_title(title, fontsize=FS_TITLE, color=INK, pad=5)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)

    # Chunk count under each bar, and the byte-shuffle share when a lossless
    # arm supplied one. Both are context for the segment above, not series.
    for i, (wl, n, _segs) in enumerate(data):
        ax.annotate(f"{n} chunks", (i, 0), xytext=(0, -15),
                    textcoords="offset points", ha="center",
                    fontsize=FS_VAL, color=INK_MUTED, annotation_clip=False)
        if wl in shuffles:
            ax.annotate(f"shuffled {shuffles[wl]:.0f}%", (i, 0), xytext=(0, -25),
                        textcoords="offset points", ha="center",
                        fontsize=FS_VAL, color=INK_MUTED, annotation_clip=False)

    seen = sorted(weight, key=lambda c: (c == "(other)", -weight[c]))
    handles = [Patch(facecolor=colour_for(c, taken), edgecolor="white",
                     linewidth=0.6, label=pretty(c)) for c in seen]
    fig.legend(handles=handles, loc="lower center", ncol=min(6, len(handles)),
               frameon=False, fontsize=FS_LEG, labelcolor=INK,
               bbox_to_anchor=(0.5, 0.005), handlelength=1.1,
               columnspacing=1.3, handletextpad=0.45)

    fig.savefig(out_path, dpi=300, facecolor="white")
    plt.close(fig)


def draw_count(data, out_path: str, title: str) -> None:
    """The same stack, absolute chunk counts instead of shares.

    Normalising to 100% makes every workload look equally sampled; this keeps
    the sample sizes visible (AI 1804 chunks against WarpX 5130) at the cost of
    making the smaller workloads' composition harder to read.
    """
    legend_h, foot_h, panel_h, head_h = 0.44, 0.26, 2.05, 0.40
    fig_h = legend_h + foot_h + panel_h + head_h
    fig = plt.figure(figsize=(FIG_W, fig_h))
    ax = fig.add_axes([0.085, (legend_h + foot_h) / fig_h, 0.895,
                       panel_h / fig_h])

    taken: dict = {}
    weight: dict = collections.defaultdict(float)
    for i, (wl, n, segs) in enumerate(data):
        bottom = 0.0
        for codec, count, pct in segs:
            weight[codec] += pct
            ax.bar(i, count, BAR_W, bottom=bottom,
                   color=colour_for(codec, taken), edgecolor="white",
                   linewidth=0.6, zorder=3)
            bottom += count
        ax.annotate(f"{n}", (i, n), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=FS_VAL, color=INK_MUTED)

    ax.set_xticks(range(len(data)))
    ax.set_xticklabels([wl for wl, _, _ in data], fontsize=FS_TICK, color=INK)
    ax.set_ylabel("chunks", fontsize=FS_TICK, color=INK)
    ax.tick_params(axis="y", labelsize=FS_TICK, colors=INK_MUTED)
    ax.set_title(title, fontsize=FS_TITLE, color=INK, pad=6)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)
    _legend(fig, weight, taken)
    fig.savefig(out_path, dpi=300, facecolor="white")
    plt.close(fig)


def _legend(fig, weight, taken, extra=None) -> None:
    """One shared legend, ordered by total share, with "(other)" pinned last.

    :param fig: the figure to attach to
    :param weight: {codec: summed share} driving the order
    :param taken: the fallback-colour map, so a codec keeps its hue
    :param extra: optional trailing Patch, e.g. the heat map's tie marker
    """
    seen = sorted(weight, key=lambda c: (c == "(other)", -weight[c]))
    handles = [Patch(facecolor=colour_for(c, taken), edgecolor="white",
                     linewidth=0.6, label=pretty(c)) for c in seen]
    if extra is not None:
        handles.append(extra)
    fig.legend(handles=handles, loc="lower center",
               ncol=min(7, len(handles)), frameon=False, fontsize=FS_LEG,
               labelcolor=INK, bbox_to_anchor=(0.5, 0.012), handlelength=1.0,
               handleheight=0.9, columnspacing=1.25, handletextpad=0.4)


def draw_timeline(series, out_path: str, title: str, nbins: int) -> None:
    """One panel per workload: y = simulation time, x = codec share at that time.

    Rows are drawn on a NORMALISED y axis, so workloads with different dump
    counts (AI 11 epochs, Nyx 559 plotfiles) fill the same height and stay
    comparable row for row.

    :param series: [(workload, bins, lo, hi, total)] in display order
    :param nbins: the bin cap that was requested; unused here but kept so the
                  three modes share one call signature
    """
    del nbins
    # Vertical budget, inches, bottom-up. Every band is explicit because an
    # earlier version let the suptitle land on top of the panel titles.
    legend_h, foot_h, panel_h, head_h = 0.44, 0.22, 2.05, 0.46
    fig_h = legend_h + foot_h + panel_h + head_h
    fig = plt.figure(figsize=(FIG_W, fig_h))
    n = len(series)
    left, right, gap = 0.052, 0.992, 0.020
    w = ((right - left) - gap * (n - 1)) / n
    y0 = (legend_h + foot_h) / fig_h

    taken: dict = {}
    weight: dict = collections.defaultdict(float)
    for i, (wl, bins, lo, hi, total) in enumerate(series):
        ax = fig.add_axes([left + i * (w + gap), y0, w, panel_h / fig_h])
        k = len(bins)
        for b, cnt in enumerate(bins):
            tot = sum(cnt.values())
            if not tot:
                continue
            x = 0.0
            for codec, c in cnt.most_common():
                pct = 100.0 * c / tot
                weight[codec] += pct
                ax.barh(b / k, pct, 1.0 / k, left=x, align="edge",
                        color=colour_for(codec, taken), edgecolor="none",
                        zorder=3)
                x += pct
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1)
        ax.set_title(wl, fontsize=FS_TICK, color=INK, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        # Step range under its own panel, not in the title: a two-line title
        # collided with the figure title above it.
        ax.annotate(f"steps {lo}\u2013{hi}", (0.5, 0), xycoords="axes fraction",
                    xytext=(0, -9), textcoords="offset points", ha="center",
                    va="top", fontsize=FS_VAL, color=INK_MUTED,
                    annotation_clip=False)
        if i == 0:
            ax.set_ylabel("early  \u2192  late", fontsize=FS_VAL, color=INK_MUTED,
                          labelpad=2)

    fig.text(0.5, 1 - 0.13 / fig_h, title, ha="center", va="top",
             fontsize=FS_TITLE, color=INK)
    _legend(fig, weight, taken)
    fig.savefig(out_path, dpi=300, facecolor="white")
    plt.close(fig)


def draw_heat(series, out_path: str, title: str, nbins: int) -> None:
    """One column per workload, y = simulation time, cell = DOMINANT codec.

    The most compact of the three modes and the most lossy: a row where two
    codecs split 51/49 draws identically to one where a single codec took every
    chunk. Rows whose winner is under two thirds are therefore hatched, so a
    near-tie cannot be read as unanimity.

    :param series: [(workload, bins, lo, hi, total)] in display order
    :param nbins: requested bin cap; each column is stretched to fill the height
    """
    del nbins
    legend_h, foot_h, panel_h, head_h = 0.44, 0.34, 2.15, 0.42
    fig_h = legend_h + foot_h + panel_h + head_h
    fig = plt.figure(figsize=(FIG_W, fig_h))
    ax = fig.add_axes([0.075, (legend_h + foot_h) / fig_h, 0.915,
                       panel_h / fig_h])

    taken: dict = {}
    weight: dict = collections.defaultdict(float)
    col_w = 0.74
    for i, (wl, bins, lo, hi, total) in enumerate(series):
        k = len(bins)
        for b, cnt in enumerate(bins):
            tot = sum(cnt.values())
            if not tot:
                continue
            codec, c = cnt.most_common(1)[0]
            weight[codec] += 1.0
            ax.add_patch(plt.Rectangle((i - col_w / 2, b / k), col_w, 1.0 / k,
                                       facecolor=colour_for(codec, taken),
                                       edgecolor="none", zorder=3))
            if c / tot < 0.667:
                ax.add_patch(plt.Rectangle((i - col_w / 2, b / k), col_w,
                                           1.0 / k, facecolor="none",
                                           edgecolor=(1, 1, 1, 0.45),
                                           hatch="......", linewidth=0.0,
                                           zorder=4))

    # Name each contiguous run of one codec, so the column reads without the
    # legend. Runs shorter than a fifth of the column have no room for text.
    for i, (wl, bins, lo, hi, total) in enumerate(series):
        k = len(bins)
        run_codec, run_start = None, 0
        winners = [(cnt.most_common(1)[0][0] if cnt else None) for cnt in bins]
        for b in range(k + 1):
            here = winners[b] if b < k else None
            if here != run_codec:
                if run_codec is not None and (b - run_start) / k >= 0.20:
                    mid = (run_start + b) / 2.0 / k
                    ax.text(i, mid, pretty(run_codec), ha="center", va="center",
                            fontsize=FS_VAL, color="white", zorder=5)
                run_codec, run_start = here, b

    ax.set_xlim(-0.62, len(series) - 0.38)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(len(series)))
    ax.set_xticklabels([wl for wl, _, _, _, _ in series], fontsize=FS_TICK,
                       color=INK)
    ax.set_yticks([])
    ax.set_ylabel("early  \u2192  late", fontsize=FS_VAL, color=INK_MUTED,
                  labelpad=2)
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(length=0, pad=4)
    for sp in ax.spines.values():
        sp.set_visible(False)

    for i, (wl, bins, lo, hi, total) in enumerate(series):
        ax.annotate(f"steps {lo}\u2013{hi}", (i, 0), xytext=(0, -9),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=FS_VAL, color=INK_MUTED, annotation_clip=False)

    fig.text(0.5, 1 - 0.13 / fig_h, title, ha="center", va="top",
             fontsize=FS_TITLE, color=INK)
    tie = Patch(facecolor="#c9c8c4", edgecolor=(1, 1, 1, 0.45), hatch="......",
                linewidth=0.0, label="winner under 2/3")
    _legend(fig, weight, taken, extra=tie)
    fig.savefig(out_path, dpi=300, facecolor="white")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camp", action="append", required=True, metavar="NAME=DIR",
                    help="workload name and its campaign directory; repeatable")
    ap.add_argument("--arm", default="neuropress", help="arm tag (default neuropress)")
    ap.add_argument("--shuffle-from", default="np_only", metavar="ARM",
                    help="LOSSLESS arm to read the exact byte-shuffle share "
                         "from; pass '' to omit the line")
    ap.add_argument("--top", type=int, default=4, help="segments before (other)")
    ap.add_argument("--mode", default="share",
                    choices=["share", "count", "timeline", "heat"],
                    help="share: 100%% stack per workload (default). count: the "
                         "same stack in absolute chunks. timeline: one panel "
                         "per workload, y = simulation time. heat: one column "
                         "per workload, cell = dominant codec at that time.")
    ap.add_argument("--bins", type=int, default=32,
                    help="timestep bins for --mode timeline/heat (default 32)")
    ap.add_argument("--out", default="live", help="output directory")
    ap.add_argument("--name", default=None,
                    help="output filename (default fig9_choices[_<mode>].png)")
    ap.add_argument("--title", default=DEFAULT_TITLE)
    args = ap.parse_args()

    camps = []
    for spec in args.camp:
        if "=" not in spec:
            print(f"--camp wants NAME=DIR, got {spec!r}", file=sys.stderr)
            return 2
        name, _, path = spec.partition("=")
        camps.append((name, path))
    order = {w: i for i, w in enumerate(WORKLOAD_ORDER)}
    camps.sort(key=lambda c: (order.get(c[0], len(order)), c[0]))

    data, shuffles = [], {}
    for name, root in camps:
        blobs = find_blobs(root, args.arm)
        if not blobs:
            print(f"  {name}: no {args.arm} arm, skipped", file=sys.stderr)
            continue
        n, segs = read_choices(blobs, args.top)
        data.append((name, n, segs))
        if args.shuffle_from:
            lossless = find_blobs(root, args.shuffle_from)
            if lossless:
                shuffles[name] = shuffle_pct(lossless)
    if not data:
        print("no arms found", file=sys.stderr)
        return 1

    os.makedirs(args.out, exist_ok=True)
    name = args.name or ("fig9_choices.png" if args.mode == "share"
                         else f"fig9_choices_{args.mode}.png")
    path = os.path.join(args.out, name)

    if args.mode in ("timeline", "heat"):
        series = []
        for name_wl, root in camps:
            blobs = find_blobs(root, args.arm)
            if not blobs:
                continue
            with open(blobs, newline="") as fh:
                rows = list(csv.DictReader(fh))
            bins, lo, hi = bin_steps(rows, args.bins)
            if not bins:
                print(f"  {name_wl}: no timestep in the blob names, skipped",
                      file=sys.stderr)
                continue
            series.append((name_wl, bins, lo, hi, len(rows)))
        if not series:
            print("no workload carried a timestep", file=sys.stderr)
            return 1
        title = args.title
        if title == DEFAULT_TITLE:
            title = "Codec chosen per chunk by NeuroPress, over simulation time"
        if args.mode == "timeline":
            draw_timeline(series, path, title, args.bins)
        else:
            draw_heat(series, path, title, args.bins)
        for wl, bins, lo, hi, total in series:
            first = next((b for b in bins if b), collections.Counter())
            last = next((b for b in reversed(bins) if b), collections.Counter())
            f = first.most_common(1)[0][0] if first else "-"
            l = last.most_common(1)[0][0] if last else "-"
            print(f"  {wl:7s} steps {lo}-{hi}   early {pretty(f)}  ->  "
                  f"late {pretty(l)}")
        print(f"wrote {path}")
        return 0

    if args.mode == "count":
        draw_count(data, path, args.title.replace("by workload",
                                                  "by workload (chunk counts)"))
    else:
        draw(data, shuffles, path, args.title)

    for wl, n, segs in data:
        top = segs[0]
        cover = 0.0
        need = 0
        for _c, _v, pct in segs:
            if cover >= 95.0:
                break
            cover += pct
            need += 1
        sh = f", shuffled {shuffles[wl]:.0f}%" if wl in shuffles else ""
        print(f"  {wl:7s} {n:5d} chunks   top {pretty(top[0])} {top[2]:.1f}%   "
              f"{need} codec(s) cover 95%{sh}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
