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
import matplotlib as mpl
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

# ---------------------------------------------------------------------------
# PAPER STYLE -- the serif, gridded, boxed-legend look, as a named style rather
# than a fork of the plotter. Everything below is a deviation from the default
# sans style above; the data path is shared.
# ---------------------------------------------------------------------------
#: Serif stack, best first. Nimbus Roman is URW's Times clone and is what a
#: Times-set paper matches; STIXGeneral is the maths-companion fallback.
SERIF = ["Nimbus Roman", "STIXGeneral", "DejaVu Serif", "serif"]

# GREYSCALE SEPARATION, which EuroSys requires outright ("Graphs and figures
# should be readable when printed in grayscale, without magnification").
# Hue alone does not survive the conversion: measured on this palette, `ans`
# and `snappy` land on 118 and 119 of 255 -- indistinguishable -- and eight
# pairs fall inside 0.06 of relative luminance. Identity is therefore carried
# by the segment's own label first and a texture second, never by colour.
# Ordered so the most-chosen codecs get the least busy fills.
CODEC_HATCH = {
    "nvcomp-ans":          "",
    "nvcomp-bitcomp":      "/",
    "nvcomp-zstd":         "\\",
    "raw(not-beneficial)": "",
    "nvcomp-lz4":          "x",
    "nvcomp-snappy":       "-",
    "nvcomp-cascaded":     ".",
    "nvcomp-gdeflate":     "o",
    "nvcomp-deflate":      "+",
    "(other)":             "",
}


def hatch_for(codec: str) -> str:
    """Texture for a codec, or "" for a solid fill."""
    return CODEC_HATCH.get(codec, "")
# EuroSys 2027 sizes the figure and its type together: the text block is
# 178 x 229 mm (7 x 9 in), and ">=10-point font ... applies to all text,
# including figures and captions". A font size only means 10 pt if the figure
# is placed at the width it was drawn at, so fig_w IS the 7 in text block and
# the caller must \includegraphics[width=\textwidth] with no rescaling.
# Shrinking this figure to one column scales every number below with it and
# breaks the rule -- redraw at fig_w 3.35 instead, do not scale.
PAPER = {
    "fig_w": 7.0,          # inches = the 178 mm text block, placed 1:1
    "plot_h": 2.30,
    "bar_w": 0.62,
    "fs_title": 11,
    "fs_axis": 10,         # y label
    "fs_tick": 10,
    "fs_val": 10,          # in-segment label
    "fs_key": 10,          # the lines under each bar
    "fs_leg": 10,
    "grid": "#b8b8b8",
    "edge": "#3a3a3a",     # bar outline
    "label_floor": 3.0,    # percent: smaller segments carry no number
    "name_floor": 9.0,     # percent: above this the segment names its codec
}

#: No title by default: in a paper the caption carries it, and a title
#: inside the figure duplicates the caption and costs plot height. Pass
#: --title to put one back, e.g. for a slide.
DEFAULT_TITLE = ""

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


def assert_no_overlap(fig, artists, legend) -> None:
    """Fail loudly if any labelled artist collides with the legend box.

    Layout here is hand-budgeted in inches, which is precise but brittle: a
    font-size or legend-row change moves a box and the collision is only
    visible by opening the PNG. This re-measures after layout instead.

    :param fig: the drawn figure, before savefig
    :param artists: text artists that must stay clear of the legend
    :param legend: the legend whose frame they must not touch
    :raises RuntimeError: naming the first collision found
    """
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    lb = legend.get_window_extent(rend)
    for a in artists:
        ab = a.get_window_extent(rend)
        if ab.overlaps(lb):
            raise RuntimeError(
                f"layout collision: {a.get_text()!r} overlaps the legend "
                f"(text y {ab.y0:.0f}-{ab.y1:.0f}, legend y {lb.y0:.0f}-"
                f"{lb.y1:.0f}); raise legend_h/key_h in PAPER")


def check_eurosys(fig, ax, segs_drawn, min_pt: float = 10.0) -> list:
    """Check a drawn figure against the EuroSys 2027 CFP, and report failures.

    The CFP sets two rules that a default chart breaks, both checked here
    rather than by opening the PNG:

      ">=10-point font ... applies to all text, including figures and captions"
        Every Text artist is measured. A point size only MEANS 10 pt if the
        figure is placed at the width it was drawn at, so the figure width is
        reported alongside for the caller to match with \includegraphics.

      "Graphs and figures should be readable when printed in grayscale"
        Hue is dropped and each vertically adjacent pair of segments is
        checked: it must differ in relative luminance, OR carry a different
        hatch, OR both be labelled. Colour alone never counts.

    :param fig: the drawn figure
    :param ax: its axis
    :param segs_drawn: [(workload, [(codec, pct, colour, hatch, labelled)])]
    :param min_pt: the CFP minimum, in points
    :return: list of human-readable failures; empty means compliant
    """
    fails = []
    w_in, h_in = fig.get_size_inches()
    if w_in > 7.001:
        fails.append(f"width {w_in:.2f} in exceeds the 7 in (178 mm) text block")

    small = sorted({round(t.get_fontsize(), 1) for t in fig.findobj(mpl.text.Text)
                    if t.get_text().strip() and t.get_fontsize() < min_pt})
    if small:
        fails.append(f"font sizes below {min_pt} pt: {small}")

    def lum(hex_or_rgb):
        r, g, b = mpl.colors.to_rgb(hex_or_rgb)
        f = lambda c: c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
        return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)

    for wl, segs in segs_drawn:
        for (c1, p1, col1, h1, lab1), (c2, p2, col2, h2, lab2) in zip(segs, segs[1:]):
            if abs(lum(col1) - lum(col2)) >= 0.06:
                continue                      # separable by tone alone
            if (h1 or "") != (h2 or ""):
                continue                      # separable by texture
            if lab1 and lab2:
                continue                      # each says its own name
            fails.append(
                f"{wl}: {pretty(c1)} and {pretty(c2)} are adjacent, differ by "
                f"{abs(lum(col1) - lum(col2)):.3f} luminance, share a hatch, "
                f"and are not both labelled -- they merge in grayscale")
    return fails


def draw_paper(data, shuffles, out_path: str, title: str) -> None:
    """The serif, gridded, boxed-legend rendering of the share chart.

    Differences from draw() that are deliberate, not incidental:

      SEGMENTS CARRY THE PERCENTAGE ONLY, not the codec name. That trades the
      legend-free reading draw() gives for a cleaner bar, and costs the reader
      one colour lookup per segment. Keep the legend below the axis so the
      lookup is short.
      EVERY CODEC GETS ITS OWN SEGMENT at a high enough --top; there is no
      "(other)" tail in this style, so pass --top 8 to match.

    :param data: [(workload, chunks, [(codec, count, pct)])] in display order
    :param shuffles: {workload: pct} or {} to omit the shuffle line
    :param out_path: PNG to write
    :param title: bold serif heading; "" draws none
    """
    P = PAPER
    with mpl.rc_context({"font.family": "serif", "font.serif": SERIF,
                         "mathtext.fontset": "stix",
                         "hatch.linewidth": 0.6}):
        # Measured, not guessed. The key under the axis holds up to three
        # 10 pt lines (workload / chunks / shuffled), the lowest of which sits
        # 40 pt below the axis; the legend box is two 10 pt rows plus its own
        # border padding, about 0.58 in tall, sitting 0.012 fig-fractions up.
        # The previous 0.52 + 0.62 put the legend's top edge through the
        # "shuffled" line. assert_no_overlap() below re-checks this after
        # layout, so a font or row-count change cannot silently reintroduce it.
        legend_h = 0.70
        key_h = 0.68 if shuffles else 0.40
        head_h = 0.34 if title else 0.12
        fig_h = legend_h + key_h + P["plot_h"] + head_h
        fig = plt.figure(figsize=(P["fig_w"], fig_h))
        ax = fig.add_axes([0.085, (legend_h + key_h) / fig_h, 0.885,
                           P["plot_h"] / fig_h])

        taken: dict = {}
        weight: dict = collections.defaultdict(float)
        segs_drawn = []
        for i, (wl, _n, segs) in enumerate(data):
            bottom = 0.0
            drawn = []
            for codec, _c, pct in segs:
                weight[codec] += pct
                ax.bar(i, pct, P["bar_w"], bottom=bottom,
                       color=colour_for(codec, taken), edgecolor=P["edge"],
                       linewidth=0.7, hatch=hatch_for(codec), zorder=3)
                if pct >= P["label_floor"]:
                    fg = INK if codec == "(other)" else "white"
                    # NAME, not just a number, wherever it fits: in greyscale
                    # the fill cannot identify the codec, so the text must.
                    txt = (f"{pretty(codec)}\n{pct:.0f}%"
                           if pct >= P["name_floor"] else f"{pct:.0f}%")
                    ax.text(i, bottom + pct / 2.0, txt, ha="center",
                            va="center", fontsize=P["fs_val"], color=fg,
                            zorder=5, linespacing=1.2)
                drawn.append((codec, pct, colour_for(codec, taken),
                              hatch_for(codec), pct >= P["name_floor"]))
                bottom += pct
            segs_drawn.append((wl, drawn))

        ax.set_xticks(range(len(data)))
        ax.set_xticklabels([wl for wl, _, _ in data], fontsize=P["fs_tick"],
                           color="black")
        ax.set_xlim(-0.62, len(data) - 0.38)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.tick_params(axis="y", labelsize=P["fs_tick"], colors="black")
        ax.set_ylabel("Percentage of chunks (%)", fontsize=P["fs_axis"],
                      color="black", labelpad=6)
        if title:
            ax.set_title(title, fontsize=P["fs_title"], color="black",
                         fontweight="bold", pad=12)
        ax.yaxis.grid(True, color=P["grid"], linewidth=0.8, linestyle=(0, (5, 4)),
                      zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("black")
            ax.spines[sp].set_linewidth(0.9)

        keys = []
        for i, (wl, n, _segs) in enumerate(data):
            keys.append(ax.annotate(
                f"{n} chunks", (i, 0), xytext=(0, -26),
                textcoords="offset points", ha="center",
                fontsize=P["fs_key"], color="black", annotation_clip=False))
            if wl in shuffles:
                keys.append(ax.annotate(
                    f"shuffled {shuffles[wl]:.0f}%", (i, 0), xytext=(0, -40),
                    textcoords="offset points", ha="center",
                    fontsize=P["fs_key"], color="black",
                    annotation_clip=False))

        seen = sorted(weight, key=lambda c: (c == "(other)", -weight[c]))
        handles = [Patch(facecolor=colour_for(c, taken), edgecolor=P["edge"],
                         linewidth=0.7, hatch=hatch_for(c), label=pretty(c))
                   for c in seen]
        leg = fig.legend(handles=handles, loc="lower center",
                         ncol=min(5, len(handles)), fontsize=P["fs_leg"],
                         labelcolor="black", bbox_to_anchor=(0.5, 0.012),
                         handlelength=1.2, handleheight=1.0,
                         columnspacing=1.5, handletextpad=0.5,
                         frameon=True, fancybox=True, borderpad=0.7)
        leg.get_frame().set_edgecolor("#cccccc")
        leg.get_frame().set_linewidth(0.9)
        leg.get_frame().set_facecolor("white")

        assert_no_overlap(fig, keys, leg)
        fails = check_eurosys(fig, ax, segs_drawn)
        if fails:
            print("EuroSys compliance FAILED:", file=sys.stderr)
            for f in fails:
                print(f"  - {f}", file=sys.stderr)
        else:
            w, h = fig.get_size_inches()
            print(f"  EuroSys checks pass: {w:.2f} x {h:.2f} in, all text "
                  f">= 10 pt, every adjacent pair separable in grayscale")
        fig.savefig(out_path, dpi=300, facecolor="white")
        plt.close(fig)


def draw(data, shuffles, out_path: str, title: str) -> None:
    """Draw the stacked chart.

    :param data: [(workload, chunks, [(codec, count, pct)])] in display order
    :param shuffles: {workload: pct} or {} to omit the shuffle line
    :param out_path: PNG to write
    :param title: axis title
    """
    # Explicit vertical budget, inches, bottom-up: legend band, then the key
    # under the axis (workload, chunk count, shuffle line), then the plot, then
    # whatever the title needs. Letting matplotlib decide left a dead band the
    # height of a title that is no longer drawn.
    legend_h = 0.44
    key_h = 0.54 if shuffles else 0.34
    head_h = 0.34 if title else 0.10
    fig_h = legend_h + key_h + PLOT_H + head_h
    fig = plt.figure(figsize=(FIG_W, fig_h))
    ax = fig.add_axes([0.055, (legend_h + key_h) / fig_h, 0.925,
                       PLOT_H / fig_h])

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
    if title:
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
    _legend(fig, weight, taken)
    fig.savefig(out_path, dpi=300, facecolor="white")
    plt.close(fig)


def draw_count(data, out_path: str, title: str) -> None:
    """The same stack, absolute chunk counts instead of shares.

    Normalising to 100% makes every workload look equally sampled; this keeps
    the sample sizes visible (AI 1804 chunks against WarpX 5130) at the cost of
    making the smaller workloads' composition harder to read.
    """
    legend_h, foot_h, panel_h = 0.44, 0.26, 2.05
    head_h = 0.40 if title else 0.14
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
    if title:
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
    legend_h, foot_h, panel_h = 0.44, 0.22, 2.05
    head_h = 0.46 if title else 0.18
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

    if title:
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
    legend_h, foot_h, panel_h = 0.44, 0.34, 2.15
    head_h = 0.42 if title else 0.16
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

    if title:
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
    ap.add_argument("--style", default="plain", choices=["plain", "paper"],
                    help="plain: the sans, label-in-bar default. paper: serif, "
                         "gridded, boxed legend, percentages only (share mode "
                         "only; pair with --top 8).")
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
    elif args.style == "paper":
        draw_paper(data, shuffles, path, args.title)
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
