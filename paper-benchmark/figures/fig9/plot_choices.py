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
LABEL_FLOOR = 6.0                  # percent: below this a segment goes unlabelled
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

#: Printed instead of the wire name; "raw" is an outcome, not a library.
PRETTY = {"raw(not-beneficial)": "stored raw"}


def pretty(codec: str) -> str:
    """Legend text for a codec, with the nvcomp- prefix dropped."""
    if codec in PRETTY:
        return PRETTY[codec]
    return codec[len("nvcomp-"):] if codec.startswith("nvcomp-") else codec


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
                ax.text(i, bottom + pct / 2.0, f"{pct:.0f}%", ha="center",
                        va="center", fontsize=FS_VAL, color=fg, zorder=4)
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
    ap.add_argument("--out", default="live", help="output directory")
    ap.add_argument("--name", default="fig9_choices.png", help="output filename")
    ap.add_argument("--title",
                    default="Codec chosen per chunk by NeuroPress, by workload")
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
    path = os.path.join(args.out, args.name)
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
