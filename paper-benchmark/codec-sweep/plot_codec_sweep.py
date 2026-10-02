#!/usr/bin/env python3
"""Plot a codec_sweep run set: ratio vs throughput, and run-to-run spread.

Reads every complete repetition under ROOT (ROOT/rep*/<wl>_chunk4m.csv whose
summary.txt holds the final table), and draws

  <wl>_tradeoff.png   compression ratio vs aggregate throughput, one panel for
                      compression and one for decompression; the horizontal bar
                      on each codec spans the min..max over the repetitions.
  <wl>_run_spread.png (max - min) / mean throughput per codec, in percent.

Aggregate throughput is total bytes over the summed CUDA-event time of the
successful round trips, the same number codec_sweep prints.

  plot_codec_sweep.py ROOT [--wl nyx] [--out DIR] [--csv <wl>_chunk4m.csv]
"""
import argparse
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.text import Text  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "..", "figures", "codec-sweep")

# Chart chrome (light surface) and the first three categorical slots, which
# validate all-pairs for a scatter.
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

# codec -> (label, family). Families carry the colour; every point is named.
CODECS = {
    "nvcomp-lz4": ("LZ4", "nvCOMP"),
    "nvcomp-snappy": ("Snappy", "nvCOMP"),
    "nvcomp-zstd": ("Zstd", "nvCOMP"),
    "nvcomp-gdeflate": ("GDeflate", "nvCOMP"),
    "nvcomp-deflate": ("Deflate", "nvCOMP"),
    "nvcomp-ans": ("ANS", "nvCOMP"),
    "nvcomp-cascaded": ("Cascaded", "nvCOMP"),
    "nvcomp-bitcomp": ("Bitcomp", "nvCOMP"),
    "ndzip": ("ndzip", "Float-specific"),
    "spspeed": ("SPspeed", "Float-specific"),
    "spratio": ("SPratio", "Float-specific"),
    "gpulz": ("GPULZ", "GPULZ"),
}
FAMILY_COLOR = {"nvCOMP": BLUE, "Float-specific": ORANGE, "GPULZ": AQUA}


def load_reps(root, csv_name):
    """Read every finished repetition under root.

    @param root     directory holding rep*/ subdirectories
    @param csv_name per-chunk CSV name inside each repetition
    @return list of (rep name, DataFrame); unfinished reps are skipped
    """
    reps = []
    for d in sorted(glob.glob(os.path.join(root, "rep*"))):
        csv = os.path.join(d, csv_name)
        summary = os.path.join(d, "summary.txt")
        if not (os.path.isdir(d) and os.path.exists(csv)):
            continue
        done = os.path.exists(summary) and any(
            line.startswith("codec ") for line in open(summary))
        if not done:
            print(f"skipping {d}: no final summary (unfinished run)",
                  file=sys.stderr)
            continue
        reps.append((os.path.basename(d), pd.read_csv(csv)))
    rows = {len(df) for _, df in reps}
    if len(rows) > 1:
        sys.exit(f"repetitions differ in row count: {sorted(rows)}")
    return reps


def per_rep_stats(reps):
    """Aggregate ratio and throughput per (rep, codec).

    @param reps output of load_reps
    @return DataFrame: rep, codec, ratio, comp_gbs, decomp_gbs, failed
    """
    out = []
    for name, df in reps:
        for codec, g in df.groupby("codec"):
            ok = g[g["ok"] == 1]
            out.append({
                "rep": name, "codec": codec,
                "failed": int((g["ok"] != 1).sum()),
                "ratio": ok["bytes"].sum() / ok["comp_bytes"].sum(),
                # bytes per ms / 1e6 = GB/s
                "comp_gbs": ok["bytes"].sum() / ok["comp_ms"].sum() / 1e6,
                "decomp_gbs": ok["bytes"].sum() / ok["decomp_ms"].sum() / 1e6,
            })
    return pd.DataFrame(out)


def summarize(stats):
    """Mean, min, max and spread over repetitions per codec.

    @param stats output of per_rep_stats
    @return DataFrame indexed by codec
    """
    agg = stats.groupby("codec").agg(
        ratio=("ratio", "mean"), failed=("failed", "sum"),
        comp_mean=("comp_gbs", "mean"), comp_min=("comp_gbs", "min"),
        comp_max=("comp_gbs", "max"), decomp_mean=("decomp_gbs", "mean"),
        decomp_min=("decomp_gbs", "min"), decomp_max=("decomp_gbs", "max"))
    for k in ("comp", "decomp"):
        agg[f"{k}_spread_pct"] = (
            100 * (agg[f"{k}_max"] - agg[f"{k}_min"]) / agg[f"{k}_mean"])
    return agg.loc[[c for c in CODECS if c in agg.index]]


def style_axes(ax):
    """Recessive grid and axes on the chart surface."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=9)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def place_labels(ax, points, labels):
    """Name every point, choosing per point the first offset whose text box
    stays inside the axes and overlaps no earlier label and no marker. Call it
    after the figure layout is final: the check is in pixels.

    @param ax     axes already holding the markers
    @param points list of (x, y) in data coordinates
    @param labels list of strings, same order
    """
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    frame = ax.get_window_extent(rend)
    marker_boxes = []
    for x, y in points:
        px, py = ax.transData.transform((x, y))
        marker_boxes.append((px - 6, py - 6, px + 6, py + 6))
    placed = []
    # Beside the point first; then a ring further out, with a leader line.
    near = [(8, 4, "left", "bottom"), (8, -4, "left", "top"),
            (-8, 4, "right", "bottom"), (-8, -4, "right", "top"),
            (0, 10, "center", "bottom"), (0, -10, "center", "top"),
            (14, 0, "left", "center"), (-14, 0, "right", "center")]
    far = [(10, -18, "left", "top"), (10, 18, "left", "bottom"),
           (-10, -18, "right", "top"), (-10, 18, "right", "bottom"),
           (0, -22, "center", "top"), (0, 22, "center", "bottom")]
    offsets = [o + (False,) for o in near] + [o + (True,) for o in far]
    leader = dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkA=0, shrinkB=5)

    def overlaps(a, b):
        return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])

    for i, ((x, y), text) in enumerate(zip(points, labels)):
        best = None
        for dx, dy, ha, va, lead in offsets:
            t = ax.annotate(text, (x, y), xytext=(dx, dy),
                            textcoords="offset points", ha=ha, va=va,
                            fontsize=9, color=INK,
                            arrowprops=leader if lead else None)
            # The text alone: an Annotation's own extent includes its leader
            # line, which always reaches the point and so its neighbours.
            t.update_positions(rend)
            bb = Text.get_window_extent(t, rend)
            box = (bb.x0 - 2, bb.y0 - 1, bb.x1 + 2, bb.y1 + 1)
            others = placed + [m for j, m in enumerate(marker_boxes) if j != i]
            inside = (box[0] >= frame.x0 and box[2] <= frame.x1 and
                      box[1] >= frame.y0 and box[3] <= frame.y1)
            if inside and not any(overlaps(box, o) for o in others):
                best = (t, box)
                break
            t.remove()
        if best is None:  # every offset collides: keep the first one
            dx, dy, ha, va, _ = offsets[0]
            t = ax.annotate(text, (x, y), xytext=(dx, dy),
                            textcoords="offset points", ha=ha, va=va,
                            fontsize=9, color=INK)
            bb = t.get_window_extent(rend)
            best = (t, (bb.x0, bb.y0, bb.x1, bb.y1))
        placed.append(best[1])


def plot_tradeoff(agg, nreps, out, wl, gib):
    """Ratio vs throughput, compression and decompression side by side.

    @param agg   output of summarize
    @param nreps repetitions behind the numbers
    @param out   PNG path
    @param wl    workload name for the title
    @param gib   input size per run, GiB
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.12, top=0.80,
                        wspace=0.06)
    labels = []
    for ax, k, title in ((axes[0], "comp", "Compression"),
                         (axes[1], "decomp", "Decompression")):
        style_axes(ax)
        pts, names = [], []
        for codec, r in agg.iterrows():
            label, fam = CODECS[codec]
            c = FAMILY_COLOR[fam]
            mean = r[f"{k}_mean"]
            if nreps > 1:
                ax.errorbar(mean, r["ratio"],
                            xerr=[[mean - r[f"{k}_min"]],
                                  [r[f"{k}_max"] - mean]],
                            fmt="none", ecolor=c, elinewidth=1.5,
                            capsize=3, zorder=2)
            ax.plot(mean, r["ratio"], "o", ms=8, color=c, mec=SURFACE,
                    mew=1.5, zorder=3)
            pts.append((mean, r["ratio"]))
            names.append(label)
        ax.set_xscale("log")
        ax.set_xlabel("Aggregate throughput (GB/s, log scale)")
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        labels.append((ax, pts, names))
    axes[0].set_ylabel("Compression ratio")
    handles = [plt.Line2D([], [], marker="o", ls="", ms=8, color=c,
                          mec=SURFACE, mew=1.5, label=f)
               for f, c in FAMILY_COLOR.items()]
    fig.legend(handles=handles, loc="upper right", ncol=3, frameon=False,
               fontsize=9, labelcolor=INK2, bbox_to_anchor=(0.99, 0.985))
    span = (f"bars span min to max over {nreps} runs (most are narrower than "
            "the marker)" if nreps > 1 else "1 run")
    fig.suptitle(f"Lossless GPU codecs on {wl}: {gib:.1f} GiB in 4 MiB "
                 "chunks, no preprocessing", x=0.01, ha="left", fontsize=12,
                 color=INK, y=0.985)
    fig.text(0.01, 0.895, "Up and to the right is better. Throughput = bytes "
             "/ summed CUDA-event time; " + span + ". A100-SXM4-40GB.",
             fontsize=9, color=INK2)
    for ax, pts, names in labels:
        place_labels(ax, pts, names)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_spread(agg, nreps, out, wl):
    """Run-to-run throughput spread per codec, compression and decompression.

    @param agg   output of summarize
    @param nreps repetitions behind the numbers
    @param out   PNG path
    @param wl    workload name for the title
    """
    order = agg.assign(
        worst=agg[["comp_spread_pct", "decomp_spread_pct"]].max(axis=1)
    ).sort_values("worst")
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.11, top=0.84)
    style_axes(ax)
    ax.grid(True, axis="x", color=GRID, linewidth=0.6)
    ax.grid(False, axis="y")
    y = np.arange(len(order))
    for col, c, name, off in (("comp_spread_pct", BLUE, "Compression", 0.12),
                              ("decomp_spread_pct", ORANGE, "Decompression",
                               -0.12)):
        ax.plot(order[col], y + off, "o", ms=8, color=c, mec=SURFACE,
                mew=1.5, label=name, zorder=3)
    ax.axvline(5, color=MUTED, linewidth=1, linestyle=(0, (3, 3)), zorder=1)
    ax.annotate("5%", (5, len(order) - 0.4), xytext=(4, 0),
                textcoords="offset points", fontsize=9, color=INK2,
                va="center")
    ax.set_yticks(y)
    ax.set_yticklabels([CODECS[c][0] for c in order.index])
    ax.set_xlim(left=0)
    ax.set_xlabel("Run-to-run spread of aggregate throughput, "
                  "(max - min) / mean (%)")
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK2)
    fig.suptitle(f"How much the {nreps} {wl} runs disagree", x=0.01,
                 ha="left", fontsize=12, color=INK)
    fig.text(0.01, 0.905, "Ratios are identical across runs (lossless); only "
             "time varies. Each run on its own node.", fontsize=9, color=INK2)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    """Load the repetitions, print the table, draw both figures."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root", help="run set directory holding rep*/")
    ap.add_argument("--out", default=DEFAULT_OUT, help="figure directory")
    ap.add_argument("--wl", default="nyx", help="workload (CSV prefix, names)")
    ap.add_argument("--csv", default=None, help="default <wl>_chunk4m.csv")
    a = ap.parse_args()
    reps = load_reps(a.root, a.csv or f"{a.wl}_chunk4m.csv")
    if not reps:
        sys.exit(f"no finished repetitions under {a.root}")
    agg = summarize(per_rep_stats(reps))
    if agg["failed"].any():
        print("WARNING: failed round trips:\n" +
              agg.loc[agg["failed"] > 0, "failed"].to_string(),
              file=sys.stderr)
    with pd.option_context("display.width", 160, "display.precision", 3,
                           "display.max_columns", None):
        print(f"{len(reps)} runs: {', '.join(n for n, _ in reps)}")
        print(agg[["ratio", "comp_mean", "comp_spread_pct", "decomp_mean",
                   "decomp_spread_pct"]])
    os.makedirs(a.out, exist_ok=True)
    df = reps[0][1]
    gib = df[df["codec"] == df["codec"].iloc[0]]["bytes"].sum() / 2**30
    plot_tradeoff(agg, len(reps), os.path.join(a.out, f"{a.wl}_tradeoff.png"),
                  a.wl, gib)
    plot_spread(agg, len(reps), os.path.join(a.out, f"{a.wl}_run_spread.png"),
                a.wl)
    print(f"wrote {a.out}/{a.wl}_tradeoff.png and {a.wl}_run_spread.png")


if __name__ == "__main__":
    main()
