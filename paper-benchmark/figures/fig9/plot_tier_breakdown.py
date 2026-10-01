#!/usr/bin/env python3
"""Where the time goes in figure 9 (a): codec, the rest of compute, device I/O.

    plot_tier_breakdown.py [--data DIR] [--out PNG] [--csv CSV]
    plot_tier_breakdown.py --campaign WL=DIR [--arm-root WL:TAG=DIR ...] --out PNG --csv CSV

For every workload and every tiering arm (Baseline, NP only, NP+Tier,
NP+Tier+Async) the bar is the measured figure 9 bar -- driver total minus
input read and H2D staging -- split three ways:

  Codec          sum of the chunks' compress_ms + preproc_ms (blobs.csv): the
                 compression kernels and the byte shuffle, CUDA-event timed.
  Other compute  the harness's compute (total - I/O) minus Codec: selection
                 (statistics, the NN, online SGD) and the runtime's per-chunk
                 path. For Baseline this is ALL of compute -- no codec runs.
  I/O            the harness's device I/O: the union of the bdev's per-write
                 intervals plus the final fsync. Untiered arms write to Lustre;
                 +Tier arms write a RAM tier that drains to node-local NVMe.

+Async drains the tier WHILE chunks are still being compressed, so its I/O
window overlaps compute and total - I/O understates it; the two compute
segments are then clipped to what the bar leaves; the codec time that ran
under the drain is drawn hatched.
--csv writes the numbers, compression ratio included.

--campaign reads a harness campaign directly (DIR = its <workload> dir, one
<tag>/fig9_runs.csv per arm) instead of the imported data/ folder, and draws
each arm as the MEAN of its recorded runs. Codec is then the mean over runs of each run's own blobs.csv. --arm-root
takes one arm of a workload from another campaign (e.g. a rerun).
"""
import statistics
import argparse
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WORKLOADS = [("vpic", "VPIC"), ("nyx", "Nyx"), ("lammps", "LAMMPS"), ("warpx", "WarpX")]
ARMS = [("baseline", "Baseline", "Base"), ("np_only", "NP only", "NP"),
        ("np_tier", "NP+Tier", "+Tier"), ("np_tier_async", "NP+Tier+Async", "+Async")]
SEG = [("codec", "Codec (compress + shuffle)", "#1c5cab"),
       ("other", "Other compute (selection, runtime)", "#8fb3de"),
       ("io", "Device I/O (write + fsync)", "#d6d5d1")]
FS = 10
INK, MUTED = "#2b2b2b", "#6b6a66"
SERIF = ["Times New Roman", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"]


def arm_row(data, wl, tag):
    """The measured split of one arm's imported figure 9 run.

    @param data figure 9 data directory.
    @param wl   workload directory name.
    @param tag  arm directory name.
    @return dict of total, io, compute, codec, other (seconds), ratio, out_mib,
            runs, and clipped (True when the codec does not fit total - io).
    """
    r = list(csv.DictReader(open(f"{data}/{wl}/{tag}/fig9.csv")))[-1]
    total, io, comp = (float(r[k]) * 60 for k in ("total_min", "io_min", "compute_min"))
    codec, out = 0.0, float(r["payload_mib"]) / float(r["ratio"])
    blobs = glob.glob(f"{data}/{wl}/{tag}/{tag}/r1/{tag}/blobs.csv")
    if blobs:
        rows = list(csv.DictReader(open(blobs[0])))
        codec = sum(float(x["compress_ms"] or 0) + float(x["preproc_ms"] or 0)
                    for x in rows) / 1e3
        out = sum(int(x["stored"]) for x in rows) / 2**20
    clipped = codec > comp
    return dict(total=total, io=io, compute=comp, codec=min(codec, comp),
                codec_measured=codec, other=max(comp - codec, 0.0),
                ratio=float(r["ratio"]), out_mib=out, runs=r["runs"], clipped=clipped)


def campaign_row(root, tag):
    """The mean split of one arm over every run a campaign recorded.

    @param root the campaign's workload directory.
    @param tag  arm directory name.
    @return the arm_row() dict, from means over runs, plus std (of the total).
    """
    runs = list(csv.DictReader(open(f"{root}/{tag}/fig9_runs.csv")))
    per = []
    for r in runs:
        blobs = glob.glob(f"{root}/{tag}/{tag}/r{r['rep']}/{tag}/blobs.csv")
        rows = list(csv.DictReader(open(blobs[0]))) if blobs else []
        per.append(dict(
            total=float(r["total_min"]) * 60, io=float(r["io_min"]) * 60,
            compute=float(r["compute_min"]) * 60, ratio=float(r["ratio"]),
            codec=sum(float(x["compress_ms"] or 0) + float(x["preproc_ms"] or 0)
                      for x in rows) / 1e3,
            out=sum(int(x["stored"]) for x in rows) / 2**20 if rows else 0.0))
    m = {k: statistics.mean(p[k] for p in per) for k in per[0]}
    return dict(total=m["total"], io=m["io"], compute=m["compute"],
                codec=min(m["codec"], m["compute"]), codec_measured=m["codec"],
                other=max(m["compute"] - m["codec"], 0.0), ratio=m["ratio"],
                out_mib=m["out"], runs=len(per), clipped=m["codec"] > m["compute"],
                std=statistics.stdev([p["total"] for p in per]) if len(per) > 1 else 0.0)


def draw(rows, workloads, out, csv_out):
    """Draw the grouped stacked bars and optionally write the table.

    @param rows      {(workload, arm tag): arm_row() or campaign_row() dict}.
    @param workloads the (dir name, label) pairs to draw, in order.
    @param out       output PNG path.
    @param csv_out   output CSV path, or None.
    """
    if csv_out:
        with open(csv_out, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["workload", "arm", "bar_s", "io_s", "compute_s", "codec_s",
                         "other_compute_s", "ratio", "written_mib", "runs"])
            for w, W in workloads:
                for t, n, _ in ARMS:
                    d = rows[(w, t)]
                    wr.writerow([W, n, f"{d['total']:.2f}", f"{d['io']:.2f}",
                                 f"{d['compute']:.2f}", f"{d['codec_measured']:.2f}",
                                 f"{d['compute'] - d['codec_measured']:.2f}",
                                 f"{d['ratio']:.3f}", f"{d['out_mib']:.0f}", d["runs"]])
        print(csv_out)

    plt.rcParams.update({"font.family": "serif", "font.serif": SERIF, "font.size": FS})
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    fig.subplots_adjust(left=0.085, right=0.995, top=0.85, bottom=0.205)
    width, gap = 0.8, 1.0
    ymax = max(d["total"] for d in rows.values()) * 1.08
    ax.set_xlim(-0.7, len(workloads) * (len(ARMS) + gap) - gap - 0.3)
    ax.set_ylim(0, ymax)
    # A value goes inside its segment only when the segment is taller than the
    # digits; the codec segment never is, so its value sits under the bar.
    px_per_s = ax.transData.transform((0, 1))[1] - ax.transData.transform((0, 0))[1]
    fit_s = FS * 0.8 * fig.dpi / 72 / px_per_s
    for g, (w, W) in enumerate(workloads):
        x0 = g * (len(ARMS) + gap)
        for i, (t, _, short) in enumerate(ARMS):
            d, x, bottom = rows[(w, t)], x0 + i, 0.0
            for key, _, col in SEG:
                ax.bar(x, d[key], bottom=bottom, width=width, color=col,
                       edgecolor="white", linewidth=0.5)
                if key != "codec" and d[key] >= fit_s:
                    ax.text(x, bottom + d[key] / 2, f"{d[key]:.1f}", ha="center",
                            va="center", fontsize=FS, color=INK)
                bottom += d[key]
            # +Async: the codec ran while the tier drained, inside the I/O window.
            # Draw that overlapped time as a hatched overlay, not a stacked segment.
            hidden = d["codec_measured"] - d["codec"]
            if hidden > 0.05:
                ax.bar(x, hidden, bottom=d["codec"] + d["other"], width=width * 0.6,
                       color="none", edgecolor=SEG[0][2], hatch="////", linewidth=0.8)
            if d["codec_measured"] > 0:
                ax.text(x, -ymax * 0.012, f"{d['codec_measured']:.1f}", ha="center",
                        va="top", fontsize=FS, color=SEG[0][2])
            ax.text(x, -ymax * 0.075, short, ha="right", va="top", fontsize=FS,
                    color=INK, rotation=40, rotation_mode="anchor")
        ax.text(x0 + (len(ARMS) - 1) / 2, -ymax * 0.25, W, ha="center", va="top",
                fontsize=FS + 1, fontweight="bold", color=INK)
    ax.set_xticks([])
    ax.set_ylabel("Time (s)", fontsize=FS, color=INK)
    ax.tick_params(axis="y", labelsize=FS, colors=MUTED, width=0.6)
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    handles = [Patch(facecolor=c, edgecolor="none", label=lab) for _, lab, c in SEG]
    handles.append(Patch(facecolor="none", edgecolor=SEG[0][2], hatch="////",
                         label="Codec that must overlap the drain (+Async)"))

    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=FS,
               bbox_to_anchor=(0.54, 1.0), handlelength=1.2, columnspacing=1.2)
    fig.savefig(out, dpi=300, facecolor="white")
    print(out)


def main():
    """Parse arguments and draw the figure."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=f"{HERE}/data")
    ap.add_argument("--out", default=f"{HERE}/live/tiering/fig9a_breakdown.png")
    ap.add_argument("--csv", default=f"{HERE}/live/tiering/fig9a_breakdown.csv")
    ap.add_argument("--campaign", nargs="*", default=[],
                    help="WL=DIR: read that workload from a campaign's workload dir")
    ap.add_argument("--arm-root", nargs="*", default=[],
                    help="WL:TAG=DIR: read that arm from another campaign")
    a = ap.parse_args()
    if a.campaign:
        camp = dict(kv.split("=", 1) for kv in a.campaign)
        over = dict(kv.split("=", 1) for kv in a.arm_root)
        workloads = [(w, W) for w, W in WORKLOADS if w in camp]
        rows = {(w, t): campaign_row(over.get(f"{w}:{t}", camp[w]), t)
                for w, _ in workloads for t, _, _ in ARMS}
    else:
        workloads = WORKLOADS
        rows = {(w, t): arm_row(a.data, w, t) for w, _ in WORKLOADS for t, _, _ in ARMS}
    draw(rows, workloads, a.out, a.csv)


if __name__ == "__main__":
    main()
