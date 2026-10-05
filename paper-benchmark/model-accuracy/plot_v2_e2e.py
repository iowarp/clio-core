#!/usr/bin/env python3
"""Measured end-to-end write and read time through Clio: NeuroPress v2 vs raw.

    plot_v2_e2e.py [--out PNG]

For each new-workloads dataset, the wall-clock time to write the whole dataset
through Clio (replay "stage+compress", setup excluded) and to read it back
(the timed "get+decompress" pass), divided by the dataset's size, for two
runs: storing it uncompressed (run_v2_workloads.sh raw, no compressor) and
NeuroPress v2 with learning only (learn) and with learning + exploration
(learnexp). The v2 bars are split
by the per-chunk phase log (phases.csv, summed over chunks). Write: v2
prediction (features + network + ranking), the element-type conversion for
v2's features, host-to-GPU copy, compress (shuffle included), exploration
(compressing and decompressing alternatives), v2 learning updates, the
decompress that measures the learning labels, and storage I/O (all host wall
clock, as the end-to-end time is). Read: decompress and storage I/O. The rest
of the measured wall time is Clio's runtime and everything else. Real I/O goes to one NVMe drive: the 4-tier bandwidths of
the cost model are not applied here.
"""
import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = "/mnt/nvme0/v2-work/runs"
FIGS = os.path.join(HERE, "..", "figures", "new-workloads", "nn-v2")
DATASETS = ("detector-frames dl-gpt2-kv dl-opt-relu-act consumer-nyx "
            "consumer-vpic-full consumer-lammps-full dl-pythia-ckpt genomics-reads "
            "hep-nanoaod dl-resnet18-train graph-orkut-full sparse-fem "
            "graph-livejournal-full dl-qwen-bf16 ref-lammps-b70-2000 "
            "ref-nyx-256-2000 ref-vpic-126-2000 ref-warpx-64x64x512-2000 "
            "astro-camels").split()
INK, INK2, RAW = "#1f2328", "#57606a", "#a0a7b0"
WRITE_PARTS = [("nn_ms", "v2 prediction (features + network + ranking)", "#2e86ab"),
               ("convert_ms", "type conversion for v2 features", "#5bc0de"),
               ("h2d_ms", "copy to GPU", "#8e7cc3"),
               ("compress_ms", "compress (shuffle included)", "#d1495b"),
               ("explore_ms", "v2 exploration (trying other codecs)", "#e08e0b"),
               ("sgd_ms", "v2 learning updates", "#7f3c8d"),
               ("label_ms", "v2 decompress for learning labels", "#f2b701"),
               ("io_ms", "storage I/O", "#3a7d44")]
READ_PARTS = [("decompress_ms", "decompress", "#d1495b"),
              ("io_ms", "storage I/O", "#3a7d44")]
OTHER = ("Clio runtime + everything else", "#d8dce1")


def wall(run):
    """(write s, read s) measured end to end from a run's stdout.log."""
    text = open(os.path.join(run, "stdout.log")).read()
    w = re.search(r"stage\+compress ([0-9.]+) s", text)
    r = re.search(r"READ 1/1.*?get\+decompress: ([0-9.]+) ms", text, re.S)
    return float(w.group(1)), float(r.group(1)) / 1e3


def breakdown(run, path, parts, total_s):
    """Seconds per part (summed over chunks) and the unexplained rest."""
    ph = pd.read_csv(os.path.join(run, "phases.csv"))
    ph = ph[ph.path == path].drop_duplicates("chunk_id")  # first (timed) read
    out = {label: (ph[col].fillna(0).sum() / 1e3 if col in ph else 0.0)
           for col, label, _ in parts}
    out[OTHER[0]] = max(0.0, total_s - sum(out.values()))
    return out


def gib(ds):
    """Dataset size in GiB (sum of its chunk files)."""
    d = os.path.join(RUNS, "..", ds, "fields")
    return sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d)) / 2**30


V2RUNS = [("learn", "learning only"), ("learnexp", "learning + exploration")]
H = 0.26  # bar height


def panel(ax, rows, side, parts):
    """Per dataset: the raw bar, then one stacked bar per v2 run, in seconds
    per GiB; each v2 bar names its run and its time against raw."""
    y = np.arange(len(rows))[::-1] * 1.15
    for yy, r in zip(y, rows):
        raw, size = r[f"raw_{side}"], r["gib"]
        ax.barh(yy + H * 1.05, raw / size, height=H, color=RAW)
        ax.text(raw / size, yy + H * 1.05, f"  uncompressed {raw:.2f} s",
                va="center", fontsize=7.2, color=INK2)
        for k, (mode, label) in enumerate(V2RUNS):
            yb = yy - H * 1.05 * k
            left = 0.0
            for part, colour in [(p[1], p[2]) for p in parts] + [OTHER]:
                w = r[f"{mode}_{side}_parts"][part] / size
                ax.barh(yb, w, left=left, height=H, color=colour)
                left += w
            v2 = r[f"{mode}_{side}"]
            f = v2 / raw
            cmp = f"{f:.1f}x slower" if f >= 1 else f"{1 / f:.1f}x faster"
            ax.text(left, yb, f"  {label} {v2:.2f} s ({cmp})", va="center",
                    fontsize=7.2, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['ds']}  ({r['gib']:.2f} GiB)" for r in rows],
                       fontsize=8.4, color=INK)
    ax.set_xlim(0, ax.get_xlim()[1] * 1.6)
    ax.set_xlabel("seconds per GiB of data (shorter = faster)", fontsize=9,
                  color=INK2)
    ax.grid(axis="x", color="#e6e9ed", lw=0.6)
    ax.tick_params(axis="x", labelsize=8, colors=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(FIGS, "v2_workloads_end_to_end.png"))
    ap.add_argument("--pdf", action="store_true",
                    help="also write a PDF next to the PNG")
    a = ap.parse_args()
    rows = []
    for ds in DATASETS:
        raw_w, raw_r = wall(os.path.join(RUNS, f"{ds}_raw"))
        row = {"ds": ds, "gib": gib(ds), "raw_write": raw_w, "raw_read": raw_r}
        for mode, _ in V2RUNS:
            run = os.path.join(RUNS, f"{ds}_{mode}")
            w, r = wall(run)
            row.update({f"{mode}_write": w, f"{mode}_read": r,
                        f"{mode}_write_parts": breakdown(run, "write", WRITE_PARTS, w),
                        f"{mode}_read_parts": breakdown(run, "read", READ_PARTS, r)})
        rows.append(row)
    rows.sort(key=lambda r: r["learnexp_write"] / r["raw_write"])
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, (aw, ar) = plt.subplots(1, 2, figsize=(18, 15), sharey=True)
    fig.patch.set_facecolor("white")
    panel(aw, rows, "write", WRITE_PARTS)
    panel(ar, rows, "read", READ_PARTS)
    aw.set_title("Writing the dataset through Clio", loc="left", fontsize=11.5,
                 color=INK)
    ar.set_title("Reading it back", loc="left", fontsize=11.5, color=INK)
    names = [("stored uncompressed (no compression)", RAW)]
    names += [(p[1], p[2]) for p in WRITE_PARTS] + [("decompress (read)", READ_PARTS[0][2]), OTHER]
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=c) for _, c in names],
               [n if i else n for i, (n, _) in enumerate(names)], loc="upper left",
               bbox_to_anchor=(0.01, 0.905), ncol=3, frameon=False, fontsize=9,
               labelcolor=INK, title="Per dataset: grey bar = uncompressed; then "
               "NeuroPress v2 with learning only, and with learning + exploration "
               "(each named beside its bar), split into where the time went:",
               title_fontsize=9)
    tot = {k: sum(r[k] for r in rows) for k in rows[0] if k.endswith(("_write", "_read"))}
    ratio = {m: (tot[f"{m}_write"] / tot["raw_write"], tot[f"{m}_read"] / tot["raw_read"])
             for m, _ in V2RUNS}
    fig.suptitle("Measured end-to-end time through Clio: storing with NeuroPress v2 "
                 "vs storing uncompressed", x=0.01, ha="left", fontsize=14,
                 color=INK, y=0.995)
    fig.text(0.01, 0.955, "How to read: each bar is the measured wall-clock time to "
             "write (left) or read (right) the whole dataset through Clio on this "
             "machine, per GiB of data; the seconds are printed beside it.\nAll "
             "writes and reads go to one NVMe SSD (the 4 storage tiers of the cost "
             "charts are NOT applied here). Every NeuroPress v2 step is inside the "
             "coloured bar: prediction, type conversion, exploration and learning "
             "included.\n"
             "All 19 datasets together, against uncompressed: learning only "
             f"{ratio['learn'][0]:.1f}x the write time and {ratio['learn'][1]:.1f}x "
             f"the read time; learning + exploration {ratio['learnexp'][0]:.1f}x and "
             f"{ratio['learnexp'][1]:.1f}x.", fontsize=9.2, color=INK2,
             va="top", linespacing=1.45)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.815, bottom=0.04, wspace=0.06)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=150)
    if a.pdf:
        fig.savefig(os.path.splitext(a.out)[0] + ".pdf")
    print("wrote", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
