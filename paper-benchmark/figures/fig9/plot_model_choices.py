#!/usr/bin/env python3
"""Per-chunk codec picks of the three lossless selectors in figure 9 (b).

    plot_model_choices.py [--data DIR] [--out PNG]

One strip per selector (XGB, HCompress, NeuroPress) per workload, one column
per chunk in write order, coloured by the codec the run stored that chunk
with. Beside each strip: the run's overall ratio, the share of chunks it
byte-shuffled, and -- for the two baselines -- the share whose pick was the
same ACTION as NeuroPress's (codec AND shuffle; a codec match with a different
shuffle stores different bytes, so it is not counted as the same pick).

Reads data/<wl>/<tag>/<tag>/r1/<tag>/blobs.csv, the run each figure 9 bar was
imported from; blobs with preproc_ms > 0 were shuffled.
"""
import argparse
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from plot_choices import CODEC_COLORS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WORKLOADS = [("vpic", "VPIC"), ("nyx", "Nyx"), ("lammps", "LAMMPS"), ("warpx", "WarpX")]
SELECTORS = [("XGBoost", "xgb_lossless"), ("HCompress", "hcompress_lossless"),
             ("NeuroPress", "np_only")]
SHORT = {"raw(not-beneficial)": "stored raw"}
FS = 10                      # IEEE 10 pt body; the figure is placed at 7 in
INK, MUTED = "#2b2b2b", "#6b6a66"
SERIF = ["Times New Roman", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"]


def load(data, wl, tag):
    """Blobs of one arm's imported run, in write order.

    @param data figure 9 data directory.
    @param wl   workload directory name.
    @param tag  arm directory name.
    @return list of (blob, codec, shuffled, bytes, stored), or None if absent.
    """
    f = glob.glob(f"{data}/{wl}/{tag}/{tag}/r1/{tag}/blobs.csv")
    if not f:
        return None
    return [(r["blob"], r["codec"], float(r["preproc_ms"] or 0) > 0,
             int(r["bytes"]), int(r["stored"])) for r in csv.DictReader(open(f[0]))]


def draw(data, out):
    """Draw the strips and their side columns, and save the PNG.

    @param data figure 9 data directory.
    @param out  output PNG path.
    """
    runs = {(w, s): load(data, w, t) for w, _ in WORKLOADS for s, t in SELECTORS}
    codecs = sorted({c for r in runs.values() if r for _, c, *_ in r},
                    key=lambda c: list(CODEC_COLORS).index(c) if c in CODEC_COLORS else 99)
    idx = {c: i for i, c in enumerate(codecs)}
    cmap = ListedColormap([CODEC_COLORS.get(c, "#d6d5d1") for c in codecs])

    plt.rcParams.update({"font.family": "serif", "font.serif": SERIF, "font.size": FS})
    # Vertical layout in inches (legend + column heads, strips, chunk-axis note),
    # so the height fits the content and the width stays the 7 in text block.
    head_in, strip_in, gap_s_in, gap_w_in, foot_in = 0.80, 0.215, 0.04, 0.26, 0.30
    height = head_in + len(WORKLOADS) * (len(SELECTORS) * strip_in + gap_w_in) + foot_in
    fig = plt.figure(figsize=(7.0, height))
    left, right = 0.115, 0.60
    top, strip = 1 - head_in / height, strip_in / height
    gap_s, gap_w = gap_s_in / height, gap_w_in / height
    y = top
    for w, name in WORKLOADS:
        ref = runs[(w, "NeuroPress")]
        ref_act = {b: (c, s) for b, c, s, *_ in ref} if ref else {}
        fig.text(left, y + 0.03 / height, name, ha="left", va="bottom", fontsize=FS + 1,
                 fontweight="bold", color=INK)
        for s, _ in SELECTORS:
            y -= strip
            r = runs[(w, s)]
            ax = fig.add_axes([left, y, right - left, strip - gap_s])
            ax.set_xticks([]), ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            fig.text(left - 0.01, y + (strip - gap_s) / 2, s, ha="right", va="center",
                     fontsize=FS, color=INK)
            if not r:
                ax.text(0.5, 0.5, "no run", ha="center", va="center", fontsize=FS,
                        color=MUTED, transform=ax.transAxes)
                continue
            ax.imshow(np.array([[idx[c] for _, c, *_ in r]]), aspect="auto", cmap=cmap,
                      vmin=-0.5, vmax=len(codecs) - 0.5, interpolation="nearest")
            ratio = sum(x[3] for x in r) / sum(x[4] for x in r)
            shuf = sum(x[2] for x in r) / len(r)
            same = ("" if s == "NeuroPress" else
                    f"{sum(1 for b, c, sh, *_ in r if ref_act.get(b) == (c, sh)) / len(r):.0%}")
            for x, txt in zip((0.70, 0.82, 0.945), (f"{ratio:.2f}×", f"{shuf:.0%}", same)):
                fig.text(x, y + (strip - gap_s) / 2, txt, ha="center", va="center",
                         fontsize=FS, color=INK)
        y -= gap_w
    # Column heads over the side columns, and the chunk axis under the strips.
    for x, txt in zip((0.70, 0.82, 0.945), ("Ratio", "Shuffled", "Same as\nNeuroPress")):
        fig.text(x, top + 0.06 / height, txt, ha="center", va="bottom", fontsize=FS, color=INK,
                 linespacing=1.0)
    fig.text((left + right) / 2, y + gap_w - 0.12 / height, "Chunks in write order →",
             ha="center", va="top", fontsize=FS, color=MUTED)
    handles = [Patch(facecolor=CODEC_COLORS.get(c, "#d6d5d1"), edgecolor="none",
                     label=SHORT.get(c, c.replace("nvcomp-", ""))) for c in codecs]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=min(5, len(handles)), frameon=False, fontsize=FS, handlelength=1.2,
               columnspacing=1.2, handletextpad=0.5)
    fig.savefig(out, dpi=300, facecolor="white")
    print(out)


def main():
    """Parse arguments and draw the figure."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=f"{HERE}/data")
    ap.add_argument("--out", default=f"{HERE}/live/choices/fig9_model_choices.png")
    a = ap.parse_args()
    draw(a.data, a.out)


if __name__ == "__main__":
    main()
