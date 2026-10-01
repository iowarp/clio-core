#!/usr/bin/env python3
"""Per-chunk codec selections of XGBoost, HCompress and NeuroPress, and where
they diverge.

For each workload the three selector arms of figure 9b (XGB+Tier,
HCompress+Tier, NeuroPress+Tier = NP+Tier+Async+Lossy) wrote the SAME chunks,
so their choices can be compared chunk by chunk. One panel per workload:

  - a strip per selector, one column per chunk in write order, coloured by the
    codec it ran;
  - under each, a thin strip marking the chunks it PREPROCESSED (quantize
    and/or byte shuffle -- blobs.csv records preprocessing time, not which);
  - an agreement strip: all three chose the same action, two did, or none.

"Action" is codec + preprocessed, so bitcomp and quantized bitcomp differ.
The agreement strip is a grey ramp, so the figure's point survives greyscale
printing (EuroSys requires it); codec identity is carried by each row's direct
label (its most frequent choice and share) and the legend, and the exact
shares are written to a CSV beside the data.

Usage:
    plot_selection_divergence.py [--data data] [--out live/choices]
Writes <out>/fig9_selection_divergence.png and
<data>/selection_divergence_summary.csv.
"""
import argparse
import collections
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from plot_choices import CODEC_COLORS, SERIF  # noqa: E402

WORKLOADS = [("vpic", "VPIC"), ("nyx", "Nyx"), ("lammps", "LAMMPS"),
             ("warpx", "WarpX")]
# Figure 9b's order: the two baselines, then NeuroPress.
SELECTORS = [("xgb_tier", "XGBoost"), ("hcompress_tier", "HCompress"),
             ("np_tier_async_lossy", "NeuroPress")]
PREP_ON, PREP_OFF = "#2b2a27", "#e9e8e3"
AGREE = {3: ("#e3e2de", "all three agree"), 2: ("#9c9b97", "two agree"),
         1: ("#3d3c39", "all three differ")}
FIG_W = 7.0          # EuroSys text block; place at \textwidth, unscaled
FONT = 10


def short(codec):
    """'nvcomp-bitcomp' -> 'bitcomp'; raw stays readable."""
    if codec.startswith("raw"):
        return "raw"
    return codec[len("nvcomp-"):] if codec.startswith("nvcomp-") else codec


def load_arm(data, wl, tag):
    """One arm's chunks in write order: [(blob, codec, preprocessed)]."""
    path = os.path.join(data, wl, tag, tag, "r1", tag, "blobs.csv")
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            prep = float(r["preproc_ms"] or 0) > 0
            out.append((r["blob"], r["codec"], prep))
    return out


def strip(values, color_of):
    """A 1 x N RGB image from per-chunk values."""
    return np.array([[to_rgb(color_of(v)) for v in values]])


def main():
    """Load the three arms per workload, draw the strips, write the summary."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="live/choices")
    a = ap.parse_args()
    plt.rcParams.update({"font.family": "serif", "font.serif": SERIF,
                         "font.size": FONT})

    panels, summary, used = [], [], collections.Counter()
    for wl, wname in WORKLOADS:
        arms = {tag: load_arm(a.data, wl, tag) for tag, _ in SELECTORS}
        order = [b for b, _, _ in arms["np_tier_async_lossy"]]
        idx = {tag: {b: (c, p) for b, c, p in rows} for tag, rows in arms.items()}
        missing = [t for t in idx if set(idx[t]) != set(order)]
        if missing:
            raise SystemExit(f"{wname}: chunk sets differ for {missing}")
        acts = {t: [idx[t][b] for b in order] for t in idx}
        agree = []
        for i in range(len(order)):
            keys = [acts[t][i] for t, _ in SELECTORS]
            agree.append(max(collections.Counter(keys).values()))
        pair = lambda x, y: 100 * np.mean(  # noqa: E731
            [acts[x][i] == acts[y][i] for i in range(len(order))])
        agree_c = collections.Counter(agree)
        n = len(order)
        for t, name in SELECTORS:
            cnt = collections.Counter(
                f"{short(c)}{'+prep' if p else ''}" for c, p in acts[t])
            for c, _ in acts[t]:
                used[c] += 1
            summary.append({
                "workload": wname, "selector": name, "chunks": n,
                "preprocessed_pct": round(100 * np.mean([p for _, p in acts[t]]), 1),
                "top_choices": "; ".join(f"{k} {100*v/n:.1f}%"
                                         for k, v in cnt.most_common(4))})
        summary.append({
            "workload": wname, "selector": "(agreement)", "chunks": n,
            "preprocessed_pct": "",
            "top_choices": (f"all three {100*agree_c[3]/n:.1f}%; two "
                            f"{100*agree_c[2]/n:.1f}%; none {100*agree_c[1]/n:.1f}%; "
                            f"XGB=HC {pair('xgb_tier','hcompress_tier'):.1f}%; "
                            f"XGB=NP {pair('xgb_tier','np_tier_async_lossy'):.1f}%; "
                            f"HC=NP {pair('hcompress_tier','np_tier_async_lossy'):.1f}%")})
        panels.append((wname, acts, agree, agree_c, n))

    # Rows per panel: (codec, prep) x 3 selectors, then agreement.
    row_h = [1.0, 0.32] * 3 + [1.0]
    fig = plt.figure(figsize=(FIG_W, 1.55 * len(panels) + 1.25))
    top, bot = 0.985, 1.2 / (1.55 * len(panels) + 1.25)
    gap = 0.018
    ph = (top - bot - gap * (len(panels) - 1)) / len(panels)
    left, right = 0.16, 0.73
    for k, (wname, acts, agree, agree_c, n) in enumerate(panels):
        y1 = top - k * (ph + gap)
        ax = fig.add_axes([left, y1 - ph, right - left, ph])
        ax.set_xlim(0, n)
        ytop, labels = sum(row_h), []
        y = ytop
        rows = []
        for t, name in SELECTORS:
            rows.append((strip([c for c, _ in acts[t]],
                               lambda c: CODEC_COLORS.get(c, "#d6d5d1")),
                         name, "codec", t))
            rows.append((strip([p for _, p in acts[t]],
                               lambda p: PREP_ON if p else PREP_OFF),
                         "", "prep", t))
        rows.append((strip(agree, lambda v: AGREE[v][0]), "agreement",
                     "agree", None))
        for (img, name, kind, t), h in zip(rows, row_h):
            ax.imshow(img, extent=[0, n, y - h, y], aspect="auto",
                      interpolation="antialiased", interpolation_stage="rgba",
                      rasterized=True)
            mid = y - h / 2
            if kind == "codec":
                ax.text(-0.012 * n, mid, name, ha="right", va="center")
                c = collections.Counter(
                    f"{short(cc)}{'+prep' if pp else ''}" for cc, pp in acts[t])
                top1, v = c.most_common(1)[0]
                ax.text(1.012 * n, mid, f"{top1} {100*v/n:.0f}%",
                        ha="left", va="center")
            elif kind == "agree":
                ax.text(-0.012 * n, mid, name, ha="right", va="center")
                ax.text(1.012 * n, mid,
                        f"all {100*agree_c[3]/n:.0f}%, none {100*agree_c[1]/n:.0f}%",
                        ha="left", va="center")
            y -= h
        ax.set_ylim(0, ytop)
        ax.set_yticks([])
        # Each panel spans ITS OWN chunks, so the count goes with the name,
        # not on a shared axis.
        ax.set_xticks([])
        if k == len(panels) - 1:
            ax.set_xlabel("chunk, in write order", labelpad=3)
        for s in ax.spines.values():
            s.set_visible(False)
        fig.text(0.012, y1 - ph / 2, f"{wname}\n{n:,} chunks", rotation=90,
                 ha="left", va="center", ma="center")

    codecs = [c for c, _ in used.most_common() if c in CODEC_COLORS]
    handles = ([Patch(facecolor=CODEC_COLORS[c], label=short(c)) for c in codecs] +
               [Patch(facecolor=PREP_ON, label="preprocessed (quantize/shuffle)")] +
               [Patch(facecolor=AGREE[v][0], edgecolor="#8c8b87", linewidth=0.5,
                      label=AGREE[v][1]) for v in (3, 2, 1)])
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=FONT, handlelength=1.2, columnspacing=1.0,
               bbox_to_anchor=(0.5, 0.0))
    os.makedirs(a.out, exist_ok=True)
    png = os.path.join(a.out, "fig9_selection_divergence.png")
    fig.savefig(png, dpi=400)
    csv_path = os.path.join(a.data, "selection_divergence_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    for r in summary:
        print(f"{r['workload']:7s} {r['selector']:12s} prep {str(r['preprocessed_pct']):>5s}%  "
              f"{r['top_choices']}")
    print(f"wrote {png} and {csv_path}")


if __name__ == "__main__":
    main()
