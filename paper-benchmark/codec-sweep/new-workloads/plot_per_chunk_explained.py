#!/usr/bin/env python3
"""What "best single codec" and "per-chunk optimal" mean, chunk by chunk.

    plot_per_chunk_explained.py [--datasets a,b,c] [--out PNG]

For each dataset, on the same four-tier hierarchy and balanced cost as
hierarchy_balanced.png (cost of a chunk = compress ms + decompress ms +
compressed bytes / its tier's bandwidth):
  strip 1  BEST SINGLE CODEC: the one setting that is cheapest when used for
           every chunk -- so every chunk has the same colour
  strip 2  PER-CHUNK OPTIMAL: each chunk takes whichever of the 139 settings is
           cheapest for that chunk -- colour = the codec it picked (hatched =
           with a pre-shuffle)
  bars     each chunk's cost under both policies; the gap is what per-chunk
           selection saves on that chunk
Chunks are grouped by array type (labelled under the bars), in run order
within each type. The box on the right gives both totals with ratio and the
compress / decompress / I/O split.
"""
import argparse
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plot_new_workloads as pw  # noqa: E402
cpc, ch, cm = pw.cpc, pw.ch, pw.cm

COLORS = dict(cpc.CODEC_COLOR)  # reference palette: one fixed hue per codec
COLORS.update({"gpulz": "#7b3fa0", "store": cpc.STORE_COLOR})
OTHER = "#c3c2b7"
BEST_EDGE = "#0b0b0b"


def codec_of(cfg):
    a = cfg.split()[0]
    return f"nvcomp-{a}" if a in pw.NVCOMP else a


def chunk_table(key):
    """Per-chunk costs of every setting for one dataset, with product and tier."""
    d = os.path.join(os.path.expanduser("~/np-newsweep"), key)
    df, _, _ = pw.load_dataset(d)
    mp = pd.read_csv(os.path.join(d, "map.csv"), header=None, names=["k", "product"])
    mp[["file", "chunk"]] = mp["k"].str.rsplit("#", n=1, expand=True)
    mp["chunk"] = mp["chunk"].astype(int)
    df = df.merge(mp[["file", "chunk", "product"]], on=["file", "chunk"])
    # totals weight each sampled chunk by the real chunks it stands for
    src = (os.path.expanduser(f"~/np-data/{key[4:]}/fields") if key.startswith("ref-")
           else os.path.expanduser(f"~/np-data/new/{key}"))
    wt = pw.vw.chunk_weights(d, src)[["file", "chunk", "weight"]]
    df = df.merge(wt, on=["file", "chunk"])
    tier = ch.place(df)
    bw = tier.map(pw.BW_HIER)
    df = df.assign(tier=tier, io_ms=df["comp_bytes"] / (bw * 1e6))
    df["cost"] = df["comp_ms"] + df["decomp_ms"] + df["io_ms"]
    best = (df["cost"] * df["weight"]).groupby(df["config"]).sum().idxmin()
    single = df[df["config"] == best].set_index(["file", "chunk"])
    opt = (df.sort_values(["file", "chunk", "cost"]).groupby(["file", "chunk"]).head(1)
           .set_index(["file", "chunk"]))
    t = single[["product", "tier", "weight", "bytes", "cost", "comp_bytes", "comp_ms", "decomp_ms", "io_ms"]].join(
        opt[["config", "cost", "comp_bytes", "comp_ms", "decomp_ms", "io_ms"]], rsuffix="_opt")
    t = t.reset_index().sort_values(["product", "file", "chunk"]).reset_index(drop=True)
    return t, best


def totals(t, suf):
    """Volume-weighted totals: each sampled chunk counts as the real chunks it
    stands for."""
    w = t["weight"]
    gib = (t["bytes"] * w).sum() / 2**30
    return ((t["bytes"] * w).sum() / (t[f"comp_bytes{suf}"] * w).sum(),
            (t[f"comp_ms{suf}"] * w).sum() / gib, (t[f"decomp_ms{suf}"] * w).sum() / gib,
            (t[f"io_ms{suf}"] * w).sum() / gib, (t[f"cost{suf}"] * w).sum() / gib)


def strip(ax, colors, hatches, label):
    n = len(colors)
    for i, (c, h) in enumerate(zip(colors, hatches)):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, facecolor=c, edgecolor=cm.SURFACE,
                                   linewidth=0.6, hatch=h))
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(-0.01, 0.5, label, transform=ax.transAxes, ha="right", va="center",
            fontsize=9, color=cm.INK)


def draw(fig, gs, key, name):
    t, best = chunk_table(key)
    n = len(t)
    sub = gs.subgridspec(3, 2, height_ratios=[0.5, 0.5, 3], width_ratios=[4.0, 1.2],
                         hspace=0.12, wspace=0.04)
    a1, a2, a3 = (fig.add_subplot(sub[i, 0]) for i in range(3))
    box = fig.add_subplot(sub[:, 1])
    bcol = COLORS.get(codec_of(best), OTHER)
    strip(a1, [bcol] * n, [""] * n, f"Best single codec\n{pw.short_config(best)} (all chunks)")
    ocol = [COLORS.get(codec_of(c), OTHER) for c in t["config"]]
    ohat = ["////" if "shuffle=" in c else "" for c in t["config"]]
    strip(a2, ocol, ohat, "Per-chunk optimal\n(each chunk its own)")
    x = np.arange(n)
    a3.bar(x + 0.5, t["cost"], width=0.86, color="#9a9890", label="cost with the best single codec")
    a3.bar(x + 0.5, t["cost_opt"], width=0.46, color=cpc.ORACLE, label="cost with the per-chunk optimal")
    a3.set_xlim(0, n)
    cm.style_axes(a3)
    a3.set_ylabel("cost of the chunk (ms)", color=cm.INK2, fontsize=8.5)
    a3.tick_params(axis="x", length=0)
    # array-type groups under the bars
    edges = [0] + list(np.flatnonzero(t["product"].values[1:] != t["product"].values[:-1]) + 1) + [n]
    a3.set_xticks([(a + b) / 2 for a, b in zip(edges[:-1], edges[1:])])
    labels, narrow = [], 0
    for a, b in zip(edges[:-1], edges[1:]):
        lab = t["product"].iloc[a]
        m = re.match(r"(.+)_(col_idx|row_ptr|values)(\..+)$", lab)
        if m:  # "<matrix>_<array>.<type>" -> two lines
            lab = f"{m.group(1)}\n{m.group(2)}{m.group(3)}"
        if (b - a) / n < 0.06:  # narrow group: stagger every other label lower
            lab = ("\n\n" if narrow % 2 else "") + lab
            narrow += 1
        labels.append(lab)
    a3.set_xticklabels(labels, fontsize=7.5, color=cm.INK2)
    for e in edges[1:-1]:
        for ax in (a1, a2, a3):
            ax.axvline(e, color=cm.INK, linewidth=1.0)
    a1.set_title(f"{name}: {n} sampled 4 MiB chunks, grouped by array type", loc="left",
                 fontsize=10.5, color=cm.INK, pad=6)
    # totals
    rs, cs, ds, ios, tots = totals(t, "")
    ro, co, do, ioo, toto = totals(t, "_opt")
    box.axis("off")
    def pct(new, old):
        return f"{100 * (new / old - 1):+.0f}%"
    lines = [("", "best single", "per-chunk", "change"),
             ("ratio", f"{rs:.2f}x", f"{ro:.2f}x", pct(ro, rs)),
             ("compress", f"{cs:.0f}", f"{co:.0f}", pct(co, cs)),
             ("decompress", f"{ds:.0f}", f"{do:.0f}", pct(do, ds)),
             ("I/O", f"{ios:.0f}", f"{ioo:.0f}", pct(ioo, ios)),
             ("total", f"{tots:.0f}", f"{toto:.0f}", pct(toto, tots))]
    y = 0.92
    box.text(0.0, y + 0.06, "totals, ms per GiB of input", fontsize=8.5, color=cm.INK2,
             transform=box.transAxes)
    for k, (lab, a_, b_, c_) in enumerate(lines):
        w = "bold" if lab in ("", "total") else "normal"
        yy = y - k * 0.1
        box.text(0.0, yy, lab, fontsize=9, color=cm.INK, transform=box.transAxes, weight=w)
        box.text(0.50, yy, a_, fontsize=9, color=cm.INK, transform=box.transAxes, ha="right", weight=w)
        box.text(0.76, yy, b_, fontsize=9, color=cpc.ORACLE if k else cm.INK,
                 transform=box.transAxes, ha="right", weight=w)
        box.text(1.0, yy, c_, fontsize=9, color=cm.INK, transform=box.transAxes, ha="right", weight=w)
    box.text(0.0, y - 6.4 * 0.1, f"per-chunk saves {100 * (1 - toto / tots):.1f}%\n"
             f"chooses the best single codec's\nsetting on {100 * (t['config'] == best).mean():.0f}% of chunks",
             fontsize=9, color=cm.INK, transform=box.transAxes, va="top")
    return {codec_of(c) for c in list(t["config"]) + [best]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="consumer-vpic-full,genomics-reads,sparse-fem")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "..", "figures", "new-workloads",
                                                  "per_chunk_explained.png"))
    a = ap.parse_args()
    keys = a.datasets.split(",")
    fig = plt.figure(figsize=(17, 4.6 * len(keys) + 1.6))
    fig.patch.set_facecolor(cm.SURFACE)
    gs = fig.add_gridspec(len(keys), 1, left=0.145, right=0.985, top=1 - 1.45 / (4.6 * len(keys) + 1.6),
                          bottom=0.04, hspace=0.42)
    used = set()
    for i, k in enumerate(keys):
        used |= draw(fig, gs[i], k, pw.DATASETS.get(k, k))
    order = [c for c in COLORS if c in used] + (["other"] if any(c not in COLORS for c in used) else [])
    handles = [Patch(color=COLORS.get(c, OTHER), label=(cm.CODECS.get(c, c) if c != "other" else "LZ4 / Snappy / GDeflate"))
               for c in order]
    handles.append(Patch(facecolor="white", edgecolor=cm.INK2, hatch="////", label="with pre-shuffle"))
    handles += [Patch(color="#9a9890", label="bar: chunk cost, best single codec"),
                Patch(color=cpc.ORACLE, label="bar: chunk cost, per-chunk optimal")]
    fig.legend(handles=handles, loc="upper left", ncol=6, frameon=False, fontsize=9,
               bbox_to_anchor=(0.005, 1 - 0.62 / (4.6 * len(keys) + 1.6)), labelcolor=cm.INK2)
    fig.suptitle("Best single codec vs per-chunk optimal, chunk by chunk (balanced cost, chunks dealt "
                 "10/30/30/30 over DRAM 12 / NVMe 1 / SSD 0.5 / HDD 0.25 GB/s)",
                 x=0.01, ha="left", fontsize=12.5, color=cm.INK, y=1 - 0.12 / (4.6 * len(keys) + 1.6))
    fig.text(0.01, 1 - 0.45 / (4.6 * len(keys) + 1.6),
             "Best single codec: ONE setting for every chunk, the one with the lowest total. "
             "Per-chunk optimal: every chunk gets the cheapest of all 139 settings for that chunk. "
             "Strip colour = codec used; bars = each chunk's cost under each policy.",
             fontsize=9, color=cm.INK2)
    fig.savefig(a.out, dpi=150, facecolor=cm.SURFACE)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
