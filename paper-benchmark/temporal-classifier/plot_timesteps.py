#!/usr/bin/env python3
"""Per-timestep view of one classify_temporal.py window: the data at T0..T3,
which block regions match the previous timestep, and a content-ID table.

    plot_timesteps.py --results DIR --window plt050-053 --root FIELDS \
        --out-dir FIGS [--field density]

Content IDs: within one field, every distinct block content (xxh3-64 hash) gets
an ID in order of first appearance (T0 index 0, 1, ... then T1 ...). A block
whose ID at Tt equals its own ID at Tt-1 matched the previous timestep at the
same index; `other_index_match` says the content already existed at a
DIFFERENT index (moved or repeated content). Written to content_ids.csv.

Figure rows:
  1. the field at T0..T3 on the mid-z slice, block grid drawn, one color scale
  2. |T(t) - T(t-1)| on that slice, its own color scale (the change itself;
     consecutive dumps differ by a few percent, invisible in row 1)
  3. per block of that slice, its state against the previous timestep
     (matched / near / changed-cold / changed-hot), labeled with block index
  4. every block of the field: state per timestep (x = block index)
"""
import argparse
import os
import re

import numpy as np
import pandas as pd

import classify_temporal as C

STATE_LABEL = ["Matched (byte-identical)", "Near match", "Changed, cold",
               "Changed, hot"]
FIRST = "#e1e0d9"   # T0 has no previous timestep


def content_ids(g):
    """Content ID per block and timestep, plus cross-index matches.

    @param g one field's rows of blocks.parquet
    @return DataFrame: block_index, coordinates, id_t0..id_t3, and for t>0
            match_prev_t (same index) and other_index_t (content seen at
            another index at any earlier point of the scan)
    """
    ids, owner = {}, {}
    out = g[["block_index", "coordinates"]].copy().reset_index(drop=True)
    for t in range(4):
        col, other = [], []
        for i, h in zip(out["block_index"], g[f"hash_t{t}"]):
            if h not in ids:
                ids[h] = len(ids)
                owner[h] = set()
            other.append(bool(owner[h] - {i}))
            owner[h].add(i)
            col.append(ids[h])
        out[f"id_t{t}"] = col
        if t:
            out[f"match_prev_t{t}"] = out[f"id_t{t}"] == out[f"id_t{t - 1}"]
            out[f"other_index_t{t}"] = other
    return out


def slice_rows(g):
    """The blocks of the mid-z slice, and the slice's block grid shape."""
    gz = g["bz"].max() + 1
    return g[g["bz"] == gz // 2], (g["by"].max() + 1, g["bx"].max() + 1)


def row_data(axs, source, field, bs):
    """Row 1: the field at T0..T3, mid-z slice, block grid, one color scale."""
    grids = [source.load(field, t) for t in range(4)]
    z = grids[0].shape[0] // 2
    lo = min(float(a[z].min()) for a in grids)
    hi = max(float(a[z].max()) for a in grids)
    for t, ax in enumerate(axs):
        im = ax.imshow(grids[t][z], cmap="Blues", vmin=lo, vmax=hi, origin="lower")
        for k in range(bs, grids[t].shape[1], bs):
            ax.axhline(k - 0.5, color="white", lw=0.4)
            ax.axvline(k - 0.5, color="white", lw=0.4)
        ax.set_title(f"T{t} = {source.frames[t]}", fontsize=9, color=C.INK)
        ax.set_xticks([]), ax.set_yticks([])
    return im, grids


def row_diff(axs, grids, bs):
    """Row 2: |T(t) - T(t-1)| on the mid-z slice, one scale for the three."""
    z = grids[0].shape[0] // 2
    diffs = [np.abs(grids[t][z].astype(np.float64) - grids[t - 1][z]) for t in (1, 2, 3)]
    hi = max(float(d.max()) for d in diffs) or 1.0
    axs[0].axis("off")
    axs[0].text(0.5, 0.5, "change vs the\nprevious timestep\n(same slice)", ha="center",
                va="center", fontsize=9, color=C.MUTED, transform=axs[0].transAxes)
    for t, (ax, d) in enumerate(zip(axs[1:], diffs), 1):
        im = ax.imshow(d, cmap="Oranges", vmin=0, vmax=hi, origin="lower")
        for k in range(bs, d.shape[0], bs):
            ax.axhline(k - 0.5, color=C.GRID, lw=0.4)
            ax.axvline(k - 0.5, color=C.GRID, lw=0.4)
        ax.set_title(f"|T{t} − T{t - 1}|: {100 * (d > 0).mean():.0f}% of cells changed",
                     fontsize=9, color=C.INK)
        ax.set_xticks([]), ax.set_yticks([])
    return im


def row_states(axs, g, cmap, plt):
    """Row 2: each slice block's state against the previous timestep."""
    s, shape = slice_rows(g)
    for t, ax in enumerate(axs):
        lab = np.full(shape, np.nan)
        if t:
            lab[s["by"], s["bx"]] = s[f"state_{C.CONSEC[t - 1]}"].to_numpy()
            ax.imshow(lab, cmap=cmap, vmin=-0.5, vmax=3.5, origin="lower",
                      interpolation="nearest")
            ax.set_title(f"T{t - 1}→T{t}", fontsize=9, color=C.INK)
        else:
            ax.imshow(np.zeros(shape), cmap=plt.matplotlib.colors.ListedColormap([FIRST]),
                      origin="lower")
            ax.set_title("T0: no previous step", fontsize=9, color=C.MUTED)
        for by, bx, bi in zip(s["by"], s["bx"], s["block_index"]):
            ax.text(bx, by, str(bi), ha="center", va="center", fontsize=4.5, color=C.INK)
        ax.set_xticks([]), ax.set_yticks([])


def row_matrix(ax, g, cmap):
    """Row 3: state per timestep for every block of the field."""
    n = len(g)
    m = np.full((4, n), np.nan)
    for t in range(1, 4):
        m[t] = g[f"state_{C.CONSEC[t - 1]}"].to_numpy()
    ax.imshow(np.where(np.isnan(m), -1, m), cmap=cmap, vmin=-1.5, vmax=3.5,
              aspect="auto", interpolation="nearest")
    ax.set_yticks(range(4), ["T0", "T1 vs T0", "T2 vs T1", "T3 vs T2"])
    ax.set_xlabel(f"Block index (0..{n - 1}; z-major, then y, then x)")
    for t in range(1, 4):
        k = int((g[f"state_{C.CONSEC[t - 1]}"] == C.EXACT).sum())
        ax.text(n + n * 0.01, t, f"{k}/{n} matched", va="center", fontsize=8,
                color=C.INK, clip_on=False)


def draw(g, source, field, bs, out, title):
    """Assemble the three rows and save the PNG."""
    from matplotlib.colors import ListedColormap
    plt = C._plt()
    cmap = ListedColormap(C.STATE_COLORS)
    cmap4 = ListedColormap([FIRST] + C.STATE_COLORS)
    fig = plt.figure(figsize=(11, 10.2))
    gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 1, 0.42], hspace=0.2, wspace=0.06,
                          top=0.92)
    ax_d = [fig.add_subplot(gs[0, t]) for t in range(4)]
    im, grids = row_data(ax_d, source, field, bs)
    fig.colorbar(im, ax=ax_d, fraction=0.015, pad=0.01,
                 label=field).ax.tick_params(labelsize=7)
    ax_c = [fig.add_subplot(gs[1, t]) for t in range(4)]
    imd = row_diff(ax_c, grids, bs)
    fig.colorbar(imd, ax=ax_c, fraction=0.015, pad=0.01,
                 label="|Δ|").ax.tick_params(labelsize=7)
    row_states([fig.add_subplot(gs[2, t]) for t in range(4)], g, cmap, plt)
    row_matrix(fig.add_subplot(gs[3, :]), g, cmap4)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in C.STATE_COLORS],
               labels=STATE_LABEL, frameon=False, ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, 0.975), fontsize=9)
    fig.suptitle(title, x=0.125, ha="left", y=0.99, fontsize=11, color=C.INK)
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--window", required=True, help="e.g. plt050-053")
    ap.add_argument("--root", required=True, help="fields directory")
    ap.add_argument("--field", nargs="*", default=["density"])
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    lo, hi = map(int, re.findall(r"\d+", a.window))
    frames = [f"plt{k:05d}" for k in range(lo, hi + 1)]
    d = pd.read_parquet(os.path.join(a.results, a.window, "blocks.parquet"))
    xyz = d["coordinates"].str.strip("()").str.split(",", expand=True).astype(int)
    d["bz"], d["by"], d["bx"] = xyz[0], xyz[1], xyz[2]
    for p in C.PAIRS:
        d[f"state_{p}"] = d[f"state_{p}"].map({s: i for i, s in enumerate(C.STATE_NAMES)})
    source = C.RawF32Source(a.root, frames)
    bs = source.shape[2] // (d["bx"].max() + 1)
    ids = pd.concat([content_ids(g).assign(field=f) for f, g in d.groupby("field", sort=False)])
    ids.to_csv(os.path.join(a.results, a.window, "content_ids.csv"), index=False)
    os.makedirs(a.out_dir, exist_ok=True)
    for f in a.field:
        g = d[d["field"] == f].reset_index(drop=True)
        out = os.path.join(a.out_dir, f"timesteps_{f}.png")
        draw(g, source, f, bs, out, f"Nyx {f}, dumps {lo}-{hi}: {bs}³ blocks, "
             "each timestep against the previous one at the same block index")
        print(out)


if __name__ == "__main__":
    main()
