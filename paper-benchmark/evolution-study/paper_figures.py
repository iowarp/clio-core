#!/usr/bin/env python3
"""The paper's heterogeneity figures, from a campaign's own measurement files.

    ./paper_figures.py fig3 --fields DUMPS --blocks EV/blocks.csv --out fig3
    ./paper_figures.py fig4 --blobs RUN/blobs.csv --out fig4

These were previously drawn outside this repository, which meant a regenerated
campaign could not reproduce them. Each subcommand reads only the standard
outputs -- `evolution.py`'s blocks.csv, a run's blobs.csv -- so a new campaign
redraws them without editing anything.

    fig3   Activity is spatially localized, and the blocks record it.
           Top: density at 0/25/50/75/100% of the run. Bottom: the same run's
           eight 1 MiB blocks -- contiguous z-slabs -- at the same five points,
           border colours matching. The share of cells unchanged since the
           previous dump starts flat near 100% and develops into a well that
           deepens AND widens as the front propagates.

           Needs a 128^3 run: a float32 field is then 8 MiB, which is exactly
           eight 1 MiB blocks. At 256^3 there are 64 and the x axis stops
           meaning "z-slab you can see in the panel above".

    fig5   The same measurement across workloads: per-block change E, and the
           share of cells bit-identical to the previous dump, both against
           NORMALIZED run time so runs of different length overlay. Takes one
           or more LABEL=evolution.json; a single workload is a valid figure,
           it just cannot make the paper's stationary-vs-not comparison.

    fields Per-field compression across the run: how each variable's ratio
           moves as the simulation evolves, and how far apart the fields are
           from each other at any one moment. The companion to fig4 -- that one
           asks how chunks of ONE field differ, this asks how the FIELDS differ.

    fig4   Compression behaviour varies within a single dump.
           (a) every chunk of one field's dump, ordered by position, with the
           spread annotated. (b) the same max/min spread for every dump in the
           run, one line per field -- the point being that (a) is not a
           single-dump accident.
"""
import argparse
import collections
import re
import csv
import gzip
import json
import importlib.util as iu
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter, LogLocator

_here = os.path.dirname(os.path.abspath(__file__))
_spec = iu.spec_from_file_location(
    "ev", os.path.join(_here, os.pardir, "evolution.py"))
ev = iu.module_from_spec(_spec)
_spec.loader.exec_module(ev)

# One colour per sampled point in the run, cool -> warm as the blast grows.
STAGE = ["#2166AC", "#67A9CF", "#B8B8B8", "#EF8A62", "#B2182B"]
FIELD_COLOR = {"density": "#0072B2", "xmom": "#D55E00", "ymom": "#009E73",
               "zmom": "#CC79A7", "rho_E": "#E69F00", "rho_e": "#56B4E9"}
# Nyx has six named fields; VPIC has sixteen and WarpX ten. Anything not named
# above draws from here in sorted order, so a 16-field plate is still readable
# instead of collapsing to one grey.
EXTRA = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9",
         "#8E7BB5", "#5EA47A", "#C25A6E", "#7E93A3", "#B0894F", "#2A6F8E",
         "#A65C2E", "#4C8C6A", "#96588A", "#6B7D3A"]


def colour_for(order):
    """-> {field: colour}, keeping the named ones and cycling for the rest."""
    out, k = {}, 0
    for f in order:
        if f in FIELD_COLOR:
            out[f] = FIELD_COLOR[f]
        else:
            out[f] = EXTRA[k % len(EXTRA)]
            k += 1
    return out
ACCENT = "#D55E00"

RC = {
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.4,
    "xtick.labelsize": 7.4, "ytick.labelsize": 7.4,
    "axes.linewidth": .7, "xtick.major.width": .7, "ytick.major.width": .7,
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#DDDDDD", "grid.linewidth": .5,
    "axes.axisbelow": True, "figure.dpi": 200,
    "savefig.bbox": "tight", "savefig.pad_inches": .015, "pdf.fonttype": 42,
}


def opener(path):
    return gzip.open if path.endswith(".gz") else open


def block_label(blocks_path, default=1048576):
    """Human block size for fig3's x axis, read from the run's own summary.

    blocks.csv carries a block INDEX and not its size, so the axis label used
    to assert 1 MiB -- true for the 128^3 study runs and wrong for any other
    --block. evolution.py writes block_bytes into the summary it drops beside
    the csv, so read it from there:

        <dir>/blocks.csv      -> <dir>/evolution.json   (evolution.py's own)
        <run>.blocks.csv[.gz] -> <run>.json             (evolution-study's)

    @param blocks_path path to the blocks.csv the figure is drawn from
    @param default bytes to assume when no summary sits beside it
    @return the size as a short string, e.g. "1 MiB" or "8 MiB"
    """
    cand = [os.path.join(os.path.dirname(blocks_path), "evolution.json"),
            re.sub(r"\.blocks\.csv(\.gz)?$", ".json", blocks_path)]
    nbytes = default
    for c in cand:
        try:
            with open(c) as f:
                nbytes = int(json.load(f).get("block_bytes", default))
            break
        except (OSError, ValueError, KeyError):
            continue
    for unit, div in (("MiB", 1 << 20), ("KiB", 1 << 10)):
        if nbytes >= div and nbytes % div == 0:
            return f"{nbytes // div} {unit}"
    return f"{nbytes} B"


def save(fig, stem):
    out = stem if stem.endswith(".png") else stem + ".png"
    fig.savefig(out)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# fig3
# ---------------------------------------------------------------------------
def fig3(a):
    rows = list(csv.DictReader(opener(a.blocks)(a.blocks, "rt")))
    rows = [r for r in rows if r["field"] == a.field]
    if not rows:
        sys.exit(f"no {a.field} rows in {a.blocks}")
    nblocks = max(int(r["block"]) for r in rows) + 1
    if nblocks != 8:
        print(f"NOTE: {nblocks} blocks per field, not 8. fig3 reads blocks as "
              f"contiguous z-slabs of the panels above, which only lines up on "
              f"a 128^3 run at --block 1048576.", file=sys.stderr)

    # % cells unchanged, by block, at each sampled pair
    per_pair = collections.defaultdict(dict)
    for r in rows:
        per_pair[int(r["step_to"])][int(r["block"])] = float(r["pct_cells_same"])
    pairs = sorted(per_pair)
    picks = [pairs[0]] + [pairs[int(round(f * (len(pairs) - 1)))]
                          for f in (.25, .5, .75, 1.0)]

    # THE TOP ROW NEEDS ONLY FIVE MID-PLANE SLICES, not the dumps. A 128^3 run
    # is 4.8 GB and a VPIC one 26 GB, none of which can be committed -- but the
    # five 2-D slices the panels actually draw are ~320 KB together. --slices
    # reads them from that cache when the dumps are gone, and --save-slices
    # writes it while they are still there.
    if a.slices and os.path.exists(a.slices):
        z = np.load(a.slices)
        vols = [z[f"s{k}"] for k in range(5)]
        print(f"  top row from {a.slices} (dumps not needed)")
    else:
        # WarpX writes openPMD, not .f32; its reader takes a DATASET PATH
        # (/data/<step>/fields/E/z) and yields the field back as "Ez", so the
        # name has to be translated or it matches nothing and returns no frames.
        want = [a.field]
        if a.source == "openpmd":
            m = re.fullmatch(r"([EBj])([xyz])", a.field)
            want = [f"{m[1]}/{m[2]}"] if m else [a.field]
        frames = list(ev.SOURCES[a.source](a.fields, np.float32, want)) if a.fields else []
        if len(frames) < 5:
            sys.exit(f"need at least 5 frames in --fields, or a --slices cache; "
                     f"found {len(frames)}")
        idx = [int(round(f * (len(frames) - 1))) for f in (0, .25, .5, .75, 1.0)]
        vols = None

    plt.rcParams.update(RC)
    fig, axes = plt.subplots(2, 5, figsize=(a.width, 3.5),
                             gridspec_kw=dict(height_ratios=[1.35, 1], hspace=.30,
                                              wspace=.12, left=.06, right=.99,
                                              top=.90, bottom=.13))

    shape = tuple(int(x) for x in a.shape.split(",")) if a.shape else None
    if vols is None:
        vols = []
        for i in idx:
            flat = frames[i][1][a.field].astype(np.float64)
            # A CUBE IS AN ASSUMPTION, NOT A FACT. WarpX's 64x64x512 grid holds
            # 2,097,152 cells, which is exactly 128^3, so the cube-root test
            # PASSES and reshapes a slab into a cube: the geometry is then
            # nonsense and the picture tiles. --shape is mandatory for any grid
            # that is not cubic.
            if shape:
                if int(np.prod(shape)) != len(flat):
                    sys.exit(f"--shape {shape} has {int(np.prod(shape))} cells, "
                             f"the field has {len(flat)}")
                v = flat.reshape(*shape, order="F")
            else:
                n = round(len(flat) ** (1 / 3))
                if n ** 3 != len(flat):
                    sys.exit(f"{len(flat)} elements is not a cube; pass --shape")
                # order="F": these dumps are Fortran-ordered and a C reshape
                # silently transposes the picture rather than failing.
                v = flat.reshape(n, n, n, order="F")
            vols.append(v[:, :, v.shape[2] // 2])       # the mid-plane, all we draw
        if a.save_slices:
            np.savez_compressed(a.save_slices,
                                **{f"s{k}": v.astype(np.float32)
                                   for k, v in enumerate(vols)})
            print(f"  wrote {a.save_slices} "
                  f"({os.path.getsize(a.save_slices) / 1024:.0f} KB) "
                  f"-- fig3 can be redrawn from this without the dumps")

    allv = np.concatenate([v.ravel() for v in vols])
    pos = allv[allv > 0]
    norm = LogNorm(vmin=np.percentile(pos, 1), vmax=np.percentile(pos, 99.9))

    for k, (ax, v) in enumerate(zip(axes[0], vols)):
        ax.imshow(v.T, origin="lower", norm=norm, cmap="inferno")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set(visible=True, color=STAGE[k], linewidth=2.0)
        ax.set_title(f"{k * 25}% of the run", fontsize=7.6, color=STAGE[k], pad=3)

    # bottom: one curve per stage, sharing the border colours
    gs = axes[1, 0].get_gridspec()
    for ax in axes[1]:
        ax.remove()
    big = fig.add_subplot(gs[1, :])
    # Stages are drawn thickest-and-largest first so that where curves COINCIDE
    # the earlier ones still show as a ring around the later. On a stationary
    # workload all five are numerically identical and a single-width line would
    # render as one curve, hiding four.
    MARK = ["o", "s", "^", "D", "v"]
    for k, p in enumerate(picks):
        d = per_pair[p]
        xs = sorted(d)
        big.plot(xs, [d[b] for b in xs], "-", marker=MARK[k], color=STAGE[k],
                 lw=2.8 - .45 * k, ms=9.5 - 1.5 * k, mfc="none", mew=1.4,
                 label=f"{k * 25}%", zorder=3 + k)
    big.set_xlabel(f"block index  (contiguous $z$-slab, {nblocks} "
                   f"$\\times$ {block_label(a.blocks)})")
    big.set_ylabel("% cells unchanged\nsince previous dump")
    vals = [v for p in picks for v in per_pair[p].values()]
    top = max(vals)
    big.set_ylim(0, 104 if top > 40 else top * 1.55)
    big.set_xlim(-0.3, nblocks - 0.7)
    big.set_xticks(range(nblocks))
    big.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
    big.legend(loc="lower center", ncol=5, frameon=False, fontsize=7.2,
               handlelength=1.4, columnspacing=1.4, borderpad=.1)

    last = per_pair[picks[-1]]
    outer = (last[0] + last[nblocks - 1]) / 2
    mid = (last[nblocks // 2] + last[nblocks // 2 - 1]) / 2
    # The claim is derived, not asserted: on a workload whose activity fills the
    # domain the outer and central slabs agree and the figure must say so.
    # mid can be 0, and 1.5 > 0 makes any noise look like localization. Require
    # a real gap AND enough signal to be talking about.
    if outer > mid * 1.5 and outer - mid > 5.0:
        head = (f"Activity is spatially localized: by the end the outermost "
                f"$z$-slabs still retain\n{outer:.0f}% of their cells between "
                f"dumps while the central slabs retain {mid:.0f}%")
    else:
        head = (f"Activity is NOT spatially localized here: every $z$-slab "
                f"retains about the same\nshare of its cells between dumps "
                f"({mid:.1f}-{outer:.1f}%) — the whole domain is active")
    fig.suptitle(head, fontsize=8.6, y=1.005, linespacing=1.35)
    save(fig, a.out)
    print(f"\n  outermost {outer:.1f}%   central {mid:.1f}%   "
          f"({nblocks} blocks, {len(pairs)} pairs)")


# ---------------------------------------------------------------------------
# fig4
# ---------------------------------------------------------------------------
def parse_blob(blob, step_scale):
    p = blob.split("/")
    chunk = int(p[-1].split("_")[1])
    if p[0].startswith("plt") or p[0].startswith("step"):
        field = p[1].split("_", 2)[2] if "_comp" in p[1] else p[1]
        return field, int("".join(c for c in p[0] if c.isdigit())) * step_scale, chunk
    return p[0], int(p[1].split("_")[1]), chunk


def fig4(a):
    rows = [r for r in csv.DictReader(opener(a.blobs)(a.blobs, "rt"))
            if r.get("rc", "0") == "0"]
    if not rows:
        sys.exit(f"no successful rows in {a.blobs}")
    for r in rows:
        r["f"], r["s"], r["c"] = parse_blob(r["blob"], a.step_scale)
        r["ratio"] = int(r["bytes"]) / int(r["stored"])

    dumps = collections.defaultdict(list)
    for r in rows:
        dumps[(r["f"], r["s"])].append(r)
    spreads = {k: max(x["ratio"] for x in v) / min(x["ratio"] for x in v)
               for k, v in dumps.items()}

    # (a) defaults to the dump with the widest spread -- the claim the panel
    # makes is about how far apart chunks of ONE dump can be.
    if a.step is None:
        key = max((k for k in dumps if k[0] == a.field), key=lambda k: spreads[k])
    else:
        key = (a.field, a.step)
        if key not in dumps:
            sys.exit(f"no {a.field} dump at step {a.step}")
    dump = sorted(dumps[key], key=lambda r: r["c"])

    plt.rcParams.update(RC)
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(a.width, 2.55),
        gridspec_kw=dict(wspace=.28, left=.08, right=.985, bottom=.19, top=.88))

    codecs = sorted({r["codec"] for r in dump})
    cpal = {c: FIELD_COLOR[f] for c, f in
            zip(codecs, ["density", "xmom", "ymom", "zmom", "rho_E", "rho_e"])}
    for r in dump:
        ax_a.bar(r["c"], r["ratio"], width=.74, color=cpal[r["codec"]],
                 edgecolor="white", linewidth=.4, zorder=3)
    lo = min(r["ratio"] for r in dump)
    hi = max(r["ratio"] for r in dump)
    ax_a.set_yscale("log")
    ax_a.set_ylim(max(1, lo * .55), hi * 3.2)
    ax_a.yaxis.set_major_locator(LogLocator(base=10))
    ax_a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}$\\times$"))
    ax_a.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax_a.set_xticks(range(0, len(dump), max(1, len(dump) // 8)))
    ax_a.set_xlabel("chunk index within the dump")
    ax_a.set_ylabel("compression ratio")
    ax_a.text(len(dump) * .5, hi * 1.9, f"{hi / lo:.0f}$\\times$ spread\n"
              f"within one snapshot", ha="center", va="center", fontsize=7.4,
              color=ACCENT, linespacing=1.2)
    ax_a.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=cpal[c], ec="white", lw=.4)
                         for c in codecs],
                labels=[c.replace("nvcomp-", "") for c in codecs],
                loc="upper left", frameon=False, fontsize=6.8, handlelength=1.1,
                handleheight=.9, borderpad=.1, labelspacing=.22)
    # Two lines: at half the figure width a one-line title of this length
    # overruns into the neighbouring panel's title.
    ax_a.set_title(f"(a)  Within one dump: the {len(dump)} chunks of {key[0]}\n"
                   f"at step {key[1]} differ by {hi / lo:.0f}$\\times$",
                   loc="left", fontsize=8.4, pad=4)

    fields_b = sorted({k[0] for k in dumps})
    cmap_b = colour_for(fields_b)
    for f in fields_b:
        pts = sorted((k[1], spreads[k]) for k in dumps if k[0] == f)
        ax_b.plot([p[0] for p in pts], [p[1] for p in pts], lw=1.05,
                  color=cmap_b[f], label=f, zorder=3)
    ax_b.set_yscale("log")
    ax_b.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}$\\times$"))
    ax_b.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax_b.axhline(1, color="#999999", lw=.7, zorder=1)
    ax_b.set_xlabel("timestep")
    ax_b.set_ylabel("within-dump ratio spread\n(max / min across chunks)")
    ax_b.legend(loc="upper center", bbox_to_anchor=(.5, -.28),
                ncol=2 if len(fields_b) <= 8 else 3, frameon=False, fontsize=6.2,
                handlelength=1.1, columnspacing=.8, labelspacing=.2)
    ax_b.set_title("(b)  Not a one-dump accident: the same spread,\n"
                   "every field, every dump", loc="left", fontsize=8.4, pad=4)

    save(fig, a.out)
    sp = sorted(spreads.values())
    med = sp[len(sp) // 2]
    multi = sum(1 for v in dumps.values() if len({x["codec"] for x in v}) > 1)
    print(f"\n  panel (a): {hi / lo:.0f}x spread, {hi:.1f}x down to {lo:.2f}x")
    print(f"  median within-dump spread {med:.0f}x over {len(dumps)} dumps")
    print(f"  dumps assigned more than one codec: {multi} of {len(dumps)} "
          f"({100 * multi / len(dumps):.0f}%)")


def fig5(a):
    runs = []
    for spec in a.run:
        if "=" not in spec:
            sys.exit(f"--run wants LABEL=evolution.json, got {spec!r}")
        label, path = spec.split("=", 1)
        d = json.load(open(path))
        runs.append((label, d))

    plt.rcParams.update(RC)
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(a.width, 2.55),
        gridspec_kw=dict(wspace=.26, left=.085, right=.985, bottom=.19, top=.88))

    pal = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
    for k, (label, d) in enumerate(runs):
        for key, ax in (("interval_means", ax_a),
                        ("interval_pct_cells_same", ax_b)):
            pts = d[key]
            # Normalized run time, so a 1,000-step run and a 2,000-step one lie
            # on the same axis: the question is how the data behaves ACROSS a
            # run, not at any particular timestep.
            last = pts[-1][1] or 1
            ax.plot([r[1] / last for r in pts], [r[2] for r in pts],
                    color=pal[k % len(pal)], lw=1.15, label=label, zorder=3)
            ax.plot(pts[-1][1] / last, pts[-1][2], "o", ms=3.2,
                    color=pal[k % len(pal)], mec="white", mew=.6, zorder=4)

    ax_a.set_yscale("log")
    ax_a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax_a.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax_a.set_xlabel("normalized run time")
    ax_a.set_ylabel("per-block change $E$\n(0 = unchanged)")
    ax_a.set_title("(a)  Does the data keep changing?  Per-block change $E$\n"
                   "between consecutive dumps", loc="left", fontsize=8.4, pad=5)
    ax_a.legend(loc="best", frameon=False, fontsize=7, handlelength=1.4,
                labelspacing=.28)

    ax_b.set_ylim(-3, 104)
    ax_b.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
    ax_b.set_xlabel("normalized run time")
    ax_b.set_ylabel("% cells bit-identical\nto previous dump")
    ax_b.set_title("(b)  Is the previous dump still usable?  Share of cells\n"
                   "bit-identical to it", loc="left", fontsize=8.4, pad=5)

    for ax in (ax_a, ax_b):
        ax.set_xlim(0, 1)
    save(fig, a.out)

    print()
    for label, d in runs:
        c = d["interval_pct_cells_same"]
        e = d["interval_means"]
        print(f"  {label:<10} E {e[0][2]:.3f} -> {e[-1][2]:.3f}   "
              f"cells identical {c[0][2]:.1f}% -> {c[-1][2]:.1f}%   "
              f"mean E {d['mean']:.3f}")
    if len(runs) < 4:
        print(f"\n  NOTE: {len(runs)} workload(s). The paper's figure contrasts "
              f"non-stationary (Nyx, WarpX) against stationary (LAMMPS, VPIC); "
              f"that comparison needs all four.")


def fields_fig(a):
    rows = [r for r in csv.DictReader(opener(a.blobs)(a.blobs, "rt"))
            if r.get("rc", "0") == "0"]
    for r in rows:
        r["f"], r["s"], r["c"] = parse_blob(r["blob"], a.step_scale)
        r["ratio"] = int(r["bytes"]) / int(r["stored"])

    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        per[r["f"]][r["s"]].append(r["ratio"])
    order = sorted(per)
    cmap = colour_for(order)

    plt.rcParams.update(RC)
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(a.width, 2.55),
        gridspec_kw=dict(wspace=.27, left=.085, right=.985, bottom=.19, top=.88))

    # (a) each field's median ratio through the run
    for f in order:
        pts = sorted((s, sorted(v)[len(v) // 2]) for s, v in per[f].items())
        ax_a.plot([p[0] for p in pts], [p[1] for p in pts], lw=1.1,
                  color=cmap[f], label=f, zorder=3)
    ax_a.set_yscale("log")
    ax_a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}$\\times$"))
    ax_a.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax_a.set_xlabel("timestep")
    ax_a.set_ylabel("median compression ratio")
    ax_a.set_title(f"(a)  Each of the {len(order)} fields, through the run",
                   loc="left", fontsize=8.4, pad=5)
    ncol = 2 if len(order) <= 8 else 3
    ax_a.legend(loc="upper center", bbox_to_anchor=(.5, -.28), ncol=ncol,
                frameon=False, fontsize=6.2, handlelength=1.1,
                columnspacing=.8, labelspacing=.2)

    # (b) how far apart the fields are at each moment -- the per-field analogue
    # of fig4's within-dump spread
    steps = sorted({s for f in per for s in per[f]})
    spread = []
    for s in steps:
        med = [sorted(per[f][s])[len(per[f][s]) // 2] for f in order if s in per[f]]
        if len(med) > 1:
            spread.append((s, max(med) / min(med)))
    ax_b.plot([p[0] for p in spread], [p[1] for p in spread], lw=1.2,
              color=ACCENT, zorder=3)
    ax_b.axhline(1, color="#999999", lw=.7, zorder=1)
    ax_b.set_yscale("log")
    ax_b.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}$\\times$"))
    ax_b.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax_b.set_xlabel("timestep")
    ax_b.set_ylabel("spread ACROSS fields\n(max / min of the field medians)")
    ax_b.set_title("(b)  ...but not at the same rate: how far apart the fields are",
                   loc="left", fontsize=8.4, pad=5)
    save(fig, a.out)

    print("\n  %-10s %-9s %-9s %-9s" % ("field", "first", "last", "median"))
    for f in order:
        pts = sorted((s, sorted(v)[len(v) // 2]) for s, v in per[f].items())
        allr = sorted(x for v in per[f].values() for x in v)
        print("  %-10s %-9.2f %-9.2f %-9.2f" % (
            f, pts[0][1], pts[-1][1], allr[len(allr) // 2]))
    if spread:
        sp = sorted(p[1] for p in spread)
        print(f"\n  across-field spread: median {sp[len(sp)//2]:.2f}x, "
              f"max {sp[-1]:.2f}x over {len(spread)} dumps")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("fig3", help="activity is spatially localized")
    q.add_argument("--fields", help="the dump directory (or use --slices)")
    q.add_argument("--source", default="f32", choices=sorted(ev.SOURCES),
                   help="dump format for the top row (openpmd for WarpX)")
    q.add_argument("--shape", default="",
                   help="NX,NY,NZ for a non-cubic grid, e.g. 64,64,512")
    q.add_argument("--slices", help="an .npz of the five mid-plane slices, "
                                    "so the top row needs no dumps")
    q.add_argument("--save-slices", help="write that .npz while the dumps exist")
    q.add_argument("--blocks", required=True, help="evolution.py's blocks.csv")
    q.add_argument("--field", default="density")
    q.add_argument("--out", required=True, help="output stem")
    q.add_argument("--width", type=float, default=7.0)
    q.set_defaults(fn=fig3)

    q = sub.add_parser("fig4", help="compression varies within a single dump")
    q.add_argument("--blobs", required=True, help="a run's blobs.csv")
    q.add_argument("--field", default="density")
    q.add_argument("--step", type=int, help="default: the widest-spread dump")
    q.add_argument("--step-scale", type=int, default=1)
    q.add_argument("--out", required=True)
    q.add_argument("--width", type=float, default=7.0)
    q.set_defaults(fn=fig4)

    q = sub.add_parser("fig5", help="the same measurement across workloads")
    q.add_argument("--run", action="append", required=True,
                   metavar="LABEL=evolution.json", help="repeatable")
    q.add_argument("--out", required=True)
    q.add_argument("--width", type=float, default=7.0)
    q.set_defaults(fn=fig5)

    q = sub.add_parser("fields", help="per-field ratio across the run")
    q.add_argument("--blobs", required=True)
    q.add_argument("--step-scale", type=int, default=1)
    q.add_argument("--out", required=True)
    q.add_argument("--width", type=float, default=7.0)
    q.set_defaults(fn=fields_fig)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
