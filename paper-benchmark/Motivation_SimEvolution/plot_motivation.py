#!/usr/bin/env python3
"""The motivation plate, and the numbers behind it.

    ./plot_motivation.py --dir FARM --blobs blobs.csv.gz \\
        --explore explore.csv.gz --out plate.png
    ./plot_motivation.py --blobs blobs.csv.gz --explore explore.csv.gz --stats

One field at three points of the run, every panel ruled into the chunks it was
stored as, every slab labelled with the maximum compression ratio it can
reach. --stats prints the spreads over ALL the slabs of ALL the dumps, not
just the three drawn, which is what the README quotes.

THE DATA AND ITS COMPRESSIBILITY ARE ONE PLATE, deliberately. Split into two
figures, the reader has to carry sixteen numbers from one to the other, and
the only thing worth seeing -- the ceiling collapsing exactly where the shell
has arrived, slab by slab and dump by dump -- is what gets lost.

WITH --explore EVERY NUMBER IS THE MAXIMUM ACHIEVABLE RATIO over all 32
actions a K=31 run measured on that chunk, not the one the run adopted. That
maximum is a property of the data and the codecs; what a run adopts also
depends on its cost model and on the ratio cap (see run.sh).

--axis x OR y, NEVER z. A chunk is a contiguous byte range of a
Fortran-ordered field, so it is a z-slab: on a z mid-plane slice every chunk
boundary lies outside the picture and there is nothing to rule.

A CHUNK IS A z-SLAB ONLY BECAUSE THE DUMPS ARE FORTRAN-ORDERED (x fastest).
Read C-ordered the same bytes interleave the whole volume, and the picture
would be a lie rather than an error; the slab bounds are computed from each
chunk's byte offset rather than assumed to divide the grid evenly.

THE RATIO IS bytes/stored, NOT the csv's own `ratio` column, which differs by
the blob header -- small but real. evolution-study/paper_figures.py's fig4 uses
bytes/stored, and two figures of one run disagreeing in the third digit is
not worth the convenience.

FRAME READING, SLICING, THE SHARED COLOUR SCALE, THE BLANK-PLATE REFUSAL AND
THE PANELS ARE ../plot/figure_evolution.py's, not a second implementation, so
this plate and the evolution figures come from the same bytes and the same
rendering. What is this figure's own is here: the csv readers, the slabs, and
the composition -- which frames, the overlay on top of them, and no title. In
the paper that line is the caption, and a plate that prints its own headline
above it reads as a mistake; the panels are titled with their timestep alone
for the same reason.
"""
import argparse
import collections
import csv
import gzip
import importlib.util as iu
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_here = os.path.dirname(os.path.abspath(__file__))
FRAME_RE = re.compile(r"^(?:plt|step_?)(\d+)$")


def evolution_plotter():
    """../plot/figure_evolution.py as a module, loaded on first use.

    The frames, the slicing and the panels come from there. It is loaded by
    path because paper-benchmark is not a package, and lazily because --stats
    draws nothing and should not need matplotlib's field readers at all.

    @return the module
    """
    spec = iu.spec_from_file_location(
        "fe", os.path.join(_here, os.pardir, "plot", "figure_evolution.py"))
    mod = iu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def opener(path):
    """@return the open function for a plain or gzipped csv."""
    return gzip.open if path.endswith(".gz") else open


def parse_blob(blob):
    """Split a blob name into the dump, the field and the chunk index.

    Two shapes reach here, and the dump component is kept VERBATIM because it
    is also the directory the bytes are read back from:

        plt00020/fab0000_comp00_density/chunk_3   Nyx, VPIC
        step00010/E_x/chunk_0                     WarpX

    @param blob the blobs.csv `blob` column
    @return (dump, field, chunk) or None when the name is neither shape
    """
    p = blob.split("/")
    if len(p) != 3 or not FRAME_RE.match(p[0]):
        return None
    field = p[1].split("_", 2)[2] if "_comp" in p[1] else p[1]
    return p[0], field, int(p[2].split("_")[1])


def read_blobs(path, field):
    """One field's chunks, grouped by dump.

    @param path a run's blobs.csv (.gz accepted)
    @param field the field to keep, e.g. "density"
    @return {dump: {chunk: {"ratio", "codec", "bytes"}}}
    """
    out = collections.defaultdict(dict)
    with opener(path)(path, "rt") as f:
        for r in csv.DictReader(f):
            got = parse_blob(r["blob"])
            if not got or got[1] != field:
                continue
            dump, _, chunk = got
            stored = int(r["stored"]) or 1
            out[dump][chunk] = {"ratio": int(r["bytes"]) / stored,
                                "codec": r["codec"], "bytes": int(r["bytes"])}
    return out


def read_explore(path, field):
    """Every measured action of every chunk, from a K=31 run's explore.csv.

    The action key is the one perchunk_oracle_tables.py uses --
    `lib|q<quantize>|s<shuffle>` -- so the two analyses name the same thing
    the same way. A library measured at several presets collapses to its best
    here, which is what "what could this codec do on this slab" means.

    @param path the run's explore.csv
    @param field the field to keep
    @return ({dump: {chunk: {action: (ratio, codec)}}}, {(dump, chunk): bytes})
    """
    per = collections.defaultdict(lambda: collections.defaultdict(dict))
    size = {}
    with opener(path)(path, "rt") as f:
        for r in csv.DictReader(f):
            got = parse_blob(r["blob"])
            if not got or got[1] != field:
                continue
            try:
                ratio = float(r["ratio"])
            except (TypeError, ValueError):
                continue
            if not ratio > 0:
                continue
            dump, _, chunk = got
            act = (f"{r['lib_name'].replace('nvcomp-', '')}"
                   f"|q{r['quantize']}|s{r['shuffle']}")
            prev = per[dump][chunk].get(act)
            if prev is None or ratio > prev[0]:
                per[dump][chunk][act] = (ratio, r["lib_name"])
            size[(dump, chunk)] = int(r["chunk_bytes"])
    return per, size


def best_fixed_action(sweep, size):
    """The one action a static policy would choose for the whole run.

    Scored the way perchunk_oracle_tables.py scores a fixed baseline: total
    stored bytes over the chunks EVERY action was measured on. An action
    missing from some chunks cannot be compared on the others without
    flattering it, so the intersection is the population.

    @param sweep read_explore's first return
    @param size read_explore's second return
    @return (action, {(dump, chunk): ratio}) or (None, {})
    """
    keys = [(d, c) for d in sweep for c in sweep[d]]
    if not keys:
        return None, {}
    common = set(sweep[keys[0][0]][keys[0][1]])
    for d, c in keys[1:]:
        common &= set(sweep[d][c])
    if not common:
        return None, {}
    best, best_bytes = None, None
    for act in sorted(common):
        tot = sum(size[(d, c)] / sweep[d][c][act][0] for d, c in keys)
        if best_bytes is None or tot < best_bytes:
            best, best_bytes = act, tot
    return best, {k: sweep[k[0]][k[1]][best][0] for k in keys}


def slab_of(chunk, chunk_bytes, itemsize, nx, ny, nz):
    """The z range a chunk covers, from its byte offset rather than by assuming.

    A chunk need not divide the grid: the last one is short whenever the field
    is not a whole multiple of the chunk size, and a slab boundary then falls
    inside a plane. Returning floats keeps that visible instead of rounding it
    away.

    @param chunk chunk index within the field
    @param chunk_bytes bytes per chunk
    @param itemsize bytes per element
    @param nx, ny, nz grid dimensions
    @return (z0, z1) in cells, clipped to the grid
    """
    per = chunk_bytes / itemsize
    plane = nx * ny
    return (min(chunk * per / plane, nz), min((chunk + 1) * per / plane, nz))


def ratio_text(r):
    """@return a ratio as the figure prints it: 357x, 21x, 3.5x."""
    return f"{r:.0f}$\\times$" if r >= 10 else f"{r:.1f}$\\times$"


def chunk_table(blobs, sweep):
    """The chunks one field was stored as, keyed by the timestep they are from.

    For a figure that draws the DATA and wants its chunks on top of it: the
    key is the dump directory's own number, which for a farm named by timestep
    IS the step a frame reader reports for that frame, so the two line up with
    no mapping table.

    @param blobs read_blobs' output
    @param sweep read_explore's first return, or {} to label each slab with
           the ratio the run adopted rather than the MAXIMUM ACHIEVABLE one
    @return {step: {chunk: {"ratio", "bytes"}}}
    """
    out = {}
    for dump, chunks in blobs.items():
        rec = {}
        for c, r in chunks.items():
            best = r["ratio"]
            if sweep.get(dump, {}).get(c):
                best = max(x for x, _ in sweep[dump][c].values())
            rec[c] = {"ratio": best, "bytes": r["bytes"]}
        out[int(FRAME_RE.match(dump)[1])] = rec
    return out


def overlay(ax, chunks, shape, itemsize):
    """Rule a panel into its chunks, and label each one with its ratio.

    The panels are ../plot/figure_evolution.py's; this is what makes the data
    and its compressibility one plate rather than two.

    imshow's default extent puts cell centres on the integers, so the boundary
    between cell z-1 and cell z is at z-0.5; drawing it at z would cut every
    slab one cell late.

    @param ax the panel
    @param chunks {chunk: {"ratio", "bytes"}} for this timestep, or None
    @param shape the volume's (nx, ny, nz)
    @param itemsize bytes per element AS WRITTEN -- a frame cast to float64
           for plotting would halve every slab
    """
    if not chunks:
        return
    nx, ny, nz = shape
    for c in sorted(chunks):
        z0, z1 = slab_of(c, chunks[c]["bytes"], itemsize, nx, ny, nz)
        if z0 > 0:
            ax.axhline(z0 - .5, color="#222222", lw=.7)
        ax.text(ny * .985, (z0 + z1) / 2 - .5,
                f"c{c}  {ratio_text(chunks[c]['ratio'])}",
                ha="right", va="center", fontsize=7.4, color="#111111",
                bbox=dict(facecolor="white", alpha=.75, edgecolor="none",
                          boxstyle="square,pad=.14"))

def spread_matrix(blobs, sweep):
    """Every slab of every dump as one (slab x dump) matrix of ratios.

    The three drawn panels sample the run; this is all of it, which is what
    makes "the ceiling moves" a measurement rather than an anecdote.

    @param blobs read_blobs' output, for the adopted fallback
    @param sweep read_explore's first return, or {} for the adopted ratio
    @return the matrix, rows chunks and columns dumps in timestep order, NaN
            where a chunk was not stored
    """
    dumps = sorted(blobs, key=lambda d: int(FRAME_RE.match(d)[1]))
    nchunk = max(max(blobs[d]) for d in dumps) + 1
    m = np.full((nchunk, len(dumps)), np.nan)
    for j, d in enumerate(dumps):
        for c, rec in blobs[d].items():
            best = rec["ratio"]
            if sweep.get(d, {}).get(c):
                best = max(r for r, _ in sweep[d][c].values())
            m[c, j] = best
    return m


def print_stats(blobs, sweep, size):
    """The numbers the figure's caption and the README quote.

    Two spreads -- down a column (the slabs of one dump) and along a row (one
    slab through the run) -- the ceiling's range, and what the best SINGLE
    FIXED action would have stored against choosing per chunk. The last is the
    honest check on the figure: at a bound where one action wins everywhere it
    is ~1.00x, and the case for adapting is the ceiling, not the codec.

    @param blobs read_blobs' output
    @param sweep read_explore's first return, or {}
    @param size read_explore's second return, or {}
    """
    m = spread_matrix(blobs, sweep)
    col = np.nanmax(m, axis=0) / np.nanmin(m, axis=0)
    row = np.nanmax(m, axis=1) / np.nanmin(m, axis=1)
    print(f"  spread within one dump, across slabs     median "
          f"{np.nanmedian(col):7.1f}x  max {np.nanmax(col):7.1f}x")
    print(f"  spread within one slab, through the run  median "
          f"{np.nanmedian(row):7.1f}x  max {np.nanmax(row):7.1f}x")
    print(f"  ceiling range                            "
          f"{np.nanmin(m):7.1f}x .. {np.nanmax(m):.1f}x over "
          f"{int(np.isfinite(m).sum())} chunks")
    if not sweep:
        return
    fixed, fixed_ratio = best_fixed_action(sweep, size)
    if not fixed:
        return
    keys = [(d, c) for d in sweep for c in sweep[d]]
    adaptive = sum(size[k] / max(r for r, _ in sweep[k[0]][k[1]].values())
                   for k in keys)
    static = sum(size[k] / fixed_ratio[k] for k in keys)
    agree = sum(1 for d, c in keys
                if max(sweep[d][c], key=lambda k: sweep[d][c][k][0]) == fixed)
    print(f"  best fixed action: {fixed}   per-chunk best stores "
          f"{static / adaptive:.2f}x fewer bytes over {len(keys)} chunks, and "
          f"that action is already the per-chunk best on "
          f"{100 * agree / len(keys):.0f}% of them")


def frames_at(fe, dump_dir, field, dtype, at):
    """The frames to draw, and the timestep of each.

    @param fe the evolution plotter module, whose readers these are
    @param dump_dir a farm of .f32 dumps, NAMED BY TIMESTEP so that the step
           in each panel's title is the real one
    @param field the field to read
    @param dtype the dumps' element type
    @param at fractions of the run to draw
    @return (steps, volumes as float64)
    """
    frames = list(fe.ev.SOURCES["f32"](dump_dir, dtype, [field]))
    if len(frames) < 2:
        sys.exit(f"{dump_dir}: need at least 2 frames, found {len(frames)}")
    steps, arrays = [], []
    for i in fe.pick_at(len(frames), at):
        step, fields = frames[i]
        if field not in fields:
            sys.exit(f"field {field!r} not in {sorted(fields)}")
        steps.append(step)
        arrays.append(fields[field].astype(np.float64))
    return steps, arrays


def draw(a, slabs):
    """The plate: three dumps, one colour scale, the chunks written on them.

    @param a the parsed arguments
    @param slabs chunk_table's output, {step: {chunk: {"ratio", "bytes"}}}
    """
    fe = evolution_plotter()
    steps, arrays = frames_at(fe, a.dir, a.field,
                              np.float64 if a.f64 else np.float32,
                              a.at or [0.05, 0.5, 1.0])
    # The csv is keyed by the dump's own number, which for a farm named by
    # timestep is the step the frame reader just reported. When the two do not
    # line up, the csvs and the dumps are from different runs, and a plate with
    # no slabs on it would hide that rather than say it.
    if not any(st in slabs for st in steps):
        sys.exit(f"{a.blobs} has chunks for steps {sorted(slabs)[:4]}... but "
                 f"these dumps are at {steps}: --blobs expects the csvs of "
                 f"the run that replayed THESE dumps")
    got = [fe.slice_of(v, a.axis) for v in arrays]
    if any(g is None for g in got):
        sys.exit(f"{a.dir}: {len(arrays[0])} elements is not a cube")
    slices = [g[0] for g in got]
    vols = [g[1] for g in got]
    fe.refuse_blank(a.field, a.axis, slices, vols)
    norm, cmap = fe.shared_norm(slices)

    fig, axes = plt.subplots(1, len(steps),
                             figsize=(4.5 * len(steps), 4.8), squeeze=False)
    axes = axes[0]
    itemsize = 8 if a.f64 else 4
    fe.draw_panels(
        fig, axes, slices, vols, [""] * len(steps), steps, norm, cmap,
        overlay=lambda ax, vol, st: overlay(ax, slabs.get(st), vol.shape,
                                            itemsize))
    if a.title:
        fig.suptitle(f"{a.field} — {a.axis} mid-plane, one shared colour "
                     f"scale — each slab is one stored chunk, labelled with "
                     f"its {'maximum achievable' if a.explore else 'adopted'}"
                     f" compression ratio", y=0.99)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=130, bbox_inches="tight")
    print(f"{a.out}  ({a.field}, steps "
          f"{'/'.join(str(st) for st in steps)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blobs", required=True,
                    help="a run's blobs.csv(.gz): one row per stored chunk")
    ap.add_argument("--out", default="",
                    help="where to write the plate; omit it with --stats to "
                         "print the numbers and draw nothing")
    ap.add_argument("--dir", default="",
                    help="the dump farm that run replayed, named by timestep. "
                         "Required to draw")
    ap.add_argument("--explore", default="",
                    help="the same run's explore.csv(.gz). With it every "
                         "number is the MAXIMUM ACHIEVABLE ratio over every "
                         "measured action rather than the adopted one")
    ap.add_argument("--field", default="density")
    ap.add_argument("--axis", default="x", choices=["x", "y"],
                    help="the plane to slice. Not z: a chunk IS a z-slab, so "
                         "a z mid-plane slice lies inside a single chunk and "
                         "there would be no boundary in the picture")
    ap.add_argument("--at", action="append", type=float, default=[],
                    help="fractions of the run to draw, repeatable "
                         "[0.05 0.5 1]; the frame nearest each is used. Not "
                         "0: a Sedov blast at step 0 is a point source in an "
                         "otherwise uniform box, and the panel reads as empty")
    ap.add_argument("--f64", action="store_true",
                    help="the dumps are float64. The slab arithmetic needs "
                         "the size ON DISK, not the float64 it is plotted as")
    ap.add_argument("--stats", action="store_true",
                    help="print the spreads over every slab of every dump, "
                         "the ceiling's range, and what one fixed action "
                         "would have stored")
    ap.add_argument("--title", action="store_true",
                    help="print a title over the panels. Off by default: in "
                         "the paper that text is the caption")
    a = ap.parse_args()
    if not a.out and not a.stats:
        sys.exit("nothing to do: --out draws the plate, --stats prints the "
                 "numbers, and both is fine")
    if a.out and not a.dir:
        sys.exit("--out needs --dir: the plate is the DATA with its chunks "
                 "drawn on top, and the dumps are where the data is")

    # Each csv is parsed once, whether one figure or both numbers want it.
    blobs = read_blobs(a.blobs, a.field)
    if not blobs:
        sys.exit(f"no {a.field} chunks in {a.blobs}")
    sweep, size = read_explore(a.explore, a.field) if a.explore else ({}, {})
    if a.out:
        draw(a, chunk_table(blobs, sweep))
    if a.stats:
        print_stats(blobs, sweep, size)


if __name__ == "__main__":
    main()
