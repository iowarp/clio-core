#!/usr/bin/env python3
"""Temporal block-evolution metric, shared by all four benchmark workloads.

    ./evolution.py --source f32     --dir /tmp/nyx-e10        --out /tmp/ev/nyx_e10
    ./evolution.py --source openpmd --dir /tmp/wx/run/diags   --out /tmp/ev/wx_mw0
    ./evolution.py --source raw     --dir /tmp/lmp-raw --f64  --out /tmp/ev/lmp_ramp

For each BLOCK -- a fixed byte range of a field, the unit Clio actually
compresses -- and each pair of consecutively sampled frames,

    E(B_t, B_t+dt) = ||B_t+dt - B_t||_2 / (||B_t+dt||_2 + ||B_t||_2 + eps)

E is 0 when the block is bit-identical to its predecessor, and approaches 1
when the two are unrelated (or opposite in sign). It is scale free, which is
what makes it comparable across four workloads whose fields span fifteen
orders of magnitude in SI units -- an absolute difference cannot be.

WHY A BLOCK AND NOT A WHOLE FIELD. The selector's decision is per chunk, so
the question the benchmark asks is whether the CHUNK's data keeps changing,
not whether the field as a whole does. A field can be busy while every chunk
of it is individually static (structure confined to one slab), and a whole-
field norm cannot see the difference. --block therefore defaults to 1 MiB,
which is WarpX's chunk size exactly and divides the other three workloads'
chunks evenly, so the same number means the same thing in all four.

WHAT IS REPORTED, and why more than a mean. A run that jumps once and then
freezes has the same mean as one that changes steadily, so the summary also
carries the per-interval series and two order statistics over it:

    p10_interval    10th percentile of the per-interval means. A single
                    spike leaves this near zero; sustained evolution does not.
    last_quarter    mean over the final quarter of the intervals. Catches the
                    run that only evolves while it is initialising.

`active` blocks are those with E >= --active-thresh (default 1e-3, i.e. a
0.1% normalised change). `nonzero` blocks are those that changed at all.

ALONGSIDE E, THE SHARE OF CELLS BIT-IDENTICAL to the previous sampled frame,
per block and over the run (`pct_cells_same`). It answers a different
question and the two disagree usefully: E says how far a block moved, this
says how much of it never moved. A shock front crossing an otherwise quiet
slab gives a large E with most cells untouched; a field nudged everywhere in
its last mantissa bit gives a tiny E with no cell untouched. The second
number is the one that explains compressibility -- its complement is what the
codec actually has to encode as new -- and it is what the Nyx and VPIC
READMEs already quote as "bit-identical to the previous dump", so the two can
be checked against each other.

NaN/Inf in any block is counted and reported rather than silently propagated:
a configuration that has gone numerically unstable must not be able to win on
a high evolution score.
"""
import argparse, glob, json, os, re, subprocess, sys, tempfile
import numpy as np

EPS = 1e-30


# ---------------------------------------------------------------------------
# Sources. Each yields (step, {field_name: 1-D numpy array}) in step order.
# ---------------------------------------------------------------------------
def frames_f32(d, dtype, want=None):
    """One directory per dump, one flat file per field. Two naming schemes:

        Nyx, VPIC:  <dir>/plt%05d/fab0000_comp%02d_<field>.f32
        WarpX:      <dir>/step%05d/<field>.f32

    Nyx and VPIC dump through the same code shape, so one reader serves both;
    the component index is part of the name but the FIELD NAME is what
    identifies a series across frames, because comp numbering is stable within
    a run and not worth depending on across them. WarpX's .f32 come from
    warpx_gen_fields.sh, which h5dumps the openPMD output and names each file
    after the dataset -- no fab/comp prefix to strip. Accepting both here is
    what lets the block metric run on a replay workload's own dumps rather
    than only on the openPMD it was extracted from.
    """
    dirs = sorted(glob.glob(os.path.join(d, "plt*"))) or \
        sorted(glob.glob(os.path.join(d, "step*")))
    for fdir in dirs:
        if not os.path.isdir(fdir):
            continue
        m = re.search(r"(?:plt|step)(\d+)", os.path.basename(fdir))
        if not m:
            continue
        step = int(m.group(1))
        fields = {}
        for p in sorted(glob.glob(os.path.join(fdir, "*.f32")) +
                        glob.glob(os.path.join(fdir, "*.f64"))):
            base = os.path.basename(p)
            mm = re.match(r"fab\d+_comp\d+_(.+)\.(f32|f64)$", base)
            name = mm.group(1) if mm else os.path.splitext(base)[0]
            if want and name not in want:
                continue
            fields[name] = np.fromfile(p, dtype=dtype)
        if fields:
            yield step, fields


def frames_openpmd(d, dtype, want=None):
    """WarpX: openpmd_%06d.h5, datasets under /data/<step>/fields/.

    Extraction is via the h5dump CLI (-b LE -o), not h5py, for the reason
    viz_openpmd.py gives: h5dump ships with HDF5 and is therefore present
    anywhere WarpX built against it, and h5py is not installed here.
    """
    dsets = want or ["E/x", "E/y", "E/z", "B/x", "B/y", "B/z",
                     "j/x", "j/y", "j/z", "rho"]
    files = []
    for p in glob.glob(os.path.join(d, "**", "*.h5"), recursive=True):
        m = re.search(r"(\d+)\.h5$", os.path.basename(p))
        if m:
            files.append((int(m.group(1)), p))
    for step, h5 in sorted(files):
        fields = {}
        for ds in dsets:
            path = f"/data/{step}/fields/{ds}"
            with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tf:
                tmp = tf.name
            try:
                r = subprocess.run(["h5dump", "-d", path, "-b", "LE", "-o", tmp, h5],
                                   capture_output=True)
                if r.returncode == 0 and os.path.getsize(tmp):
                    fields[ds.replace("/", "")] = np.fromfile(tmp, dtype=dtype)
            finally:
                os.unlink(tmp)
        if fields:
            yield step, fields


def frames_raw(d, dtype, want=None):
    """LAMMPS: <field>_step_<N>_chunk_<c>.bin, the bytes handed to the codec.

    Chunks of one field at one step are concatenated back in chunk order, so
    --block re-splits them on this tool's own boundary rather than inheriting
    whatever --chunk the run happened to use.
    """
    by_step = {}
    for p in glob.glob(os.path.join(d, "*_step_*_chunk_*.bin")):
        m = re.match(r"(.+)_step_(\d+)_chunk_(\d+)\.bin$", os.path.basename(p))
        if not m:
            continue
        name, step, chunk = m.group(1), int(m.group(2)), int(m.group(3))
        if want and name not in want:
            continue
        by_step.setdefault(step, {}).setdefault(name, []).append((chunk, p))
    for step in sorted(by_step):
        fields = {}
        for name, parts in by_step[step].items():
            fields[name] = np.concatenate(
                [np.fromfile(p, dtype=dtype) for _, p in sorted(parts)])
        yield step, fields


SOURCES = {"f32": frames_f32, "openpmd": frames_openpmd, "raw": frames_raw}


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------
def block_evolution(prev, cur, elems):
    """-> [(block_index, E, bad, same_cells, n_cells)] for one field, one pair.

    Norms accumulate in float64 whatever the field's own width: a 1 MiB
    float32 block is 262,144 terms and a float32 sum of squares of a field
    whose values reach 1e12 overflows outright.

    same_cells counts the elements BIT-IDENTICAL to the previous frame, which
    is a different question from E and the one that explains compressibility:
    E says how far a block moved, this says how much of it did not move at
    all. A block can carry a large E while most of its cells are untouched
    (a shock front crossing an otherwise quiet slab), and a block can have
    every cell changed by a hair and score a tiny E. The Nyx and VPIC
    READMEs quote this number as "bit-identical to the previous dump".

    Compared on the integer view of the bytes, not with ==, so it really is
    bit equality: -0.0 == 0.0 is true in float and these are different bytes,
    which a compressor sees and an equality test does not.
    """
    n = min(len(prev), len(cur))
    uint = np.uint32 if prev.dtype.itemsize == 4 else np.uint64
    pu, cu = prev[:n].view(uint), cur[:n].view(uint)
    out = []
    for i in range(0, n, elems):
        a = prev[i:i + elems].astype(np.float64)
        b = cur[i:i + elems].astype(np.float64)
        same = int(np.count_nonzero(pu[i:i + elems] == cu[i:i + elems]))
        ncell = len(a)
        bad = int(not (np.isfinite(a).all() and np.isfinite(b).all()))
        if bad:
            out.append((i // elems, float("nan"), 1, same, ncell))
            continue
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        e = np.linalg.norm(b - a) / (na + nb + EPS)
        out.append((i // elems, float(e), 0, same, ncell))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=sorted(SOURCES))
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--block", type=int, default=1048576,
                    help="bytes per block (default 1 MiB = WarpX's chunk)")
    ap.add_argument("--interval", type=int, default=0,
                    help="sample every N steps; 0 = every dump present")
    ap.add_argument("--stride", type=int, default=1,
                    help="use every Nth dump (a second sampling lever, applied "
                         "before --interval)")
    ap.add_argument("--step-scale", type=int, default=1,
                    help="timesteps per dump. Nyx and VPIC number their dump "
                         "directories by DUMP INDEX, not timestep, so a run "
                         "dumping every 10 steps needs --step-scale 10 for the "
                         "reported steps and --interval to mean timesteps.")
    ap.add_argument("--f64", action="store_true", help="fields are float64")
    ap.add_argument("--fields", default="", help="comma-separated subset")
    ap.add_argument("--active-thresh", type=float, default=1e-3)
    ap.add_argument("--skip-first", action="store_true",
                    help="drop the first dump (VPIC's step 0 is identically zero)")
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    dtype = np.float64 if a.f64 else np.float32
    elems = a.block // np.dtype(dtype).itemsize
    want = set(x for x in a.fields.split(",") if x) or None
    os.makedirs(a.out, exist_ok=True)

    rows, series, prev = [], [], None
    nbad = ndump = nsame = ncells = 0
    for step, fields in SOURCES[a.source](a.dir, dtype, want):
        ndump += 1
        if a.stride > 1 and (ndump - 1) % a.stride:
            continue
        step *= a.step_scale
        if a.interval and step % a.interval:
            continue
        if prev is None:
            if a.skip_first:
                prev = None
                a.skip_first = False
                continue
            prev = (step, fields)
            continue
        pstep, pf = prev
        vals, isame, icell = [], 0, 0
        for name in sorted(set(pf) & set(fields)):
            for idx, e, bad, same, ncell in block_evolution(
                    pf[name], fields[name], elems):
                rows.append((pstep, step, name, idx, e, bad,
                             100.0 * same / ncell if ncell else 0.0))
                nbad += bad
                isame += same
                icell += ncell
                if not bad:
                    vals.append(e)
        if vals:
            series.append((pstep, step, float(np.mean(vals)), len(vals),
                           100.0 * isame / icell if icell else 0.0))
        nsame += isame
        ncells += icell
        prev = (step, fields)

    if not rows:
        sys.exit(f"no frame pairs found under {a.dir}")

    with open(os.path.join(a.out, "blocks.csv"), "w") as f:
        f.write("step_from,step_to,field,block,evolution,nonfinite,pct_cells_same\n")
        for r in rows:
            f.write("%d,%d,%s,%d,%.9g,%d,%.6f\n" % r)

    e = np.array([r[4] for r in rows if not r[5]], dtype=np.float64)
    imeans = np.array([s[2] for s in series])
    q = max(1, len(imeans) // 4)
    summary = {
        "label": a.label or os.path.basename(a.out),
        "source": a.source, "dir": a.dir,
        "block_bytes": a.block, "interval": a.interval,
        "stride": a.stride, "step_scale": a.step_scale,
        # Measured from the dumps actually paired, not derived from the flags:
        # openPMD names its files by timestep and the f32 sources by dump
        # index, so only the data knows what the interval really was.
        "sample_interval_steps": int(np.median([s[1] - s[0] for s in series])),
        "dtype": "float64" if a.f64 else "float32",
        "frames": len(series) + 1, "pairs": len(series),
        "blocks_total": len(e), "nonfinite_blocks": nbad,
        "mean": float(e.mean()), "median": float(np.median(e)),
        "max": float(e.max()), "min": float(e.min()),
        "std": float(e.std()),
        "pct_active": float(100.0 * (e >= a.active_thresh).mean()),
        "pct_nonzero": float(100.0 * (e > 0).mean()),
        "active_thresh": a.active_thresh,
        "p10_interval": float(np.percentile(imeans, 10)),
        "last_quarter": float(imeans[-q:].mean()),
        "first_interval": float(imeans[0]), "max_interval": float(imeans.max()),
        # Cells BIT-IDENTICAL to the previous sampled frame. The complement is
        # what the compressor has to encode as new; a run where this stays high
        # is one where a fixed codec would do well and a per-chunk selector has
        # little to notice.
        "pct_cells_same": float(100.0 * nsame / ncells) if ncells else 0.0,
        "pct_cells_same_first": float(series[0][4]),
        "pct_cells_same_last": float(series[-1][4]),
        "cells_compared": int(ncells),
        "interval_means": [[s[0], s[1], s[2]] for s in series],
        "interval_pct_cells_same": [[s[0], s[1], s[4]] for s in series],
    }
    with open(os.path.join(a.out, "evolution.json"), "w") as f:
        json.dump(summary, f, indent=1)

    print(f"== {summary['label']}: {summary['frames']} frames, "
          f"{summary['pairs']} pairs, {summary['blocks_total']} block samples "
          f"({a.block // 1024} KiB blocks)")
    print(f"   mean {summary['mean']:.4f}  median {summary['median']:.4f}  "
          f"max {summary['max']:.4f}  min {summary['min']:.4f}")
    print(f"   active (>={a.active_thresh:g}) {summary['pct_active']:.1f}%   "
          f"changed at all {summary['pct_nonzero']:.1f}%")
    print(f"   sustained: p10 interval {summary['p10_interval']:.4f}, "
          f"last quarter {summary['last_quarter']:.4f}, "
          f"first {summary['first_interval']:.4f}")
    print(f"   cells bit-identical to previous frame: "
          f"{summary['pct_cells_same']:.2f}%  "
          f"(first pair {summary['pct_cells_same_first']:.2f}%, "
          f"last pair {summary['pct_cells_same_last']:.2f}%)")
    if nbad:
        print(f"   !! {nbad} block(s) held NaN/Inf -- configuration is unstable")


if __name__ == "__main__":
    main()
