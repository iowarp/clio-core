#!/usr/bin/env python3
"""Four-timestep temporal classifier: which parts of a simulation are Hot, Cold
or Deduplicable across T0 < T1 < T2 < T3.

    classify_temporal.py --root DIR --frames 150 151 152 153 --out RESULTS \
        --fig-dir FIGS [--block 16] [--block-sizes 8 16 32 64]
    classify_temporal.py --self-test

Every field is cut into the same spatial blocks at every timestep. Each block is
compared across the consecutive transitions T0->T1, T1->T2, T2->T3 (and the
long ones T0->T2, T0->T3, T1->T3), and every transition gets one state:

    EXACT  byte-identical (hash match, verified byte for byte)
    NEAR   not byte-identical, but at most --near-frac of its elements moved by
           more than --rel-tol (relative); default: every element within 1e-6
    COLD   change score <= the COLD/HOT threshold
    HOT    change score >  the COLD/HOT threshold

The change score (--score) defaults to `rms`: the block's RMS change divided by
the WHOLE FIELD's RMS (mean of the two timesteps), so every block of a field is
measured on one scale. `l2` is the spec's block-normalized L2 change; it divides
by the block's own values, so a block near zero (momentum at a blast centre)
reads ~1 however small its change -- reported for every block, but a poor
ranking of how much a region evolves.

The COLD/HOT threshold is not assumed. Unless --threshold is given it is derived
from the pooled distribution of log10(change score) over every
consecutive transition that is neither EXACT nor NEAR: a two-component Gaussian
mixture boundary when the populations are clearly separate (Ashman D >= 2),
Otsu's split otherwise. Every candidate is printed.

A block's window class is the majority of its three consecutive states at the
top level (Dedup / Cold / Hot); three different states resolve to the most
severe, and a Dedup majority is EXACT only when two of its states are EXACT.
A block that is HOT in any transition without being classed HOT is flagged
`transient_hot`, and the stricter "most severe state" class is reported too.

Percentages are by bytes (and by blocks, separately). Loading is pluggable:
subclass FieldSource for HDF5 / ADIOS2 / Zarr; RawF32Source reads the flat
float32 dumps of paper-benchmark/nyx/gen_fields.sh.
"""
import argparse
import glob
import json
import os
import re
import resource
import sys
import time

import numpy as np
import pandas as pd
import xxhash

EPS = 1e-30
EXACT, NEAR, COLD, HOT = 0, 1, 2, 3
STATE_NAMES = ["EXACT", "NEAR", "COLD", "HOT"]
CLASSES = ["EXACT_DEDUP", "NEAR_DEDUP", "COLD", "HOT"]
CONSEC = ["t0_t1", "t1_t2", "t2_t3"]
LONG = ["t0_t2", "t0_t3", "t1_t3"]
PAIRS = {"t0_t1": (0, 1), "t1_t2": (1, 2), "t2_t3": (2, 3),
         "t0_t2": (0, 2), "t0_t3": (0, 3), "t1_t3": (1, 3)}
SCORE_NAME = {"rms": "RMS change / field RMS", "l2": "block-normalized L2 change"}
ARROW = {p: f"T{a}→T{b}" for p, (a, b) in PAIRS.items()}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
class FieldSource:
    """One simulation, four timesteps, any number of named fields.

    Subclasses provide fields(), shape and load(field, t); everything else
    reads through this interface, so another format is one subclass."""

    def fields(self):
        raise NotImplementedError

    def load(self, field, t):
        """@return the field at timestep index t (0..3) as a C-order array."""
        raise NotImplementedError


class RawF32Source(FieldSource):
    """Flat float32 dumps: <root>/<frame>/fab0000_compNN_<field>.f32.

    AMReX writes x fastest, so a (nz, ny, nx) C-order reshape is the grid.
    @param root   the fields directory (holds pltNNNNN/ frame directories)
    @param frames four frame directory names, oldest first
    @param grid   (nz, ny, nx); default: gen.json's ncell, cubed
    """

    def __init__(self, root, frames, grid=None):
        self.root, self.frames = root, frames
        if grid is None:
            gen = json.load(open(os.path.join(root, "gen.json")))
            n = int(gen["ncell"])
            grid = (n, n, n)
        self.shape = tuple(grid)
        self.paths = {}
        for f in sorted(glob.glob(os.path.join(root, frames[0], "*.f32"))):
            name = re.sub(r"^fab\d+_comp\d+_", "", os.path.basename(f))[:-4]
            self.paths[name] = os.path.basename(f)

    def fields(self):
        return list(self.paths)

    def load(self, field, t):
        path = os.path.join(self.root, self.frames[t], self.paths[field])
        n = int(np.prod(self.shape))
        a = np.fromfile(path, dtype=np.float32)
        if a.size != n:
            raise ValueError(f"{path}: {a.size} values, grid {self.shape} needs {n}")
        return a.reshape(self.shape)


class ArraySource(FieldSource):
    """In-memory arrays {field: [T0, T1, T2, T3]}; used by the self-test."""

    def __init__(self, arrays):
        self.arrays = arrays
        self.shape = next(iter(arrays.values()))[0].shape

    def fields(self):
        return list(self.arrays)

    def load(self, field, t):
        return np.ascontiguousarray(self.arrays[field][t], dtype=np.float32)


# ---------------------------------------------------------------------------
# Chunking, hashing and per-transition metrics
# ---------------------------------------------------------------------------
def to_blocks(a, bs):
    """Cut a (nz, ny, nx) grid into (nblocks, block volume) rows.

    @param a  the grid
    @param bs (bz, by, bx); must divide the grid
    @return rows (contiguous copy), block coordinates (nblocks, 3), byte offset
            of each block's first element in the flat file
    """
    nz, ny, nx = a.shape
    bz, by, bx = bs
    if nz % bz or ny % by or nx % bx:
        raise ValueError(f"block {bs} does not divide grid {a.shape}")
    gz, gy, gx = nz // bz, ny // by, nx // bx
    rows = (a.reshape(gz, bz, gy, by, gx, bx).transpose(0, 2, 4, 1, 3, 5)
            .reshape(gz * gy * gx, bz * by * bx).copy())
    coords = np.indices((gz, gy, gx)).reshape(3, -1).T
    off = ((coords[:, 0] * bz) * ny * nx + (coords[:, 1] * by) * nx
           + coords[:, 2] * bx) * a.itemsize
    return rows, coords, off


def block_hashes(rows):
    """xxh3-64 of every block row, hex."""
    return [xxhash.xxh3_64_hexdigest(r) for r in rows]


class Timestep:
    """One timestep's block rows plus what every pair reuses: the float64
    copy, |x|, and each block's L1 and L2 norm (computed once, not per pair)."""

    def __init__(self, rows):
        self.raw = rows
        self.x = rows.astype(np.float64)
        self.ax = np.abs(self.x)
        self.l1 = self.ax.sum(1)
        self.l2 = np.sqrt(np.einsum("ij,ij->i", self.x, self.x))


def pair_metrics(A, B, rel_tol, scale):
    """Change of every block from A to B (two Timestep objects).

    @param rel_tol an element has CHANGED when |b - a| > rel_tol * max(|a|,|b|)
    @param scale   the field's RMS (mean of the two timesteps), for `rms`
    @return dict of per-block arrays: exact (byte-identical), mad, l1, l2
            (block-normalized), rms (RMS change / field RMS), max, frac
            (changed fraction), bitfrac (any bit differs), mass (sum |b - a|)
    """
    ua, ub = A.raw.view(np.uint32), B.raw.view(np.uint32)
    d = B.x - A.x
    ad = np.abs(d)
    ss = np.einsum("ij,ij->i", d, d)
    return {
        "exact": np.all(ua == ub, axis=1),
        "mad": ad.mean(1),
        "l1": ad.sum(1) / (A.l1 + B.l1 + EPS),
        "l2": np.sqrt(ss) / (A.l2 + B.l2 + EPS),
        "rms": np.sqrt(ss / d.shape[1]) / (scale + EPS),
        "max": ad.max(1),
        "frac": (ad > rel_tol * np.maximum(A.ax, B.ax)).mean(1),
        "bitfrac": (ua != ub).mean(1),
        "mass": ad.sum(1),
    }


def exact_relation(ex):
    """Which timesteps are byte-identical, e.g. 'T1=T2=T3' or 'T0=T2'.

    @param ex dict pair -> bool for one block
    """
    parent = list(range(4))

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i
    for p, (a, b) in PAIRS.items():
        if ex[p]:
            parent[find(b)] = find(a)
    groups = {}
    for t in range(4):
        groups.setdefault(find(t), []).append(t)
    parts = ["=".join(f"T{t}" for t in g) for g in groups.values() if len(g) > 1]
    return ";".join(parts) if parts else "none"


RELATION = [exact_relation({p: bool(c >> i & 1) for i, p in enumerate(PAIRS)})
            for c in range(1 << len(PAIRS))]


def field_block_table(field, grids, bs, rel_tol, timing):
    """Per-block metrics of one field at one block size.

    @param grids the field at T0..T3
    @return DataFrame, one row per block
    """
    t0 = time.perf_counter()
    rows, coords, off = zip(*(to_blocks(g, bs) for g in grids))
    coords, off = coords[0], off[0]
    df = pd.DataFrame({"field": field,
                       "block_index": np.arange(len(coords)),
                       "bz": coords[:, 0], "by": coords[:, 1], "bx": coords[:, 2],
                       "coordinates": [f"({z},{y},{x})" for z, y, x in coords],
                       "offset": off,
                       "size_bytes": rows[0].shape[1] * rows[0].itemsize})
    hashes = [block_hashes(r) for r in rows]
    for t in range(4):
        df[f"hash_t{t}"] = hashes[t]
    ts = [Timestep(r) for r in rows]
    rms = [float(np.sqrt((t.l2 ** 2).sum() / t.x.size)) for t in ts]
    for p, (a, b) in PAIRS.items():
        m = pair_metrics(ts[a], ts[b], rel_tol, 0.5 * (rms[a] + rms[b]))
        hash_eq = np.array([x == y for x, y in zip(hashes[a], hashes[b])])
        timing["hash_collisions"] += int(np.sum(hash_eq & ~m["exact"]))
        for k, v in m.items():
            df[f"{k}_{p}"] = v
    code = sum(df[f"exact_{p}"].to_numpy().astype(int) << i
               for i, p in enumerate(PAIRS))
    df["exact_duplicate"] = [RELATION[c] for c in code]
    timing["analysis_s"] += time.perf_counter() - t0
    return df


def measure(source, block_sizes, rel_tol):
    """Stream the fields one at a time; metrics for every block size.

    @return ({block size: DataFrame over all fields}, timing dict)
    """
    timing = {"io_s": 0.0, "analysis_s": 0.0, "hash_collisions": 0,
              "bytes_read": 0}
    tables = {bs: [] for bs in block_sizes}
    for field in source.fields():
        t0 = time.perf_counter()
        grids = [source.load(field, t) for t in range(4)]
        timing["io_s"] += time.perf_counter() - t0
        timing["bytes_read"] += sum(g.nbytes for g in grids)
        for bs in block_sizes:
            tables[bs].append(field_block_table(field, grids, bs, rel_tol, timing))
        del grids
    out = {}
    for bs, parts in tables.items():
        df = pd.concat(parts, ignore_index=True)
        df.insert(0, "block_id", np.arange(len(df)))
        out[bs] = df
    return out, timing


# ---------------------------------------------------------------------------
# Thresholds (derived from the data, never assumed)
# ---------------------------------------------------------------------------
def describe(x):
    """min/max/mean/median/std and the requested percentiles."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {}
    q = np.percentile(x, [10, 25, 50, 75, 90, 95, 99])
    return {"n": int(x.size), "min": x.min(), "max": x.max(), "mean": x.mean(),
            "median": np.median(x), "std": x.std(), "P10": q[0], "P25": q[1],
            "P50": q[2], "P75": q[3], "P90": q[4], "P95": q[5], "P99": q[6]}


def otsu(x, bins=256):
    """Otsu's split of a 1-D sample: the cut maximizing between-class variance."""
    h, e = np.histogram(x, bins)
    c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h).astype(float)
    w1 = w0[-1] - w0
    m = np.cumsum(h * c)
    mu0 = m / np.maximum(w0, 1)
    mu1 = (m[-1] - m) / np.maximum(w1, 1)
    k = int(np.argmax((w0 * w1 * (mu0 - mu1) ** 2)[:-1]))
    return float(e[k + 1])


def gmm_split(x):
    """Two-component Gaussian mixture on a 1-D sample.

    @return (boundary where the posteriors cross between the means, Ashman's
            D separation, the fitted means and sigmas); boundary None when the
            posteriors never cross between the means
    """
    from sklearn.mixture import GaussianMixture
    g = GaussianMixture(2, random_state=0).fit(x.reshape(-1, 1))
    mu = g.means_.ravel()
    sd = np.sqrt(g.covariances_.ravel())
    lo, hi = np.argsort(mu)
    grid = np.linspace(mu[lo], mu[hi], 2001).reshape(-1, 1)
    p_hi = g.predict_proba(grid)[:, hi]
    cross = np.nonzero(p_hi >= 0.5)[0]
    boundary = float(grid[cross[0], 0]) if cross.size else None
    d = float(np.sqrt(2) * abs(mu[hi] - mu[lo]) / np.sqrt(sd[lo] ** 2 + sd[hi] ** 2))
    return boundary, d, (float(mu[lo]), float(sd[lo]), float(mu[hi]), float(sd[hi]))


def valley(x, bins=64):
    """Deepest histogram point between the two tallest smoothed peaks, or None."""
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    h, e = np.histogram(x, bins)
    s = gaussian_filter1d(h.astype(float), 2)
    peaks, _ = find_peaks(s)
    if len(peaks) < 2:
        return None
    a, b = sorted(peaks[np.argsort(s[peaks])[-2:]])
    k = a + int(np.argmin(s[a:b + 1]))
    return float((e[k] + e[k + 1]) / 2)


def derive_threshold(scores, given=None):
    """Choose the COLD/HOT split on the change score.

    @param scores change score of every non-dedup consecutive transition
    @param given  a user threshold, which then wins
    @return dict: chosen threshold, method, every candidate, reasoning text
    """
    info = {"n_scores": int(len(scores)), "given": given}
    pos = scores[scores > 0]
    if given is not None:
        info.update(threshold=float(given), method="user-provided (--threshold)")
        info["why"] = f"--threshold {given} was given; no derivation."
        return info
    if pos.size < 10:
        info.update(threshold=float("inf"), method="none: too few changing blocks")
        info["why"] = (f"only {pos.size} non-dedup transitions; nothing is split, "
                       "every changing block is COLD.")
        return info
    lx = np.log10(pos)
    info["otsu"] = 10 ** otsu(lx)
    b, d, fit = gmm_split(lx)
    info["gmm"] = None if b is None else 10 ** b
    info["ashman_d"] = d
    info["gmm_fit_log10"] = fit
    v = valley(lx)
    info["valley"] = None if v is None else 10 ** v
    for q in (50, 75, 90):
        info[f"P{q}"] = float(np.percentile(pos, q))
    if info["gmm"] is not None and d >= 2.0:
        info.update(threshold=info["gmm"], method="Gaussian-mixture boundary")
        info["why"] = (f"log10(change) is bimodal: a 2-component mixture separates "
                       f"its populations with Ashman D = {d:.2f} (>= 2), so the "
                       f"posterior crossing {info['gmm']:.3g} is a real boundary.")
    else:
        info.update(threshold=info["otsu"], method="Otsu split")
        info["why"] = (f"no clean two-population structure (Ashman D = {d:.2f} < 2"
                       f"{'' if info['gmm'] is not None else ', no posterior crossing'}"
                       f"), so Otsu's variance-maximizing cut {info['otsu']:.3g} is "
                       "used: it is a split point, not a population boundary.")
    return info


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def transition_states(df, thr, near_frac, score):
    """EXACT / NEAR / COLD / HOT for every block and transition (in place)."""
    for p in PAIRS:
        s = np.where(df[f"{score}_{p}"] <= thr, COLD, HOT)
        near = ~df[f"exact_{p}"] & (df[f"frac_{p}"] <= near_frac)
        s = np.where(near, NEAR, s)
        df[f"state_{p}"] = np.where(df[f"exact_{p}"], EXACT, s)


def window_class(states):
    """Window class of each block from its three consecutive states.

    @param states (n, 3) of EXACT..HOT
    @return (majority class index into CLASSES, strict class = most severe)
    """
    top = np.where(states <= NEAR, 0, np.where(states == COLD, 1, 2))
    counts = np.stack([(top == k).sum(1) for k in range(3)], 1)
    win = np.where(counts.max(1) >= 2, counts.argmax(1), 2)
    dedup = np.where((states == EXACT).sum(1) >= 2, 0, 1)
    cls = np.where(win == 0, dedup, win + 1)
    return cls, states.max(1)


def classify(df, near_frac, threshold=None, score="rms"):
    """Derive the threshold, then label every transition and block (in place).

    @return the threshold info dict
    """
    for p in PAIRS:
        df[f"prelim_{p}"] = ~df[f"exact_{p}"] & (df[f"frac_{p}"] > near_frac)
    scores = np.concatenate([df.loc[df[f"prelim_{p}"], f"{score}_{p}"].to_numpy()
                             for p in CONSEC])
    info = derive_threshold(scores, threshold)
    df.drop(columns=[f"prelim_{p}" for p in PAIRS], inplace=True)
    info["score"] = score
    transition_states(df, info["threshold"], near_frac, score)
    st = df[[f"state_{p}" for p in CONSEC]].to_numpy()
    cls, strict = window_class(st)
    df["classification"] = [CLASSES[c] for c in cls]
    df["strict_classification"] = [CLASSES[c] for c in strict]
    df["transient_hot"] = (st == HOT).any(1) & (cls != HOT)
    ch = df[[f"{score}_{p}" for p in CONSEC]].to_numpy()
    for p in PAIRS:
        df[f"change_{p}"] = df[f"{score}_{p}"]
        df[f"changed_fraction_{p}"] = df[f"frac_{p}"]
    df["mean_change"] = ch.mean(1)
    df["max_change"] = ch.max(1)
    df["change_variance"] = ch.var(1)
    return info


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def shares(df, col):
    """% of bytes and % of blocks per class in `col`."""
    tot_b = df["size_bytes"].sum()
    out = {}
    for c in CLASSES:
        m = df[col] == c
        out[c] = (df.loc[m, "size_bytes"].sum(), 100 * df.loc[m, "size_bytes"].sum()
                  / tot_b, 100 * m.mean())
    return out


def transition_table(df):
    """% of bytes in each state, per transition."""
    tot = df["size_bytes"].sum()
    return {p: [100 * df.loc[df[f"state_{p}"] == s, "size_bytes"].sum() / tot
                for s in range(4)] for p in PAIRS}


def markov(df):
    """State-to-state matrix over (T0→T1, T1→T2) and (T1→T2, T2→T3), and
    the counts of each full three-transition path."""
    prev = np.concatenate([df["state_t0_t1"], df["state_t1_t2"]])
    nxt = np.concatenate([df["state_t1_t2"], df["state_t2_t3"]])
    m = np.zeros((4, 4))
    np.add.at(m, (prev, nxt), 1)
    paths = df[[f"state_{p}" for p in CONSEC]].apply(
        lambda r: "→".join(STATE_NAMES[s] for s in r), axis=1).value_counts()
    return m, paths


def neighbour_clustering(df):
    """Per class: P(a 6-neighbour block has the same class) / class share.

    Computed on each field's block grid and pooled; 1 means spatially random,
    above 1 means the class forms connected regions."""
    same = {c: 0 for c in CLASSES}
    slots = {c: 0 for c in CLASSES}
    for _, g in df.groupby("field"):
        shape = (g["bz"].max() + 1, g["by"].max() + 1, g["bx"].max() + 1)
        lab = np.empty(shape, dtype=object)
        lab[g["bz"], g["by"], g["bx"]] = g["classification"].to_numpy()
        for ax in range(3):
            a = np.moveaxis(lab, ax, 0)
            x, y = a[:-1].ravel(), a[1:].ravel()
            for c in CLASSES:
                same[c] += 2 * np.sum((x == c) & (y == c))
                slots[c] += np.sum(x == c) + np.sum(y == c)
    share = df["classification"].value_counts(normalize=True)
    return {c: (same[c] / slots[c]) / share[c] if slots[c] and c in share else None
            for c in CLASSES}


def concentration(df):
    """Share of temporal change held by the top 1/5/10/20% of blocks.

    Two measures, ranked by themselves: change magnitude (sum |Δ| over the
    three consecutive transitions, normalized per field so fields in different
    units can be pooled) and changed bytes (elements whose bits differ x 4)."""
    mass = df[[f"mass_{p}" for p in CONSEC]].sum(1)
    norm = mass / mass.groupby(df["field"]).transform("sum").replace(0, np.nan)
    nb = (df[[f"bitfrac_{p}" for p in CONSEC]].sum(1) * df["size_bytes"])
    out = {}
    for name, v, scale in (("change magnitude", norm.fillna(0), df["field"].nunique()),
                           ("changed bytes", nb, nb.sum())):
        s = np.sort(v.to_numpy())[::-1]
        cum = np.cumsum(s) / (scale if scale else 1)
        out[name] = {q: float(cum[max(1, int(np.ceil(q / 100 * len(s)))) - 1])
                     for q in (1, 5, 10, 20)}
        out[name + "_curve"] = cum
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def gb(x):
    return f"{x / 1e9:.3f} GB" if x >= 1e8 else f"{x / 2 ** 20:.1f} MiB"


def fmt_stats(name, d):
    if not d:
        return f"  {name}: (no samples)"
    keys = ["min", "P10", "P25", "P50", "P75", "P90", "P95", "P99", "max", "mean", "std"]
    return f"  {name:<24s} n={d['n']:6d} " + " ".join(f"{k}={d[k]:.3g}" for k in keys)


def threshold_text(info, rel_tol, near_frac):
    L = ["-" * 72, "THRESHOLDS (how they were determined)", "-" * 72,
         f"  changed element : |b - a| > {rel_tol:g} * max(|a|,|b|)  (--rel-tol)",
         f"  NEAR_DEDUP      : not byte-identical and changed fraction <= "
         f"{near_frac:g}  (--near-frac)",
         f"  COLD/HOT split  : {SCORE_NAME[info['score']]} <= {info['threshold']:.4g}"
         " is COLD  (--score)",
         f"  method          : {info['method']}", f"  why             : {info['why']}",
         f"  samples         : {info['n_scores']} non-dedup consecutive transitions"]
    if "otsu" in info:
        L.append("  candidates      : " + ", ".join(
            f"{k}={info[k]:.4g}" for k in ("otsu", "gmm", "valley", "P50", "P75", "P90")
            if info.get(k) is not None) + f"  (Ashman D {info['ashman_d']:.2f})")
    return L


def summary_text(label, frames, source, bs, df, info, timing, args):
    """The dataset-level report (printed and written to summary.txt)."""
    per_t = df["size_bytes"].sum()
    L = ["=" * 72, "FOUR-TIMESTEP TEMPORAL CLASSIFICATION", "=" * 72,
         f"Dataset          : {label}", "Timesteps        : " + ", ".join(
             f"T{i}={f}" for i, f in enumerate(frames)),
         f"Fields           : {', '.join(source.fields())}",
         f"Grid             : {' x '.join(map(str, source.shape))} per field",
         f"Size per timestep: {gb(per_t)}",
         f"Block            : {' x '.join(map(str, bs))}"
         f" ({df['size_bytes'].iloc[0] // 1024} KiB)",
         f"Blocks           : {len(df)} ({len(df) // df['field'].nunique()} per field)"]
    L += threshold_text(info, args.rel_tol, args.near_frac)
    sh = shares(df, "classification")
    L += ["-" * 72, "BY DATA SIZE (window class; blocks are equal-sized, so % of "
          "blocks is identical)", "-" * 72]
    ded = sh["EXACT_DEDUP"][0] + sh["NEAR_DEDUP"][0]
    for name, (b, pb, _) in (("Hot", sh["HOT"]), ("Cold", sh["COLD"]),
                             ("Deduplicable", (ded, 100 * ded / per_t, 0))):
        L.append(f"  {name:<14s} {gb(b):>12s}  {pb:6.2f}%")
    L += [f"    Exact dup    {gb(sh['EXACT_DEDUP'][0]):>12s}  {sh['EXACT_DEDUP'][1]:6.2f}%",
          f"    Near dup     {gb(sh['NEAR_DEDUP'][0]):>12s}  {sh['NEAR_DEDUP'][1]:6.2f}%",
          f"  {'TOTAL':<14s} {gb(per_t):>12s}  "
          f"{sum(v[1] for v in sh.values()):6.2f}%"]
    tr = df["transient_hot"]
    L.append(f"  transient hot (HOT in >=1 transition, not classed HOT): "
             f"{100 * tr.mean():.2f}% of bytes")
    st = shares(df, "strict_classification")
    L.append("  strict class (most severe transition): " + ", ".join(
        f"{c} {st[c][1]:.2f}%" for c in CLASSES))
    L += ["-" * 72, "PER TRANSITION (% of bytes)", "-" * 72,
          "  " + f"{'':12s}" + "".join(f"{ARROW[p]:>9s}" for p in PAIRS)]
    tt = transition_table(df)
    for s in range(4):
        L.append(f"  {STATE_NAMES[s]:<12s}" + "".join(f"{tt[p][s]:8.2f}%" for p in PAIRS))
    L += ["-" * 72, "DISTRIBUTIONS (consecutive transitions)", "-" * 72]
    for p in CONSEC:
        nd = ~df[f"exact_{p}"]
        L.append(fmt_stats(f"score {ARROW[p]}", describe(df.loc[nd, f"{info['score']}_{p}"])))
    pool = lambda k: np.concatenate([df.loc[~df[f"exact_{p}"], f"{k}_{p}"] for p in CONSEC])
    L.append(fmt_stats("RMS change/field, pooled", describe(pool("rms"))))
    L.append(fmt_stats("block-norm. L2, pooled", describe(pool("l2"))))
    L.append("  element level (the dedup ceiling of any block size): " + ", ".join(
        f"{ARROW[p]} {100 * (1 - np.average(df[f'bitfrac_{p}'], weights=df['size_bytes'])):.2f}%"
        for p in CONSEC) + " of values bit-identical")
    L.append(fmt_stats("changed fraction, pooled", describe(pool("frac"))))
    for f, g in df.groupby("field", sort=False):
        L.append(fmt_stats(f"max |Δ| {f}", describe(np.concatenate(
            [g.loc[~g[f"exact_{p}"], f"max_{p}"] for p in CONSEC]))))
    return L


def dynamics_text(df):
    """Transition matrix, persistence, spatial clustering, concentration."""
    m, paths = markov(df)
    rows = m / np.maximum(m.sum(1, keepdims=True), 1)
    L = ["-" * 72, "STATE TRANSITION MATRIX (row = previous transition's state)",
         "-" * 72, "  " + f"{'':8s}" + "".join(f"{s:>8s}" for s in STATE_NAMES) + "      n"]
    for i in range(4):
        L.append(f"  {STATE_NAMES[i]:<8s}" + "".join(f"{100 * rows[i, j]:7.1f}%"
                                                    for j in range(4))
                 + f"  {int(m[i].sum()):5d}")
    L.append("  most common three-transition paths (% of blocks):")
    for k, v in paths.head(8).items():
        L.append(f"    {k:<24s} {100 * v / len(df):6.2f}%")
    L += ["-" * 72, "SPATIAL CLUSTERING (same-class neighbour rate / class share;"
          " 1 = random)", "-" * 72]
    for c, v in neighbour_clustering(df).items():
        L.append(f"  {c:<12s} {'n/a' if v is None else f'{v:.2f}x'}")
    con = concentration(df)
    L += ["-" * 72, "CHANGE CONCENTRATION (share of the total held by the top blocks)",
          "-" * 72]
    for k in ("change magnitude", "changed bytes"):
        L.append(f"  {k:<18s}" + "  ".join(f"top {q:>2d}%: {100 * v:5.1f}%"
                                           for q, v in con[k].items()))
    return L


def answers_text(df, sens, info):
    """Plain answers to the research questions, computed from the tables."""
    sh = shares(df, "classification")
    tt = transition_table(df)
    exact_c = np.mean([tt[p][EXACT] for p in CONSEC])
    near_c = np.mean([tt[p][NEAR] for p in CONSEC])
    st = df[[f"state_{p}" for p in CONSEC]].to_numpy()
    hot = df["classification"] == "HOT"
    cold = df["classification"] == "COLD"
    clu = neighbour_clustering(df)
    con = concentration(df)
    L = ["-" * 72, "ANSWERS", "-" * 72,
         f"1. Exactly unchanged between consecutive timesteps: {exact_c:.2f}% of bytes "
         f"on average ({', '.join(f'{ARROW[p]} {tt[p][EXACT]:.2f}%' for p in CONSEC)}).",
         f"2. Deduplicable over the window: {sh['EXACT_DEDUP'][1] + sh['NEAR_DEDUP'][1]:.2f}%"
         f" (exact {sh['EXACT_DEDUP'][1]:.2f}%, near {sh['NEAR_DEDUP'][1]:.2f}%; near-"
         f"identical transitions average {near_c:.2f}%).",
         f"3. Cold (slowly changing): {sh['COLD'][1]:.2f}%.",
         f"4. Hot (substantially changing): {sh['HOT'][1]:.2f}% "
         f"(split at {SCORE_NAME[info['score']]} {info['threshold']:.4g}, {info['method']}).",
         "5. Hot regions spatially concentrated: " + (
             f"same-class neighbour rate {clu['HOT']:.2f}x the random expectation."
             if clu.get("HOT") else "no HOT blocks."),
         "6. Cold regions persistent: " + (
             f"{100 * np.mean((st[cold] == COLD).all(1)):.1f}% of COLD blocks are COLD in "
             f"all three transitions; clustering {clu['COLD']:.2f}x." if cold.any()
             else "no COLD blocks."),
         "7. Same regions hot throughout: " + (
             f"{100 * np.mean((st[hot] == HOT).all(1)):.1f}% of HOT blocks are HOT in all "
             f"three transitions; {100 * (st == HOT).all(1).mean():.2f}% of all blocks."
             if hot.any() else "no HOT blocks.")]
    if sens is not None:
        L.append("8. Granularity (block edge: Hot/Cold/Dedup %, re-derived threshold): "
                 + "; ".join(f"{r.block}³ {r.hot:.1f}/{r.cold:.1f}/{r.dedup:.1f}"
                             for r in sens.itertuples() if r.mode == "re-derived"))
    L.append(f"9-10. The top 10% of blocks hold {100 * con['change magnitude'][10]:.1f}% "
             f"of the change magnitude and {100 * con['changed bytes'][10]:.1f}% of the "
             "changed bytes (top 1/5/10/20%: " + ", ".join(
                 f"{100 * v:.1f}%" for v in con["change magnitude"].values()) + ").")
    return L


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
COLORS = {"EXACT_DEDUP": "#2a78d6", "HOT": "#eb6834", "COLD": "#1baf7a",
          "NEAR_DEDUP": "#eda100"}
STATE_COLORS = [COLORS["EXACT_DEDUP"], COLORS["NEAR_DEDUP"], COLORS["COLD"],
                COLORS["HOT"]]
LABEL = {"EXACT_DEDUP": "Exact dedup", "NEAR_DEDUP": "Near dedup", "COLD": "Cold",
         "HOT": "Hot"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 10,
                         "font.serif": ["Times New Roman", "Nimbus Roman",
                                        "STIXGeneral", "DejaVu Serif"],
                         "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    return plt


def _style(ax, grid_axis="y"):
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_distributions(df, info, args, out):
    """Figures 1 and 2: normalized change and changed-fraction distributions."""
    plt = _plt()
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    x = np.concatenate([df.loc[~df[f"exact_{p}"], f"{info['score']}_{p}"] for p in CONSEC])
    x = x[x > 0]
    ax = axs[0]
    if x.size:
        ax.hist(np.log10(x), bins=60, color="#2a78d6", edgecolor="white", linewidth=0.4)
    for key, ls in (("otsu", ":"), ("gmm", "--"), ("valley", "-.")):
        v = info.get(key)
        if v:
            ax.axvline(np.log10(v), color=MUTED, ls=ls, lw=1, label=f"{key} {v:.3g}")
    if np.isfinite(info["threshold"]):
        ax.axvline(np.log10(info["threshold"]), color=INK, lw=1.4,
                   label=f"chosen {info['threshold']:.3g}")
    ax.set_xlabel(f"log10 {SCORE_NAME[info['score']]} (non-identical transitions)")
    ax.set_ylabel("Block transitions")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fr = np.concatenate([df.loc[~df[f"exact_{p}"], f"frac_{p}"] for p in CONSEC])
    ax = axs[1]
    ax.hist(fr, bins=50, range=(0, 1), color="#2a78d6", edgecolor="white", linewidth=0.4)
    ax.set_xlabel(f"Fraction of elements changed (rel. tol {args.rel_tol:g})")
    ax.set_ylabel("Block transitions")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def fig_breakdown(df, out):
    """Figure 3: share of bytes per window class, directly labeled."""
    plt = _plt()
    sh = shares(df, "classification")
    order = ["HOT", "COLD", "NEAR_DEDUP", "EXACT_DEDUP"]
    fig, ax = plt.subplots(figsize=(6, 2.6))
    vals = [sh[c][1] for c in order]
    ax.barh(range(4), vals, color=[COLORS[c] for c in order], height=0.62,
            edgecolor="white", linewidth=2)
    for i, v in enumerate(vals):
        ax.text(v + 1, i, f"{v:.1f}%", va="center", color=INK, fontsize=9)
    ax.set_yticks(range(4), [LABEL[c] for c in order])
    ax.set_xlim(0, 110)
    ax.set_xlabel("% of bytes (window class)")
    _style(ax, "x")
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _stacked(ax, rows, labels):
    """100%-stacked horizontal bars of per-state shares, labeled when >= 6%."""
    for i, r in enumerate(rows):
        left = 0.0
        for s, v in enumerate(r):
            ax.barh(i, v, left=left, color=STATE_COLORS[s], height=0.6,
                    edgecolor="white", linewidth=1.5)
            if v >= 6:
                ax.text(left + v / 2, i, f"{v:.0f}%", ha="center", va="center",
                        fontsize=8, color=INK)
            left += v
    ax.set_yticks(range(len(rows)), labels)
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    _style(ax, "x")


def fig_transitions(df, out):
    """Figure 4: state shares per transition, consecutive and long."""
    plt = _plt()
    tt = transition_table(df)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    _stacked(ax, [tt[p] for p in PAIRS], [ARROW[p] for p in PAIRS])
    ax.set_xlabel("% of bytes")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in STATE_COLORS],
              labels=["Exact", "Near", "Cold", "Hot"], frameon=False, ncol=4,
              loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def fig_spatial(df, source, out):
    """Figure 5: window class on the mid-z block slice of every field, beside
    the first field's data at T3 on the same slice."""
    from matplotlib.colors import ListedColormap
    plt = _plt()
    fields = list(df["field"].unique())
    n = len(fields) + 1
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.6 * rows + 0.5),
                            squeeze=False)
    cmap = ListedColormap([COLORS[c] for c in CLASSES])
    for ax, f in zip(axs.flat, fields):
        g = df[df["field"] == f]
        gz = g["bz"].max() + 1
        s = g[g["bz"] == gz // 2]
        lab = np.full((g["by"].max() + 1, g["bx"].max() + 1), np.nan)
        lab[s["by"], s["bx"]] = [CLASSES.index(c) for c in s["classification"]]
        ax.imshow(lab, cmap=cmap, vmin=-0.5, vmax=3.5, origin="lower",
                  interpolation="nearest")
        ax.set_title(f, fontsize=9, color=INK)
        ax.set_xticks([]), ax.set_yticks([])
    ax = axs.flat[len(fields)]
    a = source.load(fields[0], 3)
    ax.imshow(a[a.shape[0] // 2], cmap="Blues", origin="lower")
    ax.set_title(f"{fields[0]} at T3 (data)", fontsize=9, color=INK)
    ax.set_xticks([]), ax.set_yticks([])
    for ax in list(axs.flat)[n:]:
        ax.axis("off")
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COLORS[c]) for c in CLASSES],
               labels=[LABEL[c] for c in CLASSES], frameon=False, ncol=4,
               loc="upper center", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=2.0)
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def fig_matrix(df, out):
    """State-to-state transition matrix as an annotated heatmap."""
    plt = _plt()
    m, _ = markov(df)
    r = 100 * m / np.maximum(m.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(r, cmap="Blues", vmin=0, vmax=100)
    for i in range(4):
        for j in range(4):
            if m[i].sum():
                ax.text(j, i, f"{r[i, j]:.0f}%", ha="center", va="center", fontsize=9,
                        color="white" if r[i, j] > 60 else INK)
    ax.set_xticks(range(4), STATE_NAMES)
    ax.set_yticks(range(4), [f"{s} (n={int(m[i].sum())})"
                             for i, s in enumerate(STATE_NAMES)])
    ax.set_xlabel("Next transition's state")
    ax.set_ylabel("Previous transition's state")
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def fig_sensitivity(sens, out):
    """Window-class shares per block size (re-derived threshold)."""
    plt = _plt()
    s = sens[sens["mode"] == "re-derived"]
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(s) + 1.3))
    _stacked(ax, [[r.exact, r.near, r.cold, r.hot] for r in s.itertuples()],
             [f"{r.block}³" for r in s.itertuples()])
    ax.set_xlabel("% of bytes")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in STATE_COLORS],
              labels=["Exact dedup", "Near dedup", "Cold", "Hot"], frameon=False,
              ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def fig_concentration(df, out):
    """Cumulative share of temporal change vs share of blocks, hottest first."""
    plt = _plt()
    con = concentration(df)
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for (k, c) in (("change magnitude", "#2a78d6"), ("changed bytes", "#eb6834")):
        y = con[k + "_curve"]
        ax.plot(100 * np.arange(1, len(y) + 1) / len(y), 100 * y, color=c, lw=2, label=k)
    ax.plot([0, 100], [0, 100], color=MUTED, lw=0.8, ls=":")
    ax.axvline(10, color=GRID, lw=1)
    ax.set_xlabel("% of blocks (largest change first)")
    ax.set_ylabel("% of total change")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _style(ax, "both")
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Granularity, validation, main
# ---------------------------------------------------------------------------
def sensitivity(tables, primary, info, args):
    """Class shares per block size: threshold re-derived per size, and the
    primary size's threshold held fixed (isolates the granularity effect)."""
    rows = []
    for bs, df0 in sorted(tables.items()):
        for mode, thr in (("re-derived", args.threshold), ("fixed", info["threshold"])):
            if bs == primary and mode == "fixed":
                continue
            df = df0.copy()
            i = classify(df, args.near_frac, thr, args.score)
            sh = shares(df, "classification")
            rows.append(dict(block=bs[0], mode=mode, threshold=i["threshold"],
                             method=i["method"], exact=sh["EXACT_DEDUP"][1],
                             near=sh["NEAR_DEDUP"][1], cold=sh["COLD"][1],
                             hot=sh["HOT"][1],
                             dedup=sh["EXACT_DEDUP"][1] + sh["NEAR_DEDUP"][1]))
    return pd.DataFrame(rows)


def self_test():
    """Known blocks must come out with their known class; shares sum to 100."""
    rng = np.random.default_rng(0)
    ex = [np.ones(4)] * 4
    cold = [np.array(v) for v in ([1, 1, 1, 1], [1, 1.01, 1, 1], [1, 1.01, 1.01, 1],
                                   [1.01, 1.01, 1.01, 1])]
    hot = [np.array(v, float) for v in ([1, 1, 1, 1], [4, 2, 7, 3], [9, 1, 8, 5],
                                         [3, 8, 2, 9])]
    def variant(kind, base, k):
        """k=0 is the spec's block verbatim; the rest vary scale AND change."""
        if k == 0:
            return base
        s = rng.uniform(0.5, 2.0)
        if kind == "COLD":      # same pattern, 0.2x-3x the spec's 0.01 steps
            u = rng.uniform(0.2, 3.0)
            return [s * (1 + u * (b - 1)) for b in base]
        if kind == "HOT":       # fresh random values every timestep
            return [s * rng.integers(1, 10, 4).astype(float) for _ in base]
        return [b * s for b in base]
    blocks, want = [], []
    for kind, base in (("EXACT_DEDUP", ex), ("COLD", cold), ("HOT", hot)):
        for k in range(60):
            blocks.append(variant(kind, base, k))
            want.append(kind)
    grids = [np.concatenate([b[t] for b in blocks]).reshape(1, 1, -1) for t in range(4)]
    src = ArraySource({"synthetic": grids})
    ok = True
    for score, thr in (("rms", None), ("l2", None), ("l2", 0.05)):
        tables, _ = measure(src, [(1, 1, 4)], 1e-6)
        df = tables[(1, 1, 4)]
        info = classify(df, 0.0, thr, score)
        got = df["classification"].tolist()
        bad = sum(g != w for g, w in zip(got, want))
        tot = sum(v[1] for v in shares(df, "classification").values())
        print(f"self-test score={score} threshold={'derived' if thr is None else thr}: "
              f"{info['method']} -> {info['threshold']:.4g}; spec blocks "
              f"{got[0]}/{got[60]}/{got[120]}; mismatches {bad}/{len(want)}; "
              f"shares sum {tot:.6f}%")
        ok &= bad == 0 and abs(tot - 100) < 1e-9
    print("SELF-TEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def write_outputs(df, info, sens, lines, timing, args):
    """Per-block CSV + Parquet, thresholds JSON, sensitivity CSV, summary."""
    os.makedirs(args.out, exist_ok=True)
    keep = (["block_id", "field", "block_index", "coordinates", "offset", "size_bytes"]
            + [f"hash_t{t}" for t in range(4)] + ["exact_duplicate"]
            + [f"change_{p}" for p in PAIRS] + [f"changed_fraction_{p}" for p in PAIRS]
            + ["mean_change", "max_change", "change_variance", "classification",
               "strict_classification", "transient_hot"]
            + [f"state_{p}" for p in PAIRS]
            + [f"{k}_{p}" for k in ("mad", "l1", "l2", "rms", "max", "bitfrac", "mass") for p in PAIRS])
    out = df[keep].copy()
    for p in PAIRS:
        out[f"state_{p}"] = [STATE_NAMES[s] for s in out[f"state_{p}"]]
    out.to_csv(os.path.join(args.out, "blocks.csv"), index=False)
    out.to_parquet(os.path.join(args.out, "blocks.parquet"), index=False)
    json.dump({k: v for k, v in info.items()}, open(os.path.join(
        args.out, "thresholds.json"), "w"), indent=1, default=float)
    if sens is not None:
        sens.to_csv(os.path.join(args.out, "granularity.csv"), index=False)
    m, _ = markov(df)
    pd.DataFrame(m, index=STATE_NAMES, columns=STATE_NAMES).to_csv(
        os.path.join(args.out, "transition_matrix.csv"))
    open(os.path.join(args.out, "summary.txt"), "w").write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", help="fields directory (holds the frame dirs)")
    ap.add_argument("--frames", nargs=4, help="four frames, oldest first: numbers "
                    "(-> plt%%05d) or directory names")
    ap.add_argument("--grid", nargs=3, type=int, help="nz ny nx (default: gen.json)")
    ap.add_argument("--block", type=int, default=16, help="cubic block edge")
    ap.add_argument("--block-sizes", nargs="*", type=int, default=[8, 16, 32, 64])
    ap.add_argument("--rel-tol", type=float, default=1e-6)
    ap.add_argument("--near-frac", type=float, default=0.0)
    ap.add_argument("--threshold", type=float, help="COLD/HOT split (else derived)")
    ap.add_argument("--score", choices=["rms", "l2"], default="rms",
                    help="change score the COLD/HOT split is made on")
    ap.add_argument("--label", default="dataset")
    ap.add_argument("--out", help="results directory")
    ap.add_argument("--fig-dir", help="figure directory")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not (args.root and args.frames and args.out and args.fig_dir):
        ap.error("--root, --frames, --out and --fig-dir are required")
    frames = [f"plt{int(f):05d}" if f.isdigit() else f for f in args.frames]
    source = RawF32Source(args.root, frames, args.grid)
    primary = (args.block,) * 3
    sizes = sorted({(b,) * 3 for b in args.block_sizes} | {primary})
    t0 = time.perf_counter()
    tables, timing = measure(source, sizes, args.rel_tol)
    df = tables[primary]
    info = classify(df, args.near_frac, args.threshold, args.score)
    sens = sensitivity(tables, primary, info, args) if len(sizes) > 1 else None
    lines = summary_text(args.label, frames, source, primary, df, info, timing, args)
    lines += dynamics_text(df)
    if sens is not None:
        lines += ["-" * 72, "GRANULARITY SENSITIVITY (% of bytes)", "-" * 72,
                  sens.to_string(index=False, float_format=lambda v: f"{v:.3g}")]
    lines += answers_text(df, sens, info)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    lines += ["-" * 72, f"RUNTIME: I/O {1000 * timing['io_s']:.0f} ms "
              f"({gb(timing['bytes_read'])} read), analysis {1000 * timing['analysis_s']:.0f}"
              f" ms over {len(sizes)} block size(s), total "
              f"{1000 * (time.perf_counter() - t0):.0f} ms; peak RSS {rss:.0f} MiB; "
              f"hash collisions {timing['hash_collisions']}",
              "COMMAND: " + " ".join(sys.argv)]
    print("\n".join(lines))
    write_outputs(df, info, sens, lines, timing, args)
    os.makedirs(args.fig_dir, exist_ok=True)
    F = lambda n: os.path.join(args.fig_dir, n)
    fig_distributions(df, info, args, F("fig1_2_change_distributions.png"))
    fig_breakdown(df, F("fig3_breakdown.png"))
    fig_transitions(df, F("fig4_transitions.png"))
    fig_spatial(df, source, F("fig5_spatial_map.png"))
    fig_matrix(df, F("fig6_transition_matrix.png"))
    fig_concentration(df, F("fig7_change_concentration.png"))
    if sens is not None:
        fig_sensitivity(sens, F("fig8_granularity.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
