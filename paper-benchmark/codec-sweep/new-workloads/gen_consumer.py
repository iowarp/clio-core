#!/usr/bin/env python3
"""Producer-consumer workloads: the ANALYSIS OUTPUTS a pipeline writes after a
simulation, computed from the dumps the four simulations already produced.

    gen_consumer.py --wl nyx|vpic|lammps --dump DIR --out DIR [--every N]

Every product is written as one flat little-endian file per frame,
<out>/<product>_f<NNNN>.<dtype>, so codec_sweep / corpus_sweep can chunk it.

  nyx, vpic (3-D float32 grids)
    cc_label     int32  connected-component label of every cell above the
                        90th density percentile (0 = background); what a
                        halo / structure finder writes
    kmeans       int32  cluster id (k = 8) of every cell, clustered on
                        (log density, |momentum| or |B|, energy or |E|)
    pdf2d        f64    1024 x 1024 joint probability density of two fields
                        (most bins empty)
  lammps (1.37 M atoms, float64)
    coordination int32  neighbours within 1.5 sigma of each atom
    cell_sorted  int32  atom ids sorted by spatial cell (a cell-list order)
    kmeans       int32  cluster id (k = 8) on per-atom speed / coordination
    rdf          f64    radial distribution histogram, 4096 bins
"""
import argparse
import glob
import os
import re

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from sklearn.cluster import MiniBatchKMeans


def write(out, name, f, arr):
    arr = np.ascontiguousarray(arr)
    ext = {np.dtype("int32"): "i32", np.dtype("float64"): "f64"}[arr.dtype]
    arr.tofile(os.path.join(out, f"{name}_f{f:04d}.{ext}"))


def kmeans_labels(feats, k=8, seed=0):
    """Fit on a 200k sample, label everything."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(feats), size=min(200_000, len(feats)), replace=False)
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3,
                         batch_size=8192).fit(feats[idx])
    return km.predict(feats).astype(np.int32)


def standardise(*cols):
    m = np.stack(cols, axis=1).astype(np.float32)
    m -= m.mean(0)
    m /= m.std(0) + 1e-12
    return m


def grid_frames(d):
    """@return [(frame index, {field: path})] for plt* dump directories."""
    out = []
    for i, p in enumerate(sorted(glob.glob(os.path.join(d, "plt*")))):
        fields = {}
        for f in glob.glob(os.path.join(p, "*.f32")):
            m = re.match(r"fab\d+_comp\d+_(.+)\.f32$", os.path.basename(f))
            if m:
                fields[m.group(1)] = f
        out.append((i, fields))
    return out


def grid_products(wl, frames, out, every):
    for i, fields in frames[::every]:
        n = os.path.getsize(next(iter(fields.values()))) // 4
        side = int(round((n) ** (1 / 3)))
        load = lambda k: np.fromfile(fields[k], dtype=np.float32).reshape(side, side, side)
        if wl == "nyx":
            a = load("density")
            b = np.sqrt(load("xmom") ** 2 + load("ymom") ** 2 + load("zmom") ** 2)
            c = load("rho_E")
        else:  # vpic
            a = np.abs(load("rhof")) + 1e-30
            b = np.sqrt(load("cbx") ** 2 + load("cby") ** 2 + load("cbz") ** 2)
            c = np.sqrt(load("ex") ** 2 + load("ey") ** 2 + load("ez") ** 2)
        mask = a > np.percentile(a, 90)
        lab, _ = ndimage.label(mask)
        write(out, "cc_label", i, lab.astype(np.int32))
        la = np.log10(np.maximum(a, 1e-30))
        write(out, "kmeans", i, kmeans_labels(standardise(la.ravel(), b.ravel(), c.ravel())))
        h, _, _ = np.histogram2d(la.ravel(), np.log10(np.maximum(c.ravel(), 1e-30)),
                                 bins=1024, density=True)
        write(out, "pdf2d", i, h.astype(np.float64))
        print(f"{wl} frame {i}: {lab.max()} components", flush=True)


def lammps_products(d, out, every):
    steps = sorted({int(m.group(1)) for f in os.listdir(d)
                    if (m := re.match(r"position_step_(\d+)_chunk_0\.bin$", f))})
    # box 70 fcc at density 0.8442 -> L = 70 * (4/0.8442)^(1/3)
    L = 70 * (4 / 0.8442) ** (1 / 3)
    for i, s in enumerate(steps[::every]):
        x = np.fromfile(os.path.join(d, f"position_step_{s}_chunk_0.bin"), dtype=np.float64).reshape(-1, 3)
        v = np.fromfile(os.path.join(d, f"velocity_step_{s}_chunk_0.bin"), dtype=np.float64).reshape(-1, 3)
        xw = np.mod(x, L)
        tree = cKDTree(xw, boxsize=L)
        coord = tree.query_ball_point(xw, r=1.5, return_length=True, workers=-1).astype(np.int32)
        write(out, "coordination", i, coord)
        cell = np.floor(xw / 1.5).astype(np.int64)
        nc = int(np.ceil(L / 1.5))
        key = (cell[:, 0] * nc + cell[:, 1]) * nc + cell[:, 2]
        write(out, "cell_sorted", i, np.argsort(key, kind="stable").astype(np.int32))
        speed = np.linalg.norm(v, axis=1)
        write(out, "kmeans", i, kmeans_labels(standardise(speed, coord)))
        sample = xw[np.random.default_rng(i).choice(len(xw), 20000, replace=False)]
        dists = tree.query_ball_point(sample, r=5.0, workers=-1)
        r = np.concatenate([np.linalg.norm(((xw[nb] - p + L / 2) % L) - L / 2, axis=1)
                            for p, nb in zip(sample, dists)])
        rdf, _ = np.histogram(r[r > 0], bins=4096, range=(0, 5.0))
        write(out, "rdf", i, rdf.astype(np.float64))
        print(f"lammps step {s}: mean coordination {coord.mean():.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wl", required=True, choices=["nyx", "vpic", "lammps"])
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.wl == "lammps":
        lammps_products(a.dump, a.out, a.every)
    else:
        grid_products(a.wl, grid_frames(a.dump), a.out, a.every)


if __name__ == "__main__":
    main()
