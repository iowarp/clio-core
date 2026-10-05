#!/usr/bin/env python3
"""pattern_lab.py for VPIC's frozen vacuum fields: which E-field slabs make
one codec bad and NeuroPress right?

    pattern_lab_vpic.py gen | score

Candidates are `ex` chunks exactly as the VPIC deck would hold them at 254^3
(a 4 MiB chunk = 16 z-planes x 256 x 256 voxels, x fastest): a uniform
background E0 (curl-free, periodic) plus either a clumpy part, the x
difference of random integer levels times a power-of-two unit, or a smooth
part, the x difference of a sine-product potential, rounded so every value is
exact (a power-of-two grid) or left in floating point. Scored as
pattern_lab.py does: the training sweep tool on all 45 settings, the
balanced 4-tier cost, and the trained model's prediction and pick.
"""
import os
import sys

import numpy as np

import pattern_lab as pl

pl.LAB = "/mnt/nvme0/tune/lab-vpic"
pl.SHAPE = (16, 256, 256)


def ex_of(phi):
    """ex = -(phi[i+1] - phi[i]) along x, periodic."""
    return -(np.roll(phi, -1, axis=2) - phi)


def patterns(seed):
    """name -> float32 array, one candidate chunk per pattern."""
    rng = np.random.default_rng(seed)
    nz, ny, nx = pl.SHAPE
    k0 = 16 * (seed + 2)                       # which slab of the 254^3 box
    z, y, x = np.meshgrid((k0 + np.arange(nz)) / 254, np.arange(ny) / 254,
                          np.arange(nx) / 254, indexing="ij")
    wave = np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y + 0.7) * np.sin(2 * np.pi * z + 1.3)
    lv = rng.integers(0, 8, pl.SHAPE).astype(float)
    p = {}
    for e0 in (0.0, 1.0):
        for u in (2.0 ** -7, 2.0 ** -4):
            p[f"clumpy_E0{e0:g}_u2^{int(np.log2(u))}"] = e0 + ex_of(lv * u)
    for e0 in (0.0, 1.0):
        for amp in (0.05, 0.3):
            pot = amp / (2 * np.pi / 254) * wave           # |ex| peaks near amp
            p[f"smooth_float_E0{e0:g}_a{amp:g}"] = e0 + ex_of(pot)
            for qb in (20, 23):
                q = 2.0 ** -qb
                p[f"smooth_exact{qb}_E0{e0:g}_a{amp:g}"] = e0 + ex_of(np.round(pot / q) * q)
    # Clumpy slab with a diagonal potential phi = level((i + k) mod n, j) * u:
    # ex and ez are then both differences of neighbouring levels along x.
    kk0 = 16 * (2 * (seed % 6)) + np.arange(16)
    ii = np.arange(257)
    lvl = rng.integers(0, 8, (512, 256)).astype(float) * 2.0 ** -7
    phd = lvl[(ii[None, None, :] + kk0[:, None, None]) % 254, np.arange(256)[None, :, None]]
    phd_z = lvl[(ii[None, None, :] + kk0[:, None, None] + 1) % 254, np.arange(256)[None, :, None]]
    phd_y = lvl[(ii[None, None, :] + kk0[:, None, None]) % 254, (np.arange(256)[None, :, None] + 1) % 256]
    p["clumpy_diag_ex"] = -(phd[:, :, 1:] - phd[:, :, :-1])
    p["clumpy_diag_ez"] = -(phd_z - phd)[:, :, :256]
    p["clumpy_diag_ey"] = -(phd_y - phd)[:, :, :256]
    p["clumpy_indep_ez"] = ex_of(lv * 2.0 ** -7)[:, :, np.random.default_rng(seed + 9).permutation(256)]
    # The deck's actual S slab (slab index 1 + seed % 6 * 2): sin^2 taper over
    # the slab's 16 planes, E0 = 1 in ez, potential rounded to a 2^-qb grid,
    # m wave periods per box; ex and ez components.
    s0 = 1 + 2 * (seed % 6)
    kk = 16 * s0 + np.arange(16)
    for m, qb in ((1, 22), (4, 23), (2, 22)):
        kw = 2 * np.pi * m / 254
        z3, y3, x3 = np.meshgrid(kk, np.arange(256), np.arange(257), indexing="ij")
        def pot(zz):
            tz = np.sin(np.pi * (zz - 16 * s0 + 0.5) / 16)
            w = np.sin(kw * x3) * np.sin(kw * y3 + 0.7) * np.sin(kw * zz + 1.3)
            q = 2.0 ** -qb
            return np.round(0.3 / kw * tz * tz * w / q) * q
        ph = pot(z3)
        exs = -(ph[:, :, 1:] - ph[:, :, :-1])
        ezs = 1.0 - (pot(z3 + 1) - ph)[:, :, :256]
        p[f"slabS_m{m}_q{qb}_ex"] = exs
        p[f"slabS_m{m}_q{qb}_ez"] = ezs
        if seed == 1:
            print(f"m={m} q=2^-{qb}: max|ex| {np.abs(exs).max():.3f}, ez range [{ezs.min():.3f}, {ezs.max():.3f}]")
    return {k: np.ascontiguousarray(v, dtype=np.float32).reshape(-1) for k, v in p.items()}


if __name__ == "__main__":
    pl.patterns = patterns
    {"gen": pl.gen, "score": pl.score}[sys.argv[1]]()
