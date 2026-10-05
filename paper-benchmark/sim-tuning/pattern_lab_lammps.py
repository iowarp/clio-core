#!/usr/bin/env python3
"""pattern_lab.py for LAMMPS float32 output: which per-atom array chunks make
one codec bad and NeuroPress right?

    pattern_lab_lammps.py gen | score

Each candidate is one 4 MiB chunk as the LAMMPS driver stages it with --f32
--order id: per-atom x, y, z interleaved (x0 y0 z0 x1 ...), atoms in ID order,
which for an fcc lattice built by create_atoms is z layer, then y row, then x,
then the 4 basis atoms. Candidates:
  lat_exact_a2      frozen fcc crystal, lattice constant 2 (coordinates are
                    small integers, exact in float32)
  lat_exact_lj      frozen fcc crystal at the LJ melt density (a = 1.6796)
  lat_cold_s*       that crystal plus Gaussian thermal noise of sigma s
  liquid            uniform random positions in the box
  vel_T*            Maxwell velocities at temperature T (normal noise)
  vel_cold_lattice  velocities of a cold crystal (tiny noise)
Scored as pattern_lab.py does: the training sweep tool on all 45 settings,
the balanced 4-tier cost, and the trained model's prediction and pick.
"""
import sys

import numpy as np

import pattern_lab as pl

pl.LAB = "/mnt/nvme0/tune/lab-lammps"
N_ATOMS = (4 << 20) // 12              # atoms in one 4 MiB chunk of x, y, z
BASIS = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])


def lattice(a, n_cell, start_atom):
    """fcc positions in create_atoms order (z, y, x, basis), N_ATOMS of them."""
    idx = start_atom + np.arange(N_ATOMS)
    m, cell = idx % 4, idx // 4
    i, j, k = cell % n_cell, (cell // n_cell) % n_cell, cell // (n_cell * n_cell)
    return (np.stack([i, j, k], axis=1) + BASIS[m]) * a


def patterns(seed):
    """name -> float32 array, one candidate chunk per pattern."""
    rng = np.random.default_rng(seed)
    n_cell = 70
    start = seed * N_ATOMS
    a_lj = (4 / 0.8442) ** (1 / 3)
    p = {"lat_exact_a2": lattice(2.0, n_cell, start),
         "lat_exact_lj": lattice(a_lj, n_cell, start)}
    for s in (0.003, 0.03, 0.1):
        p[f"lat_cold_s{s:g}"] = lattice(a_lj, n_cell, start) + rng.normal(0, s, (N_ATOMS, 3))
    p["liquid"] = rng.uniform(0, n_cell * a_lj, (N_ATOMS, 3))
    for t in (0.05, 1.0, 6.0):
        p[f"vel_T{t:g}"] = rng.normal(0, np.sqrt(t), (N_ATOMS, 3))
    out = {}
    for k, v in p.items():
        flat = np.ascontiguousarray(v, dtype=np.float32).reshape(-1)
        out[k] = np.concatenate([flat, flat[:1]])          # pad to exactly 4 MiB
    return out


if __name__ == "__main__":
    pl.patterns = patterns
    {"gen": pl.gen, "score": pl.score}[sys.argv[1]]()
