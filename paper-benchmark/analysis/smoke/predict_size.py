#!/usr/bin/env python3
"""How much data will a benchmark cell produce? Answer it BEFORE submitting.

Every workload's payload is a closed form -- frames x fields x bytes-per-field
-- and nothing about the run is stochastic, so the size is known from the
parameters alone. What varies is only which of `steps`, the dump interval and
the grid you choose to move.

  payload = frames x n_fields x (cells x element_size)

The +1 on some frame counts is not cosmetic. Nyx, LAMMPS and WarpX dump at
step 0 as well as every interval after it; VPIC deliberately does NOT, because
at step 0 its field array is still identically zero and a frame of pure zeros
would compress ~infinitely and dominate any averaged ratio.

  ./predict_size.py                       # the settings this repo ships
  ./predict_size.py --target 30           # what steps reach 30 GiB
"""
import argparse

GiB = 1024 ** 3


def nyx(ncell=128, steps=200, interval=10):
    frames = steps // interval + 1          # step 0 included
    return frames, 6, ncell ** 3 * 4        # density xmom ymom zmom rho_E rho_e


def vpic(ncell=126, steps=200, interval=25):
    frames = steps // interval              # step 0 SKIPPED, see module docstring
    return frames, 16, (ncell + 2) ** 3 * 4  # ghost layer -> (N+2)^3


def lammps(box=40, steps=200, interval=50):
    frames = steps // interval + 1
    natoms = 4 * box ** 3                   # fcc lattice
    return frames, 3, natoms * 3 * 8        # x/v/f, 3 components, float64


def warpx(nx=64, ny=64, nz=512, steps=40, interval=10):
    frames = steps // interval + 1
    return frames, 10, nx * ny * nz * 4     # B{xyz} E{xyz} j{xyz} rho


W = {"nyx": nyx, "vpic": vpic, "lammps": lammps, "warpx": warpx}


def size(w, **kw):
    frames, nfields, per = W[w](**kw)
    return frames, nfields, per, frames * nfields * per


def steps_for(w, target_gib, interval, **kw):
    """Smallest step count reaching the target, holding the grid fixed."""
    _, nfields, per, _ = size(w, steps=interval, interval=interval, **kw)
    frames = -(-int(target_gib * GiB) // (nfields * per))
    return frames * interval


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, help="GiB per cell to solve for")
    a = ap.parse_args()

    print(f"{'workload':8}{'frames':>8}{'fields':>8}{'per field':>12}"
          f"{'payload':>12}   parameters")
    shipped = {
        "nyx":    dict(ncell=128, steps=200, interval=10),
        "vpic":   dict(ncell=126, steps=200, interval=25),
        "lammps": dict(box=40, steps=200, interval=50),
        "warpx":  dict(nx=64, ny=64, nz=512, steps=40, interval=10),
    }
    for w, kw in shipped.items():
        fr, nf, per, tot = size(w, **kw)
        print(f"{w:8}{fr:8d}{nf:8d}{per/1024**2:9.1f} MiB{tot/GiB:9.2f} GiB   {kw}")

    if a.target:
        print(f"\nsteps needed for {a.target} GiB per cell, grid unchanged:")
        for w, kw in shipped.items():
            iv = kw["interval"]
            grid = {k: v for k, v in kw.items() if k not in ("steps", "interval")}
            s = steps_for(w, a.target, iv, **grid)
            fr, nf, per, tot = size(w, steps=s, interval=iv, **grid)
            print(f"   {w:8} --steps {s:<7} (interval {iv})  -> {fr:5d} frames,"
                  f" {tot/GiB:6.2f} GiB, {tot//(8*1024**2):5d} chunks @8MiB")
