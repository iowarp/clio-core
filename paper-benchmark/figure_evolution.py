#!/usr/bin/env python3
"""One field at the beginning, middle and last frame, side by side.

    ./figure_evolution.py --source f32     --dir DUMPS   --out FIG.png
    ./figure_evolution.py --source openpmd --dir DIAGS   --out FIG.png --field Ez
    ./figure_evolution.py --source raw     --dir RAW --atoms --out FIG.png

Section 5 of the benchmark spec: show that the workload's data actually
evolves. Three frames, ONE field, and identical settings across the panels --
which is the whole point, so the difference you see is the data and not the
rendering.

THE COLOR SCALE IS SHARED ACROSS THE THREE PANELS, computed once over all
three frames together. Per-panel autoscaling is the standard way to make a
static field look like it is evolving: matplotlib will happily stretch noise
to full contrast. A shared scale means a panel that looks empty IS empty
relative to the others.

FRAME READING IS evolution.py's, not a second implementation, so the pictures
and the numbers in evolution.csv come from exactly the same bytes and the same
source handling -- including the h5dump path for openPMD and the chunk-order
concatenation for raw.

A VECTOR COMPONENT IS ANTISYMMETRIC ABOUT THE MID-PLANE NORMAL TO ITS OWN
AXIS, so slicing z-momentum on the z mid-plane gives ~0 everywhere and a blank
panel from perfectly healthy data. Measured on Nyx at 128^3: zmom reaches 4.24
globally and 0.048 on the z mid-plane. --axis picks another plane; the
per-workload defaults below already avoid the trap.
"""
import argparse, importlib.util as iu, os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

_here = os.path.dirname(os.path.abspath(__file__))
_spec = iu.spec_from_file_location("ev", os.path.join(_here, "evolution.py"))
ev = iu.module_from_spec(_spec); _spec.loader.exec_module(ev)

# A field that actually shows the physics, per workload. Not the first
# alphabetically, which is how you end up plotting div_b_err.
DEFAULT_FIELD = {"warpx": "Ez", "vpic": "cby", "nyx": "density", "lammps": "position"}


def slice_of(flat, axis, shape=None):
    """1-D field -> a mid-plane 2-D slice of the volume it came from.

    A CUBE IS AN ASSUMPTION, NOT A FACT, and getting it wrong does not raise:
    WarpX's 64x64x512 grid has 2,097,152 cells, which is exactly 128^3, so the
    cube-root test PASSES and reshapes a slab into a cube. The geometry is then
    nonsense and the mid-plane happens to land on zeros -- a blank panel from a
    field reaching 3.4e11. Pass --shape for any grid that is not cubic.
    """
    if shape:
        if int(np.prod(shape)) != len(flat):
            return None
        v = flat.reshape(*shape, order="F")
    else:
        n = round(len(flat) ** (1 / 3))
        if n ** 3 != len(flat):
            return None
        # order="F" for the same reason viz_fields.py uses it: these dumps are
        # written in Fortran order and a C-order reshape silently transposes
        # the picture rather than failing.
        v = flat.reshape(n, n, n, order="F")
    i, j, k = (d // 2 for d in v.shape)
    return {"x": v[i, :, :], "y": v[:, j, :], "z": v[:, :, k]}[axis], v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=sorted(ev.SOURCES))
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--field", default="")
    ap.add_argument("--workload", default="", help="only to pick a default field")
    ap.add_argument("--axis", default="z", choices=["x", "y", "z"])
    ap.add_argument("--shape", default="",
                    help="NX,NY,NZ for a non-cubic grid, e.g. 64,64,512")
    ap.add_argument("--atoms", action="store_true",
                    help="rows are atom xyz, not a field cube: scatter instead")
    ap.add_argument("--f64", action="store_true")
    ap.add_argument("--step-scale", type=int, default=1)
    a = ap.parse_args()

    field = a.field or DEFAULT_FIELD.get(a.workload, "")
    dtype = np.float64 if a.f64 else np.float32
    # THE openPMD READER TAKES A DATASET PATH, NOT THE NAME IT YIELDS: it looks
    # up /data/<step>/fields/E/z and then hands the field back as "Ez". Asking
    # for "Ez" therefore matches nothing and the reader returns zero frames
    # rather than an error. Translate Ez -> E/z; anything else (rho) passes
    # through unchanged.
    want = [field] if field else None
    if a.source == "openpmd" and field:
        m = re.fullmatch(r"([EBj])([xyz])", field)
        want = [f"{m[1]}/{m[2]}"] if m else [field]
    frames = list(ev.SOURCES[a.source](a.dir, dtype, want))
    if len(frames) < 2:
        sys.exit(f"{a.dir}: need at least 2 frames, found {len(frames)}")

    pick = [0, len(frames) // 2, len(frames) - 1]
    labels = ["beginning", "middle", "last"]
    steps, arrays = [], []
    for i in pick:
        step, fields = frames[i]
        if field not in fields:
            sys.exit(f"field {field!r} not in {sorted(fields)}")
        steps.append(step * a.step_scale)
        arrays.append(fields[field].astype(np.float64))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))

    if a.atoms:
        # natoms x 3. Project onto x-y and colour by z so depth is visible;
        # the axis limits are shared so drift and expansion are readable.
        pts = [v.reshape(-1, 3) for v in arrays]
        lim = [(min(p[:, k].min() for p in pts), max(p[:, k].max() for p in pts))
               for k in range(3)]
        for ax, p, lab, st in zip(axes, pts, labels, steps):
            ax.scatter(p[:, 0], p[:, 1], c=p[:, 2], s=0.4, cmap="viridis",
                       vmin=lim[2][0], vmax=lim[2][1], linewidths=0)
            ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1]); ax.set_aspect("equal")
            ax.set_title(f"{lab} — step {st}")
            ax.set_xticks([]); ax.set_yticks([])
    else:
        shape = tuple(int(x) for x in a.shape.split(",")) if a.shape else None
        got = [slice_of(v, a.axis, shape) for v in arrays]
        if any(g is None for g in got):
            sys.exit(f"{a.dir}: {len(arrays[0])} elements does not match "
                     f"{'shape ' + a.shape if shape else 'a cube'}; "
                     f"use --shape, or --atoms if these are atom rows")
        slices = [g[0] for g in got]
        vols = [g[1] for g in got]
        # REFUSE TO WRITE A BLANK PLATE. A slice that is identically zero while
        # the volume is not means the plane or the shape is wrong, not that the
        # data is static -- a vector component is antisymmetric about the
        # mid-plane normal to its own axis, and a mis-shaped reshape lands
        # anywhere. Either way the picture would be a lie, so stop.
        if (max(abs(s).max() for s in slices) == 0
                and max(abs(v).max() for v in vols) > 0):
            sys.exit(f"{field}: identically zero on the {a.axis} mid-plane while "
                     f"the volume reaches {max(abs(v).max() for v in vols):.4g}. "
                     f"Wrong plane or wrong --shape; refusing to write a blank figure.")
        both = np.concatenate([s.ravel() for s in slices])
        finite = both[np.isfinite(both)]
        lo, hi = np.percentile(finite, [1, 99]) if finite.size else (0, 1)
        if lo == hi:
            lo, hi = (lo - 1e-12, hi + 1e-12)
        # Log scale only when the data is strictly positive and spans enough
        # decades to need it -- Sedov density does, a signed field cannot.
        if finite.min() > 0 and hi / max(lo, 1e-300) > 50:
            norm, cmap = LogNorm(vmin=max(lo, finite.min()), vmax=hi), "inferno"
        else:
            m = max(abs(lo), abs(hi))
            if finite.min() < 0:
                norm, cmap = Normalize(vmin=-m, vmax=m), "RdBu_r"
            else:
                norm, cmap = Normalize(vmin=lo, vmax=hi), "inferno"
        for ax, s, lab, st in zip(axes, slices, labels, steps):
            im = ax.imshow(s.T, origin="lower", norm=norm, cmap=cmap)
            ax.set_title(f"{lab} — step {st}")
            ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)

    fig.suptitle(f"{a.workload or a.source}: {field} — "
                 f"{a.axis} mid-plane, one shared color scale"
                 if not a.atoms else
                 f"{a.workload or a.source}: {field} — x–y projection, colour = z",
                 y=0.99)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=130, bbox_inches="tight")
    print(f"{a.out}  ({field}, steps {steps[0]}/{steps[1]}/{steps[2]})")


if __name__ == "__main__":
    main()
