#!/usr/bin/env python3
"""Render the flat field dumps a compression sweep replays.

    ./viz_fields.py --fields /tmp/nyx-quick --out /tmp/nyx-viz
    ./viz_fields.py --fields /tmp/vpic-quick --out /tmp/vpic-viz --field ex

Shared by the nyx and vpic benchmarks, which dump the same way: one directory
per frame, one flat cube per component, named
<frame>/fab%04d_comp%02d_<field>.<ext>. Nothing here is Nyx-specific except the
two blast-wave panels, which turn themselves off when the first frame is not a
uniform background (see below).

The benchmark's two phases (gen_fields.sh -> run_sweep.sh) exchange flat
float32 files, and those files -- not Nyx's AMReX plotfiles -- are what the
compressor actually sees.  So this reads them directly: no yt, no VisIt, no
AMReX, just numpy.  A dump is a bare ncell^3 C-ordered float32 array, so the
side length is recovered from the file size and there is nothing to parse.

What it produces, per field:

  <field>_montage.png   mid-plane slices across the whole run, one panel per
                        dump, on a color scale shared by every panel so the
                        blast really is expanding rather than the colorbar
                        moving under it
  <field>.gif           the same slices as an animation
  evolution.png         the physics and the compressibility on one time axis:
                        shock radius, fraction of the domain disturbed, and a
                        zlib stand-in for the ratio the sweep measures

The ratio curve is a STAND-IN.  zlib is not nvcomp-zstd and these numbers will
not match results/; what carries over is the shape -- ratio falling as the
shock fills the box -- and the shuffle gap, which is the benchmark's headline
result reproduced here in three lines of python.

Sim times come from the AMReX plotfiles if --plt points at them.  gen_fields.sh
writes those into a mktemp working directory and deletes it, so by default
there are none and frames are labelled by dump index instead.
"""
import argparse, glob, os, sys, zlib
import numpy as np

# Nyx's hydro state. VPIC names its own (ex, cbz, jfz, ...) and passes --field;
# nothing below assumes this list.
FIELDS = ["density", "xmom", "ymom", "zmom", "rho_E", "rho_e"]


def load(path):
    """A dump is a bare cube of float32; the side comes from the file size.

    FORTRAN ORDER, and it matters. The patch packs each component with AMReX's
    copyToMem, which walks the box in the FAB's own layout -- x fastest. Read
    it C-ordered and the axes come back reversed: harmless for density, which
    is spherically symmetric, but it silently slices xmom along x, where the
    field is antisymmetric and the midplane is ~0. The frames look empty and
    nothing says why.
    """
    a = np.fromfile(path, dtype=np.float32)
    n = round(len(a) ** (1 / 3))
    if n ** 3 != len(a):
        sys.exit(f"{path}: {len(a)} elements is not a cube")
    return a.reshape(n, n, n, order="F")


def frames_for(fields_dir, field, ext=".f32"):
    """Dumps are <fields>/plt%05d/fab0000_comp%02d_<field>.f32, one dir a dump."""
    pat = os.path.join(fields_dir, "plt*", f"fab0000_comp*_{field}{ext}")
    return sorted(glob.glob(pat))


def plot_times(plt_dir):
    """AMReX plotfile Header: line 2 is the variable count, then time follows
    the names and the dimension. Absent -> frames get indices, not times."""
    if not plt_dir:
        return None
    out = []
    for d in sorted(glob.glob(os.path.join(plt_dir, "plt*"))):
        try:
            L = open(os.path.join(d, "Header")).read().split("\n")
            out.append(float(L[2 + int(L[1]) + 1]))
        except (OSError, ValueError, IndexError):
            return None
    return out or None


def ratio(buf, stride):
    """zlib on the raw bytes, and on the same bytes byte-shuffled at `stride`
    -- the transform NeuroPress encodes as a single bit meaning 4."""
    raw = buf.tobytes()
    if stride > 1:
        raw = np.frombuffer(raw, np.uint8).reshape(-1, stride).T.tobytes()
    return len(buf.tobytes()) / len(zlib.compress(raw, 6))


def shock_radius(vol, quiet):
    """Radius enclosing everything that has moved off the ambient state. The
    Sedov front is a thin shell, so the max disturbed radius IS the front."""
    n = vol.shape[0]
    moved = np.abs(vol - quiet) > 1e-3 * max(abs(quiet), 1e-30)
    if not moved.any():
        return 0.0
    c = (n - 1) / 2
    ax = (np.arange(n) - c) / n           # box is the unit cube
    r = np.sqrt(sum(g ** 2 for g in np.meshgrid(ax, ax, ax, indexing="ij")))
    return float(r[moved].max())


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fields", required=True, help="dump directory (plt*/ inside)")
    p.add_argument("--out", required=True, help="where the PNGs and GIFs go")
    p.add_argument("--field", action="append", help="repeatable; default density")
    p.add_argument("--ext", default=".f32", help="dump extension [.f32]")
    p.add_argument("--axis", default="z", choices=["x", "y", "z"],
                   help="mid-plane to slice [z]. A vector component is "
                        "antisymmetric about the plane normal to its own axis, "
                        "so zmom on the z mid-plane renders blank -- give that "
                        "field its own plane with --axis-for x:zmom.")
    p.add_argument("--axis-for", action="append", default=[], metavar="AXIS:FIELD",
                   help="repeatable per-field slice axis, e.g. x:zmom")
    p.add_argument("--evolve-field", help="field for evolution.png "
                                          "[first --field, or density]")
    p.add_argument("--plt", help="AMReX plotfile dir, for real sim times")
    p.add_argument("--no-ratio", action="store_true", help="skip the zlib curve")
    a = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize

    fields = a.field or ["density"]
    # --axis-for x:zmom -> {"zmom": "x"}
    axis_for = {}
    for spec in a.axis_for:
        ax_, _, fld = spec.partition(":")
        if not fld or ax_ not in ("x", "y", "z"):
            sys.exit(f"--axis-for wants AXIS:FIELD with AXIS in x/y/z, got {spec!r}")
        axis_for[fld] = ax_
    os.makedirs(a.out, exist_ok=True)
    times = plot_times(a.plt)

    for field in fields:
        files = frames_for(a.fields, field, a.ext)
        if not files:
            sys.exit(f"no {field} dumps under {a.fields}")
        vols = [load(f) for f in files]
        n = vols[0].shape[0]
        lab = ([f"t={t:.4g}" for t in times] if times and len(times) == len(vols)
               else [f"dump {i}" for i in range(len(vols))])
        print(f"{field}: {len(vols)} dumps at {n}^3")

        # One color scale for the whole run, so the blast really is expanding
        # rather than the colorbar moving under it. Density spans three decades
        # once the blast has evacuated the centre, so log; momenta are signed,
        # so diverging about zero.
        #
        # Signed fields are scaled to a PERCENTILE, not the max. The initial
        # energy deposit puts ~100 into a couple of cells while the shell that
        # matters carries ~10, and scaling to that spike renders every later
        # frame blank.
        pool = np.concatenate([v.ravel()[::7] for v in vols])
        lo = min(float(v.min()) for v in vols)
        hi = max(float(v.max()) for v in vols)
        if lo > 0 and hi / lo > 50:
            norm, cmap = LogNorm(vmin=lo, vmax=hi), "inferno"
        elif lo < 0:
            m = float(np.percentile(np.abs(pool), 99.9)) or max(abs(lo), hi)
            norm, cmap = Normalize(-m, m), "RdBu_r"
        else:
            norm, cmap = Normalize(lo, float(np.percentile(pool, 99.9))), "inferno"

        # WHICH PLANE TO CUT, and it is not cosmetic for a vector component.
        # A component of a spherically symmetric outflow is ANTISYMMETRIC about
        # the mid-plane normal to its own axis: zmom on the z mid-plane is ~0
        # everywhere by symmetry, and the montage comes out blank while the
        # field itself is perfectly healthy. Measured on the Sedov run at
        # 128^3: zmom reaches 4.24 globally and 0.048 on the z mid-plane,
        # against xmom's 4.23 on that same plane. Slice zmom along x or y
        # instead, with --axis-for, so one invocation can still do all six.
        axis = axis_for.get(field, a.axis)
        mid = n // 2
        if axis == "x":
            slices = [v[mid, :, :] for v in vols]
        elif axis == "y":
            slices = [v[:, mid, :] for v in vols]
        else:
            slices = [v[:, :, mid] for v in vols]

        cols = min(7, len(slices))
        rows = -(-len(slices) // cols)
        fig, axes = plt.subplots(rows, cols, figsize=(2.0 * cols, 2.1 * rows))
        for ax, sl, t in zip(np.atleast_1d(axes).ravel(), slices, lab):
            im = ax.imshow(sl.T, origin="lower", norm=norm, cmap=cmap)
            ax.set_title(t, fontsize=7); ax.set_xticks([]); ax.set_yticks([])
        for ax in np.atleast_1d(axes).ravel()[len(slices):]:
            ax.axis("off")
        fig.suptitle(f"{field}, {axis}={mid} slice, {n}^3", fontsize=11)
        fig.colorbar(im, ax=axes, shrink=0.6, label=field)
        out = os.path.join(a.out, f"{field}_montage.png")
        fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
        print(f"  {out}")

        # GIF, via the same norm+cmap so it matches the montage panel for panel.
        from PIL import Image
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        imgs = []
        for sl, t in zip(slices, lab):
            rgb = (sm.to_rgba(sl.T[::-1]) * 255).astype(np.uint8)[..., :3]
            img = Image.fromarray(rgb).resize((n * 4, n * 4), Image.NEAREST)
            imgs.append(img)
        gif = os.path.join(a.out, f"{field}.gif")
        imgs[0].save(gif, save_all=True, append_images=imgs[1:],
                     duration=200, loop=0)
        print(f"  {gif}")

    # The physics and the compressibility share a time axis, which is the whole
    # point on a blast wave: the ratio falls because the shock fills the box.
    ev = a.evolve_field or (fields[0] if fields else "density")
    vols = [load(f) for f in frames_for(a.fields, ev, a.ext)]
    if not vols:
        sys.exit(f"no {ev} dumps under {a.fields} -- pass --evolve-field")
    # The blast-wave panels need a QUIET BACKGROUND to measure against: shock
    # radius and "% off ambient" are both defined relative to the undisturbed
    # value. Sedov starts as a uniform box and has one; a VPIC Weibel run
    # starts already structured and does not, so those two panels would report
    # "100% disturbed at every frame" and mean nothing. Detect it rather than
    # asking the caller which workload this is.
    v0 = vols[0]
    sedov = bool(v0.min() == v0.max())
    quiet = float(v0.flat[0])
    x = times[:len(vols)] if times and len(times) >= len(vols) else list(range(len(vols)))
    xl = "sim time" if times and len(times) >= len(vols) else "dump"
    rad = [shock_radius(v, quiet) for v in vols]
    touched = [100.0 * float(np.mean(np.abs(v - quiet) > 1e-3)) for v in vols]
    # Temporal redundancy: cells BIT-identical to the previous dump. Dump 0 has
    # no predecessor, so the series starts at 1.
    same = [100.0 * float(np.mean(vols[i] == vols[i - 1])) for i in range(1, len(vols))]

    panels = ["range", "same"] + ([] if a.no_ratio else ["ratio"])
    if sedov:
        panels = ["radius", "touched"] + panels[1:]   # drop "range" for those two
    fig, ax = plt.subplots(1, len(panels), figsize=(4.3 * len(panels), 3.6))
    P = dict(zip(panels, ax))
    if "radius" in P:
        P["radius"].plot(x, rad, "o-", ms=3)
        P["radius"].set(xlabel=xl, ylabel="shock radius (box units)",
                        title="blast expands")
        P["touched"].plot(x, touched, "o-", ms=3, color="tab:red")
        P["touched"].set(xlabel=xl, ylabel="% of cells off ambient",
                         title="domain disturbed")
    if "range" in P:
        P["range"].plot(x, [float(v.max()) for v in vols], "o-", ms=3, label="max")
        P["range"].plot(x, [float(v.min()) for v in vols], "o-", ms=3, label="min")
        P["range"].set(xlabel=xl, ylabel=ev, title="amplitude grows")
        P["range"].legend(fontsize=8)
    P["same"].plot(x[1:], same, "o-", ms=3, color="tab:purple")
    P["same"].set(xlabel=xl, ylabel="% bit-identical to previous dump",
                  title="temporal redundancy", ylim=(0, 105))
    if "ratio" in P:
        r0 = [ratio(v, 1) for v in vols]
        r4 = [ratio(v, 4) for v in vols]
        P["ratio"].plot(x, r0, "o-", ms=3, label="no shuffle")
        P["ratio"].plot(x, r4, "s-", ms=3, label="4-byte shuffle")
        P["ratio"].set(xlabel=xl, ylabel="zlib ratio (stand-in)", yscale="log",
                       title="compressibility")
        P["ratio"].legend(fontsize=8)
        print(f"{ev} zlib: no-shuffle {r0[0]:.2f}x -> {r0[-1]:.2f}x, "
              f"shuffle-4 {r4[0]:.2f}x -> {r4[-1]:.2f}x")
    print(f"{ev} bit-identical to previous dump: {same[0]:.2f}% -> {same[-1]:.2f}%")
    for b in ax:
        b.grid(alpha=0.3)
    fig.suptitle(f"{ev}: what the compressor is being handed", fontsize=11)
    fig.tight_layout()
    out = os.path.join(a.out, "evolution.png")
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"  {out}")


if __name__ == "__main__":
    main()
