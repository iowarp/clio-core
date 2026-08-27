#!/usr/bin/env python3
"""Render the LAMMPS atom state that the compressor is handed.

    ./run_config.sh dynamic --box 20 --steps 500 --gap 20 --chunk 768000 \
        --require-device --raw /tmp/lmp-raw --results /tmp/lmp-lossless --tag ll
    ./viz_atoms.py --raw /tmp/lmp-raw --out /tmp/lmp-viz

This workload is IN SITU -- LAMMPS runs as a library inside the benchmark
process and no file is ever written -- so there is nothing on disk to look at
unless the run asked for it. `--raw DIR` makes the driver write each staged
blob's bytes, and those bytes are exactly what NeuroPress compressed. This
reads them back: `<field>_step_<N>_chunk_0.bin`, float64, natoms x 3, in atom-ID
order (what `dump h5md` would have written).

ONE CHUNK PER FIELD PER FRAME is assumed, so run with `--chunk` equal to
natoms*3*8 or the files are fragments and will not reshape.

What it draws:

  atoms_montage.png   a thin z-slab of the box, x-y, colored by speed. A slab
                      and not the whole box: 32,000 atoms projected through
                      33 sigma of depth is a uniform smear at every timestep,
                      and the melt is invisible. One layer thick, the fcc rows
                      are obvious at step 0 and gone by the end.
  atoms.gif           the same slab as an animation
  evolution.png       mean-squared displacement, g(r), temperature, the share
                      of cells bit-identical to the previous frame, and a zlib
                      stand-in for the ratio

The zlib curve is a STAND-IN and its magnitudes do not transfer; what carries
over is that on this workload it barely moves at all, which is the finding.
"""
import argparse, glob, os, re, sys
import numpy as np

FIELDS = ["position", "velocity", "force"]


def frames_in(raw_dir, field):
    """-> [(step, path)], in step order. Steps are LAMMPS timesteps, not 0..n."""
    out = []
    for p in glob.glob(os.path.join(raw_dir, f"{field}_step_*_chunk_0.bin")):
        m = re.search(r"_step_(\d+)_chunk_", os.path.basename(p))
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def load(path):
    a = np.fromfile(path, dtype=np.float64)
    if a.size % 3:
        sys.exit(f"{path}: {a.size} doubles is not a multiple of 3 -- was the "
                 f"run chunked at natoms*3*8?")
    return a.reshape(-1, 3)


def box_length(frames):
    """LAMMPS wraps x into the box, so the coordinate span IS the box to within
    one atom spacing. Taken over the whole run, not one frame."""
    lo = min(x.min() for x in frames)
    hi = max(x.max() for x in frames)
    return float(hi - lo), float(lo)


def msd(frames, L):
    """Mean-squared displacement from WRAPPED coordinates.

    Straight x[t]-x[0] is wrong here: an atom that leaves one face re-enters the
    other and reads as a box-sized jump. At step 500 that put a real -0.37
    displacement in at +33.2. So accumulate the minimum-image step-to-step
    displacement instead, which is valid as long as an atom moves less than L/2
    between frames -- true at any sane --gap.
    """
    disp = np.zeros_like(frames[0])
    out = [0.0]
    for a, b in zip(frames, frames[1:]):
        d = b - a
        d -= L * np.round(d / L)
        disp += d
        out.append(float(np.mean(np.sum(disp ** 2, axis=1))))
    return out


def rdf(x, L, nbins=120, rmax=None, nref=1500, seed=0):
    """Radial distribution function g(r), the melting signature: an fcc lattice
    gives sharp peaks at the shell radii, a liquid a broad first peak and
    little beyond it.

    Sampled from `nref` reference atoms rather than all pairs -- 32,000^2 is a
    billion distances and g(r) converges long before that."""
    n = len(x)
    rmax = rmax or L / 2
    rng = np.random.default_rng(seed)
    ref = x[rng.choice(n, size=min(nref, n), replace=False)]
    h = np.zeros(nbins)
    edges = np.linspace(0, rmax, nbins + 1)
    for r0 in ref:
        d = x - r0
        d -= L * np.round(d / L)          # minimum image
        r = np.sqrt(np.sum(d ** 2, axis=1))
        h += np.histogram(r[r > 0], bins=edges)[0]
    shell = 4 / 3 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    return 0.5 * (edges[1:] + edges[:-1]), h / (len(ref) * shell * n / L ** 3)


def ratio(buf, stride):
    """zlib on the raw bytes, and on the same bytes byte-shuffled at `stride`."""
    import zlib
    raw = buf.tobytes()
    if stride > 1:
        raw = np.frombuffer(raw, np.uint8).reshape(-1, stride).T.tobytes()
    return len(buf.tobytes()) / len(zlib.compress(raw, 6))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", required=True, help="directory written by --raw")
    p.add_argument("--out", required=True)
    p.add_argument("--slab", type=float, default=1.2,
                   help="slab thickness in sigma (default 1.2, ~one fcc layer)")
    p.add_argument("--cols", type=int, default=5)
    p.add_argument("--no-ratio", action="store_true")
    a = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    pos = frames_in(a.raw, "position")
    vel = frames_in(a.raw, "velocity")
    if not pos:
        sys.exit(f"no position dumps under {a.raw} -- was the run given --raw?")
    steps = [s for s, _ in pos]
    X = [load(p_) for _, p_ in pos]
    V = [load(p_) for _, p_ in vel] if vel else None
    L, lo = box_length(X)
    n = len(X[0])
    print(f"{n} atoms, {len(X)} frames, steps {steps[0]}..{steps[-1]}, "
          f"box {L:.3f} sigma")

    os.makedirs(a.out, exist_ok=True)

    # A slab one layer thick, in the MIDDLE of the box, held fixed across
    # frames -- atoms enter and leave it, which is itself the diffusion.
    zc = lo + L / 2
    speeds = [np.sqrt(np.sum(v ** 2, axis=1)) for v in V] if V else None
    smax = float(np.percentile(np.concatenate(speeds), 99.5)) if V else 1.0
    norm = Normalize(0, smax)

    def slab(i):
        m = np.abs(X[i][:, 2] - zc) < a.slab / 2
        return X[i][m], (speeds[i][m] if V else None)

    cols = min(a.cols, len(X))
    rows = -(-len(X) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.9 * cols, 3.0 * rows),
                             squeeze=False)
    for ax, i in zip(axes.ravel(), range(len(X))):
        xs, sp = slab(i)
        sc = ax.scatter(xs[:, 0], xs[:, 1], c=(sp if V else "tab:blue"),
                        s=3.2, norm=norm if V else None, cmap="viridis",
                        linewidths=0)
        ax.set(title=f"step {steps[i]}", xlim=(lo, lo + L), ylim=(lo, lo + L),
               xticks=[], yticks=[], aspect="equal")
        ax.title.set_fontsize(7)
    for ax in axes.ravel()[len(X):]:
        ax.axis("off")
    fig.suptitle(f"LJ melt - {a.slab:g} sigma slab at mid-box, "
                 f"{n} atoms, colored by speed", fontsize=11)
    if V:
        fig.colorbar(sc, ax=axes, shrink=0.6, label="|v|")
    out = os.path.join(a.out, "atoms_montage.png")
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"  {out}")

    # GIF: one rendered panel per frame, same limits and same color scale.
    from PIL import Image
    imgs = []
    for i in range(len(X)):
        f1, a1 = plt.subplots(figsize=(3.2, 3.2), dpi=100)
        xs, sp = slab(i)
        a1.scatter(xs[:, 0], xs[:, 1], c=(sp if V else "tab:blue"), s=2.2,
                   norm=norm if V else None, cmap="viridis", linewidths=0)
        a1.set(xlim=(lo, lo + L), ylim=(lo, lo + L), xticks=[], yticks=[],
               aspect="equal", title=f"step {steps[i]}")
        f1.tight_layout(pad=0.3)
        f1.canvas.draw()
        imgs.append(Image.frombytes("RGB", f1.canvas.get_width_height(),
                                    f1.canvas.tostring_rgb()))
        plt.close(f1)
    gif = os.path.join(a.out, "atoms.gif")
    imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=200, loop=0)
    print(f"  {gif}")

    # ---- evolution ----
    m = msd(X, L)
    # Element-level identity SATURATES on this workload: every atom moves every
    # step, so it is 0.00% at every frame and says nothing. The informative
    # version is per BYTE POSITION within the double -- the sign and exponent
    # bytes barely change while the mantissa is new every step, which is both
    # the real picture of how the block evolves and the reason an 8-byte
    # shuffle helps here.
    same = [100.0 * float(np.mean(X[i] == X[i - 1])) for i in range(1, len(X))]
    W = X[0].dtype.itemsize
    bytepos = np.zeros(W)
    for i in range(1, len(X)):
        A = X[i].view(np.uint8).reshape(-1, W)
        B = X[i - 1].view(np.uint8).reshape(-1, W)
        bytepos += (A == B).mean(axis=0)
    bytepos = 100.0 * bytepos / (len(X) - 1)
    same_bytes = [100.0 * float(np.mean(X[i].view(np.uint8) == X[i - 1].view(np.uint8)))
                  for i in range(1, len(X))]
    temp = ([float(np.mean(np.sum(v ** 2, axis=1)) / 3.0) for v in V]
            if V else None)          # LJ units, mass 1: T = <v^2>/3

    ncol = 4 if a.no_ratio else 5
    fig, ax = plt.subplots(1, ncol, figsize=(4.1 * ncol, 3.5))
    ax[0].plot(steps, m, "o-", ms=3)
    ax[0].set(xlabel="timestep", ylabel="MSD (sigma^2)", title="atoms diffuse")
    for i, lab in [(0, "start"), (len(X) // 2, "middle"), (len(X) - 1, "end")]:
        r, g = rdf(X[i], L)
        ax[1].plot(r, g, lw=1.4, label=f"step {steps[i]}")
    ax[1].set(xlabel="r (sigma)", ylabel="g(r)", title="lattice -> liquid")
    ax[1].legend(fontsize=8)
    if temp:
        ax[2].plot(steps, temp, "o-", ms=3, color="tab:red")
        ax[2].set(xlabel="timestep", ylabel="T (LJ units)",
                  title="temperature equilibrates")
    ax[3].bar(np.arange(W), bytepos, color="tab:purple")
    ax[3].plot(steps[:0], [], " ")  # keep the legend handle count honest
    ax[3].set(xlabel=f"byte position in the {W}-byte value  (0 = LSB)",
              ylabel="% identical to previous frame", ylim=(0, 105),
              xticks=np.arange(W),
              title="redundancy is per byte, not per value")
    ax[3].axhline(np.mean(same_bytes), ls="--", lw=1, color="0.4")
    ax[3].annotate(f"whole values: {np.mean(same):.2f}%\n"
                   f"whole bytes:  {np.mean(same_bytes):.1f}% (dashed)",
                   xy=(0.03, 0.86), xycoords="axes fraction", fontsize=8,
                   va="top", family="monospace")
    if not a.no_ratio:
        r0 = [ratio(x, 1) for x in X]
        r8 = [ratio(x, 8) for x in X]
        ax[4].plot(steps, r0, "o-", ms=3, label="no shuffle")
        ax[4].plot(steps, r8, "s-", ms=3, label="8-byte shuffle")
        # Log: step 0 is a perfect lattice at ~7.6x and everything after it
        # sits near 1.1x, so a linear axis spends its whole range on one point.
        ax[4].set(xlabel="timestep", ylabel="zlib ratio, position (stand-in)",
                  title="compressibility collapses, then flatlines", yscale="log")
        ax[4].legend(fontsize=8)
        print(f"position zlib: no-shuffle {r0[0]:.3f}x -> {r0[-1]:.3f}x, "
              f"shuffle-8 {r8[0]:.3f}x -> {r8[-1]:.3f}x")
    for b in ax:
        b.grid(alpha=0.3)
    fig.suptitle("LAMMPS LJ melt: what the compressor is being handed", fontsize=11)
    fig.tight_layout()
    out = os.path.join(a.out, "evolution.png")
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"  {out}")
    print(f"MSD 0 -> {m[-1]:.2f} sigma^2; whole values identical between "
          f"frames {np.mean(same):.2f}%, whole bytes {np.mean(same_bytes):.1f}%")
    print("  per byte position: " +
          "  ".join(f"{k}:{v:.1f}%" for k, v in enumerate(bytepos)))


if __name__ == "__main__":
    main()
