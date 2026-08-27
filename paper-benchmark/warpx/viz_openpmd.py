#!/usr/bin/env python3
"""Render the openPMD fields a WarpX run wrote, alongside what Clio stored.

    ./run_config.sh dynamic --ncell "64 64 512" --steps 40 --interval 10 \
        --stage-h2d --results /tmp/wx-lossless --tag ll
    ./viz_openpmd.py --run /tmp/wx-lossless/ll --out /tmp/wx-viz

This workload is IN SITU and needs no dump flag: WarpX writes openPMD-HDF5
exactly as it always does and the VOL compresses on the way past, so the native
.h5 files are sitting in <run>/run/diags/diag1/ afterwards. They are the same
bytes the compressor saw.

NO h5py REQUIRED. Datasets are extracted with the h5dump CLI
(`-b LE -o file`), which ships with HDF5 and is therefore already present
anywhere WarpX built against it. Slower than h5py and worth replacing if h5py
turns up, but it removes the dependency entirely.

The laser-wakefield run is a pulse travelling along z, so slices are taken in
the x-z plane at mid-y: the pulse and the wake behind it are visible, and both
move across the frames.
"""
import argparse, glob, os, re, subprocess, sys, tempfile
import numpy as np

# WarpX writes SI. These span fifteen orders of magnitude in a single run,
# which is the whole point of the plate this produces.
FIELDS = [("E/z", "Ez"), ("E/x", "Ex"), ("B/y", "By"), ("B/x", "Bx"),
          ("j/z", "jz"), ("rho", "rho")]


def steps_in(run_dir):
    """-> [(step, h5path)] in step order, from openpmd_%06d.h5."""
    out = []
    for p in glob.glob(os.path.join(run_dir, "run", "diags", "diag1", "*.h5")):
        m = re.search(r"(\d+)\.h5$", os.path.basename(p))
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def read_field(h5, step, dset, cache):
    """One dataset as a numpy array, via h5dump -b LE. Shape comes from h5ls."""
    key = (h5, dset)
    if key in cache:
        return cache[key]
    path = f"/data/{step}/fields/{dset}"
    shp = subprocess.run(["h5ls", "-r", f"{h5}/{path}"], capture_output=True,
                         text=True).stdout
    m = re.search(r"\{([\d,\s]+)\}", shp)
    if not m:
        return None
    shape = tuple(int(x) for x in m.group(1).split(","))
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tf:
        tmp = tf.name
    try:
        r = subprocess.run(["h5dump", "-d", path, "-b", "LE", "-o", tmp, h5],
                           capture_output=True)
        if r.returncode != 0 or not os.path.getsize(tmp):
            return None
        a = np.fromfile(tmp, dtype=np.float32)
        if a.size != int(np.prod(shape)):
            return None
        a = a.reshape(shape)
    finally:
        os.unlink(tmp)
    cache[key] = a
    return a


def ratio(buf, stride):
    import zlib
    raw = buf.tobytes()
    if stride > 1:
        raw = np.frombuffer(raw, np.uint8).reshape(-1, stride).T.tobytes()
    return len(buf.tobytes()) / len(zlib.compress(raw, 6))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="a run_config.sh store")
    p.add_argument("--out", required=True)
    p.add_argument("--field", action="append", help="repeatable; openPMD path e.g. E/z")
    p.add_argument("--cols", type=int, default=5)
    a = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    steps = steps_in(a.run)
    if not steps:
        sys.exit(f"no openPMD .h5 under {a.run}/run/diags/diag1")
    fields = [(f, f.replace("/", "")) for f in a.field] if a.field else FIELDS
    os.makedirs(a.out, exist_ok=True)
    cache = {}
    print(f"{len(steps)} diagnostics, steps {steps[0][0]}..{steps[-1][0]}")

    summary = []
    for dset, name in fields:
        vols = []
        for st, h5 in steps:
            v = read_field(h5, st, dset, cache)
            if v is None:
                print(f"  {dset}: not readable, skipping"); vols = []; break
            vols.append(v)
        if not vols:
            continue
        # A symmetric percentile scale: these fields are signed and the pulse is
        # orders of magnitude above the background, so a max-based scale renders
        # everything except a handful of cells as zero.
        pool = np.concatenate([v.ravel()[::37] for v in vols])
        m = float(np.percentile(np.abs(pool), 99.9)) or float(np.abs(pool).max()) or 1.0
        norm = Normalize(-m, m)
        ny = vols[0].shape[1] // 2
        # openPMD writes (z, y, x). Slice at mid-y and leave z on the VERTICAL
        # axis: the box is 512 x 64, so laying z across the page gives panels
        # eight times wider than they are tall and the pulse is a smear.
        sl = [v[:, ny, :] for v in vols]

        cols = min(a.cols, len(sl)); rows = -(-len(sl) // cols)
        fig, axes = plt.subplots(rows, cols, figsize=(1.7 * cols, 4.4 * rows),
                                 squeeze=False)
        for ax, s, (st, _) in zip(axes.ravel(), sl, steps):
            im = ax.imshow(s, origin="lower", norm=norm, cmap="RdBu_r",
                           aspect="auto", interpolation="nearest")
            ax.set(title=f"step {st}", xticks=[], yticks=[])
            ax.title.set_fontsize(8)
        for ax in axes.ravel()[len(sl):]:
            ax.axis("off")
        fig.suptitle(f"WarpX laser wakefield - {dset}, x-z slice at mid-y "
                     f"(z upward), |scale| {m:.3g}", fontsize=11, y=1.01)
        fig.colorbar(im, ax=axes, shrink=0.7, label=dset)
        out = os.path.join(a.out, f"{name}_montage.png")
        fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
        print(f"  {out}")

        rng = float(max(v.max() for v in vols) - min(v.min() for v in vols))
        summary.append((dset, rng, [float(np.abs(v).max()) for v in vols],
                        [ratio(v, 1) for v in vols], [ratio(v, 4) for v in vols],
                        [100.0 * float(np.mean(vols[i] == vols[i - 1]))
                         for i in range(1, len(vols))]))

    if not summary:
        sys.exit("nothing rendered")
    x = [s for s, _ in steps]
    fig, ax = plt.subplots(1, 4, figsize=(18, 3.8))
    for dset, rng, amp, r0, r4, same in summary:
        ax[0].plot(x, amp, "o-", ms=3, label=dset)
        ax[1].plot(x, r0, "o-", ms=3, label=dset)
        ax[2].plot(x[1:], same, "o-", ms=3, label=dset)
    ax[0].set(xlabel="timestep", ylabel="max |field| (SI)", yscale="log",
              title="fifteen orders of magnitude apart")
    ax[1].set(xlabel="timestep", ylabel="zlib ratio (stand-in)", yscale="log",
              title="compressibility per field")
    ax[2].set(xlabel="timestep", ylabel="% bit-identical to previous diag",
              ylim=(0, 105), title="temporal redundancy")
    # The number that decides whether an absolute bound is meaningful for a
    # field: the bound against that field's own range.
    names = [s[0] for s in summary]
    y = np.arange(len(names))
    for k, eb in enumerate([0.001, 0.01, 0.1]):
        ax[3].barh(y + (k - 1) * 0.26, [100 * eb / s[1] for s in summary],
                   height=0.24, label=f"eb {eb:g}")
    ax[3].axvline(100, ls="--", lw=1, color="0.3")
    ax[3].set(yticks=y, yticklabels=names, xscale="log",
              xlabel="bound as % of the field's own range",
              title="dashed: bound exceeds the whole field")
    for b in ax:
        b.grid(alpha=0.3); b.legend(fontsize=7)
    fig.suptitle("WarpX: what the compressor is being handed", fontsize=12)
    fig.tight_layout()
    out = os.path.join(a.out, "evolution.png")
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"  {out}")
    for dset, rng, amp, r0, r4, same in summary:
        print(f"{dset:<6} range {rng:.3e}  eb=0.01 is {100*0.01/rng:.3g}% of it")


if __name__ == "__main__":
    main()
