#!/usr/bin/env python3
"""Synthetic dumps: a periodic field, a random one, and their mixture.

  ./gen_synth.py --out DIR [--frames 8] [--shape 128,128,64] [--seed 7]

The four simulation workloads all hand the compressor data that is smooth in
places and structured everywhere, and their per-chunk time therefore says as
much about the physics as about the codec. These three patterns bracket it:

  periodic  a sum of sinusoids -- what a codec does when the data is
            compressible and the ratio is high, so the write is short and the
            selector's fixed cost is the largest share of it
  random    uniform noise -- incompressible, so the codec works hardest and
            stores the most, the opposite end of the same axis
  mixed     periodic + 20% noise -- compressible but not trivially, which is
            where a real field usually sits

Every field is EXACTLY the chunk size by default (128x128x64 float32 = 4 MiB),
so one field is one chunk and the per-chunk numbers need no apportioning.
Frames advance the phase and redraw the noise, so the data evolves the way a
run's does rather than repeating one buffer.

The layout is the one the replay driver reads and the plotters parse:
<out>/plt<NNNNN>/<pattern>.f32, the directory naming the timestep.
"""
import argparse
import os

import numpy as np


def periodic(shape, phase):
    """A smooth, strongly compressible field.

    Three sinusoids at different wavelengths along the three axes: enough
    structure that a codec's entropy coder has something to do, no noise for
    it to fail on.

    @param shape the grid (nx, ny, nz)
    @param phase radians to advance, so successive frames differ
    @return the field, float32
    """
    nx, ny, nz = shape
    x = np.linspace(0, 4 * np.pi, nx, dtype=np.float32)[:, None, None]
    y = np.linspace(0, 6 * np.pi, ny, dtype=np.float32)[None, :, None]
    z = np.linspace(0, 2 * np.pi, nz, dtype=np.float32)[None, None, :]
    return (np.sin(x + phase) * np.cos(y - phase)
            + 0.5 * np.sin(z + 2 * phase)).astype(np.float32)


def write_frame(out, step, shape, phase, rng, patterns):
    """One dump directory: one .f32 per pattern.

    @param out the dump root
    @param step the timestep the directory is named for
    @param shape the grid
    @param phase this frame's phase
    @param rng the generator behind the noise
    @param patterns which of periodic/random/mixed to write
    @return bytes written
    """
    d = os.path.join(out, f"plt{step:05d}")
    os.makedirs(d, exist_ok=True)
    base = periodic(shape, phase)
    noise = rng.random(shape, dtype=np.float32)
    field = {"periodic": base,
             "random": noise,
             "mixed": (base + 0.2 * noise).astype(np.float32)}
    wrote = 0
    for name in patterns:
        path = os.path.join(d, f"{name}.f32")
        field[name].tofile(path)
        wrote += os.path.getsize(path)
    return wrote


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="dump root")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--shape", default="128,128,64",
                    help="nx,ny,nz [128,128,64 = 4 MiB of float32]")
    ap.add_argument("--interval", type=int, default=10,
                    help="timesteps between frames, for the directory names")
    ap.add_argument("--patterns", default="periodic,random,mixed")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    shape = tuple(int(v) for v in a.shape.split(","))
    patterns = [p.strip() for p in a.patterns.split(",") if p.strip()]
    unknown = set(patterns) - {"periodic", "random", "mixed"}
    if unknown:
        raise SystemExit(f"unknown pattern(s): {', '.join(sorted(unknown))}")
    # gen_fields.sh wipes its output directory on entry and so does this, for
    # the same reason: a frame left over from a wider shape replays fine and
    # silently changes the chunk count.
    if os.path.isdir(a.out):
        for root, _dirs, files in os.walk(a.out):
            for f in files:
                if f.endswith(".f32"):
                    os.remove(os.path.join(root, f))
    os.makedirs(a.out, exist_ok=True)

    rng = np.random.default_rng(a.seed)
    total = 0
    for i in range(a.frames):
        total += write_frame(a.out, i * a.interval, shape,
                             i * np.pi / a.frames, rng, patterns)
    print(f"== synthetic: {a.frames} frame(s) x {len(patterns)} pattern(s) "
          f"({', '.join(patterns)}), {int(np.prod(shape))} values of float32 "
          f"= {int(np.prod(shape)) * 4 / 2**20:.2f} MiB each, "
          f"{total / 2**20:.1f} MiB total -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
