"""A Gray-Scott reaction-diffusion campaign: the real data behind every workload phase.

    python3 gs_campaign.py [out_dir]        (once; ~2-3 min; cached)

Simulates U/V on a 512x512 periodic grid (F=0.035, k=0.065, Du=0.16, Dv=0.08, dt=1) from a
uniform state with seeded perturbations, and keeps what a real campaign produces:
    snapshots.npy  64 snapshots (every 200 steps) of U and V, float64, 4 MB each
    log.txt        the simulation's own per-step log: one line per rank of a 4x4 decomposition
                   (step, subdomain statistics, step time)
    events.bin     spot-detection records every 20 steps (step, id, x, y, peak V, area), 32 B each
    products.bin   the snapshots compressed with zstd (already-compressed data products)

The fields start smooth and nearly uniform and develop spot/stripe patterns as the campaign
runs, so the same output stream changes in compressibility over the campaign.
"""

import os
import struct
import sys
import time

import numpy as np

N, F, K, DU, DV, DT = 512, 0.035, 0.065, 0.16, 0.08, 1.0
N_SNAP, EVERY = 64, 200
DETECT_EVERY = 20         # spot detection cadence (steps)
RANKS = 4                 # the log reports per-rank statistics of a RANKS x RANKS decomposition
DEFAULT_DIR = os.environ.get("GS_CAMPAIGN", "/ssc-bench/gs_campaign")


def lap(a):
    return np.roll(a, 1, 0) + np.roll(a, -1, 0) + np.roll(a, 1, 1) + np.roll(a, -1, 1) - 4 * a


def spots(v, step, thresh=0.2):
    """Spot detection: connected regions of V above thresh, by a simple flood over a coarse grid."""
    mask = v > thresh
    coarse = mask.reshape(N // 8, 8, N // 8, 8).any(axis=(1, 3))
    seen = np.zeros_like(coarse)
    out = []
    for i, j in zip(*np.nonzero(coarse)):
        if seen[i, j]:
            continue
        stack, cells = [(i, j)], []
        seen[i, j] = True
        while stack:
            a, b = stack.pop()
            cells.append((a, b))
            for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                x, y = (a + da) % coarse.shape[0], (b + db) % coarse.shape[1]
                if coarse[x, y] and not seen[x, y]:
                    seen[x, y] = True
                    stack.append((x, y))
        ys, xs = zip(*cells)
        cy, cx = float(np.mean(ys)) * 8 + 4, float(np.mean(xs)) * 8 + 4
        peak = float(v[int(cy) % N, int(cx) % N])
        out.append(struct.pack("<IIffffQ", step, len(out), cx, cy, peak, float(len(cells) * 64), 0))
    return out


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(7)
    u, v = np.ones((N, N)), np.zeros((N, N))
    for _ in range(24):                                   # seeded perturbations
        y, x = rng.integers(0, N - 16, 2)
        u[y:y + 16, x:x + 16], v[y:y + 16, x:x + 16] = 0.5, 0.25
    u += 0.01 * rng.standard_normal((N, N))
    v += 0.01 * rng.standard_normal((N, N))
    snaps, events, log = [], [], []
    t0 = time.time()
    for step in range(N_SNAP * EVERY):
        ts = time.perf_counter()
        uvv = u * v * v
        u += DT * (DU * lap(u) - uvv + F * (1 - u))
        v += DT * (DV * lap(v) + uvv - (F + K) * v)
        dt_wall = time.perf_counter() - ts
        stamp = f"2026-11-10T{8 + step // 3600:02d}:{step // 60 % 60:02d}:{step % 60:02d}"
        su, sv = (a.reshape(RANKS, N // RANKS, RANKS, N // RANKS) for a in (u, v))
        st = [f(axis=(1, 3)) for a in (su, sv) for f in (a.min, a.max, a.mean)]
        for r in range(RANKS * RANKS):
            i, j = divmod(r, RANKS)
            log.append(f"{stamp} INFO gray_scott rank={r} step={step} t={step * DT:.1f} "
                       f"U[min={st[0][i, j]:.5f} max={st[1][i, j]:.5f} mean={st[2][i, j]:.6f}] "
                       f"V[min={st[3][i, j]:.5f} max={st[4][i, j]:.5f} mean={st[5][i, j]:.6f}] "
                       f"dt={DT} wall={dt_wall * 1e3:.2f}ms\n")
        if (step + 1) % DETECT_EVERY == 0:
            events += spots(v, step + 1)
        if (step + 1) % EVERY == 0:
            snaps.append(np.stack([u, v]).copy())
            print(f"snapshot {len(snaps):2d}/{N_SNAP} step {step + 1} spots {len(events)} "
                  f"({time.time() - t0:.0f} s)", flush=True)
    snaps = np.array(snaps)
    np.save(os.path.join(out, "snapshots.npy"), snaps)
    with open(os.path.join(out, "log.txt"), "w") as f:
        f.writelines(log)
    with open(os.path.join(out, "events.bin"), "wb") as f:
        f.write(b"".join(events))
    import zstandard
    c = zstandard.ZstdCompressor(level=3)
    with open(os.path.join(out, "products.bin"), "wb") as f:
        for s in snaps:
            f.write(c.compress(s.tobytes()))
    print("done", out, flush=True)


if __name__ == "__main__":
    main()
