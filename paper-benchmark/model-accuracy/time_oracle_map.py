#!/usr/bin/env python3
"""Write a workload's time-oracle map: each chunk's setting with the lowest end-to-end time,
from the stored exhaustive search, at the workload's real disk speed fitted to a measured
best-static run of the same setup.

    time_oracle_map.py WORKLOAD --fixed-tag TAG [--out MAP]

The real speed is the bandwidth at which the model (compress + READS x decompress +
(1 + READS) x stored bytes / speed, every chunk with the run's setting) gives the run's
measured application time. The map ("blob,setting", read by Clio through ORACLE_MAP) holds
each chunk's lowest-time candidate setting at that speed. Prints and writes the speed
(MAP.speed) so that the paper can state it.
"""
import argparse
import os

import numpy as np
import pandas as pd

import compare_parallel_runs as cp
import eval_v2_workloads as ev
import final_config as fc
import replay_learning as rl

RUNS = "/mnt/nvme0/v2-work/runs"


def stored_setting(run):
    """@return the setting the best-static run stored its chunks with."""
    m = pd.read_csv(os.path.join(run, "v2_measured.csv"))
    return int(m[m.role == "primary"].setting.mode().iloc[0])


def model_time(meas, nb, store, pick, bw):
    """@return model end-to-end s of one setting per chunk at speed bw (bytes per ms)."""
    i = np.arange(len(nb))
    ct, dt, r = meas[i, pick].T
    raw = (pick == store) | ~(r > 1)
    st = np.where(raw, nb, nb / np.where(raw, 1.0, r))
    t = np.where(pick == store, 0.0, ct) + fc.READS * np.where(raw, 0.0, dt) + (1 + fc.READS) * st / bw
    return t.sum() / 1e3


def fit_speed(meas, nb, store, setting, app_s):
    """@return the speed (bytes per ms) at which one setting's model time equals app_s."""
    pick = np.full(len(nb), setting)
    lo, hi = 0.05e6, 20e6
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if model_time(meas, nb, store, pick, mid) > app_s else (lo, mid)
    return (lo + hi) / 2


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("workload", choices=list(fc.WORKLOADS))
    ap.add_argument("--fixed-tag", required=True, help="tag of the best-static run, e.g. km10b1gfin1p1i1w1-40-5")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ds = fc.WORKLOADS[a.workload][0]
    run = os.path.join(RUNS, f"{ds}-p0_fixed_{a.fixed_tag}")
    app_s = cp.parallel_row(ds, 1, "fixed", a.fixed_tag)["app_s"]
    names, store, order, meas, _ = rl.load(ds)
    nb = order.bytes.to_numpy(float)
    s = stored_setting(run)
    bw = fit_speed(meas, nb, store, s, app_s)
    ct, dt, r = meas[..., 0], meas[..., 1], meas[..., 2]
    kept = r > 1
    st = np.where(kept, nb[:, None] / np.where(kept, r, 1.0), nb[:, None])
    t = ct + fc.READS * np.where(kept, dt, 0.0) + (1 + fc.READS) * st / bw
    t[:, store] = (1 + fc.READS) * nb / bw
    best = np.nanargmin(ev.for_selection(t), axis=1)
    out = a.out or os.path.join(RUNS, f"{ds}_oracle_map_time_{a.fixed_tag}.csv")
    pd.DataFrame({"blob": order.blob, "setting": best}).to_csv(out, index=False)
    i = np.arange(len(nb))
    with open(out + ".speed", "w") as f:
        f.write(f"{bw:.1f}\n")
    print(f"{a.workload}: best static {names[s]} measured {app_s:.1f} s -> real speed {bw / 1e6:.3f} GB/s; "
          f"time oracle model {t[i, best].sum() / 1e3:.1f} s, ratio {nb.sum() / st[i, best].sum():.2f}; wrote {out}")


if __name__ == "__main__":
    main()
