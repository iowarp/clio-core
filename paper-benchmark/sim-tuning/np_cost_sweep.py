#!/usr/bin/env python3
"""Sweep cost-model combinations on one workload's stored CSV files (no run)
and score NeuroPress against the fastest single codec by runtime.

    np_cost_sweep.py DATASET [--workers 48] [--predict-s 3.3] [--limit N]

For each combination (bandwidth BW, reads R, weights w), every option selects
by the SAME cost model (user's rule):
  * the best single codec: the one setting with the lowest total cost under w
    over all chunks;
  * NeuroPress: the offline replay of Clio's selection with the weights w at
    BW (as trained, and learning for 1 pass: lr 0.5, threshold 0.30, no
    measured decompress time on the write path), plus its prediction time;
  * the oracle: each chunk's lowest-cost setting under w.
Every value comes from the stored exhaustive search (measured compress time,
decompress time and ratio of every setting on every chunk).
Weights are written as multipliers of the runtime weights 1 / R / R + 1:
(m_ct, m_dt, m_io) gives w = (m_ct, m_dt x R, m_io x (R + 1)).
Every option is then scored by its modelled runtime of 1 write + R reads at BW
(compress + R x decompress + (1 + R) x stored bytes / BW), its cost under w and
its ratio, against the best single codec. The fastest single codec by runtime
is kept as an information column.
Output: runs/np_cost_sweep_<DATASET>.csv (one row per combination).
"""
import argparse
import itertools
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import costmodel_whatif as cw  # noqa: E402

rl = cw.rl
DATA = {}


def init(ds):
    """Load the workload once per worker process."""
    names, store, order, meas, _ = rl.load(ds)
    nb = order.bytes.to_numpy(float)
    ct, dt, st = cw.outcomes(meas, nb, store)
    DATA.update(ds=ds, names=names, store=store, order=order, meas=meas, nb=nb,
                ct=ct, dt=dt, st=st)


def score(job):
    """One combination: fastest single, NeuroPress (frozen, learning) and oracle."""
    bw_gbs, reads, mult, predict_s = job
    d = DATA
    bw = bw_gbs * 1e6
    w = (mult[0], mult[1] * reads, mult[2] * (reads + 1))
    bwv = np.full(len(d["nb"]), bw)
    cost = cw.truth_cost(d["meas"], d["nb"], d["store"], w, bwv)
    run = d["ct"] + reads * d["dt"] + (1 + reads) * d["st"] / bw
    ok = ~np.isnan(cost).any(axis=1) & ~np.isnan(run).any(axis=1)
    r = np.arange(ok.sum())
    run_ok, st_ok, nb = run[ok], d["st"][ok], d["nb"][ok].sum()
    cost_ok = cost[ok]
    # The best single codec and the oracle select from the candidate settings
    # (eval_v2_workloads.DROPPED_SETTINGS left out).
    best = int(np.argmin(cw.ev.for_selection(cost_ok).sum(axis=0)))   # lowest total cost under w
    fast = int(np.argmin(cw.ev.for_selection(run_ok).sum(axis=0)))    # information only
    t0, ratio0, c0 = (run_ok[:, best].sum() / 1e3, nb / st_ok[:, best].sum(),
                      cost_ok[:, best].sum())

    def stats(picks, extra_s=0.0):
        p = picks[ok]
        t = run_ok[r, p].sum() / 1e3 + extra_s
        return (100 * (t / t0 - 1), 100 * (nb / st_ok[r, p].sum() / ratio0 - 1),
                100 * (cost_ok[r, p].sum() / c0 - 1))
    sc = {"w": w, "bw": bw}
    picks = {m: p for m, p, _ in cw.replays(d["ds"], (d["names"], d["store"], d["order"],
                                                      d["meas"], cost), sc, 1)}
    oracle = np.argmin(cw.ev.for_selection(np.where(np.isnan(cost), np.inf, cost)), axis=1)
    row = {"bw_GBs": bw_gbs, "reads": reads, "mult": "/".join(f"{m:g}" for m in mult),
           "w_ct": w[0], "w_dt": w[1], "w_io": w[2], "best_single": d["names"][best],
           "best_single_runtime_s": t0, "best_single_ratio": ratio0,
           "fastest_by_runtime": d["names"][fast],
           "fastest_runtime_vs_best_single_pct": 100 * (run_ok[:, fast].sum() / 1e3 / t0 - 1)}
    for key, p, extra in (("np_frozen", picks[cw.FROZEN], predict_s),
                          ("np_learn1", picks[cw.LEARN1], predict_s), ("oracle", oracle, 0.0)):
        row[f"{key}_runtime_pct"], row[f"{key}_ratio_pct"], row[f"{key}_cost_pct"] = stats(p, extra)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dataset")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--predict-s", type=float, default=3.3,
                    help="NeuroPress prediction time added to its runtime (s)")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N (timing test)")
    a = ap.parse_args()
    bws = (0.1, 0.25, 0.52, 1, 2, 3, 5, 8)
    reads = (1, 2, 4, 10)
    mults = ((1, 1, 1), (1, 1, 1.25), (1, 1, 1.5), (1, 1, 2), (1, 1, 3), (1, 1, 5),
             (0.5, 1, 1), (0, 1, 1), (1, 0.5, 1), (1, 2, 1), (1, 1, 0.75), (0.5, 0.5, 1))
    jobs = [(b, r, m, a.predict_s) for b, r, m in itertools.product(bws, reads, mults)]
    if a.limit:
        jobs = jobs[:a.limit]
    t = time.time()
    with Pool(min(a.workers, len(jobs)), initializer=init, initargs=(a.dataset,)) as pool:
        rows = pool.map(score, jobs, chunksize=1)
    out = pd.DataFrame(rows)
    path = os.path.join(rl.RUNS, f"np_cost_sweep_{a.dataset}{'_test' if a.limit else ''}.csv")
    out.to_csv(path, index=False)
    print(f"{len(rows)} combinations in {time.time() - t:.0f} s -> {path}")


if __name__ == "__main__":
    main()
