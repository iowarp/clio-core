#!/usr/bin/env python3
"""Offline test of NeuroPress learning variants on one probe (no run).

    learn_variants.py DATASET [--passes 1]

Replays Clio's selection (replay_learning.py) under the tuning cost model with:
  * as trained (no learning) and the current rule (NLMS on the picked
    setting, lr 0.5, threshold 0.30);
  * NLMS with `shared` (every setting moves by part of the same correction);
  * backprop through all layers;
  * runner-up labels: every N-th chunk the setting with the second-lowest
    predicted cost is also compressed and learned from (the chunk is still
    stored with the first pick); its compress time is added to the cost.
Prints NeuroPress's cost against the best single codec, and the median
predicted / measured ratio of the best single codec at the end.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model-accuracy"))
import replay_learning as rl  # noqa: E402


def replay_runner_up(data, lr, every, thr=rl.MAPE_THRESHOLD):
    """The current rule plus a runner-up label every `every` chunks.

    @return (total cost of the picks plus the extra compressions, extra
             compressions, predictions [chunk x setting x 3])
    """
    names, store, order, meas, cost = data
    net = rl.Net()
    n, ns = len(order), len(names)
    X = order[["x0", "x1", "x2", "x3"]].to_numpy(np.float64)
    nbytes = order.bytes.to_numpy(float)
    bw = order.tier_bw.to_numpy(float)
    preds = np.empty((n, ns, 3))
    total, extra_ct, n_extra = 0.0, 0.0, 0

    def learn(a, i, s, lp):
        lab = meas[i, s]
        if s == store or not np.isfinite(lab[0]) or lab[2] <= 0:
            return
        ct, dt, r = np.exp(lp[s])
        pc = ct + dt + nbytes[i] / (r * bw[i])
        act = lab[0] + dt + nbytes[i] / (lab[2] * bw[i])   # no measured dt on write
        if abs(act - pc) / act > thr:
            tl = np.log(np.array([lab[0], np.nan, lab[2]]))
            rl.step_nlms(net, a, s, tl, lr, 0.0)

    for i in range(n):
        a = net.forward(X[i])
        lp = net.log_pred(a[-1])
        preds[i] = lp
        ct, dt, r = np.exp(lp[:, 0]), np.exp(lp[:, 1]), np.exp(lp[:, 2])
        pc = ct + dt + nbytes[i] / (r * bw[i])
        order_pc = np.argsort(pc)
        s = int(order_pc[0])
        total += cost[i, s] if np.isfinite(cost[i, s]) else 0.0
        learn(a, i, s, lp)
        if every and i % every == 0:
            s2 = next((int(k) for k in order_pc[1:] if k != store), None)
            if s2 is not None and np.isfinite(meas[i, s2, 0]):
                extra_ct += meas[i, s2, 0]
                n_extra += 1
                learn(a, i, s2, lp)
    return total + extra_ct, n_extra, preds


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dataset")
    a = ap.parse_args()
    data = rl.load(a.dataset)
    names, store, order, meas, cost = data
    tot_s = np.nansum(rl.ev.for_selection(cost), axis=0)
    bf = int(np.argmin(tot_s))
    best = np.nansum(cost[:, bf])
    rows = []

    def add(label, total, preds, note=""):
        pr = np.nanmedian(np.exp(preds[:, bf, 2]) / meas[:, bf, 2])
        rows.append({"variant": label, "cost_vs_best_single_pct": 100 * (total / best - 1),
                     f"{names[bf]}_pred_over_meas_ratio": pr, "note": note})

    for label, rule, lr, shared in (("as trained (no learning)", "nlms", 0.0, 0.0),
                                    ("current: NLMS lr 0.5", "nlms", 0.5, 0.0),
                                    ("NLMS lr 0.5, shared 0.25", "nlms", 0.5, 0.25),
                                    ("NLMS lr 0.5, shared 0.5", "nlms", 0.5, 0.5),
                                    ("backprop lr 0.01", "backprop", 0.01, 0.0),
                                    ("backprop lr 0.05", "backprop", 0.05, 0.0)):
        thr = rl.MAPE_THRESHOLD if lr else np.inf
        rec, preds, _ = rl.replay(a.dataset, rule, lr, shared=shared, thr=thr, data=data)
        add(label, np.nansum(rec.cost_pick), preds)
    for every in (50, 20, 10, 5):
        total, n_extra, preds = replay_runner_up(data, 0.5, every)
        add(f"current + runner-up label every {every}", total, preds,
            f"{n_extra} extra compressions counted")
    t = pd.DataFrame(rows)
    print(f"{a.dataset}: best single {names[bf]} {best:.0f} ms (tuning model); oracle "
          f"{100 * (np.nansum(np.nanmin(rl.ev.for_selection(cost), axis=1)) / best - 1):+.1f}%")
    with pd.option_context("display.width", 200):
        print(t.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
