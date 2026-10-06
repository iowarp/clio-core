#!/usr/bin/env python3
"""Where does NeuroPress v2 lose against the best single codec on a probe?

    np_mistakes.py DATASET [--lr 0.5]

Replays Clio's selection and learning (replay_learning.py) once over the
probe and groups the extra cost of NeuroPress's picks over the best single
codec by the setting it picked and by field: chunks, extra cost, and for the
costliest picks the predicted / measured ratio and compress time.
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model-accuracy"))
import replay_learning as rl  # noqa: E402
import probe_eval as pe  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dataset")
    ap.add_argument("--lr", type=float, default=0.5)
    a = ap.parse_args()
    data = rl.load(a.dataset)
    names, store, order, meas, cost = data
    bf = int(np.argmin(np.nansum(rl.ev.for_selection(cost), axis=0)))   # candidates only
    rec, preds, _ = rl.replay(a.dataset, "nlms", a.lr, thr=rl.MAPE_THRESHOLD if a.lr else np.inf, data=data)
    n = np.arange(len(rec))
    r = rec.assign(pick_name=[names[k] for k in rec.pick],
                   field=[pe.field(b) for b in rec.blob],
                   extra=rec.cost_pick - cost[n, bf])
    tot = np.nansum(cost[:, bf])
    print(f"{a.dataset}: best single {names[bf]} {tot:.0f} ms; NeuroPress extra {r.extra.sum():+.0f} ms "
          f"({100 * r.extra.sum() / tot:+.1f}%)")
    g = r.groupby("pick_name").agg(chunks=("pick", "size"), extra_ms=("extra", "sum"))
    g = g.sort_values("extra_ms", ascending=False)
    for name, row in g.head(6).iterrows():
        k = names.index(name)
        i = np.where(r.pick_name == name)[0]
        rp = np.median(np.exp(preds[i, k, 2]) / meas[i, k, 2]) if k != store else np.nan
        print(f"   {name:34s} {int(row.chunks):4d} chunks  extra {row.extra_ms:+8.1f} ms  "
              f"pred/meas ratio {rp:6.2f}  measured ratio {np.median(meas[i, k, 2]):5.2f} "
              f"vs {names[bf]} {np.median(meas[i, bf, 2]):5.2f}")
    print(r.groupby("field").extra.sum().sort_values(ascending=False).round(1).to_string())
    top = g.head(4).index
    t = r[r.pick_name.isin(top)].pivot_table(index="field", columns="pick_name", values="extra",
                                             aggfunc="sum", fill_value=0.0)
    print("extra ms by field and pick (the 4 costliest picks):")
    print(t.round(1).to_string())


if __name__ == "__main__":
    main()
