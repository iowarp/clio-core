#!/usr/bin/env python3
"""Score one simulation tuning probe: does per-chunk choice beat one codec,
and does NeuroPress v2 learning get there?

    probe_eval.py DATASET [--note TEXT]

From the probe's stored exhaustive search (all 45 settings on every chunk,
balanced 4-tier cost): the best single codec, the opportunity (per-chunk best
vs best single), per field the gain and the cheapest settings; and
NeuroPress v2's picks replayed through Clio's exact selection and learning
loop (replay_learning.py, which reproduces the Clio runs' picks) at the run's
own length and over 5 passes, as cost vs the best single codec (< 0 = cheaper).
Appends one row to runs/nyx_tuning.csv (or runs/vpic_tuning.csv for a VPIC probe).
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "model-accuracy"))
import eval_v2_workloads as ev  # noqa: E402
import replay_learning as rl  # noqa: E402


def field(blob):
    """Field name of a chunk."""
    m = (re.search(r"comp\d+_([A-Za-z_0-9]+)\.f32", blob)
         or re.search(r"__(force|position|velocity)_step", blob)
         or re.search(r"__plt\d+__([A-Za-z_0-9]+)\.f32", blob)   # WarpX: plt<step>/<field>.f32
         or re.search(r"__([A-Za-z][A-Za-z_0-9-]*?)(?:[-_]\d+x\d+(?:x\d+)*)?(?:_\d+_\d+_\d+)?\.[a-z0-9]+__c\d{4}__", blob))
    # last one: any staged file, <name>[-dims].<ext> (SDRBench and others)
    return m.group(1) if m else "?"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dataset")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    data = rl.load(a.dataset)
    names, store, order, meas, cost = data
    ok = np.isfinite(cost).all(axis=1)
    cost = ev.for_selection(cost)   # best single and oracle: candidates only
    tot = np.nansum(cost[ok], axis=0)
    bf = int(np.argmin(tot))
    fixed = tot[bf]
    oracle = np.nanmin(cost[ok], axis=1).sum()
    f = np.array([field(b) for b in order.blob])
    print(f"{a.dataset}: {len(order)} chunks, best single {names[bf]}, "
          f"opportunity {100 * (fixed - oracle) / fixed:.1f}%, other setting cheapest on "
          f"{100 * np.mean(np.nanargmin(cost[ok], axis=1) != bf):.0f}% of chunks")
    # the benchmark's cost model: 1 write + 10 reads, weights 1/10/10, one 1 GB/s tier
    _, cb = ev.load_truth(os.path.join(rl.STORE, a.dataset, "exhaustive"), len(names), store,
                          w=(1.0, 10.0, 10.0), bw=1e6)
    cb = ev.for_selection(cb)
    okb = np.isfinite(cb).all(axis=1)
    tb = cb[okb].sum(axis=0)
    bb = int(np.argmin(tb))
    print(f"   benchmark model (1/10/10 at 1 GB/s): best single {names[bb]}, opportunity "
          f"{100 * (tb[bb] - cb[okb].min(axis=1).sum()) / tb[bb]:.1f}%")
    for fv in sorted(set(f)):
        m = (f == fv) & ok
        c = cost[m]
        g = 100 * (c[:, bf].sum() - c.min(axis=1).sum()) / c[:, bf].sum()
        w = pd.Series([names[k] for k in c.argmin(axis=1)]).value_counts(normalize=True)
        print(f"   {fv:14s} share of cost {100 * c[:, bf].sum() / fixed:5.1f}%  gain {g:5.1f}%  "
              + ", ".join(f"{k} {100 * v:.0f}%" for k, v in w.head(3).items()))
    row = {"probe": a.dataset, "note": a.note, "chunks": len(order),
           "best_single": names[bf], "opportunity_pct": 100 * (fixed - oracle) / fixed}
    for label, rule, lr, passes in (("frozen", "nlms", 0.0, 1), ("learn", "nlms", 0.5, 5)):
        big = rl.tile(data, passes)
        rec, _, _ = rl.replay(a.dataset, rule, lr, thr=rl.MAPE_THRESHOLD if lr else np.inf, data=big)
        n1 = len(order)
        for p in range(passes):
            r = rec.iloc[p * n1:(p + 1) * n1]
            v = 100 * (r.cost_pick.sum() / fixed - 1)
            if label == "frozen":
                row["np_frozen_pct"] = v
            elif p in (0, passes - 1):
                row[f"np_learn_pass{p + 1}_pct"] = v
    print(f"   NeuroPress vs best single (cost; < 0 = cheaper): as trained "
          f"{row['np_frozen_pct']:+.1f}%, learning pass 1 {row['np_learn_pass1_pct']:+.1f}%, "
          f"pass 5 {row['np_learn_pass5_pct']:+.1f}%")
    out = os.path.join(ev.RUNS, "vpic_tuning.csv" if "vpic" in a.dataset else "lammps_tuning.csv" if "lmp" in a.dataset or "lammps" in a.dataset else "nyx_tuning.csv")
    t = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame()
    t = pd.concat([t[t.probe != a.dataset] if len(t) else t, pd.DataFrame([row])])
    t.to_csv(out, index=False)


if __name__ == "__main__":
    main()
