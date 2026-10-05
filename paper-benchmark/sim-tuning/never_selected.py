#!/usr/bin/env python3
"""Which of the 45 settings is never selected: not the best single codec of
any workload and not the best setting of any chunk (from the stored CSV files;
no run).

    never_selected.py [--models balanced,w011,w145] [--out CSV]

For every workload with a stored exhaustive search (baselines/<ds>/exhaustive)
and every cost model:
  balanced  w = 1/1/1, each chunk at the bandwidth of the tier it was written
            to (12 / 1 / 0.5 / 0.25 GB/s), the cost the searches were made with
  w011      w = 0/1/1 at 0.52 GB/s (one PFS tier)
  w145      w = 1/4/5 at 0.52 GB/s (1 write + 4 reads)
the best single codec is the setting with the lowest total cost over the
chunks that have every setting measured, and each chunk's best setting is its
lowest measured cost (eval_v2_workloads.load_truth: a setting that does not
shrink a chunk stores it raw).
Output: one row per setting with the cases where it is the best single codec,
the chunks where it is the best setting and the workloads of those chunks,
and the list of settings that are never selected.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "model-accuracy"))
import eval_v2_workloads as ev  # noqa: E402

STORE = "/mnt/nvme0/v2-work/baselines"
MODELS = {"balanced": ((1.0, 1.0, 1.0), None),
          "w011": ((0.0, 1.0, 1.0), 520000.0),
          "w145": ((1.0, 4.0, 5.0), 520000.0)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", default="balanced,w011,w145")
    ap.add_argument("--out", default="/mnt/nvme0/v2-work/runs/never_selected.csv")
    a = ap.parse_args()
    names, store = ev.settings_list()
    ns = len(names)
    single = {m: np.zeros(ns, int) for m in MODELS}
    chunks = {m: np.zeros(ns, int) for m in MODELS}
    wls = {m: [set() for _ in range(ns)] for m in MODELS}
    dsets = sorted(d for d in os.listdir(STORE)
                   if os.path.exists(os.path.join(STORE, d, "exhaustive", "v2_measured.csv")))
    for ds in dsets:
        run = os.path.join(STORE, ds, "exhaustive")
        for m in a.models.split(","):
            w, bw = MODELS[m]
            _, cost = ev.load_truth(run, ns, store, w, bw)
            ok = ~np.isnan(cost).any(axis=1)
            single[m][int(np.argmin(cost[ok].sum(axis=0)))] += 1
            has = ~np.isnan(cost).all(axis=1)
            best = np.nanargmin(np.where(np.isnan(cost[has]), np.inf, cost[has]), axis=1)
            cnt = np.bincount(best, minlength=ns)
            chunks[m] += cnt
            for s in np.nonzero(cnt)[0]:
                wls[m][s].add(ds)
        print(f"{ds}: done", flush=True)
    rows = []
    for s in range(ns):
        r = {"setting": s, "name": names[s]}
        for m in a.models.split(","):
            r[f"best_single_{m}"] = single[m][s]
            r[f"best_chunks_{m}"] = chunks[m][s]
            r[f"best_chunk_workloads_{m}"] = len(wls[m][s])
        rows.append(r)
    t = pd.DataFrame(rows)
    sel = [c for c in t.columns if c.startswith(("best_single", "best_chunks"))]
    t["never_selected"] = (t[sel] == 0).all(axis=1)
    t.to_csv(a.out, index=False)
    with pd.option_context("display.width", 250):
        print(t.to_string(index=False))
    print(f"\n{len(dsets)} workloads; never selected under any model:")
    print(t[t.never_selected][["setting", "name"]].to_string(index=False))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
