#!/usr/bin/env python3
"""Write the oracle baseline's per-chunk setting map for one workload.

    oracle_map.py DATASET [--out CSV]

From the stored exhaustive search (baselines/<DATASET>/exhaustive), each
chunk's cheapest setting at the tier it was written to, by the balanced
4-tier cost model, or the weights and one bandwidth given by --w and --bw
(eval_v2_workloads.load_truth: a setting that does not
shrink the chunk costs compress + raw I/O; storing raw costs raw I/O). Output
"blob,setting" (setting index; the store setting means raw), read by Clio
through CLIO_NEUROPRESS_SETTING_MAP (run_v2_workloads.sh oracle).
"""
import argparse
import os

import numpy as np
import pandas as pd

import eval_v2_workloads as ev

STORE = "/mnt/nvme0/v2-work/baselines"
RUNS = "/mnt/nvme0/v2-work/runs"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dataset")
    ap.add_argument("--out", default=None)
    ap.add_argument("--w", default="1,1,1",
                    help="cost weights W_CT,W_DT,W_IO [1,1,1]")
    ap.add_argument("--bw", type=float, default=None,
                    help="one bandwidth in bytes per ms for every chunk "
                         "[default: each chunk's 4-tier bandwidth]")
    a = ap.parse_args()
    w = tuple(float(x) for x in a.w.split(","))
    names, store = ev.settings_list()
    blobs, cost = ev.load_truth(os.path.join(STORE, a.dataset, "exhaustive"),
                                len(names), store, w=w, bw=a.bw)
    best = np.nanargmin(ev.for_selection(cost), axis=1)   # candidates only
    out = a.out or os.path.join(RUNS, f"{a.dataset}_oracle_map.csv")
    pd.DataFrame({"blob": blobs, "setting": best}).to_csv(out, index=False)
    share = pd.Series([names[k] for k in best]).value_counts()
    print(f"{a.dataset}: {len(blobs)} chunks, {len(share)} settings "
          f"(top: {', '.join(f'{k} {v}' for k, v in share.head(3).items())}) -> {out}")


if __name__ == "__main__":
    main()
