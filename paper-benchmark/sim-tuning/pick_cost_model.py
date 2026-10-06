#!/usr/bin/env python3
"""Rank the cost models of np_cost_sweep.py's table by how well NeuroPress
learning does in BOTH modelled runtime and ratio against the best single codec
(no run).

    pick_cost_model.py DATASET [--top 12]

Score = min(-runtime change %, ratio change %) of NeuroPress learning (1
pass): a model ranks high only if NeuroPress is faster AND compresses more.
Reads runs/np_cost_sweep_DATASET.csv; prints the best models with the oracle
and the best single codec next to them.
"""
import argparse
import os

import pandas as pd

RUNS = "/mnt/nvme0/v2-work/runs"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dataset")
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()
    t = pd.read_csv(os.path.join(RUNS, f"np_cost_sweep_{a.dataset}.csv"))
    t = t[t.best_single != "store"].copy()
    t["score"] = (-t.np_learn1_runtime_pct).clip(upper=None).combine(t.np_learn1_ratio_pct, min)
    t = t.sort_values("score", ascending=False)
    cols = ["bw_GBs", "reads", "w_ct", "w_dt", "w_io", "best_single", "np_learn1_runtime_pct",
            "np_learn1_ratio_pct", "oracle_runtime_pct", "oracle_ratio_pct", "score"]
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(f"{a.dataset}: {len(t)} cost models; NeuroPress learning vs best single "
              f"(runtime < 0 and ratio > 0 are better)")
        print(t[cols].head(a.top).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
