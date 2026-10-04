#!/usr/bin/env python3
"""How bad is one workload's best codec on another? Numbers for the case for
automatic selection, from the sweep (~/np-newsweep), volume-weighted.

    cross_penalty.py [--out DIR]

Per dataset, over all 139 settings (balanced cost on the 10/30/30/30
hierarchy at 12 / 1 / 0.5 / 0.25 GB/s, as in hierarchy_balanced.png):
  own.csv      best single setting by cost and by ratio; per-chunk optimal
               ratio and cost for both objectives
  cross.csv    setting that is best on dataset A, used on dataset B: B's cost
               and ratio relative to B's own best single setting
  products.csv the same inside one dataset, between its array types (the
               best setting of one array type used for another)
"""
import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plot_new_workloads as pw  # noqa: E402
ch, vw = pw.ch, pw.vw


def load(key):
    """Weighted per-chunk rows with tier cost and array type."""
    d = os.path.join(os.path.expanduser("~/np-newsweep"), key)
    df, _, _ = pw.load_dataset(d)
    src = (os.path.expanduser(f"~/np-data/{key[4:]}/fields") if key.startswith("ref-")
           else os.path.expanduser(f"~/np-data/new/{key}"))
    df = vw.apply(df, vw.chunk_weights(d, src))
    bw = ch.place(df).map(pw.BW_HIER)
    df["cost"] = df.comp_ms + df.decomp_ms + df.comp_bytes / (bw * 1e6)
    return df


def totals(df):
    """Per setting: total cost (ms) and ratio."""
    g = df.groupby("config")[["cost", "bytes", "comp_bytes"]].sum()
    g["ratio"] = g.bytes / g.comp_bytes
    return g


def own(df):
    """Best single and per-chunk optimal, by cost and by ratio."""
    t = totals(df)
    bc, br = t.cost.idxmin(), t.ratio.idxmax()
    pc = df.loc[df.groupby(["file", "chunk"]).cost.idxmin()]
    pr = df.loc[df.groupby(["file", "chunk"]).comp_bytes.idxmin()]
    return {"best_cost_setting": bc, "best_cost_ms": t.cost[bc], "best_cost_ratio": t.ratio[bc],
            "perchunk_cost_ms": pc.cost.sum(), "perchunk_cost_ratio": pc.bytes.sum() / pc.comp_bytes.sum(),
            "best_ratio_setting": br, "best_ratio": t.ratio[br],
            "perchunk_max_ratio": pr.bytes.sum() / pr.comp_bytes.sum()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "cross-penalty"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data = {k: load(k) for k in pw.DATASETS}
    tot = {k: totals(df) for k, df in data.items()}
    o = pd.DataFrame({k: own(df) for k, df in data.items()}).T
    o.to_csv(os.path.join(a.out, "own.csv"))
    rows = []
    for ka in data:
        sa = o.loc[ka, "best_cost_setting"]
        for kb in data:
            tb = tot[kb]
            sb = o.loc[kb, "best_cost_setting"]
            rows.append({"best_of": ka, "used_on": kb, "setting": sa,
                         "cost_penalty_pct": 100 * (tb.cost[sa] / tb.cost[sb] - 1),
                         "ratio": tb.ratio[sa], "own_ratio": tb.ratio[sb]})
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "cross.csv"), index=False)
    prow = []
    for k, df in data.items():
        tp = {p: totals(g) for p, g in df.groupby("product")}
        if len(tp) < 2:
            continue
        for pa, ta in tp.items():
            sa = ta.cost.idxmin()
            for pb, tb in tp.items():
                if pa == pb:
                    continue
                sb = tb.cost.idxmin()
                prow.append({"dataset": k, "best_of": pa, "used_on": pb, "setting": sa,
                             "own_setting": sb, "cost_penalty_pct": 100 * (tb.cost[sa] / tb.cost[sb] - 1),
                             "ratio": tb.ratio[sa], "own_ratio": tb.ratio[sb],
                             "used_on_share": tb.bytes[sb] / df.groupby("config").bytes.sum().iloc[0]})
    pd.DataFrame(prow).to_csv(os.path.join(a.out, "products.csv"), index=False)
    print("wrote", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
