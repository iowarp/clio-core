#!/usr/bin/env python3
"""Field subsets with the largest possible gain, from a workload's stored
exhaustive search (no run).

    subset_search.py DATASET [--min-gb 15] [--strip-suffix REGEX] [--top 10]

For every cost model of opp_grid.py's grid (realistic models only: w_io >=
w_dt), start from all fields and remove one field at a time, each time the
field whose removal raises the possible gain most, while the kept data stays
>= --min-gb GB (10^9 B). --strip-suffix removes a part of the field name (e.g.
a snapshot number "_s\\d{3}$") so that one field over all snapshots is kept or
removed together. Prints the best subsets with their gain and best single codec.
"""
import argparse
import re

import numpy as np
import pandas as pd

import opp_grid as og


def field_sums(fields, ct, dt, st, tier, model):
    """Per field: the summed cost of each setting and the summed per-chunk minimum.

    @return (field names, F [field x setting], M [field]) for one cost model
    """
    w_dt, w_io, bw = model
    b = tier[:, None] if bw == "tiers" else bw * 1e6
    cost = og.ev.for_selection(ct + w_dt * dt + w_io * st / b)
    ok = np.isfinite(cost).all(axis=1)
    names = sorted(set(fields))
    F = np.array([cost[ok & (fields == f)].sum(axis=0) for f in names])
    M = np.array([cost[ok & (fields == f)].min(axis=1).sum() for f in names])
    return names, F, M


def greedy(fields, raw, ct, dt, st, tier, model, min_b):
    """Backward greedy removal for one model.

    @return list of (gain %, best single idx, kept fields, kept bytes) per step
    """
    names, F, M = field_sums(fields, ct, dt, st, tier, model)
    nb = np.array([raw[fields == f].sum() for f in names])
    keep = np.ones(len(names), bool)

    def score(k):
        tot = F[k].sum(axis=0)
        bf = int(np.argmin(tot))
        return 100.0 * (tot[bf] - M[k].sum()) / tot[bf], bf

    g, bf = score(keep)
    steps = [(g, bf, [n for n, k in zip(names, keep) if k], nb[keep].sum())]
    while keep.sum() > 1:
        best = None
        for i in np.flatnonzero(keep):
            k = keep.copy(); k[i] = False
            if nb[k].sum() < min_b:
                continue
            gg, bb = score(k)
            if best is None or gg > best[0]:
                best = (gg, bb, i)
        if best is None:
            break
        keep[best[2]] = False
        steps.append((best[0], best[1], [n for n, k in zip(names, keep) if k], nb[keep].sum()))
    return steps


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dataset")
    ap.add_argument("--min-gb", type=float, default=15.0)
    ap.add_argument("--strip-suffix", default="")
    ap.add_argument("--top", type=int, default=10)
    a = ap.parse_args()
    fields, ct, dt, st, raw, tier, names, store = og.matrices(a.dataset)
    if a.strip_suffix:
        fields = np.array([re.sub(a.strip_suffix, "", f) for f in fields])
    rows = []
    for w_dt in og.W_DT:
        for w_io in og.W_IO:
            if w_io < w_dt or w_io == 0:
                continue
            for bw in og.BW:
                for g, bf, keep, nb in greedy(fields, raw, ct, dt, st, tier, (w_dt, w_io, bw),
                                              a.min_gb * 1e9):
                    if names[bf] != "store":
                        rows.append({"w_dt": w_dt, "w_io": w_io, "bw_GBs": bw, "gain_pct": g,
                                     "best_single": names[bf], "GB": nb / 1e9,
                                     "n_fields": len(keep), "fields": ",".join(keep)})
    t = pd.DataFrame(rows).sort_values("gain_pct", ascending=False)
    t = t.drop_duplicates(["w_dt", "w_io", "bw_GBs"])
    print(f"{a.dataset}: {len(set(fields))} fields, best subsets >= {a.min_gb:g} GB")
    with pd.option_context("display.width", 250, "display.max_colwidth", 200):
        print(t.head(a.top).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
