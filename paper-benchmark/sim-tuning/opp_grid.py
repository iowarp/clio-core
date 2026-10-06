#!/usr/bin/env python3
"""Possible gain of per-chunk codec choice over many cost models, from a
workload's stored exhaustive search (no run).

    opp_grid.py DATASET [--fields A,B,...] [--top 12]

For every cost model w_ct x compress + w_dt x decompress + w_io x stored
bytes / bandwidth (w_ct = 1; w_dt, w_io and the bandwidth from the grid
below; "tiers" = each chunk's own tier from the search) the best single codec
(lowest total cost over the chunks) and the possible gain = (best single -
per-chunk best) / best single. Same rule as eval_v2_workloads.load_truth: a
setting that does not shrink the chunk costs its compress time plus raw I/O;
storing raw costs the raw I/O. --fields keeps only those fields (probe_eval
field names). Prints the models with the largest gain, the largest among the
realistic ones (w_io >= w_dt: each read also moves the bytes; the best single
is a codec, not raw storage) and two reference models.
"""
import argparse
import os
import sys

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "model-accuracy"))
import eval_v2_workloads as ev  # noqa: E402
import probe_eval as pe  # noqa: E402

STORE = "/mnt/nvme0/v2-work/baselines"
W_DT = (0.0, 1.0, 2.0, 4.0, 10.0, 20.0)
W_IO = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0)
BW = ("tiers", 0.25, 0.5, 1.0, 2.0, 5.0, 12.0)   # GB/s, or each chunk's tier


def matrices(ds):
    """@return (fields, ct, dt, stored bytes, raw bytes, tier bw B/ms, names,
    store index), chunk x setting, NaN where unmeasured."""
    names, store = ev.settings_list()
    run = os.path.join(STORE, ds, "exhaustive")
    pred = pd.read_csv(os.path.join(run, "v2_pred.csv"), usecols=["blob", "bytes", "tier_bw"])
    m = pd.read_csv(os.path.join(run, "v2_measured.csv"))
    m = m[~ev.raw_primary(m) & (m.decomp_ms > 0)].drop_duplicates(["blob", "setting"])
    idx = {b: i for i, b in enumerate(pred.blob)}
    n, s = len(pred), len(names)
    ct, dt, st = (np.full((n, s), np.nan) for _ in range(3))
    r, c = m.blob.map(idx).to_numpy(), m.setting.to_numpy()
    raw = pred.bytes.to_numpy(float)
    kept = m.ratio.to_numpy() > 1
    ct[r, c] = m.comp_ms
    dt[r, c] = np.where(kept, m.decomp_ms, 0.0)          # raw-stored: no decompress
    st[r, c] = np.where(kept, raw[r] / m.ratio.to_numpy(), raw[r])
    ct[:, store], dt[:, store], st[:, store] = 0.0, 0.0, raw
    fields = np.array([pe.field(b) for b in pred.blob])
    return fields, ct, dt, st, raw, pred.tier_bw.to_numpy(float), names, store


def gain(ct, dt, st, tier, w_dt, w_io, bw):
    """@return (possible gain %, best single index) for one cost model."""
    b = tier[:, None] if bw == "tiers" else bw * 1e6
    cost = ev.for_selection(ct + w_dt * dt + w_io * st / b)
    ok = np.isfinite(cost).all(axis=1)
    tot = cost[ok].sum(axis=0)
    bf = int(np.argmin(tot))
    return 100.0 * (tot[bf] - cost[ok].min(axis=1).sum()) / tot[bf], bf


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dataset")
    ap.add_argument("--fields", default="")
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()
    fields, ct, dt, st, raw, tier, names, store = matrices(a.dataset)
    if a.fields:
        keep = np.isin(fields, a.fields.split(","))
        ct, dt, st, tier = ct[keep], dt[keep], st[keep], tier[keep]
    rows = []
    for w_dt in W_DT:
        for w_io in W_IO:
            if w_dt == 0 and w_io == 0:
                continue
            for bw in BW:
                if w_io == 0 and bw != "tiers":
                    continue   # the bandwidth does not matter without I/O
                g, bf = gain(ct, dt, st, tier, w_dt, w_io, bw)
                rows.append({"w_dt": w_dt, "w_io": w_io, "bw_GBs": bw, "gain_pct": g,
                             "best_single": names[bf]})
    t = pd.DataFrame(rows).sort_values("gain_pct", ascending=False)
    label = a.dataset + (f" [{a.fields}]" if a.fields else "")
    print(f"{label}: {len(ct)} chunks, {len(t)} cost models (w_ct = 1)")
    print(t.head(a.top).round(1).to_string(index=False))
    # realistic: every read also moves the bytes (w_io >= w_dt), and the best
    # single codec is a codec, not "store raw"
    real = t[(t.w_io >= t.w_dt) & (t.best_single != "store")]
    print(f"   realistic models (w_io >= w_dt, best single is a codec): top {min(5, len(real))}")
    print(real.head(5).round(1).to_string(index=False))
    for w_dt, w_io, bw in ((1.0, 1.0, "tiers"), (10.0, 10.0, 1.0)):
        r = t[(t.w_dt == w_dt) & (t.w_io == w_io) & (t.bw_GBs == bw)].iloc[0]
        print(f"   reference 1/{w_dt:g}/{w_io:g} at {bw}: {r.gain_pct:.1f}% ({r.best_single})")


if __name__ == "__main__":
    main()
