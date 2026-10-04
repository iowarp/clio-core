#!/usr/bin/env python3
"""Score NeuroPress v2 adaptation runs (run_v2_adapt.sh) against the sweep.

    eval_v2_adapt.py --dataset vpic RUN_DIR [RUN_DIR ...] [--out PREFIX]

Each run streamed the dataset's measured chunks through Clio in order and
logged, per chunk, v2's predictions for all 45 settings BEFORE learning from
that chunk (v2_pred.csv), plus the setting finally stored (selection.csv).
Ground truth is the codec sweep's measurement of every setting on the same
chunk bytes (~/np-newsweep/ref-<dataset>-...).

Per chunk, in stream order:
  ape_ct / ape_dt / ape_ratio  mean |pred - true| / true over the 45 settings
  regret_pick    true cost of v2's own pick / best true cost - 1
  regret_stored  the same for the setting finally stored (exploration can
                 replace the pick)
Cost = compress ms + decompress ms + bytes / (ratio * 1 GB/s), as in the runs.
Writes PREFIX_chunks.csv and PREFIX_summary.csv; prints the summary (first
and second half of the stream, and the whole stream).
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "codec-sweep"))
import eval_nn_v2_real as ev  # noqa: E402

SWEEPS = {"vpic": "~/np-newsweep/ref-vpic-126-2000",
          "nyx": "~/np-newsweep/ref-nyx-256-2000"}
BW = 1e6  # bytes per ms (1 GB/s), CLIO_NEUROPRESS_COST_BW in the runs


def sweep_key(blob):
    """'fields/plt00000__x.f32__c0/chunk_0' -> 'plt00000/x.f32#0'."""
    name = blob.split("/")[1]
    ts, field, chunk = name.split("__")
    return f"{ts}/{field}#{chunk[1:]}"


def cost(ct, dt, ratio, nbytes):
    """Cost in ms per (chunk, setting)."""
    return ct + dt + nbytes[:, None] / (ratio * BW)


def score_run(run, settings, truth_keys, nbytes_t, truth):
    """Per-chunk scores of one run, in stream order."""
    pred = pd.read_csv(os.path.join(run, "v2_pred.csv"))
    idx = {k: i for i, k in enumerate(truth_keys)}
    pred = pred.assign(key=pred.blob.map(sweep_key))
    pred = pred[pred.key.isin(idx)].reset_index(drop=True)
    rows = np.array([idx[k] for k in pred.key])
    t = np.exp(truth[rows])                       # chunk x S x 3
    n = nbytes_t[rows]
    S = len(settings)
    p = np.stack([pred[[f"ct{s}", f"dt{s}", f"r{s}"]].to_numpy(float)
                  for s in range(S)], axis=1)      # chunk x S x 3
    valid = np.isfinite(p).all(axis=2)
    ape = np.abs(p - t) / t
    ape[~valid] = np.nan
    tc = cost(t[..., 0], t[..., 1], t[..., 2], n)
    pc = np.where(valid, cost(p[..., 0], p[..., 1], p[..., 2], n), np.inf)
    best = tc.min(axis=1)
    pick = pc.argmin(axis=1)
    r = np.arange(len(pick))
    sel = pd.read_csv(os.path.join(run, "selection.csv"))
    final = sel.groupby("blob").tail(1).set_index("blob")
    store = settings.index("store")
    stored = np.array([
        (int(final.loc[b, "preset"]) if int(final.loc[b, "wire_lib"]) == 24
         else store) if b in final.index else -1 for b in pred.blob])
    stored_cost = np.where(stored >= 0, tc[r, np.maximum(stored, 0)], np.nan)
    return pd.DataFrame({
        "i": r, "blob": pred.blob, "updates": pred.updates,
        "ape_ct": np.nanmean(ape[..., 0], axis=1),
        "ape_dt": np.nanmean(ape[..., 1], axis=1),
        "ape_ratio": np.nanmean(ape[..., 2], axis=1),
        "pick": [settings[k] for k in pick],
        "regret_pick": tc[r, pick] / best - 1,
        "stored": [settings[k] if k >= 0 else "" for k in stored],
        "regret_stored": stored_cost / best - 1})


def summarise(name, d):
    """First half, second half and whole stream."""
    h = len(d) // 2
    out = []
    for part, x in (("first half", d.iloc[:h]), ("second half", d.iloc[h:]),
                    ("all", d)):
        out.append({"run": name, "part": part, "chunks": len(x),
                    "mape_ct": x.ape_ct.mean(), "mape_dt": x.ape_dt.mean(),
                    "mape_ratio": x.ape_ratio.mean(),
                    "medape_ct": x.ape_ct.median(),
                    "medape_ratio": x.ape_ratio.median(),
                    "regret_pick": x.regret_pick.mean(),
                    "regret_stored": x.regret_stored.mean(),
                    "med_regret_pick": x.regret_pick.median(),
                    "updates_end": int(x.updates.iloc[-1])})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True, choices=sorted(SWEEPS))
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    meta = json.load(open(ev.WEIGHTS[:-5] + ".json"))
    settings = [ev.canon(s) for s in meta["outputs"]["settings"]]
    keys, nbytes, truth = ev.load_measured(os.path.expanduser(SWEEPS[a.dataset]),
                                           settings)
    summary, chunks = [], []
    for run in a.runs:
        name = os.path.basename(run.rstrip("/"))
        d = score_run(run, settings, keys, nbytes, truth)
        chunks.append(d.assign(run=name))
        summary += summarise(name, d)
    s = pd.DataFrame(summary)
    pd.set_option("display.width", 220)
    print((s.set_index(["run", "part"])).round(4).to_string())
    if a.out:
        s.to_csv(a.out + "_summary.csv", index=False)
        pd.concat(chunks).to_csv(a.out + "_chunks.csv", index=False)
        print("wrote", a.out + "_{summary,chunks}.csv")


if __name__ == "__main__":
    main()
