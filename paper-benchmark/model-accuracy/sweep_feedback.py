#!/usr/bin/env python3
"""Sweep HCompress's feedback settings, so the reported number is not one pick.

    sweep_feedback.py --inputs DIR --out CSV [--jobs 8]
                      [--intervals 1,8,32,128] [--forget 1.0,0.98]
                      [--scopes executed,all]

The paper leaves the feedback interval n configurable and does not state a
forgetting factor, and both change what the baseline can do:

  n          how often the measured cost of the executed choice is folded in.
  forget     dlib rls's forget factor. At 1.0 (its default) every observation
             ever seen counts equally, so feedback moves a model already fitted
             on 401k profiler rows very little. Below 1.0 the fit tracks the
             workload in front of it.
  scope      `executed` is HCompress's own information budget -- one
             configuration per chunk. `all` is what our NN actually received in
             this campaign, where exploration was forced and all 32 were
             measured. Both are reported so the comparison cannot be accused of
             starving the baseline.

Reports the HCompress (+ feedback) row only: the other rows do not depend on
these settings.
"""
from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from accuracy_table import METRICS, SETTINGS, mape, medape, r2  # noqa: E402
from models_offline import PRED_TIME_FLOOR_MS, RATIO_CAP  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "hcompress_ccp_eval")


def one(job):
    setting, label, inputs, n, forget, scope = job
    d = os.path.join(inputs, setting)
    rows = pd.read_csv(os.path.join(d, "rows.csv"))
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "hc.csv")
        cmd = [TOOL, "--seed", os.path.join(inputs, "seed.csv"),
               "--eval", os.path.join(d, "eval.csv"), "--out", out,
               "--feedback-interval", str(n), "--forget", str(forget),
               "--feedback-scope", scope]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            return [{"setting": label, "n": n, "forget": forget, "scope": scope,
                     "metric": "ERROR", "detail": r.stderr.strip()[:200]}]
        hc = pd.read_csv(out)
    recs = []
    for metric, col, _ in METRICS:
        if metric == "psnr":
            continue  # HCompress has no quality output at all
        act = rows[col].to_numpy(dtype=float)
        key = {"ct_ms": "pred_ct_ms_fb", "dt_ms": "pred_dt_ms_fb",
               "ratio": "pred_ratio_fb"}[col]
        pr = hc[key].to_numpy(dtype=float)
        pr = (np.minimum(RATIO_CAP, pr) if col == "ratio"
              else np.maximum(PRED_TIME_FLOOR_MS, pr))
        m = np.isfinite(act) & (act > 0) & np.isfinite(pr)
        if m.sum() == 0:
            continue
        recs.append({"setting": label, "n": n, "forget": forget, "scope": scope,
                     "metric": metric, "n_rows": int(m.sum()),
                     "mape": mape(pr[m], act[m]), "median_ape": medape(pr[m], act[m]),
                     "r2": r2(pr[m], act[m]), "detail": ""})
    return recs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--intervals", default="1,8,32,128")
    ap.add_argument("--forget", default="1.0,0.98")
    ap.add_argument("--scopes", default="executed,all")
    a = ap.parse_args()

    ns = [int(x) for x in a.intervals.split(",")]
    fs = [float(x) for x in a.forget.split(",")]
    scopes = a.scopes.split(",")
    jobs = [(k, lbl, a.inputs, n, f, s)
            for (k, lbl) in SETTINGS
            if os.path.exists(os.path.join(a.inputs, k, "eval.csv"))
            for n, f, s in itertools.product(ns, fs, scopes)]
    print(f"{len(jobs)} run(s) over {len(ns)}x{len(fs)}x{len(scopes)} settings, "
          f"{a.jobs} at a time")
    out = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for i, recs in enumerate(ex.map(one, jobs), 1):
            out.extend(recs)
            print(f"  {i}/{len(jobs)}", end="\r", flush=True)
    df = pd.DataFrame(out)
    df.to_csv(a.out, index=False, float_format="%.4f")
    print(f"\nwrote {a.out}")
    err = df[df.metric == "ERROR"] if "metric" in df else df.iloc[:0]
    if not err.empty:
        print(err.to_string(index=False))
    # Does the interval matter at all? That is the question the task asks.
    piv = (df[df.metric != "ERROR"]
           .pivot_table(index=["setting", "metric", "scope", "forget"],
                        columns="n", values="mape"))
    print(piv.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
