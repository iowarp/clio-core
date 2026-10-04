#!/usr/bin/env python3
"""Mean +- std over reps of the pipeline runs, per workload and policy.

    summarize.py --out DIR      (DIR/<workload>/runs.csv from pipeline.py)

Prints, per workload: end-to-end, producer and consumer seconds, the phase
split, the producer data's compression ratio, and each policy's change in
end-to-end time against storing uncompressed ("none") and against the best
single codec.
"""
import argparse
import glob
import json
import os

import pandas as pd

PHASES = ["prod_compress_s", "prod_d2h_s", "prod_write_s", "cons_read_s", "cons_h2d_s",
          "cons_decompress_s", "cons_analysis_s", "cons_out_compress_s", "cons_out_write_s"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    for f in sorted(glob.glob(os.path.join(a.out, "*", "runs.csv"))):
        d = pd.read_csv(f)
        w = d.workload.iloc[0]
        meta = json.load(open(os.path.join(os.path.dirname(f), "plan.json")))["meta"]
        print(f"\n== {w}: {d.frames.iloc[0]} frames, {d.prod_in.iloc[0] / 1e9:.2f} GB produced, "
              f"{d.rep.max()} reps; best single = {meta['best_setting']}; "
              f"disk {meta['bw_write_gbs']:.2f} / {meta['bw_read_gbs']:.2f} GB/s write / read")
        g = d.groupby("policy", sort=False)
        e2e = g.end_to_end_s.agg(["mean", "std"])
        print(f"{'policy':9s} {'end-to-end s':>15s} {'producer s':>11s} {'consumer s':>11s} "
              f"{'ratio':>7s} {'vs none':>8s} {'vs best':>8s}")
        for pol, r in e2e.iterrows():
            x = d[d.policy == pol]
            ratio = x.prod_in.sum() / x.prod_stored.sum()
            vs = lambda ref: (f"{100 * (r['mean'] / e2e.loc[ref, 'mean'] - 1):+.1f}%"  # noqa: E731
                              if ref in e2e.index else "")
            print(f"{pol:9s} {r['mean']:8.2f} +- {r['std']:4.2f} {x.prod_total_s.mean():11.2f} "
                  f"{x.cons_total_s.mean():11.2f} {ratio:7.3f} {vs('none'):>8s} {vs('best'):>8s}")
        print("phase means (s): " + ", ".join(p[:-2] for p in PHASES))
        for pol in e2e.index:
            print(f"  {pol:9s} " + " ".join(f"{d[d.policy == pol][p].mean():7.2f}" for p in PHASES))


if __name__ == "__main__":
    main()
