#!/usr/bin/env python3
"""The end-to-end exhaustive table: every setting's measured single-codec run.

    collect_single_codecs.py DATASET [--settings all|2,34,...] [--tag-prefix km4s]
                             [--bw 520000] [--w 0,1,1 --w 1,4,5] [--out-dir DIR]

Each run runs/<DATASET>_fixed_<prefix><setting> (run_exhaustive_e2e.sh, or
run_v2_workloads.sh fixed with READS and KMEANS) stored every chunk with one
setting, wrote once and read every chunk back R times through the k-means
consumer. One row per setting with a complete run:
  measured     e2e_s (the application's wall-clock time: write_s + read_s +
               kmeans_s), each read pass, ratio, stored bytes, compress and
               decompress times (R passes), model_pfs_s (this run's compress +
               decompress + (1 + R) x stored bytes / BW)
  cost         for each --w: w_ct x compress + w_dt x decompress of one pass +
               w_io x stored bytes / BW (s), from this run's own times, and
               the rank of that cost (1 = the best single codec under w);
               rank_e2e ranks the measured application time
  csv_*        the same setting in the stored exhaustive search (codec times
               and ratio only; it has no application time): compress,
               decompress (R passes), ratio, stored bytes, the modelled time,
               the cost and rank under each --w, and the chunks it did not
               measure
  checks       chunks stored with the setting, chunks stored raw (ratio <= 1)
               and failed, bit-exact reads, the largest relative difference of
               the centroids to the first run's and whether it is <= 1e-12 (the
               GPU's atomic sums can change the last bit of a double)
The ranks cover the candidate settings in the table only
(eval_v2_workloads.candidate_settings; a dropped setting's row has no rank).
Output: runs/<DATASET>_exhaustive_e2e.csv (and a copy in --out-dir).
"""
import argparse
import os
import re
import shutil
import sys

import numpy as np
import pandas as pd

import compare_kmeans_runs as ck
import eval_v2_workloads as ev

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sim-tuning"))
import costmodel_whatif as cw  # noqa: E402


def wlabel(w):
    """@return the column label of weights w, e.g. w0-1-1."""
    return "w" + "-".join(f"{x:g}" for x in w)


def csv_columns(ds, settings, reads, bw):
    """Each setting's codec-only values from the stored exhaustive search.

    @param ds       dataset
    @param settings setting indices
    @param reads    number of reads R
    @param bw       bandwidth of the model, bytes per ms
    @return {setting: {csv_* column: value}}
    """
    _, store, order, meas, _ = cw.rl.load(ds)
    nb = order.bytes.to_numpy(float)
    ct, dt, st = cw.outcomes(meas, nb, store)
    out = {}
    for s in settings:
        ok = ~np.isnan(st[:, s])
        comp, dec = ct[ok, s].sum() / 1e3, reads * dt[ok, s].sum() / 1e3
        out[s] = {"csv_compress_s": comp, "csv_decompress_s": dec,
                  "csv_ratio": nb[ok].sum() / st[ok, s].sum(),
                  "csv_stored_GB": st[ok, s].sum() / 1e9,
                  "csv_model_s": comp + dec + (1 + reads) * st[ok, s].sum() / bw / 1e3,
                  "csv_unmeasured_chunks": int((~ok).sum())}
    return out


def setting_row(run, s, name, bw, weights):
    """One setting's measured row and centroids.

    @param run     run directory
    @param s       setting index
    @param name    setting name
    @param bw      bandwidth of the cost model, bytes per ms
    @param weights list of (w_ct, w_dt, w_io)
    @return (row dict, centroids by field)
    """
    row, cents = ck.run_row(run, name, "fixed", bw)
    text = open(os.path.join(run, "stdout.log")).read()
    counts = re.search(r"compressed: (\d+)\s+stored raw: (\d+)\s+failed: (\d+)", text)
    stored = ck.phase_totals(run, row["reads"])["stored_bytes"]
    meas = os.path.join(run, "v2_measured.csv")
    with_s = (int((pd.read_csv(meas, usecols=["setting"]).setting == s).sum())
              if os.path.exists(meas) else np.nan)
    row.update({"setting": s, "stored_GB": stored / 1e9, "chunks_with_setting": with_s,
                "compressed": int(counts.group(1)) if counts else np.nan,
                "stored_raw": int(counts.group(2)) if counts else np.nan,
                "failed": int(counts.group(3)) if counts else np.nan,
                "verified_reads": text.count("VERIFIED:"),
                "run": os.path.basename(os.path.realpath(run))})
    for w in weights:
        row[f"cost_{wlabel(w)}_s"] = (w[0] * row["compress_s"]
                                      + w[1] * row["decompress_s"] / row["reads"]
                                      + w[2] * stored / bw / 1e3)
    return row, cents


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dataset")
    ap.add_argument("--settings", default="all", help="all, or a comma list of indices")
    ap.add_argument("--tag-prefix", default="km4s")
    ap.add_argument("--bw", type=float, default=520000.0, help="bytes per ms")
    ap.add_argument("--w", action="append", default=None,
                    help="cost weights w_ct,w_dt,w_io (repeat; default 0,1,1 and 1,4,5)")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    names, _ = ev.settings_list()
    settings = (range(len(names)) if a.settings == "all"
                else [int(x) for x in a.settings.split(",")])
    weights = [tuple(float(x) for x in w.split(",")) for w in (a.w or ["0,1,1", "1,4,5"])]
    rows, ref, missing = [], None, []
    for s in settings:
        run = os.path.join(ck.RUNS, f"{a.dataset}_fixed_{a.tag_prefix}{s}")
        log = os.path.join(run, "stdout.log")
        if not os.path.exists(log) or "READ check" not in open(log).read():
            missing.append(str(s))
            continue
        row, cents = setting_row(run, s, names[s], a.bw, weights)
        if ref is None:
            ref = cents
        row["centroids_rel_diff"] = max((abs(u - v) / max(abs(u), 1e-300)
                                         for f, c in ref.items()
                                         for u, v in zip(c, cents.get(f, []))), default=np.nan)
        row["centroids_equal_first"] = bool(row["centroids_rel_diff"] <= 1e-12)
        rows.append(row)
    if missing:
        print("settings with no complete run yet:", ",".join(missing))
    if not rows:
        sys.exit("no complete run")
    t = pd.DataFrame(rows)
    csv = csv_columns(a.dataset, t.setting.tolist(), int(t.reads.iloc[0]), a.bw)
    t = t.join(pd.DataFrame([csv[s] for s in t.setting], index=t.index))
    t["candidate"] = t.setting.isin(ev.candidate_settings(len(names)))

    def rank(col):
        """Rank among the candidate rows (1 = lowest); none for the others."""
        return t[col].where(t.candidate).rank(method="min").astype("Int64")
    t["rank_e2e"] = rank("e2e_s")
    reads = t.reads.iloc[0]
    for w in weights:
        t[f"rank_{wlabel(w)}"] = rank(f"cost_{wlabel(w)}_s")
        t[f"csv_cost_{wlabel(w)}_s"] = (w[0] * t.csv_compress_s + w[1] * t.csv_decompress_s / reads
                                        + w[2] * t.csv_stored_GB * 1e9 / a.bw / 1e3)
        t[f"csv_rank_{wlabel(w)}"] = rank(f"csv_cost_{wlabel(w)}_s")
    out = os.path.join(ck.RUNS, f"{a.dataset}_exhaustive_e2e.csv")
    t.to_csv(out, index=False)
    cols = (["setting", "method", "e2e_s", "rank_e2e", "write_s", "read_s", "kmeans_s",
             "ratio", "compress_s", "decompress_s", "csv_compress_s", "csv_decompress_s",
             "csv_ratio"] + [f"rank_{wlabel(w)}" for w in weights]
            + ["stored_raw", "failed", "verified_reads", "centroids_equal_first"])
    with pd.option_context("display.width", 250, "display.max_colwidth", 30):
        print(t[cols].round(3).to_string(index=False))
    for w in weights:
        b = t.loc[t[f"rank_{wlabel(w)}"].idxmin()]
        c = t.loc[t[f"csv_rank_{wlabel(w)}"].idxmin()]
        print(f"best single by cost {wlabel(w)}: {b.method} (e2e {b.e2e_s:.1f} s, "
              f"ratio {b.ratio:.3f}); by the exhaustive CSV's cost: {c.method}")
    print(f"{len(t)} of {len(settings)} settings; ranks cover the {int(t.candidate.sum())} "
          f"candidate settings among them\nwrote", out)
    if a.out_dir:
        shutil.copy(out, os.path.join(a.out_dir, os.path.basename(out)))
        print("copied to", a.out_dir)


if __name__ == "__main__":
    main()
