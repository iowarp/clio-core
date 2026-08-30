#!/usr/bin/env python3
"""Backfill the primary row's measured quality into explore.csv.

SUPERSEDED for logs written after the primary-quality fix: the primary now
carries its own measurement in explore.csv and quality_measured reads 1, so
there is nothing to backfill. Kept for logs written BEFORE that fix, where
every primary row has -1 in the five measured columns and the measurement
exists only in selection.csv.quality. Running it on a current log is a no-op
(it reports 0 backfilled).

In those older logs the primary's quality reached only selection.csv.quality,
keyed by blob, one row per chunk. The join below is exact: the sidecar holds
the PRIMARY's configuration, so a row whose (quantized, shuffle) does not match
is refused rather than merged.

This joins the two, so every row of the output carries a measurement and
`quality_measured` is 1 everywhere. Writes a NEW file; explore.csv is the
raw record and is left untouched.
"""
import csv, sys, os

MEAS = ["meas_rmse", "meas_max_error", "meas_psnr_db",
        "meas_ssim", "meas_ssim_deviation"]
SIDE = ["rmse", "max_error", "psnr_db", "ssim", "ssim_deviation"]

def main(run_dir, out=None):
    ex = os.path.join(run_dir, "explore.csv")
    sq = os.path.join(run_dir, "selection.csv.quality")
    out = out or os.path.join(run_dir, "explore_full.csv")
    q = {r["blob"]: r for r in csv.DictReader(open(sq))}

    rows = list(csv.DictReader(open(ex)))
    hdr = list(rows[0].keys())
    filled = missing = mismatch = 0
    for r in rows:
        if r["quality_measured"] != "0":
            continue
        s = q.get(r["blob"])
        if s is None:
            missing += 1
            continue
        # the sidecar measures the PRIMARY's config; refuse a wrong join
        if (s["quantized"], s["shuffle"]) != (r["quantize"], r["shuffle"]):
            mismatch += 1
            continue
        for m, k in zip(MEAS, SIDE):
            r[m] = s[k]
        r["quality_measured"] = "1"
        filled += 1

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        w.writeheader()
        w.writerows(rows)

    still = sum(1 for r in rows if r["quality_measured"] == "0")
    print(f"rows            {len(rows)}")
    print(f"backfilled      {filled}")
    print(f"no sidecar row  {missing}")
    print(f"config mismatch {mismatch}")
    print(f"still unmeasured{still:>5}")
    print(f"wrote {out}")
    return 0 if (missing == 0 and mismatch == 0 and still == 0) else 1

if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
