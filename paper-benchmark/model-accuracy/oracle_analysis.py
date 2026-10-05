#!/usr/bin/env python3
"""How much is there to gain from choosing a codec per chunk? Per workload, from
the exhaustive runs (every setting measured on every chunk).

    oracle_analysis.py [RUN_DIR ...] [--out PREFIX]

A chunk's measured compress time, decompress time and ratio do not depend on
the storage tier, so its cost at every tier follows from them: cost = compress
+ decompress + bytes / (ratio x bandwidth) (a setting that does not shrink the
chunk is stored raw: compress + bytes / bandwidth; storing raw: bytes /
bandwidth). For each tier (12 / 1 / 0.5 / 0.25 GB/s) and for the 4-tier mix
as run (each chunk at its own tier):
  oracle            sum over chunks of the cheapest setting per chunk
  best fixed        the single setting with the lowest total, and its total
  oracle gain       (best fixed - oracle) / best fixed, in %
  winners           share of chunks each setting wins (oracle distribution)
Writes PREFIX_gain.csv (one row per workload and tier) and PREFIX_winners.csv
(share of chunks per workload, tier and winning setting).
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

import eval_v2_workloads as ev

RUNS = "/mnt/nvme0/v2-work/runs"
TIERS = {"12 GB/s": 12e6, "1 GB/s": 1e6, "0.5 GB/s": 0.5e6, "0.25 GB/s": 0.25e6}


def measurements(run):
    """Chunk x setting arrays of compress ms, decompress ms, ratio; bytes;
    each chunk's run tier; setting names (store last)."""
    pred = pd.read_csv(os.path.join(run, "v2_pred.csv"), usecols=["blob", "bytes", "tier_bw"])
    m = pd.read_csv(os.path.join(run, "v2_measured.csv"))
    raw = (m.role == "primary") & ((m.comp_ms <= 0) | (m.ratio <= 1))
    m = m[~raw & (m.decomp_ms > 0)].drop_duplicates(["blob", "setting"])
    specs = sorted(m.spec.unique())
    piv = {k: m.pivot(index="blob", columns="spec", values=v).reindex(
        index=pred.blob, columns=specs).to_numpy(float)
        for k, v in (("ct", "comp_ms"), ("dt", "decomp_ms"), ("r", "ratio"))}
    return piv, pred.bytes.to_numpy(float), pred.tier_bw.to_numpy(float), specs


def costs(piv, nbytes, bw):
    """Chunk x (settings + store) cost at bandwidth bw (array per chunk)."""
    io = (nbytes / bw)[:, None]
    kept = piv["r"] > 1
    c = np.where(kept, piv["ct"] + piv["dt"] + io / piv["r"], piv["ct"] + io)
    return np.hstack([c, io])


def analyse(run):
    """Gain and winner rows for one exhaustive run."""
    base = os.path.basename(run.rstrip("/"))
    # A baseline-store directory (baselines/<ds>/exhaustive) or a run directory.
    ds = (os.path.basename(os.path.dirname(run.rstrip("/"))) if base == "exhaustive"
          else base.replace("_exhaustive", "").replace("_nolog", ""))
    piv, nbytes, run_bw, specs = measurements(run)
    names = specs + ["raw (store)"]
    all_names, _ = ev.settings_list()
    dropped = {all_names[k] for k in ev.DROPPED_SETTINGS}
    drop = [i for i, n in enumerate(names) if n in dropped]
    gain, winners = [], []
    tiers = dict(TIERS)
    tiers["4-tier mix (as run)"] = None
    for tier, bw in tiers.items():
        c = costs(piv, nbytes, run_bw if bw is None else bw)
        ok = ~np.isnan(c).any(axis=1)
        c = c[ok]
        sel = c.copy()   # the best single and the oracle select from the candidates
        sel[:, drop] = np.inf
        oracle = sel.min(axis=1).sum()
        totals = c.sum(axis=0)
        k = int(np.argmin(sel.sum(axis=0)))
        gain.append({"workload": ds, "tier": tier, "chunks": int(ok.sum()),
                     "oracle_ms": oracle, "best_fixed": names[k],
                     "best_fixed_ms": totals[k],
                     "oracle_gain_pct": 100 * (totals[k] - oracle) / totals[k],
                     "n_winning_settings": len(np.unique(sel.argmin(axis=1)))})
        share = pd.Series([names[j] for j in sel.argmin(axis=1)]).value_counts(normalize=True)
        winners += [{"workload": ds, "tier": tier, "setting": s, "share_pct": 100 * p}
                    for s, p in share.items()]
    return gain, winners


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--out", default=os.path.join(RUNS, "oracle"))
    a = ap.parse_args()
    runs = a.runs or sorted(p for p in glob.glob(os.path.join(RUNS, "*_exhaustive"))
                            if not os.path.basename(p).startswith(("smoke-", "vpic2k")))
    gain, winners = [], []
    for run in runs:
        if not os.path.exists(os.path.join(run, "v2_measured.csv")):
            continue
        g, w = analyse(run)
        gain += g
        winners += w
    g = pd.DataFrame(gain)
    pd.set_option("display.width", 220)
    mix = g[g.tier == "4-tier mix (as run)"].sort_values("oracle_gain_pct", ascending=False)
    print(mix[["workload", "chunks", "best_fixed", "oracle_gain_pct",
               "n_winning_settings"]].round(1).to_string(index=False))
    print("\noracle gain over the best fixed codec, per tier (%):")
    print(g.pivot(index="workload", columns="tier", values="oracle_gain_pct")
          .reindex(mix.workload)[list(TIERS) + ["4-tier mix (as run)"]].round(1).to_string())
    g.to_csv(a.out + "_gain.csv", index=False)
    pd.DataFrame(winners).to_csv(a.out + "_winners.csv", index=False)
    print("wrote", a.out + "_{gain,winners}.csv")


if __name__ == "__main__":
    main()
