#!/usr/bin/env python3
"""Collect and check the final paper-evaluation runs (run_final_suite.sh).

    final_check.py [--fig-root DIR]

Per workload of final_config.py and iteration, the four options are measured in the standard
way (compare_parallel_runs.parallel_row: write window + timed read passes, ratio, bit-exact
checks): best static (fixed), NeuroPress (learn), cost oracle (oracle) and time oracle
(oracle, tag + "to"). Writes FIG_ROOT/final_runs.csv (one row per run), FIG_ROOT/final_summary.csv
(per workload and option: mean, min and max over the iterations of the time, the ratio and
their change against the best static of the same iteration) and FIG_ROOT/final_checks.txt.

Checks, per run: last read bit-exact and timed reads complete; READS k-means iterations done
with no failed chunk; read I/O per stored GB and read within READ_SPEED_TOLERANCE of the median
of the same workload and option over the iterations (otherwise the run is flagged as disturbed;
each option stores a different data layout, and the read speed per GB depends on it). Per workload (iteration means):
NeuroPress faster than the best static and a higher ratio; the time oracle the fastest option
and a ratio at least NeuroPress's; and, when FIG_ROOT/<workload>/tiering_<workload>.csv exists
(tiering_final.py), the tiered NeuroPress gain of the same sign as the measured one and within
8 points of it. Exit status 1 when any check fails.
"""
import argparse
import os
import re

import numpy as np
import pandas as pd

import compare_parallel_runs as cp
import final_config as fc

RUNS = "/mnt/nvme0/v2-work/runs"
HERE = os.path.dirname(os.path.abspath(__file__))
OPTIONS = (("best static", "fixed", ""), ("NeuroPress", "learn", ""),
           ("cost oracle", "oracle", ""), ("time oracle", "oracle", "to"))
TIER_TOLERANCE = 8.0       # points between the tiered and the measured NeuroPress gain


def tag(ds, w, bw, it):
    """@return the run tag of one workload and iteration (as run_final_suite.sh names it)."""
    return f"km{fc.READS}b{fc.bw_label(bw)}gfin{it}p1i1w{'-'.join(f'{x:g}' for x in w)}"


def run_extras(run):
    """@return (read I/O s per stored GB and read, k-means iterations done, k-means failures)."""
    p = pd.read_csv(os.path.join(run, "phases.csv"), usecols=["path", "io_ms", "stored_bytes"])
    w, r = p[p.path == "write"], p[p.path != "write"]
    gb = w.stored_bytes.sum() / 1e9
    reads = len(r) / max(len(w), 1)
    log = open(os.path.join(run, "stdout.log")).read()
    done = len(re.findall(rf"^KMEANS iteration \d+/{fc.READS} done: \d+ chunk\(s\) skipped, 0 failed", log, re.M))
    failed = len(re.findall(r"^KMEANS iteration \d+/\d+ done: .* [1-9]\d* failed", log, re.M))
    return r.io_ms.sum() / 1e3 / gb / reads, done, failed


def collect():
    """@return one row per finished run of every workload, iteration and option."""
    rows = []
    for wl, (ds, w, bw) in fc.WORKLOADS.items():
        for it in range(1, fc.ITERATIONS + 1):
            t = tag(ds, w, bw, it)
            for name, mode, suffix in OPTIONS:
                run = os.path.join(RUNS, f"{ds}-p0_{mode}_{t}{suffix}")
                if not os.path.exists(os.path.join(run, "stdout.log")):
                    continue
                row = cp.parallel_row(ds, 1, mode, t + suffix)
                io_gb, km_done, km_failed = run_extras(run)
                rows.append({"workload": wl, "iteration": it, "option": name, "run": os.path.basename(run),
                             "model": fc.model_name(w), "cost_bw_GBs": bw / 1e6,
                             "app_s": row["app_s"], "write_s": row["write_s"], "read_s": row["read_s"],
                             "ratio": row["ratio"], "digest_ok": row["digest_ok"], "timed_ok": row["timed_ok"],
                             "kmeans_done": km_done, "kmeans_failed": km_failed, "read_io_s_per_GB": io_gb})
    return pd.DataFrame(rows)


def summarize(t):
    """@return per workload and option: mean / min / max of time, ratio and change vs best static."""
    base = t[t.option == "best static"].set_index(["workload", "iteration"])
    idx = list(zip(t.workload, t.iteration))
    ok = [k in base.index for k in idx]
    t = t[ok].copy()
    idx = list(zip(t.workload, t.iteration))
    t["time_vs_static_pct"] = 100 * (t.app_s.to_numpy() / base.loc[idx, "app_s"].to_numpy() - 1)
    t["ratio_vs_static_pct"] = 100 * (t.ratio.to_numpy() / base.loc[idx, "ratio"].to_numpy() - 1)
    g = t.groupby(["workload", "option"], sort=False)
    out = g.agg(runs=("app_s", "size"), app_s=("app_s", "mean"), app_s_min=("app_s", "min"), app_s_max=("app_s", "max"),
                ratio=("ratio", "mean"), time_vs_static_pct=("time_vs_static_pct", "mean"),
                time_vs_static_min=("time_vs_static_pct", "min"), time_vs_static_max=("time_vs_static_pct", "max"),
                ratio_vs_static_pct=("ratio_vs_static_pct", "mean"),
                ratio_vs_static_min=("ratio_vs_static_pct", "min"), ratio_vs_static_max=("ratio_vs_static_pct", "max"))
    return out.reset_index(), t


def check(t, s, fig_root):
    """@return list of (check, ok, detail) lines."""
    out = []
    for _, r in t.iterrows():
        name = f"{r.workload} it{r.iteration} {r.option}"
        out.append((f"{name}: bit-exact and timed reads", bool(r.digest_ok and r.timed_ok), ""))
        out.append((f"{name}: k-means {fc.READS} iterations, 0 failed", r.kmeans_done == fc.READS and r.kmeans_failed == 0,
                    f"{r.kmeans_done} done, {r.kmeans_failed} failed"))
    med = t.groupby(["workload", "option"]).read_io_s_per_GB.median()
    for _, r in t.iterrows():
        dev = r.read_io_s_per_GB / med[(r.workload, r.option)] - 1
        out.append((f"{r.workload} it{r.iteration} {r.option}: read speed normal (not disturbed)",
                    abs(dev) <= fc.READ_SPEED_TOLERANCE,
                    f"{r.read_io_s_per_GB:.3f} s/GB per read, {100 * dev:+.1f}% vs the option's median"))
    for wl in fc.WORKLOADS:
        x = s[s.workload == wl].set_index("option")
        if not {"best static", "NeuroPress", "time oracle"} <= set(x.index):
            out.append((f"{wl}: all options measured", False, "missing options"))
            continue
        npv, to = x.loc["NeuroPress"], x.loc["time oracle"]
        out.append((f"{wl}: NeuroPress faster than the best static", npv.time_vs_static_pct < 0, f"{npv.time_vs_static_pct:+.1f}%"))
        out.append((f"{wl}: NeuroPress ratio higher than the best static", npv.ratio_vs_static_pct > 0, f"{npv.ratio_vs_static_pct:+.1f}%"))
        others = x.drop(index="time oracle")
        out.append((f"{wl}: time oracle is the fastest option", to.app_s < others.app_s.min(),
                    f"time oracle {to.app_s:.1f} s, fastest other {others.app_s.min():.1f} s ({others.app_s.idxmin()})"))
        out.append((f"{wl}: time oracle ratio at least NeuroPress's", to.ratio >= npv.ratio, f"{to.ratio:.2f} vs {npv.ratio:.2f}"))
        tf = os.path.join(fig_root, wl.lower(), f"tiering_{wl.lower()}.csv")
        if os.path.exists(tf):
            tt = pd.read_csv(tf)
            tt = tt[tt.main].set_index("config").time_s
            tier_gain = 100 * (tt["NeuroPress + tiering"] / tt["Best static + tiering"] - 1)
            same = np.sign(tier_gain) == np.sign(npv.time_vs_static_pct)
            close = abs(tier_gain - npv.time_vs_static_pct) <= TIER_TOLERANCE
            out.append((f"{wl}: tiering agrees with the measured run", bool(same and close),
                        f"tiered {tier_gain:+.1f}% vs measured {npv.time_vs_static_pct:+.1f}%"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fig-root", default=os.path.join(HERE, "..", "figures", "new-workloads", "sim-tuning", "final"))
    a = ap.parse_args()
    os.makedirs(a.fig_root, exist_ok=True)
    t = collect()
    if t.empty:
        print("no final runs found")
        return 1
    s, t = summarize(t)
    t.to_csv(os.path.join(a.fig_root, "final_runs.csv"), index=False)
    s.to_csv(os.path.join(a.fig_root, "final_summary.csv"), index=False)
    res = check(t, s, a.fig_root)
    lines = [f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({d})" if d else "") for name, ok, d in res]
    with open(os.path.join(a.fig_root, "final_checks.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    bad = [x for x in lines if x.startswith("FAIL")]
    print(f"{len(lines) - len(bad)} of {len(lines)} checks pass")
    for x in bad:
        print(x)
    with pd.option_context("display.width", 200):
        print(s[["workload", "option", "runs", "app_s", "ratio", "time_vs_static_pct", "ratio_vs_static_pct"]].round(2).to_string(index=False))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
