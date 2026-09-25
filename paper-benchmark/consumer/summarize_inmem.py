#!/usr/bin/env python3
"""Summarize an inmem.sh campaign: per arm, the write, the timed read-backs of
the consumer's fields into device memory, and the consumer's end-to-end time.

  summarize_inmem.py <out dir>     # writes <out>/summary.csv, prints a table

WRITE is the driver's stage+compress time (the input read excluded, as in
cluster.sh; the H2D staging included for every arm alike). READ is each timed
read-back's get+decompress wall clock (windowed: first request to last
completion). END-TO-END is one write plus `reads` read-backs, each at the arm's
median read -- the consumer that writes once and clusters five times. An arm's
numbers count only when every timed read verified all its blobs. The bound
column reports the untimed check read: `ok`, or the worst error when it exceeds
the bound (cuSZp3 does, by float32 rounding on values near 1e5 -- the dagger of
the cluster figures), which is reported, not hidden.

cuSZ builds and releases a resource manager for every chunk it compresses
(cusz_setup.csv). WRITE counts it, as measured; WRITE-SETUP removes the union
of those spans inside the write window, as the cluster figures do. With one
chunk in flight (VPIC) the union is exact; with several (Nyx) it also removes
time other chunks spent working, so it is a lower bound on the write there.
"""
import csv
import json
import os
import re
import statistics as st
import sys

import summarize_cluster as sc

ARMS = ["Baseline", "Best fixed nvCOMP", "ndzip", "cuSZp3", "cuSZ", "NeuroPress"]


def slug(label):
    """Result-dir name of an arm label, as inmem.sh derives it."""
    return re.sub("_+", "_", re.sub("[^a-z0-9]", "_", label.lower())).rstrip("_")


def parse(path):
    """(write s, ratio, [read s], read bytes, all verified, bound ok) of a run."""
    log = open(path).read()
    t = re.search(r"stage\+compress ([\d.]+) s", log)
    r = re.search(r"stored ratio ([\d.]+)", log)
    if not (t and r):
        return None
    reads, nbytes, ok = [], 0, True
    parts = re.split(r"^READ ", log, flags=re.M)[1:]
    for part in parts:
        v = re.search(r"^(VERIFIED|FAILED): (\d+) of (\d+)", part, re.M)
        g = re.search(r"get\+decompress: ([\d.]+) ms for (\d+) B", part)
        good = bool(v and v.group(1) == "VERIFIED" and v.group(2) == v.group(3))
        if part.startswith("check"):
            ok = ok and good
            continue
        ok = ok and good
        if g:
            reads.append(float(g.group(1)) / 1e3)
            nbytes = int(g.group(2))
    bound = re.search(r"^BOUND (OK|FAILED).*worst max\|err\|=([\d.e+-]+)", log, re.M)
    bound_s = "n/a" if bound is None else ("ok" if bound.group(1) == "OK" else f"max {float(bound.group(2)):.3g}")
    return float(t.group(1)), float(r.group(1)), reads, nbytes, ok, bound_s


def setup_s(run_dir):
    """Union of cuSZ's per-chunk manager spans inside the write window, s."""
    path = os.path.join(run_dir, "stdout.log")
    w = re.search(r"window: start_ns (\d+)\s+end_ns (\d+)", open(path).read())
    if not w or not os.path.exists(os.path.join(run_dir, "cusz_setup.csv")):
        return 0.0
    iv = sc.intervals(os.path.join(run_dir, "cusz_setup.csv"), "start_ns", "ms",
                      int(w.group(1)), int(w.group(2)))
    return sc.union_s(iv)


def main(out):
    run = json.load(open(os.path.join(out, "run.json")))
    n_reads = int(run["reads"])
    rows = []
    # Arms beyond the standard set (inmem.sh EXTRA_ARMS) are named by their dir.
    extra = sorted(x for x in os.listdir(os.path.join(out, "runs"))
                   if x not in {slug(a) for a in ARMS})
    for arm in ARMS + extra:
        d = os.path.join(out, "runs", slug(arm))
        if not os.path.isdir(d):
            continue
        dirs = [os.path.join(d, r) for r in sorted(os.listdir(d))
                if os.path.exists(os.path.join(d, r, "stdout.log"))]
        runs = [x + (setup_s(rd),) for x, rd in ((parse(os.path.join(rd, "stdout.log")), rd)
                                                for rd in dirs) if x]
        good = [x for x in runs if x[4]]
        if not good:
            rows.append({"arm": arm, "valid_reps": 0, "reps": len(runs)})
            continue
        writes = [x[0] for x in good]
        w_nosetup = st.median(x[0] - x[6] for x in good)
        reads = [s for x in good for s in x[2]]
        w, rd = st.median(writes), st.median(reads)
        nbytes = good[0][3]
        rows.append({"arm": arm, "valid_reps": len(good), "reps": len(runs),
                     "ratio": round(st.median(x[1] for x in good), 2),
                     "write_med_s": round(w, 3), "write_min_s": round(min(writes), 3),
                     "write_max_s": round(max(writes), 3),
                     "read_med_s": round(rd, 4), "read_min_s": round(min(reads), 4),
                     "read_max_s": round(max(reads), 4), "reads_n": len(reads),
                     "read_gbps": round(nbytes / rd / 1e9, 2),
                     "e2e_s": round(w + n_reads * rd, 3),
                     "write_minus_setup_s": round(w_nosetup, 3),
                     "e2e_minus_setup_s": round(w_nosetup + n_reads * rd, 3),
                     "bound": ",".join(sorted({x[5] for x in good}))})
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k)
                  if k in rows[0] else 99)
    with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    print(f"{run['workload']}: RAM tier, reads into GPU memory, {run['window']} in flight, "
          f"{run['threads']} workers, {n_reads} reads per rep, eb {run['eb']}")
    print(f"{'arm':20s} {'reps':>5s} {'ratio':>6s} {'write s':>16s} {'read s':>18s} {'GB/s':>6s} {'1W+'+str(n_reads)+'R s':>8s}")
    for r in rows:
        if not r.get("valid_reps"):
            print(f"{r['arm']:20s}  no valid rep ({r['reps']} run)")
            continue
        print(f"{r['arm']:20s} {r['valid_reps']:>2d}/{r['reps']:<2d} {r['ratio']:6.2f} "
              f"{r['write_med_s']:6.3f} [{r['write_min_s']:.3f}-{r['write_max_s']:.3f}] "
              f"{r['read_med_s']:7.4f} [{r['read_min_s']:.3f}-{r['read_max_s']:.3f}] "
              f"{r['read_gbps']:6.1f} {r['e2e_s']:8.3f}  bound {r['bound']}"
              + (f"   [cuSZ setup removed: write {r['write_minus_setup_s']:.3f}, 1W+{n_reads}R {r['e2e_minus_setup_s']:.3f}]"
                 if r['write_minus_setup_s'] < r['write_med_s'] - 1e-3 else ""))
    np_row = next((r for r in rows if r["arm"] == "NeuroPress" and r.get("valid_reps")), None)
    if np_row:
        for r in rows:
            if r["arm"] != "NeuroPress" and r.get("valid_reps"):
                print(f"  NeuroPress vs {r['arm']:18s}: write x{r['write_med_s'] / np_row['write_med_s']:.2f}"
                      f"  read x{r['read_med_s'] / np_row['read_med_s']:.2f}"
                      f"  end-to-end x{r['e2e_s'] / np_row['e2e_s']:.2f}"
                      + (f" (setup removed x{r['e2e_minus_setup_s'] / np_row['e2e_s']:.2f})"
                         if r['write_minus_setup_s'] < r['write_med_s'] - 1e-3 else "")
                      + "   (>1 = NeuroPress faster)")
    print(f"-> {os.path.join(out, 'summary.csv')}")


if __name__ == "__main__":
    main(sys.argv[1])
