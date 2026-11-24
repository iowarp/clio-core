"""Summarise a benchmark directory: end to end across schedules, and best/worst case per workload.

    python3 paper_compare.py <dir with <policy>_s<seed>_r<rep>.jsonl> [--grid grid.jsonl]

1. End to end. For every schedule, each policy's total time (I/O + recomposition + blocking
   decision stalls, as in e2e_compare) is averaged over reps and divided into the best fixed
   setup's time on that schedule. Across schedules: geometric-mean speedup, min/max, schedules
   won, and the safety counts (lost writes, search misses) summed over every run.
2. Per workload (phase). Best and worst case = the fastest and slowest composition that meets the
   phase's requirements in the measured grid (mean window time). Each policy's mean window time
   in that phase is placed between them: score 1 = as fast as the best case, 0 = as slow as the
   worst (it can fall outside [0, 1]).
"""

import argparse
import glob
import json
import math
import os
import re
import statistics
from collections import defaultdict

import compose
from controller import DECLARED

BASE = "fixed-best"
ORDER = ["fixed-worst", BASE, "heuristic2", "agentjev-async", "qwen3-async", "agentjev-blocking",
         "qwen3-blocking", "oracle"]


def total(rows):
    io = sum(r["wall_s"] for r in rows)
    rc = sum(r.get("recompose_s", 0) for r in rows)
    stall = sum(r.get("decision_latency_s", 0) for r in rows if not r.get("async"))
    return io + rc + stall


def grid_cases(path):
    """{phase: (best_s, best_name, worst_s, worst_name)} over compositions meeting the requirements."""
    agg = defaultdict(lambda: {"t": [], "lost": 0})
    for l in open(path):
        r = json.loads(l)
        a = agg[(r["phase"], compose.name(r["composition"]))]
        a["t"].append(r["wall_s"])
        a["lost"] += r.get("failed_puts", 0)
        a["c"] = r["composition"]
    out = {}
    for ph in {k[0] for k in agg}:
        if ph not in DECLARED:
            continue
        need_idx = DECLARED[ph]["searched_later"] == "yes"
        ok = [(statistics.mean(a["t"]), n) for (p, n), a in agg.items()
              if p == ph and not a["lost"] and a["c"]["indexer"] == need_idx]
        if ok:
            out[ph] = (*min(ok), *max(ok))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--grid", default="/ssc-bench/live_results/grid_gs.jsonl")
    a = ap.parse_args()
    t = defaultdict(lambda: defaultdict(list))       # policy -> schedule -> [totals]
    lost, miss, runs, lat = defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(list)
    by_phase = defaultdict(lambda: defaultdict(list))  # policy -> phase -> [window seconds]
    for p in sorted(glob.glob(os.path.join(a.dir, "*.jsonl"))):
        m = re.match(r"(.+)_s(\d+)(?:_r\d+)?\.jsonl$", os.path.basename(p))
        if not m:
            continue
        rows = [json.loads(l) for l in open(p)]
        if not rows:
            continue
        pol, sched = m.group(1), m.group(2)
        t[pol][sched].append(total(rows))
        lost[pol] += sum(r["failed_puts"] for r in rows)
        miss[pol] += sum(1 for r in rows if r["search_miss"])
        runs[pol] += 1
        lat[pol] += [r["decision_latency_s"] for r in rows if "decision_latency_s" in r]
        for r in rows:
            by_phase[pol][r["phase"]].append(r["wall_s"] + r.get("recompose_s", 0))
    pols = [p for p in ORDER if p in t] + sorted(set(t) - set(ORDER))

    print("END TO END (speedup over the best fixed setup, per schedule)")
    print(f"{'policy':20s} {'geomean x':>9s} {'min x':>6s} {'max x':>6s} {'wins':>6s} {'lost':>6s} "
          f"{'miss':>5s} {'decide s (median)':>17s} runs")
    for pol in pols:
        sp = [statistics.mean(t[BASE][s]) / statistics.mean(v) for s, v in t[pol].items() if s in t[BASE]]
        if not sp:
            continue
        g = math.exp(sum(math.log(x) for x in sp) / len(sp))
        med = f"{statistics.median(lat[pol]):.2f}" if lat[pol] else "-"
        print(f"{pol:20s} {g:9.2f} {min(sp):6.2f} {max(sp):6.2f} {sum(x > 1 for x in sp):3d}/{len(sp):<2d} "
              f"{lost[pol]:6d} {miss[pol]:5d} {med:>17s} {runs[pol]}")

    if not os.path.exists(a.grid):
        return
    cases = grid_cases(a.grid)
    print("\nPER WORKLOAD: best / worst case from the grid; each policy's mean window time (s) and score "
          "(1 = best case, 0 = worst case)")
    for ph in sorted(cases):
        b, bn, w, wn = cases[ph]
        print(f"{ph}: best {b:.2f} s ({bn}), worst {w:.2f} s ({wn}), range {w / b:.1f}x")
        for pol in pols:
            xs = by_phase[pol].get(ph)
            if xs:
                m = statistics.mean(xs)
                print(f"    {pol:20s} {m:7.2f} s  score {(w - m) / (w - b) if w > b else 1.0:5.2f}")


if __name__ == "__main__":
    main()
