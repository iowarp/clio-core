"""Summarise end-to-end runs: total time and requirement violations.

Total = I/O + recomposition + decision stalls. A blocking decision (the controller waits for the
model before the next window) stalls the workload, so its latency counts; an async decision
overlaps the workload and costs only staleness, which shows up in the I/O time.

    python3 e2e_compare.py <dir with *_r<rep>.jsonl>
"""

import glob
import json
import os
import statistics
import sys
from collections import defaultdict

ORDER = ["fixed_disk-cache-idx-zstd", "fixed_dram-nocache-idx-zstd", "fixed_dram-cache-idx-lz4", "agentjev",
         "agentjev-v3", "agentjev-v4", "agentjev-v3-rules", "agentjev-async", "heuristic", "heuristic2",
         "qwen3-blocking", "qwen3-async", "oracle_reactive", "oracle"]
LABEL = {"fixed_disk-cache-idx-zstd": "worst single valid setup",
         "fixed_dram-nocache-idx-zstd": "fixed dram-nocache-idx-zstd",
         "fixed_dram-cache-idx-lz4": "best single valid setup",
         "agentjev": "SSC v2 + AgentJev (System 1)",
         "agentjev-v3": "SSC v3 + AgentJev",
         "agentjev-v4": "SSC v4 (current hidden)",
         "agentjev-v3-rules": "SSC v3 + safety rules",
         "heuristic": "expert rules + safety rules",
         "heuristic2": "expert rules (fair, +capacity rule)",
         "agentjev-async": "System 1: AgentJev, async",
         "qwen3-blocking": "System 2: qwen3-8B, blocking",
         "qwen3-async": "System 2: qwen3-8B, async",
         "oracle_reactive": "oracle, one window late",
         "oracle": "oracle (knows the future)"}


def main():
    runs = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(sys.argv[1], "*_r*.jsonl"))):
        pol = os.path.basename(p).rsplit("_r", 1)[0]
        rows = [json.loads(l) for l in open(p)]
        if not rows:
            continue
        runs[pol].append({
            "io_s": sum(r["wall_s"] for r in rows),
            "recompose_s": sum(r.get("recompose_s", 0) for r in rows),
            "n_recompose": sum(1 for r in rows if "recompose_mode" in r),
            "n_restart": sum(1 for r in rows if r.get("recompose_mode", "").startswith("restart")),
            "lost": sum(r["failed_puts"] for r in rows),
            "search_miss": sum(1 for r in rows if r["search_miss"]),
            "decide_s": sum(r.get("decision_latency_s", 0) for r in rows),
            "stall_s": sum(r.get("decision_latency_s", 0) for r in rows if not r.get("async")),
            "windows": len(rows),
            "by_phase": {ph: sum(r["wall_s"] for r in rows if r["phase"] == ph) for ph in {r["phase"] for r in rows}},
        })
    base = None
    print(f"{'policy':32s} {'total s':>14s} {'vs best-fixed':>13s} {'recomp':>7s} {'lost':>5s} {'srch':>5s} {'decide s':>8s} reps")
    for pol in ORDER + sorted(set(runs) - set(ORDER)):
        if pol not in runs:
            continue
        rs = runs[pol]
        tot = [r["io_s"] + r["recompose_s"] + r["stall_s"] for r in rs]
        m, sd = statistics.mean(tot), (statistics.stdev(tot) if len(tot) > 1 else 0.0)
        if pol == "fixed_dram-cache-idx-lz4":
            base = m
        print(f"{LABEL.get(pol, pol):32s} {m:7.1f} ± {sd:4.1f} "
              f"{(f'{base / m:.2f}x' if base else '-'):>13s} "
              f"{statistics.mean(r['n_recompose'] for r in rs):7.1f} {statistics.mean(r['lost'] for r in rs):5.0f} "
              f"{statistics.mean(r['search_miss'] for r in rs):5.1f} {statistics.mean(r['decide_s'] for r in rs):8.1f} {len(rs)}")
    print("\nper phase (mean s):")
    phases = sorted({ph for rs in runs.values() for r in rs for ph in r["by_phase"]})
    print(f"{'policy':32s} " + " ".join(f"{ph[:12]:>12s}" for ph in phases))
    for pol in ORDER:
        if pol in runs:
            print(f"{LABEL[pol]:32s} " + " ".join(
                f"{statistics.mean(r['by_phase'].get(ph, 0) for r in runs[pol]):12.2f}" for ph in phases))


if __name__ == "__main__":
    main()
