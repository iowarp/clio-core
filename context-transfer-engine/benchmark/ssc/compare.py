"""Compare controller runs on the same schedule.

    python3 compare.py loop_static.jsonl loop_oracle.jsonl loop_agentjev.jsonl

Per policy:
    efficiency     mean of window MB/s / best measured MB/s for that phase (grid, requirements met)
    lost puts      puts refused because the tiers were full
    search misses  windows of a searched-later stream that ran without the indexer
    recompose s    total time spent restarting the runtime
    axis match     share of windows whose composition matches the measured best, per axis
"""

import json
import sys

from controller import DECLARED

AXES = ("placement", "cache", "indexer", "compression")


def load(path):
    return [json.loads(l) for l in open(path)]


def summarize(rows):
    eff = [r["mb_per_s"] / r["best_mb_per_s"] for r in rows if r.get("best_mb_per_s")]
    match = {ax: sum(r["composition"][ax] == r["best_composition"][ax] for r in rows if r.get("best_composition"))
             / max(1, sum(1 for r in rows if r.get("best_composition"))) for ax in AXES}
    return {"windows": len(rows),
            "efficiency": sum(eff) / len(eff) if eff else None,
            "lost_puts": sum(r.get("failed_puts", 0) for r in rows),
            "search_misses": sum(1 for r in rows if DECLARED[r["phase"]]["searched_later"] == "yes"
                                 and not r["composition"]["indexer"]),
            "recompose_s": sum(r.get("recompose_s", 0) for r in rows),
            "recompositions": sum(1 for r in rows if r.get("recomposed_to") or r.get("recompose_s", 0) > 0),
            "axis_match": match}


def main():
    print(f"{'policy':10s} {'eff':>6s} {'lost':>6s} {'srch':>5s} {'recomp':>7s} {'n_rc':>5s}  "
          + "  ".join(f"{a[:5]:>5s}" for a in AXES))
    for p in sys.argv[1:]:
        rows = load(p)
        s = summarize(rows)
        print(f"{rows[0]['policy']:10s} {100 * s['efficiency']:5.1f}% {s['lost_puts']:6d} {s['search_misses']:5d} "
              f"{s['recompose_s']:6.1f}s {s['recompositions']:5d}  "
              + "  ".join(f"{100 * s['axis_match'][a]:4.0f}%" for a in AXES))


if __name__ == "__main__":
    main()
