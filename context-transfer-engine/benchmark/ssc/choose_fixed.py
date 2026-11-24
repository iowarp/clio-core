"""Best and worst single composition for a whole campaign, from the measured grid.

    python3 choose_fixed.py [grid.jsonl] [phases]      -> prints "<best> <worst>"

A fixed composition must meet every phase's requirements for the whole run: no lost writes in
any phase, and the indexer on if any phase is searched later. It is scored by the sum over
phases of its mean window time (each phase weighted once).
"""

import json
import statistics
import sys
from collections import defaultdict

import compose
import workload
from controller import DECLARED


def main():
    grid = sys.argv[1] if len(sys.argv) > 1 else "/ssc-bench/live_results/grid_gs.jsonl"
    phases = sys.argv[2].split(",") if len(sys.argv) > 2 else list(workload.BASE_PHASES)
    agg = defaultdict(lambda: defaultdict(list))
    lost, comp = defaultdict(int), {}
    for l in open(grid):
        r = json.loads(l)
        n = compose.name(r["composition"])
        agg[n][r["phase"]].append(r["wall_s"])
        lost[n] += r.get("failed_puts", 0) if r["phase"] in phases else 0
        comp[n] = r["composition"]
    need_idx = any(DECLARED[p]["searched_later"] == "yes" for p in phases)
    score = {n: sum(statistics.mean(v[p]) for p in phases) for n, v in agg.items()
             if all(p in v for p in phases) and not lost[n] and comp[n]["indexer"] == need_idx}
    ranked = sorted(score, key=score.get)
    print(ranked[0], ranked[-1])
    for n in ranked:
        print(f"# {n:28s} {score[n]:7.2f} s", file=sys.stderr)


if __name__ == "__main__":
    main()
