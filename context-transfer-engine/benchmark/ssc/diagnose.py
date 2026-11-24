"""Where does a model-driven run lose time? Per window: what was deployed, what the grid says is
best for that phase, the regret, and what the model answered (and about which window).

    python3 diagnose.py run.jsonl [--grid grid.jsonl]

Regret of a window = its time minus the grid's best-case time for its phase. The summary groups
regret by phase and by axis on which the deployed composition differed from the best one, which
points at the SSC evidence (or question) that misled the model.
"""

import argparse
import json
from collections import defaultdict

import compose
from paper_compare import grid_cases

AXES = ("placement", "cache", "indexer", "compression")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--grid", default="/ssc-bench/live_results/grid_gs.jsonl")
    ap.add_argument("-q", action="store_true", help="summary only")
    a = ap.parse_args()
    cases = grid_cases(a.grid)
    by_phase, by_axis, n_axis = defaultdict(float), defaultdict(float), defaultdict(int)
    for path in a.runs:
        if not a.q:
            print(f"== {path}")
        for r in (json.loads(l) for l in open(path)):
            ph = r["phase"]
            if ph not in cases:
                continue
            best_s, best_n = cases[ph][0], cases[ph][1]
            best = dict(zip(AXES, [best_n.split("-")[0], best_n.split("-")[1] == "cache",
                                   best_n.split("-")[2] == "idx", best_n.split("-")[3]]))
            dep = r["composition"]
            regret = r["wall_s"] + r.get("recompose_s", 0) - best_s
            diff = [ax for ax in AXES if dep[ax] != best[ax]]
            by_phase[ph] += regret
            for ax in diff:
                by_axis[(ph, ax)] += regret / len(diff)
                n_axis[(ph, ax)] += 1
            if not a.q:
                print(f"{r['step']:2d} {ph:16s} {compose.name(dep):26s} best {best_n:26s} regret {regret:6.2f} s"
                      f"{'  diff ' + ','.join(diff) if diff else ''}"
                      f"{'  answers(w' + str(r.get('answer_for_step', r['step'])) + ') ' + str(r['answers']) if r.get('answers') else ''}")
    print("\nregret by phase (s):")
    for ph, v in sorted(by_phase.items(), key=lambda x: -x[1]):
        print(f"  {ph:16s} {v:7.2f}")
    print("regret by phase and wrong axis (s, windows):")
    for k, v in sorted(by_axis.items(), key=lambda x: -x[1])[:12]:
        print(f"  {k[0]:16s} {k[1]:12s} {v:7.2f}  {n_axis[k]}")


if __name__ == "__main__":
    main()
