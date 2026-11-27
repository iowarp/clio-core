"""Online-exploration baselines: learn the composition by trying it during the run (no model, no SSC).

    python3 explore_baseline.py <results dir> [node app ...]

The natural non-LLM alternative to the SSC: with 24 compositions, why not try them while the
workload runs and keep the fastest? Each arm sees only what a run can see: the measured time of
the composition it deployed for a window (one repeat of the grid, drawn at random, so measurement
noise is real), keyed by phase. The phase identity is given to the arms for free (a generous
assumption; the SSC must infer similarity from the window's trace). Same guards as every other
controller: the indexer is on for streams declared searched later, and a composition whose
uncompressed window does not fit in the free space is never tried (the capacity reflex).

Arms (fresh per run: nothing is known about the workload when the run starts, as for the SSC):
    explore-first  try each untried valid composition once (fixed order), then exploit the best seen
    ucb            UCB1 on negative log time, per phase
    egreedy        epsilon-greedy (eps = 0.2) after one try of the default stack
    explore-warm   explore-first that keeps what it learned across the 10 test schedules (a site
                   that has already run this workflow 0..9 times; the SSC never measures it)

Windows are the ones the other arms replayed (the 'guards' arm of each offline scenario), so all
arms are scored on the same windows with the same metric (paper_plots.slowdown). Output rows
go to <results>/offline/<node>_<app>_explore.jsonl.
"""

import glob
import json
import math
import os
import random
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_plots as pp      # noqa: E402
import workload               # noqa: E402

AXES = ("placement", "cache", "indexer", "compression")
TIERS = {"1g": (1, 2), "05g": (0.5, 1.5), "4g": (4, 8), "05gp": (0.5, 2), "16g": (16, 48), "16gp": (16, 48)}


def grid_repeats(path):
    """(phase, key) -> list of (wall_s, lost) over repeats."""
    acc = defaultdict(list)
    for l in open(path):
        r = json.loads(l)
        c = r["composition"]
        acc[(r["phase"], (c["placement"], c["cache"], c["indexer"], c["compression"]))].append(
            (r["wall_s"], r.get("failed_puts", 0) > 0))
    return acc


def valid(phase, node):
    """Compositions a guarded controller may deploy for this phase."""
    blob, n, _, _ = (workload.LARGE if node.startswith("16g") else workload.PHASES)[phase]
    need = blob * n / (1 << 30)
    free = sum(TIERS[node])
    idx = phase in pp.SEARCHED
    out = []
    for p in ("dram", "disk"):
        for c in (False, True):
            for z in ("none", "lz4", "zstd"):
                if z == "none" and need > free:
                    continue                    # capacity reflex: never deploy a raw window that cannot fit
                out.append((p, c, idx, z))
    return out


def run(arm, windows, node, reps, rng, memory=None):
    """Replay one schedule; returns rows. memory: per-phase observations carried across runs."""
    obs = memory if memory is not None else defaultdict(lambda: defaultdict(list))
    rows = []
    for w in windows:
        ph = w["phase"]
        cand = [k for k in valid(ph, node) if (ph, k) in reps]
        seen = obs[ph]
        mean = {k: statistics.mean(v) for k, v in seen.items()}
        if arm in ("explore-first", "explore-warm"):
            untried = [k for k in cand if k not in seen]
            k = untried[0] if untried else min(mean, key=mean.get)
        elif arm == "ucb":
            untried = [k for k in cand if k not in seen]
            if untried:
                k = untried[0]
            else:
                t = sum(len(v) for v in seen.values())
                k = min(cand, key=lambda x: mean[x] - math.sqrt(2 * math.log(t) / len(seen[x])))
        elif arm == "egreedy":
            default = (pp.DEFAULT_STACK[0], False, ph in pp.SEARCHED, "none")
            if not seen:
                k = default if default in cand else cand[0]
            elif rng.random() < 0.2:
                k = rng.choice(cand)
            else:
                k = min(mean, key=mean.get)
        t, lost = rng.choice(reps[(ph, k)])          # what this run measures for its window
        seen[k].append(math.log(t) + (math.log(1e3) if lost else 0.0))
        rows.append({"variant": arm, "seed": w["seed"], "step": w["step"], "phase": ph,
                     "recommended": dict(zip(AXES, k)), "latency_s": 0.0, "violation": False,
                     "explored": len(seen) <= len(cand) and len(seen[k]) == 1})
    return rows


def main():
    res = sys.argv[1]
    keys = list(zip(sys.argv[2::2], sys.argv[3::2])) or None
    off = pp.load_offline(res)
    for (node, app), d in sorted(off.items()):
        if keys and (node, app) not in keys or "guards" not in d:
            continue
        reps = grid_repeats(os.path.join(res, "grid", f"grid_{node}.jsonl"))
        grid = pp.load_grid(os.path.join(res, "grid", f"grid_{node}.jsonl"))
        by_seed = defaultdict(list)
        for r in d["guards"]:
            by_seed[r["seed"]].append(r)
        out = []
        rng = random.Random(7)
        for arm in ("explore-first", "ucb", "egreedy"):
            for seed in sorted(by_seed):
                out += run(arm, sorted(by_seed[seed], key=lambda r: r["step"]), node, reps, rng)
        warm = defaultdict(lambda: defaultdict(list))
        for seed in sorted(by_seed):
            out += run("explore-warm", sorted(by_seed[seed], key=lambda r: r["step"]), node, reps, rng, warm)
        path = os.path.join(res, "offline", f"{node}_{app}_explore.jsonl")
        with open(path, "w") as f:
            for r in out:
                r["score"] = 1.0 / pp.slowdown(r, grid)
                f.write(json.dumps(r) + "\n")
        summ = defaultdict(list)
        for r in out:
            summ[r["variant"]].append(pp.slowdown(r, grid))
        print(node, app, {v: f"{100 * statistics.mean(s <= pp.ACC_WITHIN for s in x):.0f}% "
                             f"{math.exp(statistics.mean(map(math.log, x))):.3f}x" for v, x in summ.items()})


if __name__ == "__main__":
    main()
