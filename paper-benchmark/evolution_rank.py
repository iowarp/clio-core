#!/usr/bin/env python3
"""Rank the configurations of one workload's evolution study.

    ./evolution_rank.py /tmp/wx-study/ev

Reads every <dir>/*/evolution.json written by evolution.py and prints the
comparison the study calls for: mean / median / max / min, the share of blocks
actively changing, the two sustain statistics, and `same%` -- the share of
CELLS bit-identical to the previous sampled frame, whose complement is what
the codec actually has to encode as new.

RANKING IS ON SUSTAINED EVOLUTION, NOT ON THE MEAN. A configuration that
changes violently once and then sits still scores a high mean and is exactly
what this study is meant to reject, so configurations are ordered by

    score = p10_interval * (pct_active / 100)

-- the 10th-percentile per-interval mean, which a single spike cannot lift,
weighted by how much of the data is moving at all. mean is reported beside it
and is usually in the same order; where the two disagree the disagreement is
the finding, and the table shows both.

A configuration with any NaN/Inf block is disqualified outright, whatever it
scores: an unstable run must not be able to win. Everything else a run can do
wrong is physics this metric cannot see -- a laser resonating in a perfectly
reflecting box evolves beautifully and means nothing -- so a study that rules
a configuration out on physics writes the reason into its evolution.json as
"disqualified", and it is honoured here.
"""
import glob, json, os, sys

args = [a for a in sys.argv[1:] if not a.startswith("-")]
MD = "--md" in sys.argv
if not args:
    sys.exit(__doc__)

runs = []
for p in sorted(glob.glob(os.path.join(args[0], "*", "evolution.json"))):
    runs.append(json.load(open(p)))
if not runs:
    sys.exit(f"no evolution.json under {sys.argv[1]}")


def score(d):
    return d["p10_interval"] * d["pct_active"] / 100.0


def why_bad(d):
    """-> reason this configuration cannot win, or None.

    NaN/Inf is caught automatically. Everything else a run can do wrong is
    physics the metric cannot see -- a laser resonating in a perfectly
    reflecting box evolves beautifully and means nothing -- so a study that
    rules a configuration out records WHY in its evolution.json under
    "disqualified", and it is honoured here rather than being remembered by
    whoever reads the table.
    """
    if d["nonfinite_blocks"]:
        return f"{d['nonfinite_blocks']} block(s) held NaN/Inf"
    return d.get("disqualified")


ok = [d for d in runs if not why_bad(d)]
bad = [d for d in runs if why_bad(d)]
ok.sort(key=score, reverse=True)

if MD:
    print("| config | parameters | mean E | median | active blocks | cells "
          "bit-identical | p10 interval | last quarter | sim s |")
    print("|---|---|---|---|---|---|---|---|---|")
    for i, d in enumerate(ok):
        name = f"**`{d['label']}`**" if i == 0 else f"`{d['label']}`"
        print(f"| {name} | `{d.get('params','')}` | "
              f"{'**' if i==0 else ''}{d['mean']:.4f}{'**' if i==0 else ''} | "
              f"{d['median']:.4f} | {d['pct_active']:.1f}% | "
              f"{d.get('pct_cells_same', float('nan')):.2f}% | "
              f"{d['p10_interval']:.4f} | {d['last_quarter']:.4f} | "
              f"{d.get('sim_wall_s',0):.0f} |")
    for d in bad:
        print(f"| `{d['label']}` | `{d.get('params','')}` | "
              f"DISQUALIFIED — {why_bad(d)} | | | | | | |")
    sys.exit(0)

w = max(len(d["label"]) for d in runs) + 1
print(f"{'config':<{w}} {'score':>7} {'mean':>7} {'median':>7} {'max':>6} "
      f"{'min':>6} {'active%':>8} {'p10':>7} {'lastQ':>7} {'first':>7} "
      f"{'same%':>7} {'sim s':>7}")
print("-" * (w + 84))
for i, d in enumerate(ok):
    print(f"{d['label']:<{w}} {score(d):>7.4f} {d['mean']:>7.4f} "
          f"{d['median']:>7.4f} {d['max']:>6.3f} {d['min']:>6.3f} "
          f"{d['pct_active']:>7.1f}% {d['p10_interval']:>7.4f} "
          f"{d['last_quarter']:>7.4f} {d['first_interval']:>7.4f} "
          f"{d.get('pct_cells_same', float('nan')):>7.2f} "
          f"{d.get('sim_wall_s', 0):>7.1f}"
          + ("   <-- BEST" if i == 0 else ""))
for d in bad:
    print(f"{d['label']:<{w}} DISQUALIFIED -- {why_bad(d)}")
print()
for d in ok + bad:
    print(f"{d['label']}: {d.get('params', '')}")
    print(f"   {d['frames']} frames, {d['pairs']} pairs, "
          f"{d['blocks_total']} blocks of {d['block_bytes'] // 1024} KiB, "
          f"sampled every {d.get('sample_interval_steps', '?')} steps")
