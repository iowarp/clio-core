#!/usr/bin/env python3
"""Did the data actually evolve, per timestep?

evolution.py cannot answer this for an IN-SITU run: it reads field FILES, and
in situ there are none -- the deck hands the compressor device pointers and no
.f32 is ever written. But blobs.csv carries an FNV-1a64 digest of every chunk
the simulation submitted, named `<var>/step_<n>/chunk_<k>`, so the question
can be answered EXACTLY rather than approximately:

    a chunk whose digest equals its own digest at the previous dump did not
    change by a single bit.

That is the complement of the `pct_cells_same` metric the VPIC and Nyx READMEs
report, at block granularity and with no threshold to choose. It cannot say
how FAR a chunk moved (a digest is not a norm) -- for that you need
evolution.py on a file-writing run -- but "did it move at all" is precisely
what decides whether a workload is exercising the selector or replaying a
frozen field.

Usage:  evolution_from_blobs.py <blobs.csv> [blobs.csv ...]
"""
import csv, sys, re
from collections import defaultdict

NAME = re.compile(r"^(?P<var>.+)/step_(?P<step>\d+)/chunk_(?P<chunk>\d+)$")


def read(path):
    """-> {(var, chunk): [(step, digest), ...]} sorted by step."""
    series = defaultdict(list)
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            m = NAME.match(row["blob"])
            if not m:
                continue
            series[(m["var"], int(m["chunk"]))].append(
                (int(m["step"]), row["fnv1a64"]))
    for k in series:
        series[k].sort()
    return series


def report(path):
    series = read(path)
    if not series:
        print(f"{path}: no step-named blobs"); return

    steps = sorted({s for v in series.values() for s, _ in v})
    per_var, total_pairs, total_same = {}, 0, 0
    frozen, alive = [], []

    for (var, chunk), pts in sorted(series.items()):
        pairs = len(pts) - 1
        if pairs <= 0:
            continue
        same = sum(1 for i in range(1, len(pts)) if pts[i][1] == pts[i - 1][1])
        distinct = len({d for _, d in pts})
        per_var[(var, chunk)] = (pairs, same, distinct)
        total_pairs += pairs
        total_same += same
        # "Frozen" is the strong statement: ONE distinct digest for the whole
        # run, i.e. the variable was dumped bit-identical every single frame.
        (frozen if distinct == 1 else alive).append(var)

    print(f"\n=== {path}")
    print(f"    {len(steps)} dumps (step {steps[0]}..{steps[-1]}), "
          f"{len(per_var)} (variable, chunk) series")
    pct = 100.0 * total_same / total_pairs if total_pairs else 0.0
    print(f"    consecutive-dump pairs bit-identical: "
          f"{total_same}/{total_pairs} = {pct:.1f}%")
    print(f"    variables that NEVER changed: {len(frozen)} of {len(per_var)}"
          + (f"  -> {', '.join(sorted(set(frozen)))}" if frozen else ""))

    print(f"    {'variable':<12} {'distinct':>8} {'of dumps':>9} "
          f"{'pairs same':>11}   verdict")
    for (var, chunk), (pairs, same, distinct) in per_var.items():
        verdict = ("FROZEN" if distinct == 1 else
                   "evolving" if same == 0 else f"partial ({same} repeats)")
        print(f"    {var:<12} {distinct:>8} {pairs + 1:>9} "
              f"{same:>11}   {verdict}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for p in sys.argv[1:]:
        report(p)
