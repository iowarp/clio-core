"""Learning-curve subsets: which Gray-Scott phases a node has measured (its SSC experience).

    python3 lc_subsets.py [per_size]      -> one line per subset: "<N> <phase,phase,...>"

Seeded random subsets of workload.MEMORY_PHASES for N = 1, 2, 3, 5 (per_size each, all of them if
fewer exist) and the full set (N = 7); the same subsets for every scenario and arm.
"""
import itertools
import random
import sys

import workload

per = int(sys.argv[1]) if len(sys.argv) > 1 else 5
mem = list(workload.MEMORY_PHASES)
rng = random.Random(2026)
for n in (1, 2, 3, 5):
    allc = list(itertools.combinations(mem, n))
    pick = allc if len(allc) <= per else rng.sample(allc, per)
    for c in pick:
        print(n, ",".join(c))
print(len(mem), ",".join(mem))
