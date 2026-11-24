"""Measured ground truth: run every workload phase under every composition.

    python3 run_grid.py out.jsonl [--phases a,b] [--repeat N] [--quick]

For each composition the runtime is restarted fresh, the client runs all phases, and the
results are appended to out.jsonl. The best composition per phase replaces the rule
oracle of SSC-Bench.
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time

import compose
import lifecycle
import workload

HERE = os.path.dirname(os.path.abspath(__file__))


def grid(quick=False):
    for placement, cache, indexer, comp in itertools.product(
            ["dram", "disk"], [False, True], [False, True], ["none", "lz4", "zstd"]):
        if quick and (indexer or placement == "disk"):
            continue
        yield {"placement": placement, "cache": cache, "indexer": indexer, "compression": comp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--phases", default=",".join(workload.PHASES))
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            done.add(compose.name(json.loads(l)["composition"]))
    for c in grid(a.quick):
        if compose.name(c) in done:
            continue
        subprocess.run(["rm", "-rf", lifecycle.ROOT])
        t0 = time.time()
        proc, ready_s = lifecycle.start(c)
        try:
            r = subprocess.run([sys.executable, os.path.join(HERE, "measure.py"), json.dumps(c), a.phases,
                                str(a.repeat)], env=lifecycle.env_for(c), capture_output=True, text=True,
                               timeout=1800)
            lines = [l for l in r.stdout.splitlines() if l.startswith("{")]
            if r.returncode != 0 or not lines:
                print(f"FAIL {compose.name(c)} rc={r.returncode}\n{r.stderr[-2000:]}", file=sys.stderr)
            with open(a.out, "a") as f:
                for l in lines:
                    x = json.loads(l)
                    x["startup_s"] = ready_s
                    f.write(json.dumps(x) + "\n")
                    print(f"{compose.name(c):32s} {x['phase']:14s} {x['mb_per_s']:9.1f} MB/s", flush=True)
        finally:
            lifecycle.stop(proc, c)
        print(f"-- {compose.name(c)} done in {time.time() - t0:.0f}s (startup {ready_s:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
