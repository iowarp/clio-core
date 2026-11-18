#!/usr/bin/env python3
"""Generate a WfBench workflow (WfCommons) for the I/O comparison runs.

  wfc_gen.py <recipe> <num_tasks> <out_dir> [--size-scale X] [--cpu-work W]
             [--seed S]

The task graph and every file size come from the WfChef recipe (a synthetic
instance shaped like the real application). WorkflowBenchmark turns it into
wfbench tasks; this script then makes every task do the same small CPU work
(--percent-cpu 1.0 --cpu-work W, one chunk), so I/O dominates:
  * WfBench scales cpu_work by the task's recorded runtime and int()s it, so
    short tasks get cpu_work 0 and no --cpu-work flag at all, and wfbench
    refuses to run without one;
  * it also scales percent_cpu down, which turns the rest of the 10 wfbench
    threads into stress-ng --vm memory hogs.
--size-scale multiplies every file size (1.0 = the recipe's own sizes).
Input files are NOT created here (wfc_stage.py writes them into the run's
data directory); the JSON and a summary JSON go to <out_dir>.
"""
import argparse
import json
import logging
import pathlib
import random
import re

import numpy as np
from wfcommons.wfbench import WorkflowBenchmark
from wfcommons.wfchef import recipes
from wfcommons.wfgen import WorkflowGenerator

RECIPES = {"montage": "MontageRecipe", "epigenomics": "EpigenomicsRecipe",
           "genome": "GenomeRecipe", "blast": "BlastRecipe",
           "srasearch": "SrasearchRecipe", "bwa": "BwaRecipe",
           "soykb": "SoykbRecipe", "seismology": "SeismologyRecipe",
           "cycles": "CyclesRecipe", "rnaseq": "RnaseqRecipe"}


def scale_sizes(wf, factor):
    """Multiply the size of every file object of every task by factor."""
    seen = set()
    for t in wf.tasks.values():
        for f in list(t.input_files) + list(t.output_files):
            if id(f) in seen:
                continue
            seen.add(id(f))
            f.size = max(1, int(f.size * factor)) if f.size else 0


def uniform_cpu(spec, cpu_work):
    """Give every task --percent-cpu 1.0 --cpu-work cpu_work."""
    for t in spec["workflow"]["execution"]["tasks"]:
        args = [a for a in t["command"]["arguments"]
                if not re.match(r"--(percent-cpu|cpu-work) ", a)]
        args.insert(1, f"--cpu-work {cpu_work}")
        args.insert(1, "--percent-cpu 1.0")
        t["command"]["arguments"] = args


def summarize(spec):
    """Footprint, staged inputs, bytes written/read, task count, depth."""
    s = spec["workflow"]["specification"]
    size = {f["id"]: f["sizeInBytes"] for f in s["files"]}
    produced = {f for t in s["tasks"] for f in t.get("outputFiles", [])}
    staged = [f for f in size if f not in produced]
    rd = sum(size[f] for t in s["tasks"] for f in t.get("inputFiles", []))
    wr = sum(size[f] for t in s["tasks"] for f in t.get("outputFiles", []))
    par = {t["id"]: t.get("parents", []) for t in s["tasks"]}
    depth = {}

    def lvl(i):
        if i not in depth:
            depth[i] = 1 + max((lvl(p) for p in par[i]), default=0)
        return depth[i]
    for i in par:
        lvl(i)
    cats = {}
    for t in s["tasks"]:
        k = t.get("category") or t["name"]
        cats[k] = cats.get(k, 0) + 1
    return {"tasks": len(s["tasks"]), "files": len(size),
            "footprint_bytes": sum(size.values()),
            "staged_files": len(staged),
            "staged_bytes": sum(size[f] for f in staged),
            "bytes_written_by_tasks": wr, "bytes_read_by_tasks": rd,
            "max_file_bytes": max(size.values()), "depth": max(depth.values()),
            "categories": cats}


def main():
    """Parse arguments, generate, post-process and summarize one workflow."""
    ap = argparse.ArgumentParser()
    ap.add_argument("recipe", choices=sorted(RECIPES))
    ap.add_argument("num_tasks", type=int)
    ap.add_argument("out_dir")
    ap.add_argument("--size-scale", type=float, default=1.0)
    ap.add_argument("--cpu-work", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    logging.disable(logging.WARNING)
    random.seed(a.seed)
    np.random.seed(a.seed)
    out = pathlib.Path(a.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "to_create.txt").unlink(missing_ok=True)
    rc = getattr(recipes, RECIPES[a.recipe])
    wf = WorkflowGenerator(rc.from_num_tasks(a.num_tasks)).build_workflow()
    wf.name = f"{a.recipe}-Benchmark"
    if a.size_scale != 1.0:
        scale_sizes(wf, a.size_scale)
    bench = WorkflowBenchmark(rc, a.num_tasks)
    path = bench.create_benchmark_from_synthetic_workflow(
        out, wf, percent_cpu=1.0, cpu_work=a.cpu_work, num_chunks=1)
    (out / "to_create.txt").unlink(missing_ok=True)
    spec = json.loads(path.read_text())
    uniform_cpu(spec, a.cpu_work)
    tag = f"{a.recipe}-{a.num_tasks}" + (f"-x{a.size_scale:g}" if a.size_scale != 1.0 else "")
    final = out / f"{tag}.json"
    final.write_text(json.dumps(spec))
    if path != final:
        path.unlink()
    summ = summarize(spec)
    summ.update(recipe=a.recipe, requested_tasks=a.num_tasks,
                size_scale=a.size_scale, cpu_work=a.cpu_work, seed=a.seed,
                json=str(final))
    (out / f"{tag}.summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
