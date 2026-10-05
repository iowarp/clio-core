#!/usr/bin/env python3
"""Compare NeuroPress v2's own overheads across code iterations.

    compare_v2_overhead.py TAG [TAG ...] [--modes learn learnexp]

Each TAG names a set of runs <ds>_<mode>_<TAG> under /mnt/nvme0/v2-work/runs
("-" for the untagged runs). Per tag and mode, summed over the 19 datasets:
measured end-to-end write and read wall time (stdout.log), and from
phases.csv each NeuroPress step twice -- host wall clock (*_ms) and GPU time
from CUDA events (*_gpu_ms, when the build logged it): prediction (nn),
learning updates (sgd), the decompress for learning labels (label) and
exploration (explore); plus the codec's own compress time (CUDA events).
"""
import argparse
import os
import re

import pandas as pd

RUNS = "/mnt/nvme0/v2-work/runs"
DATASETS = ("detector-frames dl-gpt2-kv dl-opt-relu-act consumer-nyx "
            "consumer-vpic-full consumer-lammps-full dl-pythia-ckpt genomics-reads "
            "hep-nanoaod dl-resnet18-train graph-orkut-full sparse-fem "
            "graph-livejournal-full dl-qwen-bf16 ref-lammps-b70-2000 "
            "ref-nyx-256-2000 ref-vpic-126-2000 ref-warpx-64x64x512-2000 "
            "astro-camels").split()
STEPS = ["nn", "sgd", "label", "explore"]


def wall(run):
    """(write s, read s) from a run's stdout.log."""
    text = open(os.path.join(run, "stdout.log")).read()
    w = re.search(r"stage\+compress ([0-9.]+) s", text)
    r = re.search(r"READ 1/1.*?get\+decompress: ([0-9.]+) ms", text, re.S)
    return float(w.group(1)), float(r.group(1)) / 1e3


def totals(tag, mode):
    """Summed seconds over the datasets for one tag and mode."""
    out = {"write_s": 0.0, "read_s": 0.0, "compress_gpu_s": 0.0}
    for step in STEPS:
        out[f"{step}_s"] = 0.0
        out[f"{step}_gpu_s"] = float("nan")
    for ds in DATASETS:
        run = os.path.join(RUNS, f"{ds}_{mode}" + ("" if tag == "-" else f"_{tag}"))
        w, r = wall(run)
        out["write_s"] += w
        out["read_s"] += r
        ph = pd.read_csv(os.path.join(run, "phases.csv"))
        ph = ph[ph.path == "write"]
        out["compress_gpu_s"] += ph.compress_ms.fillna(0).sum() / 1e3
        for step in STEPS:
            col = f"{step}_ms"
            if col in ph:
                out[f"{step}_s"] += ph[col].fillna(0).sum() / 1e3
            gcol = f"{step}_gpu_ms"
            if gcol in ph:
                g = ph[gcol].fillna(0).clip(lower=0).sum() / 1e3
                prev = out[f"{step}_gpu_s"]
                out[f"{step}_gpu_s"] = g if prev != prev else prev + g
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--modes", nargs="+", default=["learn", "learnexp"])
    a = ap.parse_args()
    rows = []
    for mode in a.modes:
        for tag in a.tags:
            rows.append({"mode": mode, "tag": tag, **totals(tag, mode)})
    t = pd.DataFrame(rows).set_index(["mode", "tag"])
    pd.set_option("display.width", 250)
    print("seconds summed over 19 datasets; *_s = host wall clock, "
          "*_gpu_s = GPU time from CUDA events")
    print(t.round(2).to_string())


if __name__ == "__main__":
    main()
