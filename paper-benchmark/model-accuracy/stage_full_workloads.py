#!/usr/bin/env python3
"""Stage the new-workloads datasets in full (every chunk, no sampling).

    stage_full_workloads.py [DATASET ...] [--out /mnt/nvme0/v2-work]

For each dataset, every array file of its source directory (the files the
codec sweep sampled from: all but .json/.log/.txt/.csv) is cut into 4 MiB
chunks written as <out>/<ds>/fields/<NNNNN>__<path>__c<KKKK>__dt-<type>.chunk.
NNNNN is the file's position in application order, so the lexical order of
the names is the replay order: time step first (the pltNNNNN directory of a
simulation dump, or a file's frame / epoch / step / window / run / batch tag:
_f0004, _e05, _s1000, _w003, _r012, _b03, _step_100_chunk_0), then the array
name; files without a tag in name order. The last chunk of a file keeps its
real (shorter) length. Element types as stage_v2_workloads.dtype_of.
"""
import argparse
import os
import re

import stage_v2_workloads as sv

CHUNK = 4 << 20
DATASETS = ["graph-livejournal-full", "dl-opt-relu-act", "dl-resnet18-train",
            "graph-orkut-full", "genomics-reads", "detector-frames", "sparse-fem",
            "dl-gpt2-kv", "dl-qwen-bf16", "consumer-nyx", "consumer-vpic-full",
            "astro-camels", "hep-nanoaod", "dl-pythia-ckpt", "consumer-lammps-full", "genomics-chr1-4", "astro-camels-6snap",
            "ref-lammps-b70-2000", "ref-vpic-126-2000", "ref-warpx-64x64x512-2000"]


def product(rel):
    """Array name without its time tag (as codec-sweep sample_chunks.py)."""
    base = os.path.basename(rel)
    m = re.match(r"fab\d+_comp\d+_(.+)$", base)
    if m:
        base = m.group(1)
    base = re.sub(r"_(f|e|s|w|i|b|r|step)\d+(?=\.)", "", base)
    return re.sub(r"_step_\d+_chunk_\d+(?=\.)", "", base)


def time_key(rel):
    """(time step, array name, path): the order the application writes files."""
    m = re.search(r"plt(\d+)", rel)
    if m:
        return (int(m.group(1)), "", rel)
    base = os.path.basename(rel)
    m = re.search(r"_step_(\d+)_chunk_(\d+)\.", base)
    if m:
        return (int(m.group(1)), product(rel), rel)
    m = re.search(r"_(?:f|e|s|w|i|b|r)(\d+)\.", base)
    if m:
        return (int(m.group(1)), product(rel), rel)
    return (-1, product(rel), rel)


def stage(ds, out):
    """Cut every array file of one dataset into chunk files; (chunks, bytes)."""
    src = sv.source_dir(ds)
    rels = []
    for root, _, files in os.walk(src):
        for f in files:
            if not f.endswith((".json", ".log", ".txt", ".csv")):
                rels.append(os.path.relpath(os.path.join(root, f), src))
    rels.sort(key=time_key)
    os.makedirs(out, exist_ok=True)
    n = total = 0
    for i, rel in enumerate(rels):
        dt = sv.dtype_of(ds, rel, product(rel))
        with open(os.path.join(src, rel), "rb") as fh:
            k = 0
            while True:
                b = fh.read(CHUNK)
                if not b:
                    break
                name = f"{i:05d}__{rel.replace('/', '__')}__c{k:04d}__dt-{dt}.chunk"
                with open(os.path.join(out, name), "wb") as o:
                    o.write(b)
                k += 1
                n += 1
                total += len(b)
    return n, total


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("datasets", nargs="*", default=DATASETS)
    ap.add_argument("--out", default="/mnt/nvme0/v2-work")
    a = ap.parse_args()
    for ds in a.datasets:
        dst = os.path.join(a.out, ds, "fields")
        if os.path.exists(dst) and os.listdir(dst):
            raise SystemExit(f"{dst} is not empty; move the sampled staging away first")
        n, total = stage(ds, dst)
        print(f"{ds:28s} {n:5d} chunks  {total / 2**30:6.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
