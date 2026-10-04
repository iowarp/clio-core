#!/usr/bin/env python3
"""Chunk list for corpus_sweep: up to K evenly spaced CHUNK-byte chunks per
product of a dataset. A product is a file name with its frame / epoch /
step / window tag removed (cc_label_f0004.i32 -> cc_label.i32); simulation
dumps (<frame>/<field>.f32 or fab*_comp*_<field>.f32) use the field name.

    sample_chunks.py --dir DIR [--per-product 24 | --fraction 0.05] [--chunk 4194304] > list
--fraction F takes ceil(F x the product's chunk count) instead of a fixed K.
Writes "path k" lines; the product of each line goes to stderr as a map
when --map FILE is given.
"""
import argparse, os, re

def product(rel):
    base = os.path.basename(rel)
    m = re.match(r"fab\d+_comp\d+_(.+)$", base)
    if m:
        base = m.group(1)
    base = re.sub(r"_(f|e|s|w|i|b|r|step)\d+(?=\.)", "", base)
    base = re.sub(r"_step_\d+_chunk_\d+(?=\.)", "", base)
    return base

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--per-product", type=int, default=24)
ap.add_argument("--chunk", type=int, default=4 << 20)
ap.add_argument("--map", default="")
ap.add_argument("--fraction", type=float, default=0.0)
a = ap.parse_args()
groups = {}
for root, _, files in os.walk(a.dir):
    for f in sorted(files):
        if f.endswith((".json", ".log", ".txt", ".csv")):
            continue
        rel = os.path.relpath(os.path.join(root, f), a.dir)
        n = -(-os.path.getsize(os.path.join(root, f)) // a.chunk)
        groups.setdefault(product(rel), []).extend((rel, k) for k in range(n))
lines = []
for p, pairs in sorted(groups.items()):
    pairs.sort()
    k = (-(-int(a.fraction * len(pairs) * 1e6) // 1_000_000) if a.fraction > 0
         else a.per_product)  # ceil(F x n) without float rounding up an exact product
    k = min(max(k, 1), len(pairs))
    for j in range(k):
        rel, c = pairs[j * (len(pairs) - 1) // max(k - 1, 1)]
        lines.append((rel, c, p))
for rel, c, _ in lines:
    print(rel, c)
if a.map:
    with open(a.map, "w") as m:
        for rel, c, p in lines:
            m.write(f"{rel}#{c},{p}\n")
