#!/usr/bin/env python3
# Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
# BSD 3-Clause License.
#
# Prepare a real OGB node-property-prediction dataset (ogbn-arxiv by default,
# ogbn-products optionally) into flat little-endian binaries the C++ GNN test
# reads directly. No torch / no ogb package required: we parse the raw CSVs.
#
# Outputs (under <out>/):
#   features.f32 : N*F float32, row-major  (node feature matrix)
#   graph.csr    : int64 N, int64 E, indptr[N+1] int64, indices[E] int64
#                  (undirected, symmetrized, deduped CSR; neighbors grouped by src)
#   labels.i64   : N int64 (node class label; -1 where missing)
#   meta.txt     : "N F E C"  (num_nodes num_feat num_undirected_edges num_classes)
#
# Usage:
#   python3 gnn_prep.py --dataset arxiv    [--out /workspace/data/ogb/arxiv]
#   python3 gnn_prep.py --dataset products [--out /workspace/data/ogb/products]

import argparse
import gzip
import os
import sys
import zipfile

import numpy as np

try:
    import pandas as pd
    HAVE_PANDAS = True
except Exception:
    HAVE_PANDAS = False

# name -> (zip url, subdir inside zip, expected N, expected F, expected directed E, num_classes)
DATASETS = {
    "arxiv": (
        "http://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip",
        "arxiv", 169343, 128, 1166243, 40),
    "products": (
        "http://snap.stanford.edu/ogb/data/nodeproppred/products.zip",
        "products", 2449029, 100, 61859140, 47),
}


def download(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"[prep] already have {dest} ({os.path.getsize(dest)} bytes)")
        return
    print(f"[prep] downloading {url} -> {dest}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    # Use wget for a robust, resumable download (present in the container).
    rc = os.system(f"wget -q -O '{dest}.part' '{url}' && mv '{dest}.part' '{dest}'")
    if rc != 0 or not os.path.exists(dest):
        print("[prep] wget failed; trying urllib")
        import urllib.request
        urllib.request.urlretrieve(url, dest)
    print(f"[prep] downloaded {os.path.getsize(dest)} bytes")


def unzip(zip_path, extract_root):
    print(f"[prep] extracting {zip_path}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(extract_root)


def read_csv_gz(path, dtype):
    """Read a headerless gzipped CSV into a 2-D numpy array of `dtype`."""
    print(f"[prep] reading {path}")
    if HAVE_PANDAS:
        df = pd.read_csv(path, header=None, compression="gzip", dtype=np.float64)
        return df.to_numpy().astype(dtype, copy=False)
    with gzip.open(path, "rt") as f:
        rows = [line.rstrip("\n").split(",") for line in f if line.strip()]
    return np.asarray(rows, dtype=dtype)


def build_csr(edges_dir, num_nodes):
    """Symmetrize directed edges into a deduped undirected CSR (neighbors by src)."""
    src = edges_dir[:, 0].astype(np.int64, copy=False)
    dst = edges_dir[:, 1].astype(np.int64, copy=False)
    # Symmetrize: add both directions.
    u = np.concatenate([src, dst])
    v = np.concatenate([dst, src])
    # Dedup undirected pairs (drops the reciprocal double-count so the mean over
    # neighbors is a clean average). Encode (u,v) into one int64 key.
    key = u.astype(np.uint64) * np.uint64(num_nodes) + v.astype(np.uint64)
    key = np.unique(key)  # sorted ascending => rows grouped by u, then v
    u = (key // np.uint64(num_nodes)).astype(np.int64)
    v = (key % np.uint64(num_nodes)).astype(np.int64)
    # CSR: counts per source, prefix sum for indptr; indices already grouped by u.
    counts = np.bincount(u, minlength=num_nodes).astype(np.int64)
    indptr = np.zeros(num_nodes + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    indices = v.astype(np.int64, copy=False)
    return indptr, indices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="arxiv", choices=list(DATASETS.keys()))
    ap.add_argument("--data-root", default="/workspace/data/ogb")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    url, subdir, expN, expF, expE, expC = DATASETS[args.dataset]
    out = args.out or os.path.join(args.data_root, args.dataset)
    os.makedirs(out, exist_ok=True)

    zip_path = os.path.join(args.data_root, f"{args.dataset}.zip")
    raw_dir = os.path.join(args.data_root, subdir, "raw")
    if not os.path.isdir(raw_dir):
        download(url, zip_path)
        unzip(zip_path, args.data_root)
    if not os.path.isdir(raw_dir):
        print(f"[prep] ERROR: raw dir {raw_dir} not found after extract")
        sys.exit(1)

    # ---- features ----
    feats = read_csv_gz(os.path.join(raw_dir, "node-feat.csv.gz"), np.float32)
    N, F = feats.shape
    print(f"[prep] features: N={N} F={F}  (expected {expN}x{expF})")
    assert N == expN and F == expF, "feature shape mismatch vs OGB stats"

    # ---- labels ----
    labels_path = os.path.join(raw_dir, "node-label.csv.gz")
    if os.path.exists(labels_path):
        labels = read_csv_gz(labels_path, np.float64)
        labels = np.nan_to_num(labels, nan=-1.0).astype(np.int64).reshape(-1)
    else:
        labels = np.full(N, -1, dtype=np.int64)
    C = int(labels.max()) + 1 if labels.max() >= 0 else expC
    print(f"[prep] labels: {labels.shape[0]} num_classes~{C} (expected {expC})")

    # ---- edges -> CSR ----
    edges = read_csv_gz(os.path.join(raw_dir, "edge.csv.gz"), np.int64)
    print(f"[prep] directed edges: {edges.shape[0]} (expected {expE})")
    indptr, indices = build_csr(edges, N)
    E = int(indices.shape[0])
    print(f"[prep] undirected CSR: E={E} indptr[-1]={indptr[-1]} "
          f"avg_deg={E / N:.2f}")
    assert indptr[-1] == E

    # ---- write flat binaries (little-endian native on x86) ----
    feats.astype(np.float32, copy=False).tofile(os.path.join(out, "features.f32"))
    labels.astype(np.int64, copy=False).tofile(os.path.join(out, "labels.i64"))
    with open(os.path.join(out, "graph.csr"), "wb") as f:
        np.array([N, E], dtype=np.int64).tofile(f)
        indptr.astype(np.int64, copy=False).tofile(f)
        indices.astype(np.int64, copy=False).tofile(f)
    with open(os.path.join(out, "meta.txt"), "w") as f:
        f.write(f"{N} {F} {E} {C}\n")

    fb = os.path.getsize(os.path.join(out, "features.f32"))
    gb = os.path.getsize(os.path.join(out, "graph.csr"))
    print(f"[prep] wrote {out}/")
    print(f"[prep]   features.f32 = {fb} bytes ({fb/2**20:.1f} MiB)")
    print(f"[prep]   graph.csr    = {gb} bytes ({gb/2**20:.1f} MiB)")
    print(f"[prep]   meta.txt     = '{N} {F} {E} {C}'")
    print("[prep] DONE")


if __name__ == "__main__":
    main()
