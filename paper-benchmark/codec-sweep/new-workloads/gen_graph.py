#!/usr/bin/env python3
"""Graph workloads from real SNAP graphs: the arrays a graph-analytics code
holds and writes, one flat little-endian file per array.

    gen_graph.py --edges FILE.txt.gz --out DIR [--format snap|dimacs|temporal]
                 [--weights2 FILE] [--symmetrize] [--pr-iters N]

  --format dimacs   a DIMACS .gr file ("a u v w" lines, 1-based): adds the
                    edge weight (int32); --weights2 adds a second weight file
                    of the same graph (e.g. travel time next to distance)
  --format temporal a SNAP temporal edge list "src dst unix_time": adds the
                    timestamps (int32), in the file's (time) order
  --symmetrize      also writes the undirected CSR (sym_row_ptr / sym_col_idx),
                    what BFS / connected-component / GNN codes keep in memory
  --pr-iters N      writes the PageRank vector after every one of N
                    iterations (pagerank_i<NN>), what an iterative analytics
                    run checkpoints; otherwise only the converged vector

  edges_src, edges_dst  int32  the edge list in the file's own order (raw input)
  row_ptr               int64  CSR row offsets (n + 1)
  col_idx               int32  CSR column indices, sorted within each row
  degree                int32  out-degree per vertex
  pagerank              f32    PageRank (d = 0.85, 50 iterations)
  bfs_level             int32  BFS hop count from the highest-degree vertex
                               (-1 = unreachable)
  cc_label              int32  weakly connected component id per vertex
"""
import argparse
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csgraph


def write(out, name, arr):
    arr = np.ascontiguousarray(arr)
    ext = {np.dtype("int32"): "i32", np.dtype("int64"): "i64",
           np.dtype("float32"): "f32"}[arr.dtype]
    arr.tofile(os.path.join(out, f"{name}.{ext}"))
    print(f"  {name}: {arr.size} x {arr.dtype} = {arr.nbytes / 2**20:.0f} MiB", flush=True)


def bfs_levels(A, src):
    n = A.shape[0]
    level = np.full(n, -1, dtype=np.int32)
    level[src] = 0
    frontier = np.zeros(n, dtype=bool)
    frontier[src] = True
    AT = A.T.tocsr()
    d = 0
    while frontier.any():
        d += 1
        reached = (AT @ frontier.astype(np.float32)) > 0
        nxt = reached & (level < 0)
        level[nxt] = d
        frontier = nxt
    return level


def pagerank(A, iters=50, damp=0.85, each=None):
    """PageRank by power iteration; each(i, r) is called after iteration i."""
    n = A.shape[0]
    deg = np.asarray(A.sum(axis=1)).ravel().astype(np.float32)
    inv = np.where(deg > 0, 1.0 / np.maximum(deg, 1), 0).astype(np.float32)
    P = (sp.diags(inv) @ A).T.tocsr().astype(np.float32)
    r = np.full(n, 1.0 / n, dtype=np.float32)
    for i in range(iters):
        dangling = r[deg == 0].sum()
        r = (damp * (P @ r) + (damp * dangling + 1 - damp) / n).astype(np.float32)
        if each:
            each(i + 1, r)
    return r.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--format", default="snap", choices=["snap", "dimacs", "temporal"])
    ap.add_argument("--weights2", default="")
    ap.add_argument("--symmetrize", action="store_true")
    ap.add_argument("--pr-iters", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.format == "dimacs":
        e = pd.read_csv(a.edges, sep=" ", comment="c", header=None, usecols=[0, 1, 2, 3],
                        names=["k", "u", "v", "w"], dtype={"k": str}, engine="c")
        e = e[e.k == "a"]
        src, dst = e.u.values.astype(np.int64) - 1, e.v.values.astype(np.int64) - 1
        write(a.out, "weight", e.w.values.astype(np.int32))
        if a.weights2:
            e2 = pd.read_csv(a.weights2, sep=" ", comment="c", header=None, usecols=[0, 3],
                             names=["k", "w"], dtype={"k": str}, engine="c")
            write(a.out, "weight2", e2[e2.k == "a"].w.values.astype(np.int32))
    else:
        e = pd.read_csv(a.edges, sep=r"\s+", comment="#", header=None,
                        dtype=np.int64, engine="c").values
        src, dst = e[:, 0], e[:, 1]
        if a.format == "temporal":
            write(a.out, "timestamp", e[:, 2].astype(np.int32))
    n = int(max(src.max(), dst.max())) + 1
    print(f"{os.path.basename(a.edges)}: {n} vertices, {len(src)} edges", flush=True)
    write(a.out, "edges_src", src.astype(np.int32))
    write(a.out, "edges_dst", dst.astype(np.int32))
    A = sp.csr_matrix((np.ones(len(src), dtype=np.float32), (src, dst)), shape=(n, n))
    A.sum_duplicates()
    A.sort_indices()
    write(a.out, "row_ptr", A.indptr.astype(np.int64))
    write(a.out, "col_idx", A.indices.astype(np.int32))
    deg = np.diff(A.indptr).astype(np.int32)
    write(a.out, "degree", deg)
    if a.symmetrize:
        S = (A + A.T).tocsr()
        S.sum_duplicates()
        S.sort_indices()
        write(a.out, "sym_row_ptr", S.indptr.astype(np.int64))
        write(a.out, "sym_col_idx", S.indices.astype(np.int32))
    if a.pr_iters:
        pagerank(A, a.pr_iters,
                 each=lambda i, r: write(a.out, f"pagerank_i{i:02d}", r))
    else:
        write(a.out, "pagerank", pagerank(A))
    write(a.out, "bfs_level", bfs_levels(A, int(np.argmax(deg))))
    _, cc = csgraph.connected_components(A, directed=True, connection="weak")
    write(a.out, "cc_label", cc.astype(np.int32))


if __name__ == "__main__":
    main()
