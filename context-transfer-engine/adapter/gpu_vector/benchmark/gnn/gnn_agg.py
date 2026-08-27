#!/usr/bin/env python3
# Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
# BSD 3-Clause License.
#
# One-time 1-hop mean neighbour-aggregation (SIGN-style precompute):
#   A[i] = mean( {X[i]} U {X[j] : j in N(i)} )     (mean-aggregate with self-loop)
# Writes A as agg_features.f32 (N*F float32) next to features.f32, so the GNN
# training test learns a classifier over the aggregated features -- the feature
# matrix that Eternia stores compressed and streams during training.
#
# Scales to ogbn-papers100M (N=1.1e8, F=128, E=3.2e9): X, the CSR and the output
# are all memory-mapped and processed in nnz-bounded row blocks, so peak RSS is
# a few GiB per worker rather than the 57 GiB matrix. Segment sums use
# np.add.reduceat over each block (float64 accumulate, as before).
#
# Usage: python3 gnn_agg.py --data /workspace/data/ogb/arxiv
#        python3 gnn_agg.py --data /tmp/gnn/data/papers100M --workers 8

import argparse
import multiprocessing as mp
import os
import time

import numpy as np

_G = {}


def _open(data, N, F, E):
    """Open the per-worker memory maps (once per process)."""
    if _G:
        return _G
    _G["X"] = np.memmap(os.path.join(data, "features.f32"), dtype=np.float32,
                        mode="r", shape=(N, F))
    # graph.csr = [N,E] int64 header, then indptr[N+1], then indices[E]
    _G["indptr"] = np.memmap(os.path.join(data, "graph.csr"), dtype=np.int64,
                             mode="r", offset=16, shape=(N + 1,))
    _G["indices"] = np.memmap(os.path.join(data, "graph.csr"), dtype=np.int64,
                              mode="r", offset=16 + 8 * (N + 1), shape=(E,))
    _G["A"] = np.memmap(os.path.join(data, "agg_features.f32"), dtype=np.float32,
                        mode="r+", shape=(N, F))
    return _G


def _init(data, N, F, E):
    _open(data, N, F, E)


def _block(task):
    """Aggregate rows [r0, r1) -> A[r0:r1]. Returns (r0, r1, seconds)."""
    r0, r1 = task
    t0 = time.time()
    g = _G
    X, indptr, indices, A = g["X"], g["indptr"], g["indices"], g["A"]
    lo, hi = int(indptr[r0]), int(indptr[r1])
    offs = np.asarray(indptr[r0:r1], dtype=np.int64) - lo
    deg = np.asarray(indptr[r0 + 1:r1 + 1], dtype=np.int64) - \
        np.asarray(indptr[r0:r1], dtype=np.int64)

    self_rows = np.asarray(X[r0:r1], dtype=np.float64)
    out = self_rows                      # start from the self term
    if hi > lo:
        nbr = np.asarray(indices[lo:hi], dtype=np.int64)
        # Gathering in sorted neighbour order keeps the mmap access sequential-ish.
        gathered = X[nbr].astype(np.float64, copy=False)
        valid = deg > 0
        if valid.any():
            # reduceat over the starts of the NON-EMPTY rows only: the span
            # between two consecutive valid starts covers exactly that row's
            # neighbours (empty rows in between contribute nothing).
            sums = np.add.reduceat(gathered, offs[valid], axis=0)
            out[valid] += sums
        del gathered, nbr
    out /= (deg + 1.0)[:, None]
    A[r0:r1] = out.astype(np.float32)
    return r0, r1, time.time() - t0


def make_blocks(indptr, N, nnz_budget, max_rows):
    """Partition rows into blocks bounded by both nnz and row count."""
    blocks = []
    r = 0
    while r < N:
        target = int(indptr[r]) + nnz_budget
        hi = int(np.searchsorted(indptr, target, side="right")) - 1
        hi = max(hi, r + 1)
        hi = min(hi, r + max_rows, N)
        blocks.append((r, hi))
        r = hi
    return blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/workspace/data/ogb/arxiv")
    ap.add_argument("--workers", type=int, default=0,
                    help="0 = auto (min(8, cpu_count))")
    ap.add_argument("--nnz-budget", type=int, default=2_000_000,
                    help="max neighbours gathered per block")
    ap.add_argument("--max-rows", type=int, default=1 << 20)
    args = ap.parse_args()

    with open(os.path.join(args.data, "meta.txt")) as f:
        N, F, E, C = [int(x) for x in f.read().split()]
    print(f"[agg] N={N} F={F} E={E} C={C}", flush=True)

    out_path = os.path.join(args.data, "agg_features.f32")
    need = N * F * 4
    if not (os.path.exists(out_path) and os.path.getsize(out_path) == need):
        # Preallocate the output file (sparse) so workers can mmap r+ slices.
        with open(out_path, "wb") as f:
            f.truncate(need)
        print(f"[agg] preallocated {out_path} ({need / 2**30:.2f} GiB)", flush=True)

    indptr = np.memmap(os.path.join(args.data, "graph.csr"), dtype=np.int64,
                       mode="r", offset=16, shape=(N + 1,))
    blocks = make_blocks(indptr, N, args.nnz_budget, args.max_rows)
    del indptr
    nw = args.workers or min(8, mp.cpu_count())
    print(f"[agg] {len(blocks)} blocks, {nw} workers", flush=True)

    t0 = time.time()
    done = 0
    if nw <= 1:
        _init(args.data, N, F, E)
        for b in blocks:
            _block(b)
            done += 1
            if done % 50 == 0:
                print(f"[agg]  {done}/{len(blocks)} blocks "
                      f"({time.time() - t0:.0f}s)", flush=True)
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(nw, initializer=_init,
                      initargs=(args.data, N, F, E)) as pool:
            for _ in pool.imap_unordered(_block, blocks, chunksize=1):
                done += 1
                if done % 50 == 0:
                    print(f"[agg]  {done}/{len(blocks)} blocks "
                          f"({time.time() - t0:.0f}s)", flush=True)
    dt = time.time() - t0

    A = np.memmap(out_path, dtype=np.float32, mode="r", shape=(N, F))
    probe = np.asarray(A[:min(N, 100000)])
    print(f"[agg] wrote {out_path}  ({need / 2**20:.1f} MiB) in {dt:.1f}s  "
          f"A[0,:3]={A[0, :3]}  mean|A[:1e5]|={np.abs(probe).mean():.4f}",
          flush=True)


if __name__ == "__main__":
    main()
