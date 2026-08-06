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
# Usage: python3 gnn_agg.py --data /workspace/data/ogb/arxiv

import argparse
import os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/workspace/data/ogb/arxiv")
    args = ap.parse_args()

    with open(os.path.join(args.data, "meta.txt")) as f:
        N, F, E, C = [int(x) for x in f.read().split()]
    X = np.fromfile(os.path.join(args.data, "features.f32"),
                    dtype=np.float32).reshape(N, F)
    with open(os.path.join(args.data, "graph.csr"), "rb") as f:
        hdr = np.fromfile(f, dtype=np.int64, count=2)
        assert hdr[0] == N
        indptr = np.fromfile(f, dtype=np.int64, count=N + 1)
        indices = np.fromfile(f, dtype=np.int64, count=int(hdr[1]))
    print(f"[agg] N={N} F={F} E={E} C={C}")

    # A = (X + sum_{j in N(i)} X[j]) / (deg_i + 1), computed by segment-sum over
    # the CSR (float64 accumulate for stability, then cast back to float32).
    deg = (indptr[1:] - indptr[:-1]).astype(np.float64)
    src = np.repeat(np.arange(N, dtype=np.int64), (indptr[1:] - indptr[:-1]))
    acc = X.astype(np.float64).copy()          # self term
    np.add.at(acc, src, X[indices].astype(np.float64))  # + neighbours
    acc /= (deg + 1.0)[:, None]
    A = acc.astype(np.float32)

    out = os.path.join(args.data, "agg_features.f32")
    A.tofile(out)
    print(f"[agg] wrote {out}  ({A.nbytes/2**20:.1f} MiB)  "
          f"A[0,:3]={A[0,:3]}  mean|A|={np.abs(A).mean():.4f}")


if __name__ == "__main__":
    main()
