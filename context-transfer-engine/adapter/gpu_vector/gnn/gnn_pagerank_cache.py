#!/usr/bin/env python3
# Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
# BSD 3-Clause License.
#
# PageRank page-access prediction vs simple caching, for a single-node
# shared-memory GNN whose feature matrix is paged and partially resident in GPU
# HBM (the rest compressed with zstd in host/shared memory -- our Eternia stack).
#
# Reproduces the (reverse-)PageRank feature-cache-prediction technique (ACM DOI
# 10.1145/3754598.3754630, reconstructed from the known method) on ogbn-arxiv and
# compares its cache HIT RATE against simple policies (LRU / first-C / random /
# degree). Each cache MISS in our system = one zstd decompress + refetch of a
# feature page; a HIT is served resident. So a higher hit rate = fewer zstd
# decompressions and fewer bytes refetched.
#
# Reverse-PageRank (PageRank on the transposed graph) approximates how often a
# node's features are READ as an in-neighbor during message passing -- the access
# predictor. We pin the top-C pages by aggregated page score.
#
# KEY SYSTEMS FINDING (see the granularity sweep): with COARSE feature pages
# (many nodes per page) every minibatch touches nearly every page, so page-level
# access is ~uniform and PageRank pinning is no better than random. The skew is at
# the NODE level; to expose it in a PAGED store you must EITHER page finely OR
# REORDER nodes by reverse-PageRank so hot nodes cluster into a few pages. With
# PR-reordering, a coarse-paged store recovers most of the node-level hit rate --
# which is exactly the layout knob our Eternia vector controls.
#
# Usage: python3 gnn_pagerank_cache.py [--data /workspace/data/ogb/arxiv]
#        [--batch 1024] [--damping 0.85] [--iters 50] [--md-out out.md]

import argparse
import os
import gzip

import numpy as np

try:
    import pandas as pd
    HAVE_PANDAS = True
except Exception:
    HAVE_PANDAS = False


def read_edges(raw_dir, edges_npy=None):
    """Load the DIRECTED edge list. Returns (src, dst) int64 arrays.

    arxiv/products ship raw CSVs; ogbn-papers100M ships an edge_index .npy
    (shape (2,E)) unpacked by gnn_prep.py, which is memory-mapped rather than
    parsed so a 1.6e9-edge list costs no extra RAM.
    """
    if edges_npy:
        ei = np.load(edges_npy, mmap_mode="r")
        if ei.shape[0] != 2:
            ei = ei.T
        return np.asarray(ei[0], dtype=np.int64), np.asarray(ei[1], dtype=np.int64)
    path = os.path.join(raw_dir, "edge.csv.gz")
    if HAVE_PANDAS:
        df = pd.read_csv(path, header=None, compression="gzip", dtype=np.int64)
        e = df.to_numpy()
    else:
        with gzip.open(path, "rt") as f:
            rows = [ln.split(",") for ln in f if ln.strip()]
        e = np.asarray(rows, dtype=np.int64)
    return e[:, 0], e[:, 1]


def build_csr(key, val, N):
    """Group `val` by `key` into CSR (indptr[N+1], indices) sorted by key."""
    deg = np.bincount(key, minlength=N).astype(np.int64)
    indptr = np.zeros(N + 1, dtype=np.int64)
    np.cumsum(deg, out=indptr[1:])
    order = np.argsort(key, kind="stable")
    indices = val[order].astype(np.int64, copy=False)
    return indptr, indices, deg


def pagerank(src, dst, N, damping=0.85, iters=50, tol=1e-10):
    """Power-iteration PageRank of the graph with edges src->dst."""
    outdeg = np.bincount(src, minlength=N).astype(np.float64)
    dangling = outdeg == 0.0
    inv_out = np.where(outdeg > 0, 1.0 / np.maximum(outdeg, 1.0), 0.0)
    pr = np.full(N, 1.0 / N, dtype=np.float64)
    last_delta = 1.0
    used = iters
    for it in range(iters):
        contrib = (pr * inv_out)[src]
        rank = np.bincount(dst, weights=contrib, minlength=N)
        dsum = pr[dangling].sum()
        newpr = (1.0 - damping) / N + damping * (rank + dsum / N)
        last_delta = np.abs(newpr - pr).sum()
        pr = newpr
        if last_delta < tol:
            used = it + 1
            break
    return pr, last_delta, used


def sample_nbrs(nodes, indptr, indices, fanout, rng):
    """Sample up to `fanout` in-neighbors per node (vectorized, with replacement).
    Realistic GraphSAGE minibatch sampling."""
    nodes = np.asarray(nodes, dtype=np.int64)
    starts = indptr[nodes]
    deg = indptr[nodes + 1] - starts
    take = np.minimum(deg, fanout)
    total = int(take.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    pick = np.repeat(np.arange(len(nodes), dtype=np.int64), take)
    rand = (rng.random(total) * deg[pick]).astype(np.int64)
    return indices[starts[pick] + rand]


def build_minibatch_node_batches(N, in_indptr, in_indices, batch, fanout, seed=0,
                                 max_batches=0):
    """PRIMARY trace: fanout-limited GraphSAGE sampling. Returns a list of per-batch
    NODE-id arrays = unique({seeds} U sampled-1hop U sampled-2hop). Page-granularity
    independent, so each granularity re-pages the same node batches."""
    rng = np.random.default_rng(seed)
    seeds_all = np.arange(N, dtype=np.int64)
    rng.shuffle(seeds_all)
    batches = []
    # A full epoch over ogbn-papers100M is ~108k minibatches; sampling a random
    # subset of them estimates the same hit rate at a fraction of the cost. The
    # seed order is already shuffled, so a prefix IS a uniform random subset.
    n_all = (N + batch - 1) // batch
    n_take = n_all if max_batches <= 0 else min(n_all, max_batches)
    for bi in range(n_take):
        b0 = bi * batch
        seeds = seeds_all[b0:b0 + batch]
        h1 = np.unique(sample_nbrs(seeds, in_indptr, in_indices, fanout[0], rng))
        h2 = (np.unique(sample_nbrs(h1, in_indptr, in_indices, fanout[1], rng))
              if h1.size else np.empty(0, dtype=np.int64))
        batches.append(np.unique(np.concatenate([seeds, h1, h2])))
    return batches


def top_c_mask(score, npages, c):
    mask = np.zeros(npages, dtype=bool)
    if c > 0:
        mask[np.argpartition(-score, min(c, npages) - 1)[:c]] = True
    return mask


def hitrate_lru(pages, capacity):
    """Dynamic LRU of `capacity` pages over the access sequence. O(len(pages))."""
    if capacity <= 0 or len(pages) == 0:
        return 0.0
    from collections import OrderedDict
    cache = OrderedDict()
    hits = 0
    for p in pages:
        pi = int(p)
        if pi in cache:
            hits += 1
            cache.move_to_end(pi)
        else:
            cache[pi] = True
            if len(cache) > capacity:
                cache.popitem(last=False)
    return hits / len(pages)


def evaluate(rows_per_page, N, node_predictor, node_deg, batches, sweep_node_seq,
             budgets, do_lru, rng, lru_limit=0):
    """Hit rates for every policy at one page granularity. Returns
    {trace: {policy: {pct: hitrate}}} plus (npages, page_seqs)."""
    npages = (N + rows_per_page - 1) // rows_per_page
    node_page = np.arange(N, dtype=np.int64) // rows_per_page

    # Node reorder by descending reverse-PR -> hot nodes cluster into low pages.
    order = np.argsort(-node_predictor, kind="stable")
    new_id = np.empty(N, dtype=np.int64)
    new_id[order] = np.arange(N, dtype=np.int64)
    new_node_page = new_id // rows_per_page

    page_pr = np.bincount(node_page, weights=node_predictor, minlength=npages)
    page_deg = np.bincount(node_page, weights=node_deg.astype(np.float64),
                           minlength=npages)

    # Access sequences (per trace): coarse page id per node access.
    #  minibatch: per batch, each needed page counted once (loaded page reused).
    prim = np.concatenate([np.unique(node_page[b]) for b in batches])
    prim_re = np.concatenate([np.unique(new_node_page[b]) for b in batches])
    sec = node_page[sweep_node_seq]
    sec_re = new_node_page[sweep_node_seq]
    traces = {"minibatch": (prim, prim_re), "full-sweep": (sec, sec_re)}

    out = {}
    for tn, (pages, pages_re) in traces.items():
        pol = {p: {} for p in ["PageRank-pin", "PageRank-REORDER+pin",
                               "degree-pin", "first-C", "random-C", "LRU"]}
        for pct in budgets:
            c = max(1, int(round(pct / 100.0 * npages)))
            pr_mask = top_c_mask(page_pr, npages, c)
            dg_mask = top_c_mask(page_deg, npages, c)
            rmask = np.zeros(npages, dtype=bool)
            rmask[rng.choice(npages, size=min(c, npages), replace=False)] = True
            pol["PageRank-pin"][pct] = float(pr_mask[pages].mean())
            pol["PageRank-REORDER+pin"][pct] = float((pages_re < c).mean())
            pol["degree-pin"][pct] = float(dg_mask[pages].mean())
            pol["first-C"][pct] = float((pages < c).mean())
            pol["random-C"][pct] = float(rmask[pages].mean())
            # LRU is a pure-Python O(n) walk; cap the prefix so a 3.2e9-access
            # papers100M sweep does not run for days (reported as a prefix estimate).
            lru_pages = pages if not lru_limit else pages[:lru_limit]
            pol["LRU"][pct] = (hitrate_lru(lru_pages, c) if do_lru else float("nan"))
        out[tn] = pol
    return npages, traces, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/workspace/data/ogb/arxiv")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--damping", type=float, default=0.85)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--md-out", default=None)
    ap.add_argument("--edges", default=None,
                    help="edge_index .npy (papers100M); default reads raw/edge.csv.gz")
    ap.add_argument("--max-batches", type=int, default=0,
                    help="cap sampled minibatches (0 = a full epoch)")
    ap.add_argument("--sweep-sample", type=int, default=0,
                    help="subsample the full-sweep read sequence to this many "
                         "accesses (0 = all; papers100M has 3.2e9)")
    ap.add_argument("--lru-limit", type=int, default=20_000_000,
                    help="max accesses fed to the O(n) Python LRU (0 = no cap)")
    args = ap.parse_args()

    with open(os.path.join(args.data, "meta.txt")) as f:
        N, F, E_und, C = [int(x) for x in f.read().split()]
    row_bytes = F * 4
    print(f"[PR] dataset N={N} F={F} C={C} row={row_bytes}B")

    src, dst = read_edges(os.path.join(args.data, "raw"), args.edges)
    E = src.shape[0]
    outdeg = np.bincount(src, minlength=N).astype(np.int64)
    indeg = np.bincount(dst, minlength=N).astype(np.int64)
    # Message passing computes u by reading its IN-neighbors; node v is READ once
    # per out-edge of v, so read-frequency(v) = outdeg(v). Reverse-PR (~outdeg
    # importance, multi-hop) is the predictor; total degree is the simple proxy.
    tot_deg = outdeg + indeg
    in_indptr, in_indices, _ = build_csr(dst, src, N)
    print(f"[PR] directed E={E}  avg_out={outdeg.mean():.2f} avg_in={indeg.mean():.2f}")

    pr_fwd, d1, i1 = pagerank(src, dst, N, args.damping, args.iters)
    pr_rev, d2, i2 = pagerank(dst, src, N, args.damping, args.iters)
    print(f"[PR] forward PR {i1} it L1={d1:.2e} | reverse PR {i2} it L1={d2:.2e}")
    assert d1 < 1e-4 and d2 < 1e-4, "PageRank did not converge"
    predictor = pr_rev

    # Node-access skew sanity: what fraction of full-sweep reads hit the top-10%
    # nodes by reverse-PR (the paper's NODE-level cache).
    reads = outdeg  # read-frequency per node in the full-sweep
    topk = order10 = np.argsort(-predictor)[:max(1, N // 10)]
    node_hit10 = reads[topk].sum() / reads.sum()
    print(f"[PR] NODE-level: top-10% nodes by reverse-PR capture "
          f"{node_hit10*100:.1f}% of full-sweep feature reads (skew check)")

    fanout = (15, 10)
    batches = build_minibatch_node_batches(N, in_indptr, in_indices, args.batch,
                                           fanout, seed=0,
                                           max_batches=args.max_batches)
    sweep_node_seq = in_indices  # each in-neighbor read once, in id order
    if args.sweep_sample and sweep_node_seq.shape[0] > args.sweep_sample:
        step = sweep_node_seq.shape[0] // args.sweep_sample
        sweep_node_seq = np.ascontiguousarray(sweep_node_seq[::step])
        print(f"[PR] full-sweep subsampled 1-in-{step} -> {len(sweep_node_seq):,} "
              f"accesses (of {len(in_indices):,})")
    prim_total = int(sum(np.unique(b).size for b in batches))
    print(f"[PR] minibatch batches={len(batches)} | full-sweep reads={len(sweep_node_seq):,}")

    # Granularity sweep: rows/page from node-level (1) to a 256 KiB page (512).
    rpp_list = [1, 8, 64, 512]  # 1 row, 4 KiB, 32 KiB, 256 KiB pages
    budgets = [1, 2, 5, 10, 20, 50]
    rng = np.random.default_rng(0)

    lines = []
    def emit(s=""):
        print(s); lines.append(s)

    # The dataset name was hardcoded to ogbn-arxiv, which mislabels every report
    # produced for any other dataset; take it from the data directory instead.
    ds_name = os.path.basename(os.path.abspath(args.data).rstrip("/")) or "unknown"
    emit(f"\n## PageRank page-access prediction vs simple caching ({ds_name})\n")
    emit(f"N={N} nodes, F={F} (row={row_bytes} B), directed E={E:,}, reverse-PR "
         f"damping {args.damping}/{args.iters} it. GraphSAGE fanout {fanout}, "
         f"batch {args.batch}. Each MISS = one zstd decompress + page refetch.\n")
    emit(f"**Node-level skew:** the top **10%** of nodes by reverse-PageRank absorb "
         f"**{node_hit10*100:.1f}%** of all feature reads in a full-graph sweep "
         f"(vs 10% for a random 10%). That skew is the whole opportunity.\n")

    # Per-granularity hit-rate @ 10% budget for the headline table, plus a full
    # budget table at node-level and at the 256 KiB page.
    gran_rows = []
    detail = {}
    for rpp in rpp_list:
        do_lru = (rpp >= 8)  # LRU loop is O(seq); skip only if seq huge (node-lvl minibatch)
        do_lru_here = do_lru or (prim_total < 3_000_000)
        npages, traces, res = evaluate(rpp, N, predictor, tot_deg, batches,
                                       sweep_node_seq, budgets, do_lru_here, rng,
                                       lru_limit=args.lru_limit)
        detail[rpp] = (npages, res)
        pk = f"{rpp} row" + ("s" if rpp != 1 else "") + f" ({rpp*row_bytes//1024 if rpp*row_bytes>=1024 else rpp*row_bytes}{'KiB' if rpp*row_bytes>=1024 else 'B'}, {npages} pages)"
        for tn in ("minibatch", "full-sweep"):
            r = res[tn]
            gran_rows.append((pk, tn, r["PageRank-pin"][10],
                              r["PageRank-REORDER+pin"][10], r["degree-pin"][10],
                              r["first-C"][10], r["random-C"][10], r["LRU"][10]))

    emit("### Hit rate @ 10% cache budget, by page granularity and trace\n")
    emit("| page granularity | trace | PR-pin | PR-REORDER+pin | degree-pin | first-C | random-C | LRU |")
    emit("|---|---|---|---|---|---|---|---|")
    for pk, tn, a, b, c_, d, e, l in gran_rows:
        lv = "n/a" if (l != l) else f"{l*100:.1f}"
        emit(f"| {pk} | {tn} | **{a*100:.1f}** | **{b*100:.1f}** | {c_*100:.1f} | {d*100:.1f} | {e*100:.1f} | {lv} |")

    # Full budget curve at node level (rpp=1) and 256 KiB page (rpp=512).
    for rpp in (1, 512):
        npages, res = detail[rpp]
        emit(f"\n### Full budget sweep @ {rpp} row/page ({npages} pages)\n")
        for tn in ("minibatch", "full-sweep"):
            emit(f"_{tn}_\n")
            emit("| policy | " + " | ".join(f"{p}%" for p in budgets) + " |")
            emit("|" + "---|" * (len(budgets) + 1))
            for pol in ["PageRank-pin", "PageRank-REORDER+pin", "degree-pin",
                        "first-C", "random-C", "LRU"]:
                vals = []
                for p in budgets:
                    v = res[tn][pol][p]
                    vals.append("n/a" if v != v else f"{v*100:.1f}")
                emit(f"| {pol} | " + " | ".join(vals) + " |")

    # Headline savings @ 10%, node level, full-sweep.
    npages1, res1 = detail[1]
    total = len(sweep_node_seq)
    pr_hit = res1["full-sweep"]["PageRank-pin"][10]
    simple = res1["full-sweep"]["random-C"][10]
    pr_miss = int(round((1 - pr_hit) * total))
    s_miss = int(round((1 - simple) * total))
    red = (s_miss - pr_miss) / s_miss * 100.0 if s_miss else 0.0
    emit(f"\n### Headline (node granularity, full-sweep, 10% budget)\n")
    emit(f"- PageRank-pin hit rate **{pr_hit*100:.1f}%** vs random **{simple*100:.1f}%**.\n"
         f"- Over one epoch ({total:,} feature reads): PageRank serves {pr_miss:,} "
         f"misses vs random {s_miss:,} -> **{red:.1f}% fewer zstd decompressions / "
         f"refetches**. Each avoided miss saves one {row_bytes}-B feature "
         f"decompress+copy.\n")

    if args.md_out:
        with open(args.md_out, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[PR] wrote {args.md_out}")


if __name__ == "__main__":
    main()
