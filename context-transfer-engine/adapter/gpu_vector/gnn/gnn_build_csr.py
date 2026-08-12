#!/usr/bin/env python3
# Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
# BSD 3-Clause License.
#
# Build a symmetrised, deduped, sorted undirected CSR from a raw edge_index
# STREAM and write it to stdout in graph.csr layout:
#
#     int64 N, int64 E, indptr[N+1] int64, indices[E] int64
#
# This is gnn_prep.build_csr_papers reworked to take the edge list on stdin
# instead of memory-mapping an unpacked .npy. The point is the same one as
# gnn_ingest: for ogbn-papers100M the unpacked edge_index alone is 25.9 GiB of
# scratch, and it exists only to be read once, sequentially.
#
# THE MEMORY TRICK. edge_index arrives row-major: all 1.6e9 source ids, then all
# 1.6e9 destination ids. Buffering both rows and then packing keys would need
# 12.9 + 12.9 + 25.9 = 51.7 GiB. Instead the key array is allocated once (2*E
# uint64 = 25.9 GiB), row 0 is read directly into its first half, and then row 1
# is streamed in chunks that rewrite that half in place:
#
#     u = key[lo:hi].copy()          # source ids parked there by pass 1
#     key[lo:hi]     = u * N + v     # forward edge
#     key[E+lo:E+hi] = v * N + u     # reverse edge (this is the symmetrisation)
#
# Peak is the key array plus one chunk. Sorting uint64 keys of u*N+v puts them
# in exactly (u, then v) order, which IS the CSR layout, so the sort does the
# grouping for free; dedup is then a linear scan.
#
#   python3 gnn_stream_npz.py --zip X.zip --member edge_index --dtype int64 \
#     | python3 gnn_build_csr.py --nodes 111059956 > /dev/null   # (pipe to consumer)
#
# Prefer piping straight into gnn_aggregate so the CSR never lands on disk.

import argparse
import sys

import numpy as np


def log(msg):
    print(f"[csr] {msg}", file=sys.stderr, flush=True)


def write_array(out, arr, chunk_elems=1 << 24):
    """Write an ndarray to a binary stream in chunks.

    np.ndarray.tofile needs a real seekable file and raises "obtaining file
    position failed" on a pipe -- which is exactly how this tool is meant to be
    run. Chunking also avoids materialising a second copy of a 13 GiB array the
    way tobytes() would.
    """
    for lo in range(0, arr.size, chunk_elems):
        out.write(arr[lo:lo + chunk_elems].tobytes())


def read_exact(fh, buf):
    """Fill `buf` (a writable memoryview) from fh; return bytes read."""
    mv = memoryview(buf).cast("B")
    got = 0
    while got < len(mv):
        n = fh.readinto(mv[got:])
        if not n:
            break
        got += n
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, required=True)
    ap.add_argument("--edges", type=int, default=0,
                    help="directed edge count; 0 = infer from stream length")
    ap.add_argument("--chunk", type=int, default=1 << 24)
    args = ap.parse_args()

    N = args.nodes
    fh = sys.stdin.buffer

    if args.edges > 0:
        E_dir = args.edges
    else:
        raise SystemExit("--edges is required when reading from a pipe "
                         "(the stream length is not knowable in advance); "
                         "get it from gnn_stream_npz.py --info")

    M = 2 * E_dir
    log(f"N={N} directed E={E_dir} -> key array {M} uint64 "
        f"({M * 8 / 2**30:.1f} GiB)")
    key = np.empty(M, dtype=np.uint64)
    Nu = np.uint64(N)

    # Pass 1: source ids land in the first half, to be rewritten in place.
    log("reading source row into key[0:E]")
    got = read_exact(fh, key[:E_dir])
    if got != E_dir * 8:
        raise SystemExit(f"short source row: {got} of {E_dir * 8} bytes")

    # Pass 2: destination ids arrive; build both directions in place.
    log("streaming destination row, packing both directions")
    chunk = max(1, args.chunk)
    vbuf = np.empty(chunk, dtype=np.uint64)
    lo = 0
    while lo < E_dir:
        hi = min(lo + chunk, E_dir)
        n = hi - lo
        got = read_exact(fh, vbuf[:n])
        if got != n * 8:
            raise SystemExit(f"short dest row at {lo}: {got} of {n * 8}")
        u = key[lo:hi].copy()
        v = vbuf[:n]
        key[lo:hi] = u * Nu + v
        key[E_dir + lo:E_dir + hi] = v * Nu + u
        lo = hi
        if (lo // chunk) % 16 == 0:
            log(f"  packed {lo}/{E_dir} ({100.0 * lo / E_dir:.1f}%)")
    del vbuf

    log("sorting keys in place (the long pole)")
    key.sort(kind="stable")

    log("deduping and emitting CSR")
    # Unique mask over the sorted keys; first element always kept.
    keep = np.empty(M, dtype=bool)
    keep[0] = True
    np.not_equal(key[1:], key[:-1], out=keep[1:])
    uniq = int(keep.sum())
    src = (key[keep] // Nu).astype(np.int64)
    dst = (key[keep] % Nu).astype(np.int64)
    del key, keep

    counts = np.bincount(src, minlength=N).astype(np.int64)
    indptr = np.zeros(N + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    if int(indptr[N]) != uniq:
        raise SystemExit(f"indptr[N]={int(indptr[N])} != uniq={uniq}")

    log(f"undirected E={uniq} (indices {uniq * 8 / 2**30:.2f} GiB)")
    out = sys.stdout.buffer
    write_array(out, np.array([N, uniq], dtype=np.int64))
    write_array(out, indptr)
    write_array(out, dst)
    out.flush()
    log("done")


if __name__ == "__main__":
    main()
