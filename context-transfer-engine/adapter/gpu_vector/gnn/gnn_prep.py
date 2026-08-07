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
    # ogbn-papers100M ships as raw .npz (not CSVs) and is far too large to hold
    # densely in RAM, so it takes a dedicated streaming path (prep_papers100m).
    "papers100M": (
        "http://snap.stanford.edu/ogb/data/nodeproppred/papers100M-bin.zip",
        "papers100M-bin", 111059956, 128, 1615685872, 172),
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


# ---------------------------------------------------------------------------
# ogbn-papers100M: streaming prep (no torch / no ogb package).
#
# The raw payload is .npz (a zip of .npy members), ~60 GB compressed. Nothing
# here materialises a dense copy of the 57 GB feature matrix in RAM: members are
# extracted to bare .npy files, then memory-mapped and converted chunk-by-chunk.
# ---------------------------------------------------------------------------

def _log(msg):
    print(f"[prep] {msg}", flush=True)


def npz_members_to_npy(npz_path, out_dir):
    """Extract each .npy member of an .npz to out_dir so it can be mmapped.

    np.load(npz) streams through the zip and cannot memory-map; unpacking the
    members first lets us mmap arrays far larger than RAM.
    """
    os.makedirs(out_dir, exist_ok=True)
    got = {}
    with zipfile.ZipFile(npz_path) as z:
        for nm in z.namelist():
            if not nm.endswith(".npy"):
                continue
            key = os.path.basename(nm)[:-4]
            dest = os.path.join(out_dir, key + ".npy")
            info = z.getinfo(nm)
            if os.path.exists(dest) and os.path.getsize(dest) == info.file_size:
                _log(f"  member {key}: already unpacked ({info.file_size} B)")
            else:
                _log(f"  member {key}: unpacking {info.file_size} B -> {dest}")
                with z.open(nm) as src, open(dest, "wb") as dst:
                    while True:
                        buf = src.read(1 << 24)
                        if not buf:
                            break
                        dst.write(buf)
            got[key] = dest
    return got


def convert_features_f32(src_npy, out_path, expN, expF, chunk_rows=1 << 20):
    """Memory-map the raw feature array and write float32 row-major in chunks."""
    X = np.load(src_npy, mmap_mode="r")
    if X.ndim != 2:
        raise SystemExit(f"unexpected feature ndim {X.ndim} shape {X.shape}")
    N, F = int(X.shape[0]), int(X.shape[1])
    _log(f"features raw: shape={X.shape} dtype={X.dtype} (expected {expN}x{expF})")
    if N != expN or F != expF:
        raise SystemExit("feature shape mismatch vs OGB stats")
    total = N * F * 4
    if os.path.exists(out_path) and os.path.getsize(out_path) == total:
        _log(f"features.f32 already complete ({total} B) - skipping")
        return N, F
    with open(out_path, "wb") as f:
        for lo in range(0, N, chunk_rows):
            hi = min(lo + chunk_rows, N)
            np.ascontiguousarray(X[lo:hi], dtype=np.float32).tofile(f)
            if (lo // chunk_rows) % 16 == 0:
                _log(f"  features {hi}/{N} rows ({100.0 * hi / N:.1f}%)")
    _log(f"wrote {out_path} ({os.path.getsize(out_path)} B)")
    return N, F


def convert_labels_i64(src_npy, out_path, N):
    """Labels are float with NaN for the unlabelled majority -> -1 (skipped in training)."""
    y = np.load(src_npy, mmap_mode="r")
    lab = np.empty(N, dtype=np.int64)
    chunk = 1 << 22
    for lo in range(0, N, chunk):
        hi = min(lo + chunk, N)
        blk = np.asarray(y[lo:hi]).reshape(hi - lo, -1)[:, 0].astype(np.float64)
        blk = np.where(np.isnan(blk), -1.0, blk)
        lab[lo:hi] = blk.astype(np.int64)
    lab.tofile(out_path)
    n_lab = int((lab >= 0).sum())
    C = int(lab.max()) + 1 if n_lab else 0
    _log(f"labels: {n_lab}/{N} labelled ({100.0 * n_lab / N:.2f}%), num_classes={C}")
    return C, n_lab


def build_csr_papers(edge_npy, N, out_csr, log_every=8):
    """Symmetrised, deduped, sorted undirected CSR for ~1.6e9 directed edges.

    Packs each undirected (u,v) into one uint64 key = u*N+v (max ~1.2e16, no
    overflow), sorts IN PLACE (so peak RAM is one key array, not two), then
    dedups and emits CSR by streaming the sorted keys. Sorted key order is
    exactly (u, then v) order, which is the CSR layout we need.
    """
    ei = np.load(edge_npy, mmap_mode="r")
    if ei.shape[0] != 2:
        ei = ei.T
    E_dir = int(ei.shape[1])
    _log(f"edge_index: shape={ei.shape} dtype={ei.dtype} directed E={E_dir}")

    M = 2 * E_dir
    _log(f"allocating key array: {M} uint64 = {M * 8 / 2**30:.1f} GiB")
    key = np.empty(M, dtype=np.uint64)
    Nu = np.uint64(N)
    chunk = 1 << 26
    for lo in range(0, E_dir, chunk):
        hi = min(lo + chunk, E_dir)
        u = np.asarray(ei[0, lo:hi], dtype=np.uint64)
        v = np.asarray(ei[1, lo:hi], dtype=np.uint64)
        key[lo:hi] = u * Nu + v          # forward
        key[E_dir + lo:E_dir + hi] = v * Nu + u   # reverse (symmetrise)
        if (lo // chunk) % log_every == 0:
            _log(f"  pack {hi}/{E_dir} ({100.0 * hi / E_dir:.1f}%)")
    del u, v

    _log("sorting keys in place (this is the long pole)...")
    key.sort(kind="stable")
    _log("sort done; deduping + emitting CSR")

    # Count unique keys per source, streaming, then write indices in order.
    counts = np.zeros(N, dtype=np.int64)
    uniq_total = 0
    prev_last = None
    scan = 1 << 26
    # Pass 1: counts
    for lo in range(0, M, scan):
        hi = min(lo + scan, M)
        blk = key[lo:hi]
        if prev_last is None:
            keep = np.empty(blk.shape[0], dtype=bool)
            keep[0] = True
            np.not_equal(blk[1:], blk[:-1], out=keep[1:])
        else:
            keep = np.empty(blk.shape[0], dtype=bool)
            keep[0] = blk[0] != prev_last
            np.not_equal(blk[1:], blk[:-1], out=keep[1:])
        prev_last = blk[-1]
        uu = (blk[keep] // Nu).astype(np.int64)
        counts += np.bincount(uu, minlength=N)
        uniq_total += int(keep.sum())
        if (lo // scan) % log_every == 0:
            _log(f"  count {hi}/{M} ({100.0 * hi / M:.1f}%) uniq so far={uniq_total}")

    E = uniq_total
    indptr = np.zeros(N + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    assert int(indptr[-1]) == E, (int(indptr[-1]), E)
    _log(f"undirected deduped CSR: E={E} avg_deg={E / N:.2f}")

    # Pass 2: stream v's out in sorted order.
    with open(out_csr, "wb") as f:
        np.array([N, E], dtype=np.int64).tofile(f)
        indptr.tofile(f)
        prev_last = None
        written = 0
        for lo in range(0, M, scan):
            hi = min(lo + scan, M)
            blk = key[lo:hi]
            keep = np.empty(blk.shape[0], dtype=bool)
            keep[0] = True if prev_last is None else (blk[0] != prev_last)
            np.not_equal(blk[1:], blk[:-1], out=keep[1:])
            prev_last = blk[-1]
            vv = (blk[keep] % Nu).astype(np.int64)
            vv.tofile(f)
            written += vv.shape[0]
            if (lo // scan) % log_every == 0:
                _log(f"  emit {written}/{E} ({100.0 * written / E:.1f}%)")
    assert written == E
    del key
    _log(f"wrote {out_csr} ({os.path.getsize(out_csr)} B)")
    return E


def prep_papers100m(args, url, subdir, expN, expF, expE, expC):
    out = args.out or os.path.join(args.data_root, "papers100M")
    os.makedirs(out, exist_ok=True)
    raw_dir = os.path.join(args.data_root, subdir, "raw")
    zip_path = os.path.join(args.data_root, "papers100M-bin.zip")

    if not os.path.isdir(raw_dir):
        download(url, zip_path)
        _log(f"extracting {zip_path} (60 GB; slow)")
        unzip(zip_path, args.data_root)
    if not os.path.isdir(raw_dir):
        raise SystemExit(f"raw dir {raw_dir} not found after extract")
    _log(f"raw dir: {raw_dir} -> {sorted(os.listdir(raw_dir))}")

    unpack = os.path.join(args.data_root, subdir, "unpacked")
    members = {}
    for npz_name in ("data.npz", "node-label.npz"):
        p = os.path.join(raw_dir, npz_name)
        if os.path.exists(p):
            _log(f"unpacking {npz_name}")
            members.update(npz_members_to_npy(p, unpack))
    _log(f"available members: {sorted(members)}")

    feat_key = next((k for k in ("node_feat", "node_feat_0", "x") if k in members), None)
    edge_key = next((k for k in ("edge_index", "edge_index_0") if k in members), None)
    lab_key = next((k for k in ("node_label", "y", "label") if k in members), None)
    if not feat_key or not edge_key:
        raise SystemExit(f"could not find feature/edge members in {sorted(members)}")

    N, F = convert_features_f32(members[feat_key],
                                os.path.join(out, "features.f32"), expN, expF)
    if lab_key:
        C, _ = convert_labels_i64(members[lab_key], os.path.join(out, "labels.i64"), N)
    else:
        np.full(N, -1, dtype=np.int64).tofile(os.path.join(out, "labels.i64"))
        C = expC
    C = max(C, 1)

    csr_path = os.path.join(out, "graph.csr")
    E = build_csr_papers(members[edge_key], N, csr_path)

    with open(os.path.join(out, "meta.txt"), "w") as f:
        f.write(f"{N} {F} {E} {C}\n")
    _log(f"meta.txt = '{N} {F} {E} {C}'")
    for nm in ("features.f32", "graph.csr", "labels.i64"):
        p = os.path.join(out, nm)
        _log(f"  {nm} = {os.path.getsize(p) / 2**30:.2f} GiB")
    _log("DONE (papers100M)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="arxiv", choices=list(DATASETS.keys()))
    ap.add_argument("--data-root", default="/workspace/data/ogb")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    url, subdir, expN, expF, expE, expC = DATASETS[args.dataset]
    if args.dataset == "papers100M":
        prep_papers100m(args, url, subdir, expN, expF, expE, expC)
        return
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
