# GNN + lossless-zstd on the compressed GPU vector — RESULTS

Single-node GraphSAGE-mean forward whose **node-feature matrix is stored losslessly
compressed with zstd** through the CTE compressor(600)→core(512) stack, on **ogbn-arxiv**,
with a **bit-exact** correctness guarantee vs an uncompressed baseline.

All numbers below are from real runs in container `iowarp-km` (image `iowarp-coh:latest`),
warm CUDA build `/workspace/build_km_cuda` (arch 90 PTX-JIT → sm_120, RTX 5060 8 GB),
`libzstd` CPU codec pinned via `CLIO_CTE_COMPRESS_LIB=zstd`.

## Dataset (Step 0)
`gnn/gnn_prep.py` downloads the OGB raw zip, parses the raw CSVs (no torch/ogb),
symmetrizes edges into a deduped undirected CSR, and writes flat little-endian binaries.

ogbn-arxiv (verified against OGB stats):
- N = 169,343 nodes, F = 128 float32 features, C = 40 classes
- directed edges = 1,166,243 → undirected deduped CSR E = 2,315,598 (avg deg 13.67)
- `features.f32` = 82.69 MiB, `graph.csr` = 19.0 MiB, `labels.i64` = 1.29 MiB

## Option A (must-have): host-staged zstd store + GraphSAGE forward + bit-exact verify

Test: `test/test_gpu_vector_gnn_gpu.cc` (target `test_gpu_vector_gnn`, port 10594).

Pipeline:
1. Store the feature matrix page-by-page (256 KiB pages = 512 whole rows; 331 pages)
   through the **zstd-pinned compressor from HOST buffers**
   (`ShmPtr{alloc_id=null, off_=(u64)host_ptr}` — zstd is a CPU codec, so it must
   see host memory; no cudaMemcpy-to-device first).
2. True stored size = **exact** sum of `core.AsyncGetBlobSize` per page (equals the
   bytes actually placed: compressed+24 B header, or raw when not beneficial).
3. Read pages back through the compressor's `Decompress` path (host `AsyncGetBlob`,
   passing the **exact** stored size so zstd is not fed trailing bytes) → `cudaMemcpy` H2D.
4. 2-layer GraphSAGE-mean forward on GPU (CSR SpMM-mean aggregation + dense combine,
   hidden 256 → 40 logits, fixed seeded weights). Kernels are deterministic (one thread
   owns one output, fixed-order reduction, **no atomics**).
5. Baseline: same forward with features resident in HBM (no compression).

**Correctness — bit-exact PASS (observed in real output):**
- Feature round-trip `memcmp(decompressed_device_bytes, original) == 0`
- Logits `memcmp(compressed_path, baseline) == 0`, `max_abs_diff == 0.0`

Lossless ⇒ the decompressed features are byte-identical, so identical deterministic
kernels produce byte-identical logits. This is the headline result.

## Step 2: ratio / throughput sweep (zstd fast=lvl1, balanced=lvl3, best=lvl19)

Overall feature-matrix compression (82.69 MiB logical) and per-component single-blob
lossless ratios:

| preset  | feature ratio | stored MiB | compress MB/s | decompress MB/s | forward ms (base / comp) | bit-exact |
|---------|--------------:|-----------:|--------------:|----------------:|-------------------------:|:---------:|
| fast    | 1.078×        | 76.72      | 326.3         | 548.8           | 35.95 / 37.17            | PASS      |
| balanced| 1.078×        | 76.73      | 335.2         | 533.8           | 36.66 / 36.08            | PASS      |
| best    | _pending_     | _pending_  | _pending_     | _pending_       | _pending_                | _pending_ |

Per-component lossless ratio (single blob through the same zstd path):

| component | logical | fast | balanced |
|-----------|--------:|-----:|---------:|
| features (dense float32) | 82.69 MiB | 1.078× | 1.078× |
| CSR indptr (int64)       | 1.29 MiB  | 7.13× | 7.09×  |
| CSR indices (int64)      | 17.67 MiB | 3.22× | 2.99×  |
| labels (int64)           | 1.29 MiB  | 10.07×| 13.78× |

**Honest reading:** dense float32 embeddings barely compress (~1.08× lossless) — the
byte entropy of learned features is high; this is the number to quote for feature
matrices. The integer CSR structure compresses far better (indptr ~7×, indices ~3×,
labels ~10–14×). So the lossless win on a GNN is concentrated in the *graph* arrays,
not the feature matrix. The value proposition of the feature store is therefore the
**bit-exact correctness guarantee + capacity/streaming**, not a large feature ratio.

Note: the forward-ms baseline vs compressed are equal within noise (both run the same
kernels; a warmup pass removes the one-time PTX-JIT cost so neither number is inflated).

## Step 3 (Option B, stretch): Vector<T> + zstd streaming — _pending_

## Step 4 (scale): ogbn-products capacity headline — _pending_

## How to reproduce
```
# data
python3 gnn/gnn_prep.py --dataset arxiv
# build
cmake --build /workspace/build_km_cuda --target test_gpu_vector_gnn -j
# run (per preset)
cd /workspace/build_km_cuda/bin
CLIO_PORT=10594 CLIO_CTE_COMPRESS_LIB=zstd CLIO_CTE_COMPRESS_PRESET=balanced \
  LD_LIBRARY_PATH=. CLIO_REPO_PATH=. CLIO_BIND_ADDR=127.0.0.1 ./test_gpu_vector_gnn
```
