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

Feature-matrix (82.69 MiB logical, 331×256 KiB pages) lossless compression, all
bit-exact PASS (real observed output, `gnn_runs/arxiv_{fast,balanced,best}.log`):

| preset (zstd lvl) | feature ratio | stored MiB | compress MB/s | decompress MB/s | forward ms (base / comp) | bit-exact |
|-------------------|--------------:|-----------:|--------------:|----------------:|-------------------------:|:---------:|
| fast (lvl 1)      | 1.078×        | 76.72      | 310.9         | 628.2           | 37.2 / 37.7              | PASS      |
| balanced (lvl 3)  | 1.078×        | 76.73      | 397.1         | 614.1           | 37.1 / 37.8              | PASS      |
| best (lvl 19)     | 1.088×        | 75.99      | 14.1          | 451.5           | 35.5 / 39.3              | PASS      |

Per-component lossless ratio (each component paged through the same zstd path):

| component | logical | fast | balanced |
|-----------|--------:|-----:|---------:|
| features (dense float32) | 82.69 MiB | 1.078× | 1.078× |
| CSR indptr (int64)       | 1.29 MiB  | 7.12× | 7.10×  |
| CSR indices (int64)      | 17.67 MiB | 3.19× | 3.08×  |
| labels (int64)           | 1.29 MiB  | 10.06×| 13.73× |

**Honest reading:**
- Dense float32 embeddings barely compress losslessly: ~1.078× at fast/balanced,
  1.088× at best. Learned-feature byte entropy is high; this is the honest number to
  quote for GNN feature matrices. Level 19 buys ~1% more ratio for ~25× slower
  compress (14 MB/s vs ~400 MB/s) — not worth it for features.
- The integer CSR *structure* compresses far better (indptr ~7×, indices ~3×,
  labels ~10–14×), so the lossless win on a GNN is concentrated in the graph arrays,
  not the feature matrix. The feature store's value is the **bit-exact guarantee +
  capacity/streaming**, not a large feature ratio.
- Forward ms baseline vs compressed are equal within noise (~37 ms) — identical
  deterministic kernels; a warmup pass before each timed forward removes the
  one-time PTX-JIT and post-store GPU-DVFS ramp so neither number is inflated.
- **zstd BEST gotcha:** level-19 optimal-parse is a pathological worst case on the
  sorted int64 CSR indices — a single ~18 MiB blob takes >800 s; even 256 KiB chunks
  take minutes. So the per-component breakdown is opt-out (`CLIO_GNN_COMPONENTS=0`)
  and the BEST sweep reports the (float) feature ratio only. Float feature pages are
  not affected (best store = 5.9 s for 82 MiB).

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
