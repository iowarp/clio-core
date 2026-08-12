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

## Step 3 (Option B, stretch): Vector<T> + zstd streaming — WORKS (bit-exact PASS)

Added CPU-codec D2H/H2D **staging** to the compressor (`compressor_runtime.cc`) so a
CPU codec (zstd) works on a **device** `blob_data_` in the gpu_vector streaming path,
without touching the GPU-codec (cuSZp) path:
- `CompressPutBlobImpl`: if (CPU codec `wire<=10` AND device ptr) `cudaMemcpy` D2H
  before `Compress`.
- `DecompressGetBlobImpl`: for a CPU codec, query the **exact** stored size
  (`GetBlobSize`) so zstd is not fed the cuSZp-style over-read; if the output is a
  device slot, decompress into a host buffer then H2D-stage. The H2D copy runs on a
  dedicated **non-blocking** stream — a default-stream copy deadlocks behind the
  on-device fault kernel that spin-waits on the very slot the copy must fill.
- Gated on `(CpuCodecWire && PtrIsDevice)` via `cudaPointerGetAttributes`, so cuSZp
  is byte-for-byte unchanged.

**Regression (observed):**
- `test_gpu_vector_compress` with `CLIO_CTE_COMPRESS_LIB=cuszp` → PASS (unchanged).
- `test_gpu_vector_compress` with `CLIO_CTE_COMPRESS_LIB=zstd` → PASS — zstd now
  round-trips through the real `gv::Vector` **on-device fault** streaming path
  (previously segfaulted / deadlocked).

**GNN streaming variant** (`test/test_gpu_vector_gnn_stream_gpu.cc`, target
`test_gpu_vector_gnn_stream`, port 10596): the same GraphSAGE forward, but features
are sourced through `gv::Vector<float>` + `SequentialTransaction` (zstd, staged),
swept into the full feature buffer, then forwarded. **Observed PASS** on ogbn-arxiv:
- feature memcmp = 0, logit memcmp = 0, max_abs_diff = 0.0 (bit-exact)
- store 169.7 MB/s; stream 238.2 MB/s with only a **window=4 pages (1 MiB) resident**
  vs the 82 MiB dataset — the capacity/streaming decoupling, now with lossless zstd.

## Step 4 (scale): ogbn-products — Option A PASS (bit-exact)

ogbn-products (verified against OGB stats): N = 2,449,029 nodes, F = 100 float32,
C = 47 classes, directed E = 61,859,140 → undirected deduped CSR E = 123,718,152
(avg deg 50.5); `features.f32` = 934.2 MiB, `graph.csr` = 962.6 MiB.

Option A on products (`CLIO_GNN_DATA=/workspace/data/ogb/products`, zstd balanced,
hidden H=128 to fit the 8 GB GPU, **8 MiB pages**, `CLIO_GNN_COMPONENTS=0`), observed
PASS:

| metric | value |
|--------|-------|
| features logical → stored | 934.23 MiB → 864.59 MiB (**1.081×**) |
| compress / decompress | 291 MB/s / 625 MB/s |
| forward ms (base / comp) | 511.8 / 539.8 |
| bit-exact | feature memcmp=0, logit memcmp=0, max_abs_diff=0.0 → **PASS** |

**Page-size gotcha (recorded):** with the default 256 KiB pages, products needs
3,739 store round-trips and the store *stalls* (>400 s) — each `AsyncPutBlob().Wait()`
is a full client↔runtime round-trip, so thousands of them dominate. Bumping to
**8 MiB pages (`CLIO_GNN_PAGE_KIB=8192`, 117 pages)** stores the 934 MiB matrix in
3.2 s. Same true ratio; far fewer round-trips. arxiv (331×256 KiB) is small enough
that this never bit. The 8 GB budget was respected (baseline resident forward of
~5.9 GB device succeeded, guarded by the H=128 activation size).

## Step 5: PageRank page-access prediction vs simple caching (single-node shared memory)

Reproduces the (reverse-)PageRank feature-cache-prediction technique (ACM DOI
10.1145/3754598.3754630, reconstructed from the known reverse-PR data-tiering method)
and compares its **cache hit rate** to simple policies, in the single-node shared-memory
regime the Eternia vector targets: the feature matrix lives compressed (zstd) in
host/shared memory, the GPU keeps a resident cache of C pages, and **each miss = one
zstd decompress + page refetch, each hit is served resident.** Tool:
`gnn/gnn_pagerank_cache.py` (numpy over the real ogbn-arxiv **directed** graph;
reverse-PR power iteration, damping 0.85, 50 it, converged L1≈2e-6). All numbers observed.

**Predictor.** Message passing computes node u by reading its in-neighbors, so node v is
read once per out-edge of v → read-frequency(v)=outdeg(v). **Reverse-PageRank** (PR on the
transposed graph) ranks nodes by this multi-hop read importance.

**Node-level access is skewed:** the top 10% of nodes by reverse-PR absorb **21.7%** of
all feature reads in a full sweep (2.2× a random 10%); on the multi-hop *sampled* trace the
skew compounds further (below). That skew is the entire opportunity.

### Hit rate @ 10% cache budget, by page granularity (GraphSAGE fanout 15/10, batch 1024)

| page granularity | trace | PR-pin | PR-REORDER+pin | degree-pin | first-C | random-C | LRU |
|---|---|---|---|---|---|---|---|
| 1 row (512 B)    | minibatch (sampled) | **30.2** | 30.2 | 22.3 | 9.9 | 10.1 | 18.6 |
| 1 row (512 B)    | full-sweep          | 21.7 | 21.7 | 31.6 | 9.9 | 10.1 | 21.9 |
| 64 rows (32 KiB) | minibatch (sampled) | 11.0 | **12.9** | 9.8 | 10.0 | 10.0 | — |
| 512 rows (256 KiB)| minibatch (sampled)| 10.0 | 10.0 | 10.0 | 10.0 | 10.0 | — |
| 512 rows (256 KiB)| full-sweep         | 11.2 | **21.7** | 10.6 | 9.9 | 9.9 | 15.4 |

Full budget curve, **node granularity, sampled trace** (the paper's regime):

| policy | 1% | 2% | 5% | 10% | 20% | 50% |
|---|---|---|---|---|---|---|
| PageRank-pin | 5.3 | 9.1 | 18.4 | **30.2** | **47.9** | 80.7 |
| degree-pin   | 1.6 | 3.6 | 10.6 | 22.3 | 41.7 | 77.4 |
| LRU          | 0.0 | 0.0 | 10.3 | 18.6 | 36.1 | 70.5 |
| first-C/random | ~1 | ~2 | ~5 | ~10 | ~20 | ~50 |

**Findings (honest):**
1. **PageRank strongly beats the simple policies** on the realistic sampled trace at fine
   granularity: **30.2% vs ~10%** (first-C/random) and **18.6%** (LRU) at a 10% budget —
   3× the naive hit rate — reaching **~48% at 20%** budget, matching the paper's ~50% regime.
2. **PageRank beats degree specifically in the multi-hop sampled regime** (30.2% vs 22.3%
   @ 10%), where PR's multi-hop importance matters. In the trivial 1-hop full sweep,
   read-frequency *equals* out-degree, so degree is the exact oracle and edges PR there
   (31.6% vs 21.7%) — reported for honesty; the sampled regime is the realistic GNN pattern.
3. **Coarse pages destroy the skew — reordering recovers it.** At 256 KiB pages a page holds
   512 arbitrary nodes, so every batch touches ~every page and PR-pin collapses to random
   (10%). **Reordering nodes by reverse-PR** so hot nodes cluster into low pages restores the
   benefit: **21.7% vs 11.2%** (no reorder) / **9.9%** (random) at 256 KiB full-sweep. This is
   the concrete feature-layout knob for the Eternia vector.

**Tie to Eternia + compression.** PR-guided residency (or PR feature reordering) lets a
smaller resident HBM window reach the same hit rate → fewer cold faults → **fewer zstd
decompress+refetch operations**: at a 10% budget, misses drop from ~90% (random) to ~70%
(sampled, PR) — **~22% fewer decompressions** — and ~13% on the 1-hop full sweep. The PR
page scores can feed the gpu_vector cache manager's existing per-page `score` (Phase-1
rescore) directly, and PR feature-reordering is a one-time preprocessing of the stored
matrix. This is where PageRank prediction turns the bounded-window Eternia store from
"correct" into "efficient."

## How to reproduce
```
# data
python3 gnn/gnn_prep.py --dataset arxiv
# PageRank hit-rate study (Step 5)
python3 gnn/gnn_pagerank_cache.py --data /workspace/data/ogb/arxiv
# build
cmake --build /workspace/build_km_cuda --target test_gpu_vector_gnn -j
# run (per preset)
cd /workspace/build_km_cuda/bin
CLIO_PORT=10594 CLIO_CTE_COMPRESS_LIB=zstd CLIO_CTE_COMPRESS_PRESET=balanced \
  LD_LIBRARY_PATH=. CLIO_REPO_PATH=. CLIO_BIND_ADDR=127.0.0.1 ./test_gpu_vector_gnn
```
