# gpu_vector full weight ownership for llama.cpp

**Goal:** llama.cpp stops managing tensor-weight loading entirely. Every weight
byte lives in the clio gpu_vector (CTE-backed, optionally compressed); every op
that consumes a weight reads it through the vector. llama's loader neither
allocates VRAM slabs for weights nor reads tensor data from the GGUF. The HBM
page cache is configurable and defaults to **80% of free GPU memory** at init.

Builds on the completed milestone (branch `gpu-vector-weight-resume`):
batch-1 F16 GEMV streaming works, compressed store round-trips bit-exactly,
windowed oversubscription works (CUDA graphs auto-disabled when windowed).

## Where the milestone falls short of the goal

Today `load_tensors` runs llama's normal path first — `create_tensors`
allocates full CUDA slabs, `load_all_data` copies every weight from the GGUF —
and the vector merely *overlays* the batch-1 F16 GEMV read path afterwards.
Everything else silently falls back to stock CUDA reading `t->data` from
llama's copy:

- batched matmul (whole prompt/prefill phase): `batched=330` of ~390 calls
- `GGML_OP_GET_ROWS` on the token embedding
- non-F16 tensors (norms, biases — F32)
- oversized tensors (`too_big`)

So weights are resident twice (llama slab + vector cache) and windowed mode
saves no VRAM. The fallbacks are load-bearing: llama's buffers cannot be
dropped until coverage is complete.

## Hard constraints

1. **No stable device pointer for paged weights.** In windowed mode pages move
   between HBM slots, so `t->data` can never point at vector storage; every
   consuming op must be intercepted and go through the slot table.
2. **ggml allocates per-context slabs, not per-tensor** — you cannot free one
   tensor out of a slab. Vector-served tensors must never enter a real slab:
   they need their own no-alloc buffer type.
3. **Graph capture:** host-driven window faults cannot run under CUDA graph
   capture (already handled: graphs disabled when windowed).

## Phases (each gated by the byte-identical correctness harness)

### Phase A — cache autosizing (trivial, first)
`ggml_cuda_clio_vector_init`: when `CLIO_GPU_PAGES` is unset or 0, query
`cudaMemGetInfo` and size the cache as
`pages = floor(free_bytes * pct / 100 / page_size)`, `pct` from
`CLIO_GPU_CACHE_PCT` (default **80**). Cap at the model's page count while the
vector holds a single model (no benefit past fully-resident; revisit when
multiple tags share one cache). Log the decision.
Note: KV cache + compute buffers allocate *after* model load, so they live in
the remaining ~20% — the env knobs exist precisely for workloads where that is
too tight.

### Phase B — full mul_mat coverage (kills the `batched` fallback)
Extend the fast kernel into a paged GEMM-lite: keep the
one-warp-per-row structure, loop over the `ne1` activation columns (activations
are small: ne1 × K floats; keep in registers/smem tile). Remove the batch-1
gate in `ggml_cuda_clio_vector_mul_mat`. Prefill throughput need not match
cuBLAS initially — correctness first, tile later.
Also lift `too_big`: a tensor larger than the window streams by chunking rows
per window load (row range → page range → process → advance window).

### Phase C — get_rows + small-tensor materialization
- Paged `get_rows` kernel for the token-embedding matrix (token id → row →
  page/slot → gather), intercepted at `GGML_OP_GET_ROWS` when src0 is
  registered.
- Small weights (norms, biases, rope factors — KB-scale, F32): materialized
  ONCE from vector pages into a dedicated small arena at load; `t->data`
  points there so stock kernels keep working. llama still never reads the
  GGUF data section — the bytes come out of CTE. (These stay resident; paging
  KB-scale tensors buys nothing.)

### Phase D — llama stops loading weights
- New `ggml_backend_clio_buffer_type`: `alloc_buffer` reserves no VRAM for
  paged tensors (distinct sentinel `t->data` values, never dereferenced —
  debug-assert trap on any non-intercepted op) and real arena space for
  Phase-C materialized tensors.
- Loader: tensors assigned to the clio buffer type skip `load_all_data`
  entirely (the data was assimilated into CTE by clio-assim / CAE already).
- After this, VRAM = vector cache (80% default) + small arena + KV/compute.
  Oversubscription (model > cache) now genuinely saves memory, and models
  larger than VRAM become loadable.

### Phase E — window policy (perf, after correctness)
The naive wrap-around window thrashes 100% on the sequential layer sweep
(~0.6 t/s at 2x oversubscription). With prefill also paged (Phase B) this
becomes the perf-critical piece: at minimum a ring window that prefetches
page p+W asynchronously while computing layer at page p; ideally overlap
decompress (host) with GEMV (device) via a second stream + pinned staging.

## Verification at every phase
`correctness.sh` three-way (stock / vector / vector-compressed) plus windowed:
greedy temp-0 32-token generation must stay byte-identical to stock. Perf A/B
with `llama-bench -r 3` (warm). VRAM audit via `nvidia-smi` before/after Phase
D — llama's weight slabs must be gone.
