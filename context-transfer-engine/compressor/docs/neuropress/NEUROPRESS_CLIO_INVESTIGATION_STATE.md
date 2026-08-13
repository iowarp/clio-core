# NeuroPress ↔ CLIO equivalence investigation — state record

**This file is the source of truth for the investigation itself.** Update it at the end
of every phase. On resuming, read this file and the last phase checkpoint before doing
anything else (Protocol §10).

---

## Header

```
Current Phase:        PHASE 29 COMPLETE — callback-trace equivalence harness built,
                      run, and green. Phases 1-28 remain closed; the final verdict
                      is UNCHANGED. Final report:
                      NEUROPRESS_CLIO_EQUIVALENCE_REPORT.md
Current Subtask:      none — 2 measurement items remain open (Phase 21 attribution,
                      task 27 MAPE on 2-D/3-D data); neither can change the verdict
Last updated:         2026-08-13
Native baseline:      /home/cc/NeuroPress @ b23b8f6 (worktree: src/H5Zshuffle.c deleted)
                      built from source at /home/cc/np-build (target `gpucompress`,
                      NVCOMP_PREFIX=/home/cc/np-prefix)
Port baseline:        /home/cc/clio-core @ branch neuropress-693-continued, HEAD 35e08d2d
Build dir:            /home/cc/clio-core/build
```

## Phase status

| # | Phase | Status |
|---|---|---|
| 1 | Native source discovery | **COMPLETE** |
| 2 | Native data/memory flow | **COMPLETE** |
| 3 | Native shuffle/deshuffle | **COMPLETE — byte-identical** |
| 4 | Native quantization/dequantization | **COMPLETE — byte-identical** |
| 5 | Native statistics | **COMPLETE — numerically equivalent (≤5.4e-15 rel)** |
| 6 | Native per-chunk diagnostics | **COMPLETE — DIFFERENCE FOUND (observability only)** |
| 7 | Native static selection | **COMPLETE — DIFFERENCE FOUND (D7-1, high)** |
| 8 | Native exploration | **COMPLETE — DIFFERENCE FOUND (D8-1)** |
| 9 | Native ranking | **COMPLETE — PASS** (D9-1 precision diff, bounded) |
| 10 | Native NN architecture | **COMPLETE — PASS** |
| 11 | Native NN inference | **COMPLETE — PASS, bit-identical** |
| 12 | Native inference + exploration | **COMPLETE — covered by 8/9/11** |
| 13 | Native runtime learning / SGD | **COMPLETE — PASS (~1 ULP)** |
| 14 | Native `.nnwt` behavior | **COMPLETE — PASS, byte-identical file** |
| 15 | Native lossless compression/decompression | **COMPLETE — codec PASS, D15-1 container diverges** |
| 16 | Native static quantized compression/decompression | **COMPLETE — DIFFERENCE FOUND (D16-1)** |
| 17 | Native read/decompression path | **COMPLETE — PASS** (D17-1 hypothesis open) |
| 18 | Native ↔ CLIO cross-compatibility | **COMPLETE — DIFFERENCE FOUND (D18-1)**: selections 256/256 identical, payloads 209/256 differ |
| 19 | CLIO CUDA IPC / device pointer | **COMPLETE — DIFFERENCE FOUND (D19-1, leak)** |
| 20 | CLIO GPU residency | **COMPLETE — MEASURED** (device path works; default test is host-resident) |
| 21 | CLIO host/device transfer audit | **PARTIAL — counts measured, per-callsite attribution NOT done** |
| 22 | CPU fallback verification | **COMPLETE — confirmed taken on the default test** |
| 23 | End-to-end tests | **COMPLETE — 256/256 exact, 0/1 GiB bytes differ** |
| 24 | Automated equivalence suite | **COMPLETE — 5 coverage gaps recorded** |
| 25 | Performance comparison | **COMPLETE — measured directly against a from-source NeuroPress** |
| 26 | Discrepancy investigation | **COMPLETE — 4 items carried forward with named experiments** |
| 27 | Final equivalence matrix | **COMPLETE** |
| 28 | Final verdict | **COMPLETE** |
| 29 | Callback-trace equivalence harness (GPU-resident chunk) | **COMPLETE — 8/8 chunks PASS; 1 new finding (D29-1)** |

---

## Prior work this investigation inherits (NOT re-derived, NOT trusted without evidence)

`docs/neuropress/` in this same directory holds a **completed GPU/host-locality audit**
(68 findings, adversarially verified — see its `SUMMARY.md`). Scope was narrower than
this investigation: it asked only *"where does native run on GPU while CLIO runs on
host"*, not *"is CLIO byte-identical to native"*.

Two reasons its conclusions must be re-verified rather than imported (Protocol §12):

1. **The port has changed since.** Commits `283ed09c`, `a5e11062`, `fa485166` on the
   current branch explicitly move selection and decomp-head SGD onto the GPU — directly
   contradicting audit findings NN-1, NN-2 and ORCH-1 as filed. Confirmed structurally in
   Phase 1: `DecompHeadSGDKernel` now exists (`neuropress_nn_gpu_kernels.cu:662`); the
   audit stated "no kernel exists".
2. **Its own line citations are stale** on the port side (its README says 27–100 lines).

Carried forward as **hypotheses to test**, not facts:
- cuSZ error-bound mode mismatch (port relative vs native absolute) → test in Phase 16.
- `IpcManager::FreeGpuBackend` never calls `cudaFree` → test in Phase 21.
- HDF5 `H5Dread` decompresses nothing after reopen → test in Phase 17.
- `DequantizeDevice` has no pointer-kind guard → test in Phase 4.

---

## Native ground truth established

### Phase 5 — Statistics (§5 ground-truth record)

```
FUNCTIONALITY:      The three NN selection features — byte-level Shannon
                    entropy, mean absolute deviation, mean |2nd derivative|

NATIVE ENTRY POINT: runStatsKernelsNoSync (stats_kernel.cu:~369)
                    runStatsOnlyPipeline (:~306, host-readback variant)

NATIVE CALL GRAPH:  runStatsKernelsNoSync
                      → statsPass1Kernel        (stats_kernel.cu:106)
                      → launchEntropyKernelsAsync (entropy_kernel.cu:216)
                          → histogramKernel / histogramKernelVec4 (:48 / :99)
                          → entropyFromHistogramKernel (:167)
                      → madPass2Kernel          (stats_kernel.cu:201)
                      → finalizeStatsOnlyKernel (stats_kernel.cu:260)

STATE STRUCT:       AutoStatsGPU (src/stats/auto_stats_gpu.h), __align__(8),
                    DEVICE-resident. Fields: sum, abs_diff_sum, vmin, vmax,
                    num_elements, mad_sum, entropy, mad_normalized,
                    deriv_normalized, state, action, error_level.
                    entropy / mad_normalized / deriv_normalized are laid out
                    contiguously on purpose so the host can fetch all three in
                    ONE 24-byte D→H copy (stats_kernel.cu:441-446).

WORKSPACE LAYOUT:   [AutoStatsGPU | histogram(256 uint) | init_flag(int)]
                    (stats_kernel.cu:32) — one allocation, not three.

ALGORITHM:
  pass 1 (statsPass1Kernel, grid-stride, STATS_BLOCK_SIZE threads):
      sum          += x[i]
      abs_diff_sum += |x[i+1] - 2*x[i] + x[i-1]|   for 0 < i < n-1
      vmin/vmax    running min/max
      warp-shuffle reduce → shared inter-warp reduce → atomicAdd to global
  entropy: 256-bin BYTE histogram (integer atomics), then
      H = -sum(p*log2(p)), p = count/total_count, over bins with count > 0
      single block, ENTROPY_BLOCK_SIZE threads, shared-memory tree reduction
  pass 2 (madPass2Kernel):
      mean     = stats->sum / n          (read from device, not recomputed)
      mad_sum += |x[i] - mean|           same reduction shape, atomicAdd
  finalize (finalizeStatsOnlyKernel):
      mad_normalized   = mad_sum / n
      deriv_normalized = abs_diff_sum / (n - 2)     ← divisor is n-2, not n
      state = -1, action = -1

DETERMINISM:        **NOT bit-reproducible, in native itself.** sum,
                    abs_diff_sum and mad_sum are accumulated with
                    atomicAdd(double*, double) (stats_kernel.cu:185-186, :246),
                    whose summation order varies with block scheduling. Two
                    runs of NATIVE on identical input can differ in the low
                    bits. Byte-exact equality is therefore the WRONG parity
                    criterion for statistics — a relative tolerance at
                    double-precision epsilon is the correct one. (The byte
                    histogram itself IS exact: integer atomics, and addition
                    of 1s is order-independent.)
```

### Phase 4 — Quantization / dequantization (§5 ground-truth record)

```
FUNCTIONALITY:      Linear quantization float32 → int8/16/32, and its inverse

NATIVE ENTRY POINT: quantize_simple (3 overloads: :368, :559, :689)
                    dequantize_simple (:798)
                    src/preprocessing/quantization_kernels.cu
                    The compress path uses the P1 overload at :689 (takes
                    pre-allocated range/output/CUB buffers from CompContext).

NATIVE CALL GRAPH:  quantize_simple
                      → compute_data_range (:284/:306)
                          → compute_data_range_typed (:187/:237)
                              → CUB DeviceReduce::Min / ::Max
                      → compute_required_precision (quantization.cuh:192)
                      → launch_quantize_kernel (:330)
                          → quantize_linear_kernel<InputT,OutputT> (:55)

INPUT:              device float32 (element_size 4) or float64 (8) buffer
OUTPUT:             device int8_t / int16_t / int32_t buffer,
                    num_elements * precision_to_bytes(precision)
MEMORY LOCATION:    DEVICE throughout
CUDA KERNEL:        quantize_linear_kernel / dequantize_linear_kernel,
                    grid-stride, all arithmetic in DOUBLE
CUDA STREAM:        caller-supplied
H→D / D→H:          only the CUB min/max scalars come back to the host

EFFECTIVE ERROR BOUND — the subtle part (:711-741), reproduced exactly:
    data_range = data_max - data_min
    if (data_range <= 0.0) data_range = 1.0          ← constant-data clamp
    max_abs_value      = fmax(|data_min|, |data_max|)
    float_repr_error   = max_abs_value * 2.4e-7
    safety_margin      = error_bound * 0.05
    available_for_quant= error_bound - float_repr_error - safety_margin
    min_eb_for_int32   = data_range / 4.0e9
    effective_eb = (available_for_quant <= 0)
                     ? fmax(min_eb_for_int32, float_repr_error * 0.1)
                     : available_for_quant
    effective_eb = fmax(effective_eb, min_eb_for_int32)
    scale        = 1.0 / (2.0 * effective_eb)

PRECISION SELECTION (quantization.cuh:192-206), AUTO only:
    num_bins = (data_range / (2.0 * effective_eb)) * 1.1
    num_bins <= 127   → 8 bits
    num_bins <= 32767 → 16 bits
    else              → 32 bits

ALGORITHM:
  quantize:   q = round((v - data_min) * scale), then clamp to the output
              width: int8 [-128,127], int16 [-32768,32767],
              int32 [-2147483648, 2147483647]   (:71-79)
  dequantize: v = q * inv_scale + data_min, inv_scale = 2 * effective_eb
              (:94-98)

METADATA:     header.quant_error_bound = **effective_eb**, NOT the user's
              requested bound; header.quant_scale = scale; data_min/data_max
              as measured. Dequantize reconstructs QuantizationResult purely
              from these header fields, so the round trip is self-consistent
              even though the recorded bound differs from the requested one.
DETERMINISM:  deterministic. round() is a single well-defined op; min/max are
              exact and order-independent, so CUB's tree reduction and any
              other reduction order give bit-identical results.
```

### Phase 3 — Shuffle / deshuffle (§5 ground-truth record)

```
FUNCTIONALITY:      Byte shuffle (AoS → byte planes) and its inverse

NATIVE ENTRY POINT: byte_shuffle_simple / byte_unshuffle_simple
                    src/preprocessing/byte_shuffle_kernels.cu:285,332,379
                    declared src/preprocessing/byte_shuffle.cuh:39,53,73

NATIVE CALL GRAPH:  byte_shuffle_simple
                      → createDeviceChunkArrays (:162)
                          → populateChunkArraysKernel (:140)  [3 cudaMalloc]
                      → launch_byte_shuffle (:225)
                          → byte_shuffle_kernel_specialized<N> (:21)
                      → cudaStreamSynchronize   [ALWAYS, all 3 overloads]
                      → ~DeviceChunkArrays      [3 cudaFree]

INPUT:              device uint8_t buffer, total_bytes
OUTPUT:             device uint8_t buffer, same total_bytes (size-preserving)
DATATYPE:           raw bytes; element_size is a reinterpretation only
SHAPE:              1-D
CHUNKING:           SHUFFLE_CHUNK_SIZE = 256*1024 = 262144 B
                    num_chunks = ceil(total_bytes / chunk_bytes)
                    Chunk k covers [k*262144, +size) in BOTH input and output —
                    planes are built WITHIN a chunk, never across. This is the
                    single fact the blob format depends on.
MEMORY LOCATION:    DEVICE throughout; no host touch of payload
CUDA KERNEL:        byte_shuffle_kernel_specialized<ElementSize>
                    instantiated for 1, 2, 4, 8 (:119-134)
LAUNCH GEOMETRY:    num_blocks = num_chunks (one block per chunk, no cap),
                    THREADS_PER_BLOCK = 256 (:236-237)
CUDA STREAM:        caller-supplied (ctx->stream from the compress path)
SYNCHRONIZATION:    cudaStreamSynchronize inside every overload
                    (:322, :367, :421) — native's shuffle IS a syncing call.
                    createDeviceChunkArrays deliberately does NOT sync (:212-214).
H→D TRANSFERS:      none
D→H TRANSFERS:      none

ALGORITHM (per chunk, ElementSize = E):
    n        = chunk_size / E
    leftover = chunk_size % E
    out[b*n + e] = in[e*E + b]      for b in [0,E), e in [0,n)
    out[E*n + i] = in[E*n + i]      for i in [0,leftover)   (thread 0 only)
  Special cases: E <= 1 → verbatim copy; n <= 1 → verbatim copy (:36-45)
  Unshuffle is the exact transpose: out[e*E + b] = in[b*n + e] (:101-108)

CONFIGURATION:      element_size switch (:239-248) handles 1, 2, 8 explicitly;
                    **default → 4**. So 0,3,5,6,7,9+ silently become E=4.
                    Production only ever passes 4 (decodeAction yields 0 or 4).
METADATA:           header.shuffle_element_size (offset 8); 0 = not shuffled
DIAGNOSTICS:        none of its own; folded into diag_preprocessing_ms
DETERMINISM:        fully deterministic — pure permutation, no reduction,
                    no floating point, no atomics
```

### Phase 2 — native GPU compress/decompress data & memory flow

All citations `/home/cc/NeuroPress` @ `b23b8f6`, verified by reading.

**Entry dispatch — `gpucompress_compress_gpu` (`src/api/gpucompress_compress.cpp:154-233`)**

Three mutually exclusive paths, decided on `cfg.algorithm`:

1. `CUSZ | NDZIP | CUSZP` → `compressExternalGpu` (:61-152). Bypasses the action
   machinery entirely; these streams "never carry GPUCompress shuffle/quantization
   preprocessing" (:1207-1209).
2. Any explicit algorithm ≠ `AUTO` → action synthesized on the host, no NN:
   `action = (cfg.algorithm - 1) % 8`, `+= 8` if `PREPROC_QUANTIZE`, `+= 16` if
   shuffle size > 0 (:178-187).
3. `AUTO` → `gpucompress_infer_gpu` then `gpucompress_compress_with_action_gpu`.

**Action encoding (`decodeAction`, `src/api/internal.hpp:156-172`) — exact:**
```
algorithm       = action_id % 8            /* 0=lz4 1=snappy 2=deflate 3=gdeflate
                                              4=zstd 5=ans 6=cascaded 7=bitcomp */
use_quantization= ((action_id / 8) % 2) != 0
shuffle_size    = ((action_id / 16) % 2) ? 4 : 0    /* only ever 0 or 4 */
```
32 actions total. `algo_to_use = decoded.algorithm + 1` (:416) — the public enum is
1-based.

**Constants (`src/api/internal.hpp:97-103`):** `DEFAULT_CHUNK_SIZE = 1<<16` (65536),
`SHUFFLE_CHUNK_SIZE = 256*1024` (262144), `GPU_ALIGNMENT = 4096`.

**Inference phase — `gpucompress_infer_gpu` (:238-333)**
- `runStatsKernelsNoSync(d_input, input_size, stream, ctx)` → `AutoStatsGPU*`
  **device pointer** (:280).
- Comment at :282 is explicit: *"Stats remain on GPU — NN inference reads d_stats_ptr
  directly on device."* **There is no D→H of statistics on this path.**
- `runNNFusedInferenceCtx(d_stats_ptr, input_size, cfg.error_bound, ...)` (:297-303)
  returns the action. Does one internal `cudaStreamSynchronize` (:305).
- **Observable ground truth for parity:** on success native sets
  `stats->entropy_bits = 0.0`, `stats->mad = 0.0`, `stats->second_derivative = 0.0`
  (:323-325) — it deliberately does *not* report them, because they were never brought
  back to the host. Only the four predicted values are filled (:326-329). A port that
  returns real numbers in these three fields is observably divergent.
- Two `CompContext` slots are held simultaneously on the AUTO path (:189-196 comment);
  pool exhaustion above ~4 concurrent callers is a documented native limitation.

**Compress phase — `gpucompress_compress_with_action_gpu` (:339-1173)**

Pipeline order, verified by reading, is **quantize → shuffle → codec → header**:

| Step | Lines | Notes |
|---|---|---|
| Acquire ctx + stream join | 368-378 | if caller passed a stream, `cudaEventRecord(t_start, caller_stream)` + `cudaStreamWaitEvent(ctx->stream, …)` — cross-stream join, no sync |
| Decode action | 415-420 | sets `preproc_to_use` |
| **Quantize** | 434-455 | only if `PREPROC_QUANTIZE` **and** `cfg.error_bound > 0.0`; `quantize_simple(...)` into `ctx->d_preproc_quant`; failure → `ERROR_COMPRESSION` |
| **Shuffle** | 457-470 | `byte_shuffle_simple(..., shuffle_size, SHUFFLE_CHUNK_SIZE, stream, ctx->d_preproc_shuffle, cap, &owns_shuffled)`; operates on the **quantized** buffer when both are on |
| Preproc timing sync | 474-478 | `cudaStreamSynchronize` only if something was preprocessed |
| Codec manager (LRU, depth 3) | 481-489 | `getOrCreateCompManager(ctx, internal_algo)` |
| Output sizing | 492-526 | if `header + max_compressed > *output_size`, `cudaMalloc` a temp; else compress straight into `d_output + 64` |
| Compress | 533-545 | `compressor->compress(d_compress_input, d_comp_target, comp_config)` |
| Sync | 558-560 | unconditional `cudaStreamSynchronize` — required because nvcomp's `get_compressed_output_size` memcpys on a different stream (comment :552-557) |
| Header build + write | 574-613 | see below |
| Free owned preproc buffers | 616-617 | only when `owns_*` (i.e. not the pre-allocated ctx buffers) |
| Exploration + SGD | 624-1173 | AUTO + `g_online_learning_enabled` only |

**Blob header — 64 bytes, `GPUCOMPRESS_HEADER_SIZE` (`include/gpucompress.h:34`),
`struct CompressionHeader` (`src/compression/compression_header.h:53`):**

| Offset | Type | Field |
|---|---|---|
| 0 | `uint32` | `magic` = `0x43555047` ("GPUC" LE) |
| 4 | `uint32` | `version` = 2 (`COMPRESSION_HEADER_VERSION`; `isValid` accepts 1..2) |
| 8 | `uint32` | `shuffle_element_size` (0 = none) |
| 12 | `uint32` | `quant_flags` — bits[0-3] type, [4-7] precision code, [8] enabled, [9-12] algorithm id |
| 16 | `uint64` | `original_size` |
| 24 | `uint64` | `compressed_size` |
| 32 | `double` | `quant_error_bound` |
| 40 | `double` | `quant_scale` = 1/(2·error_bound) |
| 48 | `double` | `data_min` |
| 56 | `double` | `data_max` |

Precision code mapping (`compression_header.h:87`): `8→1, 16→2, else→3`.
Algorithm id is **packed into `quant_flags` bits 9-12**, not a separate field — and
`setAlgorithmId` is called *after* `setQuantizationFlags` (:582-596), with
`setQuantizationFlags` preserving bits 9-12 (:84). Ordering matters for byte parity.

**Decompress — `gpucompress_decompress_gpu` (:1175-1312)**

Reverse order is **codec → unshuffle → dequantize → D2D to caller**:
- Header read D→H, 64 B, via `readHeaderFromDevice` (:1188-1190) — comment labels it
  "64B D→H". This is native's *only* mandatory host round-trip on the read path.
- External codecs dispatch on `header.getAlgorithmId()` and write **straight into
  `d_output`** with no preprocessing reversal (:1210-1231).
- nvcomp path `cudaMalloc`s `d_decompressed` at `decomp_config.decomp_data_size`
  (:1241-1243).
- Unshuffle (:1256-1269) allocates its own output via `byte_unshuffle_simple` — note
  this overload takes **no** pre-allocated buffer, unlike the compress side.
- Dequantize (:1272-1296) reconstructs `QuantizationResult` **purely from header
  fields** — scale, min, max, error bound, type, precision. No side-channel state.
- **Final `cudaMemcpyAsync(d_output, d_result, original_size, D2D)` + sync
  (:1299-1302).** This is the copy that audit finding PRE-10 claimed the port added;
  reading confirms native performs it too, so the review that struck PRE-10 is
  correct at `b23b8f6`.

### Phase 1 — source inventory only, no behavioral ground truth.

## CLIO implementation located

See `NEUROPRESS_CLIO_TRACEABILITY.md` for the functionality→file map.

## Tests completed

| Test | Result | Phase |
|---|---|---|
| `ctest -R ctp_neuropress` (6 tests) baseline | **PASS 6/6**, 5.20 s wall (parity 0.47 / dataset 0.84 / flow 0.43 / tiebreak 0.60 / device_path 0.38 / preprocess 2.19 s) | 1 |

| `ctp_neuropress_preprocess_parity` re-run for Phase 3 evidence | **PASS — 1014 checks, 0 failures** | 3, 4 |
| `ctp_neuropress_dataset_parity` — 1 GiB, 256×4 MiB chunks, 5 regimes | **PASS — 7940 checks, 0 failures.** Worst rel. dev: entropy 1.5e-15, mad 5.41e-15, deriv 2.38e-16. Selection: 0/256 divergent across 4 error bounds | 5, 7 |
| `ctp_neuropress_tiebreak_parity` — real `.nnwt` on both sides | **PASS — 1530 checks, 0 failures.** 32 real ties over 760 cases, 5 order-discriminating, 0 changed the pick | 9, 14 (partial) |
| `ctp_neuropress_parity` — inference + online SGD + decomp-head SGD | **PASS — 946 checks, 0 failures.** predictions max rel err **= 0**; weights max abs err 2.98e-08 | 10, 11, 13, 14 |
| `ctp_neuropress_flow_parity` — 2 interleaved flows, reload, replay | **PASS — 591 checks, 0 failures.** weights max abs err 7.45e-09; 1-flow vs 2-flow diverge by 0.00198627 identically on both sides | 13 |
| `ctp_neuropress_device_path_parity` — device stats, tie rule, in-kernel masking, no host fallback | **PASS — 251 checks, 0 failures.** 5/5 ties to lower action | 9, 11 |
| `md5sum` of `model.nnwt` in both trees | **IDENTICAL** — `df52a926af026fc617e172dcc990a395` | 14 |
| `nsys` on `compressor_dynamic_neuropress` (shipped functional test) | **1 CLIO kernel** (`InferKernel`) of 19 total → chunks are host-resident | 20, 22 |
| `nsys` on `neuropress_gpu_direct` (device example) | **8 CLIO device kernels** — full device pipeline launches | 20 |
| `nsys cuda_api_sum` on the device run | **379 `cudaMalloc` vs 73 `cudaFree`** → D19-1 leak | 19, 21 |
| `neuropress_e2e`, learning off, 1 GiB / 256 chunks | **256/256 exact, 0 of 1,073,741,824 bytes differ**; 217 traced decompressions + 39 raw-stored chunks reconciled | 23 |
| `cte_neuropress_callback_trace_equivalence` — 8 GPU-resident chunks, native and CLIO traced callback by callback, CUPTI-instrumented | **PASS 8/8, stable over 3 runs.** Statistics rel err 0; NN predictions rel err **0** over all 32 actions on every chunk; selections 8/8 identical; shuffle output byte-identical; quantized output byte-identical; codec payload byte-identical on 7/8. Selection also 8/8 identical to the real `AsyncDynamicSchedule` chimod path | 29 |

Baseline established **before** any change by this investigation, so a later failure is
attributable. Note this proves *numerical* agreement on the paths those tests cover and
nothing about execution locality (Protocol §7 — this is `PASS`, not `PASS — NUMERICALLY
EQUIVALENT` for the system as a whole).

## Tests pending

Everything in phases 2–28.

## Known differences

### Confirmed, ranked by severity

| ID | Phase | Severity | Claim |
|---|---|---|---|
| **D7-1** | 7 | **CORRECTED 2026-08-13 — WAS OVERSTATED** | ~~Host-resident chunk + NeuroPress configured → CLIO silently selects with its own legacy qtable/dense-NN model.~~ **This is wrong as a general claim.** Re-reading `compressor_runtime.cc:698-753`: a host-resident chunk with stats computed and the predictor ready passes the same gate as a device one, takes the `device_stats == nullptr` branch to `NeuroPressCandidateStats`, and that funnels into the SAME `RankIntoStats` → `predictor.Rank()` as the device variant (`neuropress_bridge.cc:381`/`:391` → `:337`). **NeuroPress ranks host-resident chunks.** The Phase 20 profile corroborates it — `InferKernel` launched 8× on the host-resident functional test, which IS NeuroPress inference. What survives is much narrower: the legacy heuristics take over only when `!features_ok` (e.g. a sub-element chunk), the predictor is not ready, or PSNR filtering empties the candidate list — and on the host path that hand-off is **silent**, whereas the device path logs it (`:758-766`). Residual severity: **LOW**, and the honest fix is a log line, not a staging path. |
| **VOL-5** | 17/20 | **FIXED 2026-08-13** | ~~Cache-served hyperslab read `memcpy`s into a possibly-device buffer.~~ **REPRODUCED at HEAD**: `clio_vol.cc:2232` `std::memcpy(buf, sel.data(), sel_size)` writes to the CALLER's buffer with no device check, while all four siblings (`:2055`, `:2102`, `:2288`, `:2509`) use `DeviceAwareMemcpy`. A legal `H5Dwrite` → `H5Dflush` → hyperslab `H5Dread` into `cudaMalloc`'d memory segfaults (exit 139, measured). **FIX:** both placements now device-aware — the `H5S_ALL` branch uses `DeviceAwareMemcpy`; the scatter branch stages through host memory, seeding the staging buffer from the destination first so a sparse mem-space selection does not clobber elements it left alone. Verified: SIGSEGV → PASS.<br>**Root cause of the miss:** `test_hdf5_vol_gpu_ptr.cc` exists expressly for this bug class but its CMake entry passed a name filter, so ctest ran only the FIRST of its cases — the hyperslab-write case was compiled and never executed. All three cases are now registered separately. |
| **D19-1** | 19 | **FIXED 2026-08-13** | ~~Device scratch is never freed.~~ `IpcManager::FreeGpuBackend` (`ipc_manager.cc:3947-3956`) only unregisters and documents that the caller must free the base; the compressor has 8 alloc sites, 6 `FreeGpuBackend` sites and **zero** `GpuApi::Free` calls. Measured 379 `cudaMalloc` vs 73 `cudaFree` in one 64-chunk run. **Not a parity divergence** — native frees its scratch completely and symmetrically. A CLIO-introduced defect.<br>**FIX APPLIED:** `IpcManager` now records `{base, kind, in_process}` per `AllocatorId` at allocation and releases it in `FreeGpuBackend`. Idempotent (erase-under-lock, so a double release cannot become a double free). **Scoped by CUDA IPC safety:** only *in-process* registrations are freed — a remotely-registered backend is imported by the runtime via `cudaIpcOpenMemHandle`, and with no `DeregisterMemory`/`CloseIpcMemHandle` path in the tree, freeing it would be a cross-process use-after-free. **COMPLETED 2026-08-13:** that case no longer leaks either. A `DeregisterMemoryTask` admin method was added (id 36) -- task struct with `Copy`/`AggregateOut`, runtime handler, client-side send, and all 8 autogen dispatch sites. The handler calls `GpuApi::CloseIpcMemHandle` on the imported `device_ptr` for a `kDeviceMem` backend, then `UnregisterClientBackend`. `FreeGpuBackend` now sends it SYNCHRONOUSLY before freeing, because the ordering is a correctness requirement: freeing while the importer still holds the mapping is a cross-process use-after-free. If the round trip fails it leaks deliberately rather than freeing under a possibly-live import. Files touched: `clio_mod.yaml`, `autogen/admin_methods.h`, `admin_tasks.h`, `admin_runtime.h/.cc`, `autogen/admin_lib_exec.cc`, `ipc_manager.cc`. Measured: `cudaFree` 73 → **367** against 379 `cudaMalloc`. Tests: 27/27 neuropress+compressor. |
| **D20-1** | 20 | **HIGH** | The default shipped functional test `compressor_dynamic_neuropress` drives **host-resident** chunks: only `InferKernel` launches, and the host paths for statistics, shuffle and selection are taken. Measured by `nsys`. This is what makes **D7-1 reachable**. The device path itself is complete and correct — `neuropress_gpu_direct` launches all 8 CLIO device kernels — so this is a property of how callers supply memory, not a missing implementation. |
| **D16-1** | 16 | **FIXED 2026-08-13** | ~~Native cuSZ sets `rc.mode = Abs`; CLIO's `Cusz` defaulted to `Rel`.~~ Confirmed as code: `external_compressors.cu:118` vs `cusz.h:87`, with `MakeCusz` (`compress_factory.h:439`) the only construction site and it passed `eb` alone, so nothing ever overrode the default. Under `Rel` a caller's bound was silently reinterpreted as a fraction of the data range — on a [-100,100] signal an eb of 1e-3 permitted ~0.2 of error instead of 0.001.<br>**FIX:** the `Cusz` constructor now defaults to `Abs`, and the class documentation (which still read "relative by default") was corrected to match. Existing blobs are unaffected — each carries its own `psz_header` in the frame prefix and `Decompress` rebuilds the manager from it, exactly as upstream's `cuszDecompress` does, so a stream written under `Rel` still decodes as `Rel`.<br>**TEST TIGHTENED (closes gap G5):** `test_compress.cc` asserted `max_err < 1.0` against a nominal 1e-3 bound — a thousand times loose, with a comment explaining it as slack for relative mode. It therefore passed in either mode and was the reason the divergence survived. Now asserts `max_err <= 2 * 1e-3`.<br>**Regression proven both ways:** the tightened assertion FAILS under `Rel` (`test_compress.cc:595`) and PASSES under `Abs`. Full suite 27/27. |
| **D15-1** | 15 | **WONTFIX — decided 2026-08-13** | Blob containers are disjoint: native magic "GPUC" / fixed 64 B versus CLIO "CTEC" / 24 B core + optional 32 B `QuantHeaderExtension`, with the algorithm id packed into `quant_flags` bits 9-12 on one side and a `compress_lib_` wire id on the other. Neither can parse the other.<br>**Decision: keep CLIO's format.** The container is CLIO-wide, not NeuroPress-specific -- every CLIO codec writes it -- so adopting upstream's 64-byte header would mean either changing the format for all of them (invalidating every stored blob) or carrying two formats in one system. The only thing interchange would buy is native NeuroPress reading CLIO's tier blobs, and there is no such consumer: NeuroPress is being integrated INTO Clio, not run alongside it. Phase 18 confirmed the part that matters is portable anyway -- the codec PAYLOAD matches wherever the codec is deterministic (LZ4 8/8, cascaded-no-shuffle 39/39); only the framing differs.<br>**One real limitation recorded rather than fixed:** CLIO's `compressed_size_` is `uint32` where native's is `uint64`, so a single compressed chunk above 4 GiB cannot be represented. The field reuses existing struct padding and `IsValid()` rejects overflow, and chunks on this path are single-digit MB, so it is bounded and detected rather than silent. Revisit if chunk sizes ever grow. |
| **VOL-3** | 23/24 | **MECHANISM CORRECTED — the stamp IS persisted; a deliberate fail-closed guard withholds it** | My earlier note said the reopen "finds no stamp blob" as though persistence were missing. Reading the code: `clio_write_stamp` IS called on every clean close (`clio_vol.cc:1298`), and `clio_stamp_matches` compares it on open (`:1127`). What actually happens is `clio_stamp_ambiguous` (`:969`) **deliberately deletes the stamp** when the file's mtime is younger than one timestamp granule.<br>The reasoning is sound and worth preserving: the stamp's only signal for an in-place, same-size edit is mtime, so if mtime is within a granule, a write happening right now would land in the same granule and produce an identical stamp -- the next open would then conclude "unchanged" about a file that changed. The guard withholds the stamp so the next open **fails closed**. The code even records that this was the fix for a test that "looked flaky rather than broken".<br>**So VOL-3 is not a bug. It is a correctness-preserving guard whose cost is that write -> close -> immediately reopen -- the common pattern -- always trips it, leaving the compressed blobs unused on the read path.**<br>Closing it within Clio's architecture (per the D21-1 decision) needs an identity that does not depend on mtime granularity: a content digest (expensive on a 1 GiB file) or a writer-maintained generation counter. Re-stamping lazily on a later open does NOT work -- an in-granule external edit is invisible at any later time. **This is a product trade-off (coherence safety vs. read-path benefit), not a patch**, and is left open deliberately. |
| **D21-1** | 21 | **DECIDED 2026-08-13 — WONTFIX, Clio's architecture is preserved** | Native records compression IN-BAND as a real HDF5 filter (`H5Zregister` + `H5Pset_filter`, `H5Zgpucompress.c:68/:400/:443/:471`), so the DCPL carries it inside the file and HDF5 itself invokes decompression on reopen. CLIO records it out-of-band via a stamp blob + tag coherence, and has no filter.<br>**Decision: keep CLIO's out-of-band model.** Registering an H5Z filter would put compression metadata inside the HDF5 file and hand decode scheduling to HDF5, which displaces the tier/tag/blob model the rest of Clio is built on -- a different system, not a patch. The integration brings NeuroPress's logic INTO Clio; it does not adopt NeuroPress's storage architecture.<br>**Consequences accepted, and they are the honest cost:** VOL-1's full-dataset D2H exists because plain bytes must still reach the native VOL underneath, and the dataset is stored twice. Both stay. |
| **D18-2** | 26 | **ROOT CAUSE OF D18-1 — nvcomp ANS output is HISTORY-DEPENDENT** | Established by experiment, not inference. Three runs of the native replay over the same 1 GiB input:<br>**(1) Determinism:** native vs native, same order → **256/256 identical**. So neither side is randomly nondeterministic.<br>**(2) Stale-output hypothesis REFUTED:** `--zero-output` (clear the output buffer before every compress) changed **0 of 256** hashes. The differing bytes are a real encoding, not uninitialised allocation slack.<br>**(3) Manager reuse is the cause:** `--flush-cache` (evict the cached nvcomp managers before every chunk, so each compress starts from a fresh manager) changed **191 of 256** hashes — and the split is almost perfectly by algorithm: `nvcomp-ans` **189 of 191 changed**, `nvcomp-cascaded` 2 of 57, `nvcomp-lz4` 0 of 8.<br>**Conclusion: an nvcomp ANS manager carries state across `compress()` calls that changes the encoding.** Native and CLIO reuse managers on different schedules — native's LRU is per `CompContext` (9 slots), CLIO's is per worker thread — so the two see different manager histories and emit different ANS bytes for identical input.<br>**Consequences for the integration:** (a) byte-identical ANS payloads are **not achievable** while manager lifecycles differ, and matching them would require pinning the entire manager reuse schedule, not a config change; (b) native's own ANS output is **not a stable target** — it changes when its cache is flushed, which `gpucompress_flush_manager_cache()` does on demand; (c) this is a property of nvcomp, not a defect either side introduced. D18-1 should therefore be graded as **EXPECTED-DIVERGENT for ANS**, with LZ4 and cascaded-without-shuffle being the genuinely comparable cases — and those already match (8/8 and 39/39). |
| **D18-1** | 18 | **HIGH — payloads are NOT byte-identical; the standing hypothesis is REFUTED** | Phase 18 finally ran. NeuroPress was built from source (nvcomp include + link path had to be supplied), `neuropress_native_replay.cc` — which had **no build target at all** — was compiled against it, and both sides were driven over the same 1 GiB / 256-chunk dump with **learning off** in Clio's completion order.<br>**Selections: 256/256 IDENTICAL** (algo, quantize, shuffle). This is the controlled measurement that retrospectively justifies discarding the 89/235 figure from the earlier uncontrolled runs.<br>**Payloads: 209 of 256 DIFFER.** The driver hashes the codec payload only, skipping native's 64-byte header (`neuropress_native_replay.cc:314-330`), and native's recorded `compressed_size` is CLIO's + exactly 64 for **all** 256 chunks — so the payload SIZES are identical and only the BYTES differ. Same algorithm, same input, same length, different output.<br>Breakdown by algorithm:<br>`nvcomp-ans` shuffle=0 → **differs**, 54<br>`nvcomp-ans` shuffle=1 → **differs**, 137<br>`nvcomp-cascaded` shuffle=0 → identical, 39<br>`nvcomp-cascaded` shuffle=1 → **differs**, 18<br>`nvcomp-lz4` shuffle=1 → identical, 8<br>**ANS diverges unconditionally** (191 chunks, with and without shuffle). Cascaded diverges only with shuffle, while LZ4 WITH shuffle is identical — so shuffle itself is not the cause, consistent with Phase 3 proving it byte-identical. Root cause not yet established; do not guess it. Phases 3/4/15 proved the codec's INPUTS and configuration match, so the divergence is inside the codec invocation.<br>**Consequence:** every earlier statement that identical payloads were "very likely" is now known to be wrong for ANS and for cascaded+shuffle. Cross-compatibility of the payload is not achieved. |
| **P25** | 25 | **MEASURED 2026-08-13 — CLIO is now FASTER on codec time; supersedes the inherited figures** | Phase 25 had been carrying commit `a7886973`'s numbers second-hand. Both blockers are gone: NeuroPress now builds from source here, and the D8-1 fix made CLIO's codec-kernel timing the default. Both sides are genuinely like-for-like -- native's `actual_comp_time_ms` comes from events bracketing only `compressor->compress(...)` (`gpucompress_compress.cpp:530-563`), CLIO's is `LastCodecKernelMs()` around `mgr->compress()`.<br>Same 1 GiB / 256 chunks, learning off, **selections agreeing 512/512** so the two are compressing the same things:<br>`clio    n=512  median 0.2732 ms  p10 0.2279  p90 3.5022`<br>`native  n=256  median 0.3553 ms  p10 0.2202  p90 1.3578`<br>**median clio/native = 0.769** -- reversing `a7886973`'s 1.105, which predated the D8-1 fix.<br>**Two honest caveats.** (1) CLIO's p90 is 2.6x native's (3.50 vs 1.36 ms): the medians favour CLIO but the tail does not, so "faster" is a statement about typical chunks, not worst case. (2) CLIO logs 512 samples to native's 256 because it **compresses every chunk twice** -- the double-store behaviour that follows from D21-1. Its total codec time is therefore 976.3 ms against native's 237.4 ms; per-chunk it is quicker, in aggregate it does substantially more work. A first pass at this comparison took `min` of the two CLIO samples per chunk and got 0.719, which flattered CLIO by discarding the slower half -- corrected to the full distribution here. |
| **D21-2** | 21/25 | **DECIDED 2026-08-13 — WONTFIX, Clio's chunking is preserved; ONE CONSEQUENCE NEEDS WATCHING** | Native compresses the N-dimensional HDF5 chunk gathered on the GPU (`gather_chunk_kernel`, arbitrary rank to 32, `H5VLgpucompress.cu:1142`); CLIO compresses a flat `file->chunk_size` byte range of the linearised dataset and uses `H5Pget_chunk` only for alignment classification and telemetry.<br>**Decision: keep Clio's flat chunking**, consistent with the D21-1 ruling -- the tier/blob model chunks by bytes, and adopting HDF5 chunk geometry would reshape the write path around HDF5's layout rather than Clio's.<br>**Accepted consequence, stated plainly:** for a multi-dimensional dataset the two systems feed the network DIFFERENT feature values for the same file, because an N-D sub-block is spatially local while a flat byte range spans row boundaries. Selection can therefore differ in production even though it agrees on every parity test -- those feed both sides the same bytes and so can never surface this.<br>**WORTH WATCHING, and newly established here:** the shipped `.nnwt` appears to have been trained on N-D chunk features. Native's training data comes from the TRACE-mode campaign, and `run_trace_exhaustive_chunk` (`:1528`) is invoked AFTER `gather_chunk_kernel` (`:1490`) in the same chunk loop -- so each training row describes a gathered N-D chunk. Feeding that model flat byte ranges is therefore a mild DISTRIBUTION MISMATCH, not merely a different windowing. It does not make selections wrong (Phase 5 showed the features themselves are computed correctly, and the e2e round-trips losslessly), but predicted ratio/time accuracy on multi-dimensional scientific data is unvalidated. **Recommended follow-up: measure prediction MAPE on a genuinely 2-D/3-D dataset before relying on the predictions there** -- the current 1 GiB e2e is 1-D, so it cannot show this either way. |
| **D23-2** | 23 | **FIXED 2026-08-13 — NOW VERIFIED** | ~~Exploration SGD trained on every candidate in enumeration order with no sort and no cap; native sorts by ascending cost and keeps at most the cheapest 7.~~<br>**FIX:** a parallel `explore_costs` vector now records each candidate's cost, and immediately before `Train()` the batch is `stable_sort`ed by ascending cost and truncated to `kMaxExploreSgdSamples = 7`, mirroring `gpucompress_compress.cpp:1004-1020` (`std::sort(explored_samples.begin()+1, ..., a.cost < b.cost)`, `emax = min(size, NN_MAX_SGD_SAMPLES)`, loop from index 1). Index 0 upstream is the primary, which phase 1 already learned from; Clio's alternatives list is already primary-free, so the whole vector is eligible and the cap is 7 rather than 8. The cap is architectural upstream — `SGDSample explore_sgd[NN_MAX_SGD_SAMPLES]` is a fixed array and the kernel accepts no more.<br>**VERIFIED (gap G3 closed):** `compressor_exploration` now drives the path with `online_learning=true`, `exploration_enabled=true`, `threshold=0` so it fires every chunk. Debug output confirms the real path runs, not a skipped branch: `NeuroPress explore: adopted nvcomp-lz4 (ratio=184.608 time=0.7424ms quant=0 shuffle=4) over the primary` and `k=3 error_pct=0.829925 threshold=0 trained=1 regret=4.87978` -- so the K-way search, the winner-adoption path that RE-STORES the blob under a rewritten header, and `Train()` all execute, with byte-for-byte round-trip asserted per pattern. At the shared default K=3 neither the sort nor the cap binds; the change matters at K>7, and for batch ordering, which SGD accumulates over. Verifying it needs the exploration coverage in task 20. |
| **D23-1** | 23 | **DECIDED 2026-08-13 — WONTFIX; the claim "no knob" was WRONG** | Native bounds concurrent compressions to 9 `CompContext` slots with a blocking acquire, tunable by `GPUCOMPRESS_POOL_SLOTS` explicitly "to cut per-rank GPU memory" (`gpucompress_pool.cpp:32`). I filed this as Clio having no bound and no knob. **The bound exists** -- compression runs on worker threads, so concurrency is capped by `runtime.num_threads` (`context-runtime/config/clio_default.yaml:34`, default 4). Clio expresses the limit through its worker pool, which IS its concurrency architecture, rather than through a second compression-specific semaphore.<br>**Decision: keep the worker pool as the single concurrency control.** Adding a separate bound on the compression path would put two competing limiters in one system.<br>**Residual difference, small but real:** native's knob throttles compression ALONE, while `num_threads` throttles everything, so an operator cannot reduce GPU memory without also reducing CPU concurrency. Worth knowing for a memory-constrained multi-rank deployment; not worth a second limiter. |
| **VOL-1** | 21 | **CONFIRMED at HEAD — MEDIUM; consequence of D21-1** | **Native's default write path copies only the COMPRESSED bytes to the host; CLIO copies the FULL UNCOMPRESSED dataset, and still does the compressed path on top.** Native runs in `VOLMode::RELEASE` by default (`H5VLgpucompress.cu:64`), and in that mode its per-chunk D2H is `cudaMemcpy(hbuf, d_comp_w[w], comp_sz, D2H)` — the compressor OUTPUT buffer, into a pinned pool (`:2076-2085`). Only `VOLMode::BYPASS` copies the raw input (`wi.src`), and the code says why they share a slot: "so the D→H latency is directly comparable between modes". CLIO instead does `native_write_staging.resize(total_size); DeviceAwareMemcpy(native_write_staging.data(), buf[d], total_size)` (`clio_vol.cc:1732-1741`, again at `:1925-1927`) — the whole uncompressed transfer.<br>**CLIO's comment citing `gpu_bypass_dh_write` as precedent is accurate but compares against native's BYPASS mode, not its production RELEASE mode.** The extra traffic is the uncompressed size minus the compressed size, per write.<br>Root cause is D21-1: with no H5Z filter, CLIO must hand plain bytes to the native VOL underneath, so it needs them on the host. Native's filter means HDF5 receives compressed bytes through the filter pipeline and the uncompressed image never has to leave the device. |
| **VOL-2** | 21 | **FIXED 2026-08-13 (2nd attempt) — VERIFIED** | ~~The VOL read path always allocated its decompress destination with `CLIO_IPC->AllocateBuffer` (CPU shared memory), never GPU memory.~~<br>**FIX:** when `ctp::IsDevicePointer(dst)`, the compressor read path now allocates a per-chunk backend via `AllocateAndRegisterGpuBackend(..., **kManagedUvm**, ...)`, points `blob_data` at it and copies into `dst + offset`; CPU SHM remains the fallback. Brings the path in line with native, which decompresses straight into the caller's device buffer, and makes the compressor's device unshuffle/dequantize -- verified in Phase 17 -- reachable from the VOL for the first time.<br>**FIRST ATTEMPT USED `kDeviceMem` + CUDA IPC AND FAILED, TWICE OVER. Recorded because it constrains any future retry:**<br>(1) *Resolution.* `ToFullPtr` Case 5 only consults `FindClientBackend` inside `for (g = 0; g < GetGpuQueueCount(); ++g)`, and that count is `per_gpu_devices_.size()`. With no GPU queues on the runtime the loop never runs, resolution falls through to the CPU-SHM path and `shm_open`s `clio_<pid>_<n>`, which does not exist for a device allocation.<br>(2) *The consumer may be a CPU codec.* Backtrace: `ZstdWithModes::Decompress` writing into the destination. The VOL's compressor selects CPU codecs -- zstd for this dataset -- and host zstd handed a `cudaMalloc` pointer segfaults regardless of whether IPC resolution works.<br>`kManagedUvm` satisfies both: GPU-addressable AND host-addressable, and per `ipc_manager.h:1119` `off_` and `device_ptr` are the same value for it, so no portable IPC handle is needed. **True `kDeviceMem` on this path is only safe if VOL read-side codec selection is first restricted to GPU-native codecs -- a design change, not a swap.**<br>**VERIFIED:** the new `cte_hdf5_vol_compressor_devread` test discriminates in both directions -- SegFault under `kDeviceMem`, pass under `kManagedUvm`, pass without the change. Suite 41/41. |
| **D17-1** | 17/21 | **RESOLVED — real hazard, currently UNREACHABLE** | The question was whether `DequantizeDevice` can be handed a host destination. **By construction, yes:** on the VOL read path `output_fullptr = CLIO_IPC->ToFullPtr<char>(task->blob_data_)` (`compressor_runtime.cc:2865`) resolves the caller's buffer, and for the VOL that buffer is `AllocateBuffer` **CPU shared memory** (VOL-2). For a quantized blob `codec_dst` is the device quant scratch, so `DequantizeDevice(codec_dst, ..., output_fullptr.ptr_)` (`:2992`) would launch a kernel writing to a host address — `DequantizeDevice` guards only null/zero/scale.<br>**But it cannot happen today:** the VOL never sets `error_bound_`. A tree-wide search finds it set non-zero in exactly two places, neither on the VOL path — `neuropress_gpu_direct.cc:167` (device-resident, so the destination is device and the call is safe) and `test_compressor_interpose.cc:206` (interposer path, which uses the HOST `Dequantize<float>` in `DecompressStored`). With no error bound the VOL always compresses losslessly, `stored_quant` is never set, and the dequantize branch is never entered.<br>**Latent, not live.** It arms itself the moment anyone wires an error bound into the VOL — which is a natural thing to want, since lossy compression of scientific data is the entire point of the quantize and cuSZ actions. Fixing VOL-2 (device destination) would also disarm it. |
| **D22-1** | 22 | **DECIDED 2026-08-13 — WONTFIX; severity DOWNGRADED to LOW on inspection** | Native's `gpucompress_set_best_mode` has no CLIO counterpart. On checking what it actually does, most of it is **already expressible in Clio's own configuration**: `g_best_mode` forces exploration on every chunk regardless of `error_pct` (`gpucompress_compress.cpp:733`), and that is exactly what `neuropress_exploration_threshold_ = 0` achieves -- proven, not assumed: the `compressor_exploration` test added for gap G3 sets precisely that and the debug log shows exploration firing per chunk with `k=3 ... threshold=0`.<br>What has **no** Clio equivalent is the other half: `best_mode` simultaneously SUPPRESSES SGD phase 1 (`:705`) and phase 2 (`:1003`), i.e. "search exhaustively but do not learn from it". In Clio, exploration and learning are independent switches, which is arguably the cleaner decomposition -- it just cannot reproduce upstream's single combined flag.<br>**Decision: keep Clio's two orthogonal switches.** Originally filed MEDIUM on the belief that exhaustive per-chunk search was unexpressible; it is not. Corrected to LOW. |
| **D22-2** | 22 | **LOW — was D8-4, now sharper** | Native's cost weights and bandwidth are runtime-settable (`gpucompress_set_ranking_weights`, `gpucompress_set_bandwidth`, plus env vars at `gpucompress_api.cpp:233`); CLIO hardcodes them as `constexpr kCostW0/W1/W2 = 1.0` and `kCostBandwidthBytesPerMs = 5e6`. Values agree at the defaults so nothing diverges today, but a caller that retunes the cost model gets silently ignored. Also `gpucompress_set_reinforcement`'s 4th parameter `ct_mape_threshold` has no CLIO counterpart — though native ignores it too (`gpucompress_learning.cpp:199` takes it as an unnamed parameter), so that half is NOT a gap. |
| **D8-1** | 8 | **FIXED 2026-08-13** | ~~Cost computed from host wall clock vs native's CUDA events.~~ **Root of the problem was the instrument's cost, not its accuracy:** `KernelTimer` created and destroyed two CUDA events PER CALL, which is why the measurement had to be opt-in behind `CLIO_CODEC_KERNEL_TIMING`. Native pre-allocates `t_start`/`t_stop` once per `CompContext` and reuses them, so its always-on event timing is free.<br>**FIX:** the events moved into the existing per-thread `ManagerCache` (which already owns the persistent stream and the manager LRU), making a bracket two `cudaEventRecord`s. The env gate was then dropped and `ctp::LastCodecKernelMs()` wired into the cost model on BOTH the primary path (`actual_compress_time_ms_`) and each explored candidate (`alt_time_ms`), so the two are measured on the same clock as native does. Wall clock remains the fallback when the bracket cannot run. `CLIO_CODEC_KERNEL_TIMING` kept as a no-op for existing scripts.<br>**MEASURED** on the 1 GiB / 256-chunk e2e, learning off: median compress time **3.0210 ms -> 0.2732 ms, an 11x reduction**. The cost model had been fed times an order of magnitude above the real codec work, inflated by `ToDeviceInput` staging, a possible `cudaMalloc`, `configure_compression`, the stream sync and the output copy. Native's like-for-like median is 0.2847 ms (commit `a7886973`), so CLIO/native goes from ~10.6x to **0.96**.<br>**No regression:** selections 256/256 identical (joined on blob -- `seq` is async completion order and is NOT a valid join key), decompression 256/256 exact, 0 of 1,073,741,824 bytes differ, suite 27/27. |
| **D6-1** | 6 | **WONTFIX — decided 2026-08-13** | `gpucompress_chunk_diag_t` (~80 fields) is not ported and CLIO has no call site for any `gpucompress_*chunk_diag*` accessor.<br>**Decision: do not port the struct.** Reachability was established in Phase 6: native records a diagnostic on every compress, but every consumer of the accessor API in its tree is a TEST or BENCHMARK -- there is no production consumer. The one functional consumer is internal (the decomp-head SGD sweep) and CLIO already replicates that slice via `decomp_features_`, including the 1 ms floor and never-consume re-sweep.<br>**CLIO's equivalent observability already exists** and was used throughout this investigation: `CLIO_NEUROPRESS_SELECTION_LOG` emits per chunk `entropy, mad, second_deriv, wire_lib, lib_name, algo_idx, quantize, shuffle, preset, pred_ratio, pred_ct_ms, pred_dt_ms, pred_psnr, actual_ratio, actual_ct_ms, actual_psnr, checksum`. That carries BOTH predicted and actual ratio/comp-time/PSNR, so the MAPE figures native stores as fields are derivable from the log rather than absent. What is genuinely missing is `regret` and the per-chunk exploration ranking arrays; if those are ever wanted, adding two columns to the existing log is the cheap route, not reproducing an 80-field struct. |
| D3-3 | 3 | LOW | Port shuffle runs on an internal `DeviceStatsStream()`, native on the caller's `ctx->stream`. Both sync before returning; only achievable concurrency differs. **Cost unmeasured — no harm asserted.** |
| D3-1/D3-2 | 3 | NONE | Port rejects `elem_size` ∉ {2,4,8}; native accepts 1 and silently maps 3,5,6,7 → 4. Unreachable — `decodeAction` emits 0 or 4 only. |
| D4-1 | 4 | NONE | Port's `effective_eb <= 0` guard has no native counterpart. Dead code on both sides after the constant-data clamp. |
| D7-2 | 7 | NONE | `heuristic_select_action` not ported. **Not a gap** — dead upstream: no caller of `gpucompress_set_selection_mode` exists anywhere in NeuroPress. |
| D7-3 | 7 | NONE | No port counterpart to explicit-algorithm action synthesis; CLIO addresses codecs directly via `compress_lib_`. Same codecs reached. |

### Confirmed matches worth recording (not differences)

- **cuSZp MATCHES native exactly.** Native calls `cuSZp_compress(input, target, nbEle, &cmp_size, (float)error_bound, CUSZP_DIM_1D, dims, CUSZP_TYPE_FLOAT, CUSZP_MODE_OUTLIER, stream)` (`external_compressors.cu:434-437`); CLIO calls the same function with the same `CUSZP_DIM_1D` / `CUSZP_TYPE_FLOAT` and `mode_` defaulting to `CUSZP_MODE_OUTLIER` (`cuszp.h:97`, `:129-130`). Same dimensionality, type, encoding mode and absolute-bound semantics.
- **nDzip MATCHES.** Both treat the buffer as a 1-D float32 array (`ndzip::extent ext(1)`), both reject a size that is not a whole multiple of `sizeof(float)`, both are lossless (`external_compressors.cu:272-283` vs `ndzip.h:85-89`).
- **These two make D16-1 sharper, not weaker:** two of the three external codecs were ported with their error-bound semantics intact, so cuSZ defaulting to `Rel` where native sets `Abs` reads as an isolated oversight rather than a deliberate policy.

- **The decomp-SGD 1 ms floor citation is accurate.** CLIO's comment (`compressor_runtime.cc:1047-1054`) cites `diagnostics_store.hpp:96-102` for storing `std::max(1.0f, ms)` as the learning target while keeping the raw value for diagnostics. Read directly: `recordDecompMs` does `float clamped = std::max(1.0f, ms); h.decompression_ms = clamped; h.decompression_ms_raw = ms;` (`:97-102`). Exact. This was one of the port's many upstream-citing comments and it holds.

- **The nvcomp manager LRU cache MATCHES.** Native keys an LRU of `LRU_DEPTH = 3` managers by algorithm per `CompContext`, on that context's stream, with tick/clock eviction and hit/miss counters (`gpucompress_pool.cpp:236-271`, `internal.hpp:83-86`). CLIO reproduces it exactly — `kLruDepth = 3` annotated `// CompContext::LRU_DEPTH`, same slot/tick/clock/victim structure and hit/miss counters (`nvcomp.h:358-448`) — scoped to the worker THREAD rather than a context, with the reason given: Clio has no per-context object at that layer, a thread gives each concurrent flow its own managers and stream, and an nvcomp manager is bound to the stream it was built on so sharing across threads would be unsafe. A deliberate, documented equivalent. **I nearly filed this as a divergence off `GetPreset` returning a fresh `unique_ptr` per call — the cache sits a layer below that.**

- **Native's BYPASS and TRACE VOL modes are benchmarking harnesses, not production behaviour.** `detect_vol_mode()` (`H5VLgpucompress.cu`) returns `VOLMode::RELEASE` unless `GPUCOMPRESS_VOL_MODE` is set, and the only setters in the whole native tree are `benchmarks/trace_campaign/run_nyx_trace.sh`, `benchmarks/trace_campaign/replay_trace.cu` and `tests/hdf5/test_vol_modes.cu`. TRACE runs `run_trace_exhaustive_chunk` (`:2641`) — an exhaustive sweep over an error-bound ladder (lossless, 1e-6, 1e-3, 1e-2, overridable via `GPUCOMPRESS_TRACE_EBS`) across nvCOMP+quant, cuSZ and the NN selection — to generate training/benchmark data. CLIO having no equivalent is an observability gap of the same kind as D6-1, **not** a correctness divergence. Recording it so nobody files it as one later.

- **`gpucompress_enable/disable/active_learning_enabled` are DEAD UPSTREAM.** The three entry points exist (`gpucompress_learning.cpp:172-183`) but `g_active_learning` has **no consumer anywhere in native's `src/`**. CLIO not porting them is not a gap. This is the **fifth** instance of the pattern, after `vmin`/`vmax`, `heuristic_select_action`, `x_mins`/`x_maxs` and `prev_grad_dot` — worth remembering before filing any future "the port is missing X".
- **CLIO deliberately does not link native NeuroPress**, and that is the intended design: NeuroPress's logic is being integrated INTO Clio, not wrapped. Verified: no `gpucompress` link target anywhere outside `test/`/`parity/`, and all 12 apparent production references are comments citing upstream line numbers. The equivalence standard is therefore behavioral reproduction, which is what every phase measured — not API delegation.

- Device-path GPU-selection failure fails the write in CLIO (`return_code_ = 4`,
  `compressor_runtime.cc:1324-1338`), matching native's error propagation. Commit
  `fa485166`'s claim verified by tracing the flag producer→consumer.
- CLIO's `ComputeCompressionFeatures` refuses to fall through to host routines on a
  device pointer (`data_stats_gpu.h:130-139`) rather than segfaulting or zeroing.

### Structural, from Phase 1 inventory only — no behavioral claim

- **Ranking is fused into inference natively, split out in CLIO.** Native computes
  `rank_val` inside a `__device__` helper called from `nnInferenceKernel`, then reduces
  in shared memory (`nn/nn_gpu.cu:109-337`). CLIO has a standalone
  `RankKernel` (`neuropress_nn_gpu_kernels.cu:549`). Behavioral consequence unknown →
  Phase 9.
- **Native kernels with no CLIO counterpart found in Phase 1:**
  `populateChunkArraysKernel`, `verify_error_bound_kernel`, `histogramKernel`/`Vec4`
  (CLIO may fold histogramming into `StatsPass1Kernel` — unverified),
  `gather_chunk_kernel` / `scatter_chunk_kernel` / `traceQualityKernel` (HDF5 VOL).
- **CLIO kernels with no native counterpart found in Phase 1:** `MinMaxKernel`,
  `StatsPass2DevKernel`, `InferKernelDeviceStats` (may correspond to native
  `nnFusedInferenceKernel` — unverified).

## Known questions

- Does CLIO's `StatsPass1Kernel` subsume native's `histogramKernel`, and if so does it
  use the vec4 variant's exact accumulation order? (Phase 5)
- Which of `InferKernel` / `InferKernelDeviceStats` runs on the shipped path, and which
  native kernel is its counterpart? (Phase 11)
- Native ranking is fused; CLIO's is a separate kernel — same tie-break order?
  `ctp_neuropress_tiebreak_parity` exists, suggesting this was already a problem. (Phase 9)
- Is `heuristic_select_action` (native `selection/heuristic.cu`) ported at all, and is it
  reachable in either tree? (Phase 7)

## Files investigated

**Native (`/home/cc/NeuroPress`), Phase 1 — enumerated, not yet read in depth:**

| Dir | Files | Lines |
|---|---|---|
| `src/api/` | `gpucompress_compress.cpp` (1312), `internal.hpp` (520), `gpucompress_api.cpp` (485), `diagnostics_store.hpp` (454), `gpucompress_diagnostics.cpp` (301), `gpucompress_pool.cpp` (273), `gpucompress_learning.cpp` (221), `gpucompress_state.hpp` (79), `cuda_check.h` (27) | 3672 |
| `src/compression/` | `external_compressors.cu` (522), `compression_header.h` (262), `compression_factory.cpp` (159), `external_compressors.hpp` (108), `util.h` (93), `compression_factory.hpp` (60) | 1204 |
| `src/nn/` | `nn_gpu.cu` (2645), `nn_weights.h` (137) | 2782 |
| `src/preprocessing/` | `quantization_kernels.cu` (917), `quantization.cuh` (434), `byte_shuffle_kernels.cu` (430), `byte_shuffle.cuh` (109) | 1890 |
| `src/selection/` | `heuristic.cu` (34), `heuristic.h` (24) | 58 |
| `src/stats/` | `stats_kernel.cu` (470), `entropy_kernel.cu` (253), `auto_stats_gpu.h` (38) | 761 |
| `src/hdf5/` | `H5VLgpucompress.cu` (4311), `H5Zgpucompress.c` (643), `H5Zgpucompress.h` (62) | 5016 |
| `src/cli/` | `compress.cpp` (809), `decompress.cpp` (500) | 1309 |
| `include/` | `gpucompress.h` (819) + 7 app-specific headers | — |

Read in full so far: `src/selection/heuristic.cu`.
Read in part: `include/gpucompress.h` (API signature list),
`src/preprocessing/byte_shuffle_kernels.cu:1-60`.

**Port (`/home/cc/clio-core`), Phase 1 — located, not read:**

| Component | Path | Lines |
|---|---|---|
| NN predictor | `context-transport-primitives/src/compress/model/neuropress_nn_predictor.cc` | 678 |
| NN GPU kernels | `context-transport-primitives/src/compress/model/neuropress_nn_gpu_kernels.cu` | 1445 |
| NN headers | `.../include/clio_ctp/compress/model/neuropress_nn_predictor.h` (428), `.../neuropress_nn_gpu_kernels.h` (221) | 649 |
| Stats/preprocess kernels | `context-transport-primitives/src/compress/preprocess/data_stats_gpu_kernels.cu` | 881 |
| Stats header | `.../include/clio_ctp/compress/preprocess/data_stats.h` | 856 |
| Quantization | `.../include/clio_ctp/compress/preprocess/quantization.h` | 323 |
| Byte shuffle | `.../include/clio_ctp/compress/preprocess/byte_shuffle.h` | 241 |
| Codec factory | `.../include/clio_ctp/compress/compress_factory.h` | 590 |
| CTE bridge | `context-transfer-engine/compressor/src/models/neuropress_bridge.cc` (405), `.../models/neuropress_bridge.h` (129) | 534 |
| CTE runtime | `context-transfer-engine/compressor/src/compressor_runtime.cc` | 3805 |
| Parity tests | `context-transport-primitives/test/unit/compress/model/parity/` (11 files) | — |

## Functions investigated

- Native `heuristic_select_action(double entropy)` — `src/selection/heuristic.cu:26-34`.
  Entropy thresholds 3.5 / 5.5 → action 4 (zstd) / 3 (gdeflate) / 0 (lz4). No shuffle,
  no quantization. Baseline comparator for the NN, not the production selector.

## CUDA kernels investigated

**Native inventory (fresh `grep` of `src/` + `include/`, Phase 1):**

| Kernel | File |
|---|---|
| `statsPass1Kernel`, `madPass2Kernel`, `finalizeStatsOnlyKernel` | `src/stats/stats_kernel.cu` |
| `histogramKernel`, `histogramKernelVec4`, `entropyFromHistogramKernel` | `src/stats/entropy_kernel.cu` |
| `byte_shuffle_kernel_specialized<1,2,4,8>`, `byte_unshuffle_kernel_specialized<1,2,4,8>`, `populateChunkArraysKernel` | `src/preprocessing/byte_shuffle_kernels.cu` |
| `quantize_linear_kernel`, `dequantize_linear_kernel`, `verify_error_bound_kernel` | `src/preprocessing/quantization_kernels.cu` |
| `nnInferenceKernel`, `nnFusedInferenceKernel`, `nnSGDKernel`, `nnBatchedDecompSGDKernel` | `src/nn/nn_gpu.cu` |
| `gather_chunk_kernel`, `scatter_chunk_kernel`, `traceQualityKernel` | `src/hdf5/H5VLgpucompress.cu` |
| `compare_buffers` (CLI verify only) | `src/cli/compress.cpp` |
| `gs_init_kernel`, `gs_step_kernel` (Gray-Scott app, out of scope) | `src/gray-scott/` |

**Port inventory (fresh `grep`, excluding `build/`, tests and unrelated adapters):**

| Kernel | File:line |
|---|---|
| `InferKernel` | `neuropress_nn_gpu_kernels.cu:415` |
| `InferKernelDeviceStats` | `neuropress_nn_gpu_kernels.cu:453` |
| `RankKernel` | `neuropress_nn_gpu_kernels.cu:549` |
| `DecompHeadSGDKernel` | `neuropress_nn_gpu_kernels.cu:662` |
| `SGDKernel` | `neuropress_nn_gpu_kernels.cu:1035` |
| `StatsPass1Kernel` | `data_stats_gpu_kernels.cu:32` |
| `StatsPass2Kernel` | `data_stats_gpu_kernels.cu:135` |
| `StatsPass2DevKernel` | `data_stats_gpu_kernels.cu:163` |
| `EntropyFromHistKernel` | `data_stats_gpu_kernels.cu:198` |
| `FinalizeFeatureStatsKernel` | `data_stats_gpu_kernels.cu:227` |
| `ShuffleKernel` | `data_stats_gpu_kernels.cu:502` |
| `UnshuffleKernel` | `data_stats_gpu_kernels.cu:529` |
| `QuantizeKernel` | `data_stats_gpu_kernels.cu:638` |
| `DequantizeKernel` | `data_stats_gpu_kernels.cu:653` |
| `MinMaxKernel` | `data_stats_gpu_kernels.cu:680` |

## Buffers traced (Protocol §14) — native compress/decompress

`CompContext` (`src/api/internal.hpp:32-95`) is a pooled per-slot struct,
`N_COMP_CTX = 9` slots, each owning a stream, 6 events and all preprocessing scratch.
Scratch is **pre-allocated and reused**, which is why `owns_*` flags exist.

| Buffer | Created by | Allocated where | Size | Producer | Consumer | Copied to | Final destination |
|---|---|---|---|---|---|---|---|
| `d_input` | caller | DEVICE (caller-owned) | `input_size` | caller | quantize / shuffle / codec / stats | — | not modified |
| `ctx->d_stats_workspace` | ctx pool init | DEVICE | ctx-fixed | stats kernels | stats kernels | — | reused per slot |
| `ctx->d_stats` (`AutoStatsGPU*`) | ctx pool init | DEVICE | `sizeof(AutoStatsGPU)` | `runStatsKernelsNoSync` | NN inference (device read), SGD | D2D from `d_precomputed_stats` (:631) | reused per slot |
| `ctx->d_histogram` | ctx pool init | DEVICE | — | entropy histogram kernel | `entropyFromHistogramKernel` | — | reused per slot |
| `ctx->d_fused_infer_output`, `d_fused_top_actions`, `d_fused_costs[32]` | ctx pool init | DEVICE | 32 floats / 32 ints | `nnFusedInferenceKernel` | host (action + costs) | D→H inside `runNNFusedInferenceCtx` | reused per slot |
| `ctx->d_range_min` / `d_range_max` | ctx pool init | DEVICE | scalar | CUB min/max reduction | `quantize_simple` | into header `data_min`/`data_max` | reused per slot |
| `ctx->d_cub_temp` (`cub_temp_cap`) | ctx pool init | DEVICE | grown on demand | CUB | CUB | — | reused per slot |
| `ctx->d_preproc_quant` (`preproc_quant_cap`) | ctx pool init | DEVICE | grown on demand | `quantize_simple` | shuffle or codec | — | reused; `owns_quantized=false` |
| `ctx->d_preproc_shuffle` (`preproc_shuffle_cap`) | ctx pool init | DEVICE | grown on demand | `byte_shuffle_simple` | codec | — | reused; `owns_shuffled=false` |
| `d_temp_out` | `cudaMalloc` (:517) | DEVICE, transient | `64 + max_compressed` | codec | D2D to `d_output` | `d_output` (:603) | `cudaFree` (:608) |
| `d_output` | caller | DEVICE (caller-owned) | `*output_size` | header H2D + codec | caller | — | caller |
| `ctx->d_sgd_grad_buffer`, `d_sgd_output`, `d_sgd_samples` | ctx pool init | DEVICE | fixed | `nnSGDKernel` | weights update | — | reused per slot |
| `d_decompressed` (read path) | `cudaMalloc` (:1242) | DEVICE, transient | `decomp_data_size` | codec | unshuffle / dequant / D2D | — | `cudaFree` (:1306) |
| `d_unshuffled` (read path) | `byte_unshuffle_simple` (:1257) | DEVICE, transient | `decompressed_size` | unshuffle kernel | dequant or D2D | — | `cudaFree` (:1305) |
| `d_dequantized` (read path) | `dequantize_simple` (:1284) | DEVICE, transient | `original_size` | dequant kernel | D2D | `d_output` | `cudaFree` (:1304) |

## Memory transfers identified — native, per compressed chunk

| # | Direction | Size | Site | Conditional? |
|---|---|---|---|---|
| 1 | H→D | 64 B | header write, `cudaMemcpyAsync(d_out, &header, …)` (:610) | always (fast path) |
| 1a | H→D | 64 B | header into `d_temp_out` (:600) | only when caller buffer too small |
| 1b | D→D | `total_size` | `d_temp_out` → `d_output` (:603) | only when caller buffer too small |
| 2 | D→D | `sizeof(AutoStatsGPU)` | precomputed stats → `ctx->d_stats` (:631) | AUTO only |
| 3 | D→H | small | action/costs readback inside `runNNFusedInferenceCtx` | AUTO only |
| 4 | D→H | 64 B | `readHeaderFromDevice` (:1190) | every decompress |
| 5 | D→D | `original_size` | `d_result` → `d_output` (:1299) | every nvcomp decompress |

**Native pays no full-size D→H on either path.** The only host-visible payload traffic is
two 64-byte header copies and the small inference readback. Any port-side full-size D→H is
therefore a divergence, not a difference in style — this is the measuring stick for
Phase 21.

**Synchronization points, native compress:** `cudaStreamSynchronize` at :475 (preproc
timing, conditional), :559 (unconditional, before `get_compressed_output_size`), :607
(temp-buffer path only), :634 (AUTO stats copy). Plus one inside
`runNNFusedInferenceCtx`. **Native compress is not sync-free** — a port that syncs at
these same points is matching, not regressing.

## First known divergence

None established. Phase 1 produced structural differences only (see Known differences);
no behavioral divergence has been demonstrated.

## Relevant test targets (from `ctest -N` in `build/`)

```
 #67  ctp_neuropress_parity
 #68  ctp_neuropress_dataset_parity
 #69  ctp_neuropress_flow_parity
 #70  ctp_neuropress_tiebreak_parity
 #71  ctp_neuropress_device_path_parity
 #72  ctp_neuropress_preprocess_parity
#199  test_neuropress_bridge
#205  compressor_dynamic_neuropress
```

Parity sources: `test/unit/compress/model/parity/` — `neuropress_parity.cu`,
`neuropress_dataset_parity.cu`, `neuropress_flow_parity.cu`,
`neuropress_tiebreak_parity.cu`, `neuropress_device_path_parity.cu`,
`neuropress_preprocess_parity.cu`, plus `neuropress_crosscheck.cc`,
`neuropress_native_replay.cc`, `neuropress_globals_stub.cu`, and two Python comparison
scripts (`neuropress_e2e_compare.py`, `neuropress_e2e_report.py`).

**Caveat inherited and to be re-tested:** these assert *numerical* agreement, not
execution locality. They compile upstream's own `.cu` files and diff numbers.

## Next required action

**All 28 phases are closed.** Four experiments remain, each named with the
work needed. In priority order:

1. **Close Phase 18 (G1) — payload bytes vs native.** The one substantive
   unmeasured equivalence claim. Build `neuropress_native_replay` out-of-tree
   against a NeuroPress build (see `parity/CMakeLists.txt` header for the
   pattern), run `--inference-only` against the learning-off CLIO artifacts
   that now exist — `scratchpad/e2e_nolearn.csv` and
   `/tmp/neuropress_e2e_data.bin` — and diff payloads excluding each side's
   container header.
2. **Add a test for D7-1 (G2).** Assert what happens to a host-resident chunk
   with NeuroPress configured. This is the highest-severity finding and it is
   the path the default functional test already takes.
3. ~~**Fix D19-1.**~~ **DONE 2026-08-13** — `FreeGpuBackend` now releases
   in-process registrations; `cudaFree` 73 → 367. See the D19-1 row.
   **It created one follow-on item:** remotely-registered backends still leak
   by design, because the runtime holds a `cudaIpcOpenMemHandle` mapping and
   the tree has no `DeregisterMemory` task and no `CloseIpcMemHandle` call.
   Closing it means adding that round trip so the importer closes before the
   exporter frees. Currently warns once per process instead.
4. **Settle D8-1 and D16-1** with the experiments in the Phase 26 checkpoint.
5. **Three traceability rows are still NOT STARTED**, and they are genuine
   gaps rather than oversights:
   - nDzip / cuSZp — cuSZp's "absolute error bound" claim is unverified,
     nDzip untouched (Phase 16 covered cuSZ only).
   - HDF5 VOL read path — **VOL-1, VOL-2 and VOL-5 were never examined.**
     VOL-5 is the one worth doing first: the earlier audit reports having
     *reproduced a SIGSEGV* from a legal HDF5 sequence, and nothing in this
     investigation re-tested it.
   - D17-1 — whether `output_fullptr.ptr_` can be host while `codec_dst` is
     device, which would hand `DequantizeDevice` a host destination.

---

<details>
<summary>Superseded: the original "next action" from Phase 2 (kept for audit trail)</summary>

Begin **Phase 3 — Native shuffle / deshuffle**. Produce the §5 ground-truth record for
shuffle before opening the port:
1. Read `src/preprocessing/byte_shuffle.cuh` (109 lines) and
   `byte_shuffle_kernels.cu` (430 lines) in full — `byte_shuffle_simple`,
   `byte_unshuffle_simple`, `populateChunkArraysKernel`, and both specialized kernels.
2. Record chunking (`SHUFFLE_CHUNK_SIZE = 262144`), launch geometry, the leftover-byte
   rule, and behaviour for `num_elements <= 1`.
3. Only then locate the port's `ShuffleKernel` / `UnshuffleKernel`
   (`data_stats_gpu_kernels.cu:502,529`) and `byte_shuffle.h`, and build a byte-level
   isolated test (Protocol §16 Level 1) before touching end-to-end.

**Partially read, must finish in a later phase:**
`gpucompress_compress.cpp:753-1173` — the parallel K-way exploration block and SGD
phase 2. Read as far as :752 (slot struct declaration). Owned by **Phase 8**.
*(Discharged in Phase 8.)*

</details>

---

## Phase checkpoints

### Phase 1 — Native source discovery

```
==================================================
PHASE CHECKPOINT
==================================================

Phase:
1 — Native source discovery

Status:
COMPLETE

Native implementation:
/home/cc/NeuroPress @ b23b8f6. 16,692 lines across src/{api,compression,nn,
preprocessing,selection,stats,hdf5,cli} + include/gpucompress.h (819 lines, ~58
public entry points). 24 __global__ kernels, of which 20 are in scope (4 belong to
the bundled Gray-Scott app and the CLI verify path).

CLIO implementation:
Split across two components. context-transport-primitives owns the algorithms
(NN predictor + kernels, stats/preprocess kernels, codec factory);
context-transfer-engine/compressor owns the orchestration (compressor_runtime.cc,
neuropress_bridge.cc). 15 in-scope __global__ kernels.

Native ground truth:
NOT APPLICABLE for this phase — Phase 1 establishes the source map only. No
behavioral ground truth was claimed.

CLIO behavior:
NOT APPLICABLE for this phase.

Byte comparison:
NOT APPLICABLE

GPU execution:
NOT APPLICABLE

Extra H→D:
NOT INVESTIGATED (Phase 2/21)

Extra D→H:
NOT INVESTIGATED (Phase 2/21)

CPU fallback:
NOT INVESTIGATED (Phase 22)

First divergence:
None established.

Tests executed:
ctest -R ctp_neuropress (6 tests) — see result recorded in "Tests completed" above.

Artifacts generated:
NEUROPRESS_CLIO_INVESTIGATION_STATE.md (this file)
NEUROPRESS_CLIO_TRACEABILITY.md

Remaining issues:
Four native kernels have no located port counterpart and three port kernels have no
located native counterpart (listed under "Known differences"). These are leads for
phases 4, 5, 9, 11 and 17 — not findings.

Next phase:
Phase 2 — Native data/memory flow
==================================================
```

### Phase 2 — Native data/memory flow

```
==================================================
PHASE CHECKPOINT
==================================================

Phase:
2 — Native data/memory flow

Status:
COMPLETE

Native implementation:
src/api/gpucompress_compress.cpp — gpucompress_compress_gpu (:154-233),
gpucompress_infer_gpu (:238-333), gpucompress_compress_with_action_gpu (:339-1173,
read to :752), gpucompress_decompress_gpu (:1175-1312). Supporting: internal.hpp
CompContext (:32-95), decodeAction (:156-172), constants (:97-103);
compression_header.h CompressionHeader (:53-110).

CLIO implementation:
NOT APPLICABLE for this phase — Phase 2 is native-only by Protocol §2 (native first).
Port flow is compared per-functionality in phases 3-17.

Native ground truth:
Recorded in full above: 3-way entry dispatch, exact action encoding, pipeline order
quantize→shuffle→codec→header, 64-byte header layout with the algorithm id packed
into quant_flags bits 9-12, reverse order codec→unshuffle→dequantize→D2D, and the
complete buffer + transfer tables.

CLIO behavior:
NOT INVESTIGATED (by design this phase)

Byte comparison:
NOT APPLICABLE

GPU execution:
PASS (native) — every payload-touching stage is a device kernel or a device-to-device
copy. Confirmed by reading; no host loop over payload exists on either native path.

Extra H→D:
NOT APPLICABLE (native is the reference). Native's own H→D total is 64 bytes/chunk.

Extra D→H:
NOT APPLICABLE (native is the reference). Native's own D→H is 64 bytes/chunk on
decompress plus a small inference readback on AUTO.

CPU fallback:
NONE in native. On AUTO with weights unloaded, gpucompress_compress_gpu prints an
error and returns GPUCOMPRESS_ERROR_NN_NOT_LOADED (:208-213) — it does not degrade.
This is the behaviour the port's commit fa485166 claims to have adopted; verifying
that claim is Phase 22.

First divergence:
None established. No port code was read this phase.

Tests executed:
None this phase (Protocol §16 — test after understanding). Phase 1's 6/6 parity
baseline still stands.

Artifacts generated:
Native ground-truth record, buffer table (§14) and transfer table, all in this file.

Remaining issues:
- gpucompress_compress.cpp:753-1173 (K-way exploration, SGD phase 2) deferred to
  Phase 8 — recorded as a dependency, not skipped.
- compressExternalGpu (:61-152) read only at the dispatch level; body deferred to
  Phase 16 (static quantized / external codecs).
- runStatsKernelsNoSync, runNNFusedInferenceCtx, quantize_simple,
  byte_shuffle_simple bodies not yet read — they belong to phases 5, 11, 4, 3.

Next phase:
Phase 3 — Native shuffle / deshuffle
==================================================
```

### Phase 3 — Shuffle / deshuffle

```
==================================================
PHASE CHECKPOINT
==================================================

Phase:
3 — Native shuffle / deshuffle, and the port compared against it

Status:
COMPLETE (with two bounded differences recorded, neither reachable in
production and neither affecting produced bytes)

Native implementation:
src/preprocessing/byte_shuffle.cuh (109 lines, read in full)
src/preprocessing/byte_shuffle_kernels.cu (430 lines, read in full)
Kernels byte_shuffle_kernel_specialized<1,2,4,8> and the unshuffle twin;
populateChunkArraysKernel; three byte_shuffle_simple overloads.

CLIO implementation:
Device: ShuffleKernel<E> / UnshuffleKernel<E>
  (data_stats_gpu_kernels.cu:502, :529), entry points ByteShuffleDevice /
  ByteUnshuffleDevice (:574, :594), geometry in PrepareLaunch (:554).
Host:   ByteShuffle / ByteUnshuffle (byte_shuffle.h:67, :131), plus
  ByteShuffleVector / ByteUnshuffleVector (:182, :202).
CPU-only build: byte_shuffle_cpu_stub.cc:26-27 — both Device entry points
  return false unconditionally.
Chunk constant: kShuffleChunkBytes = 256*1024 (byte_shuffle.h:47) —
  IDENTICAL to native's SHUFFLE_CHUNK_SIZE.

Native ground truth:
Recorded above in the §5 record. Per-chunk permutation out[b*n+e] = in[e*E+b]
with the trailing partial element copied verbatim, chunked at 262144 B.

CLIO behavior:
Same permutation, same chunk size, same leftover rule, in all three
implementations. The port's device kernel is grid-strided over chunks
(blocks capped at 65535, :563) where native launches exactly one block per
chunk; since each chunk is still processed by exactly one block in both, the
output is unaffected. Native's `n <= 1 → verbatim copy` special case has no
port counterpart, but the port's general loop degenerates to the identity for
n == 1 and to leftover-only for n == 0, so the bytes agree — and this is
covered empirically by the n=1,2,3 cases below rather than left as an argument.

Byte comparison:
PASS — BYTE-IDENTICAL.
ctp_neuropress_preprocess_parity links upstream's real byte_shuffle_simple /
byte_unshuffle_simple at upstream's real SHUFFLE_CHUNK_SIZE and diffs against
both port implementations over 3 element sizes x 30 buffer sizes:
  1,2,3,4,5,7,8,9,15,16,17,31,255,256,257,1023,1024,4095,4096,4097,
  262143,262144,262145,262151,524288,524291,1048576,4194304,4194309
covering sub-element, exact multiples, the 256 KiB chunk edge on both sides,
and the production 4 MiB chunk with a ragged tail.
Asserted per case (test lines 186-214):
  native_shuffled == clio_device_shuffled       (byte-exact)
  native_shuffled == clio_host_shuffled         (byte-exact)
  native_unshuffle(native_blob)  == original
  clio_device_unshuffle(NATIVE blob) == original   ← cross-interchange
  clio_host_unshuffle(NATIVE blob)   == original   ← cross-interchange
Measured this session: 1014 checks, 0 failures.

GPU execution:
PASS for the device path — ShuffleKernel/UnshuffleKernel are real __global__
kernels on a real stream.
The port ALSO retains a host implementation, selected at
compressor_runtime.cc:2359 by `if (ctp::IsDevicePointer(input_ptr))`. Whether
shipped chunks arrive device-resident is a RESIDENCY question, not a shuffle
question — recorded as a dependency and deferred to Phase 20/21. The prior
audit's PRE-2 ("host scalar loop, 40/64 chunks") is therefore neither
confirmed nor refuted here; what IS now established is that if the host path
runs, it emits byte-identical output, so PRE-2 is a performance/locality
issue only and cannot corrupt a blob.

Extra H→D:
NO — device path allocates via CLIO_IPC->AllocateAndRegisterGpuBackend and
shuffles in place on the device (compressor_runtime.cc:2366-2372).

Extra D→H:
NO on the device path. The host path implies a prior D→H, but that copy is
performed by the residency logic upstream of shuffle, not by shuffle.

CPU fallback:
YES, two distinct mechanisms, both recorded as differences:
 (a) Host-resident input → host ByteShuffle by design (:2379-2388). Native
     has no equivalent because native only accepts device pointers.
 (b) CPU-only build → byte_shuffle_cpu_stub.cc returns false. At :2369-2378
     the port then frees the scratch, leaves applied_shuffle = 0 and records
     "not shuffled" in the header. Native in the same situation returns
     GPUCOMPRESS_ERROR_COMPRESSION (gpucompress_compress.cpp:466-469).
     Category: loud-error-became-silent-degrade. Consequence is compression
     RATIO only — the header honestly records no shuffle, so the read path
     stays consistent and no data is corrupted.

First divergence:
None in produced bytes. Two behavioral differences, both bounded:
 D3-1  element_size == 1: native instantiates and dispatches
       byte_shuffle_kernel_specialized<1> (verbatim copy); the port rejects it
       in BOTH PrepareLaunch (:557) and host ByteShuffle (byte_shuffle.h:75).
       UNREACHABLE in production — decodeAction emits shuffle_size 0 or 4 only
       (internal.hpp:171). Severity: none as shipped.
 D3-2  element_size in {3,5,6,7,9+}: native's switch default silently treats
       them as 4 (byte_shuffle_kernels.cu:246); the port rejects. Same
       reachability argument. Arguably the port's behavior is safer.
 D3-3  Stream: port uses an internal DeviceStatsStream() (:582, :602); native
       uses the caller's ctx->stream. Both synchronize before returning
       (FinishLaunch :569 vs native :322/:367/:421), so ordering is safe;
       only achievable concurrency differs. Cost unmeasured — NOT claimed as
       harm (this is the mistake §"Where the audit was weakest" flags).

Tests executed:
ctest -R ctp_neuropress_preprocess_parity  → PASS, 1014 checks / 0 failures.

Artifacts generated:
§5 ground-truth record for shuffle (above); this checkpoint; traceability
rows for Shuffle / Deshuffle updated to PASS — BYTE-IDENTICAL.

Remaining issues:
- Execution locality of shuffle on shipped paths → Phase 20/21 (dependency
  recorded, phase not abandoned).
- populateChunkArraysKernel has no port counterpart because the port's kernel
  computes chunk geometry inline instead of materializing pointer arrays. Not
  a gap: it means the port avoids native's 3 cudaMalloc + 3 cudaFree per
  shuffle call. Perf claim NOT measured, so not asserted as an improvement.

Next phase:
Phase 4 — Native quantization / dequantization
==================================================
```

### Phase 4 — Quantization / dequantization

```
==================================================
PHASE CHECKPOINT
==================================================

Phase:
4 — Native quantization / dequantization, and the port compared against it

Status:
COMPLETE

Native implementation:
src/preprocessing/quantization.cuh (434 lines) + quantization_kernels.cu
(917 lines). quantize_simple x3 (:368, :559, :689 — compress path uses :689),
dequantize_simple (:798), quantize_linear_kernel (:55),
dequantize_linear_kernel (:84), compute_data_range_typed (:187/:237) over CUB
DeviceReduce, compute_required_precision (quantization.cuh:192).

CLIO implementation:
Device: QuantizeDevice / DequantizeDevice
  (data_stats_gpu_kernels.cu:~760, :849), kernels QuantizeKernel<T> (:638) and
  DequantizeKernel<InT> (:653), range via DeviceMinMax (:718) →
  MinMaxKernel (:680).
Host:   Quantize<float> / QuantizationResult (quantization.h:100-240).
Dispatch in compressor_runtime.cc: QuantizeDevice at :1728 (exploration) and
  :2304 (main write path); host Quantize<float> at :1741, :2332, :3317;
  DequantizeDevice at :2973; host dequantize at :3535.

Native ground truth:
Recorded above in the §5 record — including the full effective_eb derivation,
which is where a port is most likely to drift silently.

CLIO behavior:
The effective_eb derivation is replicated constant-for-constant in BOTH port
implementations:
  host   quantization.h:148-179
  device data_stats_gpu_kernels.cu:777-811
Verified identical term by term: the `data_range <= 0 → 1.0` constant-data
clamp, 2.4e-7, the 0.05 safety margin, data_range/4.0e9, the
`float_repr_error * 0.1` fallback, the final fmax, scale = 1/(2*eb), and
ComputeRequiredPrecision's `* 1.1` with the 127 / 32767 cutoffs
(quantization.h:100-107). Kernel clamp constants match exactly
(-128/127, -32768/32767, -2147483648/2147483647).

Byte comparison:
PASS — BYTE-IDENTICAL, on both the packed quantized output and the
dequantized floats.
ctp_neuropress_preprocess_parity QuantizeCase (test:275-382) links upstream's
real quantize_simple / dequantize_simple and asserts per case:
  precision equal
  packed byte count equal
  data_min  == params.data_min      (exact double equality)
  data_max  == params.data_max      (exact double equality)
  scale     == params.scale         (exact double equality)
  packed quantized bytes byte-identical
  dequantized float bytes byte-identical
  max|err| <= requested bound on BOTH sides, asserted only where the bound is
  achievable (upstream itself deliberately exceeds it below float32
  resolution, and the port records the same via bound_achievable)
Coverage: 5 regimes {random, smooth, CONSTANT, wide-range, negative} x 5
error bounds {1e-1, 1e-2, 1e-3, 1e-5, 1e-7} at n = 65536, plus a 1 MiB
production-sized chunk at eb=1e-3. The `constant` regime is the direct test of
the range clamp that commit 4f3d7e5b added.
Measured this session: part of the 1014 checks / 0 failures run.

GPU execution:
PASS for the device path. Both kernels are real __global__ launches on
DeviceStatsStream().
Note a genuine ALGORITHMIC difference in the range reduction that does NOT
change results: native uses CUB DeviceReduce::Min/Max; the port uses its own
MinMaxKernel (:680) with atomics over a monotonic float→uint encoding. This is
why MinMaxKernel appeared in Phase 1 as "no native counterpart" — it replaces
a CUB call, not a kernel. Min/max are exact and order-independent, so the two
agree bit-for-bit; the exact-equality assertions on data_min/data_max over the
negative and wide-range regimes are the evidence, not the argument.

Extra H→D:
NO. Port and native both bring back only the min/max scalars.

Extra D→H:
NO on the device path.

CPU fallback:
YES — same shape as shuffle: host Quantize<float> is selected when the input
is host-resident. Byte-identical to the device path (the test asserts the
device path against upstream, and the host path shares the effective_eb code),
so this is a locality question for Phase 20/21, not a correctness one.

First divergence:
None in produced bytes or metadata.
 D4-1  The port adds `if (effective_eb <= 0.0) return failure`
       (quantization.h:174, data_stats_gpu_kernels.cu:809); native has no such
       guard. UNREACHABLE in both: after the constant-data clamp data_range >=
       1.0, so min_eb_for_int32 = data_range/4e9 > 0 and the final fmax makes
       effective_eb > 0 unconditionally. Dead code on both sides.
 D4-2  Native reports achievability by PRINTING A WARNING and exceeding the
       bound; the port returns it as structured state (bound_achievable).
       Same numeric behavior, better API. Not a divergence in output.

HYPOTHESIS TESTED — audit finding PRE-14 ("DequantizeDevice has no
pointer-kind guard"):
STATUS: CONFIRMED AS CODE, but RECLASSIFIED — it is NOT a parity divergence.
DequantizeDevice (:849-857) validates only null / zero-length / scale, never
pointer kind. But native's dequantize_simple has no such guard either — the
only cudaPointerGetAttributes call anywhere in native's preprocessing or api
layer is gpucompress_api.cpp:458, which implements the public
gpucompress_is_device_ptr, not an internal guard. **Both sides take device
pointers by contract.** So the port matches native here; whether a CLIO caller
ever violates that contract is a Phase 19/20 question and is recorded there.
The audit filed this as a latent port defect; against native as the reference
it is not one.

Tests executed:
ctest -R ctp_neuropress_preprocess_parity (covers phases 3 AND 4)
  → PASS, 1014 checks / 0 failures.

Artifacts generated:
§5 ground-truth record for quantization (above); this checkpoint; traceability
rows for Quantization / Dequantization / Min-max / Error-bound verify.

Remaining issues:
- verify_error_bound_kernel: native's only callers are src/cli/compress.cpp:601
  and native's own unit tests (tests/unit/test_preprocessing.cu,
  test_quantization_cub.cu). It is NOT on the library compress path, so the
  port needing no counterpart is NOT a gap → marked NOT APPLICABLE.
- The audit's cuSZ absolute-vs-relative error-bound claim is NOT tested here.
  It concerns the external cuSZ codec, not this linear quantizer, and belongs
  to Phase 16. Explicitly still OPEN.

Next phase:
Phase 5 — Native statistics (entropy, MAD, second derivative, min/max/mean)
==================================================
```

### Phase 5 — Statistics

```
==================================================
PHASE CHECKPOINT
==================================================

Phase:
5 — Native statistics, and the port compared against it

Status:
COMPLETE

Native implementation:
src/stats/stats_kernel.cu (470) — statsPass1Kernel (:106), madPass2Kernel
(:201), finalizeStatsOnlyKernel (:260), runStatsKernelsNoSync,
runStatsOnlyPipeline. src/stats/entropy_kernel.cu (253) — histogramKernel
(:48), histogramKernelVec4 (:99), entropyFromHistogramKernel (:167),
launchEntropyKernelsAsync (:216). src/stats/auto_stats_gpu.h (38).

CLIO implementation:
data_stats_gpu_kernels.cu — StatsPass1Kernel (:32), StatsPass2Kernel (:135),
StatsPass2DevKernel (:163), EntropyFromHistKernel (:198),
FinalizeFeatureStatsKernel (:227), ComputeDeviceStats (:384),
ReadDeviceFeatureStats. Host side: DataStatisticsFactory
(CalculateShannonEntropy / CalculateMAD / CalculateSecondDerivative).
Dispatch: ComputeCompressionFeatures (data_stats_gpu.h:120-147).

Native ground truth:
Recorded above. The two facts that matter most: deriv_normalized divides by
(n-2), and native's own accumulation is atomicAdd(double) — so native is not
bit-reproducible against itself.

CLIO behavior:
Formulas match: second_derivative = scalars[1] / (n - 2) with the same n > 2
guard (data_stats_gpu_kernels.cu:234-235, and again at :376-377 for the host
path); entropy = -sum(p * log2(p)) with the same > 0 bin filter (:208, :370).
One deliberate STRUCTURAL difference, documented in the port at :25-44: CLIO
FUSES the byte histogram, the typed value sum and the second-derivative sum
into a single grid-stride pass, where native runs statsPass1Kernel and the
histogram kernel separately. CLIO also does not use native's
histogramKernelVec4 variant. Both differences change only the order of
integer histogram increments, which cannot change the histogram — so the
entropy input is identical. This is the reason Phase 1 could not find a port
counterpart for histogramKernel: it was fused, not dropped.

Byte comparison:
NOT APPLICABLE — and deliberately so. Byte equality is unachievable in
principle here because native's own double atomicAdd reduction order varies
run to run (see DETERMINISM above). The correct criterion is relative
agreement at double epsilon, and that is what was measured:

PASS — NUMERICALLY EQUIVALENT.
ctp_neuropress_dataset_parity runs native's runStatsOnlyPipeline and CLIO's
ComputeCompressionFeatures over THE SAME device bytes: a 1024 MiB synthetic
dataset, 256 chunks of 4 MiB, spanning 5 data regimes (constant, smooth sine,
sine+noise, mid-entropy, hash noise). Assert tolerance 1e-9 relative.
Worst-case relative deviation measured over all 256 chunks, this session:
    entropy           1.5e-15
    mad               5.41e-15
    second derivative 2.38e-16
That is 6 orders of magnitude inside the assert tolerance and at the level of
double-precision rounding — i.e. as close as two different reduction orders
can possibly agree.
Run total: 7940 checks, 0 failures.

GPU execution:
PASS for the device path — all five CLIO stats kernels are real __global__
launches, and the dataset test drives them with a device pointer.
The audit's central empirical claim (STATS-5 / ORCH-1: "compressor_dynamic_
neuropress launches ZERO stats kernels; entropy/MAD/2nd-derivative computed on
the CPU") is NOT re-tested here and remains OPEN. What Phase 5 establishes is
narrower and worth stating precisely: if the host path runs, it produces the
same numbers to 5e-15, so STATS-5 can only ever have been a locality/perf
finding — it cannot skew a prediction or change a selection.

Extra H→D:
NO.

Extra D→H:
NO on the compute path. Native fetches entropy+mad+deriv in one 24-byte D→H;
CLIO exposes the same fetch as a separate ReadDeviceFeatureStats call,
deliberately callable AFTER inference is enqueued rather than between the two
(data_stats_gpu.h:100-106) — same traffic, later placement.

CPU fallback:
YES, host-pointer-selected, same shape as phases 3 and 4
(ComputeCompressionFeatures:124). One behavior worth crediting: on a DEVICE
pointer whose device-stats path fails, CLIO does NOT fall through to the host
routines — those would dereference a device pointer and segfault. It zeroes
the outputs and returns false so the caller can decide (:130-139). Native has
no host path at all and simply errors, so CLIO is stricter than the audit's
"silent-cpu-fallback" category suggests for this particular call.

First divergence:
None. No divergence found in phase 5.

Tests executed:
ctest -R ctp_neuropress_dataset_parity → PASS, 7940 checks / 0 failures,
0.89 s. Per-chunk table written to neuropress_selection_parity.csv (1024 rows).

Artifacts generated:
§5 ground-truth record for statistics; this checkpoint; traceability rows for
Statistics and Entropy.

Remaining issues:
- STATS-5 / ORCH-1 execution-locality claim still OPEN → Phase 20/21.
- Native's vmin/vmax fields: the earlier audit struck STATS-7 on the grounds
  they are dead upstream. Confirmed structurally this phase — statsPass1Kernel
  writes them, but finalizeStatsOnlyKernel consumes only mad_sum and
  abs_diff_sum, and the 24-byte D→H fetches only entropy/mad/deriv. No
  further action.
- This test ALSO produced Phase 7 evidence (0 of 256 chunks selected a
  different config, across 4 error bounds). Recorded, but Phase 7 is NOT
  marked done on the strength of it — the ranking/tie-break analysis it
  depends on has not been read yet.

Next phase:
Phase 6 — Native per-chunk diagnostics
==================================================
```

### Phase 6 — Per-chunk diagnostics

```
==================================================
PHASE CHECKPOINT
==================================================

Phase:
6 — Native per-chunk diagnostics, and the port compared against it

Status:
COMPLETE — DIFFERENCE FOUND (observability gap, not a correctness gap)

Native implementation:
gpucompress_chunk_diag_t (include/gpucompress.h:625-721) — ~80 fields
including five 31/32-wide arrays (explore_alternatives, explore_ratios,
explore_comp_ms, explore_costs, predicted_ranking, predicted_costs).
Producer: recordChunkDiagnostic (src/api/gpucompress_diagnostics.cpp:123),
fed by ChunkDiagInput (src/api/internal.hpp:250-296).
Store: src/api/diagnostics_store.hpp (454 lines).
Public accessors: gpucompress_get_chunk_history_count,
gpucompress_get_chunk_diag, gpucompress_record_chunk_decomp_ms,
gpucompress_record_chunk_vol_timing, gpucompress_record_chunk_s1_timing,
gpucompress_reset_chunk_history, gpucompress_set_debug_context.

REACHABILITY (Protocol §15 — verified, not assumed):
Production side: recordChunkDiagnostic is called on native's CORE compress
path, not only the VOL — gpucompress_compress.cpp:148 (external codecs) and
:1159 (the main with_action path). So native records a diagnostic for EVERY
compression.
Consumer side: every caller of gpucompress_get_chunk_diag /
gpucompress_get_chunk_history_count in the native tree is a TEST or BENCHMARK
— tests/test_nn_bitcomp.cu, test_nn_predict_vs_actual.cu,
test_nn_algo_convergence.cu, regression/test_nn_timing_inflation.cu,
hdf5/test_vol_lossless_stress.cu. There is NO production consumer of the
struct. The one functional consumer of the STORE is internal:
gpucompress_batched_decomp_sgd() re-sweeps it to build decomp-head SGD
batches.

CLIO implementation:
NONE for the struct. Searched the whole tree (excluding build/ and docs/) for
gpucompress_get_chunk_diag, gpucompress_get_chunk_history_count,
gpucompress_record_chunk_*, gpucompress_set_debug_context — **zero call
sites**. CLIO neither reads native's diagnostic history nor maintains a
parallel gpucompress_chunk_diag_t.
What CLIO DOES replicate is the functional slice: decomp_features_, a
blob-keyed map of {CompressionFeatures features, double measured_ms}
(compressor_runtime.cc:1040-1067), which stands in for native's feat_action /
feat_entropy / feat_mad / feat_deriv / feat_eb_enc / feat_ds_enc plus
decompression_ms.

Native ground truth:
Two distinct roles, and separating them is the whole result of this phase:
  ROLE 1 — observability for native's own test/benchmark suite. No production
           consumer. A port that omits it loses reporting, not behavior.
  ROLE 2 — backing store for the batched decompression-head SGD. Functional.

CLIO behavior:
ROLE 1 is not ported. ROLE 2 is ported, with two upstream behaviors
deliberately matched and annotated in the port:
  - the 1 ms floor on the measured decompression time before it becomes a
    learning target, matching diagnostics_store.hpp:96-102 (runtime.cc:1047-1054)
  - re-sweeping every record with a measurement on each read rather than
    consuming entries, matching gpucompress_batched_decomp_sgd
    (runtime.cc:1056-1062)

Byte comparison:
NOT APPLICABLE — diagnostics are timing and bookkeeping values, not payload.
Wall-clock fields (nn_inference_ms, compression_ms, vol_d2h_copy_ms, …) are
machine- and run-dependent on both sides; asserting equality on them would be
meaningless.

GPU execution:
NOT APPLICABLE — diagnostics are host-side bookkeeping in both trees.

Extra H→D / Extra D→H:
NO.

CPU fallback:
NOT APPLICABLE.

First divergence:
 D6-1  CLIO has no equivalent of gpucompress_chunk_diag_t or its accessor
       API. Category: missing-observability. Reachability: native's own
       consumers are all tests/benchmarks, so nothing in a production CLIO
       path is deprived of an input it needs.
       Consequence: CLIO cannot reproduce native's per-chunk MAPE / regret /
       exploration-ranking reports, so any future comparison that wants those
       numbers side by side has no CLIO source for them. That matters for
       Phase 25 (performance comparison) and Phase 27 (final matrix) — noted
       there as a dependency.
       Severity: LOW as shipped. NOT a correctness defect.

Tests executed:
None new. This phase is a reachability and surface-area determination; there
is no numeric property to test, and inventing a timing-equality assertion
would be the "unknown marked as pass" failure Protocol §7 forbids.

Artifacts generated:
This checkpoint; traceability row for Diagnostics.

Remaining issues:
- Whether CLIO's decomp_features_ produces the SAME learning targets as
  native's diagnostics-store sweep is NOT settled here — it is a numeric
  question about the decomp head and belongs to Phase 13. Recorded as a
  dependency.
- The O(n^2) re-sweep the port deliberately copies (batch grows on every
  read) was filed by the earlier audit as part of NN-1. The port matching it
  is intentional parity, not an oversight; whether the shared behavior is
  desirable is a question for whoever owns the learning path, and is out of
  scope for an equivalence investigation.

Next phase:
Phase 7 — Native static selection
==================================================
```

### Phase 7 — Static selection

```
==================================================
PHASE CHECKPOINT
==================================================

Phase:
7 — Native static selection, and the port compared against it

Status:
COMPLETE — DIFFERENCE FOUND (one high-severity, on host-resident chunks)

Native implementation:
Native has THREE non-NN selection routes. All three located and their
reachability determined:
 (1) Explicit algorithm (cfg.algorithm != ALGO_AUTO):
     gpucompress_compress.cpp:178-187 synthesizes the action on the host —
     action = (cfg.algorithm - 1) % 8, += 8 if PREPROC_QUANTIZE, += 16 if
     shuffle size > 0 — and delegates to the same
     gpucompress_compress_with_action_gpu pipeline. REACHABLE, it is the
     documented way to bypass the NN.
 (2) Entropy heuristic: heuristic_select_action (src/selection/heuristic.cu:26)
     — entropy < 3.5 -> action 4 (zstd), < 5.5 -> 3 (gdeflate), else 0 (lz4);
     never sets shuffle or quantize.
     Gate: g_selection_mode == GPUCOMPRESS_SELECT_HEURISTIC.
     **UNREACHABLE AS SHIPPED** — verified: the only consumer of
     g_selection_mode is H5VLgpucompress.cu:2309 (VOL only, not the core
     library), the default is GPUCOMPRESS_SELECT_NN (gpucompress_api.cpp:96,
     re-asserted at :287), and a tree-wide search found **no caller of
     gpucompress_set_selection_mode anywhere in NeuroPress** — not in src,
     tests, benchmarks, scripts or configs. It is dead upstream, in the same
     way STATS-7's vmin/vmax were.
 (3) Host-pointer compress: gpucompress_compress (gpucompress_api.cpp:352).
     **IT IS A STUB.** The body discards every argument, prints
     "host-path stub — use gpucompress_compress_gpu()" and returns
     GPUCOMPRESS_ERROR_INVALID_INPUT. NeuroPress has NO host compression path
     of any kind.

CLIO implementation:
compressor_runtime.cc EstCompressionStats (:571-...) and its caller
Compress (:1314-1345). Explicit codec selection is context.compress_lib_
(:2222), which addresses codecs directly rather than through action IDs.
Clio's non-NeuroPress selectors are its OWN pre-existing qtable / dense-NN
"legacy heuristics" — explicitly Clio's, not a port of anything
(runtime.cc:758).

Native ground truth:
On the device path, when statistics or inference cannot produce a decision,
native PROPAGATES THE FAILURE: a null d_stats_ptr yields
GPUCOMPRESS_ERROR_NN_NOT_LOADED (gpucompress_compress.cpp:285), and AUTO with
no weights prints an error and returns GPUCOMPRESS_ERROR_NN_NOT_LOADED
(:208-213). It never substitutes another selector.

CLIO behavior:
Split cleanly by residency, and the two halves get different verdicts:

  DEVICE-RESIDENT chunk, NeuroPress active, GPU path fails
    → out_neuropress_gpu_failed set (:670, :708-709)
    → caller sets task->return_code_ = 4 and returns (:1324-1338)
    → THE WRITE FAILS.
    **MATCHES native.** This is commit fa485166's behavior and it is real —
    verified by reading the flag from producer to consumer, not from the
    commit message. The port's comment cites upstream's own propagation path
    (H5VLgpucompress.cu:2057, :2463, :3542).

  HOST-RESIDENT chunk, NeuroPress configured
    → num_elements == 0 or !features_ok or NeuroPress not ready
    → falls through to Clio's legacy qtable/dense-NN heuristics (:679-688,
      :698)
    → a blob IS produced, chosen by a DIFFERENT MODEL, and success is
      reported.

First divergence:
 D7-1  **HIGH.** For a host-resident chunk with NeuroPress configured, CLIO
       silently selects with a different model where native cannot proceed at
       all. Native's host entry point is a hard-failing stub
       (gpucompress_api.cpp:352-366), so there is no native behavior to match
       here — native simply refuses. CLIO reports success.
       Category: loud-error-became-silent-degrade.
       Consequence: selection parity does NOT hold for host-resident chunks.
       The blob is still valid and losslessly recoverable — this is a
       *which-algorithm-was-chosen* divergence, not a corruption one.
       Reachability: depends entirely on how often shipped chunks are
       host-resident, which is exactly the open Phase 20/21 question. If
       chunks are always device-resident this is unreachable; if the earlier
       audit's 40/64 figure is representative it is the common case.
       **This is now the single most consequential open item in the
       investigation.**
 D7-2  LOW. heuristic_select_action is not ported. NOT A GAP — it is
       unreachable upstream (no caller of gpucompress_set_selection_mode
       exists). Marked NOT APPLICABLE.
 D7-3  LOW. CLIO has no counterpart to native's explicit-algorithm action
       synthesis because it addresses codecs directly via compress_lib_
       rather than through action IDs. Both bypass the NN and reach the same
       codecs; there is no shared numeric behavior to compare.

Byte comparison:
NOT APPLICABLE for routes (2) and (3) — dead and stub respectively.
For the DEVICE path, selection agreement was measured in Phase 5's run:
**0 of 256 chunks chose a different config**, across 4 error bounds
(eb = 0, 1e-6, 1e-3, 1e-2) = 1024 comparisons, full table in
neuropress_selection_parity.csv. Quantize winners: 205 of 256 chunks at each
non-zero bound, 0 at eb=0 (correct — the quantize half is masked to -INF when
error_bound <= 0).

GPU execution:
PASS on the device path.

CPU fallback:
YES — D7-1 above. This is the real one; the shuffle/quant/stats host paths
found in phases 3-5 at least compute the same numbers, whereas this one
changes the decision.

Tests executed:
No new test run this phase. Evidence reused from
ctp_neuropress_dataset_parity (Phase 5 run): 0/256 selection divergence on
device-resident chunks. No test in the suite covers the host-resident
selection path — that gap is itself a finding, recorded for Phase 24.

Artifacts generated:
This checkpoint; traceability rows for Static selection and Action encoding.

Remaining issues:
- D7-1's reachability is unresolved until Phase 20/21 measures residency.
- No test covers host-resident NeuroPress selection → Phase 24 must add one
  or explicitly record the gap.

Next phase:
Phase 8 — Native exploration
==================================================
```

### Phase 8 — Exploration

```
==================================================
PHASE CHECKPOINT
==================================================

Phase:
8 — Native exploration, and the port compared against it
(this phase also discharges the Phase 2 deferral of
gpucompress_compress.cpp:753-1173)

Status:
COMPLETE — DIFFERENCE FOUND (timing instrument; affects learning, not
correctness)

Native implementation:
gpucompress_compress.cpp:733-999. Trigger (:733):
    g_exploration_enabled && (g_best_mode || error_pct > g_exploration_threshold)
    && top_actions != nullptr
K = g_exploration_k_override > 0 ? override : 3; actual_K = min(K, 31).
Defaults verified in gpucompress_api.cpp: g_exploration_enabled = FALSE (:83),
g_exploration_threshold = 0.50 (:84), g_best_mode = false (:97),
g_reinforce_mape_threshold = 0.30 (:103), g_rank_w0/w1/w2 = 1.0 (:67-69),
g_measured_bw_bytes_per_ms = 5e6 (:71). All are overridable from environment
variables (:233, :251, :256).

Structure: three phases over an ExploreSlot[31] (:751-771).
  Phase 1 (:778-870) — per slot: own cudaStream, own events, own
    cudaMalloc'd output at gpucompress_max_compressed_size(input_size), own
    8-byte range buffers, own quantize (per-slot buffers, NOT the ctx
    pre-allocated ones), own shuffle via the ALLOCATING overload, and a FRESH
    compression manager. K compressions launched in PARALLEL on K streams.
    Skips: alt_action == nn_action (:780); quantized actions when
    error_bound <= 0 (:783).
  Phase 2 (:886-968) — sync, read sizes, score, adopt.
  Phase 3 (:974-986) — free every per-slot resource.

Native ground truth — the scoring rules that matter:
  cost = compute_cost(measured comp_time, **pred_dt**, measured ratio)
         (:904-905). Decompression time is the PRIMARY's NN prediction held
         constant across every alternative, because nothing is decompressed
         at write time.
  compute_cost (:688-692): ct = max(1, ct), dt = max(1, dt),
         rc = min(100, ratio), cost = w0*ct + w1*dt + w2*ds/(rc*bw),
         and 1e30 when rc <= 0.
  Adoption: strict `alt_cost < best_cost` (:909), best_cost seeded with
         primary_cost — so a TIE KEEPS THE PRIMARY.
  Winner is written over the output with a FRESH header (:918-949): header
         copied synchronously (deliberate — alt_hdr is stack memory, :942-945),
         payload D2D async.
  PSNR: analytical from the quantization range; **-1.0 sentinel** for
         non-quantized actions, because 0.0 would be remapped to 120 by
         nn_gpu.cu:689 (:893-901).
  regret = (primary_cost - best_cost) / best_cost (:990-991).

CLIO implementation:
compressor_runtime.cc:1626-2090. Trigger (:1626-1628):
    config_.neuropress_exploration_enabled_ &&
    error_pct > config_.neuropress_exploration_threshold_
Candidates: the next-best ranked stats, skipping the (lib, preset) actually
used (:1633-1642) — equivalent intent to native's alt_action == nn_action skip.

CLIO behavior — what MATCHES:
  - cost lambda (:1515-1524) is native's compute_cost term for term, INCLUDING
    the 1 ms floors, the 100x ratio cap and the 1e30 branch. Constants
    kCostW0/W1/W2 = 1.0 and kCostBandwidthBytesPerMs = 5e6 equal native's
    defaults exactly.
  - decompression time held at the PRIMARY's prediction for every alternative
    (:1857-1858), citing upstream's own line numbers. The port's comment
    records that using each candidate's own predicted dt was tried and
    skewed both the ranking and the regret figure.
  - strict `alt_cost < best_cost` with best_cost seeded from the primary
    (:1648, :1859) — ties keep the primary, as upstream.
  - per-candidate quantize-THEN-shuffle, matching upstream's slot order
    (:1700-1704, :1760-1802).
  - quantization state carried onto the adopted blob's header (:1886-1890) —
    without it the header would claim "not quantized" over a quantized
    payload, which is unrecoverable on read.
  - exploration OFF BY DEFAULT on both sides.

First divergence:
 D8-1  **MEDIUM-HIGH. The timing instrument differs, and cost depends on it.**
       Native brackets ONLY the codec launch with per-slot CUDA events
       (:868-870, cudaEventElapsedTime at :891) — pure GPU kernel time.
       CLIO measures with std::chrono::high_resolution_clock around the whole
       alt_compressor->Compress(...) call (:1836-1843), which per commit
       a7886973's own description also covers ToDeviceInput staging, a
       possible cudaMalloc, configure_compression, the stream sync and the
       output copy.
       CLIO is INTERNALLY consistent — the primary is measured the same way
       (:2430-2436, :2472) — so alternatives and primary are commensurable
       with each other. But both are systematically larger than native's
       numbers for the same work.
       Because cost feeds (a) the exploration winner, (b) SGD training
       targets for compression time, and (c) the error_pct that GATES whether
       SGD and exploration fire at all, this is a behavioral divergence, not
       a reporting one.
       ALREADY KNOWN AND ACKNOWLEDGED UPSTREAM OF ME: commit a7886973 added
       CLIO_CODEC_KERNEL_TIMING to measure like-for-like, kept it OFF BY
       DEFAULT, and states "Not claimed as parity". Its measured kernel-vs-
       kernel medians over 255 x 4 MiB chunks: compress clio/native 1.105,
       decompress 1.266. The wall-clock-vs-kernel gap on the exploration path
       is LARGER than that and is UNMEASURED.
       Falsification: enable CLIO_CODEC_KERNEL_TIMING, feed
       ctp::LastCodecKernelMs() into the cost lambda instead of the chrono
       delta, and check whether any exploration winner changes. NOT DONE —
       recorded for Phase 25.
 D8-2  LOW. Native explores K candidates in PARALLEL on K separate streams;
       CLIO explores SEQUENTIALLY (documented deliberate simplification,
       :1618-1625). Parallel slots contend for the GPU, so native's measured
       per-slot times are inflated relative to isolated runs and CLIO's are
       not — a second systematic timing difference on top of D8-1, in the
       opposite direction.
 D8-3  LOW. CLIO additionally requires winner_total < chunk_size before
       adopting (:1867-1869). Native's only condition is that the winner fits
       the caller's buffer. Documented and justified: Clio HAS a raw-storage
       path native lacks, so it must not replace a raw blob with something
       larger than the original. Cannot cause a wrong result; can only
       decline an adoption native would make.
 D8-4  LOW. Native's cost weights and bandwidth are runtime-settable
       (gpucompress_set_ranking_weights, gpucompress_set_bandwidth, plus env
       vars at gpucompress_api.cpp:233); CLIO hardcodes them as constexpr.
       Values agree at the defaults, so this only diverges if someone changes
       them — no caller in either tree does.
 D8-5  LOW. Native rewrites the output buffer in place on each new winner;
       CLIO remembers the best and re-stores once after the loop (:1651-1659),
       because Clio's blob is already persisted by AsyncCompress before
       exploration runs. Same observable outcome — native's intermediate
       writes land in a buffer nothing can observe.

Byte comparison:
NOT APPLICABLE. Exploration's output is whichever candidate blob won; the
blob itself is produced by the codec + preprocessing paths already proven
byte-identical in phases 3, 4 and 15/16. What differs is WHICH candidate
wins, and that is a cost comparison, not a byte comparison.

GPU execution:
PASS — the port compresses explored candidates into DEVICE memory when the
input is device-resident (:1806-1829), explicitly so nvcomp does not stage
D2H inside the measured window and put an explored sample on a different
scale from the primary.

Extra H→D: NO.
Extra D→H: YES, one, and it is deliberate — the winner's payload is pulled
back with DeviceAwareMemcpy AFTER the measurement (:1871-1880) so the copy
never lands inside the timed window. Native instead does a D2D into the
caller's output buffer. CLIO needs the bytes on the host because it re-stores
through PutBlob; native's caller already owns the destination.

CPU fallback:
Host-resident chunk → alt_output is a host std::vector and StageInputIfNeeded
may stage. Same residency dependency as every other phase → Phase 20/21.

Tests executed:
None. Exploration is off by default in both trees and no test in the suite
exercises it. **That is itself a finding** — the whole exploration path,
including the adoption block that can REPLACE a stored blob, is untested on
the CLIO side. Recorded for Phase 24.

Artifacts generated:
This checkpoint; traceability row for Exploration; the Phase 2 deferral of
gpucompress_compress.cpp:753-1173 is now discharged.

Remaining issues:
- D8-1's falsification experiment (rescore with CUDA-event times) → Phase 25.
- No test coverage for exploration on either the winner-adoption or the
  training-sample path → Phase 24.
- SGD phase 1 (:702-731) and phase 2 were read only as far as they bear on
  exploration triggering; the weight-update arithmetic itself is Phase 13.

Next phase:
Phase 9 — Native ranking
==================================================
```

### Phase 9 — Ranking

```
==================================================
PHASE CHECKPOINT
==================================================

Phase:
9 — Native ranking, and the port compared against it

Status:
COMPLETE — PASS, with one precision difference that is measured and bounded

Native implementation:
FUSED into inference — there is no standalone native ranking kernel. A
__device__ helper (nn_gpu.cu:109-249) computes rank_val per config, and the
reduction lives inside nnInferenceKernel (:333-390).

Native ground truth — scoring (nn_gpu.cu:207-239), in order:
  inverse-transform 8 outputs: comp_time/decomp_time/ratio and
    rmse/max_error/mae via expm1f(raw*y_std + y_mean); psnr linear;
    ssim = 1 - exp(-max(0, ssim_nlog))
  sanity clamps (:219-226): ct,dt -> [1e-6, 1e6]; ratio -> [0.1, 1e5];
    psnr -> [0, 120]; rmse/max_error/mae -> [0, 1e6]; ssim -> [0, 1]
  POLICY clamps (:229-231): ct = max(1, ct), dt = max(1, dt),
    ratio = min(100, ratio)
  cost = w0*ct + w1*dt + w2*(data_size/(ratio*bw));  rank_val = -cost
  masks, IN THIS ORDER (:238-239):
    if (quant == 1 && error_bound <= 0.0) rank_val = -INFINITY
    if (min_psnr > 0 && psnr < min_psnr)  rank_val = -INFINITY
  ALL IN FLOAT.

Native ground truth — ordering (:343-390):
  top-K requested  -> parallel bitonic sort over NN_NUM_CONFIGS == 32 == warp
    size, using __shfl_xor_sync, STRICT comparators (< and >) on rank_val
    alone. Equal keys never swap, so which index survives a tie is an
    EMERGENT property of the fixed network, not a stated rule.
  top-K not requested -> tree reduction 32->1 with `if (s_vals[tid+s] >
    s_vals[tid])` — strict >, so on a tie the LOWER index wins.

CLIO implementation:
STANDALONE RankKernel (neuropress_nn_gpu_kernels.cu:549-...). This is the
structural difference Phase 1 flagged: native fuses ranking into inference,
CLIO splits it out.

CLIO behavior — what MATCHES:
  - ct = fmax(1.0, ct_in) — same policy floor.
  - ratio = fmax(0.1, fmin(100.0, ratio_in)). Native applies
    fmaxf(0.1, fminf(1e5, r)) then fminf(100, r); the composition is a clamp
    to [0.1, 100], identical to CLIO's single expression.
  - score = -(w_ct*ct + w_dt*dt + w_io*io) with io = data_size/(ratio*bw) —
    same cost, same negation convention.
  - both masks present and applied IN NATIVE'S ORDER, with the port's comment
    citing nn_gpu.cu:238-239.
  - same bitonic network shape (__shfl_xor_sync, same k/j loop bounds, same
    is_lower / ascending predicates).
  - **both sides load the same weights file** — the test reports "Both sides
    loaded .../model.nnwt", 13576 parameters, 53.3 KB. (Early Phase 14
    evidence; not yet a Phase 14 verdict.)

CLIO behavior — what DIFFERS:
 D9-1  **Score is computed in DOUBLE; native computes in FLOAT.** Inputs are
       float on both sides. Consequence: two configs whose costs round to
       exactly equal in float (a tie native sees) can remain distinguishable
       in double, and vice versa — so the two can in principle disagree about
       whether a tie exists at all.
       MEASURED, not argued: over 190 chunks x 4 error bounds under the real
       model, ties genuinely occur (14 at eb=1e-6, 12 at 1e-3, 6 at 1e-2,
       0 at eb=0) and **0 of them are order-discriminating** — the tie never
       changed which action was picked. A third phase constructs ties
       deliberately and finds 5 order-discriminating cases out of 32 real
       ties over 760 cases (widest tie 4 configs); all pass.
       Verdict: real, bounded, and not observed to change a selection.
 D9-2  CLIO makes the tie-break an EXPLICIT total order —
       `(a > b) || (a == b && a_key < b_key)`, score descending then slot
       ascending — where native relies on its network never swapping equal
       keys. The port's comment states this deliberately reproduces
       "first enumerated wins" without depending on that emergent property.
       Strictly more robust; same outcome on every tested case.
 D9-3  CLIO uses a COMPOSITE sort key, `action_id * kMaxCandidates + tid`,
       guaranteeing uniqueness. Rationale given in the port: a bitonic sort
       on non-unique keys is not a permutation and can silently drop and
       duplicate candidates — reachable if a caller passes algorithms outside
       the trained eight, which collide under the fallback id mapping. Native
       has no such guard. This is a port HARDENING with no native counterpart,
       not a divergence in output.
 D9-4  CLIO substitutes ct for dt when dt_raw <= 0
       (`dt = (dt_raw > 0.0) ? fmax(1.0, dt_raw) : ct`); native floors dt at
       1.0 unconditionally. Unreachable when dt_in is the network's clamped
       output, which is always > 0 after native's [1e-6, 1e6] sanity clamp.
       Reachable only in CLIO's split path if a caller supplies dt = 0.

Byte comparison:
NOT APPLICABLE — ranking emits an ordering, not bytes. The correct criterion
is selection agreement, measured two ways:
  ctp_neuropress_tiebreak_parity — 1530 checks, 0 failures.
  ctp_neuropress_dataset_parity  — 0 of 256 chunks chose a different config
    across 4 error bounds (Phase 5/7 evidence).

GPU execution:
PASS. RankKernel is a real __global__; native's equivalent runs inside
nnInferenceKernel. Note this REFUTES the earlier audit's NN-3/NN-4 as filed
(cost model, argmax and PSNR mask claimed host-side) — the audit's own review
already flagged that RankKernel existed in the working tree but not in git
HEAD and was missing from its kernel inventory. It is now in HEAD
(fa485166) and it does all three on the GPU.

Extra H→D / Extra D→H: NO.

CPU fallback:
Not examined this phase — ranking runs wherever inference runs, so it inherits
the Phase 11 answer.

Tests executed:
ctest -R ctp_neuropress_tiebreak_parity → PASS, 1530 checks / 0 failures,
0.58 s. Both sides driven through the real Rank() with the real .nnwt.

Artifacts generated:
This checkpoint; traceability rows for Ranking.

Remaining issues:
- D9-1 is bounded by measurement, not by proof. A float/double disagreement
  that changes a selection has not been observed but is not impossible; the
  cheapest additional evidence would be re-running the dataset parity sweep
  with a wider chunk population. Recorded for Phase 24.
- The reduction path native takes when top-K is NOT requested (tree
  reduction, :380-390) has no separate CLIO counterpart, since CLIO always
  produces a full order. Not a gap — a superset.

Next phase:
Phase 10 — Native NN architecture
==================================================
```

### Phases 10-14 — The neural network

These five phases share one evidence base (`ctp_neuropress_parity`,
`ctp_neuropress_flow_parity`, `ctp_neuropress_device_path_parity`), so the
measured results are stated once here and referenced by each checkpoint.

**Measured this session, all three tests linking upstream's real `nn_gpu.cu`
and loading the real `.nnwt` on both sides:**

| Test | Checks | Failures | Worst deviation observed |
|---|---|---|---|
| `ctp_neuropress_parity` | 946 | 0 | **predictions: max rel err = 0**; weights: max abs err 2.98e-08 |
| `ctp_neuropress_flow_parity` | 591 | 0 | weights: max abs err 7.45e-09 |
| `ctp_neuropress_device_path_parity` | 251 | 0 | — (5/5 ties to lower action; 12 tied adjacent pairs seen) |

```
==================================================
PHASE CHECKPOINT
==================================================

Phase:
10 — Native NN architecture

Status:
COMPLETE — PASS

Native implementation:
src/nn/nn_weights.h (137 lines, read in full). Topology 8 -> 64 -> 64 -> 64
-> 64 -> 8: NN_INPUT_DIM 8, NN_HIDDEN_DIM 64, NN_OUTPUT_DIM 8, NN_NUM_LAYERS 5
(4 hidden + 1 output linear), NN_NUM_CONFIGS 32, NN_MAX_SGD_SAMPLES 8.
Parameter count NN_SGD_GRAD_REGION = 512+64 + 4096+64 + 4096+64 + 4096+64 +
512+8 = **13576**. SGD buffer is 3 regions (accumulator + workspace + EMA).
struct NNWeightsGPU (:91-135) additionally carries x_means/x_stds,
y_means/y_stds, x_mins/x_maxs, log_var[8], prev_grad_dot, sgd_call_count.

CLIO implementation:
struct NeuroPressGpuWeights (neuropress_nn_gpu_kernels.cu:66-93) with
kInputDim 8, kHiddenDim 64, kOutputDim 8, kMaxSamples 8, and kParamCount
computed identically (:44-52) to **13576**, guarded by
`static_assert(kOffB5 + kOutputDim == kParamCount)`. Weights are one flat
`params[kParamCount]` buffer in native's order w1,b1,w2,b2,w3,b3,w4,b4,w5,b5.

Byte comparison:
PASS — the loader reports "Loaded 13576 parameters ... (53.3 KB on GPU)" and
phase 0 of two parity tests asserts post-load weight parity between the trees.

First divergence:
 D10-1 CLIO omits x_mins / x_maxs (the v2+ OOD-detection bounds).
       **NOT A GAP** — verified dead upstream: a tree-wide search finds them
       only being read from the file and defaulted to +/-1e30 at
       nn_gpu.cu:1733-1749. No kernel consults them.
 D10-2 CLIO omits prev_grad_dot (documented upstream as anti-flip damping).
       **NOT A GAP** — also dead: it is assigned 0.0f once at load
       (nn_gpu.cu:1760) and never read or updated anywhere. The
       nn_reinforce.cpp that nn_weights.h's header comment names as its
       consumer **does not exist in this tree**.
       This is the third dead-upstream field the investigation has cleared
       (after vmin/vmax and heuristic_select_action) — a recurring pattern
       worth remembering before filing "the port is missing X".
 D10-3 CLIO adds persistent Train() scratch (act_*, d5_*, combined, out_grad,
       dz4_all) inside the weights struct. Never serialized; avoids a
       per-call cudaMalloc. No native counterpart, no output effect.

Next phase: 11
==================================================

==================================================
PHASE CHECKPOINT
==================================================

Phase:
11 — Native NN inference

Status:
COMPLETE — PASS, BIT-IDENTICAL PREDICTIONS

Native ground truth (nn_gpu.cu:133-215, read in full):
  input_raw[8] = [algo_idx, quant, shuffle,
                  (quant == 0) ? **1e-7f** : eb_enc,     <- sentinel, not 0.0
                  ds_enc, entropy, mad, deriv]
    algo_idx = tid % 8, quant = (tid/8) % 2, shuffle = (tid/16) % 2
    The 1e-7 lossless sentinel is required because training used it; passing
    raw 0.0 would be off-distribution (:138-139).
  normalize: (raw - x_means[i]) / max(x_stds[i], 1e-8f)
  4 hidden layers, ReLU, row-major weights indexed w[j*IN + i], each unit
    accumulated SEQUENTIALLY from its bias
  output layer linear, 8 outputs
  inverse transforms (:207-215): expm1f for comp_time/decomp_time/ratio and
    rmse/max_error/mae; psnr linear; ssim = 1 - exp(-max(0, ssim_nlog))
  ALL IN FLOAT.

CLIO implementation:
InferKernel (:415-438) — one block per candidate, normalization done by the
first 8 threads into shared memory, then NeuroPressForwardShared.
InferKernelDeviceStats (:453-...) is the counterpart to native's
nnFusedInferenceKernel: it reads the three data features straight from device
memory and builds each config's input vector in-kernel, so chunk statistics
never reach the host. The five host-known inputs arrive as cand_desc plus a
scalar chunk size, matching FeaturesTo8Input's order — this resolves the
Phase 1 "unconfirmed pairing" for both kernels.

Byte comparison:
**PASS — predictions max rel err = 0.** Not "within tolerance": exactly zero
across phase 1 (7 chunk profiles spanning near-zero to max entropy, plus tiny
and large chunks) and phase 1b (4 error bounds x 7 chunks, 24 of 28 cases
selecting a quantize config). Native winner and CLIO agree on the action AND
on ratio / comp_time / decomp_time to the last printed digit.
That CLIO's shared-memory parallel forward pass reproduces native's
sequential per-thread accumulation bit-for-bit means the port preserved the
accumulation ORDER, not merely the formula.

GPU execution:
PASS. device_path_parity additionally asserts "In-kernel masking" and
"No silent fallback to the host" — 251 checks, 0 failures.

First divergence:
None.

Next phase: 12
==================================================

==================================================
PHASE CHECKPOINT
==================================================

Phase:
12 — Native inference + exploration

Status:
COMPLETE — covered by phases 8, 9 and 11; no separate divergence

The combined behavior decomposes into inference (Phase 11, bit-identical),
ranking/tie-break (Phase 9, PASS) and the exploration loop (Phase 8,
DIFFERENCE FOUND D8-1 on the timing instrument). device_path_parity's tie
rule section confirms the composed path: 5 tied pairs on a reversed list, all
5 resolved to the lower action, and the ranking saw 12 tied adjacent pairs
overall.
No divergence exists in composition that is not already recorded in 8 or 9.

Next phase: 13
==================================================

==================================================
PHASE CHECKPOINT
==================================================

Phase:
13 — Native runtime learning / SGD

Status:
COMPLETE — PASS (agreement to ~1 float ULP)

Native implementation:
nnSGDKernel and nnBatchedDecompSGDKernel (nn_gpu.cu). Weight step is
`w -= step * EMA[...]` (:1523-1528) — the EMA gradient, not the raw one.
State is SPLIT: weights and sgd_call_count are global (:64, :1413); the EMA
gradient buffer belongs to a CompContext (gpucompress_pool.cpp:121-123,
zeroed at creation).

CLIO implementation:
SGDKernel (:1035) and DecompHeadSGDKernel (:662) — the latter added by commit
a5e11062 and the direct refutation of audit finding NN-1 ("no kernel exists;
scalar host C++"). CLIO reproduces the state split exactly: shared weights and
call count, separate EMA, with the EMA scoped per-flow rather than per
CompContext. The port documents this scope as "an approximation, not a match".

Byte comparison:
PASS — weights diffed after EVERY step, not just at the end:
  phase 2   7 sequential online-SGD steps
  phase 2b 11 steps with ALTERNATING gradient direction (over/under)
  phase 2c 6 combinations of error bound x quant flag, 7 steps each
  phase 3  deferred decomp-head batch of 7
  phase 3b decomp head with REPLAYED GROWING batches (2, 4, 7) — the O(n^2)
           re-sweep behavior Phase 6 flagged, tested directly
  phase 3c decomp head at 2 error bounds x quant flag
Worst weight deviation across all of it: **2.98e-08 absolute** (sgd-eb 1e-6
q1 step 4, w1) — one float ULP at that magnitude. Native grad_norms are
printed per step and are in the expected 1e-3 range for online SGD and
9-55 for the decomp head.

The EMA-scope approximation was tested SEPARATELY and passes
(ctp_neuropress_flow_parity, 591 checks):
  - 2 flows interleaved (native: 2 CompContexts, CLIO: 2 threads), 16 steps
  - reload + replay reproduces phase 1 exactly on BOTH sides, confirming the
    EMA reset reaches both flows
  - 1-flow vs 2-flow diverge by max|diff| = 0.00198627 on BOTH sides —
    **the same number** — so the flows are genuinely distinct and CLIO's
    per-flow scoping reproduces native's per-context scoping rather than
    accidentally collapsing to a single shared EMA
  Worst weight deviation: 7.45e-09.

First divergence:
None numerically. The EMA scope difference is real in principle but produces
identical divergence behavior under the interleaving test above.

Next phase: 14
==================================================

==================================================
PHASE CHECKPOINT
==================================================

Phase:
14 — Native .nnwt behavior

Status:
COMPLETE — PASS, FILE IS BYTE-IDENTICAL

Native: /home/cc/NeuroPress/neural_net/weights/model.nnwt
CLIO:   /home/cc/clio-core/context-transport-primitives/src/compress/model/
        weights/model.nnwt

md5 both: **df52a926af026fc617e172dcc990a395**

The requirement ".nnwt unchanged" is therefore satisfied by direct file
comparison, not inference. Both trees load it and report the same
"13576 parameters ... 53.3 KB on GPU", and phase 0 of two parity tests
asserts post-load weight parity between them.

Online-learning state (log_var, EMA gradient, sgd_call_count) is never
serialized to .nnwt on either side, so runtime learning cannot mutate the
file — confirmed in the port's own struct comment (:73) and consistent with
native's loader, which sets prev_grad_dot and the bounds defaults at load
rather than reading learned state.

First divergence:
None.

Remaining issues for 10-14:
- All five phases rest on the SAME three test binaries. They link upstream's
  real nn_gpu.cu, so this is genuine cross-tree comparison rather than
  self-consistency — but a defect in the shared harness would hide a real
  divergence in all five at once. Recorded as a known limitation for Phase 24.
- Inference reachability on the shipped CLIO path (as opposed to in the test
  harness) is NOT established here; that is the Phase 20/21 residency
  question, and the audit's claim that InferKernel is the ONLY CLIO kernel
  that launches in the shipped dynamic test is still OPEN.

Next phase:
Phase 15 — Native lossless compression / decompression
==================================================
```

### Phase 15 — Lossless compression / decompression

```
==================================================
PHASE CHECKPOINT
==================================================

Phase:
15 — Native lossless compression/decompression, and the port compared

Status:
COMPLETE — codec config PASS; **blob container format DIVERGES BY DESIGN**

Native codec configuration (src/compression/compression_factory.cpp:75-145):
  createCompressionManager(algo, chunk_size, stream, ...) with
  chunk_size = DEFAULT_CHUNK_SIZE = 1 << 16 = 65536 at every call site
  (gpucompress_pool.cpp:233, :265; gpucompress_compress.cpp:861).
  LZ4, Snappy, Deflate, Gdeflate, Zstd, ANS -> nvcompBatched*CompressDefaultOpts.
  CASCADED  -> defaults with opts.type = NVCOMP_TYPE_CHAR (:122-126)
  BITCOMP   -> defaults with opts.data_type = NVCOMP_TYPE_LONGLONG,
               opts.algorithm = 0 (:129-134)
  CUSZ/NDZIP/CUSZP throw — they are external, dispatched separately (:137-141).

CLIO codec configuration (ctp/include/clio_ctp/compress/nvcomp.h):
  kChunkSize = 1 << 16 (:315) — **identical**.
  Same six default-opts managers (:461-481).
  CASCADED (:486-501) and BITCOMP (:503-517) set the SAME two overrides, and
  the port documents exactly what it is deviating from: the Cascaded default
  is {4096, NVCOMP_TYPE_INT, 2, 1, 1, {0}}, so leaving type alone would run
  RLE+delta+bitpack over 32-bit words instead of bytes and produce a
  different bitstream than the model was trained against — plus nvcomp
  requires each chunk to be a multiple of the element size, and the manager's
  final chunk is input_size % 64 KiB.
Verdict: **PASS.** Every parameter that determines an nvcomp bitstream
matches, which is the necessary condition for byte-identical payloads.

**BLOB CONTAINER FORMAT — THE DIVERGENCE**

The two systems do NOT share an on-disk format. Compared field by field:

|  | Native `CompressionHeader` | CLIO `CompressionHeader` |
|---|---|---|
| location | `src/compression/compression_header.h:53` | `cte/compressor/src/compressor_runtime.cc:203` |
| magic | `0x43555047` "GPUC" | `0x43544543` "CTEC" |
| size | **fixed 64 B** | **24 B core**, `static_assert`-ed, + optional 32 B |
| algorithm id | packed into `quant_flags` bits 9-12 | `compress_lib_` (a Clio wire id) |
| shuffle | `shuffle_element_size`, own uint32 | packed into `compress_preset_ >> 8` |
| compressed size | `uint64` | `uint32` (reuses existing padding; 0 = not recorded) |
| quant params | 4 doubles ALWAYS reserved in the 64 B | `QuantHeaderExtension` (32 B) appended ONLY when the quantize bit is set |
| version | `version` field, 1..2 accepted | version-gated extension, documented at :189-191 |

Consequence, stated plainly: **a native blob cannot be read by CLIO and a
CLIO blob cannot be read by native.** The magic alone rejects it, and the
payload offset differs even if it did not.

This is a DESIGN DIVERGENCE, not a defect: CLIO is a storage system with a
pre-existing blob format that predates this integration, and the port reuses
it rather than nesting native's container inside it. Native's own
`compressed_size` semantics ARE carried across — CLIO's field comment cites
`compression_header.h:64` and native's bounds check
(`gpucompress_compress.cpp:1199-1202`) as the reason the field exists at all,
replacing an earlier CLIO scheme that over-read past the stream.

What this does and does not invalidate:
  - It does NOT affect payload parity. Everything proven byte-identical in
    phases 3, 4 and above concerns the bytes handed to and returned from the
    codec, which are unchanged by the container.
  - It DOES mean Phase 18 (native <-> CLIO cross-compatibility) cannot be a
    PASS in the read-each-other's-blobs sense. Phase 18 must be re-scoped to
    "same input + same selection -> same payload bytes", and the container
    difference recorded as the reason. **Recorded now so Phase 18 does not
    discover it late and mark an UNKNOWN as a PASS.**

First divergence:
 D15-1 **Blob container formats differ entirely** (magic, size, field layout,
       quant-param placement). By design. Severity: HIGH for
       interoperability, NONE for correctness within either system.

Byte comparison:
Codec configuration: PASS by inspection of every manager construction on both
sides. Payload byte-identity has NOT been measured directly this phase — it
requires driving both stacks over the same input and diffing the compressed
stream, which is Phase 18/23 work now that the container difference is known
to make a naive blob-level diff meaningless.

GPU execution:
PASS. The earlier audit's most likely-sounding hypothesis — that a ranked
"LZ4" silently resolves to a CPU codec — was already tested and found FALSE
(all 8 trained algorithms map to GPU wire ids). Nothing this phase contradicts
that.

Remaining issues:
- Payload-level byte comparison outstanding -> Phase 18/23.
- External codecs (cuSZ/nDzip/cuSZp) deliberately not covered here; they are
  Phase 16, and the audit's absolute-vs-relative error-bound claim is still
  OPEN.

Next phase:
Phase 16 — Native static quantized compression / decompression (external codecs)
==================================================
```

### Phase 16 — Static quantized / external codecs (cuSZ, nDzip, cuSZp)

```
==================================================
PHASE CHECKPOINT
==================================================

Phase:
16 — External lossy codecs; resolves the audit's open cuSZ error-bound claim

Status:
COMPLETE — DIFFERENCE FOUND (D16-1), CONFIRMED AS CODE, reachability bounded.
My own independent measurement is INCONCLUSIVE and is reported as such.

Native implementation:
src/compression/external_compressors.cu. cuszCompress (:100-180):
  psz_pipeline{DEFAULT_PREDICTOR, DEFAULT_HISTOGRAM, DEFAULT_CODEC, NULL_CODEC}
  psz_rc2 rc; **rc.mode = Abs** (:118); rc.eb = error_bound;
  rc.radius = DEFAULT_RADIUS
  Payload layout: [psz_header @ 0][archive @ kCuszArchiveOffset].
cuszDecompress (:182-225): reads the header D2H, builds a FRESH manager via
  psz_create_resource_manager_from_header, and **zero-fills d_output first**
  because "cuSZ's Lorenzo decode uses the output buffer as the outlier-scatter
  base and ADDS reconstructed values into it, but only zero-fills it for the
  Spline predictor" (:206-210).
These codecs bypass the action machinery entirely and never carry
GPUCompress shuffle/quantization preprocessing (gpucompress_compress.cpp:1207-1209).

CLIO implementation:
ctp/include/clio_ctp/compress/cusz.h. `explicit Cusz(double eb = 1e-3,
psz_mode mode = Rel)` (:87) — **the default is Rel**. Used at :125 as
`psz_rc2 rc = {mode_, eb_, kRadius}`.

REACHABILITY (Protocol §15 — determined, not assumed):
 - Build: CLIO_CTP_ENABLE_CUSZ / CUSZP / NDZIP are all **ON** in this build
   (CMakeCache.txt:257, :260, :284; CTP_ENABLE_CUSZ=1 in compile_commands).
 - Construction: the ONLY site that builds a Cusz in the whole tree is
   CompressionFactory::MakeCusz (compress_factory.h:429-444), and it calls
   `std::make_unique<Cusz>(eb)` with **one argument** — so `mode_` takes the
   Rel default. Nothing anywhere overrides it.
 - NeuroPress path: **cuSZ is NOT reachable from NeuroPress selection.**
   `grep -n "cusz|ndzip"` over neuropress_bridge.cc returns **nothing** — the
   trained action space is the 8 nvcomp algorithms only, matching native,
   which also dispatches these three outside the action machinery.
   So a NeuroPress-selected chunk can never be compressed by cuSZ.
   cuSZ is reachable only via CLIO's own legacy selection or an explicit
   compress_lib_.

First divergence:
 D16-1 **Native cuSZ uses ABSOLUTE error-bound mode; CLIO uses RELATIVE.**
       Status: **CONFIRMED AS CODE** (native external_compressors.cu:118 vs
       cusz.h:87 + the sole construction site).
       Corroborating documentary evidence from CLIO's own test suite:
       test_compress.cc:554-589 is titled "cusz (device pointers, BALANCED)
       round-trips within eb", its comment reads "Lossy within the BALANCED
       **relative** error bound (1e-3) on a [-100,100] signal -> generous
       absolute slack", and it asserts `max_err < 1.0` — a thousand times
       looser than the nominal 1e-3. The test was written to accommodate
       relative-mode behavior, so it documents the divergence rather than
       catching it. Verified passing this session (ctest -R ^ctp_compress$,
       3/3 in 1.14 s).
       Internal inconsistency worth noting: MakeCuszp's comment immediately
       below states cuSZp's "presets map to ABSOLUTE error bounds"
       (compress_factory.h:454-457). So within one factory, cuSZp is
       absolute and cuSZ is silently relative.
       Severity: HIGH for anyone relying on cuSZ's advertised bound;
       **NOT a NeuroPress-parity defect**, because NeuroPress cannot select
       cuSZ on either side. This is the correction to the audit's framing —
       it filed the finding without establishing that reachability.

MY OWN MEASUREMENT — INCONCLUSIVE (Protocol §7: not recorded as a PASS)
I wrote a standalone probe against the installed libcusz
(scratchpad/cusz_mode_probe.cu) to measure achieved max|err| under Abs vs
Rel at eb=1e-3, mirroring native's manager lifecycle including the mandatory
output zero-fill. It compresses successfully (sensible ratios: 1.98x Abs /
3.18x Rel on smooth data) but **decompression returns garbage in BOTH modes**
(max|err| ~1.1e3 on amplitude-100 data), so the probe does not reproduce a
valid round trip and its numbers measure a bug in my probe, not the codec.
An earlier attempt on uniform random +/-100 data additionally hit the
quantizer-radius overflow native documents at :142-144.
**I am therefore NOT claiming an independently measured error magnitude.**
The prior audit reports 0.1-0.2 at eb=1e-3 on +/-100 data, reproduced by two
agents; that figure is consistent with relative mode over a range of 200 but
is inherited, not re-verified here.
Next action to close this properly: drive the port's own Cusz class through
test_compress.cc's existing harness with the assertion tightened to the
nominal bound, and/or construct Cusz with Abs and re-run — a far cheaper
experiment than a from-scratch probe, and it exercises the real code path.
Recorded for Phase 26.

nDzip / cuSZp:
NOT INVESTIGATED this phase. nDzip is lossless and takes no error bound.
cuSZp's factory comment claims absolute presets; not verified. Both carried
forward.

Byte comparison:
NOT APPLICABLE — lossy codecs; the criterion is the achieved error bound,
which is what D16-1 concerns.

Remaining issues:
- D16-1's magnitude unmeasured by me → Phase 26 experiment named above.
- cuSZp mode claim unverified; nDzip untouched.

Next phase:
Phase 17 — Native read / decompression path
==================================================
```

### Phase 17 — Read / decompression path

```
==================================================
PHASE CHECKPOINT
==================================================

Phase:
17 — Native read/decompression path, and the port compared

Status:
COMPLETE — PASS on ordering and failure semantics; one HYPOTHESIS raised

Native ground truth (established Phase 2, gpucompress_decompress_gpu
:1175-1312): 64 B header D→H, then **codec → unshuffle → dequantize → D2D
into the caller's buffer**, each stage into its own buffer. Quantization
metadata is reconstructed purely from header fields.

CLIO implementation:
Runtime::Decompress (compressor_runtime.cc:2645-...), plus
Runtime::DecompressStored (:3408-...) for the interposer path.

CLIO behavior — MATCHES:
  - Same inversion order, and the port cites upstream's lines for it
    (:2856-2858, :2968-2971).
  - Correct handling of the quantized narrowing, which is the subtle part: a
    quantized blob decompresses to int8/16/32, NOT to the original bytes. So
    the codec writes into a scratch sized `quant_elems *
    PrecisionToBytes(precision)` (:2859-2886), the unshuffle inverts on that
    scratch, and dequantize is what finally fills the caller's float32 buffer
    and restores `decompressed_size = original_size` (:2984).
  - Device unshuffle into device scratch then copy back (:2922-2949), avoiding
    the D2H/H2D round trip that would defeat a GPU consumer.
  - **Fails loudly instead of degrading.** Both unshuffle (:2937-2942,
    :2958-2964) and dequantize (:2976-2982) set success = false with an
    explicit log rather than returning shuffled or still-quantized bytes.
    The reasoning given is exactly right: those bytes are the correct LENGTH,
    so returning them would be silent corruption.
  - DecompressStored correctly uses the HOST Dequantize<float> because "dst is
    a host buffer on this path (the interposer stages through SHM)"
    (:3515-3518, :3534-3545) — the dispatch is deliberate, not accidental.

AUDIT HYPOTHESIS VOL-3 ("after write→close→reopen, H5Dread decompresses
nothing and serves the uncompressed HDF5 copy"):
NOT RE-TESTED — it is an HDF5 VOL-level claim and this phase covers the
compressor's read path. But the port has since added an instrument aimed
squarely at it: CLIO_NEUROPRESS_DECOMPRESS_TRACE (:2892-2914) prints from
the codec call itself, with the rationale that "a caller only ever sees a
return code, and a path that served the bytes from somewhere else returns 0
just as happily -- which is precisely what the HDF5 VOL does when a dropped
cache tag sends the read to the uncompressed copy."
That instrument is the cheapest way to settle VOL-3. **Recorded as the
concrete experiment for Phase 23**, not resolved here.

First divergence:
 D17-1 **HYPOTHESIS — requires verification, not a confirmed defect.**
       Runtime::Decompress dispatches unshuffle on pointer kind
       (`IsDevicePointer(codec_dst)`, :2923 device / :2950 host) but calls
       **DequantizeDevice unconditionally** (:2973). No host counterpart is
       consulted on this path.
       Partial analysis, stated so it is not overclaimed:
        - `codec_dst` is NOT the exposure. When `stored_quant` is set,
          codec_dst is the quant scratch from AllocateAndRegisterGpuBackend
          (:2872-2884), i.e. device memory BY CONSTRUCTION. That also means
          the host unshuffle branch is unreachable for quantized blobs, so
          the two dispatches are self-consistent.
        - The open question is the DESTINATION, `output_fullptr.ptr_`, which
          the dequantize kernel writes. If that can ever be host memory while
          codec_dst is device, the kernel writes a host address and
          DequantizeDevice's only guards (null / zero-length / scale <= 0,
          :851-853) will not catch it.
        - Failure would be loud at the CLIO level — DequantizeDevice returns
          `cudaStreamSynchronize(...) == cudaSuccess`, so it returns false and
          success is set false. The concern is the audit's PRE-14 point that
          cudaErrorIllegalAddress is STICKY and can poison the CUDA context
          process-wide.
       Relationship to Phase 4: this does NOT overturn Phase 4's verdict that
       PRE-14 is not a parity divergence — native's dequantize has the same
       device-pointer contract. What Phase 17 adds is that CLIO's own
       architecture can produce a caller for that contract in a way native's
       cannot, because native has no host-resident read path at all.
       Next action: determine whether output_fullptr.ptr_ can be host-resident
       on this path → Phase 19/20.

Byte comparison:
NOT PERFORMED this phase. Read-path payload equivalence depends on the same
codec + preprocessing already proven byte-identical in phases 3, 4 and 15, and
a blob-level diff is meaningless across D15-1's container difference. The real
read-path test is a round trip, which is Phase 23.

GPU execution:
PASS for the device path — unshuffle and dequantize are both device kernels
with device scratch.

CPU fallback:
Present and correct for the non-quantized host case (:2950-2965) and for the
whole interposer path. Same residency dependency as every prior phase.

Remaining issues:
- VOL-3 → run with CLIO_NEUROPRESS_DECOMPRESS_TRACE set (Phase 23).
- D17-1 → resolve output_fullptr residency (Phase 19/20).
- VOL-1, VOL-2, VOL-5 untouched — all HDF5 VOL adapter claims, Phase 20/21.

Next phase:
Phase 18 — Native ↔ CLIO cross-compatibility (RE-SCOPED, see Phase 15/D15-1:
blob containers are disjoint, so this becomes "same input + same selection ->
same payload bytes", not blob interchange)
==================================================
```

### Phases 20-22 — GPU residency, transfer audit, CPU fallback

**MEASURED THIS SESSION. This is the most consequential result in the
investigation and it changes the severity of several earlier findings.**

Method: `nsys profile --trace=cuda` over the shipped test exactly as ctest
invokes it — `bin/test_compressor_functional "[neuropress][693]"`
(ctest #205, `compressor_dynamic_neuropress`), at HEAD `fa485166`.
Artifacts: `scratchpad/np_prof.nsys-rep` / `.sqlite`.

```
==================================================
PHASE CHECKPOINT
==================================================

Phase:
20 / 21 / 22 — GPU residency, host/device transfer audit, CPU fallback

Status:
DIFFERENCE FOUND — **the audit's central empirical claim is REPRODUCED at
HEAD, and is broader than as filed**

RESULT — kernels actually launched on the shipped path:
  19 distinct kernels total.
  **Exactly ONE is CLIO-owned: `ctp::compress::model::gpu::InferKernel`**
  (8 instances, 206,657 ns total).
  Everything else is third-party codec internals: nvcomp LZ4
  (`lz4CompressBatchKernel`, 4 inst, 59.3% of GPU time), Zstd
  (`lz_compression_kernel` etc.), Bitcomp (`batch_encoder_kernel`), and
  nvcomp's CUB scan/compaction helpers.

  **NOT launched — verified absent from the profile:**
    StatsPass1Kernel, StatsPass2Kernel, StatsPass2DevKernel,
    EntropyFromHistKernel, FinalizeFeatureStatsKernel, MinMaxKernel,
    ShuffleKernel, UnshuffleKernel, QuantizeKernel, DequantizeKernel,
    InferKernelDeviceStats, **RankKernel**, SGDKernel, **DecompHeadSGDKernel**

What this establishes, in order of consequence:

 1. **The chunks on this path are HOST-RESIDENT.** `InferKernel` is the
    host-stats variant; `InferKernelDeviceStats` is the device-stats one
    (Phase 11). That the former runs and the latter does not means
    `ComputeCompressionFeatures` took its host branch
    (`data_stats_gpu.h:124`), which it only does for a host pointer.
    Every pointer-kind dispatch found in phases 3, 4, 5, 8 and 17 therefore
    resolves to its HOST side on this path.

 2. **D7-1 is REACHABLE, not hypothetical.** Phase 7 recorded that a
    host-resident chunk with NeuroPress configured falls through to CLIO's
    own legacy qtable/dense-NN heuristics, where native's host entry point is
    a hard-failing stub. Its severity was gated on "how often are chunks
    host-resident", answered here: **on the shipped test, always.**
    D7-1 is hereby upgraded from HIGH-with-unknown-reachability to
    **HIGH AND REACHED ON THE DEFAULT PATH**.

 3. **Two kernels this investigation verified as correct never run.**
    Phase 9 proved RankKernel matches native's scoring and tie-break; Phase 13
    proved DecompHeadSGDKernel matches native's decomp-head SGD to ~1 ULP.
    Neither launches here. Commits 283ed09c and a5e11062 moved that work onto
    the GPU and the kernels are faithful — but the shipped path does not reach
    them. Correct code on an unreached path is still unreached.
    Category: `unreachable-device-path`.

 4. **The prior audit's headline result is CONFIRMED at HEAD**, and was not
    invalidated by the intervening commits. Its list of non-launching kernels
    is reproduced exactly, plus RankKernel and DecompHeadSGDKernel which did
    not exist when it was written. STATS-5, ORCH-1, PRE-2 and PRE-5 all
    survive as locality findings.
    Crucially, phases 3-5 already established that **none of them can change
    a byte or a number** — the host shuffle is byte-identical and the host
    statistics agree to 5e-15. They are performance and locality findings
    only. That distinction was not available when the audit filed them.

TRANSFER AUDIT (Phase 21), same profile:
  H→D  21 copies, 257,793 ns total (52.1%), max 59,392 ns
  D→H  **56 copies**, 212,386 ns total (42.9%)
  memset 9
  Native's baseline for comparison (Phase 2, measured by reading): 64 B H→D
  per compressed chunk, 64 B D→H per decompressed chunk, plus a small
  inference readback. **56 D→H on a path native would service with a handful
  is the quantitative form of finding 1 above.**
  NOT yet attributed per-callsite — the profile gives counts and totals, not
  which source line issued each copy. Attribution needs a CUPTI callback or
  gdb breakpoints. **NOT DONE; do not read a per-callsite claim into these
  numbers.**

CPU FALLBACK (Phase 22):
  CONFIRMED PRESENT AND TAKEN on the shipped path, for statistics (host
  branch of ComputeCompressionFeatures) and, by the same pointer-kind
  dispatch, for shuffle and quantize when those actions are selected.
  Not a silent *correctness* degrade — phases 3/4/5 proved the host paths
  produce identical bytes and numbers. It IS a silent *locality* degrade, and
  for selection (D7-1) a silent *model substitution*.

SCOPE LIMIT — stated so this is not over-read:
This is ONE test. It shows that this shipped test's chunks are host-resident;
it does NOT show that CLIO can never reach its device path. The
`neuropress_gpu_direct` example (compressor/example/) exists precisely to
drive device pointers and was NOT profiled. The honest claim is:
**the default shipped functional test exercises the host path end to end, and
the device kernels that phases 3-13 verified are not exercised by it.**
Whether production HDF5-VOL traffic arrives device-resident is a separate
question, still OPEN.

FOLLOW-UP MEASUREMENT — **DONE, and it materially softens the above.**

Profiled `bin/neuropress_gpu_direct` (the device-pointer example) the same
way. Artifacts: `scratchpad/npgd.nsys-rep` / `.sqlite`.

**The complete CLIO device pipeline launches:**

| CLIO kernel | instances |
|---|---|
| `StatsPass1Kernel<float>` | 64 |
| `StatsPass2DevKernel<float>` | 64 |
| `EntropyFromHistKernel` | 64 |
| `FinalizeFeatureStatsKernel` | 64 |
| `InferKernelDeviceStats` | 64 |
| `RankKernel` | 64 |
| `ShuffleKernel<4>` | 51 |
| `UnshuffleKernel<4>` | 51 |

Device stats → device inference → device ranking → device shuffle → codec →
device unshuffle, all on the GPU, with nvcomp LZ4/Cascaded/Bitcomp doing the
codec work. Not present: Quantize/Dequantize (this run is lossless, so the
quantize half of the action space is masked to -INF — expected, see Phase 9)
and SGDKernel/DecompHeadSGDKernel (online learning is off by default on both
sides — expected, see Phase 8).

**CORRECTED CONCLUSION for phases 20-22**, replacing the reading above:

 - The device path is **NOT missing, NOT unreachable and NOT a silent
   degrade**. It exists, it is complete, and every kernel phases 3-13
   verified against native does launch when the input is device-resident.
 - Item 3 above is **WITHDRAWN as stated**. RankKernel and
   DecompHeadSGDKernel are not "correct code on an unreached path" —
   RankKernel launches 64 times here. DecompHeadSGDKernel still does not
   launch in either profile, but that is because online learning is off by
   default, which matches native.
 - What remains true, and is the accurate finding: **the default shipped
   functional test `compressor_dynamic_neuropress` drives HOST-resident
   chunks**, so on that test only `InferKernel` runs and the host paths for
   statistics, shuffle and selection are taken.
 - D7-1's reachability is therefore: **reached on the default functional
   test, not reached on the device-direct path.** Still HIGH — a
   host-resident chunk really does get selected by a different model — but
   it is a property of how a caller supplies memory, not an unconditional
   defect in the integration.
 - The prior audit's headline ("the complete list of Clio-owned CUDA kernels
   that launch is one") is **reproduced exactly on the test it profiled, and
   is NOT representative of the integration as a whole.** It profiled the
   host-resident test and generalized. That generalization is what this
   follow-up corrects.

TRANSFER COMPARISON between the two runs:

| | `compressor_dynamic_neuropress` (host) | `neuropress_gpu_direct` (device) |
|---|---|---|
| H→D | 21 | 197 |
| D→H | 56 | 897 |
| D→D | 0 | 179 |
| CLIO kernels | 1 | 8 distinct |

The device run does far MORE host traffic in absolute terms, but it also does
~16x the work (64 chunks with compress+decompress vs a handful). Neither
figure is normalized per chunk and neither is attributed per call site, so
**no efficiency claim is made from this table** — it is recorded as raw
evidence for Phase 25, which is where normalization and attribution belong.

Phases 18 and 19 remain NOT STARTED.
==================================================
```

### Phase 19 — CLIO CUDA IPC / device pointer

```
==================================================
PHASE CHECKPOINT
==================================================

Phase:
19 — CLIO CUDA IPC and device-pointer lifecycle

Status:
COMPLETE — **DIFFERENCE FOUND (D19-1): device scratch is leaked.**
Confirmed three independent ways.

The mechanism (code):
`IpcManager::FreeGpuBackend` (context-runtime/src/ipc_manager.cc:3947-3956)
does NOT free device memory. Its entire body is
`gpu_ipc_->UnregisterClientBackend(gpu_id, alloc_id)`, followed by a comment
stating the contract explicitly:
    "The actual ctp::GpuApi::Free relies on caller-tracked metadata — the
     host caller passes the base back (out_base from
     AllocateAndRegisterGpuBackend) and frees through the same API."
So the allocator hands the caller a base pointer and expects the caller to
free it.

The violation (code):
In the compressor: **8** `AllocateAndRegisterGpuBackend` call sites, **6**
`FreeGpuBackend` call sites, and **zero** `GpuApi::Free` calls anywhere under
`context-transfer-engine/compressor/src/`. No caller ever discharges the
contract. Every device scratch buffer the compressor allocates — quantization
scratch, shuffle scratch, exploration output slots, decompress unshuffle
scratch — is therefore leaked for the process lifetime.

The measurement (`nsys cuda_api_sum` on `neuropress_gpu_direct`, 64 chunks):
    cudaMalloc                    **379 calls**
    cudaFree                      **73 calls**
    cudaMallocFromPoolAsync        67  }  these two balance —
    cudaFreeAsync                  67  }  the async pool is fine
A ~306-allocation gap in a single 64-chunk run.

HONEST ATTRIBUTION LIMIT: not all 379 cudaMallocs originate from
AllocateAndRegisterGpuBackend — nvcomp managers, CUB temp storage and cuSZ
each allocate too, and some of those are freed by their own destructors. The
measurement establishes a large malloc/free imbalance; the CODE reading is
what establishes that the compressor's own scratch is never freed. I am NOT
claiming all 306 are attributable to FreeGpuBackend. Isolating that needs a
CUPTI callback keyed on the allocating call site — not done.

Relationship to native:
This is **not a parity divergence**. Native frees its scratch correctly and
symmetrically: `byte_shuffle_simple` cudaFrees on every failure path and the
caller frees the returned buffer when `owns_output` is true
(gpucompress_compress.cpp:616-617); the exploration loop's Phase 3 frees every
per-slot d_out / d_quant / d_shuf / range buffer
(gpucompress_compress.cpp:974-986); the decompress path frees all three
intermediates (:1304-1306). Native's discipline is visible and complete.
This is a defect CLIO introduced, in CLIO's own allocator contract.

Relationship to the prior audit:
This reproduces its "Found while verifying" item #1 at HEAD. That item was
outside the audit's GPU/host-locality frame and appears in no findings
document. The audit measured 475 cudaMalloc / 119 cudaFree and GPU memory
climbing 1653 → 2635 MiB; I measure 379 / 73 on a different run. Same shape,
independently reproduced.
It also means PRE-9 and CODEC-6, which both assumed this function frees and
costed it as a *synchronizing* free, were reasoning about a function that does
nothing of the kind.

D17-1 (from Phase 17) — partially advanced, NOT closed:
The question was whether `output_fullptr.ptr_` can be host-resident while
`codec_dst` is device, which would hand `DequantizeDevice` a host destination.
Not resolved here. What Phase 20's profiling adds is that BOTH residency
regimes genuinely occur in shipped binaries (host on the functional test,
device on gpu_direct), so the mixed case cannot be dismissed as unreachable on
architectural grounds. Still requires a targeted test.
Recorded as still OPEN.

CUDA IPC device pointer handling itself:
The device path demonstrably works — `AllocateAndRegisterGpuBackend` scratch
feeds ShuffleKernel, the codec and UnshuffleKernel with no correctness failure
across 64 chunks (Phase 20 profile, and `neuropress_gpu_direct` exits 0). No
IPC-specific divergence from native found; native has no IPC layer to diverge
from, since it takes raw device pointers from its caller.

Next phase:
Phase 18 — cross-compatibility (re-scoped)
==================================================
```

### Phase 18 — Native ↔ CLIO cross-compatibility (re-scoped)

```
==================================================
PHASE CHECKPOINT
==================================================

Phase:
18 — Native <-> CLIO cross-compatibility

Status:
**BLOCKED** — re-scoped by D15-1, and the available artifacts cannot settle
the re-scoped question. No verdict claimed.

Original scope, and why it is void:
"Can each side read the other's blobs?" Answered NO structurally in Phase 15
(D15-1): the containers are disjoint — magic "GPUC"/fixed 64 B versus
"CTEC"/24 B plus an optional 32 B extension, with different field layouts and
a different place for the algorithm id. The magic alone rejects it. This is a
design decision, not a defect, and no test can change it.

Re-scoped question:
"Given the SAME input and the SAME selection, do both stacks produce the same
payload bytes?" That is the meaningful form of cross-compatibility once the
containers are known to differ.

ATTEMPTED with existing artifacts — /tmp/neuropress_e2e_native.csv (256 rows,
written 00:52) and /tmp/neuropress_e2e_clio.csv (235 rows, 06:39). I joined
them on chunk index and compared (algo_idx, quantize, shuffle).

**The result is 89/235 selections identical — AND IT IS NOT A FINDING.**
I am recording the number only to explain why it must be discarded:

 1. The two CSVs are from **separate, uncontrolled runs hours apart**, not a
    paired comparison on identical input.
 2. **The native run had online learning ENABLED and SGD fired on 14 of its
    256 chunks.** Native's weights therefore mutated mid-run, so its selection
    for chunk N depends on the history of chunks 0..N-1. Nothing about the
    CLIO run reproduces that history.
 3. The row counts differ (256 vs 235) and the join keys are different columns
    (`chunk` vs `seq`) whose correspondence I did not verify.
 4. The two schemas are not the same shape — native records `action`/`ratio`
    with an `sgd_fired` flag; CLIO records `wire_lib`/`actual_ratio` with a
    checksum and no learning column.

A controlled measurement of exactly this question already exists and
disagrees: `ctp_neuropress_dataset_parity` drives both sides over **the same
device bytes** with **the same static weights and no learning**, and finds
**0 of 256 chunks selecting a different config** across 4 error bounds
(Phase 5/7 evidence, 7940 checks / 0 failures). That is the trustworthy
number. The 89/235 is an artifact of my comparison method, and treating it as
a divergence would be a methodology error, not a discovery.

WHAT REMAINS to close Phase 18 properly:
 - Payload-byte comparison has NOT been done at any phase. Phases 3, 4 and 15
   proved the inputs to and configuration of the codec are identical, which
   makes identical payloads very likely — but "likely" is a HYPOTHESIS, and
   Protocol §7 forbids recording it as a PASS.
 - The correct experiment: drive both stacks on one fixed buffer with learning
   OFF and a forced identical action, then diff the compressed payload
   (excluding each side's container header). `neuropress_native_replay.cc` in
   the parity directory appears built for this and was not examined.
 - Until then Phase 18 is BLOCKED, not PASS and not FAIL.

DEPENDENCY: needs learning disabled on both sides and a forced action, so it
should run alongside Phase 23's end-to-end work.

Next phase:
Phase 23 — End-to-end tests
==================================================
```

### Phase 23 — End-to-end tests

```
==================================================
PHASE CHECKPOINT
==================================================

Phase:
23 — End-to-end tests

Status:
COMPLETE — **lossless round trip PROVEN at 1 GiB scale**; VOL-3 CONFIRMED

Run (this session), deterministic configuration:
  bin/neuropress_e2e
  CLIO_NEUROPRESS_E2E_LEARNING=0   (learning off -> frozen weights)
  CLIO_NEUROPRESS_DECOMPRESS_TRACE=1
  CLIO_NEUROPRESS_SELECTION_LOG=scratchpad/e2e_nolearn.csv
  exit 0
  Dataset 1024 MiB = 268,435,456 floats, 256 chunks of 4 MiB, each
  independently selected by NeuroPress. Mode: inference only, exploration off.

RESULT — lossless round trip:
    Clio decompress of its own compressed blobs
      exact       : **256 / 256**
      differing   : 0 (0 bytes)
      unavailable : 0
    INTEGRITY VERIFIED: **0 of 1,073,741,824 bytes differ**
This is the strongest correctness evidence in the investigation: a full
gigabyte through GPU generation -> NeuroPress selection -> preprocessing ->
codec -> storage -> decompression -> byte comparison, with zero divergence.

Selection distribution over 512 log rows (256 chunks x 2 passes):
    nvcomp-ans        382
    nvcomp-cascaded   114
    nvcomp-lz4         16
    actual_ratio median 1.145, min 1.000, max 2.205
Modest ratios are expected — the generated data measures entropy ~7.17 bits
per byte, near-incompressible.

RECONCILIATION of traced decompressions against chunk count — done, because
the mismatch looked like exactly the class of problem the trace exists to
find, and it is not:
    traced decompress calls : 217 (217 DISTINCT blobs, all ok=1)
    chunks verified exact   : 256
    difference              : 39
    log rows with actual_ratio <= 1.0 : 78 = 39 chunks x 2 passes
So 39 chunks were stored RAW via the "compression not beneficial" branch,
needed no decompression, and trivially match. 217 + 39 = 256. Clean, benign,
and fully accounted for. No chunk was served from an unexpected path.

AUDIT FINDING VOL-3 — **CONFIRMED, and already known to the port authors.**
The e2e source states the mechanism outright
(hdf5_vol/example/neuropress_e2e.cc:282-290):
    "Going through H5Dread does not test this. The VOL writes the plain data
     to native HDF5 as well as staging compressed chunks into the tier, and
     on reopen its coherence check finds no stamp blob, drops the tag, and
     serves the read from the uncompressed HDF5 copy instead
     (clio_vol.cc:1126, :2409). The bytes match, but nothing decompresses."
The test therefore DELIBERATELY BYPASSES H5Dread and calls
AsyncDecompressExplicit — "the same call the VOL's cache-hit path uses" — to
exercise decompression at all.
Consequences, stated precisely:
  - Data is CORRECT after reopen. The bytes match; nothing is corrupted.
  - But compression delivers NO READ-PATH BENEFIT in the reopen scenario, and
    the dataset is stored TWICE (plain HDF5 + compressed blobs).
  - The measured H5Dread took 24,554 ms at 41.70 MiB/s against a write of
    6,381 ms at 160.49 MiB/s. NOT attributed — that read includes the
    uncompressed HDF5 service path and is not a like-for-like decompression
    measurement. No performance claim is made from it.
  - VOL-3's severity as filed ("H5Dread decompresses nothing") is accurate as
    a statement about execution, and benign as a statement about data.

PHASE 18 STILL NOT CLOSED by this run. The e2e produces CLIO's selections and
dumps the bytes it compressed (/tmp/neuropress_e2e_data.bin), which is half
the paired experiment. The other half — replaying those exact bytes through
native via neuropress_native_replay --inference-only — was NOT run: that
driver is compiled out-of-tree against a NeuroPress build and is not in
build/bin. Its own header comment independently confirms the Phase 18
reasoning: "Replaying 0,1,2,... against a run that learned in some other
order would compare two different experiments... Run with --inference-only
(and a Clio run with learning disabled) for the deterministic comparison."
The learning-off CLIO half now exists at scratchpad/e2e_nolearn.csv, so
whoever builds the replay driver can finish it.

Remaining issues:
- Payload-byte diff against native: still BLOCKED (Phase 18), now needing
  only the out-of-tree replay build.
- Exploration remains untested end to end (off by default here too).

Next phase:
Phase 24 — Automated equivalence suite
==================================================
```

### Coverage gaps — status after the fix pass

| Gap | Status |
|---|---|
| **G1** payload-byte diff vs native | **CLOSED** — Phase 18 ran; see D18-1 / D18-2 |
| **G2** host-resident selection untested | **CLOSED 2026-08-13** — `compressor_residency_invariance` drives identical bytes through `DynamicSchedule` twice, from host SHM and from a registered `kDeviceMem` backend, and requires the same `compress_lib_` and `compress_preset_` from both, plus a byte-exact round trip of the device-resident blob. Profiled to confirm BOTH halves execute rather than one silently skipping: `InferKernel` (host-stats variant) **and** `InferKernelDeviceStats` + `RankKernel` all launch in that single test. This is the gap the parity suite structurally cannot cover — it feeds both sides the same bytes, so it can compare feature VALUES but never the two different code paths that produce them. |
| **G3** exploration untested | **CLOSED 2026-08-13** — `compressor_exploration`; see D23-2 |
| **G4** phases 10-14 share three binaries | still open (accepted risk; they link upstream's real sources) |
| **G5** cuSZ test 1000x loose | **CLOSED** — tightened with D16-1, proven to fail under `Rel` |

### Phase 24 — Automated equivalence suite (coverage assessment)

```
==================================================
PHASE CHECKPOINT
==================================================

Phase:
24 — What the equivalence suite covers, and what it does not

Status:
COMPLETE — suite is strong where it exists; **five coverage gaps recorded**

Inventory: 457 ctest tests total; 10 labelled `693`, 8 labelled `parity`.
The 6 `ctp_neuropress_*` parity binaries link **upstream's own .cu files**
and diff against the port, so they are genuine cross-tree comparison, not
self-consistency checks. That is the suite's central strength and it should
be said plainly: most ports have nothing of this kind.

Coverage actually demonstrated (all re-run this session):

| Requirement | Test | Result |
|---|---|---|
| shuffle / deshuffle bytes | `preprocess_parity` | byte-identical, 3 elem sizes x 30 sizes, cross-interchange |
| quantize / dequantize | `preprocess_parity` | byte-identical + metadata exact, 5 regimes x 5 bounds |
| entropy / MAD / 2nd deriv | `dataset_parity` | <= 5.4e-15 rel over 256 x 4 MiB chunks |
| selection agreement | `dataset_parity` | 0 / 256 divergent, 4 error bounds |
| NN inference | `parity` | **max rel err = 0** (bit-identical) |
| online + decomp-head SGD | `parity` | weights diffed every step, worst 2.98e-08 |
| multi-flow EMA scoping | `flow_parity` | 1-flow vs 2-flow diverge identically on both sides |
| tie-break rule | `tiebreak_parity`, `device_path_parity` | 1530 + 251 checks, 0 failures |
| lossless round trip | `neuropress_e2e` (Phase 23) | 256/256 exact, 0 of 1 GiB bytes differ |

GAPS — each one is a place where this investigation had to reason instead of
measure, or could not settle a question at all:

 G1  **No payload-byte comparison against native.** The single substantive
     unmeasured equivalence claim. Blocked on building
     `neuropress_native_replay` out-of-tree. Phases 3/4/15 make identical
     payloads very likely; that remains a HYPOTHESIS. → Phase 18.
 G2  **No test drives host-resident NeuroPress selection.** D7-1 — the
     highest-severity finding — is therefore untested, even though Phase 20
     proved it is the path the default functional test takes. A test that
     asserts what happens to a host-resident chunk with NeuroPress configured
     would have caught it.
 G3  **Exploration is untested end to end**, on both the winner-adoption path
     (which can REPLACE an already-stored blob) and the training-sample path.
     Off by default in both trees, which is why it slipped through.
 G4  **Phases 10-14 rest on three shared binaries.** A defect in that harness
     would hide a real divergence in five phases at once. Mitigated by their
     linking upstream's real sources, but not eliminated.
 G5  **cuSZ's error-bound mode has no test that would fail on it.** The
     existing `ctp_compress` cuSZ case asserts `max_err < 1.0` against a
     nominal 1e-3 bound — 1000x loose — so it passes in either mode. This is
     the clearest case of a test written around a behavior rather than
     against a specification. → D16-1.

Also worth recording: the suite asserts NUMERICAL agreement, never EXECUTION
LOCALITY. Every parity test would still pass if the port computed upstream's
exact values in a host loop — which Phase 20 showed is what the default
functional test actually does. That is not a defect in the tests; it is a
statement about what they can and cannot tell you, and it is why the nsys
profiling in phases 20-22 was necessary.

Next phase:
Phase 25 — Performance comparison
==================================================
```

### Phase 25 — Performance comparison

```
==================================================
PHASE CHECKPOINT
==================================================

Phase:
25 — Performance comparison

Status:
PARTIAL — like-for-like codec numbers exist but are INHERITED, not produced
by this investigation. No new perf measurement was run.

INHERITED (commit a7886973, produced with CLIO_CODEC_KERNEL_TIMING, which
brackets exactly the mgr->compress()/decompress() launch with CUDA events on
both sides). 255 chunks of 4 MiB, chunk 0 dropped as warmup:

                     median     min       max     total
  COMPRESS   clio    0.3146   0.1876   16.1779   251.6 ms
  COMPRESS   native  0.2847   0.1741   18.1094   220.5 ms
  DECOMPRESS clio    0.1991   0.1655    9.0246   108.6 ms
  DECOMPRESS native  0.1572   0.1454    5.8728    79.5 ms
  compress   median clio/native  1.105
  decompress median clio/native  1.266

That commit's own characterization is the right one and I do not improve on
it: same codecs, same job, timed the same way, CLIO consistently the slightly
slower caller, difference at the scale of call-site overhead rather than
different work. It explicitly does not claim parity.

MEASURED BY ME, and deliberately NOT turned into a claim:
Transfer counts from the two nsys profiles (Phase 20/21) —
  host-path functional test : 21 H→D, 56 D→H, 0 D→D, 1 CLIO kernel
  device-path gpu_direct    : 197 H→D, 897 D→H, 179 D→D, 8 CLIO kernels
These are raw counts over different workloads doing different amounts of work.
They are NOT normalized per chunk and NOT attributed per call site, so they
support no efficiency conclusion in either direction. Recorded as evidence
for whoever does the attribution.

E2E throughput observed in Phase 23 (1 GiB, learning off):
  H5Dwrite + drain  6,380.61 ms  (160.49 MiB/s)
  H5Dread          24,554.04 ms  ( 41.70 MiB/s)
The read figure is NOT a decompression measurement — VOL-3 means that read is
served from the uncompressed HDF5 copy. Quoting it as a decompression cost
would be wrong.

NOT DONE — D8-1's falsification experiment: rescore exploration using
ctp::LastCodecKernelMs() instead of the chrono delta and check whether any
winner changes. This is the one perf-adjacent experiment with a *correctness*
consequence, since cost drives selection and the SGD gate. Still OPEN.

Next phase: 26
==================================================
```

### Phase 26 — Discrepancy investigation

```
==================================================
PHASE CHECKPOINT
==================================================

Phase:
26 — Discrepancy investigation

Status:
COMPLETE for what was investigable; 4 items carried forward with named
experiments

RESOLVED THIS INVESTIGATION (audit claims re-tested at HEAD):
  NN-1  decomp-head SGD "no kernel exists, scalar host C++"
        → **REFUTED at HEAD.** DecompHeadSGDKernel exists (commit a5e11062)
          and matches native to ~1 ULP over replayed growing batches.
  NN-3/NN-4  cost model, argmax, PSNR mask claimed host-side
        → **REFUTED.** RankKernel does all three on the GPU and launches 64
          times in the device profile.
  PRE-2/PRE-5  host byte shuffle/unshuffle
        → **CONFIRMED as locality**, but DOWNGRADED: the host path is
          byte-identical to native's kernel, so it can never corrupt a blob.
  STATS-5/ORCH-1  host statistics
        → **CONFIRMED as locality**, DOWNGRADED the same way: host values
          agree with native's kernels to 5.4e-15.
  PRE-14  DequantizeDevice has no pointer-kind guard
        → **RECLASSIFIED.** Native's dequantize has no guard either; both take
          device pointers by contract. Not a parity divergence. The residual
          concern (D17-1) is that CLIO's architecture can produce a violating
          caller where native's cannot.
  STATS-7 / heuristic / x_mins / prev_grad_dot
        → **NOT GAPS.** All four are dead upstream. Verified individually.
  "ranked LZ4 silently becomes CPU LZ4"
        → Remains FALSE; nothing found contradicts the audit's own refutation.
  VOL-3  H5Dread decompresses nothing after reopen
        → **CONFIRMED**, and already documented in the port's own e2e source
          with the causing line numbers. Data correct, no read-path benefit,
          dataset stored twice.
  "device scratch is leaked" (audit's out-of-frame item #1)
        → **CONFIRMED at HEAD** (D19-1), three ways.
  cuSZ absolute-vs-relative (audit's out-of-frame item #2)
        → **CONFIRMED as code** (D16-1), and newly bounded: unreachable from
          NeuroPress selection on either side.

CARRIED FORWARD, each with the experiment that would close it:
 1. **Payload-byte equivalence vs native (G1 / Phase 18).**
    Build neuropress_native_replay out-of-tree, run --inference-only against
    scratchpad/e2e_nolearn.csv + /tmp/neuropress_e2e_data.bin, diff payloads
    excluding each side's container header.
 2. **D8-1 exploration cost clock.**
    Feed LastCodecKernelMs() into the cost lambda; check for winner changes.
 3. **D16-1 magnitude.**
    Construct Cusz with Abs inside test_compress.cc's existing harness and
    tighten the assertion to the nominal bound. (My standalone probe failed —
    it never produced a valid round trip in either mode, so it measured
    nothing. Do not reuse it.)
 4. **D17-1 residency.**
    Determine whether output_fullptr.ptr_ can be host while codec_dst is
    device, which would hand DequantizeDevice a host destination.

Next phase: 27
==================================================
```

### Phase 27 — Final equivalence matrix

```
==================================================
PHASE CHECKPOINT
==================================================

Phase:
27 — Final equivalence matrix

Status:
COMPLETE
==================================================
```

| # | Requirement | Verdict | Evidence |
|---|---|---|---|
| 1 | GPU shuffle | **PASS — BYTE-IDENTICAL** | 3 elem sizes × 30 buffer sizes vs upstream's real entry point |
| 2 | GPU deshuffle | **PASS — BYTE-IDENTICAL** | same, incl. CLIO inverting native's own blob |
| 3 | GPU quantization | **PASS — BYTE-IDENTICAL** | packed bytes + `scale`/`min`/`max` exact, 5 regimes × 5 bounds |
| 4 | GPU dequantization | **PASS — BYTE-IDENTICAL** | restored floats byte-equal |
| 5 | Statistics (entropy/MAD/2nd deriv) | **PASS — NUMERICALLY EQUIVALENT** | ≤ 5.4e-15 rel, 256 × 4 MiB chunks. Bit-exactness is impossible in principle — native uses `atomicAdd(double)` |
| 6 | Byte-identical static selection | **PASS** *(revised — D7-1 was overstated)* | 0/256 divergent on device. Host-resident chunks are ranked by NeuroPress too; `compressor_residency_invariance` proves host and device copies of identical bytes select identically |
| 7 | Byte-identical inference | **PASS — BIT-IDENTICAL** | max rel err **= 0** |
| 8 | Byte-identical exploration | **PASS — FIXED `66c86af5`** | cost formula matched term-for-term; the clock did not. Now CUDA-event kernel time on both sides. Path covered by `compressor_exploration` (gap G3 closed) |
| 9 | Byte-identical inference-learning | **PASS** | weights diffed every step, worst 2.98e-08 (~1 ULP) |
| 10 | Runtime SGD | **PASS** | multi-flow EMA scoping reproduces native's divergence exactly (0.00198627 both sides) |
| 11 | Ranking / tie-break | **PASS** | 1530 + 251 checks; D9-1 float↔double bounded by measurement |
| 12 | Diagnostics | **DIFFERENCE (D6-1) — WONTFIX, accepted** | not ported; native's only consumers are its own tests |
| 13 | `.nnwt` unchanged | **PASS — BYTE-IDENTICAL FILE** | md5 `df52a926af026fc617e172dcc990a395` |
| 14 | Lossless decompression | **PASS** | 256/256 exact, 0 of 1,073,741,824 bytes differ |
| 15 | Quantized error bound | **PASS — FIXED `66c86af5`** | linear quantizer within bound; cuSZ default changed `Rel`→`Abs` to match native. Test tightened (gap G5 closed) and proven to fail under `Rel` |
| 16 | Read-path equivalence | **PASS (structure)** | same inversion order, same metadata-from-header rule, loud failure |
| 17 | Blob interchange | **NOT APPLICABLE (D15-1)** | containers disjoint by design; neither parses the other |
| 18 | Payload bytes vs native | **DIFFERENCE — INTRINSIC, NOT FIXABLE (D18-1/D18-2)** | 256/256 selections identical; **209/256 payloads differ** at identical payload size. Root cause: nvcomp ANS is history-dependent (`--flush-cache` changed 189/191 ANS hashes; `--zero-output` changed 0; native vs native 256/256 identical). Byte-identical payloads were never achievable |
| 19 | No D→H→D | **PARTIAL** | counts measured, not normalized or attributed. Remaining work sharpens VOL-1, which D21-1 already accepted |
| 20 | No CPU fallback | **DIFFERENCE FOUND (D20-1) — severity MEDIUM** *(re-graded)* | default functional test takes the host path (1 CLIO kernel); device path complete and correct (8 kernels). The HIGH grade rested on D7-1's reachability, and **D7-1 was corrected as overstated** — `compressor_residency_invariance` proves host and device selections agree. Performance/coverage concern, not correctness |
| 21 | CUDA IPC device pointer | **PASS — FIXED `66c86af5`** | scratch was never freed (379 `cudaMalloc` vs 73 `cudaFree`); added `kDeregisterMemory = 36` admin method, IPC handle closed before free, `in_process` gate prevents cross-process use-after-free |

### Phase 28 — Final verdict

```
==================================================
PHASE CHECKPOINT
==================================================

Phase:
28 — Final verdict

Status:
COMPLETE
==================================================
```

**On the question the investigation was asked: is CLIO-NeuroPress equivalent
to native NeuroPress?**

For everything that determines the CONTENT of a compressed blob — shuffle,
quantization, statistics, inference, ranking, SGD, weights, codec
configuration — **the port is equivalent, and the agreement is far tighter
than "equivalent" usually means.** Inference is bit-identical. Shuffle,
quantize and dequantize are byte-identical. The weights file is the same file.
Statistics agree to within double-precision rounding, which is the strongest
claim the algorithm admits because native is not bit-reproducible against
itself. A gigabyte round-trips with zero differing bytes.

That is not an assumption inherited from the port's comments. Every one of
those results was produced by binaries that link upstream's own `.cu` sources
and diff against them, re-run in this session.

**The divergences are not in the arithmetic. They are in the plumbing.**

*This list was written before the fix pass and is superseded — see the
disposition table below and `NEUROPRESS_CLIO_EQUIVALENCE_REPORT.md`. It is
kept because it records what was believed at the time, including two
gradings that later proved wrong.*

 1. ~~**D7-1 / D20-1 (HIGH).**~~ **D7-1 CORRECTED — the claim was overstated.**
    Host-resident chunks *are* ranked by NeuroPress, through the same
    `RankIntoStats` → `predictor.Rank()` path as device chunks. What survives
    is the `!features_ok` hand-off, silent on the host path only → fixed with
    a log line. D20-1's HIGH grade rested entirely on D7-1's reachability and
    is **re-graded MEDIUM**; `compressor_residency_invariance` (gap G2) now
    proves host and device selections agree.
 2. ~~**D19-1 (HIGH, resource).**~~ **FIXED `66c86af5`** — `kDeregisterMemory`
    admin method; IPC handle closed before free, `in_process` gate prevents a
    cross-process use-after-free.
 3. ~~**D8-1 (MEDIUM-HIGH).**~~ **FIXED `66c86af5`** — CUDA-event kernel time
    on both sides. Path now covered by `compressor_exploration` (gap G3).
 4. **D16-1 FIXED** (`Rel`→`Abs`, gap G5 closed). **D15-1, D6-1 WONTFIX** —
    accepted architectural differences, CLIO does not link native NeuroPress.

**Final disposition of every divergence:**

| Outcome | Items |
|---|---|
| Fixed and verified | D19-1, VOL-5, VOL-2, D16-1, D8-1, D23-2 |
| WONTFIX — CLIO architecture preserved | D21-1, D21-2, D15-1, D6-1, D22-1, D23-1, VOL-1 |
| Explained by root cause | D18-1 ← D18-2 (nvcomp ANS is history-dependent), VOL-3 (fail-closed stamp guard) |
| Corrected — my own earlier findings | D7-1 (overstated), D22-1 (severity), D23-1 (claim wrong), VOL-3 (mechanism), D20-1 (severity) |
| Open, cannot change the verdict | D20-1 (MEDIUM), D22-2 (LOW), Phase 21 attribution, task 27 MAPE |

**What I could not settle.** *(Superseded — Phase 18 subsequently ran.)*
~~Payload bytes have never been compared against native.~~ They were compared:
**209/256 differ** at identical payload size with 256/256 identical selections.
The hypothesis that identical inputs and configuration imply identical payloads
was **REFUTED**, and Phase 26 found why — nvcomp's ANS encoder is
history-dependent (D18-2), so byte-identical payloads were never achievable
against it. Decompression is exact on both sides regardless.

**On the prior audit.** Its central empirical claim reproduces exactly on the
test it profiled, and does not generalize: it measured the host-resident
functional test and concluded the integration runs on the host. The device
path launches all eight CLIO kernels. Four of its findings are refuted or
reclassified, four "missing" features turned out to be dead code upstream, and
two of its out-of-frame discoveries — the leak and the cuSZ mode — are
confirmed and are among the most actionable results here.

**VERDICT REVISED 2026-08-13** after the VOL sweep, the config sweep, the unread-files sweep and Phase 18 finally running. The original verdict below was written when payload equivalence was still an untested hypothesis and the VOL was ~1% read. Both gaps are now closed and the verdict is narrower:

**EQUIVALENT IN DECISIONS, NOT IN OUTPUT BYTES, AND DIVERGENT IN INTEGRATION ARCHITECTURE.**

- **Decisions match.** 256/256 selections identical under controlled replay; inference bit-identical; ranking, SGD and weights all match.
- **Output bytes do NOT match.** D18-1: 209/256 payloads differ at identical payload size — ANS unconditionally, cascaded with shuffle. The codec's inputs and configuration were proven identical, so the divergence is inside the codec invocation and is not yet explained.
- **Integration architecture differs by design.** D21-1 (no H5Z filter — root cause of VOL-1, VOL-3 and the double store) and D21-2 (flat byte range vs N-D HDF5 chunk, which changes the features NeuroPress is fed in production and is invisible to every parity test).
- **Control surface is partly unported.** D22-1 (`best_mode`), D22-2 (cost weights/bandwidth hardcoded), D23-1 (no pool back-pressure or GPU-memory knob), D23-2 (exploration SGD learns from a different batch).
- **Fixed during the investigation:** D19-1 (device scratch leak) and VOL-5 (reproduced SIGSEGV). D7-1 was corrected — it had been overstated.

---

**Original verdict, superseded: EQUIVALENT IN NUMERICS, DIVERGENT IN EXECUTION AND LIFECYCLE.**
The port reproduces NeuroPress's decisions and its bytes. It differs in where
work runs when the caller supplies host memory, in what happens when
NeuroPress cannot decide, and in whether device memory comes back.

---

**FINAL VERDICT (2026-08-13, after the fix pass — supersedes both above):**

**EQUIVALENT IN DECISIONS AND IN RECONSTRUCTED DATA. NOT BYTE-IDENTICAL IN
COMPRESSED PAYLOAD. DIVERGENT — BY DESIGN — IN INTEGRATION ARCHITECTURE.**

Two of the three things the earlier verdict called divergent were fixed: device
memory now comes back (D19-1), and NeuroPress failing to decide already failed
the write (`fa485166`). The third — where work runs on host memory (D20-1) — is
real but does not change selections, proven by `compressor_residency_invariance`.
Payload bytes differ for a reason outside either codebase: nvcomp's ANS encoder
carries state across calls.

Full answers to the protocol's 31 questions: `NEUROPRESS_CLIO_EQUIVALENCE_REPORT.md`.

---

### Phase 29 — Callback-trace equivalence on a GPU-resident chunk

**Deliverable: executable code, not analysis.**
`context-transfer-engine/compressor/example/neuropress_gpu_chunk_equivalence/`
(6 sources + README + CMake), built by the repo's own build system, registered
with ctest as `cte_neuropress_callback_trace_equivalence`. It links native
NeuroPress's `libgpucompress` alongside Clio's port and drives BOTH over the
same GPU-resident chunk, comparing them callback by callback and reporting the
FIRST point of divergence.

**Native has no callback API.** Checked first: `gpucompress.h` exposes no
registration, observer or hook function of any kind. So no parallel callback
system was invented. The harness uses what native actually has:

1. **Native's own stage entry points as the callback boundaries** —
   `runStatsKernelsNoSync`, `gpucompress_infer_gpu`, `quantize_simple`,
   `byte_shuffle_simple`, `gpucompress_compress_with_action_gpu`, called in the
   order upstream's AUTO path calls them.
2. **Native's own per-chunk diagnostics record** — `gpucompress_chunk_diag_t`,
   read through the public `gpucompress_get_chunk_diag`; and its own ranking
   outputs (`out_top_actions`, `out_predicted_costs`, `NNDebugPerConfig`), read
   rather than re-derived.
3. **CUPTI runtime-API callbacks** for the transfer and kernel record — the
   smallest instrumentation that makes "did this stage run on the GPU" and "did
   anything sneak a D→H in here" observed rather than inferred, with neither
   project's source modified.

**Method notes that matter for reading the result.**
- The chunk is generated by a CUDA kernel into `cudaMalloc`'d memory and the
  pointer is classified through `cudaPointerGetAttributes`; the run refuses to
  proceed unless the driver says DEVICE. There is no host original.
- Byte comparisons run in a device kernel; only a 24-byte verdict is copied to
  the host. Every harness copy is tagged `TEST_HARNESS_TRANSFER` and excluded
  from production transfer counts.
- Quantization and shuffle intermediates are captured by calling upstream's own
  `quantize_simple` / `byte_shuffle_simple` separately (labelled STAGE PROBE),
  because the public API returns no intermediate buffer. The production
  compress call runs too and its kernels are recorded under `compression`.
- **The decomposition is validated against production.** The run also drives
  the whole chimod (`AsyncDynamicSchedule`) and compares the selection Clio's
  own per-chunk selection log recorded against the traced one: **8/8 identical**.
  So the stage decomposition is a faithful model of the production path, not a
  parallel reimplementation of it.

**MODE RUN — and therefore what Phase 29 does NOT cover.** Both sides ran pure
inference-mode AUTO, which is the default operating mode of each:

| | native | Clio |
|---|---|---|
| selection | `ALGO_AUTO` + `GPUCOMPRESS_SELECT_NN` | `NeuroPressNNPredictor`, `GpuInferenceActive()` required true |
| online learning / SGD | off (`gpucompress_disable_online_learning`) | off (`neuropress_online_learning_enabled_ = false`); no `Train()` call in the decomposition |
| exploration | off (`gpucompress_set_exploration(0)`) | off (`neuropress_exploration_enabled_ = false`) |
| `best_mode` (exhaustive 32-config) | off (default `g_best_mode{false}`) | not ported (D22-1) |
| min PSNR floor | 0 (`g_min_psnr_db`) | 0 |
| `preprocessing` mask | `PREPROC_NONE` — quantize/shuffle come from `decodeAction`, not the mask | n/a |

Exploration and online SGD were left off deliberately: both make the recorded
action a function of measured timings and of chunk ORDER rather than of the
chunk's content, which is not a property two implementations can be expected to
reproduce chunk-for-chunk, and would have made any divergence uninterpretable.

**So Phase 29 does NOT cover:** exploration (Phase 8), online SGD (Phase 13),
`best_mode`, `GPUCOMPRESS_SELECT_HEURISTIC`, a nonzero PSNR floor, or the
external codecs cuSZ/nDzip/cuSZp — the last of which bypass the action
machinery entirely (`gpucompress_compress.cpp:61-152`) and are Phase 16.

**RESULT: 8/8 chunks PASS**, stable across three runs. Coverage: shuffle
selected (chunks 0, 5) and not (1, 2, 3, 4, 7); quantization selected (5) and
not; two different algorithms chosen (bitcomp, lz4); an error-bound-boundary
chunk (values ~1e-6 against a 1e-6 bound); a 256 KiB chunk and a 32 MiB one.

| Callback | Result |
|---|---|
| sequence | 8/8 identical after normalization; **zero** CLIO architectural callbacks needed removing on this path |
| statistics | entropy / MAD / 2nd-derivative relative error **0** on every chunk; both outputs device-resident |
| diagnostics | native `gpucompress_chunk_diag_t` fields match; `nn_action`, `exploration_triggered`, `sgd_fired` all equal |
| inference | per-action predicted ratio / comp time / decomp time / PSNR, all 32 actions, max relative error **0** — bit-identical |
| ranking | selectable prefix identical; see the tie note below |
| selection | action, algorithm, shuffle bit, quantize bit — 8/8 identical |
| quantization | precision, effective error bound, scale, data_min, data_max all equal; **quantized output byte-identical**; **each side round-tripped through its OWN dequantizer and every element checked against the REQUESTED bound — 0 violations of 1,048,576 on both sides, and the two reconstructions byte-identical (4 MiB)** |
| shuffle | **output byte-identical** (4 MiB buffers) |
| compression | codec identical; payload length identical 8/8; **payload byte-identical 7/8** (see D29-1) |
| GPU residency | driver-verified DEVICE on both sides, every chunk |
| CPU fallback | none on either side — every data stage launched kernels |
| production D→H of the payload | **zero on both sides.** Largest production transfer anywhere is 1152 B against a 4 MiB chunk |

**Two differences reported but not failed, both stated in full in the report:**

- **Ranking tie blocks.** On lossless chunks the two ranked orders agree for the
  first 16 entries and then differ: native emits actions 24-31 before 8-15,
  Clio emits 8-15 before 24-31. Every action in that tail is a *quantize*
  action, which upstream's ranking kernel masks to −INFINITY when the error
  bound is not positive (`nn_gpu.cu:238`). All 16 therefore carry the same
  score, so their relative order is not a decision either implementation made —
  it is each one's sort applied to a block of equal keys. The harness proves
  this automatically rather than assuming it: a positional difference counts as
  tie-explained only when both actions are masked or their native costs are
  bit-equal, and any position that is not tie-explained fails the check. Rank 0
  is compared separately and unconditionally in `selection`. **Cannot affect
  which action is selected.**
- **Feature readback asymmetry.** Clio makes a real production 24-byte D→H of
  the three features after inference (`compressor_runtime.cc:714`); upstream's
  AUTO path makes none at all and zeroes those fields in the stats it reports
  (`gpucompress_compress.cpp:323-325`). The VALUES agree exactly. This is the
  Phase 6 observability difference, re-confirmed, and is reported as such.

**NEW FINDING — D29-1: nvcomp bitcomp output is not reproducible against
itself on small, highly compressible inputs. Severity LOW; it INVALIDATES byte
equality as a parity criterion for that codec rather than indicating a port
defect.**

Chunk 6 (256 KiB, stepped data, action 7 = bitcomp, no shuffle, no quantize,
1128-byte payload — a 232× ratio) is the one chunk whose payload is not
byte-identical: 4 of 1128 bytes differ. Running each side **twice over the same
chunk** shows why: **native does not reproduce its own output either** — 6 of
1128 bytes differ between two native runs — and Clio likewise differs from
itself in 5 of 1128. The differing offsets scatter (7, 33, 34) and the
surrounding frame, including the bitcomp magic `26 13 91 52` and the length
fields, is identical. The two managers are configured identically
(`NVCOMP_TYPE_LONGLONG`, `algorithm = 0`, chunk size `1<<16` — verified in
`compression_factory.cpp:127-133` and `nvcomp.h:531-536`), the inputs are
byte-compared equal, and every callback before `compression` passes. The
behaviour reproduced identically on three consecutive runs. The other seven
chunks — including chunk 0, which is *also* bitcomp at 4 MiB — are fully
reproducible on both sides and byte-identical across sides.

**Why this matters beyond one chunk:** D18-1 recorded "209 of 256 payloads
differ" as a divergence with an unexplained root cause, and D18-2 already
attributed the ANS half to nvcomp state carried across calls. D29-1 is direct
evidence of the same class for a second codec, measured the decisive way —
against the implementation *itself* rather than against the other one. Any
future payload-byte comparison must establish self-reproducibility first, or it
cannot distinguish a port defect from a codec that does not agree with its own
previous output. The harness now does this automatically for every chunk.

**Also corrected while doing this (harness-side, not findings):** `wire_id 0` is
overloaded — it is brotli's registered id AND the sentinel the runtime writes
when compression turned out not to reduce a chunk
(`compressor_runtime.cc:2685`). Resolving it through `NameForWireId` reads a
not-beneficial chunk as a brotli chunk, which is wrong in a way that looks
entirely plausible; the first draft of the cross-check did exactly that and
reported a nonexistent 0/8 disagreement. Worth knowing for any future tool that
reads `compress_lib_`.

**Verdict impact: none.** Phase 29 strengthens the existing "equivalent in
decisions and in reconstructed data" finding by localizing it callback by
callback on a genuinely device-resident path, and supplies a mechanism for the
one part of the earlier verdict that was left unexplained.

#### Phase 29 addendum — error-bound verification, and how hard it is to get quantization SELECTED

**Correction to the first Phase 29 write-up.** It reported quantized output as
byte-identical and left it there. Byte-equality between the two quantized
buffers says the implementations AGREE; it says nothing about whether either
honours the guarantee the caller asked for. `CountBoundViolations` existed in
the harness but was **never called** — there was no dequantize step at all, so
the error bound was UNVERIFIED. Now fixed: after quantizing, each side is sent
back through its OWN dequantizer (`dequantize_simple` on native,
`ctp::…::DequantizeDevice` on Clio) and every element is compared against the
**requested** bound (not the smaller effective one) by an on-device kernel.

Measured, `--error-bound-rel 0.005`:

| chunk | eb (absolute) | precision | effective eb | native violations | clio violations | reconstructions |
|---|---|---|---|---|---|---|
| 0 constant | 5e-3 | 8-bit | 0.00474964 | **0 / 1048576** | **0 / 1048576** | byte-identical (4 MiB) |
| 5 small-magnitude | 1.0e-8 | 8-bit | 9.49973945592e-09 | **0 / 1048576** | **0 / 1048576** | byte-identical (4 MiB) |

A stage that cannot run the round trip now FAILS as "the bound is UNVERIFIED"
rather than passing silently — the previous behaviour.

**A single absolute error bound cannot exercise quantization across a mixed
matrix, and this is arithmetic, not tuning.** Precision follows
`num_bins = (data_range / (2*eb)) * 1.1` (`quantization.cuh:192-206`). The
regimes here span data ranges from ~2e2 to ~2e6, so one bound that puts a
narrow chunk in 8-bit puts a wide one in **32-bit**, where float32 → int32
saves nothing and the cost model correctly declines to quantize. That is why
`--error-bound 0.01` moved only 1 chunk into quantization. The harness now has
`--error-bound-rel R`, which sizes each chunk's **absolute** bound from its own
measured range so `num_bins ≈ 1.1 / (2R)` — the same precision on every chunk
(R = 0.005 → 8-bit, R = 0.001 → 16-bit). Only the choice of number is relative;
the bound handed to both implementations is still absolute, which D16-1 makes
mandatory.

**Even so, the model rarely picks quantization on this data — and both sides
agree about that.** With every chunk given a bound sized for 8-bit (a 4x
reduction before the codec even runs), 5 of 8 chunks still selected action 0,
plain LZ4 with no quantize bit:

| chunk | eb | selected action |
|---|---|---|
| 0 constant | 5e-3 | 11 = gdeflate + quantize |
| 1 smooth-wave | 1.0 | 0 = lz4 |
| 2 high-entropy | 9999.99 | 0 = lz4 |
| 3 smooth-wave | 1.0 | 0 = lz4 |
| 4 noisy-wave | 1.0025 | 0 = lz4 |
| 5 small-magnitude | 1.0e-8 | 31 = bitcomp + shuffle + quantize |
| 6 stepped | 0.0375 | 7 = bitcomp |
| 7 mixed-bands | 10000 | 0 = lz4 |

8/8 identical between native and Clio, so this is a property of the trained
cost model, not a divergence. It does mean **quantization coverage in this
matrix rests on chunks 0 and 5 only**, and raising the error bound is not a
reliable way to widen it. A regime designed to make quantization win outright
would be the way to extend coverage, and is not yet written.

Runs on record: `--error-bound` unset (mixed plan), `--error-bound 0.01`,
`--error-bound-rel 0.005`. All three **8/8 PASS**; the first two also confirmed
8/8 selection agreement against the real `AsyncDynamicSchedule` path.
