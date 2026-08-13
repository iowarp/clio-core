# Compression backends (nvcomp + external GPU codecs), factory, header — execution-locality audit

NATIVE `/home/cc/NeuroPress` @ b23b8f6 — PORT `/home/cc/clio-core` @ neuropress-693-continued

**10 findings** (1 high, 4 medium/medium-high, 5 low), 9 subcategories explicitly cleared, 1 out-of-scope observation.

**Empirical work done** (not code-reading alone): built and ran 4 probes against the real installed nvcomp 5.3.0.16 / cuSZ / cuSZp on the A100 in this checkout — a registry-resolution probe, an nvcomp device-path + manager-cache probe, an nvcomp capacity probe, and two cuSZ/cuSZp dirty-output-buffer probes.

## The top-priority hypothesis is FALSE — stated up front

The brief flagged "a port that maps a selected LZ4 to CPU LZ4 while native ran nvcomp LZ4 on device" as the highest-severity finding available. **It is not present on the NeuroPress-ranked path.** Verified empirically:

```
base_id=13 -> name=nvcomp-lz4       wire=11 is_gpu=1  NameForWireId=nvcomp-lz4
base_id=14 -> name=nvcomp-snappy    wire=12 is_gpu=1  NameForWireId=nvcomp-snappy
base_id=15 -> name=nvcomp-zstd      wire=13 is_gpu=1  NameForWireId=nvcomp-zstd
base_id=16 -> name=nvcomp-gdeflate  wire=14 is_gpu=1  NameForWireId=nvcomp-gdeflate
base_id=17 -> name=nvcomp-deflate   wire=15 is_gpu=1  NameForWireId=nvcomp-deflate
base_id=18 -> name=nvcomp-ans       wire=16 is_gpu=1  NameForWireId=nvcomp-ans
base_id=23 -> name=nvcomp-cascaded  wire=21 is_gpu=1  NameForWireId=nvcomp-cascaded
base_id=24 -> name=nvcomp-bitcomp   wire=22 is_gpu=1  NameForWireId=nvcomp-bitcomp
```
(compiled `compress_factory.h` standalone, driving the exact chain `neuropress_bridge.cc:270-272` uses). All 8 trained algorithms resolve to GPU wire ids with `is_gpu=1`, so `StageInputIfNeeded` (`compress_factory.h:239-247`) returns the device pointer untouched and `GetPreset` yields `NvComp`. The `is_gpu` flags at `compress_factory.h:496-510` are all correct.

Port's `ctp::NvComp` run end-to-end against real nvcomp with device in/out:
```
nvcomp-lz4       in=1048576 worst_cap=1102028 -> ok=1 out=1052657
nvcomp-zstd      in=1048576 worst_cap=1102028 -> ok=1 out=939258
nvcomp-cascaded  in=1048576 worst_cap=1102028 -> ok=1 out=1049048
nvcomp-bitcomp   in=1048576 worst_cap=1102028 -> ok=1 out=695372
nvcomp-ans       in=1048576 worst_cap=1102028 -> ok=1 out=989326
manager cache hits=0 misses=5 (LRU depth 3, 5 distinct algos)
last codec kernel ms=5.3674
```

---

## CODEC-1: NeuroPress declining to rank silently re-selects a **CPU** codec; native refuses to compress at all
- **Claim:** When the NeuroPress ranking yields no candidates (or the predictor isn't ready), the port falls through to a legacy heuristic whose entire candidate list is CPU-only, so the chunk is compressed by host zstd/lz4/bzip2/zlib — and for a device-resident chunk, `StageInputIfNeeded` first D2H-stages the whole buffer. Native returns `GPUCOMPRESS_ERROR_NN_NOT_LOADED` and compresses nothing.
- **Native evidence:** `src/api/gpucompress_compress.cpp:208-213`
  ```c
  if (ie != GPUCOMPRESS_SUCCESS || action < 0) {
      fprintf(stderr, "gpucompress ERROR: ALGO_AUTO requested but NN inference failed ...");
      return GPUCOMPRESS_ERROR_NN_NOT_LOADED;
  }
  ```
  Also `:285-286`, `:274-275`.
- **Port evidence:** `compressor_runtime.cc:771-777`
  ```cpp
  candidate_lib_configs = {
      {10, 0},  // ZSTD balanced
      {10, 3},  // ZSTD fast
      {4, 3},   // LZ4 fast
      {1, 1},   // BZIP2 best
      {9, 0},   // ZLIB balanced
  };
  ```
  Empirically: `legacy wire=10 -> zstd is_gpu=0`, `wire=4 -> lz4 is_gpu=0`, `wire=1 -> bzip2 is_gpu=0`, `wire=9 -> zlib is_gpu=0`. The selected id reaches `context.compress_lib_` at `:1351`, then `:2197-2198` `StageInputIfNeeded` D2H-stages, then `:2405` runs the CPU codec.
- **Category:** silent-cpu-fallback
- **Severity:** high
- **Reachability:** Four gates, of differing loudness.
  1. `:697-698` — `neuropress_predictor_` null or `!IsReady()` -> the whole NeuroPress block is skipped with **no log at all**. Fully silent.
  2. `neuropress_bridge.cc:152-168` — on a build with `CTP_ENABLE_NVCOMP=0`, `kAvailable` is empty, every candidate filtered out, ranking returns empty -> this CPU fallback fires on **every** chunk. (nvcomp is ON in this checkout, so not active here.)
  3. `:722-735` PSNR filter empties the list when `context.target_psnr_ > 0` — silent on the host path.
  4. Device path only (`:752-758`) logs `kError`; `:683-687` logs `kWarning` for failed device stats.
- **How to falsify:** Show `EstCompressionStats` can never return a non-empty `results` from `:781-836` while `context.dynamic_compress_ != 1`, or that the caller distinguishes a NeuroPress selection from a legacy one (it cannot: `ranked_by_cost` at `:1340` is internal and `task->return_code_` is 0 either way).

## CODEC-2: A GPU codec failure is silently converted into "stored raw, success"; native returns a compression error
- **Claim:** If `compressor->Compress()` returns false, the port takes the "not beneficial" branch, PutBlobs the caller's original bytes, zeroes `compress_lib_`, and reports return code 0. The caller cannot tell a codec failure from a genuinely incompressible chunk.
- **Native evidence:** `gpucompress_compress.cpp:533-545`
  ```c
  try { compressor->compress(d_compress_input, d_comp_target, comp_config); }
  catch (const std::exception& e) { ... return GPUCOMPRESS_ERROR_COMPRESSION; }
  catch (...)                     { ... return GPUCOMPRESS_ERROR_COMPRESSION; }
  ```
  and `:485-489` (null manager -> `GPUCOMPRESS_ERROR_COMPRESSION`). No raw-storage path anywhere in `gpucompress_compress_with_action_gpu`.
- **Port evidence:** `compressor_runtime.cc:2560-2606`. `:2562` `HLOG(kDebug, "Compression not beneficial, storing original data")` — the *same* debug line for both cases; `:2570` `context.compress_lib_ = 0;`; `:2572-2579` unconditional `AsyncPutBlob` with `task->return_code_ = put_task->return_code_`.
- **Category:** silent-cpu-fallback (to the no-codec host path)
- **Severity:** medium
- **Reachability:** Any `success == false` from `:2405`: nvcomp exception (caught at `nvcomp.h:238-241`), `cudaMalloc` failure at `nvcomp.h:216`, `CachedStream()` failure at `nvcomp.h:183-185`, or `comp_size > output_size` at `nvcomp.h:228`.
- **How to falsify:** Show a caller-visible signal distinguishing the two. Only the learning gate at `:1470-1471` consumes `actual_compression_ratio_ == 0.0`.

## CODEC-3: The interposed `PutBlob`/`GetBlob` writer runs quantize + byte-shuffle on the **host** and hands the GPU codec host buffers
- **Claim:** `Runtime::CompressIntoShm` / `Runtime::DecompressStored` perform both preprocessing transforms with host loops over `std::vector`, with no device branch at all, and compress into a host `std::vector<char>`. When `compress_lib_` names an nvcomp codec, `NvComp::Compress` then D2H-copies the compressed output back out of the temp device buffer.
- **Native evidence:** `gpucompress_compress.cpp:437-446` (device quantize with pooled scratch), `:459-463` (device shuffle), `:534` (device->device compress). Decompress `:1256-1296` — `byte_unshuffle_simple` / `dequantize_simple`, both device.
- **Port evidence:** `compressor_runtime.cc:3289-3298` host `Quantize<float>`; `:3304-3311` host `ByteShuffle`; `:3316-3319`
  ```cpp
  std::vector<char> compressed(size + (size / 20) + 1024);
  size_t compressed_size = compressed.size();
  if (!compressor->Compress(compressed.data(), compressed_size,
                            const_cast<char *>(compress_src), compress_size)) {
  ```
  Read side: `:3472-3479`, `:3491-3503`, `:3515-3516`. The comments at `:3269-3273` and `:3487-3490` assert `src` is host SHM, which the code never checks.
- **Category:** host-compute
- **Severity:** medium-high
- **Reachability:** `Runtime::PutBlob` `:3552-3558` -> `CompressIntoShm`, gated at `:3542-3544`. `compress_lib_` is caller-supplied and can be an nvcomp id — `test_compressor_functional.cc:1054` sets exactly that (`context.compress_lib_ = CompLib::NVCOMP_LZ4;`). Note this is *not* the path `DynamicSchedule` uses (that goes through `AsyncCompress` -> `Runtime::Compress`, which does have device branches at `:2269`/`:2331`).
- **How to falsify:** Find a device branch inside `CompressIntoShm`/`DecompressStored` (`IsDevicePointer` does not appear between `:3239` and `:3531`), or prove `Runtime::PutBlob` is unreachable with `compress_lib_ >= 11`.

## CODEC-4: A UVM/managed-memory chunk demotes the whole compress pipeline to the host path
- **Claim:** `ctp::IsDevicePointer` accepts only `cudaMemoryTypeDevice`, not `cudaMemoryTypeManaged`. Every device/host branch in `Runtime::Compress` keys off it, so a managed input runs the host quantizer, the host shuffle, and allocates a **host** output buffer — after which `NvComp` (whose own `IsDeviceAccessible` *does* accept managed) compresses in place on device and then D2H-copies the result.
- **Native evidence:** `gpucompress_compress.cpp:425-470` operates on `d_input` unconditionally; `:510-526` selects `d_comp_target` — always device. No pointer-kind query exists in the file.
- **Port evidence:** `include/clio_ctp/util/gpu_api.h:288` `return attributes.type == cudaMemoryTypeDevice;` (managed excluded), via `gpu_api.h:564-565`. Contrast `nvcomp.h:329-330` `return attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged;`. Consumers: `compressor_runtime.cc:2269`, `:2331`, and decisively `:2374` `bool output_on_device = ctp::IsDevicePointer(input_ptr);` -> `:2386-2388` host `compressed_buffer.resize(...)` -> `nvcomp.h:212-221` temp `cudaMalloc` + `:232-234` D2H.
- **Category:** device-struct-became-host
- **Severity:** medium
- **Reachability:** Any blob backed by `MemKind::kManagedUvm` (`ipc_manager.cc:3845-3846`, `admin_runtime.cc:1227-1228`, `admin_tasks.h:1412`). Not the mainline (`kDeviceMem` is) but a first-class supported memory kind.
- **How to falsify:** Show blob data never arrives as managed memory in any deployed configuration, or that `cudaPointerGetAttributes(...).type` reports `cudaMemoryTypeDevice` for `cudaMallocManaged` (it does not).

## CODEC-5: Exploration winner is round-tripped D2H -> host SHM -> PutBlob; native writes the winner device-to-device
- **Native evidence:** `gpucompress_compress.cpp:946-949`
  ```c
  cudaMemcpy(d_out, &alt_hdr, sizeof(CompressionHeader), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_out + hdr_sz, s.d_out, s.comp_size,
                  cudaMemcpyDeviceToDevice, stream);
  ```
  Only the 64-byte header touches the host.
- **Port evidence:** `compressor_runtime.cc:1845-1852` `winner_payload.resize(...); ctp::DeviceAwareMemcpy(winner_payload.data(), alt_out_ptr, alt_compressed_size);` then `:1958-1971` host SHM `AllocateBuffer` + `std::memcpy`, then `:1982-1987` `AsyncPutBlob`. Inconsistent with the port's own primary path, which keeps the compressed output on device (`:2487-2503`).
- **Category:** added-host-roundtrip
- **Severity:** medium
- **Reachability:** `:1598-1600` — requires `neuropress_exploration_enabled_` **and** `error_pct > neuropress_exploration_threshold_`, then `:1841`. Off by default.
- **How to falsify:** Show `DeviceAwareMemcpy` at `gpu_api.h:492` short-circuits when both pointers are device — it cannot; `winner_payload.data()` is host.

## CODEC-6: Up to three per-chunk device allocations + IPC registrations, where native uses pre-allocated per-context device buffers
- **Native evidence:** `src/api/gpucompress_pool.cpp:135-144`
  ```c
  static constexpr size_t kPreallocChunk = 16UL << 20;  /* 16 MB */
  static constexpr size_t kCubTemp       = 64UL << 10;  /* 64 KB */
  if (cudaMalloc(&ctx.d_preproc_quant, kPreallocChunk) != cudaSuccess) goto fail;
  ```
  consumed at `gpucompress_compress.cpp:443-445`, `:462`; compress target `:510-526` reuses the caller's `d_output` unless too small.
- **Port evidence:** `compressor_runtime.cc:2271-2273`, `:2338-2340`, `:2379-2381` — three `AllocateAndRegisterGpuBackend(...)` per chunk, freed by `DeviceScratchGuard` at `:2231-2242`. Each does a raw `cudaMalloc` plus registration (`ipc_manager.cc:3847-3849`, `:3866-3886`); on a non-runtime `IpcManager` also `cudaIpcGetMemHandle` and a blocking `reg_future.Wait()` RPC (`:3901-3932`).
- **Category:** added-host-roundtrip
- **Severity:** medium
- **Reachability:** Every device-resident chunk on `Runtime::Compress`. `cudaMalloc` is synchronizing, so 1-3 device-wide serializations per chunk. The blocking-RPC leg needs `!IsRuntime()`, unlikely for a chimod task; the malloc/free churn is unconditional.
- **How to falsify:** Show these allocations are pooled — they are not (`ipc_manager.cc:3848`). They do sit outside the *timed* region (`:2402-2408`), which is why the model's label is unaffected.

## CODEC-7: An unrecognized wire id silently resolves to **CPU zstd** and reports `is_gpu=false`
- **Native evidence:** `src/compression/compression_factory.cpp:151-158` no default; `gpucompress_compress.cpp:1192` `if (!header.isValid()) return GPUCOMPRESS_ERROR_INVALID_HEADER;`; `compression_factory.cpp:143-145` `default: throw std::runtime_error("Unsupported compression algorithm");`
- **Port evidence:** `compress_factory.h:199-202`
  ```cpp
  static std::string NameForWireId(int wire_id) {
    const CompressorInfo* info = FindByWireId(wire_id);
    return info ? info->name : "zstd";
  }
  ```
  `:257-260` `WireIdForName` mirrors it. Empirically: `out-of-range wire=99 -> zstd is_gpu=0`. Consumed unchecked at `compressor_runtime.cc:2143-2144`, `:2780-2781`, `:3244`, `:3396-3397`.
- **Category:** silent-cpu-fallback
- **Severity:** low
- **Reachability:** Requires `compress_lib_` outside 0-22 — a caller-supplied static id or a corrupt header. Not reachable from the NeuroPress ranking.
- **How to falsify:** Show every `compress_lib_` reaching `NameForWireId` is validated first. It is not (`:2128` only checks `<= 0`; `:2741` reads the header field raw).

## CODEC-8: cuSZ/cuSZp decompress omit native's device zero-fill of the output buffer
- **Native evidence:** `src/compression/external_compressors.cu:206-215`
  ```c
  /* cuSZ's Lorenzo decode uses the output buffer as the outlier-scatter
   * base and ADDS reconstructed values into it, but only zero-fills it for
   * the Spline predictor (psz/src/compressor.inl). A recycled device buffer
   * with stale contents silently corrupts every value ... */
  ce = cudaMemsetAsync(d_output, 0, original_size, stream);
  ```
  and `:482-485` for cuSZp.
- **Port evidence:** `cusz.h:196-202` no memset before `psz_decompress_float` at `:223`; identically `cuszp.h:222-228` before `cuSZp_decompress` at `:244`.
- **Category:** missing-gpu-work
- **Severity:** low
- **Reachability:** Every `Cusz`/`Cuszp` `Decompress` with a device output. Outside NeuroPress's trained action space on both sides — explicit/static selection only.
- **How to falsify:** **Tried and could not reproduce a corruption.** Compiled both port wrappers against installed cuSZ/cuSZp v3, decompressed the same blob twice into the same 1 MiB device buffer — once pre-zeroed, once pre-filled `0xAB`:
  ```
  cuSZp : elements differing (zeroed vs dirty output buffer): 0 / 1048576
  cuSZ  : elements differing (zeroed vs dirty output buffer): 0 / 1048576
  ```
  Reported as a factual missing device operation and a latent version-dependent divergence, **not** a demonstrated corruption.

## CODEC-9: cuSZ/cuSZp/ndzip create and destroy a CUDA stream on every call; native runs them on the caller's stream
- **Native evidence:** `external_compressors.cu:82-85` signature takes `cudaStream_t stream`; `:111` `psz_create_resource_manager(F4, len, pipeline, stream);`. Caller supplies it: `gpucompress_compress.cpp:68-69`. No `cudaStreamCreate` anywhere in `external_compressors.cu`.
- **Port evidence:** `cusz.h:100-103`/`:170-173`; `cuszp.h:110-113`/`:192-195`; `ndzip.h:94-97`/`:163-166` — all `cudaStreamCreate(&stream)` on entry, `cudaStreamDestroy` at `cusz.h:160`/`:239`, `cuszp.h:182`/`:263`, `ndzip.h:148`/`:202`. `nvcomp.h:398-404` gets this right with a cached per-thread stream; the three external wrappers were not given the same treatment.
- **Category:** added-host-roundtrip
- **Severity:** low
- **Reachability:** Every cuSZ/cuSZp/ndzip call (explicit/static selection only). `cudaStreamDestroy` implies synchronization against outstanding work.
- **How to falsify:** Show a cached stream exists in these three wrappers — `grep -c CachedStream cusz.h cuszp.h ndzip.h` is 0.

## CODEC-10: Exploration compresses K alternatives sequentially; native runs them concurrently on K CUDA streams
- **Claim:** Work stays on the GPU on both sides — this is lost device concurrency, not relocated compute.
- **Native evidence:** `gpucompress_compress.cpp:744-747` `/* ---- Parallel exploration: K alternatives on K separate streams ---- */`; `:797` `cudaStreamCreate(&s.stream)` per slot; `:869-870` compress inside the launch loop; `:880-884` separate Phase-2 sync loop.
- **Port evidence:** `compressor_runtime.cc:1591-1597` — the comment states the divergence outright; implemented at `:1809-1811` inside the `for (const auto* alt : alternatives)` loop opened at `:1642`; `nvcomp.h:226` syncs the single shared per-thread stream on each call.
- **Category:** missing-gpu-work
- **Severity:** low
- **Reachability:** Same opt-in gate as CODEC-5. Off by default on both sides.
- **How to falsify:** Show the port dispatches alternatives concurrently. It does not; `nvcomp.h:398-404` hands every call on a thread the *same* stream.

---

# Subcategories that yielded nothing (explicit)

- **CPU substitution on the NeuroPress-ranked path** — none. All 8 trained algorithms resolve to GPU wire ids and GPU wrappers; verified empirically. The highest-severity hypothesis in the brief is false.
- **nvcomp manager caching** — parity. `nvcomp.h:371-449` is a faithful thread-local LRU-3 keyed by algorithm with hit/miss counters, matching `gpucompress_pool.cpp:236-271` and `internal.hpp:83-86` (`LRU_DEPTH = 3`). Chunk size matches (`nvcomp.h:315` `1 << 16` vs `internal.hpp:97`). Confirmed at runtime: `hits=0 misses=5` for 5 distinct algorithms against a depth-3 cache. Not caching the decompress manager (`nvcomp.h:251-255`) correctly mirrors `compression_factory.cpp:151-158`.
- **nvcomp per-algorithm options** — no divergence with execution consequence. Cascaded `NVCOMP_TYPE_CHAR` (`nvcomp.h:496-498`) and Bitcomp `NVCOMP_TYPE_LONGLONG`/`algorithm=0` (`nvcomp.h:510-513`) match `compression_factory.cpp:122-133`, and both overrides are genuinely needed (defaults verified in `/usr/include/libnvcomp5-dev-cuda-12/nvcomp/{cascaded,bitcomp}.h:81,65`). The port omits native's explicit LZ4 `opts.data_type = NVCOMP_TYPE_CHAR`, but the nvcomp default is already `NVCOMP_TYPE_CHAR` (`lz4.h:90-91`) — a no-op.
- **Silent build exclusion of a GPU codec** — none in this checkout. `build/CMakeCache.txt`: `CLIO_CTP_ENABLE_NVCOMP=ON`, `CUSZ=ON`, `NDZIP=ON`, `CUSZP=ON`, `LIBPRESSIO=ON`; `ZFP_SYCL=OFF` (not in either action space). Confirmed empirically: `ldd build/bin/libclio_cte_compressor_runtime.so` links `libnvcomp.so.5`, `libcusz.so`, `libndzip-cuda.so`, `libcuSZp.so`; `nm -C` shows all 8 `nvcomp::*Manager` constructors referenced.
- **nvcomp dispatching to a CPU backend** — not possible. `nvcompDecompressBackend_t` has only `DEFAULT`/`HARDWARE`/`CUDA` (`shared_types.h:70-90`); `libnvcomp_cpu.so.5` exists on the system but is not linked (confirmed by `ldd`).
- **Compressed output forced through a temp buffer** — no systematic extra copy. Device-direct output requires `output_size >= cfg.max_compressed_buffer_size` (`nvcomp.h:213`). Measured against real nvcomp with Clio's `worst_case_size = n + n/20 + 1024` (`compressor_runtime.cc:2181`):
  ```
  n=    4096 worst_cap=    5324 | lz4=    4208  zstd=    4200  casc=    4208  bitc=    8320*
  n=   65536 worst_cap=   69836 | lz4=   65888  zstd=   65640  casc=   65648  bitc=   65720
  n= 1048576 worst_cap= 1102028 | lz4= 1053128  zstd= 1049160  casc= 1049048  bitc= 1050440
  n= 4194304 worst_cap= 4405043 | lz4= 4212296  zstd= 4196424  casc= 4195928  bitc= 4201544
  ```
  Only a 4 KiB Bitcomp chunk exceeds capacity. For realistic chunk sizes the device-direct path is taken, matching native's `:515-526`.
- **Decompress-side H2D of the compressed stream** — parity, not a finding. Initially suspected the port added one (`compressor_runtime.cc:2682` host SHM `temp_buffer`, `nvcomp.h:271` H2D). Native does the same: `H5VLgpucompress.cu:3287-3291` `vol_memcpy(d_compressed, h_comp_r[item.slot], item.comp_sz, cudaMemcpyHostToDevice)`. The only residual difference is native reusing a pre-allocated `d_compressed` vs the port's per-call `cudaMalloc` (`nvcomp.h:346`) — subsumed by CODEC-6.
- **The compressed-blob header** — no execution-locality divergence. Both keep the payload untouched. Format differences (24 vs 64 bytes) are out of scope.
- **`Runtime::Compress` primary path** — clean. Quantize (`:2269-2291`), shuffle (`:2331-2350`), output (`:2378-2385`) and header (`:2494-2501`) all stay device-resident for a `kDeviceMem` input, and `StageInputIfNeeded` correctly no-ops for GPU wire ids. This is the port's good path and it matches native's shape.

---

# Out-of-scope observation (measured, not a locality finding)

The port's `Cusz` defaults to **relative** error-bound mode (`cusz.h:87` `explicit Cusz(double eb = 1e-3, psz_mode mode = Rel)`), and `MakeCusz` (`compress_factory.h:429-443`) constructs it with only the bound, so `Rel` is what runs. Native uses **absolute**: `external_compressors.cu:118` `rc.mode = Abs;`. Measured on the same 1 M-float array (range +/-100): the port's round-trip max absolute error was **0.200005** at `eb=1e-3`, i.e. ~200x native's bound for the same nominal preset. Changes fidelity and ratio, not where work runs — flagged because it fell out of the memset experiment and likely matters to whoever owns lossy parity.

Also noted: `build/bin/test_compress_model_exec "StageInputIfNeeded*"` **segfaults** in this checkout at `test_compress_factory_gpu.cc:42` (`ctp::GpuApi::Memcpy` immediately after a successful `Malloc`), despite CUDA working fine in the same sandbox. The binary also emits a gcov checksum error. Likely a test-harness init problem rather than a product defect, but the GPU factory test is not currently providing coverage.
