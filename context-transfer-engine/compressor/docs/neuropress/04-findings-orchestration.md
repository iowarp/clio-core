# Orchestration / public-API / control-flow audit — NeuroPress (native) vs clio-core compressor (port)

Native: `/home/cc/NeuroPress` @ b23b8f6. Port: `/home/cc/clio-core` @ neuropress-693-continued.

**16 findings.**

**Framing fact underpinning several findings.** Native has *no host compression path at all*. `gpucompress_compress()` / `gpucompress_decompress()` — the two functions `include/gpucompress.h:197-211`, `:220-229` document as host-memory entry points — are deliberately stubbed:
```
gpucompress_api.cpp:360-365
    (void)input; ... fprintf(stderr, "gpucompress_compress: host-path stub - use "
        "gpucompress_compress_gpu() ... instead.\n");
    return GPUCOMPRESS_ERROR_INVALID_INPUT;
```
Every real entry point takes `d_input`/`d_output` device pointers and never dereferences them on the host except for a 64-byte header and a 24-byte stats struct. The port re-introduces the host path native refused to ship, and takes it by default.

---

## ORCH-1: The port's entire compress pipeline (stats, quantize, shuffle, codec) runs on the host whenever the chunk is host-resident — a path native does not implement
- **Claim:** For a host-resident chunk the port computes the three selection statistics on the CPU, quantizes on the CPU, byte-shuffles on the CPU, and lets the codec do its own H2D/D2H staging. Native performs all four stages on the GPU on a CUDA stream and has no host variant to fall to.
- **Native evidence:** `src/api/gpucompress_compress.cpp:273` then `:280` `AutoStatsGPU* d_stats_ptr = gpucompress::runStatsKernelsNoSync(d_input, input_size, stream, ctx);` with `:282` `/* Stats remain on GPU - NN inference reads d_stats_ptr directly on device. */`; `:440-445` device `quantize_simple(...)`; `:459-463` device `byte_shuffle_simple(..., stream, ...)`; `gpucompress_api.cpp:360-365` host entry point is a stub returning an error.
- **Port evidence:** `compressor_runtime.cc:630-636` `const bool features_ok = np_device_path ? device_stats != nullptr : (num_elements > 0 && ctp::ComputeCompressionFeatures(chunk, num_elements, data_type, &entropy, &mad, &second_derivative_mean));` and `data_stats_gpu.h:141-146` routes a non-device pointer to host loops. `compressor_runtime.cc:2268-2296` host quantize; `:2324-2333` host shuffle; `:2347` `bool output_on_device = ctp::IsDevicePointer(input_ptr);` false here, so `compressed_buffer.resize(worst_case_size)` (`:2360`) and the codec's output returns D2H (`nvcomp.h:232-234`).
- **Category:** host-compute
- **Severity:** high
- **Reachability:** `Runtime::DynamicSchedule` -> `EstCompressionStats` -> `Runtime::Compress`. `chunk_data` is a device pointer only when the HDF5 VOL staged into a device backend, requiring the application's own `src` to be device memory AND a compressor pool: `clio_vol.cc:1827-1828`. Any ordinary host-buffer `H5Dwrite` takes the host branch at `clio_vol.cc:1854-1867`. The device-backend allocation also falls back to host SHM on failure (`clio_vol.cc:1850-1852`) with no error to the caller.
- **How to falsify:** Show `ctp::IsDevicePointer(chunk_data)` is true for every chunk in the intended deployment, or find a device implementation actually invoked from `compressor_runtime.cc:2277` / `:2326`. Empirically: run with a host `malloc` source buffer under nsys and show quantize/shuffle/stats kernels still launch.

## ORCH-2: NeuroPress NN inference silently falls back to a CPU re-implementation; the orchestrator never checks whether the GPU path is live
- **Claim:** The port ships a host C++ forward pass and uses it whenever device weights are absent. The predictor still reports `IsReady() == true`, `EstCompressionStats` returns a normal ranking, and the caller cannot tell. Native has no CPU network at all.
- **Native evidence:** `gpucompress_compress.cpp:274-275` `if (num_elements == 0 || !gpucompress_nn_is_loaded_impl()) return GPUCOMPRESS_ERROR_NN_NOT_LOADED;`; `:296-308`; `:208-213` loud stderr + error.
- **Port evidence:** `neuropress_nn_predictor.cc:526-529` `// CPU fallback: same per-candidate math as before, looped.` — reached because the GPU branch is `#if CTP_ENABLE_NEUROPRESS_GPU` + `if (gpu_weights_)` (`:483-484`). `:157-177` on `NeuroPressGpuLoad` returning null the code prints one stderr warning then `is_ready_ = true; return true;`. On a `CTP_ENABLE_NEUROPRESS_GPU=0` build (`compress/model/CMakeLists.txt:69`) even that warning is compiled out. `neuropress_nn_predictor.h:116` exposes `GpuInferenceActive()`, and `:164` claims "a caller that must not diverge checks GpuInferenceActive() and declines" — but `grep -rn GpuInferenceActive` finds **no** call outside the header, that comment, and one parity test. `compressor_runtime.cc:551-553` gates only on `IsReady()`.
- **Category:** silent-cpu-fallback
- **Severity:** high
- **Reachability:** `EstCompressionStats:670-671` -> `NeuroPressCandidateStats` -> `neuropress_bridge.cc:260` -> `predictor.h:367` -> the CPU loop. Any `CTP_ENABLE_NEUROPRESS_GPU=0` build, or a CUDA build where the device was unavailable at `Load()`.
- **How to falsify:** Show any runtime caller consulting `GpuInferenceActive()` before trusting a ranking, or show `gpu_weights_` can never be null once `is_ready_` is true.

## ORCH-3: A NeuroPress ranking that produces no candidates degrades to a CPU-only legacy candidate list, with no log on the host path — native returns `GPUCOMPRESS_ERROR_NN_NOT_LOADED`
- **Claim:** When the NeuroPress ranking yields nothing, the port falls through to a hardcoded 5-candidate list consisting entirely of **CPU** libraries and compresses on the host. Native fails the call loudly with a dedicated error code.
- **Native evidence:** `gpucompress_compress.cpp:208-213`, plus `:285-286`. Code declared at `include/gpucompress.h:106` `GPUCOMPRESS_ERROR_NN_NOT_LOADED = -10`.
- **Port evidence:** `compressor_runtime.cc:709-717` returns the NeuroPress list only `if (!neuropress_stats.empty())`. `:725-731` the "NOT re-ranking it on the host" `HLOG(kError, ...)` is guarded by `if (device_stats != nullptr && neuropress_stats.empty())` — **on the host-stats path there is no log at all**. `:744-750` the fallback set: `{10,0} ZSTD balanced, {10,3} ZSTD fast, {4,3} LZ4 fast, {1,1} BZIP2 best, {9,0} ZLIB balanced`. Wire ids 10/4/1/9 are all `is_gpu=false` in `compress_factory.h:485-495`. `:792-797` with no predictor at all it invents `pred.compression_ratio = 2.0;`. `Runtime::Compress:2170-2171` then D2H-stages a device chunk for the CPU codec via `StageInputIfNeeded`. The caller sees `task->return_code_ = 0`.
- **Category:** loud-error-became-silent-degrade / silent-cpu-fallback
- **Severity:** high
- **Reachability:** Default. Reached when `neuropress_model_path_` is unset (ORCH-16), when `kAvailable` in `neuropress_bridge.cc:152-168` strips every trained algorithm on a build without nvcomp, when the PSNR filter at `:695-708` empties the list, or when `PredictBatch`/`PredictBatchDeviceStats` returns `{}`.
- **How to falsify:** Show `neuropress_stats` can never be empty in a supported configuration, or that reaching `candidate_lib_configs` still yields a GPU codec.

## ORCH-4: Failure to allocate the device output buffer silently reroutes the compressed output through host memory; native returns `GPUCOMPRESS_ERROR_OUT_OF_MEMORY`
- **Native evidence:** `gpucompress_compress.cpp:515-526` `if (cudaMalloc(&d_temp_out, total_max_needed) != cudaSuccess) { ... return GPUCOMPRESS_ERROR_OUT_OF_MEMORY; }`, with the temp->caller move D2D at `:603-604`.
- **Port evidence:** `compressor_runtime.cc:2351-2361` `if (device_output_alloc_id.IsNull()) { output_on_device = false;  // Fall back to the host buffer below. }` ... `:2477-2494` `std::memcpy(compressed_shm.ptr_ + header_size, compressed_buffer.data(), compressed_size);`. No log, no error code.
- **Category:** silent-cpu-fallback / added-host-roundtrip
- **Severity:** medium
- **Reachability:** `Runtime::Compress`, device-resident chunk, GPU allocator exhausted.
- **How to falsify:** Show a log or non-zero return code on this branch, or that `AllocateAndRegisterGpuBackend` cannot return a null id.

## ORCH-5: Device quantize / byte-shuffle failure silently continues without the transform; native returns `GPUCOMPRESS_ERROR_COMPRESSION`
- **Native evidence:** `gpucompress_compress.cpp:446-453` `if (quant_result.isValid()) { ... } else { return GPUCOMPRESS_ERROR_COMPRESSION; }`; `:464-469` same for shuffle.
- **Port evidence:** `compressor_runtime.cc:2261-2267` and `:2319-2323` + `:2334-2336`. Beyond locality: the training label at `:1540-1545` still credits the chunk to the *ranked* action, and `:2426-2433` computes PSNR only `if (applied_quant)`.
- **Category:** loud-error-became-silent-degrade
- **Severity:** medium
- **Reachability:** `Runtime::Compress` device branch, any `QuantizeDevice`/`ByteShuffleDevice` failure — including builds where those are the CPU stubs that unconditionally `return false`.
- **How to falsify:** Show these cannot fail once their allocations succeed, or find an error return/log on these branches.

## ORCH-6: Compression and decompression are timed with a host wall clock over a window that includes H2D/D2H staging, manager lookup, `cudaMalloc` and a stream sync — native uses CUDA events around the codec launch alone, and the port's comment claims parity
- **Claim:** The port's comment asserts "NeuroPress brackets exactly this with CUDA events ... so including the output allocation and the D2H/H2D staging above would feed the model a systematically inflated comp_time". The staging it claims to exclude happens **inside** the call it times, and it times it with `std::chrono`, not CUDA events. That number feeds the cost model, the MAPE gate and the training labels.
- **Native evidence:** `gpucompress_compress.cpp:529-563` — `cudaEventRecord(ctx->t_start, stream); ... compressor->compress(...); cudaEventRecord(ctx->t_stop, stream); cudaStreamSynchronize(stream); cudaEventElapsedTime(&primary_comp_time_ms, ctx->t_start, ctx->t_stop);`. `get_compressed_output_size` (`:568`) and the header write (`:598-612`) are outside it. NN/stats times likewise from events at `:217-218`. Decompression event-timed by the VOL at `H5VLgpucompress.cu:3302-3311`.
- **Port evidence:** `compressor_runtime.cc:2368-2384` `auto compress_start = std::chrono::high_resolution_clock::now(); bool success = compressor->Compress(...);`. What `Compress()` does inside that window (`nvcomp.h:196-235`): `ToDeviceInput(...)` (`:199`, H2D + `cudaMalloc` at `:346-351`), `GetOrCreateManager` (`:202`), `configure_compression` (`:204`), `cudaMalloc(&d_out, ...)` (`:215`), the kernel, `cudaStreamSynchronize` (`:226`), the delivery `cudaMemcpy` (`:234`). `:2390` a real event-based number exists (`ctp::LastCodecKernelMs()`, `nvcomp.h:120-168`) but is only written to the payload log (`:2400`); `:2417` `context.actual_compress_time_ms_ = compress_time;` is the host clock, feeding `cost(...)` at `:1477-1479`, `error_pct` at `:1480`, and the SGD label at `:1544`. Decompression: `:2775` -> `:2943-2946` -> `:2950` -> `:2955` `LearnDecompTime(...)`.
- **Category:** host-compute (timing fidelity) / added-host-roundtrip
- **Severity:** medium
- **Reachability:** Every compress and decompress, unconditionally.
- **How to falsify:** Show `LastCodecKernelMs()` (or any CUDA-event value) reaching `context.actual_compress_time_ms_`, or that `compressor->Compress()` performs no staging/allocation/sync inside the timed window.

## ORCH-7: SGD features are recomputed with a second full pass over the chunk (host, by default) instead of reusing the device stats — and with a different element-type interpretation than inference used
- **Native evidence:** `gpucompress_compress.cpp:220-222` `/* Pre-computed stats live in infer_ctx->d_stats - pass to _with_action_gpu so it can reuse them for SGD without recomputing. */`; `:630-637` D2D copy; SGD consumes `d_stats_ptr` directly at `:721-723`.
- **Port evidence:** `compressor_runtime.cc:1352-1366` `ctp::DataType feat_type = (context.data_type_ == 1) ? FLOAT32 : UINT8; ... ComputeCompressionFeatures(chunk_data, feat_num_elements, feat_type, ...);` vs the inference-side type at `:569-573` `neuropress_active ? FLOAT32 : ((context.data_type_ == 1) ? FLOAT32 : UINT8)`. With the default `data_type_ == 0` (`core_tasks.h:1637`) inference features are float32-typed and SGD features uint8-typed over the same bytes.
- **Category:** added-host-roundtrip / host-compute
- **Severity:** medium
- **Reachability:** `DynamicSchedule`, gated on `config_.neuropress_online_learning_enabled_` (default `false`, `compressor_tasks.h:73`).
- **How to falsify:** Show the device stats buffer from `EstCompressionStats` being reused here, or `feat_type` resolving to FLOAT32 for the same chunks.

## ORCH-8: Exploration is serialized on one stream, timed with a host clock, and its winner is staged D2H and re-put through host SHM — native runs K alternatives on K parallel streams with per-slot CUDA events and writes the winner D2D
- **Native evidence:** `gpucompress_compress.cpp:744-747` (`/* Parallel exploration: K alternatives on K separate streams */`); `:797` `cudaStreamCreate(&s.stream)`; `:801-802` `cudaEventCreate`; `:868-871` event-bracketed compress; `:884-891` `cudaStreamSynchronize(s.stream); ... cudaEventElapsedTime(...)`; winner write D2D at `:946-949`.
- **Port evidence:** `compressor_runtime.cc:1615` `for (const auto* alt : alternatives) {` — sequential; each `alt_compressor->Compress(...)` (`:1782-1784`) uses nvcomp's single per-thread `CachedStream()` (`nvcomp.h:180`, `:359-370`). `:1781-1788` host chrono timing. `:1818-1825` winner staged to host via `DeviceAwareMemcpy`. `:1931-1944` re-put through host SHM with three `std::memcpy`s. The port's comment at `:1565-1570` acknowledges the serialization but not that the per-candidate *time* — the quantity the winner is chosen on (`:1802-1804`) and trained on (`:1885-1888`) — also changed clock domain.
- **Category:** host-compute (timing) + added-host-roundtrip; serialization noted separately
- **Severity:** medium
- **Reachability:** `DynamicSchedule:1571-1573`, gated on `config_.neuropress_exploration_enabled_` (default `false`, `compressor_tasks.h:103`) AND `error_pct > 0.50`. Off by default on both sides (`gpucompress_api.cpp:83` `g_exploration_enabled{false}`), so this is a fidelity gap in an opt-in mode.
- **How to falsify:** Show K concurrent CUDA streams in the port's exploration loop, or `alt_time_ms` derived from CUDA events.

## ORCH-9: The interposer write path `CompressIntoShm()` has no device branch at all — quantize, shuffle and codec input are unconditionally host
- **Native evidence:** `gpucompress_compress.cpp:434-470` device quantize + shuffle on `stream`; `:534` `compressor->compress(d_compress_input, d_comp_target, comp_config);` both pointers device-resident.
- **Port evidence:** `compressor_runtime.cc:3212-3351`: `:3242-3246` comment "This path is host-side"; `:3262-3264` host `Quantize<float>`; `:3277-3281` host `ByteShuffle`; `:3289-3292` `std::vector<char> compressed(...); compressor->Compress(compressed.data(), ...)` — host destination, so nvcomp D2Hs the result; `:3323-3328` three host `std::memcpy`s into SHM.
- **Category:** host-compute / missing-gpu-work
- **Severity:** high
- **Reachability:** `Runtime::PutBlob:3506-3554` -> `:3529-3531`, for any whole-blob put with `ctx.compress_lib_ > 0`, `replica_ == 0`, `!emulate_`. Also per-record from `MultiPutBlob:3721-3733`.
- **How to falsify:** Find a device branch inside `CompressIntoShm`, or show `src` can never derive from device-resident data in any deployment using the interposer.

## ORCH-10: The interposer read path `DecompressStored()` performs byte-unshuffle and dequantization entirely on the host; native does both on device
- **Native evidence:** `gpucompress_compress.cpp:1256-1305` — `byte_unshuffle_simple(..., stream)`, `dequantize_simple(d_result, qr, stream)`, then D2D `cudaMemcpyAsync`.
- **Port evidence:** `compressor_runtime.cc:3464-3499`. `:3481` `qr.quantized_.assign(quant_staging.begin(), quant_staging.end());` is an extra full host copy. `Runtime::GetBlob:3609-3646` adds another host `std::memcpy` per region (`:3643`).
- **Category:** host-compute
- **Severity:** high
- **Reachability:** `Runtime::GetBlob:3556-3656`, every read of a blob carrying `kBlobTransformCompressed` through the interposer.
- **How to falsify:** Show `ByteUnshuffleDevice`/`DequantizeDevice` called from `DecompressStored`, or show this path is dead in favour of `Runtime::Decompress`.

## ORCH-11: `Runtime::Decompress` stages the whole compressed stream through host SHM and reads the header with a host dereference; native reads a 64-byte header D->H and keeps the payload device-resident
- **Native evidence:** `gpucompress_compress.cpp:1188-1192` `/* Read header from GPU (64B D->H) */ ... readHeaderFromDevice(d_input, header, stream); ... if (!header.isValid()) return GPUCOMPRESS_ERROR_INVALID_HEADER;`, then `:1204-1205` payload pointer stays on device through `decompressor->decompress` at `:1246`.
- **Port evidence:** `compressor_runtime.cc:2655-2666` allocates host SHM and `AsyncGetBlob`s the whole blob into it; `:2694` `auto* header = reinterpret_cast<CompressionHeader*>(temp_buffer.ptr_);`; `:2787` `char* compressed_data = temp_buffer.ptr_ + header_size;` handed to `decompressor->Decompress(...)` at `:2833-2835`, which H2D-stages it again (`nvcomp.h:271`).
- **Category:** added-host-roundtrip
- **Severity:** medium
- **Reachability:** Every `Runtime::Decompress` call.
- **How to falsify:** Show `AsyncGetBlob` can deliver into a device-backed buffer here.

## ORCH-12: Native's pre-allocated per-slot device scratch (`CompContext`) has no port equivalent — per-chunk device alloc/free plus per-chunk host `std::vector` intermediates
- **Native evidence:** `src/api/internal.hpp:32-88` — `CompContext` holds stream, six events, `d_stats_workspace`, `d_stats`, `d_histogram`, `d_fused_infer_output`, `d_fused_top_actions`, `d_fused_costs`, `d_sgd_grad_buffer`, `d_sgd_output`, `d_sgd_samples`, `d_range_min/max`, `d_preproc_quant`, `d_preproc_shuffle`, `d_cub_temp`, and an LRU-3 nvcomp manager cache; `:61-64` `/* P1: Pre-allocated preprocessing buffers. Eliminates per-chunk cudaMalloc/cudaFree in quantize + shuffle. */`; `:26` `N_COMP_CTX = 9`; acquired at `gpucompress_compress.cpp:369`.
- **Port evidence:** `compressor_runtime.cc:2204-2215` — the port's only per-chunk device state is a three-slot RAII guard over `AllocateAndRegisterGpuBackend` ids, allocated at `:2244-2246`, `:2311-2313`, `:2352-2354` and freed at scope exit, every chunk. Host per-chunk buffers at `:2169`, `:2178`, `:2193`, `:2349`. `compressor_runtime.h:264-278` shows the runtime's only persistent state is the host predictors.
- **Category:** device-struct-became-host
- **Severity:** medium
- **Reachability:** Every `Runtime::Compress`.
- **How to falsify:** Find a pooled device-scratch object equivalent to `CompContext`.

## ORCH-13: The port writes the compressed header with a blocking `cudaMemcpy` on the default stream where native uses `cudaMemcpyAsync` on the context stream
- **Native evidence:** `gpucompress_compress.cpp:610-612` `cudaMemcpyAsync(d_out, &header, sizeof(CompressionHeader), cudaMemcpyHostToDevice, stream);` — sync deferred to `:1040`.
- **Port evidence:** `compressor_runtime.cc:2467-2473` two `ctp::GpuApi::Memcpy(...)` calls; `GpuApi::Memcpy` is `cudaMemcpy(dst, src, size, cudaMemcpyDefault)` on the legacy default stream (`gpu_api.h:250-258`). Also `:2889-2890` a full-size blocking D2D copy-back after the device unshuffle, where native swaps pointers and copies once at the end.
- **Category:** added-host-roundtrip
- **Severity:** low
- **Reachability:** Every device-path compress; every device-path shuffled decompress.
- **How to falsify:** Show `GpuApi::Memcpy` resolving to an async, stream-scoped copy in the build under audit.

## ORCH-14: Native's per-chunk diagnostics (CUDA-event breakdown, 40+ fields, `DiagnosticsStore`) have no port equivalent; the port records host wall-clock telemetry only
- **Native evidence:** `gpucompress_compress.cpp:1072-1165` builds `ChunkDiagInput` with `nn_inference_ms`/`stats_ms` from `cudaEventElapsedTime` (`:217-218`), `compression_ms`/`compression_ms_raw` from `cudaEventElapsedTime` (`:562`), plus `regret`, `ratio_mape`, `top_actions` (32), `predicted_costs`, `explore_alternatives[31]`, then `:1159` `recordChunkDiagnostic(di)`. Accessors at `gpucompress_diagnostics.cpp:70-95`. Native's host-phase timers are explicitly host (`:44-49` `DT_START`/`DT_MS`) — the event/clock split is deliberate.
- **Port evidence:** `compressor_runtime.cc:2518-2522` a 7-field `CompressionTelemetry` with `compress_time` (host clock), and `LogCompressionTelemetry:3005-3027` whose in-memory sink is a TODO (`:3007-3010`). No per-chunk store, no regret/MAPE record, no predicted-ranking record, no CUDA-event field. The NN's own inference time is host-clocked and divided by batch size (`neuropress_nn_predictor.cc:512-515`, `:598-600`) and never consumed by the runtime.
- **Category:** missing-gpu-work (diagnostics fidelity)
- **Severity:** medium
- **Reachability:** Always.
- **How to falsify:** Find a per-chunk diagnostics store in the port with device-timed fields.

## ORCH-15: Diagnostic logging stages the whole chunk and the whole compressed payload D->H; native's debug diagnostics copy 24 bytes of stats
- **Native evidence:** `gpucompress_compress.cpp:1106-1115` — one `cudaMemcpy(&h_stats, d_stats_ptr, sizeof(AutoStatsGPU), cudaMemcpyDeviceToHost)`.
- **Port evidence:** `compressor_runtime.cc:1411-1425` stages the full chunk then a byte loop; `:1119-1130` in `LogCompressedPayload` stages the full payload. `DeviceAwareMemcpy` does `cudaMemcpyAsync` + `cudaStreamSynchronize` per call (`gpu_api.h:519-524`).
- **Category:** added-host-roundtrip
- **Severity:** low
- **Reachability:** Only when `CLIO_NEUROPRESS_SELECTION_LOG` is set (`:1165-1171`); honestly documented as diagnostic cost at `:1403-1409`.
- **How to falsify:** Show `SelectionLogEnabled()` defaulting on, or the staging being skipped.

## ORCH-16: The GPU selection path is off by default and the default fallback is a CPU-only candidate list; native's default config is a GPU codec and `ALGO_AUTO` without weights is a hard error
- **Native evidence:** `gpucompress_api.cpp:316-324` `config.algorithm = GPUCOMPRESS_ALGO_LZ4;` — nvcomp GPU LZ4 (`internal.hpp:114`). `:240-244` a failed weight load only warns; the refusal comes unconditionally at `gpucompress_compress.cpp:208-213` for `ALGO_AUTO`. Non-AUTO explicit algorithms bypass the NN and still run on device (`:178-187`).
- **Port evidence:** `compressor_tasks.h:64` `std::string neuropress_model_path_;` default empty; `Runtime::Create` constructs the predictor only `if (!config_.neuropress_model_path_.empty())` (`compressor_runtime.cc:408`), with `:428-432` logging at `kDebug` only. With no predictor, `EstCompressionStats:551-553` sets `neuropress_active = false`, the ranking block at `:670-671` is skipped, and `:744-750` supplies the CPU-only list with `:792-797` inventing `pred.compression_ratio = 2.0`. `neuropress_online_learning_enabled_ = false` (`:73`) and `neuropress_exploration_enabled_ = false` (`:103`) — the latter matches native's `g_exploration_enabled{false}`, so exploration-off is parity; model-path-off is not, because native's alternative to the model is still a GPU codec.
- **Category:** unreachable-device-path
- **Severity:** medium
- **Reachability:** Default configuration.
- **How to falsify:** Show a deployment default (compose YAML, `CreateParams`) that populates `neuropress_model_path_`, or the fallback candidate list resolving to GPU wire ids.

---

## Subcategories that yielded nothing

- **Native GPU work stubbed out in the port with a `return false`/no-op:** none in the orchestration layer. The device entry points do have unconditional-`false` stubs, but only in the explicitly CPU-only TU (`byte_shuffle_cpu_stub.cc:24-37`, guarded by `#if !defined(CTP_ENABLE_CUDA) || !CTP_ENABLE_CUDA`).
- **A global "CUDA init failed -> route everything to CPU" switch:** no single such flag exists. The effect is emergent, covered by ORCH-2, ORCH-1/ORCH-3 (`ctp::IsDevicePointer` false — including when `cudaPointerGetAttributes` fails on a driverless host, `gpu_api.h:501-511`) and ORCH-4. Nothing logs "CUDA unavailable, compressing on CPU" at runtime.
- **Header/format divergences, preset packing, quantization-extension layout, PSNR sentinel handling:** examined, not reported — real differences, no execution-locality consequence.
- **False-parity comment check:** the port's cited upstream line numbers are, with one exception, accurate. Verified: `:157`<->`:37-41`; `:599`<->`:274-275`; `:628`<->`:285-286`; `:1598`<->`:912`; `:1796`<->`:904-905`; `:2190`<->`:440-452`; `:2218`<->`:433-470`; `:2410`<->`:642-643`; `:2422`<->`:388`,`:653-658`; `:2681-2683`<->`:1192`; `:2937`<->`H5VLgpucompress.cu:3302-3311`. `:606` cites `:281` for "Stats remain on GPU" where the comment is at `:282` — off-by-one. The one comment whose *claim* does not survive checking the code beneath it is `:2368-2373` — that is ORCH-6.
- **`neuropress_bridge.cc:229` `dynamic_cast` device-path guard:** if the cast failed while `device_stats != nullptr`, `RankIntoStats` would silently rank on all-zero statistics (`:321-322`). Not reported: the member is declared as the concrete type (`compressor_runtime.h:277-278`), so the cast cannot fail today. Latent hazard, not live.
