# Preprocessing (quantization + byte shuffle): NATIVE vs PORT execution-locality audit

NATIVE `/home/cc/NeuroPress` @ b23b8f6 — PORT `/home/cc/clio-core` @ neuropress-693-continued

**14 findings.** Native ground truth established first: NeuroPress has **no host implementation of quantize, dequantize, shuffle or unshuffle anywhere** (`src/hdf5/H5Zgpucompress.c:224-269` only passes parameters). Native quantize = CUB `DeviceReduce::Min/Max` (`quantization_kernels.cu:204-218`) + `quantize_linear_kernel` (`:55-81`, launched `:341`); dequantize = `dequantize_linear_kernel` (`:84-99`) with **no sync at all** (`:861-863` "S4 fix: sync removed"); shuffle/unshuffle = `byte_shuffle_kernels.cu:285-330` / `:332-375`. Native allocates nothing per chunk: `gpucompress_pool.cpp:128-142` pre-allocates range scalars, 16 MB quant scratch, 16 MB shuffle scratch and 64 KB CUB temp per slot, passed at `gpucompress_compress.cpp:442-444`/`:462`, on a per-context stream created once (`gpucompress_pool.cpp:101`, used `gpucompress_compress.cpp:373`).

## PRE-1: Runtime::Compress quantizes on the CPU (min/max scan included) for host-resident chunks
- **Claim:** For a host-resident chunk the port runs the entire quantizer in host C++ — serial min/max scan plus serial per-element round/clamp — where native has only a CUB device reduction plus a kernel. Native has no host quantizer to correspond to.
- **Native evidence:** `src/preprocessing/quantization_kernels.cu:204` `cub::DeviceReduce::Min(nullptr, temp_bytes, d_input, d_buf_min, (int)num_elements, stream)`; `:341` `quantize_linear_kernel<InputT, OutputT><<<num_blocks, BLOCK_SIZE, 0, stream>>>(...)`. Sole call site `src/api/gpucompress_compress.cpp:440-445`.
- **Port evidence:** `context-transfer-engine/compressor/src/compressor_runtime.cc:2268` `} else if (want_quant) {` -> `:2277` `auto host_q = ctp::compress::preprocess::Quantize<float>(`. Body: `context-transport-primitives/include/clio_ctp/compress/preprocess/quantization.h:136-139` `for (size_t i = 1; i < num_elements; ++i) { if (data[i] < min_val) min_val = data[i]; ... }` and `:187-204` the quantize loop.
- **Category:** host-compute
- **Severity:** high
- **Reachability:** `want_quant` (`:2239-2241`: quantize bit + `context.error_bound_ > 0.0` + size % 4 == 0) **and** `IsDevicePointer(input_ptr)` false at `:2242`. Host pointers arise from host-SHM blobs or from `StageInputIfNeeded` having staged D2H for a CPU codec (`compress_factory.h:239-247`, called `compressor_runtime.cc:2170`). Same host branch in exploration at `:1682-1703`.
- **How to falsify:** show `:2268` is unreachable (every eb>0 chunk is device-resident), or find a host quantizer NeuroPress actually calls.

## PRE-2: Runtime::Compress byte-shuffles on the CPU for host-resident chunks
- **Claim:** The port has two shuffle implementations selected by pointer kind; the host one is a triple-nested scalar loop. Native has only the kernel.
- **Native evidence:** `src/preprocessing/byte_shuffle_kernels.cu:314` `err = launch_byte_shuffle(const_cast<const uint8_t**>(arrays.d_input_ptrs), ...)`; only caller `gpucompress_compress.cpp:459-463`.
- **Port evidence:** `compressor_runtime.cc:2324-2332` `} else { shuffle_staging.resize(compress_input_size); if (ctp::compress::preprocess::ByteShuffle(...))`. Body: `include/clio_ctp/compress/preprocess/byte_shuffle.h:96-101`.
- **Category:** host-compute
- **Severity:** high
- **Reachability:** `shuffle_elem != 0` (`:2303`) + `IsDevicePointer` false (`:2304`). Needs **no** error bound, and shuffle is the dimension the model picks most (port's own comment `:80-85`). Every chunk of `neuropress_compress_dir.cc` takes it (D2H at `:173-175`, host SHM at `:201-203`). Same branch in exploration `:1736-1746`.
- **How to falsify:** counter in `ByteShuffle` vs `ShuffleKernel` during a VOL/compress_dir run showing the device branch taken.

## PRE-3: The interposed PutBlob path (`CompressIntoShm`) has no device branch at all
- **Claim:** `Runtime::CompressIntoShm` always quantizes and shuffles on the host — no `IsDevicePointer` test, no device call anywhere in the function.
- **Native evidence:** `gpucompress_compress.cpp:434-470` — single preprocessing block, both steps device-only.
- **Port evidence:** `compressor_runtime.cc:3260-3262` `if (requested_quant && ctx.error_bound_ > 0.0 && ...) { quant_result = ctp::compress::preprocess::Quantize<float>(`; `:3276-3283` `if (requested_shuffle != 0) { ... ByteShuffle(`. Comment at `:3241-3243` states the assumption.
- **Category:** host-compute
- **Severity:** high
- **Reachability:** `Runtime::PutBlob` -> `:3529-3530` `CompressIntoShm(ctx, src_full.ptr_, task->size_, ...)`, for every whole-blob put with `compress_lib_ > 0` (`:3514-3517`).
- **How to falsify:** show `Runtime::PutBlob` is never interposed in a NeuroPress deployment.

## PRE-4: `DecompressStored` inverts both transforms on the host, with three extra full-size host buffers
- **Claim:** The interposer read path unshuffles and dequantizes in host loops and copies the payload three extra times (`quant_staging` -> `qr.quantized_` -> `restored` -> `dst`); native does both with kernels into device buffers.
- **Native evidence:** `gpucompress_compress.cpp:1257` `d_unshuffled = byte_unshuffle_simple(d_decompressed, decompressed_size, header.shuffle_element_size, ...)`; `:1284` `d_dequantized = dequantize_simple(d_result, qr, stream);`.
- **Port evidence:** `compressor_runtime.cc:3466` `if (!ctp::compress::preprocess::ByteUnshuffle(`; `:3481` `qr.quantized_.assign(quant_staging.begin(), quant_staging.end());`; `:3488-3489` `std::vector<float> restored = ctp::compress::preprocess::Dequantize<float>(qr);`; `:3498` `std::memcpy(dst, restored.data(), ...)`. Loops at `quantization.h:248-265`, `byte_shuffle.h:158-166`.
- **Category:** host-compute
- **Severity:** high
- **Reachability:** `Runtime::GetBlob` interposer -> `:3616` `int rc = DecompressStored(stored.ptr_, stored_size, scratch.ptr_, ...)`, for any read of a `kBlobTransformCompressed` blob (`:3563-3566`).
- **How to falsify:** show GetBlob interposition is disabled in the deployed config.

## PRE-5: Runtime::Decompress unshuffles on the host when the destination is host memory
- **Claim:** The main decompress path keeps a host unshuffle branch plus a `std::memcpy` back over the codec output; native's decompress has one path and it is the kernel.
- **Native evidence:** `gpucompress_compress.cpp:1256-1262` `if (header.hasShuffleApplied()) { d_unshuffled = byte_unshuffle_simple(...); if (d_unshuffled) { d_result = d_unshuffled; }` — no host alternative.
- **Port evidence:** `compressor_runtime.cc:2895-2902` `} else { std::vector<char> unshuffled(decompressed_size); if (ByteUnshuffle(...)) { std::memcpy(codec_dst, unshuffled.data(), decompressed_size); }`.
- **Category:** host-compute
- **Severity:** medium
- **Reachability:** `:2867` shuffle recorded + `:2868` `IsDevicePointer(codec_dst)` false — e.g. all 64 verification reads in `neuropress_compress_dir.cc:297-302`.
- **How to falsify:** show `codec_dst` is always device-resident here.

## PRE-6: Every port device preprocessing kernel runs on the NULL stream and blocks the host
- **Claim:** All four device entry points hardcode `stream = 0` then synchronize, serializing against every other blocking stream (codec, NN inference, sibling workers). Native enqueues on a per-context stream created once.
- **Native evidence:** `src/api/gpucompress_pool.cpp:101` `if (cudaStreamCreate(&ctx.stream) != cudaSuccess) goto fail;`; `gpucompress_compress.cpp:373` `cudaStream_t stream = ctx->stream;` passed at `:445`, `:461`.
- **Port evidence:** `context-transport-primitives/src/compress/preprocess/data_stats_gpu_kernels.cu:539` `cudaStream_t stream = 0;` (again `:559`); `:526` `return cudaStreamSynchronize(stream) == cudaSuccess;`; `:693` `MinMaxKernel<<<grid, kBlockSize>>>(...)` and `:766-777`, `:807-817` — all launched with no stream argument. No overload accepts a stream (`quantization.h:306-319`, `byte_shuffle.h:232-237`).
- **Category:** added-host-roundtrip
- **Severity:** medium
- **Reachability:** every device call: `compressor_runtime.cc:1673`, `:1726`, `:2249`, `:2315`, `:2879`, `:2918`.
- **How to falsify:** show no other stream is ever in flight concurrently, or find a stream threaded in.

## PRE-7: Port adds `cudaDeviceSynchronize()` where native syncs one stream — and where native syncs nothing
- **Claim:** `DeviceMinMax`, `QuantizeDevice` and `DequantizeDevice` each end in a device-wide sync. Native's quantize has exactly one `cudaStreamSynchronize(stream)`; native's dequantize has none, deliberately.
- **Native evidence:** `quantization_kernels.cu:225` `err = cudaStreamSynchronize(stream);` (only sync in `quantize_simple`); `:861-863` `/* S4 fix: sync removed - callers use result on same stream... */ return d_output;`.
- **Port evidence:** `data_stats_gpu_kernels.cu:694-695` `ok = cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;`; `:779` `if (cudaDeviceSynchronize() != cudaSuccess) return false;`; `:820` `return cudaDeviceSynchronize() == cudaSuccess;`.
- **Category:** added-host-roundtrip
- **Severity:** medium
- **Reachability:** two device-wide syncs per quantized chunk on write, one per chunk on read (`:1673`, `:2249`, `:2918`).
- **How to falsify:** show device-wide sync costs the same as native's stream sync here.

## PRE-8: Min/max reduction allocates, blocking-copies and frees per call
- **Claim:** `DeviceMinMax` does `cudaMalloc` + blocking H2D + kernel + `cudaDeviceSynchronize` + blocking D2H + `cudaFree` per chunk; native's production overload uses pre-allocated scalars and a pre-allocated CUB temp with `cudaMemcpyAsync` and one stream sync, allocating nothing.
- **Native evidence:** `quantization_kernels.cu:253-254` `if (d_cub_temp && cub_temp_cap >= temp_bytes) { d_temp = d_cub_temp; }`; `:270-274` `cudaMemcpyAsync(&h_min, d_buf_min, sizeof(T), cudaMemcpyDeviceToHost, stream); ... cudaStreamSynchronize(stream);`; buffers from `gpucompress_pool.cpp:128-142`.
- **Port evidence:** `data_stats_gpu_kernels.cu:684` `cudaMalloc(&d_res, 2 * sizeof(unsigned int))`; `:686` H2D `cudaMemcpy`; `:699` D2H `cudaMemcpy`; `:701` `cudaFree(d_res);`.
- **Category:** added-host-roundtrip
- **Severity:** medium
- **Reachability:** `QuantizeDevice:721` `if (!DeviceMinMax(in, num_elements, &data_min, &data_max)) return false;` — every device quantize.
- **How to falsify:** find a pre-allocated path in the port, or show the 8-byte malloc/copy/free is free relative to the kernel.

## PRE-9: Per-chunk device scratch allocation/free where native pre-allocates once
- **Claim:** Each device quantize/shuffle allocates a fresh device buffer via `AllocateAndRegisterGpuBackend` (cudaMalloc + backend registration) and frees via `FreeGpuBackend` (cudaFree — itself device-synchronizing). Native passes pre-allocated per-slot scratch.
- **Native evidence:** `gpucompress_pool.cpp:138-142` `cudaMalloc(&ctx.d_preproc_quant, kPreallocChunk) ... cudaMalloc(&ctx.d_preproc_shuffle, kPreallocChunk) ... cudaMalloc(&ctx.d_cub_temp, kCubTemp)` (`kPreallocChunk = 16UL << 20`, `:135`); consumed `gpucompress_compress.cpp:443`, `:462`; `quantization_kernels.cu:747-753` and `byte_shuffle_kernels.cu:392-399` set `owns_output = false`.
- **Port evidence:** `compressor_runtime.cc:2244-2246` `quant_device_alloc = CLIO_IPC->AllocateAndRegisterGpuBackend(/*gpu_id=*/0, ...kDeviceMem, input_size, &quant_buf);`; `:2311-2313` shuffle; frees `:2209`, `:2262`, `:2320`; decompress `:2817-2819`, `:2874-2877`; exploration `:1665-1669`, `:1720-1724`. Implementation `context-runtime/src/ipc_manager.cc:3847-3851` + registration `:3867-3885`.
- **Category:** added-host-roundtrip
- **Severity:** medium
- **Reachability:** every device-path quantize/shuffle/unshuffle, xK per chunk with exploration on.
- **How to falsify:** show these allocations are pool-served and never reach cudaMalloc/cudaFree.

## PRE-10: Decompress device unshuffle adds a blocking full-buffer D2D copy native avoids
- **Claim:** After `ByteUnshuffleDevice` writes scratch, the port copies the whole buffer back over `codec_dst` with the blocking `cudaMemcpy` wrapper; native just retargets the pointer.
- **Native evidence:** `gpucompress_compress.cpp:1261-1262` `if (d_unshuffled) { d_result = d_unshuffled; }` — no copy.
- **Port evidence:** `compressor_runtime.cc:2889-2890` `ctp::GpuApi::Memcpy(codec_dst, scratch, decompressed_size);` = `include/clio_ctp/util/gpu_api.h:254` `CUDA_ERROR_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDefault));`.
- **Category:** added-host-roundtrip
- **Severity:** low
- **Reachability:** device-resident read of a shuffled blob (`:2867-2868` both true).
- **How to falsify:** show `codec_dst` cannot be retargeted (a later consumer holds it), making the copy necessary.

## PRE-11: The quantized payload became a host `std::vector` in the port's non-device result type
- **Claim:** Native's `QuantizationResult` holds `void* d_quantized` — the payload never leaves the device. The port's host-path result holds `std::vector<uint8_t>`, and every use copies it again.
- **Native evidence:** `src/preprocessing/quantization.cuh:110` `void* d_quantized;              // Device pointer to quantized data`.
- **Port evidence:** `include/clio_ctp/compress/preprocess/quantization.h:48` `std::vector<uint8_t> quantized_;`; extra copies at `compressor_runtime.cc:2281-2282`, `:3266-3267`, `:1690-1691`, `:3481`.
- **Category:** device-struct-became-host
- **Severity:** medium
- **Reachability:** all host quantize paths (PRE-1, PRE-3) and `DecompressStored` (PRE-4). The device path uses `DeviceQuantizeParams` (`quantization.h:278-286`), scalars only — that part is fine.
- **How to falsify:** show the host `QuantizationResult` is never instantiated live (which would convert PRE-1/PRE-3 into dead-code findings instead).

## PRE-12: The ported device quantize/dequantize kernels are unreachable through the HDF5 VOL — no error bound is plumbed anywhere
- **Claim:** `QuantizeDevice`/`DequantizeDevice` need `context.error_bound_ > 0.0`, and nothing in the runtime or VOL sets it: the VOL passes a default-constructed `Context`, and `CompressorConfig` has no error-bound field. Native exposes the bound through HDF5 filter parameters.
- **Native evidence:** `src/hdf5/H5Zgpucompress.c:264-269` `config.error_bound = unpack_double(cd_values[3], cd_values[4]); if (config.error_bound > 0.0) { config.preprocessing |= GPUCOMPRESS_PREPROC_QUANTIZE; }` -> `gpucompress_compress.cpp:434`.
- **Port evidence:** `context-transfer-engine/adapter/hdf5_vol/clio_vol.cc:1902` `..., blob_data, -1.0f, clio::cte::core::Context(), 0, cte_client->pool_id_);` (also `:1908`, `:2480`, `:2487`); `error_bound_(0.0)` at `context-transfer-engine/core/include/clio_cte/core/core_tasks.h:1633`. The only non-test assignment in the tree is `compressor/example/neuropress_gpu_direct.cc:148` `ctx.error_bound_ = kErrorBound;` (env `CLIO_NEUROPRESS_ERROR_BOUND`, default 0.0, `:64-67`). Gate: `compressor_runtime.cc:2239`.
- **Category:** unreachable-device-path
- **Severity:** medium
- **Reachability:** none via VOL or `neuropress_compress_dir`; only `neuropress_gpu_direct` with the env var, or `parity/neuropress_preprocess_parity.cu:300`. Note native also masks quantize at eb=0 — the divergence is that the port has *no knob at all*.
- **How to falsify:** find any production caller or config path that sets `Context::error_bound_`.

## PRE-13: The ported shuffle kernels are bypassed for every host-SHM chunk, including the whole `neuropress_compress_dir` pipeline
- **Claim:** `ShuffleKernel`/`UnshuffleKernel` are faithful device ports, but the current end-to-end example never executes them — its chunks are staged D2H before reaching the compressor, so PRE-2/PRE-5 host loops run instead.
- **Native evidence:** `gpucompress_compress.cpp:457-459` `unsigned int shuffle_size = GPUCOMPRESS_GET_SHUFFLE_SIZE(preproc_to_use); if (shuffle_size > 0) { d_shuffled = byte_shuffle_simple(` — unconditional, no pointer-kind test in the file.
- **Port evidence:** `adapter/hdf5_vol/example/neuropress_compress_dir.cc:173-175` `CUDA_CHECK(cudaMemcpy(source.data(), d_data, kDatasetBytes, cudaMemcpyDeviceToHost));` then `:201-203` `auto in_buf = CLIO_IPC->AllocateBuffer(kChunkBytes); ... std::memcpy(in_buf.ptr_, source.data() + c * kChunkBytes, kChunkBytes);` -> `compressor_runtime.cc:2304` false -> `:2326` host `ByteShuffle`. The device branch *is* live from the VOL proper when the app's H5Dwrite buffer is device memory and a compressor pool is set (`clio_vol.cc:1827-1828`, `:1838-1846`) and from `neuropress_gpu_direct`.
- **Category:** unreachable-device-path
- **Severity:** medium
- **How to falsify:** counter inside `ShuffleKernel` firing during a `neuropress_compress_dir` run.

## PRE-14: `Runtime::Decompress` calls `DequantizeDevice` unconditionally, including on host destinations
- **Claim:** Unlike the unshuffle just above it, the dequantize has no pointer-kind branch and no host fallback: the kernel is handed `output_fullptr.ptr_`, host SHM for any host consumer. Native always has a device destination; the port answers the host case by failing the read (`rc = 5`) rather than doing the work anywhere.
- **Native evidence:** `gpucompress_compress.cpp:1284` `d_dequantized = dequantize_simple(d_result, qr, stream);` with device buffers throughout (`:1242`, `:1299`).
- **Port evidence:** `compressor_runtime.cc:2918-2920` `if (!ctp::compress::preprocess::DequantizeDevice(codec_dst, quant_elems, stored_quant_params, output_fullptr.ptr_)) {` — compare `:2868` `if (ctp::IsDevicePointer(codec_dst)) {` for the unshuffle, which *does* branch. `output_fullptr` from `:2793-2794`; host SHM in `neuropress_compress_dir.cc:297-302`.
- **Category:** missing-gpu-work
- **Severity:** medium
- **Reachability:** any `AsyncDecompressExplicit` of a quantized blob into a host buffer; latent today because PRE-12 keeps the write-side gate closed.
- **How to falsify:** show `output_fullptr.ptr_` is always device-resident here.

---

## Subcategories that yielded NOTHING (explicit)

1. **`byte_shuffle_cpu_stub.cc` is NOT a silent CPU fallback.** Whole body inside `#if !defined(CTP_ENABLE_CUDA) || !CTP_ENABLE_CUDA` (`:22`); `CTP_ENABLE_CUDA` from `CLIO_CORE_ENABLE_CUDA` via `context-transport-primitives/CMakeLists.txt:279-284`, and `CLIO_CORE_ENABLE_CUDA:BOOL=ON` in `build/CMakeCache.txt`. **Verified empirically:** `nm -C build/bin/libclio_ctp_compress_model.a` shows the four `*Device` symbols defined (`T`) only in `data_stats_gpu_kernels.cu.o`; the extracted `byte_shuffle_cpu_stub.cc.o` defines no functions at all. On a CPU-only build the stubs return `false` and callers treat that as "don't do it" (`:2319-2322`) or log an error (`:2882-2887`, `:2921-2926`) — never a silent host substitution. The filename is misleading; the code is correct.
2. **Quantization PARAMETERS are not host-computed on the port's device path.** Native computes min/max on device then derives scale/effective_eb/precision on the host (`quantization_kernels.cu:407-486`); the port does the same via `MinMaxKernel` (`data_stats_gpu_kernels.cu:637-662`, called `:693`) plus identical host arithmetic at `:739-758`. No divergence.
3. **No "stage device -> host, preprocess, push back" round trip exists.** Every host branch is entered on a pointer that was already host, or one `StageInputIfNeeded` had staged D2H *for the codec*.
4. **The port's device kernels themselves are faithful** — `ShuffleKernel`/`UnshuffleKernel` (`data_stats_gpu_kernels.cu:458-507`) and `QuantizeKernel`/`DequantizeKernel` (`:594-618`) contain no hidden host loops, same 256 KiB chunked plane layout (`byte_shuffle.h:47` vs `internal.hpp:99`), same clamps, same arithmetic. The upstream citations in `quantization.h:143-147, 151-157, 182-185, 237-240` and `byte_shuffle.h:36-45, 79-84, 102-104` were each checked and are accurate. (One stale doc reference, no execution consequence: `byte_shuffle.h:215` says the kernels live in `byte_shuffle_gpu_kernels.cu`; they are in `data_stats_gpu_kernels.cu`.)
5. **No missing GPU work in the shuffle direction** — both device shuffle entry points exist and are wired to real call sites in the CUDA build.
