# Stats / feature-extraction audit: NATIVE (/home/cc/NeuroPress, b23b8f6) vs PORT (/home/cc/clio-core)

**8 findings.** Scope: entropy, MAD, second-derivative mean, min/max/mean/stddev.

## Per-statistic map

| Statistic | Native kernel + reduction | Port device-resident path | Port `ComputeDeviceStats` path | Port host path |
|---|---|---|---|---|
| byte histogram | `histogramKernel`/`histogramKernelVec4`, per-warp shared privatization -> global atomics (`entropy_kernel.cu:48,99`) | fused into `StatsPass1Kernel`, per-**block** shared hist -> global atomics (`data_stats_gpu_kernels.cu:32-88`) | same kernel | host loop (`data_stats.h:100-103`) |
| Shannon entropy | `entropyFromHistogramKernel`, 1 block/256 threads, shared tree reduction, writes `&d_stats->entropy` (`entropy_kernel.cu:167-202`, launched `stats_kernel.cu:337`) | `EntropyFromHistKernel`, same shape (`data_stats_gpu_kernels.cu:155-175`, launched :254) | **host serial loop over a D2H'd histogram** (:322-329) — STATS-1 | host serial loop (`data_stats.h:106-113`) |
| sum (-> mean) | `statsPass1Kernel` warp-shuffle -> shared -> `atomicAdd(&stats->sum)` (`stats_kernel.cu:139-189`) | `StatsPass1Kernel` shared tree -> `atomicAdd(sum_out)` (:69-87) | same kernel, **then D2H'd and divided on host** (:304-308) — STATS-2 | host loop (`data_stats.h:131-134`) |
| MAD | `madPass2Kernel`, mean read **on device** from `stats->sum` (`stats_kernel.cu:201-213`), normalized by `finalizeStatsOnlyKernel` (:260-281) | `StatsPass2DevKernel` reads mean on device (:120-144), `FinalizeFeatureStatsKernel` (:184-194) | `StatsPass2Kernel` takes the mean as a **host argument** (:91-108, launched :309) | host two-pass loop (`data_stats.h:126-143`) |
| 2nd derivative | fused into `statsPass1Kernel`, `atomicAdd(&stats->abs_diff_sum)` (`stats_kernel.cu:133-136,186`) | fused into `StatsPass1Kernel` (:59-67) | same kernel, normalized on host (:333-335) | host loop (`data_stats.h:249-261`) |
| min / max | `atomicMinFloat`/`atomicMaxFloat` into `stats->vmin/vmax` (`stats_kernel.cu:72-90,187-188`) | **not computed** — STATS-7 | not computed | not computed |
| stddev | not computed by either side | — | — | — |

---

## STATS-1: Shannon entropy is summed in a host loop on the `ComputeDeviceStats` path, and the port's own faithful device entropy kernel is not reached from there
- **Claim:** Native always reduces entropy on the GPU (`entropyFromHistogramKernel` writes straight into `&d_stats->entropy`). The port has two device paths: `ComputeDeviceStatsResident*` correctly launches `EntropyFromHistKernel`, but `ComputeDeviceStats` (the one `ComputeCompressionFeatures` calls for a device-resident chunk) copies all 256 histogram bins D2H and computes `-sum p*log2(p)` in a serial host loop instead. The device kernel that would do this sits in the same file, unlaunched on that path, so the two port paths can differ in the last ulp — and only the resident one reproduces upstream's reduction order, which that kernel's own comment says is the point.
- **Native evidence:** `/home/cc/NeuroPress/src/stats/stats_kernel.cu:337-338`
  ```
  int entropy_rc = launchEntropyKernelsAsync(d_input, input_size, d_histogram,
                            &d_stats->entropy, stream);
  ```
  plus `/home/cc/NeuroPress/src/stats/entropy_kernel.cu:167-202` (`entropyFromHistogramKernel`, shared-memory tree reduction, `*entropy_out = s_partial[0]`). There is no host entropy loop anywhere in NeuroPress.
- **Port evidence:** `context-transport-primitives/src/compress/preprocess/data_stats_gpu_kernels.cu:311-329`
  ```
         cudaMemcpy(h_hist, d_hist, kHistBins * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost) == cudaSuccess &&
  ...
  double entropy = 0.0;
  for (int i = 0; i < kHistBins; i++) {
    if (h_hist[i] > 0) {
      double p = static_cast<double>(h_hist[i]) / static_cast<double>(num_bytes);
      entropy += -p * std::log2(p);
  ```
  `EntropyFromHistKernel` (same file, :155-175) is launched only from `ComputeDeviceStatsResidentTyped` (:254). Its comment at :149-153 ("The reduction ORDER is part of the port ... the whole point is to land on the value upstream's kernel produces") is true of the resident path and false of this one.
- **Category:** host-compute (+ unreachable-device-path for `EntropyFromHistKernel` on this route)
- **Severity:** high — moves a reduction native only ever performs on-device onto the CPU, adds a 1 KB D2H, and can produce a different last-ulp entropy than the port's own resident path for identical bytes.
- **Reachability:** `ctp::ComputeCompressionFeatures` -> `ComputeDeviceStats` -> `ComputeDeviceStatsTyped`, taken when `IsDevicePointer(chunk)` is true but the resident path was not chosen. Two call sites: (a) `context-transfer-engine/compressor/src/compressor_runtime.cc:634` when `np_device_path` is false — a device-resident chunk with `context.dynamic_compress_ == 1`, or `neuropress_predictor_` null/not ready (legacy Q-table / dense-NN ranking); (b) `compressor_runtime.cc:1364`, the online-learning snapshot, which is on the NeuroPress path itself (STATS-6). Not reached when the device path at `compressor_runtime.cc:619` succeeds.
- **How to falsify:** show `ComputeDeviceStatsTyped` is dead — that no configuration reaches `compressor_runtime.cc:634` or `:1364` with `IsDevicePointer(chunk) == true`. Or instrument `data_stats_gpu_kernels.cu:324` and run with a `GpuMalloc`-backed blob and `dynamic_compress_ == 1`; if it never fires, this is wrong.

## STATS-2: `ComputeDeviceStatsTyped` breaks the two stats passes apart with a D2H of the sum and a host divide for the mean
- **Claim:** Native keeps both passes on one stream and never brings the sum to the host — `madPass2Kernel` computes `stats->sum / stats->num_elements` in a `__shared__` slot on-device. The port's `ComputeDeviceStatsTyped` does a blocking D2H of the partial sums between pass 1 and pass 2, divides on the host, and passes the mean back down as a kernel argument.
- **Native evidence:** `/home/cc/NeuroPress/src/stats/stats_kernel.cu:206-213`
  ```
      __shared__ double s_mean;
      if (threadIdx.x == 0) {
          s_mean = stats->sum / static_cast<double>(stats->num_elements);
      }
      __syncthreads();
  ```
  launched back-to-back with pass 1 on one stream (`stats_kernel.cu:333-347`), no intervening copy or sync.
- **Port evidence:** `data_stats_gpu_kernels.cu:301-310`
  ```
    StatsPass1Kernel<T><<<grid, kBlockSize>>>(data, num_elements, d_hist,
                                               d_scalars, d_scalars + 1);
    ok = cudaGetLastError() == cudaSuccess &&
         cudaMemcpy(h_sum_and_d2, d_scalars, 2 * sizeof(double),
                    cudaMemcpyDeviceToHost) == cudaSuccess;
  }
  if (ok) {
    mean = h_sum_and_d2[0] / static_cast<double>(num_elements);
    StatsPass2Kernel<T>
        <<<grid, kBlockSize>>>(data, num_elements, mean, d_scalars + 2);
  ```
  The port documents this itself at :110-118 ("the host-argument form forces a D2H, a host divide and a relaunch between them") — the device-mean variant it wrote to fix it is used only by the resident path.
- **Category:** added-host-roundtrip
- **Severity:** medium — the mean is bit-identical either way; the cost is a forced host sync inside what native runs as one async chain.
- **Reachability:** identical to STATS-1 (same function, same two call sites, device-resident chunk).
- **How to falsify:** show `mean` is not a kernel argument (it is, :310), or that `StatsPass2Kernel` (:91-108) has no live caller.

## STATS-3: `ComputeDeviceStatsTyped` runs on the legacy default stream with blocking copies and per-call `cudaMalloc`/`cudaFree`; native uses a preallocated per-context workspace on a private stream
- **Claim:** Native preallocates the whole stats workspace (`AutoStatsGPU` + 256-bin histogram + flag) once per pool context and runs every stage on `ctx->stream` with async memset/memcpy. The port allocates and frees two device buffers per chunk and launches with no stream argument (legacy default stream — no `--default-stream per-thread` anywhere in the build), then uses blocking `cudaMemcpy`. Both the `cudaFree`s and the default-stream launches are device-wide serialization points native does not have here.
- **Native evidence:** `/home/cc/NeuroPress/src/api/gpucompress_pool.cpp:109-112`
  ```
        if (cudaMalloc(&ctx.d_stats_workspace, kPoolStatsWSZ) != cudaSuccess) goto fail;
        ctx.d_stats     = static_cast<AutoStatsGPU*>(ctx.d_stats_workspace);
        ctx.d_histogram = reinterpret_cast<unsigned int*>(...);
  ```
  and every stage in `stats_kernel.cu:379-409` takes `, stream>>>` / `...Async(..., stream)`.
- **Port evidence:** `data_stats_gpu_kernels.cu:284-319` — `cudaMalloc` x2 (:284,:288), blocking `cudaMemcpy` x3 (:304,:312,:314), `cudaFree` x2 (:318-319), launches `<<<grid, kBlockSize>>>` with no stream (:301,:309). No `--default-stream per-thread` in `/home/cc/clio-core/CMakeLists.txt:1168` or `context-transport-primitives/src/CMakeLists.txt`. The file concedes it at :196-203 ("ComputeDeviceStatsTyped below still does four cudaMalloc/cudaFree per chunk").
- **Category:** added-host-roundtrip
- **Severity:** medium — perf/sync only, no numeric change; but on the legacy default stream it serializes against every blocking stream in the process, including the per-thread stats/inference stream created at :220.
- **Reachability:** same as STATS-1/2.
- **How to falsify:** find a `--default-stream per-thread` / `CUDA_API_PER_THREAD_DEFAULT_STREAM` applied to this TU, or show the entry points are unreachable.

## STATS-4: In a build without CUDA the device stats entry points compile to stubs and the caller silently computes all three statistics on the CPU
- **Claim:** `ComputeDeviceStats`, `ComputeDeviceStatsResident`, `DeviceStatsStream`, `ReadDeviceFeatureStats` have `#if !CTP_ENABLE_CUDA` definitions returning `false`/`nullptr`. With `IsDevicePointer` also false on such a build, `ComputeCompressionFeatures` falls through to `DataStatisticsFactory`'s host loops and returns `true`, so `EstCompressionStats` ranks with NeuroPress on host-computed features with no error and no log line. NeuroPress has no such configuration and its host entry point prints and errors.
- **Native evidence:** `/home/cc/NeuroPress/src/api/gpucompress_api.cpp:352-366`
  ```
      fprintf(stderr, "gpucompress_compress: host-path stub - use "
                      "gpucompress_compress_gpu() or "
                      "gpucompress_compress_with_action_gpu() instead.\n");
      return GPUCOMPRESS_ERROR_INVALID_INPUT;
  ```
  No CPU implementation of any stats stage exists in the tree.
- **Port evidence:** `context-transport-primitives/src/compress/preprocess/byte_shuffle_cpu_stub.cc:22,54-77`
  ```
  #if !defined(CTP_ENABLE_CUDA) || !CTP_ENABLE_CUDA
  ...
  bool ComputeDeviceStats(const void *, size_t, DataType, double *out_entropy, ...) {
    if (out_entropy) *out_entropy = 0.0;  ...  return false;
  }
  const void *ComputeDeviceStatsResident(const void *, size_t, DataType, void *) {
    return nullptr;
  }
  ```
  plus the host fall-through returning `true` at `include/clio_ctp/compress/preprocess/data_stats_gpu.h:141-146`.
- **Category:** silent-cpu-fallback
- **Severity:** medium — numbers are the correct host equivalents, but a whole GPU pipeline is replaced by CPU loops and the caller is told it succeeded.
- **Reachability:** any `CLIO_CORE_ENABLE_CUDA=OFF` build (the stub compiles unconditionally; `data_stats_gpu_kernels.cu` is added only under that flag, `context-transport-primitives/src/CMakeLists.txt:96-108`). Also at runtime in a CUDA build on a driverless host: `GpuApi::IsDevicePointer` swallows the failed `cudaPointerGetAttributes` and returns false (`gpu_api.h:283-288`).
- **How to falsify:** show `neuropress_predictor_->IsReady()` can never be true without CUDA — that would exempt NeuroPress specifically (it would still apply to the legacy Q-table branch, which consumes the same three features).

## STATS-5: A host-resident chunk gets all three statistics from CPU loops with no diagnostic, including pinned-host "GPU" buffers
- **Claim:** When `IsDevicePointer(chunk)` is false, `ComputeCompressionFeatures` computes entropy, MAD and second derivative with three separate host passes (`CalculateMAD` alone is two) and returns `true`; `EstCompressionStats` then ranks with NeuroPress as if the GPU pipeline had run. Native has no equivalent — every stats entry takes `d_input` and the host-input API is a loud stub. Notably, Clio's `GpuShmMmap` "GPU shared memory" backend allocates **pinned host** memory (`cudaMallocHost`), which reports `cudaMemoryTypeHost`, so a GPU-facing workload on that backend takes the CPU path even though the buffer is device-accessible.
- **Native evidence:** `gpucompress_api.cpp:352-366` (stub, quoted above) and `/home/cc/NeuroPress/src/api/gpucompress_compress.cpp:280`
  ```
      AutoStatsGPU* d_stats_ptr = gpucompress::runStatsKernelsNoSync(d_input, input_size, stream, ctx);
  ```
- **Port evidence:** `context-transport-primitives/include/clio_ctp/compress/preprocess/data_stats_gpu.h:141-146`
  ```
    *out_entropy =
        DataStatisticsFactory::CalculateShannonEntropy(chunk, num_elements, type);
    *out_mad = DataStatisticsFactory::CalculateMAD(chunk, num_elements, type);
    *out_second_derivative =
        DataStatisticsFactory::CalculateSecondDerivative(chunk, num_elements, type);
    return true;
  ```
  Host impls: `data_stats.h:100-113`, `:131-142`, `:252-260`. Residency test: `gpu_api.h:284-288` (`attributes.type == cudaMemoryTypeDevice`). Pinned-host backend: `context-transport-primitives/include/clio_ctp/memory/backend/gpu_shm_mmap.h:106-124` (`GpuApi::MallocHost<char>(backend_size)`).
- **Category:** silent-cpu-fallback
- **Severity:** high in practice (the entire stats pipeline on the CPU over the whole chunk, feeding a model native only ever feeds from GPU reductions) — with the honest caveat that for a genuinely host-resident chunk some host work is unavoidable; the objectionable part is that it is silent and that it also captures pinned-host and managed buffers the GPU could read directly.
- **Reachability:** `compressor_runtime.cc:630-636` whenever `IsDevicePointer(chunk)` is false. `chunk_data` comes from `CLIO_IPC->ToFullPtr<char>(task->blob_data_)` (`compressor_runtime.cc:1254-1256`), so it is host memory for every POSIX-shm-backed blob **and** every `GpuShmMmap`-backed blob; only a `GpuMalloc`-backed (true `cudaMalloc`) blob takes the device path. Also captures `cudaMallocManaged` buffers (`cudaMemoryTypeManaged != cudaMemoryTypeDevice`), e.g. anything from `GpuApi::MallocManaged` (`gpu_api.h:208-221`).
- **How to falsify:** demonstrate `cudaPointerGetAttributes(...).type == cudaMemoryTypeDevice` for `cudaMallocHost`/`cudaMallocManaged` pointers (it is not), or show the compressor's blob data is always `GpuMalloc`-backed on the GPU path.

## STATS-6: The online-learning feature snapshot recomputes all three statistics through the host path instead of reusing the device-resident stats native reuses via a D2D copy
- **Claim:** After `EstCompressionStats` has produced device-resident statistics, `Runtime::DynamicSchedule` calls `ComputeCompressionFeatures` a second time on the same chunk to obtain host doubles for SGD. On a device-resident chunk that lands in `ComputeDeviceStatsTyped` — an extra full two-pass sweep plus the host entropy loop of STATS-1, the mid-pipeline D2H of STATS-2, and the four allocator calls of STATS-3. Native recomputes nothing: it copies the already-computed `AutoStatsGPU` device-to-device and hands that device pointer to the SGD kernel.
- **Native evidence:** `/home/cc/NeuroPress/src/api/gpucompress_compress.cpp:630-636`
  ```
      if (cfg.algorithm == GPUCOMPRESS_ALGO_AUTO && d_precomputed_stats) {
          cuda_err = cudaMemcpyAsync(ctx->d_stats, d_precomputed_stats, sizeof(AutoStatsGPU),
                                     cudaMemcpyDeviceToDevice, stream);
  ```
  consumed at `:721` `gpucompress::runNNSGDCtx(d_stats_ptr, primary_sgd, 1, ...)`. The only host copy native makes is the 24-byte diagnostics read at `:1106-1114`, after the stream is already synced.
- **Port evidence:** `context-transfer-engine/compressor/src/compressor_runtime.cc:1364-1366`
  ```
        neuropress_feat_valid = ctp::ComputeCompressionFeatures(
            chunk_data, feat_num_elements, feat_type, &neuropress_entropy,
            &neuropress_mad, &neuropress_second_deriv);
  ```
  The device stats from `:619` are still live and were already read to the host at `:681-684`; neither is reused. Secondary detail: `feat_type` here follows `context.data_type_` (`:1352-1354`, default `UINT8`) while `EstCompressionStats` forces `FLOAT32` for NeuroPress (`:569-573`), so the recomputed values are not even the statistics the ranking used.
- **Category:** added-host-roundtrip (also device-struct-became-host: the port's SGD consumes host doubles because `NeuroPressNNPredictor`'s SGD is a CPU port of `nnSGDKernel` — `neuropress_nn_predictor.h:186-190`, "Ports NeuroPress's nnSGDKernel ... to plain scalar CPU code"; that half is NN scope, but it is why the stats must come to the host at all)
- **Severity:** high — a second full pass over the chunk plus a host entropy reduction native never performs.
- **Reachability:** `config_.neuropress_online_learning_enabled_ && context.dynamic_compress_ != 1 && neuropress_predictor_->IsReady()` (`compressor_runtime.cc:1349-1351`). Off by default (`compressor_tasks.h:73`), enabled via the `neuropress_online_learning_enabled` YAML key. On a host-resident chunk it degenerates to STATS-5's host loops.
- **How to falsify:** show the block is unreachable in shipped configurations, or that `ComputeCompressionFeatures` at `:1364` reuses `device_stats` (it does not — it takes `chunk_data` and recomputes).

## STATS-7: Native's on-device min/max reduction has no counterpart in the port
- **Claim:** `statsPass1Kernel` computes `vmin`/`vmax` with CAS-loop float atomics into the device stats struct. The port's `StatsPass1Kernel` computes only sum, second-derivative sum and histogram; `DeviceFeatureStats` has no min/max fields and nothing in the port's stats path computes them.
- **Native evidence:** `/home/cc/NeuroPress/src/stats/stats_kernel.cu:184-188`
  ```
          if (lane == 0) {
              atomicAdd(&stats->sum, t_sum);
              atomicAdd(&stats->abs_diff_sum, t_deriv);
              atomicMinFloat(&stats->vmin, t_min);
              atomicMaxFloat(&stats->vmax, t_max);
  ```
  (helpers at `:72-90`, sentinels seeded H2D at `:323-326`).
- **Port evidence:** `data_stats_gpu_kernels.cu:84-87` (pass 1 emits only `sum_out` and `sum_abs_d2_out`); `include/clio_ctp/compress/preprocess/data_stats_gpu.h:56-60` (`DeviceFeatureStats` = `{entropy, mad, second_derivative}`).
- **Category:** missing-gpu-work
- **Severity:** low — and the honest caveat is that `vmin`/`vmax` are **dead in native too**: `grep -rn "vmin\|vmax" src/ include/` in NeuroPress returns only the definition, the two atomics and the sentinel seeding. No consumer, and they are not NN inputs (`nn_gpu.cu` reads only `entropy`, `mad_normalized`, `deriv_normalized`). Nothing downstream changes.
- **Reachability:** always, on both port paths.
- **How to falsify:** find a reader of `AutoStatsGPU::vmin`/`vmax` in NeuroPress — that would upgrade this to a real gap.

## STATS-8: `DeviceMinMax` ends with `cudaDeviceSynchronize` and per-call allocation; native syncs only its context stream and can reuse a preallocated buffer
- **Claim:** Both sides reduce the data range on the GPU, so this is not host-compute — but the port's version stops the whole device where native stops only the caller's stream, and native has an overload avoiding the per-call `cudaMalloc`.
- **Native evidence:** `/home/cc/NeuroPress/src/preprocessing/quantization_kernels.cu:274-276`
  ```
      err = cudaStreamSynchronize(stream);
      if (err != cudaSuccess) { if (owns_temp) cudaFree(d_temp); return -1; }
  ```
  with the preallocated-temp overload at `:236-282` (`if (d_cub_temp && cub_temp_cap >= temp_bytes) d_temp = d_cub_temp;`).
- **Port evidence:** `data_stats_gpu_kernels.cu:684-701`
  ```
    if (cudaMalloc(&d_res, 2 * sizeof(unsigned int)) != cudaSuccess) return false;
  ...
      MinMaxKernel<<<grid, kBlockSize>>>(d_in, n, d_res, d_res + 1);
      ok = cudaGetLastError() == cudaSuccess &&
           cudaDeviceSynchronize() == cudaSuccess;
  ```
  (`QuantizeDevice`/`DequantizeDevice` do the same at :779 and :820.)
- **Category:** added-host-roundtrip
- **Severity:** medium — sync/perf only; min/max are exact and order-independent so values are identical. Flagged because min/max is named in this audit's scope; overlaps a quantization/preprocessing reviewer's territory.
- **Reachability:** `QuantizeDevice` -> `DeviceMinMax`, i.e. any device-resident chunk whose selected action sets the quantize bit (`compressor_runtime.cc:2242`).
- **How to falsify:** show `cudaDeviceSynchronize` is equivalent here because no concurrent device work is ever in flight (check whether the per-thread stats/inference stream at `data_stats_gpu_kernels.cu:220` or another worker's stream can be live).

---

## Subcategories with NO finding (checked, genuinely matches)

- **The NeuroPress device path in `EstCompressionStats` is genuinely device-resident.** `compressor_runtime.cc:613-621` calls `ComputeDeviceStatsResident`, which launches all four stages on one stream with no copies and no sync (`data_stats_gpu_kernels.cu:236-260`); `NeuroPressCandidateStatsDevice` -> `PredictBatchDeviceStats` passes the device pointer to the inference kernel without reading it (`neuropress_bridge.cc:229-258`, `neuropress_nn_predictor.cc:562-593`). The comments citing `gpucompress_compress.cpp:280-282` are accurate **for this path**.
- **The `ReadDeviceFeatureStats` D2H at `compressor_runtime.cc:681` is not an added round trip.** It happens after the ranking has already synchronized, and native does the same for diagnostics (`gpucompress_compress.cpp:1106-1114`, blocking `cudaMemcpy` of the whole `AutoStatsGPU` post-sync). Its `cudaStreamSynchronize` (`data_stats_gpu_kernels.cu:424`) is stream-scoped, matching native.
- **Mean/MAD normalization and the `n > 2` second-derivative guard** are on-device in the port's resident path (`FinalizeFeatureStatsKernel`, :184-194) and match `finalizeStatsOnlyKernel` (`stats_kernel.cu:260-281`) including the guard.
- **stddev/variance:** neither side computes it in the selection feature set. `CalculateByteFrequencyVariance` (`data_stats.h:274-299`) is host-only but has no native counterpart and no caller on the selection path.
- **Histogram kernel shape.** Native's vectorized `histogramKernelVec4` (`entropy_kernel.cu:99-151`) vs the port's scalar per-block-privatized histogram fused into `StatsPass1Kernel`: both on-device, identical counts. Device-vs-device throughput, not execution locality — deliberately not reported.
- **Stream creation.** Port uses `cudaStreamCreate` (`data_stats_gpu_kernels.cu:220`); native's pool contexts use `cudaStreamCreate` too (`gpucompress_pool.cpp:101`). Both blocking — no difference.
- **`heuristic_select_action`** (`selection/heuristic.cu:25-34`) is plain host C++ in native as well (no `__device__`, no kernel despite the `.cu`), fed by a D2H'd entropy at `H5VLgpucompress.cu:2316-2330`. The port has no port of it, but its absence is not a host/device relocation.
