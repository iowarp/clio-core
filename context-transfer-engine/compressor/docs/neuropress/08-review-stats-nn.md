# Adversarial verification — STATS-1..8 and NN-1..10

Default posture: every finding is wrong until it survives an attempt to break it.
Each finding's own "How to falsify" field was attempted first, with experiments
preferred over argument (nsys/CUPTI stream and byte-size breakdowns, standalone probes
compiled against the real TUs, and side-by-side native-vs-port runs in the parity harness).

**Counts:** REFUTED 1 · CONFIRMED 9 · CONFIRMED-WITH-CORRECTION 8 · UNPROVEN 0.
**Severity after review:** high 4 · medium 7 · low 6 · none 1. The original set had 8 "high".

## Two findings up front

**1. The tree moved during this pass.** `neuropress_nn_gpu_kernels.cu` went from md5
`a4755c11…` (1152 lines) to `262b2d4a…` (1182) and `neuropress_nn_predictor.cc` from
`c07755b2…` (1220) to `d919cab3…` (1244) between the first and last read — both already
different from the md5s the NN audit recorded. **Every port line number in
`02-findings-nn.md` is stale.** Citations below are anchored to the 2026-08-13T06:49Z
snapshot. `data_stats_gpu_kernels.cu` and `compressor_runtime.cc` were stable.

**2. A fourth production kernel exists that the audit never saw.** `RankKernel`
(`neuropress_nn_gpu_kernels.cu:549`) does the cost model, both `-INFINITY` masks and a
32-lane bitonic sort **on the GPU**. It is absent from git HEAD, present in the working
tree and the built `.so`, and **launched 32-64 times in every device-path run profiled**.
It directly undercuts NN-3 and NN-4. See the correction note in `00-kernel-inventory.md`.

---

## STATS-1: CONFIRMED
- **Disproof attempts:** Tried to show `ComputeDeviceStatsTyped` dead. Profiled `neuropress_gpu_demo` (online learning on) looking for the non-resident kernels. Built a standalone probe calling both entry points on the same buffer to test whether the claimed ulp divergence is real or hypothetical.
- **Evidence:** The path is live:
  ```
  StatsPass1Kernel<unsigned char>   32 instances  stream=7
  StatsPass2Kernel<unsigned char>   32 instances  stream=7
  memcpy D2H bytes=1024 count=32    <- the 256-bin histogram, per chunk
  ```
  `EntropyFromHistKernel` ran only 32 times, matching the 32 *resident* `StatsPass1Kernel<float>` calls on stream 13 — so entropy for the other 32 chunks was summed on the CPU. The ulp claim is not hypothetical; probe on 1 Mi floats:
  ```
  ComputeDeviceStats      entropy=7.1733422533101487
  ComputeDeviceStatsRes   entropy=7.1733422533101479
  entropy delta = 8.88e-16  (bitwise-equal=0)   mad/d2 bitwise-equal=1
  ```
  Citations exact: host loop `data_stats_gpu_kernels.cu:324-329`, histogram D2H `:312-313`, `EntropyFromHistKernel:155-175` launched only at `:254`.
- **Correction:** Reachability route (a) is `compressor_runtime.cc:661` (not `:634`); route (b) is `:1396` (not `:1364`). Route (b) is the one observed firing.
- **Corrected severity:** medium — real and reachable, but the divergence is 1 ulp, the transfer is 1 KB/chunk, and on the reachable route the value feeds SGD, not the ranking.

## STATS-2: CONFIRMED
- **Disproof attempts:** Stated falsification (show `mean` is not a kernel argument / `StatsPass2Kernel` has no live caller) — both fail. Measured actual copies with nsys API tracing rather than trusting source.
- **Evidence:** `data_stats_gpu_kernels.cu:301-310` exact. Probe shows 3 blocking D2H per `ComputeDeviceStats` call; per-stream breakdown `(D2H, 16, stream 7)`, `(1024, 7)`, `(8, 7)` — the 16-byte one is the `[sum, sum_abs_d2]` readback. Both `StatsPass2Kernel<float>` and `<unsigned char>` observed launching live.
- **Corrected severity:** low — the finding concedes the mean is bit-identical (probe confirms `mad` bitwise-equal). Sync-shape divergence with no measurable cost in any run profiled (see NN-10).

## STATS-3: CONFIRMED
- **Disproof attempts:** Searched the whole build for `--default-stream per-thread` / `CUDA_API_PER_THREAD_DEFAULT_STREAM` (`grep -rn` over all CMake) — **zero hits**. Confirmed stream assignment empirically rather than inferring.
- **Evidence:** CUPTI stream table — `StatsPass1/2Kernel` (non-resident) on **stream 7** (null stream), while `EntropyFromHistKernel`/`StatsPass2DevKernel`/`FinalizeFeatureStatsKernel` (resident) are on **stream 13**, the created scratch stream. API counts per call: `cudaMalloc` x2, `cudaMemset` x2, blocking `cudaMemcpy` x3, `cudaFree` x2. Native preallocation `gpucompress_pool.cpp:109-112`; every stage stream-qualified `stats_kernel.cu:379-409`.
- **Corrected severity:** low — architecturally real, but **zero** cross-stream kernel overlap measured in these workloads even among non-default streams, so the serialization cost is latent, not observed.

## STATS-4: CONFIRMED-WITH-CORRECTION
- **Disproof attempts:** The brief asked whether a build nobody ships matters. Checked whether anybody ships it — then compiled the stub TU for real and ran it.
- **Evidence:** `CMakeLists.txt:145`: `option(CLIO_CORE_ENABLE_CUDA "Enable CUDA support" OFF)`. **The CPU-only build is the project default.** Compiled `byte_shuffle_cpu_stub.cc` with no `CTP_ENABLE_CUDA`:
  ```
  --- calling ComputeCompressionFeatures (watch for diagnostics) ---
  --- returned ---                      <- nothing printed in between
  ComputeCompressionFeatures = 1  entropy=7.1257413605984716 mad=63.85… d2=6.39e-05
  ComputeDeviceStats (stub)  = 0  entropy=0 mad=0 d2=0
  ```
  Native stub verified verbatim at `gpucompress_api.cpp:352-366`.
- **Correction:** The CMake citation is **wrong** — `data_stats_gpu_kernels.cu` is added at `compress/model/CMakeLists.txt:29-38`, not `src/CMakeLists.txt:96-108`; the stub is listed at `model/CMakeLists.txt:19`. Severity moves *up*, not down: the finding treats this as a fringe build; it is the default one.
- **Corrected severity:** medium — numbers are correct host equivalents, so this is a silence/observability defect rather than a wrong answer, but it fires in the default cmake configuration.

## STATS-5: CONFIRMED
- **Disproof attempts:** Both stated falsifications. (a) Measured `cudaPointerGetAttributes(...).type` for pinned-host and managed allocations. (b) Independently re-profiled `compressor_dynamic_neuropress` rather than relying on doc 07 — and hit the stale-`.sqlite` trap once, which silently produced an empty report; deleted the `.sqlite` and regenerated (18 rows verified).
- **Evidence:**
  ```
  cudaMallocHost    ptr type=1 (Host)
  cudaMallocManaged ptr type=3 (Managed)
  enum: Unregistered=0 Host=1 Device=2 Managed=3
  ```
  `gpu_api.h:289` `return attributes.type == cudaMemoryTypeDevice;` — neither qualifies. `GpuShmMmap` allocates via `GpuApi::MallocHost<char>` (`gpu_shm_mmap.h:124`). Independent re-profile, complete 18-row kernel list: `InferKernel` x8, nvcomp LZ4/Zstd/Bitcomp, plumbing — and **not one `Stats*`/`EntropyFromHist*` kernel**.
- **Correction:** `chunk_data` origin drifted to `compressor_runtime.cc:1286-1288`; residency test is `gpu_api.h:283-289`.
- **Corrected severity:** high — the ordinary path of the shipped test, and native has no host path at all. The finding's own caveat (some host work is unavoidable for a genuinely host-resident chunk) is fair and stands.

## STATS-6: CONFIRMED-WITH-CORRECTION
- **Disproof attempts:** Tried to show the block unreachable. Default is `false` (`compressor_tasks.h:73`) — but two shipped examples set it true (`neuropress_gpu_demo.cc:114` unconditionally, `neuropress_e2e.cc:134` by env flag). Profiled `neuropress_gpu_demo`.
- **Evidence:** The recompute happens and the type mismatch is visible:
  ```
  StatsPass1Kernel<float>          32   stream 13   <- EstCompressionStats, resident, FLOAT32
  StatsPass1Kernel<unsigned char>  32   stream 7    <- SGD snapshot, non-resident, UINT8
  StatsPass2Kernel<unsigned char>  32   stream 7
  ```
  Same 32 chunks, two full sweeps, two element widths — the second 4x longer in elements. Native D2D reuse verified at `gpucompress_compress.cpp:630-636`, consumed `:721`.
- **Correction:** Lines drifted: recompute `:1396-1398`, gate `:1381-1383`, `feat_type` `:1384-1386`, FLOAT32 forcing `:596-600`. Minor: native's D2D copy *is* followed by a `cudaStreamSynchronize` — not fully async as implied — but still a 64-byte D2D, not a recompute.
- **Corrected severity:** medium — genuinely a second full pass native never does, but behind an opt-in that is off by default.

## STATS-7: REFUTED (non-issue)
- **Disproof attempts:** Ran the finding's own falsification: search NeuroPress for any reader of `AutoStatsGPU::vmin/vmax`.
- **Evidence:**
  ```
  $ grep -rn "vmin\|vmax" src/ include/ tests/
  src/stats/stats_kernel.cu:112     (comment)
  src/stats/stats_kernel.cu:187,188 (the two atomics)
  src/stats/stats_kernel.cu:323,325,384,386 (sentinel seeding)
  src/stats/auto_stats_gpu.h:18,19  (field declarations)
  tests/…                            (tests only)
  ```
  No production consumer anywhere.
- **Correction:** The finding's own concession is correct and it should never have been filed. Missing dead work is not a gap. **Strike it, don't downgrade it.**
- **Corrected severity:** none

## STATS-8: CONFIRMED-WITH-CORRECTION
- **Disproof attempts:** Attacked reachability first — nothing sets an error bound by default, so the quantize path was suspected dead. Re-ran `neuropress_gpu_direct` with `CLIO_NEUROPRESS_ERROR_BOUND=1e-3`. Then attacked severity by measuring whether `cudaDeviceSynchronize` costs anything.
- **Evidence:** Reachable: `MinMaxKernel` 51, `QuantizeKernel<short>` 51, `DequantizeKernel<short>` 51. Port code exact: `cudaMalloc` `:684`, `cudaDeviceSynchronize()` `:695`, `cudaFree` `:701`; also `:779`, `:820`. Native `quantization_kernels.cu:274-276` with the preallocated-temp overload at `:236-282`. **But the severity attack succeeds:** across 863 kernel executions in the demo run, **0** cross-stream overlapping pairs even among non-default streams.
- **Correction:** Gate drifted to `compressor_runtime.cc:2274`. "medium" is not supported by measurement.
- **Corrected severity:** low

---

## NN-1: CONFIRMED-WITH-CORRECTION
- **Disproof attempts:** Three. (1) Looked for a `__global__` decomp-head kernel in the port — found `RankKernel`, which the audit missed, but it ranks, it does not update the head. (2) Tried to show `nnBatchedDecompSGDKernel` dead upstream — it is called from `gpucompress_learning.cpp:101`, reached from `H5VLgpucompress.cu:3355`, a real VOL call site. (3) Tried to show `TrainDecompHead` unreachable — ran `neuropress_e2e` with learning on.
- **Evidence:** Reachable and hot:
  ```
  $ grep -c 'decomp-head SGD' npe2e_out.txt
  217
  compressor_runtime.cc:1073 LearnDecompTime NeuroPress decomp-head SGD: batch=1 trained=1
  … batch=2 … batch=3 …
  ```
  Batch size grows monotonically (`LearnDecompTime` re-sweeps every recorded record, `:1058-1064`), so host cost is **O(n²) over a run, not O(n)**. Side-by-side in the parity profile: native `nnBatchedDecompSGDKernel` **8 instances, stream 14**; port — no such kernel exists.
- **Correction:** Every port line number is wrong (~+70): forward pass `neuropress_nn_predictor.cc:1158`, head output `:1160-1163`, gradient accumulation `:1179-1184`, weight update `:1221-1233`. The claim that the only `__global__` functions are the three listed is **stale and wrong** — there are four, at `:415`, `:453`, `:549` (`RankKernel`), `:822`.
- **Corrected severity:** high

## NN-2: CONFIRMED
- **Disproof attempts:** Stated falsification (show the download/upload elided, or only 65 floats moving). An `LD_PRELOAD` `cudaMemcpy` interposer segfaulted inside the CUDA runtime, so it was abandoned for CUPTI byte-size histograms — stronger anyway.
- **Evidence:** `kParamCount = 512 + 3x4096 + 512 + 256 + 8 = 13,576`; `sizeof(host_params) = 54,304 B`. Parity-harness profile, byte-exact:
  ```
  (kind=H2D, bytes=54304, count=9)     <- 1 Load() + 8 UploadWeights
  (kind=D2H, bytes=54304, count=156)   <- DownloadWeights (incl. Upload's RMW read)
  ```
  Against **8** native `nnBatchedDecompSGDKernel` launches that move **zero** weights. ~9 MB of D2H traffic to edit 65 floats eight times. Code exact: download `:1141-1144`, upload `:1237-1240`; transfers `neuropress_nn_gpu_kernels.cu:273` (D2H), `:298` (second D2H inside the RMW), `:314` (H2D).
- **Correction:** Line numbers drifted; `model_mutex_` is at `:1120`.
- **Corrected severity:** high

## NN-3: CONFIRMED-WITH-CORRECTION
- **Disproof attempts:** Ran the stated falsification — "point to a device reduction/sort in the port that yields the selected action" — and **it succeeded for the device path**.
- **Evidence:** `neuropress_nn_gpu_kernels.cu:514-578`, `RankKernel`: computes `score = -(w_ct*ct + w_dt*dt + w_io*io)` (`:528-534`), applies both `-INFINITY` masks (`:540-543`), and runs *the same bitonic network as `nn_gpu.cu:499-518`* using `__shfl_xor_sync` (`:557-575`), emitting the full ranked order. Launched on the stats stream at `:673`. Observed **32-64 instances in every device-path run**. The bridge consumes it without re-ranking (`neuropress_bridge.cc:276-294`). The host-path claim survives: `NeuroPressGpuInferBatch` still emits only raw metrics (`:382-385`), `Rank()` -> `ScoreAndSort` -> `std::stable_sort` (`predictor.h:399`), and that is what `compressor_dynamic_neuropress` takes.
- **Correction:** "Reachability: **both** port paths" is **false** — the device path ranks on the GPU. The evidence line "*Grep for `__shfl`/`bitonic` in port production `.cu` files: none*" is **now false**.
- **Corrected severity:** medium — half the claim was fixed in the tree between the audit and this review.

## NN-4: CONFIRMED-WITH-CORRECTION
- **Disproof attempts:** Both stated falsifications. (1) "Show a min-PSNR argument threaded into the port's kernels" — **succeeded**. (2) "Show upstream's `g_min_psnr_db` default is 0 and Clio never sets `target_psnr_`" — **half succeeded**.
- **Evidence:** `RankKernel` takes `double min_psnr` (`:521`) and applies it in-kernel exactly as native does:
  ```
  :540  if (is_quant && error_bound <= 0.0) score = -CUDART_INF;
  :541  if (min_psnr > 0.0 && (double)psnr_in[tid] < min_psnr) score = -CUDART_INF;
  ```
  vs `nn_gpu.cu:238-239`. Threaded from `context.target_psnr_` at `compressor_runtime.cc:704` -> bridge `:261` -> `neuropress_nn_predictor.cc:648`. Host filter now gated to the host path: `:727` `if (device_stats == nullptr && context.target_psnr_ > 0)`. Defaults: native `g_min_psnr_db = 0.0f` (`gpucompress_api.cpp:70`); Clio `target_psnr_(0)` (`core_tasks.h:1632`). **By default neither side applies any floor.**
- **Correction:** The claim that neither inference kernel takes a min-PSNR argument is true but irrelevant — the mask moved to `RankKernel`. The cited host filter is now `:727-740` *and* carries a `device_stats == nullptr` guard the finding does not mention.
- **Corrected severity:** low

## NN-5: CONFIRMED
- **Disproof attempts:** The brief asked whether a build nobody ships counts. Checked whether anybody ships it, then compiled the *production* TU exactly as the CPU build compiles it and ran it.
- **Evidence:** `CMakeLists.txt:145` — `CLIO_CORE_ENABLE_CUDA` defaults **OFF**. Compiled `neuropress_nn_predictor.cc` unmodified with `-DCTP_ENABLE_NEUROPRESS_GPU=0`, loaded the shipped `model.nnwt`:
  ```
  --- calling Load() (watch for any diagnostic) ---
  --- Load() returned ---                          <- nothing printed
  CTP_ENABLE_NEUROPRESS_GPU = 0
  Load()    = 1
  IsReady() = 1
  GpuInferenceActive() = 0
  PredictBatch -> 1 prediction(s)
    ratio=3.574776 comp_ms=4.365405 decomp_ms=1.936136 psnr=119.850418
  Train() = 1
  TrainDecompHead() = 1
  ```
  Every claim holds: silent load, `IsReady()` true (so `compressor_runtime.cc:580/698/1383` all open), full host forward pass, host backprop, host decomp head. `GpuInferenceActive()` correctly reports 0 and **nothing consults it**. Port lines `:148-181`, `:172-179`, `:183` are **exact**.
- **Correction:** Trivial — the `else()` carrying `CTP_ENABLE_NEUROPRESS_GPU=0` belongs to the `if` at `:43`, not `:29`. Answering the brief directly: this is **not** "a build nobody ships" — it is the default cmake configuration.
- **Corrected severity:** high

## NN-6: CONFIRMED
- **Disproof attempts:** Looked for a native SGD entry that *does* read back — found one, `runNNSGD` (`nn_gpu.cu:2076-2134`). Then checked whether it has any production caller. It does not.
- **Evidence:** Every production call site uses the fire-and-forget `runNNSGDCtx`: `gpucompress_compress.cpp:721`, `:1023`, `H5VLgpucompress.cu:1719`, `:1753`. Native comment verified verbatim at `nn_gpu.cu:2325-2334`. Port: `cudaMalloc` x2, H2D, `SGDKernel<<<1, kHiddenDim>>>` (`:1172`), blocking `cudaMemcpy(&applied, …, D2H)` (`:1176`), `cudaFree` x2 — all under `model_mutex_`. Measured: `SGDKernel` 61 launches and exactly `(D2H, bytes=1, count=61)` — one blocking 1-byte readback per launch. In the demo run `SGDKernel` was the single largest device consumer at 8.07 ms across 6 launches, all on the null stream.
- **Correction:** Lines drifted (`:1013-1027`->`:1156-1178`; lock `:708`, readers `:475`/`:593`).
- **Corrected severity:** medium — real and reachable, but the serialization loss could not be measured (see NN-10). "high" needs a contended multi-worker benchmark neither the finding nor this pass supplies.

## NN-7: CONFIRMED
- **Disproof attempts:** Both falsifications — show it unreachable, or show `InferScratch` covers it.
- **Evidence:** Independently reproduced (not taken from doc 07): own `compressor_dynamic_neuropress` profile lists `InferKernel` **8 instances** and no `InferKernelDeviceStats`. It also runs in the *device*-oriented `neuropress_gpu_demo` (32 instances alongside 32 `InferKernelDeviceStats`), so it is not confined to host-only workloads. `InferScratch` does **not** cover it: `Infer()` is referenced only inside `NeuroPressGpuInferBatchDeviceStats` (`:634-700`); `NeuroPressGpuInferBatch` (`:710-758`) still does 5 `cudaMalloc`, blocking H2D, `InferKernel` with no stream (`:737`), `cudaDeviceSynchronize()` (`:739`), 4 blocking D2H, 5 `cudaFree`. Native: `cudaDeviceSynchronize` appears **0 times** in all of `nn_gpu.cu`.
- **Correction:** Lines drifted (`:575-579`/`:586-588`/`:601-605` -> `:726-730`/`:737-739`/`:753-757`).
- **Corrected severity:** medium

## NN-8: CONFIRMED-WITH-CORRECTION
- **Disproof attempts:** Stated falsifications — find a port SGD entry taking a device stats pointer (none: `NeuroPressGpuTrain` takes only `NeuroPressGpuSGDSample*`), or show native's kernel receives host-built features (it does not).
- **Evidence:** Native `nnSGDKernel(NNWeightsGPU*, const AutoStatsGPU* __restrict__ d_stats, …)` (`nn_gpu.cu:620-630`), reading device memory at `:656-658`; `SGDSample` is 20 B (`nn_weights.h:38-44`). The "second full statistics pass on a device chunk" consequence is **measured**: `StatsPass1Kernel<unsigned char>` x32 + `StatsPass2Kernel<unsigned char>` x32 in the demo run, on the same 32 chunks the resident FLOAT32 pass already covered.
- **Correction:** Struct is at `neuropress_nn_gpu_kernels.h:150-156` (not `:122-128`); host assembly and runtime materialization at `compressor_runtime.cc:1396-1398` under `:1373-1377`.
- **Corrected severity:** medium

## NN-9: CONFIRMED
- **Disproof attempts:** Both falsifications — show the port emits them (it does not), or show native's 4-7 influence selection (they do not).
- **Evidence:** Port `NeuroPressForwardShared` computes all eight (`:359-362`) then transforms and stores only 0-3 (`:366-386`); `s_y[4..7]` dead. Native inverse-transforms all eight (`nn_gpu.cu:207-215`) and fills `NNDebugPerConfig` for all 32 configs on device (`:455-466`). Consumers traced: `rmse`/`max_error`/`mae`/`ssim` reach only `gpucompress_diagnostics.cpp:153-156` and VOL write-info records — never `rank_val`.
- **Correction:** The cost half of `NNDebugPerConfig` is now partially recovered — `RankKernel` returns per-candidate `scores` to the bridge — so only the quality metrics 4-7 are genuinely missing.
- **Corrected severity:** low

## NN-10: CONFIRMED-WITH-CORRECTION
- **Disproof attempts:** Both falsifications, empirically. (1) Searched the entire build for `--default-stream per-thread` — zero hits. (2) Took the finding's own proposed disproof — "*an nsys timeline showing `SGDKernel` overlapping stats-stream work disproves it*" — and ran exactly that test.
- **Evidence:** Stream assignment measured side-by-side with native in the parity run:
  ```
  PORT   InferKernel                 stream 7  (legacy default)   n=35
  PORT   SGDKernel                   stream 7  (legacy default)   n=61
  NATIVE nnInferenceKernel           stream 13                    n=35
  NATIVE nnSGDKernel                 stream 13                    n=61
  NATIVE nnBatchedDecompSGDKernel    stream 14                    n=8
  ```
  Overlap analysis on the demo run (863 kernel executions, 6 distinct streams):
  ```
  stream-7 (legacy default) kernels: 168
  other-stream kernels:              695
  stream-7 kernels overlapping ANY other stream:  0
  ```
  The finding's own disproof fails: `SGDKernel` never overlaps stats-stream work.
- **Correction:** (a) Lines drifted: `InferKernel` launch `:737`, `SGDKernel` launch `:1172`, `InferKernelDeviceStats` `:662`; `RankKernel` (`:673`) is a *second* correctly-streamed exception the finding does not know about. (b) **Severity attack partially succeeds**: non-default streams also showed 0 cross-stream overlapping pairs in the demo (2 in `gpu_direct`). "No concurrency was available" cannot be cleanly separated from "the null stream destroyed it" — the 168 interleaved null-stream fences are a plausible but unproven cause.
- **Corrected severity:** medium

---

## Summary

| ID | Verdict | Corrected severity |
|---|---|---|
| STATS-1 | CONFIRMED | medium |
| STATS-2 | CONFIRMED | low |
| STATS-3 | CONFIRMED | low |
| STATS-4 | CONFIRMED-WITH-CORRECTION | medium |
| STATS-5 | CONFIRMED | high |
| STATS-6 | CONFIRMED-WITH-CORRECTION | medium |
| STATS-7 | **REFUTED** | none |
| STATS-8 | CONFIRMED-WITH-CORRECTION | low |
| NN-1 | CONFIRMED-WITH-CORRECTION | high |
| NN-2 | CONFIRMED | high |
| NN-3 | CONFIRMED-WITH-CORRECTION | medium |
| NN-4 | CONFIRMED-WITH-CORRECTION | low |
| NN-5 | CONFIRMED | high |
| NN-6 | CONFIRMED | medium |
| NN-7 | CONFIRMED | medium |
| NN-8 | CONFIRMED-WITH-CORRECTION | medium |
| NN-9 | CONFIRMED | low |
| NN-10 | CONFIRMED-WITH-CORRECTION | medium |

## Where the audit was weak

1. **STATS-7 should be struck.** Its own concession refutes it. Reporting a missing port of dead upstream code is noise.
2. **NN-3 and NN-4 were written against a tree that no longer exists.** `RankKernel` — a real device cost model, both `-INFINITY` masks, and upstream's bitonic network — is in the working tree, in the shipped binary, and executed 32-64x in every device-path run. It is missing from `00-kernel-inventory.md`, which advertises itself as the cross-check for exactly this.
3. **NN-4's severity was inflated** by not checking that native's `g_min_psnr_db` also defaults to 0 — by default neither side filters.
4. **Perf/sync severities were assigned without measurement.** STATS-2, STATS-3, STATS-8, NN-10 all claim serialization harm; **0** cross-stream overlap was measured *anywhere* in these workloads. The divergences are real; the costs are unquantified and possibly nil at current concurrency.
5. **STATS-4/NN-5 severity was assigned backwards.** They are treated as fringe build-gated issues. `CLIO_CORE_ENABLE_CUDA` defaults to **OFF** — this is the default configuration, proven by compiling and running the real TUs.

## Where the audit was right and stronger than it claimed

- **STATS-1/2/3/6 and NN-8 collapse into one observed event.** `neuropress_gpu_demo` runs `StatsPass1Kernel<unsigned char>` and `StatsPass2Kernel<unsigned char>` 32 times on the null stream, with 32x 1 KB histogram D2H — the SGD snapshot recomputing, at the wrong element width, statistics that already exist device-resident from the FLOAT32 resident pass on stream 13. Five findings, one profile.
- **NN-1/NN-2 are proven quantitatively**, which the audit did not manage: 8 native `nnBatchedDecompSGDKernel` launches moving zero weights, versus 9x 54,304-byte H2D and 156x 54,304-byte D2H in the port, and 217 live `TrainDecompHead` calls in a shipped example with a batch that grows every read.
- **STATS-1's ulp claim is real, not theoretical** — 8.88e-16, bitwise-unequal, measured.

## Caveats on this pass

- The `build/` binaries profiled were compiled from an earlier tree state than the final snapshot; the two NN model source files gained ~30 lines each *during* this review. Behavioral conclusions rest on the built binaries, source citations on the 06:49Z snapshot.
- The stale-`.sqlite` trap was hit once (an empty CSV that would have produced a spurious "no `ctp::` kernels launched"). Every count above comes from a report regenerated after deleting the `.sqlite`, with row counts verified non-empty.
- `ctest -R ctp_neuropress` — 6/6 pass, re-run at the end. Consistent with the position that these findings are about locality, not numbers.
- `ComputeDeviceStatsTyped` could not be reached via reachability route (a) (device chunk + non-NeuroPress dynamic mode) in any shipped binary; route (b), the SGD snapshot, is the one observed. Settling (a) needs a device-resident blob driven through `DynamicSchedule` with `dynamic_compress_ == 1`, which no shipped example does.
