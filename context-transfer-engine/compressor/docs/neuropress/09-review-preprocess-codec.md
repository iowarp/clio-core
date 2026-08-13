# Adversarial verification — PRE-1..14 and CODEC-1..10

Default posture: every finding is wrong until it survives an attempt to break it.

**Counts:** REFUTED 1 · CONFIRMED 6 · CONFIRMED-WITH-CORRECTION 17 · UNPROVEN 0.
**Severity movement:** 5 high -> 2 high (PRE-2, PRE-5 hold; PRE-1/PRE-3/PRE-4/CODEC-1 down), 1 raised (PRE-14), 1 zeroed (PRE-10). Net: 10 of 24 downgraded, 1 upgraded.

## Method note (applies throughout)

Everything below rests on runs, not reading:

- **gdb breakpoint counting** on the out-of-line symbols `ctp::compress::preprocess::{ByteShuffle,ByteUnshuffle,Quantize<float>,Dequantize<float>,ByteShuffleDevice,ByteUnshuffleDevice,QuantizeDevice,DequantizeDevice}` and on `IpcManager::{AllocateAndRegisterGpuBackend,FreeGpuBackend}` — all have real symbols in `build/bin/libclio_cte_compressor_runtime.so`, so host-vs-device execution is directly observable.
- **`nsys profile -t cuda`** + `cuda_gpu_kern_sum` / `cuda_api_sum`, stale `.sqlite` deleted and row counts verified non-empty each time.
- **gcov**: the build is `--coverage`-instrumented; counters redirected via `GCOV_PREFIX` into scratch so the build tree's `.gcda` were not polluted. Caveat found and worked around: GCC does not attribute lines inside `CLIO_TASK_BODY_BEGIN` coroutine bodies, so gcov was usable only for non-coroutine functions; gdb covered the rest.
- **Compiled probes** linking the shipped `libclio_ctp_cuda.so` and the installed cuSZ.

No source file in either repo was modified.

---

## PRE-1: CONFIRMED-WITH-CORRECTION
- **Disproof attempts:** Attacked reachability, which is where it breaks. Grepped every `Context::error_bound_` assignment tree-wide (including YAML/JSON/config/test) and inspected `CompressorConfig`. Ran gdb on `Quantize<float>` across `compressor_dynamic_neuropress`, `neuropress_compress_dir`, `neuropress_gpu_direct` (eb=0) and `neuropress_gpu_direct` with `CLIO_NEUROPRESS_ERROR_BOUND=0.01`.
- **Evidence:** Code claim holds at shifted lines — `compressor_runtime.cc:2271`, `:2300`, `:2309`. Host min/max scan `quantization.h:136-139`, quantize loop `:186-204` — accurate. Native has no host quantizer. **Host `Quantize<float>` breakpoint hits: 0 in all four runs.** Cost if ever reached, measured on 4 MiB float32: host `Quantize` **5.007 ms/call** vs `QuantizeDevice` **0.221 ms/call** — 22.7x.
- **Correction:** Lines drifted (2268->2271, 2277->2309). The reachability paragraph overstates: this branch is not entered by *any* shipped configuration. Both named routes are hypothetical — `StageInputIfNeeded` only stages down for a CPU codec, and NeuroPress's ranking never yields one for a device chunk.
- **Corrected severity:** medium

## PRE-2: CONFIRMED
- **Disproof attempts:** Ran exactly the proposed falsification — a counter in `ByteShuffle` vs `ShuffleKernel` during a `compress_dir` run — hoping to see the device branch taken. It was not. Took a backtrace to confirm the call site, and checked whether the host cost is material against the codec it precedes.
- **Evidence:** gdb on `./bin/neuropress_compress_dir`:
  ```
  40 HIT ByteShuffle
  40 HIT ByteUnshuffle
  (ByteShuffleDevice: 0, ByteUnshuffleDevice: 0, QuantizeDevice: 0)
  ```
  Backtrace:
  ```
  #0 ctp::compress::preprocess::ByteShuffle (input=0x7fc04060f288, num_bytes=4194304,
       elem_size=4, output=0x7fc0883f8010) at .../byte_shuffle.h:70
  #1 clio::cte::compressor::Runtime::Compress(...) at compressor_runtime.cc:2358
  ```
  Same on `compressor_dynamic_neuropress`: `4 HIT ByteShuffle`, 0 device. Manifest confirms independently — 40 shuffled / 24 not, matching the 40 hits exactly. **Cost:** 4 MiB elem=4 shuffle at **1.204 ms host vs 0.070 ms device (17.3x)**, byte-identical output. Median GPU codec time per chunk in the same run is **1.659 ms** — so the host shuffle adds ~73% on top of the codec it feeds, where the device kernel would add 4%.
- **Correction:** Lines shifted: `:2324-2332` -> **`:2356-2364`**; exploration twin `:1736-1746` -> **`:1765-1778`**. Native citation exact.
- **Corrected severity:** high

## PRE-3: CONFIRMED-WITH-CORRECTION
- **Disproof attempts:** Tried the proposed falsification (show `PutBlob` never interposed). It *is* — ctest #317/#318/#319 all exercise it. So attacked the other side: profiled the interposer test under nsys, and used gcov to confirm the host branch executes.
- **Evidence:** `nsys stats --report cuda_gpu_kern_sum interpose.nsys-rep` -> `SKIPPED: interpose.sqlite does not contain CUDA kernel data` — **zero CUDA kernels launched in the entire process**, while the test reports a completed lossy round trip (`worst |orig - dequantized| = 0.0474625, bound 0.05`). gcov:
  ```
       2  L3292   if (requested_quant && ctx.error_bound_ > 0.0 && size >= sizeof(float) &&
       2  L3308   if (requested_shuffle != 0) {
  ```
  `CompressIntoShm` contains no `IsDevicePointer` call at all.
- **Correction:** Lines shifted (`:3260-3262`->`:3292-3298`, `:3276-3283`->`:3308-3318`, call site `:3529-3530`->`:3562-3563`). Severity over-stated at high: every shipped caller of this path supplies a **CPU** `compress_lib_` (four call sites, all `= 1`), so host preprocessing feeding a host codec is not obviously the wrong place for the work. What is genuinely divergent is that native has no such path at all.
- **Corrected severity:** medium

## PRE-4: CONFIRMED-WITH-CORRECTION
- **Evidence:** gcov of `cte_compressor_interpose`:
  ```
       1  L3513   qr.quantized_.assign(quant_staging.begin(), quant_staging.end());
       1  L3521       ctp::compress::preprocess::Dequantize<float>(qr);
       1  L3530   std::memcpy(dst, restored.data(), restored.size() * sizeof(float));
   #####  L3498   if (!ctp::compress::preprocess::ByteUnshuffle(   // no shuffle in this test
  ```
  with zero CUDA kernels in the process. Three extra host buffers confirmed. Native counterparts device-only.
- **Correction:** Lines shifted (`:3466`->`:3498`, `:3481`->`:3513`, `:3488-3489`->`:3520-3521`, `:3498`->`:3530`, call site `:3616`->`:3648`). Same severity argument as PRE-3 — this is the CPU-codec interposer read path.
- **Corrected severity:** medium

## PRE-5: CONFIRMED-WITH-CORRECTION
- **Disproof attempts:** Tried to show `codec_dst` always device-resident — it is not. Confirmed caller identity with a backtrace rather than trusting the line number, since `ByteUnshuffle` has two call sites and the finding could have pointed at the wrong one.
- **Evidence:**
  ```
  #0 ctp::compress::preprocess::ByteUnshuffle (num_bytes=4194304, elem_size=4)
       at .../byte_shuffle.h:134
  #1 clio::cte::compressor::Runtime::Decompress(...) at compressor_runtime.cc:2929
  ```
  40 hits over 64 verification reads, on 4 MiB host buffers, 0 `ByteUnshuffleDevice`. The device branch above it does fire elsewhere — `neuropress_gpu_direct` shows `51 HIT ByteUnshuffleDevice` and `UnshuffleKernel<4>` x26.
- **Correction:** Lines shifted `:2895-2902` -> **`:2926-2937`**, gate `:2867-2868` -> **`:2898-2900`**. Severity: this is the read-side twin of PRE-2, fires on the same fraction of chunks (40/64) at the same measured cost — grading it medium while PRE-2 is high was inconsistent.
- **Corrected severity:** high

## PRE-6: CONFIRMED-WITH-CORRECTION
- **Disproof attempts:** Took the proposed falsification seriously — "show no other stream is ever in flight concurrently." Extracted per-stream kernel attribution and cross-stream overlap from the nsys SQLite for a run in which all four device entry points fire.
- **Evidence:** Every code citation verified character-for-character. Other streams *do* exist:
  ```
  stream=7   MinMax/Quantize/Dequantize/Shuffle/Unshuffle   (Clio preprocess, default stream)
  stream=13  StatsPass1/2Dev, EntropyFromHist, Finalize, InferKernelDeviceStats, RankKernel
  stream=14  nvcomp compress
  stream=16  nvcomp decompress
  ```
  Both stream 13 and nvcomp's are created with default flags, i.e. blocking against the legacy default stream — so the serialization is real in principle. **But the contention does not materialize:** of 1279 kernels only **12 cross-stream overlapping pairs**, and GPU busy time is 62.9 ms out of a 482.5 ms span (13%). The pipeline is already serialized by the caller.
- **Correction:** Call-site lines drifted (`:1673,:1726,:2249,:2315,:2879,:2918` -> `:1705, :1758, :2281, :2347, :2911, :2950`). Severity rests on contention measurement does not show.
- **Corrected severity:** low

## PRE-7: CONFIRMED-WITH-CORRECTION
- **Disproof attempts:** Attempted the proposed falsification by measuring both sync forms from the CUDA API trace of a run with 51 quantizes and 51 dequantizes.
- **Evidence:** Citations exact. Reachability confirmed exactly as claimed: **155 `cudaDeviceSynchronize` calls** = 51x2 (write) + 51 (read) + 2. But the cost is not where the finding implies: those 155 total **2.96 ms** (19 us each), against `cudaStreamSynchronize` 628 calls / 71.3 ms and `cudaMalloc` 475 calls / 106.4 ms.
- **Correction:** The `:861-863` citation is `:863-865`. The device-wide sync is measurably not a hot spot; it is a correctness-hazard-if-concurrent, not a demonstrated cost.
- **Corrected severity:** low

## PRE-8: CONFIRMED
- **Disproof attempts:** Tried the proposed falsification — "show the 8-byte malloc/copy/free is free relative to the kernel." It is the opposite of free.
- **Evidence:** Citations exact. **`QuantizeDevice` on 4 MiB costs 221 us end to end, of which `MinMaxKernel` is 10.6 us and `QuantizeKernel<int8>` 6.6 us — 92% of the call is allocation, blocking copies and syncs.** `cudaMalloc` median in this process is 110 us. Native's pooled overload verified at `quantization_kernels.cu:253-256` with `cudaMemcpyAsync` + one `cudaStreamSynchronize` at `:270-275`.
- **Corrected severity:** medium

## PRE-9: CONFIRMED-WITH-CORRECTION (and the correction matters)
- **Disproof attempts:** Ran the proposed falsification (show the allocations are pool-served — they are not). But in checking the *free* side, found the finding has the mechanism backwards.
- **Evidence:** Allocation confirmed: `ipc_manager.cc:3848` `base = ctp::GpuApi::Malloc<char>(bytes);` -> `gpu_api.h:197 cudaMalloc`, no pool. gdb: **346 `AllocateAndRegisterGpuBackend` calls** over 64 chunks in `neuropress_gpu_direct`; **0** in the host-path `compressor_dynamic_neuropress`.
  **The correction:** `FreeGpuBackend` does **not** call `cudaFree`.
  ```
  ipc_manager.cc:3947  void IpcManager::FreeGpuBackend(u32 gpu_id, const AllocatorId &alloc_id) {
  ipc_manager.cc:3950    if (gpu_ipc_) gpu_ipc_->UnregisterClientBackend(gpu_id, alloc_id);
  ipc_manager.cc:3952    // The actual ctp::GpuApi::Free relies on caller-tracked metadata — ...
  ```
  and `gpu2cpu_init_hip.cc:147-153` `UnregisterClientBackend` only erases a map entry. `grep -n "GpuApi::Free\|cudaFree" compressor_runtime.cc` returns nothing. Corroborated: **475 `cudaMalloc` vs 119 `cudaFree`**, and `nvidia-smi` sampled during the run shows GPU memory climbing 1653 -> 2635 MiB.
- **Correction:** "frees via `FreeGpuBackend` (cudaFree — itself device-synchronizing)" is **false**. There is no `cudaFree` on this path at all, so the claimed free-side sync cost does not exist — but the device scratch is **leaked**, ~4-6 MiB per chunk on the device path. That is a more serious defect than the one the finding describes, and outside the locality frame the audit used, so it appears nowhere else in these documents.
- **Corrected severity:** medium (as a locality finding); **the leak it mis-describes is separately high**

## PRE-10: REFUTED
- **Disproof attempts:** Read native's decompress epilogue in full instead of stopping at the pointer retarget the finding quotes.
- **Evidence:** The finding quotes `gpucompress_compress.cpp:1261-1262` and concludes "no copy". **Thirty-six lines later native does exactly the copy the port is accused of adding:**
  ```
  gpucompress_compress.cpp:1298    /* D->D copy to caller's output buffer */
                          :1299    cuda_err = cudaMemcpyAsync(d_output, d_result, header.original_size,
                          :1300                               cudaMemcpyDeviceToDevice, stream);
                          :1302    cuda_err = cudaStreamSynchronize(stream);
  ```
  Buffer count for a lossless shuffled blob: **native** = `cudaMalloc(d_decompressed)` + `byte_unshuffle_simple` allocates `d_unshuffled` + one full-size D2D -> 2 allocations, 1 full copy. **Port** = codec decompresses straight into the caller's buffer, unshuffle into one scratch, one full copy back -> 1 allocation, 1 full copy. The port does *fewer* allocations and the same number of full-size copies. For a quantized blob the port copies `quant_bytes` (1/4 the size at int8) where native copies full `original_size`.
- **Correction:** "native just retargets the pointer" is wrong — native retargets and then still performs a full-size blocking D2D. The port adds no copy relative to native.
- **Corrected severity:** none

## PRE-11: CONFIRMED-WITH-CORRECTION
- **Evidence:** Both citations exact. Extra copies confirmed at **`:2313-2314`**, **`:3298-3299`**, **`:1722-1723`**, **`:3513`**. The device path's `DeviceQuantizeParams` is scalars only, as conceded.
- **Correction:** Not an independent divergence — it is the data-structure shadow of PRE-1/PRE-3/PRE-4, and its only marginal cost beyond those is one extra host memcpy per chunk, small next to the 5 ms host quantize that produced it. Grading it medium double-counts.
- **Corrected severity:** low

## PRE-12: CONFIRMED
- **Disproof attempts:** The exhaustive version — grepped every `error_bound_` assignment across all source and config file types, read `CompressorConfig` in full, checked every `Context()` construction in the VOL, then *proved reachability from the other direction* by turning the one knob that exists and watching for the kernels.
- **Evidence:** Exactly two assignments exist tree-wide:
  ```
  compressor/example/neuropress_gpu_direct.cc:148  ctx.error_bound_ = kErrorBound;
  test/unit/test_compressor_interpose.cc:206       ctx.error_bound_ = kBound;
  ```
  `CompressorConfig` has no error-bound field. VOL passes default `Context()`; `core_tasks.h:1633` defaults `error_bound_(0.0)`. Positive control with `CLIO_NEUROPRESS_ERROR_BOUND=0.01`:
  ```
  51 HIT QuantizeDevice   51 HIT DequantizeDevice   26 HIT ByteShuffleDevice
  nsys: MinMaxKernel 51, QuantizeKernel<int8> 51, DequantizeKernel<int8> 51,
        ShuffleKernel<4> 26, UnshuffleKernel<4> 26
  ```
  Without the env var: `0 HIT QuantizeDevice`. The kernels are correct and reachable — through nothing but that one example's environment variable.
- **Correction:** Minor — the finding says "the only non-test assignment", which is right, but omits `test_compressor_interpose.cc:206`, which matters because it is what makes PRE-3/PRE-4 empirically observable.
- **Corrected severity:** medium

## PRE-13: CONFIRMED
- **Disproof attempts:** The flagged trap. Kernel absence is equally consistent with "the model never selected a shuffle action" — a different mechanism. So did not rely on kernel absence: ran the finding's own proposed falsification and checked *which* function fired.
- **Evidence:** The mechanism is exactly as claimed. In `neuropress_compress_dir`, shuffle actions **were** selected — 40 of 64 chunks with the shuffle bit set — and all 40 went to the **host** routine, called from `Runtime::Compress:2358` on a host pointer. `ByteShuffleDevice`: 0 hits. Staging cause at `neuropress_compress_dir.cc:176-177` then `:197-199`. The "device branch is live elsewhere" half also verified: `neuropress_gpu_direct` gives 51/51 device hits and 26+26 kernel instances; the VOL keeps device buffers device-resident when a compressor client is set.
- **Corrected severity:** medium

## PRE-14: CONFIRMED-WITH-CORRECTION (severity raised)
- **Disproof attempts:** Tried to show `output_fullptr.ptr_` always device-resident. It is on every runnable path. Rather than leave it latent, built a probe calling `DequantizeDevice` with a host destination directly, to test the finding's *claimed consequence* ("rc = 5").
- **Evidence:** The asymmetry is real — `:2900` guards the unshuffle with `IsDevicePointer`, while `:2950-2952` `DequantizeDevice(...)` has no pointer-kind test at all. Probe result:
  ```
  [PRE-14] calling DequantizeDevice(device_in, host_dst) ...
  [PRE-14] returned 0 ; cudaGetLastError = 700 (an illegal memory access was encountered)
  [PRE-14] subsequent cudaMalloc = 700 (...)  -> context POISONED (sticky error)
  ```
- **Correction:** The consequence is worse than described. Not a clean `rc = 5`: the kernel faults with `cudaErrorIllegalAddress`, which is **sticky** — every subsequent CUDA call in the process fails, including an unrelated `cudaMalloc`. In a long-lived compressor runtime that is a **process-wide GPU outage** triggered by one client passing a host destination. Reframes this from a "missing GPU work" nit into a latent availability bug.
- **Corrected severity:** high (latent — unreachable until an error bound is plumbed)

---

## CODEC-1: CONFIRMED-WITH-CORRECTION
- **Disproof attempts:** Spent the effort on the four reachability gates — read `Create()`'s load-failure handling and `IsReady()`, grepped every `target_psnr_` assignment, checked 72 real chunk selections across two configured runs.
- **Evidence:** List is CPU-only — `:776-782` emits wire ids 10, 10, 4, 1, 9; registry `compress_factory.h:485-495` gives `is_gpu=false` for all.
  **Gate 1 is materially misdescribed.** A failed `Load()` with a configured model path does **not** fall through:
  ```
  compressor_runtime.cc:435   HLOG(kError, "NeuroPress was requested (model path '{}') but could not be loaded...");
                        :441   task->SetReturnCode(1);
                        :442   CLIO_CO_RETURN;
  ```
  There is no state in which a *configured* NeuroPress silently degrades to the CPU list. The predictor is null only when `neuropress_model_path_` was never set — not a NeuroPress deployment, where using Clio's own heuristics is designed behavior, not substitution.
  **Gate 3 is dead.** `grep -rn "target_psnr_\s*="` returns exactly one hit, a serialization round-trip test. Nothing in production sets it.
  **Gate 2** is real but requires `CTP_ENABLE_NVCOMP=0`; the cache has it ON.
  Empirically: **0 of 72 chunks** took the fallback. The `compress_dir` manifest shows only `nvcomp-ans` (47), `nvcomp-cascaded` (5), `nvcomp-lz4` (2) and 10 stored-raw rows (`wire_id=0`, `ratio=1.0`) — the "brotli" label on those is a display artifact of the base-id->name lookup at `neuropress_compress_dir.cc:87-91`, not a CPU codec.
- **Correction:** The high rating rests on gate 1 being a silent degradation of a working NeuroPress deployment, and it is not — that case aborts pool creation.
- **Corrected severity:** low

## CODEC-2: CONFIRMED-WITH-CORRECTION
- **Disproof attempts:** Ran the proposed falsification — find a caller-visible signal distinguishing codec failure from an incompressible chunk. Found one the finding missed.
- **Evidence:** The shared branch is real. But `:2599-2610` explicitly distinguishes them and writes the distinction into the context returned to the caller:
  ```
  if (success) { task->context_.actual_compression_ratio_ = input_size / compressed_size; ... }
  else        { task->context_.actual_compression_ratio_ = 0.0;
                task->context_.actual_compress_time_ms_  = 0.0; }
  ```
  A caller reading `actual_compression_ratio_` sees 0.0 for a codec failure and a real measured ratio for a genuinely incompressible chunk.
- **Correction:** "The caller cannot tell a codec failure from a genuinely incompressible chunk" is **false** — only the *return code* is indistinguishable. Also this is not an execution-locality finding: nothing moves to the host, the bytes are stored raw. Filed under the wrong category.
- **Corrected severity:** low

## CODEC-3: CONFIRMED-WITH-CORRECTION
- **Disproof attempts:** Checked the one citation that distinguishes this from PRE-3/PRE-4 — that `compress_lib_` reaching `CompressIntoShm` can be an nvcomp id.
- **Evidence:** Host-preprocessing half confirmed (zero CUDA kernels in a quantized interposer round trip). **The distinguishing citation is wrong.** `test_compressor_functional.cc:1054` sets `compress_lib_ = CompLib::NVCOMP_LZ4` and then calls `AsyncCompress(...)` at `:1057-1060` — that is `Runtime::Compress`, which the finding itself says *does* have device branches. It never reaches `CompressIntoShm`. Every shipped caller that does reach `CompressIntoShm` uses a CPU codec.
- **Correction:** The cited proof of nvcomp-on-the-interposer-path does not prove it. Stripped of that, this is a duplicate of PRE-3 + PRE-4 with different line numbers.
- **Corrected severity:** medium (duplicate); the nvcomp-specific half: none

## CODEC-4: CONFIRMED-WITH-CORRECTION
- **Evidence:** Probe confirms the mechanism:
  ```
  [CODEC-4] cudaMallocManaged -> attributes.type = 3 (Managed);
            IsDevicePointer-equivalent (==Device) = 0 ;
            nvcomp IsDeviceAccessible-equivalent (Device||Managed) = 1
  ```
  Reachability: `grep -rn "kManagedUvm"` finds it only in the allocator switch, the admin memory-type mapping, the enum, and two CTP unit tests. **No shipped path allocates compressor blob data as managed memory.**
- **Correction:** One-line citation drift (`:288`->`:289`), and the severity assumes a memory kind nothing in the product hands to the compressor.
- **Corrected severity:** low

## CODEC-5: CONFIRMED-WITH-CORRECTION
- **Evidence:** Port side confirmed at shifted lines (`:1852-1855`, `:1963`, `:1969-1975`, `:1988-1992`). Native D2D winner write verified. Gate `:1603-1605`; `neuropress_exploration_enabled_ = false` by default and native matches (`g_exploration_enabled{false}`).
- **Correction:** Opt-in on both sides, off by default, additionally needs `error_pct > 0.50`, and the port's own comment states it is off the storage critical path.
- **Corrected severity:** low

## CODEC-6: CONFIRMED-WITH-CORRECTION
- **Evidence:** Three per-chunk allocations verified at `:2276-2278`, `:2343-2345`, `:2384-2386`, freed by `DeviceScratchGuard` at `:2236-2247`. Runtime counts: **346 calls over 64 chunks**; **0** on the host path. `cudaMalloc` 475 calls / **106.4 ms** in a process whose entire compress phase is 309 ms. The "outside the timed region" observation is correct.
- **Correction:** Same as PRE-9 — the implicit assumption that these are freed is wrong; `FreeGpuBackend` never calls `cudaFree`, so this is malloc churn *plus a leak*, not malloc/free churn.
- **Corrected severity:** medium

## CODEC-7: CONFIRMED
- **Evidence:** `compress_factory.h:199-202` confirmed; registry has no `wire_id == 0` entry, so **0 also resolves to zstd**. Native refuses instead (`compression_factory.cpp:143-145` throw; `gpucompress_compress.cpp:1192` invalid-header error).
- **Correction:** Consumer lines drift ~5 (`:2149, :2786, :3249, :3401`, plus `:1238, :1525, :1649, :2022` the finding did not list). The finding already grades this low and scopes it out of the NeuroPress ranking path — accurate.
- **Corrected severity:** low

## CODEC-8: CONFIRMED-WITH-CORRECTION
- **Disproof attempts:** An independent, harder reproduction rather than accepting either the claim or the original author's negative: three data regimes including one built specifically to generate Lorenzo outliers (periodic 9000.0 spikes in a +/-50 sine), decompressing the same archive twice into the same device buffer — once memset 0, once memset 0xAB. Also checked whether both sides even use the same predictor.
- **Evidence:** The factual omission is real, and both sides use Lorenzo (port `cusz.h:114`, native `external_compressors.cu:106`), so it is not a config artifact. Reproduction:
  ```
  mode=0 (smooth)  ok1=1 ok2=1  differing(clean vs dirty)=0/1048576
  mode=1 (spikes)  ok1=1 ok2=1  differing(clean vs dirty)=0/1048576
  mode=2 (noise)   ok1=1 ok2=1  differing(clean vs dirty)=0/1048576
  ```
  Two independent authors, three data regimes, zero observed divergence. (Independently reproducing the out-of-scope note: `maxerr_clean=0.100002` at `eb=1e-3` in Rel mode — ~100x the nominal bound.)
- **Correction:** Should survive only as a *defensive-code* difference, not a corruption risk. Presenting it as `missing-gpu-work` with a corruption rationale neither author could demonstrate overstates it. Reachability is narrow: cuSZ/cuSZp are outside the trained action space on both sides.
- **Corrected severity:** low (informational)

## CODEC-9: CONFIRMED
- **Evidence:** All twelve cited lines verified exactly. `grep -c CachedStream` on all three = 0, while `nvcomp.h:398-404` has it. Native takes the caller's stream with no `cudaStreamCreate` anywhere in the file.
- **Corrected severity:** low

## CODEC-10: CONFIRMED
- **Evidence:** Port loop `:1647`, codec call `:1814-1816`, strictly sequential; every alternative on a thread gets the same `CachedStream()`. Native launches K slots on K streams. The port's own comment at `:1595-1602` states the divergence outright. Same opt-in gate as CODEC-5, off by default on both sides.
- **Corrected severity:** low

---

## Additional check requested: the `byte_shuffle_cpu_stub.cc` clearing

The preprocess agent's clearing is **correct**, independently re-verified:

```
$ ar x libclio_ctp_compress_model.a byte_shuffle_cpu_stub.cc.o && nm -C byte_shuffle_cpu_stub.cc.o | grep " T \| W "
(no output)

$ nm -C --defined-only libclio_ctp_compress_model.a | awk '/\.o:$/{f=$0} /ByteShuffleDevice|QuantizeDevice/{print f"  "$0}'
data_stats_gpu_kernels.cu.o:  T ctp::compress::preprocess::QuantizeDevice(...)
data_stats_gpu_kernels.cu.o:  T ctp::compress::preprocess::ByteShuffleDevice(...)
```

The four device entry points come only from `data_stats_gpu_kernels.cu.o`. Not a false alarm to reinstate.

---

## Summary

| ID | Verdict | Corrected severity |
|---|---|---|
| PRE-1 | CONFIRMED-WITH-CORRECTION | medium |
| PRE-2 | CONFIRMED | high |
| PRE-3 | CONFIRMED-WITH-CORRECTION | medium |
| PRE-4 | CONFIRMED-WITH-CORRECTION | medium |
| PRE-5 | CONFIRMED-WITH-CORRECTION | high |
| PRE-6 | CONFIRMED-WITH-CORRECTION | low |
| PRE-7 | CONFIRMED-WITH-CORRECTION | low |
| PRE-8 | CONFIRMED | medium |
| PRE-9 | CONFIRMED-WITH-CORRECTION | medium |
| PRE-10 | **REFUTED** | none |
| PRE-11 | CONFIRMED-WITH-CORRECTION | low |
| PRE-12 | CONFIRMED | medium |
| PRE-13 | CONFIRMED | medium |
| PRE-14 | CONFIRMED-WITH-CORRECTION | **high** (raised) |
| CODEC-1 | CONFIRMED-WITH-CORRECTION | low |
| CODEC-2 | CONFIRMED-WITH-CORRECTION | low |
| CODEC-3 | CONFIRMED-WITH-CORRECTION | medium (duplicate) |
| CODEC-4 | CONFIRMED-WITH-CORRECTION | low |
| CODEC-5 | CONFIRMED-WITH-CORRECTION | low |
| CODEC-6 | CONFIRMED-WITH-CORRECTION | medium |
| CODEC-7 | CONFIRMED | low |
| CODEC-8 | CONFIRMED-WITH-CORRECTION | low (informational) |
| CODEC-9 | CONFIRMED | low |
| CODEC-10 | CONFIRMED | low |

### Blunt assessment

The two findings that survive at full strength are **PRE-2 and PRE-5** — the byte shuffle and unshuffle really do run as scalar host loops on the shipped NeuroPress path, on 40 of 64 chunks in `neuropress_compress_dir` and 4 of 8 in `compressor_dynamic_neuropress`, at 1.204 ms per 4 MiB against a 0.070 ms device kernel and a 1.66 ms median codec. That is the real finding in this pair of documents and it is well supported.

Most of the rest is over-graded. PRE-1/PRE-3/PRE-4 are correct code claims about paths that either no shipped configuration enters (PRE-1) or that only ever run with a CPU codec (PRE-3/PRE-4). PRE-6 and PRE-7 are stream-hygiene claims whose asserted contention does not appear in the trace — 12 overlapping kernel pairs out of 1279, 13% GPU occupancy. CODEC-1's "high" collapses once you read `Create()`.

### Three things the audit got wrong or missed that are worth more than most of what it found

1. **PRE-10 is backwards.** Native performs the same full-size D2D copy into the caller's buffer that the port is accused of adding (`gpucompress_compress.cpp:1298-1302`); the finding stopped reading 36 lines early. The port actually uses one fewer device buffer.

2. **`FreeGpuBackend` never frees.** PRE-9 and CODEC-6 both assume it calls `cudaFree` and cost it as a synchronizing free. It does not (`ipc_manager.cc:3947-3956` only unregisters a map entry). 346 `AllocateAndRegisterGpuBackend` calls in a 64-chunk run, 475 `cudaMalloc` against 119 `cudaFree`, ~1 GB of GPU memory growth during the run. **The device scratch is leaked** — a worse defect than the churn either finding describes, and it appears nowhere in these documents.

3. **PRE-14's consequence is a process-wide GPU outage, not a failed read.** `DequantizeDevice` with a host destination raises sticky `cudaErrorIllegalAddress`; the probe shows the very next unrelated `cudaMalloc` also returning 700. Latent behind PRE-12 today, but it should be graded as an availability bug.
