# Adversarial verification — ORCH-1..16 and VOL-1..10

Default posture: every finding is wrong until it survives an attempt to break it.
Each finding's own "How to falsify" field was attempted first. Controlled experiments
(nsys kernel/memcpy profiles, decompress-trace runs with a causal control variable, and
a purpose-built device-pointer probe under gdb) were preferred over argument.

**Counts:** REFUTED 1 · CONFIRMED 14 · CONFIRMED-WITH-CORRECTION 11 · UNPROVEN 0.
**Severity after review:** high 5 (down from 12 as filed) · medium 7 · low 14.

---

## ORCH-1: CONFIRMED
- **Disproof attempts:** (a) Re-ran the nsys profile independently on `test_compressor_functional "[neuropress][693]"`, deleting the stale `.sqlite` and verifying a non-empty 21-row CSV. (b) Profiled `neuropress_gpu_demo`, which writes from a real `cudaMalloc`'d buffer through a compressor pool — the best chance to show the device path is the norm. (c) Re-located every cited line by content.
- **Evidence:** Dynamic ctest: **zero** `StatsPass1/2Kernel`, `EntropyFromHistKernel`, `MinMaxKernel`, `Quantize/Shuffle`, `InferKernelDeviceStats`; only `InferKernel` x8 + nvcomp kernels. `neuropress_gpu_demo` splits cleanly: 32 device-resident write chunks launch `StatsPass1Kernel<float>`x32 -> `InferKernelDeviceStats`x32 -> `RankKernel`x32, while 32 host-resident re-staged chunks launch `InferKernel`x32 with **no stats kernel at all**. Current code: `compressor_runtime.cc:657-663`, `:2300-2329`, `:2356-2365`, `:2379`, `:2392`; `nvcomp.h:199/215/226/230-234`.
- **Correction:** `compressor_runtime.cc` numbers in the filing are stale by ~30-100 lines; `data_stats_gpu.h:141-146` is exact.
- **Corrected severity:** high

## ORCH-2: REFUTED
- **Disproof attempts:** Ran the stated falsification ("show `gpu_weights_` can never be null once `is_ready_` is true") — it **succeeded**. Diffed against HEAD to check whether the auditor read an older revision.
- **Evidence:** `neuropress_nn_predictor.cc:157-180` — on `NeuroPressGpuLoad` returning null: `"clio ERROR: ... refusing to load. NeuroPress has no CPU path"`, then `weights_.clear(); biases_.clear(); is_ready_ = false; return false;`. The finding quotes this block as "prints one stderr warning then `is_ready_ = true; return true;`" — it read `:183-184`, which is **after** the `#endif` and only reached when `raw != nullptr`. `PredictBatch`'s GPU branch ends `return {};` (`:530-534`); the host forward pass lives inside `#else` (`:536-576`) and **is not compiled** when `CTP_ENABLE_NEUROPRESS_GPU=1`, which the build sets (`clio_ctp_compress_model.dir/flags.make`). `git diff` confirms both are in the audited tree.
- **Correction:** The claimed mechanism does not exist on any CUDA build. What survives: on a `CTP_ENABLE_NEUROPRESS_GPU=0` build the host forward pass is the implementation and nothing announces it — a configuration in which upstream cannot be built either. The sub-observation that `GpuInferenceActive()` has no runtime caller is true but moot, since `IsReady()` already implies it.
- **Corrected severity:** low

## ORCH-3: CONFIRMED-WITH-CORRECTION
- **Evidence:** NeuroPress list returned only `if (!neuropress_stats.empty())` at `:741-749`; the `HLOG(kError, ...)` at `:757-763` is guarded by `if (device_stats != nullptr && ...)`, so the host-stats fall-through is indeed **completely silent**. Fallback list `:776-782` = wire ids {10,4,1,9}, `is_gpu=false` for all (`compress_factory.h:484-494`); `pred.compression_ratio = 2.0` at `:826`.
- **Correction:** "Reachability: Default" is wrong and double-counts ORCH-16. With `neuropress_model_path_` unset there is no predictor, so the block at `:697` is never entered — nothing degrades. The genuinely-unlogged fall-through requires a *live* predictor plus one of: `NeuroPressGpuInferBatch` failing, `target_psnr_ > 0` (default 0, `core_tasks.h:1632`), or a build without nvcomp. Native's `GPUCOMPRESS_ERROR_NN_NOT_LOADED` also only applies to `ALGO_AUTO`.
- **Corrected severity:** low

## ORCH-4: CONFIRMED
- **Evidence:** `compressor_runtime.cc:2383-2393` — no log, no rc. `ipc_manager.cc:3839-3851` returns a null `AllocatorId` when `GpuApi::Malloc` fails. Native: `gpucompress_compress.cpp:~513-520` returns `GPUCOMPRESS_ERROR_OUT_OF_MEMORY`.
- **Correction:** Requires GPU-allocator exhaustion; consequence is a slower correct write, not a wrong one.
- **Corrected severity:** low

## ORCH-5: CONFIRMED-WITH-CORRECTION
- **Evidence:** `compressor_runtime.cc:2293-2299` and `:2351-2355` + `:2366-2367`. The SGD label at `:1551-1557` builds from the **ranked** `best_preset`, not `applied_shuffle`/`applied_quant`; PSNR at `:2458` only `if (applied_quant)`. Native errors at `gpucompress_compress.cpp:449-451`, `:466-469`.
- **Correction:** The reachability clause about CPU stubs is a non-sequitur — `byte_shuffle_cpu_stub.cc` is guarded by `#if !CTP_ENABLE_CUDA`, and on such a build `IsDevicePointer` is the always-false stub, so the device branch containing those calls is never entered. The header/read side records what actually ran, so this is a training-label fidelity defect, not a data-integrity one.
- **Corrected severity:** low

## ORCH-6: CONFIRMED-WITH-CORRECTION
- **Evidence:** `compressor_runtime.cc:2407-2416` `std::chrono` around `Compress(...)`; `:2422` `LastCodecKernelMs()` used **only** by `LogCompressedPayload` at `:2429-2432`; `:2449` `context.actual_compress_time_ms_ = compress_time;` feeding `cost(...)` `:1509-1511`, `error_pct` `:1512-1514`, SGD label `:1576`. Inside the window: `nvcomp.h:199/202/204/215/226/230-234`. Native uses events at `gpucompress_compress.cpp:530/548/558/561`. **Runtime finding:** every `[clio-decompress]` trace printed `kernel_ms=-1.000000`, because `KernelTimer::Enabled()` (`nvcomp.h:134-140`) requires `CLIO_CODEC_KERNEL_TIMING`.
- **Correction:** (1) The "false-parity comment" charge does not hold — the comment at `:2400-2405` says the port's *own* staging above the timer is excluded (true), and `:2417-2421` states outright that compress_time "also covers staging, allocation and the output copy, so it is not comparable with another implementation's kernel time". The port documents the gap rather than claiming parity. (2) The finding understates it: `LastCodecKernelMs()` is not merely "only logged" — by default it is never computed and returns -1.
- **Corrected severity:** medium

## ORCH-7: CONFIRMED
- **Evidence:** `compressor_runtime.cc:1384-1398` `feat_type = (data_type_ == 1) ? FLOAT32 : UINT8` then a fresh `ComputeCompressionFeatures`; inference-side type `:596-600` is `neuropress_active ? FLOAT32 : ...`; `data_type_` defaults to 0. `device_stats` is a local and never returned. **nsys of `neuropress_gpu_demo` shows both passes over the same 32 chunks with different element types:** `StatsPass1Kernel<float>`x32 + `StatsPass2DevKernel<float>`x32 (inference) **and** `StatsPass1Kernel<unsigned char>`x32 + `StatsPass2Kernel<unsigned char>`x32 (SGD).
- **Correction:** "host, by default" is right for an ordinary host chunk, but on a device chunk the second pass runs on the GPU — a duplicate device pass, not a host round-trip. The locality half is weaker than filed; the type-mismatch half is a real bug, now empirically proven.
- **Corrected severity:** medium

## ORCH-8: CONFIRMED
- **Evidence:** `compressor_runtime.cc:1647` sequential loop; `:1813-1820` `std::chrono`; single per-thread `CachedStream()` (`nvcomp.h:398-403`); `:1851-1853` D2H winner; `:1969-1976` three host memcpys. Native `gpucompress_compress.cpp:744-746`, `:795`, `:866-869`, `:884-892`, D2D at `:947-949`.
- **Correction:** Off by default on both sides — "medium" too high for an opt-in mode.
- **Corrected severity:** low

## ORCH-9: CONFIRMED-WITH-CORRECTION
- **Evidence:** `compressor_runtime.cc:3244-3383`, reached from `PutBlob:3547-3563` and `MultiPutBlob:3753-3765`.
- **Correction:** Reachability badly overstated. (1) The interposer is **commented out** in the shipped default config (`context-runtime/config/clio_default.yaml:210-214`); live only in `test_transparent_compress_config.yaml`. (2) This path performs **no NeuroPress selection at all** — it consumes a caller-supplied `ctx.compress_lib_`; comparing it to `gpucompress_compress_gpu` compares a Clio-only feature to NeuroPress. (3) `src` is host on every client-facing route: `core_client.h:606-614` calls `StageDeviceBlobForPut` for any non-runtime caller.
- **Corrected severity:** low

## ORCH-10: CONFIRMED-WITH-CORRECTION
- **Evidence:** `compressor_runtime.cc:3496-3532` host `ByteUnshuffle` + host `Dequantize<float>`; extra full copy at `:3513`. `GetBlob:3660-3677` adds a host memcpy at `:3675`.
- **Correction:** Same reachability correction as ORCH-9. The extra copy fires only for **quantized** blobs, requiring `error_bound_ > 0` which defaults to 0 — on the default path it never happens. `dst` is host SHM by construction, so no fault risk.
- **Corrected severity:** low

## ORCH-11: CONFIRMED-WITH-CORRECTION
- **Evidence:** Falsification attempted — `AsyncGetBlob` cannot deliver into a device buffer here: `:2687` unconditionally `AllocateBuffer` (host SHM). `:2726` host header dereference; `:2819` -> `:2865-2867` H2D again.
- **Correction:** The system-level contrast is much weaker than "added-host-roundtrip" implies: native's own VOL also brings compressed bytes through a pinned host pool and H2Ds them (`H5VLgpucompress.cu:3196-3199`, `:3290-3293`) — which the companion VOL doc itself files under "yielded nothing". Genuine residual delta: a 24-byte host header read vs a 64-byte D2H, plus nvcomp's per-call malloc+H2D inside the timed window (that is ORCH-6).
- **Corrected severity:** low

## ORCH-12: CONFIRMED-WITH-CORRECTION
- **Evidence:** Falsification **succeeded in part** — two pooled device-scratch objects exist that the finding missed: `data_stats_gpu_kernels.cu:209-228` `static thread_local DeviceStatsScratch` holding a persistent `cudaStream_t`, `d_hist`, `d_scalars`, `d_stats`; and `nvcomp.h:373-403` a thread-local LRU-3 manager cache plus persistent stream, structurally the same as native's `comp_mgr[LRU_DEPTH]`.
- **Correction:** "No port equivalent" is wrong. Genuinely missing: the *preprocessing/output* scratch (`d_preproc_quant`, `d_preproc_shuffle`, `d_cub_temp`, temp output) and the six CUDA events, allocated per chunk at `:2276-2278`, `:2343-2345`, `:2384-2386`. The supporting citation `compressor_runtime.h:264-278` proves only that the *chimod class* holds no device state — the pooling lives one layer down in CTP.
- **Corrected severity:** low

## ORCH-13: CONFIRMED
- **Evidence:** `clio_ctp/util/gpu_api.h:249-259` fully synchronous `cudaMemcpy`, not on the codec stream. Call sites `:2499-2506`, `:2921-2922`. Native `gpucompress_compress.cpp:610-611` async with sync deferred to `:1040`.
- **Correction:** For fairness — native itself uses a synchronous `cudaMemcpy` for the exploration-winner header (`:946-947`), so the async-vs-sync split is not absolute upstream either.
- **Corrected severity:** low

## ORCH-14: CONFIRMED-WITH-CORRECTION
- **Evidence:** Native's ~40-field `ChunkDiagInput` + `recordChunkDiagnostic` (`gpucompress_compress.cpp:1071-1160`). Port `:2550-2554` 7-field telemetry; `LogCompressionTelemetry:3037-3059` in-memory sink still TODO at `:3039-3042`.
- **Correction:** Three of the four "no X" clauses are false. `LogNeuroPressSelection` (`:1224-1275`) writes a **per-chunk 20-column record** including predicted ratio/comp-time/decomp-time/PSNR alongside actuals and the three input features — so "no per-chunk store" and "no predicted-ranking record" are wrong; `LogCompressedPayload` (`:1145-1195`) writes `compress_kernel_ms`, which **is** a CUDA-event field. Genuinely absent: an in-memory queryable store, and regret/MAPE (HLOG-debug only). The category "missing-gpu-work" is a stretch — no GPU work is missing.
- **Corrected severity:** low

## ORCH-15: CONFIRMED
- **Evidence:** `:1197-1203` env-gated; `:1443-1457` stages the whole chunk then FNV-loops it; `:1151-1157` stages the whole payload. `DeviceAwareMemcpy` = `cudaMemcpyAsync` + `cudaStreamSynchronize` (`util/gpu_api.h:522-524`). Native copies 24 bytes.
- **Correction:** None. Accurate and already self-limited to "diagnostic cost, honestly documented".
- **Corrected severity:** low

## ORCH-16: CONFIRMED-WITH-CORRECTION
- **Evidence:** Falsification run hard — grepped every `.yaml`/`.yml`/`.cc`/`.h` for anything setting `neuropress_model_path_` and read `CompressorConfig::LoadConfig` line by line. Only setters are three examples and one test. Fallback wire ids 10/4/1/9 all `is_gpu=false`. **Runtime confirmation:** the shipped VOL ctest logs `compressor_runtime.cc:456 ... "No compression predictor configured, dynamic compression prediction disabled"` and then compresses with CPU **zstd**. Native default is `GPUCOMPRESS_ALGO_LZ4`.
- **Correction:** (1) "`:428-432` logging at `kDebug` only" is stale — `Runtime::Create` now logs `kError` and **hard-fails the pool** (`SetReturnCode(1)`) at `:434-443` when a requested model fails to load. (2) The finding is weaker than the truth: `LoadConfig` (`compressor_tasks.h:161-207`) parses six `neuropress_*` keys but there is **no `neuropress_model_path` key at all**, so a compose-YAML deployment cannot turn NeuroPress on by any means. It is programmatic-API-only.
- **Corrected severity:** medium

---

## VOL-1: CONFIRMED
- **Disproof attempts:** (a) Traced every exit from `clio_dataset_write` for a device `buf[d]` that returns success without reaching `H5VLdataset_write`. (b) Grepped all D2H copies inside native's `gpu_aware_chunked_write`. (c) Measured it via nsys memory-transfer breakdown of `neuropress_gpu_demo` (64 MiB device write buffer).
- **Evidence:** `clio_vol.cc:1923-1933` unconditional on the cacheable branch; `:1730-1746` on the uncacheable one; the only early `continue`s are `:1708` and `:1782`, neither a successful write. Native's only D2H in `gpu_aware_chunked_write` is `comp_sz` bytes (compressed) plus a 4-byte entropy peek; `write_chunk_to_native:1286-1307` writes compressed bytes with `filters = 0`. **Measurement:** `bytes >= 32MiB` -> `(D2H, 67108864, 3)` and `(H2D, 67108864, 1)`. Two of the three 64 MiB D2Hs are the example's own verification copies; **one is the VOL's `native_write_staging` at `clio_vol.cc:1927`**; the single 64 MiB H2D is the read-miss `DeviceAwareMemcpy` at `:2509`.
- **Corrected severity:** high

## VOL-2: CONFIRMED
- **Disproof attempts:** (a) Grepped for every `AllocateAndRegisterGpuBackend` in `clio_vol.cc` — exactly one, line 1840, write side. (b) **Forced the cache-hit read path to execute** (`CLIO_VOL_STAMP_GRANULARITY_NS=1`, defeating VOL-3) with a device `H5Dread` destination, then profiled looking for the device unshuffle branch.
- **Evidence:** `:2036-2047` host SHM `blob_data` for `AsyncDecompressExplicit`, then `:2055` `DeviceAwareMemcpy`. Downstream `compressor_runtime.cc:2844` `codec_dst = output_fullptr.ptr_`, so `IsDevicePointer(codec_dst)` at `:2900` is false and the host branch `:2927-2942` runs. **Empirical proof:** in the forced cache-hit run, 32 `[clio-decompress]` calls executed and the round trip verified exactly, yet the kernel list contains `ShuffleKernel<4>`x34 (write side) and **zero `UnshuffleKernel` launches**. Shuffled blobs were unshuffled with correct results and no device kernel.
- **Correction:** Because of VOL-3 this path almost never executes in the default configuration, so practical exposure today is smaller than filed — a worse fact, not a better one.
- **Corrected severity:** high

## VOL-3: CONFIRMED — strongest result in the set; if anything the finding *understates* it
- **Disproof attempts:** Ran the falsification twice with a causal control. `test_hdf5_vol_compressor_write` (host buffers) and `neuropress_gpu_demo` (device buffers, real NeuroPress model) under `CLIO_NEUROPRESS_DECOMPRESS_TRACE=1`, then re-ran each with `CLIO_VOL_STAMP_GRANULARITY_NS=1` — the single variable the finding blames — to confirm the trace mechanism works and that stamp granularity is causal rather than coincidental. Verified the code, not the example's comment.
- **Evidence:**

  | run | granularity | `[clio-decompress]` calls | `Compress` calls | H5Dread time |
  |---|---|---|---|---|
  | `test_hdf5_vol_compressor_write` | default 10 ms | **0** | 8 (4 write + 4 read-miss re-stage) | — |
  | `test_hdf5_vol_compressor_write` | 1 ns | **4** | 4 | — |
  | `neuropress_gpu_demo` (device bufs) | default 10 ms | **0** | 64 (32 + 32 re-stage) | 2425 ms |
  | `neuropress_gpu_demo` (device bufs) | 1 ns | **32** | 32 | 304 ms |

  Both tests **PASS in every case** — byte equality is met either way, exactly as predicted. The demo prints `"H5Dread (NeuroPress decompress -> GPU buffer): 2425.59 ms"` while decompressing nothing. Code re-verified at the cited lines (no drift in this file): `clio_vol.cc:969-984`, `:1000-1011`, `:1126-1140`, `:2408-2427`, `:2508-2510`. Example comment verified verbatim at `neuropress_e2e.cc:282-286`.
- **Correction:** None. One addition the finding misses: on every such read the VOL **re-compresses the whole dataset** through the read-miss staging loop (`:2458-2498`), so the default write->close->reopen->read cycle pays for compression twice and decompression zero times.
- **Corrected severity:** high

## VOL-4: CONFIRMED-WITH-CORRECTION
- **Evidence:** `:1827-1828` and `:1854-1868` — the host-SHM fallback carries only a comment, no `HLOG`, no `stderr`, no rc change. `data_stats_gpu.h:121-147` else-branch is unconditional host loops. Trigger (a) verified: `clio_resolve_compressor_pool` returns `GetNull()` (`:826-843`) -> null client (`:1066-1075`) -> `src_is_device` false. Empirically shown in the demo profile: the 32 host-staged chunks reach `InferKernel` with no stats kernel.
- **Correction:** Two of the four "independent default-reachable triggers" are not demotions of GPU-resident data at all. (b) "host input buffer" — an ordinary `H5Dwrite` from `malloc` memory has nothing on the device to demote; native would `abort()` on that buffer. (c) "non-CUDA build" — same, no device exists. (a) is documented design (`:1821-1826`). The genuinely silent demotion of device-resident data is (d), allocation failure. Much narrower than "high".
- **Corrected severity:** medium

## VOL-5: CONFIRMED — upgraded from latent to reproduced
- **Disproof attempts:** Ran both falsifications (no device-pointer guard exists between `:2347` and `clio_serve_selection`; `H5Dgather` is host-only), then tried to prove the path *unreachable* by building a probe. It reproduced the fault instead. Wrote `vol5_probe.cc` in the job scratch dir (no repo file touched), linked against the existing build tree: whole-dataset `H5Dwrite` -> `H5Dflush` (no close, so VOL-3's stamp path is out of the picture and the cache is provably live) -> hyperslab `H5Dread` with `mem_space_id = H5S_ALL` into a `cudaMalloc`'d buffer.
- **Evidence:** Under gdb:
  ```
  Thread 1 "vol5_probe" received signal SIGSEGV, Segmentation fault.
  #0  0x00007ffff5188db7 in ?? () from /lib/x86_64-linux-gnu/libc.so.6
  #1  0x00007ffff7beffb4 in clio_serve_selection (..., buf=0x7fc0cb000000)
      at .../clio_vol.cc:2232
  #2  0x00007ffff7bf0d98 in clio_dataset_read (...) at .../clio_vol.cc:2348
  #3  H5VL_dataset_read () ... #5 H5Dread () ... #6 main () at vol5_probe.cc:100
  ```
  Crash inside `memcpy`, called from **exactly the cited line** `:2232`, reached from **exactly the cited call site** `:2348`. `clio_read_is_whole` (`:1493-1496`) requires *both* spaces full, so `mem_space=H5S_ALL` + hyperslab file space routes here; nothing in `cacheable_flat` (`:2321-2326`) checks device-ness. `:2232` is the only raw `std::memcpy` into a user buffer in the file — every sibling uses `ctp::DeviceAwareMemcpy`.
- **Correction:** The finding calls this "a latent device-pointer fault". It is not latent — it is a reproducible application crash on a legal HDF5 call sequence.
- **Corrected severity:** high

## VOL-6: CONFIRMED
- **Evidence:** `:1715-1761` five-way bail does a full-selection D2H at `:1732-1742` and forwards uncompressed, with no diagnostic of any kind. `clio_is_whole_read` (`:1342-1344`) requires both spaces literally `H5S_ALL`. Native dispatch (`H5VLgpucompress.cu:3510-3535`) consults only `s_vol_mode` and the DCPL filter; non-contiguous chunks go to `gather_chunk_kernel` (`:2262-2274`).
- **Correction:** Severity down from high — this is a correctness-preserving bail-out (bytes land in the authoritative file; the dataset is invalidated at `:1753-1755`), not a silent wrong answer. The defect is the absent diagnostic and the lost GPU work.
- **Corrected severity:** medium

## VOL-7: CONFIRMED
- **Evidence:** `:1830-1849` allocation inside the per-chunk loop, followed by synchronous `GpuApi::Memcpy`. `ipc_manager.cc:3847-3850`, `:3867` (`IsRuntime() && gpu_ipc_`), `:3911`, `:3927-3928` `reg_future.Wait()`. All four citations line-exact. **Runtime confirmation the RPC path is taken even in embedded-runtime examples:** `neuropress_gpu_demo` emits ~33 `TryLazyRegisterClientSegment: resolving (...) before its RegisterMemory round-trip landed` warnings, one per chunk allocation. Native reuses 8 device + 16 pinned buffers per session (`:1935-1978`) and is zero-copy for full-size contiguous chunks (`:2237-2240`).
- **Corrected severity:** medium

## VOL-8: CONFIRMED
- **Evidence:** `:154-170` declares the four pending vectors; `:1847` and `:1863-1867` push one entry per chunk with no cap; `drain_dataset_puts` (`:339`) called only from `H5Dflush`/`H5Dclose`/`H5Fclose`, never inside the loop. `clio_tier_accepting()` skips staging rather than throttling. Native's pool is fixed and back-pressured (`:1899-1904`, `:1976-1978`). Soft corroboration: a 1 GiB `neuropress_e2e` run was SIGKILLed (exit 137) at ~chunk 95 of 256 — consistent with, though not proof of, unbounded staging growth.
- **Corrected severity:** medium

## VOL-9: CONFIRMED
- **Evidence:** `H5Pget_chunk` appears exactly once, `:1621`, inside the telemetry probe; `chunk_dims` used only at `:1645` and `:1679-1682`. Compression unit `:1786-1787` from `file->chunk_size`, default 1 MiB (`clio_vol.h:25`). `grep -c __global__ clio_vol.cc` -> **0**. Native drives everything off `H5Pget_chunk`.
- **Correction:** None; the finding already concedes this is "only partly a locality regression". That self-limitation is correct.
- **Corrected severity:** low

## VOL-10: CONFIRMED-WITH-CORRECTION
- **Evidence:** `test_hdf5_vol_compressor_write.cc:128-134`/`:158-160` use `std::vector<int>` and assert only byte equality — proven above to pass with zero decompressions. `ctest -N` confirms neither `neuropress_gpu_demo` nor `neuropress_e2e` is registered as a test, so nothing in CI touches the device read path.
- **Correction:** Two factual errors. (1) `neuropress_compress_dir.cc` is **tracked** — `git ls-files` resolves it; committed in `8abe6473`. Its "no HDF5" property still holds. (2) The headline reads as if no example passes device pointers to `H5Dread`; both `neuropress_gpu_demo.cc:226-231` and `neuropress_e2e.cc` do. The claim survives only in its narrow (circular) sense: nothing reaches the decompress path because of VOL-3, and nothing asserts that it did. That narrow sense is the real point and it stands.
- **Corrected severity:** low

---

## Summary

| ID | Verdict | Corrected severity |
|---|---|---|
| ORCH-1 | CONFIRMED | high |
| ORCH-2 | **REFUTED** | low |
| ORCH-3 | CONFIRMED-WITH-CORRECTION | low |
| ORCH-4 | CONFIRMED | low |
| ORCH-5 | CONFIRMED-WITH-CORRECTION | low |
| ORCH-6 | CONFIRMED-WITH-CORRECTION | medium |
| ORCH-7 | CONFIRMED | medium |
| ORCH-8 | CONFIRMED | low |
| ORCH-9 | CONFIRMED-WITH-CORRECTION | low |
| ORCH-10 | CONFIRMED-WITH-CORRECTION | low |
| ORCH-11 | CONFIRMED-WITH-CORRECTION | low |
| ORCH-12 | CONFIRMED-WITH-CORRECTION | low |
| ORCH-13 | CONFIRMED | low |
| ORCH-14 | CONFIRMED-WITH-CORRECTION | low |
| ORCH-15 | CONFIRMED | low |
| ORCH-16 | CONFIRMED-WITH-CORRECTION | medium |
| VOL-1 | CONFIRMED | high |
| VOL-2 | CONFIRMED | high |
| VOL-3 | CONFIRMED | high |
| VOL-4 | CONFIRMED-WITH-CORRECTION | medium |
| VOL-5 | CONFIRMED | high |
| VOL-6 | CONFIRMED | medium |
| VOL-7 | CONFIRMED | medium |
| VOL-8 | CONFIRMED | medium |
| VOL-9 | CONFIRMED | low |
| VOL-10 | CONFIRMED-WITH-CORRECTION | low |

### Blunt notes
- **VOL-3 is right, and it is the headline.** A controlled single-variable experiment settles it: default 10 ms stamp granularity -> 0 decompressions and a *double* compression of the whole dataset; `CLIO_VOL_STAMP_GRANULARITY_NS=1` -> 32 decompressions and an 8x faster read. Both tests pass either way, and `neuropress_gpu_demo` prints "NeuroPress decompress -> GPU buffer" while decompressing nothing.
- **VOL-5 is worse than filed.** Reproduced a SIGSEGV at the exact cited line from a legal `H5Dread`. Not latent.
- **ORCH-2 is the false alarm.** The auditor read across a `#endif`: `Load()` hard-fails and the host network is not even compiled on a CUDA build. That was the second-highest-severity claim in the ORCH set and it is simply wrong.
- **ORCH-12 and ORCH-14 were written without looking one layer down.** The port does have persistent device scratch and an LRU-3 manager cache (in CTP, not the chimod), and it does record per-chunk predicted-vs-actual plus a CUDA-event kernel time.
- **ORCH-3/9/10/11 are true but graded on reachability that does not survive contact with the defaults** — the interposer is commented out of the shipped config and is not a NeuroPress path at all, and ORCH-3's "Default" reachability is really ORCH-16 counted twice.
- **Citation hygiene:** `clio_vol.cc`, `ipc_manager.cc`, `data_stats_gpu.h`, `neuropress_e2e.cc` and all NeuroPress citations are line-accurate. `compressor_runtime.cc` and `neuropress_nn_predictor.cc` citations are stale by 27-100 lines (the tree moved mid-audit); each was re-located by content and corrected above.
