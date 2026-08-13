# HDF5 integration layer — execution-locality audit (NATIVE vs PORT)

NATIVE: `/home/cc/NeuroPress` @ b23b8f6 — `src/hdf5/H5VLgpucompress.cu` (4311 lines)
PORT: `/home/cc/clio-core` — `context-transfer-engine/adapter/hdf5_vol/clio_vol.cc`

**10 findings.** High: 6 (VOL-1,2,3,4,5,6). Medium: 4 (VOL-7,8,9,10).

## Native ground truth (established first)

The NeuroPress VOL is **device-pointer-only by contract**. Both dataset callbacks test the user buffer with `gpucompress_is_device_ptr()` and `abort()` on a host pointer:
- `H5VLgpucompress.cu:3505-3556` (`dataset_write`) and `:3455-3502` (`dataset_read`).
- `:3544-3548` — `"gpucompress VOL FATAL: dataset_write[%zu] received a host pointer ... only accepts CUDA device pointers"` then `abort()`.
- Detection: `src/api/gpucompress_api.cpp:455-464` — `cudaPointerGetAttributes` / `attrs.type == cudaMemoryTypeDevice`.

On the default path (`VOLMode::RELEASE`; `detect_vol_mode()` `:66-73` returns RELEASE when `GPUCOMPRESS_VOL_MODE` unset) the write goes to `gpu_aware_chunked_write` (`:1782`), which slices per **HDF5 chunk** (`H5Pget_chunk`, `:1815`) directly off the device pointer with no copy for full-size contiguous chunks (`:2237-2240`, `wi.src = raw`) or a **device gather kernel** for non-contiguous ones (`:2268`); computes stats + NN inference on device (`:2311`, `:2331`); compresses on device (`:2034`/`:2044`); D2H-copies **only the compressed output** into a bounded 16-slot pinned pool (`:2080-2085`); and writes those compressed bytes via `H5VL_NATIVE_DATASET_CHUNK_WRITE` (`write_chunk_to_native`, `:1286-1307`, `filters = 0`).

Read (`gpu_aware_chunked_read`, `:3111`): compressed bytes are H2D'd (`:3290-3293`) and `gpucompress_decompress_gpu` writes **straight into the user's device buffer at the correct offset** (`:3279`, `:3303`), with a device `scatter_chunk_kernel` (`:3342`) for non-contiguous chunks.

Native's only full-raw-buffer D2H copies live in explicitly non-default paths: `gpu_bypass_dh_write` (`:2860`, only `GPUCOMPRESS_VOL_MODE=bypass`) and `gpu_fallback_dh_write` (`:2963`, only when the DCPL has **no** gpucompress filter — and it prints a loud warning, `:2997-3000`).

---

## VOL-1: Every H5Dwrite of a device buffer pays a full-dataset D2H copy that native's default path never pays
- **Claim:** The port always writes the raw, uncompressed data through to the native VOL in addition to (not instead of) staging compressed chunks, and because libhdf5 dereferences the buffer on the host, it first D2H-copies the **entire** dataset into a `std::vector<char>`. Native's RELEASE path writes only compressed chunk payloads and never moves the raw image to host.
- **Native evidence:** `H5VLgpucompress.cu:1286-1307` — `write_chunk_to_native(...)` sets `native_args.chunk_write.filters = 0;` / `.buf = buf;` and issues `H5VL_NATIVE_DATASET_CHUNK_WRITE`; the `buf` is the pinned host copy of the **compressed** chunk (`:2080-2085`, `comp_sz` bytes). No raw-image D2H anywhere in `gpu_aware_chunked_write`.
- **Port evidence:** `clio_vol.cc:1923-1933`
  ```
  const void *native_write_buf = buf[d];
  std::vector<char> native_write_staging;
  if (ctp::IsDevicePointer(buf[d])) {
    native_write_staging.resize(total_size);
    ctp::DeviceAwareMemcpy(native_write_staging.data(), buf[d], total_size);
    native_write_buf = native_write_staging.data();
  }
  herr_t rc = H5VLdataset_write(1, &dataset->obj.under_object, ...
  ```
  `total_size` is the whole dataset (`:1785`).
- **Category:** added-host-roundtrip
- **Severity:** high
- **Reachability:** Default, always. `H5Dwrite` -> `clio_dataset_write` (registered `:3141`) -> cacheable branch -> `:1925`. No env var or policy disables it — `clio_admit_policy()`/`clio_tier_accepting()` only gate the *staging* loop (`:1830`), never the native write at `:1930`. Also hit on the uncacheable branch (`:1732-1741`). The port's comment at `:1921` cites `gpu_bypass_dh_write` as precedent, but that native function is `VOLMode::BYPASS`-only (`:3521`), not default.
- **How to falsify:** Show a path through `clio_dataset_write` that returns successfully without executing `H5VLdataset_write` at `:1930` or `:1743` for a device `buf[d]`; or show native's RELEASE path also D2H-copies the full raw image (`:2084` is the only D2H in `gpu_aware_chunked_write` and it is `comp_sz` bytes).

## VOL-2: The port never hands the decompressor a device destination — the whole device decompress path is unreachable from the VOL
- **Claim:** On the cache-hit read path the port allocates a **host** SHM buffer as the decompression output and only afterwards copies to the user's (possibly device) buffer. Native decompresses directly into the user's device buffer.
- **Native evidence:** `H5VLgpucompress.cu:3277-3279` `dst_ptr = static_cast<uint8_t*>(d_buf) + off;` and `:3303-3305` `gpucompress_decompress_gpu(d_compressed, item.comp_sz, dst_ptr, &decomp_size, scatter_stream);` — `d_buf` is the user's `H5Dread` device pointer, from `:3478`.
- **Port evidence:** `clio_vol.cc:2036-2047`
  ```
  auto buffer = CLIO_IPC->AllocateBuffer(this_size);     // host SHM
  ctp::ipc::ShmPtr<> blob_data = buffer.shm_.template Cast<void>();
  futures.push_back(compressor_client->AsyncDecompressExplicit(..., blob_data, ...));
  ```
  then `:2055` `ctp::DeviceAwareMemcpy(dst + offset, buffers[i].ptr_, this_size);`. Downstream, `compressor_runtime.cc:2793-2812` sets `codec_dst = output_fullptr.ptr_` from `task->blob_data_` — that host SHM buffer — so `ctp::IsDevicePointer(codec_dst)` at `:2868` is false and the **host** unshuffle branch (`:2895-2910`) runs. Codec input is host too (`temp_buffer`, `:2655`, passed as `compressed_data` at `:2787`).
- **Category:** unreachable-device-path
- **Severity:** high
- **Reachability:** `H5Dread` -> `clio_dataset_read` (`:2303`) -> whole-read HIT branch (`:2511-2515`) -> `clio_read_cached_image` (`:2000`) -> `:2036`. Also `clio_serve_selection` -> `:2213`. `AllocateAndRegisterGpuBackend` appears in `clio_vol.cc` exactly once, at `:1840`, on the **write** side.
- **How to falsify:** Find a call site in `clio_vol.cc` building the `blob_data` argument to `AsyncDecompressExplicit`/`AsyncGetBlob` from `AllocateAndRegisterGpuBackend` rather than `CLIO_IPC->AllocateBuffer`; or run with `CLIO_NEUROPRESS_DECOMPRESS_TRACE=1` and a device `H5Dread` destination and show the device unshuffle branch executing.

## VOL-3: After a normal write->close->reopen, the port's H5Dread does not decompress at all — it serves the uncompressed HDF5 copy from host
- **Claim:** The port writes a coherence stamp at file close, but `clio_stamp_ambiguous()` **refuses** to write it whenever the file's mtime is younger than 10 ms — always true for a program that writes and closes. The next open sees `kAbsent`, deletes the tag (dropping every compressed chunk), so every read misses and is answered by the native uncompressed copy read into a host staging buffer and H2D'd. No GPU decompression occurs.
- **Native evidence:** `H5VLgpucompress.cu:3470-3480` — `dataset_read` dispatches on the DCPL filter to `gpu_aware_chunked_read`. The file holds compressed chunks (`write_chunk_to_native`, `:1286`), so there is no uncompressed copy to fall back to and no cross-process coherence state to invalidate.
- **Port evidence:**
  - `clio_vol.cc:969-984` — `clio_stamp_ambiguous()` returns true when `age_ns < clio_stamp_granularity_ns()`; default 10 ms (`:950`).
  - `clio_vol.cc:1000-1011` — on ambiguous, the stamp blob is **deleted** and the function returns without stamping.
  - `clio_vol.cc:1126-1140` — on reopen, `verdict != Stamp::kMatched` => `AsyncDelTag(tag_name)` + fresh empty tag.
  - `clio_vol.cc:2408-2427` — with the tag empty, `cached == 0`, so the MISS branch reads through native into `native_read_staging` (host `std::vector<char>`) and `:2508-2510` `ctp::DeviceAwareMemcpy(dst, native_read_staging.data(), total_size);`.
  - The port's **own** example states this outright — `adapter/hdf5_vol/example/neuropress_e2e.cc:282-286`:
    ```
    // Going through H5Dread does not test this. The VOL writes the plain data
    // to native HDF5 as well as staging compressed chunks into the tier, and
    // on reopen its coherence check finds no stamp blob, drops the tag, and
    // serves the read from the uncompressed HDF5 copy instead
    // (clio_vol.cc:1126, :2409). The bytes match, but nothing decompresses.
    ```
- **Category:** missing-gpu-work (also silent-cpu-fallback — the application sees a successful `H5Dread` with no indication)
- **Severity:** high
- **Reachability:** The default single-process write-then-reopen sequence — exactly what both examples (`neuropress_e2e.cc:246-351`, `neuropress_gpu_demo.cc:172-231`) and the unit test (`test_hdf5_vol_compressor_write.cc:110-160`) do. Non-default escapes: a very small `CLIO_VOL_STAMP_GRANULARITY_NS`, or keeping the file open for a same-process second read.
- **How to falsify:** Run `neuropress_e2e` with `CLIO_NEUROPRESS_DECOMPRESS_TRACE=1` and show `[clio-decompress]` lines emitted from inside the `H5Dread` window (not from the example's explicit `AsyncDecompressExplicit` loop at `neuropress_e2e.cc:307`); or show `clio_write_stamp` reaching `:1014` rather than returning at `:1011` for a just-written file.

## VOL-4: Silent host-SHM staging fallback demotes NeuroPress's selection features from GPU to CPU
- **Claim:** The port stages a chunk into device memory **only** when `IsDevicePointer(src) && dataset->file->compressor_client` and the GPU backend allocation succeeds. On any miss it D2H-copies into host SHM with no diagnostic, and `EstCompressionStats` then computes entropy/MAD/2nd-derivative on the **CPU**. Native computes those features only on the GPU and refuses the chunk otherwise; its uncompressed fallback prints a loud warning.
- **Native evidence:** `H5VLgpucompress.cu:2311-2313` `AutoStatsGPU* d_heur = gpucompress::runStatsKernelsNoSync(wi.src, wi.sz, infer_ctx->stream, infer_ctx);` and `:2331-2336` `gpucompress_infer_gpu(wi.src, wi.sz, ...)`. No host entropy/MAD routine in the VOL at all. `:2997-3000` — the only uncompressed fallback announces itself loudly.
- **Port evidence:** `clio_vol.cc:1827-1828` `const bool src_is_device = ctp::IsDevicePointer(src) && dataset->file->compressor_client;`; `:1854-1867` the fallback, whose own comment concedes the demotion (*"...even though it can no longer keep NeuroPress's candidate set GPU-only"*), then `auto buffer = CLIO_IPC->AllocateBuffer(this_size);` / `ctp::DeviceAwareMemcpy(buffer.ptr_, src + offset, this_size);`. No `HLOG`, no `stderr`, no return-code change. Downstream: `compressor_runtime.cc:615-616` -> false -> `:633-636` `ComputeCompressionFeatures(...)` -> `data_stats_gpu.h:141-146` host CPU loops.
- **Category:** silent-cpu-fallback
- **Severity:** high
- **Reachability:** Four independent default-reachable triggers:
  (a) **No compressor pool configured** — `clio_resolve_compressor_pool` returns `GetNull()` (`:826-840`) so `file->compressor_client` is null (`:1066`); `src_is_device` is then false for *every* write even from a device buffer, and the chunk is additionally never compressed at all (`:1905-1909`).
  (b) **Host input buffer** — the ordinary HDF5 application; `test_hdf5_vol_compressor_write.cc:128-134` is exactly this.
  (c) **Non-CUDA build** — the device block is inside `#if CTP_ENABLE_CUDA || CTP_ENABLE_ROCM || CTP_ENABLE_SYCL` (`:1836-1853`); `CMakeLists.txt:46-51` only links `ctp::cuda_cxx` when `CLIO_CORE_ENABLE_CUDA`.
  (d) `AllocateAndRegisterGpuBackend` failure at `:1840`.
- **How to falsify:** Show an `HLOG`/`stderr` emission on the `!staged_on_device` path; or show `ComputeCompressionFeatures` dispatching to a GPU kernel for a host pointer (`data_stats_gpu.h:124-147` — the `on_device` test is `IsDevicePointer(chunk)`, the else-branch is unconditionally host).

## VOL-5: Cache-served hyperslab reads do all gather/scatter on the CPU, over a host copy of the whole image — and `memcpy` into a device buffer
- **Claim:** `clio_serve_selection` materializes the **entire** dataset image in a host `std::vector<char>`, runs `H5Dgather`/`H5Dscatter` on the host, and in the `H5S_ALL` mem-space case uses a plain `std::memcpy` into the user buffer — which faults if that buffer is device memory. Native does the equivalent with a device `scatter_chunk_kernel`.
- **Native evidence:** `H5VLgpucompress.cu:3342-3346` `scatter_chunk_kernel<<<blocks, threads, 0, scatter_stream>>>(...)` (kernel at `:1180`), operating on `d_decompressed` -> `d_buf`, both device.
- **Port evidence:** `clio_vol.cc:2212-2237`
  ```
  std::vector<char> full(total_size);                       // whole image, host
  if (!clio_read_cached_image(dataset, total_size, full.data(), span_lo, span_hi)) break;
  ...
  std::vector<char> sel(sel_size);
  if (H5Dgather(fspace, full.data(), mem_type_id, sel_size, sel.data(), nullptr, nullptr) < 0) break;
  if (mem_space_id == H5S_ALL) {
    std::memcpy(buf, sel.data(), sel_size);                 // buf may be a device pointer
    ok = true;
  } else {
    clio_scatter_ctx ctx{sel.data(), sel_size};
    ok = (H5Dscatter(clio_scatter_cb, &ctx, mem_type_id, mem_space_id, buf) >= 0);
  }
  ```
  `:2232` is a raw `std::memcpy`, unlike the `ctp::DeviceAwareMemcpy` used everywhere else in this file (`:1739`, `:1865`, `:2055`, `:2102`, `:2288`, `:2509`).
- **Category:** host-compute
- **Severity:** high (locality regression plus a latent device-pointer fault)
- **Reachability:** `H5Dread` with any non-full selection -> `clio_dataset_read:2347` `if (!clio_read_is_whole(...))` -> `:2348` `clio_serve_selection(dataset, ..., buf[d])` — `buf[d]` passed through with no device check. Gated on `clio_cache_populated(dataset)` (`:2173`), so it fires only on a cache hit; combined with VOL-3 that is rare cross-process but reachable within one open file handle after a whole write.
- **How to falsify:** Find a device-pointer guard between `:2347` and `clio_serve_selection`; or show `H5Dgather` runs on the GPU.

## VOL-6: A partial / collective / type-mismatched H5Dwrite from a device buffer is silently demoted to a full D2H copy and stored uncompressed
- **Claim:** The port's uncacheable write branch D2H-copies the whole selection to host and hands it to the native VOL, performing **no** compression and **no** GPU work, with no diagnostic. Native applies the identical GPU compression pipeline regardless of selection shape.
- **Native evidence:** `H5VLgpucompress.cu:3517-3535` — the dispatch tests only `s_vol_mode` and the presence of the gpucompress filter in the DCPL; selection shape, collectivity and datatype are never consulted. Non-contiguous chunks handled at `:2262-2274` by `gather_chunk_kernel<<<...>>>`.
- **Port evidence:** `clio_vol.cc:1715-1746`
  ```
  if (!clio_cache_usable(dataset->file) || !dataset->cacheable ||
      !clio_is_whole_read(mem_space_id[d], file_space_id[d]) ||
      !clio_type_is_cacheable(mem_type_id[d]) ||
      !clio_type_matches_file(dataset, mem_type_id[d], dxpl_id) ||
      clio_is_collective(dxpl_id)) {
    ...
    if (ctp::IsDevicePointer(buf[d])) { ... DeviceAwareMemcpy(native_write_staging.data(), buf[d], total_size); ... }
    herr_t rc = H5VLdataset_write(1, ...);   // uncompressed, host
    continue;
  ```
- **Category:** silent-cpu-fallback / missing-gpu-work
- **Severity:** high
- **Reachability:** Default. Any hyperslab `H5Dwrite` (the normal incremental/parallel pattern), any MPI collective write (`clio_is_collective`, `:1383`), any compound/vlen/string type (`clio_type_is_cacheable`, `:1407`), or any file where the CTE runtime is unreachable (`clio_cache_usable`, `:446`). None warn.
- **How to falsify:** Show a warning/`HLOG` in the `:1715-1761` block, or show native also refuses to compress hyperslab writes.

## VOL-7: Per-chunk device allocation + blocking IPC-registration RPC to the daemon on the port's device write path; native reuses persistent buffers with zero copies
- **Claim:** For each 1 MiB chunk the port `cudaMalloc`s a fresh device buffer, takes a CUDA IPC handle, and blocks on a **TCP RPC round trip** to the runtime to register it, then does a synchronous D2D `cudaMemcpy`. Native's device compression buffers are allocated once per dataset and cached across writes, and full-size contiguous chunks are compressed in place with no copy.
- **Native evidence:** `H5VLgpucompress.cu:2237-2240` `/* Full-size contiguous chunk: no copy needed */ wi.src = raw; wi.sz = actual_bytes; wi.d_owned = NULL;`. Buffers session-persistent: `:1935-1975` allocates `wctx->d_comp_w[w]` (8 buffers, `M4_N_COMP_WORKERS = 8` at `:106`) and `wctx->io_pool_bufs[i]` (16 pinned, `:107`) only on first use or size growth; `:1926-1934` reclaims from a process-global cache.
- **Port evidence:** `clio_vol.cc:1830-1849`, inside `for (size_t i = 0; ... i < num_chunks; ++i)`:
  ```
  ctp::ipc::AllocatorId gpu_alloc_id =
      CLIO_IPC->AllocateAndRegisterGpuBackend(
          /*gpu_id=*/0, clio::run::gpu::IpcManager::MemKind::kDeviceMem,
          this_size, &device_ptr);
  if (!gpu_alloc_id.IsNull()) { ctp::GpuApi::Memcpy(device_ptr, src + offset, this_size); }
  ```
  `AllocateAndRegisterGpuBackend` from a client process: `context-runtime/src/ipc_manager.cc:3847-3850` (`cudaMalloc`), `:3910-3912` (`GetIpcMemHandle`), `:3923-3928` `auto reg_future = IpcCpu2CpuZmq::SendIn(this, reg_task, IpcMode::kTcp); reg_future.Wait();`. `GpuApi::Memcpy` is the synchronous `cudaMemcpy` (`gpu_api.h:249-256`), not the stream-async `DeviceAwareMemcpy`.
- **Category:** added-host-roundtrip
- **Severity:** medium
- **Reachability:** The port's *only* device-resident write path (`:1837`). The in-process short-circuit at `ipc_manager.cc:3867` (`CLIO_RUNTIME_MANAGER->IsRuntime()`) does **not** apply: the VOL is linked into the HDF5 application, a client process. A 1 GiB dataset issues 1024 blocking RPCs at the 1 MiB default (`clio_vol.h:25`).
- **How to falsify:** Show `IsRuntime()` is true inside an HDF5 application linking `clio_hdf5_vol`; or show the VOL hoists the allocation out of the loop.

## VOL-8: Bounded, reused device/pinned buffer pool became an unbounded per-chunk host allocation list
- **Claim:** Native holds a fixed pool (8 device compression buffers + 16 pinned host I/O buffers) with condition-variable back-pressure, persistent across writes. The port allocates one staging buffer **per chunk** and keeps every one alive until dataset close, so peak staging memory equals the whole dataset — and on the common path that memory is host SHM.
- **Native evidence:** `H5VLgpucompress.cu:1976-1978` `for (int w = 0; w < N_COMP_WORKERS; w++) d_comp_w[w] = wctx->d_comp_w[w]; for (int i = 0; i < N_IO_BUFS; i++) pool_free.push(wctx->io_pool_bufs[i]);` with `pool_acquire` blocking on `pool_cv` (`:1899-1904`). `VolWriteCtx` owned by the dataset (`o->write_ctx`, `:1922`).
- **Port evidence:** `clio_vol.cc:1863-1867` and `:1847` — `auto buffer = CLIO_IPC->AllocateBuffer(this_size); ... dataset->pending_buffers.push_back(std::move(buffer));`. Both vectors declared at `:161`/`:170`, drained only in `drain_dataset_puts` (`:339-388`), called from dataset/file close (`:1283`). No cap, no back-pressure on the loop at `:1830`.
- **Category:** device-struct-became-host
- **Severity:** medium
- **Reachability:** Default write path, both branches. Bounded only incidentally by `clio_tier_accepting()` (`:1805`), which skips staging entirely rather than throttling it.
- **How to falsify:** Find a per-dataset cap or a blocking acquire in the staging loop (`:1830-1912`), or show `pending_buffers` drained inside the loop rather than at close.

## VOL-9: The port never compresses per HDF5 chunk and has no GPU gather/scatter at all
- **Claim:** Native derives its compression unit from the DCPL chunk dimensions and gathers non-contiguous chunks with a device kernel. The port reads `H5Pget_chunk` only for telemetry and compresses fixed 1 MiB linear slices of a whole-dataset image; it contains no CUDA kernel of any kind.
- **Native evidence:** `H5VLgpucompress.cu:1814-1815` `if (H5Pget_chunk(o->dcpl_id, ndims, chunk_dims) < 0) goto done;` — drives `chunk_bytes` (`:1843`) and the chunk loop (`:2196`); gather kernel at `:1142`/`:2268`.
- **Port evidence:** `clio_vol.cc:1620-1621` is the only `H5Pget_chunk` call and it sits in the telemetry layout probe (stringified into a trace record at `:1679-1682`). The compression unit is `:1786-1787` `size_t chunk_size = dataset->file->chunk_size; size_t num_chunks = (total_size + chunk_size - 1) / chunk_size;` with `chunk_size` defaulting to 1 MiB (`clio_vol.h:25`).
- **Category:** missing-gpu-work
- **Severity:** medium
- **Reachability:** Always. Note this is only *partly* a locality regression: the port's linear slices are contiguous in the source buffer, so `src + offset` (`:1844`) needs no gather. The regression is that a non-whole-dataset write — the case that would need a gather — is bailed out to host instead (VOL-6), so the gather kernel is not merely absent, its workload is diverted to CPU.
- **How to falsify:** Show `dataset->chunk_dims` influencing `chunk_size`/`num_chunks`; or show the port handling a non-contiguous device selection without the `:1715` bail.

## VOL-10: No test or example in the port exercises a device pointer through the decompress path; the one that claims to bypasses HDF5 entirely
- **Claim:** The port has nothing that would catch VOL-2 or VOL-3. The tracked unit test uses host buffers only. `neuropress_compress_dir.cc` — the newest artifact and currently **untracked in git** — does not use HDF5 at all.
- **Port evidence:**
  - `context-transfer-engine/test/unit/adapters/hdf5_vol/test_hdf5_vol_compressor_write.cc:128-134` — `std::vector<int> wbuf(kNumElems); ... H5Dwrite(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, wbuf.data())`, host pointer; `:158-160` reads back into `std::vector<int> rbuf`. Per VOL-3 this passes whether or not anything decompressed.
  - `adapter/hdf5_vol/example/neuropress_compress_dir.cc` — UNTRACKED. Include list (`:43-58`) has `cuda_runtime.h`, `clio_runtime.h`, `core_client.h`, `compressor_client.h` and **no** `hdf5.h`; it calls `compressor.AsyncDynamicSchedule(...)` at `:212`. No `H5Fcreate`, `H5Dwrite` or `H5Pset_vol` anywhere.
  - `neuropress_e2e.cc:292-338` reaches the compressed data by calling `dec_client.AsyncDecompressExplicit` itself (`:307`) rather than through `H5Dread`, precisely because `H5Dread` does not decompress.
- **Category:** unreachable-device-path (evidence)
- **Severity:** medium
- **How to falsify:** Point to a test that passes a `cudaMalloc`'d pointer to `H5Dread` through the clio VOL and asserts the compressor's decompress path ran (via `LastCodecKernelMs()` or the decompress trace counter), rather than only asserting byte equality.

---

## Subcategories that yielded nothing

- **Device->host->device round trips forced by the CUDA-IPC hop on the *write* side:** checked and NOT found. When `src_is_device` holds and the GPU backend allocation succeeds, the device pointer survives into `DynamicSchedule` (`compressor_runtime.cc:1254-1256`) and on into `Compress`, which keeps stats (`:617-620`), quantize scratch (`:2242-2245`), shuffle scratch (`:2304-2311`) and the compressed **output** (`:2347-2366`, `:2460-2476`) device-resident. The `StageDeviceBlobForPut` demotion (`core/src/core_client.cc:45-60`) applies only to the **raw** `AsyncPutBlob` branch — which is why `clio_vol.cc:1827` deliberately requires `compressor_client` before staging on device. That gating is correct, not a defect.
- **`clio_vol_trace.h`:** zero GPU/CUDA references. Nothing to report.
- **A device-pointer detection function that is wrong or stubbed:** `ctp::IsDevicePointer` (`gpu_api.h:260-302`, wrapper `:564-566`) is a faithful `cudaPointerGetAttributes`/`cudaMemoryTypeDevice` check matching `gpucompress_api.cpp:455-464`. `CMakeLists.txt:46-51` links `ctp::cuda_cxx` so it is not the always-false stub in a CUDA build.
- **Compressed-bytes host staging on read:** the port fetches compressed bytes into host SHM (`compressor_runtime.cc:2655`), but native does the same via its pinned prefetch pool (`H5VLgpucompress.cu:3196-3199`) followed by an H2D (`:3290-3293`). Parity — not a finding. (The *output* side is VOL-2.)

---

**The headline:** native's VOL refuses host pointers outright and keeps the entire pipeline device-resident, writing only compressed chunks to the file. The Clio VOL is architecturally a *cache in front of an unmodified native HDF5 write*, so it must always produce a host-dereferenceable copy of the raw data — meaning every device-buffer `H5Dwrite` pays a full-dataset D2H that native never pays (VOL-1). Its device path exists on the write side only, gated behind a compressor pool being configured (VOL-4). On the read side there is no device path at all: the decompressor's destination is always host SHM (VOL-2), and in practice the coherence stamp is dropped on every ordinary write-then-close, so `H5Dread` serves the uncompressed HDF5 copy and decompresses nothing (VOL-3) — a fact the port's own example documents in a comment.
