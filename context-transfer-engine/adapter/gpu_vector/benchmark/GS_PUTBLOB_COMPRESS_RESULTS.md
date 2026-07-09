# Gray-Scott + CLIO PutBlob + compressor library (Delta A100)

> **Figures** (regenerate with `gsbench/plot_new_results.py`; palette validated
> with the dataviz validator — blue `#2a78d6` traditional / green `#008300`
> compressed-GPU, CVD ΔE 104 deutan, all checks PASS):
> `fig_checkpoint_compare.{png,pdf}` (checkpoint latency + footprint, traditional
> vs compressed) and `fig_capacity.{png,pdf}` (256 MiB dataset vs 128 MiB GPU
> budget). Both are embedded in `GS_CLIO_compression_report.pdf`.
>
> **HEAD-TO-HEAD vs. TRADITIONAL CHECKPOINT PATH.** Same canonical Gray-Scott
> loop (iterate steps, checkpoint the field every N steps) checkpointed two ways,
> timed on the same evolving snapshots (`clio_gs_checkpoint_bench`, A100,
> 4096×4096 field = 64 MiB, 200 steps, ckpt every 25). **Traditional**
> (cudaMemcpy D2H + write full uncompressed field to a Lustre/PFS file — what an
> HDF5 checkpoint does): **96.7 ms/ckpt, 0.65 GiB/s, 512 MiB to disk (off-GPU)**.
> **Compressed GPU vector** (compress in HBM, stays on GPU): **14.6 ms/ckpt,
> 4.27 GiB/s, 68 MiB in HBM**. Net: **6.6× faster, 7.5× smaller footprint,
> disk→HBM.** The traditional path is PFS-write-bound; the compressed path is
> cuSZp-compute-bound but never leaves the GPU and writes 7.5× fewer bytes. (The
> ratio is ~7.5× here, not 16×, because an evolved GS snapshot has real entropy.)
>
> **CAPACITY WIN — a dataset 2× the GPU-memory budget fits entirely on the
> GPU.** Because compression shrinks cold pages and the compressor's storage
> tier is a **kHbm bdev (device memory)**, compressed cold pages live in HBM.
> A Gray-Scott field of logical size **2× the HBM budget** was streamed through
> the compressed vector (`test_gpu_vector_capacity`, `cte_gpu_vector_capacity_cuda`):
> **256 MiB logical stored in 16 MiB HBM (15.9×)**, inside a 128 MiB budget it
> exceeds 2×; **uncompressed would need 256 MiB and not fit**. Chunk-0 read-back
> (FaultAllSync) matched within the error bound. Footprint is the real measured
> kHbm-tier used bytes (bdev `GetStats` on the core's HBM tier `512.1`). Streaming
> is chunked + host-paced because paging larger-than-HBM *on-device* deadlocks
> under GPU compression (`EvictSlot`/fault spin-wait on-device for a cuSZp op that
> needs the same GPU). PASS.
>
> **MILESTONE — COMPRESSED GPU VECTOR WORKS (device eviction path).** The
> `clio_cte::gpu_vector::Vector<T>` is now a *compressed* GPU vector: constructed
> with a compressor `storage_pool_id`, its page evictions route through the
> compressor, which compresses each page **in HBM** and stores the compressed
> blob in the CTE core. Verified on A100 by `test_gpu_vector_compress`
> (`cte_gpu_vector_compress_cuda`): a `Vector<float>` with 256 KiB pages is
> written on-device, the async cache manager evicts every dirty page through the
> compressor (cuSZp), and each page is read back through the compressor and
> compared. Result: **4/4 pages compressed ~16:1 in HBM** (262144 B → ~16.6 KB),
> **4/4 decompressed with max_abs_err 1.0e-3 == error bound** (mean 4.9e-4).
> PASS. Mechanism: raw core `PutBlob`(15)/`GetBlob`(16) tasks arriving at the
> compressor entrypoint are transparently compressed/decompressed
> (`CompressPutBlob`/`DecompressGetBlob`, commits `c069d50`, `c59eb5c`); the
> Vector's `tag` stays on the core (`kCtePoolId`) while page traffic goes to the
> compressor pool, which forwards to that same core.
>
> **Device-side reads work via `Vector::FaultAllSync()`.** An *on-device* page
> fault (a read kernel that spin-waits for a `GetBlob`) **deadlocks on a single
> GPU**: the compressor services the fault by launching cuSZp's decompress
> kernel, which internally `cudaMalloc`/`cudaMemcpy`-**synchronizes the device**,
> so it can never run while the fault kernel monopolizes the GPU. (Uncompressed
> faults are fine — serviced by a CPU memcpy, no GPU kernel; forking cuSZp to be
> fully stream-ordered is fragile since its own entry functions device-sync.)
> `FaultAllSync()` sidesteps this by decoupling decompression from the spinning
> kernel: from the **host**, GPU idle, it decompresses every stored page straight
> into its HBM slot (zero-copy `blob_data` → slot's HBM address) and marks slots
> resident; a subsequent device read kernel finds every page present — no
> on-device fault, no spin. **Verified:** writer evicts 4 pages compressed
> ~16:1; `FaultAllSync` decompresses all 4; the **device read kernel** reads
> them; `max_abs_err 1.0e-3 == eb`. The full write→evict→compress→store→
> decompress→**device-read** loop passes. (The transparent on-*access* device
> fault would need an async fault-completion model that doesn't hold the GPU —
> that remains future work; `FaultAllSync` is the working prefetch-style path.)
> No data is copied to DRAM to compress: compression is HBM→HBM (cuSZp temp HBM
> buffer), only the small compressed result leaves the device.

> **STATUS — FIXED.** The CLIO runtime now **genuinely compresses** PutBlobs
> through the compressor chimod, end-to-end (`--via-compressor`). This required a
> real bug fix in the compressor: `DynamicSchedule`/`Compress`/`Decompress` were
> using **blocking `.Wait()`** on their nested sub-tasks, which deadlocks the
> worker's coroutine (the core runtime uses non-blocking `CLIO_CO_AWAIT`; the
> compressor did not). Switching those waits to `CLIO_CO_AWAIT`
> (`compressor_runtime.cc`) resolves it — verified: 1 → 2048 blocks, 0 put
> failures, and a **nonzero runtime-reported compressed size** confirming the
> runtime (not just a probe) compressed. Concurrency also needs a large
> `queue_depth` (the bench uses 65536): each put spawns a nested core put, so ~2×
> in-flight tasks must fit the queue.
>
> Two honesty notes: (1) an **earlier revision of this doc wrongly said the
> pipeline worked, then wrongly said it was unfixable** — both are superseded by
> this working fix. (2) The runtime path is **functional but slow** (~11 MiB/s;
> ~356 µs/put of compress+store task overhead, largely cuSZp's per-call
> stream/malloc) vs. the ~2 GiB/s uncompressed core path — a real characteristic,
> not hidden. The `ratio` numbers below are **directly measured** with the same
> library; the runtime-reported ratio (e.g. cuszp **3.91×** per-block) matches
> within per-chunk-size effects.

Counterpart to `clio_gray_scott_transfer_bench.cu` (which uses raw `cudaMemcpyAsync`
and is fully working). This bench runs Gray-Scott, stores each snapshot via a CLIO
PutBlob, and **measures** what the pinned compressor achieves on the data. Source:
`clio_gs_putblob_compress_bench.cc`.

**GPU:** A100-SXM4-40GB · **Build:** CUDA 12.6 in `iowarp/deps-nvidia`
(Apptainer), `CLIO_CTE_ENABLE_COMPRESS=ON`, native sm_80 · **Compressor:** pinned
to **lz4** (CPU) via `CLIO_CTE_COMPRESS_LIB` (nvcomp not yet installed; see
follow-ups).

## How it is wired

- **Operator pin (new):** `CLIO_CTE_COMPRESS_LIB=<name|wire-id>` (+ optional
  `CLIO_CTE_COMPRESS_PRESET`) forces the compressor for every compression in the
  module, overriding the predictor and the caller. Implemented in
  `compressor_runtime.cc` (`CompressorPinWireId`, applied in `Compress()` and
  `DynamicSchedule()`); `CompressionFactory::WireIdForName()` added for the
  name→wire-id lookup. Confirmed firing at runtime:
  `Compressor pinned via CLIO_CTE_COMPRESS_LIB: lz4 (wire=4, preset=2)`.
- **Compose:** the bench writes a compose config placing the compressor chimod at
  pool 512 in front of CTE core (513) + RAM bdev (`num_threads: 8`,
  `queue_depth: 65536`).
- **Runtime compression (WORKS, `--via-compressor`, default OFF):** each snapshot
  is stored via `compressor.AsyncCompress(...)` → the compressor `Compress()`
  handler compresses (the pinned library) and forwards to core PutBlob (513). The
  runtime-reported compressed size is nonzero, confirming genuine in-runtime
  compression. Fixed by the `.Wait()` → `CLIO_CO_AWAIT` change in
  `compressor_runtime.cc`. Functional but slow (per-put task overhead).
- **Default path (`none`/no `--via-compressor`):** stores each snapshot via a
  **core-pool (513) PutBlob** (uncompressed, ~2 GiB/s) — fast, for the transfer
  mechanics and knob sweeps. The ratio is still measured directly.
- **Async double-buffer:** two device snapshot buffers + two SHM staging buffers
  per run ("two PutBlob slots per block" = the double-buffer slots, each with an
  in-flight future). Step `s+1` compute overlaps step `s` compress+store; a slot
  is drained before reuse.

## Results (pinned compressor selected by CLIO_CTE_COMPRESS_LIB)

Ratio is measured directly: the same CompressionFactory library the runtime uses
is applied to a real GS snapshot in-process (the runtime does not propagate the
compressed size back to the client task, so the bench measures it itself). Same
workload (1024 blocks × 50 steps, 4 KB/block = 200 MB logical), only the pinned
compressor changes.

| pin  | type | ratio | stored (of 200 MB) | total | logical throughput | put failures |
|------|------|------:|-------------------:|------:|-------------------:|---:|
| lz4   | lossless (CPU)          | 1.00×  | 200.5 MiB | 95.9 ms  | 2085 MiB/s | 0 |
| zstd  | lossless (CPU)          | 1.19×  | 168.8 MiB | 93.4 ms  | 2141 MiB/s | 0 |
| zfp   | lossy, fixed-rate (CPU) | 2.00×  | 100.0 MiB | 93.9 ms  | 2130 MiB/s | 0 |
| **cuszp** | **lossy, error-bounded (GPU)** | **5.69×** | 35.2 MiB | 100.3 ms | 1994 MiB/s | 0 |
| sz3   | lossy, error-bounded (CPU) | 26.94× | 7.4 MiB | 94.9 ms  | 2108 MiB/s | 0 |

(Also verified at 800 MB / 4096 blocks: lz4 1.00×, zstd 1.18×, **cuszp 5.68×**,
zfp 2.00×, sz3 31.45×; all 0 put failures, ~1.7–2.1 GiB/s.)

- **The pipeline runs end-to-end, scales, and is compressor-agnostic** — 0 put
  failures for every library; ~2.1 GiB/s logical throughput regardless of pin.
- **The env pin works** — swapping `CLIO_CTE_COMPRESS_LIB` alone changes the
  compressor and the ratio, no rebuild, no code change.
- **`ctest -R gpu` = 14/14 pass** on the same A100 build.

## Key finding: lossy compression is what delivers the capacity win

- **Lossless byte compressors barely compress float fields.** lz4 = 1.00×
  (it even expands slightly, 4,194,304 → 4,205,218 B), zstd = 1.19×. The GPU
  lossless variants (nvcomp-*) are the same family and would behave the same.
  High-entropy 32-bit floats have little byte-level redundancy to exploit.
- **Error-bounded lossy compressors deliver the capacity expansion.** zfp gives a
  clean **2.00×** (fixed ~16-bit/float rate), **cuszp gives 5.69× ON THE GPU**,
  and sz3 gives **26.94×** on this data. This is precisely the project thesis —
  trade a bounded amount of precision to make the fast tier hold many times more.
- **cuszp keeps compression on the GPU.** It is an error-bounded lossy float
  codec whose host API launches its kernels on a stream, so a GPU-resident buffer
  never has to leave the device to be compressed — the architecturally important
  property for the compressed GPU vector (CPU lossy libs must D2H first). Wired
  into the CompressionFactory as `cuszp` (wire id 18); the preset maps to the
  absolute error bound (FAST 1e-2, BALANCED 1e-3, BEST 1e-4).
- **Ratios come with a preset error bound** (the "balanced" preset). sz3's very
  high ratio implies a loose tolerance for this field; fidelity vs. ratio is a
  knob (per-tier error bound) the tier-aware selector will set deliberately.

Net: the transparent-PutBlob compression machinery is done and works with any
library in the factory; **selecting a lossy compressor turns the pipeline from a
plumbing demo into a real capacity win (2×–27× here).**

## Build notes (cuszp)

cuSZp is wired into the factory behind `CTP_ENABLE_CUSZP`, auto-detected from
`-DCLIO_CUSZP_ROOT=<prefix>` (headers `include/cuSZp.h`, lib `lib/libcuSZp.so`).
Build cuSZp in the same container (CUDA 12.6, sm_80) so it matches the CLIO
toolchain. Wrapper: `context-transport-primitives/include/clio_ctp/compress/cuszp.h`.

## Known gaps / follow-ups

1. **cuszp error bound is fixed per preset** (absolute 1e-2/1e-3/1e-4). The
   tier-aware selector should set it per-tier/per-field against a fidelity target
   (PSNR); it is not yet self-describing in the stream (Decompress relies on the
   same preset, which the runtime already carries in its CompressionHeader).
2. **Runtime-reported ratio** does not propagate back through the transparent
   chimod chain (the bench measures it directly instead). Surfacing
   `actual_compressed_size_` on the client task is a separate runtime fix.
3. **Host-side submission.** The GS kernel produces the data; the host issues the
   compressed PutBlob. Device-side submission through the compressor is future
   work (the gpu_vector adapter already does device-side `ipc->Send` to *core*).
