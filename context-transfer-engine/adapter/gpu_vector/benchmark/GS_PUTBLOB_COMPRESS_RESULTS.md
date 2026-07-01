# Gray-Scott async-compressed PutBlob — CLIO runtime path (Delta A100)

The production-vector-shaped counterpart to `clio_gray_scott_transfer_bench.cu`:
each per-block Gray-Scott snapshot is shipped to storage through a **real CLIO
transparent compressed PutBlob** (compressor chimod → CTE core → RAM bdev),
instead of a raw `cudaMemcpyAsync`. Source: `clio_gs_putblob_compress_bench.cc`.

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
- **Transparent compression:** the bench writes a compose config placing the
  compressor chimod at the CTE entrypoint pool (512) in front of CTE core (513)
  and a RAM bdev, then issues a normal `cte_client->AsyncPutBlob(..., ctx)` with
  `ctx.dynamic_compress_=1`. This is the supported path (mirrors
  `test_transparent_compress.cc`); a hand-rolled `AsyncCompress` to a separate
  pool deadlocked and was the original hang.
- **Async double-buffer:** two device snapshot buffers + two SHM staging buffers
  per run ("two PutBlob slots per block" = the double-buffer slots, each with an
  in-flight future). Step `s+1` compute overlaps step `s` compress+store; a slot
  is drained before reuse.

## Results (pinned compressor selected by CLIO_CTE_COMPRESS_LIB)

Ratio is measured directly: the same CompressionFactory library the runtime uses
is applied to a real GS snapshot in-process (the runtime does not propagate the
compressed size back to the client task, so the bench measures it itself).

| pin  | config (blocks × steps, 4 KB/block) | logical | total | µs/step | ratio | logical throughput |
|------|---|---:|---:|---:|---:|---:|
| lz4  | 1024 × 50 | 200 MB | 93.3 ms  | 1866 | 1.00× | 2144 MiB/s |
| lz4  | 4096 × 50 | 800 MB | 409.9 ms | 8198 | 1.00× | 1952 MiB/s |
| zstd | 1024 × 50 | 200 MB | 98.8 ms  | 1975 | 1.19× | 2025 MiB/s |
| zstd | 4096 × 50 | 800 MB | 488.8 ms | 9776 | 1.18× | 1637 MiB/s |

- **The pipeline runs end-to-end and scales** — 0 put failures across 51,200–
  204,800 compressed PutBlobs; logical throughput ~1.6–2.1 GiB/s.
- **The env pin works** — lz4 vs zstd give visibly different ratios and timings
  from the identical workload, confirming the pin selects the compressor.
- **`ctest -R gpu` = 14/14 pass** on the same A100 build (prerequisite check).

## Key finding: lossless byte compressors barely compress float fields

The measured ratios are **~1.0× (lz4)** and **~1.19× (zstd)**. This is expected,
and it is the important lesson for the project's compressor selection: **lossless
byte-oriented compressors (lz4/zstd/snappy — and nvcomp's GPU variants, which are
also lossless) get almost nothing on high-entropy 32-bit float data.** (lz4 here
even expands slightly: 4,194,304 → 4,205,218 B.) The whole-domain-perturbation
init used to keep the field non-degenerate is deliberately high-entropy, which is
the worst case for lossless coding.

Real capacity gains on scientific float data require **error-bounded lossy**
compression (cuSZp / SZ / zfp), exactly as the Part-1 transfer benchmark used
(cuSZp targets ~4× on structured fields). So the transparent-PutBlob machinery is
compressor-agnostic and working; the *ratio* is a property of the chosen library
+ data, and a lossy GPU compressor is what turns this into a capacity win.

## Known gaps / follow-ups

1. **Wire an error-bounded lossy GPU compressor** (cuSZp/zfp) into the factory +
   pin to get a meaningful ratio on this data; lossless libraries are a
   correctness/plumbing demo only.
2. **nvcomp not installed** in the container (only zfp). Installing nvcomp +
   `CTP_ENABLE_NVCOMP` makes `CLIO_CTE_COMPRESS_LIB=nvcomp-*` a one-env-var swap —
   though note nvcomp is lossless too, so it will also be ~1× here.
3. **Runtime-reported ratio** still does not propagate back through the
   transparent chimod chain (the bench measures it directly instead). Surfacing
   `actual_compressed_size_` on the client task is a separate runtime fix.
4. **Host-side submission.** The GS kernel produces the data; the host issues the
   compressed PutBlob. Device-side submission through the compressor is future
   work (the gpu_vector adapter already does device-side `ipc->Send` to *core*).
