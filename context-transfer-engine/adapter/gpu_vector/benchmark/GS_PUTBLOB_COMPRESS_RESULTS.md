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

## Results (pinned compressor = lz4, best-effort single run)

| config (blocks × steps, 4 KB/block) | logical | total | µs/step | logical throughput | put failures |
|---|---:|---:|---:|---:|---:|
| 256 × 20   | 20 MB  | 13.3 ms  | 665  | 1505 MiB/s | 0 |
| 1024 × 50  | 200 MB | 122.3 ms | 2446 | 1636 MiB/s | 0 |
| 4096 × 50  | 800 MB | 439.3 ms | 8786 | **1821 MiB/s** | 0 |

- **The pipeline runs end-to-end and scales** — 0 put failures across 5,120–
  204,800 compressed PutBlobs; logical throughput climbs to ~1.8 GiB/s as the
  per-step batch amortizes fixed overhead.
- **`ctest -R gpu` = 14/14 pass** on the same A100 build (prerequisite check).

## Known gaps / follow-ups

1. **Compression ratio not surfaced** (`ratio=0.00x`). `put_failures=0` and the
   pin firing confirm the compressor is in the path, but
   `actual_compressed_size_` does not propagate back to the client's PutBlobTask
   context through the transparent chimod chain. Next: read it from the CTE-core
   RAM-tier usage, enable compressor trace logs, or add a GetBlob round-trip
   byte-verify.
2. **GPU compressor.** nvcomp is absent from the container, so the pin currently
   selects a CPU library. Installing nvcomp + `CTP_ENABLE_NVCOMP` makes
   `CLIO_CTE_COMPRESS_LIB=nvcomp-lz4` a one-env-var swap (no code change).
3. **Host-side submission.** The GS kernel produces the data; the host issues the
   compressed PutBlob. Device-side submission through the compressor (the
   gpu_vector adapter already does device-side `ipc->Send` to *core*) is future
   work.
