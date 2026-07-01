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
compressed size back to the client task, so the bench measures it itself). Same
workload (1024 blocks × 50 steps, 4 KB/block = 200 MB logical), only the pinned
compressor changes.

| pin  | type | ratio | stored (of 200 MB) | total | logical throughput | put failures |
|------|------|------:|-------------------:|------:|-------------------:|---:|
| lz4  | lossless          | 1.00×  | 200.5 MiB | 95.9 ms | 2085 MiB/s | 0 |
| zstd | lossless          | 1.19×  | 168.8 MiB | 93.4 ms | 2141 MiB/s | 0 |
| **zfp**  | **lossy (fixed-rate)**   | **2.00×**  | 100.0 MiB | 93.9 ms | 2130 MiB/s | 0 |
| **sz3**  | **lossy (error-bounded)** | **26.94×** | 7.4 MiB   | 94.0 ms | 2128 MiB/s | 0 |

(Also verified at 800 MB / 4096 blocks: lz4 1.00×, zstd 1.18×, ~1.6–2.0 GiB/s.)

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
  clean **2.00×** (its balanced preset is a fixed ~16-bit/float rate), and sz3
  gives **26.94×** on this data. This is precisely the project thesis — trade a
  bounded amount of precision for making the fast tier hold many times more data.
- **Ratios come with a preset error bound** (the "balanced" preset). sz3's very
  high ratio implies a loose tolerance for this field; fidelity vs. ratio is a
  knob (per-tier error bound) the tier-aware selector will set deliberately.

Net: the transparent-PutBlob compression machinery is done and works with any
library in the factory; **selecting a lossy compressor turns the pipeline from a
plumbing demo into a real capacity win (2×–27× here).**

## Known gaps / follow-ups

1. **GPU-side lossy compressor.** zfp/sz3 here run on the CPU (LibPressio). A
   GPU error-bounded compressor (**cuSZp**, already built at `/u/rpawar/cuSZp`;
   or zfp-sycl) would keep compression on-device and off the PCIe path. Wiring
   cuSZp into the factory (a `Compressor` subclass + registry row + build flag,
   mirroring `nvcomp.h`) is the natural next step; the env pin then selects it
   unchanged.
2. **Runtime-reported ratio** does not propagate back through the transparent
   chimod chain (the bench measures it directly instead). Surfacing
   `actual_compressed_size_` on the client task is a separate runtime fix.
3. **Host-side submission.** The GS kernel produces the data; the host issues the
   compressed PutBlob. Device-side submission through the compressor is future
   work (the gpu_vector adapter already does device-side `ipc->Send` to *core*).
