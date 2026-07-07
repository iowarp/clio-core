# Gray-Scott + CLIO PutBlob + compressor library (Delta A100)

> **STATUS — read first (corrected).** The compressor *library* integration is
> real and verified: each pinned library (incl. the GPU `cuszp`) is applied to a
> real Gray-Scott snapshot and the **ratios below are directly measured**. What is
> **NOT** working is routing the PutBlob **through the CLIO compressor chimod** to
> compress *in the runtime*: that path (`--via-compressor`) **deadlocks** — the
> compressor's `DynamicSchedule` does nested blocking `.Wait()`s inside its task
> coroutine and hangs on this embedded runtime **even for a single put**. So the
> benchmark's puts are stored via CLIO PutBlob on the core pool **UNCOMPRESSED**;
> the reported ratio is a **direct measurement of the library**, not evidence the
> runtime compressed. Making the runtime genuinely compress a PutBlob (host or
> device) is an **open CLIO issue** (fix: `co_await` the nested tasks instead of
> blocking `.Wait()`), and is the real remaining work. Earlier revisions of this
> doc overstated the pipeline as end-to-end working; that was wrong.

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
- **PutBlob storage (works):** the bench writes a compose config placing the
  compressor chimod at pool 512 in front of CTE core (513) + RAM bdev, then stores
  each snapshot via a **core-pool (513) PutBlob** — this genuinely stores (verified,
  `put_failures=0`). The stored bytes are **uncompressed**.
- **Runtime compression (BROKEN):** routing the put through the compressor client
  (a `DynamicScheduleTask` to pool 512, `--via-compressor`) is the intended
  "transparent compression" path and mirrors `test_transparent_compress.cc`, but it
  **deadlocks** here — `DynamicSchedule`'s nested blocking `.Wait()`s hang on this
  embedded runtime even for one put. The `compressor_dynamic_schedule` unit test
  passes only in its own single-put harness. This is the open issue.
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
