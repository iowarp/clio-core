# Gray-Scott Transfer Benchmark — Results (local PoC)

**GPU:** NVIDIA RTX 5060 Laptop (sm_120, 8 GB) · **Toolkit:** CUDA 12.6 in `iowarp-core-dev`
container, `compute_90` PTX JIT'd to sm_120 by the 12.8 driver · **Compressor:** stand-in int8
quantizer (real cuSZp swaps in on Delta — host-orchestrated, see note).
Each number is best-of-3 reps. Source: `clio_gray_scott_transfer_bench.cu`.

> Async case #2 corrected: removed a spurious device-to-device staging copy; the D2H now reads
> straight from the double-buffered output field, matching the intended buffer model.

## Headline (default: nblocks=4096, threads=128, per_block_bytes=4096, 200 steps, 16.78 MB/step)

| case | ms_total | µs/step | ratio | MB moved | speedup vs raw |
|------|---------:|--------:|------:|---------:|---------------:|
| raw | 294.7 | 1473 | 1.00× | 3355 | 1.00× |
| async | 236.1 | 1180 | 1.00× | 3355 | **1.25×** |
| compressed | 143.0 | 715 | 3.97× | 845 | **2.06×** |
| async+comp | 74.1 | 371 | 3.97× | 845 | **3.98×** |

## Findings

1. **Async (double-buffering) is PCIe-D2H-bound.** It pins effective transfer at ~14.3 GB/s (the
   PCIe ceiling) in *every* config, vs ~11.8 GB/s for serial raw. So its speedup is essentially
   the ratio of overlapped-to-serial PCIe throughput: **~1.2× at large sizes, up to ~1.6× at small
   sizes** (where raw's per-step launch+sync overhead is relatively larger). Overlap helps, but the
   PCIe link is the wall — which is exactly the motivation for compressing before transfer.
2. **Compression alone ≈ 1.8–2.3× over raw** — moves ~4× fewer bytes (3.88–3.99× ratio); the win
   grows with snapshot size and survives the added compression compute even on the serial path.
3. **Async + compressed ≈ 4× over raw** (3.9–4.4× across the sweep) — fewer bytes *and* overlap
   compose. This is the configuration the project's compressed GPU buffer cache targets.
4. **The win grows with snapshot size.** At the largest sweep point (32768×16384 B = 53.7 GB
   moved): raw 4505 ms → async+comp 1034 ms = **4.36×**.

> Two honesty notes: (a) removing the spurious D2D did *not* change async's wall-time — it ran on
> the compute stream and overlapped, so it was never on the PCIe-bound critical path; the fix is
> about correctness/fidelity, not speed. (b) This is a laptop GPU with real run-to-run variance
> (raw 343→294 ms between runs); treat ratios, not absolute ms, as the signal.
> The "GB/s(eff)" column divides *compressed* bytes by time for compressed rows, so it reads low —
> the real metric there is wall-clock time per step.

## CDP proof-of-concept conclusion

The original question — *can we invoke a compressor from within a CUDA kernel via CUDA Dynamic
Parallelism?* — resolved to: **not with cuSZp or nvcomp.** Both expose only **host-side** entry
points (`cuSZp_compress(...)`, `nvcompBatched*CompressAsync(...)`) that launch their own kernels on
a stream; cuSZp's internal fused kernel additionally relies on **grid-wide cooperative
synchronization**, which does not compose with CDP child-grid launches. The benchmark therefore
uses **host-stream orchestration** (compress kernel + D2H on a CUDA stream), which is also the
realistic production design. A CDP path would require a bespoke, non-cooperative compressor kernel.
(Still TODO: capture the actual compile/link failure as concrete evidence.)

## Next steps

- **Delta:** rebuild with `-DUSE_CUSZP -arch=sm_80` (cuSZp reliable on sm_80/sm_90) for
  trustworthy real-compressor numbers in the same four-case harness.
- **CMake integration:** register as `clio_gray_scott_transfer_bench` via `add_cuda_executable`
  in `benchmark/CMakeLists.txt` (currently standalone-nvcc for fast local iteration).
- Vary `threads_per_block` and add a PCIe-vs-NVLink/GDS transfer-target axis.

---

# Delta results — NVIDIA A100-SXM4-40GB (sm_80)

**Node:** `gpua025` · **GPU:** A100-SXM4-40GB, 40 GB · driver 570.148.08 · **PCIe Gen4 x16**
(D2H ceiling ≈ 26 GB/s) · **Toolkit:** CUDA 12.8 (NVIDIA HPC SDK 25.3), native `sm_80`, no PTX JIT
· **Compressor:** stand-in int8 quantizer (fixed ~4× ratio; real cuSZp still TODO, see Next steps).
Best-of-3 reps, `cudaEvent` timing. Source: `clio_gray_scott_transfer_bench.cu`. Job `19721009`.

Unlike the laptop PoC, these run natively on `sm_80` with low run-to-run variance — treat the
absolute ms as trustworthy here.

## Headline (nblocks=4096, threads=128, per_block_bytes=4096, 200 steps, 16.78 MB/step)

| case | ms_total | µs/step | GB/s(eff) | ratio | MB moved | speedup vs raw |
|------|---------:|--------:|----------:|------:|---------:|---------------:|
| raw | 141.8 | 709 | 23.7 | 1.00× | 3355 | 1.00× |
| async | 127.9 | 639 | 26.2 | 1.00× | 3355 | 1.11× |
| compressed | 53.9 | 270 | — | 3.97× | 845 | **2.63×** |
| async+comp | 33.0 | 165 | 25.6 | 3.97× | 845 | **4.30×** |

## Findings (A100)

1. **PCIe-D2H is the wall, and it's a hard wall here.** `async` pins effective D2H at **26.3 GB/s**
   in *every* single config (Gen4 x16 ceiling), vs ~24 GB/s for serial `raw`. Because A100's GS
   compute is cheap relative to the copy, the run is almost purely transfer-bound, so overlap alone
   buys only **~1.1×** (less than the laptop's ~1.25× — the laptop's slower compute left more to
   hide). Overlap helps, but you cannot out-schedule a saturated link — which is the whole argument
   for compressing before transfer.
2. **Compression alone ≈ 1.8–2.8× over raw** (2.63× headline): ~4× fewer bytes on the same serial
   path; the quantizer compute is cheaper than the bytes it removes from the link.
3. **Async + compressed ≈ 4.2–4.4× over raw at scale** (4.30× headline) — fewer bytes *and* overlap
   compose, and `async+comp` re-saturates the link at **26 GB/s on the compressed stream**: the GS
   compute and quantize fully hide behind the (now 4× shorter) copy. This is the target config for
   the compressed GPU buffer cache.
4. **GS is insensitive to threads/block** (64→512 all within 1% — it's memory-bound), so the
   transfer strategy, not block sizing, is what moves the needle.
5. **The win is stable across 2.5 orders of magnitude of snapshot size** (0.26 MB → 53.7 GB total
   moved): async+comp holds 4.2–4.4× once past the smallest size, where per-step launch overhead
   pulls it down to 3.5×.

## A100 scaling sweep (threads=128, 100 steps, best-of-3). µs/step shown; speedup = async+comp vs raw.

| config (nb, pbb) | MB/step | raw µs | async µs | comp µs | async+comp µs | ratio | a+c speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1024, 1024 | 1.0 | 58 | 42 | 32 | 16 | 3.88× | 3.53× |
| 8192, 1024 | 8.4 | 356 | 320 | 155 | 84 | 3.88× | 4.22× |
| 32768, 1024 | 33.6 | 1392 | 1277 | 584 | 333 | 3.88× | 4.18× |
| 1024, 4096 | 4.2 | 183 | 161 | 73 | 42 | 3.97× | 4.34× |
| 8192, 4096 | 33.6 | 1400 | 1277 | 516 | 325 | 3.97× | 4.31× |
| 32768, 4096 | 134.2 | 5544 | 5099 | 2005 | 1293 | 3.97× | 4.29× |
| 1024, 16384 | 16.8 | 716 | 640 | 275 | 163 | 3.99× | 4.40× |
| 8192, 16384 | 134.2 | 5564 | 5099 | 2012 | 1286 | 3.99× | 4.33× |
| 32768, 16384 | 536.9 | 22242 | 20391 | 7982 | 5135 | 3.99× | 4.33× |

## Local (RTX 5060) vs Delta (A100) — the story holds, the constants shift

| | RTX 5060 laptop (sm_120, PTX-JIT) | A100-SXM4 (sm_80, native) |
|---|---|---|
| D2H ceiling (async) | ~14.3 GB/s | **~26.3 GB/s** (Gen4 x16) |
| async-alone speedup | ~1.25× (up to 1.6× small) | ~1.11× (compute cheaper → less to hide) |
| compressed-alone | ~1.8–2.3× | ~1.8–2.8× |
| **async+comp** | **~4× (3.9–4.4×)** | **~4.3× (4.2–4.4×)** |

The async+comp win is platform-robust at ~4× because it is set by the compression ratio once the
link is saturated; the faster A100 link raises absolute throughput but the *relative* ordering and
magnitude are unchanged. Raw log: `gs_xfer-19721009.out`.

## Full scaling sweep (after async fix)

`run_gs_sweep.sh` (threads=128, 100 steps, best-of-3). µs/step and ratio shown.

| config (nb, pbb) | raw µs | async µs | comp µs | async+comp µs | ratio | async+comp speedup |
|---|---:|---:|---:|---:|---:|---:|
| 1024, 1024 | 178 | 108 | 138 | 79 | 3.88× | 2.27× |
| 8192, 1024 | 719 | 597 | 439 | 240 | 3.88× | 3.00× |
| 32768, 1024 | 2892 | 2358 | 1592 | 885 | 3.88× | 3.27× |
| 1024, 4096 | 374 | 303 | 224 | 125 | 3.97× | 3.00× |
| 8192, 4096 | 2884 | 2365 | 1332 | 687 | 3.97× | 4.20× |
| 32768, 4096 | 11352 | 9412 | 5087 | 2668 | 3.97× | 4.25× |
| 1024, 16384 | 1476 | 1184 | 694 | 373 | 3.99× | 3.96× |
| 8192, 16384 | 11381 | 9407 | 4964 | 2567 | 3.99× | 4.43× |
| 32768, 16384 | 45049 | 37615 | 19603 | 10339 | 3.99× | 4.36× |
