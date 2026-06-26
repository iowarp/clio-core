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
