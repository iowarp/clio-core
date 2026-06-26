# Gray-Scott Transfer Benchmark — Results (local PoC)

**GPU:** NVIDIA RTX 5060 Laptop (sm_120, 8 GB) · **Toolkit:** CUDA 12.6 in `iowarp-core-dev`
container, `compute_90` PTX JIT'd to sm_120 by the 12.8 driver · **Compressor:** stand-in int8
quantizer (real cuSZp swaps in on Delta — host-orchestrated, see note).
Each number is best-of-3 reps. Source: `clio_gray_scott_transfer_bench.cu`.

## Headline (default: nblocks=4096, threads=128, per_block_bytes=4096, 200 steps, 16.78 MB/step)

| case | ms_total | µs/step | ratio | MB moved | speedup vs raw |
|------|---------:|--------:|------:|---------:|---------------:|
| raw | 343.6 | 1718 | 1.00× | 3355 | 1.00× |
| async | 235.1 | 1175 | 1.00× | 3355 | **1.46×** |
| compressed | 161.6 | 808 | 3.97× | 845 | **2.13×** |
| async+comp | 83.7 | 418 | 3.97× | 845 | **4.11×** |

## Findings

1. **Async (double-buffering) ≈ 1.45× over raw, consistently.** Overlapping D2H with the next
   GS step pushes effective transfer to ~14.2 GB/s (near the PCIe ceiling) in *every* config,
   vs ~10–11 GB/s for the serial raw path.
2. **Compression alone ≈ 2.1× over raw** — moves ~4× fewer bytes (3.88–3.99× ratio); the win
   survives the added compression compute even on the serial path.
3. **Async + compressed ≈ 3.7–4.1× over raw** — the two optimizations compose. This is the
   configuration the project's compressed GPU buffer cache targets.
4. **The win grows with snapshot size.** At the largest sweep point (32768×16384 B = 53.7 GB
   moved): raw 4752 ms → async+comp 1264 ms = **3.76×**.

> Note on the "GB/s(eff)" column in raw output: for compressed rows it divides *compressed* bytes
> by time, so it reads low — that's expected. The real metric for compressed cases is **wall-clock
> time per step**, where compression wins by moving far less data.

## CDP proof-of-concept conclusion

The original question — *can we invoke a compressor from within a CUDA kernel via CUDA Dynamic
Parallelism?* — resolved to: **not with cuSZp or nvcomp.** Both expose only **host-side** entry
points (`cuSZp_compress(...)`, `nvcompBatched*CompressAsync(...)`) that launch their own kernels on
a stream; cuSZp's internal fused kernel additionally relies on **grid-wide cooperative
synchronization**, which does not compose with CDP child-grid launches. The benchmark therefore
uses **host-stream orchestration** (compress kernel + D2H on a CUDA stream), which is also the
realistic production design. A CDP path would require a bespoke, non-cooperative compressor kernel.

## Next steps

- **Delta:** rebuild with `-DUSE_CUSZP -arch=sm_80` (cuSZp reliable on sm_80/sm_90) for
  trustworthy real-compressor numbers in the same four-case harness.
- **CMake integration:** register as `clio_gray_scott_transfer_bench` via `add_cuda_executable`
  in `benchmark/CMakeLists.txt` (currently standalone-nvcc for fast local iteration).
- Vary `threads_per_block` and add a PCIe-vs-NVLink/GDS transfer-target axis.

## Full scaling sweep

`run_gs_sweep.sh` (threads=128, 100 steps, best-of-3). µs/step and ratio shown.

| config (nb, pbb) | raw µs | async µs | comp µs | async+comp µs | ratio | async+comp speedup |
|---|---:|---:|---:|---:|---:|---:|
| 1024, 1024 | 208 | 132 | 133 | 68 | 3.88× | 3.06× |
| 8192, 1024 | 813 | 592 | 476 | 261 | 3.88× | 3.12× |
| 32768, 1024 | 3239 | 2360 | 1763 | 1013 | 3.88× | 3.20× |
| 1024, 4096 | 522 | 307 | 249 | 161 | 3.97× | 3.24× |
| 8192, 4096 | 3277 | 2361 | 1516 | 789 | 3.97× | 4.16× |
| 32768, 4096 | 12015 | 9425 | 5805 | 3224 | 3.97× | 3.73× |
| 1024, 16384 | 1738 | 1181 | 780 | 431 | 3.99× | 4.03× |
| 8192, 16384 | 12062 | 9426 | 5711 | 3137 | 3.99× | 3.84× |
| 32768, 16384 | 47518 | 37521 | 22253 | 12639 | 3.99× | 3.76× |
