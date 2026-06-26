# CDP Proof-of-Concept — Can we compress from within a CUDA kernel?

Empirical answer to the original PoC question, on RTX 5060 (sm_120), CUDA 12.6 in the
`iowarp-core-dev` container (`compute_90` PTX JIT'd to sm_120, `-rdc=true`).

## Result: YES with a custom kernel, NO with cuSZp/nvcomp

| | Can it be launched from inside a kernel (CDP)? | Evidence |
|---|---|---|
| **cuSZp / nvcomp** | **No** | `cdp_hostapi_fail.cu` — compile error |
| **Custom non-cooperative compressor** | **Yes** | `clio_cdp_compress_demo.cu` — runs, verified |

### Part 1 — why cuSZp/nvcomp can't (`cdp_hostapi_fail.cu`)

Both expose only **host** entry points (`cuSZp_compress(...)`, `nvcompBatched*CompressAsync(...)`).
Calling one from `__global__` device code fails to compile:

```
cdp_hostapi_fail.cu(29): error: calling a __host__ function("cuSZp_compress_like(...)")
from a __global__ function("GsThenCompressParent") is not allowed
cdp_hostapi_fail.cu(29): error: identifier "cuSZp_compress_like" is undefined in device code
```

cuSZp's *internal* fused kernel is `__global__`, but it relies on **grid-wide cooperative
synchronization** (a global barrier across all blocks), which CDP child grids do not provide — so
even reaching past the host API doesn't yield a CDP-launchable compressor.

### Part 2 — a custom compressor that can (`clio_cdp_compress_demo.cu`)

Each Gray-Scott parent block computes its row, does `__threadfence()`, and (thread 0)
**launches a child quantizer kernel over its own row** via `QuantChildKernel<<<...>>>` from device
code. A block only compresses the row its own threads wrote (fenced before launch) → no cross-block
visibility hazard, **no grid-wide cooperation** — the exact property cuSZp lacks.

```
# rows=256 cols=1024  (parent grid: 256 blocks, each CDP-launches a child)
CDP child-launches executed : 256  (expected 256)
compression ratio           : 3.97x
max abs reconstruction error: 0.001264
RESULT: CDP compress-from-within-kernel WORKS
```

Build: `nvcc -O3 -rdc=true -gencode arch=compute_90,code=compute_90 -o cdp_demo
clio_cdp_compress_demo.cu -lcudadevrt` (CDP needs separable compilation + `cudadevrt`).

## Interpretation

- **Feasibility ≠ performance.** This proves the *mechanism* works; it does not show CDP is faster.
  CDP child launches carry real overhead, and the production transfer pipeline (see
  `GS_TRANSFER_RESULTS.md`) uses **host-stream orchestration**, which is simpler and avoids
  `-rdc=true`. CDP becomes interesting only if we want the GS kernel to trigger compression with
  **zero host round-trip** (e.g. a fully on-device, self-paging compressed buffer cache).
- **Design consequence:** if we ever want in-kernel compression, we need a **bespoke,
  non-cooperative** GPU compressor (block-local, like the demo's quantizer) — we cannot reuse
  cuSZp/nvcomp for that path. cuSZp/nvcomp remain the choice for the host-orchestrated path.
