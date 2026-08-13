# CUDA kernel inventory — native NeuroPress vs Clio port

Gathered independently of the six cross-examination agents, by grepping `__global__`
in both trees and by `cuobjdump -elf` on Clio's built `bin/libclio_ctp_cuda.so`.
Use this as a cross-check: a native production kernel with no port counterpart is a
strong prior for "GPU work missing / moved to host", and any agent finding that
contradicts this table deserves extra scrutiny.

## Native production kernels (`/home/cc/NeuroPress`)

| Kernel | File:line | Area |
|---|---|---|
| `statsPass1Kernel` | src/stats/stats_kernel.cu:106 | stats |
| `madPass2Kernel` | src/stats/stats_kernel.cu:201 | stats |
| `finalizeStatsOnlyKernel` | src/stats/stats_kernel.cu:260 | stats |
| `histogramKernel` | src/stats/entropy_kernel.cu:48 | entropy |
| `histogramKernelVec4` | src/stats/entropy_kernel.cu:99 | entropy (vectorized) |
| `entropyFromHistogramKernel` | src/stats/entropy_kernel.cu:167 | entropy |
| `nnInferenceKernel` | src/nn/nn_gpu.cu:280 | NN |
| `nnFusedInferenceKernel` | src/nn/nn_gpu.cu:418 | NN |
| `nnSGDKernel` | src/nn/nn_gpu.cu:620 | NN learning |
| `nnBatchedDecompSGDKernel` | src/nn/nn_gpu.cu:2382 | NN learning (batched) |
| `quantize_linear_kernel` | src/preprocessing/quantization_kernels.cu:55 | preprocess |
| `dequantize_linear_kernel` | src/preprocessing/quantization_kernels.cu:84 | preprocess |
| `verify_error_bound_kernel` | src/preprocessing/quantization_kernels.cu:106 | preprocess |
| `byte_shuffle_kernel_specialized<1,2,4,8>` | src/preprocessing/byte_shuffle_kernels.cu:21 | preprocess |
| `byte_unshuffle_kernel_specialized<1,2,4,8>` | src/preprocessing/byte_shuffle_kernels.cu:74 | preprocess |
| `populateChunkArraysKernel` | src/preprocessing/byte_shuffle_kernels.cu:140 | preprocess |
| `gather_chunk_kernel` | src/hdf5/H5VLgpucompress.cu:1142 | HDF5 VOL |
| `scatter_chunk_kernel` | src/hdf5/H5VLgpucompress.cu:1180 | HDF5 VOL |
| `traceQualityKernel` | src/hdf5/H5VLgpucompress.cu:2578 | HDF5 VOL diagnostics |

(`gs_init_kernel` / `gs_step_kernel` are the Gray-Scott simulator, not part of the
compression pipeline — out of scope.)

## Clio port production kernels

All of them live in exactly two files:

`context-transport-primitives/src/compress/preprocess/data_stats_gpu_kernels.cu`
| Kernel | Line |
|---|---|
| `StatsPass1Kernel` | 32 |
| `StatsPass2Kernel` | 92 |
| `StatsPass2DevKernel` | 120 |
| `EntropyFromHistKernel` | 155 |
| `FinalizeFeatureStatsKernel` | 184 |
| `ShuffleKernel` | 459 |
| `UnshuffleKernel` | 486 |
| `QuantizeKernel` | 595 |
| `DequantizeKernel` | 610 |
| `MinMaxKernel` | 637 |

`context-transport-primitives/src/compress/model/neuropress_nn_gpu_kernels.cu`
| Kernel | Line |
|---|---|
| `InferKernel` | 415 |
| `InferKernelDeviceStats` | 453 |
| `RankKernel` | 549 |
| `SGDKernel` | 822 |

> **CORRECTION (added after the verification pass).** This table originally listed
> three kernels and omitted `RankKernel`. That was wrong, and it matters: `RankKernel`
> performs the cost model, *both* `-INFINITY` masks and upstream's 32-lane bitonic
> sort **on the GPU**, which directly undercuts findings NN-3 and NN-4.
>
> The omission has a cause worth recording: **this working tree is being edited by
> another session while the audit runs.** `RankKernel` is present in the working tree
> and in the rebuilt `bin/libclio_ctp_cuda.so`
> (`_ZN3ctp8compress5model3gpu10RankKernelEPKfS4_S4_S4_S4_idddddddPiPd`) but is **not in
> git HEAD** (`git show HEAD:…neuropress_nn_gpu_kernels.cu | grep -c RankKernel` -> 0).
> The inventory above was taken before that edit landed.
>
> Consequence for every document in this directory: **port-side line numbers drift.**
> `neuropress_nn_gpu_kernels.cu` and `neuropress_nn_predictor.cc` each gained ~30 lines
> mid-audit; `compressor_runtime.cc` citations run 27-100 lines stale. Native
> (`/home/cc/NeuroPress` @ b23b8f6) citations are stable and were confirmed accurate.
> Findings should be located by content, not by line number.

Confirmed present in the shipped binary (`cuobjdump -elf bin/libclio_ctp_cuda.so`),
with `StatsPass1/2`, `StatsPass2Dev`, `Quantize`, `Dequantize`, `Shuffle`,
`Unshuffle` instantiated for the expected type/width sets.

Port kernels found only in EXAMPLE or TEST code, not production:
- `CompareBuffers`, `CompareWithinBound`, `FillRegimes` —
  `context-transfer-engine/compressor/example/neuropress_gpu_direct_kernel.cu`
- `GenerateKernel` — `context-transfer-engine/adapter/hdf5_vol/example/neuropress_e2e_kernel.cu`
- `GsStepKernel` — `context-transfer-engine/adapter/hdf5_vol/example/neuropress_gpu_demo_kernel.cu`
- `FillDataset` — the dataset parity harness

## Native kernels with NO port counterpart (structural leads only — must be verified)

1. `nnBatchedDecompSGDKernel` (nn_gpu.cu:2382) — port has only `SGDKernel`.
2. `verify_error_bound_kernel` (quantization_kernels.cu:106) — the port's only
   comparable code, `CompareWithinBound`, is in an example, not the pipeline.
3. `populateChunkArraysKernel` (byte_shuffle_kernels.cu:140).
4. `gather_chunk_kernel` / `scatter_chunk_kernel` (H5VLgpucompress.cu:1142/1180) —
   the Clio VOL (`context-transfer-engine/adapter/hdf5_vol/clio_vol.cc`) contains
   no CUDA kernels at all.
5. `traceQualityKernel` (H5VLgpucompress.cu:2578).
6. `histogramKernelVec4` (entropy_kernel.cu:99) — vectorized variant; likely
   perf-only, low severity, and only a finding if native takes it by default.
7. `nnFusedInferenceKernel` (nn_gpu.cu:418) — establish which of native's two
   inference kernels the default path uses before calling this a gap.

A missing kernel is NOT automatically a finding: the port may fold the same work
into a differently-named kernel (e.g. `MinMaxKernel` + `StatsPass2Kernel` may
together cover `madPass2Kernel`), or the native kernel may be dead upstream.
Reachability on BOTH sides has to be established.

## Baseline: parity suite state before this audit

`ctest -R ctp_neuropress` — all 6 parity tests PASS (2026-08-13):
`ctp_neuropress_parity`, `_dataset_parity`, `_flow_parity`, `_tiebreak_parity`,
`_device_path_parity`, `_preprocess_parity`.

This matters for judging findings: these harnesses compile native's own
`nn_gpu.cu` / `stats_kernel.cu` / `entropy_kernel.cu` and diff numbers against the
port. So a finding claiming the port produces DIFFERENT NUMBERS from native on a
covered path contradicts a passing test and is probably wrong. A finding claiming
the port computes the same numbers in the WRONG PLACE (host instead of device) is
NOT contradicted by these tests — they do not assert execution locality.

## Empirical tooling available for verification

- GPU: NVIDIA A100-PCIE-40GB.
- `nsys`, `ncu`, `nvprof`, `compute-sanitizer` all present under `/usr/local/cuda/bin`.
- `nsys profile --stats=true` on a Clio compress run yields the actual launched-kernel
  list — the decisive test for "does this really run on the GPU". A native kernel
  absent from a Clio profile, on a path the port claims is device-resident, is proof.
- Runnable Clio binaries in `build/bin`: `neuropress_gpu_direct`, `neuropress_e2e`,
  `neuropress_gpu_demo`, `neuropress_compress_dir`, plus the `*_parity_exec` harnesses.
- ctest targets exercising the live pipeline: `compressor_dynamic_neuropress`,
  `compressor_static_gpu_device_ptr`, `compressor_nvcomp_gpu_roundtrip`,
  `cte_hdf5_vol_compressor_write`.
- NeuroPress itself has no build tree yet; the parity harness sidesteps this by
  compiling native `.cu` files directly (see
  `context-transport-primitives/test/unit/compress/model/parity/CMakeLists.txt`).
