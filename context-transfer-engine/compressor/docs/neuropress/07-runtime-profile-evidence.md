# Runtime profile evidence — which kernels actually launch

Gathered by the orchestrating session (not by the scope agents), on the machine
running this audit, to hold the source-derived findings to an empirical standard.
Per project practice, a claim about where work executes is only settled by running
the code, not by reading it.

- GPU: NVIDIA A100-PCIE-40GB
- Tool: `nsys profile -t cuda` + `nsys stats --report cuda_gpu_kern_sum`
- Build: `/home/cc/clio-core/build`, `CLIO_CORE_ENABLE_CUDA=ON`
- Date: 2026-08-13

## Method

Two shipped ctest binaries were profiled directly (both embed their own runtime,
so a single process captures the whole pipeline):

```
nsys profile -t cuda -o np_static ./bin/test_compressor_functional "[gpu][static][693]"
nsys profile -t cuda -o np_dyn    ./bin/test_compressor_functional "[neuropress][693]"
```

`[gpu][static][693]` is `compressor_static_gpu_device_ptr`; `[neuropress][693]` is
`compressor_dynamic_neuropress` — the shipped test of the NeuroPress selection path.

A caveat on method: an intermediate `nsys stats` invocation in this session emitted
an empty CSV and produced spurious zero counts. The numbers below come from a
regenerated report (stale `.sqlite` deleted first, stderr captured, non-empty output
verified: 21 rows). Any "kernel absent" claim here rests on that verified run.

## Result: `compressor_dynamic_neuropress` — complete list of launched kernels

| Kernel | Instances | Origin |
|---|---|---|
| `ctp::compress::model::gpu::InferKernel` | 8 | Clio NN inference (**host-stats variant**) |
| `nvcomp::lowlevel::lz4CompressBatchKernel` | 4 | nvcomp GPU LZ4 |
| `zstd::lz_compression_kernel` | 2 | nvcomp GPU Zstd |
| `zstd::sequence_compression_kernel` | 2 | nvcomp GPU Zstd |
| `zstd::literal_compression_kernel` | 2 | nvcomp GPU Zstd |
| `zstd::setup_frame_compress`, `init_buffers`, `compact_compressed_frames` | 2 each | nvcomp GPU Zstd |
| `bitcomp::batch_encoder_kernel` | 2 | nvcomp GPU Bitcomp |
| `nvcomp::setup_comp_llif_buffers<LZ4/Zstd/Bitcomp FormatSpecHeader>` | 4/2/2 | nvcomp plumbing |
| `nvcomp::compact_comp_buffers_and_header_output` | 8 | nvcomp plumbing |
| `nvcomp::round_up_alignment_kernel` | 8 | nvcomp plumbing |
| `nvcomp::cub::…::DeviceScanKernel` / `DeviceScanInitKernel` | 8 / 8 | nvcomp plumbing |
| `nvcomp::max_reduce_device_status_kernel` | 3 | nvcomp plumbing |

**Kernels that did NOT launch at all** — every one of Clio's own statistics and
preprocessing kernels, plus the device-stats inference variant and SGD:

`StatsPass1Kernel`, `StatsPass2Kernel`, `StatsPass2DevKernel`,
`EntropyFromHistKernel`, `FinalizeFeatureStatsKernel`, `MinMaxKernel`,
`ShuffleKernel`, `UnshuffleKernel`, `QuantizeKernel`, `DequantizeKernel`,
`InferKernelDeviceStats`, `SGDKernel`.

## What this establishes

1. **Entropy, MAD and second-derivative were computed on the CPU in this run.**
   No stats kernel launched, yet the run completed a NeuroPress-ranked selection.
   The features feeding the network therefore came from the host loops in
   `data_stats.h`. This is direct empirical confirmation of the reachability
   claims in STATS-5 and ORCH-1 — the ones asserting the host branch is the
   ordinary case, not a corner case.

2. **The device-stats inference path was not taken.** `InferKernel` ran, not
   `InferKernelDeviceStats`. That is the entry point NN-7 describes as doing five
   `cudaMalloc`s, a `cudaDeviceSynchronize()` and five `cudaFree`s per chunk, and
   the one that launches on the legacy default stream (NN-10). Confirms NN-7's
   reachability claim on the shipped test.

3. **No preprocessing ran on the GPU.** Consistent with PRE-12/PRE-13: the
   quantize path is gated behind an error bound nothing sets, and the shuffle
   device branch requires a device-resident chunk this test does not supply.

4. **The codecs themselves genuinely run on the GPU.** nvcomp LZ4, Zstd and
   Bitcomp all executed as device kernels. Whatever the codec-scope agent reports,
   the "selected algorithm silently resolves to a CPU codec" hypothesis is *not*
   what happens on this path — the ranked NeuroPress selection reached real nvcomp
   GPU codecs. Three distinct algorithms in one run also shows per-chunk selection
   is live.

## Result: `compressor_static_gpu_device_ptr`

Only nvcomp LZ4 compress/decompress kernels plus nvcomp plumbing. No stats, NN or
preprocessing kernels — expected, since this test pins the algorithm and bypasses
the model entirely. Recorded so the dynamic run above has a contrast case.

## What this does NOT establish

- It does not disprove any finding about a path this test never enters. The device
  path (`ComputeDeviceStatsResident` -> `InferKernelDeviceStats`) is exercised by
  `ctp_neuropress_device_path_parity`, which passes; these two runs simply show the
  ordinary compressor test does not reach it.
- It does not measure the HDF5 VOL path at all (`cte_hdf5_vol_compressor_write` was
  not profiled here); VOL-1 through VOL-10 remain source-derived.
- It says nothing about the CPU-only build findings (STATS-4, NN-5, ORCH-2), which
  are build-gated and cannot appear in a CUDA-build profile.

## Baseline parity suite

`ctest -R ctp_neuropress` — 6/6 pass before and independent of this audit
(`ctp_neuropress_parity`, `_dataset_parity`, `_flow_parity`, `_tiebreak_parity`,
`_device_path_parity`, `_preprocess_parity`). These assert numerical agreement with
upstream's own compiled `.cu` files. They do **not** assert execution locality, so
they neither confirm nor refute any finding in this audit — a point the review pass
must keep straight.
