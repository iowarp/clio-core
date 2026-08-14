# NeuroPress large-chunk transfer audit

Runs **2 × 256 MiB** of device-generated data through Clio's real store path
under NeuroPress **inference-only** selection with a **ratio cost model**, and
uses **Nsight Systems** to find transfers worth eliminating.

```bash
nsys profile --trace=cuda,nvtx --cuda-memory-usage=true \
     -o npaudit bin/neuropress_transfer_audit --chunks 2 --chunk-mib 256
nsys stats --report cuda_gpu_mem_size_sum npaudit.nsys-rep   # creates .sqlite
python3 analyze_nsys.py npaudit.sqlite                       # per-phase detail
```

| flag | env | default |
|---|---|---|
| `--chunks N` | `NPAUDIT_CHUNKS` | 2 |
| `--chunk-mib M` | `NPAUDIT_CHUNK_MIB` | 256 |
| `--no-readback` | `NPAUDIT_READBACK=0` | readback on |

The cost-model weights come from `CLIO_NEUROPRESS_COST_W_CT` / `_W_DT` / `_W_IO`
/ `CLIO_NEUROPRESS_COST_BW`, which the program sets to the ratio model (`0/0/1`)
unless the caller already chose values. Unset, Clio uses upstream's defaults
(all 1.0, bw 5e6) — so these knobs are for experiments only and change nothing
about a normal write.

## Why this is separate from `neuropress_data_path_trace`

The sibling tracer asks *does the payload stay on the device* and answers with
its own CUPTI record. This asks *which transfers are redundant*, which needs a
profiler's view — every copy, its stream, its duration, including copies made by
libraries this process never calls directly.

**NVTX is what makes that usable.** Each phase pushes a range, so every
`cudaMemcpy` can be attributed to the phase that caused it. Without it the
profile is 43 copies in one undifferentiated list.

## Why 256 MiB

At 4–16 MiB the per-chunk control traffic (~1 KiB) is invisible next to the
payload. Scaling the chunk 16× while control traffic stays **fixed** is what
separates transfers that scale with the data (unavoidable) from those that do
not (fixed overhead, and the only candidates for elimination).

Measured, and this is the whole point:

| | 16 MiB chunk | 256 MiB chunk |
|---|---|---|
| control bytes / chunk | ~1 KiB | **~1140 B** |
| as % of payload | 0.006% | **0.0004%** |

## Measured result, 2 × 256 MiB

Both chunks selected `nvcomp-lz4`, ratio 210.04, round trip 0 / 134,217,728
elements differ.

```
phase                     direction              bytes  count
stage-into-ipc-backend    Device-to-Device 268,435,456      2   <-- 62% of all memcpy time
compress-and-store        Host-to-Device           128      2       candidate action list
compress-and-store        Device-to-Host           128     10       order + ct/dt/ratio/psnr
compress-and-store        Device-to-Host           256      2       scores (double[32])
compress-and-store        Device-to-Host            24      2       feature stats
compress-and-store        Device-to-Host             4      2       nvcomp status
compress-and-store        Device-to-Host            64      2       nvcomp output size
compress-and-store        Host-to-Device            24      2       nvcomp descriptor
compress-and-store        Device-to-Host     1,278,043      2   <-- the compressed result
clio-setup                Host-to-Device        54,304      1       NN weights, ONE TIME
clio-setup                Host-to-Device            32      4       x/y means and stds
readback-to-device        Host-to-Device     1,278,019      2       blob back down to decompress
```

**The 256 MiB payload never crossed to the host.** Total D2H is 2.558 MB, of
which 2.556 MB is the compressed output — leaving ~2 KB of control traffic for
half a gigabyte of data.

## What is actually worth eliminating

**1. The staging D2D — 536.87 MB, 815 µs, 62% of all memcpy time.** By two
orders of magnitude the largest transfer in the trace. It exists because this
program generates into its own buffer and then copies into the registered IPC
backend. A producer that wrote *directly into* the registered backend would not
make it. This is the finding; everything below is rounding error beside it.

**It is removable, and `--no-staging` removes it.** Registration happens first,
then the generator writes straight into the returned pointer. Measured:

| | staging (default) | `--no-staging` |
|---|---|---|
| memcpy count | 43 | **41** |
| bytes moved | 516.930 MiB | **4.930 MiB** |
| memcpy GPU time | 1.289 ms | **0.359 ms** |
| device memory | +512 MiB for `src` | **none** |
| codec / ratio | nvcomp-lz4 / 210.040 | nvcomp-lz4 / 210.040 |
| compressed size | 1,278,043 B | 1,278,043 B |
| round trip | 0 / 134,217,728 differ | 0 / 134,217,728 differ |

**99.05% of all bytes moved, gone, with byte-identical output.**

Why it is safe: `AllocateAndRegisterGpuBackend` is a plain `cudaMalloc` in the
calling process plus a registration record (`ipc_manager.cc:3840-3849`), so the
pointer it returns is ordinary device memory this process can launch kernels
into — there is no runtime-owned initialization to trample. And the compressor
treats its input as read-only: quantize and shuffle each allocate their own
destination and read `input_ptr` (`compressor_runtime.cc:2440-2447`).

Two honest caveats:

- With staging off the round-trip check compares the readback against *the same
  buffer the compressor read*, so it could not detect in-place modification of
  the input. The evidence that no such modification happens is the source above
  plus the staged run — which compares against an independent private buffer and
  yields the identical compressed size. A mutated input would change that size.
- This run had the runtime **in-process** (`CLIO_INIT(..., /*with_runtime=*/true)`),
  so registration took the `registered_in_process` short-circuit. The
  cross-process path registers through `RegisterMemoryTask` and the runtime
  imports with `cudaIpcOpenMemHandle`; writing after registration should still
  be visible since both mappings alias the same physical pages, but **that path
  is not exercised here**. Test it before relying on no-staging with an external
  daemon.

**2. Four 128 B prediction arrays per chunk (512 B).** `out_comp_time_ms`,
`out_decomp_time_ms`, `out_ratio`, `out_psnr_db` are fetched for all 32
candidates, but only the winner's four values are ever read — the exploration
path reads `compress_lib_`/`compress_preset_` and re-measures the rest. Native
returns the winner's scalars in one 36 B struct instead.

**3. The 256 B scores array** is `double[32]`; native's `predicted_costs` is
`float[32]` (128 B). Same information, twice the width.

**4. The 24 B feature-stats readback** is consumed only by
`LogNeuroPressSelection`, which returns immediately when
`CLIO_NEUROPRESS_SELECTION_LOG` is unset. The copy is not gated, the consumer
is.

Together 2–4 are **664 B of the 1140 B per-chunk control traffic**, and 12 of
the 36 sub-KiB copies. Those copies are latency-bound at ~2.7 µs each
regardless of size, so eliminating them saves roughly **32 µs** — against a
120 ms compress phase, 0.03%. Worth doing for tidiness, not for speed.

## The thing the transfer numbers do not show

First chunk compresses in **103.8 ms**, second in **2.83 ms** — a ~100 ms
one-time cost inside `compress-and-store` (nvcomp manager construction, first
`cudaMalloc` of the 16.8 MB scratch, codec autotuning). That dwarfs every
transfer finding here and is invisible in a memcpy summary. Any benchmark
reporting a single-chunk compress time is measuring that warmup.

## Reading the per-phase output

NVTX ranges are stamped in **CPU** time; memcpy start/end are **GPU** time. An
async copy enqueued at the end of a phase can execute after that range closed
and land in the next one — the two staging copies show up one per phase for
exactly this reason. Totals are exact; boundary attribution is approximate.
Check against the global count before concluding a phase made a copy it did not.
