# Clio-NeuroPress data path trace

Follows a GPU-generated chunk through Clio's **real** store path and records
every byte that moves — in order, with direction, size, thread, and which named
memory region each end of the copy falls in.

It answers one question:

> Data is generated in GPU memory and handed to Clio. Does it **stay** on the
> device until it has been compressed, and only then come back to the host?

```bash
bin/neuropress_data_path_trace [--chunks N] [--chunk-mib M] [--out DIR]
                               [--incompressible] [--no-readback]
```

| flag | env | default |
|---|---|---|
| `--chunks N` | `NPPATH_CHUNKS` | 2 |
| `--chunk-mib M` | `NPPATH_CHUNK_MIB` | 16 |
| `--out DIR` | `NPPATH_OUT_DIR` | `.` |
| `--incompressible` | `NPPATH_REGIME=1` | compressible |
| `--no-readback` | `NPPATH_READBACK=0` | readback on |

Outputs `data_path_report.txt` (timeline + verdict) and
`data_path_timeline.json`.

## How it differs from the sibling harness

`neuropress_gpu_chunk_equivalence` compares **native NeuroPress against Clio**,
callback by callback. This traces **Clio alone**, chronologically, and needs no
native NeuroPress build — only CUPTI.

## Why the numbers are trustworthy

- **CUPTI runtime-API callbacks.** Every `cudaMemcpy` and kernel launch is
  observed as the runtime makes it. Nothing is inferred from source reading.
- **The runtime is in-process** (`CLIO_INIT` with `default_with_runtime = true`),
  which is what makes the compressor's own worker threads visible. Without it
  the interesting half of the path would run in another process and none of it
  would appear. Thread ids in the timeline show the handoff.
- **Region tagging.** The driver registers its buffers by name, so a line reads
  `D2D 16.00 MiB [chunk0.src -> chunk0.ipc]` rather than "a copy happened".
  This is what lets the report say *the chunk* left the device, rather than
  *something* did.
- **Harness copies are tagged** and excluded from every production total.

## Measured result, 2 x 16 MiB

Compressible regime, both chunks selected `nvcomp-lz4`, ratio 209.875:

```
PAYLOAD-SIZED PRODUCTION TRANSFERS
    seq 27  D2D  16.00 MiB  stage-into-ipc-backend  [chunk0.src -> chunk0.ipc]
    seq 28  D2D  16.00 MiB  stage-into-ipc-backend  [chunk1.src -> chunk1.ipc]

TOTALS   H2D: 11 copies, 209.6 KiB
         D2H: 28 copies, 158.4 KiB
         D2D:  2 copies,  32.00 MiB

RESIDENCY VERDICT
    Payload-sized D->H : 0
```

The compress phase reads, in order: `StatsPass1Kernel`, `EntropyFromHistKernel`,
`StatsPass2DevKernel`, `FinalizeFeatureStatsKernel`, `InferKernelDeviceStats`,
`RankKernel`, then nvcomp's `lz4CompressBatchKernel` and friends — with only
128/256-byte metadata copies between them. The payload's only trip to the host
is at the very end:

```
    seq  99   D2H  78.1 KiB   <- chunk 0's compressed result
    seq 100   D2H  78.1 KiB   <- chunk 1's compressed result
```

78.1 KiB is exactly 16 MiB / 209.875. **The 16 MiB never crossed the bus; only
the compressed output did.**

## The other path, which is worth knowing about

With `--incompressible` the codec output comes out larger than the input, the
runtime discards it and stores the original bytes
(`compressor_runtime.cc:2685`), and the trace shows the difference plainly:

```
    seq 97   D2H  16.00 MiB  compress-and-store  [chunk0.ipc -> HOST]
    seq 98   D2H  16.00 MiB  compress-and-store  [chunk1.ipc -> HOST]
```

So the residency property is a property of **compression succeeding**, not of
the device path in general. Both regimes are worth tracing, which is why the
choice is a flag rather than a fixed dataset.
