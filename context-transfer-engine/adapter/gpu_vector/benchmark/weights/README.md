# weights — repeated full-array re-reads (inference-weight streaming)

Variants: `clio_weights_{mpi,nvshmem,bam,paged}_bench` (no NCCL edition).
MPI launch: `mpirun -n N clio_weights_mpi_bench ...` (flat element range
split per rank, one Allreduce per pass).

## What moves what (measured, RTX 4070 Laptop 8 GB, single rank)

From `pipelines/workload_understanding/weights_mpi_sweep.yaml`:

| knob | flag | effect |
|---|---|---|
| dataset size | `--data-mb` (512) | **VRAM axis**: peak ≈ data_mb + ~140 MB (measured 256→394 MB, 2048→2186 MB). Runtime linear: 3 passes over 256 MB = 4.3 ms → 2048 MB = 33.4 ms (~190 GB/s — this workload is pure bandwidth). |
| passes | `--passes` (3) | runtime linear (1/3/9 passes = 1.5/4.3/12.9 ms at 256 MB); VRAM unchanged. |
| compressibility | `--flat-pct` (0) | fraction of flat (compressible) pages — only meaningful to the paged edition's codec path. |

No checkpoint phase. CAUTION: rank 0 verifies the sum on the HOST
serially, so wall clock grows with `--data-mb` far faster than GPU time —
use the printed GPU ms, never the process runtime.

## Paged-edition knobs

`--pages` (pages per block walked), `--slots`, `--page-kb`, `--hbm-mb`,
`--nvme-mb`, `--compressed`/`--cpu-codec` (codec path), `--baseline`.
Machine line on stderr: `GVW mode=... ms= GB/s= checksum= faults= ...`.

## Pipelines

- `../../pipelines/workload_understanding/weights_mpi_sweep.yaml`
- `../../pipelines/memory_pressure/weights_pressure.yaml`
