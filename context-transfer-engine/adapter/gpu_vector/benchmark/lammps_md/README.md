# lammps_md — LJ melt MD (the checkpointing workload)

Variants: `clio_lammps_md_{mpi,nccl,nvshmem,bam,paged}_bench` — the only
workload with all five substrates AND a real checkpoint/IO phase.
MPI launch: `mpirun -n N clio_lammps_md_mpi_bench ...` (z-plane slab
decomposition of the bin grid; needs `nb/npes >= 3`). Always pass `--md`
for the physics deck — without it the bench runs the ballistic gate only.

## What moves what (measured, RTX 4070 Laptop 8 GB, single rank, --md)

From `pipelines/workload_understanding/lammps_md_mpi_sweep.yaml`:

| knob | flag | effect |
|---|---|---|
| lattice | `--lattice` (20) | natoms = 4·lattice³. **VRAM axis**: measured peak 158/218/322 MB at lattice 20/30/40 (x+v+f + resort + neighbor list; the bench prints its analytic per-rank breakdown, and the neighbor list `--maxneigh` dominates at scale). **Runtime axis**: 100 steps = 96/177/429 ms. |
| steps | `--steps` (100) | runtime linear (lattice 40: 50→214 ms, 100→429 ms). |
| checkpoint period | `--ckpt` (0 = never) | **the I/O axis**. Every N steps: stage D2H + durable write (with `--ckpt-dir`). Measured at lattice 40/100 steps: ckpt every 5 = 20 checkpoints × (1.0 ms stage + durable) = ~6% on top of the run; checkpoint size = 11.9 MB at lattice 40 (2 arrays × natoms × 12 B... see the `checkpoints:` summary line). At small sizes the durable write is absorbed by the page cache (prints 0.0 ms; the stage D2H is the honest floor). |
| durable dir | `--ckpt-dir` (empty) | empty = DRAM staging only (`[DRAM ONLY -- NOT durable]` tag); set a path for real O_DSYNC file writes. |
| resort period | `--rebin` (20) | resort + list rebuild cadence; shows in the `phases (total ms)` breakdown. |

## Paged-edition checkpointing

The paged edition checkpoints with **CTE fault handlers**: `--ckpt N`
takes a `vector.Copy` snapshot of x and v (lazy copy tags, no bytes
moved at snapshot time) on top of the write-site flushes, and the final
readback verifies the SNAPSHOT tag through the restore path. Cache
budget: `--vram-mb` (MB across all vectors; 0 = size for residency),
`--page-kb`, `--slots`. Real VRAM is verified with `cudaMemGetInfo`
deltas (`REAL free-VRAM delta` lines).

## Pipelines

- `../../pipelines/workload_understanding/lammps_md_mpi_sweep.yaml`
- `../../pipelines/memory_pressure/lammps_md_pressure.yaml`
- `../../pipelines/register_eval/register_eval.yaml` (all five substrates)
