# gmx — PME order-4 B-spline charge spread + gather

Variants: `clio_gmx_{mpi,nvshmem,bam,paged}_bench` (no NCCL).
MPI launch: `mpirun -n N clio_gmx_mpi_bench ...` (mesh z-planes split
K/nranks; atoms are REPLICATED on every rank).

## What moves what (measured, RTX 4070 Laptop 8 GB, single rank)

From `pipelines/workload_understanding/gmx_mpi_sweep.yaml`:

| knob | flag | effect |
|---|---|---|
| mesh | `--k` (128) | **VRAM axis**: the K³ fixed-point mesh at 8 B/node — measured peak ≈ 140 MB context + K³·8 B (K=256 → 274 MB). Gather time also grows with K (more nodes per atom neighborhood contribute): K=64→256 at 200k atoms takes gather 1.6→5.7 ms. |
| atoms | `--atoms` (200000) | **runtime axis** for spread (linear: 100k/200k/400k = 0.9/1.8/3.5 ms spread at K=64); atom arrays are small (~MBs) next to the mesh. |
| ranks | `mpirun -n N` | splits mesh planes; atoms replicated, so per-rank VRAM ≈ context + K³·8B/N + atoms. |

One-shot kernels: there is NO steps knob and NO checkpoint phase.
Gates are exact (integer fixed-point): conservation, mesh checksum,
gather energy.

VRAM measurement note: the small-K runs finish in single-digit ms, which
a 50 ms `nvidia-smi` sampler can miss — trust the analytic K³·8B + 140 MB
line (it matches wherever the run is long enough to catch, e.g. K≥192).

## Paged-edition knobs

`--cap` (TOTAL cache in PAGES, 0 = resident = K+2 planes), `--page-kb`
(one XY plane per page), `--repeat`. Paging counters print as
`paging: faults= evicts= puts= get_errors= put_errors=`.

## Pipelines

- `../../pipelines/workload_understanding/gmx_mpi_sweep.yaml`
- `../../pipelines/memory_pressure/gmx_pressure.yaml`
