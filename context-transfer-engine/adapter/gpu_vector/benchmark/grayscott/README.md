# grayscott — 3D reaction-diffusion stencil (z-slab decomposition)

Variants: `clio_grayscott_{mpi,nvshmem,bam,paged}_bench` (no NCCL).
MPI launch: `mpirun -n N clio_grayscott_mpi_bench ...` (z-slabs with 2
ghost planes per rank, halo exchange per step).

## What moves what (measured, RTX 4070 Laptop 8 GB, single rank)

From `pipelines/workload_understanding/grayscott_mpi_sweep.yaml`:

| knob | flag | effect |
|---|---|---|
| grid size | `--data-mb` (2048) | **VRAM axis**: peak ≈ data_mb + ~150 MB (measured 512→658 MB, 4096→4240 MB). Runtime linear: 16 steps at 512 MB = 45.4 ms → 4096 MB = 298.9 ms. The grid is DERIVED: plane = page_kb·1024/4 cells factored into nx×ny, nz = data_mb·1 MB/4B/(4·plane) — data_mb counts ALL FOUR device fields (u, v, u', v'). |
| steps | `--steps` (4) | runtime linear (4/16/64 steps = 11.3/45.4/181 ms at 512 MB); VRAM unchanged. |
| page size | `--page-kb` (1024) | shapes nx×ny (one plane = one page in the paged edition); keep it identical across variants for cell-for-cell comparability. |

No checkpoint phase. Gate: `v_checksum` vs a host reference
(`CSUM GATE`), tolerance `--check-tol`.

## Paged-edition knobs

`--slots` (pages per block), `--hbm-mb`, `--nvme-mb`/`--nvme-path` (file
tier = full stack), `--repeat`, `--baseline`. Machine line on stderr:
`GRAYSCOTT mode=... ms= GB/s= v_checksum= faults= evicts= ...` plus a
`TIER SPLIT: kHBM used=... | host used=...` line showing where the pages
actually landed.

## Pipelines

- `../../pipelines/workload_understanding/grayscott_mpi_sweep.yaml`
- `../../pipelines/memory_pressure/grayscott_pressure.yaml`
