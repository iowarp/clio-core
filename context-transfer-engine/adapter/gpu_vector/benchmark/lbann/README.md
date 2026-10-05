# lbann — 2-layer MLP training (weights paged / model-parallel)

Variants: `clio_lbann_{mpi,nvshmem,bam,paged}_bench` (no NCCL).
MPI launch: `mpirun -n N clio_lbann_mpi_bench ...`, model-parallel: H and
O sliced by rank. **Hard requirement: `hidden % nranks == 0` and
`out % nranks == 0`** or the binary exits rc=2.

## What moves what (measured, RTX 4070 Laptop 8 GB, single rank)

From `pipelines/workload_understanding/lbann_mpi_sweep.yaml`:

| knob | flag | effect |
|---|---|---|
| hidden width | `--hidden` (4096) | weights W1 = in×H, W2 = H×out. Runtime linear in H (5 steps, batch 64: H=2048→16384 gives 21.0→133.8 ms). VRAM grows ~5.4 KB per unit H at in=256/batch 64-ish (measured H=16384 → 226 MB peak) — to reach multi-GB footprints raise `--in` and `--batch` together with H. |
| batch | `--batch` (64) | runtime ×~4 from batch 64→256 (H=4096: 34.5→135.3 ms); activation footprint scales with batch·H. |
| steps | `--steps` (5) | runtime linear; VRAM unchanged. |
| widths | `--in` (256) / `--out` (64) | W1/W2 sizes; `--in` is the cheap VRAM multiplier (W1 = in·H·4 B). |

No checkpoint phase. Gates: per-step loss bit-equality and a weight
digest vs the dense reference.

## Paged-edition knobs

`--cap` (TOTAL cache in PAGES, 0 = resident), `--page-kb`. Weights
[W1|b1|W2|b2] live in ONE paged vector; activations stay resident.
Summary: `N steps: paged X ms/step, dense Y ms/step` + paging counters.

## Pipelines

- `../../pipelines/workload_understanding/lbann_mpi_sweep.yaml`
- `../../pipelines/memory_pressure/lbann_pressure.yaml`
