# kmeans — streaming Lloyd iterations over a point set

Variants: `clio_kmeans_{mpi,nvshmem,bam,paged}_bench` (no NCCL edition).
The MPI/NVSHMEM/BaM editions are self-contained (no clio runtime); the
paged edition hosts the runtime in-process and composes its own CTE stack.
MPI launch: `mpirun -n N clio_kmeans_mpi_bench ...` (contiguous point
shards per rank, one Allreduce of sums/counts per iteration).

## What moves what (measured, RTX 4070 Laptop 8 GB, single rank)

Numbers from the `pipelines/workload_understanding/kmeans_mpi_sweep.yaml`
sweep (`results.csv` has the full grid):

| knob | flag | effect |
|---|---|---|
| dataset size | `--data-mb` (default 256) | **VRAM axis**: peak VRAM ≈ data_mb + ~140 MB CUDA context, measured 256→394 MB, 2048→2186 MB (each rank holds its whole shard on-device). **Runtime axis too**: linear, 4 iters over 256 MB = 81.9 ms → 2048 MB = 528.4 ms (~16 GB/s effective). |
| iterations | `--iters` (default 4) | runtime linear in iters (2→40.5 ms, 16→265.4 ms at 256 MB); VRAM unchanged. |
| point shape | `--dims` (32) / `--clusters` (16) | work per point; `dims` must divide the page's float count in the paged edition. |
| ranks | `mpirun -n N` | shards the dataset: per-rank VRAM ≈ data_mb/N + context. |

No checkpoint phase → no I/O axis in the MPI edition.

## Paged-edition knobs

`--slots` (pages per block), `--page-kb`, `--hbm-mb` (kHBM tier),
`--nvme-mb`/`--nvme-path` (adds the file tier = full CTE stack),
`--repeat`, `--baseline`. Summary is one machine-parseable stderr line
(`KMEANS mode=... ms= GB/s= faults= evicts= ...`). Total page cache in
bytes is `blocks × slots × page_kb` — sweep the TOTAL, not `slots`, when
`blocks` also varies.

## Pipelines

- `../../pipelines/workload_understanding/kmeans_mpi_sweep.yaml`
- `../../pipelines/memory_pressure/kmeans_pressure.yaml`

## The centroid checksum is NOT a comparison key

`centroid_checksum` is **not reproducible run to run**, on any substrate, at
any rank count. Measured back to back on one binary:

    kokkos n=1 : 30719.999734 / 30719.999501 / 30719.999658
    mpi    n=1 : 30719.999983 / 30719.999768 / 30720.000128
    paged (cuda): 30720.001988 / 30720.000665

Centroid accumulation is atomic, so the summation order varies between runs.
Comparing this number -- across backends, across rank counts, or against a
value written down last week -- manufactures a "regression" that is not there.
That is not a defect to fix: the MPI, Kokkos and paged editions all behave
this way, so it is the deck, not any one port.

What IS exact, and what the gate actually checks: the integer assignment
COUNT, plus `faults`/`evicts`/`get_errors`/`put_errors` on the paged rows
(131072/131072/0/0 for the default deck on both CUDA and SYCL). Compare those.
