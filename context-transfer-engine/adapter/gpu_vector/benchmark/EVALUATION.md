# Evaluation on Aurora: enacting the plan

The plan is `external/eternia/eternia/sections/evaluation-plan.tex` (in the
main checkout). This file records what has actually run, in the order the
plan prescribes: 4-node baselines for every workload on every Aurora
substrate first, then the studies. Every number here is from a job log
under `build-spike/pbs/`.

Rules in force while the system is being brought up: 4 nodes, every job
under 5 minutes; the non-Eternia (baseline) workloads first, Eternia last;
then the decks grow towards 15-minute runs, never past.

## Substrates on Aurora

| substrate | binary | data plane |
|---|---|---|
| MPI | `clio_<wl>_mpi_bench` | MPICH (built with Level Zero), GPU-aware; device buffers straight to MPI |
| oneCCL | `clio_<wl>_ccl_bench` | the NCCL analogue: collectives and send/recv on device buffers |
| Intel SHMEM | `clio_<wl>_ishmem_bench` | the NVSHMEM analogue: symmetric heap, one-sided puts, fcollect, reductions |
| Eternia | `clio_<wl>_paged_newcoro_aot` | the system under test (clio runtime, coroutine editions) |

The three baseline editions share one SYCL source per workload
(`sycl_baseline/`), built by `build_baselines_aurora.sh`, run by
`pbs_baseline_aurora.sh` (one debug-scaling allocation carries all three,
one rank per node, one tile per rank).

## Stage 1: 4-node baselines, small decks (bring-up)

Small decks first, to prove every (workload, substrate) cell runs and
gates on 4 nodes. Times are the edition's own timed region; "comm" is the
time spent in the substrate's collectives and exchanges.

| workload | deck | MPI | oneCCL | Intel SHMEM |
|---|---|---|---|---|
| kmeans | 4 GB global (1 GB/rank), 4 iters | 2465 ms (comm 1026) PASS | 2540 ms (comm 1062) PASS | 2045 ms (comm 622) PASS |
| grayscott | 4 GB global, 4 steps, 1 MB page | pending | pending | pending |
| gmx | K=512, 4 M atoms | pending | pending | pending |
| lbann | 1024 -> 4096 -> 256, batch 64, 5 steps | pending | pending | pending |
| lammps_md | pending | | | |

Findings on the way:

- Intel SHMEM's default symmetric heap refused a 1 GB allocation; the job
  script sets `ISHMEM_SYMMETRIC_SIZE` to 40 GB, and the editions keep bulk
  data no peer names in plain USM (`AllocLocal`).
- The debug queue caps at two nodes; 4-node jobs go to `debug-scaling`.
