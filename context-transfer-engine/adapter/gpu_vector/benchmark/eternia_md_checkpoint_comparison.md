# Five substrates, workload resident in GPU memory, with and without checkpointing

Same melt deck, same physics, same gates, one variable changed per bench.
Measured 2026-08-21, RTX 4070 Laptop (8188 MiB), CUDA 13.3.

**Deck:** `--md --lattice 112 --steps 80` = 5,619,712 atoms, 67³ bins,
Verlet list kept. Every substrate reproduces PE/atom −6.7733681 against the
host double reference, with the statics, resort and NVE gates green.

**Checkpoint:** x and v staged device->pinned host DRAM, then written to a
file on NVMe and **fsync'd**, 8 times (every 10 steps) = **8 x 293.7 MB =
2.29 GB made durable**. Forces and the neighbour list are derived and a
restart rebuilds them, which is what a LAMMPS restart file carries too.

The fsync is not optional. Without it the measurement is the page cache,
not the device, and a crash still eats the checkpoint. Raw device speed for
reference: `dd oflag=direct` writes 3.6 GB/s; the benches achieve 2.24-2.39
GB/s including fsync.

## Result

| substrate | axis | no checkpoint | 8 durable checkpoints | checkpoint cost each |
|---|---|---|---|---|
| paged `gv::Vector` | capacity | **83.7 ms/step** | 83.7 (unchanged) | **0** by design -- but see the caveat |
| MPI (two-sided, host-staged) | transport | **64.7** | **80.4** (+24%) | 143.7 ms (23.8 stage + 119.9 write) |
| NCCL (two-sided, GPU-resident) | transport | **64.7** | **80.4** (+24%) | 144.4 ms |
| NVSHMEM (one-sided, in-kernel) | transport | **65.8** | **81.8** (+24%) | 146.5 ms |
| BaM (list via GPU page cache, 2 GB) | capacity | **1174.5** | **1188.9** (+1.2%) | 151.9 ms |

A durable checkpoint costs every non-persistent substrate ~144 ms: 23.8 ms
of PCIe D2H plus ~120 ms of NVMe write. **The write is 5x the staging** --
which is exactly why measuring only the D2H hop, as an earlier version of
this document did, understated the cost 6x and has to be called out rather
than quietly corrected.

## What that buys, priced honestly

Over 80 steps the transports pay ~1150 ms to make 2.29 GB durable. Paging
pays 19 ms/step for its abstraction against those same transports (83.7 vs
64.7), or 1520 ms over the same 80 steps.

So paging is **4% behind** on total wall clock (6696 ms against MPI's 6432
including its checkpoints), not the 8x it appeared to be when checkpoints
were priced as a DRAM copy. **Break-even is ~10.6 checkpoints per 80 steps,
i.e. checkpointing about every 7.5 steps.** Checkpoint more often than that
and the paged vector wins outright; less often and staging to NVMe is
cheaper than carrying paging's per-step overhead all run.

That is a real and defensible result for the paged design, and it only
becomes visible once the checkpoint is actually durable.

## Capacity ceilings, which is the other half of the story

At **lattice 128** (8,388,608 atoms, 6.2 GB of state):

| substrate | result |
|---|---|
| paged `gv::Vector` | 111.8 ms/step, fully resident, 0 faults |
| MPI | 99.5 ms/step |
| NCCL | 99.6 ms/step |
| NVSHMEM | **refuses**: 6463.9 MB of per-PE state, "no tier to spill to" |
| BaM | **OOM** in its page cache unless `--bam-cache-mb` is set explicitly |

NVSHMEM's ceiling is 1 × VRAM by construction and its own guard says so.
That is the capacity claim the paged vector exists to answer — and it does
answer it, at 1.3x the step cost of the transports.

Two operational notes. NVSHMEM refuses spuriously when run back-to-back
because the previous process has not released VRAM yet ("3226 MB free" on an
idle 7820 MB card); leave a settle gap between runs. BaM's `--bam-cache-mb`
default of auto ("fits the list") over-sizes and OOMs at both lattices; 2 GB
works and is the mode its design intends — 1174 ms/step, 18x the transports,
which is what paging the list from host DRAM costs.
