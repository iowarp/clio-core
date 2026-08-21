# Five substrates, workload resident in GPU memory, with and without checkpointing

Same melt deck, same physics, same gates, one variable changed per bench.
Measured 2026-08-21, RTX 4070 Laptop (8188 MiB), CUDA 13.3.

**Deck:** `--md --lattice 112 --steps 80` = 5,619,712 atoms, 67³ bins,
Verlet list kept. Every substrate reproduces PE/atom −6.7733681 against the
host double reference, with the statics, resort and NVE gates green.

**Checkpoint:** x and v staged device→pinned host DRAM, 8 times (every 10
steps) = **8 × 293.7 MB = 2.29 GB written**. Forces and the neighbour list
are derived and a restart rebuilds them, which is what a LAMMPS restart file
carries too. Host DRAM is the destination on purpose: the paged vector
persists into a CTE tier stack whose only configured target is a RAM tier,
so this lands the bytes in the same storage class rather than comparing two
destinations.

## Result

| substrate | axis | no checkpoint | 8 checkpoints | checkpoint cost |
|---|---|---|---|---|
| paged `gv::Vector` | capacity | **83.7 ms/step** | 83.7 (unchanged) | **0** by design |
| MPI (two-sided, host-staged) | transport | **64.7** | 68.5 (+5.8%) | 23.8 ms each, 190 ms |
| NCCL (two-sided, GPU-resident) | transport | **64.7** | 68.6 (+6.0%) | 23.8 ms each, 190 ms |
| NVSHMEM (one-sided, in-kernel) | transport | **65.8** | 69.6 (+5.7%) | 23.8 ms each, 190 ms |
| BaM (list via GPU page cache, 2 GB) | capacity | **1174.5** | 1177.3 (+0.2%) | 23.8 ms each, 190 ms |

Checkpointing costs every non-persistent substrate the same 23.8 ms per
checkpoint — 12.07 GB/s of D2H, which is PCIe, not the substrate. The paged
vector pays none of it.

## What that actually buys, priced honestly

Free persistence is worth 190 ms over 80 steps. Paging costs **19 ms per
step** against the transport substrates (83.7 vs 64.7), or **1520 ms** over
the same 80 steps. So at this checkpoint frequency the paged vector loses by
8x on the very trade it exists to win.

Break-even is **~64 checkpoints per 80 steps** — checkpointing roughly every
1.3 steps. Below that cadence, staging x+v to host DRAM is simply cheaper
than paying paging's per-step overhead all run long.

## THE CAVEAT THAT MATTERS MOST

The paged column's "0" is **not verified at this scale**. `--ckpt` on the
paged bench (FlushAllKernel into the tier stack) **crashes** at lattice 112
and 128, at the first checkpoint. So during these runs nothing flushed:
the tier holds only the INITIAL placement and the evolved state never
reached it. Paging's zero is therefore "no work done", not "work done for
free", and the honest reading is that its free-persistence claim is
UNDEMONSTRATED here, not confirmed.

Pre-existing, not from this session's gpu2cpu work: verified by reverting
`c210b54f` and reproducing identically. Succeeds at lattice 40 and 80, fails
at 112 and 128. Not tier capacity (`GV_DRAM_MB=24576` fails the same) and
not page size (192KB fails the same). The failing copy is now identified
exactly: a well-formed D2H, `dst` pinned host, `src` device, one whole page
(393216 B, or 196608 B at 192KB), landing 64 bytes into a RAM page, both
pointers valid, rejected with `cudaErrorInvalidValue`. Next suspect is the
copy crossing the end of a `cudaHostRegister`'d span in the mem bdev's
SHM-backed mapping.

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
