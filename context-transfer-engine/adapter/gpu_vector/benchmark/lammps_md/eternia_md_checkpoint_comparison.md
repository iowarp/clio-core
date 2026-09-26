# Five substrates, workload resident in GPU memory, with and without checkpointing

Same melt deck, same physics, same gates, one variable changed per bench.
Measured 2026-08-21, RTX 4070 Laptop (8188 MiB), CUDA 13.3.

## Read this first: every earlier number in this file was mis-measured

The four transport/capacity benches default to `--blocks 128 --threads 64`.
The paged bench defaults to `--blocks 64 --threads 256`. Earlier revisions of
this document ran each bench **at its own default**, so the paged vector was
measured at its favourable geometry and everything else at an unfavourable
one. Geometry is worth up to 1.7x on this kernel, so that comparison was not
like-for-like and its headline conclusion (paging "4% behind", break-even at
~10.6 checkpoints) was wrong.

**Everything below is run at a single matched geometry: `--blocks 64
--threads 256` for all five substrates.** Any future row added here must
state its geometry or it is not a result.

## Result: lattice 100, 80 steps, 64 x 256

4,000,000 atoms. The only deck where all five substrates are resident.

| substrate | axis | no checkpoint | 8 durable checkpoints | durable write each |
|---|---|---|---|---|
| MPI (two-sided, host-staged) | transport | **23.918 ms/step** | **35.228** | 87.4 ms (2.23 GB/s) |
| NCCL (two-sided, GPU-resident) | transport | **24.012** | **35.164** | 86.6 ms |
| NVSHMEM (one-sided, in-kernel) | transport | **25.069** | **35.603** | 84.4 ms + 16.3 stage |
| paged `gv::Vector` | capacity | **58.724** | 0 by design -- **but unverified, see below** | -- |
| BaM (list via GPU page cache) | capacity | **135.451** | **146.992** | 90.6 ms |

At matched geometry the paged vector costs **2.46x MPI** on this deck. The
earlier 1.29x was the geometry artifact. This is the honest standing of the
abstraction against a transport baseline when the problem fits in VRAM.

## Result: lattice 112, 80 steps, 64 x 256

5,619,712 atoms. BaM cannot be resident here (the list needs 3524.6 MB of
cache; the largest that allocates on this GPU is ~3072 MB), and NVSHMEM
refuses unless the GPU is fully idle -- see the operational notes.

| substrate | no checkpoint | 8 durable checkpoints |
|---|---|---|
| MPI | **36.187 ms/step** | **51.878** (120.4 ms write each) |
| NCCL | **36.167** | **51.983** (119.9 ms) |
| paged `gv::Vector` | **83.278** | crash -- see below |
| BaM, 2 GB cache | 2453.459 **(out of core, not a resident result)** | 2475.230 |

Paged is **2.30x MPI** here, consistent with the 2.46x at lattice 100.

## The paged vector's "zero checkpoint cost" is a design claim, not a measurement

The premise is that the vector is persistent, so no checkpoint is needed
beyond the initial placement. That only holds if the contents are genuinely
durable in the CTE tier stack rather than sitting dirty in the VRAM page
cache -- where a crash loses them exactly like an unsaved transport run.

The one path that forces that durability, `--ckpt` on the paged bench,
**crashes**, at lattice 100 and 112 alike:

    [memcpyasync-fail] dst=... (rc=0 type=1) src=... (rc=0 type=2) size=393216
    FATAL MemcpyAsync CUDA Error 0: no error

So the "0" column is what the design intends, not something these runs
demonstrate. Until that flush path works, the paged vector's durability
story is unverified and the comparison below should be read as favourable to
paging by assumption.

## Break-even, recomputed

Over 80 steps at lattice 100, a durable checkpoint costs the transports
~113 ms all-in (16.3 ms PCIe stage + ~87 ms NVMe write + fsync). Paging
pays 34.8 ms/step against MPI, or 2784 ms over the run.

**Break-even is ~24.6 checkpoints per 80 steps -- checkpointing every ~3.3
steps.** Lattice 112 gives the same answer (~24.0). That is far more
aggressive than any realistic MD checkpoint cadence, so on this workload
**staging to NVMe is cheaper than carrying paging's per-step overhead**,
and the earlier "every 7.5 steps, paging wins" claim is withdrawn.

The fsync is not optional. Without it the measurement is the page cache, not
the device. Raw device speed for reference: `dd oflag=direct` writes 3.6
GB/s; the benches achieve 2.23-2.39 GB/s including fsync.

## Where the paged abstraction does win: run length

BaM and the paged vector are the two capacity substrates, and they diverge
with horizon. Same deck, same 64 x 256, lattice 100:

| steps | BaM resident (2600 MB) | paged `gv::Vector` |
|---|---|---|
| 20 | **35.500 ms/step** | 45.956 |
| 80 | 135.517 | **58.862** |

At a short horizon BaM is 23% faster; by 80 steps it is 2.3x slower. BaM's
direct-mapped page cache degrades as atoms drift and the neighbour-list
access pattern loses the locality the cache was mapped for; the paged
vector's holds do not. **This -- not per-access translation cost -- is the
abstraction difference that favours ours**, and it only appears if the
measurement runs long enough. A 20-step benchmark reports the opposite
conclusion.

Note that the earlier claim of a 3x resident force-pass advantage over BaM
was the geometry artifact, and the claim that it came from BaM's missing
acquire-then-stream API was also wrong: routing the force loop through
`bam_ptr`, which amortizes exactly that, moved nothing (126.08 vs 125.87).

## Capacity ceilings, which is the other half of the story

At **lattice 128** (8,388,608 atoms, 6.2 GB of state), at each bench's
default geometry -- these rows predate the matched-geometry rule and are
kept only for the pass/refuse outcome, not the timings:

| substrate | result |
|---|---|
| paged `gv::Vector` | runs, fully resident, 0 faults |
| MPI / NCCL | run |
| NVSHMEM | **refuses**: 6463.9 MB of per-PE state, "no tier to spill to" |
| BaM | **OOM** in its page cache unless `--bam-cache-mb` is set explicitly |

NVSHMEM's ceiling is 1 x VRAM by construction and its own guard says so.
That is the capacity claim the paged vector exists to answer, and it does
answer it -- at 2.3-2.5x the step cost of the transports.

## Operational notes

**NVSHMEM needs a genuinely idle GPU.** It refused at lattice 100 with
"3017.4 MB of per-PE state against 3226.0 MB free VRAM" even after a 30 s
settle, while `nvidia-smi` showed 7820 MB free and no compute apps. Run it
standalone; it then completes at 25.069 ms/step. A refusal in a back-to-back
sweep is not a capacity result.

**BaM's `--bam-cache-mb` default (auto, "fits the list") OOMs.** Set it
explicitly. Below the list size it goes out of core, and the bench's own
banner forbids quoting that as a BaM result.

**NVMe is emulated as an asynchronous bulk copy from pinned host memory**
(warp-cooperative `uint4`, 32 lanes in flight), not as a queue pair. The
medium is DRAM, so miss cost is understated relative to real flash; that
belongs next to any miss-heavy number. Two findings from the abandoned
queue-pair emulation, each of which cost a session: a doorbell in mapped
host memory needs device atomics on host memory, which this GPU cannot do
(`cudaDevAttrHostNativeAtomicSupported = 0`); and a host controller's plain
`cudaMemcpy` serializes on the legacy default stream against the very kernel
awaiting its completion.
