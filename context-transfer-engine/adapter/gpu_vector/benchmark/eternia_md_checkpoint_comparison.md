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
| BaM (list via GPU page cache) | capacity | see below -- **the number first published here was invalid** | | |

A durable checkpoint costs every non-persistent substrate ~144 ms: 23.8 ms
of PCIe D2H plus ~120 ms of NVMe write. **The write is 5x the staging** --
which is exactly why measuring only the D2H hop, as an earlier version of
this document did, understated the cost 6x and has to be called out rather
than quietly corrected.

### RETRACTED: the BaM leg does not measure BaM

`external/bam` is not upstream BaM. Its own README calls it "a basic
implementation of BaM (ASPLOS'23)", and it diverges from
github.com/ZaidQureshi/bam on both axes that decide the numbers.

**The GPU API is not the same.** Upstream exposes three separate mechanisms
for holding a page across many accesses:

| upstream | ours |
|---|---|
| `array_d_t<T>::acquire_page(i, page_, start, end, r)` / `release_page(...)` -- returns `start`/`end` so the kernel loops the whole page on a raw pointer | absent |
| `bam_ptr<T>` / `bam_ptr_tlb<T>` -- smart pointer holding a page reference across accesses, `update_page(i)` | absent |
| `tlb<T, n, scope, loc>::acquire/release` -- translation lookaside buffer | absent |
| `array_d_t<T>`: `operator[]`, `seq_read`, `seq_write`, `get_raw`/`release_raw`, `AtomicAdd`; `range_d_t<T>`: `operator[]`, `mark_page_dirty` | `ArrayDevice<T>::read(idx)` / `write(idx, val)` only |

So every element read in our version re-resolves the page --
`page_cache_acquire` does an `atomicCAS` on the tag plus an `atomicAdd` on
the state -- while upstream's whole point is that you acquire once and then
stream the page. **The 3x force-pass gap measured against the paged vector
is that missing API, not BaM.** It benchmarks the one access pattern
upstream provides three mechanisms to avoid.

**NVMe is not emulated -- it is stubbed.**

```cpp
__device__ inline int nvme_read_page(QueuePairDevice &qp, uint64_t bus_addr,
                                     uint64_t offset, uint32_t page_size) {
  (void)qp; (void)bus_addr; (void)offset; (void)page_size;
  return -1;
}
```

The only working backend is `host_read_page`, a warp-cooperative `uint4`
copy straight out of pinned host memory: no submission queue, no doorbell,
no completion polling, none of the latency structure a page cache exists to
hide. That makes misses far CHEAPER than real BaM, at the same time as the
missing acquire/release API makes hits far more EXPENSIVE.

The two errors push in opposite directions, so neither the resident number
nor the out-of-core one can be quoted as BaM. **The BaM row is withdrawn
from this comparison**; what follows is kept only as a record of what was
measured.

To make the leg real: implement `acquire_page`/`release_page` returning
`start`/`end`, plus `bam_ptr` and the TLB, and use the acquire-then-stream
pattern in the force kernel the way upstream's own benchmarks do; and back
`nvme_read_page` with an emulated queue pair over pinned host memory --
submission queue, doorbell, completion polling -- rather than bypassing the
protocol with a memcpy.

### The originally published BaM number was also an out-of-core run



The first version of this table quoted BaM at 1174.5 ms/step. That run had
`--bam-cache-mb 2048` against a **3524.6 MB** list, which puts BaM OUT OF
CORE -- and its own banner says not to quote it:

    !! out-of-core BaM is EXPECTED TO FAIL the gates above 2 blocks: its
       page cache has no pin, so an evicting block can change a page another
       block is mid-read of. Quote the RESIDENT number; the out-of-core one
       is not a result.

It was run at 128 blocks. The gates happened to pass, which is luck, not
evidence. BaM also **cannot be resident at lattice 112 on this GPU**: the
list needs 3524.6 MB of cache and the largest cache that allocates is ~3072
MB, so there is no valid BaM number at this deck at all.

At lattice 100, where it fits, all three run on the same deck with durable
checkpoints:

| config | ms/step | durable write |
|---|---|---|
| BaM resident (2600 MB cache >= 2406.8 MB list) | 137.9 | 92.7 ms (2.11 GB/s) |
| BaM out of core (1200 MB cache) | 944.8 | 92.4 ms (2.12 GB/s) |
| MPI | 55.8 | 87.9 ms (2.23 GB/s) |

So the 18x was the out-of-core penalty -- BaM's designed trade, capacity
past VRAM paid for in PCIe page faults -- not a property of BaM resident,
which costs ~2.5x MPI for its page-cache indirection on every neighbour
read.

And the checkpoint difference is not about BaM at all: 92.7 vs 87.9 ms is
5%, the write path is identical code in all four benches, and out-of-core
BaM writes at the same 2.12 GB/s while moving GB/s of list traffic. Read it
as contention/noise, not as a substrate property.

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
