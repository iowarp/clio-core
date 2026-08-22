# gpu_vector: the design, in intentions

What this abstraction is trying to be, why each path exists, and which
properties everything else depends on. No code. Where a detail is inferred
rather than verified it is marked **(unverified)**.

---

## 1. The problem

A GPU kernel can only touch what fits in VRAM. The usual answers are to shrink
the problem, to split it across more GPUs, or to stage data in and out between
kernel launches — each of which pushes the capacity question back onto the
application.

`gpu_vector` answers it differently: **give the kernel an array larger than
VRAM and let it fault.** The kernel indexes what looks like a flat vector; the
bytes it touches are brought in on demand from the CTE tier stack (VRAM, host
DRAM, NVMe) and evicted again when the space is needed. The application writes
one loop, not a staging schedule.

The cost of that promise is everything below: a software page cache that lives
inside the kernel, and a protocol for a kernel to stop mid-computation, wait
for the runtime, and resume.

---

## 2. The central mental model: a private cache per block

**Every CUDA block has its own page table.** Not one shared cache with locks
around it — `nblocks_` independent tables of `pages_per_block_` slots each.

This is the single most important design decision, and most other properties
follow from it:

- A block resolving, claiming, or evicting a page touches only its own table,
  so the lock it takes is a **per-block** lock. Blocks never contend with each
  other, and there is no global structure to serialize on.
- The same page may be resident in several blocks at once, each with its own
  copy. Reads are cheap to replicate; **writes are not shared**, which is why
  the write story is "each block owns a disjoint region" rather than
  "coherence".
- A block's working set is bounded by its own table, so one block cannot
  starve another by faulting heavily.

The rule that falls out and must not be broken: **per-block locking is fine,
cross-block locking is not.** Any design that needs one block to wait on
another's table is wrong for this structure.

---

## 3. Anatomy

**Page** — the unit of transfer and of residency. A fixed span of elements.
Chosen so a page divides the application's natural row/tile, because a request
that straddles two pages costs two holds instead of one.

**Slot** — a frame in a block's table that may hold a page. Carries: which
page it currently holds, a pointer to its bytes, a claim generation, and the
state bits that make concurrent access safe (below).

**Table** — one block's array of slots. Sized by the cache budget.

**Placement** — a page does not go in an arbitrary slot. Way 0 is the direct
map (`page mod slots`), which makes a contiguous sweep collision-free and lets
a lookup succeed on the first probe; further ways are hashed segments, so a
pathological stride degrades to a short chain rather than a linear scan. The
intent is that the common case — sequential or stencil access — is a single
probe.

**Slot state** — every bit exists to answer "may I use this slot right now?":
- *dirty* — written since it was faulted or last flushed, so it must be
  written back before the slot is reused.
- *fetching* — bytes are on the way and the slot is **not yet readable**. It
  distinguishes a per-page fetch, a page riding an async batch, and
  compressed bytes that have landed but not been decoded.
- *flushing* — a writeback is outstanding; a batch writeback is marked
  distinctly because its completion belongs to the batch, not the page.
- *evicting* — the slot is being taken over.
- *pins* — how many readers currently hold a raw pointer into this page. A
  pinned page cannot be evicted.
- *generation* — bumped on every claim, so a reader that captured it can prove
  the slot was not recycled underneath it.

**Scores** — eviction ranks by a score with a deliberate split: an *access*
score maintained by the cache (bumped on touch, aged on claim scans, reset on
refill) and a separate *user* score the kernel can set, with a flag
distinguishing "the user asked for zero" from "the user never said". Ties
break on last-access time. The intent is that the cache's own heuristic never
silently overwrites an application hint.

---

## 4. Two regimes, deliberately different

**Resident.** The whole vector fits, is placed once up front, and nothing ever
evicts. In this regime the design switches costs off rather than paying them
cheaply: no eviction bookkeeping, no pinning (nothing can be taken away), no
recency stamping. The contract is *zero faults*, and it is enforced as a test
gate rather than hoped for — because a fault here means the placement was
wrong, and silently limping is worse than failing.

**Out of core.** The vector exceeds the cache. Everything above is live:
faults, eviction, writeback, pinning, admission.

The regimes must produce **identical results**. That equivalence is the main
correctness gate on the whole design: same physics, same digits, resident and
out of core.

---

## 5. The access path

The kernel's only way to reach data is to take a **hold** — a guard that
returns a raw pointer and keeps the page in place for as long as the guard
lives. The guard *is* the pin; there is no separate lock/unlock to forget.

Intentions:

- **Write intent is declared at the hold, not inferred per element.** A hold
  taken for writing marks the page dirty up front. Element access is then
  side-effect-free, which means a read through a guard can never accidentally
  dirty a page and force a spurious writeback.
- **A hold reports how far you can walk.** The caller gets a pointer and a run
  length, so a loop over a page's worth of elements does one resolution and
  then plain pointer arithmetic. Resolving per element was measured as
  catastrophic; the API is shaped to make that impossible to write by accident.
- **Holds are block-collective.** Every thread in the block takes the same
  hold and the block takes one path. This is not an optimization — a hold can
  suspend, and a suspend that splits a block across a barrier is a hang or a
  hardware trap.

### 5.1 The fast path

Most holds are hits. The intent is that a hit costs a probe and nothing else:

- **No lock is taken.** A resident page needs no exclusive access; only
  claiming or evicting does. Taking the block lock on every hit serialized the
  whole block through one atomic and was slower than faulting.
- **The probe is lock-free and safe against a concurrent fault**, because a
  slot publishes "fetching" before it publishes its page number. A scanner that
  sees the new page number necessarily also sees that it is not yet readable,
  and falls through to the locked path.
- **The outcome is voted across the block, not decided per thread.** On a
  shared table another block's eviction can flip a slot between two threads'
  probes; one thread would succeed where another missed, and the block would
  split. The block takes the fast path only if *every* thread hit; otherwise
  every thread falls to the slow path together, and any thread that did
  succeed gives its pin back first.

### 5.2 The slow path — the fault

A miss cannot be serviced inside the kernel, and this is the load-bearing
constraint of the whole design: **the servicer cannot run while the faulting
kernel occupies the GPU.** A kernel that spins waiting for a fill blocks the
work that would fill it.

So a fault is not a wait. It is:

1. **Retire what has landed.** Completed fetches and writebacks are reaped
   first, so the slot search sees the true state.
2. **Make room.** If a victim must be displaced and it is dirty, its writeback
   is *submitted* and the fault yields — the writeback must finish before its
   slot can be reused, but that waiting happens outside the kernel.
3. **Ask for the page.** A fetch is submitted for the missing page, batched
   with its neighbours when the access pattern allows, so one request covers
   many pages rather than one round trip each.
4. **Park.** The block records what it is waiting for and the kernel
   **returns**. The runtime services the request and relaunches; the block
   resumes where it left off.
5. **Re-check and retry.** On resume the condition is re-evaluated, not
   assumed. A claim that failed because every candidate slot was busy is
   re-issued rather than waited on, which turns a transient failure into one
   extra round instead of a livelock.

**Park, not spin, is the point of the coroutine.** The kernel must be able to
exit mid-computation and resume with its live state intact; that state lives
in the coroutine frame. It is also why **shared memory does not survive a
fault** — the kernel genuinely ended, so anything staged in shared memory is
gone on resume and must be rebuilt.

### 5.3 Admission

A block that holds several pages while waiting for one more can deadlock
against itself: its own held pages occupy the slots its next fault needs.
Admission control exists to make the block declare its working set up front
and be admitted only when the table can satisfy all of it. It is opt-in
because the reservation costs throughput; it buys a much smaller reliable
cache floor.

---

## 6. Writeback

Writes are absorbed into the cache and written back in bulk. Intentions:

- **Flushing is asynchronous and batched**: submit puts for a range, return
  immediately, and await completion separately. One task per batch of pages,
  never one per page.
- **A dirty page must never be dropped.** Eviction of a dirty victim submits
  its writeback first. Losing this is silent data loss, which this design has
  paid for before.
- **A page's dirty flag clears only when its own record reports success.** A
  batch that partially failed must not clear the pages that did not land.
- **Completion belongs to whoever owns it.** A page written back as part of a
  batch does not carry its own completion; it is marked as riding the batch,
  and the batch's completion settles it. A page that is "landed but unsettled"
  must not be treated as clean, or later flushes will drop it.

---

## 7. Prefetch and preload

These exist so the common case is not a fault storm at kernel entry.

**Preload** fills the vector's backing store from host data before any kernel
runs. It is a bulk path, not a fault path.

**Prefetch** places a range of pages into the block tables ahead of the
launch, establishing residency deliberately rather than by demand. In the
resident regime this is what makes the zero-fault contract achievable: the
whole vector is placed, and the direct-mapped way guarantees every page has a
slot it can occupy.

The intended sequence is **preload → clear the cache → prefetch → launch**,
so the tables start in a known state rather than inheriting whatever the
previous phase left.

**In-kernel prefetch hints (unverified in detail):** a fetch may be submitted
as a hint for pages the kernel expects to need, so the transfer overlaps
compute rather than stalling on it.

---

## 8. Reorganization

Three distinct things, deliberately separated:

**Eviction** chooses a victim by score and recency, refuses pinned and
in-flight pages, writes back if dirty, and re-claims the slot. It is the only
path that takes the block lock in the steady state.

**Rescoring** pushes the *application's* notion of value down into the CTE
tier system so that placement across VRAM/DRAM/NVMe reflects what the kernel
knows. It is fire-and-forget and batched, with its own completion, because a
score is advisory — nothing should stall on it. It is kept separate from
eviction scoring so that a tier decision and a cache decision do not
overwrite each other.

**Clearing** resets the tables to empty. It exists so a phase change can start
from a known placement instead of inheriting a stale one, and it is why the
documented sequence puts it before prefetch.

---

## 9. Concurrency model

- **Block-collective**: every thread of a block executes the same holds and
  takes the same branches. Divergence around a suspend is a hang.
- **Per-block lock**, held only for claiming, eviction, and batch bookkeeping
  — never for a hit.
- **Publish order**: "not readable" is published before "this is page P", so a
  lock-free probe can never see a slot as resident before its bytes exist.
- **Pins** make eviction safe against readers; a pin is held by the guard.
- **Generations** let a reader prove after the fact that its slot was not
  recycled.
- **Nothing blocks on the host while holding the lock.** *(This is the
  intended rule. It is currently violated in several places, which is the
  subject of active work: some batch-settle paths spin on a host completion
  under the block lock instead of declining and letting the caller park.)*

---

## 10. The invariants everything rests on

1. A page is never readable before its bytes are there.
2. A dirty page is never dropped.
3. A pinned page is never evicted.
4. A block never waits on another block.
5. A suspend is block-uniform.
6. Resident and out-of-core produce identical results.
7. A fault never spins in the kernel — it parks.
8. `__shared__` state does not survive a fault.

Most defects in this component's history are violations of one of these, and
most of them presented as *plausible wrong answers* rather than crashes — a
stale page read as valid data, a dropped writeback, an energy that drifts. That
is why the gates are digit-exact comparisons in both regimes rather than
smoke tests.

---

## 11. What the design costs — measured, per piece

Not estimated. `benchmark/gpu_vector_regprobe.cc` compiles one kernel per
internal function, **each in its own translation unit** (compiled together,
every coroutine kernel inherits the module's worst one and all rungs report the
same number). sm_89, `-O3`, SASS instruction count and ptxas register count.

| piece | SASS | REG | over baseline |
|---|---|---|---|
| baseline (kernel shell + `CLIO_GPU_INIT`) | 56 | 22 | — |
| **empty coroutine + yield driver** | **200** | **28** | **+144** |
| `Find` | 360 | 34 | +304 |
| `IsResident` | 440 | 39 | +384 |
| `SettleBatchLocked` | 464 | 39 | +408 |
| `EnterHoldSet` | 488 | 32 | +432 |
| `ReapFlushed` | 504 | 39 | +448 |
| **`ProbeHold`** | **600** | **44** | **+544** |
| **`TryHoldFast` (the whole hit path)** | **632** | **46** | **+576** |
| `ReapFetched` | 1,392 | 36 | +1,336 |
| `EvictPages` | 1,480 | 48 | +1,424 |
| `ClaimSlotWindowLocked` | 1,576 | 36 | +1,520 |
| `StartEvictionAsync` | 1,848 | 44 | +1,792 |
| `BeginFetchRunLocked` | 3,104 | 56 | +3,048 |
| `FlushRangeBatchedAsyncLocked` | 3,120 | 62 | +3,064 |
| `BeginFetch` | 3,160 | 48 | +3,104 |
| `AwaitFlush` | 4,232 | 46 | +4,176 |
| **`FetchPagesBatchedLocked`** | **8,208** | **72** | **+8,152** |
| **`HoldPage` (everything)** | **20,000** | **90** | **+19,944** |

### What this says

**The hit path is cheap.** `TryHoldFast` — probe, block-uniform vote, pin — is
**632 instructions, 3% of a hold**. The lookup itself (`ProbeHold`) is 600.
Nothing about resident access is expensive.

**The coroutine is not the cost.** The empty coroutine plus its driver is
**200 instructions and 28 registers**, 1% of a hold. Every theory blaming
`co_await` for the kernel bloat was wrong, and this rung is the disproof.

**The miss path is 97% of a hold.** ~19,400 of the 20,000 instructions are
code that runs only when a page is absent — and it is compiled into every
kernel, at every hold site, whether or not it ever executes.

**The single largest piece is `FetchPagesBatchedLocked`: 8,208 instructions
and 72 registers — 41% of a hold by itself.** The fetch/submit family
(`BeginFetch`, `FetchPagesBatchedLocked`, `BeginFetchRunLocked`,
`FlushRangeBatchedAsyncLocked`, `AwaitFlush`) accounts for essentially all of
the rest. That family is **task submission**: constructing a runtime task,
filling it with shared-memory pointers, pushing it on the gpu2cpu ring,
waiting on a future, and settling the batch afterwards.

For scale, a comparable GPU page cache (BaM) implements its **entire** cache —
tag check, slot mapping, pin, miss handling — in ~1,672 instructions, which is
smaller than a single one of our fetch-submission functions.

### The conclusion this forces

The bloat is not paging, not coroutines, and not the lookup. It is that **the
kernel builds and submits runtime tasks itself**, and that machinery is
duplicated into every kernel and every hold site.

The structural fix follows directly and is not yet done: a fault already ends
with the kernel exiting, so the orchestration that precedes that exit — claim,
evict, submit, settle — does not need to be inside the compute kernel at all.
Recording "this block needs page P" and parking would leave the compute kernel
with the 632-instruction hit path and drop the other ~19,400.

