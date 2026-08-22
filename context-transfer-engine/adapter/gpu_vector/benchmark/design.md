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


---

# Part II — The target design

Part I describes what exists. This part describes what it should be, given the
measurements in section 11 and three rules:

1. **On return from a `co_await`, the data is resident.** No re-check, no
   retry loop, no "declined" path in the kernel.
2. **Anything that can wait is a `co_await`.** No `future.Wait()`, no spin, no
   blocking call anywhere in device code.
3. **The vector states its requirements and the kernel meets them.** It does
   not bend to accommodate access patterns it cannot serve well.

---

## 12. What the measurements actually indict

The hit path is fine: 632 instructions, 46 registers, 3% of a hold. Keep it.

The indictment is narrow and specific: **the kernel builds and submits runtime
tasks.** `FetchPagesBatchedLocked` alone is 8,208 instructions and 72
registers; with `BeginFetch`, `BeginFetchRunLocked`,
`FlushRangeBatchedAsyncLocked` and `AwaitFlush` it is essentially the entire
20,000. That code constructs a task, populates it with shared-memory pointers,
pushes it on the gpu2cpu ring, waits on a future, and settles a batch --
**inside a kernel whose next act, on the path that needs it, is to exit.**

It is also duplicated at every hold site in every kernel, so a kernel with
eighteen holds carries eighteen copies of a fetch engine it may never run.

The current design reached this state honestly: it made the fault path
self-service so a block could make progress without leaving. But a block
*cannot* make progress without leaving -- the servicer cannot run while the
kernel holds the GPU. The self-service machinery buys nothing and costs
everything.

---

## 13. The target: the kernel resolves, the servicer provides

**The kernel's device-side page-cache code reduces to two things: a probe, and
a fault record plus a park.** No claim, no eviction, no writeback, no task
construction, no batching, no settlement.

### 13.1 The fault protocol

1. **Probe.** Block-uniform lookup in this block's table, voted. A hit returns
   a pointer and a run length. This is the 632-instruction path, unchanged.
2. **Record.** On a miss, thread 0 writes one small record -- page number,
   write intent, block identity -- into this block's fault slot. A few stores.
3. **Park.** The block suspends with a wait tag naming its fault slot, and the
   kernel returns. This is the existing park; nothing new is needed.
4. **Serve.** The servicer -- the host-side driver that already relaunches
   parked kernels -- reads the outstanding fault records across *all* blocks
   and, for each: chooses a victim, writes it back if dirty, issues the fetch
   (batching freely across blocks), waits for completion, publishes the page
   into the slot, and only then clears the fault.
5. **Resume.** The kernel relaunches and the block re-probes. **By rule 1 this
   is guaranteed to hit**, because the servicer does not clear the fault until
   the page is resident and readable.

The kernel-side cost of a miss becomes a few stores and a suspend. Everything
that made the miss path 19,400 instructions moves to the host, where it is
written once, in ordinary C++, with no register budget and no instruction-cache
pressure.

### 13.2 What this deletes outright

- `FetchPagesBatchedLocked`, `BeginFetch`, `BeginFetchRunLocked` -- task
  submission leaves the device.
- `FlushRangeBatchedAsyncLocked` -- writeback becomes the servicer's job,
  triggered by eviction or an explicit flush epoch.
- `SettleOneLocked`, `SettlePutLocked`, `SettleScoreLocked`,
  `SettleBatchLocked`, `ReapFetched`, `ReapFlushed` -- nothing to settle on the
  device if nothing was submitted there.
- Every `future.Wait()` -- rule 2, achieved structurally rather than by
  discipline.
- `ClaimSlotWindowLocked`, `StartEvictionAsync`, `EvictPages` -- slot policy
  becomes the servicer's.
- `RetryLostFetch` and the re-issue loop -- rule 1 makes a resume that misses
  impossible, so there is nothing to retry.
- The block lock in the steady state: with no claiming or eviction on the
  device, the only device-side table access is a read-only probe.

Projected kernel cost per hold site: **~700 instructions instead of 20,000**,
registers in the mid-40s instead of 90 -- the hit path plus a store and a
suspend.

### 13.3 What it buys beyond size

- **Better batching, not worse.** The servicer sees every block's outstanding
  faults at once. A single block can only batch its own; the servicer can
  coalesce across the whole grid and issue one transfer where the current
  design issues many.
- **The eviction policy gets a global view** instead of a per-block one, and
  can change without recompiling a kernel.
- **Rule 2 becomes structural.** There is no device-side future to wait on, so
  no one can reintroduce a spin.
- **The block lock stops being a latency risk.** Today a device thread can spin
  on a host completion while holding it.

### 13.4 What it costs, honestly

- **A fault always costs a kernel exit and relaunch.** Today a fault that could
  be satisfied from an already-landed batch is retired in-kernel; that case
  disappears. This is only acceptable because faults are meant to be rare --
  and where they are not, the answer is prefetch, not in-kernel servicing.
- **Per-fault latency may rise** even as throughput improves, because
  everything round-trips through the servicer. The mitigation is cross-block
  batching, which lowers the *per-page* cost.
- **The servicer becomes load-bearing** and sits on the critical path of every
  fault.
- **The fault slot is a new shared structure** needing its own discipline --
  though one writer per block is far simpler than the current multi-state page
  table.

---

## 13.5 Two principles that shrink the fault path

**Principle 1: eviction takes clean pages only.** A page whose bytes are
already in the backing store can be dropped instantly -- no I/O, no waiting,
no completion to track. Eviction becomes a pointer operation.

**Principle 2: the vector never writes back implicitly. Flushing is the
caller's job.** There are explicit flush verbs for exactly this reason. The
vector does not decide when the user's data is written; the user does.

Together these delete the largest remaining piece of the fault path. If
eviction never writes back, then:

- there is no dirty-victim writeback to submit, so `StartEvictionAsync`
  (1,848 instructions) and `FlushRangeBatchedAsyncLocked` (3,120) leave the
  fault path entirely;
- there is nothing to wait for mid-fault -- the fault's "make room" step stops
  being an I/O operation and becomes a scan for a clean frame;
- the put/settle machinery disappears from the miss path along with it.

The measured miss path was ~19,400 instructions. Removing task submission
(section 13) and implicit writeback (here) accounts for nearly all of it.

**What happens when there is no clean victim.** Every frame in the block's
table is dirty, so the fault cannot be served without violating principle 2.
This is a *caller* error -- the block's dirty working set exceeds its table --
and the right response is to say so, loudly, not to paper over it with a
writeback the user did not ask for. It joins the requirements in section 14:
flush before you exceed your table, or size the table for your dirty set.

This is also why a fault must be able to fail. Section 13.1 says a resumed
block finds its page resident; the honest form of that guarantee is "resident,
or the run stops with a diagnosable error" -- never "resident eventually,
after some I/O you did not request".

---

## 14. Requirements on the caller

The vector serves kernels that meet these. It should refuse or loudly warn
about kernels that do not, rather than silently degrading.

### 14.1 Geometry

- **Page size is a power of two.** Every index derivation becomes a shift and a
  mask. Non-power-of-two sizes force integer division, which a GPU has no
  instruction for -- measured at 316 div/mod operations per hold today.
- **Pages per block and block count are powers of two**, for the same reason.
- **A page divides the kernel's access unit.** A row, tile or record must lie
  wholly within one page. This is the highest-value requirement in this list:
  it removes the two-span split, and with it a second hold, a second guard, and
  a select on every element access.
- **Block size is known at compile time.** Template the kernel on it, so the
  launch shape can be declared and the compiler can derive the register count
  for the real configuration instead of guessing.

### 14.2 Access discipline

- **Access is page-granular.** Take a hold, get a pointer and a run, loop over
  the run. Per-element resolution is not a supported pattern.
- **Holds are block-uniform.** Every thread of the block takes the same hold; a
  hold only some threads take will hang at the suspend.
- **The working set is declared and fits the block's table.** A block that
  holds more pages than it has slots cannot progress; the vector should refuse
  that configuration at launch rather than deadlock at runtime.
- **Holds are scoped tightly.** A guard alive across the next hold keeps its
  page unevictable -- correct for an intended multi-page working set, a slot
  leak otherwise.

### 14.3 Data ownership

- **A dirty page belongs to exactly one block.** Page-granular writeback means
  one block's eviction discards another's writes. Partition writes by page.
- **Flush before your dirty set exceeds your table.** The vector never writes
  back on its own (13.5), so a dirty page is unevictable until you flush it. A
  block whose dirty pages fill its table cannot fault in anything new, and the
  vector will say so rather than silently writing your data out.
- **Read sharing is free.** The same page may be resident in many blocks.

### 14.4 Kernel structure

- **`__shared__` state does not survive a hold that can fault.** A fault exits
  the kernel; anything staged in shared memory is gone on resume. Use registers
  or per-block global scratch, or re-stage after every hold.
- **Prefetch what you can predict.** The design targets the resident case with
  faults as the exception. A kernel whose access pattern is known ahead of time
  should declare it rather than discover it by faulting.

### 14.5 The contract in one line

> Give the vector page-aligned, block-uniform, page-granular access with a
> declared working set that fits, and it will give you an array larger than
> VRAM at close to resident speed. Violate any of those and it will still be
> correct, but it will not be fast -- and the vector should say so.

---

## 15. First implementation attempt — what it proved and what it did not

Attempted 2026-08-22. Skeleton committed; the design is **not yet working**.

### What was built

- `FaultReq`, a per-block mailbox in pinned host memory (page number, write
  intent, pending flag).
- `DeviceVector::HoldPageServiced` -- probe, and on a miss record the fault,
  park on the mailbox's `pending` field, re-probe on resume. No claim, no
  fetch, no task submission, no future.
- `Vector::EnableServicedFaults()` / `Vector::ServiceFaults()`, the host side.
- A driver that runs the host servicer between kernel rounds.

### What the device side proved

**The protocol works.** The kernel recorded faults, parked, and resumed; the
host observed every request through the mailbox and acknowledged it. The park
and resume path needed no new machinery -- the existing yield driver already
relaunches parked blocks, and its `service` callback is the right hook.

### What it did not prove: the host lacks the primitive

Both attempts to service a fault failed, for two separate reasons, and both are
gaps in the HOST side rather than the protocol:

1. **First-touch writes cannot be fetched.** A page that has never been written
   has no blob, so `Prefetch` returns 0 and nothing becomes resident. The
   servicer needs an *allocate-empty* case beside *fetch*: claim a slot, zero
   the frame, publish it. The legacy in-kernel path does this implicitly; the
   host path has no equivalent.
2. **`Prefetch` was the wrong primitive, and correctly refused.** With the
   table full it placed nothing: instrumented readback showed the page absent
   from the device table while `Prefetch` reported having fetched it.

   That refusal is **right, not a defect**. Prefetch is ADVISORY -- a
   recommendation about pages that may be wanted soon. An advisory operation
   must never evict: it would discard a page some block is using to make room
   for one nobody has asked for yet. When there is no free slot, the correct
   answer is to do nothing.

   Servicing a fault is the opposite: MANDATORY. A block is parked and cannot
   proceed until that exact page is resident, so the servicer must be willing
   to displace something. Using the advisory primitive for the mandatory job
   was the mistake.

   (It also sets `no_evict_` from whether its own range was fully placed --
   another reason it cannot be reused as-is, and another sign these are
   different operations.)

Symptom of both: the block re-faulted on the same page every round until the
driver's 200,000-round cap. The data check still read zero mismatches, so an
earlier version of the test **passed while livelocked** -- convergence has to be
asserted separately from correctness, and the round count is the assertion.

### Advisory vs mandatory placement

The attempt clarified a distinction the design should state outright, because
conflating the two is what broke it:

| | advisory (`Prefetch`) | mandatory (fault servicing) |
|---|---|---|
| who asked | nobody yet -- a prediction | a parked block, by page number |
| may evict a **clean** page | yes | yes |
| may evict a **dirty** page | **never** | **never** |
| may decline | yes, silently | only by failing loudly |
| page absent from backing store | skip it | allocate an empty frame |

Both columns may take any **unmodified** page: its bytes are already in the
backing store, so dropping the frame costs nothing and loses nothing. Neither
may take a dirty one -- see the next principle.

`Prefetch`'s current refusal to evict *at all* is stricter than it needs to be;
it may reclaim clean frames. What it must never do is silently write back.

### What the design needs before it can work

A host-side **claim-with-eviction** implementing the right column, which does
not currently exist:

- choose a **clean** victim in that block's table by score and recency,
- refuse pinned, dirty, or in-flight slots,
- fetch the requested page, or zero the frame if it has no blob yet,
- publish `page_num` last, and only then clear `pending`,
- and if no clean victim exists, fail loudly rather than write back (13.5).

Note what is *absent*: no writeback, and therefore no waiting inside the
servicer either. Eviction is a scan and a pointer swap.

That is the same policy the device path implements today, moved to the host --
which is the whole point, since on the host it costs no registers and no
instruction cache. It is the real work of Part II, and it is not a small
change: it must reproduce the victim selection and writeback ordering that the
device path gets right, including the rule that a dirty page is never dropped.

Until it exists, `HoldPageServiced` is a skeleton and the legacy in-kernel path
remains the only working fault path.
