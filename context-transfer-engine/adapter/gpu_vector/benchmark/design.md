# gpu_vector: the device vector

The design, after the brief and the round of hole-poking. Everything that was a
hole is now either a rule below or an open decision in section 8.

Two facts the whole design rests on, because forgetting either is what produced
most of the complexity being removed:

- **A block is only ever inside one public call at a time.** There is no
  concurrency between `HoldPage` and `BeginFlush` to reconcile.
- **Every task object is per-block.** A block owns its tasks the way it owns its
  page table.

---

## 1. Principles

1. On return from a `co_await`, the data is resident. No re-check loop, no
   "declined" path in the kernel.
2. Anything that can wait is a `co_await`. No `Wait()`, no spin, no blocking
   call in device code.
3. The vector states its requirements; the kernel meets them. It does not bend
   to accommodate access patterns it cannot serve well.
4. The caller owns flushing. The vector never writes back on its own.
5. Prefetch is advisory. It may decline silently; it may never evict a dirty
   page.

---

## 2. The contract with the caller

The vector serves kernels that meet these, and should refuse or warn loudly
rather than silently degrade.

### 2.1 Collectivity

**One `HoldPage` call is block-collective and names one page for the whole
block.** A block builds a multi-page working set by making several calls, each
collective, each keeping its guard alive.

This costs the workloads nothing — it is already how they are written. The
LAMMPS pair loop accumulates bin spans through a loop of sequential collective
holds (`md_bench.cc:573–589`); a straddling row takes a second hold
(`:577–582`); a stencil takes three. Every hold argument in that kernel is
block-uniform (`pg * epp`, `rbase[q]`, `fbase`, `rb = f(wz, rl[t], nb, cap)`).
Nothing derives an offset from a lane id.

**The one access this cannot serve is per-lane gather** — GNN feature rows
indexed through a neighbour list, where 32 lanes want 32 unrelated rows and
there is no locality to exploit. That is not a paging problem and `HoldPage`
should not pretend to solve it. See section 8.

### 2.2 Geometry

- `page_bytes` is a power of two. Every index derivation becomes a shift and a
  mask; non-power-of-two forces integer division, which the GPU has no
  instruction for — measured at 316 div/mod ops per hold.
- **The block's access unit — a row, tile, or record — lies wholly within one
  page.** This is the highest-value requirement here: it removes the two-span
  split, and with it a second hold, a second guard, and a select on every
  element access.
- Block size is known at compile time, so the launch shape can be declared.

Geometry is checkable at construction. The access-unit rule is a property of
the algorithm and can only be stated — but violating it is what produced the
out-of-core read corruption, where a stale slot moved a hold's pointer by whole
pages, and only non-power-of-two page sizes took that arm.

### 2.3 Data ownership

- **A dirty page belongs to exactly one block, and flush ranges never overlap.**
  This is the invariant that removes deduplication from the design entirely —
  nothing to collapse host-side, no find-or-claim device-side. The vector
  cannot check it; violating it means one block's writeback discards another's
  writes, which is how 59% of a PME charge grid vanished with every counter
  reading clean.
- Flush before the dirty set fills the table. A dirty page is never a victim,
  so a block whose table is all-dirty cannot fault and will fail loudly.
- Read sharing is free — the same page may be resident in many blocks, each in
  its own private table.

### 2.4 Kernel structure

- **`__shared__` state does not survive a hold that can fault.** The yield
  driver exits the kernel on a suspension; anything staged in shared memory is
  garbage on resume. Use registers or per-block global scratch, and re-stage
  after every suspension point (`md_bench.cc:612`).
- Holds are scoped tightly. A guard alive across the next hold pins its frame —
  correct for an intended working set, a slot leak otherwise.

---

## 3. Class structure

```
template <typename T> class DeviceVector {
 public:   // variables   — what a kernel may read
 private:  // variables   — header, page table, the three tasks
 public:   // functions   — HoldPage, BeginFlush, EndFlush, Prefetch
 private:  // functions   — Probe, EvictPages, Submit, Reap
};
```

`Held<T>` and `Page` are separate types, not nested, so the four sections stay
literally four. No function exceeds 100 lines.

---

## 4. State

### 4.1 Per frame

| field | meaning |
|---|---|
| `page_num` | resident page, or `kNoPage`. Published **last**, read **first** |
| `state` | `FREE` / `FETCHING` / `RESIDENT`, mutated only by atomic CAS |
| `pins` | atomic counter; non-zero means unevictable |
| `dirty` | set at hold time when `write` was declared |
| `flushing` | put outstanding; unclaimable until retired |
| `evict_rank` | eviction score. Higher means keep |
| `last_access` | recency, the tiebreak |

The publish-last / read-first ordering on `page_num` is the whole correctness
argument for the lock-free path.

### 4.2 Per block: three tasks

| task | used by | shape |
|---|---|---|
| `GetBlob` | synchronous fault | scalar — one collective call, one page |
| `MultiGetBlob` | asynchronous prefetch | bulk + counter — a range is many pages |
| `MultiPutBlob` | flush | bulk + counter — a block flushes its dirty set at once |

**Bulk where the caller names a range, scalar where the caller names a page.**
This does not contradict the bulk-only directive: that rule is about never
issuing a task per *element*, and about the flush path specifically, where
scalar puts capped throughput at ~9 GB/s.

Sizing: the flush task holds up to `slots` entries, so the footprint is
`nblocks × slots` entries per task type, fixed at construction.

**Reuse across a park** is the one hazard here: a block rearming its own task
must clear the completion flag and `run_ctx_` first, or a resumed block reads
the previous round's completion and proceeds on data that never landed. A
per-block sequence number checked on resume makes that an assertion rather than
an assumption.

---

## 5. Public API

All entries are block-collective. Every thread of the block enters and leaves
together.

> **Predicate the stores, never the suspension.** `if (tid == 0) { … }` then
> `__syncthreads()` then `co_await`, in that order, always. A partial arrival
> at the yield driver's block-wide barrier traps on sm_89 as an illegal
> instruction — this was the LAMMPS 256k-atom crash.

### 5.1 `Held<T> co_await HoldPage(off, count, write)`

```
1.  Probe. One page number for the whole block.
2.  Vote the OUTCOME: __syncthreads_and over "resident".
3.  All hit → pin, return. No suspension.

    On a miss:
4.  co_await EvictPages(1, 1)                  — secure a frame
5.  Parameterize the GetBlob task              [tid 0]
6.  Submit                                     [tid 0]
7.  __syncthreads()
8.  co_await completion                        [unpredicated, whole block]
9.  Publish page_num, mark RESIDENT
10. Re-probe. Resident by contract; otherwise trap.
11. Pin, return.
```

**Step 2 votes the outcome, not each thread's own probe.** On a shared table
another block's eviction can flip a slot between two threads' reads; returning
per-thread answers splits the block across the suspend. All-fast only when
every thread held the page.

**Step 10 is an assertion, not defensive coding** — it is what makes principle
1 checkable rather than hoped-for. A livelock here previously read as a *pass*
with correct data while a round cap silently ended the run.

### 5.2 `Held<T>` — the guard

| member | kind | notes |
|---|---|---|
| `ptr()` | accessor | base of the run |
| `run()` | accessor | elements reachable without another hold |
| `RescorePage(rank)` | mutator | sets this frame's `evict_rank` |

`RescorePage` is the whole of rescoring, and deliberately **not** a task, not
asynchronous, not a `co_await`. The caller already holds the page pinned, so
setting its rank is one store to a frame nobody can take away.

- **Higher rank means keep**; `EvictPages` takes the lowest. State the direction
  explicitly — the CTE blob score runs the other way in the DPE's preferred
  group, and that inversion has bitten this code before.
- It is a store, so predicate it. No barrier: nothing in the block reads it
  back, only eviction does.
- **Rank is cache-local and dies with the frame.** Evict and it is gone; fault
  back and it is the default. It is a hint about *this* residency, not durable
  metadata. The default should be a named constant so "never rescored" is a
  legible state rather than a magic float.
- It never touches `blob_score` (§7).

### 5.3 `co_await BeginFlush(range)` / `co_await EndFlush()`

`BeginFlush` marks the range's dirty frames `flushing`, augments the
`MultiPutBlob` with one entry per frame, submits, and returns without waiting.
`EndFlush` awaits the outstanding put and retires the flags.

- A `flushing` frame is never a valid victim.
- **Retirement clears `flushing` and `dirty` together**, never one alone.
- Landed-but-unretired flushes have wedged this vector before — a frame left
  mid-retire caused later flushes to be dropped silently.

### 5.4 `co_await Prefetch(range)` — advisory

Takes `FREE` frames and may evict **clean** ones. Never evicts dirty, never
writes back, never fails: with no room it does nothing, because silence is the
correct answer to a request nobody is blocked on.

The difference from a fault is **policy, not shape**: mandatory may displace and
must fail loudly; advisory may not and fails silently.

---

## 6. Private API

### 6.1 `co_await EvictPages(min, max)` — mandatory

1. Scan **this page's d-left ways only**, not the whole table.
2. If any way already holds the page → return it. (A hit, not a dedup: the
   block is collective, so there is no second claimant inside the table.)
3. Skip pinned, dirty, fetching, flushing. What remains is the candidate set.
4. Take `FREE` frames first, then candidates in victim order.
5. Fewer than `min` and a flush is in flight → `co_await` it, then retry.
6. Still fewer than `min` → **fail loudly**: print which state blocked it
   (dirty / pinned / flushing / fetching, with the offending slots) and trap.

Step 5 is not a policy violation of principle 4. The only put that can be in
flight is one the caller started with `BeginFlush`; waiting on it is observing
the caller's own I/O, not initiating a writeback.

Step 6's message must name the blocking state. "All candidates unusable" sends
the reader guessing between three unrelated causes: unflushed writes (a caller
error), a leaked guard (a pin held too long), or in-flight I/O (timing).

**Victim order.** Sort candidates ascending by `(evict_rank, last_access)`:

```
worse_victim(a, b) =  a.evict_rank  <  b.evict_rank        ? true
                   :  a.evict_rank  >  b.evict_rank        ? false
                   :  a.last_access <  b.last_access
```

Score first, LRU second; recency only ever decides ties. Both keys ascend, so
the pair reads naturally: lowest rank is least wanted, and among equally-wanted
frames the stalest goes.

The tiebreak is load-bearing, not cosmetic. Equal ranks are the *common* case —
a block that never calls `RescorePage` has uniform ranks and the policy is pure
LRU, so the second key does all the work. Dropping it changed a scoring test's
fault count from 14 to 13.

It is a two-key min-scan over `nways` frames, not a sort. Nothing is
materialized.

---

## 7. Two things called "score"

They are unrelated and must never share a field:

- **`evict_rank`** — eviction order. Unbounded, block-local, higher means keep.
- **`blob_score`** — CTE tier preference in `[0,1]`. `PutBlob` **rejects > 1.0
  with rc=5**, and the DPE's sort direction depends on its value.

Conflating them has caused silent data loss twice: an unclamped `evict_rank`
forwarded into a put made every page touched more than once fail with rc=5
while the flush counter still read success. The conversion clamps.

---

## 8. Open decisions

### 8.1 Who resolves the fault — the kernel or a host servicer?

**This is the one architectural question left, and it should be settled before
any code is written.** The protocol in §5.1 follows the brief: the kernel
evicts, parameterizes a `GetBlob`, submits it, and awaits completion.

What is implemented today is the opposite: the kernel records the page in a
mailbox and parks, and a **host servicer** claims the frame, fetches, and
publishes. That change was made because of a measurement — in-kernel fetch
machinery was **19,400 of a hold's 20,000 instructions**, dominated by
`FetchPagesBatchedLocked` (8,208), `BeginFetch` (3,160), `BeginFetchRunLocked`
(3,104), `StartEvictionAsync` (1,848), `ClaimSlotWindowLocked` (1,576),
`EvictPages` (1,480). Moving it host-side cut the hold to 2,184.

The brief's version is far slimmer than what was removed — a scalar get, no
batching, no settle/reap, and no writeback-on-evict since dirty frames are
never victims. But it is the same *category* of work: victim selection and task
submission, inlined at every hold site in every kernel.

The honest options:

| | kernel resolves (§5.1) | host servicer (today) |
|---|---|---|
| device cost | `EvictPages` + submit, inlined per hold site | probe + record + park |
| host servicer loop | none needed | required |
| flush retirement | device reaper on resume | either side |
| matches the brief | yes | no |

Worth measuring before choosing: build the §5.1 path and price a hold with
`gpu_vector_regprobe.cc`. If it lands near 2,184 the brief's version wins on
simplicity; if it lands near 20,000 the measurement that drove the servicer
still applies.

### 8.2 Gather

Per-lane pages cannot be served by a collective `HoldPage` (§2.1). GNN needs
something: a separate batched-gather verb, or that workload does not page.

### 8.3 Task footprint

`nblocks × slots` entries per task type is real pinned memory at large block
counts. Worth capping flush to a partial table instead?

### 8.4 Shared tables

The vote in §5.1 step 2 exists because eternia's read-only shards are shared
between blocks, so another block's eviction can flip a slot mid-scan. If shared
tables remain supported, that is the one place the lock-free protocol has a
real adversary. If they are dropped, section 6 simplifies further.

---

## 9. Rules that came from bugs

Kept here so the code does not need to carry them as comments. Each cost real
time to find.

| rule | what happened without it |
|---|---|
| `TryHoldFast` must open with `__syncthreads()` | 8 force writebacks instead of 123; energy −4.6731 vs −4.7639 — wrong, and plausible enough to pass an eyeball |
| Vote the outcome, not per-thread probes | block split across the suspend; sm_89 illegal instruction at 256k atoms |
| `__shared__` does not survive a park | run-varying, config-dependent near-miss answers, no crash |
| Never evict a dirty page; the caller flushes | silent loss of another block's writes |
| Clamp `evict_rank` before it becomes `blob_score` | rc=5 on every twice-touched page, flush counter still reading success |
| LRU tiebreak in victim order | scoring test read 13 where the policy gives 14 |
| Host page transfers go through the composed pool | raw zstd frames memcpy'd into page frames; gnn features corrupt |
| Assert convergence separately from correctness | a livelocked test *passed* with correct data |
| One rung per translation unit when measuring registers | every coroutine kernel inherits the module's worst; all rungs report the same number |
