# Shared Caching

In a separate branch, let's try making caches shared.
I want to make an associative set cache.

Today every CUDA block owns a private page table, so a page touched by N
blocks is stored N times. Measured on the MD workload (lattice 50, 500k atoms,
`--blocks 64 --threads 256`, 128 KB pages):

| | |
|---|---|
| unique data across all six vectors | 347 MB |
| cache actually spent | 1472 MB (**4.2x**) |
| x: distinct pages | 79 |
| x: frames allocated (64 blocks x 24) | 1536 (**16x**, and still evicting) |
| x pages held by >1 block | 100.0%, mean degree 8.22, max 17 |
| total process VRAM | 1830 MiB vs MPI's 494 MiB (**3.7x**) |

The five atom vectors are 12 MB each. A shared cache can hold **all of them
resident for 60 MB** where today's private tables spend 960 MB and still
fault. That is the prize; the list (sharing degree 1.39) stays roughly as-is.

---

## Terminology

"Block" below means an **associative set**, NOT `blockIdx.x`. There are
`num_blocks` sets and a page's set is `Hash(page_id) % num_blocks` -- a
function of the PAGE, not of who is asking. That single change is what makes
the cache shared. Where CUDA blocks are meant, this doc says "CUDA block".

Physically this reuses today's layout: one flat frame array, `num_blocks`
contiguous runs of `set_size` frames each. `Pages()` stops being
`pages_ + Table() * pages_per_block_` and becomes
`pages_ + Set(pn) * set_size_`. Frame indices become GLOBAL, which matters for
`BlockTasks::fetch_slot[]`/`flush_slot[]` -- those stay per CUDA block (task
submission is still per CUDA block) but must now store a global frame index.

## Sizing

We will have a single device memory segment for holding pages.
Each set stores an in-memory array of cached pages.
The number of pages that each set will initially store is calculated
automatically. It is the user-specified `2 * ((HBM cache size) / page_size /
blocks)`. The `2*` here is for collisions.

Worked example at a 400 MB shared budget, 128 KB pages, 64 sets:
`400 MB / 128 KB = 3200 frames`, `3200 / 64 = 50`, `set_size = 100`.
The gather pins ~14 pages per CUDA block; 64 CUDA blocks = ~896 pins spread
over 64 sets = **~14 pins per set against 100 slots**. Comfortable, but see
"Pinned working set is per SET" below -- that ratio, not the global one, is
what decides whether this wedges.

`PlanCaches` collapses when this lands: no more per-vector floors scaled by
rows-per-block, because those floors exist only to survive per-block
replication. One budget, one cache, and the question becomes "does the working
set fit" instead of "how many frames per block per vector".

## Maintaining Cache Size

Current cache size should be stored as a 64-bit atomic integer.
We need to know that the cache size constraints are being maintained.
Our host-side prefetching system should ensure to set this as well.

The counter is in FRAMES (or bytes -- pick one and never mix; frames is
cheaper and partial validity never changes a frame's cost). Note that
`Prefetch`, `ClearCache` and `InitPageTable` all populate or drop frames
host-side today and every one of them has to maintain this counter, or the
device will evict against a number that has drifted.

## IsResident(PageId)

`hash(PageId)` to get the set and then perform an iteration of the set. The set
is really a list that must be iterated in order. This will be a warp-wide
iterator + `reduce_max_sync`. If a page is not faulted, place the offset in the
reduction.

`__reduce_max_sync` / `__reduce_min_sync` are **sm_80+**. This host is sm_89 so
it is fine here, but it becomes a hard floor for the adapter -- Volta and
Turing would need a shuffle-based fallback.

## FindFreePage(block_id)

Find a page that is free in the cache using a warp-level reduction like this:
```
__device__ inline int blockReduceMinSimple(int val) {
    __shared__ int s_min;

    // 1. Thread 0 initializes the shared minimum
    if (threadIdx.x == 0) s_min = INT_MAX;
    __syncthreads();

    // 2. Intra-warp reduction (1 instruction)
    int warp_min = __reduce_min_sync(0xFFFFFFFF, val);

    // 3. Across-warp reduction: only lane 0 of each warp updates shared memory
    if ((threadIdx.x & 31) == 0) {
        atomicMin(&s_min, warp_min);
    }
    __syncthreads();

    return s_min; // All threads in block read the result
}
```

`__shared__` is safe here because nothing on this path suspends -- see the
lock decision below. The rule it has to keep obeying: a suspension EXITS the
kernel and the grid is compacted on resume, so `__shared__` state is garbage
afterwards (this has already cost us run-to-run-varying wrong answers in
eternia). If a reduction ever does need to span a `co_await`, its accumulator
belongs in `CLIO_SHARED_PERSIST`, not in `__shared__`.

## bool AllocatePage(page_id, Page *&page)

A private helper method for FetchRange.
`block_id = Hash(page_id) % num_blocks` to select the associative set.
Guard Lock the set.
If `IsResident(page_id)` return false.
`Page *page = FindFreePage(block_id)`
if (page == nullptr) {
  `EvictPage(block_id)`;
  `page = FindFreePage(block_id)`;
  if (page == nullptr) { abort with error; }
}
Mark page as valid AND pinned.

### The set lock: a SPIN LOCK, not CAS

DECIDED: a real lock on the set, spun on when contended.

**CAS is rejected, and the reason is duplication.** BaM's lock-free scheme
claims a *slot* atomically, which is enough when each slot is a fixed home for
one page. It is NOT enough here: `IsResident` -> `AllocatePage` is a
check-then-act over the WHOLE set, and two CUDA blocks fetching the same page
would each miss, each claim a different free slot, and each fetch it. Two
copies of one page in one set -- the exact replication this design exists to
kill -- plus two in-flight transfers into two frames, of which one silently
wins. Only a lock that spans the search and the claim makes "is it here, and if
not it is mine to fill" a single decision.

**The lock SPINS, and `AllocatePage` is NOT a coroutine.** The
never-spin-wait rule is about waiting on I/O, not about a bounded critical
section. This one is bounded: scan the set, pick a slot, mark it. No transfer,
no suspend, no kernel exit inside the lock.

That follows from the invariant below.

### INVARIANT: eviction performs no I/O

`EvictPage` and `EvictPageFrom` select a victim and drop it. They never
transfer, never submit a task, never await. A frame that is mid-flush or
mid-fetch is not a candidate; a frame that is unpinned and quiet is dropped on
the spot.

Everything else in this design rests on it: eviction is a scan -> the critical
section is bounded -> a spin lock is correct -> `AllocatePage` is not a
coroutine -> no suspend on the allocation path -> the
`__shared__`-across-`co_await` hazard cannot arise here. Today's `EvictPages`
violates this -- it `co_await`s `EndFlush()` before giving up -- and that path
goes away.

Build around:

- One lock word per set. Pad it so two sets never share a cache line, or
  unrelated sets serialize against each other.
- Nothing that can suspend may be called under the lock. That is an invariant
  worth asserting in debug builds, because it is what keeps the spin bounded.
- Pins are taken under the lock, so a page cannot be selected as a victim
  between being found free and being marked pinned.

## FetchRange

1. First convert the range into page ids.
2. For each page id, if `AllocateUniquePage(page_id, page)` {}
3. Build the multi get task for all non-resident pages and collect the total
   size of data being fetched (`fetch_size`). This should be number of pages *
   page size. Partial paging doesn't affect this number. Partial paging only
   affects the amount of data transferred to the page, not the size of the page
   in the cache.
4. Atomic increment the cache size (which should return the previous size as
   well).
5. If previous size was smaller than the allotted space, then evict
   `fetch_size - (max - previous)`. If it was larger, then evict `fetch_size`.
6. Submit the task.

This function needs to pin pages so that future evictions don't accidentally
take the pinned pages. Evictions are not necessarily from only this block
anymore, so we will need to ensure they don't evict pages we need.

### Notes

- Step 5, second branch: the debt is `previous + fetch_size - max`, which is
  `fetch_size + (previous - max)` -- strictly more than `fetch_size` when we
  are already over. Evicting only `fetch_size` leaves the overshoot permanently
  on the books. Same expression covers both branches; clamp at zero.
- `kPodMultiMax` is 64, so a range needing more than 64 pages already has to
  submit in waves. With sharing, another CUDA block may fill some of those
  pages between waves -- which is a saving, not a bug, but wave N+1 must
  re-check residency rather than trust the page list computed at step 1.
- Partial validity survives: a frame still carries `valid_lo`/`valid_hi`, and
  `SubmitFetch` still widens to the union of the resident interval and the
  request. Under sharing that widening mutates a frame two CUDA blocks can
  reach, so it happens **under the set lock**, alongside the residency check
  that decided to widen -- same critical section, same reason as `IsResident`
  -> `AllocatePage`.
- Generations: `p->generation` is now shared, so a generational fetch by one
  CUDA block satisfies every other one. Good. The stale-page reset
  (`valid_lo = valid_hi = 0`) is a shared mutation and goes under the same lock.

## HoldPage

`hash(pageid) % num_blocks` to get the set.
Find the page in the associative cache using warp-level reduction.
This will NOT increment the page references and will NOT auto unpin pages.
This function will now only get a reference to the page.

This inverts `Held`: today "the guard IS the pin" and `~Held()` calls
`atomicSub(&page_->pins, 1)`. That destructor must go, and with it the RAII
property -- the pin's lifetime becomes `FetchRange` -> `UnpinRange`, which the
caller controls explicitly. Worth being clear-eyed that this trades a
compiler-enforced invariant for a caller-enforced one; a missing `UnpinRange`
becomes a leaked pin that shrinks the effective set until the kernel aborts.
A debug-build counter of pins outstanding at kernel exit would catch it cheaply.

## UnpinRange(0, 100)

Should unpin all pages in the range.
This is now an explicit call since FetchRange will be the pinner of pages.

## bool EvictPage(this_block, fetch_block)

(Written `co_await EvictPage` above. Synchronous: eviction performs no I/O.)

Evicts one page from the cache.
if (`EvictPage(fetch_block)`) { return; }
if (`EvictPage(this_block)`) { return; }
for each block if `EvictPage(block)` { return; }
abort kernel.

## bool EvictPageFrom(block_id)

(Also synchronous, same reason.)

Can only evict pages that are not currently pinned or actively being
flushed/fetched.
First iterate over this block's page set and identify the Page that has the
lowest score, LRU.
All warps in the block should participate in this search.
This operation can fail, in which case it returns false.
It should be something like this:
```
__device__ inline int blockReduceMinSimple(int val) {
    __shared__ int s_min;

    // 1. Thread 0 initializes the shared minimum
    if (threadIdx.x == 0) s_min = INT_MAX;
    __syncthreads();

    // 2. Intra-warp reduction (1 instruction)
    int warp_min = __reduce_min_sync(0xFFFFFFFF, val);

    // 3. Across-warp reduction: only lane 0 of each warp updates shared memory
    if ((threadIdx.x & 31) == 0) {
        atomicMin(&s_min, warp_min);
    }
    __syncthreads();

    return s_min; // All threads in block read the result
}
```

### Pinned working set is per SET, not global

Freeing space in set B does not help a page that hashes to set A. So the
cross-set fallback in `EvictPage` recovers *budget*, not *slots*: if every slot
in A is pinned, A is stuck no matter how much global room exists. The binding
constraint is therefore

> max pages simultaneously pinned that hash to the same set  <=  set_size

For the MD gather that is ~14 against 100, so we have ~7x headroom -- but it is
a distribution, not an average, and `Hash` decides the tail. Two things follow:
`Hash` should mix (page_num alone means a strided access pattern hits one set),
and the abort path must say *which set overflowed and how many of its slots
were pinned*, because "no free frame" with no set identity is the message we
have already spent a day decoding once.

DECIDED: **pages are pinned in fetch.** `FetchRange` is the only pinner and
`UnpinRange` the only releaser, so the pinned working set is entirely the
caller's declaration -- it is exactly the ranges the caller said it needed and
has not yet released.

That makes an all-slots-pinned set a **caller error, and it aborts** -- the
same class as holding a page that was never fetched. The caller asked for more
simultaneously-live pages hashing to one set than the set can hold, and no
eviction policy can rescue that. The abort message must name the set, its
`set_size`, and how many slots are pinned, or it is the "no free frame after
EvictPages" message all over again, which cost a day to decode.

With a non-suspending allocate there is no lock-ordering deadlock: no CUDA
block ever holds one set's lock while waiting for another's. What remains is
pure capacity -- a CUDA block still holds pins across the `AwaitFetch` suspend,
so pins from many blocks coexist in a set. That cannot deadlock; it can only
overflow, and overflow aborts. Which is why the hash-distribution measurement
below is a gate rather than a nicety.

### Two different evictions, do not conflate them

The doc uses `EvictPage` for both, and they answer different questions:

- **Slot eviction** (`AllocatePage` -> `EvictPageFrom(block_id)`): "this page
  hashes to set A and set A has no free slot." Only set A can help. Freeing
  space anywhere else is irrelevant, because the page's home is fixed by the
  hash.
- **Budget eviction** (`FetchRange` step 5, and `EvictPage(this_block,
  fetch_block)` with its fall-through over every set): "the cache is over its
  byte budget." Any set will do, which is exactly why that one walks all sets.

The fall-through list belongs to the second and must not be used for the
first, or a full set will appear to succeed at freeing space and then still
fail to produce a slot.

## Remove the "Flushed" bit

We can remove the dirty boolean.
We do not need to indicate a page is dirty, only that it is being actively
fetched, flushed, or pinned.

DECIDED: **removed entirely.** No dirty state, not even in debug builds.

**Flushing what you wrote is part of the contract.** A writer publishes its
bytes with `Flush` before it unpins them. That is not a convention the cache
hedges against -- it is the same class of requirement as fetching a page before
holding it, which the vector already refuses to paper over. `HoldPage` does not
fault on your behalf; eviction does not flush on your behalf. The caller
declared the range, wrote it, and publishes it.

A dirty bit is the cache guessing at what the caller already knows. It exists
only to tell an evictor to write something back -- and **eviction performs no
I/O**, so there is nothing for it to tell. The bit has no reader. Delete it.

What that buys, restated from the invariant above: eviction is a scan,
`AllocatePage` needs no `co_await`, the set lock's critical section is
bounded, and `EvictPages`' `co_await EndFlush()` retry path disappears
outright. Write the flush at the write site.

---

## Verification plan

- **A/B, byte-identical.** Same kernel, private tables vs shared sets, outputs
  compared exactly. This is the gate; everything else is diagnosis.
- **Concurrency stress on one page.** Many CUDA blocks racing evict-vs-pin on a
  single page id. This is the regression test the eternia TOCTOU bugs never
  had, and both of those were found in production, twice, in the same shape.
- **Set overflow, deliberately.** Size a set so the pinned working set cannot
  fit and assert it aborts with the set id and pin count -- **loudly, not
  wedged**. A livelock here looks exactly like a slow run.
- **Hash distribution.** Report max/mean occupancy per set for the real MD
  access pattern before trusting the 7x headroom above.
- Existing gates unchanged: 17/17 single-node, md drift, distributed halo,
  CTE generations.

## Expected outcome

If this works, the atom vectors go 960 MB -> ~60 MB (fully resident, no
eviction) and the list stays private at ~512 MB, for roughly **572 MB of cache
against 1472 MB today** -- process VRAM near MPI's 494 MiB rather than 3.7x it.
Host RSS is unaffected: 1094 MiB of that is the CTE tier holding the dataset,
which sharing does not touch.

## Decisions taken

1. **Set lock is a real lock and it SPINS.** CAS is rejected: it claims a slot,
   not a set, so two CUDA blocks fetching the same page would each miss and
   each fetch it. Spinning is fine because the critical section holds no I/O.
2. **Pages are pinned in `FetchRange`**, released by `UnpinRange`. A set whose
   slots are all pinned is a caller error and aborts, naming the set.
3. **`AllocatePage` is not a coroutine**, and nothing suspending runs under the
   lock.
4. **The dirty bit is removed entirely**, debug builds included. That is what
   makes eviction synchronous and hence (1) and (3) sound.

## Still open

**Which vectors go in the shared cache in v1.**

To answer the "what is f" -- these are the md_bench vectors, all
`gv::Vector<float>` over the same bin-major atom layout:

| vector | what it holds | access |
|---|---|---|
| `x` | atom positions | read by force + build through 9-row stencils; **shared, degree 8.22** |
| `v` | velocities | read/written by the integrator, own rows only |
| `f` | forces | written by the force pass, own rows only |
| `x2`, `v2` | resort ping-pong destinations | written by the gather, own rows only |
| `list` | neighbour list | streamed a row at a time; **degree 1.39, effectively private** |

So `f` is per-atom forces. It, `v`, `x2` and `v2` are all
written-to-own-rows; only `x` is genuinely read-shared, and only the `list` is
genuinely private.

Written vectors are admissible: writers are byte-disjoint, `Flush` is ranged
and byte-exact, and publishing at the write site is already how md_bench is
written. Nothing about sharing changes that.

The only reason to stage it is blast radius while the new set machinery is
being proven, not doubt about the write path. `x` alone captures the effect
that motivates the whole design (100% shared, degree 8.22) and exercises
lookup, allocation and eviction under real contention without touching flush;
adding `v`, `f`, `x2`, `v2` afterwards is then a change of scope, not of
mechanism. Doing all five at once is defensible too -- it is one extra
dimension in the A/B rather than a new failure mode.
