---
name: gpu-vector
description: Use when writing, porting, optimizing, debugging, or evaluating GPU kernels that use CLIO's paged out-of-core vector (gpu_vector, gv::Vector / gv::DeviceVector) -- choosing a data layout and page size, sizing the GPU page cache, placing fetches/holds/flushes, making data survive eviction, overlapping I/O with compute, going multi-node with generations, or designing a tier-placement policy. Covers the current device API (Fetch/HoldPage/UnpinRange/Flush, coroutine forms, relaunch driver), general kernel and layout design rules, the measured cost model, correctness traps, and validation. Triggers on "gpu_vector", "gv::Vector", "DeviceVector", "paged vector", "CoFetch", "HoldPage", "out-of-core kernel", "CLIO_COROC_RUN", "data organizer", "ReorganizeHint", "tiering policy".
---

# Programming the gpu_vector paged vector

A `gv::Vector<T>` is one logical array whose pages live in CLIO's tiered store
(GPU memory, host DRAM, NVMe, object stores, parallel filesystems, across
nodes) and are cached on each GPU in a shared, set-associative page cache.
Kernels read and write it through block-collective coroutine verbs: a block
that has to wait for data **exits the kernel** and the host relaunches it when
the data lands.

This skill is general. It describes the API, rules that hold for any access
pattern, and measured costs. It deliberately contains no worked
implementations: design your kernel from the workload in front of you.

## Files in this skill -- read in this order

| file | read when |
|---|---|
| **API.md** | before writing any code. The *current* contract; many older docs and comments in the tree describe a deleted API. |
| **DESIGN.md** | before choosing a data layout or decomposition. The rules that decide whether a kernel is 1x or 10x its baseline. |
| **COSTS.md** | before optimizing, choosing page/cache sizes, or adding a tier policy. |
| **PITFALLS.md** | before running anything, and whenever a result looks too good, too bad, or hangs. Also: validation and debugging. |
| ORGANIZER.md | only when designing a tier-placement policy (data organizer, phase hints, prefetch). |

## The mental model in eight lines

1. `Fetch` names the element ranges a block is about to touch, **pins** those pages, and brings them in. It is the only verb that moves data in.
2. `HoldPage` never faults. It resolves an offset to a pointer into a pinned, resident frame, and the result never crosses a page (`run()` elements).
3. `UnpinRange` gives the pin back. Every fetched page needs exactly one.
4. `Flush` writes back exactly the byte ranges you name. **Nothing is ever written back for you**: eviction does no I/O, so an unpinned unflushed page can silently lose its writes.
5. A fault costs ~110 us of round trip for ~6 us of copy. **Batching and not faulting are the levers, not bandwidth.**
6. Every wait is a kernel exit plus relaunch (~1.3-1.6 ms per round). Keep waits out of heavy kernels.
7. Paging code is expensive in registers (hold path ~+120 regs). Keep coroutine bodies thin; put compute in `noinline` plain functions over raw pointers.
8. Distributed coherence is by **generation**: a consumer demands the generation it needs on a peer's page; the generation is the barrier.

## Workflow for a new kernel or benchmark

1. **Write the baseline first** (a plain GPU kernel over a device array). You will compare against it at *matched geometry* (same blocks x threads), and its compute kernel is often reusable unchanged via the two-phase pattern (DESIGN.md).
2. **Characterize the access pattern**: order (sequential, strided, data-dependent), reuse distance, read/write mix, and phase boundaries. Everything below follows from it.
3. **Choose the layout so a page is a work unit** and every page has exactly one writer (DESIGN.md sec 1). Decide what is paged (the big arrays) and what stays resident (indices, tables, small hot shared state).
4. **Compute each block's working set from geometry** -- which pages it fetches per step -- and derive the cache floor from it. Refuse configurations below the floor loudly on the host.
5. **Write the kernel** in the coroutine form for your target (API.md sec 3): fetch the working set in as few calls as possible (<= 8 ranges, <= 64 pages per call), hold once per page or chunk, compute through a raw pointer, flush written ranges before unpinning.
6. **Validate resident first** (cache >= data): zero faults and zero evictions, answer equal to the baseline. Then **out of core** (cache well below data): evictions > 0 and the *same* answer. PITFALLS.md sec 3.
7. **Then optimize -- do not stop at the first working draft.** A first draft that is correct is typically several times slower than it needs to be, resident and out of core alike. Measure it against the baseline at matched geometry, then work down COSTS.md's list in order: fault count and batching, rounds and waits, registers, hold amortization, publish volume. Re-run the step-6 gates after every change; stop when the remaining gap is explained by a cost you cannot remove, and say which one.
8. **Only then** consider tier placement (ORGANIZER.md) -- and expect it to lose unless COSTS.md's fill-cost rule says otherwise.

## Non-negotiable rules

- Every verb is **block-collective**: all threads call it, never under a thread-dependent branch; vote any branch that reaches a barrier with `__syncthreads_or`. Stores may be predicated; suspensions may not.
- `Init(yv.Block())`, never `blockIdx.x` -- the driver compacts the grid between rounds.
- Fetch before hold; hold only bytes the fetch named (use `PageLo`/`PageSpan` to fetch whole pages).
- Flush written ranges, then `UnpinRange`. Never flush bytes you did not write.
- Data generated on the GPU exists only in the cache until flushed: flush it before it can be evicted, or it is gone.
- Plain `__shared__` and registers do not survive a park (C++20 coroutine locals do; coroc locals must be trivially copyable). Use `CLIO_SHARED_PERSIST` or re-stage after every suspend.
- Check `HitRoundCap()` after every run: hitting the round cap is a livelock that otherwise looks like a pass.
- A run with zero evictions proves nothing about out-of-core behavior.
- Do not trust comments or docs that mention `FetchPagesBatched`, `WaitFetch`, `FlushBlockBatched`, `FlushAsync`/`AwaitFlush`, dirty bits, "the hold is the pin", private per-block tables, or a resident fast path: that API is gone (API.md, "Stale").
