# CTE Shared-Memory Metadata Cache (issue #783)

Status: **IMPLEMENTED** (phases 0-7 complete)
Branch: `783-cte-shm-metadata-cache`

## Scaling: how many blobs and tags fit?

**The maps are FIXED CAPACITY and never rehash** (see §5.3b — a rehash would
free a table out from under untracked cross-process readers). Capacity is set
once at pool creation and is the permanent ceiling. Defaults: 64K tags, 256K
blobs per CTE pool, overridable via `CLIO_CTE_SHM_TAG_CAPACITY` /
`CLIO_CTE_SHM_BLOB_CAPACITY`.

Useful occupancy is **7/8 of capacity** (load cap, below), so the defaults hold
~229K blobs. **A million blobs does NOT fit the default** — it gets ~23%
coverage, and the other 77% silently use the RPC path.

Memory is **resident, not sparse**: every slot is constructed at creation.

| Slot | Size | 1M blobs needs | Table size |
|---|---|---|---|
| blob | 376 B | 2,097,152 slots | **0.79 GB** |
| tag (name→id) | 80 B | — | — |
| tag (id→info) | 56 B | — | — |

So 1M blobs is reachable (0.79 GB inside a 6 GB segment) but must be configured
deliberately; 10M would need 6.3 GB and does not fit today.

### The load cap exists because a full table was slower than not caching

Measured on a 65536-slot table, before the cap:

| load | hit | **miss** | insert |
|---|---|---|---|
| 50% | 0.042 µs | 0.056 µs | 0.15 µs |
| 90% | 0.037 µs | 0.184 µs | 0.22 µs |
| 99% | 0.071 µs | **11.8 µs** | 0.85 µs |
| 100% | 0.046 µs | **108.9 µs** | 1.71 µs |

A probe run only stops at an EMPTY slot, so a table with none makes every miss
scan all of it. At 100% a miss cost **more than the ~90 µs RPC the cache exists
to avoid** — and the caller still paid that RPC afterwards. A saturated cache
was strictly worse than no cache.

Fix: refuse NEW keys past 7/8 load (overwrites still accepted, so a saturated
cache stays fresh rather than going stale), plus a 1024-probe backstop. Miss
cost is now **0.09–0.24 µs at every fill level**.

A cautionary detail: the first probe cap I tried (64) became the *binding*
constraint instead of the load factor — linear-probing clusters run far longer
than the average, so a 65536-slot table stopped accepting keys at ~51%
occupancy and 1M coverage *fell* from 26% to 18%. The backstop must be loose
enough that the load factor is what binds.

### Still open

Overflow is **silent**: blobs beyond capacity simply are not cached and quietly
take the RPC path. That is the same visibility gap as #787 — the correct
behaviour is invisible, so a badly-sized deployment looks identical to a
well-sized one. Exposing cached/refused counters is the follow-up.

## Result

### Write-then-read cycle (the pattern callers actually perform)

The read benchmarks below are **steady-state**: they time reads with no
interleaved writes. That measures the read in isolation. For a caller that
writes then reads back, the cycle is **write-dominated**, because writes are
NOT accelerated by this work:

| Measurement | SHM | RPC | Ratio |
|---|---|---|---|
| write alone | 79-140 µs | (same) | — |
| **write + metadata read** | **81-114 µs** | 162-195 µs | **~2x** |
| **write + payload read** | **83-128 µs** | 187-278 µs | **~2x** |

So the honest headline for write-then-read is **~2x**, not the several-hundred-x
figure the read-only benchmarks show. The read component goes from ~90-140 µs to
~0.2 µs — effectively free — but Amdahl caps the cycle at the cost of the write.
The read-only numbers are the right measure for read-mostly workloads (repeated
`GetBlobSize`/`GetBlob` on cached blobs); the cycle numbers are the right measure
for write-then-read.

Note the run-to-run spread: the RPC baseline swings 90-280 µs on this host while
the SHM path stays at ~0.2 µs, so quote ranges, not single figures.


Client reads of node-local CTE data now complete with **zero IPC**. Measured
cross-process (external daemon), same query and same blobs on both paths:

| Read type | SHM fast path | RPC path | Speedup |
|---|---|---|---|
| **Payload (4 KB blob)** | **0.26 µs/op** | **120.2 µs/op** | **461x** |
| **Metadata** | **0.15 µs/op** | **90.0 µs/op** | **590x** |

Design target was < 5 µs; both paths land at 0.15-0.26 µs. The starting point
recorded in §1 was a 72 µs transport floor that got worse under concurrency.

Wired into the real APIs: `Tag::GetBlobSize` and `Tag::GetBlob` both take the
fast path when the blob qualifies and fall through to RPC otherwise, so callers
get the speedup without changing a line.

Test coverage: ctp primitives 43 + 6191 + 28 assertions; CTE SHM data model
7/7; CTE functional 12/12 (incl. write-then-read and payload correctness);
bdev chimod 21/21.

## 1. Problem

Blobs are arbitrary-sized, so the fixed-size page-cache trick does not apply. Every
metadata lookup is a runtime round-trip, and the round-trip is expensive no matter
which transport we pick. Measured on a 6-core WSL host (`clio_run_thrpt_bench
--test-case latency`, 4 reps, medians):

| Transport | 1 thread | 4 threads | 8 threads |
|---|---|---|---|
| SHM | 70.6 µs | 145.3 µs | 240.5 µs |
| IPC (unix socket) | 124.5 µs | 175.5 µs | 297.1 µs |
| TCP (loopback) | 471.7 µs | 574.8 µs | 691.5 µs |

The floor is ~71 µs and it degrades with concurrency. No hardware extension helps.
The only way to make small reads cheap is to remove the round-trip entirely.

**Target: node-local RAM-resident `GetBlob` in < 5 µs with zero IPC.**

## 2. Model

- The **runtime owns** all shared metadata. It is the only writer.
- **Clients are strictly read-only**, mapping the segment `PROT_READ`.
- **Reads are synchronous** (in-process SHM lookup). **Writes are fully asynchronous**
  (write-through to the runtime; the SHM copy is updated by the runtime afterward).
- Clients are **untrusted for liveness**: any client may be `SIGKILL`ed at any
  instant, including mid-read. Nothing the runtime does may ever block on a client.

That last point is the load-bearing constraint. It rules out the obvious designs.

## 3. What already exists (and what does not)

Useful groundwork already in the tree:

- `ctp::ipc::vector<AllocT>` and `ShmContainer<AllocT>` —
  `context-transport-primitives/include/clio_ctp/data_structures/ipc/`
- A segment/allocator pattern: `backend.shm_init(alloc_id, size, name)` →
  `MakeAlloc<T>()` on the runtime, `AttachAlloc<T>()` on the client
  (`context-runtime/src/ipc_manager.cc:839-945`)
- `MemorySegment` enum with `kMainSegment` / `kClientDataSegment` / `kQueueSegment`
  (`context-runtime/include/clio_runtime/types.h:596`)
- `TagInfo`/`BlobInfo` are **already allocator-parameterized** (`priv::string`,
  `priv::vector`, `CLIO_PRIV_ALLOC`) — they are not hard-wired to `new`/`delete`.

Missing, and must be built:

- **`ipc::string`** — does not exist.
- **`ipc::unordered_map`** — does not exist. Only `priv::unordered_map_ll`
  (malloc-backed, lock-taking) and `priv::unordered_map_lhash` exist.
- Any read-only / lock-free lookup path.
- SHM-backed RAM bdev. Today `MemBdevTransport` uses `new char[1 GiB]` pages
  (`mem_bdev_transport.cc:80`).

Note `CLIO_PRIV_ALLOC` expands to `CTP_MALLOC` on host (`types.h:551`) — the `priv::*`
containers are allocator-aware but currently bound to malloc. Retargeting them is
plausible; it is *not* free, because the map values are `std::shared_ptr`.

## 4. Architecture

### 4.1 Metadata segment

New `kMetadataSegment` in the `MemorySegment` enum. 8 GB fixed size, created at
context-runtime startup, **runtime-wide** (not CTE-owned — other modules will want
it). Not pre-faulted: `shm_open` + `ftruncate(8 GB)` + `mmap` is lazily populated, so
only touched pages cost RAM. `IpcManager` creates it on the server path and attaches
it read-only on the client path, exposing it next to `main_allocator_`.

### 4.2 Containers

- `ipc::string` — offset-based, SSO optional, no heap pointers.
- `ipc::unordered_map` — open addressing with tombstones, power-of-two capacity,
  per-slot generation counter. Chosen over chaining because a lock-free reader can
  probe a flat array safely; chasing chain pointers while the writer recycles nodes
  needs epoch protection on *every* node.
- Reuse `ipc::vector` for `blocks_` and `aliases_`.

All internal references are **segment-relative offsets**, never raw pointers — the
segment maps at different base addresses in each process.

### 4.3 Data structures

`ShmTagInfo` / `ShmBlobInfo` mirror the existing structs with SHM-safe members.
Notable conversions:

| Current | Becomes | Note |
|---|---|---|
| `priv::string blob_name_` | `ipc::string` | |
| `priv::vector<BlobBlock> blocks_` | `ipc::vector<BlobBlock>` | verify `BlobBlock` is POD-safe; it embeds a `bdev::Client` |
| `ctp::Mutex prealloc_lock_` | runtime-side only | clients never take it; must not be on the RO read path |
| `std::shared_ptr<T>` in maps | offset + generation | see §5.1 — this is the hard part |

### 4.4 Client read path

```
GetBlob(tag, name)
  -> seqlock-read blob slot from SHM map
  -> all blocks node-local && kRam?
       yes -> memcpy from SHM RAM bdev, re-validate generation, return
       no  -> fall back to existing async RPC path
```

Fallback is always available, which is what makes incremental rollout safe.

## 5. The hard problems

### 5.1 Reclamation without `shared_ptr`

`core_runtime.h:366` documents the current safety property explicitly: *"Values are
`std::shared_ptr`, so a concurrent erase just drops the map's reference while any
in-flight handle keeps the object alive — no use-after-free."* There are 61
`shared_ptr<TagInfo>`/`shared_ptr<BlobInfo>` sites. Shared memory has no `shared_ptr`,
and the readers are in other processes.

Proposal: **epoch-based reclamation.**
- Global epoch counter in SHM, bumped by the runtime.
- Each attached client publishes a per-client epoch slot (entering/leaving a read).
- The runtime defers freeing a slot until every live client has advanced past the
  epoch in which it was retired.
- **Liveness:** a client that dies inside a read section would pin the epoch forever.
  Each slot carries a PID + heartbeat; the runtime reaps slots whose owner is gone.
  Reaping must be conservative (a false reap is a use-after-free).

This is the single riskiest piece of the design and deserves its own standalone
test harness before it is wired into CTE.

### 5.2 Optimistic reads only

Because the runtime may never wait on a client, **no client-held lock may exist**.
All reads are seqlock-style: read sequence, read payload, re-read sequence, retry on
mismatch or odd value.

Direct consequence: **clients cannot call `unordered_map_ll::find()`** — it acquires
locks, i.e. it writes, which both blocks the runtime and faults a `PROT_READ` mapping.
The runtime's writer path and the client's reader path are therefore *different code*
over the *same* layout. That asymmetry needs to be explicit in the API, not implied.

### 5.3 Coherence against the DataOrganizer

The frecency organizer (`data_organizer/frecency_organizer.cc`) moves and evicts
blobs. Sequence that corrupts data:

1. Client reads `BlobInfo`, gets block offsets into the RAM bdev.
2. Runtime reorganizes the blob; blocks are freed and reused by another blob.
3. Client `memcpy`s from those offsets → **silently returns another blob's bytes.**

No error surfaces.

**Scope note:** locking only the *metadata copy-out* does not fix this. The hazard is
the **payload** read. Whatever mechanism we use must span the `memcpy` from the RAM
bdev, not just the `BlobInfo` field read.

#### Chosen design: generation for correctness, lease for progress

A per-blob **timed lease** (proposed: 500 ms) that the reorganizer respects is the
right shape, but it cannot be the *correctness* mechanism, for four reasons found in
the existing code:

1. **A lock requires the client to write.** Clients map `PROT_READ` (§5.5). Acquiring
   any mutex is a store. This forces the lock words into a **separate small
   read-write mapped region**, with the payload staying read-only. Consequence: a
   buggy or hostile client can corrupt lock state, and corrupting it to *unlocked*
   breaks safety rather than merely liveness. So the lock cannot be load-bearing.
2. **`ctp::Mutex` is a ticket lock and cannot be reclaimed.**
   `thread/lock/mutex.h:47` — `Lock()` does `lock_.fetch_add(1)` and spins until
   `head_ == tkt`. If a client dies holding it, `head_` never advances and *every*
   later waiter blocks forever. Forcibly advancing `head_` releases multiple waiters
   at once, destroying mutual exclusion. A reclaimable lease needs a different
   primitive: a CAS word carrying **owner PID + acquire timestamp**.
3. **The reorganizer is a coroutine on a runtime worker.**
   `ReorganizeBlobInternal` / `DynamicReorganize` return `TaskResume`, and
   `core_tasks.h:820-831` already warns that a thread-blocking lock "CANNOT be used
   here — it would deadlock the single worker the instant the holder suspends at a
   `co_await`." A 500 ms blocking wait would stall a whole worker; with 4 workers, a
   few stuck blobs wedge the runtime. **The reorganizer must `try_lock` and skip** —
   it is background work and is free to reorganize a different blob.
4. **Timeout-and-steal is probabilistic, not correct.** If the timeout fires on a
   *live but slow* client — page fault under memory pressure (§5.7),
   descheduled, swapped — the reorganizer moves data under an active reader, which is
   exactly the corruption being prevented. 500 ms makes that rare; rare **plus
   silent** is the worst failure mode to ship.

Therefore:

- **Correctness** = a per-blob **generation counter**, validated before *and* after
  the payload copy. Requires no client writes, costs nothing on the read path, and
  stays correct even if the lease is stolen, corrupted, or skipped.
- **The lease** = a *progress* optimization, so readers do not livelock retrying while
  the organizer churns a hot blob.
- **Neither side ever blocks on the other.** Client: `try_lock` with a tiny budget,
  else fall back to RPC. Runtime: `try_lock`, else skip this blob this pass.
- **500 ms becomes the stale-lease janitor threshold, not a wait.** If a lease has
  been held longer than that, check the owner PID for liveness and reclaim only if it
  is gone — never as a blind timeout.

**Size cap.** The motivation is *small* I/O; for a 1 GB blob a 71 µs round-trip is
already noise. Restricting the fast path to blobs under ~1 MB bounds a client's
legitimate lease hold to well under 100 µs, which lets the janitor threshold be far
tighter than 500 ms and shrinks the damage a dead client can do. 500 ms is ~100,000x
the 5 µs target — it is sized for large-blob copies that should not use this path at
all. **Exact cap is TBD from measurement (open question 6).**

**Lock ordering:** `TagInfo` before `BlobInfo`, enforced identically on both sides.

### 5.3b `TimedMutex` and `Lease` (replaces `shared_ptr`)

Decision: a reclaimable `TimedMutex` (`clio_ctp/thread/lock/timed_mutex.h`, beside
`Mutex`) plus a SHM smart pointer `Lease` (`clio_ctp/memory/smart_ptr/lease.h`, beside
`shared_ptr`/`unique_ptr`). A `Lease` acquires the target's `TimedMutex` on
construction and releases on destruction; because the lock is reclaimable, a client
that dies holding one is eventually serviced. Clients **copy metadata out and release
immediately** rather than holding a lease across work.

This replaces epochs/hazard pointers for the **lifetime** problem. It does *not*
replace the generation counter, which remains the **coherence** mechanism (§5.3).
They are complementary: the lease stops the object being freed under a reader; the
generation catches the payload being moved under a reader.

Four constraints on the implementation:

**1. Mapping is `PROT_READ|PROT_WRITE`; read-only is enforced by discipline.**
DECIDED: clients map the segment read-write, because acquiring a lease is a store and
splitting lock words into a separate region is not worth the layout complexity. **The
only bytes a client may write are lock words. Metadata is never client-writable.**

This trades a hardware guarantee for a convention, so two invariants become
load-bearing:
- The runtime **must never read the SHM cache as authoritative**. It keeps its own
  structures; the cache is derived state, and a client that scribbles on it can only
  harm other clients, never the runtime. (This is why Phase 7 keeps it a pure cache.)
- A corrupted cache must be **droppable and rebuildable** wholesale, since we can no
  longer rely on `PROT_READ` to prevent corruption from happening at all.

Acceptable because §7.1 already accepts a single-tenant trust model. Under
multi-tenancy this decision would have to be revisited along with §5.6.

**2. Default to a lock-free read; take a lease only when you need a stable view.**
An exclusive `TimedMutex` serializes concurrent readers of the same hot blob, and the
CAS ping-pongs that cache line across every reading core — the opposite of what a
read-mostly cache wants. So the common path is a **seqlock read that performs zero
stores** (read generation, copy, re-read generation, retry), and `Lease` is reserved
for operations needing a stable view for longer than a retry loop tolerates. Keeping
leases *rare* is also what keeps reclamation simple: an exclusive lock has a single
owner, so a dead holder is identifiable, which a shared/reader-counted lease would
not be (that would need per-holder tracking, i.e. epochs again by another name).

**What if a client dies mid-seqlock-read?** Nothing happens, and this is the whole
reason to prefer a seqlock for the default path. A seqlock *reader* acquires nothing
and stores nothing — it reads the sequence, copies, and re-reads the sequence. A dead
reader therefore leaves **no state to release, reclaim, or unwind**; there is no
counter to decrement and no lock to steal. It has copied garbage into its own address
space, but it is dead, so nobody consumes it. No cross-process effect whatsoever.

The seqlock's one death hazard is on the **writer** side: a writer that dies between
its two sequence bumps leaves the sequence odd forever, and readers retry forever. But
the writer is always the **runtime** — the single trusted owner — and if the runtime
dies the whole session is being torn down anyway. The asymmetry is exactly the one we
want: the untrusted-liveness side (clients) is stateless, and the stateful side is the
one process whose death is already fatal.

This is precisely why the seqlock is the *default* and `Lease` is the *exception*: a
lease is the only client-side construct whose abandonment needs servicing, so the
design minimizes how often one is held.

**3. Reclamation needs PID + process start time, not a bare timeout.** A timeout alone
cannot distinguish a dead holder from a live-but-slow one (§5.3, point 4). The lock
word carries `{owner_pid, process_start_time, acquire_timestamp}`; a waiter past the
threshold checks liveness and steals **only if the owner is genuinely gone**. Start
time is required because PIDs are recycled — on Linux, field 22 of `/proc/<pid>/stat`.
A steal must bump a steal counter so a resurrected holder can detect it lost the lock.

**4. The runtime may wait on a lease — but the *task* waits, never the *worker
thread*.** DECIDED: waiting is unavoidable, since a lease that the runtime could
ignore would not protect anything. The constraint is not *whether* to wait but *how*.

**Thread-blocking waits are ACCEPTED for the initial implementation**, on the strength
of the issue #781 work-orchestrator mechanisms (elastic pool + stall detection + worker
spawn/steal), which detect a wedged worker and spawn a replacement. This branch merges
`origin/dev` at PR #782 specifically so those mechanisms are present — they landed in
dev *after* this branch was cut, and without them a blocking lease wait would reproduce
the exact deadlock class #781 exists to prevent.

Historical constraint, retained because it still applies to any path that must not
depend on stall detection: `core_tasks.h:820-831` documents that a thread-blocking lock
"CANNOT be used here — it would deadlock the single worker the instant the holder
suspends at a `co_await`," and demonstrates the alternative — the `write_owner_`
contender "busy-polls via `co_await yield()` ... lost-wakeup-proof because it re-checks
every worker iteration."

So, in order of preference:

- The reorganizer, being pure background work, still prefers **`try_lock`-and-skip** —
  it has no obligation to reorganize any particular blob right now, so it should never
  wait at all.
- Metadata writes may **block the worker**, bounded by the `TimedMutex` threshold and
  backstopped by #781 stall detection.
- If stall detection proves insufficient under load, the `co_await` retry loop above is
  the fallback — same semantics, no worker occupancy. Worth revisiting if Phase 5/6
  benchmarks show worker starvation.

Two consequences to keep in view:
- The `TimedMutex` timeout now **bounds runtime write latency**, not just reclamation
  lag. A 500 ms threshold means a metadata write can stall that long behind a wedged
  client. This is the strongest argument for the §5.3 size cap: bounding legitimate
  lease hold time lets the threshold come down proportionally.
- A client holding a lease can now delay runtime progress, which is a denial-of-service
  vector even under a single-tenant model — a merely *buggy* client that leaks a lease
  is enough. The timeout is the backstop, and clients must never hold a lease across a
  syscall, an I/O, or anything else unbounded.

### 5.4 Read-your-own-writes

Fully-async write-through means a client that does `PutBlob` then `GetBlob` can miss
its own write — which breaks the filesystem adapter's expectations.

Mitigation: a client-local `std::unordered_map` of **pending writes**, consulted
before every fast-path read; a hit forces the slow path until the write is
acknowledged. This is client-local plain heap — no SHM, no cross-process concern.

Two requirements that are easy to miss:
- It must track **deletes, truncates and renames**, not just puts. Otherwise a client
  reads a blob it just deleted straight out of the stale SHM cache.
- It must be consulted by **metadata-only** fast paths (`GetBlobSize`, `GetBlobScore`)
  too, not only payload reads — a stale size is just as wrong as stale bytes.

### 5.5 `PROT_READ` discipline

A stray write on the client read path is a `SIGSEGV`, not a wrong answer. No lock
acquisition, no lazy init, no refcount updates, no `operator[]` (which inserts).
Worth enforcing in CI with a test that maps the segment read-only and exercises every
client path.

### 5.6 Trust boundary — **needs an explicit decision**

A shared metadata segment lets *any* client read *every* tag and blob name/size on
the node. Today a client sees only what the runtime returns. This is a genuine change
to the isolation model. Options: accept it (single-tenant assumption), or partition
segments per tenant/pool. This should be decided deliberately and recorded, not
inherited by accident.

### 5.7 Segment sizing — it is RAM, not `/dev/shm`

**Corrected during Phase 1 implementation.** The original text here assumed these
segments are `/dev/shm` files and that this host's 64 MB `/dev/shm` was the binding
constraint. That is wrong.

On Linux the runtime creates segments with **`memfd_create()`**
(`SystemInfo::CreateNewSharedMemory`, `system_info.cc:519`), then publishes a symlink
under a per-user directory (`/tmp/clio_<user>/`) pointing at `/proc/<pid>/fd/<n>` so
other processes can find and open them. memfd objects live in the kernel's internal
shmem pool — bounded by **RAM + swap and the memory cgroup** — and are *not* subject
to the `/dev/shm` mount size at all. Verified: `/proc/<pid>/maps` shows
`/memfd:chi_metadata_segment_...`, and `/dev/shm` stays 0% used with a 1 GB main
segment live.

This also explains an older observation that a 1 GB main segment and a 128 MB ring
"work anyway" on a host with a 64 MB `/dev/shm` — not sparseness, just the wrong
filesystem.

Consequences:
- **Do not clamp against `std::filesystem::space("/dev/shm")`.** It measures an
  unrelated filesystem and needlessly shrinks the cache (an early Phase 1 draft cut
  8 GB to 57 MB this way).
- The reservation is close to free — sparse and never pre-faulted, so cost is only
  pages actually written. Verified: with a ~6 GB metadata segment mapped, runtime RSS
  is ~35 MB.
- The real limit is the **live set** against RAM/cgroup. Exhausting shmem surfaces as
  `SIGBUS` or the OOM killer rather than a clean allocation failure, so Phase 1 clamps
  the reservation to **half of physical RAM** as a sanity bound.
- A "cache full → fall back to RPC" path is still required, since allocator exhaustion
  inside the segment must degrade rather than fail.

## 6. Phasing

Dual-write is what keeps a 127-call-site refactor from becoming a flag-day.

| Phase | Work | Exit criterion |
|---|---|---|
| 0 | CTE metadata benchmark harness | DONE -- baseline SHM 72.1 µs (see §6b) |
| 1 | `kMetadataSegment` + `IpcManager` plumbing | Runtime creates it; client attaches RO |
| 2 | `ipc::string`, `ipc::unordered_map`, `TimedMutex`, `Lease` | Standalone torture test, incl. random reader kills mid-lease |
| 3 | `ShmTagInfo`/`ShmBlobInfo`; runtime **dual-writes** | Old path still authoritative; all CTE tests green |
| 4 | Propagate map locations via `CreateTask` | Client holds valid handles |
| 5 | Client fast path for **metadata-only** ops (`GetBlobSize`, `GetBlobScore`) | Correctness parity; latency win measured |
| 6 | SHM RAM bdev + client payload fast path | `GetBlob` < 5 µs, zero IPC |
| 7 | Retire the *scaffolding*, keep SHM as a pure cache | Steady state (see below) |

**Phase 7 revised.** The original plan was to retire the legacy structures and make SHM
the single source of truth. That is now rejected, primarily because of §5.3b constraint
1: clients map the segment **read-write**, so a buggy client can corrupt it. Derived
state that can be dropped and rebuilt is a recoverable annoyance; *authoritative* state
that a client can corrupt is data loss. The runtime therefore keeps its own structures
permanently, and the SHM segment stays a **pure read cache** it may defer updating or
invalidate wholesale at any time.

(A secondary argument also applies: an authoritative cache would put client-held leases
on the critical path of every metadata write, rather than merely on cache freshness.
The runtime is willing to wait on leases — §5.3b constraint 4 — but there is no reason
to make correctness, rather than staleness, depend on that wait.)

Phase 7 therefore only removes benchmarking scaffolding and redundant bookkeeping, not
the legacy path.

Phases 0-2 are self-contained and carry essentially no risk to existing behavior.
Phase 3 is where the blast radius starts. Phase 5 before Phase 6 deliberately: proving
the read path on metadata-only ops is far cheaper to debug than on payload reads.

## 6b. Implementation status

**Phase 1 (metadata segment) — done.**
- `kMetadataSegment` added to `MemorySegment` (`types.h`).
- `ConfigManager`: `metadata_segment_name_`, `metadata_segment_size_`,
  `CalculateMetadataSegmentSize()`, plus the name switch case.
- `IpcManager`: `metadata_backend_` / `metadata_allocator_id_` /
  `metadata_allocator_`, `GetMetadataAllocator()`, `HasMetadataSegment()`.
- Server creates it at `pid.3` (allocator index reservation bumped 3 -> 4);
  client attaches best-effort. **Both paths are non-fatal**: if the segment is
  missing the runtime still boots and clients silently use the RPC path.
- Reservation sanity-clamped to half of RAM (see §5.7).
- Verified live: segment mapped as `/memfd:chi_metadata_segment_<user>_<port>`,
  runtime RSS ~35 MB with ~6 GB reserved (confirming it is not pre-faulted), and a
  client logs a successful attach.

**Phase 2 (primitives) — 3 of 4 done.**
- `thread/lock/timed_mutex.h`: unfair CAS lock with PID + process-start-time
  reclamation, ABA-safe release, `ScopedTimedLock`, `steal_count_`. Verified by
  fork/SIGKILL: a killed holder is reclaimed at the threshold; a live-but-slow
  holder is never stolen from.
- `data_structures/ipc/string.h`: offset-based, SSO (23 bytes inline), one-way
  growth, deleted implicit copy, FNV-1a `hash()` (NOT `std::hash`, which is not
  stable across processes), `EqualsBytes` for allocation-free probing.
  10 cases / 43 assertions.
- `data_structures/ipc/unordered_map.h`: open addressing, **fixed capacity**,
  per-slot seqlock, tombstones, `TryGetBytes` so clients never allocate.
  11 cases / 6191 assertions.
- `memory/smart_ptr/lease.h`: `Leaseable` mix-in carrying BOTH mechanisms
  (`gen_` seqlock for correctness, `lease_mutex_` for progress/lifetime),
  `SeqRead()` optimistic-read helper, and move-only RAII `Lease<T>` whose
  `get()` returns nullptr unless owned, so a missed `Owns()` check fails loudly
  instead of silently reading unprotected memory. 5 cases / 28 assertions.

**Phase 2 exit criterion MET.** The torture test forks 6 seqlock readers plus 2
lease-holding readers, SIGKILLs them at random mid-read (and the lease holders
while they still hold the lease) against a writer that deliberately widens its
update window. Result over a 3s run:

| metric | value |
|---|---|
| writes | 503,400 |
| successful cross-process reads | 25,554,659 |
| **torn reads** | **0** |
| reader retries | 242,412,642 |
| leases reclaimed from dead holders | 1 |

Zero tears across 25.5M reads while readers were being killed, and the writer
never wedged behind an abandoned lease.

**Caveat to carry into Phase 5:** the retry rate here is ~90%, because this
writer holds its update window open almost continuously by design. Real writes
are rare and short, but reader retry rate under realistic write load is a
number worth measuring rather than assuming — a hot blob under sustained
rewrite could starve readers into the RPC fallback.

- Still to build: nothing in Phase 2.

**Phase 3 (SHM records) — data model done; dual-write wiring pending.**
`clio_cte/core/shm_metadata_cache.h` defines `ShmBlockDesc`, `ShmBlobRecord`,
`ShmTagRecord`, the three maps, and `ShmMetadataCacheRoot`. 7 tests pass,
including a **forked child** that attaches the segment independently and reads
back 200 entries plus a spilled long key — the property that validates the
whole offset-based design.

Two shape decisions forced by the primitives:

- **Map values are strictly POD.** The map's reader copies the value out
  between two generation reads, so a value containing an `ipc::string` (which
  is non-copyable by design) would defeat the seqlock. The blob NAME is
  therefore the map key, not a field of the record.
- **Block lists are fixed-size inline (`kMaxInlineBlocks = 8`), not vectors.**
  The fast path only serves small blobs, and a small blob has few blocks.
  Capping keeps the record POD with no nested allocation and no second pointer
  chase for a lock-free reader; a blob with more blocks is flagged
  `kShmBlobTruncated` and is never payload-read from the cache.

`ShmBlobRecord::placement_gen_` is separate from the map's slot seqlock on
purpose: the slot generation protects the METADATA copy, while §5.3's hazard is
the DataOrganizer relocating blocks during the PAYLOAD read. A client must
re-check `placement_gen_` after copying bytes.

**PHASE 6 COMPLETE — zero-IPC payload reads working and measured.**

| Read type | SHM fast path | RPC path | Speedup |
|---|---|---|---|
| **Payload (4 KB blob)** | **0.207 µs/op** | **94.3 µs/op** | **455x** |
| Metadata (cross-process) | 0.170 µs/op | 82.7 µs/op | 488x |

`BuildShmBlobRecord` sets `kShmBlobDirectReadable` only when EVERY block is
node-local and `kRam`; the flag starts true and is cleared by the first block
that fails, so the default is *refuse* and an unknown target can never be
mistaken for readable. `Client::TryReadBlobShm` attaches the bdev segment
lazily (cached per pool, negative results cached too) and validates
`placement_gen_` **before and after** the copy — the metadata seqlock protects
only the record, while the real hazard is the DataOrganizer relocating blocks
mid-`memcpy`.

**Test isolation was the hard part, not the code.** The first version
registered a RAM target on the *shared* fixture pool, where file targets had
accumulated from earlier tests and the DPE placed the blob on one. The blob was
therefore correctly not direct-readable and the payload path was never
exercised — the test warned loudly instead of passing vacuously. Giving it its
own CTE pool with a RAM target as the ONLY target is what made it a real test.

**Phase 6a — RAM bdev is shared-memory backed.**

A `kRam` device's bytes now live in a dedicated SHM segment, mapped as ONE
contiguous sparse region rather than a page table. Since these are memfd
segments an untouched reservation costs nothing, so a flat mapping is both
simpler and faster to index than the old 1 GiB page vector — `EnsureRamPage`
now allocates nothing and is pure arithmetic. The paged API is retained only so
existing call sites are unchanged.

Segment name derives purely from **(runtime pid, pool id)**, so a client can
reconstruct it from what it already has — server pid from ClientConnect, target
pool id from each cached block descriptor. No name or page table needs
publishing through a side channel.

`kPinned` stays on the private heap (`cudaMallocHost` memory cannot be
SHM-backed). Backing is best-effort: failure costs only the client fast path.
`Destroy` clears `ready_` before unmapping so a client mid-attach refuses
rather than reading a dying segment.

Verified: bdev chimod suite **21/21** with three devices reporting themselves
SHM-backed.

*Retracted (was: "unrelated pre-existing bug").* `clio_run_thrpt_bench
--test-case bdev_allocation` was segfaulting against an external daemon, and I
recorded it here as pre-existing because the file path crashed too. **That
conclusion was wrong.** After destroying `build-cpu` and rebuilding the whole
repo from scratch it does not reproduce (6/6 clean runs, both bdev types).

Root cause was **ABI skew from my own partial rebuilds**: this work changed the
layout of `IpcManager` and added an OUT field to `ClientConnectTask`, but
`clio_run_thrpt_bench` was not rebuilt, so the bench carried the old handshake
layout while the daemon used the new one. The client faulted on the first task
after `ClientConnect` — exactly where the crash appeared — and the daemon was
unaffected because only the client's view was stale.

Filed as #786 and closed as invalid. The lesson is the one already recorded in
Phase 0: **a crash that appears after a layout change is a stale-build suspect
first.** Rebuild the runtime, every module `.so`, and any test/bench binaries
together — or do a clean build before concluding anything about the cause.

**Phase 5 COMPLETE — fast path wired and MEASURED.**

`Tag::GetBlobSize` now answers from the SHM cache when the blob is cached and
falls through to RPC on any miss or inconsistent read. Measured on this host,
same query, same blobs, same process:

| Configuration | SHM fast path | RPC path | Speedup |
|---|---|---|---|
| **External daemon (cross-process — the real target)** | **0.170 µs/op** | **82.7 µs/op** | **488x** |
| In-process runtime (`CLIO_INIT` with runtime) | 0.075 µs/op | 5.57 µs/op | 74x |

**Design target was < 5 µs; achieved 0.17 µs cross-process — ~29x better than
target.** The 82.7 µs RPC figure lines up with the §1 baseline (72 µs transport
floor plus CTE handler work), which is the sanity check that the benchmark is
measuring the right thing.

Note the two rows measure genuinely different baselines: with an in-process
runtime there is no cross-process round-trip at all, so its 5.6 µs is not the
number this design set out to beat. Quote the cross-process row.

The benchmark asserts correctness alongside speed — both paths must return the
same size, and every fast-path lookup must hit — so a regression that turned
the fast path into a miss-storm would fail rather than silently look fast.
CTE suite: **11/11** in both configurations.

**Phase 4 COMPLETE — client attach + write-then-read tests passing.**
Two bugs surfaced only by writing the test:

1. **CreateTask propagation alone does not work.** The OUT offset is written
   only by the process that actually causes pool creation, and compose creates
   the CTE pool at startup — so every real client called GetOrCreatePool, hit
   the existing pool, never ran the ChiMod's `Create`, and got offset 0.
   Discovery must not depend on being the first caller. Added a
   `MetadataDirectory` as the first object in the metadata segment, published
   to every client in the **ClientConnect handshake** beside the allocator ids.
   The CreateTask path is kept and tried first, then falls back to the
   directory.
2. **The blob mirror was published too early.** `BlobInfo` is inserted into the
   map empty and only gains blocks/`total_size_cache_` later in `ExtendBlob`,
   so mirroring at insert time published size 0 — a client reading its own
   write got the wrong answer. The mirror now runs at the successful end of
   `PutBlobImpl`.

The write-then-read test asserts what makes the cache *safe*, not merely fast:
a miss is reported as not-cached and never as absent; the cached size AGREES
with the authoritative RPC answer; an overwrite is reflected rather than served
stale; payload reads stay refused while blobs live on file targets. It also
fails loudly rather than passing vacuously when the cache is unavailable — the
first version silently skipped every assertion and "passed".

CTE functional suite: **10/10**.

**Phase 3 COMPLETE — dual-write wired and verified end-to-end.**
`ShmMetadataCache` (runtime-side owner) is created in `Runtime::Create` and
mirrored at every blob insert (3 sites: PutBlob, WAL replay, metadata restore),
every blob erase (3 sites), and tag creation. Every mirror call is best-effort
and swallows failure: the cache is derived state, so the right response to any
problem is to stop caching, never to reject the metadata write the caller was
actually performing. Mirrors run AFTER the authoritative insert, so the cache
can only lag, never lead.

Verified against a live daemon by attaching the metadata segment from an
UNRELATED process and reading the maps:

```
ready=1 version=1
tag_name_to_id  : size=11  cap=65536
tag_id_to_info  : size=11  cap=65536
blob_key_to_info: size=87  cap=262144
```

CTE functional suite stays 9/9 with the mirror active.

**Method note worth keeping.** The first "green" run proved nothing: the suite
that passed (`cte_core_functionality_simple_tests`) never creates a CTE pool, so
`Runtime::Create` — and therefore the whole dual-write path — never executed.
Confirming the cache was actually *enabled* (`root_off=68072` in the log) and
then reading the populated maps from another process is what made the result
mean something. A green suite that does not reach the new code is not evidence.

`kShmBlobDirectReadable` is deliberately NOT set yet: proving a block is
node-local AND RAM-backed needs the target registry, and the RAM bdev is not
SHM-backed until Phase 6. Until then the cache serves METADATA only and every
payload read still goes through RPC — the safe direction to be wrong in.

**Design refinement made during implementation: the map never rehashes.** Growth
is the one operation that cannot be made safe for untracked cross-process
readers — a rehash must free the old table, and knowing when the last reader has
left it requires exactly the epoch machinery §5.3b rejected. So the table is
sized up front and a full table is an insert failure, which routes into the
"cache full -> fall back to RPC" path that already had to exist. This removes an
entire class of use-after-free rather than trying to synchronize it.

**Also found: `AllocateOffset` THROWS `OUT_OF_MEMORY`** (`arena_allocator.h:143`)
rather than returning null. Both containers now catch it, because the whole
fallback strategy depends on exhaustion being a recoverable condition rather
than an exception escaping into a client read path.

**Phase 0 (baseline) — done.** Measured on this branch after a consistent full
rebuild, single thread, `--test-case latency`: **SHM 72.1 µs**, IPC 109.7 µs, TCP
517.5 µs. Matches the pre-merge numbers (70.6 / 124.5 / 471.7), so merging `dev`
introduced no latency regression. **SHM 72 µs is the number the cache must beat**;
the target is < 5 µs.

**Resolved false alarm — ABI skew, not a `dev` bug.** During Phase 1,
`clio_run_thrpt_bench` aborted just after client init with `Fatal glibc error:
tpp.c:83 (__pthread_tpp_change_priority)`, in every transport mode including TCP. It
looked like a pre-existing `dev` regression. It was not: adding members to
`IpcManager` changes its layout, and rebuilding only `clio_run` left the module
`.so`s (`clio_*_runtime`) compiled against the old layout. The resulting ABI skew
presented as unrelated crashes — including a startup segfault when the skew ran the
other way. A consistent rebuild of runtime + all modules + bench made it vanish.

**Rule for this work:** after touching the layout of any widely-included type, rebuild
the modules in the same command, e.g.
`cmake --build build-cpu --target clio_run clio_admin_runtime clio_bdev_runtime
clio_MOD_NAME_runtime clio_cte_core_runtime clio_cae_core_runtime
clio_run_thrpt_bench -j 6`. A partial rebuild will produce crashes that look like
someone else's bug.

## 7. Decisions

1. **Trust boundary — DECIDED: accept node-wide metadata visibility.** Any client may
   read every tag/blob name and size on the node. Single-tenant assumption; revisit if
   multi-tenancy becomes a requirement.
2. **Reclamation — DECIDED: `Lease` + `TimedMutex` (§5.3b).** Not epochs, not hazard
   pointers, not never-reclaim. Lifetime is handled by a reclaimable lease; coherence
   stays with the generation counter.
3. **Both maps in SHM — DECIDED.** `string -> TagId` *and* `TagId -> TagInfo` (and the
   blob equivalents). No client-private ID caching scheme.
4. **`BlobBlock` — RESOLVED BY INSPECTION: a slimmed POD descriptor is required.**
   `BlobBlock` embeds a `bdev::Client`, which derives from `ContainerClient`
   (`container.h:590`). Its *data* is only `PoolId pool_id_` + `u32 return_code_`, but
   it declares `virtual void Init()` and `virtual ~ContainerClient()`, so the object
   **carries a vtable pointer**. A vptr is a process-local address (and moves under
   ASLR / differing .so load addresses), so it is meaningless — and dangerous — in a
   segment mapped by another process. The SHM block descriptor must therefore be plain
   POD: `{PoolId target_pool, PoolQuery target_query, u64 target_offset, u64 size,
   u64 capacity}`, with the `bdev::Client` reconstructed runtime-side on demand.
   **General rule for this work: no virtual functions in any type stored in SHM.**
5. **`kPinned` — DECIDED: stays on the private heap.** `cudaMallocHost` memory is not
   SHM-backed; the GPU-pinned RAM bdev keeps the existing path and simply does not
   participate in the client fast path.
6. **Fast-path blob size cap — measure in Phase 0/6.** Set it where the SHM path stops
   beating RPC; expected near 256 KB - 1 MB.
7. **Lock primitive — DECIDED: new `TimedMutex`** in `clio_ctp/thread/lock/timed_mutex.h`,
   alongside `mutex.h` / `rwlock.h` / `spin_lock.h` / `cvrwlock.h`.
8. **Smart pointer — DECIDED: new `Lease`** in `clio_ctp/memory/smart_ptr/lease.h`,
   alongside `shared_ptr.h` / `unique_ptr.h`. See §5.3b for its four constraints.

9. **Mapping — DECIDED: `PROT_READ|PROT_WRITE`.** Clients map read-write; only lock
   words are client-writable, enforced by convention rather than by the MMU. See
   §5.3b constraint 1 for the two invariants this makes load-bearing.
10. **Runtime waiting — DECIDED: the runtime may wait on leases.** Unavoidable, since
    an ignorable lease protects nothing. But it waits by `co_await` retry, never by
    blocking a worker thread (§5.3b constraint 4).

### Still open

- The exact fast-path size cap (item 6) — needs measurement, not a decision.
- Whether the `TimedMutex` threshold stays at 500 ms or drops once the size cap bounds
  legitimate hold time. This now matters more than it did: with the runtime waiting on
  leases (item 10), the threshold bounds **runtime metadata-write latency**, not just
  reclamation lag.
