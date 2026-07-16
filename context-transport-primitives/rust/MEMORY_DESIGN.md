# ctp-memory: Shared-Memory Design for the Rust Port (issue #756)

The memory module is the crux of the migration: every task, queue, and blob
buffer lives in shared memory addressed by `(AllocatorId, offset)` pairs.
This document is the contract the `ctp-memory` crate implements.

## Pillar 1 — one segment, one owner

A segment's **allocator metadata** (free lists, locks, heap bump pointer) is
mutated by exactly ONE implementation — the language that created it. Other
processes/languages may:

- **map** the segment,
- **resolve** any `ShmPtr` into it (registry lookup + `base + off`),
- **read/write the payload bytes** those pointers address,

but must **delegate allocation/free to the owner** (via task round-trip or
FFI), exactly like the transition single-owner rule for task ctors/dtors.
This removes cross-language lock-free-allocator races from the critical
path. If profiling ever demands co-owned segments, that becomes a separate,
stress-gated project against this same segment format.

## Pillar 2 — frozen pointer ABI

```
AllocatorId  = #[repr(C)] { major: u32, minor: u32 }     // == MemoryBackendId
ShmPtr<T>    = #[repr(C)] { alloc_id: AllocatorId, off: u64 }
OffsetPtr<T> = #[repr(C)] { off: u64 }                   // allocator implied
```

- Byte-compatible with the C++ `ShmPtrBase`/`OffsetPtrBase` and serialized
  identically (`alloc_id` then `off`).
- `off == u64::MAX` is the null sentinel (`IS_NULL`); `off` is relative to
  the segment's **heap base** (see layout below).
- `Copy + Send + Sync`, **no `Drop`**: a `ShmPtr` is a reference into shared
  state; releasing memory is always an explicit `free` on the owning
  allocator (mirrors the C++ FreeBuffer discipline and the task-teardown
  protocol from the task-ABI discussion).

## Pillar 3 — position-independence by construction

Nothing inside a segment stores an absolute pointer. All intra-segment links
are offsets; cross-process mutual exclusion uses in-segment atomics
(`AtomicU32` spinlock; a process crash while holding it is recovered by the
same daemon-restart machinery the C++ relies on). Any process mapping the
segment at any base is therefore correct by definition — this is stricter
than the C++ (which patches per-process bases into headers) and removes an
entire class of fixup code.

## Segment layout (v1)

```
[ SegmentHeader | heap ............................................. ]
SegmentHeader (64-byte aligned):
  magic: u64 = 0x4354_505f_5348_4d31  ("CTP_SHM1")
  version: u32
  owner_impl: u32          // 1 = C++, 2 = Rust (enforces Pillar 1)
  alloc_id: AllocatorId
  heap_size: u64
  lock: AtomicU32          // allocator spinlock
  free_head: u64           // offset of first free block (MAX = none)
  bump: u64                // high-water mark for never-freed space
Block header (16 bytes, precedes every allocation):
  size: u64                // payload size incl. padding
  next_free: u64           // valid only while on the free list
```

Allocation: first-fit over the free list (with block split), else bump.
Free: push to free-list head. Coalescing is deliberately deferred (v1 is
fragmentation-prone under adversarial churn — the C++ buddy allocator port
is the planned answer for slab-heavy workloads; see the #646 lineage).

## Rust safety model

- One `unsafe` core: segment mapping + `ShmPtr` resolution to raw pointers.
- Public typed surface: `alloc::<T>() -> ShmPtr<T>` and reads/writes are
  restricted to `T: ShmSafe` — a marker trait implemented only for
  POD-shaped types (no references, no Drop, `'static`), derived manually per
  type like the task-ABI POD structs.
- Registry: process-global `OnceLock<RwLock<HashMap<AllocatorId, Mapping>>>`
  (read-mostly; NOT thread_local, per project rule). Unregistered ids
  resolve to `None`, never UB.
- Cross-process correctness is proven by test: the suite re-spawns its own
  binary as a child (via ctp-introspect) that maps the same segment,
  resolves parent-written `ShmPtr`s, allocates, and hands pointers back.

## What this deliberately defers

- Buddy/slab/thread-local allocator ports (same segment format, new
  strategies within it).
- GPU-backed segments (gpu_shm_mmap) — after the ctp-gpu managed-memory path.
- Cross-language co-owned segments (see Pillar 1).
- C++-side regeneration of ShmPtrBase against this spec (task-ABI phase).
