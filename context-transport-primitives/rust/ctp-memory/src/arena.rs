// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).
//
//! Arena (bump-pointer) allocator and the `Heap` bump helper, over the v1
//! segment format of `MEMORY_DESIGN.md`.
//!
//! Ported from `clio_ctp/memory/allocator/heap.h` and
//! `clio_ctp/memory/allocator/arena_allocator.h`.
//!
//! [`Heap`] is the bump primitive: a monotonically increasing offset bounded
//! by `max_offset`. [`ArenaAllocator`] wraps it as a real segment allocator
//! that grows upwards and never frees individual allocations — memory comes
//! back only in bulk via [`ArenaAllocator::reset`], exactly as in the C++.
//!
//! All allocator state (bump position, bounds, allocation counter) lives in
//! the segment header as offsets and in-segment atomics, so any process that
//! maps the segment at any base address sees the same arena (design pillar 3).
//!
//! # Segment layout
//!
//! The arena header opens with the **byte-identical 56-byte prefix** of the
//! v1 `SegmentHeader` (see `allocator.rs`), so a reader can identify the
//! segment from `magic`/`version`/`alloc_id`/`heap_size` without knowing
//! which allocator owns it. The v1 `bump` slot at offset 48 *is* the arena's
//! bump pointer (C++ `Heap::heap_`), which is precisely that field's v1
//! meaning ("high-water mark for never-freed space"). Arena-specific state
//! follows in an extension region, and the heap begins at 128:
//!
//! ```text
//! 0    magic u64 ("CTP_SHM1")     32   lock u32 (unused: bump is lock-free)
//! 8    version u32                36   _pad u32
//! 12   owner_impl u32             40   free_head u64 (always NULL: no free list)
//! 16   alloc_major u32            48   heap.heap AtomicU64   == v1 `bump`
//! 20   alloc_minor u32          --- arena extension ---
//! 24   heap_size u64              56   heap.max_offset AtomicU64
//!                                 64   kind u64 ("CTP_AREN")
//!                                 72   heap_begin u64   (reset target)
//!                                 80   heap_max u64     (reset target)
//!                                 88   total_alloc AtomicU64
//!                                 96   _reserved [u64; 4]
//! 128  heap ...
//! ```
//!
//! Offsets handed out are relative to the heap base (segment base + 128),
//! matching the crate-wide convention the registry resolves against.
//!
//! # C++ → Rust name mapping
//!
//! | C++ (`heap.h`, `arena_allocator.h`, `allocator.h`) | Rust (this module) |
//! |---|---|
//! | `Heap<ATOMIC>` | [`Heap`] |
//! | `Heap::Heap()` / `Heap(initial, max)` | [`Heap::new`] / [`Heap::with_bounds`] |
//! | `Heap::Init` | [`Heap::init`] |
//! | `Heap::Allocate` | [`Heap::allocate`] |
//! | `Heap::GetOffset` | [`Heap::offset`] |
//! | `Heap::GetMaxOffset` | [`Heap::max_offset`] |
//! | `Heap::GetMaxSize` | [`Heap::max_size`] |
//! | `Heap::GetRemainingSize` | [`Heap::remaining_size`] |
//! | `_ArenaAllocator<ATOMIC>` / `ArenaAllocator<>` | [`ArenaAllocator`] |
//! | `shm_init(backend, region_size = 0)` | [`ArenaAllocator::create`] / [`ArenaAllocator::create_with_region`] |
//! | `shm_attach(backend)` | [`ArenaAllocator::open`] |
//! | `AllocateOffset(size, alignment)` | [`ArenaAllocator::allocate_offset_aligned`] |
//! | `AllocateOffset(size)` | [`ArenaAllocator::allocate_offset`] |
//! | `BaseAllocator::Allocate<T>` | [`ArenaAllocator::alloc`] / [`ArenaAllocator::alloc_bytes`] |
//! | `FreeOffsetNoNullCheck` / `BaseAllocator::Free` | [`ArenaAllocator::free_bytes`] / [`ArenaAllocator::free`] (no-ops) |
//! | `ReallocateOffsetNoNullCheck` | *(absent — divergence 4)* |
//! | `GetCurrentlyAllocatedSize` | [`ArenaAllocator::currently_allocated_size`] |
//! | `GetHeapOffset` | [`ArenaAllocator::heap_offset`] |
//! | `GetRemainingSize` | [`ArenaAllocator::remaining_size`] |
//! | `Reset` | [`ArenaAllocator::reset`] |
//! | `CreateTls` / `FreeTls` | [`ArenaAllocator::create_tls`] / [`ArenaAllocator::free_tls`] (no-ops) |
//! | `PushArenaState` / `PopArenaState` | [`ArenaAllocator::push_arena_state`] / [`ArenaAllocator::pop_arena_state`] |
//! | `ArenaState` | [`ArenaState`] |
//! | `Allocator::GetId` | [`ArenaAllocator::id`] |
//! | `Allocator::GetBackend` | [`ArenaAllocator::backend`] |
//! | `Allocator::GetAllocatorDataSize` | [`ArenaAllocator::capacity`] |
//! | `Allocator::GetBackendDataCapacity` | [`ArenaAllocator::heap_size`] |
//! | `AllocatorHeader::total_alloc_` | `ArenaHeader::total_alloc` (in-segment) |
//!
//! # Semantic divergences from the C++
//!
//! 1. **Failure is `None`, not `0`.** C++ `Heap::Allocate` returns `0` for
//!    OOM, which collides with a legitimate allocation at offset 0 — the
//!    arena papers over it with `off == 0 && heap_.GetOffset() != 0`. Here
//!    `allocate` returns `Option<u64>`, so offset 0 is allocatable and the
//!    disambiguation hack is gone. This matters: Rust heap offsets are
//!    heap-base-relative, so the *first* allocation is genuinely at 0,
//!    whereas the C++ offsets are backend-relative and start past the
//!    allocator object (never 0).
//! 2. **No `ATOMIC` template parameter.** `Heap<false>` (a plain `size_t`
//!    bump pointer) has no analog: a bump pointer shared across processes
//!    must be an in-segment atomic, and a non-atomic one would be UB the
//!    moment a second mapping touched it. [`Heap`] is always atomic
//!    (`SeqCst`, mirroring the C++ `opt_atomic` default ordering).
//! 3. **`CTP_ALLOC_TRACK_SIZE` is always on.** C++ gates `total_alloc_`
//!    updates behind that macro (off by default, so
//!    `GetCurrentlyAllocatedSize()` reports 0 in stock builds). Tracking is
//!    unconditional here — one `fetch_add` on the already-hot header line —
//!    because the crate cannot add a Cargo feature for it (single-file
//!    ownership rule) and an always-zero counter is worse than a correct one.
//! 4. **Exceptions → types.** `CTP_THROW_ERROR(OUT_OF_MEMORY)` becomes
//!    `None`. `ReallocateOffsetNoNullCheck`, which unconditionally throws
//!    `NOT_IMPLEMENTED`, is expressed by the *absence* of the method: an
//!    unsupported operation is a compile error rather than a runtime throw.
//! 5. **Alignment is validated.** C++ computes
//!    `(size + alignment - 1) & ~(alignment - 1)` with no checks — zero or
//!    non-power-of-two alignment silently corrupts the size (alignment 0
//!    yields size 0). Here a non-power-of-two (including 0) alignment
//!    returns `None`.
//! 6. **Overflow saturates instead of wrapping.** C++ `heap_ + size` and
//!    `size + alignment - 1` wrap on overflow, which can turn a huge request
//!    into a passing bounds check. Rust uses checked/saturating arithmetic
//!    and reports exhaustion.
//! 7. **No `this_` / `shift_` / `data_start_` / `GetBackendData()`.** The C++
//!    allocator object lives *inside* the region it manages and reconstructs
//!    pointers from `this`; per design pillar 3 nothing here stores an
//!    absolute address, so those fields collapse to compile-time constants
//!    (`data_start_` → `HEADER_SIZE`, `this_` → 0) and pointer
//!    reconstruction is the registry's job. Consequently `GetAllocatorDataOff`
//!    is always `HEADER_SIZE` and is not exposed.
//! 8. **`region_size` is clamped to the segment.** C++ takes `region_size` on
//!    trust; a too-large value lets the heap bump past the mapping (OOB).
//!    Here it is clamped to the backend size. A `region_size` *smaller* than
//!    the header degrades to a permanently-full arena, which is what the C++
//!    arithmetic also produces (`heap_max_ < heap_begin_`), reached via
//!    `saturating_sub` rather than by unsigned wraparound.
//! 9. **Over-aligned `T` is refused, not misaligned.** The C++ arena aligns
//!    the *size*, not the offset, so mixing alignments can hand back an
//!    offset that is unaligned for the type (benign-ish in C++, instant UB in
//!    Rust). [`ArenaAllocator::alloc`] verifies the resulting address and
//!    returns `None` instead of producing a misaligned reference. With the
//!    default `size_of::<usize>()` alignment and uniform usage this never
//!    fires; the raw [`ArenaAllocator::allocate_offset_aligned`] keeps the
//!    C++ size-rounding semantics untouched.
//! 10. **`Heap` is not [`ShmSafe`]** even though it lives in the segment: the
//!     marker requires `Copy`, and atomics are not `Copy`. It is `repr(C)`,
//!     `Drop`-free and position-independent, which is what the segment
//!     actually requires; `ShmSafe` governs the *payload* types the typed
//!     `alloc` hands out.
//! 11. **`free` is safe.** `FreeListAllocator::free_bytes` is an `unsafe fn`
//!     because it mutates a free list; the arena's free is a documented no-op
//!     with no invariants to uphold, so marking it `unsafe` would be noise.
//! 12. **`PushArenaState`'s out-params become a return value.** C++ signals
//!     "no sub-arena" with `bool` + two out-params; the Rust form returns
//!     `Option<(ArenaState, OffsetPtr<u8>)>` — always `None`, since an arena
//!     is already a bump allocator. The `Arena<AllocT>` RAII handle and
//!     `BaseAllocator::CreateSubAllocator` (which would host an arena inside
//!     another allocator's region) are not ported here — they belong to the
//!     `BaseAllocator` layer, not to this module's one file.
//! 13. **`size_t` → `u64`.** C++ offsets are `size_t`; the frozen ABI fixes
//!     them at `u64` (`MEMORY_DESIGN.md` pillar 2), so this port is
//!     word-size-independent where the C++ is not.
//!
//! # Concurrency note (not a divergence)
//!
//! [`Heap::allocate`] reproduces the C++ `fetch_add` / rollback-with-
//! `fetch_sub` algorithm verbatim. It never hands the same region to two
//! callers, but under contention a thread whose `fetch_add` overshoots
//! `max_offset` transiently pushes the bump pointer past the limit, so a
//! concurrent request can fail spuriously even though space exists once the
//! overshoot rolls back. That conservative behavior is the C++ behavior and
//! is preserved deliberately.

use crate::backend::SharedMemBackend;
use crate::ptr::{AllocatorId, OffsetPtr, ShmPtr, ShmSafe, NULL_OFFSET};
use crate::registry;
use std::sync::atomic::{AtomicU64, Ordering};

/// v1 segment magic ("CTP_SHM1"), shared with `allocator.rs`.
const MAGIC: u64 = 0x4354_505f_5348_4d31;
/// v1 segment version, shared with `allocator.rs`.
const VERSION: u32 = 1;
/// `owner_impl` value for Rust-owned segments (design pillar 1).
const OWNER_RUST: u32 = 2;
/// Arena extension tag ("CTP_AREN") at header offset 64: makes an arena
/// segment self-describing, since the v1 prefix carries no allocator kind.
const ARENA_KIND: u64 = 0x4354_505f_4152_454e;
/// Arena header size; the heap starts here. 128 = the 56-byte v1 prefix plus
/// the arena extension, rounded to the next 64-byte line.
const HEADER_SIZE: u64 = 128;
/// C++ `AllocateOffset`'s default alignment: `sizeof(size_t)`.
const DEFAULT_ALIGNMENT: u64 = std::mem::size_of::<usize>() as u64;
/// The C++ `opt_atomic` default ordering is sequentially consistent.
const ORD: Ordering = Ordering::SeqCst;

/// Bump-pointer helper (C++ `Heap<ATOMIC>`).
///
/// Not an allocator: a utility for allocators that need monotonically
/// increasing offset allocation. `repr(C)`, `Drop`-free and made of plain
/// integers, so it may live in a shared segment (see divergence 10 for why it
/// is not [`ShmSafe`]).
#[repr(C)]
pub struct Heap {
    /// Current heap offset (C++ `heap_`).
    heap: AtomicU64,
    /// Maximum offset the heap can reach (C++ `max_offset_`), i.e.
    /// `initial_offset + max_size`.
    max_offset: AtomicU64,
}

impl Heap {
    /// C++ `Heap()` — an empty, zero-capacity heap.
    pub const fn new() -> Self {
        Self {
            heap: AtomicU64::new(0),
            max_offset: AtomicU64::new(0),
        }
    }

    /// C++ `Heap(initial_offset, max_offset)`.
    pub const fn with_bounds(initial_offset: u64, max_offset: u64) -> Self {
        Self {
            heap: AtomicU64::new(initial_offset),
            max_offset: AtomicU64::new(max_offset),
        }
    }

    /// C++ `Heap::Init` — (re)set the bump position and the bound.
    ///
    /// Takes `&self`: the fields are atomics, which is also what lets a
    /// mapped-in-shared-memory heap be reset through a shared reference.
    pub fn init(&self, initial_offset: u64, max_offset: u64) {
        self.max_offset.store(max_offset, ORD);
        self.heap.store(initial_offset, ORD);
    }

    /// C++ `Heap::Allocate` — bump by `size`, returning the offset of the
    /// region, or `None` when it would cross `max_offset` (C++: `0`).
    ///
    /// See the module's concurrency note: an overshooting `fetch_add` is
    /// rolled back with `fetch_sub`, exactly as the C++ does.
    pub fn allocate(&self, size: u64) -> Option<u64> {
        let max = self.max_offset.load(ORD);

        // C++: `if (heap_.load() + size > max_offset_) return 0;`
        // Saturating: the C++ sum wraps and can wrongly pass the check.
        if self.heap.load(ORD).saturating_add(size) > max {
            return None;
        }

        // Atomically fetch the current offset and advance the heap.
        let off = self.heap.fetch_add(size, ORD);
        let end_off = off.saturating_add(size);

        // A concurrent allocation may have consumed the space between the
        // check and the fetch_add: roll back and report failure.
        if end_off > max {
            self.heap.fetch_sub(size, ORD);
            return None;
        }
        Some(off)
    }

    /// C++ `Heap::GetOffset` — the current top of the heap.
    pub fn offset(&self) -> u64 {
        self.heap.load(ORD)
    }

    /// C++ `Heap::GetMaxOffset` — the maximum offset the heap can reach.
    pub fn max_offset(&self) -> u64 {
        self.max_offset.load(ORD)
    }

    /// C++ `Heap::GetMaxSize`. Kept for surface parity: despite the name it
    /// returns the max *offset*, identical to [`Heap::max_offset`] — the C++
    /// misnomer is preserved rather than silently "fixed".
    pub fn max_size(&self) -> u64 {
        self.max_offset()
    }

    /// C++ `Heap::GetRemainingSize` — bytes left, saturating at 0 when the
    /// heap has overshot (see the concurrency note).
    pub fn remaining_size(&self) -> u64 {
        // C++: `(current < max_offset_) ? (max_offset_ - current) : 0`.
        self.max_offset.load(ORD).saturating_sub(self.heap.load(ORD))
    }
}

impl Default for Heap {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for Heap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Heap")
            .field("offset", &self.offset())
            .field("max_offset", &self.max_offset())
            .finish()
    }
}

/// C++ `ArenaState` — bump state for a sub-arena carved out of an allocator's
/// region. Present for surface parity; the arena allocator never activates one
/// (see divergence 12).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ArenaState {
    /// Offset of the arena block (C++ `arena_off_`).
    pub arena_off: u64,
    /// Current bump position within the arena (C++ `arena_cur_`).
    pub arena_cur: u64,
    /// End of the arena block (C++ `arena_end_`).
    pub arena_end: u64,
}

impl ArenaState {
    /// C++ `ArenaState::IsActive`.
    pub const fn is_active(&self) -> bool {
        self.arena_off != 0
    }
}

// SAFETY: three plain u64 offsets — no references, no Drop, valid for any bit
// pattern, and position-independent.
unsafe impl ShmSafe for ArenaState {}

/// The in-segment arena header. See the module-level layout diagram; the
/// first 56 bytes are the v1 `SegmentHeader` prefix verbatim.
#[repr(C)]
struct ArenaHeader {
    magic: u64,
    version: u32,
    owner_impl: u32,
    alloc_major: u32,
    alloc_minor: u32,
    /// Bytes of heap in the segment (`backend.size() - HEADER_SIZE`).
    heap_size: u64,
    /// v1 allocator spinlock. Unused: bump allocation is lock-free.
    lock: u32,
    _pad: u32,
    /// v1 free-list head. Always null: the arena has no free list.
    free_head: u64,
    /// C++ `_ArenaAllocator::heap_`. Its first word occupies the v1 `bump`
    /// slot, which is exactly that field's v1 meaning.
    heap: Heap,
    /// Arena self-description tag, checked by [`ArenaAllocator::open`].
    kind: u64,
    /// C++ `_ArenaAllocator::heap_begin_` — the reset target.
    heap_begin: u64,
    /// C++ `_ArenaAllocator::heap_max_` — the reset bound. Duplicates the
    /// live `heap.max_offset` exactly as the C++ duplicates it, so that
    /// `reset` restores from the initial parameters.
    heap_max: u64,
    /// C++ `AllocatorHeader::total_alloc_`.
    total_alloc: AtomicU64,
    _reserved: [u64; 4],
}

const _: () = assert!(std::mem::size_of::<ArenaHeader>() as u64 == HEADER_SIZE);
// The v1 SegmentHeader prefix must stay byte-identical (allocator.rs).
const _: () = assert!(std::mem::offset_of!(ArenaHeader, heap_size) == 24);
const _: () = assert!(std::mem::offset_of!(ArenaHeader, free_head) == 40);
// `heap.heap` == the v1 `bump` slot at 48.
const _: () = assert!(std::mem::offset_of!(ArenaHeader, heap) == 48);
const _: () = assert!(std::mem::offset_of!(ArenaHeader, kind) == 64);

/// Bump/arena allocator over a shared-memory segment (C++
/// `ArenaAllocator<ATOMIC>` = `BaseAllocator<_ArenaAllocator<ATOMIC>>`).
///
/// Allocations only move the bump pointer upwards; [`free`](Self::free) is a
/// no-op and memory returns in bulk from [`reset`](Self::reset). Like
/// [`FreeListAllocator`](crate::FreeListAllocator) this handle owns its
/// mapping and registers itself in the process registry on create/open,
/// unregistering on drop.
pub struct ArenaAllocator {
    backend: SharedMemBackend,
    id: AllocatorId,
}

impl ArenaAllocator {
    /// C++ `shm_init(backend)` — initialize a fresh arena over the whole
    /// segment (C++ `region_size = 0` → `backend.data_capacity_`). This
    /// process/language becomes the segment's owner (design pillar 1).
    ///
    /// # Panics
    /// If the segment is not larger than the 128-byte arena header.
    pub fn create(backend: SharedMemBackend, id: AllocatorId) -> Self {
        Self::create_with_region(backend, id, 0)
    }

    /// C++ `shm_init(backend, region_size)` — initialize a fresh arena whose
    /// managed region spans `region_size` bytes *including* the header, so
    /// the usable capacity is `region_size - HEADER_SIZE` (C++
    /// `GetAllocatorDataSize()`). `region_size == 0` means the whole segment.
    ///
    /// `region_size` is clamped to the segment size (divergence 8); values
    /// below the header size yield a permanently-full arena, matching the
    /// C++ arithmetic.
    ///
    /// # Panics
    /// If the segment is not larger than the 128-byte arena header.
    pub fn create_with_region(backend: SharedMemBackend, id: AllocatorId, region_size: u64) -> Self {
        let seg_size = backend.size() as u64;
        assert!(
            seg_size > HEADER_SIZE,
            "arena segment ({seg_size} B) must exceed the {HEADER_SIZE} B header"
        );
        let heap_size = seg_size - HEADER_SIZE;
        // C++: `if (region_size == 0) region_size = backend.data_capacity_;`
        let region_size = if region_size == 0 {
            seg_size
        } else {
            region_size.min(seg_size)
        };
        // C++: heap_begin_ = GetAllocatorDataOff() = this_ + data_start_;
        //      heap_max_   = GetAllocatorDataOff() + region_size - data_start_
        //                  = this_ + region_size.
        // Heap-relative (this_ = 0, data_start_ = HEADER_SIZE) that is:
        let heap_begin = 0u64;
        let heap_max = region_size.saturating_sub(HEADER_SIZE);

        let hdr = backend.base() as *mut ArenaHeader;
        // SAFETY: `hdr` is the base of a freshly mapped, exclusively owned
        // segment larger than the header (asserted above); a mapping base is
        // page-aligned, hence suitably aligned for ArenaHeader.
        unsafe {
            hdr.write(ArenaHeader {
                magic: MAGIC,
                version: VERSION,
                owner_impl: OWNER_RUST,
                alloc_major: id.major,
                alloc_minor: id.minor,
                heap_size,
                lock: 0,
                _pad: 0,
                free_head: NULL_OFFSET,
                heap: Heap::with_bounds(heap_begin, heap_max),
                kind: ARENA_KIND,
                heap_begin,
                heap_max,
                total_alloc: AtomicU64::new(0),
                _reserved: [0; 4],
            });
        }
        let me = Self { backend, id };
        me.register();
        me
    }

    /// C++ `shm_attach(backend)` — attach to an existing arena segment.
    ///
    /// The C++ attach is a no-op because all arena state is in shared memory;
    /// that remains true here, so this only validates that the segment really
    /// is an arena with the expected id and registers the mapping.
    pub fn open(backend: SharedMemBackend, expect_id: AllocatorId) -> std::io::Result<Self> {
        let ok = (backend.size() as u64) > HEADER_SIZE && {
            let hdr = backend.base() as *const ArenaHeader;
            // SAFETY: the mapping exceeds the header (checked left of `&&`,
            // which short-circuits); every field read here is immutable after
            // create, so no torn read is possible even with live writers.
            unsafe {
                (*hdr).magic == MAGIC
                    && (*hdr).version == VERSION
                    && (*hdr).kind == ARENA_KIND
                    && (*hdr).alloc_major == expect_id.major
                    && (*hdr).alloc_minor == expect_id.minor
            }
        };
        if !ok {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "not a CTP_SHM1 arena segment or id mismatch",
            ));
        }
        let me = Self {
            backend,
            id: expect_id,
        };
        me.register();
        Ok(me)
    }

    fn register(&self) {
        registry::register_allocator(self.id, self.heap_base(), self.heap_size());
    }

    /// C++ `Allocator::GetId`.
    pub fn id(&self) -> AllocatorId {
        self.id
    }

    /// C++ `Allocator::GetBackend`.
    pub fn backend(&self) -> &SharedMemBackend {
        &self.backend
    }

    fn header(&self) -> &ArenaHeader {
        // SAFETY: create/open established that the mapping is larger than
        // HEADER_SIZE and holds an ArenaHeader; the mapping outlives `&self`.
        // A shared reference suffices for the whole header: the mutable
        // fields are atomics, and the rest is immutable after create — so
        // concurrent mappings in other processes cannot invalidate it.
        unsafe { &*(self.backend.base() as *const ArenaHeader) }
    }

    fn heap_base(&self) -> *mut u8 {
        // SAFETY: the segment is larger than HEADER_SIZE (create/open), so
        // this stays within the mapping.
        unsafe { self.backend.base().add(HEADER_SIZE as usize) }
    }

    /// Total heap bytes in the segment (C++
    /// `Allocator::GetBackendDataCapacity` net of the header). Independent of
    /// `region_size`: see [`capacity`](Self::capacity) for what this arena may
    /// actually hand out.
    pub fn heap_size(&self) -> u64 {
        self.header().heap_size
    }

    /// C++ `Allocator::GetAllocatorDataSize` — the arena's managed capacity,
    /// i.e. `region_size - HEADER_SIZE`.
    pub fn capacity(&self) -> u64 {
        let h = self.header();
        h.heap_max.saturating_sub(h.heap_begin)
    }

    /// C++ `AllocateOffset(size, alignment)`.
    ///
    /// `size` is rounded **up to a multiple of `alignment`** so that the next
    /// allocation starts aligned — the C++ aligns the size, not the offset
    /// (see divergence 9). Returns `None` on exhaustion (C++ throws
    /// `OUT_OF_MEMORY`) or if `alignment` is not a power of two (C++: UB).
    pub fn allocate_offset_aligned(&self, size: u64, alignment: u64) -> Option<OffsetPtr<u8>> {
        if !alignment.is_power_of_two() {
            return None;
        }
        // C++: `size = (size + alignment - 1) & ~(alignment - 1);` — checked
        // because the C++ sum wraps (divergence 6).
        let size = size.checked_add(alignment - 1)? & !(alignment - 1);

        let h = self.header();
        // C++ pre-check: an early OUT_OF_MEMORY before touching the heap.
        if h.heap.remaining_size() < size {
            return None;
        }
        let off = h.heap.allocate(size)?;
        h.total_alloc.fetch_add(size, ORD);
        Some(OffsetPtr::new(off))
    }

    /// C++ `AllocateOffset(size)` — [`allocate_offset_aligned`] with the C++
    /// default alignment of `sizeof(size_t)`.
    ///
    /// [`allocate_offset_aligned`]: Self::allocate_offset_aligned
    pub fn allocate_offset(&self, size: u64) -> Option<OffsetPtr<u8>> {
        self.allocate_offset_aligned(size, DEFAULT_ALIGNMENT)
    }

    /// Allocate `size` payload bytes. Crate-parity alias for
    /// [`allocate_offset`](Self::allocate_offset), mirroring
    /// `FreeListAllocator::alloc_bytes`.
    pub fn alloc_bytes(&self, size: u64) -> Option<OffsetPtr<u8>> {
        self.allocate_offset(size)
    }

    /// Typed allocation, zero-initialized (C++ `BaseAllocator::Allocate<T>`).
    ///
    /// Returns `None` on exhaustion, or if `T`'s alignment cannot be honoured
    /// by the bump pointer's current position (divergence 9) — impossible for
    /// types aligned to at most `size_of::<usize>()` when every allocation
    /// uses the default alignment.
    pub fn alloc<T: ShmSafe>(&self) -> Option<ShmPtr<T>> {
        let align = std::mem::align_of::<T>() as u64;
        let off = self.allocate_offset_aligned(
            std::mem::size_of::<T>() as u64,
            align.max(DEFAULT_ALIGNMENT),
        )?;
        // The heap base is only HEADER_SIZE-aligned, so check the resulting
        // address rather than the offset. Leaking the bumped bytes is the
        // arena's normal mode: only `reset` reclaims.
        let addr = self.heap_base() as usize + off.off as usize;
        if !addr.is_multiple_of(std::mem::align_of::<T>()) {
            return None;
        }
        // SAFETY: the block was just bumped out of this arena, so
        // [off, off + size_of::<T>()) lies inside the mapped heap and is
        // owned by this allocation alone.
        unsafe {
            std::ptr::write_bytes(addr as *mut u8, 0, std::mem::size_of::<T>());
        }
        Some(off.to_shm(self.id).cast())
    }

    /// C++ `FreeOffsetNoNullCheck` — **a no-op, intentionally, not an error**.
    /// An arena frees only in bulk; see [`reset`](Self::reset).
    pub fn free_bytes(&self, _ptr: OffsetPtr<u8>) {}

    /// C++ `BaseAllocator::Free<T>` — a no-op, as [`free_bytes`](Self::free_bytes).
    pub fn free<T>(&self, _ptr: ShmPtr<T>) {}

    /// C++ `GetCurrentlyAllocatedSize` — total bytes handed out since the last
    /// [`reset`](Self::reset) (rounded sizes, matching what the heap consumed).
    /// Never decreases short of a reset: the arena has no per-block free.
    pub fn currently_allocated_size(&self) -> u64 {
        self.header().total_alloc.load(ORD)
    }

    /// C++ `GetHeapOffset` — the current bump position (heap-relative).
    pub fn heap_offset(&self) -> u64 {
        self.header().heap.offset()
    }

    /// C++ `GetRemainingSize` — bytes still allocatable from the arena.
    pub fn remaining_size(&self) -> u64 {
        self.header().heap.remaining_size()
    }

    /// C++ `Reset` — rewind the bump pointer to its initial position, freeing
    /// every allocation at once, and zero the allocation counter.
    ///
    /// Every outstanding pointer into this arena dangles afterwards: the next
    /// allocation hands the same bytes to someone else. That is the C++
    /// contract too — the caller must have quiesced all users first, which is
    /// why this cannot be expressed through Rust lifetimes (offsets are not
    /// borrows).
    pub fn reset(&self) {
        let h = self.header();
        h.total_alloc.store(0, ORD);
        // C++: heap_.Init(heap_begin_, heap_max_) — restored from the stored
        // init parameters, not from the heap's live bound.
        h.heap.init(h.heap_begin, h.heap_max);
    }

    /// C++ `CreateTls` — no TLS is needed for an arena; a no-op.
    /// (Rust `thread_local!` is banned project-wide regardless.)
    pub fn create_tls(&self) {}

    /// C++ `FreeTls` — no TLS is needed for an arena; a no-op.
    pub fn free_tls(&self) {}

    /// C++ `PushArenaState` — always `None` (C++: `false`): an arena is
    /// already a bump allocator, so it hosts no sub-arena. See divergence 12.
    pub fn push_arena_state(&self, _size: u64) -> Option<(ArenaState, OffsetPtr<u8>)> {
        None
    }

    /// C++ `PopArenaState` — a no-op, the counterpart of
    /// [`push_arena_state`](Self::push_arena_state).
    pub fn pop_arena_state(&self, _prior: &ArenaState, _block: OffsetPtr<u8>) {}
}

impl Drop for ArenaAllocator {
    fn drop(&mut self) {
        registry::unregister_allocator(self.id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    // ---- Heap (pure logic; no shared memory needed) --------------------

    #[test]
    fn heap_default_is_empty_and_full() {
        let h = Heap::new();
        assert_eq!(h.offset(), 0);
        assert_eq!(h.max_offset(), 0);
        assert_eq!(h.remaining_size(), 0);
        // A zero-capacity heap allocates nothing but zero bytes.
        assert_eq!(h.allocate(0), Some(0));
        assert_eq!(h.allocate(1), None);
        assert_eq!(Heap::default().offset(), 0);
    }

    #[test]
    fn heap_bumps_and_reports_bounds() {
        let h = Heap::with_bounds(64, 64 + 256);
        assert_eq!(h.offset(), 64);
        assert_eq!(h.max_offset(), 320);
        // GetMaxSize is the C++ misnomer for GetMaxOffset (divergence note).
        assert_eq!(h.max_size(), h.max_offset());
        assert_eq!(h.remaining_size(), 256);

        assert_eq!(h.allocate(100), Some(64));
        assert_eq!(h.offset(), 164);
        assert_eq!(h.remaining_size(), 156);
        assert_eq!(h.allocate(56), Some(164));
        assert_eq!(h.remaining_size(), 100);

        // Exact fit succeeds and leaves the heap exactly at max.
        assert_eq!(h.allocate(100), Some(220));
        assert_eq!(h.offset(), 320);
        assert_eq!(h.remaining_size(), 0);
        // One byte past is refused, and the heap is untouched.
        assert_eq!(h.allocate(1), None);
        assert_eq!(h.offset(), 320);
        // A zero-byte request still succeeds at the very top.
        assert_eq!(h.allocate(0), Some(320));
    }

    #[test]
    fn heap_zero_size_does_not_advance() {
        let h = Heap::with_bounds(0, 16);
        assert_eq!(h.allocate(0), Some(0));
        assert_eq!(h.offset(), 0);
        // Offset 0 is a real allocation here, not the C++ OOM sentinel.
        assert_eq!(h.allocate(8), Some(0));
        assert_eq!(h.allocate(0), Some(8));
        assert_eq!(h.offset(), 8);
    }

    #[test]
    fn heap_saturates_instead_of_wrapping() {
        let h = Heap::with_bounds(8, 4096);
        // C++ `heap_ + size` would wrap to 7 and pass the bounds check.
        assert_eq!(h.allocate(u64::MAX), None);
        assert_eq!(h.offset(), 8, "a rejected request must not move the heap");
        assert_eq!(h.allocate(u64::MAX - 7), None);
        assert_eq!(h.offset(), 8);
    }

    #[test]
    fn heap_init_resets_position_and_bound() {
        let h = Heap::with_bounds(0, 32);
        assert_eq!(h.allocate(32), Some(0));
        assert_eq!(h.allocate(1), None);
        h.init(4, 68);
        assert_eq!(h.offset(), 4);
        assert_eq!(h.max_offset(), 68);
        assert_eq!(h.remaining_size(), 64);
        assert_eq!(h.allocate(64), Some(4));
    }

    #[test]
    fn heap_remaining_saturates_when_overshot() {
        // Shrinking the bound below the live offset (or a transient overshoot)
        // must report 0 remaining, never an underflowed huge number.
        let h = Heap::with_bounds(0, 128);
        assert_eq!(h.allocate(128), Some(0));
        h.max_offset.store(64, ORD);
        assert_eq!(h.remaining_size(), 0);
        assert_eq!(h.allocate(1), None);
    }

    #[test]
    fn heap_concurrent_allocations_are_disjoint() {
        use std::sync::{Arc, Mutex};
        const THREADS: u64 = 8;
        const PER_THREAD: u64 = 64;
        const BLOCK: u64 = 16;
        // Deliberately half the space the threads ask for: the rollback path
        // (fetch_add overshoot → fetch_sub) is only reachable under contention.
        const CAP: u64 = THREADS * PER_THREAD * BLOCK / 2;
        let h = Arc::new(Heap::with_bounds(0, CAP));
        let got: Arc<Mutex<Vec<u64>>> = Arc::new(Mutex::new(Vec::new()));
        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                let h = Arc::clone(&h);
                let got = Arc::clone(&got);
                std::thread::spawn(move || {
                    let mut mine = Vec::new();
                    for _ in 0..PER_THREAD {
                        if let Some(off) = h.allocate(BLOCK) {
                            mine.push(off);
                        }
                    }
                    got.lock().unwrap().extend(mine);
                })
            })
            .collect();
        for t in handles {
            t.join().unwrap();
        }
        let mut offs = got.lock().unwrap().clone();
        let n = offs.len() as u64;
        offs.sort_unstable();
        offs.dedup();
        assert_eq!(offs.len() as u64, n, "the same region was handed out twice");
        // Every allocation is in-bounds and block-aligned; the heap never
        // ends up past its bound despite transient overshoots.
        assert!(offs.iter().all(|&o| o % BLOCK == 0 && o + BLOCK <= CAP));
        assert!(h.offset() <= CAP);
        assert_eq!(h.offset(), n * BLOCK);
    }

    // ---- ArenaAllocator ------------------------------------------------

    /// Unique AllocatorId per call: the registry is process-global and tests
    /// run in parallel, so a shared id would let one test's Drop unregister
    /// another's live mapping. Major 701 keeps arenas clear of allocator.rs's
    /// 700.
    fn next_id() -> AllocatorId {
        static NEXT_MINOR: AtomicU32 = AtomicU32::new(1);
        AllocatorId::new(701, NEXT_MINOR.fetch_add(1, Ordering::Relaxed))
    }

    fn seg(tag: &str, bytes: usize) -> (SharedMemBackend, String) {
        let name = format!("ctp_rs_arena_{}_{}_{:?}", tag, std::process::id(), {
            static N: AtomicU32 = AtomicU32::new(0);
            N.fetch_add(1, Ordering::Relaxed)
        });
        let backend = SharedMemBackend::create(&name, bytes).unwrap();
        (backend, name)
    }

    fn fresh(tag: &str, bytes: usize) -> ArenaAllocator {
        let (backend, _name) = seg(tag, bytes);
        backend.destroy(); // unlink the name eagerly (POSIX); mapping stays
        ArenaAllocator::create(backend, next_id())
    }

    #[test]
    fn header_layout_matches_v1_prefix() {
        // Compile-time asserts cover the offsets; this pins the sizes.
        assert_eq!(std::mem::size_of::<ArenaHeader>(), 128);
        assert_eq!(std::mem::size_of::<Heap>(), 16);
        assert_eq!(std::mem::align_of::<Heap>(), 8);
        assert_eq!(std::mem::size_of::<ArenaState>(), 24);
    }

    #[test]
    fn create_spans_whole_segment() {
        let a = fresh("whole", 1 << 16);
        assert_eq!(a.heap_size(), (1 << 16) - HEADER_SIZE);
        assert_eq!(a.capacity(), (1 << 16) - HEADER_SIZE);
        assert_eq!(a.remaining_size(), a.capacity());
        assert_eq!(a.heap_offset(), 0);
        assert_eq!(a.currently_allocated_size(), 0);
    }

    #[test]
    fn allocations_bump_upwards_and_never_overlap() {
        let a = fresh("bump", 1 << 16);
        let p1 = a.allocate_offset(100).unwrap();
        let p2 = a.allocate_offset(100).unwrap();
        let p3 = a.allocate_offset(100).unwrap();
        // First allocation sits at heap-relative 0 — a legitimate offset here,
        // unlike the C++ where 0 doubles as the OOM sentinel (divergence 1).
        assert_eq!(p1.off, 0);
        // 100 rounds up to 104 with the default 8-byte alignment.
        assert_eq!(p2.off, 104);
        assert_eq!(p3.off, 208);
        assert_eq!(a.heap_offset(), 312);
        assert_eq!(a.currently_allocated_size(), 312);
        assert_eq!(a.remaining_size(), a.capacity() - 312);
    }

    #[test]
    fn alignment_rounds_size_not_offset() {
        let a = fresh("align", 1 << 16);
        // Default alignment: sizeof(size_t) == 8.
        assert_eq!(a.allocate_offset(1).unwrap().off, 0);
        assert_eq!(a.allocate_offset(1).unwrap().off, 8);
        // Explicit alignment rounds the size up to its multiple.
        assert_eq!(a.allocate_offset_aligned(1, 64).unwrap().off, 16);
        assert_eq!(a.heap_offset(), 80);
        assert_eq!(a.allocate_offset_aligned(65, 64).unwrap().off, 80);
        assert_eq!(a.heap_offset(), 80 + 128);
        // Exactly-aligned sizes are untouched.
        assert_eq!(a.allocate_offset_aligned(32, 32).unwrap().off, 208);
        assert_eq!(a.heap_offset(), 240);
    }

    #[test]
    fn invalid_alignment_is_refused() {
        let a = fresh("badalign", 1 << 16);
        // C++ computes garbage for these (alignment 0 collapses size to 0).
        assert!(a.allocate_offset_aligned(16, 0).is_none());
        assert!(a.allocate_offset_aligned(16, 3).is_none());
        assert!(a.allocate_offset_aligned(16, 24).is_none());
        assert_eq!(a.heap_offset(), 0, "a refused request must not bump");
        // ...and a valid one still works afterwards.
        assert!(a.allocate_offset_aligned(16, 16).is_some());
    }

    #[test]
    fn size_overflow_reports_exhaustion() {
        let a = fresh("ovf", 1 << 16);
        // size + alignment - 1 would wrap in C++.
        assert!(a.allocate_offset(u64::MAX).is_none());
        assert!(a.allocate_offset_aligned(u64::MAX - 3, 8).is_none());
        assert_eq!(a.heap_offset(), 0);
        assert_eq!(a.currently_allocated_size(), 0);
    }

    #[test]
    fn zero_size_allocation_does_not_advance() {
        let a = fresh("zero", 1 << 16);
        let p = a.allocate_offset(0).unwrap();
        assert_eq!(p.off, 0);
        assert_eq!(a.heap_offset(), 0);
        // A real allocation then starts at the same offset.
        assert_eq!(a.allocate_offset(8).unwrap().off, 0);
        assert_eq!(a.allocate_offset(0).unwrap().off, 8);
    }

    #[test]
    fn exhaustion_returns_none_and_leaves_the_arena_usable() {
        // 4096-byte segment → 3968 bytes of capacity.
        let a = fresh("oom", 4096);
        let cap = a.capacity();
        assert!(a.allocate_offset(cap - 8).is_some());
        assert_eq!(a.remaining_size(), 8);
        // Too big: refused, nothing bumped.
        assert!(a.allocate_offset(16).is_none());
        assert_eq!(a.remaining_size(), 8);
        assert!(a.allocate_offset(cap).is_none());
        // The remaining 8 bytes are still allocatable.
        assert!(a.allocate_offset(8).is_some());
        assert_eq!(a.remaining_size(), 0);
        assert!(a.allocate_offset(1).is_none());
        assert_eq!(a.currently_allocated_size(), cap);
    }

    #[test]
    fn free_is_a_noop_and_reset_reclaims_everything() {
        let a = fresh("reset", 1 << 16);
        let p1 = a.allocate_offset(1024).unwrap();
        let _p2 = a.allocate_offset(1024).unwrap();
        assert_eq!(a.heap_offset(), 2048);

        // Freeing an individual allocation is intentionally a no-op: the
        // space is NOT reclaimed and the next allocation does not reuse it.
        a.free_bytes(p1);
        assert_eq!(a.heap_offset(), 2048);
        assert_eq!(a.currently_allocated_size(), 2048);
        let p3 = a.allocate_offset(1024).unwrap();
        assert_eq!(p3.off, 2048);

        // Reset rewinds everything at once.
        a.reset();
        assert_eq!(a.heap_offset(), 0);
        assert_eq!(a.remaining_size(), a.capacity());
        assert_eq!(a.currently_allocated_size(), 0);
        assert_eq!(a.allocate_offset(1024).unwrap().off, p1.off);
        // Reset is idempotent on an empty arena.
        a.reset();
        a.reset();
        assert_eq!(a.heap_offset(), 0);
    }

    #[test]
    fn reset_restores_a_sub_region_bound_not_the_segment() {
        let (backend, _n) = seg("resetregion", 1 << 16);
        backend.destroy();
        // region_size covers header + 1024 usable bytes.
        let a = ArenaAllocator::create_with_region(backend, next_id(), HEADER_SIZE + 1024);
        assert_eq!(a.capacity(), 1024);
        assert!(a.allocate_offset(1024).is_some());
        assert!(a.allocate_offset(8).is_none());
        a.reset();
        // The bound comes back as region_size, NOT as the whole segment.
        assert_eq!(a.capacity(), 1024);
        assert_eq!(a.remaining_size(), 1024);
        assert!(a.allocate_offset(1024).is_some());
        assert!(a.allocate_offset(8).is_none());
    }

    #[test]
    fn region_size_limits_capacity_below_the_segment() {
        let (backend, _n) = seg("region", 1 << 16);
        backend.destroy();
        let a = ArenaAllocator::create_with_region(backend, next_id(), HEADER_SIZE + 256);
        assert_eq!(a.heap_size(), (1 << 16) - HEADER_SIZE);
        assert_eq!(a.capacity(), 256);
        assert_eq!(a.remaining_size(), 256);
        assert!(a.allocate_offset(256).is_some());
        assert!(a.allocate_offset(1).is_none());
    }

    #[test]
    fn region_size_at_or_below_the_header_is_permanently_full() {
        // C++ arithmetic yields heap_max_ <= heap_begin_ here; the arena is
        // simply always out of memory (reached by saturation, not wraparound).
        for region in [1u64, 64, HEADER_SIZE] {
            let (backend, _n) = seg("tinyregion", 4096);
            backend.destroy();
            let a = ArenaAllocator::create_with_region(backend, next_id(), region);
            assert_eq!(a.capacity(), 0);
            assert_eq!(a.remaining_size(), 0);
            assert!(a.allocate_offset(1).is_none());
            // A zero-byte request is still satisfiable, as in the C++.
            assert!(a.allocate_offset(0).is_some());
        }
    }

    #[test]
    fn region_size_larger_than_the_segment_is_clamped() {
        let (backend, _n) = seg("clamp", 4096);
        backend.destroy();
        // C++ would take this on trust and bump straight past the mapping.
        let a = ArenaAllocator::create_with_region(backend, next_id(), 1 << 30);
        assert_eq!(a.capacity(), 4096 - HEADER_SIZE);
        assert!(a.allocate_offset(4096 - HEADER_SIZE).is_some());
        assert!(a.allocate_offset(1).is_none());
    }

    #[test]
    fn typed_alloc_is_zeroed_and_resolves_via_the_registry() {
        let a = fresh("typed", 1 << 16);
        // Reuse a region a previous allocation dirtied, to prove zeroing.
        let dirty = a.allocate_offset(64).unwrap();
        // SAFETY: live 64-byte allocation inside the mapped heap.
        unsafe { std::ptr::write_bytes(a.heap_base().add(dirty.off as usize), 0xFF, 64) };
        a.reset();

        let p: ShmPtr<u64> = a.alloc().unwrap();
        let raw = registry::resolve(p).unwrap();
        // SAFETY: freshly allocated, zeroed, 8-aligned u64 in a registered
        // segment; nothing else holds this offset.
        unsafe {
            assert_eq!(*raw, 0);
            *raw = 0xDEAD_BEEF;
            assert_eq!(*registry::resolve(p).unwrap(), 0xDEAD_BEEF);
        }
        // Two typed allocations never alias.
        let q: ShmPtr<u64> = a.alloc().unwrap();
        assert_ne!(p.off, q.off);
        assert_eq!(a.currently_allocated_size(), 16);
        // Free is a no-op; the offset is not recycled.
        a.free(p);
        let r: ShmPtr<u64> = a.alloc().unwrap();
        assert_ne!(r.off, p.off);
    }

    #[test]
    fn typed_alloc_reports_exhaustion() {
        let a = fresh("typedoom", 4096);
        assert!(a.allocate_offset(a.capacity()).is_some());
        assert!(a.alloc::<u64>().is_none());
        assert!(a.alloc::<[u64; 64]>().is_none());
    }

    #[test]
    fn open_attaches_to_live_arena_state() {
        let (backend, name) = seg("attach", 1 << 16);
        let id = next_id();
        let a = ArenaAllocator::create(backend, id);
        let p = a.allocate_offset(256).unwrap();

        // A second mapping of the same segment sees the same arena.
        let b = ArenaAllocator::open(SharedMemBackend::open(&name, 1 << 16).unwrap(), id).unwrap();
        assert_eq!(b.heap_offset(), a.heap_offset());
        assert_eq!(b.capacity(), a.capacity());
        assert_eq!(b.currently_allocated_size(), 256);

        // Bumping through one handle is visible through the other.
        let q = b.allocate_offset(64).unwrap();
        assert_eq!(q.off, p.off + 256);
        assert_eq!(a.heap_offset(), 320);
        assert_eq!(a.currently_allocated_size(), 320);
        // ...and so is a reset.
        b.reset();
        assert_eq!(a.heap_offset(), 0);

        drop(b);
        drop(a);
        SharedMemBackend::open(&name, 1 << 16)
            .map(|s| s.destroy())
            .ok();
    }

    #[test]
    fn open_rejects_foreign_segments_and_wrong_ids() {
        let (backend, name) = seg("reject", 1 << 16);
        let id = next_id();
        let a = ArenaAllocator::create(backend, id);

        // Wrong allocator id.
        let other = AllocatorId::new(id.major, id.minor.wrapping_add(1000));
        assert!(
            ArenaAllocator::open(SharedMemBackend::open(&name, 1 << 16).unwrap(), other).is_err()
        );

        // Not an arena at all: a zeroed segment fails the magic check, and a
        // v1 free-list segment fails the arena `kind` check.
        let (blank, blank_name) = seg("blank", 4096);
        blank.destroy();
        assert!(ArenaAllocator::open(blank, id).is_err());
        let _ = blank_name;

        let (fl_seg, fl_name) = seg("freelist", 1 << 16);
        let fl_id = next_id();
        let fl = crate::FreeListAllocator::create(fl_seg, fl_id);
        assert!(
            ArenaAllocator::open(SharedMemBackend::open(&fl_name, 1 << 16).unwrap(), fl_id).is_err(),
            "a free-list segment must not be mistaken for an arena"
        );
        drop(fl);

        drop(a);
        SharedMemBackend::open(&name, 1 << 16)
            .map(|s| s.destroy())
            .ok();
        SharedMemBackend::open(&fl_name, 1 << 16)
            .map(|s| s.destroy())
            .ok();
    }

    #[test]
    fn open_rejects_a_segment_smaller_than_the_header() {
        let (tiny, _n) = seg("tiny", 64);
        tiny.destroy();
        assert!(ArenaAllocator::open(tiny, next_id()).is_err());
    }

    #[test]
    fn tls_and_sub_arena_hooks_are_noops() {
        let a = fresh("noops", 4096);
        a.create_tls();
        a.free_tls();
        assert!(a.push_arena_state(128).is_none());
        a.pop_arena_state(&ArenaState::default(), OffsetPtr::null());
        // The no-ops must not disturb the arena.
        assert_eq!(a.heap_offset(), 0);
        assert_eq!(a.remaining_size(), a.capacity());

        let st = ArenaState {
            arena_off: 0,
            arena_cur: 0,
            arena_end: 0,
        };
        assert!(!st.is_active());
        assert!(ArenaState {
            arena_off: 8,
            ..st
        }
        .is_active());
    }

    #[test]
    fn concurrent_allocations_are_disjoint_and_counted() {
        use std::sync::{Arc, Mutex};
        const THREADS: u64 = 8;
        const PER_THREAD: u64 = 200;
        const BLOCK: u64 = 64;
        let a = Arc::new(fresh("mt", 1 << 20));
        let got: Arc<Mutex<Vec<u64>>> = Arc::new(Mutex::new(Vec::new()));
        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                let a = Arc::clone(&a);
                let got = Arc::clone(&got);
                std::thread::spawn(move || {
                    let mut mine = Vec::with_capacity(PER_THREAD as usize);
                    for _ in 0..PER_THREAD {
                        let p = a.allocate_offset(BLOCK).unwrap();
                        let raw = registry::resolve(p.to_shm(a.id())).unwrap();
                        // SAFETY: a live BLOCK-byte allocation owned solely by
                        // this thread (proved disjoint below).
                        unsafe { std::ptr::write_bytes(raw, 0x77, BLOCK as usize) };
                        mine.push(p.off);
                    }
                    got.lock().unwrap().extend(mine);
                })
            })
            .collect();
        for t in handles {
            t.join().unwrap();
        }
        let mut offs = got.lock().unwrap().clone();
        assert_eq!(offs.len() as u64, THREADS * PER_THREAD);
        offs.sort_unstable();
        offs.dedup();
        assert_eq!(
            offs.len() as u64,
            THREADS * PER_THREAD,
            "an offset was handed out twice"
        );
        // Bump semantics: the offsets tile the heap exactly, with no gaps.
        for (i, off) in offs.iter().enumerate() {
            assert_eq!(*off, i as u64 * BLOCK);
        }
        let total = THREADS * PER_THREAD * BLOCK;
        assert_eq!(a.heap_offset(), total);
        assert_eq!(a.currently_allocated_size(), total);
        assert_eq!(a.remaining_size(), a.capacity() - total);
    }
}
