// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).
//
//! Slab allocation, ported from two C++ headers:
//!
//! * `memory/allocator/slab_allocator.h` — `PrivateSlabAllocator`, an O(1)
//!   fixed-size-class allocator over a shared-memory segment: a bump
//!   pointer for cold space plus a per-class free ring (fast path) backed
//!   by a per-class intrusive free stack (overflow path). Every allocation
//!   carries a 16-byte header holding its data size, so `free` needs no
//!   lookup.
//! * `memory/allocator/slab_cache_allocator.h` — `SlabAllocator<T, AllocT>`,
//!   a per-thread cache of freed fixed-size regions in front of a backing
//!   allocator, ported here as [`SlabCache`] + [`SlabCacheCtx`].
//!
//! The segment layout follows `MEMORY_DESIGN.md`: a 64-byte segment header,
//! then the heap. Offsets handed out are **heap-relative**, so a [`ShmPtr`]
//! built from one resolves through [`crate::registry`] exactly like a
//! [`crate::FreeListAllocator`] pointer. The slab's own control block lives
//! at heap offset 0 (the analogue of the C++ allocator header living at
//! `this_` inside the backend), and the bump pointer starts after it.
//!
//! Size classes (payload bytes, header excluded): 32, 64, 128, 256, 512,
//! 1024, 2048, 4096. Requests above 4096 bump-allocate and are **not**
//! recycled (leaked on free) — faithful to the C++, which accepts this for
//! short-lived GPU scratch buffers.
//!
//! # C++ → Rust name mapping
//!
//! | C++ (`ctp::ipc`) | Rust |
//! |---|---|
//! | `PrivateSlabAllocator` | [`SlabAllocator`] |
//! | `PrivateSlabAllocator::SlabHeader` | `SlabBlockHeader` (private) |
//! | `SlabHeader::kFreeMask` / `MarkFree` / `MarkAllocated` / `IsFree` / `GetSize` | `FREE_MASK` / `mark_free` / `mark_allocated` / `is_free` / `size` |
//! | `kHeaderSize` (16) | `BLOCK_HEADER_SIZE` |
//! | `kNumClasses` (8) | `NUM_CLASSES` |
//! | `kMinLog2` (5) | `MIN_LOG2` |
//! | `kMaxClassSize` (4096) | `MAX_CLASS_SIZE` |
//! | `kRingCap` (1024) | `RING_CAP` |
//! | `FreeRing` (`buf_`, `head_`, `tail_`) | `FreeRing` (`buf_off`, `head`, `tail`) |
//! | `FreeNode { FreeNode *next_; }` | intrusive `u64` next-**offset** written into the payload (`free_next_slot`) |
//! | `free_lists_[]`, `free_rings_[]`, `bump_`, `bump_end_`, `base_` | `SlabControl { free_lists, rings, bump, bump_end }`, in-segment; no `base_` |
//! | `shm_init(backend, region_size)` | [`SlabAllocator::create`] / [`SlabAllocator::create_in_region`] / [`SlabAllocator::open`] |
//! | `AllocateOffset(size)` | [`SlabAllocator::alloc_bytes`] |
//! | `FreeOffset(p)` | [`SlabAllocator::free_bytes`] |
//! | `FreeOffsetNoNullCheck(p)` | [`SlabAllocator::free_bytes_no_null_check`] |
//! | `ReallocateOffsetNoNullCheck(p, n)` | [`SlabAllocator::realloc_bytes`] (always `None`) |
//! | `GetSizeClass(size)` | `size_class(size) -> Option<usize>` |
//! | `ClassToSize(cls)` | `class_to_size(cls)` |
//! | `ArenaState` | [`SlabArenaState`] |
//! | `PushArenaState` / `PopArenaState` | [`SlabAllocator::push_arena_state`] / [`SlabAllocator::pop_arena_state`] |
//! | `CreateTls` / `FreeTls` | — (no-ops in C++; omitted) |
//! | `SlabAllocator<T, AllocT>` (slab_cache_allocator.h) | [`SlabCache<S>`] |
//! | `SlabTls` + `thread::ThreadLocalKey key_` | [`SlabCacheCtx`] (explicit, caller-owned) |
//! | `ext_ring_buffer<void *, MallocAllocator> regions_` | `VecDeque<R>` bounded by `cap` |
//! | `MallocAllocator` / `CTP_MALLOC` | [`MallocRegionSource`] |
//! | `AllocT::Allocate<void>(size)` / `AllocT::Free(FullPtr)` | [`RegionSource::alloc_region`] / [`RegionSource::free_region`] |
//! | `FullPtr<void>` | `S::Region` (`*mut u8`, or `OffsetPtr<u8>` for a segment source) |
//!
//! # Semantic divergences from the C++
//!
//! 1. **Position-independence.** The C++ caches `base_` (a raw pointer) *in
//!    the segment* and links free slots with absolute `FreeNode *next_`
//!    pointers, so its metadata is only valid in the process that built it.
//!    This port stores heap-relative `u64` offsets everywhere (design pillar
//!    3), so any process mapping the segment at any base is correct. Ring
//!    slots are `u64` rather than C++ `size_t` for the same reason (a 32-bit
//!    peer must agree on the layout).
//! 2. **Synchronization.** `PrivateSlabAllocator` is deliberately
//!    unsynchronized (single-thread / per-GPU-thread "private" use); its
//!    `&self`-shaped operations would be data races in Rust. This port takes
//!    the same in-segment `AtomicU32` spinlock `FreeListAllocator` uses, so
//!    [`SlabAllocator`] is `Sync` and cross-process safe. Cost: one
//!    uncontended CAS per op, which the C++ fast path does not pay.
//! 3. **Overflow.** C++ computes `(requested_size + 15) & ~15` and
//!    `bump_ + total` in wrapping `size_t`: a near-`SIZE_MAX` request wraps
//!    and silently corrupts the heap. This port uses checked arithmetic and
//!    returns `None`.
//! 4. **Sentinels.** `GetSizeClass` returns `-1` for oversized; here
//!    `Option<usize>`. `AllocateOffset` returns a null `OffsetPtr` on OOM;
//!    here `Option`. (The C++ narrows `size - 1` to 32-bit `unsigned` before
//!    `__builtin_clz`; harmless, since `> 4096` returns early. Reproduced.)
//! 5. **Segment magic.** `CTP_SLB1`, distinct from the free-list allocator's
//!    `CTP_SHM1`, so [`crate::FreeListAllocator::open`] cannot silently
//!    adopt a slab segment (the header tail means different things). The
//!    header prefix (magic/version/owner/id/heap_size/lock) is
//!    layout-identical. `open` also enforces design pillar 1 by rejecting a
//!    segment whose `owner_impl` is not the Rust implementation.
//! 6. **Zeroing.** [`SlabAllocator::alloc`] zeroes its payload, matching
//!    `FreeListAllocator::alloc`; `alloc_bytes` does not, matching C++
//!    `AllocateOffset` (recycled slots retain stale bytes in both).
//! 7. **Region clamping.** `shm_init` trusts `region_size`; this port clamps
//!    it to the mapped heap and rejects a segment too small for the control
//!    block, rather than trusting the caller with out-of-bounds writes.
//! 8. **`PushArenaState` quirks preserved.** The C++ writes `prior.arena_off_`
//!    twice (pre-alloc, then post-alloc bump) while `arena_cur_` keeps the
//!    *pre*-alloc bump, and computes an unused `block_off`. The offsets are
//!    reproduced; the unused local is dropped. On failure C++ leaves `prior`
//!    partially written — here `None` is returned and nothing is written.
//!    `PopArenaState` ignores `prior` and just frees the block, as in C++.
//! 9. **TLS → explicit context.** `SlabTls` hangs off `CTP_THREAD_MODEL`'s
//!    TLS and is created lazily per thread. `thread_local!` is banned by
//!    project rule (Windows per-DLL duplication), so the cache is a
//!    caller-owned [`SlabCacheCtx`] passed explicitly. Consequences: no lazy
//!    per-thread creation, and `&mut` proves the single-threaded access the
//!    C++ only assumes. [`SlabCache::drain_ctx`] is added because the C++
//!    silently leaks a thread's cached regions at thread exit; a dropped
//!    `SlabCacheCtx` likewise leaks unless drained (it has no handle to its
//!    source).
//! 10. **Cache overflow behavior.** `ext_ring_buffer` is
//!     `SPSC | DYNAMIC_SIZE`, but its resize is unimplemented ("would need
//!     to resize vector - not implemented for now"), so it *rejects* when
//!     full — and, unlike the `ErrorOnNoSpace` path, it does **not** roll
//!     back its speculative `tail_.fetch_add(1)`. A full C++ SlabTls
//!     therefore inflates `tail_` on every rejected free until the queue
//!     wedges permanently (`Emplace` always full, `Pop` always hits an
//!     unwritten slot), silently degrading to pass-through forever. This
//!     port implements the *intended* bounded semantics: reject when full,
//!     stay usable. Related: the C++ ring reserves one sentinel slot, so a
//!     `cap` of 1024 caches 1023 regions; [`SlabCacheCtx`] caches `cap`.
//!     (The in-segment `FreeRing` keeps its sentinel: `RING_CAP` 1024 → 1023
//!     entries, since that ring's modular arithmetic is ported verbatim.)
//! 11. **Uniform region size.** C++ `Allocate(size)` lets a caller pull a
//!     region of arbitrary `size` from the *same* per-thread cache that
//!     serves `slab_size` requests; freeing a smaller-than-`slab_size`
//!     region caches it and the next default `Allocate()` hands it back —
//!     a latent overflow. This port caches only `slab_size` regions and
//!     exposes no sized override.
//! 12. **`RegionSource::free_region` takes the size.** Rust's `Layout`
//!     requires it at deallocation; C++ `MallocAllocator::Free` recovers it
//!     from its own block header. Only regions of `slab_size` are cached, so
//!     the cache always knows it.
//! 13. **Null regions.** C++ `SlabAllocator::Free` skips `nullptr`; a
//!     generic `S::Region` has no null, so the check is absent here.
//!
//! Not ported (no C++ behavior to carry): `CreateTls`/`FreeTls` (empty).
//! [`SlabAllocator::bump_remaining`], [`SlabAllocator::ring_enabled`],
//! [`SlabAllocator::owner_impl`] and [`SlabAllocator::heap_size`] are
//! additions for diagnostics and tests. [`SlabAllocator::is_free_slot`]
//! re-exposes C++ `SlabHeader::IsFree()`, whose header struct is public in
//! C++ but private here.

use crate::backend::SharedMemBackend;
use crate::ptr::{AllocatorId, OffsetPtr, ShmPtr, ShmSafe, NULL_OFFSET};
use crate::registry;
use std::collections::VecDeque;
use std::io;
use std::sync::atomic::{AtomicU32, Ordering};

/// Segment magic, "CTP_SLB1" (cf. `CTP_SHM1` for the free-list allocator).
const MAGIC: u64 = 0x4354_505f_534c_4231;
const VERSION: u32 = 1;
/// `owner_impl` discriminant for the Rust implementation (design pillar 1).
const OWNER_RUST: u32 = 2;
/// Segment header size, 64-byte aligned; the heap (and `SlabControl`) starts here.
const SEGMENT_HEADER_SIZE: u64 = 64;

/// C++ `kHeaderSize`: `SlabHeader` padded to 16 for alignment.
const BLOCK_HEADER_SIZE: u64 = 16;
/// C++ `kNumClasses`.
const NUM_CLASSES: usize = 8;
/// C++ `kMinLog2` — log2 of the smallest class (32).
const MIN_LOG2: u32 = 5;
/// Smallest size class, `1 << MIN_LOG2`.
const MIN_CLASS_SIZE: u64 = 1 << MIN_LOG2;
/// C++ `kMaxClassSize`.
const MAX_CLASS_SIZE: u64 = 4096;
/// C++ `kRingCap`. One slot is a sentinel, so a ring holds `RING_CAP - 1`.
const RING_CAP: u64 = 1024;
/// Payload alignment (C++ rounds every request to 16).
const ALIGN: u64 = 16;

/// Bit 63 of `SlabBlockHeader::size` marks a slot free (C++ `kFreeMask`).
const FREE_MASK: u64 = 1 << 63;

/// Segment header. Prefix-compatible with the free-list allocator's, but
/// carries no free-list/bump fields: the slab keeps those in `SlabControl`.
#[repr(C)]
struct SegmentHeader {
    magic: u64,
    version: u32,
    owner_impl: u32,
    alloc_major: u32,
    alloc_minor: u32,
    heap_size: u64,
    lock: u32, // accessed as AtomicU32
    _pad: u32,
}

const _: () = assert!(std::mem::size_of::<SegmentHeader>() as u64 <= SEGMENT_HEADER_SIZE);

/// Per-allocation header (C++ `SlabHeader`, padded to `kHeaderSize`).
///
/// While the slot is free, the first 8 bytes of the *payload* are reused as
/// the intrusive next-offset link — exactly as C++ reuses them as `FreeNode`.
#[repr(C)]
struct SlabBlockHeader {
    /// Payload size excluding this header; bit 63 = free flag.
    size: u64,
    _pad: u64,
}

const _: () = assert!(std::mem::size_of::<SlabBlockHeader>() as u64 == BLOCK_HEADER_SIZE);

impl SlabBlockHeader {
    /// C++ `Init(data_size)` — stores the size, allocated (free bit clear).
    fn init(&mut self, data_size: u64) {
        self.size = data_size;
        self._pad = 0;
    }

    /// C++ `GetSize()` — size with the free flag masked off.
    fn size(&self) -> u64 {
        self.size & !FREE_MASK
    }

    /// C++ `MarkFree()`.
    fn mark_free(&mut self) {
        self.size |= FREE_MASK;
    }

    /// C++ `MarkAllocated()`.
    fn mark_allocated(&mut self) {
        self.size &= !FREE_MASK;
    }

    /// C++ `IsFree()`.
    fn is_free(&self) -> bool {
        (self.size & FREE_MASK) != 0
    }
}

/// C++ `FreeRing`: a fixed circular buffer of freed payload offsets, one per
/// size class, stored in the slab-managed region. `buf_ == nullptr` (ring
/// disabled, no room reserved) is `buf_off == NULL_OFFSET` here.
#[repr(C)]
struct FreeRing {
    buf_off: u64,
    head: u64,
    tail: u64,
}

impl FreeRing {
    /// C++ `Empty()`.
    fn is_empty(&self) -> bool {
        self.head == self.tail
    }

    /// C++ `Full()` — one slot is kept as a sentinel.
    fn is_full(&self) -> bool {
        ((self.tail + 1) % RING_CAP) == self.head
    }

    fn enabled(&self) -> bool {
        self.buf_off != NULL_OFFSET
    }
}

/// The slab's in-segment control block, at heap offset 0. Holds what the C++
/// keeps in the `PrivateSlabAllocator` object itself — minus `base_`, which
/// must never live in a segment (pillar 3).
#[repr(C)]
struct SlabControl {
    /// Per-class intrusive free stacks: head payload offset, `NULL_OFFSET` = empty.
    free_lists: [u64; NUM_CLASSES],
    /// Per-class fixed ring buffers.
    rings: [FreeRing; NUM_CLASSES],
    /// Current bump offset (heap-relative).
    bump: u64,
    /// End of the managed region (heap-relative).
    bump_end: u64,
}

/// Size of the control block, rounded to `ALIGN` so the bump region — and
/// therefore every block header and payload — stays 16-aligned.
const CONTROL_SIZE: u64 = {
    let s = std::mem::size_of::<SlabControl>() as u64;
    s.next_multiple_of(ALIGN)
};

/// C++ `ArenaState` (renamed to avoid colliding with the arena module's own
/// type). Offsets are heap-relative here, backend-relative in C++.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SlabArenaState {
    /// Offset of the arena block.
    pub arena_off: u64,
    /// Current bump position within the arena.
    pub arena_cur: u64,
    /// End of the arena block.
    pub arena_end: u64,
}

// SAFETY: plain POD of u64s, valid for any bit pattern, no process-local
// addresses (heap-relative offsets only).
unsafe impl ShmSafe for SlabArenaState {}

impl SlabArenaState {
    /// C++ `ArenaState::IsActive()`.
    pub fn is_active(&self) -> bool {
        self.arena_off != 0
    }
}

/// C++ `GetSizeClass(size)`: `None` replaces the `-1` sentinel.
///
/// `cls = clamp(ceil_log2(size) - 5)`, computed branchlessly from the leading
/// zero count, exactly as the C++ does.
const fn size_class(size: u64) -> Option<usize> {
    if size > MAX_CLASS_SIZE {
        return None;
    }
    if size <= MIN_CLASS_SIZE {
        return Some(0);
    }
    // size is in 33..=4096, so `v` is in 32..=4095 and never 0 (C++ would
    // hit UB on `__builtin_clz(0)`; unreachable there for the same reason).
    let v = (size - 1) as u32;
    let log2_val = 31 - v.leading_zeros() as i32;
    // subtract (kMinLog2 - 1) = 4
    let mut cls = log2_val - 4;
    if cls < 0 {
        cls = 0;
    }
    if cls >= NUM_CLASSES as i32 {
        cls = NUM_CLASSES as i32 - 1;
    }
    Some(cls as usize)
}

/// C++ `ClassToSize(cls)` — `32 << cls`.
const fn class_to_size(cls: usize) -> u64 {
    MIN_CLASS_SIZE << cls
}

/// C++ `PrivateSlabAllocator`: O(1) slab allocator over a shared segment.
///
/// Owns its backend mapping; registers in the process registry on
/// create/open and unregisters on drop (mirroring `FreeListAllocator`).
pub struct SlabAllocator {
    backend: SharedMemBackend,
    id: AllocatorId,
}

impl SlabAllocator {
    /// C++ `shm_init(backend, 0)`: initialize a fresh segment over the whole
    /// heap. This process/language becomes the owner (design pillar 1).
    pub fn create(backend: SharedMemBackend, id: AllocatorId) -> io::Result<Self> {
        Self::create_in_region(backend, id, 0)
    }

    /// C++ `shm_init(backend, region_size)`: `region_size == 0` means the
    /// whole heap. Unlike C++, an over-large `region_size` is clamped to the
    /// mapping rather than trusted.
    pub fn create_in_region(
        backend: SharedMemBackend,
        id: AllocatorId,
        region_size: u64,
    ) -> io::Result<Self> {
        let total = backend.size() as u64;
        if total <= SEGMENT_HEADER_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "segment smaller than its header",
            ));
        }
        let heap_size = total - SEGMENT_HEADER_SIZE;
        let region = if region_size == 0 {
            heap_size
        } else {
            region_size.min(heap_size)
        };
        // Need the control block plus at least one minimum-class slot.
        if region < CONTROL_SIZE + BLOCK_HEADER_SIZE + MIN_CLASS_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "region too small for the slab control block",
            ));
        }

        let hdr = backend.base() as *mut SegmentHeader;
        // SAFETY: freshly mapped, exclusively owned segment, checked above to
        // exceed SEGMENT_HEADER_SIZE.
        unsafe {
            hdr.write(SegmentHeader {
                magic: MAGIC,
                version: VERSION,
                owner_impl: OWNER_RUST,
                alloc_major: id.major,
                alloc_minor: id.minor,
                heap_size,
                lock: 0,
                _pad: 0,
            });
        }

        let me = Self { backend, id };

        // SAFETY: the heap is >= CONTROL_SIZE (checked) and exclusively ours
        // until `register` publishes it below.
        unsafe {
            let c = &mut *me.control();
            for slot in c.free_lists.iter_mut() {
                *slot = NULL_OFFSET;
            }
            for ring in c.rings.iter_mut() {
                // C++ `free_rings_[i].Init(nullptr)` — disabled until reserved.
                ring.buf_off = NULL_OFFSET;
                ring.head = 0;
                ring.tail = 0;
            }
            // Bump pointer starts right after the allocator header.
            c.bump = CONTROL_SIZE;
            c.bump_end = region;

            // Reserve ring storage up-front (single-thread fast free-list).
            // This avoids pointer chasing through freed objects on GPU.
            let ring_bytes = (RING_CAP * 8).next_multiple_of(ALIGN);
            for ring in c.rings.iter_mut() {
                if c.bump + ring_bytes > c.bump_end {
                    // Out of space; rings stay disabled for remaining classes.
                    break;
                }
                ring.buf_off = c.bump;
                ring.head = 0;
                ring.tail = 0;
                c.bump += ring_bytes;
            }
        }

        me.register();
        Ok(me)
    }

    /// Attach to an existing slab segment (possibly created by another
    /// process) and register it. Validates magic, version, owner and id.
    pub fn open(backend: SharedMemBackend, expect_id: AllocatorId) -> io::Result<Self> {
        if (backend.size() as u64) <= SEGMENT_HEADER_SIZE + CONTROL_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "segment too small to be a CTP_SLB1 slab",
            ));
        }
        let hdr = backend.base() as *const SegmentHeader;
        // SAFETY: mapping is larger than the header (checked above).
        let (magic, version, owner, major, minor) = unsafe {
            (
                (*hdr).magic,
                (*hdr).version,
                (*hdr).owner_impl,
                (*hdr).alloc_major,
                (*hdr).alloc_minor,
            )
        };
        if magic != MAGIC || version != VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "not a CTP_SLB1 slab segment",
            ));
        }
        if owner != OWNER_RUST {
            // Design pillar 1: only the owning implementation mutates
            // allocator metadata.
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "slab segment is owned by another implementation",
            ));
        }
        if major != expect_id.major || minor != expect_id.minor {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "slab segment allocator id mismatch",
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
        let heap_size = self.backend.size() as u64 - SEGMENT_HEADER_SIZE;
        registry::register_allocator(self.id, self.heap_base(), heap_size);
    }

    pub fn id(&self) -> AllocatorId {
        self.id
    }

    pub fn backend(&self) -> &SharedMemBackend {
        &self.backend
    }

    /// Total heap bytes (mapping minus the segment header).
    pub fn heap_size(&self) -> u64 {
        // SAFETY: header mapped for the lifetime of the backend.
        unsafe { (*self.header()).heap_size }
    }

    /// The segment's owner discriminant (2 = Rust). See design pillar 1.
    pub fn owner_impl(&self) -> u32 {
        // SAFETY: header mapped for the lifetime of the backend.
        unsafe { (*self.header()).owner_impl }
    }

    /// C++ `SlabHeader::IsFree()` for the slot addressed by `ptr`. The C++
    /// accessor is public on the header struct; that struct is private here,
    /// so it is re-exposed as a diagnostic (nothing in the allocator itself
    /// consults it — the C++ does not either).
    ///
    /// # Safety
    /// `ptr` must address a slot produced by this allocator (live or freed).
    pub unsafe fn is_free_slot(&self, ptr: OffsetPtr<u8>) -> bool {
        let _g = self.lock();
        // SAFETY: caller contract — a block header precedes the payload.
        unsafe { (*self.block_header(ptr.off)).is_free() }
    }

    /// Whether class `cls` got ring storage reserved at init. Classes whose
    /// rings did not fit fall back to the intrusive free stack (C++ breaks
    /// out of the reservation loop on the first class that does not fit).
    pub fn ring_enabled(&self, cls: usize) -> bool {
        if cls >= NUM_CLASSES {
            return false;
        }
        let _g = self.lock();
        // SAFETY: control block mapped; read under the segment lock.
        unsafe { (*self.control()).rings[cls].enabled() }
    }

    /// Remaining never-bumped space (diagnostic; ignores recycled slots).
    pub fn bump_remaining(&self) -> u64 {
        let _g = self.lock();
        // SAFETY: control block mapped; read under the segment lock.
        unsafe {
            let c = &*self.control();
            c.bump_end - c.bump
        }
    }

    /// C++ `AllocateOffset(requested_size)`.
    ///
    /// Fast path: pop the size class's ring, else its intrusive free stack.
    /// Slow path: bump. Returns `None` on exhaustion (C++: null `OffsetPtr`)
    /// or when the rounded size overflows (C++: silent wrap).
    pub fn alloc_bytes(&self, requested_size: u64) -> Option<OffsetPtr<u8>> {
        // Round up to minimum and align to 16 bytes.
        let want = requested_size
            .max(MIN_CLASS_SIZE)
            .checked_add(ALIGN - 1)?
            & !(ALIGN - 1);

        let cls = size_class(want);

        let _g = self.lock();
        // SAFETY: control block and heap are mapped; all metadata mutation
        // happens under the segment lock. Offsets read from the metadata
        // were produced by this allocator and are in-bounds by construction.
        unsafe {
            let c = &mut *self.control();

            // Fast path: pop from free list (most common after warmup).
            if let Some(cls) = cls {
                if c.rings[cls].enabled() {
                    if let Some(off) = self.ring_pop(&mut c.rings[cls]) {
                        (*self.block_header(off)).mark_allocated();
                        return Some(OffsetPtr::new(off));
                    }
                }
                let node_off = c.free_lists[cls];
                if node_off != NULL_OFFSET {
                    c.free_lists[cls] = *self.free_next_slot(node_off);
                    (*self.block_header(node_off)).mark_allocated();
                    return Some(OffsetPtr::new(node_off));
                }
            }

            // Slow path: bump allocate.
            let slot_data_size = match cls {
                Some(cls) => class_to_size(cls),
                None => want,
            };
            let total = BLOCK_HEADER_SIZE.checked_add(slot_data_size)?;
            if c.bump.checked_add(total)? > c.bump_end {
                return None; // Out of memory
            }

            let hdr_off = c.bump;
            (*self.header_at(hdr_off)).init(slot_data_size);
            let data_off = hdr_off + BLOCK_HEADER_SIZE;
            c.bump += total;

            Some(OffsetPtr::new(data_off))
        }
    }

    /// C++ `FreeOffsetNoNullCheck(offset)`: read the size from the header,
    /// compute the class, push to the ring (fast path) or the intrusive
    /// stack. Oversized allocations (no class) are leaked, as in C++.
    ///
    /// # Safety
    /// `ptr` must have been returned by `alloc_bytes`/`alloc` on THIS
    /// allocator (from any process) and not freed since. Unlike
    /// [`Self::free_bytes`] this performs no null or range check.
    pub unsafe fn free_bytes_no_null_check(&self, ptr: OffsetPtr<u8>) {
        // SAFETY: forwarded caller contract — a live block header precedes
        // the payload at `ptr`.
        unsafe { self.free_locked(ptr.off) }
    }

    /// C++ `FreeOffset(offset)` — null-safe.
    ///
    /// # Safety
    /// Same contract as [`Self::free_bytes_no_null_check`], except that null
    /// is accepted and ignored.
    pub unsafe fn free_bytes(&self, ptr: OffsetPtr<u8>) {
        if ptr.is_null() {
            return;
        }
        // Defensive range check (not in C++): a payload always sits after
        // the control block and its own header, and inside the heap.
        if ptr.off < CONTROL_SIZE + BLOCK_HEADER_SIZE || ptr.off >= self.heap_size() {
            return;
        }
        // SAFETY: forwarded caller contract.
        unsafe { self.free_locked(ptr.off) }
    }

    /// # Safety
    /// `off` must be a live payload offset from this allocator.
    unsafe fn free_locked(&self, off: u64) {
        let _g = self.lock();
        // SAFETY: caller contract — a live block header precedes the payload;
        // metadata mutation is serialized by the segment lock.
        unsafe {
            let c = &mut *self.control();
            let hdr = self.block_header(off);
            let data_size = (*hdr).size();
            (*hdr).mark_free();

            let Some(cls) = size_class(data_size) else {
                // For oversized allocations, memory is leaked. This is
                // acceptable for GPU scratch buffers that are short-lived.
                return;
            };

            // Prefer the ring buffer (single-thread fast path).
            if c.rings[cls].enabled() && self.ring_push(&mut c.rings[cls], off) {
                return;
            }

            // Fallback: push to the free list — reuse the payload's first 8
            // bytes as the next link (C++ reuses them as a `FreeNode`).
            *self.free_next_slot(off) = c.free_lists[cls];
            c.free_lists[cls] = off;
        }
    }

    /// C++ `ReallocateOffsetNoNullCheck` — not supported; always `None`
    /// (C++ returns a null `OffsetPtr`). Allocate new and copy manually.
    pub fn realloc_bytes(&self, _ptr: OffsetPtr<u8>, _new_size: u64) -> Option<OffsetPtr<u8>> {
        None
    }

    /// Typed allocation, zero-initialized, in the cross-process form.
    /// (Zeroing is this port's addition — see divergence 6.)
    pub fn alloc<T: ShmSafe>(&self) -> Option<ShmPtr<T>> {
        let off = self.alloc_bytes(std::mem::size_of::<T>() as u64)?;
        // SAFETY: freshly allocated slot of at least size_of::<T>() bytes,
        // 16-aligned, exclusively owned by this caller.
        unsafe {
            std::ptr::write_bytes(
                self.heap_base().add(off.off as usize),
                0,
                std::mem::size_of::<T>(),
            );
        }
        Some(off.to_shm(self.id).cast())
    }

    /// Free a typed allocation.
    ///
    /// # Safety
    /// Same contract as [`Self::free_bytes`].
    pub unsafe fn free<T>(&self, ptr: ShmPtr<T>) {
        // SAFETY: forwarded caller contract.
        unsafe { self.free_bytes(OffsetPtr::new(ptr.off)) }
    }

    /// C++ `PushArenaState(prior, block, size)` — the bump-allocator subset.
    /// Returns the prior state and the block, or `None` where C++ returns
    /// `false` (allocation failed).
    pub fn push_arena_state(&self, size: u64) -> Option<(SlabArenaState, OffsetPtr<u8>)> {
        let (bump, bump_end) = {
            let _g = self.lock();
            // SAFETY: control block mapped; read under the segment lock.
            unsafe {
                let c = &*self.control();
                (c.bump, c.bump_end)
            }
        };
        let mut prior = SlabArenaState {
            arena_off: bump,
            arena_cur: bump,
            arena_end: bump_end,
        };
        // NOTE: not atomic w.r.t. concurrent allocations; the C++ is
        // single-threaded, and the same caveat applies to its own bump reads.
        let block = self.alloc_bytes(size)?;
        // C++ overwrites arena_off_ with the post-allocation bump ("Save
        // state after allocation") while arena_cur_ keeps the pre-allocation
        // value. Reproduced verbatim (divergence 8).
        prior.arena_off = {
            let _g = self.lock();
            // SAFETY: control block mapped; read under the segment lock.
            unsafe { (*self.control()).bump }
        };
        Some((prior, block))
    }

    /// C++ `PopArenaState(prior, block)` — frees the block; `prior` is
    /// ignored (as in C++, which never restores the bump pointer).
    ///
    /// # Safety
    /// `block` must be null or a live allocation from this allocator.
    pub unsafe fn pop_arena_state(&self, _prior: &SlabArenaState, block: OffsetPtr<u8>) {
        if !block.is_null() {
            // SAFETY: forwarded caller contract.
            unsafe { self.free_bytes(block) }
        }
    }

    // ---- internals -------------------------------------------------------

    fn header(&self) -> *mut SegmentHeader {
        self.backend.base() as *mut SegmentHeader
    }

    fn heap_base(&self) -> *mut u8 {
        // SAFETY: create/open both verify the mapping exceeds
        // SEGMENT_HEADER_SIZE, so this is one-past-the-header, in-bounds.
        unsafe { self.backend.base().add(SEGMENT_HEADER_SIZE as usize) }
    }

    fn control(&self) -> *mut SlabControl {
        // The control block occupies heap offset 0; 64-byte aligned because
        // the mapping is page-aligned and the header is 64 bytes.
        self.heap_base() as *mut SlabControl
    }

    /// Header preceding the payload at `data_off`.
    ///
    /// # Safety
    /// `data_off` must be a payload offset produced by this allocator.
    unsafe fn block_header(&self, data_off: u64) -> *mut SlabBlockHeader {
        // SAFETY: caller contract — payloads always follow their header, so
        // `data_off - BLOCK_HEADER_SIZE` is in-bounds.
        unsafe { self.heap_base().add((data_off - BLOCK_HEADER_SIZE) as usize) as *mut SlabBlockHeader }
    }

    /// Header at its own offset (bump path).
    ///
    /// # Safety
    /// `hdr_off + BLOCK_HEADER_SIZE` must be within the heap.
    unsafe fn header_at(&self, hdr_off: u64) -> *mut SlabBlockHeader {
        // SAFETY: caller contract.
        unsafe { self.heap_base().add(hdr_off as usize) as *mut SlabBlockHeader }
    }

    /// The intrusive next-link stored in a free slot's payload (C++
    /// `FreeNode::next_`, as an offset rather than a pointer).
    ///
    /// # Safety
    /// `data_off` must be a payload offset of a slot of at least 8 bytes
    /// (guaranteed: the smallest class is 32).
    unsafe fn free_next_slot(&self, data_off: u64) -> *mut u64 {
        // SAFETY: caller contract; payloads are 16-aligned so the u64 is
        // well-aligned.
        unsafe { self.heap_base().add(data_off as usize) as *mut u64 }
    }

    /// # Safety
    /// `ring` must be enabled and its storage reserved by `create_in_region`.
    unsafe fn ring_slot(&self, buf_off: u64, idx: u64) -> *mut u64 {
        // SAFETY: caller contract — `idx < RING_CAP` and the ring's
        // RING_CAP * 8 bytes were reserved inside the heap at init.
        unsafe { self.heap_base().add((buf_off + idx * 8) as usize) as *mut u64 }
    }

    /// C++ `FreeRing::Pop`.
    ///
    /// # Safety
    /// `ring` must be enabled (storage reserved).
    unsafe fn ring_pop(&self, ring: &mut FreeRing) -> Option<u64> {
        if ring.is_empty() {
            return None;
        }
        // SAFETY: head < RING_CAP and the ring's storage is reserved.
        let off = unsafe { *self.ring_slot(ring.buf_off, ring.head) };
        ring.head = (ring.head + 1) % RING_CAP;
        Some(off)
    }

    /// C++ `FreeRing::Push`.
    ///
    /// # Safety
    /// `ring` must be enabled (storage reserved).
    unsafe fn ring_push(&self, ring: &mut FreeRing, off: u64) -> bool {
        if ring.is_full() {
            return false;
        }
        // SAFETY: tail < RING_CAP and the ring's storage is reserved.
        unsafe { *self.ring_slot(ring.buf_off, ring.tail) = off };
        ring.tail = (ring.tail + 1) % RING_CAP;
        true
    }

    /// In-segment spinlock (cross-process capable). See divergence 2 — the
    /// C++ `PrivateSlabAllocator` has no equivalent.
    fn lock(&self) -> SegmentLockGuard<'_> {
        // SAFETY: `lock` lives in the mapped header and is 4-aligned; it is
        // only ever touched through this atomic view after `create` writes it.
        let atom = unsafe { AtomicU32::from_ptr(std::ptr::addr_of_mut!((*self.header()).lock)) };
        while atom
            .compare_exchange_weak(0, 1, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            std::hint::spin_loop();
            std::thread::yield_now();
        }
        SegmentLockGuard { atom }
    }
}

impl Drop for SlabAllocator {
    fn drop(&mut self) {
        registry::unregister_allocator(self.id);
    }
}

struct SegmentLockGuard<'a> {
    atom: &'a AtomicU32,
}

impl Drop for SegmentLockGuard<'_> {
    fn drop(&mut self) {
        self.atom.store(0, Ordering::Release);
    }
}

// ---------------------------------------------------------------------------
// slab_cache_allocator.h — SlabAllocator<T, AllocT> / SlabTls
// ---------------------------------------------------------------------------

/// Backing allocator for [`SlabCache`] (C++ `AllocT`, default
/// `MallocAllocator`). Mirrors the `Allocate<void>(size)` / `Free(FullPtr)`
/// surface the C++ template requires.
pub trait RegionSource {
    /// C++ `FullPtr<void>`.
    type Region: Copy;

    /// C++ `AllocT::Allocate<void>(size)`; `None` on failure.
    fn alloc_region(&self, size: usize) -> Option<Self::Region>;

    /// C++ `AllocT::Free(region)`.
    ///
    /// # Safety
    /// `region` must have been returned by [`Self::alloc_region`] on THIS
    /// source for exactly `size` bytes, and not freed since.
    unsafe fn free_region(&self, region: Self::Region, size: usize);
}

/// C++ `MallocAllocator` / `CTP_MALLOC`: the process heap.
#[derive(Debug, Clone, Copy)]
pub struct MallocRegionSource {
    align: usize,
}

impl Default for MallocRegionSource {
    fn default() -> Self {
        Self::new()
    }
}

impl MallocRegionSource {
    /// 16-byte alignment, matching the C++ allocators' rounding discipline.
    pub const fn new() -> Self {
        Self { align: ALIGN as usize }
    }

    /// `align` must be a non-zero power of two, else allocation returns `None`.
    pub const fn with_align(align: usize) -> Self {
        Self { align }
    }

    fn layout(&self, size: usize) -> Option<std::alloc::Layout> {
        std::alloc::Layout::from_size_align(size, self.align).ok()
    }
}

impl RegionSource for MallocRegionSource {
    type Region = *mut u8;

    fn alloc_region(&self, size: usize) -> Option<*mut u8> {
        if size == 0 {
            // Zero-size allocation is UB in Rust; C++ malloc(0) is
            // implementation-defined. Reject.
            return None;
        }
        let layout = self.layout(size)?;
        // SAFETY: layout has non-zero size (checked above).
        let p = unsafe { std::alloc::alloc(layout) };
        if p.is_null() {
            None
        } else {
            Some(p)
        }
    }

    unsafe fn free_region(&self, region: *mut u8, size: usize) {
        let Some(layout) = self.layout(size) else {
            return;
        };
        if layout.size() == 0 {
            return;
        }
        // SAFETY: caller contract — `region` came from `alloc_region` with
        // this `size` and this source's alignment, so the layout matches.
        unsafe { std::alloc::dealloc(region, layout) }
    }
}

/// A [`SlabAllocator`] segment as a region source, so [`SlabCache`] can front
/// shared-memory slabs (the C++ template is likewise allocator-agnostic).
impl RegionSource for SlabAllocator {
    type Region = OffsetPtr<u8>;

    fn alloc_region(&self, size: usize) -> Option<OffsetPtr<u8>> {
        self.alloc_bytes(size as u64)
    }

    unsafe fn free_region(&self, region: OffsetPtr<u8>, _size: usize) {
        // SAFETY: forwarded caller contract.
        unsafe { self.free_bytes(region) }
    }
}

/// C++ `SlabTls`: one thread's cache of freed regions.
///
/// The C++ hangs this off `CTP_THREAD_MODEL`'s TLS; `thread_local!` is banned
/// by project rule, so the caller owns one of these per thread and passes it
/// explicitly. `&mut` then *proves* the single-producer/single-consumer
/// access that the C++ `ext_ring_buffer` merely assumes.
///
/// Dropping a non-empty context leaks its cached regions (as the C++ does at
/// thread exit) — call [`SlabCache::drain_ctx`] first.
pub struct SlabCacheCtx<R> {
    regions: VecDeque<R>,
    cap: usize,
}

impl<R> SlabCacheCtx<R> {
    /// Number of regions currently cached.
    pub fn len(&self) -> usize {
        self.regions.len()
    }

    /// Whether the cache holds no regions.
    pub fn is_empty(&self) -> bool {
        self.regions.is_empty()
    }

    /// Maximum regions this context will cache.
    pub fn cap(&self) -> usize {
        self.cap
    }
}

/// C++ `SlabAllocator<T, AllocT>` (slab_cache_allocator.h): a per-thread
/// cache of freed fixed-size regions in front of a backing allocator.
///
/// `alloc` reuses a cached region, falling back to the source on a miss;
/// `free` returns the region to the calling thread's cache, or to the source
/// when the cache is full. The C++ `T` template parameter (whose `sizeof`
/// supplies the default slab size) becomes the explicit `slab_size`.
pub struct SlabCache<S: RegionSource> {
    source: S,
    slab_size: usize,
    cap: usize,
}

impl<S: RegionSource> SlabCache<S> {
    /// C++ `SlabAllocator(alloc, slab_size, cap = 1024)`.
    pub fn new(source: S, slab_size: usize, cap: usize) -> Self {
        Self {
            source,
            slab_size,
            cap,
        }
    }

    /// The backing allocator (C++ `alloc_`).
    pub fn source(&self) -> &S {
        &self.source
    }

    /// C++ `slab_size_`.
    pub fn slab_size(&self) -> usize {
        self.slab_size
    }

    /// C++ `cap_` — per-thread cache capacity.
    pub fn cap(&self) -> usize {
        self.cap
    }

    /// Create a cache context for the calling thread (C++ `Tls()`'s lazy
    /// `new SlabTls(CTP_MALLOC, cap_)`, made explicit).
    pub fn new_ctx(&self) -> SlabCacheCtx<S::Region> {
        SlabCacheCtx {
            regions: VecDeque::with_capacity(self.cap),
            cap: self.cap,
        }
    }

    /// C++ `Allocate()`: reuse a cached region, else allocate `slab_size`
    /// from the backing allocator.
    pub fn alloc(&self, ctx: &mut SlabCacheCtx<S::Region>) -> Option<S::Region> {
        if let Some(region) = ctx.regions.pop_front() {
            return Some(region);
        }
        self.source.alloc_region(self.slab_size)
    }

    /// C++ `Free(region)`: cache it, or hand it back to the backing
    /// allocator when this thread's cache is full.
    ///
    /// # Safety
    /// `region` must have come from [`Self::alloc`] on a cache with the same
    /// source and `slab_size`, and not been freed since.
    pub unsafe fn free(&self, ctx: &mut SlabCacheCtx<S::Region>, region: S::Region) {
        if ctx.regions.len() < ctx.cap {
            ctx.regions.push_back(region);
            return;
        }
        // SAFETY: forwarded caller contract.
        unsafe { self.source.free_region(region, self.slab_size) }
    }

    /// Return every cached region to the backing allocator. No C++ analogue:
    /// a C++ `SlabTls` leaks its cache at thread exit.
    ///
    /// # Safety
    /// `ctx` must have been created by this cache (same source/`slab_size`).
    pub unsafe fn drain_ctx(&self, ctx: &mut SlabCacheCtx<S::Region>) {
        while let Some(region) = ctx.regions.pop_front() {
            // SAFETY: forwarded caller contract.
            unsafe { self.source.free_region(region, self.slab_size) }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// Unique AllocatorId + segment name per call: the registry is
    /// process-global and tests run in parallel, so a shared id would let one
    /// test's Drop unregister another's live mapping.
    fn fresh(tag: &str, bytes: usize) -> SlabAllocator {
        static NEXT_MINOR: AtomicU32 = AtomicU32::new(1);
        let minor = NEXT_MINOR.fetch_add(1, Ordering::Relaxed);
        let name = format!("ctp_rs_slab_{}_{}_{}", tag, std::process::id(), minor);
        let backend = SharedMemBackend::create(&name, bytes).unwrap();
        backend.destroy(); // unlink the name eagerly (POSIX); mapping stays
        SlabAllocator::create(backend, AllocatorId::new(7561, minor)).unwrap()
    }

    /// A segment big enough for all 8 rings (8 * 8 KiB) plus working space.
    fn big(tag: &str) -> SlabAllocator {
        fresh(tag, 1 << 20)
    }

    // ---- pure logic ------------------------------------------------------

    #[test]
    fn size_class_matches_cpp_table() {
        // Everything at or below the minimum class lands in class 0.
        assert_eq!(size_class(0), Some(0));
        assert_eq!(size_class(1), Some(0));
        assert_eq!(size_class(31), Some(0));
        assert_eq!(size_class(32), Some(0));
        // Boundaries: a class holds (prev, cur].
        assert_eq!(size_class(33), Some(1));
        assert_eq!(size_class(64), Some(1));
        assert_eq!(size_class(65), Some(2));
        assert_eq!(size_class(128), Some(2));
        assert_eq!(size_class(129), Some(3));
        assert_eq!(size_class(256), Some(3));
        assert_eq!(size_class(512), Some(4));
        assert_eq!(size_class(1024), Some(5));
        assert_eq!(size_class(2048), Some(6));
        assert_eq!(size_class(2049), Some(7));
        assert_eq!(size_class(4096), Some(7));
        // Above kMaxClassSize there is no class (C++ returns -1).
        assert_eq!(size_class(4097), None);
        assert_eq!(size_class(u64::MAX), None);
    }

    #[test]
    fn class_to_size_covers_its_class() {
        let expect = [32u64, 64, 128, 256, 512, 1024, 2048, 4096];
        for (cls, want) in expect.iter().enumerate() {
            assert_eq!(class_to_size(cls), *want);
        }
        // The slot for a size is never smaller than the request.
        for size in 1..=MAX_CLASS_SIZE {
            let cls = size_class(size).unwrap();
            assert!(class_to_size(cls) >= size, "class {cls} too small for {size}");
        }
    }

    #[test]
    fn block_header_free_flag_roundtrip() {
        let mut h = SlabBlockHeader { size: 0, _pad: 0 };
        h.init(4096);
        assert_eq!(h.size(), 4096);
        assert!(!h.is_free());
        h.mark_free();
        assert!(h.is_free());
        assert_eq!(h.size(), 4096, "free flag must not corrupt the size");
        h.mark_allocated();
        assert!(!h.is_free());
        assert_eq!(h.size(), 4096);
        // Idempotent, like the C++ bit ops.
        h.mark_allocated();
        assert!(!h.is_free());
    }

    #[test]
    fn shm_layout_invariants() {
        assert_eq!(std::mem::size_of::<SlabBlockHeader>() as u64, BLOCK_HEADER_SIZE);
        assert!(std::mem::size_of::<SegmentHeader>() as u64 <= SEGMENT_HEADER_SIZE);
        // Control block is 16-aligned so every header/payload after it is.
        assert_eq!(CONTROL_SIZE % ALIGN, 0);
        assert!(CONTROL_SIZE >= std::mem::size_of::<SlabControl>() as u64);
    }

    // ---- allocator core --------------------------------------------------

    #[test]
    fn create_rejects_undersized_segments() {
        let name = format!("ctp_rs_slab_tiny_{}", std::process::id());
        let backend = SharedMemBackend::create(&name, 4096).unwrap();
        backend.destroy();
        // 4096 - 64 heap < CONTROL_SIZE + header + min class? CONTROL_SIZE is
        // ~272, so this one fits; a truly tiny region must not.
        let a = SlabAllocator::create_in_region(backend, AllocatorId::new(7562, 1), 64);
        assert!(a.is_err(), "region smaller than the control block must fail");
    }

    #[test]
    fn alloc_is_class_rounded_and_recycled_via_ring() {
        let a = big("ring");
        assert!(a.ring_enabled(0), "1 MiB segment reserves every ring");
        let p1 = a.alloc_bytes(32).unwrap();
        let after_first = a.bump_remaining();
        // SAFETY: p1 is live and from this allocator.
        unsafe { a.free_bytes(p1) };
        // Recycled from the ring: no new bump space consumed.
        let p2 = a.alloc_bytes(32).unwrap();
        assert_eq!(p2.off, p1.off);
        assert_eq!(a.bump_remaining(), after_first);
        // A request of 1 still yields a 32-byte class-0 slot.
        // SAFETY: p2 is live.
        unsafe { a.free_bytes(p2) };
        let p3 = a.alloc_bytes(0).unwrap();
        assert_eq!(p3.off, p1.off);
    }

    #[test]
    fn distinct_live_allocations_never_alias() {
        let a = big("alias");
        let mut seen = HashSet::new();
        let mut ptrs = Vec::new();
        for size in [1u64, 32, 33, 64, 100, 256, 1000, 4096] {
            for _ in 0..16 {
                let p = a.alloc_bytes(size).unwrap();
                assert!(seen.insert(p.off), "offset {} handed out twice", p.off);
                ptrs.push(p);
            }
        }
        // Payload regions must not overlap their neighbours' headers either:
        // every offset is 16-aligned and past the control block.
        for p in &ptrs {
            assert_eq!(p.off % ALIGN, 0);
            assert!(p.off >= CONTROL_SIZE + BLOCK_HEADER_SIZE);
            assert!(p.off < a.heap_size());
        }
        for p in ptrs {
            // SAFETY: each pointer is live and from this allocator.
            unsafe { a.free_bytes(p) };
        }
    }

    #[test]
    fn ring_is_fifo_and_overflow_spills_to_free_list_lifo() {
        // 1023 ring slots + spill; class 0 (32B) slots are 48B each.
        let a = fresh("overflow", 1 << 18);
        let n = (RING_CAP + 76) as usize; // 1100: fills the ring, spills 77
        let ptrs: Vec<_> = (0..n).map(|_| a.alloc_bytes(32).unwrap()).collect();
        let exhausted = a.bump_remaining();

        for p in &ptrs {
            // SAFETY: live allocation from this allocator.
            unsafe { a.free_bytes(*p) };
        }

        // Re-allocating all of them touches no new bump space.
        let again: Vec<_> = (0..n).map(|_| a.alloc_bytes(32).unwrap()).collect();
        assert_eq!(a.bump_remaining(), exhausted, "recycling must not bump");

        // The ring holds RING_CAP - 1 entries (one sentinel slot), FIFO.
        let ring_n = (RING_CAP - 1) as usize;
        for i in 0..ring_n {
            assert_eq!(again[i].off, ptrs[i].off, "ring must pop FIFO at {i}");
        }
        // The spill went to the intrusive stack, so it pops LIFO.
        for (k, i) in (ring_n..n).enumerate() {
            assert_eq!(again[i].off, ptrs[n - 1 - k].off, "free list must pop LIFO");
        }
        // Nothing was lost or duplicated.
        let a_set: HashSet<u64> = ptrs.iter().map(|p| p.off).collect();
        let b_set: HashSet<u64> = again.iter().map(|p| p.off).collect();
        assert_eq!(a_set, b_set);
    }

    #[test]
    fn free_list_fallback_when_ring_storage_did_not_fit() {
        // 32 KiB heap: 8 KiB per ring means only the first few classes get
        // one; the C++ breaks out of the reservation loop on the first miss.
        let a = fresh("noring", 1 << 15);
        let disabled = (0..NUM_CLASSES).find(|c| !a.ring_enabled(*c));
        let cls = disabled.expect("a 32 KiB segment cannot reserve all 8 rings");
        // Once one class misses, every later class is disabled too.
        for c in cls..NUM_CLASSES {
            assert!(!a.ring_enabled(c), "class {c} must stay ring-less");
        }

        let size = class_to_size(cls);
        let p1 = a.alloc_bytes(size).unwrap();
        let p2 = a.alloc_bytes(size).unwrap();
        let before = a.bump_remaining();
        // SAFETY: both live and from this allocator.
        unsafe {
            a.free_bytes(p1);
            a.free_bytes(p2);
        }
        // Intrusive stack: LIFO, and no bump space consumed.
        assert_eq!(a.alloc_bytes(size).unwrap().off, p2.off);
        assert_eq!(a.alloc_bytes(size).unwrap().off, p1.off);
        assert_eq!(a.bump_remaining(), before);
    }

    #[test]
    fn free_list_link_lives_in_the_payload() {
        // The intrusive link overwrites the payload's first 8 bytes — the
        // recycled slot must still be fully writable afterwards.
        let a = fresh("intrusive", 1 << 15);
        let cls = (0..NUM_CLASSES)
            .find(|c| !a.ring_enabled(*c))
            .expect("ring-less class");
        let size = class_to_size(cls);
        let p = a.alloc_bytes(size).unwrap();
        // SAFETY: p is a live `size`-byte payload in a registered segment.
        unsafe {
            let raw = registry::resolve(p.to_shm(a.id())).unwrap();
            std::ptr::write_bytes(raw, 0xEE, size as usize);
            a.free_bytes(p);
            let q = a.alloc_bytes(size).unwrap();
            assert_eq!(q.off, p.off);
            let raw = registry::resolve(q.to_shm(a.id())).unwrap();
            std::ptr::write_bytes(raw, 0x11, size as usize);
            let bytes = std::slice::from_raw_parts(raw, size as usize);
            assert!(bytes.iter().all(|&b| b == 0x11));
        }
    }

    #[test]
    fn oversized_allocations_bypass_classes_and_leak_on_free() {
        let a = big("oversize");
        let huge = MAX_CLASS_SIZE + 16; // no size class
        assert_eq!(size_class(huge), None);
        let before = a.bump_remaining();
        let p1 = a.alloc_bytes(huge).unwrap();
        assert_eq!(before - a.bump_remaining(), BLOCK_HEADER_SIZE + huge);
        // SAFETY: live allocation.
        unsafe { a.free_bytes(p1) };
        // C++ leaks oversized frees: the next one takes fresh bump space.
        let mid = a.bump_remaining();
        let p2 = a.alloc_bytes(huge).unwrap();
        assert_ne!(p2.off, p1.off, "oversized slots are not recycled");
        assert_eq!(mid - a.bump_remaining(), BLOCK_HEADER_SIZE + huge);
    }

    #[test]
    fn exhaustion_returns_none_not_panic() {
        let a = fresh("oom", 1 << 15);
        // Larger than the whole segment.
        assert!(a.alloc_bytes(1 << 20).is_none());
        // Drain the remaining bump space with ring-less, class-sized slots.
        let mut count = 0;
        while a.alloc_bytes(4096).is_some() {
            count += 1;
            assert!(count < 1000, "should exhaust quickly");
        }
        assert!(a.alloc_bytes(4096).is_none());
        // A class-0 request may still fit in the tail; either way, no panic.
        let _ = a.alloc_bytes(32);
    }

    #[test]
    fn absurd_sizes_saturate_to_none() {
        let a = big("overflow");
        // C++ wraps `(size + 15) & ~15` here and corrupts the heap; we don't.
        assert!(a.alloc_bytes(u64::MAX).is_none());
        assert!(a.alloc_bytes(u64::MAX - 8).is_none());
        assert!(a.alloc_bytes(u64::MAX / 2).is_none());
        // The allocator is still usable afterwards.
        assert!(a.alloc_bytes(32).is_some());
    }

    #[test]
    fn free_ignores_null_and_out_of_range() {
        let a = big("nullfree");
        // SAFETY: null and out-of-range offsets are explicitly tolerated by
        // `free_bytes` (they return without touching metadata).
        unsafe {
            a.free_bytes(OffsetPtr::null());
            a.free_bytes(OffsetPtr::new(0)); // inside the control block
            a.free_bytes(OffsetPtr::new(a.heap_size())); // past the end
            a.free_bytes(OffsetPtr::new(u64::MAX - 1));
        }
        // Metadata intact.
        assert!(a.alloc_bytes(64).is_some());
    }

    #[test]
    fn free_flag_tracks_slot_state_through_recycling() {
        let a = big("freeflag");
        let p = a.alloc_bytes(64).unwrap();
        // SAFETY: p addresses a slot from this allocator throughout.
        unsafe {
            assert!(!a.is_free_slot(p), "fresh slot is allocated");
            a.free_bytes(p);
            assert!(a.is_free_slot(p), "freed slot carries the free bit");
            // Popping from the ring must clear the bit again (MarkAllocated).
            let q = a.alloc_bytes(64).unwrap();
            assert_eq!(q.off, p.off);
            assert!(!a.is_free_slot(q));
        }
        // Same for the intrusive-stack path (a ring-less class).
        let b = fresh("freeflag2", 1 << 15);
        let cls = (0..NUM_CLASSES)
            .find(|c| !b.ring_enabled(*c))
            .expect("ring-less class");
        let size = class_to_size(cls);
        let r = b.alloc_bytes(size).unwrap();
        // SAFETY: r addresses a slot from allocator `b` throughout.
        unsafe {
            assert!(!b.is_free_slot(r));
            b.free_bytes(r);
            assert!(b.is_free_slot(r));
            let s = b.alloc_bytes(size).unwrap();
            assert_eq!(s.off, r.off);
            assert!(!b.is_free_slot(s));
        }
    }

    #[test]
    fn realloc_is_unsupported() {
        let a = big("realloc");
        let p = a.alloc_bytes(64).unwrap();
        assert!(a.realloc_bytes(p, 128).is_none());
    }

    #[test]
    fn typed_alloc_is_zeroed_and_resolves_via_registry() {
        let a = big("typed");
        let p: ShmPtr<u64> = a.alloc().unwrap();
        let raw = registry::resolve(p).unwrap();
        // SAFETY: freshly allocated, zeroed u64 in a registered segment.
        unsafe {
            assert_eq!(*raw, 0);
            *raw = 0xDEAD_BEEF;
            assert_eq!(*registry::resolve(p).unwrap(), 0xDEAD_BEEF);
            a.free(p);
        }
        // Recycled slots are NOT re-zeroed by alloc_bytes (C++ parity), but
        // the typed alloc must always hand back zeroes.
        let q: ShmPtr<u64> = a.alloc().unwrap();
        assert_eq!(q.off, p.off);
        // SAFETY: freshly allocated, zeroed u64.
        unsafe { assert_eq!(*registry::resolve(q).unwrap(), 0) };
    }

    #[test]
    fn arena_push_pop_tracks_the_bump_pointer() {
        let a = big("arena");
        let start = a.bump_remaining();
        let (prior, block) = a.push_arena_state(256).unwrap();
        // arena_cur_ keeps the PRE-allocation bump; arena_off_ is overwritten
        // with the POST-allocation bump (faithful C++ quirk).
        assert!(prior.arena_off > prior.arena_cur);
        assert_eq!(prior.arena_off, prior.arena_cur + BLOCK_HEADER_SIZE + 256);
        assert_eq!(prior.arena_end, a.heap_size());
        assert!(prior.is_active());
        assert_eq!(block.off, prior.arena_cur + BLOCK_HEADER_SIZE);
        assert_eq!(start - a.bump_remaining(), BLOCK_HEADER_SIZE + 256);
        // SAFETY: `block` is live and from this allocator.
        unsafe { a.pop_arena_state(&prior, block) };
        // Pop recycles the block (C++ calls FreeOffset), so it comes back.
        assert_eq!(a.alloc_bytes(256).unwrap().off, block.off);
        // Null blocks are tolerated.
        // SAFETY: null is explicitly allowed.
        unsafe { a.pop_arena_state(&prior, OffsetPtr::null()) };
    }

    #[test]
    fn arena_push_fails_when_out_of_memory() {
        let a = fresh("arena_oom", 1 << 15);
        assert!(a.push_arena_state(1 << 20).is_none());
        assert!(!SlabArenaState::default().is_active());
    }

    // ---- cross-process / position independence ---------------------------

    #[test]
    fn second_mapping_of_the_same_segment_is_position_independent() {
        // Two independent mappings of one segment, at different base
        // addresses in this process: proves no absolute pointer is stored.
        let name = format!("ctp_rs_slab_shared_{}", std::process::id());
        let id = AllocatorId::new(7563, 1);
        let b1 = SharedMemBackend::create(&name, 1 << 20).unwrap();
        let a1 = SlabAllocator::create(b1, id).unwrap();
        let p1 = a1.alloc_bytes(64).unwrap();

        {
            let b2 = SharedMemBackend::open(&name, 1 << 20).unwrap();
            let a2 = SlabAllocator::open(b2, id).unwrap();
            assert_ne!(
                a1.backend().base(),
                a2.backend().base(),
                "distinct mappings expected"
            );
            assert_eq!(a2.owner_impl(), OWNER_RUST);

            // The second mapping sees the first's allocation state.
            let p2 = a2.alloc_bytes(64).unwrap();
            assert_ne!(p2.off, p1.off, "shared bump pointer");

            // Write through mapping 2, read through mapping 1 at the same
            // offset — offsets are the only currency.
            // SAFETY: p1/p2 are live 64-byte payloads in both mappings.
            unsafe {
                let w = a2.backend().base().add(SEGMENT_HEADER_SIZE as usize).add(p1.off as usize);
                std::ptr::write_bytes(w, 0x5A, 64);
                let r = a1.backend().base().add(SEGMENT_HEADER_SIZE as usize).add(p1.off as usize);
                assert!(std::slice::from_raw_parts(r, 64).iter().all(|&b| b == 0x5A));
                // Free through mapping 2; mapping 1 must recycle it.
                a2.free_bytes(p2);
            }
            assert_eq!(a1.alloc_bytes(64).unwrap().off, p2.off);
        }
        a1.backend().destroy();
    }

    #[test]
    fn open_rejects_foreign_and_mismatched_segments() {
        let name = format!("ctp_rs_slab_reject_{}", std::process::id());
        let b = SharedMemBackend::create(&name, 1 << 16).unwrap();
        let a = SlabAllocator::create(b, AllocatorId::new(7564, 1)).unwrap();

        // Wrong id.
        let b2 = SharedMemBackend::open(&name, 1 << 16).unwrap();
        assert!(SlabAllocator::open(b2, AllocatorId::new(7564, 99)).is_err());

        // Not a slab segment at all (zeroed magic).
        let name2 = format!("ctp_rs_slab_reject2_{}", std::process::id());
        let raw = SharedMemBackend::create(&name2, 1 << 16).unwrap();
        raw.destroy();
        assert!(SlabAllocator::open(raw, AllocatorId::new(7564, 1)).is_err());

        // Too small to hold the control block.
        let name3 = format!("ctp_rs_slab_reject3_{}", std::process::id());
        let tiny = SharedMemBackend::create(&name3, 64).unwrap();
        tiny.destroy();
        assert!(SlabAllocator::open(tiny, AllocatorId::new(7564, 1)).is_err());

        a.backend().destroy();
    }

    #[test]
    fn concurrent_alloc_free_never_hands_out_a_live_slot() {
        use std::sync::Arc;
        let a = Arc::new(big("mt"));
        let handles: Vec<_> = (0..8)
            .map(|t| {
                let a = Arc::clone(&a);
                std::thread::spawn(move || {
                    let pattern = 0x10u8 + t as u8;
                    for _ in 0..200 {
                        let p = a.alloc_bytes(64).unwrap();
                        let raw = registry::resolve(p.to_shm(a.id())).unwrap();
                        // SAFETY: a live 64-byte allocation owned solely by
                        // this thread until it is freed below.
                        unsafe {
                            std::ptr::write_bytes(raw, pattern, 64);
                            // If another thread were handed the same live
                            // slot, this read-back would see its pattern.
                            let seen = std::slice::from_raw_parts(raw, 64);
                            assert!(
                                seen.iter().all(|&b| b == pattern),
                                "slot aliased between threads"
                            );
                            a.free_bytes(p);
                        }
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        // 1600 allocations of 64B were recycled through the class-1 ring, so
        // the segment is nowhere near exhausted.
        assert!(a.bump_remaining() > (1 << 20) - 8 * 200 * 80);
    }

    // ---- slab cache (slab_cache_allocator.h) -----------------------------

    #[test]
    fn cache_reuses_freed_regions_fifo() {
        let c = SlabCache::new(MallocRegionSource::new(), 128, 4);
        let mut ctx = c.new_ctx();
        assert_eq!(c.slab_size(), 128);
        assert_eq!(c.cap(), 4);
        assert!(ctx.is_empty());

        let r1 = c.alloc(&mut ctx).unwrap();
        let r2 = c.alloc(&mut ctx).unwrap();
        assert_ne!(r1, r2);
        // SAFETY: both regions came from this cache.
        unsafe {
            c.free(&mut ctx, r1);
            c.free(&mut ctx, r2);
        }
        assert_eq!(ctx.len(), 2);
        // ext_ring_buffer pops FIFO.
        assert_eq!(c.alloc(&mut ctx).unwrap(), r1);
        assert_eq!(c.alloc(&mut ctx).unwrap(), r2);
        assert!(ctx.is_empty());
        // SAFETY: regions from this cache; drained back to the source.
        unsafe {
            c.free(&mut ctx, r1);
            c.free(&mut ctx, r2);
            c.drain_ctx(&mut ctx);
        }
        assert!(ctx.is_empty());
    }

    #[test]
    fn cached_regions_are_writable_across_reuse() {
        let c = SlabCache::new(MallocRegionSource::new(), 256, 2);
        let mut ctx = c.new_ctx();
        let r = c.alloc(&mut ctx).unwrap();
        // SAFETY: r is a live 256-byte region.
        unsafe {
            std::ptr::write_bytes(r, 0xAB, 256);
            assert!(std::slice::from_raw_parts(r, 256).iter().all(|&b| b == 0xAB));
            c.free(&mut ctx, r);
            let r2 = c.alloc(&mut ctx).unwrap();
            assert_eq!(r2, r);
            std::ptr::write_bytes(r2, 0xCD, 256);
            assert!(std::slice::from_raw_parts(r2, 256).iter().all(|&b| b == 0xCD));
            c.free(&mut ctx, r2);
            c.drain_ctx(&mut ctx);
        }
    }

    #[test]
    fn cache_overflow_returns_regions_to_the_source() {
        let c = SlabCache::new(MallocRegionSource::new(), 64, 2);
        let mut ctx = c.new_ctx();
        let regions: Vec<_> = (0..3).map(|_| c.alloc(&mut ctx).unwrap()).collect();
        // SAFETY: all three regions are live and from this cache.
        unsafe {
            for r in &regions {
                c.free(&mut ctx, *r);
            }
        }
        // Capacity 2: the third free went straight back to the source.
        assert_eq!(ctx.len(), 2);
        assert_eq!(ctx.cap(), 2);
        // SAFETY: drains the two cached regions.
        unsafe { c.drain_ctx(&mut ctx) };
        assert!(ctx.is_empty());
    }

    #[test]
    fn zero_capacity_cache_is_pass_through() {
        let c = SlabCache::new(MallocRegionSource::new(), 64, 0);
        let mut ctx = c.new_ctx();
        let r = c.alloc(&mut ctx).unwrap();
        // SAFETY: r is live and from this cache.
        unsafe { c.free(&mut ctx, r) };
        assert!(ctx.is_empty(), "cap 0 must never cache");
        let r2 = c.alloc(&mut ctx).unwrap();
        // SAFETY: r2 is live and from this cache.
        unsafe { c.free(&mut ctx, r2) };
    }

    #[test]
    fn contexts_are_independent_like_the_cpp_tls() {
        let c = SlabCache::new(MallocRegionSource::new(), 64, 8);
        let mut ctx1 = c.new_ctx();
        let mut ctx2 = c.new_ctx();
        let r = c.alloc(&mut ctx1).unwrap();
        // SAFETY: r is live and from this cache.
        unsafe { c.free(&mut ctx1, r) };
        assert_eq!(ctx1.len(), 1);
        // A different thread's cache must not see it (C++: per-thread TLS).
        assert_eq!(ctx2.len(), 0);
        let other = c.alloc(&mut ctx2).unwrap();
        assert_ne!(other, r);
        // SAFETY: both regions are live and from this cache.
        unsafe {
            c.free(&mut ctx2, other);
            c.drain_ctx(&mut ctx1);
            c.drain_ctx(&mut ctx2);
        }
    }

    #[test]
    fn cache_over_a_shared_memory_slab_segment() {
        // The C++ template is allocator-agnostic; front a slab segment.
        let c = SlabCache::new(big("cache_src"), 256, 2);
        let mut ctx = c.new_ctx();
        let r1 = c.alloc(&mut ctx).unwrap();
        assert_eq!(r1.off % ALIGN, 0);
        // SAFETY: r1 came from this cache's slab source.
        unsafe { c.free(&mut ctx, r1) };
        assert_eq!(ctx.len(), 1);
        // Served from the context cache, not the segment.
        let before = c.source().bump_remaining();
        let r2 = c.alloc(&mut ctx).unwrap();
        assert_eq!(r2.off, r1.off);
        assert_eq!(c.source().bump_remaining(), before);
        // SAFETY: r2 is live; drain returns it to the slab segment.
        unsafe {
            c.free(&mut ctx, r2);
            c.drain_ctx(&mut ctx);
        }
        // Returned to the segment's class-3 ring, so it is handed out again.
        assert_eq!(c.source().alloc_bytes(256).unwrap().off, r1.off);
    }

    #[test]
    fn malloc_source_rejects_zero_size_and_bad_align() {
        let s = MallocRegionSource::new();
        assert!(s.alloc_region(0).is_none());
        let r = s.alloc_region(32).unwrap();
        // SAFETY: r came from this source with size 32.
        unsafe { s.free_region(r, 32) };
        // Non-power-of-two alignment cannot form a Layout.
        let bad = MallocRegionSource::with_align(3);
        assert!(bad.alloc_region(32).is_none());
        // A zero-size slab cache therefore never allocates.
        let c = SlabCache::new(MallocRegionSource::new(), 0, 4);
        let mut ctx = c.new_ctx();
        assert!(c.alloc(&mut ctx).is_none());
    }
}
