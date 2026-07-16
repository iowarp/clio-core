// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).
//
//! Coalescing buddy allocator — the answer MEMORY_DESIGN.md names for the v1
//! free-list allocator's fragmentation ("Coalescing is deliberately deferred
//! ... the C++ buddy allocator port is the planned answer").
//!
//! Adapted from `clio_ctp/memory/allocator/buddy_allocator.h`. Same three
//! pillars as [`crate::allocator`]: **one segment one owner**, the **frozen
//! `ShmPtr` ABI**, and **position-independence by construction** — every link
//! in the segment is a heap-relative offset and the only mutual exclusion is
//! an in-segment `AtomicU32` spinlock, so any process may map the segment at
//! any base and allocate/free correctly.
//!
//! Blocks are exact powers of two (32-byte header included). Allocation pops
//! the smallest sufficient order and splits down to the requested one; free
//! merges with the XOR buddy (`off ^ (1 << order)`) as far up as the buddy is
//! free and of the same order. The heap is seeded by a greedy largest-first
//! carve (the shape of the C++ `DivideArenaIntoPages`), so a non-power-of-two
//! segment becomes a descending run of power-of-two "root" blocks.
//!
//! # C++ → Rust name mapping
//!
//! | C++ (`buddy_allocator.h`) | Rust (this module) |
//! |---|---|
//! | `_BuddyAllocator<MODE>` / `BuddyAllocator` | [`BuddyAllocator`] |
//! | `BuddyPage<MODE>` (16 B, slist node + `size_`) | `BlockHeader` (32 B, order + dlist links + `req_size`) |
//! | `BuddyPage::size_` bit 63 / `kFreeMask` | `BlockHeader::meta` bit 63 / `META_FREE` |
//! | `MarkFree` / `MarkAllocated` / `IsFree` | `push_free` (marks free) / `write_meta(order)` / `META_FREE` test |
//! | `BuddyPage::GetSize()` | `(1 << order) - BLOCK_HEADER_SIZE`, i.e. [`BuddyAllocator::capacity_of`] |
//! | `small_pages_[10]` + `large_pages_[6]` | `BuddySegmentHeader::free_lists[order - MIN_ORDER]` (35 orders) |
//! | `kMinSize` = 32 (min data size) | [`MIN_PAYLOAD`] = 32 (= `MIN_BLOCK - BLOCK_HEADER_SIZE`) |
//! | `kMinLog2` = 5 / `kMaxLog2` = 20 | `MIN_ORDER` = 6 / `MAX_ORDER` = 40 (block, not payload, exponents) |
//! | `kSmallThreshold` / `kSmallArenaSize` / `kSmallArenaPages` | (gone — one uniform order ladder, no small/large split) |
//! | `Heap<false> big_heap_` / `small_arena_` | (gone — the carve seeds every heap byte into buddy roots) |
//! | `shm_init` / `shm_attach` | [`BuddyAllocator::create`] / [`BuddyAllocator::open`] |
//! | `AllocateOffset` | [`BuddyAllocator::alloc_bytes`] |
//! | `AllocateSmall` / `AllocateLarge` / `FindFirstFit` | (folded into `alloc_bytes` — the order lists are exact-fit) |
//! | `FreeOffset` / `FreeOffsetNoNullCheck` | [`BuddyAllocator::free_bytes`] (null check folded in) |
//! | `ReallocateOffset` | [`BuddyAllocator::realloc_bytes`] |
//! | `DivideArenaIntoPages` | `carve_region` (greedy largest-first) |
//! | `AddRemainderToFreeList` | `push_free` (no remainders exist — see divergence 6) |
//! | `FinalizeAllocation` | tail of `alloc_bytes` |
//! | `GetSmall/LargePageListIndexForAlloc/Free` | `order_for` (alloc, round up) / `BlockHeader::meta` order (free, exact) |
//! | `EmplaceToList` / `PopFromList` | `push_free` / `pop_free` / `unlink_free` |
//! | `ctp::CeilLog2` / `ctp::FloorLog2` | `ceil_log2` / `floor_log2` |
//! | `ctp::Mutex lock_` / `ScopedMutex` | in-segment `AtomicU32` + `SegmentLockGuard` |
//! | `OffsetToPage` / `PageToOffset` | `block_at` (offsets never leave the module) |
//! | `ForwardingTable` / `Entry` | [`ForwardingTable`] / [`ForwardingEntry`] |
//! | `dbg_alloc_count_` / `dbg_free_count_` / `dbg_net_bytes_` (`CTP_BUDDY_ALLOC_DEBUG`) | [`BuddyStats`] via [`BuddyAllocator::stats`] (always on) |
//! | `CoalesceBuddyPage` (rb-tree node, declared but never used in C++) | (dropped — coalescing is the XOR-buddy merge in `free_bytes`) |
//! | `MemMode::kPrivate` / `PrivateBuddyAllocator` | (dropped — see divergence 8) |
//!
//! # Semantic divergences
//!
//! 1. **The C++ "buddy" allocator does not coalesce.** It is a segregated
//!    power-of-two free-list allocator over a bump heap: `FreeOffset` refiles
//!    a page into the list for its size class and adjacent free pages are
//!    never merged (`CoalesceBuddyPage` is declared and never used; there is
//!    no buddy computation anywhere in the header). This port is a true
//!    binary buddy — split on alloc, XOR-buddy merge on free — because
//!    coalescing is the entire reason MEMORY_DESIGN.md schedules this port.
//!    It is the largest divergence and it is deliberate.
//! 2. **Segment header is buddy-specific.** Magic is `"CTP_SHMB"` (v1 is
//!    `"CTP_SHM1"`), the reserved header is 512 B (v1: 64 B) because it holds
//!    35 free-list heads instead of one `free_head`, and there is no `bump`
//!    (the carve consumes the heap up front). The v1 `SegmentHeader` cannot
//!    express per-order lists and is private to [`crate::allocator`] anyway.
//!    Conventions are identical: 64-B-aligned header at the segment base,
//!    heap starts after it, all offsets heap-relative, in-segment spinlock,
//!    registry registration, `ShmPtr` ABI. The distinct magic makes opening a
//!    v1 segment as a buddy segment (or the reverse) a clean `InvalidData`
//!    error instead of silent corruption.
//! 3. **Block geometry.** C++ pages are `sizeof(BuddyPage) + 2^k` data bytes;
//!    here a block *is* `2^order` bytes with its 32-byte header inside, so
//!    payload capacity is `2^order - 32`. The header grew from 16 B to 32 B
//!    because buddy merging must unlink a block from the *middle* of a free
//!    list (doubly-linked: `next_free` + `prev_free`), plus `req_size`.
//!    `MIN_ORDER` = 6 keeps the C++ minimum data size of 32 bytes exactly.
//! 4. **Max single allocation** is `MAX_PAYLOAD` = 2^40 − 32; larger requests
//!    return `None`. C++ serves arbitrarily large blocks from `big_heap_`'s
//!    bump pointer (its `kMaxSize` = 1 MB only caps the *size class*). Heaps
//!    larger than 1 TiB are still fully usable — they carve into several
//!    1-TiB roots; only a single >1-TiB object is refused.
//! 5. **`Expand()` is not ported.** It appends a new region to `big_heap_`;
//!    the Rust segment's size is fixed when the backend is mapped, so there
//!    is no second region to add. Growth means a new segment.
//! 6. **`Compact()` / `Compact`'s producer role are not ported** (the
//!    [`ForwardingTable`] *is*, for parity). Compaction slides live blocks
//!    left, which (a) destroys the power-of-two alignment every buddy
//!    computation depends on and (b) invalidates every `ShmPtr` already
//!    handed to other processes, which the frozen ABI gives no way to fix up
//!    (C++ can do it because `ForwardingTable::Resolve` is applied by the
//!    single owning process to its own pointers). Coalescing addresses the
//!    fragmentation `Compact()` existed for. The C++ hazards `Compact()` and
//!    `AddRemainderToFreeList` guard against (#680: undersized remainder
//!    pages, free-list migration losing tail bytes) are structurally absent
//!    here: a block's order fully determines its size and its buddy, so
//!    there are no remainders and no recorded size to lose.
//! 7. **`ArenaState` / `PushArenaState` / `PopArenaState` are not ported.**
//!    The bump-arena fast path makes `FreeOffset` a silent no-op for any
//!    offset inside the active arena, which would defeat the buddy
//!    invariants; this crate has a dedicated `arena` module for that pattern.
//! 8. **`MemMode::kPrivate` / `PrivateBuddyAllocator` are not ported.** That
//!    mode stores raw pointers in `priv_slist` nodes; pillar 3 forbids
//!    absolute pointers inside a segment. Only the shared, position-
//!    independent mode exists.
//! 9. **`shm_init`'s `region_size` / `shifted` parameters are dropped.** The
//!    Rust allocator always owns `[SEGMENT_HEADER_SIZE, backend.size())`; the
//!    `shifted`/`this_`/`data_start_` pointer-fixup machinery exists only
//!    because C++ places the allocator object *inside* the backend.
//! 10. **Double free and out-of-range free are ignored, not corrupting.**
//!     C++ re-pushes the page onto a free list (a cycle, then corruption);
//!     this port tests the `META_FREE` bit and returns. Frees of offsets
//!     outside the heap, or of blocks with an implausible order, likewise
//!     return instead of scribbling.
//! 11. **Locking**: C++ uses a blocking `ctp::Mutex` in the allocator header;
//!     this port uses the in-segment `AtomicU32` spinlock of the v1 allocator
//!     (cross-process capable; a holder crashing is recovered by the same
//!     daemon-restart machinery pillar 3 assumes). `realloc_bytes` reads the
//!     old block's order under the lock and then releases it before
//!     allocating — like C++, it is *not* atomic against a concurrent free of
//!     the same pointer.
//! 12. **Debug counters are always compiled in** (`CTP_BUDDY_ALLOC_DEBUG` is
//!     opt-in in C++): three `u64` stores under a lock already held. C++
//!     `dbg_net_bytes_` is asymmetric (adds the requested size, subtracts the
//!     page's recorded size); [`BuddyStats::live_bytes`] adds and subtracts
//!     the requested size, so it returns to 0 when everything is freed.
//! 13. **GPU (`CTP_IS_GPU` / `atomicCAS`) paths are not ported** — device
//!     allocation belongs to `ctp-gpu` (MIGRATION.md), and the CPU spinlock
//!     covers the host path the C++ `#else` branch takes.

use crate::backend::SharedMemBackend;
use crate::ptr::{AllocatorId, OffsetPtr, ShmPtr, ShmSafe, NULL_OFFSET};
use crate::registry;
use std::sync::atomic::{AtomicU32, Ordering};

/// `"CTP_SHMB"` — buddy sibling of the v1 `"CTP_SHM1"` magic.
const MAGIC: u64 = 0x4354_505f_5348_4d42;
const VERSION: u32 = 1;
/// 1 = C++, 2 = Rust (MEMORY_DESIGN.md pillar 1).
const OWNER_RUST: u32 = 2;

/// Reserved segment header; the heap starts here. 64-byte aligned like v1.
pub const SEGMENT_HEADER_SIZE: u64 = 512;
/// Per-block header, inside every block (both free and allocated).
pub const BLOCK_HEADER_SIZE: u64 = 32;

/// Smallest block order: 2^6 = 64-byte block => 32-byte payload, exactly the
/// C++ `kMinSize` minimum data size.
const MIN_ORDER: u32 = 6;
/// Largest block order: 2^40 = 1 TiB. Bounds the in-header free-list array;
/// bigger heaps simply carve into several 1-TiB roots.
const MAX_ORDER: u32 = 40;
const NUM_ORDERS: usize = (MAX_ORDER - MIN_ORDER + 1) as usize;
const MIN_BLOCK: u64 = 1 << MIN_ORDER;

/// Smallest payload a block can hold (C++ `kMinSize`). Requests below this
/// are rounded up to it.
pub const MIN_PAYLOAD: u64 = MIN_BLOCK - BLOCK_HEADER_SIZE;
/// Largest payload a single allocation can hold (see divergence 4).
pub const MAX_PAYLOAD: u64 = (1u64 << MAX_ORDER) - BLOCK_HEADER_SIZE;

/// Free flag in `BlockHeader::meta` — bit 63, mirroring C++
/// `BuddyPage::kFreeMask`.
const META_FREE: u64 = 1 << 63;
/// Block order lives in the low 8 bits of `meta`.
const META_ORDER_MASK: u64 = 0xFF;

#[repr(C)]
struct BuddySegmentHeader {
    magic: u64,
    version: u32,
    owner_impl: u32,
    alloc_major: u32,
    alloc_minor: u32,
    heap_size: u64,
    lock: u32, // accessed as AtomicU32
    _pad: u32,
    /// Head of the free list for each order; index = order - MIN_ORDER.
    /// `NULL_OFFSET` = empty. (C++ `small_pages_` + `large_pages_`.)
    free_lists: [u64; NUM_ORDERS],
    alloc_count: u64,
    free_count: u64,
    live_bytes: u64,
}

const _: () = assert!(std::mem::size_of::<BuddySegmentHeader>() as u64 <= SEGMENT_HEADER_SIZE);

/// Header of every block (C++ `BuddyPage`). Allocation header *and* free-list
/// node, like the C++ one; unlike it, the list is doubly linked so a buddy can
/// be unlinked from the middle during a merge.
#[repr(C)]
struct BlockHeader {
    /// Bit 63 = free flag (C++ `kFreeMask`); low 8 bits = order. The block
    /// spans `1 << order` bytes *including* this header.
    meta: u64,
    /// Free-list forward link; valid only while free (C++ `slist_node::next_`).
    next_free: u64,
    /// Free-list backward link; valid only while free.
    prev_free: u64,
    /// Requested payload size; valid only while allocated.
    req_size: u64,
}

const _: () = assert!(std::mem::size_of::<BlockHeader>() as u64 == BLOCK_HEADER_SIZE);
const _: () = assert!(MIN_BLOCK > BLOCK_HEADER_SIZE);

/// C++ `ctp::FloorLog2`. Returns 0 for 0 (C++ leaves it undefined).
const fn floor_log2(v: u64) -> u32 {
    if v == 0 {
        0
    } else {
        63 - v.leading_zeros()
    }
}

/// C++ `ctp::CeilLog2`.
const fn ceil_log2(v: u64) -> u32 {
    if v <= 1 {
        0
    } else {
        64 - (v - 1).leading_zeros()
    }
}

const fn valid_order(order: u32) -> bool {
    MIN_ORDER <= order && order <= MAX_ORDER
}

/// Smallest order whose payload holds `size`, or `None` when `size` exceeds
/// [`MAX_PAYLOAD`]. Mirrors C++ `GetSmallPageListIndexForAlloc`'s round-up
/// (and its `kMinSize` floor), minus the small/large split.
const fn order_for(size: u64) -> Option<u32> {
    let total = match size.checked_add(BLOCK_HEADER_SIZE) {
        Some(t) => t,
        None => return None, // u64::MAX request: saturating add would lie
    };
    let order = ceil_log2(total);
    let order = if order < MIN_ORDER { MIN_ORDER } else { order };
    if order > MAX_ORDER {
        None
    } else {
        Some(order)
    }
}

/// Snapshot of allocator state (C++ `CTP_BUDDY_ALLOC_DEBUG` counters, plus a
/// free-list walk that makes coalescing observable to tests).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BuddyStats {
    /// C++ `dbg_alloc_count_`.
    pub alloc_count: u64,
    /// C++ `dbg_free_count_`.
    pub free_count: u64,
    /// C++ `dbg_net_bytes_` (symmetric here — see divergence 12).
    pub live_bytes: u64,
    /// Total bytes on the free lists, counting whole blocks (header included).
    pub free_bytes: u64,
    /// Number of blocks on the free lists.
    pub free_blocks: u64,
    /// Size of the largest free block; 0 when the heap is exhausted.
    pub largest_free_block: u64,
}

/// A coalescing buddy allocator over one shared-memory segment.
///
/// Owns its backend mapping; registers in the process registry on
/// create/open and unregisters on drop, exactly like
/// [`crate::allocator::FreeListAllocator`], so `ShmPtr`s minted here resolve
/// through [`crate::registry::resolve`].
pub struct BuddyAllocator {
    backend: SharedMemBackend,
    id: AllocatorId,
}

impl BuddyAllocator {
    /// Initialize a fresh segment (this process becomes the owner per pillar
    /// 1) and seed its free lists. C++ `shm_init`.
    ///
    /// # Panics
    /// If the segment cannot hold the header plus one minimum block — the
    /// C++ `shm_init` returns `false` for `region_size < kMinSize`; the Rust
    /// sibling `FreeListAllocator::create` asserts, and this matches it.
    pub fn create(backend: SharedMemBackend, id: AllocatorId) -> Self {
        assert!(
            backend.size() as u64 >= SEGMENT_HEADER_SIZE + MIN_BLOCK,
            "segment too small for a buddy heap"
        );
        let heap_size = backend.size() as u64 - SEGMENT_HEADER_SIZE;
        let hdr = backend.base() as *mut BuddySegmentHeader;
        // SAFETY: freshly mapped, exclusively owned segment, large enough for
        // the header (asserted above).
        unsafe {
            hdr.write(BuddySegmentHeader {
                magic: MAGIC,
                version: VERSION,
                owner_impl: OWNER_RUST,
                alloc_major: id.major,
                alloc_minor: id.minor,
                heap_size,
                lock: 0,
                _pad: 0,
                free_lists: [NULL_OFFSET; NUM_ORDERS],
                alloc_count: 0,
                free_count: 0,
                live_bytes: 0,
            });
        }
        let me = Self { backend, id };
        me.register();
        {
            let _g = me.lock();
            // SAFETY: lock held; the heap is exclusively ours and mapped.
            unsafe { me.carve_region(heap_size) };
        }
        me
    }

    /// Attach to an existing buddy segment. C++ `shm_attach` (a no-op there —
    /// C++ trusts the caller; this validates the format first).
    pub fn open(backend: SharedMemBackend, expect_id: AllocatorId) -> std::io::Result<Self> {
        let hdr = backend.base() as *const BuddySegmentHeader;
        let size = backend.size() as u64;
        // SAFETY: mapping is at least SEGMENT_HEADER_SIZE (checked first, and
        // `&&` short-circuits before the header read).
        let ok = size >= SEGMENT_HEADER_SIZE + MIN_BLOCK
            && unsafe {
                (*hdr).magic == MAGIC
                    && (*hdr).version == VERSION
                    && (*hdr).owner_impl == OWNER_RUST
                    && (*hdr).alloc_major == expect_id.major
                    && (*hdr).alloc_minor == expect_id.minor
                    && (*hdr).heap_size == size - SEGMENT_HEADER_SIZE
            };
        if !ok {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "not a CTP_SHMB buddy segment, or id/size mismatch",
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

    pub fn id(&self) -> AllocatorId {
        self.id
    }

    pub fn backend(&self) -> &SharedMemBackend {
        &self.backend
    }

    /// Bytes of heap this allocator manages (segment minus its header).
    pub fn heap_size(&self) -> u64 {
        self.backend.size() as u64 - SEGMENT_HEADER_SIZE
    }

    /// Allocate `size` payload bytes, 16-byte aligned (blocks are at least
    /// 64-byte aligned and the 32-byte header preserves that). Returns a
    /// heap-relative offset pointer, or `None` when the heap cannot serve the
    /// request. C++ `AllocateOffset`.
    pub fn alloc_bytes(&self, size: u64) -> Option<OffsetPtr<u8>> {
        let want = order_for(size)?;
        let _g = self.lock();
        // SAFETY: lock held; every offset below comes from allocator metadata
        // or from splitting a block we just popped, so all are in-heap.
        unsafe {
            // Smallest non-empty order >= want (C++ walks its size-class
            // lists upward the same way).
            let mut order = want;
            while order <= MAX_ORDER && self.free_head(order) == NULL_OFFSET {
                order += 1;
            }
            if order > MAX_ORDER {
                return None;
            }
            let off = self.pop_free(order);
            // Split down, filing each buddy half on its order's list.
            while order > want {
                order -= 1;
                self.write_meta(off, order as u64 | META_FREE);
                self.push_free(off + (1u64 << order), order);
            }
            self.write_meta(off, want as u64); // free bit cleared = allocated
            self.write_req_size(off, size);
            self.bump_counter(HdrCounter::AllocCount, 1);
            self.add_live_bytes(size as i64);
            Some(OffsetPtr::new(off + BLOCK_HEADER_SIZE))
        }
    }

    /// Return an allocation to the heap, merging with its buddy as far up as
    /// possible. C++ `FreeOffset` (+ `FreeOffsetNoNullCheck`), plus the
    /// coalescing the C++ never had.
    ///
    /// # Safety
    /// `ptr` must have been returned by `alloc_bytes`/`alloc`/`realloc_bytes`
    /// on THIS allocator (from any process) and not freed since. A double
    /// free or a foreign offset is *detected* and ignored where it can be
    /// (divergence 10), but that is a backstop, not a license.
    pub unsafe fn free_bytes(&self, ptr: OffsetPtr<u8>) {
        if ptr.is_null() || ptr.off < BLOCK_HEADER_SIZE {
            return;
        }
        let _g = self.lock();
        // SAFETY: caller contract — a live block header precedes the payload;
        // the guards below reject offsets this allocator cannot have minted.
        unsafe {
            let heap_size = self.read_heap_size();
            let mut off = ptr.off - BLOCK_HEADER_SIZE;
            if off >= heap_size {
                return;
            }
            let meta = self.read_meta(off);
            let mut order = (meta & META_ORDER_MASK) as u32;
            // Double free / corrupt header: ignore rather than cycle a list.
            if meta & META_FREE != 0 || !valid_order(order) {
                return;
            }
            let size = 1u64 << order;
            if off & (size - 1) != 0 || off + size > heap_size {
                return;
            }
            self.add_live_bytes(-(self.read_req_size(off) as i64));
            self.bump_counter(HdrCounter::FreeCount, 1);

            // Coalesce. The buddy of a block is its offset with bit `order`
            // toggled. Merging is safe iff the buddy is in-heap, free, and of
            // the SAME order: the greedy carve gives roots strictly
            // descending orders (except a run of MAX_ORDER roots, which the
            // `order < MAX_ORDER` bound stops), so an equal-order buddy can
            // only ever be the other half of a parent we split ourselves —
            // two roots never merge into a block that was never carved.
            while order < MAX_ORDER {
                let size = 1u64 << order;
                let buddy = off ^ size;
                if buddy + size > heap_size {
                    break; // buddy would run past the heap (carve tail)
                }
                let bmeta = self.read_meta(buddy);
                if bmeta & META_FREE == 0 || (bmeta & META_ORDER_MASK) as u32 != order {
                    break; // allocated, or split into smaller blocks
                }
                self.unlink_free(buddy, order);
                off = if buddy < off { buddy } else { off };
                order += 1;
            }
            self.push_free(off, order);
        }
    }

    /// Grow (or keep) an allocation. C++ `ReallocateOffset`: a null pointer
    /// allocates; a request that still fits the block's capacity is answered
    /// in place; otherwise allocate-copy-free. On failure the original
    /// allocation is untouched and `None` is returned (C++ returns null).
    ///
    /// # Safety
    /// Same contract as [`Self::free_bytes`] for `ptr`.
    pub unsafe fn realloc_bytes(&self, ptr: OffsetPtr<u8>, new_size: u64) -> Option<OffsetPtr<u8>> {
        if ptr.is_null() {
            return self.alloc_bytes(new_size);
        }
        if ptr.off < BLOCK_HEADER_SIZE {
            return None;
        }
        let block = ptr.off - BLOCK_HEADER_SIZE;
        // Scoped: the spinlock is not reentrant, so it must be released
        // before alloc_bytes/free_bytes below.
        let old_capacity = {
            let _g = self.lock();
            // SAFETY: caller contract — live block header precedes payload.
            unsafe {
                if block >= self.read_heap_size() {
                    return None;
                }
                let meta = self.read_meta(block);
                let order = (meta & META_ORDER_MASK) as u32;
                if meta & META_FREE != 0 || !valid_order(order) {
                    return None;
                }
                let capacity = (1u64 << order) - BLOCK_HEADER_SIZE;
                if new_size <= capacity {
                    // In place, like C++ (never shrinks the block).
                    self.add_live_bytes(new_size as i64 - self.read_req_size(block) as i64);
                    self.write_req_size(block, new_size);
                    return Some(ptr);
                }
                capacity
            }
        };
        let new_ptr = self.alloc_bytes(new_size)?;
        // SAFETY: distinct live blocks; the source holds `old_capacity`
        // payload bytes and the destination holds `new_size > old_capacity`.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.heap_base().add(ptr.off as usize),
                self.heap_base().add(new_ptr.off as usize),
                old_capacity as usize,
            );
            self.free_bytes(ptr);
        }
        Some(new_ptr)
    }

    /// Payload capacity of a live allocation — always >= the requested size,
    /// since blocks are powers of two. C++ `BuddyPage::GetSize()`. Returns 0
    /// for a null/invalid pointer.
    ///
    /// # Safety
    /// `ptr` must be null or a live allocation from THIS allocator.
    pub unsafe fn capacity_of(&self, ptr: OffsetPtr<u8>) -> u64 {
        if ptr.is_null() || ptr.off < BLOCK_HEADER_SIZE {
            return 0;
        }
        let _g = self.lock();
        // SAFETY: caller contract — live block header precedes the payload.
        unsafe {
            let block = ptr.off - BLOCK_HEADER_SIZE;
            if block >= self.read_heap_size() {
                return 0;
            }
            let meta = self.read_meta(block);
            let order = (meta & META_ORDER_MASK) as u32;
            if meta & META_FREE != 0 || !valid_order(order) {
                return 0;
            }
            (1u64 << order) - BLOCK_HEADER_SIZE
        }
    }

    /// Typed allocation, zero-initialized, in the cross-process form.
    pub fn alloc<T: ShmSafe>(&self) -> Option<ShmPtr<T>> {
        let off = self.alloc_bytes(std::mem::size_of::<T>() as u64)?;
        // SAFETY: freshly allocated block of >= size_of::<T>() payload bytes,
        // 16-byte aligned (T is ShmSafe: POD, alignment <= 16 in practice for
        // the ABI types this crate carries).
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
        unsafe { self.free_bytes(OffsetPtr::new(ptr.off)) };
    }

    /// Counters plus a free-list walk (O(free blocks)); makes split/merge
    /// observable. C++ exposes only the `CTP_BUDDY_ALLOC_DEBUG` counters.
    pub fn stats(&self) -> BuddyStats {
        let _g = self.lock();
        // SAFETY: lock held; the walk follows in-heap free-list links only.
        unsafe {
            let mut free_bytes = 0u64;
            let mut free_blocks = 0u64;
            let mut largest_free_block = 0u64;
            for order in MIN_ORDER..=MAX_ORDER {
                let mut cur = self.free_head(order);
                while cur != NULL_OFFSET {
                    free_blocks += 1;
                    free_bytes += 1u64 << order;
                    if (1u64 << order) > largest_free_block {
                        largest_free_block = 1u64 << order;
                    }
                    cur = self.read_next(cur);
                }
            }
            BuddyStats {
                alloc_count: self.read_counter(HdrCounter::AllocCount),
                free_count: self.read_counter(HdrCounter::FreeCount),
                live_bytes: self.read_counter(HdrCounter::LiveBytes),
                free_bytes,
                free_blocks,
                largest_free_block,
            }
        }
    }

    // ---- internals: every fn below requires the segment lock held ----

    /// Seed the free lists: greedy largest-power-of-two-first, the shape of
    /// the C++ `DivideArenaIntoPages` (which walks size classes high to low).
    /// Each emitted block is a "root"; roots have strictly descending orders
    /// (modulo a run at MAX_ORDER) and are self-aligned, which is exactly the
    /// invariant the merge guard in `free_bytes` leans on.
    unsafe fn carve_region(&self, heap_size: u64) {
        let mut off = 0u64;
        let mut remain = heap_size;
        while remain >= MIN_BLOCK {
            let order = floor_log2(remain).min(MAX_ORDER);
            let size = 1u64 << order;
            debug_assert_eq!(off & (size - 1), 0, "greedy carve keeps roots aligned");
            // SAFETY: [off, off+size) is inside the heap by construction.
            unsafe { self.push_free(off, order) };
            off += size;
            remain -= size;
        }
        // A tail of < MIN_BLOCK bytes is unusable and stays out of the lists,
        // like the C++ arena remainder that no size class can hold.
    }

    /// Mark `off` free at `order` and push it on that order's list head
    /// (C++ `EmplaceToList`, doubly linked here).
    unsafe fn push_free(&self, off: u64, order: u32) {
        // SAFETY: caller passes an in-heap, `1 << order`-aligned block.
        unsafe {
            let head = self.free_head(order);
            self.write_meta(off, order as u64 | META_FREE);
            self.write_next(off, head);
            self.write_prev(off, NULL_OFFSET);
            if head != NULL_OFFSET {
                self.write_prev(head, off);
            }
            self.set_free_head(order, off);
        }
    }

    /// Pop the head of `order`'s list, or `NULL_OFFSET` (C++ `PopFromList`,
    /// which returns 0 — 0 is a valid heap offset here, so null is
    /// `NULL_OFFSET`).
    unsafe fn pop_free(&self, order: u32) -> u64 {
        // SAFETY: list heads are in-heap block offsets or NULL_OFFSET.
        unsafe {
            let head = self.free_head(order);
            if head != NULL_OFFSET {
                self.unlink_free(head, order);
            }
            head
        }
    }

    /// Remove a block from the middle of its free list — the operation the
    /// C++ singly-linked `slist` could not do, and the reason coalescing
    /// needs a doubly-linked list.
    unsafe fn unlink_free(&self, off: u64, order: u32) {
        // SAFETY: `off` is on `order`'s list; its links are in-heap or null.
        unsafe {
            let prev = self.read_prev(off);
            let next = self.read_next(off);
            if prev == NULL_OFFSET {
                self.set_free_head(order, next);
            } else {
                self.write_next(prev, next);
            }
            if next != NULL_OFFSET {
                self.write_prev(next, prev);
            }
        }
    }

    fn header(&self) -> *mut BuddySegmentHeader {
        self.backend.base() as *mut BuddySegmentHeader
    }

    fn heap_base(&self) -> *mut u8 {
        // SAFETY: segment >= SEGMENT_HEADER_SIZE + MIN_BLOCK (create/open).
        unsafe { self.backend.base().add(SEGMENT_HEADER_SIZE as usize) }
    }

    fn block_at(&self, off: u64) -> *mut BlockHeader {
        // SAFETY: callers pass offsets produced by this allocator's metadata,
        // all within the heap and at least 64-byte aligned.
        unsafe { self.heap_base().add(off as usize) as *mut BlockHeader }
    }

    /// In-segment spinlock (cross-process capable), as in [`crate::allocator`].
    fn lock(&self) -> SegmentLockGuard<'_> {
        // SAFETY: `lock` is within the mapped header and 4-byte aligned.
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

    unsafe fn read_heap_size(&self) -> u64 {
        // SAFETY: header is mapped; field is 8-byte aligned.
        unsafe { std::ptr::addr_of!((*self.header()).heap_size).read() }
    }

    unsafe fn free_head(&self, order: u32) -> u64 {
        // SAFETY: order is in MIN_ORDER..=MAX_ORDER, so the index is within
        // the free_lists array.
        unsafe {
            std::ptr::addr_of!((*self.header()).free_lists)
                .cast::<u64>()
                .add((order - MIN_ORDER) as usize)
                .read()
        }
    }

    unsafe fn set_free_head(&self, order: u32, off: u64) {
        // SAFETY: see `free_head`.
        unsafe {
            std::ptr::addr_of_mut!((*self.header()).free_lists)
                .cast::<u64>()
                .add((order - MIN_ORDER) as usize)
                .write(off);
        }
    }

    unsafe fn read_counter(&self, which: HdrCounter) -> u64 {
        // SAFETY: header is mapped; counters are 8-byte aligned.
        unsafe { self.counter_ptr(which).read() }
    }

    unsafe fn bump_counter(&self, which: HdrCounter, by: u64) {
        // SAFETY: header is mapped; lock held (counters are plain u64s
        // protected by the segment lock, not atomics).
        unsafe {
            let p = self.counter_ptr(which);
            p.write(p.read().wrapping_add(by));
        }
    }

    /// `live_bytes` moves by a signed delta (alloc adds, free subtracts);
    /// wrapping keeps a mismatched pair from panicking in debug builds.
    unsafe fn add_live_bytes(&self, delta: i64) {
        // SAFETY: header is mapped; lock held.
        unsafe {
            let p = self.counter_ptr(HdrCounter::LiveBytes);
            p.write(p.read().wrapping_add(delta as u64));
        }
    }

    unsafe fn counter_ptr(&self, which: HdrCounter) -> *mut u64 {
        let h = self.header();
        // SAFETY: header is mapped; each arm names a real field.
        unsafe {
            match which {
                HdrCounter::AllocCount => std::ptr::addr_of_mut!((*h).alloc_count),
                HdrCounter::FreeCount => std::ptr::addr_of_mut!((*h).free_count),
                HdrCounter::LiveBytes => std::ptr::addr_of_mut!((*h).live_bytes),
            }
        }
    }

    unsafe fn read_meta(&self, off: u64) -> u64 {
        // SAFETY: `off` is an in-heap block offset; u64 accepts any bits.
        unsafe { std::ptr::addr_of!((*self.block_at(off)).meta).read() }
    }

    unsafe fn write_meta(&self, off: u64, meta: u64) {
        // SAFETY: see `read_meta`.
        unsafe { std::ptr::addr_of_mut!((*self.block_at(off)).meta).write(meta) }
    }

    unsafe fn read_next(&self, off: u64) -> u64 {
        // SAFETY: see `read_meta`.
        unsafe { std::ptr::addr_of!((*self.block_at(off)).next_free).read() }
    }

    unsafe fn write_next(&self, off: u64, val: u64) {
        // SAFETY: see `read_meta`.
        unsafe { std::ptr::addr_of_mut!((*self.block_at(off)).next_free).write(val) }
    }

    unsafe fn read_prev(&self, off: u64) -> u64 {
        // SAFETY: see `read_meta`.
        unsafe { std::ptr::addr_of!((*self.block_at(off)).prev_free).read() }
    }

    unsafe fn write_prev(&self, off: u64, val: u64) {
        // SAFETY: see `read_meta`.
        unsafe { std::ptr::addr_of_mut!((*self.block_at(off)).prev_free).write(val) }
    }

    unsafe fn read_req_size(&self, off: u64) -> u64 {
        // SAFETY: see `read_meta`.
        unsafe { std::ptr::addr_of!((*self.block_at(off)).req_size).read() }
    }

    unsafe fn write_req_size(&self, off: u64, val: u64) {
        // SAFETY: see `read_meta`.
        unsafe { std::ptr::addr_of_mut!((*self.block_at(off)).req_size).write(val) }
    }
}

#[derive(Clone, Copy)]
enum HdrCounter {
    AllocCount,
    FreeCount,
    LiveBytes,
}

impl Drop for BuddyAllocator {
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

/// One `old_off -> new_off` mapping (C++ `ForwardingTable::Entry`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ForwardingEntry {
    pub old_off: u64,
    pub new_off: u64,
}

/// Maps old user-data offsets to new offsets after a compaction pass
/// (C++ `ForwardingTable`).
///
/// Ported for API parity and because it is pure, host-side logic; note that
/// this port has **no `Compact()` to produce one** (divergence 6) — sliding
/// live blocks would break every buddy invariant and invalidate `ShmPtr`s
/// held by other processes. Entries must be recorded in ascending `old_off`
/// order (the C++ left-to-right heap scan), because [`Self::resolve`] binary
/// searches them.
#[derive(Debug, Clone, Default)]
pub struct ForwardingTable {
    entries: Vec<ForwardingEntry>,
}

impl ForwardingTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// C++ `Record`.
    pub fn record(&mut self, old_off: u64, new_off: u64) {
        self.entries.push(ForwardingEntry { old_off, new_off });
    }

    /// Resolve a raw offset; returns `old_off` unchanged when not found
    /// (C++ `Resolve(size_t)`).
    pub fn resolve(&self, old_off: u64) -> u64 {
        let (mut lo, mut hi) = (0usize, self.entries.len());
        while lo < hi {
            let mid = (lo + hi) / 2;
            let e = self.entries[mid];
            if e.old_off == old_off {
                return e.new_off;
            }
            if e.old_off < old_off {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        old_off
    }

    /// Resolve a typed offset pointer; null is returned unchanged
    /// (C++ `Resolve(OffsetPtr<T>)`).
    pub fn resolve_ptr<T>(&self, ptr: OffsetPtr<T>) -> OffsetPtr<T> {
        if ptr.is_null() {
            return ptr;
        }
        OffsetPtr::new(self.resolve(ptr.off))
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unique AllocatorId per allocator: the registry is process-global and
    /// tests run in parallel, so a shared id would let one test's Drop
    /// unregister another test's live mapping (same reasoning as the v1
    /// allocator's test helper).
    fn next_id() -> AllocatorId {
        use std::sync::atomic::AtomicU32;
        static NEXT_MINOR: AtomicU32 = AtomicU32::new(1);
        AllocatorId::new(701, NEXT_MINOR.fetch_add(1, Ordering::Relaxed))
    }

    fn seg_name(tag: &str) -> String {
        format!("ctp_rs_buddy_{}_{}", tag, std::process::id())
    }

    /// `heap_bytes` of usable heap (segment = header + heap).
    fn fresh(tag: &str, heap_bytes: u64) -> BuddyAllocator {
        let backend =
            SharedMemBackend::create(&seg_name(tag), (SEGMENT_HEADER_SIZE + heap_bytes) as usize)
                .unwrap();
        backend.destroy(); // unlink the name eagerly (POSIX); mapping stays
        BuddyAllocator::create(backend, next_id())
    }

    #[test]
    fn log2_and_order_math_edges() {
        assert_eq!(floor_log2(0), 0); // C++ leaves this undefined; we clamp
        assert_eq!(floor_log2(1), 0);
        assert_eq!(floor_log2(63), 5);
        assert_eq!(floor_log2(64), 6);
        assert_eq!(floor_log2(65), 6);
        assert_eq!(floor_log2(u64::MAX), 63);
        assert_eq!(ceil_log2(0), 0);
        assert_eq!(ceil_log2(1), 0);
        assert_eq!(ceil_log2(2), 1);
        assert_eq!(ceil_log2(65), 7);
        assert_eq!(ceil_log2(1u64 << 40), 40);

        // Zero and sub-minimum requests floor at MIN_ORDER (C++ kMinSize).
        assert_eq!(order_for(0), Some(MIN_ORDER));
        assert_eq!(order_for(1), Some(MIN_ORDER));
        assert_eq!(order_for(MIN_PAYLOAD), Some(MIN_ORDER)); // 32 + 32 = 64
        assert_eq!(order_for(MIN_PAYLOAD + 1), Some(MIN_ORDER + 1)); // needs 128
        assert_eq!(order_for(96), Some(7)); // 96 + 32 = 128 exactly
        assert_eq!(order_for(97), Some(8));
        // Boundaries of the order ladder, and overflow/saturation.
        assert_eq!(order_for(MAX_PAYLOAD), Some(MAX_ORDER));
        assert_eq!(order_for(MAX_PAYLOAD + 1), None);
        assert_eq!(order_for(u64::MAX), None); // checked_add, no wrap
        assert_eq!(order_for(u64::MAX - 16), None);
    }

    #[test]
    fn carve_seeds_one_root_for_power_of_two_heap() {
        let a = fresh("carve_pow2", 1 << 16);
        let s = a.stats();
        assert_eq!(s.free_blocks, 1);
        assert_eq!(s.free_bytes, 1 << 16);
        assert_eq!(s.largest_free_block, 1 << 16);
        assert_eq!(s.alloc_count, 0);
        assert_eq!(s.live_bytes, 0);
    }

    #[test]
    fn carve_splits_ragged_heap_into_descending_roots_and_drops_short_tail() {
        // 64 KiB + 16 KiB + 100 B: greedy carve => roots of order 16, 14, 6,
        // and a 36-byte tail too small for any block.
        let a = fresh("carve_ragged", (1 << 16) + (1 << 14) + 100);
        let s = a.stats();
        assert_eq!(s.free_blocks, 3);
        assert_eq!(s.free_bytes, (1 << 16) + (1 << 14) + 64);
        assert_eq!(s.largest_free_block, 1 << 16);
    }

    #[test]
    fn alloc_splits_down_and_free_merges_all_the_way_back() {
        let a = fresh("split_merge", 1 << 16);
        // One 32-byte request off a 64 KiB root splits orders 15..=6, leaving
        // exactly one free buddy at each of those 10 orders.
        let p = a.alloc_bytes(32).unwrap();
        let s = a.stats();
        assert_eq!(s.free_blocks, 10);
        assert_eq!(s.free_bytes, (1 << 16) - 64);
        assert_eq!(s.largest_free_block, 1 << 15);
        assert_eq!(s.alloc_count, 1);
        assert_eq!(s.live_bytes, 32);
        // The payload is 16-byte aligned and the block is the minimum one.
        assert_eq!(p.off % 16, 0);
        // SAFETY: p is live.
        assert_eq!(unsafe { a.capacity_of(p) }, MIN_PAYLOAD);

        // Freeing it merges 6 -> 7 -> ... -> 16: the heap is whole again.
        // SAFETY: p is live and from this allocator.
        unsafe { a.free_bytes(p) };
        let s = a.stats();
        assert_eq!(s.free_blocks, 1);
        assert_eq!(s.free_bytes, 1 << 16);
        assert_eq!(s.largest_free_block, 1 << 16);
        assert_eq!(s.free_count, 1);
        assert_eq!(s.live_bytes, 0);
    }

    #[test]
    fn buddies_merge_only_pairwise_and_only_when_both_free() {
        let a = fresh("pairwise", 1 << 16);
        // Two 32 KiB-order blocks (payload 32736 => order 15) fill the root.
        let p1 = a.alloc_bytes((1 << 15) - BLOCK_HEADER_SIZE).unwrap();
        let p2 = a.alloc_bytes((1 << 15) - BLOCK_HEADER_SIZE).unwrap();
        assert_eq!(a.stats().free_blocks, 0);
        assert!(a.alloc_bytes(1).is_none()); // exhausted: nothing left at all

        // Freeing one half cannot merge — its buddy is still allocated.
        // SAFETY: p1 is live.
        unsafe { a.free_bytes(p1) };
        let s = a.stats();
        assert_eq!(s.free_blocks, 1);
        assert_eq!(s.largest_free_block, 1 << 15);
        // Freeing the second half merges the pair back into the root.
        // SAFETY: p2 is live.
        unsafe { a.free_bytes(p2) };
        let s = a.stats();
        assert_eq!(s.free_blocks, 1);
        assert_eq!(s.largest_free_block, 1 << 16);
    }

    #[test]
    fn roots_never_merge_across_carve_boundaries() {
        // 96 KiB = 2^16 + 2^15: roots (16)@0 and (15)@65536 are XOR buddies
        // at order 15/16 only by accident of arithmetic — merging them would
        // fabricate a 128 KiB block the heap does not own.
        let a = fresh("roots", (1 << 16) + (1 << 15));
        assert_eq!(a.stats().free_bytes, (1 << 16) + (1 << 15));
        let payload = (1u64 << 15) - BLOCK_HEADER_SIZE;
        let ps: Vec<_> = (0..3).map(|_| a.alloc_bytes(payload).unwrap()).collect();
        assert_eq!(a.stats().free_blocks, 0);
        for p in ps {
            // SAFETY: p is live and from this allocator.
            unsafe { a.free_bytes(p) };
        }
        let s = a.stats();
        assert_eq!(s.free_bytes, (1 << 16) + (1 << 15)); // nothing lost
        assert_eq!(s.free_blocks, 2); // the two roots, not one 96 KiB block
        assert_eq!(s.largest_free_block, 1 << 16);
        assert_eq!(s.live_bytes, 0);
    }

    #[test]
    fn exhaustion_returns_none_then_recovers_fully() {
        let a = fresh("exhaust", 1 << 16);
        // 16 blocks of order 12 (4 KiB) exactly fill a 64 KiB heap.
        let payload = (1u64 << 12) - BLOCK_HEADER_SIZE;
        let mut live = Vec::new();
        for _ in 0..16 {
            live.push(a.alloc_bytes(payload).unwrap());
        }
        assert!(a.alloc_bytes(payload).is_none());
        assert!(a.alloc_bytes(1).is_none()); // not even a minimum block
        assert_eq!(a.stats().free_bytes, 0);
        assert_eq!(a.stats().largest_free_block, 0);
        for p in live {
            // SAFETY: each p is live and from this allocator.
            unsafe { a.free_bytes(p) };
        }
        // Full coalescing: the heap is one root again and can serve the
        // largest block it ever could.
        let s = a.stats();
        assert_eq!(s.free_blocks, 1);
        assert_eq!(s.largest_free_block, 1 << 16);
        assert_eq!(s.alloc_count, 16);
        assert_eq!(s.free_count, 16);
        assert_eq!(s.live_bytes, 0);
        assert!(a.alloc_bytes((1 << 16) - BLOCK_HEADER_SIZE).is_some());
    }

    #[test]
    fn oversized_and_zero_requests() {
        let a = fresh("sizes", 1 << 16);
        // Bigger than the heap, but a legal order: no block, no panic.
        assert!(a.alloc_bytes(1 << 20).is_none());
        // Beyond MAX_PAYLOAD: rejected before touching the heap.
        assert!(a.alloc_bytes(MAX_PAYLOAD + 1).is_none());
        assert!(a.alloc_bytes(u64::MAX).is_none());
        // Zero-size gets a real, minimum-sized block (C++ kMinSize floor).
        let p = a.alloc_bytes(0).unwrap();
        // SAFETY: p is live.
        assert_eq!(unsafe { a.capacity_of(p) }, MIN_PAYLOAD);
        // SAFETY: p is live.
        unsafe { a.free_bytes(p) };
        assert_eq!(a.stats().free_bytes, 1 << 16);
    }

    #[test]
    fn allocations_do_not_overlap_and_payloads_survive() {
        let a = fresh("overlap", 1 << 18);
        let sizes = [1u64, 32, 33, 100, 1000, 4096, 16384];
        let mut ptrs = Vec::new();
        for (i, &sz) in sizes.iter().enumerate() {
            let p = a.alloc_bytes(sz).unwrap();
            let raw = registry::resolve(p.to_shm(a.id())).unwrap();
            // SAFETY: live allocation of at least `sz` bytes.
            unsafe { std::ptr::write_bytes(raw, i as u8 + 1, sz as usize) };
            ptrs.push((p, sz, i as u8 + 1));
        }
        for &(p, sz, tag) in &ptrs {
            let raw = registry::resolve(p.to_shm(a.id())).unwrap();
            // SAFETY: live allocation of at least `sz` bytes.
            let bytes = unsafe { std::slice::from_raw_parts(raw, sz as usize) };
            assert!(bytes.iter().all(|&b| b == tag), "block {tag} was clobbered");
            // Capacity is the block's payload, always >= the request.
            // SAFETY: p is live.
            assert!(unsafe { a.capacity_of(p) } >= sz);
        }
        for (p, _, _) in ptrs {
            // SAFETY: each p is live and from this allocator.
            unsafe { a.free_bytes(p) };
        }
        assert_eq!(a.stats().free_bytes, 1 << 18);
    }

    #[test]
    fn double_free_and_bogus_pointers_are_ignored() {
        let a = fresh("badfree", 1 << 16);
        let p = a.alloc_bytes(64).unwrap();
        // SAFETY: p is live.
        unsafe { a.free_bytes(p) };
        let after_first = a.stats();
        // Second free is detected via the META_FREE bit and ignored, instead
        // of cycling the free list like the C++ would.
        // SAFETY: exercising the documented backstop for a double free.
        unsafe { a.free_bytes(p) };
        assert_eq!(a.stats(), after_first);
        // Null, sub-header, and out-of-heap offsets are no-ops.
        // SAFETY: exercising the documented guards.
        unsafe {
            a.free_bytes(OffsetPtr::null());
            a.free_bytes(OffsetPtr::new(0));
            a.free_bytes(OffsetPtr::new(BLOCK_HEADER_SIZE - 1));
            a.free_bytes(OffsetPtr::new(1 << 20));
        }
        assert_eq!(a.stats(), after_first);
        // SAFETY: null/invalid pointers report zero capacity.
        unsafe {
            assert_eq!(a.capacity_of(OffsetPtr::null()), 0);
            assert_eq!(a.capacity_of(OffsetPtr::new(1 << 20)), 0);
        }
    }

    #[test]
    fn realloc_in_place_then_moves_and_copies() {
        let a = fresh("realloc", 1 << 16);
        // Null pointer allocates (C++ ReallocateOffset).
        // SAFETY: null input is explicitly allowed.
        let p = unsafe { a.realloc_bytes(OffsetPtr::null(), 100).unwrap() };
        let raw = registry::resolve(p.to_shm(a.id())).unwrap();
        // SAFETY: live 100-byte allocation.
        unsafe { std::ptr::write_bytes(raw, 0xC3, 100) };

        // 100 bytes lives in an order-8 block (capacity 224): growing within
        // capacity keeps the same block, like C++.
        // SAFETY: p is live.
        let same = unsafe { a.realloc_bytes(p, 200).unwrap() };
        assert_eq!(same.off, p.off);
        assert_eq!(a.stats().live_bytes, 200);

        // Growing past capacity moves the block and copies the old payload.
        // SAFETY: p is live.
        let moved = unsafe { a.realloc_bytes(p, 4000).unwrap() };
        assert_ne!(moved.off, p.off);
        let raw = registry::resolve(moved.to_shm(a.id())).unwrap();
        // SAFETY: live 4000-byte allocation; the first 100 bytes came from p.
        let bytes = unsafe { std::slice::from_raw_parts(raw, 100) };
        assert!(bytes.iter().all(|&b| b == 0xC3));
        assert_eq!(a.stats().live_bytes, 4000);

        // A request past MAX_PAYLOAD fails without disturbing the original.
        // SAFETY: moved is live.
        assert!(unsafe { a.realloc_bytes(moved, MAX_PAYLOAD + 1) }.is_none());
        // SAFETY: moved is still live after the failed realloc.
        assert_eq!(
            unsafe { a.capacity_of(moved) },
            (1u64 << 12) - BLOCK_HEADER_SIZE
        );
        // SAFETY: moved is live.
        unsafe { a.free_bytes(moved) };
        assert_eq!(a.stats().free_bytes, 1 << 16);
    }

    #[test]
    fn typed_alloc_is_zeroed_and_resolves_via_registry() {
        let a = fresh("typed", 1 << 16);
        let p: ShmPtr<u64> = a.alloc().unwrap();
        let raw = registry::resolve(p).unwrap();
        // SAFETY: freshly allocated, zeroed u64 in a registered segment.
        unsafe {
            assert_eq!(*raw, 0);
            *raw = 0xDEAD_BEEF;
            assert_eq!(*registry::resolve(p).unwrap(), 0xDEAD_BEEF);
            a.free(p);
        }
        assert_eq!(a.stats().free_bytes, 1 << 16);
    }

    #[test]
    fn open_validates_the_segment_format() {
        let name = seg_name("open");
        let size = (SEGMENT_HEADER_SIZE + (1 << 16)) as usize;
        let backend = SharedMemBackend::create(&name, size).unwrap();
        let id = next_id();
        let a = BuddyAllocator::create(backend, id);

        // Wrong id is rejected.
        let b2 = SharedMemBackend::open(&name, size).unwrap();
        assert!(BuddyAllocator::open(b2, AllocatorId::new(701, 9999)).is_err());
        // Wrong mapped size is rejected (heap_size mismatch).
        let b3 = SharedMemBackend::open(&name, size - 64).unwrap();
        assert!(BuddyAllocator::open(b3, id).is_err());

        // A non-buddy (all-zero / v1-magic) segment is rejected rather than
        // misparsed: distinct magic, per divergence 2.
        let other = SharedMemBackend::create(&seg_name("open_zero"), size).unwrap();
        other.destroy();
        assert!(BuddyAllocator::open(other, id).is_err());

        drop(a);
        SharedMemBackend::create(&name, size).unwrap().destroy();
    }

    #[test]
    fn segment_contains_no_absolute_pointers() {
        // Pillar 3: every link is heap-relative, so the whole segment must be
        // free of anything that looks like this process's mapping base.
        let a = fresh("pic", 1 << 16);
        let mut live = Vec::new();
        for i in 0..32 {
            live.push(a.alloc_bytes(64 * (i % 7 + 1)).unwrap());
        }
        for p in live.drain(..).step_by(2) {
            // SAFETY: p is live and from this allocator.
            unsafe { a.free_bytes(p) };
        }
        let base = a.backend().base() as usize as u64;
        // SAFETY: the whole segment is mapped and readable.
        let words = unsafe {
            std::slice::from_raw_parts(
                a.backend().base() as *const u64,
                a.backend().size() / std::mem::size_of::<u64>(),
            )
        };
        let page = !0xFFFu64;
        assert!(
            !words.iter().any(|&w| w & page == base & page && w != 0),
            "segment stores an address inside its own mapping"
        );
    }

    #[test]
    fn concurrent_churn_keeps_the_heap_consistent() {
        use std::sync::Arc;
        const THREADS: u64 = 8;
        const ITERS: u64 = 250;
        let a = Arc::new(fresh("churn", 1 << 20));
        let handles: Vec<_> = (0..THREADS)
            .map(|t| {
                let a = Arc::clone(&a);
                std::thread::spawn(move || {
                    let mut held: Vec<(OffsetPtr<u8>, u64, u8)> = Vec::new();
                    for i in 0..ITERS {
                        // Sizes straddle several orders so threads race on
                        // split and merge, not just on one free list.
                        let sz = 1 + ((i * 37 + t * 11) % 3000);
                        let tag = (t as u8) << 4 | (i as u8 & 0xF);
                        let p = a.alloc_bytes(sz).expect("1 MiB heap, <=8 live blocks");
                        let raw = registry::resolve(p.to_shm(a.id())).unwrap();
                        // SAFETY: live allocation of at least `sz` bytes.
                        unsafe { std::ptr::write_bytes(raw, tag, sz as usize) };
                        held.push((p, sz, tag));
                        // Keep a couple live so blocks interleave, then free
                        // the oldest — this is what forces real coalescing.
                        if held.len() > 2 {
                            let (op, osz, otag) = held.remove(0);
                            let raw = registry::resolve(op.to_shm(a.id())).unwrap();
                            // SAFETY: op is still live here.
                            let bytes = unsafe { std::slice::from_raw_parts(raw, osz as usize) };
                            assert!(
                                bytes.iter().all(|&b| b == otag),
                                "another thread wrote into a live block"
                            );
                            // SAFETY: op is live and from this allocator.
                            unsafe { a.free_bytes(op) };
                        }
                    }
                    for (p, _, _) in held {
                        // SAFETY: p is live and from this allocator.
                        unsafe { a.free_bytes(p) };
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        // Every block came back and every buddy merged: the heap is one root
        // again, with no bytes lost to the churn.
        let s = a.stats();
        assert_eq!(s.alloc_count, THREADS * ITERS);
        assert_eq!(s.free_count, THREADS * ITERS);
        assert_eq!(s.live_bytes, 0);
        assert_eq!(s.free_bytes, 1 << 20);
        assert_eq!(s.free_blocks, 1);
        assert_eq!(s.largest_free_block, 1 << 20);
    }

    #[test]
    fn forwarding_table_resolves_by_binary_search() {
        let mut t = ForwardingTable::new();
        assert!(t.is_empty());
        assert_eq!(t.len(), 0);
        // Empty table: everything resolves to itself.
        assert_eq!(t.resolve(42), 42);
        // Entries are recorded in ascending old_off order (C++ heap scan).
        for i in 0..64u64 {
            t.record(i * 128, i * 64);
        }
        assert_eq!(t.len(), 64);
        assert!(!t.is_empty());
        assert_eq!(t.resolve(0), 0); // first
        assert_eq!(t.resolve(128), 64);
        assert_eq!(t.resolve(63 * 128), 63 * 64); // last
        assert_eq!(t.resolve(129), 129); // miss: unchanged
        assert_eq!(t.resolve(u64::MAX), u64::MAX); // past the end

        // Typed form: null passes through untouched.
        let null = OffsetPtr::<u64>::null();
        assert!(t.resolve_ptr(null).is_null());
        assert_eq!(t.resolve_ptr(OffsetPtr::<u64>::new(128)).off, 64);
    }
}
