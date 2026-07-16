// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).
//
//! Multi-lane ring buffer — port of
//! `clio_ctp/data_structures/ipc/multi_ring_buffer.h` (`ctp::ipc`).
//!
//! A [`MultiRingBuffer`] is a flat collection of independent ring buffers
//! addressed by `(lane_id, prio)`, laid out row-major: the buffer for a lane
//! and priority lives at `lane_id * num_prios + prio`. That index arithmetic,
//! the `num_lanes × num_prios` sizing, and the size-planning formula are the
//! whole of the C++ class — everything else it inherits from `ring_buffer`
//! and `vector`.
//!
//! Because `ring_buffer.rs` is still a stub while this module is written,
//! the lane type is a **generic parameter** bounded by [`RingLane`], and this
//! file carries a local stand-in lane ([`LaneRingBuffer`]) so the container
//! and the three C++ lane aliases are usable and testable today. See
//! divergence 1 for the hand-off to `ring_buffer.rs`.
//!
//! # C++ → Rust name mapping
//!
//! | C++ (`ctp::ipc`) | Rust (`ctp_ds::multi_ring_buffer`) | Notes |
//! |---|---|---|
//! | `multi_ring_buffer<T, AllocT, FLAGS>` | [`MultiRingBuffer<R>`] | lane type generic; divergences 1–2 |
//! | `multi_ring_buffer(alloc, num_lanes, num_prios, depth)` | [`MultiRingBuffer::new`] | + [`try_new`](MultiRingBuffer::try_new); divergence 4 |
//! | `multi_ring_buffer::CalculateSize(num_lanes, num_prios, depth)` | [`MultiRingBuffer::calculate_size`] | divergence 3 |
//! | `multi_ring_buffer(const&) = delete` | no `Clone` impl | |
//! | `multi_ring_buffer(&&) = delete` | — | divergence 6 |
//! | `~multi_ring_buffer()` (empty) | no `Drop` impl | |
//! | `GetLane(lane_id, prio)` | [`lane`](MultiRingBuffer::lane) / [`try_lane`](MultiRingBuffer::try_lane) | `assert` → panic / `Option`; divergence 5 |
//! | `GetLane(lane_id, prio) const` | [`lane_mut`](MultiRingBuffer::lane_mut) / [`try_lane_mut`](MultiRingBuffer::try_lane_mut) | const/non-const → `&`/`&mut`; divergence 5 |
//! | `GetNumLanes()` | [`num_lanes`](MultiRingBuffer::num_lanes) | |
//! | `GetNumPrios()` | [`num_prios`](MultiRingBuffer::num_prios) | |
//! | `GetTotalBuffers()` | [`total_buffers`](MultiRingBuffer::total_buffers) | |
//! | `lanes_` (private) | [`lanes`](MultiRingBuffer::lanes) / [`lanes_mut`](MultiRingBuffer::lanes_mut) | flat row-major view |
//! | `multi_ext_ring_buffer<T, AllocT>` | [`MultiExtRingBuffer<T>`] | |
//! | `multi_spsc_ring_buffer<T, AllocT>` | [`MultiSpscRingBuffer<T>`] | |
//! | `multi_mpsc_ring_buffer<T, AllocT>` | [`MultiMpscRingBuffer<T>`] | |
//! | `AllocT` / `ShmContainer<AllocT>` | — | divergence 2 |
//! | `ring_buffer_type` / `ring_buffer<T, AllocT, FLAGS>` | [`RingLane`] + [`LaneRingBuffer<T, FLAGS>`] | divergence 1 |
//! | `ring_buffer_type::CalculateSize(depth)` | [`RingLane::calculate_size`] | |
//! | `entry_type` / `RingBufferEntry<T>` | [`RingBufferEntry<T>`] | |
//! | `RingQueueFlag` enumerators | [`RING_BUFFER_SPSC_FLAGS`] … [`RING_BUFFER_LOCK_POP`] | `enum : uint32_t` → `pub const u32` |
//!
//! Lane surface carried by the local stand-in ([`LaneRingBuffer`], divergence 8):
//!
//! | C++ `ring_buffer` | Rust | Notes |
//! |---|---|---|
//! | `ring_buffer(alloc, depth = 1024)` | [`LaneRingBuffer::new`] | `depth+1` slots, one reserved |
//! | `Size()` | [`len`](LaneRingBuffer::len) | |
//! | `Capacity()` | [`capacity`](LaneRingBuffer::capacity) | |
//! | `GetDepth()` | [`depth`](LaneRingBuffer::depth) | slot count = capacity + 1 |
//! | `Empty()` | [`is_empty`](LaneRingBuffer::is_empty) | |
//! | `Full()` | [`is_full`](LaneRingBuffer::is_full) | |
//! | `Push(val)` / `TryPush(val)` / `Emplace(args…)` | [`push`](LaneRingBuffer::push) / [`try_push`](LaneRingBuffer::try_push) / [`emplace`](LaneRingBuffer::emplace) | variadic emplace → one value |
//! | `Pop(T&)` / `TryPop(T&)` | [`pop`](LaneRingBuffer::pop) / [`try_pop`](LaneRingBuffer::try_pop) | divergence 11 |
//! | `GetHead()` / `GetTail()` | [`head`](LaneRingBuffer::head) / [`tail`](LaneRingBuffer::tail) | |
//! | `Clear()` / `Reset()` | [`clear`](LaneRingBuffer::clear) / [`reset`](LaneRingBuffer::reset) | |
//! | `GetAssignedWorkerId()` / `SetAssignedWorkerId(id)` | [`assigned_worker_id`](LaneRingBuffer::assigned_worker_id) / [`set_assigned_worker_id`](LaneRingBuffer::set_assigned_worker_id) | divergence 15 |
//! | `GetSignalFd()` / `SetSignalFd(fd)` | [`signal_fd`](LaneRingBuffer::signal_fd) / [`set_signal_fd`](LaneRingBuffer::set_signal_fd) | divergence 15 |
//! | `GetTid()` / `SetTid(tid)` | [`tid`](LaneRingBuffer::tid) / [`set_tid`](LaneRingBuffer::set_tid) | divergence 14 |
//! | `IsActive()` / `SetActive(b)` | [`is_active`](LaneRingBuffer::is_active) / [`set_active`](LaneRingBuffer::set_active) | divergence 7 |
//! | `IsSPSC` … `IsAtomic` (`static constexpr`) | [`LaneRingBuffer::IS_SPSC`] … [`IS_ATOMIC`](LaneRingBuffer::IS_ATOMIC) | associated consts |
//!
//! # Semantic divergences from the C++
//!
//! 1. **The lane type is a generic parameter, not `ring_buffer<T, AllocT,
//!    FLAGS>`.** The C++ hard-wires its lane type through the `T`/`FLAGS`
//!    template parameters; here [`MultiRingBuffer<R>`] takes the lane itself,
//!    constrained by the two operations the C++ container actually uses on it
//!    (construct-with-depth, `CalculateSize`) — that is the [`RingLane`]
//!    trait. `ring_buffer.rs` was an empty stub when this module was written
//!    (parallel port), so a local stand-in lane, [`LaneRingBuffer<T, FLAGS>`],
//!    is provided here and the aliases point at it.
//!
//!    **Hand-off.** `ring_buffer.rs` has since landed, and its `RingBuffer<T,
//!    FLAGS>` is deliberately an *unsized* type (a trailing
//!    `[RingBufferEntry<T>]`) that is placed into caller-supplied memory by
//!    `unsafe RingBuffer::init_at(mem, depth) -> &Self`. It therefore cannot
//!    implement [`RingLane`], whose `with_depth(depth) -> Self` requires an
//!    owned, `Sized` lane — and that is the correct outcome, not an oversight:
//!    a lane that must be *placed* belongs to the in-segment container
//!    described in divergence 2. The finished port allocates one block of
//!    `RingBuffer::calculate_size(depth)` bytes per lane and `init_at`s it at
//!    stride `lane_id * num_prios + prio` — which is precisely the arithmetic
//!    [`MultiRingBuffer::calculate_size`] already computes. At that point this
//!    file keeps its index/sizing logic and drops [`LaneRingBuffer`],
//!    [`RingBufferEntry`] and the `RING_BUFFER_*` constants, all of which
//!    `ring_buffer.rs` also defines from the same C++ enum.
//!
//! 2. **No allocator, and not shared-memory resident.** The C++ is a
//!    `ShmContainer<AllocT>` holding `vector<ring_buffer, AllocT> lanes_`, so
//!    the whole structure lives inside a segment and is reachable from any
//!    process that maps it. This port stores lanes in a process-local
//!    `Vec<R>`, because the in-segment vector it would need
//!    (`shm_vector.rs`) is itself an unfinished stub. Concretely missing
//!    versus the C++: `#[repr(C)]`/`ShmSafe` placement of the container,
//!    offset-based (rather than pointer-based) lane storage, the `AllocT`
//!    constructor parameter, and cross-process addressing of lanes. The lane
//!    *logic* (index math, sizing, per-lane state) is complete and unaffected;
//!    what changes is only where the lane array lives. `MultiRingBuffer` is
//!    `Send`/`Sync` when `R` is, so lanes are shareable across threads within
//!    one process today.
//!
//! 3. **`calculate_size` is saturating and reports Rust sizes.** The formula
//!    is the C++ one verbatim — `sizeof(Self) + num_lanes * num_prios *
//!    lane::CalculateSize(depth)` — but `sizeof` is the Rust type's size, so
//!    the number is not byte-comparable with the C++ (see divergence 2). Both
//!    versions are *planning estimates*: they ignore allocator block headers
//!    and inter-object alignment padding. Every multiplication/addition
//!    saturates instead of wrapping (C++ `size_t` overflow would silently
//!    produce an absurdly small size); a saturated result is `usize::MAX`,
//!    which no allocation can satisfy — the honest answer.
//!
//! 4. **Construction validates the lane count.** `num_lanes * num_prios`
//!    overflowing `size_t` is silent wraparound in C++, and the vector ctor
//!    then allocates the wrong count. [`MultiRingBuffer::try_new`] returns
//!    `None` on overflow or allocation failure; [`MultiRingBuffer::new`]
//!    panics with a message. `try_new` has no C++ counterpart.
//!
//! 5. **`GetLane`'s `assert`s become unconditional panics.** The C++ asserts
//!    are compiled out in release builds, where an out-of-range `lane_id`/
//!    `prio` reads out of bounds (UB). [`lane`](MultiRingBuffer::lane) and
//!    [`lane_mut`](MultiRingBuffer::lane_mut) panic in *all* builds, per
//!    MEMORY_DESIGN.md's "never UB" rule; [`try_lane`](MultiRingBuffer::try_lane)
//!    / [`try_lane_mut`](MultiRingBuffer::try_lane_mut) are non-C++ additions
//!    returning `Option`. Note the C++ checks each coordinate separately
//!    rather than the product, so `(0, num_prios)` and `(1, 0)` are distinct
//!    outcomes even though both resolve to index `num_prios`; that
//!    per-coordinate check is reproduced exactly.
//!
//! 6. **Moves are permitted.** C++ deletes both the copy and the move ctor so
//!    an IPC structure can only be built in place through the allocator. Copy
//!    is likewise absent here (no `Clone`), but a Rust move is a bitwise
//!    relocation of a `Vec` handle that no lane observes — nothing inside a
//!    lane stores its own address (MEMORY_DESIGN.md pillar 3), so it is safe.
//!    A shared-memory-resident version (divergence 2) would be pinned in place
//!    by construction and the point becomes moot.
//!
//! 7. **`head_`/`tail_`/`active_` are always real atomics.** The C++ selects
//!    `opt_atomic<T, IsAtomic>` — plain storage for SPSC, `std::atomic` for
//!    MPSC. Choosing a type from a *computed* `const` generic
//!    (`FLAGS & MPSC != 0`) needs `generic_const_exprs`, which is unstable, so
//!    this port always uses `ctp_types::atomic::Atomic`. This costs SPSC lanes
//!    some performance, changes no observable single-producer behavior, and
//!    strictly removes UB when an "SPSC" lane is misused from several threads.
//!    `active_` is additionally stored as `Atomic<u32>` (`0`/`1`) rather than
//!    `opt_atomic<bool>`, because `ctp-types` deliberately restricts its
//!    atomics to integer primitives (its own divergence 3); the accessors
//!    still take and return `bool`.
//!
//! 8. **The local lane carries the subset `multi_ring_buffer` needs.** Present:
//!    construction/sizing, `Size`/`Capacity`/`GetDepth`/`Empty`/`Full`,
//!    `Push`/`TryPush`/`Emplace`, `Pop`/`TryPop`, `GetHead`/`GetTail`,
//!    `Clear`/`Reset`, and the worker metadata accessors. Deliberately absent
//!    (they belong to `ring_buffer.rs`, and no `multi_*` alias uses them):
//!    `PushSystem`, `PopDevice`, `GetHeadDevice`/`GetTailDevice`, `Peek`,
//!    `Resize`, the copy constructor, and the GPU device-scope entry helpers
//!    (`IsReadyDevice`, `SetReadySystem`) — per MIGRATION.md, kernels stay
//!    CUDA C++ and Rust only ever compiles the host branch, so the
//!    device-scope variants would be exact duplicates of their host twins.
//!
//! 9. **A space policy is mandatory; `RING_BUFFER_LOCK_POP` is ignored.**
//!    A `FLAGS` value with none of `WAIT_FOR_SPACE`/`ERROR_ON_NO_SPACE`/
//!    `DYNAMIC_SIZE` makes the C++ `Emplace` skip the space check entirely and
//!    overwrite live entries (the `circular_mpsc_ring_buffer` alias) — a
//!    genuine data race with a concurrent `Pop`, which Rust cannot sanction
//!    behind a safe API. [`LaneRingBuffer`] therefore rejects such `FLAGS` at
//!    compile time (a `const` assertion fired from
//!    [`new`](LaneRingBuffer::new)); no `multi_*` alias uses that combination,
//!    and modelling `circular_mpsc_ring_buffer` soundly is `ring_buffer.rs`'s
//!    problem. `RING_BUFFER_LOCK_POP` is accepted and ignored; no `multi_*`
//!    alias sets it, and divergence 16 explains why that is sound here — and
//!    why the C++ header's stated reason for the flag is wrong.
//!
//! 16. **Lanes are single-consumer, and the C++ header is wrong about why.**
//!     `ring_buffer::Pop`'s comment claims `RING_BUFFER_LOCK_POP` is a mere
//!     optimization — "the CAS in the lock-free path already prevents
//!     duplicate delivery", so the lock only removes "wasted cycles and unfair
//!     scheduling". The CAS does prevent duplicate *delivery*, but it does not
//!     make the lock-free path multi-consumer-safe, because `head_` is
//!     advanced by a **plain store of a locally-read value**
//!     (`head_.store_system(head + 1)`), not a CAS. Consumer B can read
//!     `head == h`, stall, and wake after consumer A has drained a full lap
//!     and the producer has refilled slot `h % slots`. B then finds that slot
//!     ready, wins its CAS, consumes the *new* element, and stores
//!     `head = h + 1` — regressing `head` by a whole lap. Every slot between
//!     the regressed `head` and the true one has `flags_ == 0`, so `Pop`
//!     returns `false` forever while `Empty()` reports non-empty, and (under
//!     `WAIT_FOR_SPACE`) producers then block permanently: the lane is wedged.
//!     This is reachable in the C++ exactly as written; it is a genuine bug,
//!     preserved here rather than papered over.
//!
//!     Consequences for this port: the `MPSC` in `multi_mpsc_ring_buffer`
//!     means *Multiple Producer, **Single** Consumer*, and that is the
//!     contract — one `pop()` caller per lane, any number of `push()` callers.
//!     Nothing is lost, since no `multi_*` alias sets `RING_BUFFER_LOCK_POP`
//!     (the C++'s own answer to multi-consumer, via `mpmc_ring_buffer`).
//!     **Memory safety is unaffected either way**, which is why `pop` can keep
//!     `&self`: the head regression corrupts the queue's logical state, never
//!     the slot ownership protocol. The ready-flag CAS still elects exactly one
//!     consumer per payload, and a regressed `head` only ever makes the
//!     producer's `size = tail - head + 1` check *larger*, i.e. strictly more
//!     conservative — so a producer can never be tricked into writing a slot a
//!     consumer is reading. The failure mode is a hang, not undefined
//!     behavior.
//!
//! 10. **The `DYNAMIC_SIZE` tail leak is preserved, not fixed.** On a full
//!     buffer, the C++ `Emplace` returns `false` *after* having incremented
//!     `tail_`, and — unlike the `ErrorOnNoSpace` branch — never restores it.
//!     The lane is then wedged forever: `head_` can never catch up to the
//!     phantom slot, so every subsequent `Pop` returns `false` while `Empty()`
//!     reports non-empty. [`MultiExtRingBuffer`] inherits this. The behavior is
//!     reproduced exactly (and pinned by
//!     `dynamic_size_full_push_wedges_the_lane_like_cpp`) rather than silently
//!     corrected, so that a fix lands as a deliberate change on both sides.
//!
//! 11. **`Pop` returns `Option<T>` instead of `bool` + out-parameter.** The
//!     C++ leaves `val` untouched on failure; `None` says the same thing
//!     without requiring a default-constructed temporary at the call site.
//!
//! 12. **All atomic orderings are `SeqCst`.** The C++ scope suffixes
//!     (`load_system`, `fetch_add_system`, `store_system`,
//!     `compare_exchange_strong_system`) select CPU↔GPU-coherent instructions
//!     on the device pass and plain `seq_cst` operations on the host; Rust
//!     only ever compiles the host branch, so this port calls
//!     `ctp-types`' identically-named host helpers. Never weaker than the C++.
//!
//! 13. **Elements are `Copy + Default`.** The C++ requires `T` to be
//!     copy-assignable and default-constructible (`entry.data_ = T()` in
//!     `Pop`); `Copy + Default` is the Rust spelling of that, and it matches
//!     the POD discipline `ShmSafe` demands of anything segment-resident.
//!
//! 14. **`pid_t tid_` → `i32`.** The C++ header itself types `pid_t` as `int`
//!     on Windows; `i32` is that type on every platform CTP supports, and it
//!     keeps the field a fixed 4 bytes (per the project rule that OS headers
//!     stay out of portable code).
//!
//! 15. **Plain-metadata setters take `&mut self`.** `SetAssignedWorkerId`,
//!     `SetSignalFd` and `SetTid` write non-atomic fields through a non-const
//!     `this` in C++; doing that through a shared reference is a data race in
//!     either language. `SetActive` keeps `&self` because `active_` is atomic
//!     in the C++ too. Reach the `&mut` setters through
//!     [`MultiRingBuffer::lane_mut`].

use ctp_types::atomic::{threadfence, threadfence_system, Atomic, AtomicApi};
use std::cell::UnsafeCell;
use std::sync::atomic::Ordering;

// ---------------------------------------------------------------------------
// RingQueueFlag — ctp::ipc::RingQueueFlag
// ---------------------------------------------------------------------------

/// Single producer single consumer mode (no atomics needed).
pub const RING_BUFFER_SPSC_FLAGS: u32 = 0x01;
/// Multiple producers single consumer mode (atomic operations required).
pub const RING_BUFFER_MPSC_FLAGS: u32 = 0x02;
/// Wait for space (block until space is available).
pub const RING_BUFFER_WAIT_FOR_SPACE: u32 = 0x04;
/// Error on no space (return error if no space).
pub const RING_BUFFER_ERROR_ON_NO_SPACE: u32 = 0x08;
/// Dynamic size (resize buffer when full).
pub const RING_BUFFER_DYNAMIC_SIZE: u32 = 0x10;
/// Fixed-size buffer (no dynamic resizing).
pub const RING_BUFFER_FIXED_SIZE: u32 = 0x20;
/// Serialize `Pop()` with a mutex (enables multi-consumer / MPMC). Accepted
/// and ignored here — see divergence 9.
pub const RING_BUFFER_LOCK_POP: u32 = 0x40;

/// The C++ default `FLAGS` template argument of both `ring_buffer` and
/// `multi_ring_buffer`.
pub const DEFAULT_RING_BUFFER_FLAGS: u32 =
    RING_BUFFER_SPSC_FLAGS | RING_BUFFER_FIXED_SIZE | RING_BUFFER_ERROR_ON_NO_SPACE;

/// `FLAGS` of `multi_ext_ring_buffer`.
pub const MULTI_EXT_RING_BUFFER_FLAGS: u32 = RING_BUFFER_SPSC_FLAGS | RING_BUFFER_DYNAMIC_SIZE;

/// `FLAGS` of `multi_spsc_ring_buffer` (identical to the default).
pub const MULTI_SPSC_RING_BUFFER_FLAGS: u32 =
    RING_BUFFER_SPSC_FLAGS | RING_BUFFER_FIXED_SIZE | RING_BUFFER_ERROR_ON_NO_SPACE;

/// `FLAGS` of `multi_mpsc_ring_buffer`.
pub const MULTI_MPSC_RING_BUFFER_FLAGS: u32 =
    RING_BUFFER_MPSC_FLAGS | RING_BUFFER_FIXED_SIZE | RING_BUFFER_WAIT_FOR_SPACE;

/// The flag sets that give `Emplace` a space policy. A `FLAGS` value carrying
/// none of these makes the C++ overwrite live entries; see divergence 9.
const SPACE_POLICY_FLAGS: u32 =
    RING_BUFFER_WAIT_FOR_SPACE | RING_BUFFER_ERROR_ON_NO_SPACE | RING_BUFFER_DYNAMIC_SIZE;

/// Bit 0 of `RingBufferEntry::flags_` — "data ready for consumption".
const READY_BIT: u32 = 1;

// ---------------------------------------------------------------------------
// RingBufferEntry<T> — ctp::ipc::RingBufferEntry
// ---------------------------------------------------------------------------

/// A slot: an atomic ready flag plus the payload — port of
/// `RingBufferEntry<T>`.
///
/// Field order and `#[repr(C)]` mirror the C++ (`flags_` then `data_`). The
/// payload sits in an `UnsafeCell` because the producer writes it through a
/// shared reference, exactly as the C++ writes it through a shared object;
/// the ready-flag protocol — described on
/// [`SAFETY`](RingBufferEntry#slot-ownership-protocol) below — is what makes
/// that sound.
///
/// # Slot ownership protocol
///
/// * A producer acquires exclusive ownership of slot `tail % depth` by
///   `fetch_add`-ing `tail_`: each `tail` value is handed to exactly one
///   producer, and the space check (mandatory — divergence 9) guarantees the
///   previous lap's consumer is finished with that slot before it is reused.
///   The producer writes `data`, then publishes with a release fence followed
///   by an atomic OR of [`READY_BIT`].
/// * A consumer acquires exclusive ownership by winning the `1 → 0`
///   compare-exchange on `flags`: exactly one consumer can observe the
///   transition. It then reads `data`, resets it, and advances `head_`.
///
/// So the payload is never read and written concurrently, and never written by
/// two threads at once.
///
/// Note this protocol is what makes the `unsafe` here sound, and it holds even
/// under the lock-free multi-consumer `head_` regression of divergence 16: a
/// regressed `head_` only inflates the producer's occupancy estimate, so it
/// makes producers wait longer, never lets one re-enter a slot early. The
/// consequence of that bug is a wedged lane, never a data race.
#[repr(C)]
pub struct RingBufferEntry<T> {
    /// C++ `abitfield32_t flags_` — bit 0 = data ready.
    flags: Atomic<u32>,
    /// C++ `T data_`.
    data: UnsafeCell<T>,
}

// SAFETY: every access to `data` is serialized by the slot ownership protocol
// documented on the type: producers are elected by the `tail_` fetch_add,
// consumers by the ready-flag compare-exchange, and the two are ordered by the
// release/acquire pair on `flags`. Sharing the entry across threads therefore
// creates no data race, provided `T` may itself move between threads.
unsafe impl<T: Send> Sync for RingBufferEntry<T> {}

impl<T: Copy + Default> RingBufferEntry<T> {
    /// C++ `RingBufferEntry()` — `flags_(0)`, `data_` default-constructed.
    fn new() -> Self {
        Self {
            flags: Atomic::new(0),
            data: UnsafeCell::new(T::default()),
        }
    }

    /// C++ `IsReady()` / `IsReadySystem()` (host branch — divergence 12).
    #[inline]
    fn is_ready(&self) -> bool {
        self.flags.load_system() & READY_BIT != 0
    }

    /// C++ `SetReady()`: `threadfence_system()` then a system-scope bit set,
    /// so the `data` write above cannot be reordered past the flag.
    #[inline]
    fn set_ready(&self) {
        threadfence_system();
        self.flags.or_system(READY_BIT);
    }

    /// The `1 → 0` claim from `PopUnlocked`. `true` means this caller now owns
    /// the slot; `false` means another consumer got there first.
    #[inline]
    fn claim(&self) -> bool {
        let mut expected = READY_BIT;
        self.flags
            .compare_exchange_strong_system(&mut expected, 0, Ordering::SeqCst)
    }

    /// C++ `ClearReady()` / `flags_.Clear()`.
    #[inline]
    fn clear_flags(&self) {
        self.flags.store_system(0);
    }

    /// # Safety
    /// The caller must own this slot as a producer (it `fetch_add`-ed the
    /// `tail_` value mapping here and the space policy proved the slot free)
    /// or as a consumer that won [`claim`](Self::claim).
    #[inline]
    unsafe fn write_data(&self, val: T) {
        // SAFETY: forwarded to this function's contract — the caller is the
        // slot's exclusive owner, so no other thread reads or writes `data`
        // for the duration of this store. `T: Copy` has no destructor, so
        // overwriting the previous value cannot run user code.
        unsafe { *self.data.get() = val };
    }

    /// # Safety
    /// The caller must own this slot as a consumer, i.e. it won
    /// [`claim`](Self::claim), whose compare-exchange also acquires the
    /// producer's release on `flags` and thus its `data` write.
    #[inline]
    unsafe fn read_data(&self) -> T {
        // SAFETY: forwarded to this function's contract — the caller is the
        // slot's exclusive owner, so `data` is fully written and quiescent.
        unsafe { *self.data.get() }
    }
}

// ---------------------------------------------------------------------------
// RingLane — what multi_ring_buffer requires of a lane
// ---------------------------------------------------------------------------

/// The lane interface [`MultiRingBuffer`] depends on.
///
/// The C++ container only ever does two things with its lane type: construct
/// `num_lanes * num_prios` of them at a given depth, and ask
/// `ring_buffer_type::CalculateSize(depth)` for the sizing formula. This trait
/// is exactly that surface, which is what lets the container be ported now and
/// re-pointed at `ring_buffer.rs`'s type later (divergence 1).
pub trait RingLane: Sized {
    /// C++ `ring_buffer(alloc, depth)` — one lane holding `depth` elements.
    fn with_depth(depth: usize) -> Self;

    /// C++ `ring_buffer_type::CalculateSize(depth)` — bytes for one lane.
    fn calculate_size(depth: usize) -> usize;
}

// ---------------------------------------------------------------------------
// LaneRingBuffer<T, FLAGS> — local stand-in for ring_buffer<T, AllocT, FLAGS>
// ---------------------------------------------------------------------------

/// Local stand-in for `ctp::ipc::ring_buffer<T, AllocT, FLAGS>` (divergence 1).
///
/// A lock-free circular queue of `depth + 1` slots — one is always reserved as
/// the full/empty sentinel, exactly as in C++, so `capacity() == depth`.
/// `head_`/`tail_` are free-running `u64` counters; a slot is `counter % depth`.
///
/// `FLAGS` must carry a space policy (divergence 9): one of
/// [`RING_BUFFER_WAIT_FOR_SPACE`], [`RING_BUFFER_ERROR_ON_NO_SPACE`] or
/// [`RING_BUFFER_DYNAMIC_SIZE`]. Violating that is a compile-time error.
pub struct LaneRingBuffer<T, const FLAGS: u32 = DEFAULT_RING_BUFFER_FLAGS> {
    /// C++ `entry_vector queue_` — `depth + 1` slots.
    queue: Vec<RingBufferEntry<T>>,
    /// C++ `head_type head_` — consumer cursor. Divergence 7.
    head: Atomic<u64>,
    /// C++ `tail_type tail_` — producer cursor. Divergence 7.
    tail: Atomic<u64>,
    /// C++ `u32 assigned_worker_id_`.
    assigned_worker_id: u32,
    /// C++ `int signal_fd_`.
    signal_fd: i32,
    /// C++ `pid_t tid_`. Divergence 14.
    tid: i32,
    /// C++ `opt_atomic<bool, IsAtomic> active_`, as `0`/`1`. Divergence 7.
    active: Atomic<u32>,
}

impl<T: Copy + Default, const FLAGS: u32> LaneRingBuffer<T, FLAGS> {
    /// C++ `static constexpr bool IsSPSC`.
    pub const IS_SPSC: bool = FLAGS & RING_BUFFER_SPSC_FLAGS != 0;
    /// C++ `static constexpr bool IsMPSC`.
    pub const IS_MPSC: bool = FLAGS & RING_BUFFER_MPSC_FLAGS != 0;
    /// C++ `static constexpr bool WaitForSpace`.
    pub const WAIT_FOR_SPACE: bool = FLAGS & RING_BUFFER_WAIT_FOR_SPACE != 0;
    /// C++ `static constexpr bool ErrorOnNoSpace`.
    pub const ERROR_ON_NO_SPACE: bool = FLAGS & RING_BUFFER_ERROR_ON_NO_SPACE != 0;
    /// C++ `static constexpr bool DynamicSize`.
    pub const DYNAMIC_SIZE: bool = FLAGS & RING_BUFFER_DYNAMIC_SIZE != 0;
    /// C++ `static constexpr bool LockPop`. Ignored — divergence 9.
    pub const LOCK_POP: bool = FLAGS & RING_BUFFER_LOCK_POP != 0;
    /// C++ `static constexpr bool IsAtomic = IsMPSC`. Reported for parity;
    /// the storage is atomic either way (divergence 7).
    pub const IS_ATOMIC: bool = Self::IS_MPSC;

    /// Rejects the C++ "circular" flag sets that skip the space check
    /// (divergence 9). Forced by [`new`](Self::new).
    const SPACE_POLICY_CHECK: () = assert!(
        FLAGS & SPACE_POLICY_FLAGS != 0,
        "LaneRingBuffer: FLAGS must set one of RING_BUFFER_WAIT_FOR_SPACE, \
         RING_BUFFER_ERROR_ON_NO_SPACE or RING_BUFFER_DYNAMIC_SIZE; without a \
         space policy a full buffer silently overwrites live entries"
    );

    /// C++ `ring_buffer(alloc, depth)`. Allocates `depth + 1` slots.
    pub fn new(depth: usize) -> Self {
        let () = Self::SPACE_POLICY_CHECK;
        let slots = depth.saturating_add(1);
        Self {
            queue: (0..slots).map(|_| RingBufferEntry::new()).collect(),
            head: Atomic::new(0),
            tail: Atomic::new(0),
            assigned_worker_id: 0,
            signal_fd: -1,
            tid: 0,
            active: Atomic::new(1),
        }
    }

    #[inline]
    fn slots(&self) -> u64 {
        self.queue.len() as u64
    }

    /// C++ `Size()` — items currently in the buffer, `0` if `tail < head`.
    #[inline]
    pub fn len(&self) -> usize {
        let head = self.head.load_system();
        let tail = self.tail.load_system();
        if tail >= head {
            usize::try_from(tail - head).unwrap_or(usize::MAX)
        } else {
            0
        }
    }

    /// C++ `Empty()` — `head_ == tail_`.
    ///
    /// Not the negation of `len() == 0`: `Size()` clamps a `tail < head`
    /// reading to zero while `Empty()` reports `false` for it. Both are the
    /// C++ definitions verbatim.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.head.load_system() == self.tail.load_system()
    }

    /// C++ `Capacity()` — usable entries, i.e. slots minus the reserved one.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.queue.len().saturating_sub(1)
    }

    /// C++ `GetDepth()` — allocated slots (`capacity() + 1`).
    #[inline]
    pub fn depth(&self) -> usize {
        self.queue.len()
    }

    /// C++ `Full()`.
    #[inline]
    pub fn is_full(&self) -> bool {
        let head = self.head.load_system();
        let tail = self.tail.load_system();
        tail.wrapping_sub(head) == self.slots() - 1
    }

    /// C++ `Push(val)`.
    #[inline]
    pub fn push(&self, val: T) -> bool {
        self.emplace(val)
    }

    /// C++ `TryPush(val)` — an alias for `Push`, as in the header.
    #[inline]
    pub fn try_push(&self, val: T) -> bool {
        self.push(val)
    }

    /// C++ `Emplace(args...)`. The variadic construct-in-place has no Rust
    /// analogue for a `Copy + Default` payload, so it takes the value.
    pub fn emplace(&self, val: T) -> bool {
        let slots = self.slots();
        // C++ loads head BEFORE claiming a slot, so the space check is
        // conservative under concurrency: it may see a stale (small) head and
        // report full early. Reproduced exactly.
        let mut head = self.head.load(Ordering::SeqCst);
        let tail = self.tail.fetch_add_system(1);

        if Self::WAIT_FOR_SPACE {
            let mut size = tail.wrapping_sub(head).wrapping_add(1);
            while size >= slots {
                std::hint::spin_loop();
                head = self.head.load_device();
                size = tail.wrapping_sub(head).wrapping_add(1);
            }
        } else if Self::ERROR_ON_NO_SPACE {
            let size = tail.wrapping_sub(head).wrapping_add(1);
            if size >= slots {
                self.tail.fetch_sub(1, Ordering::SeqCst);
                return false;
            }
        } else if Self::DYNAMIC_SIZE {
            let size = tail.wrapping_sub(head).wrapping_add(1);
            if size >= slots {
                // C++: "Would need to resize vector - not implemented for now".
                // `tail_` is deliberately NOT restored here — divergence 10.
                return false;
            }
        }

        let idx = (tail % slots) as usize;
        let entry = &self.queue[idx];
        // SAFETY: this thread `fetch_add`-ed the `tail` value that maps to
        // `idx`, so it is the only producer for this slot; the space policy
        // above established that the consumer of the previous lap has released
        // it (`head` has passed `tail - capacity`). Per the slot ownership
        // protocol on `RingBufferEntry`, no consumer touches `data` until
        // `set_ready` publishes it below.
        unsafe { entry.write_data(val) };
        // C++ threadfence() before SetReady(), which fences again at system
        // scope: the data write must not sink past the ready flag.
        threadfence();
        entry.set_ready();
        true
    }

    /// C++ `Pop(T&)` — the lock-free `PopUnlocked` body; `RING_BUFFER_LOCK_POP`
    /// is ignored (divergence 9). Returns `None` where the C++ returns `false`
    /// (divergence 11).
    ///
    /// A `None` does not prove the lane is empty: a producer may have claimed
    /// the slot at `head` without having published it yet, exactly as in C++.
    pub fn pop(&self) -> Option<T> {
        let head = self.head.load_system();
        let tail = self.tail.load_system();
        if head >= tail {
            return None;
        }
        let idx = (head % self.slots()) as usize;
        let entry = &self.queue[idx];
        if !entry.is_ready() {
            return None;
        }
        if !entry.claim() {
            // Another consumer atomically claimed this entry first.
            return None;
        }
        // SAFETY: `claim()` won the 1 -> 0 compare-exchange on the ready flag,
        // which (a) elects this thread the slot's exclusive consumer — no other
        // consumer can observe that transition — and (b) synchronizes with the
        // producer's `set_ready` release, so the payload write happens-before
        // this read. The producer cannot re-enter the slot until `head`
        // advances a full lap, which only the store below permits.
        let val = unsafe { entry.read_data() };
        // C++ releases the slot's copy so it cannot pin owning resources (e.g.
        // a Future's shared_ptr) until the slot is overwritten — see the #620
        // leak note in the header. Cheap for a Copy payload.
        // SAFETY: as above — this thread still exclusively owns the slot.
        unsafe { entry.write_data(T::default()) };
        self.head.store_system(head.wrapping_add(1));
        Some(val)
    }

    /// C++ `TryPop(T&)` — an alias for `Pop`, as in the header.
    #[inline]
    pub fn try_pop(&self) -> Option<T> {
        self.pop()
    }

    /// C++ `GetHead()` — oldest valid event ID.
    #[inline]
    pub fn head(&self) -> u64 {
        self.head.load_system()
    }

    /// C++ `GetTail()` — next event ID to be written.
    #[inline]
    pub fn tail(&self) -> u64 {
        self.tail.load_system()
    }

    /// C++ `Clear()` — zero the cursors and every entry's flags. As in C++ the
    /// payloads are left as they are; the flags are what gate reads.
    pub fn clear(&mut self) {
        self.head.store_system(0);
        self.tail.store_system(0);
        for entry in &self.queue {
            entry.clear_flags();
        }
    }

    /// C++ `Reset()` — alias for `Clear()`.
    #[inline]
    pub fn reset(&mut self) {
        self.clear();
    }

    /// C++ `GetAssignedWorkerId()`.
    #[inline]
    pub fn assigned_worker_id(&self) -> u32 {
        self.assigned_worker_id
    }

    /// C++ `SetAssignedWorkerId(worker_id)`. Divergence 15.
    #[inline]
    pub fn set_assigned_worker_id(&mut self, worker_id: u32) {
        self.assigned_worker_id = worker_id;
    }

    /// C++ `GetSignalFd()`.
    #[inline]
    pub fn signal_fd(&self) -> i32 {
        self.signal_fd
    }

    /// C++ `SetSignalFd(signal_fd)`. Divergence 15.
    #[inline]
    pub fn set_signal_fd(&mut self, signal_fd: i32) {
        self.signal_fd = signal_fd;
    }

    /// C++ `GetTid()`. Divergence 14.
    #[inline]
    pub fn tid(&self) -> i32 {
        self.tid
    }

    /// C++ `SetTid(tid)`. Divergences 14 and 15.
    #[inline]
    pub fn set_tid(&mut self, tid: i32) {
        self.tid = tid;
    }

    /// C++ `IsActive()` — worker accepting tasks, vs blocked in `epoll_wait`.
    #[inline]
    pub fn is_active(&self) -> bool {
        self.active.load_system() != 0
    }

    /// C++ `SetActive(active)`. Keeps `&self`: `active_` is atomic in C++ too.
    #[inline]
    pub fn set_active(&self, active: bool) {
        self.active.store_system(u32::from(active));
    }
}

impl<T: Copy + Default, const FLAGS: u32> RingLane for LaneRingBuffer<T, FLAGS> {
    #[inline]
    fn with_depth(depth: usize) -> Self {
        Self::new(depth)
    }

    /// C++ `ring_buffer::CalculateSize(depth)` —
    /// `sizeof(ring_buffer) + (depth + 1) * sizeof(entry_type)`. Saturating
    /// (divergence 3).
    fn calculate_size(depth: usize) -> usize {
        let base_size = std::mem::size_of::<Self>();
        let entry_size = std::mem::size_of::<RingBufferEntry<T>>();
        let entries_size = depth.saturating_add(1).saturating_mul(entry_size);
        base_size.saturating_add(entries_size)
    }
}

impl<T: Copy + Default, const FLAGS: u32> std::fmt::Debug for LaneRingBuffer<T, FLAGS> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LaneRingBuffer")
            .field("flags", &FLAGS)
            .field("head", &self.head())
            .field("tail", &self.tail())
            .field("capacity", &self.capacity())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// multi_ring_buffer<T, AllocT, FLAGS>
// ---------------------------------------------------------------------------

/// Multi-lane ring buffer — port of `ctp::ipc::multi_ring_buffer`.
///
/// `num_lanes × num_prios` independent ring buffers in one flat, row-major
/// array: the lane for `(lane_id, prio)` is at `lane_id * num_prios + prio`.
///
/// ```
/// use ctp_ds::multi_ring_buffer::MultiSpscRingBuffer;
///
/// // 4 lanes × 2 priorities, 16 elements per ring buffer.
/// let q = MultiSpscRingBuffer::<u64>::new(4, 2, 16);
/// assert_eq!(q.total_buffers(), 8);
///
/// q.lane(3, 1).push(42);
/// assert!(q.lane(3, 0).pop().is_none()); // a different lane entirely
/// assert_eq!(q.lane(3, 1).pop(), Some(42));
/// ```
pub struct MultiRingBuffer<R> {
    /// C++ `lanes_vector lanes_`. Process-local storage — divergence 2.
    lanes: Vec<R>,
    /// C++ `size_t num_lanes_`.
    num_lanes: usize,
    /// C++ `size_t num_prios_`.
    num_prios: usize,
}

impl<R: RingLane> MultiRingBuffer<R> {
    /// C++ `static size_t CalculateSize(num_lanes, num_prios, depth)` — bytes
    /// needed for a `multi_ring_buffer` with these parameters. Saturating, and
    /// an estimate; see divergence 3.
    pub fn calculate_size(num_lanes: usize, num_prios: usize, depth: usize) -> usize {
        // Base size includes all member variables.
        let base_size = std::mem::size_of::<Self>();
        // Each ring buffer's size.
        let per_ring_buffer_size = R::calculate_size(depth);
        // Total ring buffers = num_lanes * num_prios.
        let total_ring_buffers = num_lanes.saturating_mul(num_prios);
        base_size.saturating_add(total_ring_buffers.saturating_mul(per_ring_buffer_size))
    }

    /// C++ `multi_ring_buffer(alloc, num_lanes, num_prios, depth)`.
    ///
    /// Builds `num_lanes * num_prios` lanes, each holding `depth` elements.
    /// Either count may be zero, yielding a container with no lanes.
    ///
    /// # Panics
    /// If `num_lanes * num_prios` overflows `usize`, or the lane array cannot
    /// be allocated. See divergence 4 and [`try_new`](Self::try_new).
    pub fn new(num_lanes: usize, num_prios: usize, depth: usize) -> Self {
        Self::try_new(num_lanes, num_prios, depth).unwrap_or_else(|| {
            panic!(
                "multi_ring_buffer: cannot allocate {num_lanes} lanes × {num_prios} priorities \
                 (depth {depth}): the lane count overflows usize or memory is exhausted"
            )
        })
    }

    /// Fallible [`new`](Self::new) — `None` on lane-count overflow or
    /// allocation failure. No C++ counterpart (divergence 4).
    pub fn try_new(num_lanes: usize, num_prios: usize, depth: usize) -> Option<Self> {
        let total = num_lanes.checked_mul(num_prios)?;
        let mut lanes = Vec::new();
        lanes.try_reserve_exact(total).ok()?;
        lanes.extend((0..total).map(|_| R::with_depth(depth)));
        Some(Self {
            lanes,
            num_lanes,
            num_prios,
        })
    }
}

impl<R> MultiRingBuffer<R> {
    /// C++ `GetNumLanes()`.
    #[inline]
    pub fn num_lanes(&self) -> usize {
        self.num_lanes
    }

    /// C++ `GetNumPrios()`.
    #[inline]
    pub fn num_prios(&self) -> usize {
        self.num_prios
    }

    /// C++ `GetTotalBuffers()` — `num_lanes * num_prios`, which is always
    /// `lanes().len()` (the constructor rejects a product that would overflow,
    /// so the saturation here can never trigger).
    #[inline]
    pub fn total_buffers(&self) -> usize {
        self.num_lanes.saturating_mul(self.num_prios)
    }

    /// The flat, row-major lane array — a safe reading of the private C++
    /// `lanes_` member.
    #[inline]
    pub fn lanes(&self) -> &[R] {
        &self.lanes
    }

    /// [`lanes`](Self::lanes), mutably.
    #[inline]
    pub fn lanes_mut(&mut self) -> &mut [R] {
        &mut self.lanes
    }

    /// C++ `GetLane(lane_id, prio) const` → `idx = lane_id * num_prios_ + prio`.
    ///
    /// # Panics
    /// If `lane_id >= num_lanes()` or `prio >= num_prios()`. The C++ `assert`s
    /// vanish in release builds and the access becomes UB; see divergence 5 and
    /// [`try_lane`](Self::try_lane).
    #[inline]
    pub fn lane(&self, lane_id: usize, prio: usize) -> &R {
        self.try_lane(lane_id, prio)
            .unwrap_or_else(|| self.out_of_range(lane_id, prio))
    }

    /// C++ `GetLane(lane_id, prio)` (non-const overload).
    ///
    /// # Panics
    /// As [`lane`](Self::lane).
    #[inline]
    pub fn lane_mut(&mut self, lane_id: usize, prio: usize) -> &mut R {
        if self.lane_index(lane_id, prio).is_none() {
            self.out_of_range(lane_id, prio);
        }
        // Recomputed rather than reusing the index above so the borrow of
        // `self` from `lane_index` ends before the mutable one begins.
        let idx = lane_id * self.num_prios + prio;
        &mut self.lanes[idx]
    }

    /// Checked [`lane`](Self::lane) — `None` when out of range. No C++
    /// counterpart (divergence 5).
    #[inline]
    pub fn try_lane(&self, lane_id: usize, prio: usize) -> Option<&R> {
        self.lanes.get(self.lane_index(lane_id, prio)?)
    }

    /// Checked [`lane_mut`](Self::lane_mut) — `None` when out of range.
    #[inline]
    pub fn try_lane_mut(&mut self, lane_id: usize, prio: usize) -> Option<&mut R> {
        let idx = self.lane_index(lane_id, prio)?;
        self.lanes.get_mut(idx)
    }

    /// `lane_id * num_prios_ + prio`, guarded by the C++ `assert`s. The two
    /// bounds are checked separately, as in C++ (divergence 5).
    #[inline]
    fn lane_index(&self, lane_id: usize, prio: usize) -> Option<usize> {
        if lane_id >= self.num_lanes || prio >= self.num_prios {
            return None;
        }
        // Cannot overflow: `lane_id <= num_lanes - 1` and `prio < num_prios`,
        // so the result is `< num_lanes * num_prios`, which the constructor
        // already proved fits in usize.
        Some(lane_id * self.num_prios + prio)
    }

    #[cold]
    #[inline(never)]
    fn out_of_range(&self, lane_id: usize, prio: usize) -> ! {
        panic!(
            "multi_ring_buffer: lane ({lane_id}, {prio}) is out of range for {} lanes × {} \
             priorities",
            self.num_lanes, self.num_prios
        );
    }
}

impl<R> std::fmt::Debug for MultiRingBuffer<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultiRingBuffer")
            .field("num_lanes", &self.num_lanes)
            .field("num_prios", &self.num_prios)
            .field("total_buffers", &self.total_buffers())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Aliases — the three C++ `multi_*_ring_buffer` typedefs
// ---------------------------------------------------------------------------

/// C++ `multi_ext_ring_buffer<T, AllocT>` — extensible, single-thread only.
///
/// The C++ never actually implements the promised resize, so a push to a full
/// lane fails *and wedges it permanently*; that quirk is preserved here — see
/// divergence 10 before reaching for this alias.
pub type MultiExtRingBuffer<T> = MultiRingBuffer<LaneRingBuffer<T, MULTI_EXT_RING_BUFFER_FLAGS>>;

/// C++ `multi_spsc_ring_buffer<T, AllocT>` — fixed size, error on no space.
pub type MultiSpscRingBuffer<T> = MultiRingBuffer<LaneRingBuffer<T, MULTI_SPSC_RING_BUFFER_FLAGS>>;

/// C++ `multi_mpsc_ring_buffer<T, AllocT>` — fixed size, multiple producers,
/// **spins** until a consumer frees space.
pub type MultiMpscRingBuffer<T> = MultiRingBuffer<LaneRingBuffer<T, MULTI_MPSC_RING_BUFFER_FLAGS>>;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering as StdOrdering};
    use std::thread;

    type Spsc = LaneRingBuffer<u64, MULTI_SPSC_RING_BUFFER_FLAGS>;

    // -- flags ---------------------------------------------------------------

    #[test]
    fn flag_values_match_the_cpp_enum() {
        assert_eq!(RING_BUFFER_SPSC_FLAGS, 0x01);
        assert_eq!(RING_BUFFER_MPSC_FLAGS, 0x02);
        assert_eq!(RING_BUFFER_WAIT_FOR_SPACE, 0x04);
        assert_eq!(RING_BUFFER_ERROR_ON_NO_SPACE, 0x08);
        assert_eq!(RING_BUFFER_DYNAMIC_SIZE, 0x10);
        assert_eq!(RING_BUFFER_FIXED_SIZE, 0x20);
        assert_eq!(RING_BUFFER_LOCK_POP, 0x40);
        // The default template argument of multi_ring_buffer.
        assert_eq!(DEFAULT_RING_BUFFER_FLAGS, 0x01 | 0x20 | 0x08);
        assert_eq!(MULTI_SPSC_RING_BUFFER_FLAGS, DEFAULT_RING_BUFFER_FLAGS);
        assert_eq!(MULTI_EXT_RING_BUFFER_FLAGS, 0x01 | 0x10);
        assert_eq!(MULTI_MPSC_RING_BUFFER_FLAGS, 0x02 | 0x20 | 0x04);
    }

    /// The C++ `static constexpr bool` members, decoded from each alias' FLAGS.
    /// Tuple order: (IsSPSC, IsMPSC, WaitForSpace, ErrorOnNoSpace, DynamicSize,
    /// LockPop, IsAtomic).
    #[test]
    fn flag_derived_constants_decode_like_if_constexpr() {
        fn decode<T: Copy + Default, const F: u32>() -> (bool, bool, bool, bool, bool, bool, bool) {
            (
                LaneRingBuffer::<T, F>::IS_SPSC,
                LaneRingBuffer::<T, F>::IS_MPSC,
                LaneRingBuffer::<T, F>::WAIT_FOR_SPACE,
                LaneRingBuffer::<T, F>::ERROR_ON_NO_SPACE,
                LaneRingBuffer::<T, F>::DYNAMIC_SIZE,
                LaneRingBuffer::<T, F>::LOCK_POP,
                LaneRingBuffer::<T, F>::IS_ATOMIC,
            )
        }

        assert_eq!(
            decode::<u64, MULTI_SPSC_RING_BUFFER_FLAGS>(),
            (true, false, false, true, false, false, false)
        );
        assert_eq!(
            decode::<u64, MULTI_MPSC_RING_BUFFER_FLAGS>(),
            // IsAtomic == IsMPSC, per the C++ (storage is atomic regardless —
            // divergence 7).
            (false, true, true, false, false, false, true)
        );
        assert_eq!(
            decode::<u64, MULTI_EXT_RING_BUFFER_FLAGS>(),
            (true, false, false, false, true, false, false)
        );
        // LockPop is decoded (and then ignored — divergence 9).
        assert_eq!(
            decode::<u64, { MULTI_MPSC_RING_BUFFER_FLAGS | RING_BUFFER_LOCK_POP }>(),
            (false, true, true, false, false, true, true)
        );
    }

    // -- CalculateSize -------------------------------------------------------

    #[test]
    fn calculate_size_follows_the_cpp_formula() {
        let depth = 16;
        let per_lane = <Spsc as RingLane>::calculate_size(depth);
        assert_eq!(
            per_lane,
            std::mem::size_of::<Spsc>()
                + (depth + 1) * std::mem::size_of::<RingBufferEntry<u64>>()
        );

        let total = MultiSpscRingBuffer::<u64>::calculate_size(4, 2, depth);
        assert_eq!(
            total,
            std::mem::size_of::<MultiSpscRingBuffer<u64>>() + 4 * 2 * per_lane
        );
    }

    #[test]
    fn calculate_size_with_zero_lanes_or_prios_is_just_the_base() {
        let base = std::mem::size_of::<MultiSpscRingBuffer<u64>>();
        assert_eq!(MultiSpscRingBuffer::<u64>::calculate_size(0, 8, 64), base);
        assert_eq!(MultiSpscRingBuffer::<u64>::calculate_size(8, 0, 64), base);
        assert_eq!(MultiSpscRingBuffer::<u64>::calculate_size(0, 0, 0), base);
    }

    #[test]
    fn calculate_size_with_zero_depth_still_counts_the_sentinel_slot() {
        // depth 0 => 1 slot: the reserved sentinel.
        let per_lane = <Spsc as RingLane>::calculate_size(0);
        assert_eq!(
            per_lane,
            std::mem::size_of::<Spsc>() + std::mem::size_of::<RingBufferEntry<u64>>()
        );
    }

    #[test]
    fn calculate_size_saturates_instead_of_wrapping() {
        // C++ size_t arithmetic would wrap and report an absurdly small size.
        assert_eq!(
            MultiSpscRingBuffer::<u64>::calculate_size(usize::MAX, usize::MAX, 8),
            usize::MAX
        );
        assert_eq!(
            MultiSpscRingBuffer::<u64>::calculate_size(2, 2, usize::MAX),
            usize::MAX
        );
        assert_eq!(<Spsc as RingLane>::calculate_size(usize::MAX), usize::MAX);
    }

    // -- construction / geometry --------------------------------------------

    #[test]
    fn construction_reports_lane_geometry() {
        let q = MultiSpscRingBuffer::<u64>::new(4, 3, 8);
        assert_eq!(q.num_lanes(), 4);
        assert_eq!(q.num_prios(), 3);
        assert_eq!(q.total_buffers(), 12);
        assert_eq!(q.lanes().len(), 12);
        for lane in q.lanes() {
            assert_eq!(lane.capacity(), 8);
            assert_eq!(lane.depth(), 9); // depth + 1 slots
            assert!(lane.is_empty());
        }
    }

    #[test]
    fn zero_lanes_or_zero_prios_yields_an_empty_container() {
        for (lanes, prios) in [(0, 4), (4, 0), (0, 0)] {
            let q = MultiSpscRingBuffer::<u64>::new(lanes, prios, 8);
            assert_eq!(q.num_lanes(), lanes);
            assert_eq!(q.num_prios(), prios);
            assert_eq!(q.total_buffers(), 0);
            assert!(q.lanes().is_empty());
            assert!(q.try_lane(0, 0).is_none());
        }
    }

    #[test]
    fn single_lane_single_prio_is_a_plain_ring_buffer() {
        let q = MultiSpscRingBuffer::<u64>::new(1, 1, 4);
        assert_eq!(q.total_buffers(), 1);
        assert!(q.lane(0, 0).push(7));
        assert_eq!(q.lane(0, 0).pop(), Some(7));
        assert!(q.try_lane(1, 0).is_none());
        assert!(q.try_lane(0, 1).is_none());
    }

    #[test]
    fn try_new_rejects_a_lane_count_that_overflows_usize() {
        // C++ would wrap silently and build the wrong number of lanes.
        assert!(MultiSpscRingBuffer::<u64>::try_new(usize::MAX, 2, 4).is_none());
        assert!(MultiSpscRingBuffer::<u64>::try_new(usize::MAX, usize::MAX, 4).is_none());
        // ... but the boundary case `MAX * 1` is only a (failing) allocation.
        assert!(MultiSpscRingBuffer::<u64>::try_new(usize::MAX, 1, 4).is_none());
        // Zero never overflows.
        assert!(MultiSpscRingBuffer::<u64>::try_new(usize::MAX, 0, 4).is_some());
    }

    #[test]
    #[should_panic(expected = "overflows usize or memory is exhausted")]
    fn new_panics_where_try_new_returns_none() {
        let _ = MultiSpscRingBuffer::<u64>::new(usize::MAX, 2, 4);
    }

    // -- GetLane index arithmetic -------------------------------------------

    #[test]
    fn get_lane_maps_to_lane_id_times_num_prios_plus_prio() {
        let num_lanes = 5;
        let num_prios = 3;
        let q = MultiSpscRingBuffer::<u64>::new(num_lanes, num_prios, 4);

        // Tag every lane with a value derived from its coordinates ...
        for lane_id in 0..num_lanes {
            for prio in 0..num_prios {
                let tag = (lane_id * 100 + prio) as u64;
                assert!(q.lane(lane_id, prio).push(tag));
            }
        }
        // ... and confirm the flat row-major position the C++ computes.
        for lane_id in 0..num_lanes {
            for prio in 0..num_prios {
                let idx = lane_id * num_prios + prio;
                let expected = (lane_id * 100 + prio) as u64;
                assert_eq!(q.lanes()[idx].pop(), Some(expected));
            }
        }
    }

    #[test]
    fn lanes_are_independent() {
        let q = MultiSpscRingBuffer::<u64>::new(2, 2, 4);
        assert!(q.lane(0, 0).push(1));
        assert!(q.lane(0, 0).push(2));

        assert_eq!(q.lane(0, 0).len(), 2);
        assert_eq!(q.lane(0, 1).len(), 0);
        assert_eq!(q.lane(1, 0).len(), 0);
        assert_eq!(q.lane(1, 1).len(), 0);
        assert!(q.lane(0, 1).pop().is_none());
        assert!(q.lane(1, 1).pop().is_none());
        assert_eq!(q.lane(0, 0).pop(), Some(1));
    }

    #[test]
    fn bounds_are_checked_per_coordinate_like_the_cpp_asserts() {
        let q = MultiSpscRingBuffer::<u64>::new(2, 3, 4);
        // In-range corners.
        assert!(q.try_lane(0, 0).is_some());
        assert!(q.try_lane(1, 2).is_some());
        // One past each coordinate. Note (0, 3) and (1, 0) both compute index
        // 3, but only the latter is legal: the C++ asserts each coordinate.
        assert!(q.try_lane(2, 0).is_none());
        assert!(q.try_lane(0, 3).is_none());
        assert!(q.try_lane(1, 0).is_some());
        assert!(q.try_lane(usize::MAX, 0).is_none());
        assert!(q.try_lane(0, usize::MAX).is_none());
    }

    #[test]
    #[should_panic(expected = "lane (2, 0) is out of range")]
    fn lane_panics_past_num_lanes() {
        let q = MultiSpscRingBuffer::<u64>::new(2, 3, 4);
        let _ = q.lane(2, 0);
    }

    #[test]
    #[should_panic(expected = "lane (0, 3) is out of range")]
    fn lane_panics_past_num_prios_even_when_the_flat_index_is_valid() {
        let q = MultiSpscRingBuffer::<u64>::new(2, 3, 4);
        // Flat index 3 exists (it is lane (1, 0)), but prio 3 does not.
        let _ = q.lane(0, 3);
    }

    #[test]
    #[should_panic(expected = "lane (9, 9) is out of range")]
    fn lane_mut_panics_out_of_range() {
        let mut q = MultiSpscRingBuffer::<u64>::new(2, 3, 4);
        let _ = q.lane_mut(9, 9);
    }

    #[test]
    fn lane_mut_reaches_the_same_lane_as_lane() {
        let mut q = MultiSpscRingBuffer::<u64>::new(3, 2, 4);
        q.lane_mut(2, 1).set_assigned_worker_id(11);
        q.lane_mut(2, 1).set_signal_fd(7);
        q.lane_mut(2, 1).set_tid(4242);
        assert_eq!(q.lane(2, 1).assigned_worker_id(), 11);
        assert_eq!(q.lanes()[2 * 2 + 1].signal_fd(), 7);
        assert_eq!(q.lane(2, 1).tid(), 4242);
        // Untouched neighbours keep the constructor's defaults.
        assert_eq!(q.lane(2, 0).assigned_worker_id(), 0);
        assert_eq!(q.lane(2, 0).signal_fd(), -1);
        assert_eq!(q.lane(2, 0).tid(), 0);

        assert!(q.try_lane_mut(0, 0).is_some());
        assert!(q.try_lane_mut(3, 0).is_none());
        q.lanes_mut()[0].set_assigned_worker_id(5);
        assert_eq!(q.lane(0, 0).assigned_worker_id(), 5);
    }

    // -- worker metadata -----------------------------------------------------

    #[test]
    fn active_defaults_to_true_and_toggles_through_a_shared_ref() {
        let q = MultiSpscRingBuffer::<u64>::new(1, 1, 4);
        assert!(q.lane(0, 0).is_active());
        q.lane(0, 0).set_active(false);
        assert!(!q.lane(0, 0).is_active());
        q.lane(0, 0).set_active(true);
        assert!(q.lane(0, 0).is_active());
    }

    // -- lane semantics: fill, boundaries, FIFO ------------------------------

    #[test]
    fn spsc_lane_fills_to_capacity_then_errors_and_restores_tail() {
        let q = MultiSpscRingBuffer::<u64>::new(1, 1, 3);
        let lane = q.lane(0, 0);

        for i in 0..3 {
            assert!(lane.push(i), "push {i} within capacity must succeed");
        }
        assert!(lane.is_full());
        assert_eq!(lane.len(), 3);
        assert_eq!(lane.tail(), 3);

        // Full: ErrorOnNoSpace reports failure AND restores the claimed slot.
        assert!(!lane.push(99));
        assert_eq!(lane.tail(), 3, "tail must be restored on a failed push");
        assert_eq!(lane.len(), 3);
        assert!(lane.is_full());

        // FIFO order, and the rejected element never entered.
        assert_eq!(lane.pop(), Some(0));
        assert_eq!(lane.pop(), Some(1));
        assert_eq!(lane.pop(), Some(2));
        assert_eq!(lane.pop(), None);
        assert!(lane.is_empty());
        assert!(!lane.is_full());

        // ... and the lane is reusable afterwards.
        assert!(lane.push(42));
        assert_eq!(lane.pop(), Some(42));
    }

    #[test]
    fn spsc_lane_wraps_around_the_slot_boundary() {
        let q = MultiSpscRingBuffer::<u64>::new(1, 1, 2);
        let lane = q.lane(0, 0);
        // Drive many laps through the 3 slots: exercises `% slots` wrap-around
        // and slot reuse after the consumer releases each entry.
        for i in 0..50u64 {
            assert!(lane.push(i));
            assert_eq!(lane.pop(), Some(i));
        }
        assert!(lane.is_empty());
        assert_eq!(lane.head(), 50);
        assert_eq!(lane.tail(), 50);
    }

    #[test]
    fn zero_depth_lane_is_both_empty_and_full() {
        let q = MultiSpscRingBuffer::<u64>::new(2, 1, 0);
        let lane = q.lane(0, 0);
        assert_eq!(lane.capacity(), 0);
        assert_eq!(lane.depth(), 1); // the reserved sentinel slot
        assert!(lane.is_empty());
        assert!(lane.is_full());
        assert!(!lane.push(1), "a zero-capacity lane can never accept a push");
        assert_eq!(lane.tail(), 0, "the failed push restored the tail");
        assert_eq!(lane.pop(), None);
        assert_eq!(lane.len(), 0);
    }

    #[test]
    fn clear_resets_cursors_and_drops_pending_entries() {
        let mut q = MultiSpscRingBuffer::<u64>::new(2, 1, 4);
        assert!(q.lane(0, 0).push(1));
        assert!(q.lane(0, 0).push(2));
        assert!(q.lane(1, 0).push(3));

        q.lane_mut(0, 0).clear();
        assert!(q.lane(0, 0).is_empty());
        assert_eq!(q.lane(0, 0).head(), 0);
        assert_eq!(q.lane(0, 0).tail(), 0);
        assert_eq!(q.lane(0, 0).pop(), None, "cleared flags gate the payloads");
        // Only the cleared lane was touched.
        assert_eq!(q.lane(1, 0).len(), 1);
        assert_eq!(q.lane(1, 0).pop(), Some(3));

        // The cleared lane still works.
        assert!(q.lane(0, 0).push(9));
        assert_eq!(q.lane(0, 0).pop(), Some(9));
        q.lane_mut(0, 0).reset(); // alias for clear()
        assert!(q.lane(0, 0).is_empty());
    }

    /// Pins the preserved C++ quirk documented in divergence 10.
    #[test]
    fn dynamic_size_full_push_wedges_the_lane_like_cpp() {
        let q = MultiExtRingBuffer::<u64>::new(1, 1, 2);
        let lane = q.lane(0, 0);

        assert!(lane.push(1));
        assert!(lane.push(2));
        assert_eq!(lane.tail(), 2);

        // Full: the DynamicSize branch returns false WITHOUT restoring tail.
        assert!(!lane.push(3));
        assert_eq!(lane.tail(), 3, "the C++ leaks the claimed slot");

        // The two live elements still come out ...
        assert_eq!(lane.pop(), Some(1));
        assert_eq!(lane.pop(), Some(2));

        // ... and then the lane is wedged forever: head can never reach the
        // phantom slot, so Pop always fails while Empty() reports non-empty.
        assert_eq!(lane.pop(), None);
        assert!(!lane.is_empty());
        assert_eq!(lane.len(), 1, "reports one element that cannot be popped");
        assert!(lane.push(4), "further pushes still 'succeed' ...");
        assert_eq!(lane.pop(), None, "... but nothing can ever be popped again");
    }

    // -- concurrency ---------------------------------------------------------

    /// MPSC lanes are the reason `multi_mpsc_ring_buffer` exists: several
    /// producers, one consumer, per lane, with WAIT_FOR_SPACE forcing producers
    /// to spin on a deliberately tiny ring.
    #[test]
    fn mpsc_lane_delivers_every_element_exactly_once() {
        const PRODUCERS: u64 = 4;
        const PER_PRODUCER: u64 = 500;
        const TOTAL: usize = (PRODUCERS * PER_PRODUCER) as usize;

        let q = MultiMpscRingBuffer::<u64>::new(1, 1, 8);
        let mut seen = vec![0usize; TOTAL];

        thread::scope(|s| {
            for p in 0..PRODUCERS {
                let lane = q.lane(0, 0);
                s.spawn(move || {
                    for i in 0..PER_PRODUCER {
                        // Unique across producers, so duplicates/losses show up.
                        assert!(lane.push(p * PER_PRODUCER + i));
                    }
                });
            }

            let lane = q.lane(0, 0);
            let seen = &mut seen;
            s.spawn(move || {
                let mut count = 0usize;
                while count < TOTAL {
                    if let Some(v) = lane.pop() {
                        seen[v as usize] += 1;
                        count += 1;
                    } else {
                        std::hint::spin_loop();
                    }
                }
            });
        });

        assert!(
            seen.iter().all(|&n| n == 1),
            "every element must be delivered exactly once"
        );
        assert!(q.lane(0, 0).is_empty());
        assert_eq!(q.lane(0, 0).head(), q.lane(0, 0).tail());
    }

    /// The container itself must be shareable: distinct lanes are touched
    /// concurrently through `&MultiRingBuffer`.
    #[test]
    fn distinct_lanes_are_driven_concurrently() {
        const LANES: usize = 4;
        const PRIOS: usize = 2;
        const ITEMS: u64 = 200;

        let q = MultiMpscRingBuffer::<u64>::new(LANES, PRIOS, 4);
        let popped = AtomicUsize::new(0);

        thread::scope(|s| {
            for lane_id in 0..LANES {
                for prio in 0..PRIOS {
                    let q = &q;
                    let popped = &popped;
                    s.spawn(move || {
                        let lane = q.lane(lane_id, prio);
                        let tag = (lane_id * PRIOS + prio) as u64 * ITEMS;
                        for i in 0..ITEMS {
                            assert!(lane.push(tag + i));
                            // Drain as we go: WAIT_FOR_SPACE would otherwise
                            // spin forever on a depth-4 lane.
                            while let Some(v) = lane.pop() {
                                assert!(v >= tag, "lane {lane_id}/{prio} saw a foreign value");
                                popped.fetch_add(1, StdOrdering::Relaxed);
                            }
                        }
                        while let Some(_v) = lane.pop() {
                            popped.fetch_add(1, StdOrdering::Relaxed);
                        }
                    });
                }
            }
        });

        assert_eq!(
            popped.load(StdOrdering::SeqCst),
            LANES * PRIOS * ITEMS as usize
        );
        for lane in q.lanes() {
            assert!(lane.is_empty());
        }
    }

    /// A single consumer racing many producers must observe every element
    /// exactly once *and in per-producer order*: each producer's own sequence
    /// is FIFO even though the lanes interleave arbitrarily.
    #[test]
    fn mpsc_lane_preserves_per_producer_order() {
        const PRODUCERS: u64 = 3;
        const PER_PRODUCER: u64 = 400;

        let q = MultiMpscRingBuffer::<u64>::new(1, 1, 4);
        let mut last = vec![None::<u64>; PRODUCERS as usize];

        thread::scope(|s| {
            for p in 0..PRODUCERS {
                let lane = q.lane(0, 0);
                s.spawn(move || {
                    for i in 0..PER_PRODUCER {
                        // Encode (producer, sequence) in one word.
                        assert!(lane.push(p * 1_000_000 + i));
                    }
                });
            }

            let lane = q.lane(0, 0);
            let last = &mut last;
            s.spawn(move || {
                let mut count = 0;
                while count < PRODUCERS * PER_PRODUCER {
                    if let Some(v) = lane.pop() {
                        let (p, i) = (v / 1_000_000, v % 1_000_000);
                        let slot = &mut last[p as usize];
                        match *slot {
                            None => assert_eq!(i, 0, "producer {p} lost its first element"),
                            Some(prev) => {
                                assert_eq!(i, prev + 1, "producer {p} delivered out of order")
                            }
                        }
                        *slot = Some(i);
                        count += 1;
                    } else {
                        std::hint::spin_loop();
                    }
                }
            });
        });

        // Every producer ran to completion.
        assert!(last.iter().all(|&l| l == Some(PER_PRODUCER - 1)));
    }

    // NOTE: there is deliberately no multi-*consumer* test. MPSC means
    // Multiple Producer, *Single* Consumer, and divergence 16 documents why
    // the C++'s lock-free Pop genuinely cannot support concurrent consumers
    // (it advances head_ with a plain store, so a stalled consumer can regress
    // head_ a full lap and wedge the lane). Such a test would hang, which is
    // the faithful behavior — not something to assert in CI.

    // -- generic container over a foreign lane type -------------------------

    /// `MultiRingBuffer` is generic over `RingLane`, which is how
    /// `ring_buffer.rs`'s type will slot in (divergence 1). A stub lane proves
    /// the container touches nothing beyond the trait.
    #[test]
    fn container_works_with_any_ring_lane_impl() {
        struct StubLane {
            depth: usize,
        }
        impl RingLane for StubLane {
            fn with_depth(depth: usize) -> Self {
                Self { depth }
            }
            fn calculate_size(depth: usize) -> usize {
                7 + depth
            }
        }

        let q: MultiRingBuffer<StubLane> = MultiRingBuffer::new(3, 4, 9);
        assert_eq!(q.total_buffers(), 12);
        assert_eq!(q.lanes().len(), 12);
        assert!(q.lanes().iter().all(|l| l.depth == 9));
        assert_eq!(q.lane(2, 3).depth, 9);
        assert_eq!(
            MultiRingBuffer::<StubLane>::calculate_size(3, 4, 9),
            std::mem::size_of::<MultiRingBuffer<StubLane>>() + 12 * (7 + 9)
        );
    }

    // -- layout --------------------------------------------------------------

    #[test]
    fn entry_layout_matches_the_cpp_struct() {
        // C++: struct { abitfield32_t flags_; T data_; } — flags first.
        assert_eq!(std::mem::size_of::<RingBufferEntry<u64>>(), 16);
        assert_eq!(std::mem::align_of::<RingBufferEntry<u64>>(), 8);
        assert_eq!(std::mem::size_of::<RingBufferEntry<u32>>(), 8);
        // POD payloads: no drop glue, as shared-memory residency demands.
        assert!(!std::mem::needs_drop::<RingBufferEntry<u64>>());
    }
}
