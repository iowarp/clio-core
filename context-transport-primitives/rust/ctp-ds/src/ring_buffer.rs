// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).

//! Lock-free IPC ring buffers — the CTP **task queues**. Port of
//! `include/clio_ctp/data_structures/ipc/ring_buffer.h` (`ctp::ipc`).
//!
//! One `#[repr(C)]` type, [`RingBuffer<T, FLAGS>`], covers every C++
//! configuration; the SPSC / MPSC / circular-MPSC / MPMC / extensible variants
//! are `FLAGS` type aliases, exactly as in the C++ (`spsc_ring_buffer`,
//! `mpsc_ring_buffer`, …).
//!
//! # Storage: an inline trailing slice, not a `vector`
//!
//! The C++ holds entries in a `vector<entry_type, AllocT>`, i.e. a separate
//! allocation reached through an `OffsetPtr`. `shm_vector` is not ported yet,
//! and a ring buffer that must resolve an `OffsetPtr` (registry lookup) on
//! every `Push`/`Pop` would be the wrong shape for a task queue anyway. This
//! port instead stores the entries **inline, immediately after the header**, as
//! a custom DST with a trailing `[RingBufferEntry<T>]`:
//!
//! ```text
//! [ head | tail | worker_id | signal_fd | tid | active | pop_lock | e0 | e1 | … | e_depth ]
//! ```
//!
//! This is what the C++ `CalculateSize(depth)` (`sizeof(ring_buffer) +
//! (depth+1) * sizeof(entry)`) already assumes, and it is *maximally*
//! position-independent per MEMORY_DESIGN.md pillar 3: the entry array is found
//! at a compile-time offset from the header's own address, so the structure
//! stores **no offsets and no pointers at all** and is correct at any mapping
//! base in any process (pinned by `position_independent_after_byte_copy`). All
//! cross-process state is in-segment atomics. Because `&Self` is a fat
//! reference covering header **and** entries, slot access is ordinary safe
//! indexing (`self.queue[idx]`) with correct provenance — the hot path carries
//! no pointer arithmetic.
//!
//! Placement: [`RingBuffer::layout`] / [`RingBuffer::calculate_size`] size the
//! block, [`RingBuffer::init_at`] constructs one in it, [`RingBuffer::from_raw`]
//! attaches to one another process built. [`RingBufferBox`] is a process-local
//! owner for tests and non-shared use.
//!
//! # C++ → Rust name mapping
//!
//! | C++ (`ctp::ipc`) | Rust | Notes |
//! |---|---|---|
//! | `enum RingQueueFlag` | `RING_BUFFER_*` consts | plain `u32` consts; used as a const-generic arg |
//! | `RING_BUFFER_SPSC_FLAGS` … `RING_BUFFER_LOCK_POP` | [`RING_BUFFER_SPSC_FLAGS`] … [`RING_BUFFER_LOCK_POP`] | same bit values |
//! | `RingBufferEntry<T>` | [`RingBufferEntry<T>`] | `#[repr(C)]`, byte-identical to the C++ |
//! | `RingBufferEntry::flags_` (`abitfield32_t`) | [`RingBufferEntry::flags`] | `Atomic<u32>`; the C++ `bitfield` wrapper collapses to its single `bits_` field |
//! | `RingBufferEntry::data_` | [`RingBufferEntry::get_data`] / [`set_data`](RingBufferEntry::set_data) | `UnsafeCell<T>` (same layout); see divergence 6 |
//! | `RingBufferEntry()` / `RingBufferEntry(const T&)` | [`RingBufferEntry::new`] / [`with_data`](RingBufferEntry::with_data) | |
//! | `IsReady` / `IsReadyDevice` / `IsReadySystem` | [`is_ready`](RingBufferEntry::is_ready) / [`is_ready_device`](RingBufferEntry::is_ready_device) / [`is_ready_system`](RingBufferEntry::is_ready_system) | `T Any(mask)` → `bool` |
//! | `SetReady` / `SetReadySystem` / `ClearReady` | [`set_ready`](RingBufferEntry::set_ready) / [`set_ready_system`](RingBufferEntry::set_ready_system) / [`clear_ready`](RingBufferEntry::clear_ready) | |
//! | `GetData()` (both overloads) | [`get_data`](RingBufferEntry::get_data) | returns `T` by value (`T: Copy`); a `&mut T` from `&self` is unrepresentable |
//! | `ring_buffer<T, AllocT, FLAGS>` | [`RingBuffer<T, FLAGS>`] | no `AllocT` param — see divergence 1 |
//! | `queue_` (`vector<entry_type, AllocT>`) | `RingBuffer::queue` (`[RingBufferEntry<T>]`) | inline trailing slice; see above |
//! | `head_` / `tail_` (`opt_atomic<u64, IsAtomic>`) | `RingBuffer::head` / `tail` (`Atomic<u64>`) | always atomic; see divergence 2 |
//! | `assigned_worker_id_` / `signal_fd_` / `tid_` | private `Atomic<u32>` / `Atomic<i32>` / `Atomic<i32>` | see divergence 3 |
//! | `active_` (`opt_atomic<bool, IsAtomic>`) | `RingBuffer::active` (`AtomicBool`) | see divergence 2 |
//! | `pop_lock_` (`ctp::ipc::Mutex`) | `RingBuffer::pop_lock` (`ctp_thread::mutex::Mutex`) | the ported ticket lock |
//! | `CalculateSize(depth)` | [`RingBuffer::calculate_size`] | + [`RingBuffer::layout`] (align too); see divergence 4 |
//! | `ring_buffer(AllocT*, depth = 1024)` | [`RingBuffer::init_at`] | placement construction; [`DEFAULT_DEPTH`] = 1024 |
//! | `ring_buffer(const ring_buffer&)` | [`RingBuffer::init_copy_at`] | copy ctor needs a destination block |
//! | `ring_buffer(ring_buffer&&) = delete` | — | no move ctor exists to port; the type is `!Sized` and never moves |
//! | `~ring_buffer()` | — | trivial in C++; no `Drop` here (MEMORY_DESIGN.md) |
//! | `Get/SetAssignedWorkerId` | [`assigned_worker_id`](RingBuffer::assigned_worker_id) / [`set_assigned_worker_id`](RingBuffer::set_assigned_worker_id) | |
//! | `Get/SetSignalFd` | [`signal_fd`](RingBuffer::signal_fd) / [`set_signal_fd`](RingBuffer::set_signal_fd) | |
//! | `Get/SetTid` (`pid_t`) | [`tid`](RingBuffer::tid) / [`set_tid`](RingBuffer::set_tid) | `pid_t` → `i32`; see divergence 3 |
//! | `IsActive` / `SetActive` | [`is_active`](RingBuffer::is_active) / [`set_active`](RingBuffer::set_active) | |
//! | `Size` / `Capacity` / `GetDepth` | [`size`](RingBuffer::size) / [`capacity`](RingBuffer::capacity) / [`get_depth`](RingBuffer::get_depth) | `get_depth` returns slot count = `depth + 1` (C++ quirk, kept) |
//! | `Empty` / `Full` | [`is_empty`](RingBuffer::is_empty) / [`is_full`](RingBuffer::is_full) | renamed per Rust convention |
//! | `Push` / `TryPush` / `Emplace` | [`push`](RingBuffer::push) / [`try_push`](RingBuffer::try_push) / [`emplace`](RingBuffer::emplace) | see divergence 5 |
//! | `PushSystem` | [`push_system`](RingBuffer::push_system) | host-equivalent to `push`; see divergence 8 |
//! | `Pop(T&)` / `TryPop(T&)` | [`pop`](RingBuffer::pop) / [`try_pop`](RingBuffer::try_pop) | `bool` + out-param → `Option<T>` (divergence 7) |
//! | `PopUnlocked(T&)` (private) | `pop_unlocked` (private) | |
//! | `PopDevice(T&)` | [`pop_device`](RingBuffer::pop_device) | host-equivalent to `pop`; see divergences 8, 12 |
//! | `Peek(u64, T&)` | [`peek`](RingBuffer::peek) | → `Option<T>`; see divergence 11 |
//! | `GetHead` / `GetTail` / `GetHeadDevice` / `GetTailDevice` | [`get_head`](RingBuffer::get_head) / [`get_tail`](RingBuffer::get_tail) / [`get_head_device`](RingBuffer::get_head_device) / [`get_tail_device`](RingBuffer::get_tail_device) | |
//! | `Clear` / `Reset` | [`clear`](RingBuffer::clear) / [`reset`](RingBuffer::reset) | |
//! | `Resize(size_t)` | — | **not ported** (divergence 9) |
//! | `ext_ring_buffer<T>` | [`ExtRingBuffer<T>`] | **wedges the queue when full** — divergence 10 |
//! | `spsc_ring_buffer<T>` | [`SpscRingBuffer<T>`] | |
//! | `mpsc_ring_buffer<T>` | [`MpscRingBuffer<T>`] | |
//! | `circular_mpsc_ring_buffer<T>` | [`CircularMpscRingBuffer<T>`] | overwrites when full — divergence 11 |
//! | `mpmc_ring_buffer<T>` | [`MpmcRingBuffer<T>`] | |
//! | — | [`RingBufferBox<T, FLAGS>`] | **added**: process-local owning handle |
//! | — | [`RingBuffer::init_at`] / [`from_raw`](RingBuffer::from_raw) | **added**: the segment-placement surface replacing `AllocT*` |
//!
//! # Semantic divergences from the C++
//!
//! 1. **No `AllocT` parameter, no `ShmContainer` base, and therefore no ABI
//!    compatibility with the C++ `ring_buffer`.** The C++ derives from
//!    `ShmContainer<AllocT>` (an `OffsetPtr<void> this_`) and stores a
//!    `vector<entry_type, AllocT>`; both are replaced by the inline trailing
//!    slice described above, so the *header* bytes differ from C++ and a
//!    ring_buffer **cannot yet be shared between the C++ and Rust
//!    implementations**. `RingBufferEntry<T>` itself *is* byte-identical, so
//!    only the header must be reconciled once `shm_vector` lands; until then,
//!    cross-language queues stay on the C++ side (MEMORY_DESIGN.md pillar 1:
//!    one segment, one owner). Rust-to-Rust cross-process use is fully
//!    supported today via [`init_at`](RingBuffer::init_at) /
//!    [`from_raw`](RingBuffer::from_raw).
//! 2. **`head`/`tail`/`active` are always real atomics.** C++ picks
//!    `opt_atomic<T, IsAtomic>`, so the SPSC variants use `nonatomic` (plain
//!    loads/stores) — a data race by both the C++ and Rust models as soon as
//!    the producer and consumer are different threads/processes, which for a
//!    *cross-process* SPSC queue they always are. The layout is identical (8/8/1
//!    bytes either way), so nothing is lost but a negligible amount of SPSC
//!    speed, and `Nonatomic<T>` is `!Sync` in Rust anyway. `active_` uses
//!    `std::sync::atomic::AtomicBool` because `ctp_types::atomic::Atomic<T>` is
//!    integers-only (its own divergence 3); layout is unchanged.
//! 3. **`assigned_worker_id_` / `signal_fd_` / `tid_` are stored as atomics.**
//!    They are plain fields in C++ with non-const setters, i.e. racy by
//!    construction (the orchestrator writes them while workers read). Making
//!    them `Atomic<u32>`/`Atomic<i32>` keeps the layout byte-identical, keeps
//!    the setters on `&self` (a `&mut self` setter is unusable on a shared
//!    queue), and removes the race. `pid_t` → `i32`: the C++ header itself does
//!    `using pid_t = int` on Windows, and it is `int` on Linux/macOS too.
//! 4. **`calculate_size` counts the real layout.** The C++ returns
//!    `sizeof(ring_buffer) + (depth+1)*sizeof(entry)`, which double-counts the
//!    `vector` header and ignores alignment padding. [`RingBuffer::layout`]
//!    returns the exact `Layout` (`Layout::extend` reproduces the `repr(C)`
//!    rule, asserted against the compiler's own layout by
//!    `layout_matches_the_compiler`); `calculate_size` is its `.size()`. It
//!    **panics** on a `depth` whose layout would overflow `isize`, where the
//!    C++ silently wraps.
//! 5. **`Emplace` collapses to `push(val)`.** The C++ variadic `Emplace` is not
//!    in-place: it does `entry.data_ = T(std::forward<Args>(args)...)`, i.e.
//!    constructs a temporary and assigns. With `T: Copy` that is exactly
//!    `push`, so the variadic form carries no information to port.
//! 6. **`T: ShmSafe + Default`, and `data_` lives in an `UnsafeCell<T>`.**
//!    `ShmSafe` (`Copy + 'static`, POD-shaped) is MEMORY_DESIGN.md's admission
//!    ticket for shared memory. `Default` supplies the C++ `T()`
//!    value-initialization used by `RingBufferEntry()` and by the slot-release
//!    store in `Pop`; C++ default-initializes `data_` (indeterminate for a POD),
//!    which Rust cannot express, so fresh slots are `T::default()` instead of
//!    uninitialized. `UnsafeCell` is `repr(transparent)`, so the entry layout is
//!    unchanged; it is required to write `data` through `&self` at all.
//! 7. **`bool Pop(T &val)` → `pop() -> Option<T>`.** Same for `TryPop`,
//!    `PopDevice` and `Peek`. The C++ leaves `val` untouched on failure, so no
//!    information is lost. `Empty`/`Full` become `is_empty`/`is_full`.
//! 8. **`*_system` / `*_device` are not stronger than their plain forms.** They
//!    reproduce the C++ **host** definitions exactly (`ctp_types::atomic`
//!    divergences 1–2: `load_system` == `load_device` == `load(SeqCst)`), so on
//!    the host `push_system` differs from `push` only in lacking the
//!    `DYNAMIC_SIZE` branch, and `pop_device` differs from `pop` as in
//!    divergence 12. The names are kept so ported call sites read identically
//!    and a future GPU backend can specialize them. Every ordering is `SeqCst`,
//!    matching the C++ (whose `load`/`store`/`fetch_add` default to `seq_cst`).
//! 9. **`Resize` is not ported.** It builds a second `ring_buffer` from the
//!    allocator and move-assigns its `vector` — impossible for an
//!    inline-storage queue, and unreachable in practice: the only `DynamicSize`
//!    alias (`ext_ring_buffer`) never calls it, because the C++ `Emplace`
//!    resize path is a `return false` stub ("not implemented for now").
//!    Re-sizing means: allocate a new block, `init_at`, drain, repeat.
//! 10. **Upstream bug, ported verbatim: a full [`ExtRingBuffer`] wedges
//!     permanently.** `Emplace`'s `DynamicSize` branch returns `false`
//!     *without* undoing its `tail_.fetch_add(1)` (the `ErrorOnNoSpace` branch
//!     does `fetch_sub(1)`). The queue is left with a claimed-but-never-marked-
//!     ready slot, so `tail` is permanently one too high and the consumer stops
//!     dead at that slot forever — every later `Pop` returns false even though
//!     entries behind it are ready. Pinned by
//!     `ext_ring_buffer_wedges_forever_after_a_full_push_cpp_bug`. Reported
//!     rather than silently fixed, since this is a cross-language type.
//! 11. **Data races the C++ contract permits are preserved, not fixed.** Two
//!     configurations let a producer write a slot a consumer is reading:
//!     (a) [`CircularMpscRingBuffer`] (no space policy at all) overwrites the
//!     slot under a lagging consumer *by design*; (b) [`peek`](RingBuffer::peek)
//!     reads a slot that a concurrent `pop` may be releasing. Both are formally
//!     data races in Rust and in C++. They stay safe (not `unsafe fn`) to match
//!     the C++ surface, and they are benign in practice for exactly the reason
//!     `ShmSafe` exists — its contract is that **any** bit pattern a cooperating
//!     process writes is a valid `T`, so a torn read yields a valid, merely
//!     stale/mixed, value. Callers wanting soundness use a space policy
//!     (`Spsc`/`Mpsc`/`Mpmc`) and confine `peek` to the consumer. The
//!     concurrent-overwrite behaviour is therefore tested single-threaded only.
//!     Note the overwrite is worse than "drops the oldest entry", which is how
//!     the C++ alias comment reads: the lapping producer writes *in place*, so
//!     the clobbered cursor delivers the **new** payload out of order, and the
//!     lapped entry's own cursor then finds its flag already spent — wedging the
//!     ring exactly as in divergence 10 (pinned by
//!     `circular_queue_overwrites_when_full`).
//! 12. **Hazards inherited from the C++ algorithm** (faithfully ported, worth
//!     knowing): `Emplace`'s `ErrorOnNoSpace` rollback (`tail_.fetch_sub(1)`)
//!     is only correct for a **single** producer — with concurrent producers it
//!     retracts *someone else's* slot and two producers then write the same
//!     entry; it is safe only because every `ErrorOnNoSpace` alias is SPSC.
//!     `PopDevice` never takes `pop_lock` even under `RING_BUFFER_LOCK_POP`, and
//!     (unlike `PopUnlocked`) does not release the slot's copy of the element,
//!     so it re-introduces the #620 leak for owning `T`s. `Clear`/`Reset` are
//!     not thread-safe against concurrent producers/consumers.
//! 13. **`if constexpr` → `if`.** The `FLAGS` tests are `const fn`s over a
//!     const-generic parameter, so each monomorphization folds them away; the
//!     difference from `if constexpr` is that dead branches must still *compile*,
//!     which they do.
//! 14. **Upstream bug found while porting: `RING_BUFFER_LOCK_POP` is a
//!     *correctness requirement* for multiple consumers, not an optimization.**
//!     The C++ `Pop()` comment states that "the CAS in the lock-free path
//!     already prevents duplicate delivery" and that the lock merely removes
//!     contention and unfairness. That is **false once the ring laps.**
//!     `PopUnlocked` samples `head`, *then* tests the slot's ready flag and
//!     claims it; nothing revalidates `head` in between. A consumer preempted in
//!     that window resumes to find its slot ready again — refilled by a producer
//!     a full lap later — wins the CAS on that **future** entry, delivers it out
//!     of order, and then stores `stale_head + 1`, **rewinding `head`** over
//!     entries other consumers already delivered. The rewound-over slots all
//!     have `flags == 0`, so `Pop` returns `None` forever while the waiting
//!     producers block on a ring that can never drain: permanent deadlock.
//!     Reproduced here in seconds (1 producer, 4 consumers, depth 64: `head`
//!     rewound 4548 → 4526, i.e. a consumer holding head 4525 claimed the entry
//!     the producer had written at cursor 4590, since 4590 % 65 == 4525 % 65).
//!     The preconditions are pinned deterministically by
//!     `stale_head_lets_an_unlocked_consumer_claim_a_future_slot_cpp_bug`, and
//!     the fix-by-configuration by `locked_pop_closes_the_stale_head_window`.
//!     **Consequences for callers:** use [`MpmcRingBuffer`] (or any
//!     `RING_BUFFER_LOCK_POP` flag set) whenever more than one consumer pops;
//!     [`MpscRingBuffer`]/[`SpscRingBuffer`] are safe *only* with a single
//!     consumer, which is what their names promise. [`pop_device`](RingBuffer::pop_device)
//!     never takes the lock and so is **never** safe for multiple consumers
//!     (divergence 12). Ported verbatim — this is a live C++ hazard to fix
//!     upstream (in both languages: sample `head` *inside* the claim, e.g. CAS
//!     `head` from `head` to `head+1` and only then claim the slot), not
//!     something to paper over in the Rust port alone.

#![deny(unsafe_op_in_unsafe_fn)]

use std::alloc::{alloc, dealloc, handle_alloc_error, Layout};
use std::cell::UnsafeCell;
use std::fmt;
use std::ops::Deref;
use std::ptr::{self, NonNull};
use std::sync::atomic::{AtomicBool, Ordering};

use ctp_memory::ShmSafe;
use ctp_thread::mutex::Mutex;
use ctp_types::atomic::{threadfence, threadfence_system, Atomic, AtomicApi};

// ---------------------------------------------------------------------------
// RingQueueFlag — `enum RingQueueFlag : uint32_t`
// ---------------------------------------------------------------------------

/// Single producer, single consumer (the C++ then picks `nonatomic` counters;
/// this port does not — see divergence 2).
pub const RING_BUFFER_SPSC_FLAGS: u32 = 0x01;
/// Multiple producers, single consumer: atomic slot claim via `fetch_add`.
pub const RING_BUFFER_MPSC_FLAGS: u32 = 0x02;
/// Block until space is available.
pub const RING_BUFFER_WAIT_FOR_SPACE: u32 = 0x04;
/// Return `false` when there is no space.
pub const RING_BUFFER_ERROR_ON_NO_SPACE: u32 = 0x08;
/// Resize when full. The C++ resize path is an unimplemented stub — see
/// divergences 9 and 10 before using this.
pub const RING_BUFFER_DYNAMIC_SIZE: u32 = 0x10;
/// Fixed-size buffer (no dynamic resizing). Documentary: nothing branches on it.
pub const RING_BUFFER_FIXED_SIZE: u32 = 0x20;
/// Serialize `Pop()` with a [`Mutex`], enabling multi-consumer (MPMC) use.
///
/// **Required** for multiple consumers — despite the C++ comment calling it a
/// contention optimization, an unlocked `Pop` from two threads deadlocks the
/// queue. See divergence 14.
pub const RING_BUFFER_LOCK_POP: u32 = 0x40;

/// The C++ default template argument for `FLAGS`.
pub const DEFAULT_RING_BUFFER_FLAGS: u32 =
    RING_BUFFER_SPSC_FLAGS | RING_BUFFER_FIXED_SIZE | RING_BUFFER_ERROR_ON_NO_SPACE;

/// The C++ default `depth` constructor argument.
pub const DEFAULT_DEPTH: usize = 1024;

/// Bit 0 of `RingBufferEntry::flags` — "data ready for consumption".
const READY_BIT: u32 = 1;

/// `IsSPSC` — single producer, single consumer.
#[inline]
pub const fn is_spsc(flags: u32) -> bool {
    (flags & RING_BUFFER_SPSC_FLAGS) != 0
}

/// `IsMPSC` — multiple producers, single consumer.
#[inline]
pub const fn is_mpsc(flags: u32) -> bool {
    (flags & RING_BUFFER_MPSC_FLAGS) != 0
}

/// `WaitForSpace` — spin until the consumer frees the claimed slot.
#[inline]
pub const fn waits_for_space(flags: u32) -> bool {
    (flags & RING_BUFFER_WAIT_FOR_SPACE) != 0
}

/// `ErrorOnNoSpace` — fail the push when full.
#[inline]
pub const fn errors_on_no_space(flags: u32) -> bool {
    (flags & RING_BUFFER_ERROR_ON_NO_SPACE) != 0
}

/// `DynamicSize` — see divergences 9 and 10.
#[inline]
pub const fn is_dynamic_size(flags: u32) -> bool {
    (flags & RING_BUFFER_DYNAMIC_SIZE) != 0
}

/// `LockPop` — serialize `Pop` with the embedded mutex.
#[inline]
pub const fn locks_pop(flags: u32) -> bool {
    (flags & RING_BUFFER_LOCK_POP) != 0
}

/// `IsAtomic` — `IsMPSC` in the C++. Reported for parity only: this port always
/// uses atomic counters (divergence 2).
#[inline]
pub const fn is_atomic(flags: u32) -> bool {
    is_mpsc(flags)
}

// ---------------------------------------------------------------------------
// RingBufferEntry<T>
// ---------------------------------------------------------------------------

/// One slot: an atomic ready flag plus the payload.
///
/// `#[repr(C)]` and **byte-identical to the C++ `RingBufferEntry<T>`**
/// (`abitfield32_t flags_` collapses to its single `opt_atomic<u32, true> bits_`
/// field, and `UnsafeCell<T>` is `repr(transparent)`).
///
/// The ready flag is the publication protocol: a producer writes `data` and then
/// sets bit 0 with release ordering; a consumer acquires the flag (and, for
/// multi-consumer use, *claims* the slot) by CAS-ing it back to 0 before reading
/// `data`.
#[repr(C)]
pub struct RingBufferEntry<T> {
    /// C++ `flags_` (bit 0 = data ready). Public, as in the C++, so a consumer
    /// can run the claim CAS itself.
    pub flags: Atomic<u32>,
    /// C++ `data_`. Written through `&self` under the ready protocol, hence the
    /// `UnsafeCell` (divergence 6).
    data: UnsafeCell<T>,
}

// SAFETY: `ShmSafe` asserts `T` is POD-shaped, holds no process-local
// addresses, and is valid for any bit pattern a cooperating process writes —
// i.e. it is trivially `Send`/`Sync` in substance. The only *safe* ways to
// touch `data` through `&self` are reads (`get_data`); every write goes through
// the `unsafe` `set_data`, whose contract carries the exclusivity requirement.
// `flags` is atomic.
unsafe impl<T: ShmSafe> Send for RingBufferEntry<T> {}
unsafe impl<T: ShmSafe> Sync for RingBufferEntry<T> {}

impl<T: ShmSafe + Default> Default for RingBufferEntry<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T: ShmSafe + Default> RingBufferEntry<T> {
    /// C++ `RingBufferEntry() : flags_(0) {}`.
    ///
    /// The C++ leaves `data_` default-initialized (indeterminate for a POD);
    /// this port value-initializes it to `T::default()` (divergence 6).
    #[inline]
    pub fn new() -> Self {
        Self {
            flags: Atomic::new(0),
            data: UnsafeCell::new(T::default()),
        }
    }
}

impl<T: ShmSafe> RingBufferEntry<T> {
    /// C++ `explicit RingBufferEntry(const T &data) : flags_(0), data_(data) {}`.
    #[inline]
    pub fn with_data(data: T) -> Self {
        Self {
            flags: Atomic::new(0),
            data: UnsafeCell::new(data),
        }
    }

    /// C++ `IsReady()` — `flags_.Any(1)`.
    #[inline]
    pub fn is_ready(&self) -> bool {
        (self.flags.load(Ordering::SeqCst) & READY_BIT) != 0
    }

    /// C++ `IsReadyDevice()` — `flags_.AnyDevice(1)`. Host-equal to
    /// [`is_ready`](Self::is_ready) (divergence 8).
    #[inline]
    pub fn is_ready_device(&self) -> bool {
        (self.flags.load_device() & READY_BIT) != 0
    }

    /// C++ `IsReadySystem()` — `flags_.AnySystem(1)`. Host-equal to
    /// [`is_ready`](Self::is_ready) (divergence 8).
    #[inline]
    pub fn is_ready_system(&self) -> bool {
        (self.flags.load_system() & READY_BIT) != 0
    }

    /// C++ `SetReady()` — `threadfence_system(); flags_.SetBitsSystem(1);`.
    ///
    /// Publishes the preceding `data` write: the fence plus the `SeqCst`
    /// read-modify-write give release ordering, which the consumer's acquiring
    /// flag load/CAS pairs with.
    #[inline]
    pub fn set_ready(&self) {
        threadfence_system();
        self.flags.or_system(READY_BIT);
    }

    /// C++ `SetReadySystem()` — `flags_.SetBitsSystem(1)`.
    ///
    /// Note the C++ doc comment claims this issues `__threadfence_system()`
    /// first; the code does not (only [`set_ready`](Self::set_ready) does). The
    /// code is ported, not the comment — harmless on the host, where the
    /// `SeqCst` RMW alone already orders the preceding `data` write.
    #[inline]
    pub fn set_ready_system(&self) {
        self.flags.or_system(READY_BIT);
    }

    /// C++ `ClearReady()` — `flags_.UnsetBits(1)`, i.e. `bits_ &= ~1`.
    #[inline]
    pub fn clear_ready(&self) {
        self.flags.fetch_and(!READY_BIT, Ordering::SeqCst);
    }

    /// C++ `bitfield::Clear()` — `bits_ = 0`. Used by
    /// [`RingBuffer::clear`](RingBuffer::clear).
    #[inline]
    pub fn clear_flags(&self) {
        self.flags.store(0, Ordering::SeqCst);
    }

    /// C++ `GetData()` — by value, since `T: Copy` and a `&mut T` cannot be
    /// handed out from `&self`.
    #[inline]
    pub fn get_data(&self) -> T {
        // SAFETY: `data` is a live, initialized `T` (every construction path
        // writes one). Reading it through `&self` races only with a concurrent
        // `set_data`, whose own contract forbids that overlap.
        unsafe { *self.data.get() }
    }

    /// Overwrite the payload. C++ spells this `entry.data_ = val`.
    ///
    /// # Safety
    /// The caller must have exclusive access to this slot's payload for the
    /// duration of the write — i.e. must have claimed it, either by owning the
    /// `tail` ticket that maps here (producer, before `set_ready`) or by winning
    /// the ready-flag CAS (consumer, in `pop`). Concurrent `set_data`/`get_data`
    /// on the same entry is a data race.
    #[inline]
    pub unsafe fn set_data(&self, val: T) {
        // SAFETY: forwarded to this function's own contract.
        unsafe { *self.data.get() = val }
    }
}

impl<T: ShmSafe> Clone for RingBufferEntry<T> {
    /// Mirrors the C++ implicit copy (`bitfield`'s copy ctor copies `bits_`;
    /// `data_` is copied).
    #[inline]
    fn clone(&self) -> Self {
        Self {
            flags: Atomic::new(self.flags.load(Ordering::SeqCst)),
            data: UnsafeCell::new(self.get_data()),
        }
    }
}

impl<T: ShmSafe + fmt::Debug> fmt::Debug for RingBufferEntry<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RingBufferEntry")
            .field("flags", &self.flags.load(Ordering::SeqCst))
            .field("data", &self.get_data())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// RingBuffer<T, FLAGS>
// ---------------------------------------------------------------------------

/// The fixed part of a [`RingBuffer`], split out so the trailing entry slice can
/// be a DST tail and so the layout is computable with [`Layout::extend`].
///
/// Field order mirrors the C++ class (minus `queue_`, which moved to the tail).
#[repr(C)]
struct RingBufferHead {
    /// C++ `head_`: consumer cursor, monotonically increasing, never wrapped by
    /// the queue length (only by `u64`).
    head: Atomic<u64>,
    /// C++ `tail_`: producer cursor / next ticket to hand out.
    tail: Atomic<u64>,
    /// C++ `assigned_worker_id_`.
    assigned_worker_id: Atomic<u32>,
    /// C++ `signal_fd_`.
    signal_fd: Atomic<i32>,
    /// C++ `tid_` (`pid_t`).
    tid: Atomic<i32>,
    /// C++ `active_`.
    active: AtomicBool,
    /// C++ `pop_lock_`; used only when `RING_BUFFER_LOCK_POP` is set.
    pop_lock: Mutex,
}

impl RingBufferHead {
    /// Matches the C++ ctor's member-initializer list: `head_(0), tail_(0),
    /// assigned_worker_id_(0), signal_fd_(-1), tid_(0), active_(true)`.
    #[inline]
    fn new() -> Self {
        Self {
            head: Atomic::new(0),
            tail: Atomic::new(0),
            assigned_worker_id: Atomic::new(0),
            signal_fd: Atomic::new(-1),
            tid: Atomic::new(0),
            active: AtomicBool::new(true),
            pop_lock: Mutex::new(),
        }
    }
}

/// Lock-free circular queue for shared memory — `ctp::ipc::ring_buffer<T,
/// AllocT, FLAGS>`.
///
/// `FLAGS` is the C++ template parameter; see the [module docs](self) for the
/// storage layout, the flag constants, and the aliases
/// ([`SpscRingBuffer`], [`MpscRingBuffer`], [`MpmcRingBuffer`], …).
///
/// The type is unsized (its entries are inline), so it is always handled behind
/// a reference: obtain one from [`RingBufferBox`] (process-local) or from
/// [`init_at`](Self::init_at) / [`from_raw`](Self::from_raw) (shared memory).
/// It has no `Drop`, contains no pointers, and is correct at any mapping base.
///
/// # Example
///
/// ```
/// use ctp_ds::ring_buffer::{MpscRingBuffer, RingBufferBox};
///
/// let q: RingBufferBox<u64, { ctp_ds::ring_buffer::MPSC_FLAGS }> = RingBufferBox::new(8);
/// assert!(q.push(42));
/// assert_eq!(q.pop(), Some(42));
/// assert_eq!(q.pop(), None);
///
/// // ... or via the C++-named alias, which needs an explicit annotation:
/// fn takes(_: &MpscRingBuffer<u64>) {}
/// takes(&q);
/// ```
#[repr(C)]
pub struct RingBuffer<T, const FLAGS: u32 = DEFAULT_RING_BUFFER_FLAGS> {
    hdr: RingBufferHead,
    /// C++ `queue_`. Length is `depth + 1`: one slot is reserved as the
    /// full/empty sentinel.
    queue: [RingBufferEntry<T>],
}

// SAFETY: every field is either an atomic or a `RingBufferEntry<T>`, which is
// `Send + Sync` for `T: ShmSafe` (see its impl). The queue's own safe API never
// writes an entry's payload without first claiming the slot (producer: an
// exclusive `tail` ticket; consumer: winning the ready-flag CAS), so sharing
// `&RingBuffer` across threads — and across processes, which is the entire
// point — is sound for the configurations with a space policy. See divergence
// 11 for the two configurations whose *C++ contract* permits an overlap.
unsafe impl<T: ShmSafe, const FLAGS: u32> Send for RingBuffer<T, FLAGS> {}
unsafe impl<T: ShmSafe, const FLAGS: u32> Sync for RingBuffer<T, FLAGS> {}

impl<T: ShmSafe + Default, const FLAGS: u32> RingBuffer<T, FLAGS> {
    // -- placement / construction -------------------------------------------

    /// Exact layout (size **and** alignment) of a `RingBuffer` holding `depth`
    /// usable entries.
    ///
    /// `Layout::extend` reproduces the `repr(C)` field-placement rule, so this
    /// agrees with the compiler byte for byte (pinned by
    /// `layout_matches_the_compiler`).
    ///
    /// # Panics
    /// If the block would exceed `isize::MAX` bytes (the C++ silently wraps).
    pub fn layout(depth: usize) -> Layout {
        let slots = depth
            .checked_add(1)
            .expect("ring_buffer depth + 1 overflows usize");
        let entries = Layout::array::<RingBufferEntry<T>>(slots)
            .expect("ring_buffer entry array exceeds isize::MAX bytes");
        let (layout, _off) = Layout::new::<RingBufferHead>()
            .extend(entries)
            .expect("ring_buffer layout exceeds isize::MAX bytes");
        layout.pad_to_align()
    }

    /// C++ `CalculateSize(depth)` — bytes needed for a `RingBuffer` of `depth`
    /// usable entries. See divergence 4.
    ///
    /// # Panics
    /// As [`layout`](Self::layout).
    #[inline]
    pub fn calculate_size(depth: usize) -> usize {
        Self::layout(depth).size()
    }

    /// Build the (fat) pointer for a `RingBuffer` of `depth` entries at `mem`.
    /// Does not read or write anything.
    #[inline]
    fn fat_ptr(mem: *mut u8, depth: usize) -> *mut Self {
        // The cast keeps the address and reinterprets the slice length as the
        // trailing-slice length of `Self`; the compiler then places `queue` at
        // its true offset.
        ptr::slice_from_raw_parts_mut(mem.cast::<RingBufferEntry<T>>(), depth + 1) as *mut Self
    }

    /// C++ `ring_buffer(AllocT *alloc, size_t depth)` — construct in place.
    ///
    /// Initializes the header (`head_ = tail_ = 0`, `assigned_worker_id_ = 0`,
    /// `signal_fd_ = -1`, `tid_ = 0`, `active_ = true`, a fresh `pop_lock_`) and
    /// all `depth + 1` entries (`flags_ = 0`, `data_ = T::default()`).
    ///
    /// # Safety
    /// * `mem` must be valid for writes for [`layout(depth)`](Self::layout)
    ///   bytes and aligned to that layout's alignment.
    /// * The block must outlive `'a`, and nothing may alias it as anything other
    ///   than this `RingBuffer` (other processes mapping the same segment and
    ///   using it *as* this `RingBuffer` are exactly the intended use).
    pub unsafe fn init_at<'a>(mem: *mut u8, depth: usize) -> &'a Self {
        let this = Self::fat_ptr(mem, depth);
        // SAFETY: `mem` is valid for the whole layout per the contract, so the
        // header slot and every entry slot are writable and correctly aligned.
        // `write` initializes without dropping the (uninitialized) old value.
        unsafe {
            ptr::addr_of_mut!((*this).hdr).write(RingBufferHead::new());
            let entries = ptr::addr_of_mut!((*this).queue).cast::<RingBufferEntry<T>>();
            for i in 0..depth + 1 {
                entries.add(i).write(RingBufferEntry::new());
            }
            &*this
        }
    }

    /// C++ `ring_buffer(const ring_buffer &other)` — copy-construct in place.
    ///
    /// The copy gets `other`'s depth, cursors, worker/fd/tid/active fields and a
    /// full copy of every entry (flags included). Like the C++ — whose
    /// `ctp::Mutex` copy ctor default-constructs — the copy gets a **fresh,
    /// unlocked** `pop_lock_`.
    ///
    /// # Safety
    /// As [`init_at`](Self::init_at), with `depth == other.capacity()`. `other`
    /// must not be concurrently mutated (the C++ copy ctor is equally racy).
    pub unsafe fn init_copy_at<'a>(mem: *mut u8, other: &Self) -> &'a Self {
        let depth = other.capacity();
        // SAFETY: forwarded to this function's contract.
        let this = unsafe { Self::init_at(mem, depth) };
        this.hdr
            .head
            .store(other.hdr.head.load(Ordering::SeqCst), Ordering::SeqCst);
        this.hdr
            .tail
            .store(other.hdr.tail.load(Ordering::SeqCst), Ordering::SeqCst);
        this.set_assigned_worker_id(other.assigned_worker_id());
        this.set_signal_fd(other.signal_fd());
        this.set_tid(other.tid());
        this.set_active(other.is_active());
        for (dst, src) in this.queue.iter().zip(other.queue.iter()) {
            // SAFETY: `this` is not published yet, so no one else can touch its
            // slots; the entry copy mirrors `queue_[i] = other.queue_[i]`.
            unsafe { dst.set_data(src.get_data()) };
            dst.flags
                .store(src.flags.load(Ordering::SeqCst), Ordering::SeqCst);
        }
        this
    }

    /// Attach to a `RingBuffer` that already lives at `mem` — the counterpart of
    /// resolving a `ShmPtr` into a mapped segment.
    ///
    /// # Safety
    /// * `mem` must point at a `RingBuffer<T, FLAGS>` previously built by
    ///   [`init_at`](Self::init_at) with this same `depth` and `T`/`FLAGS`,
    ///   valid for reads and writes for `'a`, and aligned.
    /// * `depth` must match the one it was created with; a mismatch silently
    ///   reinterprets memory.
    #[inline]
    pub unsafe fn from_raw<'a>(mem: *const u8, depth: usize) -> &'a Self {
        // SAFETY: forwarded to this function's contract.
        unsafe { &*Self::fat_ptr(mem as *mut u8, depth) }
    }

    // -- lane metadata ------------------------------------------------------

    /// C++ `GetAssignedWorkerId()`.
    #[inline]
    pub fn assigned_worker_id(&self) -> u32 {
        self.hdr.assigned_worker_id.load(Ordering::SeqCst)
    }

    /// C++ `SetAssignedWorkerId(u32)`.
    #[inline]
    pub fn set_assigned_worker_id(&self, worker_id: u32) {
        self.hdr
            .assigned_worker_id
            .store(worker_id, Ordering::SeqCst);
    }

    /// C++ `GetSignalFd()`. `-1` when unset.
    #[inline]
    pub fn signal_fd(&self) -> i32 {
        self.hdr.signal_fd.load(Ordering::SeqCst)
    }

    /// C++ `SetSignalFd(int)`.
    #[inline]
    pub fn set_signal_fd(&self, signal_fd: i32) {
        self.hdr.signal_fd.store(signal_fd, Ordering::SeqCst);
    }

    /// C++ `GetTid()` (`pid_t` → `i32`, divergence 3).
    #[inline]
    pub fn tid(&self) -> i32 {
        self.hdr.tid.load(Ordering::SeqCst)
    }

    /// C++ `SetTid(pid_t)`.
    #[inline]
    pub fn set_tid(&self, tid: i32) {
        self.hdr.tid.store(tid, Ordering::SeqCst);
    }

    /// C++ `IsActive()` — worker is accepting tasks rather than blocked in
    /// `epoll_wait`.
    #[inline]
    pub fn is_active(&self) -> bool {
        self.hdr.active.load(Ordering::SeqCst)
    }

    /// C++ `SetActive(bool)`.
    #[inline]
    pub fn set_active(&self, active: bool) {
        self.hdr.active.store(active, Ordering::SeqCst);
    }

    // -- observers ----------------------------------------------------------

    /// Slot count, `depth + 1` (C++ `queue_.size()`).
    #[inline]
    fn slots(&self) -> u64 {
        self.queue.len() as u64
    }

    /// The entry `cursor` maps to. `cursor` is a monotonic head/tail value.
    #[inline]
    fn slot(&self, cursor: u64) -> &RingBufferEntry<T> {
        &self.queue[(cursor % self.slots()) as usize]
    }

    /// C++ `Size()` — number of items currently queued.
    #[inline]
    pub fn size(&self) -> usize {
        let head = self.hdr.head.load(Ordering::SeqCst);
        let tail = self.hdr.tail.load(Ordering::SeqCst);
        if tail >= head {
            (tail - head) as usize
        } else {
            0
        }
    }

    /// C++ `Capacity()` — usable entries, i.e. the constructor's `depth`.
    #[inline]
    pub fn capacity(&self) -> usize {
        let slots = self.queue.len();
        if slots > 0 {
            slots - 1
        } else {
            0
        }
    }

    /// C++ `GetDepth()` — allocated **slot count**, i.e. `capacity() + 1`.
    ///
    /// Note the C++ naming trap, kept for parity: this does *not* return the
    /// `depth` passed to the constructor — that is [`capacity`](Self::capacity).
    #[inline]
    pub fn get_depth(&self) -> usize {
        self.queue.len()
    }

    /// C++ `Empty()`.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.hdr.head.load(Ordering::SeqCst) == self.hdr.tail.load(Ordering::SeqCst)
    }

    /// C++ `Full()`.
    #[inline]
    pub fn is_full(&self) -> bool {
        let head = self.hdr.head.load(Ordering::SeqCst);
        let tail = self.hdr.tail.load(Ordering::SeqCst);
        // C++ computes `tail - head` on unsigned counters, which wraps; plain
        // `-` would panic in a Rust debug build.
        tail.wrapping_sub(head) == self.slots() - 1
    }

    /// C++ `GetHead()` — oldest valid event id.
    #[inline]
    pub fn get_head(&self) -> u64 {
        self.hdr.head.load(Ordering::SeqCst)
    }

    /// C++ `GetTail()` — next event id to be written.
    #[inline]
    pub fn get_tail(&self) -> u64 {
        self.hdr.tail.load(Ordering::SeqCst)
    }

    /// C++ `GetHeadDevice()`. Host-equal to [`get_head`](Self::get_head).
    #[inline]
    pub fn get_head_device(&self) -> u64 {
        self.hdr.head.load_device()
    }

    /// C++ `GetTailDevice()`. Host-equal to [`get_tail`](Self::get_tail).
    #[inline]
    pub fn get_tail_device(&self) -> u64 {
        self.hdr.tail.load_device()
    }

    // -- producer -----------------------------------------------------------

    /// C++ `Push(const T&)` — enqueue.
    ///
    /// Returns `false` only when the buffer is full *and* `FLAGS` says to fail
    /// (`RING_BUFFER_ERROR_ON_NO_SPACE` / `RING_BUFFER_DYNAMIC_SIZE`). With
    /// `RING_BUFFER_WAIT_FOR_SPACE` it spins until a consumer frees the claimed
    /// slot; with neither, it overwrites (see divergence 11).
    #[inline]
    pub fn push(&self, val: T) -> bool {
        self.emplace(val)
    }

    /// C++ `TryPush(const T&)` — alias for [`push`](Self::push), as in the C++
    /// (it does *not* mean "never block": with `RING_BUFFER_WAIT_FOR_SPACE` this
    /// still waits).
    #[inline]
    pub fn try_push(&self, val: T) -> bool {
        self.push(val)
    }

    /// C++ `Emplace(Args&&...)` — see divergence 5; identical to
    /// [`push`](Self::push).
    pub fn emplace(&self, val: T) -> bool {
        // Load head, then claim a slot by taking a ticket. The order matters:
        // `head` is sampled *before* the ticket, so the space check is
        // conservative (it may see a stale, smaller head, never a too-large one).
        let head = self.hdr.head.load(Ordering::SeqCst);
        let tail = self.hdr.tail.fetch_add_system(1);
        let slots = self.slots();

        // One slot is always kept empty as the sentinel, hence `>=`.
        if waits_for_space(FLAGS) {
            let mut head = head;
            while tail.wrapping_sub(head).wrapping_add(1) >= slots {
                head = self.hdr.head.load_device();
                std::hint::spin_loop();
            }
        } else if errors_on_no_space(FLAGS) {
            if tail.wrapping_sub(head).wrapping_add(1) >= slots {
                // Retract the ticket. Correct for one producer only —
                // divergence 12.
                self.hdr.tail.fetch_sub(1, Ordering::SeqCst);
                return false;
            }
        } else if is_dynamic_size(FLAGS) && tail.wrapping_sub(head).wrapping_add(1) >= slots {
            // C++: "Would need to resize vector - not implemented for now".
            // NOTE: the ticket is *not* retracted here — divergence 10.
            return false;
        }
        // Otherwise (e.g. `circular_mpsc_ring_buffer`): no space check at all,
        // so a lagging consumer's slot is overwritten. Divergence 11.

        let entry = self.slot(tail);
        // SAFETY: `tail` is this producer's exclusive ticket, so no other
        // producer maps to this slot until the cursor laps the ring; the
        // consumer will not touch it until `set_ready` publishes it. (In the
        // no-space-policy configuration a lap *can* happen concurrently — the
        // documented C++ contract, divergence 11.)
        unsafe { entry.set_data(val) };
        // Publish: fence, then the release-flavoured flag write.
        threadfence();
        entry.set_ready();
        true
    }

    /// C++ `PushSystem(const T&)` — the GPU→CPU-visible push.
    ///
    /// Host-equal to [`push`](Self::push) except that the C++ omits the
    /// `DYNAMIC_SIZE` branch here (so a `DYNAMIC_SIZE` queue silently overwrites
    /// via this entry point) and skips the explicit `threadfence` before
    /// publishing. Divergence 8.
    pub fn push_system(&self, val: T) -> bool {
        let head = self.hdr.head.load_system();
        let tail = self.hdr.tail.fetch_add_system(1);
        let slots = self.slots();

        if waits_for_space(FLAGS) {
            let mut head = head;
            while tail.wrapping_sub(head).wrapping_add(1) >= slots {
                head = self.hdr.head.load_system();
                std::hint::spin_loop();
            }
        } else if errors_on_no_space(FLAGS) && tail.wrapping_sub(head).wrapping_add(1) >= slots {
            // C++ `tail_.fetch_add_system((u64)-1)`: the wrapping way to
            // subtract one.
            self.hdr.tail.fetch_add_system(u64::MAX);
            return false;
        }

        let entry = self.slot(tail);
        // SAFETY: as in `emplace` — `tail` is this producer's exclusive ticket.
        unsafe { entry.set_data(val) };
        entry.set_ready_system();
        true
    }

    // -- consumer -----------------------------------------------------------

    /// C++ `Pop(T&)` — dequeue the oldest ready entry.
    ///
    /// Returns `None` when the queue is empty **or** when the head slot has been
    /// claimed by a producer that has not published it yet, **or** when another
    /// consumer won the claim. Callers therefore retry rather than treating
    /// `None` as "empty"; [`is_empty`](Self::is_empty) is the emptiness test.
    ///
    /// # Only one consumer, unless `RING_BUFFER_LOCK_POP` is set
    ///
    /// With `RING_BUFFER_LOCK_POP` the whole body runs under the embedded mutex,
    /// which is what makes an MPSC queue a correct MPMC one — use
    /// [`MpmcRingBuffer`]. **Without it, calling `pop` from more than one thread
    /// deadlocks the queue permanently** (a stale `head` plus a lapped ring lets
    /// a consumer claim a future entry and rewind `head`). The C++ comment
    /// claims the lock is only a contention optimization; it is not. See
    /// divergence 14 for the mechanism and the reproduction.
    #[inline]
    pub fn pop(&self) -> Option<T> {
        if locks_pop(FLAGS) {
            let _guard = self.hdr.pop_lock.scoped_lock(0);
            self.pop_unlocked()
        } else {
            self.pop_unlocked()
        }
    }

    /// C++ `TryPop(T&)` — alias for [`pop`](Self::pop).
    #[inline]
    pub fn try_pop(&self) -> Option<T> {
        self.pop()
    }

    /// C++ `PopUnlocked(T&)` — the lock-free body of [`pop`](Self::pop).
    fn pop_unlocked(&self) -> Option<T> {
        let head = self.hdr.head.load_system();
        let tail = self.hdr.tail.load_system();
        if head >= tail {
            return None;
        }

        let entry = self.slot(head);
        if !entry.is_ready_system() {
            // Claimed by a producer that has not published yet.
            return None;
        }
        // Claim the entry: exactly one consumer can win this.
        let mut expected = READY_BIT;
        if !entry
            .flags
            .compare_exchange_strong_system(&mut expected, 0, Ordering::SeqCst)
        {
            return None;
        }
        let val = entry.get_data();
        // Release the slot's copy now that we own it (the producer cannot
        // re-emplace here until `head` advances a full lap). Leaving the copy in
        // place would pin whatever the element owns — one leaked task per
        // enqueue on an idle ring; see the C++ comment and issue #620.
        // SAFETY: winning the CAS above makes this consumer the slot's exclusive
        // owner until `head` is advanced below.
        unsafe { entry.set_data(T::default()) };
        self.hdr.head.store_system(head.wrapping_add(1));
        Some(val)
    }

    /// C++ `PopDevice(T&)` — the GPU→GPU pop.
    ///
    /// Host-equal to [`pop`](Self::pop) apart from two *real* differences the
    /// C++ has and this port keeps (divergence 12): it never takes `pop_lock`
    /// (even under `RING_BUFFER_LOCK_POP`), and it does not release the slot's
    /// copy of the element.
    pub fn pop_device(&self) -> Option<T> {
        let head = self.hdr.head.load_device();
        let tail = self.hdr.tail.load_device();
        if head >= tail {
            return None;
        }
        let entry = self.slot(head);
        if !entry.is_ready_device() {
            return None;
        }
        let mut expected = READY_BIT;
        if !entry
            .flags
            .compare_exchange_strong(&mut expected, 0, Ordering::SeqCst)
        {
            return None;
        }
        threadfence();
        let val = entry.get_data();
        self.hdr
            .head
            .store(head.wrapping_add(1), Ordering::SeqCst);
        Some(val)
    }

    /// C++ `Peek(u64 idx, T &val)` — read by absolute (monotonic) event id
    /// without advancing `head`.
    ///
    /// `None` if `idx` is outside `[head, tail)` or the entry is not ready.
    ///
    /// Only safe against a *quiescent* consumer: a concurrent [`pop`](Self::pop)
    /// of the same slot races with this read (divergence 11).
    pub fn peek(&self, idx: u64) -> Option<T> {
        let head = self.hdr.head.load(Ordering::SeqCst);
        let tail = self.hdr.tail.load(Ordering::SeqCst);
        if idx < head || idx >= tail {
            return None;
        }
        let entry = self.slot(idx);
        if !entry.is_ready() {
            return None;
        }
        Some(entry.get_data())
    }

    // -- lifecycle ----------------------------------------------------------

    /// C++ `Clear()` — reset the cursors and clear every entry's flags.
    ///
    /// Payloads are left as they are (as in C++); an entry only becomes readable
    /// again once a producer republishes it. Not thread-safe against concurrent
    /// producers or consumers.
    pub fn clear(&self) {
        self.hdr.head.store(0, Ordering::SeqCst);
        self.hdr.tail.store(0, Ordering::SeqCst);
        for entry in self.queue.iter() {
            entry.clear_flags();
        }
    }

    /// C++ `Reset()` — alias for [`clear`](Self::clear).
    #[inline]
    pub fn reset(&self) {
        self.clear();
    }
}

impl<T: ShmSafe + Default + fmt::Debug, const FLAGS: u32> fmt::Debug for RingBuffer<T, FLAGS> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RingBuffer")
            .field("flags", &format_args!("{FLAGS:#04x}"))
            .field("head", &self.get_head())
            .field("tail", &self.get_tail())
            .field("size", &self.size())
            .field("capacity", &self.capacity())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Aliases — the C++ `using` declarations
// ---------------------------------------------------------------------------

/// Flag set of [`ExtRingBuffer`], spelled out for use as a const-generic arg.
pub const EXT_FLAGS: u32 = RING_BUFFER_SPSC_FLAGS | RING_BUFFER_DYNAMIC_SIZE;
/// Flag set of [`SpscRingBuffer`].
pub const SPSC_FLAGS: u32 =
    RING_BUFFER_SPSC_FLAGS | RING_BUFFER_FIXED_SIZE | RING_BUFFER_ERROR_ON_NO_SPACE;
/// Flag set of [`MpscRingBuffer`].
pub const MPSC_FLAGS: u32 =
    RING_BUFFER_MPSC_FLAGS | RING_BUFFER_FIXED_SIZE | RING_BUFFER_WAIT_FOR_SPACE;
/// Flag set of [`CircularMpscRingBuffer`].
pub const CIRCULAR_MPSC_FLAGS: u32 = RING_BUFFER_MPSC_FLAGS | RING_BUFFER_FIXED_SIZE;
/// Flag set of [`MpmcRingBuffer`].
pub const MPMC_FLAGS: u32 = RING_BUFFER_MPSC_FLAGS
    | RING_BUFFER_FIXED_SIZE
    | RING_BUFFER_WAIT_FOR_SPACE
    | RING_BUFFER_LOCK_POP;

/// C++ `ext_ring_buffer<T>` — "extensible", single-thread only.
///
/// **Read divergence 10 first**: the C++ resize path is an unimplemented stub
/// that leaks a `tail` ticket, so a full `ext_ring_buffer` wedges permanently.
pub type ExtRingBuffer<T> = RingBuffer<T, EXT_FLAGS>;

/// C++ `spsc_ring_buffer<T>` — fixed size, one producer, one consumer, push
/// fails when full.
pub type SpscRingBuffer<T> = RingBuffer<T, SPSC_FLAGS>;

/// C++ `mpsc_ring_buffer<T>` — fixed size, many producers, one consumer, push
/// waits for space.
///
/// "One consumer" is load-bearing: popping this from two threads deadlocks the
/// queue (divergence 14). Use [`MpmcRingBuffer`] for multiple consumers.
pub type MpscRingBuffer<T> = RingBuffer<T, MPSC_FLAGS>;

/// C++ `circular_mpsc_ring_buffer<T>` — like [`MpscRingBuffer`] but **wraps
/// around and overwrites** instead of waiting. See divergence 11.
pub type CircularMpscRingBuffer<T> = RingBuffer<T, CIRCULAR_MPSC_FLAGS>;

/// C++ `mpmc_ring_buffer<T>` — [`MpscRingBuffer`] plus a `Pop` serialized by the
/// embedded mutex, so many consumers can pop safely and without CAS contention.
pub type MpmcRingBuffer<T> = RingBuffer<T, MPMC_FLAGS>;

// ---------------------------------------------------------------------------
// RingBufferBox — process-local owner (no C++ counterpart)
// ---------------------------------------------------------------------------

/// An owning, process-local handle to a [`RingBuffer`] — **not** part of the C++
/// API.
///
/// [`RingBuffer`] is unsized and lives wherever it was placed; in C++ that is
/// always an allocator's segment. This box is the equivalent for a queue that
/// does not need to be shared between processes (and for tests): it allocates
/// with the global allocator, constructs in place, and frees on drop.
/// `Deref`s to the queue, so all queue methods are available.
///
/// For a shared-memory queue, allocate through the owning allocator instead and
/// use [`RingBuffer::init_at`] / [`RingBuffer::from_raw`].
pub struct RingBufferBox<T: ShmSafe + Default, const FLAGS: u32 = DEFAULT_RING_BUFFER_FLAGS> {
    ptr: NonNull<RingBuffer<T, FLAGS>>,
    layout: Layout,
}

// SAFETY: the box is a unique owner of a `RingBuffer<T, FLAGS>`, which is itself
// `Send + Sync` (see its impls). `NonNull` only suppresses the auto-derivation.
unsafe impl<T: ShmSafe + Default, const FLAGS: u32> Send for RingBufferBox<T, FLAGS> {}
unsafe impl<T: ShmSafe + Default, const FLAGS: u32> Sync for RingBufferBox<T, FLAGS> {}

impl<T: ShmSafe + Default, const FLAGS: u32> RingBufferBox<T, FLAGS> {
    /// Allocate and construct a queue with `depth` usable entries — the
    /// equivalent of C++ `ring_buffer(alloc, depth)`.
    ///
    /// # Panics
    /// If [`RingBuffer::layout`] does, or on allocation failure.
    pub fn new(depth: usize) -> Self {
        let layout = RingBuffer::<T, FLAGS>::layout(depth);
        // SAFETY: `layout` has non-zero size — it always contains at least the
        // header and one entry.
        let mem = unsafe { alloc(layout) };
        if mem.is_null() {
            handle_alloc_error(layout);
        }
        // SAFETY: `mem` is a fresh allocation of exactly `layout`, so it is
        // valid, aligned, unaliased, and lives until `drop` frees it.
        let this = unsafe { RingBuffer::<T, FLAGS>::init_at(mem, depth) };
        Self {
            ptr: NonNull::from(this),
            layout,
        }
    }

    /// Bytes owned by this box (`== RingBuffer::calculate_size(capacity())`).
    #[inline]
    pub fn byte_size(&self) -> usize {
        self.layout.size()
    }
}

impl<T: ShmSafe + Default, const FLAGS: u32> Default for RingBufferBox<T, FLAGS> {
    /// C++ default `depth` argument: 1024.
    #[inline]
    fn default() -> Self {
        Self::new(DEFAULT_DEPTH)
    }
}

impl<T: ShmSafe + Default, const FLAGS: u32> Deref for RingBufferBox<T, FLAGS> {
    type Target = RingBuffer<T, FLAGS>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        // SAFETY: `ptr` was constructed in `new` from a live allocation this box
        // owns and frees only in `drop`.
        unsafe { self.ptr.as_ref() }
    }
}

impl<T: ShmSafe + Default, const FLAGS: u32> Drop for RingBufferBox<T, FLAGS> {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from `alloc(self.layout)` in `new` and is freed
        // exactly once, here. `RingBuffer` has no `Drop` glue (all fields are
        // atomics or `ShmSafe` PODs), so there is nothing to drop in place.
        unsafe { dealloc(self.ptr.cast::<u8>().as_ptr(), self.layout) }
    }
}

impl<T: ShmSafe + Default + fmt::Debug, const FLAGS: u32> fmt::Debug for RingBufferBox<T, FLAGS> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::sync::atomic::AtomicUsize;
    use std::thread;

    /// A non-trivial POD payload: catches torn/misattributed entries that a
    /// `u64` counter would hide.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    #[repr(C)]
    struct Item {
        producer: u32,
        seq: u32,
    }

    // SAFETY: `Item` is a `Copy`, `'static`, pointer-free POD of two `u32`s;
    // every bit pattern is a valid value.
    unsafe impl ShmSafe for Item {}

    impl Item {
        fn new(producer: u32, seq: u32) -> Self {
            Self { producer, seq }
        }
    }

    type Spsc<T> = RingBufferBox<T, SPSC_FLAGS>;
    type Mpsc<T> = RingBufferBox<T, MPSC_FLAGS>;
    type Mpmc<T> = RingBufferBox<T, MPMC_FLAGS>;
    type Circular<T> = RingBufferBox<T, CIRCULAR_MPSC_FLAGS>;
    type Ext<T> = RingBufferBox<T, EXT_FLAGS>;

    // -----------------------------------------------------------------------
    // Flags
    // -----------------------------------------------------------------------

    #[test]
    fn flag_values_match_cpp() {
        assert_eq!(RING_BUFFER_SPSC_FLAGS, 0x01);
        assert_eq!(RING_BUFFER_MPSC_FLAGS, 0x02);
        assert_eq!(RING_BUFFER_WAIT_FOR_SPACE, 0x04);
        assert_eq!(RING_BUFFER_ERROR_ON_NO_SPACE, 0x08);
        assert_eq!(RING_BUFFER_DYNAMIC_SIZE, 0x10);
        assert_eq!(RING_BUFFER_FIXED_SIZE, 0x20);
        assert_eq!(RING_BUFFER_LOCK_POP, 0x40);
        assert_eq!(DEFAULT_RING_BUFFER_FLAGS, 0x01 | 0x20 | 0x08);
    }

    #[test]
    fn flag_predicates_decode_each_alias() {
        // spsc_ring_buffer
        assert!(is_spsc(SPSC_FLAGS) && !is_mpsc(SPSC_FLAGS));
        assert!(errors_on_no_space(SPSC_FLAGS) && !waits_for_space(SPSC_FLAGS));
        assert!(!is_atomic(SPSC_FLAGS) && !locks_pop(SPSC_FLAGS));
        // mpsc_ring_buffer
        assert!(is_mpsc(MPSC_FLAGS) && is_atomic(MPSC_FLAGS));
        assert!(waits_for_space(MPSC_FLAGS) && !errors_on_no_space(MPSC_FLAGS));
        assert!(!locks_pop(MPSC_FLAGS));
        // circular_mpsc_ring_buffer: no space policy whatsoever
        assert!(is_mpsc(CIRCULAR_MPSC_FLAGS));
        assert!(!waits_for_space(CIRCULAR_MPSC_FLAGS));
        assert!(!errors_on_no_space(CIRCULAR_MPSC_FLAGS));
        assert!(!is_dynamic_size(CIRCULAR_MPSC_FLAGS));
        // mpmc_ring_buffer
        assert!(locks_pop(MPMC_FLAGS) && waits_for_space(MPMC_FLAGS));
        // ext_ring_buffer
        assert!(is_dynamic_size(EXT_FLAGS) && is_spsc(EXT_FLAGS));
        assert!(!errors_on_no_space(EXT_FLAGS));
    }

    // -----------------------------------------------------------------------
    // ABI / layout / shared-memory shape
    // -----------------------------------------------------------------------

    /// `RingBufferEntry<T>` is the one type this port keeps byte-compatible
    /// with the C++ (`abitfield32_t flags_; T data_;`).
    #[test]
    fn entry_layout_matches_cpp() {
        use std::mem::{align_of, offset_of, size_of};
        assert_eq!(size_of::<RingBufferEntry<u32>>(), 8);
        assert_eq!(size_of::<RingBufferEntry<u64>>(), 16);
        assert_eq!(align_of::<RingBufferEntry<u64>>(), 8);
        assert_eq!(offset_of!(RingBufferEntry<u64>, flags), 0);
        assert_eq!(offset_of!(RingBufferEntry<u64>, data), 8);
        // The flag word is exactly the C++ bitfield's single `bits_` member.
        assert_eq!(size_of::<Atomic<u32>>(), 4);
        // `UnsafeCell` must not perturb the payload.
        assert_eq!(size_of::<RingBufferEntry<Item>>(), 4 + 8);
        assert_eq!(offset_of!(RingBufferEntry<Item>, data), 4);
    }

    /// Our `Layout` arithmetic must agree with the compiler's real `repr(C)`
    /// placement, since `init_at`/`from_raw` size the block with it.
    #[test]
    fn layout_matches_the_compiler() {
        fn check<T: ShmSafe + Default, const FLAGS: u32>(depth: usize) {
            let b = RingBufferBox::<T, FLAGS>::new(depth);
            let computed = RingBuffer::<T, FLAGS>::layout(depth);
            let actual = std::mem::size_of_val(&*b);
            assert_eq!(computed.size(), actual, "size mismatch at depth {depth}");
            assert_eq!(computed.size(), b.byte_size());
            assert_eq!(
                computed.align(),
                std::mem::align_of_val(&*b),
                "align mismatch at depth {depth}"
            );
            // The entry slice really does start where `Layout::extend` says.
            let (_, off) = Layout::new::<RingBufferHead>()
                .extend(Layout::array::<RingBufferEntry<T>>(depth + 1).unwrap())
                .unwrap();
            let base = &*b as *const RingBuffer<T, FLAGS> as *const u8;
            let entries = ptr::addr_of!(b.queue) as *const u8;
            assert_eq!(entries as usize - base as usize, off);
            assert_eq!(b.queue.len(), depth + 1);
        }
        for depth in [0usize, 1, 2, 7, 8, 1023] {
            check::<u64, SPSC_FLAGS>(depth);
            check::<u32, MPSC_FLAGS>(depth);
            check::<Item, MPMC_FLAGS>(depth);
        }
    }

    #[test]
    fn calculate_size_accounts_for_every_slot() {
        let entry = std::mem::size_of::<RingBufferEntry<u64>>();
        let a = RingBuffer::<u64, SPSC_FLAGS>::calculate_size(4);
        let b = RingBuffer::<u64, SPSC_FLAGS>::calculate_size(5);
        // One extra usable entry costs exactly one entry.
        assert_eq!(b - a, entry);
        // A depth-0 queue still has its sentinel slot.
        let z = RingBuffer::<u64, SPSC_FLAGS>::calculate_size(0);
        assert_eq!(a - z, 4 * entry);
        assert!(z >= std::mem::size_of::<RingBufferHead>() + entry);
    }

    #[test]
    #[should_panic(expected = "overflow")]
    fn layout_panics_instead_of_wrapping() {
        // C++ `CalculateSize` would silently wrap here.
        let _ = RingBuffer::<u64, SPSC_FLAGS>::layout(usize::MAX);
    }

    /// MEMORY_DESIGN.md: nothing in a segment may have `Drop` glue.
    #[test]
    fn no_drop_glue_anywhere() {
        use std::mem::needs_drop;
        // `RingBuffer` itself is unsized, so check every field's type: if none
        // has drop glue, neither can the struct.
        assert!(!needs_drop::<RingBufferHead>());
        assert!(!needs_drop::<RingBufferEntry<u64>>());
        assert!(!needs_drop::<RingBufferEntry<Item>>());
        assert!(!needs_drop::<Atomic<u64>>());
        assert!(!needs_drop::<Mutex>());
    }

    #[test]
    fn queue_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync + ?Sized>() {}
        assert_send_sync::<RingBuffer<Item, MPMC_FLAGS>>();
        assert_send_sync::<RingBufferEntry<Item>>();
        assert_send_sync::<RingBufferBox<Item, MPMC_FLAGS>>();
    }

    /// MEMORY_DESIGN.md pillar 3: a segment mapped at *any* base must work.
    /// Copying the raw bytes to a different address and continuing to use the
    /// copy proves no absolute address is stored anywhere inside.
    #[test]
    fn position_independent_after_byte_copy() {
        let depth = 4;
        let original = Spsc::<Item>::new(depth);
        original.set_assigned_worker_id(9);
        original.set_signal_fd(33);
        original.set_tid(4242);
        original.set_active(false);
        for i in 0..4u32 {
            assert!(original.push(Item::new(1, i)));
        }
        // Consume one so head != 0 and the copy has a non-trivial cursor state.
        assert_eq!(original.pop(), Some(Item::new(1, 0)));

        let layout = RingBuffer::<Item, SPSC_FLAGS>::layout(depth);
        // SAFETY: a fresh block of exactly the queue's layout; we memcpy the
        // (quiesced, single-threaded) source bytes into it and then reinterpret
        // it as the same type it was built as.
        unsafe {
            let mem = alloc(layout);
            assert!(!mem.is_null());
            let src = &*original as *const RingBuffer<Item, SPSC_FLAGS> as *const u8;
            ptr::copy_nonoverlapping(src, mem, layout.size());
            assert_ne!(src as usize, mem as usize, "copy must live elsewhere");

            let moved = RingBuffer::<Item, SPSC_FLAGS>::from_raw(mem, depth);
            // Metadata survived.
            assert_eq!(moved.assigned_worker_id(), 9);
            assert_eq!(moved.signal_fd(), 33);
            assert_eq!(moved.tid(), 4242);
            assert!(!moved.is_active());
            // Cursors and contents survived, at a different base address.
            assert_eq!(moved.get_head(), 1);
            assert_eq!(moved.get_tail(), 4);
            assert_eq!(moved.size(), 3);
            for i in 1..4u32 {
                assert_eq!(moved.pop(), Some(Item::new(1, i)));
            }
            assert_eq!(moved.pop(), None);
            // And it is still a working queue at its new home.
            assert!(moved.push(Item::new(7, 7)));
            assert_eq!(moved.pop(), Some(Item::new(7, 7)));
            dealloc(mem, layout);
        }
    }

    // -----------------------------------------------------------------------
    // Entry
    // -----------------------------------------------------------------------

    #[test]
    fn entry_ready_flag_protocol() {
        let e = RingBufferEntry::<u64>::new();
        assert!(!e.is_ready() && !e.is_ready_device() && !e.is_ready_system());
        assert_eq!(e.get_data(), 0);

        e.set_ready();
        assert!(e.is_ready() && e.is_ready_device() && e.is_ready_system());
        // SetReady is an OR: setting twice is idempotent, as the wrap-around
        // path in the circular queue relies on.
        e.set_ready_system();
        assert_eq!(e.flags.load(Ordering::SeqCst), 1);

        e.clear_ready();
        assert!(!e.is_ready());
        assert_eq!(e.flags.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn entry_clear_ready_preserves_other_bits() {
        let e = RingBufferEntry::<u64>::new();
        // Only bit 0 is the ready flag; the C++ `UnsetBits(1)` masks the rest.
        e.flags.store(0b1011, Ordering::SeqCst);
        e.clear_ready();
        assert_eq!(e.flags.load(Ordering::SeqCst), 0b1010);
        // `Clear()` wipes everything.
        e.clear_flags();
        assert_eq!(e.flags.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn entry_with_data_and_clone() {
        let e = RingBufferEntry::with_data(Item::new(3, 4));
        assert_eq!(e.get_data(), Item::new(3, 4));
        assert!(!e.is_ready(), "explicit ctor also starts not-ready");
        e.set_ready();
        let c = e.clone();
        assert_eq!(c.get_data(), Item::new(3, 4));
        assert!(c.is_ready(), "clone copies flags, as the C++ copy does");
        // The clone is independent.
        c.clear_ready();
        assert!(e.is_ready());
    }

    #[test]
    fn entry_set_data_round_trips() {
        let e = RingBufferEntry::<Item>::default();
        assert_eq!(e.get_data(), Item::default());
        // SAFETY: single-threaded, exclusive access to `e`.
        unsafe { e.set_data(Item::new(1, 2)) };
        assert_eq!(e.get_data(), Item::new(1, 2));
    }

    // -----------------------------------------------------------------------
    // Construction / observers
    // -----------------------------------------------------------------------

    #[test]
    fn fresh_queue_matches_cpp_ctor() {
        let q = Spsc::<u64>::new(8);
        assert_eq!(q.get_head(), 0);
        assert_eq!(q.get_tail(), 0);
        assert_eq!(q.assigned_worker_id(), 0);
        assert_eq!(q.signal_fd(), -1, "C++ ctor: signal_fd_(-1)");
        assert_eq!(q.tid(), 0);
        assert!(q.is_active(), "C++ ctor: active_(true)");
        assert!(q.is_empty());
        assert!(!q.is_full());
        assert_eq!(q.size(), 0);
        assert_eq!(q.capacity(), 8);
        assert_eq!(q.get_depth(), 9, "GetDepth() is the slot count, depth + 1");
        for e in q.queue.iter() {
            assert!(!e.is_ready());
            assert_eq!(e.get_data(), 0);
        }
    }

    #[test]
    fn default_box_uses_the_cpp_default_depth() {
        let q = RingBufferBox::<u64>::default();
        assert_eq!(q.capacity(), DEFAULT_DEPTH);
        assert_eq!(q.get_depth(), DEFAULT_DEPTH + 1);
    }

    #[test]
    fn lane_metadata_round_trips() {
        let q = Mpsc::<u64>::new(2);
        q.set_assigned_worker_id(u32::MAX);
        assert_eq!(q.assigned_worker_id(), u32::MAX);
        q.set_signal_fd(7);
        assert_eq!(q.signal_fd(), 7);
        q.set_tid(-1);
        assert_eq!(q.tid(), -1);
        q.set_active(false);
        assert!(!q.is_active());
        q.set_active(true);
        assert!(q.is_active());
    }

    #[test]
    fn size_capacity_empty_full_track_each_push() {
        let q = Spsc::<u64>::new(3);
        assert!(q.is_empty());
        for i in 0..3u64 {
            assert!(q.push(i));
            assert_eq!(q.size(), (i + 1) as usize);
            assert!(!q.is_empty());
        }
        assert!(q.is_full(), "capacity reached with the sentinel slot free");
        assert_eq!(q.size(), q.capacity());
        assert_eq!(q.pop(), Some(0));
        assert!(!q.is_full());
        assert_eq!(q.size(), 2);
    }

    // -----------------------------------------------------------------------
    // Single-threaded queue semantics
    // -----------------------------------------------------------------------

    #[test]
    fn push_pop_is_fifo() {
        let q = Spsc::<Item>::new(16);
        for i in 0..10u32 {
            assert!(q.push(Item::new(0, i)));
        }
        for i in 0..10u32 {
            assert_eq!(q.pop(), Some(Item::new(0, i)));
        }
        assert_eq!(q.pop(), None);
    }

    #[test]
    fn pop_on_empty_returns_none_and_moves_nothing() {
        let q = Spsc::<u64>::new(4);
        assert_eq!(q.pop(), None);
        assert_eq!(q.try_pop(), None);
        assert_eq!(q.pop_device(), None);
        assert_eq!(q.get_head(), 0);
        assert_eq!(q.get_tail(), 0);
    }

    /// The `ErrorOnNoSpace` path must retract its ticket, or `tail` would drift.
    #[test]
    fn spsc_push_fails_when_full_and_retracts_the_ticket() {
        let q = Spsc::<u64>::new(2);
        assert!(q.push(1));
        assert!(q.push(2));
        assert_eq!(q.get_tail(), 2);
        for _ in 0..5 {
            assert!(!q.push(3), "full: ErrorOnNoSpace");
            assert!(!q.try_push(3));
            assert_eq!(q.get_tail(), 2, "failed push must not advance tail");
            assert_eq!(q.size(), 2);
        }
        // Freeing one slot lets exactly one more in.
        assert_eq!(q.pop(), Some(1));
        assert!(q.push(3));
        assert!(!q.push(4));
        assert_eq!(q.pop(), Some(2));
        assert_eq!(q.pop(), Some(3));
        assert_eq!(q.pop(), None);
    }

    /// A `depth == 0` queue has one slot, all of it sentinel: never usable.
    #[test]
    fn depth_zero_queue_is_permanently_full() {
        let q = Spsc::<u64>::new(0);
        assert_eq!(q.capacity(), 0);
        assert_eq!(q.get_depth(), 1);
        assert!(q.is_empty());
        assert!(q.is_full(), "tail - head == 0 == slots - 1");
        assert!(!q.push(1));
        assert_eq!(q.get_tail(), 0, "ticket retracted");
        assert_eq!(q.pop(), None);
    }

    #[test]
    fn depth_one_queue_holds_exactly_one() {
        let q = Spsc::<u64>::new(1);
        assert_eq!(q.capacity(), 1);
        assert!(q.push(7));
        assert!(q.is_full());
        assert!(!q.push(8));
        assert_eq!(q.pop(), Some(7));
        assert!(q.is_empty());
        assert!(q.push(8));
        assert_eq!(q.pop(), Some(8));
    }

    /// Cursors are monotonic and only *slots* wrap; drive several full laps.
    #[test]
    fn wraparound_across_many_laps() {
        let q = Spsc::<u64>::new(3);
        for i in 0..100u64 {
            assert!(q.push(i));
            assert_eq!(q.pop(), Some(i));
            assert!(q.is_empty());
        }
        assert_eq!(q.get_head(), 100);
        assert_eq!(q.get_tail(), 100);
        // Interleaved, keeping the ring partly full across the wrap.
        for i in 100..200u64 {
            assert!(q.push(i));
            if i >= 102 {
                assert_eq!(q.pop(), Some(i - 2));
            }
        }
        assert_eq!(q.pop(), Some(198));
        assert_eq!(q.pop(), Some(199));
        assert_eq!(q.pop(), None);
    }

    /// #620: `Pop` must release the slot's copy, or an idle ring pins whatever
    /// the element owns until the slot is eventually overwritten.
    #[test]
    fn pop_releases_the_slots_copy() {
        let q = Spsc::<Item>::new(4);
        assert!(q.push(Item::new(5, 6)));
        assert_eq!(q.slot(0).get_data(), Item::new(5, 6));
        assert_eq!(q.pop(), Some(Item::new(5, 6)));
        assert_eq!(
            q.slot(0).get_data(),
            Item::default(),
            "the consumed copy must be value-initialized away"
        );
        assert!(!q.slot(0).is_ready());
    }

    /// `PopDevice` deliberately does *not* clear the slot (divergence 12).
    #[test]
    fn pop_device_leaves_the_slot_copy_cpp_quirk() {
        let q = Spsc::<Item>::new(4);
        assert!(q.push(Item::new(5, 6)));
        assert_eq!(q.pop_device(), Some(Item::new(5, 6)));
        assert_eq!(
            q.slot(0).get_data(),
            Item::new(5, 6),
            "PopDevice keeps the copy — the C++ #620 hazard, ported as-is"
        );
        assert_eq!(q.get_head(), 1);
        assert!(q.is_empty());
    }

    #[test]
    fn push_system_and_pop_device_interoperate_with_push_pop() {
        let q = Spsc::<u64>::new(4);
        assert!(q.push_system(1));
        assert!(q.push(2));
        assert_eq!(q.pop_device(), Some(1));
        assert_eq!(q.pop(), Some(2));
        assert_eq!(q.pop_device(), None);
        // push_system honours ErrorOnNoSpace, including the `(u64)-1` rollback.
        let full = Spsc::<u64>::new(1);
        assert!(full.push_system(1));
        assert!(!full.push_system(2));
        assert_eq!(full.get_tail(), 1, "the (u64)-1 fetch_add retracted it");
    }

    /// A pop cannot see a slot whose producer has not published yet, even though
    /// the ticket (and thus `tail`) is already visible.
    #[test]
    fn unpublished_slot_blocks_the_consumer() {
        let q = Mpsc::<u64>::new(4);
        // Simulate a producer that claimed a ticket and stalled before writing.
        q.hdr.tail.fetch_add(1, Ordering::SeqCst);
        assert_eq!(q.size(), 1, "Size() counts the claimed-but-unready slot");
        assert!(!q.is_empty());
        assert_eq!(q.pop(), None, "not ready: the consumer must wait");
        assert_eq!(q.get_head(), 0, "and head must not move");
        // Publishing it releases the consumer.
        let e = q.slot(0);
        // SAFETY: single-threaded; we are standing in for the stalled producer.
        unsafe { e.set_data(99) };
        e.set_ready();
        assert_eq!(q.pop(), Some(99));
    }

    #[test]
    fn peek_reads_without_consuming() {
        let q = Spsc::<Item>::new(4);
        assert_eq!(q.peek(0), None, "empty: nothing in [head, tail)");
        for i in 0..3u32 {
            assert!(q.push(Item::new(0, i)));
        }
        // Absolute (monotonic) ids, not slot indices.
        assert_eq!(q.peek(0), Some(Item::new(0, 0)));
        assert_eq!(q.peek(2), Some(Item::new(0, 2)));
        assert_eq!(q.peek(0), Some(Item::new(0, 0)), "peek does not consume");
        assert_eq!(q.size(), 3);
        assert_eq!(q.peek(3), None, "idx >= tail");
        assert_eq!(q.pop(), Some(Item::new(0, 0)));
        assert_eq!(q.peek(0), None, "idx < head");
        assert_eq!(q.peek(1), Some(Item::new(0, 1)));
    }

    #[test]
    fn clear_resets_cursors_and_flags() {
        let q = Spsc::<u64>::new(4);
        for i in 0..4 {
            assert!(q.push(i));
        }
        q.clear();
        assert_eq!(q.get_head(), 0);
        assert_eq!(q.get_tail(), 0);
        assert!(q.is_empty());
        assert_eq!(q.size(), 0);
        for e in q.queue.iter() {
            assert!(!e.is_ready(), "Clear() clears every entry's flags");
        }
        assert_eq!(q.pop(), None);
        // Usable again.
        assert!(q.push(42));
        assert_eq!(q.pop(), Some(42));
        // Reset is the same thing.
        assert!(q.push(1));
        q.reset();
        assert!(q.is_empty());
    }

    #[test]
    fn copy_construction_duplicates_state_and_contents() {
        let depth = 4;
        let src = Mpmc::<Item>::new(depth);
        src.set_assigned_worker_id(11);
        src.set_signal_fd(12);
        src.set_tid(13);
        src.set_active(false);
        for i in 0..3u32 {
            assert!(src.push(Item::new(2, i)));
        }
        assert_eq!(src.pop(), Some(Item::new(2, 0)));

        let layout = RingBuffer::<Item, MPMC_FLAGS>::layout(depth);
        // SAFETY: fresh block of the right layout; `src` is quiescent.
        unsafe {
            let mem = alloc(layout);
            assert!(!mem.is_null());
            let copy = RingBuffer::<Item, MPMC_FLAGS>::init_copy_at(mem, &src);
            assert_eq!(copy.capacity(), depth);
            assert_eq!(copy.assigned_worker_id(), 11);
            assert_eq!(copy.signal_fd(), 12);
            assert_eq!(copy.tid(), 13);
            assert!(!copy.is_active());
            assert_eq!(copy.get_head(), 1);
            assert_eq!(copy.get_tail(), 3);
            assert_eq!(copy.pop(), Some(Item::new(2, 1)));
            assert_eq!(copy.pop(), Some(Item::new(2, 2)));
            assert_eq!(copy.pop(), None);
            // Independent of the source.
            assert_eq!(src.pop(), Some(Item::new(2, 1)));
            dealloc(mem, layout);
        }
    }

    // -----------------------------------------------------------------------
    // Configuration-specific behaviour (incl. ported C++ bugs)
    // -----------------------------------------------------------------------

    /// `circular_mpsc_ring_buffer` has no space policy: a push into a full ring
    /// overwrites the oldest slot instead of failing or waiting. Single-threaded
    /// on purpose — concurrent overwrite is the documented data race
    /// (divergence 11).
    #[test]
    fn circular_queue_overwrites_when_full() {
        // depth 2 => 3 slots. With no space policy, even the sentinel slot is
        // written, so the ring only laps on the *fourth* push.
        let q = Circular::<u64>::new(2);
        assert!(q.push(1)); // cursor 0 -> slot 0
        assert!(q.push(2)); // cursor 1 -> slot 1
        assert!(q.is_full(), "tail - head == slots - 1");
        // No error and no wait: the ticket is handed out regardless of Full().
        assert!(q.push(3), "circular pushes always 'succeed'"); // cursor 2 -> slot 2
        assert_eq!(q.get_tail(), 3);
        assert_eq!(q.capacity(), 2);
        assert_eq!(
            q.size(),
            3,
            "Size() now exceeds Capacity() — the cursors, not the slots, are truth"
        );

        // Cursor 3 laps the ring (3 % 3 == 0) and clobbers slot 0 in place,
        // which still holds the undelivered entry 1.
        assert!(q.push(4));
        assert_eq!(q.slot(0).get_data(), 4, "entry 1 was overwritten in place");

        // Delivery is now wrong in every way an overwrite can make it wrong:
        // entry 1 is gone, and the entry that took its slot is delivered first,
        // out of order.
        assert_eq!(q.pop(), Some(4), "cursor 0 delivers cursor 3's payload");
        assert_eq!(q.pop(), Some(2));
        assert_eq!(q.pop(), Some(3));
        // ...and because cursor 3's slot was consumed by cursor 0's pop, head can
        // never reach tail: the ring wedges exactly like the ext_ring_buffer case.
        assert_eq!(q.get_head(), 3);
        assert_eq!(q.get_tail(), 4);
        assert!(!q.is_empty());
        for _ in 0..5 {
            assert_eq!(q.pop(), None, "wedged: head < tail but the slot is spent");
        }
    }

    /// Upstream bug (divergence 10): `Emplace`'s DynamicSize branch returns
    /// false *without* retracting the ticket, so a full `ext_ring_buffer` is
    /// wedged forever — even after the consumer drains it.
    #[test]
    fn ext_ring_buffer_wedges_forever_after_a_full_push_cpp_bug() {
        let q = Ext::<u64>::new(2);
        assert!(q.push(1));
        assert!(q.push(2));
        assert!(!q.push(3), "full: the resize path is a `return false` stub");
        assert_eq!(
            q.get_tail(),
            3,
            "BUG (ported): the DynamicSize branch never undoes its fetch_add"
        );
        assert_eq!(q.size(), 3, "phantom entry: tail counts a slot never written");
        // The queue still drains what was really written...
        assert_eq!(q.pop(), Some(1));
        assert_eq!(q.pop(), Some(2));
        // ...and then jams on the phantom slot forever: head < tail, so it is
        // not "empty", but the slot at head is not ready and never will be.
        assert!(!q.is_empty());
        for _ in 0..10 {
            assert_eq!(q.pop(), None, "wedged on the phantom slot");
        }
        // Worse: later pushes land *behind* the phantom and are unreachable.
        assert!(q.push(4));
        assert_eq!(q.pop(), None, "still wedged; entry 4 is unreachable");
    }

    /// With `RING_BUFFER_LOCK_POP`, `Pop` runs under the embedded mutex. Verify
    /// the lock is actually released (a leaked ticket would deadlock the next
    /// pop) and that popping through it is otherwise ordinary.
    #[test]
    fn lock_pop_serializes_and_releases() {
        let q = Mpmc::<u64>::new(4);
        for i in 0..4 {
            assert!(q.push(i));
        }
        for i in 0..4 {
            assert_eq!(q.pop(), Some(i));
        }
        // Empty pops take and release the lock too.
        for _ in 0..4 {
            assert_eq!(q.pop(), None);
        }
        // The lock is free: this would hang if a guard had leaked.
        q.hdr.pop_lock.lock(0);
        q.hdr.pop_lock.unlock();
    }

    // -----------------------------------------------------------------------
    // Concurrency — the whole point of the type
    // -----------------------------------------------------------------------

    /// MPSC: many producers, one consumer. Every entry must be delivered exactly
    /// once, and each producer's entries must arrive in order (the `fetch_add`
    /// ticket per producer is monotonic, and the consumer drains in ticket
    /// order). Depth is deliberately tiny so the ring wraps constantly and
    /// producers actually hit the wait-for-space path.
    #[test]
    fn mpsc_delivers_every_entry_exactly_once() {
        const PRODUCERS: u32 = 4;
        const PER_PRODUCER: u32 = 5_000;
        const TOTAL: usize = (PRODUCERS * PER_PRODUCER) as usize;

        let q = Mpsc::<Item>::new(8);
        let mut received: Vec<Item> = Vec::with_capacity(TOTAL);

        thread::scope(|s| {
            for p in 0..PRODUCERS {
                let q = &q;
                s.spawn(move || {
                    for seq in 0..PER_PRODUCER {
                        assert!(q.push(Item::new(p, seq)), "MPSC push waits, never fails");
                    }
                });
            }
            // Consumer: this thread.
            while received.len() < TOTAL {
                match q.pop() {
                    Some(item) => received.push(item),
                    None => std::hint::spin_loop(),
                }
            }
        });

        assert_eq!(received.len(), TOTAL);
        // Nothing lost, nothing duplicated.
        let unique: HashSet<Item> = received.iter().copied().collect();
        assert_eq!(unique.len(), TOTAL, "duplicate or lost entries");
        for p in 0..PRODUCERS {
            for seq in 0..PER_PRODUCER {
                assert!(unique.contains(&Item::new(p, seq)), "lost {p}/{seq}");
            }
        }
        // Per-producer FIFO order survived the interleaving.
        let mut next = vec![0u32; PRODUCERS as usize];
        for item in &received {
            assert_eq!(
                item.seq, next[item.producer as usize],
                "producer {} delivered out of order",
                item.producer
            );
            next[item.producer as usize] += 1;
        }
        assert!(q.is_empty());
        assert_eq!(q.get_head(), q.get_tail());
        assert_eq!(q.get_tail(), TOTAL as u64);
    }

    /// MPMC: many producers *and* many consumers, `Pop` serialized by the
    /// embedded mutex. Again: exactly-once delivery.
    #[test]
    fn mpmc_delivers_every_entry_exactly_once() {
        const PRODUCERS: u32 = 4;
        const CONSUMERS: usize = 3;
        const PER_PRODUCER: u32 = 2_500;
        const TOTAL: usize = (PRODUCERS * PER_PRODUCER) as usize;

        let q = Mpmc::<Item>::new(16);
        let popped = AtomicUsize::new(0);
        let mut all: Vec<Item> = Vec::with_capacity(TOTAL);

        thread::scope(|s| {
            for p in 0..PRODUCERS {
                let q = &q;
                s.spawn(move || {
                    for seq in 0..PER_PRODUCER {
                        assert!(q.push(Item::new(p, seq)));
                    }
                });
            }
            let handles: Vec<_> = (0..CONSUMERS)
                .map(|_| {
                    let q = &q;
                    let popped = &popped;
                    s.spawn(move || {
                        let mut mine = Vec::new();
                        // Stop once the *queue* has yielded everything; a
                        // consumer that never wins a race still terminates.
                        while popped.load(Ordering::SeqCst) < TOTAL {
                            match q.pop() {
                                Some(item) => {
                                    mine.push(item);
                                    popped.fetch_add(1, Ordering::SeqCst);
                                }
                                None => std::hint::spin_loop(),
                            }
                        }
                        mine
                    })
                })
                .collect();
            for h in handles {
                all.extend(h.join().unwrap());
            }
        });

        assert_eq!(all.len(), TOTAL, "lost or duplicated entries");
        let unique: HashSet<Item> = all.iter().copied().collect();
        assert_eq!(unique.len(), TOTAL, "an entry was delivered twice");
        for p in 0..PRODUCERS {
            for seq in 0..PER_PRODUCER {
                assert!(unique.contains(&Item::new(p, seq)), "lost {p}/{seq}");
            }
        }
        assert!(q.is_empty());
        assert_eq!(q.get_head(), TOTAL as u64);
    }

    /// SPSC across two threads: strict global FIFO, nothing lost, and the
    /// producer's `ErrorOnNoSpace` rollback must never corrupt `tail` (it would
    /// show up as a duplicate or a skipped value).
    #[test]
    fn spsc_across_threads_preserves_strict_fifo() {
        const COUNT: u64 = 50_000;

        let q = Spsc::<u64>::new(4);
        thread::scope(|s| {
            let qp = &q;
            s.spawn(move || {
                for i in 0..COUNT {
                    // Full is expected constantly at depth 4: retry.
                    while !qp.push(i) {
                        std::hint::spin_loop();
                    }
                }
            });
            let qc = &q;
            let consumer = s.spawn(move || {
                let mut expect = 0u64;
                while expect < COUNT {
                    if let Some(v) = qc.pop() {
                        assert_eq!(v, expect, "SPSC must be strictly FIFO");
                        expect += 1;
                    } else {
                        std::hint::spin_loop();
                    }
                }
                expect
            });
            assert_eq!(consumer.join().unwrap(), COUNT);
        });
        assert!(q.is_empty());
        assert_eq!(q.get_tail(), COUNT, "no ticket drift from failed pushes");
    }

    /// Divergence 14, deterministically: a second consumer on a queue **without**
    /// `RING_BUFFER_LOCK_POP` can hold a `head` that a ring lap has invalidated,
    /// and the slot it is about to claim then holds a *future* entry.
    ///
    /// This reproduces, single-threaded and without timing luck, every
    /// precondition of the wedge that the threaded version hits within seconds:
    /// `PopUnlocked` samples `head`, and only afterwards tests the slot's ready
    /// flag. Nothing in the algorithm revalidates `head` between the two, so the
    /// stale consumer wins the CAS on an entry that is not the one it sampled,
    /// and then stores `stale_head + 1` — rewinding `head` over entries other
    /// consumers already delivered.
    #[test]
    fn stale_head_lets_an_unlocked_consumer_claim_a_future_slot_cpp_bug() {
        // 3 slots, so cursor N and cursor N+3 share a slot.
        let q = Mpsc::<Item>::new(2);
        assert!(q.push(Item::new(0, 0)));
        assert!(q.push(Item::new(0, 1)));

        // Consumer A enters PopUnlocked and samples head... then is preempted.
        let stale_head = q.get_head();
        assert_eq!(stale_head, 0);
        let sampled = q.slot(stale_head).get_data();
        assert_eq!(sampled, Item::new(0, 0), "A intended to deliver entry 0");

        // Consumers B/C drain the ring while A sleeps.
        assert_eq!(q.pop(), Some(Item::new(0, 0)));
        assert_eq!(q.pop(), Some(Item::new(0, 1)));
        assert_eq!(q.get_head(), 2);

        // The producer laps the ring: cursor 3 reuses slot 0 (3 % 3 == 0).
        assert!(q.push(Item::new(0, 2)));
        assert!(q.push(Item::new(0, 3)));
        assert_eq!(q.get_tail(), 4);

        // A resumes. Its stale head still maps to slot 0 — which is ready again,
        // but with a *different, future* entry. A's ready check passes and A wins
        // the claim CAS.
        let entry = q.slot(stale_head);
        assert!(
            entry.is_ready_system(),
            "the stale slot looks ready: A's ready check cannot tell it apart"
        );
        assert_eq!(
            entry.get_data(),
            Item::new(0, 3),
            "A would deliver entry 3 (cursor 3), not the entry 0 it sampled"
        );
        // ...and A would then `head_.store(stale_head + 1)` == 1, rewinding head
        // from 2 back over entries 0 and 1, which were already delivered.
        assert!(
            stale_head + 1 < q.get_head(),
            "storing stale_head+1 rewinds head: {} < {}",
            stale_head + 1,
            q.get_head()
        );
        // Empirically (4 consumers, depth 64) this rewinds head within seconds —
        // observed 4548 -> 4526 — after which every slot between the rewound head
        // and the true head has flags == 0, so Pop returns None forever and the
        // waiting producers block on a ring that can never drain: a total
        // deadlock. Hence: multi-consumer REQUIRES MpmcRingBuffer.
    }

    /// The safe multi-consumer configuration is [`MpmcRingBuffer`] — the mutex
    /// keeps the `head` sample and the claim in one critical section, so the
    /// stale-head window above cannot open. This is the contrast case for
    /// divergence 14: same shape as the test above, but locked.
    #[test]
    fn locked_pop_closes_the_stale_head_window() {
        const COUNT: u32 = 20_000;
        const CONSUMERS: usize = 4;

        // Depth 4: the ring laps constantly, which is exactly what breaks the
        // unlocked configuration.
        let q = Mpmc::<Item>::new(4);
        let popped = AtomicUsize::new(0);
        let mut all: Vec<Item> = Vec::with_capacity(COUNT as usize);

        thread::scope(|s| {
            let qp = &q;
            s.spawn(move || {
                for seq in 0..COUNT {
                    assert!(qp.push(Item::new(0, seq)));
                }
            });
            let handles: Vec<_> = (0..CONSUMERS)
                .map(|_| {
                    let q = &q;
                    let popped = &popped;
                    s.spawn(move || {
                        let mut mine = Vec::new();
                        while popped.load(Ordering::SeqCst) < COUNT as usize {
                            if let Some(item) = q.pop() {
                                mine.push(item);
                                popped.fetch_add(1, Ordering::SeqCst);
                            } else {
                                std::hint::spin_loop();
                            }
                        }
                        mine
                    })
                })
                .collect();
            for h in handles {
                all.extend(h.join().unwrap());
            }
        });

        assert_eq!(all.len(), COUNT as usize);
        let unique: HashSet<Item> = all.iter().copied().collect();
        assert_eq!(unique.len(), COUNT as usize, "an entry was delivered twice");
        assert!(q.is_empty());
        assert_eq!(q.get_head(), COUNT as u64, "head advanced monotonically");
    }

    /// The metadata fields are written by the orchestrator while workers read
    /// them; as atomics (divergence 3) that is race-free by construction.
    #[test]
    fn lane_metadata_is_safe_to_touch_concurrently() {
        let q = Mpsc::<u64>::new(4);
        thread::scope(|s| {
            let qw = &q;
            s.spawn(move || {
                for i in 0..10_000u32 {
                    qw.set_assigned_worker_id(i);
                    qw.set_active(i % 2 == 0);
                }
            });
            let qr = &q;
            s.spawn(move || {
                for _ in 0..10_000 {
                    let id = qr.assigned_worker_id();
                    assert!(id < 10_000);
                    let _ = qr.is_active();
                }
            });
        });
    }
}
