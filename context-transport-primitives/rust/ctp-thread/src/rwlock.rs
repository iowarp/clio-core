// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).

//! Reader-writer locks — port of `clio_ctp/thread/lock/rwlock.h` (the spinning,
//! SHM-capable `ctp::RwLock`) and `clio_ctp/thread/lock/cvrwlock.h` (the
//! host-only, parking `ctp::CvRwLock`).
//!
//! Both C++ headers live in `thread/lock/` and both implement a reader-writer
//! lock, so both land here; the crate's module list has no `cvrwlock` slot and
//! `lib.rs` is not ours to edit.
//!
//! # `RwLock` at a glance
//!
//! Reader-preferring spin lock with a **ticket** ordering among writers and a
//! **batched-fairness** bound on starvation. `mode` is the exclusion gate
//! (`NONE`/`READ`/`WRITE`); `readers`/`writers` are population counts that lag
//! the mode, and `update_mode` is what collapses a drained phase back to
//! `NONE` so the other side can take over. Once a phase has served
//! [`RwLock::FAIRNESS_BATCH`] acquisitions *and* the opposite side is waiting,
//! new same-side acquirers defer until the phase drains.
//!
//! # C++ → Rust name mapping
//!
//! | C++ (`rwlock.h`) | Rust | Notes |
//! |---|---|---|
//! | `RwLockMode::Type` | [`RwLockModeType`] (`i32`) | stays a plain int, not an enum — see divergence 1 |
//! | `RwLockMode::kNone` | [`RwLockMode::NONE`] | `0` |
//! | `RwLockMode::kWrite` | [`RwLockMode::WRITE`] | `1` |
//! | `RwLockMode::kRead` | [`RwLockMode::READ`] | `2` |
//! | `RwLock` | [`RwLock`] | `#[repr(C)]`, layout-compatible |
//! | `RwLock::mode_` | [`RwLock::mode`] | `ipc::atomic<RwLockMode::Type>` → `AtomicI32` |
//! | `RwLock::readers_` | [`RwLock::readers`] | `ipc::atomic<reg_uint>` → `AtomicU32` |
//! | `RwLock::writers_` | [`RwLock::writers`] | `ipc::atomic<reg_uint>` → `AtomicU32` |
//! | `RwLock::cur_writer_` | [`RwLock::cur_writer`] | `ipc::atomic<reg_uint>` → `AtomicU32` |
//! | `RwLock::ticket_` | [`RwLock::ticket`] | `ipc::atomic<big_uint>` → `AtomicU64` |
//! | `RwLock::ops_since_switch_` | [`RwLock::ops_since_switch`] | `ipc::atomic<reg_uint>` → `AtomicU32` |
//! | `RwLock::kFairnessBatch` | [`RwLock::FAIRNESS_BATCH`] | `32` |
//! | `RwLock()` / `Init()` | [`RwLock::new`] / [`RwLock::init`] | |
//! | `RwLock(const RwLock&)` | `<RwLock as Clone>::clone` | yields a fresh, unheld lock — divergence 4 |
//! | `RwLock(RwLock&&)` | (native Rust move) | divergence 4 |
//! | `ReadLock(owner)` | [`RwLock::read_lock`] | `writer_priority = false` default |
//! | `ReadLock(owner, writer_priority)` | [`RwLock::read_lock_with`] | divergence 3 |
//! | `ReadUnlock()` | [`RwLock::read_unlock`] | |
//! | `WriteLock(owner)` | [`RwLock::write_lock`] | |
//! | `WriteUnlock()` | [`RwLock::write_unlock`] | |
//! | `IsWriteLocked()` | [`RwLock::is_write_locked`] | |
//! | `UpdateMode(mode&)` | `RwLock::update_mode` (private) | |
//! | `ScopedRwReadLock` | [`ScopedRwReadLock`] | RAII via `Drop` |
//! | `ScopedRwWriteLock` | [`ScopedRwWriteLock`] | RAII via `Drop` |
//! | `ScopedRw*Lock::is_locked_` | `is_locked` (private) | |
//! | `ScopedRw*Lock::Lock/Unlock` | `lock` / `unlock` | re-lockable, same idempotence |
//!
//! | C++ (`cvrwlock.h`) | Rust | Notes |
//! |---|---|---|
//! | `CvRwLock` | [`CvRwLock`] | host-only in both; NOT SHM-capable |
//! | `readers_` (`atomic<int64_t>`) | `readers` (`AtomicI64`) | private, as in C++ |
//! | `write_intent_` | `write_intent` (`AtomicI64`) | private |
//! | `mtx_` + `writer_active_` | `writer_active: Mutex<bool>` | divergence 8 |
//! | `cv_` | `cv: Condvar` | |
//! | `Init()` | [`CvRwLock::init`] | no-op, for parity |
//! | `ReadLock()` / `ReadUnlock()` | [`CvRwLock::read_lock`] / [`CvRwLock::read_unlock`] | no `owner` param, as in C++ |
//! | `WriteLock()` / `WriteUnlock()` | [`CvRwLock::write_lock`] / [`CvRwLock::write_unlock`] | |
//!
//! # Semantic divergences
//!
//! 1. **`RwLockMode` is an `i32`, not an `enum`.** `mode` is an atomic word in
//!    shared memory that a peer process (or a stale/corrupt mapping) can set to
//!    any bit pattern; a Rust `enum` with 3 variants would make an unexpected
//!    value instant UB. The C++ `typedef int Type` + `CLS_CONST` constants are
//!    reproduced literally as `i32` associated consts. Unknown values behave as
//!    in C++: they match neither arm, so the acquirer keeps spinning.
//! 2. **Yielding goes through the ported thread model, which today has only a
//!    `StdThread` backend.** Every spin point calls
//!    `thread_model().yield_now()`, i.e. the Rust `CTP_THREAD_MODEL->Yield()`,
//!    so this module is faithful and automatically inherits any backend the
//!    thread-model port later gains. The C++ can select Argobots/pthread/fiber
//!    backends at build time; until `thread_model.rs` grows those, a spin here
//!    yields the OS thread rather than rescheduling a fiber. That gap lives in
//!    `thread_model.rs`, not here — this module has no independent yield path.
//! 3. **No default arguments.** `ReadLock(owner, writer_priority = false)`
//!    splits into [`RwLock::read_lock`] (the `false` default that reentrant
//!    read→read callers depend on) and [`RwLock::read_lock_with`].
//! 4. **Copy/move.** The C++ copy ctor/assign deliberately produce a *fresh,
//!    unheld* lock (containers need *some* copy ctor). [`Clone`] reproduces
//!    that exactly — cloning a held lock yields an unheld one, which is
//!    surprising by Rust convention but is the ported contract. The C++ move
//!    ctor copies the raw counter values; a Rust move is a bitwise copy of the
//!    same words, so it is already equivalent and needs no code.
//! 5. **`owner` is accepted and ignored**, exactly as in the C++ (the parameter
//!    is documented there as "legacy/unused"). Kept in the signature for
//!    call-site parity; named `_owner`.
//! 6. **Ticket/`cur_writer` width mismatch is preserved.** `ticket` is 64-bit
//!    and `cur_writer` is 32-bit, and `WriteLock` compares them
//!    (`cur_writer == tkt`, with the C++ integer promotion reproduced as
//!    `u64::from(cur_writer) == tkt`). After 2^32 *write* acquisitions
//!    `cur_writer` wraps to 0 while `ticket` keeps climbing, so the comparison
//!    can never match and that writer spins forever. This is a latent defect in
//!    the C++ (not a porting artifact); it is reproduced rather than silently
//!    fixed, because fixing it changes the on-wire meaning of a `#[repr(C)]`
//!    field shared with live C++ processes. Fixing it in both languages at once
//!    (widen `cur_writer_` to 64-bit, or compare modulo 2^32) is a follow-up.
//!    `write_lock_boundary_at_u32_max_ticket` pins the last ticket that works.
//! 7. **Counters wrap, they do not saturate.** `readers`/`writers`/
//!    `ops_since_switch` are unsigned and use `fetch_add`/`fetch_sub`, which
//!    wrap in Rust (in debug *and* release) exactly as C++ unsigned atomics do.
//!    An unbalanced `read_unlock` therefore wraps `readers` to `u32::MAX`
//!    rather than panicking, and a wrapped `ops_since_switch` transiently
//!    re-opens the fairness gate. Both are pinned by tests.
//! 8. **`CvRwLock` mutex/poisoning.** C++ has no poisoning; `Mutex` lock errors
//!    are unwrapped with `into_inner()` so a panic in one thread does not turn
//!    every later acquisition into a panic. `writer_active_` (a `bool` guarded
//!    by `mtx_`) becomes `Mutex<bool>`, which is the same thing with the
//!    guard-relationship made explicit rather than by convention.
//! 9. **No `ShmSafe` marker on [`RwLock`].** The marker lives in `ctp-memory`,
//!    which is not a dependency of `ctp-thread` and may not be added. `RwLock`
//!    satisfies the contract structurally (`#[repr(C)]`, no `Drop`, no
//!    pointers/references, `'static`, all-atomic fields); the `unsafe impl
//!    ShmSafe for RwLock {}` belongs wherever the dependency edge exists.
//! 10. **No lock-upgrade API — none exists in the C++.** `ctp::RwLock` has no
//!     read→write upgrade path (the word "upgrade" appears in `rwlock.h` only in
//!     a comment about a *migration* writer observed via `IsWriteLocked`).
//!     Nothing was dropped; there is nothing to port. Attempting a read→write
//!     upgrade by taking `write_lock` while holding a read hold self-deadlocks
//!     in both languages, since the writer spins for the read phase to drain.
//! 11. **GPU (`CTP_IS_DEVICE_PASS`) paths are absent.** The C++ compiles for
//!     device with the yields `#if`-ed out and uses `load_device()` for cross-SM
//!     L2 visibility. On the host `load_device()` *is* a `seq_cst` load, so
//!     every atomic op here uses [`Ordering::SeqCst`] and is exactly the C++
//!     host behavior. Device support is out of scope for this crate.

use crate::thread_model::{thread_model, ThreadModel};
use std::sync::atomic::{AtomicI32, AtomicI64, AtomicU32, AtomicU64, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard};

/// Every atomic op mirrors the C++ host `ipc::atomic`, whose defaults — and
/// whose `load_device()` on the host pass — are all `memory_order_seq_cst`.
const ORD: Ordering = Ordering::SeqCst;

/// Single yield point for the whole module: `CTP_THREAD_MODEL->Yield()`.
#[inline]
fn yield_now() {
    thread_model().yield_now();
}

// ---------------------------------------------------------------------------
// RwLockMode
// ---------------------------------------------------------------------------

/// Mirrors `ctp::RwLockMode::Type` (`typedef int`). See divergence 1 for why
/// this is a raw `i32` rather than an `enum`.
pub type RwLockModeType = i32;

/// Namespace for the lock-mode constants (mirrors the C++ `class RwLockMode`,
/// which is a constant holder, not an instantiable type).
pub struct RwLockMode;

impl RwLockMode {
    /// `RwLockMode::kNone` — unheld; the next acquirer CASes the mode to its
    /// own side.
    pub const NONE: RwLockModeType = 0;
    /// `RwLockMode::kWrite` — a write phase; the ticket holder equal to
    /// `cur_writer` holds the lock.
    pub const WRITE: RwLockModeType = 1;
    /// `RwLockMode::kRead` — a read phase; any number of readers hold.
    pub const READ: RwLockModeType = 2;
}

// ---------------------------------------------------------------------------
// RwLock
// ---------------------------------------------------------------------------

/// A reader-writer lock — port of `ctp::RwLock`.
///
/// `#[repr(C)]` with the C++ field *declaration* order, so an instance is
/// byte-compatible with the C++ struct and can live in a shared segment mapped
/// by both languages (32 bytes, 8-byte aligned; pinned by
/// `shm_layout_matches_cpp`). No `Drop`, no pointers — see divergence 9 for the
/// missing `ShmSafe` marker.
///
/// Fields are public because the C++ `struct` members are public and callers
/// (and the FFI layer) read them; mutating them directly outside the lock
/// protocol is as unsupported here as it is there.
///
/// # Example
///
/// ```
/// use ctp_thread::rwlock::{RwLock, ScopedRwReadLock, ScopedRwWriteLock};
///
/// let lock = RwLock::new();
/// {
///     let _w = ScopedRwWriteLock::new(&lock, 0); // exclusive
///     assert!(lock.is_write_locked());
/// }
/// let _r1 = ScopedRwReadLock::new(&lock, 0); // shared
/// let _r2 = ScopedRwReadLock::new(&lock, 0);
/// ```
#[repr(C)]
#[derive(Debug)]
pub struct RwLock {
    /// `mode_`: the exclusion gate. One of the [`RwLockMode`] constants.
    pub mode: AtomicI32,
    /// `readers_`: readers holding *or* spinning to hold. Lags `mode`.
    pub readers: AtomicU32,
    /// `writers_`: writers holding *or* queued. Lags `mode`.
    pub writers: AtomicU32,
    /// `cur_writer_`: ticket currently authorized to write. 32-bit — see
    /// divergence 6.
    pub cur_writer: AtomicU32,
    /// `ticket_`: monotonically issued writer tickets.
    pub ticket: AtomicU64,
    /// `ops_since_switch_`: acquisitions granted since the last flip to
    /// [`RwLockMode::NONE`]; the batched-fairness budget.
    pub ops_since_switch: AtomicU32,
}

impl RwLock {
    /// `kFairnessBatch`: ops granted in one phase before yielding to a waiting
    /// opposite side.
    pub const FAIRNESS_BATCH: u32 = 32;

    /// Default constructor: a fresh, unheld lock.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            mode: AtomicI32::new(RwLockMode::NONE),
            readers: AtomicU32::new(0),
            writers: AtomicU32::new(0),
            cur_writer: AtomicU32::new(0),
            ticket: AtomicU64::new(0),
            ops_since_switch: AtomicU32::new(0),
        }
    }

    /// Explicit initializer (`Init()`): reset to the unheld state.
    ///
    /// Takes `&self` — the C++ `Init()` assigns through atomics, which is
    /// interior mutability, and in-place init of a lock living in a shared
    /// segment only ever has a shared reference to work with. Resetting a lock
    /// other parties are using is a caller error in both languages.
    pub fn init(&self) {
        self.readers.store(0, ORD);
        self.writers.store(0, ORD);
        self.ticket.store(0, ORD);
        self.mode.store(RwLockMode::NONE, ORD);
        self.cur_writer.store(0, ORD);
        self.ops_since_switch.store(0, ORD);
    }

    /// Acquire the read lock (`writer_priority = false`, the C++ default).
    ///
    /// Reader-preferring: enters immediately whenever the lock is already in
    /// read mode, even with a writer queued. Reentrant read→read callers depend
    /// on this.
    pub fn read_lock(&self, owner: u32) {
        self.read_lock_with(owner, false);
    }

    /// Acquire the read lock, optionally deferring to waiting writers.
    ///
    /// `writer_priority = true` makes this otherwise reader-preferring lock
    /// writer-preferring **for this acquisition only**: defer entry while a
    /// writer is waiting on or holds the lock. This is what keeps a sustained
    /// reader stream from starving writers (the C++ notes an icx/Windows ctest
    /// timeout in the `unordered_map_ll` growth stress test as the observed
    /// symptom).
    ///
    /// The wait happens **before** registering as a reader: incrementing
    /// `readers` first and only then waiting for `writers == 0` would deadlock
    /// against a writer, which spins for `readers == 0`. Stragglers that slip
    /// past before a writer bumps `writers` are bounded — the writer's
    /// ticket-based [`write_lock`](Self::write_lock) drains them — so no reader
    /// is blocked forever.
    pub fn read_lock_with(&self, _owner: u32, writer_priority: bool) {
        let mut mode = self.mode.load(ORD);

        // Batched fairness: if a writer is waiting and this read phase has
        // already served a full batch, let the current readers drain
        // (readers -> 0) so the waiting writer can take over before we pile on
        // another reader. New readers gate here too, so `readers` actually
        // reaches 0. Checked once (bounded).
        if self.writers.load(ORD) > 0
            && mode == RwLockMode::READ
            && self.ops_since_switch.load(ORD) >= Self::FAIRNESS_BATCH
        {
            while self.readers.load(ORD) > 0 {
                yield_now();
            }
        }
        self.ops_since_switch.fetch_add(1, ORD);

        if writer_priority {
            while self.writers.load(ORD) > 0 {
                yield_now();
            }
        }

        // Increment # readers, then wait until we are in read mode.
        self.readers.fetch_add(1, ORD);

        loop {
            self.update_mode(&mut mode);
            if mode == RwLockMode::READ {
                return;
            }
            if mode == RwLockMode::NONE {
                match self
                    .mode
                    .compare_exchange_weak(mode, RwLockMode::READ, ORD, ORD)
                {
                    Ok(_) => return,
                    // The C++ CAS writes the observed value back into its
                    // by-reference `expected`; mirrored for fidelity even
                    // though `update_mode` reloads it at the top of the loop.
                    Err(actual) => mode = actual,
                }
            }
            yield_now();
        }
    }

    /// Release the read lock.
    pub fn read_unlock(&self) {
        self.readers.fetch_sub(1, ORD);
    }

    /// Acquire the write lock. Writers are served in ticket order.
    pub fn write_lock(&self, _owner: u32) {
        let mut mode = self.mode.load(ORD);

        // Batched fairness: if readers are waiting and this write phase has
        // served a full batch, let the current writers drain (writers -> 0) so
        // the waiting readers can take over before we queue another writer. New
        // writers gate here too, so `writers` actually reaches 0. Checked once
        // (bounded).
        if self.readers.load(ORD) > 0
            && mode == RwLockMode::WRITE
            && self.ops_since_switch.load(ORD) >= Self::FAIRNESS_BATCH
        {
            while self.writers.load(ORD) > 0 {
                yield_now();
            }
        }
        self.ops_since_switch.fetch_add(1, ORD);

        // Increment # writers & get ticket.
        self.writers.fetch_add(1, ORD);
        let tkt = self.ticket.fetch_add(1, ORD);

        loop {
            self.update_mode(&mut mode);
            if mode == RwLockMode::NONE {
                let _ = self
                    .mode
                    .compare_exchange_weak(mode, RwLockMode::WRITE, ORD, ORD);
                // The C++ re-loads unconditionally here (discarding whatever
                // the CAS wrote into `expected`), so the CAS result is ignored.
                mode = self.mode.load(ORD);
            }
            if mode == RwLockMode::WRITE {
                let cur_writer = self.cur_writer.load(ORD);
                // C++ compares a 32-bit `cur_writer` against a 64-bit ticket
                // via integer promotion. See divergence 6.
                if u64::from(cur_writer) == tkt {
                    return;
                }
            }
            yield_now();
        }
    }

    /// Release the write lock and hand the phase to the next ticket.
    pub fn write_unlock(&self) {
        self.writers.fetch_sub(1, ORD);
        self.cur_writer.fetch_add(1, ORD);
    }

    /// True while a writer holds (or is waiting/draining for) this lock.
    ///
    /// Used by the C++ `ContainerPtr::IsPlugged()` so a reader can observe that
    /// a migration/upgrade writer is in progress on the container.
    #[must_use]
    pub fn is_write_locked(&self) -> bool {
        self.writers.load(ORD) > 0
    }

    /// Update the mode of the lock (`UpdateMode`).
    ///
    /// When `readers` hits 0 there is a lag before the mode leaves `READ`, and
    /// likewise for `writers`/`WRITE`; this collapses a drained phase to
    /// `NONE`. On a successful flip the batch counter restarts so the next
    /// phase gets a fresh [`FAIRNESS_BATCH`](Self::FAIRNESS_BATCH) budget.
    ///
    /// Faithful subtlety: a C++ CAS leaves its by-reference `expected`
    /// **unchanged on success**, so after a successful flip the caller's `mode`
    /// still reads `READ`/`WRITE` (not `NONE`) and falls through to another
    /// spin iteration. Callers here observe the same thing.
    fn update_mode(&self, mode: &mut RwLockModeType) {
        *mode = self.mode.load(ORD);
        if (self.readers.load(ORD) == 0 && *mode == RwLockMode::READ)
            || (self.writers.load(ORD) == 0 && *mode == RwLockMode::WRITE)
        {
            match self
                .mode
                .compare_exchange_weak(*mode, RwLockMode::NONE, ORD, ORD)
            {
                Ok(_) => self.ops_since_switch.store(0, ORD),
                Err(actual) => *mode = actual,
            }
        }
    }
}

impl Default for RwLock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for RwLock {
    /// Mirrors the C++ copy ctor: construct a **fresh, unheld** lock rather
    /// than cloning live state. See divergence 4.
    fn clone(&self) -> Self {
        Self::new()
    }

    /// Mirrors the C++ copy-assignment operator, which stores zeros into the
    /// destination and ignores the source entirely.
    fn clone_from(&mut self, _source: &Self) {
        self.init();
    }
}

// ---------------------------------------------------------------------------
// Scoped guards
// ---------------------------------------------------------------------------

/// Acquire the read lock for a scope (`ctp::ScopedRwReadLock`).
///
/// Releases on `Drop`. Like the C++, [`lock`](Self::lock) and
/// [`unlock`](Self::unlock) are idempotent, so an explicit `unlock` followed by
/// the destructor releases exactly once.
#[derive(Debug)]
pub struct ScopedRwReadLock<'a> {
    lock: &'a RwLock,
    is_locked: bool,
}

impl<'a> ScopedRwReadLock<'a> {
    /// Acquire the read lock.
    #[must_use]
    pub fn new(lock: &'a RwLock, owner: u32) -> Self {
        let mut guard = Self {
            lock,
            is_locked: false,
        };
        guard.lock(owner);
        guard
    }

    /// Explicitly acquire the read lock (no-op if already held).
    pub fn lock(&mut self, owner: u32) {
        if !self.is_locked {
            self.lock.read_lock(owner);
            self.is_locked = true;
        }
    }

    /// Explicitly release the read lock (no-op if not held).
    pub fn unlock(&mut self) {
        if self.is_locked {
            self.lock.read_unlock();
            self.is_locked = false;
        }
    }
}

impl Drop for ScopedRwReadLock<'_> {
    fn drop(&mut self) {
        self.unlock();
    }
}

/// Acquire the write lock for a scope (`ctp::ScopedRwWriteLock`).
///
/// Releases on `Drop`. Like the C++, [`lock`](Self::lock) and
/// [`unlock`](Self::unlock) are idempotent.
#[derive(Debug)]
pub struct ScopedRwWriteLock<'a> {
    lock: &'a RwLock,
    is_locked: bool,
}

impl<'a> ScopedRwWriteLock<'a> {
    /// Acquire the write lock.
    #[must_use]
    pub fn new(lock: &'a RwLock, owner: u32) -> Self {
        let mut guard = Self {
            lock,
            is_locked: false,
        };
        guard.lock(owner);
        guard
    }

    /// Explicitly acquire the write lock (no-op if already held).
    pub fn lock(&mut self, owner: u32) {
        if !self.is_locked {
            self.lock.write_lock(owner);
            self.is_locked = true;
        }
    }

    /// Explicitly release the write lock (no-op if not held).
    pub fn unlock(&mut self) {
        if self.is_locked {
            self.lock.write_unlock();
            self.is_locked = false;
        }
    }
}

impl Drop for ScopedRwWriteLock<'_> {
    fn drop(&mut self) {
        self.unlock();
    }
}

// ---------------------------------------------------------------------------
// CvRwLock (cvrwlock.h)
// ---------------------------------------------------------------------------

/// Writer-preferring reader/writer lock with a lock-free parallel reader path
/// and condition-variable blocking on the slow path — port of `ctp::CvRwLock`.
///
/// A sleep-friendly alternative to the spin-based [`RwLock`]; method names
/// mirror it so the two can be swapped for comparison.
///
/// Readers are **parallel**: the common no-writer read path is lock-free (a
/// single atomic increment on `readers` guarded by an optimistic re-check of
/// the writer-intent counter) and never touches the mutex, so concurrent
/// readers do not serialize on it. The mutex and condvar are used only on the
/// slow path — when a reader must park behind a writer, or when a writer must
/// wait for readers to drain.
///
/// Writer preference: a writer bumps `write_intent` on arrival, which makes new
/// readers back off, so a steady reader stream cannot starve writers.
///
/// **HOST ONLY and NOT SHM-capable**, exactly as in C++ (`#if
/// !CTP_IS_DEVICE_PASS`, and `std::mutex`/`std::condition_variable` are
/// process-local). Non-copyable and non-movable in C++; in Rust it is simply
/// neither [`Clone`] nor [`Copy`], and callers share it by reference.
#[derive(Debug, Default)]
pub struct CvRwLock {
    /// Active readers. Modified lock-free on the reader fast path; read under
    /// the mutex by a waiting writer's predicate.
    readers: AtomicI64,
    /// Writers waiting OR active. Readers back off (do not enter) while > 0.
    write_intent: AtomicI64,
    /// The C++ `mtx_` + `writer_active_` pair: the bool and the mutex that
    /// guards it, fused (divergence 8).
    writer_active: Mutex<bool>,
    cv: Condvar,
}

impl CvRwLock {
    /// Construct an unheld lock.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            readers: AtomicI64::new(0),
            write_intent: AtomicI64::new(0),
            writer_active: Mutex::new(false),
            cv: Condvar::new(),
        }
    }

    /// No-op initializer, for API parity with [`RwLock`].
    pub fn init(&self) {}

    /// Lock the state mutex, ignoring poisoning (C++ has no such concept).
    fn lock_state(&self) -> MutexGuard<'_, bool> {
        self.writer_active
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Acquire the lock in shared (read) mode. Lock-free when no writer is
    /// waiting/active; otherwise parks on the condition variable.
    pub fn read_lock(&self) {
        loop {
            if self.write_intent.load(ORD) == 0 {
                // Optimistically register as a reader, then re-check for a
                // writer that may have arrived in between. If none, we hold the
                // lock — no mutex.
                self.readers.fetch_add(1, ORD);
                if self.write_intent.load(ORD) == 0 {
                    return;
                }
                // A writer appeared; back out. If we were the last reader, wake
                // it.
                if self.readers.fetch_sub(1, ORD) == 1 {
                    let _state = self.lock_state();
                    self.cv.notify_all();
                }
            }
            // Slow path: park until no writer is waiting/active, then retry.
            let mut state = self.lock_state();
            while self.write_intent.load(ORD) != 0 {
                state = self
                    .cv
                    .wait(state)
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
            }
        }
    }

    /// Release a shared (read) hold. Lock-free unless we are the last reader
    /// and a writer is waiting, in which case we wake it.
    pub fn read_unlock(&self) {
        if self.readers.fetch_sub(1, ORD) == 1 && self.write_intent.load(ORD) > 0 {
            let _state = self.lock_state();
            self.cv.notify_all();
        }
    }

    /// Acquire the lock in exclusive (write) mode. Announces intent so new
    /// readers back off, then waits until no writer holds it and readers drain.
    pub fn write_lock(&self) {
        self.write_intent.fetch_add(1, ORD); // block new readers
        let mut state = self.lock_state();
        while *state || self.readers.load(ORD) != 0 {
            state = self
                .cv
                .wait(state)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
        *state = true;
    }

    /// Release an exclusive (write) hold; wake the next writer and/or readers.
    pub fn write_unlock(&self) {
        let mut state = self.lock_state();
        *state = false;
        self.write_intent.fetch_sub(1, ORD); // re-admit readers if none waits
        self.cv.notify_all();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, offset_of, size_of};
    use std::sync::atomic::AtomicBool;
    use std::thread;
    use std::time::Duration;

    /// Long enough that a thread which *can* make progress will have, short
    /// enough to keep the suite fast. Only used for "must NOT have progressed"
    /// assertions, which are conservative: a slow machine makes them pass, not
    /// flake.
    const SETTLE: Duration = Duration::from_millis(50);

    // -- layout / ABI ----------------------------------------------------

    #[test]
    fn shm_layout_matches_cpp() {
        // C++: int(4) u32(4) u32(4) u32(4) | u64(8) | u32(4) + 4 pad = 32.
        assert_eq!(offset_of!(RwLock, mode), 0);
        assert_eq!(offset_of!(RwLock, readers), 4);
        assert_eq!(offset_of!(RwLock, writers), 8);
        assert_eq!(offset_of!(RwLock, cur_writer), 12);
        assert_eq!(offset_of!(RwLock, ticket), 16);
        assert_eq!(offset_of!(RwLock, ops_since_switch), 24);
        assert_eq!(size_of::<RwLock>(), 32);
        assert_eq!(align_of::<RwLock>(), 8);
    }

    #[test]
    fn rwlock_is_send_and_sync_and_dropless() {
        fn assert_shm_shaped<T: Send + Sync + 'static>() {}
        assert_shm_shaped::<RwLock>();
        // No Drop: required for a type living in a shared segment.
        assert!(!std::mem::needs_drop::<RwLock>());
    }

    #[test]
    fn mode_constants_match_cpp() {
        assert_eq!(RwLockMode::NONE, 0);
        assert_eq!(RwLockMode::WRITE, 1);
        assert_eq!(RwLockMode::READ, 2);
        assert_eq!(RwLock::FAIRNESS_BATCH, 32);
    }

    // -- construction / init ---------------------------------------------

    #[test]
    fn new_and_default_are_unheld() {
        for lock in [RwLock::new(), RwLock::default()] {
            assert_eq!(lock.mode.load(ORD), RwLockMode::NONE);
            assert_eq!(lock.readers.load(ORD), 0);
            assert_eq!(lock.writers.load(ORD), 0);
            assert_eq!(lock.cur_writer.load(ORD), 0);
            assert_eq!(lock.ticket.load(ORD), 0);
            assert_eq!(lock.ops_since_switch.load(ORD), 0);
            assert!(!lock.is_write_locked());
        }
    }

    #[test]
    fn init_resets_every_field() {
        let lock = RwLock::new();
        lock.mode.store(RwLockMode::WRITE, ORD);
        lock.readers.store(7, ORD);
        lock.writers.store(3, ORD);
        lock.cur_writer.store(9, ORD);
        lock.ticket.store(11, ORD);
        lock.ops_since_switch.store(31, ORD);

        lock.init();

        assert_eq!(lock.mode.load(ORD), RwLockMode::NONE);
        assert_eq!(lock.readers.load(ORD), 0);
        assert_eq!(lock.writers.load(ORD), 0);
        assert_eq!(lock.cur_writer.load(ORD), 0);
        assert_eq!(lock.ticket.load(ORD), 0);
        assert_eq!(lock.ops_since_switch.load(ORD), 0);
    }

    #[test]
    fn clone_yields_a_fresh_unheld_lock() {
        // Divergence 4: the C++ copy ctor deliberately does NOT clone state.
        let lock = RwLock::new();
        lock.write_lock(0);
        assert!(lock.is_write_locked());

        let copy = lock.clone();
        assert_eq!(copy.mode.load(ORD), RwLockMode::NONE);
        assert_eq!(copy.writers.load(ORD), 0);
        assert_eq!(copy.ticket.load(ORD), 0);
        assert!(!copy.is_write_locked());

        lock.write_unlock();
    }

    #[test]
    fn clone_from_resets_destination_and_ignores_source() {
        let source = RwLock::new();
        source.readers.store(5, ORD);
        let mut dest = RwLock::new();
        dest.writers.store(2, ORD);
        dest.mode.store(RwLockMode::WRITE, ORD);

        dest.clone_from(&source);

        assert_eq!(dest.readers.load(ORD), 0); // NOT 5 — source is ignored
        assert_eq!(dest.writers.load(ORD), 0);
        assert_eq!(dest.mode.load(ORD), RwLockMode::NONE);
    }

    // -- single-threaded acquire/release ---------------------------------

    #[test]
    fn read_lock_enters_read_mode_and_counts_readers() {
        let lock = RwLock::new();
        lock.read_lock(0);
        assert_eq!(lock.mode.load(ORD), RwLockMode::READ);
        assert_eq!(lock.readers.load(ORD), 1);
        assert_eq!(lock.ops_since_switch.load(ORD), 1);
        assert!(!lock.is_write_locked());

        lock.read_unlock();
        assert_eq!(lock.readers.load(ORD), 0);
        // The mode LAGS: it stays READ until some update_mode observes the
        // drain. This lag is load-bearing for the reader-preferring fast path.
        assert_eq!(lock.mode.load(ORD), RwLockMode::READ);
    }

    #[test]
    fn nested_read_locks_are_reentrant() {
        // Reader-preferring: a second read hold enters while the first is held.
        let lock = RwLock::new();
        lock.read_lock(0);
        lock.read_lock(0);
        lock.read_lock(0);
        assert_eq!(lock.readers.load(ORD), 3);
        assert_eq!(lock.mode.load(ORD), RwLockMode::READ);
        for _ in 0..3 {
            lock.read_unlock();
        }
        assert_eq!(lock.readers.load(ORD), 0);
    }

    #[test]
    fn write_lock_enters_write_mode_and_takes_a_ticket() {
        let lock = RwLock::new();
        lock.write_lock(0);
        assert_eq!(lock.mode.load(ORD), RwLockMode::WRITE);
        assert_eq!(lock.writers.load(ORD), 1);
        assert_eq!(lock.ticket.load(ORD), 1); // one ticket issued
        assert_eq!(lock.cur_writer.load(ORD), 0); // ours
        assert!(lock.is_write_locked());

        lock.write_unlock();
        assert_eq!(lock.writers.load(ORD), 0);
        assert_eq!(lock.cur_writer.load(ORD), 1); // handed to the next ticket
        assert!(!lock.is_write_locked());
    }

    #[test]
    fn sequential_write_locks_advance_the_ticket() {
        let lock = RwLock::new();
        for i in 0..5u64 {
            lock.write_lock(0);
            assert_eq!(lock.ticket.load(ORD), i + 1);
            assert_eq!(u64::from(lock.cur_writer.load(ORD)), i);
            lock.write_unlock();
        }
        assert_eq!(lock.cur_writer.load(ORD), 5);
    }

    #[test]
    fn read_after_write_flips_the_mode() {
        let lock = RwLock::new();
        lock.write_lock(0);
        lock.write_unlock();
        assert_eq!(lock.mode.load(ORD), RwLockMode::WRITE); // lagging

        lock.read_lock(0); // must drain the stale WRITE mode and enter READ
        assert_eq!(lock.mode.load(ORD), RwLockMode::READ);
        assert_eq!(lock.readers.load(ORD), 1);
        lock.read_unlock();
    }

    #[test]
    fn write_after_read_resets_the_batch_counter() {
        // Pins the exact algorithm, including the C++ subtlety that a
        // successful CAS leaves `expected` unchanged (see `update_mode`):
        // write_lock bumps ops to 2, then its first update_mode flips
        // READ->NONE and stores ops=0, and the acquisition completes on the
        // NEXT iteration — so ops reads 0, not 1, once we hold the lock.
        let lock = RwLock::new();
        lock.read_lock(0);
        lock.read_unlock();
        assert_eq!(lock.ops_since_switch.load(ORD), 1);

        lock.write_lock(0);
        assert_eq!(lock.mode.load(ORD), RwLockMode::WRITE);
        assert_eq!(lock.ops_since_switch.load(ORD), 0);
        lock.write_unlock();
    }

    // -- reader/writer preference ----------------------------------------

    #[test]
    fn default_reader_enters_despite_a_waiting_writer() {
        // Reader-preferring default: in READ mode a new reader does not care
        // that a writer is queued (below the fairness batch).
        let lock = RwLock::new();
        lock.read_lock(0); // mode = READ, readers = 1
        lock.writers.fetch_add(1, ORD); // simulate a queued writer

        lock.read_lock(0); // must not block
        assert_eq!(lock.readers.load(ORD), 2);

        lock.read_unlock();
        lock.read_unlock();
        lock.writers.fetch_sub(1, ORD);
    }

    #[test]
    fn writer_priority_reader_defers_while_a_writer_is_registered() {
        let lock = RwLock::new();
        lock.writers.fetch_add(1, ORD); // simulate a writer waiting/holding
        let entered = AtomicBool::new(false);

        thread::scope(|s| {
            s.spawn(|| {
                lock.read_lock_with(0, true);
                entered.store(true, ORD);
                lock.read_unlock();
            });

            thread::sleep(SETTLE);
            assert!(!entered.load(ORD), "writer_priority reader must not enter");
            // Critically, it must not even REGISTER as a reader: doing so
            // before the wait would deadlock a writer spinning for readers==0.
            assert_eq!(lock.readers.load(ORD), 0);

            lock.writers.fetch_sub(1, ORD); // writer leaves -> reader proceeds
        });

        assert!(entered.load(ORD));
        assert_eq!(lock.readers.load(ORD), 0);
    }

    #[test]
    fn writer_priority_reader_enters_when_no_writer_is_registered() {
        let lock = RwLock::new();
        lock.read_lock_with(0, true); // no writers -> no deferral
        assert_eq!(lock.readers.load(ORD), 1);
        assert_eq!(lock.mode.load(ORD), RwLockMode::READ);
        lock.read_unlock();
    }

    // -- batched fairness -------------------------------------------------

    #[test]
    fn read_fairness_gate_defers_a_new_reader_once_the_batch_is_spent() {
        let lock = RwLock::new();
        lock.mode.store(RwLockMode::READ, ORD);
        lock.readers.store(1, ORD); // an in-flight reader
        lock.writers.store(1, ORD); // a waiting writer
        lock.ops_since_switch.store(RwLock::FAIRNESS_BATCH, ORD);
        let entered = AtomicBool::new(false);

        thread::scope(|s| {
            s.spawn(|| {
                lock.read_lock(0); // gated: waits for readers -> 0
                entered.store(true, ORD);
                lock.read_unlock();
            });

            thread::sleep(SETTLE);
            assert!(!entered.load(ORD), "reader must defer to the waiting writer");
            assert_eq!(lock.readers.load(ORD), 1, "must not pile on");

            lock.readers.store(0, ORD); // the in-flight reader drains
        });

        assert!(entered.load(ORD));
        assert_eq!(
            lock.ops_since_switch.load(ORD),
            RwLock::FAIRNESS_BATCH + 1,
            "gated reader still counts against the batch"
        );
    }

    #[test]
    fn read_fairness_gate_is_open_just_below_the_batch_boundary() {
        // Boundary: the gate is `>= FAIRNESS_BATCH`, so BATCH-1 must not defer.
        let lock = RwLock::new();
        lock.mode.store(RwLockMode::READ, ORD);
        lock.readers.store(1, ORD);
        lock.writers.store(1, ORD);
        lock.ops_since_switch.store(RwLock::FAIRNESS_BATCH - 1, ORD);

        lock.read_lock(0); // must NOT block despite readers>0 && writers>0
        assert_eq!(lock.readers.load(ORD), 2);
        lock.read_unlock();
    }

    #[test]
    fn read_fairness_gate_is_open_when_no_writer_waits() {
        // All three conditions must hold; writers==0 keeps the gate open even
        // with the batch fully spent.
        let lock = RwLock::new();
        lock.mode.store(RwLockMode::READ, ORD);
        lock.readers.store(1, ORD);
        lock.writers.store(0, ORD);
        lock.ops_since_switch.store(RwLock::FAIRNESS_BATCH * 4, ORD);

        lock.read_lock(0);
        assert_eq!(lock.readers.load(ORD), 2);
        lock.read_unlock();
        lock.read_unlock();
    }

    #[test]
    fn write_fairness_gate_defers_a_new_writer_once_the_batch_is_spent() {
        let lock = RwLock::new();
        lock.mode.store(RwLockMode::WRITE, ORD);
        lock.writers.store(1, ORD); // an in-flight writer
        lock.readers.store(1, ORD); // a waiting reader
        lock.ops_since_switch.store(RwLock::FAIRNESS_BATCH, ORD);
        let entered = AtomicBool::new(false);

        thread::scope(|s| {
            s.spawn(|| {
                lock.write_lock(0); // gated: waits for writers -> 0
                entered.store(true, ORD);
                lock.write_unlock();
            });

            thread::sleep(SETTLE);
            assert!(!entered.load(ORD), "writer must defer to the waiting reader");
            assert_eq!(lock.writers.load(ORD), 1, "must not queue up");
            assert_eq!(lock.ticket.load(ORD), 0, "must not take a ticket yet");

            lock.writers.store(0, ORD); // the in-flight writer drains
        });

        assert!(entered.load(ORD));
    }

    #[test]
    fn write_fairness_gate_is_open_just_below_the_batch_boundary() {
        let lock = RwLock::new();
        lock.mode.store(RwLockMode::WRITE, ORD);
        lock.writers.store(0, ORD);
        lock.readers.store(1, ORD);
        lock.ops_since_switch.store(RwLock::FAIRNESS_BATCH - 1, ORD);

        lock.write_lock(0); // must not block
        assert!(lock.is_write_locked());
        lock.write_unlock();
    }

    #[test]
    fn update_mode_resets_the_batch_on_a_phase_flip() {
        let lock = RwLock::new();
        lock.mode.store(RwLockMode::READ, ORD);
        lock.readers.store(0, ORD); // phase drained
        lock.ops_since_switch.store(500, ORD);

        let mut mode = RwLockMode::NONE;
        lock.update_mode(&mut mode);

        assert_eq!(lock.mode.load(ORD), RwLockMode::NONE, "flipped");
        assert_eq!(lock.ops_since_switch.load(ORD), 0, "batch restarted");
        // The C++ CAS leaves `expected` untouched on success, so the caller
        // still sees the OLD mode and spins once more. Pinned deliberately.
        assert_eq!(mode, RwLockMode::READ);
    }

    #[test]
    fn update_mode_does_not_flip_a_live_phase() {
        let lock = RwLock::new();
        lock.mode.store(RwLockMode::READ, ORD);
        lock.readers.store(2, ORD); // still live
        lock.ops_since_switch.store(7, ORD);

        let mut mode = RwLockMode::NONE;
        lock.update_mode(&mut mode);

        assert_eq!(lock.mode.load(ORD), RwLockMode::READ);
        assert_eq!(lock.ops_since_switch.load(ORD), 7, "batch untouched");
        assert_eq!(mode, RwLockMode::READ);
    }

    // -- overflow / wraparound edges -------------------------------------

    #[test]
    fn write_lock_boundary_at_u32_max_ticket() {
        // Divergence 6: u32::MAX is the LAST ticket that can ever be served,
        // because `cur_writer` is 32-bit while `ticket` is 64-bit. This pins
        // that the boundary itself still works.
        let lock = RwLock::new();
        lock.ticket.store(u64::from(u32::MAX), ORD);
        lock.cur_writer.store(u32::MAX, ORD);

        lock.write_lock(0); // tkt == u32::MAX == cur_writer -> acquires
        assert!(lock.is_write_locked());
        assert_eq!(lock.ticket.load(ORD), u64::from(u32::MAX) + 1);

        lock.write_unlock();
        // cur_writer wraps to 0 while ticket is now 2^32: from here the
        // comparison can never match again. Reproduced, not fixed.
        assert_eq!(lock.cur_writer.load(ORD), 0);
        assert_eq!(lock.ticket.load(ORD), 1u64 << 32);
    }

    #[test]
    fn unbalanced_read_unlock_wraps_rather_than_panicking() {
        // Divergence 7: C++ unsigned atomics wrap; Rust `fetch_sub` wraps in
        // debug too. Caller error either way, but it must not panic.
        let lock = RwLock::new();
        lock.read_unlock();
        assert_eq!(lock.readers.load(ORD), u32::MAX);
    }

    #[test]
    fn unbalanced_write_unlock_wraps_rather_than_panicking() {
        let lock = RwLock::new();
        lock.write_unlock();
        assert_eq!(lock.writers.load(ORD), u32::MAX);
        assert_eq!(lock.cur_writer.load(ORD), 1);
    }

    #[test]
    fn ops_since_switch_wraps_at_u32_max() {
        // Divergence 7: the batch counter wraps (it does not saturate), which
        // transiently re-opens the fairness gate. Documented, not fixed.
        let lock = RwLock::new();
        lock.ops_since_switch.store(u32::MAX, ORD);

        lock.read_lock(0);
        assert_eq!(lock.ops_since_switch.load(ORD), 0);
        lock.read_unlock();
    }

    #[test]
    fn unknown_mode_value_is_not_ub() {
        // Divergence 1: a peer process can store any i32 into `mode`. It must
        // simply match no arm; here the reader spins until we hand it a valid
        // mode, rather than tripping enum UB.
        let lock = RwLock::new();
        lock.mode.store(0x7fff_ffff, ORD);
        let entered = AtomicBool::new(false);

        thread::scope(|s| {
            s.spawn(|| {
                lock.read_lock(0);
                entered.store(true, ORD);
                lock.read_unlock();
            });

            thread::sleep(SETTLE);
            assert!(!entered.load(ORD), "unknown mode must not admit a reader");
            lock.mode.store(RwLockMode::READ, ORD); // recover
        });

        assert!(entered.load(ORD));
    }

    // -- scoped guards ----------------------------------------------------

    #[test]
    fn scoped_read_guard_locks_and_releases() {
        let lock = RwLock::new();
        {
            let _g = ScopedRwReadLock::new(&lock, 0);
            assert_eq!(lock.readers.load(ORD), 1);
        }
        assert_eq!(lock.readers.load(ORD), 0);
    }

    #[test]
    fn scoped_write_guard_locks_and_releases() {
        let lock = RwLock::new();
        {
            let _g = ScopedRwWriteLock::new(&lock, 0);
            assert!(lock.is_write_locked());
        }
        assert!(!lock.is_write_locked());
        assert_eq!(lock.cur_writer.load(ORD), 1);
    }

    #[test]
    fn scoped_read_guard_lock_unlock_are_idempotent() {
        let lock = RwLock::new();
        let mut g = ScopedRwReadLock::new(&lock, 0);
        assert_eq!(lock.readers.load(ORD), 1);

        g.lock(0); // already held: no-op, must not double-count
        assert_eq!(lock.readers.load(ORD), 1);

        g.unlock();
        assert_eq!(lock.readers.load(ORD), 0);
        g.unlock(); // not held: no-op, must not underflow
        assert_eq!(lock.readers.load(ORD), 0);

        drop(g); // dtor after explicit unlock: still exactly one release
        assert_eq!(lock.readers.load(ORD), 0);
    }

    #[test]
    fn scoped_write_guard_lock_unlock_are_idempotent() {
        let lock = RwLock::new();
        let mut g = ScopedRwWriteLock::new(&lock, 0);
        assert_eq!(lock.writers.load(ORD), 1);

        g.lock(0); // no-op: must not take a second ticket
        assert_eq!(lock.writers.load(ORD), 1);
        assert_eq!(lock.ticket.load(ORD), 1);

        g.unlock();
        assert!(!lock.is_write_locked());
        g.unlock(); // no-op
        assert_eq!(lock.writers.load(ORD), 0);
        assert_eq!(lock.cur_writer.load(ORD), 1);

        drop(g);
        assert_eq!(lock.cur_writer.load(ORD), 1);
    }

    #[test]
    fn scoped_guard_relock_after_unlock() {
        let lock = RwLock::new();
        let mut g = ScopedRwReadLock::new(&lock, 0);
        g.unlock();
        g.lock(0); // re-acquire through the same guard
        assert_eq!(lock.readers.load(ORD), 1);
        drop(g);
        assert_eq!(lock.readers.load(ORD), 0);
    }

    // -- concurrency ------------------------------------------------------

    /// Shared mutual-exclusion harness: writers must never overlap each other
    /// or a reader. Uses only atomics, so a violation is observed, not UB.
    struct Witness {
        in_write: AtomicBool,
        counter: AtomicU64,
        violations: AtomicU32,
    }

    impl Witness {
        const fn new() -> Self {
            Self {
                in_write: AtomicBool::new(false),
                counter: AtomicU64::new(0),
                violations: AtomicU32::new(0),
            }
        }

        /// Non-atomic-style read-modify-write: only correct under real
        /// exclusion, so a broken lock shows up as a lost update.
        fn write_critical_section(&self) {
            if self.in_write.swap(true, ORD) {
                self.violations.fetch_add(1, ORD); // another writer was inside
            }
            let v = self.counter.load(ORD);
            thread::yield_now(); // widen the window
            self.counter.store(v + 1, ORD);
            self.in_write.store(false, ORD);
        }

        fn read_critical_section(&self) {
            if self.in_write.load(ORD) {
                self.violations.fetch_add(1, ORD); // a writer was inside
            }
        }
    }

    #[test]
    fn concurrent_writers_are_mutually_exclusive() {
        const WRITERS: u64 = 4;
        const ITERS: u64 = 250;
        let lock = RwLock::new();
        let w = Witness::new();

        thread::scope(|s| {
            for _ in 0..WRITERS {
                s.spawn(|| {
                    for _ in 0..ITERS {
                        let _g = ScopedRwWriteLock::new(&lock, 0);
                        w.write_critical_section();
                    }
                });
            }
        });

        assert_eq!(w.violations.load(ORD), 0);
        assert_eq!(w.counter.load(ORD), WRITERS * ITERS, "lost update");
        assert_eq!(lock.writers.load(ORD), 0);
        assert_eq!(lock.ticket.load(ORD), WRITERS * ITERS);
        assert_eq!(u64::from(lock.cur_writer.load(ORD)), WRITERS * ITERS);
    }

    #[test]
    fn concurrent_readers_and_writers_never_overlap() {
        const WRITERS: u64 = 3;
        const READERS: u64 = 5;
        const ITERS: u64 = 200;
        let lock = RwLock::new();
        let w = Witness::new();

        thread::scope(|s| {
            for _ in 0..WRITERS {
                s.spawn(|| {
                    for _ in 0..ITERS {
                        let _g = ScopedRwWriteLock::new(&lock, 0);
                        w.write_critical_section();
                    }
                });
            }
            for _ in 0..READERS {
                s.spawn(|| {
                    for _ in 0..ITERS {
                        let _g = ScopedRwReadLock::new(&lock, 0);
                        w.read_critical_section();
                    }
                });
            }
        });

        assert_eq!(w.violations.load(ORD), 0, "reader/writer overlap");
        assert_eq!(w.counter.load(ORD), WRITERS * ITERS);
        assert_eq!(lock.readers.load(ORD), 0);
        assert_eq!(lock.writers.load(ORD), 0);
    }

    #[test]
    fn writers_make_progress_against_a_sustained_reader_stream() {
        // The batched-fairness bound is exactly what this exercises: with a
        // reader-preferring lock and no fairness batch, this hangs (the C++
        // bug lineage: an icx/Windows ctest timeout).
        const READERS: usize = 4;
        let lock = RwLock::new();
        let stop = AtomicBool::new(false);
        let writes = AtomicU64::new(0);

        thread::scope(|s| {
            for _ in 0..READERS {
                s.spawn(|| {
                    while !stop.load(ORD) {
                        let _g = ScopedRwReadLock::new(&lock, 0);
                    }
                });
            }
            let writer = s.spawn(|| {
                for _ in 0..50 {
                    let _g = ScopedRwWriteLock::new(&lock, 0);
                    writes.fetch_add(1, ORD);
                }
            });
            writer.join().unwrap(); // must terminate: no writer starvation
            stop.store(true, ORD);
        });

        assert_eq!(writes.load(ORD), 50);
    }

    #[test]
    fn concurrent_readers_hold_in_parallel() {
        const READERS: usize = 4;
        let lock = RwLock::new();
        let inside = AtomicU32::new(0);
        let max_seen = AtomicU32::new(0);

        thread::scope(|s| {
            for _ in 0..READERS {
                s.spawn(|| {
                    let _g = ScopedRwReadLock::new(&lock, 0);
                    let n = inside.fetch_add(1, ORD) + 1;
                    max_seen.fetch_max(n, ORD);
                    thread::sleep(Duration::from_millis(20));
                    inside.fetch_sub(1, ORD);
                });
            }
        });

        assert!(
            max_seen.load(ORD) > 1,
            "readers must be able to hold concurrently"
        );
        assert_eq!(lock.readers.load(ORD), 0);
    }

    // -- CvRwLock ---------------------------------------------------------

    #[test]
    fn cv_new_and_default_are_unheld() {
        for lock in [CvRwLock::new(), CvRwLock::default()] {
            lock.init(); // no-op, for parity
            assert_eq!(lock.readers.load(ORD), 0);
            assert_eq!(lock.write_intent.load(ORD), 0);
            assert!(!*lock.lock_state());
        }
    }

    #[test]
    fn cv_read_lock_is_lock_free_and_reentrant() {
        let lock = CvRwLock::new();
        lock.read_lock();
        lock.read_lock();
        assert_eq!(lock.readers.load(ORD), 2);
        assert_eq!(lock.write_intent.load(ORD), 0);
        lock.read_unlock();
        lock.read_unlock();
        assert_eq!(lock.readers.load(ORD), 0);
    }

    #[test]
    fn cv_write_lock_sets_intent_and_active() {
        let lock = CvRwLock::new();
        lock.write_lock();
        assert_eq!(lock.write_intent.load(ORD), 1);
        assert!(*lock.lock_state());

        lock.write_unlock();
        assert_eq!(lock.write_intent.load(ORD), 0);
        assert!(!*lock.lock_state());
    }

    #[test]
    fn cv_writer_waits_for_readers_and_blocks_new_readers() {
        let lock = CvRwLock::new();
        let writer_entered = AtomicBool::new(false);
        let reader_entered = AtomicBool::new(false);

        lock.read_lock(); // main thread holds a read hold

        thread::scope(|s| {
            s.spawn(|| {
                lock.write_lock();
                writer_entered.store(true, ORD);
                thread::sleep(Duration::from_millis(20));
                lock.write_unlock();
            });

            thread::sleep(SETTLE);
            assert!(
                !writer_entered.load(ORD),
                "writer must wait for readers to drain"
            );

            s.spawn(|| {
                lock.read_lock();
                reader_entered.store(true, ORD);
                lock.read_unlock();
            });

            thread::sleep(SETTLE);
            assert!(
                !reader_entered.load(ORD),
                "writer preference: a new reader must back off"
            );

            lock.read_unlock(); // release -> writer runs, then the reader
        });

        assert!(writer_entered.load(ORD));
        assert!(reader_entered.load(ORD));
        assert_eq!(lock.readers.load(ORD), 0);
        assert_eq!(lock.write_intent.load(ORD), 0);
    }

    #[test]
    fn cv_concurrent_readers_and_writers_never_overlap() {
        const WRITERS: u64 = 3;
        const READERS: u64 = 5;
        const ITERS: u64 = 200;
        let lock = CvRwLock::new();
        let w = Witness::new();

        thread::scope(|s| {
            for _ in 0..WRITERS {
                s.spawn(|| {
                    for _ in 0..ITERS {
                        lock.write_lock();
                        w.write_critical_section();
                        lock.write_unlock();
                    }
                });
            }
            for _ in 0..READERS {
                s.spawn(|| {
                    for _ in 0..ITERS {
                        lock.read_lock();
                        w.read_critical_section();
                        lock.read_unlock();
                    }
                });
            }
        });

        assert_eq!(w.violations.load(ORD), 0, "reader/writer overlap");
        assert_eq!(w.counter.load(ORD), WRITERS * ITERS, "lost update");
        assert_eq!(lock.readers.load(ORD), 0);
        assert_eq!(lock.write_intent.load(ORD), 0);
    }

    #[test]
    fn cv_readers_hold_in_parallel() {
        const READERS: usize = 4;
        let lock = CvRwLock::new();
        let inside = AtomicI64::new(0);
        let max_seen = AtomicI64::new(0);

        thread::scope(|s| {
            for _ in 0..READERS {
                s.spawn(|| {
                    lock.read_lock();
                    let n = inside.fetch_add(1, ORD) + 1;
                    max_seen.fetch_max(n, ORD);
                    thread::sleep(Duration::from_millis(20));
                    inside.fetch_sub(1, ORD);
                    lock.read_unlock();
                });
            }
        });

        assert!(
            max_seen.load(ORD) > 1,
            "the reader path must not serialize on the mutex"
        );
    }

    #[test]
    fn cv_survives_a_poisoned_mutex() {
        // Divergence 8: C++ has no poisoning, so a panic elsewhere must not
        // turn later acquisitions into panics.
        let lock = CvRwLock::new();
        let caught = thread::scope(|s| {
            s.spawn(|| {
                let _state = lock.lock_state();
                panic!("poison the state mutex");
            })
            .join()
        });
        assert!(caught.is_err());

        lock.write_lock(); // must not panic
        lock.write_unlock();
        lock.read_lock();
        lock.read_unlock();
        assert_eq!(lock.readers.load(ORD), 0);
    }
}
