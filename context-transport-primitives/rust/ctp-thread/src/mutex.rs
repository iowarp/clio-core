// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).

//! `ctp::Mutex` — the shared-memory-capable **fair ticket lock**.
//!
//! Port of `include/clio_ctp/thread/lock/mutex.h`. This is the lock embedded in
//! SHM-resident state (`BlobInfo::prealloc_lock_`, `MpAllocator::lock_`,
//! `BuddyAllocator::lock_`, `RingBuffer::pop_lock_`, `vector<ctp::Mutex>` in
//! `unordered_map_lhash`), so it is deliberately **not** `std::sync::Mutex`:
//! std's mutex is not usable across process boundaries and carries no
//! guaranteed layout. This type is `#[repr(C)]`, position-independent (pure
//! in-segment atomics, no pointers, no OS handles) and has no `Drop`, per
//! MEMORY_DESIGN.md Pillars 2 and 3.
//!
//! # Algorithm
//!
//! A ticket lock. `ticket` dispenses monotonically increasing ticket numbers;
//! `head` is the ticket currently being served. `lock()` takes a ticket and
//! spins until `head` reaches it; `unlock()` bumps `head`. Acquisition is
//! therefore **FIFO-fair**: waiters are served in arrival order and cannot be
//! barged. All operations are `SeqCst`, matching the C++ (whose `fetch_add` /
//! `load` / `load_device` all default to `memory_order_seq_cst` — on the host
//! `std_atomic::load_device()` is verbatim `x.load(std::memory_order_seq_cst)`).
//!
//! # C++ → Rust name mapping
//!
//! | C++ (`ctp::`) | Rust | Notes |
//! |---|---|---|
//! | `struct Mutex` | [`Mutex`] | `#[repr(C)]`, same 24-byte layout |
//! | `Mutex::lock_` (`ipc::atomic<min_u64>`) | `Mutex::ticket` (`AtomicU64`) | offset 0; ticket dispenser. Renamed: `lock_` names a *counter*, not the lock |
//! | `Mutex::head_` (`ipc::atomic<min_u64>`) | `Mutex::head` (`AtomicU64`) | offset 8; ticket now being served |
//! | `Mutex::try_lock_` (`ipc::atomic<min_u32>`) | `Mutex::try_guard` (`AtomicU32`) | offset 16; `TryLock` entry guard |
//! | `Mutex()` (default ctor) | [`Mutex::new`] / [`Mutex::default`] | `const fn`; zeroes all three fields |
//! | `Mutex(const Mutex&)` (copy ctor) | `impl Clone for Mutex` | both produce a *fresh unlocked* mutex; state is not copied |
//! | `void Init()` | [`Mutex::init`] | takes `&self` (atomics = interior mutability) |
//! | `void Lock(u32 owner)` | [`Mutex::lock`] | `owner` accepted and ignored, as in C++ |
//! | `bool TryLock(u32 owner)` | [`Mutex::try_lock`] | see divergence 2 |
//! | `void Unlock()` | [`Mutex::unlock`] | |
//! | `void Backoff(u32, min_u64)` | [`Mutex::backoff`] | associated fn (the C++ member touches no state) |
//! | `struct ScopedMutex` | [`ScopedMutex`] | RAII guard; `Drop` = C++ dtor |
//! | `ScopedMutex(Mutex&, u32)` | [`ScopedMutex::new`] | acquires on construction |
//! | `ScopedMutex::lock_` (`Mutex&`) | `ScopedMutex::lock` (`&'a Mutex`) | lifetime replaces the raw C++ reference |
//! | `ScopedMutex::is_locked_` (public field) | [`ScopedMutex::is_locked`] | private field + accessor |
//! | `ScopedMutex::{Lock,TryLock,Unlock}` | [`ScopedMutex::lock`] / [`try_lock`](ScopedMutex::try_lock) / [`unlock`](ScopedMutex::unlock) | `&mut self` |
//! | `ctp::ipc::Mutex`, `ctp::ipc::ScopedMutex` (aliases) | — | Rust has one path; no `ipc` re-export namespace |
//! | — | [`Mutex::scoped_lock`] | **added** sugar for `ScopedMutex::new(&m, owner)` |
//!
//! # Semantic divergences from the C++
//!
//! 1. **Thread-model dispatch is not wired.** C++ `Backoff` calls
//!    `CTP_THREAD_MODEL->Yield()` and consults `GetType()`, sleeping only for
//!    `kPthread`/`kStdThread` and staying purely cooperative for user-level
//!    threads (Argobots), where sleeping an execution stream would block the
//!    sibling ULT that may hold the lock. `crate::thread_model` is a parallel
//!    port whose API does not exist yet, and I may not add a dependency, so
//!    this file uses `std::thread::yield_now()` + `std::thread::sleep`
//!    unconditionally. For a pure-Rust caller this is *behaviour-preserving*:
//!    Rust threads are OS threads, i.e. exactly the `kStdThread` branch the C++
//!    would select. It becomes wrong only if this lock is ever waited on from a
//!    ULT/coroutine context; wiring `backoff` to the thread model is the
//!    follow-up.
//! 2. **`try_lock` reproduces an upstream defect (`try_guard` is leaked).**
//!    C++ `TryLock` increments `try_lock_` on entry but decrements it **only on
//!    the failure path**; neither `Unlock()` nor `Init()` ever clears it. So a
//!    `TryLock` that *succeeds* leaves `try_lock_ == 1` forever, and every
//!    later `TryLock` on that object returns `false` unconditionally — one
//!    successful `TryLock` permanently poisons `TryLock` for that mutex. This
//!    is ported **verbatim** rather than silently fixed: `Mutex` is a
//!    cross-language SHM type, so the Rust and C++ sides must agree on both the
//!    bytes and the state machine. The behaviour is pinned by
//!    `try_lock_succeeds_once_then_is_permanently_poisoned_cpp_quirk` and
//!    `init_does_not_reset_try_guard_cpp_quirk`, so fixing it upstream fails
//!    these tests loudly. Impact today is nil: `TryLock` has **no callers
//!    anywhere** in the C++ tree outside `lock/`, which is why the bug is
//!    latent. See the issue #756 report.
//! 3. **`try_lock` is racy by construction (upstream, preserved).** Even absent
//!    (2), the `ticket > head` test and the subsequent `Lock()` are not atomic:
//!    another thread may take a ticket in between, so a "successful"
//!    `try_lock` can still block. Faithfully ported.
//! 4. **Wraparound is explicit.** C++ `tkt - cur` and `++spin_count` are
//!    unsigned and wrap by definition. Plain Rust `-`/`+=` would *panic* in
//!    debug builds, so `wrapping_sub`/`wrapping_add` preserve C++ semantics
//!    exactly (see `ahead_matches_cpp_unsigned_wraparound` and
//!    `ticket_and_head_wrap_around_u64_max`).
//! 5. **No `ShmSafe` impl.** MEMORY_DESIGN.md asks SHM types to carry the
//!    `ShmSafe` marker, but (a) `ctp-thread` does not depend on `ctp-memory`
//!    and I may not edit `Cargo.toml`, and (b) `ShmSafe` is declared
//!    `unsafe trait ShmSafe: Copy + 'static`, which **no lock can satisfy** —
//!    atomics are not `Copy`, and a `Copy` mutex would be nonsense anyway. The
//!    marker needs an atomic-friendly relaxation before any in-segment lock can
//!    implement it, even though Pillar 3 mandates exactly such in-segment
//!    atomics. Reported as a cross-module design gap. `Mutex` is otherwise
//!    fully SHM-shaped: `repr(C)`, no `Drop`, no pointers, layout asserted by
//!    test.
//! 6. **GPU paths dropped.** The `CTP_IS_GPU` stuck-detector (`printf` every
//!    5,000,000 spins) and the device-pass busy-spin variant have no meaning in
//!    this host-only crate. `load_device()` collapses to `load(SeqCst)`, which
//!    is what it already is on the host.
//! 7. **Timings.** Per the ms convention: this API takes **no** timing
//!    argument. The one internal sleep stays in **microseconds** because the
//!    C++ says so explicitly (`std::chrono::microseconds`); `ahead` is a queue
//!    position, not a duration, that the C++ reinterprets as µs.
//! 8. **No poisoning, no data payload.** `ctp::Mutex` guards *adjacent* state
//!    rather than owning a `T`, so the guard intentionally grants no data
//!    access (unlike `std::sync::Mutex<T>`). A panic while holding it leaks the
//!    lock; a process crash while holding it is recovered by daemon restart,
//!    exactly as MEMORY_DESIGN.md Pillar 3 describes for the C++.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::Duration;

/// Cooperative-yield iterations before [`Mutex::backoff`] starts descheduling
/// the thread. Mirrors the C++ `spin_count < 64` fast path.
const COOPERATIVE_SPINS: u32 = 64;

/// Upper bound (µs) on a single backoff sleep, so worst-case handoff latency
/// stays bounded. Mirrors the C++ `ahead < 100 ? ahead : 100`.
const MAX_BACKOFF_MICROS: u64 = 100;

/// Sleep duration (µs) for a waiter `ahead` positions behind the head.
///
/// Mirrors the C++ exactly: scale with queue position (we provably cannot
/// acquire until `ahead` `Unlock()`s happen), clamp at [`MAX_BACKOFF_MICROS`],
/// and never sleep zero.
#[inline]
const fn backoff_micros(ahead: u64) -> u64 {
    let us = if ahead < MAX_BACKOFF_MICROS {
        ahead
    } else {
        MAX_BACKOFF_MICROS
    };
    if us == 0 {
        1
    } else {
        us
    }
}

/// Queue distance from `cur` to `tkt`, with C++ unsigned wraparound.
///
/// C++ computes `tkt - cur` on `min_u64`; unsigned overflow is defined to wrap,
/// whereas Rust's `-` panics in debug. See divergence 4.
#[inline]
const fn ahead_of(tkt: u64, cur: u64) -> u64 {
    tkt.wrapping_sub(cur)
}

/// A fair (FIFO) ticket lock that works inside a shared-memory segment.
///
/// Layout is frozen and matches the C++ `ctp::Mutex` byte for byte: three
/// atomics at offsets 0, 8, 16 with `size == 24`, `align == 8`. Contains no
/// pointers and no OS handles, so it is correct at any mapping base and may be
/// shared across processes.
///
/// `Send`/`Sync` are derived, not asserted: the type is nothing but atomics.
///
/// # Example
///
/// ```
/// use ctp_thread::mutex::Mutex;
///
/// let m = Mutex::new();
/// {
///     let _guard = m.scoped_lock(0); // released on drop
/// }
/// m.lock(0);
/// m.unlock();
/// ```
#[derive(Debug)]
#[repr(C)]
pub struct Mutex {
    /// C++ `lock_`: monotonically increasing ticket dispenser.
    ticket: AtomicU64,
    /// C++ `head_`: the ticket currently being served.
    head: AtomicU64,
    /// C++ `try_lock_`: `TryLock` entry guard. See divergence 2.
    try_guard: AtomicU32,
}

impl Mutex {
    /// Creates an unlocked mutex with all counters zeroed.
    ///
    /// `const`, so a `Mutex` can live in a `static` or in a const-initialized
    /// SHM header.
    #[inline]
    pub const fn new() -> Self {
        Self {
            ticket: AtomicU64::new(0),
            head: AtomicU64::new(0),
            try_guard: AtomicU32::new(0),
        }
    }

    /// Explicit (re)initialization for a mutex living in raw shared memory,
    /// mirroring C++ `Init()`.
    ///
    /// Takes `&self` because atomics provide interior mutability and callers
    /// hold a shared reference into a mapped segment.
    ///
    /// Resets `ticket` and `head` but **not** `try_guard` — faithful to the
    /// C++, which omits it. Combined with divergence 2, this means `init()`
    /// cannot rescue a mutex whose `try_lock` has been poisoned.
    ///
    /// Only meaningful when no thread holds or is waiting on the lock;
    /// resetting under contention strands waiters, exactly as in C++.
    #[inline]
    pub fn init(&self) {
        self.ticket.store(0, Ordering::SeqCst);
        self.head.store(0, Ordering::SeqCst);
    }

    /// Acquires the lock, blocking until it is held.
    ///
    /// `owner` is accepted for signature parity and ignored — the C++ ignores
    /// it too (it is vestigial; the lock records no owner).
    #[inline]
    pub fn lock(&self, owner: u32) {
        let _ = owner;
        let tkt = self.ticket.fetch_add(1, Ordering::SeqCst);
        let mut spin_count: u32 = 0;
        loop {
            let cur = self.head.load(Ordering::SeqCst);
            if tkt == cur {
                return;
            }
            // C++ `++spin_count` on a u32 wraps rather than panicking.
            spin_count = spin_count.wrapping_add(1);
            Self::backoff(spin_count, ahead_of(tkt, cur));
        }
    }

    /// Adaptive contention backoff for the ticket-wait path.
    ///
    /// A short cooperative-yield phase keeps handoff latency low in the common
    /// lightly-contended case; under sustained contention the OS thread is
    /// additionally descheduled so a preempted lock *holder* can run.
    ///
    /// This matters: the C++ comment records issue #483, where a far-back
    /// waiter busy-spinning on a bare PAUSE starved the holder on macOS — whose
    /// scheduler will not preempt a PAUSE-spinning thread — livelocking the
    /// lock for seconds. The uncontended path never reaches here ([`lock`]
    /// returns on its first iteration), so this costs the fast path nothing.
    ///
    /// Associated rather than a method: the C++ member reads no state.
    ///
    /// [`lock`]: Self::lock
    #[inline]
    pub fn backoff(spin_count: u32, ahead: u64) {
        // Cooperative yield first: cheap, preserves handoff latency.
        std::thread::yield_now();
        if spin_count < COOPERATIVE_SPINS {
            return;
        }
        // Sustained contention: deschedule so the holder can make progress.
        // See divergence 1 re: thread-model dispatch.
        std::thread::sleep(Duration::from_micros(backoff_micros(ahead)));
    }

    /// Attempts to acquire without blocking; returns `true` if acquired.
    ///
    /// # Upstream defect, ported verbatim
    ///
    /// A **successful** `try_lock` permanently poisons `try_lock` on this
    /// mutex: `try_guard` is incremented on entry and never decremented on the
    /// success path, so every subsequent call returns `false` forever, and
    /// [`init`](Self::init) does not clear it. A *failed* `try_lock` is
    /// harmless (it restores `try_guard`). Ported as-is for byte/state parity
    /// with the C++; see divergence 2. Prefer [`lock`](Self::lock).
    ///
    /// Also inherently racy: a `true` return may still have blocked, because
    /// the check and the acquire are not atomic (divergence 3).
    #[inline]
    pub fn try_lock(&self, owner: u32) -> bool {
        // `||` short-circuits in both languages: when the guard is already
        // taken, the ticket/head comparison is not evaluated.
        if self.try_guard.fetch_add(1, Ordering::SeqCst) > 0
            || self.ticket.load(Ordering::SeqCst) > self.head.load(Ordering::SeqCst)
        {
            self.try_guard.fetch_sub(1, Ordering::SeqCst);
            return false;
        }
        self.lock(owner);
        true
    }

    /// Releases the lock by serving the next ticket.
    ///
    /// Calling this without holding the lock corrupts the lock (it hands the
    /// critical section to a waiter that was not granted it) — the same
    /// contract as the C++, which is why callers should prefer
    /// [`scoped_lock`](Self::scoped_lock).
    #[inline]
    pub fn unlock(&self) {
        self.head.fetch_add(1, Ordering::SeqCst);
    }

    /// Acquires the lock and returns an RAII guard that releases it on drop.
    ///
    /// Convenience for [`ScopedMutex::new`]; the C++ spells this
    /// `ScopedMutex guard(lock_, 0);`.
    #[inline]
    pub fn scoped_lock(&self, owner: u32) -> ScopedMutex<'_> {
        ScopedMutex::new(self, owner)
    }
}

impl Default for Mutex {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Mirrors the C++ copy constructor `Mutex(const Mutex &other) {}`, which
/// ignores `other` and default-constructs its members: cloning yields a
/// **fresh unlocked mutex**, never a copy of lock state. (The C++ ctor exists
/// so `vector<ctp::Mutex>` can grow.)
impl Clone for Mutex {
    #[inline]
    fn clone(&self) -> Self {
        Self::new()
    }
}

/// RAII guard for [`Mutex`], mirroring C++ `ctp::ScopedMutex`.
///
/// Acquires on construction and releases on drop. Unlike `std::sync::MutexGuard`
/// it grants no data access: `ctp::Mutex` guards adjacent state rather than
/// owning a payload (divergence 8).
///
/// The guard tracks whether it currently holds the lock, so an explicit
/// [`unlock`](Self::unlock) followed by a drop releases exactly once.
pub struct ScopedMutex<'a> {
    lock: &'a Mutex,
    is_locked: bool,
}

impl<'a> ScopedMutex<'a> {
    /// Acquires `lock` and wraps it in a guard (C++ `ScopedMutex(Mutex&, u32)`).
    #[inline]
    pub fn new(lock: &'a Mutex, owner: u32) -> Self {
        let mut guard = Self {
            lock,
            is_locked: false,
        };
        guard.lock(owner);
        guard
    }

    /// Acquires the lock if this guard does not already hold it (idempotent).
    #[inline]
    pub fn lock(&mut self, owner: u32) {
        if !self.is_locked {
            self.lock.lock(owner);
            self.is_locked = true;
        }
    }

    /// Tries to acquire if not already held; returns whether the guard now
    /// holds the lock.
    ///
    /// Inherits the [`Mutex::try_lock`] defect (divergence 2). Note this is
    /// effectively unreachable in the C++ too: the only constructor already
    /// acquires, so the guard always arrives locked.
    #[inline]
    pub fn try_lock(&mut self, owner: u32) -> bool {
        if !self.is_locked {
            self.is_locked = self.lock.try_lock(owner);
        }
        self.is_locked
    }

    /// Releases the lock early if held; idempotent.
    #[inline]
    pub fn unlock(&mut self) {
        if self.is_locked {
            self.lock.unlock();
            self.is_locked = false;
        }
    }

    /// Whether this guard currently holds the lock (C++ public `is_locked_`).
    #[inline]
    pub fn is_locked(&self) -> bool {
        self.is_locked
    }
}

impl Drop for ScopedMutex<'_> {
    /// C++ `~ScopedMutex() { Unlock(); }` — `unlock` is guarded by
    /// `is_locked`, so this never double-unlocks.
    #[inline]
    fn drop(&mut self) {
        self.unlock();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // ABI / shared-memory shape
    // -----------------------------------------------------------------------

    /// The C++ `ctp::Mutex` is `{atomic<min_u64>, atomic<min_u64>,
    /// atomic<min_u32>}`; `ipc::atomic<T>` is `std_atomic<T>` == a bare
    /// `std::atomic<T>`, so `sizeof == 24` / `alignof == 8` with 4 bytes of
    /// tail padding. Rust must agree byte for byte or SHM interop breaks.
    #[test]
    fn layout_matches_cpp_abi() {
        assert_eq!(std::mem::size_of::<Mutex>(), 24);
        assert_eq!(std::mem::align_of::<Mutex>(), 8);
        assert_eq!(std::mem::offset_of!(Mutex, ticket), 0);
        assert_eq!(std::mem::offset_of!(Mutex, head), 8);
        assert_eq!(std::mem::offset_of!(Mutex, try_guard), 16);
    }

    /// MEMORY_DESIGN.md: SHM-resident types must have no `Drop`.
    #[test]
    fn mutex_has_no_drop_glue() {
        assert!(!std::mem::needs_drop::<Mutex>());
    }

    /// The lock must be shareable across threads/processes by reference.
    #[test]
    fn mutex_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Mutex>();
    }

    #[test]
    fn new_is_zeroed_and_unlocked() {
        let m = Mutex::new();
        assert_eq!(m.ticket.load(Ordering::SeqCst), 0);
        assert_eq!(m.head.load(Ordering::SeqCst), 0);
        assert_eq!(m.try_guard.load(Ordering::SeqCst), 0);
        // An uncontended acquire returns immediately.
        m.lock(0);
        m.unlock();
    }

    #[test]
    fn default_matches_new() {
        let m = Mutex::default();
        assert_eq!(m.ticket.load(Ordering::SeqCst), 0);
        assert_eq!(m.head.load(Ordering::SeqCst), 0);
        assert_eq!(m.try_guard.load(Ordering::SeqCst), 0);
    }

    // -----------------------------------------------------------------------
    // Core lock/unlock behaviour
    // -----------------------------------------------------------------------

    #[test]
    fn lock_then_unlock_is_repeatable() {
        let m = Mutex::new();
        for i in 0..100u64 {
            m.lock(0);
            assert_eq!(m.ticket.load(Ordering::SeqCst), i + 1);
            assert_eq!(m.head.load(Ordering::SeqCst), i);
            m.unlock();
            assert_eq!(m.head.load(Ordering::SeqCst), i + 1);
        }
    }

    #[test]
    fn unlock_advances_head_exactly_once() {
        let m = Mutex::new();
        m.lock(0);
        m.unlock();
        assert_eq!(m.head.load(Ordering::SeqCst), 1);
        assert_eq!(m.ticket.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn owner_argument_is_ignored() {
        // C++ takes `u32 owner` and never reads it; differing owners must not
        // change observable state.
        let a = Mutex::new();
        a.lock(0);
        a.unlock();
        let b = Mutex::new();
        b.lock(u32::MAX);
        b.unlock();
        assert_eq!(
            a.ticket.load(Ordering::SeqCst),
            b.ticket.load(Ordering::SeqCst)
        );
        assert_eq!(a.head.load(Ordering::SeqCst), b.head.load(Ordering::SeqCst));
    }

    #[test]
    fn init_resets_ticket_and_head() {
        let m = Mutex::new();
        m.lock(0);
        m.unlock();
        m.lock(0);
        m.unlock();
        assert_eq!(m.head.load(Ordering::SeqCst), 2);
        m.init();
        assert_eq!(m.ticket.load(Ordering::SeqCst), 0);
        assert_eq!(m.head.load(Ordering::SeqCst), 0);
        // Usable again after re-init.
        m.lock(0);
        m.unlock();
    }

    /// Pins C++ quirk: `Init()` resets `lock_`/`head_` but never `try_lock_`,
    /// so it cannot un-poison `TryLock`. See divergence 2.
    #[test]
    fn init_does_not_reset_try_guard_cpp_quirk() {
        let m = Mutex::new();
        assert!(m.try_lock(0));
        m.unlock();
        assert_eq!(m.try_guard.load(Ordering::SeqCst), 1);
        m.init();
        assert_eq!(
            m.try_guard.load(Ordering::SeqCst),
            1,
            "C++ Init() does not touch try_lock_; the port must not either"
        );
        assert!(
            !m.try_lock(0),
            "try_lock stays poisoned across init(), exactly as in C++"
        );
    }

    // -----------------------------------------------------------------------
    // try_lock — including the faithfully-ported upstream defect
    // -----------------------------------------------------------------------

    /// The headline quirk (divergence 2). If this test ever fails, the C++ was
    /// fixed and this port must be updated in lockstep.
    #[test]
    fn try_lock_succeeds_once_then_is_permanently_poisoned_cpp_quirk() {
        let m = Mutex::new();
        assert!(m.try_lock(0), "first try_lock on a fresh mutex succeeds");
        m.unlock();
        // try_guard was incremented on the success path and never decremented.
        assert_eq!(m.try_guard.load(Ordering::SeqCst), 1);
        // ...so every later try_lock fails, even though the mutex is FREE.
        assert_eq!(
            m.ticket.load(Ordering::SeqCst),
            m.head.load(Ordering::SeqCst)
        );
        for _ in 0..5 {
            assert!(
                !m.try_lock(0),
                "C++ TryLock never releases try_lock_ on success"
            );
        }
        // The blocking path is unaffected — which is why the bug is latent.
        m.lock(0);
        m.unlock();
    }

    #[test]
    fn try_lock_fails_while_lock_is_held() {
        let m = Mutex::new();
        m.lock(0);
        assert!(!m.try_lock(0), "ticket > head while held");
        m.unlock();
    }

    /// A *failed* try_lock must restore `try_guard` (the failure path does
    /// decrement), so failures are not sticky.
    #[test]
    fn failed_try_lock_restores_try_guard() {
        let m = Mutex::new();
        m.lock(0);
        for _ in 0..3 {
            assert!(!m.try_lock(0));
            assert_eq!(
                m.try_guard.load(Ordering::SeqCst),
                0,
                "failure path decrements try_lock_"
            );
        }
        m.unlock();
        // Still able to succeed once, since no success has poisoned it yet.
        assert!(m.try_lock(0));
        m.unlock();
    }

    // -----------------------------------------------------------------------
    // Backoff policy: boundaries and saturation
    // -----------------------------------------------------------------------

    #[test]
    fn backoff_micros_clamps_and_never_sleeps_zero() {
        // Never zero: C++ `if (us == 0) us = 1;`
        assert_eq!(backoff_micros(0), 1);
        assert_eq!(backoff_micros(1), 1);
        // Scales with queue position below the cap.
        assert_eq!(backoff_micros(50), 50);
        // Boundary: `ahead < 100 ? ahead : 100`, so 99 -> 99 and 100 -> 100.
        assert_eq!(backoff_micros(99), 99);
        assert_eq!(backoff_micros(100), 100);
        assert_eq!(backoff_micros(101), 100);
        // Saturates: a wrapped `ahead` must not sleep for eons.
        assert_eq!(backoff_micros(u64::MAX), MAX_BACKOFF_MICROS);
    }

    /// `ahead` comes from `tkt - cur` on unsigned C++ counters, which wraps.
    /// Rust's `-` would panic in debug; `wrapping_sub` preserves C++ semantics.
    #[test]
    fn ahead_matches_cpp_unsigned_wraparound() {
        assert_eq!(ahead_of(10, 3), 7);
        assert_eq!(ahead_of(5, 5), 0);
        // head has passed the ticket (or the counters wrapped): a huge value,
        // which backoff_micros then clamps rather than sleeping ~forever.
        assert_eq!(ahead_of(0, 1), u64::MAX);
        assert_eq!(ahead_of(3, u64::MAX), 4);
        assert_eq!(backoff_micros(ahead_of(0, 1)), MAX_BACKOFF_MICROS);
    }

    /// The cooperative fast path must not sleep; the sustained path may.
    #[test]
    fn backoff_below_threshold_is_cooperative_only() {
        let start = std::time::Instant::now();
        for i in 1..COOPERATIVE_SPINS {
            Mutex::backoff(i, 100);
        }
        // 63 yields must not take anywhere near 63 * 100us of sleeping.
        // Generous bound: this asserts "no sleep", not scheduler precision.
        assert!(
            start.elapsed() < Duration::from_millis(500),
            "spin_count < 64 must not sleep"
        );
    }

    #[test]
    fn backoff_at_threshold_sleeps() {
        let start = std::time::Instant::now();
        Mutex::backoff(COOPERATIVE_SPINS, 100);
        assert!(
            start.elapsed() >= Duration::from_micros(100),
            "spin_count >= 64 sleeps for min(ahead, 100)us"
        );
    }

    // -----------------------------------------------------------------------
    // Counter wraparound (end-to-end)
    // -----------------------------------------------------------------------

    /// Tickets are u64 and wrap; the lock stays correct across the boundary
    /// because only `tkt == head` equality is tested. C++ unsigned atomics wrap
    /// and Rust's `fetch_add` wraps too, so parity holds.
    #[test]
    fn ticket_and_head_wrap_around_u64_max() {
        let m = Mutex::new();
        m.ticket.store(u64::MAX, Ordering::SeqCst);
        m.head.store(u64::MAX, Ordering::SeqCst);
        // tkt == u64::MAX, head == u64::MAX -> acquire immediately.
        m.lock(0);
        assert_eq!(m.ticket.load(Ordering::SeqCst), 0, "dispenser wrapped");
        m.unlock();
        assert_eq!(m.head.load(Ordering::SeqCst), 0, "head wrapped");
        // And the post-wrap mutex still works.
        m.lock(0);
        assert_eq!(m.ticket.load(Ordering::SeqCst), 1);
        m.unlock();
        assert_eq!(m.head.load(Ordering::SeqCst), 1);
    }

    // -----------------------------------------------------------------------
    // ScopedMutex (RAII)
    // -----------------------------------------------------------------------

    #[test]
    fn scoped_mutex_locks_on_new_and_unlocks_on_drop() {
        let m = Mutex::new();
        {
            let guard = ScopedMutex::new(&m, 0);
            assert!(guard.is_locked());
            assert_eq!(m.head.load(Ordering::SeqCst), 0, "still held");
        }
        assert_eq!(m.head.load(Ordering::SeqCst), 1, "released on drop");
        // The lock is genuinely free again.
        m.lock(0);
        m.unlock();
    }

    #[test]
    fn scoped_lock_helper_matches_scoped_mutex_new() {
        let m = Mutex::new();
        {
            let guard = m.scoped_lock(0);
            assert!(guard.is_locked());
        }
        assert_eq!(m.head.load(Ordering::SeqCst), 1);
    }

    /// The `is_locked` flag exists precisely so this does not double-unlock —
    /// a double unlock would corrupt `head` and hand out the critical section.
    #[test]
    fn scoped_mutex_explicit_unlock_then_drop_releases_once() {
        let m = Mutex::new();
        {
            let mut guard = ScopedMutex::new(&m, 0);
            guard.unlock();
            assert!(!guard.is_locked());
            assert_eq!(m.head.load(Ordering::SeqCst), 1);
            // Dropping now must NOT advance head again.
        }
        assert_eq!(
            m.head.load(Ordering::SeqCst),
            1,
            "drop after explicit unlock must not double-unlock"
        );
        assert_eq!(m.ticket.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn scoped_mutex_unlock_is_idempotent() {
        let m = Mutex::new();
        let mut guard = ScopedMutex::new(&m, 0);
        guard.unlock();
        guard.unlock();
        guard.unlock();
        assert_eq!(m.head.load(Ordering::SeqCst), 1);
    }

    /// Re-locking through the guard after an explicit unlock takes a new ticket.
    #[test]
    fn scoped_mutex_relock_after_unlock() {
        let m = Mutex::new();
        {
            let mut guard = ScopedMutex::new(&m, 0);
            guard.unlock();
            guard.lock(0);
            assert!(guard.is_locked());
            assert_eq!(m.ticket.load(Ordering::SeqCst), 2);
            assert_eq!(m.head.load(Ordering::SeqCst), 1);
        }
        assert_eq!(m.head.load(Ordering::SeqCst), 2);
    }

    /// C++ `Lock()` on an already-locked guard is a no-op (guarded by
    /// `is_locked_`) — it must not take a second ticket and self-deadlock.
    #[test]
    fn scoped_mutex_lock_while_held_is_noop() {
        let m = Mutex::new();
        let mut guard = ScopedMutex::new(&m, 0);
        guard.lock(0);
        guard.lock(0);
        assert_eq!(
            m.ticket.load(Ordering::SeqCst),
            1,
            "no extra ticket taken; a second would deadlock"
        );
    }

    /// `try_lock` on a guard that already holds the lock short-circuits to
    /// `true` without touching `try_guard` (so it dodges the divergence-2 bug).
    #[test]
    fn scoped_mutex_try_lock_while_held_is_noop() {
        let m = Mutex::new();
        let mut guard = ScopedMutex::new(&m, 0);
        assert!(guard.try_lock(0));
        assert_eq!(m.try_guard.load(Ordering::SeqCst), 0, "try_lock_ untouched");
        assert_eq!(m.ticket.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn clone_yields_fresh_unlocked_mutex() {
        let m = Mutex::new();
        m.lock(0); // held: ticket=1, head=0
        let copy = m.clone();
        assert_eq!(copy.ticket.load(Ordering::SeqCst), 0);
        assert_eq!(copy.head.load(Ordering::SeqCst), 0);
        assert_eq!(copy.try_guard.load(Ordering::SeqCst), 0);
        // The clone is independent and immediately lockable.
        copy.lock(0);
        copy.unlock();
        m.unlock();
    }

    // -----------------------------------------------------------------------
    // Concurrency
    // -----------------------------------------------------------------------

    /// Mutual-exclusion proof with zero `unsafe`: a deliberately non-atomic
    /// read-modify-write (load then store) inside the critical section. Any
    /// overlap loses updates and the total comes up short.
    #[test]
    fn mutual_exclusion_under_contention() {
        const THREADS: usize = 8;
        const ITERS: usize = 500;

        let m = Mutex::new();
        let counter = AtomicU64::new(0);

        std::thread::scope(|s| {
            for t in 0..THREADS {
                let m = &m;
                let counter = &counter;
                s.spawn(move || {
                    for _ in 0..ITERS {
                        m.lock(t as u32);
                        // Non-atomic RMW: correct ONLY under real exclusion.
                        let v = counter.load(Ordering::Relaxed);
                        counter.store(v + 1, Ordering::Relaxed);
                        m.unlock();
                    }
                });
            }
        });

        assert_eq!(
            counter.load(Ordering::Relaxed),
            (THREADS * ITERS) as u64,
            "lost updates mean the lock let threads overlap"
        );
        assert_eq!(m.ticket.load(Ordering::SeqCst), (THREADS * ITERS) as u64);
        assert_eq!(m.head.load(Ordering::SeqCst), (THREADS * ITERS) as u64);
    }

    /// Directly observes that at most one thread is ever inside the critical
    /// section, rather than inferring it from a counter.
    #[test]
    fn never_two_threads_inside_critical_section() {
        const THREADS: usize = 8;
        const ITERS: usize = 200;

        let m = Mutex::new();
        let inside = AtomicU64::new(0);
        let violations = AtomicU64::new(0);

        std::thread::scope(|s| {
            for _ in 0..THREADS {
                let m = &m;
                let inside = &inside;
                let violations = &violations;
                s.spawn(move || {
                    for _ in 0..ITERS {
                        m.lock(0);
                        if inside.fetch_add(1, Ordering::SeqCst) != 0 {
                            violations.fetch_add(1, Ordering::SeqCst);
                        }
                        std::thread::yield_now();
                        inside.fetch_sub(1, Ordering::SeqCst);
                        m.unlock();
                    }
                });
            }
        });

        assert_eq!(violations.load(Ordering::SeqCst), 0);
        assert_eq!(inside.load(Ordering::SeqCst), 0);
    }

    /// The RAII guard must provide the same exclusion as manual lock/unlock.
    #[test]
    fn scoped_mutex_provides_exclusion_under_contention() {
        const THREADS: usize = 4;
        const ITERS: usize = 300;

        let m = Mutex::new();
        let counter = AtomicU64::new(0);

        std::thread::scope(|s| {
            for _ in 0..THREADS {
                let m = &m;
                let counter = &counter;
                s.spawn(move || {
                    for _ in 0..ITERS {
                        let _guard = m.scoped_lock(0);
                        let v = counter.load(Ordering::Relaxed);
                        counter.store(v + 1, Ordering::Relaxed);
                    }
                });
            }
        });

        assert_eq!(counter.load(Ordering::Relaxed), (THREADS * ITERS) as u64);
        assert_eq!(m.head.load(Ordering::SeqCst), (THREADS * ITERS) as u64);
    }

    /// A ticket lock is FIFO-fair: with N threads each looping, no thread can
    /// be starved. Assert every thread completes (a barging lock could let one
    /// starve indefinitely) and that tickets and head balance exactly.
    #[test]
    fn all_waiters_make_progress_no_starvation() {
        const THREADS: usize = 6;
        const ITERS: usize = 100;

        let m = Mutex::new();
        let per_thread: Vec<AtomicU64> = (0..THREADS).map(|_| AtomicU64::new(0)).collect();

        std::thread::scope(|s| {
            for slot in per_thread.iter() {
                let m = &m;
                s.spawn(move || {
                    for _ in 0..ITERS {
                        m.lock(0);
                        slot.fetch_add(1, Ordering::Relaxed);
                        m.unlock();
                    }
                });
            }
        });

        for (i, slot) in per_thread.iter().enumerate() {
            assert_eq!(
                slot.load(Ordering::Relaxed),
                ITERS as u64,
                "thread {i} was starved"
            );
        }
        assert_eq!(
            m.ticket.load(Ordering::SeqCst),
            m.head.load(Ordering::SeqCst)
        );
    }
}
