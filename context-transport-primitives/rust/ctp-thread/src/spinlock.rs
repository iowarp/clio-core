// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).

//! Cross-process ticket spin lock — port of
//! `clio_ctp/thread/lock/spin_lock.h` (`ctp::SpinLock` / `ctp::ScopedSpinLock`).
//!
//! Despite the name, the C++ `SpinLock` is **not** a test-and-set lock: it is a
//! FIFO **ticket lock**. `Lock()` takes a ticket via `lock_.fetch_add(1)` and
//! busy-waits until `head_` reaches it; `Unlock()` bumps `head_`, handing the
//! lock to the next ticket in arrival order. Fairness is therefore a property
//! of the algorithm, not an extra, and this port preserves it.
//!
//! All state is plain atomics inside a `#[repr(C)]`, `Drop`-free struct, so an
//! instance is usable directly inside a shared-memory segment mapped at
//! different base addresses in different processes (MEMORY_DESIGN.md pillar 3:
//! no absolute pointers, in-segment atomics only). An all-zero byte pattern is
//! a valid, unheld lock — which is exactly what the C++ `util/singleton.h`
//! relies on when it reinterprets a zeroed `char[sizeof(SpinLock)]` as a
//! `SpinLock` without ever running a constructor.
//!
//! # C++ → Rust name mapping
//!
//! | C++ (`ctp::` / `ctp::ipc::`)        | Rust (`ctp_thread::spinlock::`)          |
//! |-------------------------------------|------------------------------------------|
//! | `struct SpinLock`                   | [`SpinLock`]                             |
//! | `SpinLock::lock_` (`atomic<min_u64>`)   | `SpinLock::ticket` (`AtomicU64`, private) |
//! | `SpinLock::head_` (`atomic<min_u64>`)   | `SpinLock::head` (`AtomicU64`, private)   |
//! | `SpinLock::try_lock_` (`atomic<min_u32>`) | `SpinLock::try_gate` (`AtomicU32`, private) |
//! | `SpinLock()` (default ctor)         | [`SpinLock::new`] / [`SpinLock::default`] |
//! | `SpinLock(const SpinLock&)`         | *(none — see divergence 1)*              |
//! | `SpinLock::Init()`                  | [`SpinLock::init`] (see divergence 2)    |
//! | `SpinLock::Lock(u32 owner)`         | [`SpinLock::lock`]                       |
//! | `SpinLock::TryLock(u32 owner)`      | [`SpinLock::try_lock`] (see divergence 3) |
//! | `SpinLock::Unlock()`                | [`SpinLock::unlock`]                     |
//! | `struct ScopedSpinLock`             | [`ScopedSpinLock`]                       |
//! | `ScopedSpinLock(SpinLock&, u32)`    | [`ScopedSpinLock::new`] / [`SpinLock::acquire`] |
//! | `ScopedSpinLock::~ScopedSpinLock()` | `impl Drop for ScopedSpinLock`           |
//! | `ScopedSpinLock::Lock(u32)`         | [`ScopedSpinLock::lock`]                 |
//! | `ScopedSpinLock::TryLock(u32)`      | [`ScopedSpinLock::try_lock`]             |
//! | `ScopedSpinLock::Unlock()`          | [`ScopedSpinLock::unlock`]               |
//! | `ScopedSpinLock::is_locked_`        | [`ScopedSpinLock::is_locked`] (accessor) |
//! | `ctp::ipc::SpinLock` (alias)        | same type — Rust has no second namespace |
//!
//! `min_u64`/`min_u32` resolve to `u64`/`u32` on the host pass
//! (`types/numbers.h`), which is what this port targets.
//!
//! # Semantic divergences from the C++
//!
//! 1. **No copy constructor.** C++ `SpinLock(const SpinLock &other) {}` has an
//!    empty body, so its `std_atomic` members are *default-initialized* — i.e.
//!    `std::atomic<u64> x;` with **indeterminate value**. Copy-constructing a
//!    `SpinLock` (which `priv::vector<SpinLock>` reallocation would do) yields a
//!    lock holding garbage tickets: latent UB. The evident *intent* — the
//!    sibling comment in `rwlock.h` spells it out — is "construct a fresh,
//!    unheld lock". Rust therefore implements neither `Clone` nor `Copy`
//!    (silently cloning a live lock is never what a caller wants); callers that
//!    want the intended behavior write [`SpinLock::new`], which is the
//!    well-defined all-zero state. **This diverges deliberately: the C++
//!    behavior is unreproducible because it is UB.**
//!
//! 2. **`init()` resets all three fields; C++ `Init()` resets only `lock_`.**
//!    C++ leaves `head_` and `try_lock_` untouched, so calling `Init()` on a
//!    lock that has ever been used leaves `lock_ == 0` while `head_ == N`,
//!    permanently deadlocking every subsequent `Lock()` (ticket 0 waits for a
//!    `head_` that already passed 0). The sibling `Mutex::Init()` resets both
//!    `lock_` and `head_`, confirming the omission is an oversight. Every
//!    reachable C++ call site runs `Init()` on fresh/zeroed memory, where
//!    "reset `lock_`" and "reset all" are indistinguishable — so this port is
//!    a strict superset that agrees with the C++ on all non-broken inputs.
//!
//! 3. **`try_lock()` releases the try-gate on success; C++ `TryLock()` leaks
//!    it.** The C++ increments `try_lock_` on entry and decrements it only on
//!    the *failure* path:
//!    ```cpp
//!    bool TryLock(u32 owner) {
//!      if (try_lock_.fetch_add(1) > 0 || lock_.load() > head_.load()) {
//!        try_lock_.fetch_sub(1);
//!        return false;
//!      }
//!      Lock(owner);
//!      return true;          // <-- try_lock_ stays 1 forever
//!    }
//!    ```
//!    `Unlock()` only touches `head_`, so after the *first* successful
//!    `TryLock()` the gate is stuck at 1 and **every later `TryLock()` on that
//!    lock returns false forever**, even when the lock is free. This port adds
//!    the missing `try_gate.fetch_sub(1)` on the success path, making the gate
//!    what it plainly means to be: a short critical section serializing the
//!    "observe free, then take a ticket" sequence between concurrent
//!    try-lockers. `spin_lock.h`'s `TryLock` has no in-tree callers or tests
//!    (the only consumer, `util/singleton.h`, uses `ScopedSpinLock`'s blocking
//!    path), so nothing depends on bug-compatibility here.
//!    `test_try_lock_is_repeatable` pins the fixed behavior. Verified against a
//!    verbatim transcription of the C++: attempt 1 returns true, and attempts 2
//!    and 3 on the now-free lock both return false.
//!
//!    **The identical defect exists in `ctp::Mutex::TryLock` (`lock/mutex.h`),
//!    where it is not latent but live:** `clio_runtime`'s
//!    `CoMutex::TryLock()` (`comutex.h:75`) calls `lock_.TryLock(0)` on a
//!    `ctp::Mutex`, so after the first successful non-reentrant `TryLock` on a
//!    given `CoMutex`, every later `TryLock()` on it returns false forever even
//!    when the mutex is free. Callers that retry in a loop will spin
//!    indefinitely. That is a C++-side fix outside this file's scope (`mutex.rs`
//!    is a separate module); it is reported to the migration owner.
//!
//! 4. **Ticket wraparound in `try_lock` is left faithful (and still latently
//!    wrong).** `lock()`/`unlock()` wrap correctly: `fetch_add` wraps in Rust
//!    exactly as unsigned overflow does in C++, and `lock()` compares tickets
//!    with `==`. But `try_lock`'s `ticket > head` "is it free?" test is a
//!    magnitude comparison, so in the single window where `ticket` has wrapped
//!    past `u64::MAX` and `head` has not, a held lock reads as free and
//!    `try_lock` falls into a **blocking** `lock()`. A wrap-safe `ticket !=
//!    head` would fix it, but the window needs 2^64 acquisitions to reach, so
//!    unlike divergence 3 (broken on the *first* call) it is documented rather
//!    than changed. `test_ticket_wraparound_at_u64_max` pins the wrap behavior
//!    of the `lock`/`unlock` path.
//!
//! 5. **`owner` is accepted and ignored, as in C++.** Every C++ lock entry
//!    point takes a `u32 owner` that the body never reads (a vestige of an
//!    ownership-tracking design); the only call site passes `0`. The parameter
//!    is kept for signature parity and to leave room for the deadlock-detection
//!    use it was reserved for. Pass [`UNUSED_OWNER`] when you have no id.
//!
//! 6. **No `ShmSafe` marker.** MEMORY_DESIGN.md asks shared-memory types to
//!    carry `ctp-memory`'s `ShmSafe` marker, but `ctp-thread/Cargo.toml` does
//!    not depend on `ctp-memory` and this port may not edit it. The type
//!    independently satisfies every structural requirement the marker attests
//!    (`repr(C)`, no `Drop`, no references, `'static`, valid when zeroed);
//!    `impl ShmSafe for SpinLock` should be added when the crates are wired
//!    together.
//!
//! 7. **`spin_loop()` hint added to the wait loop.** C++ `Lock()` spins in a
//!    bare `do { for (int i = 0; i < 1; ++i) { ... } } while (true)` — the inner
//!    loop iterates exactly once and is a no-op that the optimizer deletes.
//!    Rust emits a `PAUSE` (x86) / `YIELD` (aarch64) hint per iteration, which
//!    changes no observable semantics but cuts the memory-order-violation
//!    penalty and hyperthread starvation of a naked spin.
//!    Notably, `SpinLock` — unlike `ctp::Mutex` — deliberately does **not**
//!    yield to `CTP_THREAD_MODEL` or sleep; `util/singleton.h` acquires it
//!    while constructing the very singletons the thread model is made of, so
//!    the wait path must not re-enter that machinery. This port preserves the
//!    no-yield, no-sleep property, and consequently `SpinLock` is only
//!    appropriate for short critical sections.
//!
//! 8. **Not reentrant** (same as C++): a thread that calls `lock()` twice
//!    without an intervening `unlock()` deadlocks against its own ticket.

#![deny(unsafe_code)]

use std::hint::spin_loop;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// Owner id to pass when you have none — the C++ ignores this argument
/// everywhere, and its only in-tree call site passes `0`. See divergence 5.
pub const UNUSED_OWNER: u32 = 0;

/// A FIFO ticket spin lock (`ctp::SpinLock`).
///
/// Layout is `#[repr(C)]` and field order matches the C++ (`lock_`, `head_`,
/// `try_lock_`) so the same bytes are interpretable from either language. There
/// is no `Drop`: the lock is a plain region of shared state, never an owning
/// handle.
///
/// All-zero bytes are a valid unheld lock, so a freshly `mmap`ed (zero-filled)
/// segment needs no constructor pass — mirroring the C++ `singleton.h` idiom.
///
/// # Example
///
/// ```
/// use ctp_thread::spinlock::{SpinLock, UNUSED_OWNER};
///
/// let lock = SpinLock::new();
/// {
///     let _guard = lock.acquire(UNUSED_OWNER);
///     // critical section; released when `_guard` drops
/// }
/// assert!(lock.try_lock(UNUSED_OWNER));
/// lock.unlock();
/// ```
#[repr(C)]
#[derive(Debug)]
pub struct SpinLock {
    /// C++ `lock_`: next ticket to hand out. Monotonic, wraps.
    ticket: AtomicU64,
    /// C++ `head_`: ticket currently being served. The lock is held iff
    /// `head != ticket`.
    head: AtomicU64,
    /// C++ `try_lock_`: serializes concurrent `try_lock` attempts so that only
    /// one of them may observe "free" and commit to taking a ticket. Without
    /// it, two try-lockers could both see a free lock, both take tickets, and
    /// the loser would *block* in `lock()` — violating try semantics.
    try_gate: AtomicU32,
}

// The C++ interoperates with these bytes; catch any accidental layout drift.
// Only asserted on 64-bit targets, where `AtomicU64` is 8-aligned (matching the
// C++ `std::atomic<u64>`); 32-bit targets may align it to 4 and shrink the
// struct, which would not be C++-ABI-compatible anyway.
#[cfg(target_pointer_width = "64")]
const _: () = {
    assert!(core::mem::size_of::<SpinLock>() == 24);
    assert!(core::mem::align_of::<SpinLock>() == 8);
};

impl SpinLock {
    /// Create a new, unheld lock (C++ default ctor `SpinLock()`).
    ///
    /// `const`, so a `SpinLock` can be placed in a `static` and is a valid
    /// all-zero bit pattern.
    #[inline]
    pub const fn new() -> Self {
        Self {
            ticket: AtomicU64::new(0),
            head: AtomicU64::new(0),
            try_gate: AtomicU32::new(0),
        }
    }

    /// Reset to the unheld state (C++ `Init()`).
    ///
    /// Only safe to call when no thread or process holds or is waiting on the
    /// lock — it is an initializer for freshly mapped memory, not a recovery
    /// primitive. Unlike the C++ this resets all three fields; see divergence 2.
    #[inline]
    pub fn init(&self) {
        self.ticket.store(0, Ordering::SeqCst);
        self.head.store(0, Ordering::SeqCst);
        self.try_gate.store(0, Ordering::SeqCst);
    }

    /// Acquire the lock, busy-waiting until this caller's ticket is served
    /// (C++ `Lock(u32 owner)`).
    ///
    /// FIFO: callers are served in the order they took tickets. `owner` is
    /// ignored (divergence 5). This never yields or sleeps (divergence 7), so
    /// only guard short critical sections with it.
    #[inline]
    pub fn lock(&self, _owner: u32) {
        // Take a ticket. SeqCst matches the C++ `ipc::atomic` default
        // (`memory_order_seq_cst`) and subsumes the acquire the lock needs.
        let tkt = self.ticket.fetch_add(1, Ordering::SeqCst);
        // Wait to be served. `==` (not `>=`) is what makes this wrap-safe.
        while self.head.load(Ordering::SeqCst) != tkt {
            spin_loop();
        }
    }

    /// Try to acquire the lock without blocking (C++ `TryLock(u32 owner)`).
    ///
    /// Returns `true` if the lock was acquired — the caller must then
    /// [`unlock`](Self::unlock) it. Returns `false` if the lock was held, or if
    /// another `try_lock` was in flight (the gate is conservative: a losing
    /// try-locker reports failure rather than blocking).
    ///
    /// Unlike the C++, this is repeatable — see divergence 3.
    #[inline]
    pub fn try_lock(&self, owner: u32) -> bool {
        // Gate: only one try-locker at a time may run the "free?" test and
        // commit to a ticket, so the `lock()` below can never actually block.
        if self.try_gate.fetch_add(1, Ordering::SeqCst) > 0
            || self.ticket.load(Ordering::SeqCst) > self.head.load(Ordering::SeqCst)
        {
            self.try_gate.fetch_sub(1, Ordering::SeqCst);
            return false;
        }
        // We are the only try-locker and observed the lock free, so this takes
        // a ticket that `head` is already serving and returns without spinning.
        self.lock(owner);
        // Release the gate. The C++ omits this, wedging TryLock permanently
        // after the first success (divergence 3).
        self.try_gate.fetch_sub(1, Ordering::SeqCst);
        true
    }

    /// Release the lock, serving the next ticket (C++ `Unlock()`).
    ///
    /// Must only be called by a caller that holds the lock. An unbalanced
    /// `unlock` cannot cause undefined behavior (the state is just two
    /// counters), but it hands the lock to a waiter that has not been served
    /// yet and so destroys mutual exclusion. Prefer [`acquire`](Self::acquire),
    /// whose guard makes the pairing structural.
    #[inline]
    pub fn unlock(&self) {
        self.head.fetch_add(1, Ordering::SeqCst);
    }

    /// True if the lock is currently held by someone.
    ///
    /// Advisory only — the answer may be stale the instant it is returned. No
    /// C++ counterpart on `SpinLock` (`RwLock::IsWriteLocked()` is the analogue
    /// there); provided for tests and diagnostics.
    #[inline]
    pub fn is_locked(&self) -> bool {
        self.ticket.load(Ordering::SeqCst) != self.head.load(Ordering::SeqCst)
    }

    /// Acquire the lock and return an RAII guard that releases it on drop
    /// (C++ `ScopedSpinLock lock(spin_lock, owner)`).
    #[inline]
    pub fn acquire(&self, owner: u32) -> ScopedSpinLock<'_> {
        ScopedSpinLock::new(self, owner)
    }

    /// Try to acquire the lock, returning an RAII guard on success.
    ///
    /// `None` means the lock was held (or lost the try-gate race); no state is
    /// left modified in that case.
    #[inline]
    pub fn try_acquire(&self, owner: u32) -> Option<ScopedSpinLock<'_>> {
        let mut guard = ScopedSpinLock::deferred(self);
        if guard.try_lock(owner) {
            Some(guard)
        } else {
            None
        }
    }
}

impl Default for SpinLock {
    /// A fresh, unheld lock — also the well-defined stand-in for the C++ copy
    /// constructor's intent (divergence 1).
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// RAII guard for a [`SpinLock`] (C++ `ctp::ScopedSpinLock`).
///
/// Releases the lock on drop. Like the C++ it tracks whether it currently holds
/// the lock, so [`unlock`](Self::unlock) is idempotent and a released guard may
/// be re-locked. The borrow of the `SpinLock` is what keeps the lock alive for
/// at least as long as the guard — something the C++ `SpinLock &lock_` member
/// cannot enforce.
///
/// This guard hands out no data: like the C++, a `SpinLock` protects state that
/// lives elsewhere (typically elsewhere in the same shared-memory segment), so
/// this is a lock, not a `std::sync::Mutex<T>`-style container.
#[derive(Debug)]
pub struct ScopedSpinLock<'a> {
    lock: &'a SpinLock,
    is_locked: bool,
}

impl<'a> ScopedSpinLock<'a> {
    /// Acquire `lock` and return a guard holding it (C++
    /// `ScopedSpinLock(SpinLock &lock, uint32_t owner)`).
    #[inline]
    pub fn new(lock: &'a SpinLock, owner: u32) -> Self {
        let mut guard = Self::deferred(lock);
        guard.lock(owner);
        guard
    }

    /// Create a guard that does **not** yet hold `lock`.
    ///
    /// No C++ counterpart: `ScopedSpinLock`'s only constructor acquires, which
    /// makes its `TryLock`/`Unlock`-then-`Lock` methods unreachable in practice
    /// (`is_locked_` is always true on entry). This constructor is what makes
    /// those ported methods — and [`SpinLock::try_acquire`] — usable.
    #[inline]
    pub fn deferred(lock: &'a SpinLock) -> Self {
        Self {
            lock,
            is_locked: false,
        }
    }

    /// Acquire the lock if this guard does not already hold it
    /// (C++ `ScopedSpinLock::Lock`).
    #[inline]
    pub fn lock(&mut self, owner: u32) {
        if !self.is_locked {
            self.lock.lock(owner);
            self.is_locked = true;
        }
    }

    /// Try to acquire the lock if this guard does not already hold it
    /// (C++ `ScopedSpinLock::TryLock`).
    ///
    /// Returns whether the guard holds the lock afterwards — so `true` if it
    /// already held it, matching the C++.
    #[inline]
    pub fn try_lock(&mut self, owner: u32) -> bool {
        if !self.is_locked {
            self.is_locked = self.lock.try_lock(owner);
        }
        self.is_locked
    }

    /// Release the lock if held; a no-op otherwise
    /// (C++ `ScopedSpinLock::Unlock`).
    #[inline]
    pub fn unlock(&mut self) {
        if self.is_locked {
            self.lock.unlock();
            self.is_locked = false;
        }
    }

    /// Whether this guard currently holds the lock (C++ `is_locked_`).
    #[inline]
    pub fn is_locked(&self) -> bool {
        self.is_locked
    }
}

impl Drop for ScopedSpinLock<'_> {
    /// C++ `~ScopedSpinLock()` — releases the lock if still held.
    #[inline]
    fn drop(&mut self) {
        self.unlock();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64 as StdAtomicU64;
    use std::sync::{Arc, Mutex};
    use std::thread;

    // -- structural / shared-memory suitability ----------------------------

    #[test]
    fn test_layout_matches_cpp() {
        // Must stay byte-compatible with the C++ struct (two u64 + one u32).
        assert_eq!(core::mem::size_of::<SpinLock>(), 24);
        assert_eq!(core::mem::align_of::<SpinLock>(), 8);
    }

    #[test]
    fn test_no_drop_glue() {
        // MEMORY_DESIGN.md: shared-memory state must never have Drop, or a
        // process unmapping a segment would "release" another process's lock.
        assert!(!std::mem::needs_drop::<SpinLock>());
    }

    #[test]
    fn test_zeroed_bytes_are_a_valid_unheld_lock() {
        // `util/singleton.h` reinterprets a zeroed char buffer as a SpinLock
        // and locks it without ever running a constructor; a freshly mmap'ed
        // shm segment is likewise zero-filled. Prove zero == unheld.
        let zeroed: SpinLock = SpinLock::new();
        assert_eq!(zeroed.ticket.load(Ordering::SeqCst), 0);
        assert_eq!(zeroed.head.load(Ordering::SeqCst), 0);
        assert_eq!(zeroed.try_gate.load(Ordering::SeqCst), 0);
        assert!(!zeroed.is_locked());
        assert!(zeroed.try_lock(UNUSED_OWNER));
    }

    #[test]
    fn test_usable_from_a_static() {
        // `const fn new()` is what lets the lock live in static/shm storage.
        static LOCK: SpinLock = SpinLock::new();
        let g = LOCK.acquire(UNUSED_OWNER);
        assert!(g.is_locked());
        drop(g);
        assert!(!LOCK.is_locked());
    }

    #[test]
    fn test_is_send_and_sync() {
        // Required to place the lock in shm shared across threads/processes.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SpinLock>();
    }

    // -- basic lock / unlock ----------------------------------------------

    #[test]
    fn test_new_is_unlocked() {
        let lock = SpinLock::new();
        assert!(!lock.is_locked());
        assert!(!SpinLock::default().is_locked());
    }

    #[test]
    fn test_lock_unlock_roundtrip() {
        let lock = SpinLock::new();
        lock.lock(UNUSED_OWNER);
        assert!(lock.is_locked());
        lock.unlock();
        assert!(!lock.is_locked());
        // Repeatable.
        lock.lock(UNUSED_OWNER);
        lock.unlock();
        assert!(!lock.is_locked());
    }

    #[test]
    fn test_init_resets_all_state() {
        // Divergence 2: C++ Init() resets only lock_, which would leave
        // head_ == 1 here and deadlock the next Lock().
        let lock = SpinLock::new();
        lock.lock(UNUSED_OWNER);
        lock.unlock();
        assert_eq!(lock.ticket.load(Ordering::SeqCst), 1);
        assert_eq!(lock.head.load(Ordering::SeqCst), 1);

        lock.init();
        assert_eq!(lock.ticket.load(Ordering::SeqCst), 0);
        assert_eq!(lock.head.load(Ordering::SeqCst), 0);
        assert_eq!(lock.try_gate.load(Ordering::SeqCst), 0);
        // A lock re-initialized this way still works.
        lock.lock(UNUSED_OWNER);
        lock.unlock();
    }

    // -- try_lock ----------------------------------------------------------

    #[test]
    fn test_try_lock_on_free_lock_succeeds() {
        let lock = SpinLock::new();
        assert!(lock.try_lock(UNUSED_OWNER));
        assert!(lock.is_locked());
        lock.unlock();
    }

    #[test]
    fn test_try_lock_fails_while_held() {
        let lock = SpinLock::new();
        lock.lock(UNUSED_OWNER);
        assert!(!lock.try_lock(UNUSED_OWNER));
        // A failed try must leave the gate exactly as it found it.
        assert_eq!(lock.try_gate.load(Ordering::SeqCst), 0);
        lock.unlock();
        assert!(lock.try_lock(UNUSED_OWNER));
        lock.unlock();
    }

    #[test]
    fn test_try_lock_is_repeatable() {
        // Pins divergence 3. Against the C++ algorithm the SECOND try_lock
        // returns false forever, because TryLock never releases try_lock_ on
        // its success path.
        let lock = SpinLock::new();
        for i in 0..100 {
            assert!(lock.try_lock(UNUSED_OWNER), "try_lock failed on attempt {i}");
            assert_eq!(
                lock.try_gate.load(Ordering::SeqCst),
                0,
                "try-gate leaked on attempt {i}"
            );
            lock.unlock();
        }
    }

    #[test]
    fn test_try_lock_never_blocks_when_contended() {
        // The gate's whole purpose: concurrent try-lockers must all return,
        // and at most one may hold the lock at a time. If the losers fell into
        // a blocking lock(), this test would hang.
        const THREADS: usize = 8;
        const ITERS: usize = 500;
        let lock = Arc::new(SpinLock::new());
        let successes = Arc::new(StdAtomicU64::new(0));
        let concurrent = Arc::new(StdAtomicU64::new(0));

        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                let lock = Arc::clone(&lock);
                let successes = Arc::clone(&successes);
                let concurrent = Arc::clone(&concurrent);
                thread::spawn(move || {
                    for _ in 0..ITERS {
                        if lock.try_lock(UNUSED_OWNER) {
                            // Nobody else may be inside at the same time.
                            let inside = concurrent.fetch_add(1, Ordering::SeqCst);
                            assert_eq!(inside, 0, "try_lock granted a held lock");
                            successes.fetch_add(1, Ordering::SeqCst);
                            concurrent.fetch_sub(1, Ordering::SeqCst);
                            lock.unlock();
                        }
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }

        assert!(successes.load(Ordering::SeqCst) > 0, "no try_lock ever won");
        assert!(!lock.is_locked());
        assert_eq!(lock.try_gate.load(Ordering::SeqCst), 0, "try-gate leaked");
    }

    // -- fairness (the defining property of a ticket lock) -----------------

    #[test]
    fn test_acquisition_is_fifo() {
        // Deterministic: each waiter is confirmed to have taken its ticket
        // (via `ticket`) before the next one is spawned, so arrival order is
        // known and the release order must match it.
        let lock = Arc::new(SpinLock::new());
        let order = Arc::new(Mutex::new(Vec::new()));

        lock.lock(UNUSED_OWNER); // main holds ticket 0; `ticket` is now 1

        let mut handles = Vec::new();
        for id in 0..4u32 {
            let waiter_lock = Arc::clone(&lock);
            let waiter_order = Arc::clone(&order);
            handles.push(thread::spawn(move || {
                let _g = waiter_lock.acquire(UNUSED_OWNER);
                waiter_order.lock().unwrap().push(id);
            }));
            // Wait until this thread has actually taken ticket `id + 1`
            // before letting the next one race for a ticket.
            while lock.ticket.load(Ordering::SeqCst) != u64::from(id) + 2 {
                spin_loop();
            }
        }

        lock.unlock(); // release ticket 0 -> waiters drain in ticket order
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(*order.lock().unwrap(), vec![0, 1, 2, 3]);
    }

    // -- mutual exclusion under contention ---------------------------------

    #[test]
    fn test_mutual_exclusion_under_contention() {
        // A deliberately non-atomic read-modify-write (load, gap, store): the
        // final count can only be exact if the lock truly serializes.
        const THREADS: usize = 8;
        const ITERS: usize = 2_000;
        let lock = Arc::new(SpinLock::new());
        let counter = Arc::new(StdAtomicU64::new(0));

        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                let lock = Arc::clone(&lock);
                let counter = Arc::clone(&counter);
                thread::spawn(move || {
                    for _ in 0..ITERS {
                        let _g = lock.acquire(UNUSED_OWNER);
                        let v = counter.load(Ordering::SeqCst);
                        spin_loop(); // widen the race window
                        counter.store(v + 1, Ordering::SeqCst);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(counter.load(Ordering::SeqCst), (THREADS * ITERS) as u64);
        assert!(!lock.is_locked());
    }

    #[test]
    fn test_no_ticket_is_lost_under_contention() {
        const THREADS: u64 = 6;
        const ITERS: u64 = 1_000;
        let lock = Arc::new(SpinLock::new());
        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                let lock = Arc::clone(&lock);
                thread::spawn(move || {
                    for _ in 0..ITERS {
                        lock.lock(UNUSED_OWNER);
                        lock.unlock();
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        // Every acquisition took exactly one ticket and served exactly one.
        assert_eq!(lock.ticket.load(Ordering::SeqCst), THREADS * ITERS);
        assert_eq!(lock.head.load(Ordering::SeqCst), THREADS * ITERS);
    }

    // -- overflow / boundary ------------------------------------------------

    #[test]
    fn test_ticket_wraparound_at_u64_max() {
        // Pins divergence 4: the lock/unlock path wraps correctly because it
        // compares tickets with `==` and `fetch_add` wraps like C++ unsigned
        // overflow.
        let lock = SpinLock::new();
        lock.ticket.store(u64::MAX, Ordering::SeqCst);
        lock.head.store(u64::MAX, Ordering::SeqCst);
        assert!(!lock.is_locked());

        let g = lock.acquire(UNUSED_OWNER); // takes ticket u64::MAX
        assert_eq!(lock.ticket.load(Ordering::SeqCst), 0, "ticket must wrap");
        assert!(lock.is_locked());
        drop(g);
        assert_eq!(lock.head.load(Ordering::SeqCst), 0, "head must wrap");
        assert!(!lock.is_locked());

        // The lock keeps working across the wrap.
        lock.lock(UNUSED_OWNER);
        assert_eq!(lock.ticket.load(Ordering::SeqCst), 1);
        lock.unlock();
        assert!(!lock.is_locked());
    }

    #[test]
    fn test_fifo_handoff_across_wraparound() {
        let lock = Arc::new(SpinLock::new());
        lock.ticket.store(u64::MAX, Ordering::SeqCst);
        lock.head.store(u64::MAX, Ordering::SeqCst);

        lock.lock(UNUSED_OWNER); // ticket u64::MAX; `ticket` wraps to 0
        let waiter = {
            let lock = Arc::clone(&lock);
            thread::spawn(move || {
                let _g = lock.acquire(UNUSED_OWNER); // ticket 0
            })
        };
        while lock.ticket.load(Ordering::SeqCst) != 1 {
            spin_loop();
        }
        lock.unlock(); // head wraps to 0, serving the waiter
        waiter.join().unwrap();
        assert!(!lock.is_locked());
    }

    #[test]
    fn test_is_locked_is_wrap_safe() {
        let lock = SpinLock::new();
        // Held state straddling the wrap boundary: ticket wrapped, head not.
        lock.head.store(u64::MAX, Ordering::SeqCst);
        lock.ticket.store(0, Ordering::SeqCst);
        assert!(lock.is_locked(), "`!=` must see a held lock across the wrap");
        // (This is exactly the state in which C++'s `lock_ > head_` try_lock
        // test reports "free" — see divergence 4.)
    }

    // -- ScopedSpinLock -----------------------------------------------------

    #[test]
    fn test_guard_releases_on_drop() {
        let lock = SpinLock::new();
        {
            let g = lock.acquire(UNUSED_OWNER);
            assert!(g.is_locked());
            assert!(lock.is_locked());
        }
        assert!(!lock.is_locked());
    }

    #[test]
    fn test_guard_unlock_is_idempotent() {
        let lock = SpinLock::new();
        let mut g = lock.acquire(UNUSED_OWNER);
        g.unlock();
        assert!(!g.is_locked());
        assert!(!lock.is_locked());
        // A second unlock must NOT bump head again (that would hand the lock
        // to an unserved waiter).
        g.unlock();
        assert_eq!(lock.head.load(Ordering::SeqCst), 1);
        drop(g); // drop of a released guard is also a no-op
        assert_eq!(lock.head.load(Ordering::SeqCst), 1);
        assert!(!lock.is_locked());
    }

    #[test]
    fn test_guard_lock_is_idempotent() {
        // Mirrors C++ ScopedSpinLock::Lock's `if (!is_locked_)` guard: a second
        // Lock() on a held guard must not re-enter the (non-reentrant) lock.
        let lock = SpinLock::new();
        let mut g = lock.acquire(UNUSED_OWNER);
        g.lock(UNUSED_OWNER); // would deadlock if it took a second ticket
        assert!(g.is_locked());
        assert_eq!(lock.ticket.load(Ordering::SeqCst), 1);
        assert!(g.try_lock(UNUSED_OWNER)); // already held -> true, no-op
        assert_eq!(lock.ticket.load(Ordering::SeqCst), 1);
        assert_eq!(lock.try_gate.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn test_guard_relock_after_unlock() {
        let lock = SpinLock::new();
        let mut g = ScopedSpinLock::deferred(&lock);
        assert!(!g.is_locked());
        assert!(!lock.is_locked());

        g.lock(UNUSED_OWNER);
        assert!(g.is_locked());
        g.unlock();
        assert!(!g.is_locked());

        assert!(g.try_lock(UNUSED_OWNER));
        assert!(g.is_locked());
        drop(g);
        assert!(!lock.is_locked());
    }

    #[test]
    fn test_deferred_guard_does_not_acquire() {
        let lock = SpinLock::new();
        {
            let g = ScopedSpinLock::deferred(&lock);
            assert!(!g.is_locked());
            assert!(!lock.is_locked());
            // Dropping a never-locked guard must not touch the lock.
        }
        assert_eq!(lock.ticket.load(Ordering::SeqCst), 0);
        assert_eq!(lock.head.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn test_try_acquire_returns_none_when_held() {
        let lock = SpinLock::new();
        let held = lock.acquire(UNUSED_OWNER);
        assert!(lock.try_acquire(UNUSED_OWNER).is_none());
        // The failed attempt must leave no residue.
        assert_eq!(lock.try_gate.load(Ordering::SeqCst), 0);
        assert_eq!(lock.ticket.load(Ordering::SeqCst), 1);
        drop(held);

        let g = lock.try_acquire(UNUSED_OWNER);
        assert!(g.is_some());
        assert!(g.as_ref().unwrap().is_locked());
        drop(g);
        assert!(!lock.is_locked());
    }

    #[test]
    fn test_guards_serialize_across_threads() {
        let lock = Arc::new(SpinLock::new());
        let log = Arc::new(Mutex::new(Vec::new()));
        let g = lock.acquire(UNUSED_OWNER);
        log.lock().unwrap().push("main-in");

        let other = {
            let lock = Arc::clone(&lock);
            let log = Arc::clone(&log);
            thread::spawn(move || {
                let _g = lock.acquire(UNUSED_OWNER);
                log.lock().unwrap().push("other-in");
            })
        };
        while lock.ticket.load(Ordering::SeqCst) != 2 {
            spin_loop();
        }
        // The other thread is provably waiting and cannot have logged yet.
        log.lock().unwrap().push("main-out");
        drop(g);
        other.join().unwrap();
        assert_eq!(*log.lock().unwrap(), vec!["main-in", "main-out", "other-in"]);
    }
}
