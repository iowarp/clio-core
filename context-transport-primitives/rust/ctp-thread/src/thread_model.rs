// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).

//! Thread model abstraction — Rust port of `clio_ctp/thread/thread_model/`
//! (`thread_model.h`, `std_thread.h`), `src/thread_model.cc` (`BusyWait`), and
//! `clio_ctp/thread/thread_model_manager.h` (the `CTP_THREAD_MODEL` singleton).
//!
//! The C++ side is *static* polymorphism: `ThreadModel` has no virtual methods;
//! each backend (`StdThread`, `Pthread`, `Argobots`, `Cuda`, `Rocm`) duck-types
//! the same surface and CMake selects one via `CTP_DEFAULT_THREAD_MODEL`, which
//! the `CTP_THREAD_MODEL` macro resolves through `CrossSingleton`. Here that
//! contract is spelled out as a [`ThreadModel`] trait with [`StdThread`] — the
//! always-available backend, and the C++ default on Windows — implementing it.
//!
//! # C++ → Rust name mapping
//!
//! | C++ | Rust |
//! |---|---|
//! | `ctp::ThreadType` (`kNone`, `kPthread`, …) | [`ThreadType`] (`None`, `Pthread`, …) |
//! | `ctp::ThreadId` (`types/numbers.h`) | [`ThreadId`] |
//! | `ThreadId::GetNull()` / `IsNull()` / `SetNull()` | [`ThreadId::get_null`] / [`ThreadId::is_null`] / [`ThreadId::set_null`] |
//! | `ctp::ThreadGroup` | [`ThreadGroup`] |
//! | `ctp::ThreadGroupContext` | [`ThreadGroupContext`] |
//! | `ctp::Thread` | [`Thread`] |
//! | `ctp::ThreadParams<FUN, Args...>` | *(none — a Rust closure captures its own args)* |
//! | `ctp::ThreadLocalKey` | *(not ported — divergence 3)* |
//! | `ctp::thread::ThreadLocalData::destroy_wrap<TLS>` | *(not ported — divergence 3)* |
//! | `ctp::thread::ThreadModel` | [`ThreadModel`] (trait) |
//! | `ThreadModel::type_` / `GetType()` | [`ThreadModel::get_type`] |
//! | `ThreadModel::BusyWait(cond)` | [`ThreadModel::busy_wait`] + free [`busy_wait`] |
//! | `ctp::thread::StdThread` | [`StdThread`] |
//! | `StdThread::Init()` | [`ThreadModel::init`] |
//! | `StdThread::Yield()` | [`ThreadModel::yield_now`] (`yield` is a Rust keyword) |
//! | `StdThread::SleepForUs(us)` | [`ThreadModel::sleep_for_us`] |
//! | `StdThread::GetTid()` | [`ThreadModel::get_tid`] |
//! | `StdThread::CreateThreadGroup(ctx)` | [`ThreadModel::create_thread_group`] |
//! | `StdThread::Spawn(group, func, args...)` | [`ThreadModel::spawn`] / [`ThreadModel::spawn_boxed`] |
//! | `StdThread::Join(thread)` | [`ThreadModel::join`] |
//! | `StdThread::TimedJoinOrDetach(thread, timeout_ms)` | [`ThreadModel::timed_join_or_detach`] |
//! | `StdThread::SetAffinity(thread, cpu_id)` | [`ThreadModel::set_affinity`] |
//! | `CTP_THREAD_MODEL` (`CrossSingleton<…>::GetInstance()`) | [`thread_model()`] |
//! | `CTP_THREAD_MODEL_T` | [`ThreadModelT`] |
//! | `ctp::thread::Pthread` | *(not ported — divergence 2)* |
//! | `Argobots` / `Cuda` / `Rocm` | *(out of scope — divergence 1)* |
//!
//! # Units
//!
//! Per the migration rules timings are milliseconds unless the C++ says
//! otherwise — it does here: `SleepForUs` is **microseconds** and is kept as
//! [`ThreadModel::sleep_for_us`]. `TimedJoinOrDetach`'s `timeout_ms` is
//! milliseconds. `BusyWait`'s internal constants stay in microseconds.
//!
//! # Semantic divergences from the C++
//!
//! 1. **GPU / Argobots backends are not ported.** `Cuda`, `Rocm` (device-side
//!    `Yield()` = `threadfence()`, no-op sleep, `GetTid()` = null) and
//!    `Argobots` (`ABT_thread_*`, an `ABT_xstream` per thread group, real
//!    `SetAffinity`) wrap C/C++ APIs — CUDA/ROCm device code and the Mochi
//!    Argobots C++ API. Per MIGRATION.md those stay behind wrapper crates
//!    (`ctp-gpu` for CUDA/ROCm; a thallium wrapper for Argobots) and should
//!    implement [`ThreadModel`] from there once those crates exist. The
//!    [`ThreadType`] enum keeps their discriminants so the type tag round-trips
//!    across the FFI boundary meanwhile.
//! 2. **The `Pthread` backend is not ported.** It is the C++ default only where
//!    `CTP_ENABLE_PTHREADS` is set (i.e. not Windows). Two of its behaviors
//!    are deliberately *not* inherited by [`StdThread`], matching C++:
//!    (a) `Pthread::Yield()` issues a bare `pause`/`yield` CPU hint (<20ns)
//!    rather than `sched_yield`, whereas `StdThread::Yield()` calls
//!    `std::this_thread::yield()`; this port keeps `std::thread::yield_now()`.
//!    (b) `Pthread::GetTid()` hands out *dense, 0-based* ids from an atomic
//!    counter cached in TLS, while `StdThread::GetTid()` returns the raw OS
//!    tid. This port returns the OS tid. Code that used pthread tids to index
//!    a per-thread array must not assume density here.
//! 3. **No TLS API** (`ThreadLocalKey`, `CreateTls`/`SetTls`/`GetTls`,
//!    `ThreadLocalData::destroy_wrap`). `thread_local!` is banned project-wide
//!    (Windows duplicates it per-DLL — the #620 Boost bug), and `StdThread`'s
//!    TLS methods are pure pass-throughs to `SystemInfo::{Create,Set,Get}Tls`,
//!    which belongs to `ctp-introspect` — not a dependency of this crate, and
//!    Cargo.toml may not be edited here. Its only in-tree consumer is the
//!    `Pthread` tid counter (divergence 2). Thread-affine state should go
//!    through explicit context or `OnceLock`/atomics instead. Note the C++
//!    `destroy_wrap` is already a no-op (its body is commented out with a
//!    "TODO: figure out why this segfaults on exit"), so no destructor
//!    semantics are lost.
//! 4. **`TimedJoinOrDetach` honors the timeout.** C++ `StdThread` documents the
//!    contract ("detach if the timeout elapses; returns false if detached") but
//!    *ignores* `timeout_ms`, performs a blocking join and always returns true,
//!    with a `TODO: implement via native handle if a deadline is required on
//!    Windows`. This port implements the documented contract — the same one
//!    `Pthread` (the reference implementation, via `pthread_timedjoin_np`)
//!    already honors — using a completion flag + `Condvar`, portably and with
//!    no native handle. So a Rust caller can observe `false` + a still-running
//!    detached thread where C++-on-Windows would have blocked. Any caller
//!    already correct against the pthread backend is unaffected. Like
//!    `pthread_timedjoin_np`, the flag is set as the closure returns, so a
//!    `true` result may still block for the microseconds the runtime needs to
//!    finish tearing the thread down. `timeout_ms == 0` polls once and detaches
//!    if not already finished. Matching C++, the handle is cleared either way
//!    (`pthread_thread_ = 0`), so a second call returns `true`.
//! 5. **`Join` returns a `Result`.** C++ `Join` returns `void`: joining a
//!    non-joinable `std::thread` throws `std::system_error`, and an exception
//!    escaping the child calls `std::terminate`. Rust cannot reproduce either,
//!    so both are reported: [`JoinFailure::NotJoinable`] and
//!    [`JoinFailure::Panicked`] (the child's panic unwinds only the child; the
//!    payload is dropped rather than propagated, so a caller that ignores the
//!    `Result` sees a *survived* process where C++ would have aborted it).
//! 6. **`Thread`'s drop detaches; C++'s terminates.** `~std::thread` on a
//!    joinable thread calls `std::terminate`; dropping a Rust `JoinHandle`
//!    silently detaches. [`Thread`] therefore has no `Drop` impl and inherits
//!    `JoinHandle`'s detach-on-drop.
//! 7. **`BusyWait` drives `self`, not the singleton.** The C++ out-of-line
//!    definition calls `CTP_THREAD_MODEL->Yield()`/`->SleepForUs()`, i.e. the
//!    process-wide singleton, even when invoked on some other instance. The
//!    default method here drives `self`. Identical for the singleton (the only
//!    instance in practice); the free [`busy_wait`] mirrors the C++ spelling.
//!    The tiered policy itself is ported verbatim from `src/thread_model.cc`:
//!    yield-spin for 5 ms, then sleep 1, 2, 4 … µs capped at 1024 µs. NB the
//!    C++ *doc comment* in `thread_model.h` advertises a third, earlier phase
//!    ("tight spin, no yield, ~200us"); no such phase exists in the shipped
//!    `.cc`. The code was ported, not the comment.
//! 8. **`ThreadId::default()` is null.** C++ `ThreadId()` leaves `tid_`
//!    *indeterminate*; Rust cannot, so `Default` yields [`ThreadId::NULL`]
//!    (`u64::MAX`), matching `GetNull()`/`SetNull()`.
//! 9. **`get_tid` does not narrow through `int`.** C++ `SystemInfo::GetTid()`
//!    returns `int` and `StdThread::GetTid()` widens it into a `u64`, so a
//!    Windows tid ≥ 2^31 sign-extends to a huge bogus id (and Linux tids are
//!    capped well below that). This port keeps the value unsigned: no
//!    sign-extension, no narrowing.
//! 10. **`sleep_for_us` is reimplemented, not delegated.** C++ routes through
//!     `SystemInfo::SleepForUs` specifically to get Windows a high-resolution
//!     waitable timer instead of `std::this_thread::sleep_for`'s coarse
//!     ~1–15.6 ms tick — which `BusyWait`'s 1 µs backoff floor depends on. This
//!     crate cannot depend on `ctp-introspect` (not in Cargo.toml, which this
//!     task may not edit), so that Win32 path is duplicated here verbatim, and
//!     POSIX calls `nanosleep` directly (no EINTR retry, matching C++, unlike
//!     `std::thread::sleep`). `CreateWaitableTimerExW` is additionally
//!     hand-declared, as windows-sys gates it behind a `Win32_Security` feature
//!     this crate's (uneditable) Cargo.toml does not enable — see the comment in
//!     `sys_sleep_for_us`. **Follow-up:** `ctp_introspect::sleep_for_us` is
//!     currently a plain `std::thread::sleep` and lacks the high-resolution
//!     path; it should adopt this implementation, and this module should then
//!     delegate to it (which would also retire the hand-declared import).
//! 11. **`ThreadId` is defined here.** In C++ it lives in `types/numbers.h`
//!     (→ `ctp-types`), whose Rust port does not expose it yet; another crate's
//!     files may not be edited under this task. It should move to `ctp-types`
//!     and be re-exported here.
//! 12. **Windows due-time saturates.** `SetWaitableTimer`'s relative due time
//!     is `-(us * 10)` in 100 ns units; C++ overflows (UB) for absurd `us`,
//!     while `win_due_time_100ns` saturates at `-i64::MAX` (~29,000 years) —
//!     Rust would otherwise panic on overflow in debug builds.
//! 13. **`spawn` takes one closure, not `(func, args...)`.** `ThreadParams` +
//!     `ArgPack` exist because C++ must store a deferred call; a Rust closure
//!     captures its own arguments. [`ThreadModel::spawn_boxed`] keeps the trait
//!     object-safe; [`ThreadModel::spawn`] is the generic sugar.
//! 14. **`ThreadGroup`/`SetAffinity` are inert**, exactly as in C++'s
//!     `StdThread`: `CreateThreadGroup` ignores its context and `SetAffinity` is
//!     a no-op. Both carry state only under the Argobots backend (divergence 1),
//!     so the types are kept as placeholders to preserve the call surface.

use std::fmt;
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// thread_model.h — ThreadType / ThreadId / ThreadGroup / Thread
// ---------------------------------------------------------------------------

/// Available thread models (`ctp::ThreadType`).
///
/// Discriminants are pinned so the tag can cross the C ABI unchanged.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ThreadType {
    /// `ThreadType::kNone`
    #[default]
    None = 0,
    /// `ThreadType::kPthread` — see divergence 2.
    Pthread = 1,
    /// `ThreadType::kArgobots` — see divergence 1.
    Argobots = 2,
    /// `ThreadType::kCuda` — see divergence 1.
    Cuda = 3,
    /// `ThreadType::kRocm` — see divergence 1.
    Rocm = 4,
    /// `ThreadType::kStdThread` — the backend implemented by this module.
    StdThread = 5,
}

/// `ctp::ThreadId` (C++ `types/numbers.h`; see divergence 11).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ThreadId {
    /// `tid_`
    pub tid: u64,
}

impl ThreadId {
    /// The null id — `(ctp::u64)-1`, as in `ThreadId::GetNull()`.
    pub const NULL: Self = Self { tid: u64::MAX };

    /// `explicit ThreadId(ctp::u64 tid)`
    #[inline]
    pub const fn new(tid: u64) -> Self {
        Self { tid }
    }

    /// `ThreadId::GetNull()`
    #[inline]
    pub const fn get_null() -> Self {
        Self::NULL
    }

    /// `ThreadId::IsNull()`
    #[inline]
    pub const fn is_null(&self) -> bool {
        self.tid == u64::MAX
    }

    /// `ThreadId::SetNull()`
    #[inline]
    pub fn set_null(&mut self) {
        self.tid = u64::MAX;
    }
}

/// Divergence 8: C++ leaves `tid_` indeterminate; null is the safe analogue.
impl Default for ThreadId {
    #[inline]
    fn default() -> Self {
        Self::NULL
    }
}

/// Mirrors `operator<<(std::ostream&, const ThreadId&)`.
impl fmt::Display for ThreadId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.tid)
    }
}

/// `ctp::ThreadGroupContext` — a placeholder in C++ too (`int nothing_`);
/// Argobots would carry scheduler settings here. See divergence 14.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ThreadGroupContext {
    /// `nothing_`
    pub nothing: i32,
}

/// `ctp::ThreadGroup`. Empty for every non-Argobots backend (the C++ struct
/// holds an `ABT_xstream` only under `CTP_ENABLE_THALLIUM`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ThreadGroup;

/// Why a [`ThreadModel::join`] did not complete normally. See divergence 5.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JoinFailure {
    /// No live thread to join — never spawned, or already joined/detached.
    /// C++ `std::thread::join()` throws `std::system_error` here.
    NotJoinable,
    /// The thread's closure panicked. C++ would have called `std::terminate`
    /// when the exception escaped the child.
    Panicked,
}

impl fmt::Display for JoinFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JoinFailure::NotJoinable => write!(f, "thread is not joinable"),
            JoinFailure::Panicked => write!(f, "thread panicked"),
        }
    }
}

impl std::error::Error for JoinFailure {}

/// Completion signal backing the timed join (divergence 4).
#[derive(Debug)]
struct Completion {
    done: Mutex<bool>,
    cv: Condvar,
}

impl Completion {
    fn new() -> Self {
        Self {
            done: Mutex::new(false),
            cv: Condvar::new(),
        }
    }

    /// Mark the closure finished and wake every waiter.
    fn signal(&self) {
        // A poisoned lock still holds a valid bool: a panicking child must
        // still be observed as "done", else timed joins would spin to timeout.
        let mut done = self.done.lock().unwrap_or_else(|e| e.into_inner());
        *done = true;
        drop(done);
        self.cv.notify_all();
    }

    /// Wait up to `timeout` for the closure to finish. `true` if it did.
    /// A zero timeout degenerates to a single poll, matching a
    /// `pthread_timedjoin_np` deadline of "now".
    fn wait_timeout(&self, timeout: Duration) -> bool {
        let done = self.done.lock().unwrap_or_else(|e| e.into_inner());
        // wait_timeout_while re-checks the predicate, so spurious wakeups and
        // partial waits cannot cut the deadline short.
        let (done, _res) = self
            .cv
            .wait_timeout_while(done, timeout, |done| !*done)
            .unwrap_or_else(|e| e.into_inner());
        *done
    }
}

/// Signals [`Completion`] however the closure leaves — return *or* panic.
/// Without the `Drop`, a panicking child would leave timed joins to time out
/// and detach a thread that had in fact already finished.
struct SignalOnDrop(Arc<Completion>);

impl Drop for SignalOnDrop {
    fn drop(&mut self) {
        self.0.signal();
    }
}

/// `ctp::Thread`. Move-only, like the `std::thread` it wraps.
///
/// Divergence 6: dropping a joinable `Thread` detaches it (Rust `JoinHandle`
/// semantics) where `~std::thread` would call `std::terminate`.
#[derive(Debug, Default)]
pub struct Thread {
    group: ThreadGroup,
    handle: Option<JoinHandle<()>>,
    completion: Option<Arc<Completion>>,
}

impl Thread {
    /// The group this thread was spawned into (`thread.group_`).
    #[inline]
    pub fn group(&self) -> ThreadGroup {
        self.group
    }

    /// `thread.std_thread_.joinable()`. False for a default-constructed
    /// [`Thread`], and after a successful join or a detach.
    #[inline]
    pub fn is_joinable(&self) -> bool {
        self.handle.is_some()
    }
}

// ---------------------------------------------------------------------------
// thread_model.h — the ThreadModel surface (+ BusyWait from thread_model.cc)
// ---------------------------------------------------------------------------

/// Yield-spin window before `BusyWait` backs off to sleeping (`kYieldUs`).
const BUSY_WAIT_YIELD_WINDOW: Duration = Duration::from_micros(5000);

/// Cap on `BusyWait`'s exponential sleep backoff (`kMaxSleepUs`, ~1 ms).
const BUSY_WAIT_MAX_SLEEP_US: u64 = 1024;

/// The generic operations of a thread (`ctp::thread::ThreadModel`).
///
/// C++ resolves this statically — no vtable, one backend chosen at build time.
/// Here it is a trait, kept object-safe (the generic [`ThreadModel::spawn`] is
/// a `Self: Sized` default) so a `&dyn ThreadModel` remains usable.
pub trait ThreadModel {
    /// `ThreadModel::GetType()` / `type_`.
    fn get_type(&self) -> ThreadType;

    /// `Init()`. A no-op for [`StdThread`]; `Argobots::Init` calls `ABT_init`.
    fn init(&self) {}

    /// `Yield()` — yield this thread's time slice.
    ///
    /// Named `yield_now` because `yield` is a reserved Rust keyword. Called
    /// from tight spin loops, so it must stay cheap; see divergence 2a for how
    /// the pthread backend differs.
    fn yield_now(&self);

    /// `SleepForUs(us)` — sleep for **microseconds** (not ms; see § Units).
    /// `us == 0` returns without sleeping.
    fn sleep_for_us(&self, us: u64);

    /// `GetTid()` — the id of the calling thread.
    fn get_tid(&self) -> ThreadId;

    /// `CreateThreadGroup(ctx)`.
    fn create_thread_group(&self, ctx: &ThreadGroupContext) -> ThreadGroup;

    /// `Spawn(group, func, args...)`, object-safe form. See divergence 13.
    fn spawn_boxed(&self, group: &ThreadGroup, func: Box<dyn FnOnce() + Send + 'static>) -> Thread;

    /// `Join(thread)` — block until the thread finishes. See divergence 5.
    fn join(&self, thread: &mut Thread) -> Result<(), JoinFailure>;

    /// `TimedJoinOrDetach(thread, timeout_ms)` — best-effort join with a
    /// **millisecond** timeout; detach if it elapses. `true` if joined cleanly,
    /// `false` if detached. See divergence 4.
    fn timed_join_or_detach(&self, thread: &mut Thread, timeout_ms: u64) -> bool;

    /// `SetAffinity(thread, cpu_id)`. A no-op here, as in C++'s `StdThread`.
    fn set_affinity(&self, thread: &mut Thread, cpu_id: i32);

    /// `Spawn(group, func, args...)` — generic sugar over [`Self::spawn_boxed`].
    fn spawn<F>(&self, group: &ThreadGroup, func: F) -> Thread
    where
        F: FnOnce() + Send + 'static,
        Self: Sized,
    {
        self.spawn_boxed(group, Box::new(func))
    }

    /// `ThreadModel::BusyWait(cond)` — ported verbatim from `thread_model.cc`.
    ///
    /// Wait until `cond()` returns true, escalating so the fast path stays
    /// low-latency without pinning a core on a slow wait:
    ///
    /// 1. **Yield-spin** for ~5 ms — returns the core each iteration but stays
    ///    hot enough to react within a scheduler tick.
    /// 2. **Exponential-backoff sleep** — 1, 2, 4 … µs, capped at 1024 µs —
    ///    near-zero CPU for long waits.
    ///
    /// `cond` is evaluated very frequently: keep it cheap and side-effect-free
    /// (e.g. an atomic flag load). It is checked before any yield or sleep, so
    /// an already-satisfied condition costs exactly one call and returns
    /// without touching the scheduler.
    ///
    /// Divergence 7: drives `self` rather than the `CTP_THREAD_MODEL` singleton
    /// (`&mut dyn FnMut` stands in for `const std::function<bool()>&`, whose
    /// const `operator()` may likewise mutate captured state).
    fn busy_wait(&self, cond: &mut dyn FnMut() -> bool) {
        // Phase 1: yield-spin.
        let phase_start = Instant::now();
        loop {
            if cond() {
                return;
            }
            self.yield_now();
            if phase_start.elapsed() >= BUSY_WAIT_YIELD_WINDOW {
                break;
            }
        }

        // Phase 2: exponential-backoff sleep until done.
        let mut sleep_us: u64 = 1;
        while !cond() {
            self.sleep_for_us(sleep_us);
            if sleep_us < BUSY_WAIT_MAX_SLEEP_US {
                sleep_us <<= 1;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// std_thread.h — the StdThread backend
// ---------------------------------------------------------------------------

/// `ctp::thread::StdThread` — always available; the C++ default on Windows and
/// the GPU fallback when neither CUDA nor ROCm is enabled.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StdThread;

impl StdThread {
    /// `StdThread()` — sets `type_` to `ThreadType::kStdThread`.
    #[inline]
    pub const fn new() -> Self {
        Self
    }
}

impl ThreadModel for StdThread {
    #[inline]
    fn get_type(&self) -> ThreadType {
        ThreadType::StdThread
    }

    #[inline]
    fn yield_now(&self) {
        std::thread::yield_now();
    }

    #[inline]
    fn sleep_for_us(&self, us: u64) {
        sys_sleep_for_us(us);
    }

    #[inline]
    fn get_tid(&self) -> ThreadId {
        ThreadId::new(sys_get_tid())
    }

    #[inline]
    fn create_thread_group(&self, _ctx: &ThreadGroupContext) -> ThreadGroup {
        // `return ThreadGroup{};` — the context is ignored, as in C++.
        ThreadGroup
    }

    fn spawn_boxed(&self, group: &ThreadGroup, func: Box<dyn FnOnce() + Send + 'static>) -> Thread {
        let completion = Arc::new(Completion::new());
        let child_completion = Arc::clone(&completion);
        let handle = std::thread::spawn(move || {
            // Signals on return *and* on unwind; see SignalOnDrop.
            let _signal = SignalOnDrop(child_completion);
            func();
        });
        Thread {
            group: *group,
            handle: Some(handle),
            completion: Some(completion),
        }
    }

    fn join(&self, thread: &mut Thread) -> Result<(), JoinFailure> {
        match thread.handle.take() {
            None => Err(JoinFailure::NotJoinable),
            Some(handle) => {
                thread.completion = None;
                handle.join().map_err(|_| JoinFailure::Panicked)
            }
        }
    }

    fn timed_join_or_detach(&self, thread: &mut Thread, timeout_ms: u64) -> bool {
        // `if (!thread.std_thread_.joinable()) return true;`
        let Some(handle) = thread.handle.take() else {
            return true;
        };
        let finished = match thread.completion.take() {
            Some(completion) => completion.wait_timeout(Duration::from_millis(timeout_ms)),
            None => true,
        };
        if finished {
            // Already done: this cannot block for more than the teardown.
            let _ = handle.join();
            return true;
        }
        // Timed out. Dropping the handle detaches, mirroring pthread_detach;
        // the handle is cleared either way, as C++ zeroes pthread_thread_.
        drop(handle);
        false
    }

    #[inline]
    fn set_affinity(&self, thread: &mut Thread, cpu_id: i32) {
        // `void SetAffinity(Thread &thread, int cpu_id) {}` — no-op in C++ too.
        let _ = (thread, cpu_id);
    }
}

// ---------------------------------------------------------------------------
// thread_model_manager.h — CTP_THREAD_MODEL
// ---------------------------------------------------------------------------

/// Backing store for [`thread_model()`]. `OnceLock`, never `thread_local!`
/// (project rule): one instance per process, exactly like `CrossSingleton`.
static THREAD_MODEL: OnceLock<StdThread> = OnceLock::new();

/// `CTP_THREAD_MODEL_T` — the pointer type the macro yields.
pub type ThreadModelT = &'static StdThread;

/// `CTP_THREAD_MODEL` — the process-wide thread model singleton.
///
/// C++ selects the concrete type at build time via `CTP_DEFAULT_THREAD_MODEL`
/// (`Pthread` when pthreads are enabled, else `StdThread`) and
/// `CTP_DEFAULT_THREAD_MODEL_GPU` (`Cuda`/`Rocm`/`StdThread`). Only
/// [`StdThread`] exists here, so this always returns it — see divergences 1–2.
#[inline]
pub fn thread_model() -> ThreadModelT {
    THREAD_MODEL.get_or_init(StdThread::new)
}

/// `CTP_THREAD_MODEL->BusyWait(cond)` — the free-function spelling used by the
/// C++ call sites. See [`ThreadModel::busy_wait`].
#[inline]
pub fn busy_wait<F: FnMut() -> bool>(mut cond: F) {
    thread_model().busy_wait(&mut cond);
}

// ---------------------------------------------------------------------------
// SystemInfo pass-throughs, reimplemented locally (divergence 10)
// ---------------------------------------------------------------------------

/// `SystemInfo::GetTid()`, without the C++ narrowing through `int`
/// (divergence 9).
fn sys_get_tid() -> u64 {
    #[cfg(target_os = "linux")]
    {
        // SAFETY: gettid() takes no arguments and cannot fail.
        unsafe { libc::gettid() as u32 as u64 }
    }
    #[cfg(target_os = "macos")]
    {
        // `pthread_threadid_np(nullptr, &tid)`: a null thread means "self".
        let mut tid: u64 = 0;
        // SAFETY: null selects the calling thread; `tid` is a live, writable u64.
        unsafe { libc::pthread_threadid_np(0, &mut tid) };
        tid
    }
    #[cfg(all(unix, not(any(target_os = "linux", target_os = "macos"))))]
    {
        // C++ falls back to GetPid() where SYS_gettid is unavailable.
        std::process::id() as u64
    }
    #[cfg(windows)]
    {
        // SAFETY: GetCurrentThreadId has no preconditions and cannot fail.
        unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() as u64 }
    }
}

/// Splits microseconds into the `(tv_sec, tv_nsec)` of `SleepForUs`'s
/// `timespec`. Extracted so the arithmetic is testable on every platform.
#[cfg_attr(not(unix), allow(dead_code))]
fn split_us(us: u64) -> (u64, u32) {
    (us / 1_000_000, ((us % 1_000_000) * 1_000) as u32)
}

/// `SetWaitableTimer`'s relative due time in 100 ns units: negative means
/// relative. Saturates rather than overflowing (divergence 12).
#[cfg_attr(not(windows), allow(dead_code))]
fn win_due_time_100ns(us: u64) -> i64 {
    let ticks = i64::try_from(us).unwrap_or(i64::MAX).saturating_mul(10);
    ticks.saturating_neg()
}

/// `SystemInfo::SleepForUs(us)`.
fn sys_sleep_for_us(us: u64) {
    if us == 0 {
        return;
    }
    #[cfg(unix)]
    {
        // POSIX: nanosleep already honors sub-microsecond requests (subject to
        // the hrtimer slack, ~50us by default). Like the C++, a signal-
        // interrupted sleep returns early rather than resuming the remainder.
        let (secs, nsecs) = split_us(us);
        let ts = libc::timespec {
            tv_sec: secs as libc::time_t,
            tv_nsec: nsecs as _,
        };
        // SAFETY: `ts` is a live, fully-initialized timespec; a null remainder
        // pointer is explicitly allowed by nanosleep(2).
        unsafe { libc::nanosleep(&ts, std::ptr::null_mut()) };
    }
    #[cfg(windows)]
    {
        use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
        use windows_sys::Win32::System::Threading::{
            SetWaitableTimer, Sleep, WaitForSingleObject, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
            INFINITE, TIMER_ALL_ACCESS,
        };

        // windows-sys gates every CreateWaitableTimer* behind its Win32_Security
        // feature (their first parameter is LPSECURITY_ATTRIBUTES), which is not
        // enabled for this crate and whose Cargo.toml this task may not edit.
        // The one import is therefore declared here. We only ever pass a null
        // attributes pointer, so `*const c_void` is ABI-identical to
        // `*const SECURITY_ATTRIBUTES` and nothing is lost. Drop this block in
        // favor of the crate's own import if Win32_Security is ever enabled.
        #[link(name = "kernel32")]
        extern "system" {
            /// `HANDLE CreateWaitableTimerExW(LPSECURITY_ATTRIBUTES, LPCWSTR,
            ///                                DWORD dwFlags, DWORD dwDesiredAccess)`
            fn CreateWaitableTimerExW(
                lptimerattributes: *const core::ffi::c_void,
                lptimername: *const u16,
                dwflags: u32,
                dwdesiredaccess: u32,
            ) -> HANDLE;
        }

        // One-shot high-resolution waitable timer. Unlike a plain Sleep (which
        // rounds up to the ~1-15.6ms global timer tick), this gives sub-ms
        // accuracy without timeBeginPeriod (no global timer rate change). The
        // timer is created/closed per call rather than cached in a thread_local
        // (project rule + per-DLL duplication hazard); CreateWaitableTimer
        // costs a few microseconds, dwarfed by any sleep worth issuing.
        //
        // SAFETY: null attributes/name are documented defaults; the returned
        // HANDLE is checked for null and closed on every path below.
        let mut timer = unsafe {
            CreateWaitableTimerExW(
                std::ptr::null(),
                std::ptr::null(),
                CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                TIMER_ALL_ACCESS,
            )
        };
        if timer.is_null() {
            // Pre-1803: high-resolution flag unsupported. Standard timer.
            // SAFETY: as above, with no flags.
            timer = unsafe {
                CreateWaitableTimerExW(std::ptr::null(), std::ptr::null(), 0, TIMER_ALL_ACCESS)
            };
        }
        if !timer.is_null() {
            let due = win_due_time_100ns(us);
            // SAFETY: `timer` is a live timer handle we own; `due` is a live
            // i64; a null completion routine + null arg means "no APC", and
            // WaitForSingleObject/CloseHandle receive that same valid handle.
            unsafe {
                if SetWaitableTimer(timer, &due, 0, None, std::ptr::null(), 0) != 0 {
                    WaitForSingleObject(timer, INFINITE);
                }
                CloseHandle(timer);
            }
            return;
        }
        // Last resort: coarse Sleep, rounded up to >= 1ms.
        let ms = us.div_ceil(1000).min(u32::MAX as u64) as u32;
        // SAFETY: Sleep has no preconditions.
        unsafe { Sleep(ms) };
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::{Cell, RefCell};
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    // -- ThreadType ---------------------------------------------------------

    #[test]
    fn thread_type_discriminants_match_cpp_enum_order() {
        assert_eq!(ThreadType::None as u32, 0);
        assert_eq!(ThreadType::Pthread as u32, 1);
        assert_eq!(ThreadType::Argobots as u32, 2);
        assert_eq!(ThreadType::Cuda as u32, 3);
        assert_eq!(ThreadType::Rocm as u32, 4);
        assert_eq!(ThreadType::StdThread as u32, 5);
        assert_eq!(ThreadType::default(), ThreadType::None);
    }

    #[test]
    fn std_thread_reports_its_type() {
        let model = StdThread::new();
        assert_eq!(model.get_type(), ThreadType::StdThread);
        assert_eq!(thread_model().get_type(), ThreadType::StdThread);
        // Init() is a no-op but must exist and be callable.
        model.init();
    }

    // -- ThreadId -----------------------------------------------------------

    #[test]
    fn thread_id_null_semantics() {
        assert_eq!(ThreadId::get_null(), ThreadId::NULL);
        assert_eq!(ThreadId::NULL.tid, u64::MAX);
        assert!(ThreadId::get_null().is_null());
        assert!(ThreadId::default().is_null()); // divergence 8
        assert!(!ThreadId::new(0).is_null());
        // u64::MAX - 1 is a normal id: only the exact sentinel is null.
        assert!(!ThreadId::new(u64::MAX - 1).is_null());

        let mut id = ThreadId::new(7);
        assert!(!id.is_null());
        id.set_null();
        assert!(id.is_null());
        assert_eq!(id, ThreadId::get_null());
    }

    #[test]
    fn thread_id_compare_and_display() {
        assert_eq!(ThreadId::new(5), ThreadId::new(5));
        assert_ne!(ThreadId::new(5), ThreadId::new(6));
        assert!(ThreadId::new(5) < ThreadId::new(6));
        assert!(ThreadId::new(6) > ThreadId::new(5));
        assert!(ThreadId::new(5) <= ThreadId::new(5));
        assert!(ThreadId::new(5) >= ThreadId::new(5));
        // Null sorts highest, being u64::MAX.
        assert!(ThreadId::new(u64::MAX - 1) < ThreadId::get_null());
        assert_eq!(ThreadId::new(42).to_string(), "42");
        assert_eq!(ThreadId::new(0).to_string(), "0");
    }

    // -- get_tid ------------------------------------------------------------

    #[test]
    fn get_tid_is_stable_within_a_thread_and_unique_across_threads() {
        let model = StdThread::new();
        let mine = model.get_tid();
        assert!(!mine.is_null());
        assert_eq!(mine, model.get_tid(), "tid must be stable within a thread");

        let group = model.create_thread_group(&ThreadGroupContext::default());
        let other = Arc::new(AtomicU64::new(0));
        let sink = Arc::clone(&other);
        let mut t = model.spawn(&group, move || {
            sink.store(StdThread::new().get_tid().tid, Ordering::SeqCst);
        });
        model.join(&mut t).expect("join");
        assert_ne!(
            other.load(Ordering::SeqCst),
            mine.tid,
            "distinct threads must report distinct tids"
        );
    }

    // -- sleep_for_us / yield_now -------------------------------------------

    #[test]
    fn sleep_for_zero_us_does_not_sleep() {
        let model = StdThread::new();
        let start = Instant::now();
        for _ in 0..1000 {
            model.sleep_for_us(0); // `if (us == 0) return;`
        }
        assert!(
            start.elapsed() < Duration::from_millis(200),
            "1000 zero-length sleeps should be nearly free, took {:?}",
            start.elapsed()
        );
    }

    #[test]
    fn sleep_for_us_waits_at_least_roughly_the_requested_time() {
        let model = StdThread::new();
        let start = Instant::now();
        model.sleep_for_us(20_000); // 20ms
                                    // Lower bound with slack: timers may fire marginally early, and the
                                    // point is that we do sleep rather than return instantly.
        assert!(
            start.elapsed() >= Duration::from_millis(10),
            "20ms sleep returned after {:?}",
            start.elapsed()
        );
    }

    #[test]
    fn yield_now_is_callable_and_cheap() {
        let model = StdThread::new();
        let start = Instant::now();
        for _ in 0..10_000 {
            model.yield_now();
        }
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "10k yields took {:?}",
            start.elapsed()
        );
    }

    #[test]
    fn split_us_boundaries() {
        assert_eq!(split_us(0), (0, 0));
        assert_eq!(split_us(1), (0, 1_000));
        assert_eq!(split_us(999_999), (0, 999_999_000)); // < 1s, max nsec
        assert_eq!(split_us(1_000_000), (1, 0)); // exact second boundary
        assert_eq!(split_us(1_500_000), (1, 500_000_000));
        assert_eq!(split_us(u64::MAX), (u64::MAX / 1_000_000, 551_615_000));
        // tv_nsec must always be a valid < 1e9 remainder.
        for us in [0u64, 1, 999_999, 1_000_000, 1_000_001, u64::MAX] {
            assert!(split_us(us).1 < 1_000_000_000, "bad tv_nsec for {us}");
        }
    }

    #[test]
    fn win_due_time_is_negative_and_saturates() {
        assert_eq!(win_due_time_100ns(0), 0);
        assert_eq!(win_due_time_100ns(1), -10);
        assert_eq!(win_due_time_100ns(1_000), -10_000); // 1ms
        assert_eq!(win_due_time_100ns(20_000), -200_000);
        // Divergence 12: must saturate, never panic on overflow. The floor is
        // -i64::MAX (saturating_mul caps at i64::MAX, then negates), not
        // i64::MIN — either is ~29,000 years, far past any real sleep.
        assert_eq!(win_due_time_100ns(u64::MAX), -i64::MAX);
        assert_eq!(win_due_time_100ns(i64::MAX as u64), -i64::MAX);
        assert_eq!(win_due_time_100ns(u64::MAX / 2), -i64::MAX);
        // The largest input that does not saturate still round-trips exactly.
        let exact = (i64::MAX / 10) as u64;
        assert_eq!(win_due_time_100ns(exact), -(exact as i64) * 10);
        // Every input must stay negative (relative due time) or zero.
        for us in [0u64, 1, 1_000, u64::MAX / 2, u64::MAX] {
            assert!(win_due_time_100ns(us) <= 0, "due time must be relative");
        }
    }

    // -- spawn / join -------------------------------------------------------

    #[test]
    fn spawn_runs_the_closure_and_join_waits_for_it() {
        let model = StdThread::new();
        let group = model.create_thread_group(&ThreadGroupContext::default());
        let flag = Arc::new(AtomicBool::new(false));
        let child = Arc::clone(&flag);
        let mut t = model.spawn(&group, move || {
            std::thread::sleep(Duration::from_millis(20));
            child.store(true, Ordering::SeqCst);
        });
        assert!(t.is_joinable());
        assert_eq!(t.group(), ThreadGroup);
        model.join(&mut t).expect("join should succeed");
        assert!(
            flag.load(Ordering::SeqCst),
            "join must wait for the closure"
        );
        assert!(!t.is_joinable(), "join clears the handle");
    }

    #[test]
    fn spawn_boxed_matches_spawn() {
        let model = StdThread::new();
        let group = ThreadGroup;
        let n = Arc::new(AtomicU64::new(0));
        let child = Arc::clone(&n);
        let mut t = model.spawn_boxed(
            &group,
            Box::new(move || {
                child.store(99, Ordering::SeqCst);
            }),
        );
        assert!(model.join(&mut t).is_ok());
        assert_eq!(n.load(Ordering::SeqCst), 99);
    }

    #[test]
    fn many_threads_all_run_and_join() {
        const N: u64 = 32;
        let model = StdThread::new();
        let group = model.create_thread_group(&ThreadGroupContext::default());
        let counter = Arc::new(AtomicU64::new(0));
        let mut threads: Vec<Thread> = (0..N)
            .map(|i| {
                let c = Arc::clone(&counter);
                model.spawn(&group, move || {
                    c.fetch_add(i, Ordering::SeqCst);
                })
            })
            .collect();
        for t in &mut threads {
            model.join(t).expect("join");
        }
        assert_eq!(counter.load(Ordering::SeqCst), (0..N).sum::<u64>());
    }

    #[test]
    fn join_of_a_non_joinable_thread_reports_not_joinable() {
        let model = StdThread::new();
        // Default-constructed: never spawned (C++ would throw system_error).
        let mut t = Thread::default();
        assert!(!t.is_joinable());
        assert_eq!(model.join(&mut t), Err(JoinFailure::NotJoinable));

        // Double join: the second must not block or panic.
        let mut t2 = model.spawn(&ThreadGroup, || {});
        assert!(model.join(&mut t2).is_ok());
        assert_eq!(model.join(&mut t2), Err(JoinFailure::NotJoinable));
    }

    #[test]
    fn join_of_a_panicking_thread_reports_panicked() {
        let model = StdThread::new();
        // Silence the child's panic output; this panic is the point of the test.
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let mut t = model.spawn(&ThreadGroup, || panic!("boom"));
        let res = model.join(&mut t);
        std::panic::set_hook(prev);
        // Divergence 5: C++ would std::terminate; we surface it instead.
        assert_eq!(res, Err(JoinFailure::Panicked));
        assert!(!t.is_joinable());
    }

    #[test]
    fn join_failure_displays() {
        assert_eq!(
            JoinFailure::NotJoinable.to_string(),
            "thread is not joinable"
        );
        assert_eq!(JoinFailure::Panicked.to_string(), "thread panicked");
    }

    // -- timed_join_or_detach (divergence 4) --------------------------------

    #[test]
    fn timed_join_succeeds_when_the_thread_finishes_in_time() {
        let model = StdThread::new();
        let done = Arc::new(AtomicBool::new(false));
        let child = Arc::clone(&done);
        let mut t = model.spawn(&ThreadGroup, move || {
            child.store(true, Ordering::SeqCst);
        });
        assert!(model.timed_join_or_detach(&mut t, 10_000));
        assert!(done.load(Ordering::SeqCst));
        assert!(!t.is_joinable());
    }

    #[test]
    fn timed_join_on_a_non_joinable_thread_returns_true() {
        let model = StdThread::new();
        let mut t = Thread::default();
        assert!(model.timed_join_or_detach(&mut t, 0));
        assert!(model.timed_join_or_detach(&mut t, 1000));
    }

    #[test]
    fn timed_join_detaches_when_the_timeout_elapses() {
        let model = StdThread::new();
        let release = Arc::new(AtomicBool::new(false));
        let child = Arc::clone(&release);
        let observed = Arc::new(AtomicBool::new(false));
        let child_observed = Arc::clone(&observed);
        let mut t = model.spawn(&ThreadGroup, move || {
            while !child.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(1));
            }
            child_observed.store(true, Ordering::SeqCst);
        });

        let start = Instant::now();
        assert!(
            !model.timed_join_or_detach(&mut t, 50),
            "a thread that outlives the timeout must report detached"
        );
        let waited = start.elapsed();
        assert!(
            waited >= Duration::from_millis(25),
            "returned too early: {waited:?}"
        );
        assert!(
            waited < Duration::from_secs(5),
            "must not block past the timeout: {waited:?}"
        );
        // Handle cleared (C++ zeroes pthread_thread_), so a retry says true.
        assert!(!t.is_joinable());
        assert!(model.timed_join_or_detach(&mut t, 0));

        // The detached thread still runs; let it finish so it can't outlive
        // the test process holding an Arc.
        release.store(true, Ordering::SeqCst);
        let deadline = Instant::now() + Duration::from_secs(5);
        while !observed.load(Ordering::SeqCst) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(
            observed.load(Ordering::SeqCst),
            "detached thread should finish"
        );
    }

    #[test]
    fn timed_join_with_zero_timeout_polls_once() {
        let model = StdThread::new();
        let release = Arc::new(AtomicBool::new(false));
        let child = Arc::clone(&release);
        let mut t = model.spawn(&ThreadGroup, move || {
            while !child.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(1));
            }
        });
        let start = Instant::now();
        // Running thread + 0ms deadline => immediate detach.
        assert!(!model.timed_join_or_detach(&mut t, 0));
        assert!(
            start.elapsed() < Duration::from_secs(2),
            "zero timeout must not block: {:?}",
            start.elapsed()
        );
        release.store(true, Ordering::SeqCst);
    }

    #[test]
    fn timed_join_of_a_panicking_thread_still_completes() {
        // SignalOnDrop must fire on unwind, else this would time out.
        let model = StdThread::new();
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let mut t = model.spawn(&ThreadGroup, || panic!("boom"));
        let joined = model.timed_join_or_detach(&mut t, 10_000);
        std::panic::set_hook(prev);
        assert!(joined, "a panicking thread is still a finished thread");
    }

    #[test]
    fn set_affinity_is_a_noop_but_callable() {
        let model = StdThread::new();
        let mut t = model.spawn(&ThreadGroup, || {});
        model.set_affinity(&mut t, 0);
        model.set_affinity(&mut t, -1);
        assert!(model.join(&mut t).is_ok());
    }

    // -- singleton ----------------------------------------------------------

    #[test]
    fn thread_model_singleton_is_stable() {
        let a: ThreadModelT = thread_model();
        let b: ThreadModelT = thread_model();
        assert!(std::ptr::eq(a, b), "CrossSingleton returns one instance");
        assert_eq!(a.get_type(), ThreadType::StdThread);
    }

    // -- busy_wait ----------------------------------------------------------

    /// Recording model: lets the ported BusyWait policy be asserted exactly,
    /// without waiting real microseconds. Single-threaded use only.
    struct MockModel {
        yields: Cell<u64>,
        sleeps: RefCell<Vec<u64>>,
    }

    impl MockModel {
        fn new() -> Self {
            Self {
                yields: Cell::new(0),
                sleeps: RefCell::new(Vec::new()),
            }
        }
    }

    impl ThreadModel for MockModel {
        fn get_type(&self) -> ThreadType {
            ThreadType::None
        }
        fn yield_now(&self) {
            self.yields.set(self.yields.get() + 1);
        }
        fn sleep_for_us(&self, us: u64) {
            self.sleeps.borrow_mut().push(us); // recorded, not slept
        }
        fn get_tid(&self) -> ThreadId {
            ThreadId::get_null()
        }
        fn create_thread_group(&self, _ctx: &ThreadGroupContext) -> ThreadGroup {
            ThreadGroup
        }
        fn spawn_boxed(
            &self,
            _group: &ThreadGroup,
            _func: Box<dyn FnOnce() + Send + 'static>,
        ) -> Thread {
            unimplemented!("MockModel only exercises the BusyWait policy")
        }
        fn join(&self, _thread: &mut Thread) -> Result<(), JoinFailure> {
            unimplemented!()
        }
        fn timed_join_or_detach(&self, _thread: &mut Thread, _timeout_ms: u64) -> bool {
            unimplemented!()
        }
        fn set_affinity(&self, _thread: &mut Thread, _cpu_id: i32) {}
    }

    #[test]
    fn busy_wait_returns_immediately_when_the_condition_already_holds() {
        let mock = MockModel::new();
        let mut calls = 0u32;
        let start = Instant::now();
        mock.busy_wait(&mut || {
            calls += 1;
            true
        });
        // cond is checked before the first yield: no yield, no sleep, one call.
        assert_eq!(calls, 1, "cond must be evaluated exactly once");
        assert_eq!(mock.yields.get(), 0, "must not touch the scheduler");
        assert!(mock.sleeps.borrow().is_empty());
        assert!(start.elapsed() < Duration::from_millis(500));
    }

    #[test]
    fn busy_wait_yield_spins_before_sleeping() {
        let mock = MockModel::new();
        let mut calls = 0u32;
        // False twice, then true: still inside the 5ms yield window.
        mock.busy_wait(&mut || {
            calls += 1;
            calls > 2
        });
        assert_eq!(calls, 3);
        assert_eq!(
            mock.yields.get(),
            2,
            "one yield per failed check in phase 1"
        );
        assert!(
            mock.sleeps.borrow().is_empty(),
            "phase 2 must not start inside the yield window"
        );
    }

    #[test]
    fn busy_wait_backoff_doubles_then_caps() {
        let mock = MockModel::new();
        // Never satisfied during phase 1 (~5ms of yield-spin), then let phase 2
        // run 14 sleeps so the cap is exercised past 1024us.
        let mock_ref = &mock;
        mock.busy_wait(&mut || mock_ref.sleeps.borrow().len() >= 14);
        let sleeps = mock.sleeps.borrow();
        assert_eq!(
            *sleeps,
            vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1024, 1024, 1024],
            "1,2,4.. doubling capped at kMaxSleepUs=1024"
        );
        assert!(mock.yields.get() > 0, "phase 1 must have yield-spun");
    }

    #[test]
    fn busy_wait_on_the_real_model_waits_for_another_thread() {
        let model = StdThread::new();
        let flag = Arc::new(AtomicBool::new(false));
        let child = Arc::clone(&flag);
        let mut t = model.spawn(&ThreadGroup, move || {
            std::thread::sleep(Duration::from_millis(30));
            child.store(true, Ordering::SeqCst);
        });
        let observed = Arc::clone(&flag);
        let start = Instant::now();
        // >5ms, so this crosses from phase 1 into the phase-2 backoff.
        model.busy_wait(&mut || observed.load(Ordering::SeqCst));
        let waited = start.elapsed();
        assert!(
            flag.load(Ordering::SeqCst),
            "must not return before cond holds"
        );
        assert!(
            waited >= Duration::from_millis(15),
            "returned early: {waited:?}"
        );
        assert!(
            waited < Duration::from_secs(10),
            "took too long: {waited:?}"
        );
        model.join(&mut t).expect("join");
    }

    #[test]
    fn free_busy_wait_drives_the_singleton() {
        let flag = Arc::new(AtomicBool::new(true));
        let observed = Arc::clone(&flag);
        // Mirrors the C++ CTP_THREAD_MODEL->BusyWait(cond) call sites.
        busy_wait(|| observed.load(Ordering::SeqCst));
    }
}
