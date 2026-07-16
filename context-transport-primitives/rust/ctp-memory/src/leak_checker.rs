// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).
//
//! Rust port of `clio_ctp/memory/allocator/leak_checker.h` — the
//! process-global registry that an allocator's *destructor* reports to when
//! it is torn down while still holding allocated-but-not-freed bytes. This
//! is the accounting side of the leak-check CI job
//! (`CLIO_CORE_ENABLE_LEAK_CHECK` / `CTP_ALLOC_TRACK_SIZE`).
//!
//! As in C++, this checker is one of two complementary facilities: it catches
//! every allocator whose destructor actually runs (the process-lifetime
//! singletons such as `CTP_MALLOC`, plus any stack/heap allocator), but it
//! cannot observe placement-constructed shared-memory allocators whose
//! destructors never fire — those stay the job of the runtime's own shutdown
//! scan (`IpcManager::ReportRuntimeLeaks`).
//!
//! # C++ → Rust name mapping
//!
//! | C++ (`ctp::ipc`) | Rust (`ctp_memory::leak_checker`) |
//! |---|---|
//! | `AllocatorLeakChecker` | [`AllocatorLeakChecker`] |
//! | `AllocatorLeakChecker::Entry` | [`Entry`] |
//! | `Entry::label` (`std::string`) | [`Entry::label`] (`String`) |
//! | `Entry::bytes` (`size_t`) | [`Entry::bytes`] (`usize`) |
//! | `AllocatorLeakChecker::Get()` | [`AllocatorLeakChecker::get`] |
//! | `Report(const char *label, size_t bytes)` | [`AllocatorLeakChecker::report`] |
//! | `TotalLeakedBytes()` | [`AllocatorLeakChecker::total_leaked_bytes`] |
//! | `LeakCount()` | [`AllocatorLeakChecker::leak_count`] |
//! | `Entries()` | [`AllocatorLeakChecker::entries`] |
//! | `Reset()` | [`AllocatorLeakChecker::reset`] |
//! | `"<unknown>"` (nullptr label) | [`UNKNOWN_LABEL`] |
//! | `std::mutex mu_` + `entries_`/`total_bytes_` | one private `Mutex<State>` |
//!
//! Preserved semantics: zero-byte reports are ignored (a clean allocator is
//! not a leak); a null/absent label records `"<unknown>"`; every accepted
//! report is emitted to the log immediately, under the lock; entries keep
//! insertion order and are snapshotted by copy; `leak_count` counts *reports*,
//! so an allocator reporting twice counts twice.
//!
//! # Semantic divergences from the C++
//!
//! 1. **Nullable label → `Option<&str>`.** C++ takes `const char *label` and
//!    maps `nullptr` to `"<unknown>"`. Rust `&str` cannot be null, so
//!    [`report`](AllocatorLeakChecker::report) takes `Option<&str>`: `None` is
//!    exactly the C++ `nullptr` case. An empty `Some("")` is recorded verbatim,
//!    matching C++ (which only special-cases `nullptr`, not `""`).
//! 2. **Total saturates instead of wrapping.** C++ `total_bytes_ += bytes` on
//!    `size_t` wraps modulo 2^64; Rust would panic in debug builds. The total
//!    uses `saturating_add`, so an absurd accumulation pins at `usize::MAX`
//!    rather than wrapping to a small (misleading) number. Per-entry byte
//!    counts are always recorded verbatim, so no information is lost.
//! 3. **Logging goes to stderr.** C++ emits `HLOG(kError, ...)`. `ctp-memory`
//!    does not depend on `ctp-util` (the crate's `Cargo.toml` is fixed for this
//!    port), so the identical message text is written with `eprintln!`. Same
//!    format string, same one-line-per-report behaviour.
//! 4. **No mutex poisoning.** `std::mutex` has no poisoning concept; a panic
//!    while a Rust `Mutex` is held would otherwise make the checker permanently
//!    unusable. Lock acquisition recovers the guard (`into_inner`), matching
//!    C++ availability. Recorded state stays consistent because every critical
//!    section is panic-free.
//! 5. **Public constructor.** The C++ ctor is private (singleton-only). Rust
//!    exposes [`AllocatorLeakChecker::new`] so tests can accumulate against an
//!    isolated instance instead of racing on process-global state. The
//!    process-global instance is still reached only through
//!    [`get`](AllocatorLeakChecker::get).
//! 6. **Singleton lifetime.** C++ deliberately heap-leaks the instance so that
//!    allocator destructors firing during static teardown always reach a live
//!    object. Rust `static`s are never dropped, so a plain `OnceLock` gives the
//!    same guarantee without the leak.
//! 7. **No `CTP_IS_HOST` / device split and no `CTP_ALLOC_TRACK_SIZE` gate.**
//!    The C++ header compiles to nothing in device translation units and its
//!    callers are gated on the tracking macro. This crate is host-only and has
//!    no such feature flag; the type is always compiled (as it is on C++ host,
//!    so tests can query it unconditionally) and callers decide when to report.

use std::sync::{Mutex, MutexGuard, OnceLock};

/// Label recorded when the reporter passes no name (C++ `nullptr` label).
pub const UNKNOWN_LABEL: &str = "<unknown>";

/// One recorded leak: which allocator, and how many bytes it still held.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    /// Reporter-supplied allocator name, or [`UNKNOWN_LABEL`].
    pub label: String,
    /// Bytes the allocator still held when it was torn down.
    pub bytes: usize,
}

/// Mutex-guarded interior state (C++ `entries_` + `total_bytes_` under `mu_`).
#[derive(Debug, Default)]
struct State {
    entries: Vec<Entry>,
    total_bytes: usize,
}

/// Process-global registry that an allocator's teardown reports to when it is
/// destroyed with a non-empty heap.
///
/// Reports only carry meaning in leak-tracking builds: allocators only call
/// [`report`](Self::report) when their outstanding-byte counter is real.
///
/// ```
/// use ctp_memory::leak_checker::AllocatorLeakChecker;
///
/// let checker = AllocatorLeakChecker::new();
/// checker.report(Some("MallocAllocator"), 128);
/// checker.report(None, 0); // clean allocator: ignored
/// assert_eq!(checker.leak_count(), 1);
/// assert_eq!(checker.total_leaked_bytes(), 128);
/// ```
#[derive(Debug, Default)]
pub struct AllocatorLeakChecker {
    state: Mutex<State>,
}

impl AllocatorLeakChecker {
    /// Create an independent checker (see divergence 5). Most code wants
    /// [`get`](Self::get).
    pub fn new() -> Self {
        Self::default()
    }

    /// Access the process-global instance (C++ `AllocatorLeakChecker::Get()`).
    ///
    /// The instance lives for the whole process and is never dropped, so
    /// teardown paths can always reach it regardless of destruction order.
    pub fn get() -> &'static AllocatorLeakChecker {
        static INSTANCE: OnceLock<AllocatorLeakChecker> = OnceLock::new();
        INSTANCE.get_or_init(AllocatorLeakChecker::new)
    }

    /// Lock the state, recovering from poisoning (divergence 4).
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Record that the allocator identified by `label` was destroyed while
    /// still holding `bytes`.
    ///
    /// Zero-byte reports are ignored (a clean allocator is not a leak). A
    /// `None` label is recorded as [`UNKNOWN_LABEL`]. Every accepted report is
    /// also emitted immediately to the log.
    pub fn report(&self, label: Option<&str>, bytes: usize) {
        if bytes == 0 {
            return;
        }
        let name = label.unwrap_or(UNKNOWN_LABEL);
        // C++ logs inside the lock_guard; keeping it there serializes the
        // report lines against concurrent reporters.
        let mut state = self.lock();
        state.entries.push(Entry {
            label: name.to_string(),
            bytes,
        });
        // Divergence 2: saturating, not wrapping.
        state.total_bytes = state.total_bytes.saturating_add(bytes);
        eprintln!("[leak][allocator] '{name}' destroyed with {bytes} bytes still allocated");
    }

    /// Total outstanding bytes across all recorded leaks.
    pub fn total_leaked_bytes(&self) -> usize {
        self.lock().total_bytes
    }

    /// Number of leak reports recorded (an allocator reporting twice counts
    /// twice — C++ `entries_.size()`).
    pub fn leak_count(&self) -> usize {
        self.lock().entries.len()
    }

    /// Snapshot of the recorded leaks, copied under the lock and in insertion
    /// order.
    pub fn entries(&self) -> Vec<Entry> {
        self.lock().entries.clone()
    }

    /// Clear all recorded leaks (mainly for tests that check specific phases).
    pub fn reset(&self) {
        let mut state = self.lock();
        state.entries.clear();
        state.total_bytes = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn fresh_checker_is_empty() {
        let checker = AllocatorLeakChecker::new();
        assert_eq!(checker.total_leaked_bytes(), 0);
        assert_eq!(checker.leak_count(), 0);
        assert!(checker.entries().is_empty());
    }

    #[test]
    fn zero_byte_reports_are_ignored() {
        let checker = AllocatorLeakChecker::new();
        checker.report(Some("CleanAllocator"), 0);
        checker.report(None, 0);
        assert_eq!(checker.leak_count(), 0);
        assert_eq!(checker.total_leaked_bytes(), 0);
        assert!(checker.entries().is_empty());
    }

    #[test]
    fn smallest_non_zero_report_is_recorded() {
        // The 0/1 boundary is the whole accept/reject rule.
        let checker = AllocatorLeakChecker::new();
        checker.report(Some("OneByte"), 1);
        assert_eq!(checker.leak_count(), 1);
        assert_eq!(checker.total_leaked_bytes(), 1);
    }

    #[test]
    fn absent_label_records_unknown() {
        let checker = AllocatorLeakChecker::new();
        checker.report(None, 8);
        assert_eq!(
            checker.entries(),
            vec![Entry {
                label: UNKNOWN_LABEL.to_string(),
                bytes: 8,
            }]
        );
    }

    #[test]
    fn empty_label_is_recorded_verbatim() {
        // C++ only special-cases nullptr, never the empty string.
        let checker = AllocatorLeakChecker::new();
        checker.report(Some(""), 4);
        assert_eq!(checker.entries()[0].label, "");
    }

    #[test]
    fn reports_accumulate_in_order() {
        let checker = AllocatorLeakChecker::new();
        checker.report(Some("MallocAllocator"), 128);
        checker.report(Some("StackAllocator"), 32);
        checker.report(Some("MallocAllocator"), 8); // same label twice
        assert_eq!(checker.total_leaked_bytes(), 168);
        assert_eq!(checker.leak_count(), 3);
        let entries = checker.entries();
        assert_eq!(
            entries.iter().map(|e| e.label.as_str()).collect::<Vec<_>>(),
            vec!["MallocAllocator", "StackAllocator", "MallocAllocator"]
        );
        assert_eq!(
            entries.iter().map(|e| e.bytes).collect::<Vec<_>>(),
            vec![128, 32, 8]
        );
    }

    #[test]
    fn entries_are_a_snapshot_not_a_view() {
        let checker = AllocatorLeakChecker::new();
        checker.report(Some("A"), 1);
        let snapshot = checker.entries();
        checker.report(Some("B"), 2);
        checker.reset();
        assert_eq!(snapshot.len(), 1);
        assert_eq!(snapshot[0].label, "A");
    }

    #[test]
    fn reset_clears_everything_and_allows_reuse() {
        let checker = AllocatorLeakChecker::new();
        checker.report(Some("A"), 64);
        checker.reset();
        assert_eq!(checker.total_leaked_bytes(), 0);
        assert_eq!(checker.leak_count(), 0);
        assert!(checker.entries().is_empty());
        // Reset on an already-empty checker is a no-op.
        checker.reset();
        assert_eq!(checker.leak_count(), 0);
        // And the checker still works afterwards.
        checker.report(Some("B"), 2);
        assert_eq!(checker.total_leaked_bytes(), 2);
        assert_eq!(checker.leak_count(), 1);
    }

    #[test]
    fn total_saturates_instead_of_wrapping() {
        // Divergence 2: C++ would wrap size_t to 0 here; we pin at usize::MAX.
        let checker = AllocatorLeakChecker::new();
        checker.report(Some("Huge"), usize::MAX);
        checker.report(Some("OneMore"), 1);
        assert_eq!(checker.total_leaked_bytes(), usize::MAX);
        assert_eq!(checker.leak_count(), 2);
        // Per-entry bytes are never clamped.
        let entries = checker.entries();
        assert_eq!(entries[0].bytes, usize::MAX);
        assert_eq!(entries[1].bytes, 1);
        // Saturation is sticky, and reset restores an exact total.
        checker.report(Some("AndAnother"), usize::MAX);
        assert_eq!(checker.total_leaked_bytes(), usize::MAX);
        checker.reset();
        assert_eq!(checker.total_leaked_bytes(), 0);
    }

    #[test]
    fn concurrent_reports_are_all_recorded() {
        let checker = Arc::new(AllocatorLeakChecker::new());
        let threads: Vec<_> = (0..8)
            .map(|t| {
                let checker = Arc::clone(&checker);
                thread::spawn(move || {
                    for _ in 0..100 {
                        checker.report(Some("Racer"), t + 1);
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().expect("reporter thread panicked");
        }
        assert_eq!(checker.leak_count(), 800);
        // sum over t in 0..8 of 100 * (t + 1)
        assert_eq!(
            checker.total_leaked_bytes(),
            100 * (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8)
        );
        assert!(checker.entries().iter().all(|e| e.label == "Racer"));
    }

    #[test]
    fn lock_survives_a_panicking_reader() {
        // Divergence 4: std::mutex does not poison; neither do we, in effect.
        let checker = Arc::new(AllocatorLeakChecker::new());
        checker.report(Some("Before"), 16);
        let poisoner = Arc::clone(&checker);
        let _ = thread::spawn(move || {
            let _guard = poisoner.lock();
            panic!("poison the mutex");
        })
        .join();
        // Still usable, and prior state intact.
        assert_eq!(checker.total_leaked_bytes(), 16);
        checker.report(Some("After"), 4);
        assert_eq!(checker.total_leaked_bytes(), 20);
        assert_eq!(checker.leak_count(), 2);
    }

    #[test]
    fn global_instance_is_a_stable_singleton() {
        // Only this test touches the process-global instance; it asserts
        // identity and its own label rather than global counts, so it cannot
        // race with (future) reporters in other tests.
        let a: &'static AllocatorLeakChecker = AllocatorLeakChecker::get();
        let b: &'static AllocatorLeakChecker = AllocatorLeakChecker::get();
        assert!(std::ptr::eq(a, b));
        a.report(Some("SingletonProbe"), 42);
        let hits: Vec<_> = b
            .entries()
            .into_iter()
            .filter(|e| e.label == "SingletonProbe")
            .collect();
        assert_eq!(
            hits,
            vec![Entry {
                label: "SingletonProbe".to_string(),
                bytes: 42
            }]
        );
        assert!(b.total_leaked_bytes() >= 42);
    }
}
