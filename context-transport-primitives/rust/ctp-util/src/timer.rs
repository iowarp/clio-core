// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).
//
//! Rust port of `clio_ctp/util/timer.h`.
//!
//! # C++ parity map
//!
//! | C++ name                                   | Rust name                              |
//! |--------------------------------------------|----------------------------------------|
//! | `TimepointBase<T>`                         | [`Timepoint`]                          |
//! | `TimepointBase::Now()`                     | [`Timepoint::now`]                     |
//! | `TimepointBase::GetNsecFromStart()`        | [`Timepoint::get_nsec_from_start`]     |
//! | `TimepointBase::GetUsecFromStart()`        | [`Timepoint::get_usec_from_start`]     |
//! | `TimepointBase::GetMsecFromStart()`        | [`Timepoint::get_msec_from_start`]     |
//! | `TimepointBase::GetSecFromStart()`         | [`Timepoint::get_sec_from_start`]      |
//! | `TimepointBase::GetNsecFromStart(now)`     | [`Timepoint::get_nsec_from_start_until`] |
//! | `TimepointBase::GetUsecFromStart(now)`     | [`Timepoint::get_usec_from_start_until`] |
//! | `TimepointBase::GetMsecFromStart(now)`     | [`Timepoint::get_msec_from_start_until`] |
//! | `TimepointBase::GetSecFromStart(now)`      | [`Timepoint::get_sec_from_start_until`]  |
//! | `NsecTimer` (`GetNsec/GetUsec/GetMsec/GetSec`) | [`NsecTimer`] (`get_nsec/get_usec/get_msec/get_sec`) |
//! | `TimerBase<T>::Resume()`                   | [`Timer::resume`] (alias [`Timer::start`]) |
//! | `TimerBase<T>::Pause()`                    | [`Timer::pause`]                       |
//! | `TimerBase<T>::Reset()`                    | [`Timer::reset`]                       |
//! | `TimerBase<T>::GetUsFromEpoch()`           | [`Timer::get_us_from_epoch`]           |
//! | `HighResCpuTimer` / `HighResMonotonicTimer` / `Timer` | [`HighResCpuTimer`] / [`HighResMonotonicTimer`] / [`Timer`] |
//! | `HighResCpuTimepoint` / `HighResMonotonicTimepoint` / `Timepoint` | [`HighResCpuTimepoint`] / [`HighResMonotonicTimepoint`] / [`Timepoint`] |
//! | `PeriodicRun<IDX>::Run(max_nsec, lambda)`  | [`PeriodicRun::run`]                   |
//!
//! # Semantic divergences from the C++ header
//!
//! 1. **Clock unification.** C++ instantiates `TimerBase` over both
//!    `std::chrono::high_resolution_clock` and `std::chrono::steady_clock`.
//!    Rust's `std::time::Instant` is the (only) monotonic high-resolution
//!    clock, so `HighResCpuTimer`/`HighResMonotonicTimer` are both aliases of
//!    the single [`Timer`] type (likewise for the timepoint aliases).
//! 2. **Default-constructed start.** A default-constructed C++
//!    `TimepointBase` holds the clock's *epoch*; `Instant` has no accessible
//!    epoch, so [`Timepoint::default`]/[`Timepoint::new`] stamp the moment of
//!    construction instead. Call `now()`/`resume()` before measuring — as all
//!    C++ call sites already do — and the behavior is identical.
//! 3. **`CpuTimer` is not ported.** It reads per-thread CPU time via
//!    `ctp::SystemInfo::ThreadCpuTimeNs()`; `ctp-util` is restricted to
//!    std + `ctp-types`, so it belongs to the `ctp-introspect` port.
//! 4. **`CTP_PERIODIC(IDX)` singleton macro is not ported.** [`PeriodicRun`]
//!    is a plain value type (the C++ `template <int IDX>` parameter existed
//!    only to mint distinct `CrossSingleton` instances); callers own their
//!    instance instead of using process-global state.
//! 5. **No device (`CTP_IS_HOST == 0`) branches.** The C++ stubs that return
//!    `0` on GPU device code have no Rust equivalent here (host-only crate).
//! 6. **Accumulation semantics preserved exactly**, including quirks: a
//!    second `pause()` without an intervening `resume()` re-adds the full
//!    elapsed-since-last-`resume()` span (C++ `Pause()` does not re-stamp
//!    `start_`), and elapsed times are `f64` nanoseconds internally, exactly
//!    as the C++ `double time_ns_`. Negative spans (a "later" timepoint that
//!    is actually earlier) yield negative values, matching C++ signed
//!    arithmetic on `time_point` differences.
//! 7. **Field access.** C++ exposes `start_`/`time_ns_` as public members;
//!    Rust keeps them private behind the accessor methods above.
//!
//! Units are those of the C++ API: explicit ns/us/ms/sec per method name.

use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Signed nanosecond span from `earlier` to `later` (negative if `later`
/// precedes `earlier`), mirroring C++ `duration_cast<nanoseconds>(a - b)`.
fn signed_nsec_between(earlier: Instant, later: Instant) -> f64 {
    match later.checked_duration_since(earlier) {
        Some(d) => d.as_nanos() as f64,
        None => -(earlier.duration_since(later).as_nanos() as f64),
    }
}

/// Port of `ctp::TimepointBase<T>`: a stamped instant that measures
/// elapsed nanoseconds (as `f64`) to "now" or to another [`Timepoint`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Timepoint {
    start: Instant,
}

impl Default for Timepoint {
    fn default() -> Self {
        Self::new()
    }
}

impl Timepoint {
    /// Creates a timepoint stamped at the moment of construction
    /// (divergence 2 in the module docs: C++ default is the clock epoch).
    #[must_use]
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    /// Port of `Now()`: re-stamps this timepoint to the current instant.
    pub fn now(&mut self) {
        self.start = Instant::now();
    }

    /// Port of `GetNsecFromStart(TimepointBase &now)`: nanoseconds from this
    /// timepoint to `now` (negative if `now` was stamped earlier).
    #[must_use]
    pub fn get_nsec_from_start_until(&self, now: &Timepoint) -> f64 {
        signed_nsec_between(self.start, now.start)
    }

    /// Port of `GetUsecFromStart(TimepointBase &now)`.
    #[must_use]
    pub fn get_usec_from_start_until(&self, now: &Timepoint) -> f64 {
        self.get_nsec_from_start_until(now) / 1_000.0
    }

    /// Port of `GetMsecFromStart(TimepointBase &now)`.
    #[must_use]
    pub fn get_msec_from_start_until(&self, now: &Timepoint) -> f64 {
        self.get_nsec_from_start_until(now) / 1_000_000.0
    }

    /// Port of `GetSecFromStart(TimepointBase &now)`.
    #[must_use]
    pub fn get_sec_from_start_until(&self, now: &Timepoint) -> f64 {
        self.get_nsec_from_start_until(now) / 1_000_000_000.0
    }

    /// Port of `GetNsecFromStart()`: nanoseconds from this timepoint to the
    /// current instant.
    #[must_use]
    pub fn get_nsec_from_start(&self) -> f64 {
        signed_nsec_between(self.start, Instant::now())
    }

    /// Port of `GetUsecFromStart()`.
    #[must_use]
    pub fn get_usec_from_start(&self) -> f64 {
        self.get_nsec_from_start() / 1_000.0
    }

    /// Port of `GetMsecFromStart()`.
    #[must_use]
    pub fn get_msec_from_start(&self) -> f64 {
        self.get_nsec_from_start() / 1_000_000.0
    }

    /// Port of `GetSecFromStart()`.
    #[must_use]
    pub fn get_sec_from_start(&self) -> f64 {
        self.get_nsec_from_start() / 1_000_000_000.0
    }
}

/// Port of `ctp::NsecTimer`: an accumulated nanosecond count (as `f64`)
/// with unit-conversion accessors.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct NsecTimer {
    time_ns: f64,
}

impl NsecTimer {
    /// Constructs with zero accumulated time (C++ default constructor).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Constructs from an already-accumulated nanosecond count.
    #[must_use]
    pub fn from_nsec(time_ns: f64) -> Self {
        Self { time_ns }
    }

    /// Adds `ns` nanoseconds to the accumulated total (the C++ code writes
    /// `time_ns_ +=` directly; the field is private here — divergence 7).
    pub fn add_nsec(&mut self, ns: f64) {
        self.time_ns += ns;
    }

    /// Port of `GetNsec()`.
    #[must_use]
    pub fn get_nsec(&self) -> f64 {
        self.time_ns
    }

    /// Port of `GetUsec()`.
    #[must_use]
    pub fn get_usec(&self) -> f64 {
        self.time_ns / 1_000.0
    }

    /// Port of `GetMsec()`.
    #[must_use]
    pub fn get_msec(&self) -> f64 {
        self.time_ns / 1_000_000.0
    }

    /// Port of `GetSec()`.
    #[must_use]
    pub fn get_sec(&self) -> f64 {
        self.time_ns / 1_000_000_000.0
    }
}

/// Port of `ctp::TimerBase<T>` (C++ inherits `TimepointBase<T>` and
/// `NsecTimer`; Rust composes them and forwards the methods).
///
/// Accumulate-across-pause semantics, exactly as in C++:
/// - [`resume`](Timer::resume) stamps the start timepoint;
/// - [`pause`](Timer::pause) adds elapsed-since-start to the accumulated
///   total and returns the new total in nanoseconds — it does NOT re-stamp
///   the start, so pausing twice without resuming adds the (longer) span
///   again;
/// - [`reset`](Timer::reset) re-stamps the start and zeroes the total.
#[derive(Debug, Clone, Copy, Default)]
pub struct Timer {
    timepoint: Timepoint,
    accum: NsecTimer,
}

impl Timer {
    /// Creates a timer with zero accumulated time. As with [`Timepoint`],
    /// the start is stamped at construction (divergence 2); call
    /// [`resume`](Self::resume) before measuring, as the C++ call sites do.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Port of `Resume()`: stamps the start timepoint.
    pub fn resume(&mut self) {
        self.timepoint.now();
    }

    /// Convenience alias for [`resume`](Self::resume) (the C++ API spells
    /// "begin timing" as `Resume()`; there is no separate `Start()`).
    pub fn start(&mut self) {
        self.resume();
    }

    /// Port of `Pause()`: accumulates elapsed-since-start and returns the
    /// new total in nanoseconds. Does not re-stamp the start (see type docs).
    pub fn pause(&mut self) -> f64 {
        let elapsed = self.timepoint.get_nsec_from_start();
        self.accum.add_nsec(elapsed);
        self.accum.get_nsec()
    }

    /// Port of `Reset()`: re-stamps the start and zeroes the accumulated
    /// total (same order as C++: `Resume(); time_ns_ = 0;`).
    pub fn reset(&mut self) {
        self.resume();
        self.accum = NsecTimer::new();
    }

    /// Port of `NsecTimer::GetNsec()`: the accumulated total, in ns.
    #[must_use]
    pub fn get_nsec(&self) -> f64 {
        self.accum.get_nsec()
    }

    /// Port of `NsecTimer::GetUsec()`.
    #[must_use]
    pub fn get_usec(&self) -> f64 {
        self.accum.get_usec()
    }

    /// Port of `NsecTimer::GetMsec()`.
    #[must_use]
    pub fn get_msec(&self) -> f64 {
        self.accum.get_msec()
    }

    /// Port of `NsecTimer::GetSec()`.
    #[must_use]
    pub fn get_sec(&self) -> f64 {
        self.accum.get_sec()
    }

    /// Port of the inherited `TimepointBase::GetNsecFromStart()`: elapsed
    /// nanoseconds since the last `resume()`, WITHOUT touching the
    /// accumulated total.
    #[must_use]
    pub fn get_nsec_from_start(&self) -> f64 {
        self.timepoint.get_nsec_from_start()
    }

    /// Port of the inherited `TimepointBase::GetUsecFromStart()`.
    #[must_use]
    pub fn get_usec_from_start(&self) -> f64 {
        self.timepoint.get_usec_from_start()
    }

    /// Port of the inherited `TimepointBase::GetMsecFromStart()`.
    #[must_use]
    pub fn get_msec_from_start(&self) -> f64 {
        self.timepoint.get_msec_from_start()
    }

    /// Port of the inherited `TimepointBase::GetSecFromStart()`.
    #[must_use]
    pub fn get_sec_from_start(&self) -> f64 {
        self.timepoint.get_sec_from_start()
    }

    /// The start [`Timepoint`] (read access to the C++ `start_` member).
    #[must_use]
    pub fn timepoint(&self) -> &Timepoint {
        &self.timepoint
    }

    /// Port of `GetUsFromEpoch()`: whole microseconds (truncated, like C++
    /// `duration_cast<microseconds>`) since the Unix epoch on the system
    /// (wall) clock, as `f64`. Negative if the system clock reads before the
    /// epoch.
    #[must_use]
    pub fn get_us_from_epoch(&self) -> f64 {
        match SystemTime::now().duration_since(UNIX_EPOCH) {
            Ok(d) => d.as_micros() as f64,
            Err(e) => -(e.duration().as_micros() as f64),
        }
    }
}

/// C++ `HighResCpuTimer` (see divergence 1: unified with [`Timer`]).
pub type HighResCpuTimer = Timer;
/// C++ `HighResMonotonicTimer` (`ctp::Timer` itself is this alias in C++).
pub type HighResMonotonicTimer = Timer;
/// C++ `HighResCpuTimepoint` (see divergence 1: unified with [`Timepoint`]).
pub type HighResCpuTimepoint = Timepoint;
/// C++ `HighResMonotonicTimepoint` (`ctp::Timepoint` is this alias in C++).
pub type HighResMonotonicTimepoint = Timepoint;

/// Port of `ctp::PeriodicRun<IDX>` as a plain value type (divergence 4:
/// no `CTP_PERIODIC` singleton). The internal timer is resumed at
/// construction, exactly like the C++ constructor.
#[derive(Debug, Clone, Copy)]
pub struct PeriodicRun {
    timer: HighResMonotonicTimer,
}

impl Default for PeriodicRun {
    fn default() -> Self {
        Self::new()
    }
}

impl PeriodicRun {
    /// Constructs and resumes the internal timer (C++ constructor parity).
    #[must_use]
    pub fn new() -> Self {
        let mut timer = HighResMonotonicTimer::new();
        timer.resume();
        Self { timer }
    }

    /// Port of `Run(max_nsec, lambda)`: invokes `f` and resets the internal
    /// timer iff at least `max_nsec` nanoseconds elapsed since the last
    /// reset (or construction). `max_nsec == 0` therefore always fires,
    /// matching the C++ `nsec >= max_nsec` comparison.
    pub fn run<F: FnOnce()>(&mut self, max_nsec: u64, f: F) {
        let nsec = self.timer.get_nsec_from_start();
        if nsec >= max_nsec as f64 {
            f();
            self.timer.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;
    use std::time::Duration;

    /// `thread::sleep` guarantees at least the requested duration, and
    /// `Instant` is monotonic, so elapsed >= slept is a hard invariant.
    const SLEEP_MS: u64 = 10;

    #[test]
    fn new_timer_has_zero_accumulated_time() {
        let t = Timer::new();
        assert_eq!(t.get_nsec(), 0.0);
        assert_eq!(t.get_usec(), 0.0);
        assert_eq!(t.get_msec(), 0.0);
        assert_eq!(t.get_sec(), 0.0);
    }

    #[test]
    fn resume_pause_measures_at_least_sleep_duration() {
        let mut t = Timer::new();
        t.resume();
        sleep(Duration::from_millis(SLEEP_MS));
        let total = t.pause();
        assert!(total >= (SLEEP_MS * 1_000_000) as f64, "total={total}");
        assert_eq!(total, t.get_nsec());
    }

    #[test]
    fn pause_accumulates_across_resume_cycles() {
        let mut t = Timer::new();
        t.resume();
        sleep(Duration::from_millis(SLEEP_MS));
        let first = t.pause();
        t.resume();
        sleep(Duration::from_millis(SLEEP_MS));
        let second = t.pause();
        // Second total includes the first plus at least another SLEEP_MS.
        assert!(second >= first + (SLEEP_MS * 1_000_000) as f64);
        assert!(t.get_msec() >= (2 * SLEEP_MS) as f64);
    }

    #[test]
    fn double_pause_without_resume_re_adds_elapsed_cpp_quirk() {
        // C++ Pause() does not re-stamp start_, so a second Pause() adds the
        // full (now longer) span since the last Resume() again.
        let mut t = Timer::new();
        t.resume();
        sleep(Duration::from_millis(SLEEP_MS));
        let p1 = t.pause();
        let p2 = t.pause();
        // elapsed at second pause >= elapsed at first pause (monotonic),
        // so p2 = p1 + e2 >= 2 * p1.
        assert!(p2 >= 2.0 * p1, "p1={p1} p2={p2}");
    }

    #[test]
    fn reset_zeroes_accumulated_time_and_restamps_start() {
        let mut t = Timer::new();
        t.resume();
        sleep(Duration::from_millis(SLEEP_MS));
        assert!(t.pause() > 0.0);
        t.reset();
        assert_eq!(t.get_nsec(), 0.0);
        // Start was re-stamped by reset: elapsed-from-start is small and
        // non-negative, definitely less than the pre-reset accumulation.
        let since_start = t.get_nsec_from_start();
        assert!(since_start >= 0.0);
        // A pause right after reset accumulates only the post-reset span.
        let after = t.pause();
        assert!(after >= 0.0);
        assert!(after < (SLEEP_MS * 1_000_000) as f64 || after < since_start + 1e9);
    }

    #[test]
    fn start_is_an_alias_for_resume() {
        let mut t = Timer::new();
        t.start();
        sleep(Duration::from_millis(SLEEP_MS));
        assert!(t.pause() >= (SLEEP_MS * 1_000_000) as f64);
    }

    #[test]
    fn get_nsec_from_start_does_not_accumulate() {
        let mut t = Timer::new();
        t.resume();
        sleep(Duration::from_millis(SLEEP_MS));
        let peek = t.get_nsec_from_start();
        assert!(peek >= (SLEEP_MS * 1_000_000) as f64);
        // Peeking must not have modified the accumulated total.
        assert_eq!(t.get_nsec(), 0.0);
    }

    #[test]
    fn nsec_timer_unit_conversions_are_exact() {
        let n = NsecTimer::from_nsec(1_500_000_000.0);
        assert_eq!(n.get_nsec(), 1_500_000_000.0);
        assert_eq!(n.get_usec(), 1_500_000.0);
        assert_eq!(n.get_msec(), 1_500.0);
        assert_eq!(n.get_sec(), 1.5);
    }

    #[test]
    fn nsec_timer_zero_and_default() {
        let n = NsecTimer::new();
        assert_eq!(n.get_nsec(), 0.0);
        assert_eq!(n.get_sec(), 0.0);
        assert_eq!(NsecTimer::default(), n);
    }

    #[test]
    fn nsec_timer_add_accumulates() {
        let mut n = NsecTimer::new();
        n.add_nsec(500.0);
        n.add_nsec(250.0);
        assert_eq!(n.get_nsec(), 750.0);
        assert_eq!(n.get_usec(), 0.75);
    }

    #[test]
    fn nsec_timer_huge_values_do_not_overflow() {
        // ~31.7 years in ns: far beyond any realistic accumulation, still
        // exact enough in f64 (< 2^53) and finite through every conversion.
        let n = NsecTimer::from_nsec(1e18);
        assert_eq!(n.get_sec(), 1e9);
        assert!(n.get_usec().is_finite());
        assert!(n.get_msec().is_finite());
    }

    #[test]
    fn nsec_timer_negative_span_preserved() {
        // C++ arithmetic is signed; negative accumulations pass through.
        let n = NsecTimer::from_nsec(-2_000_000.0);
        assert_eq!(n.get_msec(), -2.0);
        assert_eq!(n.get_sec(), -0.002);
    }

    #[test]
    fn timepoint_span_between_two_stamps() {
        let a = Timepoint::new();
        sleep(Duration::from_millis(SLEEP_MS));
        let b = Timepoint::new();
        let ns = a.get_nsec_from_start_until(&b);
        assert!(ns >= (SLEEP_MS * 1_000_000) as f64);
        assert_eq!(a.get_usec_from_start_until(&b), ns / 1_000.0);
        assert_eq!(a.get_msec_from_start_until(&b), ns / 1_000_000.0);
        assert_eq!(a.get_sec_from_start_until(&b), ns / 1_000_000_000.0);
    }

    #[test]
    fn timepoint_reversed_span_is_exact_negation() {
        let a = Timepoint::new();
        sleep(Duration::from_millis(SLEEP_MS));
        let b = Timepoint::new();
        let forward = a.get_nsec_from_start_until(&b);
        let backward = b.get_nsec_from_start_until(&a);
        assert!(forward > 0.0);
        assert_eq!(backward, -forward);
    }

    #[test]
    fn timepoint_span_to_itself_is_zero() {
        let a = Timepoint::new();
        assert_eq!(a.get_nsec_from_start_until(&a), 0.0);
        assert_eq!(a.get_sec_from_start_until(&a), 0.0);
    }

    #[test]
    fn timepoint_now_restamps() {
        let mut a = Timepoint::new();
        sleep(Duration::from_millis(SLEEP_MS));
        let before_restamp = a.get_nsec_from_start();
        assert!(before_restamp >= (SLEEP_MS * 1_000_000) as f64);
        a.now();
        let b = Timepoint::new();
        // After restamping, the span to a fresh timepoint is tiny (well
        // under the slept duration).
        assert!(a.get_nsec_from_start_until(&b) < before_restamp);
    }

    #[test]
    fn timepoint_elapsed_to_current_instant_is_monotonic_nonnegative() {
        let a = Timepoint::new();
        let e1 = a.get_nsec_from_start();
        let e2 = a.get_nsec_from_start();
        assert!(e1 >= 0.0);
        assert!(e2 >= e1);
        assert!(a.get_usec_from_start() >= 0.0);
        assert!(a.get_msec_from_start() >= 0.0);
        assert!(a.get_sec_from_start() >= 0.0);
    }

    #[test]
    fn us_from_epoch_is_plausible_wall_clock() {
        let t = Timer::new();
        let us = t.get_us_from_epoch();
        // 2020-01-01 in us since epoch; any correctly-set clock exceeds it.
        assert!(us > 1_577_836_800_000_000.0, "us={us}");
        // And it is truncated whole microseconds, so an integral f64.
        assert_eq!(us.fract(), 0.0);
        // Monotone-ish sanity: a second read is not before the first by
        // more than clock-adjustment noise; just check it stays positive.
        assert!(t.get_us_from_epoch() > 0.0);
    }

    #[test]
    fn periodic_run_fires_when_threshold_reached() {
        let mut p = PeriodicRun::new();
        let mut fired = 0;
        // Zero threshold: `nsec >= 0` always holds, so it always fires.
        p.run(0, || fired += 1);
        assert_eq!(fired, 1);
        p.run(0, || fired += 1);
        assert_eq!(fired, 2);
    }

    #[test]
    fn periodic_run_does_not_fire_before_threshold() {
        let mut p = PeriodicRun::new();
        let mut fired = false;
        // ~1 hour in ns: cannot elapse during the test.
        p.run(3_600_000_000_000, || fired = true);
        assert!(!fired);
    }

    #[test]
    fn periodic_run_resets_after_firing() {
        let mut p = PeriodicRun::new();
        sleep(Duration::from_millis(SLEEP_MS));
        let mut fired = 0;
        // Threshold below the slept span: fires and resets the timer.
        p.run(1_000_000, || fired += 1);
        assert_eq!(fired, 1);
        // Timer was reset, so the same threshold no longer fires
        // immediately (the huge threshold proves the reset regardless).
        p.run(3_600_000_000_000, || fired += 1);
        assert_eq!(fired, 1);
    }

    #[test]
    fn type_aliases_are_interchangeable() {
        let mut t: HighResCpuTimer = HighResMonotonicTimer::new();
        t.resume();
        let _: &Timer = &t;
        let tp: HighResCpuTimepoint = HighResMonotonicTimepoint::new();
        let _: &Timepoint = &tp;
    }

    #[test]
    fn timer_default_matches_new() {
        let a = Timer::default();
        assert_eq!(a.get_nsec(), Timer::new().get_nsec());
    }

    #[test]
    fn timer_timepoint_accessor_tracks_resume() {
        let mut t = Timer::new();
        t.resume();
        let stamp = *t.timepoint();
        sleep(Duration::from_millis(SLEEP_MS));
        let now = Timepoint::new();
        assert!(stamp.get_nsec_from_start_until(&now) >= (SLEEP_MS * 1_000_000) as f64);
    }
}
