// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).
//
// Stackful fiber backend wrapping Boost.Context through
// shim/boost_fiber_shim.cc (feature `boost-fibers`). Mirrors the C++
// runtime's CLIO_CORE_ENABLE_BOOST_COROUTINES mode so C++ and Rust tasks
// can share one scheduler during migration.

use crate::{Coroutine, CoroutineState};
use std::ffi::c_void;

extern "C" {
    fn ctp_fiber_create(
        stack_size: usize,
        entry: unsafe extern "C" fn(*mut c_void, *mut c_void),
        arg: *mut c_void,
    ) -> *mut c_void;
    fn ctp_fiber_resume(fiber: *mut c_void) -> i32;
    fn ctp_fiber_yield(fiber: *mut c_void);
    fn ctp_fiber_destroy(fiber: *mut c_void);
}

/// Handed to the fiber body; the only sanctioned way to suspend.
///
/// Not `Send`/`Sync` (raw pointer field) and only usable from inside the
/// fiber it was created for — both enforced by construction: the yielder is
/// created by the trampoline on the fiber's own stack and never escapes the
/// body's lifetime.
pub struct Yielder {
    fiber: *mut c_void,
}

impl Yielder {
    /// Suspend the fiber; the current `resume()` call returns `Yielded`.
    pub fn yield_now(&self) {
        // SAFETY: `fiber` is the live handle of the fiber we are executing
        // on (constructed by the trampoline), and yield is only reachable
        // from inside the body per the type's non-escape construction.
        unsafe { ctp_fiber_yield(self.fiber) };
    }
}

type FiberBody = Box<dyn FnOnce(&Yielder)>;

/// Trampoline running on the fiber stack. Catches panics: unwinding across
/// the Boost frames is UB, so a panicking body aborts with a message
/// instead.
unsafe extern "C" fn fiber_trampoline(arg: *mut c_void, fiber: *mut c_void) {
    // SAFETY: `arg` is the Box::into_raw of the FiberBody created in
    // `BoostFiberCoroutine::new`, passed through create exactly once.
    let body = unsafe { Box::from_raw(arg as *mut FiberBody) };
    let yielder = Yielder { fiber };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        (*body)(&yielder);
    }));
    if result.is_err() {
        eprintln!("ctp-coroutine: panic inside boost fiber body; aborting");
        std::process::abort();
    }
}

/// A stackful coroutine on a Boost.Context fiber.
pub struct BoostFiberCoroutine {
    handle: *mut c_void,
    done: bool,
}

impl BoostFiberCoroutine {
    /// Create a suspended fiber; user code runs on the first `resume()`.
    /// `stack_size == 0` selects Boost's default stack size.
    pub fn new(stack_size: usize, body: impl FnOnce(&Yielder) + 'static) -> Self {
        let boxed: Box<FiberBody> = Box::new(Box::new(body));
        // SAFETY: entry/arg pair is the trampoline contract above.
        let handle = unsafe {
            ctp_fiber_create(
                stack_size,
                fiber_trampoline,
                Box::into_raw(boxed) as *mut c_void,
            )
        };
        assert!(!handle.is_null(), "ctp_fiber_create failed");
        Self {
            handle,
            done: false,
        }
    }
}

impl Coroutine for BoostFiberCoroutine {
    fn resume(&mut self) -> CoroutineState {
        if self.done {
            return CoroutineState::Complete;
        }
        // SAFETY: handle is live (not destroyed) and not currently running.
        let finished = unsafe { ctp_fiber_resume(self.handle) } != 0;
        if finished {
            self.done = true;
            CoroutineState::Complete
        } else {
            CoroutineState::Yielded
        }
    }
}

impl Drop for BoostFiberCoroutine {
    fn drop(&mut self) {
        // Run-to-completion before destroy: unwinding Rust frames parked on
        // a suspended fiber stack across the FFI is UB, so drive the fiber
        // to its natural end first (bodies are cooperative by contract).
        while !self.done {
            // SAFETY: as in resume().
            self.done = unsafe { ctp_fiber_resume(self.handle) } != 0;
        }
        // SAFETY: fiber completed; handle owned uniquely by self.
        unsafe { ctp_fiber_destroy(self.handle) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Worker;
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn fiber_yields_and_completes() {
        let log = Rc::new(RefCell::new(Vec::<u32>::new()));
        let log2 = Rc::clone(&log);
        let mut f = BoostFiberCoroutine::new(0, move |y| {
            log2.borrow_mut().push(1);
            y.yield_now();
            log2.borrow_mut().push(2);
        });
        assert_eq!(f.resume(), CoroutineState::Yielded);
        assert_eq!(*log.borrow(), vec![1]);
        assert_eq!(f.resume(), CoroutineState::Complete);
        assert_eq!(*log.borrow(), vec![1, 2]);
    }

    #[test]
    fn mixed_backends_share_one_worker() {
        // The migration-critical property: a stackful (boost) task and a
        // stackless (async) task interleave on the same round-robin worker.
        let log = Rc::new(RefCell::new(Vec::<String>::new()));
        let mut w = Worker::new();

        let l = Rc::clone(&log);
        w.spawn_coroutine(Box::new(BoostFiberCoroutine::new(0, move |y| {
            for i in 0..3 {
                l.borrow_mut().push(format!("fiber{i}"));
                y.yield_now();
            }
        })));

        let l = Rc::clone(&log);
        w.spawn(async move {
            for i in 0..3 {
                l.borrow_mut().push(format!("async{i}"));
                crate::yield_now().await;
            }
        });

        w.run_until_idle();
        assert_eq!(
            *log.borrow(),
            vec!["fiber0", "async0", "fiber1", "async1", "fiber2", "async2"]
        );
    }
}
