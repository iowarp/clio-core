// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).
//
// One coroutine abstraction, two backends — mirroring the C++ runtime's
// stackless (C++20) / stackful (CLIO_CORE_ENABLE_BOOST_COROUTINES) duality:
//
//  * `StacklessCoroutine` — native Rust async/await. A task is a
//    `Future<Output = ()>`; `resume()` polls it once. Cooperative
//    suspension uses [`yield_now`], exactly like `co_await yield()` on the
//    C++ worker. This is the default and the target for new pure-Rust code.
//
//  * `BoostFiberCoroutine` (feature `boost-fibers`) — stackful fibers
//    wrapping Boost.Context through a small C++ shim. Exists for interop
//    with the C++ worker during migration: C++ boost-fiber tasks and Rust
//    tasks can share one scheduler.
//
// Both implement [`Coroutine`], and [`Worker`] runs any mix of them with a
// clio-style round-robin loop.

#![deny(unsafe_op_in_unsafe_fn)]

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, Waker};

#[cfg(feature = "boost-fibers")]
pub mod boost;

/// Result of driving a coroutine one step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoroutineState {
    /// The coroutine suspended cooperatively; resume it again later.
    Yielded,
    /// The coroutine ran to completion; do not resume it again.
    Complete,
}

/// A resumable unit of work — the common face of both backends.
///
/// Contract: after `Complete` is returned once, `resume` must not be called
/// again (the [`Worker`] upholds this by dropping finished coroutines).
pub trait Coroutine {
    fn resume(&mut self) -> CoroutineState;
}

// ---------------------------------------------------------------------------
// Stackless backend: native Rust async
// ---------------------------------------------------------------------------

/// Cooperative yield — the async equivalent of the C++ worker's
/// `co_await yield()`. The first poll suspends; the next resume completes.
pub fn yield_now() -> YieldNow {
    YieldNow { yielded: false }
}

pub struct YieldNow {
    yielded: bool,
}

impl Future for YieldNow {
    type Output = ();
    fn poll(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<()> {
        if self.yielded {
            Poll::Ready(())
        } else {
            self.yielded = true;
            Poll::Pending
        }
    }
}

/// Stackless coroutine: a pinned future polled once per `resume`.
///
/// The clio worker model is busy-poll cooperative scheduling — a suspended
/// task is simply re-polled on the next worker iteration — so `Pending`
/// maps to `Yielded` and no waker plumbing is required. (Futures that wait
/// on external events still work: they are re-polled every round, which is
/// exactly the lost-wakeup-proof busy-poll the C++ side uses for the
/// per-blob write token.)
pub struct StacklessCoroutine {
    future: Pin<Box<dyn Future<Output = ()>>>,
}

impl StacklessCoroutine {
    pub fn new(future: impl Future<Output = ()> + 'static) -> Self {
        Self {
            future: Box::pin(future),
        }
    }
}

impl Coroutine for StacklessCoroutine {
    fn resume(&mut self) -> CoroutineState {
        let waker = Waker::noop();
        let mut cx = Context::from_waker(waker);
        match self.future.as_mut().poll(&mut cx) {
            Poll::Pending => CoroutineState::Yielded,
            Poll::Ready(()) => CoroutineState::Complete,
        }
    }
}

// ---------------------------------------------------------------------------
// Worker: clio-style round-robin cooperative scheduler
// ---------------------------------------------------------------------------

/// Minimal cooperative executor mirroring the clio worker loop: coroutines
/// are resumed round-robin; a `Yielded` coroutine goes to the back of the
/// queue, a `Complete` one is dropped.
#[derive(Default)]
pub struct Worker {
    run_queue: VecDeque<Box<dyn Coroutine>>,
}

impl Worker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Enqueue any coroutine (stackless or stackful).
    pub fn spawn_coroutine(&mut self, coro: Box<dyn Coroutine>) {
        self.run_queue.push_back(coro);
    }

    /// Convenience: enqueue an async task on the stackless backend.
    pub fn spawn(&mut self, future: impl Future<Output = ()> + 'static) {
        self.spawn_coroutine(Box::new(StacklessCoroutine::new(future)));
    }

    /// Number of live (unfinished) coroutines.
    pub fn len(&self) -> usize {
        self.run_queue.len()
    }

    pub fn is_empty(&self) -> bool {
        self.run_queue.is_empty()
    }

    /// One scheduler round: resume every queued coroutine once.
    /// Returns the number still alive.
    pub fn run_round(&mut self) -> usize {
        for _ in 0..self.run_queue.len() {
            let mut coro = self.run_queue.pop_front().expect("len checked");
            if coro.resume() == CoroutineState::Yielded {
                self.run_queue.push_back(coro);
            }
        }
        self.run_queue.len()
    }

    /// Drive everything to completion.
    pub fn run_until_idle(&mut self) {
        while self.run_round() > 0 {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn yield_now_suspends_once() {
        let mut c = StacklessCoroutine::new(async {
            yield_now().await;
        });
        assert_eq!(c.resume(), CoroutineState::Yielded);
        assert_eq!(c.resume(), CoroutineState::Complete);
    }

    #[test]
    fn worker_interleaves_cooperatively() {
        // Two tasks yielding between steps must interleave a,b,a,b — the
        // defining property of the cooperative round-robin worker.
        let log = Rc::new(RefCell::new(Vec::<String>::new()));
        let mut w = Worker::new();
        for name in ["a", "b"] {
            let log = Rc::clone(&log);
            w.spawn(async move {
                for i in 0..3 {
                    log.borrow_mut().push(format!("{name}{i}"));
                    yield_now().await;
                }
            });
        }
        w.run_until_idle();
        assert_eq!(*log.borrow(), vec!["a0", "b0", "a1", "b1", "a2", "b2"]);
        assert!(w.is_empty());
    }

    #[test]
    fn nested_awaits_run_to_completion() {
        async fn inner(n: u64) -> u64 {
            yield_now().await;
            n * 2
        }
        let out = Rc::new(RefCell::new(0u64));
        let out2 = Rc::clone(&out);
        let mut w = Worker::new();
        w.spawn(async move {
            let mut acc = 0;
            for i in 1..=4 {
                acc += inner(i).await;
            }
            *out2.borrow_mut() = acc;
        });
        w.run_until_idle();
        assert_eq!(*out.borrow(), 20);
    }
}
