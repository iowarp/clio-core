/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * Task implementation
 *
 * The task's coroutine/fiber driving mechanics live here (not in the worker):
 * a Task owns its RunContext and therefore its coroutine frame, so starting and
 * resuming the coroutine is the Task's own responsibility. The coroutine handle
 * (run_ctx_->coro_handle_) is touched ONLY from these Task members — it is not
 * an externally-accessible thing.
 */

#include "clio_runtime/task.h"

#if !defined(CLIO_ENABLE_BOOST_COROUTINES)
#include <coroutine>
#endif

#include "clio_runtime/container.h"
#include "clio_runtime/ipc_manager.h"
#include "clio_runtime/singletons.h"
#include "clio_runtime/boost_stack_allocator.h"

namespace clio::run {

// Task::~Task() is defined inline in task.h (CTP_CROSS_FUN CLIO_VIRTUAL); the
// owned RunContext is a ctp::unique_ptr member (run_ctx_) freed automatically,
// so there is no longer a custom DestroyRunCtx().

#if defined(CLIO_ENABLE_BOOST_COROUTINES)
// Set up this task's RunContext to run on a fresh Boost.Context fiber, recording
// the current worker as the one responsible for the fiber state. The fiber state
// lives in run_ctx_->fiber_state_, and the fiber stack comes from the process-
// wide, per-thread-cached BoostStackPool() via BoostStackPoolAllocator (the
// Boost StackAllocator interface) — so a reused stack costs no malloc. The entry
// runs the container directly off the RunContext. Lazy: the body runs on first
// resume().
clio::run::detail::FiberHandle Task::MakeTaskFiber(
    clio::run::shared_ptr<Task> &self) {
  // Capture the owning task handle by value in the fiber entry so the task
  // stays alive for the whole (possibly suspended) fiber execution — the fiber
  // outlives this call. `mutable` so the captured handle can be passed to Run,
  // whose signature takes a non-const shared_ptr<Task>&.
  self->FiberStateRef().done = false;
  self->FiberStateRef().worker_ = CLIO_CUR_WORKER;
  self->FiberStateRef().task_ = boost::context::fiber{
      std::allocator_arg, BoostStackPoolAllocator{},
      [self](boost::context::fiber &&caller) mutable -> boost::context::fiber {
        self->FiberStateRef().caller_ = std::move(caller);
        // Must not let an exception escape the fiber entry (Boost.Context calls
        // std::terminate if one does).
        try {
          ContainerHold c = self->ExecContainer().get();
          if (c) {
            c->Run(self->method_, self);
          }
        } catch (...) {
        }
        self->FiberStateRef().done = true;
        return std::move(self->FiberStateRef().caller_);
      }};
  return clio::run::detail::FiberHandle(&self->FiberStateRef());
}
#endif  // CLIO_ENABLE_BOOST_COROUTINES

void Task::StartCoroutine(clio::run::shared_ptr<Task> &self) {
  // Set the current task for this worker thread
  SetCurrentTask(self);

  // Per-execution initialization (merged from the former BeginOnRuntime). This
  // runs on the worker that first executes the task, so CLIO_CUR_WORKER is
  // valid. Worker id / lane / event queue / future are already bound by
  // Worker::ProcessNewTask; here we set the remaining execution state.
  SetYielded(false);  // Initially not blocked
  // Adaptive polling fields for periodic tasks. TruePeriodNs() now reads
  // period_ns_ directly, so there is nothing to copy here.
  if (IsPeriodic()) {
    SetYieldTimeUs(period_ns_ / 1000.0);  // Initialize with true period
    SetDidWork(false);                    // Initially no work done
  } else {
    SetYieldTimeUs(0.0);
    SetDidWork(false);
  }

  // Read the current (most-recently-upgraded) container version for this
  // dispatch (the resume that runs the coroutine/fiber to its first suspension).
  ContainerHold container = ExecContainer().get();

  // Populate predicted_stat_ from the container so downstream routing / load
  // tracking can read the task's payload size without re-doing GetTaskStats.
  if (container) {
    PredictedStat() = container->GetTaskStats(self.get());
  }

  // New task execution - increment work count for non-periodic tasks
  if (container && !IsPeriodic()) {
    // Increment work remaining in the container for non-periodic tasks
    container->UpdateWork(self, 1);
  }

  if (!container) {
    HLOG(kWarning, "Container not found in RunContext for pool_id: {}",
         pool_id_);
    return;
  }

  // Call the container's Run function which returns a TaskResume coroutine/fiber.
  try {
#if defined(CLIO_ENABLE_BOOST_COROUTINES)
    // Boost.Context path (issue #620): one fiber per task whose entry runs
    // Container::Run natively on the fiber stack. The whole task — including
    // nested helper coroutines, which run inline — executes as ordinary C++, so
    // reference parameters and locals behave exactly as in the C++20 stackless
    // backend. The fiber state lives in run_ctx_->fiber_state_ (only the stack
    // is heap-allocated); MakeTaskFiber's entry dispatches container->Run().
    // The fiber frame is owned by this RunContext and freed when the RunContext
    // is destroyed — we never destroy it here.
    {
      SetCoroCompleted(false);
      run_ctx_->coro_handle_ = MakeTaskFiber(self);
      if (run_ctx_->coro_handle_) {
        run_ctx_->coro_handle_.resume();
      }
    }
#else
    TaskResume task_resume = container->Run(method_, self);

    // Standard C++20 coroutine path
    auto handle = task_resume.release();
    run_ctx_->coro_handle_ = handle;

    // Set the executing task in the coroutine's promise so it can access it
    if (handle) {
      auto typed_handle =
          TaskResume::handle_type::from_address(handle.address());
      typed_handle.promise().set_task(self.get());
      // Mark this as the top-level task coroutine so its (and only its)
      // final_suspend raises the task's coro_completed_ flag on real task
      // completion (issue #485). Nested co_await'd coroutines are never marked.
      typed_handle.promise().is_top_level_ = true;

      // Fresh coroutine frame: clear the completion flag the promise's
      // final_suspend will raise when this task finishes (issue #485).
      SetCoroCompleted(false);

      // Resume the coroutine to run until first suspension point or completion.
      // initial_suspend returns suspend_always, so we resume to start execution.
      handle.resume();

      // Stackless: if the task completed in this slice (no suspension points),
      // destroy the top-level frame now. The RunContext destructor does NOT own
      // the stackless frame (coro_handle_ is a shared slot reused by nested
      // coroutines), so completion is the single owning teardown point. Read the
      // completion flag, never handle.done() on a possibly cross-thread-freed
      // frame (#485); when set, await_resume has repointed coro_handle_ back to
      // this (valid) top-level handle.
      if (IsCoroCompleted()) {
        handle.destroy();
        run_ctx_->coro_handle_ = nullptr;
      }
    }
#endif // CLIO_ENABLE_BOOST_COROUTINES vs C++20 stackless
  } catch (const std::exception &e) {
    HLOG(kError, "Task execution failed: {}", e.what());
  } catch (...) {
    HLOG(kError, "Task execution failed with unknown exception");
  }
}

void Task::ResumeCoroutine(clio::run::shared_ptr<Task> &self) {
  // Set the current task for this worker thread
  SetCurrentTask(self);

  // Clear yielded flag before resumption
  SetYielded(false);

  // Check if we have a valid coroutine handle
  if (!run_ctx_ || !run_ctx_->coro_handle_) {
    HLOG(kWarning,
         "Attempted to resume task without coroutine handle. "
         "Task method: {} Pool: {}",
         method_, pool_id_);
    return;
  }

  // Resume the coroutine/fiber - it will run until next suspension or
  // completion. Completion is detected via the RunContext completion flag (set
  // by the top-level coroutine's final_suspend), never coro_handle_.done() on a
  // possibly cross-thread-freed frame (#485).
  try {
    run_ctx_->coro_handle_.resume();

#if !defined(CLIO_ENABLE_BOOST_COROUTINES)
    // Stackless: destroy the top-level frame on completion (the RunContext
    // destructor does not own it — coro_handle_ is a shared slot). When the
    // flag is set, await_resume has repointed coro_handle_ back to the (valid)
    // top-level handle, so destroy() targets a live frame.
    if (IsCoroCompleted()) {
      run_ctx_->coro_handle_.destroy();
      run_ctx_->coro_handle_ = nullptr;
    }
#endif
    // Boost: the fiber frame is owned by this RunContext (inline fiber_state_)
    // and freed when the RunContext is destroyed — no destroy() here.
  } catch (const std::exception &e) {
    HLOG(kError, "Task resume failed: {}", e.what());
  } catch (...) {
    HLOG(kError, "Task resume failed with unknown exception");
  }
}

}  // namespace clio::run
