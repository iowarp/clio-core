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

#ifndef CLIO_RUNTIME_INCLUDE_IPC_CPU2SELF_H_
#define CLIO_RUNTIME_INCLUDE_IPC_CPU2SELF_H_

#include "clio_runtime/types.h"
#include "clio_runtime/task.h"

namespace clio::run {

class IpcManager;
class Worker;

/**
 * IPC transport for runtime-internal tasks (CPU runtime sending to itself).
 *
 * No serialization/deserialization. Tasks are passed by pointer via
 * MakePointerFuture. Worker threads use ClientMapTask to pick a lane;
 * non-worker threads use full RouteTask with force_enqueue.
 *
 * Lifecycle:
 *   SendIn  -> enqueue Future<Task> to worker lane
 *   RecvIn  -> no-op (task is already a pointer)
 *   SendOut -> set FUTURE_COMPLETE or enqueue to parent event queue
 *   RecvOut -> poll FUTURE_COMPLETE in shared memory
 */
struct IpcCpu2Self {
  /**
   * Client sends a task to the runtime (from within the runtime process).
   *
   * Worker threads: creates pointer future, sets parent RunContext,
   *   allocates RunContext via BeginTask, enqueues via ClientMapTask.
   * Non-worker threads: creates pointer future, allocates RunContext,
   *   routes via RouteTask with force_enqueue=true.
   *
   * @param ipc IpcManager instance
   * @param task_ptr Task to enqueue
   * @return Future wrapping the task pointer (no serialization)
   */
  static Future<Task> SendIn(IpcManager *ipc,
                                 const clio::run::shared_ptr<Task> &task_ptr);

  /**
   * Runtime receives a self-sent task.
   *
   * No deserialization needed — the task is already a valid pointer.
   * Returns the task pointer directly from the future.
   *
   * @param future Future containing the task pointer
   * @return shared_ptr to the task (same handle from SendIn)
   */
  static clio::run::shared_ptr<Task> RecvIn(Future<Task> &future);

  /**
   * Runtime sends the response after task execution.
   *
   * Three paths:
   *   1. Parent has event queue: enqueue Future to parent's event queue
   *      (completion signaled later by ProcessEventQueue).
   *   2. No parent: set FUTURE_COMPLETE directly via system-scope atomic.
   *   3. Task was copied (FUTURE_WAS_COPIED): delegate to IpcManager::SendRuntime.
   *
   * @param task_ptr Executed task
   * @param send_transport SHM transport (only if was_copied)
   */
  static void SendOut(const clio::run::shared_ptr<Task> &task_ptr,
                           ctp::lbm::Transport *send_transport);

  /**
   * Client waits for task completion (runtime-internal poll).
   *
   * Polls FUTURE_COMPLETE flag in shared memory with optional timeout.
   *
   * @param future Future to wait on
   * @param max_sec Maximum seconds to wait (0 = forever)
   * @return true if completed, false if timed out
   */
  template <typename TaskT, typename AllocT>
  static bool RecvOut(Future<TaskT, AllocT> &future, float max_sec,
                         ctp::ipc::FullPtr<RunContext> future_full) {
    // Poll per-process completion on the task (set by SendOut on this runtime).
    (void)future_full;
    TaskT *task_ptr = future.get();
    auto start = std::chrono::steady_clock::now();
    size_t spins = 0;
    while (!task_ptr->IsComplete()) {
      // Adaptive backoff. Busy-yield for the first burst so the common case (the
      // worker completes our task in microseconds) stays low-latency, then sleep
      // so we CEDE the CPU to the runtime workers. Without the sleep, many
      // in-process waiter threads (same-blob write stress, a FUSE thread pool,
      // etc.) all spin in Yield() and starve the very workers that must run and
      // complete their tasks -- a livelock. std::this_thread::yield() does not
      // cede to the workers on Windows, so this shows up as a deterministic hang
      // there (cte_concurrent_same_blob_all) while Linux only slows down;
      // reproduced on Linux with 64 waiter threads pinned to 1 CPU. Same
      // oversubscription class as the FUSE xfstests hangs.
      if (++spins < 256) {
        CTP_THREAD_MODEL->Yield();
      } else {
        CTP_THREAD_MODEL->SleepForUs(50);
      }
      if (max_sec > 0) {
        float elapsed = std::chrono::duration<float>(
                            std::chrono::steady_clock::now() - start)
                            .count();
        if (elapsed >= max_sec) return false;
      }
    }
    return true;
  }
};

}  // namespace clio::run

#endif  // CLIO_RUNTIME_INCLUDE_IPC_CPU2SELF_H_
