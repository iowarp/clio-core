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
 *   ClientSend  -> enqueue Future<Task> to worker lane
 *   RuntimeRecv -> no-op (task is already a pointer)
 *   RuntimeSend -> set FUTURE_COMPLETE or enqueue to parent event queue
 *   ClientRecv  -> poll FUTURE_COMPLETE in shared memory
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
  static Future<Task> ClientSend(IpcManager *ipc,
                                 const ctp::ipc::FullPtr<Task> &task_ptr);

  /**
   * Runtime receives a self-sent task.
   *
   * No deserialization needed — the task is already a valid pointer.
   * Returns the task pointer directly from the future.
   *
   * @param future Future containing the task pointer
   * @return FullPtr to the task (same pointer from ClientSend)
   */
  static ctp::ipc::FullPtr<Task> RuntimeRecv(Future<Task> &future);

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
   * @param run_ctx RunContext with future and execution state
   * @param container Container for serialization (only if was_copied)
   * @param send_transport SHM transport (only if was_copied)
   */
  static void RuntimeSend(const FullPtr<Task> &task_ptr,
                           RunContext *run_ctx,
                           Container *container,
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
  static bool ClientRecv(Future<TaskT, AllocT> &future, float max_sec,
                         ctp::ipc::FullPtr<FutureShm> future_full) {
    // Poll FUTURE_COMPLETE in shared memory
    ctp::abitfield32_t &flags = future_full->flags_;
    auto start = std::chrono::steady_clock::now();
    while (!flags.Any(FutureShm::FUTURE_COMPLETE)) {
      CTP_THREAD_MODEL->Yield();
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
