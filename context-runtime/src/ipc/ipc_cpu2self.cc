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

#include "clio_runtime/ipc/ipc_cpu2self.h"
#include "clio_runtime/ipc_manager.h"
#include "clio_runtime/worker.h"
#include "clio_runtime/singletons.h"

namespace clio::run {

Future<Task> IpcCpu2Self::ClientSend(IpcManager *ipc,
                                      const ctp::ipc::FullPtr<Task> &task_ptr) {
  Worker *worker = CLIO_CUR_WORKER;

  // Create pointer future (no serialization)
  Future<Task> future = ipc->MakePointerFuture(task_ptr);

  if (worker != nullptr) {
    // Worker thread path: set parent RunContext so EndTask can resume parent
    RunContext *run_ctx = worker->GetCurrentRunContext();
    if (run_ctx != nullptr) {
      future.SetParentTask(run_ctx);
    }

    // Allocate RunContext before enqueueing
    if (!task_ptr->task_flags_.Any(TASK_RUN_CTX_EXISTS)) {
      ipc->BeginTask(future, nullptr, nullptr);
    }

    // Use ClientMapTask to pick a lane and enqueue
    if (ipc->scheduler_ != nullptr) {
      u32 lane_id = ipc->scheduler_->ClientMapTask(ipc, future);
      if (!ipc->worker_queues_.IsNull()) {
        auto &dest_lane = ipc->worker_queues_->GetLane(lane_id, 0);
        dest_lane.Push(future);
        // Always signal — see ipc_cpu2cpu_impl.h for the race.
        ipc->AwakenWorker(&dest_lane);
      }
    }
  } else {
    // Non-worker thread path: full routing with force_enqueue
    if (!task_ptr->task_flags_.Any(TASK_RUN_CTX_EXISTS)) {
      ipc->BeginTask(future, nullptr, nullptr);
    }
    ipc->RouteTask(future, /*force_enqueue=*/true);
  }

  return future;
}

ctp::ipc::FullPtr<Task> IpcCpu2Self::RuntimeRecv(Future<Task> &future) {
  // No deserialization needed — task is a direct pointer
  return future.GetTaskPtr();
}

void IpcCpu2Self::RuntimeSend(const FullPtr<Task> &task_ptr,
                               RunContext *run_ctx,
                               Container *container,
                               ctp::lbm::Transport *send_transport) {
  auto future_shm = run_ctx->future_.GetFutureShm();
  if (future_shm.IsNull()) {
    return;
  }
  bool was_copied = future_shm->flags_.Any(FutureShm::FUTURE_WAS_COPIED);
  u32 origin = future_shm->origin_;

  // Delegate to origin-based SendRuntime for non-self origins
  if (was_copied || origin != FutureShm::FUTURE_CLIENT_SHM) {
    CLIO_IPC->SendRuntime(task_ptr, run_ctx, container, send_transport);
    return;
  }

  RunContext *parent_task = run_ctx->future_.GetParentTask();
  if (parent_task && parent_task->event_queue_) {
    // Runtime subtask with parent: enqueue Future to parent worker's event
    // queue. FUTURE_COMPLETE is NOT set here — it will be set by
    // ProcessEventQueue on the parent's worker thread.
    auto *parent_event_queue =
        reinterpret_cast<ctp::ipc::mpsc_ring_buffer<Future<Task, CLIO_QUEUE_ALLOC_T>,
                                                ctp::ipc::MallocAllocator> *>(
            parent_task->event_queue_);
    parent_event_queue->Emplace(run_ctx->future_);
    if (parent_task->lane_) {
      // Always signal — see ipc_cpu2cpu_impl.h for the race.
      CLIO_IPC->AwakenWorker(parent_task->lane_);
    }
  } else {
    // Top-level client task: set FUTURE_COMPLETE directly.
    // Use SetBitsSystem for CPU→GPU visibility in UVM.
    future_shm->flags_.SetBitsSystem(FutureShm::FUTURE_COMPLETE);
  }
}

}  // namespace clio::run
