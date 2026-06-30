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

Future<Task> IpcCpu2Self::SendIn(IpcManager *ipc,
                                      const clio::run::shared_ptr<Task> &task_ptr) {
  Worker *worker = CLIO_CUR_WORKER;

  // Create pointer future (no serialization): the Future owns a fresh FutureShm
  // via shared_ptr. Self-send origin is SHM and the task is a direct pointer
  // (vaddr 0, no FUTURE_WAS_COPIED) so SendOut resumes the parent coroutine
  // rather than routing a response back to a client.
  Future<Task> future(task_ptr->pool_id_, task_ptr->method_, task_ptr);
  {
    auto fs = future.GetFutureShm();
    fs->origin_ = ClientOrigin::kClientShm;
  }

  if (worker != nullptr) {
    // Worker thread path: set the parent task so EndTask can resume its
    // coroutine when this self-send completes.
    clio::run::shared_ptr<Task> &cur_task = worker->GetCurrentTask();
    if (!cur_task.IsNull()) {
      future.SetParentTask(cur_task);
    }

    // Allocate the task's RunContext (and resolve its container) before
    // enqueueing, so RouteTask / the worker have an active RunContext.
    future.GetTaskPtr()->BeginRunContext();

    // Use ClientMapTask to pick a lane and enqueue.
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
    // Non-worker thread path (e.g. ServerInit's synchronous admin pool
    // creation, where CLIO_CUR_WORKER is null): allocate the RunContext +
    // container, then full routing with force_enqueue. BeginRunContext does no
    // worker-specific setup, so it is safe off a worker thread.
    future.GetTaskPtr()->BeginRunContext();
    ipc->RouteTask(future, /*force_enqueue=*/true);
  }

  return future;
}

clio::run::shared_ptr<Task> IpcCpu2Self::RecvIn(Future<Task> &future) {
  // No deserialization needed — task is a direct pointer
  return future.GetTaskPtr();
}

void IpcCpu2Self::SendOut(const clio::run::shared_ptr<Task> &task_ptr,
                               ctp::lbm::Transport *send_transport) {
  auto future_shm = task_ptr->RunFuture().GetFutureShm();
  if (future_shm.IsNull()) {
    return;
  }
  ClientOrigin origin = future_shm->origin_;

  // Delegate to origin-based SendRuntime for non-self origins
  if (origin != ClientOrigin::kClientShm) {
    CLIO_IPC->SendRuntime(task_ptr, send_transport);
    return;
  }

  const clio::run::shared_ptr<Task> &parent_task = task_ptr->GetParentTask();
  if (!parent_task.IsNull() &&
      parent_task->EventQueue()) {
    // Runtime subtask with parent: enqueue Future to parent worker's event
    // queue. FUTURE_COMPLETE is NOT set here — it will be set by
    // ProcessEventQueue on the parent's worker thread.
    auto *parent_event_queue =
        reinterpret_cast<ctp::ipc::mpsc_ring_buffer<Future<Task, CLIO_QUEUE_ALLOC_T>,
                                                ctp::ipc::MallocAllocator> *>(
            parent_task->EventQueue());
    parent_event_queue->Emplace(task_ptr->RunFuture());
    if (parent_task->Lane()) {
      // Always signal — see ipc_cpu2cpu_impl.h for the race.
      CLIO_IPC->AwakenWorker(parent_task->Lane());
    }
  } else if (task_ptr->task_flags_.Any(TASK_EXTERNAL_CLIENT)) {
    // Top-level task from an EXTERNAL (cross-process) SHM client. This is a
    // daemon-local deserialized copy (IpcCpu2Cpu::RecvIn AllocLoadTask'd it and
    // stamped TASK_EXTERNAL_CLIENT + the waiter pid/tid); the client process is
    // blocked polling its own MPSC mailbox clio-<pid>-<tid>, not this copy's
    // is_complete_ flag. Just flipping the local flag (the in-process branch
    // below) would never reach it — the client hangs forever (the
    // cr_ipc_transport_shm / cr_client_retry_*_shm timeouts on the macOS + boost
    // configs). Serialize the result and send it to the client's mailbox.
    IpcCpu2Cpu::SendOut(CLIO_IPC, task_ptr, send_transport);
  } else {
    // Top-level in-process self-send: client and runtime share this task
    // object, so flipping the completion flag is observed directly.
    task_ptr->SetComplete();
  }
}

}  // namespace clio::run
