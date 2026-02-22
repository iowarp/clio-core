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

// Copyright 2024 IOWarp contributors
#include "chimaera/scheduler/default_sched.h"

#include "chimaera/config_manager.h"
#include "chimaera/ipc_manager.h"
#include "chimaera/work_orchestrator.h"
#include "chimaera/worker.h"

namespace chi {

void DefaultScheduler::DivideWorkers(WorkOrchestrator *work_orch) {
  if (!work_orch) {
    return;
  }

  u32 total_workers = work_orch->GetTotalWorkerCount();

  scheduler_worker_ = nullptr;
  io_workers_.clear();
  net_worker_ = nullptr;
  gpu_worker_ = nullptr;

  // Worker 0 is always the scheduler worker
  scheduler_worker_ = work_orch->GetWorker(0);

  // Network worker is always the last worker
  net_worker_ = work_orch->GetWorker(total_workers - 1);

  // GPU worker is worker N-2 if we have more than 2 workers
  if (total_workers > 2) {
    gpu_worker_ = work_orch->GetWorker(total_workers - 2);
  }

  // I/O workers are workers 1..N-2 (empty if N <= 2)
  if (total_workers > 2) {
    for (u32 i = 1; i < total_workers - 1; ++i) {
      Worker *worker = work_orch->GetWorker(i);
      if (worker) {
        io_workers_.push_back(worker);
      }
    }
  }

  // Number of scheduling queues excludes the network worker
  IpcManager *ipc = CHI_IPC;
  if (ipc) {
    ipc->SetNumSchedQueues(1);
  }

  HLOG(kInfo,
       "DefaultScheduler: 1 scheduler worker (0), {} I/O workers, "
       "1 network worker ({}), gpu_worker={}",
       io_workers_.size(), total_workers - 1,
       gpu_worker_ ? (int)gpu_worker_->GetId() : -1);
}

u32 DefaultScheduler::ClientMapTask(IpcManager *ipc_manager,
                                    const Future<Task> &task) {
  u32 num_lanes = ipc_manager->GetNumSchedQueues();
  if (num_lanes == 0) {
    return 0;
  }

  Task *task_ptr = task.get();

  // Network tasks (Send/Recv from admin pool) → last lane
  if (task_ptr != nullptr && task_ptr->pool_id_ == chi::kAdminPoolId) {
    u32 method_id = task_ptr->method_;
    if (method_id == 14 || method_id == 15) {
      return num_lanes - 1;
    }
  }

  // Default: scheduler worker (lane 0)
  return 0;
}

u32 DefaultScheduler::RuntimeMapTask(Worker *worker, const Future<Task> &task,
                                      Container *container) {
  Task *task_ptr = task.get();

  // ---- Task group affinity: return early if group already pinned ----
  // Use the caller-supplied container directly (no static container lookup).
  Container *grp_container =
      (container != nullptr && task_ptr != nullptr &&
       !task_ptr->task_group_.IsNull())
          ? container
          : nullptr;
  if (grp_container != nullptr) {
    int64_t group_id = task_ptr->task_group_.id_;
    ScopedCoRwReadLock read_lock(grp_container->task_group_lock_);
    auto it = grp_container->task_group_map_.find(group_id);
    if (it != grp_container->task_group_map_.end() && it->second != nullptr) {
      return it->second->GetId();
    }
  }

  // ---- Normal routing: determine selected worker ----
  Worker *selected = nullptr;

  // Periodic Send/Recv → network worker
  if (task_ptr != nullptr && task_ptr->IsPeriodic()) {
    if (task_ptr->pool_id_ == chi::kAdminPoolId) {
      u32 method_id = task_ptr->method_;
      if ((method_id == 14 || method_id == 15) && net_worker_ != nullptr) {
        selected = net_worker_;
      }
    }
  }

  // Route large I/O to dedicated I/O workers (round-robin)
  if (selected == nullptr && task_ptr != nullptr && !io_workers_.empty()) {
    size_t io_size = task_ptr->stat_.io_size_;
    if (io_size >= kLargeIOThreshold) {
      u32 idx = next_io_idx_.fetch_add(1, std::memory_order_relaxed) %
                static_cast<u32>(io_workers_.size());
      selected = io_workers_[idx];
    }
  }

  // Small I/O / metadata → scheduler worker
  if (selected == nullptr && scheduler_worker_ != nullptr) {
    selected = scheduler_worker_;
  }

  // Fallback to current worker
  if (selected == nullptr) {
    selected = worker;
  }

  // ---- Update group map after routing decision ----
  if (grp_container != nullptr && task_ptr != nullptr && selected != nullptr) {
    int64_t group_id = task_ptr->task_group_.id_;
    ScopedCoRwWriteLock write_lock(grp_container->task_group_lock_);
    auto it = grp_container->task_group_map_.find(group_id);
    if (it == grp_container->task_group_map_.end() || it->second == nullptr) {
      grp_container->task_group_map_[group_id] = selected;
    }
  }

  if (selected != nullptr) {
    return selected->GetId();
  }
  return 0;
}

void DefaultScheduler::RebalanceWorker(Worker *worker) { (void)worker; }

void DefaultScheduler::AdjustPolling(RunContext *run_ctx) {
  if (!run_ctx) {
    return;
  }
  // Adaptive polling disabled for now - restore the true period
  // This is critical because co_await on Futures sets yield_time_us_ = 0,
  // so we must restore it here to prevent periodic tasks from busy-looping
  run_ctx->yield_time_us_ = run_ctx->true_period_ns_ / 1000.0;
}

}  // namespace chi
