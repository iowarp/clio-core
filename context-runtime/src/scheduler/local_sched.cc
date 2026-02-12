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
#include "chimaera/scheduler/local_sched.h"

#include <functional>

#include "chimaera/config_manager.h"
#include "chimaera/ipc_manager.h"
#include "chimaera/work_orchestrator.h"
#include "chimaera/worker.h"

namespace chi {

void LocalScheduler::DivideWorkers(WorkOrchestrator *work_orch) {
  if (!work_orch) {
    return;
  }

  ConfigManager *config = CHI_CONFIG_MANAGER;
  if (!config) {
    HLOG(kError,
         "LocalScheduler::DivideWorkers: ConfigManager not available");
    return;
  }

  u32 thread_count = config->GetNumThreads();
  u32 total_workers = work_orch->GetTotalWorkerCount();

  scheduler_workers_.clear();
  net_worker_ = nullptr;
  gpu_worker_ = nullptr;

  net_worker_ = work_orch->GetWorker(total_workers - 1);

  if (total_workers > 2) {
    gpu_worker_ = work_orch->GetWorker(total_workers - 2);
  }

  u32 num_sched_workers = (total_workers == 1) ? 1 : (total_workers - 1);
  for (u32 i = 0; i < num_sched_workers; ++i) {
    Worker *worker = work_orch->GetWorker(i);
    if (worker) {
      scheduler_workers_.push_back(worker);
    }
  }

  IpcManager *ipc = CHI_IPC;
  if (ipc) {
    ipc->SetNumSchedQueues(num_sched_workers);
  }

  HLOG(kInfo,
       "LocalScheduler: {} scheduler workers, 1 network worker (worker {})"
       ", gpu_worker={}",
       scheduler_workers_.size(), total_workers - 1,
       gpu_worker_ ? (int)gpu_worker_->GetId() : -1);
}

u32 LocalScheduler::ClientMapTask(IpcManager *ipc_manager,
                                   const Future<Task> &task) {
  u32 num_lanes = ipc_manager->GetNumSchedQueues();
  if (num_lanes == 0) {
    return 0;
  }

  Task *task_ptr = task.get();
  if (task_ptr != nullptr && task_ptr->pool_id_ == chi::kAdminPoolId) {
    u32 method_id = task_ptr->method_;
    if (method_id == 14 || method_id == 15) {
      return num_lanes - 1;
    }
  }

  u32 lane = MapByPidTid(num_lanes);
  return lane;
}

u32 LocalScheduler::RuntimeMapTask(Worker *worker, const Future<Task> &task) {
  Task *task_ptr = task.get();
  if (task_ptr != nullptr && task_ptr->IsPeriodic()) {
    if (task_ptr->pool_id_ == chi::kAdminPoolId) {
      u32 method_id = task_ptr->method_;
      if (method_id == 14 || method_id == 15) {
        if (net_worker_ != nullptr) {
          return net_worker_->GetId();
        }
      }
    }
  }

  if (gpu_worker_ != nullptr && worker == gpu_worker_ &&
      !scheduler_workers_.empty()) {
    u32 idx = next_sched_idx_.fetch_add(1, std::memory_order_relaxed)
              % scheduler_workers_.size();
    return scheduler_workers_[idx]->GetId();
  }

  if (worker != nullptr) {
    return worker->GetId();
  }
  return 0;
}

void LocalScheduler::RebalanceWorker(Worker *worker) {
  (void)worker;
}

void LocalScheduler::AdjustPolling(RunContext *run_ctx) {
  if (!run_ctx) {
    return;
  }
  // Adaptive polling disabled for now - restore the true period
  // This is critical because co_await on Futures sets yield_time_us_ = 0,
  // so we must restore it here to prevent periodic tasks from busy-looping
  run_ctx->yield_time_us_ = run_ctx->true_period_ns_ / 1000.0;
}

u32 LocalScheduler::MapByPidTid(u32 num_lanes) {
  auto *sys_info = HSHM_SYSTEM_INFO;
  pid_t pid = sys_info->pid_;
  auto tid = HSHM_THREAD_MODEL->GetTid();

  size_t combined_hash =
      std::hash<pid_t>{}(pid) ^ (std::hash<hshm::u64>{}(tid.tid_) << 1);
  return static_cast<u32>(combined_hash % num_lanes);
}

}  // namespace chi
