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

#include <functional>

#include "chimaera/config_manager.h"
#include "chimaera/ipc_manager.h"
#include "chimaera/work_orchestrator.h"
#include "chimaera/worker.h"

namespace chi {

void DefaultScheduler::DivideWorkers(WorkOrchestrator *work_orch) {
  if (!work_orch) {
    return;
  }

  // Get worker counts from configuration
  ConfigManager *config = CHI_CONFIG_MANAGER;
  if (!config) {
    HLOG(kError,
         "DefaultScheduler::DivideWorkers: ConfigManager not available");
    return;
  }

  u32 thread_count = config->GetNumThreads();
  u32 total_workers = work_orch->GetTotalWorkerCount();

  // Clear any existing worker assignments
  scheduler_workers_.clear();
  net_worker_ = nullptr;

  // Network worker is always the last worker
  net_worker_ = work_orch->GetWorker(total_workers - 1);

  // Scheduler workers are all workers except the last one (unless only 1
  // worker)
  u32 num_sched_workers = (total_workers == 1) ? 1 : (total_workers - 1);
  for (u32 i = 0; i < num_sched_workers; ++i) {
    Worker *worker = work_orch->GetWorker(i);
    if (worker) {
      scheduler_workers_.push_back(worker);
    }
  }

  // Update IpcManager with the number of workers
  IpcManager *ipc = CHI_IPC;
  if (ipc) {
    ipc->SetNumSchedQueues(total_workers);
  }

  HLOG(kInfo,
       "DefaultScheduler: {} scheduler workers, 1 network worker (worker {})",
       scheduler_workers_.size(), total_workers - 1);
}

u32 DefaultScheduler::ClientMapTask(IpcManager *ipc_manager,
                                    const Future<Task> &task) {
  // Get number of scheduling queues
  u32 num_lanes = ipc_manager->GetNumSchedQueues();
  if (num_lanes == 0) {
    return 0;
  }

  // Check if this is a network task (Send or Recv from admin pool)
  Task *task_ptr = task.get();
  if (task_ptr != nullptr && task_ptr->pool_id_ == chi::kAdminPoolId) {
    u32 method_id = task_ptr->method_;
    if (method_id == 14 || method_id == 15) {  // kSend or kRecv
      // Route to network worker (last worker)
      return num_lanes - 1;
    }
  }

  // Use PID+TID hash-based mapping for other tasks
  u32 lane = MapByPidTid(num_lanes);

  return lane;
}

u32 DefaultScheduler::RuntimeMapTask(Worker *worker, const Future<Task> &task) {
  // Check if this is a periodic Send or Recv task from admin pool
  Task *task_ptr = task.get();
  if (task_ptr != nullptr && task_ptr->IsPeriodic()) {
    if (task_ptr->pool_id_ == chi::kAdminPoolId) {
      u32 method_id = task_ptr->method_;
      if (method_id == 14 || method_id == 15) {  // kSend or kRecv
        // Schedule on network worker
        if (net_worker_ != nullptr) {
          return net_worker_->GetId();
        }
      }
    }
  }

  // All other tasks execute on the current worker
  if (worker != nullptr) {
    return worker->GetId();
  }
  return 0;
}

void DefaultScheduler::RebalanceWorker(Worker *worker) {
  // No rebalancing in default scheduler
  (void)worker;
}

void DefaultScheduler::AdjustPolling(RunContext *run_ctx) {
  if (!run_ctx) {
    return;
  }

  // TEMPORARY: Disable adaptive polling to test if it resolves hanging issues
  // Just return early without adjusting - tasks will use their configured
  // period
  return;

  // Maximum polling interval in microseconds (100ms)
  const double kMaxPollingIntervalUs = 100000.0;

  if (run_ctx->did_work_) {
    // Task did work - use the true (responsive) period
    run_ctx->yield_time_us_ = run_ctx->true_period_ns_ / 1000.0;
  } else {
    // Task didn't do work - increase polling interval (exponential backoff)
    double current_interval = run_ctx->yield_time_us_;

    // If uninitialized, start backoff from the true period
    if (current_interval <= 0.0) {
      current_interval = run_ctx->true_period_ns_ / 1000.0;
    }

    // Exponential backoff: double the interval
    double new_interval = current_interval * 2.0;

    // Cap at maximum polling interval
    if (new_interval > kMaxPollingIntervalUs) {
      new_interval = kMaxPollingIntervalUs;
    }

    run_ctx->yield_time_us_ = new_interval;
  }
}

u32 DefaultScheduler::MapByPidTid(u32 num_lanes) {
  // Use HSHM_SYSTEM_INFO to get both PID and TID for lane hashing
  auto *sys_info = HSHM_SYSTEM_INFO;
  pid_t pid = sys_info->pid_;
  auto tid = HSHM_THREAD_MODEL->GetTid();

  // Combine PID and TID for hashing to ensure different processes/threads use
  // different lanes
  size_t combined_hash =
      std::hash<pid_t>{}(pid) ^ (std::hash<void *>{}(&tid) << 1);
  return static_cast<u32>(combined_hash % num_lanes);
}

}  // namespace chi
