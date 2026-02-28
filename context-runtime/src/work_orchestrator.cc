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
 * Work orchestrator implementation
 */

#include "chimaera/work_orchestrator.h"

#include <chrono>
#include <cstdlib>
#include <iostream>

#include "chimaera/container.h"
#include "chimaera/pool_manager.h"
#include "chimaera/singletons.h"
#include "chimaera/ipc_manager.h"

// Global pointer variable definition for Work Orchestrator singleton
HSHM_DEFINE_GLOBAL_PTR_VAR_CC(chi::WorkOrchestrator, g_work_orchestrator);

namespace chi {

//===========================================================================
// Work Orchestrator Implementation
//===========================================================================

// Constructor and destructor removed - handled by HSHM singleton pattern

bool WorkOrchestrator::Init() {
  if (is_initialized_) {
    return true;
  }

  // Initialize HSHM TLS key for workers
  if (!chi_cur_worker_key_created_) {
    HSHM_THREAD_MODEL->CreateTls<class Worker>(chi_cur_worker_key_, nullptr);
    chi_cur_worker_key_created_ = true;
  }

  // Initialize scheduling state
  next_worker_index_for_scheduling_.store(0);
  active_lanes_ = nullptr;

  // Initialize HSHM thread group first
  auto thread_model = HSHM_THREAD_MODEL;
  thread_group_ = thread_model->CreateThreadGroup({});

  ConfigManager *config = CHI_CONFIG_MANAGER;
  if (!config) {
    return false;  // Configuration manager not initialized
  }

  // Get total worker count from configuration
  // Note: max(1, num_threads-1) are task workers, last worker is dedicated network worker
  // Exception: If num_threads=1, that single worker serves both roles
  u32 num_threads = config->GetNumThreads();
  u32 total_workers = num_threads;
  u32 num_task_workers = (num_threads > 1) ? (num_threads - 1) : 1;

  if (num_threads == 1) {
    HLOG(kInfo, "Creating 1 worker (serves both task and network roles)");
  } else {
    HLOG(kInfo, "Creating {} workers ({} task + 1 dedicated network)",
         total_workers, num_task_workers);
  }

  // Create all workers
  // The scheduler will partition them into groups via DivideWorkers()
  for (u32 i = 0; i < total_workers; ++i) {
    if (!CreateWorker()) {
      return false;
    }
  }

  // Mark as initialized so GetWorker() works during DivideWorkers
  is_initialized_ = true;

  // Get scheduler from IpcManager (IpcManager is the single owner)
  scheduler_ = CHI_IPC->GetScheduler();
  HLOG(kDebug, "WorkOrchestrator: Using scheduler from IpcManager");

  // Let the scheduler partition workers into groups (sched, slow, net)
  // This sets thread types and populates scheduler_workers_, slow_workers_, net_worker_
  if (scheduler_) {
    scheduler_->DivideWorkers(this);
    HLOG(kDebug, "WorkOrchestrator: Scheduler DivideWorkers completed");
  }

  return true;
}

void WorkOrchestrator::Finalize() {
  if (!is_initialized_) {
    return;
  }

  // Stop workers if running
  if (workers_running_) {
    StopWorkers();
  }

  // StopWorkers already joined threads (with timeout), just clear the vectors
  worker_threads_.clear();

  // Clear worker containers
  all_workers_.clear();
  workers_.clear();

  is_initialized_ = false;
}

bool WorkOrchestrator::StartWorkers() {
  if (!is_initialized_ || workers_running_) {
    return false;
  }

  // Spawn worker threads using HSHM thread model
  if (!SpawnWorkerThreads()) {
    return false;
  }

  workers_running_ = true;
  return true;
}

void WorkOrchestrator::StopWorkers() {
  if (!workers_running_) {
    return;
  }

  HLOG(kDebug, "Stopping {} worker threads...", all_workers_.size());

  // Stop all workers and wake them from epoll_wait
  pid_t runtime_pid = getpid();
  for (auto *worker : all_workers_) {
    if (worker) {
      worker->Stop();
      // Wake worker from epoll_wait so it can observe is_running_ == false
      TaskLane *lane = worker->GetLane();
      if (lane) {
        pid_t tid = lane->GetTid();
        if (tid > 0) {
          hshm::lbm::EventManager::Signal(runtime_pid, tid);
        }
      }
    }
  }

  // Wait for worker threads to finish using HSHM thread model with timeout
  auto thread_model = HSHM_THREAD_MODEL;
  auto start_time = std::chrono::steady_clock::now();
  const auto timeout_duration = std::chrono::seconds(5); // 5 second timeout

  size_t joined_count = 0;
  for (auto &thread : worker_threads_) {
    auto elapsed = std::chrono::steady_clock::now() - start_time;
    if (elapsed > timeout_duration) {
      HLOG(kError, "Warning: Worker thread join timeout reached. Some threads "
                    "may not have stopped gracefully.");
      break;
    }

    thread_model->Join(thread);
    joined_count++;
  }

  HLOG(kDebug, "Joined {} of {} worker threads", joined_count,
        worker_threads_.size());
  workers_running_ = false;
}

Worker *WorkOrchestrator::GetWorker(u32 worker_id) const {
  if (!is_initialized_ || worker_id >= all_workers_.size()) {
    return nullptr;
  }

  return all_workers_[worker_id];
}

size_t WorkOrchestrator::GetWorkerCount() const {
  return is_initialized_ ? all_workers_.size() : 0;
}

bool WorkOrchestrator::IsInitialized() const { return is_initialized_; }

bool WorkOrchestrator::AreWorkersRunning() const { return workers_running_; }

bool WorkOrchestrator::SpawnWorkerThreads() {
  // Get IPC Manager to access worker queues
  IpcManager *ipc = CHI_IPC;
  if (!ipc) {
    return false;
  }

  // Get the worker queues (task queue)
  TaskQueue *worker_queues = ipc->GetTaskQueue();
  if (!worker_queues) {
    HLOG(kError,
          "WorkOrchestrator: Worker queues not available for lane mapping");
    return false;
  }

  u32 num_lanes = worker_queues->GetNumLanes();
  if (num_lanes == 0) {
    HLOG(kError, "WorkOrchestrator: Worker queues have no lanes");
    return false;
  }

  // All workers process tasks - assign each worker to a lane using 1:1 mapping
  u32 num_workers = static_cast<u32>(all_workers_.size());
  HLOG(kInfo, "WorkOrchestrator: num_workers={}, num_lanes={}",
        num_workers, num_lanes);

  if (num_workers == 0) {
    HLOG(kError, "WorkOrchestrator: No workers available for lane mapping");
    return false;
  }

  // Map lanes to workers using 1:1 mapping
  // Each worker gets exactly one lane
  for (u32 worker_idx = 0; worker_idx < num_workers; ++worker_idx) {
    Worker *worker = all_workers_[worker_idx];
    if (worker) {
      // Direct 1:1 mapping: worker i gets lane i
      u32 lane_id = worker_idx;
      TaskLane *lane = &worker_queues->GetLane(lane_id, 0);

      // Set the worker's assigned lane
      worker->SetLane(lane);

      // Mark the lane with the assigned worker ID
      lane->SetAssignedWorkerId(worker->GetId());

      HLOG(kInfo, "WorkOrchestrator: Mapped worker {} (ID {}) to lane {}",
            worker_idx, worker->GetId(), lane_id);
    } else {
      HLOG(kWarning, "WorkOrchestrator: Worker at index {} is null", worker_idx);
    }
  }

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
  // Assign GPU lanes only to the designated GPU worker
  size_t num_gpus = ipc->GetGpuQueueCount();
  if (num_gpus > 0 && scheduler_) {
    Worker *gpu_worker = scheduler_->GetGpuWorker();
    if (gpu_worker) {
      std::vector<TaskLane *> gpu_lanes;
      gpu_lanes.reserve(num_gpus);
      for (size_t gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
        TaskQueue *gpu_queue = ipc->GetGpuQueue(gpu_id);
        if (gpu_queue) {
          TaskLane *gpu_lane = &gpu_queue->GetLane(0, 0);
          gpu_lanes.push_back(gpu_lane);
          gpu_lane->SetAssignedWorkerId(gpu_worker->GetId());
        }
      }
      gpu_worker->SetGpuLanes(gpu_lanes);
      HLOG(kInfo, "WorkOrchestrator: Assigned {} GPU lane(s) to GPU worker {}",
           gpu_lanes.size(), gpu_worker->GetId());
    } else {
      HLOG(kWarning, "WorkOrchestrator: {} GPU queue(s) available but no GPU worker designated",
           num_gpus);
    }
  }
#endif

  // Use HSHM thread model to spawn worker threads
  auto thread_model = HSHM_THREAD_MODEL;
  worker_threads_.reserve(all_workers_.size());

  try {
    for (size_t i = 0; i < all_workers_.size(); ++i) {
      auto *worker = all_workers_[i];
      if (worker) {
        // Spawn thread using HSHM thread model
        hshm::thread::Thread thread = thread_model->Spawn(
            thread_group_, [worker](int tid) { worker->Run(); },
            static_cast<int>(i));
        worker_threads_.emplace_back(std::move(thread));
      }
    }

    // Note: DivideWorkers() is called in Init() before workers are spawned
    // Workers are already partitioned into groups at this point

    return true;
  } catch (const std::exception &e) {
    return false;
  }
}

bool WorkOrchestrator::CreateWorker() {
  u32 worker_id = static_cast<u32>(all_workers_.size());
  auto worker = std::make_unique<Worker>(worker_id);

  if (!worker->Init()) {
    return false;
  }

  Worker *worker_ptr = worker.get();
  all_workers_.push_back(worker_ptr);

  // Add to ownership container (workers_ owns all worker unique_ptrs)
  workers_.push_back(std::move(worker));

  return true;
}

bool WorkOrchestrator::CreateWorkers(u32 count) {
  for (u32 i = 0; i < count; ++i) {
    if (!CreateWorker()) {
      return false;
    }
  }

  return true;
}

//===========================================================================
// Lane Scheduling Methods
//===========================================================================

bool WorkOrchestrator::ServerInitQueues(u32 num_lanes) {
  // Initialize process queues for different priorities
  bool success = true;

  // No longer creating local queues - external queue is managed by IPC Manager
  return success;
}

bool WorkOrchestrator::HasWorkRemaining(u64 &total_work_remaining) const {
  total_work_remaining = 0;

  // Get PoolManager to access all containers in the system
  auto *pool_manager = CHI_POOL_MANAGER;
  if (!pool_manager || !pool_manager->IsInitialized()) {
    return false; // No pool manager means no work
  }

  // Get all pool IDs from the pool manager
  std::vector<PoolId> all_pool_ids = pool_manager->GetAllPoolIds();

  for (const auto &pool_id : all_pool_ids) {
    const PoolInfo *info = pool_manager->GetPoolInfo(pool_id);
    if (!info) continue;
    for (const auto &pair : info->containers_) {
      if (pair.second) {
        total_work_remaining += pair.second->GetWorkRemaining();
      }
    }
  }

  return total_work_remaining > 0;
}

} // namespace chi