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
    HLOG(kError, "DefaultScheduler::DivideWorkers: ConfigManager not available");
    return;
  }

  u32 sched_count = config->GetSchedulerWorkerCount();
  u32 slow_count = config->GetSlowWorkerCount();
  u32 net_count = 1;  // Hardcoded to 1 network worker for now

  u32 total_workers = work_orch->GetTotalWorkerCount();
  u32 expected_workers = sched_count + slow_count + net_count;

  if (total_workers != expected_workers) {
    HLOG(kWarning,
         "DefaultScheduler::DivideWorkers: Worker count mismatch. "
         "Expected {}, got {}",
         expected_workers, total_workers);
  }

  // Clear any existing worker group assignments
  scheduler_workers_.clear();
  slow_workers_.clear();
  net_worker_ = nullptr;

  // Assign workers to groups based on configuration
  // Order: first sched_count workers -> scheduler_workers_
  //        next slow_count workers -> slow_workers_
  //        last worker -> net_worker_
  u32 worker_idx = 0;

  // Assign scheduler workers (fast tasks)
  for (u32 i = 0; i < sched_count && worker_idx < total_workers; ++i) {
    Worker *worker = work_orch->GetWorker(worker_idx);
    if (worker) {
      worker->SetThreadType(kSchedWorker);
      scheduler_workers_.push_back(worker);
      HLOG(kDebug, "DefaultScheduler: Worker {} assigned as kSchedWorker",
           worker_idx);
    }
    ++worker_idx;
  }

  // Assign slow workers (long-running tasks)
  for (u32 i = 0; i < slow_count && worker_idx < total_workers; ++i) {
    Worker *worker = work_orch->GetWorker(worker_idx);
    if (worker) {
      worker->SetThreadType(kSlow);
      slow_workers_.push_back(worker);
      HLOG(kDebug, "DefaultScheduler: Worker {} assigned as kSlow", worker_idx);
    }
    ++worker_idx;
  }

  // Assign network worker (last worker)
  for (u32 i = 0; i < net_count && worker_idx < total_workers; ++i) {
    Worker *worker = work_orch->GetWorker(worker_idx);
    if (worker) {
      worker->SetThreadType(kNetWorker);
      net_worker_ = worker;
      HLOG(kDebug, "DefaultScheduler: Worker {} assigned as kNetWorker",
           worker_idx);
    }
    ++worker_idx;
  }

  HLOG(kInfo,
       "DefaultScheduler::DivideWorkers: Partitioned {} workers "
       "(sched={}, slow={}, net={})",
       total_workers, sched_count, slow_count, net_count);
}

u32 DefaultScheduler::ClientMapTask(IpcManager *ipc_manager,
                                     const Future<Task> &task) {
  // Get number of scheduling queues
  u32 num_lanes = ipc_manager->GetNumSchedQueues();
  HLOG(kDebug, "ClientMapTask: num_sched_queues={}", num_lanes);
  if (num_lanes == 0) {
    return 0;
  }

  // Always use PID+TID hash-based mapping
  u32 lane = MapByPidTid(num_lanes);
  HLOG(kDebug, "ClientMapTask: PID+TID hash mapped to lane {}", lane);
  return lane;
}

u32 DefaultScheduler::RuntimeMapTask(Worker *worker, const Future<Task> &task) {
  // Return current worker - no migration in default scheduler
  // The task will execute on whichever worker picked it up
  (void)task;  // Unused in default scheduler
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

void DefaultScheduler::AssignToWorkerType(ThreadType thread_type,
                                          Future<Task> &future) {
  if (future.IsNull()) {
    return;
  }

  // Select target worker vector based on thread type
  std::vector<Worker *> *target_workers = nullptr;
  std::atomic<size_t> *idx = nullptr;

  if (thread_type == kSchedWorker) {
    target_workers = &scheduler_workers_;
    idx = &scheduler_idx_;
  } else if (thread_type == kSlow) {
    target_workers = &slow_workers_;
    idx = &slow_idx_;
  } else {
    // Process reaper or other types - not supported for task routing
    return;
  }

  if (target_workers->empty()) {
    HLOG(kWarning, "AssignToWorkerType: No workers of type {}",
          static_cast<int>(thread_type));
    return;
  }

  // Round-robin assignment
  size_t worker_idx = idx->fetch_add(1) % target_workers->size();
  Worker *worker = (*target_workers)[worker_idx];

  // Get the worker's assigned lane and emplace the task
  TaskLane *lane = worker->GetLane();
  if (lane != nullptr) {
    // Emplace the Future into the lane
    lane->Emplace(future);
  }
}

}  // namespace chi
