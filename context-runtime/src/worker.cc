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
 * Worker implementation
 *
 * Uses C++20 stackless coroutines for task suspension and resumption.
 * Coroutines are managed via std::coroutine_handle stored in RunContext.
 */

#include "chimaera/worker.h"

#include <sys/syscall.h>
#include <unistd.h>

#include <coroutine>
#include <cstdlib>
#include <iostream>
#include <unordered_set>

// Include task_queue.h before other chimaera headers to ensure proper
// resolution
#include "chimaera/admin/admin_client.h"
#include "chimaera/container.h"
#include "chimaera/ipc_manager.h"
#include "chimaera/pool_manager.h"
#include "chimaera/singletons.h"
#include "chimaera/task.h"
#include "chimaera/task_archives.h"
#include "chimaera/task_queue.h"
#include "chimaera/work_orchestrator.h"

namespace chi {

// Stack detection is now handled by WorkOrchestrator during initialization

Worker::Worker(u32 worker_id)
    : worker_id_(worker_id),
      is_running_(false),
      is_initialized_(false),
      did_work_(false),
      task_did_work_(false),
      current_run_context_(nullptr),
      assigned_lane_(nullptr),
      event_queue_(nullptr),
      last_long_queue_check_(0),
      iteration_count_(0),
      idle_iterations_(0),
      current_sleep_us_(0),
      sleep_count_(0) {
  // std::queue is initialized with default constructors in member
  // initialization No pre-allocation of capacity is needed or possible with
  // std::queue

  // Record worker spawn time
  spawn_time_.Now();
}

Worker::~Worker() {
  if (is_initialized_) {
    Finalize();
  }
}

bool Worker::Init() {
  if (is_initialized_) {
    return true;
  }

  // Stack management simplified - no pool needed
  // Note: assigned_lane_ will be set by WorkOrchestrator during external queue
  // initialization

  // Allocate and initialize event queue from malloc allocator (temporary
  // runtime data). Stores Future<Task> objects to avoid stale RunContext* pointers.
  event_queue_ = HSHM_MALLOC
                     ->template NewObj<hshm::ipc::mpsc_ring_buffer<
                         Future<Task, CHI_MAIN_ALLOC_T>, hshm::ipc::MallocAllocator>>(
                         HSHM_MALLOC, EVENT_QUEUE_DEPTH)
                     .ptr_;

  // Get scheduler from IpcManager (IpcManager is the single owner)
  scheduler_ = CHI_IPC->GetScheduler();
  HLOG(kDebug, "Worker {}: Using scheduler from IpcManager", worker_id_);

  // Create SHM lightbeam transports for worker-side transport
  shm_send_transport_ = hshm::lbm::TransportFactory::Get(
      "", hshm::lbm::TransportType::kShm,
      hshm::lbm::TransportMode::kClient);
  shm_recv_transport_ = hshm::lbm::TransportFactory::Get(
      "", hshm::lbm::TransportType::kShm,
      hshm::lbm::TransportMode::kServer);

  is_initialized_ = true;
  return true;
}

void Worker::SetTaskDidWork(bool did_work) { task_did_work_ = did_work; }

bool Worker::GetTaskDidWork() const { return task_did_work_; }

WorkerStats Worker::GetWorkerStats() const {
  WorkerStats stats;

  // Basic worker info
  stats.worker_id_ = worker_id_;
  stats.is_running_ = is_running_;
  stats.idle_iterations_ = idle_iterations_;

  // Calculate number of queued tasks (tasks waiting in the assigned lane)
  stats.num_queued_tasks_ = 0;
  stats.is_active_ = false;
  if (assigned_lane_) {
    stats.num_queued_tasks_ = assigned_lane_->Size();
    stats.is_active_ = assigned_lane_->IsActive();
  }

  // Count blocked tasks across all blocked queues
  stats.num_blocked_tasks_ = 0;
  for (u32 i = 0; i < NUM_BLOCKED_QUEUES; ++i) {
    stats.num_blocked_tasks_ += blocked_queues_[i].size();
  }

  // Count periodic tasks across all periodic queues
  stats.num_periodic_tasks_ = 0;
  for (u32 i = 0; i < NUM_PERIODIC_QUEUES; ++i) {
    stats.num_periodic_tasks_ += periodic_queues_[i].size();
  }

  // Get suspend period (time until next periodic task or 0 if none)
  double suspend_period = GetSuspendPeriod();
  stats.suspend_period_us_ =
      (suspend_period < 0) ? 0 : static_cast<u32>(suspend_period);

  // Note: num_tasks_processed_ would require adding a counter to the worker
  // For now, set to 0 - can be added later if needed
  stats.num_tasks_processed_ = 0;

  return stats;
}

hshm::lbm::EventManager& Worker::GetEventManager() { return event_manager_; }

void Worker::Finalize() {
  if (!is_initialized_) {
    return;
  }

  Stop();

  // Note: Context cache cleanup removed - RunContext is now embedded in Task

  // Clean up all blocked queues (2 queues)
  for (u32 i = 0; i < NUM_BLOCKED_QUEUES; ++i) {
    while (!blocked_queues_[i].empty()) {
      RunContext *run_ctx = blocked_queues_[i].front();
      blocked_queues_[i].pop();
      // RunContexts in blocked queues are still in use - don't free them
      // They will be cleaned up when the tasks complete or by stack cache
      (void)run_ctx;  // Suppress unused variable warning
    }
  }

  // Clear assigned lane reference (don't delete - it's in shared memory)
  assigned_lane_ = nullptr;

  is_initialized_ = false;
}

void Worker::Run() {
  if (!is_initialized_) {
    return;
  }
  HLOG(kInfo, "Worker {}: Running", worker_id_);

  // Set current worker once for the entire thread duration
  SetAsCurrentWorker();
  is_running_ = true;

  // Set up thread ID and signal event via EventManager
  pid_t tid = static_cast<pid_t>(syscall(SYS_gettid));
  if (assigned_lane_) {
    assigned_lane_->SetTid(tid);
  }
  event_manager_.AddSignalEvent(nullptr);
  HLOG(kInfo, "Worker {}: EventManager signal event added, tid={}", worker_id_,
       tid);

  // Main worker loop - process tasks from assigned lane
  while (is_running_) {
    did_work_ = false;  // Reset work tracker at start of each loop iteration
    task_did_work_ = false;  // Reset task-level work tracker

    // Process tasks from assigned lane
    if (assigned_lane_) {
      u32 count = ProcessNewTasks(assigned_lane_);
      if (count > 0) did_work_ = true;
    }
#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
    for (auto *gpu_lane : gpu_lanes_) {
      u32 count = ProcessNewTasks(gpu_lane);
      if (count > 0) did_work_ = true;
    }
#endif

    // Check blocked queue for completed tasks at end of each iteration
    ContinueBlockedTasks(false);

    // Copy task output data to copy space for streaming (low-priority
    // operation) Only do this when worker would otherwise idle, with minimal
    // time budget
    if (!did_work_) {
      CopyTaskOutputToClient();
    }

    // Increment iteration counter
    iteration_count_++;

    if (!did_work_) {
      // No work was done - suspend worker with adaptive sleep
      SuspendMe();
    }

    if (did_work_) {
      // Work was done - reset idle counters
      idle_iterations_ = 0;
      current_sleep_us_ = 0;
      sleep_count_ = 0;
      did_work_ = false;
    }
  }

  // EventManager destructor handles signalfd and epoll cleanup
}

void Worker::Stop() { is_running_ = false; }

void Worker::SetLane(TaskLane *lane) {
  assigned_lane_ = lane;
  // Mark lane as active when assigned to worker
  if (assigned_lane_) {
    assigned_lane_->SetActive(true);
  }
}

TaskLane *Worker::GetLane() const { return assigned_lane_; }

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
void Worker::SetGpuLanes(const std::vector<TaskLane *> &lanes) {
  gpu_lanes_ = lanes;
}

const std::vector<TaskLane *> &Worker::GetGpuLanes() const {
  return gpu_lanes_;
}
#endif


hipc::FullPtr<Task> Worker::GetOrCopyTaskFromFuture(Future<Task> &future,
                                                    Container *container,
                                                    u32 method_id) {
  auto future_shm = future.GetFutureShm();
  FullPtr<Task> task_full_ptr = future.GetTaskPtr();

  // Check FUTURE_COPY_FROM_CLIENT flag to determine if task needs to be loaded
  if (future_shm->flags_.Any(FutureShm::FUTURE_COPY_FROM_CLIENT) &&
      !future_shm->flags_.Any(FutureShm::FUTURE_WAS_COPIED)) {
    // CLIENT PATH: Load task from serialized data in FutureShm copy_space
    // Only copy if not already copied (FUTURE_WAS_COPIED not set)

    // Build SHM context for transfer
    hshm::lbm::LbmContext ctx;
    ctx.copy_space = future_shm->copy_space;
    ctx.shm_info_ = &future_shm->input_;

    // Receive via SHM transport (blocking - spins until client sends)
    LoadTaskArchive archive;
    auto info = shm_recv_transport_->Recv(archive, ctx);
    (void)info;

    // Allocate and deserialize task
    task_full_ptr = container->AllocLoadTask(method_id, archive);

    // Update the Future's task pointer
    future.GetTaskPtr() = task_full_ptr;

    // Mark as copied to prevent re-copying if task migrates between workers
    future_shm->flags_.SetBits(FutureShm::FUTURE_WAS_COPIED);
  }
  // RUNTIME PATH or ALREADY COPIED: Task pointer is already set in future

  return task_full_ptr;
}

u32 Worker::ProcessNewTasks(TaskLane *lane) {
  const u32 MAX_TASKS_PER_ITERATION = 16;
  u32 tasks_processed = 0;

  if (!lane) {
    return 0;
  }

  while (tasks_processed < MAX_TASKS_PER_ITERATION) {
    if (ProcessNewTask(lane)) {
      tasks_processed++;
    } else {
      break;
    }
  }

  return tasks_processed;
}

bool Worker::ProcessNewTask(TaskLane *lane) {
  Future<Task> future;
  // Pop Future<Task> from lane
  if (!lane->Pop(future)) {
    return false;
  }

  HLOG(kDebug, "Worker {}: Popped future from lane, processing task",
       worker_id_);
  SetCurrentRunContext(nullptr);

  // Get FutureShm (allocator is pre-registered by Admin::RegisterMemory)
  auto future_shm = future.GetFutureShm();
  if (future_shm.IsNull()) {
    HLOG(kError, "Worker {}: Failed to get FutureShm (null pointer)",
         worker_id_);
    return true;
  }

  // Get pool_id and method_id from FutureShm
  PoolId pool_id = future_shm->pool_id_;
  u32 method_id = future_shm->method_id_;

  // Get container for routing
  auto *pool_manager = CHI_POOL_MANAGER;
  Container *container = pool_manager->GetContainer(pool_id);

  if (!container) {
    // Container not found - mark as complete with error
    HLOG(kError, "Worker {}: Container not found for pool_id={}, method={}",
         worker_id_, pool_id, method_id);
    // Set both error bit AND FUTURE_COMPLETE so client doesn't hang
    future_shm->flags_.SetBits(1 | FutureShm::FUTURE_COMPLETE);
    return true;
  }

  // Get or copy task from Future (handles deserialization if needed)
  FullPtr<Task> task_full_ptr =
      GetOrCopyTaskFromFuture(future, container, method_id);

  // Check if task deserialization failed
  if (task_full_ptr.IsNull()) {
    HLOG(kError,
         "Worker {}: Failed to deserialize task for pool_id={}, method={}",
         worker_id_, pool_id, method_id);
    // Mark as complete with error so client doesn't hang
    future_shm->flags_.SetBits(1 | FutureShm::FUTURE_COMPLETE);
    return true;
  }

  HLOG(kDebug,
       "Worker {}: Task deserialized successfully, task_ptr={}, checking "
       "if routed",
       worker_id_, (void *)task_full_ptr.ptr_);

  // Allocate RunContext before routing (skip if already created)
  if (!task_full_ptr->task_flags_.Any(TASK_RUN_CTX_EXISTS)) {
    HLOG(kDebug, "Worker {}: RunContext not yet created, calling BeginTask",
         worker_id_);
    BeginTask(future, container, lane);
  }

  // Route task using consolidated routing function
  if (RouteTask(future, lane, container)) {
    // Routing successful, execute the task
#if HSHM_IS_HOST
    RunContext *run_ctx = task_full_ptr->run_ctx_.get();
    ExecTask(task_full_ptr, run_ctx, false);
#endif
  }
  // Note: RouteTask returning false doesn't always indicate an error
  // Real errors are handled within RouteTask itself

  return true;
}

double Worker::GetSuspendPeriod() const {
  // Scan all periodic queues to find the maximum yield_time (polling period)
  // We use the maximum yield_time directly, not the remaining time, to avoid
  // desynchronization issues when multiple tasks have the same period but
  // slightly different block_start timestamps
  double max_yield_time_us = 0;
  bool found_task = false;

  // Check all periodic queues (0-3)
  for (u32 queue_idx = 0; queue_idx < NUM_PERIODIC_QUEUES; ++queue_idx) {
    const std::queue<RunContext *> &queue = periodic_queues_[queue_idx];

    if (queue.empty()) {
      continue;
    }

    // Check just the front task of each queue (representative of the queue's
    // period)
    RunContext *run_ctx = queue.front();

    if (!run_ctx || run_ctx->task_.IsNull()) {
      continue;
    }

    // Use the yield_time directly - this is the adaptive polling period
    // No elapsed time calculation to avoid desynchronization
    if (!found_task || run_ctx->yield_time_us_ > max_yield_time_us) {
      max_yield_time_us = run_ctx->yield_time_us_;
      found_task = true;
    }
  }

  // Return -1 if no periodic tasks (means wait indefinitely in epoll_wait)
  // Otherwise return the maximum yield_time across all periodic queues
  return found_task ? max_yield_time_us : -1;
}

void Worker::SuspendMe() {
#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
  // GPU workers must never sleep — they need to poll GPU lanes continuously
  if (!gpu_lanes_.empty()) {
    return;
  }
#endif

  // No work was done in this iteration - increment idle counter
  idle_iterations_++;

  // Set idle start time on first idle iteration
  if (idle_iterations_ == 1) {
    idle_start_.Now();
  }

  // Get configuration parameters
  auto *config = CHI_CONFIG_MANAGER;
  u32 first_busy_wait = config->GetFirstBusyWait();
  u32 max_sleep = config->GetMaxSleep();

  // Calculate actual elapsed idle time
  hshm::Timepoint current_time;
  current_time.Now();
  double elapsed_idle_us = idle_start_.GetUsecFromStart(current_time);

  if (elapsed_idle_us < first_busy_wait) {
    // Still in busy wait period - just return
    return;
  } else {
    // Past busy wait period - use epoll
    // Before sleeping, check blocked queues with force=true
    ContinueBlockedTasks(true);

    // If task_did_work_ is true, blocked tasks were found - don't sleep
    if (GetTaskDidWork()) {
      return;
    }

    // Mark worker as inactive (blocked in epoll_wait)
    if (assigned_lane_) {
      assigned_lane_->SetActive(false);
    }

    // Calculate timeout from periodic tasks
    double suspend_period_us = GetSuspendPeriod();
    int timeout_us = (suspend_period_us < 0) ? -1
        : static_cast<int>(suspend_period_us);

    // Wait for signal using EventManager
    int nfds = event_manager_.Wait(timeout_us);

    // Mark worker as active again
    if (assigned_lane_) {
      assigned_lane_->SetActive(true);
    }

    if (nfds == 0) {
      sleep_count_++;
    }

    // Force immediate rescan of all periodic tasks after waking
    ContinueBlockedTasks(true);
  }
}

u32 Worker::GetId() const { return worker_id_; }

bool Worker::IsRunning() const { return is_running_; }

RunContext *Worker::GetCurrentRunContext() const {
  return current_run_context_;
}

RunContext *Worker::SetCurrentRunContext(RunContext *rctx) {
  current_run_context_ = rctx;
  return current_run_context_;
}

FullPtr<Task> Worker::GetCurrentTask() const {
  RunContext *run_ctx = GetCurrentRunContext();
  if (!run_ctx) {
    return FullPtr<Task>::GetNull();
  }
  return run_ctx->task_;
}

Container *Worker::GetCurrentContainer() const {
  RunContext *run_ctx = GetCurrentRunContext();
  if (!run_ctx) {
    return nullptr;
  }
  return run_ctx->container_;
}

TaskLane *Worker::GetCurrentLane() const {
  RunContext *run_ctx = GetCurrentRunContext();
  if (!run_ctx) {
    return nullptr;
  }
  return run_ctx->lane_;
}

void Worker::SetAsCurrentWorker() {
  HSHM_THREAD_MODEL->SetTls(chi_cur_worker_key_,
                            static_cast<class Worker *>(this));
}

void Worker::ClearCurrentWorker() {
  HSHM_THREAD_MODEL->SetTls(chi_cur_worker_key_,
                            static_cast<class Worker *>(nullptr));
}

bool Worker::RouteTask(Future<Task> &future, TaskLane *lane,
                       Container *container) {
  // Get task pointer from future
  FullPtr<Task> task_ptr = future.GetTaskPtr();

  if (task_ptr.IsNull()) {
    HLOG(kWarning, "Worker {}: RouteTask - task_ptr is null", worker_id_);
    return false;
  }

  // Check if task has already been routed - if so, return true immediately
  if (task_ptr->IsRouted()) {
    return (container != nullptr);
  }

  // Initialize exec_mode to kExec by default
  if (task_ptr->run_ctx_) {
    RunContext *run_ctx = task_ptr->run_ctx_.get();
    run_ctx->exec_mode_ = ExecMode::kExec;
  }

  // Resolve pool query and route task to container
  // Note: ResolveDynamicQuery may override exec_mode to kDynamicSchedule
  std::vector<PoolQuery> pool_queries =
      ResolvePoolQuery(task_ptr->pool_query_, task_ptr->pool_id_, task_ptr);

  // Check if pool_queries is empty - this indicates an error in resolution
  if (pool_queries.empty()) {
    HLOG(kError,
         "Worker {}: Task routing failed - no pool queries resolved. "
         "Pool ID: {}, Method: {}",
         worker_id_, task_ptr->pool_id_, task_ptr->method_);

    // RunContext is already allocated, just return false to indicate local
    // execution
    return false;
  }

  // Check if task should be processed locally
  bool is_local = IsTaskLocal(task_ptr, pool_queries);
  if (is_local) {
    // Route task locally using container query and Monitor with kLocalSchedule
    return RouteLocal(future, lane, container);
  } else {
    // Route task globally using admin client's ClientSendTaskIn method
    // RouteGlobal never fails, so no need for fallback logic
    RouteGlobal(future, pool_queries);
    return false;  // No local execution needed
  }
}

bool Worker::IsTaskLocal(const FullPtr<Task> &task_ptr,
                         const std::vector<PoolQuery> &pool_queries) {
  // If task has TASK_FORCE_NET flag, force it through network code
  if (task_ptr->task_flags_.Any(TASK_FORCE_NET)) {
    return false;
  }

  // If there's only one node, all tasks are local
  auto *ipc_manager = CHI_IPC;
  if (ipc_manager && ipc_manager->GetNumHosts() == 1) {
    return true;
  }

  // Task is local only if there is exactly one pool query
  if (pool_queries.size() != 1) {
    return false;
  }

  const PoolQuery &query = pool_queries[0];

  // Check routing mode first, then specific conditions
  RoutingMode routing_mode = query.GetRoutingMode();

  switch (routing_mode) {
    case RoutingMode::Local:
      return true;  // Always local

    case RoutingMode::Dynamic:
      // Dynamic mode routes to Monitor first, then may be resolved locally
      // Treat as local for initial routing to allow Monitor to process
      return true;

    case RoutingMode::Physical: {
      // Physical mode is local only if targeting local node
      auto *ipc_manager = CHI_IPC;
      u64 local_node_id = ipc_manager ? ipc_manager->GetNodeId() : 0;
      return query.GetNodeId() == local_node_id;
    }

    case RoutingMode::DirectId:
    case RoutingMode::DirectHash:
    case RoutingMode::Range:
    case RoutingMode::Broadcast:
      // These modes should have been resolved to Physical queries by now
      // If we still see them here, they are not local
      return false;
  }

  return false;
}

bool Worker::RouteLocal(Future<Task> &future, TaskLane *lane,
                        Container *container) {
  // Get task pointer from future
  FullPtr<Task> task_ptr = future.GetTaskPtr();

  // Mark as routed so the task is not re-routed on subsequent passes.
  // Tasks are already placed on the correct worker's lane by
  // ClientMapTask/Send, so we always execute locally here.
  task_ptr->SetFlags(TASK_ROUTED);

  // Execute task locally (container is provided by caller)
  if (!container) {
    HLOG(kError, "Worker {}: RouteLocal - container not found for pool_id={}",
         worker_id_, task_ptr->pool_id_);
    return false;
  }

  // Set the completer_ field to track which container will execute this task
  task_ptr->SetCompleter(container->container_id_);

  auto *ipc_manager = CHI_IPC;
  u32 node_id = ipc_manager->GetNodeId();

  // Routing successful - caller should execute the task locally
  return true;
}

bool Worker::RouteGlobal(Future<Task> &future,
                         const std::vector<PoolQuery> &pool_queries) {
  // Get task pointer from future
  FullPtr<Task> task_ptr = future.GetTaskPtr();

  auto *ipc_manager = CHI_IPC;

  // Log the global routing for debugging
  if (!pool_queries.empty()) {
    const auto &query = pool_queries[0];
    HLOG(kInfo,
         "Worker {}: RouteGlobal - routing task method={}, pool_id={} to node "
         "{} (routing_mode={})",
         worker_id_, task_ptr->method_, task_ptr->pool_id_, query.GetNodeId(),
         static_cast<int>(query.GetRoutingMode()));
  }

  // Store pool_queries in task's RunContext for SendIn to access
  if (task_ptr->run_ctx_) {
    RunContext *run_ctx = task_ptr->run_ctx_.get();
    run_ctx->pool_queries_ = pool_queries;
  }

  // Enqueue the original task directly to net_queue_ priority 0 (SendIn)
  ipc_manager->EnqueueNetTask(future, NetQueuePriority::kSendIn);

  // Set TASK_ROUTED flag on original task
  task_ptr->SetFlags(TASK_ROUTED);

  HLOG(kInfo, "Worker {}: RouteGlobal - task enqueued to net_queue",
       worker_id_);

  // Always return true (never fail)
  return true;
}

std::vector<PoolQuery> Worker::ResolvePoolQuery(const PoolQuery &query,
                                                PoolId pool_id,
                                                const FullPtr<Task> &task_ptr) {
  // Basic validation
  if (pool_id.IsNull()) {
    return {};  // Invalid pool ID
  }

  RoutingMode routing_mode = query.GetRoutingMode();
  std::vector<PoolQuery> result;

  switch (routing_mode) {
    case RoutingMode::Local:
      result = ResolveLocalQuery(query, task_ptr);
      break;
    case RoutingMode::Dynamic:
      result = ResolveDynamicQuery(query, pool_id, task_ptr);
      break;
    case RoutingMode::DirectId:
      result = ResolveDirectIdQuery(query, pool_id, task_ptr);
      break;
    case RoutingMode::DirectHash:
      result = ResolveDirectHashQuery(query, pool_id, task_ptr);
      break;
    case RoutingMode::Range:
      result = ResolveRangeQuery(query, pool_id, task_ptr);
      break;
    case RoutingMode::Broadcast:
      result = ResolveBroadcastQuery(query, pool_id, task_ptr);
      break;
    case RoutingMode::Physical:
      result = ResolvePhysicalQuery(query, pool_id, task_ptr);
      break;
  }

  // Set ret_node_ on all resolved queries to this node's ID
  auto *ipc_manager = CHI_IPC;
  u32 this_node_id = ipc_manager->GetNodeId();
  for (auto &pq : result) {
    pq.SetReturnNode(this_node_id);
  }

  return result;
}

std::vector<PoolQuery> Worker::ResolveLocalQuery(
    const PoolQuery &query, const FullPtr<Task> &task_ptr) {
  // Local routing - process on current node
  return {query};
}

std::vector<PoolQuery> Worker::ResolveDynamicQuery(
    const PoolQuery &query, PoolId pool_id, const FullPtr<Task> &task_ptr) {
  // Ensure RunContext is initialized
  if (!task_ptr->run_ctx_) {
    task_ptr->run_ctx_ = std::make_unique<RunContext>();
  }

  // Use the current RunContext that is owned by the task
  RunContext *run_ctx = task_ptr->run_ctx_.get();

  // Set execution mode to kDynamicSchedule
  // This tells ExecTask to call RerouteDynamicTask instead of EndTask
  run_ctx->exec_mode_ = ExecMode::kDynamicSchedule;

  // Return Local query for execution
  // After task completes, RerouteDynamicTask will re-route with updated
  // pool_query
  std::vector<PoolQuery> result;
  result.push_back(PoolQuery::Local());
  return result;
}

std::vector<PoolQuery> Worker::ResolveDirectIdQuery(
    const PoolQuery &query, PoolId pool_id, const FullPtr<Task> &task_ptr) {
  auto *pool_manager = CHI_POOL_MANAGER;
  if (pool_manager == nullptr) {
    return {query};  // Fallback to original query
  }

  // Get the container ID from the query
  ContainerId container_id = query.GetContainerId();

  // Boundary case optimization: Check if container exists on this node
  if (pool_manager->HasContainer(pool_id, container_id)) {
    // Container is local, resolve to Local query
    return {PoolQuery::Local()};
  }

  // Get the physical node ID for this container
  u32 node_id = pool_manager->GetContainerNodeId(pool_id, container_id);

  // Create a Physical PoolQuery to that node
  return {PoolQuery::Physical(node_id)};
}

std::vector<PoolQuery> Worker::ResolveDirectHashQuery(
    const PoolQuery &query, PoolId pool_id, const FullPtr<Task> &task_ptr) {
  auto *pool_manager = CHI_POOL_MANAGER;
  if (pool_manager == nullptr) {
    return {query};  // Fallback to original query
  }

  // Get pool info to find the number of containers
  const PoolInfo *pool_info = pool_manager->GetPoolInfo(pool_id);
  if (pool_info == nullptr || pool_info->num_containers_ == 0) {
    return {query};  // Fallback to original query
  }

  // Hash to get container ID
  u32 hash_value = query.GetHash();
  ContainerId container_id = hash_value % pool_info->num_containers_;

  // Boundary case optimization: Check if container exists on this node
  if (pool_manager->HasContainer(pool_id, container_id)) {
    // Container is local, resolve to Local query
    return {PoolQuery::Local()};
  }

  // Get the physical node ID for this container
  u32 node_id = pool_manager->GetContainerNodeId(pool_id, container_id);

  // Create a Physical PoolQuery to that node
  return {PoolQuery::Physical(node_id)};
}

std::vector<PoolQuery> Worker::ResolveRangeQuery(
    const PoolQuery &query, PoolId pool_id, const FullPtr<Task> &task_ptr) {
  // Set execution mode to normal execution
  if (task_ptr->run_ctx_) {
    RunContext *run_ctx = task_ptr->run_ctx_.get();
    run_ctx->exec_mode_ = ExecMode::kExec;
  }

  auto *pool_manager = CHI_POOL_MANAGER;
  if (pool_manager == nullptr) {
    return {query};  // Fallback to original query
  }

  auto *config_manager = CHI_CONFIG_MANAGER;
  if (config_manager == nullptr) {
    return {query};  // Fallback to original query
  }

  u32 range_offset = query.GetRangeOffset();
  u32 range_count = query.GetRangeCount();

  // Validate range
  if (range_count == 0) {
    return {};  // Empty range
  }

  // Boundary case optimization: Check if single-container range is local
  if (range_count == 1) {
    ContainerId container_id = range_offset;
    if (pool_manager->HasContainer(pool_id, container_id)) {
      // Container is local, resolve to Local query
      return {PoolQuery::Local()};
    }
  }

  std::vector<PoolQuery> result_queries;

  // Get neighborhood size from configuration (maximum number of queries)
  u32 neighborhood_size = config_manager->GetNeighborhoodSize();

  // Calculate queries needed, capped at neighborhood_size
  u32 ideal_queries = (range_count + neighborhood_size - 1) / neighborhood_size;
  u32 queries_to_create = std::min(ideal_queries, neighborhood_size);

  // Create one query per container
  if (queries_to_create <= 1) {
    queries_to_create = range_count;
  }

  u32 containers_per_query = range_count / queries_to_create;
  u32 remaining_containers = range_count % queries_to_create;

  u32 current_offset = range_offset;
  for (u32 i = 0; i < queries_to_create; ++i) {
    u32 current_count = containers_per_query;
    if (i < remaining_containers) {
      current_count++;  // Distribute remainder across first queries
    }

    if (current_count > 0) {
      result_queries.push_back(PoolQuery::Range(current_offset, current_count));
      current_offset += current_count;
    }
  }

  return result_queries;
}

std::vector<PoolQuery> Worker::ResolveBroadcastQuery(
    const PoolQuery &query, PoolId pool_id, const FullPtr<Task> &task_ptr) {
  auto *pool_manager = CHI_POOL_MANAGER;
  if (pool_manager == nullptr) {
    return {query};  // Fallback to original query
  }

  // Get pool info to find the total number of containers
  const PoolInfo *pool_info = pool_manager->GetPoolInfo(pool_id);
  if (pool_info == nullptr || pool_info->num_containers_ == 0) {
    return {query};  // Fallback to original query
  }

  // Create a Range query that covers all containers, then resolve it
  PoolQuery range_query = PoolQuery::Range(0, pool_info->num_containers_);
  return ResolveRangeQuery(range_query, pool_id, task_ptr);
}

std::vector<PoolQuery> Worker::ResolvePhysicalQuery(
    const PoolQuery &query, PoolId pool_id, const FullPtr<Task> &task_ptr) {
  // Physical routing - query is already resolved to a specific node
  return {query};
}

// Note: AllocateContext and DeallocateContext functions have been removed
// RunContext is now embedded directly in the Task object, eliminating the need
// for separate allocation/deallocation and context caching

void Worker::BeginTask(Future<Task> &future, Container *container,
                       TaskLane *lane) {
  FullPtr<Task> task_ptr = future.GetTaskPtr();
  if (task_ptr.IsNull()) {
    return;
  }

#if HSHM_IS_HOST
  // Initialize or reset the task's owned RunContext
  task_ptr->run_ctx_ = std::make_unique<RunContext>();
  RunContext *run_ctx = task_ptr->run_ctx_.get();

  // Clear and initialize RunContext for new task execution
  run_ctx->worker_id_ = worker_id_;
  run_ctx->task_ = task_ptr;        // Store task in RunContext
  run_ctx->is_yielded_ = false;     // Initially not blocked
  run_ctx->container_ = container;  // Store container for CHI_CUR_CONTAINER
  run_ctx->lane_ = lane;            // Store lane for CHI_CUR_LANE
  run_ctx->event_queue_ = event_queue_;  // Set pointer to worker's event queue
  run_ctx->future_ = future;             // Store future in RunContext
  run_ctx->coro_handle_ = nullptr;       // Coroutine not started yet

  // Initialize adaptive polling fields for periodic tasks
  if (task_ptr->IsPeriodic()) {
    run_ctx->true_period_ns_ = task_ptr->period_ns_;
    run_ctx->yield_time_us_ =
        task_ptr->period_ns_ / 1000.0;  // Initialize with true period
    run_ctx->did_work_ = false;         // Initially no work done
  } else {
    run_ctx->true_period_ns_ = 0.0;
    run_ctx->yield_time_us_ = 0.0;
    run_ctx->did_work_ = false;
  }

  // Mark that RunContext now exists for this task
  task_ptr->SetFlags(TASK_RUN_CTX_EXISTS);

  // Set current run context
  SetCurrentRunContext(run_ctx);
#endif
}

void Worker::StartCoroutine(const FullPtr<Task> &task_ptr,
                            RunContext *run_ctx) {
  // Set current run context
  SetCurrentRunContext(run_ctx);

  // New task execution - increment work count for non-periodic tasks
  if (run_ctx->container_ && !task_ptr->IsPeriodic()) {
    // Increment work remaining in the container for non-periodic tasks
    run_ctx->container_->UpdateWork(task_ptr, *run_ctx, 1);
  }

  // Get the container from RunContext
  Container *container = run_ctx->container_;
  if (!container) {
    HLOG(kWarning, "Container not found in RunContext for pool_id: {}",
         task_ptr->pool_id_);
    return;
  }

  // Call the container's Run function which returns a TaskResume coroutine
  try {
    TaskResume task_resume =
        container->Run(task_ptr->method_, task_ptr, *run_ctx);

    // Store the coroutine handle in RunContext for later resumption
    auto handle = task_resume.release();
    run_ctx->coro_handle_ = handle;

    // Set the run context in the coroutine's promise so it can access it
    if (handle) {
      auto typed_handle =
          TaskResume::handle_type::from_address(handle.address());
      typed_handle.promise().set_run_context(run_ctx);

      // Resume the coroutine to run until first suspension point or completion
      // initial_suspend returns suspend_always, so we need to resume to start
      // execution
      handle.resume();

      // Check if coroutine completed (no suspension points)
      if (handle.done()) {
        // Coroutine completed - clean up
        handle.destroy();
        run_ctx->coro_handle_ = nullptr;
      }
    }
  } catch (const std::exception &e) {
    HLOG(kError, "Task execution failed: {}", e.what());
    // Clean up coroutine handle on exception
    if (run_ctx->coro_handle_) {
      run_ctx->coro_handle_.destroy();
      run_ctx->coro_handle_ = nullptr;
    }
  } catch (...) {
    HLOG(kError, "Task execution failed with unknown exception");
    // Clean up coroutine handle on exception
    if (run_ctx->coro_handle_) {
      run_ctx->coro_handle_.destroy();
      run_ctx->coro_handle_ = nullptr;
    }
  }
}

void Worker::ResumeCoroutine(const FullPtr<Task> &task_ptr,
                             RunContext *run_ctx) {
  // Set current run context
  SetCurrentRunContext(run_ctx);

  // Clear yielded flag before resumption
  run_ctx->is_yielded_ = false;

  // Check if we have a valid coroutine handle
  if (!run_ctx->coro_handle_) {
    HLOG(kWarning,
         "Worker {}: Attempted to resume task without coroutine handle. "
         "Task method: {} Pool: {}",
         worker_id_, task_ptr->method_, task_ptr->pool_id_);
    return;
  }

  // Resume the coroutine - it will run until next co_await or co_return
  try {
    HLOG(kDebug,
         "ResumeCoroutine: About to resume coro_handle_={} for task method={}",
         (void *)run_ctx->coro_handle_.address(), task_ptr->method_);
    run_ctx->coro_handle_.resume();
    HLOG(kDebug, "ResumeCoroutine: Returned from resume, coro_handle_={}",
         (void *)(run_ctx->coro_handle_ ? run_ctx->coro_handle_.address()
                                        : nullptr));

    // Check if coroutine completed after resumption
    if (run_ctx->coro_handle_.done()) {
      // Coroutine completed - clean up
      run_ctx->coro_handle_.destroy();
      run_ctx->coro_handle_ = nullptr;
    }
  } catch (const std::exception &e) {
    HLOG(kError, "Task resume failed: {}", e.what());
    // Clean up coroutine handle on exception
    if (run_ctx->coro_handle_) {
      run_ctx->coro_handle_.destroy();
      run_ctx->coro_handle_ = nullptr;
    }
  } catch (...) {
    HLOG(kError, "Task resume failed with unknown exception");
    // Clean up coroutine handle on exception
    if (run_ctx->coro_handle_) {
      run_ctx->coro_handle_.destroy();
      run_ctx->coro_handle_ = nullptr;
    }
  }
}

void Worker::ExecTask(const FullPtr<Task> &task_ptr, RunContext *run_ctx,
                      bool is_started) {
  // Non-periodic tasks always count as real work.
  // Periodic tasks must express work via run_ctx->did_work_.
  if (!task_ptr->IsPeriodic()) {
    SetTaskDidWork(true);
  }

  // Check if task is null or run context is null
  if (task_ptr.IsNull() || !run_ctx) {
    return;  // Consider null tasks as completed
  }

  // Call appropriate coroutine function based on task state
  if (is_started) {
    ResumeCoroutine(task_ptr, run_ctx);
  } else {
    StartCoroutine(task_ptr, run_ctx);
    task_ptr->SetFlags(TASK_STARTED);
  }

  // For periodic tasks, only set task_did_work_ if the task reported
  // actual work done (e.g., received data, sent data). This prevents
  // idle polling from keeping the worker awake.
  if (task_ptr->IsPeriodic() && run_ctx->did_work_) {
    SetTaskDidWork(true);
  }

  // Only set did_work_ if the task actually did work
  if (GetTaskDidWork() && run_ctx->exec_mode_ != ExecMode::kDynamicSchedule) {
    did_work_ = true;
  }

  // Check if coroutine is done or yielded
  bool coro_done = run_ctx->coro_handle_ && run_ctx->coro_handle_.done();

  // If coroutine yielded (not done and is_yielded_ set), don't clean up
  if (run_ctx->is_yielded_ && !coro_done) {
    // Task is blocked - don't clean up, will be resumed later
    return;  // Task is not completed, blocked for later resume
  }

  // Check if this is a dynamic scheduling task
  if (run_ctx->exec_mode_ == ExecMode::kDynamicSchedule) {
    // Dynamic scheduling - re-route task with updated pool_query
    RerouteDynamicTask(task_ptr, run_ctx);
    return;
  }

  // End task execution and cleanup (handles periodic rescheduling internally)
  EndTask(task_ptr, run_ctx, true);
}

void Worker::EndTaskShmTransfer(const FullPtr<Task> &task_ptr,
                                RunContext *run_ctx,
                                Container *container) {
  auto future_shm = run_ctx->future_.GetFutureShm();

  // Build SHM context for transfer (output reuses same copy_space)
  future_shm->output_.copy_space_size_ = future_shm->input_.copy_space_size_;
  hshm::lbm::LbmContext ctx;
  ctx.copy_space = future_shm->copy_space;
  ctx.shm_info_ = &future_shm->output_;

  // Serialize outputs
  SaveTaskArchive archive(MsgType::kSerializeOut, shm_send_transport_.get());
  container->SaveTask(task_ptr->method_, archive, task_ptr);

  // Send via SHM transport (blocking)
  shm_send_transport_->Send(archive, ctx);

  // Set FUTURE_COMPLETE and clean up task
  future_shm->flags_.SetBits(FutureShm::FUTURE_COMPLETE);
  container->DelTask(task_ptr->method_, task_ptr);
}

void Worker::EndTask(const FullPtr<Task> &task_ptr, RunContext *run_ctx,
                     bool can_resched) {
  // Check container once at the beginning
  Container *container = run_ctx->container_;
  if (container == nullptr) {
    HLOG(kError, "EndTask: container is null");
    return;
  }

  // Get task properties at the start
  bool is_remote = task_ptr->IsRemote();
  bool is_periodic = task_ptr->IsPeriodic();

  // Handle periodic task rescheduling
  if (is_periodic && can_resched) {
    ReschedulePeriodicTask(run_ctx, task_ptr);
    return;
  }

  // Decrement work remaining for non-periodic tasks
  if (!is_periodic) {
    container->UpdateWork(task_ptr, *run_ctx, -1);
  }

  // If task is remote, enqueue to net_queue_ for SendOut
  if (is_remote) {
    CHI_IPC->EnqueueNetTask(run_ctx->future_, NetQueuePriority::kSendOut);
    return;
  }

  // Copy variables from future_shm to stack BEFORE any SetComplete() call
  // This prevents use-after-free since client may free future_shm after
  // SetComplete()
  auto future_shm = run_ctx->future_.GetFutureShm();
  bool was_copied = future_shm->flags_.Any(FutureShm::FUTURE_WAS_COPIED);

  // Copy parent task pointer before transfer begins (may be modified during
  // transfer)
  RunContext *parent_task = run_ctx->future_.GetParentTask();

  // Handle client transfer based on origin transport mode
  if (was_copied) {
    u32 origin = future_shm->origin_;
    switch (origin) {
      case FutureShm::FUTURE_CLIENT_SHM:
        EndTaskShmTransfer(task_ptr, run_ctx, container);
        break;
      case FutureShm::FUTURE_CLIENT_TCP:
        CHI_IPC->EnqueueNetTask(run_ctx->future_, NetQueuePriority::kClientSendTcp);
        break;
      case FutureShm::FUTURE_CLIENT_IPC:
        CHI_IPC->EnqueueNetTask(run_ctx->future_, NetQueuePriority::kClientSendIpc);
        break;
      default:
        EndTaskShmTransfer(task_ptr, run_ctx, container);
        break;
    }
  } else if (parent_task && parent_task->event_queue_) {
    // Runtime subtask with parent: enqueue Future to parent worker's event queue.
    // FUTURE_COMPLETE is NOT set here — it will be set by ProcessEventQueue on the
    // parent's worker thread. This prevents the race where the parent sees
    // FUTURE_COMPLETE early, completes, frees memory, and a stale event resumes
    // a different task that reused the same address.
    auto *parent_event_queue = reinterpret_cast<
        hipc::mpsc_ring_buffer<Future<Task, CHI_MAIN_ALLOC_T>,
                               hshm::ipc::MallocAllocator> *>(
        parent_task->event_queue_);
    bool was_empty = parent_event_queue->Empty();
    parent_event_queue->Emplace(run_ctx->future_);
    if (was_empty && parent_task->lane_) {
      CHI_IPC->AwakenWorker(parent_task->lane_);
    }
  } else {
    // Runtime task without parent (top-level client task) - set FUTURE_COMPLETE
    // directly so the client's Wait() can see it
    future_shm->flags_.SetBits(FutureShm::FUTURE_COMPLETE);
  }
}

void Worker::RerouteDynamicTask(const FullPtr<Task> &task_ptr,
                                RunContext *run_ctx) {
  // Dynamic scheduling complete - now re-route task with updated pool_query
  // The task's pool_query_ should have been updated during execution
  // (e.g., from Dynamic to Local or Broadcast)

  Container *container = run_ctx->container_;
  TaskLane *lane = run_ctx->lane_;

  // Reset flags so the task can be re-routed and executed again
  task_ptr->ClearFlags(TASK_STARTED | TASK_ROUTED);

  // Re-route the task using the updated pool_query
  if (RouteTask(run_ctx->future_, lane, container)) {
    // Avoids recursive call to RerouteDynamicTask
    if (run_ctx->exec_mode_ == ExecMode::kDynamicSchedule) {
      EndTask(task_ptr, run_ctx, false);
      return;
    }
    // Successfully re-routed - execute the task again
    // Note: ExecTask will call BeginFiber since TASK_STARTED is unset
    ExecTask(task_ptr, run_ctx, false);
  }
  // RouteTask returned false means task was routed globally
}

void Worker::ProcessBlockedQueue(std::queue<RunContext *> &queue,
                                 u32 queue_idx) {
  (void)queue_idx;  // Unused parameter, kept for API consistency

  // Process only first 8 tasks in the queue
  size_t queue_size = queue.size();
  size_t check_limit = std::min(queue_size, size_t(8));

  for (size_t i = 0; i < check_limit; i++) {
    if (queue.empty()) {
      break;
    }

    RunContext *run_ctx = queue.front();
    queue.pop();

    if (!run_ctx || run_ctx->task_.IsNull()) {
      // Invalid entry, don't re-add
      continue;
    }

    // Determine if this is a resume (task was started before) or first
    // execution
    bool is_started = run_ctx->task_->task_flags_.Any(TASK_STARTED);

    // Skip if task was started but coroutine already completed
    // This can happen with orphan events from parallel subtasks
    if (is_started &&
        (!run_ctx->coro_handle_ || run_ctx->coro_handle_.done())) {
      continue;
    }

    run_ctx->yield_count_ = 0;

    // CRITICAL: Clear the is_yielded_ flag before resuming the task
    // This allows the task to call Wait() again if needed
    run_ctx->is_yielded_ = false;

    // Execute task with existing RunContext
    ExecTask(run_ctx->task_, run_ctx, is_started);

    // Don't re-add to queue
    continue;

    // Re-add to appropriate blocked queue based on current block count
    // AddToBlockedQueue will increment yield_count_ and determine the queue
    AddToBlockedQueue(run_ctx);
  }
}

void Worker::ProcessPeriodicQueue(std::queue<RunContext *> &queue,
                                  u32 queue_idx) {
  (void)queue_idx;  // Unused parameter, kept for API consistency

  // Check up to 8 tasks from the queue
  size_t check_limit = 8;
  size_t queue_size = queue.size();
  size_t actual_limit = std::min(queue_size, check_limit);

  // Capture SINGLE timestamp for ALL tasks processed in this batch
  // This prevents timestamp desynchronization between tasks with same
  // yield_time
  hshm::Timepoint batch_timestamp;
  batch_timestamp.Now();

  // Get current time for all checks
  for (size_t i = 0; i < actual_limit; i++) {
    if (queue.empty()) {
      break;
    }

    RunContext *run_ctx = queue.front();
    queue.pop();

    if (!run_ctx || run_ctx->task_.IsNull()) {
      // Invalid entry, don't re-add
      continue;
    }

    // Check if the time threshold has been surpassed using batch timestamp
    // Add 2ms tolerance to account for timing variance and ms/us precision
    // mismatch
    double elapsed_us = run_ctx->block_start_.GetUsecFromStart(batch_timestamp);
    if (elapsed_us + 2000.0 >= run_ctx->yield_time_us_) {
      // Time threshold reached (within tolerance) - execute the task
      bool is_started = run_ctx->task_->task_flags_.Any(TASK_STARTED);

      // CRITICAL: Clear the is_yielded_ flag before resuming the task
      // This allows the task to call Wait() again if needed
      run_ctx->is_yielded_ = false;

      // For periodic tasks, unmark TASK_ROUTED and route again
      run_ctx->task_->ClearFlags(TASK_ROUTED);
      Container *container = run_ctx->container_;

      // Use batch timestamp for rescheduling to prevent desynchronization
      // This ensures all tasks in this batch get the same block_start time
      run_ctx->block_start_ = batch_timestamp;

      // Route task again - this will handle both local and distributed routing
      if (RouteTask(run_ctx->future_, run_ctx->lane_, container)) {
        // Routing successful, execute the task
        ExecTask(run_ctx->task_, run_ctx, is_started);
      }
    } else {
      // Time threshold not reached yet - re-add to same queue
      queue.push(run_ctx);
    }
  }
}

void Worker::ProcessEventQueue() {
  // Process all subtask futures in the event queue.
  // Each entry is a Future<Task> from a completed subtask. We set
  // FUTURE_COMPLETE on it here (on the parent worker's thread), then resume
  // the parent coroutine. This avoids stale RunContext* pointers since
  // FUTURE_COMPLETE is never set before the event is consumed.
  Future<Task, CHI_MAIN_ALLOC_T> future;
  while (event_queue_->Pop(future)) {
    // Mark the subtask's future as complete
    future.Complete();

    // Get the parent RunContext that is waiting for this subtask.
    // Safe to dereference because FUTURE_COMPLETE was not set until just now,
    // so the parent coroutine could not have seen completion, could not have
    // finished, and its RunContext has not been freed.
    RunContext *run_ctx = future.GetParentTask();
    if (!run_ctx || run_ctx->task_.IsNull()) {
      continue;
    }

    // Skip if coroutine handle is null or already completed
    if (!run_ctx->coro_handle_ || run_ctx->coro_handle_.done()) {
      continue;
    }

    HLOG(kDebug, "ProcessEventQueue: Resuming task method={}, coro_handle_={}",
         run_ctx->task_->method_, (void *)run_ctx->coro_handle_.address());

    // Reset the is_yielded_ flag before executing the task
    run_ctx->is_yielded_ = false;

    // Reset is_notified_ so this task can be notified again for subsequent
    // co_await
    run_ctx->is_notified_.store(false);

    // Execute the task
    ExecTask(run_ctx->task_, run_ctx, true);
  }
}

void Worker::ContinueBlockedTasks(bool force) {
  // Process event queue to wake up tasks waiting for subtask completion
  ProcessEventQueue();

  if (force) {
    // Force mode: process all blocked queues regardless of iteration count
    for (u32 i = 0; i < NUM_BLOCKED_QUEUES; ++i) {
      ProcessBlockedQueue(blocked_queues_[i], i);
    }
    // Also process all periodic queues in force mode
    for (u32 i = 0; i < NUM_PERIODIC_QUEUES; ++i) {
      ProcessPeriodicQueue(periodic_queues_[i], i);
    }
  } else {
    // Normal mode: check blocked queues based on iteration count
    // blocked_queues_[0] every 2 iterations
    if (iteration_count_ % 2 == 0) {
      ProcessBlockedQueue(blocked_queues_[0], 0);
    }

    // blocked_queues_[1] every 4 iterations
    if (iteration_count_ % 4 == 0) {
      ProcessBlockedQueue(blocked_queues_[1], 1);
    }

    // blocked_queues_[2] every 8 iterations
    if (iteration_count_ % 8 == 0) {
      ProcessBlockedQueue(blocked_queues_[2], 2);
    }

    // blocked_queues_[3] every 16 iterations
    if (iteration_count_ % 16 == 0) {
      ProcessBlockedQueue(blocked_queues_[3], 3);
    }

    // Process periodic queues with different checking frequencies
    // periodic_queues_[0] (<=50us) every 4 iterations
    if (iteration_count_ % 4 == 0) {
      ProcessPeriodicQueue(periodic_queues_[0], 0);
    }

    // periodic_queues_[1] (<=200us) every 8 iterations
    if (iteration_count_ % 8 == 0) {
      ProcessPeriodicQueue(periodic_queues_[1], 1);
    }

    // periodic_queues_[2] (<=50ms) every 64 iterations
    if (iteration_count_ % 64 == 0) {
      ProcessPeriodicQueue(periodic_queues_[2], 2);
    }

    // periodic_queues_[3] (>50ms) every 128 iterations
    if (iteration_count_ % 128 == 0) {
      ProcessPeriodicQueue(periodic_queues_[3], 3);
    }
  }
}

void Worker::AddToBlockedQueue(RunContext *run_ctx, bool wait_for_task) {
  if (!run_ctx || run_ctx->task_.IsNull()) {
    return;
  }

  // If wait_for_task is true, do not add to blocked queue
  // The task is waiting for subtask completion and will be woken by event queue
  if (wait_for_task) {
    return;
  }

  // Check if task should go to blocked queue or periodic queue
  // Go to blocked queue if: block_time is 0 OR task is already started
  if (run_ctx->yield_time_us_ == 0.0) {
    // Cooperative task waiting for subtasks - add to blocked queue
    // Increment block count for cooperative tasks
    run_ctx->yield_count_++;

    // Determine which blocked queue based on block count:
    // Queue[0]: Tasks blocked <=2 times (checked every % 2 iterations)
    // Queue[1]: Tasks blocked <= 4 times (checked every % 4 iterations)
    // Queue[2]: Tasks blocked <= 8 times (checked every % 8 iterations)
    // Queue[3]: Tasks blocked > 8 times (checked every % 16 iterations)
    u32 queue_idx;
    if (run_ctx->yield_count_ <= 2) {
      queue_idx = 0;
    } else if (run_ctx->yield_count_ <= 4) {
      queue_idx = 1;
    } else if (run_ctx->yield_count_ <= 8) {
      queue_idx = 2;
    } else {
      queue_idx = 3;
    }

    // Add to the appropriate blocked queue
    blocked_queues_[queue_idx].push(run_ctx);
  } else {
    // Time-based periodic task - add to periodic queue
    // Record the time when task was blocked (if not already set recently)
    // Check if timestamp was set within last 10ms (indicates batch processing)
    double elapsed_since_block_us = run_ctx->block_start_.GetUsecFromStart();
    if (elapsed_since_block_us > 10000.0 || elapsed_since_block_us < 0) {
      // Timestamp is stale or uninitialized - set it now
      run_ctx->block_start_.Now();
    }
    // else: timestamp is fresh (< 10ms old), keep it to maintain
    // synchronization

    // Determine which periodic queue based on yield_time_us_:
    // Queue[0]: yield_time_us_ <= 50us
    // Queue[1]: yield_time_us_ <= 200us
    // Queue[2]: yield_time_us_ <= 50ms (50000us)
    // Queue[3]: yield_time_us_ > 50ms
    u32 queue_idx;
    if (run_ctx->yield_time_us_ <= 50.0) {
      queue_idx = 0;
    } else if (run_ctx->yield_time_us_ <= 200.0) {
      queue_idx = 1;
    } else if (run_ctx->yield_time_us_ <= 50000.0) {
      queue_idx = 2;
    } else {
      queue_idx = 3;
    }

    // Add to the appropriate periodic queue
    periodic_queues_[queue_idx].push(run_ctx);
  }
}

void Worker::ReschedulePeriodicTask(RunContext *run_ctx,
                                    const FullPtr<Task> &task_ptr) {
  if (!run_ctx || task_ptr.IsNull() || !task_ptr->IsPeriodic()) {
    return;
  }

  // Get the lane from the run context
  TaskLane *lane = run_ctx->lane_;
  if (!lane) {
    // No lane information, cannot reschedule
    return;
  }

  // Unset TASK_STARTED when rescheduling periodic task
  task_ptr->ClearFlags(TASK_STARTED);

  // Adjust polling rate based on whether task did work
  if (scheduler_) {
    scheduler_->AdjustPolling(run_ctx);
  } else {
    // Fallback: use the true period if no scheduler available
    run_ctx->yield_time_us_ = task_ptr->period_ns_ / 1000.0;
  }

  // Reset did_work_ for the next execution
  run_ctx->did_work_ = false;

  // Add to blocked queue - block count will be incremented automatically
  AddToBlockedQueue(run_ctx);
}

RunContext *GetCurrentRunContextFromWorker() {
  Worker *worker = CHI_CUR_WORKER;
  if (worker) {
    return worker->GetCurrentRunContext();
  }
  return nullptr;
}

void Worker::CopyTaskOutputToClient() {
  // Process transfers in client_copy_ queue
  // Mark did_work_ if there are any transfers to prevent worker suspension
  if (!client_copy_.empty()) {
    did_work_ = true;
  }

  while (!client_copy_.empty()) {
    LocalTransfer &transfer = client_copy_.front();

    // Try to send data using LocalTransfer (5ms = 5000us time budget)
    bool send_complete = transfer.Send(5000);

    if (send_complete) {
      // Transfer complete - remove from queue
      client_copy_.pop();
    } else {
      // Transfer not complete - move to back of queue for fairness
      LocalTransfer t = std::move(client_copy_.front());
      client_copy_.pop();
      client_copy_.push(std::move(t));
      break;  // Process other work before continuing this transfer
    }
  }
}

}  // namespace chi