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

#include "clio_runtime/worker.h"

#ifndef __NVCOMPILER
#include <coroutine>
#endif
#include <cstdlib>
#include <iostream>
#include <unordered_set>

// Include task_queue.h before other chimaera headers to ensure proper
// resolution
#include "clio_runtime/admin/admin_client.h"
#include "clio_runtime/container.h"
#include "clio_runtime/device_memcpy.h"
#include "clio_runtime/ipc_manager.h"
#include "clio_runtime/pool_manager.h"
#include "clio_runtime/singletons.h"
#include "clio_runtime/task.h"
#include "clio_runtime/task_archives.h"
#include "clio_runtime/local_task_archives.h"
#include "clio_runtime/work_orchestrator.h"

namespace clio::run {

// Detect whether a task's coroutine has run to completion WITHOUT
// dereferencing the coroutine frame. The top-level coroutine's final_suspend
// sets run_ctx->coro_completed_ (issue #485); reading that flag is valid even
// if a cross-thread completion already freed the frame, whereas
// coro_handle_.done() would be a use-after-free (the GPFLT in
// coroutine_handle::done() observed on macOS). The NVHPC fiber path has no
// such flag and is not subject to the same cross-thread free, so it keeps
// using FiberHandle::done().
namespace {
inline bool CoroCompleted(const RunContext *run_ctx) {
#ifndef __NVCOMPILER
  return run_ctx->coro_completed_.load(std::memory_order_acquire);
#else
  return run_ctx->coro_handle_ && run_ctx->coro_handle_.done();
#endif
}
}  // namespace

// Stack detection is now handled by WorkOrchestrator during initialization

Worker::Worker(u32 worker_id)
    : worker_id_(worker_id),
      is_running_(false),
      is_initialized_(false),
      load_(0),
      did_work_(false),
      task_did_work_(false),
      current_run_context_(nullptr),
      assigned_lane_(nullptr),
      event_queue_(nullptr),
      num_tasks_processed_(0),
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
  // runtime data). Stores Future<Task> objects to avoid stale RunContext*
  // pointers.
  event_queue_ =
      CTP_MALLOC
          ->template NewObj<ctp::ipc::mpsc_ring_buffer<
              Future<Task, CLIO_QUEUE_ALLOC_T>, ctp::ipc::MallocAllocator>>(
              CTP_MALLOC, EVENT_QUEUE_DEPTH)
          .ptr_;

  // Get scheduler from IpcManager (IpcManager is the single owner)
  scheduler_ = CLIO_IPC->GetScheduler();
  HLOG(kDebug, "Worker {}: Using scheduler from IpcManager", worker_id_);

  // Create SHM lightbeam transports for worker-side transport
  shm_send_transport_ = ctp::lbm::TransportFactory::Get(
      "", ctp::lbm::TransportType::kShm, ctp::lbm::TransportMode::kClient);
  shm_recv_transport_ = ctp::lbm::TransportFactory::Get(
      "", ctp::lbm::TransportType::kShm, ctp::lbm::TransportMode::kServer);

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

  // Count retry tasks
  stats.num_retry_tasks_ = retry_queue_.size();

  // Get suspend period (time until next periodic task or 0 if none)
  double suspend_period = GetSuspendPeriod();
  stats.suspend_period_us_ =
      (suspend_period < 0) ? 0 : static_cast<u32>(suspend_period);

  stats.num_tasks_processed_ = num_tasks_processed_;
  stats.load_ = load_;

  return stats;
}

ctp::lbm::EventManager &Worker::GetEventManager() { return event_manager_; }

void Worker::Finalize() {
  if (!is_initialized_) {
    return;
  }

  Stop();

  // Note: Context cache cleanup removed - RunContext is now embedded in Task

  // Clean up all blocked queues
  for (u32 i = 0; i < NUM_BLOCKED_QUEUES; ++i) {
    while (!blocked_queues_[i].empty()) {
      RunContext *run_ctx = blocked_queues_[i].front();
      blocked_queues_[i].pop();
      // RunContexts in blocked queues are still in use - don't free them
      // They will be cleaned up when the tasks complete or by stack cache
      (void)run_ctx;  // Suppress unused variable warning
    }
  }

  // Clean up all periodic queues
  for (u32 i = 0; i < NUM_PERIODIC_QUEUES; ++i) {
    while (!periodic_queues_[i].empty()) {
      periodic_queues_[i].pop();
    }
  }

  // Clean up retry queue
  while (!retry_queue_.empty()) {
    retry_queue_.pop();
  }

  // Clear assigned lane reference (don't delete - it's in shared memory)
  assigned_lane_ = nullptr;

  is_initialized_ = false;
}

void Worker::Run() {
  if (!is_initialized_) {
    return;
  }
  // Set current worker once for the entire thread duration
  SetAsCurrentWorker();
  is_running_ = true;

  // Set up thread ID and signal event via EventManager
  int tid = ctp::SystemInfo::GetTid();
  if (assigned_lane_) {
    assigned_lane_->SetTid(tid);
    assigned_lane_->SetRuntimePid(ctp::SystemInfo::GetPid());
  }
  event_manager_.AddSignalEvent(nullptr);

  // Main worker loop - process tasks from assigned lane
  while (is_running_) {
    did_work_ = false;  // Reset work tracker at start of each loop iteration
    task_did_work_ = false;  // Reset task-level work tracker

    // Process tasks from assigned lane
    if (assigned_lane_) {
      u32 count = ProcessNewTasks(assigned_lane_);
      if (count > 0) did_work_ = true;
    }
    u32 gpu_count = ProcessNewTasksGpu();
    if (gpu_count > 0) did_work_ = true;

    // Check blocked queue for completed tasks at end of each iteration
    ContinueBlockedTasks(false);

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

void Worker::SetGpuLanes(const std::vector<GpuTaskLane *> &lanes) {
  gpu_lanes_ = lanes;
}

const std::vector<GpuTaskLane *> &Worker::GetGpuLanes() const {
  return gpu_lanes_;
}

u32 Worker::ProcessNewTasksGpu() {
  u32 total = 0;
  for (auto *gpu_lane : gpu_lanes_) {
    if (ProcessNewTaskGpu(gpu_lane)) {
      ++total;
    }
  }
  return total;
}

bool Worker::ProcessNewTaskGpu(GpuTaskLane *gpu_lane) {
  // Producer-only gpu2cpu pop path.
  //
  // The kernel pre-allocated a Task+FutureShm pair in a registered
  // GPU client backend (kPinnedHost, kManagedUvm, or kDeviceMem) and
  // pushed a gpu::Future<Task> carrying ShmPtrs (with the raw device-
  // accessible address stashed in `off_`) for both the task and its
  // co-located gpu::FutureShm.
  //
  // For kPinnedHost / kManagedUvm the worker dereferences both raw
  // addresses directly: the CPU and GPU share visibility. For
  // kDeviceMem the worker D2H-copies the POD bytes (gpu::FutureShm and
  // the Task struct) into per-thread host scratch and runs the chimod
  // on those copies; RuntimeSend H2D-copies the mutated Task POD back
  // to the original device address before signaling FUTURE_COMPLETE.
  // The chi::FutureShm carries the original device pointers
  // (gpu_task_device_ptr_ / gpu_fshm_device_ptr_) plus task size so
  // RuntimeSend can issue the writeback memcpys.
  gpu::Future<Task> gpu_future;
  if (!gpu_lane->Pop(gpu_future)) {
    return false;
  }
  HLOG(kDebug, "Worker {}: ProcessNewTaskGpu: popped task from gpu2cpu queue",
       worker_id_);

  SetCurrentRunContext(nullptr);

  ctp::ipc::ShmPtr<gpu::FutureShm> gpu_fshm_shmptr = gpu_future.GetFutureShmPtr();
  ctp::ipc::ShmPtr<Task> task_shmptr = gpu_future.GetTaskPtr().shm_;
  if (gpu_fshm_shmptr.IsNull() || task_shmptr.IsNull()) {
    HLOG(kError, "Worker {}: ProcessNewTaskGpu: null ShmPtr in queue entry",
         worker_id_);
    return true;
  }

  void *gpu_fshm_raw = reinterpret_cast<void *>(
      gpu_fshm_shmptr.off_.load());
  void *gpu_task_raw = reinterpret_cast<void *>(task_shmptr.off_.load());
  if (!gpu_fshm_raw || !gpu_task_raw) {
    HLOG(kError, "Worker {}: ProcessNewTaskGpu: null off_ in queue entry",
         worker_id_);
    return true;
  }

  // Detect whether the FutureShm / Task structs sit in pure device
  // memory (host cannot dereference them). g_is_device_pointer is
  // installed by ServerInitGpuQueues; absent on host-only builds.
  auto is_device_ptr = chi::g_is_device_pointer.load(
      std::memory_order_acquire);
  bool fshm_on_device =
      is_device_ptr && is_device_ptr(gpu_fshm_raw);
  bool task_on_device =
      is_device_ptr && is_device_ptr(gpu_task_raw);

  // Pull gpu::FutureShm contents into a local copy (D2H if needed).
  // task_size_ tells us how many bytes the Task POD occupies.
  alignas(8) char fshm_buf[sizeof(gpu::FutureShm)];
  if (fshm_on_device) {
    chi::DeviceAwareMemcpy(fshm_buf, gpu_fshm_raw,
                           sizeof(gpu::FutureShm));
  } else {
    std::memcpy(fshm_buf, gpu_fshm_raw, sizeof(gpu::FutureShm));
  }
  auto &fshm_copy = *reinterpret_cast<gpu::FutureShm *>(fshm_buf);
  u32 task_pod_size = fshm_copy.task_size_;
  if (task_pod_size == 0) {
    HLOG(kError,
         "Worker {}: ProcessNewTaskGpu: gpu::FutureShm.task_size_=0 — "
         "kernel did not call Reset(sizeof(TaskT)) before Send",
         worker_id_);
    return true;
  }

  // Per-thread scratch for the host-resident task copy. Sized to fit
  // any reasonable POD task (PutBlobTask is ~480 bytes today).
  static constexpr size_t kTaskScratchBytes = 4096;
  alignas(64) thread_local char task_scratch[kTaskScratchBytes];
  if (task_pod_size > kTaskScratchBytes) {
    HLOG(kError,
         "Worker {}: ProcessNewTaskGpu: task_pod_size {} exceeds "
         "scratch capacity {}",
         worker_id_, task_pod_size, kTaskScratchBytes);
    return true;
  }
  Task *task_raw = nullptr;
  if (task_on_device) {
    chi::DeviceAwareMemcpy(task_scratch, gpu_task_raw, task_pod_size);
    task_raw = reinterpret_cast<Task *>(task_scratch);
  } else {
    task_raw = static_cast<Task *>(gpu_task_raw);
  }

  PoolId pool_id = task_raw->pool_id_;
  u32 method_id = task_raw->method_;

  ctp::ipc::FullPtr<Task> task_full_ptr(task_raw);

  Future<Task> future = CLIO_IPC->MakePointerFuture(task_full_ptr);
  if (future.GetFutureShmPtr().IsNull()) {
    HLOG(kError,
         "Worker {}: ProcessNewTaskGpu: MakePointerFuture failed "
         "(pool={}, method={})",
         worker_id_, pool_id, method_id);
    if (!fshm_on_device) {
      static_cast<gpu::FutureShm *>(gpu_fshm_raw)
          ->flags_.SetBitsSystem(gpu::FutureShm::FUTURE_COMPLETE);
    }
    return true;
  }

  auto chi_fshm = future.GetFutureShm();
  chi_fshm->pool_id_ = pool_id;
  chi_fshm->method_id_ = method_id;
  chi_fshm->origin_ = FutureShm::FUTURE_CLIENT_GPU2CPU;
  // Stash original device-side pointers + size so RuntimeSend can
  // H2D-copy the mutated POD back and signal FUTURE_COMPLETE on the
  // device-side gpu::FutureShm (cudaMemcpy when in kDeviceMem).
  chi_fshm->gpu_fshm_device_ptr_ =
      reinterpret_cast<uintptr_t>(gpu_fshm_raw);
  chi_fshm->gpu_task_device_ptr_ =
      task_on_device ? reinterpret_cast<uintptr_t>(gpu_task_raw) : 0;
  chi_fshm->gpu_task_size_ = task_pod_size;

  auto *pool_manager = CLIO_POOL_MANAGER;
  Container *container = pool_manager->GetStaticContainer(pool_id);
  if (!container) {
    HLOG(kError,
         "Worker {}: ProcessNewTaskGpu: Container not found "
         "(pool={}, method={})",
         worker_id_, pool_id, method_id);
    chi_fshm->flags_.SetBits(1 | FutureShm::FUTURE_COMPLETE);
    if (!fshm_on_device) {
      static_cast<gpu::FutureShm *>(gpu_fshm_raw)
          ->flags_.SetBitsSystem(gpu::FutureShm::FUTURE_COMPLETE);
    } else {
      // Best-effort: still flip the device flag via cudaMemcpy.
      u32 v = gpu::FutureShm::FUTURE_COMPLETE;
      chi::DeviceAwareMemcpy(
          &static_cast<gpu::FutureShm *>(gpu_fshm_raw)->flags_.bits_.x,
          &v, sizeof(u32));
    }
    return true;
  }

  // Fix up SSO/SVO `data_` pointers in the host-resident task copy if
  // we D2H-copied it. The chimod's container override dispatches by
  // method id to the per-task FixupAfterCopy(). Skip when the task
  // never moved (kPinnedHost / kManagedUvm path).
  if (task_on_device) {
    container->FixupAfterCopy(method_id, task_full_ptr);
  }

  if (!task_full_ptr->task_flags_.Any(TASK_RUN_CTX_EXISTS)) {
    CLIO_IPC->BeginTask(future, container, assigned_lane_);
  } else {
    RunContext *run_ctx = task_full_ptr->GetRunCtx();
    if (run_ctx) {
      run_ctx->worker_id_ = worker_id_;
      run_ctx->lane_ = assigned_lane_;
      run_ctx->event_queue_ = event_queue_;
    }
  }

  RouteResult route_result = CLIO_IPC->RouteTask(future, /*force_enqueue=*/true);
  HLOG(kDebug, "Worker {}: ProcessNewTaskGpu: RouteTask returned {} "
       "pool={} method={}",
       worker_id_, (int)route_result, pool_id, method_id);
  return true;
}

ctp::ipc::FullPtr<Task> Worker::GetOrCopyTaskFromFuture(Future<Task> &future,
                                                    Container *container,
                                                    u32 method_id) {
  return CLIO_IPC->RecvRuntime(future, container, method_id,
                              shm_recv_transport_.get());
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
  HLOG(kDebug, "Worker {}: ProcessNewTask popped pool={} method={}",
       worker_id_, pool_id, method_id);

  // Get static container for task deserialization (stateless operation)
  auto *pool_manager = CLIO_POOL_MANAGER;
  Container *container = pool_manager->GetStaticContainer(pool_id);

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

  // Allocate RunContext before routing (skip if already created)
  if (!task_full_ptr->task_flags_.Any(TASK_RUN_CTX_EXISTS)) {
    CLIO_IPC->BeginTask(future, container, lane);
  } else {
    // Task was re-enqueued from another worker (e.g., by RouteLocal).
    // Update worker-specific RunContext fields to match this worker,
    // so subtask completion events go to the correct event queue.
    RunContext *run_ctx = task_full_ptr->GetRunCtx();
    if (run_ctx) {
      run_ctx->worker_id_ = worker_id_;
      run_ctx->lane_ = lane;
      run_ctx->event_queue_ = event_queue_;
    }
  }

  // Route task using consolidated routing function
  // RouteTask handles Retry/Dne internally via AddToRetryQueue
  RouteResult route_result = CLIO_IPC->RouteTask(future);
  if (route_result == RouteResult::ExecHere) {
#if CTP_IS_HOST
    // Re-fetch task pointer from future in case RouteTask changed it
    RunContext *run_ctx = task_full_ptr->GetRunCtx();
    bool is_started = task_full_ptr->task_flags_.Any(TASK_STARTED);
    ExecTask(task_full_ptr, run_ctx, is_started);
#endif
  }

  return true;
}

double Worker::GetSuspendPeriod() const {
  // Scan all periodic queues to find the minimum yield_time (polling period)
  // We must wake up for the fastest periodic task to avoid starving it
  double min_yield_time_us = 0;
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
    if (!found_task || run_ctx->yield_time_us_ < min_yield_time_us) {
      min_yield_time_us = run_ctx->yield_time_us_;
      found_task = true;
    }
  }

  // Return -1 if no periodic tasks (means wait indefinitely in epoll_wait)
  // Otherwise return the minimum yield_time across all periodic queues
  return found_task ? min_yield_time_us : -1;
}

void Worker::SuspendMe() {
  // GPU workers must never sleep — they need to poll GPU lanes continuously
  if (!gpu_lanes_.empty()) {
    return;
  }

  // No work was done in this iteration - increment idle counter
  idle_iterations_++;

  // Set idle start time on first idle iteration
  if (idle_iterations_ == 1) {
    idle_start_.Now();
  }

  // Get configuration parameters
  auto *config = CLIO_CONFIG_MANAGER;
  u32 first_busy_wait = config->GetFirstBusyWait();
  u32 max_sleep = config->GetMaxSleep();

  // Calculate actual elapsed idle time
  ctp::Timepoint current_time;
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

    // Last-chance recheck: any producer push that landed before we got
    // here is already visible to a plain Empty() load. signal_mpmc_queue
    // fires SIGUSR1 automatically on an empty→non-empty push, so any push
    // that lands AFTER this recheck will signal us out of epoll_pwait2.
    // The active_ park-flag handshake was removed because it caused a
    // lost-wakeup race at scale (4n 256m, multi-tier bdev).
    if (assigned_lane_) {
      bool work_pending = !assigned_lane_->Empty();
      if (!work_pending && event_queue_) {
        work_pending = !event_queue_->Empty();
      }
      if (work_pending) {
        return;
      }
    }

    // Calculate timeout from periodic tasks
    double suspend_period_us = GetSuspendPeriod();
    int timeout_us =
        (suspend_period_us < 0) ? -1 : static_cast<int>(suspend_period_us);

    // Wait for signal using EventManager
    int nfds = event_manager_.Wait(timeout_us);

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
  CTP_THREAD_MODEL->SetTls(chi_cur_worker_key_,
                            static_cast<class Worker *>(this));
}

void Worker::ClearCurrentWorker() {
  CTP_THREAD_MODEL->SetTls(chi_cur_worker_key_,
                            static_cast<class Worker *>(nullptr));
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

  // Call the container's Run function which returns a TaskResume coroutine/fiber
  try {
    TaskResume task_resume =
        container->Run(task_ptr->method_, task_ptr, *run_ctx);

#ifndef __NVCOMPILER
    // Standard C++20 coroutine path
    auto handle = task_resume.release();
    run_ctx->coro_handle_ = handle;

    // Set the run context in the coroutine's promise so it can access it
    if (handle) {
      auto typed_handle =
          TaskResume::handle_type::from_address(handle.address());
      typed_handle.promise().set_run_context(run_ctx);
      // Mark this as the top-level task coroutine so its (and only its)
      // final_suspend raises run_ctx->coro_completed_ on real task completion
      // (issue #485). Nested co_await'd coroutines are never marked.
      typed_handle.promise().is_top_level_ = true;

      // Fresh coroutine frame: clear the completion flag the promise's
      // final_suspend will raise when this task finishes (issue #485).
      run_ctx->coro_completed_.store(false, std::memory_order_relaxed);

      // Resume the coroutine to run until first suspension point or completion
      // initial_suspend returns suspend_always, so we need to resume to start
      // execution
      handle.resume();

      // Check if coroutine completed (no suspension points). Read the
      // RunContext completion flag rather than handle.done(): if the task
      // completed cross-thread during resume() the frame may already be freed,
      // and done() on a freed frame is a use-after-free (issue #485). When the
      // flag is set, await_resume has repointed things so `handle` (the
      // top-level frame) is still valid to destroy.
      if (run_ctx->coro_completed_.load(std::memory_order_acquire)) {
        // Coroutine completed - clean up
        handle.destroy();
        run_ctx->coro_handle_ = nullptr;
      }
    }
#else // __NVCOMPILER - ucontext_t fiber path
    auto fhandle = task_resume.release();
    run_ctx->coro_handle_ = fhandle;

    if (fhandle) {
      // Resume the fiber to run until first suspension point or completion
      run_ctx->coro_handle_.resume();

      // Check if fiber completed
      if (run_ctx->coro_handle_.done()) {
        run_ctx->coro_handle_.destroy();
        run_ctx->coro_handle_ = chi::detail::FiberHandle{};
      }
    }
#endif // __NVCOMPILER
  } catch (const std::exception &e) {
    HLOG(kError, "Task execution failed: {}", e.what());
    // Clean up handle on exception
    if (run_ctx->coro_handle_) {
      run_ctx->coro_handle_.destroy();
#ifndef __NVCOMPILER
      run_ctx->coro_handle_ = nullptr;
#else
      run_ctx->coro_handle_ = chi::detail::FiberHandle{};
#endif
    }
  } catch (...) {
    HLOG(kError, "Task execution failed with unknown exception");
    // Clean up handle on exception
    if (run_ctx->coro_handle_) {
      run_ctx->coro_handle_.destroy();
#ifndef __NVCOMPILER
      run_ctx->coro_handle_ = nullptr;
#else
      run_ctx->coro_handle_ = chi::detail::FiberHandle{};
#endif
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

  // Resume the coroutine/fiber - it will run until next suspension or completion
  try {
    run_ctx->coro_handle_.resume();

    // Check if coroutine/fiber completed after resumption. For C++20
    // coroutines this reads the RunContext completion flag (set by the
    // top-level coroutine's final_suspend) instead of coro_handle_.done(): the
    // task may have completed cross-thread during resume() and freed the
    // coroutine frame, which would make done() a use-after-free of a heap
    // coro_handle_ (the GPFLT seen on macOS, issue #485). The RunContext
    // outlives the frame; when the flag is set, await_resume has already
    // repointed coro_handle_ to the (valid) top-level frame so destroy() is
    // safe.
    if (CoroCompleted(run_ctx)) {
      // Completed - clean up
      run_ctx->coro_handle_.destroy();
#ifndef __NVCOMPILER
      run_ctx->coro_handle_ = nullptr;
#else
      run_ctx->coro_handle_ = chi::detail::FiberHandle{};
#endif
    }
  } catch (const std::exception &e) {
    HLOG(kError, "Task resume failed: {}", e.what());
    // Clean up handle on exception
    if (run_ctx->coro_handle_) {
      run_ctx->coro_handle_.destroy();
#ifndef __NVCOMPILER
      run_ctx->coro_handle_ = nullptr;
#else
      run_ctx->coro_handle_ = chi::detail::FiberHandle{};
#endif
    }
  } catch (...) {
    HLOG(kError, "Task resume failed with unknown exception");
    // Clean up handle on exception
    if (run_ctx->coro_handle_) {
      run_ctx->coro_handle_.destroy();
#ifndef __NVCOMPILER
      run_ctx->coro_handle_ = nullptr;
#else
      run_ctx->coro_handle_ = chi::detail::FiberHandle{};
#endif
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
    return;
  }

  // Resolve the container fresh each time (may change during migration)
  auto *pool_manager = CLIO_POOL_MANAGER;
  bool is_plugged = false;
  ContainerId container_id = task_ptr->pool_query_.GetContainerId();
  Container *exec_container =
      pool_manager->GetContainer(task_ptr->pool_id_, container_id, is_plugged);
  if (exec_container && !is_plugged) {
    run_ctx->container_ = exec_container;
  }

  // Start CPU and wall timers before execution
  run_ctx->cpu_timer_.Resume();
  run_ctx->wall_timer_.Resume();

  // Call appropriate coroutine function based on task state
  if (is_started) {
    ResumeCoroutine(task_ptr, run_ctx);
  } else {
    StartCoroutine(task_ptr, run_ctx);
    task_ptr->SetFlags(TASK_STARTED);

    // Predict load for new tasks. predicted_stat_ is populated by
    // BeginTask via container->GetTaskStats(task), so derive the model
    // inferences from the already-cached stat instead of re-calling
    // GetTaskStats here.
    if (run_ctx->container_) {
      run_ctx->predicted_load_ = run_ctx->container_->InferCpuTime(task_ptr->method_, run_ctx->predicted_stat_);
      run_ctx->predicted_wall_us_ = run_ctx->container_->InferWallClockTime(task_ptr->method_, run_ctx->predicted_stat_);
      load_ += run_ctx->predicted_load_;
    }
  }

  // Pause CPU and wall timers after execution
  run_ctx->cpu_timer_.Pause();
  run_ctx->wall_timer_.Pause();

  // For periodic tasks, only set task_did_work_ if the task reported
  // actual work done (e.g., received data, sent data). This prevents
  // idle polling from keeping the worker awake.
  if (task_ptr->IsPeriodic() && run_ctx->did_work_) {
    SetTaskDidWork(true);
  }

  // Only set did_work_ if the task actually did work
  if (GetTaskDidWork()) {
    did_work_ = true;
  }

  // Check if coroutine is done or yielded. Use the RunContext completion flag
  // (issue #485) rather than dereferencing the possibly-freed coroutine frame
  // via coro_handle_.done().
  bool coro_done = CoroCompleted(run_ctx);

  // If coroutine yielded (not done and is_yielded_ set), don't clean up
  if (run_ctx->is_yielded_ && !coro_done) {
    // yield_time_us_ > 0 means cooperative yield (polling) — add to periodic
    // queue so the worker re-checks after the requested delay.
    // yield_time_us_ == 0 means waiting for a Future event — the event queue
    // will resume it, so we must NOT add it to any queue here.
    if (run_ctx->yield_time_us_ > 0) {
      AddToBlockedQueue(run_ctx);
    }
    return;
  }

  // End task execution and cleanup (handles periodic rescheduling internally)
  EndTask(task_ptr, run_ctx, true);
}


void Worker::EndTask(const FullPtr<Task> &task_ptr, RunContext *run_ctx,
                     bool can_resched) {
  // Check container once at the beginning
  Container *container = run_ctx->container_;
  if (container == nullptr) {
    HLOG(kError, "EndTask: container is null");
    return;
  }

  // Track completed tasks
  ++num_tasks_processed_;

  // Subtract predicted load from worker
  load_ -= run_ctx->predicted_load_;

  // Reinforce model with actual CPU and wall time
  float actual_cpu_us = static_cast<float>(run_ctx->cpu_timer_.GetUsec());
  float actual_wall_us = static_cast<float>(run_ctx->wall_timer_.GetUsec());
  container->ReinforceCpuModel(
      task_ptr->method_, run_ctx->predicted_load_, actual_cpu_us,
      run_ctx->predicted_stat_);
  container->ReinforceWallModel(
      task_ptr->method_, run_ctx->predicted_wall_us_, actual_wall_us,
      run_ctx->predicted_stat_);

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

  // Fire-and-forget: skip all response paths, just delete the task
  if (task_ptr->task_flags_.Any(TASK_FIRE_AND_FORGET)) {
    task_ptr->ClearFlags(TASK_DATA_OWNER);
    container->DelTask(task_ptr->method_, task_ptr);
    return;
  }

  // If task is remote, enqueue to net_queue_ for SendOut. Choose the
  // latency vs I/O lane from the task's cached predicted_stat_ so a
  // small ACK / heartbeat reply doesn't queue behind a 1 MiB GetBlob
  // response on the wire.
  if (is_remote) {
    size_t io_size = run_ctx->predicted_stat_.io_size_;
    NetQueuePriority prio = (io_size >= kNetQueueIoThreshold)
                                ? NetQueuePriority::kSendOutIO
                                : NetQueuePriority::kSendOutLatency;
    CLIO_IPC->EnqueueNetTask(run_ctx->future_, prio);
    return;
  }

  // Copy variables from future_shm to stack BEFORE any SetComplete() call
  // This prevents use-after-free since client may free future_shm after
  // SetComplete()
  auto future_shm = run_ctx->future_.GetFutureShm();
  if (future_shm.IsNull()) {
    HLOG(kError, "EndTask: future_shm is NULL for pool={} method={}",
         task_ptr->pool_id_, task_ptr->method_);
    return;
  }
  bool was_copied = future_shm->flags_.Any(FutureShm::FUTURE_WAS_COPIED);

  // Copy parent task pointer before transfer begins (may be modified during
  // transfer)
  RunContext *parent_task = run_ctx->future_.GetParentTask();

  // Dispatch response via transport class
  IpcCpu2Self::RuntimeSend(task_ptr, run_ctx, container,
                           shm_send_transport_.get());
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
    // This can happen with orphan events from parallel subtasks. Use the
    // completion flag instead of coro_handle_.done() to avoid a use-after-free
    // on a cross-thread-freed coroutine frame (issue #485).
    if (is_started &&
        (!run_ctx->coro_handle_ || CoroCompleted(run_ctx))) {
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
  ctp::Timepoint batch_timestamp;
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
      // RouteTask handles Retry/Dne internally via AddToRetryQueue
      if (CLIO_IPC->RouteTask(run_ctx->future_) == RouteResult::ExecHere) {
        ExecTask(run_ctx->task_, run_ctx, is_started);

        // If task re-yielded with a polling interval, ExecTask already
        // re-added it to the periodic queue via AddToBlockedQueue.
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
  Future<Task, CLIO_QUEUE_ALLOC_T> future;
  while (event_queue_->Pop(future)) {
    HLOG(kDebug, "Worker {}: ProcessEventQueue popped subtask future",
         worker_id_);
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

    // Skip if coroutine handle is null or already completed. Use the
    // completion flag instead of coro_handle_.done() to avoid dereferencing a
    // coroutine frame a cross-thread completion may already have freed (#485).
    if (!run_ctx->coro_handle_ || CoroCompleted(run_ctx)) {
      continue;
    }

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

  // Process retry queue every 32 iterations (or on force)
  if (force || iteration_count_ % 32 == 0) {
    ProcessRetryQueue();
  }

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

void Worker::AddToRetryQueue(RunContext *run_ctx_ptr) {
  retry_queue_.push(run_ctx_ptr);
}

void Worker::ProcessRetryQueue() {
  size_t count = retry_queue_.size();
  for (size_t i = 0; i < count; ++i) {
    RunContext *run_ctx = retry_queue_.front();
    retry_queue_.pop();

    FullPtr<Task> task_ptr = run_ctx->task_;
    if (task_ptr.IsNull()) {
      continue;  // Skip invalid entries
    }

    // Clear TASK_ROUTED so RouteTask re-evaluates.
    // RouteTask handles Retry/Dne internally via AddToRetryQueue.
    task_ptr->ClearFlags(TASK_ROUTED);
    RouteResult result =
        CLIO_IPC->RouteTask(run_ctx->future_, /*force_enqueue=*/true);
    if (result == RouteResult::ExecHere) {
      // force_enqueue=true means this shouldn't happen, but handle it
      ExecTask(task_ptr, run_ctx, false);
    }
  }
}

void Worker::ReschedulePeriodicTask(RunContext *run_ctx,
                                    const FullPtr<Task> &task_ptr) {
  if (!run_ctx || task_ptr.IsNull() || !task_ptr->IsPeriodic()) {
    return;
  }

  // Reset timers and predictions for next period
  run_ctx->cpu_timer_.time_ns_ = 0;
  run_ctx->wall_timer_.time_ns_ = 0;
  run_ctx->predicted_load_ = 0;
  run_ctx->predicted_wall_us_ = 0;
  run_ctx->predicted_stat_ = TaskStat();

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
  Worker *worker = CLIO_CUR_WORKER;
  if (worker) {
    return worker->GetCurrentRunContext();
  }
  return nullptr;
}

}  // namespace clio::run