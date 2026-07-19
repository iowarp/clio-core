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

#ifndef CLIO_RUNTIME_INCLUDE_WORKERS_WORKER_H_
#define CLIO_RUNTIME_INCLUDE_WORKERS_WORKER_H_

#include <chrono>
// <coroutine> only for the C++20 stackless backend, not the Boost stackful one.
// (task.h, included below, derives CLIO_ENABLE_BOOST_COROUTINES from this.)
#if !defined(CLIO_ENABLE_BOOST_COROUTINES)
#include <coroutine>
#endif
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "clio_runtime/container.h"
#include "clio_runtime/pool_query.h"
#include "clio_runtime/task.h"
#include "clio_runtime/types.h"
#include "clio_runtime/scheduler/scheduler.h"
#include "clio_ctp/lightbeam/event_manager.h"
#include "clio_ctp/lightbeam/transport_factory_impl.h"
#include "clio_ctp/memory/allocator/malloc_allocator.h"

namespace clio::run {

// Forward declaration to avoid circular dependency
using WorkQueue =
    ctp::ipc::mpsc_ring_buffer<ctp::ipc::ShmPtr<TaskLane>, CLIO_QUEUE_ALLOC_T>;

// Forward declarations
class Task;

// Note: CachedContext is no longer needed since RunContext is now embedded
// directly in the Task object. Context caching has been eliminated.

/**
 * Structure to hold worker statistics for monitoring
 */
struct WorkerStats {
  u64 num_tasks_processed_;  /**< Total number of tasks this worker has processed */
  u32 num_queued_tasks_;     /**< Number of tasks waiting to be processed */
  u32 num_blocked_tasks_;    /**< Number of tasks in blocked queue */
  u32 num_periodic_tasks_;   /**< Number of periodic tasks on this worker */
  u32 num_retry_tasks_;      /**< Number of tasks in retry queue */
  u32 suspend_period_us_;    /**< Time in microseconds before the worker would suspend */
  u32 idle_iterations_;      /**< Number of consecutive idle iterations */
  bool is_running_;          /**< Whether the worker is currently running */
  bool is_active_;           /**< Whether the worker's lane is currently active (processing tasks) */
  u32 worker_id_;            /**< Worker identifier */
  float load_;               /**< Current estimated load in microseconds */

  /** Default constructor */
  WorkerStats()
      : num_tasks_processed_(0),
        num_queued_tasks_(0),
        num_blocked_tasks_(0),
        num_periodic_tasks_(0),
        num_retry_tasks_(0),
        suspend_period_us_(0),
        idle_iterations_(0),
        is_running_(false),
        is_active_(false),
        worker_id_(0),
        load_(0) {}

  template <typename Archive>
  void save(Archive& ar) const {
    ar(num_tasks_processed_, num_queued_tasks_, num_blocked_tasks_,
       num_periodic_tasks_, num_retry_tasks_, suspend_period_us_,
       idle_iterations_, is_running_, is_active_, worker_id_, load_);
  }

  template <typename Archive>
  void load(Archive& ar) {
    ar(num_tasks_processed_, num_queued_tasks_, num_blocked_tasks_,
       num_periodic_tasks_, num_retry_tasks_, suspend_period_us_,
       idle_iterations_, is_running_, is_active_, worker_id_, load_);
  }
};

// Macro for accessing CTP thread-local storage (worker thread context)
// This macro allows access to the current worker from any thread
// Example usage in ChiMod container code:
//   Worker* worker = CLIO_CUR_WORKER;
//   clio::run::shared_ptr<Task> current_task = worker->GetCurrentTask();
//   RunContext* run_ctx = worker->GetCurrentRunContext();
#define CLIO_CUR_WORKER \
  (CTP_THREAD_MODEL->GetTls<clio::run::Worker>(clio::run::chi_cur_worker_key_))
// Backward-compat alias (clio_run rebrand). External code that still
// uses the legacy CHI_* spelling keeps working unchanged.

/**
 * Worker class for executing tasks
 *
 * Manages active and cold lane queues, executes tasks using boost::fiber,
 * and provides task execution environment with stack allocation.
 */
class Worker {
 public:
  /**
   * Constructor
   * @param worker_id Unique worker identifier
   */
  Worker(u32 worker_id);

  /**
   * Destructor
   */
  ~Worker();

  /**
   * Initialize worker
   * @return true if initialization successful, false otherwise
   */
  bool Init();

  /**
   * Finalize and cleanup worker resources
   */
  void Finalize();

  /**
   * Main worker loop - processes tasks from queues
   */
  void Run();

  /**
   * Stop the worker loop
   */
  void Stop();

  /**
   * Get worker ID
   * @return Worker identifier
   */
  u32 GetId() const;

  /** OS thread id of this worker (valid once Run() has started). #642 */
  u32 GetTid() const { return tid_; }

  /**
   * Get the event queue for this worker
   * @return Pointer to this worker's event queue
   */
  auto *GetEventQueue() { return event_queue_; }

  /**
   * Check if worker is running
   * @return true if worker is active, false otherwise
   */
  bool IsRunning() const;

  /**
   * Set the task currently executing on this worker thread (the worker sets
   * this before every Run/resume so handlers can reach it via GetCurrentTask).
   * @param task The executing task (null to clear)
   */
  void SetCurrentTask(const clio::run::shared_ptr<Task> &task);

  /**
   * Get the task currently executing on this worker thread.
   * @return Reference to the current task handle (null if idle)
   */
  clio::run::shared_ptr<Task> &GetCurrentTask();

  /**
   * Get current lane from the current RunContext
   * @return Pointer to current lane or nullptr if no RunContext
   */
  TaskLane *GetCurrentLane() const;

  /**
   * Set this worker as the current worker in thread-local storage
   */
  void SetAsCurrentWorker();

  /**
   * Clear the current worker from thread-local storage
   */
  static void ClearCurrentWorker();

  /**
   * Set whether the current task did actual work
   * @param did_work true if task did work, false if idle/no work
   */
  void SetTaskDidWork(bool did_work);

  /**
   * Get whether the current task did actual work
   * @return true if task did work, false if idle/no work
   */
  bool GetTaskDidWork() const;

  /**
   * Get worker statistics for monitoring
   * @return WorkerStats struct containing current worker statistics
   */
  WorkerStats GetWorkerStats() const;

  /**
   * Get the EventManager for this worker
   * @return Reference to this worker's EventManager
   */
  ctp::lbm::EventManager& GetEventManager();

  /**
   * Add a task to the blocked queue based on block count
   * @param task The blocked task
   * @param wait_for_task If true, do not add to blocked queue (task is waiting
   * for subtask completion)
   */
  void AddToBlockedQueue(const clio::run::shared_ptr<Task> &task,
                         bool wait_for_task = false);

  /**
   * Add a task to the retry queue (container migrated or plugged)
   * @param task The task to retry
   */
  void AddToRetryQueue(const clio::run::shared_ptr<Task> &task);

  /**
   * Reschedule a periodic task for next execution
   * Checks if lane still maps to this worker - if so, adds to blocked queue
   * Otherwise, reschedules task back to the lane
   * @param task_ptr Full pointer to the periodic task
   */
  void ReschedulePeriodicTask(clio::run::shared_ptr<Task> &task_ptr);

  /**
   * Set the worker's assigned lane
   * @param lane Pointer to the TaskLane assigned to this worker
   */
  void SetLane(TaskLane *lane);

  /**
   * Get the worker's assigned lane
   * @return Pointer to the TaskLane assigned to this worker
   */
  TaskLane *GetLane() const;

  /**
   * Set GPU lanes for this worker to process
   * @param lanes Vector of TaskLane pointers for GPU queues
   */
  void SetGpuLanes(const std::vector<GpuTaskLane *> &lanes);

  /**
   * Get the worker's assigned GPU lanes
   * @return Reference to vector of GPU TaskLanes
   */
  const std::vector<GpuTaskLane *> &GetGpuLanes() const;

  /**
   * Poll all GPU lanes for new tasks.
   * Called from PollOnce when this worker has GPU lanes assigned.
   * @return Number of GPU tasks processed
   */
  u32 ProcessNewTasksGpu();

  /**
   * Process a single task from a GPU lane.
   * Pops from the lane; the task pointer is already resolved by the inbound
   * Ipc call that enqueued it.
   * @param gpu_lane TaskLane to poll
   * @return true if a task was processed
   */
  bool ProcessNewTaskGpu(GpuTaskLane *gpu_lane);

 private:
  /**
   * Process a blocked queue, checking tasks and re-queuing as needed
   * @param queue Reference to the ext_ring_buffer to process
   * @param queue_idx Index of the queue being processed (0-3)
   */
  void ProcessBlockedQueue(std::queue<clio::run::shared_ptr<Task>> &queue, u32 queue_idx);

  /**
   * Process a periodic queue, checking time-based tasks and executing if ready
   * @param queue Reference to the ext_ring_buffer to process
   * @param queue_idx Index of the queue being processed (0-3)
   */
  void ProcessPeriodicQueue(std::queue<clio::run::shared_ptr<Task>> &queue, u32 queue_idx);

  /**
   * Process event queue for waking up tasks when subtasks complete
   * Iterates over event_queue_, removes tasks from blocked_queue_, and calls
   * ExecTask
   */
  void ProcessEventQueue();

  /**
   * Process retry queue: re-check containers and re-route as needed
   * Tasks with plugged containers stay in retry queue.
   * Tasks with nullptr containers get re-routed via RouteGlobal.
   */
  void ProcessRetryQueue();


 public:
  /**
   * End task execution and perform cleanup
   * @param task_ptr Full pointer to task to end
   * @param can_resched Whether task can be rescheduled (false on error)
   */
  void EndTask(clio::run::shared_ptr<Task> &task_ptr, bool can_resched);

 private:
  /**
   * Continue processing blocked tasks that are ready to resume
   * @param force If true, process both queues regardless of iteration count
   */
  void ContinueBlockedTasks(bool force);

  /**
   * Process tasks from a given lane
   * Processes up to MAX_TASKS_PER_ITERATION tasks per call
   * @param lane The TaskLane to process tasks from
   * @return Number of tasks processed
   */
  u32 ProcessNewTasks(TaskLane *lane);

  /**
   * Process a single task from a given lane
   * Handles task retrieval, deserialization, routing, and execution
   * @param lane The TaskLane to pop a task from
   * @return true if a task was processed, false if lane was empty
   */
  bool ProcessNewTask(TaskLane *lane);

  /**
   * Get the time remaining before the next periodic task should resume
   * Scans all periodic queues to find the task with the shortest remaining time
   * @return Time in microseconds until next periodic task, or 0 if no periodic tasks
   */
  double GetSuspendPeriod() const;

  /**
   * Suspend worker when there is no work available
   * Implements adaptive sleep algorithm with busy wait and linear increment
   */
  void SuspendMe();

  /**
   * Execute task with context switching capability
   * Uses C++20 coroutines for suspension and resumption
   * @param task_ptr Full pointer to task to execute
   * @param is_started True if task is resuming, false for new task
   */
  void ExecTask(clio::run::shared_ptr<Task> &task_ptr, bool is_started);

  // NOTE: StartCoroutine / ResumeCoroutine / MakeTaskFiber now live on Task —
  // the task owns its RunContext (and thus its coroutine frame), so driving the
  // coroutine is the task's responsibility. ExecTask calls task->StartCoroutine
  // / task->ResumeCoroutine.

  u32 worker_id_;
  u32 tid_ = 0;  /**< OS thread id, set in Run() (#642) */
  bool is_running_;
  bool is_initialized_;
  float load_;          // Estimated total CPU time (us) of active tasks
  bool did_work_;       // Tracks if any work was done in current loop iteration
  bool task_did_work_;  // Tracks if current task did actual work (set by tasks
                        // via CLIO_CUR_WORKER)

  // Task currently executing on this worker thread (null when idle). The
  // RunContext lives inside this Task; it is never held as a bare pointer.
  clio::run::shared_ptr<Task> current_task_;

  // Single lane assigned to this worker (one lane per worker)
  TaskLane *assigned_lane_;

  // GPU lanes assigned to this worker (one lane per GPU, empty when no GPU)
  std::vector<GpuTaskLane *> gpu_lanes_;

  // Note: RunContext cache removed - RunContext is now embedded in Task

  // Blocked queue system for cooperative tasks (waiting for subtasks):
  // - Queue[0]: Tasks blocked <=2 times (checked every % 2 iterations)
  // - Queue[1]: Tasks blocked <= 4 times (checked every % 4 iterations)
  // - Queue[2]: Tasks blocked <= 8 times (checked every % 8 iterations)
  // - Queue[3]: Tasks blocked > 8 times (checked every % 16 iterations)
  // Using std::queue for O(1) enqueue/dequeue operations
  static constexpr u32 NUM_BLOCKED_QUEUES = 4;
  static constexpr u32 BLOCKED_QUEUE_SIZE = 1024;
  std::queue<clio::run::shared_ptr<Task>> blocked_queues_[NUM_BLOCKED_QUEUES];

  // Retry queue for tasks whose containers migrated away
  // Tasks are retried every 32 iterations. During retry:
  //   - If container is plugged: put back in retry queue
  //   - If container is nullptr: re-route via RouteGlobal
  //   - If container is available: execute locally
  std::queue<clio::run::shared_ptr<Task>> retry_queue_;

  // Event queue for completing subtask futures on the parent worker's thread.
  // Stores Future<Task> objects to set FUTURE_COMPLETE, avoiding stale
  // RunContext* pointers. Allocated from malloc allocator (temporary runtime
  // data, not IPC), sized in Init() to EVENT_QUEUE_DEPTH_MULTIPLIER x the
  // configured per-worker task-queue depth (GetQueueDepth()): a parent fills at
  // most ~queue_depth subtask slots in its lane, so giving the completion ring
  // 2x that headroom keeps the WAIT_FOR_SPACE Emplace from ever blocking on a
  // single parent's fan-out (which otherwise self-deadlocks the worker, #620).
  static constexpr u32 EVENT_QUEUE_DEPTH_MULTIPLIER = 2;
  ctp::ipc::mpsc_ring_buffer<Future<Task, CLIO_QUEUE_ALLOC_T>, ctp::ipc::MallocAllocator> *event_queue_;

  // Boost fiber stacks are pooled by the process-wide BoostStackPool() (a
  // per-thread SlabAllocator over BoostStackAllocator), reused by AllocateStack
  // to avoid malloc/free of the (256 KiB) stack per task. No per-worker member.

  // Periodic queue system for time-based periodic tasks:
  // - Queue[0]: Tasks with yield_time_us_ <= 50us (checked every 16 iterations)
  // - Queue[1]: Tasks with yield_time_us_ <= 200us (checked every 32
  // iterations)
  // - Queue[2]: Tasks with yield_time_us_ <= 50ms/50000us (checked every 64
  // iterations)
  // - Queue[3]: Tasks with yield_time_us_ > 50ms (checked every 128 iterations)
  // Using std::queue for O(1) enqueue/dequeue operations
  static constexpr u32 NUM_PERIODIC_QUEUES = 4;
  static constexpr u32 PERIODIC_QUEUE_SIZE = 1024;
  std::queue<clio::run::shared_ptr<Task>> periodic_queues_[NUM_PERIODIC_QUEUES];

  // Worker spawn time
  ctp::Timepoint spawn_time_;  // Time when worker was spawned

  // Task completion counter (incremented in EndTask)
  u64 num_tasks_processed_;  // Total tasks completed by this worker

  // Iteration counter for periodic blocked queue checks
  u64 iteration_count_;  // Number of iterations completed

  // Sleep management for idle workers
  u64 idle_iterations_;   // Number of consecutive iterations with no work
  u32 current_sleep_us_;  // Current sleep duration in microseconds
  u64 sleep_count_;  // Number of times sleep was called in current idle period
  ctp::Timepoint idle_start_;  // Time when worker became idle

  // EventManager for efficient worker suspension and event monitoring
  ctp::lbm::EventManager event_manager_;

  // SHM lightbeam transport (worker-side)
  ctp::lbm::TransportPtr shm_send_transport_;  // For IpcManager::SendRuntime
  ctp::lbm::TransportPtr shm_recv_transport_;  // For ProcessNewTask

  // Scheduler pointer (owned by IpcManager, not Worker)
  Scheduler *scheduler_;
};

}  // namespace clio::run

#endif  // CLIO_RUNTIME_INCLUDE_WORKERS_WORKER_H_