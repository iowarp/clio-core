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

#ifndef CHIMAERA_INCLUDE_CHIMAERA_WORKERS_WORKER_H_
#define CHIMAERA_INCLUDE_CHIMAERA_WORKERS_WORKER_H_

#include <chrono>
#include <coroutine>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "chimaera/container.h"
#include "chimaera/integer_timer.h"
#include "chimaera/local_transfer.h"
#include "chimaera/pool_query.h"
#include "chimaera/task.h"
#include "chimaera/task_queue.h"
#include "chimaera/types.h"
#include "chimaera/scheduler/scheduler.h"
#include "hermes_shm/lightbeam/event_manager.h"
#include "hermes_shm/lightbeam/transport_factory_impl.h"
#include "hermes_shm/memory/allocator/malloc_allocator.h"

namespace chi {

// Forward declaration to avoid circular dependency
using WorkQueue =
    hshm::ipc::mpsc_ring_buffer<hipc::ShmPtr<TaskLane>, CHI_MAIN_ALLOC_T>;

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
  u32 suspend_period_us_;    /**< Time in microseconds before the worker would suspend */
  u32 idle_iterations_;      /**< Number of consecutive idle iterations */
  bool is_running_;          /**< Whether the worker is currently running */
  bool is_active_;           /**< Whether the worker's lane is currently active (processing tasks) */
  u32 worker_id_;            /**< Worker identifier */

  /** Default constructor */
  WorkerStats()
      : num_tasks_processed_(0),
        num_queued_tasks_(0),
        num_blocked_tasks_(0),
        num_periodic_tasks_(0),
        suspend_period_us_(0),
        idle_iterations_(0),
        is_running_(false),
        is_active_(false),
        worker_id_(0) {}

  template <typename Archive>
  void save(Archive& ar) const {
    ar(num_tasks_processed_, num_queued_tasks_, num_blocked_tasks_,
       num_periodic_tasks_, suspend_period_us_, idle_iterations_,
       is_running_, is_active_, worker_id_);
  }

  template <typename Archive>
  void load(Archive& ar) {
    ar(num_tasks_processed_, num_queued_tasks_, num_blocked_tasks_,
       num_periodic_tasks_, suspend_period_us_, idle_iterations_,
       is_running_, is_active_, worker_id_);
  }
};

// Macro for accessing HSHM thread-local storage (worker thread context)
// This macro allows access to the current worker from any thread
// Example usage in ChiMod container code:
//   Worker* worker = CHI_CUR_WORKER;
//   FullPtr<Task> current_task = worker->GetCurrentTask();
//   RunContext* run_ctx = worker->GetCurrentRunContext();
#define CHI_CUR_WORKER \
  (HSHM_THREAD_MODEL->GetTls<chi::Worker>(chi::chi_cur_worker_key_))

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

  /**
   * Check if worker is running
   * @return true if worker is active, false otherwise
   */
  bool IsRunning() const;

  /**
   * Get current RunContext for this worker thread
   * @return Pointer to current RunContext or nullptr
   */
  RunContext *GetCurrentRunContext() const;

  /**
   * Set current RunContext for this worker thread
   * @param rctx Pointer to RunContext to set as current
   * @return Pointer to the set RunContext
   */
  RunContext *SetCurrentRunContext(RunContext *rctx);

  /**
   * Get current task from the current RunContext
   * @return FullPtr to current task or null if no RunContext
   */
  FullPtr<Task> GetCurrentTask() const;

  /**
   * Get current container from the current RunContext
   * @return Pointer to current container or nullptr if no RunContext
   */
  Container *GetCurrentContainer() const;

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
  hshm::lbm::EventManager& GetEventManager();

  /**
   * Add run context to blocked queue based on block count
   * @param run_ctx_ptr Pointer to run context (task accessible via
   * run_ctx_ptr->task)
   * @param wait_for_task If true, do not add to blocked queue (task is waiting
   * for subtask completion)
   */
  void AddToBlockedQueue(RunContext *run_ctx_ptr, bool wait_for_task = false);

  /**
   * Reschedule a periodic task for next execution
   * Checks if lane still maps to this worker - if so, adds to blocked queue
   * Otherwise, reschedules task back to the lane
   * @param run_ctx_ptr Pointer to run context
   * @param task_ptr Full pointer to the periodic task
   */
  void ReschedulePeriodicTask(RunContext *run_ctx_ptr,
                              const FullPtr<Task> &task_ptr);

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

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
  /**
   * Set GPU lanes for this worker to process
   * @param lanes Vector of TaskLane pointers for GPU queues
   */
  void SetGpuLanes(const std::vector<TaskLane *> &lanes);

  /**
   * Get the worker's assigned GPU lanes
   * @return Reference to vector of GPU TaskLanes
   */
  const std::vector<TaskLane *> &GetGpuLanes() const;
#endif

  /**
   * Route a task by calling ResolvePoolQuery and determining local vs global
   * scheduling
   * @param future Future containing the task to route
   * @param lane Pointer to the task lane for execution context
   * @param container The container to use for task execution
   * @return true if task was successfully routed, false otherwise
   */
  bool RouteTask(Future<Task> &future, TaskLane *lane, Container *container);

  /**
   * Resolve a pool query into concrete physical addresses
   * @param query Pool query to resolve
   * @param pool_id Pool ID for the query
   * @param task_ptr Task pointer (needed for Dynamic routing)
   * @return Vector of pool queries for routing
   */
  std::vector<PoolQuery> ResolvePoolQuery(const PoolQuery &query,
                                          PoolId pool_id,
                                          const FullPtr<Task> &task_ptr);

 private:
  // Pool query resolution helper functions
  std::vector<PoolQuery> ResolveLocalQuery(const PoolQuery &query,
                                           const FullPtr<Task> &task_ptr);
  std::vector<PoolQuery> ResolveDynamicQuery(const PoolQuery &query,
                                             PoolId pool_id,
                                             const FullPtr<Task> &task_ptr);
  std::vector<PoolQuery> ResolveDirectIdQuery(const PoolQuery &query,
                                              PoolId pool_id,
                                              const FullPtr<Task> &task_ptr);
  std::vector<PoolQuery> ResolveDirectHashQuery(const PoolQuery &query,
                                                PoolId pool_id,
                                                const FullPtr<Task> &task_ptr);
  std::vector<PoolQuery> ResolveRangeQuery(const PoolQuery &query,
                                           PoolId pool_id,
                                           const FullPtr<Task> &task_ptr);
  std::vector<PoolQuery> ResolveBroadcastQuery(const PoolQuery &query,
                                               PoolId pool_id,
                                               const FullPtr<Task> &task_ptr);
  std::vector<PoolQuery> ResolvePhysicalQuery(const PoolQuery &query,
                                              PoolId pool_id,
                                              const FullPtr<Task> &task_ptr);

  /**
   * Process a blocked queue, checking tasks and re-queuing as needed
   * @param queue Reference to the ext_ring_buffer to process
   * @param queue_idx Index of the queue being processed (0-3)
   */
  void ProcessBlockedQueue(std::queue<RunContext *> &queue, u32 queue_idx);

  /**
   * Process a periodic queue, checking time-based tasks and executing if ready
   * @param queue Reference to the ext_ring_buffer to process
   * @param queue_idx Index of the queue being processed (0-3)
   */
  void ProcessPeriodicQueue(std::queue<RunContext *> &queue, u32 queue_idx);

  /**
   * Process event queue for waking up tasks when subtasks complete
   * Iterates over event_queue_, removes tasks from blocked_queue_, and calls
   * ExecTask
   */
  void ProcessEventQueue();

  /**
   * Copy task output data to copy space for streaming to clients
   * Processes client_copy_ queue with time budget of up to 10ms per call
   * For each task:
   * - Acquires reader lock on IpcManager::allocator_map_lock_
   * - Checks if FUTURE_NEW_DATA flag is set
   * - If not set: copies data to copy space, sets FUTURE_NEW_DATA flag
   * - Waits up to 1ms for client to consume data (unset FUTURE_NEW_DATA)
   * - If still set after 1ms: pushes task to back of queue, continues to next
   * - If data fully copied: sets FUTURE_COMPLETE, removes from queue
   */
  void CopyTaskOutputToClient();

 public:
  /**
   * Check if task should be processed locally based on task flags and pool
   * queries
   * @param task_ptr Full pointer to task to check for TASK_FORCE_NET flag
   * @param pool_queries Vector of pool queries from ResolvePoolQuery
   * @return true if task should be processed locally, false for global routing
   */
  bool IsTaskLocal(const FullPtr<Task> &task_ptr,
                   const std::vector<PoolQuery> &pool_queries);

  /**
   * Route task locally using container query and Monitor with kLocalSchedule
   * @param future Future containing the task to route locally
   * @param lane Pointer to the task lane for execution context
   * @param container The container to use for task execution
   * @return true if local routing successful, false otherwise
   */
  bool RouteLocal(Future<Task> &future, TaskLane *lane, Container *container);

  /**
   * Route task globally using admin client's ClientSendTaskIn method
   * @param future Future containing the task to route globally
   * @param pool_queries Vector of pool queries for global routing
   * @return true if global routing successful, false otherwise
   */
  bool RouteGlobal(Future<Task> &future,
                   const std::vector<PoolQuery> &pool_queries);

  /**
   * Begin client transfer for task outputs
   * Called only when task was copied from client (was_copied = true)
   * @param task_ptr Task to serialize
   * @param run_ctx Runtime context
   * @param container Container for serialization
   */
  void EndTaskShmTransfer(const FullPtr<Task> &task_ptr,
                             RunContext *run_ctx, Container *container);

  /**
   * End task execution and perform cleanup
   * @param task_ptr Full pointer to task to end
   * @param run_ctx Pointer to RunContext for task
   * @param can_resched Whether task can be rescheduled (false on error)
   */
  void EndTask(const FullPtr<Task> &task_ptr, RunContext *run_ctx,
               bool can_resched);

 private:
  /**
   * Begin task execution
   * @param future Future object containing the task and completion state
   * @param container Container for the task
   * @param lane Lane for the task (can be nullptr)
   */
  void BeginTask(Future<Task> &future, Container *container, TaskLane *lane);

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
   * Get task pointer from Future, copying from client if needed
   * Deserializes task if FUTURE_COPY_FROM_CLIENT flag is set
   * @param future Future object containing FutureShm
   * @param container Container to allocate task in
   * @param method_id Method ID for task creation
   * @return FullPtr to task (either existing or newly deserialized)
   */
  hipc::FullPtr<Task> GetOrCopyTaskFromFuture(Future<Task> &future,
                                               Container *container,
                                               u32 method_id);

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
   * @param run_ctx_ptr Pointer to existing RunContext
   * @param is_started True if task is resuming, false for new task
   */
  void ExecTask(const FullPtr<Task> &task_ptr, RunContext *run_ctx_ptr,
                bool is_started);

  /**
   * Start coroutine execution for a new task
   * Creates the coroutine and runs until first suspension point
   * @param task_ptr Full pointer to task to execute
   * @param run_ctx Pointer to RunContext for task
   */
  void StartCoroutine(const FullPtr<Task> &task_ptr, RunContext *run_ctx);

  /**
   * Resume coroutine execution for a yielded/blocked task
   * @param task_ptr Full pointer to task to resume
   * @param run_ctx Pointer to RunContext for task
   */
  void ResumeCoroutine(const FullPtr<Task> &task_ptr, RunContext *run_ctx);

  /**
   * End dynamic scheduling task and re-route with updated pool query
   * @param task_ptr Full pointer to task to re-route
   * @param run_ctx Pointer to RunContext for task
   */
  void RerouteDynamicTask(const FullPtr<Task> &task_ptr, RunContext *run_ctx);

  u32 worker_id_;
  bool is_running_;
  bool is_initialized_;
  bool did_work_;       // Tracks if any work was done in current loop iteration
  bool task_did_work_;  // Tracks if current task did actual work (set by tasks
                        // via CHI_CUR_WORKER)

  // Current RunContext for this worker thread
  RunContext *current_run_context_;

  // Single lane assigned to this worker (one lane per worker)
  TaskLane *assigned_lane_;

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
  // GPU lanes assigned to this worker (one lane per GPU)
  std::vector<TaskLane *> gpu_lanes_;
#endif

  // Note: RunContext cache removed - RunContext is now embedded in Task

  // Blocked queue system for cooperative tasks (waiting for subtasks):
  // - Queue[0]: Tasks blocked <=2 times (checked every % 2 iterations)
  // - Queue[1]: Tasks blocked <= 4 times (checked every % 4 iterations)
  // - Queue[2]: Tasks blocked <= 8 times (checked every % 8 iterations)
  // - Queue[3]: Tasks blocked > 8 times (checked every % 16 iterations)
  // Using std::queue for O(1) enqueue/dequeue operations
  static constexpr u32 NUM_BLOCKED_QUEUES = 4;
  static constexpr u32 BLOCKED_QUEUE_SIZE = 1024;
  std::queue<RunContext *> blocked_queues_[NUM_BLOCKED_QUEUES];

  // Event queue for completing subtask futures on the parent worker's thread
  // Stores Future<Task> objects to set FUTURE_COMPLETE, avoiding stale RunContext* pointers
  // Allocated from malloc allocator (temporary runtime data, not IPC)
  static constexpr u32 EVENT_QUEUE_DEPTH = 1024;
  hshm::ipc::mpsc_ring_buffer<Future<Task, CHI_MAIN_ALLOC_T>, hshm::ipc::MallocAllocator> *event_queue_;

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
  std::queue<RunContext *> periodic_queues_[NUM_PERIODIC_QUEUES];

  // Worker spawn time and queue processing tracking
  hshm::Timepoint spawn_time_;  // Time when worker was spawned
  u64 last_long_queue_check_;   // Last time (in 10us units) long queue was
                                // processed

  // Iteration counter for periodic blocked queue checks
  u64 iteration_count_;  // Number of iterations completed

  // Sleep management for idle workers
  u64 idle_iterations_;   // Number of consecutive iterations with no work
  u32 current_sleep_us_;  // Current sleep duration in microseconds
  u64 sleep_count_;  // Number of times sleep was called in current idle period
  hshm::Timepoint idle_start_;  // Time when worker became idle

  // EventManager for efficient worker suspension and event monitoring
  hshm::lbm::EventManager event_manager_;

  // Client copy queue - LocalTransfer objects streaming output data to clients
  std::queue<LocalTransfer> client_copy_;

  // SHM lightbeam transport (worker-side)
  std::unique_ptr<hshm::lbm::Transport> shm_send_transport_;  // For EndTaskShmTransfer
  std::unique_ptr<hshm::lbm::Transport> shm_recv_transport_;  // For ProcessNewTask

  // Scheduler pointer (owned by IpcManager, not Worker)
  Scheduler *scheduler_;
};

}  // namespace chi

#endif  // CHIMAERA_INCLUDE_CHIMAERA_WORKERS_WORKER_H_