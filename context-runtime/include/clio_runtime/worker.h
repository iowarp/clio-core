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

#include <atomic>
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
#include <unordered_map>
#include <vector>

#include "clio_runtime/batch_groups.h"
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
  // issue #785: atomic — Stop() is called from another thread while Run() polls
  // it. TSan flagged the plain bool.
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
  u32 GetTid() const { return tid_.load(std::memory_order_acquire); }

  /**
   * Get the event queue for this worker
   * @return Pointer to this worker's event queue
   */
  auto *GetEventQueue() {
    return event_queue_.load(std::memory_order_acquire);
  }

  /** Event-queue element type (completion futures for parked parents).
   * issue #822: ext_spsc_queue (mutex-serialized, growable) instead of a
   * WAIT_FOR_SPACE MPSC ring. The old ring relied on a sizing invariant —
   * "a parent fills at most ~queue_depth subtask slots in its lane, so 2x
   * that is enough completion headroom" (#620) — that no longer holds now
   * that task lanes GROW past queue_depth; overflowing it would busy-spin
   * the pushing worker forever. A growable queue removes the invariant. */
  using EventQueue =
      ctp::ipc::ext_spsc_queue<Future<Task, CLIO_QUEUE_ALLOC_T>,
                               ctp::ipc::MallocAllocator>;

  /**
   * issue #785: take ownership of a stalled worker's event queue.
   *
   * The insight that makes this cheap: a parked task holds a raw pointer to the
   * queue OBJECT (stamped into its RunContext at ProcessNewTask). Transferring
   * the object — rather than draining its contents — therefore redirects both
   * the events already queued AND every event a still-running subtask will push
   * later. No per-task re-pointing and no producer-side handshake is needed,
   * because nothing about the tasks changes.
   *
   * Safe because the donor is stalled inside ExecTask and so is provably not in
   * ProcessEventQueue. Called on the monitor thread.
   */
  void AdoptEventQueue(EventQueue *q) {
    if (q == nullptr) {
      return;
    }
    // APPEND, never replace. An earlier version stored over event_queue_, which
    // silently orphaned whatever this worker already owned — a recycled
    // replacement can hold events for tasks that still point at its original
    // queue, and those tasks would then never wake. That is task LOSS, not just
    // a leak. A worker therefore drains its own queue plus every queue it has
    // adopted.
    std::lock_guard<std::mutex> lk(park_mtx_);
    adopted_event_queues_.push_back(q);
  }

  /**
   * issue #785: hand off this worker's event queue and install a fresh empty
   * one, so the donor still has somewhere to put completions for subtasks its
   * own in-flight task spawns after it recovers.
   * @return the queue that was released.
   */
  EventQueue *ReplaceEventQueue();

  /**
   * issue #785: hand every queue this worker has ADOPTED to \a dst.
   *
   * Rescues cascade — a replacement adopts a donor's queue, then wedges and is
   * rescued in turn. Without this its adopted queues stay with it and nobody
   * drains them, so the parked tasks pointing at those queues are stranded
   * permanently: exactly the failure the adoption was meant to prevent, just
   * one level deeper. Only the worker's OWN queue moves via ReplaceEventQueue;
   * this moves the inherited ones.
   *
   * Preserves the one-consumer-per-queue invariant: entries are MOVED, never
   * copied.
   */
  void TransferAdoptedEventQueuesTo(Worker *dst) {
    if (dst == nullptr || dst == this) {
      return;
    }
    // std::lock, matching MigrateParkedTo. Only the monitor thread calls this
    // today, so a fixed this->dst order would work — but leaving two different
    // lock orders in the same class is a landmine for whoever adds the second
    // caller.
    std::lock(park_mtx_, dst->park_mtx_);
    std::lock_guard<std::mutex> lk_src(park_mtx_, std::adopt_lock);
    std::lock_guard<std::mutex> lk_dst(dst->park_mtx_, std::adopt_lock);
    for (EventQueue *q : adopted_event_queues_) {
      dst->adopted_event_queues_.push_back(q);
    }
    adopted_event_queues_.clear();
  }

  /**
   * issue #785: move this worker's PARKED state (blocked / periodic / retry
   * queues) to \a dst. Completes the rescue: the lane covers tasks not yet
   * started and the event queue covers tasks parked on a co_await, but a task
   * that yielded with a timer, or is waiting to be re-routed, sits in a plain
   * worker-private std::queue that no other thread can reach.
   *
   * Periodic tasks matter most here. ProcessPeriodicQueue re-routes each task
   * through RouteTask every period, so a non-pinned periodic task self-heals
   * its placement as soon as ANYONE scans it — moving the queue to a live
   * worker is the whole fix. A wedged worker never scans, so its periodic
   * tasks simply stop running.
   *
   * Both workers' park locks are taken with std::lock to avoid deadlock, and
   * the lock is never held across ExecTask (see park_mtx_), so a wedged donor
   * can never block the monitor thread here.
   *
   * @param dst worker to receive the parked tasks.
   * @return number of tasks moved.
   */
  size_t MigrateParkedTo(Worker *dst);

  /**
   * Check if worker is running
   * @return true if worker is active, false otherwise
   */
  bool IsRunning() const;

  // --- issue #781: measured-load + stall-detection accessors ---

  /** issue #785: tasks completed by this worker. Read by the monitor thread's
   *  progress watchdog; relaxed because it only needs to observe CHANGE. */
  u64 TasksProcessed() const {
    return num_tasks_processed_.load(std::memory_order_relaxed);
  }

  /** issue #785: completions of NON-PERIODIC tasks only.
   *
   *  The progress watchdog must use this rather than TasksProcessed(). Periodic
   *  tasks — the network and client-IPC pollers — complete and reschedule
   *  continuously whether or not the runtime is making real progress, so a
   *  counter that includes them never stops advancing and the watchdog can
   *  never fire. Measured: with every worker occupied by a 14 s blocking task
   *  and nothing else able to run, the all-tasks counter kept climbing and no
   *  alarm was raised. Polling is not progress. */
  u64 RealTasksProcessed() const {
    return num_nonperiodic_processed_.load(std::memory_order_relaxed);
  }

  /** Estimated queued CPU time (us), bumped/decremented around ExecTask. */
  float Load() const { return load_.load(std::memory_order_relaxed); }

  /** True while a task is actively executing on this worker. */
  bool IsExecuting() const {
    return last_exec_start_us_.load(std::memory_order_relaxed) != 0;
  }

  /**
   * Measured real-time load used for routing (issue #781):
   *   load_             predicted cost of the task currently EXECUTING
   *   queued_load_us_   predicted cost of tasks assigned/queued but not started
   *   elapsed           how long the executing task has OVERRUN wall-clock
   * The predicted terms come from the learned per-method model (converges after
   * a few runs); the elapsed term is the pure-measured signal that catches a
   * task overrunning its prediction (or a mislabeled non-yielding spin). Together
   * they steer new work off both busy AND backlogged workers.
   * @param now_us current steady-clock time in microseconds.
   */
  double RealtimeLoad(double now_us) const {
    long long start = last_exec_start_us_.load(std::memory_order_relaxed);
    double elapsed = (start != 0) ? (now_us - static_cast<double>(start)) : 0.0;
    return static_cast<double>(load_.load(std::memory_order_relaxed)) +
           static_cast<double>(queued_load_us_.load(std::memory_order_relaxed)) +
           elapsed;
  }

  /** #781: reserve predicted cost on this worker when a task is mapped to it.
   *  Integer atomic (µs) — atomic<double> fetch_add is not portable (icx/MSVC). */
  void ReserveLoad(double us) {
    if (us > 0.0) {
      queued_load_us_.fetch_add(static_cast<long long>(us),
                                std::memory_order_relaxed);
    }
  }
  /** #781: release the reservation when the task starts executing (or is
   *  cancelled) — the executing cost is then tracked by load_ instead. */
  void ReleaseReservation(double us) {
    if (us > 0.0) {
      long long amt = static_cast<long long>(us);
      long long prev = queued_load_us_.fetch_sub(amt, std::memory_order_relaxed);
      if (prev - amt < 0) queued_load_us_.store(0, std::memory_order_relaxed);
    }
  }

  /**
   * True if this worker has been executing one task longer than \a threshold_sec
   * — i.e. a non-yielding/mislabeled task is stalling it (issue #781).
   * @param now_us current steady-clock time in microseconds.
   */
  bool IsStalled(double now_us, double threshold_sec) const {
    long long start = last_exec_start_us_.load(std::memory_order_relaxed);
    return start != 0 &&
           (now_us - static_cast<double>(start)) > threshold_sec * 1e6;
  }

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
   * issue #785: take over \a lane from a stalled worker (D5 lane transfer).
   *
   * Lanes are allocated 1:1 with num_threads at startup with no spare, so an
   * elastic worker cannot be given a fresh lane — the rescue moves the wedged
   * worker's lane instead, which is a swap rather than an allocation. It also
   * avoids draining a single-consumer MPSC ring from a foreign thread, which is
   * the "steal-safe pop" problem #781 left open.
   *
   * Safe because the donor worker is, by definition of stalled, inside ExecTask
   * and provably not popping. Republishes the lane's tid so AwakenWorker's
   * tgkill reaches THIS thread.
   *
   * Called on the monitor thread.
   * @param lane The lane to adopt (may be null to release without adopting).
   */
  void AdoptLane(TaskLane *lane);

  /**
   * issue #785: give up this worker's lane without taking another. Used on the
   * stalled donor: it keeps running (we cannot interrupt it) but owns nothing,
   * so nothing new is stranded behind it.
   * @return The lane that was released (null if it had none).
   */
  TaskLane *ReleaseLane();

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

  /** issue #807: bind a resolved Future to this worker, route it, and execute it
   *  INLINE if it maps here (else RouteTask enqueues it to the right worker).
   *  Shared by the lane path (ProcessNewTask) and the inline SHM-ingest path
   *  (DrainMyShard). `lane` is where the task's RunContext should point for
   *  subtask wakeups (the popped lane, or this worker's own lane on ingest). */
  void RouteAndExec(Future<Task> &future, TaskLane *lane);

  /** issue #807/#820: bind + route only. Returns the resolved task when it
   *  should run HERE (so the caller may batch it before executing), null
   *  otherwise (RouteTask already enqueued it elsewhere). */
  clio::run::shared_ptr<Task> RouteOnly(Future<Task> &future, TaskLane *lane);

  /**
   * issue #820: the batching phase, over the SHM INGRESS.
   *
   * Take a bounded run of freshly arrived client requests, and give each one's
   * container the
   * chance to COALESCE it with its neighbours instead of running it now. Tasks
   * the container declines run as-is, in arrival order, first; the merged tasks
   * the container produces are submitted after.
   *
   * The dequeue bound is what keeps this honest: it can only ever batch what is
   * ALREADY queued, so a lone task finds an empty lane behind it, forms a group
   * of one, and is emitted unchanged. Batching therefore costs nothing when
   * there is no backlog to amortize, and only engages under the contention it
   * exists to remove.
   *
   * @return number of tasks dequeued
   */
  u32 BatchIngest(TaskLane *lane, u32 budget);

  /** issue #820: same phase, sourced from the LANE (where same-blob tasks
   *  converge). Experimental, CLIO_BATCH_LANE=1. */
  u32 BatchLane(TaskLane *lane, u32 budget);

  /** Shared body: collect from the lane or the shard, batch, execute. */
  u32 BatchCollect(TaskLane *lane, u32 budget, bool from_lane);

  /** issue #820: whether the batching phase runs at all (CLIO_TASK_BATCHING=0
   *  restores the pre-batching dequeue loop verbatim). */
  static bool BatchingEnabled();

  /** issue #820: complete the parents a merged task subsumed. Called from
   *  EndTask when the finished task carries TASK_BATCH_MERGED. */
  void CompleteBatchParents(clio::run::shared_ptr<Task> &merged);

  /** issue #807: drain this worker's inbound SHM shard, executing each request
   *  INLINE (no lane round-trip, no per-request wakeup). Returns count handled. */
  u32 DrainMyShard();

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
  // issue #785: atomic. Run() publishes it on the worker thread; AdoptLane
  // reads it on the monitor thread to republish lane ownership. TSan flagged
  // the plain u32.
  std::atomic<u32> tid_{0};  /**< OS thread id, set in Run() (#642) */
  // issue #785: atomic — Stop() is called from another thread while Run()
  // polls it. TSan flagged the plain bool.
  std::atomic<bool> is_running_;
  bool is_initialized_;
  // issue #785: atomic. Written only by the owning worker (ExecTask/EndTask)
  // but read by the monitor thread via RealtimeLoad() on every LoadBalance
  // tick, which TSan flags as a race on a plain float. Only load/store is used
  // — never fetch_add — because atomic<float> RMW is not portable (the #781
  // atomic<double> fetch_add broke MSVC-STL and icx).
  std::atomic<float> load_;  // Estimated total CPU time (us) of active tasks
  // issue #781: wall-clock timestamp stamped at the top of ExecTask and cleared
  // when the task returns/yields. While executing, (now() - last_exec_start_) is
  // the current task's elapsed time; the monitor thread uses this to detect a
  // worker stalled on a non-yielding task ( > kStallThresholdSec ) so the
  // runtime can spawn a replacement and steal the backlog. std::atomic so the
  // monitor thread can read it without a lock.
  std::atomic<long long> last_exec_start_us_{0};  // µs, 0 == not executing
  // issue #781: sum of predicted costs of tasks mapped to this worker but not
  // yet started (the queue). Added in RuntimeMapTask, released in ExecTask when
  // the task begins. Lets the mapper avoid a worker with a heavy task queued
  // even before it starts running — the queued-behind-heavy fix.
  std::atomic<long long> queued_load_us_{0};  // µs
  bool did_work_;       // Tracks if any work was done in current loop iteration
  bool task_did_work_;  // Tracks if current task did actual work (set by tasks
                        // via CLIO_CUR_WORKER)

  // Task currently executing on this worker thread (null when idle). The
  // RunContext lives inside this Task; it is never held as a bare pointer.
  clio::run::shared_ptr<Task> current_task_;

  // Single lane assigned to this worker (one lane per worker).
  // issue #785: atomic because the monitor thread reassigns it during a stall
  // rescue (AdoptLane/ReleaseLane) while this worker is running. Run() re-reads
  // it every iteration so a wedged worker picks up the change the moment it
  // finally returns from the bad task.
  std::atomic<TaskLane *> assigned_lane_;

  // GPU lanes assigned to this worker (one lane per GPU, empty when no GPU)
  std::vector<GpuTaskLane *> gpu_lanes_;

  // Note: RunContext cache removed - RunContext is now embedded in Task

  // Blocked queue system for cooperative tasks (waiting for subtasks):
  // - Queue[0]: Tasks blocked <=2 times (checked every % 2 iterations)
  // - Queue[1]: Tasks blocked <= 4 times (checked every % 4 iterations)
  // - Queue[2]: Tasks blocked <= 8 times (checked every % 8 iterations)
  // - Queue[3]: Tasks blocked > 8 times (checked every % 16 iterations)
  // Using std::queue for O(1) enqueue/dequeue operations
  // issue #785: guards blocked_queues_, periodic_queues_ and retry_queue_ —
  // plain std::queues that, before this, only the owning thread ever touched.
  // The monitor thread takes it to migrate parked work off a stalled worker.
  //
  // INVARIANT: never held across ExecTask. Callers pop under the lock, RELEASE,
  // then execute. That is what guarantees a worker wedged inside a task holds
  // nothing the monitor thread needs.
  mutable std::mutex park_mtx_;

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
  // configured per-worker task-queue depth (GetQueueDepth()). issue #822:
  // this is now only the INITIAL capacity — the queue grows when full
  // (ext_spsc_queue), so the old #620 requirement that the multiplier
  // out-size any single parent's fan-out no longer gates correctness.
  static constexpr u32 EVENT_QUEUE_DEPTH_MULTIPLIER = 2;
  // issue #785: atomic because the monitor thread reassigns it during a stall
  // rescue. ProcessEventQueue re-reads it each drain.
  std::atomic<EventQueue *> event_queue_;

  // issue #785: queues inherited from rescued workers. Drained alongside
  // event_queue_ so no parked task is ever orphaned. Guarded by park_mtx_.
  std::vector<EventQueue *> adopted_event_queues_;

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

  // ---- issue #820: worker-local task batching ----------------------------
  // Reused across phases (cleared, never reallocated) so the batching path
  // allocates nothing in steady state.
  BatchGroups batch_groups_;
  // Tasks their container declined to batch; they run as-is, in arrival order.
  // A plain vector, not a queue: this is produced and consumed by THIS worker
  // thread within one phase, so there is no handoff to synchronize.
  std::vector<clio::run::shared_ptr<Task>> batch_passthrough_;
  // issue #820/#822: the "merged uid -> parent tasks" registry is PROCESS-GLOBAL
  // (see worker.cc BatchPendingRegistry), NOT a per-worker member. A merged task
  // is an I/O task: it suspends on the bdev put/get and its continuation resumes
  // on WHATEVER worker the scheduler picks, which is frequently not the worker
  // that emitted it. A per-worker map made the completing worker miss the entry
  // and orphan the parents (client Wait() hangs forever). The global registry
  // also uses a global-unique key because CreateTaskId().unique_ is only unique
  // within one worker thread (thread-local counter), so two workers collide.

  // Worker spawn time
  ctp::Timepoint spawn_time_;  // Time when worker was spawned

  // Task completion counter (incremented in EndTask)
  // issue #785: atomic — the monitor thread's progress watchdog reads it.
  std::atomic<u64> num_tasks_processed_;  // Total tasks completed by this worker
  // issue #785: non-periodic completions only — see RealTasksProcessed().
  std::atomic<u64> num_nonperiodic_processed_{0};

  // Iteration counter for periodic blocked queue checks
  u64 iteration_count_;  // Number of iterations completed

  // Sleep management for idle workers
  // issue #785: atomic. GetWorkerStats() is now called from the monitor thread
  // on every watchdog tick, so this plain counter became a live cross-thread
  // race (TSan). Relaxed: it is a diagnostic, only its approximate value
  // matters.
  std::atomic<u64> idle_iterations_;  // consecutive iterations with no work
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