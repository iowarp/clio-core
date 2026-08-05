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

// <coroutine> only for the C++20 stackless backend, not the Boost stackful one.
// (CLIO_ENABLE_BOOST_COROUTINES is defined by task.h, included below, in terms of
// CLIO_ENABLE_BOOST_COROUTINES.)
#if !defined(CLIO_ENABLE_BOOST_COROUTINES)
#include <coroutine>
#endif
#include <cerrno>
#include <cstdlib>
#include <iostream>
#include <unordered_set>

// Include task_queue.h before other clio headers to ensure proper
// resolution
#include "clio_runtime/admin/admin_client.h"
#include "clio_runtime/container.h"
#include "clio_ctp/util/gpu_api.h"
#include "clio_runtime/ipc_manager.h"
#include "clio_runtime/pool_manager.h"
#include "clio_runtime/singletons.h"
#include "clio_runtime/task.h"
#include "clio_runtime/task_archives.h"
#include "clio_runtime/local_task_archives.h"
#include "clio_runtime/work_orchestrator.h"
#include "clio_runtime/boost_stack_allocator.h"

namespace clio::run {


// Stack detection is now handled by WorkOrchestrator during initialization

Worker::Worker(u32 worker_id)
    : worker_id_(worker_id),
      is_running_(false),
      is_initialized_(false),
      load_(0),
      did_work_(false),
      task_did_work_(false),
      current_task_(),
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
  // pointers. Sized to 2x the configured per-worker task-queue depth so a
  // single parent's completion fan-out can never fill this WAIT_FOR_SPACE ring
  // and block the worker inside its own completion Emplace (issue #620).
  u32 task_queue_depth = CLIO_CONFIG_MANAGER->GetQueueDepth();
  if (task_queue_depth == 0) {
    task_queue_depth = 1024;  // fallback if config is unset
  }
  u32 event_queue_depth = EVENT_QUEUE_DEPTH_MULTIPLIER * task_queue_depth;
  event_queue_.store(CTP_MALLOC
                         ->template NewObj<EventQueue>(CTP_MALLOC,
                                                       event_queue_depth)
                         .ptr_,
                     std::memory_order_release);

  // Boost fiber stacks now come from the process-wide, per-thread-cached
  // BoostStackPool() (see AllocateStack/FreeStack); no per-worker pool needed.

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
  stats.idle_iterations_ = idle_iterations_.load(std::memory_order_relaxed);

  // Calculate number of queued tasks (tasks waiting in the assigned lane)
  stats.num_queued_tasks_ = 0;
  stats.is_active_ = false;
  TaskLane *stats_lane = assigned_lane_.load(std::memory_order_acquire);
  if (stats_lane) {
    stats.num_queued_tasks_ = stats_lane->Size();
    stats.is_active_ = stats_lane->IsActive();
  }

  // Count blocked tasks across all blocked queues
  stats.num_blocked_tasks_ = 0;
  {
    std::lock_guard<std::mutex> lk(park_mtx_);  // #785
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
  }

  // Get suspend period (time until next periodic task or 0 if none)
  double suspend_period = GetSuspendPeriod();
  stats.suspend_period_us_ =
      (suspend_period < 0) ? 0 : static_cast<u32>(suspend_period);

  stats.num_tasks_processed_ =
      num_tasks_processed_.load(std::memory_order_relaxed);
  stats.load_ = load_.load(std::memory_order_relaxed);

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
  {
  std::lock_guard<std::mutex> lk(park_mtx_);  // #785: monitor may splice these
  for (u32 i = 0; i < NUM_BLOCKED_QUEUES; ++i) {
    while (!blocked_queues_[i].empty()) {
      // Each entry is an owning shared_ptr<Task>; popping drops the worker's
      // reference. The task (and its RunContext) is freed once its last owner
      // drops.
      blocked_queues_[i].pop();
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

  }  // release park_mtx_

  // Boost fiber stacks are owned by the process-wide BoostStackPool() (a
  // per-thread SlabAllocator cache); cached stacks live for the process and are
  // reclaimed at exit — no per-worker drain here.

  // Clear assigned lane reference (don't delete - it's in shared memory)
  assigned_lane_.store(nullptr, std::memory_order_release);

  is_initialized_ = false;
}

void Worker::Run() {
  if (!is_initialized_) {
    return;
  }
  // Set current worker once for the entire thread duration
  SetAsCurrentWorker();
  is_running_ = true;

  // Create this worker thread's IpcManagerTls on its OWN thread, which
  // ServerInit's the named MPSC SHM receive server "clio-<pid>-<tid>" (#642).
  // Producers reach this worker by that name; the DONTWAIT drain of it is wired
  // together with the ipc_cpu2cpu send side. Must run on the worker thread so
  // the segment is keyed to this thread's tid.
  CLIO_IPC->GetTls();

  // Set up the signal event BEFORE publishing the tid. AwakenWorker
  // tgkill(SIGUSR1)s any published tid unconditionally; if a producer fires
  // in the window between SetTid and AddSignalEvent (which blocks SIGUSR1
  // on this thread), the default disposition kills the whole process
  // (issue #520). On Windows the same order matters for liveness: Signal()
  // on a tid without a registered event is a lost wakeup.
  int tid = ctp::SystemInfo::GetTid();
  tid_.store(static_cast<u32>(tid), std::memory_order_release);  // #642
  event_manager_.AddSignalEvent(nullptr);
  if (TaskLane *lane = assigned_lane_.load(std::memory_order_acquire)) {
    lane->SetTid(tid);
  }
  // issue #807: become the consumer of this worker's inbound SHM shard ring, so
  // a producing client's SignalConsumerIfParked SIGUSR1s THIS worker (its tid is
  // now published on the ring). Must run after AddSignalEvent, which blocks
  // SIGUSR1 on this thread — otherwise the wake would kill the process.
  CLIO_IPC->RegisterShardConsumer(worker_id_);

  // Main worker loop - process tasks from assigned lane
  while (is_running_) {
    did_work_ = false;  // Reset work tracker at start of each loop iteration
    task_did_work_ = false;  // Reset task-level work tracker

    // issue #807: drain THIS worker's inbound SHM shard ring (worker w owns
    // shard w) and execute each request INLINE — no lane round-trip, no
    // per-request wakeup. A no-op for workers that own no shard / non-SHM.
    if (DrainMyShard() > 0) {
      did_work_ = true;
    }
    // issue #807: also drain any DEFERRED SHM responses (opt-in async send path)
    // onto the client rings, using this worker's own send transport. A cheap
    // no-op on the inline-default path (queues stay empty). Bounded so it can't
    // starve this worker's own task processing.
    if (CLIO_IPC->DrainShmSends(shm_send_transport_.get(), 16) > 0) {
      did_work_ = true;
    }

    // Process tasks from assigned lane
    // issue #785: re-read every iteration. The monitor thread may have moved
    // this lane to a replacement worker while we were wedged in a task, so a
    // cached pointer would keep us consuming a lane we no longer own.
    if (TaskLane *lane = assigned_lane_.load(std::memory_order_acquire)) {
      u32 count = ProcessNewTasks(lane);
      if (count > 0) did_work_ = true;
    }
    u32 gpu_count = ProcessNewTasksGpu();
    if (gpu_count > 0) did_work_ = true;

    // Check blocked queue for completed tasks at end of each iteration
    ContinueBlockedTasks(false);

    // ManyToOne: flush due collective batches on the neighborhood leader.
    // Driven from a single worker to avoid redundant locking; FlushDue is a
    // cheap no-op when no batches are pending. Treat pending batches as work so
    // this worker keeps polling and honors the short (e.g. 10us) batch window
    // instead of deep-sleeping.
    if (worker_id_ == 0) {
      BatchManager *bm = CLIO_IPC->GetBatchManager();
      if (bm != nullptr && bm->FlushDue(this)) {
        did_work_ = true;
      }
    }

    // Increment iteration counter
    iteration_count_++;

    if (!did_work_) {
      // No work was done - suspend worker with adaptive sleep
      SuspendMe();
    }

    if (did_work_) {
      // Work was done - reset idle counters
      idle_iterations_.store(0, std::memory_order_relaxed);
      current_sleep_us_ = 0;
      sleep_count_ = 0;
      did_work_ = false;
    }
  }

  // issue #807: stop being the shard ring's consumer before this thread's tid
  // can be recycled by an unrelated thread that does not block SIGUSR1.
  CLIO_IPC->UnregisterShardConsumer(worker_id_);

  // EventManager destructor handles signalfd and epoll cleanup
}

void Worker::Stop() { is_running_ = false; }

void Worker::SetLane(TaskLane *lane) {
  assigned_lane_.store(lane, std::memory_order_release);
  // Mark lane as active when assigned to worker
  if (lane) {
    lane->SetActive(true);
  }
}

TaskLane *Worker::GetLane() const {
  return assigned_lane_.load(std::memory_order_acquire);
}

void Worker::AdoptLane(TaskLane *lane) {
  if (lane != nullptr) {
    // Republish ownership BEFORE the lane becomes visible to our Run loop, so a
    // producer that signals between these two stores still targets this thread.
    lane->SetAssignedWorkerId(worker_id_);
    u32 my_tid = tid_.load(std::memory_order_acquire);
    if (my_tid != 0) {
      lane->SetTid(static_cast<int>(my_tid));
    }
    lane->SetActive(true);
  }
  assigned_lane_.store(lane, std::memory_order_release);
  HLOG(kWarning, "[#785] worker {} adopted lane {}", worker_id_,
       static_cast<const void *>(lane));
}

Worker::EventQueue *Worker::ReplaceEventQueue() {
  u32 depth = CLIO_CONFIG_MANAGER->GetQueueDepth();
  if (depth == 0) {
    depth = 1024;
  }
  EventQueue *fresh =
      CTP_MALLOC
          ->template NewObj<EventQueue>(CTP_MALLOC,
                                        EVENT_QUEUE_DEPTH_MULTIPLIER * depth)
          .ptr_;
  return event_queue_.exchange(fresh, std::memory_order_acq_rel);
}

size_t Worker::MigrateParkedTo(Worker *dst) {
  if (dst == nullptr || dst == this) {
    return 0;
  }
  // std::lock takes both without a fixed order, so two concurrent migrations
  // involving the same pair cannot deadlock. Neither lock is ever held across
  // ExecTask, so a wedged donor cannot block us here.
  std::lock(park_mtx_, dst->park_mtx_);
  std::lock_guard<std::mutex> lk_src(park_mtx_, std::adopt_lock);
  std::lock_guard<std::mutex> lk_dst(dst->park_mtx_, std::adopt_lock);

  size_t moved = 0;
  for (u32 i = 0; i < NUM_BLOCKED_QUEUES; ++i) {
    while (!blocked_queues_[i].empty()) {
      dst->blocked_queues_[i].push(blocked_queues_[i].front());
      blocked_queues_[i].pop();
      ++moved;
    }
  }
  for (u32 i = 0; i < NUM_PERIODIC_QUEUES; ++i) {
    while (!periodic_queues_[i].empty()) {
      clio::run::shared_ptr<Task> t = periodic_queues_[i].front();
      periodic_queues_[i].pop();
      // Reset the block timestamp. A periodic task stranded behind a wedged
      // worker has elapsed >> its period, so every migrated task would fire on
      // the destination's very first scan — a thundering herd proportional to
      // how long the stall lasted. Restarting the clock costs at most one
      // period of delay and keeps the cadence sane.
      if (!t.IsNull()) {
        t->BlockStart().Now();
      }
      dst->periodic_queues_[i].push(t);
      ++moved;
    }
  }
  while (!retry_queue_.empty()) {
    dst->retry_queue_.push(retry_queue_.front());
    retry_queue_.pop();
    ++moved;
  }
  return moved;
}

TaskLane *Worker::ReleaseLane() {
  TaskLane *old = assigned_lane_.exchange(nullptr, std::memory_order_acq_rel);
  return old;
}

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
  // All gpu2cpu task/future deserialization lives in IpcGpu2Cpu::RecvIn — the
  // worker never touches device task/future bytes.
#if CTP_ENABLE_CUDA || CTP_ENABLE_ROCM || CTP_ENABLE_SYCL
  return IpcGpu2Cpu::RecvIn(CLIO_IPC, gpu_lane, this);
#else
  (void)gpu_lane;
  return false;
#endif
}

u32 Worker::ProcessNewTasks(TaskLane *lane) {
  const u32 MAX_TASKS_PER_ITERATION = 16;
  u32 tasks_processed = 0;

  if (!lane) {
    return 0;
  }

  // issue #820, experimental (CLIO_BATCH_LANE=1): batch on the LANE.
  //
  // This is where same-blob tasks actually converge -- RouteTask sends every
  // task for a blob to that blob's container, i.e. to ONE worker's lane -- so
  // it is the only place a batch of same-blob work can form. Batching at the
  // SHM ingress cannot see it: the ingesting worker is usually not the
  // executing one, so it routes the task onward and never gets to merge it.
  //
  // Off by default because deferring lane tasks is not yet proven safe: the
  // lane also carries internal and re-routed work that other in-flight tasks
  // may be synchronously waiting on.
  static const bool lane_batch = [] {
    const char *e = std::getenv("CLIO_BATCH_LANE");
    return e != nullptr && !(std::string(e) == "0" || std::string(e) == "false");
  }();
  if (lane_batch && BatchingEnabled()) {
    // issue #820: size the batch to the CURRENT lane depth so a deep backlog
    // (e.g. a burst of same-blob writes that RouteTask funneled onto this one
    // lane) collapses in a SINGLE merged task instead of being chopped at a
    // fixed budget. Size() is a single-consumer snapshot -- this worker is the
    // only consumer of its lane (mpsc), so it can only UNDER-read when a
    // producer enqueues concurrently, and those late arrivals simply batch on
    // the next iteration; it never over-reads. Floor at the previous fixed
    // budget so a shallow lane never batches LESS than before, and the natural
    // ceiling is the ring's capacity (Size() cannot exceed it).
    const u32 depth = static_cast<u32>(lane->Size());
    const u32 batch_budget = std::max(depth, MAX_TASKS_PER_ITERATION * 4);
    return BatchLane(lane, batch_budget);
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

bool Worker::BatchingEnabled() {
  // issue #820. On by default; CLIO_TASK_BATCHING=0 restores the pre-batching
  // dequeue loop verbatim, which is the escape hatch for bisecting a regression
  // to this phase rather than to a container's policy.
  static const bool v = [] {
    if (const char *e = std::getenv("CLIO_TASK_BATCHING")) {
      return !(std::string(e) == "0" || std::string(e) == "false");
    }
    return true;
  }();
  return v;
}

bool Worker::ProcessNewTask(TaskLane *lane) {
  Future<Task> future;
  // Pop Future<Task> from lane
  if (!lane->Pop(future)) {
    return false;
  }


  SetCurrentTask(clio::run::shared_ptr<Task>());

  // The task carries its own identity now (the Future no longer owns a FutureShm).
  Task *new_task = future.get();
  if (new_task == nullptr) {
    HLOG(kError, "Worker {}: ProcessNewTask popped a null task", worker_id_);
    return true;
  }

  // Get pool_id and method_id from the task
  PoolId pool_id = new_task->pool_id_;
  u32 method_id = new_task->method_;
  HLOG(kDebug, "Worker {}: ProcessNewTask popped pool={} method={}",
       worker_id_, pool_id, method_id);

  // Get static container for task deserialization (stateless operation)
  auto *pool_manager = CLIO_POOL_MANAGER;
  auto container = pool_manager->GetStaticContainer(pool_id).get();

  // No local container for this pool: fail the task so the client doesn't hang.
  if (!container) {
    HLOG(kError, "Worker {}: cannot service pool_id={} method={}; "
         "container not found", worker_id_, pool_id, method_id);
    future.SetComplete();  // unblock the waiter on the error path
    return true;
  }

  // The Ipc call that enqueued this task onto the lane already deserialized /
  // copied it, so the Future's task pointer is already resolved — just read it.
  clio::run::shared_ptr<Task> task_full_ptr = future.GetTaskPtr();

  // Check if the task pointer is missing (enqueue bug / failed deserialize)
  if (task_full_ptr.IsNull()) {
    HLOG(kError,
         "Worker {}: Failed to deserialize task for pool_id={}, method={}",
         worker_id_, pool_id, method_id);
    // Mark as complete with error so client doesn't hang
    future.SetComplete();
    return true;
  }

  // Bind + route + execute (shared with the inline SHM-ingest path).
  RouteAndExec(future, lane);
  return true;
}

clio::run::shared_ptr<Task> Worker::RouteOnly(Future<Task> &future,
                                              TaskLane *lane) {
  clio::run::shared_ptr<Task> task_full_ptr = future.GetTaskPtr();
  if (task_full_ptr.IsNull()) {
    return clio::run::shared_ptr<Task>();
  }
  task_full_ptr->SetRunWorkerId(worker_id_);
  task_full_ptr->Lane() = lane;
  task_full_ptr->SetEventQueue(event_queue_.load(std::memory_order_acquire));
  task_full_ptr->RunFuture() = future;

  RouteResult route_result = CLIO_IPC->RouteTask(future);
  if (route_result != RouteResult::ExecHere) {
    // RouteTask already enqueued it wherever it belongs.
    return clio::run::shared_ptr<Task>();
  }
  return task_full_ptr;
}



namespace {
// issue #820/#822: process-global registry of "merged-task key -> parent tasks
// it must complete". Global (not per-worker) because a merged task is an I/O
// task whose continuation resumes on an arbitrary worker after the bdev put/get
// suspends, so the worker that COMPLETES it is often not the one that EMITTED
// it. g_batch_key_seq mints a process-unique key per merged task: it cannot use
// CreateTaskId().unique_ because that counter is thread-local, so two workers
// mint identical unique_ values and would clobber each other's parents in a
// shared map. Keys start at a high base so they never alias a normal
// per-thread task unique_ (defense in depth; the map is only keyed by these).
struct BatchPendingRegistry {
  std::mutex mu;
  std::unordered_map<clio::run::u64,
                     std::vector<clio::run::shared_ptr<Task>>> pending;
};
BatchPendingRegistry &GlobalBatchPending() {
  static BatchPendingRegistry reg;
  return reg;
}
std::atomic<clio::run::u64> g_batch_key_seq{clio::run::u64{1} << 48};

/** issue #820: hands a container's merged tasks back to the worker. */
class WorkerBatchSink : public BatchSink {
 public:
  using RunAsIs = std::function<void(const clio::run::shared_ptr<Task> &)>;

  WorkerBatchSink(RunAsIs run_as_is, RunAsIs run_merged)
      : run_as_is_(std::move(run_as_is)),
        run_merged_(std::move(run_merged)) {}

  void Passthrough(const clio::run::shared_ptr<Task> &task) override {
    if (task.IsNull()) {
      return;
    }
    // Unchanged: it keeps the id, future and RunContext it was routed with.
    run_as_is_(task);
  }

  void Emit(const clio::run::shared_ptr<Task> &merged,
            std::vector<clio::run::shared_ptr<Task>> &&parents) override {
    if (merged.IsNull()) {
      return;
    }
    // A fresh id and a Local query, so this task gets its OWN RunContext and
    // future rather than inheriting a member's.
    merged->task_id_ = CreateTaskId();
    // Stamp a PROCESS-GLOBAL key into unique_ so the (arbitrary) worker that
    // later completes this task finds its parents in the global registry.
    // CreateTaskId().unique_ is thread-local and collides across workers.
    merged->task_id_.unique_ =
        g_batch_key_seq.fetch_add(1, std::memory_order_relaxed);
    merged->pool_query_ = PoolQuery::Local();
    merged->SetFlags(TASK_BATCH_MERGED);
    if (!parents.empty()) {
      auto &reg = GlobalBatchPending();
      std::lock_guard<std::mutex> lk(reg.mu);
      reg.pending[merged->task_id_.unique_] = std::move(parents);
    }
    // Run it HERE rather than CLIO_IPC->Send().
    //
    // Send() self-sends, which routes with force_enqueue and pushes onto a
    // worker lane. Called from inside the drain loop that consumes that very
    // lane, under a deep burst the lane is full and the push waits for space
    // only this thread could make -- the producer==consumer deadlock
    // IpcCpu2Self::SendIn warns about in its own comment. Measured: Send never
    // returned, every thread asleep, nothing spinning.
    //
    // Running inline is also simply correct: every member routed to this
    // container on this worker, so the merged task belongs here by
    // construction and never needs to enter a lane at all.
    run_merged_(merged);
  }

 private:
  RunAsIs run_as_is_;
  RunAsIs run_merged_;
};
}  // namespace

u32 Worker::BatchLane(TaskLane *lane, u32 budget) {
  return BatchCollect(lane, budget, /*from_lane=*/true);
}

u32 Worker::BatchIngest(TaskLane *lane, u32 budget) {
  return BatchCollect(lane, budget, /*from_lane=*/false);
}

u32 Worker::BatchCollect(TaskLane *lane, u32 budget, bool from_lane) {
  // Bounded so batching can only ever amortize a backlog that already exists:
  // with an empty lane behind it a task forms a group of one and is emitted
  // unchanged, so the no-contention path pays nothing.
  const u32 kBatchDequeue = budget;
  u32 popped = 0;
  auto *pool_manager = CLIO_POOL_MANAGER;
  // Arrival order across the whole phase. Overlap resolution ("last writer
  // wins") is only meaningful against submission order, so the worker stamps it
  // here rather than letting a container infer it.
  static thread_local u64 batch_seq = 0;

  while (popped < kBatchDequeue) {
    Future<Task> future;
    if (from_lane) {
      if (!lane->Pop(future)) {
        break;  // lane empty
      }
    } else {
      future = IpcCpu2Cpu::RecvIn(CLIO_IPC, worker_id_);
      if (future.get() == nullptr) {
        break;  // shard empty
      }
    }
    ++popped;
    SetCurrentTask(clio::run::shared_ptr<Task>());
    Task *new_task = future.get();
    if (new_task == nullptr) {
      continue;
    }
    PoolId pool_id = new_task->pool_id_;
    u32 method_id = new_task->method_;
    auto container = pool_manager->GetStaticContainer(pool_id).get();
    if (!container) {
      HLOG(kError,
           "Worker {}: cannot service pool_id={} method={}; container not found",
           worker_id_, pool_id, method_id);
      future.SetComplete();
      continue;
    }
    clio::run::shared_ptr<Task> task_ptr = RouteOnly(future, lane);
    if (task_ptr.IsNull()) {
      continue;  // routed elsewhere, or undeserializable
    }
    // A task that has already STARTED is not a candidate: it is a coroutine
    // parked mid-execution and re-queued to resume (ContinueBlockedTasks
    // pushes those back onto the lane). It must resume, not be folded into a
    // fresh merged task -- doing that abandons its in-flight state and its
    // waiter never completes. Offer only never-started tasks.
    if (task_ptr->IsStarted()) {
      batch_passthrough_.push_back(task_ptr);
      continue;
    }
    // Offer it to the container's batching policy. The default declines, so a
    // container that has not opted in behaves exactly as before.
    if (container->BuildBatch(method_id, task_ptr, batch_groups_)) {
      for (auto &kv : batch_groups_) {
        if (!kv.second.empty() && kv.second.back().task.get() == task_ptr.get()) {
          kv.second.back().seq = batch_seq++;
          break;
        }
      }
      continue;
    }
    batch_passthrough_.push_back(task_ptr);
  }

  // Declined tasks run first, in arrival order: they keep the semantics they
  // had before batching existed, and a merged task then sees a settled world.
  for (auto &t : batch_passthrough_) {
    ExecTask(t, t->IsStarted());
  }
  batch_passthrough_.clear();

  // Then let each container collapse whatever it parked.
  if (!batch_groups_.empty()) {
    // Clear the current-task slot FIRST. IpcCpu2Self::SendIn parents a
    // self-sent task to whatever GetCurrentTask() returns, and by now that is
    // a stale, already-finished passthrough task from the loop above. A merged
    // task adopted by a dead parent has its completion routed into that
    // parent's coroutine (SendOut resumes the parent rather than signalling
    // normally), so nothing ever completes the merged task's own future or its
    // parked parents -- every worker then idles and the client waits forever.
    // That is the lost wakeup this phase was hanging on: no thread spinning, no
    // futex held, everything asleep.
    SetCurrentTask(clio::run::shared_ptr<Task>());
    WorkerBatchSink sink(
        [this](const clio::run::shared_ptr<Task> &t) {
          clio::run::shared_ptr<Task> copy = t;
          ExecTask(copy, copy->IsStarted());
        },
        [this, lane](const clio::run::shared_ptr<Task> &m) {
          // Give the merged task its OWN future + RunContext -- it has no
          // client waiter of its own; its job is to complete the parents it
          // subsumed -- then bind and run it on this worker.
          clio::run::shared_ptr<Task> t = m;
          Future<Task> f(t->pool_id_, t->method_, t);
          {
            auto fs = f.GetFutureShm();
            fs->origin_ = ClientOrigin::kClientShm;
          }
          t->BeginRunContext();
          t->SetRunWorkerId(worker_id_);
          t->Lane() = lane;
          t->SetEventQueue(event_queue_.load(std::memory_order_acquire));
          t->RunFuture() = f;
          ExecTask(t, /*is_started=*/false);
        });
    // Distinct pools present in the groups; each container picks out its own
    // keys and must leave the map empty.
    std::vector<PoolId> pools;
    for (auto &kv : batch_groups_) {
      bool seen = false;
      for (const auto &p : pools) {
        if (p == kv.first.pool_id) { seen = true; break; }
      }
      if (!seen) pools.push_back(kv.first.pool_id);
    }
    for (const auto &p : pools) {
      auto container = pool_manager->GetStaticContainer(p).get();
      if (container) {
        container->SmashBatch(batch_groups_, sink);
      }
    }
    // Anything a container left behind would otherwise never run: fail loudly
    // rather than silently dropping a client's task.
    if (!batch_groups_.empty()) {
      for (auto &kv : batch_groups_) {
        HLOG(kError,
             "Worker {}: SmashBatch left {} task(s) parked for pool={} method={}"
             " — completing them unbatched to avoid a hang",
             worker_id_, kv.second.size(), kv.first.pool_id, kv.first.method);
        for (auto &m : kv.second) {
          ExecTask(m.task, m.task->IsStarted());
        }
      }
      batch_groups_.clear();
    }
  }
  return popped;
}

void Worker::CompleteBatchParents(clio::run::shared_ptr<Task> &merged) {
  std::vector<clio::run::shared_ptr<Task>> parents;
  {
    auto &reg = GlobalBatchPending();
    std::lock_guard<std::mutex> lk(reg.mu);
    auto it = reg.pending.find(merged->task_id_.unique_);
    if (it == reg.pending.end()) {
      // A merged task with no registered parents (parents.empty() at Emit) is a
      // legitimate no-op; a genuine miss cannot happen now that the registry is
      // global and the key is process-unique.
      return;
    }
    parents = std::move(it->second);
    reg.pending.erase(it);
  }
  u32 rc = merged->GetReturnCode();
  ContainerId completer = merged->GetCompleter();
  DynamicContainer container = merged->ExecContainer();
  for (auto &parent : parents) {
    // No payload copy-back: a merged task is built so that each parent's own
    // output buffer is what the runtime already filled (that is what the
    // vectored task shape in part A is for). Only the status has to travel.
    parent->SetReturnCode(rc);
    parent->SetCompleter(completer);
    if (container.IsValid()) {
      parent->ExecContainer() = container;
    }
    EndTask(parent, false);
  }
}

void Worker::RouteAndExec(Future<Task> &future, TaskLane *lane) {
  clio::run::shared_ptr<Task> task_full_ptr = future.GetTaskPtr();
  if (task_full_ptr.IsNull()) {
    return;
  }
  // The RunContext (with its container resolved) was allocated by the ipc
  // receive/send site that introduced this task (Task::BeginRunContext). Bind it
  // to THIS worker/lane and record the future so subtask-completion events and
  // the eventual response go to the right place — this also covers tasks
  // re-enqueued across workers by RouteLocal.
  task_full_ptr->SetRunWorkerId(worker_id_);
  task_full_ptr->Lane() = lane;
  task_full_ptr->SetEventQueue(event_queue_.load(std::memory_order_acquire));
  task_full_ptr->RunFuture() = future;

  // Route task using consolidated routing function
  // RouteTask handles Retry/Dne internally via AddToRetryQueue
  RouteResult route_result = CLIO_IPC->RouteTask(future);
  if (route_result == RouteResult::ExecHere) {
#if CTP_IS_HOST
    // Re-check in case RouteTask changed the task's RunContext.
    bool is_started = task_full_ptr->IsStarted();
    ExecTask(task_full_ptr, is_started);
#endif
  }
}

u32 Worker::DrainMyShard() {
#if CTP_IS_HOST
  // issue #807: pop requests off this worker's inbound SHM shard and execute
  // each INLINE — bind, route, and (when it maps here, the common case for a
  // quick local task) run it right here, then the response is sent inline. No
  // lane push, no per-request AwakenWorker SIGUSR1: those were the bulk of the
  // latency gap vs the original inline design. A task that maps elsewhere is
  // enqueued by RouteTask as usual.
  // issue #820: THIS is where client tasks actually arrive. Under the #807 SHM
  // transport a client's PutBlob/GetBlob is delivered to its worker's shard
  // ring and executed inline here -- it never enters the lane. Batching wired
  // to the lane therefore never saw the hot path at all (measured: zero tasks
  // parked under an 8-writer same-blob load). So the batch phase hooks the
  // ingress, and the lane path is left exactly as it was.
  //
  // Deferring a LANE task is also unsafe in a way deferring an ingress task is
  // not: the lane carries internal and re-routed work that other in-flight
  // tasks may be synchronously waiting on, so holding one back can deadlock.
  // Freshly ingested client requests have no such dependents yet.
  // Raised 16 -> 256 (issue #862 experiment): with inline execution per
  // ingested task, a larger quantum amortizes the drain loop; fairness to the
  // lane is preserved because each ingested task still runs to completion (or
  // routes away) before the next pop.
  constexpr u32 kShardDrainBudget = 256;
  TaskLane *my_lane = assigned_lane_.load(std::memory_order_acquire);
  if (!BatchingEnabled()) {
    u32 n = 0;
    while (n < kShardDrainBudget) {
      Future<Task> f = IpcCpu2Cpu::RecvIn(CLIO_IPC, worker_id_);
      if (f.get() == nullptr) {
        break;
      }
      RouteAndExec(f, my_lane);
      ++n;
    }
    return n;
  }
  return BatchIngest(my_lane, kShardDrainBudget);
#else
  return 0;
#endif
}

double Worker::GetSuspendPeriod() const {
  // Scan all periodic queues to find the minimum yield_time (polling period)
  // We must wake up for the fastest periodic task to avoid starving it
  double min_yield_time_us = 0;
  bool found_task = false;

  // issue #785: the periodic queues are no longer worker-private — the monitor
  // thread splices them during a rescue — so this scan must hold park_mtx_.
  // TSan caught it racing MigrateParkedTo on the deque's internal iterators.
  std::lock_guard<std::mutex> lk(park_mtx_);

  // Check all periodic queues (0-3)
  for (u32 queue_idx = 0; queue_idx < NUM_PERIODIC_QUEUES; ++queue_idx) {
    const std::queue<clio::run::shared_ptr<Task>> &queue =
        periodic_queues_[queue_idx];

    if (queue.empty()) {
      continue;
    }

    // Check just the front task of each queue (representative of the queue's
    // period)
    const clio::run::shared_ptr<Task> &task = queue.front();
    if (task.IsNull()) {
      continue;
    }

    // Use the yield_time directly - this is the adaptive polling period
    if (!found_task || task->YieldTimeUs() < min_yield_time_us) {
      min_yield_time_us = task->YieldTimeUs();
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
  idle_iterations_.fetch_add(1, std::memory_order_relaxed);

  // Set idle start time on first idle iteration
  if (idle_iterations_.load(std::memory_order_relaxed) == 1) {
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
    // Still in busy-wait period. Yield the core before returning to the poll
    // loop. std::this_thread::yield only reschedules when another thread is
    // runnable, so on a well-provisioned host (spare cores) this is a no-op
    // and the low-latency busy-wait is preserved. When the runtime is
    // oversubscribed — more runnable workers + FUSE/ZMQ threads than cores, as
    // on CI's 2-core runners — a bare spin here monopolizes the core and
    // starves the worker that must run a blocked task's dependency (or the
    // FUSE thread that must submit it), livelocking the whole pipeline. That
    // is why the embedded-FUSE xfstests (generic/006/007/011/013/089/100/113/
    // 127/286/363/438/471) pass on a 16-core box but hang in CI. Yielding lets
    // the runnable thread get scheduled so forward progress resumes. (issue
    // #807: measured a bare tight-spin here at only ~1us over the yield on a
    // spare-core box — sched_yield is already ~free when a core is idle — so it
    // is not worth risking the oversubscription livelock the yield prevents.)
    CTP_THREAD_MODEL->Yield();
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
    // here is already visible to a plain Empty() load. Producers now
    // unconditionally send SIGUSR1 (see IpcManager::AwakenWorker), so we
    // don't need the active_ park-flag handshake — any push that lands
    // AFTER this recheck will signal us out of epoll_pwait2 regardless.
    // The handshake previously here tripped a lost-wakeup race at scale
    // (4n 256m, multi-tier bdev) where active_=true at the producer's
    // load suppressed the signal while the worker was already past its
    // post-store recheck and committed to epoll_pwait2.
    if (TaskLane *lane = assigned_lane_.load(std::memory_order_acquire)) {
      bool work_pending = !lane->Empty();
      EventQueue *eq_chk = event_queue_.load(std::memory_order_acquire);
      if (!work_pending) {
        std::lock_guard<std::mutex> lk(park_mtx_);
        for (EventQueue *aq : adopted_event_queues_) {
          if (aq && !aq->Empty()) {
            work_pending = true;
            break;
          }
        }
      }
      if (!work_pending && eq_chk) {
        work_pending = !eq_chk->Empty();
      }
      if (work_pending) {
        return;
      }
    }
    // issue #807: last-chance check of this worker's inbound SHM shard, so a
    // request that arrived before we park is not missed.
    if (!CLIO_IPC->ShardEmpty(worker_id_)) {
      return;
    }

    // Calculate timeout from periodic tasks, then cap it by max_sleep so a
    // worker NEVER sleeps indefinitely. GetSuspendPeriod() returns -1 when this
    // worker has no periodic task, which previously became an infinite epoll
    // wait: the worker then only wakes on the producer's SIGUSR1. That is a
    // cross-process lost-wakeup hang — an external SHM client pushes a task to a
    // shared lane and signals this worker out of epoll_pwait2, but if the signal
    // is ever missed (it races the worker's commit to epoll_pwait2, and on
    // macOS/boost configs the timing reliably loses) the worker sleeps forever
    // with a task pending (the cr_ipc_transport_shm / cr_client_retry_*_shm
    // timeouts). max_sleep (50 ms) is exactly the "maximum sleep" knob meant to
    // bound this; honor it so the worker self-heals by re-polling its lane.
    double suspend_period_us = GetSuspendPeriod();
    int timeout_us = (suspend_period_us < 0)
                         ? static_cast<int>(max_sleep)
                         : std::min(static_cast<int>(suspend_period_us),
                                    static_cast<int>(max_sleep));

    // issue #807: park protocol for the inbound SHM shard. The ORDER is
    // load-bearing and must match the transport's documented rendezvous:
    // publish "parked" BEFORE the final empty re-check, so it interlocks with a
    // producer that does "reserve slot (tail++) THEN check consumer_parked_".
    // The store(parked) and the load(tail, via ShardEmpty) form a Dekker
    // StoreLoad pair — if a request landed after our earlier fast-path check, we
    // now either observe its slot here (and skip the park) or it observes us
    // parked (and signals). Doing the re-check BEFORE setting parked (as an
    // earlier cut did) loses the wakeup: the producer pushes in the gap, sees
    // parked==0, skips the signal, and we park anyway — bounded only by the
    // 50ms max_sleep re-poll, which under the ARM weak-memory model fires
    // constantly and crawled cr_cli_cfs_parallel_stress into its timeout (x86
    // TSO mostly hid it).
    // Unified park handshake — this is what lets producers STOP signalling a
    // running worker (see IpcManager::AwakenWorker). Publish "parked" on BOTH
    // wakeup surfaces before the final re-check: the inbound SHM shard
    // (SetShardParked -> consumer_parked_, for client submits) AND this worker's
    // lane (SetActive(false), for intra-runtime lane pushes + completion
    // event-queue emplaces). The seq_cst fence between the publish and the
    // re-check makes it a Dekker StoreLoad: a producer that pushes THEN loads
    // the flag either sees us parked (and signals) or we see its task here (and
    // skip the park). A residual miss self-heals within max_sleep (<=50ms) — the
    // worker re-polls unconditionally — so gating can cost bounded latency but
    // never a lost-wakeup hang (which is why the earlier gated attempt, made
    // before the max_sleep cap existed, could hang and this one cannot).
    TaskLane *park_lane = assigned_lane_.load(std::memory_order_acquire);
    EventQueue *park_eq = event_queue_.load(std::memory_order_acquire);
    if (park_lane) park_lane->SetActive(false);
    CLIO_IPC->SetShardParked(worker_id_, true);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    bool late_work = false;
    if (park_lane && !park_lane->Empty()) late_work = true;
    if (!late_work && park_eq && !park_eq->Empty()) late_work = true;
    if (!late_work) {
      std::lock_guard<std::mutex> lk(park_mtx_);
      for (EventQueue *aq : adopted_event_queues_) {
        if (aq && !aq->Empty()) { late_work = true; break; }
      }
    }
    if (!late_work && !CLIO_IPC->ShardEmpty(worker_id_)) late_work = true;
    if (late_work) {
      if (park_lane) park_lane->SetActive(true);
      CLIO_IPC->SetShardParked(worker_id_, false);
      return;  // a task landed during park setup — go drain it
    }
    // Wait for signal using EventManager
    int nfds = event_manager_.Wait(timeout_us);
    if (park_lane) park_lane->SetActive(true);
    CLIO_IPC->SetShardParked(worker_id_, false);

    if (nfds == 0) {
      sleep_count_++;
    }

    // Force immediate rescan of all periodic tasks after waking
    ContinueBlockedTasks(true);
  }
}

u32 Worker::GetId() const { return worker_id_; }

bool Worker::IsRunning() const { return is_running_; }

void Worker::SetCurrentTask(const clio::run::shared_ptr<Task> &task) {
  current_task_ = task;
}

clio::run::shared_ptr<Task> &Worker::GetCurrentTask() { return current_task_; }

TaskLane *Worker::GetCurrentLane() const {
  if (current_task_.IsNull()) {
    return nullptr;
  }
  return current_task_->Lane();
}

void Worker::SetAsCurrentWorker() {
  CTP_THREAD_MODEL->SetTls(chi_cur_worker_key_,
                            static_cast<class Worker *>(this));
}

void Worker::ClearCurrentWorker() {
  CTP_THREAD_MODEL->SetTls(chi_cur_worker_key_,
                            static_cast<class Worker *>(nullptr));
}

void Worker::ExecTask(clio::run::shared_ptr<Task> &task_ptr, bool is_started) {
  // Non-periodic tasks always count as real work.
  // Periodic tasks must express work via the task's did_work_ flag.
  if (!task_ptr->IsPeriodic()) {
    SetTaskDidWork(true);
  }

  // Check if task is null or has no RunContext (not executing)
  if (task_ptr.IsNull()) {
    return;
  }

  // The execution container was resolved once at routing time and cached in
  // the task's RunContext (a DynamicContainer); read its current — i.e. most-
  // recently-upgraded — version for this execution slice.
  ContainerHold exec = task_ptr->ExecContainer().get();

  // Per-RPC access control. This is the owning host (the container is resolved
  // locally and the method is about to run) and the only place reached exactly
  // once per locally-executed task — including remote-forwarded ones, which
  // re-enter here on arrival with TASK_EXTERNAL_CLIENT preserved across the hop.
  // Reject an external user client's call to a private method before the
  // handler runs; deliver EACCES through the normal completion path. Internal
  // callers and public methods pass through. Only on first execution (resumes
  // of an already-admitted task carry no new authorization decision).
  if (!is_started && exec &&
      !exec->IsRpcAllowed(
          task_ptr->method_,
          task_ptr->task_flags_.Any(TASK_EXTERNAL_CLIENT))) {
    HLOG(kWarning,
         "Worker {}: denying external client call to private method {} on "
         "pool {}",
         worker_id_, task_ptr->method_, task_ptr->pool_id_);
    task_ptr->SetReturnCode(EACCES);
    EndTask(task_ptr, /*can_resched=*/false);
    return;
  }

  // issue #781: stamp the wall-clock entry time so the monitor thread can see a
  // worker that never returns from a non-yielding task. Set BEFORE driving the
  // coroutine; cleared after it returns/yields below. A cooperative task that
  // co_awaits returns here (clears), so it never looks stalled; a spinning task
  // never returns, so last_exec_start_us_ stays set and IsStalled() fires.
  last_exec_start_us_.store(
      static_cast<long long>(
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now().time_since_epoch())
              .count()),
      std::memory_order_relaxed);

  // Start CPU and wall timers before execution
  task_ptr->RunCpuTimer().Resume();
  task_ptr->RunWallTimer().Resume();

  // Call appropriate coroutine function based on task state. Driving the
  // coroutine is the Task's own responsibility (it owns its RunContext/frame).
  if (is_started) {
    task_ptr->ResumeCoroutine(task_ptr);
  } else {
    task_ptr->StartCoroutine(task_ptr);
    task_ptr->SetStarted();

    // Predict load for new tasks. predicted_stat_ is populated by
    // BeginTask via container->GetTaskStats(task), so derive the model
    // inferences from the already-cached stat instead of re-calling
    // GetTaskStats here.
    // issue #781: if RuntimeMapTask already predicted this task's cost at map
    // time (sched_reserved_us_ != 0), that value is AUTHORITATIVE — it was
    // computed from a fresh GetTaskStats, whereas PredictedStat() here can still
    // be 0 (BeginTask populates it just now). So only (re)compute PredictedLoad
    // when the task was NOT map-accounted.
    float reserved = task_ptr->SchedReservedUs();
    if (exec) {
      if (reserved == 0.0f) {
        task_ptr->SetPredictedLoad(
            exec->InferCpuTime(task_ptr->method_, task_ptr->PredictedStat()));
      }
      task_ptr->SetPredictedWallUs(exec->InferWallClockTime(
          task_ptr->method_, task_ptr->PredictedStat()));
    }
    // The task is now EXECUTING: count its predicted cost in load_ (whether the
    // value came from the map or from here), and release the queued reservation
    // so it is not double-counted. EndTask subtracts the same PredictedLoad.
    load_.store(load_.load(std::memory_order_relaxed) +
                    task_ptr->PredictedLoad(),
                std::memory_order_relaxed);
    if (reserved != 0.0f) {
      ReleaseReservation(reserved);
      task_ptr->SetSchedReservedUs(0);
    }
  }

  // Pause CPU and wall timers after execution
  task_ptr->RunCpuTimer().Pause();
  task_ptr->RunWallTimer().Pause();

  // issue #781: task returned/yielded — no longer stalling this worker.
  last_exec_start_us_.store(0, std::memory_order_relaxed);

  // For periodic tasks, only set task_did_work_ if the task reported
  // actual work done (e.g., received data, sent data). This prevents
  // idle polling from keeping the worker awake.
  if (task_ptr->IsPeriodic() && task_ptr->DidWork()) {
    SetTaskDidWork(true);
  }

  // Only set did_work_ if the task actually did work
  if (GetTaskDidWork()) {
    did_work_ = true;
  }

  // Check if coroutine is done or yielded. Use the RunContext completion flag
  // (issue #485) rather than dereferencing the possibly-freed coroutine frame
  // via coro_handle_.done().
  bool coro_done = task_ptr->IsCoroCompleted();

  // If coroutine yielded (not done and is_yielded_ set), don't clean up
  if (task_ptr->IsYielded() && !coro_done) {
    // yield_time_us_ > 0 means cooperative yield (polling) — add to periodic
    // queue so the worker re-checks after the requested delay.
    // yield_time_us_ == 0 means waiting for a Future event — the event queue
    // will resume it, so we must NOT add it to any queue here.
    if (task_ptr->YieldTimeUs() > 0) {
      AddToBlockedQueue(task_ptr);
    }
    // The worker is no longer running this coroutine; drop the current task so
    // between-task main-loop code (below) can't read it. See the note at the
    // end of this function.
    SetCurrentTask(clio::run::shared_ptr<Task>());
    return;
  }

  // End task execution and cleanup (handles periodic rescheduling internally)
  EndTask(task_ptr, true);

  // Clear the worker's current task now that it is no longer executing.
  // Main-loop code that runs *between* tasks — notably
  // BatchManager::FlushDue -> CreateTaskId -> Worker::GetCurrentTask — must not
  // observe a task whose RunContext has been freed. A completed remote task's
  // RunContext is destroyed on the network (ZMQ) thread, so leaving the current
  // task set here would let between-task code reach a freed RunContext (observed
  // as a heap-use-after-free in CreateTaskId during a ManyToOne flush). Clear
  // the handle so nothing dereferences it past this point.
  SetCurrentTask(clio::run::shared_ptr<Task>());
}


void Worker::EndTask(clio::run::shared_ptr<Task> &task_ptr, bool can_resched) {
  // Hold a reader on the container for the whole of EndTask: ReinforceCpuModel /
  // UpdateWork below touch it, and a migration must not swap it out mid-update.
  ContainerHold container = task_ptr->ExecContainer().get();
  if (container == nullptr) {
    HLOG(kError, "EndTask: container is null");
    return;
  }

  // Track completed tasks
  num_tasks_processed_.fetch_add(1, std::memory_order_relaxed);
  if (!task_ptr->IsPeriodic()) {
    // #785: real work, as opposed to a poller re-arming itself.
    num_nonperiodic_processed_.fetch_add(1, std::memory_order_relaxed);
  }

  // Subtract predicted load from worker
  load_.store(load_.load(std::memory_order_relaxed) -
                  task_ptr->PredictedLoad(),
              std::memory_order_relaxed);

  // issue #781: defensively release any still-held queued reservation for tasks
  // that reach EndTask WITHOUT executing (e.g. the early-error EACCES path), so a
  // reservation can never leak onto queued_load_. Normal tasks already released
  // it in ExecTask (sched_reserved_us_ == 0 here), so this is a no-op for them.
  float reserved = task_ptr->SchedReservedUs();
  if (reserved != 0.0f) {
    ReleaseReservation(reserved);
    task_ptr->SetSchedReservedUs(0);
  }

  // Reinforce model with actual CPU and wall time
  float actual_cpu_us = static_cast<float>(task_ptr->RunCpuTimer().GetUsec());
  float actual_wall_us = static_cast<float>(task_ptr->RunWallTimer().GetUsec());
  container->ReinforceCpuModel(
      task_ptr->method_, task_ptr->PredictedLoad(), actual_cpu_us,
      task_ptr->PredictedStat());
  container->ReinforceWallModel(
      task_ptr->method_, task_ptr->PredictedWallUs(), actual_wall_us,
      task_ptr->PredictedStat());
  // issue #781: fold the measured cost into the scheduler's perf-bin PDF so the
  // monitor thread can report the live workload distribution (telemetry).
  if (scheduler_ != nullptr) {
    scheduler_->RecordCompletion(task_ptr->method_, actual_cpu_us,
                                 actual_wall_us);
  }

  // Break the RunContext self-cycle for a task that is about to be released.
  // ProcessNewTask binds RunFuture (run_ctx_->future_) with a copied task_ptr_
  // that points back at this very task, so task -> RunContext -> future_ ->
  // task_ptr_ -> task is a reference cycle that pins the whole graph once every
  // external owner drops (one leaked task per RPC). RouteGlobal/RouteManyToOne
  // already break it for origin/batch tasks; net-received tasks reach the
  // worker via RouteLocal and are only broken here. Rebind as a NON-OWNING
  // handle rather than reset(): GetFutureShm() now resolves THROUGH TaskRaw(),
  // so a plain reset() would null the route and the in-flight response would
  // never complete (see fix #c3e05ab). The non-owning handle keeps TaskRaw()
  // valid while removing the ownership edge. NOT done on the periodic-reschedule
  // path: a rescheduled task stays live and re-executes against this future.
  auto break_self_cycle = [&]() {
    task_ptr->RunFuture().GetTaskPtr() =
        clio::run::shared_ptr<Task>::WrapNonOwning(task_ptr.get());
  };

  // ManyToOne aggregate task: this synthetic task has no external waiter. On
  // completion, broadcast its OUT to the batched originals (which completes
  // each of them), then delete the aggregate. Done after the model
  // reinforcement above so load accounting stays consistent.
  if (task_ptr->task_flags_.Any(TASK_BATCH_AGGREGATE)) {
    BatchManager *bm = CLIO_IPC->GetBatchManager();
    if (bm != nullptr) {
      bm->OnAggregateComplete(this, task_ptr);
    }
    container->UpdateWork(task_ptr, -1);
    task_ptr->ClearFlags(TASK_DATA_OWNER);
    // Task is freed via RAII when the RunContext's shared_ptr owners drop.
    break_self_cycle();
    return;
  }

  // issue #820: a merged task from SmashBatch. Like the aggregate above it has
  // no external waiter of its own — its job is to complete the PARENTS it
  // subsumed, each of which does have one. The parents' payloads were already
  // written by the merged task itself (a vectored put/get names each parent's
  // own buffer), so only status travels.
  if (task_ptr->task_flags_.Any(TASK_BATCH_MERGED)) {
    CompleteBatchParents(task_ptr);
    container->UpdateWork(task_ptr, -1);
    task_ptr->ClearFlags(TASK_DATA_OWNER);
    break_self_cycle();
    return;
  }

  // Get task properties at the start
  bool is_remote = task_ptr->IsRemote();
  bool is_periodic = task_ptr->IsPeriodic();

  // Handle periodic task rescheduling
  if (is_periodic && can_resched) {
    ReschedulePeriodicTask(task_ptr);
    return;
  }

  // Decrement work remaining for non-periodic tasks
  if (!is_periodic) {
    container->UpdateWork(task_ptr, -1);
  }

  // Fire-and-forget: skip all response paths. The task frees via RAII when the
  // RunContext's shared_ptr owners drop.
  if (task_ptr->task_flags_.Any(TASK_FIRE_AND_FORGET)) {
    task_ptr->ClearFlags(TASK_DATA_OWNER);
    break_self_cycle();
    return;
  }

  // If task is remote, enqueue to net_queue_ for SendOut. Choose the
  // latency vs I/O lane from the task's cached predicted_stat_ so a
  // small ACK / heartbeat reply doesn't queue behind a 1 MiB GetBlob
  // response on the wire.
  if (is_remote) {
    size_t io_size = task_ptr->PredictedStat().io_size_;
    NetQueuePriority prio = (io_size >= kNetQueueIoThreshold)
                                ? NetQueuePriority::kSendOutIO
                                : NetQueuePriority::kSendOutLatency;
    CLIO_IPC->EnqueueNetTask(task_ptr->RunFuture(), prio);
    // EnqueueNetTask copied an owning Future into the net queue (keeps the task
    // alive through SendOut), so this RunContext's back-reference can drop its
    // ownership now.
    break_self_cycle();
    return;
  }

  // Copy variables from future_shm to stack BEFORE any SetComplete() call
  // This prevents use-after-free since client may free future_shm after
  // SetComplete()
  auto future_shm = task_ptr->RunFuture().GetFutureShm();
  if (future_shm.IsNull()) {
    HLOG(kError, "EndTask: future_shm is NULL for pool={} method={}",
         task_ptr->pool_id_, task_ptr->method_);
    return;
  }
  // A top-level in-process task shares its Task object (and RunContext) with the
  // waiting client. In that case SendOut() below flips this task's completion
  // flag directly (IpcCpu2Self::SendOut -> SetComplete), which unblocks the
  // client's Wait(); the client then tears down the RunContext. break_self_cycle
  // mutates that same RunContext's RunFuture.task_ptr_, so running it AFTER the
  // signal races the just-unblocked client on that shared_ptr — a
  // use-after-free / double-release that corrupts the task allocator and shows
  // up as a SIGSEGV in _BuddyAllocator::AllocateOffset (#680; TSan flags
  // concurrent shared_ptr<Task>::Release from Worker::EndTask and the client's
  // ~RunContext). So for this case break the cycle BEFORE signaling. For a
  // subtask, SendOut enqueues an OWNING RunFuture copy onto the parent's event
  // queue, so the self-cycle must stay owning until after SendOut — break after.
  const bool signals_inprocess_client =
      future_shm->origin_ == ClientOrigin::kClientShm &&
      !task_ptr->task_flags_.Any(TASK_EXTERNAL_CLIENT) &&
      (task_ptr->GetParentTask().IsNull() ||
       task_ptr->GetParentTask()->EventQueue() == nullptr);
  if (signals_inprocess_client) {
    break_self_cycle();
    IpcCpu2Self::SendOut(task_ptr, shm_send_transport_.get());
  } else {
    IpcCpu2Self::SendOut(task_ptr, shm_send_transport_.get());
    // SendOut enqueued the Future copy onto the parent's event queue (subtask)
    // or shipped it to a remote/external client; the local back-reference can
    // now drop its ownership and let the finished task free by RAII.
    break_self_cycle();
  }
}

void Worker::ProcessBlockedQueue(std::queue<clio::run::shared_ptr<Task>> &queue,
                                 u32 queue_idx) {
  (void)queue_idx;  // Unused parameter, kept for API consistency

  // Process only first 8 tasks in the queue.
  // issue #785: size() reads the deque's internal iterators, which the monitor
  // thread mutates in MigrateParkedTo — TSan flagged this even though the pops
  // below are already guarded. It is only a batch hint, so a snapshot is fine.
  size_t queue_size;
  {
    std::lock_guard<std::mutex> lk(park_mtx_);
    queue_size = queue.size();
  }
  size_t check_limit = std::min(queue_size, size_t(8));

  for (size_t i = 0; i < check_limit; i++) {
    // issue #785: pop under park_mtx_, then RELEASE before ExecTask. Holding it
    // across execution would let a wedged worker block the monitor thread.
    clio::run::shared_ptr<Task> task;
    {
      std::lock_guard<std::mutex> lk(park_mtx_);
      if (queue.empty()) {
        break;
      }
      // The queue OWNS the task (shared_ptr), which keeps both the task and its
      // RunContext (owned by the task) alive while blocked.
      task = queue.front();
      queue.pop();
    }

    if (task.IsNull()) {
      continue;
    }

    // Determine if this is a resume (task was started before) or first
    // execution
    bool is_started = task->IsStarted();

    // Skip if task was started but coroutine already completed. This can happen
    // with orphan events from parallel subtasks. Uses the task's completion
    // query (which reads the completion flag, not coro_handle_.done(), to avoid
    // a use-after-free on a cross-thread-freed coroutine frame — issue #485).
    if (is_started && task->IsCoroCompleted()) {
      continue;
    }

    task->SetYieldCount(0);

    // CRITICAL: Clear the is_yielded_ flag before resuming the task
    // This allows the task to call Wait() again if needed
    task->SetYielded(false);

    // Execute task with existing RunContext
    ExecTask(task, is_started);

    // Don't re-add to queue
    continue;

    // Re-add to appropriate blocked queue based on current block count
    // AddToBlockedQueue will increment yield_count_ and determine the queue
    AddToBlockedQueue(task);
  }
}

void Worker::ProcessPeriodicQueue(std::queue<clio::run::shared_ptr<Task>> &queue,
                                  u32 queue_idx) {
  (void)queue_idx;  // Unused parameter, kept for API consistency

  // Check up to 8 tasks from the queue
  size_t check_limit = 8;
  size_t queue_size;
  {
    std::lock_guard<std::mutex> lk(park_mtx_);  // #785: see ProcessBlockedQueue
    queue_size = queue.size();
  }
  size_t actual_limit = std::min(queue_size, check_limit);

  // Capture SINGLE timestamp for ALL tasks processed in this batch
  // This prevents timestamp desynchronization between tasks with same
  // yield_time
  ctp::Timepoint batch_timestamp;
  batch_timestamp.Now();

  // Get current time for all checks
  for (size_t i = 0; i < actual_limit; i++) {
    // issue #785: same rule as ProcessBlockedQueue — pop under the lock,
    // execute outside it.
    clio::run::shared_ptr<Task> task;
    {
      std::lock_guard<std::mutex> lk(park_mtx_);
      if (queue.empty()) {
        break;
      }
      task = queue.front();
      queue.pop();
    }

    if (task.IsNull()) {
      continue;
    }

    // Check if the time threshold has been surpassed using batch timestamp
    // Add 2ms tolerance to account for timing variance and ms/us precision
    // mismatch
    double elapsed_us = task->BlockStart().GetUsecFromStart(batch_timestamp);
    if (elapsed_us + 2000.0 >= task->YieldTimeUs()) {
      // Time threshold reached (within tolerance) - execute the task
      bool is_started = task->IsStarted();

      // CRITICAL: Clear the is_yielded_ flag before resuming the task
      // This allows the task to call Wait() again if needed
      task->SetYielded(false);

      // For periodic tasks, unmark routed and route again
      task->SetRouted(false);

      // Use batch timestamp for rescheduling to prevent desynchronization
      // This ensures all tasks in this batch get the same block_start time
      task->BlockStart() = batch_timestamp;

      // Route task again - this will handle both local and distributed routing
      // RouteTask handles Retry/Dne internally via AddToRetryQueue
      if (CLIO_IPC->RouteTask(task->RunFuture()) == RouteResult::ExecHere) {
        ExecTask(task, is_started);

        // If task re-yielded with a polling interval, ExecTask already
        // re-added it to the periodic queue via AddToBlockedQueue.
      }
    } else {
      // Time threshold not reached yet - re-add to same queue
      std::lock_guard<std::mutex> lk(park_mtx_);
      queue.push(task);
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
  // issue #785: drain this worker's own queue AND every queue inherited from a
  // rescued worker. Snapshot the list under park_mtx_ and release before
  // popping, since ExecTask runs below.
  std::vector<EventQueue *> queues;
  {
    std::lock_guard<std::mutex> lk(park_mtx_);
    EventQueue *own = event_queue_.load(std::memory_order_acquire);
    if (own != nullptr) {
      queues.push_back(own);
    }
    queues.insert(queues.end(), adopted_event_queues_.begin(),
                  adopted_event_queues_.end());
  }
  for (EventQueue *eq : queues) {
  while (eq->Pop(future)) {
    HLOG(kDebug, "Worker {}: ProcessEventQueue popped subtask future",
         worker_id_);
    // Mark the subtask's future as complete
    future.Complete();

    // Get the parent task that is waiting for this subtask.
    // Safe to dereference because FUTURE_COMPLETE was not set until just now,
    // so the parent coroutine could not have seen completion, could not have
    // finished, and its RunContext has not been freed.
    clio::run::shared_ptr<Task> &parent = future.GetParentTask();
    if (parent.IsNull()) {
      continue;
    }

    // Skip if the parent's coroutine already completed. Uses the completion
    // query (the flag, not coro_handle_.done()) to avoid dereferencing a
    // coroutine frame a cross-thread completion may already have freed (#485).
    if (parent->IsCoroCompleted()) {
      continue;
    }

    // Resume the parent ONLY if this event's future is the one the parent is
    // suspended on (issue #705). SendIn registers the parent on EVERY subtask
    // future at submit, so with several subtasks in flight (e.g. ReadData's
    // per-block AsyncReads) an out-of-order completion would otherwise resume
    // the parent mid-await on a DIFFERENT subtask — the await returns while
    // the awaited handler is still running, and the caller reads its outputs
    // early (short reads/writes) or frees buffers under it (SEGFAULT).
    // Skipping is safe: FUTURE_COMPLETE was already set above, so when the
    // parent's own await reaches this future it completes without suspending,
    // and the future the parent IS waiting on delivers its own event.
    //
    // The match must be EXACT — including when awaited is null (issue #856).
    // Null means the parent is not suspended on any future right now: it is
    // either running or cooperatively yielded (a periodic task, e.g.
    // HeartbeatProbe, parked in the periodic queue between cycles). Resuming
    // it here executes the fiber while it is still queued, so the periodic
    // pop later resumes the same fiber a second time — two executions share
    // one fiber stack and its frame locals are freed twice (the recovery
    // leader's free(): invalid pointer abort in TriggerRecovery during
    // leader-election). Both await paths record the awaited future before
    // suspending, and a future with a null FutureShm matches null == null,
    // so exact equality cannot strand a legitimate waiter.
    const void* awaited = parent->AwaitedFshm();
    if (awaited != future.GetFutureShm().ptr_) {
      continue;
    }
    parent->SetAwaitedFshm(nullptr);

    // Reset the is_yielded_ flag before executing the task
    parent->SetYielded(false);

    // Reset is_notified_ so this task can be notified again for subsequent
    // co_await
    parent->SetNotified(false);

    // Execute the task
    ExecTask(parent, true);
  }
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

void Worker::AddToBlockedQueue(const clio::run::shared_ptr<Task> &task,
                               bool wait_for_task) {
  if (task.IsNull()) {
    return;
  }

  // If wait_for_task is true, do not add to blocked queue
  // The task is waiting for subtask completion and will be woken by event queue
  if (wait_for_task) {
    return;
  }

  // Check if task should go to blocked queue or periodic queue
  // Go to blocked queue if: block_time is 0 OR task is already started
  if (task->YieldTimeUs() == 0.0) {
    // Cooperative task waiting for subtasks - add to blocked queue
    // Increment block count for cooperative tasks
    task->SetYieldCount(task->YieldCount() + 1);

    // Determine which blocked queue based on block count:
    // Queue[0]: Tasks blocked <=2 times (checked every % 2 iterations)
    // Queue[1]: Tasks blocked <= 4 times (checked every % 4 iterations)
    // Queue[2]: Tasks blocked <= 8 times (checked every % 8 iterations)
    // Queue[3]: Tasks blocked > 8 times (checked every % 16 iterations)
    u32 queue_idx;
    if (task->YieldCount() <= 2) {
      queue_idx = 0;
    } else if (task->YieldCount() <= 4) {
      queue_idx = 1;
    } else if (task->YieldCount() <= 8) {
      queue_idx = 2;
    } else {
      queue_idx = 3;
    }

    // Add to the appropriate blocked queue. Store the task (owning shared_ptr)
    // so the task + its RunContext stay alive while blocked.
    std::lock_guard<std::mutex> lk(park_mtx_);  // #785
    blocked_queues_[queue_idx].push(task);
  } else {
    // Time-based periodic task - add to periodic queue
    // Record the time when task was blocked (if not already set recently)
    // Check if timestamp was set within last 10ms (indicates batch processing)
    double elapsed_since_block_us = task->BlockStart().GetUsecFromStart();
    if (elapsed_since_block_us > 10000.0 || elapsed_since_block_us < 0) {
      // Timestamp is stale or uninitialized - set it now
      task->BlockStart().Now();
    }
    // else: timestamp is fresh (< 10ms old), keep it to maintain
    // synchronization

    // Determine which periodic queue based on yield_time_us_:
    // Queue[0]: yield_time_us_ <= 50us
    // Queue[1]: yield_time_us_ <= 200us
    // Queue[2]: yield_time_us_ <= 50ms (50000us)
    // Queue[3]: yield_time_us_ > 50ms
    u32 queue_idx;
    if (task->YieldTimeUs() <= 50.0) {
      queue_idx = 0;
    } else if (task->YieldTimeUs() <= 200.0) {
      queue_idx = 1;
    } else if (task->YieldTimeUs() <= 50000.0) {
      queue_idx = 2;
    } else {
      queue_idx = 3;
    }

    // Add to the appropriate periodic queue (store the owning task handle).
    std::lock_guard<std::mutex> lk(park_mtx_);  // #785
    periodic_queues_[queue_idx].push(task);
  }
}

void Worker::AddToRetryQueue(const clio::run::shared_ptr<Task> &task) {
  std::lock_guard<std::mutex> lk(park_mtx_);  // #785
  retry_queue_.push(task);
}

void Worker::ProcessRetryQueue() {
  size_t count;
  {
    std::lock_guard<std::mutex> lk(park_mtx_);
    count = retry_queue_.size();
  }
  for (size_t i = 0; i < count; ++i) {
    clio::run::shared_ptr<Task> task_ptr;
    {
      std::lock_guard<std::mutex> lk(park_mtx_);  // #785
      if (retry_queue_.empty()) {
        break;
      }
      task_ptr = retry_queue_.front();
      retry_queue_.pop();
    }

    if (task_ptr.IsNull()) {
      continue;  // Skip invalid entries
    }

    // Clear routed so RouteTask re-evaluates.
    // RouteTask handles Retry/Dne internally via AddToRetryQueue.
    task_ptr->SetRouted(false);
    RouteResult result =
        CLIO_IPC->RouteTask(task_ptr->RunFuture(), /*force_enqueue=*/true);
    if (result == RouteResult::ExecHere) {
      // force_enqueue=true means this shouldn't happen, but handle it
      ExecTask(task_ptr, false);
    }
  }
}

void Worker::ReschedulePeriodicTask(clio::run::shared_ptr<Task> &task_ptr) {
  if (task_ptr.IsNull() || !task_ptr->IsPeriodic()) {
    return;
  }

  // Reset timers and predictions for next period
  task_ptr->RunCpuTimer().time_ns_ = 0;
  task_ptr->RunWallTimer().time_ns_ = 0;
  task_ptr->SetPredictedLoad(0);
  task_ptr->SetPredictedWallUs(0);
  task_ptr->PredictedStat() = TaskStat();

  // Get the lane from the run context.
  TaskLane *lane = task_ptr->Lane();
  if (!lane) {
    // issue #785: this used to return silently, DROPPING the periodic task —
    // it would simply stop running with no diagnostic anywhere. Fall back to
    // this worker's current lane so the task survives, and say so loudly if
    // even that is unavailable. A periodic task that vanishes is how a runtime
    // quietly stops polling.
    lane = assigned_lane_.load(std::memory_order_acquire);
    if (!lane) {
      HLOG(kError,
           "[#785] DROPPING periodic task (pool={} method={}): its RunContext "
           "has no lane and this worker has none either",
           task_ptr->pool_id_, task_ptr->method_);
      return;
    }
    task_ptr->Lane() = lane;
    HLOG(kWarning,
         "[#785] periodic task (pool={} method={}) had no lane; recovered onto "
         "worker {}'s lane",
         task_ptr->pool_id_, task_ptr->method_, worker_id_);
  }

  // Unset started when rescheduling periodic task
  task_ptr->SetStarted(false);

  // Adjust polling rate based on whether task did work
  if (scheduler_) {
    scheduler_->AdjustPolling(task_ptr);
  } else {
    // Fallback: use the true period if no scheduler available
    task_ptr->SetYieldTimeUs(task_ptr->period_ns_ / 1000.0);
  }

  // Reset did_work_ for the next execution
  task_ptr->SetDidWork(false);

  // Add to blocked queue - block count will be incremented automatically
  AddToBlockedQueue(task_ptr);
}

clio::run::shared_ptr<Task> &GetCurrentTask() {
  Worker *worker = CLIO_CUR_WORKER;
  if (worker) {
    return worker->GetCurrentTask();
  }
  // Non-worker thread: consult the per-thread fallback holder (set by tests /
  // direct Container::Run callers via clio::run::SetCurrentTask). The TLS slot
  // stores a pointer to a heap-allocated shared_ptr<Task> so GetCurrentTask can
  // hand back a stable reference. Lazily create it so a reader that runs before
  // any SetCurrentTask still gets a valid (null) handle.
  if (!chi_cur_runctx_key_created_) {
    CTP_THREAD_MODEL->CreateTls<clio::run::shared_ptr<Task>>(
        chi_cur_runctx_key_, nullptr);
    chi_cur_runctx_key_created_ = true;
  }
  auto *holder = CTP_THREAD_MODEL->GetTls<clio::run::shared_ptr<Task>>(
      chi_cur_runctx_key_);
  if (!holder) {
    holder = new clio::run::shared_ptr<Task>();
    CTP_THREAD_MODEL->SetTls(chi_cur_runctx_key_, holder);
  }
  return *holder;
}

void SetCurrentTask(const clio::run::shared_ptr<Task> &task) {
  // MUST mirror GetCurrentTask(): on a worker thread the current task lives in
  // worker->current_task_, so set THAT (not the TLS fallback). Otherwise
  // SetCurrentTask writes the TLS holder while GetCurrentTask reads the worker's
  // (never-updated, null) current_task_, and every handler's GetCurrentTask()
  // comes back null.
  Worker *worker = CLIO_CUR_WORKER;
  if (worker) {
    worker->SetCurrentTask(task);
    return;
  }
  // Non-worker thread: per-thread fallback holder (tests / direct
  // Container::Run callers).
  if (!chi_cur_runctx_key_created_) {
    CTP_THREAD_MODEL->CreateTls<clio::run::shared_ptr<Task>>(
        chi_cur_runctx_key_, nullptr);
    chi_cur_runctx_key_created_ = true;
  }
  auto *holder = CTP_THREAD_MODEL->GetTls<clio::run::shared_ptr<Task>>(
      chi_cur_runctx_key_);
  if (!holder) {
    holder = new clio::run::shared_ptr<Task>();
    CTP_THREAD_MODEL->SetTls(chi_cur_runctx_key_, holder);
  }
  *holder = task;
}

}  // namespace clio::run