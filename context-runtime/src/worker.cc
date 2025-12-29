/**
 * Worker implementation
 */

#include "chimaera/worker.h"

#include <signal.h>
#include <sys/epoll.h>
#include <sys/signalfd.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <boost/context/detail/fcontext.hpp>
#include <cstdlib>
#include <iostream>
#include <unordered_set>

// Include task_queue.h before other chimaera headers to ensure proper
// resolution
#include "chimaera/admin/admin_client.h"
#include "chimaera/container.h"
#include "chimaera/future.h"
#include "chimaera/pool_manager.h"
#include "chimaera/singletons.h"
#include "chimaera/task.h"
#include "chimaera/task_archives.h"
#include "chimaera/task_queue.h"
#include "chimaera/work_orchestrator.h"

namespace chi {

// Stack detection is now handled by WorkOrchestrator during initialization

Worker::Worker(u32 worker_id, ThreadType thread_type)
    : worker_id_(worker_id),
      thread_type_(thread_type),
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
      sleep_count_(0),
      epoll_fd_(-1) {
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

  // Allocate and initialize event queue from main allocator
  auto *alloc = CHI_IPC->GetMainAlloc();
  event_queue_ =
      alloc
          ->template NewObj<
              hipc::mpsc_ring_buffer<RunContext *, CHI_MAIN_ALLOC_T>>(
              alloc, EVENT_QUEUE_DEPTH)
          .ptr_;

  // Create epoll file descriptor for efficient worker suspension
  epoll_fd_ = epoll_create1(0);
  if (epoll_fd_ == -1) {
    HLOG(kError, "Worker {}: Failed to create epoll file descriptor",
         worker_id_);
    return false;
  }

  is_initialized_ = true;
  return true;
}

void Worker::SetTaskDidWork(bool did_work) { task_did_work_ = did_work; }

bool Worker::GetTaskDidWork() const { return task_did_work_; }

int Worker::GetEpollFd() const { return epoll_fd_; }

void Worker::Finalize() {
  if (!is_initialized_) {
    return;
  }

  Stop();

  // Clean up cached stacks and RunContexts
  while (!stack_cache_.empty()) {
    StackAndContext cached_entry = stack_cache_.front();
    stack_cache_.pop();

    // Free the cached stack
    if (cached_entry.stack_base_for_free) {
      free(cached_entry.stack_base_for_free);
    }

    // Free the cached RunContext
    if (cached_entry.run_ctx) {
      cached_entry.run_ctx->~RunContext();
      free(cached_entry.run_ctx);
    }
  }

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

  // Close epoll file descriptor
  if (epoll_fd_ != -1) {
    close(epoll_fd_);
    epoll_fd_ = -1;
  }

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

  // Set up signalfd and store in TaskLane
  // Get current thread ID
  pid_t tid = static_cast<pid_t>(syscall(SYS_gettid));
  assigned_lane_->SetTid(tid);

  // Create signal mask for custom user signal (SIGUSR1)
  sigset_t mask;
  sigemptyset(&mask);
  sigaddset(&mask, SIGUSR1);

  // Block the signal so it's handled by signalfd instead of default handler
  if (pthread_sigmask(SIG_BLOCK, &mask, nullptr) != 0) {
    HLOG(kError, "Worker {}: Failed to block SIGUSR1 signal", worker_id_);
    is_running_ = false;
    return;
  }

  // Create signalfd
  int signal_fd = signalfd(-1, &mask, SFD_NONBLOCK | SFD_CLOEXEC);
  if (signal_fd == -1) {
    HLOG(kError, "Worker {}: Failed to create signalfd", worker_id_);
    is_running_ = false;
    return;
  }

  // Store signal_fd in TaskLane
  assigned_lane_->SetSignalFd(signal_fd);

  // Add signal_fd to epoll
  struct epoll_event ev;
  ev.events = EPOLLIN;
  ev.data.fd = signal_fd;
  if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, signal_fd, &ev) == -1) {
    HLOG(kError, "Worker {}: Failed to add signal_fd to epoll", worker_id_);
    close(signal_fd);
    assigned_lane_->SetSignalFd(-1);
    is_running_ = false;
    return;
  }

  HLOG(kDebug, "Worker {}: Set up signalfd={} for tid={}", worker_id_,
       signal_fd, tid);

  // Main worker loop - process tasks from assigned lane
  while (is_running_) {
    did_work_ = false;  // Reset work tracker at start of each loop iteration

    // Process tasks from assigned lane
    ProcessNewTasks();

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
      //   if (sleep_count_ > 0) {
      //     HLOG(kInfo, "Worker {}: Woke up after {} sleeps", worker_id_,
      //           sleep_count_);
      //   }
      idle_iterations_ = 0;
      current_sleep_us_ = 0;
      sleep_count_ = 0;
    }
  }

  // Cleanup signalfd when worker exits
  int cleanup_signal_fd = assigned_lane_->GetSignalFd();
  if (cleanup_signal_fd != -1) {
    close(cleanup_signal_fd);
    assigned_lane_->SetSignalFd(-1);
  }
}

void Worker::Stop() { is_running_ = false; }

void Worker::SetLane(TaskLane *lane) {
  assigned_lane_ = lane;
  // Mark lane as active when assigned to worker
  assigned_lane_->SetActive(true);
}

TaskLane *Worker::GetLane() const { return assigned_lane_; }

u32 Worker::ProcessNewTasks() {
  // Process up to 16 tasks from this worker's lane per iteration
  const u32 MAX_TASKS_PER_ITERATION = 16;
  u32 tasks_processed = 0;

  while (tasks_processed < MAX_TASKS_PER_ITERATION) {
    Future<Task> future;
    // Pop Future<Task> from assigned lane
    if (assigned_lane_->Pop(future)) {
      tasks_processed++;
      SetCurrentRunContext(nullptr);

      // Fix the allocator pointer after popping from ring buffer
      auto *ipc_manager = CHI_IPC;
      future.SetAllocator(ipc_manager->GetMainAlloc());

      // Get pool_id and method_id from FutureShm
      auto &future_shm = future.GetFutureShm();
      PoolId pool_id = future_shm->pool_id_;
      u32 method_id = future_shm->method_id_;

      // Get container for routing
      auto *pool_manager = CHI_POOL_MANAGER;
      Container *container = pool_manager->GetContainer(pool_id);

      if (!container) {
        // Container not found - mark as complete with error
        future_shm->is_complete_.store(1);
        continue;
      }

      // Check if Future has null task pointer (indicates task needs to be
      // loaded)
      FullPtr<Task> task_full_ptr = future.GetTaskPtr();
      bool destroy_in_end_task = false;

      if (task_full_ptr.IsNull()) {
        // CLIENT PATH: Load task from serialized data in FutureShm
        std::vector<char> serialized_data(future_shm->serialized_task_.begin(),
                                          future_shm->serialized_task_.end());
        LocalLoadTaskArchive archive(serialized_data);
        task_full_ptr = container->LocalAllocLoadTask(method_id, archive);

        // Update the Future's task pointer
        future.GetTaskPtr() = task_full_ptr;

        destroy_in_end_task =
            true;  // Task was created from serialized data, should be destroyed
      } else {
        // RUNTIME PATH: Task pointer is already set, no need to load
        destroy_in_end_task = false;  // Task should not be destroyed
      }

      // Allocate stack and RunContext before routing
      if (!task_full_ptr->IsRouted()) {
        BeginTask(future, container, assigned_lane_, destroy_in_end_task);
      }

      // Route task using consolidated routing function
      if (RouteTask(future, assigned_lane_, container)) {
        // Routing successful, execute the task
        RunContext *run_ctx = task_full_ptr->run_ctx_;
        ExecTask(task_full_ptr, run_ctx, false);
      }
      // Note: RouteTask returning false doesn't always indicate an error
      // Real errors are handled within RouteTask itself
    } else {
      // No more tasks in this lane
      break;
    }
  }

  return tasks_processed;
}

void Worker::SuspendMe() {
  // No work was done in this iteration - increment idle counter
  idle_iterations_++;

  // If idle iterations is less than 32, don't sleep
  if (idle_iterations_ < 64) {
    return;
  }

  // Set idle start time on 32 idle iteration
  if (idle_iterations_ == 64) {
    idle_start_.Now();
  }

  // Get configuration parameters
  auto *config = CHI_CONFIG_MANAGER;
  u32 first_busy_wait = config->GetFirstBusyWait();
  u32 sleep_increment = config->GetSleepIncrement();
  u32 max_sleep = config->GetMaxSleep();

  // Calculate actual elapsed idle time using timer
  hshm::Timepoint current_time;
  current_time.Now();
  double elapsed_idle_us = idle_start_.GetUsecFromStart(current_time);

  if (elapsed_idle_us < first_busy_wait) {
    // Still in busy wait period - just yield CPU
    HSHM_THREAD_MODEL->Yield();
  } else {
    // Past busy wait period - start sleeping with linear increment
    // Calculate how many sleep increments have passed since busy wait ended
    current_sleep_us_ += sleep_increment;

    // Cap at maximum sleep
    if (current_sleep_us_ > max_sleep) {
      current_sleep_us_ = max_sleep;
    }

    // Before sleeping, check blocked queues with force=true
    // This will process both queues regardless of iteration count
    ContinueBlockedTasks(true);

    // If task_did_work_ is true, blocked tasks were found - don't sleep
    if (GetTaskDidWork()) {
      return;
    }

    // if (sleep_count_ == 0) {
    //   HLOG(kInfo, "Worker {}: Sleeping for {} us", worker_id_,
    //         current_sleep_us_);
    // }

    // Mark worker as inactive (blocked in epoll_wait)
    assigned_lane_->SetActive(false);

    // Wait for signal using epoll_wait with calculated timeout
    int timeout_ms = static_cast<int>(
        current_sleep_us_ / 1000);  // Convert microseconds to milliseconds
    int nfds =
        epoll_wait(epoll_fd_, epoll_events_, MAX_EPOLL_EVENTS, timeout_ms);

    // Mark worker as active again
    assigned_lane_->SetActive(true);

    if (nfds > 0) {
      // Events received - read and discard all signal info
      for (int i = 0; i < nfds; ++i) {
        struct signalfd_siginfo si;
        ssize_t bytes_read = read(epoll_events_[i].data.fd, &si, sizeof(si));
        (void)bytes_read;  // Suppress unused variable warning
      }
    } else if (nfds == 0) {
      // Timeout occurred - normal sleep expiration
      sleep_count_++;
    }
    // nfds < 0 means error, but we just continue anyway
  }
}

u32 Worker::GetId() const { return worker_id_; }

ThreadType Worker::GetThreadType() const { return thread_type_; }

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
  return run_ctx->task;
}

Container *Worker::GetCurrentContainer() const {
  RunContext *run_ctx = GetCurrentRunContext();
  if (!run_ctx) {
    return nullptr;
  }
  return run_ctx->container;
}

TaskLane *Worker::GetCurrentLane() const {
  RunContext *run_ctx = GetCurrentRunContext();
  if (!run_ctx) {
    return nullptr;
  }
  return run_ctx->lane;
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
                       Container *&container) {
  // Get task pointer from future
  FullPtr<Task> task_ptr = future.GetTaskPtr();

  if (task_ptr.IsNull()) {
    HLOG(kDebug, "Worker::RouteTask - task_ptr is null, returning false");
    return false;
  }

  // Check if task has already been routed - if so, return true immediately
  if (task_ptr->IsRouted()) {
    HLOG(kDebug, "Worker::RouteTask - task already routed, getting container");
    auto *pool_manager = CHI_POOL_MANAGER;
    container = pool_manager->GetContainer(task_ptr->pool_id_);
    HLOG(kDebug, "Worker::RouteTask - already routed, container={}",
         (void *)container);
    return (container != nullptr);
  }

  // Initialize exec_mode to kExec by default
  RunContext *run_ctx = task_ptr->run_ctx_;
  if (run_ctx != nullptr) {
    run_ctx->exec_mode = ExecMode::kExec;
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
    HLOG(kDebug, "Worker::RouteTask - calling RouteGlobal");
    RouteGlobal(future, pool_queries);
    HLOG(kDebug,
         "Worker::RouteTask - RouteGlobal done, returning false (no local "
         "exec)");
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
                        Container *&container) {
  // Get task pointer from future
  FullPtr<Task> task_ptr = future.GetTaskPtr();

  // Check task execution time estimate
  // Tasks with EstCpuTime >= 50us are considered slow
  size_t est_cpu_time = task_ptr->EstCpuTime();

  // Route slow tasks to kSlow workers if we're not already a slow worker
  //   if ((est_cpu_time >= 50 || task_ptr->stat_.io_size_ > 0) &&
  //       thread_type_ != kSlow) {
  //     // This is a slow task and we're a fast worker - route to slow workers
  //     auto *work_orchestrator = CHI_WORK_ORCHESTRATOR;
  //     work_orchestrator->AssignToWorkerType(kSlow, task_ptr);
  //     return false; // Task routed to slow workers, don't execute here
  //   }

  // Fast tasks (< 50us) stay on any worker, slow tasks stay on kSlow workers
  // Get the container for execution
  auto *pool_manager = CHI_POOL_MANAGER;
  container = pool_manager->GetContainer(task_ptr->pool_id_);
  if (!container) {
    return false;
  }

  // Set the completer_ field to track which container will execute this task
  task_ptr->SetCompleter(container->container_id_);

  auto *ipc_manager = CHI_IPC;
  u32 node_id = ipc_manager->GetNodeId();

  // Task is local and should be executed directly
  // Set TASK_ROUTED flag to indicate this task has been routed
  task_ptr->SetFlags(TASK_ROUTED);

  // Routing successful - caller should execute the task locally
  return true;
}

bool Worker::RouteGlobal(Future<Task> &future,
                         const std::vector<PoolQuery> &pool_queries) {
  // Get task pointer from future
  FullPtr<Task> task_ptr = future.GetTaskPtr();

  HLOG(kDebug,
       "Worker::RouteGlobal START - method={}, pool_id={}, num_queries={}",
       task_ptr->method_, task_ptr->pool_id_, pool_queries.size());

  for (size_t i = 0; i < pool_queries.size(); ++i) {
    HLOG(kDebug, "Worker::RouteGlobal - query[{}] routing_mode={}, node_id={}",
         i, static_cast<int>(pool_queries[i].GetRoutingMode()),
         pool_queries[i].GetNodeId());
  }

  auto *ipc_manager = CHI_IPC;

  // Store pool_queries in task's RunContext for SendIn to access
  RunContext *run_ctx = task_ptr->run_ctx_;
  if (run_ctx != nullptr) {
    run_ctx->pool_queries = pool_queries;
  }

  HLOG(kDebug, "Worker::RouteGlobal - enqueueing task directly to net_queue_");

  // Enqueue the original task directly to net_queue_ priority 0 (SendIn)
  ipc_manager->EnqueueNetTask(future, NetQueuePriority::kSendIn);

  HLOG(kDebug, "Worker::RouteGlobal - enqueued to net_queue_");

  // Set TASK_ROUTED flag on original task
  task_ptr->SetFlags(TASK_ROUTED);

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
  HLOG(kDebug, "Worker::ResolveDynamicQuery START - pool_id={}, method={}",
       pool_id, task_ptr->method_);

  // Use the current RunContext that was allocated by BeginTask
  RunContext *run_ctx = task_ptr->run_ctx_;
  if (run_ctx == nullptr) {
    HLOG(kDebug,
         "Worker::ResolveDynamicQuery - run_ctx is nullptr, returning empty");
    return {};  // Return empty vector if no RunContext
  }

  // Set execution mode to kDynamicSchedule
  // This tells ExecTask to call RerouteDynamicTask instead of EndTask
  run_ctx->exec_mode = ExecMode::kDynamicSchedule;
  HLOG(kDebug, "Worker::ResolveDynamicQuery - set exec_mode=kDynamicSchedule");

  // Return Local query for execution
  // After task completes, RerouteDynamicTask will re-route with updated
  // pool_query
  std::vector<PoolQuery> result;
  result.push_back(PoolQuery::Local());
  HLOG(kDebug, "Worker::ResolveDynamicQuery - returning Local query");
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
  RunContext *run_ctx = task_ptr->run_ctx_;
  if (run_ctx != nullptr) {
    run_ctx->exec_mode = ExecMode::kExec;
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
  HLOG(kDebug, "Worker::ResolveBroadcastQuery START - pool_id={}, method={}",
       pool_id, task_ptr->method_);

  auto *pool_manager = CHI_POOL_MANAGER;
  if (pool_manager == nullptr) {
    HLOG(kDebug,
         "Worker::ResolveBroadcastQuery - pool_manager is null, returning "
         "original query");
    return {query};  // Fallback to original query
  }

  // Get pool info to find the total number of containers
  const PoolInfo *pool_info = pool_manager->GetPoolInfo(pool_id);
  if (pool_info == nullptr || pool_info->num_containers_ == 0) {
    HLOG(kDebug,
         "Worker::ResolveBroadcastQuery - pool_info is null or 0 containers, "
         "returning original query");
    return {query};  // Fallback to original query
  }

  HLOG(
      kDebug,
      "Worker::ResolveBroadcastQuery - num_containers={}, creating Range query",
      pool_info->num_containers_);

  // Create a Range query that covers all containers, then resolve it
  PoolQuery range_query = PoolQuery::Range(0, pool_info->num_containers_);
  auto result = ResolveRangeQuery(range_query, pool_id, task_ptr);
  HLOG(kDebug,
       "Worker::ResolveBroadcastQuery - ResolveRangeQuery returned {} queries",
       result.size());
  return result;
}

std::vector<PoolQuery> Worker::ResolvePhysicalQuery(
    const PoolQuery &query, PoolId pool_id, const FullPtr<Task> &task_ptr) {
  // Physical routing - query is already resolved to a specific node
  return {query};
}

RunContext *Worker::AllocateStackAndContext(size_t size) {
  // Try to get from cache first
  if (!stack_cache_.empty()) {
    StackAndContext cached_entry = stack_cache_.front();
    stack_cache_.pop();

    if (cached_entry.run_ctx && cached_entry.stack_base_for_free) {
      RunContext *run_ctx = cached_entry.run_ctx;
      run_ctx->Clear();
      return run_ctx;
    }
  }

  // Normalize size to page-aligned
  const size_t page_size = 4096;
  size = ((size + page_size - 1) / page_size) * page_size;

  // Cache miss or size mismatch - allocate new stack and RunContext
  void *stack_base = nullptr;
  int ret = posix_memalign(&stack_base, page_size, size);
  RunContext *new_run_ctx = new RunContext();

  if (ret == 0 && stack_base && new_run_ctx) {
    // Store the malloc base pointer for freeing later
    new_run_ctx->stack_base_for_free = stack_base;
    new_run_ctx->stack_size = size;

    // Set the correct stack pointer based on stack growth direction from work
    // orchestrator
    WorkOrchestrator *orchestrator = CHI_WORK_ORCHESTRATOR;
    bool grows_downward = orchestrator ? orchestrator->IsStackDownward()
                                       : true;  // Default to downward

    if (grows_downward) {
      // Stack grows downward: point to aligned end of the malloc buffer
      // Ensure 16-byte alignment (required for x86-64 ABI)
      char *stack_top = static_cast<char *>(stack_base) + size;
      new_run_ctx->stack_ptr = reinterpret_cast<void *>(
          reinterpret_cast<uintptr_t>(stack_top) & ~static_cast<uintptr_t>(15));
    } else {
      // Stack grows upward: point to the beginning of the malloc buffer
      // Ensure 16-byte alignment
      new_run_ctx->stack_ptr = reinterpret_cast<void *>(
          (reinterpret_cast<uintptr_t>(stack_base) + 15) &
          ~static_cast<uintptr_t>(15));
    }

    return new_run_ctx;
  }

  // Cleanup on failure
  if (stack_base) free(stack_base);
  if (new_run_ctx) delete new_run_ctx;

  return nullptr;
}

void Worker::DeallocateStackAndContext(RunContext *run_ctx) {
  if (!run_ctx) {
    return;
  }

  // Add to cache for reuse instead of freeing
  // Create StackAndContext entry with the stack and RunContext
  StackAndContext cache_entry(run_ctx->stack_base_for_free, run_ctx->stack_size,
                              run_ctx);

  // Add to cache
  stack_cache_.push(cache_entry);
  // std::queue always succeeds in pushing (will grow dynamically)
}

void Worker::BeginTask(Future<Task> &future, Container *container,
                       TaskLane *lane, bool destroy_in_end_task) {
  FullPtr<Task> task_ptr = future.GetTaskPtr();
  if (task_ptr.IsNull()) {
    return;
  }

  // Allocate stack and RunContext together for new task
  RunContext *run_ctx = AllocateStackAndContext(65536);  // 64KB default

  if (!run_ctx) {
    // FATAL: Stack allocation failure - this is a critical error
    HLOG(kFatal,
         "Worker {}: Failed to allocate stack for task execution. Task "
         "method: {}, pool: {}",
         worker_id_, task_ptr->method_, task_ptr->pool_id_);
    std::abort();  // Fatal failure
  }

  // Initialize RunContext for new task
  run_ctx->thread_type = thread_type_;
  run_ctx->worker_id = worker_id_;
  run_ctx->task = task_ptr;        // Store task in RunContext
  run_ctx->is_yielded_ = false;    // Initially not blocked
  run_ctx->container = container;  // Store container for CHI_CUR_CONTAINER
  run_ctx->lane = lane;            // Store lane for CHI_CUR_LANE
  run_ctx->event_queue_ = event_queue_;  // Set pointer to worker's event queue
  run_ctx->destroy_in_end_task_ = destroy_in_end_task;  // Set destroy flag
  run_ctx->future_ = future;  // Store future in RunContext
  // Set RunContext pointer in task
  task_ptr->run_ctx_ = run_ctx;

  // Set current run context
  SetCurrentRunContext(run_ctx);
}

void Worker::BeginFiber(const FullPtr<Task> &task_ptr, RunContext *run_ctx,
                        void (*fiber_fn)(boost::context::detail::transfer_t)) {
  // Set current run context
  SetCurrentRunContext(run_ctx);

  // New task execution - increment work count for non-periodic tasks
  if (run_ctx->container && !task_ptr->IsPeriodic()) {
    // Increment work remaining in the container for non-periodic tasks
    run_ctx->container->UpdateWork(task_ptr, *run_ctx, 1);
  }

  // Create fiber context for this task using provided fiber function
  // stack_ptr is already correctly positioned based on stack growth direction
  bctx::fcontext_t fiber_fctx =
      bctx::make_fcontext(run_ctx->stack_ptr, run_ctx->stack_size, fiber_fn);

  // Jump to fiber context to execute the task
  bctx::transfer_t fiber_result = bctx::jump_fcontext(fiber_fctx, nullptr);

  // Update yield_context with current worker context so task can return here
  // The fiber_result contains the worker context for the task to use
  run_ctx->yield_context = fiber_result;

  // Update resume_context only if the task actually yielded (is_yielded_ =
  // true)
  if (run_ctx->is_yielded_) {
    run_ctx->resume_context = fiber_result;
  }
}

void Worker::ResumeFiber(const FullPtr<Task> &task_ptr, RunContext *run_ctx) {
  // Set current run context
  SetCurrentRunContext(run_ctx);

  // Resume execution - jump back to where the task yielded

  // Validate resume_context before jumping
  if (!run_ctx->resume_context.fctx) {
    HLOG(kFatal,
         "Worker {}: resume_context.fctx is null when resuming task. "
         "Stack: {} Size: {} Task method: {} Pool: {}",
         worker_id_, run_ctx->stack_ptr, run_ctx->stack_size, task_ptr->method_,
         task_ptr->pool_id_);
    std::abort();
  }

  // Check if stack pointer is still valid
  if (!run_ctx->stack_ptr || !run_ctx->stack_base_for_free) {
    HLOG(kFatal,
         "Worker {}: Stack context is invalid when resuming task. "
         "stack_ptr: {} stack_base: {} Task method: {} Pool: {}",
         worker_id_, run_ctx->stack_ptr, run_ctx->stack_base_for_free,
         task_ptr->method_, task_ptr->pool_id_);
    std::abort();
  }

  // Validate that resume_context.fctx points within the allocated stack range
  uintptr_t fctx_addr =
      reinterpret_cast<uintptr_t>(run_ctx->resume_context.fctx);
  uintptr_t stack_start =
      reinterpret_cast<uintptr_t>(run_ctx->stack_base_for_free);
  uintptr_t stack_end = stack_start + run_ctx->stack_size;

  if (fctx_addr < stack_start || fctx_addr > stack_end) {
    HLOG(kWarning,
         "Worker {}: resume_context.fctx ({:#x}) is outside stack range "
         "[{:#x}, {:#x}]. "
         "Task method: {} Pool: {}",
         worker_id_, fctx_addr, stack_start, stack_end, task_ptr->method_,
         task_ptr->pool_id_);
  }

  HLOG(kDebug,
       "Worker {}: Resuming task - fctx: {:#x}, stack: [{:#x}, {:#x}], "
       "method: {}",
       worker_id_, fctx_addr, stack_start, stack_end, task_ptr->method_);

  // Resume execution - jump back to task's yield point
  // Use temporary variables to avoid read/write conflict on resume_context
  bctx::fcontext_t resume_fctx = run_ctx->resume_context.fctx;
  void *resume_data = run_ctx->resume_context.data;

  // Jump to task's yield point and capture the result
  bctx::transfer_t resume_result =
      bctx::jump_fcontext(resume_fctx, resume_data);

  // Update yield_context with current worker context so task can return here
  run_ctx->yield_context = resume_result;

  // Update resume_context only if the task yielded again (is_yielded_ = true)
  if (run_ctx->is_yielded_) {
    run_ctx->resume_context = resume_result;
  }
}

void Worker::ExecTask(const FullPtr<Task> &task_ptr, RunContext *run_ctx,
                      bool is_started) {
  // Set task_did_work_ to true by default (tasks can override via
  // CHI_CUR_WORKER)
  // This comes before the null check since the task was scheduled
  SetTaskDidWork(true);

  // Check if task is null or run context is null
  if (task_ptr.IsNull() || !run_ctx) {
    HLOG(kDebug, "Worker::ExecTask - null task or run_ctx, returning");
    return;  // Consider null tasks as completed
  }

  // Call appropriate fiber function based on task state
  if (is_started) {
    ResumeFiber(task_ptr, run_ctx);
  } else {
    BeginFiber(task_ptr, run_ctx, FiberExecutionFunction);
    task_ptr->SetFlags(TASK_STARTED);
  }

  // Only set did_work_ if the task actually did work
  if (GetTaskDidWork() && run_ctx->exec_mode != ExecMode::kDynamicSchedule) {
    did_work_ = true;
  }

  // Common cleanup logic for both fiber and direct execution
  if (run_ctx->is_yielded_) {
    // Task is blocked - don't clean up, will be resumed later
    HLOG(kDebug, "Worker::ExecTask - task yielded, returning (blocked)");
    return;  // Task is not completed, blocked for later resume
  }

  // Check if this is a dynamic scheduling task
  if (run_ctx->exec_mode == ExecMode::kDynamicSchedule) {
    // Dynamic scheduling - re-route task with updated pool_query
    HLOG(kDebug, "Worker::ExecTask - calling RerouteDynamicTask");
    RerouteDynamicTask(task_ptr, run_ctx);
    return;
  }

  // End task execution and cleanup (handles periodic rescheduling internally)
  EndTask(task_ptr, run_ctx, true);
}

void Worker::EndTask(const FullPtr<Task> &task_ptr, RunContext *run_ctx,
                     bool can_resched) {
  // Get task properties at the start
  bool is_remote = task_ptr->IsRemote();
  bool is_periodic = task_ptr->IsPeriodic();

  HLOG(kInfo,
       "[TRACE] EndTask - task_id={}, pool_id={}, method={}, is_remote={}, "
       "is_periodic={}, can_resched={}",
       task_ptr->task_id_, task_ptr->pool_id_, task_ptr->method_, is_remote,
       is_periodic, can_resched);

  // Handle periodic task rescheduling
  if (is_periodic && can_resched) {
    ReschedulePeriodicTask(run_ctx, task_ptr);
    return;
  }

  // Decrement work remaining for non-periodic tasks
  if (!is_periodic && run_ctx->container != nullptr) {
    run_ctx->container->UpdateWork(task_ptr, *run_ctx, -1);
  }

  // If task is remote, enqueue to net_queue_ for SendOut and return immediately
  if (is_remote) {
    auto *ipc_manager = CHI_IPC;
    ipc_manager->EnqueueNetTask(run_ctx->future_, NetQueuePriority::kSendOut);
    HLOG(kDebug,
         "[TRACE] EndTask - enqueued remote task {} to net_queue_ for SendOut",
         task_ptr->task_id_);
    return;
  }

  // Local task completion using Future
  // 1. Serialize outputs using container->LocalSaveTask (only if task will be
  // destroyed)
  if (run_ctx->destroy_in_end_task_) {
    LocalSaveTaskArchive archive(LocalMsgType::kSerializeOut);
    if (run_ctx->container != nullptr) {
      run_ctx->container->LocalSaveTask(task_ptr->method_, archive, task_ptr);
    }

    // Copy serialized outputs to FutureShm
    const std::vector<char> &serialized = archive.GetData();
    auto &future_shm = run_ctx->future_.GetFutureShm();
    future_shm->serialized_task_.resize(serialized.size());
    std::memcpy(future_shm->serialized_task_.data(), serialized.data(),
                serialized.size());
  }

  // 2. Mark task as complete
  HLOG(kInfo, "[TRACE] EndTask - calling SetComplete for task_id={}",
       task_ptr->task_id_);
  run_ctx->future_.SetComplete();
  HLOG(kInfo, "[TRACE] EndTask - SetComplete done for task_id={}",
       task_ptr->task_id_);

  // 2.5. Wake up parent task if waiting for this subtask
  RunContext *parent_task = run_ctx->future_.GetParentTask();
  if (parent_task != nullptr && parent_task->event_queue_ != nullptr) {
    auto *parent_event_queue = reinterpret_cast<
        hipc::mpsc_ring_buffer<RunContext *, CHI_MAIN_ALLOC_T> *>(
        parent_task->event_queue_);
    parent_event_queue->Emplace(parent_task);
  }

  // 3. Delete task using container->DelTask (only if destroy_in_end_task is
  // true)
  if (run_ctx->destroy_in_end_task_) {
    run_ctx->container->DelTask(task_ptr->method_, task_ptr);
  }

  // Deallocate stack and context
  DeallocateStackAndContext(run_ctx);
}

void Worker::RerouteDynamicTask(const FullPtr<Task> &task_ptr,
                                RunContext *run_ctx) {
  HLOG(kDebug,
       "Worker::RerouteDynamicTask START - method={}, pool_id={}, "
       "new_routing_mode={}",
       task_ptr->method_, task_ptr->pool_id_,
       static_cast<int>(task_ptr->pool_query_.GetRoutingMode()));

  // Dynamic scheduling complete - now re-route task with updated pool_query
  // The task's pool_query_ should have been updated during execution
  // (e.g., from Dynamic to Local or Broadcast)

  Container *container = run_ctx->container;
  TaskLane *lane = run_ctx->lane;

  // Reset the TASK_STARTED flag so the task can be executed again
  task_ptr->ClearFlags(TASK_STARTED | TASK_ROUTED);
  HLOG(kDebug,
       "Worker::RerouteDynamicTask - cleared TASK_STARTED|TASK_ROUTED flags");

  // Re-route the task using the updated pool_query
  HLOG(kDebug, "Worker::RerouteDynamicTask - calling RouteTask");
  if (RouteTask(run_ctx->future_, lane, container)) {
    HLOG(kDebug,
         "Worker::RerouteDynamicTask - RouteTask returned true, exec_mode={}",
         static_cast<int>(run_ctx->exec_mode));
    // Avoids recursive call to RerouteDynamicTask
    if (run_ctx->exec_mode == ExecMode::kDynamicSchedule) {
      HLOG(kDebug,
           "Worker::RerouteDynamicTask - still kDynamicSchedule, calling "
           "EndTask");
      EndTask(task_ptr, run_ctx, false);
      return;
    }
    // Successfully re-routed - execute the task again
    // Note: ExecTask will call BeginFiber since TASK_STARTED is unset
    HLOG(kDebug, "Worker::RerouteDynamicTask - calling ExecTask");
    ExecTask(task_ptr, run_ctx, false);
  } else {
    HLOG(kDebug,
         "Worker::RerouteDynamicTask - RouteTask returned false (routed "
         "globally)");
  }
}

void Worker::ProcessBlockedQueue(std::queue<RunContext *> &queue,
                                 u32 queue_idx) {
  // Process only first 8 tasks in the queue
  size_t queue_size = queue.size();
  size_t check_limit = std::min(queue_size, size_t(8));

  if (queue_size > 0) {
    HLOG(kInfo,
         "[TRACE] ProcessBlockedQueue - queue_idx={}, queue_size={}, "
         "check_limit={}",
         queue_idx, queue_size, check_limit);
  }

  for (size_t i = 0; i < check_limit; i++) {
    if (queue.empty()) {
      break;
    }

    RunContext *run_ctx = queue.front();
    queue.pop();

    if (!run_ctx || run_ctx->task.IsNull()) {
      // Invalid entry, don't re-add
      continue;
    }

    HLOG(kInfo,
         "[TRACE] ProcessBlockedQueue - processing task_id={}, pool_id={}, "
         "method={}",
         run_ctx->task->task_id_, run_ctx->task->pool_id_,
         run_ctx->task->method_);

    // Always execute tasks from blocked queue
    // (Event queue will handle subtask completion wakeup)
    // Determine if this is a resume (task was started before) or first
    // execution
    bool is_started = run_ctx->task->task_flags_.Any(TASK_STARTED);

    run_ctx->yield_count_ = 0;

    // CRITICAL: Clear the is_yielded_ flag before resuming the task
    // This allows the task to call Wait() again if needed
    run_ctx->is_yielded_ = false;

    HLOG(kInfo, "[TRACE] ProcessBlockedQueue - calling ExecTask, is_started={}",
         is_started);

    // Execute task with existing RunContext
    ExecTask(run_ctx->task, run_ctx, is_started);

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

  // Get current time for all checks
  for (size_t i = 0; i < actual_limit; i++) {
    if (queue.empty()) {
      break;
    }

    RunContext *run_ctx = queue.front();
    queue.pop();

    if (!run_ctx || run_ctx->task.IsNull()) {
      // Invalid entry, don't re-add
      continue;
    }

    // Check if the time threshold has been surpassed
    if (run_ctx->block_start.GetUsecFromStart() >= run_ctx->yield_time_us_) {
      // Time threshold reached - execute the task
      bool is_started = run_ctx->task->task_flags_.Any(TASK_STARTED);

      // CRITICAL: Clear the is_yielded_ flag before resuming the task
      // This allows the task to call Wait() again if needed
      run_ctx->is_yielded_ = false;

      // For periodic tasks, unmark TASK_ROUTED and route again
      run_ctx->task->ClearFlags(TASK_ROUTED);
      Container *container = run_ctx->container;

      // Route task again - this will handle both local and distributed routing
      if (RouteTask(run_ctx->future_, run_ctx->lane, container)) {
        // Routing successful, execute the task
        ExecTask(run_ctx->task, run_ctx, is_started);
      }
    } else {
      // Time threshold not reached yet - re-add to same queue
      queue.push(run_ctx);
    }
  }
}

void Worker::ProcessEventQueue() {
  // Process all tasks in the event queue
  RunContext *run_ctx;
  while (event_queue_->Pop(run_ctx)) {
    if (!run_ctx || run_ctx->task.IsNull()) {
      continue;
    }

    // Reset the is_yielded_ flag before executing the task
    run_ctx->is_yielded_ = false;

    // Execute the task
    ExecTask(run_ctx->task, run_ctx, true);
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
    // periodic_queues_[0] (<=50us) every 16 iterations
    if (iteration_count_ % 16 == 0) {
      ProcessPeriodicQueue(periodic_queues_[0], 0);
    }

    // periodic_queues_[1] (<=200us) every 32 iterations
    if (iteration_count_ % 32 == 0) {
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
  if (!run_ctx || run_ctx->task.IsNull()) {
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
    // Record the time when task was blocked
    run_ctx->block_start.Now();

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
  TaskLane *lane = run_ctx->lane;
  if (!lane) {
    // No lane information, cannot reschedule
    return;
  }

  // Unset TASK_STARTED flag when rescheduling periodic task
  task_ptr->ClearFlags(TASK_STARTED);

  // Add to blocked queue - block count will be incremented automatically
  run_ctx->yield_time_us_ = task_ptr->period_ns_ / 1000.0;
  AddToBlockedQueue(run_ctx);
}

void Worker::FiberExecutionFunction(boost::context::detail::transfer_t t) {
  // This function runs in the fiber context
  // Use thread-local storage to get context
  Worker *worker = CHI_CUR_WORKER;
  RunContext *run_ctx = worker->GetCurrentRunContext();
  FullPtr<Task> task_ptr =
      worker ? worker->GetCurrentTask() : FullPtr<Task>::GetNull();

  if (!task_ptr.IsNull() && worker && run_ctx) {
    // Store the worker's context (from parameter t) - this is where we jump
    // back when yielding or when task completes
    run_ctx->yield_context = t;
    // Execute the task directly - merged TaskExecutionFunction logic
    try {
      // Get the container from RunContext
      Container *container = run_ctx->container;

      if (container) {
        // Call the container's Run function with the task
        container->Run(task_ptr->method_, task_ptr, *run_ctx);
      } else {
        // Container not found - this is an error condition
        HLOG(kWarning, "Container not found in RunContext for pool_id: {}",
             task_ptr->pool_id_);
      }
    } catch (const std::exception &e) {
      // Handle execution errors
      HLOG(kError, "Task execution failed: {}", e.what());
    } catch (...) {
      // Handle unknown errors
      HLOG(kError, "Task execution failed with unknown exception");
    }

    // Task completion and work count handling is done in ExecTask
    // This avoids duplicate logic and ensures proper ordering
  }

  // Jump back to worker context when task completes
  // Use temporary variables to avoid potential read/write conflicts
  bctx::fcontext_t worker_fctx = run_ctx->yield_context.fctx;
  void *worker_data = run_ctx->yield_context.data;
  bctx::jump_fcontext(worker_fctx, worker_data);
}

}  // namespace chi