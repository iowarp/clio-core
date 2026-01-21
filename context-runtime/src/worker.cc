/**
 * Worker implementation
 *
 * Uses C++20 stackless coroutines for task suspension and resumption.
 * Coroutines are managed via std::coroutine_handle stored in RunContext.
 */

#include "chimaera/worker.h"

#include <signal.h>
#include <sys/epoll.h>
#include <sys/signalfd.h>
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

bool Worker::RegisterEpollFd(int fd, u32 events, void *user_data) {
  if (epoll_fd_ == -1 || fd < 0) {
    return false;
  }

  struct epoll_event ev;
  ev.events = events;
  ev.data.ptr = user_data;

  // Lock to protect epoll_ctl from concurrent access
  hshm::ScopedMutex lock(epoll_mutex_, 0);
  if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, fd, &ev) == -1) {
    HLOG(kWarning, "Failed to register fd {} with worker {} epoll: {}", fd,
         worker_id_, strerror(errno));
    return false;
  }
  return true;
}

bool Worker::UnregisterEpollFd(int fd) {
  if (epoll_fd_ == -1 || fd < 0) {
    return false;
  }

  // Lock to protect epoll_ctl from concurrent access
  hshm::ScopedMutex lock(epoll_mutex_, 0);
  if (epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, fd, nullptr) == -1) {
    HLOG(kWarning, "Failed to unregister fd {} from worker {} epoll: {}", fd,
         worker_id_, strerror(errno));
    return false;
  }
  return true;
}

bool Worker::ModifyEpollFd(int fd, u32 events, void *user_data) {
  if (epoll_fd_ == -1 || fd < 0) {
    return false;
  }

  struct epoll_event ev;
  ev.events = events;
  ev.data.ptr = user_data;

  // Lock to protect epoll_ctl from concurrent access
  hshm::ScopedMutex lock(epoll_mutex_, 0);
  if (epoll_ctl(epoll_fd_, EPOLL_CTL_MOD, fd, &ev) == -1) {
    HLOG(kWarning, "Failed to modify fd {} in worker {} epoll: {}", fd,
         worker_id_, strerror(errno));
    return false;
  }
  return true;
}

void Worker::Finalize() {
  if (!is_initialized_) {
    return;
  }

  Stop();

  // Clean up cached RunContexts
  while (!context_cache_.empty()) {
    CachedContext cached_entry = context_cache_.front();
    context_cache_.pop();

    // Free the cached RunContext
    if (cached_entry.run_ctx) {
      delete cached_entry.run_ctx;
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
    return false;
  }

  // Check if task has already been routed - if so, return true immediately
  if (task_ptr->IsRouted()) {
    auto *pool_manager = CHI_POOL_MANAGER;
    container = pool_manager->GetContainer(task_ptr->pool_id_);
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

  auto *ipc_manager = CHI_IPC;

  // Store pool_queries in task's RunContext for SendIn to access
  RunContext *run_ctx = task_ptr->run_ctx_;
  if (run_ctx != nullptr) {
    run_ctx->pool_queries = pool_queries;
  }

  // Enqueue the original task directly to net_queue_ priority 0 (SendIn)
  ipc_manager->EnqueueNetTask(future, NetQueuePriority::kSendIn);

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
  // Use the current RunContext that was allocated by BeginTask
  RunContext *run_ctx = task_ptr->run_ctx_;
  if (run_ctx == nullptr) {
    return {};  // Return empty vector if no RunContext
  }

  // Set execution mode to kDynamicSchedule
  // This tells ExecTask to call RerouteDynamicTask instead of EndTask
  run_ctx->exec_mode = ExecMode::kDynamicSchedule;

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

RunContext *Worker::AllocateContext() {
  // Try to get from cache first
  if (!context_cache_.empty()) {
    CachedContext cached_entry = context_cache_.front();
    context_cache_.pop();

    if (cached_entry.run_ctx) {
      RunContext *run_ctx = cached_entry.run_ctx;
      run_ctx->Clear();
      return run_ctx;
    }
  }

  // Cache miss - allocate new RunContext
  // With C++20 stackless coroutines, no stack allocation is needed
  RunContext *new_run_ctx = new RunContext();
  return new_run_ctx;
}

void Worker::DeallocateContext(RunContext *run_ctx) {
  if (!run_ctx) {
    return;
  }

  // Destroy coroutine handle if it exists
  if (run_ctx->coro_handle_) {
    run_ctx->coro_handle_.destroy();
    run_ctx->coro_handle_ = nullptr;
  }

  // Add to cache for reuse instead of freeing
  CachedContext cache_entry(run_ctx);
  context_cache_.push(cache_entry);
}

void Worker::BeginTask(Future<Task> &future, Container *container,
                       TaskLane *lane, bool destroy_in_end_task) {
  FullPtr<Task> task_ptr = future.GetTaskPtr();
  if (task_ptr.IsNull()) {
    return;
  }

  // Allocate RunContext for new task
  // With C++20 stackless coroutines, no stack allocation is needed
  RunContext *run_ctx = AllocateContext();

  if (!run_ctx) {
    // FATAL: Context allocation failure - this is a critical error
    HLOG(kFatal,
         "Worker {}: Failed to allocate context for task execution. Task "
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
  run_ctx->future_ = future;        // Store future in RunContext
  run_ctx->coro_handle_ = nullptr;  // Coroutine not started yet
  // Set RunContext pointer in task
  task_ptr->run_ctx_ = run_ctx;

  // Set current run context
  SetCurrentRunContext(run_ctx);
}

void Worker::StartCoroutine(const FullPtr<Task> &task_ptr,
                            RunContext *run_ctx) {
  // Set current run context
  SetCurrentRunContext(run_ctx);

  // New task execution - increment work count for non-periodic tasks
  if (run_ctx->container && !task_ptr->IsPeriodic()) {
    // Increment work remaining in the container for non-periodic tasks
    run_ctx->container->UpdateWork(task_ptr, *run_ctx, 1);
  }

  // Get the container from RunContext
  Container *container = run_ctx->container;
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
  // Set task_did_work_ to true by default (tasks can override via
  // CHI_CUR_WORKER)
  // This comes before the null check since the task was scheduled
  SetTaskDidWork(true);

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

  // Only set did_work_ if the task actually did work
  if (GetTaskDidWork() && run_ctx->exec_mode != ExecMode::kDynamicSchedule) {
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
  if (run_ctx->exec_mode == ExecMode::kDynamicSchedule) {
    // Dynamic scheduling - re-route task with updated pool_query
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
  run_ctx->future_.SetComplete();

  // 2.5. Wake up parent task if waiting for this subtask
  // Only wake parent if:
  // 1. Parent exists and has valid event queue and coroutine handle
  // 2. Parent hasn't already been notified by another subtask
  //    (prevents duplicate event queue additions causing SIGILL)
  RunContext *parent_task = run_ctx->future_.GetParentTask();
  if (parent_task != nullptr && parent_task->event_queue_ != nullptr &&
      parent_task->coro_handle_ && !parent_task->coro_handle_.done()) {
    // Use atomic compare_exchange to ensure only one subtask notifies the
    // parent
    bool expected = false;
    if (parent_task->is_notified_.compare_exchange_strong(expected, true)) {
      auto *parent_event_queue = reinterpret_cast<
          hipc::mpsc_ring_buffer<RunContext *, CHI_MAIN_ALLOC_T> *>(
          parent_task->event_queue_);
      parent_event_queue->Emplace(parent_task);
      // Awaken parent worker in case it's sleeping
      if (parent_task->lane != nullptr) {
        CHI_IPC->AwakenWorker(parent_task->lane);
      }
    }
  }

  // 3. Delete task using container->DelTask (only if destroy_in_end_task is
  // true)
  if (run_ctx->destroy_in_end_task_) {
    run_ctx->container->DelTask(task_ptr->method_, task_ptr);
  }

  // Deallocate context
  DeallocateContext(run_ctx);
}

void Worker::RerouteDynamicTask(const FullPtr<Task> &task_ptr,
                                RunContext *run_ctx) {
  // Dynamic scheduling complete - now re-route task with updated pool_query
  // The task's pool_query_ should have been updated during execution
  // (e.g., from Dynamic to Local or Broadcast)

  Container *container = run_ctx->container;
  TaskLane *lane = run_ctx->lane;

  // Reset the TASK_STARTED flag so the task can be executed again
  task_ptr->ClearFlags(TASK_STARTED | TASK_ROUTED);

  // Re-route the task using the updated pool_query
  if (RouteTask(run_ctx->future_, lane, container)) {
    // Avoids recursive call to RerouteDynamicTask
    if (run_ctx->exec_mode == ExecMode::kDynamicSchedule) {
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

    if (!run_ctx || run_ctx->task.IsNull()) {
      // Invalid entry, don't re-add
      continue;
    }

    // Determine if this is a resume (task was started before) or first
    // execution
    bool is_started = run_ctx->task->task_flags_.Any(TASK_STARTED);

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
    HLOG(kDebug, "ProcessEventQueue: Popped run_ctx={}", (void *)run_ctx);
    if (!run_ctx || run_ctx->task.IsNull()) {
      HLOG(kDebug, "ProcessEventQueue: Skipping null run_ctx or task");
      continue;
    }

    // Skip if coroutine handle is null or already completed
    // This can legitimately happen when:
    // 1. Multiple parallel subtasks complete and each posts an event to wake
    // parent
    //    Only the first event is needed; subsequent events are orphans
    // 2. Parent already completed and was destroyed before events were
    // processed
    // 3. Coroutine completed synchronously (no suspension point hit)
    if (!run_ctx->coro_handle_ || run_ctx->coro_handle_.done()) {
      HLOG(kDebug, "ProcessEventQueue: Skipping - coro_handle_={}, done={}",
           (void *)run_ctx->coro_handle_.address(),
           run_ctx->coro_handle_ ? run_ctx->coro_handle_.done() : false);
      continue;
    }

    HLOG(kDebug, "ProcessEventQueue: Resuming task method={}, coro_handle_={}",
         run_ctx->task->method_, (void *)run_ctx->coro_handle_.address());

    // Reset the is_yielded_ flag before executing the task
    run_ctx->is_yielded_ = false;

    // Reset is_notified_ so this task can be notified again for subsequent
    // co_await
    run_ctx->is_notified_.store(false);

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

RunContext* GetCurrentRunContextFromWorker() {
  Worker* worker = CHI_CUR_WORKER;
  if (worker) {
    return worker->GetCurrentRunContext();
  }
  return nullptr;
}

}  // namespace chi