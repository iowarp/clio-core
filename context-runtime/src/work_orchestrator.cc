/**
 * Work orchestrator implementation
 */

#include "chimaera/work_orchestrator.h"

#include <boost/context/detail/fcontext.hpp>
#include <chrono>
#include <cstdlib>
#include <iostream>

#include "chimaera/container.h"
#include "chimaera/singletons.h"

// Global pointer variable definition for Work Orchestrator singleton
HSHM_DEFINE_GLOBAL_PTR_VAR_CC(chi::WorkOrchestrator, g_work_orchestrator);

namespace chi {

//===========================================================================
// Stack Growth Direction Detection (for boost::context)
//===========================================================================

// Structure to pass data to stack detection function
struct StackDetectionData {
  void *middle_ptr;
  bool *detection_complete;
};

// Function called in boost context to detect stack growth direction
static void StackDetectionFunction(boost::context::detail::transfer_t t) {
  // Get the data passed from the context
  StackDetectionData *data = static_cast<StackDetectionData *>(t.data);

  // Create a local array to check stack addresses
  char local_array[64];
  void *array_start = &local_array[0];

  // For debugging - check where we are relative to middle
  bool is_below_middle = (array_start < data->middle_ptr);

  // Debug output to understand what's happening
  HLOG(kDebug,
        "Stack detection: middle_ptr={}, array_start={}, is_below_middle={}",
        data->middle_ptr, array_start, is_below_middle);

  *data->detection_complete = true;

  // Jump back to caller
  boost::context::detail::jump_fcontext(t.fctx, nullptr);
}

// Detect stack growth direction at runtime using boost context
static bool DetectStackGrowthDirection() {
  // Allocate 128KB test stack
  const size_t test_stack_size = 128 * 1024;
  void *test_stack = malloc(test_stack_size);
  if (!test_stack) {
    // Fallback to downward assumption
    HLOG(kDebug,
          "Stack detection: Failed to allocate test stack, assuming downward");
    return true;
  }

  // Calculate middle pointer
  void *middle_ptr = static_cast<char *>(test_stack) + (test_stack_size / 2);

  // Prepare detection data
  bool detection_complete = false;
  StackDetectionData data = {middle_ptr, &detection_complete};

  // Try high-end pointer first (correct for downward-growing stacks)
  HLOG(kDebug, "Testing stack detection with high-end pointer...");
  void *high_end_ptr = static_cast<char *>(test_stack) + test_stack_size;

  bool stack_grows_downward = true; // Default assumption

  try {
    auto context = boost::context::detail::make_fcontext(
        high_end_ptr, test_stack_size, StackDetectionFunction);
    boost::context::detail::jump_fcontext(context, &data);
  } catch (...) {
    HLOG(kDebug, "High-end pointer attempt failed");
    detection_complete = false;
  }

  if (detection_complete) {
    // If high-end pointer worked, it means make_fcontext expects high-end
    // pointer This is the correct behavior for downward-growing stacks
    stack_grows_downward = true;
    HLOG(kDebug, "High-end pointer succeeded - stack grows downward (correct "
                  "for x86_64)");
  } else {
    // If first attempt failed, try low end pointer (upward growth)
    HLOG(kDebug, "Testing stack detection with low-end pointer...");
    detection_complete = false;

    try {
      auto context2 = boost::context::detail::make_fcontext(
          test_stack, test_stack_size, StackDetectionFunction);
      boost::context::detail::jump_fcontext(context2, &data);
    } catch (...) {
      HLOG(kDebug, "Low-end pointer attempt also failed");
      detection_complete = false;
    }

    if (detection_complete) {
      // If low-end pointer worked, it means make_fcontext expects low-end
      // pointer This would be for upward-growing stacks (rare)
      stack_grows_downward = false;
      HLOG(kDebug, "Low-end pointer succeeded - stack grows upward (unusual "
                    "architecture)");
    } else {
      // Fallback to downward assumption
      HLOG(kDebug,
            "Both attempts failed, falling back to downward assumption");
      stack_grows_downward = true;
    }
  }

  free(test_stack);

  // Log the detection result
  HLOG(kDebug, "Stack growth direction detected: {}",
        (stack_grows_downward ? "downward" : "upward"));

  return stack_grows_downward;
}

//===========================================================================
// Work Orchestrator Implementation
//===========================================================================

// Constructor and destructor removed - handled by HSHM singleton pattern

bool WorkOrchestrator::Init() {
  if (is_initialized_) {
    return true;
  }

  // Detect stack growth direction once at orchestrator initialization
  stack_is_downward_ = DetectStackGrowthDirection();

  // Initialize HSHM TLS key for workers
  HSHM_THREAD_MODEL->CreateTls<class Worker>(chi_cur_worker_key_, nullptr);

  // Initialize scheduling state
  next_worker_index_for_scheduling_.store(0);
  active_lanes_ = nullptr;
  net_worker_ = nullptr;

  // Initialize HSHM thread group first
  auto thread_model = HSHM_THREAD_MODEL;
  thread_group_ = thread_model->CreateThreadGroup({});

  ConfigManager *config = CHI_CONFIG_MANAGER;
  if (!config) {
    return false; // Configuration manager not initialized
  }

  // Get worker counts from configuration
  u32 sched_count = config->GetSchedulerWorkerCount();
  u32 slow_count = config->GetSlowWorkerCount();

  // Create scheduler workers (fast tasks)
  for (u32 i = 0; i < sched_count; ++i) {
    if (!CreateWorker(kSchedWorker)) {
      return false;
    }
  }

  // Create slow workers (long-running tasks)
  for (u32 i = 0; i < slow_count; ++i) {
    if (!CreateWorker(kSlow)) {
      return false;
    }
  }

  // Create dedicated network worker (hardcoded to 1 for now)
  if (!CreateWorker(kNetWorker)) {
    return false;
  }

  is_initialized_ = true;
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

  // Cleanup worker threads using HSHM thread model
  auto thread_model = HSHM_THREAD_MODEL;
  for (auto &thread : worker_threads_) {
    thread_model->Join(thread);
  }
  worker_threads_.clear();

  // Clear worker containers
  all_workers_.clear();
  sched_workers_.clear();

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

  // Stop all workers
  for (auto *worker : all_workers_) {
    if (worker) {
      worker->Stop();
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

std::vector<Worker *>
WorkOrchestrator::GetWorkersByType(ThreadType thread_type) const {
  std::vector<Worker *> workers;
  if (!is_initialized_) {
    return workers;
  }

  for (auto *worker : all_workers_) {
    if (worker && worker->GetThreadType() == thread_type) {
      workers.push_back(worker);
    }
  }

  return workers;
}

size_t WorkOrchestrator::GetWorkerCount() const {
  return is_initialized_ ? all_workers_.size() : 0;
}

u32 WorkOrchestrator::GetWorkerCountByType(ThreadType thread_type) const {
  ConfigManager *config = CHI_CONFIG_MANAGER;
  return config->GetWorkerThreadCount(thread_type);
}

bool WorkOrchestrator::IsInitialized() const { return is_initialized_; }

bool WorkOrchestrator::AreWorkersRunning() const { return workers_running_; }

bool WorkOrchestrator::IsStackDownward() const { return stack_is_downward_; }

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

  // Map lanes to sched workers (only sched workers process tasks from worker
  // queues)
  u32 num_sched_workers = static_cast<u32>(sched_workers_.size());
  if (num_sched_workers == 0) {
    HLOG(kError,
          "WorkOrchestrator: No sched workers available for lane mapping");
    return false;
  }

  // Number of lanes should equal number of sched workers (configured in
  // IpcManager) Each worker gets exactly one lane for 1:1 mapping
  for (u32 worker_idx = 0; worker_idx < num_sched_workers; ++worker_idx) {
    Worker *worker = sched_workers_[worker_idx].get();
    if (worker) {
      // Direct 1:1 mapping: worker i gets lane i
      u32 lane_id = worker_idx;
      TaskLane *lane = &worker_queues->GetLane(lane_id, 0);

      // Set the worker's assigned lane
      worker->SetLane(lane);

      // Mark the lane with the assigned worker ID
      lane->SetAssignedWorkerId(worker->GetId());

      HLOG(kDebug,
            "WorkOrchestrator: Mapped sched worker {} (ID {}) to worker "
            "queue lane {}",
            worker_idx, worker->GetId(), lane_id);
    }
  }

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
    return true;
  } catch (const std::exception &e) {
    return false;
  }
}

bool WorkOrchestrator::CreateWorker(ThreadType thread_type) {
  u32 worker_id = static_cast<u32>(all_workers_.size());
  auto worker = std::make_unique<Worker>(worker_id, thread_type);

  if (!worker->Init()) {
    return false;
  }

  Worker *worker_ptr = worker.get();
  all_workers_.push_back(worker_ptr);

  // Add to type-specific container
  switch (thread_type) {
  case kSchedWorker:
    sched_workers_.push_back(std::move(worker));
    scheduler_workers_.push_back(worker_ptr);
    break;
  case kSlow:
    sched_workers_.push_back(std::move(worker));
    slow_workers_.push_back(worker_ptr);
    break;
  case kNetWorker:
    sched_workers_.push_back(std::move(worker));
    net_worker_ = worker_ptr;
    break;
  default:
    // Unknown worker type
    return false;
  }

  return true;
}

bool WorkOrchestrator::CreateWorkers(ThreadType thread_type, u32 count) {
  for (u32 i = 0; i < count; ++i) {
    if (!CreateWorker(thread_type)) {
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

  // Get all container pool IDs from the pool manager
  std::vector<PoolId> all_pool_ids = pool_manager->GetAllPoolIds();

  for (const auto &pool_id : all_pool_ids) {
    // Get container for each pool
    Container *container = pool_manager->GetContainer(pool_id);
    if (container) {
      total_work_remaining += container->GetWorkRemaining();
    }
  }

  return total_work_remaining > 0;
}

void WorkOrchestrator::AssignToWorkerType(ThreadType thread_type,
                                          const FullPtr<Task> &task_ptr) {
  if (task_ptr.IsNull()) {
    return;
  }

  // Select target worker vector based on thread type
  std::vector<Worker *> *target_workers = nullptr;
  if (thread_type == kSchedWorker) {
    target_workers = &scheduler_workers_;
  } else if (thread_type == kSlow) {
    target_workers = &slow_workers_;
  } else {
    // Process reaper or other types - not supported for task routing
    return;
  }

  if (target_workers->empty()) {
    HLOG(kWarning, "AssignToWorkerType: No workers of type {}",
          static_cast<int>(thread_type));
    return;
  }

  // Round-robin assignment using static atomic counters
  static std::atomic<size_t> scheduler_idx{0};
  static std::atomic<size_t> slow_idx{0};

  std::atomic<size_t> &idx =
      (thread_type == kSchedWorker) ? scheduler_idx : slow_idx;
  size_t worker_idx = idx.fetch_add(1) % target_workers->size();
  Worker *worker = (*target_workers)[worker_idx];

  // Get the worker's assigned lane and emplace the task
  TaskLane *lane = worker->GetLane();
  if (lane) {
    // RUNTIME PATH: Create Future with task pointer set (no serialization)
    auto *ipc_manager = CHI_IPC;
    auto *alloc = ipc_manager->GetMainAlloc();
    Future<Task> future(alloc, task_ptr);

    // Emplace the Future into the lane
    lane->Emplace(future);
  }
}

} // namespace chi