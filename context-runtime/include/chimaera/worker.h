#ifndef CHIMAERA_INCLUDE_CHIMAERA_WORKERS_WORKER_H_
#define CHIMAERA_INCLUDE_CHIMAERA_WORKERS_WORKER_H_

#include <boost/context/detail/fcontext.hpp>
#include <chrono>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "chimaera/container.h"
#include "chimaera/integer_timer.h"
#include "chimaera/pool_query.h"
#include "chimaera/task.h"
#include "chimaera/task_queue.h"
#include "chimaera/types.h"

namespace chi {

// Forward declaration to avoid circular dependency
using WorkQueue = hshm::ipc::mpsc_ring_buffer<hipc::ShmPtr<TaskLane>, CHI_MAIN_ALLOC_T>;

// Forward declarations
class Task;

/**
 * Structure to hold a cached stack and RunContext together
 * Used for efficient reuse of stack allocations
 */
struct StackAndContext {
  void *stack_base_for_free; /**< Base pointer for freeing the stack */
  size_t stack_size;         /**< Size of the stack in bytes */
  RunContext *run_ctx;       /**< Pointer to the RunContext */

  StackAndContext()
      : stack_base_for_free(nullptr), stack_size(0), run_ctx(nullptr) {}

  StackAndContext(void *stack_base, size_t size, RunContext *ctx)
      : stack_base_for_free(stack_base), stack_size(size), run_ctx(ctx) {}
};

// Macro for accessing HSHM thread-local storage (worker thread context)
// This macro allows access to the current worker from any thread
// Example usage in ChiMod container code:
//   Worker* worker = CHI_CUR_WORKER;
//   FullPtr<Task> current_task = worker->GetCurrentTask();
//   RunContext* run_ctx = worker->GetCurrentRunContext();
#define CHI_CUR_WORKER                                                         \
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
   * @param thread_type Type of worker thread
   */
  Worker(u32 worker_id, ThreadType thread_type);

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
   * Get worker thread type
   * @return Type of worker thread
   */
  ThreadType GetThreadType() const;

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
   * Add run context to blocked queue based on block count
   * @param run_ctx_ptr Pointer to run context (task accessible via
   * run_ctx_ptr->task)
   */
  void AddToBlockedQueue(RunContext *run_ctx_ptr);

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

  /**
   * Route a task by calling ResolvePoolQuery and determining local vs global
   * scheduling
   * @param task_ptr Full pointer to task to route
   * @param lane Pointer to the task lane for execution context
   * @param container Output parameter for the container to use for task
   * execution
   * @return true if task was successfully routed, false otherwise
   */
  bool RouteTask(const FullPtr<Task> &task_ptr, TaskLane *lane,
                 Container *&container);

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
  void ProcessBlockedQueue(std::queue<RunContext *> &queue,
                           u32 queue_idx);

  /**
   * Process a periodic queue, checking time-based tasks and executing if ready
   * @param queue Reference to the ext_ring_buffer to process
   * @param queue_idx Index of the queue being processed (0-3)
   */
  void ProcessPeriodicQueue(std::queue<RunContext *> &queue,
                            u32 queue_idx);

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
   * @param task_ptr Full pointer to task to route locally
   * @param lane Pointer to the task lane for execution context
   * @param container Output parameter for the container to use for task
   * execution
   * @return true if local routing successful, false otherwise
   */
  bool RouteLocal(const FullPtr<Task> &task_ptr, TaskLane *lane,
                  Container *&container);

  /**
   * Route task globally using admin client's ClientSendTaskIn method
   * @param task_ptr Full pointer to task to route globally
   * @param pool_queries Vector of pool queries for global routing
   * @return true if global routing successful, false otherwise
   */
  bool RouteGlobal(const FullPtr<Task> &task_ptr,
                   const std::vector<PoolQuery> &pool_queries);

private:
  /**
   * Allocate stack and RunContext for task execution (64KB default)
   * @param size Stack size in bytes
   * @return RunContext pointer with stack_ptr set
   */
  RunContext *AllocateStackAndContext(size_t size = 65536); // 64KB default

  /**
   * Deallocate task execution stack and RunContext
   * @param run_ctx_ptr Pointer to RunContext containing stack info to
   * deallocate
   */
  void DeallocateStackAndContext(RunContext *run_ctx_ptr);

  /**
   * Begin task execution using boost::fiber for context switching
   * @param task_ptr Full pointer to task to execute (RunContext will be
   * allocated and set in task)
   * @param container Container for the task
   * @param lane Lane for the task (can be nullptr)
   */
  void BeginTask(const FullPtr<Task> &task_ptr, Container *container,
                 TaskLane *lane);

  /**
   * Continue processing blocked tasks that are ready to resume
   * @param force If true, process both queues regardless of iteration count
   */
  void ContinueBlockedTasks(bool force);

  /**
   * Process tasks from the worker's assigned lane
   * Processes up to MAX_TASKS_PER_ITERATION tasks per call
   * @return Number of tasks processed
   */
  u32 ProcessNewTasks();

  /**
   * Suspend worker when there is no work available
   * Implements adaptive sleep algorithm with busy wait and linear increment
   */
  void SuspendMe();

  /**
   * Execute task with context switching capability
   * @param task_ptr Full pointer to task to execute
   * @param run_ctx_ptr Pointer to existing RunContext
   * @param is_started True if task is resuming, false for new task
   */
  void ExecTask(const FullPtr<Task> &task_ptr, RunContext *run_ctx_ptr,
                bool is_started);

  /**
   * Begin fiber execution for a new task
   * @param task_ptr Full pointer to task to execute
   * @param run_ctx Pointer to RunContext for task
   * @param fiber_fn Function pointer for fiber execution
   */
  void BeginFiber(const FullPtr<Task> &task_ptr, RunContext *run_ctx,
                  void (*fiber_fn)(boost::context::detail::transfer_t));

  /**
   * Resume fiber execution for a yielded/blocked task
   * @param task_ptr Full pointer to task to resume
   * @param run_ctx Pointer to RunContext for task
   */
  void ResumeFiber(const FullPtr<Task> &task_ptr, RunContext *run_ctx);

  /**
   * End task execution and perform cleanup
   * @param task_ptr Full pointer to task to end
   * @param run_ctx Pointer to RunContext for task
   * @param should_complete Whether task should be marked as complete
   * @param is_remote Whether task is remote and needs to send outputs back
   */
  void EndTask(const FullPtr<Task> &task_ptr, RunContext *run_ctx,
               bool should_complete, bool is_remote);

  /**
   * End dynamic scheduling task and re-route with updated pool query
   * @param task_ptr Full pointer to task to re-route
   * @param run_ctx Pointer to RunContext for task
   */
  void RerouteDynamicTask(const FullPtr<Task> &task_ptr, RunContext *run_ctx);

  /**
   * Static function for boost::fiber execution context
   * @param t Transfer context for boost::fiber
   */
  static void FiberExecutionFunction(boost::context::detail::transfer_t t);

  u32 worker_id_;
  ThreadType thread_type_;
  bool is_running_;
  bool is_initialized_;
  bool did_work_;       // Tracks if any work was done in current loop iteration
  bool task_did_work_;  // Tracks if current task did actual work (set by tasks via CHI_CUR_WORKER)

  // Current RunContext for this worker thread
  RunContext *current_run_context_;

  // Single lane assigned to this worker (one lane per worker)
  TaskLane *assigned_lane_;

  // Stack and RunContext cache for efficient reuse
  // Using ext_ring_buffer for O(1) enqueue/dequeue operations
  std::queue<StackAndContext> stack_cache_;

  // Blocked queue system for cooperative tasks (waiting for subtasks):
  // - Queue[0]: Tasks blocked <=2 times (checked every % 2 iterations)
  // - Queue[1]: Tasks blocked <= 4 times (checked every % 4 iterations)
  // - Queue[2]: Tasks blocked <= 8 times (checked every % 8 iterations)
  // - Queue[3]: Tasks blocked > 8 times (checked every % 16 iterations)
  // Using std::queue for O(1) enqueue/dequeue operations
  static constexpr u32 NUM_BLOCKED_QUEUES = 4;
  static constexpr u32 BLOCKED_QUEUE_SIZE = 1024;
  std::queue<RunContext *> blocked_queues_[NUM_BLOCKED_QUEUES];

  // Periodic queue system for time-based periodic tasks:
  // - Queue[0]: Tasks with block_time_us <= 50us (checked every 16 iterations)
  // - Queue[1]: Tasks with block_time_us <= 200us (checked every 32 iterations)
  // - Queue[2]: Tasks with block_time_us <= 50ms/50000us (checked every 64 iterations)
  // - Queue[3]: Tasks with block_time_us > 50ms (checked every 128 iterations)
  // Using std::queue for O(1) enqueue/dequeue operations
  static constexpr u32 NUM_PERIODIC_QUEUES = 4;
  static constexpr u32 PERIODIC_QUEUE_SIZE = 1024;
  std::queue<RunContext *> periodic_queues_[NUM_PERIODIC_QUEUES];

  // Worker spawn time and queue processing tracking
  hshm::Timepoint spawn_time_; // Time when worker was spawned
  u64 last_long_queue_check_;  // Last time (in 10us units) long queue was
                               // processed

  // Iteration counter for periodic blocked queue checks
  u64 iteration_count_; // Number of iterations completed

  // Sleep management for idle workers
  u64 idle_iterations_;     // Number of consecutive iterations with no work
  u32 current_sleep_us_;    // Current sleep duration in microseconds
  u64 sleep_count_;         // Number of times sleep was called in current idle period
  hshm::Timepoint idle_start_;  // Time when worker became idle
};

} // namespace chi

#endif // CHIMAERA_INCLUDE_CHIMAERA_WORKERS_WORKER_H_