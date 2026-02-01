#ifndef CHIMAERA_INCLUDE_CHIMAERA_TASK_H_
#define CHIMAERA_INCLUDE_CHIMAERA_TASK_H_

#include <sys/types.h>

#include <atomic>
#include <coroutine>
#include <memory>
#include <sstream>
#include <vector>

#include "chimaera/pool_query.h"
#include "chimaera/types.h"
#include "hermes_shm/data_structures/ipc/shm_container.h"
#include "hermes_shm/data_structures/ipc/vector.h"
#include "hermes_shm/memory/allocator/allocator.h"
#include "hermes_shm/util/logging.h"

// Include cereal for serialization
#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>

// Forward declare chi::priv::string for cereal support
namespace hshm::priv {
template <typename T, typename AllocT, size_t SmallSize>
class basic_string;
}

namespace chi {

// Forward declarations
class Task;
class Container;
class IpcManager;
struct RunContext;
class Worker;

/**
 * Get the current RunContext from thread-local Worker storage
 * This function is implemented in worker.cc to avoid circular dependency
 * between task.h and worker.h
 * @return Pointer to current RunContext, or nullptr if not in a worker thread
 */
RunContext* GetCurrentRunContextFromWorker();

/**
 * Task statistics for I/O and compute time tracking
 * Used to route tasks to appropriate worker groups
 */
struct TaskStat {
  size_t io_size_{0}; /**< I/O size in bytes */
  size_t compute_{0}; /**< Normalized compute time in microseconds */
};

// Define macros for container template
#define CLASS_NAME Task
#define CLASS_NEW_ARGS

/**
 * Base task class for Chimaera distributed execution
 *
 * All tasks represent C++ functions similar to RPCs that can be executed
 * across the distributed system. Tasks are now allocated in private memory
 * using standard new/delete.
 */
class Task {
 public:
  typedef CHI_MAIN_ALLOC_T AllocT;
  IN PoolId pool_id_;       /**< Pool identifier for task execution */
  IN TaskId task_id_;       /**< Task identifier for task routing */
  IN PoolQuery pool_query_; /**< Pool query for execution location */
  IN MethodId method_;      /**< Method identifier for task type */
  IN ibitfield task_flags_; /**< Task properties and flags */
  IN double period_ns_;     /**< Period in nanoseconds for periodic tasks */
  IN std::unique_ptr<RunContext> run_ctx_; /**< Runtime context owned by task (RAII) */
  OUT hipc::atomic<u32>
      return_code_; /**< Task return code (0=success, non-zero=error) */
  OUT hipc::atomic<ContainerId>
      completer_; /**< Container ID that completed this task */
  TaskStat stat_; /**< Task statistics for I/O and compute tracking */

  /**
   * Default constructor
   */
  Task() { SetNull(); }

  /**
   * Emplace constructor with task initialization
   */
  explicit Task(const TaskId& task_id, const PoolId& pool_id,
                const PoolQuery& pool_query, const MethodId& method) {
    // Initialize task
    task_id_ = task_id;
    pool_id_ = pool_id;
    method_ = method;
    task_flags_.SetBits(0);
    pool_query_ = pool_query;
    period_ns_ = 0.0;
    // run_ctx_ is initialized by its default constructor
    return_code_.store(0);  // Initialize as success
    completer_.store(0);    // Initialize as null (0 is invalid container ID)
  }

  /**
   * Copy from another task (assumes this task is already constructed)
   *
   * IMPORTANT: Derived classes that override Copy MUST call Task::Copy(other)
   * first before copying their own fields.
   *
   * @param other Pointer to the source task to copy from
   */
  HSHM_CROSS_FUN void Copy(const hipc::FullPtr<Task>& other) {
    pool_id_ = other->pool_id_;
    task_id_ = other->task_id_;
    pool_query_ = other->pool_query_;
    method_ = other->method_;
    task_flags_ = other->task_flags_;
    period_ns_ = other->period_ns_;
    // Note: run_ctx_ is not copied - each task maintains its own RunContext
    // The RunContext will be initialized when the task is executed
    return_code_.store(other->return_code_.load());
    completer_.store(other->completer_.load());
    stat_ = other->stat_;
  }

  /**
   * SetNull implementation
   */
  HSHM_INLINE_CROSS_FUN void SetNull() {
    pool_id_ = PoolId::GetNull();
    task_id_ = TaskId();
    pool_query_ = PoolQuery();
    method_ = 0;
    task_flags_.Clear();
    period_ns_ = 0.0;
    run_ctx_.reset();  // Reset the unique_ptr (destroys RunContext if allocated)
    return_code_.store(0);  // Initialize as success
    completer_.store(0);    // Initialize as null (0 is invalid container ID)
    stat_.io_size_ = 0;
    stat_.compute_ = 0;
  }

  /**
   * Check if task is periodic
   * @return true if task has periodic flag set
   */
  HSHM_CROSS_FUN bool IsPeriodic() const {
    return task_flags_.Any(TASK_PERIODIC);
  }

  /**
   * Check if task has been routed
   * @return true if task has routed flag set
   */
  HSHM_CROSS_FUN bool IsRouted() const { return task_flags_.Any(TASK_ROUTED); }

  /**
   * Check if task is the data owner
   * @return true if task has data owner flag set
   */
  HSHM_CROSS_FUN bool IsDataOwner() const {
    return task_flags_.Any(TASK_DATA_OWNER);
  }

  /**
   * Check if task is a remote task (received from another node)
   * @return true if task has remote flag set
   */
  HSHM_CROSS_FUN bool IsRemote() const { return task_flags_.Any(TASK_REMOTE); }

  /**
   * Get task execution period in specified time unit
   * @param unit Time unit constant (kNano, kMicro, kMilli, kSec, kMin, kHour)
   * @return Period in specified unit, 0 if not periodic
   */
  HSHM_CROSS_FUN double GetPeriod(double unit) const {
    return period_ns_ / unit;
  }

  /**
   * Set task execution period in specified time unit
   * @param period Period value in the specified unit
   * @param unit Time unit constant (kNano, kMicro, kMilli, kSec, kMin, kHour)
   */
  HSHM_CROSS_FUN void SetPeriod(double period, double unit) {
    period_ns_ = period * unit;
  }

  /**
   * Set task flags
   * @param flags Bitfield of task flags to set
   */
  HSHM_CROSS_FUN void SetFlags(u32 flags) { task_flags_.SetBits(flags); }

  /**
   * Clear task flags
   * @param flags Bitfield of task flags to clear
   */
  HSHM_CROSS_FUN void ClearFlags(u32 flags) { task_flags_.UnsetBits(flags); }

  /**
   * Serialize data structures to chi::priv::string using cereal
   * @param alloc Malloc allocator for memory management (private data)
   * @param output_str The string to store serialized data
   * @param args The arguments to serialize
   */
  template <typename... Args>
  static void Serialize(hipc::MallocAllocator* alloc, chi::priv::string& output_str,
                        const Args&... args) {
    std::ostringstream os;
    cereal::BinaryOutputArchive archive(os);
    archive(args...);

    std::string serialized = os.str();
    output_str = chi::priv::string(alloc, serialized);
  }

  /**
   * Deserialize data structure from chi::ipc::string using cereal
   * @param input_str The string containing serialized data
   * @return The deserialized object
   */
  template <typename OutT>
  static OutT Deserialize(const chi::priv::string& input_str) {
    std::string data = input_str.str();
    std::istringstream is(data);
    cereal::BinaryInputArchive archive(is);

    OutT result;
    archive(result);
    return result;
  }

  /**
   * Serialize task for incoming network transfer (IN and INOUT parameters)
   * This method serializes the base Task fields first, then should be
   * overridden by derived classes to serialize their specific fields.
   *
   * IMPORTANT: Derived classes MUST call Task::SerializeIn(ar) first before
   * serializing their own fields.
   *
   * @param ar Archive to serialize to
   */
  template <typename Archive>
  void SerializeIn(Archive& ar) {
    // Serialize base Task fields (IN and INOUT parameters)
    ar(pool_id_, task_id_, pool_query_, method_, task_flags_, period_ns_,
       return_code_);
  }

  /**
   * Serialize task for outgoing network transfer (OUT and INOUT parameters)
   * This method serializes the base Task OUT fields first, then should be
   * overridden by derived classes to serialize their specific OUT fields.
   *
   * IMPORTANT: Derived classes MUST call Task::SerializeOut(ar) first before
   * serializing their own OUT fields.
   *
   * @param ar Archive to serialize to
   */
  template <typename Archive>
  void SerializeOut(Archive& ar) {
    // Serialize base Task OUT fields only
    // Only serialize OUT fields - do NOT re-serialize IN fields
    // (pool_id_, task_id_, pool_query_, method_, task_flags_, period_ns_ are
    // all IN) Only return_code_ and completer_ are OUT fields that need to be
    // sent back
    ar(return_code_, completer_);
  }

  /**
   * Get the task return code
   * @return Return code (0=success, non-zero=error)
   */
  HSHM_CROSS_FUN u32 GetReturnCode() const { return return_code_.load(); }

  /**
   * Set the task return code
   * @param return_code Return code to set (0=success, non-zero=error)
   */
  HSHM_CROSS_FUN void SetReturnCode(u32 return_code) {
    return_code_.store(return_code);
  }

  /**
   * Get the completer container ID (which container completed this task)
   * @return Container ID that completed this task
   */
  HSHM_CROSS_FUN ContainerId GetCompleter() const { return completer_.load(); }

  /**
   * Post-wait callback called after task completion
   * Called by Future::Wait() and co_await Future after task is complete.
   * Derived classes can override this to perform post-completion actions.
   * Default implementation does nothing.
   */
  void PostWait() {
    // Base implementation does nothing
  }

  /**
   * Set the completer container ID (which container completed this task)
   * @param completer Container ID to set
   */
  HSHM_CROSS_FUN void SetCompleter(ContainerId completer) {
    completer_.store(completer);
  }

  /**
   * Base aggregate method - propagates return codes and completer from replica
   * tasks Sets this task's return code to the replica's return code if replica
   * has non-zero return code Accepts any task type that inherits from Task
   *
   * IMPORTANT: Derived classes that override Aggregate MUST call
   * Task::Aggregate(replica_task) first before aggregating their own fields.
   *
   * @param replica_task The replica task to aggregate from
   */
  template <typename TaskT>
  HSHM_CROSS_FUN void Aggregate(const hipc::FullPtr<TaskT>& replica_task) {
    // Cast to base Task for aggregation
    auto base_replica = replica_task.template Cast<Task>();
    // Propagate return code from replica to this task
    if (!base_replica.IsNull() && base_replica->GetReturnCode() != 0) {
      SetReturnCode(base_replica->GetReturnCode());
    }
    // Copy the completer from the replica task
    if (!base_replica.IsNull()) {
      SetCompleter(base_replica->GetCompleter());
    }
    HLOG(kDebug, "[COMPLETER] Aggregated task {} with completer {}", task_id_,
         GetCompleter());
  }

  /**
   * Estimate CPU time for this task based on I/O size and compute time
   * Formula: (io_size / 4GBps) + compute + 5us
   * @return Estimated CPU time in microseconds
   */
  HSHM_CROSS_FUN size_t EstCpuTime() const;
};

/**
 * Execution mode for task processing
 */
enum class ExecMode : u32 {
  kExec = 0,           /**< Normal task execution */
  kDynamicSchedule = 1 /**< Dynamic scheduling - route after execution */
};

// ============================================================================
// FutureShm and Future classes (must be before RunContext which uses Future)
// ============================================================================

/**
 * FutureShm - Shared memory container for task future state
 *
 * This container holds the serialized task data and completion status
 * for asynchronous task operations.
 */
template <typename AllocT = CHI_MAIN_ALLOC_T>
class FutureShm : public hipc::ShmContainer<AllocT> {
 public:
  /** Pool ID for the task */
  PoolId pool_id_;

  /** Method ID for the task */
  u32 method_id_;

  /** Serialized task data */
  hipc::vector<char, AllocT> serialized_task_;

  /** Atomic completion flag (0=not complete, 1=complete) */
  hipc::atomic<u32> is_complete_;

  /** Size of the shared memory segment containing this FutureShm
   *  Used for lazy registration - shm_name derived from alloc_id (pid.count)
   */
  size_t shm_size_;

  /**
   * SHM default constructor
   */
  explicit FutureShm(AllocT* alloc)
      : hipc::ShmContainer<AllocT>(alloc), serialized_task_(alloc) {
    pool_id_ = PoolId::GetNull();
    method_id_ = 0;
    is_complete_.store(0);
    shm_size_ = 0;
  }
};

/**
 * Future - Template class for asynchronous task operations
 *
 * Future provides a handle to an asynchronous task operation, allowing
 * the caller to check completion status and retrieve results.
 *
 * @tparam TaskT The task type (e.g., CreateTask, CustomTask)
 * @tparam AllocT The allocator type (defaults to CHI_MAIN_ALLOC_T)
 */
template <typename TaskT, typename AllocT = CHI_MAIN_ALLOC_T>
class Future {
 public:
  using FutureT = FutureShm<AllocT>;

  // Allow all Future instantiations to access each other's private members
  // This enables the Cast method to work across different task types
  template <typename OtherTaskT, typename OtherAllocT>
  friend class Future;

 private:
  /** FullPtr to the task (wraps private memory with null allocator) */
  hipc::FullPtr<TaskT> task_ptr_;

  /** FullPtr to the shared FutureShm object */
  hipc::FullPtr<FutureT> future_shm_;

  /** Parent task RunContext pointer (nullptr if no parent waiting) */
  RunContext* parent_task_;

  /** Flag indicating if this Future owns the task and should destroy it */
  bool is_owner_;

  /**
   * Implementation of await_suspend
   * Defined after RunContext to access its members
   */
  bool await_suspend_impl(std::coroutine_handle<> handle) noexcept;

 public:
  /**
   * Constructor with allocator - allocates new FutureShm
   * @param alloc Allocator to use for FutureShm allocation
   * @param task_ptr FullPtr to the task (wraps private memory with null
   * allocator)
   */
  Future(AllocT* alloc, hipc::FullPtr<TaskT> task_ptr)
      : task_ptr_(task_ptr), parent_task_(nullptr), is_owner_(false) {
    // Allocate FutureShm object (null check for safety)
    if (alloc != nullptr) {
      future_shm_ =
          alloc->template NewObj<FutureT>(alloc).template Cast<FutureT>();
      // Copy pool_id to FutureShm
      if (!task_ptr_.IsNull() && !future_shm_.IsNull()) {
        future_shm_->pool_id_ = task_ptr_->pool_id_;
        future_shm_->method_id_ = task_ptr_->method_;
      }
    }
  }

  /**
   * Constructor with allocator and existing FutureShm
   * @param alloc Allocator (for FullPtr construction)
   * @param task_ptr FullPtr to the task (wraps private memory with null
   * allocator)
   * @param future_shm ShmPtr to existing FutureShm object
   */
  Future(AllocT* alloc, hipc::FullPtr<TaskT> task_ptr,
         hipc::ShmPtr<FutureT> future_shm)
      : task_ptr_(task_ptr),
        future_shm_(alloc, future_shm),
        parent_task_(nullptr),
        is_owner_(false) {}

  /**
   * Constructor from FullPtr<FutureShm> and FullPtr<Task>
   * @param future_shm FullPtr to existing FutureShm object
   * @param task_ptr FullPtr to the task (wraps private memory with null
   * allocator)
   */
  Future(hipc::FullPtr<FutureT> future_shm, hipc::FullPtr<TaskT> task_ptr)
      : task_ptr_(task_ptr),
        future_shm_(future_shm),
        parent_task_(nullptr),
        is_owner_(false) {
    // No need to copy pool_id - FutureShm already has it
  }

  /**
   * Default constructor - creates null future
   */
  Future() : parent_task_(nullptr), is_owner_(false) {}

  /**
   * Constructor from ShmPtr<FutureShm> - used by ring buffer deserialization
   * Task pointer will be null and must be set later
   * @param future_shm_ptr ShmPtr to FutureShm object
   */
  explicit Future(const hipc::ShmPtr<FutureT>& future_shm_ptr)
      : future_shm_(nullptr, future_shm_ptr),
        parent_task_(nullptr),
        is_owner_(false) {
    // Task pointer starts null - will be set in ProcessNewTasks
    task_ptr_.SetNull();
  }

  /**
   * Fix the allocator pointer after construction from ShmPtr
   * Call this immediately after popping from ring buffer
   * Uses IpcManager::ToFullPtr to resolve the allocator from the ShmPtr
   */
  void SetAllocator();

  /**
   * Destructor - destroys the task if this Future owns it
   */
  ~Future() {
    if (is_owner_) {
      Destroy();
    }
  }

  /**
   * Destroy the task using CHI_IPC->DelTask if not null
   * Sets the task pointer to null afterwards
   */
  void Destroy();

  /**
   * Copy constructor - does not transfer ownership
   * @param other Future to copy from
   */
  Future(const Future& other)
      : task_ptr_(other.task_ptr_),
        future_shm_(other.future_shm_),
        parent_task_(other.parent_task_),
        is_owner_(false) {}  // Copy does not transfer ownership

  /**
   * Copy assignment operator - does not transfer ownership
   * @param other Future to copy from
   * @return Reference to this future
   */
  Future& operator=(const Future& other) {
    if (this != &other) {
      // Destroy existing task if we own it
      if (is_owner_) {
        Destroy();
      }
      task_ptr_ = other.task_ptr_;
      future_shm_ = other.future_shm_;
      parent_task_ = other.parent_task_;
      is_owner_ = false;  // Copy does not transfer ownership
    }
    return *this;
  }

  /**
   * Move constructor - transfers ownership
   * @param other Future to move from
   */
  Future(Future&& other) noexcept
      : task_ptr_(std::move(other.task_ptr_)),
        future_shm_(std::move(other.future_shm_)),
        parent_task_(other.parent_task_),
        is_owner_(other.is_owner_) {  // Transfer ownership
    other.parent_task_ = nullptr;
    other.is_owner_ = false;  // Source no longer owns
  }

  /**
   * Move assignment operator - transfers ownership
   * @param other Future to move from
   * @return Reference to this future
   */
  Future& operator=(Future&& other) noexcept {
    if (this != &other) {
      // Destroy existing task if we own it
      if (is_owner_) {
        Destroy();
      }
      task_ptr_ = std::move(other.task_ptr_);
      future_shm_ = std::move(other.future_shm_);
      parent_task_ = other.parent_task_;
      is_owner_ = other.is_owner_;  // Transfer ownership
      other.parent_task_ = nullptr;
      other.is_owner_ = false;  // Source no longer owns
    }
    return *this;
  }

  /**
   * Get raw pointer to the task
   * @return Pointer to the task object
   */
  TaskT* get() const { return task_ptr_.ptr_; }

  /**
   * Get the FullPtr to the task (non-const version)
   * @return FullPtr to the task object
   */
  hipc::FullPtr<TaskT>& GetTaskPtr() { return task_ptr_; }

  /**
   * Get the FullPtr to the task (const version)
   * @return FullPtr to the task object
   */
  const hipc::FullPtr<TaskT>& GetTaskPtr() const { return task_ptr_; }

  /**
   * Dereference operator - access task members
   * @return Reference to the task object
   */
  TaskT& operator*() const { return *task_ptr_.ptr_; }

  /**
   * Arrow operator - access task members
   * @return Pointer to the task object
   */
  TaskT* operator->() const { return task_ptr_.ptr_; }

  /**
   * Check if the task is complete
   * @return True if task has completed, false otherwise
   */
  bool IsComplete() const {
    if (future_shm_.IsNull()) {
      return false;
    }
    return future_shm_->is_complete_.load() != 0;
  }

  /**
   * Wait for task completion (blocking)
   * Calls IpcManager::Recv() to handle task completion and deserialization
   */
  void Wait();

  /**
   * Mark the task as complete
   */
  void Complete() {
    if (!future_shm_.IsNull()) {
      future_shm_->is_complete_.store(1);
    }
  }

  /**
   * Mark the task as complete (alias for Complete)
   */
  void SetComplete() { Complete(); }

  /**
   * Check if this future is null
   * @return True if future is null, false otherwise
   */
  bool IsNull() const { return task_ptr_.IsNull(); }

  /**
   * Get the FutureShm FullPtr
   * @return FullPtr to the FutureShm object
   */
  hipc::FullPtr<FutureT>& GetFutureShm() { return future_shm_; }

  /**
   * Get the FutureShm FullPtr (const version)
   * @return FullPtr to the FutureShm object
   */
  const hipc::FullPtr<FutureT>& GetFutureShm() const { return future_shm_; }

  /**
   * Get the pool ID from the FutureShm
   * @return Pool ID for the task
   */
  PoolId GetPoolId() const {
    if (future_shm_.IsNull()) {
      return PoolId::GetNull();
    }
    return future_shm_->pool_id_;
  }

  /**
   * Set the pool ID in the FutureShm
   * @param pool_id Pool ID to set
   */
  void SetPoolId(const PoolId& pool_id) {
    if (!future_shm_.IsNull()) {
      future_shm_->pool_id_ = pool_id;
    }
  }

  /**
   * Cast this Future to a Future of a different task type
   *
   * This is a safe operation because Future<TaskT> and Future<NewTaskT>
   * have identical memory layouts - they both store the same underlying
   * pointers (task_ptr_, future_shm_, parent_task_).
   *
   * Note: Cast does not transfer ownership - the original Future retains it.
   *
   * @tparam NewTaskT The new task type to cast to
   * @return Future<NewTaskT> with the same underlying state (non-owning)
   */
  template <typename NewTaskT>
  Future<NewTaskT, AllocT> Cast() const {
    Future<NewTaskT, AllocT> result;
    // Use reinterpret_cast to copy the memory layout
    // This works because Future<TaskT> and Future<NewTaskT> have identical
    // sizes
    result.task_ptr_ = task_ptr_.template Cast<NewTaskT>();
    result.future_shm_ = future_shm_;
    result.parent_task_ = parent_task_;
    result.is_owner_ = false;  // Cast does not transfer ownership
    return result;
  }

  /**
   * Get the parent task RunContext pointer
   * @return Pointer to parent RunContext or nullptr
   */
  RunContext* GetParentTask() const { return parent_task_; }

  /**
   * Set the parent task RunContext pointer
   * @param parent_task Pointer to parent RunContext
   */
  void SetParentTask(RunContext* parent_task) { parent_task_ = parent_task; }

  // =========================================================================
  // C++20 Coroutine Awaitable Interface
  // These methods allow `co_await future` in runtime coroutines
  // Note: Template methods defer instantiation, so RunContext access is OK
  // =========================================================================

  /**
   * Check if the awaitable is ready (coroutine await_ready)
   *
   * If the task is already complete, the coroutine won't suspend.
   * @return True if task is complete, false if coroutine should suspend
   */
  bool await_ready() const noexcept {
    return IsComplete();
  }

  /**
   * Suspend the coroutine and register for resumption (coroutine await_suspend)
   *
   * This is called when await_ready returns false. It stores the coroutine
   * handle in the RunContext so the worker can resume it when the task
   * completes. Also marks this Future as the owner of the task.
   *
   * Uses type-erased coroutine handle and gets RunContext from thread-local
   * Worker storage rather than from the promise to ensure proper template
   * instantiation.
   *
   * @param handle The coroutine handle to resume when task completes
   * @return True to suspend, false to continue without suspending
   */
  bool await_suspend(std::coroutine_handle<> handle) noexcept {
    // Mark this Future as owner of the task
    is_owner_ = true;

    // Get RunContext via helper function (defined in worker.cc)
    // This avoids needing RunContext to be complete at this point
    return await_suspend_impl(handle);
  }

  /**
   * Get the result after resumption (coroutine await_resume)
   *
   * Returns void to avoid GCC 11 "statement has no effect" warning
   * which causes the compiler to skip the await machinery entirely.
   * Marks this Future as the owner if not already set (for await_ready=true
   * case). Calls PostWait() on the task for post-completion actions.
   */
  void await_resume() noexcept {
    // If await_ready returned true, await_suspend wasn't called, so set
    // ownership here
    is_owner_ = true;
    // Call PostWait() callback on the task for post-completion actions
    if (!task_ptr_.IsNull()) {
      task_ptr_->PostWait();
    }
  }
};

// ============================================================================
// Task Queue types (must be after Future for TaskLane typedef)
// Placed before RunContext so RunContext can use TaskLane*
// ============================================================================

/**
 * Custom header for tracking lane state (stored per-lane)
 */
struct TaskQueueHeader {
  PoolId pool_id;
  WorkerId assigned_worker_id;
  u32 task_count;    // Number of tasks currently in the queue
  bool is_enqueued;  // Whether this queue is currently enqueued in worker
  int signal_fd_;    // Signal file descriptor for awakening worker
  pid_t tid_;        // Thread ID of the worker owning this lane
  std::atomic<bool> active_;  // Whether worker is accepting tasks (true) or
                              // blocked in epoll_wait (false)

  TaskQueueHeader()
      : pool_id(),
        assigned_worker_id(0),
        task_count(0),
        is_enqueued(false),
        signal_fd_(-1),
        tid_(0) {
    active_.store(true);
  }

  TaskQueueHeader(PoolId pid, WorkerId wid = 0)
      : pool_id(pid),
        assigned_worker_id(wid),
        task_count(0),
        is_enqueued(false),
        signal_fd_(-1),
        tid_(0) {
    active_.store(true);
  }
};

// Type alias for individual lanes with per-lane headers (moved outside
// TaskQueue class) Worker queues store Future<Task> objects directly
using TaskLane =
    hipc::multi_mpsc_ring_buffer<Future<Task>,
                                 CHI_MAIN_ALLOC_T>::ring_buffer_type;

/**
 * Simple wrapper around hipc::multi_mpsc_ring_buffer
 *
 * This wrapper adds custom enqueue and dequeue functions while maintaining
 * compatibility with existing code that expects the multi_mpsc_ring_buffer
 * interface.
 */
typedef hipc::multi_mpsc_ring_buffer<Future<Task>, CHI_MAIN_ALLOC_T> TaskQueue;

// ============================================================================
// RunContext (uses Future<Task> and TaskLane* - both must be complete above)
// ============================================================================

/**
 * Context passed to task execution methods
 *
 * RunContext holds the execution state for a task, including the coroutine
 * handle for C++20 stackless coroutines. When a task yields (co_await),
 * the coro_handle_ is used to resume execution later.
 */
struct RunContext {
  /** Coroutine handle for C++20 stackless coroutines */
  std::coroutine_handle<> coro_handle_;
  ThreadType thread_type_;      /**< Type of worker thread executing this task */
  u32 worker_id_;               /**< Worker ID executing this task */
  FullPtr<Task> task_;          /**< Task being executed by this context */
  bool is_yielded_;             /**< Task is waiting for completion */
  double yield_time_us_;        /**< Time in microseconds for task to yield */
  hshm::Timepoint block_start_; /**< Time when task was blocked (real time) */
  Container* container_;        /**< Current container being executed */
  TaskLane* lane_;              /**< Current lane being processed */
  ExecMode exec_mode_;          /**< Execution mode (kExec or kDynamicSchedule) */
  void* event_queue_;           /**< Pointer to worker's event queue */
  std::vector<PoolQuery>
      pool_queries_;            /**< Pool queries for task distribution */
  std::vector<FullPtr<Task>> subtasks_; /**< Replica tasks for this execution */
  u32 completed_replicas_;              /**< Count of completed replicas */
  u32 yield_count_;                     /**< Number of times task has yielded */
  Future<Task, CHI_MAIN_ALLOC_T>
      future_;               /**< Future for async completion tracking */
  bool destroy_in_end_task_; /**< Flag to indicate if task should be destroyed
                                in EndTask */
  std::atomic<bool> is_notified_; /**< Atomic flag to prevent duplicate event
                                     queue additions */
  double true_period_ns_;       /**< Original period from task->period_ns_ */
  bool did_work_;               /**< Whether task did work in last execution */

  RunContext()
      : coro_handle_(nullptr),
        thread_type_(kSchedWorker),
        worker_id_(0),
        is_yielded_(false),
        yield_time_us_(0.0),
        block_start_(),
        container_(nullptr),
        lane_(nullptr),
        exec_mode_(ExecMode::kExec),
        event_queue_(nullptr),
        completed_replicas_(0),
        yield_count_(0),
        destroy_in_end_task_(false),
        is_notified_(false),
        true_period_ns_(0.0),
        did_work_(false) {}

  /**
   * Move constructor
   */
  RunContext(RunContext&& other) noexcept
      : coro_handle_(other.coro_handle_),
        thread_type_(other.thread_type_),
        worker_id_(other.worker_id_),
        task_(std::move(other.task_)),
        is_yielded_(other.is_yielded_),
        yield_time_us_(other.yield_time_us_),
        block_start_(other.block_start_),
        container_(other.container_),
        lane_(other.lane_),
        exec_mode_(other.exec_mode_),
        event_queue_(other.event_queue_),
        pool_queries_(std::move(other.pool_queries_)),
        subtasks_(std::move(other.subtasks_)),
        completed_replicas_(other.completed_replicas_),
        yield_count_(other.yield_count_),
        future_(std::move(other.future_)),
        destroy_in_end_task_(other.destroy_in_end_task_),
        is_notified_(other.is_notified_.load()),
        true_period_ns_(other.true_period_ns_),
        did_work_(other.did_work_) {
    other.coro_handle_ = nullptr;
    other.event_queue_ = nullptr;
  }

  /**
   * Move assignment operator
   */
  RunContext& operator=(RunContext&& other) noexcept {
    if (this != &other) {
      coro_handle_ = other.coro_handle_;
      thread_type_ = other.thread_type_;
      worker_id_ = other.worker_id_;
      task_ = std::move(other.task_);
      is_yielded_ = other.is_yielded_;
      yield_time_us_ = other.yield_time_us_;
      block_start_ = other.block_start_;
      container_ = other.container_;
      lane_ = other.lane_;
      exec_mode_ = other.exec_mode_;
      event_queue_ = other.event_queue_;
      pool_queries_ = std::move(other.pool_queries_);
      subtasks_ = std::move(other.subtasks_);
      completed_replicas_ = other.completed_replicas_;
      yield_count_ = other.yield_count_;
      future_ = std::move(other.future_);
      destroy_in_end_task_ = other.destroy_in_end_task_;
      is_notified_.store(other.is_notified_.load());
      true_period_ns_ = other.true_period_ns_;
      did_work_ = other.did_work_;
      other.coro_handle_ = nullptr;
      other.event_queue_ = nullptr;
    }
    return *this;
  }

  // Delete copy constructor and copy assignment
  RunContext(const RunContext&) = delete;
  RunContext& operator=(const RunContext&) = delete;

  /**
   * Clear all STL containers for reuse
   * Does not touch pointers or primitive types
   */
  void Clear() {
    pool_queries_.clear();
    subtasks_.clear();
    completed_replicas_ = 0;
    yield_time_us_ = 0.0;
    block_start_ = hshm::Timepoint();
    yield_count_ = 0;
    is_notified_.store(false);
    true_period_ns_ = 0.0;
    did_work_ = false;
  }
};

// ============================================================================
// Future::await_suspend_impl implementation (must be after RunContext definition)
// ============================================================================

template <typename TaskT, typename AllocT>
bool Future<TaskT, AllocT>::await_suspend_impl(std::coroutine_handle<> handle) noexcept {
  // Get RunContext from the current worker's thread-local storage
  // Uses helper function to avoid circular dependency with worker.h
  RunContext* run_ctx = GetCurrentRunContextFromWorker();

  if (!run_ctx) {
    // No RunContext available, don't suspend
    HLOG(kWarning, "Future::await_suspend: run_ctx is null, not suspending!");
    return false;
  }
  // Store parent context for resumption tracking
  SetParentTask(run_ctx);
  // Store coroutine handle in RunContext for worker to resume
  run_ctx->coro_handle_ = handle;
  run_ctx->is_yielded_ = true;
  run_ctx->yield_time_us_ = 0.0;
  return true;  // Suspend the coroutine
}

// ============================================================================
// TaskResume and YieldAwaiter (must be after RunContext for member access)
// ============================================================================

/**
 * TaskResume - Coroutine return type for runtime methods
 *
 * This lightweight class serves as the return type for ChiMod Run methods
 * that use C++20 coroutines. It holds a coroutine handle and provides
 * methods to resume and check completion status.
 */
class TaskResume {
 public:
  /**
   * Promise type for C++20 coroutine machinery
   *
   * The promise_type defines how the coroutine behaves at various points:
   * - initial_suspend: suspend immediately (lazy start)
   * - final_suspend: resume caller if exists, else suspend
   * - return_void: coroutines return void
   */
  struct promise_type {
    /** Pointer to the RunContext for this coroutine */
    RunContext* run_ctx_ = nullptr;
    /** Handle to the caller coroutine (for nested coroutine support) */
    std::coroutine_handle<> caller_handle_ = nullptr;

    /**
     * Create the TaskResume object from this promise
     * @return TaskResume wrapping the coroutine handle
     */
    TaskResume get_return_object() {
      return TaskResume(
          std::coroutine_handle<promise_type>::from_promise(*this));
    }

    /**
     * Suspend immediately on coroutine start (lazy evaluation)
     * @return Always suspend
     */
    std::suspend_always initial_suspend() noexcept { return {}; }

    /**
     * Awaiter for final_suspend that resumes the caller coroutine
     * if one exists, enabling nested coroutine support.
     */
    struct FinalAwaiter {
      std::coroutine_handle<> caller_;

      bool await_ready() noexcept { return false; }

      /**
       * Resume the caller coroutine if it exists, otherwise use noop
       * @return Handle to resume (caller or noop)
       */
      std::coroutine_handle<> await_suspend(std::coroutine_handle<>) noexcept {
        return caller_ ? caller_ : std::noop_coroutine();
      }

      void await_resume() noexcept {}
    };

    /**
     * Suspend at final suspension point and resume caller if exists
     * @return FinalAwaiter that handles resuming the caller
     */
    FinalAwaiter final_suspend() noexcept {
      return FinalAwaiter{caller_handle_};
    }

    /**
     * Handle void return from coroutine
     */
    void return_void() {}

    /**
     * Handle unhandled exceptions by terminating
     */
    void unhandled_exception() { std::terminate(); }

    /**
     * Set the RunContext for this coroutine
     * @param ctx Pointer to RunContext
     */
    void set_run_context(RunContext* ctx) { run_ctx_ = ctx; }

    /**
     * Get the RunContext for this coroutine
     * @return Pointer to RunContext
     */
    RunContext* get_run_context() const { return run_ctx_; }

    /**
     * Set the caller coroutine handle
     * @param caller Handle to the caller coroutine
     */
    void set_caller(std::coroutine_handle<> caller) { caller_handle_ = caller; }
  };

  using handle_type = std::coroutine_handle<promise_type>;

 private:
  /** The coroutine handle */
  handle_type handle_;
  /** Stored caller handle for await_resume to update run_ctx */
  std::coroutine_handle<> caller_handle_ = nullptr;

 public:
  /**
   * Construct from coroutine handle
   * @param h The coroutine handle
   */
  explicit TaskResume(handle_type h) : handle_(h) {}

  /**
   * Default constructor - null handle
   */
  TaskResume() : handle_(nullptr) {}

  /**
   * Move constructor
   * @param other TaskResume to move from
   */
  TaskResume(TaskResume&& other) noexcept
      : handle_(other.handle_), caller_handle_(other.caller_handle_) {
    other.handle_ = nullptr;
    other.caller_handle_ = nullptr;
  }

  /**
   * Move assignment operator
   * @param other TaskResume to move from
   * @return Reference to this
   */
  TaskResume& operator=(TaskResume&& other) noexcept {
    if (this != &other) {
      if (handle_) {
        handle_.destroy();
      }
      handle_ = other.handle_;
      caller_handle_ = other.caller_handle_;
      other.handle_ = nullptr;
      other.caller_handle_ = nullptr;
    }
    return *this;
  }

  /** Disable copy constructor */
  TaskResume(const TaskResume&) = delete;

  /** Disable copy assignment */
  TaskResume& operator=(const TaskResume&) = delete;

  /**
   * Destructor - destroys the coroutine handle
   */
  ~TaskResume() {
    if (handle_) {
      handle_.destroy();
    }
  }

  /**
   * Get the coroutine handle
   * @return The coroutine handle
   */
  handle_type get_handle() const { return handle_; }

  /**
   * Check if coroutine is done
   * @return True if coroutine has completed
   */
  bool done() const { return handle_ && handle_.done(); }

  /**
   * Resume the coroutine
   */
  void resume() {
    if (handle_ && !handle_.done()) {
      handle_.resume();
    }
  }

  /**
   * Destroy the coroutine handle manually
   */
  void destroy() {
    if (handle_) {
      handle_.destroy();
      handle_ = nullptr;
    }
  }

  /**
   * Check if the handle is valid
   * @return True if handle is not null
   */
  explicit operator bool() const { return handle_ != nullptr; }

  /**
   * Release ownership of the handle without destroying it
   * @return The coroutine handle
   */
  handle_type release() {
    handle_type h = handle_;
    handle_ = nullptr;
    return h;
  }

  // ============================================================
  // Awaiter interface - allows TaskResume to be used with co_await
  // ============================================================

  /**
   * Check if the coroutine is already done
   * @return True if coroutine completed, false otherwise
   */
  bool await_ready() const noexcept { return handle_ && handle_.done(); }

  /**
   * Suspend the calling coroutine and run this one to completion or suspension
   *
   * This runs the inner coroutine (TaskResume) until it either:
   * - Completes (co_return)
   * - Suspends at a co_await (is_yielded_ is true)
   *
   * IMPORTANT: Propagates the RunContext from the caller to the inner coroutine
   * so that nested co_await calls on Futures work correctly.
   *
   * When the inner coroutine suspends on a Future, the caller is also
   * suspended. When the awaited Future completes, the inner coroutine's handle
   * (stored in run_ctx->coro_handle_) is resumed. The await_resume of this
   * TaskResume will then continue running the inner coroutine to completion.
   *
   * @tparam PromiseT The promise type of the calling coroutine
   * @param caller_handle The coroutine handle of the caller
   * @return True if we should suspend (inner suspended), false if inner
   * completed
   */
  template <typename PromiseT>
  bool await_suspend(std::coroutine_handle<PromiseT> caller_handle) noexcept {
    if (!handle_) {
      return false;  // Nothing to run, don't suspend
    }

    // Store caller handle for await_resume to use when updating run_ctx
    caller_handle_ = caller_handle;

    // CRITICAL: Propagate RunContext from caller to inner coroutine
    // This allows nested co_await on Futures to properly suspend
    RunContext* caller_run_ctx = caller_handle.promise().get_run_context();
    if (caller_run_ctx) {
      handle_.promise().set_run_context(caller_run_ctx);
    }

    // NOTE: We do NOT set caller_handle in inner's promise yet!
    // If the inner coroutine completes synchronously during resume(),
    // final_suspend would try to resume caller while we're still inside
    // await_suspend, causing undefined behavior. We only set it after
    // confirming suspension.

    // Resume the inner coroutine
    handle_.resume();

    // Check if inner coroutine is done
    if (handle_.done()) {
      // Inner completed synchronously, destroy it
      handle_.destroy();
      handle_ = nullptr;
      return false;  // Don't suspend caller
    }

    // Inner coroutine suspended (on co_await Future or yield)
    // NOW it's safe to set caller_handle - the inner will complete
    // asynchronously and final_suspend will properly resume the caller
    handle_.promise().set_caller(caller_handle);

    // The inner's handle is now stored in run_ctx->coro_handle_ by
    // Future::await_suspend When the awaited Future completes, worker will
    // resume inner via run_ctx->coro_handle_ When inner eventually completes,
    // final_suspend will resume the caller (this coroutine)
    return true;
  }

  /**
   * Resume after await - cleanup inner coroutine and update run_ctx
   *
   * This is called when the caller is resumed after the inner coroutine
   * completes. The inner's final_suspend resumes the caller, which triggers
   * this method. We need to:
   * 1. Get run_ctx from inner's promise (before destroying)
   * 2. Destroy inner's handle (it's done)
   * 3. Update run_ctx->coro_handle_ to caller's handle so subsequent events
   *    properly resume the caller (outer) coroutine
   */
  void await_resume() noexcept {
    // Get run_ctx from inner's promise before destroying
    RunContext* run_ctx = nullptr;
    if (handle_) {
      run_ctx = handle_.promise().get_run_context();
      // Inner coroutine is done (final_suspend just resumed us), destroy it
      handle_.destroy();
      handle_ = nullptr;
    }

    // Update run_ctx->coro_handle_ to caller's handle
    // This ensures if caller suspends again on another co_await,
    // or if caller completes, the worker can properly handle it
    if (run_ctx != nullptr && caller_handle_) {
      run_ctx->coro_handle_ = caller_handle_;
    }
  }
};

/**
 * YieldAwaiter - Awaitable for yielding control in coroutines
 *
 * This class implements the awaitable interface for cooperative yielding
 * within ChiMod runtime coroutines. It allows tasks to yield control
 * back to the worker with an optional delay before resumption.
 *
 * Usage:
 *   co_await chi::yield();       // Yield immediately
 *   co_await chi::yield(25.0);   // Yield with 25 microsecond delay
 */
class YieldAwaiter {
 private:
  /** Time in microseconds to delay before resumption */
  double yield_time_us_;

 public:
  /**
   * Construct a YieldAwaiter with optional delay
   * @param us Microseconds to delay before resumption (default: 0)
   */
  explicit YieldAwaiter(double us = 0.0) : yield_time_us_(us) {}

  /**
   * Yield is never immediately ready - always suspends
   * @return Always false (always suspend)
   */
  bool await_ready() const noexcept { return false; }

  /**
   * Suspend the coroutine and mark for yielded resumption
   *
   * @tparam PromiseT The promise type of the calling coroutine
   * @param handle The coroutine handle to resume after yield
   * @return True to suspend, false if no RunContext available
   */
  template <typename PromiseT>
  bool await_suspend(std::coroutine_handle<PromiseT> handle) noexcept {
    auto* run_ctx = handle.promise().get_run_context();
    if (!run_ctx) {
      // No RunContext available, don't suspend
      return false;
    }
    // Store coroutine handle in RunContext for worker to resume
    run_ctx->coro_handle_ = handle;
    run_ctx->is_yielded_ = true;
    run_ctx->yield_time_us_ = yield_time_us_;
    return true;  // Suspend the coroutine
  }

  /**
   * Resume after yield - nothing to return
   */
  void await_resume() noexcept {}
};

/**
 * Create a YieldAwaiter for cooperative yielding in coroutines
 *
 * This function provides a clean syntax for yielding control within
 * ChiMod runtime coroutines.
 *
 * @param us Microseconds to delay before resumption (default: 0)
 * @return YieldAwaiter object that can be co_awaited
 *
 * Usage:
 *   co_await chi::yield();       // Yield immediately
 *   co_await chi::yield(25.0);   // Yield with 25 microsecond delay
 */
inline YieldAwaiter yield(double us = 0.0) { return YieldAwaiter(us); }

// Cleanup macros
#undef CLASS_NAME
#undef CLASS_NEW_ARGS

/**
 * SFINAE-based compile-time detection and invocation for Aggregate method
 * Usage: CHI_AGGREGATE_OR_COPY(origin_ptr, replica_ptr)
 */
namespace detail {
// Primary template - assumes no Aggregate method, calls Copy
template <typename T, typename = void>
struct aggregate_or_copy {
  static void call(hipc::FullPtr<T> origin, hipc::FullPtr<T> replica) {
    origin->Copy(replica);
  }
};

// Specialization for types with Aggregate method - calls Aggregate
template <typename T>
struct aggregate_or_copy<T, std::void_t<decltype(std::declval<T*>()->Aggregate(
                                std::declval<hipc::FullPtr<T>>()))>> {
  static void call(hipc::FullPtr<T> origin, hipc::FullPtr<T> replica) {
    origin->Aggregate(replica);
  }
};
}  // namespace detail

// Macro for convenient usage - automatically dispatches to Aggregate or Copy
#define CHI_AGGREGATE_OR_COPY(origin_ptr, replica_ptr)         \
  chi::detail::aggregate_or_copy<typename std::remove_pointer< \
      decltype((origin_ptr).ptr_)>::type>::call((origin_ptr), (replica_ptr))

}  // namespace chi

#endif  // CHIMAERA_INCLUDE_CHIMAERA_TASK_H_
