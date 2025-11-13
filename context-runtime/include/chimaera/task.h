#ifndef CHIMAERA_INCLUDE_CHIMAERA_TASK_H_
#define CHIMAERA_INCLUDE_CHIMAERA_TASK_H_

#include <atomic>
#include <boost/context/detail/fcontext.hpp>
#include <sstream>
#include <vector>

#include "chimaera/pool_query.h"
#include "chimaera/task_queue.h"
#include "chimaera/types.h"

// Include cereal for serialization
#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>

// TaskQueue types are now available via include

namespace chi {

// Forward declarations
class Task;
class Container;
struct RunContext;

/**
 * Task statistics for I/O and compute time tracking
 * Used to route tasks to appropriate worker groups
 */
struct TaskStat {
  size_t io_size_{0};    /**< I/O size in bytes */
  size_t compute_{0};    /**< Normalized compute time in microseconds */
};

// Define macros for container template
#define CLASS_NAME Task
#define CLASS_NEW_ARGS

/**
 * Base task class for Chimaera distributed execution
 *
 * Inherits from hipc::ShmContainer to support shared memory operations.
 * All tasks represent C++ functions similar to RPCs that can be executed
 * across the distributed system.
 */
class Task : public hipc::ShmContainer {
public:
  IN PoolId pool_id_;       /**< Pool identifier for task execution */
  IN TaskId task_id_;       /**< Task identifier for task routing */
  IN PoolQuery pool_query_; /**< Pool query for execution location */
  IN MethodId method_;      /**< Method identifier for task type */
  IN ibitfield task_flags_; /**< Task properties and flags */
  IN double period_ns_;     /**< Period in nanoseconds for periodic tasks */
  IN RunContext *run_ctx_; /**< Pointer to runtime context for task execution */
  std::atomic<u32> is_complete_; /**< Atomic flag indicating task completion
                                   (0=not complete, 1=complete) */
  std::atomic<u32>
      return_code_; /**< Task return code (0=success, non-zero=error) */
  OUT std::atomic<ContainerId> completer_; /**< Container ID that completed this task */
  TaskStat stat_;   /**< Task statistics for I/O and compute tracking */

  /**
   * SHM default constructor
   */
  explicit Task(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc)
      : hipc::ShmContainer() {
    SetNull();
  }

  /**
   * Emplace constructor with task initialization
   */
  explicit Task(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc,
                const TaskId &task_id, const PoolId &pool_id,
                const PoolQuery &pool_query, const MethodId &method)
      : hipc::ShmContainer() {
    // Initialize task
    task_id_ = task_id;
    pool_id_ = pool_id;
    method_ = method;
    task_flags_.SetBits(0);
    pool_query_ = pool_query;
    period_ns_ = 0.0;
    run_ctx_ = nullptr;
    is_complete_.store(0); // Initialize as not complete
    return_code_.store(0); // Initialize as success
    completer_.store(0); // Initialize as null (0 is invalid container ID)
  }

  /**
   * Copy constructor
   */
  HSHM_CROSS_FUN explicit Task(const Task &other) {
    SetNull();
    shm_strong_copy_main(other);
  }

  /**
   * Strong copy implementation
   */
  template <typename ContainerT>
  HSHM_CROSS_FUN void shm_strong_copy_main(const ContainerT &other) {
    pool_id_ = other.pool_id_;
    task_id_ = other.task_id_;
    pool_query_ = other.pool_query_;
    method_ = other.method_;
    task_flags_ = other.task_flags_;
    period_ns_ = other.period_ns_;
    run_ctx_ = other.run_ctx_;
    return_code_.store(other.return_code_.load());
    completer_.store(other.completer_.load());
    stat_ = other.stat_;
    // Explicitly initialize as not complete for copied tasks
    is_complete_.store(0);
  }

  /**
   * Copy from another task (assumes this task is already constructed)
   * @param other Pointer to the source task to copy from
   */
  HSHM_CROSS_FUN void Copy(const hipc::FullPtr<Task> &other) {
    pool_id_ = other->pool_id_;
    task_id_ = other->task_id_;
    pool_query_ = other->pool_query_;
    method_ = other->method_;
    task_flags_ = other->task_flags_;
    period_ns_ = other->period_ns_;
    run_ctx_ = other->run_ctx_;
    return_code_.store(other->return_code_.load());
    completer_.store(other->completer_.load());
    stat_ = other->stat_;
    // Explicitly initialize as not complete for copied tasks
    is_complete_.store(0);
  }

  /**
   * Move constructor
   */
  HSHM_CROSS_FUN Task(Task &&other) {
    shm_move_op<false>(
        HSHM_MEMORY_MANAGER->GetDefaultAllocator<CHI_MAIN_ALLOC_T>(),
        std::move(other));
  }

  template <bool IS_ASSIGN>
  HSHM_CROSS_FUN void
  shm_move_op(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc,
              Task &&other) noexcept {
    // For simplified Task class, just copy the data
    shm_strong_copy_main(other);
    other.SetNull();
  }

  /**
   * IsNull check
   */
  HSHM_INLINE_CROSS_FUN bool IsNull() const {
    return false; // Base task is never null
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
    run_ctx_ = nullptr;
    is_complete_.store(0); // Initialize as not complete
    return_code_.store(0); // Initialize as success
    completer_.store(0); // Initialize as null (0 is invalid container ID)
    stat_.io_size_ = 0;
    stat_.compute_ = 0;
  }

  /**
   * Destructor implementation
   */
  HSHM_INLINE_CROSS_FUN void shm_destroy_main() {
    // Base task has no dynamic resources to clean up
  }

  /**
   * Virtual destructor
   */
  HSHM_CROSS_FUN virtual ~Task() = default;

  /**
   * Wait for task completion (blocking)
   * @param block_time_us Blocking duration in microseconds (default: 0.0 for cooperative tasks)
   * @param from_yield If true, do not add subtasks to RunContext (default:
   * false)
   */
  HSHM_CROSS_FUN void Wait(double block_time_us = 0.0,
                           bool from_yield = false);

  /**
   * Check if task is complete
   * @return true if task is complete, false otherwise
   */
  HSHM_CROSS_FUN bool IsComplete() const;

  /**
   * Yield execution back to worker by waiting for task completion
   * @param block_time_us Blocking duration in microseconds (default: 0.0 for cooperative tasks)
   */
  HSHM_CROSS_FUN void Yield(double block_time_us = 0.0);

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
   * Get shared memory pointer representation
   */
  HSHM_CROSS_FUN hipc::Pointer GetShmPointer() const {
    return hipc::Pointer::GetNull();
  }

  /**
   * Get the allocator (stub implementation for compatibility)
   */
  HSHM_CROSS_FUN hipc::CtxAllocator<CHI_MAIN_ALLOC_T> GetAllocator() const {
    return HSHM_MEMORY_MANAGER->GetDefaultAllocator<CHI_MAIN_ALLOC_T>();
  }

  /**
   * Get context allocator (stub implementation for compatibility)
   */
  HSHM_CROSS_FUN hipc::CtxAllocator<CHI_MAIN_ALLOC_T> GetCtxAllocator() const {
    return HSHM_MEMORY_MANAGER->GetDefaultAllocator<CHI_MAIN_ALLOC_T>();
  }

  /**
   * Serialize data structures to chi::ipc::string using cereal
   * @param alloc Context allocator for memory management
   * @param output_str The string to store serialized data
   * @param args The arguments to serialize
   */
  template <typename... Args>
  static void Serialize(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc,
                        hipc::string &output_str, const Args &...args) {
    std::ostringstream os;
    cereal::BinaryOutputArchive archive(os);
    archive(args...);

    std::string serialized = os.str();
    output_str = hipc::string(alloc, serialized);
  }

  /**
   * Deserialize data structure from chi::ipc::string using cereal
   * @param input_str The string containing serialized data
   * @return The deserialized object
   */
  template <typename OutT>
  static OutT Deserialize(const hipc::string &input_str) {
    std::string data = input_str.str();
    std::istringstream is(data);
    cereal::BinaryInputArchive archive(is);

    OutT result;
    archive(result);
    return result;
  }

  /**
   * Serialize base task fields for incoming network transfer (IN and INOUT
   * parameters) This method serializes the common task fields that are shared
   * across all task types. Called automatically by archives when they detect
   * Task inheritance.
   * @param ar Archive to serialize to
   */
  template <typename Archive> void BaseSerializeIn(Archive &ar) {
    // Handle atomic return_code_ by loading/storing its value
    u32 return_code_value = return_code_.load();
    ar(pool_id_, task_id_, pool_query_, method_, task_flags_, period_ns_,
       return_code_value);
    return_code_.store(return_code_value);
  }

  /**
   * Serialize base task fields for outgoing network transfer (OUT and INOUT
   * parameters) This method serializes the common task fields that are shared
   * across all task types. Called automatically by archives when they detect
   * Task inheritance.
   * @param ar Archive to serialize to
   */
  template <typename Archive> void BaseSerializeOut(Archive &ar) {
    // Only serialize OUT fields - do NOT re-serialize IN fields
    // (pool_id_, task_id_, pool_query_, method_, task_flags_, period_ns_ are
    // all IN) Only return_code_ and completer_ are OUT fields that need to be sent back
    u32 return_code_value = return_code_.load();
    ContainerId completer_value = completer_.load();
    ar(return_code_value, completer_value);
    return_code_.store(return_code_value);
    completer_.store(completer_value);
  }

  /**
   * Serialize task for incoming network transfer (IN and INOUT parameters)
   * This method should be implemented by each specific task type.
   * Archives automatically call BaseSerializeIn first, then this method.
   * @param ar Archive to serialize to
   */
  template <typename Archive> void SerializeIn(Archive &ar) {
    // Base implementation does nothing - derived classes override to serialize
    // their IN/INOUT fields
  }

  /**
   * Serialize task for outgoing network transfer (OUT and INOUT parameters)
   * This method should be implemented by each specific task type.
   * Archives automatically call BaseSerializeOut first, then this method.
   * @param ar Archive to serialize to
   */
  template <typename Archive> void SerializeOut(Archive &ar) {
    // Base implementation does nothing - derived classes override to serialize
    // their OUT/INOUT fields
  }

  /**
   * Yield execution back to worker (runtime) or sleep briefly (non-runtime)
   * In runtime: Jumps back to worker fiber context with estimated completion
   * time Outside runtime: Uses SleepForUs when worker is null
   */
  HSHM_CROSS_FUN void YieldBase();

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
   * Set the completer container ID (which container completed this task)
   * @param completer Container ID to set
   */
  HSHM_CROSS_FUN void SetCompleter(ContainerId completer) {
    completer_.store(completer);
  }

  /**
   * Base aggregate method - propagates return codes from replica tasks
   * Sets this task's return code to the replica's return code if replica has non-zero return code
   * @param replica_task The replica task to aggregate from
   */
  HSHM_CROSS_FUN void Aggregate(const hipc::FullPtr<Task> &replica_task);

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
  kExec = 0,              /**< Normal task execution */
  kDynamicSchedule = 1    /**< Dynamic scheduling - route after execution */
};

/**
 * Context passed to task execution methods
 */
struct RunContext {
  void *stack_ptr; // Stack pointer (positioned for boost::context based on
                   // stack growth)
  void *stack_base_for_free; // Original malloc pointer for freeing
  size_t stack_size;
  ThreadType thread_type;
  u32 worker_id;
  FullPtr<Task> task; // Task being executed by this context
  bool is_blocked;    // Task is waiting for completion
  double est_load;    // Estimated time until task should wake up (microseconds)
  double block_time_us;  // Time in microseconds for task to block
  hshm::Timepoint block_start; // Time when task was blocked (real time)
  boost::context::detail::transfer_t
      yield_context; // boost::context transfer from FiberExecutionFunction
                     // parameter - used for yielding back
  boost::context::detail::transfer_t
      resume_context;    // boost::context transfer for resuming into yield
                         // function
  Container *container;  // Current container being executed
  TaskLane *lane;        // Current lane being processed
  ExecMode exec_mode;    // Execution mode (kExec or kDynamicSchedule)
  std::vector<FullPtr<Task>>
      waiting_for_tasks; // Tasks this task is waiting for completion
  std::vector<PoolQuery> pool_queries;  // Pool queries for task distribution
  std::vector<FullPtr<Task>> subtasks_; // Replica tasks for this execution
  std::atomic<u32> completed_replicas_; // Count of completed replicas
  u32 block_count_;  // Number of times task has been blocked

  RunContext()
      : stack_ptr(nullptr), stack_base_for_free(nullptr), stack_size(0),
        thread_type(kSchedWorker), worker_id(0), is_blocked(false),
        est_load(0.0), block_time_us(0.0), block_start(), yield_context{}, resume_context{},
        container(nullptr), lane(nullptr), exec_mode(ExecMode::kExec),
        completed_replicas_(0), block_count_(0) {}

  /**
   * Move constructor - required because of atomic member
   */
  RunContext(RunContext &&other) noexcept
      : stack_ptr(other.stack_ptr),
        stack_base_for_free(other.stack_base_for_free),
        stack_size(other.stack_size), thread_type(other.thread_type),
        worker_id(other.worker_id), task(std::move(other.task)),
        is_blocked(other.is_blocked), est_load(other.est_load),
        block_time_us(other.block_time_us), block_start(other.block_start),
        yield_context(other.yield_context), resume_context(other.resume_context),
        container(other.container), lane(other.lane),
        exec_mode(other.exec_mode),
        waiting_for_tasks(std::move(other.waiting_for_tasks)),
        pool_queries(std::move(other.pool_queries)),
        subtasks_(std::move(other.subtasks_)),
        completed_replicas_(other.completed_replicas_.load()),
        block_count_(other.block_count_) {}

  /**
   * Move assignment operator - required because of atomic member
   */
  RunContext &operator=(RunContext &&other) noexcept {
    if (this != &other) {
      stack_ptr = other.stack_ptr;
      stack_base_for_free = other.stack_base_for_free;
      stack_size = other.stack_size;
      thread_type = other.thread_type;
      worker_id = other.worker_id;
      task = std::move(other.task);
      is_blocked = other.is_blocked;
      est_load = other.est_load;
      block_time_us = other.block_time_us;
      block_start = other.block_start;
      yield_context = other.yield_context;
      resume_context = other.resume_context;
      container = other.container;
      lane = other.lane;
      exec_mode = other.exec_mode;
      waiting_for_tasks = std::move(other.waiting_for_tasks);
      pool_queries = std::move(other.pool_queries);
      subtasks_ = std::move(other.subtasks_);
      completed_replicas_.store(other.completed_replicas_.load());
      block_count_ = other.block_count_;
    }
    return *this;
  }

  // Delete copy constructor and copy assignment
  RunContext(const RunContext &) = delete;
  RunContext &operator=(const RunContext &) = delete;

  /**
   * Clear all STL containers for reuse
   * Does not touch pointers or primitive types
   */
  void Clear() {
    waiting_for_tasks.clear();
    pool_queries.clear();
    subtasks_.clear();
    completed_replicas_.store(0);
    est_load = 0.0;
    block_time_us = 0.0;
    block_start = hshm::Timepoint();
    block_count_ = 0;
  }

  /**
   * Check if all subtasks this task is waiting for are completed
   * @return true if all subtasks are completed, false otherwise
   */
  bool AreSubtasksCompleted() const {
    // Check each task in the waiting_for_tasks vector
    for (const auto &waiting_task : waiting_for_tasks) {
      if (!waiting_task.IsNull()) {
        // Check if the waiting task is completed using atomic flag
        if (waiting_task->is_complete_.load() == 0) {
          return false; // Found a subtask that's not completed yet
        }
      }
    }
    return true; // All subtasks are completed (or no subtasks)
  }
};

// Cleanup macros
#undef CLASS_NAME
#undef CLASS_NEW_ARGS

/**
 * SFINAE-based compile-time detection and invocation for Aggregate method
 * Usage: CHI_AGGREGATE_OR_COPY(origin_ptr, replica_ptr)
 */
namespace detail {
// Primary template - assumes no Aggregate method, calls Copy
template <typename T, typename = void> struct aggregate_or_copy {
  static void call(hipc::FullPtr<T> origin, hipc::FullPtr<T> replica) {
    origin->Copy(replica);
  }
};

// Specialization for types with Aggregate method - calls Aggregate
template <typename T>
struct aggregate_or_copy<T, std::void_t<decltype(std::declval<T *>()->Aggregate(
                                std::declval<hipc::FullPtr<T>>()))>> {
  static void call(hipc::FullPtr<T> origin, hipc::FullPtr<T> replica) {
    origin->Aggregate(replica);
  }
};
} // namespace detail

// Macro for convenient usage - automatically dispatches to Aggregate or Copy
#define CHI_AGGREGATE_OR_COPY(origin_ptr, replica_ptr)                         \
  chi::detail::aggregate_or_copy<typename std::remove_pointer<                 \
      decltype((origin_ptr).ptr_)>::type>::call((origin_ptr), (replica_ptr))

} // namespace chi

// Namespace alias for convenience - removed to avoid circular reference

#endif // CHIMAERA_INCLUDE_CHIMAERA_TASK_H_