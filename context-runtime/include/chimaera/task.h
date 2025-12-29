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

// TaskQueue types are now available via include

// Forward declare chi::priv::string for cereal support
namespace hshm::priv {
template <typename T, typename AllocT, size_t SmallSize>
class basic_string;
}


namespace chi {

// Forward declarations
class Task;
class Container;
struct RunContext;
template<typename AllocT> class FutureShm;

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
  IN RunContext *run_ctx_; /**< Pointer to runtime context for task execution */
  OUT u32 return_code_; /**< Task return code (0=success, non-zero=error) */
  OUT ContainerId completer_; /**< Container ID that completed this task */
  TaskStat stat_;   /**< Task statistics for I/O and compute tracking */

  /**
   * Default constructor
   */
  Task() {
    SetNull();
  }

  /**
   * Emplace constructor with task initialization
   */
  explicit Task(const TaskId &task_id, const PoolId &pool_id,
                const PoolQuery &pool_query, const MethodId &method) {
    // Initialize task
    task_id_ = task_id;
    pool_id_ = pool_id;
    method_ = method;
    task_flags_.SetBits(0);
    pool_query_ = pool_query;
    period_ns_ = 0.0;
    run_ctx_ = nullptr;
    return_code_ = 0; // Initialize as success
    completer_ = 0; // Initialize as null (0 is invalid container ID)
  }

  /**
   * Copy from another task (assumes this task is already constructed)
   *
   * IMPORTANT: Derived classes that override Copy MUST call Task::Copy(other)
   * first before copying their own fields.
   *
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
    return_code_ = other->return_code_;
    completer_ = other->completer_;
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
    run_ctx_ = nullptr;
    return_code_ = 0; // Initialize as success
    completer_ = 0; // Initialize as null (0 is invalid container ID)
    stat_.io_size_ = 0;
    stat_.compute_ = 0;
  }

  /**
   * Wait for task completion (blocking)
   * Uses Future's is_complete flag to determine when waiting is done
   * @param is_complete Reference to atomic completion flag from Future
   * @param yield_time_us Yield duration in microseconds (default: 0.0 for cooperative tasks)
   */
  HSHM_CROSS_FUN void Wait(std::atomic<u32>& is_complete, double yield_time_us = 0.0);

  /**
   * Yield execution back to worker by waiting for task completion
   * @param yield_time_us Yield duration in microseconds (default: 0.0 for cooperative tasks)
   */
  HSHM_CROSS_FUN void Yield(double yield_time_us = 0.0);

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
   * Serialize data structures to chi::ipc::string using cereal
   * @param alloc Context allocator for memory management
   * @param output_str The string to store serialized data
   * @param args The arguments to serialize
   */
  template <typename... Args>
  static void Serialize(AllocT* alloc,
                        chi::priv::string &output_str, const Args &...args) {
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
  static OutT Deserialize(const chi::priv::string &input_str) {
    std::string data = input_str.str();
    std::istringstream is(data);
    cereal::BinaryInputArchive archive(is);

    OutT result;
    archive(result);
    return result;
  }

  /**
   * Serialize task for incoming network transfer (IN and INOUT parameters)
   * This method serializes the base Task fields first, then should be overridden
   * by derived classes to serialize their specific fields.
   *
   * IMPORTANT: Derived classes MUST call Task::SerializeIn(ar) first before
   * serializing their own fields.
   *
   * @param ar Archive to serialize to
   */
  template <typename Archive> void SerializeIn(Archive &ar) {
    // Serialize base Task fields (IN and INOUT parameters)
    ar(pool_id_, task_id_, pool_query_, method_, task_flags_, period_ns_,
       return_code_);
  }

  /**
   * Serialize task for outgoing network transfer (OUT and INOUT parameters)
   * This method serializes the base Task OUT fields first, then should be overridden
   * by derived classes to serialize their specific OUT fields.
   *
   * IMPORTANT: Derived classes MUST call Task::SerializeOut(ar) first before
   * serializing their own OUT fields.
   *
   * @param ar Archive to serialize to
   */
  template <typename Archive> void SerializeOut(Archive &ar) {
    // Serialize base Task OUT fields only
    // Only serialize OUT fields - do NOT re-serialize IN fields
    // (pool_id_, task_id_, pool_query_, method_, task_flags_, period_ns_ are all IN)
    // Only return_code_ and completer_ are OUT fields that need to be sent back
    ar(return_code_, completer_);
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
  HSHM_CROSS_FUN u32 GetReturnCode() const { return return_code_; }

  /**
   * Set the task return code
   * @param return_code Return code to set (0=success, non-zero=error)
   */
  HSHM_CROSS_FUN void SetReturnCode(u32 return_code) {
    return_code_ = return_code;
  }

  /**
   * Get the completer container ID (which container completed this task)
   * @return Container ID that completed this task
   */
  HSHM_CROSS_FUN ContainerId GetCompleter() const { return completer_; }

  /**
   * Set the completer container ID (which container completed this task)
   * @param completer Container ID to set
   */
  HSHM_CROSS_FUN void SetCompleter(ContainerId completer) {
    completer_ = completer;
  }

  /**
   * Base aggregate method - propagates return codes and completer from replica tasks
   * Sets this task's return code to the replica's return code if replica has non-zero return code
   * Accepts any task type that inherits from Task
   *
   * IMPORTANT: Derived classes that override Aggregate MUST call Task::Aggregate(replica_task)
   * first before aggregating their own fields.
   *
   * @param replica_task The replica task to aggregate from
   */
  template<typename TaskT>
  HSHM_CROSS_FUN void Aggregate(const hipc::FullPtr<TaskT> &replica_task) {
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
    HLOG(kDebug, "[COMPLETER] Aggregated task {} with completer {}", task_id_, GetCompleter());
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
  bool is_yielded_;    // Task is waiting for completion
  double est_load;    // Estimated time until task should wake up (microseconds)
  double yield_time_us_;  // Time in microseconds for task to yield
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
  void *event_queue_;    // Pointer to worker's event queue (mpsc_queue<RunContext*>)
  std::vector<PoolQuery> pool_queries;  // Pool queries for task distribution
  std::vector<FullPtr<Task>> subtasks_; // Replica tasks for this execution
  u32 completed_replicas_; // Count of completed replicas
  u32 yield_count_;  // Number of times task has yielded
  Future<Task> future_;  // Future for async completion tracking
  bool destroy_in_end_task_;  // Flag to indicate if task should be destroyed in EndTask

  RunContext()
      : stack_ptr(nullptr), stack_base_for_free(nullptr), stack_size(0),
        thread_type(kSchedWorker), worker_id(0), is_yielded_(false),
        est_load(0.0), yield_time_us_(0.0), block_start(), yield_context{}, resume_context{},
        container(nullptr), lane(nullptr), exec_mode(ExecMode::kExec),
        event_queue_(nullptr),
        completed_replicas_(0), yield_count_(0), destroy_in_end_task_(false) {
  }

  /**
   * Move constructor
   */
  RunContext(RunContext &&other) noexcept
      : stack_ptr(other.stack_ptr),
        stack_base_for_free(other.stack_base_for_free),
        stack_size(other.stack_size), thread_type(other.thread_type),
        worker_id(other.worker_id), task(std::move(other.task)),
        is_yielded_(other.is_yielded_), est_load(other.est_load),
        yield_time_us_(other.yield_time_us_), block_start(other.block_start),
        yield_context(other.yield_context), resume_context(other.resume_context),
        container(other.container), lane(other.lane),
        exec_mode(other.exec_mode),
        event_queue_(other.event_queue_),
        pool_queries(std::move(other.pool_queries)),
        subtasks_(std::move(other.subtasks_)),
        completed_replicas_(other.completed_replicas_),
        yield_count_(other.yield_count_),
        future_(std::move(other.future_)),
        destroy_in_end_task_(other.destroy_in_end_task_) {
    other.event_queue_ = nullptr;
  }

  /**
   * Move assignment operator
   */
  RunContext &operator=(RunContext &&other) noexcept {
    if (this != &other) {
      stack_ptr = other.stack_ptr;
      stack_base_for_free = other.stack_base_for_free;
      stack_size = other.stack_size;
      thread_type = other.thread_type;
      worker_id = other.worker_id;
      task = std::move(other.task);
      is_yielded_ = other.is_yielded_;
      est_load = other.est_load;
      yield_time_us_ = other.yield_time_us_;
      block_start = other.block_start;
      yield_context = other.yield_context;
      resume_context = other.resume_context;
      container = other.container;
      lane = other.lane;
      exec_mode = other.exec_mode;
      event_queue_ = other.event_queue_;
      pool_queries = std::move(other.pool_queries);
      subtasks_ = std::move(other.subtasks_);
      completed_replicas_ = other.completed_replicas_;
      yield_count_ = other.yield_count_;
      future_ = std::move(other.future_);
      destroy_in_end_task_ = other.destroy_in_end_task_;
      other.event_queue_ = nullptr;
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
    pool_queries.clear();
    subtasks_.clear();
    completed_replicas_ = 0;
    est_load = 0.0;
    yield_time_us_ = 0.0;
    block_start = hshm::Timepoint();
    yield_count_ = 0;
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