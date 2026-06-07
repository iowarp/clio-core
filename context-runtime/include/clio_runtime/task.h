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

#ifndef CHIMAERA_INCLUDE_CHIMAERA_TASK_H_
#define CHIMAERA_INCLUDE_CHIMAERA_TASK_H_

#ifdef _WIN32
using pid_t = int;
#else
#include <sys/types.h>
#endif

#include <atomic>
#ifndef __NVCOMPILER
#include <coroutine>
#endif
#include <memory>
#include <sstream>
#include <vector>

#include "clio_runtime/pool_query.h"
#include "clio_runtime/types.h"
#include "clio_ctp/data_structures/ipc/shm_container.h"
#include "clio_ctp/data_structures/ipc/vector.h"
#include "clio_ctp/lightbeam/shm_transport.h"
#include "clio_ctp/memory/allocator/allocator.h"
#include "clio_ctp/util/logging.h"

// Include GlobalSerialize for architecture-portable serialization
#include <clio_ctp/data_structures/serialization/global_serialize.h>

// Forward declare chi::priv::string for cereal support
namespace ctp::priv {
template <typename T, typename AllocT, size_t SmallSize>
class basic_string;
}

// ============================================================================
// NVHPC ucontext_t fiber infrastructure
// Replaces C++20 coroutines for NVHPC compiler (which crashes on libstdc++
// coroutines with ICE in llc).
// ============================================================================
#ifdef __NVCOMPILER
#include <ucontext.h>
#include <functional>
namespace clio::run::detail {
  static constexpr size_t FIBER_STACK_SIZE = 256 * 1024;
  // Forward declare RunContext for FiberState
  struct FiberState;
  inline thread_local FiberState* tls_current_fiber = nullptr;

  struct FiberCallable {
    virtual void call() = 0;
    virtual ~FiberCallable() = default;
  };

  template<typename F>
  struct FiberCallableT : FiberCallable {
    F fn;
    explicit FiberCallableT(F&& f) : fn(std::move(f)) {}
    void call() override { fn(); }
  };

  struct FiberState {
    ucontext_t fiber_ctx;
    ucontext_t caller_ctx;
    bool done = false;
    std::unique_ptr<FiberCallable> fn;
    std::unique_ptr<char[]> stack;
    FiberState() : done(false), stack(new char[FIBER_STACK_SIZE]) {}
  };

  static void fiber_trampoline() {
    auto* fs = tls_current_fiber;
    try { fs->fn->call(); } catch(...) {}
    fs->done = true;
    swapcontext(&fs->fiber_ctx, &fs->caller_ctx);
  }

  class FiberHandle {
  public:
    FiberState* state_ = nullptr;
    FiberHandle() = default;
    explicit FiberHandle(FiberState* s) : state_(s) {}
    bool done() const noexcept { return !state_ || state_->done; }
    void resume() {
      if (!state_ || state_->done) return;
      auto* prev = tls_current_fiber;
      tls_current_fiber = state_;
      swapcontext(&state_->caller_ctx, &state_->fiber_ctx);
      tls_current_fiber = prev;
    }
    void destroy() { delete state_; state_ = nullptr; }
    explicit operator bool() const { return state_ != nullptr; }
    bool operator!() const { return state_ == nullptr; }
    FiberHandle& operator=(std::nullptr_t) noexcept { state_ = nullptr; return *this; }
    bool operator==(std::nullptr_t) const noexcept { return state_ == nullptr; }
    bool operator!=(std::nullptr_t) const noexcept { return state_ != nullptr; }
  };
}  // namespace clio::run::detail
#endif // __NVCOMPILER

namespace clio::run {

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
 * TaskGroup - Identifies a scheduling affinity group
 *
 * Tasks in the same group are pinned to the same worker once routed.
 * A null TaskGroup (id_ == -1) means no affinity group.
 */
struct TaskGroup {
  int64_t id_{-1}; /**< Group identifier; -1 = null (no group) */

  /** Default constructor - null group */
  CTP_CROSS_FUN TaskGroup() : id_(-1) {}

  /** Construct with explicit group ID */
  CTP_CROSS_FUN explicit TaskGroup(int64_t id) : id_(id) {}

  /** Check if this group is null (unassigned) */
  CTP_CROSS_FUN bool IsNull() const { return id_ == -1; }

  /** Equality operators */
  CTP_CROSS_FUN bool operator==(const TaskGroup& o) const {
    return id_ == o.id_;
  }
  CTP_CROSS_FUN bool operator!=(const TaskGroup& o) const {
    return id_ != o.id_;
  }
};

/**
 * Task statistics for I/O and compute time tracking
 * Represents group-level properties: when a task belongs to a TaskGroup,
 * these stats describe the characteristics of the entire group (not just
 * the individual task) and are used to route all group tasks consistently.
 */
struct TaskStat {
  size_t io_size_{0};  /**< I/O size in bytes */
  size_t compute_{0};  /**< Normalized compute time in microseconds */
  float wall_time_{0}; /**< Normalized wall time input for InferWallClockTime */
};

// Define macros for container template
#define CLASS_NAME Task
#define CLASS_NEW_ARGS

/**
 * Base task class for CLIO Runtime distributed execution
 *
 * All tasks represent C++ functions similar to RPCs that can be executed
 * across the distributed system. Tasks are now allocated in private memory
 * using standard new/delete.
 */
class Task {
 public:
  typedef CLIO_QUEUE_ALLOC_T AllocT;
  IN PoolId pool_id_;       /**< Pool identifier for task execution */
  IN TaskId task_id_;       /**< Task identifier for task routing */
  IN PoolQuery pool_query_; /**< Pool query for execution location */
  IN MethodId method_;      /**< Method identifier for task type */
  IN ibitfield task_flags_; /**< Task properties and flags */
  IN double period_ns_;     /**< Period in nanoseconds for periodic tasks */
  IN TaskGroup
      task_group_; /**< Scheduling affinity group (null = no affinity) */
  OUT ctp::ipc::atomic<u32>
      return_code_; /**< Task return code (0=success, non-zero=error) */
  OUT ctp::ipc::atomic<ContainerId>
      completer_; /**< Container ID that completed this task */
  IN u32 pod_size_; /**< sizeof(TaskT) for POD copy transport */
  /** Raw pointer to host RunContext (cast to RunContext* on CPU, always 0 on GPU).
   *  Unconditional so sizeof(Task) is identical on CPU and GPU. */
  TEMP uintptr_t host_run_ctx_ = 0;

#if CTP_IS_HOST
  RunContext* GetRunCtx() { return reinterpret_cast<RunContext*>(host_run_ctx_); }
  const RunContext* GetRunCtx() const { return reinterpret_cast<const RunContext*>(host_run_ctx_); }
  void SetRunCtx(RunContext* ctx) { host_run_ctx_ = reinterpret_cast<uintptr_t>(ctx); }
  /** Implemented out-of-line (task.cc) because RunContext is incomplete here */
  void DestroyRunCtx();
#endif

  /**
   * Destructor — must explicitly free RunContext since we no longer use unique_ptr.
   * Host pass uses the out-of-line definition in task.cc; any device pass
   * (CUDA/ROCm/SYCL) uses `= default`. Switching from CTP_IS_HOST to
   * !CTP_IS_DEVICE_PASS lets DPC++'s SYCL device pass — where CTP_IS_HOST=1 —
   * pick the inline default destructor instead of an unresolved declaration.
   */
#if !CTP_IS_DEVICE_PASS
  ~Task();
#else
  ~Task() = default;
#endif

  /**
   * Default constructor
   */
  CTP_CROSS_FUN Task() { pod_size_ = 0; host_run_ctx_ = 0; SetNull(); }

  /**
   * Emplace constructor with task initialization
   */
  CTP_CROSS_FUN explicit Task(const TaskId& task_id, const PoolId& pool_id,
                               const PoolQuery& pool_query,
                               const MethodId& method) {
    // Initialize task
    task_id_ = task_id;
    pool_id_ = pool_id;
    method_ = method;
    task_flags_.SetBits(0);
    pool_query_ = pool_query;
    period_ns_ = 0.0;
    pod_size_ = 0;
    host_run_ctx_ = 0;
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
  CTP_CROSS_FUN void Copy(const ctp::ipc::FullPtr<Task>& other) {
    pool_id_ = other->pool_id_;
    task_id_ = other->task_id_;
    pool_query_ = other->pool_query_;
    method_ = other->method_;
    task_flags_ = other->task_flags_;
    period_ns_ = other->period_ns_;
    // host_run_ctx_ is not copied — each task owns its own RunContext
    return_code_.store(other->return_code_.load());
    completer_.store(other->completer_.load());
    task_group_ = other->task_group_;
  }

  /**
   * SetNull implementation
   */
  CTP_INLINE_CROSS_FUN void SetNull() {
    pool_id_ = PoolId::GetNull();
    task_id_ = TaskId();
    pool_query_ = PoolQuery();
    method_ = 0;
    task_flags_.Clear();
    period_ns_ = 0.0;
#if CTP_IS_HOST
    DestroyRunCtx();
#endif
    return_code_.store(0);  // Initialize as success
    completer_.store(0);    // Initialize as null (0 is invalid container ID)
    task_group_ = TaskGroup();  // null group
  }

  /**
   * Check if task is periodic
   * @return true if task has periodic flag set
   */
  CTP_CROSS_FUN bool IsPeriodic() const {
    return task_flags_.Any(TASK_PERIODIC);
  }

  /**
   * Check if task has been routed
   * @return true if task has routed flag set
   */
  CTP_CROSS_FUN bool IsRouted() const { return task_flags_.Any(TASK_ROUTED); }

  /**
   * Check if task is the data owner
   * @return true if task has data owner flag set
   */
  CTP_CROSS_FUN bool IsDataOwner() const {
    return task_flags_.Any(TASK_DATA_OWNER);
  }

  /**
   * Check if task is a remote task (received from another node)
   * @return true if task has remote flag set
   */
  CTP_CROSS_FUN bool IsRemote() const { return task_flags_.Any(TASK_REMOTE); }

  /**
   * Get task execution period in specified time unit
   * @param unit Time unit constant (kNano, kMicro, kMilli, kSec, kMin, kHour)
   * @return Period in specified unit, 0 if not periodic
   */
  CTP_CROSS_FUN double GetPeriod(double unit) const {
    return period_ns_ / unit;
  }

  /**
   * Set task execution period in specified time unit
   * @param period Period value in the specified unit
   * @param unit Time unit constant (kNano, kMicro, kMilli, kSec, kMin, kHour)
   */
  CTP_CROSS_FUN void SetPeriod(double period, double unit) {
    period_ns_ = period * unit;
  }

  /**
   * Set task flags
   * @param flags Bitfield of task flags to set
   */
  CTP_CROSS_FUN void SetFlags(u32 flags) { task_flags_.SetBits(flags); }

  /**
   * Clear task flags
   * @param flags Bitfield of task flags to clear
   */
  CTP_CROSS_FUN void ClearFlags(u32 flags) { task_flags_.UnsetBits(flags); }

  /**
   * Serialize data structures to chi::priv::string using GlobalSerialize
   * @param alloc Allocator for memory management (CLIO_PRIV_ALLOC_T)
   * @param output_str The string to store serialized data
   * @param args The arguments to serialize
   */
  template <typename... Args>
  static void Serialize(CLIO_PRIV_ALLOC_T* alloc,
                        chi::priv::string& output_str, const Args&... args) {
    std::vector<char> buffer;
    ctp::ipc::GlobalSerialize<std::vector<char>> ar(buffer);
    ar(args...);
    ar.Finalize();
    std::string serialized(buffer.begin(), buffer.end());
    output_str = chi::priv::string(alloc, serialized);
  }

  /**
   * Deserialize data structure from chi::ipc::string using GlobalDeserialize
   * @param input_str The string containing serialized data
   * @return The deserialized object
   */
  template <typename OutT>
  static OutT Deserialize(const chi::priv::string& input_str) {
    std::vector<char> data(input_str.data(),
                           input_str.data() + input_str.size());
    ctp::ipc::GlobalDeserialize<std::vector<char>> ar(data);
    OutT result;
    ar(result);
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
  CTP_CROSS_FUN void SerializeIn(Archive& ar) {
    ar.range(pool_id_, task_id_, pool_query_, method_, task_flags_,
             period_ns_, task_group_, return_code_, completer_);
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
  CTP_CROSS_FUN void SerializeOut(Archive& ar) {
    ar.range(return_code_, completer_);
  }

  /**
   * Get the task return code
   * @return Return code (0=success, non-zero=error)
   */
  CTP_CROSS_FUN u32 GetReturnCode() const { return return_code_.load(); }

  /**
   * Set the task return code
   * @param return_code Return code to set (0=success, non-zero=error)
   */
  CTP_CROSS_FUN void SetReturnCode(u32 return_code) {
    return_code_.store(return_code);
  }

  /**
   * Get the completer container ID (which container completed this task)
   * @return Container ID that completed this task
   */
  CTP_CROSS_FUN ContainerId GetCompleter() const { return completer_.load(); }

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
  CTP_CROSS_FUN void SetCompleter(ContainerId completer) {
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
  void Aggregate(const ctp::ipc::FullPtr<Task>& replica_task) {
    // Propagate return code from replica to this task
    if (!replica_task.IsNull() && replica_task->GetReturnCode() != 0) {
      SetReturnCode(replica_task->GetReturnCode());
    }
    // Copy the completer from the replica task
    if (!replica_task.IsNull()) {
      SetCompleter(replica_task->GetCompleter());
    }
    HLOG(kDebug, "[COMPLETER] Aggregated task {} with completer {}", task_id_,
         GetCompleter());
  }

  /**
   * Get the copy space size for serialized task output
   * Derived classes can override to specify custom copy space sizes
   * Default is 4KB (4096 bytes) for most tasks
   * @return Size in bytes for the serialized_task_ capacity
   */
  CTP_CROSS_FUN size_t GetCopySpaceSize() const {
    return 4096;  // Default 4KB for most tasks
  }

  /**
   * Fix up internal pointers after a cudaMemcpy/memcpy (POD copy path).
   * Override in tasks that contain priv::vector or other pointer-bearing
   * fields to re-seat inline (SVO) pointers to the new host address.
   * Default is a no-op for pure POD tasks.
   */
  CTP_CROSS_FUN void FixupAfterCopy() {}
};

}  // namespace clio::run

// FutureShm and Future classes (must be before RunContext which uses Future)
#include "clio_runtime/future.h"

// FutureShm and Future are defined in chimaera/future.h (included above)

namespace clio::run {

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
    ctp::ipc::multi_mpsc_ring_buffer<Future<Task>,
                                 CLIO_QUEUE_ALLOC_T>::ring_buffer_type;

/**
 * Simple wrapper around ctp::ipc::multi_mpsc_ring_buffer
 *
 * This wrapper adds custom enqueue and dequeue functions while maintaining
 * compatibility with existing code that expects the multi_mpsc_ring_buffer
 * interface.
 */
typedef ctp::ipc::multi_mpsc_ring_buffer<Future<Task>, CLIO_QUEUE_ALLOC_T> TaskQueue;

}  // namespace clio::run

// GPU Future types (must be after chi::Future and chi::Task are complete)
#include "clio_runtime/gpu/future.h"

namespace clio::run {

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
  /** Coroutine handle for C++20 stackless coroutines (or fiber handle for NVHPC) */
#ifndef __NVCOMPILER
  std::coroutine_handle<> coro_handle_;
#else
  chi::detail::FiberHandle coro_handle_;
#endif
  u32 worker_id_;               /**< Worker ID executing this task */
  FullPtr<Task> task_;          /**< Task being executed by this context */
  bool is_yielded_;             /**< Task is waiting for completion */
  double yield_time_us_;        /**< Time in microseconds for task to yield */
  ctp::Timepoint block_start_; /**< Time when task was blocked (real time) */
  Container* container_;        /**< Current container being executed */
  TaskLane* lane_;              /**< Current lane being processed */
  void* event_queue_;           /**< Pointer to worker's event queue */
  std::vector<PoolQuery>
      pool_queries_; /**< Pool queries for task distribution */
  std::vector<FullPtr<Task>> subtasks_; /**< Replica tasks for this execution */
  // Atomic so SendIn (net_send_worker), RecvOut (net_recv_worker) and
  // FlushStaleStateForNode can update it concurrently without losing
  // increments — a missed bump leaves completed_ < subtasks_.size()
  // forever and the origin task's future never fires, which manifests
  // as a writer hang under heavy 4n+ FPP load.
  std::atomic<u32> completed_replicas_;
  u32 yield_count_;                     /**< Number of times task has yielded */
  Future<Task, CLIO_QUEUE_ALLOC_T>
      future_;                    /**< Future for async completion tracking */
  std::atomic<bool> is_notified_; /**< Atomic flag to prevent duplicate event
                                     queue additions */
  // Set true by the top-level coroutine's final_suspend when the task
  // completes. The worker reads THIS instead of coro_handle_.done() to detect
  // completion, so it never dereferences a coroutine frame that a cross-thread
  // completion may already have freed — done() on a freed frame GPFLTs (issue
  // #485). The RunContext outlives the frame, so this read is always valid.
  std::atomic<bool> coro_completed_;
  double true_period_ns_;         /**< Original period from task->period_ns_ */
  bool did_work_;            /**< Whether task did work in last execution */
  ctp::CpuTimer cpu_timer_; /**< Accumulates thread CPU time across yields */
  float predicted_load_ = 0; /**< Predicted CPU time from InferModel (us) */
  ctp::HighResMonotonicTimer wall_timer_; /**< Wall clock time across yields */
  float predicted_wall_us_ =
      0; /**< Predicted wall time from InferWallClockTime */
  TaskStat
      predicted_stat_; /**< TaskStat used for prediction (for reinforcement) */

  RunContext()
      : coro_handle_(),
        worker_id_(0),
        is_yielded_(false),
        yield_time_us_(0.0),
        block_start_(),
        container_(nullptr),
        lane_(nullptr),
        event_queue_(nullptr),
        completed_replicas_(0),
        yield_count_(0),
        is_notified_(false),
        coro_completed_(false),
        true_period_ns_(0.0),
        did_work_(false) {}

  /**
   * Move constructor
   */
  RunContext(RunContext&& other) noexcept
      : coro_handle_(std::move(other.coro_handle_)),
        worker_id_(other.worker_id_),
        task_(std::move(other.task_)),
        is_yielded_(other.is_yielded_),
        yield_time_us_(other.yield_time_us_),
        block_start_(other.block_start_),
        container_(other.container_),
        lane_(other.lane_),
        event_queue_(other.event_queue_),
        pool_queries_(std::move(other.pool_queries_)),
        subtasks_(std::move(other.subtasks_)),
        completed_replicas_(other.completed_replicas_.load()),
        yield_count_(other.yield_count_),
        future_(std::move(other.future_)),
        is_notified_(other.is_notified_.load()),
        coro_completed_(other.coro_completed_.load()),
        true_period_ns_(other.true_period_ns_),
        did_work_(other.did_work_),
        cpu_timer_(other.cpu_timer_),
        predicted_load_(other.predicted_load_),
        wall_timer_(other.wall_timer_),
        predicted_wall_us_(other.predicted_wall_us_),
        predicted_stat_(other.predicted_stat_) {
#ifndef __NVCOMPILER
    other.coro_handle_ = nullptr;
#else
    other.coro_handle_ = chi::detail::FiberHandle{};
#endif
    other.event_queue_ = nullptr;
  }

  /**
   * Move assignment operator
   */
  RunContext& operator=(RunContext&& other) noexcept {
    if (this != &other) {
      coro_handle_ = std::move(other.coro_handle_);
      worker_id_ = other.worker_id_;
      task_ = std::move(other.task_);
      is_yielded_ = other.is_yielded_;
      yield_time_us_ = other.yield_time_us_;
      block_start_ = other.block_start_;
      container_ = other.container_;
      lane_ = other.lane_;
      event_queue_ = other.event_queue_;
      pool_queries_ = std::move(other.pool_queries_);
      subtasks_ = std::move(other.subtasks_);
      completed_replicas_.store(other.completed_replicas_.load());
      yield_count_ = other.yield_count_;
      future_ = std::move(other.future_);
      is_notified_.store(other.is_notified_.load());
      coro_completed_.store(other.coro_completed_.load());
      true_period_ns_ = other.true_period_ns_;
      did_work_ = other.did_work_;
      cpu_timer_ = other.cpu_timer_;
      predicted_load_ = other.predicted_load_;
      wall_timer_ = other.wall_timer_;
      predicted_wall_us_ = other.predicted_wall_us_;
      predicted_stat_ = other.predicted_stat_;
#ifndef __NVCOMPILER
      other.coro_handle_ = nullptr;
#else
      other.coro_handle_ = chi::detail::FiberHandle{};
#endif
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
    completed_replicas_.store(0);
    yield_time_us_ = 0.0;
    block_start_ = ctp::Timepoint();
    yield_count_ = 0;
    is_notified_.store(false);
    coro_completed_.store(false);
    true_period_ns_ = 0.0;
    did_work_ = false;
    cpu_timer_.time_ns_ = 0;
    predicted_load_ = 0;
    wall_timer_.time_ns_ = 0;
    predicted_wall_us_ = 0;
    predicted_stat_ = TaskStat();
  }
};

// ============================================================================
// Future::await_suspend_impl implementation (must be after RunContext
// definition)
// ============================================================================

#ifndef __NVCOMPILER
template <typename TaskT, typename AllocT>
bool Future<TaskT, AllocT>::await_suspend_impl(
    std::coroutine_handle<> handle) noexcept {
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
#endif // !__NVCOMPILER

// ============================================================================
// TaskResume and YieldAwaiter (must be after RunContext for member access)
// ============================================================================

#ifndef __NVCOMPILER
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
     * True only for the top-level task coroutine the worker starts directly
     * (set in Worker::StartCoroutine). Nested co_await'd coroutines leave it
     * false. Used by final_suspend to set run_ctx_->coro_completed_ on the
     * real task completion only — caller_handle_ is unreliable for this
     * because a nested coroutine that finishes synchronously still has a null
     * caller_handle_ at that instant (issue #485).
     */
    bool is_top_level_ = false;

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
      // When the TOP-LEVEL task coroutine reaches final suspend the whole task
      // is complete, so record it in the RunContext. The worker checks
      // run_ctx_->coro_completed_ instead of coro_handle_.done(); the coroutine
      // frame may already be freed by a cross-thread completion, which would
      // make done() a use-after-free (issue #485), whereas the RunContext
      // outlives the frame. By the time the top-level final_suspend runs,
      // await_resume has repointed coro_handle_ back to this (top-level)
      // handle, so the worker's subsequent destroy() still targets a valid
      // frame. We gate on is_top_level_ (not caller_handle_): a nested
      // coroutine that completes synchronously also momentarily has a null
      // caller_handle_, and must NOT be mistaken for task completion.
      if (is_top_level_ && run_ctx_ != nullptr) {
        run_ctx_->coro_completed_.store(true, std::memory_order_release);
      }
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

#else // __NVCOMPILER - use ucontext_t fiber-based TaskResume

/**
 * TaskResume (NVHPC) - Fiber-based return type for runtime methods
 *
 * Wraps a chi::detail::FiberHandle instead of a coroutine handle.
 * Used when compiling with NVHPC which crashes on C++20 coroutines.
 */
class TaskResume {
  chi::detail::FiberHandle handle_;

public:
  TaskResume() = default;
  explicit TaskResume(chi::detail::FiberHandle h) : handle_(std::move(h)) {}

  TaskResume(TaskResume&& o) noexcept : handle_(o.handle_) {
    o.handle_ = chi::detail::FiberHandle{};
  }

  TaskResume& operator=(TaskResume&& o) noexcept {
    if (this != &o) {
      if (handle_) handle_.destroy();
      handle_ = o.handle_;
      o.handle_ = chi::detail::FiberHandle{};
    }
    return *this;
  }

  TaskResume(const TaskResume&) = delete;
  TaskResume& operator=(const TaskResume&) = delete;

  ~TaskResume() {
    if (handle_) handle_.destroy();
  }

  bool done() const { return handle_.done(); }
  void resume() { handle_.resume(); }
  void destroy() { handle_.destroy(); }

  chi::detail::FiberHandle& get_handle() { return handle_; }
  const chi::detail::FiberHandle& get_handle() const { return handle_; }

  /**
   * Release ownership of the fiber handle without destroying it
   */
  chi::detail::FiberHandle release() {
    auto h = handle_;
    handle_ = chi::detail::FiberHandle{};
    return h;
  }

  explicit operator bool() const { return bool(handle_); }
  bool operator!() const { return !handle_; }
};

#endif // __NVCOMPILER

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

#ifndef __NVCOMPILER
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
#endif // !__NVCOMPILER

  /**
   * Resume after yield - nothing to return
   */
  void await_resume() noexcept {}

  /**
   * Get the yield time in microseconds (used by fiber_co_await for NVHPC)
   */
  double get_yield_time_us() const noexcept { return yield_time_us_; }
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

}  // namespace clio::run

// ============================================================================
// NVHPC fiber_co_await overloads and make_task_fiber
// (outside chi namespace, in chi::detail namespace)
// ============================================================================
#ifdef __NVCOMPILER
namespace clio::run::detail {

/// Yield awaiter overload: suspends fiber and marks rctx as yielded
inline void fiber_co_await(chi::YieldAwaiter ya, chi::RunContext& rctx) {
  auto* fs = tls_current_fiber;
  if (!fs) return;
  rctx.is_yielded_ = true;
  rctx.yield_time_us_ = ya.get_yield_time_us();
  swapcontext(&fs->fiber_ctx, &fs->caller_ctx);
  rctx.is_yielded_ = false;
}

/// Future overload: waits for async task completion
template<typename TaskT, typename AllocT>
inline void fiber_co_await(chi::Future<TaskT, AllocT>& future, chi::RunContext& rctx) {
  if (future.IsReady()) return;
  auto* fs = tls_current_fiber;
  if (!fs) return;
  future.SetParentTask(&rctx);
  rctx.is_yielded_ = true;
  rctx.yield_time_us_ = 0.0;
  swapcontext(&fs->fiber_ctx, &fs->caller_ctx);
  rctx.is_yielded_ = false;
}

/// Future rvalue overload (for temporaries)
template<typename TaskT, typename AllocT>
inline void fiber_co_await(chi::Future<TaskT, AllocT>&& future, chi::RunContext& rctx) {
  fiber_co_await(future, rctx);
}

/// TaskResume fiber overload: runs inner fiber until it completes, yielding
/// outer fiber whenever inner suspends
inline void fiber_co_await(chi::TaskResume inner, chi::RunContext& rctx) {
  if (!inner) return;
  while (!inner.done()) {
    inner.get_handle().resume();
    if (!inner.done() && rctx.is_yielded_) {
      // Inner fiber yielded - propagate yield upward
      rctx.is_yielded_ = false;
      auto* fs = tls_current_fiber;
      if (fs) {
        rctx.is_yielded_ = true;
        swapcontext(&fs->fiber_ctx, &fs->caller_ctx);
        rctx.is_yielded_ = false;
      }
    }
  }
}

/// Create a TaskResume wrapping a new fiber
template<typename F>
inline chi::TaskResume make_task_fiber(F&& fn) {
  auto* state = new FiberState();
  state->fn = std::make_unique<FiberCallableT<typename std::decay<F>::type>>(std::forward<F>(fn));
  getcontext(&state->fiber_ctx);
  state->fiber_ctx.uc_stack.ss_sp = state->stack.get();
  state->fiber_ctx.uc_stack.ss_size = FIBER_STACK_SIZE;
  state->fiber_ctx.uc_link = nullptr;
  makecontext(&state->fiber_ctx, fiber_trampoline, 0);
  return chi::TaskResume(FiberHandle(state));
}

}  // namespace clio::run::detail
#endif // __NVCOMPILER

// ============================================================================
// Cross-compiler macros for task bodies (co_await / co_return replacements)
// ============================================================================
#ifndef __NVCOMPILER
#  define CLIO_TASK_BODY_BEGIN
#  define CLIO_TASK_BODY_END
#  define CLIO_CO_AWAIT(expr)  co_await (expr)
#  define CLIO_CO_RETURN       co_return
#else
#  define CLIO_TASK_BODY_BEGIN return chi::detail::make_task_fiber([=, &rctx]() mutable {
#  define CLIO_TASK_BODY_END   });
#  define CLIO_CO_AWAIT(expr)  chi::detail::fiber_co_await((expr), rctx)
// Use plain return so RAII destructors (e.g. ScopedCoMutex) run before the
// fiber stack is freed. fiber_trampoline handles the final swapcontext back
// to the worker after the lambda returns.
#  define CLIO_CO_RETURN       return
#endif
// Backward-compat aliases (clio_run rebrand). External code that still
// uses the legacy CHI_* spelling keeps working unchanged.
#define CHI_TASK_BODY_BEGIN  CLIO_TASK_BODY_BEGIN
#define CHI_TASK_BODY_END    CLIO_TASK_BODY_END
#define CHI_CO_AWAIT         CLIO_CO_AWAIT
#define CHI_CO_RETURN        CLIO_CO_RETURN

#endif  // CHIMAERA_INCLUDE_CHIMAERA_TASK_H_
