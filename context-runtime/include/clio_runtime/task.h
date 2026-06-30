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

#ifndef CLIO_RUNTIME_INCLUDE_TASK_H_
#define CLIO_RUNTIME_INCLUDE_TASK_H_

#ifdef _WIN32
using pid_t = int;
#else
#include <sys/types.h>
#endif

#include <atomic>
// ============================================================================
// Coroutine backend selection.
//   * Default: C++20 stackless coroutines (std::coroutine_handle).
//   * Boost.Context stackful "fiber" backend, selected by
//     CLIO_ENABLE_BOOST_COROUTINES. This is the only stackful backend. CMake
//     forces it ON for compilers that cannot compile C++20 coroutines (e.g.
//     NVHPC, which ICEs in llc on libstdc++ coroutines), so the compiler
//     decision lives in the build — not in #ifdefs here.
// ============================================================================
#ifndef CLIO_ENABLE_BOOST_COROUTINES
#include <coroutine>
#endif
#include <memory>
#include <sstream>
#include <vector>

#include "clio_runtime/dynamic_container.h"
#include "clio_runtime/pool_query.h"
#include "clio_runtime/types.h"
#include "clio_ctp/data_structures/ipc/shm_container.h"
#include "clio_ctp/data_structures/ipc/vector.h"
#include "clio_ctp/lightbeam/shm_transport.h"
#include "clio_ctp/memory/allocator/allocator.h"
#include "clio_ctp/util/logging.h"

// Include GlobalSerialize for architecture-portable serialization
#include <clio_ctp/data_structures/serialization/global_serialize.h>

// Forward declare clio::run::priv::string for cereal support
namespace ctp::priv {
template <typename T, typename AllocT, size_t SmallSize>
class basic_string;
}

// ============================================================================
// Boost.Context stackful fiber infrastructure (issue #620).
// The stackful backend, active whenever CLIO_ENABLE_BOOST_COROUTINES is defined
// (opt-in, or forced by CMake for compilers without C++20 coroutines such as
// NVHPC). Provides the FiberHandle/FiberState surface the worker and task-body
// macros drive; RunContext stays backend-agnostic.
//
// There is intentionally no "current fiber" thread_local: the running fiber is
// reached through the worker's current RunContext (rctx.coro_handle_.state_),
// which already flows across DLLs via the exported GetCurrentRunContextFromWorker
// (a header thread_local would duplicate per-DLL on Windows — issue #620).
// ============================================================================
#ifdef CLIO_ENABLE_BOOST_COROUTINES
#include <boost/context/fiber.hpp>
// fixedsize_stack (plain malloc), NOT protected_fixedsize_stack: the latter
// pulls in windows.h (VirtualAlloc/VirtualProtect) on Windows, whose macros
// clash with the IPC headers (error C2760), and its per-fiber mmap+mprotect
// would undermine the "avoid malloc" goal of the Boost backend (#620).
#include <boost/context/fixedsize_stack.hpp>
#include <cstdlib>
#include <functional>
namespace clio::run { class Worker; }  // for FiberState::worker_
namespace clio::run::detail {
  // (KiB). Defaults to 256 KiB: the whole task — incl. nested helper coroutines
  // and the DPE/allocation/cereal serialization call chain — runs on this one
  // stack, but with stacks reused per task a large default wastes memory across
  // many concurrent fibers. Bump CLIO_BOOST_STACK_SIZE if a deep call chain
  // overflows.
  inline size_t boost_stack_size() {
    static const size_t sz = []() -> size_t {
      const char* e = std::getenv("CLIO_BOOST_STACK_SIZE");
      size_t kib = (e && *e) ? std::strtoul(e, nullptr, 10) : 256;
      if (kib < 64) kib = 64;
      return kib * 1024;
    }();
    return sz;
  }
  // Per-task fiber state. It lives INSIDE the task's RunContext (see
  // RunContext::fiber_state_), so starting a task fiber allocates ONLY the
  // stack — there is no heap FiberState and no type-erased callable. `task_`
  // holds the suspended task continuation (entered by resume()); `caller_`
  // holds the continuation back to the worker, refreshed on every suspend;
  // `worker_` is the worker that created the fiber (Worker::make_task_fiber) —
  // the one originally responsible for driving/cleaning it up. A
  // finished/destroyed fiber unwinds its own stack via RAII.
  struct FiberState {
    boost::context::fiber task_;
    boost::context::fiber caller_;
    bool done = false;
    clio::run::Worker* worker_ = nullptr;
  };

  // Suspend the running fiber back to the worker (inverse of FiberHandle::resume).
  inline void fiber_suspend_to_caller(FiberState* fs) {
    fs->caller_ = std::move(fs->caller_).resume();
  }

  // Non-owning handle to a FiberState (owned by the RunContext). resume() drives
  // the task fiber; destroy() detaches and frees the stack (the FiberState
  // itself is freed with its RunContext).
  class FiberHandle {
  public:
    FiberState* state_ = nullptr;
    FiberHandle() = default;
    explicit FiberHandle(FiberState* s) : state_(s) {}
    bool done() const noexcept { return !state_ || state_->done; }
    void resume() {
      if (!state_ || state_->done) return;
      state_->task_ = std::move(state_->task_).resume();
    }
    void destroy() {
      if (state_) {
        state_->task_ = boost::context::fiber{};    // free the fiber stack
        state_->caller_ = boost::context::fiber{};
        state_->done = false;
      }
      state_ = nullptr;
    }
    explicit operator bool() const { return state_ != nullptr; }
    bool operator!() const { return state_ == nullptr; }
    FiberHandle& operator=(std::nullptr_t) noexcept { state_ = nullptr; return *this; }
    bool operator==(std::nullptr_t) const noexcept { return state_ == nullptr; }
    bool operator!=(std::nullptr_t) const noexcept { return state_ != nullptr; }
  };
}  // namespace clio::run::detail
#endif // CLIO_ENABLE_BOOST_COROUTINES

namespace clio::run {

// Forward declarations
class Task;
class Container;
class IpcManager;
struct RunContext;
class Worker;
// Future is defined later in this header (after Task); forward-declared here so
// Task's RunContext accessors can name Future<Task, AllocT> by reference.
// (TaskLane is a `using` alias defined later and cannot be forward-declared, so
// the lane accessor below uses a deduced return type instead.)
template <typename TaskT, typename AllocT>
class Future;

/**
 * Canonical accessor for the task currently executing on this worker thread.
 * Module method handlers and runtime internals call this instead of receiving a
 * RunContext (the worker sets the current task before every Run/resume), then
 * reach the execution state through the Task's accessors. On a worker thread the
 * worker's own current task is returned; off a worker thread a thread-local
 * fallback (set via SetCurrentTask) is returned. The returned reference is to a
 * shared_ptr<Task> that is null when nothing is executing. Defined in worker.cc.
 */
clio::run::shared_ptr<Task>& GetCurrentTask();

/**
 * Set the fallback current task for THIS thread (used by tests / non-worker
 * callers that invoke Container::Run directly). On a worker thread the worker's
 * own current task takes precedence in GetCurrentTask(); this only applies when
 * there is no current worker. Pass a null handle to clear. Defined in worker.cc.
 */
void SetCurrentTask(const clio::run::shared_ptr<Task>& task);

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
 * FutureInfo - self-contained completion state embedded in every Task.
 *
 * Holds the bits that must travel WITH the task so it is self-describing for
 * completion + waiter wakeup, replacing the separate (gpu::)FutureShm:
 *  - is_complete_: CPU/host completion signal, set by the completing worker (or
 *    the client recv thread when the response lands) and polled by the waiter /
 *    the GPU kernel. (Was Task::is_complete_ / FutureShm::flags_ FUTURE_COMPLETE.)
 *  - task_size_: sizeof(the concrete TaskT) — the GPU worker needs it to D2H/H2D
 *    the POD task; also the POD-transport size. (Was Task::pod_size_ /
 *    FutureShm::gpu_task_size_ / gpu::FutureShm::task_size_.)
 *  - waiter_pid_/waiter_tid_: the thread to EventManager::Signal on completion.
 * Per-process / not the wire format (the enclosing member is TEMP). Uses
 * ctp::ipc::atomic so the Task POD layout is identical on host and device.
 */
struct FutureInfo {
  ctp::ipc::atomic<u32> is_complete_;  /**< CPU/host completion signal */
  u32 task_size_;                      /**< sizeof(concrete TaskT) */
  u32 waiter_pid_;                     /**< PID of the thread awaiting completion */
  u32 waiter_tid_;                     /**< TID of the thread awaiting completion */

  CTP_CROSS_FUN FutureInfo() : task_size_(0), waiter_pid_(0), waiter_tid_(0) {
    is_complete_.store(0);
  }
};

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
  /** Self-contained completion state: completion flag, sizeof(TaskT), and the
   *  awaiting thread's pid/tid. TEMP — per-process, NOT serialized or copied
   *  (like run_ctx_); for the GPU POD path it rides along in the memcpy'd Task
   *  bytes. Replaces the separate (gpu::)FutureShm: the Task is its own future
   *  completion record. (task_size_ replaces the former pod_size_.) */
  TEMP FutureInfo fut_;
  /** Per-process "new streaming data available" signal (CPU streaming). Replaces
   *  FutureShm::flags_ FUTURE_NEW_DATA. Same not-serialized/not-copied rules. */
  TEMP ctp::ipc::atomic<u32> is_new_data_;
  /** Owned host RunContext (null on GPU and whenever the task is not executing).
   *  A unique_ptr so the Task frees it automatically — no custom DestroyRunCtx.
   *  Same size on CPU and GPU (three pointers); RunContext stays merely
   *  forward-declared here because ctp::unique_ptr destroys via a type-erased
   *  deleter captured by make_unique (where RunContext is complete). */
  TEMP clio::run::unique_ptr<RunContext> run_ctx_;

#if CTP_IS_HOST
  /** Allocate a fresh RunContext for this task (begin executing). Frees any
   *  previously-held context. The RunContext is this Task's private execution-
   *  state extension — there is deliberately NO accessor that returns it; all
   *  access goes through the Task accessors below. Defined out-of-line once
   *  RunContext is a complete type. Called at the ipc_*.cc BeginTask sites. */
  void BeginRunContext();
  /** Allocate the RunContext storage if absent (lightweight; no container
   *  resolution) so the task's embedded routing state (run_ctx_->route_) exists.
   *  Used by the Future ctor / client SendIn before GetFutureShm(). */
  void EnsureRunCtx();
  /** Free this task's RunContext (back to not-executing). */
  void ResetRunCtx() { run_ctx_.reset(); }

#if CTP_IS_HOST
  // Coroutine driving lives on the Task (not the Worker): the Task owns its
  // RunContext and therefore its coroutine frame, so starting/resuming it is the
  // Task's responsibility. `self` is this task's owning shared_ptr handle (the
  // worker passes its current_task_); it is needed because Container::Run and the
  // fiber entry take a shared_ptr<Task>&. Frame teardown is NOT done here — the
  // RunContext destructor frees the frame when execution ends (RAII).
  /** First execution: create the coroutine/fiber frame, wire its promise, and
   *  run it to the first suspension (or completion). */
  void StartCoroutine(clio::run::shared_ptr<Task> &self);
  /** Resume a suspended coroutine/fiber after a subtask completes. */
  void ResumeCoroutine(clio::run::shared_ptr<Task> &self);
#if defined(CLIO_ENABLE_BOOST_COROUTINES)
  /** Set up this task's RunContext to run on a fresh Boost.Context fiber (the
   *  fiber state lives inline in the RunContext; only the stack is pooled). */
  clio::run::detail::FiberHandle MakeTaskFiber(clio::run::shared_ptr<Task> &self);
#endif
#endif  // CTP_IS_HOST

  // ---------------------------------------------------------------------------
  // RunContext accessors. RunContext is the task's private execution-state
  // extension; ALL access goes through these. Each one null-checks run_ctx_ and
  // throws via CLIO_THROW, so callers never touch RunContext fields directly nor
  // dereference a null RunContext. Defined out-of-line below, once RunContext is
  // a complete type. Reference-returning accessors exist for the complex members
  // (timers, future, vectors, coroutine handle) that callers mutate in place.
  // ---------------------------------------------------------------------------
  u32 RunWorkerId() const;
  void SetRunWorkerId(u32 v);
  bool IsYielded() const;
  void SetYielded(bool v = true);
  double YieldTimeUs() const;
  void SetYieldTimeUs(double v);
  ctp::Timepoint& BlockStart();
  DynamicContainer& ExecContainer();
  const DynamicContainer& ExecContainer() const;
  /** Mutable reference to the lane pointer (TaskLane*&). Deduced return type so
   *  this can be declared before TaskLane (a `using` alias) is defined.
   *  Read: `task->Lane()`; write: `task->Lane() = lane`. */
  auto &Lane();
  void* EventQueue() const;
  void SetEventQueue(void* v);
  std::vector<PoolQuery>& PoolQueries();
  std::vector<clio::run::shared_ptr<Task>>& Subtasks();
  std::atomic<u32>& CompletedReplicas();
  u32 YieldCount() const;
  void SetYieldCount(u32 v);
  Future<Task, CLIO_QUEUE_ALLOC_T>& RunFuture();
  /** Pointer to this task's RunContext (which now holds the routing/transport
   *  state, formerly FutureShm). Null if the task has no active RunContext. */
  RunContext* RunCtxPtr();
  bool IsNotified() const;
  void SetNotified(bool v);
  /** Whether this task's coroutine/fiber has run to completion, without
   *  dereferencing the (possibly cross-thread-freed) coroutine frame. */
  bool IsCoroCompleted() const;
  void SetCoroCompleted(bool v);
  double TruePeriodNs() const;
  bool DidWork() const;
  void SetDidWork(bool v = true);
  bool IsRouted() const;
  void SetRouted(bool v = true);
  bool IsStarted() const;
  void SetStarted(bool v = true);
  ctp::CpuTimer& RunCpuTimer();
  float PredictedLoad() const;
  void SetPredictedLoad(float v);
  ctp::HighResMonotonicTimer& RunWallTimer();
  float PredictedWallUs() const;
  void SetPredictedWallUs(float v);
  TaskStat& PredictedStat();
  /** The parent task waiting on this task's completion (i.e. whose coroutine
   *  resumes when this task finishes), read through this task's own future.
   *  Null if this is a top-level / client-originated task. */
  const clio::run::shared_ptr<Task>& GetParentTask() const;
  // Coroutine/fiber handle accessor, used by the coroutine await machinery and
  // this task's own coroutine drivers (StartCoroutine/ResumeCoroutine in
  // task.cc). The worker does NOT touch the handle.
#ifndef CLIO_ENABLE_BOOST_COROUTINES
  std::coroutine_handle<>& CoroHandle();
#else
  clio::run::detail::FiberHandle& CoroHandle();
  clio::run::detail::FiberState& FiberStateRef();
#endif
  /** Reset the per-execution STL/scalar state for reuse (RunContext::Clear). */
  void ClearRunState();
#endif

  /**
   * Destructor — VIRTUAL so that a clio::run::shared_ptr<Task> base view
   * destructs the concrete derived task correctly when its last reference
   * drops (tasks are allocated by runtime method id, so the owning handle is
   * often the base Task type). The run_ctx_ unique_ptr member is destroyed
   * automatically (type-erased deleter), so the body is empty.
   *
   * CLIO_VIRTUAL is now `virtual` on BOTH host and device (issue #556) so the
   * vtable pointer — and therefore sizeof(Task) and every field offset — is
   * identical across a host<->device cudaMemcpy. Device code never dispatches
   * through this vtable (it uses typed tasks, never a base-Task view), so the
   * host vtable is simply ignored on the device side. CTP_CROSS_FUN gives the
   * dtor a __host__ __device__ execution space matching the derived task
   * destructors (also CTP_CROSS_FUN), so nvcc accepts the override.
   */
  CTP_CROSS_FUN CLIO_VIRTUAL ~Task() {}

  /**
   * Default constructor
   */
  CTP_CROSS_FUN Task() { fut_.task_size_ = 0; SetNull(); }

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
    fut_.task_size_ = 0;
    return_code_.store(0);  // Initialize as success
    completer_.store(0);    // Initialize as null (0 is invalid container ID)
    fut_.is_complete_.store(0);
    is_new_data_.store(0);
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
    // run_ctx_ is not copied — each task owns its own RunContext (unique_ptr)
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
    run_ctx_.reset();
#endif
    return_code_.store(0);  // Initialize as success
    completer_.store(0);    // Initialize as null (0 is invalid container ID)
    fut_.is_complete_.store(0);
    is_new_data_.store(0);
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
   * Serialize data structures to clio::run::priv::string using GlobalSerialize
   * @param alloc Allocator for memory management (CLIO_PRIV_ALLOC_T)
   * @param output_str The string to store serialized data
   * @param args The arguments to serialize
   */
  template <typename... Args>
  static void Serialize(CLIO_PRIV_ALLOC_T* alloc,
                        clio::run::priv::string& output_str, const Args&... args) {
    std::vector<char> buffer;
    ctp::ipc::GlobalSerialize<std::vector<char>> ar(buffer);
    ar(args...);
    ar.Finalize();
    std::string serialized(buffer.begin(), buffer.end());
    output_str = clio::run::priv::string(alloc, serialized);
  }

  /**
   * Deserialize data structure from clio::run::ipc::string using GlobalDeserialize
   * @param input_str The string containing serialized data
   * @return The deserialized object
   */
  template <typename OutT>
  static OutT Deserialize(const clio::run::priv::string& input_str) {
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

  // Per-process completion / new-data signals (CPU/host paths). Managed by the
  // Future; replace FutureShm::flags_ FUTURE_COMPLETE / FUTURE_NEW_DATA.
  CTP_CROSS_FUN void SetComplete() { fut_.is_complete_.store(1); }
  CTP_CROSS_FUN void UnsetComplete() { fut_.is_complete_.store(0); }
  CTP_CROSS_FUN bool IsComplete() const { return fut_.is_complete_.load() != 0; }
  /** sizeof(concrete TaskT) carried with the task (replaces pod_size_). */
  CTP_CROSS_FUN u32 GetTaskSize() const { return fut_.task_size_; }
  CTP_CROSS_FUN void SetTaskSize(u32 v) { fut_.task_size_ = v; }
  /** Thread to wake (EventManager::Signal) when this task completes. */
  CTP_CROSS_FUN u32 WaiterPid() const { return fut_.waiter_pid_; }
  CTP_CROSS_FUN u32 WaiterTid() const { return fut_.waiter_tid_; }
  CTP_CROSS_FUN void SetWaiter(u32 pid, u32 tid) {
    fut_.waiter_pid_ = pid;
    fut_.waiter_tid_ = tid;
  }
  CTP_CROSS_FUN void SetNewData() { is_new_data_.store(1); }
  CTP_CROSS_FUN void UnsetNewData() { is_new_data_.store(0); }
  CTP_CROSS_FUN bool IsNewData() const { return is_new_data_.load() != 0; }

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
   * IMPORTANT: Derived classes that override AggregateOut MUST call
   * Task::AggregateOut(replica_task) first before aggregating their own fields.
   *
   * @param replica_task The replica task to aggregate from
   */
  void AggregateOut(const ctp::ipc::FullPtr<Task>& replica_task) {
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
   * Combine a batched member's INPUTS into this aggregate task (ManyToOne).
   * Counterpart to AggregateOut: AggregateOut merges a replica's OUT fields
   * (N->1 gather), while AggregateIn folds a member's IN fields into the single
   * collective aggregate task that runs once for the batch. Default is a no-op
   * (aggregate = copy of the first member: a barrier/dedup collective). Derived
   * tasks whose collective combines inputs (sum, max, concat, ...) override it.
   *
   * @param member_task The batched member whose inputs are folded in
   */
  void AggregateIn(const ctp::ipc::FullPtr<Task>& member_task) {
    (void)member_task;
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

// FutureShm and Future are defined in clio/future.h (included above)

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

// GPU Future types (must be after clio::run::Future and clio::run::Task are complete)
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
class RunContext {
  // RunContext is the Task's PRIVATE execution-state extension: all data
  // members are private and reached only through Task's accessors (Task is a
  // friend). No code outside Task/RunContext touches these fields, so there is
  // no way to dereference a null RunContext by accident.
  friend class Task;

  /** Per-execution lifecycle flags, packed into one word instead of four
   *  separate bools (is_yielded_ / did_work_ / routed_ / started_). */
  enum RunCtxFlag : u32 {
    RCTX_YIELDED = 1u << 0,   /**< Task is waiting for completion */
    RCTX_DID_WORK = 1u << 1,  /**< Task did work in last execution */
    RCTX_ROUTED = 1u << 2,    /**< RouteTask has placed this task */
    RCTX_STARTED = 1u << 3,   /**< Execution has begun */
  };

  /** Coroutine handle for C++20 stackless coroutines (or Boost fiber handle) */
#ifndef CLIO_ENABLE_BOOST_COROUTINES
  std::coroutine_handle<> coro_handle_;
#else
  clio::run::detail::FiberHandle coro_handle_;
  /** Inline fiber state for the Boost backend (so only the stack is heap
   *  allocated, never the FiberState). coro_handle_ points at this. */
  clio::run::detail::FiberState fiber_state_;
#endif
  u32 worker_id_;               /**< Worker ID executing this task */
  double yield_time_us_;        /**< Time in microseconds for task to yield */
  ctp::Timepoint block_start_; /**< Time when task was blocked (real time) */
  DynamicContainer container_;  /**< Resolved-once handle to the execution
                                 *   container (always the most-recently-
                                 *   upgraded version) */
  TaskLane* lane_;              /**< Current lane being processed */
  void* event_queue_;           /**< Pointer to worker's event queue */
  std::vector<PoolQuery>
      pool_queries_; /**< Pool queries for task distribution */
  std::vector<clio::run::shared_ptr<Task>>
      subtasks_; /**< Replica tasks for this execution (owning handles) */
  // Atomic so SendIn (net_send_worker), RecvOut (net_recv_worker) and
  // FlushStaleStateForNode can update it concurrently without losing
  // increments — a missed bump leaves completed_ < subtasks_.size()
  // forever and the origin task's future never fires, which manifests
  // as a writer hang under heavy 4n+ FPP load.
  std::atomic<u32> completed_replicas_;
  u32 yield_count_;                     /**< Number of times task has yielded */
  Future<Task, CLIO_QUEUE_ALLOC_T>
      future_;                    /**< Future for async completion tracking */
 public:
  // ---- Routing / transport state (formerly the separate FutureShm; now just
  //      public fields of the RunContext, reached via Future::GetFutureShm() ->
  //      &run_ctx_). Set on the server at RecvIn (and on the client at SendIn)
  //      and read at SendOut.
  ClientOrigin origin_;            /**< Origin transport mode (completion path) */
  u32 client_pid_;                 /**< Client PID for per-client routing */
  /** Client's net_key (task vaddr) captured at RecvIn and restored onto
   *  task_id_.net_key_ at SendOut, because AllocLoadTask reassigns the server
   *  task's identity. The ZMQ recv thread keys pending_zmq_futures_ by this, so
   *  the response must carry it for the client to match (else it hangs). */
  uintptr_t client_net_key_;
  ctp::lbm::ShmTransferInfo input_;   /**< SHM transfer info (client -> worker) */
  ctp::lbm::ShmTransferInfo output_;  /**< SHM transfer info (worker -> client) */
  ctp::lbm::Transport* response_transport_; /**< Transport for the response */
  char response_identity_[64];     /**< ZMQ echo-back identity (fallback path) */
  u32 response_identity_len_;
  int response_fd_;                /**< Socket fd for routing response (IPC) */
  ctp::abitfield32_t gpu_flags_;   /**< GPU device-completion bit (gpu2gpu) */
  uintptr_t gpu_task_device_ptr_;  /**< Device addr of the task POD (kDeviceMem) */
  u32 gpu_task_size_;              /**< sizeof(TaskT) for the H2D writeback */

 private:
  std::atomic<bool> is_notified_; /**< Atomic flag to prevent duplicate event
                                     queue additions */
  // Set true by the top-level coroutine's final_suspend when the task
  // completes. The worker reads THIS instead of coro_handle_.done() to detect
  // completion, so it never dereferences a coroutine frame that a cross-thread
  // completion may already have freed — done() on a freed frame GPFLTs (issue
  // #485). The RunContext outlives the frame, so this read is always valid.
  std::atomic<bool> coro_completed_;
  /** is_yielded_ / did_work_ / routed_ / started_ packed into one word. The
   *  true period is NOT duplicated here — Task::TruePeriodNs() reads the task's
   *  own period_ns_ directly. */
  ctp::bitfield32_t flags_;
  ctp::CpuTimer cpu_timer_; /**< Accumulates thread CPU time across yields */
  float predicted_load_ = 0; /**< Predicted CPU time from InferModel (us) */
  ctp::HighResMonotonicTimer wall_timer_; /**< Wall clock time across yields */
  float predicted_wall_us_ =
      0; /**< Predicted wall time from InferWallClockTime */
  TaskStat
      predicted_stat_; /**< TaskStat used for prediction (for reinforcement) */

 public:
  RunContext()
      : coro_handle_(),
        worker_id_(0),
        yield_time_us_(0.0),
        block_start_(),
        container_(),
        lane_(nullptr),
        event_queue_(nullptr),
        completed_replicas_(0),
        yield_count_(0),
        origin_(ClientOrigin::kClientShm),
        client_pid_(0),
        client_net_key_(0),
        response_transport_(nullptr),
        response_identity_len_(0),
        response_fd_(-1),
        gpu_task_device_ptr_(0),
        gpu_task_size_(0),
        is_notified_(false),
        coro_completed_(false),
        flags_() {
    gpu_flags_.Clear();
  }

  /** Sentinel AllocatorId used by SendCpuToGpu to mark ShmPtrs whose offset is
   *  a raw pinned-host address (formerly FutureShm::GetCpu2GpuAllocId). */
  static ctp::ipc::AllocatorId GetCpu2GpuAllocId() {
    ctp::ipc::AllocatorId id;
    id.major_ = UINT32_MAX - 1;
    id.minor_ = 0;
    return id;
  }
  /** GPU device-completion bit (formerly FutureShm::FUTURE_COMPLETE). */
  static constexpr u32 FUTURE_COMPLETE = 1;

  /**
   * Move constructor
   */
  RunContext(RunContext&& other) noexcept
      : coro_handle_(std::move(other.coro_handle_)),
        worker_id_(other.worker_id_),
        yield_time_us_(other.yield_time_us_),
        block_start_(other.block_start_),
        container_(std::move(other.container_)),
        lane_(other.lane_),
        event_queue_(other.event_queue_),
        pool_queries_(std::move(other.pool_queries_)),
        subtasks_(std::move(other.subtasks_)),
        completed_replicas_(other.completed_replicas_.load()),
        yield_count_(other.yield_count_),
        future_(std::move(other.future_)),
        is_notified_(other.is_notified_.load()),
        coro_completed_(other.coro_completed_.load()),
        flags_(other.flags_),
        cpu_timer_(other.cpu_timer_),
        predicted_load_(other.predicted_load_),
        wall_timer_(other.wall_timer_),
        predicted_wall_us_(other.predicted_wall_us_),
        predicted_stat_(other.predicted_stat_) {
#ifndef CLIO_ENABLE_BOOST_COROUTINES
    other.coro_handle_ = nullptr;
#else
    other.coro_handle_ = clio::run::detail::FiberHandle{};
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
      yield_time_us_ = other.yield_time_us_;
      block_start_ = other.block_start_;
      container_ = std::move(other.container_);
      lane_ = other.lane_;
      event_queue_ = other.event_queue_;
      pool_queries_ = std::move(other.pool_queries_);
      subtasks_ = std::move(other.subtasks_);
      completed_replicas_.store(other.completed_replicas_.load());
      yield_count_ = other.yield_count_;
      future_ = std::move(other.future_);
      is_notified_.store(other.is_notified_.load());
      coro_completed_.store(other.coro_completed_.load());
      flags_ = other.flags_;
      cpu_timer_ = other.cpu_timer_;
      predicted_load_ = other.predicted_load_;
      wall_timer_ = other.wall_timer_;
      predicted_wall_us_ = other.predicted_wall_us_;
      predicted_stat_ = other.predicted_stat_;
#ifndef CLIO_ENABLE_BOOST_COROUTINES
      other.coro_handle_ = nullptr;
#else
      other.coro_handle_ = clio::run::detail::FiberHandle{};
#endif
      other.event_queue_ = nullptr;
    }
    return *this;
  }

  /**
   * Destructor: frees the Boost fiber frame this RunContext owns.
   *
   * Boost backend: the fiber state lives INLINE in fiber_state_, so the fiber
   * (and its pooled stack) is freed automatically when this RunContext is
   * destroyed — frame lifetime is tied to the RunContext, and so to the Task
   * that owns it via run_ctx_, with no scattered destroy() in the worker. We
   * detach the FiberHandle first so it cannot be resumed after this point.
   *
   * Stackless backend: coro_handle_ is NOT destroyed here. Unlike the inline
   * fiber state, a C++20 coroutine frame is a separately-heap-allocated object,
   * and coro_handle_ is a *shared slot* that is repointed to whichever (top-
   * level or nested-helper) coroutine is currently active — nested-helper frames
   * are owned by their TaskResume awaiter. So the top-level frame is destroyed
   * by the task driver on completion (see Task::StartCoroutine/ResumeCoroutine),
   * not here.
   */
  ~RunContext() {
#if defined(CLIO_ENABLE_BOOST_COROUTINES)
    coro_handle_ = clio::run::detail::FiberHandle{};
    // fiber_state_ (and its boost::context::fiber) destructs next, freeing the
    // stack back to the pool.
#endif
  }

  // Delete copy constructor and copy assignment
  RunContext(const RunContext&) = delete;
  RunContext& operator=(const RunContext&) = delete;

  // Execution-lifecycle flag accessors. These hold per-execution state that
  // used to live in Task::task_flags_ (TASK_ROUTED / TASK_STARTED) but is
  // runtime-local and must not be serialized with the task.
  // Each flag lives in flags_; setting passes the bit, clearing unsets it.
  void SetFlag(RunCtxFlag f, bool v) {
    if (v) {
      flags_.SetBits(f);
    } else {
      flags_.UnsetBits(f);
    }
  }
  bool IsRouted() const { return flags_.Any(RCTX_ROUTED); }
  void SetRouted(bool v = true) { SetFlag(RCTX_ROUTED, v); }
  bool IsStarted() const { return flags_.Any(RCTX_STARTED); }
  void SetStarted(bool v = true) { SetFlag(RCTX_STARTED, v); }
  bool IsYielded() const { return flags_.Any(RCTX_YIELDED); }
  void SetYielded(bool v = true) { SetFlag(RCTX_YIELDED, v); }
  bool DidWork() const { return flags_.Any(RCTX_DID_WORK); }
  void SetDidWork(bool v = true) { SetFlag(RCTX_DID_WORK, v); }

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
    flags_.UnsetBits(RCTX_DID_WORK);
    cpu_timer_.time_ns_ = 0;
    predicted_load_ = 0;
    wall_timer_.time_ns_ = 0;
    predicted_wall_us_ = 0;
    predicted_stat_ = TaskStat();
  }
};

// ============================================================================
// Task RunContext accessors (defined here, where RunContext is complete).
// Each null-checks run_ctx_ and throws via CLIO_THROW so no caller ever
// dereferences a null RunContext or touches a field directly. Host-only:
// RunContext is incomplete on a device pass.
// ============================================================================
#if CTP_IS_HOST
// Log + throw on a null RunContext so the failure is visible (not a silent
// terminate inside a coroutine). pool/method/task id are logged for context.
#define CLIO_RCTX_NULL(NAME)                                                   \
  HLOG(kError,                                                                 \
       "Task::" #NAME ": null RunContext — task not executing (pool={} "      \
       "method={} task_id={})",                                               \
       pool_id_, method_, task_id_.unique_);                                  \
  CLIO_THROW(std::runtime_error(#NAME ": null RunContext"))
// Value getter: `RET Task::NAME() const { return run_ctx_->FIELD; }`
#define CLIO_RCTX_GET(RET, NAME, FIELD)                                       \
  inline RET Task::NAME() const {                                            \
    if (!run_ctx_) {                                                          \
      CLIO_RCTX_NULL(NAME);                                                   \
    }                                                                        \
    return run_ctx_->FIELD;                                                  \
  }
// Value setter: `void Task::NAME(ARG v) { run_ctx_->FIELD = v; }`
#define CLIO_RCTX_SET(ARG, NAME, FIELD)                                       \
  inline void Task::NAME(ARG v) {                                            \
    if (!run_ctx_) {                                                          \
      CLIO_RCTX_NULL(NAME);                                                   \
    }                                                                        \
    run_ctx_->FIELD = v;                                                     \
  }
// Mutable reference accessor: `RET& Task::NAME() { return run_ctx_->FIELD; }`
#define CLIO_RCTX_REF(RET, NAME, FIELD)                                       \
  inline RET& Task::NAME() {                                                 \
    if (!run_ctx_) {                                                          \
      CLIO_RCTX_NULL(NAME);                                                   \
    }                                                                        \
    return run_ctx_->FIELD;                                                  \
  }

CLIO_RCTX_GET(u32, RunWorkerId, worker_id_)
CLIO_RCTX_SET(u32, SetRunWorkerId, worker_id_)
// Flag accessors delegate to the RunContext's bitfield (flags_); they can't use
// the plain-field CLIO_RCTX_GET/SET macros.
#define CLIO_RCTX_FLAG(GETTER, SETTER)                                          \
  inline bool Task::GETTER() const {                                           \
    if (!run_ctx_) {                                                           \
      CLIO_RCTX_NULL(GETTER);                                                  \
    }                                                                          \
    return run_ctx_->GETTER();                                                 \
  }                                                                            \
  inline void Task::SETTER(bool v) {                                           \
    if (!run_ctx_) {                                                           \
      CLIO_RCTX_NULL(SETTER);                                                  \
    }                                                                          \
    run_ctx_->SETTER(v);                                                       \
  }
CLIO_RCTX_FLAG(IsYielded, SetYielded)
CLIO_RCTX_GET(double, YieldTimeUs, yield_time_us_)
CLIO_RCTX_SET(double, SetYieldTimeUs, yield_time_us_)
CLIO_RCTX_REF(ctp::Timepoint, BlockStart, block_start_)
CLIO_RCTX_REF(DynamicContainer, ExecContainer, container_)
CLIO_RCTX_GET(void*, EventQueue, event_queue_)
CLIO_RCTX_SET(void*, SetEventQueue, event_queue_)
CLIO_RCTX_REF(std::vector<PoolQuery>, PoolQueries, pool_queries_)
CLIO_RCTX_REF(std::vector<clio::run::shared_ptr<Task>>, Subtasks, subtasks_)
CLIO_RCTX_REF(std::atomic<u32>, CompletedReplicas, completed_replicas_)
CLIO_RCTX_GET(u32, YieldCount, yield_count_)
CLIO_RCTX_SET(u32, SetYieldCount, yield_count_)
// The "true period" is the task's own period_ns_ (set via SetPeriod); it is no
// longer duplicated in the RunContext, so this reads the task field directly
// and needs no RunContext.
inline double Task::TruePeriodNs() const { return period_ns_; }
CLIO_RCTX_FLAG(DidWork, SetDidWork)
CLIO_RCTX_FLAG(IsRouted, SetRouted)
CLIO_RCTX_FLAG(IsStarted, SetStarted)
CLIO_RCTX_REF(ctp::CpuTimer, RunCpuTimer, cpu_timer_)
CLIO_RCTX_GET(float, PredictedLoad, predicted_load_)
CLIO_RCTX_SET(float, SetPredictedLoad, predicted_load_)
CLIO_RCTX_REF(ctp::HighResMonotonicTimer, RunWallTimer, wall_timer_)
CLIO_RCTX_GET(float, PredictedWallUs, predicted_wall_us_)
CLIO_RCTX_SET(float, SetPredictedWallUs, predicted_wall_us_)
CLIO_RCTX_REF(TaskStat, PredictedStat, predicted_stat_)
#ifndef CLIO_ENABLE_BOOST_COROUTINES
CLIO_RCTX_REF(std::coroutine_handle<>, CoroHandle, coro_handle_)
#else
CLIO_RCTX_REF(clio::run::detail::FiberHandle, CoroHandle, coro_handle_)
CLIO_RCTX_REF(clio::run::detail::FiberState, FiberStateRef, fiber_state_)
#endif

#undef CLIO_RCTX_GET
#undef CLIO_RCTX_SET
#undef CLIO_RCTX_REF

// BeginRunContext() is defined out-of-line in ipc_manager.cc: it allocates the
// RunContext AND resolves the execution container (GetRealOrStaticContainer),
// which needs the PoolManager singleton not visible here.

// Deduced-return and atomic/aggregate accessors need bespoke bodies.
inline auto &Task::Lane() {
  if (!run_ctx_) {
    CLIO_RCTX_NULL(Lane);
  }
  return run_ctx_->lane_;
}
inline Future<Task, CLIO_QUEUE_ALLOC_T>& Task::RunFuture() {
  if (!run_ctx_) {
    CLIO_RCTX_NULL(RunFuture);
  }
  return run_ctx_->future_;
}
inline RunContext* Task::RunCtxPtr() {
  // Returns null (rather than throwing) when there is no RunContext, so the
  // RunCtx().IsNull() defensive checks across the IPC paths keep working.
  // Callers that must stamp routing (SendIn) ensure a RunContext first
  // (the Future ctor calls EnsureRunCtx()).
  return run_ctx_.get();
}
inline const clio::run::shared_ptr<Task>& Task::GetParentTask() const {
  if (!run_ctx_) {
    CLIO_RCTX_NULL(GetParentTask);
  }
  return run_ctx_->future_.GetParentTask();
}
inline const DynamicContainer& Task::ExecContainer() const {
  if (!run_ctx_) {
    CLIO_RCTX_NULL(ExecContainer);
  }
  return run_ctx_->container_;
}
inline bool Task::IsNotified() const {
  if (!run_ctx_) {
    CLIO_RCTX_NULL(IsNotified);
  }
  return run_ctx_->is_notified_.load();
}
inline void Task::SetNotified(bool v) {
  if (!run_ctx_) {
    CLIO_RCTX_NULL(SetNotified);
  }
  run_ctx_->is_notified_.store(v);
}
inline bool Task::IsCoroCompleted() const {
  if (!run_ctx_) {
    CLIO_RCTX_NULL(IsCoroCompleted);
  }
#ifndef CLIO_ENABLE_BOOST_COROUTINES
  // Stackless: read the flag the top-level coroutine's final_suspend sets
  // (issue #485). Valid even if a cross-thread completion already freed the
  // frame, whereas coro_handle_.done() would be a use-after-free.
  return run_ctx_->coro_completed_.load();
#else
  // Boost fiber path has no such flag and is not subject to the same cross-
  // thread free, so it uses the fiber's own done() state.
  return run_ctx_->coro_handle_ && run_ctx_->coro_handle_.done();
#endif
}
inline void Task::SetCoroCompleted(bool v) {
  if (!run_ctx_) {
    CLIO_RCTX_NULL(SetCoroCompleted);
  }
  run_ctx_->coro_completed_.store(v);
}
inline void Task::ClearRunState() {
  if (!run_ctx_) {
    CLIO_RCTX_NULL(ClearRunState);
  }
  run_ctx_->Clear();
}
#undef CLIO_RCTX_NULL
#endif  // CTP_IS_HOST

// ============================================================================
// Future::await_suspend_impl implementation (must be after RunContext
// definition)
// ============================================================================

// CTP_IS_HOST so this host-only coroutine machinery is excluded from a GPU
// device pass entirely: nvcc parses (and member-checks) the whole TU in the
// device pass, and this body references #if CTP_IS_HOST-only Task accessors
// (CoroHandle/SetYielded/...). The GPU path uses gpu::Future, never this.
#if CTP_IS_HOST && !defined(CLIO_ENABLE_BOOST_COROUTINES)
template <typename TaskT, typename AllocT>
bool Future<TaskT, AllocT>::await_suspend_impl(
    std::coroutine_handle<> handle) noexcept {
  // Get the executing task from the current worker's thread-local storage
  // Uses helper function to avoid circular dependency with worker.h
  clio::run::shared_ptr<Task>& task = GetCurrentTask();

  if (task.IsNull()) {
    // No executing task available, don't suspend
    HLOG(kWarning, "Future::await_suspend: no current task, not suspending!");
    return false;
  }
  // Store parent task for resumption tracking
  SetParentTask(task);
  // Store coroutine handle in the task's RunContext for worker to resume
  task->CoroHandle() = handle;
  task->SetYielded(true);
  task->SetYieldTimeUs(0.0);
  return true;  // Suspend the coroutine
}
#endif // !CLIO_ENABLE_BOOST_COROUTINES

// ============================================================================
// TaskResume and YieldAwaiter (must be after RunContext for member access)
// ============================================================================

// Host-only: the coroutine return type / awaiters touch #if CTP_IS_HOST-only
// Task accessors, and nvcc parses this in the device pass too. Excluded from
// any device pass (the GPU path is producer-only and uses gpu::Future).
#if CTP_IS_HOST
#ifndef CLIO_ENABLE_BOOST_COROUTINES
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
    /** Non-owning pointer to the executing Task for this coroutine. The task
     *  outlives its coroutine frame, and its RunContext is reached through Task
     *  accessors — the coroutine never holds a bare RunContext pointer. */
    Task* task_ = nullptr;
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
      if (is_top_level_ && task_ != nullptr) {
        // SetCoroCompleted is a host-only RunContext accessor (declared under
        // #if CTP_IS_HOST). The C++20 coroutine promise is parsed but never
        // executed on the CUDA device pass (CTP_IS_HOST=0), so guard the call
        // to keep that pass compiling.
#if CTP_IS_HOST
        task_->SetCoroCompleted(true);
#endif
      }
      return FinalAwaiter{caller_handle_};
    }

    /**
     * Handle void return from coroutine
     */
    void return_void() {}

    /**
     * Log the unhandled coroutine exception (e.g. a CLIO_THROW from a null-
     * RunContext accessor) BEFORE terminating, so the failure is visible in the
     * log instead of being a silent std::terminate inside the coroutine frame.
     */
    void unhandled_exception() {
      try {
        std::rethrow_exception(std::current_exception());
      } catch (const std::exception &e) {
        HLOG(kError, "Coroutine task threw an unhandled exception: {}",
             e.what());
      } catch (...) {
        HLOG(kError, "Coroutine task threw an unhandled (non-std) exception");
      }
      std::terminate();
    }

    /**
     * Set the executing Task for this coroutine
     * @param task Non-owning pointer to the Task
     */
    void set_task(Task* task) { task_ = task; }

    /**
     * Get the executing Task for this coroutine
     * @return Non-owning pointer to the Task
     */
    Task* get_task() const { return task_; }

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

    // CRITICAL: Propagate the executing Task from caller to inner coroutine
    // This allows nested co_await on Futures to properly suspend
    Task* caller_task = caller_handle.promise().get_task();
    if (caller_task) {
      handle_.promise().set_task(caller_task);
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
    // Get the executing task from inner's promise before destroying
    Task* task = nullptr;
    if (handle_) {
      task = handle_.promise().get_task();
      // Inner coroutine is done (final_suspend just resumed us), destroy it
      handle_.destroy();
      handle_ = nullptr;
    }

    // Update the task's coro handle to caller's handle so that if the caller
    // suspends again on another co_await, or completes, the worker can handle
    // it. (When the inner completed synchronously inside await_suspend, handle_
    // is already null and the caller never suspended, so there is nothing to
    // restore — coro_handle_ still holds the caller's own handle.)
    if (task != nullptr && caller_handle_) {
      // CoroHandle() is a host-only RunContext accessor; this awaiter is parsed
      // but never executed on the CUDA device pass (CTP_IS_HOST=0).
#if CTP_IS_HOST
      task->CoroHandle() = caller_handle_;
#endif
    }
  }
};

#else // CLIO_ENABLE_BOOST_COROUTINES - Boost.Context fiber-based TaskResume

/**
 * TaskResume (fiber backend) - return type for runtime methods
 *
 * Wraps a clio::run::detail::FiberHandle instead of a coroutine handle.
 * Used by the Boost.Context stackful backend (CLIO_ENABLE_BOOST_COROUTINES).
 */
class TaskResume {
  clio::run::detail::FiberHandle handle_;

public:
  TaskResume() = default;
  explicit TaskResume(clio::run::detail::FiberHandle h) : handle_(std::move(h)) {}

  TaskResume(TaskResume&& o) noexcept : handle_(o.handle_) {
    o.handle_ = clio::run::detail::FiberHandle{};
  }

  TaskResume& operator=(TaskResume&& o) noexcept {
    if (this != &o) {
      if (handle_) handle_.destroy();
      handle_ = o.handle_;
      o.handle_ = clio::run::detail::FiberHandle{};
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

  clio::run::detail::FiberHandle& get_handle() { return handle_; }
  const clio::run::detail::FiberHandle& get_handle() const { return handle_; }

  /**
   * Release ownership of the fiber handle without destroying it
   */
  clio::run::detail::FiberHandle release() {
    auto h = handle_;
    handle_ = clio::run::detail::FiberHandle{};
    return h;
  }

  explicit operator bool() const { return bool(handle_); }
  bool operator!() const { return !handle_; }
};

#endif // CLIO_ENABLE_BOOST_COROUTINES
#endif // CTP_IS_HOST (TaskResume is host-only)

/**
 * YieldAwaiter - Awaitable for yielding control in coroutines
 *
 * This class implements the awaitable interface for cooperative yielding
 * within ChiMod runtime coroutines. It allows tasks to yield control
 * back to the worker with an optional delay before resumption.
 *
 * Usage:
 *   co_await clio::run::yield();       // Yield immediately
 *   co_await clio::run::yield(25.0);   // Yield with 25 microsecond delay
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

#if CTP_IS_HOST && !defined(CLIO_ENABLE_BOOST_COROUTINES)
  /**
   * Suspend the coroutine and mark for yielded resumption
   *
   * @tparam PromiseT The promise type of the calling coroutine
   * @param handle The coroutine handle to resume after yield
   * @return True to suspend, false if no RunContext available
   */
  template <typename PromiseT>
  bool await_suspend(std::coroutine_handle<PromiseT> handle) noexcept {
    Task* task = handle.promise().get_task();
    if (!task) {
      // No executing task available, don't suspend
      return false;
    }
    // Store coroutine handle in the task's RunContext for worker to resume
    task->CoroHandle() = handle;
    task->SetYielded(true);
    task->SetYieldTimeUs(yield_time_us_);
    return true;  // Suspend the coroutine
  }
#endif // !CLIO_ENABLE_BOOST_COROUTINES

  /**
   * Resume after yield - nothing to return
   */
  void await_resume() noexcept {}

  /**
   * Get the yield time in microseconds (used by boost_await on the fiber backend)
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
 *   co_await clio::run::yield();       // Yield immediately
 *   co_await clio::run::yield(25.0);   // Yield with 25 microsecond delay
 */
inline YieldAwaiter yield(double us = 0.0) { return YieldAwaiter(us); }

// Cleanup macros
#undef CLASS_NAME
#undef CLASS_NEW_ARGS

}  // namespace clio::run

// ============================================================================
// Boost fiber backend: boost_await overloads and make_task_fiber
// (outside chi namespace, in clio::run::detail namespace)
// ============================================================================
#ifdef CLIO_ENABLE_BOOST_COROUTINES
namespace clio::run::detail {

// Boost backend model (issue #620): the WORKER creates ONE fiber per task (via
// make_task_fiber, with the entry calling Container::Run) and the whole task —
// including any nested helper coroutines — runs NATIVELY on that fiber's stack.
// CLIO_TASK_BODY_BEGIN/END are therefore empty (no lambda, so reference
// parameters and locals behave exactly like ordinary C++), nested helper calls
// run inline (so awaiting their returned TaskResume is a no-op), and an await
// simply suspends the one fiber back to the worker until the future completes.

// boost_await fetches the running task's RunContext itself, via the single
// exported GetCurrentRunContextFromWorker() (the canonical CTP_THREAD_MODEL-
// backed accessor, same path as CLIO_CUR_WORKER). So CLIO_CO_AWAIT does not
// assume a local `rctx`, and there is no per-DLL "current fiber" thread_local to
// go stale on Windows (issue #620). The fiber to suspend is rctx->fiber_state_.

/// Yield: suspend the running fiber back to the worker.
inline void boost_await(clio::run::YieldAwaiter ya) {
  // Hold the running task by VALUE, not by reference into the worker's
  // current_task_ slot: the worker reassigns that slot (to null, then to other
  // tasks) while this fiber is suspended, and with shared_ptr-owned tasks that
  // reassignment can drop the slot's reference. A by-reference capture would
  // then dangle and the post-resume task->SetYielded(false) would touch a freed
  // (or wrong) task — the boost-backend SEGFAULT on every co_await/yield path
  // (bdev WriteBlocks, the co-aware locks behind cte_query/cte_tag, etc.). The
  // owning copy keeps this task alive across the suspend and is correct
  // regardless of which worker resumes the fiber.
  clio::run::shared_ptr<clio::run::Task> task = clio::run::GetCurrentTask();
  if (task.IsNull()) return;
  task->SetYielded(true);
  task->SetYieldTimeUs(ya.get_yield_time_us());
  fiber_suspend_to_caller(&task->FiberStateRef());
  task->SetYielded(false);
}

/// Future: suspend the fiber until the worker observes the future complete.
template<typename TaskT, typename AllocT>
inline void boost_await(clio::run::Future<TaskT, AllocT>& future) {
  if (future.IsComplete()) return;
  // By value — see the YieldAwaiter overload above for why a reference into
  // current_task_ dangles across the suspend.
  clio::run::shared_ptr<clio::run::Task> task = clio::run::GetCurrentTask();
  if (task.IsNull()) return;
  future.SetParentTask(task);
  task->SetYielded(true);
  task->SetYieldTimeUs(0.0);
  fiber_suspend_to_caller(&task->FiberStateRef());
  task->SetYielded(false);
}

/// Future rvalue overload (for temporaries).
template<typename TaskT, typename AllocT>
inline void boost_await(clio::run::Future<TaskT, AllocT>&& future) {
  boost_await(future);
}

/// Nested helper coroutine: it already ran to completion inline (empty
/// CLIO_TASK_BODY_BEGIN means it executed as a plain call on this fiber's
/// stack, suspending the fiber itself at its own awaits), so there is nothing
/// left to drive.
inline void boost_await(clio::run::TaskResume&&) {}
inline void boost_await(clio::run::TaskResume&) {}

}  // namespace clio::run::detail
#endif // CLIO_ENABLE_BOOST_COROUTINES

// ============================================================================
// Cross-compiler macros for task bodies (co_await / co_return replacements)
// ============================================================================
#if defined(CLIO_ENABLE_BOOST_COROUTINES)
// Boost.Context: the worker runs the whole method natively on one fiber stack
// (issue #620). The body is NOT wrapped in a lambda, so begin/end are empty,
// references/locals behave like ordinary C++, awaits suspend the fiber, and a
// method returns a (trivial) TaskResume by value.
#  define CLIO_TASK_BODY_BEGIN
#  define CLIO_TASK_BODY_END
#  define CLIO_CO_AWAIT(expr)  clio::run::detail::boost_await((expr))
#  define CLIO_CO_RETURN       return clio::run::TaskResume{}
#else
// C++20 stackless coroutines.
#  define CLIO_TASK_BODY_BEGIN
#  define CLIO_TASK_BODY_END
#  define CLIO_CO_AWAIT(expr)  co_await (expr)
#  define CLIO_CO_RETURN       co_return
#endif
// Backward-compat aliases (clio_run rebrand). External code that still
// uses the legacy CHI_* spelling keeps working unchanged.

#endif  // CLIO_RUNTIME_INCLUDE_TASK_H_
