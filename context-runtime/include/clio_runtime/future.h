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

#ifndef CHIMAERA_INCLUDE_CHIMAERA_FUTURE_H_
#define CHIMAERA_INCLUDE_CHIMAERA_FUTURE_H_

#include <coroutine>

#include "clio_runtime/types.h"
#include "clio_ctp/lightbeam/shm_transport.h"
#include "clio_ctp/memory/allocator/allocator.h"
#include "clio_ctp/util/logging.h"

namespace clio::run {

// Forward declarations
class Task;
class IpcManager;
struct RunContext;

// ============================================================================
// FutureShm - Shared memory container for task future state
// ============================================================================

/**
 * FutureShm - Fixed-size shared memory structure for task futures
 *
 * This structure contains metadata and a copy space buffer for task
 * serialization. The copy space is a flexible array member allocated as part of
 * the structure.
 *
 * Memory layout:
 * - Fixed-size header fields (pool_id, method_id, etc.)
 * - Flexible array: char copy_space[]
 *
 * Allocation: AllocateBuffer(sizeof(FutureShm) + copy_space_size)
 */
struct FutureShm {
  // Bitfield flags for flags_
  static constexpr u32 FUTURE_COMPLETE = 1; /**< Task execution is complete */
  static constexpr u32 FUTURE_NEW_DATA = 2; /**< New output data available */
  static constexpr u32 FUTURE_COPY_FROM_CLIENT =
      4; /**< Task needs to be copied from client serialization */
  static constexpr u32 FUTURE_WAS_COPIED =
      8; /**< Task was already copied from client (don't re-copy) */
  static constexpr u32 FUTURE_DEVICE_SCOPE =
      16; /**< GPU->GPU path: use device-scope atomics (no system fence) */
  static constexpr u32 FUTURE_POD_COPY =
      32; /**< POD cudaMemcpy path: no serialization, task is raw memcpy'd */

  // Origin constants: how the client submitted this task
  static constexpr u32 FUTURE_CLIENT_SHM = 0; /**< Client used shared memory */
  static constexpr u32 FUTURE_CLIENT_TCP = 1; /**< Client used ZMQ TCP */
  static constexpr u32 FUTURE_CLIENT_IPC =
      2; /**< Client used ZMQ IPC (Unix domain socket) */
  static constexpr u32 FUTURE_CLIENT_CPU2GPU =
      3; /**< CPU->GPU transfer via cudaMemcpy */
  static constexpr u32 FUTURE_CLIENT_GPU2CPU =
      4; /**< GPU->CPU transfer via cudaMemcpy */

  /**
   * Get the sentinel AllocatorId used by SendCpuToGpu to mark ShmPtrs
   * whose offset is a raw pinned-host address (not an SHM offset).
   * @return AllocatorId sentinel {UINT32_MAX-1, 0}
   */
  CTP_CROSS_FUN static ctp::ipc::AllocatorId GetCpu2GpuAllocId() {
    ctp::ipc::AllocatorId id;
    id.major_ = UINT32_MAX - 1;
    id.minor_ = 0;
    return id;
  }

  /** Pool ID for the task */
  PoolId pool_id_;

  /** Method ID for the task */
  u32 method_id_;

  /** Origin transport mode (FUTURE_CLIENT_SHM, _TCP, or _IPC) */
  u32 origin_;

  /** Virtual address of client's task (for ZMQ response routing) */
  uintptr_t client_task_vaddr_;

  /** Client PID for per-client response routing */
  u32 client_pid_;

  /** SHM transfer info for input direction (client -> worker) */
  ctp::lbm::ShmTransferInfo input_;

  /** SHM transfer info for output direction (worker -> client) */
  ctp::lbm::ShmTransferInfo output_;

  /** Transport to use for sending response back to client */
  ctp::lbm::Transport* response_transport_;

  /** Socket fd for routing response (IPC mode) */
  int response_fd_;

  /** ZMQ identity for routing response back to client (TCP mode) */
  char response_identity_[64];
  u32 response_identity_len_;

  /** Atomic bitfield for completion and data availability flags */
  ctp::abitfield32_t flags_;

  /**
   * Opaque pointer to the parent's GPU RunContext (chi::gpu::RunContext*).
   * Set by Future::await_suspend on GPU so that the worker completing
   * this sub-task can directly resume the parent coroutine (same thread,
   * no event queue needed). Null for top-level (client-originated) tasks.
   * Typed as void* to avoid circular dependency with gpu/container.h.
   */
  void *parent_gpu_rctx_;

  /** Cross-warp: warps increment on done */
  ctp::ipc::atomic<u32> completion_counter_;
  /** Number of warps sharing this FutureShm */
  u32 total_warps_;

  /**
   * GPU device-memory pointer to the *task POD* (set when the kernel
   * placed the Task struct in kDeviceMem). The CPU worker D2H-copies
   * `gpu_task_size_` bytes from here into a host scratch slot for
   * dispatch, and on completion H2D-copies the (mutated) POD bytes
   * back to this address so the kernel sees output fields. Zero when
   * the task is in kPinnedHost / kManagedUvm (host-dereferenceable).
   */
  uintptr_t gpu_task_device_ptr_;

  /**
   * sizeof(TaskT) for the H2D writeback copy. Mirrors
   * gpu::FutureShm::task_size_ which the kernel filled in via Reset.
   */
  u32 gpu_task_size_;

  /**
   * GPU device-memory pointer to the *gpu::FutureShm* co-located with
   * the task. RuntimeSend writes FUTURE_COMPLETE to this address (via
   * cudaMemcpy when the FutureShm itself is in kDeviceMem) so the
   * kernel poll-loop unblocks. Always non-zero on the GPU origin path.
   */
  uintptr_t gpu_fshm_device_ptr_;

  /** Copy space for serialized task data (flexible array member).
   *  Must be 4-byte aligned for WarpMemCpy uint32_t strided access. */
  char copy_space[];

  /**
   * Default constructor - initializes fields
   * Note: copy_space is allocated as part of the buffer, not separately
   */
  CTP_CROSS_FUN FutureShm() {
    pool_id_ = PoolId::GetNull();
    method_id_ = 0;
    origin_ = FUTURE_CLIENT_SHM;
    client_task_vaddr_ = 0;
    client_pid_ = 0;
    response_transport_ = nullptr;
    response_fd_ = -1;
    response_identity_len_ = 0;
    parent_gpu_rctx_ = nullptr;
    completion_counter_.store(0);
    total_warps_ = 1;
    gpu_task_device_ptr_ = 0;
    gpu_task_size_ = 0;
    gpu_fshm_device_ptr_ = 0;
    flags_.Clear();
  }

  /**
   * Lightweight reset for per-task reuse on GPU.
   * Only resets fields that change between tasks or that the
   * orchestrator reads before processing. Avoids redundant atomic
   * stores to fields that stay constant (origin_, response_*, etc.).
   *
   * Call this instead of placement-new when reusing a cached FutureShm.
   */
  CTP_CROSS_FUN void Reset(PoolId pool_id, u32 method_id) {
    pool_id_ = pool_id;
    method_id_ = method_id;
    client_task_vaddr_ = 0;
    parent_gpu_rctx_ = nullptr;
    gpu_task_device_ptr_ = 0;
    gpu_task_size_ = 0;
    gpu_fshm_device_ptr_ = 0;
    flags_.Clear();
    input_.total_written_.store(0);
    input_.total_read_.store(0);
    output_.total_written_.store(0);
    output_.total_read_.store(0);
  }
};

// ============================================================================
// Future - Template class for asynchronous task operations
// ============================================================================

/**
 * Future - Template class for asynchronous task operations
 *
 * Future provides a handle to an asynchronous task operation, allowing
 * the caller to check completion status and retrieve results.
 *
 * @tparam TaskT The task type (e.g., CreateTask, CustomTask)
 * @tparam AllocT The allocator type (defaults to CLIO_QUEUE_ALLOC_T)
 */
template <typename TaskT, typename AllocT = CLIO_QUEUE_ALLOC_T>
class Future {
 public:
  using FutureT = FutureShm;

  // Allow all Future instantiations to access each other's private members
  // This enables the Cast method to work across different task types
  template <typename OtherTaskT, typename OtherAllocT>
  friend class Future;
  friend struct IpcCpu2Self;
  friend struct IpcCpu2Gpu;
  friend struct IpcGpu2Cpu;

 private:
  /** FullPtr to the task (wraps private memory with null allocator) */
  ctp::ipc::FullPtr<TaskT> task_ptr_;

  /** ShmPtr to the shared FutureShm object */
  ctp::ipc::ShmPtr<FutureT> future_shm_;

  /** Parent task RunContext pointer (nullptr if no parent waiting) */
  RunContext* parent_task_;

  /** Whether Destroy(true) was called (via Wait/await_resume) */
  bool consumed_;

  /** Cross-warp sub-range for this future */
  struct Range {
    u32 off;
    u32 width;
    CTP_CROSS_FUN Range() : off(0), width(0) {}
    CTP_CROSS_FUN Range(u32 o, u32 w) : off(o), width(w) {}
  } range_;

  /**
   * Implementation of await_suspend
   * Defined after RunContext to access its members
   */
  bool await_suspend_impl(std::coroutine_handle<> handle) noexcept;

 public:
  /**
   * Constructor from ShmPtr<FutureShm> and FullPtr<Task>
   * @param future_shm ShmPtr to existing FutureShm object
   * @param task_ptr FullPtr to the task (wraps private memory with null
   * allocator)
   */
  CTP_CROSS_FUN Future(ctp::ipc::ShmPtr<FutureT> future_shm,
                        const ctp::ipc::FullPtr<TaskT>& task_ptr)
      : future_shm_(future_shm), parent_task_(nullptr), consumed_(false) {
    // Manually initialize task_ptr_ to avoid FullPtr copy constructor bug on
    // GPU Copy shm_ directly, then reconstruct ptr_ from it
    task_ptr_.shm_ = task_ptr.shm_;
    task_ptr_.ptr_ = task_ptr.ptr_;
  }

  /**
   * Default constructor - creates null future
   */
  CTP_CROSS_FUN Future() : parent_task_(nullptr), consumed_(false) {}

  /**
   * Constructor from ShmPtr<FutureShm> - used by ring buffer deserialization
   * Task pointer will be null and must be set later
   * @param future_shm_ptr ShmPtr to FutureShm object
   */
  CTP_CROSS_FUN explicit Future(const ctp::ipc::ShmPtr<FutureT>& future_shm_ptr)
      : future_shm_(future_shm_ptr), parent_task_(nullptr), consumed_(false) {
    // Task pointer starts null - will be set in ProcessNewTasks
    task_ptr_.SetNull();
  }

  /**
   * Destructor - frees the task if this Future was consumed (via
   * Wait/await_resume). Defined out-of-line in ipc_manager.h where
   * CLIO_IPC is available.
   */
  CTP_CROSS_FUN ~Future();

  /**
   * Destroy the task using CLIO_IPC->DelTask if not null
   * Sets the task pointer to null afterwards
   */
  CTP_CROSS_FUN void Destroy(bool post_wait = false);

  /**
   * Explicitly delete the underlying task via CLIO_IPC->DelTask
   */
  CTP_CROSS_FUN void DelTask();

  /**
   * Copy constructor - does not transfer ownership
   * @param other Future to copy from
   */
  CTP_CROSS_FUN Future(const Future& other)
      : future_shm_(other.future_shm_),
        parent_task_(other.parent_task_),
        consumed_(false) {  // Copy is not consumed
    // Manually copy task_ptr_ to avoid FullPtr copy constructor bug on GPU
    task_ptr_.shm_ = other.task_ptr_.shm_;
    task_ptr_.ptr_ = other.task_ptr_.ptr_;
  }

  /**
   * Copy assignment operator - does not transfer ownership
   * @param other Future to copy from
   * @return Reference to this future
   */
  CTP_CROSS_FUN Future& operator=(const Future& other) {
    if (this != &other) {
      // Manually copy task_ptr_ to avoid FullPtr copy assignment bug on GPU
      task_ptr_.shm_ = other.task_ptr_.shm_;
      task_ptr_.ptr_ = other.task_ptr_.ptr_;
      future_shm_ = other.future_shm_;
      parent_task_ = other.parent_task_;
      consumed_ = false;  // Copy is not consumed
    }
    return *this;
  }

  /**
   * Move constructor - transfers ownership
   * @param other Future to move from
   */
  CTP_CROSS_FUN Future(Future&& other) noexcept
      : future_shm_(std::move(other.future_shm_)),
        parent_task_(other.parent_task_),
        consumed_(other.consumed_) {
    // Manually move task_ptr_ to avoid FullPtr move constructor bug on GPU
    task_ptr_.shm_ = other.task_ptr_.shm_;
    task_ptr_.ptr_ = other.task_ptr_.ptr_;
    other.task_ptr_.SetNull();
    other.parent_task_ = nullptr;
    other.consumed_ = false;
  }

  /**
   * Move assignment operator - transfers ownership
   * @param other Future to move from
   * @return Reference to this future
   */
  CTP_CROSS_FUN Future& operator=(Future&& other) noexcept {
    if (this != &other) {
      // Manually move task_ptr_ to avoid FullPtr move assignment bug on GPU
      task_ptr_.shm_ = other.task_ptr_.shm_;
      task_ptr_.ptr_ = other.task_ptr_.ptr_;
      future_shm_ = std::move(other.future_shm_);
      parent_task_ = other.parent_task_;
      consumed_ = other.consumed_;
      other.task_ptr_.SetNull();
      other.future_shm_.SetNull();
      other.parent_task_ = nullptr;
      other.consumed_ = false;
    }
    return *this;
  }

  /**
   * Get raw pointer to the task
   * @return Pointer to the task object
   */
  CTP_CROSS_FUN TaskT* get() const { return task_ptr_.ptr_; }

  /**
   * Get the FullPtr to the task (non-const version)
   * @return FullPtr to the task object
   */
  ctp::ipc::FullPtr<TaskT>& GetTaskPtr() { return task_ptr_; }

  /**
   * Get the FullPtr to the task (const version)
   * @return FullPtr to the task object
   */
  const ctp::ipc::FullPtr<TaskT>& GetTaskPtr() const { return task_ptr_; }

  /**
   * Fail-loud sanity check for the dereference operators below.
   *
   * Dereferencing a Future whose task_ptr_ is null is always a bug: a
   * correctly built runtime never resumes a coroutine into a null awaited
   * task. A null here means either (a) an ABI/layout mismatch in the runtime
   * libraries from a mismatched toolchain or dependency stack, or (b) an
   * unchecked allocation failure -- await_ready() deliberately treats a null
   * Future as ready so the coroutine resumes instead of hanging, and callers
   * are expected to check IsNull() before dereferencing. Catch it with a
   * clear, actionable FATAL message instead of a silent SIGSEGV deep in
   * coroutine resume. Define CLIO_NO_FUTURE_NULL_CHECK to compile this out on
   * the hot path for maximum-performance builds.
   */
  CTP_CROSS_FUN void CheckDerefNonNull() const {
#if CTP_IS_HOST && !defined(CLIO_NO_FUTURE_NULL_CHECK)
    if (task_ptr_.IsNull()) {
      HLOG(kFatal,
           "Future: dereferenced (-> or *) a Future whose task_ptr_ is null. "
           "A correctly built runtime never resumes into a null awaited task; "
           "this indicates a broken runtime build (ABI/layout mismatch from a "
           "mismatched toolchain or dependency stack) or an unchecked "
           "allocation failure. Check IsNull() before dereferencing -- see "
           "Future::await_ready().");
    }
#endif
  }

  /**
   * Dereference operator - access task members
   * @return Reference to the task object
   */
  CTP_CROSS_FUN TaskT& operator*() const {
    CheckDerefNonNull();
    return *task_ptr_.ptr_;
  }

  /**
   * Arrow operator - access task members
   * @return Pointer to the task object
   */
  CTP_CROSS_FUN TaskT* operator->() const {
    CheckDerefNonNull();
    return task_ptr_.ptr_;
  }

  /** Get the cross-warp range offset */
  CTP_CROSS_FUN u32 RangeOffset() const { return range_.off; }

  /** Get the cross-warp range width */
  CTP_CROSS_FUN u32 RangeWidth() const { return range_.width; }

  /** Set the cross-warp sub-range */
  CTP_CROSS_FUN void SetRange(u32 off, u32 width) {
    range_.off = off;
    range_.width = width;
  }

  /** Get the warp offset index for this future's range */
  CTP_CROSS_FUN u32 GetTaskWarpOffset() const { return range_.off / 32; }

  /**
   * Check if the task is complete.
   * Dispatches to the correct variant based on origin.
   * @return True if task has completed, false otherwise
   */
  CTP_CROSS_FUN bool IsComplete() const;

  /**
   * CPU-to-CPU completion check.
   * Reads FUTURE_COMPLETE via normal atomic load.
   * @return True if task has completed
   */
  CTP_HOST_FUN bool IsCompleteCpu2Cpu() const;

  /**
   * GPU-to-CPU completion check.
   * Reads FUTURE_COMPLETE via system-scope atomic (visible across GPU/CPU).
   * @return True if task has completed
   */
  CTP_HOST_FUN bool IsCompleteGpu2Cpu() const;

  /**
   * CPU-to-GPU completion check.
   * Polls device-resident FutureShm flags via D-to-H memcpy.
   * @return True if task has completed
   */
  CTP_HOST_FUN bool IsCompleteCpu2Gpu() const;

  /**
   * GPU-to-GPU completion check.
   * Reads FUTURE_COMPLETE via device-scope atomic on GPU.
   * @return True if task has completed
   */
  CTP_GPU_FUN bool IsCompleteGpu2Gpu() const;

  /**
   * Wait for task completion (blocking with optional timeout).
   * Automatically dispatches to the correct path based on origin:
   *   WaitCpu2Cpu, WaitGpu2Cpu, WaitCpu2Gpu, or WaitGpu2Gpu.
   * @param max_sec Wait policy:
   *                 -1 = wait indefinitely (default; never times out)
   *                  0 = poll once; return immediately if not yet complete
   *                       (does NOT enter the recv path when incomplete)
   *                 >0 = block up to that many seconds; return false on
   *                       timeout
   * @param reuse_task If true, skip task deletion on destroy so the task
   *                   object can be resubmitted. Caller owns the task lifetime.
   * @return true if task completed, false if not (timeout or non-blocking
   *         poll that found the future incomplete)
   */
  CTP_CROSS_FUN bool Wait(float max_sec = -1, bool reuse_task = false);

  /**
   * CPU-to-CPU wait path (SHM / ZMQ / IPC).
   * Runtime: polls FUTURE_COMPLETE in shared memory.
   * Client: calls IpcManager::Recv() for lightbeam or ZMQ streaming.
   * @param max_sec Maximum seconds to wait (0 = wait indefinitely)
   * @param reuse_task If true, skip task deletion on destroy
   * @return true if task completed, false if timed out
   */
  CTP_HOST_FUN bool WaitCpu2Cpu(float max_sec = 0, bool reuse_task = false);

  /**
   * GPU-to-CPU wait path (GPU future polled from client on host).
   * Polls FUTURE_COMPLETE with system-scope atomics, then deserializes
   * output from the FutureShm ring buffer.
   * @param max_sec Maximum seconds to wait (0 = wait indefinitely)
   * @param reuse_task If true, skip task deletion on destroy
   * @return true if task completed, false if timed out
   */
  CTP_HOST_FUN bool WaitGpu2Cpu(float max_sec = 0, bool reuse_task = false);

  /**
   * CPU-to-GPU wait path (POD transfer via cudaMemcpy).
   * Polls device-resident FutureShm flags via D-to-H copy, then copies
   * the completed task back to host memory.
   * @param max_sec Maximum seconds to wait (0 = wait indefinitely)
   * @param reuse_task If true, skip task deletion on destroy
   * @return true if task completed, false if timed out
   */
  CTP_HOST_FUN bool WaitCpu2Gpu(float max_sec = 0, bool reuse_task = false);

  /**
   * GPU-to-GPU wait path (task submitted and completed entirely on GPU).
   * All warp lanes enter RecvGpu for warp-cooperative deserialization.
   * @param max_sec Maximum seconds to wait (0 = wait indefinitely)
   * @param reuse_task If true, skip task deletion on destroy
   * @return true if task completed, false if timed out
   */
  CTP_GPU_FUN bool WaitGpu2Gpu(float max_sec = 0, bool reuse_task = false);

  /** Wait phase 1: spin until FUTURE_COMPLETE (no deserialization) */
  CTP_CROSS_FUN void WaitPoll(float max_sec = 0, bool reuse_task = false);

  /** Wait phase 2: deserialize output + cleanup (call after WaitPoll) */
  CTP_CROSS_FUN void WaitRecv(float max_sec = 0, bool reuse_task = false);

  /**
   * Mark the task as complete
   */
  void Complete() {
    if (!future_shm_.IsNull()) {
      auto future_shm = GetFutureShm();
      if (!future_shm.IsNull()) {
        future_shm->flags_.SetBits(FutureT::FUTURE_COMPLETE);
      }
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
  CTP_CROSS_FUN bool IsNull() const { return task_ptr_.IsNull(); }

  /**
   * Get the internal ShmPtr to FutureShm (for internal use)
   * @return ShmPtr to the FutureShm object
   */
  CTP_CROSS_FUN ctp::ipc::ShmPtr<FutureT> GetFutureShmPtr() const {
    return future_shm_;
  }

  /**
   * Get the FutureShm FullPtr (for access to copy_space and flags_)
   * Converts the internal ShmPtr to FullPtr using IpcManager
   * @return FullPtr to the FutureShm object
   * Note: Implementation provided in ipc_manager.h where CLIO_IPC is defined
   */
  CTP_CROSS_FUN ctp::ipc::FullPtr<FutureT> GetFutureShm() const;

  /**
   * Get the pool ID from the FutureShm
   * @return Pool ID for the task
   */
  PoolId GetPoolId() const {
    if (future_shm_.IsNull()) {
      return PoolId::GetNull();
    }
    auto future_shm = GetFutureShm();
    if (future_shm.IsNull()) {
      return PoolId::GetNull();
    }
    return future_shm->pool_id_;
  }

  /**
   * Set the pool ID in the FutureShm
   * @param pool_id Pool ID to set
   */
  void SetPoolId(const PoolId& pool_id) {
    if (!future_shm_.IsNull()) {
      auto future_shm = GetFutureShm();
      if (!future_shm.IsNull()) {
        future_shm->pool_id_ = pool_id;
      }
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
    result.consumed_ = false;  // Cast does not transfer ownership
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
  CTP_CROSS_FUN bool await_ready() const noexcept {
    // A null future (e.g. SendGpu allocation failure) is immediately ready
    // to prevent suspending with awaited_fshm_=nullptr, which would resume
    // the coroutine into a null task_ptr_ dereference.
    if (future_shm_.IsNull() && task_ptr_.IsNull()) return true;
    if (IsComplete()) return true;
    if (!task_ptr_.IsNull() &&
        task_ptr_->task_flags_.Any(TASK_FIRE_AND_FORGET)) {
      return true;
    }
    return false;
  }

  /**
   * Suspend the coroutine and register for resumption (coroutine await_suspend)
   *
   * CPU: stores the coroutine handle in RunContext for the worker to resume
   * when the task completes.
   * GPU: stores coroutine handle + FutureShm pointer in RunContext so the
   * Worker can check FUTURE_COMPLETE and resume without spin-waiting.
   *
   * @param handle The coroutine handle to resume when task completes
   * @return True to suspend, false to continue without suspending
   */
  template <typename PromiseT>
  CTP_CROSS_FUN bool await_suspend(
      std::coroutine_handle<PromiseT> handle) noexcept {
#if CTP_IS_HOST
    // Get RunContext via helper function (defined in worker.cc)
    // This avoids needing RunContext to be complete at this point
    return await_suspend_impl(handle);
#else
    // GPU: genuinely suspend -- store handle and FutureShm in RunContext.
    // Also store parent RunContext on FutureShm so the worker completing
    // the sub-task can directly resume the parent (same thread, no polling).
    if constexpr (requires { handle.promise().get_run_context(); }) {
      auto *ctx = handle.promise().get_run_context();
      if (ctx) {
#if CTP_IS_GPU_COMPILER
        u32 lane = threadIdx.x % 32;
#else
        u32 lane = 0;
#endif
        ctx->coro_handles_[lane] = handle;
        ctx->is_yielded_ = true;
        auto fshm_full = GetFutureShm();
        ctx->awaited_fshm_ = fshm_full.IsNull() ? nullptr : fshm_full.ptr_;
        ctx->awaited_task_ = task_ptr_.IsNull() ? nullptr : task_ptr_.ptr_;
        if (!fshm_full.IsNull()) {
          fshm_full.ptr_->parent_gpu_rctx_ = ctx;
        }
        return true;  // Genuinely suspend
      }
    }
    return false;  // Fallback: no RunContext, can't suspend
#endif
  }

  /**
   * Get the result after resumption (coroutine await_resume)
   *
   * CPU: calls Destroy(true) to clean up after worker-driven resumption.
   * GPU: Worker already confirmed FUTURE_COMPLETE; deserialize output
   * and cleanup.
   */
  CTP_CROSS_FUN void await_resume() noexcept {
    if (!task_ptr_.IsNull() &&
        task_ptr_->task_flags_.Any(TASK_FIRE_AND_FORGET)) {
      // Fire-and-forget: detach without destroying. The task is still
      // in-flight on the network; the remote EndTask will clean it up.
      task_ptr_.SetNull();
      future_shm_.SetNull();
      return;
    }
#if CTP_IS_HOST
    Destroy(true);
#else
    // GPU: Worker already deserialized output before resuming this coroutine.
    // Just mark consumed.
    Destroy(true);
#endif
  }
};

}  // namespace clio::run

#endif  // CHIMAERA_INCLUDE_CHIMAERA_FUTURE_H_
