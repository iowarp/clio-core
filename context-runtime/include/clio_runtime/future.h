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

#ifndef CLIO_RUNTIME_INCLUDE_FUTURE_H_
#define CLIO_RUNTIME_INCLUDE_FUTURE_H_

#include <coroutine>
#include <memory>

#include "clio_runtime/types.h"
#include "clio_ctp/lightbeam/shm_transport.h"
#include "clio_ctp/memory/allocator/allocator.h"
#include "clio_ctp/util/logging.h"

namespace clio::run {

// Forward declarations
class Task;
class IpcManager;
struct RunContext;

/**
 * How the client submitted this task — selects the completion/response path.
 * Stored in FutureShm::origin_ and read by Future::IsComplete()/Wait() dispatch.
 */
enum class ClientOrigin : u32 {
  kClientShm = 0,      ///< Client used shared memory
  kClientTcp = 1,      ///< Client used ZMQ TCP
  kClientIpc = 2,      ///< Client used ZMQ IPC (Unix domain socket)
  kClientCpu2Gpu = 3,  ///< CPU->GPU transfer via cudaMemcpy
  kClientGpu2Cpu = 4,  ///< GPU->CPU transfer via cudaMemcpy
};

// ============================================================================
// FutureShm - Shared memory container for task future state
// ============================================================================

/**
 * FutureShm - Fixed-size structure for task future state
 *
 * This structure contains task metadata and completion/transfer state. It is a
 * plain fixed-size POD: the host-side Future owns it through a
 * std::shared_ptr<FutureShm> (allocated with std::make_shared in the Future
 * constructor), and device code references it via an offset-based ShmPtr. The
 * former flexible-array `copy_space[]` member has been removed — the MPSC SHM
 * transport carries serialized task bytes over the named MPSC server, not an
 * inline copy buffer.
 */
struct FutureShm {
  // flags_ now carries only the GPU device-completion bit. CPU/host completion
  // moved to Task::is_complete_ / Task::is_new_data_.
  static constexpr u32 FUTURE_COMPLETE = 1; /**< Task execution is complete */

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

  /** Origin transport mode — selects the completion/response path. */
  ClientOrigin origin_;

  /** Virtual address of client's task (for ZMQ response routing) */
  uintptr_t client_task_vaddr_;

  /** Client PID for per-client response routing */
  u32 client_pid_;

  /** PID + TID of the thread blocked in Recv waiting for this task to
   *  complete. The completer (SendOut) passes these to the transport via
   *  LbmContext and, after a brief busy-wait, calls EventManager::Signal to
   *  wake the waiter — replacing the busy-poll on FUTURE_COMPLETE. For an
   *  external client this is the client thread; for a runtime self-send it is
   *  the awaiting worker. 0 = no waiter registered (busy-poll fallback). */
  u32 waiter_pid_;
  u32 waiter_tid_;

  /** SHM transfer info for input direction (client -> worker) */
  ctp::lbm::ShmTransferInfo input_;

  /** SHM transfer info for output direction (worker -> client) */
  ctp::lbm::ShmTransferInfo output_;

  /**
   * Transport used to return this task's response to the originating client.
   * Fast path (TCP): a dedicated dial-back DEALER, built (or fetched from the
   * IpcManager connection cache) at RecvIn from the sender's "hostname:pid"
   * identity + the archive's client_port_, then used verbatim by SendOut. A
   * DEALER has a single peer so it auto-routes with no identity frame.
   * Fallback path (TCP): when dial-back is impossible — the client advertised
   * no response port (client_port_ == 0, e.g. the synchronous fallback-runtime
   * client) or its routing identity is not a parseable "hostname:pid" (e.g. a
   * ZMQ auto-assigned identity) — this points at the inbound ROUTER and the
   * response is echoed back over it using response_identity_ (see below). Non-
   * owning — lifetime is held by IpcManager's connection cache / the server.
   */
  ctp::lbm::Transport* response_transport_;

  /**
   * Captured ZMQ routing identity for the echo-back fallback path. Populated
   * at RecvIn only when response_transport_ is the inbound ROUTER (dial-back
   * was not possible); SendOut prepends it as the ROUTER identity frame so the
   * response is delivered over the same socket the request arrived on. Length 0
   * means the dial-back DEALER is in use and no identity frame is needed.
   */
  char response_identity_[64];
  u32 response_identity_len_;

  /** Socket fd for routing response (IPC mode) */
  int response_fd_;

  /** GPU device-completion bit (gpu2gpu reads it on-device; gpu2cpu signals
   *  the device gpu::FutureShm). CPU/host completion lives on Task instead. */
  ctp::abitfield32_t flags_;

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
   * the task. SendOut writes FUTURE_COMPLETE to this address (via
   * cudaMemcpy when the FutureShm itself is in kDeviceMem) so the
   * kernel poll-loop unblocks. Always non-zero on the GPU origin path.
   */
  uintptr_t gpu_fshm_device_ptr_;

  /**
   * Default constructor - initializes fields
   */
  CTP_CROSS_FUN FutureShm() {
    pool_id_ = PoolId::GetNull();
    method_id_ = 0;
    origin_ = ClientOrigin::kClientShm;
    client_task_vaddr_ = 0;
    client_pid_ = 0;
    waiter_pid_ = 0;
    waiter_tid_ = 0;
    response_transport_ = nullptr;
    response_identity_len_ = 0;
    response_fd_ = -1;
    gpu_task_device_ptr_ = 0;
    gpu_task_size_ = 0;
    gpu_fshm_device_ptr_ = 0;
    flags_.Clear();
  }

  /**
   * Parameterized constructor - initializes fields and stamps the task
   * identity. Used by the Future constructor (via std::make_shared) so the
   * pool/method are set at construction; transports then set origin_ and the
   * client/waiter routing fields directly on the FutureShm.
   */
  CTP_CROSS_FUN FutureShm(PoolId pool_id, u32 method_id) : FutureShm() {
    pool_id_ = pool_id;
    method_id_ = method_id;
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
  /** Owning handle to the task.
   *  Host: clio::run::shared_ptr (RAII, refcounted, private MallocAllocator) --
   *  the Future is an owner; the last owner frees the task automatically.
   *  Device: an offset/raw FullPtr (shared_ptr/make_shared are host-only), and
   *  the kernel never owns/frees tasks. */
#if CTP_IS_HOST
  clio::run::shared_ptr<TaskT> task_ptr_;
#else
  ctp::ipc::FullPtr<TaskT> task_ptr_;
#endif

  /**
   * Handle to the FutureShm.
   * Host: the Future OWNS the FutureShm via a std::shared_ptr created with
   * std::make_shared in the parameterized constructor; copies/moves/Cast share
   * the same object and the last owner frees it. Device: an offset-based ShmPtr
   * (std::shared_ptr / make_shared are illegal in device code), unchanged.
   */
#if CTP_IS_HOST
  std::shared_ptr<FutureT> future_shm_;
#else
  ctp::ipc::ShmPtr<FutureT> future_shm_;
#endif

  /** Null-check on future_shm_ that works for both the shared_ptr (host) and
   *  the ShmPtr (device) representations. */
  CTP_HOST_FUN bool FutureShmIsNull() const {
#if CTP_IS_HOST
    return future_shm_ == nullptr;
#else
    return future_shm_.IsNull();
#endif
  }

  /** Null-out future_shm_ for both representations. */
  CTP_HOST_FUN void FutureShmSetNull() {
#if CTP_IS_HOST
    future_shm_ = nullptr;
#else
    future_shm_.SetNull();
#endif
  }

  /** Raw task pointer (host: shared_ptr::get; device: FullPtr::ptr_). */
  CTP_HOST_FUN TaskT* TaskRaw() const {
#if CTP_IS_HOST
    return task_ptr_.get();
#else
    return task_ptr_.ptr_;
#endif
  }

  /** Drop/clear the task handle (host: shared_ptr::reset; device:
   *  FullPtr::SetNull). On host this releases this Future's reference. */
  CTP_HOST_FUN void TaskSetNull() {
#if CTP_IS_HOST
    task_ptr_.reset();
#else
    task_ptr_.SetNull();
#endif
  }

  /** Parent task whose coroutine resumes when this future completes (null if no
   *  parent waiting). Stores the owning Task handle — not a RunContext pointer —
   *  so RunContext is never held outside its Task. */
  clio::run::shared_ptr<Task> parent_task_;

  /** Whether Destroy(true) was called (via Wait/await_resume) */
  bool consumed_;

  /** Cross-warp sub-range for this future */
  struct Range {
    u32 off;
    u32 width;
    CTP_HOST_FUN Range() : off(0), width(0) {}
    CTP_HOST_FUN Range(u32 o, u32 w) : off(o), width(w) {}
  } range_;

  /**
   * Implementation of await_suspend
   * Defined after RunContext to access its members
   */
  bool await_suspend_impl(std::coroutine_handle<> handle) noexcept;

 public:
  /**
   * Parameterized constructor (host) - creates and OWNS a fresh FutureShm.
   *
   * Allocates the FutureShm via std::make_shared (host-only — make_shared is
   * illegal in device code) stamped with the task identity. The transport that
   * builds the Future then sets origin_ and the client/waiter routing fields
   * directly on GetFutureShm(). Copies/moves of this Future share ownership of
   * the same FutureShm; the last owner frees it.
   *
   * @param pool_id Pool ID for the task
   * @param method_id Method ID for the task
   * @param task_ptr FullPtr to the task (wraps private memory with null
   * allocator)
   */
#if CTP_IS_HOST
  Future(PoolId pool_id, u32 method_id,
         clio::run::shared_ptr<TaskT> task_ptr)
      : task_ptr_(std::move(task_ptr)),
        future_shm_(std::make_shared<FutureT>(pool_id, method_id)),
        parent_task_(),
        consumed_(false) {}
#endif

#if !CTP_IS_HOST
  /**
   * Constructor from ShmPtr<FutureShm> and FullPtr<Task> (device).
   * @param future_shm ShmPtr to existing FutureShm object
   * @param task_ptr FullPtr to the task
   */
  CTP_HOST_FUN Future(ctp::ipc::ShmPtr<FutureT> future_shm,
                        const ctp::ipc::FullPtr<TaskT>& task_ptr)
      : future_shm_(future_shm), parent_task_(), consumed_(false) {
    task_ptr_.shm_ = task_ptr.shm_;
    task_ptr_.ptr_ = task_ptr.ptr_;
  }

  /**
   * Constructor from ShmPtr<FutureShm> (device) - task pointer set later.
   * @param future_shm_ptr ShmPtr to FutureShm object
   */
  CTP_HOST_FUN explicit Future(const ctp::ipc::ShmPtr<FutureT>& future_shm_ptr)
      : future_shm_(future_shm_ptr), parent_task_(), consumed_(false) {
    task_ptr_.SetNull();
  }
#endif

  /**
   * Default constructor - creates null future
   */
  CTP_HOST_FUN Future() : parent_task_(), consumed_(false) {}

  /**
   * Destructor - drops this Future's reference to the task (RAII via the
   * shared_ptr member on host) and cleans up the response archive if consumed.
   * Defined out-of-line in ipc_manager.h where CLIO_IPC is available.
   */
  CTP_HOST_FUN ~Future();

  /**
   * Mark the future consumed (calls PostWait on the task). The task is freed
   * automatically when its last shared_ptr owner drops — no explicit delete.
   */
  CTP_HOST_FUN void Destroy(bool post_wait = false);

  /**
   * Copy constructor - does not transfer ownership
   * @param other Future to copy from
   */
  CTP_HOST_FUN Future(const Future& other)
      :
#if CTP_IS_HOST
        task_ptr_(other.task_ptr_),  // shares ownership (increments refcount)
#endif
        future_shm_(other.future_shm_),
        parent_task_(other.parent_task_),
        consumed_(false) {  // Copy is not consumed
#if !CTP_IS_HOST
    // Manually copy task_ptr_ to avoid FullPtr copy constructor bug on GPU
    task_ptr_.shm_ = other.task_ptr_.shm_;
    task_ptr_.ptr_ = other.task_ptr_.ptr_;
#endif
  }

  /**
   * Copy assignment operator - does not transfer ownership
   * @param other Future to copy from
   * @return Reference to this future
   */
  CTP_HOST_FUN Future& operator=(const Future& other) {
    if (this != &other) {
#if CTP_IS_HOST
      task_ptr_ = other.task_ptr_;  // shares ownership (refcount)
#else
      // Manually copy task_ptr_ to avoid FullPtr copy assignment bug on GPU
      task_ptr_.shm_ = other.task_ptr_.shm_;
      task_ptr_.ptr_ = other.task_ptr_.ptr_;
#endif
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
  CTP_HOST_FUN Future(Future&& other) noexcept
      :
#if CTP_IS_HOST
        task_ptr_(std::move(other.task_ptr_)),  // transfers ownership
#endif
        future_shm_(std::move(other.future_shm_)),
        parent_task_(other.parent_task_),
        consumed_(other.consumed_) {
#if !CTP_IS_HOST
    // Manually move task_ptr_ to avoid FullPtr move constructor bug on GPU
    task_ptr_.shm_ = other.task_ptr_.shm_;
    task_ptr_.ptr_ = other.task_ptr_.ptr_;
    other.task_ptr_.SetNull();
#endif
    other.parent_task_.reset();
    other.consumed_ = false;
  }

  /**
   * Move assignment operator - transfers ownership
   * @param other Future to move from
   * @return Reference to this future
   */
  CTP_HOST_FUN Future& operator=(Future&& other) noexcept {
    if (this != &other) {
#if CTP_IS_HOST
      task_ptr_ = std::move(other.task_ptr_);  // transfers ownership
#else
      // Manually move task_ptr_ to avoid FullPtr move assignment bug on GPU
      task_ptr_.shm_ = other.task_ptr_.shm_;
      task_ptr_.ptr_ = other.task_ptr_.ptr_;
      other.task_ptr_.SetNull();
#endif
      future_shm_ = std::move(other.future_shm_);
      parent_task_ = other.parent_task_;
      consumed_ = other.consumed_;
      other.FutureShmSetNull();
      other.parent_task_.reset();
      other.consumed_ = false;
    }
    return *this;
  }

  /**
   * Get raw pointer to the task
   * @return Pointer to the task object
   */
  CTP_HOST_FUN TaskT* get() const { return TaskRaw(); }

  /**
   * Get the owning task handle.
   * Host: clio::run::shared_ptr<TaskT>&. Device: FullPtr<TaskT>&.
   */
#if CTP_IS_HOST
  clio::run::shared_ptr<TaskT>& GetTaskPtr() { return task_ptr_; }
  const clio::run::shared_ptr<TaskT>& GetTaskPtr() const { return task_ptr_; }

  /**
   * Non-owning FullPtr view of the task (null allocator) for transport /
   * serialization code that still speaks FullPtr. Does not affect ownership.
   */
  ctp::ipc::FullPtr<TaskT> GetTaskFullPtr() const {
    return ctp::ipc::FullPtr<TaskT>(task_ptr_.get());
  }
#else
  ctp::ipc::FullPtr<TaskT>& GetTaskPtr() { return task_ptr_; }
  const ctp::ipc::FullPtr<TaskT>& GetTaskPtr() const { return task_ptr_; }
  CTP_HOST_FUN ctp::ipc::FullPtr<TaskT> GetTaskFullPtr() const {
    return task_ptr_;
  }
#endif

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
  CTP_HOST_FUN void CheckDerefNonNull() const {
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
  CTP_HOST_FUN TaskT& operator*() const {
    CheckDerefNonNull();
    return *TaskRaw();
  }

  /**
   * Arrow operator - access task members
   * @return Pointer to the task object
   */
  CTP_HOST_FUN TaskT* operator->() const {
    CheckDerefNonNull();
    return TaskRaw();
  }

  /** Get the cross-warp range offset */
  CTP_HOST_FUN u32 RangeOffset() const { return range_.off; }

  /** Get the cross-warp range width */
  CTP_HOST_FUN u32 RangeWidth() const { return range_.width; }

  /** Set the cross-warp sub-range */
  CTP_HOST_FUN void SetRange(u32 off, u32 width) {
    range_.off = off;
    range_.width = width;
  }

  /** Get the warp offset index for this future's range */
  CTP_HOST_FUN u32 GetTaskWarpOffset() const { return range_.off / 32; }

  /**
   * Check if the task is complete.
   * Dispatches to the correct variant based on origin.
   * @return True if task has completed, false otherwise
   */
  CTP_HOST_FUN bool IsComplete() const;

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
  CTP_HOST_FUN bool Wait(float max_sec = -1, bool reuse_task = false);

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
  CTP_HOST_FUN void WaitPoll(float max_sec = 0, bool reuse_task = false);

  /** Wait phase 2: deserialize output + cleanup (call after WaitPoll) */
  CTP_HOST_FUN void WaitRecv(float max_sec = 0, bool reuse_task = false);

  /**
   * Mark the task as complete. CPU/host completion lives on Task::is_complete_
   * (managed by the Future); the worker/client reads it via IsComplete().
   */
  void Complete() {
    if (!task_ptr_.IsNull()) {
      task_ptr_->SetComplete();
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
  CTP_HOST_FUN bool IsNull() const { return task_ptr_.IsNull(); }

  /**
   * Get the internal ShmPtr to FutureShm (for internal use)
   * @return ShmPtr to the FutureShm object
   */
  CTP_HOST_FUN ctp::ipc::ShmPtr<FutureT> GetFutureShmPtr() const {
#if CTP_IS_HOST
    // Host owns the FutureShm via shared_ptr; synthesize a null-allocator
    // ShmPtr whose offset is the raw address so .IsNull()/offset consumers
    // keep working.
    ctp::ipc::ShmPtr<FutureT> p;
    if (future_shm_ == nullptr) {
      p.SetNull();
      return p;
    }
    p.alloc_id_.SetNull();
    p.off_ = reinterpret_cast<size_t>(future_shm_.get());
    return p;
#else
    return future_shm_;
#endif
  }

  /**
   * Get the FutureShm as a FullPtr (for access to flags_ and routing fields).
   * Host: wraps the shared_ptr's raw pointer with a null allocator. Device:
   * resolves the ShmPtr offset.
   * @return FullPtr to the FutureShm object
   * Note: Implementation provided in ipc_manager.h where CLIO_IPC is defined
   */
  CTP_HOST_FUN ctp::ipc::FullPtr<FutureT> GetFutureShm() const;

  /**
   * Get the pool ID from the FutureShm
   * @return Pool ID for the task
   */
  PoolId GetPoolId() const {
    if (FutureShmIsNull()) {
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
    if (!FutureShmIsNull()) {
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
   * Get the parent task (whose coroutine resumes when this future completes).
   * @return Owning handle to the parent Task (null if none)
   */
  clio::run::shared_ptr<Task>& GetParentTask() { return parent_task_; }
  const clio::run::shared_ptr<Task>& GetParentTask() const { return parent_task_; }

  /**
   * Set the parent task (whose coroutine resumes when this future completes).
   * @param parent_task Owning handle to the parent Task
   */
  void SetParentTask(const clio::run::shared_ptr<Task>& parent_task) {
    parent_task_ = parent_task;
  }

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
  CTP_HOST_FUN bool await_ready() const noexcept {
    // A null future (e.g. SendGpu allocation failure) is immediately ready
    // to prevent suspending with awaited_fshm_=nullptr, which would resume
    // the coroutine into a null task_ptr_ dereference.
    if (FutureShmIsNull() && task_ptr_.IsNull()) return true;
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
  CTP_HOST_FUN bool await_suspend(
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
  CTP_HOST_FUN void await_resume() noexcept {
    if (!task_ptr_.IsNull() &&
        task_ptr_->task_flags_.Any(TASK_FIRE_AND_FORGET)) {
      // Fire-and-forget: detach without destroying. The task is still
      // in-flight on the network; the remote EndTask will clean it up. On host
      // this drops our local reference (the serialized copy lives remotely).
      TaskSetNull();
      FutureShmSetNull();
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

#endif  // CLIO_RUNTIME_INCLUDE_FUTURE_H_
