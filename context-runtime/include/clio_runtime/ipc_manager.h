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

#ifndef CHIMAERA_INCLUDE_CHIMAERA_MANAGERS_IPC_MANAGER_H_
#define CHIMAERA_INCLUDE_CHIMAERA_MANAGERS_IPC_MANAGER_H_

#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "clio_ctp/data_structures/priv/array_vector.h"
#include "clio_ctp/memory/allocator/round_robin_allocator.h"
#include "clio_ctp/util/gpu_intrinsics.h"
#include "clio_runtime/manager.h"
#include "clio_runtime/corwlock.h"
#include "clio_runtime/scheduler/scheduler.h"
#include "clio_runtime/task.h"
#include "clio_runtime/task_archives.h"
#include "clio_runtime/types.h"
#include "clio_runtime/worker.h"
#include "clio_runtime/ipc/ipc_cpu2self.h"
#include "clio_runtime/ipc/ipc_cpu2cpu.h"
#include "clio_runtime/ipc/ipc_cpu2cpu_zmq.h"
#include "clio_runtime/ipc/ipc_gpu2cpu.h"
#include "clio_ctp/data_structures/serialization/serialize_common.h"
#include "clio_ctp/lightbeam/transport_factory_impl.h"
#include "clio_ctp/memory/backend/posix_shm_mmap.h"
#include "clio_runtime/gpu/gpu_info.h"

#if CTP_ENABLE_CUDA || CTP_ENABLE_ROCM || CTP_ENABLE_SYCL
#include "clio_runtime/gpu/gpu_ipc_manager.h"
#endif
#if CTP_ENABLE_CUDA || CTP_ENABLE_ROCM
#include "clio_ctp/memory/allocator/arena_allocator.h"
#include "clio_ctp/memory/backend/gpu_malloc.h"
#include "clio_ctp/memory/backend/gpu_shm_mmap.h"
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
#endif

namespace clio::run {

// Forward declaration of gpu::IpcManager
namespace gpu { class IpcManager; }

// Forward declarations — full definitions in local_task_archives.h,
// included after CLIO_IPC is defined at the bottom of this header.
template <typename BufferT> class LocalSaveTaskArchive;
template <typename BufferT> class LocalLoadTaskArchive;
using DefaultSaveArchive = LocalSaveTaskArchive<chi::priv::vector<char>>;
using DefaultLoadArchive = LocalLoadTaskArchive<chi::priv::vector<char>>;
enum class LocalMsgType : uint8_t;

/**
 * Network queue priority levels for send operations
 */
enum class NetQueuePriority : u32 {
  // Cross-node sends are split into latency vs I/O lanes by the
  // routing layer (RouteGlobal / Worker::EndTask) using
  // container->GetTaskStats(task).io_size_:
  //   io_size_ <  kLargeIOThreshold (4 KiB) -> kSendIn/OutLatency
  //   io_size_ >= kLargeIOThreshold          -> kSendIn/OutIO
  // The Send periodic drains all latency-lane tasks first, then a
  // bounded byte budget (~8 MiB) of IO-lane tasks, then re-checks
  // latency. This is the same admission control the scheduler uses
  // for routing tasks to scheduler vs I/O workers, applied to the
  // network plane: SWIM heartbeats (io_size_ == 0) and small ACKs
  // never sit behind a 1 MiB PutBlob.
  kSendInLatency = 0,  ///< Cross-node forward, <4 KiB (probes, metadata)
  kSendInIO = 1,       ///< Cross-node forward, >=4 KiB (bulk data)
  kSendOutLatency = 2, ///< Cross-node response, <4 KiB
  kSendOutIO = 3,      ///< Cross-node response, >=4 KiB
  kClientSendTcp = 4,  ///< Client response via TCP
  kClientSendIpc = 5,  ///< Client response via IPC
};
static constexpr u32 kNetQueueNumPriorities = 6;
static constexpr size_t kNetQueueIoThreshold = 4096;        // 4 KiB
static constexpr size_t kNetQueueIoByteBudget = 8u << 20;   // 8 MiB / tick

/**
 * Network queue for storing Future<SendTask> objects
 * One lane with two priorities (SendIn and SendOut)
 */
using NetQueue = ctp::ipc::multi_mpsc_ring_buffer<Future<Task>, CLIO_QUEUE_ALLOC_T>;

/**
 * Typedef for worker queue type to simplify usage
 */
using WorkQueue = chi::ipc::mpsc_ring_buffer<ctp::ipc::ShmPtr<TaskLane>>;

/**
 * Metadata for client <-> server communication via lightbeam
 * Compatible with lightbeam Send/RecvMetadata via duck typing
 * (has send, recv, send_bulks, recv_bulks fields)
 */
struct ClientTaskMeta {
  std::vector<ctp::lbm::Bulk> send;
  std::vector<ctp::lbm::Bulk> recv;
  size_t send_bulks = 0;
  size_t recv_bulks = 0;
  std::vector<char> wire_data;

  template <class Archive>
  void serialize(Archive &ar) {
    ar(send, recv, send_bulks, recv_bulks, wire_data);
  }
};

/**
 * Information about a per-process shared memory segment
 * Used for registering client memory with the runtime
 */
struct ClientShmInfo {
  std::string shm_name;        // Shared memory name (chimaera_{pid}_{count})
  pid_t owner_pid;             // PID of the owning process
  u32 shm_index;               // Index within the owner's shm segments
  size_t size;                 // Size of the shared memory segment
  ctp::ipc::AllocatorId alloc_id;  // Allocator ID for this segment

  ClientShmInfo() : owner_pid(0), shm_index(0), size(0) {}

  ClientShmInfo(const std::string &name, pid_t pid, u32 idx, size_t sz,
                const ctp::ipc::AllocatorId &id)
      : shm_name(name),
        owner_pid(pid),
        shm_index(idx),
        size(sz),
        alloc_id(id) {}

  /**
   * Serialization support for cereal
   */
  template <class Archive>
  void serialize(Archive &ar) {
    ar(shm_name, owner_pid, shm_index, size, alloc_id.major_, alloc_id.minor_);
  }
};

// IpcManagerGpuInfo and IpcManagerGpu are defined in gpu_info.h

/**
 * IPC Manager singleton for inter-process communication
 *
 * Manages ZeroMQ server using lightbeam from CTP, three memory segments,
 * and priority queues for task processing.
 * Uses CTP global cross pointer variable singleton pattern.
 */
class IpcManager {
  friend struct IpcCpu2Self;
  friend struct IpcCpu2Cpu;
  friend struct IpcCpu2CpuZmq;
#if CTP_ENABLE_CUDA || CTP_ENABLE_ROCM || CTP_ENABLE_SYCL
  friend struct IpcGpu2Cpu;
#endif

 public:
  /**
   * Initialize client components
   * @return true if initialization successful, false otherwise
   */
  bool ClientInit();

  /**
   * Initialize server/runtime components
   * @return true if initialization successful, false otherwise
   */
  bool ServerInit();

  /**
   * Client finalize - does nothing for now
   */
  void ClientFinalize();

  /**
   * Reset transport objects before workers are destroyed.
   * Must be called after StopWorkers() but before WorkOrchestrator::Finalize()
   * to avoid use-after-free: transports hold raw EventManager* pointers that
   * are owned by Worker objects freed during Finalize().
   */
  void ClearTransports();

  /**
   * Server finalize - cleanup all IPC resources
   */
  void ServerFinalize();




  /**
   * Returns the per-thread task allocator (CLIO_TASK_ALLOC_T*).
   * Host: main_allocator_ (MultiProcessAllocator)
   */
  CLIO_TASK_ALLOC_T *GetMainAllocator() {
    return main_allocator_;
  }

  // GetIpcManagerGpuInfo / GetIpcManagerGpu were the orchestrator-info
  // accessors and have been removed. Use
  // GetGpuIpcManager()->GetGpuInfo(gpu_id) to obtain per-device info.


  /**
   * Host-side stubs for GPU warp utilities (always lane 0 / warp 0)
   * These are used by host code that still references these methods
   * (e.g., Future templates). GPU code uses gpu::IpcManager versions.
   */
  static inline u32 GetWarpId() { return 0; }
  static inline u32 GetLaneId() { return 0; }
  static inline bool IsWarpScheduler() { return true; }

  /**
   * Base task allocation: Task + append_size extra bytes.
   * The caller is responsible for constructing any co-located objects
   * (FutureShm, RunContext, etc.) in the appended region.
   */
  template <typename TaskT, typename... Args>
  ctp::ipc::FullPtr<TaskT> NewTaskBase(size_t append_size,
                                   Args &&...args) {
    (void)append_size;
    TaskT *ptr = new TaskT(std::forward<Args>(args)...);
    return ctp::ipc::FullPtr<TaskT>(ptr);
  }

  /**
   * Create a task for client or orchestrator use.
   */
  template <typename TaskT, typename... Args>
  ctp::ipc::FullPtr<TaskT> NewTask(Args &&...args) {
    TaskT *ptr = new TaskT(std::forward<Args>(args)...);
    ptr->pod_size_ = static_cast<u32>(sizeof(TaskT));
    ctp::ipc::FullPtr<TaskT> result(ptr);
    return result;
  }

  /**
   * Allocate a task for orchestrator-side execution (copy path).
   * Same as NewTask, but with explicit stack_size parameter.
   * Called by autogenerated AllocTaskImpl in LocalAllocLoadDeser.
   */
  template <typename TaskT, typename... Args>
  ctp::ipc::FullPtr<TaskT> NewTaskExec(size_t stack_size,
                                   Args &&...args) {
    (void)stack_size;
    return NewTask<TaskT>(std::forward<Args>(args)...);
  }

  /**
   * Delete a task
   * Destructor + operator delete
   */
  template <typename TaskT>
  void DelTask(ctp::ipc::FullPtr<TaskT> task_ptr) {
    if (task_ptr.IsNull()) return;
    task_ptr.ptr_->~TaskT();
    void *raw = static_cast<void *>(task_ptr.ptr_);
    ::operator delete(raw);
  }

  /**
   * Delete a heap-allocated object.
   * Destructor + operator delete.
   * @param obj_ptr FullPtr to object to delete
   */
  template <typename T>
  void DelObj(ctp::ipc::FullPtr<T> obj_ptr) {
    if (obj_ptr.IsNull()) return;
    obj_ptr.ptr_->~T();
    void *raw = static_cast<void *>(obj_ptr.ptr_);
    ::operator delete(raw);
  }

  /**
   * Allocate buffer in appropriate memory segment
   * Client uses cdata segment, runtime uses rdata segment
   * Yields until buffer is allocated successfully
   * @param size Size in bytes to allocate
   * @return FullPtr<char> to allocated memory
   */
  FullPtr<char> AllocateBuffer(size_t size);

  /**
   * Push a bump arena for fast allocation.
   * All subsequent AllocateBuffer/NewObj/NewTask calls will bump-allocate
   * from this arena until it is popped (RAII).
   *
   * @param size Arena size in bytes
   * @return Arena RAII handle
   */
  ctp::ipc::Arena<ctp::ipc::RoundRobinAllocator> PushArena(size_t size);

  // AllocateDeviceData / AllocateGpuBuffer were removed along with the
  // GPU runtime concept. Kernel-side buffer allocation now goes
  // through chi::gpu::IpcManager::AllocateBuffer (carved out of the
  // gpu2cpu_copy_backend), and host-side device allocation uses
  // ctp::GpuApi::Malloc directly.

  /**
   * Free buffer from appropriate memory segment
   * Uses allocator's Free method
   * @param buffer_ptr FullPtr to buffer to free
   *
   * Host body lives in ipc_manager.cc. Under any device pass we provide an
   * inline empty stub so DPC++'s SYCL device pass can parse through call
   * sites in chimod ~Task destructors (traced via the autogen alloc kernel)
   * without an unresolved external reference.
   */
#if !CTP_IS_DEVICE_PASS
  void FreeBuffer(FullPtr<char> buffer_ptr);
#else
  CTP_INLINE_CROSS_FUN void FreeBuffer(FullPtr<char> /*buffer_ptr*/) {}
#endif

  /**
   * Free buffer from appropriate memory segment (ctp::ipc::ShmPtr<> overload)
   * Converts ctp::ipc::ShmPtr<> to FullPtr<char> and calls the main FreeBuffer
   * @param buffer_ptr ctp::ipc::ShmPtr<> to buffer to free
   */
  void FreeBuffer(ctp::ipc::ShmPtr<char> buffer_ptr) {
    if (buffer_ptr.IsNull()) {
      return;
    }
    // Convert ctp::ipc::ShmPtr<> to FullPtr<char> and call main FreeBuffer
    ctp::ipc::FullPtr<char> full_ptr(ToFullPtr<char>(buffer_ptr));
    FreeBuffer(full_ptr);
  }

  /**
   * Allocate and construct an object using placement new
   * Combines AllocateBuffer and placement new construction
   * @tparam T Type of object to construct
   * @tparam Args Constructor argument types
   * @param args Constructor arguments
   * @return FullPtr<T> to constructed object
   */
  template <typename T, typename... Args>
  ctp::ipc::FullPtr<T> NewObj(Args &&...args) {
    // Allocate from bulk buffer
    ctp::ipc::FullPtr<char> buffer = AllocateBuffer(sizeof(T));
    if (buffer.IsNull()) {
      return ctp::ipc::FullPtr<T>();
    }
    T *obj = new (buffer.ptr_) T(std::forward<Args>(args)...);
    return buffer.Cast<T>();
  }

  /**
   * Create Future by copying/serializing task
   * Serializes the task into FutureShm's copy_space
   *
   * @tparam TaskT Task type (must derive from Task)
   * @param task_ptr Task to serialize into Future
   * @return Future<TaskT> with serialized task data
   */
  template <typename TaskT>
  Future<TaskT> MakeCopyFuture(ctp::ipc::FullPtr<TaskT> task_ptr) {
    if (task_ptr.IsNull()) {
      return Future<TaskT>();
    }

    // Allocate FutureShm with copy_space (lightbeam handles the data transfer)
    size_t copy_space_size = task_ptr->GetCopySpaceSize();
    if (copy_space_size == 0) copy_space_size = KILOBYTES(4);
    size_t alloc_size = sizeof(FutureShm) + copy_space_size;
    ctp::ipc::FullPtr<char> buffer = AllocateBuffer(alloc_size);
    if (buffer.IsNull()) {
      return Future<TaskT>();
    }

    // Construct FutureShm in-place
    FutureShm *future_shm_ptr = new (buffer.ptr_) FutureShm();
    future_shm_ptr->pool_id_ = task_ptr->pool_id_;
    future_shm_ptr->method_id_ = task_ptr->method_;
    future_shm_ptr->origin_ = FutureShm::FUTURE_CLIENT_SHM;
    future_shm_ptr->client_task_vaddr_ =
        reinterpret_cast<uintptr_t>(task_ptr.ptr_);
    future_shm_ptr->input_.copy_space_size_ = copy_space_size;
    future_shm_ptr->flags_.SetBits(FutureShm::FUTURE_COPY_FROM_CLIENT);

    // Create and return Future
    ctp::ipc::ShmPtr<FutureShm> future_shm_shmptr =
        buffer.shm_.template Cast<FutureShm>();
    return Future<TaskT>(future_shm_shmptr, task_ptr);
  }





  /**
   * Create Future by wrapping task pointer (runtime-only, no serialization)
   * Used by runtime workers to avoid unnecessary copying
   *
   * @tparam TaskT Task type (must derive from Task)
   * @param task_ptr Task to wrap in Future
   * @return Future<TaskT> wrapping task pointer directly
   */
  template <typename TaskT>
  Future<TaskT> MakePointerFuture(ctp::ipc::FullPtr<TaskT> task_ptr) {
    // Check task_ptr validity
    if (task_ptr.IsNull()) {
      return Future<TaskT>();
    }

    // Allocate and construct FutureShm (no copy_space for runtime path)
    ctp::ipc::FullPtr<FutureShm> future_shm = NewObj<FutureShm>();
    if (future_shm.IsNull()) {
      return Future<TaskT>();
    }

    // Initialize FutureShm fields
    future_shm.ptr_->pool_id_ = task_ptr->pool_id_;
    future_shm.ptr_->method_id_ = task_ptr->method_;
    future_shm.ptr_->origin_ = FutureShm::FUTURE_CLIENT_SHM;
    future_shm.ptr_->client_task_vaddr_ = 0;
    // No copy_space in runtime path — ShmTransferInfo defaults are fine

    // Create Future with ShmPtr and task_ptr (no serialization)
    Future<TaskT> future(future_shm.shm_, task_ptr);
    return future;
  }

  /**
   * Create a Future for a task with optional serialization
   * Used internally by Send and as a public interface for future creation
   *
   * Two execution paths:
   * - Client thread (IsClientThread=true): Serialize the task into the Future
   * - Runtime thread (IsClientThread=false): Wrap task_ptr directly without
   * serialization
   *
   * @tparam TaskT Task type (must derive from Task)
   * @param task_ptr Task to wrap in Future
   * @return Future<TaskT> wrapping the task
   */
  template <typename TaskT>
  Future<TaskT> MakeFuture(const ctp::ipc::FullPtr<TaskT> &task_ptr) {
    bool is_runtime = CLIO_RUNTIME_MANAGER->IsRuntime();
    Worker *worker = CLIO_CUR_WORKER;

    // Runtime path requires BOTH IsRuntime AND worker to be non-null
    bool use_runtime_path = is_runtime && worker != nullptr;

    if (!use_runtime_path) {
      // CLIENT PATH: Use MakeCopyFuture to serialize the task
      return MakeCopyFuture(task_ptr);
    } else {
      // RUNTIME PATH: Use MakePointerFuture to wrap pointer without
      // serialization
      return MakePointerFuture(task_ptr);
    }
  }

  /**
   * Send task asynchronously (serializes into Future)
   * Creates a Future wrapper, serializes task inputs, and enqueues to worker
   *
   * Two execution paths:
   * - Client thread (IsClientThread=true): Serialize task and copy Future with
   * null task pointer
   * - Runtime thread (IsClientThread=false): Create Future with task pointer
   * directly (no copy)
   *
   * @param task_ptr Task to send
   * @param awake_event Whether to awaken worker after enqueueing
   * @return Future<TaskT> for polling completion and retrieving results
   */

  template <typename TaskT>
  Future<TaskT> Send(const ctp::ipc::FullPtr<TaskT> &task_ptr,
                     bool awake_event = true) {
    bool is_runtime = CLIO_RUNTIME_MANAGER->IsRuntime();
    Worker *worker = CLIO_CUR_WORKER;

    // Client TCP/IPC path
    if (!is_runtime && ipc_mode_ != IpcMode::kShm) {
      return IpcCpu2CpuZmq::ClientSend(this, task_ptr, ipc_mode_);
    }

    // Client SHM path
    if (!is_runtime) {
      return IpcCpu2Cpu::ClientSend(this, task_ptr);
    }

    // Runtime self-send: enqueue task by pointer (no serialization)
    Future<Task> base_future =
        IpcCpu2Self::ClientSend(this, task_ptr.template Cast<Task>());
    return base_future.Cast<TaskT>();
  }


  /**
   * Receive a task on the runtime side: deserialize from Future if needed.
   * Handles origin-based dispatch (SHM, GPU2CPU, CPU2GPU, etc.).
   *
   * @param future Future containing the task
   * @param container Container for deserialization
   * @param method_id Method ID for task allocation
   * @param recv_transport SHM transport for receiving serialized data
   * @return FullPtr to the deserialized/retrieved task
   */
  ctp::ipc::FullPtr<Task> RecvRuntime(
      Future<Task> &future, Container *container, u32 method_id,
      ctp::lbm::Transport *recv_transport);

  /**
   * Send the runtime response back to the client after task execution.
   * Serializes outputs, completes futures, and handles cleanup for each
   * origin transport mode (SHM, TCP, IPC, GPU2CPU, CPU2GPU).
   *
   * @param task_ptr Executed task
   * @param run_ctx RunContext with future and execution state
   * @param container Container for serialization
   * @param send_transport SHM transport for sending serialized data
   */
  void SendRuntime(const FullPtr<Task> &task_ptr, RunContext *run_ctx,
                   Container *container,
                   ctp::lbm::Transport *send_transport);

  /**
   * Initialize RunContext for a task before routing.
   * @param future Future containing the task
   * @param container Container for the task (can be nullptr)
   * @param lane Lane for the task (can be nullptr)
   */
  void BeginTask(Future<Task> &future, Container *container, TaskLane *lane);

  /** Route a task: resolve pool query, determine local vs global.
   * If force_enqueue is true, always enqueue to the destination worker's lane
   * (used by EnqueueRuntime which cannot execute tasks directly). */
  RouteResult RouteTask(Future<Task> &future, bool force_enqueue = false);

  /** Resolve a pool query into concrete physical addresses */
  std::vector<PoolQuery> ResolvePoolQuery(const PoolQuery &query,
                                          PoolId pool_id,
                                          const FullPtr<Task> &task_ptr);

  /** Check if task should be processed locally.
   *  @param originally_local True iff the user-facing API call was made
   *    with PoolQuery::Local() — snapshotted in RouteTask BEFORE
   *    ScheduleTask (Dynamic → concrete) or ResolvePoolQuery (DirectHash /
   *    DirectId → Local boundary-case rewrite) gets a chance to mutate
   *    the query. Lets CLIO_FORCE_NET honor explicit Local requests while
   *    pushing every other query through the network path. */
  bool IsTaskLocal(const FullPtr<Task> &task_ptr,
                   const std::vector<PoolQuery> &pool_queries,
                   bool originally_local);

  /** Route task locally.
   * If force_enqueue is true, always enqueue even if dest == current worker. */
  RouteResult RouteLocal(Future<Task> &future, bool force_enqueue = false);

  /** Route task globally via network */
  RouteResult RouteGlobal(Future<Task> &future,
                          const std::vector<PoolQuery> &pool_queries);

  // RouteToGpu / cpu→gpu dispatch removed along with the GPU runtime.

  /**
   * Receive task results on the client side.
   * Dispatches to the appropriate transport class based on origin.
   */
  template <typename TaskT>
  bool Recv(Future<TaskT> &future, float max_sec = 0) {
    bool is_runtime = CLIO_RUNTIME_MANAGER->IsRuntime();
    if (is_runtime) return true;

    auto future_shm = future.GetFutureShm();
    u32 origin = future_shm->origin_;

    if (origin == FutureShm::FUTURE_CLIENT_SHM && server_alive_.load()) {
      return IpcCpu2Cpu::ClientRecv(this, future, max_sec);
    }
    return IpcCpu2CpuZmq::ClientRecv(this, future, max_sec);
  }

  /**
   * Set the IsClientThread flag for the current thread
   * @param is_client_thread true if thread is running client code, false
   * otherwise
   */
  void SetIsClientThread(bool is_client_thread);

  /**
   * Get the IsClientThread flag for the current thread
   * @return true if thread is running client code, false otherwise
   */
  bool GetIsClientThread() const;

  /**
   * Get TaskQueue for task processing
   * @return Pointer to the TaskQueue or nullptr if not available
   */
  TaskQueue *GetTaskQueue();

  /**
   * Check if IPC manager is initialized
   * @return true if initialized, false otherwise
   */
  bool IsInitialized() const;

  /**
   * Get the current IPC transport mode
   * @return IpcMode enum value (kTcp, kIpc, or kShm)
   */
  IpcMode GetIpcMode() const { return ipc_mode_; }

  /**
   * Get the server's generation counter
   * @return Server generation value, 0 if not available
   */
  u64 GetServerGeneration() const {
    return server_generation_.load(std::memory_order_acquire);
  }

  u64 GetWorkerQueuesOffset() const { return worker_queues_off_; }

  /**
   * Check if the runtime server process is alive
   * SHM mode: checks runtime PID via kill(pid, 0)
   * Other modes: returns true (assume alive until timeout)
   * @return true if server is believed alive
   */
  bool IsServerAlive() const;

  /** Check cached server liveness (set by heartbeat thread) */
  bool IsServerAliveCache() const {
    return server_alive_.load(std::memory_order_acquire);
  }

  /** Background heartbeat thread function */
  void HeartbeatThread();

  /**
   * Reconnect to a restarted server (all transports)
   * Re-attaches SHM, re-verifies server via ClientConnectTask
   * @return true if reconnection succeeded
   */
  bool ReconnectToOriginalHost();

  /**
   * Wait for server to come back and reconnect
   * Polls with 1-second intervals up to client_retry_timeout_
   * @param start Time point when the wait started (for overall timeout)
   * @return true if reconnection succeeded within timeout
   */
  bool WaitForServerAndReconnect(std::chrono::steady_clock::time_point start);

  /**
   * Reconnect the ZMQ transport to a different host.
   * Stops recv thread, destroys old transport, creates new TCP transport
   * to new_addr, restarts recv thread, verifies connectivity.
   * Forces ipc_mode_ to kTcp (SHM/IPC are same-machine only).
   * @param new_addr IP address of the new host
   * @return true if successfully connected to the new host
   */
  bool ReconnectToNewHost(const std::string &new_addr);

  /**
   * Get number of workers from shared memory header
   * @return Number of workers, 0 if not initialized
   */
  u32 GetWorkerCount();

  /**
   * Get number of scheduling queues from shared memory header
   * @return Number of scheduling queues, 0 if not initialized
   */
  u32 GetNumSchedQueues() const;

  /**
   * Set number of scheduling queues in shared memory header
   * Called by scheduler after DivideWorkers to inform IpcManager of actual
   * scheduler worker count
   * @param num_sched_queues Number of scheduler workers that process tasks
   */
  void SetNumSchedQueues(u32 num_sched_queues);

  /**
   * Awaken a worker by sending a signal to its thread
   * Sends SIGUSR1 to the worker's thread ID stored in the TaskLane
   * Only sends signal if the worker is inactive (blocked in epoll_wait)
   * @param lane Pointer to the TaskLane containing the worker's tid and active
   * status
   */
  void AwakenWorker(TaskLane *lane);

  /**
   * Set the node ID in the shared memory header
   * @param hostname Hostname string to hash and store
   */
  void SetNodeId(const std::string &hostname);

  /**
   * Get the node ID from the shared memory header
   * @return 64-bit node ID, 0 if not initialized
   */
  u64 GetNodeId() const;

  /**
   * Load hostfile and populate hostfile map
   * Uses hostfile path from ConfigManager
   * @return true if loaded successfully, false otherwise
   */
  bool LoadHostfile();

  /**
   * Get Host struct by node ID
   * @param node_id 64-bit node ID
   * @return Pointer to Host struct if found, nullptr otherwise
   */
  const Host *GetHost(u64 node_id) const;

  /**
   * Get Host struct by IP address
   * @param ip_address IP address string
   * @return Pointer to Host struct if found, nullptr otherwise
   */
  const Host *GetHostByIp(const std::string &ip_address) const;

  /**
   * Get all hosts from hostfile
   * @return Const reference to vector of all Host structs
   */
  const std::vector<Host> &GetAllHosts() const;

  /**
   * Get number of hosts in the cluster
   * @return Number of hosts
   */
  size_t GetNumHosts() const;

  /**
   * Check if a node is believed to be alive
   * @param node_id Node to check
   * @return true if alive, false if dead or unknown
   */
  bool IsAlive(u64 node_id) const;

  /**
   * Mark a node as dead and record it for retry tracking
   * Removes cached client connections for the dead node
   * @param node_id Node to mark as dead
   */
  void SetDead(u64 node_id);

  /**
   * Mark a node as alive and remove it from dead-node tracking
   * @param node_id Node to mark as alive
   */
  void SetAlive(u64 node_id);

  /**
   * Get the SWIM node state for a node
   * @param node_id Node to query
   * @return NodeState (kDead for unknown nodes)
   */
  NodeState GetNodeState(u64 node_id) const;

  /**
   * Set the SWIM node state and update state_changed_at timestamp
   * @param node_id Node to update
   * @param new_state New state to set
   */
  void SetNodeState(u64 node_id, NodeState new_state);

  /**
   * Set self-fenced status (partition detection)
   * @param fenced true if this node should fence itself
   */
  void SetSelfFenced(bool fenced);

  /**
   * Check if this node is self-fenced
   * @return true if self-fenced
   */
  bool IsSelfFenced() const { return self_fenced_; }

  /**
   * Get the leader node ID (lowest alive node_id)
   * All nodes compute the same leader deterministically from local state
   */
  u64 GetLeaderNodeId() const;

  /**
   * Check if this node is the current leader
   */
  bool IsLeader() const;

  struct DeadNodeEntry {
    u64 node_id;
    std::chrono::steady_clock::time_point detected_at;
  };

  /**
   * Get the list of dead nodes for retry queue scanning
   * @return Const reference to dead_nodes_ vector
   */
  const std::vector<DeadNodeEntry> &GetDeadNodes() const { return dead_nodes_; }

  /**
   * Add a new node to the internal hostfile
   * @param ip_address IP address of the new node
   * @param port Port of the new node's runtime
   * @return Assigned node ID for the new node
   */
  u64 AddNode(const std::string &ip_address, u32 port);

  /**
   * Identify current host from hostfile by attempting TCP server binding
   * Uses hostfile path from ConfigManager
   * @return true if host identified successfully, false otherwise
   */
  bool IdentifyThisHost();

  /**
   * Get current hostname identified during host identification
   * @return Current hostname string
   */
  const std::string &GetCurrentHostname() const;

  /**
   * Set lane mapping policy for task distribution
   * @param policy Lane mapping policy to use
   */
  /**
   * Get the main ZeroMQ server for network communication
   * @return Pointer to main server or nullptr if not initialized
   */
  ctp::lbm::Transport *GetMainTransport() const;

  /**
   * Get this host identified during host identification
   * @return Const reference to this Host struct
   */
  const Host &GetThisHost() const;

  /**
   * Get the lightbeam server for receiving client tasks
   * @param mode IPC mode (kTcp or kIpc)
   * @return Lightbeam Server pointer, or nullptr
   */
  ctp::lbm::Transport *GetClientTransport(IpcMode mode) const;

  /**
   * Client-side thread that receives completed task outputs via lightbeam
   */
  void RecvZmqClientThread();

  /**
   * Clean up a response archive and its zmq_msg_t handles
   * Called from Future::Destroy() to free zero-copy recv buffers
   * @param net_key Net key (client_task_vaddr_) used as map key
   */
  void CleanupResponseArchive(size_t net_key);

  /**
   * Start local ZeroMQ server
   * Uses ZMQ port + 1 for local server operations
   * Must be called after ServerInit completes to ensure runtime is ready
   * @return true if successful, false otherwise
   */
  bool StartLocalServer();

  /**
   * Convert ShmPtr to FullPtr by checking allocator IDs
   * Handles three cases:
   * 1. AllocatorId::GetNull() - offset is the actual memory address (raw
   * pointer)
   * 2. Main allocator - runtime shared memory for queues/futures
   * 3. Per-process shared memory allocators via alloc_map_
   * Acquires reader lock on allocator_map_lock_ for thread-safe access
   * @param shm_ptr The ShmPtr to convert
   * @return FullPtr with matching allocator and pointer, or null FullPtr if no
   * match
   */
  template <typename T>
  ctp::ipc::FullPtr<T> ToFullPtr(const ctp::ipc::ShmPtr<T> &shm_ptr) {
    // Full allocator lookup implementation
    // Case 1: AllocatorId is null - offset IS the raw memory address
    // This is used for private memory allocations (new/delete)
    if (shm_ptr.alloc_id_ == ctp::ipc::AllocatorId::GetNull()) {
      // The offset field contains the raw pointer address
      T *raw_ptr = reinterpret_cast<T *>(shm_ptr.off_.load());
      return ctp::ipc::FullPtr<T>(raw_ptr);
    }

    // Case 2: Check main allocator (runtime shared memory)
    if (main_allocator_ && shm_ptr.alloc_id_ == main_allocator_->GetId()) {
      return ctp::ipc::FullPtr<T>(main_allocator_, shm_ptr);
    }

    // Case 3: Check per-process shared memory allocators via alloc_map_
    // Acquire reader lock for thread-safe access to allocator_map_
    allocator_map_lock_.ReadLock();

    // Convert AllocatorId to lookup key (combine major and minor)
    u64 alloc_key = (static_cast<u64>(shm_ptr.alloc_id_.major_) << 32) |
                    static_cast<u64>(shm_ptr.alloc_id_.minor_);
    auto it = alloc_map_.find(alloc_key);
    ctp::ipc::FullPtr<T> result;
    if (it != alloc_map_.end()) {
      result = ctp::ipc::FullPtr<T>(it->second, shm_ptr);
    }

    // Release the lock before continuing
    allocator_map_lock_.ReadUnlock();
    if (result.ptr_ != nullptr) return result;

    // Case 4: Check GPU client backends (kPinnedHost / kManagedUvm /
    // kDeviceMem). The IpcGpu2Cpu producer convention stashes the raw
    // device-or-host-accessible address in `off_` directly, so resolution
    // is the same as the null-alloc_id case once we've confirmed the
    // alloc_id refers to a registered GPU backend. Callers that operate
    // on the resolved ptr_ via DeviceAwareMemcpy (which dispatches
    // through cudaMemcpyDefault / hipMemcpyDefault / sycl::queue::memcpy)
    // can copy from kDeviceMem pointers without first staging through
    // the host.
#if (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM || CTP_ENABLE_SYCL) && CTP_IS_HOST
    if (gpu_ipc_) {
      size_t ngpu = gpu_ipc_->GetGpuQueueCount();
      for (size_t g = 0; g < ngpu; ++g) {
        if (gpu_ipc_->FindClientBackend(static_cast<u32>(g),
                                         shm_ptr.alloc_id_)) {
          T *raw_ptr = reinterpret_cast<T *>(shm_ptr.off_.load());
          return ctp::ipc::FullPtr<T>(raw_ptr);
        }
      }
    }
#endif

    return result;
  }

  /**
   * Convert raw pointer to FullPtr by checking allocators
   * Uses ContainsPtr() on each allocator to find the matching one
   * Checks main allocator first, then per-process allocators
   * If no allocator contains the pointer, returns a FullPtr with null allocator
   * (private memory)
   * Acquires reader lock on allocator_map_lock_ for thread-safe access
   * @param ptr The raw pointer to convert
   * @return FullPtr with matching allocator and pointer, or FullPtr with null
   * allocator if no match (private memory)
   */
  template <typename T>
  ctp::ipc::FullPtr<T> ToFullPtr(T *ptr) {
    // Full allocator lookup implementation
    if (ptr == nullptr) {
      return ctp::ipc::FullPtr<T>();
    }

    // Check main allocator
    if (main_allocator_ && main_allocator_->ContainsPtr(ptr)) {
      return ctp::ipc::FullPtr<T>(main_allocator_, ptr);
    }

    // Check per-process shared memory allocators
    // Acquire reader lock for thread-safe access
    allocator_map_lock_.ReadLock();

    ctp::ipc::FullPtr<T> result;
    for (auto *alloc : alloc_vector_) {
      if (alloc && alloc->ContainsPtr(ptr)) {
        result = ctp::ipc::FullPtr<T>(alloc, ptr);
        allocator_map_lock_.ReadUnlock();
        return result;
      }
    }

    // Release the lock before returning
    allocator_map_lock_.ReadUnlock();

    // No matching allocator found - treat as private memory
    // Return FullPtr with the raw pointer (null allocator ID)
    return ctp::ipc::FullPtr<T>(ptr);
  }

  /**
   * Get or create a persistent ZeroMQ client connection from the pool
   * Creates a new connection if one doesn't exist for the given address:port
   * Thread-safe using internal mutex protection
   * @param addr IP address to connect to
   * @param port Port number to connect to
   * @return Pointer to the ZeroMQ client (owned by the pool)
   */
  ctp::lbm::Transport *GetOrCreateClient(const std::string &addr, int port);

  /**
   * Clear all cached client connections
   * Should be called during shutdown
   */
  void ClearClientPool();

  /**
   * Set the net worker's lane pointers for signaling on EnqueueNetTask.
   * Called by scheduler after DivideWorkers assigns the net workers.
   *
   * With the recv/send split, the cross-node Send priorities
   * (kSendIn{Latency,IO} / kSendOut{Latency,IO}) are drained by the
   * send worker (peer DEALERs), while kClientSendTcp / kClientSendIpc
   * are drained by the recv worker (client-facing ROUTER).
   * EnqueueNetTask picks the right lane to awaken based on the
   * priority enqueued.
   *
   * Passing both lanes equal (single-net-worker fallback) is fine — the
   * same worker handles both directions.
   */
  void SetNetLane(TaskLane *send_lane, TaskLane *recv_lane) {
    net_send_lane_ = send_lane;
    net_recv_lane_ = recv_lane;
    net_lane_ = send_lane;  // back-compat for callers that haven't updated
  }

  /**
   * Enqueue a Future<SendTask> to the network queue.
   * @param future Future containing the SendTask to enqueue
   * @param priority Network queue priority (see NetQueuePriority for
   *                 the latency-vs-IO lane split).
   */
  void EnqueueNetTask(Future<Task> future, NetQueuePriority priority);

  /**
   * Try to pop a Future<SendTask> from the network queue
   * @param priority Network queue priority to pop from
   * @param future Output parameter for the popped Future
   * @return true if a Future was popped, false if queue is empty
   */
  bool TryPopNetTask(NetQueuePriority priority, Future<Task> &future);

  /**
   * Approximate number of tasks currently queued on the given priority.
   * Snapshot at call time (MPSC head/tail atomics, no lock); producers
   * may push more after this returns. Used by the Send periodic to
   * bound its drain loop to the depth seen on entry — prevents one
   * priority's hot stream from monopolising the tick at the expense
   * of the other priorities, retries, and SWIM probes that share the
   * same periodic.
   * @return Current size of lane 0 at the given priority, or 0 if the
   *         net_queue_ isn't initialized.
   */
  size_t GetNetQueueSize(NetQueuePriority priority) const {
    if (net_queue_.IsNull()) {
      return 0;
    }
    return net_queue_->GetLane(0, static_cast<u32>(priority)).Size();
  }

  /**
   * Get the network queue for direct access
   * @return Pointer to the network queue or nullptr if not initialized
   */
  NetQueue *GetNetQueue() { return net_queue_.ptr_; }

  /**
   * Get number of GPU→CPU queues (one per GPU device).
   * Forwards to gpu::IpcManager::per_gpu_devices_ — there is no longer a
   * separate gpu_queues_ vector on chi::IpcManager.
   */
  size_t GetGpuQueueCount() const {
#if CTP_IS_HOST && (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM || CTP_ENABLE_SYCL)
    if (gpu_ipc_) return gpu_ipc_->GetGpuQueueCount();
#endif
    return 0;
  }

  /**
   * Get GPU→CPU queue by index (CPU worker polls this).
   */
  GpuTaskQueue *GetGpuQueue(size_t gpu_id) {
#if CTP_IS_HOST && (CTP_ENABLE_CUDA || CTP_ENABLE_ROCM || CTP_ENABLE_SYCL)
    if (gpu_ipc_) return gpu_ipc_->GetGpuQueue(static_cast<u32>(gpu_id));
#endif
    (void)gpu_id;
    return nullptr;
  }

#if CTP_ENABLE_CUDA || CTP_ENABLE_ROCM || CTP_ENABLE_SYCL
  /** Get the GPU IPC manager for direct access to GPU operations. */
  gpu::IpcManager *GetGpuIpcManager() { return gpu_ipc_.get(); }
  const gpu::IpcManager *GetGpuIpcManager() const { return gpu_ipc_.get(); }
#endif  // CTP_ENABLE_CUDA || CTP_ENABLE_ROCM || CTP_ENABLE_SYCL

  /**
   * Assign every per-device gpu2cpu_queue lane to the GPU worker so it
   * polls them. Call after gpu_ipc_->ServerInitGpuQueues() completes.
   */
  void AssignGpuLanesToWorker();

  /**
   * Get the scheduler instance
   * IpcManager is the single owner of the scheduler.
   * WorkOrchestrator and Worker should use this method to get the scheduler.
   * @return Pointer to the scheduler or nullptr if not initialized
   */
  Scheduler *GetScheduler() { return scheduler_.get(); }

  /**
   * Register an existing shared memory segment into the IpcManager
   * Called by worker when encountering an unknown allocator in a FutureShm
   * Derives shm_name from alloc_id: chimaera_{pid}_{index}
   * @param alloc_id Allocator ID (major=pid, minor=index)
   * @return true if successful (or already registered), false on error
   */
  bool RegisterMemory(const ctp::ipc::AllocatorId &alloc_id);

  /**
   * Get the current process's shared memory info for registration
   * @param index Index of the shared memory segment (0 to shm_count_-1)
   * @return ClientShmInfo for the specified segment
   */
  ClientShmInfo GetClientShmInfo(u32 index) const;

  /**
   * Reap shared memory segments from dead processes
   *
   * Iterates over all registered shared memory segments and checks if the
   * owning process (identified by pid = AllocatorId.major) is still alive.
   * For segments belonging to dead processes, destroys the shared memory
   * backend and removes tracking entries.
   *
   * Does not reap:
   * - Segments owned by the current process
   * - The main allocator segment (AllocatorId 1.0)
   *
   * @return Number of shared memory segments reaped
   */
  size_t WreapDeadIpcs();

  /**
   * Reap all shared memory segments
   *
   * Destroys all shared memory backends (except main allocator) and clears
   * all tracking structures. This is typically called during shutdown to
   * clean up all IPC resources.
   *
   * @return Number of shared memory segments reaped
   */
  size_t WreapAllIpcs();

#if CTP_ENABLE_CUDA || CTP_ENABLE_ROCM || CTP_ENABLE_SYCL
  /**
   * Allocate a client-owned device-memory backend and register it with the
   * runtime so the CPU GPU worker can resolve task ShmPtrs popped off
   * gpu2cpu_queue. Producer-only model: clients pre-allocate everything
   * before kernel launch.
   *
   * `kind` selects the memory type:
   *   - kPinnedHost (cudaHostAlloc / sycl::malloc_host): CPU readable directly,
   *     device sees it through PCIe — fastest path, lowest latency.
   *   - kManagedUvm  (cudaMallocManaged / sycl::malloc_shared): page-faulting.
   *   - kDeviceMem   (cudaMalloc / sycl::malloc_device): worker copies POD
   *     bytes via cudaMemcpy on every pop.
   *
   * Returns a fresh AllocatorId on success and writes the host-visible base
   * pointer (or device pointer for kDeviceMem) to *out_base. Returns an empty
   * AllocatorId on failure.
   */
  ctp::ipc::AllocatorId AllocateAndRegisterGpuBackend(
      u32 gpu_id, gpu::IpcManager::MemKind kind, size_t bytes,
      char **out_base);

  /** Unregister and free a previously allocated GPU backend. */
  void FreeGpuBackend(u32 gpu_id, const ctp::ipc::AllocatorId &alloc_id);
#endif

  /**
   * Clear all memfd symlinks from the per-user chimaera directory.
   *
   * Called during RuntimeInit to clean up leftover memfd symlinks
   * from previous runs or crashed processes. Since the directory is
   * per-user, all entries are cleaned up.
   *
   * @return Number of memfd symlinks successfully removed
   */
  size_t ClearUserIpcs();

 private:
  // Pool query resolution helpers
  std::vector<PoolQuery> ResolveLocalQuery(const PoolQuery &query,
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
   * Initialize memory segments for server
   * @return true if successful, false otherwise
   */
  bool ServerInitShm();

  /**
   * Initialize memory segments for client
   * @return true if successful, false otherwise
   */
  bool ClientInitShm();

  /**
   * Initialize priority queues for server
   * @return true if successful, false otherwise
   */
  bool ServerInitQueues();

  // GPU orchestrator-era init / lifecycle helpers (ServerInitGpuQueues,
  // InitCpu2GpuSendPools, PauseGpuOrchestrator, ResumeGpuOrchestrator,
  // SetGpuOrchestratorBlocks, PrintGpuOrchestratorProfile, etc.) are gone
  // with the producer-only redesign. Per-device init is invoked via
  // ChiServerBootstrap{Hip,Sycl}Gpu from IpcManager::ServerInit.

 private:
  /**
   * Initialize priority queues for client
   * @return true if successful, false otherwise
   */
  bool ClientInitQueues();

  /**
   * Wait for local server to become available via lightbeam transport
   * Sends a ClientConnectTask and waits for response with timeout
   * Uses CHI_WAIT_SERVER environment variable for timeout (default 30s)
   * @return true if server responded, false on timeout
   */
  bool WaitForLocalServer();

  /**
   * Try to start main server on given hostname
   * Helper method for host identification
   * Uses ZMQ port from ConfigManager and sets main_transport_
   * @param hostname Hostname to bind to
   * @return true if server started successfully, false otherwise
   */
  bool TryStartMainServer(const std::string &hostname);

  bool is_initialized_ = false;

  // CLIO_FORCE_NET: read once at ServerInit (and only on the server
  // path — clients route via the server's IsTaskLocal decision so a
  // client-side override would be a no-op).  When set, every task
  // whose PoolQuery isn't explicitly Local() is treated as remote so
  // the ZMQ serialize / send / deserialize loop gets exercised even
  // on a single-node bench.  See IsTaskLocal for the predicate.
  bool force_net_ = false;

  // Shared memory backend for main segment (task data, FutureShm)
  ctp::ipc::PosixShmMmap main_backend_;

  // Allocator ID for main segment
  ctp::ipc::AllocatorId main_allocator_id_;

  // Main allocator pointer for runtime shared memory (task data, FutureShm)
  // CPU: MultiProcessAllocator — shared across runtime + client processes
  // GPU: unused (gpu_alloc_table_ provides per-thread BuddyAllocator)
  CLIO_TASK_ALLOC_T *main_allocator_ = nullptr;

  // Shared memory backend for queue segment (TaskQueue ring buffers)
  ctp::ipc::PosixShmMmap queue_backend_;

  // Allocator ID for queue segment
  ctp::ipc::AllocatorId queue_allocator_id_;

  // Queue allocator pointer — ArenaAllocator for all TaskQueue structures
  CLIO_QUEUE_ALLOC_T *queue_allocator_ = nullptr;

  // Number of workers for which queues are allocated
  u32 num_workers_ = 0;

  // Number of scheduling queues for task distribution
  u32 num_sched_queues_ = 0;

  // PID of the runtime process (for tgkill)
  pid_t runtime_pid_ = 0;

  // Monotonic counter, set from epoch nanos at init
  std::atomic<u64> server_generation_{0};

  // The worker task queues (multi-lane queue)
  ctp::ipc::FullPtr<TaskQueue> worker_queues_;
  // SHM offset of worker_queues_ within queue_allocator_ (server sets it;
  // client receives it via ClientConnectTask and stores here for
  // ClientInitQueues)
  u64 worker_queues_off_ = 0;

  // Network queue for send operations (one lane, two priorities)
  ctp::ipc::FullPtr<NetQueue> net_queue_;

  // Net workers' lane pointers for signaling on EnqueueNetTask. With the
  // recv/send split, send-side priorities wake net_send_lane_ and
  // client-response priorities wake net_recv_lane_. net_lane_ remains as
  // an alias for legacy callers / logging.
  TaskLane *net_lane_ = nullptr;
  TaskLane *net_send_lane_ = nullptr;
  TaskLane *net_recv_lane_ = nullptr;

  // GPU per-device queues now live on gpu::IpcManager::per_gpu_devices_.

  // Local ZeroMQ transport (server mode, using lightbeam)
  ctp::lbm::TransportPtr local_transport_;

  // Main ZeroMQ transport (server mode) for distributed communication
  ctp::lbm::TransportPtr main_transport_;

  // IPC transport mode (TCP default, configurable via CHI_IPC_MODE)
  IpcMode ipc_mode_ = IpcMode::kTcp;

  // SHM lightbeam transport (for SendShm / RecvShm)
  ctp::lbm::TransportPtr shm_send_transport_;
  ctp::lbm::TransportPtr shm_recv_transport_;

  // Client-side: DEALER transport for sending tasks and receiving responses
  ctp::lbm::TransportPtr zmq_transport_;
  std::mutex zmq_client_send_mutex_;

  // Server-side: ROUTER transport for receiving client tasks and sending
  // responses
  ctp::lbm::TransportPtr client_tcp_transport_;
  // Server-side: Socket transport for IPC client communication
  ctp::lbm::TransportPtr client_ipc_transport_;

  // EventManager for the client recv thread.  Must outlive zmq_transport_ so
  // ~SocketTransport() can safely call em_->RemoveEvent() without use-after-free.
  ctp::lbm::EventManager zmq_client_em_;

  // Client recv thread (receives completed task outputs via lightbeam)
  std::thread zmq_recv_thread_;
  std::atomic<bool> zmq_recv_running_{false};

  // Background heartbeat thread for server liveness detection
  std::thread heartbeat_thread_;
  std::atomic<bool> heartbeat_running_{false};
  std::atomic<bool> server_alive_{true};

  // Pending futures (client-side, keyed by net_key)
  std::unordered_map<size_t, FutureShm *> pending_zmq_futures_;
  std::mutex pending_futures_mutex_;

  // Pending response archives (client-side, keyed by net_key)
  // Archives stay alive after Recv() deserialization so that zmq zero-copy
  // buffers (stored in recv[].desc) remain valid until Future::Destroy().
  std::unordered_map<size_t, std::unique_ptr<LoadTaskArchive>>
      pending_response_archives_;

  // Dead node tracking for failure detection
  std::vector<DeadNodeEntry> dead_nodes_;

  // Self-fencing flag for partition detection (SWIM protocol)
  bool self_fenced_ = false;

  // Hostfile management
  std::unordered_map<u64, Host> hostfile_map_;  // Map node_id -> Host
  mutable std::vector<Host>
      hosts_cache_;  // Cached vector of hosts for GetAllHosts
  mutable bool hosts_cache_valid_ = false;  // Flag to track cache validity
  Host this_host_;                          // Identified host for this node

  // Client-side server waiting configuration (from environment variables)
  // Semantics: 0 = fail immediately, -1 = wait forever, >0 = timeout in seconds
  float wait_server_timeout_ =
      30.0f;  // CHI_WAIT_SERVER: timeout in seconds (default 30)
  u32 poll_server_interval_ =
      1;  // CHI_POLL_SERVER: poll interval in seconds (default 1)

  // Client-side retry configuration
  // Semantics: 0 = fail immediately, -1 = wait forever, >0 = timeout in seconds
  u64 client_generation_ = 0;  // Cached server generation at connect time
  float client_retry_timeout_ =
      60.0f;                        // CHI_CLIENT_RETRY_TIMEOUT (default 60s)
  int client_try_new_servers_ = 0;  // CHI_CLIENT_TRY_NEW_SERVERS (default 0)
  std::atomic<bool> reconnecting_{false};  // Guards against recursive reconnect

  // Persistent ZeroMQ transport connection pool
  // Key format: "ip_address:port"
  std::unordered_map<std::string, ctp::lbm::TransportPtr> client_pool_;
  mutable std::mutex client_pool_mutex_;  // Mutex for thread-safe pool access

  // Scheduler for task routing
  std::unique_ptr<Scheduler> scheduler_;

  //============================================================================
  // Per-Process Shared Memory Management
  //============================================================================

  /** Counter for shared memory segments created by this process (starts at 0)
   */
  std::atomic<u32> shm_count_{0};

  /**
   * Map of AllocatorId -> Allocator for all registered shared memory segments
   * Key is the allocator ID (major.minor), value is the allocator pointer
   * Used by ToFullPtr to find the correct allocator for a ShmPtr
   * Protected by allocator_map_lock_ for thread-safe access
   */
  std::unordered_map<u64, ctp::ipc::MultiProcessAllocator *> alloc_map_;

  /**
   * Map of AllocatorId -> {data_ptr, capacity} for GPU backend memory
   * Used by ToFullPtr to resolve ShmPtrs allocated by GPU kernels.
   * GPU backends use pinned host memory, so data_ptr is CPU-accessible.
   */
 public:
  /**
   * Wait for local server to stop by polling with ClientConnectTask.
   * Sends repeated ClientConnectTask probes with a short timeout.
   * Returns true once the runtime stops responding (connection fails/times
   * out).
   * @param timeout_sec Maximum time to wait for the runtime to stop
   * @return true if runtime stopped, false if still running after timeout
   */
  bool WaitForLocalRuntimeStop(u32 timeout_sec = 30);

  /**
   * RwLock for protecting allocator_map_ access
   * Reader lock: for normal ToFullPtr lookups and allocation attempts
   * Writer lock: for IpcManager cleanup and memory increase operations
   */
  chi::CoRwLock allocator_map_lock_;


#if CTP_ENABLE_CUDA || CTP_ENABLE_ROCM || CTP_ENABLE_SYCL
  /** GPU IPC manager: owns the per-device gpu2cpu_queues and the
   *  AllocatorId → ClientBackend registry. Producer-only design: GPU
   *  kernels just push, CPU worker pops and resolves. */
  std::unique_ptr<gpu::IpcManager> gpu_ipc_;
#else
  /** Layout placeholder — keeps struct size/offsets identical whether
   *  any GPU backend is enabled or not, preventing ODR violations when test
   *  binaries link chimaera_cxx without GPU support. */
  void *gpu_ipc_placeholder_ = nullptr;
#endif

 private:
#if CTP_IS_HOST
  /**
   * Create a new per-process shared memory segment and register it with the
   * runtime Client-only: sends Admin::RegisterMemory and waits for the server
   * to attach
   * @param size Size in bytes to allocate (32MB will be added for metadata)
   * @return true if successful, false otherwise
   */
  bool IncreaseClientShm(size_t size);

  /**
   * Vector of allocators owned by this process
   * Used for allocation attempts before calling IncreaseClientShm
   */
  std::vector<ctp::ipc::MultiProcessAllocator *> alloc_vector_;

  /**
   * Vector of backends owned by this process
   * Stored to ensure backends outlive allocators
   */
  std::vector<std::unique_ptr<ctp::ipc::PosixShmMmap>> client_backends_;

  /**
   * Most recently accessed allocator for fast allocation path
   * Checked first in AllocateBuffer before iterating alloc_vector_
   */
  ctp::ipc::MultiProcessAllocator *last_alloc_ = nullptr;

  /** Mutex for thread-safe access to shared memory structures */
  mutable std::mutex shm_mutex_;
#endif

  /** Metadata overhead to add to each shared memory segment: 32MB */
  static constexpr size_t kShmMetadataOverhead = 32ULL * 1024 * 1024;

  /** Multiplier for shared memory allocation to ensure space for metadata */
  static constexpr float kShmAllocationMultiplier = 2.5f;
};

}  // namespace clio::run

// Global pointer variable declaration for IPC manager singleton
CLIO_RUN_DEFINE_GLOBAL_PTR_VAR_H(chi::IpcManager, g_ipc_manager);

#define CLIO_IPC CTP_GET_GLOBAL_PTR_VAR(::chi::IpcManager, g_ipc_manager)
#define CLIO_CPU_IPC CLIO_IPC

// Backward-compat aliases (clio_run rebrand). External code that still
// uses the legacy CHI_* spelling keeps working unchanged. These are
// text-substitution aliases — they resolve to whichever CLIO_* def is
// active in the current pass (host / GPU host pass / GPU device pass /
// SYCL device pass), surviving any later #undef/#define cycle on
// CLIO_IPC / CLIO_CPU_IPC further down in this file.
#define CHI_IPC      CLIO_IPC
#define CHI_CPU_IPC  CLIO_CPU_IPC

// Include local_task_archives after CLIO_IPC is defined, since on GPU
// CLIO_PRIV_ALLOC expands to chi::GetPrivAllocGpu() (defined below)
#include "clio_runtime/local_task_archives.h"

// ================================================================
// GPU translation unit support: override CLIO_IPC for device code
// ================================================================
#if CTP_IS_GPU_COMPILER

namespace clio::run {
namespace gpu {
CTP_CROSS_FUN inline IpcManager *GetGpuIpcManager() {
#if CTP_IS_GPU
  return IpcManager::GetBlockIpcManager();
#else
  return nullptr;
#endif
}
}  // namespace gpu
}  // namespace clio::run

// CLIO_IPC needs different expansions in nvcc/hipcc's two passes:
//   - Device pass (CTP_IS_GPU=1): GetBlockIpcManager() — the per-block
//     `__shared__` singleton initialized by CHIMAERA_GPU_INIT.
//   - Host pass (CTP_IS_GPU=0): the global host pointer accessor —
//     same as the non-GPU-compiler default. Host-only client code
//     (bdev_client::AsyncCreate, etc.) gets compiled in this pass too
//     when the test .cc lives in an nvcc TU; it must reach the real
//     host IpcManager, not nullptr. Mirrors the SYCL two-form override.
#undef CLIO_IPC
#if CTP_IS_GPU
#define CLIO_IPC (::chi::gpu::GetGpuIpcManager())
#else
#define CLIO_IPC CTP_GET_GLOBAL_PTR_VAR(::chi::IpcManager, g_ipc_manager)
#endif
#undef CLIO_CPU_IPC
#define CLIO_CPU_IPC CTP_GET_GLOBAL_PTR_VAR(::chi::IpcManager, g_ipc_manager)

namespace clio::run {
// Producer-only model: kernels do not allocate. The legacy
// GetPrivAllocGpu / GetSharedAllocGpu helpers (which returned a private
// BuddyAllocator and the shared RoundRobinAllocator) returned null /
// gpu_alloc_ respectively; both are gone with the GPU runtime concept.
// Any device-pass code that still calls these gets a nullptr stub so it
// links cleanly while we excise the call sites.
#if !CTP_IS_HOST
CTP_GPU_FUN inline ctp::ipc::PrivateBuddyAllocator *GetPrivAllocGpu() {
  return nullptr;
}
CTP_GPU_FUN inline ctp::ipc::RoundRobinAllocator *GetSharedAllocGpu() {
  return nullptr;
}
#endif
}  // namespace clio::run

#endif  // CTP_IS_GPU_COMPILER

#if CTP_IS_SYCL_COMPILER

// SYCL has no analogue of CUDA's __shared__-backed GetBlockIpcManager(),
// and DPC++ rejects function-local static variables in device code. The
// CUDA path lets CLIO_IPC auto-resolve via a static method; the SYCL path
// instead binds a kernel-scope local variable named `g_ipc_manager_ptr`
// in the CHIMAERA_GPU_*_INIT macros (see gpu_ipc_manager.h), and CLIO_IPC
// is a macro that resolves to that local via plain C++ name lookup.
//
// Consequence: CLIO_IPC works inside the kernel body and inside any
// device function called from the kernel where `g_ipc_manager_ptr` is
// in lexical scope (typically because the function takes it as a
// parameter or is inlined into the kernel). Free functions that take
// no parameters and reach for CLIO_IPC will not compile under SYCL —
// pass the IpcManager pointer through explicitly. The chimaera runtime
// follows this convention: chimod methods are called from the worker's
// kernel body, where g_ipc_manager_ptr is in scope.
//
// CLIO_CPU_IPC remains the host-side global pointer accessor for code
// that runs on the CPU even in a SYCL build.
namespace clio::run {
// SYCL stubs for the per-warp allocator getters that types.h's
// CLIO_PRIV_ALLOC macro expands to under any device pass. Code that wants
// a private allocator under SYCL should reach CLIO_IPC->gpu_alloc_
// directly; these stubs preserve build-time compatibility for paths that
// happened to call them.
inline ctp::ipc::PrivateBuddyAllocator *GetPrivAllocGpu() { return nullptr; }
inline ctp::ipc::RoundRobinAllocator *GetSharedAllocGpu() { return nullptr; }

}  // namespace clio::run

// Global-namespace fallback for `g_ipc_manager_ptr`. Code inside the
// kernel scope shadows this with a local established by
// CHIMAERA_GPU_*_INIT and gets the real IpcManager pointer; host-only
// methods that get parsed (but never emitted) in the SYCL device pass —
// e.g. bdev_client's AsyncMonitor — find this nullptr fallback so they
// parse cleanly. They are never reachable from a kernel, so DPC++ does
// not emit device code for them and the nullptr is never dereferenced
// on device.
//
// Declared at global namespace scope (rather than in `chi`) so the
// unqualified name `g_ipc_manager_ptr` resolves to it from any
// surrounding namespace via the standard C++ unqualified-lookup walk.
//
// Declared `inline` so multiple TUs sharing this header don't generate
// conflicting definitions.
inline ::chi::gpu::IpcManager *g_ipc_manager_ptr = nullptr;

// CLIO_IPC under SYCL needs different expansions in the two compilation
// passes that DPC++ runs over a SYCL TU:
//
//   - Device pass (CTP_IS_SYCL_DEVICE=1): resolve to the kernel-scope
//     local `g_ipc_manager_ptr` established by CHIMAERA_GPU_*_INIT, picked
//     up via unqualified C++ name lookup from the enclosing function.
//   - Host pass: keep using the global pointer accessor — host-only
//     functions (e.g. bdev_client::AsyncMonitor) get compiled in this
//     pass too even when they're never called from device, and they
//     legitimately want the host singleton.
//
// The two-form expansion lets CTP_CROSS_FUN-tagged code (compiled in
// both passes) get the right pointer in each.
#undef CLIO_IPC
#if CTP_IS_SYCL_DEVICE
#define CLIO_IPC (g_ipc_manager_ptr)
#else
#define CLIO_IPC CTP_GET_GLOBAL_PTR_VAR(::chi::IpcManager, g_ipc_manager)
#endif
#undef CLIO_CPU_IPC
#define CLIO_CPU_IPC CTP_GET_GLOBAL_PTR_VAR(::chi::IpcManager, g_ipc_manager)

#endif  // CTP_IS_SYCL_COMPILER

// ================================================================
// Future method implementations (unified for CPU and GPU TUs)
// ================================================================
namespace clio::run {

// ~Future() - frees resources if consumed (via Wait/await_resume)
template <typename TaskT, typename AllocT>
CTP_CROSS_FUN Future<TaskT, AllocT>::~Future() {
  if (consumed_) {
    // Clean up zero-copy response archive (TCP/IPC only, never used on GPU)
    if (!future_shm_.IsNull()) {
#if CTP_IS_HOST
      ctp::ipc::FullPtr<FutureShm> fs = CLIO_CPU_IPC->ToFullPtr(future_shm_);
      if (!fs.IsNull() && (fs->origin_ == FutureShm::FUTURE_CLIENT_TCP ||
                           fs->origin_ == FutureShm::FUTURE_CLIENT_IPC)) {
        CLIO_CPU_IPC->CleanupResponseArchive(fs->client_task_vaddr_);
      }
      // Free FutureShm (host only)
      ctp::ipc::ShmPtr<char> buffer_shm = future_shm_.template Cast<char>();
      CLIO_CPU_IPC->FreeBuffer(buffer_shm);
      future_shm_.SetNull();
#endif
    }
    // Auto-free the task (only when consumed to avoid double-free
    // from runtime-internal Future copies in event queues / RunContext)
    DelTask();
  }
}

// GetFutureShm() - converts internal ShmPtr to FullPtr
template <typename TaskT, typename AllocT>
CTP_CROSS_FUN ctp::ipc::FullPtr<typename Future<TaskT, AllocT>::FutureT>
Future<TaskT, AllocT>::GetFutureShm() const {
  if (future_shm_.IsNull()) {
    return ctp::ipc::FullPtr<FutureT>();
  }
#if CTP_IS_GPU
  return CLIO_IPC->ToFullPtr(future_shm_);
#else
  return CLIO_CPU_IPC->ToFullPtr(future_shm_);
#endif
}

// ----------------------------------------------------------------
// IsComplete variants
// ----------------------------------------------------------------

template <typename TaskT, typename AllocT>
CTP_CROSS_FUN bool Future<TaskT, AllocT>::IsComplete() const {
  if (future_shm_.IsNull()) {
    return false;
  }
#if CTP_IS_GPU
  return IsCompleteGpu2Gpu();
#else
#if CTP_ENABLE_CUDA || CTP_ENABLE_ROCM
  if (future_shm_.alloc_id_ == FutureShm::GetCpu2GpuAllocId()) {
    return IsCompleteCpu2Gpu();
  }
#endif
  auto future_shm = GetFutureShm();
  if (future_shm.IsNull()) {
    return false;
  }
  bool is_gpu_future =
      future_shm->flags_.Any(FutureShm::FUTURE_COPY_FROM_CLIENT) &&
      (future_shm->client_task_vaddr_ == 0);
  if (is_gpu_future) {
    return IsCompleteGpu2Cpu();
  }
  return IsCompleteCpu2Cpu();
#endif
}

template <typename TaskT, typename AllocT>
CTP_HOST_FUN bool Future<TaskT, AllocT>::IsCompleteCpu2Cpu() const {
  auto future_shm = GetFutureShm();
  if (future_shm.IsNull()) {
    return false;
  }
  return future_shm->flags_.Any(FutureShm::FUTURE_COMPLETE);
}

template <typename TaskT, typename AllocT>
CTP_HOST_FUN bool Future<TaskT, AllocT>::IsCompleteGpu2Cpu() const {
  auto future_shm = GetFutureShm();
  if (future_shm.IsNull()) {
    return false;
  }
  return future_shm->flags_.AnySystem(FutureShm::FUTURE_COMPLETE);
}

template <typename TaskT, typename AllocT>
CTP_HOST_FUN bool Future<TaskT, AllocT>::IsCompleteCpu2Gpu() const {
#if CTP_ENABLE_CUDA || CTP_ENABLE_ROCM
  // ShmPtr offset points to pinned-host gpu::FutureShm
  void *host_fshm = reinterpret_cast<void *>(future_shm_.off_.load());
  u32 flags_val = 0;
  size_t flags_offset = offsetof(gpu::FutureShm, flags_);
  ctp::GpuApi::Memcpy(
      &flags_val,
      reinterpret_cast<u32 *>(
          static_cast<char *>(host_fshm) + flags_offset),
      sizeof(u32));
  return (flags_val & gpu::FutureShm::FUTURE_COMPLETE) != 0;
#else
  return false;
#endif
}

#if CTP_IS_GPU_COMPILER
template <typename TaskT, typename AllocT>
CTP_GPU_FUN bool Future<TaskT, AllocT>::IsCompleteGpu2Gpu() const {
  auto future_shm = GetFutureShm();
  if (future_shm.IsNull()) {
    return false;
  }
  return future_shm->flags_.AnyDevice(FutureShm::FUTURE_COMPLETE);
}
#endif

// ----------------------------------------------------------------
// Wait dispatcher
// ----------------------------------------------------------------

template <typename TaskT, typename AllocT>
CTP_CROSS_FUN bool Future<TaskT, AllocT>::Wait(float max_sec,
                                                 bool reuse_task) {
#if CTP_IS_GPU
  return WaitGpu2Gpu(max_sec, reuse_task);
#else
  if (task_ptr_.IsNull() || future_shm_.IsNull()) {
    return true;
  }
  // Fire-and-forget: return immediately without waiting.
  if (task_ptr_->task_flags_.Any(TASK_FIRE_AND_FORGET)) {
    task_ptr_.SetNull();
    future_shm_.SetNull();
    return true;
  }

  // Non-blocking poll (max_sec == 0): check FUTURE_COMPLETE without
  // entering the recv path. If the task is still in flight, return
  // false immediately so the caller can do other work. If it IS
  // complete, fall through; the underlying recv will see the flag and
  // deserialize without blocking.
  if (max_sec == 0.0f) {
    ctp::ipc::FullPtr<FutureShm> future_full_poll =
        CLIO_CPU_IPC->ToFullPtr(future_shm_);
    if (future_full_poll.IsNull()) {
      return false;
    }
    if (!future_full_poll.ptr_->flags_.Any(FutureShm::FUTURE_COMPLETE)) {
      return false;
    }
    // Complete -> fall through to normal wait path (cheap; flag is set).
  }

  bool is_runtime = CLIO_RUNTIME_MANAGER->IsRuntime();

#if CTP_ENABLE_CUDA || CTP_ENABLE_ROCM
  // CPU→GPU POD path: sentinel allocator ID marks device pointers.
  if (is_runtime &&
      future_shm_.alloc_id_ == FutureShm::GetCpu2GpuAllocId()) {
    return WaitCpu2Gpu(max_sec, reuse_task);
  }
#endif

  // Resolve FutureShm for non-GPU paths
  ctp::ipc::FullPtr<FutureShm> future_full = CLIO_CPU_IPC->ToFullPtr(future_shm_);
  if (future_full.IsNull()) {
    HLOG(kError, "Future::Wait: ToFullPtr returned null");
    return false;
  }

  if (is_runtime) {
    return WaitCpu2Cpu(max_sec, reuse_task);
  }

  // Client path: detect GPU-originated futures
  bool is_gpu_future =
      future_full->flags_.Any(FutureShm::FUTURE_COPY_FROM_CLIENT) &&
      (future_full->client_task_vaddr_ == 0);
  if (is_gpu_future) {
    return WaitGpu2Cpu(max_sec, reuse_task);
  }
  return WaitCpu2Cpu(max_sec, reuse_task);
#endif
}

// ----------------------------------------------------------------
// GPU Wait paths (CTP_GPU_FUN)
// ----------------------------------------------------------------

#if CTP_IS_GPU_COMPILER
template <typename TaskT, typename AllocT>
CTP_GPU_FUN bool Future<TaskT, AllocT>::WaitGpu2Gpu(float max_sec,
                                                      bool reuse_task) {
  // chi::Future should not be used for GPU-to-GPU paths.
  // Use gpu::Future::WaitGpu2Gpu instead.
  (void)max_sec; (void)reuse_task;
  return true;
}
#endif

// ----------------------------------------------------------------
// Host Wait paths (CTP_HOST_FUN)
// ----------------------------------------------------------------

template <typename TaskT, typename AllocT>
CTP_HOST_FUN bool Future<TaskT, AllocT>::WaitCpu2Gpu(float max_sec,
                                                       bool reuse_task) {
  // CPU->GPU transport was deleted with the GPU runtime concept; only the
  // signature is kept so existing dispatch tables compile. Producer-only
  // model never lands here at runtime.
  (void)max_sec; (void)reuse_task;
  return true;
}

template <typename TaskT, typename AllocT>
CTP_HOST_FUN bool Future<TaskT, AllocT>::WaitCpu2Cpu(float max_sec,
                                                       bool reuse_task) {
  bool is_runtime = CLIO_RUNTIME_MANAGER->IsRuntime();
  if (is_runtime) {
    // Runtime self-send: poll FUTURE_COMPLETE in SHM
    ctp::ipc::FullPtr<FutureShm> future_full =
        CLIO_CPU_IPC->ToFullPtr(future_shm_);
    if (future_full.IsNull()) return false;
    bool ok = IpcCpu2Self::ClientRecv(*this, max_sec, future_full);
    if (!ok) return false;
  } else {
    // Client: SHM or ZMQ recv path
    if (!CLIO_CPU_IPC->Recv(*this, max_sec)) {
      task_ptr_->SetReturnCode(static_cast<u32>(-1));
      return false;
    }
  }
  if (reuse_task) task_ptr_.SetNull();
  Destroy(true);
  return true;
}

template <typename TaskT, typename AllocT>
CTP_HOST_FUN bool Future<TaskT, AllocT>::WaitGpu2Cpu(float max_sec,
                                                       bool reuse_task) {
#if CTP_ENABLE_CUDA || CTP_ENABLE_ROCM
  // Host-side polling path (test harness): polls chi::FutureShm with
  // system-scope atomics. The GPU kernel uses IpcGpu2Cpu::ClientRecv
  // (device-side) which polls gpu::FutureShm instead.
  ctp::ipc::FullPtr<FutureShm> future_full =
      CLIO_CPU_IPC->ToFullPtr(future_shm_);
  if (future_full.IsNull()) {
    HLOG(kError, "Future::WaitGpu2Cpu: ToFullPtr returned null");
    return false;
  }
  ctp::abitfield32_t &flags = future_full->flags_;
  auto start = std::chrono::steady_clock::now();
  while (!flags.AnySystem(FutureShm::FUTURE_COMPLETE)) {
    CTP_THREAD_MODEL->Yield();
    if (max_sec > 0) {
      float elapsed = std::chrono::duration<float>(
                          std::chrono::steady_clock::now() - start)
                          .count();
      if (elapsed >= max_sec) {
        task_ptr_->SetReturnCode(static_cast<u32>(-3));
        return false;
      }
    }
  }

  // Deserialize output from ring buffer if present
  if (future_full->output_.total_written_.load() > 0) {
    ctp::lbm::LbmContext ctx;
    ctx.copy_space = future_full->copy_space;
    ctx.shm_info_ = &future_full->output_;
    chi::priv::vector<char> load_buf(CLIO_PRIV_ALLOC);
    load_buf.reserve(256);
    DefaultLoadArchive load_ar(load_buf);
    load_ar.SetMsgType(LocalMsgType::kSerializeOut);
    ctp::lbm::ShmTransport::Recv(load_ar, ctx);
    task_ptr_->SerializeOut(load_ar);
  }

  if (reuse_task) task_ptr_.SetNull();
  Destroy(true);
  return true;
#else
  (void)max_sec; (void)reuse_task;
  return true;
#endif
}

// ----------------------------------------------------------------
// Shared helpers (CTP_CROSS_FUN)
// ----------------------------------------------------------------

template <typename TaskT, typename AllocT>
CTP_CROSS_FUN void Future<TaskT, AllocT>::Destroy(bool post_wait) {
  if (post_wait && !task_ptr_.IsNull()) {
    task_ptr_->PostWait();
  }
  consumed_ = true;
}

template <typename TaskT, typename AllocT>
CTP_CROSS_FUN void Future<TaskT, AllocT>::DelTask() {
#if CTP_IS_GPU
  // Producer-only model: kernels do not own/free tasks. The host owns
  // the registered device-memory backend; just clear our handle.
  task_ptr_.SetNull();
#else
  if (!task_ptr_.IsNull()) {
    CLIO_CPU_IPC->DelTask(task_ptr_);
    task_ptr_.SetNull();
  }
#endif
}

// ----------------------------------------------------------------
// WaitPoll / WaitRecv (GPU-only, stubs on host)
// ----------------------------------------------------------------

template <typename TaskT, typename AllocT>
CTP_CROSS_FUN void Future<TaskT, AllocT>::WaitPoll(float max_sec,
                                                     bool reuse_task) {
  (void)max_sec; (void)reuse_task;
#if CTP_IS_GPU
  if (threadIdx.x != 0) return;
  auto fshm_full = GetFutureShm();
  if (fshm_full.IsNull()) return;
  auto *fshm = fshm_full.ptr_;

  // Spin-wait on FUTURE_COMPLETE (device-scope atomics)
  while (!fshm->flags_.AnyDevice(FutureShm::FUTURE_COMPLETE)) {
    CTP_THREAD_MODEL->Yield();
  }
  ctp::ipc::threadfence();
#endif
}

template <typename TaskT, typename AllocT>
CTP_CROSS_FUN void Future<TaskT, AllocT>::WaitRecv(float max_sec,
                                                     bool reuse_task) {
  (void)max_sec; (void)reuse_task;
#if CTP_IS_GPU
  if (threadIdx.x != 0) return;
  auto fshm_full = GetFutureShm();
  if (fshm_full.IsNull()) return;
  auto *fshm = fshm_full.ptr_;

  // Read output if any was written
  size_t output_written = fshm->output_.total_written_.load_device();
  if (output_written > 0) {
    ctp::ipc::threadfence();

    // Inline deserialization: create local buffer + archive
    ctp::ipc::FullPtr<char> fp;
    fp.ptr_ = fshm->copy_space;
    fp.shm_.alloc_id_.SetNull();
    fp.shm_.off_ = reinterpret_cast<size_t>(fp.ptr_);
    ctp::priv::wrap_vector buffer;
    buffer.set(fp, output_written);
    buffer.resize(output_written);
    GpuLoadTaskArchive load_ar(buffer);
    load_ar.SetMsgType(LocalMsgType::kSerializeOut);
    task_ptr_.ptr_->SerializeOut(load_ar);
  }

  // Cleanup
  Destroy(true);
  if (reuse_task) {
    task_ptr_.SetNull();
  }
#endif
}

// gpu::Future is fully defined inline in chimaera/gpu/future.h after the
// producer-only redesign — no out-of-line method implementations needed.
// gpu::Future::Wait() is defined in chimaera/ipc/ipc_gpu2cpu_impl.h
// alongside IpcGpu2Cpu::ClientSend.

}  // namespace clio::run

// Template implementations for transport classes (need full IpcManager definition)
#include "clio_runtime/ipc/ipc_cpu2cpu_impl.h"
#include "clio_runtime/ipc/ipc_cpu2cpu_zmq_impl.h"

#endif  // CHIMAERA_INCLUDE_CHIMAERA_MANAGERS_IPC_MANAGER_H_