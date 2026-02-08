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

#include "chimaera/chimaera_manager.h"
#include "chimaera/corwlock.h"
#include "chimaera/local_task_archives.h"
#include "chimaera/local_transfer.h"
#include "chimaera/scheduler/scheduler.h"
#include "chimaera/task.h"
#include "chimaera/task_queue.h"
#include "chimaera/types.h"
#include "chimaera/worker.h"
#include "hermes_shm/memory/backend/posix_shm_mmap.h"

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
#include "hermes_shm/memory/backend/gpu_malloc.h"
#include "hermes_shm/memory/backend/gpu_shm_mmap.h"
#include "hermes_shm/memory/allocator/buddy_allocator.h"
#endif

namespace chi {

/**
 * Network queue priority levels for send operations
 */
enum class NetQueuePriority : u32 {
  kSendIn = 0,  ///< Priority 0: SendIn operations (sending task inputs)
  kSendOut = 1  ///< Priority 1: SendOut operations (sending task outputs)
};

/**
 * Network queue for storing Future<SendTask> objects
 * One lane with two priorities (SendIn and SendOut)
 */
using NetQueue = hipc::multi_mpsc_ring_buffer<Future<Task>, CHI_MAIN_ALLOC_T>;

/**
 * Typedef for worker queue type to simplify usage
 */
using WorkQueue = chi::ipc::mpsc_ring_buffer<hipc::ShmPtr<TaskLane>>;

/**
 * Custom header structure for shared memory allocator
 * Contains shared data structures
 */
struct IpcSharedHeader {
  TaskQueue worker_queues;  // Multi-lane worker task queue in shared memory
  u32 num_workers;          // Number of workers for which queues are allocated
  u32 num_sched_queues;     // Number of scheduling queues for task distribution
  u64 node_id;        // 64-bit hash of the hostname for node identification
  pid_t runtime_pid;  // PID of the runtime process (for tgkill)
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
  hipc::AllocatorId alloc_id;  // Allocator ID for this segment

  ClientShmInfo() : owner_pid(0), shm_index(0), size(0) {}

  ClientShmInfo(const std::string &name, pid_t pid, u32 idx, size_t sz,
                const hipc::AllocatorId &id)
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

/**
 * Host structure for hostfile management
 * Contains IP address and corresponding 64-bit node ID
 */
struct Host {
  std::string ip_address;  // IP address as string (IPv4 or IPv6)
  u64 node_id;             // 64-bit representation of IP address

  /**
   * Default constructor
   */
  Host() : node_id(0) {}

  /**
   * Constructor with IP address and node ID (required)
   * Node IDs are assigned based on linear offset in hostfile
   * @param ip IP address string
   * @param id Node ID (typically offset in hostfile)
   */
  Host(const std::string &ip, u64 id) : ip_address(ip), node_id(id) {}

  /**
   * Stream output operator for Host
   * @param os Output stream
   * @param host Host object to print
   * @return Reference to output stream
   */
  friend std::ostream &operator<<(std::ostream &os, const Host &host) {
    os << "Host(ip=" << host.ip_address << ", node_id=" << host.node_id << ")";
    return os;
  }
};

/**
 * IPC Manager singleton for inter-process communication
 *
 * Manages ZeroMQ server using lightbeam from HSHM, three memory segments,
 * and priority queues for task processing.
 * Uses HSHM global cross pointer variable singleton pattern.
 */
class IpcManager {
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
   * Server finalize - cleanup all IPC resources
   */
  void ServerFinalize();

  /**
   * Initialize GPU client components
   * Sets up GPU-specific fields without calling constructor
   * @param backend GPU memory backend
   * @param allocator Pre-initialized GPU allocator
   */
  HSHM_CROSS_FUN
  void ClientGpuInit(const hipc::MemoryBackend &backend,
                     hipc::ArenaAllocator<false> *allocator) {
    gpu_backend_ = backend;
    gpu_backend_initialized_ = true;
    gpu_thread_allocator_ = allocator;
  }

  /**
   * Create a new task in private memory
   * Host: uses standard new
   * GPU: uses AllocateBuffer from shared memory
   * @param args Constructor arguments for the task
   * @return FullPtr wrapping the task with null allocator
   */
  template <typename TaskT, typename... Args>
  HSHM_CROSS_FUN hipc::FullPtr<TaskT> NewTask(Args &&...args) {
#if HSHM_IS_HOST
    // Host path: use standard new
    TaskT *ptr = new TaskT(std::forward<Args>(args)...);
    hipc::FullPtr<TaskT> result(ptr);
    return result;
#else
    // GPU path: allocate from shared memory buffer
    hipc::FullPtr<char> buffer = AllocateBuffer(sizeof(TaskT));
    TaskT *ptr = new (buffer.ptr_) TaskT(std::forward<Args>(args)...);
    hipc::FullPtr<TaskT> result(ptr);
    return result;
#endif
  }

  /**
   * Delete a task from private memory
   * Host: uses standard delete
   * GPU: uses FreeBuffer
   * @param task_ptr FullPtr to task to delete
   */
  template <typename TaskT>
  HSHM_CROSS_FUN void DelTask(hipc::FullPtr<TaskT> task_ptr) {
    if (task_ptr.IsNull()) return;
#if HSHM_IS_HOST
    // Host path: use standard delete
    delete task_ptr.ptr_;
#else
    // GPU path: call destructor and free buffer
    task_ptr.ptr_->~TaskT();
    FreeBuffer(hipc::FullPtr<char>(reinterpret_cast<char*>(task_ptr.ptr_)));
#endif
  }

  /**
   * Allocate buffer in appropriate memory segment
   * Client uses cdata segment, runtime uses rdata segment
   * Yields until buffer is allocated successfully
   * @param size Size in bytes to allocate
   * @return FullPtr<char> to allocated memory
   */
  HSHM_CROSS_FUN FullPtr<char> AllocateBuffer(size_t size);

  /**
   * Free buffer from appropriate memory segment
   * Client uses cdata segment, runtime uses rdata segment
   * @param buffer_ptr FullPtr to buffer to free
   */
  void FreeBuffer(FullPtr<char> buffer_ptr);

  /**
   * Free buffer from appropriate memory segment (hipc::ShmPtr<> overload)
   * Converts hipc::ShmPtr<> to FullPtr<char> and calls the main FreeBuffer
   * @param buffer_ptr hipc::ShmPtr<> to buffer to free
   */
  void FreeBuffer(hipc::ShmPtr<char> buffer_ptr) {
    if (buffer_ptr.IsNull()) {
      return;
    }
    // Convert hipc::ShmPtr<> to FullPtr<char> and call main FreeBuffer
    hipc::FullPtr<char> full_ptr(ToFullPtr<char>(buffer_ptr));
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
  hipc::FullPtr<T> NewObj(Args &&...args) {
    // Allocate buffer for the object
    hipc::FullPtr<char> buffer = AllocateBuffer(sizeof(T));
    if (buffer.IsNull()) {
      return hipc::FullPtr<T>();
    }

    // Construct object using placement new
    T *obj = new (buffer.ptr_) T(std::forward<Args>(args)...);

    // Return FullPtr<T> by reinterpreting the buffer's ptr and shm
    hipc::FullPtr<T> result;
    result.ptr_ = obj;
    result.shm_ = buffer.shm_.template Cast<T>();
    return result;
  }

  /**
   * Create Future by copying/serializing task (GPU-compatible)
   * Always serializes the task into FutureShm's copy_space
   * Used by clients and GPU kernels
   *
   * @tparam TaskT Task type (must derive from Task)
   * @param task_ptr Task to serialize into Future
   * @return Future<TaskT> with serialized task data
   */
  template <typename TaskT>
  HSHM_CROSS_FUN Future<TaskT> MakeCopyFuture(hipc::FullPtr<TaskT> task_ptr) {
    // Check task_ptr validity
    if (task_ptr.IsNull()) {
      return Future<TaskT>();
    }

    // Serialize the task
    LocalSaveTaskArchive archive(LocalMsgType::kSerializeIn);
    archive << (*task_ptr.ptr_);

    // Get serialized data
    const std::vector<char> &serialized = archive.GetData();
    size_t serialized_size = serialized.size();

    // Get recommended copy space size from task, but use actual size if larger
    size_t recommended_size = task_ptr->GetCopySpaceSize();
    size_t copy_space_size = std::max(recommended_size, serialized_size);

    // Allocate and construct FutureShm with appropriately sized copy_space
    size_t alloc_size = sizeof(FutureShm) + copy_space_size;
    hipc::FullPtr<char> buffer = AllocateBuffer(alloc_size);
    if (buffer.IsNull()) {
      return Future<TaskT>();
    }

    // Construct FutureShm in-place using placement new
    FutureShm *future_shm_ptr = new (buffer.ptr_) FutureShm();

    // Initialize FutureShm fields
    future_shm_ptr->pool_id_ = task_ptr->pool_id_;
    future_shm_ptr->method_id_ = task_ptr->method_;
    future_shm_ptr->capacity_.store(copy_space_size);

    // Copy serialized data to copy_space
    memcpy(future_shm_ptr->copy_space, serialized.data(), serialized_size);
    future_shm_ptr->input_size_.store(serialized_size,
                                      std::memory_order_release);

    // Memory fence: Ensure copy_space and input_size_ writes are visible before flag
    std::atomic_thread_fence(std::memory_order_release);

    // Set FUTURE_COPY_FROM_CLIENT flag - worker will deserialize from copy_space
    future_shm_ptr->flags_.SetBits(FutureShm::FUTURE_COPY_FROM_CLIENT);

    // Create ShmPtr to FutureShm
    hipc::ShmPtr<FutureShm> future_shm_shmptr =
        buffer.shm_.template Cast<FutureShm>();

    // Return Future preserving the original task_ptr
    Future<TaskT> future(future_shm_shmptr, task_ptr);
    return future;
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
  Future<TaskT> MakePointerFuture(hipc::FullPtr<TaskT> task_ptr) {
    // Check task_ptr validity
    if (task_ptr.IsNull()) {
      return Future<TaskT>();
    }

    // Allocate and construct FutureShm (no copy_space for runtime path)
    hipc::FullPtr<FutureShm> future_shm = NewObj<FutureShm>();
    if (future_shm.IsNull()) {
      return Future<TaskT>();
    }

    // Initialize FutureShm fields
    future_shm.ptr_->pool_id_ = task_ptr->pool_id_;
    future_shm.ptr_->method_id_ = task_ptr->method_;
    future_shm.ptr_->capacity_.store(0);  // No copy_space in runtime path

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
  Future<TaskT> MakeFuture(hipc::FullPtr<TaskT> task_ptr) {
    // Check task_ptr validity
    if (task_ptr.IsNull()) {
      HLOG(kError, "MakeFuture: called with null task_ptr");
      return Future<TaskT>();
    }

    bool is_runtime = CHI_CHIMAERA_MANAGER->IsRuntime();
    Worker *worker = CHI_CUR_WORKER;

    // Runtime path requires BOTH IsRuntime AND worker to be non-null
    bool use_runtime_path = is_runtime && worker != nullptr;

    if (!use_runtime_path) {
      // CLIENT PATH: Use MakeCopyFuture to serialize the task
      return MakeCopyFuture(task_ptr);
    } else {
      // RUNTIME PATH: Use MakePointerFuture to wrap pointer without serialization
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
  HSHM_CROSS_FUN Future<TaskT> Send(hipc::FullPtr<TaskT> task_ptr, bool awake_event = true) {
    // 1. Create Future using MakeFuture (handles both client and runtime paths)
    // In CLIENT mode: MakeFuture serializes task and sets
    // FUTURE_COPY_FROM_CLIENT flag In RUNTIME mode: MakeFuture wraps task
    // pointer directly without serialization
    Future<TaskT> future = MakeFuture(task_ptr);

    // 2. Get current worker (needed for runtime parent task tracking)
    // On GPU: worker is always null, so use client path
#if HSHM_IS_HOST
    Worker *worker = CHI_CUR_WORKER;
    bool is_runtime = CHI_CHIMAERA_MANAGER->IsRuntime();
#else
    Worker *worker = nullptr;
    bool is_runtime = false;
#endif

    // Runtime path requires BOTH IsRuntime AND worker to be non-null
    bool use_runtime_path = is_runtime && worker != nullptr;

    // 3. Set parent task RunContext from current worker (runtime only)
    if (use_runtime_path) {
      RunContext *run_ctx = worker->GetCurrentRunContext();
      if (run_ctx != nullptr) {
        future.SetParentTask(run_ctx);
      }
    }

    // 4. Map task to lane using scheduler
    LaneId lane_id;
    Future<Task> task_future_for_sched = future.template Cast<Task>();
    if (use_runtime_path) {
      lane_id = scheduler_->RuntimeMapTask(worker, task_future_for_sched);
    } else {
      lane_id = scheduler_->ClientMapTask(this, task_future_for_sched);
    }

    // 5. Enqueue the Future object to the worker queue
    auto &lane_ref = worker_queues_->GetLane(lane_id, 0);
    Future<Task> task_future_for_queue = future.template Cast<Task>();
    lane_ref.Push(task_future_for_queue);

    // 6. Awaken worker for this lane
    AwakenWorker(&lane_ref);

    // 7. Return the same Future (no separate user_future/queue_future)
    return future;
  }

  /**
   * Receive task results (deserializes from completed Future)
   * Called after Future::Wait() has confirmed task completion
   *
   * Two execution paths:
   * - Path 1 (fits): Data fits in serialized_task_capacity, deserialize
   * directly
   * - Path 2 (streaming): Data larger than capacity, assemble from stream
   * - Runtime mode (IsRuntime): No-op (task already has correct outputs)
   *
   * @param future Future containing completed task
   */
  template <typename TaskT>
  void Recv(Future<TaskT> &future) {
    bool is_runtime = CHI_CHIMAERA_MANAGER->IsRuntime();
    Worker *worker = CHI_CUR_WORKER;

    // Runtime path requires BOTH IsRuntime AND worker to be non-null
    bool use_runtime_path = is_runtime && worker != nullptr;

    if (!use_runtime_path) {
      // CLIENT PATH: Deserialize task outputs from FutureShm using
      // LocalTransfer
      auto future_shm = future.GetFutureShm();
      TaskT *task_ptr = future.get();

      // Wait for first data to be available (signaled by FUTURE_NEW_DATA or
      // FUTURE_COMPLETE) This ensures output_size_ is valid before we read it
      hshm::abitfield32_t &flags = future_shm->flags_;
      while (!flags.Any(FutureShm::FUTURE_NEW_DATA) &&
             !flags.Any(FutureShm::FUTURE_COMPLETE)) {
        HSHM_THREAD_MODEL->Yield();
      }

      // Memory fence: Ensure we see worker's writes to output_size_
      std::atomic_thread_fence(std::memory_order_acquire);

      // Get output size from FutureShm (now valid)
      size_t output_size = future_shm->output_size_.load();

      // Use LocalTransfer to receive all data
      LocalTransfer receiver(future_shm, output_size);

      // Receive all data (blocks until complete)
      bool recv_complete = receiver.Recv();
      if (!recv_complete) {
        HLOG(kError, "Recv: LocalTransfer failed - received {}/{} bytes",
             receiver.GetBytesTransferred(), output_size);
      }

      // Wait for FUTURE_COMPLETE to ensure all data has been sent
      while (!flags.Any(FutureShm::FUTURE_COMPLETE)) {
        HSHM_THREAD_MODEL->Yield();
      }

      // Create LocalLoadTaskArchive with kSerializeOut mode
      LocalLoadTaskArchive archive(receiver.GetData());
      archive.SetMsgType(LocalMsgType::kSerializeOut);

      // Deserialize task outputs into the Future's task pointer
      archive >> (*task_ptr);
    }
    // RUNTIME PATH: No deserialization needed - task already has correct
    // outputs
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
  hshm::lbm::Server *GetMainServer() const;

  /**
   * Get the heartbeat socket for polling heartbeat requests
   * @return Raw ZMQ REP socket pointer, or nullptr if not initialized
   */
  void *GetHeartbeatSocket() const;

  /**
   * Get this host identified during host identification
   * @return Const reference to this Host struct
   */
  const Host &GetThisHost() const;

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
  HSHM_CROSS_FUN hipc::FullPtr<T> ToFullPtr(const hipc::ShmPtr<T> &shm_ptr) {
#if HSHM_IS_GPU
    // GPU PATH: Simple conversion using the warp allocator
    if (shm_ptr.IsNull()) {
      return hipc::FullPtr<T>();
    }
    // Convert ShmPtr offset to pointer (assumes GPU path uses simple offset scheme)
    return hipc::FullPtr<T>(gpu_thread_allocator_, shm_ptr);
#else
    // HOST PATH: Full allocator lookup implementation
    // Case 1: AllocatorId is null - offset IS the raw memory address
    // This is used for private memory allocations (new/delete)
    if (shm_ptr.alloc_id_ == hipc::AllocatorId::GetNull()) {
      // The offset field contains the raw pointer address
      T *raw_ptr = reinterpret_cast<T *>(shm_ptr.off_.load());
      return hipc::FullPtr<T>(raw_ptr);
    }

    // Case 2: Check main allocator (runtime shared memory)
    if (main_allocator_ && shm_ptr.alloc_id_ == main_allocator_->GetId()) {
      return hipc::FullPtr<T>(main_allocator_, shm_ptr);
    }

    // Case 3: Check per-process shared memory allocators via alloc_map_
    // Acquire reader lock for thread-safe access to allocator_map_
    allocator_map_lock_.ReadLock(0);

    // Convert AllocatorId to lookup key (combine major and minor)
    u64 alloc_key = (static_cast<u64>(shm_ptr.alloc_id_.major_) << 32) |
                    static_cast<u64>(shm_ptr.alloc_id_.minor_);
    auto it = alloc_map_.find(alloc_key);
    hipc::FullPtr<T> result;
    if (it != alloc_map_.end()) {
      result = hipc::FullPtr<T>(it->second, shm_ptr);
    }

    // Release the lock before returning
    allocator_map_lock_.ReadUnlock();

    return result;
#endif
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
  HSHM_CROSS_FUN hipc::FullPtr<T> ToFullPtr(T *ptr) {
#if HSHM_IS_GPU
    // GPU PATH: Wrap raw pointer with warp allocator
    if (ptr == nullptr) {
      return hipc::FullPtr<T>();
    }
    return hipc::FullPtr<T>(gpu_thread_allocator_, ptr);
#else
    // HOST PATH: Full allocator lookup implementation
    if (ptr == nullptr) {
      return hipc::FullPtr<T>();
    }

    // Check main allocator
    if (main_allocator_ && main_allocator_->ContainsPtr(ptr)) {
      return hipc::FullPtr<T>(main_allocator_, ptr);
    }

    // Check per-process shared memory allocators
    // Acquire reader lock for thread-safe access
    allocator_map_lock_.ReadLock(0);

    hipc::FullPtr<T> result;
    for (auto *alloc : alloc_vector_) {
      if (alloc && alloc->ContainsPtr(ptr)) {
        result = hipc::FullPtr<T>(alloc, ptr);
        allocator_map_lock_.ReadUnlock();
        return result;
      }
    }

    // Release the lock before returning
    allocator_map_lock_.ReadUnlock();

    // No matching allocator found - treat as private memory
    // Return FullPtr with the raw pointer (null allocator ID)
    return hipc::FullPtr<T>(ptr);
#endif
  }

  /**
   * Get or create a persistent ZeroMQ client connection from the pool
   * Creates a new connection if one doesn't exist for the given address:port
   * Thread-safe using internal mutex protection
   * @param addr IP address to connect to
   * @param port Port number to connect to
   * @return Pointer to the ZeroMQ client (owned by the pool)
   */
  hshm::lbm::Client *GetOrCreateClient(const std::string &addr, int port);

  /**
   * Clear all cached client connections
   * Should be called during shutdown
   */
  void ClearClientPool();

  /**
   * Enqueue a Future<SendTask> to the network queue
   * @param future Future containing the SendTask to enqueue
   * @param priority Network queue priority (kSendIn or kSendOut)
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
   * Get the network queue for direct access
   * @return Pointer to the network queue or nullptr if not initialized
   */
  NetQueue *GetNetQueue() { return net_queue_.ptr_; }

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
  /**
   * Get number of GPU queues
   * @return Number of GPU queues (one per GPU device)
   */
  size_t GetGpuQueueCount() const { return gpu_queues_.size(); }

  /**
   * Get GPU queue by index
   * @param gpu_id GPU device ID (0-based)
   * @return Pointer to GPU TaskQueue or nullptr if invalid gpu_id
   */
  TaskQueue *GetGpuQueue(size_t gpu_id) {
    if (gpu_id < gpu_queues_.size()) {
      return gpu_queues_[gpu_id].ptr_;
    }
    return nullptr;
  }
#endif

  /**
   * Get the scheduler instance
   * IpcManager is the single owner of the scheduler.
   * WorkOrchestrator and Worker should use this method to get the scheduler.
   * @return Pointer to the scheduler or nullptr if not initialized
   */
  Scheduler *GetScheduler() { return scheduler_.get(); }

  /**
   * Increase memory by creating a new per-process shared memory segment
   * Creates shared memory with name chimaera_{pid}_{shm_count_}
   * Registers the new segment with the runtime via Admin::RegisterMemory
   * @param size Size in bytes to allocate (32MB will be added for metadata)
   * @return true if successful, false otherwise
   */
  bool IncreaseMemory(size_t size);

  /**
   * Register an existing shared memory segment into the IpcManager
   * Called by worker when encountering an unknown allocator in a FutureShm
   * Derives shm_name from alloc_id: chimaera_{pid}_{index}
   * @param alloc_id Allocator ID (major=pid, minor=index)
   * @return true if successful (or already registered), false on error
   */
  bool RegisterMemory(const hipc::AllocatorId &alloc_id);

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

  /**
   * Clear all chimaera_* shared memory segments from /dev/shm
   *
   * Called during RuntimeInit to clean up leftover shared memory segments
   * from previous runs or crashed processes. Attempts to remove all files
   * matching "chimaera_*" pattern in /dev/shm directory.
   *
   * Permission errors are silently ignored to allow multi-user systems where
   * other users may have active Chimaera processes.
   *
   * @return Number of shared memory segments successfully removed
   */
  size_t ClearUserIpcs();

  /**
   * Register GPU accelerator memory backend (GPU kernel use only)
   *
   * Called from GPU kernels to store GPU memory backend reference.
   * Per-thread BuddyAllocators are initialized in CHIMAERA_GPU_INIT macro.
   *
   * @param backend GPU memory backend to register
   * @return true on success, false on failure
   */
  HSHM_CROSS_FUN
  bool RegisterAcceleratorMemory(const hipc::MemoryBackend &backend);

 private:
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

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
  /**
   * Initialize GPU queues for server (one ring buffer per GPU)
   * Uses pinned host memory with NUMA awareness
   * @return true if successful, false otherwise
   */
  bool ServerInitGpuQueues();
#endif

  /**
   * Initialize priority queues for client
   * @return true if successful, false otherwise
   */
  bool ClientInitQueues();

  /**
   * Wait for local server to become available using heartbeat mechanism
   * Sends ZMQ_REQ heartbeat and waits for ZMQ_REP response with timeout
   * Uses CHI_WAIT_SERVER environment variable for timeout (default 30s)
   * @return true if heartbeat response received, false on timeout
   */
  bool WaitForLocalServer();

  /**
   * Try to start main server on given hostname
   * Helper method for host identification
   * Uses ZMQ port from ConfigManager and sets main_server_
   * @param hostname Hostname to bind to
   * @return true if server started successfully, false otherwise
   */
  bool TryStartMainServer(const std::string &hostname);

  bool is_initialized_ = false;

  // Shared memory backend for main segment (contains IpcSharedHeader,
  // TaskQueue)
  hipc::PosixShmMmap main_backend_;

  // Allocator ID for main segment
  hipc::AllocatorId main_allocator_id_;

  // Main allocator pointer for runtime shared memory (queues, FutureShm)
  CHI_MAIN_ALLOC_T *main_allocator_ = nullptr;

  // Pointer to shared header containing the task queue pointer
  IpcSharedHeader *shared_header_ = nullptr;

  // The worker task queues (multi-lane queue)
  hipc::FullPtr<TaskQueue> worker_queues_;

  // Network queue for send operations (one lane, two priorities)
  hipc::FullPtr<NetQueue> net_queue_;

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
  // GPU memory backends (one per GPU device, using pinned host memory)
  std::vector<std::unique_ptr<hipc::GpuShmMmap>> gpu_backends_;

  // GPU task queues (one ring buffer per GPU device)
  std::vector<hipc::FullPtr<TaskQueue>> gpu_queues_;
#endif

  // Local ZeroMQ server (using lightbeam)
  std::unique_ptr<hshm::lbm::Server> local_server_;

  // Main ZeroMQ server for distributed communication
  std::unique_ptr<hshm::lbm::Server> main_server_;

  // Heartbeat server for client connection verification (ZMQ_REP)
  void *heartbeat_ctx_;     ///< ZMQ context for heartbeat server
  void *heartbeat_socket_;  ///< ZMQ REP socket for heartbeat server

  // Hostfile management
  std::unordered_map<u64, Host> hostfile_map_;  // Map node_id -> Host
  mutable std::vector<Host>
      hosts_cache_;  // Cached vector of hosts for GetAllHosts
  mutable bool hosts_cache_valid_ = false;  // Flag to track cache validity
  Host this_host_;                          // Identified host for this node

  // Client-side server waiting configuration (from environment variables)
  u32 wait_server_timeout_ =
      30;  // CHI_WAIT_SERVER: timeout in seconds (default 30)
  u32 poll_server_interval_ =
      1;  // CHI_POLL_SERVER: poll interval in seconds (default 1)

  // Persistent ZeroMQ client connection pool
  // Key format: "ip_address:port"
  std::unordered_map<std::string, std::unique_ptr<hshm::lbm::Client>>
      client_pool_;
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
  std::unordered_map<u64, hipc::MultiProcessAllocator *> alloc_map_;

 public:
  /**
   * RwLock for protecting allocator_map_ access
   * Reader lock: for normal ToFullPtr lookups and allocation attempts
   * Writer lock: for IpcManager cleanup and memory increase operations
   */
  chi::CoRwLock allocator_map_lock_;

  //============================================================================
  // GPU Memory Management (public for CHIMAERA_GPU_INIT macro access)
  //============================================================================

  /** GPU memory backend for device memory (GPU kernels only) */
  hipc::MemoryBackend gpu_backend_;

  /** Pointer to current thread's GPU ArenaAllocator (GPU kernel only) */
  hipc::ArenaAllocator<false> *gpu_thread_allocator_ = nullptr;

  /** Flag indicating if GPU backend is initialized */
  bool gpu_backend_initialized_ = false;

 private:
#if HSHM_IS_HOST
  /**
   * Vector of allocators owned by this process
   * Used for allocation attempts before calling IncreaseMemory
   */
  std::vector<hipc::MultiProcessAllocator *> alloc_vector_;

  /**
   * Vector of backends owned by this process
   * Stored to ensure backends outlive allocators
   */
  std::vector<std::unique_ptr<hipc::PosixShmMmap>> client_backends_;

  /**
   * Most recently accessed allocator for fast allocation path
   * Checked first in AllocateBuffer before iterating alloc_vector_
   */
  hipc::MultiProcessAllocator *last_alloc_ = nullptr;

  /** Mutex for thread-safe access to shared memory structures */
  mutable std::mutex shm_mutex_;
#endif

  /** Metadata overhead to add to each shared memory segment: 32MB */
  static constexpr size_t kShmMetadataOverhead = 32ULL * 1024 * 1024;

  /** Multiplier for shared memory allocation to ensure space for metadata */
  static constexpr float kShmAllocationMultiplier = 1.2f;
};

}  // namespace chi

// Global pointer variable declaration for IPC manager singleton
HSHM_DEFINE_GLOBAL_PTR_VAR_H(chi::IpcManager, g_ipc_manager);

// Macro for accessing the IPC manager singleton using global pointer variable
#define CHI_IPC HSHM_GET_GLOBAL_PTR_VAR(::chi::IpcManager, g_ipc_manager)

// GPU kernel initialization macro
// Creates a shared IPC manager instance in GPU __shared__ memory
// Each thread has its own ArenaAllocator for memory allocation
// Supports 1D, 2D, and 3D thread blocks (max 1024 threads per block)
//
// Usage in GPU kernel:
//   __global__ void my_kernel(const hipc::MemoryBackend* backend) {
//     CHIMAERA_GPU_INIT(*backend);
//     // Now CHI_IPC->AllocateBuffer() works for this thread
//   }
#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
#define CHIMAERA_GPU_INIT(backend)                                                          \
  __shared__ char g_ipc_manager_storage[sizeof(chi::IpcManager)];                           \
  __shared__ chi::IpcManager *g_ipc_manager_ptr;                                            \
  __shared__ hipc::ArenaAllocator<false> *g_arena_alloc;                                    \
  /* Compute linear thread ID for 1D/2D/3D blocks */                                        \
  int thread_id = threadIdx.x +                                                              \
                  threadIdx.y * blockDim.x +                                                 \
                  threadIdx.z * blockDim.x * blockDim.y;                                     \
  if (thread_id == 0) {                                                                      \
    /* Place ArenaAllocator at the beginning of backend's data region */                    \
    g_arena_alloc = reinterpret_cast<hipc::ArenaAllocator<false>*>(backend.data_);         \
    new (g_arena_alloc) hipc::ArenaAllocator<false>();                                      \
    g_arena_alloc->shm_init(backend, backend.data_capacity_);                              \
    /* Point to IpcManager storage without calling constructor */                           \
    /* Do NOT use placement new - IpcManager has STL members that can't init on GPU */      \
    g_ipc_manager_ptr = reinterpret_cast<chi::IpcManager*>(g_ipc_manager_storage);         \
    /* Initialize GPU-specific fields */                                                    \
    g_ipc_manager_ptr->ClientGpuInit(backend, g_arena_alloc);                              \
  }                                                                                           \
  __syncthreads();                                                                           \
  chi::IpcManager &g_ipc_manager = *g_ipc_manager_ptr
#endif

// Define Future methods after IpcManager and CHI_IPC are fully defined
// This avoids circular dependency issues between task.h and ipc_manager.h
namespace chi {

// GPU device implementation of AllocateBuffer
// ToFullPtr implementations are inline in the class above
#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
inline __device__ hipc::FullPtr<char> IpcManager::AllocateBuffer(size_t size) {
  // GPU PATH: Use per-warp ArenaAllocator
  if (gpu_backend_initialized_ && gpu_thread_allocator_ != nullptr) {
    return gpu_thread_allocator_->AllocateObjs<char>(size);
  }
  return hipc::FullPtr<char>::GetNull();
}
#endif

// GetFutureShm() implementation - converts internal ShmPtr to FullPtr
template <typename TaskT, typename AllocT>
hipc::FullPtr<typename Future<TaskT, AllocT>::FutureT>
Future<TaskT, AllocT>::GetFutureShm() const {
  if (future_shm_.IsNull()) {
    return hipc::FullPtr<FutureT>();
  }
  return CHI_IPC->ToFullPtr(future_shm_);
}

template <typename TaskT, typename AllocT>
void Future<TaskT, AllocT>::Wait() {
  // Mark this Future as owner of the task (will be destroyed on Future
  // destruction) Caller should NOT manually call DelTask() after Wait()
  is_owner_ = true;

  if (!task_ptr_.IsNull() && !future_shm_.IsNull()) {
    // Convert ShmPtr to FullPtr to access flags_
    hipc::FullPtr<FutureShm> future_full = CHI_IPC->ToFullPtr(future_shm_);
    if (future_full.IsNull()) {
      HLOG(kError, "Future::Wait: ToFullPtr returned null for future_shm_");
      return;
    }

    // Determine path: client vs runtime
    bool is_runtime = CHI_CHIMAERA_MANAGER->IsRuntime();
    Worker *worker = CHI_CUR_WORKER;
    bool use_runtime_path = is_runtime && worker != nullptr;

    if (use_runtime_path) {
      // RUNTIME PATH: Wait for FUTURE_COMPLETE first (task outputs are direct)
      // No deserialization needed, just wait for completion signal
      hshm::abitfield32_t &flags = future_full->flags_;
      while (!flags.Any(FutureShm::FUTURE_COMPLETE)) {
        HSHM_THREAD_MODEL->Yield();
      }
    } else {
      // CLIENT PATH: Call Recv() first to handle streaming
      // Recv() uses LocalTransfer which will consume chunks as they arrive
      // FUTURE_COMPLETE will be set by worker after all data is sent
      // Don't wait for FUTURE_COMPLETE first - that causes deadlock for streaming
      CHI_IPC->Recv(*this);
    }

    // Call PostWait() callback on the task for post-completion actions
    task_ptr_->PostWait();

    // Don't free future_shm here - let the destructor handle it since is_owner_
    // = true
  }
}

template <typename TaskT, typename AllocT>
void Future<TaskT, AllocT>::Destroy() {
#if HSHM_IS_HOST
  // Host path: use CHI_IPC thread-local
  // Destroy the task using CHI_IPC->DelTask if not null
  if (!task_ptr_.IsNull()) {
    CHI_IPC->DelTask(task_ptr_);
    task_ptr_.SetNull();
  }
  // Also free FutureShm if it wasn't freed in Wait()
  if (!future_shm_.IsNull()) {
    // Cast ShmPtr<FutureShm> to ShmPtr<char> for FreeBuffer
    hipc::ShmPtr<char> buffer_shm = future_shm_.template Cast<char>();
    CHI_IPC->FreeBuffer(buffer_shm);
    future_shm_.SetNull();
  }
#else
  // GPU path: Don't actually free resources - just null out pointers
  // Tasks created on GPU are submitted to CPU queues for processing
  // The CPU side handles the actual cleanup when tasks complete
  // Trying to access g_ipc_manager here would fail because it's only
  // defined within CHIMAERA_GPU_INIT macro scope
  task_ptr_.SetNull();
  future_shm_.SetNull();
#endif
  is_owner_ = false;
}

}  // namespace chi

#endif  // CHIMAERA_INCLUDE_CHIMAERA_MANAGERS_IPC_MANAGER_H_