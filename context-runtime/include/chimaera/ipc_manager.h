#ifndef CHIMAERA_INCLUDE_CHIMAERA_MANAGERS_IPC_MANAGER_H_
#define CHIMAERA_INCLUDE_CHIMAERA_MANAGERS_IPC_MANAGER_H_

#include <atomic>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "chimaera/chimaera_manager.h"
#include "chimaera/local_task_archives.h"
#include "chimaera/scheduler/scheduler.h"
#include "chimaera/task.h"
#include "chimaera/task_queue.h"
#include "chimaera/types.h"
#include "chimaera/worker.h"
#include "hermes_shm/memory/backend/posix_shm_mmap.h"

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
  std::string shm_name;       // Shared memory name (chimaera_{pid}_{count})
  pid_t owner_pid;            // PID of the owning process
  u32 shm_index;              // Index within the owner's shm segments
  size_t size;                // Size of the shared memory segment
  hipc::AllocatorId alloc_id; // Allocator ID for this segment

  ClientShmInfo() : owner_pid(0), shm_index(0), size(0) {}

  ClientShmInfo(const std::string &name, pid_t pid, u32 idx, size_t sz,
                const hipc::AllocatorId &id)
      : shm_name(name), owner_pid(pid), shm_index(idx), size(sz), alloc_id(id) {}

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
   * Create a new task in private memory (using standard new)
   * @param args Constructor arguments for the task
   * @return FullPtr wrapping the task with null allocator
   */
  template <typename TaskT, typename... Args>
  hipc::FullPtr<TaskT> NewTask(Args &&...args) {
    TaskT *ptr = new TaskT(std::forward<Args>(args)...);
    // Create a FullPtr with null allocator ID and zero offset (private memory)
    // Use explicit initialization to avoid template constructor overload issues
    hipc::FullPtr<TaskT> result(ptr);
    return result;
  }

  /**
   * Delete a task from private memory (using standard delete)
   * @param task_ptr FullPtr to task to delete
   */
  template <typename TaskT>
  void DelTask(hipc::FullPtr<TaskT> task_ptr) {
    if (task_ptr.IsNull()) return;

    delete task_ptr.ptr_;
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
   * Free a FutureShm object using the correct allocator
   * Looks up allocator by alloc_id in the shm_ member
   * @tparam FutureT The FutureShm type
   * @param future_shm FullPtr to the FutureShm to free
   */
  template <typename FutureT>
  void FreeFutureShm(hipc::FullPtr<FutureT> &future_shm) {
    if (future_shm.IsNull()) {
      return;
    }

    // Get allocator ID from the shm_ member
    hipc::AllocatorId alloc_id = future_shm.shm_.alloc_id_;

    // Check if allocator ID is null (shouldn't happen for FutureShm)
    if (alloc_id == hipc::AllocatorId::GetNull()) {
      // Null allocator - FutureShm allocated in private memory (unusual)
      return;
    }

    // Check main allocator
    if (main_allocator_ && alloc_id == main_allocator_id_) {
      main_allocator_->DelObj(future_shm);
      return;
    }

    // Check per-process shared memory allocators via alloc_map_
    u64 alloc_key = (static_cast<u64>(alloc_id.major_) << 32) |
                    static_cast<u64>(alloc_id.minor_);
    auto it = alloc_map_.find(alloc_key);
    if (it != alloc_map_.end()) {
      it->second->DelObj(future_shm);
      return;
    }

    HLOG(kWarning, "FreeFutureShm: Could not find allocator for alloc_id ({}.{})",
         alloc_id.major_, alloc_id.minor_);
  }

  /**
   * Send task asynchronously (serializes into Future)
   * Creates a Future wrapper, serializes task inputs, and enqueues to worker
   *
   * Two execution paths:
   * - Client mode (!IsRuntime): Serialize task and copy Future with null task
   * pointer
   * - Runtime mode (IsRuntime): Create Future with task pointer directly (no
   * copy)
   *
   * @param task_ptr Task to send
   * @return Future<TaskT> for polling completion and retrieving results
   */
  template <typename TaskT>
  Future<TaskT> Send(hipc::FullPtr<TaskT> task_ptr, bool awake_event = true) {
    if (!CHI_CHIMAERA_MANAGER->IsRuntime()) {
      // CLIENT PATH: Serialize task and create two Future objects
      // - One for the queue (with null task pointer)
      // - One for the user (with task pointer set)

      // 1. Get allocator from per-process shared memory (created during ClientInit)
      CHI_MAIN_ALLOC_T *alloc = last_alloc_;
      size_t shm_size = 0;
      if (alloc != nullptr) {
        // Get the shm_size from the allocator's backend
        shm_size = alloc->GetBackend().backend_size_;
      }

      // 2. Create Future with allocator and task_ptr (for user)
      Future<TaskT> user_future(alloc, task_ptr);

      // 3. Serialize task using LocalSaveTaskArchive with kSerializeIn mode
      LocalSaveTaskArchive archive(LocalMsgType::kSerializeIn);
      archive << (*task_ptr.ptr_);

      // 4. Get serialized data and copy to FutureShm's hipc::vector
      const std::vector<char> &serialized = archive.GetData();
      auto &future_shm = user_future.GetFutureShm();
      future_shm->serialized_task_.resize(serialized.size());
      memcpy(future_shm->serialized_task_.data(), serialized.data(),
             serialized.size());

      // 5. Set shm_size for lazy registration by worker
      future_shm->shm_size_ = shm_size;

      // 6. Create a separate Future for the queue with null task pointer
      // This Future shares the same FutureShm but has a null task pointer
      hipc::FullPtr<TaskT> null_task_ptr;
      null_task_ptr.SetNull();
      Future<TaskT> queue_future(user_future.GetFutureShm(), null_task_ptr);

      // 7. Map task to lane using scheduler
      // Route Send/Recv tasks to net worker's lane
      LaneId lane_id;
      if (IsNetworkTask(task_ptr)) {
        // Get net worker's lane (last lane in the queue)
        lane_id = shared_header_->num_workers - 1;
      } else {
        // Convert Future<TaskT> to Future<Task> for scheduler
        Future<Task> task_future = queue_future.template Cast<Task>();
        lane_id = scheduler_->ClientMapTask(this, task_future);
      }

      // 8. Enqueue the Future object to the worker queue
      auto &lane_ref = worker_queues_->GetLane(lane_id, 0);
      // Convert Future<TaskT> to Future<Task> for the queue
      Future<Task> task_future = queue_future.template Cast<Task>();
      lane_ref.Push(task_future);

      // 9. Awaken worker for this lane
      AwakenWorker(&lane_ref);

      // 10. Return the Future with task pointer set for the user
      return user_future;
    } else {
      // RUNTIME PATH: Create Future with task pointer directly (no
      // serialization copy)

      // 1. Get allocator from per-process shared memory (created during ServerInit)
      // ServerInit calls IncreaseMemory() which sets last_alloc_ for runtime use
      CHI_MAIN_ALLOC_T *alloc = last_alloc_;
      if (alloc == nullptr) {
        // Fall back to main_allocator_ if last_alloc_ is not set
        alloc = main_allocator_;
        if (alloc == nullptr) {
          HLOG(kError, "Send: No allocator available in runtime path");
          return Future<TaskT>();  // Return null Future
        }
      }

      // 2. Create Future with allocator and task_ptr (task pointer is set)
      Future<TaskT> future(alloc, task_ptr);

      // 2. Get current worker (needed for scheduler and parent task tracking)
      Worker *worker = CHI_CUR_WORKER;

      // 3. Set the parent task RunContext from current worker (if available and
      // awake_event is true)
      if (awake_event && worker != nullptr) {
        RunContext *run_ctx = worker->GetCurrentRunContext();
        if (run_ctx != nullptr) {
          future.SetParentTask(run_ctx);
        }
      }

      // 4. Map task to lane using scheduler
      // Route Send/Recv tasks to net worker's lane
      LaneId lane_id;
      if (IsNetworkTask(task_ptr)) {
        // Get net worker's lane (last lane in the queue)
        lane_id = shared_header_->num_workers - 1;
      } else {
        // Convert Future<TaskT> to Future<Task> for scheduler
        Future<Task> task_future = future.template Cast<Task>();
        lane_id = scheduler_->RuntimeMapTask(worker, task_future);
      }

      // 4. Enqueue the Future object to the worker queue
      auto &lane_ref = worker_queues_->GetLane(lane_id, 0);
      // Convert Future<TaskT> to Future<Task> for the queue
      Future<Task> task_future = future.template Cast<Task>();
      lane_ref.Push(task_future);

      // 5. Awaken worker for this lane
      AwakenWorker(&lane_ref);

      // 6. Return the Future with task pointer
      return future;
    }
  }

  /**
   * Receive task results (deserializes from completed Future)
   * Called after Future::Wait() has confirmed task completion
   *
   * Two execution paths:
   * - Client mode (!IsRuntime): Deserialize task outputs from FutureShm
   * - Runtime mode (IsRuntime): No-op (task already has correct outputs)
   *
   * @param future Future containing completed task
   */
  template <typename TaskT>
  void Recv(Future<TaskT> &future) {
    if (!CHI_CHIMAERA_MANAGER->IsRuntime()) {
      // CLIENT PATH: Deserialize task outputs from FutureShm
      auto &future_shm = future.GetFutureShm();
      TaskT *task_ptr = future.get();

      // Get the serialized task data from FutureShm
      hipc::vector<char, CHI_MAIN_ALLOC_T> &serialized =
          future_shm->serialized_task_;

      // Convert hipc::vector to std::vector for LocalLoadTaskArchive
      std::vector<char> buffer(serialized.begin(), serialized.end());

      // Create LocalLoadTaskArchive with kSerializeOut mode
      LocalLoadTaskArchive archive(buffer);
      archive.SetMsgType(LocalMsgType::kSerializeOut);

      // Deserialize task outputs into the Future's task pointer
      archive >> (*task_ptr);
    }
    // RUNTIME PATH: No deserialization needed - task already has correct
    // outputs
  }

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
   * 1. AllocatorId::GetNull() - offset is the actual memory address (raw pointer)
   * 2. Main allocator - runtime shared memory for queues/futures
   * 3. Per-process shared memory allocators via alloc_map_
   * @param shm_ptr The ShmPtr to convert
   * @return FullPtr with matching allocator and pointer, or null FullPtr if no
   * match
   */
  template <typename T>
  hipc::FullPtr<T> ToFullPtr(const hipc::ShmPtr<T> &shm_ptr) {
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
    // Convert AllocatorId to lookup key (combine major and minor)
    u64 alloc_key = (static_cast<u64>(shm_ptr.alloc_id_.major_) << 32) |
                    static_cast<u64>(shm_ptr.alloc_id_.minor_);
    auto it = alloc_map_.find(alloc_key);
    if (it != alloc_map_.end()) {
      return hipc::FullPtr<T>(it->second, shm_ptr);
    }

    // No matching allocator found
    return hipc::FullPtr<T>();
  }

  /**
   * Convert raw pointer to FullPtr by checking allocators
   * Uses ContainsPtr() on each allocator to find the matching one
   * Checks main allocator first, then per-process allocators
   * If no allocator contains the pointer, returns a FullPtr with null allocator
   * (private memory)
   * @param ptr The raw pointer to convert
   * @return FullPtr with matching allocator and pointer, or FullPtr with null
   * allocator if no match (private memory)
   */
  template <typename T>
  hipc::FullPtr<T> ToFullPtr(T *ptr) {
    if (ptr == nullptr) {
      return hipc::FullPtr<T>();
    }

    // Check main allocator
    if (main_allocator_ && main_allocator_->ContainsPtr(ptr)) {
      return hipc::FullPtr<T>(main_allocator_, ptr);
    }

    // Check per-process shared memory allocators
    for (auto *alloc : alloc_vector_) {
      if (alloc && alloc->ContainsPtr(ptr)) {
        return hipc::FullPtr<T>(alloc, ptr);
      }
    }

    // No matching allocator found - treat as private memory
    // Return FullPtr with the raw pointer (null allocator ID)
    return hipc::FullPtr<T>(ptr);
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
   * @param shm_size Size of the shared memory segment
   * @return true if successful (or already registered), false on error
   */
  bool RegisterMemory(const hipc::AllocatorId &alloc_id, size_t shm_size);

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

 private:
  /**
   * Check if task is a network task (Send or Recv)
   * Network tasks are routed to the dedicated network worker
   * @param task_ptr Task to check
   * @return true if task is a Send or Recv admin task
   */
  template <typename TaskT>
  bool IsNetworkTask(const hipc::FullPtr<TaskT> &task_ptr) const {
    if (task_ptr.IsNull()) {
      return false;
    }
    // Admin kSend = 14, kRecv = 15
    constexpr u32 kAdminSend = 14;
    constexpr u32 kAdminRecv = 15;
    const Task *task = task_ptr.ptr_;
    return task->pool_id_ == kAdminPoolId &&
           (task->method_ == kAdminSend || task->method_ == kAdminRecv);
  }

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

  // Shared memory backend for main segment (contains IpcSharedHeader, TaskQueue)
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

  /** Counter for shared memory segments created by this process (starts at 0) */
  std::atomic<u32> shm_count_{0};

  /**
   * Map of AllocatorId -> Allocator for all registered shared memory segments
   * Key is the allocator ID (major.minor), value is the allocator pointer
   * Used by ToFullPtr to find the correct allocator for a ShmPtr
   */
  std::unordered_map<u64, hipc::MultiProcessAllocator *> alloc_map_;

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

// Define Future::Wait() after IpcManager is fully defined
// This avoids circular dependency issues between future.h and ipc_manager.h
namespace chi {

template <typename TaskT, typename AllocT>
void Future<TaskT, AllocT>::Wait() {
  // Mark this Future as owner of the task (will be destroyed on Future
  // destruction) Caller should NOT manually call DelTask() after Wait()
  is_owner_ = true;

  if (!task_ptr_.IsNull() && !future_shm_.IsNull()) {
    // Wait for completion by polling is_complete atomic
    // Busy-wait with thread yielding - works for both client and runtime
    // contexts Coroutine contexts should use co_await Future instead
    hipc::atomic<u32> &is_complete = future_shm_->is_complete_;
    while (is_complete.load() == 0) {
      HSHM_THREAD_MODEL->Yield();
    }

    // Call IpcManager::Recv() to deserialize results (client path only)
    CHI_IPC->Recv(*this);

    // Call PostWait() callback on the task for post-completion actions
    task_ptr_->PostWait();

    // Free the FutureShm object using the correct allocator
    // FutureShm is allocated from per-process shared memory, look up by alloc_id
    CHI_IPC->FreeFutureShm(future_shm_);
    future_shm_.SetNull();
  }
}

template <typename TaskT, typename AllocT>
void Future<TaskT, AllocT>::Destroy() {
  // Destroy the task using CHI_IPC->DelTask if not null
  if (!task_ptr_.IsNull()) {
    CHI_IPC->DelTask(task_ptr_);
    task_ptr_.SetNull();
  }
  // Also free FutureShm if it wasn't freed in Wait()
  // FutureShm is allocated from per-process shared memory, look up by alloc_id
  if (!future_shm_.IsNull()) {
    CHI_IPC->FreeFutureShm(future_shm_);
    future_shm_.SetNull();
  }
  is_owner_ = false;
}

template <typename TaskT, typename AllocT>
void Future<TaskT, AllocT>::SetAllocator() {
  // Use IpcManager::ToFullPtr to resolve the allocator from the ShmPtr
  // This handles all allocator lookup including null allocator (private memory)
  // Note: RegisterMemory must be called before this if the allocator is unknown
  future_shm_ = CHI_IPC->ToFullPtr(future_shm_.shm_);
}

}  // namespace chi

#endif  // CHIMAERA_INCLUDE_CHIMAERA_MANAGERS_IPC_MANAGER_H_