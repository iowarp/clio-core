#ifndef CHIMAERA_INCLUDE_CHIMAERA_MANAGERS_IPC_MANAGER_H_
#define CHIMAERA_INCLUDE_CHIMAERA_MANAGERS_IPC_MANAGER_H_

#include "chimaera/chimaera_manager.h"
#include "chimaera/task.h"
#include "chimaera/task_queue.h"
#include "chimaera/types.h"
#include "chimaera/worker.h"
#include "chimaera/future.h"
#include "chimaera/local_task_archives.h"
#include "hermes_shm/memory/backend/posix_shm_mmap.h"
#include <atomic>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace chi {

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
  u64 node_id;              // 64-bit hash of the hostname for node identification
};

/**
 * Host structure for hostfile management
 * Contains IP address and corresponding 64-bit node ID
 */
struct Host {
  std::string ip_address; // IP address as string (IPv4 or IPv6)
  u64 node_id;            // 64-bit representation of IP address

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
    TaskT* ptr = new TaskT(std::forward<Args>(args)...);
    // Create a FullPtr with null allocator ID and zero offset (private memory)
    // Use explicit initialization to avoid template constructor overload issues
    hipc::FullPtr<TaskT> result(ptr);
    return result;
  }

  /**
   * Delete a task from private memory (using standard delete)
   * @param task_ptr FullPtr to task to delete
   */
  template <typename TaskT> void DelTask(hipc::FullPtr<TaskT> task_ptr) {
    if (task_ptr.IsNull())
      return;

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
   * Send task asynchronously (serializes into Future)
   * Creates a Future wrapper, serializes task inputs, and enqueues to worker
   * @param task_ptr Task to send
   * @return Future<TaskT> for polling completion and retrieving results
   */
  template <typename TaskT> Future<TaskT> Send(hipc::FullPtr<TaskT> task_ptr) {
    // 1. Get main allocator for FutureShm allocation
    auto *alloc = GetMainAlloc();

    // 2. Create Future with allocator and task_ptr
    Future<TaskT> future(alloc, task_ptr);

    // 3. Serialize task using LocalSaveTaskArchive with kSerializeIn mode
    LocalSaveTaskArchive archive(LocalMsgType::kSerializeIn);
    archive << (*task_ptr.ptr_);

    // 4. Get serialized data and copy to FutureShm's hipc::vector
    const std::vector<char>& serialized = archive.GetData();
    auto& future_shm = future.GetFutureShm();
    future_shm->serialized_task_.resize(serialized.size());
    memcpy(future_shm->serialized_task_.data(), serialized.data(), serialized.size());

    // 5. Map task to lane using configured policy
    u32 num_lanes = shared_header_->num_sched_queues;
    if (num_lanes == 0)
      return future; // Avoid division by zero

    // 6. Enqueue the FutureShm's ShmPtr to the worker queue
    LaneId lane_id = MapTaskToLane(num_lanes);
    auto& lane_ref = worker_queues_->GetLane(lane_id, 0);
    lane_ref.Push(future_shm.shm_);

    // 7. Return the Future
    return future;
  }

  /**
   * Receive task results (deserializes from completed Future)
   * Polls for completion and deserializes task outputs into task
   * @param future Future containing completed task
   */
  template <typename TaskT> void Recv(Future<TaskT>& future) {
    // 1. Poll for completion by checking is_complete atomic in FutureShm
    auto& future_shm = future.GetFutureShm();
    while (future_shm->is_complete_.load() == 0) {
      // Busy wait or could add a short sleep
      std::this_thread::yield();
    }

    // 2. Get the serialized task data from FutureShm
    hipc::vector<char, CHI_MAIN_ALLOC_T>& serialized = future_shm->serialized_task_;

    // 3. Convert hipc::vector to std::vector for LocalLoadTaskArchive
    std::vector<char> buffer(serialized.begin(), serialized.end());

    // 4. Create LocalLoadTaskArchive with kSerializeOut mode
    LocalLoadTaskArchive archive(buffer);
    archive.SetMsgType(LocalMsgType::kSerializeOut);

    // 5. Deserialize task outputs into the Future's task pointer
    TaskT* task_ptr = future.get();
    archive >> (*task_ptr);
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
   * Get main allocator (alias for GetMainAlloc)
   * @return Pointer to main allocator or nullptr if not available
   */
  CHI_MAIN_ALLOC_T *GetMainAlloc() { return main_allocator_; }

  /**
   * Get client data allocator
   * @return Pointer to client data allocator or nullptr if not available
   */
  CHI_CDATA_ALLOC_T *GetDataAlloc() { return client_data_allocator_; }

  /**
   * Get runtime data allocator (same as client data allocator)
   * @return Pointer to runtime data allocator or nullptr if not available
   */
  CHI_RDATA_ALLOC_T *GetRdataAlloc() { return runtime_data_allocator_; }

  /**
   * Get number of workers from shared memory header
   * @return Number of workers, 0 if not initialized
   */
  u32 GetWorkerCount();

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
  void SetLaneMapPolicy(LaneMapPolicy policy);

  /**
   * Get current lane mapping policy
   * @return Current lane mapping policy
   */
  LaneMapPolicy GetLaneMapPolicy() const;

  /**
   * Get the main ZeroMQ server for network communication
   * @return Pointer to main server or nullptr if not initialized
   */
  hshm::lbm::Server *GetMainServer() const;

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
   * Matches the ShmPtr's allocator ID against main, data, and rdata allocators
   * @param shm_ptr The ShmPtr to convert
   * @return FullPtr with matching allocator and pointer, or null FullPtr if no match
   */
  template <typename T>
  hipc::FullPtr<T> ToFullPtr(const hipc::ShmPtr<T> &shm_ptr) {
    // Check main allocator
    if (main_allocator_ && shm_ptr.alloc_id_ == main_allocator_->GetId()) {
      return hipc::FullPtr<T>(main_allocator_, shm_ptr);
    }

    // Check client data allocator
    if (client_data_allocator_ && shm_ptr.alloc_id_ == client_data_allocator_->GetId()) {
      return hipc::FullPtr<T>(client_data_allocator_, shm_ptr);
    }

    // Check runtime data allocator
    if (runtime_data_allocator_ && shm_ptr.alloc_id_ == runtime_data_allocator_->GetId()) {
      return hipc::FullPtr<T>(runtime_data_allocator_, shm_ptr);
    }

    // No matching allocator found
    return hipc::FullPtr<T>();
  }

  /**
   * Convert raw pointer to FullPtr by checking allocators
   * Uses ContainsPtr() on each allocator to find the matching one
   * @param ptr The raw pointer to convert
   * @return FullPtr with matching allocator and pointer, or null FullPtr if no match
   */
  template <typename T>
  hipc::FullPtr<T> ToFullPtr(T *ptr) {
    // Check main allocator
    if (main_allocator_ && main_allocator_->ContainsPtr(ptr)) {
      return hipc::FullPtr<T>(main_allocator_, ptr);
    }

    // Check client data allocator
    if (client_data_allocator_ && client_data_allocator_->ContainsPtr(ptr)) {
      return hipc::FullPtr<T>(client_data_allocator_, ptr);
    }

    // Check runtime data allocator
    if (runtime_data_allocator_ && runtime_data_allocator_->ContainsPtr(ptr)) {
      return hipc::FullPtr<T>(runtime_data_allocator_, ptr);
    }

    // No matching allocator found
    return hipc::FullPtr<T>();
  }

private:
  /**
   * Map task to lane ID using the configured policy
   * Dispatches to the appropriate policy-specific function
   * @param num_lanes Number of available lanes
   * @return Lane ID to use
   */
  LaneId MapTaskToLane(u32 num_lanes);

  /**
   * Map task to lane by PID+TID hash
   * @param num_lanes Number of available lanes
   * @return Lane ID to use
   */
  LaneId MapByPidTid(u32 num_lanes);

  /**
   * Map task to lane using round-robin
   * @param num_lanes Number of available lanes
   * @return Lane ID to use
   */
  LaneId MapRoundRobin(u32 num_lanes);

  /**
   * Map task to lane randomly
   * @param num_lanes Number of available lanes
   * @return Lane ID to use
   */
  LaneId MapRandom(u32 num_lanes);

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
   * Test connection to local server
   * Creates lightbeam client and attempts connection to local server
   * Does not print any logging output
   * @return true if connection successful, false otherwise
   */
  bool TestLocalServer();

  /**
   * Wait for local server to become available
   * Polls TestLocalServer until server is available or timeout expires
   * Uses CHI_WAIT_SERVER and CHI_POLL_SERVER environment variables
   * Inherits logging from TestLocalServer attempts
   * @return true if server becomes available, false on timeout
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

  // Shared memory backends for the three segments
  hipc::PosixShmMmap main_backend_;
  hipc::PosixShmMmap client_data_backend_;
  hipc::PosixShmMmap runtime_data_backend_;

  // Allocator IDs for each segment
  hipc::AllocatorId main_allocator_id_;
  hipc::AllocatorId client_data_allocator_id_;
  hipc::AllocatorId runtime_data_allocator_id_;

  // Cached allocator pointers for performance
  CHI_MAIN_ALLOC_T *main_allocator_ = nullptr;
  CHI_CDATA_ALLOC_T *client_data_allocator_ = nullptr;
  CHI_RDATA_ALLOC_T *runtime_data_allocator_ = nullptr;

  // Pointer to shared header containing the task queue pointer
  IpcSharedHeader *shared_header_ = nullptr;

  // The worker task queues (multi-lane queue)
  hipc::FullPtr<TaskQueue> worker_queues_;

  // Local ZeroMQ server (using lightbeam)
  std::unique_ptr<hshm::lbm::Server> local_server_;

  // Main ZeroMQ server for distributed communication
  std::unique_ptr<hshm::lbm::Server> main_server_;

  // Hostfile management
  std::unordered_map<u64, Host> hostfile_map_; // Map node_id -> Host
  mutable std::vector<Host>
      hosts_cache_; // Cached vector of hosts for GetAllHosts
  mutable bool hosts_cache_valid_ = false; // Flag to track cache validity
  Host this_host_;                         // Identified host for this node

  // Lane mapping policy
  LaneMapPolicy lane_map_policy_ = LaneMapPolicy::kRoundRobin;
  std::atomic<u32> round_robin_counter_{0}; // Counter for round-robin policy

  // Client-side server waiting configuration (from environment variables)
  u32 wait_server_timeout_ =
      30; // CHI_WAIT_SERVER: timeout in seconds (default 30)
  u32 poll_server_interval_ =
      1; // CHI_POLL_SERVER: poll interval in seconds (default 1)
};

} // namespace chi

// Global pointer variable declaration for IPC manager singleton
HSHM_DEFINE_GLOBAL_PTR_VAR_H(chi::IpcManager, g_ipc_manager);

// Macro for accessing the IPC manager singleton using global pointer variable
#define CHI_IPC HSHM_GET_GLOBAL_PTR_VAR(::chi::IpcManager, g_ipc_manager)

// Define Future::Wait() after IpcManager is fully defined
// This avoids circular dependency issues between future.h and ipc_manager.h
namespace chi {

template<typename TaskT, typename AllocT>
void Future<TaskT, AllocT>::Wait() {
  if (!task_ptr_.IsNull()) {
    // Call IpcManager::Recv() to poll for completion and deserialize results
    CHI_IPC->Recv(*this);
  }
}

}  // namespace chi

#endif // CHIMAERA_INCLUDE_CHIMAERA_MANAGERS_IPC_MANAGER_H_