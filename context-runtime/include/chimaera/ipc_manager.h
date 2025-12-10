#ifndef CHIMAERA_INCLUDE_CHIMAERA_MANAGERS_IPC_MANAGER_H_
#define CHIMAERA_INCLUDE_CHIMAERA_MANAGERS_IPC_MANAGER_H_

#include "chimaera/chimaera_manager.h"
#include "chimaera/task.h"
#include "chimaera/task_queue.h"
#include "chimaera/types.h"
#include "chimaera/worker.h"
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
  TaskQueue external_queue; // External/Process TaskQueue in shared memory
  chi::ipc::vector<WorkQueue>
      worker_queues; // Vector of worker active queues
  u32 num_workers;   // Number of workers for which queues are allocated
  u64 node_id;       // 64-bit hash of the hostname for node identification
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
   * Create a new task in shared memory (always uses main segment)
   * @param args Constructor arguments for the task
   * @return FullPtr to allocated task
   */
  template <typename TaskT, typename... Args>
  FullPtr<TaskT> NewTask(Args &&...args) {
    if (!main_allocator_) {
      return FullPtr<TaskT>();
    }

    return main_allocator_->template NewObj<TaskT>(main_allocator_,
                                                   std::forward<Args>(args)...);
  }

  /**
   * Delete a task from shared memory (always uses main segment)
   * @param task_ptr FullPtr to task to delete
   */
  template <typename TaskT> void DelTask(FullPtr<TaskT> task_ptr) {
    if (task_ptr.IsNull() || !main_allocator_)
      return;

    main_allocator_->template DelObj<TaskT>(task_ptr);
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
   * Enqueue task to process queue (priority 0 - normal tasks)
   * Uses the configured lane mapping policy to select the target lane
   * @param task_ptr Task to enqueue
   */
  template <typename TaskT> void Enqueue(FullPtr<TaskT> &task_ptr) {
    if (!external_queue_.IsNull() && external_queue_.ptr_) {
      // Create ShmPtr from the task FullPtr
      hipc::ShmPtr<Task> typed_ptr(task_ptr.shm_);

      u32 num_lanes = external_queue_->GetNumLanes();
      if (num_lanes == 0)
        return; // Avoid division by zero

      // Map task to lane using configured policy
      LaneId lane_id = MapTaskToLane(num_lanes);

      // Get lane as FullPtr and use TaskQueue's EmplaceTask method
      // Priority 0 for normal task submission
      auto &lane_ref = external_queue_->GetLane(lane_id, 0);
      lane_ref.Push(task_ptr.template Cast<Task>().shm_);
    }
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
   * Get main allocator for creating worker queues
   * @return Pointer to main allocator or nullptr if not available
   */
  CHI_MAIN_ALLOC_T *GetMainAllocator() { return main_allocator_; }

  /**
   * Get main allocator (alias for GetMainAllocator)
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
   * Initialize worker queues in shared memory
   * @param num_workers Number of worker queues to create
   * @return true if initialization successful, false otherwise
   */
  bool InitializeWorkerQueues(u32 num_workers);

  /**
   * Get worker active queue by worker ID
   * @param worker_id Worker identifier (0-based)
   * @return FullPtr to worker's active queue or null if invalid
   */
  hipc::FullPtr<WorkQueue> GetWorkerQueue(u32 worker_id);

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
  hipc::FullPtr<T> ToFullPtr(const hipc::ShmPtr<T> &shm_ptr);

  /**
   * Convert raw pointer to FullPtr by checking allocators
   * Uses ContainsPtr() on each allocator to find the matching one
   * @param ptr The raw pointer to convert
   * @return FullPtr with matching allocator and pointer, or null FullPtr if no match
   */
  template <typename T>
  hipc::FullPtr<T> ToFullPtr(T *ptr);

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

  // The actual external TaskQueue instance
  hipc::FullPtr<TaskQueue> external_queue_;

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

#endif // CHIMAERA_INCLUDE_CHIMAERA_MANAGERS_IPC_MANAGER_H_