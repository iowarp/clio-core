/**
 * IPC manager implementation
 */

#include "chimaera/ipc_manager.h"

#include "chimaera/config_manager.h"
#include "chimaera/chimaera_manager.h"
#include "chimaera/admin/admin_client.h"
#include "chimaera/task_queue.h"
#include "chimaera/scheduler/scheduler_factory.h"
#include <arpa/inet.h>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <endian.h>
#include <functional>
#include <iostream>
#include <memory>
#include <netdb.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#include <zmq.h>

// Global pointer variable definition for IPC manager singleton
HSHM_DEFINE_GLOBAL_PTR_VAR_CC(chi::IpcManager, g_ipc_manager);

namespace chi {

// Host struct methods

// IpcManager methods

// Constructor and destructor removed - handled by HSHM singleton pattern

bool IpcManager::ClientInit() {
  HLOG(kDebug, "IpcManager::ClientInit");
  if (is_initialized_) {
    return true;
  }

  // Wait for local server to become available - critical for client
  // functionality TestLocalServer sends heartbeat to verify connectivity
  if (!WaitForLocalServer()) {
    HLOG(kError, "CRITICAL ERROR: Cannot connect to local server.");
    HLOG(kError, "Client initialization failed. Exiting.");
    return false;
  }

  // Initialize memory segments for client
  if (!ClientInitShm()) {
    return false;
  }

  // Initialize priority queues
  if (!ClientInitQueues()) {
    return false;
  }

  // Create per-process shared memory for client allocations
  // Use configured client_data_segment_size from config
  auto *config = CHI_CONFIG_MANAGER;
  size_t initial_size = config && config->IsValid()
      ? config->GetMemorySegmentSize(kClientDataSegment)
      : hshm::Unit<size_t>::Megabytes(256);  // Default 256MB
  if (!IncreaseMemory(initial_size)) {
    HLOG(kError, "IpcManager::ClientInit: Failed to create per-process shared memory");
    return false;
  }

  // Retrieve node ID from shared header and store in this_host_
  if (shared_header_) {
    this_host_.node_id = shared_header_->node_id;
    HLOG(kDebug, "Retrieved node ID from shared memory: 0x{:x}",
          this_host_.node_id);
  } else {
    HLOG(kError, "Warning: Could not access shared header during ClientInit");
    this_host_ = Host(); // Default constructor gives node_id = 0
  }

  // Initialize HSHM TLS key for task counter
  HSHM_THREAD_MODEL->CreateTls<TaskCounter>(chi_task_counter_key_, nullptr);

  // Initialize thread-local task counter for this client thread
  auto *counter = new TaskCounter();
  HSHM_THREAD_MODEL->SetTls(chi_task_counter_key_, counter);

  // Set current worker to null for client-only mode
  HSHM_THREAD_MODEL->SetTls(chi_cur_worker_key_,
                            static_cast<Worker *>(nullptr));

  // Create scheduler using factory
  if (config && config->IsValid()) {
    std::string sched_name = config->GetLocalSched();
    scheduler_ = SchedulerFactory::Get(sched_name);
    HLOG(kDebug, "Scheduler initialized: {}", sched_name);
  }

  is_initialized_ = true;
  return true;
}

bool IpcManager::ServerInit() {
  if (is_initialized_) {
    return true;
  }

  // Initialize memory segments for server
  if (!ServerInitShm()) {
    return false;
  }

  // Initialize priority queues
  if (!ServerInitQueues()) {
    return false;
  }

  // Identify this host and store node ID in shared header
  if (!IdentifyThisHost()) {
    HLOG(kError, "Warning: Could not identify host, using default node ID");
    this_host_ = Host(); // Default constructor gives node_id = 0
    if (shared_header_) {
      shared_header_->node_id = this_host_.node_id;
    }
  } else {
    // Store the identified host's node ID in shared header
    if (shared_header_) {
      shared_header_->node_id = this_host_.node_id;
    }

    HLOG(kDebug, "Node ID stored in shared memory: 0x{:x}",
          this_host_.node_id);
  }

  // Initialize HSHM TLS key for task counter (needed for CreateTaskId in
  // runtime)
  HSHM_THREAD_MODEL->CreateTls<TaskCounter>(chi_task_counter_key_, nullptr);

  // Create scheduler using factory
  auto *config = CHI_CONFIG_MANAGER;
  if (config && config->IsValid()) {
    std::string sched_name = config->GetLocalSched();
    scheduler_ = SchedulerFactory::Get(sched_name);
    HLOG(kDebug, "Scheduler initialized: {}", sched_name);
  }

  // Create per-process shared memory for runtime allocations
  // Use configured client_data_segment_size from config
  size_t initial_size = config && config->IsValid()
      ? config->GetMemorySegmentSize(kClientDataSegment)
      : hshm::Unit<size_t>::Megabytes(256);  // Default 256MB
  if (!IncreaseMemory(initial_size)) {
    HLOG(kError, "IpcManager::ServerInit: Failed to create per-process shared memory");
    return false;
  }

  is_initialized_ = true;
  return true;
}

void IpcManager::ClientFinalize() {
  // Clean up thread-local task counter
  TaskCounter *counter =
      HSHM_THREAD_MODEL->GetTls<TaskCounter>(chi_task_counter_key_);
  if (counter) {
    delete counter;
    HSHM_THREAD_MODEL->SetTls(chi_task_counter_key_,
                              static_cast<TaskCounter *>(nullptr));
  }

  // Clients should not destroy shared resources
}

void IpcManager::ServerFinalize() {
  if (!is_initialized_) {
    return;
  }

  // Cleanup servers
  local_server_.reset();
  main_server_.reset();

  // Cleanup task queue in shared header (queue handles cleanup automatically)
  // Only the last process to detach will actually destroy shared data
  shared_header_ = nullptr;

  // Clear main allocator pointer
  main_allocator_ = nullptr;

  is_initialized_ = false;
}

// Template methods (NewTask, DelTask, AllocateBuffer, Enqueue) are implemented
// inline in the header

TaskQueue *IpcManager::GetTaskQueue() { return worker_queues_.ptr_; }

bool IpcManager::IsInitialized() const { return is_initialized_; }

u32 IpcManager::GetWorkerCount() {
  if (!shared_header_) {
    return 0;
  }
  return shared_header_->num_workers;
}

u32 IpcManager::GetNumSchedQueues() const {
  if (!shared_header_) {
    return 0;
  }
  return shared_header_->num_sched_queues;
}

void IpcManager::AwakenWorker(TaskLane* lane) {
  if (!lane) {
    HLOG(kWarning, "AwakenWorker: lane is null");
    return;
  }

  // Always send signal to ensure worker wakes up
  // The worker may transition from active->inactive between our check and signal send
  // Sending signal when already active is safe - it's a no-op if worker is processing
  pid_t tid = lane->GetTid();
  if (tid > 0) {
    // Get runtime PID from shared header (client's getpid() won't work for runtime threads)
    pid_t runtime_pid = shared_header_ ? shared_header_->runtime_pid : getpid();

    // Send SIGUSR1 to the worker thread in the runtime process
    int result = syscall(SYS_tgkill, runtime_pid, tid, SIGUSR1);
    if (result == 0) {
      HLOG(kDebug, "AwakenWorker: Sent SIGUSR1 to runtime_pid={}, tid={} (active={}) - SUCCESS",
            runtime_pid, tid, lane->IsActive());
    } else {
      HLOG(kError, "AwakenWorker: Failed to send SIGUSR1 to runtime_pid={}, tid={} (active={}) - errno={}",
            runtime_pid, tid, lane->IsActive(), errno);
    }
  } else {
    HLOG(kWarning, "AwakenWorker: tid={} (invalid), cannot send signal", tid);
  }
}

bool IpcManager::ServerInitShm() {
  ConfigManager *config = CHI_CONFIG_MANAGER;

  try {
    // Set allocator ID for main segment
    main_allocator_id_ = hipc::AllocatorId::Get(1, 0);

    // Get configurable segment name
    std::string main_segment_name =
        config->GetSharedMemorySegmentName(kMainSegment);

    // Initialize main backend with custom header size
    if (!main_backend_.shm_init(
            main_allocator_id_,
            hshm::Unit<size_t>::Bytes(config->GetMemorySegmentSize(kMainSegment)),
            main_segment_name)) {
      return false;
    }

    // Create main allocator using backend's MakeAlloc method
    main_allocator_ = main_backend_.MakeAlloc<CHI_MAIN_ALLOC_T>();
    if (!main_allocator_) {
      return false;
    }

    // Add main allocator to alloc_map_ for ToFullPtr lookup
    u64 alloc_key = (static_cast<u64>(main_allocator_id_.major_) << 32) |
                    static_cast<u64>(main_allocator_id_.minor_);
    alloc_map_[alloc_key] = reinterpret_cast<hipc::MultiProcessAllocator*>(main_allocator_);

    return true;
  } catch (const std::exception &e) {
    return false;
  }
}

bool IpcManager::ClientInitShm() {
  ConfigManager *config = CHI_CONFIG_MANAGER;

  try {
    // Set allocator ID (must match server)
    main_allocator_id_ = hipc::AllocatorId(1, 0);

    // Get configurable segment name with environment variable expansion
    std::string main_segment_name =
        config->GetSharedMemorySegmentName(kMainSegment);

    // Attach to existing main shared memory segment created by server
    if (!main_backend_.shm_attach(main_segment_name)) {
      return false;
    }

    // Attach to main allocator using backend's AttachAlloc method
    main_allocator_ = main_backend_.AttachAlloc<CHI_MAIN_ALLOC_T>();
    if (!main_allocator_) {
      return false;
    }

    // Add main allocator to alloc_map_ for ToFullPtr lookup
    u64 alloc_key = (static_cast<u64>(main_allocator_id_.major_) << 32) |
                    static_cast<u64>(main_allocator_id_.minor_);
    alloc_map_[alloc_key] = reinterpret_cast<hipc::MultiProcessAllocator*>(main_allocator_);

    return true;
  } catch (const std::exception &e) {
    return false;
  }
}

bool IpcManager::ServerInitQueues() {
  if (!main_allocator_) {
    return false;
  }

  try {
    // Get the custom header from the backend
    shared_header_ =
        main_backend_.template GetSharedHeader<IpcSharedHeader>();

    if (!shared_header_) {
      return false;
    }

    // Initialize shared header
    shared_header_->node_id = 0; // Will be set after host identification
    shared_header_->runtime_pid = getpid(); // Store runtime's PID for client tgkill

    // Get worker counts from ConfigManager
    ConfigManager *config = CHI_CONFIG_MANAGER;
    u32 sched_count = config->GetSchedulerWorkerCount();
    u32 slow_count = config->GetSlowWorkerCount();
    u32 net_worker_count = 1;  // Dedicated network worker (hardcoded to 1)
    u32 total_workers = sched_count + slow_count + net_worker_count;

    // Number of scheduling queues equals number of sched workers
    u32 num_sched_queues = sched_count;

    // Store worker count and scheduling queue count
    shared_header_->num_workers = total_workers;
    shared_header_->num_sched_queues = num_sched_queues;

    // Initialize TaskQueue in shared header
    // Number of lanes equals total worker count (including net worker)
    new (&shared_header_->worker_queues) TaskQueue(
        main_allocator_,
        total_workers,  // num_lanes equals total worker count
        2,      // num_priorities (2 priorities: 0=normal, 1=resumed tasks)
        1024);  // depth_per_queue

    // Create FullPtr reference to the shared TaskQueue
    worker_queues_ = hipc::FullPtr<TaskQueue>(main_allocator_,
                                              &shared_header_->worker_queues);

    // Initialize network queue for send operations
    // One lane with two priorities (SendIn and SendOut)
    net_queue_ = main_allocator_->NewObj<NetQueue>(
        main_allocator_,
        1,     // num_lanes: single lane for network operations
        2,     // num_priorities: 0=SendIn, 1=SendOut
        1024); // depth_per_queue

    return !worker_queues_.IsNull() && !net_queue_.IsNull();
  } catch (const std::exception &e) {
    return false;
  }
}

bool IpcManager::ClientInitQueues() {
  if (!main_allocator_) {
    return false;
  }

  try {
    // Get the custom header from the backend
    shared_header_ =
        main_backend_.template GetSharedHeader<IpcSharedHeader>();

    if (!shared_header_) {
      return false;
    }

    // Client accesses the server's shared TaskQueue
    // Create FullPtr reference to the shared TaskQueue
    worker_queues_ = hipc::FullPtr<TaskQueue>(main_allocator_,
                                              &shared_header_->worker_queues);

    return !worker_queues_.IsNull();
  } catch (const std::exception &e) {
    return false;
  }
}

bool IpcManager::StartLocalServer() {
  ConfigManager *config = CHI_CONFIG_MANAGER;

  try {
    // Start local ZeroMQ server using HSHM Lightbeam
    std::string addr = "127.0.0.1";
    std::string protocol = "tcp";
    u32 port = config->GetPort() + 1; // Use ZMQ port + 1 for local server

    local_server_ = hshm::lbm::TransportFactory::GetServer(
        addr, hshm::lbm::Transport::kZeroMq, protocol, port);

    if (local_server_ != nullptr) {
      HLOG(kInfo, "Successfully started local server at {}:{}", addr, port);
      return true;
    }

    HLOG(kError, "Failed to start local server at {}:{}", addr, port);
    return false;
  } catch (const std::exception &e) {
    HLOG(kError, "Exception starting local server: {}", e.what());
    return false;
  }
}

bool IpcManager::WaitForLocalServer() {
  ConfigManager *config = CHI_CONFIG_MANAGER;

  // Read environment variables for wait configuration
  const char *wait_env = std::getenv("CHI_WAIT_SERVER");
  if (wait_env != nullptr) {
    wait_server_timeout_ = static_cast<u32>(std::atoi(wait_env));
  }

  // Heartbeat server runs on port+2 (main server on port, PULL on port+1)
  u32 heartbeat_port = config->GetPort() + 2;
  HLOG(kInfo,
       "Waiting for runtime heartbeat on 127.0.0.1:{} (timeout={}s)",
       heartbeat_port, wait_server_timeout_);

  // Create ZeroMQ REQ socket for heartbeat request/response
  void *hb_ctx = zmq_ctx_new();
  if (hb_ctx == nullptr) {
    HLOG(kError, "Failed to create ZMQ context");
    return false;
  }

  void *hb_socket = zmq_socket(hb_ctx, ZMQ_REQ);
  if (hb_socket == nullptr) {
    HLOG(kError, "Failed to create ZMQ REQ socket");
    zmq_ctx_destroy(hb_ctx);
    return false;
  }

  // Set linger to 0 so close doesn't block
  int linger = 0;
  zmq_setsockopt(hb_socket, ZMQ_LINGER, &linger, sizeof(linger));

  // Set receive timeout in milliseconds
  int timeout_ms = static_cast<int>(wait_server_timeout_ * 1000);
  zmq_setsockopt(hb_socket, ZMQ_RCVTIMEO, &timeout_ms, sizeof(timeout_ms));

  // Connect to heartbeat server
  std::string url = "tcp://127.0.0.1:" + std::to_string(heartbeat_port);
  int rc = zmq_connect(hb_socket, url.c_str());
  if (rc == -1) {
    HLOG(kError, "Failed to connect to heartbeat server at {}: {}",
         url, zmq_strerror(zmq_errno()));
    zmq_close(hb_socket);
    zmq_ctx_destroy(hb_ctx);
    return false;
  }

  // Send heartbeat request (value 1)
  int32_t request = 1;
  rc = zmq_send(hb_socket, &request, sizeof(request), 0);
  if (rc == -1) {
    HLOG(kError, "Failed to send heartbeat request: {}",
         zmq_strerror(zmq_errno()));
    zmq_close(hb_socket);
    zmq_ctx_destroy(hb_ctx);
    return false;
  }
  HLOG(kDebug, "Sent heartbeat request, waiting for response...");

  // RECEIVE heartbeat response (blocking with timeout)
  int32_t response = -1;
  rc = zmq_recv(hb_socket, &response, sizeof(response), 0);
  if (rc == -1) {
    int err = zmq_errno();
    if (err == EAGAIN) {
      HLOG(kError,
           "Timeout waiting for runtime after {} seconds",
           wait_server_timeout_);
    } else {
      HLOG(kError, "Failed to receive heartbeat response: {}",
           zmq_strerror(err));
    }
    HLOG(kError, "This usually means:");
    HLOG(kError, "1. Chimaera runtime is not running");
    HLOG(kError, "2. Runtime failed to start heartbeat server");
    HLOG(kError, "3. Network connectivity issues");
    zmq_close(hb_socket);
    zmq_ctx_destroy(hb_ctx);
    return false;
  }

  // Check response value (0 = success)
  HLOG(kInfo, "Received heartbeat response: {}", response);
  zmq_close(hb_socket);
  zmq_ctx_destroy(hb_ctx);

  if (response == 0) {
    HLOG(kInfo, "Successfully connected to runtime (heartbeat received)");
    return true;
  }

  HLOG(kError, "Runtime responded with error code: {}", response);
  return false;
}

void IpcManager::SetNodeId(const std::string &hostname) {
  (void)hostname; // Unused parameter
  if (!shared_header_) {
    return;
  }

  // Set the node ID from this_host_ which was identified during
  // IdentifyThisHost
  shared_header_->node_id = this_host_.node_id;
}

u64 IpcManager::GetNodeId() const {
  // Return the node ID from the identified host
  return this_host_.node_id;
}

bool IpcManager::LoadHostfile() {
  ConfigManager *config = CHI_CONFIG_MANAGER;
  std::string hostfile_path = config->GetHostfilePath();

  // Clear existing hostfile map
  hostfile_map_.clear();
  hosts_cache_valid_ = false;

  if (hostfile_path.empty()) {
    // No hostfile configured - assume localhost as node 0
    HLOG(kDebug, "No hostfile configured, using localhost as node 0");
    Host host("127.0.0.1", 0);
    hostfile_map_[0] = host;
    return true;
  }

  try {
    // Use HSHM to parse hostfile
    std::vector<std::string> host_ips =
        hshm::ConfigParse::ParseHostfile(hostfile_path);

    // Create Host structs and populate map using linear offset-based node IDs
    HLOG(kInfo, "=== Container to Node ID Mapping (Linear Offset) ===");
    for (size_t offset = 0; offset < host_ips.size(); ++offset) {
      u64 node_id = static_cast<u64>(offset);
      Host host(host_ips[offset], node_id);
      hostfile_map_[node_id] = host;
      HLOG(kInfo, "  Hostfile[{}]: {} -> Node ID: {}", offset,
            host_ips[offset], node_id);
    }
    HLOG(kInfo, "=== Total hosts loaded: {} ===", hostfile_map_.size());
    if (hostfile_map_.empty()) {
      HLOG(kFatal, "There were no hosts in the hostfile {}", hostfile_path);
    }
    return true;

  } catch (const std::exception &e) {
    HLOG(kError, "Error loading hostfile {}: {}", hostfile_path, e.what());
    return false;
  }
}

const Host *IpcManager::GetHost(u64 node_id) const {
  auto it = hostfile_map_.find(node_id);
  if (it == hostfile_map_.end()) {
    // Log all available node IDs when lookup fails
    HLOG(kError,
          "GetHost: Looking for node_id {} but not found. Available nodes:",
          node_id);
    for (const auto &pair : hostfile_map_) {
      HLOG(kError, "  Node ID: {} -> IP: {}", pair.first,
            pair.second.ip_address);
    }
    return nullptr;
  }
  return &it->second;
}

const Host *IpcManager::GetHostByIp(const std::string &ip_address) const {
  // Search through hostfile_map_ for matching IP address
  for (const auto &pair : hostfile_map_) {
    if (pair.second.ip_address == ip_address) {
      return &pair.second;
    }
  }
  return nullptr;
}

const std::vector<Host> &IpcManager::GetAllHosts() const {
  // Rebuild cache if invalid
  if (!hosts_cache_valid_) {
    hosts_cache_.clear();
    hosts_cache_.reserve(hostfile_map_.size());

    for (const auto &pair : hostfile_map_) {
      hosts_cache_.push_back(pair.second);
    }

    hosts_cache_valid_ = true;
  }

  return hosts_cache_;
}

size_t IpcManager::GetNumHosts() const { return hostfile_map_.size(); }

bool IpcManager::IdentifyThisHost() {
  HLOG(kDebug, "Identifying current host");

  // Load hostfile if not already loaded
  if (hostfile_map_.empty()) {
    if (!LoadHostfile()) {
      HLOG(kError, "Error: Failed to load hostfile");
      return false;
    }
  }

  if (hostfile_map_.empty()) {
    HLOG(kError, "ERROR: No hosts available for identification");
    return false;
  }

  HLOG(kDebug, "Attempting to identify host among {} candidates",
        hostfile_map_.size());

  // Get port number for error reporting
  ConfigManager *config = CHI_CONFIG_MANAGER;
  u32 port = config->GetPort();

  // Collect list of attempted hosts for error reporting
  std::vector<std::string> attempted_hosts;

  // Try to start TCP server on each host IP
  for (const auto &pair : hostfile_map_) {
    const Host &host = pair.second;
    attempted_hosts.push_back(host.ip_address);
    HLOG(kDebug, "Trying to bind TCP server to: {}", host.ip_address);

    try {
      if (TryStartMainServer(host.ip_address)) {
        HLOG(kInfo, "SUCCESS: Main server started on {} (node={})",
              host.ip_address, host.node_id);
        this_host_ = host;
        return true;
      }
    } catch (const std::exception &e) {
      HLOG(kDebug, "Failed to bind to {}: {}", host.ip_address, e.what());
    } catch (...) {
      HLOG(kDebug, "Failed to bind to {}: Unknown error", host.ip_address);
    }
  }

  // Build detailed error message with hosts and port
  HLOG(kError, "ERROR: Could not start TCP server on any host from hostfile");
  HLOG(kError, "Port attempted: {}", port);
  HLOG(kError, "Hosts checked ({} total):", attempted_hosts.size());
  for (const auto &host_ip : attempted_hosts) {
    HLOG(kError, "  - {}", host_ip);
  }
  HLOG(kError, "");
  HLOG(
      kError,
      "This usually means another process is already running on the same port");
  HLOG(kError, "");
  HLOG(kError, "To check which process is using port {}, run:", port);
  HLOG(kError, "  Linux:   sudo lsof -i :{} -P -n", port);
  HLOG(kError, "           sudo netstat -tulpn | grep :{}", port);
  HLOG(kError, "  macOS:   sudo lsof -i :{} -P -n", port);
  HLOG(kError, "           sudo lsof -nP -iTCP:{} | grep LISTEN", port);
  HLOG(kError, "");
  HLOG(kError, "To stop the Chimaera runtime, run:");
  HLOG(kError, "  chimaera_stop_runtime");
  HLOG(kError, "");
  HLOG(kError, "Or kill the process directly:");
  HLOG(kError, "  pkill -9 chimaera_start_runtime");
  HLOG(kFatal, "  kill -9 <PID>");
  return false;
}

const std::string &IpcManager::GetCurrentHostname() const {
  return this_host_.ip_address;
}


bool IpcManager::TryStartMainServer(const std::string &hostname) {
  ConfigManager *config = CHI_CONFIG_MANAGER;

  try {
    // Create main server using Lightbeam TransportFactory
    std::string protocol = "tcp";
    u32 port = config->GetPort();

    HLOG(kDebug, "Attempting to start main server on {}:{}", hostname, port);

    main_server_ = hshm::lbm::TransportFactory::GetServer(
        hostname, hshm::lbm::Transport::kZeroMq, protocol, port);

    if (!main_server_) {
      HLOG(kDebug,
            "Failed to create main server on {}:{} - server creation returned "
            "null",
            hostname, port);
      return false;
    }

    HLOG(kDebug, "Main server successfully bound to {}:{}", hostname, port);

    // Create heartbeat server on port+2 for client connection verification
    // Heartbeat server binds to loopback (127.0.0.1) since it's only used
    // for local client verification, not distributed communication
    u32 heartbeat_port = port + 2;
    std::string heartbeat_host = "127.0.0.1";
    HLOG(kDebug, "Starting heartbeat server on {}:{}", heartbeat_host, heartbeat_port);

    // Create raw ZMQ context and REP socket for heartbeat
    heartbeat_ctx_ = zmq_ctx_new();
    if (heartbeat_ctx_ == nullptr) {
      HLOG(kError, "Failed to create ZMQ context for heartbeat server");
      return false;
    }

    heartbeat_socket_ = zmq_socket(heartbeat_ctx_, ZMQ_REP);
    if (heartbeat_socket_ == nullptr) {
      HLOG(kError, "Failed to create ZMQ REP socket for heartbeat server");
      zmq_ctx_destroy(heartbeat_ctx_);
      heartbeat_ctx_ = nullptr;
      return false;
    }

    std::string heartbeat_url =
        protocol + "://" + heartbeat_host + ":" + std::to_string(heartbeat_port);
    int rc = zmq_bind(heartbeat_socket_, heartbeat_url.c_str());
    if (rc == -1) {
      HLOG(kError, "Failed to bind heartbeat server to {}: {}",
           heartbeat_url, zmq_strerror(zmq_errno()));
      zmq_close(heartbeat_socket_);
      zmq_ctx_destroy(heartbeat_ctx_);
      heartbeat_socket_ = nullptr;
      heartbeat_ctx_ = nullptr;
      return false;
    }

    HLOG(kInfo, "Heartbeat server started on {}", heartbeat_url);

    return true;

  } catch (const std::exception &e) {
    HLOG(kDebug, "Failed to start main server on {}:{} - exception: {}",
          hostname, config->GetPort(), e.what());
    return false;
  } catch (...) {
    HLOG(kDebug, "Failed to start main server on {}:{} - unknown exception",
          hostname, config->GetPort());
    return false;
  }
}

hshm::lbm::Server *IpcManager::GetMainServer() const {
  return main_server_.get();
}

void *IpcManager::GetHeartbeatSocket() const {
  return heartbeat_socket_;
}

const Host &IpcManager::GetThisHost() const { return this_host_; }

FullPtr<char> IpcManager::AllocateBuffer(size_t size) {
  // Use per-process shared memory allocation strategy for both client and runtime
  // 1. Check last accessed allocator first (fast path)
  if (last_alloc_ != nullptr) {
    FullPtr<char> buffer = last_alloc_->AllocateObjs<char>(size);
    if (!buffer.IsNull()) {
      return buffer;
    }
  }

  // 2. Check all allocators in alloc_vector_
  {
    std::lock_guard<std::mutex> lock(shm_mutex_);
    for (auto *alloc : alloc_vector_) {
      if (alloc != nullptr && alloc != last_alloc_) {
        FullPtr<char> buffer = alloc->AllocateObjs<char>(size);
        if (!buffer.IsNull()) {
          last_alloc_ = alloc;  // Update last accessed
          return buffer;
        }
      }
    }
  }

  // 3. All existing allocators are full - create new shared memory segment
  // Calculate segment size: (requested_size + 32MB metadata) * 1.2 multiplier
  size_t new_size = static_cast<size_t>(
      (size + kShmMetadataOverhead) * kShmAllocationMultiplier);
  if (!IncreaseMemory(new_size)) {
    HLOG(kError, "AllocateBuffer: Failed to increase memory for {} bytes", size);
    return FullPtr<char>::GetNull();
  }

  // 4. Retry allocation from the newly created allocator (last_alloc_)
  if (last_alloc_ != nullptr) {
    FullPtr<char> buffer = last_alloc_->AllocateObjs<char>(size);
    if (!buffer.IsNull()) {
      return buffer;
    }
  }

  HLOG(kError, "AllocateBuffer: Failed to allocate {} bytes even after increasing memory", size);
  return FullPtr<char>::GetNull();
}

void IpcManager::FreeBuffer(FullPtr<char> buffer_ptr) {
  if (buffer_ptr.IsNull()) {
    return;
  }

  // Check if allocator ID is null (private memory - use free())
  if (buffer_ptr.shm_.alloc_id_ == hipc::AllocatorId::GetNull()) {
    // Private memory - use free
    char *raw_ptr = reinterpret_cast<char *>(buffer_ptr.shm_.off_.load());
    free(raw_ptr);
    return;
  }

  // Check main allocator
  if (main_allocator_ && buffer_ptr.shm_.alloc_id_ == main_allocator_id_) {
    main_allocator_->Free(buffer_ptr);
    return;
  }

  // Check per-process shared memory allocators via alloc_map_
  u64 alloc_key = (static_cast<u64>(buffer_ptr.shm_.alloc_id_.major_) << 32) |
                  static_cast<u64>(buffer_ptr.shm_.alloc_id_.minor_);
  auto it = alloc_map_.find(alloc_key);
  if (it != alloc_map_.end()) {
    it->second->Free(buffer_ptr);
    return;
  }

  HLOG(kWarning, "FreeBuffer: Could not find allocator for alloc_id ({}.{})",
       buffer_ptr.shm_.alloc_id_.major_, buffer_ptr.shm_.alloc_id_.minor_);
}

hshm::lbm::Client* IpcManager::GetOrCreateClient(const std::string& addr,
                                                  int port) {
  // Create key for the pool map
  std::string key = addr + ":" + std::to_string(port);

  // Lock the pool for thread-safe access
  std::lock_guard<std::mutex> lock(client_pool_mutex_);

  // Check if client already exists
  auto it = client_pool_.find(key);
  if (it != client_pool_.end()) {
    HLOG(kDebug, "[ClientPool] Reusing existing connection to {}", key);
    return it->second.get();
  }

  // Create new persistent client connection
  HLOG(kInfo, "[ClientPool] Creating new persistent connection to {}", key);
  auto client = hshm::lbm::TransportFactory::GetClient(
      addr, hshm::lbm::Transport::kZeroMq, "tcp", port);

  if (!client) {
    HLOG(kError, "[ClientPool] Failed to create client for {}", key);
    return nullptr;
  }

  // Store in pool and return raw pointer
  hshm::lbm::Client* raw_ptr = client.get();
  client_pool_[key] = std::move(client);

  HLOG(kInfo, "[ClientPool] Connection established to {}", key);
  return raw_ptr;
}

void IpcManager::ClearClientPool() {
  std::lock_guard<std::mutex> lock(client_pool_mutex_);
  HLOG(kInfo, "[ClientPool] Clearing {} persistent connections",
        client_pool_.size());
  client_pool_.clear();
}

void IpcManager::EnqueueNetTask(Future<Task> future, NetQueuePriority priority) {
  if (net_queue_.IsNull()) {
    HLOG(kError, "EnqueueNetTask: net_queue_ is null");
    return;
  }

  // Get lane 0 (single lane) with the specified priority
  u32 priority_idx = static_cast<u32>(priority);
  auto& lane = net_queue_->GetLane(0, priority_idx);
  lane.Push(future);

  HLOG(kDebug, "EnqueueNetTask: Enqueued task to priority {} queue", priority_idx);
}

bool IpcManager::TryPopNetTask(NetQueuePriority priority, Future<Task>& future) {
  if (net_queue_.IsNull()) {
    return false;
  }

  // Get lane 0 (single lane) with the specified priority
  u32 priority_idx = static_cast<u32>(priority);
  auto& lane = net_queue_->GetLane(0, priority_idx);

  if (lane.Pop(future)) {
    // Fix the allocator pointer after popping using IpcManager::ToFullPtr
    future.SetAllocator();
    return true;
  }

  return false;
}

//==============================================================================
// Per-Process Shared Memory Management
//==============================================================================

bool IpcManager::IncreaseMemory(size_t size) {
  std::lock_guard<std::mutex> lock(shm_mutex_);

  pid_t pid = getpid();
  u32 index = shm_count_.fetch_add(1, std::memory_order_relaxed);

  // Create shared memory name: chimaera_{pid}_{index}
  std::string shm_name = "chimaera_" + std::to_string(pid) + "_" + std::to_string(index);

  // Add 32MB metadata overhead
  size_t total_size = size + kShmMetadataOverhead;

  HLOG(kInfo, "IpcManager::IncreaseMemory: Creating {} with size {} ({} + {} overhead)",
       shm_name, total_size, size, kShmMetadataOverhead);

  try {
    // Create the shared memory backend
    auto backend = std::make_unique<hipc::PosixShmMmap>();

    // Create allocator ID: major = pid, minor = index
    hipc::AllocatorId alloc_id(static_cast<u32>(pid), index);

    // Initialize shared memory using backend's shm_init method
    if (!backend->shm_init(alloc_id,
                           hshm::Unit<size_t>::Bytes(total_size),
                           shm_name)) {
      HLOG(kError, "IpcManager::IncreaseMemory: Failed to create shm for {}",
           shm_name);
      shm_count_.fetch_sub(1, std::memory_order_relaxed);
      return false;
    }

    // Create allocator using backend's MakeAlloc method
    hipc::MultiProcessAllocator *allocator =
        backend->MakeAlloc<hipc::MultiProcessAllocator>();

    if (allocator == nullptr) {
      HLOG(kError, "IpcManager::IncreaseMemory: Failed to create allocator for {}",
           shm_name);
      shm_count_.fetch_sub(1, std::memory_order_relaxed);
      return false;
    }

    // Add to our tracking structures
    u64 alloc_key = (static_cast<u64>(alloc_id.major_) << 32) |
                    static_cast<u64>(alloc_id.minor_);
    alloc_map_[alloc_key] = allocator;
    alloc_vector_.push_back(allocator);
    client_backends_.push_back(std::move(backend));
    last_alloc_ = allocator;

    HLOG(kInfo, "IpcManager::IncreaseMemory: Created allocator {} with ID ({}.{})",
         shm_name, alloc_id.major_, alloc_id.minor_);

    // Note: Registration with runtime is now done lazily in SetAllocator()
    // when the worker first encounters a FutureShm from this client's memory

    return true;

  } catch (const std::exception &e) {
    HLOG(kError, "IpcManager::IncreaseMemory: Exception creating {}: {}",
         shm_name, e.what());
    shm_count_.fetch_sub(1, std::memory_order_relaxed);
    return false;
  }
}

bool IpcManager::RegisterMemory(const hipc::AllocatorId &alloc_id, size_t shm_size) {
  std::lock_guard<std::mutex> lock(shm_mutex_);

  // Derive shm_name from alloc_id: chimaera_{pid}_{index}
  pid_t owner_pid = static_cast<pid_t>(alloc_id.major_);
  u32 shm_index = alloc_id.minor_;
  std::string shm_name = "chimaera_" + std::to_string(owner_pid) + "_" + std::to_string(shm_index);

  HLOG(kInfo, "IpcManager::RegisterMemory: Registering {} (size={}) from pid {}",
       shm_name, shm_size, owner_pid);

  // Check if already registered
  u64 alloc_key = (static_cast<u64>(alloc_id.major_) << 32) |
                  static_cast<u64>(alloc_id.minor_);
  if (alloc_map_.find(alloc_key) != alloc_map_.end()) {
    HLOG(kInfo,
         "IpcManager::RegisterMemory: {} already registered, skipping",
         shm_name);
    return true;  // Already registered
  }

  try {
    // Attach to the shared memory backend (already created by client)
    auto backend = std::make_unique<hipc::PosixShmMmap>();
    if (!backend->shm_attach(shm_name)) {
      HLOG(kError,
           "IpcManager::RegisterMemory: Failed to attach to shm {}",
           shm_name);
      return false;
    }

    // Attach to the existing allocator in the backend
    hipc::MultiProcessAllocator *allocator =
        backend->AttachAlloc<hipc::MultiProcessAllocator>();

    if (allocator == nullptr) {
      HLOG(kError,
           "IpcManager::RegisterMemory: Failed to attach allocator for {}",
           shm_name);
      return false;
    }

    // Add to our tracking structures
    alloc_map_[alloc_key] = allocator;
    // Note: Don't add to alloc_vector_ since this is not our memory
    // (we don't allocate from it, just need to resolve ShmPtrs)
    client_backends_.push_back(std::move(backend));

    HLOG(kInfo, "IpcManager::RegisterMemory: Successfully registered {}",
         shm_name);

    return true;

  } catch (const std::exception &e) {
    HLOG(kError, "IpcManager::RegisterMemory: Exception registering {}: {}",
         shm_name, e.what());
    return false;
  }
}

ClientShmInfo IpcManager::GetClientShmInfo(u32 index) const {
  std::lock_guard<std::mutex> lock(shm_mutex_);

  if (index >= alloc_vector_.size()) {
    return ClientShmInfo();  // Return empty info
  }

  pid_t pid = getpid();
  std::string shm_name = "chimaera_" + std::to_string(pid) + "_" + std::to_string(index);

  hipc::MultiProcessAllocator *allocator = alloc_vector_[index];
  hipc::AllocatorId alloc_id = allocator->GetId();

  // Get size from backend if available, otherwise use 0
  size_t size = 0;
  if (index < client_backends_.size() && client_backends_[index]) {
    size = client_backends_[index]->backend_size_;
  }

  return ClientShmInfo(shm_name, pid, index, size, alloc_id);
}

size_t IpcManager::WreapDeadIpcs() {
  std::lock_guard<std::mutex> lock(shm_mutex_);

  pid_t current_pid = getpid();
  size_t reaped_count = 0;

  // Build list of allocator keys to remove (can't modify map while iterating)
  std::vector<u64> keys_to_remove;

  for (const auto &pair : alloc_map_) {
    u64 alloc_key = pair.first;

    // Extract pid from allocator key (major is in upper 32 bits)
    u32 major = static_cast<u32>(alloc_key >> 32);
    u32 minor = static_cast<u32>(alloc_key & 0xFFFFFFFF);

    // Skip main allocator (1.0)
    if (major == 1 && minor == 0) {
      continue;
    }

    // Skip our own process's segments
    pid_t owner_pid = static_cast<pid_t>(major);
    if (owner_pid == current_pid) {
      continue;
    }

    // Check if the owning process is still alive
    // kill(pid, 0) returns 0 if process exists, -1 with ESRCH if not
    if (kill(owner_pid, 0) == -1 && errno == ESRCH) {
      // Process is dead - mark for removal
      HLOG(kInfo, "WreapDeadIpcs: Process {} is dead, marking allocator ({}.{}) for removal",
           owner_pid, major, minor);
      keys_to_remove.push_back(alloc_key);
    }
  }

  // Remove marked allocators and their backends
  for (u64 key : keys_to_remove) {
    // Find the allocator in the map
    auto map_it = alloc_map_.find(key);
    if (map_it == alloc_map_.end()) {
      continue;
    }

    hipc::MultiProcessAllocator *allocator = map_it->second;

    // Get the allocator ID to construct shm_name
    hipc::AllocatorId alloc_id = allocator->GetId();
    std::string shm_name = "chimaera_" + std::to_string(alloc_id.major_) +
                           "_" + std::to_string(alloc_id.minor_);

    // Find and destroy the corresponding backend
    for (auto backend_it = client_backends_.begin();
         backend_it != client_backends_.end(); ++backend_it) {
      if (*backend_it && (*backend_it)->header_ &&
          (*backend_it)->header_->id_.major_ == alloc_id.major_ &&
          (*backend_it)->header_->id_.minor_ == alloc_id.minor_) {
        // Destroy the shared memory
        HLOG(kInfo, "WreapDeadIpcs: Destroying shared memory {} for allocator ({}.{})",
             shm_name, alloc_id.major_, alloc_id.minor_);
        (*backend_it)->shm_destroy();
        client_backends_.erase(backend_it);
        break;
      }
    }

    // Remove from alloc_vector_ if present
    auto vec_it = std::find(alloc_vector_.begin(), alloc_vector_.end(), allocator);
    if (vec_it != alloc_vector_.end()) {
      alloc_vector_.erase(vec_it);
    }

    // Clear last_alloc_ if it points to this allocator
    if (last_alloc_ == allocator) {
      last_alloc_ = alloc_vector_.empty() ? nullptr : alloc_vector_.back();
    }

    // Remove from alloc_map_
    alloc_map_.erase(map_it);
    reaped_count++;
  }

  if (reaped_count > 0) {
    HLOG(kInfo, "WreapDeadIpcs: Reaped {} shared memory segments from dead processes",
         reaped_count);
  }

  return reaped_count;
}

size_t IpcManager::WreapAllIpcs() {
  std::lock_guard<std::mutex> lock(shm_mutex_);

  size_t reaped_count = 0;

  // Build list of all allocator keys except main allocator (1.0)
  std::vector<u64> keys_to_remove;

  for (const auto &pair : alloc_map_) {
    u64 alloc_key = pair.first;

    // Extract pid from allocator key (major is in upper 32 bits)
    u32 major = static_cast<u32>(alloc_key >> 32);
    u32 minor = static_cast<u32>(alloc_key & 0xFFFFFFFF);

    // Skip main allocator (1.0) - it's managed separately
    if (major == 1 && minor == 0) {
      continue;
    }

    keys_to_remove.push_back(alloc_key);
  }

  // Destroy all backends and remove from tracking structures
  for (u64 key : keys_to_remove) {
    auto map_it = alloc_map_.find(key);
    if (map_it == alloc_map_.end()) {
      continue;
    }

    hipc::MultiProcessAllocator *allocator = map_it->second;

    // Get the allocator ID to construct shm_name
    hipc::AllocatorId alloc_id = allocator->GetId();
    std::string shm_name = "chimaera_" + std::to_string(alloc_id.major_) +
                           "_" + std::to_string(alloc_id.minor_);

    // Find and destroy the corresponding backend
    for (auto backend_it = client_backends_.begin();
         backend_it != client_backends_.end(); ++backend_it) {
      if (*backend_it && (*backend_it)->header_ &&
          (*backend_it)->header_->id_.major_ == alloc_id.major_ &&
          (*backend_it)->header_->id_.minor_ == alloc_id.minor_) {
        // Destroy the shared memory
        HLOG(kInfo, "WreapAllIpcs: Destroying shared memory {} for allocator ({}.{})",
             shm_name, alloc_id.major_, alloc_id.minor_);
        (*backend_it)->shm_destroy();
        client_backends_.erase(backend_it);
        break;
      }
    }

    // Remove from alloc_map_
    alloc_map_.erase(map_it);
    reaped_count++;
  }

  // Clear remaining structures
  alloc_vector_.clear();
  last_alloc_ = nullptr;

  // Note: client_backends_ may still have some entries if backends were
  // not found in the loop above (shouldn't happen in normal operation)
  if (!client_backends_.empty()) {
    HLOG(kWarning, "WreapAllIpcs: {} backends remaining after cleanup",
         client_backends_.size());
    // Destroy any remaining backends
    for (auto &backend : client_backends_) {
      if (backend) {
        backend->shm_destroy();
        reaped_count++;
      }
    }
    client_backends_.clear();
  }

  HLOG(kInfo, "WreapAllIpcs: Reaped {} shared memory segments", reaped_count);

  return reaped_count;
}

} // namespace chi