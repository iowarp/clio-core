/**
 * IPC manager implementation
 */

#include "chimaera/ipc_manager.h"

#include "chimaera/config_manager.h"
#include "chimaera/task_queue.h"
#include <arpa/inet.h>
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

  // Read lane mapping policy from configuration
  auto *config = CHI_CONFIG_MANAGER;
  if (config && config->IsValid()) {
    lane_map_policy_ = config->GetLaneMapPolicy();
    HLOG(kDebug, "Lane mapping policy set to: {}",
          static_cast<int>(lane_map_policy_));
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

  // Read lane mapping policy from configuration
  auto *config = CHI_CONFIG_MANAGER;
  if (config && config->IsValid()) {
    lane_map_policy_ = config->GetLaneMapPolicy();
    HLOG(kDebug, "Lane mapping policy set to: {}",
          static_cast<int>(lane_map_policy_));
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

  // Clear cached allocator pointers
  main_allocator_ = nullptr;
  client_data_allocator_ = nullptr;
  runtime_data_allocator_ = nullptr;

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

void IpcManager::AwakenWorker(TaskLane* lane) {
  if (!lane) {
    return;
  }

  // Only send signal if worker is inactive (blocked in epoll_wait)
  if (!lane->IsActive()) {
    pid_t tid = lane->GetTid();
    if (tid > 0) {
      // Send SIGUSR1 to the worker thread
      syscall(SYS_tgkill, getpid(), tid, SIGUSR1);
    }
  }
}

bool IpcManager::ServerInitShm() {
  ConfigManager *config = CHI_CONFIG_MANAGER;

  try {
    // Set allocator IDs for each segment
    main_allocator_id_ = hipc::AllocatorId::Get(1, 0);
    client_data_allocator_id_ = hipc::AllocatorId::Get(2, 0);
    runtime_data_allocator_id_ = hipc::AllocatorId::Get(3, 0);

    // Get configurable segment names
    std::string main_segment_name =
        config->GetSharedMemorySegmentName(kMainSegment);
    std::string client_data_segment_name =
        config->GetSharedMemorySegmentName(kClientDataSegment);
    std::string runtime_data_segment_name =
        config->GetSharedMemorySegmentName(kRuntimeDataSegment);

    // Initialize main backend with custom header size
    size_t custom_header_size = sizeof(IpcSharedHeader);
    if (!main_backend_.shm_init(
            main_allocator_id_,
            hshm::Unit<size_t>::Bytes(config->GetMemorySegmentSize(kMainSegment)),
            main_segment_name)) {
      return false;
    }

    // Initialize client data backend
    if (!client_data_backend_.shm_init(
            client_data_allocator_id_,
            hshm::Unit<size_t>::Bytes(
                config->GetMemorySegmentSize(kClientDataSegment)),
            client_data_segment_name)) {
      return false;
    }

    // Initialize runtime data backend
    if (!runtime_data_backend_.shm_init(
            runtime_data_allocator_id_,
            hshm::Unit<size_t>::Bytes(
                config->GetMemorySegmentSize(kRuntimeDataSegment)),
            runtime_data_segment_name)) {
      return false;
    }

    // Create allocators using backend's MakeAlloc method
    main_allocator_ = main_backend_.MakeAlloc<CHI_MAIN_ALLOC_T>();
    if (!main_allocator_) {
      return false;
    }

    client_data_allocator_ = client_data_backend_.MakeAlloc<CHI_CDATA_ALLOC_T>();
    if (!client_data_allocator_) {
      return false;
    }

    runtime_data_allocator_ = runtime_data_backend_.MakeAlloc<CHI_RDATA_ALLOC_T>();
    if (!runtime_data_allocator_) {
      return false;
    }

    return true;
  } catch (const std::exception &e) {
    return false;
  }
}

bool IpcManager::ClientInitShm() {
  ConfigManager *config = CHI_CONFIG_MANAGER;

  try {
    // Set allocator IDs (must match server)
    main_allocator_id_ = hipc::AllocatorId(1, 0);
    client_data_allocator_id_ = hipc::AllocatorId(2, 0);
    runtime_data_allocator_id_ = hipc::AllocatorId(3, 0);

    // Get configurable segment names with environment variable expansion
    std::string main_segment_name =
        config->GetSharedMemorySegmentName(kMainSegment);
    std::string client_data_segment_name =
        config->GetSharedMemorySegmentName(kClientDataSegment);
    std::string runtime_data_segment_name =
        config->GetSharedMemorySegmentName(kRuntimeDataSegment);

    // Attach to existing shared memory segments created by server
    if (!main_backend_.shm_attach(main_segment_name)) {
      return false;
    }

    if (!client_data_backend_.shm_attach(client_data_segment_name)) {
      return false;
    }

    if (!runtime_data_backend_.shm_attach(runtime_data_segment_name)) {
      return false;
    }

    // Attach to allocators using backend's AttachAlloc method
    main_allocator_ = main_backend_.AttachAlloc<CHI_MAIN_ALLOC_T>();
    if (!main_allocator_) {
      return false;
    }

    client_data_allocator_ = client_data_backend_.AttachAlloc<CHI_CDATA_ALLOC_T>();
    if (!client_data_allocator_) {
      return false;
    }

    runtime_data_allocator_ = runtime_data_backend_.AttachAlloc<CHI_RDATA_ALLOC_T>();
    if (!runtime_data_allocator_) {
      return false;
    }

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

void IpcManager::SetLaneMapPolicy(LaneMapPolicy policy) {
  lane_map_policy_ = policy;
}

LaneMapPolicy IpcManager::GetLaneMapPolicy() const { return lane_map_policy_; }

LaneId IpcManager::MapByPidTid(u32 num_lanes) {
  // Use HSHM_SYSTEM_INFO to get both PID and TID for lane hashing
  auto *sys_info = HSHM_SYSTEM_INFO;
  pid_t pid = sys_info->pid_;
  auto tid = HSHM_THREAD_MODEL->GetTid();

  // Combine PID and TID for hashing to ensure different processes/threads use
  // different lanes
  size_t combined_hash =
      std::hash<pid_t>{}(pid) ^ (std::hash<void *>{}(&tid) << 1);
  return static_cast<LaneId>(combined_hash % num_lanes);
}

LaneId IpcManager::MapRoundRobin(u32 num_lanes) {
  // Use atomic counter for round-robin distribution
  u32 counter = round_robin_counter_.fetch_add(1, std::memory_order_relaxed);
  return static_cast<LaneId>(counter % num_lanes);
}

LaneId IpcManager::MapRandom(u32 num_lanes) {
  // Use thread-local random number generator for efficiency
  thread_local std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<u32> dist(0, num_lanes - 1);
  return static_cast<LaneId>(dist(rng));
}

LaneId IpcManager::MapTaskToLane(u32 num_lanes) {
  if (num_lanes == 0) {
    return 0; // Avoid division by zero
  }

  switch (lane_map_policy_) {
  case LaneMapPolicy::kMapByPidTid:
    return MapByPidTid(num_lanes);

  case LaneMapPolicy::kRoundRobin:
    return MapRoundRobin(num_lanes);

  case LaneMapPolicy::kRandom:
    return MapRandom(num_lanes);

  default:
    // Fallback to round-robin
    return MapRoundRobin(num_lanes);
  }
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
    u32 heartbeat_port = port + 2;
    HLOG(kDebug, "Starting heartbeat server on {}:{}", hostname, heartbeat_port);

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
        protocol + "://" + hostname + ":" + std::to_string(heartbeat_port);
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
  auto *chimaera_manager = CHI_CHIMAERA_MANAGER;

  // Determine which allocator to use
  CHI_RDATA_ALLOC_T *allocator = nullptr;
  if (chimaera_manager && chimaera_manager->IsRuntime()) {
    // Runtime uses rdata segment
    if (!runtime_data_allocator_) {
      return FullPtr<char>::GetNull();
    }
    allocator = runtime_data_allocator_;
  } else {
    // Client uses cdata segment
    if (!client_data_allocator_) {
      return FullPtr<char>::GetNull();
    }
    allocator = reinterpret_cast<CHI_RDATA_ALLOC_T *>(client_data_allocator_);
  }

  // Loop until allocation succeeds
  FullPtr<char> buffer = FullPtr<char>::GetNull();
  while (buffer.IsNull()) {
    buffer = allocator->AllocateObjs<char>(size);
    if (buffer.IsNull()) {
      // Allocation failed - yield to allow other tasks to run
      HSHM_THREAD_MODEL->Yield();
    }
  }

  return buffer;
}

void IpcManager::FreeBuffer(FullPtr<char> buffer_ptr) {
  if (buffer_ptr.IsNull()) {
    return;
  }

  // Try runtime data allocator first, then client data allocator
  auto *rdata_alloc = CHI_IPC->GetRdataAlloc();
  if (rdata_alloc && buffer_ptr.shm_.alloc_id_ == runtime_data_allocator_id_) {
    rdata_alloc->Free(buffer_ptr);
  } else {
    auto *cdata_alloc = CHI_IPC->GetDataAlloc();
    if (cdata_alloc) {
      cdata_alloc->Free(buffer_ptr);
    }
  }
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
    // Fix the allocator pointer after popping
    future.SetAllocator(main_allocator_);
    return true;
  }

  return false;
}

} // namespace chi