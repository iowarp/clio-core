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
#include <sys/socket.h>
#include <unistd.h>
#include <zmq.h>

// Global pointer variable definition for IPC manager singleton
HSHM_DEFINE_GLOBAL_PTR_VAR_CC(chi::IpcManager, g_ipc_manager);

namespace chi {

// Host struct methods

// IpcManager methods

// Constructor and destructor removed - handled by HSHM singleton pattern

bool IpcManager::ClientInit() {
  HILOG(kDebug, "IpcManager::ClientInit");
  if (is_initialized_) {
    return true;
  }

  // Wait for local server to become available - critical for client
  // functionality TestLocalServer sends heartbeat to verify connectivity
  if (!WaitForLocalServer()) {
    HELOG(kError, "CRITICAL ERROR: Cannot connect to local server.");
    HELOG(kError, "Client initialization failed. Exiting.");
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
    HILOG(kDebug, "Retrieved node ID from shared memory: 0x{:x}",
          this_host_.node_id);
  } else {
    HELOG(kError, "Warning: Could not access shared header during ClientInit");
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
    HILOG(kDebug, "Lane mapping policy set to: {}",
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
    HELOG(kError, "Warning: Could not identify host, using default node ID");
    this_host_ = Host(); // Default constructor gives node_id = 0
    if (shared_header_) {
      shared_header_->node_id = this_host_.node_id;
    }
  } else {
    // Store the identified host's node ID in shared header
    if (shared_header_) {
      shared_header_->node_id = this_host_.node_id;
    }

    HILOG(kDebug, "Node ID stored in shared memory: 0x{:x}",
          this_host_.node_id);
  }

  // Initialize HSHM TLS key for task counter (needed for CreateTaskId in
  // runtime)
  HSHM_THREAD_MODEL->CreateTls<TaskCounter>(chi_task_counter_key_, nullptr);

  // Read lane mapping policy from configuration
  auto *config = CHI_CONFIG_MANAGER;
  if (config && config->IsValid()) {
    lane_map_policy_ = config->GetLaneMapPolicy();
    HILOG(kDebug, "Lane mapping policy set to: {}",
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

TaskQueue *IpcManager::GetTaskQueue() { return external_queue_.ptr_; }

bool IpcManager::IsInitialized() const { return is_initialized_; }

bool IpcManager::InitializeWorkerQueues(u32 num_workers) {
  if (!main_allocator_ || !shared_header_) {
    return false;
  }

  try {
    // Initialize worker queues vector in shared header
    // Construct the vector with the allocator
    new (&shared_header_->worker_queues) chi::ipc::vector<WorkQueue>(main_allocator_);

    // Resize to num_workers (default-initializes each WorkQueue)
    shared_header_->worker_queues.resize(num_workers);

    // Store worker count
    shared_header_->num_workers = num_workers;

    return true;
  } catch (const std::exception &e) {
    return false;
  }
}

hipc::FullPtr<WorkQueue> IpcManager::GetWorkerQueue(u32 worker_id) {
  if (!shared_header_) {
    return hipc::FullPtr<WorkQueue>::GetNull();
  }

  if (worker_id >= shared_header_->num_workers) {
    return hipc::FullPtr<WorkQueue>::GetNull();
  }

  // Get the vector of worker queues
  auto &worker_queues_vector = shared_header_->worker_queues;

  if (worker_id >= worker_queues_vector.size()) {
    return hipc::FullPtr<WorkQueue>::GetNull();
  }

  // Return FullPtr reference to the specific worker's queue in the vector
  return ToFullPtr<WorkQueue>(&worker_queues_vector[worker_id]);
}

u32 IpcManager::GetWorkerCount() {
  if (!shared_header_) {
    return 0;
  }
  return shared_header_->num_workers;
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
    main_allocator_ = main_backend_.MakeAlloc<CHI_MAIN_ALLOC_T>(custom_header_size);
    if (!main_allocator_) {
      return false;
    }

    client_data_allocator_ = client_data_backend_.MakeAlloc<CHI_CDATA_ALLOC_T>(0);
    if (!client_data_allocator_) {
      return false;
    }

    runtime_data_allocator_ = runtime_data_backend_.MakeAlloc<CHI_RDATA_ALLOC_T>(0);
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
        main_backend_.template GetCustomHeader<IpcSharedHeader>();

    if (!shared_header_) {
      return false;
    }

    // Initialize shared header
    shared_header_->num_workers = 0;
    shared_header_->node_id = 0; // Will be set after host identification

    // Get number of sched workers from ConfigManager
    // Number of lanes equals number of sched workers for optimal distribution
    ConfigManager *config = CHI_CONFIG_MANAGER;
    u32 num_lanes = config->GetWorkerThreadCount(kSchedWorker);

    // Initialize TaskQueue in shared header
    new (&shared_header_->external_queue) TaskQueue(
        main_allocator_,
        num_lanes, // num_lanes equals sched worker count
        2,         // num_priorities (2 priorities: 0=normal, 1=resumed tasks)
        1024);     // depth_per_queue

    // Create FullPtr reference to the shared TaskQueue
    external_queue_ = hipc::FullPtr<TaskQueue>(main_allocator_,
                                               &shared_header_->external_queue);

    return !external_queue_.IsNull();
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
        main_backend_.template GetCustomHeader<IpcSharedHeader>();

    if (!shared_header_) {
      return false;
    }

    // Client accesses the server's shared TaskQueue
    // Create FullPtr reference to the shared TaskQueue
    external_queue_ = hipc::FullPtr<TaskQueue>(main_allocator_,
                                               &shared_header_->external_queue);

    return !external_queue_.IsNull();
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
      HILOG(kInfo, "Successfully started local server at {}:{}", addr, port);
      return true;
    }

    HELOG(kError, "Failed to start local server at {}:{}", addr, port);
    return false;
  } catch (const std::exception &e) {
    HELOG(kError, "Exception starting local server: {}", e.what());
    return false;
  }
}

bool IpcManager::TestLocalServer() {
  try {
    ConfigManager *config = CHI_CONFIG_MANAGER;
    std::string addr = "127.0.0.1";
    std::string protocol = "tcp";
    u32 port = config->GetPort() + 1;

    auto client = hshm::lbm::TransportFactory::GetClient(
        addr, hshm::lbm::Transport::kZeroMq, protocol, port);

    if (!client) {
      return false;
    }

    // Create empty metadata with heartbeat message type
    chi::SaveTaskArchive archive(chi::MsgType::kHeartbeat, client.get());

    // Use synchronous send (single attempt, no retry)
    hshm::lbm::LbmContext ctx(hshm::lbm::LBM_SYNC);
    int rc = client->Send(archive, ctx);

    if (rc == 0) {
      HILOG(kDebug, "Successfully sent heartbeat to local server");
      return true;
    }

    HELOG(kDebug, "Failed to send heartbeat with error code {}", rc);
    return false;
  } catch (const std::exception &e) {
    HELOG(kWarning, "Exception during heartbeat send: {}", e.what());
    return false;
  }
}

bool IpcManager::WaitForLocalServer() {
  ConfigManager *config = CHI_CONFIG_MANAGER;

  // Read environment variables for wait configuration
  const char *wait_env = std::getenv("CHI_WAIT_SERVER");
  const char *poll_env = std::getenv("CHI_POLL_SERVER");

  if (wait_env) {
    wait_server_timeout_ = static_cast<u32>(std::atoi(wait_env));
  }
  if (poll_env) {
    poll_server_interval_ = static_cast<u32>(std::atoi(poll_env));
  }

  // Ensure poll interval is at least 1 second to avoid busy-waiting
  if (poll_server_interval_ == 0) {
    poll_server_interval_ = 1;
  }

  u32 port = config->GetPort() + 1;
  HILOG(kInfo,
        "Waiting for local server at 127.0.0.1:{} (timeout={}s, "
        "poll_interval={}s)",
        port, wait_server_timeout_, poll_server_interval_);

  u32 elapsed = 0;
  u32 attempt = 0;

  while (elapsed < wait_server_timeout_) {
    attempt++;

    if (TestLocalServer()) {
      HILOG(kInfo,
            "Successfully connected to local server after {} seconds ({} "
            "attempts)",
            elapsed, attempt);
      return true;
    }

    HILOG(kDebug, "Local server not available yet (attempt {}, elapsed {}s)",
          attempt, elapsed);

    // Sleep for poll interval
    sleep(poll_server_interval_);
    elapsed += poll_server_interval_;
  }

  HELOG(kError,
        "Timeout waiting for local server after {} seconds ({} attempts)",
        wait_server_timeout_, attempt);
  HELOG(kError, "This usually means:");
  HELOG(kError, "1. Chimaera runtime is not running");
  HELOG(kError, "2. Local server failed to start");
  HELOG(kError, "3. Network connectivity issues");
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
    HILOG(kDebug, "No hostfile configured, using localhost as node 0");
    Host host("127.0.0.1", 0);
    hostfile_map_[0] = host;
    return true;
  }

  try {
    // Use HSHM to parse hostfile
    std::vector<std::string> host_ips =
        hshm::ConfigParse::ParseHostfile(hostfile_path);

    // Create Host structs and populate map using linear offset-based node IDs
    HILOG(kInfo, "=== Container to Node ID Mapping (Linear Offset) ===");
    for (size_t offset = 0; offset < host_ips.size(); ++offset) {
      u64 node_id = static_cast<u64>(offset);
      Host host(host_ips[offset], node_id);
      hostfile_map_[node_id] = host;
      HILOG(kInfo, "  Hostfile[{}]: {} -> Node ID: {}", offset,
            host_ips[offset], node_id);
    }
    HILOG(kInfo, "=== Total hosts loaded: {} ===", hostfile_map_.size());
    if (hostfile_map_.empty()) {
      HELOG(kFatal, "There were no hosts in the hostfile {}", hostfile_path);
    }
    return true;

  } catch (const std::exception &e) {
    HELOG(kError, "Error loading hostfile {}: {}", hostfile_path, e.what());
    return false;
  }
}

const Host *IpcManager::GetHost(u64 node_id) const {
  auto it = hostfile_map_.find(node_id);
  if (it == hostfile_map_.end()) {
    // Log all available node IDs when lookup fails
    HILOG(kError,
          "GetHost: Looking for node_id {} but not found. Available nodes:",
          node_id);
    for (const auto &pair : hostfile_map_) {
      HILOG(kError, "  Node ID: {} -> IP: {}", pair.first,
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
  HILOG(kDebug, "Identifying current host");

  // Load hostfile if not already loaded
  if (hostfile_map_.empty()) {
    if (!LoadHostfile()) {
      HELOG(kError, "Error: Failed to load hostfile");
      return false;
    }
  }

  if (hostfile_map_.empty()) {
    HELOG(kError, "ERROR: No hosts available for identification");
    return false;
  }

  HILOG(kDebug, "Attempting to identify host among {} candidates",
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
    HILOG(kDebug, "Trying to bind TCP server to: {}", host.ip_address);

    try {
      if (TryStartMainServer(host.ip_address)) {
        HILOG(kInfo, "SUCCESS: Main server started on {} (node={})",
              host.ip_address, host.node_id);
        this_host_ = host;
        return true;
      }
    } catch (const std::exception &e) {
      HILOG(kDebug, "Failed to bind to {}: {}", host.ip_address, e.what());
    } catch (...) {
      HILOG(kDebug, "Failed to bind to {}: Unknown error", host.ip_address);
    }
  }

  // Build detailed error message with hosts and port
  HELOG(kError, "ERROR: Could not start TCP server on any host from hostfile");
  HELOG(kError, "Port attempted: {}", port);
  HELOG(kError, "Hosts checked ({} total):", attempted_hosts.size());
  for (const auto &host_ip : attempted_hosts) {
    HELOG(kError, "  - {}", host_ip);
  }
  HELOG(kError, "");
  HELOG(
      kError,
      "This usually means another process is already running on the same port");
  HELOG(kError, "");
  HELOG(kError, "To check which process is using port {}, run:", port);
  HELOG(kError, "  Linux:   sudo lsof -i :{} -P -n", port);
  HELOG(kError, "           sudo netstat -tulpn | grep :{}", port);
  HELOG(kError, "  macOS:   sudo lsof -i :{} -P -n", port);
  HELOG(kError, "           sudo lsof -nP -iTCP:{} | grep LISTEN", port);
  HELOG(kError, "");
  HELOG(kError, "To stop the Chimaera runtime, run:");
  HELOG(kError, "  chimaera_stop_runtime");
  HELOG(kError, "");
  HELOG(kError, "Or kill the process directly:");
  HELOG(kError, "  pkill -9 chimaera_start_runtime");
  HELOG(kFatal, "  kill -9 <PID>");
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

    HILOG(kDebug, "Attempting to start main server on {}:{}", hostname, port);

    main_server_ = hshm::lbm::TransportFactory::GetServer(
        hostname, hshm::lbm::Transport::kZeroMq, protocol, port);

    if (!main_server_) {
      HILOG(kDebug,
            "Failed to create main server on {}:{} - server creation returned "
            "null",
            hostname, port);
      return false;
    }

    HILOG(kDebug, "Main server successfully bound to {}:{}", hostname, port);
    return true;

  } catch (const std::exception &e) {
    HILOG(kDebug, "Failed to start main server on {}:{} - exception: {}",
          hostname, config->GetPort(), e.what());
    return false;
  } catch (...) {
    HILOG(kDebug, "Failed to start main server on {}:{} - unknown exception",
          hostname, config->GetPort());
    return false;
  }
}

hshm::lbm::Server *IpcManager::GetMainServer() const {
  return main_server_.get();
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
      Worker *worker = CHI_CUR_WORKER;
      if (worker) {
        // We're in a task context - yield from the current task
        FullPtr<Task> current_task = worker->GetCurrentTask();
        if (!current_task.IsNull()) {
          current_task->Yield();
        } else {
          // No current task - yield from thread model
          HSHM_THREAD_MODEL->Yield();
        }
      } else {
        // Not in worker context - yield from thread model
        HSHM_THREAD_MODEL->Yield();
      }
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

template <typename T>
hipc::FullPtr<T> IpcManager::ToFullPtr(const hipc::ShmPtr<T> &shm_ptr) {
  if (shm_ptr.IsNull()) {
    return hipc::FullPtr<T>();
  }

  // Check main allocator
  if (shm_ptr.alloc_id_ == main_allocator_id_) {
    return hipc::FullPtr<T>(main_allocator_, shm_ptr);
  }

  // Check client data allocator
  if (shm_ptr.alloc_id_ == client_data_allocator_id_) {
    return hipc::FullPtr<T>(client_data_allocator_, shm_ptr);
  }

  // Check runtime data allocator
  if (shm_ptr.alloc_id_ == runtime_data_allocator_id_) {
    return hipc::FullPtr<T>(runtime_data_allocator_, shm_ptr);
  }

  // No matching allocator found
  return hipc::FullPtr<T>();
}

template <typename T>
hipc::FullPtr<T> IpcManager::ToFullPtr(T *ptr) {
  if (!ptr) {
    return hipc::FullPtr<T>();
  }

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

} // namespace chi