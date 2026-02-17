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

/**
 * IPC manager implementation
 */

#include "chimaera/ipc_manager.h"

#include <arpa/inet.h>
#include <dirent.h>
#include <endian.h>
#include <netdb.h>
#include <signal.h>
#include <sys/epoll.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#include <zmq.h>
#include <hermes_shm/lightbeam/transport_factory_impl.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>

#include "chimaera/admin.h"
#include "chimaera/admin/admin_client.h"
#include "chimaera/chimaera_manager.h"
#include "chimaera/config_manager.h"
#include "chimaera/scheduler/scheduler_factory.h"
#include "chimaera/task_queue.h"

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

  // Parse CHI_IPC_MODE environment variable (default: TCP)
  const char *ipc_mode_env = std::getenv("CHI_IPC_MODE");
  if (ipc_mode_env != nullptr) {
    std::string mode_str(ipc_mode_env);
    if (mode_str == "SHM" || mode_str == "shm") {
      ipc_mode_ = IpcMode::kShm;
    } else if (mode_str == "IPC" || mode_str == "ipc") {
      ipc_mode_ = IpcMode::kIpc;
    } else {
      ipc_mode_ = IpcMode::kTcp;  // Default
    }
  }
  HLOG(kInfo, "IpcManager::ClientInit: IPC mode = {}",
       ipc_mode_ == IpcMode::kShm ? "SHM" :
       ipc_mode_ == IpcMode::kIpc ? "IPC" : "TCP");

  // Create lightbeam transport for client-server communication
  {
    auto *config = CHI_CONFIG_MANAGER;
    u32 port = config->GetPort();

    if (ipc_mode_ == IpcMode::kIpc) {
      // IPC mode: Unix domain socket transport
      std::string ipc_path =
          "/tmp/chimaera_" + std::to_string(port) + ".ipc";
      try {
        zmq_transport_ = hshm::lbm::TransportFactory::Get(
            ipc_path, hshm::lbm::TransportType::kSocket,
            hshm::lbm::TransportMode::kClient, "ipc", 0);
        HLOG(kInfo, "IpcManager: IPC transport connected to {}", ipc_path);
      } catch (const std::exception &e) {
        HLOG(kError,
             "IpcManager::ClientInit: Failed to create IPC transport: {}",
             e.what());
        return false;
      }
    } else {
      // TCP mode: ZMQ DEALER transport
      try {
        zmq_transport_ = hshm::lbm::TransportFactory::Get(
            "127.0.0.1", hshm::lbm::TransportType::kZeroMq,
            hshm::lbm::TransportMode::kClient, "tcp", port + 3);
        HLOG(kInfo, "IpcManager: DEALER transport connected to port {}",
             port + 3);
      } catch (const std::exception &e) {
        HLOG(kError,
             "IpcManager::ClientInit: Failed to create DEALER transport: {}",
             e.what());
        return false;
      }
    }

    zmq_recv_running_.store(true);
    zmq_recv_thread_ = std::thread([this]() { RecvZmqClientThread(); });
  }

  // Wait for local server using lightbeam transport
  if (!WaitForLocalServer()) {
    HLOG(kError, "CRITICAL ERROR: Cannot connect to local server.");
    HLOG(kError, "Client initialization failed. Exiting.");
    zmq_recv_running_.store(false);
    if (zmq_recv_thread_.joinable()) zmq_recv_thread_.join();
    zmq_transport_.reset();
    return false;
  }

  // SHM mode: Attach to main SHM segment and initialize queues
  if (ipc_mode_ == IpcMode::kShm) {
    if (!ClientInitShm()) {
      return false;
    }
    if (!ClientInitQueues()) {
      return false;
    }

    // Create per-process shared memory for client allocations
    auto *config = CHI_CONFIG_MANAGER;
    size_t initial_size =
        config && config->IsValid()
            ? config->GetMemorySegmentSize(kClientDataSegment)
            : hshm::Unit<size_t>::Megabytes(256);  // Default 256MB
    if (!IncreaseClientShm(initial_size)) {
      HLOG(kError,
           "IpcManager::ClientInit: Failed to create per-process shared memory");
      return false;
    }

    // Create SHM lightbeam transports for client-side transport
    shm_send_transport_ = hshm::lbm::TransportFactory::Get(
        "", hshm::lbm::TransportType::kShm,
        hshm::lbm::TransportMode::kClient);
    shm_recv_transport_ = hshm::lbm::TransportFactory::Get(
        "", hshm::lbm::TransportType::kShm,
        hshm::lbm::TransportMode::kServer);
  }

  // Retrieve node ID from shared header and store in this_host_
  if (shared_header_) {
    this_host_.node_id = shared_header_->node_id;
    HLOG(kDebug, "Retrieved node ID from shared memory: 0x{:x}",
         this_host_.node_id);
  } else {
    HLOG(kError, "Warning: Could not access shared header during ClientInit");
    this_host_ = Host();  // Default constructor gives node_id = 0
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
  auto *config = CHI_CONFIG_MANAGER;
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

  // Clear leftover shared memory segments from previous runs
  ClearUserIpcs();

  // Initialize memory segments for server
  if (!ServerInitShm()) {
    return false;
  }

  // Initialize priority queues
  if (!ServerInitQueues()) {
    return false;
  }

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
  // Initialize GPU queues (one ring buffer per GPU)
  if (!ServerInitGpuQueues()) {
    return false;
  }
#endif

  // Identify this host and store node ID in shared header
  if (!IdentifyThisHost()) {
    HLOG(kError, "Warning: Could not identify host, using default node ID");
    this_host_ = Host();  // Default constructor gives node_id = 0
    if (shared_header_) {
      shared_header_->node_id = this_host_.node_id;
    }
  } else {
    // Store the identified host's node ID in shared header
    if (shared_header_) {
      shared_header_->node_id = this_host_.node_id;
    }

    HLOG(kDebug, "Node ID stored in shared memory: 0x{:x}", this_host_.node_id);
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

  // Create lightbeam transports for client task reception
  {
    u32 port = config->GetPort();

    try {
      // TCP ROUTER server on port+3
      client_tcp_transport_ = hshm::lbm::TransportFactory::Get(
          "0.0.0.0", hshm::lbm::TransportType::kZeroMq,
          hshm::lbm::TransportMode::kServer, "tcp", port + 3);
      HLOG(kInfo, "IpcManager: TCP ROUTER transport bound on port {}", port + 3);
    } catch (const std::exception &e) {
      HLOG(kError, "IpcManager::ServerInit: Failed to bind TCP server: {}",
           e.what());
    }

    try {
      // IPC server on Unix domain socket
      std::string ipc_path =
          "/tmp/chimaera_" + std::to_string(port) + ".ipc";
      client_ipc_transport_ = hshm::lbm::TransportFactory::Get(
          ipc_path, hshm::lbm::TransportType::kSocket,
          hshm::lbm::TransportMode::kServer, "ipc", 0);
      HLOG(kInfo, "IpcManager: IPC lightbeam server bound on {}", ipc_path);
    } catch (const std::exception &e) {
      HLOG(kError, "IpcManager::ServerInit: Failed to bind IPC server: {}",
           e.what());
    }
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

  // Stop recv thread
  if (zmq_recv_running_.load()) {
    zmq_recv_running_.store(false);
    if (zmq_recv_thread_.joinable()) {
      zmq_recv_thread_.join();
    }
  }

  // Clean up lightbeam transport objects
  zmq_transport_.reset();

  // Clients should not destroy shared resources
}

void IpcManager::ServerFinalize() {
  if (!is_initialized_) {
    return;
  }

  // Cleanup servers
  local_transport_.reset();
  main_transport_.reset();

  // Clean up lightbeam client transport objects
  client_tcp_transport_.reset();
  client_ipc_transport_.reset();

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

void IpcManager::SetNumSchedQueues(u32 num_sched_queues) {
  if (!shared_header_) {
    HLOG(kError, "IpcManager::SetNumSchedQueues: shared_header_ is null");
    return;
  }
  shared_header_->num_sched_queues = num_sched_queues;
  HLOG(kInfo, "IpcManager: Updated num_sched_queues to {}", num_sched_queues);
}

void IpcManager::AwakenWorker(TaskLane *lane) {
  if (!lane) {
    HLOG(kWarning, "AwakenWorker: lane is null");
    return;
  }

  // Always send signal to ensure worker wakes up
  // The worker may transition from active->inactive between our check and
  // signal send Sending signal when already active is safe - it's a no-op if
  // worker is processing
  pid_t tid = lane->GetTid();
  if (tid > 0) {
    // Get runtime PID from shared header (client's getpid() won't work for
    // runtime threads)
    pid_t runtime_pid = shared_header_ ? shared_header_->runtime_pid : getpid();

    // Send SIGUSR1 to the worker thread in the runtime process
    int result = hshm::lbm::EventManager::Signal(runtime_pid, tid);
    if (result == 0) {
      HLOG(kDebug,
           "AwakenWorker: Sent SIGUSR1 to runtime_pid={}, tid={} (active={}) - "
           "SUCCESS",
           runtime_pid, tid, lane->IsActive());
    } else {
      HLOG(kError,
           "AwakenWorker: Failed to send SIGUSR1 to runtime_pid={}, tid={} "
           "(active={}) - errno={}",
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

    // Use calculated or explicit main_segment_size
    size_t main_segment_size = config->CalculateMainSegmentSize();

    HLOG(kInfo, "Initializing main shared memory segment: {} bytes ({} MB)",
         main_segment_size, main_segment_size / (1024 * 1024));

    // Initialize main backend with custom header size
    if (!main_backend_.shm_init(main_allocator_id_,
                                hshm::Unit<size_t>::Bytes(main_segment_size),
                                main_segment_name)) {
      return false;
    }

    // Create main allocator using backend's MakeAlloc method
    main_allocator_ = main_backend_.MakeAlloc<CHI_MAIN_ALLOC_T>();
    if (!main_allocator_) {
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
    shared_header_ = main_backend_.template GetSharedHeader<IpcSharedHeader>();

    if (!shared_header_) {
      return false;
    }

    // Initialize shared header
    shared_header_->node_id = 0;  // Will be set after host identification
    shared_header_->runtime_pid =
        getpid();  // Store runtime's PID for client tgkill

    // Get worker counts from ConfigManager
    ConfigManager *config = CHI_CONFIG_MANAGER;
    u32 thread_count = config->GetNumThreads();
    // Note: Last worker serves dual roles as both task worker and network
    // worker
    u32 total_workers = thread_count;

    // Store worker count and scheduling queue count
    shared_header_->num_workers = total_workers;
    shared_header_->num_sched_queues = thread_count;

    // Get configured queue depth (no longer hardcoded)
    u32 queue_depth = config->GetQueueDepth();

    HLOG(kInfo,
         "Initializing {} worker queues with depth {} (last worker serves dual "
         "role)",
         total_workers, queue_depth);

    // Initialize TaskQueue in shared header
    // Number of lanes equals total worker count
    new (&shared_header_->worker_queues) TaskQueue(
        main_allocator_,
        total_workers,  // num_lanes equals total worker count
        2,  // num_priorities (2 priorities: 0=normal, 1=resumed tasks)
        queue_depth);  // Use configured depth instead of hardcoded 1024

    // Create FullPtr reference to the shared TaskQueue
    worker_queues_ = hipc::FullPtr<TaskQueue>(main_allocator_,
                                              &shared_header_->worker_queues);

    // Initialize network queue for send operations
    // One lane with four priorities (SendIn, SendOut, ClientSendTcp, ClientSendIpc)
    net_queue_ = main_allocator_->NewObj<NetQueue>(
        main_allocator_,
        1,             // num_lanes: single lane for network operations
        4,             // num_priorities: 0=SendIn, 1=SendOut, 2=ClientSendTcp, 3=ClientSendIpc
        queue_depth);  // Use configured depth instead of hardcoded 1024

    return !worker_queues_.IsNull() && !net_queue_.IsNull();
  } catch (const std::exception &e) {
    return false;
  }
}

#if HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM
bool IpcManager::ServerInitGpuQueues() {
  // Get number of GPUs on the system
  int num_gpus = hshm::GpuApi::GetDeviceCount();
  if (num_gpus == 0) {
    HLOG(kDebug, "No GPUs detected, skipping GPU queue initialization");
    return true;  // Not an error - just no GPUs available
  }

  HLOG(kInfo, "Initializing {} GPU queue(s) with pinned host memory", num_gpus);

  try {
    // Get configured queue depth
    ConfigManager *config = CHI_CONFIG_MANAGER;
    u32 queue_depth = config->GetQueueDepth();

    // Get configured GPU segment size (default to 64MB per GPU)
    size_t gpu_segment_size = config && config->IsValid()
        ? config->GetMemorySegmentSize("gpu_segment")
        : hshm::Unit<size_t>::Megabytes(64);

    // Reserve space for GPU backends and queues
    gpu_backends_.reserve(num_gpus);
    gpu_queues_.reserve(num_gpus);

    // Create one segment and ring buffer per GPU
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
      // Create unique URL for this GPU's shared memory
      std::string gpu_url = "/chi_gpu_queue_" + std::to_string(gpu_id);

      // Create GPU backend ID
      hipc::MemoryBackendId backend_id(1000 + gpu_id, 0);  // Use high IDs for GPU backends

      // Create GpuShmMmap backend (pinned host memory, GPU-accessible)
      auto gpu_backend = std::make_unique<hipc::GpuShmMmap>();
      if (!gpu_backend->shm_init(backend_id, gpu_segment_size, gpu_url, gpu_id)) {
        HLOG(kError, "Failed to initialize GPU backend for GPU {}", gpu_id);
        return false;
      }

      // Create allocator for this GPU segment
      auto *gpu_allocator = gpu_backend->template MakeAlloc<CHI_MAIN_ALLOC_T>(
          gpu_backend->data_capacity_);
      if (!gpu_allocator) {
        HLOG(kError, "Failed to create allocator for GPU {}", gpu_id);
        return false;
      }

      // Create TaskQueue in GPU segment (one ring buffer)
      // Single lane for now, 2 priorities (normal and resumed)
      hipc::FullPtr<TaskQueue> gpu_queue = gpu_allocator->template NewObj<TaskQueue>(
          gpu_allocator,
          1,             // num_lanes: single lane per GPU
          2,             // num_priorities: normal and resumed
          queue_depth);  // configured depth

      if (gpu_queue.IsNull()) {
        HLOG(kError, "Failed to create TaskQueue for GPU {}", gpu_id);
        return false;
      }

      HLOG(kInfo, "GPU {} queue initialized: segment_size={}, queue_depth={}",
           gpu_id, gpu_segment_size, queue_depth);

      // Store backend and queue
      gpu_backends_.push_back(std::move(gpu_backend));
      gpu_queues_.push_back(gpu_queue);
    }

    return true;
  } catch (const std::exception &e) {
    HLOG(kError, "Exception during GPU queue initialization: {}", e.what());
    return false;
  }
}
#endif

bool IpcManager::ClientInitQueues() {
  if (!main_allocator_) {
    return false;
  }

  try {
    // Get the custom header from the backend
    shared_header_ = main_backend_.template GetSharedHeader<IpcSharedHeader>();

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
    u32 port = config->GetPort() + 1;  // Use ZMQ port + 1 for local server

    local_transport_ = hshm::lbm::TransportFactory::Get(
        addr, hshm::lbm::TransportType::kZeroMq,
        hshm::lbm::TransportMode::kServer, protocol, port);

    if (local_transport_ != nullptr) {
      HLOG(kSuccess, "Successfully started local server at {}:{}", addr, port);
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
  // Read environment variables for wait configuration
  const char *wait_env = std::getenv("CHI_WAIT_SERVER");
  if (wait_env != nullptr) {
    wait_server_timeout_ = static_cast<u32>(std::atoi(wait_env));
  }

  HLOG(kInfo, "Waiting for runtime via lightbeam (timeout={}s)",
       wait_server_timeout_);

  // Send a ClientConnectTask via the lightbeam transport
  auto task = NewTask<chimaera::admin::ClientConnectTask>(
      CreateTaskId(), kAdminPoolId, PoolQuery::Local());
  auto future = SendZmq(task, ipc_mode_);

  // Wait for response with timeout
  if (!future.Wait(static_cast<float>(wait_server_timeout_))) {
    HLOG(kError, "Timeout waiting for runtime after {} seconds",
         wait_server_timeout_);
    HLOG(kError, "This usually means:");
    HLOG(kError, "1. Chimaera runtime is not running");
    HLOG(kError, "2. Runtime failed to start");
    HLOG(kError, "3. Network connectivity issues");
    return false;
  }

  if (task->response_ == 0) {
    HLOG(kInfo, "Successfully connected to runtime");
    return true;
  }

  HLOG(kError, "Runtime responded with error code: {}", task->response_);
  return false;
}

void IpcManager::SetNodeId(const std::string &hostname) {
  (void)hostname;  // Unused parameter
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
      HLOG(kInfo, "  Hostfile[{}]: {} -> Node ID: {}", offset, host_ips[offset],
           node_id);
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

bool IpcManager::IsAlive(u64 node_id) const {
  auto it = hostfile_map_.find(node_id);
  if (it == hostfile_map_.end()) return false;
  return it->second.state == NodeState::kAlive;
}

void IpcManager::SetDead(u64 node_id) {
  auto it = hostfile_map_.find(node_id);
  if (it == hostfile_map_.end()) return;
  if (it->second.state == NodeState::kDead) return;  // Already dead

  SetNodeState(node_id, NodeState::kDead);

  // Record dead-node entry for retry tracking
  DeadNodeEntry entry;
  entry.node_id = node_id;
  entry.detected_at = std::chrono::steady_clock::now();
  dead_nodes_.push_back(entry);

  // Remove cached client connections to the dead node
  {
    std::lock_guard<std::mutex> lock(client_pool_mutex_);
    auto *config_manager = CHI_CONFIG_MANAGER;
    int port = static_cast<int>(config_manager->GetPort());
    std::string key = it->second.ip_address + ":" + std::to_string(port);
    client_pool_.erase(key);
  }

  HLOG(kWarning, "IpcManager: Node {} ({}) marked as DEAD",
       node_id, it->second.ip_address);
}

void IpcManager::SetAlive(u64 node_id) {
  auto it = hostfile_map_.find(node_id);
  if (it == hostfile_map_.end()) return;
  if (it->second.state == NodeState::kAlive) return;  // Already alive

  SetNodeState(node_id, NodeState::kAlive);

  // Remove from dead_nodes_ list
  dead_nodes_.erase(
      std::remove_if(dead_nodes_.begin(), dead_nodes_.end(),
                     [node_id](const DeadNodeEntry &e) {
                       return e.node_id == node_id;
                     }),
      dead_nodes_.end());

  HLOG(kInfo, "IpcManager: Node {} ({}) marked as ALIVE",
       node_id, it->second.ip_address);
}

NodeState IpcManager::GetNodeState(u64 node_id) const {
  auto it = hostfile_map_.find(node_id);
  if (it == hostfile_map_.end()) return NodeState::kDead;
  return it->second.state;
}

void IpcManager::SetNodeState(u64 node_id, NodeState new_state) {
  auto it = hostfile_map_.find(node_id);
  if (it == hostfile_map_.end()) return;
  it->second.state = new_state;
  it->second.state_changed_at = std::chrono::steady_clock::now();
  hosts_cache_valid_ = false;
}

void IpcManager::SetSelfFenced(bool fenced) {
  self_fenced_ = fenced;
}

u64 IpcManager::GetLeaderNodeId() const {
  u64 leader = std::numeric_limits<u64>::max();
  for (const auto& [id, host] : hostfile_map_) {
    if (host.state == NodeState::kAlive && host.node_id < leader) {
      leader = host.node_id;
    }
  }
  return (leader == std::numeric_limits<u64>::max()) ? 0 : leader;
}

bool IpcManager::IsLeader() const {
  return GetNodeId() == GetLeaderNodeId();
}

u64 IpcManager::AddNode(const std::string& ip_address, u32 port) {
  (void)port;  // Port stored elsewhere (ConfigManager) for now

  // Check if node already exists
  for (const auto& pair : hostfile_map_) {
    if (pair.second.ip_address == ip_address) {
      HLOG(kInfo, "AddNode: Node {} already registered as node_id={}",
           ip_address, pair.first);
      return pair.first;
    }
  }

  // Assign next node ID (linear offset)
  u64 new_node_id = static_cast<u64>(hostfile_map_.size());
  Host host(ip_address, new_node_id);
  hostfile_map_[new_node_id] = host;
  hosts_cache_valid_ = false;

  HLOG(kInfo, "AddNode: Registered {} as node_id={}", ip_address, new_node_id);
  return new_node_id;
}

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

    main_transport_ = hshm::lbm::TransportFactory::Get(
        hostname, hshm::lbm::TransportType::kZeroMq,
        hshm::lbm::TransportMode::kServer, protocol, port);

    if (!main_transport_) {
      HLOG(kDebug,
           "Failed to create main server on {}:{} - server creation returned "
           "null",
           hostname, port);
      return false;
    }

    HLOG(kDebug, "Main server successfully bound to {}:{}", hostname, port);

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

hshm::lbm::Transport *IpcManager::GetMainTransport() const {
  return main_transport_.get();
}

hshm::lbm::Transport *IpcManager::GetClientTransport(IpcMode mode) const {
  if (mode == IpcMode::kTcp) return client_tcp_transport_.get();
  if (mode == IpcMode::kIpc) return client_ipc_transport_.get();
  return nullptr;
}


const Host &IpcManager::GetThisHost() const { return this_host_; }

FullPtr<char> IpcManager::AllocateBuffer(size_t size) {
#if HSHM_IS_HOST
  // HOST-ONLY PATH: The device implementation is in ipc_manager.h

  // RUNTIME PATH: Use private memory (HSHM_MALLOC) — runtime never uses
  // per-process shared memory segments
  if (CHI_CHIMAERA_MANAGER && CHI_CHIMAERA_MANAGER->IsRuntime()) {
    // Use HSHM_MALLOC allocator for private memory allocation
    FullPtr<char> buffer = HSHM_MALLOC->AllocateObjs<char>(size);
    if (buffer.IsNull()) {
      HLOG(kError, "AllocateBuffer: HSHM_MALLOC failed for {} bytes", size);
    }
    return buffer;
  }

  // CLIENT TCP/IPC PATH: Use private memory (no shared memory needed)
  if (ipc_mode_ != IpcMode::kShm) {
    FullPtr<char> buffer = HSHM_MALLOC->AllocateObjs<char>(size);
    if (buffer.IsNull()) {
      HLOG(kError, "AllocateBuffer: HSHM_MALLOC failed for {} bytes (client ZMQ mode)", size);
    }
    return buffer;
  }

  // CLIENT SHM PATH: Use per-process shared memory allocation strategy
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
  size_t new_size = static_cast<size_t>((size + kShmMetadataOverhead) *
                                        kShmAllocationMultiplier);
  if (!IncreaseClientShm(new_size)) {
    HLOG(kError, "AllocateBuffer: Failed to increase memory for {} bytes",
         size);
    return FullPtr<char>::GetNull();
  }

  // 4. Retry allocation from the newly created allocator (last_alloc_)
  if (last_alloc_ != nullptr) {
    FullPtr<char> buffer = last_alloc_->AllocateObjs<char>(size);
    if (!buffer.IsNull()) {
      return buffer;
    }
  }

  HLOG(kError,
       "AllocateBuffer: Failed to allocate {} bytes even after increasing "
       "memory",
       size);
  return FullPtr<char>::GetNull();
#else
  // GPU PATH: Implementation is in ipc_manager.h as inline function
  return FullPtr<char>::GetNull();
#endif  // HSHM_IS_HOST
}

void IpcManager::FreeBuffer(FullPtr<char> buffer_ptr) {
#if HSHM_IS_HOST
  // HOST PATH: Check various allocators
  if (buffer_ptr.IsNull()) {
    return;
  }

  // Check if allocator ID is null (private memory allocated with HSHM_MALLOC)
  if (buffer_ptr.shm_.alloc_id_ == hipc::AllocatorId::GetNull()) {
    // Private memory - use HSHM_MALLOC->Free() for RUNTIME-allocated buffers
    // In RUNTIME mode, AllocateBuffer uses HSHM_MALLOC which adds MallocPage
    // header
    HSHM_MALLOC->Free(buffer_ptr);
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
#else
  // GPU PATH: Implementation is in ipc_manager.h as inline function
#endif  // HSHM_IS_HOST
}

hshm::lbm::Transport *IpcManager::GetOrCreateClient(const std::string &addr,
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
  auto transport = hshm::lbm::TransportFactory::Get(
      addr, hshm::lbm::TransportType::kZeroMq,
      hshm::lbm::TransportMode::kClient, "tcp", port);

  if (!transport) {
    HLOG(kError, "[ClientPool] Failed to create client for {}", key);
    return nullptr;
  }

  // Store in pool and return raw pointer
  hshm::lbm::Transport *raw_ptr = transport.get();
  client_pool_[key] = std::move(transport);

  HLOG(kInfo, "[ClientPool] Connection established to {}", key);
  return raw_ptr;
}

void IpcManager::ClearClientPool() {
  std::lock_guard<std::mutex> lock(client_pool_mutex_);
  HLOG(kInfo, "[ClientPool] Clearing {} persistent connections",
       client_pool_.size());
  client_pool_.clear();
}

void IpcManager::EnqueueNetTask(Future<Task> future,
                                NetQueuePriority priority) {
  if (net_queue_.IsNull()) {
    HLOG(kError, "EnqueueNetTask: net_queue_ is null");
    return;
  }

  // Get lane 0 (single lane) with the specified priority
  u32 priority_idx = static_cast<u32>(priority);
  auto &lane = net_queue_->GetLane(0, priority_idx);
  lane.Push(future);

  HLOG(kDebug, "EnqueueNetTask: Enqueued task to priority {} queue",
       priority_idx);
}

bool IpcManager::TryPopNetTask(NetQueuePriority priority,
                               Future<Task> &future) {
  if (net_queue_.IsNull()) {
    return false;
  }

  // Get lane 0 (single lane) with the specified priority
  u32 priority_idx = static_cast<u32>(priority);
  auto &lane = net_queue_->GetLane(0, priority_idx);

  if (lane.Pop(future)) {
    return true;
  }

  return false;
}

//==============================================================================
// Per-Process Shared Memory Management
//==============================================================================

bool IpcManager::IncreaseClientShm(size_t size) {
  HLOG(kDebug, "IncreaseClientShm CALLED: size={}", size);
  std::lock_guard<std::mutex> lock(shm_mutex_);
  // Acquire writer lock on allocator_map_lock_ during memory increase
  // This ensures exclusive access to the allocator_map_ structures
  allocator_map_lock_.WriteLock(0);

  pid_t pid = getpid();
  u32 index = shm_count_.fetch_add(1, std::memory_order_relaxed);

  // Create shared memory name: chimaera_{pid}_{index}
  std::string shm_name =
      "chimaera_" + std::to_string(pid) + "_" + std::to_string(index);

  // Add 32MB metadata overhead
  size_t total_size = size + kShmMetadataOverhead;

  HLOG(
      kInfo,
      "IpcManager::IncreaseClientShm: Creating {} with size {} ({} + {} overhead)",
      shm_name, total_size, size, kShmMetadataOverhead);

  try {
    // Create the shared memory backend
    auto backend = std::make_unique<hipc::PosixShmMmap>();

    // Create allocator ID: major = pid, minor = index
    hipc::AllocatorId alloc_id(static_cast<u32>(pid), index);

    // Initialize shared memory using backend's shm_init method
    if (!backend->shm_init(alloc_id, hshm::Unit<size_t>::Bytes(total_size),
                           shm_name)) {
      HLOG(kError, "IpcManager::IncreaseClientShm: Failed to create shm for {}",
           shm_name);
      shm_count_.fetch_sub(1, std::memory_order_relaxed);
      allocator_map_lock_
          .WriteUnlock();  // CRITICAL: Release lock before returning
      return false;
    }

    // Create allocator using backend's MakeAlloc method
    hipc::MultiProcessAllocator *allocator =
        backend->MakeAlloc<hipc::MultiProcessAllocator>();

    if (allocator == nullptr) {
      HLOG(kError,
           "IpcManager::IncreaseClientShm: Failed to create allocator for {}",
           shm_name);
      shm_count_.fetch_sub(1, std::memory_order_relaxed);
      allocator_map_lock_
          .WriteUnlock();  // CRITICAL: Release lock before returning
      return false;
    }

    // Add to our tracking structures
    u64 alloc_key = (static_cast<u64>(alloc_id.major_) << 32) |
                    static_cast<u64>(alloc_id.minor_);
    alloc_map_[alloc_key] = allocator;
    alloc_vector_.push_back(allocator);
    client_backends_.push_back(std::move(backend));
    last_alloc_ = allocator;

    HLOG(kInfo,
         "IpcManager::IncreaseClientShm: Created allocator {} with ID ({}.{})",
         shm_name, alloc_id.major_, alloc_id.minor_);

    // Release the lock before returning
    allocator_map_lock_.WriteUnlock();

    // Tell the runtime server to attach to this new shared memory segment.
    // Use kAdminPoolId directly (not admin_client->pool_id_) because
    // the admin client may not be initialized yet during ClientInit.
    auto reg_task = NewTask<chimaera::admin::RegisterMemoryTask>(
        chi::CreateTaskId(), chi::kAdminPoolId,
        chi::PoolQuery::Local(), alloc_id);
    SendZmq(reg_task, IpcMode::kTcp).Wait();

    return true;

  } catch (const std::exception &e) {
    allocator_map_lock_.WriteUnlock();
    HLOG(kError, "IpcManager::IncreaseClientShm: Exception creating {}: {}",
         shm_name, e.what());
    shm_count_.fetch_sub(1, std::memory_order_relaxed);
    return false;
  }
}

bool IpcManager::RegisterMemory(const hipc::AllocatorId &alloc_id) {
  HLOG(kDebug, "RegisterMemory CALLED: alloc_id=({}.{})", alloc_id.major_,
       alloc_id.minor_);
  std::lock_guard<std::mutex> lock(shm_mutex_);
  // Acquire writer lock on allocator_map_lock_ during memory registration
  allocator_map_lock_.WriteLock(0);

  // Derive shm_name from alloc_id: chimaera_{pid}_{index}
  pid_t owner_pid = static_cast<pid_t>(alloc_id.major_);
  u32 shm_index = alloc_id.minor_;
  std::string shm_name =
      "chimaera_" + std::to_string(owner_pid) + "_" + std::to_string(shm_index);

  HLOG(kInfo, "IpcManager::RegisterMemory: Registering {} from pid {}",
       shm_name, owner_pid);

  // Check if already registered
  u64 alloc_key = (static_cast<u64>(alloc_id.major_) << 32) |
                  static_cast<u64>(alloc_id.minor_);
  if (alloc_map_.find(alloc_key) != alloc_map_.end()) {
    HLOG(kInfo, "IpcManager::RegisterMemory: {} already registered, skipping",
         shm_name);
    allocator_map_lock_.WriteUnlock();
    return true;  // Already registered
  }

  try {
    // Attach to the shared memory backend (already created by client)
    auto backend = std::make_unique<hipc::PosixShmMmap>();
    if (!backend->shm_attach(shm_name)) {
      HLOG(kError, "IpcManager::RegisterMemory: Failed to attach to shm {}",
           shm_name);
      allocator_map_lock_
          .WriteUnlock();  // CRITICAL: Release lock before returning
      return false;
    }

    // Attach to the existing allocator in the backend
    hipc::MultiProcessAllocator *allocator =
        backend->AttachAlloc<hipc::MultiProcessAllocator>();

    if (allocator == nullptr) {
      HLOG(kError,
           "IpcManager::RegisterMemory: Failed to attach allocator for {}",
           shm_name);
      allocator_map_lock_
          .WriteUnlock();  // CRITICAL: Release lock before returning
      return false;
    }

    // Add to our tracking structures
    alloc_map_[alloc_key] = allocator;
    // Note: Don't add to alloc_vector_ since this is not our memory
    // (we don't allocate from it, just need to resolve ShmPtrs)
    client_backends_.push_back(std::move(backend));

    HLOG(kInfo, "IpcManager::RegisterMemory: Successfully registered {}",
         shm_name);

    // Release the lock before returning
    allocator_map_lock_.WriteUnlock();

    return true;

  } catch (const std::exception &e) {
    allocator_map_lock_.WriteUnlock();
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
  std::string shm_name =
      "chimaera_" + std::to_string(pid) + "_" + std::to_string(index);

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
  HLOG(kDebug, "WreapDeadIpcs CALLED");
  std::lock_guard<std::mutex> lock(shm_mutex_);
  // Acquire writer lock on allocator_map_lock_ during reaping
  allocator_map_lock_.WriteLock(0);

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
      HLOG(kInfo,
           "WreapDeadIpcs: Process {} is dead, marking allocator ({}.{}) for "
           "removal",
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
    std::string shm_name = "chimaera_" + std::to_string(alloc_id.major_) + "_" +
                           std::to_string(alloc_id.minor_);

    // Find and destroy the corresponding backend
    for (auto backend_it = client_backends_.begin();
         backend_it != client_backends_.end(); ++backend_it) {
      if (*backend_it && (*backend_it)->header_ &&
          (*backend_it)->header_->id_.major_ == alloc_id.major_ &&
          (*backend_it)->header_->id_.minor_ == alloc_id.minor_) {
        // Destroy the shared memory
        HLOG(kInfo,
             "WreapDeadIpcs: Destroying shared memory {} for allocator ({}.{})",
             shm_name, alloc_id.major_, alloc_id.minor_);
        (*backend_it)->shm_destroy();
        client_backends_.erase(backend_it);
        break;
      }
    }

    // Remove from alloc_vector_ if present
    auto vec_it =
        std::find(alloc_vector_.begin(), alloc_vector_.end(), allocator);
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
    HLOG(kInfo,
         "WreapDeadIpcs: Reaped {} shared memory segments from dead processes",
         reaped_count);
  }

  // Release the lock before returning
  allocator_map_lock_.WriteUnlock();

  return reaped_count;
}

size_t IpcManager::WreapAllIpcs() {
  HLOG(kDebug, "WreapAllIpcs CALLED");
  std::lock_guard<std::mutex> lock(shm_mutex_);
  // Acquire writer lock on allocator_map_lock_ during cleanup
  allocator_map_lock_.WriteLock(0);

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
    std::string shm_name = "chimaera_" + std::to_string(alloc_id.major_) + "_" +
                           std::to_string(alloc_id.minor_);

    // Find and destroy the corresponding backend
    for (auto backend_it = client_backends_.begin();
         backend_it != client_backends_.end(); ++backend_it) {
      if (*backend_it && (*backend_it)->header_ &&
          (*backend_it)->header_->id_.major_ == alloc_id.major_ &&
          (*backend_it)->header_->id_.minor_ == alloc_id.minor_) {
        // Destroy the shared memory
        HLOG(kInfo,
             "WreapAllIpcs: Destroying shared memory {} for allocator ({}.{})",
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

  // Release the lock before returning
  allocator_map_lock_.WriteUnlock();

  return reaped_count;
}

size_t IpcManager::ClearUserIpcs() {
  size_t removed_count = 0;
  const char *memfd_dir = "/tmp/chimaera_memfd";
  const char *prefix = "chimaera_";
  size_t prefix_len = strlen(prefix);

  // Open memfd symlink directory
  DIR *dir = opendir(memfd_dir);
  if (dir == nullptr) {
    // Directory may not exist yet, that's fine
    return 0;
  }

  // Iterate through directory entries
  struct dirent *entry;
  while ((entry = readdir(dir)) != nullptr) {
    // Skip "." and ".."
    if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
      continue;
    }

    // Check if filename starts with "chimaera_"
    if (strncmp(entry->d_name, prefix, prefix_len) != 0) {
      continue;
    }

    // Construct full path and remove the symlink
    std::string full_path = std::string(memfd_dir) + "/" + entry->d_name;
    if (unlink(full_path.c_str()) == 0) {
      HLOG(kDebug, "ClearUserIpcs: Removed memfd symlink: {}",
           entry->d_name);
      removed_count++;
    } else {
      if (errno != EACCES && errno != EPERM && errno != ENOENT) {
        HLOG(kDebug, "ClearUserIpcs: Could not remove {} ({}): {}",
             entry->d_name, errno, strerror(errno));
      }
    }
  }

  closedir(dir);

  if (removed_count > 0) {
    HLOG(kInfo,
         "ClearUserIpcs: Removed {} memfd symlinks from previous runs",
         removed_count);
  }

  return removed_count;
}

void IpcManager::SetIsClientThread(bool is_client_thread) {
  // Create TLS key if not already created
  HSHM_THREAD_MODEL->CreateTls<bool>(chi_is_client_thread_key_, nullptr);

  // Set the flag for the current thread
  bool *flag = new bool(is_client_thread);
  HSHM_THREAD_MODEL->SetTls(chi_is_client_thread_key_, flag);

  HLOG(kDebug, "SetIsClientThread: Set to {} for current thread",
       is_client_thread);
}

bool IpcManager::GetIsClientThread() const {
  // Get the TLS value, defaulting to false if not set
  bool *flag = HSHM_THREAD_MODEL->GetTls<bool>(chi_is_client_thread_key_);
  if (!flag) {
    return false;
  }
  return *flag;
}

//==============================================================================
// GPU Memory Management
//==============================================================================

//==============================================================================
// ZMQ Transport Methods
//==============================================================================

void IpcManager::RecvZmqClientThread() {
  // Client-side thread: polls for completed task responses from the server
  // DEALER transport receives responses on the same socket used for sending
  if (!zmq_transport_) {
    HLOG(kError, "RecvZmqClientThread: No DEALER transport");
    return;
  }

  // Set up EventManager for ZMQ transport polling
  hshm::lbm::EventManager em;
  zmq_transport_->RegisterEventManager(em);

  while (zmq_recv_running_.load()) {
    // Drain all available messages first
    bool drained_any = false;
    bool got_message = true;
    while (got_message) {
      got_message = false;
      auto archive = std::make_unique<LoadTaskArchive>();
      auto info = zmq_transport_->Recv(*archive);
      int rc = info.rc;
      if (rc == EAGAIN) break;
      if (rc != 0) {
        zmq_transport_->ClearRecvHandles(*archive);
        if (!zmq_recv_running_.load()) break;
        HLOG(kDebug, "RecvZmqClientThread: Recv returned: {}", rc);
        continue;
      }
      got_message = true;
      drained_any = true;

      // Look up pending future by net_key from task_infos
      if (archive->task_infos_.empty()) {
        HLOG(kError, "RecvZmqClientThread: No task_infos in response");
        continue;
      }
      size_t net_key = archive->task_infos_[0].task_id_.net_key_;

      std::lock_guard<std::mutex> lock(pending_futures_mutex_);
      auto it = pending_zmq_futures_.find(net_key);
      if (it == pending_zmq_futures_.end()) {
        HLOG(kError, "RecvZmqClientThread: No pending future for net_key {}",
             net_key);
        zmq_transport_->ClearRecvHandles(*archive);
        continue;
      }

      FutureShm *future_shm = it->second;

      // Store the archive for Recv() to pick up
      pending_response_archives_[net_key] = std::move(archive);

      // Memory fence before setting complete
      std::atomic_thread_fence(std::memory_order_release);

      // Signal completion
      future_shm->flags_.SetBits(FutureShm::FUTURE_NEW_DATA |
                                  FutureShm::FUTURE_COMPLETE);

      // Remove from pending futures map
      pending_zmq_futures_.erase(it);
    }

    // Only block on epoll when the drain loop found nothing;
    // if we just processed messages, loop back immediately.
    if (!drained_any) {
      em.Wait(10000);  // 10ms in microseconds
    }
  }
}

void IpcManager::CleanupResponseArchive(size_t net_key) {
  std::lock_guard<std::mutex> lock(pending_futures_mutex_);
  auto it = pending_response_archives_.find(net_key);
  if (it != pending_response_archives_.end()) {
    zmq_transport_->ClearRecvHandles(*(it->second));
    pending_response_archives_.erase(it);
  }
}

bool IpcManager::RegisterAcceleratorMemory(const hipc::MemoryBackend &backend) {
#if !HSHM_ENABLE_CUDA && !HSHM_ENABLE_ROCM
  HLOG(kError,
       "RegisterAcceleratorMemory: GPU support not enabled at compile time");
  return false;
#else
  // Store the GPU backend for later use
  // This is called from GPU kernels where we have limited capability
  // The actual allocation happens in CHIMAERA_GPU_INIT macro where
  // each thread gets its own ArenaAllocator instance
  gpu_backend_ = backend;
  gpu_backend_initialized_ = true;

  // Note: In GPU kernels, each thread maintains its own ArenaAllocator
  // The macro CHIMAERA_GPU_INIT handles per-thread allocator setup
  // No need to initialize allocators here as they're created per-thread in __shared__ memory

  return true;
#endif
}

}  // namespace chi