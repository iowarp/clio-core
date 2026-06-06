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

#include "clio_runtime/ipc_manager.h"

#include <clio_ctp/lightbeam/transport_factory_impl.h>
#include <zmq.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <set>

#include "clio_runtime/admin.h"
#include "clio_runtime/admin/admin_client.h"
#include "clio_runtime/manager.h"
#include "clio_runtime/config_manager.h"
#include "clio_runtime/container.h"
#include "clio_runtime/local_task_archives.h"
#include "clio_runtime/pool_manager.h"
#include "clio_runtime/scheduler/scheduler_factory.h"
#include "clio_runtime/task_archives.h"

#if CTP_ENABLE_CUDA || CTP_ENABLE_ROCM
#include <clio_ctp/util/gpu_api.h>
#endif

// Global pointer variable definition for IPC manager singleton
CLIO_RUN_DEFINE_GLOBAL_PTR_VAR_CC(chi::IpcManager, g_ipc_manager);

#include <clio_runtime/device_memcpy.h>

namespace clio::run {

// Definitions of the device-aware memcpy + IsDevicePointer hooks
// declared in chimaera/device_memcpy.h. ServerInitGpuQueuesSycl (or
// its CUDA/ROCm equivalent) installs function pointers here at
// server-init time so the bdev runtime — built without -fsycl — can
// route memcpys involving device USM through the GPU runtime, and
// stage through host buffers only when the data is actually on the
// device.
CLIO_RUN_API std::atomic<DeviceAwareMemcpyFn> g_device_aware_memcpy{nullptr};
CLIO_RUN_API std::atomic<IsDevicePointerFn> g_is_device_pointer{nullptr};

}  // namespace clio::run

namespace clio::run {

namespace {

// Issue #482: libzmq's TCP-loopback engine on macOS fails to route ROUTER
// replies back to a connected DEALER (zmq_send -> EHOSTUNREACH) even though the
// ZMTP handshake succeeds, which blocks every same-host client that reaches the
// runtime over the local client ROUTER (the port+3 endpoint). libzmq's ipc://
// (unix-domain) transport carries the identical ROUTER/DEALER identity routing
// but over a different engine that is not affected. So run the LOCAL
// client<->runtime ROUTER/DEALER over ipc:// there. The cross-node ROUTER (the
// main port) is untouched, so multi-node TCP is unaffected; on Linux/Windows
// nothing changes. Override the platform default with CLIO_ZMQ_LOCAL_IPC=0/1.
inline bool UseLocalZmqIpc() {
  if (const char *env = chi::env::GetCompat("ZMQ_LOCAL_IPC")) {
    return *env != '\0' && std::strcmp(env, "0") != 0;
  }
#ifdef __APPLE__
  return true;
#else
  return false;
#endif
}

// Unix-domain socket path for the local ZMQ ROUTER/DEALER. Distinct from the
// non-ZMQ SocketTransport endpoint (chimaera_<port>.ipc) so the two local
// servers never collide.
inline std::string LocalZmqIpcPath(u32 port) {
  return ctp::SystemInfo::GetMemfdPath(
      "chimaera_zmq_" + std::to_string(port + 3) + ".ipc");
}

}  // namespace

// Host struct methods

// IpcManager methods

// Constructor and destructor removed - handled by CTP singleton pattern

bool IpcManager::ClientInit() {
  HLOG(kDebug, "IpcManager::ClientInit");
  if (is_initialized_) {
    return true;
  }

  // Parse CHI_IPC_MODE environment variable (default: TCP)
  const char *ipc_mode_env = chi::env::GetCompat("IPC_MODE");
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
       ipc_mode_ == IpcMode::kShm   ? "SHM"
       : ipc_mode_ == IpcMode::kIpc ? "IPC"
                                    : "TCP");

  // Parse retry timeout environment variable
  // Semantics: 0 = fail immediately, -1 = wait forever, >0 = timeout in seconds
  const char *retry_env = chi::env::GetCompat("CLIENT_RETRY_TIMEOUT");
  if (retry_env) {
    client_retry_timeout_ = static_cast<float>(std::atof(retry_env));
  }
  HLOG(kInfo, "IpcManager::ClientInit: retry_timeout = {}s",
       client_retry_timeout_);

  // Parse CHI_CLIENT_TRY_NEW_SERVERS environment variable
  const char *try_new_env = chi::env::GetCompat("CLIENT_TRY_NEW_SERVERS");
  if (try_new_env) {
    client_try_new_servers_ = std::atoi(try_new_env);
  }
  HLOG(kInfo, "IpcManager::ClientInit: try_new_servers = {}",
       client_try_new_servers_);

  // Load hostfile so Phase 2 failover has hosts to try
  if (client_try_new_servers_ > 0) {
    if (LoadHostfile()) {
      HLOG(kInfo, "IpcManager::ClientInit: Loaded {} hosts from hostfile",
           hostfile_map_.size());
    } else {
      HLOG(kWarning, "IpcManager::ClientInit: Failed to load hostfile, "
           "Phase 2 failover will be disabled");
    }
  }

  // Create lightbeam transport for client-server communication
  {
    auto *config = CLIO_CONFIG_MANAGER;
    u32 port = config->GetPort();

    if (ipc_mode_ == IpcMode::kIpc) {
      // IPC mode: Unix domain socket transport
      std::string ipc_path =
          ctp::SystemInfo::GetMemfdPath("chimaera_" + std::to_string(port) + ".ipc");
      try {
        zmq_transport_ = ctp::lbm::TransportFactory::Get(
            ipc_path, ctp::lbm::TransportType::kSocket,
            ctp::lbm::TransportMode::kClient, "ipc", 0);
        HLOG(kInfo, "IpcManager: IPC transport connected to {}", ipc_path);
      } catch (const std::exception &e) {
        HLOG(kError,
             "IpcManager::ClientInit: Failed to create IPC transport: {}",
             e.what());
        return false;
      }
    } else if (UseLocalZmqIpc()) {
      // macOS (issue #482): keep the ZMQ DEALER but carry it over a local
      // ipc:// unix socket instead of TCP loopback, which libzmq cannot route
      // replies over on macOS. Identity routing is unchanged.
      try {
        ctp::SystemInfo::EnsureMemfdDir();
        std::string zmq_ipc = LocalZmqIpcPath(port);
        zmq_transport_ = ctp::lbm::TransportFactory::Get(
            zmq_ipc, ctp::lbm::TransportType::kZeroMq,
            ctp::lbm::TransportMode::kClient, "ipc", 0);
        HLOG(kInfo, "IpcManager: DEALER transport connected over ipc {}",
             zmq_ipc);
      } catch (const std::exception &e) {
        HLOG(kError,
             "IpcManager::ClientInit: Failed to create ipc DEALER transport: {}",
             e.what());
        return false;
      }
    } else {
      // TCP mode: ZMQ DEALER transport
      try {
        zmq_transport_ = ctp::lbm::TransportFactory::Get(
            config->GetServerAddr(), ctp::lbm::TransportType::kZeroMq,
            ctp::lbm::TransportMode::kClient, "tcp", port + 3);
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

  // Initialize CTP TLS key for task counter before calling WaitForLocalServer,
  // which calls CreateTaskId(). Without the key registered first, GetTls() on
  // the zero-initialized key may return a stale/freed pointer → crash.
  CTP_THREAD_MODEL->CreateTls<TaskCounter>(chi_task_counter_key_, nullptr);
  auto *tls_counter = new TaskCounter();
  CTP_THREAD_MODEL->SetTls(chi_task_counter_key_, tls_counter);

  // Wait for local server using lightbeam transport
  if (!WaitForLocalServer()) {
    HLOG(kError, "CRITICAL ERROR: Cannot connect to local server.");
    HLOG(kError, "Client initialization failed. Exiting.");
    zmq_recv_running_.store(false);
    if (zmq_recv_thread_.joinable()) zmq_recv_thread_.join();
    zmq_transport_.reset();
    return false;
  }

  // Start heartbeat thread for server liveness detection
  server_alive_.store(true);
  heartbeat_running_.store(true);
  heartbeat_thread_ = std::thread([this]() { HeartbeatThread(); });

  // Create TLS key for current worker if not already created.
  // Must happen before any CoRwLock/CoMutex operations (e.g. IncreaseClientShm).
  // Server mode creates it earlier in WorkOrchestrator::Init.
  if (!chi_cur_worker_key_created_) {
    CTP_THREAD_MODEL->CreateTls<Worker>(chi_cur_worker_key_, nullptr);
    chi_cur_worker_key_created_ = true;
  }
  CTP_THREAD_MODEL->SetTls(chi_cur_worker_key_,
                            static_cast<Worker *>(nullptr));

  // SHM mode: Attach to main SHM segment and initialize queues
  if (ipc_mode_ == IpcMode::kShm) {
    if (!ClientInitShm()) {
      return false;
    }
    if (!ClientInitQueues()) {
      return false;
    }

    // Create per-process shared memory for client allocations
    auto *config = CLIO_CONFIG_MANAGER;
    size_t initial_size =
        config && config->IsValid()
            ? config->GetMemorySegmentSize(kClientDataSegment)
            : ctp::Unit<size_t>::Megabytes(256);  // Default 256MB
    if (!IncreaseClientShm(initial_size)) {
      HLOG(
          kError,
          "IpcManager::ClientInit: Failed to create per-process shared memory");
      return false;
    }

    // Create SHM lightbeam transports for client-side transport
    shm_send_transport_ = ctp::lbm::TransportFactory::Get(
        "", ctp::lbm::TransportType::kShm, ctp::lbm::TransportMode::kClient);
    shm_recv_transport_ = ctp::lbm::TransportFactory::Get(
        "", ctp::lbm::TransportType::kShm, ctp::lbm::TransportMode::kServer);
  }

  // Default host until identified
  this_host_ = Host();

  // Task counter TLS key was already created before WaitForLocalServer (above).
  // Do NOT create it again here — doing so leaks the previous pthread key and
  // causes all TLS operations to collide on key 0.

  // Create scheduler using factory
  auto *config = CLIO_CONFIG_MANAGER;
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

  // CLIO_FORCE_NET (legacy CHI_FORCE_NET also honored via GetCompat):
  // when set to anything non-empty, every task whose PoolQuery isn't
  // explicitly Local() is routed via the network path even on a
  // single-node deployment. Used by the bench to stress the ZMQ
  // serialize/send/recv loop without needing a real multi-node
  // setup.  Read once here; IsTaskLocal consults force_net_ on the
  // hot path.
  if (const char *env = chi::env::GetCompat("FORCE_NET")) {
    if (*env != '\0' && std::strcmp(env, "0") != 0) {
      force_net_ = true;
      HLOG(kInfo, "IpcManager: CLIO_FORCE_NET=1 — routing all non-Local "
                  "tasks via network path");
    }
  }

  // Create chi_cur_worker_key_ TLS key early in server path.
  // ServerInitGpuQueues() calls RegisterGpuAllocator() which acquires a
  // CoRwLock, which calls GetCurrentLockOwnerId() → pthread_getspecific().
  // Without a valid TLS key, pthread_getspecific(0) returns a garbage pointer
  // that crashes on dereference.  WorkOrchestrator::Init() normally creates
  // this key, but it runs after ServerInit(), so we create it here first.
  if (!chi_cur_worker_key_created_) {
    CTP_THREAD_MODEL->CreateTls<Worker>(chi_cur_worker_key_, nullptr);
    chi_cur_worker_key_created_ = true;
  }
  CTP_THREAD_MODEL->SetTls(chi_cur_worker_key_, static_cast<Worker *>(nullptr));

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

#if CTP_ENABLE_CUDA || CTP_ENABLE_ROCM
  // CUDA / ROCm slim path: GPU is a pure task producer that pushes onto
  // gpu2cpu_queue. The bootstrap mirrors the SYCL one — pinned host
  // gpu2cpu_queue + gpu2cpu_copy_backend, on-device GpuTaskQueue
  // construction, then install the chi::DeviceAwareMemcpy /
  // IsDevicePointer hooks. Source lives in src/gpu/gpu2cpu_init_hip.cc
  // and is compiled by nvcc/hipcc so the kernel launch syntax resolves.
  {
    ConfigManager *config = CLIO_CONFIG_MANAGER;
    u32 queue_depth = config->GetQueueDepth();
    constexpr size_t kHipClientBackendBytes = 64 * 1024 * 1024;  // 64 MB
    extern bool ChiServerBootstrapHipGpu(IpcManager *self, u32 queue_depth,
                                          size_t backend_bytes);
    if (!ChiServerBootstrapHipGpu(this, queue_depth,
                                   kHipClientBackendBytes)) {
      return false;
    }
  }
#elif CTP_ENABLE_SYCL
  // SYCL backend: same shape as the CUDA/HIP path above. Bootstrap
  // helper lives in chimaera_cxx_gpu (gpu2cpu_init_sycl.cc) — call
  // into it via a free function with normal linkage; both libraries
  // see the same IpcManager layout because CTP_ENABLE_SYCL=1 is set
  // on both.
  {
    ConfigManager *config = CLIO_CONFIG_MANAGER;
    u32 queue_depth = config->GetQueueDepth();
    constexpr size_t kSyclClientBackendBytes = 64 * 1024 * 1024;  // 64 MB
    extern bool ChiServerBootstrapSyclGpu(IpcManager *self, u32 queue_depth,
                                           size_t backend_bytes);
    if (!ChiServerBootstrapSyclGpu(this, queue_depth,
                                    kSyclClientBackendBytes)) {
      return false;
    }
  }
#endif

  // Identify this host
  if (!IdentifyThisHost()) {
    HLOG(kError, "Warning: Could not identify host, using default node ID");
    this_host_ = Host();  // Default constructor gives node_id = 0
  } else {
    HLOG(kDebug, "Node ID identified: 0x{:x}", this_host_.node_id);
  }

  // Initialize CTP TLS key for task counter (needed for CreateTaskId in
  // runtime)
  CTP_THREAD_MODEL->CreateTls<TaskCounter>(chi_task_counter_key_, nullptr);

  // Create scheduler using factory
  auto *config = CLIO_CONFIG_MANAGER;
  if (config && config->IsValid()) {
    std::string sched_name = config->GetLocalSched();
    scheduler_ = SchedulerFactory::Get(sched_name);
    HLOG(kDebug, "Scheduler initialized: {}", sched_name);
  }

  // Create lightbeam transports for client task reception
  {
    u32 port = config->GetPort();

    try {
      // TCP ROUTER server on port+3. Honor CLIO_BIND_ADDR so this matches
      // whatever LoadHostfile picked; otherwise tests on Windows can't
      // avoid the Defender Firewall prompt on the ROUTER port even when
      // the main server is on loopback.
      std::string router_bind = "0.0.0.0";
      if (const char *env = chi::env::GetCompat("BIND_ADDR")) {
        if (*env) router_bind = env;
      }
      if (UseLocalZmqIpc()) {
        // macOS (issue #482): bind the local client ROUTER on an ipc:// unix
        // socket so replies route reliably; same-host clients connect their
        // DEALER to the matching path. Cross-node clients are served by the
        // main net ROUTER, which is unaffected.
        ctp::SystemInfo::EnsureMemfdDir();
        std::string zmq_ipc = LocalZmqIpcPath(port);
        client_tcp_transport_ = ctp::lbm::TransportFactory::Get(
            zmq_ipc, ctp::lbm::TransportType::kZeroMq,
            ctp::lbm::TransportMode::kServer, "ipc", 0);
        HLOG(kInfo, "IpcManager: client ZMQ ROUTER bound on ipc {}", zmq_ipc);
      } else {
        client_tcp_transport_ = ctp::lbm::TransportFactory::Get(
            router_bind, ctp::lbm::TransportType::kZeroMq,
            ctp::lbm::TransportMode::kServer, "tcp", port + 3);
        HLOG(kInfo, "IpcManager: TCP ROUTER transport bound on {}:{}",
             router_bind, port + 3);
      }
    } catch (const std::exception &e) {
      HLOG(kError, "IpcManager::ServerInit: Failed to bind TCP server: {}",
           e.what());
    }

    try {
      // IPC server on Unix domain socket. Ensure the per-user bookkeeping
      // directory exists first -- on Windows it lives under %TEMP% and is
      // not created by default, so binding the socket would otherwise fail.
      ctp::SystemInfo::EnsureMemfdDir();
      std::string ipc_path =
          ctp::SystemInfo::GetMemfdPath("chimaera_" + std::to_string(port) + ".ipc");
      client_ipc_transport_ = ctp::lbm::TransportFactory::Get(
          ipc_path, ctp::lbm::TransportType::kSocket,
          ctp::lbm::TransportMode::kServer, "ipc", 0);
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
      CTP_THREAD_MODEL->GetTls<TaskCounter>(chi_task_counter_key_);
  if (counter) {
    delete counter;
    CTP_THREAD_MODEL->SetTls(chi_task_counter_key_,
                              static_cast<TaskCounter *>(nullptr));
  }

  // Stop heartbeat thread
  if (heartbeat_running_.load()) {
    heartbeat_running_.store(false);
    if (heartbeat_thread_.joinable()) {
      heartbeat_thread_.join();
    }
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

void IpcManager::RegisterTransportShutdownHook(std::function<void()> hook) {
  std::lock_guard<std::mutex> lock(transport_shutdown_hooks_mutex_);
  transport_shutdown_hooks_.push_back(std::move(hook));
}

void IpcManager::ClearTransports() {
  // Stop any module-owned threads that hold a raw transport pointer BEFORE we
  // free the transports below. Hooks run once; move them out under the lock so
  // a hook that re-enters (or a second ClearTransports call from
  // ServerFinalize) sees an empty list.
  std::vector<std::function<void()>> hooks;
  {
    std::lock_guard<std::mutex> lock(transport_shutdown_hooks_mutex_);
    hooks.swap(transport_shutdown_hooks_);
  }
  for (auto &hook : hooks) {
    if (hook) hook();
  }

  local_transport_.reset();
  main_transport_.reset();
  client_tcp_transport_.reset();
  client_ipc_transport_.reset();
}

void IpcManager::ServerFinalize() {
  if (!is_initialized_) {
    return;
  }

  // GPU orchestrator finalization removed along with the GPU runtime.
  // gpu2cpu_queue + gpu2cpu_copy_backend are torn down by
  // gpu::IpcManager::FinalizeGpuQueuesHip / FinalizeGpuQueuesSycl
  // when gpu_ipc_'s unique_ptr is destroyed.

  // Close persistent outbound DEALER sockets before resetting transports
  ClearClientPool();

  // Transports may have already been reset by ClearTransports() (called
  // earlier in the shutdown sequence before workers are freed); these are
  // no-ops in that case.
  ClearTransports();

  // Clear main allocator pointer
  main_allocator_ = nullptr;

  is_initialized_ = false;
}

// Template methods (NewTask, DelTask, AllocateBuffer, Enqueue) are implemented
// inline in the header

TaskQueue *IpcManager::GetTaskQueue() { return worker_queues_.ptr_; }

bool IpcManager::IsInitialized() const { return is_initialized_; }

u32 IpcManager::GetWorkerCount() {
  return num_workers_;
}

u32 IpcManager::GetNumSchedQueues() const {
  return num_sched_queues_;
}

void IpcManager::SetNumSchedQueues(u32 num_sched_queues) {
  num_sched_queues_ = num_sched_queues;
  HLOG(kInfo, "IpcManager: Updated num_sched_queues to {}", num_sched_queues);
}


bool IpcManager::ServerInitShm() {
  ConfigManager *config = CLIO_CONFIG_MANAGER;

  try {
    // Set allocator ID for main segment
    main_allocator_id_ = ctp::ipc::AllocatorId::Get(1, 0);

    // Get configurable segment name
    std::string main_segment_name =
        config->GetSharedMemorySegmentName(kMainSegment);

    // Use calculated or explicit main_segment_size
    size_t main_segment_size = config->CalculateMainSegmentSize();

    HLOG(kInfo, "Initializing main shared memory segment: {} bytes ({} MB)",
         main_segment_size, main_segment_size / (1024 * 1024));

    // Initialize main backend with custom header size
    if (!main_backend_.shm_init(main_allocator_id_,
                                ctp::Unit<size_t>::Bytes(main_segment_size),
                                main_segment_name)) {
      return false;
    }

    // Create main allocator (CLIO_TASK_ALLOC_T = BuddyAllocator) for task data
    main_allocator_ = main_backend_.MakeAlloc<CLIO_TASK_ALLOC_T>();
    if (!main_allocator_) {
      return false;
    }

    // Initialize queue segment (CLIO_QUEUE_ALLOC_T = ArenaAllocator) for TaskQueues
    queue_allocator_id_ = ctp::ipc::AllocatorId::Get(2, 0);
    std::string queue_segment_name =
        config->GetSharedMemorySegmentName(kQueueSegment);
    size_t queue_segment_size = config->CalculateQueueSegmentSize();
    HLOG(kInfo, "Initializing queue shared memory segment: {} bytes ({} KB)",
         queue_segment_size, queue_segment_size / 1024);
    if (!queue_backend_.shm_init(queue_allocator_id_,
                                 ctp::Unit<size_t>::Bytes(queue_segment_size),
                                 queue_segment_name)) {
      return false;
    }
    queue_allocator_ = queue_backend_.MakeAlloc<CLIO_QUEUE_ALLOC_T>();
    if (!queue_allocator_) {
      return false;
    }

    return true;
  } catch (const std::exception &e) {
    return false;
  }
}

bool IpcManager::ClientInitShm() {
  ConfigManager *config = CLIO_CONFIG_MANAGER;

  try {
    // Set allocator IDs (must match server)
    main_allocator_id_ = ctp::ipc::AllocatorId(1, 0);
    queue_allocator_id_ = ctp::ipc::AllocatorId(2, 0);

    // Get configurable segment names with environment variable expansion
    std::string main_segment_name =
        config->GetSharedMemorySegmentName(kMainSegment);
    std::string queue_segment_name =
        config->GetSharedMemorySegmentName(kQueueSegment);

    // Attach to existing main shared memory segment created by server
    if (!main_backend_.shm_attach(main_segment_name)) {
      return false;
    }

    // Attach to main allocator (CLIO_TASK_ALLOC_T = BuddyAllocator)
    main_allocator_ = main_backend_.AttachAlloc<CLIO_TASK_ALLOC_T>();
    if (!main_allocator_) {
      return false;
    }

    // Attach to queue segment (CLIO_QUEUE_ALLOC_T = ArenaAllocator)
    if (!queue_backend_.shm_attach(queue_segment_name)) {
      return false;
    }
    queue_allocator_ = queue_backend_.AttachAlloc<CLIO_QUEUE_ALLOC_T>();
    if (!queue_allocator_) {
      return false;
    }

    return true;
  } catch (const std::exception &e) {
    return false;
  }
}

bool IpcManager::ServerInitQueues() {
  if (!queue_allocator_) {
    return false;
  }

  try {
    // Initialize runtime metadata
    runtime_pid_ = ctp::SystemInfo::GetPid();
    server_generation_.store(
        static_cast<u64>(
            std::chrono::steady_clock::now().time_since_epoch().count()),
        std::memory_order_release);

    // Get worker counts from ConfigManager
    ConfigManager *config = CLIO_CONFIG_MANAGER;
    u32 thread_count = config->GetNumThreads();
    // Note: Last worker serves dual roles as both task worker and network
    // worker
    u32 total_workers = thread_count;

    // Store worker count and scheduling queue count
    num_workers_ = total_workers;
    num_sched_queues_ = thread_count;

    // Get configured queue depth (no longer hardcoded)
    u32 queue_depth = config->GetQueueDepth();

    HLOG(kInfo,
         "Initializing {} worker queues with depth {} (last worker serves dual "
         "role)",
         total_workers, queue_depth);

    // Allocate TaskQueue in queue segment (CLIO_QUEUE_ALLOC_T = ArenaAllocator)
    worker_queues_ = queue_allocator_->NewObj<TaskQueue>(
        queue_allocator_,
        total_workers,  // num_lanes equals total worker count
        2,  // num_priorities (2 priorities: 0=normal, 1=resumed tasks)
        queue_depth);  // Use configured depth instead of hardcoded 1024
    worker_queues_off_ = worker_queues_.shm_.off_.load();

    // Initialize network queue for send operations.
    // Cross-node sends are split latency vs I/O so SWIM probes and
    // small ACKs never queue behind bulk PutBlob/GetBlob payloads.
    // See NetQueuePriority for the priority order and the drain
    // strategy in Runtime::Send.
    net_queue_ = queue_allocator_->NewObj<NetQueue>(
        queue_allocator_,
        1,                          // num_lanes
        kNetQueueNumPriorities,     // num_priorities
        queue_depth);

    return !worker_queues_.IsNull() && !net_queue_.IsNull();
  } catch (const std::exception &e) {
    return false;
  }
}

void IpcManager::AssignGpuLanesToWorker() {
  size_t num_gpus = GetGpuQueueCount();
  if (num_gpus == 0 || !scheduler_) return;
  Worker *gpu_worker = scheduler_->GetGpuWorker();
  if (!gpu_worker) return;
  std::vector<GpuTaskLane *> gpu_lanes;
  gpu_lanes.reserve(num_gpus);
  for (size_t gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
    GpuTaskQueue *gpu_queue = GetGpuQueue(gpu_id);
    if (gpu_queue) {
      GpuTaskLane *gpu_lane = &gpu_queue->GetLane(0, 0);
      gpu_lanes.push_back(gpu_lane);
      gpu_lane->SetAssignedWorkerId(gpu_worker->GetId());
    }
  }
  gpu_worker->SetGpuLanes(gpu_lanes);

  // Wake the GPU worker in case it's sleeping in epoll_wait.
  // The worker may have entered sleep before gpu_lanes_ was set.
  TaskLane *lane = gpu_worker->GetLane();
  if (lane) {
    int tid = lane->GetTid();
    if (tid > 0) {
      ctp::lbm::EventManager::Signal(ctp::SystemInfo::GetPid(), tid);
    }
  }
}


bool IpcManager::ClientInitQueues() {
  if (!queue_allocator_) {
    return false;
  }

  try {
    // Reconstruct the worker_queues_ FullPtr from the SHM offset received
    // during WaitForLocalServer() (via ClientConnectTask::worker_queues_off_).
    // The offset is relative to queue_allocator_->GetBackendData().
    if (worker_queues_off_ == 0) {
      HLOG(kError, "ClientInitQueues: worker_queues_off_ not set "
           "(server did not send queue offset)");
      return false;
    }

    worker_queues_.shm_.off_ = worker_queues_off_;
    worker_queues_.shm_.alloc_id_ = queue_allocator_->GetId();
    worker_queues_.ptr_ = reinterpret_cast<TaskQueue *>(
        queue_allocator_->GetBackendData() + worker_queues_off_);

    return !worker_queues_.IsNull();
  } catch (const std::exception &e) {
    return false;
  }
}

bool IpcManager::StartLocalServer() {
  ConfigManager *config = CLIO_CONFIG_MANAGER;

  try {
    // Start local ZeroMQ server using CTP Lightbeam
    std::string addr = "127.0.0.1";
    std::string protocol = "tcp";
    u32 port = config->GetPort() + 1;  // Use ZMQ port + 1 for local server

    local_transport_ = ctp::lbm::TransportFactory::Get(
        addr, ctp::lbm::TransportType::kZeroMq,
        ctp::lbm::TransportMode::kServer, protocol, port);

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
  // Semantics: 0 = fail immediately, -1 = wait forever, >0 = timeout in seconds
  const char *wait_env = chi::env::GetCompat("WAIT_SERVER");
  if (wait_env != nullptr) {
    wait_server_timeout_ = static_cast<float>(std::atof(wait_env));
  }

  HLOG(kInfo, "Waiting for runtime via lightbeam (timeout={}s)",
       wait_server_timeout_);

  // 0 = don't wait at all
  if (wait_server_timeout_ == 0) {
    HLOG(kError, "CHI_WAIT_SERVER=0: not waiting for runtime");
    return false;
  }

  // At scale (>=64 chimaera daemons) the daemon's local 9416 ROUTER's I/O
  // thread is starved by initial cross-node SWIM probes when this DEALER
  // first connects, the ZMTP greeting EPIPE's, and the DEALER ends up in
  // a half-open state ZMQ's auto-reconnect cannot recover from. Sending a
  // ClientConnectTask through that DEALER then sits in Future.Wait()
  // forever — IsServerAlive's TCP-level connect() probe still succeeds
  // (the ROUTER does accept()), so server_alive_ stays true and the
  // ClientRecv spin loop never triggers WaitForServerAndReconnect.
  // Defend ourselves with a per-attempt timeout + DEALER recreate loop;
  // we keep the total wait budget = wait_server_timeout_ but split it
  // across attempts so a single dead greeting can't burn the whole window.
  float total_timeout = wait_server_timeout_ > 0 ? wait_server_timeout_ : 0;
  float per_attempt = total_timeout > 0 ? std::min(total_timeout, 15.0f) : 0;
  auto attempt_start = std::chrono::steady_clock::now();
  int attempt_idx = 0;

retry_attempt:
  ++attempt_idx;
  // Send a ClientConnectTask via the lightbeam transport
  auto task = NewTask<clio::run::admin::ClientConnectTask>(
      CreateTaskId(), kAdminPoolId, PoolQuery::Local());
  auto future = IpcCpu2CpuZmq::ClientSend(this,task, ipc_mode_);

  // Wait for response with per-attempt timeout
  if (!future.Wait(per_attempt)) {
    DelTask(task);
    float elapsed = std::chrono::duration<float>(
        std::chrono::steady_clock::now() - attempt_start).count();
    if (total_timeout > 0 && elapsed >= total_timeout) {
      HLOG(kError, "Timeout waiting for runtime after {} seconds ({} attempts)",
           wait_server_timeout_, attempt_idx);
      HLOG(kError, "This usually means:");
      HLOG(kError, "1. Chimaera runtime is not running");
      HLOG(kError, "2. Runtime failed to start");
      HLOG(kError, "3. Network connectivity issues");
      return false;
    }
    HLOG(kWarning, "Attempt {} timed out after {:.1f}s; recreating DEALER",
         attempt_idx, per_attempt);
    if (ipc_mode_ == IpcMode::kTcp) {
      auto *config = CLIO_CONFIG_MANAGER;
      u32 port = config->GetPort();
      if (zmq_recv_running_.load()) {
        zmq_recv_running_.store(false);
        if (zmq_recv_thread_.joinable()) zmq_recv_thread_.join();
      }
      zmq_transport_.reset();
      {
        std::lock_guard<std::mutex> lock(pending_futures_mutex_);
        pending_zmq_futures_.clear();
        pending_response_archives_.clear();
      }
      try {
        if (UseLocalZmqIpc()) {
          // Mirror ClientInit: recreate the DEALER over the local ipc://
          // endpoint on macOS (issue #482).
          ctp::SystemInfo::EnsureMemfdDir();
          std::string zmq_ipc = LocalZmqIpcPath(port);
          zmq_transport_ = ctp::lbm::TransportFactory::Get(
              zmq_ipc, ctp::lbm::TransportType::kZeroMq,
              ctp::lbm::TransportMode::kClient, "ipc", 0);
        } else {
          zmq_transport_ = ctp::lbm::TransportFactory::Get(
              config->GetServerAddr(), ctp::lbm::TransportType::kZeroMq,
              ctp::lbm::TransportMode::kClient, "tcp", port + 3);
        }
      } catch (const std::exception &e) {
        HLOG(kError, "WaitForLocalServer: DEALER recreate failed: {}",
             e.what());
        return false;
      }
      zmq_recv_running_.store(true);
      zmq_recv_thread_ = std::thread([this]() { RecvZmqClientThread(); });
    }
    goto retry_attempt;
  }

  if (task->response_ == 0) {
    client_generation_ = task->server_generation_;
    worker_queues_off_ = task->worker_queues_off_;
    if (task->server_pid_ > 0) {
      runtime_pid_ = static_cast<int>(task->server_pid_);
    }
    HLOG(kInfo, "Successfully connected to runtime (generation={}, server_pid={})",
         client_generation_, runtime_pid_);

    // Client-side GPU queue init was for the cpu2gpu / gpu2gpu queues
    // of the GPU runtime. With the runtime gone, kernels submit
    // directly via gpu2cpu_queue from server-init's pinned-host
    // backend; no client-side attach needed.

    // Task cleanup is handled by ~Future() since Wait() marked it consumed.
    return true;
  }

  // A response arrived but carries a non-zero code. The server's
  // ClientConnect handler ALWAYS sets response_ = 0 (admin_runtime.cc), so a
  // non-zero value here is not an intentional rejection — it is a transient
  // artifact (e.g. a stale/mismatched correlation while the daemon is reaping
  // an abruptly-dead client that left an in-flight response on the wire, the
  // cr_client_retry_client_death_* reconnect race, issue #486). Treat it like
  // a timed-out attempt: retry within the remaining wait budget instead of
  // hard-failing, so the reconnect is deterministic rather than flaky.
  {
    float elapsed = std::chrono::duration<float>(
        std::chrono::steady_clock::now() - attempt_start).count();
    if (total_timeout > 0 && elapsed >= total_timeout) {
      HLOG(kError,
           "Runtime responded with error code {} on every attempt within {}s "
           "({} attempts)",
           task->response_, wait_server_timeout_, attempt_idx);
      // Task cleanup is handled by ~Future() since Wait() marked it consumed.
      return false;
    }
    HLOG(kWarning,
         "Attempt {} got transient non-zero response {}; retrying connect",
         attempt_idx, task->response_);
    goto retry_attempt;
  }
}

bool IpcManager::WaitForLocalRuntimeStop(u32 timeout_sec) {
  HLOG(kInfo, "Waiting for runtime to stop (timeout={}s)", timeout_sec);

  // Temporarily disable reconnection so that Recv() returns false
  // immediately when the heartbeat detects the server is dead, instead
  // of blocking in WaitForServerAndReconnect for up to 60 seconds.
  float saved_retry = client_retry_timeout_;
  int saved_try_new = client_try_new_servers_;
  client_retry_timeout_ = 0;
  client_try_new_servers_ = 0;

  for (u32 elapsed = 0; elapsed < timeout_sec; ++elapsed) {
    // Send a ClientConnectTask with a 1-second timeout
    auto task = NewTask<clio::run::admin::ClientConnectTask>(
        CreateTaskId(), kAdminPoolId, PoolQuery::Local());
    auto future = IpcCpu2CpuZmq::ClientSend(this,task, ipc_mode_);

    if (!future.Wait(1.0f)) {
      // Timeout or server dead: runtime is no longer responding
      client_retry_timeout_ = saved_retry;
      client_try_new_servers_ = saved_try_new;
      HLOG(kInfo, "Runtime stopped (no response after {}s)", elapsed + 1);
      return true;
    }

    // Runtime still responded — it's still alive, keep waiting
    HLOG(kDebug, "Runtime still alive after {}s, retrying...", elapsed + 1);
  }

  client_retry_timeout_ = saved_retry;
  client_try_new_servers_ = saved_try_new;
  HLOG(kError, "Runtime still running after {}s timeout", timeout_sec);
  return false;
}

void IpcManager::SetNodeId(const std::string &hostname) {
  (void)hostname;  // Unused parameter
}

u64 IpcManager::GetNodeId() const {
  // Return the node ID from the identified host
  return this_host_.node_id;
}

bool IpcManager::LoadHostfile() {
  ConfigManager *config = CLIO_CONFIG_MANAGER;
  std::string hostfile_path = config->GetHostfilePath();

  // Clear existing hostfile map
  hostfile_map_.clear();
  hosts_cache_valid_ = false;

  if (hostfile_path.empty()) {
    // No hostfile configured: bind on all local interfaces (0.0.0.0) by
    // default. GetServerAddr() defaults to 127.0.0.1 — fine for the
    // client DEALER target on a single host, but useless as a hostfile
    // entry because IdentifyThisHost matches entries against
    // gethostname() and on real multi-rail hosts (e.g. Aurora's
    // `x4315c7s0b0n0`) the hostname is never literally `127.0.0.1`.
    // Pushing "0.0.0.0" here, combined with the wildcard match in
    // IdentifyThisHost, lets the runtime come up anywhere without
    // forcing every user to write a one-line hostfile.
    //
    // CLIO_BIND_ADDR env override: when set, replaces the wildcard with
    // the requested address. Used by tests on Windows to pin to
    // 127.0.0.1 so the Defender Firewall doesn't pop "Allow access?"
    // for every new test binary that binds a fresh port.
    std::string bind_addr = "0.0.0.0";
    if (const char *env = chi::env::GetCompat("BIND_ADDR")) {
      if (*env) bind_addr = env;
    }
    HLOG(kDebug, "No hostfile configured, binding {} as node 0", bind_addr);
    Host host(bind_addr, 0);
    hostfile_map_[0] = host;
    return true;
  }

  try {
    // Use CTP to parse hostfile
    std::vector<std::string> host_ips =
        ctp::ConfigParse::ParseHostfile(hostfile_path);

    // Create Host structs and populate map using linear offset-based node IDs
    HLOG(kDebug, "=== Container to Node ID Mapping (Linear Offset) ===");
    for (size_t offset = 0; offset < host_ips.size(); ++offset) {
      u64 node_id = static_cast<u64>(offset);
      Host host(host_ips[offset], node_id);
      hostfile_map_[node_id] = host;
      HLOG(kDebug, "  Hostfile[{}]: {} -> Node ID: {}", offset,
           host_ips[offset], node_id);
    }
    HLOG(kDebug, "=== Total hosts loaded: {} ===", hostfile_map_.size());
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
    auto *config_manager = CLIO_CONFIG_MANAGER;
    int port = static_cast<int>(config_manager->GetPort());
    std::string key = it->second.ip_address + ":" + std::to_string(port);
    client_pool_.erase(key);
  }

  HLOG(kWarning, "IpcManager: Node {} ({}) marked as DEAD", node_id,
       it->second.ip_address);
}

void IpcManager::SetAlive(u64 node_id) {
  auto it = hostfile_map_.find(node_id);
  if (it == hostfile_map_.end()) return;
  if (it->second.state == NodeState::kAlive) return;  // Already alive

  SetNodeState(node_id, NodeState::kAlive);

  // Remove from dead_nodes_ list
  dead_nodes_.erase(std::remove_if(dead_nodes_.begin(), dead_nodes_.end(),
                                   [node_id](const DeadNodeEntry &e) {
                                     return e.node_id == node_id;
                                   }),
                    dead_nodes_.end());

  HLOG(kInfo, "IpcManager: Node {} ({}) marked as ALIVE", node_id,
       it->second.ip_address);
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

void IpcManager::SetSelfFenced(bool fenced) { self_fenced_ = fenced; }

u64 IpcManager::GetLeaderNodeId() const {
  u64 leader = std::numeric_limits<u64>::max();
  for (const auto &[id, host] : hostfile_map_) {
    if (host.state == NodeState::kAlive && host.node_id < leader) {
      leader = host.node_id;
    }
  }
  return (leader == std::numeric_limits<u64>::max()) ? 0 : leader;
}

bool IpcManager::IsLeader() const { return GetNodeId() == GetLeaderNodeId(); }

u64 IpcManager::AddNode(const std::string &ip_address, u32 port) {
  (void)port;  // Port stored elsewhere (ConfigManager) for now

  // Check if node already exists
  for (const auto &pair : hostfile_map_) {
    if (pair.second.ip_address == ip_address) {
      HLOG(kInfo, "AddNode: Node {} already registered as node_id={}",
           ip_address, pair.first);
      SetAlive(pair.first);
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

namespace {

// Collect every IPv4/IPv6 address bound to a local network interface
// (loopback included). Used so IdentifyThisHost can recognize a hostfile
// entry that is an IP literal (or a hostname resolving to one of our
// interface IPs) as "this node" — hostname string matching alone breaks
// when the hostfile uses addresses instead of names.
std::set<std::string> CollectLocalInterfaceIps() {
  auto v = ctp::SystemInfo::GetLocalInterfaceIps();
  return std::set<std::string>(v.begin(), v.end());
}

// True when `entry` (an IP literal or a resolvable hostname from the
// hostfile) names an address that is bound to one of this node's local
// interfaces. Handles the case the hostname-only matcher misses:
// hostfiles written with raw IPs, or DNS/hosts names that resolve to a
// local NIC IP whose reverse name differs from gethostname().
bool HostMatchesLocalIp(const std::string &entry,
                        const std::set<std::string> &local_ips) {
  if (entry.empty() || local_ips.empty()) return false;
  if (local_ips.count(entry)) return true;  // already an IP literal we hold
  for (const auto &ip : ctp::SystemInfo::ResolveHostname(entry)) {
    if (local_ips.count(ip)) return true;
  }
  return false;
}

}  // namespace

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
  ConfigManager *config = CLIO_CONFIG_MANAGER;
  u32 port = config->GetPort();

  // Collect list of attempted hosts for error reporting
  std::vector<std::string> attempted_hosts;

  // Resolve our local hostname so we can identify which hostfile entry
  // corresponds to this node *without* using bind-failure as the test.
  // Bind-failure-based identity used to work, but breaks on multi-rail
  // fabrics like Aurora's Slingshot HSN: binding to a specific FQDN
  // succeeds on whichever rail the FQDN resolves to, then peers
  // routing via the *other* rail get silently dropped (the listener
  // is on the wrong interface). Solution: identify by hostname match,
  // then bind the actual server on "0.0.0.0" so it listens on every
  // local interface (mirrors how `client_tcp_transport_` is bound).
  std::string local_host = ctp::SystemInfo::GetHostname();
  if (local_host.empty()) {
    HLOG(kError, "Error: GetHostname() failed");
    return false;
  }
  std::string local_short =
      local_host.substr(0, local_host.find('.'));

  // All IPs bound to local interfaces, so a hostfile entry written as a
  // raw IP (or a name resolving to a local NIC) is recognized as this
  // node even when its reverse name differs from gethostname(). This also
  // covers containerized deployments (Docker networks) where the hostfile
  // lists IPs but gethostname() returns the compose service name.
  const std::set<std::string> local_ips = CollectLocalInterfaceIps();

  // Try to identify (by hostname OR local-IP match) and start the server.
  for (const auto &pair : hostfile_map_) {
    const Host &host = pair.second;
    attempted_hosts.push_back(host.ip_address);
    std::string entry_short =
        host.ip_address.substr(0, host.ip_address.find('.'));

    // Treat the synthetic "0.0.0.0" wildcard (pushed by LoadHostfile()
    // when no hostfile is configured) and loopback addresses as "always
    // me" so the runtime binds without needing the user to predeclare
    // the local hostname.
    bool is_loopback = (host.ip_address == "127.0.0.1") ||
                       (host.ip_address == "localhost") ||
                       (host.ip_address == "::1");
    // The hostfile may use NIC-suffixed names (e.g. "ares-comp-31-40g")
    // that resolve to a non-default fabric, while gethostname() returns
    // the plain short name ("ares-comp-31"). Accept a match when the
    // entry's short name starts with `<local_short>-` so suffixed
    // hostnames identify the same node correctly.
    bool suffix_match =
        entry_short.size() > local_short.size() + 1 &&
        entry_short.compare(0, local_short.size(), local_short) == 0 &&
        entry_short[local_short.size()] == '-';
    bool is_me = (host.ip_address == "0.0.0.0") ||
                 is_loopback ||
                 (host.ip_address == local_host) ||
                 (entry_short == local_short) ||
                 suffix_match ||
                 HostMatchesLocalIp(host.ip_address, local_ips);
    if (!is_me) continue;

    // Bind to whatever address the hostfile entry advertises so an
    // override like CLIO_BIND_ADDR=127.0.0.1 actually pins the listener
    // to loopback (no Defender Firewall prompt). The fallback "0.0.0.0"
    // path is preserved for the synthetic wildcard and hostname-only
    // entries that don't resolve to a literal local IP.
    std::string bind_target =
        (host.ip_address == "0.0.0.0" ||
         host.ip_address == local_host ||
         entry_short == local_short || suffix_match)
            ? std::string("0.0.0.0")
            : host.ip_address;
    HLOG(kDebug, "Hostfile entry {} matches local host {}; binding {}",
         host.ip_address, local_host, bind_target);

    try {
      if (TryStartMainServer(bind_target)) {
        HLOG(kInfo,
             "SUCCESS: Main server started on {}:{} "
             "(advertised as {}, node={})",
             bind_target, port, host.ip_address, host.node_id);
        this_host_ = host;
        return true;
      }
    } catch (const std::exception &e) {
      HLOG(kDebug, "Failed to bind {}:{} for {}: {}", bind_target,
           port, host.ip_address, e.what());
    } catch (...) {
      HLOG(kDebug, "Failed to bind 0.0.0.0:{} for {}: unknown error",
           port, host.ip_address);
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
  HLOG(kError, "  chimaera runtime stop");
  HLOG(kError, "");
  HLOG(kError, "Or kill the process directly:");
  HLOG(kError, "  pkill -9 chimaera");
  HLOG(kFatal, "  kill -9 <PID>");
  return false;
}

const std::string &IpcManager::GetCurrentHostname() const {
  return this_host_.ip_address;
}

bool IpcManager::TryStartMainServer(const std::string &hostname) {
  ConfigManager *config = CLIO_CONFIG_MANAGER;

  try {
    // Create main server using Lightbeam TransportFactory
    std::string protocol = "tcp";
    u32 port = config->GetPort();

    HLOG(kDebug, "Attempting to start main server on {}:{}", hostname, port);

    main_transport_ = ctp::lbm::TransportFactory::Get(
        hostname, ctp::lbm::TransportType::kZeroMq,
        ctp::lbm::TransportMode::kServer, protocol, port);

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

ctp::lbm::Transport *IpcManager::GetMainTransport() const {
  return main_transport_.get();
}

ctp::lbm::Transport *IpcManager::GetClientTransport(IpcMode mode) const {
  if (mode == IpcMode::kTcp) return client_tcp_transport_.get();
  if (mode == IpcMode::kIpc) return client_ipc_transport_.get();
  return nullptr;
}

const Host &IpcManager::GetThisHost() const { return this_host_; }

FullPtr<char> IpcManager::AllocateBuffer(size_t size) {
#if CTP_IS_HOST
  // HOST-ONLY PATH: The device implementation is in ipc_manager.h

  // RUNTIME PATH: Use private memory (CTP_MALLOC) — runtime never uses
  // per-process shared memory segments
  if (CLIO_RUNTIME_MANAGER && CLIO_RUNTIME_MANAGER->IsRuntime()) {
    // Use CTP_MALLOC allocator for private memory allocation
    FullPtr<char> buffer = CTP_MALLOC->AllocateObjs<char>(size);
    if (buffer.IsNull()) {
      HLOG(kError, "AllocateBuffer: CTP_MALLOC failed for {} bytes", size);
    }
    return buffer;
  }

  // CLIENT TCP/IPC PATH: Use private memory (no shared memory needed)
  if (ipc_mode_ != IpcMode::kShm) {
    FullPtr<char> buffer = CTP_MALLOC->AllocateObjs<char>(size);
    if (buffer.IsNull()) {
      HLOG(kError,
           "AllocateBuffer: CTP_MALLOC failed for {} bytes (client ZMQ mode)",
           size);
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
#endif  // CTP_IS_HOST
}

void IpcManager::FreeBuffer(FullPtr<char> buffer_ptr) {
#if CTP_IS_HOST
  // HOST PATH: Check various allocators
  if (buffer_ptr.IsNull()) {
    return;
  }

  // Check if allocator ID is null (private memory allocated with CTP_MALLOC)
  if (buffer_ptr.shm_.alloc_id_ == ctp::ipc::AllocatorId::GetNull()) {
    // Private memory - use CTP_MALLOC->Free() for RUNTIME-allocated buffers
    // In RUNTIME mode, AllocateBuffer uses CTP_MALLOC which adds MallocPage
    // header
    CTP_MALLOC->Free(buffer_ptr);
    return;
  }

  // Check main allocator
  if (main_allocator_ && buffer_ptr.shm_.alloc_id_ == main_allocator_id_) {
    main_allocator_->Free(buffer_ptr);
    return;
  }

  // Check per-process shared memory allocators via alloc_map_.
  //
  // alloc_map_ is a std::unordered_map; mutation (IncreaseClientShm,
  // RegisterMemory, WreapDeadIpcs, KillIpcs) is serialised under
  // allocator_map_lock_'s write-side. A bare find() here races with
  // those writers, and a concurrent rehash can deref a stale bucket
  // pointer — caught here under sustained write load as a segfault
  // in the runtime's bdev path. Match ToFullPtr's read-locked pattern.
  u64 alloc_key = (static_cast<u64>(buffer_ptr.shm_.alloc_id_.major_) << 32) |
                  static_cast<u64>(buffer_ptr.shm_.alloc_id_.minor_);
  ctp::ipc::MultiProcessAllocator *resolved_alloc = nullptr;
  {
    allocator_map_lock_.ReadLock();
    auto it = alloc_map_.find(alloc_key);
    if (it != alloc_map_.end()) {
      resolved_alloc = it->second;
    }
    allocator_map_lock_.ReadUnlock();
  }
  if (resolved_alloc != nullptr) {
    resolved_alloc->Free(buffer_ptr);
    return;
  }

  // GPU client-registered backends use AllocatorIds outside alloc_map_ —
  // the host never frees them here (the client owns the device memory and
  // releases it through FreeGpuBackend / admin DeregisterMemory). Silently
  // skip the free for those allocator ids by checking gpu_ipc_ first.
#if CTP_ENABLE_CUDA || CTP_ENABLE_ROCM || CTP_ENABLE_SYCL
  if (gpu_ipc_) {
    for (const auto &dev : gpu_ipc_->per_gpu_devices_) {
      if (dev.client_backends.find(alloc_key) != dev.client_backends.end()) {
        return;
      }
    }
  }
#endif

  HLOG(kWarning, "FreeBuffer: Could not find allocator for alloc_id ({}.{})",
       buffer_ptr.shm_.alloc_id_.major_, buffer_ptr.shm_.alloc_id_.minor_);
#else
  // GPU PATH: Implementation is in ipc_manager.h as inline function
#endif  // CTP_IS_HOST
}

ctp::lbm::Transport *IpcManager::GetOrCreateClient(const std::string &addr,
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
  auto transport = ctp::lbm::TransportFactory::Get(
      addr, ctp::lbm::TransportType::kZeroMq,
      ctp::lbm::TransportMode::kClient, "tcp", port);

  if (!transport) {
    HLOG(kError, "[ClientPool] Failed to create client for {}", key);
    return nullptr;
  }

  // Store in pool and return raw pointer
  ctp::lbm::Transport *raw_ptr = transport.get();
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
  bool was_empty = lane.Empty();
  lane.Push(future);

  // Pick the worker that drains this priority's queue. Cross-node Send
  // priorities (kSendIn{Latency,IO} / kSendOut{Latency,IO}) are owned
  // by net_send_worker; client response priorities (kClientSendTcp /
  // kClientSendIpc) are owned by net_recv_worker (the ROUTER socket is
  // shared with ClientRecv).
  if (was_empty) {
    TaskLane *wake_lane = nullptr;
    switch (priority) {
      case NetQueuePriority::kSendInLatency:
      case NetQueuePriority::kSendInIO:
      case NetQueuePriority::kSendOutLatency:
      case NetQueuePriority::kSendOutIO:
        wake_lane = net_send_lane_ ? net_send_lane_ : net_lane_;
        break;
      case NetQueuePriority::kClientSendTcp:
      case NetQueuePriority::kClientSendIpc:
        wake_lane = net_recv_lane_ ? net_recv_lane_ : net_lane_;
        break;
    }
  }

  HLOG(kDebug,
       "EnqueueNetTask: priority={}, was_empty={}, send_lane={}, recv_lane={}",
       priority_idx, was_empty, net_send_lane_ != nullptr,
       net_recv_lane_ != nullptr);
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
  allocator_map_lock_.WriteLock();

  int pid = ctp::SystemInfo::GetPid();
  u32 index = shm_count_.fetch_add(1, std::memory_order_relaxed);

  // Create shared memory name: chimaera_{pid}_{index}
  std::string shm_name =
      "chimaera_" + std::to_string(pid) + "_" + std::to_string(index);

  // Add 32MB metadata overhead
  size_t total_size = size + kShmMetadataOverhead;

  HLOG(kInfo,
       "IpcManager::IncreaseClientShm: Creating {} with size {} ({} + {} "
       "overhead)",
       shm_name, total_size, size, kShmMetadataOverhead);

  try {
    // Create the shared memory backend
    auto backend = std::make_unique<ctp::ipc::PosixShmMmap>();

    // Create allocator ID: major = pid, minor = index
    ctp::ipc::AllocatorId alloc_id(static_cast<u32>(pid), index);

    // Initialize shared memory using backend's shm_init method
    if (!backend->shm_init(alloc_id, ctp::Unit<size_t>::Bytes(total_size),
                           shm_name)) {
      HLOG(kError, "IpcManager::IncreaseClientShm: Failed to create shm for {}",
           shm_name);
      shm_count_.fetch_sub(1, std::memory_order_relaxed);
      allocator_map_lock_
          .WriteUnlock();  // CRITICAL: Release lock before returning
      return false;
    }

    // Create allocator using backend's MakeAlloc method
    ctp::ipc::MultiProcessAllocator *allocator =
        backend->MakeAlloc<ctp::ipc::MultiProcessAllocator>();

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
    auto reg_task = NewTask<clio::run::admin::RegisterMemoryTask>(
        chi::CreateTaskId(), chi::kAdminPoolId, chi::PoolQuery::Local(),
        alloc_id);
    IpcCpu2CpuZmq::ClientSend(this,reg_task, IpcMode::kTcp).Wait();

    return true;

  } catch (const std::exception &e) {
    allocator_map_lock_.WriteUnlock();
    HLOG(kError, "IpcManager::IncreaseClientShm: Exception creating {}: {}",
         shm_name, e.what());
    shm_count_.fetch_sub(1, std::memory_order_relaxed);
    return false;
  }
}

bool IpcManager::RegisterMemory(const ctp::ipc::AllocatorId &alloc_id) {
  HLOG(kDebug, "RegisterMemory CALLED: alloc_id=({}.{})", alloc_id.major_,
       alloc_id.minor_);
  std::lock_guard<std::mutex> lock(shm_mutex_);
  // Acquire writer lock on allocator_map_lock_ during memory registration
  allocator_map_lock_.WriteLock();

  // Derive shm_name from alloc_id: chimaera_{pid}_{index}
  int owner_pid = static_cast<int>(alloc_id.major_);
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
    auto backend = std::make_unique<ctp::ipc::PosixShmMmap>();
    if (!backend->shm_attach(shm_name)) {
      HLOG(kError, "IpcManager::RegisterMemory: Failed to attach to shm {}",
           shm_name);
      allocator_map_lock_
          .WriteUnlock();  // CRITICAL: Release lock before returning
      return false;
    }

    // Attach to the existing allocator in the backend
    ctp::ipc::MultiProcessAllocator *allocator =
        backend->AttachAlloc<ctp::ipc::MultiProcessAllocator>();

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

  int pid = ctp::SystemInfo::GetPid();
  std::string shm_name =
      "chimaera_" + std::to_string(pid) + "_" + std::to_string(index);

  ctp::ipc::MultiProcessAllocator *allocator = alloc_vector_[index];
  ctp::ipc::AllocatorId alloc_id = allocator->GetId();

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
  allocator_map_lock_.WriteLock();

  int current_pid = ctp::SystemInfo::GetPid();
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
    int owner_pid = static_cast<int>(major);
    if (owner_pid == current_pid) {
      continue;
    }

    // Check if the owning process is still alive.
    if (!ctp::SystemInfo::IsProcessAlive(owner_pid)) {
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

    ctp::ipc::MultiProcessAllocator *allocator = map_it->second;

    // Get the allocator ID to construct shm_name
    ctp::ipc::AllocatorId alloc_id = allocator->GetId();
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
  allocator_map_lock_.WriteLock();

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

    ctp::ipc::MultiProcessAllocator *allocator = map_it->second;

    // Get the allocator ID to construct shm_name
    ctp::ipc::AllocatorId alloc_id = allocator->GetId();
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
  std::string memfd_dir = ctp::SystemInfo::GetMemfdDir();

  for (const auto &name : ctp::SystemInfo::ListDirectory(memfd_dir)) {
    std::string full_path = memfd_dir + "/" + name;
    if (ctp::SystemInfo::RemoveFile(full_path)) {
      HLOG(kDebug, "ClearUserIpcs: Removed memfd symlink: {}", name);
      removed_count++;
    } else {
      HLOG(kDebug, "ClearUserIpcs: Could not remove {}", name);
    }
  }

  if (removed_count > 0) {
    HLOG(kInfo, "ClearUserIpcs: Removed {} memfd symlinks from previous runs",
         removed_count);
  }

  return removed_count;
}

void IpcManager::SetIsClientThread(bool is_client_thread) {
  // Create TLS key if not already created
  CTP_THREAD_MODEL->CreateTls<bool>(chi_is_client_thread_key_, nullptr);

  // Set the flag for the current thread
  bool *flag = new bool(is_client_thread);
  CTP_THREAD_MODEL->SetTls(chi_is_client_thread_key_, flag);

  HLOG(kDebug, "SetIsClientThread: Set to {} for current thread",
       is_client_thread);
}

bool IpcManager::GetIsClientThread() const {
  // Get the TLS value, defaulting to false if not set
  bool *flag = CTP_THREAD_MODEL->GetTls<bool>(chi_is_client_thread_key_);
  if (!flag) {
    return false;
  }
  return *flag;
}

//==============================================================================
// GPU Memory Management
//==============================================================================

//==============================================================================
// Client Retry / Reconnect Methods
//==============================================================================

bool IpcManager::IsServerAlive() const {
  if (!zmq_transport_) return false;
  ctp::lbm::LbmContext ctx;
  if (ipc_mode_ == IpcMode::kShm) {
    ctx.server_pid_ = static_cast<int>(runtime_pid_);
  }
  return zmq_transport_->IsServerAlive(ctx);
}

bool IpcManager::ReconnectToOriginalHost() {
  HLOG(kInfo, "ReconnectToOriginalHost: Attempting to reconnect to restarted server");

  if (ipc_mode_ == IpcMode::kShm) {
    // Detach old shared memory (don't destroy — server owns it)
    main_allocator_ = nullptr;
    worker_queues_ = ctp::ipc::FullPtr<TaskQueue>();
    main_backend_ = ctp::ipc::PosixShmMmap();

    // Re-attach to new shared memory
    if (!ClientInitShm()) return false;
    if (!ClientInitQueues()) return false;

    // Re-create SHM lightbeam transports
    shm_send_transport_ = ctp::lbm::TransportFactory::Get(
        "", ctp::lbm::TransportType::kShm, ctp::lbm::TransportMode::kClient);
    shm_recv_transport_ = ctp::lbm::TransportFactory::Get(
        "", ctp::lbm::TransportType::kShm, ctp::lbm::TransportMode::kServer);

    // Re-register per-process shared memory segments with new server
    for (auto *alloc : alloc_vector_) {
      auto alloc_id = alloc->GetId();
      auto reg_task = NewTask<clio::run::admin::RegisterMemoryTask>(
          chi::CreateTaskId(), chi::kAdminPoolId, chi::PoolQuery::Local(),
          alloc_id);
      IpcCpu2CpuZmq::ClientSend(this,reg_task, IpcMode::kTcp).Wait();
    }
  }

  // For TCP mode the original WaitForLocalServer DEALER may have died
  // mid-greeting (e.g. starved by SWIM I/O at startup) and now sits in a
  // half-open state that ZMQ's auto-reconnect can't recover from — the
  // ROUTER already saw an EPIPE on this identity and HANDSHAKE keeps
  // failing on every retry. Tear the DEALER fully down and rebuild it
  // so the next WaitForLocalServer goes through a fresh socket.
  if (ipc_mode_ == IpcMode::kTcp) {
    auto *config = CLIO_CONFIG_MANAGER;
    u32 port = config->GetPort();

    if (zmq_recv_running_.load()) {
      zmq_recv_running_.store(false);
      if (zmq_recv_thread_.joinable()) {
        zmq_recv_thread_.join();
      }
    }
    zmq_transport_.reset();
    {
      std::lock_guard<std::mutex> lock(pending_futures_mutex_);
      pending_zmq_futures_.clear();
      pending_response_archives_.clear();
    }
    try {
      zmq_transport_ = ctp::lbm::TransportFactory::Get(
          config->GetServerAddr(), ctp::lbm::TransportType::kZeroMq,
          ctp::lbm::TransportMode::kClient, "tcp", port + 3);
    } catch (const std::exception &e) {
      HLOG(kError, "ReconnectToOriginalHost: TCP transport recreate failed: {}",
           e.what());
      return false;
    }
    zmq_recv_running_.store(true);
    zmq_recv_thread_ = std::thread([this]() { RecvZmqClientThread(); });
  }

  // Re-verify server via ClientConnectTask (updates client_generation_)
  if (!WaitForLocalServer()) return false;

  server_alive_.store(true, std::memory_order_release);
  HLOG(kInfo, "ReconnectToOriginalHost: Reconnected, new generation={}",
       client_generation_);
  return true;
}

bool IpcManager::ReconnectToNewHost(const std::string &new_addr) {
  HLOG(kInfo, "ReconnectToNewHost: Switching to {}", new_addr);
  auto *config = CLIO_CONFIG_MANAGER;
  u32 port = config->GetPort();

  // Stop recv thread
  if (zmq_recv_running_.load()) {
    zmq_recv_running_.store(false);
    if (zmq_recv_thread_.joinable()) {
      zmq_recv_thread_.join();
    }
  }

  // Destroy old transport
  zmq_transport_.reset();

  // Clear orphaned pending state
  {
    std::lock_guard<std::mutex> lock(pending_futures_mutex_);
    pending_zmq_futures_.clear();
    pending_response_archives_.clear();
  }

  // Disable SHM/IPC — remote hosts require TCP
  ipc_mode_ = IpcMode::kTcp;
  shm_send_transport_.reset();
  shm_recv_transport_.reset();
  main_allocator_ = nullptr;
  runtime_pid_ = 0;

  // Create new ZMQ DEALER transport
  try {
    zmq_transport_ = ctp::lbm::TransportFactory::Get(
        new_addr, ctp::lbm::TransportType::kZeroMq,
        ctp::lbm::TransportMode::kClient, "tcp", port + 3);
  } catch (const std::exception &e) {
    HLOG(kError, "ReconnectToNewHost: Transport to {} failed: {}",
         new_addr, e.what());
    return false;
  }

  // Restart recv thread
  zmq_recv_running_.store(true);
  zmq_recv_thread_ = std::thread([this]() { RecvZmqClientThread(); });

  // Verify connectivity — the server should respond almost instantly
  // if it's alive.  No long timer; just a quick round-trip check.
  float saved_timeout = wait_server_timeout_;
  wait_server_timeout_ = 0.5f;
  bool ok = WaitForLocalServer();
  wait_server_timeout_ = saved_timeout;
  if (!ok) {
    HLOG(kWarning, "ReconnectToNewHost: {} not responding", new_addr);
    return false;
  }

  server_alive_.store(true, std::memory_order_release);
  HLOG(kInfo, "ReconnectToNewHost: Connected to {} (generation={})",
       new_addr, client_generation_);
  return true;
}

bool IpcManager::WaitForServerAndReconnect(
    std::chrono::steady_clock::time_point start) {
  // Guard against recursive re-entry (WaitForLocalServer → Recv → here)
  reconnecting_.store(true, std::memory_order_release);

  // Phase 1: Try reconnecting to the original server
  // Skip entirely when client_retry_timeout_==0 (go straight to Phase 2).
  // Use a short WaitForLocalServer timeout so each attempt doesn't
  // block for the full 30s default.
  float saved_timeout = wait_server_timeout_;
  if (client_retry_timeout_ != 0) {
    float per_attempt_timeout = std::min(wait_server_timeout_, 3.0f);
    wait_server_timeout_ = per_attempt_timeout;
    while (true) {
      float elapsed =
          std::chrono::duration<float>(std::chrono::steady_clock::now() - start)
              .count();
      if (client_retry_timeout_ >= 0 && elapsed >= client_retry_timeout_) {
        HLOG(kWarning, "WaitForServerAndReconnect: Original server timed out "
             "after {:.1f}s", elapsed);
        break;
      }
      std::this_thread::sleep_for(std::chrono::seconds(1));
      if (ReconnectToOriginalHost()) {
        wait_server_timeout_ = saved_timeout;
        reconnecting_.store(false, std::memory_order_release);
        return true;
      }
    }
    wait_server_timeout_ = saved_timeout;
  } else {
    HLOG(kInfo, "WaitForServerAndReconnect: retry_timeout=0, "
         "skipping Phase 1, going straight to Phase 2");
  }

  // Phase 2: Try random hosts from the hostfile
  if (client_try_new_servers_ <= 0 || hostfile_map_.empty()) {
    reconnecting_.store(false, std::memory_order_release);
    return false;
  }

  const auto &hosts = GetAllHosts();
  if (hosts.empty()) {
    HLOG(kWarning, "WaitForServerAndReconnect: No hosts in hostfile");
    reconnecting_.store(false, std::memory_order_release);
    return false;
  }

  // Pick random hosts and try each (may retry same host — that's fine)
  std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<size_t> dist(0, hosts.size() - 1);

  HLOG(kInfo, "WaitForServerAndReconnect: Trying {} random hosts",
       client_try_new_servers_);
  for (int i = 0; i < client_try_new_servers_; ++i) {
    size_t idx = dist(rng);
    const std::string &addr = hosts[idx].ip_address;
    HLOG(kInfo, "WaitForServerAndReconnect: Trying {}/{}: {}",
         i + 1, client_try_new_servers_, addr);
    if (ReconnectToNewHost(addr)) {
      reconnecting_.store(false, std::memory_order_release);
      return true;
    }
  }

  HLOG(kError, "WaitForServerAndReconnect: All {} random hosts failed",
       client_try_new_servers_);
  reconnecting_.store(false, std::memory_order_release);
  return false;
}

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

  // Set up EventManager for ZMQ transport polling.
  // Use the member zmq_client_em_ (not a local) so the EventManager outlives
  // the transport reset in ClientFinalize() and the ~SocketTransport()
  // destructor can safely call em_->RemoveEvent().
  zmq_transport_->RegisterEventManager(zmq_client_em_);

  // Instrumentation: count of responses this client has received and signaled
  // (FUTURE_COMPLETE set). Mismatch vs daemon-side send count = lost responses.
  size_t recv_count = 0;
  size_t miss_count = 0;

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
        // ETERM means the ZMQ context is being shut down (zmq_ctx_shutdown was
        // called).  Exit immediately so the context destructor is not blocked.
        if (rc == ETERM) return;
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
        ++miss_count;
        HLOG(kError,
             "[CountClientRecv] miss#{}: No pending future for net_key {} "
             "(received={}, misses={})",
             miss_count, net_key, recv_count, miss_count);
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
      ++recv_count;
      if ((recv_count & 0xff) == 0) {
        HLOG(kDebug,
             "[CountClientRecv] cumulative responses received = {} "
             "(misses so far = {})",
             recv_count, miss_count);
      }
    }

    // Only block on epoll when the drain loop found nothing;
    // if we just processed messages, loop back immediately.
    if (!drained_any) {
      zmq_client_em_.Wait(100);  // 100μs (precise with epoll_pwait2)
    }
  }
  // `em` is about to be destroyed (stack-allocated). The transport
  // stashed a raw pointer to it in RegisterEventManager — clear that
  // before unwinding, otherwise ClientFinalize's later ~SocketTransport
  // calls em_->RemoveEvent on freed memory (ASan: heap-use-after-free
  // in EventManager::RemoveEvent → std::unordered_map::find).
  if (zmq_transport_) {
    zmq_transport_->UnregisterEventManager();
  }
}

void IpcManager::HeartbeatThread() {
  while (heartbeat_running_.load()) {
    bool alive = IsServerAlive();
    server_alive_.store(alive, std::memory_order_release);
    std::this_thread::sleep_for(std::chrono::seconds(1));
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

// RegisterAcceleratorMemory was the GPU-runtime hook for staging device
// memory inside the now-removed GPU orchestrator. After the producer-only
// redesign, GPU client backends are registered through the admin
// RegisterMemory path, which calls
// gpu::IpcManager::RegisterClientBackend directly.

#if CTP_ENABLE_CUDA || CTP_ENABLE_ROCM || CTP_ENABLE_SYCL
ctp::ipc::AllocatorId IpcManager::AllocateAndRegisterGpuBackend(
    u32 gpu_id, gpu::IpcManager::MemKind kind, size_t bytes,
    char **out_base) {
  ctp::ipc::AllocatorId result;
  result.SetNull();
  if (out_base) *out_base = nullptr;

  char *base = nullptr;
  switch (kind) {
    case gpu::IpcManager::MemKind::kPinnedHost:
      base = ctp::GpuApi::MallocHost<char>(bytes);
      break;
    case gpu::IpcManager::MemKind::kManagedUvm:
      base = ctp::GpuApi::MallocManaged<char>(bytes);
      break;
    case gpu::IpcManager::MemKind::kDeviceMem:
      ctp::GpuApi::SetDevice(static_cast<int>(gpu_id));
      base = ctp::GpuApi::Malloc<char>(bytes);
      break;
  }
  if (!base) {
    HLOG(kError, "AllocateAndRegisterGpuBackend: alloc failed (kind={}, "
         "bytes={}, gpu_id={})", static_cast<int>(kind), bytes, gpu_id);
    return result;
  }

  // Mint AllocatorId from PID + a counter (mirror IncreaseClientShm).
  u32 idx = shm_count_.fetch_add(1, std::memory_order_relaxed);
  ctp::ipc::AllocatorId alloc_id(
      static_cast<u32>(ctp::SystemInfo::GetPid()), idx);

  // In-process registration: when this IpcManager *is* the runtime
  // (kServer mode), short-circuit the admin RegisterMemoryTask round-trip
  // and call gpu_ipc_->RegisterClientBackend directly. Otherwise send the
  // admin task over the wire so the runtime can register it on our behalf.
  if (CLIO_RUNTIME_MANAGER->IsRuntime() && gpu_ipc_) {
    gpu::IpcManager::ClientBackend b;
    b.alloc_id = alloc_id;
    b.gpu_id = gpu_id;
    b.capacity = bytes;
    b.kind = kind;
    b.host_view = (kind == gpu::IpcManager::MemKind::kDeviceMem) ? nullptr
                                                                  : base;
    b.device_ptr = base;
    if (!gpu_ipc_->RegisterClientBackend(b)) {
      HLOG(kError, "AllocateAndRegisterGpuBackend: in-process register "
           "failed");
      return result;
    }
  } else {
    clio::run::admin::MemoryType admin_kind =
        clio::run::admin::MemoryType::kPinnedHostMemory;
    switch (kind) {
      case gpu::IpcManager::MemKind::kPinnedHost:
        admin_kind = clio::run::admin::MemoryType::kPinnedHostMemory;
        break;
      case gpu::IpcManager::MemKind::kManagedUvm:
        admin_kind = clio::run::admin::MemoryType::kManagedUvm;
        break;
      case gpu::IpcManager::MemKind::kDeviceMem:
        admin_kind = clio::run::admin::MemoryType::kGpuDeviceMemory;
        break;
    }
    ctp::ipc::MemoryBackendId backend_id(alloc_id.major_, alloc_id.minor_);
    char ipc_handle_bytes[64] = {0};
    std::memcpy(ipc_handle_bytes, &base, sizeof(char *));

    auto reg_task = NewTask<clio::run::admin::RegisterMemoryTask>(
        chi::CreateTaskId(), chi::kAdminPoolId, chi::PoolQuery::Local(),
        backend_id, admin_kind, gpu_id, static_cast<u64>(bytes),
        ipc_handle_bytes);
    IpcCpu2CpuZmq::ClientSend(this, reg_task, IpcMode::kTcp).Wait();
  }

  result = alloc_id;
  if (out_base) *out_base = base;
  return result;
}

void IpcManager::FreeGpuBackend(u32 gpu_id,
                                 const ctp::ipc::AllocatorId &alloc_id) {
  if (gpu_ipc_) {
    gpu_ipc_->UnregisterClientBackend(gpu_id, alloc_id);
  }
  // The actual ctp::GpuApi::Free relies on caller-tracked metadata —
  // the host caller passes the base back (out_base from
  // AllocateAndRegisterGpuBackend) and frees through the same API. In a
  // future iteration we could fold that bookkeeping into ClientBackend.
}
#endif  // CTP_ENABLE_CUDA || CTP_ENABLE_ROCM || CTP_ENABLE_SYCL

void IpcManager::BeginTask(Future<Task> &future, Container *container,
                           TaskLane *lane) {
  FullPtr<Task> task_ptr = future.GetTaskPtr();
  if (task_ptr.IsNull()) {
    HLOG(kError, "BeginTask: task_ptr is null!");
    return;
  }
#if CTP_IS_HOST
  Worker *worker = CLIO_CUR_WORKER;

  // Initialize or reset the task's owned RunContext
  task_ptr->SetRunCtx(new RunContext());
  RunContext *run_ctx = task_ptr->GetRunCtx();

  // Clear and initialize RunContext for new task execution
  run_ctx->worker_id_ = worker ? worker->GetId() : 0;
  run_ctx->task_ = task_ptr;        // Store task in RunContext
  run_ctx->is_yielded_ = false;     // Initially not blocked
  run_ctx->container_ = container;  // Store container for CHI_CUR_CONTAINER
  run_ctx->lane_ = lane;            // Store lane for CHI_CUR_LANE
  run_ctx->event_queue_ =
      worker ? worker->GetEventQueue() : nullptr;  // Set event queue
  run_ctx->future_ = future;        // Store future in RunContext
  run_ctx->coro_handle_ = nullptr;  // Coroutine not started yet

  // Initialize adaptive polling fields for periodic tasks
  if (task_ptr->IsPeriodic()) {
    run_ctx->true_period_ns_ = task_ptr->period_ns_;
    run_ctx->yield_time_us_ =
        task_ptr->period_ns_ / 1000.0;  // Initialize with true period
    run_ctx->did_work_ = false;         // Initially no work done
  } else {
    run_ctx->true_period_ns_ = 0.0;
    run_ctx->yield_time_us_ = 0.0;
    run_ctx->did_work_ = false;
  }

  // Populate predicted_stat_ from the container so downstream routing
  // (RouteGlobal's latency-vs-IO lane choice; worker.cc's predicted-load
  // tracking) can read the task's actual payload size without re-doing
  // the GetTaskStats(task) work. Scheduler-class code (RuntimeMapTask)
  // already calls GetTaskStats; pre-populating it here keeps a single
  // source of truth and makes the value available before RouteTask.
  if (container) {
    run_ctx->predicted_stat_ = container->GetTaskStats(task_ptr.ptr_);
  }

  // Mark that RunContext now exists for this task
  task_ptr->SetFlags(TASK_RUN_CTX_EXISTS);

  // NOTE: Do NOT call SetCurrentRunContext here. BeginTask may be called
  // from SendRuntimeClient inside a running coroutine. Overwriting the
  // current RunContext would cause await_suspend_impl to store the parent
  // coroutine handle on the subtask's RunContext instead of the parent's,
  // leading to premature task completion. StartCoroutine and ResumeCoroutine
  // already set the current RunContext before executing the task.
#endif
}

RouteResult IpcManager::RouteTask(Future<Task> &future, bool force_enqueue) {
  // Get task pointer from future
  FullPtr<Task> task_ptr = future.GetTaskPtr();

  if (task_ptr.IsNull()) {
    Worker *worker = CLIO_CUR_WORKER;
    HLOG(kWarning, "Worker {}: RouteTask - task_ptr is null",
         worker ? worker->GetId() : 0);
    return RouteResult::Dne;
  }

  // Check if task has already been routed - if so, return ExecHere
  if (task_ptr->IsRouted()) {
    return RouteResult::ExecHere;
  }

  // Only call ScheduleTask for Dynamic pool queries.
  // ScheduleTask resolves Dynamic routing into concrete modes (e.g.,
  // Broadcast, DirectHash, Local). Concrete routing modes (Range, Physical,
  // Local, Broadcast, etc.) were set by a previous routing step and must
  // not be overridden — doing so would cause infinite re-broadcast loops
  // when tasks arrive at remote nodes (e.g., GetOrCreatePool returns
  // Broadcast on every node since the pool doesn't exist yet).
  auto *pool_manager = CLIO_POOL_MANAGER;
  Container *static_container =
      pool_manager->GetStaticContainer(task_ptr->pool_id_);
  PoolQuery resolved_query = task_ptr->pool_query_;
  if (static_container && resolved_query.IsDynamicMode()) {
    resolved_query = static_container->ScheduleTask(task_ptr);
    task_ptr->pool_query_ = resolved_query;
  }

  // Snapshot the routing intent AFTER ScheduleTask resolves Dynamic but
  // BEFORE ResolvePoolQuery's DirectHash/DirectId → Local boundary-case
  // rewrite.  IsTaskLocal uses this to gate CLIO_FORCE_NET:
  //   - admin tasks that go through Dynamic-resolved-to-Local on single-
  //     node stay local (avoids dragging SaveTaskArchive through ZMQ);
  //   - DirectHash/DirectId/Range/Broadcast/Physical that the resolver
  //     would collapse to Local for the local-container case still take
  //     the network path under force_net_.
  const bool originally_local =
      resolved_query.GetRoutingMode() == RoutingMode::Local;

  // Resolve pool query into concrete physical addresses
  std::vector<PoolQuery> pool_queries =
      ResolvePoolQuery(resolved_query, task_ptr->pool_id_, task_ptr);

  // Check if pool_queries is empty - this indicates an error in resolution
  if (pool_queries.empty()) {
    Worker *worker = CLIO_CUR_WORKER;
    HLOG(kError,
         "Worker {}: Task routing failed - no pool queries resolved. "
         "Pool ID: {}, Method: {}",
         worker ? worker->GetId() : 0, task_ptr->pool_id_, task_ptr->method_);
    return RouteResult::Dne;
  }

  // Check if task should be processed locally
  bool is_local = IsTaskLocal(task_ptr, pool_queries, originally_local);
  if (is_local) {
    RouteResult result = RouteLocal(future, force_enqueue);
    // If container is plugged or gone, add to retry queue
    if (result == RouteResult::Retry || result == RouteResult::Dne) {
      Worker *worker = CLIO_CUR_WORKER;
      HLOG(kError, "RouteTask: RouteLocal returned {} for pool={} method={}, worker={}",
           (int)result, task_ptr->pool_id_, task_ptr->method_,
           worker ? (int)worker->GetId() : -1);
      if (worker && task_ptr->GetRunCtx()) {
        worker->AddToRetryQueue(task_ptr->GetRunCtx());
      }
    }
    return result;
  } else {
    return RouteGlobal(future, pool_queries);
  }
}

bool IpcManager::IsTaskLocal(const FullPtr<Task> & /*task_ptr*/,
                             const std::vector<PoolQuery> &pool_queries,
                             bool originally_local) {
  // CLIO_FORCE_NET stress mode: routing is determined entirely by the
  // caller's original intent.  Explicit PoolQuery::Local() stays local;
  // anything else (Dynamic, DirectHash, DirectId, Range, Broadcast,
  // Physical) takes the network path, even on single-node deployments
  // where ResolveDirectHashQuery / ResolveDirectIdQuery would otherwise
  // short-circuit to Local() via their boundary-case optimization.
  // force_net_ is read once in ServerInit; see force_net_ in
  // ipc_manager.h.
  if (force_net_) {
    return originally_local;
  }

  // A single Local() query — whether the user-facing API picked it or
  // ScheduleTask / ResolvePoolQuery collapsed it to Local — is local.
  if (pool_queries.size() == 1 &&
      pool_queries[0].GetRoutingMode() == RoutingMode::Local) {
    return true;
  }

  // If there's only one node, all tasks are local
  if (GetNumHosts() == 1) {
    return true;
  }

  // Task is local only if there is exactly one pool query
  if (pool_queries.size() != 1) {
    return false;
  }

  const PoolQuery &query = pool_queries[0];

  // Check routing mode first, then specific conditions
  RoutingMode routing_mode = query.GetRoutingMode();

  switch (routing_mode) {
    case RoutingMode::Local:
      return true;  // Always local

    case RoutingMode::Dynamic:
      // Dynamic queries should have been resolved by ScheduleTask before
      // reaching here. Treat as local as a safe fallback.
      return true;

    case RoutingMode::Physical: {
      // Physical mode is local only if targeting local node
      u64 local_node_id = GetNodeId();
      return query.GetNodeId() == local_node_id;
    }

    case RoutingMode::DirectId:
    case RoutingMode::DirectHash:
    case RoutingMode::Range:
    case RoutingMode::Broadcast:
      // These modes should have been resolved to Physical queries by now
      // If we still see them here, they are not local
      return false;

    case RoutingMode::ToLocalCpu:
      return true;  // GPU producer-only path: always local

    case RoutingMode::Null:
      return true;  // Null mode is a no-op, treat as local
  }

  return false;
}

RouteResult IpcManager::RouteLocal(Future<Task> &future, bool force_enqueue) {
  // Get task pointer from future
  FullPtr<Task> task_ptr = future.GetTaskPtr();

  // Mark as routed so the task is not re-routed on subsequent passes.
  task_ptr->SetFlags(TASK_ROUTED);

  // Resolve the actual execution container
  auto *pool_manager = CLIO_POOL_MANAGER;
  bool is_plugged = false;
  ContainerId container_id = task_ptr->pool_query_.GetContainerId();
  Container *exec_container =
      pool_manager->GetContainer(task_ptr->pool_id_, container_id, is_plugged);

  if (!exec_container) {
    HLOG(kError, "RouteLocal: Container not found for pool={} container_id={} method={}",
         task_ptr->pool_id_, container_id, task_ptr->method_);
    return RouteResult::Dne;
  }
  if (is_plugged) {
    HLOG(kWarning, "RouteLocal: Container plugged for pool={}", task_ptr->pool_id_);
    return RouteResult::Retry;
  }

  // RouteToGpu was the cpu→gpu dispatch for the now-removed GPU
  // runtime. ToLocalGpu / LocalGpuBcast routing modes are no longer
  // honored — kernels are pure task producers, not consumers.

  // Set the completer_ field to track which container will execute this task
  task_ptr->SetCompleter(exec_container->container_id_);

  // Update RunContext to use the resolved execution container
  if (task_ptr->GetRunCtx()) {
    task_ptr->GetRunCtx()->container_ = exec_container;
  }

  // Use scheduler to pick the destination worker
  Worker *worker = CLIO_CUR_WORKER;
  u32 dest_worker_id =
      scheduler_->RuntimeMapTask(worker, future, exec_container);

  // If destination matches this worker and not forced to enqueue, execute directly
  if (!force_enqueue && worker && dest_worker_id == worker->GetId()) {
    return RouteResult::ExecHere;
  }

  // Enqueue to the destination worker's lane
  auto &dest_lane = worker_queues_->GetLane(dest_worker_id, 0);
  dest_lane.Push(future);
  return RouteResult::Local;
}

RouteResult IpcManager::RouteGlobal(Future<Task> &future,
                             const std::vector<PoolQuery> &pool_queries) {
  // Get task pointer from future
  FullPtr<Task> task_ptr = future.GetTaskPtr();

  // Log the global routing for debugging
  if (!pool_queries.empty()) {
    Worker *worker = CLIO_CUR_WORKER;
    const auto &query = pool_queries[0];
    HLOG(kDebug,
         "Worker {}: RouteGlobal - routing task method={}, pool_id={} to node "
         "{} (routing_mode={})",
         worker ? worker->GetId() : 0, task_ptr->method_, task_ptr->pool_id_,
         query.GetNodeId(), static_cast<int>(query.GetRoutingMode()));
  }

  // Store pool_queries in task's RunContext for SendIn to access
  if (task_ptr->GetRunCtx()) {
    RunContext *run_ctx = task_ptr->GetRunCtx();
    run_ctx->pool_queries_ = pool_queries;
  }

  // Pick the latency vs I/O SendIn lane based on the task's actual
  // payload size — small probes / metadata sit on kSendInLatency so
  // they're not buried behind 1 MiB PutBlob bulks on the wire. The
  // scheduler (BeginTask / pre-routing) populates RunContext::
  // predicted_stat_ from container->GetTaskStats(task), so we just
  // read it here instead of recomputing.
  size_t io_size = 0;
  if (task_ptr->GetRunCtx()) {
    io_size = task_ptr->GetRunCtx()->predicted_stat_.io_size_;
  }
  NetQueuePriority sendin_prio = (io_size >= kNetQueueIoThreshold)
                                     ? NetQueuePriority::kSendInIO
                                     : NetQueuePriority::kSendInLatency;
  EnqueueNetTask(future, sendin_prio);

  // Set TASK_ROUTED flag on original task
  task_ptr->SetFlags(TASK_ROUTED);

  Worker *worker = CLIO_CUR_WORKER;
  HLOG(kDebug, "Worker {}: RouteGlobal - task enqueued to net_queue",
       worker ? worker->GetId() : 0);

  return RouteResult::Network;
}


std::vector<PoolQuery> IpcManager::ResolvePoolQuery(
    const PoolQuery &query, PoolId pool_id, const FullPtr<Task> &task_ptr) {
  // Basic validation
  if (pool_id.IsNull()) {
    return {};  // Invalid pool ID
  }

  RoutingMode routing_mode = query.GetRoutingMode();
  std::vector<PoolQuery> result;

  switch (routing_mode) {
    case RoutingMode::Local:
      result = ResolveLocalQuery(query, task_ptr);
      break;
    case RoutingMode::Dynamic:
      // Dynamic queries should have been resolved by Container::ScheduleTask
      // before reaching ResolvePoolQuery. Fall through to Local as safe default.
      result = ResolveLocalQuery(query, task_ptr);
      break;
    case RoutingMode::DirectId:
      result = ResolveDirectIdQuery(query, pool_id, task_ptr);
      break;
    case RoutingMode::DirectHash:
      result = ResolveDirectHashQuery(query, pool_id, task_ptr);
      break;
    case RoutingMode::Range:
      result = ResolveRangeQuery(query, pool_id, task_ptr);
      break;
    case RoutingMode::Broadcast:
      result = ResolveBroadcastQuery(query, pool_id, task_ptr);
      break;
    case RoutingMode::Physical:
      result = ResolvePhysicalQuery(query, pool_id, task_ptr);
      break;
    case RoutingMode::ToLocalCpu:
    case RoutingMode::Null:
      // GPU producer-only ToLocalCpu and Null modes pass through.
      result = {query};
      break;
  }

  // Set ret_node_ on all resolved queries to this node's ID
  u32 this_node_id = GetNodeId();
  for (auto &pq : result) {
    pq.SetReturnNode(this_node_id);
  }

  return result;
}

std::vector<PoolQuery> IpcManager::ResolveLocalQuery(
    const PoolQuery &query, const FullPtr<Task> &task_ptr) {
  // Local routing - process on current node
  return {query};
}

std::vector<PoolQuery> IpcManager::ResolveDirectIdQuery(
    const PoolQuery &query, PoolId pool_id, const FullPtr<Task> &task_ptr) {
  auto *pool_manager = CLIO_POOL_MANAGER;
  if (pool_manager == nullptr) {
    return {query};  // Fallback to original query
  }

  // Get the container ID from the query
  ContainerId container_id = query.GetContainerId();

  // Boundary case optimization: Check if container exists on this node
  if (pool_manager->HasContainer(pool_id, container_id)) {
    // Container is local, resolve to Local query
    return {PoolQuery::Local()};
  }

  // Get the physical node ID for this container
  u32 node_id = pool_manager->GetContainerNodeId(pool_id, container_id);

  // Create a Physical PoolQuery to that node
  return {PoolQuery::Physical(node_id)};
}

std::vector<PoolQuery> IpcManager::ResolveDirectHashQuery(
    const PoolQuery &query, PoolId pool_id, const FullPtr<Task> &task_ptr) {
  auto *pool_manager = CLIO_POOL_MANAGER;
  if (pool_manager == nullptr) {
    return {query};  // Fallback to original query
  }

  // Get pool info to find the number of containers
  const PoolInfo *pool_info = pool_manager->GetPoolInfo(pool_id);
  if (pool_info == nullptr || pool_info->num_containers_ == 0) {
    return {query};  // Fallback to original query
  }

  // Hash to get container ID
  u32 hash_value = query.GetHash();
  ContainerId container_id = hash_value % pool_info->num_containers_;

  // Boundary case optimization: Check if container exists on this node
  if (pool_manager->HasContainer(pool_id, container_id)) {
    // Container is local, resolve to Local query
    return {PoolQuery::Local()};
  }

  // Check if the address_map_ points this container to the local node.
  // After migration, the container may be mapped here but not yet in
  // containers_ (e.g., forwarded tasks arriving at the destination).
  // Returning Local() prevents an infinite forwarding loop.
  u32 mapped_node = pool_manager->GetContainerNodeId(pool_id, container_id);
  if (mapped_node == GetNodeId()) {
    return {PoolQuery::Local()};
  }

  // Resolve to DirectId so SendIn can dynamically look up the current
  // node via GetContainerNodeId.  This preserves the container_id through
  // the routing chain, which is required for retry-after-recovery: if the
  // original node dies and the container is recovered elsewhere, the retry
  // queue re-resolves DirectId to the new node.
  return {PoolQuery::DirectId(container_id)};
}

std::vector<PoolQuery> IpcManager::ResolveRangeQuery(
    const PoolQuery &query, PoolId pool_id, const FullPtr<Task> &task_ptr) {
  auto *pool_manager = CLIO_POOL_MANAGER;
  if (pool_manager == nullptr) {
    return {query};  // Fallback to original query
  }

  auto *config_manager = CLIO_CONFIG_MANAGER;
  if (config_manager == nullptr) {
    return {query};  // Fallback to original query
  }

  u32 range_offset = query.GetRangeOffset();
  u32 range_count = query.GetRangeCount();

  // Validate range
  if (range_count == 0) {
    return {};  // Empty range
  }

  // Boundary case optimization: Check if single-container range is local
  if (range_count == 1) {
    ContainerId container_id = range_offset;
    if (pool_manager->HasContainer(pool_id, container_id)) {
      // Container is local, resolve to Local query
      return {PoolQuery::Local()};
    }
    // Check if address_map_ maps this container to the local node
    u32 mapped_node = pool_manager->GetContainerNodeId(pool_id, container_id);
    if (mapped_node == GetNodeId()) {
      return {PoolQuery::Local()};
    }
    // Resolve to DirectId to preserve container info for retry-after-recovery
    return {PoolQuery::DirectId(container_id)};
  }

  std::vector<PoolQuery> result_queries;

  // Get neighborhood size from configuration (maximum number of queries)
  u32 neighborhood_size = config_manager->GetNeighborhoodSize();

  // Calculate queries needed, capped at neighborhood_size
  u32 ideal_queries = (range_count + neighborhood_size - 1) / neighborhood_size;
  u32 queries_to_create = std::min(ideal_queries, neighborhood_size);

  // Create one query per container
  if (queries_to_create <= 1) {
    queries_to_create = range_count;
  }

  u32 containers_per_query = range_count / queries_to_create;
  u32 remaining_containers = range_count % queries_to_create;

  u32 current_offset = range_offset;
  for (u32 i = 0; i < queries_to_create; ++i) {
    u32 current_count = containers_per_query;
    if (i < remaining_containers) {
      current_count++;  // Distribute remainder across first queries
    }

    if (current_count > 0) {
      result_queries.push_back(PoolQuery::Range(current_offset, current_count));
      current_offset += current_count;
    }
  }

  return result_queries;
}

std::vector<PoolQuery> IpcManager::ResolveBroadcastQuery(
    const PoolQuery &query, PoolId pool_id, const FullPtr<Task> &task_ptr) {
  auto *pool_manager = CLIO_POOL_MANAGER;
  if (pool_manager == nullptr) {
    return {query};  // Fallback to original query
  }

  // Get pool info to find the total number of containers
  const PoolInfo *pool_info = pool_manager->GetPoolInfo(pool_id);
  if (pool_info == nullptr || pool_info->num_containers_ == 0) {
    return {query};  // Fallback to original query
  }

  // Create a Range query that covers all containers, then resolve it
  PoolQuery range_query = PoolQuery::Range(0, pool_info->num_containers_);
  return ResolveRangeQuery(range_query, pool_id, task_ptr);
}

std::vector<PoolQuery> IpcManager::ResolvePhysicalQuery(
    const PoolQuery &query, PoolId pool_id, const FullPtr<Task> &task_ptr) {
  // Physical routing - query is already resolved to a specific node
  return {query};
}

ctp::ipc::FullPtr<Task> IpcManager::RecvRuntime(
    Future<Task> &future, Container *container, u32 method_id,
    ctp::lbm::Transport *recv_transport) {
  auto future_shm = future.GetFutureShm();

  // Self-send path: no deserialization needed
  if (!future_shm->flags_.Any(FutureShm::FUTURE_COPY_FROM_CLIENT) ||
      future_shm->flags_.Any(FutureShm::FUTURE_WAS_COPIED)) {
    return IpcCpu2Self::RuntimeRecv(future);
  }

  u32 origin = future_shm->origin_;
  switch (origin) {
#if CTP_ENABLE_CUDA || CTP_ENABLE_ROCM || CTP_ENABLE_SYCL
    case FutureShm::FUTURE_CLIENT_GPU2CPU:
      return IpcGpu2Cpu::RuntimeRecv(this, future, container,
                                      method_id, recv_transport);
#endif
    case FutureShm::FUTURE_CLIENT_SHM:
    default:
      return IpcCpu2Cpu::RuntimeRecv(this, future, container,
                                      method_id, recv_transport);
  }
}

void IpcManager::SendRuntime(
    const FullPtr<Task> &task_ptr, RunContext *run_ctx,
    Container *container, ctp::lbm::Transport *send_transport) {
  auto future_shm = run_ctx->future_.GetFutureShm();
  u32 origin = future_shm->origin_;

  switch (origin) {
    case FutureShm::FUTURE_CLIENT_SHM:
    default:
      IpcCpu2Cpu::RuntimeSend(this, task_ptr, run_ctx, container,
                               send_transport);
      break;
    case FutureShm::FUTURE_CLIENT_TCP:
    case FutureShm::FUTURE_CLIENT_IPC:
      IpcCpu2CpuZmq::EnqueueRuntimeSend(this, run_ctx, origin);
      break;
#if CTP_ENABLE_CUDA || CTP_ENABLE_ROCM || CTP_ENABLE_SYCL
    case FutureShm::FUTURE_CLIENT_GPU2CPU:
      IpcGpu2Cpu::RuntimeSend(this, task_ptr, run_ctx, container);
      break;
#endif
    // FUTURE_CLIENT_CPU2GPU dispatch was removed with the GPU runtime.
  }
}

}  // namespace clio::run