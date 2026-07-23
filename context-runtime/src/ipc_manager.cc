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
#include <filesystem>
#include <fstream>
#ifdef __linux__
#include <sys/mman.h>
#endif
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
CLIO_RUN_DEFINE_GLOBAL_PTR_VAR_CC(clio::run::IpcManager, g_ipc_manager);

namespace clio::run {

// Bind address for the local server sockets (client ROUTER, response listener,
// single-node host). CLIO_BIND_ADDR wins; otherwise loopback under
// CLIO_TEST_MODE so a unit-test binary binding a fresh port never trips the
// Windows Defender Firewall "Allow access?" prompt — every unit test is
// single-node loopback. Falls back to 0.0.0.0 for real multi-interface
// deployments. Keep all bind sites going through this so none silently default
// to 0.0.0.0 and accumulate per-binary firewall rules.
static std::string DefaultServerBindAddr() {
  if (const char *env = clio::run::env::GetCompat("BIND_ADDR")) {
    if (*env) return std::string(env);
  }
  if (const char *tm = clio::run::env::GetCompat("TEST_MODE")) {
    if (*tm && std::string(tm) != "0") return std::string("127.0.0.1");
  }
  return std::string("0.0.0.0");
}

// ChiServerBootstrap{Hip,Sycl}Gpu are defined in the GPU companion lib
// (clio_run_cxx_gpu) and called from ServerInit below. Declare them at
// namespace scope (not block scope) so MSVC mangles the references as
// clio::run::ChiServerBootstrap*Gpu to match the GPU lib's exports — a
// block-scope `extern` declaration binds to the global namespace on MSVC
// (it binds to the enclosing namespace on GCC, which is why it worked there).
CLIO_RUN_GPU_API bool ChiServerBootstrapHipGpu(IpcManager *self,
                                               u32 queue_depth,
                                               size_t backend_bytes);
CLIO_RUN_GPU_API bool ChiServerBootstrapSyclGpu(IpcManager *self,
                                                u32 queue_depth,
                                                size_t backend_bytes);

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
  if (const char *env = clio::run::env::GetCompat("ZMQ_LOCAL_IPC")) {
    return *env != '\0' && std::strcmp(env, "0") != 0;
  }
#ifdef __APPLE__
  return true;
#else
  return false;
#endif
}

// Unix-domain socket path for the local ZMQ ROUTER/DEALER. Distinct from the
// non-ZMQ SocketTransport endpoint (clio_<port>.ipc) so the two local
// servers never collide.
inline std::string LocalZmqIpcPath(u32 port) {
  return ctp::SystemInfo::GetMemfdPath(
      "clio_zmq_" + std::to_string(port + 3) + ".ipc");
}

}  // namespace

// Host struct methods

// IpcManager methods

// Constructor and destructor removed - handled by CTP singleton pattern

// Auto-select the fastest client IPC transport that is actually usable, when
// CLIO_IPC_MODE is unset. Probe order (fastest first, issue #768):
//
//   1. SHM  -- a same-host runtime always creates its main shared segment
//              (ServerInitShm is unconditional), so the segment existing means
//              a local server is up and the shared-memory data path is usable.
//   2. IPC  -- a unix-domain (AF_UNIX) socket the server bound for the local
//              control path; only present when the server itself runs in IPC
//              mode (or on macOS, issue #482). Probed by the socket file.
//   3. TCP  -- always available; the cross-host and last-resort fallback.
//
// The probes are cheap and side-effect free (an open/close on the segment
// handle; a filesystem stat on the socket path). An explicit CLIO_IPC_MODE
// bypasses this entirely.
IpcMode IpcManager::SelectBestIpcMode() {
  ConfigManager *config = CLIO_CONFIG_MANAGER;

  // 1. SHM: is the server's main segment present on this host?
  if (config) {
    std::string main_seg = config->GetSharedMemorySegmentName(kMainSegment);
    if (ctp::SystemInfo::SharedMemoryExists(main_seg)) {
      HLOG(kDebug, "SelectBestIpcMode: SHM segment '{}' present -> SHM",
           main_seg);
      return IpcMode::kShm;
    }
    HLOG(kDebug, "SelectBestIpcMode: SHM segment '{}' absent", main_seg);
  }

  // 2. IPC: did the server bind a local unix-domain socket?
  u32 port = GetEffectivePort();
  std::string ipc_path =
      ctp::SystemInfo::GetMemfdPath("clio_" + std::to_string(port) + ".ipc");
  std::error_code ec;
  if (std::filesystem::exists(ipc_path, ec) && !ec) {
    HLOG(kDebug, "SelectBestIpcMode: IPC socket '{}' present -> IPC", ipc_path);
    return IpcMode::kIpc;
  }
  HLOG(kDebug, "SelectBestIpcMode: IPC socket '{}' absent", ipc_path);

  // 3. TCP: always available.
  HLOG(kDebug, "SelectBestIpcMode: falling back to TCP");
  return IpcMode::kTcp;
}

bool IpcManager::ClientInit() {
  HLOG(kDebug, "IpcManager::ClientInit");
  if (is_initialized_) {
    return true;
  }
  // Optional Windows timer-resolution bump (CLIO_WIN_TIMER_MS, issue #768).
  ctp::SystemInfo::RequestTimerResolutionFromEnv();

  // Resolve the client IPC mode. An explicit CLIO_IPC_MODE forces that exact
  // transport (no probing); when unset, auto-select the fastest path that is
  // actually available. On this host a same-host SHM round-trip is ~190x
  // faster than TCP and ~5x faster than the unix-socket path (issue #768), so
  // defaulting to TCP left the slowest transport as the default.
  if (const char *ipc_mode_env = clio::run::env::GetCompat("IPC_MODE")) {
    std::string mode_str(ipc_mode_env);
    if (mode_str == "SHM" || mode_str == "shm") {
      ipc_mode_ = IpcMode::kShm;
    } else if (mode_str == "IPC" || mode_str == "ipc") {
      ipc_mode_ = IpcMode::kIpc;
    } else {
      ipc_mode_ = IpcMode::kTcp;
    }
    HLOG(kInfo, "IpcManager::ClientInit: IPC mode = {} (from CLIO_IPC_MODE)",
         ipc_mode_ == IpcMode::kShm   ? "SHM"
         : ipc_mode_ == IpcMode::kIpc ? "IPC"
                                      : "TCP");
  } else {
    ipc_mode_ = SelectBestIpcMode();
    HLOG(kInfo, "IpcManager::ClientInit: IPC mode = {} (auto-selected)",
         ipc_mode_ == IpcMode::kShm   ? "SHM"
         : ipc_mode_ == IpcMode::kIpc ? "IPC"
                                      : "TCP");
  }

  // Parse retry timeout environment variable
  // Semantics: 0 = fail immediately, -1 = wait forever, >0 = timeout in seconds
  const char *retry_env = clio::run::env::GetCompat("CLIENT_RETRY_TIMEOUT");
  if (retry_env) {
    client_retry_timeout_ = static_cast<float>(std::atof(retry_env));
  }
  HLOG(kInfo, "IpcManager::ClientInit: retry_timeout = {}s",
       client_retry_timeout_);

  // Parse CLIO_CLIENT_TRY_NEW_SERVERS environment variable
  const char *try_new_env = clio::run::env::GetCompat("CLIENT_TRY_NEW_SERVERS");
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
    // Effective port: a fallback client connects to the MAIN runtime's port.
    u32 port = GetEffectivePort();

    if (ipc_mode_ == IpcMode::kIpc || UseLocalZmqIpc()) {
      // Unix-domain SocketTransport for the client<->runtime control path.
      //
      // IPC mode always uses it. On macOS (issue #482) every other local mode
      // uses it too: the ZMQ ROUTER->DEALER reply path is broken cross-process
      // on macOS 14+ even over ipc:// (the in-process #490 ipc:// workaround did
      // not fix the cross-process case -- the daemon receives the request and
      // the reply reports no error but never reaches the client's DEALER). This
      // connection-oriented unix socket routes replies back over the same fd
      // with no identity routing, so it is unaffected; the kIpc transport tests
      // (cr_ipc_transport_ipc, cr_client_retry_*_ipc) pass on macOS 14+. Only
      // the control path moves -- ipc_mode_ is unchanged, so SHM mode keeps its
      // shared-memory data path and the mode assertions still hold.
      ctp::SystemInfo::EnsureMemfdDir();
      std::string ipc_path =
          ctp::SystemInfo::GetMemfdPath("clio_" + std::to_string(port) + ".ipc");
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

      // Decoupled response path: bind an ephemeral ROUTER on which this client
      // receives task responses, independent of the request DEALER above. Its
      // OS-assigned port is advertised to the runtime via client_port_ so the
      // runtime opens a dedicated dial-back connection (see RecvIn). This
      // keeps the client's RX off the request socket — no shared sock_mtx_
      // between send and recv.
      //
      // Constructed directly (not via the factory) for two reasons: (1) the
      // factory maps port 0 -> 8192, but we need a genuinely OS-assigned
      // ephemeral port so multiple clients on one host don't collide; (2) it
      // must bind on the shared, leaked-at-exit ZMQ context (use_shared_ctx)
      // — a ROUTER on its own owned context would zmq_ctx_destroy at clean exit
      // and abort on Windows (libzmq signaler WSASTARTUP assertion).
#if CTP_ENABLE_ZMQ
      try {
        client_response_listener_ = ctp::lbm::TransportPtr(
            new ctp::lbm::ZeroMqTransport(ctp::lbm::TransportMode::kServer,
                                          DefaultServerBindAddr(), "tcp",
                                          /*port=*/0,
                                          /*use_shared_ctx=*/true));
        client_response_port_ = client_response_listener_->GetBoundPort();
        HLOG(kInfo, "IpcManager: client response listener bound to port {}",
             client_response_port_);
      } catch (const std::exception &e) {
        HLOG(kError,
             "IpcManager::ClientInit: Failed to bind response listener: {}",
             e.what());
        return false;
      }
#endif
    }

    zmq_recv_running_.store(true);
    zmq_recv_thread_ = std::thread([this]() { RecvZmqClientThread(); });
  }

  // Initialize CTP TLS key for task counter before calling WaitForLocalServer,
  // which calls CreateTaskId(). Without the key registered first, GetTls() on
  // the zero-initialized key may return a stale/freed pointer → crash.
  // Guard against re-creating the global key: a fallback runtime brings up a
  // second IpcManager (client) inside a process that already created it.
  if (!chi_task_counter_key_created_) {
    CTP_THREAD_MODEL->CreateTls<TaskCounter>(chi_task_counter_key_, nullptr);
    chi_task_counter_key_created_ = true;
  }
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

    // Start the dedicated SHM response recv thread: it drains shm_out_server_
    // and wakes waiters (the SHM analogue of zmq_recv_thread_). App threads now
    // block on their EventManager in RecvOut instead of polling a per-thread
    // ring, so this thread MUST be running before any response can land.
    shm_recv_running_.store(true);
    shm_recv_thread_ = std::thread([this]() { RecvShmClientThread(); });
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
  // Optional Windows timer-resolution bump (CLIO_WIN_TIMER_MS, issue #768).
  ctp::SystemInfo::RequestTimerResolutionFromEnv();
  if (is_initialized_) {
    return true;
  }

  // CLIO_FORCE_NET (legacy CLIO_FORCE_NET also honored via GetCompat):
  // when set to anything non-empty, every task whose PoolQuery isn't
  // explicitly Local() is routed via the network path even on a
  // single-node deployment. Used by the bench to stress the ZMQ
  // serialize/send/recv loop without needing a real multi-node
  // setup.  Read once here; IsTaskLocal consults force_net_ on the
  // hot path.
  if (const char *env = clio::run::env::GetCompat("FORCE_NET")) {
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
  // construction. Source lives in src/gpu/gpu2cpu_init_hip.cc and is
  // compiled by nvcc/hipcc so the kernel launch syntax resolves.
  // (Device-aware memcpy is now ctp::DeviceAwareMemcpy in gpu_api.h.)
  {
    ConfigManager *config = CLIO_CONFIG_MANAGER;
    u32 queue_depth = config->GetQueueDepth();
    constexpr size_t kHipClientBackendBytes = 64 * 1024 * 1024;  // 64 MB
    if (!ChiServerBootstrapHipGpu(this, queue_depth,
                                   kHipClientBackendBytes)) {
      return false;
    }
  }
#elif CTP_ENABLE_SYCL
  // SYCL backend: same shape as the CUDA/HIP path above. Bootstrap
  // helper lives in clio_run_cxx_gpu (gpu2cpu_init_sycl.cc) — call
  // into it via a free function with normal linkage; both libraries
  // see the same IpcManager layout because CTP_ENABLE_SYCL=1 is set
  // on both.
  {
    ConfigManager *config = CLIO_CONFIG_MANAGER;
    u32 queue_depth = config->GetQueueDepth();
    constexpr size_t kSyclClientBackendBytes = 64 * 1024 * 1024;  // 64 MB
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
    HLOG(kDebug, "Node ID identified: {}", this_host_.node_id);
  }

  // Initialize CTP TLS key for task counter (needed for CreateTaskId in
  // runtime). Guarded so a later fallback-client bring-up in this process does
  // not re-create the global key.
  if (!chi_task_counter_key_created_) {
    CTP_THREAD_MODEL->CreateTls<TaskCounter>(chi_task_counter_key_, nullptr);
    chi_task_counter_key_created_ = true;
  }

  // Create scheduler using factory
  auto *config = CLIO_CONFIG_MANAGER;
  if (config && config->IsValid()) {
    std::string sched_name = config->GetLocalSched();
    scheduler_ = SchedulerFactory::Get(sched_name);
    HLOG(kDebug, "Scheduler initialized: {}", sched_name);
  }

  // ManyToOne collective batch/aggregation manager (leader-side).
  batch_manager_ = std::make_unique<BatchManager>(this);

  // Create lightbeam transports for client task reception
  {
    u32 port = config->GetPort();

    try {
      // TCP ROUTER server on port+3. Bind via DefaultServerBindAddr so it
      // honors CLIO_BIND_ADDR / loopback-under-test-mode and never defaults to
      // 0.0.0.0 (which trips the Windows Defender Firewall prompt on the ROUTER
      // port even when the main server is on loopback).
      std::string router_bind = DefaultServerBindAddr();
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
          ctp::SystemInfo::GetMemfdPath("clio_" + std::to_string(port) + ".ipc");
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
  // Mark shutdown so ZeroMqTransport leaks shared-context sockets instead of
  // zmq_close-ing them on Windows (avoids libzmq's signaler WSASTARTUP abort).
  ctp::lbm::sock::SetSocketLibShutdown();

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

  // Stop the SHM response recv thread and tear down the response ring. Join
  // BEFORE Shutdown so the thread cannot touch a detached segment.
#if CTP_IS_HOST
  if (shm_recv_running_.load()) {
    shm_recv_running_.store(false);
    // Wake it if parked so the join is prompt (see StopShmServerRecvThread).
    shm_out_server_.SignalConsumerIfParked();
    if (shm_recv_thread_.joinable()) {
      shm_recv_thread_.join();
    }
  }
  if (shm_out_server_ok_) {
    shm_out_server_.Shutdown();  // unmap + unlink "clio-<pid>-shm-out"
    shm_out_server_ok_ = false;
  }
#endif

  // Clean up lightbeam transport objects. The recv thread that reads the
  // response listener is already stopped above, so closing it here is safe.
  // NOTE (Windows): the listener is a ROUTER on the shared ZMQ context; at
  // process exit libzmq's static-destructor context shutdown touches it and
  // can abort in the signaler ("Successful WSASTARTUP not yet performed") —
  // the same teardown landmine the codebase bypasses via TerminateProcess in
  // tests. Closing vs leaking the listener does not change that, so we close
  // it cleanly (no leak on POSIX, where teardown is well-behaved).
  client_response_listener_.reset();
  ClearClientPool();
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

  // Close the persistent outbound DEALER sockets (run-to-run peer clients)
  // BEFORE destroying the owned server contexts below. On Windows, destroying
  // the last owned ZMQ context tears Winsock down (WSACleanup); a DEALER
  // closed afterwards trips libzmq's signaler assertion ("Successful WSASTARTUP
  // not yet performed", signaler.cpp) -> abort() -> the process wedges until
  // the ctest timeout. This first ClearTransports() runs while the owned
  // contexts (and thus Winsock) are still alive, so the DEALERs close cleanly;
  // the later ClearClientPool() in ServerFinalize is then a harmless no-op.
  ClearClientPool();

  local_transport_.reset();
  main_transport_.reset();
  client_tcp_transport_.reset();
  client_ipc_transport_.reset();
}

void IpcManager::ServerFinalize() {
  if (!is_initialized_) {
    return;
  }

  // Mark shutdown so ZeroMqTransport leaks shared-context sockets instead of
  // zmq_close-ing them on Windows (avoids libzmq's signaler WSASTARTUP abort).
  ctp::lbm::sock::SetSocketLibShutdown();

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

  // Tear down the single inbound SHM ring. Join its recv thread FIRST so it
  // cannot touch a detached segment (or push onto lanes whose workers are
  // already stopped). The admin ChiMod also stops it via a transport-shutdown
  // hook on the normal path; both call sites are idempotent.
#if CTP_IS_HOST
  StopShmServerRecvThread();
  if (shm_in_server_ok_) {
    for (auto &ring : shm_in_servers_) {
      if (ring) ring->Shutdown();  // unmap + unlink clio-<pid>-shm-in-<k>
    }
    shm_in_servers_.clear();
    shm_in_server_ok_ = false;
  }
#endif

  // Clear main allocator pointer
  main_allocator_ = nullptr;
  // issue #783: metadata segment is runtime-owned; clients only detach.
  metadata_allocator_ = nullptr;

  // Leak scan while the SHM segments are still mapped (alloc_vector_ is not
  // cleared here). This is the reliable trigger for the IpcManager leak scan:
  // the CLIO_IPC global is intentionally leaked, so ~IpcManager rarely runs.
  // No-op unless built with CTP_ALLOC_TRACK_SIZE (CLIO_CORE_ENABLE_LEAK_CHECK).
  ReportRuntimeLeaks("ServerFinalize");

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

void IpcManager::AwakenWorker(TaskLane *lane) {
  if (!lane) {
    // No lane to target — wake every worker so a task parked with no resolvable
    // owning lane is still re-checked (lost-wakeup safety net).
    HLOG(kWarning, "AwakenWorker: lane is null; waking all workers");
    CLIO_WORK_ORCHESTRATOR->AwakenAllWorkers();
    return;
  }

  // ALWAYS send SIGUSR1, never skip on active_=true. Past attempts to
  // gate this on the park-flag tripped a lost-wakeup race at scale (4n
  // 256m FPP) where the producer observed active_=true, skipped the
  // signal, and the worker then stored active_=false and entered
  // epoll_pwait2 before noticing the just-pushed task. The
  // post-store-recheck handshake in Worker::SuspendMe is supposed to
  // catch this but doesn't fire reliably under heavy multi-tier
  // scheduling pressure. Skipping the tgkill saved a syscall; the
  // observed cost was hangs that never recovered. The extra signal is
  // absorbed harmlessly by signalfd — at worst the worker wakes one
  // extra time and re-checks its (empty) queue. Worth it.

  int tid = lane->GetTid();
  if (tid > 0) {
    int runtime_pid = runtime_pid_ ? runtime_pid_ : ctp::SystemInfo::GetPid();

    // Send SIGUSR1 to the worker thread in the runtime process
    int result = ctp::lbm::EventManager::Signal(runtime_pid, tid);
    if (result != 0) {
      HLOG(kError,
           "AwakenWorker: Failed to send SIGUSR1 to runtime_pid={}, tid={} "
           "(active={}) - errno={}",
           runtime_pid, tid, lane->IsActive(), errno);
    }
  } else {
    // The target lane has no worker tid (only a worker's OWN assigned_lane_
    // ever gets a tid, so a task parked on any secondary lane reads tid==0).
    // A targeted signal is impossible, but some worker DOES own this task's
    // event queue, so wake them all and let the owner re-check and resume the
    // parked parent. This closes a lost-wakeup that hung sustained O_APPEND
    // writes (#680 generic/069): a completed PutBlob subtask emplaced its
    // result on the parent WriteTask's event queue but could not signal, so
    // the parent slept forever while all workers idled in epoll.
    CLIO_WORK_ORCHESTRATOR->AwakenAllWorkers();
  }
}

IpcManagerTls *IpcManager::GetTls() {
  // One-time key registration (double-checked under the mutex). The key is
  // process-wide; the per-thread value below is what differs per thread.
  if (!ipc_tls_key_created_.load(std::memory_order_acquire)) {
    std::lock_guard<std::mutex> lk(ipc_tls_key_mutex_);
    if (!ipc_tls_key_created_.load(std::memory_order_relaxed)) {
      CTP_THREAD_MODEL->CreateTls<IpcManagerTls>(ipc_tls_key_, nullptr);
      ipc_tls_key_created_.store(true, std::memory_order_release);
    }
  }
  // Lazily allocate this thread's IpcManagerTls. Its EventManager ctor runs on
  // THIS thread, registering this thread's (pid, tid) signal event.
  IpcManagerTls *tls = CTP_THREAD_MODEL->GetTls<IpcManagerTls>(ipc_tls_key_);
  if (tls == nullptr) {
    tls = new IpcManagerTls();
    CTP_THREAD_MODEL->SetTls(ipc_tls_key_, tls);
  }
  return tls;
}

ctp::lbm::ShmMpscTransport *IpcManager::GetOrCreateShmConn(
    const std::string &name) {
  std::lock_guard<std::mutex> lk(shm_conns_mutex_);
  auto it = shm_conns_.find(name);
  if (it != shm_conns_.end()) {
    return it->second.get();
  }
  auto conn = std::make_unique<ctp::lbm::ShmMpscTransport>();
  if (!conn->ClientInit(name)) {
    return nullptr;
  }
  ctp::lbm::ShmMpscTransport *raw = conn.get();
  shm_conns_[name] = std::move(conn);
  return raw;
}

/**
 * Memory limit imposed on this process by its cgroup, or 0 when unlimited or
 * unreadable (cgroup v2 first, then v1).
 *
 * Needed because SystemInfo::GetRamCapacity() reports the HOST's physical
 * memory even inside a container -- sizing a shared-memory reservation off
 * that number over-commits badly in CI images and constrained deployments.
 */
static size_t ReadCgroupMemoryLimit() {
#ifdef __linux__
  const char *paths[] = {"/sys/fs/cgroup/memory.max",
                         "/sys/fs/cgroup/memory/memory.limit_in_bytes"};
  for (const char *path : paths) {
    std::ifstream f(path);
    if (!f.is_open()) {
      continue;
    }
    std::string v;
    if (!(f >> v)) {
      continue;
    }
    if (v == "max") {
      return 0;  // explicitly unlimited
    }
    try {
      unsigned long long n = std::stoull(v);
      // cgroup v1 reports a sentinel near SIZE_MAX when unlimited.
      if (n > 0 && n < (1ULL << 62)) {
        return static_cast<size_t>(n);
      }
    } catch (...) {
    }
  }
#endif
  return 0;
}

/**
 * Keep a large shared mapping out of core dumps (Linux MADV_DONTDUMP).
 *
 * The runtime shuts down via std::abort() (admin_runtime.cc), so every clean
 * `clio_run stop` raises SIGABRT and the kernel dumps core wherever cores are
 * enabled. With a multi-gigabyte metadata segment mapped that dump takes long
 * enough to blow the 60s shutdown budget in the CLI lifecycle tests
 * (cr_cli_runtime_command_tests, cr_cli_cte_wal_restart, cr_cli_compose_restart
 * all failed this way on docker AND ubuntu-arm). It never reproduced locally
 * because this dev box has `ulimit -c 0`, so no core is written at all.
 *
 * The segment is a cache; its contents have no diagnostic value in a core, so
 * excluding it costs nothing and removes gigabytes of dump work.
 */
static void ExcludeFromCoreDump(void *addr, size_t size, const char *what) {
#if defined(__linux__) && defined(MADV_DONTDUMP)
  if (addr == nullptr || size == 0) {
    return;
  }
  if (madvise(addr, size, MADV_DONTDUMP) != 0) {
    HLOG(kWarning, "MADV_DONTDUMP failed for {} ({} bytes): {}", what, size,
         strerror(errno));
  } else {
    HLOG(kInfo, "{}: excluded {} bytes from core dumps", what, size);
  }
#else
  (void)addr;
  (void)size;
  (void)what;
#endif
}

bool IpcManager::ServerInitShm() {
  ConfigManager *config = CLIO_CONFIG_MANAGER;

  try {
    // Allocator IDs are based on this runtime's pid so that multiple runtimes
    // on one node (the fallback topology) own globally-distinct allocators
    // instead of every runtime claiming (1,0)/(2,0). Convention: pid.1 = main,
    // pid.2 = queue. Clients learn these dynamically via ClientConnect.
    u32 pid = static_cast<u32>(ctp::SystemInfo::GetPid());
    main_allocator_id_ = ctp::ipc::AllocatorId::Get(pid, 1);

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
    queue_allocator_id_ = ctp::ipc::AllocatorId::Get(pid, 2);
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

    // issue #783: runtime-wide metadata segment (pid.3). Backs the CTE
    // shared-memory metadata cache. Deliberately NON-FATAL: if it cannot be
    // created the runtime still starts and every client simply falls back to
    // the RPC path. That matters because the reservation is huge (8 GB by
    // default) and a host with a small /dev/shm can legitimately refuse it --
    // losing an optimization is acceptable, failing to boot is not.
    metadata_allocator_id_ = ctp::ipc::AllocatorId::Get(pid, 3);
    std::string metadata_segment_name =
        config->GetSharedMemorySegmentName(kMetadataSegment);
    size_t metadata_segment_size = config->CalculateMetadataSegmentSize();

    // Sanity-clamp the reservation against the memory this process may
    // actually use -- which is NOT the same as physical RAM in a container.
    //
    // SystemInfo::GetRamCapacity() reads sysinfo().totalram, i.e. the HOST's
    // memory, and knows nothing about cgroup limits. Inside the docker CI
    // image that made this reserve half the RUNNER's RAM, several times the
    // container's own limit, which showed up as daemon lifecycle timeouts
    // (cr_cli_runtime_command_tests, cr_cli_cte_wal_restart,
    // cr_cli_compose_restart) in a job that is green on dev. Read the cgroup
    // limit first and fall back to physical RAM only when there is none.
    // Sanity-clamp the reservation against physical RAM.
    //
    // NOTE ON WHAT BACKS THIS: on Linux these segments are memfd_create()
    // objects (SystemInfo::CreateNewSharedMemory), NOT files under /dev/shm.
    // memfd lives in the kernel's internal shmem pool, bounded by RAM+swap and
    // the memory cgroup -- the /dev/shm mount size is irrelevant here, which is
    // why a 1 GB main segment works fine on a host whose /dev/shm is 64 MB.
    // Do not "fix" this by measuring std::filesystem::space("/dev/shm"); that
    // reads an unrelated filesystem and needlessly shrinks the cache.
    //
    // The reservation itself is nearly free because the segment is sparse and
    // never pre-faulted. The hazard is only the LIVE SET: pages actually
    // written come out of RAM, and exhausting shmem surfaces as SIGBUS or the
    // OOM killer rather than a clean allocation failure. Clamping to half of
    // RAM keeps the default a no-op on real nodes while protecting small
    // dev boxes and constrained containers.
    {
      size_t budget = ctp::SystemInfo::GetRamCapacity();
      size_t cgroup_limit = ReadCgroupMemoryLimit();
      if (cgroup_limit > 0 && (budget == 0 || cgroup_limit < budget)) {
        budget = cgroup_limit;
      }
      if (budget > 0 && metadata_segment_size > budget / 2) {
        size_t clamped = budget / 2;
        HLOG(kWarning,
             "Metadata segment: requested {} bytes exceeds half the memory "
             "budget ({} bytes; cgroup_limit={}); clamping to {} bytes",
             metadata_segment_size, budget, cgroup_limit, clamped);
        metadata_segment_size = clamped;
      }
    }

    HLOG(kInfo,
         "Initializing metadata shared memory segment: {} bytes ({} MB), "
         "sparse/not pre-faulted",
         metadata_segment_size, metadata_segment_size / (1024 * 1024));
    if (metadata_backend_.shm_init(
            metadata_allocator_id_,
            ctp::Unit<size_t>::Bytes(metadata_segment_size),
            metadata_segment_name)) {
      metadata_allocator_ = metadata_backend_.MakeAlloc<CLIO_TASK_ALLOC_T>();
    }
    if (!metadata_allocator_) {
      HLOG(kWarning,
           "ServerInitShm: metadata segment '{}' unavailable ({} bytes) -- "
           "SHM metadata caching disabled, clients will use the RPC path",
           metadata_segment_name, metadata_segment_size);
    } else {
      ExcludeFromCoreDump(metadata_backend_.region_,
                          metadata_backend_.backend_size_,
                          "metadata segment");
      // Publish the directory of well-known roots as the FIRST object in the
      // segment. Clients learn its offset from ClientConnect, which is what
      // makes discovery independent of who created a pool first.
      try {
        auto dir_fp =
            metadata_allocator_->template Allocate<MetadataDirectory>(
                sizeof(MetadataDirectory));
        if (!dir_fp.IsNull()) {
          auto *dir = dir_fp.ptr_;
          std::memset(dir, 0, sizeof(MetadataDirectory));
          dir->version_ = MetadataDirectory::kVersion;
          metadata_dir_off_ = static_cast<u64>(
              reinterpret_cast<char *>(dir) -
              reinterpret_cast<char *>(metadata_allocator_));
          HLOG(kInfo, "ServerInitShm: metadata directory at offset {}",
               metadata_dir_off_);
        }
      } catch (const std::exception &e) {
        HLOG(kWarning, "ServerInitShm: metadata directory alloc failed: {}",
             e.what());
      }
    }

    // Reserve segment indices 0-3 of this pid's allocator-id space: index 1 is
    // the main segment, index 2 the queue segment, index 3 the metadata
    // segment (see above). Runtime data segments created on demand by
    // IncreaseClientShm (for AllocateBuffer / FutureShm) start at index 4 so
    // their AllocatorId never collides. Clients use a different pid, so they
    // keep starting at 0.
    shm_count_.store(4, std::memory_order_relaxed);

#if CTP_IS_HOST
    // Single inbound MPSC ring shared by every SHM client (replaces the old
    // per-worker "clio-<pid>-<tid>" servers). Clients SendBytes serialized tasks
    // here; the dedicated RecvShmServerThread drains it (-> IpcCpu2Cpu::RecvIn)
    // and fans tasks out to worker lanes, mirroring how the ZMQ path funnels
    // all inbound client traffic through one dedicated recv thread. Sized
    // large (~128MB) so bursty async fan-out from many clients rarely blocks a
    // producer in SendBytes. The name is derived from the runtime pid so a
    // client can form it from the server_pid_ it learns via ClientConnect.
    {
      // issue #807: S parallel inbound rings clio-<pid>-shm-in-<k>, each drained
      // by its own thread, replacing the single ring. Spreads MPSC-tail
      // contention and deserialize+route work across cores. Each ring is sized
      // down proportionally so total inbound SHM stays ~128MB regardless of S.
      // issue #807: 0 = auto = one shard per worker (default). Worker w owns
      // shard w, so cap at the worker count — a shard beyond the last worker
      // would have no consumer and its clients would hang.
      u32 nworkers = CLIO_CONFIG_MANAGER->GetNumThreads();
      u32 shards = CLIO_CONFIG_MANAGER->GetShmInShards();
      if (shards == 0 || shards > nworkers) shards = nworkers;
      if (shards < 1) shards = 1;
      shm_in_shards_ = shards;
      size_t per_ring_mb = std::max<size_t>(8, 128 / shards);
      shm_in_servers_.clear();
      shm_in_server_ok_ = true;
      for (u32 k = 0; k < shards; ++k) {
        std::string in_name =
            "clio-" + std::to_string(pid) + "-shm-in-" + std::to_string(k);
        auto ring = std::make_unique<ctp::lbm::ShmMpscTransport>();
        if (!ring->ServerInit(in_name,
                              ctp::Unit<size_t>::Megabytes(per_ring_mb))) {
          HLOG(kError, "ServerInitShm: failed to create inbound SHM ring '{}'",
               in_name);
          shm_in_server_ok_ = false;
          return false;
        }
        shm_in_servers_.push_back(std::move(ring));
      }
      HLOG(kInfo, "ServerInitShm: {} inbound SHM ring(s) '{}-shm-in-0..{}' ({}MB each)",
           shards, "clio-" + std::to_string(pid), shards - 1, per_ring_mb);
    }
#endif

    return true;
  } catch (const std::exception &e) {
    return false;
  }
}

bool IpcManager::ClientInitShm() {
  ConfigManager *config = CLIO_CONFIG_MANAGER;

  try {
    // Allocator IDs are NOT hardcoded: the server's are pid-based (pid.1 main,
    // pid.2 queue) and differ per runtime. They are recovered from each
    // segment's header on attach (and also reported by ClientConnect).

    // Get configurable segment names with environment variable expansion.
    std::string main_segment_name =
        config->GetSharedMemorySegmentName(kMainSegment);
    std::string queue_segment_name =
        config->GetSharedMemorySegmentName(kQueueSegment);

    // Attach to existing main shared memory segment created by server
    if (!main_backend_.shm_attach(main_segment_name)) {
      HLOG(kError, "ClientInitShm: shm_attach(main='{}') failed",
           main_segment_name);
      return false;
    }

    // Attach to main allocator (CLIO_TASK_ALLOC_T = BuddyAllocator)
    main_allocator_ = main_backend_.AttachAlloc<CLIO_TASK_ALLOC_T>();
    if (!main_allocator_) {
      HLOG(kError, "ClientInitShm: AttachAlloc(main='{}') failed",
           main_segment_name);
      return false;
    }
    // Recover the server's actual (pid-based) allocator id from the header.
    main_allocator_id_ = main_allocator_->GetId();

    // Attach to queue segment (CLIO_QUEUE_ALLOC_T = ArenaAllocator)
    if (!queue_backend_.shm_attach(queue_segment_name)) {
      HLOG(kError, "ClientInitShm: shm_attach(queue='{}') failed",
           queue_segment_name);
      return false;
    }
    queue_allocator_ = queue_backend_.AttachAlloc<CLIO_QUEUE_ALLOC_T>();
    if (!queue_allocator_) {
      HLOG(kError, "ClientInitShm: AttachAlloc(queue='{}') failed",
           queue_segment_name);
      return false;
    }
    queue_allocator_id_ = queue_allocator_->GetId();

    // issue #783: attach the runtime-wide metadata segment for SHM metadata
    // caching. BEST-EFFORT -- unlike main/queue, a failure here is not fatal.
    // The segment is an optimization; a client that cannot attach it just uses
    // the RPC path. It is also legitimately absent when the runtime chose not
    // to create it (small /dev/shm), so this must not log at error level.
    //
    // Mapped read-write, not PROT_READ: acquiring a lease is a store. Clients
    // are trusted by convention to write ONLY lock words, never metadata --
    // the runtime never treats this segment as authoritative, so a
    // misbehaving client can degrade other clients but cannot corrupt the
    // runtime's own state.
    {
      std::string metadata_segment_name =
          config->GetSharedMemorySegmentName(kMetadataSegment);
      if (metadata_backend_.shm_attach(metadata_segment_name)) {
        metadata_allocator_ = metadata_backend_.AttachAlloc<CLIO_TASK_ALLOC_T>();
      }
      if (metadata_allocator_) {
        metadata_allocator_id_ = metadata_allocator_->GetId();
        HLOG(kInfo, "ClientInitShm: attached metadata segment '{}'",
             metadata_segment_name);
      } else {
        HLOG(kInfo,
             "ClientInitShm: metadata segment '{}' unavailable -- SHM metadata "
             "caching disabled for this client (RPC path still works)",
             metadata_segment_name);
      }
    }

#if CTP_IS_HOST
    // Single per-process response ring (replaces the old per-thread
    // "clio-<pid>-<tid>" servers). The runtime routes every response for this
    // process here (IpcCpu2Cpu::SendOut -> "clio-<client_pid>-shm-out"); the
    // dedicated RecvShmClientThread drains it and wakes waiters. ~1MB is ample
    // for the response stream from one runtime, continuously drained.
    {
      std::string out_name =
          "clio-" + std::to_string(ctp::SystemInfo::GetPid()) + "-shm-out";
      shm_out_server_ok_ =
          shm_out_server_.ServerInit(out_name, ctp::Unit<size_t>::Megabytes(1));
      if (!shm_out_server_ok_) {
        HLOG(kError, "ClientInitShm: failed to create response SHM ring '{}'",
             out_name);
        return false;
      }
      HLOG(kInfo, "ClientInitShm: response SHM ring '{}' (1MB)", out_name);
    }
#endif

    return true;
  } catch (const std::exception &e) {
    HLOG(kError, "ClientInitShm: exception: {}", e.what());
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
    // issue #785: reserve lanes for elastic replacements as well — see
    // ConfigManager::GetElasticLaneHeadroom. Sizing in
    // CalculateQueueSegmentSize accounts for the same headroom.
    u32 total_workers = thread_count + config->GetElasticLaneHeadroom();

    // Store worker count and scheduling queue count
    num_workers_ = thread_count;
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
  const char *wait_env = clio::run::env::GetCompat("WAIT_SERVER");
  if (wait_env != nullptr) {
    wait_server_timeout_ = static_cast<float>(std::atof(wait_env));
  }

  HLOG(kInfo, "Waiting for runtime via lightbeam (timeout={}s)",
       wait_server_timeout_);

  // 0 = don't wait at all
  if (wait_server_timeout_ == 0) {
    HLOG(kError, "CLIO_WAIT_SERVER=0: not waiting for runtime");
    return false;
  }

  // At scale (>=64 clio daemons) the daemon's local 9416 ROUTER's I/O
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
  auto future = IpcCpu2CpuZmq::SendIn(this,task, ipc_mode_);

  // Wait for response with per-attempt timeout
  if (!future.Wait(per_attempt)) {
    task.reset();
    float elapsed = std::chrono::duration<float>(
        std::chrono::steady_clock::now() - attempt_start).count();
    if (total_timeout > 0 && elapsed >= total_timeout) {
      HLOG(kError, "Timeout waiting for runtime after {} seconds ({} attempts)",
           wait_server_timeout_, attempt_idx);
      HLOG(kError, "This usually means:");
      HLOG(kError, "1. Clio runtime is not running");
      HLOG(kError, "2. Runtime failed to start");
      HLOG(kError, "3. Network connectivity issues");
      return false;
    }
    HLOG(kWarning, "Attempt {} timed out after {}s; recreating DEALER",
         attempt_idx, per_attempt);
    if (ipc_mode_ == IpcMode::kTcp) {
      auto *config = CLIO_CONFIG_MANAGER;
      // Effective port: a fallback client recreates its DEALER to the MAIN port.
      u32 port = GetEffectivePort();
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
          // Mirror ClientInit: on macOS the local control path uses the unix
          // SocketTransport (not ZMQ), so recreate it the same way (issue #482).
          ctp::SystemInfo::EnsureMemfdDir();
          std::string ipc_path = ctp::SystemInfo::GetMemfdPath(
              "clio_" + std::to_string(port) + ".ipc");
          zmq_transport_ = ctp::lbm::TransportFactory::Get(
              ipc_path, ctp::lbm::TransportType::kSocket,
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
    worker_tids_.assign(task->worker_tids_,
                        task->worker_tids_ + task->num_worker_tids_);
    // Adopt the runtime's dynamic (pid-based) allocator ids rather than
    // assuming (1,0)/(2,0).
    main_allocator_id_ = task->main_alloc_id_;
    queue_allocator_id_ = task->queue_alloc_id_;
    // issue #783: learn where the metadata-segment directory lives so this
    // client can find module cache roots without depending on having been the
    // process that created the pool.
    metadata_dir_off_ = task->metadata_dir_off_;
    if (task->server_pid_ > 0) {
      runtime_pid_ = static_cast<int>(task->server_pid_);
    }
    // issue #807: learn how many inbound SHM rings to shard requests across.
    if (task->shm_in_shards_ >= 1) {
      shm_in_shards_ = task->shm_in_shards_;
    }
    HLOG(kInfo,
         "Successfully connected to runtime (generation={}, server_pid={}, "
         "shm_in_shards={})",
         client_generation_, runtime_pid_, shm_in_shards_);

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
    auto future = IpcCpu2CpuZmq::SendIn(this,task, ipc_mode_);

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
    std::string bind_addr = DefaultServerBindAddr();
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

u64 IpcManager::GetNeighborhoodLeaderNodeId(u64 for_node) const {
  // The neighborhood is the window [base, base + N) where N is the configured
  // neighborhood size and base = floor(for_node / N) * N. The leader is the
  // lowest *alive* node in that window. Computed from local SWIM membership,
  // so every node in the neighborhood agrees deterministically. Falls back to
  // for_node itself if the window has no other alive members.
  u32 n = CLIO_CONFIG_MANAGER->GetNeighborhoodSize();
  if (n == 0) {
    n = 1;
  }
  u64 base = (for_node / n) * n;
  u64 end = base + n;
  u64 leader = std::numeric_limits<u64>::max();
  for (const auto &[id, host] : hostfile_map_) {
    if (host.state == NodeState::kAlive && host.node_id >= base &&
        host.node_id < end && host.node_id < leader) {
      leader = host.node_id;
    }
  }
  return (leader == std::numeric_limits<u64>::max()) ? for_node : leader;
}

u64 IpcManager::GetNeighborhoodLeaderNodeId() const {
  return GetNeighborhoodLeaderNodeId(GetNodeId());
}

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
  HLOG(kError, "To stop the Clio runtime, run:");
  HLOG(kError, "  clio runtime stop");
  HLOG(kError, "");
  HLOG(kError, "Or kill the process directly:");
  HLOG(kError, "  pkill -9 clio");
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
    HLOG(kError, "Failed to start main server on {}:{} - exception: {}",
         hostname, config->GetPort(), e.what());
    return false;
  } catch (...) {
    HLOG(kError, "Failed to start main server on {}:{} - unknown exception",
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

size_t IpcManager::GetRuntimeHeapAllocatedBytes() const {
#if CTP_IS_HOST
  // CTP_MALLOC is the private heap backing NewObj and the client-ZMQ
  // AllocateBuffer path. GetCurrentlyAllocatedSize() returns 0 unless built
  // with CTP_ALLOC_TRACK_SIZE (CLIO_CORE_ENABLE_LEAK_CHECK).
  // Tasks are carved from CTP_MALLOC (ctp::make_shared(CTP_MALLOC, ...) in
  // NewTask) and freed by RAII when the last shared_ptr owner drops, so a leaked
  // task is already visible in GetCurrentlyAllocatedSize() below. (The former
  // RuntimeTaskAllocBytes side-counter — needed when NewTask used global
  // operator new — was removed with DelTask; keeping its now-undecremented
  // increment made every freed task look leaked.)
  size_t total = CTP_MALLOC->GetCurrentlyAllocatedSize();
#if defined(CTP_ALLOC_TRACK_SIZE)
  // The runtime's AllocateBuffer draws from the per-process SHM
  // MultiProcessAllocator segments (not CTP_MALLOC), so add their outstanding
  // bytes too — otherwise a leaked runtime buffer is invisible here.
  {
    std::lock_guard<std::mutex> lock(shm_mutex_);
    for (auto *alloc : alloc_vector_) {
      if (alloc != nullptr) {
        total += alloc->GetCurrentlyAllocatedSize();
      }
    }
  }
#endif
  return total;
#else
  return 0;
#endif
}

size_t IpcManager::ReportRuntimeLeaks(const char *phase) const {
#if defined(CTP_ALLOC_TRACK_SIZE) && CTP_IS_HOST
  // Private heap (NewTask/NewObj/client-ZMQ buffers). At shutdown this
  // legitimately still holds process-lifetime runtime state (pools, config,
  // module manager) that is only released at static teardown -- so report it at
  // INFO, NOT as a leak, to avoid false positives. Genuinely unfreed CTP_MALLOC
  // allocations are caught for real by the MallocAllocator destructor
  // (ctp::ipc::AllocatorLeakChecker) which runs at static teardown, after that
  // process-lifetime state is gone.
  const size_t priv = CTP_MALLOC->GetCurrentlyAllocatedSize();
  if (priv != 0) {
    HLOG(kInfo,
         "[leak][runtime] {}: CTP_MALLOC private heap holds {} bytes (may be "
         "process-lifetime state; verified clean at static teardown)",
         phase, priv);
  }

  // Per-process SHM segments the runtime's AllocateBuffer draws from. These
  // MultiProcessAllocators are placement-constructed in shared memory and never
  // get a C++ destructor, so this scan is the ONLY place their leaks surface.
  // Every buffer MUST be freed by shutdown, so any outstanding bytes here are a
  // real shared-memory leak (reported at ERROR). This is the return value.
  size_t shm_leaked = 0;
  {
    std::lock_guard<std::mutex> lock(shm_mutex_);
    for (size_t i = 0; i < alloc_vector_.size(); ++i) {
      auto *alloc = alloc_vector_[i];
      if (alloc == nullptr) {
        continue;
      }
      const size_t out = alloc->GetCurrentlyAllocatedSize();
      if (out != 0) {
        shm_leaked += out;
        HLOG(kError,
             "[leak][runtime] {}: SHM allocator #{} leaked {} bytes "
             "(outstanding at shutdown)",
             phase, i, out);
      }
    }
  }

  if (shm_leaked == 0) {
    HLOG(kInfo, "[leak][runtime] {}: no outstanding SHM buffers", phase);
  } else {
    HLOG(kError, "[leak][runtime] {}: {} total SHM bytes leaked", phase,
         shm_leaked);
  }
  return shm_leaked;
#else
  (void)phase;
  return 0;
#endif
}

IpcManager::~IpcManager() {
#if defined(CTP_ALLOC_TRACK_SIZE) && CTP_IS_HOST
  ReportRuntimeLeaks("~IpcManager");
#endif
}

FullPtr<char> IpcManager::AllocateBuffer(size_t size) {
#if CTP_IS_HOST
  // HOST-ONLY PATH: The device implementation is in ipc_manager.h

  // RUNTIME PATH: draw from per-process shared-memory segments (same growable
  // MultiProcessAllocator strategy as the SHM client below), NOT CTP_MALLOC.
  // Buffers and FutureShm allocated here must be resolvable from another
  // process so a task this runtime punts to its fallback (main) can have its
  // FutureShm completed IN PLACE by main. MultiProcessAllocator gives each
  // worker thread its own lock-free block (like malloc's per-thread arenas), so
  // unlike the single-lock main BuddyAllocator it absorbs the runtime's
  // high-concurrency churn without contending. The runtime falls through to the
  // shared steps 1-4 (it is the SHM owner; IsRuntime() gates registration in
  // IncreaseClientShm so it does not ClientSend to itself).
  const bool is_runtime =
      (CLIO_RUNTIME_MANAGER != nullptr) && CLIO_RUNTIME_MANAGER->IsRuntime();

  // CLIENT TCP/IPC PATH: Use private memory (no shared memory needed). The
  // runtime always uses the SHM path regardless of ipc_mode_.
  if (!is_runtime && ipc_mode_ != IpcMode::kShm) {
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

  // Create new persistent client connection. TransportFactory::Get throws
  // (std::runtime_error) when the address is unroutable; a malformed client
  // identity must never terminate the whole runtime, so swallow it and return
  // nullptr — the caller falls back to echoing the response over the inbound
  // ROUTER (see IpcCpu2CpuZmq::RecvIn).
  HLOG(kInfo, "[ClientPool] Creating new persistent connection to {}", key);
  ctp::lbm::TransportPtr transport;
  try {
    transport = ctp::lbm::TransportFactory::Get(
        addr, ctp::lbm::TransportType::kZeroMq,
        ctp::lbm::TransportMode::kClient, "tcp", port);
  } catch (const std::exception &e) {
    HLOG(kError, "[ClientPool] Failed to dial {}: {}", key, e.what());
    return nullptr;
  }

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

ctp::lbm::Transport *IpcManager::GetOrCreateClientByIdentity(
    const std::string &key_id, const std::string &dial_addr, int port) {
  // Cache key: routing identity + advertised response port. Two clients that
  // happen to pick the same ephemeral port still differ by identity, and the
  // same client reusing a port across reconnects re-resolves to a fresh dial.
  size_t hkey = std::hash<std::string>{}(key_id + ":" + std::to_string(port));

  // Fast path: already have a dial-back connection for this client.
  if (ctp::lbm::Transport **found = client_conn_cache_.find(hkey)) {
    return *found;
  }

  // Miss: open (and own, via client_pool_) a DEALER to the client's listener.
  ctp::lbm::Transport *transport = GetOrCreateClient(dial_addr, port);
  if (transport == nullptr) {
    HLOG(kError, "[ConnCache] Failed to dial back to {}:{} (id={})", dial_addr,
         port, key_id);
    return nullptr;
  }
  // insert_or_assign is idempotent under a race: whichever thread lands second
  // just overwrites with the same client_pool_-owned pointer.
  client_conn_cache_.insert_or_assign(hkey, transport);
  HLOG(kDebug, "[ConnCache] dial-back to {}:{} cached (id={}, key={})",
       dial_addr, port, key_id, hkey);
  return transport;
}

void IpcManager::ClearClientPool() {
  std::lock_guard<std::mutex> lock(client_pool_mutex_);
  HLOG(kInfo, "[ClientPool] Clearing {} persistent connections",
       client_pool_.size());
  client_conn_cache_.clear();
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
    if (wake_lane) {
      AwakenWorker(wake_lane);
    }
    HLOG(kDebug,
         "[TRACE768] t={} EnqueueNetTask prio={} was_empty={} wake_lane={} "
         "lane_tid={} send_lane_set={} recv_lane_set={} net_lane_set={}",
         std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(), priority_idx, was_empty, wake_lane != nullptr,
         wake_lane ? wake_lane->GetTid() : -1, net_send_lane_ != nullptr,
         net_recv_lane_ != nullptr, net_lane_ != nullptr);
  } else {
    HLOG(kDebug, "[TRACE768] t={} EnqueueNetTask prio={} was_empty=FALSE (no wake)",
         std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(), priority_idx);
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

  // Create shared memory name: clio_{pid}_{index}
  std::string shm_name =
      "clio_" + std::to_string(pid) + "_" + std::to_string(index);

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

    // RUNTIME PATH: the server owns this segment and already inserted it into
    // its own alloc_map_ above, so it resolves its own buffers/FutureShm
    // directly — there is no server to ask and no client DEALER to send on (a
    // self-directed ClientSend().Wait() would block forever).
    if (CLIO_RUNTIME_MANAGER && CLIO_RUNTIME_MANAGER->IsRuntime()) {
      return true;
    }

    // CLIENT PATH: tell the runtime server to attach to this new shared memory
    // segment. Use kAdminPoolId directly (not admin_client->pool_id_) because
    // the admin client may not be initialized yet during ClientInit.
    auto reg_task = NewTask<clio::run::admin::RegisterMemoryTask>(
        clio::run::CreateTaskId(), clio::run::kAdminPoolId, clio::run::PoolQuery::Local(),
        alloc_id);
    IpcCpu2CpuZmq::SendIn(this,reg_task, IpcMode::kTcp).Wait();

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

  // Derive shm_name from alloc_id: clio_{pid}_{index}
  int owner_pid = static_cast<int>(alloc_id.major_);
  u32 shm_index = alloc_id.minor_;
  std::string shm_name =
      "clio_" + std::to_string(owner_pid) + "_" + std::to_string(shm_index);

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

bool IpcManager::TryLazyRegisterClientSegment(
    const ctp::ipc::AllocatorId &alloc_id) {
#if CTP_IS_HOST
  // Runtime only: clients never resolve other processes' segments, and the
  // attach below is the server-side registration RegisterMemory performs.
  if (CLIO_RUNTIME_MANAGER == nullptr || !CLIO_RUNTIME_MANAGER->IsRuntime()) {
    return false;
  }
  if (alloc_id == ctp::ipc::AllocatorId::GetNull()) {
    return false;
  }
  HLOG(kWarning,
       "IpcManager::TryLazyRegisterClientSegment: resolving ({}.{}) before "
       "its RegisterMemory round-trip landed — attaching on demand (#807)",
       alloc_id.major_, alloc_id.minor_);
  // Idempotent: returns true if the segment is already registered.
  return RegisterMemory(alloc_id);
#else
  (void)alloc_id;
  return false;
#endif
}

ClientShmInfo IpcManager::GetClientShmInfo(u32 index) const {
  std::lock_guard<std::mutex> lock(shm_mutex_);

  if (index >= alloc_vector_.size()) {
    return ClientShmInfo();  // Return empty info
  }

  int pid = ctp::SystemInfo::GetPid();
  std::string shm_name =
      "clio_" + std::to_string(pid) + "_" + std::to_string(index);

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
    std::string shm_name = "clio_" + std::to_string(alloc_id.major_) + "_" +
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
    std::string shm_name = "clio_" + std::to_string(alloc_id.major_) + "_" +
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
  int current_pid = ctp::SystemInfo::GetPid();

  for (const auto &name : ctp::SystemInfo::ListDirectory(memfd_dir)) {
    std::string full_path = memfd_dir + "/" + name;

    // The per-user memfd dir is shared by every runtime on this node (the
    // fallback topology runs several). Only reap leftovers from DEAD processes
    // — never clobber a segment a still-running runtime owns, or we break its
    // clients (and its fallbacks). memfd entries are symlinks to
    // /proc/<pid>/fd/N; if that pid is alive and isn't us, keep the entry.
    std::error_code ec;
    auto target = std::filesystem::read_symlink(full_path, ec);
    if (!ec) {
      const std::string t = target.string();
      constexpr const char *kProc = "/proc/";
      if (t.rfind(kProc, 0) == 0) {
        int owner_pid = std::atoi(t.c_str() + std::strlen(kProc));
        if (owner_pid > 0 && owner_pid != current_pid &&
            ctp::SystemInfo::IsProcessAlive(owner_pid)) {
          HLOG(kDebug,
               "ClearUserIpcs: keeping {} (owned by live pid {})", name,
               owner_pid);
          continue;
        }
      }
    }

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
    // issue #783: the metadata mapping points into the OLD server's segment;
    // drop it so ClientInitShm re-attaches the restarted server's one instead
    // of leaving a dangling pointer into an unmapped region.
    metadata_allocator_ = nullptr;
    metadata_backend_ = ctp::ipc::PosixShmMmap();

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
          clio::run::CreateTaskId(), clio::run::kAdminPoolId, clio::run::PoolQuery::Local(),
          alloc_id);
      IpcCpu2CpuZmq::SendIn(this,reg_task, IpcMode::kTcp).Wait();
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
      if (UseLocalZmqIpc()) {
        // macOS (issue #482): the original control transport was the unix
        // SocketTransport (ClientInit / WaitForLocalServer do the same). A ZMQ
        // DEALER here cannot receive the restarted server's ROUTER reply on
        // macOS 14+, so the reconnect would time out forever. Recreate the
        // SocketTransport against the restarted daemon's IPC endpoint.
        ctp::SystemInfo::EnsureMemfdDir();
        std::string ipc_path = ctp::SystemInfo::GetMemfdPath(
            "clio_" + std::to_string(port) + ".ipc");
        zmq_transport_ = ctp::lbm::TransportFactory::Get(
            ipc_path, ctp::lbm::TransportType::kSocket,
            ctp::lbm::TransportMode::kClient, "ipc", 0);
      } else {
        zmq_transport_ = ctp::lbm::TransportFactory::Get(
            config->GetServerAddr(), ctp::lbm::TransportType::kZeroMq,
            ctp::lbm::TransportMode::kClient, "tcp", port + 3);
      }
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
  // issue #783: remote host -> no shared memory at all, so the metadata cache
  // is unavailable and every read must take the RPC path.
  metadata_allocator_ = nullptr;
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
             "after {}s", elapsed);
        break;
      }
      CTP_THREAD_MODEL->SleepForUs(1000000);
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

// Milliseconds a blocking zmq_poll waits before returning to re-check the
// running flag. zmq_poll returns immediately when a response is ready, so this
// only bounds idle re-poll + shutdown-join latency; while idle it also caps how
// long a concurrent Send can be held off (PollRecv holds the socket mutex).
// Replaces the EventManager/WSAEventSelect wait, which can't watch ZMQ_FD on
// Windows and there falls back to a tick-floored ::Sleep (~15.6ms → the 40ms
// TCP round-trip).
static constexpr int kZmqPollTimeoutMs = 1;

void IpcManager::RecvZmqClientThread() {
  // Client-side thread: blocks for completed task responses. In TCP mode these
  // arrive on the dedicated response listener (an ephemeral ROUTER bound in
  // ClientInit) that the runtime dials back into — fully decoupled from the
  // request DEALER, so there is no shared sock_mtx_ between send and recv. In
  // IPC mode there is no listener; responses come back over the unix-socket
  // transport (zmq_transport_) as before. Uses ZMQ's native zmq_poll (via
  // Transport::PollRecv) rather than the EventManager, because ZMQ_FD cannot be
  // registered with WSAEventSelect on Windows.
  ctp::lbm::Transport *recv_transport = client_response_listener_
                                            ? client_response_listener_.get()
                                            : zmq_transport_.get();
  if (!recv_transport) {
    HLOG(kError, "RecvZmqClientThread: No response transport");
    return;
  }

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
      auto info = recv_transport->Recv(*archive);
      int rc = info.rc;
      if (rc == EAGAIN) break;
      if (rc != 0) {
        recv_transport->ClearRecvHandles(*archive);
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
        recv_transport->ClearRecvHandles(*archive);
        continue;
      }

      Task *task = it->second.task;

      // Store the archive for Recv() to pick up
      pending_response_archives_[net_key] = std::move(archive);

      // Memory fence before setting complete
      std::atomic_thread_fence(std::memory_order_release);

      // Signal completion on the client's task (per-process; polled by the
      // client thread's WaitCpu2Cpu loop).
      task->SetNewData();
      task->SetComplete();
      // Wake the client thread blocked in ClientRecv — it sleeps on its
      // EventManager rather than busy-polling. The waiter (pid,tid) lives on the
      // task's FutureInfo (recorded in ClientSend).
      if (task->WaiterPid() != 0) {
        ctp::lbm::EventManager::Signal(static_cast<int>(task->WaiterPid()),
                                       static_cast<int>(task->WaiterTid()));
      }

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

    // Only block when the drain loop found nothing; if we just processed
    // messages, loop back immediately to drain more. zmq_poll wakes the instant
    // a response arrives, so this adds no latency to message delivery.
    if (!drained_any) {
      recv_transport->PollRecv(kZmqPollTimeoutMs);
    }
  }
}

namespace {
/**
 * Lost-wakeup safety net for the parked SHM drain loops.
 *
 * The park/signal handshake (ShmMpscTransport::SignalConsumerIfParked) is
 * designed so a wake can never be missed, so in principle these threads could
 * block forever. They don't: a bounded wait means a bug in that handshake — or
 * a producer killed between reserving its slot and signalling — degrades to a
 * few milliseconds of extra latency instead of a permanently wedged runtime.
 * This codebase has shipped that exact failure mode more than once (#768,
 * #774), and 20 idle wakeups/sec/thread is not a measurable cost.
 *
 * This is NOT the polling interval: under load the drainer is woken by the
 * producer's signal and never reaches the timeout.
 */
constexpr int kShmParkTimeoutUs = 50'000;  // 50ms

/**
 * How long a drainer stays hot before parking, in failed drain attempts.
 *
 * Parking is not free: the wake costs the producer a tgkill and the consumer a
 * signalfd read plus an epoll wakeup. Measured on the task round-trip, parking
 * on EVERY message is worse than either alternative — the drain thread ends up
 * asleep whenever a request arrives, so the signal round-trip lands on the
 * critical path:
 *
 *   bdev_allocation, 4 threads    park always   spin always   spin-then-park
 *   SHM throughput                  5,303/s      14,227/s        (below)
 *
 * So spin while there is any reason to believe more work is imminent, and park
 * only once the ring has been quiet for a while. Under load the drainer never
 * reaches the budget, stays unparked, and producers skip the signal entirely
 * (SignalConsumerIfParked checks parked_ first) — the fast path costs nothing.
 * When traffic genuinely stops, the drainer parks once and the machine goes
 * fully idle: no polling, no periodic wakeups beyond the safety-net timeout.
 */
constexpr size_t kShmSpinBudget = 4096;

/**
 * Block until this ring has work, using the park/signal handshake.
 *
 * Publishing `parked` BEFORE the final IsEmpty() re-check is what makes this
 * race-free: a producer that reserves its slot after our re-check must observe
 * parked_ and signal us, and one that reserved before it is seen by the
 * re-check so we never block on a non-empty ring.
 */
void ShmParkUntilWork(ctp::lbm::ShmMpscTransport &ring,
                      ctp::lbm::EventManager *em) {
  ring.SetConsumerParked(true);
  if (ring.IsEmpty()) {
    em->Wait(kShmParkTimeoutUs);
  }
  ring.SetConsumerParked(false);
}

/**
 * One idle iteration of a drain loop: spin while warm, then park.
 * `spins` is the caller's counter; it is reset by the caller on any work, and
 * here after a park so a woken drainer gets a fresh hot window.
 */
void ShmDrainIdle(ctp::lbm::ShmMpscTransport &ring, ctp::lbm::EventManager *em,
                  size_t &spins) {
  if (++spins < kShmSpinBudget) {
    CTP_THREAD_MODEL->Yield();
    return;
  }
  ShmParkUntilWork(ring, em);
  spins = 0;
}
}  // namespace

void IpcManager::RecvShmClientThread() {
#if CTP_IS_HOST
  // Client-side SHM analogue of RecvZmqClientThread. Drains the single
  // per-process response ring (shm_out_server_). For each response: match it to
  // the waiting Future by net_key (stamped in SendIn, echoed by SendOut), park
  // the archive for RecvOut to deserialize (RecvOut owns the concrete task
  // type), mark the client task complete, and wake its waiter via
  // EventManager::Signal. This replaces the old model where every client thread
  // polled its own per-thread ring in RecvOut.
  size_t recv_count = 0;
  size_t miss_count = 0;
  // GetTls() registers this thread's signalfd-backed EventManager (its ctor
  // calls AddSignalEvent, which also BLOCKS SIGUSR1 here — mandatory before
  // publishing ourselves as the drainer, or a producer's wake would kill the
  // process via SIGUSR1's default disposition).
  ctp::lbm::EventManager *em = &GetTls()->event_manager_;
  shm_out_server_.RegisterConsumer();
  size_t idle_spins = 0;
  while (shm_recv_running_.load()) {
    // issue #807: the shared drainer. When app threads are actively waiting they
    // drain inline (RecvOut) and win the try-lock, so this thread mostly falls
    // through to park; when they are parked, it wins the lock and drains for
    // them. Either way exactly one consumer runs at a time.
    bool drained_any = DrainShmResponses();
    if (drained_any) {
      idle_spins = 0;
    } else {
      ShmDrainIdle(shm_out_server_, em, idle_spins);
    }
  }
  shm_out_server_.UnregisterConsumer();
  (void)recv_count;
  (void)miss_count;
#endif
}

bool IpcManager::DrainShmResponses() {
#if CTP_IS_HOST
  if (!shm_out_server_ok_) {
    return false;
  }
  // Sole-consumer gate: if another thread (a spinning waiter or this fallback
  // thread) is already draining, back off — the caller polls its own IsComplete.
  std::unique_lock<std::mutex> drain_lk(shm_out_drain_mutex_, std::try_to_lock);
  if (!drain_lk.owns_lock()) {
    return false;
  }
  bool any = false;
  while (true) {
    // Cheap check before allocating: a spinning waiter calls this every
    // iteration, so allocating a LoadTaskArchive only to hit EAGAIN on an empty
    // ring dominated the spin (measured +7us). IsEmpty is two atomic loads and
    // is safe here — we hold the sole-consumer lock.
    if (shm_out_server_.IsEmpty()) {
      break;
    }
    auto archive = std::make_unique<LoadTaskArchive>();
    ctp::lbm::ClientInfo info =
        shm_out_server_.Recv(*archive, ctp::lbm::SHM_MPSC_DONTWAIT);
    if (info.rc != 0) {
      break;  // EAGAIN: nothing more in flight right now
    }
    any = true;
    if (archive->GetTaskInfos().empty()) {
      HLOG(kError, "DrainShmResponses: response with no task_infos");
      continue;
    }
    size_t net_key = archive->GetTaskInfos()[0].task_id_.net_key_;

    std::lock_guard<std::mutex> lock(pending_futures_mutex_);
    auto it = pending_zmq_futures_.find(net_key);
    if (it == pending_zmq_futures_.end()) {
      HLOG(kError, "DrainShmResponses: no pending future for net_key {}",
           net_key);
      continue;
    }
    Task *task = it->second.task;
    // Park the archive; RecvOut moves it out and deserializes into the task.
    // Demuxing here (not just for our own net_key) is what lets one draining
    // waiter serve every waiter's response.
    pending_response_archives_[net_key] = std::move(archive);
    // Publish the parked archive before the waiter observes completion.
    std::atomic_thread_fence(std::memory_order_release);
    task->SetNewData();
    task->SetComplete();
    if (task->WaiterPid() != 0) {
      ctp::lbm::EventManager::Signal(static_cast<int>(task->WaiterPid()),
                                     static_cast<int>(task->WaiterTid()));
    }
    pending_zmq_futures_.erase(it);
  }
  return any;
#else
  return false;
#endif
}

// issue #807: RecvShmServerThread removed — workers drain shards inline
// (Worker::DrainMyShard). No dedicated inbound drain thread exists.


void IpcManager::StartShmServerRecvThread() {
#if CTP_IS_HOST
  // issue #807: NO dedicated inbound drain threads. The workers drain the shard
  // rings themselves (RegisterShardConsumer / DrainShard from Worker::Run), so
  // parallel ingress comes from oversubscribing the existing pool rather than
  // from spawning threads that contend for cores — which is what made the first
  // cut regress. Intentionally a no-op; rings are created in ServerInitShm and
  // consumed by workers.
  shm_in_recv_running_.store(true, std::memory_order_release);
#endif
}

void IpcManager::StopShmServerRecvThread() {
#if CTP_IS_HOST
  shm_in_recv_running_.store(false, std::memory_order_release);
#endif
}

// --- issue #807: worker-driven shard draining --------------------------------

void IpcManager::RegisterShardConsumer(u32 worker_id) {
#if CTP_IS_HOST
  if (!shm_in_server_ok_ || worker_id >= shm_in_servers_.size() ||
      shm_in_servers_[worker_id] == nullptr) {
    return;
  }
  // Publishes THIS worker thread's tid as the ring's consumer, so a producing
  // client's SignalConsumerIfParked SIGUSR1s this worker — the same signalfd
  // path Worker::SuspendMe already parks on.
  shm_in_servers_[worker_id]->RegisterConsumer();
#else
  (void)worker_id;
#endif
}

void IpcManager::UnregisterShardConsumer(u32 worker_id) {
#if CTP_IS_HOST
  if (!shm_in_server_ok_ || worker_id >= shm_in_servers_.size() ||
      shm_in_servers_[worker_id] == nullptr) {
    return;
  }
  shm_in_servers_[worker_id]->UnregisterConsumer();
#else
  (void)worker_id;
#endif
}

bool IpcManager::ShardEmpty(u32 worker_id) const {
#if CTP_IS_HOST
  if (!shm_in_server_ok_ || worker_id >= shm_in_servers_.size() ||
      shm_in_servers_[worker_id] == nullptr) {
    return true;
  }
  return shm_in_servers_[worker_id]->IsEmpty();
#else
  (void)worker_id;
  return true;
#endif
}

void IpcManager::SetShardParked(u32 worker_id, bool parked) {
#if CTP_IS_HOST
  if (!shm_in_server_ok_ || worker_id >= shm_in_servers_.size() ||
      shm_in_servers_[worker_id] == nullptr) {
    return;
  }
  shm_in_servers_[worker_id]->SetConsumerParked(parked);
#else
  (void)worker_id;
  (void)parked;
#endif
}

// ===========================================================================
// issue #807 D2: nonblocking, per-destination, background response sender.
// ===========================================================================

void IpcManager::EnqueueShmSend(const std::string &dest,
                                clio::run::Future<Task> &&future) {
#if CTP_IS_HOST
  ShmSendQueue *q = nullptr;
  {
    std::lock_guard<std::mutex> lk(shm_send_queues_mutex_);
    auto it = shm_send_queues_.find(dest);
    if (it == shm_send_queues_.end()) {
      it = shm_send_queues_
               .emplace(dest, std::make_unique<ShmSendQueue>())
               .first;
    }
    q = it->second.get();
  }
  {
    std::lock_guard<std::mutex> lk(q->mtx);
    q->pending.push_back(std::move(future));
  }
  // No wake needed: the workers drain these queues in their poll loop
  // (DrainShmSends), and the enqueuing worker itself will drain on its next
  // iteration; the 50ms max_sleep cap bounds the worst case if all workers park.
#else
  (void)dest;
  (void)future;
#endif
}

u32 IpcManager::DrainShmSends(ctp::lbm::Transport *transport, u32 budget) {
#if CTP_IS_HOST
  // issue #807: called from the WORKERS' poll loop (and the shutdown flush), so
  // deferred response send is oversubscribed onto the existing pool rather than
  // owned by a dedicated thread. Bounded per call so one worker cannot spend a
  // whole iteration here and starve its own lane. A no-op when the queues are
  // empty, which is always the case on the inline-default path.
  //
  // Multiple workers may drain concurrently: pops are atomic under each queue's
  // mutex, and conn->Send serialises per client ring via its own send_mu_, so
  // two workers transferring to the same client are safe. The client demuxes
  // responses by net_key, so cross-worker delivery order does not matter.
  u32 sent = 0;
  std::vector<ShmSendQueue *> queues;
  {
    std::lock_guard<std::mutex> lk(shm_send_queues_mutex_);
    queues.reserve(shm_send_queues_.size());
    for (auto &kv : shm_send_queues_) queues.push_back(kv.second.get());
  }
  for (ShmSendQueue *q : queues) {
    while (sent < budget) {
      clio::run::Future<Task> fut;
      {
        std::lock_guard<std::mutex> lk(q->mtx);
        if (q->pending.empty()) break;
        fut = std::move(q->pending.front());
        q->pending.pop_front();
      }
      clio::run::shared_ptr<Task> task = fut.GetTaskPtr();
      if (!task.IsNull()) {
        IpcCpu2Cpu::SendOutTransfer(this, task, transport);
      }
      ++sent;
      // `fut` drops here, freeing the task by RAII on this worker thread.
    }
    if (sent >= budget) break;
  }
  return sent;
#else
  (void)transport;
  (void)budget;
  return 0;
#endif
}

void IpcManager::StartShmServerSendThread() {
#if CTP_IS_HOST
  // issue #807: no dedicated sender thread. The default (inline) send transfers
  // on the executing worker; the opt-in async send is drained by the workers via
  // DrainShmSends in their poll loop. Kept as a no-op so the admin bring-up call
  // site is unchanged.
  shm_send_running_.store(true, std::memory_order_release);
#endif
}

void IpcManager::StopShmServerSendThread() {
#if CTP_IS_HOST
  shm_send_running_.store(false, std::memory_order_release);
  // Shutdown flush: workers have stopped draining, so transfer whatever
  // responses are still queued instead of dropping them (which would hang the
  // waiting clients). A temp transport on this thread; bounded so a pathological
  // producer cannot hang shutdown.
  ctp::lbm::TransportPtr flush_transport = ctp::lbm::TransportFactory::Get(
      "", ctp::lbm::TransportType::kShm, ctp::lbm::TransportMode::kClient);
  int empty_passes = 0;
  int total_passes = 0;
  while (empty_passes < 2 && total_passes < 10000) {
    empty_passes = (DrainShmSends(flush_transport.get(), 4096) > 0)
                       ? 0
                       : empty_passes + 1;
    ++total_passes;
  }
#endif
}

void IpcManager::HeartbeatThread() {
  while (heartbeat_running_.load()) {
    bool alive = IsServerAlive();
    server_alive_.store(alive, std::memory_order_release);
    CTP_THREAD_MODEL->SleepForUs(1000000);
  }
}

void IpcManager::CleanupResponseArchive(size_t net_key) {
  std::lock_guard<std::mutex> lock(pending_futures_mutex_);
  auto it = pending_response_archives_.find(net_key);
  if (it != pending_response_archives_.end()) {
    // Frees ZMQ zero-copy recv handles (bulk.desc); a no-op for a SHM archive
    // (its recv bulks carry no desc). Reaching here means the future was dropped
    // WITHOUT a Recv — the consumed path erases the entry in RecvOut. NOTE: a
    // SHM archive's malloc'd recv payloads (bulk.data.ptr_) are intentionally
    // NOT freed here: bulk.data.ptr_ on the TCP/IPC path can be a CHI-allocated
    // (non-malloc) buffer, so a blanket free corrupts the heap. The consumed
    // path adopts SHM payloads into the task (TASK_DATA_OWNER); only a truly
    // dropped-before-Recv SHM future leaks its payload, which is rare and bounded.
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
        clio::run::CreateTaskId(), clio::run::kAdminPoolId, clio::run::PoolQuery::Local(),
        backend_id, admin_kind, gpu_id, static_cast<u64>(bytes),
        ipc_handle_bytes);
    IpcCpu2CpuZmq::SendIn(this, reg_task, IpcMode::kTcp).Wait();
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

// Allocate this task's owned RunContext and resolve its execution container.
//
// Called at the ipc_*.cc receive/send sites (NOT necessarily on a worker
// thread — e.g. the net recv worker, or the main thread during ServerInit's
// synchronous admin pool creation). It therefore does ONLY thread-independent
// setup: allocate the RunContext and resolve the container. The worker-specific
// per-execution parameters (worker id, lane, event queue, future, polling,
// predicted stats) are set later, on the worker, in Worker::ProcessNewTask and
// Task::StartCoroutine. Giving the task an active RunContext here is what lets
// RouteTask run before the worker picks the task up.
void Task::EnsureRunCtx() {
  // Allocate the RunContext storage if the task does not already have one. This
  // is the lightweight half of BeginRunContext (no container resolution): it is
  // what gives the task its embedded routing state (run_ctx_->route_) so the
  // client SendIn / server RecvIn can stamp it via GetFutureShm() before the
  // worker formally begins the task.
  if (!run_ctx_) {
    run_ctx_ = ctp::make_unique<RunContext>(CTP_MALLOC);
  }
}

void Task::BeginRunContext() {
  // Reuse an existing RunContext (e.g. one created at RecvIn to hold the
  // response-routing state) rather than clobbering it — re-allocating here
  // would wipe run_ctx_->route_ that the transport set before dispatch.
  EnsureRunCtx();
  // Fresh execution starts not-complete (the waiter must not observe a stale
  // completion from a prior life of a recycled task).
  UnsetComplete();
  UnsetNewData();
#if CTP_IS_HOST
  // Resolve the execution container: the real (local) container if one exists
  // (the common case), otherwise the static container.
  ExecContainer() = CLIO_POOL_MANAGER->GetRealOrStaticContainer(pool_id_);
#endif
}

RouteResult IpcManager::RouteTask(Future<Task> &future, bool force_enqueue) {
  // Get task pointer from future
  clio::run::shared_ptr<Task> task_ptr = future.GetTaskPtr();

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

  // Collective (ManyToOne / AllToOne) routing is handled before the normal
  // resolve path: forward to the neighborhood leader, or park into the batch
  // manager if we are the leader. Both modes share this routing; they differ
  // only in the BatchManager flush condition (time window vs. all-containers
  // barrier). (The aggregate task the leader later runs is a plain Local task
  // and does not re-enter this branch.)
  if (task_ptr->pool_query_.IsCollectiveMode()) {
    return RouteManyToOne(future);
  }

  // Only call ScheduleTask for Dynamic pool queries.
  // ScheduleTask resolves Dynamic routing into concrete modes (e.g.,
  // Broadcast, DirectHash, Local). Concrete routing modes (Range, Physical,
  // Local, Broadcast, etc.) were set by a previous routing step and must
  // not be overridden — doing so would cause infinite re-broadcast loops
  // when tasks arrive at remote nodes (e.g., GetOrCreatePool returns
  // Broadcast on every node since the pool doesn't exist yet).
  auto *pool_manager = CLIO_POOL_MANAGER;
  auto static_container =
      pool_manager->GetStaticContainer(task_ptr->pool_id_).get();
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
      if (worker) {
        worker->AddToRetryQueue(task_ptr);
      }
    }
    return result;
  } else {
    return RouteGlobal(future, pool_queries);
  }
}

RouteResult IpcManager::RouteManyToOne(Future<Task> &future) {
  clio::run::shared_ptr<Task> task_ptr = future.GetTaskPtr();
  u64 leader = GetNeighborhoodLeaderNodeId();
  if (leader == GetNodeId()) {
    // We are the neighborhood leader: park the task into its batch group. The
    // batch flusher aggregates, runs once, and completes each member later.
    // RouteResult::Local tells the worker the task is owned elsewhere now and
    // must not be executed here.
    //
    // Anchor the FutureShm to the task's RunContext: the batch holds the member
    // task by pointer, and the queued Future that owns the FutureShm is dropped
    // once routing returns. Without this bind the FutureShm is freed before the
    // batch flusher runs, so OnAggregateComplete can't signal the member's
    // waiter (future_shm is null) and the client's Wait() hangs. Mirrors the
    // bind in RouteGlobal; the batch path skips ProcessNewTask, which is the
    // only other place RunFuture is bound.
    task_ptr->RunFuture() = future;
    // Break the strong back-reference cycle (task -> RunContext -> future_ ->
    // task_ptr_ -> task) that would otherwise leak the member after the batch
    // completes, but keep the Future's task pointer pointing AT the member as a
    // NON-OWNING handle: GetFutureShm() now resolves the routing state through
    // the task (TaskRaw()->RunCtxPtr()), so a plain reset() would make
    // OnAggregateComplete -> EndTask see a null route and never signal the
    // member's waiter (the client's Wait() would hang).
    task_ptr->RunFuture().GetTaskPtr() =
        clio::run::shared_ptr<Task>::WrapNonOwning(task_ptr.get());
    task_ptr->SetRouted();
    batch_manager_->Add(task_ptr);
    return RouteResult::Local;
  }
  // Forward to the leader. The forwarded task keeps its collective pool_query_
  // (ManyToOne or AllToOne) so the leader re-enters this path and batches; only
  // the network address is the leader node.
  std::vector<PoolQuery> pool_queries = {PoolQuery::Physical((u32)leader)};
  pool_queries[0].SetReturnNode(GetNodeId());
  return RouteGlobal(future, pool_queries);
}

bool IpcManager::IsTaskLocal(const clio::run::shared_ptr<Task> & /*task_ptr*/,
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
    case RoutingMode::ManyToOne:
    case RoutingMode::AllToOne:
      // These modes should have been resolved (ManyToOne/AllToOne → Local on
      // the neighborhood leader, else Physical) by now. If we still see them
      // here, they are not local.
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
  clio::run::shared_ptr<Task> task_ptr = future.GetTaskPtr();

  // Mark as routed so the task is not re-routed on subsequent passes.
  task_ptr->SetRouted();

  // Resolve the actual execution container (resolve-once: this handle is cached
  // in the RunContext below and reused for the task's whole lifetime).
  auto *pool_manager = CLIO_POOL_MANAGER;
  ContainerId container_id = task_ptr->pool_query_.GetContainerId();
  DynamicContainer exec_dc =
      pool_manager->GetContainer(task_ptr->pool_id_, container_id);
  ContainerHold exec_container = exec_dc.get();

  if (!exec_container) {
    HLOG(kError, "RouteLocal: Container not found for pool={} container_id={} method={}",
         task_ptr->pool_id_, container_id, task_ptr->method_);
    return RouteResult::Dne;
  }
  if (exec_dc.IsPlugged()) {
    HLOG(kWarning, "RouteLocal: Container plugged for pool={}", task_ptr->pool_id_);
    return RouteResult::Retry;
  }

  // RouteToGpu was the cpu→gpu dispatch for the now-removed GPU
  // runtime. ToLocalGpu / LocalGpuBcast routing modes are no longer
  // honored — kernels are pure task producers, not consumers.

  // Set the completer_ field to track which container will execute this task
  task_ptr->SetCompleter(exec_container->container_id_);

  // Update RunContext to use the resolved execution container
  task_ptr->ExecContainer() = exec_dc;

  // Use scheduler to pick the destination worker
  Worker *worker = CLIO_CUR_WORKER;
  u32 dest_worker_id =
      scheduler_->RuntimeMapTask(worker, future, exec_container);

  // If destination matches this worker and not forced to enqueue, execute directly
  if (!force_enqueue && worker && dest_worker_id == worker->GetId()) {
    return RouteResult::ExecHere;
  }

  // Self-send deadlock avoidance: a worker force-enqueuing a subtask onto its
  // OWN lane busy-spins in the WAIT_FOR_SPACE ring Push when the lane is full,
  // and can never drain it — it IS the consumer, blocked here in Push rather
  // than in its Run loop. Redirect to an alternate worker whose own thread
  // drains it, converting the deadlock into transient backpressure. An earlier
  // "only redirect to a non-full sibling" guard failed under the mmap-writeback
  // storm (all lanes saturate → no non-full sibling → fell back to self-spin);
  // redirect UNCONDITIONALLY — briefly spinning on a *sibling's* full lane is
  // safe because that sibling's thread drains it. (generic/438: mmap read fault
  // -> GetBlob -> bdev::AsyncRead -> SendIn all on the scheduler worker; the
  // bdev subtask's predicted io_size is 0 so RuntimeMapTask routes it back to
  // the scheduler worker = self.)
  if (force_enqueue && worker && dest_worker_id == worker->GetId()) {
    Worker *alt = scheduler_->PickAltWorker(dest_worker_id);
    if (alt != nullptr) {
      dest_worker_id = alt->GetId();
    }
  }

  // Enqueue to the destination worker's lane, then ALWAYS signal. Gating the
  // wakeup on was_empty is the exact lost-wakeup race AwakenWorker's own
  // comment warns against: the consumer can drain the lane and park in
  // epoll_pwait2 in the window between our Empty() check and Push, so a
  // "non-empty" observation skips a wakeup the worker actually needed. The
  // extra SIGUSR1 is absorbed harmlessly by signalfd if the worker is already
  // awake. (Surfaced as a permanent hang in generic/208 aio-dio once self-sent
  // subtasks began routing to otherwise-idle I/O workers.)
  auto &dest_lane = worker_queues_->GetLane(dest_worker_id, 0);
  dest_lane.Push(future);
  AwakenWorker(&dest_lane);
  return RouteResult::Local;
}

RouteResult IpcManager::RouteGlobal(Future<Task> &future,
                             const std::vector<PoolQuery> &pool_queries) {
  // Get task pointer from future
  clio::run::shared_ptr<Task> task_ptr = future.GetTaskPtr();

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
  task_ptr->PoolQueries() = pool_queries;

  // Anchor the FutureShm to the task's RunContext. run2run's send_map_ holds a
  // net-forwarded task by Task pointer only; the queued Future that owns the
  // FutureShm is dropped as soon as the net-send worker pops and forwards it.
  // Without this bind the FutureShm is freed before the peer response returns,
  // so RecvOut -> EndTask sees a null future_shm and never sends the response,
  // hanging the waiting client. ProcessNewTask binds RunFuture for locally
  // executed tasks; net-routed tasks skip ProcessNewTask, so bind it here.
  task_ptr->RunFuture() = future;
  // Break the strong self-reference cycle (task -> RunContext -> future_ ->
  // task_ptr_ -> task) that would leak the origin task after send_map_ erases
  // it, but keep the Future's task pointer pointing AT the task as a NON-OWNING
  // handle: GetFutureShm() resolves the routing state through the task
  // (TaskRaw()->RunCtxPtr()), so a plain reset() would make RecvIn/RecvOut ->
  // EndTask see a null route and never send the response (hanging the client).
  task_ptr->RunFuture().GetTaskPtr() =
      clio::run::shared_ptr<Task>::WrapNonOwning(task_ptr.get());

  // Pick the latency vs I/O SendIn lane based on the task's actual
  // payload size — small probes / metadata sit on kSendInLatency so
  // they're not buried behind 1 MiB PutBlob bulks on the wire. The
  // scheduler (BeginTask / pre-routing) populates RunContext::
  // predicted_stat_ from container->GetTaskStats(task), so we just
  // read it here instead of recomputing.
  size_t io_size = task_ptr->PredictedStat().io_size_;
  NetQueuePriority sendin_prio = (io_size >= kNetQueueIoThreshold)
                                     ? NetQueuePriority::kSendInIO
                                     : NetQueuePriority::kSendInLatency;
  EnqueueNetTask(future, sendin_prio);

  // Set TASK_ROUTED flag on original task
  task_ptr->SetRouted();

  Worker *worker = CLIO_CUR_WORKER;
  HLOG(kDebug, "Worker {}: RouteGlobal - task enqueued to net_queue",
       worker ? worker->GetId() : 0);

  return RouteResult::Network;
}


std::vector<PoolQuery> IpcManager::ResolvePoolQuery(
    const PoolQuery &query, PoolId pool_id, const clio::run::shared_ptr<Task> &task_ptr) {
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
    case RoutingMode::ManyToOne:
    case RoutingMode::AllToOne:
      // Collective modes are intercepted in RouteTask (RouteManyToOne) before
      // reaching here. If one ever falls through, resolve to the neighborhood
      // leader.
      result = {PoolQuery::Physical(
          (u32)GetNeighborhoodLeaderNodeId())};
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
    const PoolQuery &query, const clio::run::shared_ptr<Task> &task_ptr) {
  // Local routing - process on current node
  return {query};
}

std::vector<PoolQuery> IpcManager::ResolveDirectIdQuery(
    const PoolQuery &query, PoolId pool_id, const clio::run::shared_ptr<Task> &task_ptr) {
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
    const PoolQuery &query, PoolId pool_id, const clio::run::shared_ptr<Task> &task_ptr) {
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
    const PoolQuery &query, PoolId pool_id, const clio::run::shared_ptr<Task> &task_ptr) {
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
    const PoolQuery &query, PoolId pool_id, const clio::run::shared_ptr<Task> &task_ptr) {
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
    const PoolQuery &query, PoolId pool_id, const clio::run::shared_ptr<Task> &task_ptr) {
  // Physical routing - query is already resolved to a specific node
  return {query};
}


void IpcManager::SendRuntime(
    const clio::run::shared_ptr<Task> &task_ptr,
    ctp::lbm::Transport *send_transport) {
  auto future_shm = task_ptr->RunFuture().GetFutureShm();
  ClientOrigin origin = future_shm->origin_;

  switch (origin) {
    case ClientOrigin::kClientShm:
    default:
      IpcCpu2Cpu::SendOut(this, task_ptr, send_transport);
      break;
    case ClientOrigin::kClientTcp:
    case ClientOrigin::kClientIpc:
      IpcCpu2CpuZmq::EnqueueSendOut(this, task_ptr, origin);
      break;
#if CTP_ENABLE_CUDA || CTP_ENABLE_ROCM || CTP_ENABLE_SYCL
    case ClientOrigin::kClientGpu2Cpu:
      IpcGpu2Cpu::SendOut(this, task_ptr);
      break;
#endif
    // FUTURE_CLIENT_CPU2GPU dispatch was removed with the GPU runtime.
  }
}

}  // namespace clio::run
