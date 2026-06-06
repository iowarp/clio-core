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
 * Runtime implementation for Admin ChiMod
 *
 * Critical ChiMod for managing ChiPools and runtime lifecycle.
 * Contains the server-side task processing logic with PoolManager integration.
 */

#include "clio_runtime/admin/admin_runtime.h"

#include <clio_runtime/manager.h>
#include <clio_runtime/module_manager.h>
#include <clio_runtime/pool_manager.h>
#include <clio_runtime/task_archives.h>
#include <clio_runtime/worker.h>
#include <clio_ctp/lightbeam/transport_factory_impl.h>
#include <clio_ctp/serialize/msgpack_wrapper.h>

#include "clio_ctp/data_structures/serialization/global_serialize.h"
#include <cerrno>
#include <chrono>
#include <climits>
#include <filesystem>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace clio::run::admin {

// ============================================================================
// Network-path instrumentation
// ============================================================================
//
// Counters and timers for diagnosing where cross-node throughput is being
// lost. Each direction is updated by exactly one worker thread (net_send
// or net_recv per DefaultScheduler::DivideWorkers), so the per-peer
// arrays don't need atomics — only the periodic-dump snapshot crosses
// thread boundaries, and that uses relaxed atomics on the timestamp +
// "owner has-flushed" flag.
//
// Look for "[NetStats] ..." lines in the runtime log. Emitted every
// kNetStatsDumpIntervalSec wall-clock seconds from the Send periodic.
//
// Per-peer rows:
//   peer=N  sin=<count>/<MiB>  sout=<count>/<MiB>  rin=<count>/<MiB>  rout=<count>/<MiB>
//   send_lbm_ms=<sum>  send_ser_ms=<sum>  recv_des_ms=<sum>
//
// Aggregate row:
//   total: ... (sum across peers)
//
// To reset between runs just restart the runtime; counters are process-
// scoped.
struct NetPeerStats {
  // SendIn (outbound cross-node task forwards)
  uint64_t sin_count = 0;
  uint64_t sin_bytes = 0;
  uint64_t sin_lbm_ns = 0;    // time inside lbm_transport->Send for sin
  uint64_t sin_ser_ns = 0;    // time inside container->SaveTask for sin
  // SendOut (outbound cross-node task responses)
  uint64_t sout_count = 0;
  uint64_t sout_bytes = 0;
  uint64_t sout_lbm_ns = 0;
  uint64_t sout_ser_ns = 0;
  // RecvIn (inbound cross-node task forwards from peer)
  uint64_t rin_count = 0;
  uint64_t rin_bytes = 0;
  uint64_t rin_des_ns = 0;    // time inside deserialize/dispatch
  // RecvOut (inbound cross-node task responses from peer)
  uint64_t rout_count = 0;
  uint64_t rout_bytes = 0;
  uint64_t rout_des_ns = 0;
};

static constexpr size_t kMaxNetPeers = 64;  // soft cap; we just modulo
static NetPeerStats g_net_peer_stats[kMaxNetPeers];

static inline NetPeerStats &NetStats(chi::u64 peer) {
  return g_net_peer_stats[peer % kMaxNetPeers];
}

static constexpr double kNetStatsDumpIntervalSec = 1.0;
static std::chrono::steady_clock::time_point g_net_stats_last_dump =
    std::chrono::steady_clock::now();

static inline void DumpNetStatsIfDue(chi::u64 self_node_id, size_t num_peers) {
  auto now = std::chrono::steady_clock::now();
  double since =
      std::chrono::duration<double>(now - g_net_stats_last_dump).count();
  if (since < kNetStatsDumpIntervalSec) return;
  g_net_stats_last_dump = now;

  // Aggregates across peers
  uint64_t tot_sin_c = 0, tot_sin_b = 0, tot_sout_c = 0, tot_sout_b = 0;
  uint64_t tot_rin_c = 0, tot_rin_b = 0, tot_rout_c = 0, tot_rout_b = 0;
  uint64_t tot_sin_lbm = 0, tot_sout_lbm = 0;
  uint64_t tot_sin_ser = 0, tot_sout_ser = 0;
  uint64_t tot_rin_des = 0, tot_rout_des = 0;

  size_t cap = std::min<size_t>(num_peers, kMaxNetPeers);
  for (size_t p = 0; p < cap; ++p) {
    if (p == self_node_id) continue;
    const auto &s = g_net_peer_stats[p];
    if (s.sin_count == 0 && s.sout_count == 0 && s.rin_count == 0 &&
        s.rout_count == 0) {
      continue;
    }
    HLOG(kDebug,
         "[NetStats] self={} peer={} sin={}/{}MiB sout={}/{}MiB "
         "rin={}/{}MiB rout={}/{}MiB "
         "sin_lbm_ms={:.1f} sout_lbm_ms={:.1f} "
         "sin_ser_ms={:.1f} sout_ser_ms={:.1f} "
         "rin_des_ms={:.1f} rout_des_ms={:.1f}",
         self_node_id, p, s.sin_count, s.sin_bytes >> 20, s.sout_count,
         s.sout_bytes >> 20, s.rin_count, s.rin_bytes >> 20, s.rout_count,
         s.rout_bytes >> 20, s.sin_lbm_ns / 1e6, s.sout_lbm_ns / 1e6,
         s.sin_ser_ns / 1e6, s.sout_ser_ns / 1e6, s.rin_des_ns / 1e6,
         s.rout_des_ns / 1e6);
    tot_sin_c += s.sin_count;
    tot_sin_b += s.sin_bytes;
    tot_sout_c += s.sout_count;
    tot_sout_b += s.sout_bytes;
    tot_rin_c += s.rin_count;
    tot_rin_b += s.rin_bytes;
    tot_rout_c += s.rout_count;
    tot_rout_b += s.rout_bytes;
    tot_sin_lbm += s.sin_lbm_ns;
    tot_sout_lbm += s.sout_lbm_ns;
    tot_sin_ser += s.sin_ser_ns;
    tot_sout_ser += s.sout_ser_ns;
    tot_rin_des += s.rin_des_ns;
    tot_rout_des += s.rout_des_ns;
  }
  HLOG(kDebug,
       "[NetStats] self={} TOTAL sin={}/{}MiB sout={}/{}MiB "
       "rin={}/{}MiB rout={}/{}MiB "
       "sin_lbm_ms={:.1f} sout_lbm_ms={:.1f} "
       "sin_ser_ms={:.1f} sout_ser_ms={:.1f} "
       "rin_des_ms={:.1f} rout_des_ms={:.1f}",
       self_node_id, tot_sin_c, tot_sin_b >> 20, tot_sout_c,
       tot_sout_b >> 20, tot_rin_c, tot_rin_b >> 20, tot_rout_c,
       tot_rout_b >> 20, tot_sin_lbm / 1e6, tot_sout_lbm / 1e6,
       tot_sin_ser / 1e6, tot_sout_ser / 1e6, tot_rin_des / 1e6,
       tot_rout_des / 1e6);
}

static inline uint64_t HrtNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}


// Method implementations for Runtime class

// Virtual method implementations (Init, Run, Del, SaveTask, LoadTask, NewCopy,
// Aggregate) now in autogen/admin_lib_exec.cc

//===========================================================================
// Method implementations
//===========================================================================

void Runtime::StopRecvThreads() {
  // Signal dedicated recv threads to exit and join them. The threads cache a
  // raw pointer to the main transport, so they MUST be joined before
  // IpcManager::ClearTransports() frees it (see RegisterTransportShutdownHook).
  recv_shutdown_.store(true, std::memory_order_release);
  if (peer_recv_thread_.joinable()) peer_recv_thread_.join();
  if (client_recv_thread_.joinable()) client_recv_thread_.join();
}

Runtime::~Runtime() {
  // Backstop: normally the IpcManager transport-shutdown hook already joined
  // these (while the transport was still alive). Safe to call again.
  StopRecvThreads();
}

chi::TaskResume Runtime::Create(ctp::ipc::FullPtr<CreateTask> task,
                                chi::RunContext &rctx) {
  CLIO_TASK_BODY_BEGIN
  // Admin container creation logic (IS_ADMIN=true)
  HLOG(kDebug, "Admin: Initializing admin container");

  // Initialize the Admin container with pool information from the task
  // Note: Admin container is already initialized by the framework before Create
  // is called

  // send_map_/recv_map_ access is now guarded by send_map_mutex_ /
  // recv_map_mutex_ since DefaultScheduler splits the net worker into a
  // send worker (SendIn / SendOut / ProcessRetryQueues) and a recv worker
  // (RecvIn / RecvOut).

  create_count_++;

  // ===========================================================================
  // Cross-node inbound network path: dedicated std::thread instead of
  // worker-periodic Recv. The worker scheduler's yield/wake/epoll cycle
  // added too much latency between socket-readable and dispatch (see
  // [PutLat] / [NetStats] dumps showing ~97% of per-blob latency
  // unaccounted in the network timers). These threads call
  // lbm_transport->Recv() directly and dispatch via RecvIn / RecvOut.
  //
  // Inbound dispatch only pushes Futures to worker lanes:
  //   - RecvIn  -> IpcManager::Send -> IpcCpu2Self::ClientSend (non-worker
  //                branch: RouteTask(force_enqueue=true), which is safe
  //                from a non-worker thread)
  //   - RecvOut -> direct worker_queues_->GetLane(...).Push(future)
  //   - RuntimeRecv (client TCP/IPC) -> direct lane push
  //
  // None of these go through CLIO_CUR_WORKER (TLS) for anything that matters.
  // The outbound Send path is unchanged: still served by the net_send_worker
  // via the AsyncSendPoll / AsyncClientSend periodics below.
  // ===========================================================================

  // Spawn periodic Send task — outbound side still runs on the worker
  // because the per-task send path is bounded by transport capacity, and
  // EnqueueNetTask is invoked from many worker threads anyway.
  client_.AsyncSendPoll(chi::PoolQuery::Local(), 0, 500);

  // Spawn periodic ClientSend task for client response sending via lightbeam
  client_.AsyncClientSend(chi::PoolQuery::Local(), 100);

  // Dedicated single peer recv thread: polls the main p2p transport
  // (port 9413 by default) and dispatches inbound task forwards/responses.
  peer_recv_thread_ = std::thread([this]() {
    ctp::SystemInfo::SetCurrentThreadName("chi-peer-recv");
    auto *ipc_manager = CLIO_IPC;
    ctp::lbm::Transport *lbm_transport = nullptr;
    for (int spin = 0; spin < 1000 && !recv_shutdown_.load(); ++spin) {
      lbm_transport = ipc_manager->GetMainTransport();
      if (lbm_transport) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (!lbm_transport) {
      HLOG(kError, "[PeerRecvThread] main transport never appeared");
      return;
    }
    HLOG(kInfo, "[PeerRecvThread] started");
    while (!recv_shutdown_.load(std::memory_order_acquire)) {
      chi::LoadTaskArchive archive;
      auto info = lbm_transport->Recv(archive);
      int rc = info.rc;
      if (rc == EAGAIN) {
        std::this_thread::sleep_for(std::chrono::microseconds(1));
        continue;
      }
      if (rc != 0) {
        if (rc != -1) {
          HLOG(kError, "[PeerRecvThread] Recv failed rc={}", rc);
        }
        continue;
      }
      RecvTask dummy_recv;
      ctp::ipc::FullPtr<RecvTask> recv_fp(&dummy_recv);
      chi::MsgType msg_type = archive.GetMsgType();
      switch (msg_type) {
        case chi::MsgType::kSerializeIn:
          RecvIn(recv_fp, archive, lbm_transport);
          break;
        case chi::MsgType::kSerializeOut:
          RecvOut(recv_fp, archive, lbm_transport);
          break;
        case chi::MsgType::kHeartbeat:
          break;
        default:
          HLOG(kError, "[PeerRecvThread] unknown msg_type={}",
               static_cast<int>(msg_type));
          break;
      }
      // Release the per-bulk zmq_msg_t handles that ZeroMqTransport::RecvBulks
      // allocated for any borrowed (caller-didn't-preallocate) bulks. RecvIn /
      // RecvOut have already copied the bulk payloads into shm-owned task
      // buffers (via LoadTaskArchive::bulk -> CLIO_IPC->AllocateBuffer +
      // memcpy + TASK_DATA_OWNER), so dropping the ZMQ frames here is safe.
      // Without this, every PutBlob/GetBlob loopback round-trip leaks one
      // 2 MiB ZMQ-owned frame per direction (~8 MiB per op under
      // direct0+FORCE_NET) — archive goes out of scope after this iteration
      // anyway, but the desc-owned zmq_msg_t lives in libzmq memory that
      // only zmq_msg_close releases; the destructor can't reach it.
      lbm_transport->ClearRecvHandles(archive);
    }
    HLOG(kInfo, "[PeerRecvThread] shutting down");
  });

  // Dedicated single client recv thread: drains TCP (port 9416) and IPC
  // (unix socket) client transports via IpcCpu2CpuZmq::RuntimeRecv.
  client_recv_thread_ = std::thread([this]() {
    ctp::SystemInfo::SetCurrentThreadName("chi-client-recv");
    auto *ipc_manager = CLIO_IPC;
    HLOG(kInfo, "[ClientRecvThread] started");
    while (!recv_shutdown_.load(std::memory_order_acquire)) {
      chi::u32 tasks_received = 0;
      bool did_work = chi::IpcCpu2CpuZmq::RuntimeRecv(ipc_manager,
                                                       tasks_received);
      if (!did_work) {
        std::this_thread::sleep_for(std::chrono::microseconds(1));
      }
    }
    HLOG(kInfo, "[ClientRecvThread] shutting down");
  });

  // Stop these recv threads at the very start of IpcManager::ClearTransports(),
  // i.e. before the main transport they poll is freed. Without this, on
  // shutdown the threads keep calling Recv() on a freed transport -- benign on
  // Linux but a hard crash on macOS (busy-spin on ENOTSOCK over freed memory).
  CLIO_IPC->RegisterTransportShutdownHook([this]() { StopRecvThreads(); });

  // Spawn periodic WreapDeadIpcs task with 1 second period
  // This task reaps shared memory segments from dead processes
  client_.AsyncWreapDeadIpcs(chi::PoolQuery::Local(), 1000000);

  // Spawn periodic HeartbeatProbe task (SWIM failure detector, 2s period)
  client_.AsyncHeartbeatProbe(chi::PoolQuery::Local(), 2000000);

  // Initialize system stats ring buffer and spawn periodic monitor task
  system_stats_ring_ = std::make_unique<
      ctp::ipc::circular_mpsc_ring_buffer<SystemStats, ctp::ipc::MallocAllocator>>(
      CTP_MALLOC, kSystemStatsRingSize);
  prev_cpu_times_ = ctp::SystemInfo::GetCpuTimes();
  client_.AsyncSystemMonitor(chi::PoolQuery::Local(), 1000000);  // 1s

  HLOG(kDebug,
       "Admin: Container created and initialized for pool: {} (ID: {}, count: "
       "{})",
       pool_name_, task->new_pool_id_, create_count_);
  HLOG(kDebug,
       "Admin: Spawned periodic Recv, Send, ClientConnect, ClientRecv, "
       "ClientSend tasks");
  (void)rctx;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::PoolQuery Runtime::ScheduleTask(const ctp::ipc::FullPtr<chi::Task> &task) {
  using namespace clio::run::admin;
  switch (task->method_) {
    case Method::kGetOrCreatePool: {
      auto typed = task.template Cast<GetOrCreatePoolTask<CreateParams>>();
      auto *pool_manager = CLIO_POOL_MANAGER;
      std::string pool_name = typed->pool_name_.str();
      chi::PoolId existing_pool_id = pool_manager->FindPoolByName(pool_name);
      if (!existing_pool_id.IsNull()) {
        return chi::PoolQuery::Local();
      }
      return chi::PoolQuery::Broadcast();
    }
    default:
      return task->pool_query_;
  }
}

chi::TaskResume Runtime::GetOrCreatePool(
    ctp::ipc::FullPtr<
        clio::run::admin::GetOrCreatePoolTask<clio::run::admin::CreateParams>>
        task,
    chi::RunContext &rctx) {
  CLIO_TASK_BODY_BEGIN
  // Debug: Log do_compose_ value
  HLOG(kDebug,
       "Admin::GetOrCreatePool ENTRY: task->do_compose_={}, task->is_admin_={}",
       task->do_compose_, task->is_admin_);

  // Get pool manager and pool name
  auto *pool_manager = CLIO_POOL_MANAGER;
  std::string pool_name = task->pool_name_.str();

  // Pool get-or-create operation logic (IS_ADMIN=false)
  HLOG(kDebug, "Admin: Executing GetOrCreatePool task - ChiMod: {}, Pool: {}",
       task->chimod_name_.str(), pool_name);

  // Initialize output values
  task->return_code_ = 0;
  task->error_message_ = "";

  try {
    // Use the simplified PoolManager API that extracts all parameters from the
    // task. CreatePool is now a coroutine that co_awaits nested Create methods.
    CLIO_CO_AWAIT(pool_manager->CreatePool(task.Cast<chi::Task>(), &rctx));

    // Check if CreatePool set an error (return code is set on the task)
    if (task->return_code_ != 0) {
      // Error already set by CreatePool
      CLIO_CO_RETURN;
    }

    // Set success results (task->new_pool_id_ is already updated by CreatePool)
    task->return_code_ = 0;
    pools_created_++;

    HLOG(kDebug,
         "Admin: Pool operation completed successfully - ID: {}, Name: {} "
         "(Total pools created: {})",
         task->new_pool_id_, pool_name, pools_created_);

  } catch (const std::exception &e) {
    task->return_code_ = 99;
    std::string error_msg =
        std::string("Exception during pool creation: ") + e.what();
    task->error_message_ = chi::priv::string(CTP_MALLOC, error_msg);
    HLOG(kError, "Admin: Pool creation failed with exception: {}", e.what());
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::Destroy(ctp::ipc::FullPtr<DestroyTask> task,
                                 chi::RunContext &rctx) {
  CLIO_TASK_BODY_BEGIN
  // DestroyTask is aliased to DestroyPoolTask, so delegate to DestroyPool
  CLIO_CO_AWAIT(DestroyPool(task, rctx));
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::DestroyPool(ctp::ipc::FullPtr<DestroyPoolTask> task,
                                     chi::RunContext &rctx) {
  CLIO_TASK_BODY_BEGIN
  HLOG(kDebug, "Admin: Executing DestroyPool task - Pool ID: {}",
       task->target_pool_id_);

  // Initialize output values
  task->return_code_ = 0;
  task->error_message_ = "";

  try {
    chi::PoolId target_pool = task->target_pool_id_;

    // Get pool manager to handle pool destruction
    auto *pool_manager = CLIO_POOL_MANAGER;
    if (!pool_manager || !pool_manager->IsInitialized()) {
      task->return_code_ = 1;
      task->error_message_ = "Pool manager not available";
      CLIO_CO_RETURN;
    }

    // Use PoolManager to destroy the complete pool including metadata
    // DestroyPool is now a coroutine for consistency
    CLIO_CO_AWAIT(pool_manager->DestroyPool(target_pool));

    // Set success results
    task->return_code_ = 0;
    pools_destroyed_++;

    HLOG(kDebug,
         "Admin: Pool destroyed successfully - ID: {} (Total pools destroyed: "
         "{})",
         target_pool, pools_destroyed_);

  } catch (const std::exception &e) {
    task->return_code_ = 99;
    std::string error_msg =
        std::string("Exception during pool destruction: ") + e.what();
    task->error_message_ = chi::priv::string(CTP_MALLOC, error_msg);
    HLOG(kError, "Admin: Pool destruction failed with exception: {}", e.what());
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::StopRuntime(ctp::ipc::FullPtr<StopRuntimeTask> task,
                                     chi::RunContext &rctx) {
  CLIO_TASK_BODY_BEGIN
  HLOG(kDebug, "Admin: Executing StopRuntime task - Grace period: {}ms",
       task->grace_period_ms_);

  // Initialize output values
  task->return_code_ = 0;
  task->error_message_ = "";

  // Die immediately. SWIM will detect the death and trigger recovery.
  is_shutdown_requested_ = true;
  HLOG(kInfo, "Admin: Runtime shutdown initiated successfully");
  InitiateShutdown(task->grace_period_ms_);
  (void)rctx;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

void Runtime::InitiateShutdown(chi::u32 grace_period_ms) {
  HLOG(kDebug, "Admin: Initiating runtime shutdown with {}ms grace period",
       grace_period_ms);

  // In a real implementation, this would:
  // 1. Signal all worker threads to stop
  // 2. Wait for current tasks to complete (up to grace period)
  // 3. Clean up all resources
  // 4. Exit the runtime process

  // For now, we'll just set a flag that other components can check
  is_shutdown_requested_ = true;

  // Get CLIO Runtime manager to initiate shutdown
  auto *runtime_manager = CLIO_RUNTIME_MANAGER;
  if (runtime_manager) {
    // runtime_manager->InitiateShutdown(grace_period_ms);
  }
  std::abort();
}

chi::TaskResume Runtime::Flush(ctp::ipc::FullPtr<FlushTask> task,
                               chi::RunContext &rctx) {
  CLIO_TASK_BODY_BEGIN
  HLOG(kDebug, "Admin: Executing Flush task");

  // Initialize output values
  task->return_code_ = 0;
  task->total_work_done_ = 0;

  try {
    // Get WorkOrchestrator to check work remaining across all containers
    auto *work_orchestrator = CLIO_WORK_ORCHESTRATOR;
    if (!work_orchestrator || !work_orchestrator->IsInitialized()) {
      task->return_code_ = 1;
      CLIO_CO_RETURN;
    }

    // Loop until all work is complete
    chi::u64 total_work_remaining = 0;
    while (work_orchestrator->HasWorkRemaining(total_work_remaining)) {
      HLOG(kDebug,
           "Admin: Flush found {} work units still remaining, waiting...",
           total_work_remaining);

      // Brief yield to avoid busy waiting
      CLIO_CO_AWAIT(chi::yield(25));
    }

    // Store the final work count (should be 0)
    task->total_work_done_ = total_work_remaining;
    task->return_code_ = 0;  // Success - all work completed

    HLOG(kDebug,
         "Admin: Flush completed - no work remaining across all containers");

  } catch (const std::exception &e) {
    task->return_code_ = 99;
    HLOG(kError, "Admin: Flush failed with exception: {}", e.what());
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

//===========================================================================
// Distributed Task Scheduling Method Implementations
//===========================================================================

/**
 * Helper function: Send task inputs to remote node
 * @param origin_task Task to send to remote nodes
 * @param rctx RunContext for managing subtasks
 */
void Runtime::SendIn(ctp::ipc::FullPtr<chi::Task> origin_task,
                     chi::RunContext &rctx) {
  auto *ipc_manager = CLIO_IPC;
  auto *pool_manager = CLIO_POOL_MANAGER;

  // Validate origin_task
  if (origin_task.IsNull()) {
    HLOG(kError, "SendIn: origin_task is null");
    return;
  }

  // Get the container associated with the origin_task
  chi::Container *container =
      pool_manager->GetStaticContainer(origin_task->pool_id_);
  if (container == nullptr) {
    HLOG(kError, "SendIn: container not found for pool_id {}",
         origin_task->pool_id_);
    return;
  }

  // Pre-allocate send_map_key using origin_task pointer
  // This ensures consistent net_key across all replicas
  size_t send_map_key = size_t(origin_task.ptr_);

  // Add the origin task to send_map before creating copies.
  // SendIn runs on net_send_worker; RecvOut/ProcessRetryQueues touch this
  // map from other threads with the worker split, so the mutex is required.
  {
    std::lock_guard<std::mutex> lk(send_map_mutex_);
    send_map_[send_map_key] = origin_task;
  }

  // Get pool_queries from task's RunContext
  if (!origin_task->GetRunCtx()) {
    HLOG(kError, "SendIn: origin_task has no RunContext");
    return;
  }
  chi::RunContext *origin_task_rctx = origin_task->GetRunCtx();

  const std::vector<chi::PoolQuery> &pool_queries =
      origin_task_rctx->pool_queries_;
  size_t num_replicas = pool_queries.size();

  // Reserve space for all replicas in subtasks vector BEFORE the loop
  // This ensures subtasks_.size() reflects the correct total replica count
  origin_task_rctx->subtasks_.resize(num_replicas);

  HLOG(kDebug, "[SendIn] Task {} to {} replicas", origin_task->task_id_,
       num_replicas);

  // Send to each target in pool_queries
  for (size_t i = 0; i < num_replicas; ++i) {
    const chi::PoolQuery &query = pool_queries[i];

    // Determine target node_id based on query type
    chi::u64 target_node_id = 0;

    if (query.IsLocalMode()) {
      target_node_id = ipc_manager->GetNodeId();
    } else if (query.IsPhysicalMode()) {
      target_node_id = query.GetNodeId();
    } else if (query.IsDirectIdMode()) {
      chi::ContainerId container_id = query.GetContainerId();
      target_node_id =
          pool_manager->GetContainerNodeId(origin_task->pool_id_, container_id);
    } else if (query.IsRangeMode()) {
      chi::u32 offset = query.GetRangeOffset();
      chi::ContainerId container_id(offset);
      target_node_id =
          pool_manager->GetContainerNodeId(origin_task->pool_id_, container_id);
    } else if (query.IsBroadcastMode()) {
      HLOG(kError,
           "Admin: Broadcast mode should be handled by "
           "TaskDispatcher, not SendIn");
      continue;
    } else if (query.IsDirectHashMode()) {
      HLOG(kError,
           "Admin: DirectHash mode should be handled by "
           "TaskDispatcher, not SendIn");
      continue;
    } else {
      HLOG(kError, "Admin: Unsupported or unrecognized query type for SendIn");
      continue;
    }

    // Get host information for target node
    const chi::Host *target_host = ipc_manager->GetHost(target_node_id);
    if (!target_host) {
      HLOG(kError, "[SendIn] Task {} FAILED: Host not found for node_id {}",
           origin_task->task_id_, target_node_id);
      continue;
    }

    // Create task copy first (needed for both send and retry)
    ctp::ipc::FullPtr<chi::Task> task_copy =
        container->NewCopyTask(origin_task->method_, origin_task, true);
    origin_task_rctx->subtasks_[i] = task_copy;

    // Set net_key in task_id to match send_map_key
    chi::TaskId &copy_id = task_copy->task_id_;
    copy_id.net_key_ = send_map_key;
    copy_id.replica_id_ = i;

    // Update the copy's pool query to current query
    task_copy->pool_query_ = query;

    // Set return node ID in the pool query
    chi::u64 this_node_id = ipc_manager->GetNodeId();
    task_copy->pool_query_.SetReturnNode(this_node_id);

    // Check aliveness before sending
    if (!ipc_manager->IsAlive(target_node_id)) {
      float net_timeout = origin_task->pool_query_.GetNetTimeout();
      if (net_timeout >= 0 && net_timeout < 0.001f) {
        // net_timeout=0: fail immediately for dead nodes, skip this replica
        HLOG(kWarning,
             "[SendIn] Task {} target node {} is dead, net_timeout=0 -> skip",
             origin_task->task_id_, target_node_id);
        origin_task_rctx->completed_replicas_++;
        continue;
      }
      HLOG(kWarning,
           "[SendIn] Task {} target node {} is dead, queuing for retry",
           origin_task->task_id_, target_node_id);
      {
        std::lock_guard<std::mutex> _rqlk(retry_queues_mutex_);
        send_in_retry_.push_back(
            {task_copy, target_node_id, std::chrono::steady_clock::now()});
      }
      continue;
    }

    // Get or create persistent Lightbeam client using connection pool
    auto *config_manager = CLIO_CONFIG_MANAGER;
    int port = static_cast<int>(config_manager->GetPort());
    ctp::lbm::Transport *lbm_transport =
        ipc_manager->GetOrCreateClient(target_host->ip_address, port);

    if (!lbm_transport) {
      HLOG(kError, "[SendIn] Task {} FAILED: Could not get client for {}:{}",
           origin_task->task_id_, target_host->ip_address, port);
      ipc_manager->SetDead(target_node_id);
      {
        std::lock_guard<std::mutex> _rqlk(retry_queues_mutex_);
        send_in_retry_.push_back(
            {task_copy, target_node_id, std::chrono::steady_clock::now()});
      }
      continue;
    }

    // Create SaveTaskArchive with SerializeIn mode and lbm_transport
    chi::SaveTaskArchive archive(chi::MsgType::kSerializeIn, lbm_transport);

    uint64_t ser_t0 = HrtNs();
    container->SaveTask(task_copy->method_, archive, task_copy);
    uint64_t ser_dt = HrtNs() - ser_t0;

    // SYNC send: lightbeam copies bulks into ZMQ during this call. After
    // Send returns ZMQ holds no reference to task_copy's buffers, so the
    // copy's lifetime (already pinned by origin_task_rctx->subtasks_[i]
    // until the response is received) is unaffected by ZMQ's I/O thread.
    ctp::lbm::LbmContext ctx(ctp::lbm::LBM_SYNC);
    HLOG(kDebug, "[SendIn] Task {} sending to node {} via lightbeam",
         origin_task->task_id_, target_node_id);
    uint64_t lbm_t0 = HrtNs();
    int rc = lbm_transport->Send(archive, ctx);
    uint64_t lbm_dt = HrtNs() - lbm_t0;
    HLOG(kDebug, "[SendIn] Task {} lightbeam Send rc={}", origin_task->task_id_,
         rc);

    if (rc == 0) {
      auto &st = NetStats(target_node_id);
      st.sin_count += 1;
      st.sin_bytes += origin_task->GetRunCtx()
                          ? origin_task->GetRunCtx()->predicted_stat_.io_size_
                          : 0;
      st.sin_lbm_ns += lbm_dt;
      st.sin_ser_ns += ser_dt;
    }

    if (rc != 0) {
      HLOG(kWarning,
           "[SendIn] Task {} Lightbeam Send rc={} — re-queueing (no dead-mark)",
           origin_task->task_id_, rc);
      if (rc != EAGAIN) {
        ipc_manager->SetDead(target_node_id);
      }
      {
        std::lock_guard<std::mutex> _rqlk(retry_queues_mutex_);
        send_in_retry_.push_back(
            {task_copy, target_node_id, std::chrono::steady_clock::now()});
      }
      continue;
    }

    if (origin_task->method_ == Method::kHeartbeatProbe ||
        origin_task->method_ == Method::kProbeRequest ||
        origin_task->method_ == Method::kHeartbeat) {
      HLOG(kInfo,
           "[ProbeTrace SendIn] method={} task={} -> node={} sent OK",
           origin_task->method_, origin_task->task_id_, target_node_id);
    }

    {
      static std::atomic<size_t> ctr{0};
      size_t t = ctr.fetch_add(1, std::memory_order_relaxed) + 1;
      if ((t & 0xff) == 0) {
        HLOG(kDebug, "[CountSendIn] sent {} cross-node tasks to peers", t);
      }
    }
  }
}

/**
 * Helper function: Send task outputs back to origin node
 * @param origin_task Completed task whose outputs need to be sent back
 */
void Runtime::SendOut(ctp::ipc::FullPtr<chi::Task> origin_task) {
  auto *ipc_manager = CLIO_IPC;
  auto *pool_manager = CLIO_POOL_MANAGER;

  // Task lifetime across the zero-copy ZMQ send is handled by lightbeam
  // (via LbmContext::on_send_complete set just before Send below). The
  // transport keeps origin_task's serialization buffer alive via an
  // atomic refcount tied to each zmq_msg_init_data ref; when every bulk
  // has flushed (or been released via zmq_msg_close on a failure path)
  // the callback fires and DelTask runs.  No deferred-delete queue,
  // no time-based deferral — the same event-driven mechanism that
  // IpcCpu2CpuZmq::RuntimeSend uses on the client-response path.

  // Validate origin_task
  if (origin_task.IsNull()) {
    HLOG(kError, "SendOut: origin_task is null");
    return;
  }

  // Get the container associated with the origin_task
  chi::Container *container =
      pool_manager->GetStaticContainer(origin_task->pool_id_);
  if (container == nullptr) {
    HLOG(kError, "SendOut: container not found for pool_id {}",
         origin_task->pool_id_);
    return;
  }

  // Remove task from recv_map as we're completing it.
  // Erase is idempotent: SendOut can be called again from send_out_retry_
  // after IsAlive() or Lightbeam Send fails. The first call erases the
  // entry; subsequent retries no-op here and still attempt the send.
  // RecvIn writes recv_map_ from net_recv_worker, so the mutex serialises
  // the erase against concurrent inserts.
  size_t recv_key = origin_task->task_id_.net_key_ ^
                    (static_cast<size_t>(origin_task->task_id_.replica_id_) *
                     0x9e3779b97f4a7c15ULL);
  {
    std::lock_guard<std::mutex> lk(recv_map_mutex_);
    recv_map_.erase(recv_key);
  }

  // Get return node from pool_query
  chi::u64 target_node_id = origin_task->pool_query_.GetReturnNode();

  // Check aliveness before sending output back
  if (!ipc_manager->IsAlive(target_node_id)) {
    HLOG(kWarning,
         "[SendOut] Task {} return node {} is dead, queuing for retry",
         origin_task->task_id_, target_node_id);
    {
      std::lock_guard<std::mutex> _rqlk(retry_queues_mutex_);
      send_out_retry_.push_back(
          {origin_task, target_node_id, std::chrono::steady_clock::now()});
    }
    return;
  }

  // Get host information
  const chi::Host *target_host = ipc_manager->GetHost(target_node_id);
  if (target_host == nullptr) {
    HLOG(kError, "[SendOut] Task {} FAILED: Host not found for node_id {}",
         origin_task->task_id_, target_node_id);
    return;
  }

  auto *config_manager = CLIO_CONFIG_MANAGER;
  int port = static_cast<int>(config_manager->GetPort());
  ctp::lbm::Transport *lbm_transport =
      ipc_manager->GetOrCreateClient(target_host->ip_address, port);

  if (lbm_transport == nullptr) {
    HLOG(kError, "[SendOut] Task {} FAILED: Could not get client for {}:{}",
         origin_task->task_id_, target_host->ip_address, port);
    ipc_manager->SetDead(target_node_id);
    {
      std::lock_guard<std::mutex> _rqlk(retry_queues_mutex_);
      send_out_retry_.push_back(
          {origin_task, target_node_id, std::chrono::steady_clock::now()});
    }
    return;
  }

  chi::SaveTaskArchive archive(chi::MsgType::kSerializeOut, lbm_transport);

  uint64_t sout_ser_t0 = HrtNs();
  container->SaveTask(origin_task->method_, archive, origin_task);
  uint64_t sout_ser_dt = HrtNs() - sout_ser_t0;

  // SYNC send: lightbeam copies our bulks into ZMQ during the call and
  // holds no reference to origin_task after Send returns, so DelTask
  // runs deterministically below — no on_send_complete, no async
  // lifetime tracking, no UAF race with ZMQ's I/O thread.
  ctp::lbm::LbmContext ctx(ctp::lbm::LBM_SYNC);
  uint64_t sout_lbm_t0 = HrtNs();
  int rc = lbm_transport->Send(archive, ctx);
  uint64_t sout_lbm_dt = HrtNs() - sout_lbm_t0;

  if (rc == 0) {
    auto &st = NetStats(target_node_id);
    st.sout_count += 1;
    st.sout_bytes += origin_task->GetRunCtx()
                         ? origin_task->GetRunCtx()->predicted_stat_.io_size_
                         : 0;
    st.sout_lbm_ns += sout_lbm_dt;
    st.sout_ser_ns += sout_ser_dt;
  }
  if (rc != 0) {
    HLOG(kWarning,
         "[SendOut] Task {} Lightbeam Send rc={} — re-queueing (no dead-mark)",
         origin_task->task_id_, rc);
    if (rc != EAGAIN) {
      ipc_manager->SetDead(target_node_id);
    }
    {
      std::lock_guard<std::mutex> _rqlk(retry_queues_mutex_);
      send_out_retry_.push_back(
          {origin_task, target_node_id, std::chrono::steady_clock::now()});
    }
    return;
  }

  if (origin_task->method_ == Method::kHeartbeatProbe ||
      origin_task->method_ == Method::kProbeRequest ||
      origin_task->method_ == Method::kHeartbeat) {
    HLOG(kInfo,
         "[ProbeTrace SendOut] method={} task={} -> node={} sent OK",
         origin_task->method_, origin_task->task_id_, target_node_id);
  }

  {
    static std::atomic<size_t> ctr{0};
    size_t t = ctr.fetch_add(1, std::memory_order_relaxed) + 1;
    if ((t & 0xff) == 0) {
      HLOG(kDebug, "[CountSendOut] sent {} cross-node responses back", t);
    }
  }

  // SYNC send: Send returned, ZMQ has copied every bulk into its own
  // buffer, no outstanding reference to origin_task remains anywhere
  // outside us. Safe to DelTask synchronously.
  container->DelTask(origin_task->method_, origin_task);
}

/**
 * Main Send periodic — drains cross-node send queues on net_send_worker.
 *
 * Strategy (see NetQueuePriority for the lane definitions):
 *   1. Drain all kSendInLatency and kSendOutLatency tasks unbounded.
 *   2. Drain bulk lanes (kSendInIO + kSendOutIO) up to a byte budget
 *      (kNetQueueIoByteBudget, default 8 MiB) per tick, alternating
 *      SendIn / SendOut so neither direction starves.
 *   3. Yield. The next periodic tick re-checks latency first.
 *
 * Retry-queue + dead-node scans still run unconditionally each tick.
 *
 * Strategy (see NetQueuePriority for the lane definitions):
 *   1. Drain all kSendInLatency and kSendOutLatency tasks unbounded.
 *      These are small (<4 KiB) — SWIM heartbeats, ACKs, small metadata
 *      — so emptying them per tick is cheap and lets control-plane
 *      round-trips meet their SLA even under bulk-data pressure.
 *   2. Drain bulk lanes (kSendInIO + kSendOutIO) up to a byte budget
 *      (kNetQueueIoByteBudget, default 8 MiB) per tick, alternating
 *      SendIn / SendOut so neither direction starves.
 *   3. Yield. The next periodic tick re-checks latency first, so any
 *      probe enqueued during the I/O burst gets prompt attention.
 *
 * Retry-queue + dead-node scans still run unconditionally each tick.
 */
chi::TaskResume Runtime::Send(ctp::ipc::FullPtr<SendTask> task,
                              chi::RunContext &rctx) {
  CLIO_TASK_BODY_BEGIN
  auto *ipc_manager = CLIO_IPC;
  chi::Future<chi::Task> queued_future;
  bool did_send = false;

  // Per-tick maintenance: retries and dead-node fanout.
  ProcessRetryQueues();
  ScanSendMapTimeouts();

  // Snapshot the depth of each priority at function entry so a hot
  // producer can't monopolise this tick.
  const size_t n_in_lat =
      ipc_manager->GetNetQueueSize(chi::NetQueuePriority::kSendInLatency);
  const size_t n_out_lat =
      ipc_manager->GetNetQueueSize(chi::NetQueuePriority::kSendOutLatency);
  const size_t n_in_io =
      ipc_manager->GetNetQueueSize(chi::NetQueuePriority::kSendInIO);
  const size_t n_out_io =
      ipc_manager->GetNetQueueSize(chi::NetQueuePriority::kSendOutIO);

  // --- Phase 1: drain latency-lane sends, bounded by entry depth ------
  for (size_t i = 0; i < n_in_lat; ++i) {
    if (!ipc_manager->TryPopNetTask(chi::NetQueuePriority::kSendInLatency,
                                    queued_future)) {
      break;
    }
    auto origin_task = queued_future.GetTaskPtr();
    if (!origin_task.IsNull()) {
      SendIn(origin_task, rctx);
      did_send = true;
    }
  }
  for (size_t i = 0; i < n_out_lat; ++i) {
    if (!ipc_manager->TryPopNetTask(chi::NetQueuePriority::kSendOutLatency,
                                    queued_future)) {
      break;
    }
    auto origin_task = queued_future.GetTaskPtr();
    if (!origin_task.IsNull()) {
      SendOut(origin_task);
      did_send = true;
    }
  }

  // --- Phase 2: drain bulk I/O up to byte budget AND entry depth ------
  size_t io_budget = chi::kNetQueueIoByteBudget;
  const size_t io_bound = n_in_io + n_out_io;
  size_t io_in_remaining = n_in_io;
  size_t io_out_remaining = n_out_io;
  for (size_t i = 0; i < io_bound && io_budget > 0; ++i) {
    bool did_any = false;
    if (io_in_remaining > 0 &&
        ipc_manager->TryPopNetTask(chi::NetQueuePriority::kSendInIO,
                                   queued_future)) {
      auto origin_task = queued_future.GetTaskPtr();
      if (!origin_task.IsNull()) {
        size_t sz = origin_task->GetRunCtx()
                        ? origin_task->GetRunCtx()->predicted_stat_.io_size_
                        : 0;
        SendIn(origin_task, rctx);
        did_send = true;
        io_budget = (sz >= io_budget) ? 0 : (io_budget - sz);
        --io_in_remaining;
        did_any = true;
      }
    }
    if (io_budget == 0) break;
    if (io_out_remaining > 0 &&
        ipc_manager->TryPopNetTask(chi::NetQueuePriority::kSendOutIO,
                                   queued_future)) {
      auto origin_task = queued_future.GetTaskPtr();
      if (!origin_task.IsNull()) {
        size_t sz = origin_task->GetRunCtx()
                        ? origin_task->GetRunCtx()->predicted_stat_.io_size_
                        : 0;
        SendOut(origin_task);
        did_send = true;
        io_budget = (sz >= io_budget) ? 0 : (io_budget - sz);
        --io_out_remaining;
        did_any = true;
      }
    }
    if (!did_any) break;
  }

  rctx.did_work_ = did_send;
  {
    auto *cfg_ipc = CLIO_IPC;
    DumpNetStatsIfDue(cfg_ipc->GetNodeId(), cfg_ipc->GetNumHosts());
  }
  task->SetReturnCode(0);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

/**
 * Helper function: Receive task inputs from remote node
 * @param task RecvTask containing control information
 * @param archive Already-parsed LoadTaskArchive containing task info
 * @param lbm_transport Lightbeam server for receiving bulk data
 */
void Runtime::RecvIn(ctp::ipc::FullPtr<RecvTask> task,
                     chi::LoadTaskArchive &archive,
                     ctp::lbm::Transport *lbm_transport) {
  auto *ipc_manager = CLIO_IPC;
  auto *pool_manager = CLIO_POOL_MANAGER;

  const auto &task_infos = archive.GetTaskInfos();

  // If no tasks to receive
  if (task_infos.empty()) {
    task->SetReturnCode(0);
    return;
  }

  for (size_t task_idx = 0; task_idx < task_infos.size(); ++task_idx) {
    const auto &task_info = task_infos[task_idx];

    // Get container associated with PoolId
    chi::Container *container =
        pool_manager->GetStaticContainer(task_info.pool_id_);
    if (!container) {
      HLOG(kError, "Admin: Container not found for pool_id {}",
           task_info.pool_id_);
      continue;
    }

    // Instrumentation: time the deserialize+alloc step so the
    // dump can show whether the recv-side cost is dominated by
    // bulk copy or by task object construction.
    uint64_t des_t0 = HrtNs();
    // Call AllocLoadTask to allocate and deserialize the task
    ctp::ipc::FullPtr<chi::Task> task_ptr =
        container->AllocLoadTask(task_info.method_id_, archive);
    uint64_t des_dt = HrtNs() - des_t0;

    if (task_ptr.IsNull()) {
      HLOG(kError, "Admin: Failed to load task");
      continue;
    }
    {
      chi::u64 from_node = task_ptr->pool_query_.GetReturnNode();
      auto &st = NetStats(from_node);
      st.rin_count += 1;
      st.rin_bytes += task_ptr->GetRunCtx()
                          ? task_ptr->GetRunCtx()->predicted_stat_.io_size_
                          : 0;
      st.rin_des_ns += des_dt;
    }

    // If the sender is a node we marked kDead, mark it alive.
    // This handles restarted nodes: their SWIM probes reach us here,
    // letting us rediscover them without requiring --induct.
    // Only trigger on kDead (not kProbeFailed/kSuspected) to avoid
    // flip-flopping during normal SWIM failure detection.
    chi::u64 sender_node = task_ptr->pool_query_.GetReturnNode();
    if (sender_node != ipc_manager->GetNodeId() &&
        ipc_manager->GetNodeState(sender_node) == chi::NodeState::kDead) {
      HLOG(kInfo, "[RecvIn] Received task from dead node {}, marking alive",
           sender_node);
      // Flush stale retry entries BEFORE marking alive so
      // ProcessRetryQueues doesn't resend old tasks to the fresh runtime.
      FlushStaleStateForNode(sender_node);
      ipc_manager->SetAlive(sender_node);
    }

    // Mark task as remote. TASK_DATA_OWNER is set only if the archive
    // had to AllocateBuffer for any BULK_EXPOSE bulk (typical read
    // response path: GetBlob's blob_data_ is allocated daemon-side). If
    // the input was BULK_XFER (write data shipped from peer), the bulk
    // buffer lives in ZMQ-owned memory and FreeBuffer on it would crash,
    // so leave TASK_DATA_OWNER clear in that case.
    //
    // TASK_RUN_CTX_EXISTS and TASK_STARTED must be cleared so the
    // receiving worker allocates a fresh RunContext via BeginTask.
    chi::u32 set_flags = TASK_REMOTE;
    if (archive.daemon_allocated_bulk_count_ > 0) {
      set_flags |= TASK_DATA_OWNER;
    }
    task_ptr->SetFlags(set_flags);
    task_ptr->ClearFlags(TASK_PERIODIC | TASK_ROUTED |
                         TASK_RUN_CTX_EXISTS | TASK_STARTED);

    // Add task to recv_map for later lookup
    // Key combines net_key and replica_id so multiple replicas targeting the
    // same node (e.g., after container migration) get distinct entries.
    // RecvIn runs on net_recv_worker; SendOut reads/erases recv_map_ from
    // net_send_worker.
    size_t recv_key = task_ptr->task_id_.net_key_ ^
                      (static_cast<size_t>(task_ptr->task_id_.replica_id_) *
                       0x9e3779b97f4a7c15ULL);
    {
      std::lock_guard<std::mutex> lk(recv_map_mutex_);
      recv_map_[recv_key] = task_ptr;
    }

    HLOG(kDebug, "[RecvIn] Task {} method={} pool_id={} dispatching to workers",
         task_ptr->task_id_, task_ptr->method_, task_ptr->pool_id_);

    if (task_ptr->method_ == Method::kHeartbeatProbe ||
        task_ptr->method_ == Method::kProbeRequest ||
        task_ptr->method_ == Method::kHeartbeat) {
      HLOG(kInfo,
           "[ProbeTrace RecvIn] method={} task={} from node={}",
           task_ptr->method_, task_ptr->task_id_, sender_node);
    }

    // Dispatch the received task directly onto a worker lane.
    //
    // We deliberately do NOT call ipc_manager->Send here because RecvIn now
    // runs on a dedicated std::thread (CLIO_CUR_WORKER == nullptr). Send's
    // non-worker branch routes through IpcManager::RouteTask, which respects
    // pool_query_ and may re-broadcast a task that arrived at us because
    // we cleared TASK_ROUTED above. Mirror IpcCpu2Self::ClientSend's worker
    // branch directly: build a pointer-Future, pick a lane via
    // scheduler->ClientMapTask, and push (signal_mpmc_queue wakes the worker).
    // No parent RunContext is set (there is no current run context on a
    // recv thread, and these are top-level remote tasks anyway).
    chi::Future<chi::Task> future =
        ipc_manager->MakePointerFuture(task_ptr);
    if (future.GetFutureShm().IsNull()) {
      HLOG(kError, "[RecvIn] MakePointerFuture failed for task {}",
           task_ptr->task_id_);
      continue;
    }
    if (!task_ptr->task_flags_.Any(TASK_RUN_CTX_EXISTS)) {
      ipc_manager->BeginTask(future, nullptr, nullptr);
    }
    // ROUTED so the worker doesn't re-route this task to peers when it
    // sees pool_query_ pointing at another node.
    task_ptr->SetFlags(TASK_ROUTED);

    if (ipc_manager->GetScheduler() != nullptr) {
      chi::u32 lane_id =
          ipc_manager->GetScheduler()->ClientMapTask(ipc_manager, future);
      auto *worker_queues = ipc_manager->GetTaskQueue();
      if (worker_queues) {
        auto &dest_lane = worker_queues->GetLane(lane_id, 0);
        dest_lane.Push(future);
      }
    }
  }

  task->SetReturnCode(0);
}

/**
 * Helper function: Receive task outputs from remote node
 * @param task RecvTask containing control information
 * @param archive Already-parsed LoadTaskArchive containing task info
 * @param lbm_transport Lightbeam server for receiving bulk data
 */
void Runtime::RecvOut(ctp::ipc::FullPtr<RecvTask> task,
                      chi::LoadTaskArchive &archive,
                      ctp::lbm::Transport *lbm_transport) {
  auto *pool_manager = CLIO_POOL_MANAGER;

  const auto &task_infos = archive.GetTaskInfos();

  // If no task outputs to receive
  if (task_infos.empty()) {
    task->SetReturnCode(0);
    return;
  }

  // Set lbm_transport in archive for bulk transfer exposure in output mode
  archive.SetTransport(lbm_transport);

  // First pass: Deserialize to expose buffers
  // LoadTask will call ar.bulk() which will expose the pointers and populate
  // archive.recv
  for (size_t task_idx = 0; task_idx < task_infos.size(); ++task_idx) {
    const auto &task_info = task_infos[task_idx];

    // Locate origin task from send_map using net_key
    size_t net_key = task_info.task_id_.net_key_;

    // SendIn (net_send_worker) populates send_map_; copy the value out
    // under the lock so the rest of the work happens without holding it.
    ctp::ipc::FullPtr<chi::Task> origin_task;
    {
      std::lock_guard<std::mutex> lk(send_map_mutex_);
      auto send_it = send_map_.find(net_key);
      if (send_it == nullptr) {
        HLOG(kError,
             "[RecvOut] Task {} FAILED: Origin task not found in send_map "
             "(size: {}) with net_key {}",
             task_info.task_id_, send_map_.size(), net_key);
        task->SetReturnCode(5);
        return;
      }
      origin_task = *send_it;
    }
    if (!origin_task->GetRunCtx()) {
      HLOG(kError, "Admin: origin_task has no RunContext");
      task->SetReturnCode(6);
      return;
    }
    chi::RunContext *origin_rctx = origin_task->GetRunCtx();

    // Locate replica in origin's run_ctx using replica_id
    chi::u32 replica_id = task_info.task_id_.replica_id_;
    if (replica_id >= origin_rctx->subtasks_.size()) {
      HLOG(kError, "Admin: Invalid replica_id {} (subtasks size: {})",
           replica_id, origin_rctx->subtasks_.size());
      task->SetReturnCode(7);
      return;
    }

    ctp::ipc::FullPtr<chi::Task> replica = origin_rctx->subtasks_[replica_id];

    // Get the container associated with the origin task
    chi::Container *container =
        pool_manager->GetStaticContainer(origin_task->pool_id_);
    if (!container) {
      HLOG(kError, "Admin: Container not found for pool_id {}",
           origin_task->pool_id_);
      task->SetReturnCode(8);
      return;
    }

    // Deserialize outputs directly into the replica task using LoadTask
    // This exposes buffers via ar.bulk() and populates archive.recv.
    // Instrumentation: time the deserialize for rout accounting.
    uint64_t rout_des_t0 = HrtNs();
    container->LoadTask(origin_task->method_, archive, replica);
    uint64_t rout_des_dt = HrtNs() - rout_des_t0;
    {
      // task_info has sender node via task_id_.node_id_ (the origin node).
      chi::u64 from_node = task_info.task_id_.node_id_;
      auto &st = NetStats(from_node);
      st.rout_count += 1;
      st.rout_bytes += origin_task->GetRunCtx()
                           ? origin_task->GetRunCtx()->predicted_stat_.io_size_
                           : 0;
      st.rout_des_ns += rout_des_dt;
    }

  }

  // Second pass: Aggregate results
  for (size_t task_idx = 0; task_idx < task_infos.size(); ++task_idx) {
    const auto &task_info = task_infos[task_idx];

    // Locate origin task from send_map using net_key. Same locking note as
    // the first pass — copy out under the lock, then process unlocked.
    size_t net_key = task_info.task_id_.net_key_;
    ctp::ipc::FullPtr<chi::Task> origin_task;
    {
      std::lock_guard<std::mutex> lk(send_map_mutex_);
      auto send_it = send_map_.find(net_key);
      if (send_it == nullptr) {
        HLOG(kError, "Admin: Origin task not found in send_map with net_key {}",
             net_key);
        continue;
      }
      origin_task = *send_it;
    }
    if (!origin_task->GetRunCtx()) {
      HLOG(kError, "Admin: origin_task has no RunContext");
      continue;
    }
    chi::RunContext *origin_rctx = origin_task->GetRunCtx();

    // Locate replica in origin's run_ctx using replica_id
    chi::u32 replica_id = task_info.task_id_.replica_id_;
    if (replica_id >= origin_rctx->subtasks_.size()) {
      HLOG(kError, "Admin: Invalid replica_id {} (subtasks size: {})",
           replica_id, origin_rctx->subtasks_.size());
      continue;
    }

    ctp::ipc::FullPtr<chi::Task> replica = origin_rctx->subtasks_[replica_id];

    // Get the container associated with the origin task
    chi::Container *container =
        pool_manager->GetStaticContainer(origin_task->pool_id_);
    if (!container) {
      HLOG(kError, "Admin: Container not found for pool_id {}",
           origin_task->pool_id_);
      continue;
    }

    // Aggregate replica results into origin task via container dispatch
    container->Aggregate(origin_task->method_, origin_task, replica);

    HLOG(kDebug, "[RecvOut] Task {}", origin_task->task_id_);

    // Increment completed replicas counter in origin's rctx.
    // Use the fetch_add return so we observe THIS thread's increment —
    // a separate load can race a parallel RecvOut/SendIn-skip increment
    // and miss the "all done" trigger condition.
    chi::u32 completed = origin_rctx->completed_replicas_.fetch_add(1) + 1;

    // If all replicas completed
    if (completed == origin_rctx->subtasks_.size()) {
      // Get pool manager to access container
      auto *pool_manager = CLIO_POOL_MANAGER;
      chi::Container *container =
          pool_manager->GetStaticContainer(origin_task->pool_id_);

      // Unmark TASK_DATA_OWNER before deleting replicas to avoid freeing the
      // same data pointers twice. Delete all origin_task replicas using
      // container->DelTask() to avoid memory leak
      for (const auto &origin_task_ptr : origin_rctx->subtasks_) {
        origin_task_ptr->ClearFlags(TASK_DATA_OWNER);
        container->DelTask(origin_task->method_, origin_task_ptr);
      }

      // Clear subtasks vector after deleting tasks
      origin_rctx->subtasks_.clear();

      // Remove origin from send_map; serialised against SendIn insertions
      // and ScanSendMapTimeouts iteration on the send worker.
      {
        std::lock_guard<std::mutex> lk(send_map_mutex_);
        send_map_.erase(net_key);
      }

      // Set container in origin RunContext (may be null if task was routed
      // globally without passing through RouteLocal — e.g. when the runtime
      // is started with CLIO_FORCE_NET=1)
      if (container) {
        origin_rctx->container_ = container;
      }

      // Complete the origin task via EndTask. When RecvOut runs on the
      // dedicated peer recv thread (CLIO_CUR_WORKER == nullptr), borrow the
      // net_recv_worker pointer for the call. EndTask's per-worker state
      // updates (num_tasks_processed_, load_) are stats only — the
      // correctness-bearing paths (EnqueueNetTask for remote origin,
      // IpcCpu2Self::RuntimeSend for local origin) are thread-safe and
      // don't depend on CLIO_CUR_WORKER being correct. The net_recv_worker
      // already serialized incoming network responses pre-split, so this
      // approximates the prior accounting.
      auto *worker = CLIO_CUR_WORKER;
      if (worker == nullptr) {
        auto *scheduler = CLIO_IPC->GetScheduler();
        worker = scheduler ? scheduler->GetNetRecvWorker() : nullptr;
      }
      if (worker) {
        worker->EndTask(origin_task, origin_rctx, true);
      } else {
        HLOG(kError,
             "[RecvOut] No worker available to call EndTask for task {}",
             origin_task->task_id_);
      }
    }
  }

  task->SetReturnCode(0);
}

/**
 * Main Recv function - receives metadata and dispatches based on mode
 * Note: This is a periodic task - only logs when actual work is done
 */
chi::TaskResume Runtime::Recv(ctp::ipc::FullPtr<RecvTask> task,
                              chi::RunContext &rctx) {
  CLIO_TASK_BODY_BEGIN
  // Get the main server from CLIO_IPC (already bound during initialization)
  auto *ipc_manager = CLIO_IPC;

  ctp::lbm::Transport *lbm_transport = ipc_manager->GetMainTransport();
  if (lbm_transport == nullptr) {
    CLIO_CO_RETURN;
  }

  // Note: No socket lock needed - single net worker processes all Recv tasks

  // Receive metadata + bulks (non-blocking)
  chi::LoadTaskArchive archive;
  auto info = lbm_transport->Recv(archive);
  int rc = info.rc;
  if (rc == EAGAIN) {
    // No message available - this is normal for polling, mark as no work done
    task->SetReturnCode(0);
    rctx.did_work_ = false;
    CLIO_CO_RETURN;
  }

  if (rc != 0) {
    if (rc != -1) {
      HLOG(kError, "Admin: Lightbeam Recv failed with error code {}", rc);
    }
    task->SetReturnCode(2);
    rctx.did_work_ = false;
    CLIO_CO_RETURN;
  }

  // Mark that we received data (did work)
  rctx.did_work_ = true;

  chi::MsgType msg_type = archive.GetMsgType();
  HLOG(kDebug, "[Recv] Received message with msg_type={}",
       static_cast<int>(msg_type));

  {
    static std::atomic<size_t> ctr_in{0};
    static std::atomic<size_t> ctr_out{0};
    if (msg_type == chi::MsgType::kSerializeIn) {
      size_t t = ctr_in.fetch_add(1, std::memory_order_relaxed) + 1;
      if ((t & 0xff) == 0)
        HLOG(kDebug, "[CountRecvIn] received {} cross-node task inputs", t);
    } else if (msg_type == chi::MsgType::kSerializeOut) {
      size_t t = ctr_out.fetch_add(1, std::memory_order_relaxed) + 1;
      if ((t & 0xff) == 0)
        HLOG(kDebug, "[CountRecvOut] received {} cross-node task outputs", t);
    }
  }

  // Dispatch based on message type
  switch (msg_type) {
    case chi::MsgType::kSerializeIn:
      HLOG(kDebug, "[Recv] Dispatching to RecvIn");
      RecvIn(task, archive, lbm_transport);
      break;
    case chi::MsgType::kSerializeOut:
      HLOG(kDebug, "[Recv] Dispatching to RecvOut");
      RecvOut(task, archive, lbm_transport);
      break;
    case chi::MsgType::kHeartbeat:
      task->SetReturnCode(0);
      break;
    default:
      HLOG(kError, "Admin: Unknown message type in Recv");
      task->SetReturnCode(3);
      break;
  }

  // The dedicated PeerRecvThread (admin_runtime.cc:249) is the real recv
  // loop for inbound peer/loopback messages; this Recv periodic task path
  // is reserved for legacy / non-peer flows that don't allocate zmq_msg_t
  // frames. No ClearRecvHandles needed here.

  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

/**
 * Handle ClientConnect - Respond to client connection request
 * Polls connect server for ZMQ REQ/REP requests and responds
 * @param task The connect task
 * @param rctx Run context
 */
chi::TaskResume Runtime::ClientConnect(ctp::ipc::FullPtr<ClientConnectTask> task,
                                       chi::RunContext &rctx) {
  CLIO_TASK_BODY_BEGIN
  task->response_ = 0;
  task->server_generation_ = CLIO_IPC->GetServerGeneration();
  task->server_pid_ = static_cast<int32_t>(getpid());
  task->worker_queues_off_ = CLIO_IPC->GetWorkerQueuesOffset();

  // GPU queue info for client attachment.
  //
  // Producer-only redesign: clients no longer attach to host-managed
  // cpu2gpu / gpu2gpu backends. They allocate their own device-memory
  // backends and register them via admin RegisterMemory. The fields
  // below are zeroed so legacy clients see "no GPU queues to attach".
  task->num_gpus_ = 0;
  task->gpu_queue_depth_ = 0;

  task->SetReturnCode(0);
  rctx.did_work_ = true;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

/**
 * Handle ClientRecv - Receive tasks from lightbeam client servers
 * Delegates to IpcCpu2CpuZmq::RuntimeRecv for the actual transport logic.
 */
chi::TaskResume Runtime::ClientRecv(ctp::ipc::FullPtr<ClientRecvTask> task,
                                    chi::RunContext &rctx) {
  chi::u32 tasks_received = 0;
  bool did_work = chi::IpcCpu2CpuZmq::RuntimeRecv(CLIO_IPC, tasks_received);
  task->tasks_received_ = tasks_received;

  rctx.did_work_ = did_work;
  task->SetReturnCode(0);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

/**
 * Handle ClientSend - Send completed task outputs to clients via lightbeam
 * Delegates to IpcCpu2CpuZmq::RuntimeSend for the actual transport logic.
 */
chi::TaskResume Runtime::ClientSend(ctp::ipc::FullPtr<ClientSendTask> task,
                                    chi::RunContext &rctx) {
  static std::vector<ctp::ipc::FullPtr<chi::Task>> deferred_deletes;
  chi::u32 tasks_sent = 0;
  bool did_work = chi::IpcCpu2CpuZmq::RuntimeSend(
      CLIO_IPC, tasks_sent, deferred_deletes);
  task->tasks_sent_ = tasks_sent;

  rctx.did_work_ = did_work;
  task->SetReturnCode(0);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::Monitor(ctp::ipc::FullPtr<MonitorTask> task,
                                 chi::RunContext &rctx) {
  CLIO_TASK_BODY_BEGIN
  if (task->query_ == "worker_stats") {
    MonitorWorkerStats(task);
  } else if (task->query_.rfind("pool_stats://", 0) == 0) {
    CLIO_CO_AWAIT(MonitorPoolStats(task));
  } else if (task->query_.rfind("system_stats", 0) == 0) {
    MonitorSystemStats(task);
  } else if (task->query_ == "bdev_stats") {
    CLIO_CO_AWAIT(MonitorBdevStats(task));
  } else if (task->query_ == "container_stats") {
    MonitorContainerStats(task);
  } else if (task->query_ == "get_host_info") {
    MonitorGetHostInfo(task);
  } else {
    // Unknown queries get empty results (forward-compatible)
    task->SetReturnCode(0);
  }
  (void)rctx;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

void Runtime::MonitorWorkerStats(ctp::ipc::FullPtr<MonitorTask> task) {
  auto *work_orchestrator = CLIO_WORK_ORCHESTRATOR;
  if (!work_orchestrator) {
    task->SetReturnCode(1);
    HLOG(kError, "Monitor(worker_stats): WorkOrchestrator not available");
    return;
  }

  size_t num_workers = work_orchestrator->GetWorkerCount();

  msgpack::sbuffer sbuf;
  msgpack::packer<msgpack::sbuffer> pk(sbuf);
  pk.pack_array(num_workers);

  for (size_t i = 0; i < num_workers; ++i) {
    chi::Worker *worker =
        work_orchestrator->GetWorker(static_cast<chi::u32>(i));
    if (!worker) {
      pk.pack_map(0);
      continue;
    }
    chi::WorkerStats stats = worker->GetWorkerStats();
    pk.pack_map(11);
    pk.pack("worker_id");
    pk.pack(stats.worker_id_);
    pk.pack("is_running");
    pk.pack(stats.is_running_);
    pk.pack("is_active");
    pk.pack(stats.is_active_);
    pk.pack("idle_iterations");
    pk.pack(stats.idle_iterations_);
    pk.pack("num_queued_tasks");
    pk.pack(stats.num_queued_tasks_);
    pk.pack("num_blocked_tasks");
    pk.pack(stats.num_blocked_tasks_);
    pk.pack("num_periodic_tasks");
    pk.pack(stats.num_periodic_tasks_);
    pk.pack("num_retry_tasks");
    pk.pack(stats.num_retry_tasks_);
    pk.pack("suspend_period_us");
    pk.pack(stats.suspend_period_us_);
    pk.pack("num_tasks_processed");
    pk.pack(stats.num_tasks_processed_);
    pk.pack("load");
    pk.pack(stats.load_);
  }

  task->results_[container_id_] = std::string(sbuf.data(), sbuf.size());
}

void Runtime::MonitorContainerStats(ctp::ipc::FullPtr<MonitorTask> task) {
  auto *pool_manager = CLIO_POOL_MANAGER;
  if (!pool_manager) {
    task->SetReturnCode(1);
    return;
  }

  msgpack::sbuffer sbuf;
  msgpack::packer<msgpack::sbuffer> pk(sbuf);

  auto pool_ids = pool_manager->GetAllPoolIds();
  pk.pack_array(pool_ids.size());

  for (const auto &pid : pool_ids) {
    const auto *info = pool_manager->GetPoolInfo(pid);
    if (!info) continue;

    // Get the static container for model data
    chi::Container *container = pool_manager->GetStaticContainer(pid);

    pk.pack_map(6);

    pk.pack("pool_id");
    pk.pack(pid.ToString());

    pk.pack("pool_name");
    pk.pack(info->pool_name_);

    pk.pack("chimod_name");
    pk.pack(info->chimod_name_);

    pk.pack("container_id");
    pk.pack(container ? container->container_id_ : 0u);

    // Model data: array of per-method entries
    if (container) {
      const auto &model = container->GetMethodModel();
      const auto &mape = container->GetMethodMapeVec();
      const auto &model_wall = container->GetMethodModelWall();
      const auto &mape_wall = container->GetMethodMapeWallVec();
      const auto &names = container->GetMethodNames();

      pk.pack("methods");
      pk.pack_array(model.size());
      for (size_t i = 0; i < model.size(); ++i) {
        pk.pack_map(6);
        pk.pack("id");
        pk.pack(static_cast<uint32_t>(i));
        pk.pack("name");
        pk.pack(i < names.size() ? names[i] : std::string());
        pk.pack("coefficient");
        pk.pack(model[i]);
        pk.pack("mape");
        pk.pack(i < mape.size() ? mape[i] : 0.0f);
        pk.pack("wall_coefficient");
        pk.pack(i < model_wall.size() ? model_wall[i] : 0.0f);
        pk.pack("wall_mape");
        pk.pack(i < mape_wall.size() ? mape_wall[i] : 0.0f);
      }

      pk.pack("learning_rate");
      pk.pack(container->GetLearningRate());
    } else {
      pk.pack("methods");
      pk.pack_array(0);
      pk.pack("learning_rate");
      pk.pack(0.0f);
    }
  }

  task->results_[container_id_] = std::string(sbuf.data(), sbuf.size());
}

chi::TaskResume Runtime::MonitorPoolStats(ctp::ipc::FullPtr<MonitorTask> task) {
#ifdef __NVCOMPILER
  chi::RunContext _dummy_rctx;
  chi::RunContext& rctx = _dummy_rctx;
#endif
  CLIO_TASK_BODY_BEGIN
  // Parse pool_stats://PoolId:PoolQuery:selector
  // Format: pool_stats://<major.minor>:<routing_mode[:params...]>:<selector>
  std::string uri_body = task->query_.substr(13);  // skip "pool_stats://"

  // 1. Extract PoolId (everything before first ':')
  size_t first_colon = uri_body.find(':');
  if (first_colon == std::string::npos) {
    task->SetReturnCode(2);
    HLOG(kError, "Monitor(pool_stats): missing ':' after PoolId in '{}'",
         task->query_);
    CLIO_CO_RETURN;
  }
  std::string pool_id_str = uri_body.substr(0, first_colon);
  chi::PoolId target_pool_id;
  try {
    target_pool_id = chi::PoolId::FromString(pool_id_str);
  } catch (const std::exception &e) {
    task->SetReturnCode(2);
    HLOG(kError, "Monitor(pool_stats): invalid PoolId '{}': {}", pool_id_str,
         e.what());
    CLIO_CO_RETURN;
  }

  // 2. Token-based parse of routing mode and its parameters
  //    Tokens after PoolId: routing_mode[:extra1[:extra2]]:selector
  //    local/broadcast/dynamic = 0 extra tokens
  //    direct_id/direct_hash/physical = 1 extra token
  //    range = 2 extra tokens
  std::string remainder = uri_body.substr(first_colon + 1);
  size_t colon_pos = remainder.find(':');
  std::string routing_token = (colon_pos != std::string::npos)
                                  ? remainder.substr(0, colon_pos)
                                  : remainder;
  // Lowercase for comparison
  std::string routing_lower = routing_token;
  std::transform(routing_lower.begin(), routing_lower.end(),
                 routing_lower.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  // Determine how many extra colon-separated tokens the mode consumes
  int extra_tokens = 0;
  if (routing_lower == "direct_id" || routing_lower == "direct_hash" ||
      routing_lower == "physical") {
    extra_tokens = 1;
  } else if (routing_lower == "range") {
    extra_tokens = 2;
  }

  // Consume routing_token + extra tokens to build PoolQuery string
  std::string pool_query_str = routing_token;
  size_t parse_pos =
      (colon_pos != std::string::npos) ? colon_pos + 1 : remainder.size();
  for (int i = 0; i < extra_tokens; ++i) {
    if (parse_pos >= remainder.size()) {
      task->SetReturnCode(2);
      HLOG(kError,
           "Monitor(pool_stats): not enough tokens for routing mode '{}' "
           "in '{}'",
           routing_token, task->query_);
      CLIO_CO_RETURN;
    }
    size_t next_colon = remainder.find(':', parse_pos);
    std::string token =
        (next_colon != std::string::npos)
            ? remainder.substr(parse_pos, next_colon - parse_pos)
            : remainder.substr(parse_pos);
    pool_query_str += ":" + token;
    parse_pos =
        (next_colon != std::string::npos) ? next_colon + 1 : remainder.size();
  }

  // Everything remaining is the selector
  std::string selector;
  if (parse_pos < remainder.size()) {
    selector = remainder.substr(parse_pos);
  }

  // 3. Build PoolQuery from string
  chi::PoolQuery target_pool_query;
  try {
    target_pool_query = chi::PoolQuery::FromString(pool_query_str);
  } catch (const std::exception &e) {
    task->SetReturnCode(2);
    HLOG(kError, "Monitor(pool_stats): invalid PoolQuery '{}': {}",
         pool_query_str, e.what());
    CLIO_CO_RETURN;
  }

  // 4. Verify the target pool exists
  auto *pool_manager = CLIO_POOL_MANAGER;
  chi::Container *container = pool_manager->GetStaticContainer(target_pool_id);
  if (!container) {
    task->SetReturnCode(3);
    HLOG(kError, "Monitor(pool_stats): pool {} not found", target_pool_id);
    CLIO_CO_RETURN;
  }

  // 5. Create sub-MonitorTask targeting the pool and dispatch it
  auto *ipc_manager = CLIO_IPC;
  auto sub_task = ipc_manager->NewTask<MonitorTask>(
      chi::CreateTaskId(), target_pool_id, target_pool_query, selector);
  chi::Future<MonitorTask> sub_future = ipc_manager->Send(sub_task);
  CLIO_CO_AWAIT(sub_future);

  // 6. Copy results from sub-task into this task
  if (sub_future->GetReturnCode() != 0) {
    task->SetReturnCode(sub_future->GetReturnCode());
    HLOG(kError, "Monitor(pool_stats): sub-task failed with rc={}",
         sub_future->GetReturnCode());
  } else {
    task->results_ = sub_future->results_;
    task->SetReturnCode(0);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

void Runtime::MonitorSystemStats(ctp::ipc::FullPtr<MonitorTask> task) {
  // system_stats or system_stats:<min_event_id>
  uint64_t min_event_id = 0;
  if (task->query_.size() > 13 && task->query_[12] == ':') {
    try {
      min_event_id = std::stoull(task->query_.substr(13));
    } catch (...) {
      // ignore parse errors, default to 0
    }
  }

  if (!system_stats_ring_) {
    task->SetReturnCode(1);
    return;
  }

  chi::u64 head = system_stats_ring_->GetHead();
  chi::u64 tail = system_stats_ring_->GetTail();
  chi::u64 start = (min_event_id > head) ? min_event_id : head;

  msgpack::sbuffer sbuf;
  msgpack::packer<msgpack::sbuffer> pk(sbuf);

  // Count entries first
  uint64_t count = (tail > start) ? (tail - start) : 0;
  pk.pack_array(static_cast<uint32_t>(count));

  for (chi::u64 idx = start; idx < tail; ++idx) {
    SystemStats s;
    if (system_stats_ring_->Peek(idx, s)) {
      pk.pack_map(16);
      pk.pack("event_id");
      pk.pack(idx);
      pk.pack("timestamp_ns");
      pk.pack(s.timestamp_ns_);
      pk.pack("wall_time_ns");
      pk.pack(s.wall_time_ns_);
      pk.pack("ram_total_bytes");
      pk.pack(s.ram_total_bytes_);
      pk.pack("ram_available_bytes");
      pk.pack(s.ram_available_bytes_);
      pk.pack("ram_usage_pct");
      pk.pack(s.ram_usage_pct_);
      pk.pack("cpu_usage_pct");
      pk.pack(s.cpu_usage_pct_);
      pk.pack("gpu_count");
      pk.pack(s.gpu_count_);
      pk.pack("gpu_usage_pct");
      pk.pack(s.gpu_usage_pct_);
      pk.pack("hbm_usage_pct");
      pk.pack(s.hbm_usage_pct_);
      pk.pack("hbm_used_bytes");
      pk.pack(s.hbm_used_bytes_);
      pk.pack("hbm_total_bytes");
      pk.pack(s.hbm_total_bytes_);
      pk.pack("hostname");
      pk.pack(CLIO_IPC->GetCurrentHostname());
      pk.pack("ip_address");
      pk.pack(CLIO_IPC->GetThisHost().ip_address);
      pk.pack("node_id");
      pk.pack(CLIO_IPC->GetNodeId());
      pk.pack("is_leader");
      pk.pack(CLIO_IPC->IsLeader());
    } else {
      pk.pack_map(0);
    }
  }

  task->results_[container_id_] = std::string(sbuf.data(), sbuf.size());
  task->SetReturnCode(0);
}

void Runtime::MonitorGetHostInfo(ctp::ipc::FullPtr<MonitorTask> task) {
  msgpack::sbuffer sbuf;
  msgpack::packer<msgpack::sbuffer> pk(sbuf);
  pk.pack_map(3);
  pk.pack("hostname");
  pk.pack(CLIO_IPC->GetCurrentHostname());
  pk.pack("ip_address");
  pk.pack(CLIO_IPC->GetThisHost().ip_address);
  pk.pack("node_id");
  pk.pack(CLIO_IPC->GetNodeId());
  task->results_[container_id_] = std::string(sbuf.data(), sbuf.size());
  task->SetReturnCode(0);
}

chi::TaskResume Runtime::AnnounceShutdown(
    ctp::ipc::FullPtr<AnnounceShutdownTask> task, chi::RunContext &rctx) {
  CLIO_TASK_BODY_BEGIN
  chi::u64 dead_node_id = task->shutting_down_node_id_;
  auto *ipc_manager = CLIO_IPC;

  HLOG(kInfo, "Admin: Received shutdown announcement for node {}",
       dead_node_id);

  // Mark the node as dead immediately — skips the SWIM detection delay
  ipc_manager->SetDead(dead_node_id);

  // If we are the new leader, trigger recovery for the departing node
  if (ipc_manager->IsLeader()) {
    CLIO_CO_AWAIT(TriggerRecovery(dead_node_id));
  }

  task->SetReturnCode(0);
  (void)rctx;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::MonitorBdevStats(ctp::ipc::FullPtr<MonitorTask> task) {
#ifdef __NVCOMPILER
  chi::RunContext _dummy_rctx;
  chi::RunContext& rctx = _dummy_rctx;
#endif
  CLIO_TASK_BODY_BEGIN
  // Collect stats from all bdev pools on this node
  auto *pool_manager = CLIO_POOL_MANAGER;
  auto *ipc_manager = CLIO_IPC;
  auto all_pool_ids = pool_manager->GetAllPoolIds();
  msgpack::sbuffer sbuf;
  msgpack::packer<msgpack::sbuffer> pk(sbuf);

  // Collect bdev pool ids first
  std::vector<chi::PoolId> bdev_pools;
  for (const auto &pid : all_pool_ids) {
    const auto *info = pool_manager->GetPoolInfo(pid);
    if (info && (info->chimod_name_ == "clio_bdev" || info->chimod_name_ == "chimaera_bdev")) {
      bdev_pools.push_back(pid);
    }
  }

  pk.pack_array(static_cast<uint32_t>(bdev_pools.size()));

  for (size_t i = 0; i < bdev_pools.size(); ++i) {
    const auto &pid = bdev_pools[i];
    const auto *info = pool_manager->GetPoolInfo(pid);
    // Create sub-MonitorTask targeting this bdev pool (local routing)
    chi::PoolQuery bdev_query;  // default = Local routing
    auto sub_task = ipc_manager->NewTask<MonitorTask>(chi::CreateTaskId(), pid,
                                                      bdev_query, "stats");
    chi::Future<MonitorTask> sub_future = ipc_manager->Send(sub_task);
    CLIO_CO_AWAIT(sub_future);

    if (sub_future->GetReturnCode() == 0 && !sub_future->results_.empty()) {
      // Wrap bdev stats with pool_id metadata
      const auto &bdev_result = sub_future->results_.begin()->second;
      pk.pack_map(2);
      pk.pack("pool_id");
      pk.pack(pid.ToString());
      pk.pack("stats");
      pk.pack_raw_msgpack(bdev_result);
    } else {
      // Pack an empty entry on failure
      pk.pack_map(2);
      pk.pack("pool_id");
      pk.pack(info ? info->pool_name_ : "unknown");
      pk.pack("stats");
      pk.pack_nil();
    }
  }

  task->results_[container_id_] = std::string(sbuf.data(), sbuf.size());
  task->SetReturnCode(0);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::SubmitBatch(ctp::ipc::FullPtr<SubmitBatchTask> task,
                                     chi::RunContext &rctx) {
  CLIO_TASK_BODY_BEGIN
  HLOG(kInfo, "Admin: Executing SubmitBatch task with {} tasks",
       task->task_infos_.size());

  auto *ipc_manager = CLIO_IPC;
  auto *pool_manager = CLIO_POOL_MANAGER;

  // Initialize output values
  task->tasks_completed_ = 0;
  task->error_message_ = "";

  // If no tasks to submit
  if (task->task_infos_.empty()) {
    task->SetReturnCode(0);
    HLOG(kInfo, "SubmitBatch: No tasks to submit");
    CLIO_CO_RETURN;
  }

  // Create DefaultLoadArchive from the serialized data
  chi::priv::vector<char> load_buf(CLIO_PRIV_ALLOC);
  load_buf.reserve(task->serialized_data_.size());
  for (size_t i = 0; i < task->serialized_data_.size(); ++i) {
    load_buf.push_back(task->serialized_data_[i]);
  }
  chi::DefaultLoadArchive archive(load_buf);

  // Process tasks in batches of 32
  constexpr size_t kMaxParallelTasks = 32;
  std::vector<chi::Future<chi::Task>> pending_futures;
  pending_futures.reserve(kMaxParallelTasks);

  size_t task_idx = 0;
  size_t total_tasks = task->task_infos_.size();

  while (task_idx < total_tasks) {
    // Submit up to kMaxParallelTasks tasks
    pending_futures.clear();

    for (size_t i = 0; i < kMaxParallelTasks && task_idx < total_tasks;
         ++i, ++task_idx) {
      const chi::LocalTaskInfo &task_info = task->task_infos_[task_idx];

      // Get the container for this task's pool
      chi::Container *container =
          pool_manager->GetStaticContainer(task_info.pool_id_);
      if (!container) {
        HLOG(kError, "SubmitBatch: Container not found for pool_id {}",
             task_info.pool_id_);
        continue;
      }

      // Deserialize and allocate the task
      ctp::ipc::FullPtr<chi::Task> sub_task_ptr =
          container->LocalAllocLoadTask(task_info.method_id_, archive);

      if (sub_task_ptr.IsNull()) {
        HLOG(kError, "SubmitBatch: Failed to load task at index {}", task_idx);
        continue;
      }

      // Submit task and collect future
      chi::Future<chi::Task> future = ipc_manager->Send(sub_task_ptr);
      pending_futures.push_back(std::move(future));
    }

    // CLIO_CO_AWAIT all pending futures in this batch
    for (auto &future : pending_futures) {
      CLIO_CO_AWAIT(future);
      task->tasks_completed_++;
    }

    HLOG(kDebug, "SubmitBatch: Completed batch, total completed: {}",
         task->tasks_completed_);
  }

  task->SetReturnCode(0);
  HLOG(kInfo, "SubmitBatch: Completed {} of {} tasks", task->tasks_completed_,
       total_tasks);

  (void)rctx;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::RegisterMemory(ctp::ipc::FullPtr<RegisterMemoryTask> task,
                                        chi::RunContext &rctx) {
  CLIO_TASK_BODY_BEGIN
  auto *ipc_manager = CLIO_IPC;
  MemoryType mem_type = static_cast<MemoryType>(task->memory_type_);

  switch (mem_type) {
    case MemoryType::kCpuMemory: {
      // Existing path: POSIX shared memory registration
      ctp::ipc::AllocatorId alloc_id(task->alloc_major_, task->alloc_minor_);
      HLOG(kInfo, "Admin::RegisterMemory: Registering CPU alloc_id ({}.{})",
           alloc_id.major_, alloc_id.minor_);
      task->success_ = ipc_manager->RegisterMemory(alloc_id);
      break;
    }
    case MemoryType::kPinnedHostMemory:
    case MemoryType::kGpuDeviceMemory:
    case MemoryType::kManagedUvm: {
#if CTP_ENABLE_CUDA || CTP_ENABLE_ROCM || CTP_ENABLE_SYCL
      auto *gpu_ipc = ipc_manager->GetGpuIpcManager();
      if (!gpu_ipc) {
        HLOG(kError, "Admin::RegisterMemory: gpu_ipc_ not initialized");
        task->success_ = false;
        break;
      }
      chi::gpu::IpcManager::ClientBackend b;
      b.alloc_id = ctp::ipc::AllocatorId(task->alloc_major_, task->alloc_minor_);
      b.gpu_id = task->gpu_id_;
      b.capacity = task->data_capacity_;
      switch (mem_type) {
        case MemoryType::kPinnedHostMemory:
          b.kind = chi::gpu::IpcManager::MemKind::kPinnedHost;
          // The client passes the host pointer in ipc_handle_bytes_[0..7] for
          // pinned host (no IPC handle needed — same address space).
          memcpy(&b.host_view, task->ipc_handle_bytes_, sizeof(char *));
          b.device_ptr = b.host_view;  // pinned host is device-accessible
          break;
        case MemoryType::kManagedUvm:
          b.kind = chi::gpu::IpcManager::MemKind::kManagedUvm;
          memcpy(&b.host_view, task->ipc_handle_bytes_, sizeof(char *));
          b.device_ptr = b.host_view;
          break;
        case MemoryType::kGpuDeviceMemory:
          b.kind = chi::gpu::IpcManager::MemKind::kDeviceMem;
          // ipc_handle_bytes_ holds a cudaIpcMemHandle_t — opening it on the
          // runtime side is left as a follow-up; for now we record the
          // handle bytes verbatim and rely on the worker pop path to copy
          // POD bytes via cudaMemcpy through a runtime-side cudaIpcOpenMemHandle.
          b.host_view = nullptr;
          memcpy(&b.device_ptr, task->ipc_handle_bytes_, sizeof(char *));
          break;
        default: break;
      }
      HLOG(kInfo, "Admin::RegisterMemory: kind={} alloc_id=({}.{}) gpu_id={} "
           "capacity={}", static_cast<int>(b.kind), b.alloc_id.major_,
           b.alloc_id.minor_, b.gpu_id, b.capacity);
      task->success_ = gpu_ipc->RegisterClientBackend(b);
#else
      HLOG(kError, "Admin::RegisterMemory: GPU support not compiled in");
      task->success_ = false;
#endif
      break;
    }
    default:
      HLOG(kError, "Admin::RegisterMemory: Unknown memory type {}", task->memory_type_);
      task->success_ = false;
      break;
  }

  task->SetReturnCode(task->success_ ? 0 : 1);
  (void)rctx;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::RestartContainers(
    ctp::ipc::FullPtr<RestartContainersTask> task, chi::RunContext &rctx) {
  CLIO_TASK_BODY_BEGIN
  HLOG(kDebug, "Admin: Executing RestartContainers task");

  task->containers_restarted_ = 0;
  task->error_message_ = "";

  try {
    auto *config_manager = CLIO_CONFIG_MANAGER;
    std::string restart_dir = config_manager->GetConfDir() + "/restart";

    namespace fs = std::filesystem;
    if (!fs::exists(restart_dir) || !fs::is_directory(restart_dir)) {
      HLOG(kDebug, "Admin: No restart directory found at {}", restart_dir);
      task->SetReturnCode(0);
      CLIO_CO_RETURN;
    }

    for (const auto &entry : fs::directory_iterator(restart_dir)) {
      if (entry.path().extension() != ".yaml") continue;

      // Load pool config from YAML file
      chi::ConfigManager temp_config;
      if (!temp_config.LoadYaml(entry.path().string())) {
        HLOG(kError, "Admin: Failed to load restart config: {}",
             entry.path().string());
        continue;
      }

      const auto &compose_config = temp_config.GetComposeConfig();
      for (const auto &pool_config : compose_config.pools_) {
        HLOG(kInfo, "Admin: Restarting pool {} (module: {})",
             pool_config.pool_name_, pool_config.mod_name_);

        auto future = client_.AsyncCompose(pool_config);
        CLIO_CO_AWAIT(future);

        chi::u32 rc = future->GetReturnCode();
        if (rc != 0) {
          HLOG(kError, "Admin: Failed to restart pool {}: rc={}",
               pool_config.pool_name_, rc);
          continue;
        }

        task->containers_restarted_++;
        HLOG(kInfo, "Admin: Successfully restarted pool {}",
             pool_config.pool_name_);
      }
    }

    task->SetReturnCode(0);
    HLOG(kInfo, "Admin: RestartContainers completed, {} containers restarted",
         task->containers_restarted_);
  } catch (const std::exception &e) {
    task->return_code_ = 99;
    std::string error_msg =
        std::string("Exception during RestartContainers: ") + e.what();
    task->error_message_ = chi::priv::string(CTP_MALLOC, error_msg);
    HLOG(kError, "Admin: RestartContainers failed: {}", e.what());
  }
  (void)rctx;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::AddNode(ctp::ipc::FullPtr<AddNodeTask> task,
                                 chi::RunContext &rctx) {
  CLIO_TASK_BODY_BEGIN
  (void)rctx;
  HLOG(kInfo, "Admin: Executing AddNode for {}:{}", task->new_node_ip_.str(),
       task->new_node_port_);

  auto *ipc_manager = CLIO_IPC;
  auto *pool_manager = CLIO_POOL_MANAGER;

  // Add the new node to the IpcManager's hostfile
  chi::u64 new_node_id =
      ipc_manager->AddNode(task->new_node_ip_.str(), task->new_node_port_);
  task->new_node_id_ = new_node_id;

  // Notify all containers about the new node
  chi::Host new_host(task->new_node_ip_.str(), new_node_id);
  std::vector<chi::PoolId> pool_ids = pool_manager->GetAllPoolIds();
  for (const auto &pool_id : pool_ids) {
    bool is_plugged = false;
    chi::Container *container = pool_manager->GetContainer(
        pool_id, chi::kInvalidContainerId, is_plugged);
    if (container) {
      container->Expand(new_host);
    }
  }

  HLOG(kInfo, "Admin: AddNode complete, assigned node_id={}", new_node_id);
  task->SetReturnCode(0);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::WreapDeadIpcs(ctp::ipc::FullPtr<WreapDeadIpcsTask> task,
                                       chi::RunContext &rctx) {
  CLIO_TASK_BODY_BEGIN
  auto *ipc_manager = CLIO_IPC;

  // Call IpcManager::WreapDeadIpcs to reap shared memory from dead processes
  // task->reaped_count_ = ipc_manager->WreapDeadIpcs();
  task->reaped_count_ = 0;

  // Mark whether we did work (for periodic task efficiency tracking)
  if (task->reaped_count_ > 0) {
    rctx.did_work_ = true;
    HLOG(kInfo, "Admin: WreapDeadIpcs reaped {} shared memory segments",
         task->reaped_count_);
  } else {
    rctx.did_work_ = false;
  }

  task->SetReturnCode(0);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::ChangeAddressTable(
    ctp::ipc::FullPtr<ChangeAddressTableTask> task, chi::RunContext &rctx) {
  CLIO_TASK_BODY_BEGIN
  (void)rctx;
  auto *pool_manager = CLIO_POOL_MANAGER;

  chi::PoolId target_pool_id = task->target_pool_id_;
  chi::ContainerId container_id = task->container_id_;
  chi::u32 new_node_id = task->new_node_id_;

  // Get old node for WAL
  chi::u32 old_node_id =
      pool_manager->GetContainerNodeId(target_pool_id, container_id);

  // Write WAL entry before applying change
  pool_manager->WriteAddressTableWAL(target_pool_id, container_id, old_node_id,
                                     new_node_id);

  // Update the address table mapping
  if (pool_manager->UpdateContainerNodeMapping(target_pool_id, container_id,
                                               new_node_id)) {
    HLOG(kInfo, "Admin: ChangeAddressTable pool {} container {} -> node {}",
         target_pool_id, container_id, new_node_id);
    task->SetReturnCode(0);
  } else {
    task->error_message_ = chi::priv::string(
        CTP_MALLOC, "Failed to update container node mapping");
    task->SetReturnCode(1);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::MigrateContainers(
    ctp::ipc::FullPtr<MigrateContainersTask> task, chi::RunContext &rctx) {
  CLIO_TASK_BODY_BEGIN
  (void)rctx;
  HLOG(kInfo, "Admin: Executing MigrateContainers task");

  auto *pool_manager = CLIO_POOL_MANAGER;
  task->num_migrated_ = 0;
  task->error_message_ = "";

  // Deserialize migrations from binary
  std::string data = task->migrations_json_.str();
  std::vector<chi::MigrateInfo> migrations;
  {
    std::vector<char> buf(data.begin(), data.end());
    ctp::ipc::GlobalDeserialize<std::vector<char>> ar(buf);
    ar(migrations);
  }

  for (const auto &info : migrations) {
    // Look up source node
    chi::u32 src_node =
        pool_manager->GetContainerNodeId(info.pool_id_, info.container_id_);

    // Plug the container to stop new tasks and wait for work to complete
    pool_manager->PlugContainer(info.pool_id_, info.container_id_);

    // Get the specific Container on this node and call Migrate
    bool is_plugged = false;
    chi::Container *container = pool_manager->GetContainer(
        info.pool_id_, info.container_id_, is_plugged);
    if (container) {
      container->Migrate(info.dest_);
    }

    // Broadcast ChangeAddressTable to all nodes
    auto change_task = client_.AsyncChangeAddressTable(
        chi::PoolQuery::Broadcast(), info.pool_id_, info.container_id_,
        info.dest_);
    CLIO_CO_AWAIT(change_task);

    if (change_task->GetReturnCode() != 0) {
      HLOG(kError,
           "Admin: Failed to change address table for pool {} container {}",
           info.pool_id_, info.container_id_);
      continue;
    }

    // Unregister the container on source node so HasContainer() returns false.
    // This causes ResolveDirectHashQuery to fall through to address_map_
    // lookup. Note: UnregisterContainer preserves static_container_ for
    // deserialization.
    pool_manager->UnregisterContainer(info.pool_id_, info.container_id_);

    task->num_migrated_++;
    HLOG(kInfo, "Admin: Migrated pool {} container {} from node {} to node {}",
         info.pool_id_, info.container_id_, src_node, info.dest_);
  }

  task->SetReturnCode(0);
  HLOG(kInfo, "Admin: MigrateContainers completed, {} migrated",
       task->num_migrated_);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

/**
 * Attempt to send a retried task to the given node
 * @param entry The retry entry containing the task
 * @param node_id The node to send to
 * @return true if send succeeded
 */
bool Runtime::RetrySendToNode(RetryEntry &entry, chi::u64 node_id) {
  auto *ipc_manager = CLIO_IPC;
  auto *pool_manager = CLIO_POOL_MANAGER;
  auto *config_manager = CLIO_CONFIG_MANAGER;

  const chi::Host *target_host = ipc_manager->GetHost(node_id);
  if (!target_host) {
    return false;
  }
  int port = static_cast<int>(config_manager->GetPort());
  ctp::lbm::Transport *lbm_transport =
      ipc_manager->GetOrCreateClient(target_host->ip_address, port);
  if (!lbm_transport) {
    return false;
  }
  chi::Container *container =
      pool_manager->GetStaticContainer(entry.task->pool_id_);
  if (!container) {
    return false;
  }
  chi::SaveTaskArchive archive(chi::MsgType::kSerializeIn, lbm_transport);
  container->SaveTask(entry.task->method_, archive, entry.task);
  ctp::lbm::LbmContext ctx(0);
  int rc = lbm_transport->Send(archive, ctx);
  return rc == 0;
}

/**
 * Re-resolve target node for a retried task whose original target is dead.
 * Uses the task's pool_query_ to look up the current container→node mapping
 * from the address_map_, which may have been updated by recovery.
 * @param entry The retry entry to re-resolve
 * @return New node ID, or 0 if re-resolution failed
 */
chi::u64 Runtime::RerouteRetryEntry(RetryEntry &entry) {
  auto *pool_manager = CLIO_POOL_MANAGER;
  const chi::PoolQuery &query = entry.task->pool_query_;

  if (query.IsDirectIdMode()) {
    chi::ContainerId container_id = query.GetContainerId();
    chi::u32 new_node =
        pool_manager->GetContainerNodeId(entry.task->pool_id_, container_id);
    if (new_node != 0 && new_node != entry.target_node_id) {
      return new_node;
    }
  } else if (query.IsRangeMode()) {
    chi::u32 offset = query.GetRangeOffset();
    chi::ContainerId container_id(offset);
    chi::u32 new_node =
        pool_manager->GetContainerNodeId(entry.task->pool_id_, container_id);
    if (new_node != 0 && new_node != entry.target_node_id) {
      return new_node;
    }
  }
  return 0;
}

void Runtime::ProcessRetryQueues() {
  auto *ipc_manager = CLIO_IPC;
  auto now = std::chrono::steady_clock::now();

  // Sender threads push to send_*_retry_ on EAGAIN / failure.
  // ProcessRetryQueues runs on the SendPoll periodic thread; serialise
  // iteration against concurrent pushes. unique_lock so we can release
  // around the recursive SendOut() call (SendOut may want this lock).
  std::unique_lock<std::mutex> _rqlk(retry_queues_mutex_);

  // Process send_in retry queue
  auto it = send_in_retry_.begin();
  while (it != send_in_retry_.end()) {
    float elapsed = std::chrono::duration<float>(now - it->enqueued_at).count();
    float task_timeout = kRetryTimeoutSec;
    float task_net_timeout = it->task->pool_query_.GetNetTimeout();
    if (task_net_timeout >= 0) {
      task_timeout = task_net_timeout;
    }
    if (elapsed >= task_timeout) {
      // Timeout: mark task as failed
      HLOG(kError, "[RetryQueue] SendIn task timed out after {}s for node {}",
           elapsed, it->target_node_id);
      it->task->SetReturnCode(kNetworkTimeoutRC);
      it = send_in_retry_.erase(it);
    } else if (ipc_manager->IsAlive(it->target_node_id)) {
      // Original node came back: retry the send
      if (RetrySendToNode(*it, it->target_node_id)) {
        HLOG(kInfo, "[RetryQueue] SendIn retry succeeded for node {}",
             it->target_node_id);
        it = send_in_retry_.erase(it);
        continue;
      }
      // Retry failed, keep in queue
      ++it;
    } else {
      // Original node still dead — try re-routing via updated address_map
      chi::u64 new_node = RerouteRetryEntry(*it);
      if (new_node != 0 && ipc_manager->IsAlive(new_node)) {
        HLOG(kInfo,
             "[RetryQueue] Re-routing task from dead node {} to "
             "recovered node {}",
             it->target_node_id, new_node);
        it->target_node_id = new_node;
        if (RetrySendToNode(*it, new_node)) {
          HLOG(kInfo,
               "[RetryQueue] SendIn re-routed retry succeeded "
               "for node {}",
               new_node);
          it = send_in_retry_.erase(it);
          continue;
        }
      }
      ++it;
    }
  }

  // Process send_out retry queue
  it = send_out_retry_.begin();
  while (it != send_out_retry_.end()) {
    float elapsed = std::chrono::duration<float>(now - it->enqueued_at).count();
    float out_task_timeout = kRetryTimeoutSec;
    float out_task_net_timeout = it->task->pool_query_.GetNetTimeout();
    if (out_task_net_timeout >= 0) {
      out_task_timeout = out_task_net_timeout;
    }
    if (elapsed >= out_task_timeout) {
      HLOG(kError, "[RetryQueue] SendOut task timed out after {}s for node {}",
           elapsed, it->target_node_id);
      // For send_out, the result is lost; origin will timeout
      it = send_out_retry_.erase(it);
    } else if (ipc_manager->IsAlive(it->target_node_id)) {
      // Node came back: retry by calling SendOut. SendOut may push to
      // send_out_retry_ on a fresh failure — release the lock around it
      // to avoid self-deadlock, then restart iteration from begin().
      ctp::ipc::FullPtr<chi::Task> retry_task = it->task;
      it = send_out_retry_.erase(it);
      _rqlk.unlock();
      SendOut(retry_task);
      _rqlk.lock();
      it = send_out_retry_.begin();
      continue;
      ++it;
    }
  }
}

void Runtime::ScanSendMapTimeouts() {
  auto *ipc_manager = CLIO_IPC;
  auto now = std::chrono::steady_clock::now();

  // Iterate dead nodes and check if any send_map_ entries target them
  const auto &dead_nodes = ipc_manager->GetDeadNodes();
  if (dead_nodes.empty()) return;

  // Build map of dead node IDs to their detected_at times for per-task checks
  std::unordered_map<chi::u64, std::chrono::steady_clock::time_point> dead_map;
  for (const auto &entry : dead_nodes) {
    dead_map[entry.node_id] = entry.detected_at;
  }

  // Scan send_map_ for tasks targeting dead nodes using for_each.
  //
  // ScanSendMapTimeouts runs on net_send_worker (same thread as SendIn
  // inserts), but RecvOut on net_recv_worker erases entries concurrently.
  // Hold the lock for the iteration, but EndTask is called only on
  // collected keys and may do nontrivial work, so the actual EndTask +
  // erase pass runs outside the iteration scope. This keeps the critical
  // section bounded to a hash-bucket walk.
  std::vector<std::pair<size_t, ctp::ipc::FullPtr<chi::Task>>> to_complete;
  {
    std::lock_guard<std::mutex> lk(send_map_mutex_);
    send_map_.for_each(
        [&](const size_t &key, ctp::ipc::FullPtr<chi::Task> &origin_task) {
          if (origin_task.IsNull() || !origin_task->GetRunCtx()) return;

          chi::RunContext *rctx = origin_task->GetRunCtx();
          // Use per-task timeout if set, otherwise kRetryTimeoutSec
          float task_timeout = kRetryTimeoutSec;
          float task_net_timeout = origin_task->pool_query_.GetNetTimeout();
          if (task_net_timeout >= 0) {
            task_timeout = task_net_timeout;
          }

          // Check if any replica targets a dead node that has exceeded the
          // timeout
          bool any_timed_out = false;
          for (const auto &pq : rctx->pool_queries_) {
            if (pq.IsPhysicalMode()) {
              auto dit = dead_map.find(pq.GetNodeId());
              if (dit != dead_map.end()) {
                float dead_elapsed =
                    std::chrono::duration<float>(now - dit->second).count();
                if (dead_elapsed >= task_timeout) {
                  any_timed_out = true;
                  break;
                }
              }
            }
          }

          if (any_timed_out) {
            to_complete.emplace_back(key, origin_task);
          }
        });
  }

  // Finalise timed-out tasks outside the map lock to keep EndTask from
  // executing under it (EndTask can re-enter scheduling code).
  for (auto &entry : to_complete) {
    auto &origin_task = entry.second;
    chi::RunContext *rctx = origin_task->GetRunCtx();
    if (!rctx) continue;
    HLOG(kError,
         "[ScanSendMapTimeouts] Task {} timed out waiting for dead node",
         origin_task->task_id_);
    origin_task->SetReturnCode(kNetworkTimeoutRC);
    auto *worker = CLIO_CUR_WORKER;
    worker->EndTask(origin_task, rctx, true);
  }

  if (!to_complete.empty()) {
    std::lock_guard<std::mutex> lk(send_map_mutex_);
    for (auto &entry : to_complete) {
      send_map_.erase(entry.first);
    }
  }
}

void Runtime::FlushStaleStateForNode(chi::u64 node_id) {
  std::lock_guard<std::mutex> _rqlk(retry_queues_mutex_);
  // 1. Discard send_in retry entries targeting this node.
  //    For each discarded entry, increment the origin task's
  //    completed_replicas so broadcast origins can still complete.
  for (auto it = send_in_retry_.begin(); it != send_in_retry_.end();) {
    if (it->target_node_id == node_id) {
      size_t net_key = it->task->task_id_.net_key_;
      ctp::ipc::FullPtr<chi::Task> origin;
      {
        std::lock_guard<std::mutex> lk(send_map_mutex_);
        auto send_it = send_map_.find(net_key);
        if (send_it != nullptr) {
          origin = *send_it;
        }
      }
      if (!origin.IsNull() && origin->GetRunCtx()) {
        origin->GetRunCtx()->completed_replicas_++;
      }
      HLOG(kInfo,
           "[FlushStale] Discarding SendIn retry for restarted node {}",
           node_id);
      it = send_in_retry_.erase(it);
    } else {
      ++it;
    }
  }

  // 2. Discard send_out retry entries targeting this node.
  //    These are responses destined for the old incarnation; drop them.
  for (auto it = send_out_retry_.begin(); it != send_out_retry_.end();) {
    if (it->target_node_id == node_id) {
      HLOG(kInfo,
           "[FlushStale] Discarding SendOut retry for restarted node {}",
           node_id);
      it = send_out_retry_.erase(it);
    } else {
      ++it;
    }
  }
}

chi::TaskResume Runtime::Heartbeat(ctp::ipc::FullPtr<HeartbeatTask> task,
                                   chi::RunContext &rctx) {
  CLIO_TASK_BODY_BEGIN
  task->SetReturnCode(0);
  rctx.did_work_ = true;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::HeartbeatProbe(ctp::ipc::FullPtr<HeartbeatProbeTask> task,
                                        chi::RunContext &rctx) {
  CLIO_TASK_BODY_BEGIN
  auto *ipc_manager = CLIO_IPC;
  auto *config_manager = CLIO_CONFIG_MANAGER;

  // Master kill-switch: when disabled, the failure-detector is a no-op.
  // Used for bring-up / perf debugging where we want to take SWIM out
  // of the picture and trust that nodes don't go away mid-run. Every
  // task that consults SWIM state (SendIn IsAlive check, SendOut return-
  // node check, recovery) still works — `kAlive` is the default node
  // state, so without SWIM mutating it, IsAlive() trivially returns true
  // and the cross-node paths just keep retrying on transient errors.
  if (!config_manager->GetSwimEnabled()) {
    rctx.did_work_ = false;
    task->SetReturnCode(0);
    CLIO_CO_RETURN;
  }

  // Pull SWIM timeouts from config each tick so the values are
  // hot-reloadable in principle and live close to the call site.
  const float kDirectProbeTimeoutSec_cfg =
      config_manager->GetSwimDirectProbeTimeoutSec();
  const float kIndirectProbeTimeoutSec_cfg =
      config_manager->GetSwimIndirectProbeTimeoutSec();
  const float kSuspicionTimeoutSec_cfg =
      config_manager->GetSwimSuspicionTimeoutSec();

  auto now = std::chrono::steady_clock::now();
  chi::u64 self_node_id = ipc_manager->GetNodeId();
  bool did_work = false;

  // 1. Check pending direct probes
  for (auto it = pending_direct_probes_.begin();
       it != pending_direct_probes_.end();) {
    if (it->future.IsComplete()) {
      // Direct probe succeeded - node is alive
      ipc_manager->SetNodeState(it->target_node_id, chi::NodeState::kAlive);
      it = pending_direct_probes_.erase(it);
      did_work = true;
    } else {
      float elapsed = std::chrono::duration<float>(now - it->sent_at).count();
      if (elapsed > kDirectProbeTimeoutSec_cfg) {
        // Direct probe timed out - escalate to indirect probing
        ipc_manager->SetNodeState(it->target_node_id,
                                  chi::NodeState::kProbeFailed);
        HLOG(
            kWarning,
            "SWIM: Direct probe to node {} timed out, starting indirect probes",
            it->target_node_id);

        // Select k random alive helpers (excluding self and target)
        const auto &hosts = ipc_manager->GetAllHosts();
        std::vector<chi::u64> candidates;
        for (const auto &h : hosts) {
          if (h.node_id != self_node_id && h.node_id != it->target_node_id &&
              h.IsAlive()) {
            candidates.push_back(h.node_id);
          }
        }
        std::shuffle(candidates.begin(), candidates.end(), probe_rng_);
        size_t num_helpers = std::min(kIndirectProbeHelpers, candidates.size());
        for (size_t i = 0; i < num_helpers; ++i) {
          auto future = client_.AsyncProbeRequest(
              chi::PoolQuery::Physical(candidates[i]), it->target_node_id);
          pending_indirect_probes_.push_back(
              {std::move(future), it->target_node_id, candidates[i],
               std::chrono::steady_clock::now()});
        }

        it = pending_direct_probes_.erase(it);
        did_work = true;
      } else {
        ++it;
      }
    }
  }

  // 2. Check pending indirect probes
  for (auto it = pending_indirect_probes_.begin();
       it != pending_indirect_probes_.end();) {
    if (it->future.IsComplete()) {
      it->future.Wait();   // Finalize (already complete — IsComplete() above)
      if (it->future->probe_result_ == 0) {
        // Indirect probe succeeded - node is alive
        ipc_manager->SetNodeState(it->target_node_id, chi::NodeState::kAlive);
        HLOG(kInfo, "SWIM: Indirect probe via node {} confirmed node {} alive",
             it->helper_node_id, it->target_node_id);
        // Remove all pending indirects for this target
        chi::u64 alive_target = it->target_node_id;
        pending_indirect_probes_.erase(
            std::remove_if(pending_indirect_probes_.begin(),
                           pending_indirect_probes_.end(),
                           [alive_target](const PendingIndirectProbe &p) {
                             return p.target_node_id == alive_target;
                           }),
            pending_indirect_probes_.end());
        did_work = true;
        break;  // Iterator invalidated, restart on next invocation
      } else {
        chi::u64 target = it->target_node_id;
        it = pending_indirect_probes_.erase(it);
        did_work = true;
        // If no more pending indirects for this target, move to suspected
        bool has_more = false;
        for (const auto &p : pending_indirect_probes_) {
          if (p.target_node_id == target) {
            has_more = true;
            break;
          }
        }
        if (!has_more &&
            ipc_manager->GetNodeState(target) == chi::NodeState::kProbeFailed) {
          ipc_manager->SetNodeState(target, chi::NodeState::kSuspected);
          HLOG(
              kWarning,
              "SWIM: All indirect probes for node {} failed, marking suspected",
              target);
        }
      }
    } else {
      float elapsed = std::chrono::duration<float>(now - it->sent_at).count();
      if (elapsed > kIndirectProbeTimeoutSec_cfg) {
        chi::u64 target = it->target_node_id;
        it = pending_indirect_probes_.erase(it);
        did_work = true;
        // If no more pending indirects for this target, move to suspected
        bool has_more = false;
        for (const auto &p : pending_indirect_probes_) {
          if (p.target_node_id == target) {
            has_more = true;
            break;
          }
        }
        if (!has_more &&
            ipc_manager->GetNodeState(target) == chi::NodeState::kProbeFailed) {
          ipc_manager->SetNodeState(target, chi::NodeState::kSuspected);
          HLOG(
              kWarning,
              "SWIM: All indirect probes for node {} failed, marking suspected",
              target);
        }
      } else {
        ++it;
      }
    }
  }

  // 3. Check suspicion timeouts
  {
    const auto &hosts = ipc_manager->GetAllHosts();
    for (const auto &h : hosts) {
      if (h.state == chi::NodeState::kSuspected) {
        float since_change =
            std::chrono::duration<float>(now - h.state_changed_at).count();
        if (since_change >= kSuspicionTimeoutSec_cfg) {
          HLOG(kError, "SWIM: Node {} confirmed dead after suspicion timeout",
               h.node_id);
          ipc_manager->SetDead(h.node_id);
          did_work = true;
          CLIO_CO_AWAIT(TriggerRecovery(h.node_id));
        }
      }
    }
  }

  // 4. Self-fencing: if majority of other nodes are suspected/dead, fence self
  {
    const auto &hosts = ipc_manager->GetAllHosts();
    size_t other_count = 0;
    size_t bad_count = 0;
    for (const auto &h : hosts) {
      if (h.node_id == self_node_id) continue;
      other_count++;
      if (h.state == chi::NodeState::kSuspected ||
          h.state == chi::NodeState::kDead) {
        bad_count++;
      }
    }
    if (other_count > 0 && bad_count * 2 > other_count) {
      if (!ipc_manager->IsSelfFenced()) {
        // HLOG(kFatal, ...) calls exit(1) (see logging.h:270), which would
        // tear down the daemon before SetSelfFenced has a chance to run and
        // before SWIM has any chance to recover. The intent here is "log
        // loudly and set the flag," not "abort." Use kError so the message
        // still stands out in stderr but the daemon stays up and keeps
        // probing — the kInfo branch below clears the fence once peers come
        // back. Killing the daemon on suspicion was the actual root cause
        // of the 4n/8n IOR hangs: under bulk-IO bursts a probe round-trip
        // can miss its window on >=2 peers, the kFatal exit-on-log fires,
        // the node goes away, and now its peers see >50% bad and self-fence
        // too -> full cluster collapse.
        HLOG(kError,
             "SWIM: Self-fencing (continuing to probe)! {} of {} other "
             "nodes are suspected/dead",
             bad_count, other_count);
        ipc_manager->SetSelfFenced(true);
      }
    } else {
      if (ipc_manager->IsSelfFenced()) {
        HLOG(kInfo, "SWIM: Clearing self-fence, cluster connectivity restored");
        ipc_manager->SetSelfFenced(false);
      }
    }
  }

  // 5. Send new direct probe (round-robin, one per invocation)
  {
    const auto &hosts = ipc_manager->GetAllHosts();
    if (hosts.size() > 1) {
      size_t start_idx = probe_round_robin_idx_;
      for (size_t i = 0; i < hosts.size(); ++i) {
        size_t idx = (start_idx + i) % hosts.size();
        const auto &h = hosts[idx];
        if (h.node_id == self_node_id) continue;
        if (h.state == chi::NodeState::kDead) continue;
        // Skip suspected nodes — let the suspicion timeout fire
        // before re-probing, otherwise the state cycles
        // kSuspected→kProbeFailed→kSuspected and resets the timer
        if (h.state == chi::NodeState::kSuspected) continue;
        if (h.state == chi::NodeState::kProbeFailed) continue;
        // Skip if already probing this node
        bool already_probing = false;
        for (const auto &p : pending_direct_probes_) {
          if (p.target_node_id == h.node_id) {
            already_probing = true;
            break;
          }
        }
        if (already_probing) continue;

        // Send direct probe
        auto future =
            client_.AsyncHeartbeat(chi::PoolQuery::Physical(h.node_id));
        pending_direct_probes_.push_back(
            {std::move(future), h.node_id, std::chrono::steady_clock::now()});
        probe_round_robin_idx_ = (idx + 1) % hosts.size();
        did_work = true;
        break;  // One probe per invocation
      }
    }
  }

  rctx.did_work_ = did_work;
  task->SetReturnCode(0);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::ProbeRequest(ctp::ipc::FullPtr<ProbeRequestTask> task,
                                      chi::RunContext &rctx) {
  CLIO_TASK_BODY_BEGIN
  // Probe the target node on behalf of the requester using cooperative yield
  auto future =
      client_.AsyncHeartbeat(chi::PoolQuery::Physical(task->target_node_id_));
  auto start = std::chrono::steady_clock::now();

  const float indirect_probe_timeout_sec =
      CLIO_CONFIG_MANAGER->GetSwimIndirectProbeTimeoutSec();
  while (!future.IsComplete()) {
    float elapsed =
        std::chrono::duration<float>(std::chrono::steady_clock::now() - start)
            .count();
    if (elapsed >= indirect_probe_timeout_sec) break;
    CLIO_CO_AWAIT(chi::yield(1000.0));
  }

  if (future.IsComplete()) {
    future.Wait();            // Finalize (already complete)
    task->probe_result_ = 0;  // alive
  } else {
    task->probe_result_ = -1;  // unreachable
  }

  task->SetReturnCode(0);
  rctx.did_work_ = true;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskStat Runtime::GetTaskStats(const chi::Task *task) const {
  if (!task) return chi::TaskStat();
  switch (task->method_) {
    case Method::kSend:
    case Method::kRecv:
    case Method::kClientRecv:
    case Method::kClientSend: {
      // Admin net periodics don't carry a variable payload of their own;
      // io_size_ here is just a stand-in so the scheduler doesn't route
      // them as zero-cost metadata. compute_ reflects the current
      // net_queue backlog so periodic schedulers can throttle if needed.
      chi::TaskStat stat;
      auto *net_queue = CLIO_IPC->GetNetQueue();
      size_t total = 0;
      if (net_queue) {
        for (chi::u32 p = 0; p < 4; ++p)
          total += net_queue->GetLane(0, p).Size();
      }
      stat.compute_ = total;
      stat.io_size_ = 1024 * 1024;
      return stat;
    }
    default: return chi::TaskStat();
  }
}

chi::u64 Runtime::GetWorkRemaining() const {
  // Accessed from non-net workers (the scheduler/shutdown path); briefly
  // acquire both map locks for a consistent snapshot.
  size_t send_size, recv_size;
  {
    std::lock_guard<std::mutex> lk(send_map_mutex_);
    send_size = send_map_.size();
  }
  {
    std::lock_guard<std::mutex> lk(recv_map_mutex_);
    recv_size = recv_map_.size();
  }
  return send_size + recv_size;
}

//===========================================================================
// Recovery Methods
//===========================================================================

std::vector<chi::RecoveryAssignment> Runtime::ComputeRecoveryPlan(
    chi::u64 dead_node_id) {
  auto *pool_manager = CLIO_POOL_MANAGER;
  auto *ipc_manager = CLIO_IPC;

  // Collect alive nodes for round-robin assignment
  std::vector<chi::u64> alive_nodes;
  for (const auto &h : ipc_manager->GetAllHosts()) {
    if (h.IsAlive()) alive_nodes.push_back(h.node_id);
  }
  if (alive_nodes.empty()) return {};

  std::vector<chi::RecoveryAssignment> assignments;
  size_t rr_idx = 0;

  for (const auto &pool_id : pool_manager->GetAllPoolIds()) {
    const chi::PoolInfo *info = pool_manager->GetPoolInfo(pool_id);
    if (!info) continue;
    for (const auto &[container_id, node_id] : info->address_map_) {
      if (node_id == static_cast<chi::u32>(dead_node_id)) {
        chi::RecoveryAssignment ra;
        ra.pool_id_ = pool_id;
        ra.chimod_name_ = info->chimod_name_;
        ra.pool_name_ = info->pool_name_;
        ra.chimod_params_ = info->chimod_params_;
        ra.container_id_ = container_id;
        ra.dead_node_id_ = static_cast<chi::u32>(dead_node_id);
        chi::u32 dest = static_cast<chi::u32>(-1);
        if (info->local_container_) {
          dest = info->local_container_->ScheduleRecover();
        }
        if (dest == static_cast<chi::u32>(-1)) {
          dest =
              static_cast<chi::u32>(alive_nodes[rr_idx % alive_nodes.size()]);
          rr_idx++;
        }
        ra.dest_node_id_ = dest;
        assignments.push_back(std::move(ra));
      }
    }
  }
  return assignments;
}

chi::TaskResume Runtime::TriggerRecovery(chi::u64 dead_node_id) {
#ifdef __NVCOMPILER
  chi::RunContext _dummy_rctx;
  chi::RunContext& rctx = _dummy_rctx;
#endif
  CLIO_TASK_BODY_BEGIN
  auto *ipc_manager = CLIO_IPC;
  if (!ipc_manager->IsLeader()) CLIO_CO_RETURN;
  if (recovery_initiated_.count(dead_node_id)) CLIO_CO_RETURN;
  recovery_initiated_.insert(dead_node_id);
  if (ipc_manager->IsSelfFenced()) {
    HLOG(kWarning, "Recovery: Skipping for node {} - self-fenced",
         dead_node_id);
    CLIO_CO_RETURN;
  }

  HLOG(kInfo, "Recovery: Leader initiating for dead node {}", dead_node_id);
  auto assignments = ComputeRecoveryPlan(dead_node_id);
  if (assignments.empty()) {
    HLOG(kInfo, "Recovery: No containers to recover from node {}",
         dead_node_id);
    CLIO_CO_RETURN;
  }

  HLOG(kInfo, "Recovery: {} containers to redistribute from node {}",
       assignments.size(), dead_node_id);
  CLIO_CO_AWAIT(client_.AsyncRecoverContainers(chi::PoolQuery::Broadcast(0),
                                          assignments, dead_node_id));
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::RecoverContainers(
    ctp::ipc::FullPtr<RecoverContainersTask> task, chi::RunContext &rctx) {
  CLIO_TASK_BODY_BEGIN
  auto *ipc_manager = CLIO_IPC;
  auto *pool_manager = CLIO_POOL_MANAGER;
  auto *module_manager = CLIO_MODULE_MANAGER;
  chi::u64 self_node_id = ipc_manager->GetNodeId();
  task->num_recovered_ = 0;

  // Deserialize assignments
  std::vector<chi::RecoveryAssignment> assignments;
  {
    std::string data = task->assignments_data_.str();
    std::vector<char> buf(data.begin(), data.end());
    ctp::ipc::GlobalDeserialize<std::vector<char>> ar(buf);
    ar(assignments);
  }

  for (const auto &ra : assignments) {
    // ALL nodes update address_map_
    pool_manager->UpdateContainerNodeMapping(ra.pool_id_, ra.container_id_,
                                             ra.dest_node_id_);
    pool_manager->WriteAddressTableWAL(ra.pool_id_, ra.container_id_,
                                       ra.dead_node_id_, ra.dest_node_id_);

    // Only dest node creates the container
    if (static_cast<chi::u64>(ra.dest_node_id_) != self_node_id) continue;

    HLOG(kInfo, "Recovery: Creating container {} for pool {} ({})",
         ra.container_id_, ra.pool_name_, ra.chimod_name_);
    chi::Container *container = module_manager->CreateContainer(
        ra.chimod_name_, ra.pool_id_, ra.pool_name_);
    if (!container) {
      HLOG(kError, "Recovery: Failed to create container for {}",
           ra.chimod_name_);
      continue;
    }
    container->Recover(ra.pool_id_, ra.pool_name_, ra.container_id_);
    pool_manager->RegisterContainer(ra.pool_id_, ra.container_id_, container,
                                    false);
    task->num_recovered_++;
  }

  task->SetReturnCode(0);
  rctx.did_work_ = true;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

//===========================================================================
// System Monitor
//===========================================================================

chi::TaskResume Runtime::SystemMonitor(ctp::ipc::FullPtr<SystemMonitorTask> task,
                                       chi::RunContext &rctx) {
  CLIO_TASK_BODY_BEGIN
  SystemStats stats;

  // Timestamps
  auto mono_now = std::chrono::steady_clock::now();
  auto wall_now = std::chrono::system_clock::now();
  stats.timestamp_ns_ = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          mono_now.time_since_epoch())
          .count());
  stats.wall_time_ns_ = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          wall_now.time_since_epoch())
          .count());

  // DRAM
  stats.ram_total_bytes_ = CTP_SYSTEM_INFO->ram_size_;
  stats.ram_available_bytes_ = ctp::SystemInfo::GetRamAvailable();
  if (stats.ram_total_bytes_ > 0) {
    stats.ram_usage_pct_ =
        (1.0f - static_cast<float>(stats.ram_available_bytes_) /
                    static_cast<float>(stats.ram_total_bytes_)) *
        100.0f;
  }

  // CPU
  ctp::CpuTimes cur = ctp::SystemInfo::GetCpuTimes();
  stats.cpu_usage_pct_ =
      ctp::SystemInfo::ComputeCpuUtilization(prev_cpu_times_, cur);
  prev_cpu_times_ = cur;

  // GPU/HBM — stub (zeroed by default constructor)

  // Push into ring buffer
  if (system_stats_ring_) {
    system_stats_ring_->Push(stats);
  }

  rctx.did_work_ = true;
  (void)task;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

chi::TaskResume Runtime::RegisterGpuContainer(
    ctp::ipc::FullPtr<RegisterGpuContainerTask> task, chi::RunContext &rctx) {
  // This task is handled on the CPU side.
  // The GPU orchestrator's gpu::PoolManager is updated via a GPU kernel launch,
  // not directly from the admin runtime. The pool_manager.cc CreatePool
  // handles the actual GPU container creation and registration.
  // This method exists as a no-op placeholder for task routing completeness.
  HLOG(kDebug, "RegisterGpuContainer: pool_id={}, container_id={}",
       task->target_pool_id_, task->container_id_);
  rctx.did_work_ = true;
  co_return;
}

//===========================================================================
// Task Serialization Method Implementations
//===========================================================================

// Task Serialization Method Implementations now in autogen/admin_lib_exec.cc

}  // namespace clio::run::admin

// Define ChiMod entry points using CLIO_TASK_CC macro
CLIO_TASK_CC(clio::run::admin::Runtime)