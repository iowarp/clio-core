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

// Method implementations for Runtime class

// Virtual method implementations (Init, Run, Del, SaveTask, LoadTask, NewCopy,
// Aggregate) now in autogen/admin_lib_exec.cc

//===========================================================================
// Method implementations
//===========================================================================

Runtime::~Runtime() {}

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

  // Start dedicated recv threads in IpcManagerRun2Run.
  CLIO_IPC->GetRun2Run()->StartRecvThreads();
  // Stop recv threads before the main transport is freed.
  CLIO_IPC->RegisterTransportShutdownHook([this]() {
    CLIO_IPC->GetRun2Run()->StopRecvThreads();
  });

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
  CLIO_IPC->GetRun2Run()->ProcessRetryQueues();
  CLIO_IPC->GetRun2Run()->ScanSendMapTimeouts();

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
      ipc_manager->GetRun2Run()->SendIn(origin_task);
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
      ipc_manager->GetRun2Run()->SendOut(origin_task);
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
        ipc_manager->GetRun2Run()->SendIn(origin_task);
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
        ipc_manager->GetRun2Run()->SendOut(origin_task);
        did_send = true;
        io_budget = (sz >= io_budget) ? 0 : (io_budget - sz);
        --io_out_remaining;
        did_any = true;
      }
    }
    if (!did_any) break;
  }

  rctx.did_work_ = did_send;
  task->SetReturnCode(0);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
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
      ipc_manager->GetRun2Run()->RecvIn(archive, lbm_transport);
      task->SetReturnCode(0);
      break;
    case chi::MsgType::kSerializeOut:
      HLOG(kDebug, "[Recv] Dispatching to RecvOut");
      ipc_manager->GetRun2Run()->RecvOut(archive, lbm_transport);
      task->SetReturnCode(0);
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
  auto *run2run = CLIO_IPC->GetRun2Run();
  return run2run->GetSendMapSize() + run2run->GetRecvMapSize();
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