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

#include "chimaera/admin/admin_runtime.h"

#include <chimaera/chimaera_manager.h>
#include <chimaera/module_manager.h>
#include <chimaera/pool_manager.h>
#include <chimaera/task_archives.h>
#include <chimaera/worker.h>
#include <hermes_shm/lightbeam/transport_factory_impl.h>

#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>
#include <chrono>
#include <memory>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <filesystem>
#include <vector>

namespace chimaera::admin {

// Method implementations for Runtime class

// Virtual method implementations (Init, Run, Del, SaveTask, LoadTask, NewCopy,
// Aggregate) now in autogen/admin_lib_exec.cc

//===========================================================================
// Method implementations
//===========================================================================

chi::TaskResume Runtime::Create(hipc::FullPtr<CreateTask> task,
                                chi::RunContext &rctx) {
  // Admin container creation logic (IS_ADMIN=true)
  HLOG(kDebug, "Admin: Initializing admin container");

  // Initialize the Admin container with pool information from the task
  // Note: Admin container is already initialized by the framework before Create
  // is called

  // Note: No locks needed - all Send/Recv tasks are routed to a single
  // dedicated network worker, ensuring thread-safe access to
  // send_map_/recv_map_

  create_count_++;

  // Spawn periodic Recv task with 25 microsecond period (default)
  // Worker will automatically reschedule periodic tasks
  client_.AsyncRecv(chi::PoolQuery::Local(), 0, 500);

  // Spawn periodic Send task with 25 microsecond period
  // This task polls net_queue_ for send operations
  client_.AsyncSendPoll(chi::PoolQuery::Local(), 0, 500);

  // Spawn periodic ClientRecv task for client task reception via lightbeam
  client_.AsyncClientRecv(chi::PoolQuery::Local(), 100);

  // Spawn periodic ClientSend task for client response sending via lightbeam
  client_.AsyncClientSend(chi::PoolQuery::Local(), 100);

  // Register client server FDs with worker's EventManager
  {
    auto *worker = CHI_CUR_WORKER;
    auto *ipc_manager = CHI_IPC;
    if (worker && ipc_manager) {
      auto &em = worker->GetEventManager();
      auto *tcp_transport = ipc_manager->GetClientTransport(chi::IpcMode::kTcp);
      if (tcp_transport) {
        tcp_transport->RegisterEventManager(em);
        HLOG(kDebug, "Admin: TCP transport registered with worker EventManager");
      }
      auto *ipc_transport = ipc_manager->GetClientTransport(chi::IpcMode::kIpc);
      if (ipc_transport) {
        ipc_transport->RegisterEventManager(em);
        HLOG(kDebug, "Admin: IPC transport registered with worker EventManager");
      }
    }
  }

  // Spawn periodic WreapDeadIpcs task with 1 second period
  // This task reaps shared memory segments from dead processes
  client_.AsyncWreapDeadIpcs(chi::PoolQuery::Local(), 1000000);

  // Spawn periodic HeartbeatProbe task (SWIM failure detector, 2s period)
  client_.AsyncHeartbeatProbe(chi::PoolQuery::Local(), 2000000);

  HLOG(kDebug,
       "Admin: Container created and initialized for pool: {} (ID: {}, count: "
       "{})",
       pool_name_, task->new_pool_id_, create_count_);
  HLOG(kDebug, "Admin: Spawned periodic Recv, Send, ClientConnect, ClientRecv, ClientSend tasks");
  (void)rctx;
  co_return;
}

chi::TaskResume Runtime::GetOrCreatePool(
    hipc::FullPtr<
        chimaera::admin::GetOrCreatePoolTask<chimaera::admin::CreateParams>>
        task,
    chi::RunContext &rctx) {
  // Debug: Log do_compose_ value
  HLOG(kDebug,
       "Admin::GetOrCreatePool ENTRY: task->do_compose_={}, task->is_admin_={}",
       task->do_compose_, task->is_admin_);

  // Get pool manager once - used by both dynamic scheduling and normal
  // execution
  auto *pool_manager = CHI_POOL_MANAGER;

  // Extract pool name once
  std::string pool_name = task->pool_name_.str();

  // Check if this is dynamic scheduling mode
  if (rctx.exec_mode_ == chi::ExecMode::kDynamicSchedule) {
    // Dynamic routing with cache optimization
    // Check if pool exists locally first to avoid unnecessary broadcast
    HLOG(kDebug,
         "Admin: Dynamic routing for GetOrCreatePool - checking local cache");

    chi::PoolId existing_pool_id = pool_manager->FindPoolByName(pool_name);

    if (!existing_pool_id.IsNull()) {
      // Pool exists locally - change pool query to Local
      HLOG(kDebug, "Admin: Pool '{}' found locally (ID: {}), using Local query",
           pool_name, existing_pool_id);
      task->pool_query_ = chi::PoolQuery::Local();
    } else {
      // Pool doesn't exist locally - update pool query to Broadcast for
      // creation
      HLOG(kDebug, "Admin: Pool '{}' not found locally, broadcasting creation",
           pool_name);
      task->pool_query_ = chi::PoolQuery::Broadcast();
    }
    co_return;
  }

  // Pool get-or-create operation logic (IS_ADMIN=false)
  HLOG(kDebug, "Admin: Executing GetOrCreatePool task - ChiMod: {}, Pool: {}",
       task->chimod_name_.str(), pool_name);

  // Initialize output values
  task->return_code_ = 0;
  task->error_message_ = "";

  try {
    // Use the simplified PoolManager API that extracts all parameters from the
    // task. CreatePool is now a coroutine that co_awaits nested Create methods.
    co_await pool_manager->CreatePool(task.Cast<chi::Task>(), &rctx);

    // Check if CreatePool set an error (return code is set on the task)
    if (task->return_code_ != 0) {
      // Error already set by CreatePool
      co_return;
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
    task->error_message_ = chi::priv::string(HSHM_MALLOC, error_msg);
    HLOG(kError, "Admin: Pool creation failed with exception: {}", e.what());
  }
  co_return;
}

chi::TaskResume Runtime::Destroy(hipc::FullPtr<DestroyTask> task,
                                 chi::RunContext &rctx) {
  // DestroyTask is aliased to DestroyPoolTask, so delegate to DestroyPool
  co_await DestroyPool(task, rctx);
  co_return;
}

chi::TaskResume Runtime::DestroyPool(hipc::FullPtr<DestroyPoolTask> task,
                                     chi::RunContext &rctx) {
  HLOG(kDebug, "Admin: Executing DestroyPool task - Pool ID: {}",
       task->target_pool_id_);

  // Initialize output values
  task->return_code_ = 0;
  task->error_message_ = "";

  try {
    chi::PoolId target_pool = task->target_pool_id_;

    // Get pool manager to handle pool destruction
    auto *pool_manager = CHI_POOL_MANAGER;
    if (!pool_manager || !pool_manager->IsInitialized()) {
      task->return_code_ = 1;
      task->error_message_ = "Pool manager not available";
      co_return;
    }

    // Use PoolManager to destroy the complete pool including metadata
    // DestroyPool is now a coroutine for consistency
    co_await pool_manager->DestroyPool(target_pool);

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
    task->error_message_ = chi::priv::string(HSHM_MALLOC, error_msg);
    HLOG(kError, "Admin: Pool destruction failed with exception: {}", e.what());
  }
  co_return;
}

chi::TaskResume Runtime::StopRuntime(hipc::FullPtr<StopRuntimeTask> task,
                                     chi::RunContext &rctx) {
  HLOG(kDebug, "Admin: Executing StopRuntime task - Grace period: {}ms",
       task->grace_period_ms_);

  // Initialize output values
  task->return_code_ = 0;
  task->error_message_ = "";

  try {
    // Set shutdown flag
    is_shutdown_requested_ = true;

    // Initiate graceful shutdown
    InitiateShutdown(task->grace_period_ms_);

    // Set success results
    task->return_code_ = 0;

    HLOG(kDebug, "Admin: Runtime shutdown initiated successfully");

  } catch (const std::exception &e) {
    task->return_code_ = 99;
    std::string error_msg =
        std::string("Exception during runtime shutdown: ") + e.what();
    task->error_message_ = chi::priv::string(HSHM_MALLOC, error_msg);
    HLOG(kError, "Admin: Runtime shutdown failed with exception: {}", e.what());
  }
  (void)rctx;
  co_return;
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

  // Get Chimaera manager to initiate shutdown
  auto *chimaera_manager = CHI_CHIMAERA_MANAGER;
  if (chimaera_manager) {
    // chimaera_manager->InitiateShutdown(grace_period_ms);
  }
  std::abort();
}

chi::TaskResume Runtime::Flush(hipc::FullPtr<FlushTask> task,
                               chi::RunContext &rctx) {
  HLOG(kDebug, "Admin: Executing Flush task");

  // Initialize output values
  task->return_code_ = 0;
  task->total_work_done_ = 0;

  try {
    // Get WorkOrchestrator to check work remaining across all containers
    auto *work_orchestrator = CHI_WORK_ORCHESTRATOR;
    if (!work_orchestrator || !work_orchestrator->IsInitialized()) {
      task->return_code_ = 1;
      co_return;
    }

    // Loop until all work is complete
    chi::u64 total_work_remaining = 0;
    while (work_orchestrator->HasWorkRemaining(total_work_remaining)) {
      HLOG(kDebug,
           "Admin: Flush found {} work units still remaining, waiting...",
           total_work_remaining);

      // Brief yield to avoid busy waiting
      co_await chi::yield(25);
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
  co_return;
}

//===========================================================================
// Distributed Task Scheduling Method Implementations
//===========================================================================

/**
 * Helper function: Send task inputs to remote node
 * @param origin_task Task to send to remote nodes
 * @param rctx RunContext for managing subtasks
 */
void Runtime::SendIn(hipc::FullPtr<chi::Task> origin_task,
                     chi::RunContext &rctx) {
  auto *ipc_manager = CHI_IPC;
  auto *pool_manager = CHI_POOL_MANAGER;

  // Validate origin_task
  if (origin_task.IsNull()) {
    HLOG(kError, "SendIn: origin_task is null");
    return;
  }

  // Get the container associated with the origin_task
  chi::Container *container = pool_manager->GetStaticContainer(origin_task->pool_id_);
  if (container == nullptr) {
    HLOG(kError, "SendIn: container not found for pool_id {}",
         origin_task->pool_id_);
    return;
  }

  // Pre-allocate send_map_key using origin_task pointer
  // This ensures consistent net_key across all replicas
  size_t send_map_key = size_t(origin_task.ptr_);

  // Add the origin task to send_map before creating copies
  // Note: No lock needed - single net worker processes all Send/Recv tasks
  send_map_[send_map_key] = origin_task;

  // Get pool_queries from task's RunContext
  if (!origin_task->run_ctx_) {
    HLOG(kError, "SendIn: origin_task has no RunContext");
    return;
  }
  chi::RunContext *origin_task_rctx = origin_task->run_ctx_.get();

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
    hipc::FullPtr<chi::Task> task_copy =
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
      HLOG(kWarning, "[SendIn] Task {} target node {} is dead, queuing for retry",
           origin_task->task_id_, target_node_id);
      send_in_retry_.push_back({task_copy, target_node_id,
                                std::chrono::steady_clock::now()});
      continue;
    }

    // Get or create persistent Lightbeam client using connection pool
    auto *config_manager = CHI_CONFIG_MANAGER;
    int port = static_cast<int>(config_manager->GetPort());
    hshm::lbm::Transport *lbm_transport =
        ipc_manager->GetOrCreateClient(target_host->ip_address, port);

    if (!lbm_transport) {
      HLOG(kError, "[SendIn] Task {} FAILED: Could not get client for {}:{}",
           origin_task->task_id_, target_host->ip_address, port);
      ipc_manager->SetDead(target_node_id);
      send_in_retry_.push_back({task_copy, target_node_id,
                                std::chrono::steady_clock::now()});
      continue;
    }

    // Create SaveTaskArchive with SerializeIn mode and lbm_transport
    chi::SaveTaskArchive archive(chi::MsgType::kSerializeIn, lbm_transport);

    // Serialize the task using container->SaveTask (Expose will be called
    // automatically for bulks)
    container->SaveTask(task_copy->method_, archive, task_copy);

    // Send using Lightbeam asynchronously (non-blocking)
    // Note: No lock needed - single net worker processes all Send/Recv tasks
    hshm::lbm::LbmContext ctx(0);  // Non-blocking async send
    int rc = lbm_transport->Send(archive, ctx);

    if (rc != 0) {
      HLOG(kError,
           "[SendIn] Task {} Lightbeam async Send FAILED with error code {}",
           origin_task->task_id_, rc);
      ipc_manager->SetDead(target_node_id);
      send_in_retry_.push_back({task_copy, target_node_id,
                                std::chrono::steady_clock::now()});
      continue;
    }
  }
}

/**
 * Helper function: Send task outputs back to origin node
 * @param origin_task Completed task whose outputs need to be sent back
 */
void Runtime::SendOut(hipc::FullPtr<chi::Task> origin_task) {
  auto *ipc_manager = CHI_IPC;
  auto *pool_manager = CHI_POOL_MANAGER;

  // Flush deferred deletes from previous invocation (zero-copy send safety)
  static std::vector<hipc::FullPtr<chi::Task>> deferred_deletes;
  for (auto &t : deferred_deletes) {
    ipc_manager->DelTask(t);
  }
  deferred_deletes.clear();

  // Validate origin_task
  if (origin_task.IsNull()) {
    HLOG(kError, "SendOut: origin_task is null");
    return;
  }

  // Get the container associated with the origin_task
  chi::Container *container = pool_manager->GetStaticContainer(origin_task->pool_id_);
  if (container == nullptr) {
    HLOG(kError, "SendOut: container not found for pool_id {}",
         origin_task->pool_id_);
    return;
  }

  // Remove task from recv_map as we're completing it
  // Key must match RecvIn: combines net_key and replica_id
  // Note: No lock needed - single net worker processes all Send/Recv tasks
  size_t recv_key = origin_task->task_id_.net_key_ ^
      (static_cast<size_t>(origin_task->task_id_.replica_id_) * 0x9e3779b97f4a7c15ULL);
  auto *it = recv_map_.find(recv_key);
  if (it == nullptr) {
    HLOG(kError,
         "[SendOut] Task {} FAILED: Not found in recv_map (size: {}) with "
         "recv_key {}",
         origin_task->task_id_, recv_map_.size(), recv_key);
    return;
  }
  recv_map_.erase(recv_key);

  // Get return node from pool_query
  chi::u64 target_node_id = origin_task->pool_query_.GetReturnNode();

  // Check aliveness before sending output back
  if (!ipc_manager->IsAlive(target_node_id)) {
    HLOG(kWarning, "[SendOut] Task {} return node {} is dead, queuing for retry",
         origin_task->task_id_, target_node_id);
    send_out_retry_.push_back({origin_task, target_node_id,
                               std::chrono::steady_clock::now()});
    return;
  }

  // Get host information
  const chi::Host *target_host = ipc_manager->GetHost(target_node_id);
  if (target_host == nullptr) {
    HLOG(kError, "[SendOut] Task {} FAILED: Host not found for node_id {}",
         origin_task->task_id_, target_node_id);
    return;
  }

  // Get or create persistent Lightbeam client using connection pool
  auto *config_manager = CHI_CONFIG_MANAGER;
  int port = static_cast<int>(config_manager->GetPort());
  hshm::lbm::Transport *lbm_transport =
      ipc_manager->GetOrCreateClient(target_host->ip_address, port);

  if (lbm_transport == nullptr) {
    HLOG(kError, "[SendOut] Task {} FAILED: Could not get client for {}:{}",
         origin_task->task_id_, target_host->ip_address, port);
    ipc_manager->SetDead(target_node_id);
    send_out_retry_.push_back({origin_task, target_node_id,
                               std::chrono::steady_clock::now()});
    return;
  }

  // Create SaveTaskArchive with SerializeOut mode and lbm_transport
  // The client will automatically call Expose internally during serialization
  chi::SaveTaskArchive archive(chi::MsgType::kSerializeOut, lbm_transport);

  // Serialize the task outputs using container->SaveTask (Expose called
  // automatically)
  container->SaveTask(origin_task->method_, archive, origin_task);

  // Use non-timed, non-sync context for SendOut
  // Note: No lock needed - single net worker processes all Send/Recv tasks
  hshm::lbm::LbmContext ctx(0);
  int rc = lbm_transport->Send(archive, ctx);
  if (rc != 0) {
    HLOG(kError, "[SendOut] Task {} Lightbeam Send FAILED with error code {}",
         origin_task->task_id_, rc);
    ipc_manager->SetDead(target_node_id);
    send_out_retry_.push_back({origin_task, target_node_id,
                               std::chrono::steady_clock::now()});
    return;
  }

  HLOG(kDebug, "[SendOut] Task {}", origin_task->task_id_);

  // Defer task deletion to next invocation for zero-copy send safety
  deferred_deletes.push_back(origin_task);
}

/**
 * Main Send function - periodic task that polls net_queue_ for send operations
 * Polls both SendIn (priority 0) and SendOut (priority 1) queues
 */
chi::TaskResume Runtime::Send(hipc::FullPtr<SendTask> task,
                              chi::RunContext &rctx) {
  auto *ipc_manager = CHI_IPC;
  chi::Future<chi::Task> queued_future;
  bool did_send = false;
  int send_in_count = 0;

  // Process retry queues before normal sends
  ProcessRetryQueues();

  // Scan send_map_ for timed-out entries from dead nodes
  ScanSendMapTimeouts();

  // Poll priority 0 (SendIn) queue - tasks waiting to be sent to remote nodes
  while (ipc_manager->TryPopNetTask(chi::NetQueuePriority::kSendIn,
                                    queued_future)) {
    // Get the original task from the Future
    auto origin_task = queued_future.GetTaskPtr();
    if (!origin_task.IsNull()) {
      HLOG(kDebug, "[Send] Processing SendIn task method={}, pool_id={}",
           origin_task->method_, origin_task->pool_id_);
      SendIn(origin_task, rctx);
      did_send = true;
      send_in_count++;
    }
  }

  if (send_in_count > 0) {
    HLOG(kDebug, "[Send] Processed {} SendIn tasks", send_in_count);
  }

  // Poll priority 1 (SendOut) queue - tasks with outputs to send back
  int send_out_count = 0;
  while (ipc_manager->TryPopNetTask(chi::NetQueuePriority::kSendOut,
                                    queued_future)) {
    // Get the original task from the Future
    auto origin_task = queued_future.GetTaskPtr();
    if (!origin_task.IsNull()) {
      HLOG(kDebug, "[Send] Processing SendOut task method={}, pool_id={}",
           origin_task->method_, origin_task->pool_id_);
      SendOut(origin_task);
      did_send = true;
      send_out_count++;
    }
  }

  if (send_out_count > 0) {
    HLOG(kDebug, "[Send] Processed {} SendOut tasks", send_out_count);
  }

  // Track whether this execution did actual work
  rctx.did_work_ = did_send;

  task->SetReturnCode(0);
  co_return;
}

/**
 * Helper function: Receive task inputs from remote node
 * @param task RecvTask containing control information
 * @param archive Already-parsed LoadTaskArchive containing task info
 * @param lbm_transport Lightbeam server for receiving bulk data
 */
void Runtime::RecvIn(hipc::FullPtr<RecvTask> task,
                     chi::LoadTaskArchive &archive,
                     hshm::lbm::Transport *lbm_transport) {
  // Set I/O size to 1MB to ensure routing to slow workers
  task->stat_.io_size_ = 1024 * 1024;  // 1MB

  auto *ipc_manager = CHI_IPC;
  auto *pool_manager = CHI_POOL_MANAGER;

  const auto &task_infos = archive.GetTaskInfos();

  // If no tasks to receive
  if (task_infos.empty()) {
    task->SetReturnCode(0);
    return;
  }

  for (size_t task_idx = 0; task_idx < task_infos.size(); ++task_idx) {
    const auto &task_info = task_infos[task_idx];

    // Get container associated with PoolId
    chi::Container *container = pool_manager->GetStaticContainer(task_info.pool_id_);
    if (!container) {
      HLOG(kError, "Admin: Container not found for pool_id {}",
           task_info.pool_id_);
      continue;
    }

    // Call AllocLoadTask to allocate and deserialize the task
    hipc::FullPtr<chi::Task> task_ptr =
        container->AllocLoadTask(task_info.method_id_, archive);

    if (task_ptr.IsNull()) {
      HLOG(kError, "Admin: Failed to load task");
      continue;
    }

    // Mark task as remote, set as data owner, clear sender-side flags
    // TASK_RUN_CTX_EXISTS and TASK_STARTED must be cleared so the receiving
    // worker allocates a fresh RunContext via BeginTask
    task_ptr->SetFlags(TASK_REMOTE | TASK_DATA_OWNER);
    task_ptr->ClearFlags(TASK_PERIODIC | TASK_FORCE_NET | TASK_ROUTED |
                         TASK_RUN_CTX_EXISTS | TASK_STARTED);

    // Add task to recv_map for later lookup
    // Key combines net_key and replica_id so multiple replicas targeting the
    // same node (e.g., after container migration) get distinct entries.
    // Note: No lock needed - single net worker processes all Send/Recv tasks
    size_t recv_key = task_ptr->task_id_.net_key_ ^
        (static_cast<size_t>(task_ptr->task_id_.replica_id_) * 0x9e3779b97f4a7c15ULL);
    recv_map_[recv_key] = task_ptr;

    HLOG(kDebug, "[RecvIn] Task {}", task_ptr->task_id_);

    // Send task for execution using IpcManager::Send with awake_event=false
    // Note: This creates a Future and enqueues it to worker lanes
    // awake_event=false prevents setting parent task for received remote tasks
    // Note: IsClientThread is false since this is runtime code
    (void)ipc_manager->Send(task_ptr, false);
  }

  task->SetReturnCode(0);
}

/**
 * Helper function: Receive task outputs from remote node
 * @param task RecvTask containing control information
 * @param archive Already-parsed LoadTaskArchive containing task info
 * @param lbm_transport Lightbeam server for receiving bulk data
 */
void Runtime::RecvOut(hipc::FullPtr<RecvTask> task,
                      chi::LoadTaskArchive &archive,
                      hshm::lbm::Transport *lbm_transport) {
  // Set I/O size to 1MB to ensure routing to slow workers
  task->stat_.io_size_ = 1024 * 1024;  // 1MB

  auto *pool_manager = CHI_POOL_MANAGER;

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

    // Note: No lock needed - single net worker processes all Send/Recv tasks
    auto send_it = send_map_.find(net_key);
    if (send_it == nullptr) {
      HLOG(kError,
           "[RecvOut] Task {} FAILED: Origin task not found in send_map "
           "(size: {}) with net_key {}",
           task_info.task_id_, send_map_.size(), net_key);
      task->SetReturnCode(5);
      return;
    }
    hipc::FullPtr<chi::Task> origin_task = *send_it;
    if (!origin_task->run_ctx_) {
      HLOG(kError, "Admin: origin_task has no RunContext");
      task->SetReturnCode(6);
      return;
    }
    chi::RunContext *origin_rctx = origin_task->run_ctx_.get();

    // Locate replica in origin's run_ctx using replica_id
    chi::u32 replica_id = task_info.task_id_.replica_id_;
    if (replica_id >= origin_rctx->subtasks_.size()) {
      HLOG(kError, "Admin: Invalid replica_id {} (subtasks size: {})",
           replica_id, origin_rctx->subtasks_.size());
      task->SetReturnCode(7);
      return;
    }

    hipc::FullPtr<chi::Task> replica = origin_rctx->subtasks_[replica_id];

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
    // This exposes buffers via ar.bulk() and populates archive.recv
    container->LoadTask(origin_task->method_, archive, replica);
  }

  // Second pass: Aggregate results
  for (size_t task_idx = 0; task_idx < task_infos.size(); ++task_idx) {
    const auto &task_info = task_infos[task_idx];

    // Locate origin task from send_map using net_key
    // Note: No lock needed - single net worker processes all Send/Recv tasks
    size_t net_key = task_info.task_id_.net_key_;
    auto send_it = send_map_.find(net_key);
    if (send_it == nullptr) {
      HLOG(kError, "Admin: Origin task not found in send_map with net_key {}",
           net_key);
      continue;
    }
    hipc::FullPtr<chi::Task> origin_task = *send_it;
    if (!origin_task->run_ctx_) {
      HLOG(kError, "Admin: origin_task has no RunContext");
      continue;
    }
    chi::RunContext *origin_rctx = origin_task->run_ctx_.get();

    // Locate replica in origin's run_ctx using replica_id
    chi::u32 replica_id = task_info.task_id_.replica_id_;
    if (replica_id >= origin_rctx->subtasks_.size()) {
      HLOG(kError, "Admin: Invalid replica_id {} (subtasks size: {})",
           replica_id, origin_rctx->subtasks_.size());
      continue;
    }

    hipc::FullPtr<chi::Task> replica = origin_rctx->subtasks_[replica_id];

    // Get the container associated with the origin task
    chi::Container *container =
        pool_manager->GetStaticContainer(origin_task->pool_id_);
    if (!container) {
      HLOG(kError, "Admin: Container not found for pool_id {}",
           origin_task->pool_id_);
      continue;
    }

    // Aggregate replica results into origin task
    container->Aggregate(origin_task->method_, origin_task, replica);

    HLOG(kDebug, "[RecvOut] Task {}", origin_task->task_id_);

    // Increment completed replicas counter in origin's rctx
    origin_rctx->completed_replicas_++;
    chi::u32 completed = origin_rctx->completed_replicas_;

    // If all replicas completed
    if (completed == origin_rctx->subtasks_.size()) {
      // Get pool manager to access container
      auto *pool_manager = CHI_POOL_MANAGER;
      chi::Container *container =
          pool_manager->GetStaticContainer(origin_task->pool_id_);

      // Unmark TASK_DATA_OWNER before deleting replicas to avoid freeing the
      // same data pointers twice Delete all origin_task replicas using
      // container->DelTask() to avoid memory leak
      if (container) {
        for (const auto &origin_task_ptr : origin_rctx->subtasks_) {
          origin_task_ptr->ClearFlags(TASK_DATA_OWNER);
          container->DelTask(origin_task_ptr->method_, origin_task_ptr);
        }
      }

      // Clear subtasks vector after deleting tasks
      origin_rctx->subtasks_.clear();

      // Remove origin from send_map
      // Note: No lock needed - single net worker processes all Send/Recv tasks
      send_map_.erase(net_key);

      // Add task back to blocked queue for both periodic and non-periodic tasks
      // ExecTask will handle checking if the task is complete and ending it
      // properly
      auto *worker = CHI_CUR_WORKER;
      worker->EndTask(origin_task, origin_rctx, true);
    }
  }

  task->SetReturnCode(0);
}

/**
 * Main Recv function - receives metadata and dispatches based on mode
 * Note: This is a periodic task - only logs when actual work is done
 */
chi::TaskResume Runtime::Recv(hipc::FullPtr<RecvTask> task,
                              chi::RunContext &rctx) {
  // Get the main server from CHI_IPC (already bound during initialization)
  auto *ipc_manager = CHI_IPC;

  hshm::lbm::Transport *lbm_transport = ipc_manager->GetMainTransport();
  if (lbm_transport == nullptr) {
    co_return;
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
    co_return;
  }

  if (rc != 0) {
    if (rc != -1) {
      HLOG(kError, "Admin: Lightbeam Recv failed with error code {}",
           rc);
    }
    task->SetReturnCode(2);
    rctx.did_work_ = false;
    co_return;
  }

  // Mark that we received data (did work)
  rctx.did_work_ = true;

  // Dispatch based on message type
  chi::MsgType msg_type = archive.GetMsgType();
  switch (msg_type) {
    case chi::MsgType::kSerializeIn:
      RecvIn(task, archive, lbm_transport);
      break;
    case chi::MsgType::kSerializeOut:
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

  co_return;
}

/**
 * Handle ClientConnect - Respond to client connection request
 * Polls connect server for ZMQ REQ/REP requests and responds
 * @param task The connect task
 * @param rctx Run context
 */
chi::TaskResume Runtime::ClientConnect(hipc::FullPtr<ClientConnectTask> task,
                                       chi::RunContext &rctx) {
  task->response_ = 0;
  task->server_generation_ = CHI_IPC->GetServerGeneration();
  task->SetReturnCode(0);
  rctx.did_work_ = true;
  co_return;
}

/**
 * Handle ClientRecv - Receive tasks from lightbeam client servers
 * Polls TCP and IPC PULL servers for incoming client task submissions
 */
chi::TaskResume Runtime::ClientRecv(hipc::FullPtr<ClientRecvTask> task,
                                    chi::RunContext &rctx) {
  auto *ipc_manager = CHI_IPC;
  auto *pool_manager = CHI_POOL_MANAGER;
  bool did_work = false;
  task->tasks_received_ = 0;

  // Process both TCP and IPC servers
  for (int mode_idx = 0; mode_idx < 2; ++mode_idx) {
    chi::IpcMode mode = (mode_idx == 0) ? chi::IpcMode::kTcp
                                         : chi::IpcMode::kIpc;
    hshm::lbm::Transport *transport = ipc_manager->GetClientTransport(mode);
    if (!transport) continue;

    // Drain all pending messages from this transport
    // (Recv handles accept internally for socket transports)
    while (true) {
      chi::LoadTaskArchive archive;
      auto recv_info = transport->Recv(archive);
      int rc = recv_info.rc;
      if (rc == EAGAIN) break;
      if (rc != 0) {
        HLOG(kError, "ClientRecv: Recv failed: {}", rc);
        break;
      }

      const auto &task_infos = archive.GetTaskInfos();
      if (task_infos.empty()) {
        HLOG(kError, "ClientRecv: No task_infos in received message");
        continue;
      }

      const auto &info = task_infos[0];
      chi::PoolId pool_id = info.pool_id_;
      chi::u32 method_id = info.method_id_;

      // Get container for deserialization
      chi::Container *container = pool_manager->GetStaticContainer(pool_id);
      if (!container) {
        HLOG(kError, "ClientRecv: Container not found for pool_id {}", pool_id);
        continue;
      }

      // Allocate and deserialize the task
      hipc::FullPtr<chi::Task> task_ptr =
          container->AllocLoadTask(method_id, archive);

      if (task_ptr.IsNull()) {
        HLOG(kError, "ClientRecv: Failed to deserialize task");
        continue;
      }

      // Create FutureShm for the task (server-side)
      hipc::FullPtr<chi::FutureShm> future_shm =
          ipc_manager->NewObj<chi::FutureShm>();
      future_shm->pool_id_ = pool_id;
      future_shm->method_id_ = method_id;
      future_shm->origin_ = (mode == chi::IpcMode::kTcp)
                                 ? chi::FutureShm::FUTURE_CLIENT_TCP
                                 : chi::FutureShm::FUTURE_CLIENT_IPC;
      future_shm->client_task_vaddr_ = info.task_id_.net_key_;
      future_shm->client_pid_ = info.task_id_.pid_;
      // Store transport and routing info for response
      future_shm->response_transport_ = transport;
      future_shm->response_fd_ = recv_info.fd_;
      // Store ZMQ identity from recv frame for response routing
      if (!recv_info.identity_.empty() &&
          recv_info.identity_.size() <= sizeof(future_shm->response_identity_)) {
        std::memcpy(future_shm->response_identity_,
                    recv_info.identity_.data(),
                    recv_info.identity_.size());
        future_shm->response_identity_len_ =
            static_cast<chi::u32>(recv_info.identity_.size());
      }
      // No copy_space for ZMQ path — ShmTransferInfo defaults are fine
      // Mark as copied so the worker routes the completed task back via lightbeam
      // rather than treating it as a runtime-internal task
      future_shm->flags_.SetBits(chi::FutureShm::FUTURE_WAS_COPIED);

      // Create Future and enqueue to worker
      chi::Future<chi::Task> future(future_shm.shm_, task_ptr);

      // Map task to lane using scheduler
      chi::LaneId lane_id =
          ipc_manager->GetScheduler()->ClientMapTask(ipc_manager, future);
      auto *worker_queues = ipc_manager->GetTaskQueue();
      auto &lane_ref = worker_queues->GetLane(lane_id, 0);
      bool was_empty = lane_ref.Empty();
      lane_ref.Push(future);
      if (was_empty) {
        ipc_manager->AwakenWorker(&lane_ref);
      }

      did_work = true;
      task->tasks_received_++;
    }
  }

  rctx.did_work_ = did_work;
  task->SetReturnCode(0);
  co_return;
}

/**
 * Handle ClientSend - Send completed task outputs to clients via lightbeam
 * Polls net_queue_ kClientSendTcp and kClientSendIpc priorities
 */
chi::TaskResume Runtime::ClientSend(hipc::FullPtr<ClientSendTask> task,
                                    chi::RunContext &rctx) {
  auto *ipc_manager = CHI_IPC;
  auto *pool_manager = CHI_POOL_MANAGER;
  bool did_work = false;
  task->tasks_sent_ = 0;

  // Flush deferred deletes from previous invocation.
  // Zero-copy send (zmq_msg_init_data) lets ZMQ's IO thread read from the
  // task buffer after zmq_msg_send returns. Deferring DelTask by one
  // invocation guarantees the IO thread has flushed the message.
  static std::vector<hipc::FullPtr<chi::Task>> deferred_deletes;
  for (auto &t : deferred_deletes) {
    ipc_manager->DelTask(t);
  }
  deferred_deletes.clear();

  // Process both TCP and IPC queues
  for (int mode_idx = 0; mode_idx < 2; ++mode_idx) {
    chi::NetQueuePriority priority =
        (mode_idx == 0) ? chi::NetQueuePriority::kClientSendTcp
                        : chi::NetQueuePriority::kClientSendIpc;
    chi::IpcMode mode =
        (mode_idx == 0) ? chi::IpcMode::kTcp : chi::IpcMode::kIpc;

    chi::Future<chi::Task> queued_future;
    while (ipc_manager->TryPopNetTask(priority, queued_future)) {
      auto origin_task = queued_future.GetTaskPtr();
      if (origin_task.IsNull()) continue;

      // Get the FutureShm to find client's net_key
      auto future_shm = queued_future.GetFutureShm();
      if (future_shm.IsNull()) continue;

      // Get container to serialize outputs
      chi::Container *container =
          pool_manager->GetStaticContainer(origin_task->pool_id_);
      if (!container) {
        HLOG(kError, "ClientSend: Container not found for pool_id {}",
             origin_task->pool_id_);
        continue;
      }

      // Get response transport and routing info from FutureShm
      hshm::lbm::Transport *response_transport =
          future_shm->response_transport_;
      if (!response_transport) {
        HLOG(kError, "ClientSend: No response transport for mode {} pid {}",
             mode_idx, future_shm->client_pid_);
        continue;
      }

      // Preserve client's net_key for response routing
      origin_task->task_id_.net_key_ = future_shm->client_task_vaddr_;

      // Serialize task outputs using network archive
      chi::SaveTaskArchive archive(chi::MsgType::kSerializeOut, response_transport);
      container->SaveTask(origin_task->method_, archive, origin_task);

      // Set routing info for the response
      if (mode == chi::IpcMode::kTcp) {
        // TCP (ZMQ ROUTER): identity-based routing
        // Use the actual ZMQ identity from the recv frame
        if (future_shm->response_identity_len_ > 0) {
          archive.client_info_.identity_ = std::string(
              future_shm->response_identity_,
              future_shm->response_identity_len_);
        } else {
          // Fallback: construct from PID (legacy 4-byte identity)
          chi::u32 client_pid = future_shm->client_pid_;
          archive.client_info_.identity_ = std::string(
              reinterpret_cast<const char *>(&client_pid),
              sizeof(client_pid));
        }
      } else if (mode == chi::IpcMode::kIpc) {
        // IPC (Socket): fd-based routing on accepted connection
        archive.client_info_.fd_ = future_shm->response_fd_;
      }

      // Send via lightbeam
      int rc = response_transport->Send(archive, hshm::lbm::LbmContext());
      if (rc != 0) {
        HLOG(kError, "ClientSend: lightbeam Send failed: {}", rc);
      }

      // Defer task deletion to next invocation for zero-copy send safety
      deferred_deletes.push_back(origin_task);

      did_work = true;
      task->tasks_sent_++;
    }
  }

  rctx.did_work_ = did_work;
  task->SetReturnCode(0);
  co_return;
}

chi::TaskResume Runtime::Monitor(hipc::FullPtr<MonitorTask> task,
                                 chi::RunContext &rctx) {
  // Get work orchestrator to access all workers
  auto *work_orchestrator = CHI_WORK_ORCHESTRATOR;
  if (!work_orchestrator) {
    task->SetReturnCode(1);
    HLOG(kError, "Monitor: WorkOrchestrator not available");
    (void)rctx;
    co_return;
  }

  // Get worker count from the work orchestrator
  size_t num_workers = work_orchestrator->GetWorkerCount();

  // Reserve space in the vector
  task->info_.reserve(num_workers);

  // Collect stats from all workers
  for (size_t i = 0; i < num_workers; ++i) {
    chi::Worker *worker =
        work_orchestrator->GetWorker(static_cast<chi::u32>(i));
    if (!worker) {
      continue;
    }

    // Get stats from this worker and add to vector
    chi::WorkerStats stats = worker->GetWorkerStats();
    task->info_.push_back(stats);
  }

  task->SetReturnCode(0);
  (void)rctx;
  co_return;
}

chi::TaskResume Runtime::SubmitBatch(hipc::FullPtr<SubmitBatchTask> task,
                                     chi::RunContext &rctx) {
  HLOG(kInfo, "Admin: Executing SubmitBatch task with {} tasks",
       task->task_infos_.size());

  auto *ipc_manager = CHI_IPC;
  auto *pool_manager = CHI_POOL_MANAGER;

  // Initialize output values
  task->tasks_completed_ = 0;
  task->error_message_ = "";

  // If no tasks to submit
  if (task->task_infos_.empty()) {
    task->SetReturnCode(0);
    HLOG(kInfo, "SubmitBatch: No tasks to submit");
    co_return;
  }

  // Create LocalLoadTaskArchive from the serialized data
  chi::LocalLoadTaskArchive archive(task->serialized_data_);

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
      hipc::FullPtr<chi::Task> sub_task_ptr =
          container->LocalAllocLoadTask(task_info.method_id_, archive);

      if (sub_task_ptr.IsNull()) {
        HLOG(kError, "SubmitBatch: Failed to load task at index {}", task_idx);
        continue;
      }

      // Submit task and collect future
      chi::Future<chi::Task> future = ipc_manager->Send(sub_task_ptr);
      pending_futures.push_back(std::move(future));
    }

    // co_await all pending futures in this batch
    for (auto &future : pending_futures) {
      co_await future;
      task->tasks_completed_++;
    }

    HLOG(kDebug, "SubmitBatch: Completed batch, total completed: {}",
         task->tasks_completed_);
  }

  task->SetReturnCode(0);
  HLOG(kInfo, "SubmitBatch: Completed {} of {} tasks", task->tasks_completed_,
       total_tasks);

  (void)rctx;
  co_return;
}

chi::TaskResume Runtime::RegisterMemory(hipc::FullPtr<RegisterMemoryTask> task,
                                        chi::RunContext &rctx) {
  auto *ipc_manager = CHI_IPC;
  hipc::AllocatorId alloc_id(task->alloc_major_, task->alloc_minor_);

  HLOG(kInfo, "Admin::RegisterMemory: Registering alloc_id ({}.{})",
       alloc_id.major_, alloc_id.minor_);

  task->success_ = ipc_manager->RegisterMemory(alloc_id);
  task->SetReturnCode(task->success_ ? 0 : 1);

  (void)rctx;
  co_return;
}

chi::TaskResume Runtime::RestartContainers(
    hipc::FullPtr<RestartContainersTask> task, chi::RunContext &rctx) {
  HLOG(kDebug, "Admin: Executing RestartContainers task");

  task->containers_restarted_ = 0;
  task->error_message_ = "";

  try {
    auto *config_manager = CHI_CONFIG_MANAGER;
    std::string restart_dir = config_manager->GetConfDir() + "/restart";

    namespace fs = std::filesystem;
    if (!fs::exists(restart_dir) || !fs::is_directory(restart_dir)) {
      HLOG(kDebug, "Admin: No restart directory found at {}", restart_dir);
      task->SetReturnCode(0);
      co_return;
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
        co_await future;

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
    task->error_message_ = chi::priv::string(HSHM_MALLOC, error_msg);
    HLOG(kError, "Admin: RestartContainers failed: {}", e.what());
  }
  (void)rctx;
  co_return;
}

chi::TaskResume Runtime::AddNode(
    hipc::FullPtr<AddNodeTask> task, chi::RunContext &rctx) {
  (void)rctx;
  HLOG(kInfo, "Admin: Executing AddNode for {}:{}",
       task->new_node_ip_.str(), task->new_node_port_);

  auto *ipc_manager = CHI_IPC;
  auto *pool_manager = CHI_POOL_MANAGER;

  // Add the new node to the IpcManager's hostfile
  chi::u64 new_node_id = ipc_manager->AddNode(
      task->new_node_ip_.str(), task->new_node_port_);
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
  co_return;
}

chi::TaskResume Runtime::WreapDeadIpcs(hipc::FullPtr<WreapDeadIpcsTask> task,
                                       chi::RunContext &rctx) {
  auto *ipc_manager = CHI_IPC;

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
  co_return;
}

chi::TaskResume Runtime::ChangeAddressTable(
    hipc::FullPtr<ChangeAddressTableTask> task, chi::RunContext &rctx) {
  (void)rctx;
  auto *pool_manager = CHI_POOL_MANAGER;

  chi::PoolId target_pool_id = task->target_pool_id_;
  chi::ContainerId container_id = task->container_id_;
  chi::u32 new_node_id = task->new_node_id_;

  // Get old node for WAL
  chi::u32 old_node_id =
      pool_manager->GetContainerNodeId(target_pool_id, container_id);

  // Write WAL entry before applying change
  pool_manager->WriteAddressTableWAL(target_pool_id, container_id,
                                      old_node_id, new_node_id);

  // Update the address table mapping
  if (pool_manager->UpdateContainerNodeMapping(target_pool_id, container_id,
                                                new_node_id)) {
    HLOG(kInfo,
         "Admin: ChangeAddressTable pool {} container {} -> node {}",
         target_pool_id, container_id, new_node_id);
    task->SetReturnCode(0);
  } else {
    task->error_message_ = chi::priv::string(
        HSHM_MALLOC, "Failed to update container node mapping");
    task->SetReturnCode(1);
  }
  co_return;
}

chi::TaskResume Runtime::MigrateContainers(
    hipc::FullPtr<MigrateContainersTask> task, chi::RunContext &rctx) {
  (void)rctx;
  HLOG(kInfo, "Admin: Executing MigrateContainers task");

  auto *pool_manager = CHI_POOL_MANAGER;
  task->num_migrated_ = 0;
  task->error_message_ = "";

  // Deserialize migrations from cereal binary
  std::string data = task->migrations_json_.str();
  std::vector<chi::MigrateInfo> migrations;
  {
    std::istringstream is(data);
    cereal::BinaryInputArchive ar(is);
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
    co_await change_task;

    if (change_task->GetReturnCode() != 0) {
      HLOG(kError,
           "Admin: Failed to change address table for pool {} container {}",
           info.pool_id_, info.container_id_);
      continue;
    }

    // Unregister the container on source node so HasContainer() returns false.
    // This causes ResolveDirectHashQuery to fall through to address_map_ lookup.
    // Note: UnregisterContainer preserves static_container_ for deserialization.
    pool_manager->UnregisterContainer(info.pool_id_, info.container_id_);

    task->num_migrated_++;
    HLOG(kInfo,
         "Admin: Migrated pool {} container {} from node {} to node {}",
         info.pool_id_, info.container_id_, src_node, info.dest_);
  }

  task->SetReturnCode(0);
  HLOG(kInfo, "Admin: MigrateContainers completed, {} migrated",
       task->num_migrated_);
  co_return;
}

/**
 * Attempt to send a retried task to the given node
 * @param entry The retry entry containing the task
 * @param node_id The node to send to
 * @return true if send succeeded
 */
bool Runtime::RetrySendToNode(RetryEntry &entry, chi::u64 node_id) {
  auto *ipc_manager = CHI_IPC;
  auto *pool_manager = CHI_POOL_MANAGER;
  auto *config_manager = CHI_CONFIG_MANAGER;

  const chi::Host *target_host = ipc_manager->GetHost(node_id);
  if (!target_host) {
    return false;
  }
  int port = static_cast<int>(config_manager->GetPort());
  hshm::lbm::Transport *lbm_transport =
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
  hshm::lbm::LbmContext ctx(0);
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
  auto *pool_manager = CHI_POOL_MANAGER;
  const chi::PoolQuery &query = entry.task->pool_query_;

  if (query.IsDirectIdMode()) {
    chi::ContainerId container_id = query.GetContainerId();
    chi::u32 new_node = pool_manager->GetContainerNodeId(
        entry.task->pool_id_, container_id);
    if (new_node != 0 && new_node != entry.target_node_id) {
      return new_node;
    }
  } else if (query.IsRangeMode()) {
    chi::u32 offset = query.GetRangeOffset();
    chi::ContainerId container_id(offset);
    chi::u32 new_node = pool_manager->GetContainerNodeId(
        entry.task->pool_id_, container_id);
    if (new_node != 0 && new_node != entry.target_node_id) {
      return new_node;
    }
  }
  return 0;
}

void Runtime::ProcessRetryQueues() {
  auto *ipc_manager = CHI_IPC;
  auto now = std::chrono::steady_clock::now();

  // Process send_in retry queue
  auto it = send_in_retry_.begin();
  while (it != send_in_retry_.end()) {
    float elapsed = std::chrono::duration<float>(now - it->enqueued_at).count();
    if (elapsed >= kRetryTimeoutSec) {
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
        HLOG(kInfo, "[RetryQueue] Re-routing task from dead node {} to "
             "recovered node {}", it->target_node_id, new_node);
        it->target_node_id = new_node;
        if (RetrySendToNode(*it, new_node)) {
          HLOG(kInfo, "[RetryQueue] SendIn re-routed retry succeeded "
               "for node {}", new_node);
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
    if (elapsed >= kRetryTimeoutSec) {
      HLOG(kError, "[RetryQueue] SendOut task timed out after {}s for node {}",
           elapsed, it->target_node_id);
      // For send_out, the result is lost; origin will timeout
      it = send_out_retry_.erase(it);
    } else if (ipc_manager->IsAlive(it->target_node_id)) {
      // Node came back: retry by calling SendOut
      SendOut(it->task);
      it = send_out_retry_.erase(it);
    } else {
      ++it;
    }
  }
}

void Runtime::ScanSendMapTimeouts() {
  auto *ipc_manager = CHI_IPC;
  auto now = std::chrono::steady_clock::now();

  // Iterate dead nodes and check if any send_map_ entries target them
  const auto &dead_nodes = ipc_manager->GetDeadNodes();
  if (dead_nodes.empty()) return;

  // Build set of dead node IDs for fast lookup
  std::unordered_set<chi::u64> dead_set;
  for (const auto &entry : dead_nodes) {
    // Only timeout entries that have been dead long enough
    float dead_elapsed = std::chrono::duration<float>(
        now - entry.detected_at).count();
    if (dead_elapsed >= kRetryTimeoutSec) {
      dead_set.insert(entry.node_id);
    }
  }

  if (dead_set.empty()) return;

  // Scan send_map_ for tasks targeting dead nodes using for_each
  std::vector<size_t> keys_to_remove;
  send_map_.for_each([&](const size_t &key,
                         hipc::FullPtr<chi::Task> &origin_task) {
    if (origin_task.IsNull() || !origin_task->run_ctx_) return;

    chi::RunContext *rctx = origin_task->run_ctx_.get();
    // Check if any replica targets a dead node
    bool any_dead = false;
    for (const auto &pq : rctx->pool_queries_) {
      if (pq.IsPhysicalMode() && dead_set.count(pq.GetNodeId())) {
        any_dead = true;
        break;
      }
    }

    if (any_dead) {
      HLOG(kError, "[ScanSendMapTimeouts] Task {} timed out waiting for dead node",
           origin_task->task_id_);
      origin_task->SetReturnCode(kNetworkTimeoutRC);
      // Complete the task as failed
      auto *worker = CHI_CUR_WORKER;
      worker->EndTask(origin_task, rctx, true);
      keys_to_remove.push_back(key);
    }
  });

  for (size_t key : keys_to_remove) {
    send_map_.erase(key);
  }
}

chi::TaskResume Runtime::Heartbeat(hipc::FullPtr<HeartbeatTask> task,
                                   chi::RunContext &rctx) {
  task->SetReturnCode(0);
  rctx.did_work_ = true;
  co_return;
}

chi::TaskResume Runtime::HeartbeatProbe(hipc::FullPtr<HeartbeatProbeTask> task,
                                        chi::RunContext &rctx) {
  auto *ipc_manager = CHI_IPC;
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
      float elapsed =
          std::chrono::duration<float>(now - it->sent_at).count();
      if (elapsed > kDirectProbeTimeoutSec) {
        // Direct probe timed out - escalate to indirect probing
        ipc_manager->SetNodeState(it->target_node_id,
                                  chi::NodeState::kProbeFailed);
        HLOG(kWarning,
             "SWIM: Direct probe to node {} timed out, starting indirect probes",
             it->target_node_id);

        // Select k random alive helpers (excluding self and target)
        const auto &hosts = ipc_manager->GetAllHosts();
        std::vector<chi::u64> candidates;
        for (const auto &h : hosts) {
          if (h.node_id != self_node_id &&
              h.node_id != it->target_node_id && h.IsAlive()) {
            candidates.push_back(h.node_id);
          }
        }
        std::shuffle(candidates.begin(), candidates.end(), probe_rng_);
        size_t num_helpers =
            std::min(kIndirectProbeHelpers, candidates.size());
        for (size_t i = 0; i < num_helpers; ++i) {
          auto future = client_.AsyncProbeRequest(
              chi::PoolQuery::Physical(candidates[i]),
              it->target_node_id);
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
      it->future.Wait(0);  // Finalize
      if (it->future->probe_result_ == 0) {
        // Indirect probe succeeded - node is alive
        ipc_manager->SetNodeState(it->target_node_id,
                                  chi::NodeState::kAlive);
        HLOG(kInfo,
             "SWIM: Indirect probe via node {} confirmed node {} alive",
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
        it = pending_indirect_probes_.erase(it);
        did_work = true;
      }
    } else {
      float elapsed =
          std::chrono::duration<float>(now - it->sent_at).count();
      if (elapsed > kIndirectProbeTimeoutSec) {
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
            ipc_manager->GetNodeState(target) ==
                chi::NodeState::kProbeFailed) {
          ipc_manager->SetNodeState(target, chi::NodeState::kSuspected);
          HLOG(kWarning,
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
        if (since_change >= kSuspicionTimeoutSec) {
          HLOG(kError, "SWIM: Node {} confirmed dead after suspicion timeout",
               h.node_id);
          ipc_manager->SetDead(h.node_id);
          did_work = true;
          TriggerRecovery(h.node_id);
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
        HLOG(kFatal,
             "SWIM: Self-fencing! {} of {} other nodes are suspected/dead",
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
        auto future = client_.AsyncHeartbeat(
            chi::PoolQuery::Physical(h.node_id));
        pending_direct_probes_.push_back(
            {std::move(future), h.node_id,
             std::chrono::steady_clock::now()});
        probe_round_robin_idx_ = (idx + 1) % hosts.size();
        did_work = true;
        break;  // One probe per invocation
      }
    }
  }

  rctx.did_work_ = did_work;
  task->SetReturnCode(0);
  co_return;
}

chi::TaskResume Runtime::ProbeRequest(hipc::FullPtr<ProbeRequestTask> task,
                                      chi::RunContext &rctx) {
  // Probe the target node on behalf of the requester using cooperative yield
  auto future = client_.AsyncHeartbeat(
      chi::PoolQuery::Physical(task->target_node_id_));
  auto start = std::chrono::steady_clock::now();

  while (!future.IsComplete()) {
    float elapsed = std::chrono::duration<float>(
        std::chrono::steady_clock::now() - start).count();
    if (elapsed >= kIndirectProbeTimeoutSec) break;
    co_await chi::yield(1000.0);
  }

  if (future.IsComplete()) {
    future.Wait(0);  // Finalize (already complete)
    task->probe_result_ = 0;  // alive
  } else {
    task->probe_result_ = -1;  // unreachable
  }

  task->SetReturnCode(0);
  rctx.did_work_ = true;
  co_return;
}

chi::u64 Runtime::GetWorkRemaining() const {
  // Note: No lock needed - single net worker processes all Send/Recv tasks
  return send_map_.size() + recv_map_.size();
}

//===========================================================================
// Recovery Methods
//===========================================================================

std::vector<chi::RecoveryAssignment> Runtime::ComputeRecoveryPlan(
    chi::u64 dead_node_id) {
  auto *pool_manager = CHI_POOL_MANAGER;
  auto *ipc_manager = CHI_IPC;

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
          dest = static_cast<chi::u32>(
              alive_nodes[rr_idx % alive_nodes.size()]);
          rr_idx++;
        }
        ra.dest_node_id_ = dest;
        assignments.push_back(std::move(ra));
      }
    }
  }
  return assignments;
}

void Runtime::TriggerRecovery(chi::u64 dead_node_id) {
  auto *ipc_manager = CHI_IPC;
  if (!ipc_manager->IsLeader()) return;
  if (recovery_initiated_.count(dead_node_id)) return;
  recovery_initiated_.insert(dead_node_id);
  if (ipc_manager->IsSelfFenced()) {
    HLOG(kWarning, "Recovery: Skipping for node {} - self-fenced",
         dead_node_id);
    return;
  }

  HLOG(kInfo, "Recovery: Leader initiating for dead node {}", dead_node_id);
  auto assignments = ComputeRecoveryPlan(dead_node_id);
  if (assignments.empty()) {
    HLOG(kInfo, "Recovery: No containers to recover from node {}",
         dead_node_id);
    return;
  }

  HLOG(kInfo, "Recovery: {} containers to redistribute from node {}",
       assignments.size(), dead_node_id);
  client_.AsyncRecoverContainers(
      chi::PoolQuery::Broadcast(), assignments, dead_node_id);
}

chi::TaskResume Runtime::RecoverContainers(
    hipc::FullPtr<RecoverContainersTask> task, chi::RunContext &rctx) {
  auto *ipc_manager = CHI_IPC;
  auto *pool_manager = CHI_POOL_MANAGER;
  auto *module_manager = CHI_MODULE_MANAGER;
  chi::u64 self_node_id = ipc_manager->GetNodeId();
  task->num_recovered_ = 0;

  // Deserialize assignments
  std::vector<chi::RecoveryAssignment> assignments;
  {
    std::istringstream is(task->assignments_data_.str());
    cereal::BinaryInputArchive ar(is);
    ar(assignments);
  }

  for (const auto &ra : assignments) {
    // ALL nodes update address_map_
    pool_manager->UpdateContainerNodeMapping(
        ra.pool_id_, ra.container_id_, ra.dest_node_id_);
    pool_manager->WriteAddressTableWAL(
        ra.pool_id_, ra.container_id_, ra.dead_node_id_, ra.dest_node_id_);

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
    pool_manager->RegisterContainer(
        ra.pool_id_, ra.container_id_, container, false);
    task->num_recovered_++;
  }

  task->SetReturnCode(0);
  rctx.did_work_ = true;
  co_return;
}

//===========================================================================
// Task Serialization Method Implementations
//===========================================================================

// Task Serialization Method Implementations now in autogen/admin_lib_exec.cc

}  // namespace chimaera::admin

// Define ChiMod entry points using CHI_TASK_CC macro
CHI_TASK_CC(chimaera::admin::Runtime)