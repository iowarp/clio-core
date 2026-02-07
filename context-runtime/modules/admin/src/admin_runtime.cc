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
#include <hermes_shm/lightbeam/zmq_transport.h>
#include <zmq.h>

#include <chrono>
#include <memory>
#include <sstream>
#include <thread>
#include <unordered_map>
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

  // Spawn periodic Heartbeat task with 5ms period
  // This task polls for ZMQ heartbeat requests and responds
  client_.AsyncHeartbeat(chi::PoolQuery::Local(), 5000);

  // Spawn periodic WreapDeadIpcs task with 1 second period
  // This task reaps shared memory segments from dead processes
  client_.AsyncWreapDeadIpcs(chi::PoolQuery::Local(), 1000000);

  HLOG(kDebug,
       "Admin: Container created and initialized for pool: {} (ID: {}, count: "
       "{})",
       pool_name_, task->new_pool_id_, create_count_);
  HLOG(kDebug, "Admin: Spawned periodic Recv, Send, and Heartbeat tasks");
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
  chi::Container *container = pool_manager->GetContainer(origin_task->pool_id_);
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

    // Get or create persistent Lightbeam client using connection pool
    auto *config_manager = CHI_CONFIG_MANAGER;
    int port = static_cast<int>(config_manager->GetPort());
    hshm::lbm::Client *lbm_client =
        ipc_manager->GetOrCreateClient(target_host->ip_address, port);

    if (!lbm_client) {
      HLOG(kError, "[SendIn] Task {} FAILED: Could not get client for {}:{}",
           origin_task->task_id_, target_host->ip_address, port);
      continue;
    }

    // Create SaveTaskArchive with SerializeIn mode and lbm_client
    chi::SaveTaskArchive archive(chi::MsgType::kSerializeIn, lbm_client);

    // Create task copy
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

    // Serialize the task using container->SaveTask (Expose will be called
    // automatically for bulks)
    container->SaveTask(task_copy->method_, archive, task_copy);

    // Send using Lightbeam asynchronously (non-blocking)
    // Note: No lock needed - single net worker processes all Send/Recv tasks
    hshm::lbm::LbmContext ctx(0);  // Non-blocking async send
    int rc = lbm_client->Send(archive, ctx);

    if (rc != 0) {
      HLOG(kError,
           "[SendIn] Task {} Lightbeam async Send FAILED with error code {}",
           origin_task->task_id_, rc);
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

  // Validate origin_task
  if (origin_task.IsNull()) {
    HLOG(kError, "SendOut: origin_task is null");
    return;
  }

  // Get the container associated with the origin_task
  chi::Container *container = pool_manager->GetContainer(origin_task->pool_id_);
  if (container == nullptr) {
    HLOG(kError, "SendOut: container not found for pool_id {}",
         origin_task->pool_id_);
    return;
  }

  // Remove task from recv_map as we're completing it (use net_key for lookup)
  // Note: No lock needed - single net worker processes all Send/Recv tasks
  size_t net_key = origin_task->task_id_.net_key_;
  auto *it = recv_map_.find(net_key);
  if (it == nullptr) {
    HLOG(kError,
         "[SendOut] Task {} FAILED: Not found in recv_map (size: {}) with "
         "net_key {}",
         origin_task->task_id_, recv_map_.size(), net_key);
    return;
  }
  recv_map_.erase(net_key);

  // Get return node from pool_query
  chi::u64 target_node_id = origin_task->pool_query_.GetReturnNode();

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
  hshm::lbm::Client *lbm_client =
      ipc_manager->GetOrCreateClient(target_host->ip_address, port);

  if (lbm_client == nullptr) {
    HLOG(kError, "[SendOut] Task {} FAILED: Could not get client for {}:{}",
         origin_task->task_id_, target_host->ip_address, port);
    return;
  }

  // Create SaveTaskArchive with SerializeOut mode and lbm_client
  // The client will automatically call Expose internally during serialization
  chi::SaveTaskArchive archive(chi::MsgType::kSerializeOut, lbm_client);

  // Serialize the task outputs using container->SaveTask (Expose called
  // automatically)
  container->SaveTask(origin_task->method_, archive, origin_task);

  // Use non-timed, non-sync context for SendOut
  // Note: No lock needed - single net worker processes all Send/Recv tasks
  hshm::lbm::LbmContext ctx(0);
  int rc = lbm_client->Send(archive, ctx);
  if (rc != 0) {
    HLOG(kError, "[SendOut] Task {} Lightbeam Send FAILED with error code {}",
         origin_task->task_id_, rc);
    return;
  }

  HLOG(kDebug, "[SendOut] Task {}", origin_task->task_id_);

  // Delete the task after sending outputs
  ipc_manager->DelTask(origin_task);
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

  // Poll priority 0 (SendIn) queue - tasks waiting to be sent to remote nodes
  while (ipc_manager->TryPopNetTask(chi::NetQueuePriority::kSendIn,
                                    queued_future)) {
    // Get the original task from the Future
    auto origin_task = queued_future.GetTaskPtr();
    if (!origin_task.IsNull()) {
      HLOG(kInfo, "[Send] Processing SendIn task method={}, pool_id={}",
           origin_task->method_, origin_task->pool_id_);
      SendIn(origin_task, rctx);
      did_send = true;
      send_in_count++;
    }
  }

  if (send_in_count > 0) {
    HLOG(kInfo, "[Send] Processed {} SendIn tasks", send_in_count);
  }

  // Poll priority 1 (SendOut) queue - tasks with outputs to send back
  int send_out_count = 0;
  while (ipc_manager->TryPopNetTask(chi::NetQueuePriority::kSendOut,
                                    queued_future)) {
    // Get the original task from the Future
    auto origin_task = queued_future.GetTaskPtr();
    if (!origin_task.IsNull()) {
      HLOG(kInfo, "[Send] Processing SendOut task method={}, pool_id={}",
           origin_task->method_, origin_task->pool_id_);
      SendOut(origin_task);
      did_send = true;
      send_out_count++;
    }
  }

  if (send_out_count > 0) {
    HLOG(kInfo, "[Send] Processed {} SendOut tasks", send_out_count);
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
 * @param lbm_server Lightbeam server for receiving bulk data
 */
void Runtime::RecvIn(hipc::FullPtr<RecvTask> task,
                     chi::LoadTaskArchive &archive,
                     hshm::lbm::Server *lbm_server) {
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

  // Allocate buffers for bulk data and expose them for receiving
  // archive.send contains sender's bulk descriptors (populated by RecvMetadata)
  for (const auto &send_bulk : archive.send) {
    hipc::FullPtr<char> buffer = ipc_manager->AllocateBuffer(send_bulk.size);
    archive.recv.push_back(
        lbm_server->Expose(buffer, send_bulk.size, send_bulk.flags.bits_));
  }

  // Receive all bulk data using Lightbeam
  int rc = lbm_server->RecvBulks(archive);
  if (rc != 0) {
    HLOG(kError, "Admin: Lightbeam RecvBulks failed with error code {}", rc);
    task->SetReturnCode(4);
    return;
  }

  for (size_t task_idx = 0; task_idx < task_infos.size(); ++task_idx) {
    const auto &task_info = task_infos[task_idx];

    // Get container associated with PoolId
    chi::Container *container = pool_manager->GetContainer(task_info.pool_id_);
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

    // Mark task as remote, set as data owner, unset periodic and TASK_FORCE_NET
    task_ptr->SetFlags(TASK_REMOTE | TASK_DATA_OWNER);
    task_ptr->ClearFlags(TASK_PERIODIC | TASK_FORCE_NET | TASK_ROUTED);

    // Add task to recv_map for later lookup (use net_key from task_id)
    // Note: No lock needed - single net worker processes all Send/Recv tasks
    size_t net_key = task_ptr->task_id_.net_key_;
    recv_map_[net_key] = task_ptr;

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
 * @param lbm_server Lightbeam server for receiving bulk data
 */
void Runtime::RecvOut(hipc::FullPtr<RecvTask> task,
                      chi::LoadTaskArchive &archive,
                      hshm::lbm::Server *lbm_server) {
  // Set I/O size to 1MB to ensure routing to slow workers
  task->stat_.io_size_ = 1024 * 1024;  // 1MB

  auto *pool_manager = CHI_POOL_MANAGER;

  const auto &task_infos = archive.GetTaskInfos();

  // If no task outputs to receive
  if (task_infos.empty()) {
    task->SetReturnCode(0);
    return;
  }

  // Set lbm_server in archive for bulk transfer exposure in output mode
  archive.SetLbmServer(lbm_server);

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
        pool_manager->GetContainer(origin_task->pool_id_);
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

  // Receive all bulk data using Lightbeam
  int rc = lbm_server->RecvBulks(archive);
  if (rc != 0) {
    HLOG(kError, "Admin: Lightbeam RecvBulks failed with error code {}", rc);
    task->SetReturnCode(4);
    return;
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
        pool_manager->GetContainer(origin_task->pool_id_);
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
          pool_manager->GetContainer(origin_task->pool_id_);

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

  hshm::lbm::Server *lbm_server = ipc_manager->GetMainServer();
  if (lbm_server == nullptr) {
    co_return;
  }

  // Note: No socket lock needed - single net worker processes all Recv tasks

  // Receive metadata first to determine mode (non-blocking)
  chi::LoadTaskArchive archive;
  int rc = lbm_server->RecvMetadata(archive);
  if (rc == EAGAIN) {
    // No message available - this is normal for polling, mark as no work done
    task->SetReturnCode(0);
    rctx.did_work_ = false;
    co_return;
  }

  if (rc != 0) {
    if (rc != -1) {
      HLOG(kError, "Admin: Lightbeam RecvMetadata failed with error code {}",
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
      RecvIn(task, archive, lbm_server);
      break;
    case chi::MsgType::kSerializeOut:
      RecvOut(task, archive, lbm_server);
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
 * Handle Heartbeat - Respond to heartbeat request
 * Polls heartbeat server for ZMQ REQ/REP requests and responds
 * Also sets task response to 0 to indicate runtime is healthy
 * @param task The heartbeat task
 * @param rctx Run context
 */
chi::TaskResume Runtime::Heartbeat(hipc::FullPtr<HeartbeatTask> task,
                                   chi::RunContext &rctx) {
  auto *ipc_manager = CHI_IPC;

  // Poll heartbeat socket - RECEIVE request and SEND response
  // This ensures clients can verify the runtime is running
  void *hb_socket = ipc_manager->GetHeartbeatSocket();
  if (hb_socket != nullptr) {
    // RECEIVE heartbeat request (non-blocking)
    int32_t request;
    int rc = zmq_recv(hb_socket, &request, sizeof(request), ZMQ_DONTWAIT);
    if (rc != -1) {
      // Received a heartbeat request - SEND response (0 = success)
      int32_t response = 0;
      zmq_send(hb_socket, &response, sizeof(response), 0);
      HLOG(kDebug, "Heartbeat: received request {}, sent response {}", request,
           response);
      // Mark that we did work (received and responded to heartbeat)
      rctx.did_work_ = true;
    } else {
      // No heartbeat request available (EAGAIN)
      rctx.did_work_ = false;
    }
  } else {
    // No heartbeat socket available
    rctx.did_work_ = false;
  }

  // Set task response to indicate runtime is healthy
  task->response_ = 0;
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
          pool_manager->GetContainer(task_info.pool_id_);
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

chi::u64 Runtime::GetWorkRemaining() const {
  // Note: No lock needed - single net worker processes all Send/Recv tasks
  return send_map_.size() + recv_map_.size();
}

//===========================================================================
// Task Serialization Method Implementations
//===========================================================================

// Task Serialization Method Implementations now in autogen/admin_lib_exec.cc

}  // namespace chimaera::admin

// Define ChiMod entry points using CHI_TASK_CC macro
CHI_TASK_CC(chimaera::admin::Runtime)