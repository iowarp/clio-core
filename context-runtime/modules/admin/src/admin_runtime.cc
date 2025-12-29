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

void Runtime::Create(hipc::FullPtr<CreateTask> task, chi::RunContext &rctx) {
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
  client_.AsyncRecv(chi::PoolQuery::Local(), 0, 25);

  // Spawn periodic Send task with 25 microsecond period
  // This task polls net_queue_ for send operations
  client_.AsyncSendPoll(chi::PoolQuery::Local(), 0, 25);

  HLOG(kDebug,
       "Admin: Container created and initialized for pool: {} (ID: {}, count: "
       "{})",
       pool_name_, task->new_pool_id_, create_count_);
  HLOG(kDebug, "Admin: Spawned periodic Recv and Send tasks with 25us period");
}

void Runtime::GetOrCreatePool(
    hipc::FullPtr<
        chimaera::admin::GetOrCreatePoolTask<chimaera::admin::CreateParams>>
        task,
    chi::RunContext &rctx) {
  // Get pool manager once - used by both dynamic scheduling and normal
  // execution
  auto *pool_manager = CHI_POOL_MANAGER;

  // Extract pool name once
  std::string pool_name = task->pool_name_.str();

  // Check if this is dynamic scheduling mode
  if (rctx.exec_mode == chi::ExecMode::kDynamicSchedule) {
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
    return;
  }

  // Pool get-or-create operation logic (IS_ADMIN=false)
  HLOG(kDebug, "Admin: Executing GetOrCreatePool task - ChiMod: {}, Pool: {}",
       task->chimod_name_.str(), pool_name);

  // Initialize output values
  task->return_code_ = 0;
  task->error_message_ = "";

  try {
    // Use the simplified PoolManager API that extracts all parameters from the
    // task
    if (!pool_manager->CreatePool(task.Cast<chi::Task>(), &rctx)) {
      task->return_code_ = 2;
      task->error_message_ = "Failed to create or get pool via PoolManager";
      return;
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
    auto alloc = CHI_IPC->GetMainAlloc();
    std::string error_msg =
        std::string("Exception during pool creation: ") + e.what();
    task->error_message_ = chi::priv::string(alloc, error_msg);
    HLOG(kError, "Admin: Pool creation failed with exception: {}", e.what());
  }
}

void Runtime::Destroy(hipc::FullPtr<DestroyTask> task, chi::RunContext &rctx) {
  // DestroyTask is aliased to DestroyPoolTask, so delegate to DestroyPool
  DestroyPool(task, rctx);
}

void Runtime::DestroyPool(hipc::FullPtr<DestroyPoolTask> task,
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
      return;
    }

    // Use PoolManager to destroy the complete pool including metadata
    if (!pool_manager->DestroyPool(target_pool)) {
      task->return_code_ = 2;
      task->error_message_ = "Failed to destroy pool via PoolManager";
      return;
    }

    // Set success results
    task->return_code_ = 0;
    pools_destroyed_++;

    HLOG(kDebug,
         "Admin: Pool destroyed successfully - ID: {} (Total pools destroyed: "
         "{})",
         target_pool, pools_destroyed_);

  } catch (const std::exception &e) {
    task->return_code_ = 99;
    auto alloc = CHI_IPC->GetMainAlloc();
    std::string error_msg =
        std::string("Exception during pool destruction: ") + e.what();
    task->error_message_ = chi::priv::string(alloc, error_msg);
    HLOG(kError, "Admin: Pool destruction failed with exception: {}", e.what());
  }
}

void Runtime::StopRuntime(hipc::FullPtr<StopRuntimeTask> task,
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
    auto alloc = CHI_IPC->GetMainAlloc();
    std::string error_msg =
        std::string("Exception during runtime shutdown: ") + e.what();
    task->error_message_ = chi::priv::string(alloc, error_msg);
    HLOG(kError, "Admin: Runtime shutdown failed with exception: {}", e.what());
  }
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

void Runtime::Flush(hipc::FullPtr<FlushTask> task, chi::RunContext &rctx) {
  HLOG(kDebug, "Admin: Executing Flush task");

  // Initialize output values
  task->return_code_ = 0;
  task->total_work_done_ = 0;

  try {
    // Get WorkOrchestrator to check work remaining across all containers
    auto *work_orchestrator = CHI_WORK_ORCHESTRATOR;
    if (!work_orchestrator || !work_orchestrator->IsInitialized()) {
      task->return_code_ = 1;
      return;
    }

    // Loop until all work is complete
    chi::u64 total_work_remaining = 0;
    while (work_orchestrator->HasWorkRemaining(total_work_remaining)) {
      HLOG(kDebug,
           "Admin: Flush found {} work units still remaining, waiting...",
           total_work_remaining);

      // Brief sleep to avoid busy waiting
      task->Yield(25);
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

  // Log host information at method entry
  const auto &this_host = CHI_IPC->GetThisHost();
  HLOG(kDebug, "SendIn executing on host {} (node_id: {})",
       this_host.ip_address, this_host.node_id);

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

  HLOG(kDebug,
       "=== [SendIn BEGIN] Task {} (pool: {}) starting distributed send ===",
       origin_task->task_id_, origin_task->pool_id_);

  // Pre-allocate send_map_key using origin_task pointer
  // This ensures consistent net_key across all replicas
  size_t send_map_key = size_t(origin_task.ptr_);

  // Add the origin task to send_map before creating copies
  // Note: No lock needed - single net worker processes all Send/Recv tasks
  send_map_[send_map_key] = origin_task;
  HLOG(kDebug, "[SendIn] Added origin task {} to send_map with net_key {}",
       origin_task->task_id_, send_map_key);

  // Get pool_queries from task's RunContext
  chi::RunContext *origin_task_rctx = origin_task->run_ctx_;
  if (origin_task_rctx == nullptr) {
    HLOG(kError, "SendIn: origin_task has no RunContext");
    return;
  }

  const std::vector<chi::PoolQuery> &pool_queries =
      origin_task_rctx->pool_queries;
  size_t num_replicas = pool_queries.size();

  // Reserve space for all replicas in subtasks vector BEFORE the loop
  // This ensures subtasks_.size() reflects the correct total replica count
  origin_task_rctx->subtasks_.resize(num_replicas);
  HLOG(kDebug, "[SendIn] Reserved space for {} replicas in subtasks vector",
       num_replicas);

  // Send to each target in pool_queries
  HLOG(kDebug, "SendIn - processing {} replicas for pool_id={}", num_replicas,
       origin_task->pool_id_);

  for (size_t i = 0; i < num_replicas; ++i) {
    const chi::PoolQuery &query = pool_queries[i];

    HLOG(kDebug,
         "SendIn - replica {} routing_mode={}, range_offset={}, range_count={}",
         i, static_cast<int>(query.GetRoutingMode()), query.GetRangeOffset(),
         query.GetRangeCount());

    // Determine target node_id based on query type
    chi::u64 target_node_id = 0;

    if (query.IsLocalMode()) {
      target_node_id = ipc_manager->GetNodeId();
      HLOG(kDebug, "SendIn - replica {} LocalMode -> target_node_id={}", i,
           target_node_id);
    } else if (query.IsPhysicalMode()) {
      target_node_id = query.GetNodeId();
      HLOG(kDebug, "SendIn - replica {} PhysicalMode -> target_node_id={}", i,
           target_node_id);
    } else if (query.IsDirectIdMode()) {
      chi::ContainerId container_id = query.GetContainerId();
      target_node_id =
          pool_manager->GetContainerNodeId(origin_task->pool_id_, container_id);
      HLOG(kDebug,
           "SendIn - replica {} DirectIdMode container_id={} -> "
           "target_node_id={}",
           i, container_id, target_node_id);
    } else if (query.IsRangeMode()) {
      chi::u32 offset = query.GetRangeOffset();
      chi::ContainerId container_id(offset);
      target_node_id =
          pool_manager->GetContainerNodeId(origin_task->pool_id_, container_id);
      HLOG(kDebug,
           "SendIn - replica {} RangeMode offset={} container_id={} -> "
           "target_node_id={}",
           i, offset, container_id, target_node_id);
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

    HLOG(kDebug, "SendIn - replica {} target_host: ip={}, node_id={}", i,
         target_host->ip_address, target_host->node_id);

    // Get or create persistent Lightbeam client using connection pool
    auto *config_manager = CHI_CONFIG_MANAGER;
    int port = static_cast<int>(config_manager->GetPort());
    HLOG(kDebug, "SendIn - getting pooled client to {}:{}",
         target_host->ip_address, port);
    hshm::lbm::Client *lbm_client =
        ipc_manager->GetOrCreateClient(target_host->ip_address, port);

    if (!lbm_client) {
      HLOG(kError, "[SendIn] Task {} FAILED: Could not get client for {}:{}",
           origin_task->task_id_, target_host->ip_address, port);
      continue;
    }

    HLOG(kDebug, "SendIn - lbm_client obtained from pool: OK");

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

    HLOG(kDebug, "SendIn - created task copy {} with net_key {} for node {}",
         task_copy->task_id_, send_map_key, target_node_id);

    // Update the copy's pool query to current query
    task_copy->pool_query_ = query;

    // Set return node ID in the pool query
    chi::u64 this_node_id = ipc_manager->GetNodeId();
    task_copy->pool_query_.SetReturnNode(this_node_id);

    // Serialize the task using container->SaveTask (Expose will be called
    // automatically for bulks)
    HLOG(kDebug, "SendIn - serializing task...");
    container->SaveTask(task_copy->method_, archive, task_copy);

    // Send using Lightbeam asynchronously (non-blocking)
    // Note: No lock needed - single net worker processes all Send/Recv tasks
    HLOG(kDebug, "SendIn - calling Lightbeam Send...");
    hshm::lbm::LbmContext ctx(0);  // Non-blocking async send
    int rc = lbm_client->Send(archive, ctx);

    if (rc != 0) {
      HLOG(kError,
           "[SendIn] Task {} Lightbeam async Send FAILED with error code {}",
           origin_task->task_id_, rc);
      continue;
    }

    HLOG(kDebug, "SendIn - task {} sent to node {} (rc={})",
         task_copy->task_id_, target_node_id, rc);
  }

  HLOG(kDebug, "SendIn END - completed sending to {} targets", num_replicas);
}

/**
 * Helper function: Send task outputs back to origin node
 * @param origin_task Completed task whose outputs need to be sent back
 */
void Runtime::SendOut(hipc::FullPtr<chi::Task> origin_task) {
  auto *ipc_manager = CHI_IPC;
  auto *pool_manager = CHI_POOL_MANAGER;

  // Log host information at method entry
  const auto &this_host = CHI_IPC->GetThisHost();
  HLOG(kDebug, "SendOut executing on host {} (node_id: {})",
       this_host.ip_address, this_host.node_id);

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

  HLOG(kDebug,
       "=== [SendOut BEGIN] Task {} (pool: {}, method: {}) sending outputs "
       "back ===",
       origin_task->task_id_, origin_task->pool_id_, origin_task->method_);

  // Remove task from recv_map as we're completing it (use net_key for lookup)
  // Note: No lock needed - single net worker processes all Send/Recv tasks
  size_t net_key = origin_task->task_id_.net_key_;
  HLOG(kDebug, "[SendOut] Removing task {} from recv_map with net_key {}",
       origin_task->task_id_, net_key);
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

  HLOG(kDebug, "[SEND] Task {} outputs sent back to node {}",
       origin_task->task_id_, target_node_id);

  // Delete the task after sending outputs
  ipc_manager->DelTask(origin_task);
  HLOG(kDebug, "=== [SendOut END] Task {} completed and deleted ===",
       origin_task->task_id_);
}

/**
 * Main Send function - periodic task that polls net_queue_ for send operations
 * Polls both SendIn (priority 0) and SendOut (priority 1) queues
 */
void Runtime::Send(hipc::FullPtr<SendTask> task, chi::RunContext &rctx) {
  auto *ipc_manager = CHI_IPC;
  chi::Future<chi::Task> queued_future;

  // Poll priority 0 (SendIn) queue - tasks waiting to be sent to remote nodes
  while (ipc_manager->TryPopNetTask(chi::NetQueuePriority::kSendIn,
                                    queued_future)) {
    // Get the original task from the Future
    auto origin_task = queued_future.GetTaskPtr();
    if (!origin_task.IsNull()) {
      SendIn(origin_task, rctx);
    }
  }

  // Poll priority 1 (SendOut) queue - tasks with outputs to send back
  while (ipc_manager->TryPopNetTask(chi::NetQueuePriority::kSendOut,
                                    queued_future)) {
    // Get the original task from the Future
    auto origin_task = queued_future.GetTaskPtr();
    if (!origin_task.IsNull()) {
      SendOut(origin_task);
    }
  }

  task->SetReturnCode(0);
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

  // Log host information at method entry
  auto &this_host = CHI_IPC->GetThisHost();
  HLOG(kDebug, "RecvIn executing on host {} (node_id: {})",
       this_host.ip_address, this_host.node_id);

  const auto &task_infos = archive.GetTaskInfos();
  HLOG(kDebug,
       "=== [RecvIn BEGIN] (node={}) Receiving {} task(s) from remote node ===",
       ipc_manager->GetNodeId(), task_infos.size());

  // If no tasks to receive
  if (task_infos.empty()) {
    HLOG(kDebug, "=== [RecvIn END] No tasks to receive ===");
    task->SetReturnCode(0);
    return;
  }

  // Log bulk counts from sender's metadata
  HLOG(kDebug, "[RecvIn] Sender's meta.send has {} entries",
       archive.send.size());
  for (size_t i = 0; i < archive.send.size(); ++i) {
    HLOG(kDebug, "[RecvIn] send[{}]: size={}, flags={}", i,
         archive.send[i].size, archive.send[i].flags.bits_);
  }

  // Allocate buffers for bulk data and expose them for receiving
  // archive.send contains sender's bulk descriptors (populated by RecvMetadata)
  for (const auto &send_bulk : archive.send) {
    hipc::FullPtr<char> buffer = ipc_manager->AllocateBuffer(send_bulk.size);
    archive.recv.push_back(
        lbm_server->Expose(buffer, send_bulk.size, send_bulk.flags.bits_));
  }

  // Log recv counts after allocation
  HLOG(kDebug, "[RecvIn] After allocation, meta.recv has {} entries",
       archive.recv.size());
  for (size_t i = 0; i < archive.recv.size(); ++i) {
    HLOG(kDebug, "[RecvIn] recv[{}]: size={}, flags={}", i,
         archive.recv[i].size, archive.recv[i].flags.bits_);
  }

  // Receive all bulk data using Lightbeam
  int rc = lbm_server->RecvBulks(archive);
  if (rc != 0) {
    HLOG(kError, "Admin: Lightbeam RecvBulks failed with error code {}", rc);
    task->SetReturnCode(4);
    return;
  }

  HLOG(kDebug, "Admin: Received {} bulk transfers via Lightbeam",
       archive.recv.size());

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

    HLOG(kDebug,
         "[RecvIn] Received task {} (pool: {}) with net_key {}, added to "
         "recv_map",
         task_ptr->task_id_, task_info.pool_id_, net_key);

    // Send task for execution using IpcManager::Send with awake_event=false
    // Note: This creates a Future and enqueues it to worker lanes
    // awake_event=false prevents setting parent task for received remote tasks
    (void)ipc_manager->Send(task_ptr, false);
    HLOG(kDebug, "[RECV] Task {} received and sent for execution",
         task_ptr->task_id_);
  }

  HLOG(kDebug, "=== [RecvIn END] Processed {} task(s) ===", task_infos.size());
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

  // Log host information at method entry
  auto &this_host = CHI_IPC->GetThisHost();
  HLOG(kDebug, "RecvOut executing on host {} (node_id: {})",
       this_host.ip_address, this_host.node_id);

  const auto &task_infos = archive.GetTaskInfos();
  HLOG(kDebug,
       "=== [RecvOut BEGIN] Receiving {} task output(s) from remote node ===",
       task_infos.size());

  // If no task outputs to receive
  if (task_infos.empty()) {
    HLOG(kDebug, "=== [RecvOut END] No task outputs to receive ===");
    task->SetReturnCode(0);
    return;
  }

  // Set lbm_server in archive for bulk transfer exposure in output mode
  archive.SetLbmServer(lbm_server);

  // Log bulk counts from sender's metadata
  HLOG(kDebug, "[RecvOut] Sender's meta.send has {} entries",
       archive.send.size());
  for (size_t i = 0; i < archive.send.size(); ++i) {
    HLOG(kDebug, "[RecvOut] send[{}]: size={}, flags={}", i,
         archive.send[i].size, archive.send[i].flags.bits_);
  }

  // First pass: Deserialize to expose buffers
  // LoadTask will call ar.bulk() which will expose the pointers and populate
  // archive.recv
  for (size_t task_idx = 0; task_idx < task_infos.size(); ++task_idx) {
    const auto &task_info = task_infos[task_idx];

    // Locate origin task from send_map using net_key
    size_t net_key = task_info.task_id_.net_key_;
    HLOG(kDebug,
         "[RecvOut] Looking up origin task for replica {} with net_key {}",
         task_info.task_id_, net_key);

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
    HLOG(kDebug, "[RecvOut] Found origin task {} for replica {}",
         origin_task->task_id_, task_info.task_id_);
    chi::RunContext *origin_rctx = origin_task->run_ctx_;

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

  // Log recv counts after deserialization
  HLOG(kDebug, "[RecvOut] After deserialization, meta.recv has {} entries",
       archive.recv.size());
  for (size_t i = 0; i < archive.recv.size(); ++i) {
    HLOG(kDebug, "[RecvOut] recv[{}]: size={}, flags={}", i,
         archive.recv[i].size, archive.recv[i].flags.bits_);
  }

  // Receive all bulk data using Lightbeam
  int rc = lbm_server->RecvBulks(archive);
  if (rc != 0) {
    HLOG(kError, "Admin: Lightbeam RecvBulks failed with error code {}", rc);
    task->SetReturnCode(4);
    return;
  }

  HLOG(kDebug, "[RecvOut] Received {} bulk transfers via Lightbeam",
       archive.recv.size());

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
    chi::RunContext *origin_rctx = origin_task->run_ctx_;

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
    HLOG(kDebug, "[RECV] Task {} outputs received and aggregated",
         origin_task->task_id_);

    // Increment completed replicas counter in origin's rctx
    origin_rctx->completed_replicas_++;
    chi::u32 completed = origin_rctx->completed_replicas_;
    HLOG(kDebug, "[RecvOut] Origin task {} completed {}/{} replicas",
         origin_task->task_id_, completed, origin_rctx->subtasks_.size());

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
      HLOG(kInfo,
           "[TRACE] RecvOut - ALL replicas complete for task {}, adding to "
           "blocked queue",
           origin_task->task_id_);
      worker->EndTask(origin_task, origin_rctx, true);
      HLOG(kInfo,
           "[TRACE] RecvOut - Added origin task {} to blocked queue, "
           "run_ctx={:#x}",
           origin_task->task_id_, reinterpret_cast<uintptr_t>(origin_rctx));
    }
  }

  HLOG(kDebug,
       "=== [RecvOut END] Processed {} task output(s) ===", task_infos.size());
  task->SetReturnCode(0);
}

/**
 * Main Recv function - receives metadata and dispatches based on mode
 * Note: This is a periodic task - only logs when actual work is done
 */
void Runtime::Recv(hipc::FullPtr<RecvTask> task, chi::RunContext &rctx) {
  // Get the main server from CHI_IPC (already bound during initialization)
  auto *ipc_manager = CHI_IPC;
  hshm::lbm::Server *lbm_server = ipc_manager->GetMainServer();
  if (!lbm_server) {
    chi::Worker *worker = CHI_CUR_WORKER;
    if (worker) {
      worker->SetTaskDidWork(false);
    }
    return;
  }

  // Note: No socket lock needed - single net worker processes all Recv tasks

  // Receive metadata first to determine mode (non-blocking)
  chi::LoadTaskArchive archive;
  int rc = lbm_server->RecvMetadata(archive);
  if (rc == EAGAIN) {
    // No message available - this is normal for polling, mark as no work done
    chi::Worker *worker = CHI_CUR_WORKER;
    if (worker) {
      worker->SetTaskDidWork(false);
    }
    task->SetReturnCode(0);
    return;
  }

  // Only log when we actually receive a message (not periodic polling noise)
  HLOG(kDebug, "Recv - RecvMetadata returned rc={}", rc);
  if (rc != 0) {
    if (rc == -1) {
      HLOG(kDebug, "Admin: RecvMetadata returned -1 (deserialization issue)");
    } else {
      HLOG(kError, "Admin: Lightbeam RecvMetadata failed with error code {}",
           rc);
    }
    task->SetReturnCode(2);
    return;
  }

  // Log message type - only when we have actual work
  chi::MsgType msg_type = archive.GetMsgType();
  const char *msg_type_str =
      (msg_type == chi::MsgType::kSerializeIn)    ? "SerializeIn"
      : (msg_type == chi::MsgType::kSerializeOut) ? "SerializeOut"
      : (msg_type == chi::MsgType::kHeartbeat)    ? "Heartbeat"
                                                  : "Unknown";
  HLOG(kDebug, "Admin: Received metadata (type: {})", msg_type_str);

  // Dispatch based on message type
  switch (msg_type) {
    case chi::MsgType::kSerializeIn:
      RecvIn(task, archive, lbm_server);
      break;
    case chi::MsgType::kSerializeOut:
      RecvOut(task, archive, lbm_server);
      break;
    case chi::MsgType::kHeartbeat:
      HLOG(kDebug, "Admin: Received heartbeat message");
      task->SetReturnCode(0);
      break;
    default:
      HLOG(kError, "Admin: Unknown message type in Recv");
      task->SetReturnCode(3);
      break;
  }

  (void)rctx;
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