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

#include <clio_runtime/ipc/ipc_run2run.h>
#include <clio_runtime/ipc_manager.h>
#include <clio_runtime/pool_manager.h>
#include <clio_runtime/config_manager.h>
#include <clio_runtime/worker.h>
#include <clio_runtime/singletons.h>
#include <clio_ctp/lightbeam/transport_factory_impl.h>
#include <clio_runtime/ipc/ipc_cpu2cpu_zmq.h>
#include <clio_ctp/introspect/system_info.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <thread>
#include <unordered_map>

namespace clio::run {

// =============================================================================
// Constructor
// =============================================================================

IpcManagerRun2Run::IpcManagerRun2Run()
    : send_map_(kNumMapBuckets), recv_map_(kNumMapBuckets) {}

// =============================================================================
// SendIn helpers
// =============================================================================

chi::u64 IpcManagerRun2Run::SendInResolveTargetNode(
    chi::IpcManager *ipc_manager,
    chi::PoolManager *pool_manager,
    ctp::ipc::FullPtr<chi::Task> origin_task,
    const chi::PoolQuery &query) {
  if (query.IsLocalMode()) {
    return ipc_manager->GetNodeId();
  }
  if (query.IsDynamicMode()) {
    // A Dynamic query that reaches SendIn was never resolved to a concrete
    // remote target (ScheduleTask/ResolvePoolQuery passed it through), so it
    // belongs on the local node.  Under CLIO_FORCE_NET=1 this is the common
    // case: a local Dynamic task is forced through the loopback network path.
    return ipc_manager->GetNodeId();
  }
  if (query.IsPhysicalMode()) {
    return query.GetNodeId();
  }
  if (query.IsDirectIdMode()) {
    chi::ContainerId container_id = query.GetContainerId();
    return pool_manager->GetContainerNodeId(origin_task->pool_id_, container_id);
  }
  if (query.IsRangeMode()) {
    chi::u32 offset = query.GetRangeOffset();
    chi::ContainerId container_id(offset);
    return pool_manager->GetContainerNodeId(origin_task->pool_id_, container_id);
  }
  if (query.IsBroadcastMode()) {
    HLOG(kError,
         "Admin: Broadcast mode should be handled by TaskDispatcher, not SendIn");
    return kInvalidNodeId;
  }
  if (query.IsDirectHashMode()) {
    HLOG(kError,
         "Admin: DirectHash mode should be handled by TaskDispatcher, not SendIn");
    return kInvalidNodeId;
  }
  HLOG(kError, "Admin: Unsupported query type for SendIn");
  return kInvalidNodeId;
}

void IpcManagerRun2Run::SendInTransmitReplica(
    chi::Container *container,
    chi::IpcManager *ipc_manager,
    ctp::ipc::FullPtr<chi::Task> task_copy,
    chi::u64 target_node_id,
    ctp::ipc::FullPtr<chi::Task> origin_task) {
  auto *config_manager = CLIO_CONFIG_MANAGER;
  const chi::Host *target_host = ipc_manager->GetHost(target_node_id);

  int port = static_cast<int>(config_manager->GetPort());
  ctp::lbm::Transport *lbm_transport =
      ipc_manager->GetOrCreateClient(target_host->ip_address, port);

  if (!lbm_transport) {
    HLOG(kError, "[SendIn] Task {} FAILED: Could not get client for {}:{}",
         origin_task->task_id_, target_host->ip_address, port);
    ipc_manager->SetDead(target_node_id);
    std::lock_guard<std::mutex> _rqlk(retry_queues_mutex_);
    send_in_retry_.push_back(
        {task_copy, target_node_id, std::chrono::steady_clock::now()});
    return;
  }

  chi::SaveTaskArchive archive(chi::MsgType::kSerializeIn, lbm_transport);
  container->SaveTask(task_copy->method_, archive, task_copy);

  ctp::lbm::LbmContext ctx(ctp::lbm::LBM_SYNC);
  int rc = lbm_transport->Send(archive, ctx);

  if (rc != 0) {
    HLOG(kWarning, "[SendIn] Task {} Lightbeam Send rc={} — re-queueing",
         origin_task->task_id_, rc);
    if (rc != EAGAIN) {
      ipc_manager->SetDead(target_node_id);
    }
    std::lock_guard<std::mutex> _rqlk(retry_queues_mutex_);
    send_in_retry_.push_back(
        {task_copy, target_node_id, std::chrono::steady_clock::now()});
    return;
  }
}

// =============================================================================
// SendIn
// =============================================================================

void IpcManagerRun2Run::SendIn(ctp::ipc::FullPtr<chi::Task> origin_task) {
  auto *ipc_manager = CLIO_IPC;
  auto *pool_manager = CLIO_POOL_MANAGER;

  if (origin_task.IsNull()) {
    HLOG(kError, "SendIn: origin_task is null");
    return;
  }

  chi::Container *container =
      pool_manager->GetStaticContainer(origin_task->pool_id_);
  if (container == nullptr) {
    HLOG(kError, "SendIn: container not found for pool_id {}",
         origin_task->pool_id_);
    return;
  }

  size_t send_map_key = size_t(origin_task.ptr_);
  {
    std::lock_guard<std::mutex> lk(send_map_mutex_);
    send_map_[send_map_key] = origin_task;
  }

  if (!origin_task->GetRunCtx()) {
    HLOG(kError, "SendIn: origin_task has no RunContext");
    return;
  }
  chi::RunContext *origin_rctx = origin_task->GetRunCtx();

  const std::vector<chi::PoolQuery> &pool_queries = origin_rctx->pool_queries_;
  size_t num_replicas = pool_queries.size();
  origin_rctx->subtasks_.resize(num_replicas);

  HLOG(kDebug, "[SendIn] Task {} to {} replicas", origin_task->task_id_,
       num_replicas);

  for (size_t i = 0; i < num_replicas; ++i) {
    const chi::PoolQuery &query = pool_queries[i];

    chi::u64 target_node_id =
        SendInResolveTargetNode(ipc_manager, pool_manager, origin_task, query);
    if (target_node_id == kInvalidNodeId) {
      continue;
    }

    const chi::Host *target_host = ipc_manager->GetHost(target_node_id);
    if (!target_host) {
      HLOG(kError, "[SendIn] Task {} FAILED: Host not found for node_id {}",
           origin_task->task_id_, target_node_id);
      continue;
    }

    ctp::ipc::FullPtr<chi::Task> task_copy =
        container->NewCopyTask(origin_task->method_, origin_task, true);
    origin_rctx->subtasks_[i] = task_copy;

    task_copy->task_id_.net_key_ = send_map_key;
    task_copy->task_id_.replica_id_ = i;
    task_copy->pool_query_ = query;
    task_copy->pool_query_.SetReturnNode(ipc_manager->GetNodeId());

    if (!ipc_manager->IsAlive(target_node_id)) {
      float net_timeout = origin_task->pool_query_.GetNetTimeout();
      if (net_timeout >= 0 && net_timeout < 0.001f) {
        HLOG(kWarning,
             "[SendIn] Task {} target node {} is dead, net_timeout=0 -> skip",
             origin_task->task_id_, target_node_id);
        origin_rctx->completed_replicas_++;
        continue;
      }
      HLOG(kWarning,
           "[SendIn] Task {} target node {} is dead, queuing for retry",
           origin_task->task_id_, target_node_id);
      std::lock_guard<std::mutex> _rqlk(retry_queues_mutex_);
      send_in_retry_.push_back(
          {task_copy, target_node_id, std::chrono::steady_clock::now()});
      continue;
    }

    SendInTransmitReplica(container, ipc_manager, task_copy, target_node_id,
                          origin_task);
  }
}

// =============================================================================
// SendOut helpers
// =============================================================================

int IpcManagerRun2Run::SendOutTransmit(
    chi::Container *container,
    chi::IpcManager *ipc_manager,
    ctp::ipc::FullPtr<chi::Task> origin_task,
    chi::u64 target_node_id,
    const chi::Host *target_host) {
  auto *config_manager = CLIO_CONFIG_MANAGER;
  int port = static_cast<int>(config_manager->GetPort());
  ctp::lbm::Transport *lbm_transport =
      ipc_manager->GetOrCreateClient(target_host->ip_address, port);

  if (lbm_transport == nullptr) {
    HLOG(kError, "[SendOut] Task {} FAILED: Could not get client for {}:{}",
         origin_task->task_id_, target_host->ip_address, port);
    ipc_manager->SetDead(target_node_id);
    std::lock_guard<std::mutex> _rqlk(retry_queues_mutex_);
    send_out_retry_.push_back(
        {origin_task, target_node_id, std::chrono::steady_clock::now()});
    return -1;
  }

  chi::SaveTaskArchive archive(chi::MsgType::kSerializeOut, lbm_transport);
  container->SaveTask(origin_task->method_, archive, origin_task);

  ctp::lbm::LbmContext ctx(ctp::lbm::LBM_SYNC);
  int rc = lbm_transport->Send(archive, ctx);

  if (rc != 0) {
    HLOG(kWarning, "[SendOut] Task {} Lightbeam Send rc={} — re-queueing",
         origin_task->task_id_, rc);
    if (rc != EAGAIN) {
      ipc_manager->SetDead(target_node_id);
    }
    std::lock_guard<std::mutex> _rqlk(retry_queues_mutex_);
    send_out_retry_.push_back(
        {origin_task, target_node_id, std::chrono::steady_clock::now()});
    return rc;
  }

  return 0;
}

// =============================================================================
// SendOut
// =============================================================================

void IpcManagerRun2Run::SendOut(ctp::ipc::FullPtr<chi::Task> origin_task) {
  auto *ipc_manager = CLIO_IPC;
  auto *pool_manager = CLIO_POOL_MANAGER;

  if (origin_task.IsNull()) {
    HLOG(kError, "SendOut: origin_task is null");
    return;
  }

  chi::Container *container =
      pool_manager->GetStaticContainer(origin_task->pool_id_);
  if (container == nullptr) {
    HLOG(kError, "SendOut: container not found for pool_id {}",
         origin_task->pool_id_);
    return;
  }

  size_t recv_key = origin_task->task_id_.net_key_ ^
                    (static_cast<size_t>(origin_task->task_id_.replica_id_) *
                     0x9e3779b97f4a7c15ULL);
  {
    std::lock_guard<std::mutex> lk(recv_map_mutex_);
    recv_map_.erase(recv_key);
  }

  chi::u64 target_node_id = origin_task->pool_query_.GetReturnNode();

  if (!ipc_manager->IsAlive(target_node_id)) {
    HLOG(kWarning,
         "[SendOut] Task {} return node {} is dead, queuing for retry",
         origin_task->task_id_, target_node_id);
    std::lock_guard<std::mutex> _rqlk(retry_queues_mutex_);
    send_out_retry_.push_back(
        {origin_task, target_node_id, std::chrono::steady_clock::now()});
    return;
  }

  const chi::Host *target_host = ipc_manager->GetHost(target_node_id);
  if (target_host == nullptr) {
    HLOG(kError, "[SendOut] Task {} FAILED: Host not found for node_id {}",
         origin_task->task_id_, target_node_id);
    return;
  }

  int rc = SendOutTransmit(container, ipc_manager, origin_task, target_node_id,
                           target_host);
  if (rc == 0) {
    container->DelTask(origin_task->method_, origin_task);
  }
}

// =============================================================================
// RecvIn helpers
// =============================================================================

bool IpcManagerRun2Run::RecvInHandleOne(
    chi::IpcManager *ipc_manager,
    chi::PoolManager *pool_manager,
    const chi::TaskInfo &task_info,
    chi::LoadTaskArchive &archive,
    ctp::lbm::Transport *lbm_transport) {
  chi::Container *container =
      pool_manager->GetStaticContainer(task_info.pool_id_);
  if (!container) {
    HLOG(kError, "Admin: Container not found for pool_id {}", task_info.pool_id_);
    return false;
  }

  ctp::ipc::FullPtr<chi::Task> task_ptr =
      container->AllocLoadTask(task_info.method_id_, archive);

  if (task_ptr.IsNull()) {
    HLOG(kError, "Admin: Failed to load task");
    return false;
  }

  chi::u64 sender_node = task_ptr->pool_query_.GetReturnNode();
  if (sender_node != ipc_manager->GetNodeId() &&
      ipc_manager->GetNodeState(sender_node) == chi::NodeState::kDead) {
    HLOG(kInfo, "[RecvIn] Received task from dead node {}, marking alive",
         sender_node);
    FlushStaleStateForNode(sender_node);
    ipc_manager->SetAlive(sender_node);
  }

  chi::u32 set_flags = TASK_REMOTE;
  if (archive.daemon_allocated_bulk_count_ > 0) {
    set_flags |= TASK_DATA_OWNER;
  }
  task_ptr->SetFlags(set_flags);
  task_ptr->ClearFlags(TASK_PERIODIC | TASK_ROUTED | TASK_RUN_CTX_EXISTS |
                       TASK_STARTED);

  size_t recv_key =
      task_ptr->task_id_.net_key_ ^
      (static_cast<size_t>(task_ptr->task_id_.replica_id_) * 0x9e3779b97f4a7c15ULL);
  {
    std::lock_guard<std::mutex> lk(recv_map_mutex_);
    recv_map_[recv_key] = task_ptr;
  }

  HLOG(kDebug, "[RecvIn] Task {} method={} pool_id={} dispatching to workers",
       task_ptr->task_id_, task_ptr->method_, task_ptr->pool_id_);

  chi::Future<chi::Task> future = ipc_manager->MakePointerFuture(task_ptr);
  if (future.GetFutureShm().IsNull()) {
    HLOG(kError, "[RecvIn] MakePointerFuture failed for task {}",
         task_ptr->task_id_);
    return false;
  }
  if (!task_ptr->task_flags_.Any(TASK_RUN_CTX_EXISTS)) {
    ipc_manager->BeginTask(future, container, nullptr);
  }
  task_ptr->SetFlags(TASK_ROUTED);

  if (ipc_manager->GetScheduler() != nullptr) {
    chi::u32 lane_id =
        ipc_manager->GetScheduler()->ClientMapTask(ipc_manager, future);
    auto *worker_queues = ipc_manager->GetTaskQueue();
    if (worker_queues) {
      auto &dest_lane = worker_queues->GetLane(lane_id, 0);
      dest_lane.Push(future);
      // Always signal — see ipc_cpu2cpu_impl.h for the lost-wakeup race the
      // unconditional AwakenWorker closes (the RING_BUFFER_SIGNAL_ON_0
      // was_empty gate dropped wakes under concurrent FUSE-adapter pushes).
      ipc_manager->AwakenWorker(&dest_lane);
    }
  }

  return true;
}

// =============================================================================
// RecvIn
// =============================================================================

int IpcManagerRun2Run::RecvIn(chi::LoadTaskArchive &archive,
                               ctp::lbm::Transport *lbm_transport) {
  auto *ipc_manager = CLIO_IPC;
  auto *pool_manager = CLIO_POOL_MANAGER;

  const auto &task_infos = archive.GetTaskInfos();
  if (task_infos.empty()) {
    return 0;
  }

  for (const auto &task_info : task_infos) {
    RecvInHandleOne(ipc_manager, pool_manager, task_info, archive, lbm_transport);
  }

  return 0;
}

// =============================================================================
// RecvOut helpers
// =============================================================================

int IpcManagerRun2Run::RecvOutDeserialize(
    chi::PoolManager *pool_manager,
    const std::vector<chi::TaskInfo> &task_infos,
    chi::LoadTaskArchive &archive) {
  for (const auto &task_info : task_infos) {
    size_t net_key = task_info.task_id_.net_key_;
    ctp::ipc::FullPtr<chi::Task> origin_task;
    {
      std::lock_guard<std::mutex> lk(send_map_mutex_);
      auto send_it = send_map_.find(net_key);
      if (send_it == nullptr) {
        HLOG(kError,
             "[RecvOut] Task {} FAILED: Origin task not found in send_map "
             "(size: {}) with net_key {}",
             task_info.task_id_, send_map_.size(), net_key);
        return 5;
      }
      origin_task = *send_it;
    }

    if (!origin_task->GetRunCtx()) {
      HLOG(kError, "Admin: origin_task has no RunContext");
      return 6;
    }
    chi::RunContext *origin_rctx = origin_task->GetRunCtx();

    chi::u32 replica_id = task_info.task_id_.replica_id_;
    if (replica_id >= origin_rctx->subtasks_.size()) {
      HLOG(kError, "Admin: Invalid replica_id {} (subtasks size: {})",
           replica_id, origin_rctx->subtasks_.size());
      return 7;
    }

    ctp::ipc::FullPtr<chi::Task> replica = origin_rctx->subtasks_[replica_id];

    chi::Container *container =
        pool_manager->GetStaticContainer(origin_task->pool_id_);
    if (!container) {
      HLOG(kError, "Admin: Container not found for pool_id {}",
           origin_task->pool_id_);
      return 8;
    }

    container->LoadTask(origin_task->method_, archive, replica);
  }

  return 0;
}

int IpcManagerRun2Run::RecvOutAggregate(
    const std::vector<chi::TaskInfo> &task_infos) {
  auto *ipc_manager = CLIO_IPC;

  for (const auto &task_info : task_infos) {
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

    chi::u32 replica_id = task_info.task_id_.replica_id_;
    if (replica_id >= origin_rctx->subtasks_.size()) {
      HLOG(kError, "Admin: Invalid replica_id {} (subtasks size: {})",
           replica_id, origin_rctx->subtasks_.size());
      continue;
    }

    ctp::ipc::FullPtr<chi::Task> replica = origin_rctx->subtasks_[replica_id];
    origin_task->Aggregate(replica);

    HLOG(kDebug, "[RecvOut] Task {}", origin_task->task_id_);

    chi::u32 completed =
        origin_rctx->completed_replicas_.fetch_add(1) + 1;
    if (completed == origin_rctx->subtasks_.size()) {
      RecvOutCompleteOriginTask(net_key, origin_task, origin_rctx);
    }
  }

  return 0;
}

void IpcManagerRun2Run::RecvOutCompleteOriginTask(
    size_t net_key,
    ctp::ipc::FullPtr<chi::Task> origin_task,
    chi::RunContext *origin_rctx) {
  auto *ipc_manager = CLIO_IPC;

  chi::Container *container =
      CLIO_POOL_MANAGER->GetStaticContainer(origin_task->pool_id_);
  if (container != nullptr) {
    origin_rctx->container_ = container;
  }

  for (const auto &subtask_ptr : origin_rctx->subtasks_) {
    subtask_ptr->ClearFlags(TASK_DATA_OWNER);
    if (container != nullptr) {
      container->DelTask(subtask_ptr->method_, subtask_ptr);
    } else {
      HLOG(kError, "[RecvOut] Container not found for pool_id {} while deleting subtask",
           origin_task->pool_id_);
    }
  }
  origin_rctx->subtasks_.clear();

  {
    std::lock_guard<std::mutex> lk(send_map_mutex_);
    send_map_.erase(net_key);
  }

  auto *worker = CLIO_CUR_WORKER;
  if (worker == nullptr) {
    auto *scheduler = ipc_manager->GetScheduler();
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

// =============================================================================
// RecvOut
// =============================================================================

int IpcManagerRun2Run::RecvOut(chi::LoadTaskArchive &archive,
                                ctp::lbm::Transport *lbm_transport) {
  auto *pool_manager = CLIO_POOL_MANAGER;

  const auto &task_infos = archive.GetTaskInfos();
  if (task_infos.empty()) {
    return 0;
  }

  archive.SetTransport(lbm_transport);

  int rc = RecvOutDeserialize(pool_manager, task_infos, archive);
  if (rc != 0) {
    return rc;
  }

  return RecvOutAggregate(task_infos);
}

// =============================================================================
// Retry helpers
// =============================================================================

bool IpcManagerRun2Run::RetrySendToNode(RetryEntry &entry, chi::u64 node_id) {
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

chi::u64 IpcManagerRun2Run::RerouteRetryEntry(RetryEntry &entry) {
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

// =============================================================================
// ProcessRetryQueues
// =============================================================================

void IpcManagerRun2Run::ProcessRetryQueues() {
  auto *ipc_manager = CLIO_IPC;
  auto now = std::chrono::steady_clock::now();

  std::unique_lock<std::mutex> _rqlk(retry_queues_mutex_);

  for (auto it = send_in_retry_.begin(); it != send_in_retry_.end();) {
    float elapsed = std::chrono::duration<float>(now - it->enqueued_at).count();
    float task_timeout = kRun2RunRetryTimeoutSec;
    float task_net_timeout = it->task->pool_query_.GetNetTimeout();
    if (task_net_timeout >= 0) {
      task_timeout = task_net_timeout;
    }

    if (elapsed >= task_timeout) {
      HLOG(kError, "[RetryQueue] SendIn task timed out after {}s for node {}",
           elapsed, it->target_node_id);
      it->task->SetReturnCode(kRun2RunNetworkTimeoutRC);
      it = send_in_retry_.erase(it);
    } else if (ipc_manager->IsAlive(it->target_node_id)) {
      if (RetrySendToNode(*it, it->target_node_id)) {
        HLOG(kInfo, "[RetryQueue] SendIn retry succeeded for node {}",
             it->target_node_id);
        it = send_in_retry_.erase(it);
      } else {
        ++it;
      }
    } else {
      chi::u64 new_node = RerouteRetryEntry(*it);
      if (new_node != 0 && ipc_manager->IsAlive(new_node)) {
        HLOG(kInfo,
             "[RetryQueue] Re-routing task from dead node {} to node {}",
             it->target_node_id, new_node);
        it->target_node_id = new_node;
        if (RetrySendToNode(*it, new_node)) {
          HLOG(kInfo,
               "[RetryQueue] SendIn re-routed retry succeeded for node {}",
               new_node);
          it = send_in_retry_.erase(it);
          continue;
        }
      }
      ++it;
    }
  }

  for (auto it = send_out_retry_.begin(); it != send_out_retry_.end();) {
    float elapsed = std::chrono::duration<float>(now - it->enqueued_at).count();
    float out_task_timeout = kRun2RunRetryTimeoutSec;
    float out_task_net_timeout = it->task->pool_query_.GetNetTimeout();
    if (out_task_net_timeout >= 0) {
      out_task_timeout = out_task_net_timeout;
    }

    if (elapsed >= out_task_timeout) {
      HLOG(kError, "[RetryQueue] SendOut task timed out after {}s for node {}",
           elapsed, it->target_node_id);
      it = send_out_retry_.erase(it);
    } else if (ipc_manager->IsAlive(it->target_node_id)) {
      ctp::ipc::FullPtr<chi::Task> retry_task = it->task;
      it = send_out_retry_.erase(it);
      _rqlk.unlock();
      SendOut(retry_task);
      _rqlk.lock();
      it = send_out_retry_.begin();
    } else {
      ++it;
    }
  }
}

// =============================================================================
// ScanSendMapTimeouts
// =============================================================================

void IpcManagerRun2Run::ScanSendMapTimeouts() {
  auto *ipc_manager = CLIO_IPC;
  auto now = std::chrono::steady_clock::now();

  const auto &dead_nodes = ipc_manager->GetDeadNodes();
  if (dead_nodes.empty()) {
    return;
  }

  std::unordered_map<chi::u64, std::chrono::steady_clock::time_point> dead_map;
  for (const auto &entry : dead_nodes) {
    dead_map[entry.node_id] = entry.detected_at;
  }

  std::vector<std::pair<size_t, ctp::ipc::FullPtr<chi::Task>>> to_complete;
  {
    std::lock_guard<std::mutex> lk(send_map_mutex_);
    send_map_.for_each(
        [&](const size_t &key, ctp::ipc::FullPtr<chi::Task> &origin_task) {
          if (origin_task.IsNull() || !origin_task->GetRunCtx()) {
            return;
          }

          chi::RunContext *rctx = origin_task->GetRunCtx();
          float task_timeout = kRun2RunRetryTimeoutSec;
          float task_net_timeout = origin_task->pool_query_.GetNetTimeout();
          if (task_net_timeout >= 0) {
            task_timeout = task_net_timeout;
          }

          bool any_timed_out = false;
          for (const auto &pq : rctx->pool_queries_) {
            if (!pq.IsPhysicalMode()) {
              continue;
            }
            auto dit = dead_map.find(pq.GetNodeId());
            if (dit == dead_map.end()) {
              continue;
            }
            float dead_elapsed =
                std::chrono::duration<float>(now - dit->second).count();
            if (dead_elapsed >= task_timeout) {
              any_timed_out = true;
              break;
            }
          }

          if (any_timed_out) {
            to_complete.emplace_back(key, origin_task);
          }
        });
  }

  for (auto &entry : to_complete) {
    auto &origin_task = entry.second;
    chi::RunContext *rctx = origin_task->GetRunCtx();
    if (!rctx) {
      continue;
    }
    HLOG(kError,
         "[ScanSendMapTimeouts] Task {} timed out waiting for dead node",
         origin_task->task_id_);
    origin_task->SetReturnCode(kRun2RunNetworkTimeoutRC);
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

// =============================================================================
// FlushStaleStateForNode
// =============================================================================

void IpcManagerRun2Run::FlushStaleStateForNode(chi::u64 node_id) {
  std::lock_guard<std::mutex> _rqlk(retry_queues_mutex_);

  for (auto it = send_in_retry_.begin(); it != send_in_retry_.end();) {
    if (it->target_node_id != node_id) {
      ++it;
      continue;
    }
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
  }

  for (auto it = send_out_retry_.begin(); it != send_out_retry_.end();) {
    if (it->target_node_id != node_id) {
      ++it;
      continue;
    }
    HLOG(kInfo,
         "[FlushStale] Discarding SendOut retry for restarted node {}",
         node_id);
    it = send_out_retry_.erase(it);
  }
}

// =============================================================================
// StartRecvThreads / StopRecvThreads
// =============================================================================

void IpcManagerRun2Run::StartRecvThreads() {
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
      chi::MsgType msg_type = archive.GetMsgType();
      switch (msg_type) {
        case chi::MsgType::kSerializeIn:
          RecvIn(archive, lbm_transport);
          break;
        case chi::MsgType::kSerializeOut:
          RecvOut(archive, lbm_transport);
          break;
        case chi::MsgType::kHeartbeat:
          break;
        default:
          HLOG(kError, "[PeerRecvThread] unknown msg_type={}",
               static_cast<int>(msg_type));
          break;
      }
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
}

void IpcManagerRun2Run::StopRecvThreads() {
  recv_shutdown_.store(true, std::memory_order_release);
  if (peer_recv_thread_.joinable()) peer_recv_thread_.join();
  if (client_recv_thread_.joinable()) client_recv_thread_.join();
}

}  // namespace clio::run
