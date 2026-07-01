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
#include <clio_ctp/thread/thread_model_manager.h>

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

clio::run::u64 IpcManagerRun2Run::SendInResolveTargetNode(
    clio::run::IpcManager *ipc_manager,
    clio::run::PoolManager *pool_manager,
    clio::run::shared_ptr<clio::run::Task> origin_task,
    const clio::run::PoolQuery &query) {
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
    clio::run::ContainerId container_id = query.GetContainerId();
    return pool_manager->GetContainerNodeId(origin_task->pool_id_, container_id);
  }
  if (query.IsRangeMode()) {
    clio::run::u32 offset = query.GetRangeOffset();
    clio::run::ContainerId container_id(offset);
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
    clio::run::IpcManager *ipc_manager,
    clio::run::shared_ptr<clio::run::Task> task_copy,
    clio::run::u64 target_node_id,
    clio::run::shared_ptr<clio::run::Task> origin_task) {
  clio::run::ContainerHold container =
      CLIO_POOL_MANAGER->GetStaticContainer(task_copy->pool_id_).get();
  auto *config_manager = CLIO_CONFIG_MANAGER;
  const clio::run::Host *target_host = ipc_manager->GetHost(target_node_id);

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

  clio::run::SaveTaskArchive archive(clio::run::MsgType::kSerializeIn, lbm_transport);
  // Advertise this node's server port as the response port. The receiver pairs
  // it with our return-node address to open a dedicated dial-back connection
  // for SendOut (see RecvInHandleOne), keyed in the connection cache by
  // return-host + this port.
  archive.client_port_ = static_cast<int>(config_manager->GetPort());
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

void IpcManagerRun2Run::SendIn(clio::run::shared_ptr<clio::run::Task> origin_task) {
  auto *ipc_manager = CLIO_IPC;
  auto *pool_manager = CLIO_POOL_MANAGER;

  if (origin_task.IsNull()) {
    HLOG(kError, "SendIn: origin_task is null");
    return;
  }

  auto container =
      pool_manager->GetStaticContainer(origin_task->pool_id_).get();
  if (container == nullptr) {
    HLOG(kError, "SendIn: container not found for pool_id {}",
         origin_task->pool_id_);
    return;
  }

  size_t send_map_key = size_t(origin_task.get());
  {
    std::lock_guard<std::mutex> lk(send_map_mutex_);
    send_map_[send_map_key] = origin_task;
  }


  const std::vector<clio::run::PoolQuery> &pool_queries =
      origin_task->PoolQueries();
  size_t num_replicas = pool_queries.size();
  origin_task->Subtasks().resize(num_replicas);

  // Per-replica target node, for the #628 task-progress scan. Left as
  // kInvalidNodeId for replicas that were never dispatched to a live/queued
  // node (those never wait on a network response, so the scan skips them).
  std::vector<clio::run::u64> replica_targets(num_replicas, kInvalidNodeId);

  HLOG(kDebug, "[SendIn] Task {} to {} replicas", origin_task->task_id_,
       num_replicas);

  for (size_t i = 0; i < num_replicas; ++i) {
    const clio::run::PoolQuery &query = pool_queries[i];

    clio::run::u64 target_node_id =
        SendInResolveTargetNode(ipc_manager, pool_manager, origin_task, query);
    if (target_node_id == kInvalidNodeId) {
      continue;
    }

    const clio::run::Host *target_host = ipc_manager->GetHost(target_node_id);
    if (!target_host) {
      HLOG(kError, "[SendIn] Task {} FAILED: Host not found for node_id {}",
           origin_task->task_id_, target_node_id);
      continue;
    }

    clio::run::shared_ptr<clio::run::Task> task_copy =
        container->NewCopyTask(origin_task->method_, origin_task, true);
    origin_task->Subtasks()[i] = task_copy;

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
        origin_task->CompletedReplicas()++;
        continue;
      }
      HLOG(kWarning,
           "[SendIn] Task {} target node {} is dead, queuing for retry",
           origin_task->task_id_, target_node_id);
      replica_targets[i] = target_node_id;
      std::lock_guard<std::mutex> _rqlk(retry_queues_mutex_);
      send_in_retry_.push_back(
          {task_copy, target_node_id, std::chrono::steady_clock::now()});
      continue;
    }

    replica_targets[i] = target_node_id;
    SendInTransmitReplica(ipc_manager, task_copy,
                          target_node_id, origin_task);
  }

  // Register this origin for the #628 task-progress scan. Admin-pool tasks are
  // excluded: QueryTaskProgress is itself an admin cross-node task, so tracking
  // it (and the other admin liveness/control probes) would recurse.
  if (!(origin_task->pool_id_ == clio::run::kAdminPoolId)) {
    RegisterOriginProgress(send_map_key, replica_targets);
  }
}

// =============================================================================
// SendOut helpers
// =============================================================================

int IpcManagerRun2Run::SendOutTransmit(
    clio::run::IpcManager *ipc_manager,
    clio::run::shared_ptr<clio::run::Task> origin_task,
    clio::run::u64 target_node_id,
    const clio::run::Host *target_host) {
  clio::run::ContainerHold container =
      CLIO_POOL_MANAGER->GetStaticContainer(origin_task->pool_id_).get();
  auto *config_manager = CLIO_CONFIG_MANAGER;
  int port = static_cast<int>(config_manager->GetPort());

  // Prefer the dedicated dial-back connection resolved at RecvIn (stored on the
  // task's FutureShm). Fall back to resolving the peer connection by address if
  // it is absent (e.g. a legacy sender that advertised no client_port_, or a
  // retry whose FutureShm was already freed).
  ctp::lbm::Transport *lbm_transport = nullptr;
  {
    ctp::ipc::FullPtr<clio::run::RunContext> fshm =
        origin_task->RunFuture().GetFutureShm();
    if (!fshm.IsNull() && fshm->response_transport_ != nullptr) {
      lbm_transport = fshm->response_transport_;
    }
  }
  if (lbm_transport == nullptr) {
    lbm_transport = ipc_manager->GetOrCreateClient(target_host->ip_address, port);
  }

  if (lbm_transport == nullptr) {
    HLOG(kError, "[SendOut] Task {} FAILED: Could not get client for {}:{}",
         origin_task->task_id_, target_host->ip_address, port);
    ipc_manager->SetDead(target_node_id);
    std::lock_guard<std::mutex> _rqlk(retry_queues_mutex_);
    send_out_retry_.push_back(
        {origin_task, target_node_id, std::chrono::steady_clock::now()});
    return -1;
  }

  clio::run::SaveTaskArchive archive(clio::run::MsgType::kSerializeOut, lbm_transport);
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

void IpcManagerRun2Run::SendOut(clio::run::shared_ptr<clio::run::Task> origin_task) {
  auto *ipc_manager = CLIO_IPC;

  if (origin_task.IsNull()) {
    HLOG(kError, "SendOut: origin_task is null");
    return;
  }

  // The recv-side FutureShm is owned by the Future's shared_ptr (created in
  // RecvInHandleOne and stored in this task's RunContext). It is freed
  // automatically when the RunContext (and its Future copy) is destroyed by
  // DelTask below — no manual capture/FreeBuffer needed.
  size_t recv_key = origin_task->task_id_.net_key_ ^
                    (static_cast<size_t>(origin_task->task_id_.replica_id_) *
                     0x9e3779b97f4a7c15ULL);
  {
    std::lock_guard<std::mutex> lk(recv_map_mutex_);
    recv_map_.erase(recv_key);
  }

  clio::run::u64 target_node_id = origin_task->pool_query_.GetReturnNode();

  if (!ipc_manager->IsAlive(target_node_id)) {
    HLOG(kWarning,
         "[SendOut] Task {} return node {} is dead, queuing for retry",
         origin_task->task_id_, target_node_id);
    std::lock_guard<std::mutex> _rqlk(retry_queues_mutex_);
    send_out_retry_.push_back(
        {origin_task, target_node_id, std::chrono::steady_clock::now()});
    return;
  }

  const clio::run::Host *target_host = ipc_manager->GetHost(target_node_id);
  if (target_host == nullptr) {
    HLOG(kError, "[SendOut] Task {} FAILED: Host not found for node_id {}",
         origin_task->task_id_, target_node_id);
    return;
  }

  int rc = SendOutTransmit(ipc_manager, origin_task,
                           target_node_id, target_host);
  (void)rc;
  // Task frees via RAII when its shared_ptr owners drop (the by-value
  // origin_task handle here, plus the RunContext/send_map_ entry) — no
  // explicit DelTask.
}

// =============================================================================
// RecvIn helpers
// =============================================================================

bool IpcManagerRun2Run::RecvInHandleOne(
    clio::run::IpcManager *ipc_manager,
    clio::run::PoolManager *pool_manager,
    const clio::run::TaskInfo &task_info,
    clio::run::LoadTaskArchive &archive,
    ctp::lbm::Transport *lbm_transport) {
  auto container =
      pool_manager->GetStaticContainer(task_info.pool_id_).get();
  if (!container) {
    HLOG(kError, "Admin: Container not found for pool_id {}", task_info.pool_id_);
    return false;
  }

  clio::run::shared_ptr<clio::run::Task> task_ptr =
      container->AllocLoadTask(task_info.method_id_, archive);

  if (task_ptr.IsNull()) {
    HLOG(kError, "Admin: Failed to load task");
    return false;
  }

  clio::run::u64 sender_node = task_ptr->pool_query_.GetReturnNode();
  if (sender_node != ipc_manager->GetNodeId() &&
      ipc_manager->GetNodeState(sender_node) == clio::run::NodeState::kDead) {
    HLOG(kInfo, "[RecvIn] Received task from dead node {}, marking alive",
         sender_node);
    FlushStaleStateForNode(sender_node);
    ipc_manager->SetAlive(sender_node);
  }

  clio::run::u32 set_flags = TASK_REMOTE;
  if (archive.daemon_allocated_bulk_count_ > 0) {
    set_flags |= TASK_DATA_OWNER;
  }
  task_ptr->SetFlags(set_flags);
  // routed_/started_ are fresh per-RunContext (BeginTask below); only the
  // serialized TASK_PERIODIC flag needs resetting on receive.
  task_ptr->ClearFlags(TASK_PERIODIC);

  size_t recv_key =
      task_ptr->task_id_.net_key_ ^
      (static_cast<size_t>(task_ptr->task_id_.replica_id_) * 0x9e3779b97f4a7c15ULL);
  {
    std::lock_guard<std::mutex> lk(recv_map_mutex_);
    recv_map_[recv_key] = task_ptr;
  }

  HLOG(kDebug, "[RecvIn] Task {} method={} pool_id={} dispatching to workers",
       task_ptr->task_id_, task_ptr->method_, task_ptr->pool_id_);

  clio::run::Future<clio::run::Task> future(task_ptr->pool_id_,
                                            task_ptr->method_, task_ptr);
  if (future.GetFutureShm().IsNull()) {
    HLOG(kError, "[RecvIn] Future construction failed for task {}",
         task_ptr->task_id_);
    return false;
  }

  // NOTE: the response connection is intentionally NOT opened here. A ZMQ socket
  // must be created and used by the same thread; this RecvIn runs on the recv
  // thread, but the response is sent by net_send_worker_ in SendOutTransmit.
  // Opening the dial-back DEALER here and sending on it there is exactly the
  // cross-thread socket sharing that forced sock_mtx_ and wedged force_net.
  // SendOut re-resolves the return node (pool_query_.GetReturnNode()) and
  // GetOrCreateClient's the connection on the send worker — same cache, same
  // endpoint (return-host:cluster-port), but single-threaded socket ownership.

  // Allocate the task's RunContext (and resolve its container) now that it is
  // deserialized, so RouteTask / the worker have an active RunContext.
  future.GetTaskPtr()->BeginRunContext();
  task_ptr->SetRouted();

  if (ipc_manager->GetScheduler() != nullptr) {
    clio::run::u32 lane_id =
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

int IpcManagerRun2Run::RecvIn(clio::run::LoadTaskArchive &archive,
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
    clio::run::PoolManager *pool_manager,
    const std::vector<clio::run::TaskInfo> &task_infos,
    clio::run::LoadTaskArchive &archive) {
  for (const auto &task_info : task_infos) {
    size_t net_key = task_info.task_id_.net_key_;
    clio::run::shared_ptr<clio::run::Task> origin_task;
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


    clio::run::u32 replica_id = task_info.task_id_.replica_id_;
    if (replica_id >= origin_task->Subtasks().size()) {
      HLOG(kError, "Admin: Invalid replica_id {} (subtasks size: {})",
           replica_id, origin_task->Subtasks().size());
      return 7;
    }

    clio::run::shared_ptr<clio::run::Task> replica =
        origin_task->Subtasks()[replica_id];

    auto container =
        pool_manager->GetStaticContainer(origin_task->pool_id_).get();
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
    const std::vector<clio::run::TaskInfo> &task_infos) {
  auto *ipc_manager = CLIO_IPC;

  for (const auto &task_info : task_infos) {
    size_t net_key = task_info.task_id_.net_key_;
    clio::run::shared_ptr<clio::run::Task> origin_task;
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


    clio::run::u32 replica_id = task_info.task_id_.replica_id_;
    if (replica_id >= origin_task->Subtasks().size()) {
      HLOG(kError, "Admin: Invalid replica_id {} (subtasks size: {})",
           replica_id, origin_task->Subtasks().size());
      continue;
    }

    clio::run::shared_ptr<clio::run::Task> replica =
        origin_task->Subtasks()[replica_id];

    // AggregateOut via the Container so the concrete task type's AggregateOut()
    // runs (which merges OUT fields like bdev's blocks_). Task::AggregateOut is
    // intentionally non-virtual to keep Task vtable-free, so calling
    // origin_task->AggregateOut(replica) here would slice to the base and drop
    // every derived OUT field — leaving callers with empty results.
    auto container =
        CLIO_POOL_MANAGER->GetStaticContainer(origin_task->pool_id_).get();
    if (!container) {
      HLOG(kError, "[RecvOut] Container not found for pool_id {}",
           origin_task->pool_id_);
      continue;
    }
    container->AggregateOut(origin_task->method_, origin_task, replica);

    HLOG(kDebug, "[RecvOut] Task {}", origin_task->task_id_);

    // Count toward completion only if this replica was not already accounted
    // for by the #628 progress scan (a Gone verdict that raced this response);
    // untracked (admin) origins always count, preserving prior behaviour.
    if (!MarkReplicaAccounted(net_key, replica_id)) {
      continue;
    }

    clio::run::u32 completed =
        origin_task->CompletedReplicas().fetch_add(1) + 1;
    if (completed == origin_task->Subtasks().size()) {
      RecvOutCompleteOriginTask(net_key, origin_task);
    }
  }

  return 0;
}

void IpcManagerRun2Run::RecvOutCompleteOriginTask(
    size_t net_key,
    clio::run::shared_ptr<clio::run::Task> origin_task) {
  auto *ipc_manager = CLIO_IPC;

  clio::run::DynamicContainer container =
      CLIO_POOL_MANAGER->GetStaticContainer(origin_task->pool_id_);
  if (container) {
    origin_task->ExecContainer() = container;
  }

  for (const auto &subtask_ptr : origin_task->Subtasks()) {
    subtask_ptr->ClearFlags(TASK_DATA_OWNER);
  }
  // Clearing the vector drops the last shared_ptr owner of each replica subtask,
  // freeing them via RAII (replaces the former per-subtask DelTask).
  origin_task->Subtasks().clear();

  {
    std::lock_guard<std::mutex> lk(send_map_mutex_);
    send_map_.erase(net_key);
    progress_map_.erase(net_key);  // #628: drop task-progress tracking
  }

  auto *worker = CLIO_CUR_WORKER;
  if (worker == nullptr) {
    auto *scheduler = ipc_manager->GetScheduler();
    worker = scheduler ? scheduler->GetNetRecvWorker() : nullptr;
  }
  if (worker) {
    worker->EndTask(origin_task, true);
  } else {
    HLOG(kError,
         "[RecvOut] No worker available to call EndTask for task {}",
         origin_task->task_id_);
  }
}

// =============================================================================
// Cross-node task-progress tracking (issue #628)
// =============================================================================

void IpcManagerRun2Run::RegisterOriginProgress(
    size_t net_key, const std::vector<clio::run::u64> &replica_targets) {
  OriginProgress prog;
  prog.enqueue_time = std::chrono::steady_clock::now();
  prog.replicas.resize(replica_targets.size());
  for (size_t i = 0; i < replica_targets.size(); ++i) {
    prog.replicas[i].target_node_id = replica_targets[i];
    // A replica never dispatched to a node has nothing to wait on -> accounted,
    // so the scan skips it and it never blocks completion.
    prog.replicas[i].accounted = (replica_targets[i] == kInvalidNodeId);
  }
  std::lock_guard<std::mutex> lk(send_map_mutex_);
  progress_map_[net_key] = std::move(prog);
}

bool IpcManagerRun2Run::MarkReplicaAccounted(size_t net_key,
                                             clio::run::u32 replica_id) {
  std::lock_guard<std::mutex> lk(send_map_mutex_);
  auto it = progress_map_.find(net_key);
  if (it == progress_map_.end()) {
    return true;  // untracked (admin) origin -> caller counts unconditionally
  }
  if (replica_id >= it->second.replicas.size()) {
    return true;  // defensive: out-of-range, don't suppress the count
  }
  if (it->second.replicas[replica_id].accounted) {
    return false;  // already accounted -> caller must not double-count
  }
  it->second.replicas[replica_id].accounted = true;
  return true;
}

std::vector<StuckReplica> IpcManagerRun2Run::CollectStuckReplicas(
    clio::run::u32 interval_ms) {
  std::vector<StuckReplica> stuck;
  if (interval_ms == 0) {
    return stuck;  // periodic validity check disabled
  }
  auto *ipc_manager = CLIO_IPC;
  auto now = std::chrono::steady_clock::now();
  auto interval = std::chrono::milliseconds(interval_ms);

  std::lock_guard<std::mutex> lk(send_map_mutex_);
  // Throttle: run a scan pass at most once per interval.
  if (last_progress_scan_.time_since_epoch().count() != 0 &&
      now - last_progress_scan_ < interval) {
    return stuck;
  }
  last_progress_scan_ = now;

  for (auto &kv : progress_map_) {
    OriginProgress &prog = kv.second;
    if (now - prog.enqueue_time < interval) {
      continue;  // give the task at least one interval before probing
    }
    for (clio::run::u32 rid = 0; rid < prog.replicas.size(); ++rid) {
      ReplicaProgress &rp = prog.replicas[rid];
      if (rp.accounted || rp.target_node_id == kInvalidNodeId) {
        continue;
      }
      // A dead target is handled by the dead-node timeout path; only probe
      // nodes that are (still / again) alive.
      if (!ipc_manager->IsAlive(rp.target_node_id)) {
        continue;
      }
      stuck.push_back({static_cast<clio::run::u64>(kv.first), rid,
                       rp.target_node_id});
    }
  }
  return stuck;
}

void IpcManagerRun2Run::HandleTaskProgressResult(clio::run::u64 net_key,
                                                 clio::run::u32 replica_id,
                                                 bool gone) {
  if (!gone) {
    return;  // still running on its node -> keep waiting
  }
  // Claim the accounting transition; bail if a real response already took it.
  if (!MarkReplicaAccounted(static_cast<size_t>(net_key), replica_id)) {
    return;
  }
  clio::run::shared_ptr<clio::run::Task> origin_task;
  {
    std::lock_guard<std::mutex> lk(send_map_mutex_);
    auto sit = send_map_.find(static_cast<size_t>(net_key));
    if (sit == nullptr) {
      return;  // origin already completed/erased
    }
    origin_task = *sit;
  }
  HLOG(kWarning,
       "[TaskProgress] replica {} of task {} is Gone on its node; completing "
       "origin with network-timeout RC (#628)",
       replica_id, origin_task->task_id_);
  // Preserve partial results already aggregated from replicas that answered;
  // signal the shortfall with a network-timeout RC.
  origin_task->SetReturnCode(kRun2RunNetworkTimeoutRC);
  clio::run::u32 completed =
      origin_task->CompletedReplicas().fetch_add(1) + 1;
  if (completed == origin_task->Subtasks().size()) {
    RecvOutCompleteOriginTask(static_cast<size_t>(net_key), origin_task);
  }
}

// =============================================================================
// RecvOut
// =============================================================================

int IpcManagerRun2Run::RecvOut(clio::run::LoadTaskArchive &archive,
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

bool IpcManagerRun2Run::RetrySendToNode(RetryEntry &entry, clio::run::u64 node_id) {
  auto *ipc_manager = CLIO_IPC;
  auto *pool_manager = CLIO_POOL_MANAGER;
  auto *config_manager = CLIO_CONFIG_MANAGER;

  const clio::run::Host *target_host = ipc_manager->GetHost(node_id);
  if (!target_host) {
    return false;
  }

  int port = static_cast<int>(config_manager->GetPort());
  ctp::lbm::Transport *lbm_transport =
      ipc_manager->GetOrCreateClient(target_host->ip_address, port);
  if (!lbm_transport) {
    return false;
  }

  auto container =
      pool_manager->GetStaticContainer(entry.task->pool_id_).get();
  if (!container) {
    return false;
  }

  clio::run::SaveTaskArchive archive(clio::run::MsgType::kSerializeIn, lbm_transport);
  container->SaveTask(entry.task->method_, archive, entry.task);
  ctp::lbm::LbmContext ctx(0);
  int rc = lbm_transport->Send(archive, ctx);
  return rc == 0;
}

clio::run::u64 IpcManagerRun2Run::RerouteRetryEntry(RetryEntry &entry) {
  auto *pool_manager = CLIO_POOL_MANAGER;
  const clio::run::PoolQuery &query = entry.task->pool_query_;

  if (query.IsDirectIdMode()) {
    clio::run::ContainerId container_id = query.GetContainerId();
    clio::run::u32 new_node =
        pool_manager->GetContainerNodeId(entry.task->pool_id_, container_id);
    if (new_node != 0 && new_node != entry.target_node_id) {
      return new_node;
    }
  } else if (query.IsRangeMode()) {
    clio::run::u32 offset = query.GetRangeOffset();
    clio::run::ContainerId container_id(offset);
    clio::run::u32 new_node =
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
      clio::run::u64 new_node = RerouteRetryEntry(*it);
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
      clio::run::shared_ptr<clio::run::Task> retry_task = it->task;
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

  std::unordered_map<clio::run::u64, std::chrono::steady_clock::time_point> dead_map;
  for (const auto &entry : dead_nodes) {
    dead_map[entry.node_id] = entry.detected_at;
  }

  std::vector<std::pair<size_t, clio::run::shared_ptr<clio::run::Task>>> to_complete;
  {
    std::lock_guard<std::mutex> lk(send_map_mutex_);
    send_map_.for_each(
        [&](const size_t &key, clio::run::shared_ptr<clio::run::Task> &origin_task) {
          if (origin_task.IsNull()) {
            return;
          }

          float task_timeout = kRun2RunRetryTimeoutSec;
          float task_net_timeout = origin_task->pool_query_.GetNetTimeout();
          if (task_net_timeout >= 0) {
            task_timeout = task_net_timeout;
          }

          bool any_timed_out = false;
          for (const auto &pq : origin_task->PoolQueries()) {
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
    HLOG(kError,
         "[ScanSendMapTimeouts] Task {} timed out waiting for dead node",
         origin_task->task_id_);
    origin_task->SetReturnCode(kRun2RunNetworkTimeoutRC);
    // Fix B (#628): this runs on the net worker's periodic tick, where
    // CLIO_CUR_WORKER can be null. Fall back to the net-recv worker rather than
    // dereferencing null (mirrors RecvOutCompleteOriginTask).
    auto *worker = CLIO_CUR_WORKER;
    if (worker == nullptr) {
      auto *scheduler = CLIO_IPC->GetScheduler();
      worker = scheduler ? scheduler->GetNetRecvWorker() : nullptr;
    }
    if (worker != nullptr) {
      worker->EndTask(origin_task, true);
    } else {
      HLOG(kError,
           "[ScanSendMapTimeouts] No worker available to complete task {}",
           origin_task->task_id_);
    }
  }

  if (!to_complete.empty()) {
    std::lock_guard<std::mutex> lk(send_map_mutex_);
    for (auto &entry : to_complete) {
      send_map_.erase(entry.first);
      progress_map_.erase(entry.first);  // #628: drop task-progress tracking
    }
  }
}

// =============================================================================
// FlushStaleStateForNode
// =============================================================================

void IpcManagerRun2Run::FlushStaleStateForNode(clio::run::u64 node_id) {
  std::lock_guard<std::mutex> _rqlk(retry_queues_mutex_);

  for (auto it = send_in_retry_.begin(); it != send_in_retry_.end();) {
    if (it->target_node_id != node_id) {
      ++it;
      continue;
    }
    size_t net_key = it->task->task_id_.net_key_;
    clio::run::shared_ptr<clio::run::Task> origin;
    {
      std::lock_guard<std::mutex> lk(send_map_mutex_);
      auto send_it = send_map_.find(net_key);
      if (send_it != nullptr) {
        origin = *send_it;
      }
    }
    if (!origin.IsNull()) {
      origin->CompletedReplicas()++;
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

// How long a recv thread blocks in EventManager::Wait() before re-checking
// recv_shutdown_ and re-draining. The fd-readability signal (epoll /
// WSAEventSelect) wakes Wait() immediately when data arrives, so this only
// bounds idle re-poll and shutdown-join latency; it is NOT added to
// message-delivery latency. Mirrors the established RecvZmqClientThread wait.
static constexpr int kRecvWaitTimeoutUs = 100;

// Milliseconds a blocking zmq_poll (Transport::PollRecv) waits before returning
// to re-check recv_shutdown_. zmq_poll wakes immediately on data, so this only
// bounds idle re-poll + shutdown latency. ZMQ transports use this instead of an
// EventManager because ZMQ_FD cannot be watched with WSAEventSelect on Windows.
static constexpr int kZmqPollTimeoutMs = 1;

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
      CTP_THREAD_MODEL->SleepForUs(1000);
    }
    if (!lbm_transport) {
      HLOG(kError, "[PeerRecvThread] main transport never appeared");
      return;
    }
    // Block on transport readability instead of spin-polling. main_transport_
    // is a ZMQ transport, so we use its native zmq_poll (Transport::PollRecv)
    // rather than an EventManager: ZMQ_FD can't be registered with
    // WSAEventSelect on Windows. Each wake drains every pending message before
    // blocking again.
    HLOG(kInfo, "[PeerRecvThread] started");
    while (!recv_shutdown_.load(std::memory_order_acquire)) {
      bool drained_any = false;
      while (!recv_shutdown_.load(std::memory_order_acquire)) {
        clio::run::LoadTaskArchive archive;
        auto info = lbm_transport->Recv(archive);
        int rc = info.rc;
        if (rc != 0) {
          // EAGAIN (nothing left) or a peer-closed/transient error ends the
          // drain. rc == -1 is the routine peer-closed case (SocketTransport
          // already cleaned up the fd inside Recv()).
          if (rc != EAGAIN && rc != -1) {
            HLOG(kError, "[PeerRecvThread] Recv failed rc={}", rc);
          }
          break;
        }
        drained_any = true;
        clio::run::MsgType msg_type = archive.GetMsgType();
        switch (msg_type) {
          case clio::run::MsgType::kSerializeIn:
            RecvIn(archive, lbm_transport);
            break;
          case clio::run::MsgType::kSerializeOut:
            RecvOut(archive, lbm_transport);
            break;
          case clio::run::MsgType::kHeartbeat:
            break;
          default:
            HLOG(kError, "[PeerRecvThread] unknown msg_type={}",
                 static_cast<int>(msg_type));
            break;
        }
        lbm_transport->ClearRecvHandles(archive);
      }
      if (!drained_any && !recv_shutdown_.load(std::memory_order_acquire)) {
        lbm_transport->PollRecv(kZmqPollTimeoutMs);
      }
    }
    HLOG(kInfo, "[PeerRecvThread] shutting down");
  });

  // Dedicated single client recv thread: drains TCP (port 9416) and IPC
  // (unix socket) client transports via IpcCpu2CpuZmq::RecvIn.
  client_recv_thread_ = std::thread([this]() {
    ctp::SystemInfo::SetCurrentThreadName("chi-client-recv");
    auto *ipc_manager = CLIO_IPC;
    // Block instead of spin-polling. The two client transports use different
    // readiness primitives: the IPC (unix-socket) transport registers with our
    // EventManager (epoll / WSAEventSelect both work); the TCP transport is ZMQ,
    // whose ZMQ_FD can't be watched with WSAEventSelect on Windows, so it blocks
    // via its native zmq_poll (Transport::PollRecv). The transports are created
    // during IpcManager init, which may race this thread's start.
    ctp::lbm::Transport *tcp_transport = nullptr;
    ctp::lbm::Transport *ipc_transport = nullptr;
    for (int spin = 0; spin < 1000 && !recv_shutdown_.load(); ++spin) {
      tcp_transport = ipc_manager->GetClientTransport(clio::run::IpcMode::kTcp);
      ipc_transport = ipc_manager->GetClientTransport(clio::run::IpcMode::kIpc);
      if (tcp_transport || ipc_transport) break;
      CTP_THREAD_MODEL->SleepForUs(1000);
    }
    if (ipc_transport) ipc_transport->RegisterEventManager(client_recv_em_);
    HLOG(kInfo, "[ClientRecvThread] started");
    while (!recv_shutdown_.load(std::memory_order_acquire)) {
      clio::run::u32 tasks_received = 0;
      bool did_work = clio::run::IpcCpu2CpuZmq::RecvIn(ipc_manager,
                                                       tasks_received);
      if (!did_work && !recv_shutdown_.load(std::memory_order_acquire)) {
        // Nothing drained: block on the IPC socket via the EventManager (its
        // bounded timeout also re-polls the TCP transport). If only TCP exists,
        // block on its native zmq_poll instead so we don't hit the EventManager's
        // tick-floored empty-handle Sleep fallback on Windows.
        if (ipc_transport) {
          client_recv_em_.Wait(kRecvWaitTimeoutUs);
        } else if (tcp_transport) {
          tcp_transport->PollRecv(kZmqPollTimeoutMs);
        }
      }
    }
    if (ipc_transport) ipc_transport->UnregisterEventManager();
    HLOG(kInfo, "[ClientRecvThread] shutting down");
  });
}

void IpcManagerRun2Run::StopRecvThreads() {
  recv_shutdown_.store(true, std::memory_order_release);
  if (peer_recv_thread_.joinable()) peer_recv_thread_.join();
  if (client_recv_thread_.joinable()) client_recv_thread_.join();
}

}  // namespace clio::run
