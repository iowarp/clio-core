/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * BSD 3-Clause License. See LICENSE file.
 */

/**
 * BatchManager implementation — see batch_manager.h for the design.
 */

#include "clio_runtime/batch_manager.h"

#include <chrono>
#include <utility>

#include "clio_runtime/container.h"
#include "clio_runtime/ipc_manager.h"
#include "clio_runtime/pool_manager.h"
#include "clio_runtime/pool_query.h"
#include "clio_runtime/singletons.h"
#include "clio_runtime/worker.h"

namespace clio::run {

u64 BatchManager::NowNs() {
  return static_cast<u64>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

void BatchManager::Add(const ctp::ipc::FullPtr<Task> &task) {
  const PoolQuery &q = task->pool_query_;
  GroupKey key{task->pool_id_, task->method_, q.GetContainerHash(),
               q.GetBatchKey()};
  std::lock_guard<std::mutex> lk(mu_);
  Group &group = groups_[key];
  if (group.members.empty()) {
    // First arrival starts the batch window.
    u64 window = q.GetBatchForNs();
    group.deadline_ns = NowNs() + window;
  }
  group.members.push_back(task);
}

bool BatchManager::FlushDue(Worker *worker) {
  std::vector<std::pair<GroupKey, Group>> due;
  bool pending_remains = false;
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (groups_.empty()) {
      return false;
    }
    u64 now = NowNs();
    for (auto it = groups_.begin(); it != groups_.end();) {
      if (now >= it->second.deadline_ns) {
        due.emplace_back(it->first, std::move(it->second));
        it = groups_.erase(it);
      } else {
        ++it;
      }
    }
    pending_remains = !groups_.empty();
  }
  for (auto &[key, group] : due) {
    BuildAndSubmit(worker, key, group);
  }
  // Report busy if we flushed anything or still have groups awaiting their
  // window, so the leader worker keeps polling rather than deep-sleeping.
  return !due.empty() || pending_remains;
}

void BatchManager::BuildAndSubmit(Worker * /*worker*/, const GroupKey &key,
                                  Group &group) {
  if (group.members.empty()) {
    return;
  }
  auto *pool_manager = CLIO_POOL_MANAGER;
  Container *container = pool_manager->GetStaticContainer(key.pool_id);
  if (container == nullptr) {
    HLOG(kError, "BatchManager: no container for pool {} (dropping {} tasks)",
         key.pool_id, group.members.size());
    return;
  }

  // Aggregate task = a copy of the first member; combine the rest in as inputs.
  ctp::ipc::FullPtr<Task> agg =
      container->NewCopyTask(key.method, group.members[0], /*deep=*/true);
  if (agg.IsNull()) {
    HLOG(kError, "BatchManager: NewCopyTask failed for pool {} method {}",
         key.pool_id, key.method);
    return;
  }
  // Combine each remaining member's INPUTS into the aggregate (AggregateIn).
  // Default is a no-op, so without a chimod override the aggregate is simply a
  // copy of the first member (barrier/dedup collective).
  for (size_t i = 1; i < group.members.size(); ++i) {
    container->AggregateIn(key.method, agg, group.members[i]);
  }

  // Run the aggregate locally on the leader as a fresh, self-owned task.
  // NewCopyTask copied the member's task_flags_ but NOT its host_run_ctx_
  // (Task::Copy leaves run ctx per-task). Clear the execution-lifecycle flags
  // the member had accumulated (routed/started/run-ctx-exists) so the submit
  // path runs BeginTask and gives this aggregate its OWN RunContext + future.
  // Without this the aggregate inherits TASK_RUN_CTX_EXISTS with a null run
  // ctx, BeginTask is skipped, and the aggregate never executes (members hang).
  agg->task_id_ = CreateTaskId();
  agg->pool_query_ = PoolQuery::Local();
  agg->SetFlags(TASK_BATCH_AGGREGATE);
  agg->ClearFlags(TASK_ROUTED | TASK_STARTED | TASK_RUN_CTX_EXISTS);

  // Remember the originals to broadcast to when the aggregate completes.
  {
    std::lock_guard<std::mutex> lk(pending_mu_);
    pending_[agg->task_id_.unique_] = std::move(group.members);
  }

  ipc_->Send(agg);
}

bool BatchManager::IsAggregate(const ctp::ipc::FullPtr<Task> &task) const {
  return !task.IsNull() && task->task_flags_.Any(TASK_BATCH_AGGREGATE);
}

void BatchManager::OnAggregateComplete(Worker *worker,
                                       const ctp::ipc::FullPtr<Task> &agg,
                                       RunContext *agg_rctx) {
  std::vector<ctp::ipc::FullPtr<Task>> members;
  {
    std::lock_guard<std::mutex> lk(pending_mu_);
    auto it = pending_.find(agg->task_id_.unique_);
    if (it == pending_.end()) {
      return;
    }
    members = std::move(it->second);
    pending_.erase(it);
  }

  auto *pool_manager = CLIO_POOL_MANAGER;
  Container *container =
      (agg_rctx != nullptr && agg_rctx->container_ != nullptr)
          ? agg_rctx->container_
          : pool_manager->GetStaticContainer(agg->pool_id_);
  if (container == nullptr) {
    HLOG(kError, "BatchManager: no container to broadcast aggregate for pool {}",
         agg->pool_id_);
    return;
  }

  u32 rc = agg->GetReturnCode();
  ContainerId completer = agg->GetCompleter();
  u32 method = agg->method_;

  // Serialize the aggregate's OUT fields once. ManyToOne distributes the single
  // collective result to every member via SerializeOut copy-back (the original
  // spec's `member->SerializeOut(aggregate)`), NOT AggregateOut — that is the
  // replica-gather (N->1) merge. Here it is one result broadcast 1->N.
  chi::priv::vector<char> out_buf(CLIO_PRIV_ALLOC);
  chi::DefaultSaveArchive save_ar(chi::LocalMsgType::kSerializeOut, out_buf);
  container->LocalSaveTask(method, save_ar, agg);

  for (auto &member : members) {
    // Copy the aggregate's OUT into this member (broadcast).
    chi::DefaultLoadArchive load_ar(save_ar.GetMutableData());
    load_ar.Reset(chi::LocalMsgType::kSerializeOut);
    container->LocalLoadTask(member->method_, load_ar, member);
    member->SetReturnCode(rc);
    member->SetCompleter(completer);
    // Complete the original: local future signal or remote SendOut.
    RunContext *m_rctx = member->GetRunCtx();
    if (m_rctx == nullptr) {
      HLOG(kError, "BatchManager: batched task has no RunContext, cannot end");
      continue;
    }
    m_rctx->container_ = container;
    worker->EndTask(member, m_rctx, false);
  }
}

}  // namespace clio::run
