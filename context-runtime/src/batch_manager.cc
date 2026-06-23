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
  const bool count_based = q.IsAllToOneMode();
  std::lock_guard<std::mutex> lk(mu_);
  Group &group = groups_[key];
  if (group.members.empty()) {
    // AllToOne: a count-based barrier — flush only once tasks from every
    // container have arrived (FlushDue checks the count against the pool's
    // container count); no time window. ManyToOne: the first arrival starts the
    // batch window.
    group.count_based = count_based;
    if (!count_based) {
      group.deadline_ns = NowNs() + q.GetBatchForNs();
    }
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
    auto *pool_manager = CLIO_POOL_MANAGER;
    for (auto it = groups_.begin(); it != groups_.end();) {
      bool ready;
      if (it->second.count_based) {
        // AllToOne barrier: ready once a task from every container in the pool
        // has been aggregated. Each member is one container's contribution
        // (the collective contract — one task per container), so the
        // accumulated member count IS the number of aggregations performed;
        // when it reaches the pool's container count, the aggregate runs.
        // num_containers is read live (not at Add) so a transiently-unregistered
        // pool just keeps the group pending until it resolves.
        const PoolInfo *pool_info =
            pool_manager ? pool_manager->GetPoolInfo(it->first.pool_id)
                         : nullptr;
        u32 num_containers =
            (pool_info != nullptr) ? pool_info->num_containers_ : 0;
        ready = (num_containers > 0) &&
                (it->second.members.size() >= num_containers);
      } else {
        // ManyToOne: ready once the time window has elapsed.
        ready = now >= it->second.deadline_ns;
      }
      if (ready) {
        if (in_flight_.count(it->first) != 0) {
          // An aggregate for this key is still running. Leave the group in
          // place so members keep accumulating; it flushes once the in-flight
          // aggregate completes (OnAggregateComplete clears in_flight_).
          ++it;
          continue;
        }
        in_flight_.insert(it->first);  // claim: this group's aggregate is ours
        due.emplace_back(it->first, std::move(it->second));
        it = groups_.erase(it);
      } else {
        ++it;
      }
    }
    // groups_ still holds not-yet-due groups AND in-flight-blocked groups; keep
    // the leader polling so a blocked group flushes promptly once released.
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
    std::lock_guard<std::mutex> lk(mu_);
    in_flight_.erase(key);  // release the claim so future batches can flush
    return;
  }

  // Aggregate task = a copy of the first member; combine the rest in as inputs.
  ctp::ipc::FullPtr<Task> agg =
      container->NewCopyTask(key.method, group.members[0], /*deep=*/true);
  if (agg.IsNull()) {
    HLOG(kError, "BatchManager: NewCopyTask failed for pool {} method {}",
         key.pool_id, key.method);
    std::lock_guard<std::mutex> lk(mu_);
    in_flight_.erase(key);
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

  // Remember the originals to broadcast to when the aggregate completes, and
  // map the aggregate back to its group key so OnAggregateComplete can release
  // the in-flight claim (allowing the next batch for this key to flush).
  {
    std::lock_guard<std::mutex> lk(pending_mu_);
    pending_[agg->task_id_.unique_] = std::move(group.members);
  }
  {
    std::lock_guard<std::mutex> lk(mu_);
    agg_group_[agg->task_id_.unique_] = key;
  }

  ipc_->Send(agg);
}

bool BatchManager::IsAggregate(const ctp::ipc::FullPtr<Task> &task) const {
  return !task.IsNull() && task->task_flags_.Any(TASK_BATCH_AGGREGATE);
}

void BatchManager::OnAggregateComplete(Worker *worker,
                                       const ctp::ipc::FullPtr<Task> &agg,
                                       RunContext *agg_rctx) {
  // Release the in-flight claim for this group key first, so a fresh batch can
  // start now that this aggregate's effects are fully settled. Done before the
  // broadcast below (which only signals the original submitters) so the leader
  // can begin the next batch promptly.
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto git = agg_group_.find(agg->task_id_.unique_);
    if (git != agg_group_.end()) {
      in_flight_.erase(git->second);
      agg_group_.erase(git);
    }
  }

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
  clio::run::priv::vector<char> out_buf(CLIO_PRIV_ALLOC);
  clio::run::DefaultSaveArchive save_ar(clio::run::LocalMsgType::kSerializeOut, out_buf);
  container->LocalSaveTask(method, save_ar, agg);

  for (auto &member : members) {
    // Copy the aggregate's OUT into this member (broadcast).
    clio::run::DefaultLoadArchive load_ar(save_ar.GetMutableData());
    load_ar.Reset(clio::run::LocalMsgType::kSerializeOut);
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
