/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * BSD 3-Clause License. See LICENSE file.
 */

#ifndef CLIO_RUNTIME_INCLUDE_BATCH_MANAGER_H_
#define CLIO_RUNTIME_INCLUDE_BATCH_MANAGER_H_

#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "clio_runtime/task.h"
#include "clio_runtime/types.h"

namespace clio::run {

class IpcManager;
class Worker;

/**
 * BatchManager — leader-side collective batching + aggregation for the
 * PoolQuery::ManyToOne routing mode.
 *
 * When a ManyToOne task reaches its neighborhood leader, it is parked here
 * (instead of being executed) and grouped by
 * (pool_id, method, container_hash, batch_key). After the batch window
 * (batch_for_ns, measured from the first arrival) elapses, the group is
 * flushed:
 *   1. A synthetic aggregate task is built as a copy of the first member,
 *      and each remaining member's INPUTS are combined in via
 *      Container::AggregateIn (the collective input combine).
 *   2. The aggregate task runs once (flagged TASK_BATCH_AGGREGATE).
 *   3. On completion (Worker::EndTask), the aggregate's OUT is broadcast back
 *      into every original task via a SerializeOut copy (LocalSaveTask/
 *      LocalLoadTask, kSerializeOut) — one result distributed 1->N, distinct
 *      from AggregateOut which is the replica-gather N->1 merge — and each
 *      original is completed through Worker::EndTask (local future signal or
 *      remote SendOut to its return node).
 */
class BatchManager {
 public:
  explicit BatchManager(IpcManager *ipc) : ipc_(ipc) {}

  /**
   * Park a ManyToOne task into its batch group. Called on the neighborhood
   * leader from IpcManager::RouteTask. The task already has a RunContext and
   * a future; both are reachable via task->GetRunCtx().
   */
  void Add(const ctp::ipc::FullPtr<Task> &task);

  /**
   * Flush every group whose batch window has elapsed. Called periodically from
   * a single worker's run loop on the leader. Safe to call when empty.
   * @return true if any group is pending or was flushed (so the caller keeps
   *         spinning to honor the short batch window instead of deep-sleeping).
   */
  bool FlushDue(Worker *worker);

  /** True if `task` is a synthetic aggregate task produced by FlushDue. */
  bool IsAggregate(const ctp::ipc::FullPtr<Task> &task) const;

  /**
   * Broadcast a completed aggregate's OUT to its batched originals and
   * complete each one. Called from Worker::EndTask for aggregate tasks.
   */
  void OnAggregateComplete(Worker *worker, const ctp::ipc::FullPtr<Task> &agg,
                           RunContext *agg_rctx);

 private:
  /** Group identity: matching tasks aggregate together. */
  struct GroupKey {
    PoolId pool_id;
    u32 method;
    u32 container_hash;
    u64 batch_key;
    bool operator==(const GroupKey &o) const {
      return pool_id == o.pool_id && method == o.method &&
             container_hash == o.container_hash && batch_key == o.batch_key;
    }
  };
  struct GroupKeyHash {
    size_t operator()(const GroupKey &k) const {
      size_t h = std::hash<u64>()(
          (static_cast<u64>(k.pool_id.major_) << 32) | k.pool_id.minor_);
      h ^= std::hash<u32>()(k.method) + 0x9e3779b97f4a7c15ULL + (h << 6) +
           (h >> 2);
      h ^= std::hash<u32>()(k.container_hash) + 0x9e3779b97f4a7c15ULL +
           (h << 6) + (h >> 2);
      h ^= std::hash<u64>()(k.batch_key) + 0x9e3779b97f4a7c15ULL + (h << 6) +
           (h >> 2);
      return h;
    }
  };
  struct Group {
    std::vector<ctp::ipc::FullPtr<Task>> members;
    u64 deadline_ns = 0;  /**< steady-clock ns of the first arrival + window */
    // AllToOne (count-based barrier): the aggregate does not run until tasks
    // from every container in the pool have arrived. When set, FlushDue ignores
    // deadline_ns and instead flushes once the accumulated member count reaches
    // the pool's container count. ManyToOne leaves this false (time-windowed).
    bool count_based = false;
  };

  /** Monotonic nanoseconds (steady clock). */
  static u64 NowNs();

  /** Build the aggregate for a flushed group and submit it to run. */
  void BuildAndSubmit(Worker *worker, const GroupKey &key, Group &group);

  IpcManager *ipc_;
  std::mutex mu_;
  std::unordered_map<GroupKey, Group, GroupKeyHash> groups_;
  // At most one aggregate per group key may be in flight at a time. While a
  // group's aggregate is running, new members keep accumulating in groups_ and
  // are NOT flushed until that aggregate completes (OnAggregateComplete clears
  // the key). This serializes collectives sharing a (pool,method,hash,key) so
  // e.g. each runs against the fully-settled output of the previous one.
  std::unordered_set<GroupKey, GroupKeyHash> in_flight_;
  /** in-flight aggregate task unique id → its group key (to release on done). */
  std::unordered_map<u64, GroupKey> agg_group_;
  /** agg task unique id → its batched originals, awaiting broadcast. */
  std::mutex pending_mu_;
  std::unordered_map<u64, std::vector<ctp::ipc::FullPtr<Task>>> pending_;
};

}  // namespace clio::run

#endif  // CLIO_RUNTIME_INCLUDE_BATCH_MANAGER_H_
