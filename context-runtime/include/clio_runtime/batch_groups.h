/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * BSD 3-Clause License. See LICENSE file.
 */

#ifndef CLIO_RUNTIME_INCLUDE_BATCH_GROUPS_H_
#define CLIO_RUNTIME_INCLUDE_BATCH_GROUPS_H_

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "clio_runtime/task.h"
#include "clio_runtime/types.h"

namespace clio::run {

/**
 * Worker-local task batching (issue #820).
 *
 * A worker collects a bounded run of ready tasks and lets each task's container
 * COALESCE them before any executes: N independent tasks collapse into a
 * minimal set of merged tasks, each completing the parents it subsumed.
 *
 * This is deliberately NOT the ManyToOne collective in BatchManager. That one
 * reduces N combinable inputs to a single aggregate and broadcasts one result
 * back to every member (all members see the same OUT). Here the output is a
 * *subset* of tasks, and each merged task completes a DIFFERENT set of parents.
 * Reduction vs. merge.
 *
 * The motivating case: the filesystem stores each 1 MiB page as one blob, so a
 * sequential 4 KiB workload sends 256 tasks at one page-blob. Each must take
 * that blob's #680 write token across its whole body, so they drain single-file
 * (measured: max ~1 s stalls). Coalescing them into one vectored PutBlob makes
 * it one token acquire and one bdev pass.
 */

/** Identity of a batch group. `key` is opaque to the worker: the container
 *  chooses it (e.g. a hash of tag id + blob name, so one group == one blob). */
struct BatchKey {
  PoolId pool_id;
  u32 method;
  u64 key;

  bool operator==(const BatchKey &o) const {
    return pool_id == o.pool_id && method == o.method && key == o.key;
  }
};

struct BatchKeyHash {
  size_t operator()(const BatchKey &k) const {
    size_t h = std::hash<u64>()(
        (static_cast<u64>(k.pool_id.major_) << 32) | k.pool_id.minor_);
    h ^= std::hash<u32>()(k.method) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    h ^= std::hash<u64>()(k.key) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    return h;
  }
};

/** One member parked in a group, with the arrival order that decides how
 *  overlapping work is resolved when the group is merged. Submission order is
 *  the only thing that makes "last writer wins" meaningful, so the worker
 *  stamps it at BuildBatch time rather than letting the container guess. */
struct BatchMember {
  clio::run::shared_ptr<Task> task;
  u64 seq = 0;
};

/** Tasks parked by BuildBatch, keyed by whatever the container chose. Owned by
 *  the worker and reused across phases (cleared, not reallocated). */
using BatchGroups =
    std::unordered_map<BatchKey, std::vector<BatchMember>, BatchKeyHash>;

/**
 * How SmashBatch hands a merged task back to the worker.
 *
 * `Emit` takes ownership of `merged` and the list of parents it completes. When
 * `merged` finishes, the worker copies its return code to each parent and
 * completes it (local future signal or remote SendOut), which is the same
 * fan-out BatchManager::OnAggregateComplete performs for the collective path.
 *
 * A merged task whose parents' outputs are already satisfied by the merged task
 * itself -- e.g. a vectored GetBlob that read directly into each parent's own
 * buffer -- needs no per-parent payload copy, which is exactly why the vectored
 * task shape (part A) exists.
 */
class BatchSink {
 public:
  virtual ~BatchSink() = default;

  /** Submit a NEW synthetic task that completes `parents` when it finishes.
   *  `merged` must be a freshly built task (e.g. from NewCopyTask), never one
   *  of the parked originals — Emit reassigns its task id, which would strand
   *  the client future already bound to an original's id. */
  virtual void Emit(const clio::run::shared_ptr<Task> &merged,
                    std::vector<clio::run::shared_ptr<Task>> &&parents) = 0;

  /** Run a parked ORIGINAL task unchanged, exactly as it would have run had it
   *  never been offered for batching. This is the right answer whenever merging
   *  is not worth it or not possible — a group of one, or a failed merge —
   *  because the task already carries its client's future and must keep it. */
  virtual void Passthrough(const clio::run::shared_ptr<Task> &task) = 0;
};

}  // namespace clio::run

#endif  // CLIO_RUNTIME_INCLUDE_BATCH_GROUPS_H_
