/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */
#ifndef CLIO_CTE_REPLICATION_REPLICATION_RUNTIME_H_
#define CLIO_CTE_REPLICATION_REPLICATION_RUNTIME_H_

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_interposer.h>
#include <clio_cte/replication/replication_client.h>
#include <clio_cte/replication/replication_tasks.h>

namespace clio::cte::replication {

/**
 * Replication chimod runtime (issue #886). The CTE core stores replicas and
 * addresses them (Context::replica_); this module owns ALL replica policy —
 * and it INTERPOSES on the core's own task interface so callers need no new
 * client: point a clio::cte::core::Client at this pool (or set
 * CLIO_CTE_POOL=561.0) and every op keeps working. PutBlob/GetBlob carry
 * the CORE's task structs and method ids: the interposed PutBlob forwards
 * the primary write to the core pool and writes through to num_replicas
 * FIXED|PERSISTENT replicas; the interposed GetBlob serves a dropped
 * primary from a covering replica and re-caches it. Every other core
 * method is forwarded to the next pool untouched. ReplicateBlob/FlushTag
 * remain the explicit verbs, numbered above the core's method space.
 */
class Runtime : public clio::cte::core::CoreInterposer {
 public:
  using CreateParams = ReplicationConfig;  // required by CLIO_TASK_CC

  Runtime() = default;
  ~Runtime() override = default;


  /**
   * Per-task cost estimate for the scheduler (see Container::GetTaskStats).
   *
   * compute_ is the feature Container::InferCpuTime multiplies its learned
   * per-method coefficient by; leaving it 0 — as every chimod but MOD_NAME and
   * admin did — collapses that model to one constant per method with no
   * dependence on request size. wall_time_ seeds InferWallClockTime at the
   * ~500 MB/s house convention. Both coefficients are then learned from real
   * completions, so these only need the right order of magnitude.
   */
  clio::run::TaskStat GetTaskStats(const clio::run::Task *task) const override {
    clio::run::TaskStat stat;
    if (task == nullptr) {
      return stat;
    }
    switch (task->method_) {
      case Method::kReplicateBlob:
        // Re-reads the source blob and writes it to a peer: the local CPU cost
        // is one copy, but the wall time is a network round trip, so the two
        // estimates deliberately diverge here.
        stat.compute_ = 30;
        stat.wall_time_ = 2000.0f;
        return stat;
      case Method::kFlushTag:
        stat.compute_ = 50;
        stat.wall_time_ = 5000.0f;
        return stat;
      case Method::kReplicateSweep:
        // Periodic scan over pending replicas; cheap when there is nothing to
        // do, which is the common case.
        stat.compute_ = 10;
        stat.wall_time_ = 20.0f;
        return stat;
      default:
        return clio::cte::core::CoreInterposer::GetTaskStats(task);
    }
  }


  // ---- Method handlers ----
  clio::run::TaskResume Create(clio::run::shared_ptr<CreateTask> &task);
  clio::run::TaskResume Destroy(clio::run::shared_ptr<DestroyTask> &task);
  clio::run::TaskResume Monitor(clio::run::shared_ptr<MonitorTask> &task);
  clio::run::TaskResume ReplicateBlob(
      clio::run::shared_ptr<ReplicateBlobTask> &task);
  clio::run::TaskResume FlushTag(clio::run::shared_ptr<FlushTagTask> &task);
  /** Interposed core put (Method::kPutBlob, core task struct): primary
   *  forwarded to the core pool, then written through to the module's fixed
   *  persistent replica set. Explicit Context::replica_ != 0 passes through
   *  untouched. */
  clio::run::TaskResume PutBlob(
      clio::run::shared_ptr<clio::cte::core::PutBlobTask> &task);
  /** Interposed core get (Method::kGetBlob, core task struct): local/primary
   *  probe, replica fallback for a dropped primary, then best-effort
   *  primary re-cache and node-local cache population. */
  clio::run::TaskResume GetBlob(
      clio::run::shared_ptr<clio::cte::core::GetBlobTask> &task);
  /** Interposed size probe (Method::kGetBlobSize): reports the blob's
   *  LOGICAL size — a dropped primary answers with its best replica's size,
   *  so size-then-read callers still reach the replica-serving GetBlob. */
  clio::run::TaskResume GetBlobSize(
      clio::run::shared_ptr<clio::cte::core::GetBlobSizeTask> &task);
  /** Periodic async write-through driver (Method::kReplicateSweep): drains
   *  the pending-replication set by re-copying each dirty blob's CURRENT
   *  primary into the persistent replica set (ReplicateOne per blob). */
  clio::run::TaskResume ReplicateSweep(
      clio::run::shared_ptr<ReplicateSweepTask> &task);
  /** Interposed batched put (Method::kMultiPutBlob): the batch runs on the
   *  core verbatim (primaries), then each record is written through to the
   *  persistent replica set — the deferred-put pipeline (#862/#878) ships
   *  small records this way, and letting it bypass the write-through would
   *  silently strip durability from every async caller. */
  clio::run::TaskResume MultiPutBlob(
      clio::run::shared_ptr<clio::cte::core::MultiPutBlobTask> &task);

  // ---- Container virtuals (defined in autogen/replication_lib_exec.cc) ----
  /** Recovery start (`clio_run restart`): Create pulls the handoff. */
  void Restart(const clio::run::PoolId &pool_id, const std::string &pool_name,
               clio::run::u32 container_id = 0) override;
  void Init(const clio::run::PoolId &pool_id, const std::string &pool_name,
            clio::run::u32 container_id = 0) override;
  clio::run::TaskResume Run(clio::run::u32 method,
                      clio::run::shared_ptr<clio::run::Task> task_ptr) override;
  clio::run::u64 GetWorkRemaining() const override;
  void LocalLoadTask(clio::run::u32 method, clio::run::DefaultLoadArchive &archive,
                     clio::run::shared_ptr<clio::run::Task>& task_ptr) override;
  clio::run::shared_ptr<clio::run::Task> LocalAllocLoadTask(
      clio::run::u32 method, clio::run::DefaultLoadArchive &archive) override;
  void LocalSaveTask(clio::run::u32 method, clio::run::DefaultSaveArchive &archive,
                     clio::run::shared_ptr<clio::run::Task>& task_ptr) override;
  void AggregateOut(clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> &orig_task,
                    const clio::run::shared_ptr<clio::run::Task> &replica_task) override;
  void AggregateIn(clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> &agg_task,
                   const clio::run::shared_ptr<clio::run::Task> &member_task) override;
  void SaveTask(clio::run::u32 method, clio::run::SaveTaskArchive &archive,
                clio::run::shared_ptr<clio::run::Task>& task_ptr) override;
  void LoadTask(clio::run::u32 method, clio::run::LoadTaskArchive &archive,
                clio::run::shared_ptr<clio::run::Task>& task_ptr) override;
  clio::run::shared_ptr<clio::run::Task> AllocLoadTask(clio::run::u32 method,
                                             clio::run::LoadTaskArchive &archive) override;
  clio::run::shared_ptr<clio::run::Task> NewCopyTask(clio::run::u32 method,
                                           clio::run::shared_ptr<clio::run::Task> &orig,
                                           bool deep) override;
  clio::run::shared_ptr<clio::run::Task> NewTask(clio::run::u32 method) override;

 private:
  /**
   * Copy one blob's primary bytes into replica `replica_idx`, chunked so a
   * large blob never needs a blob-sized bounce buffer. rc uses ReplicateBlob's
   * return-code space: 2 alloc failure, 10+x GetBlobSize, 20+x GetBlob,
   * 30+x PutBlob.
   */
  clio::run::TaskResume ReplicateOne(const TagId &tag_id,
                                     const std::string &blob_name,
                                     int replica_idx,
                                     const Context &context,
                                     clio::run::u64 &bytes_copied,
                                     clio::run::u32 &rc,
                                     float put_score = -1.0f);

  /**
   * Copy replica `replica_idx`'s FULL contents back into the primary,
   * sequentially from offset 0 in bounded chunks, so an interruption leaves
   * a valid prefix (future CachedGets treat uncovered ranges as misses).
   * Best-effort: a failed chunk stops the copy without failing the read that
   * triggered it. recached reports bytes restored.
   */
  // ---- remote copies and failover (replication_remote.cc) ----
  /** Delete a blob; at its owner, also its remote copies. */
  clio::run::TaskResume DelBlob(
      clio::run::shared_ptr<clio::cte::core::DelBlobTask> &task);
  /** Truncate a blob; at its owner, also its remote copies. */
  clio::run::TaskResume TruncateBlob(
      clio::run::shared_ptr<clio::cte::core::TruncateBlobTask> &task);
  /** Hand back to `owner_` every blob this container changed for it. */
  clio::run::TaskResume HandoffPull(clio::run::shared_ptr<HandoffPullTask> &task);
  /** Periodic: hand changes back to owners that are alive again. */
  clio::run::TaskResume HandoffSweep(
      clio::run::shared_ptr<HandoffSweepTask> &task);
  /** The original PutBlob: primary plus this node's replicas. */
  clio::run::TaskResume PutBlobLocal(
      clio::run::shared_ptr<clio::cte::core::PutBlobTask> &task);
  /** The original MultiPutBlob: primary batch plus this node's replicas. */
  clio::run::TaskResume MultiPutBlobLocal(
      clio::run::shared_ptr<clio::cte::core::MultiPutBlobTask> &task);

  clio::run::TaskResume RecachePrimary(const TagId &tag_id,
                                       const std::string &blob_name,
                                       int replica_idx,
                                       clio::run::u64 rep_size,
                                       clio::run::u64 &recached);

  /**
   * Before a primary write that starts at `write_off`: if the primary holds
   * fewer than `write_off` bytes but a replica holds more (the primary's
   * volatile blocks were dropped by a restart), copy that replica back into
   * the primary first. Otherwise the write would re-grow the primary over
   * [0, write_off) with unfilled blocks, which then shadow the intact
   * replica: every read of that range returns zeros.
   * @param tag_id blob's tag
   * @param blob_name blob name
   * @param write_off lowest offset the pending write touches
   */
  clio::run::TaskResume RefillPrimaryBeforeWrite(const TagId &tag_id,
                                                 const std::string &blob_name,
                                                 clio::run::u64 write_off);

  /**
   * Populate THIS node's local cache copy of a remote blob (issue #886
   * distributed coherence): chunked copy of the owner's primary into the
   * LOCAL container (PoolQuery::Local puts), then register this node with
   * the owner (RegisterReplicaContainer) so the next primary write
   * invalidates the copy. Best-effort — a failed chunk abandons the local
   * copy without failing the read that triggered it, and registration only
   * happens after a COMPLETE copy (a partial local copy is never served
   * because CachedGet requires coverage, but registering it would earn a
   * pointless invalidation).
   */
  clio::run::TaskResume CacheLocalCopy(const TagId &tag_id,
                                       const std::string &blob_name,
                                       clio::run::u64 total,
                                       bool &cached);

  /** Lazily bind the core client (compose next_pool_id, default kCtePoolId) */
  clio::cte::core::Client *GetCoreClient();

  // CorePoolId/CoreContainer/ForwardToCore + the lib_exec dispatch
  // defaults come from clio::cte::core::CoreInterposer (shared base).

  /** Mark a blob dirty for the async write-through (deduped by key). */
  void EnqueueReplication(const TagId &tag_id, const std::string &blob_name);

  ReplicationConfig config_;
  std::unique_ptr<clio::cte::core::Client> core_client_;

  /** Async write-through state (issue #886): blobs whose primary is newer
   *  than their replicas, keyed "major.minor.blob" so rapid overwrites of
   *  one blob coalesce into a single pending entry. The sweep swaps the map
   *  out under the lock and replicates outside it; a put racing the sweep
   *  simply re-inserts and is caught next period. */
  std::mutex pending_mtx_;
  std::unordered_map<std::string, std::pair<TagId, std::string>> pending_;

  // ---- remote copies and failover (replication_remote.cc) ----
  /** A blob changed here while its owner was down. */
  struct HandoffEntry {
    TagId tag_;
    std::string name_;
    bool deleted_ = false;
  };
  /** @return containers in this pool. */
  clio::run::u32 NumContainers() const;
  /** @return the container that owns a blob by hash. */
  clio::run::u32 OwnerOf(const TagId &tag, const std::string &name) const;
  /** @return true if `container`'s node is alive. */
  bool ContainerAlive(clio::run::u32 container) const;
  /**
   * Record a change made here on behalf of a dead owner.
   * @param owner the owner container
   * @param tag blob's tag
   * @param name blob name
   * @param deleted true for a delete, false for a write/truncate
   */
  void NoteHandoff(clio::run::u32 owner, const TagId &tag,
                   const std::string &name, bool deleted);
  /**
   * Mirror one written range to this blob's remote copies (owner side).
   * @param tag blob's tag
   * @param name blob name
   * @param off offset of the range
   * @param size size of the range
   * @param data the bytes
   * @param score blob score
   */
  clio::run::TaskResume MirrorRange(TagId tag, std::string name,
                                    clio::run::u64 off, clio::run::u64 size,
                                    ctp::ipc::ShmPtr<> data, float score);
  /**
   * Hand back every change recorded for `owner` to it.
   * @param owner the returned owner
   * @param pushed receives the number of blobs handed back
   */
  clio::run::TaskResume PushHandoff(clio::run::u32 owner, clio::run::u32 *pushed);
  /**
   * Copy this container's shadow of one blob to its owner.
   * @param e the blob
   * @param owner owner container
   * @param ok receives true on success
   */
  clio::run::TaskResume PushOne(HandoffEntry e, clio::run::u32 owner, bool *ok);
  /** Client bound to THIS (replication) pool: mirrored ops run through the
   *  target node's replication container, so its local replicas apply. */
  clio::cte::core::Client *Self();
  std::unique_ptr<clio::cte::core::Client> self_client_;
  std::mutex handoff_mu_;
  std::unordered_map<clio::run::u32,
                     std::unordered_map<std::string, HandoffEntry>> handoff_;
  bool is_restart_ = false;
};

}  // namespace clio::cte::replication

#endif  // CLIO_CTE_REPLICATION_REPLICATION_RUNTIME_H_
