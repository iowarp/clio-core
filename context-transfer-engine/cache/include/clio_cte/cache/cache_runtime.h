/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */
#ifndef CLIO_CTE_CACHE_CACHE_RUNTIME_H_
#define CLIO_CTE_CACHE_CACHE_RUNTIME_H_

#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <string>

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_interposer.h>
#include <clio_cte/cache/cache_client.h>
#include <clio_cte/cache/cache_tasks.h>

namespace clio::cte::cache {

/**
 * Cache chimod runtime (issue #886 cache/replication split) — the TOP of
 * the interposition chain (cache -> [compressor] -> replication -> core).
 * Keeps a node-local UNTRANSFORMED copy of each blob in the core's
 * REPLICA_CACHE slot, which the SHM zero-IPC fast path and raw task reads
 * serve directly, with ASYNCHRONOUS WRITE-THROUGH semantics (the same
 * discipline as the replication chimod):
 *
 *  - PutBlob/MultiPutBlob forward the AUTHORITATIVE write down `next`
 *    FIRST — it must succeed before the ack, so the source of truth is
 *    current after every ack and a crash loses nothing acked — then write
 *    the raw cache copy (best-effort; a cache failure degrades reads,
 *    never the put). The layers below keep their own async machinery
 *    (replication's durable copies are sweep-driven); this layer defers
 *    nothing and tracks no dirty state.
 *  - GetBlob serves from the cache replica when it covers the request
 *    (never stale: write-through updates it before the ack), else forwards
 *    down (decompressed below) and best-effort re-populates.
 *  - GetBlobSize answers from the cache replica (raw size == logical size)
 *    before consulting the chain.
 */
class Runtime : public clio::cte::core::CoreInterposer {
 public:
  using CreateParams = CacheConfig;  // required by CLIO_TASK_CC

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
    // The cache interposes the core blob path and adds only a lookup on top,
    // so the base interposer estimate (size-driven) is the right shape.
    return clio::cte::core::CoreInterposer::GetTaskStats(task);
  }


  // ---- Method handlers ----
  clio::run::TaskResume Create(clio::run::shared_ptr<CreateTask> &task);
  clio::run::TaskResume Destroy(clio::run::shared_ptr<DestroyTask> &task);
  clio::run::TaskResume Monitor(clio::run::shared_ptr<MonitorTask> &task);
  clio::run::TaskResume PutBlob(
      clio::run::shared_ptr<clio::cte::core::PutBlobTask> &task);
  clio::run::TaskResume GetBlob(
      clio::run::shared_ptr<clio::cte::core::GetBlobTask> &task);
  clio::run::TaskResume GetBlobSize(
      clio::run::shared_ptr<clio::cte::core::GetBlobSizeTask> &task);
  clio::run::TaskResume MultiPutBlob(
      clio::run::shared_ptr<clio::cte::core::MultiPutBlobTask> &task);
  clio::run::TaskResume PodMultiGetBlob(
      clio::run::shared_ptr<clio::cte::core::PodMultiGetBlobTask> &task);
  clio::run::TaskResume EnrollTag(clio::run::shared_ptr<TagCacheTask> &task);
  clio::run::TaskResume InvalidateTag(clio::run::shared_ptr<TagCacheTask> &task);
  clio::run::TaskResume ForgetTag(clio::run::shared_ptr<TagCacheTask> &task);
  // ---- Tag metadata cache (write-through) ----
  clio::run::TaskResume GetOrCreateTag(
      clio::run::shared_ptr<
          clio::cte::core::GetOrCreateTagTask<clio::cte::core::CreateParams>>
          &task);
  clio::run::TaskResume DelTag(
      clio::run::shared_ptr<clio::cte::core::DelTagTask> &task);
  clio::run::TaskResume RenameTag(
      clio::run::shared_ptr<clio::cte::core::RenameTagTask> &task);
  clio::run::TaskResume TagCachePut(
      clio::run::shared_ptr<TagCacheUpdateTask> &task);
  clio::run::TaskResume TagCacheDrop(
      clio::run::shared_ptr<TagCacheUpdateTask> &task);

  // ---- Container virtuals (defined in autogen/cache_lib_exec.cc) ----
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

  /**
   * Reader-local read routing (issue #886 distributed coherence): Dynamic
   * GetBlob/GetBlobSize resolve to THIS node's cache container, which
   * serves from the node-local raw copy and, on a miss, fetches from the
   * blob's owner chain, populates the local copy, and registers it for the
   * owner's put-invalidation. Everything else (writes, explicit-replica
   * ops, module verbs) routes exactly as the core does (owner by blob
   * hash) via the CoreInterposer delegation.
   */
  clio::run::PoolQuery ScheduleTask(
      const clio::run::shared_ptr<clio::run::Task> &task) override;

 private:
  /** Aim a put context at THE cache replica slot (raw bytes, score floor). */
  void AimAtCacheReplica(Context *ctx) const;

  /**
   * True when THIS node hosts the blob's owner container (issue #894).
   * The owner node keeps NO cache-slot copy: its primary is already
   * node-local DRAM (and zero-IPC readable), and an owner-side copy sits
   * outside the register/invalidate protocol — origin registration skips
   * self, so nothing would ever invalidate it and it serves stale bytes
   * after a foreign write (the CI-caught fragmented winner split).
   */
  bool IsBlobOwnerLocal(const TagId &tag_id, const std::string &blob_name);
  /** Whether this node serves the blob for its dead owner (failover). */
  bool IsStandingIn(const TagId &tag_id, const std::string &blob_name);

  /**
   * Best-effort population of THIS node's raw local copy from the blob's
   * authoritative owner chain, plus coherence registration (the owner's
   * next primary write invalidates the copy). Sequential from 0, so an
   * interruption leaves a valid prefix.
   */
  clio::run::TaskResume PopulateLocal(const TagId &tag_id,
                                      const std::string &blob_name);

  /** Lazily bind the next-pool client (compose next_pool_id). */
  clio::cte::core::Client *GetNextClient();
  /** Lazily bind a direct core client for node-LOCAL copy access. */
  clio::cte::core::Client *GetLocalClient();

  // ---- Tag-scoped caching (pager path) ----
  /** Per enrolled tag: the pages this node holds a cache copy of, so a tag
   *  invalidation can release them without scanning the core. */
  struct TagCache {
    std::unordered_set<std::string> cached;
    clio::run::u64 hits = 0, misses = 0, populated = 0;
  };
  static clio::run::u64 TagKey(const TagId &t) {
    return (static_cast<clio::run::u64>(t.major_) << 32) | t.minor_;
  }
  bool TagEnrolled(const TagId &tag_id, bool auto_enroll);
  clio::run::TaskResume DropTagCopies(const TagId &tag_id,
                                      clio::run::u64 &dropped);
  std::mutex tags_mu_;
  std::unordered_map<clio::run::u64, TagCache> tags_;

  /** Tag metadata cache: every name of a tag -> its id, and id -> names so a
   *  delete/rename drops all of them. Static metadata only: a tag's byte
   *  size changes on every page write and is NOT cached. */
  std::mutex tmeta_mu_;
  std::unordered_map<std::string, TagId> tmeta_by_name_;
  std::unordered_map<clio::run::u64, std::unordered_set<std::string>>
      tmeta_names_;
  bool TagMetaLookup(const std::string &name, TagId *id);
  void TagMetaInsert(const std::string &name, const TagId &id);
  void TagMetaDrop(const TagId &id, const std::string &name);
  /** Broadcast a put/drop to every node's cache and wait for all of them. */
  clio::run::Future<TagCacheUpdateTask> SendTagMeta(
      clio::run::u32 method, const std::string &name, const TagId &id,
      const clio::run::PoolId &fault_pool_id,
      const std::string &fault_pool_name, const std::string &fault_params);

  CacheConfig config_;
  std::unique_ptr<clio::cte::core::Client> next_client_;
  std::unique_ptr<clio::cte::core::Client> local_client_;
};

}  // namespace clio::cte::cache

#endif  // CLIO_CTE_CACHE_CACHE_RUNTIME_H_
