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

#ifndef WRPCTE_CORE_RUNTIME_H_
#define WRPCTE_CORE_RUNTIME_H_

#include <memory>
#include <atomic>
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/comutex.h>
#include <clio_runtime/corwlock.h>
#include <clio_ctp/data_structures/priv/unordered_map_ll.h>
#include <clio_ctp/data_structures/ipc/ring_buffer.h>
#include <clio_ctp/memory/allocator/malloc_allocator.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_config.h>
#include <clio_cte/core/dpe/dpe.h>
#include <clio_cte/core/core_tasks.h>
#include <clio_cte/core/data_organizer/data_organizer.h>
#include <clio_cte/core/gpu_metadata_cache.h>
#include <clio_cte/core/transaction_log.h>
#include <clio_ctp/search/regex_search_engine.h>

// Forward declarations to avoid circular dependency
namespace clio::cte::core {
class Config;
}

namespace clio::cte::core {


/**
 * Determine whether a target is eligible for blob placement under TTL rules.
 *
 * @param target Target metadata, including persistence and predicted TTL.
 * @param min_persistence_level Minimum persistence level required by the blob.
 * @return True when the target may receive the blob.
 */
bool IsTargetEligibleForBlob(const TargetInfo &target,
                             int min_persistence_level);

/**
 * CTE Core Runtime Container
 * Implements target management and tag/blob operations
 */
class Runtime : public clio::run::Container {
public:
  using CreateParams = clio::cte::core::CreateParams; // Required for CLIO_TASK_CC

  Runtime() = default;
  // Virtual destructor performs the same metadata/WAL cleanup as the Destroy()
  // method, so deleting the container on graceful shutdown (PoolManager::
  // DestroyAllContainers) frees this module's runtime-heap state — closing the
  // WAL files and clearing the tag/blob/target maps — instead of leaking it
  // until process exit.
  ~Runtime() override;

  /**
   * Fix up POD task members (clio::run::priv::string SSO data_ pointers,
   * etc.) after a GPU2CPU D2H POD memcpy. Dispatched by the GPU pop
   * path on the worker before Run.
   */
  void FixupAfterCopy(clio::run::u32 method,
                      clio::run::shared_ptr<clio::run::Task>& task_ptr) override;

  /**
   * Create the container (Method::kCreate)
   * This method both creates and initializes the container
   * Returns TaskResume for coroutine-based async operations
   */
  clio::run::TaskResume Create(clio::run::shared_ptr<CreateTask> &task);

  /**
   * Monitor container state (Method::kMonitor)
   */
  clio::run::TaskResume Monitor(clio::run::shared_ptr<MonitorTask> &task);

  /**
   * Destroy the container (Method::kDestroy)
   */
  clio::run::TaskResume Destroy(clio::run::shared_ptr<DestroyTask> &task);

  /**
   * Register a target (Method::kRegisterTarget)
   * Returns TaskResume for coroutine-based async operations
   */
  clio::run::TaskResume RegisterTarget(clio::run::shared_ptr<RegisterTargetTask> &task);

  /**
   * Unregister a target (Method::kUnregisterTarget)
   */
  clio::run::TaskResume UnregisterTarget(clio::run::shared_ptr<UnregisterTargetTask> &task);

  /**
   * List registered targets (Method::kListTargets)
   */
  clio::run::TaskResume ListTargets(clio::run::shared_ptr<ListTargetsTask> &task);

  /**
   * Update target statistics (Method::kStatTargets)
   */
  clio::run::TaskResume StatTargets(clio::run::shared_ptr<StatTargetsTask> &task);

  /**
   * Get target information (Method::kGetTargetInfo)
   * Returns target score, remaining space, and performance metrics
   */
  clio::run::TaskResume GetTargetInfo(clio::run::shared_ptr<GetTargetInfoTask> &task);

  /**
   * Get or create a tag (Method::kGetOrCreateTag)
   */
  template <typename CreateParamsT = CreateParams>
  clio::run::TaskResume GetOrCreateTag(clio::run::shared_ptr<GetOrCreateTagTask<CreateParamsT>> &task);

  /**
   * Put/Get/Reorganize blob handlers. The real logic lives once in the
   * *Impl<TaskT> member templates; the public handlers (and the POD variants
   * Pod*Blob, Method::kPod*) are thin wrappers. Both the priv::string and
   * fixed_string blob-name types expose .str(), so the impls are agnostic to
   * which task carries them — that is the code-dedup for issue #556.
   */
  template <typename TaskT>
  clio::run::TaskResume PutBlobImpl(clio::run::shared_ptr<TaskT> &task);
  template <typename TaskT>
  clio::run::TaskResume GetBlobImpl(clio::run::shared_ptr<TaskT> &task);
  template <typename TaskT>
  clio::run::TaskResume ReorganizeBlobImpl(clio::run::shared_ptr<TaskT> &task);

  /** Put blob (Method::kPutBlob) - allocates and writes data to blob. */
  clio::run::TaskResume PutBlob(clio::run::shared_ptr<PutBlobTask> &task);
  /** Get blob (Method::kGetBlob) - reads data from existing blob. */
  clio::run::TaskResume GetBlob(clio::run::shared_ptr<GetBlobTask> &task);
  /** Reorganize single blob (Method::kReorganizeBlob) - update score. */
  clio::run::TaskResume ReorganizeBlob(clio::run::shared_ptr<ReorganizeBlobTask> &task);

  /**
   * Rescore-and-move one blob without spawning a ReorganizeBlobTask (issue
   * #738). This is the body shared by the ReorganizeBlob task handler and
   * the internal data organizers, which call it directly from the periodic
   * DynamicReorganize coroutine. Applies the configured
   * score_difference_threshold (small deltas are a successful no-op).
   * @param tag_id Owning tag
   * @param blob_name Blob to rescore
   * @param new_score Target score in [0, 1]
   * @param rc Output: 0 on success/skip, non-zero error code otherwise
   */
  clio::run::TaskResume ReorganizeBlobInternal(const TagId &tag_id,
                                               const std::string &blob_name,
                                               float new_score,
                                               clio::run::u32 &rc);

  /**
   * Periodic internal data-organizer driver (Method::kDynamicReorganize).
   * One-liner delegation: all organization logic lives in the configured
   * DataOrganizer (issue #738).
   */
  clio::run::TaskResume DynamicReorganize(
      clio::run::shared_ptr<DynamicReorganizeTask> &task);

  /**
   * Snapshot organizer-relevant metadata for this replica's partition of the
   * local blob map. Partitioning hashes each blob's composite key modulo the
   * configured organizer_tasks, so the `organizer_tasks` periodic replicas
   * cover the blob space disjointly. Empty blobs are skipped (nothing to
   * move).
   * @param replica_id 0-based replica index of the calling organizer task
   * @param out Filled with one entry per blob owned by this replica
   */
  void CollectOrganizerBlobStats(clio::run::u32 replica_id,
                                 std::vector<OrganizerBlobStat> &out);

  /** Fully-POD, GPU-compatible blob handlers (issue #556). Thin wrappers over
   *  the *Impl<TaskT> templates above. */
  clio::run::TaskResume PodPutBlob(clio::run::shared_ptr<PodPutBlobTask> &task);
  clio::run::TaskResume PodGetBlob(clio::run::shared_ptr<PodGetBlobTask> &task);
  clio::run::TaskResume PodReorganizeBlob(
      clio::run::shared_ptr<PodReorganizeBlobTask> &task);

  /**
   * Delete blob operation - removes blob and decrements tag size
   * Returns TaskResume for coroutine-based async operations
   */
  clio::run::TaskResume DelBlob(clio::run::shared_ptr<DelBlobTask> &task);

  /**
   * Truncate blob (Method::kTruncateBlob) - resize a blob to an exact logical
   * size (grow/shrink) via the shared ResizeBlob helper.
   */
  clio::run::TaskResume TruncateBlob(clio::run::shared_ptr<TruncateBlobTask> &task);

  /**
   * Rename tag (Method::kRenameTag) - change a tag's name in place, keeping
   * its TagId (and all blobs). Broadcast op; shares no data movement.
   */
  clio::run::TaskResume RenameTag(clio::run::shared_ptr<RenameTagTask> &task);

  /**
   * GetOrCreateTagAlias (Method::kGetOrCreateTagAlias) - bind an extra name to
   * an existing tag's id (hard link at the tag level). Broadcast op.
   */
  clio::run::TaskResume GetOrCreateTagAlias(
      clio::run::shared_ptr<GetOrCreateTagAliasTask> &task);

  /**
   * Delete tag operation - removes all blobs from tag and removes tag
   * Returns TaskResume for coroutine-based async operations
   */
  clio::run::TaskResume DelTag(clio::run::shared_ptr<DelTagTask> &task);

  /**
   * GetTagName (Method::kGetTagName) - resolve a TagId to its full, absolute
   * tag name by walking the stored relative "$tagid{parent}/leaf" references.
   * Broadcast op; the container owning the tag's metadata answers.
   */
  clio::run::TaskResume GetTagName(clio::run::shared_ptr<GetTagNameTask> &task);

  /**
   * Get tag size operation - returns total size of all blobs in tag
   */
  clio::run::TaskResume GetTagSize(clio::run::shared_ptr<GetTagSizeTask> &task);

  /**
   * Get max (total) capacity — sum of max_capacity_ over targets registered on
   * this node. Broadcast to sum across the cluster (AggregateOut adds replicas).
   */
  clio::run::TaskResume GetCapacity(clio::run::shared_ptr<GetCapacityTask> &task);

  /**
   * GetNumAliases (Method::kGetNumAliases) - number of extra names (tag-level
   * hard links) bound to a tag, by name or id. Excludes the canonical name, so
   * the POSIX link count is num_aliases_ + 1. Broadcast op; the container that
   * owns the tag answers.
   */
  clio::run::TaskResume GetNumAliases(clio::run::shared_ptr<GetNumAliasesTask> &task);

  /**
   * Schedule a task by resolving Dynamic pool queries.
   */
  clio::run::PoolQuery ScheduleTask(const clio::run::shared_ptr<clio::run::Task> &task) override;

  // Pure virtual methods - implementations are in autogen/core_lib_exec.cc
  void Init(const clio::run::PoolId &pool_id, const std::string &pool_name,
            clio::run::u32 container_id = 0) override;
  void Restart(const clio::run::PoolId &pool_id, const std::string &pool_name,
               clio::run::u32 container_id = 0) override;
  clio::run::TaskResume Run(clio::run::u32 method,
                      clio::run::shared_ptr<clio::run::Task> task_ptr) override;
  clio::run::u64 GetWorkRemaining() const override;

  /**
   * Override GetTaskStats so PutBlob / GetBlob report their actual byte
   * payload size (size_) for routing. Otherwise the scheduler buckets
   * every blob op as "metadata" and lands them on the scheduler worker,
   * making large blob transfers compete with latency-sensitive admin
   * traffic instead of going to dedicated I/O workers.
   */
  clio::run::TaskStat GetTaskStats(const clio::run::Task *task) const override;

  // Container virtual method implementations (defined in autogen/core_lib_exec.cc)
  void SaveTask(clio::run::u32 method, clio::run::SaveTaskArchive &archive,
                clio::run::shared_ptr<clio::run::Task>& task_ptr) override;
  void LoadTask(clio::run::u32 method, clio::run::LoadTaskArchive &archive,
                clio::run::shared_ptr<clio::run::Task>& task_ptr) override;
  clio::run::shared_ptr<clio::run::Task> AllocLoadTask(clio::run::u32 method, clio::run::LoadTaskArchive &archive) override;
  clio::run::shared_ptr<clio::run::Task> NewCopyTask(clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> &orig_task_ptr,
                                        bool deep) override;
  clio::run::shared_ptr<clio::run::Task> NewTask(clio::run::u32 method) override;
  void AggregateOut(clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> &orig_task,
                 const clio::run::shared_ptr<clio::run::Task>& replica_task) override;
  void LocalLoadTask(clio::run::u32 method, clio::run::DefaultLoadArchive &archive,
                     clio::run::shared_ptr<clio::run::Task>& task_ptr) override;
  clio::run::shared_ptr<clio::run::Task> LocalAllocLoadTask(clio::run::u32 method,
                                               clio::run::DefaultLoadArchive &archive) override;
  void LocalSaveTask(clio::run::u32 method, clio::run::DefaultSaveArchive &archive,
                     clio::run::shared_ptr<clio::run::Task>& task_ptr) override;

private:
  // Queue ID constants (REQUIRED: Use semantic names, not raw integers)
  static const clio::run::QueueId kTargetManagementQueue = 0;
  static const clio::run::QueueId kTagManagementQueue = 1;
  static const clio::run::QueueId kBlobOperationsQueue = 2;
  static const clio::run::QueueId kStatsQueue = 3;

  // Client for this ChiMod
  Client client_;

  // Target management data structures.
  //
  // Two parallel structures, kept in sync under target_lock_:
  //   - registered_targets_: hash map for O(1) lookup by PoolId
  //   - target_list_:        contiguous vector for O(N_live) iteration
  //                          (avoids scanning the map's empty slots; size()
  //                          returns exact live count)
  // The DPE consumes std::vector<TargetInfo>, so ExtendBlob hands it
  // target_list_ directly with no materialization step.
  ctp::priv::unordered_map_ll<clio::run::PoolId, TargetInfo> registered_targets_;
  std::vector<TargetInfo> target_list_;
  ctp::priv::unordered_map_ll<std::string, clio::run::PoolId>
      target_name_to_id_; // reverse lookup: target_name -> target_id

  // Tag management data structures (using ctp::priv::unordered_map_ll for thread-safe
  // concurrent access)
  ctp::priv::unordered_map_ll<std::string, TagId>
      tag_name_to_id_;                                   // tag_name -> tag_id
  ctp::priv::unordered_map_ll<TagId, std::shared_ptr<TagInfo>> tag_id_to_info_; // tag_id -> TagInfo
  ctp::priv::unordered_map_ll<std::string, std::shared_ptr<BlobInfo>>
      tag_blob_name_to_info_; // "tag_id.blob_name" -> BlobInfo

  // Secondary search index: absolute resolved tag name -> tag id. Lets TagQuery
  // answer regex queries via a trigram prefilter instead of scanning every tag
  // (#598). Maintained in lockstep with tag_name_to_id_/tag_id_to_info_ under
  // tag_map_lock_ (created in GetOrAssignTagId, removed in DelTag, moved in
  // RenameTag, rebuilt after WAL/metadata restore).
  ctp::search::RegexSearchEngine<TagId> tag_search_;

  // Atomic counters for thread-safe ID generation
  std::atomic<clio::run::u32>
      next_tag_id_minor_; // Minor counter for TagId UniqueId generation

  // Map sizes for data structures (must be large enough for expected entries)
  // Initial bucket count only; the maps grow (rehash) as entries are added.
  // Kept small so for_each() (which scans every bucket) stays O(entries) rather
  // than O(1M): the old 1M/100K pre-size made every full-map scan visit a
  // million empty buckets, which is what made metadata-heavy workloads (e.g.
  // generic/089) crawl to a timeout (issue #680).
  static const size_t kBlobMapSize = 100;
  static const size_t kTagMapSize = 100;

  // Synchronization primitives for thread-safe access to data structures
  // Single lock per data structure ensures all operations synchronize correctly
  clio::run::CoRwLock target_lock_;  // For registered_targets_ + target_name_to_id_
  // tag_map_lock_ and blob_map_lock_ removed: the tag/blob maps are self-locking
  // unordered_map_ll and tag_search_ is now internally synchronized, so the outer
  // coroutine locks were redundant and deadlock-prone (issue #680). Values are
  // std::shared_ptr, so a concurrent erase just drops the map's reference while
  // any in-flight handle keeps the object alive -- no use-after-free.
  // Use a set of locks based on maximum number of lanes for better concurrency
  static const size_t kMaxLocks =
      64; // Maximum number of locks (matches max lanes)
  std::vector<std::unique_ptr<clio::run::CoRwLock>>
      target_locks_; // For registered_targets_ (DEPRECATED - use target_lock_)
  std::vector<std::unique_ptr<clio::run::CoRwLock>>
      tag_locks_; // For tag management structures (DEPRECATED - use tag_map_lock_ / blob_map_lock_)

  // Storage configuration (parsed from config file)
  std::vector<StorageDeviceConfig> storage_devices_;

  // CTE configuration (replaces ConfigManager singleton)
  Config config_;

  // Cached Data Placement Engine — built once from config_ in Create() so
  // ExtendBlob does not allocate a fresh DPE on every PutBlob.
  std::unique_ptr<DataPlacementEngine> dpe_;

  // Internal data organizer — built once from config_ in Create() (issue
  // #738). nullptr when config_.organizer_.name_ is "none"/unknown; the
  // periodic DynamicReorganize task then degrades to a no-op.
  std::unique_ptr<DataOrganizer> organizer_;

  // Restart flag: set by Restart() before calling Init()/Create()
  bool is_restart_ = false;

  // -----------------------------------------------------------------
  // GPU metadata cache (optional; populated when
  // config_.gpu_metadata_cache_.enabled_ is true).
  //
  // Owned by the CTE Core server. The header pointer addresses
  // GPU-managed USM (sycl::malloc_shared on SYCL, cudaMallocManaged on
  // CUDA). All mutations go through GpuCache* helpers below, which are
  // the ONLY places that touch this state from PutBlob / GetOrCreateTag /
  // DelBlob / DelTag.
  // -----------------------------------------------------------------
  GpuMetadataCacheHeader *gpu_cache_ = nullptr;
  size_t gpu_cache_bytes_ = 0;

  /**
   * Allocate and initialize the GPU metadata cache region. No-op if
   * the feature is disabled or no GPU backend is built in. Sets
   * gpu_cache_ / gpu_cache_bytes_ on success.
   *
   * @return true on success (or feature-disabled), false on allocation
   *         failure.
   */
  bool GpuCacheCreate();

  /** Free the GPU metadata cache region. Safe to call when disabled. */
  void GpuCacheDestroy();

  /**
   * Add or update a blob entry in the cache. Called from PutBlob right
   * after the blob has been placed. Resolves the bdev_type of the
   * blob's first block via registered_targets_ + storage_devices_,
   * and either projects the blob into the cache (DRAM-tier targets) or
   * removes any stale projection (file/noop targets).
   *
   * The PutBlob call site stays a single line; all bdev-type lookup
   * logic lives here.
   */
  void GpuCacheOnPutBlob(const TagId &tag_id, const std::string &blob_name,
                         const BlobInfo &blob_info);

  /**
   * Resolve the bdev_type string ("ram", "hbm", "pinned", "file", "noop",
   * "") for a blob given its BlobInfo. Returns an empty string when
   * registered_targets_ has no entry for the first block's pool id.
   *
   * Caller must hold no locks (this acquires target_lock_ for read).
   */
  std::string GetBdevTypeForBlob(const BlobInfo &blob_info);

  /** Remove a blob entry. Called from DelBlob. */
  void GpuCacheOnDelBlob(const TagId &tag_id, const std::string &blob_name);

  /** Add (or refresh) a tag entry. Called from GetOrCreateTag. */
  void GpuCacheOnGetOrCreateTag(const TagId &tag_id,
                                const std::string &tag_name);

  /** Remove a tag entry and cascade-remove all blob entries owning it. */
  void GpuCacheOnDelTag(const TagId &tag_id);

  /**
   * Read-only accessor (host-side) for tests / diagnostics. Returns
   * the cache header pointer if enabled, else nullptr.
   */
  GpuMetadataCacheHeader *GetGpuCache() const { return gpu_cache_; }

  // Telemetry ring buffer for performance monitoring
  static inline constexpr size_t kTelemetryRingSize = 1024; // Ring buffer size
  std::unique_ptr<ctp::ipc::circular_mpsc_ring_buffer<CteTelemetry, ctp::ipc::MallocAllocator>> telemetry_log_;
  std::atomic<std::uint64_t>
      telemetry_counter_; // Atomic counter for logical time

  // Write-Ahead Transaction Logs (per-worker)
  std::vector<std::unique_ptr<TransactionLog>> blob_txn_logs_;
  std::vector<std::unique_ptr<TransactionLog>> tag_txn_logs_;

  /**
   * Get access to configuration manager
   */
  const Config &GetConfig() const;

  /**
   * Helper function to get manual score for a target from storage device config
   * @param target_name Name of the target to look up
   * @return Manual score (0.0-1.0) if configured, -1.0f if not set (use
   * automatic)
   */
  float GetManualScoreForTarget(const std::string &target_name);

  /**
   * Get the persistence level for a target from its storage device config
   */
  clio::run::bdev::PersistenceLevel GetPersistenceLevelForTarget(
      const std::string &target_name);

  /**
   * Helper function to get or assign a tag ID
   */
  TagId GetOrAssignTagId(const std::string &tag_name,
                         const TagId &preferred_id = TagId::GetNull());

  /**
   * Get-or-create the chain of tags for an absolute path, returning the id of
   * the deepest tag. "/a/b/c" creates "/", "/a", "/a/b", "/a/b/c" (each child
   * stored relative to its parent as "$tagid{parent}/leaf") and returns the id
   * of "/a/b/c". Non-absolute names are created as a single flat tag.
   * preferred_id (if set) is applied to the deepest tag only.
   */
  TagId GetOrCreateTagChain(const std::string &name,
                            const TagId &preferred_id = TagId::GetNull());

  /**
   * Resolve an absolute path to an existing tag id by walking the hierarchy
   * (no creation). Returns TagId::GetNull() if any component is missing.
   * Must be called while holding tag_map_lock_.
   */
  TagId ResolvePathToIdLocked(const std::string &path);

  /**
   * Expand a stored (possibly relative "$tagid{parent}/leaf") tag name into its
   * full absolute name by recursively resolving parent references against
   * tag_id_to_info_. Flat names and the root "/" resolve to themselves. Lock
   * free: callers iterate the tag maps with the same discipline used elsewhere.
   * `depth` guards against pathological/cyclic references.
   */
  std::string ResolveTagName(const std::string &stored_name, int depth = 0);

  /**
   * Rebuild tag_search_ from scratch out of tag_id_to_info_. Order-independent
   * (every tag's parents are present), so it is the safe way to repopulate the
   * index after WAL/metadata restore. Caller must hold tag_map_lock_ (write).
   */
  void RebuildTagSearchIndexLocked();

  /**
   * Helper function to generate a new TagId using node_id as major and atomic
   * counter as minor
   */
  TagId GenerateNewTagId();

  /**
   * Get target lock index based on TargetId hash
   */
  size_t GetTargetLockIndex(const clio::run::PoolId &target_id) const;

  /**
   * Get tag lock index based on tag name hash
   */
  size_t GetTagLockIndex(const std::string &tag_name) const;

  /**
   * Get tag lock index based on tag ID hash
   */
  size_t GetTagLockIndex(const TagId &tag_id) const;

  /**
   * Allocate space from a target for new blob data
   * @param target_info Target to allocate from
   * @param size Size to allocate
   * @param allocated_offset Output parameter for allocated offset
   * @param success Output parameter: true if allocation succeeded, false otherwise
   * Returns TaskResume for coroutine-based async operations
   */
  clio::run::TaskResume AllocateFromTarget(TargetInfo &target_info, clio::run::u64 size,
                                     clio::run::u64 &allocated_offset, bool &success);

  /**
   * Free all blocks from a blob back to their respective targets
   * @param blob_info BlobInfo containing blocks to free
   * @param error_code Output: 0 on success, non-zero on error
   * Returns TaskResume for coroutine-based async operations
   */
  clio::run::TaskResume FreeAllBlobBlocks(BlobInfo &blob_info, clio::run::u32 &error_code);

  /**
   * Check if blob exists and return pointer to BlobInfo if found
   * @param blob_name Blob name to search for (required)
   * @param tag_id Tag ID to search within
   * @return Pointer to BlobInfo if found, nullptr if not found
   */
  std::shared_ptr<BlobInfo> CheckBlobExists(const std::string &blob_name, const TagId &tag_id);

  /**
   * Create new blob with given parameters
   * @param blob_name Name for the new blob (required)
   * @param tag_id Tag ID to associate blob with
   * @param blob_score Score/priority for the blob
   * @return Pointer to created BlobInfo, nullptr on failure
   */
  std::shared_ptr<BlobInfo> CreateNewBlob(const std::string &blob_name, const TagId &tag_id,
                          float blob_score);

  /**
   * Clear all blocks from a blob if this is a full replacement.
   * Conditions: score in [0,1], offset == 0, size >= current blob size.
   * @param blob_info Blob to potentially clear
   * @param blob_score Score of the incoming put
   * @param offset Write offset
   * @param size Write size
   * @param cleared Output: true if blocks were cleared
   */
  clio::run::TaskResume ClearBlob(BlobInfo &blob_info, float blob_score,
                            clio::run::u64 offset, clio::run::u64 size, bool &cleared);

  /**
   * Extend blob by allocating new data blocks if offset + size > current size.
   * If offset + size <= current size, returns immediately (no-op).
   * Runs DPE to select targets, then allocates from bdev.
   * @param blob_info Blob to extend
   * @param offset Offset where data starts
   * @param size Size of data to write
   * @param blob_score Score for target selection
   * @param error_code Output: 0 for success, non-zero for failure
   * @param min_persistence_level Minimum persistence level for target filtering
   */
  clio::run::TaskResume ExtendBlob(BlobInfo &blob_info, clio::run::u64 offset, clio::run::u64 size,
                             float blob_score, clio::run::u32 &error_code,
                             int min_persistence_level = 0);

  /**
   * Resize a blob to exactly new_size: grow (allocate appended blocks via
   * ExtendBlob) or shrink (free trailing blocks, trim the boundary block).
   * new_size == 0 frees all blocks. Shared by PutBlob's replace path
   * (kCtePutReplace) and the explicit TruncateBlob op.
   * @param blob_info Blob to resize
   * @param new_size Target logical size in bytes
   * @param blob_score Score for target selection on grow
   * @param error_code Output: 0 for success, non-zero for failure
   * @param min_persistence_level Minimum persistence level for target filtering
   */
  clio::run::TaskResume ResizeBlob(BlobInfo &blob_info, clio::run::u64 new_size,
                             float blob_score, clio::run::u32 &error_code,
                             int min_persistence_level = 0);

  /**
   * Write data to existing blob blocks
   * @param blocks Vector of blob blocks to write to
   * @param data Pointer to data to write
   * @param data_size Size of data to write
   * @param data_offset_in_blob Offset within blob where data starts
   * @param error_code Output: 0 for success, 1 for failure
   * Returns TaskResume for coroutine-based async operations
   */
  // start_block_idx / start_block_offset_in_blob: an optional fast-path hint for
  // writes known to target the blob tail (e.g. an append). The scan then begins
  // at block `start_block_idx`, whose first byte sits at `start_block_offset_in_blob`
  // in the blob, instead of re-walking from block 0. Callers MUST pass a
  // consistent pair (start_block_offset_in_blob == sum of blocks[0..start_block_idx)
  // sizes) and only for writes whose region lies entirely at/after that offset;
  // the default (0,0) reproduces the exact full-scan behavior. This turns an
  // O(blocks) rescan per append into O(1), fixing the O(N^2) blowup on files
  // built by millions of tiny O_APPEND writes (generic/069).
  clio::run::TaskResume ModifyExistingData(const clio::run::priv::vector<BlobBlock> &blocks,
                                     ctp::ipc::ShmPtr<> data, size_t data_size,
                                     size_t data_offset_in_blob, clio::run::u32 &error_code,
                                     size_t start_block_idx = 0,
                                     size_t start_block_offset_in_blob = 0);

  /**
   * Read existing blob data from blocks
   * @param blocks Vector of blob blocks to read from
   * @param data Output buffer to read data into
   * @param data_size Size of data to read
   * @param data_offset_in_blob Offset within blob where reading starts
   * @param error_code Output: 0 for success, 1 for failure
   * Returns TaskResume for coroutine-based async operations
   */
  clio::run::TaskResume ReadData(const clio::run::priv::vector<BlobBlock> &blocks, ctp::ipc::ShmPtr<> data,
                           size_t data_size, size_t data_offset_in_blob, clio::run::u32 &error_code);

  /**
   * Model the wall time of a data transfer over the blocks overlapping
   * [offset, offset+size) WITHOUT performing it (I/O emulation, issue #747).
   * Bytes are aggregated per target; each target contributes
   * latency + bytes/bandwidth from its measured PerfMetrics (fallback
   * constants when a target is unknown or reports zero), and the result is
   * the max across targets — blocks on different targets transfer
   * concurrently, blocks on the same target serialize on the device.
   * @param blocks Block layout of the blob (snapshot or live under the
   *        write token)
   * @param offset Transfer start offset within the blob
   * @param size Transfer size in bytes
   * @param is_write true = use write latency/bandwidth, false = read
   * @return Modeled duration in nanoseconds (0 if no blocks overlap)
   */
  clio::run::u64 EstimateIoTimeNs(const clio::run::priv::vector<BlobBlock> &blocks,
                                  clio::run::u64 offset, clio::run::u64 size,
                                  bool is_write);

  /**
   * Log telemetry data for CTE operations
   * @param op Operation type
   * @param off Offset within blob
   * @param size Size of operation
   * @param tag_id Tag ID involved
   * @param mod_time Last modification time
   * @param read_time Last read time
   */
  void LogTelemetry(CteOp op, size_t off, size_t size, const TagId &tag_id,
                    const Timestamp &mod_time, const Timestamp &read_time);

  /**
   * Get telemetry queue size for monitoring
   * @return Current number of entries in telemetry queue
   */
  size_t GetTelemetryQueueSize();

  /**
   * Parse capacity string to bytes
   * @param capacity_str Capacity string (e.g., "1TB", "500GB", "100MB")
   * @return Capacity in bytes
   */
  clio::run::u64 ParseCapacityToBytes(const std::string &capacity_str);

  /**
   * Restore metadata from persistent log during restart
   */
  void RestoreMetadataFromLog();

  /**
   * Replay transaction logs on top of restored snapshot during restart
   */
  void ReplayTransactionLogs();

  /**
   * Retrieve telemetry entries for analysis (non-destructive peek)
   * @param entries Vector to store retrieved entries
   * @param max_entries Maximum number of entries to retrieve
   * @return Number of entries actually retrieved
   */
  size_t GetTelemetryEntries(std::vector<CteTelemetry> &entries,
                             size_t max_entries = 100);

  /**
   * Poll telemetry log (Method::kPollTelemetryLog)
   * @param task PollTelemetryLog task containing parameters and results
   * @param rctx Runtime context for task execution
   */
  clio::run::TaskResume PollTelemetryLog(clio::run::shared_ptr<PollTelemetryLogTask> &task);

  /**
   * Get blob score operation - returns the score of a blob
   * @param task GetBlobScore task containing blob lookup parameters and results
   * @param rctx Runtime context for task execution
   */
  clio::run::TaskResume GetBlobScore(clio::run::shared_ptr<GetBlobScoreTask> &task);

  /**
   * Get blob size operation - returns the size of a blob in bytes
   * @param task GetBlobSize task containing blob lookup parameters and results
   * @param rctx Runtime context for task execution
   */
  clio::run::TaskResume GetBlobSize(clio::run::shared_ptr<GetBlobSizeTask> &task);

  /**
   * Get contained blobs operation - returns all blob names in a tag
   * @param task GetContainedBlobs task containing tag ID and results
   * @param rctx Runtime context for task execution
   */
  clio::run::TaskResume GetContainedBlobs(clio::run::shared_ptr<GetContainedBlobsTask> &task);

  /**
   * Query tags by regex pattern (Method::kTagQuery)
   * @param task TagQuery task containing regex pattern and results
   * @param rctx Runtime context for task execution
   */
  clio::run::TaskResume TagQuery(clio::run::shared_ptr<TagQueryTask> &task);

  /**
   * Query blobs by tag and blob regex patterns (Method::kBlobQuery)
   * @param task BlobQuery task containing regex patterns and results
   * @param rctx Runtime context for task execution
   */
  clio::run::TaskResume BlobQuery(clio::run::shared_ptr<BlobQueryTask> &task);

  /**
   * BM25 keyword search over blob contents (Method::kSemanticSearch).
   * Filters by tag+blob regex like BlobQuery, then tokenizes each
   * candidate blob's bytes and scores them against query_text_ using
   * Okapi BM25 with corpus stats computed over the matched working
   * set. Returns top-k results sorted by descending score.
   */
  clio::run::TaskResume SemanticSearch(clio::run::shared_ptr<SemanticSearchTask> &task);

  /**
   * Timestamp-window search over blob metadata (Method::kTemporalSearch).
   * Pure metadata scan — no blob bytes are read. Filters by tag+blob
   * regex then returns blobs whose last_modified_ falls within
   * [time_begin_, time_end_], sorted by ascending last_modified_.
   */
  clio::run::TaskResume TemporalSearch(clio::run::shared_ptr<TemporalSearchTask> &task);

  /**
   * Get comprehensive blob metadata (Method::kGetBlobInfo)
   * Returns score, total size, and block placement information
   * @param task GetBlobInfo task containing blob lookup parameters and results
   * @param rctx Runtime context for task execution
   */
  clio::run::TaskResume GetBlobInfo(clio::run::shared_ptr<GetBlobInfoTask> &task);

  /**
   * Flush metadata to durable storage (Method::kFlushMetadata)
   */
  clio::run::TaskResume FlushMetadata(clio::run::shared_ptr<FlushMetadataTask> &task);

  /**
   * Flush data from volatile to non-volatile targets (Method::kFlushData)
   */
  clio::run::TaskResume FlushData(clio::run::shared_ptr<FlushDataTask> &task);

private:
  /**
   * Helper function to compute hash-based pool query for blob operations
   * @param tag_id Tag ID for the blob
   * @param blob_name Blob name
   * @return PoolQuery with DirectHash based on tag_id and blob_name
   */
  clio::run::PoolQuery HashBlobToContainer(const TagId &tag_id,
                                     const std::string &blob_name);
};

} // namespace clio::cte::core

#endif // WRPCTE_CORE_RUNTIME_H_
