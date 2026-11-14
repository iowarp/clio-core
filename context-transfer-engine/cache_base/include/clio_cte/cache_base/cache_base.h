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

#ifndef CLIO_CTE_CACHE_BASE_CACHE_BASE_H_
#define CLIO_CTE_CACHE_BASE_CACHE_BASE_H_

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/admin/admin_tasks.h>
#include <clio_cte/core/blob_batch.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_interposer.h>
#include <clio_cte/cache_base/cache_config.h>

namespace clio::cte::cache_base {

using TagId = clio::cte::core::TagId;
using Context = clio::cte::core::Context;

/**
 * Module verb shared by both policy caches: drop every cached copy of one
 * tag on THIS container (sent as a broadcast by DelTag). Carries a
 * core::DelTagTask; numbered above the core's id space (interposer rule).
 */
GLOBAL_CROSS_CONST clio::run::u32 kDropTagCopies = 100;
/** Highest method id + 1 across the policy cache modules. */
GLOBAL_CROSS_CONST clio::run::u32 kCacheMaxMethodId = 101;

/** Return codes the core uses when a tier cannot hold a put (11..13). */
inline bool IsNoSpaceRc(clio::run::u32 rc) { return rc >= 11 && rc <= 13; }

/**
 * One volatile copy the cache tracks. A copy is either ABSENT (no entry) or
 * COMPLETE: it is created whole (a put covering the blob, or a fill) and
 * only removed whole, so a present copy is always the full blob.
 */
struct CacheEntry {
  TagId tag_;
  std::string name_;
  clio::run::u64 size_ = 0;            ///< logical size of the copy
  bool dirty_ = false;                 ///< newer than the persistent blob
  clio::run::u64 gen_ = 0;             ///< bumped by every write to the copy
  clio::run::u32 pins_ = 0;            ///< ops reading/writing the copy now
  bool busy_ = false;                  ///< being filled, evicted or dropped
  bool flushing_ = false;              ///< a flush is copying it out
  clio::run::u64 last_access_ns_ = 0;  ///< for LRU
  clio::run::u64 hits_ = 0;            ///< for LFU
};

/** Result of trying to pin a copy. */
enum class PinState {
  kCached,  ///< present and pinned: the caller must Unpin
  kAbsent,  ///< no copy
  kBusy,    ///< being filled/evicted/dropped: yield and retry
};

/**
 * Shared runtime of the policy-named caches (clio_cte_write_thru_cache and
 * clio_cte_write_back_cache).
 *
 * The cache sits above two DISCONNECTED cores: a volatile core reached with
 * an ordinary client at PoolQuery::Local(), and a persistent chain that is
 * the interposer's next pool (every verb this class does not handle is
 * forwarded there). All policy state lives here: the map of volatile copies
 * with their dirty bits, eviction (DelBlob on the volatile core), fill on
 * read miss, flushing, and the volatile-full fallback.
 *
 * Blob verbs route to the blob's owner container exactly as the core does,
 * so every operation on one blob passes through one cache container — the
 * coherence argument of the design rests on that.
 */
class CacheBase : public clio::cte::core::DualTargetInterposer {
 public:
  explicit CacheBase(CachePolicy policy) : policy_(policy) {}
  ~CacheBase() override;

  /** Size-driven cost estimate, same shape as the other interposers. */
  clio::run::TaskStat GetTaskStats(
      const clio::run::Task *task) const override {
    return clio::cte::core::CoreInterposer::GetTaskStats(task);
  }

  /** Owner routing for blob verbs; FlushData and DropTagCopies stay local. */
  clio::run::PoolQuery ScheduleTask(
      const clio::run::shared_ptr<clio::run::Task> &task) override;

  /** Container lifecycle no-ops (the cache keeps only in-memory state). */
  clio::run::TaskResume Destroy(
      clio::run::shared_ptr<clio::run::admin::DestroyTask> &task);
  clio::run::TaskResume Monitor(
      clio::run::shared_ptr<clio::run::admin::MonitorTask> &task);

  // ---- Interposed core verbs ----
  clio::run::TaskResume PutBlob(
      clio::run::shared_ptr<clio::cte::core::PutBlobTask> &task);
  clio::run::TaskResume GetBlob(
      clio::run::shared_ptr<clio::cte::core::GetBlobTask> &task);
  clio::run::TaskResume GetBlobSize(
      clio::run::shared_ptr<clio::cte::core::GetBlobSizeTask> &task);
  clio::run::TaskResume DelBlob(
      clio::run::shared_ptr<clio::cte::core::DelBlobTask> &task);
  clio::run::TaskResume DelTag(
      clio::run::shared_ptr<clio::cte::core::DelTagTask> &task);
  clio::run::TaskResume TruncateBlob(
      clio::run::shared_ptr<clio::cte::core::TruncateBlobTask> &task);
  clio::run::TaskResume MultiPutBlob(
      clio::run::shared_ptr<clio::cte::core::MultiPutBlobTask> &task);
  clio::run::TaskResume SyncTag(
      clio::run::shared_ptr<clio::cte::core::SyncTagTask> &task);
  clio::run::TaskResume FlushData(
      clio::run::shared_ptr<clio::cte::core::FlushDataTask> &task);
  /** Module verb kDropTagCopies (see its declaration). */
  clio::run::TaskResume DropTagCopies(
      clio::run::shared_ptr<clio::cte::core::DelTagTask> &task);

  /**
   * Every other core verb: make the persistent side current where the verb
   * reads or changes blob state (flush dirty copies; drop copies a mutation
   * would make stale), then forward it to the persistent chain verbatim.
   * @param method core method id
   * @param task the task
   */
  clio::run::TaskResume ForwardCoherent(
      clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> task);

  /** Copies currently dirty (for tests and Monitor). */
  size_t DirtyCount();

 protected:
  /**
   * Bind targets and start the policy's background work. Call from the
   * derived Create once the config is known.
   * @param params the module config
   * @param self_pool this cache's pool id
   */
  void InitCache(const PolicyCacheParams &params,
                 const clio::run::PoolId &self_pool);

 private:
  using EntryMap = std::unordered_map<std::string, CacheEntry>;
  using KeyFilter = std::function<bool(const CacheEntry &)>;

  static std::string Key(const TagId &tag, const std::string &name);
  bool IsWriteBack() const { return policy_ == CachePolicy::kWriteBack; }

  // ---- Map state (mu_ held only inside these, never across co_await) ----
  PinState Pin(const std::string &key);
  void Unpin(const std::string &key);
  /** Record a write into the copy (creating the entry if absent). */
  void NoteWrite(const std::string &key, const TagId &tag,
                 const std::string &name, clio::run::u64 end, bool dirty);
  /** Mark an absent key busy for a fill or create; false if present/busy. */
  bool Reserve(const std::string &key, const TagId &tag,
               const std::string &name);
  /** Release a reservation; keep the entry only when the copy was written. */
  void ReleaseReserve(const std::string &key, bool written,
                      clio::run::u64 size, bool dirty);
  /** Mark a present, unpinned, idle copy busy; false otherwise. */
  bool MarkBusy(const std::string &key);
  /** Erase an entry (the copy has been deleted). */
  void Erase(const std::string &key);
  /** Keys of entries matching `filter` (dirty-only when asked). */
  std::vector<std::string> Collect(const KeyFilter &filter, bool dirty_only);
  /** Whether a fill should happen for this miss (fill mode bookkeeping). */
  bool ShouldFill(const std::string &key);

  // ---- Volatile / persistent I/O helpers ----
  clio::run::TaskResume VolatilePutRegions(
      clio::run::shared_ptr<clio::cte::core::PutBlobTask> task,
      clio::run::u32 &rc);
  clio::run::TaskResume VolatileGetRegions(
      clio::run::shared_ptr<clio::cte::core::GetBlobTask> task,
      clio::run::u32 &rc);
  clio::run::TaskResume PersistentSize(TagId tag, std::string name,
                                       clio::run::u64 &size,
                                       clio::run::u32 &rc);
  clio::run::TaskResume DeleteVolatileCopy(TagId tag, std::string name);
  /**
   * Copy a blob's first `total` bytes from the persistent chain into the
   * volatile core, chunk by chunk, evicting (MakeRoom) once if the volatile
   * core is full. The caller holds the entry's reservation.
   * @param tag the blob's tag
   * @param name the blob's name
   * @param total bytes to copy (the persistent blob's size)
   * @param rc 0, or the failing get/put's return code
   */
  clio::run::TaskResume CopyPersistentToVolatile(TagId tag, std::string name,
                                                 clio::run::u64 total,
                                                 clio::run::u32 &rc);

  // ---- Policy steps ----
  clio::run::TaskResume WriteThruPut(
      clio::run::shared_ptr<clio::cte::core::PutBlobTask> task);
  clio::run::TaskResume WriteBackPut(
      clio::run::shared_ptr<clio::cte::core::PutBlobTask> task);
  clio::run::TaskResume WriteBackCreate(
      clio::run::shared_ptr<clio::cte::core::PutBlobTask> task,
      std::string key, clio::run::u64 end, clio::run::u32 &outcome);
  clio::run::TaskResume WriteBackFallback(
      clio::run::shared_ptr<clio::cte::core::PutBlobTask> task,
      std::string key);
  clio::run::TaskResume MirrorIntoCopy(
      clio::run::shared_ptr<clio::cte::core::PutBlobTask> task,
      std::string key, clio::run::u64 end);
  clio::run::TaskResume Fill(TagId tag, std::string name,
                             ctp::ipc::ShmPtr<> whole_buf,
                             clio::run::u64 whole_len);
  clio::run::TaskResume FlushEntry(std::string key, clio::run::u32 &rc);
  clio::run::TaskResume FlushKeys(std::vector<std::string> keys,
                                  clio::run::u32 &rc);
  /** Wait until the copy is idle, then flush it (optionally) and delete it. */
  clio::run::TaskResume DropCopy(std::string key, bool flush_first,
                                 clio::run::u32 &rc);
  /** Flush (optionally) and delete a copy the caller already marked busy. */
  clio::run::TaskResume DropBusyCopy(std::string key, bool flush_first,
                                     clio::run::u32 &rc);
  clio::run::TaskResume DropKeys(std::vector<std::string> keys,
                                 bool flush_first);
  clio::run::TaskResume MakeRoom(clio::run::u64 needed,
                                 clio::run::u64 &freed);

  /** Register the stop hook that flushes every dirty copy (write-back). */
  void RegisterStopFlush(const clio::run::PoolId &self_pool);

  CachePolicy policy_;
  PolicyCacheParams params_;
  clio::cte::core::Client vol_client_;   ///< volatile core, Local queries
  clio::cte::core::Client pers_client_;  ///< persistent chain, Dynamic
  std::mutex mu_;
  EntryMap entries_;
  std::unordered_map<std::string, clio::run::u32> miss_counts_;
  clio::run::u64 stop_hook_id_ = 0;
};

}  // namespace clio::cte::cache_base

#endif  // CLIO_CTE_CACHE_BASE_CACHE_BASE_H_
