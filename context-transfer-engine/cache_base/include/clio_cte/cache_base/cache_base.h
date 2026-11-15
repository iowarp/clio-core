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

#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
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
 * Module verb: drop every cached copy of one tag on THIS container (sent as
 * a broadcast by DelTag). Carries a core::DelTagTask.
 */
GLOBAL_CROSS_CONST clio::run::u32 kDropTagCopies = 100;
/** Highest method id + 1 across the policy cache modules. */
GLOBAL_CROSS_CONST clio::run::u32 kCacheMaxMethodId = 101;

/** A byte range [first, second) of a blob. */
using Extent = std::pair<clio::run::u64, clio::run::u64>;

/**
 * Add [begin, end) to a sorted list of disjoint ranges, merging any it
 * overlaps or touches.
 */
inline void AddExtent(std::vector<Extent> &ext, clio::run::u64 begin,
                      clio::run::u64 end) {
  if (begin >= end) return;
  std::vector<Extent> out;
  out.reserve(ext.size() + 1);
  bool placed = false;
  for (const Extent &e : ext) {
    if (e.second < begin) {
      out.push_back(e);
    } else if (end < e.first) {
      if (!placed) out.emplace_back(begin, end);
      placed = true;
      out.push_back(e);
    } else {
      begin = std::min(begin, e.first);
      end = std::max(end, e.second);
    }
  }
  if (!placed) out.emplace_back(begin, end);
  ext.swap(out);
}

/** Return codes the core uses when a tier cannot hold a put (11..13). */
inline bool IsNoSpaceRc(clio::run::u32 rc) { return rc >= 11 && rc <= 13; }

/**
 * Persistent-tier caches (core_persist_write_thru / core_persist_write_back).
 *
 * The volatile core and the persistent chain place a blob on the SAME owner
 * container (same hash), and every blob verb routes to that owner, so each
 * blob has exactly one volatile copy in the cluster, kept by its owner's
 * container. All state here is owner-local and every operation on a blob
 * is serialized by a per-blob lock: no cross-node coherence is needed.
 *
 *  - write-through: a put goes to persistent, then the volatile copy (if
 *    any) is updated; reads try volatile, then persistent (and fill).
 *  - write-back: a put goes to volatile and its range is logged dirty; the
 *    dirty log is flushed to persistent periodically, on SyncTag, before
 *    eviction, and before any verb that must see persistent state.
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

  /** Owner routing for blob verbs; FlushData and DropTagCopies stay put. */
  clio::run::PoolQuery ScheduleTask(
      const clio::run::shared_ptr<clio::run::Task> &task) override;

  clio::run::TaskResume Destroy(
      clio::run::shared_ptr<clio::run::admin::DestroyTask> &task);
  clio::run::TaskResume Monitor(
      clio::run::shared_ptr<clio::run::admin::MonitorTask> &task);
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
  /** kDropTagCopies: drop this container's copies of a tag. */
  clio::run::TaskResume DropTagCopies(
      clio::run::shared_ptr<clio::cte::core::DelTagTask> &task);
  /**
   * Every other core verb: flush the dirty data it must observe (the blob,
   * or the tag), drop copies a device-side write would make stale, then
   * forward to the persistent chain.
   */
  clio::run::TaskResume ForwardCoherent(
      clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> task);

 protected:
  /** Bind both targets, start the periodic flusher (write-back). */
  void InitCache(const PolicyCacheParams &params,
                 const clio::run::PoolId &self_pool);
  bool IsWriteBack() const { return policy_ == CachePolicy::kWriteBack; }

 private:
  /** One volatile copy this owner keeps (always the whole blob). */
  struct Entry {
    TagId tag_;
    std::string name_;
    clio::run::u64 size_ = 0;      ///< logical size of the copy
    std::vector<Extent> dirty_;    ///< ranges newer than persistent
    clio::run::u64 last_ns_ = 0;   ///< LRU
  };

  static std::string Key(const TagId &tag, const std::string &name);
  /** Serialize operations on one blob (yields while another holds it). */
  clio::run::TaskResume Lock(std::string key);
  void Unlock(const std::string &key);
  bool Cached(const std::string &key, clio::run::u64 *size = nullptr);
  void Touch(const std::string &key);
  bool ShouldFill(const std::string &key);

  /** Write the blob's dirty ranges to persistent (lock held). */
  clio::run::TaskResume Flush(std::string key, clio::run::u32 *rc);
  /** Flush (unless discarding) and drop the volatile copy (lock held). */
  clio::run::TaskResume Drop(std::string key, bool flush, clio::run::u32 *rc);
  /** Delete a blob from volatile whether or not it is recorded here. */
  clio::run::TaskResume DropVolatile(TagId tag, std::string name);
  /** Flush and drop least-recently-used unlocked copies. */
  clio::run::TaskResume Evict(std::string keep, clio::run::u32 n);
  /** Copy the whole persistent blob into volatile (lock held). */
  clio::run::TaskResume FillFromPersistent(TagId tag, std::string name,
                                           clio::run::u64 size,
                                           clio::run::u32 *rc);
  /** Put a task's regions into volatile, evicting once on no-space. */
  clio::run::TaskResume PutVolatile(
      clio::run::shared_ptr<clio::cte::core::PutBlobTask> task,
      clio::run::u32 *rc);
  /** Put one record (batch) into volatile, evicting once on no-space. */
  clio::run::TaskResume PutVolatileRaw(TagId tag, std::string name,
                                       clio::run::u64 off, clio::run::u64 size,
                                       const char *src, clio::run::u32 *rc);
  /**
   * Write-back absorb of one write to (tag, name) covering [off, end):
   * fills a partial write to a persistent-only blob first, then `put`
   * writes the volatile copy; falls back to persistent when volatile is
   * full (lock held).
   */
  clio::run::TaskResume PrepareAbsorb(TagId tag, std::string name,
                                      clio::run::u64 off, clio::run::u64 end,
                                      clio::run::u32 *rc);
  void NoteWrite(const std::string &key, const TagId &tag,
                 const std::string &name, clio::run::u64 off,
                 clio::run::u64 end, bool dirty);
  /** Keys of this container's copies (of one tag, or all). */
  std::vector<std::string> KeysOf(const TagId *tag);
  /** Flush every dirty copy of a tag (or all) on this container. */
  clio::run::TaskResume FlushAll(TagId tag, bool all_tags, clio::run::u32 *rc);
  void RegisterStopFlush(const clio::run::PoolId &self_pool);

  CachePolicy policy_;
  PolicyCacheParams params_;
  clio::cte::core::Client vol_client_;   ///< volatile core (owner-local)
  clio::cte::core::Client pers_client_;  ///< persistent chain (owner-local)
  std::mutex mu_;                         ///< guards entries_ / busy_
  std::unordered_map<std::string, Entry> entries_;
  std::unordered_set<std::string> busy_;
  /** on_hit_count fill: misses per uncached key. */
  std::unordered_map<std::string, clio::run::u32> miss_counts_;
  clio::run::u64 stop_hook_id_ = 0;
};

}  // namespace clio::cte::cache_base

#endif  // CLIO_CTE_CACHE_BASE_CACHE_BASE_H_
