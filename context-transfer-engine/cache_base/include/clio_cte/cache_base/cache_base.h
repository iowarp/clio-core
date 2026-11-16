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
#include <atomic>
#include <deque>
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
/** kDropTagCopies modes, carried in DelTagTask::posix_unlink_. */
GLOBAL_CROSS_CONST clio::run::u32 kCopiesDiscard = 0;    ///< drop, dirty lost (delete)
GLOBAL_CROSS_CONST clio::run::u32 kCopiesFlushDrop = 1;  ///< flush, then drop
GLOBAL_CROSS_CONST clio::run::u32 kCopiesFlush = 2;      ///< flush, keep
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
  /** kDropTagCopies: flush and/or drop this container's copies of a tag
   *  (null tag: every copy), per the mode in posix_unlink_. */
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
  /** The outputs of a SyncTag, as the sharers of its round receive them. */
  struct SyncOut {
    clio::run::u32 deferred_ = 0;
    clio::run::u64 blobs_moved_ = 0;
    clio::run::u64 bdevs_synced_ = 0;
    clio::run::u32 containers_ = 0;
    clio::run::u64 liveness_change_ns_ = 0;
  };
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
  /**
   * Make room for a write by dropping clean copies only: with write-around
   * on, a put that still finds no room is written through itself rather
   * than flushing other blobs' dirty bytes inline (a full cache would
   * otherwise pay a synchronous flush per put).
   */
  bool EvictCleanOnly() const {
    return IsWriteBack() && params_.fallback_to_persistent_;
  }

  /** Write the blob's dirty ranges to persistent (lock held). */
  clio::run::TaskResume Flush(std::string key, clio::run::u32 *rc);
  /** Flush (unless discarding) and drop the volatile copy (lock held). */
  clio::run::TaskResume Drop(std::string key, bool flush, clio::run::u32 *rc);
  /** Delete a blob from volatile whether or not it is recorded here. */
  clio::run::TaskResume DropVolatile(TagId tag, std::string name);
  /**
   * Flush and drop least-recently-used unlocked copies.
   * @param keep key never dropped (the blob being written)
   * @param n copies to drop at most
   * @param clean_only drop only copies with no dirty range
   * @param done set to the number of copies dropped
   * @param need_bytes when not 0: stop once the dropped copies held this
   *        many bytes (at least one copy is dropped)
   */
  clio::run::TaskResume Evict(std::string keep, clio::run::u32 n,
                              bool clean_only, clio::run::u32 *done,
                              clio::run::u64 need_bytes = 0);
  /**
   * Rebuild the eviction list: the least-recently-used unlocked copies
   * (clean ones only when clean_only), at most kVictimQueue of them, so
   * that most evictions take a victim without scanning every copy.
   */
  void RefillVictims(const std::string &keep, bool clean_only);
  /**
   * Take and lock the next copy of the eviction list that can still be
   * dropped.
   * @param key set to its key (its lock is held on return)
   * @param size set to the bytes it holds
   * @return false when the list is empty
   */
  bool TakeVictim(const std::string &keep, bool clean_only, std::string *key,
                  clio::run::u64 *size);
  /** Record that volatile refused a put of `size` bytes for lack of room. */
  void NoteNoSpace(clio::run::u64 size);
  /** Record that a volatile copy was dropped (its space is free again). */
  void NoteSpaceFreed() { full_size_.store(0, std::memory_order_relaxed); }
  /** Record that a copy became clean (a clean-only eviction can take it). */
  void NoteClean() { no_clean_.store(false, std::memory_order_relaxed); }
  /**
   * True while volatile is known to refuse a put of `size` bytes. The
   * knowledge expires (kFullProbeNs), after which one put probes again.
   */
  bool KnownFull(clio::run::u64 size);
  /**
   * Before absorbing a write-back put to a blob with no volatile copy:
   * when volatile is known full, make room by dropping clean copies, or
   * report that the put should go straight to persistent because every
   * copy is dirty (no failed volatile put, no eviction scan).
   * @param key the blob's key (lock held)
   * @param size bytes the put writes
   * @param bypass set true when the put should skip volatile
   */
  clio::run::TaskResume MakeRoom(std::string key, clio::run::u64 size,
                                 bool *bypass);
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
  /**
   * Fill-level flush trigger (flush_fill_pct): when there is unflushed
   * data and the volatile core is at least that full, start one flush of
   * every dirty copy on this container. Called after writes; it asks the
   * volatile core for its fill level at most once per kFillCheckNs.
   */
  clio::run::TaskResume CheckFill();
  /** Keys and dirty byte counts of this container's dirty copies (of one
   *  tag, or all), sorted by key. */
  std::vector<std::pair<std::string, clio::run::u64>> DirtyKeysOf(
      const TagId *tag);
  /**
   * Write the dirty ranges of several blobs to persistent at once (every
   * key's lock held): all reads from volatile are issued together, then
   * all writes to persistent.
   * @param keys the blobs to flush
   * @param rc set to the first failure, else 0
   */
  clio::run::TaskResume FlushBatch(std::vector<std::string> keys,
                                   clio::run::u32 *rc);
  /** Flush every dirty copy of a tag (or all) on this container, several
   *  blobs at a time. */
  clio::run::TaskResume FlushAll(TagId tag, bool all_tags, clio::run::u32 *rc);
  /**
   * Flush this container's dirty copies of one tag; concurrent callers
   * share one flush (see JoinRound).
   */
  clio::run::TaskResume FlushTagShared(TagId tag, clio::run::u32 *rc);
  /**
   * Join the shared work named `id`: at most one round of it runs at a
   * time, and every caller that arrives while a round runs shares the NEXT
   * round (which starts after the caller arrived, so it covers everything
   * the caller needs covered).
   * @param id name of the work (e.g. sync of one tag)
   * @param leader set true when the caller must do the work and then call
   *        LeaveRound; false when a round that covers it has completed
   * @param rc when not leader: the result of the round that covered it
   * @param out when not leader and not null: that round's SyncTag outputs
   */
  clio::run::TaskResume JoinRound(std::string id, bool *leader,
                                  clio::run::u32 *rc, SyncOut *out = nullptr);
  /** End the round the caller led, publishing its result (and, for a sync,
   *  its SyncTag outputs) to the sharers. */
  void LeaveRound(const std::string &id, clio::run::u32 rc,
                  const SyncOut *out = nullptr);
  void RegisterStopFlush(const clio::run::PoolId &self_pool);

  /**
   * The container that applies this cache's policy to a blob: its owner
   * (BlobHash, with the core's failover). Tasks reach this pool on other
   * containers too -- an interposer above forwards to its LOCAL container
   * -- and are sent on to the owner, which holds the blob's only copy.
   */
  clio::run::u32 OwnerOf(const TagId &tag, const std::string &name);
  /** Send a put (regions one by one when vectored) to the blob's owner. */
  clio::run::TaskResume PutAtOwner(
      clio::run::shared_ptr<clio::cte::core::PutBlobTask> task,
      clio::run::u32 owner);
  /** Send a get (regions one by one when vectored) to the blob's owner. */
  clio::run::TaskResume GetAtOwner(
      clio::run::shared_ptr<clio::cte::core::GetBlobTask> task,
      clio::run::u32 owner);
  /**
   * Apply a kDropTagCopies mode to a tag's copies (null: all) everywhere.
   * @param rc set to the first container's flush failure, else 0
   */
  clio::run::TaskResume TagCopiesEverywhere(TagId tag, clio::run::u32 mode,
                                            clio::run::u32 *rc);

  CachePolicy policy_;
  PolicyCacheParams params_;
  clio::cte::core::Client self_client_;  ///< this cache's pool (to owners)
  clio::cte::core::Client vol_client_;   ///< volatile core (owner-local)
  clio::cte::core::Client pers_client_;  ///< persistent chain (owner-local)
  std::mutex mu_;                         ///< guards entries_ / busy_
  std::unordered_map<std::string, Entry> entries_;
  std::unordered_set<std::string> busy_;
  /** Copies kept on the eviction list between scans. */
  static constexpr size_t kVictimQueue = 256;
  /** Eviction list (RefillVictims / TakeVictim). Guarded by mu_. */
  std::deque<std::string> victims_;
  /** on_hit_count fill: misses per uncached key. */
  std::unordered_map<std::string, clio::run::u32> miss_counts_;
  clio::run::u64 stop_hook_id_ = 0;
  /** Smallest put volatile refused for lack of room (0: not known full). */
  std::atomic<clio::run::u64> full_size_{0};
  /** When full_size_ was last set (ns). */
  std::atomic<clio::run::u64> full_ns_{0};
  /** The last clean-only eviction found no clean copy to drop. */
  std::atomic<bool> no_clean_{false};
  /** Counts writes that left a dirty range; flushed_gen_ is its value when
   *  the last complete flush of everything began (equal: nothing dirty). */
  std::atomic<clio::run::u64> dirty_gen_{0};
  std::atomic<clio::run::u64> flushed_gen_{0};
  /** When the fill level was last asked for (ns). */
  std::atomic<clio::run::u64> fill_check_ns_{0};
  /** A fill-level flush has been sent and has not finished. */
  std::atomic<bool> fill_flush_pending_{false};
  /** One kind of shared work (JoinRound). Guarded by mu_. */
  struct Round {
    clio::run::u64 started_ = 0;  ///< rounds started
    clio::run::u64 done_ = 0;     ///< rounds completed
    bool running_ = false;
    clio::run::u32 rc_ = 0;       ///< result of the last completed round
    clio::run::u32 users_ = 0;    ///< callers inside JoinRound..LeaveRound
    SyncOut out_;                 ///< SyncTag outputs of that round
  };
  std::unordered_map<std::string, Round> rounds_;
};

}  // namespace clio::cte::cache_base

#endif  // CLIO_CTE_CACHE_BASE_CACHE_BASE_H_
