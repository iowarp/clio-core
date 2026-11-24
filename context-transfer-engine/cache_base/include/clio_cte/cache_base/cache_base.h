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

#include <map>
#include <set>
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

  /**
   * Per-task cost estimate for every method a policy cache serves.
   *
   * Core verbs take the core's estimate (CoreInterposer::GetTaskStats). A
   * write-through put writes the persistent chain AND refreshes the volatile
   * copy, so it adds one memory-tier copy of the payload on top; a
   * write-back put writes the volatile copy only (the flusher pays the
   * persistent write later, under FlushData / SyncTag / DropTagCopies).
   * kDropTagCopies flushes a tag's dirty copies (kCopiesFlush*) or just
   * forgets them (kCopiesDiscard); its volume is unknown up front.
   * @param task the task to estimate
   * @return the task's cost features
   */
  clio::run::TaskStat GetTaskStats(
      const clio::run::Task *task) const override {
    namespace cost = clio::run::cost;
    namespace core = clio::cte::core;
    if (task == nullptr) {
      return clio::run::TaskStat();
    }
    clio::run::TaskStat stat = core::CoreInterposer::GetTaskStats(task);
    clio::run::u64 bytes = 0;
    switch (task->method_) {
      case kDropTagCopies: {
        const auto *t = static_cast<const core::DelTagTask *>(task);
        return t->posix_unlink_ == kCopiesDiscard ? cost::Fixed(20, 50.0f)
                                                  : cost::Fixed(50, 500.0f);
      }
      case core::Method::kPutBlob:
        bytes = core::CoreBlobBytes(static_cast<const core::PutBlobTask *>(task));
        break;
      case core::Method::kMultiPutBlob:
        bytes = static_cast<const core::MultiPutBlobTask *>(task)->data_len_;
        break;
      default:
        return stat;
    }
    if (!IsWriteBack()) {
      const clio::run::TaskStat copy =
          cost::Data(bytes, cost::kCopyBytesPerCpuUs,
                     cost::kMemoryBytesPerWallUs, 0);
      stat.compute_ += copy.compute_;
      stat.wall_time_ += copy.wall_time_;
    }
    return stat;
  }

  /** Owner routing for blob verbs; FlushData and DropTagCopies stay put. */
  clio::run::PoolQuery ScheduleTask(
      const clio::run::shared_ptr<clio::run::Task> &task) override;

  /**
   * Module Destroy (kDestroy), honoring the admin destroy flags:
   * write-back: stop buffering (draining_), wait for buffered puts, flush
   * every dirty copy to persistent (unless NoFlush, which discards them),
   * then on a final destroy drop the stop hook and, with DeleteData, every
   * volatile copy. QuiesceOnly keeps the pool and its hook. Idempotent: a
   * second call finds nothing dirty. A nonzero return keeps the pool running.
   * @param task the destroy task (flags in, flush counts and error out)
   * @return the coroutine resume handle
   */
  clio::run::TaskResume Destroy(
      clio::run::shared_ptr<clio::run::admin::DestroyTask> &task);
  /**
   * Where tasks addressed to this pool go once it is removed: the persistent
   * chain (the write-back or write-through cache's data lives there).
   * @return the persistent chain's pool, or a null id without one
   */
  clio::run::PoolId GetUnplugSuccessor() const override {
    return HasPersistent() ? CorePoolId() : clio::run::PoolId::GetNull();
  }
  /**
   * Monitor queries: "stats" answers a msgpack map of the policy, draining
   * state, dirty and cached totals and every counter (see PackStats);
   * "dirty_count" answers the number of dirty blobs as msgpack.
   */
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

  /**
   * One write-back put's claim on buffering (buffering_), taken before the
   * put decides where its bytes go. Constructing it counts the put as
   * buffering and then reads draining_ (both seq_cst, a Dekker pair with
   * BeginDrain). A put that finds the cache draining gives its count back
   * at once and takes the ordered path; a put that keeps the claim releases
   * it when it is destroyed, so every CLIO_CO_RETURN gives it back.
   */
  class BufferSlot {
   public:
    /**
     * @param buffering the cache's count of buffering puts
     * @param draining the cache's draining flag
     * @param write_back false for write-through: never admitted
     */
    BufferSlot(std::atomic<clio::run::u64> *buffering,
               const std::atomic<bool> *draining, bool write_back)
        : buffering_(buffering) {
      if (!write_back) return;
      buffering_->fetch_add(1, std::memory_order_seq_cst);
      if (draining->load(std::memory_order_seq_cst)) {
        buffering_->fetch_sub(1, std::memory_order_seq_cst);
        return;
      }
      held_ = true;
    }
    ~BufferSlot() {
      if (held_) buffering_->fetch_sub(1, std::memory_order_seq_cst);
    }
    BufferSlot(const BufferSlot &) = delete;
    BufferSlot &operator=(const BufferSlot &) = delete;
    /** @return true when the put may buffer its bytes in volatile */
    bool Admitted() const { return held_; }

   private:
    std::atomic<clio::run::u64> *buffering_;
    bool held_ = false;
  };

  /** Totals of one flush: the blobs and bytes it made durable. */
  struct FlushStats {
    clio::run::u64 blobs_ = 0;
    clio::run::u64 bytes_ = 0;
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
   * Continuous flush trigger (flush_dirty_mb): once that much new dirty
   * data arrived since the last triggered flush started, start another.
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
  /** One dirty range of a flush in flight (FlushAll). */
  struct FlushOp {
    size_t key_ = 0;  ///< index into the flush's FlushKey list
    TagId tag_;
    std::string name_;
    Extent ext_;
    ctp::ipc::FullPtr<char> buf_;
    clio::run::u32 owner_ = 0;  ///< the blob's owner (where it lands)
    clio::run::Future<core::GetBlobTask> get_;
    clio::run::Future<core::PutBlobTask> put_;            ///< owned here
    clio::run::Future<core::MultiPutBlobTask> land_;      ///< owned elsewhere
  };
  /** A blob of a flush in flight: its lock is held until left_ reaches 0. */
  struct FlushKey {
    std::string key_;
    size_t left_ = 0;         ///< its ranges not yet landed
    clio::run::u32 rc_ = 0;   ///< first failure among its ranges
    bool foreign_ = false;    ///< another node owns it: drop once landed
    clio::run::u32 owner_ = 0;  ///< its owner
    clio::run::u64 bytes_ = 0;  ///< bytes of the dirty ranges it queued
    bool flushed_ = false;      ///< every range landed: the blob is durable
  };
  /** Ranges of several blobs landing together on one owner (FlushAway). */
  struct LandBatch {
    ctp::ipc::FullPtr<char> buf_;              ///< the ranges' bytes
    clio::run::u64 cap_ = 0;                   ///< buf_'s size
    clio::run::u64 used_ = 0;                  ///< bytes placed in buf_
    std::vector<core::MultiPutDesc> descs_;    ///< one per range
    std::vector<size_t> keys_;                 ///< FlushKeys it carries
    clio::run::Future<core::MultiPutBlobTask> fut_;
  };
  /**
   * Flush blobs other containers own: their ranges land on each owner in
   * batches of up to kLandBatchBytes (see FlushAll).
   * @param keys the blobs, in key order; their rc_ receive the results
   */
  clio::run::TaskResume FlushAway(std::vector<FlushKey> *keys);
  /**
   * Send a batch of ranges to their owner, which writes them through.
   * @param b the batch (its buffer and descriptors filled)
   * @param owner the owner
   */
  void LandSend(LandBatch *b, clio::run::u32 owner);
  /**
   * A batch landed (or failed): free its buffer and account it to each of
   * its blobs (FlushRangeDone).
   * @param keys the flush's blobs
   * @param b the batch
   * @param rc its result
   */
  void LandBatchDone(std::vector<FlushKey> *keys, LandBatch *b,
                     clio::run::u32 rc);
  /**
   * Write-back batch put, common case: lock every record's blob, ask for
   * all sizes at once, put every record into volatile at once, and log
   * them dirty. Anything else (a special put, a full volatile tier, a
   * partial write that needs persistent's other bytes, a failed put)
   * leaves *done false and the caller takes the per-record path.
   * @param task the batch
   * @param done set true when the whole batch was absorbed here
   */
  clio::run::TaskResume AbsorbBatchFast(
      clio::run::shared_ptr<core::MultiPutBlobTask> task, bool *done);
  /**
   * Lock a dirty blob of a flush and queue its dirty ranges. A blob with
   * nothing dirty by then is unlocked again at once.
   * @param keys the flush's blobs
   * @param i the blob to open
   * @param queued receives its ranges
   */
  clio::run::TaskResume FlushOpenKey(std::vector<FlushKey> *keys, size_t i,
                                     std::deque<FlushOp> *queued);
  /**
   * Account one landed (or failed) range of a flush. When it was the blob's
   * last, a blob whose ranges all landed is marked clean, and its lock is
   * released either way (a failed blob stays dirty for the next flush).
   * @param keys the flush's blobs
   * @param i the range's blob
   * @param rc the range's result
   */
  void FlushRangeDone(std::vector<FlushKey> *keys, size_t i, clio::run::u32 rc);
  /**
   * Flush every dirty copy of a tag (or all) on this container through a
   * window of ranges in flight: as one range lands the next starts, so
   * the persistent tier always has up to kFlushInflightOps writes.
   * @param tag the tag (ignored with all_tags)
   * @param all_tags flush every tag
   * @param rc set to the first failure, else 0
   * @param stats when not null: the blobs and bytes this call made durable
   *        (added to, not reset)
   */
  clio::run::TaskResume FlushAll(TagId tag, bool all_tags, clio::run::u32 *rc,
                                 FlushStats *stats = nullptr);
  /**
   * Add the blobs of a flush that landed (FlushKey::flushed_) to the stats
   * and to the cache's counters.
   * @param keys the flush's blobs
   * @param stats the flush's totals, or null
   */
  void TallyFlush(const std::vector<FlushKey> &keys, FlushStats *stats);
  /**
   * Join the shared "flushall" round (every dirty copy of this container):
   * the periodic flush, fill-level flushes, stop flushes and Destroy share it.
   * @param rc set to the round's result
   * @param stats set to the round's flushed blobs and bytes
   */
  clio::run::TaskResume FlushEverythingShared(clio::run::u32 *rc,
                                              FlushStats *stats);
  /**
   * Flush everything dirty here until nothing is left, bounded in rounds, for
   * a write-back Destroy (BeginDrain must have run). With no_flush only the
   * dirty copies are counted as discarded.
   * @param no_flush skip the flush and discard the dirty copies
   * @param rc set to the failure (kSyncIoRc when dirty data remains), else 0
   * @param stats the blobs and bytes this destroy made durable (added to)
   * @param error set to a description when the flush failed
   */
  clio::run::TaskResume UnplugFlush(bool no_flush, clio::run::u32 *rc,
                                    FlushStats *stats, std::string *error);
  /**
   * Stop buffering new write-back puts and wait until none is in flight: the
   * first step of a write-back Destroy (draining_ and buffering_, seq_cst).
   */
  clio::run::TaskResume BeginDrain();
  /**
   * Drop every volatile copy this container holds (kDestroyPoolDeleteData),
   * each under its blob's lock, then clear the bookkeeping.
   */
  clio::run::TaskResume DropEveryCopy();
  /** Remove this cache's stop hook (the pool is going away). */
  void RemoveStopFlush();
  /**
   * Sum the dirty copies held here.
   * @param blobs set to the number of copies with dirty ranges
   * @param bytes set to the dirty bytes they hold
   */
  void DirtyTotals(clio::run::u64 *blobs, clio::run::u64 *bytes);
  /**
   * @return the "stats" Monitor answer: a msgpack map of the policy, the
   *         draining state, dirty and cached totals and every counter
   */
  std::string PackStats();
  /**
   * Flush blobs through a window of ranges in flight (see FlushAll). Blobs
   * another container owns (FlushKey::foreign_) land on their owner; the
   * others are written to persistent here.
   * @param keys the blobs, in key order; their rc_ receive the results
   */
  clio::run::TaskResume FlushWindow(std::vector<FlushKey> *keys);
  /**
   * @param key a cached blob
   * @return the container that owns it (this one when it is not cached)
   */
  clio::run::u32 KeyOwner(const std::string &key);
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
   * @param flushed when not leader and not null: that round's flush totals
   */
  clio::run::TaskResume JoinRound(std::string id, bool *leader,
                                  clio::run::u32 *rc, SyncOut *out = nullptr,
                                  FlushStats *flushed = nullptr);
  /** End the round the caller led, publishing its result (and, for a sync,
   *  its SyncTag outputs; for a flush, its totals) to the sharers. */
  void LeaveRound(const std::string &id, clio::run::u32 rc,
                  const SyncOut *out = nullptr,
                  const FlushStats *flushed = nullptr);
  void RegisterStopFlush(const clio::run::PoolId &self_pool);

  /**
   * The container that applies this cache's policy to a blob: its owner
   * (BlobHash, with the core's failover). Tasks reach this pool on other
   * containers too -- an interposer above forwards to its LOCAL container
   * -- and are sent on to the owner, which holds the blob's only copy.
   */
  clio::run::u32 OwnerOf(const TagId &tag, const std::string &name);
  /**
   * Write a put through: into persistent here when this container owns the
   * blob, else to its owner.
   * @param task the put
   * @param owner the blob's owner
   */
  clio::run::TaskResume PutThrough(
      clio::run::shared_ptr<clio::cte::core::PutBlobTask> task,
      clio::run::u32 owner);
  /**
   * A put that persistent takes first, with the blob's lock held: a special
   * put (persistent decides) or, on write-back, a put that may not buffer
   * (draining). The buffered copy is flushed and dropped first so an older
   * dirty range cannot overwrite these bytes later; a write-through put
   * then refreshes the volatile copy.
   * @param task the put
   * @param key its blob's key
   * @param special true for a special put (persistent decides)
   * @param off first byte the put writes
   * @param end one past the last byte the put writes
   */
  clio::run::TaskResume PutPersistentFirst(
      clio::run::shared_ptr<clio::cte::core::PutBlobTask> task,
      std::string key, bool special, clio::run::u64 off, clio::run::u64 end);
  /**
   * Write-back absorb of a put this container owns or may buffer, with the
   * blob's lock held: makes the copy whole, puts the range into volatile and
   * logs it dirty, or writes the put through when volatile cannot hold it.
   * @param task the put (its return code set)
   * @param key its blob's key
   * @param owner the blob's owner
   * @param off first byte the put writes
   * @param end one past the last byte the put writes
   */
  clio::run::TaskResume AbsorbLocked(
      clio::run::shared_ptr<clio::cte::core::PutBlobTask> task,
      std::string key, clio::run::u32 owner, clio::run::u64 off,
      clio::run::u64 end);
  /**
   * A batch whose records are not all owned here: each record is sent to its
   * owner at once (fan-out), and the batch's counts are set.
   * @param task the batch
   * @param batch its attached view
   */
  clio::run::TaskResume MultiPutFanOut(
      clio::run::shared_ptr<clio::cte::core::MultiPutBlobTask> task,
      clio::cte::core::MultiPutBatchView &batch);
  /**
   * Write-back absorb of a batch's records one by one (the path after
   * AbsorbBatchFast declined): each blob is locked, made whole, put into
   * volatile and logged dirty, or written through when volatile cannot hold
   * it. Sets the batch's num_ok_ and first_rc_.
   * @param task the batch
   * @param batch its attached view
   */
  clio::run::TaskResume MultiPutAbsorbRecords(
      clio::run::shared_ptr<clio::cte::core::MultiPutBlobTask> task,
      clio::cte::core::MultiPutBatchView &batch);
  /**
   * A batch the cache does not absorb (write-through, draining, or a range
   * another node flushed here): every record's lock is held across the
   * forward and the drop of its copies (see MultiPutBlob), and persistent
   * takes the batch.
   * @param task the batch
   * @param batch its attached view (when attached)
   * @param attached false when the batch could not be attached
   */
  clio::run::TaskResume MultiPutThrough(
      clio::run::shared_ptr<clio::cte::core::MultiPutBlobTask> task,
      clio::cte::core::MultiPutBatchView &batch, bool attached);
  /**
   * Write-back: land this container's buffered copy of a blob another
   * container owns (flush it to the owner) and drop it, so the owner serves
   * what this node wrote. Takes the blob's lock.
   * @param key the blob
   * @param rc set to the flush's failure, else 0
   */
  clio::run::TaskResume PublishBuffered(std::string key, clio::run::u32 *rc);
  /**
   * Send one dirty range of a blob another container owns to that owner,
   * which writes it through (see MultiPutBlob's land path).
   * @param tag the blob's tag
   * @param name the blob's name
   * @param x the range
   * @param buf the range's bytes
   * @param owner the blob's owner
   * @return the landing's future
   */
  clio::run::Future<clio::cte::core::MultiPutBlobTask> SendLand(
      const TagId &tag, const std::string &name, const Extent &x,
      const ctp::ipc::FullPtr<char> &buf, clio::run::u32 owner);
  /**
   * The result of a landing: its return code, else its first record's.
   * @param f the landing's (completed) future
   * @return 0 when the range landed
   */
  static clio::run::u32 LandRc(
      clio::run::Future<clio::cte::core::MultiPutBlobTask> &f);
  /**
   * @param key a blob
   * @return true when this container holds a copy of it with nothing dirty
   */
  bool CopyClean(const std::string &key);
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
  clio::cte::core::Client flush_client_;  ///< where flushed ranges land
  std::mutex mu_;                         ///< guards entries_ / busy_
  std::unordered_map<std::string, Entry> entries_;
  std::unordered_set<std::string> busy_;
  /** (tag major, tag minor): the key the dirty index groups copies by. */
  using TagKey = std::pair<clio::run::u32, clio::run::u32>;
  /**
   * Keys of the copies with dirty ranges, by tag, in key order. A sync of
   * one tag reads its set instead of scanning every entry under mu_ (512
   * ranks' fsyncs each asked every node, and every ask scanned the whole
   * cache). Guarded by mu_; kept by IndexDirty / UnindexDirty.
   */
  std::map<TagKey, std::set<std::string>> dirty_tags_;
  /**
   * Record in the dirty index whether a copy has dirty ranges. Call with
   * mu_ held after any change to the entry's dirty_.
   * @param key the copy's key
   * @param e its entry
   */
  void IndexDirty(const std::string &key, const Entry &e);
  /**
   * Take a copy out of the dirty index. Call with mu_ held before the
   * entry is erased.
   * @param key the copy's key
   */
  void UnindexDirty(const std::string &key);
  /**
   * @param tag a tag
   * @return true when this container holds a dirty copy of one of its blobs
   */
  bool TagDirty(const TagId &tag);
  /** Copies kept on the eviction list between scans. */
  static constexpr size_t kVictimQueue = 256;
  /** Eviction list (RefillVictims / TakeVictim). Guarded by mu_. */
  std::deque<std::string> victims_;
  /** on_hit_count fill: misses per uncached key. */
  std::unordered_map<std::string, clio::run::u32> miss_counts_;
  clio::run::u64 stop_hook_id_ = 0;
  /**
   * Set by a write-back Destroy: no put buffers any more, every put takes the
   * ordered write-through path. Stays set once the cache is removed (or after
   * a quiesce, until the removal that follows it).
   */
  std::atomic<bool> draining_{false};
  /** Write-back puts holding a BufferSlot (Destroy waits for zero). */
  std::atomic<clio::run::u64> buffering_{0};
  /** Counters for Monitor "stats" (see PackStats). */
  std::atomic<clio::run::u64> flush_rounds_{0};       ///< FlushAll calls
  std::atomic<clio::run::u64> blobs_flushed_{0};      ///< blobs made durable
  std::atomic<clio::run::u64> bytes_flushed_{0};      ///< bytes made durable
  std::atomic<clio::run::u64> flush_failures_{0};     ///< FlushAll failures
  std::atomic<clio::run::u64> unplug_flushes_{0};     ///< Destroys that flushed
  std::atomic<clio::run::u64> unplug_blobs_flushed_{0};  ///< by Destroy
  std::atomic<clio::run::u64> unplug_bytes_flushed_{0};  ///< by Destroy
  std::atomic<clio::run::u64> discarded_blobs_{0};    ///< dirty, NoFlush
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
  /** A fill-level or continuous flush has been sent and has not finished. */
  std::atomic<bool> fill_flush_pending_{false};
  /** Dirty bytes written since the last triggered flush started
   *  (flush_dirty_mb). */
  std::atomic<clio::run::u64> dirty_since_flush_{0};
  /** Guards rounds_ only: waiters poll it, and must not hold up the
   *  entry map (mu_) while they do. */
  std::mutex round_mu_;
  /** One kind of shared work (JoinRound). Guarded by round_mu_. */
  struct Round {
    clio::run::u64 started_ = 0;  ///< rounds started
    clio::run::u64 done_ = 0;     ///< rounds completed
    bool running_ = false;
    clio::run::u32 rc_ = 0;       ///< result of the last completed round
    clio::run::u32 users_ = 0;    ///< callers inside JoinRound..LeaveRound
    SyncOut out_;                 ///< SyncTag outputs of that round
    FlushStats flushed_;          ///< flush totals of that round
  };
  std::unordered_map<std::string, Round> rounds_;
};

}  // namespace clio::cte::cache_base

#endif  // CLIO_CTE_CACHE_BASE_CACHE_BASE_H_
