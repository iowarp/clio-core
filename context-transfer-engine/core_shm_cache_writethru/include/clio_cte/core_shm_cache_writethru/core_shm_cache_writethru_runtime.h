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

#ifndef CLIO_CTE_CORE_SHM_CACHE_WRITETHRU_RUNTIME_H_
#define CLIO_CTE_CORE_SHM_CACHE_WRITETHRU_RUNTIME_H_

#include <atomic>
#include <chrono>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include <clio_runtime/bdev/bdev_client.h>
#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_interposer.h>
#include <clio_cte/core/shm_wtc.h>
#include <clio_cte/core_shm_cache_writethru/core_shm_cache_writethru_client.h>
#include <clio_cte/core_shm_cache_writethru/core_shm_cache_writethru_tasks.h>

namespace clio::cte::shm_wtc {

using clio::cte::core::TagId;

/** One coalesced, flush-ready range. */
struct FlushRecord {
  clio::run::u32 slot_ = 0;    ///< blob table slot
  clio::run::u64 off_ = 0;     ///< offset inside the blob
  clio::run::u64 size_ = 0;    ///< bytes (clamped by ReadRange)
  clio::run::u32 ranges_ = 0;  ///< raw log ranges merged into this record
  BlobKey key_;                ///< blob identity
  clio::run::u32 home_ = 0;    ///< container of the blob's home cache
  clio::run::u64 payload_off_ = 0;  ///< offset in its home group's staging
  clio::run::u32 put_flags_ = 0;    ///< put flags of the blob
  /** Zero-copy flush: the record's bytes in the cache segment, pinned
   *  until the batch lands (nullptr when staged instead). */
  const char *ptr_ = nullptr;
};

/** One flush batch in flight: popped, staged and sent, not yet landed. */
struct InflightBatch {
  std::vector<FlushRecord> recs_;  ///< coalesced records it carries
  clio::run::u64 popped_ = 0;      ///< raw log ranges popped for it
  clio::run::u64 bytes_ = 0;       ///< payload bytes sent
  std::map<clio::run::u32, ctp::ipc::FullPtr<char>> staging_;  ///< per home
  std::vector<clio::run::u32> homes_;  ///< home of each future below
  std::vector<clio::run::Future<clio::cte::core::MultiPutBlobTask>> futs_;
  /** Ranges popped for this batch but not resent (an earlier flush already
   *  carried their bytes): slot -> count, retired when this batch lands. */
  std::unordered_map<clio::run::u32, clio::run::u32> skipped_;
  /** Records sent by pointer into the cache (pinned until landed). */
  bool pinned_ = false;
};

/** What the flusher last copied of a blob with no write in progress. */
struct FlushMark {
  clio::run::u64 gen_ = 0;  ///< every write up to this generation included
  clio::run::u64 off_ = 0;  ///< copied range
  clio::run::u64 end_ = 0;
};

/** Nodes holding a blob, as tracked by the blob's home. */
struct SharerSet {
  std::set<clio::run::u32> nodes_;    ///< containers that may cache it
  std::set<clio::run::u32> writers_;  ///< containers whose writes landed
};

/**
 * core_shm_cache_writethru runtime: owns the segment, drains its log, answers
 * every core verb for task callers, and keeps cached copies coherent across
 * nodes (blob homes track sharers and invalidate them when a write lands).
 */
class Runtime : public clio::cte::core::CoreInterposer {
 public:
  using CreateParams = WtcConfig;  // required by CLIO_TASK_CC

  Runtime() = default;
  ~Runtime() override;

  // ---- Method handlers ----
  clio::run::TaskResume Create(clio::run::shared_ptr<CreateTask> &task);
  clio::run::TaskResume Destroy(clio::run::shared_ptr<DestroyTask> &task);
  clio::run::TaskResume Monitor(clio::run::shared_ptr<MonitorTask> &task);
  clio::run::TaskResume GetOrCreateTag(
      clio::run::shared_ptr<clio::cte::core::GetOrCreateTagTask<
          clio::cte::core::CreateParams>> &task);
  clio::run::TaskResume PutBlob(
      clio::run::shared_ptr<clio::cte::core::PutBlobTask> &task);
  clio::run::TaskResume GetBlob(
      clio::run::shared_ptr<clio::cte::core::GetBlobTask> &task);
  clio::run::TaskResume DelBlob(
      clio::run::shared_ptr<clio::cte::core::DelBlobTask> &task);
  clio::run::TaskResume DelTag(
      clio::run::shared_ptr<clio::cte::core::DelTagTask> &task);
  clio::run::TaskResume GetBlobSize(
      clio::run::shared_ptr<clio::cte::core::GetBlobSizeTask> &task);
  clio::run::TaskResume MultiPutBlob(
      clio::run::shared_ptr<clio::cte::core::MultiPutBlobTask> &task);
  clio::run::TaskResume SyncTag(
      clio::run::shared_ptr<clio::cte::core::SyncTagTask> &task);
  clio::run::TaskResume FlushLog(clio::run::shared_ptr<FlushLogTask> &task);
  /**
   * Wake the worker that runs the flusher when it is parked and the dirty
   * log holds ranges. Cache-path writes push to the log without waking
   * anyone, and a parked worker polls its periodic tasks only every 1 to
   * max_sleep (50) ms (its idle backoff), so a task that waits on the
   * flusher (for room, a pin, or a barrier) calls this on each poll. No
   * syscall when the worker is running.
   */
  void KickFlusher();
  /**
   * Whether a put refused for room should keep waiting for the flusher:
   * ranges buffered before its first refusal are still unflushed (once
   * landed they are clean and evictable) and the wait is under
   * WtcConfig::nospace_wait_ms_.
   * @param target pushed-range count at the put's first refusal
   * @param t0 time of the put's first refusal
   * @return true to keep waiting without spending a no-space retry
   */
  bool RoomMayFollowFlush(clio::run::u64 target,
                          std::chrono::steady_clock::time_point t0) const;
  clio::run::TaskResume Invalidate(clio::run::shared_ptr<InvalidateTask> &task);
  clio::run::TaskResume HomeFlush(clio::run::shared_ptr<HomeFlushTask> &task);
  /** Any other core method: drain the local log, then forward verbatim. */
  clio::run::TaskResume BarrierForward(clio::run::u32 method,
                                       clio::run::shared_ptr<clio::run::Task> task);

  /** Route data verbs submitter-local (the segment is per node). */
  clio::run::PoolQuery ScheduleTask(
      const clio::run::shared_ptr<clio::run::Task> &task) override;

  /** Segment view (tests). */
  clio::cte::core::wtc::View &GetView() { return view_; }

  // ---- Container virtuals (autogen/core_shm_cache_writethru_lib_exec.cc) ----
  void Init(const clio::run::PoolId &pool_id, const std::string &pool_name,
            clio::run::u32 container_id = 0) override;
  clio::run::TaskResume Run(clio::run::u32 method,
                            clio::run::shared_ptr<clio::run::Task> task_ptr) override;
  clio::run::u64 GetWorkRemaining() const override;
  void LocalLoadTask(clio::run::u32 method, clio::run::DefaultLoadArchive &archive,
                     clio::run::shared_ptr<clio::run::Task> &task_ptr) override;
  clio::run::shared_ptr<clio::run::Task> LocalAllocLoadTask(
      clio::run::u32 method, clio::run::DefaultLoadArchive &archive) override;
  void LocalSaveTask(clio::run::u32 method, clio::run::DefaultSaveArchive &archive,
                     clio::run::shared_ptr<clio::run::Task> &task_ptr) override;
  void AggregateOut(clio::run::u32 method,
                    clio::run::shared_ptr<clio::run::Task> &orig_task,
                    const clio::run::shared_ptr<clio::run::Task> &replica_task) override;
  void AggregateIn(clio::run::u32 method,
                   clio::run::shared_ptr<clio::run::Task> &agg_task,
                   const clio::run::shared_ptr<clio::run::Task> &member_task) override;
  void SaveTask(clio::run::u32 method, clio::run::SaveTaskArchive &archive,
                clio::run::shared_ptr<clio::run::Task> &task_ptr) override;
  void LoadTask(clio::run::u32 method, clio::run::LoadTaskArchive &archive,
                clio::run::shared_ptr<clio::run::Task> &task_ptr) override;
  clio::run::shared_ptr<clio::run::Task> AllocLoadTask(
      clio::run::u32 method, clio::run::LoadTaskArchive &archive) override;
  clio::run::shared_ptr<clio::run::Task> NewCopyTask(
      clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> &orig,
      bool deep) override;
  clio::run::shared_ptr<clio::run::Task> NewTask(clio::run::u32 method) override;

 private:
  /** Create and publish the segment. */
  bool OpenSegment();
  /**
   * Log the segment's counters (puts and refusals by reason, flushes,
   * fills, read hits and misses) at warning level, so a run's log shows
   * where the cache's time went. Called from a runtime stop hook.
   */
  void LogStats();
  /**
   * Zero-copy flush: pin every record's bytes in the cache segment and
   * keep their addresses (FlushRecord::ptr_). All or nothing: when a
   * record cannot be pinned (extent mode), the ones pinned are released
   * and the batch is staged instead.
   * @param recs the batch's resolved records
   * @return true when every record with bytes is pinned
   */
  bool PinRecords(std::vector<FlushRecord> &recs);
  /** Release the pins PinRecords took. */
  void UnpinRecords(std::vector<FlushRecord> &recs);
  /**
   * Send a pinned batch: one task per record, its payload the record's
   * bytes in the cache (no staging copy). A record of another home lands
   * there through HomeFlush; one homed here goes into the core below.
   * @param recs the pinned records
   * @param homes receives the home of each future
   * @param futs receives one future per record sent
   * @return payload bytes sent
   */
  clio::run::u64 SendPinned(
      const std::vector<FlushRecord> &recs, std::vector<clio::run::u32> *homes,
      std::vector<clio::run::Future<clio::cte::core::MultiPutBlobTask>> *futs);
  /**
   * The flusher landed writes of these tags: invalidate their cached tag
   * metadata on every node, in the background. At most one invalidation
   * per tag is in flight; a landing while one is in flight marks the tag
   * so exactly one more goes out after it (so the last landing is always
   * covered). The flusher does not wait; SyncTag waits for all of them.
   * @param tags the tags of the landed batch (duplicates are fine)
   */
  void NoteTagsLanded(const std::vector<TagId> &tags);
  /**
   * Retire the finished tag-metadata invalidations, sending the follow-up
   * for each tag that landed again while one was in flight.
   * @return true when none is in flight
   */
  bool ReapTagNotes();
  /** Send one tag-metadata invalidation to the tag's home (no wait). */
  clio::run::Future<InvalidateTask> SendTagNote(const TagId &tag);
  /** A tag's background metadata invalidation (NoteTagsLanded). */
  struct TagNote {
    clio::run::Future<InvalidateTask> fut_;  ///< the one in flight
    bool again_ = false;  ///< landed again since it was sent
  };
  std::mutex notes_mtx_;  ///< guards tag_notes_
  std::unordered_map<clio::run::u64, TagNote> tag_notes_;  ///< by tag
  clio::run::u64 stats_hook_id_ = 0;  ///< stop hook that runs LogStats
  /** Lease data_capacity bytes from the data_pool RAM bdev (extent mode). */
  clio::run::TaskResume LeaseExtents();
  /** Core client bound to the next pool in the chain. */
  clio::cte::core::Client *Next();
  /** This node's id. */
  clio::run::u32 SelfNode() const;
  /** Container (of this pool) that is a blob's home. */
  clio::run::u32 HomeOf(const TagId &tag, const std::string &name) const;
  /**
   * Whether a task's put/get context can be served by the segment: no codec,
   * replica, emulation, generation, versioning, persistence target or fault
   * handler -- the segment stores raw bytes and none of those properties.
   * Task-path requests only; a client's shm path takes no context at all.
   * @param ctx operation context
   * @return true for a plain context
   */
  /**
   * Cost of this cache's tasks for the scheduler: the data size of each
   * (io_size_ selects the bulk network lane; wall_time_ drives worker
   * placement), on the core's model (~10 KB per CPU us, ~500 MB/s wall).
   * Core methods this cache does not handle specially use the interposer's.
   * @param task the task
   * @return its estimated size and cost
   */
  clio::run::TaskStat GetTaskStats(
      const clio::run::Task *task) const override {
    clio::run::TaskStat stat;
    if (task == nullptr) return stat;
    constexpr float kBytesPerComputeUs = 10000.0f;
    constexpr float kBytesPerWallUs = 500.0f;
    switch (task->method_) {
      case Method::kHomeFlush:
      case Method::kMultiPutBlob: {
        const auto *t =
            static_cast<const clio::cte::core::MultiPutBlobTask *>(task);
        stat.io_size_ = t->data_len_;
        stat.compute_ =
            static_cast<size_t>(t->data_len_ / kBytesPerComputeUs) + 2;
        stat.wall_time_ = static_cast<float>(t->data_len_) / kBytesPerWallUs;
        return stat;
      }
      case Method::kInvalidate: {
        const auto *t = static_cast<const InvalidateTask *>(task);
        stat.io_size_ = t->keys_.size();
        stat.compute_ = 2;
        stat.wall_time_ = 5.0f;
        return stat;
      }
      case Method::kFlushLog:
      case Method::kTagAttrs:
        stat.compute_ = 2;
        stat.wall_time_ = 5.0f;
        return stat;
      default:
        return clio::cte::core::CoreInterposer::GetTaskStats(task);
    }
  }

  /**
   * Whether a put/get context can be served by the segment: nothing in it
   * changes what bytes are stored or where they must go.
   *
   * `preallocate_` is ignored: it only sizes the core's block allocation and
   * never changes the stored bytes. clio-fs passes it on every page write, so
   * refusing it sent all file data around the segment and no read could hit.
   * The flush lands the coalesced ranges without the hint.
   * @param ctx the task's operation context
   * @return true when the segment may absorb (or serve) the operation
   */
  static bool PlainContext(const clio::cte::core::Context &ctx) {
    static const clio::cte::core::Context kDef;
    return ctx.persistence_target_ == kDef.persistence_target_ &&
           ctx.min_persistence_level_ == kDef.min_persistence_level_ &&
           !ctx.emulate_ &&
           ctx.transform_flags_ == 0 && ctx.replica_ == 0 &&
           ctx.replica_flags_ == 0 && ctx.version_ == kDef.version_ &&
           !ctx.create_on_get_ && ctx.op_flags_ == 0 &&
           ctx.fault_params_[0] == '\0' &&
           ctx.generation_ == kDef.generation_ &&
           ctx.dynamic_compress_ == kDef.dynamic_compress_ &&
           ctx.compress_lib_ == kDef.compress_lib_;
  }
  /** Pop up to one batch of log ranges and coalesce them. */
  std::vector<FlushRecord> PopAndCoalesce(
      clio::run::u64 *popped,
      std::unordered_map<clio::run::u32, clio::run::u32> *skipped);
  /** True if an earlier flush copy already carried this logged range. */
  bool Superseded(const clio::cte::core::wtc::DirtyRange &r) const;
  /**
   * Resolve each record's key and home container.
   * @param recs   coalesced records (a record whose entry cannot be read is
   *               zeroed and reported: it should never happen while pending)
   * @param totals out: payload bytes per home
   */
  void ResolveRecords(std::vector<FlushRecord> &recs,
                      std::map<clio::run::u32, clio::run::u64> *totals);
  /**
   * Allocate one staging buffer per home, waiting (never dropping) when the
   * runtime data segment is momentarily full: the ranges are already popped
   * from the log, so a dropped batch would be a lost acknowledged write.
   * @param totals  payload bytes per home
   * @param staging out: staging buffer per home
   */
  clio::run::TaskResume AllocStaging(
      const std::map<clio::run::u32, clio::run::u64> &totals,
      std::map<clio::run::u32, ctp::ipc::FullPtr<char>> *staging);
  /** Copy every resolved record's current bytes into its home's buffer. */
  void CopyRecords(std::vector<FlushRecord> &recs,
                   std::map<clio::run::u32, ctp::ipc::FullPtr<char>> &staging,
                   std::map<clio::run::u32, clio::run::u64> *staged);
  /**
   * Split one home's descriptors into at most flush_shards shards by blob,
   * so a home's batch lands on several workers at once; one blob never
   * spans two shards, which keeps its ranges in log order.
   */
  void ShardDescs(const std::vector<clio::cte::core::MultiPutDesc> &descs,
                  std::vector<std::vector<clio::cte::core::MultiPutDesc>> *shards);
  /**
   * Send every home's staged records as sharded MultiPutBlob (local home)
   * or HomeFlush (remote home) tasks.
   * @return payload bytes sent
   */
  clio::run::u64 SendGroups(
      const std::vector<FlushRecord> &recs,
      std::map<clio::run::u32, ctp::ipc::FullPtr<char>> &staging,
      std::map<clio::run::u32, clio::run::u64> &staged,
      std::vector<clio::run::u32> *homes,
      std::vector<clio::run::Future<clio::cte::core::MultiPutBlobTask>> *futs);
  /**
   * Await every sent group; replay a home whose shards reported a failure,
   * and invalidate other sharers of the local home's blobs once.
   */
  clio::run::TaskResume LandGroups(
      const std::vector<FlushRecord> &recs,
      std::map<clio::run::u32, ctp::ipc::FullPtr<char>> &staging,
      const std::vector<clio::run::u32> &homes,
      std::vector<clio::run::Future<clio::cte::core::MultiPutBlobTask>> &futs);
  /**
   * Pop, coalesce, stage and send one batch, appending it to inflight_. If
   * any of its blobs is still landing in an older batch, those batches land
   * first (oldest-first), so a blob's bytes reach the chain in staging order.
   * @param launched out: false when nothing was popped
   */
  clio::run::TaskResume LaunchBatch(bool *launched);
  /**
   * Await the oldest in-flight batch, retry or invalidate as needed, and
   * retire it: complete its ranges and advance the flushed_ watermark.
   * Batches retire strictly oldest-first, so flushed_ still means "the
   * first N popped ranges have landed" (the SyncTag barrier depends on it).
   */
  clio::run::TaskResume LandOldest();
  /** True when a record of `recs` names a slot an in-flight batch carries. */
  bool ConflictsInflight(const std::vector<FlushRecord> &recs) const;
  /** Descriptors of one home group. */
  static std::vector<clio::cte::core::MultiPutDesc> GroupDescs(
      const std::vector<FlushRecord> &recs, clio::run::u32 home);
  /** Re-send a failed group record by record; record sticky errors. */
  clio::run::TaskResume RetryGroup(const std::vector<FlushRecord> &recs,
                                   clio::run::u32 home,
                                   ctp::ipc::FullPtr<char> staging);
  /** Home side: record the writer, collect who to invalidate. */
  std::map<clio::run::u32, std::vector<BlobKey>> HomeApply(
      const std::vector<BlobKey> &keys, clio::run::u32 writer);
  /** Send invalidations and wait for every ack. */
  clio::run::TaskResume SendInvalidations(
      std::map<clio::run::u32, std::vector<BlobKey>> by_node);
  /** Reader side: register as a sharer with the blob's home. */
  clio::run::TaskResume RegisterSharer(const TagId &tag,
                                       const std::string &name);
  /** Ask the blobs' homes to drop every other container's copy. */
  /**
   * After a forwarded verb that changes cached state behind the cache
   * (truncate, Pod writes, multi deletes, renames, name updates), drop what
   * it changed here and at the homes, so no node serves the old bytes,
   * size or name.
   * @param method core method id
   * @param task   the forwarded task (already landed)
   */
  clio::run::TaskResume InvalidateAfterForward(
      clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> task);
  /**
   * TagAttrs (module verb, a core::GetTagSizeTask sent Local by a client):
   * the tag's cluster-wide size and times from the segment, or fetched once
   * (Broadcast down the chain) and cached, registered with the tag's home.
   */
  clio::run::TaskResume TagAttrs(
      clio::run::shared_ptr<clio::cte::core::GetTagSizeTask> &task);
  /**
   * Fetch a tag's cluster-wide size/times (draining this node's log first,
   * registered with the tag's home) and cache them unless a drop or a local
   * write raced the fetch.
   */
  clio::run::TaskResume FetchTagAttrs(const TagId &tag,
                                      clio::cte::core::wtc::TagAttrs *a,
                                      clio::run::u32 *rc);
  /** A write changed these tags: drop their cached attributes on every
   *  node (here: a local write counter; elsewhere: through each home). */
  clio::run::TaskResume NoteTagsWritten(const std::vector<TagId> &tags,
                                        bool count_local = true);
  /**
   * After a write to one blob landed in the chain: tell the blob's home and
   * the tag's home (its cached tag metadata) at the same time, and count
   * the write locally. Equivalent to FanoutInvalidate(tag, name) followed
   * by NoteTagsWritten({tag}), with the two sends in flight together.
   * @param tag the written blob's tag
   * @param name the written blob's name
   */
  clio::run::TaskResume FanoutInvalidateWrite(const TagId &tag,
                                              const std::string &name);
  /**
   * Send write invalidations to the keys' homes and return once every key
   * has been delivered. Keys bound for the same home that are waiting at
   * the same time travel in one message: a home has at most one such
   * message in flight, and the keys that arrive meanwhile form the next.
   * A key is always sent after this call queued it, so the caller's write
   * is ordered before its invalidation exactly as with one message per key.
   * @param keys the written keys (duplicates bound for one home are merged)
   */
  clio::run::TaskResume FanoutBatched(std::vector<BlobKey> keys);
  /** Drop every node's cached blobs and metadata of these tags. */
  clio::run::TaskResume DropTagDataEverywhere(const std::vector<TagId> &tags);
  clio::run::TaskResume FanoutInvalidate(const TagId &tag,
                                         const std::string &name);
  /** Wait until a blob has no unflushed ranges. */
  clio::run::TaskResume WaitBlobFlushed(const TagId &tag,
                                        const std::string &name);
  /**
   * Enter a chain read of a blob this container may be flushing: waits until
   * the blob has no unflushed ranges and no flush batch is landing, and keeps
   * FlushLog from starting one until EndChainRead. Without it a miss could
   * read the chain while this node's own flusher is mid-way through writing
   * newer bytes of the same blob (a torn read).
   * @param tag  blob's tag
   * @param name blob name
   */
  clio::run::TaskResume BeginChainRead(const TagId &tag,
                                       const std::string &name);
  /** Leave a chain read entered with BeginChainRead (same blob). */
  void EndChainRead(const TagId &tag, const std::string &name);
  /**
   * Enter a task-path write of ONE blob down the chain (explicit-score or
   * flagged put, write-around, DelBlob): waits for in-flight chain reads of
   * that blob and keeps new ones out until EndChainWrite. Without it a miss
   * could snapshot the entry after this write's invalidation, fetch the
   * chain BEFORE the write lands, and install the older bytes as complete.
   * Per blob, so streaming write-arounds never starve misses of other blobs.
   */
  clio::run::TaskResume BeginChainWrite(const TagId &tag,
                                        const std::string &name);
  /** Leave a per-blob chain write (same blob). */
  void EndChainWrite(const TagId &tag, const std::string &name);
  /** Tag-wide variant (DelTag): fences chain reads of EVERY blob. */
  clio::run::TaskResume BeginChainWriteAll();
  /** Leave a BeginChainWriteAll. */
  void EndChainWriteAll() { chain_writes_.fetch_sub(1); }
  /**
   * Copy one put into the segment, retrying the refusals that clear on
   * their own: kNoSpace (wait for the flusher to free room) and kBusy (the
   * entry is pinned by an in-flight flush and must grow). Retries yield
   * the coroutine and hold no lock, so the flush holding the pin (and the
   * home's invalidation of this node it waits on) can finish. Bounded:
   * after the budget the caller takes the non-cache path.
   * @param tag    tag of the blob
   * @param name   blob name
   * @param off    offset within the blob
   * @param size   bytes to write
   * @param src    source bytes (stay valid until this completes)
   * @param flags  put flags
   * @param st     out: the last TryPut status (kOk when absorbed)
   */
  clio::run::TaskResume TryPutRetrying(const TagId &tag,
                                       const std::string &name,
                                       clio::run::u64 off,
                                       clio::run::u64 size, const char *src,
                                       clio::run::u32 flags,
                                       clio::cte::core::wtc::Status *st);
  /**
   * Write one record around the cache (segment full): drain the blob's
   * logged ranges, fence chain reads, drop the local copy, put down the
   * chain, then invalidate other nodes' copies.
   * @param rc out: the chain put's return code
   */
  clio::run::TaskResume WriteAroundRecord(const TagId &tag,
                                          const std::string &name,
                                          clio::run::u64 off,
                                          clio::run::u64 size, const char *src,
                                          clio::run::u32 flags,
                                          clio::run::u32 *rc);

  /** Fetch the whole blob from the chain into the segment. */
  clio::run::TaskResume FillFromChain(const TagId &tag, const std::string &name,
                                      bool *filled);

  WtcConfig config_;
  clio::cte::core::wtc::View view_;
  /** This cache's Locator in the metadata segment (null when none was
   *  published); Destroy clears its ready flag so the core stops handing
   *  out keys for a cache that is gone. */
  clio::cte::core::wtc::Locator *locator_ = nullptr;
  /** Blocks leased from the data_pool RAM bdev (returned at Destroy). */
  std::vector<clio::run::bdev::Block> leased_;
  std::unique_ptr<clio::cte::core::Client> next_client_;
  /** Batches staged or in flight (the chain-read fence reads it as "a
   *  flush is landing"); incremented before a batch checks for readers. */
  std::atomic<int> flush_active_{0};
  /** One FlushLog coroutine drains the log at a time. */
  std::atomic<bool> flush_running_{false};
  /** Lane of the worker that last ran FlushLog (KickFlusher wakes it). */
  std::atomic<clio::run::TaskLane *> flusher_lane_{nullptr};
  /** Chain reads backed off by an active flush: no new batch starts while
   *  one waits, so in-flight batches drain and the read gets in. */
  std::atomic<int> reads_waiting_{0};
  std::deque<InflightBatch> inflight_;  ///< oldest first (flusher only)
  std::unordered_map<clio::run::u32, clio::run::u32> inflight_slots_;
  /** Per slot with ranges pending: the last quiescent flush copy. */
  std::unordered_map<clio::run::u32, FlushMark> flush_marks_;
  /** Chain reads in flight (BeginChainRead); FlushLog waits for zero. */
  std::atomic<int> chain_reads_{0};
  /** Tag-wide chain writes in flight (BeginChainWriteAll). */
  std::atomic<int> chain_writes_{0};
  std::mutex fence_mu_;  ///< guards blob_reads_ / blob_writes_
  /** Remote homes each tag's flushed ranges landed on since the tag's last
   *  SyncTag here: a SyncTag must make them durable there too (the home
   *  wrote them into its own next pool). Key: (major << 32) | minor. */
  std::unordered_map<clio::run::u64, std::set<clio::run::u32>> sync_homes_;
  std::mutex sync_homes_mu_;  ///< guards sync_homes_
  /** Note that a batch's records landed on their homes (before flushed_). */
  void NoteLandedHomes(const std::vector<FlushRecord> &recs);
  /** Take (and clear) the remote homes of a tag. */
  std::set<clio::run::u32> TakeSyncHomes(const TagId &tag);
  std::unordered_map<std::string, int> blob_reads_;   ///< chain reads per blob
  std::unordered_map<std::string, int> blob_writes_;  ///< chain writes per blob
  std::mutex sharers_mtx_;
  /** Write invalidations waiting for, or on their way to, one home. */
  struct InvLane {
    std::vector<BlobKey> pending_;  ///< keys of the batch numbered open_seq_
    clio::run::u64 open_seq_ = 1;   ///< batch that pending_ will leave in
    clio::run::u64 done_seq_ = 0;   ///< last batch the home acknowledged
    bool busy_ = false;             ///< a batch is in flight
  };
  std::mutex inv_mtx_;  ///< guards inv_lanes_
  std::unordered_map<clio::run::u32, InvLane> inv_lanes_;  ///< by home node
  std::map<std::string, SharerSet> sharers_;
};

/** Sharer-map key of a blob. */
inline std::string SharerKey(clio::run::u32 major, clio::run::u32 minor,
                             const std::string &name) {
  return std::to_string(major) + "." + std::to_string(minor) + "/" + name;
}

}  // namespace clio::cte::shm_wtc

#endif  // CLIO_CTE_CORE_SHM_CACHE_WRITETHRU_RUNTIME_H_
