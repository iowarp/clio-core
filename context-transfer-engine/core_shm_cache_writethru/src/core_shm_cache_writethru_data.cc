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

/**
 * core_shm_cache_writethru data verbs for TASK callers (remote nodes, TCP/IPC
 * clients, runtime-internal clients, and SHM clients whose op the segment
 * could not express). A plain put lands in the segment exactly as a client's
 * direct write would; everything else drains the blob's pending ranges and
 * goes down the chain.
 */

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <clio_cte/core/blob_batch.h>
#include <clio_cte/core_shm_cache_writethru/core_shm_cache_writethru_runtime.h>

namespace clio::cte::shm_wtc {

/** Poll period of a write waiting for its invalidation batch (us). */
static constexpr double kInvPollUs = 20.0;

/** Wall-clock ms (CLOCK_REALTIME) for cross-node trace ordering. */
static double WallMs() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

namespace wtc = clio::cte::core::wtc;
using clio::cte::core::BlobRegion;

static constexpr double kBarrierPollUs = 10.0;
static constexpr double kNoSpacePollUs = 50.0;
/** Waits for segment room before a put writes around the cache (~1 ms),
 *  once no range buffered before the put's first refusal is still
 *  unflushed. A segment full of clean small buffers may never yield a large
 *  one. Before that, a refused put waits up to WtcConfig::nospace_wait_ms_
 *  (default 250 ms) for those ranges to land: once flushed they are clean
 *  and evictable, so room follows. Writing around instead would take the
 *  same chain. */
static constexpr int kNoSpaceRetries = 20;
/** Retries of a put refused because its entry is pinned (kBusy) before it
 *  writes around the cache. The pin is an in-flight flush, released when
 *  the blob's home acks (a network round trip). */
static constexpr int kBusyRetries = 12;
/** First back-off between kBusy retries (us); doubles per retry. */
static constexpr double kBusyPollStartUs = 10.0;
/** Back-off cap between kBusy retries (us); with the start above the whole
 *  budget is ~6 ms of yields before the put writes around the cache. */
static constexpr double kBusyPollMaxUs = 1000.0;

bool Runtime::RoomMayFollowFlush(
    clio::run::u64 target, std::chrono::steady_clock::time_point t0) const {
  if (view_.FlushedThrough(target)) return false;
  const double waited_ms = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - t0)
                               .count();
  return waited_ms < static_cast<double>(config_.nospace_wait_ms_);
}

clio::run::TaskResume Runtime::TryPutRetrying(const TagId &tag,
                                              const std::string &name,
                                              clio::run::u64 off,
                                              clio::run::u64 size,
                                              const char *src,
                                              clio::run::u32 flags,
                                              wtc::Status *st) {
  CLIO_TASK_BODY_BEGIN
  {
    int nospace_tries = 0;
    int busy_tries = 0;
    double busy_us = kBusyPollStartUs;
    bool waiting_room = false;
    clio::run::u64 room_target = 0;
    std::chrono::steady_clock::time_point room_t0;
    while (true) {
      *st = view_.TryPut(tag.major_, tag.minor_, name.c_str(), name.size(),
                         off, size, src, flags, /*may_block=*/false);
      if (*st == wtc::Status::kNoSpace && !waiting_room) {
        waiting_room = true;
        room_target = view_.PushedNow();
        room_t0 = std::chrono::steady_clock::now();
      }
      // Room and pins come back only as the flusher lands buffered ranges:
      // wake it if its worker is parked, else the wait outlasts the budget.
      if (*st == wtc::Status::kNoSpace &&
          RoomMayFollowFlush(room_target, room_t0)) {
        view_.GetRoot()->stats_.waits_.fetch_add(1);
        KickFlusher();
        CLIO_CO_AWAIT(clio::run::yield(kNoSpacePollUs));
      } else if (*st == wtc::Status::kNoSpace &&
                 nospace_tries < kNoSpaceRetries) {
        ++nospace_tries;
        view_.GetRoot()->stats_.waits_.fetch_add(1);
        KickFlusher();
        CLIO_CO_AWAIT(clio::run::yield(kNoSpacePollUs));
      } else if (*st == wtc::Status::kBusy && busy_tries < kBusyRetries) {
        ++busy_tries;
        view_.GetRoot()->stats_.waits_.fetch_add(1);
        KickFlusher();
        CLIO_CO_AWAIT(clio::run::yield(busy_us));
        busy_us = std::min(busy_us * 2.0, kBusyPollMaxUs);
      } else {
        break;
      }
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::WaitBlobFlushed(const TagId &tag,
                                               const std::string &name) {
  CLIO_TASK_BODY_BEGIN
  if (view_.Ready()) {
    while (view_.BlobPending(tag.major_, tag.minor_, name.c_str(),
                             name.size()) != 0) {
      KickFlusher();
      CLIO_CO_AWAIT(clio::run::yield(kBarrierPollUs));
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

/** Fence key of one blob. */
static std::string FenceKey(const TagId &tag, const std::string &name) {
  return std::to_string(tag.major_) + "." + std::to_string(tag.minor_) + "/" +
         name;
}

clio::run::TaskResume Runtime::BeginChainRead(const TagId &tag,
                                              const std::string &name) {
  CLIO_TASK_BODY_BEGIN
  while (true) {
    // The blob's own unflushed ranges must land first: wait for them WITHOUT
    // holding the flusher off (it is what lands them).
    if (view_.Ready() && view_.BlobPending(tag.major_, tag.minor_,
                                           name.c_str(), name.size()) != 0) {
      KickFlusher();
      CLIO_CO_AWAIT(clio::run::yield(kBarrierPollUs));
      continue;
    }
    chain_reads_.fetch_add(1);  // the flusher waits for this to drain
    const bool flushing = flush_active_.load() != 0;
    bool held = false;
    if (!flushing && chain_writes_.load() == 0 &&
        (!view_.Ready() || view_.BlobPending(tag.major_, tag.minor_,
                                             name.c_str(), name.size()) == 0)) {
      std::lock_guard<std::mutex> lk(fence_mu_);
      const std::string key = FenceKey(tag, name);
      if (blob_writes_.find(key) == blob_writes_.end()) {
        ++blob_reads_[key];
        held = true;  // no batch landing, none can start, no write of it
      }
    }
    if (held) break;
    chain_reads_.fetch_sub(1);
    // Only a landing flush is drained for us: ask the flusher to stop
    // starting batches until this read is through.
    if (flushing) reads_waiting_.fetch_add(1);
    CLIO_CO_AWAIT(clio::run::yield(kBarrierPollUs));
    if (flushing) reads_waiting_.fetch_sub(1);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

void Runtime::EndChainRead(const TagId &tag, const std::string &name) {
  {
    std::lock_guard<std::mutex> lk(fence_mu_);
    auto it = blob_reads_.find(FenceKey(tag, name));
    if (it != blob_reads_.end() && --it->second == 0) blob_reads_.erase(it);
  }
  chain_reads_.fetch_sub(1);
}

clio::run::TaskResume Runtime::BeginChainWrite(const TagId &tag,
                                               const std::string &name) {
  CLIO_TASK_BODY_BEGIN
  {
    const std::string key = FenceKey(tag, name);
    {
      std::lock_guard<std::mutex> lk(fence_mu_);
      ++blob_writes_[key];  // new reads of this blob back off from here
    }
    while (true) {
      {
        std::lock_guard<std::mutex> lk(fence_mu_);
        if (blob_reads_.find(key) == blob_reads_.end()) break;
      }
      CLIO_CO_AWAIT(clio::run::yield(kBarrierPollUs));
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

void Runtime::EndChainWrite(const TagId &tag, const std::string &name) {
  std::lock_guard<std::mutex> lk(fence_mu_);
  auto it = blob_writes_.find(FenceKey(tag, name));
  if (it != blob_writes_.end() && --it->second == 0) blob_writes_.erase(it);
}

clio::run::TaskResume Runtime::BeginChainWriteAll() {
  CLIO_TASK_BODY_BEGIN
  // seq_cst pairs with BeginChainRead (increment, then check the other).
  chain_writes_.fetch_add(1);
  while (chain_reads_.load() != 0) {
    CLIO_CO_AWAIT(clio::run::yield(kBarrierPollUs));
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::FanoutInvalidate(const TagId &tag,
                                                const std::string &name) {
  CLIO_TASK_BODY_BEGIN
  if (CLIO_IPC->GetNumHosts() > 1) {
    // A named local, not a braced temporary inside the co_await: GCC 16.2
    // hits an internal compiler error (build_special_member_call) on the
    // temporary initializer_list in a coroutine await expression.
    std::vector<BlobKey> keys{BlobKey{tag.major_, tag.minor_, name}};
    CLIO_CO_AWAIT(FanoutBatched(std::move(keys)));
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::FanoutBatched(std::vector<BlobKey> keys) {
  CLIO_TASK_BODY_BEGIN
  {
    /** One key of this call: its home and the batch it was queued into. */
    struct Wait {
      clio::run::u32 home_;
      clio::run::u64 seq_;
    };
    std::vector<Wait> waits;
    {
      std::lock_guard<std::mutex> lk(inv_mtx_);
      for (BlobKey &k : keys) {
        const clio::run::u32 home =
            HomeOf(TagId(k.major_, k.minor_), k.name_);
        InvLane &lane = inv_lanes_[home];
        bool dup = false;
        for (const BlobKey &p : lane.pending_) {
          dup = dup || (p.major_ == k.major_ && p.minor_ == k.minor_ &&
                        p.name_ == k.name_);
        }
        if (!dup) lane.pending_.push_back(std::move(k));
        waits.push_back(Wait{home, lane.open_seq_});
      }
    }
    auto *ipc = CLIO_IPC;
    while (!waits.empty()) {
      // Drop the keys already delivered; lead the first lane that is idle
      // with this call's batch next in line.
      std::vector<BlobKey> batch;
      Wait led{0, 0};
      {
        std::lock_guard<std::mutex> lk(inv_mtx_);
        for (size_t i = 0; i < waits.size();) {
          InvLane &lane = inv_lanes_[waits[i].home_];
          if (lane.done_seq_ >= waits[i].seq_) {
            waits[i] = waits.back();
            waits.pop_back();
            continue;
          }
          if (batch.empty() && !lane.busy_ &&
              lane.open_seq_ == waits[i].seq_) {
            lane.busy_ = true;
            batch.swap(lane.pending_);
            ++lane.open_seq_;
            led = waits[i];
          }
          ++i;
        }
      }
      if (batch.empty()) {
        if (!waits.empty()) CLIO_CO_AWAIT(clio::run::yield(kInvPollUs));
        continue;
      }
      auto inv = ipc->NewTask<InvalidateTask>(
          clio::run::CreateTaskId(), pool_id_,
          clio::run::PoolQuery::DirectId(led.home_), kInvalidateFanout,
          container_id_, PackKeys(batch));
      auto fut = ipc->Send(inv);
      CLIO_CO_AWAIT(fut);
      std::lock_guard<std::mutex> lk(inv_mtx_);
      InvLane &lane = inv_lanes_[led.home_];
      lane.done_seq_ = led.seq_;
      lane.busy_ = false;
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::FanoutInvalidateWrite(
    const TagId &tag, const std::string &name) {
  CLIO_TASK_BODY_BEGIN
  {
    const bool meta = view_.Ready() && !tag.IsNull();
    if (meta) view_.NoteLocalWrite();
    // invalidate_on_flush: the write-back cache below sends nothing until it
    // flushes, and its flush lands through this pool on the home, which
    // invalidates then (MultiPutBlob). Deletes still invalidate at once.
    if (CLIO_IPC->GetNumHosts() > 1 && !config_.invalidate_on_flush_) {
      std::vector<BlobKey> keys;
      keys.push_back(BlobKey{tag.major_, tag.minor_, name});
      if (meta) {
        keys.push_back(BlobKey{
            tag.major_, tag.minor_,
            std::string(wtc::kTagMetaName, wtc::kTagMetaNameLen)});
      }
      CLIO_CO_AWAIT(FanoutBatched(std::move(keys)));
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

// ===========================================================================
// Writes
// ===========================================================================

clio::run::TaskResume Runtime::PutBlob(
    clio::run::shared_ptr<clio::cte::core::PutBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    const std::string name = task->GetBlobName();
    const bool plain =
        view_.Ready() &&
        (task->flags_ & ~wtc::kPerBlobPutFlags) == 0 &&
        task->score_ < 0.0f && PlainContext(task->context_) &&
        name.size() <= wtc::kMaxBlobName;
    bool absorbed = plain;
    if (plain) {
      std::vector<BlobRegion> regions;
      clio::cte::core::ForEachBlobRegion(*task, [&regions](const BlobRegion &r) {
        regions.push_back(r);
        return true;
      });
      auto *ipc = CLIO_IPC;
      for (size_t i = 0; i < regions.size() && absorbed; ++i) {
        if (regions[i].size_ == 0) {
          continue;
        }
        const char *src =
            ipc->ToFullPtr<char>(regions[i].data_.template Cast<char>()).ptr_;
        if (src == nullptr) {
          task->SetReturnCode(1);
          CLIO_CO_RETURN;
        }
        wtc::Status st = wtc::Status::kNoSpace;
        CLIO_CO_AWAIT(TryPutRetrying(task->tag_id_, name,
                                     regions[i].blob_off_, regions[i].size_,
                                     src, task->flags_, &st));
        // Not absorbed (a flags conflict, still no room, or still pinned by
        // a flush): the whole put
        // goes down the chain below, ordered after this blob's logged
        // ranges -- an earlier absorbed region lands again, unchanged.
        absorbed = st == wtc::Status::kOk;
      }
      if (absorbed) {
        task->SetReturnCode(0);
        CLIO_CO_RETURN;
      }
    }
    // Not expressible in the segment: drain this blob's buffered ranges so
    // the chain orders this write after them, then drop the local copy. No
    // miss may fill from the chain until this write has landed.
    CLIO_CO_AWAIT(WaitBlobFlushed(task->tag_id_, name));
    CLIO_CO_AWAIT(BeginChainWrite(task->tag_id_, name));
    if (view_.Ready()) {
      view_.Invalidate(task->tag_id_.major_, task->tag_id_.minor_,
                       name.c_str(), name.size());
    }
    if (task->segments_.empty()) {
      auto f = Next()->AsyncPutBlob(task->tag_id_, name, task->offset_,
                                    task->size_, task->blob_data_,
                                    task->score_, task->context_, task->flags_,
                                    clio::run::PoolQuery::Dynamic());
      CLIO_CO_AWAIT(f);
      // INOUT fields: an append reports the offset its bytes landed at, and
      // the name may be completed handler-side (clio-fs log appends read
      // the landed offset back).
      task->context_ = f->context_;
      task->offset_ = f->offset_;
      task->blob_name_ = f->blob_name_;
      task->SetReturnCode(f->GetReturnCode());
    } else {
      CLIO_CO_AWAIT(ForwardToCore(Method::kPutBlob,
                                  task.template Cast<clio::run::Task>()));
    }
    // Drop again now that the write landed: a local reader may have cached
    // the pre-write size (or bytes) between the drop above and the landing,
    // and neither the home (it skips the writer) nor a single-node run
    // (no fan-out) would invalidate it.
    if (view_.Ready()) {
      view_.Invalidate(task->tag_id_.major_, task->tag_id_.minor_,
                       name.c_str(), name.size());
    }
    EndChainWrite(task->tag_id_, name);
    CLIO_CO_AWAIT(FanoutInvalidateWrite(task->tag_id_, name));
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::WriteAroundRecord(
    const TagId &tag, const std::string &name, clio::run::u64 off,
    clio::run::u64 size, const char *src, clio::run::u32 flags,
    clio::run::u32 *rc) {
  CLIO_TASK_BODY_BEGIN
  // Same order as a task-path put: this blob's logged ranges land first,
  // no fill may race the write, and the stale local copy is dropped.
  CLIO_CO_AWAIT(WaitBlobFlushed(tag, name));
  CLIO_CO_AWAIT(BeginChainWrite(tag, name));
  if (view_.Ready()) {
    view_.Invalidate(tag.major_, tag.minor_, name.c_str(), name.size());
  }
  {
    auto f = Next()->AsyncPutBlob(tag, name, off, size, src, -1.0f,
                                  clio::cte::core::Context(), flags,
                                  clio::run::PoolQuery::Dynamic());
    CLIO_CO_AWAIT(f);
    *rc = f.get() == nullptr ? 1u : f->GetReturnCode();
  }
  if (view_.Ready()) {  // see PutBlob: drop what a reader cached meanwhile
    view_.Invalidate(tag.major_, tag.minor_, name.c_str(), name.size());
  }
  EndChainWrite(tag, name);
  CLIO_CO_AWAIT(FanoutInvalidateWrite(tag, name));
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::MultiPutBlob(
    clio::run::shared_ptr<clio::cte::core::MultiPutBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    clio::cte::core::MultiPutBatchView batch;
    const bool attached = clio::cte::core::MultiPutBatchView::Attach(*task, &batch);
    if (view_.Ready() && view_.MdOnly() && attached) {
      // Metadata only: the batch goes down as one task, then every
      // record's cached metadata is dropped here and at its home.
      std::vector<BlobKey> keys;
      std::vector<TagId> tags;
      for (size_t i = 0; i < batch.size(); ++i) {
        const auto &d = batch.descs_[i];
        keys.push_back(BlobKey{d.tag_id_.major_, d.tag_id_.minor_,
                               d.blob_name_});
        tags.push_back(d.tag_id_);
      }
      CLIO_CO_AWAIT(BarrierForward(Method::kMultiPutBlob,
                                   task.template Cast<clio::run::Task>()));
      for (const BlobKey &k : keys) {
        view_.Invalidate(k.major_, k.minor_, k.name_.c_str(), k.name_.size());
      }
      // The records' blobs and their tags' cached metadata, invalidated at
      // their homes together (one batch per home).
      if (view_.Ready() && !tags.empty()) view_.NoteLocalWrite();
      if (CLIO_IPC->GetNumHosts() > 1) {
        const std::string meta(wtc::kTagMetaName, wtc::kTagMetaNameLen);
        std::set<std::pair<clio::run::u32, clio::run::u32>> seen;
        for (const TagId &t : tags) {
          if (view_.Ready() && !t.IsNull() &&
              seen.insert({t.major_, t.minor_}).second) {
            keys.push_back(BlobKey{t.major_, t.minor_, meta});
          }
        }
        CLIO_CO_AWAIT(FanoutBatched(std::move(keys)));
      }
      CLIO_CO_RETURN;
    }
    if (view_.Ready() && attached && PlainContext(task->context_)) {
      task->num_ok_ = 0;
      task->first_rc_ = 0;
      for (size_t i = 0; i < batch.size(); ++i) {
        const auto &d = batch.descs_[i];
        if (!batch.RecordValid(i) || d.blob_name_.size() > wtc::kMaxBlobName) {
          if (task->first_rc_ == 0) task->first_rc_ = 1;
          continue;
        }
        wtc::Status st = wtc::Status::kNoSpace;
        CLIO_CO_AWAIT(TryPutRetrying(d.tag_id_, d.blob_name_, d.offset_,
                                     d.size_, batch.base_ + d.payload_off_,
                                     d.flags_, &st));
        if (st != wtc::Status::kOk) {
          // No room (kNoSpace), still pinned by a flush (kBusy),
          // conflicting put flags (kMiss) or a name the segment cannot
          // hold: write this record around the cache.
          clio::run::u32 rc = 0;
          CLIO_CO_AWAIT(WriteAroundRecord(d.tag_id_, d.blob_name_, d.offset_,
                                          d.size_, batch.base_ + d.payload_off_,
                                          d.flags_, &rc));
          st = rc == 0 ? wtc::Status::kOk : wtc::Status::kMiss;
        }
        if (st != wtc::Status::kOk) {
          if (task->first_rc_ == 0) task->first_rc_ = 1;
          continue;
        }
        ++task->num_ok_;
      }
      task->SetReturnCode(task->first_rc_ == 0 ? 0 : 1);
      CLIO_CO_RETURN;
    }
  }
  CLIO_CO_AWAIT(BarrierForward(Method::kMultiPutBlob,
                               task.template Cast<clio::run::Task>()));
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::DelBlob(
    clio::run::shared_ptr<clio::cte::core::DelBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    const std::string name = task->blob_name_.str();
    CLIO_CO_AWAIT(WaitBlobFlushed(task->tag_id_, name));
    CLIO_CO_AWAIT(BeginChainWrite(task->tag_id_, name));
    if (view_.Ready()) {
      view_.DropBlob(task->tag_id_.major_, task->tag_id_.minor_, name.c_str(),
                     name.size());
    }
    auto f = Next()->AsyncDelBlob(task->tag_id_, name,
                                  clio::run::PoolQuery::Dynamic(),
                                  task->del_flags_);
    CLIO_CO_AWAIT(f);
    EndChainWrite(task->tag_id_, name);
    task->SetReturnCode(f->GetReturnCode());
    CLIO_CO_AWAIT(FanoutInvalidateWrite(task->tag_id_, name));
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

// ===========================================================================
// Reads
// ===========================================================================

clio::run::TaskResume Runtime::RegisterSharer(const TagId &tag,
                                              const std::string &name) {
  CLIO_TASK_BODY_BEGIN
  {
    const clio::run::u32 home = HomeOf(tag, name);
    if (home == container_id_) {
      // The home never tracks itself: every write from another container
      // lands here and drops this copy directly (HomeApply / fanout).
    } else {
      auto *ipc = CLIO_IPC;
      BlobKey k{tag.major_, tag.minor_, name};
      auto reg = ipc->NewTask<InvalidateTask>(
          clio::run::CreateTaskId(), pool_id_,
          clio::run::PoolQuery::DirectId(home), kInvalidateRegister,
          container_id_, PackKeys({k}));
      auto fut = ipc->Send(reg);
      CLIO_CO_AWAIT(fut);
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::FillFromChain(const TagId &tag,
                                             const std::string &name,
                                             bool *filled) {
  CLIO_TASK_BODY_BEGIN
  *filled = false;
  {
    clio::run::u64 gen = 0;
    clio::run::u64 epoch = 0;
    if (!view_.SnapshotGens(tag.major_, tag.minor_, name.c_str(), name.size(),
                            &gen, &epoch)) {
      CLIO_CO_RETURN;
    }
    // Register BEFORE fetching: a write landing at the home after this point
    // invalidates us, which bumps the epoch and voids the fill below.
    CLIO_CO_AWAIT(RegisterSharer(tag, name));
    auto sz = Next()->AsyncGetBlobSize(tag, name);
    CLIO_CO_AWAIT(sz);
    const clio::run::u64 size = sz->size_;
    if (sz->GetReturnCode() != 0 || size == 0 || size > config_.max_fill_) {
      CLIO_CO_RETURN;
    }
    auto *ipc = CLIO_IPC;
    ctp::ipc::FullPtr<char> buf = ipc->AllocateBuffer(size);
    if (buf.IsNull()) {
      CLIO_CO_RETURN;
    }
    auto get = Next()->AsyncGetBlob(tag, name, 0, size, 0,
                                    ctp::ipc::ShmPtr<>(buf.shm_),
                                    clio::run::PoolQuery::Dynamic());
    CLIO_CO_AWAIT(get);
    if (get->GetReturnCode() == 0) {
      *filled = view_.Fill(tag.major_, tag.minor_, name.c_str(), name.size(),
                           buf.ptr_, size, gen, epoch);
    }
    ipc->FreeBuffer(buf);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::GetBlob(
    clio::run::shared_ptr<clio::cte::core::GetBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    const std::string name = task->GetBlobName();
    auto *ipc = CLIO_IPC;
    char *dst = task->blob_data_.IsNull()
                    ? nullptr
                    : ipc->ToFullPtr<char>(task->blob_data_.template Cast<char>()).ptr_;
    // Metadata-only: bytes are never cached here, so no fill either.
    const bool eligible = view_.Ready() && !view_.MdOnly() &&
                          task->flags_ == 0 &&
                          PlainContext(task->context_) &&
                          task->segments_.empty() && dst != nullptr &&
                          task->size_ > 0 && name.size() <= wtc::kMaxBlobName;
    const TagId tag = task->tag_id_;
    if (eligible) {
      if (view_.TryGet(tag.major_, tag.minor_, name.c_str(), name.size(),
                       task->offset_, task->size_, dst) == wtc::Status::kOk) {
        task->SetReturnCode(0);
        CLIO_CO_RETURN;
      }
      CLIO_CO_AWAIT(BeginChainRead(tag, name));
      bool filled = false;
      CLIO_CO_AWAIT(FillFromChain(tag, name, &filled));
      if (filled &&
          view_.TryGet(tag.major_, tag.minor_, name.c_str(), name.size(),
                       task->offset_, task->size_, dst) == wtc::Status::kOk) {
        EndChainRead(tag, name);
        task->SetReturnCode(0);
        CLIO_CO_RETURN;
      }
    } else {
      CLIO_CO_AWAIT(BeginChainRead(tag, name));
    }
    if (task->segments_.empty()) {
      auto f = Next()->AsyncGetBlob(tag, name, task->offset_, task->size_,
                                    task->flags_, task->blob_data_,
                                    clio::run::PoolQuery::Dynamic(),
                                    task->context_);
      CLIO_CO_AWAIT(f);
      task->context_ = f->context_;
      task->SetReturnCode(f->GetReturnCode());
    } else {
      CLIO_CO_AWAIT(ForwardToCore(Method::kGetBlob,
                                  task.template Cast<clio::run::Task>()));
    }
    EndChainRead(tag, name);
    {
      const char *tr = std::getenv("CLIO_WTC_TRACE");
      if (tr != nullptr && tr[0] == '1' && task->GetReturnCode() != 0) {
        HLOG(kInfo, "[WTC-TRACE] t={} chainread-miss at={} blob={}.{}/{} home={} "
             "rc={}", WallMs(), container_id_, tag.major_, tag.minor_, name,
             HomeOf(tag, name), task->GetReturnCode());
      }
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::GetBlobSize(
    clio::run::shared_ptr<clio::cte::core::GetBlobSizeTask> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    const std::string name = task->blob_name_.str();
    const TagId tag = task->tag_id_;
    clio::run::u64 size = 0;
    if (task->replica_ == 0 && view_.Ready() &&
        view_.TryGetSize(tag.major_, tag.minor_, name.c_str(), name.size(),
                         &size) == wtc::Status::kOk) {
      task->size_ = size;
      task->lost_bytes_ = 0;
      task->SetReturnCode(0);
      CLIO_CO_RETURN;
    }
    CLIO_CO_AWAIT(WaitBlobFlushed(tag, name));
    // Cache the answer (metadata only): snapshot the entry's generations and
    // register with the blob's home BEFORE asking the chain, so a write that
    // lands after this point invalidates us and voids the fill.
    clio::run::u64 gen = 0;
    clio::run::u64 epoch = 0;
    const bool cacheable =
        task->replica_ == 0 && view_.Ready() &&
        view_.SnapshotGens(tag.major_, tag.minor_, name.c_str(), name.size(),
                           &gen, &epoch);
    if (cacheable) {
      CLIO_CO_AWAIT(RegisterSharer(tag, name));
    }
    auto f = Next()->AsyncGetBlobSize(tag, name, clio::run::PoolQuery::Dynamic(),
                                      task->replica_);
    CLIO_CO_AWAIT(f);
    task->size_ = f->size_;
    task->lost_bytes_ = f->lost_bytes_;
    task->SetReturnCode(f->GetReturnCode());
    if (cacheable && f->GetReturnCode() == 0 && f->lost_bytes_ == 0) {
      view_.FillSize(tag.major_, tag.minor_, name.c_str(), name.size(),
                     f->size_, gen, epoch);
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

}  // namespace clio::cte::shm_wtc
