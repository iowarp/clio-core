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
 * core_shm_cache_writethru flusher and coherence.
 *
 * FlushLog is the dirty log's only consumer. One pass pops a batch of
 * ranges, coalesces them per blob, copies each coalesced range's CURRENT
 * bytes into one staging buffer per home container, and ships each group as
 * ONE MultiPutBlob: straight down the chain when this container is the home,
 * else as a HomeFlush to the home, which forwards it down and invalidates
 * the other containers caching those blobs before it acks.
 */

#include <algorithm>
#include <deque>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <clio_cte/core/blob_batch.h>
#include <clio_cte/core_shm_cache_writethru/core_shm_cache_writethru_runtime.h>
#include <clio_runtime/evlat.h>

namespace clio::cte::shm_wtc {

/** Wall-clock ms (CLOCK_REALTIME) for cross-node trace ordering. */
static double WallMs() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

namespace wtc = clio::cte::core::wtc;
using clio::cte::core::Context;
using clio::cte::core::MultiPutBlobTask;
using clio::cte::core::MultiPutDesc;

static constexpr size_t kMaxRangesPerFlush = 8192;
/** Raw log ranges (resent or skipped) one batch may pop. */
static constexpr size_t kMaxPopsPerFlush = 65536;
static constexpr clio::run::u32 kWriterMarker = 0xFFFFFFF7u;
static constexpr double kStagingRetryUs = 50.0;

/** CLIO_WTC_TRACE=1: log every flush group, home flush and barrier. */
bool WtcTrace() {
  static const bool v = [] {
    const char *e = std::getenv("CLIO_WTC_TRACE");
    return e != nullptr && e[0] == '1';
  }();
  return v;
}

void Runtime::NoteLandedHomes(const std::vector<FlushRecord> &recs) {
  std::lock_guard<std::mutex> g(sync_homes_mu_);
  for (const FlushRecord &fr : recs) {
    if (fr.size_ == 0 || fr.home_ == container_id_) continue;
    clio::run::u64 k = (static_cast<clio::run::u64>(fr.key_.major_) << 32) |
                       fr.key_.minor_;
    sync_homes_[k].insert(fr.home_);
  }
}

std::set<clio::run::u32> Runtime::TakeSyncHomes(const TagId &tag) {
  clio::run::u64 k = (static_cast<clio::run::u64>(tag.major_) << 32) |
                     tag.minor_;
  std::lock_guard<std::mutex> g(sync_homes_mu_);
  auto it = sync_homes_.find(k);
  if (it == sync_homes_.end()) return {};
  std::set<clio::run::u32> out = std::move(it->second);
  sync_homes_.erase(it);
  return out;
}

bool Runtime::Superseded(const wtc::DirtyRange &r) const {
  auto it = flush_marks_.find(r.slot_);
  if (it == flush_marks_.end()) {
    return false;
  }
  const FlushMark &m = it->second;
  return view_.RangeGen(r) <= m.gen_ && r.off_ >= m.off_ &&
         r.off_ + r.size_ <= m.end_;
}

std::vector<FlushRecord> Runtime::PopAndCoalesce(
    clio::run::u64 *popped,
    std::unordered_map<clio::run::u32, clio::run::u32> *skipped) {
  std::vector<wtc::DirtyRange> rs;
  clio::run::u64 bytes = 0;
  clio::run::u64 n = 0;
  wtc::DirtyRange r{};
  wtc::DirtyLog *log = view_.Log();
  // Overwrites: a range whose bytes an earlier flush copy already carried
  // (older generation, covered range) is retired with this batch instead of
  // being resent, and does not count against the batch budget.
  while (rs.size() < kMaxRangesPerFlush && n < kMaxPopsPerFlush &&
         bytes < config_.flush_batch_bytes_ && log->Pop(r)) {
    ++n;
    if (Superseded(r)) {
      ++(*skipped)[r.slot_];
      continue;
    }
    rs.push_back(r);
    bytes += r.size_;
  }
  *popped = n;
  std::sort(rs.begin(), rs.end(),
            [](const wtc::DirtyRange &a, const wtc::DirtyRange &b) {
              return a.slot_ != b.slot_ ? a.slot_ < b.slot_ : a.off_ < b.off_;
            });
  std::vector<FlushRecord> recs;
  for (const wtc::DirtyRange &x : rs) {
    if (!recs.empty() && recs.back().slot_ == x.slot_ &&
        x.off_ <= recs.back().off_ + recs.back().size_) {
      FlushRecord &cur = recs.back();
      clio::run::u64 end = std::max(cur.off_ + cur.size_, x.off_ + x.size_);
      cur.size_ = end - cur.off_;
      ++cur.ranges_;
      continue;
    }
    FlushRecord fr;
    fr.slot_ = x.slot_;
    fr.off_ = x.off_;
    fr.size_ = x.size_;
    fr.ranges_ = 1;
    recs.push_back(fr);
  }
  return recs;
}

void Runtime::ResolveRecords(std::vector<FlushRecord> &recs,
                             std::map<clio::run::u32, clio::run::u64> *totals) {
  for (FlushRecord &fr : recs) {
    clio::run::u64 none = 0;
    std::string name;
    if (!view_.ReadRange(fr.slot_, fr.off_, &none, nullptr, &fr.key_.major_,
                         &fr.key_.minor_, &name, &fr.put_flags_)) {
      HLOG(kError, "core_shm_cache_writethru: logged range of slot {} [{}, +{}) "
           "has no resident entry; its bytes cannot be flushed",
           fr.slot_, fr.off_, fr.size_);
      fr.size_ = 0;
      continue;
    }
    fr.key_.name_ = std::move(name);
    fr.home_ = HomeOf(TagId(fr.key_.major_, fr.key_.minor_), fr.key_.name_);
    (*totals)[fr.home_] += fr.size_;
  }
}

clio::run::TaskResume Runtime::AllocStaging(
    const std::map<clio::run::u32, clio::run::u64> &totals,
    std::map<clio::run::u32, ctp::ipc::FullPtr<char>> *staging) {
  CLIO_TASK_BODY_BEGIN
  for (const auto &t : totals) {
    if (t.second == 0) {
      continue;
    }
    clio::run::u32 waits = 0;
    while (true) {
      ctp::ipc::FullPtr<char> buf = CLIO_IPC->AllocateBuffer(t.second);
      if (!buf.IsNull()) {
        (*staging)[t.first] = buf;
        break;
      }
      if (waits++ % 10000 == 0) {
        HLOG(kWarning, "core_shm_cache_writethru: no room for a {} B flush staging "
             "buffer; waiting", t.second);
      }
      CLIO_CO_AWAIT(clio::run::yield(kStagingRetryUs));
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

bool Runtime::PinRecords(std::vector<FlushRecord> &recs) {
  for (size_t i = 0; i < recs.size(); ++i) {
    FlushRecord &fr = recs[i];
    if (fr.size_ == 0) continue;
    clio::run::u64 size = fr.size_;
    clio::run::u64 gen = 0;
    const char *p = view_.PinRange(fr.slot_, fr.off_, &size, &gen);
    if (p == nullptr) {
      for (size_t j = 0; j < i; ++j) {
        if (recs[j].ptr_ != nullptr) view_.UnpinRange(recs[j].slot_);
        recs[j].ptr_ = nullptr;
      }
      return false;
    }
    fr.ptr_ = p;
    fr.size_ = size;
    if (gen != 0 && size != 0) {
      flush_marks_[fr.slot_] = FlushMark{gen, fr.off_, fr.off_ + size};
    }
  }
  return true;
}

void Runtime::UnpinRecords(std::vector<FlushRecord> &recs) {
  for (FlushRecord &fr : recs) {
    if (fr.ptr_ != nullptr) view_.UnpinRange(fr.slot_);
    fr.ptr_ = nullptr;
  }
}

clio::run::u64 Runtime::SendPinned(
    const std::vector<FlushRecord> &recs, std::vector<clio::run::u32> *homes,
    std::vector<clio::run::Future<MultiPutBlobTask>> *futs) {
  auto *ipc = CLIO_IPC;
  clio::run::u64 bytes = 0;
  for (const FlushRecord &fr : recs) {
    if (fr.size_ == 0 || fr.ptr_ == nullptr) continue;
    MultiPutDesc d;
    d.tag_id_ = TagId(fr.key_.major_, fr.key_.minor_);
    d.blob_name_ = fr.key_.name_;
    d.offset_ = fr.off_;
    d.size_ = fr.size_;
    d.payload_off_ = 0;
    d.flags_ = fr.put_flags_;
    const bool local = fr.home_ == container_id_;
    auto t = ipc->NewTask<MultiPutBlobTask>(
        clio::run::CreateTaskId(), local ? CorePoolId() : pool_id_,
        local ? clio::run::PoolQuery::Local()
              : clio::run::PoolQuery::DirectId(fr.home_),
        local ? d.tag_id_ : TagId(container_id_, kWriterMarker), d.blob_name_,
        ctp::ipc::ShmPtr<>::FromRaw(const_cast<char *>(fr.ptr_)), fr.size_,
        clio::cte::core::EncodeMultiPutDescs({d}), Context());
    if (!local) t->method_ = Method::kHomeFlush;
    homes->push_back(fr.home_);
    futs->push_back(ipc->Send(t));
    bytes += fr.size_;
  }
  return bytes;
}

void Runtime::CopyRecords(
    std::vector<FlushRecord> &recs,
    std::map<clio::run::u32, ctp::ipc::FullPtr<char>> &staging,
    std::map<clio::run::u32, clio::run::u64> *staged) {
  for (FlushRecord &fr : recs) {
    auto it = staging.find(fr.home_);
    if (fr.size_ == 0 || it == staging.end()) {
      fr.size_ = 0;
      continue;
    }
    clio::run::u64 &cursor = (*staged)[fr.home_];
    clio::run::u64 size = fr.size_;
    std::string ignored;
    clio::run::u32 a = 0;
    clio::run::u32 b = 0;
    clio::run::u32 f = 0;
    clio::run::u64 gen = 0;
    if (!view_.ReadRange(fr.slot_, fr.off_, &size, it->second.ptr_ + cursor,
                         &a, &b, &ignored, &f, &gen)) {
      HLOG(kError, "core_shm_cache_writethru: entry of slot {} vanished while "
           "pending; range [{}, +{}) not flushed", fr.slot_, fr.off_, fr.size_);
      size = 0;
    } else if (gen != 0 && size != 0) {
      flush_marks_[fr.slot_] = FlushMark{gen, fr.off_, fr.off_ + size};
    }
    fr.size_ = size;
    fr.payload_off_ = cursor;
    cursor += size;
  }
}

std::vector<MultiPutDesc> Runtime::GroupDescs(
    const std::vector<FlushRecord> &recs, clio::run::u32 home) {
  std::vector<MultiPutDesc> descs;
  for (const FlushRecord &fr : recs) {
    if (fr.home_ != home || fr.size_ == 0) {
      continue;
    }
    MultiPutDesc d;
    d.tag_id_ = TagId(fr.key_.major_, fr.key_.minor_);
    d.blob_name_ = fr.key_.name_;
    d.offset_ = fr.off_;
    d.size_ = fr.size_;
    d.payload_off_ = fr.payload_off_;
    d.flags_ = fr.put_flags_;
    descs.push_back(std::move(d));
  }
  return descs;
}

clio::run::TaskResume Runtime::RetryGroup(const std::vector<FlushRecord> &recs,
                                          clio::run::u32 home,
                                          ctp::ipc::FullPtr<char> staging) {
  CLIO_TASK_BODY_BEGIN
  // A batch reported a failure: replay its records one by one so a bad
  // record is attributed to its own tag (sticky rc, returned by SyncTag)
  // instead of failing every tag in the batch.
  for (size_t i = 0; i < recs.size(); ++i) {
    if (recs[i].home_ != home || recs[i].size_ == 0) {
      continue;
    }
    clio::run::u32 rc = 1;
    for (clio::run::u32 attempt = 0;
         attempt < config_.flush_retries_ && rc != 0; ++attempt) {
      // Pinned records retry from the cache bytes; staged ones from staging.
      char *src = recs[i].ptr_ != nullptr
                      ? const_cast<char *>(recs[i].ptr_)
                      : staging.ptr_ + recs[i].payload_off_;
      auto f = Next()->AsyncPutBlob(
          TagId(recs[i].key_.major_, recs[i].key_.minor_), recs[i].key_.name_,
          recs[i].off_, recs[i].size_, ctp::ipc::ShmPtr<>::FromRaw(src),
          -1.0f, clio::cte::core::Context(), recs[i].put_flags_);
      CLIO_CO_AWAIT(f);
      rc = f->GetReturnCode();
    }
    if (rc != 0) {
      HLOG(kError, "core_shm_cache_writethru: flush of {}.{}/{} failed rc={}",
           recs[i].key_.major_, recs[i].key_.minor_, recs[i].key_.name_, rc);
      view_.SetTagError(recs[i].key_.major_, recs[i].key_.minor_, rc);
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

void Runtime::ShardDescs(const std::vector<MultiPutDesc> &descs,
                         std::vector<std::vector<MultiPutDesc>> *shards) {
  const size_t k = std::max<size_t>(
      1, std::min<size_t>(config_.flush_shards_, descs.size()));
  shards->assign(k, {});
  std::hash<std::string> h;
  for (const MultiPutDesc &d : descs) {
    // By blob: one blob's records always share a shard, so its ranges land
    // in log order even with the shards in flight together.
    const size_t i = (h(d.blob_name_) ^ (static_cast<size_t>(d.tag_id_.major_) << 32 |
                                         d.tag_id_.minor_)) % k;
    (*shards)[i].push_back(d);
  }
}

clio::run::u64 Runtime::SendGroups(
    const std::vector<FlushRecord> &recs,
    std::map<clio::run::u32, ctp::ipc::FullPtr<char>> &staging,
    std::map<clio::run::u32, clio::run::u64> &staged,
    std::vector<clio::run::u32> *homes,
    std::vector<clio::run::Future<MultiPutBlobTask>> *futs) {
  auto *ipc = CLIO_IPC;
  clio::run::u64 bytes = 0;
  for (const auto &g : staging) {
    if (g.second.IsNull()) continue;
    std::vector<MultiPutDesc> descs = GroupDescs(recs, g.first);
    if (descs.empty()) continue;
    const bool local = g.first == container_id_;
    std::vector<std::vector<MultiPutDesc>> shards;
    if (local) {
      ShardDescs(descs, &shards);
    } else {
      // A HomeFlush ships its whole staging buffer: one per remote home.
      shards.push_back(std::move(descs));
    }
    for (const auto &sh : shards) {
      if (sh.empty()) continue;
      // Every shard shares the home's staging buffer: payload_off_ indexes it.
      auto t = ipc->NewTask<MultiPutBlobTask>(
          clio::run::CreateTaskId(), local ? CorePoolId() : pool_id_,
          local ? clio::run::PoolQuery::Local()
                : clio::run::PoolQuery::DirectId(g.first),
          local ? sh.front().tag_id_ : TagId(container_id_, kWriterMarker),
          sh.front().blob_name_, ctp::ipc::ShmPtr<>(g.second.shm_),
          staged[g.first], clio::cte::core::EncodeMultiPutDescs(sh),
          Context());
      if (!local) t->method_ = Method::kHomeFlush;
      if (WtcTrace()) {
        HLOG(kInfo, "[WTC-TRACE] t={} flush from={} home={} recs={} bytes={} "
             "first={}.{}/{}", WallMs(), container_id_, g.first, sh.size(),
             staged[g.first], sh.front().tag_id_.major_,
             sh.front().tag_id_.minor_, sh.front().blob_name_);
      }
      homes->push_back(g.first);
      futs->push_back(ipc->Send(t));
    }
    bytes += staged[g.first];
  }
  return bytes;
}

clio::run::TaskResume Runtime::LandGroups(
    const std::vector<FlushRecord> &recs,
    std::map<clio::run::u32, ctp::ipc::FullPtr<char>> &staging,
    const std::vector<clio::run::u32> &homes,
    std::vector<clio::run::Future<MultiPutBlobTask>> &futs) {
  CLIO_TASK_BODY_BEGIN
  {
    std::set<clio::run::u32> failed;
    std::set<clio::run::u32> seen;
    for (size_t i = 0; i < futs.size(); ++i) {
      CLIO_CO_AWAIT(futs[i]);
      if (WtcTrace()) {
        HLOG(kInfo, "[WTC-TRACE] t={} flush-done from={} home={} rc={} "
             "first_rc={}", WallMs(), container_id_, homes[i],
             futs[i]->GetReturnCode(), futs[i]->first_rc_);
      }
      if (futs[i]->GetReturnCode() != 0 || futs[i]->first_rc_ != 0) {
        failed.insert(homes[i]);
      }
      seen.insert(homes[i]);
    }
    for (clio::run::u32 home : failed) {
      // Replays every record of the home (puts are idempotent).
      CLIO_CO_AWAIT(RetryGroup(recs, home, staging[home]));
    }
    if (seen.count(container_id_) != 0) {
      std::vector<BlobKey> keys;
      for (const FlushRecord &fr : recs) {
        if (fr.home_ == container_id_ && fr.size_ > 0) keys.push_back(fr.key_);
      }
      CLIO_CO_AWAIT(SendInvalidations(HomeApply(keys, container_id_)));
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

bool Runtime::ConflictsInflight(const std::vector<FlushRecord> &recs) const {
  for (const FlushRecord &fr : recs) {
    if (inflight_slots_.count(fr.slot_) != 0) return true;
  }
  return false;
}

clio::run::TaskResume Runtime::LaunchBatch(bool *launched) {
  CLIO_TASK_BODY_BEGIN
  *launched = false;
  {
    InflightBatch b;
    b.recs_ = PopAndCoalesce(&b.popped_, &b.skipped_);
    if (b.popped_ != 0) {
      // A blob still landing in an older batch: land those first, so this
      // batch stages (and lands) the blob's newer bytes after them.
      while (ConflictsInflight(b.recs_)) {
        CLIO_CO_AWAIT(LandOldest());
      }
      std::map<clio::run::u32, clio::run::u64> totals;
      ResolveRecords(b.recs_, &totals);
      if (PinRecords(b.recs_)) {
        // Zero copy: the tasks point at the bytes in the cache itself.
        b.pinned_ = true;
        b.bytes_ = SendPinned(b.recs_, &b.homes_, &b.futs_);
      } else {
        CLIO_CO_AWAIT(AllocStaging(totals, &b.staging_));
        std::map<clio::run::u32, clio::run::u64> staged;
        CopyRecords(b.recs_, b.staging_, &staged);
        b.bytes_ = SendGroups(b.recs_, b.staging_, staged, &b.homes_, &b.futs_);
      }
      for (const FlushRecord &fr : b.recs_) ++inflight_slots_[fr.slot_];
      inflight_.push_back(std::move(b));
      *launched = true;
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::LandOldest() {
  CLIO_TASK_BODY_BEGIN
  {
    InflightBatch b = std::move(inflight_.front());
    inflight_.pop_front();
    const unsigned long long ev_land0 = clio::run::evlat::Now();
    CLIO_CO_AWAIT(LandGroups(b.recs_, b.staging_, b.homes_, b.futs_));
    clio::run::evlat::Span(clio::run::evlat::kWtcLand, ev_land0,
                           clio::run::evlat::Now());
    if (b.pinned_) UnpinRecords(b.recs_);
    auto *ipc = CLIO_IPC;
    for (auto &g : b.staging_) {
      if (!g.second.IsNull()) ipc->FreeBuffer(g.second);
    }
    for (const FlushRecord &fr : b.recs_) {
      view_.CompleteRanges(fr.slot_, fr.ranges_);
      auto it = inflight_slots_.find(fr.slot_);
      if (it != inflight_slots_.end() && --it->second == 0) {
        inflight_slots_.erase(it);
      }
    }
    // Before flushed_ moves: a SyncTag that sees these ranges flushed must
    // also find their homes.
    NoteLandedHomes(b.recs_);
    // Skipped ranges retire only now: every older batch (including the one
    // that carried their bytes) has landed.
    clio::run::u64 nskipped = 0;
    for (const auto &s : b.skipped_) {
      view_.CompleteRanges(s.first, s.second);
      nskipped += s.second;
    }
    // A mark is only useful while the slot has ranges left in the log;
    // one that does is in a later batch, which checks again.
    auto prune = [this](clio::run::u32 slot) {
      if (view_.Pending(slot) == 0) flush_marks_.erase(slot);
    };
    for (const FlushRecord &fr : b.recs_) prune(fr.slot_);
    for (const auto &s : b.skipped_) prune(s.first);
    wtc::Root *root = view_.GetRoot();
    root->stats_.flush_skipped_.fetch_add(nskipped);
    // The batch's writes are in the chain now: other nodes' cached
    // attributes of these tags are stale (once per tag per batch).
    {
      std::vector<TagId> written;
      for (const FlushRecord &fr : b.recs_) {
        if (fr.size_ > 0) written.push_back(TagId(fr.key_.major_, fr.key_.minor_));
      }
      const unsigned long long ev_note0 = clio::run::evlat::Now();
      ReapTagNotes();
      NoteTagsLanded(written);
      clio::run::evlat::Span(clio::run::evlat::kWtcNote, ev_note0,
                             clio::run::evlat::Now());
    }
    root->stats_.flush_batches_.fetch_add(b.futs_.size());
    root->stats_.flush_ranges_.fetch_add(b.popped_);
    root->stats_.flush_bytes_.fetch_add(b.bytes_);
    // Oldest-first retirement keeps flushed_ a prefix of the pop order.
    root->flushed_.fetch_add(b.popped_, std::memory_order_acq_rel);
    flush_active_.fetch_sub(1);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

void Runtime::KickFlusher() {
  clio::run::TaskLane *lane = flusher_lane_.load(std::memory_order_acquire);
  if (lane == nullptr || !view_.Ready() || view_.Log()->Empty()) return;
  auto *ipc = CLIO_IPC;
  ipc->AwakenWorker(lane);  // skipped (no syscall) while the worker runs
}

clio::run::TaskResume Runtime::FlushLog(clio::run::shared_ptr<FlushLogTask> &task) {
  CLIO_TASK_BODY_BEGIN
  task->ranges_ = 0;
  task->bytes_ = 0;
  {
    // Where waiters find the flusher; a periodic task may move to another
    // worker (stalled-worker rescue), so republish on every run.
    auto *worker = CLIO_CUR_WORKER;
    if (worker != nullptr) {
      flusher_lane_.store(worker->GetLane(), std::memory_order_release);
    }
  }
  if (!view_.Ready() || flush_running_.exchange(true)) {
    task->SetReturnCode(0);
    CLIO_CO_RETURN;
  }
  // Pipeline: keep up to flush_depth batches in flight, staging the next
  // while older ones land; land oldest-first; stop starting batches while a
  // chain read is in progress or waiting, so the fence can open.
  while (true) {
    while (inflight_.size() < std::max<clio::run::u32>(1, config_.flush_depth_) &&
           !view_.Log()->Empty() && reads_waiting_.load() == 0) {
      // seq_cst pairs with BeginChainRead (increment, then check the other).
      flush_active_.fetch_add(1);
      if (chain_reads_.load() != 0) {
        flush_active_.fetch_sub(1);
        break;
      }
      bool launched = false;
      CLIO_CO_AWAIT(LaunchBatch(&launched));
      if (!launched) {
        flush_active_.fetch_sub(1);
        break;
      }
      task->ranges_ += inflight_.back().popped_;
      task->bytes_ += inflight_.back().bytes_;
    }
    if (inflight_.empty()) break;
    CLIO_CO_AWAIT(LandOldest());
  }
  flush_running_.store(false);
  task->SetReturnCode(0);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

// ===========================================================================
// Coherence
// ===========================================================================

std::map<clio::run::u32, std::vector<BlobKey>> Runtime::HomeApply(
    const std::vector<BlobKey> &keys, clio::run::u32 writer) {
  std::map<clio::run::u32, std::vector<BlobKey>> by_node;
  std::lock_guard<std::mutex> lk(sharers_mtx_);
  // The directory tracks OTHER containers only. This home's own copy is
  // dropped directly whenever another container's write lands here, so a
  // home writing its own blobs keeps no entries -- and never invalidates
  // its own fresh copies when the bounded directory overflows.
  for (const BlobKey &k : keys) {
    const std::string sk = SharerKey(k.major_, k.minor_, k.name_);
    if (writer != container_id_) by_node[container_id_].push_back(k);
    auto it = sharers_.find(sk);
    if (it != sharers_.end()) {
      bool foreign_writer = false;
      for (clio::run::u32 w : it->second.writers_) {
        foreign_writer = foreign_writer || w != writer;
      }
      for (clio::run::u32 n : it->second.nodes_) {
        if (n != writer) by_node[n].push_back(k);
      }
      if (foreign_writer) {
        // The writer's buffer may lack another writer's bytes.
        by_node[writer].push_back(k);
      }
    }
    if (writer == container_id_) {
      if (it != sharers_.end()) sharers_.erase(it);
    } else {
      SharerSet &s = sharers_[sk];
      s.nodes_.clear();
      s.writers_.clear();
      s.nodes_.insert(writer);
      s.writers_.insert(writer);
    }
  }
  while (sharers_.size() > config_.max_sharer_entries_) {
    // Bounded directory: forget an entry by invalidating its holders.
    auto victim = sharers_.begin();
    BlobKey vk;
    size_t slash = victim->first.find('/');
    size_t dot = victim->first.find('.');
    vk.major_ = static_cast<clio::run::u32>(std::stoul(victim->first.substr(0, dot)));
    vk.minor_ = static_cast<clio::run::u32>(
        std::stoul(victim->first.substr(dot + 1, slash - dot - 1)));
    vk.name_ = victim->first.substr(slash + 1);
    for (clio::run::u32 n : victim->second.nodes_) by_node[n].push_back(vk);
    sharers_.erase(victim);
  }
  return by_node;
}

clio::run::TaskResume Runtime::SendInvalidations(
    std::map<clio::run::u32, std::vector<BlobKey>> by_node) {
  CLIO_TASK_BODY_BEGIN
  {
    auto *ipc = CLIO_IPC;
    std::vector<clio::run::Future<InvalidateTask>> futs;
    for (const auto &e : by_node) {
      if (e.first == container_id_) {
        // A cache without a segment (serving tasks only) has nothing here.
        for (size_t i = 0; view_.Ready() && i < e.second.size(); ++i) {
          const BlobKey &k = e.second[i];
          view_.Invalidate(k.major_, k.minor_, k.name_.c_str(), k.name_.size());
        }
        continue;
      }
      auto t = ipc->NewTask<InvalidateTask>(
          clio::run::CreateTaskId(), pool_id_,
          clio::run::PoolQuery::DirectId(e.first), kInvalidateDrop,
          container_id_, PackKeys(e.second));
      futs.push_back(ipc->Send(t));
    }
    for (size_t i = 0; i < futs.size(); ++i) {
      CLIO_CO_AWAIT(futs[i]);
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::HomeFlush(clio::run::shared_ptr<HomeFlushTask> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    const clio::run::u32 writer = task->route_tag_id_.major_;
    const unsigned long long ev_home0 = clio::run::evlat::Now();
    if (WtcTrace()) {
      HLOG(kInfo, "[WTC-TRACE] t={} homeflush-recv at={} writer={}", WallMs(),
           container_id_, writer);
    }
    std::vector<MultiPutDesc> descs =
        clio::cte::core::DecodeMultiPutDescs(task->descs_);
    if (!descs.empty()) {
      task->route_tag_id_ = descs.front().tag_id_;
    }
    // Land the batch below this cache, then invalidate before acking, so a
    // writer's barrier means no other container still serves the old bytes.
    task->method_ = clio::cte::core::Method::kMultiPutBlob;
    CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kMultiPutBlob,
                                task.template Cast<clio::run::Task>()));
    task->method_ = Method::kHomeFlush;
    if (WtcTrace()) {
      HLOG(kInfo, "[WTC-TRACE] t={} homeflush at={} writer={} recs={} rc={} "
           "first_rc={} first={}.{}/{}", WallMs(), container_id_, writer, descs.size(),
           task->GetReturnCode(), task->first_rc_,
           descs.empty() ? 0u : descs.front().tag_id_.major_,
           descs.empty() ? 0u : descs.front().tag_id_.minor_,
           descs.empty() ? std::string() : descs.front().blob_name_);
    }
    std::vector<BlobKey> keys;
    for (const MultiPutDesc &d : descs) {
      keys.push_back(BlobKey{d.tag_id_.major_, d.tag_id_.minor_, d.blob_name_});
    }
    CLIO_CO_AWAIT(SendInvalidations(HomeApply(keys, writer)));
    clio::run::evlat::Span(clio::run::evlat::kWtcHome, ev_home0,
                           clio::run::evlat::Now());
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Invalidate(
    clio::run::shared_ptr<InvalidateTask> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    std::vector<BlobKey> keys = UnpackKeys(task->keys_.str());
    const clio::run::u32 sender = task->sender_node_;
    std::map<clio::run::u32, std::vector<BlobKey>> fan;
    if (task->mode_ == kInvalidateDrop && view_.Ready()) {
      for (const BlobKey &k : keys) {
        if (WtcTrace() && k.name_.size() == wtc::kTagMetaNameLen &&
            k.name_.compare(0, wtc::kTagMetaNameLen, wtc::kTagMetaName) == 0) {
          HLOG(kInfo, "[WTC-TRACE] tagmeta-drop at={} tag={}.{} from={}",
               container_id_, k.major_, k.minor_, sender);
        }
        view_.Invalidate(k.major_, k.minor_, k.name_.c_str(), k.name_.size());
      }
    } else if (task->mode_ == kInvalidateDropTagData && view_.Ready()) {
      for (const BlobKey &k : keys) {
        view_.DropTagBlobs(k.major_, k.minor_);
        view_.DropTagMeta(k.major_, k.minor_);
      }
    } else if (task->mode_ == kInvalidateDropTag && view_.Ready()) {
      for (const BlobKey &k : keys) {
        view_.DropTagBlobs(k.major_, k.minor_);
        view_.DropTag(k.major_, k.minor_);
      }
    } else if (task->mode_ == kInvalidateRegister) {
      std::lock_guard<std::mutex> lk(sharers_mtx_);
      for (const BlobKey &k : keys) {
        sharers_[SharerKey(k.major_, k.minor_, k.name_)].nodes_.insert(sender);
      }
    } else if (task->mode_ == kInvalidateFanout) {
      std::lock_guard<std::mutex> lk(sharers_mtx_);
      for (const BlobKey &k : keys) {
        auto it = sharers_.find(SharerKey(k.major_, k.minor_, k.name_));
        if (WtcTrace() && k.name_.size() == wtc::kTagMetaNameLen &&
            k.name_.compare(0, wtc::kTagMetaNameLen, wtc::kTagMetaName) == 0) {
          HLOG(kInfo, "[WTC-TRACE] tagmeta-fanout at={} tag={}.{} from={} "
               "sharers={}", container_id_, k.major_, k.minor_, sender,
               it == sharers_.end() ? 0 : it->second.nodes_.size());
        }
        // This home's copy is untracked: another container's write drops it.
        if (sender != container_id_) fan[container_id_].push_back(k);
        if (it == sharers_.end()) continue;
        for (clio::run::u32 n : it->second.nodes_) {
          if (n != sender && n != container_id_) fan[n].push_back(k);
        }
        sharers_.erase(it);
      }
    }
    if (!fan.empty()) {
      CLIO_CO_AWAIT(SendInvalidations(fan));
    }
    task->SetReturnCode(0);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

}  // namespace clio::cte::shm_wtc
