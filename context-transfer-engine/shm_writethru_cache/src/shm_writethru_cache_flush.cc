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
 * shm_writethru_cache flusher and coherence.
 *
 * FlushLog is the dirty log's only consumer. One pass pops a batch of
 * ranges, coalesces them per blob, copies each coalesced range's CURRENT
 * bytes into one staging buffer per home container, and ships each group as
 * ONE MultiPutBlob: straight down the chain when this container is the home,
 * else as a HomeFlush to the home, which forwards it down and invalidates
 * the other containers caching those blobs before it acks.
 */

#include <algorithm>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <clio_cte/core/blob_batch.h>
#include <clio_cte/shm_writethru_cache/shm_writethru_cache_runtime.h>

namespace clio::cte::shm_wtc {

namespace wtc = clio::cte::core::wtc;
using clio::cte::core::Context;
using clio::cte::core::MultiPutBlobTask;
using clio::cte::core::MultiPutDesc;

static constexpr size_t kMaxRangesPerFlush = 8192;
static constexpr clio::run::u32 kWriterMarker = 0xFFFFFFF7u;

std::vector<FlushRecord> Runtime::PopAndCoalesce(clio::run::u64 *popped) {
  std::vector<wtc::DirtyRange> rs;
  clio::run::u64 bytes = 0;
  wtc::DirtyRange r{};
  wtc::DirtyLog *log = view_.Log();
  while (rs.size() < kMaxRangesPerFlush && bytes < config_.flush_batch_bytes_ &&
         log->Pop(r)) {
    rs.push_back(r);
    bytes += r.size_;
  }
  *popped = rs.size();
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

void Runtime::StageRecords(
    std::vector<FlushRecord> &recs,
    std::map<clio::run::u32, ctp::ipc::FullPtr<char>> *staging,
    std::map<clio::run::u32, clio::run::u64> *staged) {
  std::map<clio::run::u32, clio::run::u64> totals;
  for (FlushRecord &fr : recs) {
    clio::run::u64 none = 0;
    std::string name;
    if (!view_.ReadRange(fr.slot_, fr.off_, &none, nullptr, &fr.key_.major_,
                         &fr.key_.minor_, &name, &fr.put_flags_)) {
      fr.size_ = 0;
      continue;
    }
    fr.key_.name_ = std::move(name);
    fr.home_ = HomeOf(TagId(fr.key_.major_, fr.key_.minor_), fr.key_.name_);
    totals[fr.home_] += fr.size_;
  }
  auto *ipc = CLIO_IPC;
  for (const auto &t : totals) {
    if (t.second > 0) {
      (*staging)[t.first] = ipc->AllocateBuffer(t.second);
      (*staged)[t.first] = 0;
    }
  }
  for (FlushRecord &fr : recs) {
    auto it = staging->find(fr.home_);
    if (fr.size_ == 0 || it == staging->end() || it->second.IsNull()) {
      fr.size_ = 0;
      continue;
    }
    clio::run::u64 &cursor = (*staged)[fr.home_];
    clio::run::u64 size = fr.size_;
    std::string ignored;
    clio::run::u32 a = 0;
    clio::run::u32 b = 0;
    clio::run::u32 f = 0;
    if (!view_.ReadRange(fr.slot_, fr.off_, &size, it->second.ptr_ + cursor,
                         &a, &b, &ignored, &f)) {
      size = 0;
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
      auto f = Next()->AsyncPutBlob(
          TagId(recs[i].key_.major_, recs[i].key_.minor_), recs[i].key_.name_,
          recs[i].off_, recs[i].size_,
          ctp::ipc::ShmPtr<>::FromRaw(staging.ptr_ + recs[i].payload_off_),
          -1.0f, clio::cte::core::Context(), recs[i].put_flags_);
      CLIO_CO_AWAIT(f);
      rc = f->GetReturnCode();
    }
    if (rc != 0) {
      HLOG(kError, "shm_writethru_cache: flush of {}.{}/{} failed rc={}",
           recs[i].key_.major_, recs[i].key_.minor_, recs[i].key_.name_, rc);
      view_.SetTagError(recs[i].key_.major_, recs[i].key_.minor_, rc);
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::FlushLog(clio::run::shared_ptr<FlushLogTask> &task) {
  CLIO_TASK_BODY_BEGIN
  task->ranges_ = 0;
  task->bytes_ = 0;
  if (!view_.Ready() || view_.Log()->Empty() || flushing_.exchange(true)) {
    task->SetReturnCode(0);
    CLIO_CO_RETURN;
  }
  {
    clio::run::u64 popped = 0;
    std::vector<FlushRecord> recs = PopAndCoalesce(&popped);
    std::map<clio::run::u32, ctp::ipc::FullPtr<char>> staging;
    std::map<clio::run::u32, clio::run::u64> staged;
    StageRecords(recs, &staging, &staged);
    auto *ipc = CLIO_IPC;
    std::vector<clio::run::u32> homes;
    std::vector<clio::run::Future<MultiPutBlobTask>> futs;
    for (const auto &g : staging) {
      if (g.second.IsNull()) {
        continue;
      }
      std::vector<MultiPutDesc> descs = GroupDescs(recs, g.first);
      if (descs.empty()) {
        continue;
      }
      const bool local = g.first == container_id_;
      auto t = ipc->NewTask<MultiPutBlobTask>(
          clio::run::CreateTaskId(), local ? CorePoolId() : pool_id_,
          local ? clio::run::PoolQuery::Local()
                : clio::run::PoolQuery::DirectId(g.first),
          local ? descs.front().tag_id_ : TagId(container_id_, kWriterMarker),
          descs.front().blob_name_, ctp::ipc::ShmPtr<>(g.second.shm_),
          staged[g.first], clio::cte::core::EncodeMultiPutDescs(descs),
          Context());
      if (!local) {
        t->method_ = Method::kHomeFlush;
      }
      task->bytes_ += staged[g.first];
      homes.push_back(g.first);
      futs.push_back(ipc->Send(t));
    }
    for (size_t i = 0; i < futs.size(); ++i) {
      CLIO_CO_AWAIT(futs[i]);
      if (futs[i]->GetReturnCode() != 0 || futs[i]->first_rc_ != 0) {
        CLIO_CO_AWAIT(RetryGroup(recs, homes[i], staging[homes[i]]));
      }
      if (homes[i] == container_id_) {
        std::vector<BlobKey> keys;
        for (const FlushRecord &fr : recs) {
          if (fr.home_ == homes[i] && fr.size_ > 0) keys.push_back(fr.key_);
        }
        CLIO_CO_AWAIT(SendInvalidations(HomeApply(keys, container_id_)));
      }
    }
    for (auto &g : staging) {
      if (!g.second.IsNull()) ipc->FreeBuffer(g.second);
    }
    for (const FlushRecord &fr : recs) {
      view_.CompleteRanges(fr.slot_, fr.ranges_);
    }
    wtc::Root *root = view_.GetRoot();
    root->stats_.flush_batches_.fetch_add(futs.size());
    root->stats_.flush_ranges_.fetch_add(popped);
    root->stats_.flush_bytes_.fetch_add(task->bytes_);
    root->flushed_.fetch_add(popped, std::memory_order_acq_rel);
    task->ranges_ = popped;
  }
  flushing_.store(false);
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
  for (const BlobKey &k : keys) {
    SharerSet &s = sharers_[SharerKey(k.major_, k.minor_, k.name_)];
    bool foreign_writer = false;
    for (clio::run::u32 w : s.writers_) {
      foreign_writer = foreign_writer || w != writer;
    }
    for (clio::run::u32 n : s.nodes_) {
      if (n != writer) by_node[n].push_back(k);
    }
    if (foreign_writer) {
      // The writer's buffer may lack another writer's bytes.
      by_node[writer].push_back(k);
    }
    s.nodes_.clear();
    s.writers_.clear();
    s.nodes_.insert(writer);
    s.writers_.insert(writer);
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
        for (const BlobKey &k : e.second) {
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
    std::vector<BlobKey> keys;
    for (const MultiPutDesc &d : descs) {
      keys.push_back(BlobKey{d.tag_id_.major_, d.tag_id_.minor_, d.blob_name_});
    }
    CLIO_CO_AWAIT(SendInvalidations(HomeApply(keys, writer)));
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
        view_.Invalidate(k.major_, k.minor_, k.name_.c_str(), k.name_.size());
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
        if (it == sharers_.end()) continue;
        for (clio::run::u32 n : it->second.nodes_) {
          if (n != sender) fan[n].push_back(k);
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
