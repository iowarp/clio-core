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
 * Stream runtime, home side: logical sizes (SizeOp) and the merge of
 * deferred appends (Plan -> ExecutePlan). A plan reserves its
 * byte range, is persisted, then copied; it stays open (and is retried at
 * the same offsets) until every copy succeeded.
 */

#include <clio_cte/stream/stream_runtime.h>

#include <algorithm>
#include <cerrno>

namespace clio::cte::stream {

namespace {
/** Poll interval while waiting for a stream's in-flight merge (us). */
constexpr double kBusyPollUs = 100.0;
/** Return code for an I/O failure (matches EIO). */
constexpr clio::run::u32 kRcIo = EIO;
/** Merged-name memory is pruned once it holds this many names. */
constexpr size_t kMaxMergedNames = 1u << 20;
/** Names older than this are pruned from the merged-name memory (ns). */
constexpr clio::run::u64 kMergedNameTtlNs = 600ULL * 1000 * 1000 * 1000;
/** Compact the log once this many bytes were appended since the last. */
constexpr clio::run::u64 kCompactBytes = 64ULL << 20;

/** @return true if `op` must wait for an in-flight merge of the stream. */
bool WaitsForMerge(StreamSizeOp op) {
  return op == StreamSizeOp::kSet || op == StreamSizeOp::kDrop;
}
}  // namespace

// ===========================================================================
// Sizes
// ===========================================================================

bool Runtime::LocalSize(const clio::cte::core::TagId &tag,
                        clio::run::u64 *size) {
  std::lock_guard<std::mutex> g(mu_);
  auto it = streams_.find(tag);
  *size = it == streams_.end() ? 0 : it->second.size_;
  return it != streams_.end();
}

clio::run::u64 Runtime::ApplySizeOpLocked(const clio::cte::core::TagId &tag,
                                          StreamSizeOp op,
                                          clio::run::u64 value,
                                          clio::run::u64 *old_size) {
  auto it = streams_.find(tag);
  *old_size = it == streams_.end() ? 0 : it->second.size_;
  if (op == StreamSizeOp::kGet) return *old_size;

  if (op == StreamSizeOp::kDrop) {
    if (it != streams_.end()) streams_.erase(it);
    dropped_[tag] = clio::cte::core::GetWallTimeNs();
    LogDropLocked(tag);
    return 0;
  }
  dropped_.erase(tag);  // a size mutation (re)creates the stream
  StreamState &st = streams_[tag];
  const clio::run::u64 before = st.size_;
  if (op == StreamSizeOp::kMax) {
    st.size_ = std::max(st.size_, value);
  } else if (op == StreamSizeOp::kReserve) {
    st.size_ += value;
  } else if (op == StreamSizeOp::kSet) {
    st.size_ = value;
  }
  if (st.size_ != before || it == streams_.end()) LogSizeLocked(tag, st.size_);
  if (log_.BytesSinceCompact() > kCompactBytes) CompactLocked();
  return st.size_;
}

clio::run::TaskResume Runtime::SizeOp(clio::run::shared_ptr<SizeOpTask> &task) {
  CLIO_TASK_BODY_BEGIN
  const auto op = static_cast<StreamSizeOp>(task->op_);
  if (op > StreamSizeOp::kDrop) {
    task->return_code_ = EINVAL;
    CLIO_CO_RETURN;
  }
  for (;;) {
    {
      std::lock_guard<std::mutex> g(mu_);
      auto it = streams_.find(task->tag_id_);
      const bool busy = it != streams_.end() && it->second.busy_;
      if (!busy || !WaitsForMerge(op)) {
        task->new_size_ = ApplySizeOpLocked(task->tag_id_, op, task->value_,
                                            &task->old_size_);
        break;
      }
    }
    // Truncate and drop order after an in-flight merge, never inside it.
    CLIO_CO_AWAIT(clio::run::yield(kBusyPollUs));
  }
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

// ===========================================================================
// Merge
// ===========================================================================

void Runtime::DedupeLocked(std::vector<AppendEntry> *entries) {
  if (merged_names_.size() > kMaxMergedNames) {
    const clio::run::u64 cutoff =
        clio::cte::core::GetWallTimeNs() - kMergedNameTtlNs;
    for (auto it = merged_names_.begin(); it != merged_names_.end();) {
      it = it->second < cutoff ? merged_names_.erase(it) : std::next(it);
    }
  }
  std::vector<AppendEntry> fresh;
  fresh.reserve(entries->size());
  for (auto &e : *entries) {
    if (merged_names_.emplace(e.staged_name_, e.clock_).second) {
      fresh.push_back(std::move(e));
    }
  }
  entries->swap(fresh);
}

clio::run::TaskResume Runtime::Plan(clio::run::shared_ptr<PlanTask> &task) {
  CLIO_TASK_BODY_BEGIN
  const clio::cte::core::TagId tag = task->tag_id_;
  std::vector<AppendEntry> entries(task->entries_.begin(),
                                   task->entries_.end());
  bool dropped = false;
  std::vector<StreamPlan> retry;
  for (;;) {  // claim the stream: one merge at a time
    {
      std::lock_guard<std::mutex> g(mu_);
      if (dropped_.count(tag) != 0) {
        dropped = true;
        break;
      }
      StreamState &st = streams_[tag];
      if (!st.busy_) {
        st.busy_ = true;
        for (auto &kv : open_plans_) {
          if (kv.second.tag_ == tag) retry.push_back(kv.second);
        }
        break;
      }
    }
    CLIO_CO_AWAIT(clio::run::yield(kBusyPollUs));
  }
  if (dropped) {  // appends that raced an unlink: discard their bytes
    CLIO_CO_AWAIT(DeleteStaged(entries));
    task->return_code_ = 0;
    CLIO_CO_RETURN;
  }
  bool ok = true;
  for (const auto &old : retry) {  // earlier plans whose copy failed
    bool rok = true;
    CLIO_CO_AWAIT(ExecutePlan(old, &rok));
    std::lock_guard<std::mutex> g(mu_);
    if (rok) {
      LogDoneLocked(old.id_);
      open_plans_.erase(old.id_);
    }
    ok = ok && rok;
  }
  StreamPlan plan;
  {
    std::lock_guard<std::mutex> g(mu_);
    DedupeLocked(&entries);
    std::sort(entries.begin(), entries.end(),
              [](const AppendEntry &a, const AppendEntry &b) {
                if (a.clock_ != b.clock_) return a.clock_ < b.clock_;
                if (a.origin_ != b.origin_) return a.origin_ < b.origin_;
                return a.counter_ < b.counter_;
              });
    StreamState &st = streams_[tag];
    plan.id_ = next_plan_id_++;
    plan.tag_ = tag;
    plan.base_ = st.size_;
    plan.entries_ = std::move(entries);
    for (const auto &e : plan.entries_) st.size_ += e.size_;
    if (!plan.entries_.empty()) {
      open_plans_[plan.id_] = plan;
      LogPlanLocked(plan);
      LogSizeLocked(tag, st.size_);
    }
  }
  if (!plan.entries_.empty()) {
    bool pok = true;
    CLIO_CO_AWAIT(ExecutePlan(plan, &pok));
    std::lock_guard<std::mutex> g(mu_);
    if (pok) {
      LogDoneLocked(plan.id_);
      open_plans_.erase(plan.id_);
    }
    ok = ok && pok;
  }
  {
    std::lock_guard<std::mutex> g(mu_);
    StreamState &st = streams_[tag];
    st.busy_ = false;
    task->new_size_ = st.size_;
  }
  task->return_code_ = ok ? 0 : kRcIo;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::ExecutePlan(StreamPlan plan, bool *ok) {
  CLIO_TASK_BODY_BEGIN
  *ok = true;
  size_t i = 0;
  clio::run::u64 off = plan.base_;
  while (i < plan.entries_.size()) {
    size_t j = i;
    clio::run::u64 bytes = 0;
    while (j < plan.entries_.size() &&
           (j == i || bytes + plan.entries_[j].size_ <= kMaxSliceBytes)) {
      bytes += plan.entries_[j].size_;
      ++j;
    }
    bool sok = true;
    CLIO_CO_AWAIT(CopySlice(&plan, i, j, off, &sok));
    *ok = *ok && sok;
    off += bytes;
    i = j;
  }
  // Staged bytes are dropped only once every copy landed; a failed plan
  // keeps them for the retry.
  if (*ok) CLIO_CO_AWAIT(DeleteStaged(plan.entries_));
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::CopySlice(const StreamPlan *plan, size_t first,
                                         size_t last, clio::run::u64 off,
                                         bool *ok) {
  CLIO_TASK_BODY_BEGIN
  auto *ipc = CLIO_IPC;
  clio::run::u64 bytes = 0;
  for (size_t k = first; k < last; ++k) bytes += plan->entries_[k].size_;
  ctp::ipc::FullPtr<char> buf = ipc->AllocateBuffer(bytes);
  if (buf.IsNull()) {
    *ok = false;
    CLIO_CO_RETURN;
  }
  // Read every staged append of the slice; runs of present ones are written
  // page by page. A staged blob that is gone was merged before a crash (this
  // is a replayed plan): its range is already in place and is skipped.
  std::vector<bool> have(last - first, false);
  clio::run::u64 pos = 0;
  for (size_t k = first; k < last; ++k) {
    const AppendEntry &e = plan->entries_[k];
    auto g = cte_.AsyncGetBlob(staging_tag_, e.staged_name_, 0, e.size_, 0u,
                               buf.ptr_ + pos);
    CLIO_CO_AWAIT(g);
    have[k - first] = g->GetReturnCode() == 0;
    if (!have[k - first]) {
      HLOG(kWarning, "stream: staged append {} of {}.{} unreadable (rc {}); "
           "treating it as merged before a restart", e.staged_name_,
           plan->tag_.major_, plan->tag_.minor_, g->GetReturnCode());
    }
    pos += e.size_;
  }
  size_t k = first;
  pos = 0;
  while (k < last && *ok) {
    if (!have[k - first]) {
      pos += plan->entries_[k].size_;
      ++k;
      continue;
    }
    const clio::run::u64 run_pos = pos;
    while (k < last && have[k - first]) pos += plan->entries_[k++].size_;
    clio::run::u64 cur = off + run_pos;
    const clio::run::u64 end = off + pos;
    while (cur < end) {
      const clio::run::u64 page_off = cur % kStreamPageSize;
      const clio::run::u64 n = std::min(kStreamPageSize - page_off, end - cur);
      auto p = cte_.AsyncPutBlob(plan->tag_, StreamPageName(cur), page_off, n,
                                 buf.ptr_ + (cur - off), -1.0f,
                                 clio::cte::core::Context(), 0u);
      CLIO_CO_AWAIT(p);
      if (p->GetReturnCode() != 0) {
        *ok = false;
        break;
      }
      cur += n;
    }
  }
  ipc->FreeBuffer(buf);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::DeleteStaged(std::vector<AppendEntry> entries) {
  CLIO_TASK_BODY_BEGIN
  for (const auto &e : entries) {
    auto d = cte_.AsyncDelBlob(staging_tag_, e.staged_name_);
    CLIO_CO_AWAIT(d);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

}  // namespace clio::cte::stream
