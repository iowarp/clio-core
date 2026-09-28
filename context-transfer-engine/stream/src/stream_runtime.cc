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
 * Stream runtime: lifecycle, the origin side of deferred appends (Append,
 * Flush, Sequence) and restart recovery. The home side (SizeOp, Collect,
 * Plan and the merge) is in stream_home.cc; persistence in stream_log.cc.
 */

#include <clio_cte/stream/stream_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <map>
#include <utility>

namespace clio::cte::stream {

namespace {
/** Longest a Flush waits for its merges before failing (microseconds). */
constexpr double kFlushDeadlineUs = 120.0 * 1e6;
/** Poll interval of a waiting Flush (microseconds). */
constexpr double kFlushPollUs = 200.0;
/** Backoff before re-shipping a batch whose home did not answer (ns). */
constexpr clio::run::u64 kShipRetryNs = 100ULL * 1000 * 1000;
/** Return code for an I/O failure (matches EIO). */
constexpr clio::run::u32 kRcIo = EIO;
/** Return code for a flush that timed out (matches ETIMEDOUT). */
constexpr clio::run::u32 kRcTimeout = ETIMEDOUT;
}  // namespace

// ===========================================================================
// Lifecycle
// ===========================================================================

clio::run::TaskResume Runtime::Create(clio::run::shared_ptr<CreateTask> &task) {
  CLIO_TASK_BODY_BEGIN
  config_ = task->GetParams();
  cte_ = clio::cte::core::Client(config_.next_pool_id_.IsNull()
                                     ? clio::cte::core::kCtePoolId
                                     : config_.next_pool_id_);
  self_.Init(task->new_pool_id_);
  auto *ipc = CLIO_IPC;
  node_ = ipc->GetNodeId();
  {
    auto st = cte_.AsyncGetOrCreateTag(kStagingTagName,
                                       clio::cte::core::TagId::GetNull(),
                                       clio::run::PoolQuery::Dynamic());
    CLIO_CO_AWAIT(st);
    if (st->GetReturnCode() != 0) {
      HLOG(kError, "stream: cannot create staging tag {}", kStagingTagName);
      task->return_code_ = kRcIo;
      CLIO_CO_RETURN;
    }
    staging_tag_ = st->tag_id_;
  }
  OpenLog();
  if (is_restart_) {
    CLIO_CO_AWAIT(FinishOpenPlans());
    CLIO_CO_AWAIT(RecoverStaged());
  }
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Destroy(
    clio::run::shared_ptr<DestroyTask> &task) {
  CLIO_TASK_BODY_BEGIN
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Monitor(
    clio::run::shared_ptr<MonitorTask> &task) {
  CLIO_TASK_BODY_BEGIN
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

// ===========================================================================
// Staged names
// ===========================================================================

std::string Runtime::MakeStagedName(const clio::cte::core::TagId &tag,
                                    clio::run::u32 home,
                                    const AppendEntry &e) {
  char buf[160];
  std::snprintf(buf, sizeof(buf), "sa.%x.%x.%x.%llx.%x.%llx", tag.major_,
                tag.minor_, home, static_cast<unsigned long long>(e.clock_),
                e.origin_, static_cast<unsigned long long>(e.counter_));
  return buf;
}

bool Runtime::ParseStagedName(const std::string &name,
                              clio::cte::core::TagId *tag,
                              clio::run::u32 *home, AppendEntry *e) {
  unsigned major = 0, minor = 0, h = 0, origin = 0;
  unsigned long long clock = 0, counter = 0;
  int consumed = 0;
  if (std::sscanf(name.c_str(), "sa.%x.%x.%x.%llx.%x.%llx%n", &major, &minor,
                  &h, &clock, &origin, &counter, &consumed) != 6 ||
      static_cast<size_t>(consumed) != name.size()) {
    return false;
  }
  tag->major_ = major;
  tag->minor_ = minor;
  *home = h;
  e->staged_name_ = name;
  e->clock_ = clock;
  e->origin_ = origin;
  e->counter_ = counter;
  return true;
}

// ===========================================================================
// Origin side
// ===========================================================================

void Runtime::EnsureSequence() {
  bool start = false;
  {
    std::lock_guard<std::mutex> g(q_mu_);
    if (!seq_started_) {
      seq_started_ = true;
      start = true;
    }
  }
  if (start) self_.AsyncSequence(kSequencePeriodUs);
}

void Runtime::Enqueue(PendingAppend p) {
  {
    std::lock_guard<std::mutex> g(q_mu_);
    p.local_seq_ = ++enq_seq_;
    local_pending_[p.tag_].insert(p.local_seq_);
    queue_.push_back(std::move(p));
  }
  pending_count_.fetch_add(1);
  EnsureSequence();
}

clio::run::TaskResume Runtime::Append(clio::run::shared_ptr<AppendTask> &task) {
  CLIO_TASK_BODY_BEGIN
  task->bytes_written_ = 0;
  if (task->size_ == 0) {
    task->return_code_ = 0;
    CLIO_CO_RETURN;
  }
  PendingAppend p;
  p.tag_ = task->tag_id_;
  p.home_ = task->home_;
  p.entry_.size_ = task->size_;
  p.entry_.origin_ = node_;
  {
    // Monotonic per node even if the wall clock steps back: one writer's
    // appends are never reordered by the merge.
    std::lock_guard<std::mutex> g(q_mu_);
    last_clock_ = std::max(last_clock_ + 1, clio::cte::core::GetWallTimeNs());
    p.entry_.clock_ = last_clock_;
    p.entry_.counter_ = ++counter_;
  }
  p.entry_.staged_name_ = MakeStagedName(p.tag_, p.home_, p.entry_);
  // The staged blob is the durable record of the append: once this put
  // returns, a crash anywhere cannot lose it (RecoverStaged re-queues it).
  auto put = cte_.AsyncPutBlob(staging_tag_, p.entry_.staged_name_, 0,
                               task->size_, task->data_, -1.0f,
                               clio::cte::core::Context(), 0u,
                               clio::run::PoolQuery::Dynamic());
  CLIO_CO_AWAIT(put);
  if (put->GetReturnCode() != 0) {
    task->return_code_ = kRcIo;
    CLIO_CO_RETURN;
  }
  Enqueue(p);
  task->bytes_written_ = task->size_;
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Flush(clio::run::shared_ptr<FlushTask> &task) {
  CLIO_TASK_BODY_BEGIN
  // Wait only for the appends accepted BEFORE this flush: later writers must
  // not be able to starve it.
  clio::run::u64 target = 0;
  {
    std::lock_guard<std::mutex> g(q_mu_);
    target = enq_seq_;
  }
  double waited_us = 0.0;
  for (;;) {
    bool done = true;
    {
      std::lock_guard<std::mutex> g(q_mu_);
      auto it = local_pending_.find(task->tag_id_);
      done = it == local_pending_.end() || it->second.empty() ||
             *it->second.begin() > target;
    }
    if (done) break;
    if (waited_us >= kFlushDeadlineUs) {
      HLOG(kError, "stream: flush of {}.{} timed out with appends unmerged",
           task->tag_id_.major_, task->tag_id_.minor_);
      task->return_code_ = kRcTimeout;
      CLIO_CO_RETURN;
    }
    CLIO_CO_AWAIT(clio::run::yield(kFlushPollUs));
    waited_us += kFlushPollUs;
  }
  auto s = self_.AsyncSizeOp(task->tag_id_, task->home_, StreamSizeOp::kGet);
  CLIO_CO_AWAIT(s);
  task->size_ = s->new_size_;
  task->return_code_ = s->GetReturnCode();
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Sequence(
    clio::run::shared_ptr<SequenceTask> &task) {
  CLIO_TASK_BODY_BEGIN
  const clio::run::u64 now = clio::cte::core::GetWallTimeNs();
  std::vector<PendingAppend> ready;
  {
    std::lock_guard<std::mutex> g(q_mu_);
    std::vector<PendingAppend> later;
    for (auto &p : queue_) {
      (p.not_before_ns_ <= now ? ready : later).push_back(std::move(p));
    }
    queue_.swap(later);
  }
  // Group by (stream, home), keeping each node's order within a group.
  std::map<std::pair<clio::run::u64, clio::run::u32>, std::vector<PendingAppend>>
      groups;
  for (auto &p : ready) {
    groups[{Client::TagKey(p.tag_), p.home_}].push_back(std::move(p));
  }
  // One Collect per stream, awaited one at a time (a coroutine may have only
  // one outstanding subtask future).
  for (auto &kv : groups) {
    std::vector<PendingAppend> &batch = kv.second;
    std::vector<AppendEntry> entries;
    entries.reserve(batch.size());
    for (const auto &p : batch) entries.push_back(p.entry_);
    auto f = self_.AsyncCollect(batch.front().tag_, batch.front().home_,
                                entries);
    CLIO_CO_AWAIT(f);
    std::lock_guard<std::mutex> g(q_mu_);
    if (f->GetReturnCode() != 0) {
      // Home unreachable or the merge failed: retry the batch later. The
      // home dedupes by staged name, so a retry never merges twice.
      for (auto &p : batch) {
        p.not_before_ns_ = now + kShipRetryNs;
        queue_.push_back(std::move(p));
      }
      continue;
    }
    auto it = local_pending_.find(batch.front().tag_);
    if (it != local_pending_.end()) {
      for (const auto &p : batch) it->second.erase(p.local_seq_);
      if (it->second.empty()) local_pending_.erase(it);
    }
    pending_count_.fetch_sub(batch.size());
  }
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

// ===========================================================================
// Restart recovery
// ===========================================================================

clio::run::TaskResume Runtime::FinishOpenPlans() {
  CLIO_TASK_BODY_BEGIN
  std::vector<StreamPlan> plans;
  {
    std::lock_guard<std::mutex> g(mu_);
    for (auto &kv : open_plans_) plans.push_back(kv.second);
  }
  std::sort(plans.begin(), plans.end(),
            [](const StreamPlan &a, const StreamPlan &b) { return a.id_ < b.id_; });
  for (const auto &plan : plans) {
    bool ok = true;
    CLIO_CO_AWAIT(ExecutePlan(plan, &ok));
    std::lock_guard<std::mutex> g(mu_);
    LogDoneLocked(plan.id_);
    open_plans_.erase(plan.id_);
  }
  if (!plans.empty()) {
    HLOG(kInfo, "stream: finished {} merge plan(s) interrupted by a restart",
         plans.size());
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::RecoverStaged() {
  CLIO_TASK_BODY_BEGIN
  auto list = cte_.AsyncGetContainedBlobs(staging_tag_,
                                          clio::run::PoolQuery::Broadcast());
  CLIO_CO_AWAIT(list);
  size_t requeued = 0;
  for (const auto &name : list->blob_names_) {
    PendingAppend p;
    if (!ParseStagedName(name, &p.tag_, &p.home_, &p.entry_)) continue;
    // Re-ship what this node accepted (its queue died with it) and what it
    // homes (in-flight batches aimed at it died with it). The home dedupes
    // by name, so an entry recovered by both nodes merges once.
    if (p.entry_.origin_ != node_ && p.home_ != node_) continue;
    auto sz = cte_.AsyncGetBlobSize(staging_tag_, name);
    CLIO_CO_AWAIT(sz);
    if (sz->GetReturnCode() != 0 || sz->size_ == 0) continue;
    p.entry_.size_ = sz->size_;
    Enqueue(p);
    ++requeued;
  }
  if (requeued != 0) {
    HLOG(kInfo, "stream: re-queued {} staged append(s) after restart",
         requeued);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

}  // namespace clio::cte::stream

CLIO_TASK_CC(clio::cte::stream::Runtime)
