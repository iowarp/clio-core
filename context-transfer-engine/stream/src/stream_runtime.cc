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
 * Flush, Sequence) and restart recovery. The home side (SizeOp,
 * Plan and the merge) is in stream_home.cc; persistence in stream_log.cc.
 */

#include <clio_cte/stream/stream_runtime.h>
#include <clio_cte/core/blob_placement.h>

#include <algorithm>
#include <chrono>
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
/** Tries (one per kStagingTagRetryUs) to register the staging tag with its
 *  owner at startup; past them the well-known id is used (#1141). */
constexpr int kStagingTagTries = 5;
/** The staging tag's well-known id: major 0x3FFFFFFF is outside every range
 *  a node, container or client mints (the root directory is minor 1). */
const clio::cte::core::TagId kStagingTagId(0x3FFFFFFFu, 2u);
/** Pause between staging-tag tries (us). */
constexpr double kStagingTagRetryUs = 1e6;
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
  staging_ = clio::cte::core::Client(config_.staging_pool_id_.IsNull()
                                         ? cte_.pool_id_
                                         : config_.staging_pool_id_);
  self_.Init(task->new_pool_id_);
  auto *ipc = CLIO_IPC;
  node_ = ipc->GetNodeId();
  {
    // The staging tag has a WELL-KNOWN id, so a node never needs the tag's
    // owner to learn it: its blobs route by blob hash and fail over like any
    // other, while the name->id lookup goes only to the owner and does NOT
    // fail over. A node restarting while that owner was down used to abort
    // its whole runtime here (#1141). Registering the name with its owner is
    // best effort: an owner that already holds an id for it (a deployment
    // older than the well-known id) is adopted; an unreachable one is not
    // waited for.
    staging_tag_ = kStagingTagId;
    for (int attempt = 0; attempt < kStagingTagTries; ++attempt) {
      auto st = staging_.AsyncGetOrCreateTag(kStagingTagName, kStagingTagId,
                                         clio::run::PoolQuery::Dynamic());
      CLIO_CO_AWAIT(st);
      if (st->GetReturnCode() == 0) {
        staging_tag_ = st->tag_id_;
        break;
      }
      if (attempt + 1 == kStagingTagTries) {
        HLOG(kWarning, "stream: the owner of staging tag {} is unreachable "
             "(rc {}); using its well-known id {}.{}", kStagingTagName,
             st->GetReturnCode(), kStagingTagId.major_, kStagingTagId.minor_);
        break;
      }
      CLIO_CO_AWAIT(clio::run::yield(kStagingTagRetryUs));
    }
  }
  OpenLog();
  if (is_restart_) {
    // Merge plans interrupted by the restart finish later, from the drain
    // tick, once clio-fs has reconciled the restored streams: a stream whose
    // file was truncated elsewhere meanwhile must drop its old plans, not
    // replay them at their pre-crash offsets (see ReconcileRestored).
    {
      std::lock_guard<std::mutex> g(mu_);
      plans_pending_ = !open_plans_.empty();
    }
    CLIO_CO_AWAIT(RecoverStaged());
    if (plans_pending_) EnsureSequence();
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

clio::run::u32 Runtime::LiveHome(clio::run::u32 home) const {
  // While the home's node is dead its successor serves the stream (the rule
  // the CTE and the filesystem use for everything the home owns); sending to
  // the dead node waits forever.
  return clio::cte::core::FailoverContainer(pool_id_, home);
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
  {
    auto *ipc = CLIO_IPC;
    const char *src = ipc->ToFullPtr<char>(task->data_.template Cast<char>()).ptr_;
    p.data_.assign(src, task->size_);  // shipped inline to the home
  }
  // The staged blob is the durable record of the append: once this put
  // returns, a crash anywhere cannot lose it (RecoverStaged re-queues it).
  auto put = staging_.AsyncPutBlob(staging_tag_, p.entry_.staged_name_, 0,
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
  const auto t0 = std::chrono::steady_clock::now();
  for (;;) {
    bool done = true;
    {
      std::lock_guard<std::mutex> g(q_mu_);
      auto it = local_pending_.find(task->tag_id_);
      done = it == local_pending_.end() || it->second.empty() ||
             *it->second.begin() > target;
    }
    if (done) break;
    const double waited_us = std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - t0).count();
    if (waited_us >= kFlushDeadlineUs) {
      HLOG(kError, "stream: flush of {}.{} timed out with appends unmerged",
           task->tag_id_.major_, task->tag_id_.minor_);
      task->return_code_ = kRcTimeout;
      CLIO_CO_RETURN;
    }
    CLIO_CO_AWAIT(clio::run::yield(kFlushPollUs));
  }
  auto s = self_.AsyncSizeOp(task->tag_id_, LiveHome(task->home_),
                             StreamSizeOp::kGet);
  CLIO_CO_AWAIT(s);
  task->size_ = s->new_size_;
  task->return_code_ = s->GetReturnCode();
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

void Runtime::SyncLogPeriodically() {
  const auto now = std::chrono::steady_clock::now();
  if (now - last_log_sync_ < std::chrono::milliseconds(kLogSyncPeriodMs)) {
    return;
  }
  last_log_sync_ = now;
  if (log_.Unsynced()) log_.Sync();
}

clio::run::TaskResume Runtime::Sequence(
    clio::run::shared_ptr<SequenceTask> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    bool finish = false;
    {
      std::lock_guard<std::mutex> g(mu_);
      if (plans_pending_ && (unverified_.empty() || !GatedAny())) {
        plans_pending_ = false;
        finish = true;
      }
    }
    if (finish) CLIO_CO_AWAIT(FinishOpenPlans());
  }
  SyncLogPeriodically();
  const clio::run::u64 now = clio::cte::core::GetWallTimeNs();
  std::vector<PendingAppend> ready;
  {
    std::lock_guard<std::mutex> g(q_mu_);
    std::vector<PendingAppend> later;
    for (auto it = hold_until_.begin(); it != hold_until_.end();) {
      it = it->second <= now ? hold_until_.erase(it) : std::next(it);
    }
    for (auto &p : queue_) {
      const bool held = hold_until_.count(p.tag_) != 0;
      (held ? later : ready).push_back(std::move(p));
    }
    queue_.swap(later);
  }
  // Group by (stream, home), keeping each node's order within a group.
  std::map<std::pair<clio::run::u64, clio::run::u32>, std::vector<PendingAppend>>
      groups;
  for (auto &p : ready) {
    groups[{Client::TagKey(p.tag_), p.home_}].push_back(std::move(p));
  }
  // Each stream's batch goes straight to its home in chunks of at most
  // kMaxSliceBytes of inline bytes, awaited one at a time (a coroutine may
  // have only one outstanding subtask future). The home serializes merges
  // per stream, so batches from several nodes interleave safely.
  for (auto &kv : groups) {
    std::vector<PendingAppend> &batch = kv.second;
    size_t i = 0;
    while (i < batch.size()) {
      std::vector<PendingAppend> chunk;
      clio::run::u64 bytes = 0;
      while (i < batch.size() &&
             (chunk.empty() || bytes + batch[i].data_.size() <= kMaxSliceBytes)) {
        bytes += batch[i].data_.size();
        chunk.push_back(std::move(batch[i++]));
      }
      bool ok = false;
      CLIO_CO_AWAIT(ShipChunk(&chunk, &ok));
      if (ok) continue;
      // Home unreachable or the merge failed: retry this chunk and the rest
      // of the batch later, ahead of anything newer for the stream (held
      // until then). The home dedupes by staged name, so a retry never
      // merges twice.
      std::lock_guard<std::mutex> g(q_mu_);
      hold_until_[chunk.front().tag_] = now + kShipRetryNs;
      std::vector<PendingAppend> requeue(std::make_move_iterator(chunk.begin()),
                                         std::make_move_iterator(chunk.end()));
      for (; i < batch.size(); ++i) requeue.push_back(std::move(batch[i]));
      for (auto &p : queue_) requeue.push_back(std::move(p));
      queue_.swap(requeue);
    }
  }
  CLIO_CO_AWAIT(ReapStaged());
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::ShipChunk(
    const std::vector<PendingAppend> *chunk, bool *ok) {
  CLIO_TASK_BODY_BEGIN
  std::vector<AppendEntry> entries;
  std::string payload;
  entries.reserve(chunk->size());
  for (const auto &p : *chunk) {
    AppendEntry e = p.entry_;
    if (p.data_.size() == e.size_) {
      e.payload_off_ = payload.size();
      payload += p.data_;
    } else {
      e.payload_off_ = kNoPayload;  // re-queued after a restart: staged only
    }
    entries.push_back(std::move(e));
  }
  auto f = self_.AsyncPlan(chunk->front().tag_,
                           LiveHome(chunk->front().home_), entries, payload);
  CLIO_CO_AWAIT(f);
  *ok = f->GetReturnCode() == 0;
  if (*ok) {
    std::lock_guard<std::mutex> g(q_mu_);
    auto it = local_pending_.find(chunk->front().tag_);
    if (it != local_pending_.end()) {
      for (const auto &p : *chunk) it->second.erase(p.local_seq_);
      if (it->second.empty()) local_pending_.erase(it);
    }
    pending_count_.fetch_sub(chunk->size());
  }
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
    CLIO_CO_AWAIT(ExecutePlan(plan, nullptr, &ok));
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
  auto list = staging_.AsyncGetContainedBlobs(staging_tag_,
                                          clio::run::PoolQuery::Broadcast());
  CLIO_CO_AWAIT(list);
  size_t requeued = 0;
  for (const auto &name : list->blob_names_) {
    PendingAppend p;
    if (!ParseStagedName(name, &p.tag_, &p.home_, &p.entry_)) continue;
    // Re-ship only what this node accepted: its queue died with it. Appends
    // other nodes accepted are still queued there and are retried until
    // their home confirms the merge; shipping them from here too would race
    // those retries and reorder a writer's appends.
    if (p.entry_.origin_ != node_) continue;
    auto sz = staging_.AsyncGetBlobSize(staging_tag_, name);
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
