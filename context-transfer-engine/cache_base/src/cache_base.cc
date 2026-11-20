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
 * Persistent-tier caches: owner-local write-through and write-back policies
 * over a volatile core and a persistent chain. See cache_base.h.
 */

#include <clio_cte/cache_base/cache_base.h>
#include <clio_cte/core/blob_placement.h>

#include <clio_runtime/evlat.h>
#include <clio_ctp/serialize/msgpack_wrapper.h>

#include <algorithm>
#include <cstdlib>
#include <chrono>

namespace clio::cte::cache_base {

namespace core = clio::cte::core;
namespace Method = core::Method;

namespace {

/** Wait between attempts to take a busy blob (us). */
constexpr double kLockPollUs = 10.0;
/** Copies evicted per no-space retry. */
constexpr clio::run::u32 kEvictBatch = 4;
/** Most copies dropped to make room for one put ahead of time. */
constexpr clio::run::u32 kMakeRoomMax = 64;
/** Resends of an op forwarded to a blob's owner that died in flight, and
 *  the wait between them while the cluster marks it dead (us). */
constexpr int kOwnerRetries = 10;
constexpr double kOwnerRetryUs = 1000000.0;
/** How long "volatile is full" is trusted before a put probes it again
 *  (ns): every path that frees space clears it at once, this bounds the
 *  cost of one that does not. */
constexpr clio::run::u64 kFullProbeNs = 100ull * 1000 * 1000;
/** Dirty ranges a flush keeps in flight to persistent, and the I/O buffer
 *  bytes they may hold together. */
constexpr size_t kFlushInflightOps = 64;
/** Bytes of one batch of flushed ranges landing on one owner. */
constexpr clio::run::u64 kLandBatchBytes = 8ull << 20;
/** Routing name of a MultiPutBlob that lands another node's flushed range
 *  on the blob's owner (write-back): the owner writes it through. */
constexpr const char kLandBlob[] = "\x01" "cache_base.land";
constexpr clio::run::u64 kFlushInflightBytes = 256ull << 20;
/** Least time between two fill-level queries of the volatile core (ns). */
constexpr clio::run::u64 kFillCheckNs = 10ull * 1000 * 1000;
/** Wait between checks for a shared round to complete (us). */
constexpr double kRoundPollUs = 50.0;
/** Read-miss counts kept for on_hit_count admission (then reset). */
constexpr size_t kMaxMissCounts = 1u << 20;
/** Wait between checks for buffered puts to finish during a drain (us). */
constexpr double kDrainPollUs = 10.0;
/** Flush rounds a write-back Destroy runs before it gives up on dirty data. */
constexpr clio::run::u32 kUnplugFlushRounds = 8;

clio::run::u64 NowNs() {
  return static_cast<clio::run::u64>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

/** Appends, replaces and conditional puts depend on the blob's current
 *  state, which only persistent decides. */
bool IsSpecial(clio::run::u32 flags, const Context &ctx) {
  return (flags & (core::kCtePutReplace | core::kCtePutAppend)) != 0 ||
         (ctx.op_flags_ & (Context::kPutIfAbsent | Context::kPutIfVersion)) !=
             0;
}

/**
 * Puts a write-back cache applies on the blob's owner instead of buffering
 * them where they are made: special puts (persistent decides) and blobs
 * holding a tag's own metadata (clio-fs directories and inodes), which its
 * readers on every node rely on being current, unlike file data.
 * @param flags the put's flags
 * @param ctx the put's context
 * @return true when the owner applies the put
 */
bool AtOwner(clio::run::u32 flags, const Context &ctx) {
  return IsSpecial(flags, ctx) || (ctx.op_flags_ & Context::kMetaBlob) != 0;
}

}  // namespace

CacheBase::~CacheBase() {
  if (stop_hook_id_ != 0) {
    auto *manager = CLIO_RUNTIME_MANAGER;
    if (manager != nullptr) manager->RemoveStopHook(stop_hook_id_);
  }
}

std::string CacheBase::Key(const TagId &tag, const std::string &name) {
  return std::to_string(tag.major_) + "." + std::to_string(tag.minor_) + "." +
         name;
}

void CacheBase::InitCache(const PolicyCacheParams &params,
                          const clio::run::PoolId &self_pool) {
  params_ = params;
  SetTargets(params.volatile_pool_, params.persistent_pool_);
  vol_client_.Init(params.volatile_pool_);
  pers_client_.Init(CorePoolId());
  self_client_.Init(self_pool);
  flush_client_.Init(params.flush_pool_.IsNull() ? self_pool
                                                 : params.flush_pool_);
  if (IsWriteBack() && HasPersistent()) {
    if (params.flush_period_ms_ > 0) {
      clio::cte::core::Client self(self_pool);
      self.AsyncFlushData(clio::run::PoolQuery::Local(), 0,
                          params.flush_period_ms_ * 1000.0);
    }
    RegisterStopFlush(self_pool);
  }
  HLOG(kInfo, "policy cache {}: {} over volatile {}.{} ({})", self_pool,
       IsWriteBack() ? "write-back" : "write-through",
       params.volatile_pool_.major_, params.volatile_pool_.minor_,
       HasPersistent() ? "persistent chain" : "no persistent chain");
}

void CacheBase::RegisterStopFlush(const clio::run::PoolId &self_pool) {
  auto *manager = CLIO_RUNTIME_MANAGER;
  if (manager == nullptr) return;
  stop_hook_id_ = manager->AddStopHook([self_pool]() {
    // Dirty copies are the only bytes a graceful stop could lose.
    auto *mgr = CLIO_RUNTIME_MANAGER;
    const float max_s =
        static_cast<float>(mgr->GetStopGracePeriodMs()) / 1000.0f;
    clio::cte::core::Client client(self_pool);
    auto flush = client.AsyncFlushData(clio::run::PoolQuery::Local(), 0);
    if (!flush.Wait(max_s)) {
      HLOG(kWarning, "write-back cache stop: dirty copies not flushed within "
           "the {} s grace period; unflushed data is lost", max_s);
    }
  });
}

clio::run::PoolQuery CacheBase::ScheduleTask(
    const clio::run::shared_ptr<clio::run::Task> &task) {
  switch (task->method_) {
    case Method::kFlushData:
    case kDropTagCopies:
      return task->pool_query_;  // per-container work, stays where sent
    case Method::kPutBlob:
    case Method::kMultiPutBlob:
    case Method::kDelBlob:
    case Method::kTruncateBlob:
      // Write-back buffers a write where it is made (the flush lands it on
      // the owner later). A delete or truncate runs here first so that no
      // later flush of this node's buffer undoes it. Reads go straight to
      // the owner: unflushed writes are visible only after a flush (fsync).
      if (IsWriteBack() && HasPersistent()) return clio::run::PoolQuery::Local();
      return CoreInterposer::ScheduleTask(task);
    default:
      return CoreInterposer::ScheduleTask(task);  // blob verbs -> owner
  }
}

void CacheBase::RemoveStopFlush() {
  if (stop_hook_id_ == 0) return;
  auto *manager = CLIO_RUNTIME_MANAGER;
  if (manager != nullptr) manager->RemoveStopHook(stop_hook_id_);
  stop_hook_id_ = 0;
}

clio::run::TaskResume CacheBase::Destroy(
    clio::run::shared_ptr<clio::run::admin::DestroyTask> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    namespace admin = clio::run::admin;
    const clio::run::u32 flags = task->destruction_flags_;
    const bool quiesce_only = (flags & admin::kDestroyPoolQuiesceOnly) != 0;
    const bool no_flush = (flags & admin::kDestroyPoolNoFlush) != 0;
    const bool delete_data = (flags & admin::kDestroyPoolDeleteData) != 0;
    clio::run::u32 rc = 0;
    FlushStats stats;
    std::string error;
    if (IsWriteBack() && HasPersistent()) {
      CLIO_CO_AWAIT(UnplugFlush(no_flush, &rc, &stats, &error));
    }
    task->blobs_flushed_ = stats.blobs_;
    task->bytes_flushed_ = stats.bytes_;
    if (rc != 0) {
      // Dirty data remains: the pool keeps running (UnplugFlush cleared the
      // drain) and the admin aborts the destroy.
      task->error_message_ = clio::run::priv::string(CTP_MALLOC, error);
      task->SetReturnCode(rc);
      CLIO_CO_RETURN;
    }
    if (!quiesce_only) {
      // The pool is going away: no more flushes from the stop hook.
      RemoveStopFlush();
      if (delete_data) CLIO_CO_AWAIT(DropEveryCopy());
    }
    task->SetReturnCode(0);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume CacheBase::BeginDrain() {
  CLIO_TASK_BODY_BEGIN
  draining_.store(true, std::memory_order_seq_cst);
  while (buffering_.load(std::memory_order_seq_cst) != 0) {
    CLIO_CO_AWAIT(clio::run::yield(kDrainPollUs));
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume CacheBase::UnplugFlush(bool no_flush, clio::run::u32 *rc,
                                             FlushStats *stats,
                                             std::string *error) {
  CLIO_TASK_BODY_BEGIN
  *rc = 0;
  CLIO_CO_AWAIT(BeginDrain());
  if (no_flush) {
    // Forced: the buffered bytes are discarded, and the count says how many.
    clio::run::u64 dirty_blobs = 0;
    clio::run::u64 dirty_bytes = 0;
    DirtyTotals(&dirty_blobs, &dirty_bytes);
    discarded_blobs_.fetch_add(dirty_blobs, std::memory_order_relaxed);
    CLIO_CO_RETURN;
  }
  clio::run::u64 left_blobs = 0;
  clio::run::u64 left_bytes = 0;
  for (clio::run::u32 round = 0; round < kUnplugFlushRounds; ++round) {
    clio::run::u32 frc = 0;
    FlushStats round_stats;
    CLIO_CO_AWAIT(FlushEverythingShared(&frc, &round_stats));
    stats->blobs_ += round_stats.blobs_;
    stats->bytes_ += round_stats.bytes_;
    DirtyTotals(&left_blobs, &left_bytes);
    if (frc != 0) {
      *rc = frc;
      break;
    }
    // Nothing left, or a round that made no progress: stop.
    if (left_blobs == 0 || round_stats.blobs_ == 0) break;
  }
  if (left_blobs != 0 && *rc == 0) *rc = core::kSyncIoRc;
  unplug_blobs_flushed_.fetch_add(stats->blobs_, std::memory_order_relaxed);
  unplug_bytes_flushed_.fetch_add(stats->bytes_, std::memory_order_relaxed);
  if (stats->blobs_ != 0) {
    unplug_flushes_.fetch_add(1, std::memory_order_relaxed);
  }
  if (*rc != 0) {
    draining_.store(false, std::memory_order_seq_cst);  // keep running
    *error = "write-back flush before destroy failed (rc " +
             std::to_string(*rc) + "): " + std::to_string(left_blobs) +
             " dirty blob(s), " + std::to_string(left_bytes) +
             " dirty byte(s) not durable";
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume CacheBase::FlushEverythingShared(clio::run::u32 *rc,
                                                       FlushStats *stats) {
  CLIO_TASK_BODY_BEGIN
  {
    const std::string id = "flushall";
    bool leader = false;
    *rc = 0;
    CLIO_CO_AWAIT(JoinRound(id, &leader, rc, nullptr, stats));
    if (leader) {
      const clio::run::u64 gen = dirty_gen_.load(std::memory_order_relaxed);
      CLIO_CO_AWAIT(FlushAll(TagId::GetNull(), true, rc, stats));
      if (*rc == 0) flushed_gen_.store(gen, std::memory_order_relaxed);
      LeaveRound(id, *rc, nullptr, stats);
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume CacheBase::DropEveryCopy() {
  CLIO_TASK_BODY_BEGIN
  {
    std::vector<std::pair<TagId, std::string>> copies;
    {
      std::lock_guard<std::mutex> g(mu_);
      for (const auto &e : entries_) {
        copies.emplace_back(e.second.tag_, e.second.name_);
      }
    }
    for (const auto &c : copies) {
      const std::string key = Key(c.first, c.second);
      CLIO_CO_AWAIT(Lock(key));
      CLIO_CO_AWAIT(DropVolatile(c.first, c.second));
      Unlock(key);
    }
    // Whatever a concurrent read filled after the snapshot is dropped too:
    // the pool is being removed, so the bookkeeping goes with it.
    std::lock_guard<std::mutex> g(mu_);
    entries_.clear();
    dirty_tags_.clear();
    victims_.clear();
    NoteSpaceFreed();
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

void CacheBase::DirtyTotals(clio::run::u64 *blobs, clio::run::u64 *bytes) {
  std::lock_guard<std::mutex> g(mu_);
  *blobs = 0;
  *bytes = 0;
  for (const auto &e : entries_) {
    if (e.second.dirty_.empty()) continue;
    ++*blobs;
    for (const Extent &x : e.second.dirty_) *bytes += x.second - x.first;
  }
}

void CacheBase::TallyFlush(const std::vector<FlushKey> &keys,
                           FlushStats *stats) {
  for (const FlushKey &k : keys) {
    if (!k.flushed_) continue;
    if (stats != nullptr) {
      stats->blobs_ += 1;
      stats->bytes_ += k.bytes_;
    }
    blobs_flushed_.fetch_add(1, std::memory_order_relaxed);
    bytes_flushed_.fetch_add(k.bytes_, std::memory_order_relaxed);
  }
}

std::string CacheBase::PackStats() {
  clio::run::u64 dirty_blobs = 0;
  clio::run::u64 dirty_bytes = 0;
  DirtyTotals(&dirty_blobs, &dirty_bytes);
  clio::run::u64 cached = 0;
  {
    std::lock_guard<std::mutex> g(mu_);
    cached = entries_.size();
  }
  msgpack::sbuffer sbuf;
  msgpack::packer<msgpack::sbuffer> pk(sbuf);
  pk.pack_map(13);
  pk.pack("policy");
  pk.pack(std::string(IsWriteBack() ? "write-back" : "write-through"));
  pk.pack("draining");
  pk.pack(draining_.load(std::memory_order_relaxed));
  pk.pack("dirty_blobs");
  pk.pack(dirty_blobs);
  pk.pack("dirty_bytes");
  pk.pack(dirty_bytes);
  pk.pack("cached_blobs");
  pk.pack(cached);
  pk.pack("flush_rounds");
  pk.pack(flush_rounds_.load(std::memory_order_relaxed));
  pk.pack("blobs_flushed");
  pk.pack(blobs_flushed_.load(std::memory_order_relaxed));
  pk.pack("bytes_flushed");
  pk.pack(bytes_flushed_.load(std::memory_order_relaxed));
  pk.pack("flush_failures");
  pk.pack(flush_failures_.load(std::memory_order_relaxed));
  pk.pack("unplug_flushes");
  pk.pack(unplug_flushes_.load(std::memory_order_relaxed));
  pk.pack("unplug_blobs_flushed");
  pk.pack(unplug_blobs_flushed_.load(std::memory_order_relaxed));
  pk.pack("unplug_bytes_flushed");
  pk.pack(unplug_bytes_flushed_.load(std::memory_order_relaxed));
  pk.pack("discarded_blobs");
  pk.pack(discarded_blobs_.load(std::memory_order_relaxed));
  return std::string(sbuf.data(), sbuf.size());
}

clio::run::TaskResume CacheBase::Monitor(
    clio::run::shared_ptr<clio::run::admin::MonitorTask> &task) {
  CLIO_TASK_BODY_BEGIN
  if (task->query_ == "stats") {
    task->results_[container_id_] = PackStats();
    task->SetReturnCode(0);
    CLIO_CO_RETURN;
  }
  if (task->query_ == "dirty_count") {
    clio::run::u64 dirty_blobs = 0;
    clio::run::u64 dirty_bytes = 0;
    DirtyTotals(&dirty_blobs, &dirty_bytes);
    msgpack::sbuffer sbuf;
    msgpack::packer<msgpack::sbuffer> pk(sbuf);
    pk.pack(dirty_blobs);
    task->results_[container_id_] = std::string(sbuf.data(), sbuf.size());
    task->SetReturnCode(0);
    CLIO_CO_RETURN;
  }
  CLIO_CO_AWAIT(ForwardToCore(Method::kMonitor,
                              task.template Cast<clio::run::Task>()));
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

// ===========================================================================
// Owner-local state
// ===========================================================================

clio::run::TaskResume CacheBase::Lock(std::string key) {
  CLIO_TASK_BODY_BEGIN
  while (true) {
    {
      std::lock_guard<std::mutex> g(mu_);
      if (busy_.insert(key).second) break;
    }
    CLIO_CO_AWAIT(clio::run::yield(kLockPollUs));
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

void CacheBase::Unlock(const std::string &key) {
  std::lock_guard<std::mutex> g(mu_);
  busy_.erase(key);
}

bool CacheBase::Cached(const std::string &key, clio::run::u64 *size) {
  std::lock_guard<std::mutex> g(mu_);
  auto it = entries_.find(key);
  if (it == entries_.end()) return false;
  if (size != nullptr) *size = it->second.size_;
  return true;
}

void CacheBase::Touch(const std::string &key) {
  std::lock_guard<std::mutex> g(mu_);
  auto it = entries_.find(key);
  if (it != entries_.end()) it->second.last_ns_ = NowNs();
}

bool CacheBase::ShouldFill(const std::string &key) {
  switch (params_.Fill()) {
    case FillMode::kOnReadMiss:
      return true;
    case FillMode::kNone:
      return false;
    case FillMode::kOnHitCount: {
      std::lock_guard<std::mutex> g(mu_);
      // Bounded: a long one-pass read leaves one count per blob behind.
      if (miss_counts_.size() > kMaxMissCounts) miss_counts_.clear();
      clio::run::u32 &n = miss_counts_[key];
      if (++n < params_.fill_hits_) return false;
      miss_counts_.erase(key);
      return true;
    }
  }
  return false;
}

void CacheBase::NoteWrite(const std::string &key, const TagId &tag,
                          const std::string &name, clio::run::u64 off,
                          clio::run::u64 end, bool dirty) {
  std::lock_guard<std::mutex> g(mu_);
  Entry &e = entries_[key];
  e.tag_ = tag;
  e.name_ = name;
  e.size_ = std::max(e.size_, end);
  if (dirty) AddExtent(e.dirty_, off, end);
  if (dirty) IndexDirty(key, e);
  e.last_ns_ = NowNs();
  if (e.dirty_.empty()) NoteClean();
  if (dirty) {
    dirty_gen_.fetch_add(1, std::memory_order_relaxed);
    dirty_since_flush_.fetch_add(end - off, std::memory_order_relaxed);
  }
}

clio::run::TaskResume CacheBase::CheckFill() {
  CLIO_TASK_BODY_BEGIN
  if (IsWriteBack() && HasPersistent() && params_.flush_dirty_mb_ != 0 &&
      dirty_since_flush_.load(std::memory_order_relaxed) >=
          (static_cast<clio::run::u64>(params_.flush_dirty_mb_) << 20) &&
      !fill_flush_pending_.exchange(true)) {
    // Continuous: the flush covers everything dirty when it starts; what is
    // written meanwhile counts toward the next one, started by the first
    // write after this one ends (FlushData clears the pending mark).
    dirty_since_flush_.store(0, std::memory_order_relaxed);
    self_client_.AsyncFlushData(clio::run::PoolQuery::Local(), 0);
  }
  if (IsWriteBack() && HasPersistent() && params_.flush_fill_pct_ != 0 &&
      !fill_flush_pending_.load(std::memory_order_relaxed) &&
      dirty_gen_.load(std::memory_order_relaxed) !=
          flushed_gen_.load(std::memory_order_relaxed)) {
    const clio::run::u64 now = NowNs();
    clio::run::u64 last = fill_check_ns_.load(std::memory_order_relaxed);
    if (now - last >= kFillCheckNs &&
        fill_check_ns_.compare_exchange_strong(last, now)) {
      auto cap = vol_client_.AsyncGetCapacity(clio::run::PoolQuery::Local());
      CLIO_CO_AWAIT(cap);
      const clio::run::u64 total = cap->total_capacity_;
      if (cap->GetReturnCode() == 0 && total != 0) {
        const clio::run::u64 used = total - cap->remaining_capacity_;
        // total / 100 first: total * pct could overflow.
        if (used >= total / 100 * params_.flush_fill_pct_ &&
            !fill_flush_pending_.exchange(true)) {
          // One-shot, not awaited: this put returns; FlushData clears the
          // pending mark when it is done.
          self_client_.AsyncFlushData(clio::run::PoolQuery::Local(), 0);
        }
      }
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

void CacheBase::NoteNoSpace(clio::run::u64 size) {
  const clio::run::u64 cur = full_size_.load(std::memory_order_relaxed);
  if (cur == 0 || size < cur) {
    full_size_.store(size == 0 ? 1 : size, std::memory_order_relaxed);
  }
  full_ns_.store(NowNs(), std::memory_order_relaxed);
}

bool CacheBase::KnownFull(clio::run::u64 size) {
  const clio::run::u64 full = full_size_.load(std::memory_order_relaxed);
  if (full == 0 || size < full) return false;
  if (NowNs() - full_ns_.load(std::memory_order_relaxed) > kFullProbeNs) {
    // Stale: let this put probe volatile (and its eviction look again).
    full_size_.store(0, std::memory_order_relaxed);
    no_clean_.store(false, std::memory_order_relaxed);
    return false;
  }
  return true;
}

clio::run::TaskResume CacheBase::MakeRoom(std::string key,
                                          clio::run::u64 size, bool *bypass) {
  CLIO_TASK_BODY_BEGIN
  *bypass = false;
  if (EvictCleanOnly() && KnownFull(size) && !Cached(key)) {
    if (!no_clean_.load(std::memory_order_relaxed)) {
      clio::run::u32 done = 0;
      // As many clean copies as hold this put's bytes, so the put below
      // is not refused (a refusal costs the core a scan of its blob map).
      CLIO_CO_AWAIT(Evict(key, kMakeRoomMax, true, &done, size));
      if (done == 0) no_clean_.store(true, std::memory_order_relaxed);
    }
    *bypass = no_clean_.load(std::memory_order_relaxed);
    // Still full for the next put; only a failed probe would say so again.
    if (*bypass) full_ns_.store(NowNs(), std::memory_order_relaxed);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

void CacheBase::IndexDirty(const std::string &key, const Entry &e) {
  const TagKey tk(e.tag_.major_, e.tag_.minor_);
  if (!e.dirty_.empty()) {
    dirty_tags_[tk].insert(key);
    return;
  }
  auto it = dirty_tags_.find(tk);
  if (it == dirty_tags_.end()) return;
  it->second.erase(key);
  if (it->second.empty()) dirty_tags_.erase(it);
}

void CacheBase::UnindexDirty(const std::string &key) {
  auto e = entries_.find(key);
  if (e == entries_.end()) return;
  auto it = dirty_tags_.find(TagKey(e->second.tag_.major_,
                                    e->second.tag_.minor_));
  if (it == dirty_tags_.end()) return;
  it->second.erase(key);
  if (it->second.empty()) dirty_tags_.erase(it);
}

bool CacheBase::TagDirty(const TagId &tag) {
  std::lock_guard<std::mutex> g(mu_);
  return dirty_tags_.count(TagKey(tag.major_, tag.minor_)) != 0;
}

std::vector<std::pair<std::string, clio::run::u64>> CacheBase::DirtyKeysOf(
    const TagId *tag) {
  std::vector<std::pair<std::string, clio::run::u64>> keys;
  {
    std::lock_guard<std::mutex> g(mu_);
    // From the dirty index: only the copies that have something to flush.
    auto add = [this, &keys](const std::set<std::string> &set) {
      for (const std::string &key : set) {
        auto e = entries_.find(key);
        if (e == entries_.end()) continue;
        clio::run::u64 bytes = 0;
        for (const Extent &x : e->second.dirty_) bytes += x.second - x.first;
        keys.emplace_back(key, bytes);
      }
    };
    if (tag != nullptr) {
      auto it = dirty_tags_.find(TagKey(tag->major_, tag->minor_));
      if (it != dirty_tags_.end()) add(it->second);
    } else {
      for (const auto &t : dirty_tags_) add(t.second);
    }
  }
  std::sort(keys.begin(), keys.end());  // the global lock order
  return keys;
}

std::vector<std::string> CacheBase::KeysOf(const TagId *tag) {
  std::vector<std::string> keys;
  std::lock_guard<std::mutex> g(mu_);
  for (const auto &e : entries_) {
    if (tag == nullptr || e.second.tag_ == *tag) keys.push_back(e.first);
  }
  return keys;
}

// ===========================================================================
// Volatile copy maintenance (callers hold the blob's lock)
// ===========================================================================

clio::run::Future<core::MultiPutBlobTask> CacheBase::SendLand(
    const TagId &tag, const std::string &name, const Extent &x,
    const ctp::ipc::FullPtr<char> &buf, clio::run::u32 owner) {
  core::MultiPutDesc d;
  d.tag_id_ = tag;
  d.blob_name_ = name;
  d.offset_ = x.first;
  d.size_ = x.second - x.first;
  d.payload_off_ = 0;
  auto *ipc = CLIO_IPC;
  auto t = ipc->NewTask<core::MultiPutBlobTask>(
      clio::run::CreateTaskId(), flush_client_.pool_id_,
      clio::run::PoolQuery::DirectId(owner), tag, std::string(kLandBlob),
      ctp::ipc::ShmPtr<>(buf.shm_), d.size_, core::EncodeMultiPutDescs({d}),
      Context());
  return ipc->Send(t);
}

clio::run::u32 CacheBase::LandRc(
    clio::run::Future<core::MultiPutBlobTask> &f) {
  if (f.get() == nullptr) return 1;
  const clio::run::u32 rc = f->GetReturnCode();
  if (rc != 0) return rc;
  return f->first_rc_ != 0 ? static_cast<clio::run::u32>(f->first_rc_) : 0;
}

bool CacheBase::CopyClean(const std::string &key) {
  std::lock_guard<std::mutex> g(mu_);
  auto it = entries_.find(key);
  return it != entries_.end() && it->second.dirty_.empty();
}

clio::run::TaskResume CacheBase::Flush(std::string key, clio::run::u32 *rc) {
  CLIO_TASK_BODY_BEGIN
  *rc = 0;
  {
    Entry e;
    {
      std::lock_guard<std::mutex> g(mu_);
      auto it = entries_.find(key);
      if (it == entries_.end() || it->second.dirty_.empty()) CLIO_CO_RETURN;
      e = it->second;
    }
    const clio::run::u32 owner =
        IsWriteBack() ? OwnerOf(e.tag_, e.name_) : container_id_;
    auto *ipc = CLIO_IPC;
    for (const Extent &x : e.dirty_) {
      const clio::run::u64 len = x.second - x.first;
      ctp::ipc::FullPtr<char> buf = ipc->AllocateBuffer(len);
      if (buf.IsNull()) {
        *rc = 12;
        break;
      }
      auto get = vol_client_.AsyncGetBlob(e.tag_, e.name_.c_str(), x.first,
                                          len, 0, ctp::ipc::ShmPtr<>(buf.shm_),
                                          clio::run::PoolQuery::Local());
      CLIO_CO_AWAIT(get);
      *rc = get->GetReturnCode();
      if (*rc == 0 && owner != container_id_) {
        auto landed = SendLand(e.tag_, e.name_, x, buf, owner);
        CLIO_CO_AWAIT(landed);
        *rc = LandRc(landed);
      } else if (*rc == 0) {
        auto put = pers_client_.AsyncPutBlob(
            e.tag_, e.name_, x.first, len, ctp::ipc::ShmPtr<>(buf.shm_), -1.0f,
            Context(), 0, clio::run::PoolQuery::Local());
        CLIO_CO_AWAIT(put);
        *rc = put->GetReturnCode();
      }
      ipc->FreeBuffer(buf);
      if (*rc != 0) break;
    }
    if (*rc == 0) {
      std::lock_guard<std::mutex> g(mu_);
      auto it = entries_.find(key);
      if (it != entries_.end()) {
        it->second.dirty_.clear();
        IndexDirty(key, it->second);
      }
      NoteClean();
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume CacheBase::Drop(std::string key, bool flush,
                                      clio::run::u32 *rc) {
  CLIO_TASK_BODY_BEGIN
  *rc = 0;
  {
    Entry e;
    {
      std::lock_guard<std::mutex> g(mu_);
      auto it = entries_.find(key);
      if (it == entries_.end()) CLIO_CO_RETURN;
      e = it->second;
    }
    if (flush && IsWriteBack()) {
      CLIO_CO_AWAIT(Flush(key, rc));
      if (*rc != 0) CLIO_CO_RETURN;  // keep the copy: it holds the only bytes
    }
    auto del = vol_client_.AsyncDelBlob(e.tag_, e.name_,
                                        clio::run::PoolQuery::Local());
    CLIO_CO_AWAIT(del);
    std::lock_guard<std::mutex> g(mu_);
    UnindexDirty(key);
    entries_.erase(key);
    NoteSpaceFreed();
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume CacheBase::DropVolatile(TagId tag, std::string name) {
  CLIO_TASK_BODY_BEGIN
  {
    // A refused or partial volatile put may have left a blob behind that
    // this cache never recorded; "not found" is an expected answer.
    auto del = vol_client_.AsyncDelBlob(tag, name,
                                        clio::run::PoolQuery::Local());
    CLIO_CO_AWAIT(del);
    std::lock_guard<std::mutex> g(mu_);
    UnindexDirty(Key(tag, name));
    entries_.erase(Key(tag, name));
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

void CacheBase::RefillVictims(const std::string &keep, bool clean_only) {
  std::vector<std::pair<clio::run::u64, std::string>> lru;
  std::lock_guard<std::mutex> g(mu_);
  for (const auto &e : entries_) {
    if (clean_only && !e.second.dirty_.empty()) continue;
    if (e.first != keep && busy_.count(e.first) == 0) {
      lru.emplace_back(e.second.last_ns_, e.first);
    }
  }
  const size_t n = std::min(lru.size(), kVictimQueue);
  std::partial_sort(lru.begin(), lru.begin() + n, lru.end());
  victims_.clear();
  for (size_t i = 0; i < n; ++i) victims_.push_back(std::move(lru[i].second));
}

bool CacheBase::TakeVictim(const std::string &keep, bool clean_only,
                           std::string *key, clio::run::u64 *size) {
  std::lock_guard<std::mutex> g(mu_);
  while (!victims_.empty()) {
    std::string k = std::move(victims_.front());
    victims_.pop_front();
    auto it = entries_.find(k);
    // The list is a snapshot: the copy may be gone, rewritten or in use.
    if (it == entries_.end() || k == keep) continue;
    if (clean_only && !it->second.dirty_.empty()) continue;
    if (!busy_.insert(k).second) continue;
    *size = it->second.size_;
    *key = std::move(k);
    return true;
  }
  return false;
}

clio::run::TaskResume CacheBase::Evict(std::string keep, clio::run::u32 n,
                                       bool clean_only,
                                       clio::run::u32 *out_done,
                                       clio::run::u64 need_bytes) {
  CLIO_TASK_BODY_BEGIN
  *out_done = 0;
  {
    // Dropping copies to admit a write does not make volatile less full for
    // the next write: keep what is known about it.
    const clio::run::u64 full = full_size_.load(std::memory_order_relaxed);
    clio::run::u32 done = 0;
    clio::run::u64 freed = 0;
    bool refilled = false;
    while (done < n && (need_bytes == 0 || freed < need_bytes || done == 0)) {
      std::string key;
      clio::run::u64 size = 0;
      if (!TakeVictim(keep, clean_only, &key, &size)) {
        if (refilled) break;  // a fresh list had nothing left to take
        RefillVictims(keep, clean_only);
        refilled = true;
        continue;
      }
      if (IsWriteBack() && !CopyClean(key) && KeyOwner(key) != container_id_) {
        // Flushing it would wait on its owner with the writer's lock held:
        // the flusher lands it instead.
        Unlock(key);
        continue;
      }
      clio::run::u32 rc = 0;
      CLIO_CO_AWAIT(Drop(key, true, &rc));
      Unlock(key);
      if (rc == 0) {
        ++done;
        freed += size;
      }
    }
    if (clean_only && full != 0 && done != 0) {
      full_size_.store(full, std::memory_order_relaxed);
      full_ns_.store(NowNs(), std::memory_order_relaxed);
    }
    *out_done = done;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume CacheBase::FillFromPersistent(TagId tag,
                                                    std::string name,
                                                    clio::run::u64 size,
                                                    clio::run::u32 *rc) {
  CLIO_TASK_BODY_BEGIN
  *rc = 0;
  {
    auto *ipc = CLIO_IPC;
    ctp::ipc::FullPtr<char> buf = ipc->AllocateBuffer(size == 0 ? 1 : size);
    if (buf.IsNull()) {
      *rc = 12;
      CLIO_CO_RETURN;
    }
    auto get = pers_client_.AsyncGetBlob(tag, name.c_str(), 0, size, 0,
                                         ctp::ipc::ShmPtr<>(buf.shm_),
                                         clio::run::PoolQuery::Local());
    CLIO_CO_AWAIT(get);
    *rc = get->GetReturnCode();
    if (*rc == 0) {
      CLIO_CO_AWAIT(PutVolatileRaw(tag, name, 0, size, buf.ptr_, rc));
    }
    ipc->FreeBuffer(buf);
    if (*rc == 0) NoteWrite(Key(tag, name), tag, name, 0, size, false);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume CacheBase::PutVolatileRaw(TagId tag, std::string name,
                                                clio::run::u64 off,
                                                clio::run::u64 size,
                                                const char *src,
                                                clio::run::u32 *rc) {
  CLIO_TASK_BODY_BEGIN
  for (int attempt = 0; attempt < 2; ++attempt) {
    auto put = vol_client_.AsyncPutBlob(tag, name, off, size, src, -1.0f,
                                        Context(), 0,
                                        clio::run::PoolQuery::Local());
    CLIO_CO_AWAIT(put);
    *rc = put->GetReturnCode();
    if (!IsNoSpaceRc(*rc) || attempt == 1) break;
    clio::run::u32 done = 0;
    CLIO_CO_AWAIT(Evict(Key(tag, name), kEvictBatch, EvictCleanOnly(), &done));
    if (done == 0) {
      // Nothing was dropped, so the same put would only fail again.
      if (EvictCleanOnly()) no_clean_.store(true, std::memory_order_relaxed);
      break;
    }
  }
  if (IsNoSpaceRc(*rc)) NoteNoSpace(size);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume CacheBase::PutVolatile(
    clio::run::shared_ptr<core::PutBlobTask> task, clio::run::u32 *rc) {
  CLIO_TASK_BODY_BEGIN
  {
    std::vector<core::BlobRegion> regions;
    core::ForEachBlobRegion(*task, [&regions](const core::BlobRegion &r) {
      regions.push_back(r);
      return true;
    });
    const std::string name = task->blob_name_.str();
    *rc = 0;
    for (size_t i = 0; i < regions.size() && *rc == 0; ++i) {
      for (int attempt = 0; attempt < 2; ++attempt) {
        auto put = vol_client_.AsyncPutBlob(
            task->tag_id_, name, regions[i].blob_off_, regions[i].size_,
            regions[i].data_, -1.0f, Context(), 0,
            clio::run::PoolQuery::Local());
        CLIO_CO_AWAIT(put);
        *rc = put->GetReturnCode();
        if (!IsNoSpaceRc(*rc) || attempt == 1) break;
        clio::run::u32 done = 0;
        CLIO_CO_AWAIT(Evict(Key(task->tag_id_, name), kEvictBatch,
                            EvictCleanOnly(), &done));
        if (done == 0) {
          // Nothing was dropped, so the same put would only fail again.
          if (EvictCleanOnly()) {
            no_clean_.store(true, std::memory_order_relaxed);
          }
          break;
        }
      }
      if (IsNoSpaceRc(*rc)) NoteNoSpace(regions[i].size_);
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume CacheBase::PrepareAbsorb(TagId tag, std::string name,
                                               clio::run::u64 off,
                                               clio::run::u64 end,
                                               clio::run::u32 *rc) {
  CLIO_TASK_BODY_BEGIN
  *rc = 0;
  if (!Cached(Key(tag, name))) {
    // A partial write to a blob that only persistent holds: make the copy
    // whole first, so the volatile copy is always the entire blob.
    auto sz = pers_client_.AsyncGetBlobSize(tag, name,
                                            clio::run::PoolQuery::Local());
    CLIO_CO_AWAIT(sz);
    const clio::run::u64 psize = sz->GetReturnCode() == 0 ? sz->size_ : 0;
    if (psize > 0 && !(off == 0 && end >= psize)) {
      CLIO_CO_AWAIT(FillFromPersistent(tag, name, psize, rc));
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume CacheBase::FlushOpenKey(std::vector<FlushKey> *keys,
                                              size_t i,
                                              std::deque<FlushOp> *queued) {
  CLIO_TASK_BODY_BEGIN
  {
    FlushKey &k = (*keys)[i];
    CLIO_CO_AWAIT(Lock(k.key_));
    const size_t first = queued->size();
    {
      // Read under the lock: a write since DirtyKeysOf may have added ranges.
      std::lock_guard<std::mutex> g(mu_);
      auto it = entries_.find(k.key_);
      if (it != entries_.end()) {
        for (const Extent &x : it->second.dirty_) {
          FlushOp op;
          op.key_ = i;
          op.tag_ = it->second.tag_;
          op.name_ = it->second.name_;
          op.ext_ = x;
          queued->push_back(std::move(op));
          ++k.left_;
          k.bytes_ += x.second - x.first;
        }
      }
    }
    if (k.left_ == 0) {
      Unlock(k.key_);  // flushed (or dropped) meanwhile
    } else {
      for (size_t j = first; j < queued->size(); ++j) {
        (*queued)[j].owner_ = k.owner_;
      }
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

void CacheBase::FlushRangeDone(std::vector<FlushKey> *keys, size_t i,
                               clio::run::u32 rc) {
  FlushKey &k = (*keys)[i];
  if (rc != 0 && k.rc_ == 0) k.rc_ = rc;
  if (--k.left_ != 0) return;
  if (k.rc_ == 0) {
    {
      std::lock_guard<std::mutex> g(mu_);
      auto it = entries_.find(k.key_);
      if (it != entries_.end()) {
        it->second.dirty_.clear();
        IndexDirty(k.key_, it->second);
      }
      NoteClean();
    }
    k.flushed_ = true;  // its last range landed: durable
  }
  Unlock(k.key_);
}

clio::run::TaskResume CacheBase::FlushWindow(std::vector<FlushKey> *keys) {
  CLIO_TASK_BODY_BEGIN
  {
    // Blobs are locked in key order (the one global order) as the window
    // reaches them, and each is released as soon as its last range lands.
    auto *ipc = CLIO_IPC;
    std::deque<FlushOp> queued, getting, putting;
    size_t next_key = 0;
    clio::run::u64 held = 0;  // I/O buffer bytes in flight
    while (true) {
      // Fill the window: read ranges out of volatile.
      while (getting.size() + putting.size() < kFlushInflightOps &&
             held < kFlushInflightBytes) {
        if (queued.empty()) {
          if (next_key == keys->size()) break;
          CLIO_CO_AWAIT(FlushOpenKey(keys, next_key++, &queued));
          continue;
        }
        FlushOp op = std::move(queued.front());
        queued.pop_front();
        const clio::run::u64 len = op.ext_.second - op.ext_.first;
        op.buf_ = ipc->AllocateBuffer(len);
        if (op.buf_.IsNull()) {
          if (!getting.empty() || !putting.empty()) {
            queued.push_front(std::move(op));  // retry once buffers return
            break;
          }
          FlushRangeDone(keys, op.key_, 12);  // no memory with none held
          continue;
        }
        op.get_ = vol_client_.AsyncGetBlob(op.tag_, op.name_.c_str(),
                                           op.ext_.first, len, 0,
                                           ctp::ipc::ShmPtr<>(op.buf_.shm_),
                                           clio::run::PoolQuery::Local());
        held += len;
        getting.push_back(std::move(op));
      }
      if (!getting.empty()) {
        // A range read out of volatile goes on to persistent.
        FlushOp op = std::move(getting.front());
        getting.pop_front();
        CLIO_CO_AWAIT(op.get_);
        const clio::run::u64 len = op.ext_.second - op.ext_.first;
        const clio::run::u32 grc = op.get_->GetReturnCode();
        if (grc != 0) {
          ipc->FreeBuffer(op.buf_);
          held -= len;
          FlushRangeDone(keys, op.key_, grc);
          continue;
        }
        if ((*keys)[op.key_].foreign_) {
          op.land_ = SendLand(op.tag_, op.name_, op.ext_, op.buf_, op.owner_);
        } else {
          op.put_ = pers_client_.AsyncPutBlob(
              op.tag_, op.name_, op.ext_.first, len,
              ctp::ipc::ShmPtr<>(op.buf_.shm_), -1.0f, Context(), 0,
              clio::run::PoolQuery::Local());
        }
        putting.push_back(std::move(op));
        continue;
      }
      if (putting.empty()) break;
      // The oldest write to persistent lands; its slot takes the next range.
      FlushOp op = std::move(putting.front());
      putting.pop_front();
      clio::run::u32 prc = 0;
      if ((*keys)[op.key_].foreign_) {
        CLIO_CO_AWAIT(op.land_);
        prc = LandRc(op.land_);
      } else {
        CLIO_CO_AWAIT(op.put_);
        prc = op.put_->GetReturnCode();
      }
      ipc->FreeBuffer(op.buf_);
      held -= op.ext_.second - op.ext_.first;
      FlushRangeDone(keys, op.key_, prc);
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

void CacheBase::LandBatchDone(std::vector<FlushKey> *keys, LandBatch *b,
                              clio::run::u32 rc) {
  CLIO_IPC->FreeBuffer(b->buf_);
  for (size_t i : b->keys_) FlushRangeDone(keys, i, rc);
}

clio::run::TaskResume CacheBase::FlushAway(std::vector<FlushKey> *keys) {
  CLIO_TASK_BODY_BEGIN
  {
    // Ranges are copied out of volatile into one buffer per owner and sent
    // as one batch per kLandBatchBytes. A blob holds its lock (only blobs
    // other containers own are locked here) until every batch carrying it
    // landed; FlushKey::left_ counts those batches, plus one while its
    // ranges are still being placed.
    auto *ipc = CLIO_IPC;
    std::map<clio::run::u32, LandBatch> open;
    std::deque<LandBatch> flying;
    clio::run::u64 held = 0;
    for (size_t i = 0; i < keys->size(); ++i) {
      FlushKey &k = (*keys)[i];
      std::deque<FlushOp> ranges;
      CLIO_CO_AWAIT(FlushOpenKey(keys, i, &ranges));
      if (k.left_ == 0) continue;  // nothing dirty by now (unlocked)
      k.left_ = 1;                 // the guard, dropped once placed
      for (FlushOp &op : ranges) {
        const clio::run::u64 len = op.ext_.second - op.ext_.first;
        LandBatch &b = open[k.owner_];
        if (!b.buf_.IsNull() && b.used_ + len > b.cap_) {
          held += b.used_;
          LandSend(&b, k.owner_);
          flying.push_back(std::move(b));
          b = LandBatch();
        }
        while (held > kFlushInflightBytes && !flying.empty()) {
          CLIO_CO_AWAIT(flying.front().fut_);
          held -= flying.front().used_;
          LandBatchDone(keys, &flying.front(), LandRc(flying.front().fut_));
          flying.pop_front();
        }
        if (b.buf_.IsNull()) {
          b.cap_ = std::max(len, kLandBatchBytes);
          b.buf_ = ipc->AllocateBuffer(b.cap_);
          while (b.buf_.IsNull() && !flying.empty()) {
            CLIO_CO_AWAIT(flying.front().fut_);
            held -= flying.front().used_;
            LandBatchDone(keys, &flying.front(), LandRc(flying.front().fut_));
            flying.pop_front();
            b.buf_ = ipc->AllocateBuffer(b.cap_);
          }
          if (b.buf_.IsNull()) {
            if (k.rc_ == 0) k.rc_ = 12;  // no memory with nothing in flight
            continue;
          }
        }
        auto get = vol_client_.AsyncGetBlob(
            op.tag_, op.name_.c_str(), op.ext_.first, len, 0,
            ctp::ipc::ShmPtr<>(b.buf_.shm_) + b.used_,
            clio::run::PoolQuery::Local());
        CLIO_CO_AWAIT(get);
        if (get->GetReturnCode() != 0) {
          if (k.rc_ == 0) k.rc_ = get->GetReturnCode();
          continue;
        }
        core::MultiPutDesc d;
        d.tag_id_ = op.tag_;
        d.blob_name_ = op.name_;
        d.offset_ = op.ext_.first;
        d.size_ = len;
        d.payload_off_ = b.used_;
        b.descs_.push_back(std::move(d));
        b.used_ += len;
        if (b.keys_.empty() || b.keys_.back() != i) {
          b.keys_.push_back(i);
          ++k.left_;
        }
      }
      FlushRangeDone(keys, i, 0);  // drop the guard
    }
    for (auto &e : open) {
      if (e.second.buf_.IsNull()) continue;
      if (e.second.descs_.empty()) {
        ipc->FreeBuffer(e.second.buf_);
        continue;
      }
      LandSend(&e.second, e.first);
      flying.push_back(std::move(e.second));
    }
    while (!flying.empty()) {
      CLIO_CO_AWAIT(flying.front().fut_);
      LandBatchDone(keys, &flying.front(), LandRc(flying.front().fut_));
      flying.pop_front();
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

void CacheBase::LandSend(LandBatch *b, clio::run::u32 owner) {
  auto *ipc = CLIO_IPC;
  auto t = ipc->NewTask<core::MultiPutBlobTask>(
      clio::run::CreateTaskId(), flush_client_.pool_id_,
      clio::run::PoolQuery::DirectId(owner), b->descs_.front().tag_id_,
      std::string(kLandBlob), ctp::ipc::ShmPtr<>(b->buf_.shm_), b->used_,
      core::EncodeMultiPutDescs(b->descs_), Context());
  b->fut_ = ipc->Send(t);
}

clio::run::TaskResume CacheBase::FlushAll(TagId tag, bool all_tags,
                                          clio::run::u32 *rc,
                                          FlushStats *stats) {
  CLIO_TASK_BODY_BEGIN
  *rc = 0;
  if (IsWriteBack()) {
    flush_rounds_.fetch_add(1, std::memory_order_relaxed);
    // Two passes, so a lock on a blob this container owns is never held
    // while waiting on another container: first the blobs other containers
    // own (they land there, holding only their own locks here), then the
    // ones owned here (written to persistent here, no network). Owners lock
    // only blobs they own when a range lands, so no wait can go in a cycle.
    std::vector<FlushKey> away, here;
    for (auto &d : DirtyKeysOf(all_tags ? nullptr : &tag)) {
      FlushKey k{std::move(d.first)};
      k.owner_ = KeyOwner(k.key_);
      k.foreign_ = k.owner_ != container_id_;
      (k.foreign_ ? away : here).push_back(std::move(k));
    }
    CLIO_CO_AWAIT(FlushAway(&away));
    CLIO_CO_AWAIT(FlushWindow(&here));
    TallyFlush(away, stats);
    TallyFlush(here, stats);
    for (const std::vector<FlushKey> *ks : {&away, &here}) {
      for (const FlushKey &k : *ks) {
        if (k.rc_ != 0 && *rc == 0) *rc = k.rc_;
      }
    }
    if (*rc != 0) flush_failures_.fetch_add(1, std::memory_order_relaxed);
    // A landed blob another node owns is served by its owner from now on:
    // drop the buffer here (unless it was written again meanwhile).
    for (size_t i = 0; i < away.size(); ++i) {
      if (away[i].rc_ != 0) continue;
      CLIO_CO_AWAIT(Lock(away[i].key_));
      if (CopyClean(away[i].key_)) {
        clio::run::u32 drc = 0;
        CLIO_CO_AWAIT(Drop(away[i].key_, false, &drc));
      }
      Unlock(away[i].key_);
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::u32 CacheBase::KeyOwner(const std::string &key) {
  TagId tag;
  std::string name;
  {
    std::lock_guard<std::mutex> g(mu_);
    auto it = entries_.find(key);
    if (it == entries_.end()) return container_id_;
    tag = it->second.tag_;
    name = it->second.name_;
  }
  return OwnerOf(tag, name);
}

clio::run::TaskResume CacheBase::JoinRound(std::string id, bool *leader,
                                           clio::run::u32 *rc, SyncOut *out,
                                           FlushStats *flushed) {
  CLIO_TASK_BODY_BEGIN
  *leader = false;
  *rc = 0;
  {
    clio::run::u64 target = 0;
    {
      std::lock_guard<std::mutex> g(round_mu_);
      Round &r = rounds_[id];
      target = r.started_ + 1;  // the first round to start after this call
      ++r.users_;
    }
    while (true) {
      {
        std::lock_guard<std::mutex> g(round_mu_);
        Round &r = rounds_[id];
        if (r.done_ >= target) {
          *rc = r.rc_;
          if (out != nullptr) *out = r.out_;
          if (flushed != nullptr) *flushed = r.flushed_;
          if (--r.users_ == 0 && !r.running_) rounds_.erase(id);
          break;
        }
        if (!r.running_) {
          r.running_ = true;
          ++r.started_;
          *leader = true;
          break;
        }
      }
      CLIO_CO_AWAIT(clio::run::yield(kRoundPollUs));
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

void CacheBase::LeaveRound(const std::string &id, clio::run::u32 rc,
                           const SyncOut *out, const FlushStats *flushed) {
  std::lock_guard<std::mutex> g(round_mu_);
  auto it = rounds_.find(id);
  if (it == rounds_.end()) return;
  Round &r = it->second;
  r.done_ = r.started_;
  r.rc_ = rc;
  if (out != nullptr) r.out_ = *out;
  if (flushed != nullptr) r.flushed_ = *flushed;
  r.running_ = false;
  if (--r.users_ == 0) rounds_.erase(it);
}

clio::run::TaskResume CacheBase::FlushTagShared(TagId tag,
                                                clio::run::u32 *rc) {
  CLIO_TASK_BODY_BEGIN
  {
    // Nothing of this tag is dirty here (a sync asks every container, and
    // most hold nothing new): no round to join. A flush still in flight
    // keeps its copies dirty until its writes return, so it is not skipped.
    if (!TagDirty(tag)) {
      *rc = 0;
      CLIO_CO_RETURN;
    }
    const std::string id = "flush." + Key(tag, "");
    bool leader = false;
    CLIO_CO_AWAIT(JoinRound(id, &leader, rc));
    if (leader) {
      CLIO_CO_AWAIT(FlushAll(tag, false, rc));
      LeaveRound(id, *rc);
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::u32 CacheBase::OwnerOf(const TagId &tag, const std::string &name) {
  const clio::run::u32 n = core::PoolContainers(pool_id_);
  if (n <= 1) return container_id_;
  return core::FailoverContainer(pool_id_, core::BlobHash(tag, name) % n);
}

clio::run::TaskResume CacheBase::PutAtOwner(
    clio::run::shared_ptr<core::PutBlobTask> task, clio::run::u32 owner) {
  CLIO_TASK_BODY_BEGIN
  {
    const std::string name = task->blob_name_.str();
    // An append is not idempotent: a lost one may have landed.
    const bool may_resend = (task->flags_ & core::kCtePutAppend) == 0;
    std::vector<core::BlobRegion> regions;
    if (task->segments_.empty()) {
      regions.push_back(core::BlobRegion{task->offset_, task->size_,
                                         task->blob_data_});
    } else {
      core::ForEachBlobRegion(*task, [&regions](const core::BlobRegion &r) {
        regions.push_back(r);
        return true;
      });
    }
    std::vector<clio::run::Future<core::PutBlobTask>> futs;
    for (const core::BlobRegion &r : regions) {
      futs.push_back(self_client_.AsyncPutBlob(
          task->tag_id_, name, r.blob_off_, r.size_, r.data_, task->score_,
          task->context_, task->flags_, clio::run::PoolQuery::DirectId(owner)));
    }
    clio::run::u32 rc = 0;
    for (size_t i = 0; i < futs.size(); ++i) {
      CLIO_CO_AWAIT(futs[i]);
      // The owner died with the put in flight: once failover names its
      // successor, write there (idempotent), as the core's forwarding does.
      for (int attempt = 0; may_resend && attempt < kOwnerRetries &&
                            core::IsNodeLostRc(futs[i]->GetReturnCode());
           ++attempt) {
        CLIO_CO_AWAIT(clio::run::yield(kOwnerRetryUs));
        const core::BlobRegion &r = regions[i];
        futs[i] = self_client_.AsyncPutBlob(
            task->tag_id_, name, r.blob_off_, r.size_, r.data_, task->score_,
            task->context_, task->flags_,
            clio::run::PoolQuery::DirectId(OwnerOf(task->tag_id_, name)));
        CLIO_CO_AWAIT(futs[i]);
      }
      if (rc == 0) rc = futs[i]->GetReturnCode();
    }
    if (task->segments_.empty()) {
      // INOUT: an append reports where it landed.
      task->context_ = futs[0]->context_;
      task->offset_ = futs[0]->offset_;
      task->blob_name_ = futs[0]->blob_name_;
    }
    task->SetReturnCode(rc);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume CacheBase::GetAtOwner(
    clio::run::shared_ptr<core::GetBlobTask> task, clio::run::u32 owner) {
  CLIO_TASK_BODY_BEGIN
  {
    const std::string name = task->blob_name_.str();
    std::vector<core::BlobRegion> regions;
    if (task->segments_.empty()) {
      regions.push_back(core::BlobRegion{task->offset_, task->size_,
                                         task->blob_data_});
    } else {
      core::ForEachBlobRegion(*task, [&regions](const core::BlobRegion &r) {
        regions.push_back(r);
        return true;
      });
    }
    std::vector<clio::run::Future<core::GetBlobTask>> futs;
    for (const core::BlobRegion &r : regions) {
      futs.push_back(self_client_.AsyncGetBlob(
          task->tag_id_, name.c_str(), r.blob_off_, r.size_, task->flags_,
          r.data_, clio::run::PoolQuery::DirectId(owner), task->context_));
    }
    clio::run::u32 rc = 0;
    for (size_t i = 0; i < futs.size(); ++i) {
      CLIO_CO_AWAIT(futs[i]);
      for (int attempt = 0; attempt < kOwnerRetries &&
                            core::IsNodeLostRc(futs[i]->GetReturnCode());
           ++attempt) {
        CLIO_CO_AWAIT(clio::run::yield(kOwnerRetryUs));
        const core::BlobRegion &r = regions[i];
        futs[i] = self_client_.AsyncGetBlob(
            task->tag_id_, name.c_str(), r.blob_off_, r.size_, task->flags_,
            r.data_,
            clio::run::PoolQuery::DirectId(OwnerOf(task->tag_id_, name)),
            task->context_);
        CLIO_CO_AWAIT(futs[i]);
      }
      if (rc == 0) rc = futs[i]->GetReturnCode();
    }
    if (task->segments_.empty()) task->context_ = futs[0]->context_;
    task->SetReturnCode(rc);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume CacheBase::TagCopiesEverywhere(TagId tag,
                                                     clio::run::u32 mode,
                                                     clio::run::u32 *rc) {
  CLIO_TASK_BODY_BEGIN
  {
    // Every live container's copies. A dead container's copies died with its
    // memory, so there is nothing of it to flush or drop: skip it, and do
    // not fail the operation when one dies mid-call.
    *rc = 0;
    auto *ipc = CLIO_IPC;
    const clio::run::u32 n = std::max<clio::run::u32>(1, core::PoolContainers(pool_id_));
    std::vector<clio::run::Future<core::DelTagTask>> futs;
    for (clio::run::u32 c = 0; c < n; ++c) {
      if (n > 1 && !core::ContainerNodeAlive(pool_id_, c)) continue;
      auto t = ipc->NewTask<core::DelTagTask>(
          clio::run::CreateTaskId(), pool_id_,
          clio::run::PoolQuery::DirectId(static_cast<clio::run::ContainerId>(c)),
          tag);
      t->method_ = kDropTagCopies;
      t->posix_unlink_ = mode;
      futs.push_back(ipc->Send(t));
    }
    for (auto &f : futs) {
      CLIO_CO_AWAIT(f);
      const clio::run::u32 r = f->GetReturnCode();
      if (r != 0 && !core::IsNodeLostRc(r) && *rc == 0) *rc = r;
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

// ===========================================================================
// Verbs
// ===========================================================================

clio::run::TaskResume CacheBase::PutThrough(
    clio::run::shared_ptr<core::PutBlobTask> task, clio::run::u32 owner) {
  CLIO_TASK_BODY_BEGIN
  if (owner == container_id_) {
    CLIO_CO_AWAIT(ForwardToCore(Method::kPutBlob,
                                task.template Cast<clio::run::Task>()));
  } else {
    CLIO_CO_AWAIT(PutAtOwner(task, owner));
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume CacheBase::PublishBuffered(std::string key,
                                                 clio::run::u32 *rc) {
  CLIO_TASK_BODY_BEGIN
  *rc = 0;
  if (Cached(key)) {
    CLIO_CO_AWAIT(Lock(key));
    CLIO_CO_AWAIT(Drop(key, true, rc));  // flush to the owner, then drop
    Unlock(key);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume CacheBase::PutPersistentFirst(
    clio::run::shared_ptr<core::PutBlobTask> task, std::string key,
    bool special, clio::run::u64 off, clio::run::u64 end) {
  CLIO_TASK_BODY_BEGIN
  {
    const std::string name = task->blob_name_.str();
    clio::run::u32 rc = 0;
    // A write-back copy is flushed and dropped before these bytes go down, so
    // an older dirty range cannot land over them later.
    if (special || IsWriteBack()) CLIO_CO_AWAIT(Drop(key, true, &rc));
    if (rc == 0) {
      CLIO_CO_AWAIT(ForwardToCore(Method::kPutBlob,
                                  task.template Cast<clio::run::Task>()));
      rc = task->GetReturnCode();
      if (!special && rc == 0 && !IsWriteBack()) {
        // Write-through: the same put goes to volatile too.
        clio::run::u32 vrc = 0;
        CLIO_CO_AWAIT(PrepareAbsorb(task->tag_id_, name, off, end, &vrc));
        if (vrc == 0) CLIO_CO_AWAIT(PutVolatile(task, &vrc));
        if (vrc == 0) {
          NoteWrite(key, task->tag_id_, name, off, end, false);
        } else {
          CLIO_CO_AWAIT(DropVolatile(task->tag_id_, name));
        }
      }
    }
    task->SetReturnCode(rc);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume CacheBase::AbsorbLocked(
    clio::run::shared_ptr<core::PutBlobTask> task, std::string key,
    clio::run::u32 owner, clio::run::u64 off, clio::run::u64 end) {
  CLIO_TASK_BODY_BEGIN
  {
    const std::string name = task->blob_name_.str();
    const bool here = owner == container_id_;
    bool bypass = false;
    CLIO_CO_AWAIT(MakeRoom(key, end - off, &bypass));
    if (bypass) {
      // Volatile is full of dirty copies and holds none of this blob: the
      // put would be refused and written through anyway.
      CLIO_CO_AWAIT(PutThrough(task, owner));
      CLIO_CO_RETURN;
    }
    // Another node's blob is buffered as the written ranges only: it is
    // never read from here (a read lands it first), so it needs no fill.
    clio::run::u32 rc = 0;
    if (here) CLIO_CO_AWAIT(PrepareAbsorb(task->tag_id_, name, off, end, &rc));
    if (rc == 0) CLIO_CO_AWAIT(PutVolatile(task, &rc));
    if (rc == 0) {
      NoteWrite(key, task->tag_id_, name, off, end, true);
    } else if (!IsNoSpaceRc(rc) || params_.fallback_to_persistent_) {
      // No whole copy could be made -- volatile is full even after
      // eviction, or persistent could not supply the blob's other bytes
      // (an owner that just died, its stand-in still catching up): write
      // this blob through instead of failing the put.
      CLIO_CO_AWAIT(Drop(key, true, &rc));
      CLIO_CO_AWAIT(DropVolatile(task->tag_id_, name));
      if (rc == 0) {
        CLIO_CO_AWAIT(PutThrough(task, owner));
        rc = task->GetReturnCode();
      }
    } else {
      // No fallback: the put fails whole, leaving nothing behind.
      if (!Cached(key)) CLIO_CO_AWAIT(DropVolatile(task->tag_id_, name));
      rc = core::kPutNoSpaceRc;
    }
    task->SetReturnCode(rc);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume CacheBase::PutBlob(
    clio::run::shared_ptr<core::PutBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    if (!HasPersistent()) {
      CLIO_CO_AWAIT(ForwardToCore(Method::kPutBlob,
                                  task.template Cast<clio::run::Task>()));
      CLIO_CO_RETURN;
    }
    // Claimed before the route is decided: Destroy waits for every claim
    // before it flushes, so a put that buffers here is flushed.
    const BufferSlot slot(&buffering_, &draining_, IsWriteBack());
    const std::string name = task->blob_name_.str();
    const clio::run::u32 owner = OwnerOf(task->tag_id_, name);
    const bool here = owner == container_id_;
    const std::string key = Key(task->tag_id_, name);
    const bool special = IsSpecial(task->flags_, task->context_);
    if (!here && (!slot.Admitted() || AtOwner(task->flags_, task->context_))) {
      // Write-through, draining, a put persistent decides, or metadata: the
      // owner applies it, after any bytes of it buffered here have landed.
      clio::run::u32 prc = 0;
      if (IsWriteBack()) CLIO_CO_AWAIT(PublishBuffered(key, &prc));
      if (prc != 0) {
        task->SetReturnCode(prc);
        CLIO_CO_RETURN;
      }
      CLIO_CO_AWAIT(PutAtOwner(task, owner));
      CLIO_CO_RETURN;
    }
    clio::run::u64 off = ~0ull;
    clio::run::u64 end = 0;
    core::ForEachBlobRegion(*task, [&off, &end](const core::BlobRegion &r) {
      off = std::min(off, r.blob_off_);
      end = std::max(end, r.blob_off_ + r.size_);
      return true;
    });
    if (off == ~0ull) off = 0;
    CLIO_CO_AWAIT(Lock(key));
    if (special || !slot.Admitted()) {
      CLIO_CO_AWAIT(PutPersistentFirst(task, key, special, off, end));
    } else {
      CLIO_CO_AWAIT(AbsorbLocked(task, key, owner, off, end));
    }
    Unlock(key);
    if (slot.Admitted() && !special) CLIO_CO_AWAIT(CheckFill());
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume CacheBase::MultiPutFanOut(
    clio::run::shared_ptr<core::MultiPutBlobTask> task,
    core::MultiPutBatchView &batch) {
  CLIO_TASK_BODY_BEGIN
  {
    // Records of other owners (an interposer above forwarded the batch to
    // its local container): each goes to its owner, all at once.
    task->num_ok_ = 0;
    task->first_rc_ = 0;
    std::vector<clio::run::Future<core::PutBlobTask>> futs;
    std::vector<size_t> rec_of;
    for (size_t i = 0; i < batch.size(); ++i) {
      const core::MultiPutDesc &d = batch.descs_[i];
      if (!batch.RecordValid(i)) {
        if (task->first_rc_ == 0) task->first_rc_ = 2;
        continue;
      }
      if (d.size_ == 0) {  // nothing to write (the client sends no task)
        ++task->num_ok_;
        continue;
      }
      futs.push_back(self_client_.AsyncPutBlob(
          d.tag_id_, d.blob_name_, d.offset_, d.size_,
          batch.base_ + d.payload_off_, -1.0f, task->context_, d.flags_,
          clio::run::PoolQuery::DirectId(OwnerOf(d.tag_id_, d.blob_name_))));
      rec_of.push_back(i);
    }
    for (size_t k = 0; k < futs.size(); ++k) {
      auto &f = futs[k];
      CLIO_CO_AWAIT(f);
      for (int a = 0; a < kOwnerRetries && core::IsNodeLostRc(f->GetReturnCode());
           ++a) {
        CLIO_CO_AWAIT(clio::run::yield(kOwnerRetryUs));
        const core::MultiPutDesc &d = batch.descs_[rec_of[k]];
        f = self_client_.AsyncPutBlob(
            d.tag_id_, d.blob_name_, d.offset_, d.size_,
            batch.base_ + d.payload_off_, -1.0f, task->context_, d.flags_,
            clio::run::PoolQuery::DirectId(OwnerOf(d.tag_id_, d.blob_name_)));
        CLIO_CO_AWAIT(f);
      }
      const clio::run::u32 rc = f->GetReturnCode();
      if (rc == 0) {
        ++task->num_ok_;
      } else if (task->first_rc_ == 0) {
        task->first_rc_ = static_cast<int>(rc);
      }
    }
    task->SetReturnCode(task->first_rc_ == 0 ? 0 : 1);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume CacheBase::MultiPutThrough(
    clio::run::shared_ptr<core::MultiPutBlobTask> task,
    core::MultiPutBatchView &batch, bool attached) {
  CLIO_TASK_BODY_BEGIN
  {
    // Write-through (or nothing to absorb): persistent takes the batch;
    // the records' volatile copies are dropped (the next read refills).
    // Every record's lock is held across the forward AND the drop, as a
    // single put holds its own: otherwise a put or read-fill of the same
    // blob sizes its volatile copy from persistent before the batch lands
    // and installs it after the drop check -- a copy missing the batch's
    // bytes, served to every later read (clio-fs pages read zeros).
    std::vector<std::string> keys;
    for (size_t i = 0; attached && i < batch.size(); ++i) {
      keys.push_back(Key(batch.descs_[i].tag_id_, batch.descs_[i].blob_name_));
    }
    std::sort(keys.begin(), keys.end());  // one global lock order
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    // A copy's dirty ranges land before the batch, which may overwrite
    // them; no copy can be refilled until the locks are released.
    for (size_t i = 0; i < keys.size(); ++i) CLIO_CO_AWAIT(Lock(keys[i]));
    for (size_t i = 0; i < keys.size(); ++i) {
      if (!Cached(keys[i])) continue;
      clio::run::u32 r = 0;
      CLIO_CO_AWAIT(Drop(keys[i], true, &r));
    }
    CLIO_CO_AWAIT(ForwardToCore(Method::kMultiPutBlob,
                                task.template Cast<clio::run::Task>()));
    for (size_t i = 0; i < keys.size(); ++i) Unlock(keys[i]);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume CacheBase::MultiPutAbsorbRecords(
    clio::run::shared_ptr<core::MultiPutBlobTask> task,
    core::MultiPutBatchView &batch) {
  CLIO_TASK_BODY_BEGIN
  {
    task->num_ok_ = 0;
    task->first_rc_ = 0;
    for (size_t i = 0; i < batch.size(); ++i) {
      const core::MultiPutDesc &d = batch.descs_[i];
      if (!batch.RecordValid(i)) {
        if (task->first_rc_ == 0) task->first_rc_ = 2;
        continue;
      }
      const std::string key = Key(d.tag_id_, d.blob_name_);
      const char *src = batch.base_ + d.payload_off_;
      const clio::run::u32 owner = OwnerOf(d.tag_id_, d.blob_name_);
      const bool here = owner == container_id_;
      CLIO_CO_AWAIT(Lock(key));
      clio::run::u32 rc = 0;
      const bool special = IsSpecial(d.flags_, task->context_);
      bool bypass = false;
      if (!special) CLIO_CO_AWAIT(MakeRoom(key, d.size_, &bypass));
      if (bypass) {
        // Volatile is full of dirty copies and holds none of this blob.
        auto put = here ? pers_client_.AsyncPutBlob(
                              d.tag_id_, d.blob_name_, d.offset_, d.size_,
                              src, -1.0f, task->context_, d.flags_,
                              clio::run::PoolQuery::Local())
                        : self_client_.AsyncPutBlob(
                              d.tag_id_, d.blob_name_, d.offset_, d.size_,
                              src, -1.0f, task->context_, d.flags_,
                              clio::run::PoolQuery::DirectId(owner));
        CLIO_CO_AWAIT(put);
        rc = put->GetReturnCode();
      } else if (!special) {
        if (here) {
          CLIO_CO_AWAIT(PrepareAbsorb(d.tag_id_, d.blob_name_, d.offset_,
                                      d.offset_ + d.size_, &rc));
        }
        if (rc == 0) {
          CLIO_CO_AWAIT(PutVolatileRaw(d.tag_id_, d.blob_name_, d.offset_,
                                       d.size_, src, &rc));
        }
        if (rc == 0) {
          NoteWrite(key, d.tag_id_, d.blob_name_, d.offset_,
                    d.offset_ + d.size_, true);
        }
      }
      // Write the record through when it cannot be absorbed whole (see
      // PutBlob); only a capacity failure with no fallback fails it.
      if (!bypass &&
          (special || (rc != 0 && (!IsNoSpaceRc(rc) ||
                                   params_.fallback_to_persistent_)))) {
        rc = 0;
        CLIO_CO_AWAIT(Drop(key, true, &rc));
        CLIO_CO_AWAIT(DropVolatile(d.tag_id_, d.blob_name_));
        if (rc == 0) {
          auto put = here ? pers_client_.AsyncPutBlob(
                                d.tag_id_, d.blob_name_, d.offset_, d.size_,
                                src, -1.0f, task->context_, d.flags_,
                                clio::run::PoolQuery::Local())
                          : self_client_.AsyncPutBlob(
                                d.tag_id_, d.blob_name_, d.offset_, d.size_,
                                src, -1.0f, task->context_, d.flags_,
                                clio::run::PoolQuery::DirectId(owner));
          CLIO_CO_AWAIT(put);
          rc = put->GetReturnCode();
        }
      }
      Unlock(key);
      if (rc == 0) {
        ++task->num_ok_;
      } else if (task->first_rc_ == 0) {
        task->first_rc_ = static_cast<int>(rc);
      }
    }
    task->SetReturnCode(task->first_rc_ == 0 ? 0 : 1);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume CacheBase::MultiPutBlob(
    clio::run::shared_ptr<core::MultiPutBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    core::MultiPutBatchView batch;
    const bool attached = core::MultiPutBatchView::Attach(*task, &batch);
    // A range another node flushed lands here on its owner: persistent
    // takes it and this cache's copies of it are dropped (write-through).
    const bool land = task->route_blob_.str() == kLandBlob;
    // Write-back claims buffering for the batch (a landing never buffers).
    const BufferSlot slot(&buffering_, &draining_, IsWriteBack() && !land);
    bool all_here = true;
    // Write-back absorbs every record where it is written (no owner hop),
    // except metadata, which its owner applies, and, while draining, every
    // record: each goes to its owner, as on write-through.
    const bool route =
        !IsWriteBack() ||
        (!land && (!slot.Admitted() || AtOwner(0, task->context_)));
    for (size_t i = 0; HasPersistent() && attached && route &&
                       i < batch.size(); ++i) {
      all_here = all_here && OwnerOf(batch.descs_[i].tag_id_,
                                     batch.descs_[i].blob_name_) == container_id_;
    }
    if (!all_here) {
      CLIO_CO_AWAIT(MultiPutFanOut(task, batch));
      CLIO_CO_RETURN;
    }
    if (!HasPersistent() || !attached || !slot.Admitted() || land) {
      CLIO_CO_AWAIT(MultiPutThrough(task, batch, attached));
      CLIO_CO_RETURN;
    }
    {
      bool done = false;
      CLIO_CO_AWAIT(AbsorbBatchFast(task, &done));
      if (done) {
        CLIO_CO_AWAIT(CheckFill());
        CLIO_CO_RETURN;
      }
    }
    CLIO_CO_AWAIT(MultiPutAbsorbRecords(task, batch));
    CLIO_CO_AWAIT(CheckFill());
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume CacheBase::AbsorbBatchFast(
    clio::run::shared_ptr<core::MultiPutBlobTask> task, bool *done) {
  CLIO_TASK_BODY_BEGIN
  *done = false;
  {
    core::MultiPutBatchView batch;
    if (!core::MultiPutBatchView::Attach(*task, &batch)) CLIO_CO_RETURN;
    const size_t n = batch.size();
    std::vector<std::string> keys(n);
    std::vector<bool> here(n);
    for (size_t i = 0; i < n; ++i) {
      const core::MultiPutDesc &d = batch.descs_[i];
      // The common case only: anything else takes the per-record path.
      if (!batch.RecordValid(i) || IsSpecial(d.flags_, task->context_) ||
          KnownFull(d.size_)) {
        CLIO_CO_RETURN;
      }
      keys[i] = Key(d.tag_id_, d.blob_name_);
      here[i] = OwnerOf(d.tag_id_, d.blob_name_) == container_id_;
    }
    std::vector<std::string> order(keys);
    std::sort(order.begin(), order.end());  // the one global lock order
    order.erase(std::unique(order.begin(), order.end()), order.end());
    for (const std::string &k : order) CLIO_CO_AWAIT(Lock(k));
    // A partial write to a blob only persistent holds needs that blob's
    // other bytes first (PrepareAbsorb): ask for every size at once.
    std::vector<clio::run::Future<core::GetBlobSizeTask>> sz(n);
    for (size_t i = 0; i < n; ++i) {
      const core::MultiPutDesc &d = batch.descs_[i];
      if (here[i] && !Cached(keys[i])) {
        sz[i] = pers_client_.AsyncGetBlobSize(d.tag_id_, d.blob_name_,
                                              clio::run::PoolQuery::Local());
      }
    }
    bool simple = true;
    for (size_t i = 0; i < n; ++i) {
      if (sz[i].get() == nullptr) continue;
      CLIO_CO_AWAIT(sz[i]);
      const core::MultiPutDesc &d = batch.descs_[i];
      const clio::run::u64 psize = sz[i]->GetReturnCode() == 0 ? sz[i]->size_ : 0;
      if (psize > 0 && !(d.offset_ == 0 && d.offset_ + d.size_ >= psize)) {
        simple = false;
      }
    }
    // Every record into volatile at once.
    std::vector<clio::run::Future<core::PutBlobTask>> puts(n);
    for (size_t i = 0; simple && i < n; ++i) {
      const core::MultiPutDesc &d = batch.descs_[i];
      puts[i] = vol_client_.AsyncPutBlob(
          d.tag_id_, d.blob_name_, d.offset_, d.size_,
          batch.base_ + d.payload_off_, -1.0f, Context(), 0,
          clio::run::PoolQuery::Local());
    }
    for (size_t i = 0; simple && i < n; ++i) {
      CLIO_CO_AWAIT(puts[i]);
      if (puts[i].get() == nullptr || puts[i]->GetReturnCode() != 0) {
        simple = false;  // the per-record path retries (puts are idempotent)
      }
    }
    if (simple) {
      for (size_t i = 0; i < n; ++i) {
        const core::MultiPutDesc &d = batch.descs_[i];
        NoteWrite(keys[i], d.tag_id_, d.blob_name_, d.offset_,
                  d.offset_ + d.size_, true);
      }
      task->num_ok_ = static_cast<clio::run::u32>(n);
      task->first_rc_ = 0;
      task->SetReturnCode(0);
      *done = true;
    }
    for (const std::string &k : order) Unlock(k);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume CacheBase::GetBlob(
    clio::run::shared_ptr<core::GetBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    if (!HasPersistent()) {
      CLIO_CO_AWAIT(ForwardToCore(Method::kGetBlob,
                                  task.template Cast<clio::run::Task>()));
      CLIO_CO_RETURN;
    }
    const std::string name = task->blob_name_.str();
    const clio::run::u32 owner = OwnerOf(task->tag_id_, name);
    if (owner != container_id_) {
      CLIO_CO_AWAIT(GetAtOwner(task, owner));
      CLIO_CO_RETURN;
    }
    const std::string key = Key(task->tag_id_, name);
    CLIO_CO_AWAIT(Lock(key));
    clio::run::u32 rc = 1;
    if (Cached(key)) {
      std::vector<core::BlobRegion> regions;
      core::ForEachBlobRegion(*task, [&regions](const core::BlobRegion &r) {
        regions.push_back(r);
        return true;
      });
      rc = 0;
      for (size_t i = 0; i < regions.size() && rc == 0; ++i) {
        auto get = vol_client_.AsyncGetBlob(
            task->tag_id_, name.c_str(), regions[i].blob_off_,
            regions[i].size_, task->flags_, regions[i].data_,
            clio::run::PoolQuery::Local(), task->context_);
        CLIO_CO_AWAIT(get);
        rc = get->GetReturnCode();
      }
      if (rc == 0) {
        Touch(key);
        task->SetReturnCode(0);
      } else {
        // The copy failed to serve: flush what it holds, then read through.
        CLIO_CO_AWAIT(Drop(key, true, &rc));
        rc = 1;
      }
    }
    if (rc != 0) {
      CLIO_CO_AWAIT(ForwardToCore(Method::kGetBlob,
                                  task.template Cast<clio::run::Task>()));
      // No fill while volatile is full of dirty copies: it would re-read
      // the blob from persistent only to have the copy refused.
      const bool full = EvictCleanOnly() && KnownFull(task->size_) &&
                        no_clean_.load(std::memory_order_relaxed);
      if (task->GetReturnCode() == 0 && !full && ShouldFill(key)) {
        auto sz = pers_client_.AsyncGetBlobSize(task->tag_id_, name,
                                                clio::run::PoolQuery::Local());
        CLIO_CO_AWAIT(sz);
        if (sz->GetReturnCode() == 0 && sz->size_ > 0) {
          clio::run::u32 frc = 0;
          CLIO_CO_AWAIT(FillFromPersistent(task->tag_id_, name, sz->size_,
                                           &frc));
        }
      }
    }
    Unlock(key);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume CacheBase::GetBlobSize(
    clio::run::shared_ptr<core::GetBlobSizeTask> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    const std::string name = task->blob_name_.str();
    const clio::run::u32 owner = OwnerOf(task->tag_id_, name);
    if (HasPersistent() && owner != container_id_) {
      auto f = self_client_.AsyncGetBlobSize(
          task->tag_id_, name, clio::run::PoolQuery::DirectId(owner),
          task->replica_);
      CLIO_CO_AWAIT(f);
      for (int a = 0; a < kOwnerRetries && core::IsNodeLostRc(f->GetReturnCode());
           ++a) {
        CLIO_CO_AWAIT(clio::run::yield(kOwnerRetryUs));
        f = self_client_.AsyncGetBlobSize(
            task->tag_id_, name,
            clio::run::PoolQuery::DirectId(OwnerOf(task->tag_id_, name)),
            task->replica_);
        CLIO_CO_AWAIT(f);
      }
      task->size_ = f->size_;
      task->lost_bytes_ = f->lost_bytes_;
      task->SetReturnCode(f->GetReturnCode());
      CLIO_CO_RETURN;
    }
    const std::string key = Key(task->tag_id_, name);
    clio::run::u64 size = 0;
    if (HasPersistent() && task->replica_ == 0 && Cached(key, &size)) {
      task->size_ = size;
      task->lost_bytes_ = 0;
      task->SetReturnCode(0);
      CLIO_CO_RETURN;
    }
    CLIO_CO_AWAIT(ForwardToCore(Method::kGetBlobSize,
                                task.template Cast<clio::run::Task>()));
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume CacheBase::DelBlob(
    clio::run::shared_ptr<core::DelBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    const std::string name = task->blob_name_.str();
    const clio::run::u32 owner = OwnerOf(task->tag_id_, name);
    bool buffered = false;
    if (HasPersistent() && owner != container_id_ && IsWriteBack()) {
      // Its bytes buffered here die with it (as the owner's copy does).
      const std::string key = Key(task->tag_id_, name);
      CLIO_CO_AWAIT(Lock(key));
      buffered = Cached(key);
      clio::run::u32 drc = 0;
      CLIO_CO_AWAIT(Drop(key, false, &drc));
      Unlock(key);
    }
    if (HasPersistent() && owner != container_id_) {
      auto f = self_client_.AsyncDelBlob(task->tag_id_, name,
                                         clio::run::PoolQuery::DirectId(owner),
                                         task->del_flags_);
      CLIO_CO_AWAIT(f);
      for (int a = 0; a < kOwnerRetries && core::IsNodeLostRc(f->GetReturnCode());
           ++a) {
        CLIO_CO_AWAIT(clio::run::yield(kOwnerRetryUs));
        f = self_client_.AsyncDelBlob(
            task->tag_id_, name,
            clio::run::PoolQuery::DirectId(OwnerOf(task->tag_id_, name)),
            task->del_flags_);
        CLIO_CO_AWAIT(f);
      }
      // A blob never flushed existed only in this node's buffer.
      task->SetReturnCode(buffered ? 0 : f->GetReturnCode());
      CLIO_CO_RETURN;
    }
    const std::string key = Key(task->tag_id_, name);
    CLIO_CO_AWAIT(Lock(key));
    const bool had = Cached(key);
    clio::run::u32 rc = 0;
    CLIO_CO_AWAIT(Drop(key, false, &rc));  // its dirty bytes die with it
    CLIO_CO_AWAIT(ForwardToCore(Method::kDelBlob,
                                task.template Cast<clio::run::Task>()));
    if (had && task->GetReturnCode() != 0) {
      task->SetReturnCode(0);  // a never-flushed blob existed only here
    }
    Unlock(key);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume CacheBase::TruncateBlob(
    clio::run::shared_ptr<core::TruncateBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    const std::string name = task->blob_name_.str();
    const clio::run::u32 owner = OwnerOf(task->tag_id_, name);
    if (HasPersistent() && owner != container_id_) {
      clio::run::u32 prc = 0;
      if (IsWriteBack()) {
        CLIO_CO_AWAIT(PublishBuffered(Key(task->tag_id_, name), &prc));
      }
      if (prc != 0) {
        task->SetReturnCode(prc);
        CLIO_CO_RETURN;
      }
      auto f = self_client_.AsyncTruncateBlob(
          task->tag_id_, name, task->new_size_,
          clio::run::PoolQuery::DirectId(owner));
      CLIO_CO_AWAIT(f);
      for (int a = 0; a < kOwnerRetries && core::IsNodeLostRc(f->GetReturnCode());
           ++a) {
        CLIO_CO_AWAIT(clio::run::yield(kOwnerRetryUs));
        f = self_client_.AsyncTruncateBlob(
            task->tag_id_, name, task->new_size_,
            clio::run::PoolQuery::DirectId(OwnerOf(task->tag_id_, name)));
        CLIO_CO_AWAIT(f);
      }
      task->SetReturnCode(f->GetReturnCode());
      CLIO_CO_RETURN;
    }
    const std::string key = Key(task->tag_id_, name);
    CLIO_CO_AWAIT(Lock(key));
    clio::run::u32 rc = 0;
    CLIO_CO_AWAIT(Drop(key, true, &rc));
    if (rc != 0) {
      task->SetReturnCode(rc);
    } else {
      CLIO_CO_AWAIT(ForwardToCore(Method::kTruncateBlob,
                                  task.template Cast<clio::run::Task>()));
    }
    Unlock(key);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume CacheBase::SyncTag(
    clio::run::shared_ptr<core::SyncTagTask> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    // The tag's dirty copies live on their blobs' owners: every container
    // flushes its own (however this task arrived), then the persistent
    // chain syncs its devices.
    //
    // Concurrent syncs of one tag (every rank of a shared file calls fsync)
    // share a round: one flushes and syncs, the others wait for the first
    // round that started after they arrived and take its result.
    namespace ev = clio::run::evlat;
    // Only syncs asking for the same thing share.
    const std::string id = "sync." + Key(task->tag_id_, "") +
                           std::to_string(task->min_persistence_) + "." +
                           std::to_string(task->sync_flags_);
    bool leader = false;
    clio::run::u32 rc = 0;
    SyncOut out;
    CLIO_CO_AWAIT(JoinRound(id, &leader, &rc, &out));
    if (!leader) {
      task->deferred_ = out.deferred_;
      task->blobs_moved_ = out.blobs_moved_;
      task->bdevs_synced_ = out.bdevs_synced_;
      task->containers_ = out.containers_;
      task->liveness_change_ns_ = out.liveness_change_ns_;
      task->SetReturnCode(rc);
      CLIO_CO_RETURN;
    }
    const unsigned long long t0 = ev::Now();
    CLIO_CO_AWAIT(FlushTagShared(task->tag_id_, &rc));
    const unsigned long long t1 = ev::Now();
    ev::Span(ev::kWbSyncLocal, t0, t1);
    // A broadcast sync already reached every container, each flushing its
    // own copies: asking them all again would make one fsync N^2 messages.
    const bool fanned = (task->sync_flags_ & core::kSyncFannedOut) != 0;
    if (rc == 0 && IsWriteBack() && !fanned &&
        core::PoolContainers(pool_id_) > 1) {
      CLIO_CO_AWAIT(TagCopiesEverywhere(task->tag_id_, kCopiesFlush, &rc));
      ev::Span(ev::kWbSyncAll, t1, ev::Now());
    }
    if (rc != 0) {
      // Any put no-space code: a tier with no room or a device that refused
      // the bytes (rc 28, e.g. a full DAOS pool) is ENOSPC, not EIO.
      task->SetReturnCode(core::PutRcIsNoSpace(rc) ? core::kSyncNoSpaceRc
                                                   : core::kSyncIoRc);
    } else {
      const unsigned long long t2 = ev::Now();
      CLIO_CO_AWAIT(ForwardToCore(Method::kSyncTag,
                                  task.template Cast<clio::run::Task>()));
      ev::Span(ev::kWbSyncCore, t2, ev::Now());
    }
    out.deferred_ = task->deferred_;
    out.blobs_moved_ = task->blobs_moved_;
    out.bdevs_synced_ = task->bdevs_synced_;
    out.containers_ = task->containers_;
    out.liveness_change_ns_ = task->liveness_change_ns_;
    LeaveRound(id, task->GetReturnCode(), &out);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume CacheBase::FlushData(
    clio::run::shared_ptr<core::FlushDataTask> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    // The periodic flush, a fill-level flush and a stop flush share rounds.
    clio::run::u32 rc = 0;
    FlushStats stats;
    CLIO_CO_AWAIT(FlushEverythingShared(&rc, &stats));
    fill_flush_pending_.store(false, std::memory_order_relaxed);
    task->blobs_flushed_ = stats.blobs_;
    task->bytes_flushed_ = stats.bytes_;
    task->SetReturnCode(rc);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume CacheBase::DelTag(
    clio::run::shared_ptr<core::DelTagTask> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    // Persistent first (it resolves a name to the id); then every
    // container drops its copies, and every core container its share.
    CLIO_CO_AWAIT(ForwardToCore(Method::kDelTag,
                                task.template Cast<clio::run::Task>()));
    const clio::run::u32 rc = task->GetReturnCode();
    const TagId tag = task->tag_id_;
    if (!tag.IsNull()) {
      auto shares = pers_client_.AsyncMultiDelTag(
          {tag}, clio::run::PoolQuery::Broadcast());
      auto *ipc = CLIO_IPC;
      auto drop = ipc->NewTask<core::DelTagTask>(
          clio::run::CreateTaskId(), pool_id_,
          clio::run::PoolQuery::Broadcast(), tag);
      drop->method_ = kDropTagCopies;
      drop->posix_unlink_ = kCopiesDiscard;
      auto fut = ipc->Send(drop);
      CLIO_CO_AWAIT(shares);
      CLIO_CO_AWAIT(fut);
    }
    task->SetReturnCode(rc);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume CacheBase::DropTagCopies(
    clio::run::shared_ptr<core::DelTagTask> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    const TagId tag = task->tag_id_;
    const clio::run::u32 mode = task->posix_unlink_;
    if (mode == kCopiesFlush) {
      // Flush and keep: several blobs at a time, shared by concurrent
      // requests for the same tag.
      clio::run::u32 frc = 0;
      if (tag.IsNull()) {
        CLIO_CO_AWAIT(FlushAll(tag, true, &frc));
      } else {
        CLIO_CO_AWAIT(FlushTagShared(tag, &frc));
      }
      task->SetReturnCode(frc);
      CLIO_CO_RETURN;
    }
    std::vector<std::string> keys = KeysOf(tag.IsNull() ? nullptr : &tag);
    clio::run::u32 first = 0;
    for (size_t i = 0; i < keys.size(); ++i) {
      CLIO_CO_AWAIT(Lock(keys[i]));
      clio::run::u32 rc = 0;
      CLIO_CO_AWAIT(Drop(keys[i], mode == kCopiesFlushDrop, &rc));
      Unlock(keys[i]);
      if (rc != 0 && first == 0) first = rc;
    }
    task->SetReturnCode(first);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume CacheBase::ForwardCoherent(
    clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> task) {
  CLIO_TASK_BODY_BEGIN
  {
    TagId tag = TagId::GetNull();
    bool any_tag = false;  // reads state across tags
    bool mutates = false;  // writes blob bytes behind the cache
    switch (method) {
      case Method::kGetTagSize:
        tag = task.template Cast<core::GetTagSizeTask>()->tag_id_;
        break;
      case Method::kGetContainedBlobs:
        tag = task.template Cast<core::GetContainedBlobsTask>()->tag_id_;
        break;
      case Method::kGetBlobInfo:
        tag = task.template Cast<core::GetBlobInfoTask>()->tag_id_;
        break;
      case Method::kGetBlobScore:
        tag = task.template Cast<core::GetBlobScoreTask>()->tag_id_;
        break;
      case Method::kPodGetBlob:
        tag = task.template Cast<core::PodGetBlobTask>()->tag_id_;
        break;
      case Method::kPodMultiGetBlob:
        tag = task.template Cast<core::PodMultiGetBlobTask>()->tag_id_;
        break;
      case Method::kPodPutBlob:
        tag = task.template Cast<core::PodPutBlobTask>()->tag_id_;
        mutates = true;
        break;
      case Method::kPodMultiPutBlob:
        tag = task.template Cast<core::PodMultiPutBlobTask>()->tag_id_;
        mutates = true;
        break;
      case Method::kBlobQuery:
      case Method::kListLocalBlobs:
      case Method::kSemanticSearch:
      case Method::kTemporalSearch:
        any_tag = true;
        break;
      default:
        break;
    }
    // The tag's copies live on its blobs' owners, all over the cluster.
    // A read must see their dirty bytes (write-back); a write behind the
    // cache makes them stale. Write-through copies are never dirty.
    const bool needed = HasPersistent() && (!tag.IsNull() || any_tag) &&
                        (mutates || IsWriteBack());
    if (needed) {
      clio::run::u32 rc = 0;
      CLIO_CO_AWAIT(TagCopiesEverywhere(any_tag ? TagId::GetNull() : tag,
                                        mutates ? kCopiesFlushDrop : kCopiesFlush,
                                        &rc));
    }
    CLIO_CO_AWAIT(ForwardToCore(method, task));
    if (needed && mutates) {
      // A read may have filled a copy of this tag from persistent while the
      // write was landing: drop again (each drop waits for that blob's fill).
      clio::run::u32 rc = 0;
      CLIO_CO_AWAIT(TagCopiesEverywhere(tag, kCopiesFlushDrop, &rc));
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

}  // namespace clio::cte::cache_base
