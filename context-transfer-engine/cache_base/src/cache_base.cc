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

#include <algorithm>
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
/** Blobs flushed to persistent at once, and the dirty bytes (so the I/O
 *  buffers held) at which a batch is closed early. */
constexpr size_t kFlushBatchBlobs = 32;
constexpr clio::run::u64 kFlushBatchBytes = 64ull << 20;
/** Least time between two fill-level queries of the volatile core (ns). */
constexpr clio::run::u64 kFillCheckNs = 10ull * 1000 * 1000;
/** Wait between checks for a shared round to complete (us). */
constexpr double kRoundPollUs = 50.0;
/** Read-miss counts kept for on_hit_count admission (then reset). */
constexpr size_t kMaxMissCounts = 1u << 20;

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
    default:
      return CoreInterposer::ScheduleTask(task);  // blob verbs -> owner
  }
}

clio::run::TaskResume CacheBase::Destroy(
    clio::run::shared_ptr<clio::run::admin::DestroyTask> &task) {
  CLIO_TASK_BODY_BEGIN
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume CacheBase::Monitor(
    clio::run::shared_ptr<clio::run::admin::MonitorTask> &task) {
  CLIO_TASK_BODY_BEGIN
  if (task->query_ == "dirty_count") {
    size_t n = 0;
    {
      std::lock_guard<std::mutex> g(mu_);
      for (const auto &e : entries_) n += e.second.dirty_.empty() ? 0 : 1;
    }
    task->results_[container_id_] = std::to_string(n);
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
  e.last_ns_ = NowNs();
  if (e.dirty_.empty()) NoteClean();
  if (dirty) dirty_gen_.fetch_add(1, std::memory_order_relaxed);
}

clio::run::TaskResume CacheBase::CheckFill() {
  CLIO_TASK_BODY_BEGIN
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

std::vector<std::pair<std::string, clio::run::u64>> CacheBase::DirtyKeysOf(
    const TagId *tag) {
  std::vector<std::pair<std::string, clio::run::u64>> keys;
  {
    std::lock_guard<std::mutex> g(mu_);
    for (const auto &e : entries_) {
      if (e.second.dirty_.empty()) continue;
      if (tag != nullptr && !(e.second.tag_ == *tag)) continue;
      clio::run::u64 bytes = 0;
      for (const Extent &x : e.second.dirty_) bytes += x.second - x.first;
      keys.emplace_back(e.first, bytes);
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
      if (*rc == 0) {
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
      if (it != entries_.end()) it->second.dirty_.clear();
      NoteClean();
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume CacheBase::FlushBatch(std::vector<std::string> keys,
                                            clio::run::u32 *rc) {
  CLIO_TASK_BODY_BEGIN
  *rc = 0;
  {
    /** One dirty range on its way from volatile to persistent. */
    struct Op {
      size_t key_;  ///< index into keys
      TagId tag_;
      std::string name_;
      Extent ext_;
      ctp::ipc::FullPtr<char> buf_;
      clio::run::u32 rc_ = 0;
      clio::run::Future<core::GetBlobTask> get_;
      clio::run::Future<core::PutBlobTask> put_;
    };
    auto *ipc = CLIO_IPC;
    std::vector<Op> ops;
    std::vector<clio::run::u32> key_rc(keys.size(), 0);
    {
      std::lock_guard<std::mutex> g(mu_);
      for (size_t k = 0; k < keys.size(); ++k) {
        auto it = entries_.find(keys[k]);
        if (it == entries_.end()) continue;
        for (const Extent &x : it->second.dirty_) {
          Op op;
          op.key_ = k;
          op.tag_ = it->second.tag_;
          op.name_ = it->second.name_;
          op.ext_ = x;
          ops.push_back(std::move(op));
        }
      }
    }
    // In waves: a wave is as many ranges as I/O buffers could be had for.
    // When the buffers run out the rest wait for this wave to return its
    // own, so a large batch never fails for memory a serial flush would
    // have found; only a range that gets no buffer with none held fails.
    size_t next = 0;
    while (next < ops.size()) {
      const size_t first = next;
      for (; next < ops.size(); ++next) {
        Op &op = ops[next];
        const clio::run::u64 len = op.ext_.second - op.ext_.first;
        op.buf_ = ipc->AllocateBuffer(len);
        if (op.buf_.IsNull()) {
          if (next > first) break;  // retry it in the next wave
          op.rc_ = 12;
          continue;
        }
        op.get_ = vol_client_.AsyncGetBlob(op.tag_, op.name_.c_str(),
                                           op.ext_.first, len, 0,
                                           ctp::ipc::ShmPtr<>(op.buf_.shm_),
                                           clio::run::PoolQuery::Local());
      }
      for (size_t i = first; i < next; ++i) {
        Op &op = ops[i];
        if (op.rc_ != 0) continue;
        CLIO_CO_AWAIT(op.get_);
        op.rc_ = op.get_->GetReturnCode();
        if (op.rc_ != 0) continue;
        op.put_ = pers_client_.AsyncPutBlob(
            op.tag_, op.name_, op.ext_.first, op.ext_.second - op.ext_.first,
            ctp::ipc::ShmPtr<>(op.buf_.shm_), -1.0f, Context(), 0,
            clio::run::PoolQuery::Local());
      }
      for (size_t i = first; i < next; ++i) {
        Op &op = ops[i];
        if (op.rc_ == 0 && !op.buf_.IsNull()) {
          CLIO_CO_AWAIT(op.put_);
          op.rc_ = op.put_->GetReturnCode();
        }
        if (!op.buf_.IsNull()) ipc->FreeBuffer(op.buf_);
        if (op.rc_ != 0 && key_rc[op.key_] == 0) key_rc[op.key_] = op.rc_;
      }
    }
    std::lock_guard<std::mutex> g(mu_);
    for (size_t k = 0; k < keys.size(); ++k) {
      if (key_rc[k] != 0) {
        if (*rc == 0) *rc = key_rc[k];
        continue;  // its ranges stay dirty for the next flush
      }
      auto it = entries_.find(keys[k]);
      if (it != entries_.end()) it->second.dirty_.clear();
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

clio::run::TaskResume CacheBase::FlushAll(TagId tag, bool all_tags,
                                          clio::run::u32 *rc) {
  CLIO_TASK_BODY_BEGIN
  *rc = 0;
  if (IsWriteBack()) {
    // Batches of blobs in key order: every batch takes its locks in the one
    // global order, flushes all its blobs at once, and releases them.
    std::vector<std::pair<std::string, clio::run::u64>> dirty =
        DirtyKeysOf(all_tags ? nullptr : &tag);
    size_t i = 0;
    while (i < dirty.size()) {
      std::vector<std::string> batch;
      clio::run::u64 bytes = 0;
      while (i < dirty.size() && batch.size() < kFlushBatchBlobs &&
             (batch.empty() || bytes + dirty[i].second <= kFlushBatchBytes)) {
        bytes += dirty[i].second;
        batch.push_back(dirty[i].first);
        ++i;
      }
      for (size_t k = 0; k < batch.size(); ++k) CLIO_CO_AWAIT(Lock(batch[k]));
      clio::run::u32 r = 0;
      CLIO_CO_AWAIT(FlushBatch(batch, &r));
      for (size_t k = 0; k < batch.size(); ++k) Unlock(batch[k]);
      if (r != 0 && *rc == 0) *rc = r;
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume CacheBase::JoinRound(std::string id, bool *leader,
                                           clio::run::u32 *rc,
                                           SyncOut *out) {
  CLIO_TASK_BODY_BEGIN
  *leader = false;
  *rc = 0;
  {
    clio::run::u64 target = 0;
    {
      std::lock_guard<std::mutex> g(mu_);
      Round &r = rounds_[id];
      target = r.started_ + 1;  // the first round to start after this call
      ++r.users_;
    }
    while (true) {
      {
        std::lock_guard<std::mutex> g(mu_);
        Round &r = rounds_[id];
        if (r.done_ >= target) {
          *rc = r.rc_;
          if (out != nullptr) *out = r.out_;
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
                           const SyncOut *out) {
  std::lock_guard<std::mutex> g(mu_);
  auto it = rounds_.find(id);
  if (it == rounds_.end()) return;
  Round &r = it->second;
  r.done_ = r.started_;
  r.rc_ = rc;
  if (out != nullptr) r.out_ = *out;
  r.running_ = false;
  if (--r.users_ == 0) rounds_.erase(it);
}

clio::run::TaskResume CacheBase::FlushTagShared(TagId tag,
                                                clio::run::u32 *rc) {
  CLIO_TASK_BODY_BEGIN
  {
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

clio::run::TaskResume CacheBase::PutBlob(
    clio::run::shared_ptr<core::PutBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    if (!HasPersistent()) {
      CLIO_CO_AWAIT(ForwardToCore(Method::kPutBlob,
                                  task.template Cast<clio::run::Task>()));
      CLIO_CO_RETURN;
    }
    const std::string name = task->blob_name_.str();
    const clio::run::u32 owner = OwnerOf(task->tag_id_, name);
    if (owner != container_id_) {
      CLIO_CO_AWAIT(PutAtOwner(task, owner));
      CLIO_CO_RETURN;
    }
    const std::string key = Key(task->tag_id_, name);
    const bool special = IsSpecial(task->flags_, task->context_);
    clio::run::u64 off = ~0ull;
    clio::run::u64 end = 0;
    core::ForEachBlobRegion(*task, [&off, &end](const core::BlobRegion &r) {
      off = std::min(off, r.blob_off_);
      end = std::max(end, r.blob_off_ + r.size_);
      return true;
    });
    if (off == ~0ull) off = 0;
    CLIO_CO_AWAIT(Lock(key));
    clio::run::u32 rc = 0;
    if (special || !IsWriteBack()) {
      // Persistent decides (special) or persistent first (write-through).
      if (special) CLIO_CO_AWAIT(Drop(key, true, &rc));
      if (rc == 0) {
        CLIO_CO_AWAIT(ForwardToCore(Method::kPutBlob,
                                    task.template Cast<clio::run::Task>()));
        rc = task->GetReturnCode();
        if (!special && rc == 0) {
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
      Unlock(key);
      CLIO_CO_RETURN;
    }
    // Write-back: absorb into the volatile copy and log the range dirty.
    bool bypass = false;
    CLIO_CO_AWAIT(MakeRoom(key, end - off, &bypass));
    if (bypass) {
      // Volatile is full of dirty copies and holds none of this blob: the
      // put would be refused and written through anyway.
      CLIO_CO_AWAIT(ForwardToCore(Method::kPutBlob,
                                  task.template Cast<clio::run::Task>()));
      Unlock(key);
      CLIO_CO_AWAIT(CheckFill());
      CLIO_CO_RETURN;
    }
    CLIO_CO_AWAIT(PrepareAbsorb(task->tag_id_, name, off, end, &rc));
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
        CLIO_CO_AWAIT(ForwardToCore(Method::kPutBlob,
                                    task.template Cast<clio::run::Task>()));
        rc = task->GetReturnCode();
      }
    } else {
      // No fallback: the put fails whole, leaving nothing behind.
      if (!Cached(key)) CLIO_CO_AWAIT(DropVolatile(task->tag_id_, name));
      rc = core::kPutNoSpaceRc;
    }
    task->SetReturnCode(rc);
    Unlock(key);
    CLIO_CO_AWAIT(CheckFill());
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
    bool all_here = true;
    for (size_t i = 0; HasPersistent() && attached && i < batch.size(); ++i) {
      all_here = all_here && OwnerOf(batch.descs_[i].tag_id_,
                                     batch.descs_[i].blob_name_) == container_id_;
    }
    if (!all_here) {
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
      CLIO_CO_RETURN;
    }
    if (!HasPersistent() || !attached || !IsWriteBack()) {
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
      CLIO_CO_RETURN;
    }
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
      CLIO_CO_AWAIT(Lock(key));
      clio::run::u32 rc = 0;
      const bool special = IsSpecial(d.flags_, task->context_);
      bool bypass = false;
      if (!special) CLIO_CO_AWAIT(MakeRoom(key, d.size_, &bypass));
      if (bypass) {
        // Volatile is full of dirty copies and holds none of this blob.
        auto put = pers_client_.AsyncPutBlob(
            d.tag_id_, d.blob_name_, d.offset_, d.size_, src, -1.0f,
            task->context_, d.flags_, clio::run::PoolQuery::Local());
        CLIO_CO_AWAIT(put);
        rc = put->GetReturnCode();
      } else if (!special) {
        CLIO_CO_AWAIT(PrepareAbsorb(d.tag_id_, d.blob_name_, d.offset_,
                                    d.offset_ + d.size_, &rc));
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
          auto put = pers_client_.AsyncPutBlob(
              d.tag_id_, d.blob_name_, d.offset_, d.size_, src, -1.0f,
              task->context_, d.flags_, clio::run::PoolQuery::Local());
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
    CLIO_CO_AWAIT(CheckFill());
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
      task->SetReturnCode(f->GetReturnCode());
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
    if (rc == 0 && IsWriteBack() && core::PoolContainers(pool_id_) > 1) {
      CLIO_CO_AWAIT(TagCopiesEverywhere(task->tag_id_, kCopiesFlush, &rc));
      ev::Span(ev::kWbSyncAll, t1, ev::Now());
    }
    if (rc != 0) {
      task->SetReturnCode(IsNoSpaceRc(rc) ? core::kSyncNoSpaceRc
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
    const std::string id = "flushall";
    bool leader = false;
    clio::run::u32 rc = 0;
    CLIO_CO_AWAIT(JoinRound(id, &leader, &rc));
    if (leader) {
      const clio::run::u64 gen = dirty_gen_.load(std::memory_order_relaxed);
      CLIO_CO_AWAIT(FlushAll(TagId::GetNull(), true, &rc));
      if (rc == 0) flushed_gen_.store(gen, std::memory_order_relaxed);
      LeaveRound(id, rc);
    }
    fill_flush_pending_.store(false, std::memory_order_relaxed);
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
