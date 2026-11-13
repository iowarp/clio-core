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
 * Policy cache base: map state, lifecycle, and the verbs that only need
 * whole-copy operations (delete, truncate, batch puts, sync, flush, and the
 * coherent default forward). The put/get/flush/evict machinery lives in
 * cache_base_io.cc.
 */

#include <clio_cte/cache_base/cache_base.h>

#include <algorithm>

namespace clio::cte::cache_base {

namespace Method = clio::cte::core::Method;

CacheBase::~CacheBase() {
  if (stop_hook_id_ != 0) {
    auto *manager = CLIO_RUNTIME_MANAGER;
    if (manager != nullptr) manager->RemoveStopHook(stop_hook_id_);
  }
}

std::string CacheBase::Key(const TagId &tag, const std::string &name) {
  return std::to_string(tag.major_) + "." + std::to_string(tag.minor_) +
         "." + name;
}

void CacheBase::InitCache(const PolicyCacheParams &params,
                          const clio::run::PoolId &self_pool) {
  params_ = params;
  SetTargets(params.volatile_pool_, params.persistent_pool_);
  vol_client_.Init(params.volatile_pool_);
  pers_client_.Init(CorePoolId());
  // Zero-IPC reads through this pool use the VOLATILE core's mirror: its
  // records are exactly the copies this cache keeps (current by
  // construction), and a blob it does not hold misses there and takes the
  // RPC path here. Without this a client bound to an interposer falls back
  // to the canonical core's mirror, which is stale for dirty copies.
  auto *ipc = CLIO_IPC;
  auto *dir = ipc != nullptr ? ipc->GetMetadataDirectory() : nullptr;
  if (dir != nullptr) {
    const clio::run::u64 root = dir->FindRoot(params.volatile_pool_.ToU64());
    if (root != 0) dir->RegisterRoot(self_pool.ToU64(), root);
  }
  if (IsWriteBack() && HasPersistent()) {
    if (params.flush_period_ms_ > 0) {
      clio::cte::core::Client self(self_pool);
      self.AsyncFlushData(clio::run::PoolQuery::Local(), 0,
                          params.flush_period_ms_ * 1000.0);
    }
    RegisterStopFlush(self_pool);
  }
  HLOG(kInfo,
       "policy cache {}: {} over volatile {}.{} and {} (fill={}, evict={}, "
       "fallback_to_persistent={})",
       self_pool, IsWriteBack() ? "write-back" : "write-through",
       params.volatile_pool_.major_, params.volatile_pool_.minor_,
       HasPersistent() ? "a persistent chain" : "no persistent chain",
       params.fill_mode_, params.evict_mode_, params.fallback_to_persistent_);
}

void CacheBase::RegisterStopFlush(const clio::run::PoolId &self_pool) {
  auto *manager = CLIO_RUNTIME_MANAGER;
  if (manager == nullptr) return;
  stop_hook_id_ = manager->AddStopHook([self_pool]() {
    // Replaces the core's old stop-time move: dirty copies are the only
    // bytes a graceful stop could lose.
    auto *mgr = CLIO_RUNTIME_MANAGER;
    const float max_s =
        static_cast<float>(mgr->GetStopGracePeriodMs()) / 1000.0f;
    clio::cte::core::Client client(self_pool);
    auto flush = client.AsyncFlushData(clio::run::PoolQuery::Local(), 0);
    if (!flush.Wait(max_s)) {
      HLOG(kWarning,
           "write-back cache stop: flushing dirty copies did not finish "
           "within the {} s grace period; unflushed data is lost (raise "
           "clio_run stop --grace-period)",
           max_s);
    }
  });
}

clio::run::PoolQuery CacheBase::ScheduleTask(
    const clio::run::shared_ptr<clio::run::Task> &task) {
  switch (task->method_) {
    case Method::kFlushData:
    case kDropTagCopies:
      // Per-container work: keep it where it was sent (the periodic flush
      // is Local; DelTag broadcasts kDropTagCopies itself).
      return task->pool_query_;
    default:
      return CoreInterposer::ScheduleTask(task);
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
    task->results_[container_id_] = std::to_string(DirtyCount());
    task->SetReturnCode(0);
    CLIO_CO_RETURN;
  }
  // Every other query describes the chain below (e.g. replication's
  // "replicate_period_ms"): answer it as the persistent side would.
  CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kMonitor,
                              task.template Cast<clio::run::Task>()));
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

// ===========================================================================
// Map state. mu_ is held only inside these helpers, never across a
// co_await.
// ===========================================================================

PinState CacheBase::Pin(const std::string &key) {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = entries_.find(key);
  if (it == entries_.end()) return PinState::kAbsent;
  if (it->second.busy_) return PinState::kBusy;
  it->second.pins_++;
  it->second.hits_++;
  it->second.last_access_ns_ = clio::cte::core::GetCurrentTimeNs();
  return PinState::kCached;
}

void CacheBase::Unpin(const std::string &key) {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = entries_.find(key);
  if (it != entries_.end() && it->second.pins_ > 0) it->second.pins_--;
}

void CacheBase::NoteWrite(const std::string &key, const TagId &tag,
                          const std::string &name, clio::run::u64 end,
                          bool dirty) {
  std::lock_guard<std::mutex> lock(mu_);
  CacheEntry &e = entries_[key];
  e.tag_ = tag;
  e.name_ = name;
  e.size_ = std::max(e.size_, end);
  e.gen_++;
  e.dirty_ = e.dirty_ || dirty;
  e.last_access_ns_ = clio::cte::core::GetCurrentTimeNs();
}

bool CacheBase::Reserve(const std::string &key, const TagId &tag,
                        const std::string &name) {
  std::lock_guard<std::mutex> lock(mu_);
  if (entries_.count(key) != 0) return false;
  CacheEntry &e = entries_[key];
  e.tag_ = tag;
  e.name_ = name;
  e.busy_ = true;
  e.last_access_ns_ = clio::cte::core::GetCurrentTimeNs();
  return true;
}

void CacheBase::ReleaseReserve(const std::string &key, bool written,
                               clio::run::u64 size, bool dirty) {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = entries_.find(key);
  if (it == entries_.end()) return;
  if (!written) {
    entries_.erase(it);
    return;
  }
  it->second.busy_ = false;
  it->second.size_ = size;
  it->second.dirty_ = dirty;
  it->second.gen_++;
}

bool CacheBase::MarkBusy(const std::string &key) {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = entries_.find(key);
  if (it == entries_.end() || it->second.busy_ || it->second.pins_ > 0) {
    return false;
  }
  it->second.busy_ = true;
  return true;
}

void CacheBase::Erase(const std::string &key) {
  std::lock_guard<std::mutex> lock(mu_);
  entries_.erase(key);
}

std::vector<std::string> CacheBase::Collect(const KeyFilter &filter,
                                            bool dirty_only) {
  std::vector<std::string> keys;
  std::lock_guard<std::mutex> lock(mu_);
  for (const auto &kv : entries_) {
    if (dirty_only && !kv.second.dirty_) continue;
    if (filter && !filter(kv.second)) continue;
    keys.push_back(kv.first);
  }
  return keys;
}

size_t CacheBase::DirtyCount() {
  std::lock_guard<std::mutex> lock(mu_);
  size_t n = 0;
  for (const auto &kv : entries_) n += kv.second.dirty_ ? 1 : 0;
  return n;
}

bool CacheBase::ShouldFill(const std::string &key) {
  if (!HasPersistent()) return false;
  switch (params_.Fill()) {
    case FillMode::kOnReadMiss:
      return true;
    case FillMode::kNone:
      return false;
    case FillMode::kOnHitCount: {
      std::lock_guard<std::mutex> lock(mu_);
      // Bounded: a one-pass scan of a huge store must not grow this forever.
      if (miss_counts_.size() > (1u << 20)) miss_counts_.clear();
      clio::run::u32 &n = miss_counts_[key];
      if (++n < params_.fill_hits_) return false;
      miss_counts_.erase(key);
      return true;
    }
  }
  return false;
}

// ===========================================================================
// Whole-copy verbs.
// ===========================================================================

clio::run::TaskResume CacheBase::DelBlob(
    clio::run::shared_ptr<clio::cte::core::DelBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    const TagId tag = task->tag_id_;
    const std::string name = task->blob_name_.str();
    const std::string key = Key(tag, name);
    bool had_copy = false;
    bool had_dirty = false;
    // Wait out a fill/evict in flight, then take the copy out of service.
    while (true) {
      {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = entries_.find(key);
        if (it == entries_.end()) break;
        if (!it->second.busy_ && it->second.pins_ == 0) {
          it->second.busy_ = true;
          had_copy = true;
          had_dirty = it->second.dirty_;
          break;
        }
      }
      CLIO_CO_AWAIT(clio::run::yield(50));
    }
    if (had_copy) {
      // Dropped without flushing: the blob is being deleted.
      CLIO_CO_AWAIT(DeleteVolatileCopy(tag, name));
      Erase(key);
    }
    if (HasPersistent()) {
      CLIO_CO_AWAIT(ForwardToCore(Method::kDelBlob,
                                  task.template Cast<clio::run::Task>()));
      // A dirty copy that never reached persistent existed only here.
      if (had_dirty && task->GetReturnCode() != 0) task->SetReturnCode(0);
    } else {
      task->SetReturnCode(had_copy ? 0 : 1);
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume CacheBase::DelTag(
    clio::run::shared_ptr<clio::cte::core::DelTagTask> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    // Persistent first (it resolves a name to the id), then every cache
    // container drops its copies of the tag: blob copies live on their
    // owners' nodes, so this is a broadcast of the module verb.
    CLIO_CO_AWAIT(ForwardToCore(Method::kDelTag,
                                task.template Cast<clio::run::Task>()));
    const clio::run::u32 rc = task->GetReturnCode();
    const TagId tag = task->tag_id_;
    if (!tag.IsNull()) {
      auto *ipc = CLIO_IPC;
      auto drop = ipc->NewTask<clio::cte::core::DelTagTask>(
          clio::run::CreateTaskId(), pool_id_,
          clio::run::PoolQuery::Broadcast(), tag);
      drop->method_ = kDropTagCopies;
      auto fut = ipc->Send(drop);
      CLIO_CO_AWAIT(fut);
    }
    task->SetReturnCode(rc);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume CacheBase::DropTagCopies(
    clio::run::shared_ptr<clio::cte::core::DelTagTask> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    const TagId tag = task->tag_id_;
    std::vector<std::string> keys = Collect(
        [tag](const CacheEntry &e) { return e.tag_ == tag; }, false);
    CLIO_CO_AWAIT(DropKeys(keys, /*flush_first=*/false));
    task->SetReturnCode(0);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume CacheBase::TruncateBlob(
    clio::run::shared_ptr<clio::cte::core::TruncateBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    // Write the copy out (write-back), drop it, and truncate the durable
    // blob: the next read fills the truncated bytes back if wanted.
    const std::string key = Key(task->tag_id_, task->blob_name_.str());
    clio::run::u32 rc = 0;
    CLIO_CO_AWAIT(DropCopy(key, /*flush_first=*/IsWriteBack(), rc));
    if (rc != 0) {
      task->SetReturnCode(rc);
      CLIO_CO_RETURN;
    }
    CLIO_CO_AWAIT(ForwardToCore(Method::kTruncateBlob,
                                task.template Cast<clio::run::Task>()));
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume CacheBase::MultiPutBlob(
    clio::run::shared_ptr<clio::cte::core::MultiPutBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    // Batches write around the cache: every record's copy is written out
    // (write-back) and dropped, then the batch lands on persistent as one
    // task. Reads fill the copies back.
    std::vector<std::string> keys;
    for (const auto &d : clio::cte::core::DecodeMultiPutDescs(task->descs_)) {
      keys.push_back(Key(d.tag_id_, d.blob_name_));
    }
    CLIO_CO_AWAIT(DropKeys(keys, /*flush_first=*/IsWriteBack()));
    CLIO_CO_AWAIT(ForwardToCore(Method::kMultiPutBlob,
                                task.template Cast<clio::run::Task>()));
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume CacheBase::SyncTag(
    clio::run::shared_ptr<clio::cte::core::SyncTagTask> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    // fsync: this container's dirty copies of the tag reach persistent
    // first, then the persistent chain syncs its devices and WAL.
    if (IsWriteBack()) {
      const TagId tag = task->tag_id_;
      std::vector<std::string> keys = Collect(
          [tag](const CacheEntry &e) { return e.tag_ == tag; }, true);
      clio::run::u32 rc = 0;
      CLIO_CO_AWAIT(FlushKeys(keys, rc));
      if (rc != 0) {
        task->SetReturnCode(IsNoSpaceRc(rc) ? clio::cte::core::kSyncNoSpaceRc
                                            : clio::cte::core::kSyncIoRc);
        CLIO_CO_RETURN;
      }
      task->blobs_moved_ = keys.size();
    }
    CLIO_CO_AWAIT(ForwardToCore(Method::kSyncTag,
                                task.template Cast<clio::run::Task>()));
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume CacheBase::FlushData(
    clio::run::shared_ptr<clio::cte::core::FlushDataTask> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    clio::run::u64 flushed = 0;
    clio::run::u32 rc = 0;
    if (IsWriteBack() && HasPersistent()) {
      std::vector<std::string> keys = Collect(nullptr, true);
      flushed = keys.size();
      CLIO_CO_AWAIT(FlushKeys(keys, rc));
    }
    if (HasPersistent()) {
      // The persistent core then syncs its devices and WAL.
      CLIO_CO_AWAIT(ForwardToCore(Method::kFlushData,
                                  task.template Cast<clio::run::Task>()));
    }
    task->blobs_flushed_ = flushed;
    if (rc != 0) task->SetReturnCode(rc);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume CacheBase::ForwardCoherent(
    clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> task) {
  CLIO_TASK_BODY_BEGIN
  {
    namespace core = clio::cte::core;
    // Which copies this verb needs settled before persistent answers it.
    TagId tag = TagId::GetNull();
    bool any_tag = false;    // verb reads state across tags
    bool mutates = false;    // verb changes blob bytes behind the cache
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
    if (!tag.IsNull() || any_tag) {
      KeyFilter filter = nullptr;
      if (!any_tag) {
        filter = [tag](const CacheEntry &e) { return e.tag_ == tag; };
      }
      if (mutates) {
        // Device-side writes bypass the cache: settle and drop the tag's
        // copies so none can serve bytes older than the write.
        CLIO_CO_AWAIT(DropKeys(Collect(filter, false), IsWriteBack()));
      } else if (IsWriteBack()) {
        clio::run::u32 rc = 0;
        CLIO_CO_AWAIT(FlushKeys(Collect(filter, true), rc));
      }
    }
    CLIO_CO_AWAIT(ForwardToCore(method, task));
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

}  // namespace clio::cte::cache_base
