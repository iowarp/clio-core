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
 * core_shm_cache_writethru runtime. See ../DESIGN.md.
 *
 * Data verbs route submitter-local: the segment is per node, so the node
 * that issued an op is the one whose cache it reads or fills. Everything the
 * cache does not serve goes down `next_pool_id` through a core client
 * (Dynamic routing) after the blob's (or the log's) pending ranges landed.
 */

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <clio_cte/core/blob_batch.h>
#include <clio_cte/core/blob_placement.h>
#include <clio_cte/core_shm_cache_writethru/core_shm_cache_writethru_runtime.h>

namespace clio::cte::shm_wtc {

/** Wall-clock ms (CLOCK_REALTIME) for cross-node trace ordering. */
static double WallMs() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

namespace wtc = clio::cte::core::wtc;

/** Poll interval for barrier waits (us). */
static constexpr double kBarrierPollUs = 10.0;

// ===========================================================================
// Lifecycle
// ===========================================================================

clio::run::TaskResume Runtime::Create(clio::run::shared_ptr<CreateTask> &task) {
  CLIO_TASK_BODY_BEGIN
  config_ = task->GetParams();
  interposer_next_pool_ = config_.next_pool_id_;
  next_client_ = std::make_unique<clio::cte::core::Client>(CorePoolId());
  if (OpenSegment()) {
    // The log's only consumer: a periodic flusher on this container.
    auto *ipc = CLIO_IPC;
    auto flush = ipc->NewTask<FlushLogTask>(clio::run::CreateTaskId(),
                                            pool_id_,
                                            clio::run::PoolQuery::Local());
    flush->SetPeriod(config_.flush_period_us_, clio::run::kMicro);
    flush->SetFlags(TASK_PERIODIC);
    ipc->Send(flush);
    HLOG(kInfo,
         "core_shm_cache_writethru {}.{}: segment {} MB, log {} ranges, next {}.{}",
         pool_id_.major_, pool_id_.minor_, config_.capacity_ >> 20,
         config_.log_depth_, CorePoolId().major_, CorePoolId().minor_);
  } else {
    HLOG(kWarning,
         "core_shm_cache_writethru {}.{}: no segment; serving tasks only",
         pool_id_.major_, pool_id_.minor_);
  }
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

bool Runtime::OpenSegment() {
  auto *ipc = CLIO_IPC;
  auto *meta = ipc->GetMetadataAllocator();
  auto *dir = ipc->GetMetadataDirectory();
  const clio::run::u32 pid = static_cast<clio::run::u32>(ipc->GetRuntimePid());
  wtc::Limits lim;
  lim.seg_size_ = config_.capacity_;
  lim.max_blobs_ = config_.max_blobs_;
  lim.max_tags_ = config_.max_tags_;
  lim.log_depth_ = config_.log_depth_;
  lim.evict_sample_ = config_.evict_sample_;
  lim.max_fill_ = config_.max_fill_;
  lim.half_life_ns_ = config_.half_life_ms_ * 1000ull * 1000ull;
  lim.md_only_ = config_.md_only_;
  const std::string name = wtc::SegmentName(pid, pool_id_);
  const ctp::ipc::MemoryBackendId id(pid, 0x57540000u | (pool_id_.major_ & 0xFFFFu));
  if (!view_.Format(name, id, lim, SelfNode())) {
    view_.Detach();
    return false;
  }
  if (meta == nullptr || dir == nullptr) {
    return true;  // task callers still get the cache; clients cannot attach
  }
  auto loc_fp = meta->template Allocate<wtc::Locator>(sizeof(wtc::Locator));
  if (loc_fp.IsNull()) {
    return true;
  }
  wtc::Locator *loc = loc_fp.ptr_;
  std::memset(static_cast<void *>(loc), 0, sizeof(wtc::Locator));
  loc->magic_ = wtc::Locator::kMagic;
  loc->runtime_pid_ = pid;
  loc->root_off_ = view_.RootOff();
  loc->seg_size_ = view_.SegSize();
  loc->entry_pool_ = config_.entry_pool_id_.IsNull()
                         ? 0
                         : config_.entry_pool_id_.ToU64();
  loc->ready_.store(1, std::memory_order_release);
  const clio::run::u64 off = static_cast<clio::run::u64>(
      reinterpret_cast<char *>(loc) - reinterpret_cast<char *>(meta));
  if (!dir->RegisterRoot(pool_id_.ToU64(), off)) {
    HLOG(kWarning, "core_shm_cache_writethru: metadata directory full");
  }
  return true;
}

clio::run::TaskResume Runtime::Destroy(clio::run::shared_ptr<DestroyTask> &task) {
  CLIO_TASK_BODY_BEGIN
  view_.Detach();
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Monitor(clio::run::shared_ptr<MonitorTask> &task) {
  CLIO_TASK_BODY_BEGIN
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

// ===========================================================================
// Helpers
// ===========================================================================

clio::cte::core::Client *Runtime::Next() {
  if (!next_client_) {
    next_client_ = std::make_unique<clio::cte::core::Client>(CorePoolId());
  }
  return next_client_.get();
}

clio::run::u32 Runtime::SelfNode() const {
  return static_cast<clio::run::u32>(CLIO_IPC->GetNodeId());
}

clio::run::u32 Runtime::HomeOf(const TagId &tag, const std::string &name) const {
  auto *pool_manager = CLIO_POOL_MANAGER;
  const clio::run::PoolInfo *info = pool_manager->GetPoolInfo(pool_id_);
  if (info == nullptr || info->num_containers_ <= 1) {
    return container_id_;
  }
  // Same hash as the core's owner routing (BlobHash), over THIS pool's
  // containers: every chain pool composes one container per node.
  return static_cast<clio::run::u32>(clio::cte::core::BlobHash(tag, name) %
                                     info->num_containers_);
}

clio::run::PoolQuery Runtime::ScheduleTask(
    const clio::run::shared_ptr<clio::run::Task> &task) {
  switch (task->method_) {
    case Method::kPutBlob:
    case Method::kGetBlob:
    case Method::kGetBlobSize:
    case Method::kMultiPutBlob:
    case Method::kGetOrCreateTag:
    case Method::kDelTag:
    case Method::kDelBlob:
    case Method::kFlushLog:
      return clio::run::PoolQuery::Local();
    default:
      return CoreInterposer::ScheduleTask(task);
  }
}

// ===========================================================================
// Tags
// ===========================================================================

clio::run::TaskResume Runtime::GetOrCreateTag(
    clio::run::shared_ptr<clio::cte::core::GetOrCreateTagTask<
        clio::cte::core::CreateParams>> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    const std::string name = task->tag_name_.str();
    // Only a plain open-by-name is memoized: no fault handler, no id given.
    const bool plain = view_.Ready() && task->tag_id_.IsNull() &&
                       task->fault_pool_id_.IsNull() &&
                       task->fault_params_.str().empty() && !name.empty() &&
                       name.size() <= wtc::kMaxTagName;
    clio::run::u32 maj = 0;
    clio::run::u32 min = 0;
    wtc::TagAttrs a;
    if (plain && view_.LookupTagByName(name.data(), name.size(), &maj, &min) &&
        view_.LookupTagAttrs(maj, min, &a)) {
      task->tag_id_ = TagId(maj, min);
      task->created_ = 0;
      task->tag_size_ = a.size_;
      task->SetReturnCode(0);
      CLIO_CO_RETURN;
    }
    const clio::run::u64 seq = plain ? view_.TagInvalSeq() : 0;
    auto f = Next()->AsyncGetOrCreateTag(
        name, task->fault_pool_id_, task->fault_pool_name_.str(),
        task->fault_params_.str(), task->tag_id_,
        clio::run::PoolQuery::Dynamic());
    CLIO_CO_AWAIT(f);
    task->tag_id_ = f->tag_id_;
    task->created_ = f->created_;
    task->tag_size_ = f->tag_size_;
    if (f->GetReturnCode() == 0 && f->created_ != 0 && view_.Ready()) {
      view_.NoteTagCreated(task->tag_id_.major_, task->tag_id_.minor_);
    }
    task->SetReturnCode(f->GetReturnCode());
    if (plain && f->GetReturnCode() == 0 && !f->tag_id_.IsNull()) {
      // Register with the tag's home, THEN confirm the binding: a rename
      // that landed before the registration reached the home would never
      // invalidate us, but it does change what this second lookup returns.
      const TagId id = f->tag_id_;
      CLIO_CO_AWAIT(RegisterSharer(
          id, std::string(wtc::kTagMetaName, wtc::kTagMetaNameLen)));
      auto again = Next()->AsyncGetOrCreateTag(name, TagId::GetNull(),
                                               clio::run::PoolQuery::Dynamic());
      CLIO_CO_AWAIT(again);
      if (again->GetReturnCode() == 0 && again->tag_id_ == id) {
        view_.FillTagName(id.major_, id.minor_, name.data(), name.size(), seq);
      }
      // tag_size_ is the tag's TOTAL bytes (the field's contract). The core
      // answers from one container's TagInfo -- only that container's share
      // when blobs spread over nodes -- so a hit (the cached total) and a
      // miss would disagree. Report the total either way, and cache it.
      clio::run::u32 rc = 0;
      CLIO_CO_AWAIT(FetchTagAttrs(id, &a, &rc));
      if (rc == 0) task->tag_size_ = a.size_;
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::DelTag(
    clio::run::shared_ptr<clio::cte::core::DelTagTask> &task) {
  CLIO_TASK_BODY_BEGIN
  if (view_.Ready()) {
    const clio::run::u64 target = view_.PushedNow();
    while (!view_.FlushedThrough(target)) {
      CLIO_CO_AWAIT(clio::run::yield(kBarrierPollUs));
    }
  }
  CLIO_CO_AWAIT(BeginChainWriteAll());
  {
    const std::string name = task->tag_name_.str();
    clio::run::Future<clio::cte::core::DelTagTask> f =
        name.empty() ? Next()->AsyncDelTag(task->tag_id_)
                     : Next()->AsyncDelTag(name, clio::run::PoolQuery::Dynamic(),
                                           task->posix_unlink_ != 0);
    CLIO_CO_AWAIT(f);
    if (!f->tag_id_.IsNull()) {
      task->tag_id_ = f->tag_id_;
    }
    task->SetReturnCode(f->GetReturnCode());
  }
  EndChainWriteAll();
  if (!task->tag_id_.IsNull()) {
    // Every node drops its copies of the tag's blobs (and its own, here).
    BlobKey k{task->tag_id_.major_, task->tag_id_.minor_, std::string()};
    auto *ipc = CLIO_IPC;
    if (ipc->GetNumHosts() > 1) {
      auto inv = ipc->NewTask<InvalidateTask>(
          clio::run::CreateTaskId(), pool_id_, clio::run::PoolQuery::Broadcast(),
          kInvalidateDropTag, container_id_, PackKeys({k}));
      auto fut = ipc->Send(inv);
      CLIO_CO_AWAIT(fut);
    } else if (view_.Ready()) {
      view_.DropTagBlobs(k.major_, k.minor_);
      view_.DropTag(k.major_, k.minor_);
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::SyncTag(
    clio::run::shared_ptr<clio::cte::core::SyncTagTask> &task) {
  CLIO_TASK_BODY_BEGIN
  clio::run::u32 sticky = 0;
  std::set<clio::run::u32> homes;
  if (view_.Ready()) {
    const clio::run::u64 target = view_.PushedNow();
    while (!view_.FlushedThrough(target)) {
      CLIO_CO_AWAIT(clio::run::yield(kBarrierPollUs));
    }
    sticky = view_.TakeTagError(task->tag_id_.major_, task->tag_id_.minor_);
    homes = TakeSyncHomes(task->tag_id_);
    const char *tr = std::getenv("CLIO_WTC_TRACE");
    if (tr != nullptr && tr[0] == '1') {
      HLOG(kInfo, "[WTC-TRACE] t={} synctag at={} tag={}.{} target={} flushed={}", WallMs(),
           container_id_, task->tag_id_.major_, task->tag_id_.minor_, target,
           view_.GetRoot()->flushed_.load());
    }
  }
  // This node's flushes landed on remote homes, which wrote them into
  // THEIR next pool: sync the tag there as well. The broadcast arm on a
  // home may have run before these ranges arrived, so it cannot cover them.
  {
    std::vector<clio::run::Future<clio::cte::core::SyncTagTask>> futs;
    for (clio::run::u32 h : homes) {
      futs.push_back(next_client_->AsyncSyncTag(
          task->tag_id_, clio::run::PoolQuery::DirectId(h),
          task->min_persistence_));
    }
    for (auto &f : futs) {
      CLIO_CO_AWAIT(f);
      clio::run::u32 rc = f.get() == nullptr ? 1 : f->GetReturnCode();
      if (rc != 0 && sticky == 0) sticky = rc;
    }
  }
  CLIO_CO_AWAIT(ForwardToCore(Method::kSyncTag,
                              task.template Cast<clio::run::Task>()));
  if (sticky != 0 && task->GetReturnCode() == 0) {
    task->SetReturnCode(sticky);  // an earlier buffered write failed
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::BarrierForward(
    clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> task) {
  CLIO_TASK_BODY_BEGIN
  if (view_.Ready()) {
    const clio::run::u64 target = view_.PushedNow();
    while (!view_.FlushedThrough(target)) {
      CLIO_CO_AWAIT(clio::run::yield(kBarrierPollUs));
    }
  }
  CLIO_CO_AWAIT(ForwardToCore(method, task));
  CLIO_CO_AWAIT(InvalidateAfterForward(method, task));
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::TagAttrs(
    clio::run::shared_ptr<clio::cte::core::GetTagSizeTask> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    const TagId tag = task->tag_id_;
    wtc::TagAttrs a;
    if (view_.Ready() && view_.LookupTagAttrs(tag.major_, tag.minor_, &a)) {
      task->tag_size_ = a.size_;
      task->ctime_ = a.ctime_;
      task->mtime_ = a.mtime_;
      task->atime_ = a.atime_;
      task->SetReturnCode(0);
      CLIO_CO_RETURN;
    }
    clio::run::u32 rc = 0;
    CLIO_CO_AWAIT(FetchTagAttrs(tag, &a, &rc));
    task->tag_size_ = a.size_;
    task->ctime_ = a.ctime_;
    task->mtime_ = a.mtime_;
    task->atime_ = a.atime_;
    task->SetReturnCode(rc);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::FetchTagAttrs(const TagId &tag,
                                             wtc::TagAttrs *a,
                                             clio::run::u32 *rc) {
  CLIO_TASK_BODY_BEGIN
  {
    // Snapshot BEFORE draining and registering: a drop or a local write
    // after this point voids the fill below. Draining puts every write
    // buffered here before the snapshot into the chain's total.
    const clio::run::u64 seq = view_.Ready() ? view_.TagInvalSeq() : 0;
    const clio::run::u64 wseq = view_.Ready() ? view_.LocalWriteSeq() : 0;
    if (view_.Ready()) {
      const clio::run::u64 target = view_.PushedNow();
      while (!view_.FlushedThrough(target)) {
        CLIO_CO_AWAIT(clio::run::yield(kBarrierPollUs));
      }
      CLIO_CO_AWAIT(RegisterSharer(
          tag, std::string(wtc::kTagMetaName, wtc::kTagMetaNameLen)));
    }
    auto f = Next()->AsyncGetTagSize(tag, clio::run::PoolQuery::Broadcast(0.0f));
    CLIO_CO_AWAIT(f);
    a->size_ = f->tag_size_;
    a->ctime_ = f->ctime_;
    a->mtime_ = f->mtime_;
    a->atime_ = f->atime_;
    *rc = f->GetReturnCode();
    if (view_.Ready() && *rc == 0) {
      view_.FillTagAttrs(tag.major_, tag.minor_, *a, seq, wseq);
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::NoteTagsWritten(const std::vector<TagId> &tags,
                                               bool count_local) {
  CLIO_TASK_BODY_BEGIN
  if (view_.Ready() && !tags.empty()) {
    // A landing flush batch passes count_local=false: its writes were
    // counted when they were absorbed, and counting them again would void
    // every fill that drained the log (the drain lands this very batch).
    if (count_local) view_.NoteLocalWrite();
    std::set<std::pair<clio::run::u32, clio::run::u32>> seen;
    for (const TagId &t : tags) {
      if (t.IsNull() || !seen.insert({t.major_, t.minor_}).second) continue;
      CLIO_CO_AWAIT(FanoutInvalidate(
          t, std::string(wtc::kTagMetaName, wtc::kTagMetaNameLen)));
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::DropTagDataEverywhere(
    const std::vector<TagId> &tags) {
  CLIO_TASK_BODY_BEGIN
  if (!tags.empty() && view_.Ready()) {
    std::vector<BlobKey> keys;
    for (const TagId &t : tags) {
      if (t.IsNull()) continue;
      keys.push_back(BlobKey{t.major_, t.minor_, std::string()});
      view_.DropTagBlobs(t.major_, t.minor_);
      view_.DropTagMeta(t.major_, t.minor_);
    }
    auto *ipc = CLIO_IPC;
    if (!keys.empty() && ipc->GetNumHosts() > 1) {
      auto inv = ipc->NewTask<InvalidateTask>(
          clio::run::CreateTaskId(), pool_id_, clio::run::PoolQuery::Broadcast(),
          kInvalidateDropTagData, container_id_, PackKeys(keys));
      auto fut = ipc->Send(inv);
      CLIO_CO_AWAIT(fut);
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::InvalidateAfterForward(
    clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> task) {
  CLIO_TASK_BODY_BEGIN
  {
    namespace core = clio::cte::core;
    std::vector<BlobKey> blobs;   // single blobs: local drop + their homes
    std::vector<TagId> tag_wide;  // every node drops the tag's entries
    std::vector<TagId> tag_meta;  // tag metadata (names) at the tag's home
    switch (method) {
      case core::Method::kTruncateBlob: {
        auto t = task.template Cast<core::TruncateBlobTask>();
        blobs.push_back(BlobKey{t->tag_id_.major_, t->tag_id_.minor_,
                                t->blob_name_.str()});
        break;
      }
      case core::Method::kMultiDelBlob: {
        auto t = task.template Cast<core::MultiDelBlobTask>();
        for (const core::MultiDelKey &k :
             core::DecodeMultiDelKeys(t->keys_)) {
          blobs.push_back(BlobKey{k.tag_id_.major_, k.tag_id_.minor_,
                                  k.blob_name_});
        }
        break;
      }
      case core::Method::kMultiDelTag: {
        auto t = task.template Cast<core::MultiDelTagTask>();
        tag_wide = core::DecodeMultiDelTags(t->tags_);
        break;
      }
      case core::Method::kPodPutBlob:
        tag_wide.push_back(task.template Cast<core::PodPutBlobTask>()->tag_id_);
        break;
      case core::Method::kPodMultiPutBlob:
        tag_wide.push_back(
            task.template Cast<core::PodMultiPutBlobTask>()->tag_id_);
        break;
      case core::Method::kRenameTag:
        tag_meta.push_back(task.template Cast<core::RenameTagTask>()->tag_id_);
        break;
      case core::Method::kGetOrCreateTagAlias:
        tag_meta.push_back(
            task.template Cast<core::GetOrCreateTagAliasTask>()->tag_id_);
        break;
      case core::Method::kUpdateTagNames: {
        auto t = task.template Cast<core::UpdateTagNamesTask>();
        std::vector<core::TagNameOpRec> ops;
        const std::string buf = t->ops_.str();
        core::DecodeTagNameOps(buf.data(), buf.size(), &ops);
        for (const auto &r : ops) tag_meta.push_back(r.tag_);
        break;
      }
      default:
        break;
    }
    if (view_.Ready()) {
      for (const BlobKey &k : blobs) {
        view_.Invalidate(k.major_, k.minor_, k.name_.c_str(), k.name_.size());
      }
      for (const TagId &t : tag_meta) {
        if (!t.IsNull()) view_.DropTagMeta(t.major_, t.minor_);
      }
    }
    for (const BlobKey &k : blobs) {
      CLIO_CO_AWAIT(FanoutInvalidate(TagId(k.major_, k.minor_), k.name_));
    }
    for (const TagId &t : tag_meta) {
      if (t.IsNull()) continue;
      CLIO_CO_AWAIT(FanoutInvalidate(
          t, std::string(wtc::kTagMetaName, wtc::kTagMetaNameLen)));
    }
    CLIO_CO_AWAIT(DropTagDataEverywhere(tag_wide));
    {
      std::vector<TagId> written;
      for (const BlobKey &k : blobs) written.push_back(TagId(k.major_, k.minor_));
      CLIO_CO_AWAIT(NoteTagsWritten(written));
      if (!tag_wide.empty() && view_.Ready()) view_.NoteLocalWrite();
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

}  // namespace clio::cte::shm_wtc

// Module entry point.
CLIO_TASK_CC(clio::cte::shm_wtc::Runtime)
