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
  // The task is NOT a CreateTask, whatever the parameter says: compose hands
  // this ChiMod a ComposeTask<PoolConfig>, a direct caller its own
  // instantiation. All of them ARE a CreatePoolFields, so every read and write
  // goes through that (see CreatePoolFields::GetParamsAs). Calling a member
  // through `task` is undefined behaviour, which UBSan reports.
  auto &fields = task.template Cast<clio::run::admin::CreatePoolFields>();
  config_ = fields->GetParamsAs<WtcConfig>();
  interposer_next_pool_ = config_.next_pool_id_;
  next_client_ = std::make_unique<clio::cte::core::Client>(CorePoolId());
  if (!config_.data_pool_.IsNull() && !config_.md_only_ &&
      config_.data_capacity_ > 0) {
    if (config_.data_pool_.minor_ == 0) {
      config_.data_pool_.minor_ = 1 + container_id_;  // this node's device
    }
    CLIO_CO_AWAIT(LeaseExtents());
  }
  if (OpenSegment()) {
    // The log's only consumer: a periodic flusher on this container.
    auto *ipc = CLIO_IPC;
    auto flush = ipc->NewTask<FlushLogTask>(clio::run::CreateTaskId(),
                                            pool_id_,
                                            clio::run::PoolQuery::Local());
    flush->SetPeriod(config_.flush_period_us_, clio::run::kMicro);
    flush->SetFlags(TASK_PERIODIC);
    ipc->Send(flush);
    auto *manager = CLIO_RUNTIME_MANAGER;
    if (manager != nullptr) {
      stats_hook_id_ = manager->AddStopHook([this]() { LogStats(); });
    }
    HLOG(kInfo,
         "core_shm_cache_writethru {}.{}: segment {} MB, log {} ranges, next {}.{}",
         pool_id_.major_, pool_id_.minor_, config_.capacity_ >> 20,
         config_.log_depth_, CorePoolId().major_, CorePoolId().minor_);
  } else {
    HLOG(kWarning,
         "core_shm_cache_writethru {}.{}: no segment; serving tasks only",
         pool_id_.major_, pool_id_.minor_);
  }
  fields->SetReturnCode(0);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

Runtime::~Runtime() {
  if (stats_hook_id_ != 0) {
    auto *manager = CLIO_RUNTIME_MANAGER;
    if (manager != nullptr) manager->RemoveStopHook(stats_hook_id_);
  }
}

void Runtime::LogStats() {
  if (!view_.Ready()) return;
  const wtc::Stats &st = view_.GetRoot()->stats_;
  auto v = [](const std::atomic<clio::run::u64> &a) { return a.load(); };
  HLOG(kWarning,
       "[shm-cache stats] node={} puts={} put_MiB={} refused: log={} slot={} "
       "buf={} busy={} miss={} waits={} flush_batches={} flush_ranges={} "
       "flush_MiB={} flush_skipped={} fills={} md_fills={} get_hits={} "
       "get_misses={} (absent={} nonres={} range={} busy={}) "
       "invalidations={} evictions={}",
       container_id_, v(st.puts_), v(st.put_bytes_) >> 20,
       v(st.put_full_log_), v(st.put_full_slot_), v(st.put_full_buf_),
       v(st.put_busy_), v(st.put_miss_), v(st.waits_), v(st.flush_batches_),
       v(st.flush_ranges_), v(st.flush_bytes_) >> 20, v(st.flush_skipped_),
       v(st.fills_), v(st.md_fills_), v(st.get_hits_), v(st.get_misses_),
       v(st.miss_absent_), v(st.miss_nonres_), v(st.miss_range_),
       v(st.miss_busy_), v(st.invalidations_), v(st.evictions_));
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
  if (lim.log_depth_ == 0) {
    // Metadata only: no byte writes are logged. Otherwise one entry per KiB
    // of segment: the log must outlast what the buffers can absorb, or puts
    // wait on the flusher long before the cache is full.
    const clio::run::u64 auto_depth =
        config_.md_only_ ? 65536ull : std::max<clio::run::u64>(
                                          65536ull, config_.capacity_ >> 10);
    lim.log_depth_ = static_cast<clio::run::u32>(
        std::min<clio::run::u64>(auto_depth, 1ull << 28));
  }
  lim.evict_sample_ = config_.evict_sample_;
  lim.max_fill_ = config_.max_fill_;
  lim.half_life_ns_ = config_.half_life_ms_ * 1000ull * 1000ull;
  lim.md_only_ = config_.md_only_;
  if (!leased_.empty()) {
    // Extent mode: adjacent leased blocks merge into one extent, so runs
    // can span them (a fresh RAM bdev usually hands out one long stretch).
    std::vector<std::pair<clio::run::u64, clio::run::u64>> ext;
    for (const auto &b : leased_) ext.emplace_back(b.offset_, b.size_);
    std::sort(ext.begin(), ext.end());
    for (const auto &x : ext) {
      if (!lim.extents_.empty() &&
          lim.extents_.back().first + lim.extents_.back().second == x.first) {
        lim.extents_.back().second += x.second;
      } else {
        lim.extents_.push_back(x);
      }
    }
    lim.data_pid_ = pid;
    lim.data_pool_major_ = config_.data_pool_.major_;
    lim.data_pool_minor_ = config_.data_pool_.minor_;
  }
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
  locator_ = loc;
  const clio::run::u64 off = static_cast<clio::run::u64>(
      reinterpret_cast<char *>(loc) - reinterpret_cast<char *>(meta));
  if (!dir->RegisterRoot(pool_id_.ToU64(), off)) {
    HLOG(kWarning, "core_shm_cache_writethru: metadata directory full");
  }
  return true;
}

clio::run::TaskResume Runtime::LeaseExtents() {
  CLIO_TASK_BODY_BEGIN
  {
    // In 64 MiB requests until data_capacity is leased or the device is
    // full; whatever was leased is used (a smaller cache, not no cache).
    clio::run::bdev::Client bdev(config_.data_pool_);
    constexpr clio::run::u64 kLeaseChunk = 64ull << 20;
    clio::run::u64 got = 0;
    while (got < config_.data_capacity_) {
      const clio::run::u64 want =
          std::min(kLeaseChunk, config_.data_capacity_ - got);
      auto f = bdev.AsyncAllocateBlocks(clio::run::PoolQuery::Local(), want);
      CLIO_CO_AWAIT(f);
      if (f->GetReturnCode() != 0 || f->blocks_.empty()) break;
      for (size_t i = 0; i < f->blocks_.size(); ++i) {
        leased_.push_back(f->blocks_[i]);
        got += f->blocks_[i].size_;
      }
    }
    HLOG(kInfo, "core_shm_cache_writethru {}.{}: leased {} MB in {} block(s) "
         "from RAM bdev {}.{}", pool_id_.major_, pool_id_.minor_, got >> 20,
         leased_.size(), config_.data_pool_.major_, config_.data_pool_.minor_);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Destroy(clio::run::shared_ptr<DestroyTask> &task) {
  CLIO_TASK_BODY_BEGIN
  if (locator_ != nullptr) {
    locator_->ready_.store(0, std::memory_order_release);
    locator_ = nullptr;
  }
  view_.Detach();
  if (!leased_.empty()) {
    clio::run::bdev::Client bdev(config_.data_pool_);
    auto f = bdev.AsyncFreeBlocks(clio::run::PoolQuery::Local(), leased_);
    CLIO_CO_AWAIT(f);
    leased_.clear();
  }
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
      KickFlusher();
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
      KickFlusher();
      CLIO_CO_AWAIT(clio::run::yield(kBarrierPollUs));
    }
    // The landed writes' tag-metadata invalidations went out in the
    // background: a sync returns only after every one finished.
    while (!ReapTagNotes()) {
      CLIO_CO_AWAIT(clio::run::yield(kBarrierPollUs));
    }
    // A put that failed for lack of space (any PutBlob no-space code) is
    // the sync's ENOSPC, so fsync reports a full store, not an I/O error.
    sticky = view_.TakeTagError(task->tag_id_.major_, task->tag_id_.minor_);
    if (clio::cte::core::PutRcIsNoSpace(sticky)) {
      sticky = clio::cte::core::kSyncNoSpaceRc;
    }
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
      KickFlusher();
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
        KickFlusher();
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

clio::run::Future<InvalidateTask> Runtime::SendTagNote(const TagId &tag) {
  auto *ipc = CLIO_IPC;
  const std::string meta(wtc::kTagMetaName, wtc::kTagMetaNameLen);
  auto inv = ipc->NewTask<InvalidateTask>(
      clio::run::CreateTaskId(), pool_id_,
      clio::run::PoolQuery::DirectId(HomeOf(tag, meta)), kInvalidateFanout,
      container_id_, PackKeys({BlobKey{tag.major_, tag.minor_, meta}}));
  return ipc->Send(inv);
}

void Runtime::NoteTagsLanded(const std::vector<TagId> &tags) {
  if (!view_.Ready() || CLIO_IPC->GetNumHosts() <= 1) return;
  std::lock_guard<std::mutex> lk(notes_mtx_);
  for (const TagId &t : tags) {
    if (t.IsNull()) continue;
    const clio::run::u64 key =
        (static_cast<clio::run::u64>(t.major_) << 32) | t.minor_;
    auto it = tag_notes_.find(key);
    if (it != tag_notes_.end()) {
      it->second.again_ = true;  // one more after the one in flight
      continue;
    }
    TagNote &n = tag_notes_[key];
    n.fut_ = SendTagNote(t);
  }
}

bool Runtime::ReapTagNotes() {
  std::lock_guard<std::mutex> lk(notes_mtx_);
  for (auto it = tag_notes_.begin(); it != tag_notes_.end();) {
    if (!it->second.fut_.IsComplete()) {
      ++it;
      continue;
    }
    if (it->second.again_) {
      const TagId t(static_cast<clio::run::u32>(it->first >> 32),
                    static_cast<clio::run::u32>(it->first & 0xffffffffu));
      it->second.fut_ = SendTagNote(t);
      it->second.again_ = false;
      ++it;
      continue;
    }
    it = tag_notes_.erase(it);
  }
  return tag_notes_.empty();
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
