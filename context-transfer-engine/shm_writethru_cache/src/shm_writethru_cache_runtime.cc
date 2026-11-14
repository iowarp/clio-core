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
 * shm_writethru_cache runtime. See ../DESIGN.md.
 *
 * Data verbs route submitter-local: the segment is per node, so the node
 * that issued an op is the one whose cache it reads or fills. Everything the
 * cache does not serve goes down `next_pool_id` through a core client
 * (Dynamic routing) after the blob's (or the log's) pending ranges landed.
 */

#include <algorithm>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <clio_cte/core/blob_batch.h>
#include <clio_cte/core/blob_placement.h>
#include <clio_cte/shm_writethru_cache/shm_writethru_cache_runtime.h>

namespace clio::cte::shm_wtc {

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
         "shm_writethru_cache {}.{}: segment {} MB, log {} ranges, next {}.{}",
         pool_id_.major_, pool_id_.minor_, config_.capacity_ >> 20,
         config_.log_depth_, CorePoolId().major_, CorePoolId().minor_);
  } else {
    HLOG(kWarning,
         "shm_writethru_cache {}.{}: no segment; serving tasks only",
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
  loc->ready_.store(1, std::memory_order_release);
  const clio::run::u64 off = static_cast<clio::run::u64>(
      reinterpret_cast<char *>(loc) - reinterpret_cast<char *>(meta));
  if (!dir->RegisterRoot(pool_id_.ToU64(), off)) {
    HLOG(kWarning, "shm_writethru_cache: metadata directory full");
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
    auto f = Next()->AsyncGetOrCreateTag(
        task->tag_name_.str(), task->fault_pool_id_,
        task->fault_pool_name_.str(), task->fault_params_.str(), task->tag_id_,
        clio::run::PoolQuery::Dynamic());
    CLIO_CO_AWAIT(f);
    task->tag_id_ = f->tag_id_;
    task->created_ = f->created_;
    task->tag_size_ = f->tag_size_;
    if (f->GetReturnCode() == 0 && f->created_ != 0 && view_.Ready()) {
      view_.NoteTagCreated(task->tag_id_.major_, task->tag_id_.minor_);
    }
    task->SetReturnCode(f->GetReturnCode());
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
  if (view_.Ready()) {
    const clio::run::u64 target = view_.PushedNow();
    while (!view_.FlushedThrough(target)) {
      CLIO_CO_AWAIT(clio::run::yield(kBarrierPollUs));
    }
    sticky = view_.TakeTagError(task->tag_id_.major_, task->tag_id_.minor_);
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
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

}  // namespace clio::cte::shm_wtc

// Module entry point.
CLIO_TASK_CC(clio::cte::shm_wtc::Runtime)
