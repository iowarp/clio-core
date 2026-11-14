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
 * shm_writethru_cache data verbs for TASK callers (remote nodes, TCP/IPC
 * clients, runtime-internal clients, and SHM clients whose op the segment
 * could not express). A plain put lands in the segment exactly as a client's
 * direct write would; everything else drains the blob's pending ranges and
 * goes down the chain.
 */

#include <cstring>
#include <string>
#include <vector>

#include <clio_cte/core/blob_batch.h>
#include <clio_cte/shm_writethru_cache/shm_writethru_cache_runtime.h>

namespace clio::cte::shm_wtc {

namespace wtc = clio::cte::core::wtc;
using clio::cte::core::BlobRegion;

static constexpr double kBarrierPollUs = 10.0;
static constexpr double kNoSpacePollUs = 50.0;

clio::run::TaskResume Runtime::WaitBlobFlushed(const TagId &tag,
                                               const std::string &name) {
  CLIO_TASK_BODY_BEGIN
  if (view_.Ready()) {
    while (view_.BlobPending(tag.major_, tag.minor_, name.c_str(),
                             name.size()) != 0) {
      CLIO_CO_AWAIT(clio::run::yield(kBarrierPollUs));
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::FanoutInvalidate(const TagId &tag,
                                                const std::string &name) {
  CLIO_TASK_BODY_BEGIN
  if (CLIO_IPC->GetNumHosts() > 1) {
    auto *ipc = CLIO_IPC;
    BlobKey k{tag.major_, tag.minor_, name};
    auto inv = ipc->NewTask<InvalidateTask>(
        clio::run::CreateTaskId(), pool_id_,
        clio::run::PoolQuery::DirectId(HomeOf(tag, name)), kInvalidateFanout,
        container_id_, PackKeys({k}));
    auto fut = ipc->Send(inv);
    CLIO_CO_AWAIT(fut);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

// ===========================================================================
// Writes
// ===========================================================================

clio::run::TaskResume Runtime::PutBlob(
    clio::run::shared_ptr<clio::cte::core::PutBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    const std::string name = task->GetBlobName();
    const bool plain =
        view_.Ready() &&
        clio::cte::core::Client::WtcAbsorbableFlags(task->flags_) &&
        task->score_ < 0.0f && PlainContext(task->context_) &&
        name.size() <= wtc::kMaxBlobName;
    bool absorbed = plain;
    if (plain) {
      std::vector<BlobRegion> regions;
      clio::cte::core::ForEachBlobRegion(*task, [&regions](const BlobRegion &r) {
        regions.push_back(r);
        return true;
      });
      auto *ipc = CLIO_IPC;
      for (size_t i = 0; i < regions.size() && absorbed; ++i) {
        if (regions[i].size_ == 0) {
          continue;
        }
        const char *src =
            ipc->ToFullPtr<char>(regions[i].data_.template Cast<char>()).ptr_;
        if (src == nullptr) {
          task->SetReturnCode(1);
          CLIO_CO_RETURN;
        }
        wtc::Status st = wtc::Status::kNoSpace;
        while ((st = view_.TryPut(task->tag_id_.major_, task->tag_id_.minor_,
                                  name.c_str(), name.size(),
                                  regions[i].blob_off_, regions[i].size_, src,
                                  task->flags_)) == wtc::Status::kNoSpace) {
          view_.GetRoot()->stats_.waits_.fetch_add(1);
          CLIO_CO_AWAIT(clio::run::yield(kNoSpacePollUs));
        }
        // kMiss only on the FIRST region (a flags conflict is per blob), so
        // nothing was absorbed when we fall through to the chain below.
        absorbed = st == wtc::Status::kOk;
      }
      if (absorbed) {
        task->SetReturnCode(0);
        CLIO_CO_RETURN;
      }
    }
    // Not expressible in the segment: drain this blob's buffered ranges so
    // the chain orders this write after them, then drop the local copy.
    CLIO_CO_AWAIT(WaitBlobFlushed(task->tag_id_, name));
    if (view_.Ready()) {
      view_.Invalidate(task->tag_id_.major_, task->tag_id_.minor_,
                       name.c_str(), name.size());
    }
    if (task->segments_.empty()) {
      auto f = Next()->AsyncPutBlob(task->tag_id_, name, task->offset_,
                                    task->size_, task->blob_data_,
                                    task->score_, task->context_, task->flags_,
                                    clio::run::PoolQuery::Dynamic());
      CLIO_CO_AWAIT(f);
      task->context_ = f->context_;
      task->SetReturnCode(f->GetReturnCode());
    } else {
      CLIO_CO_AWAIT(ForwardToCore(Method::kPutBlob,
                                  task.template Cast<clio::run::Task>()));
    }
    CLIO_CO_AWAIT(FanoutInvalidate(task->tag_id_, name));
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::MultiPutBlob(
    clio::run::shared_ptr<clio::cte::core::MultiPutBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    clio::cte::core::MultiPutBatchView batch;
    const bool attached = clio::cte::core::MultiPutBatchView::Attach(*task, &batch);
    if (view_.Ready() && attached && PlainContext(task->context_)) {
      task->num_ok_ = 0;
      task->first_rc_ = 0;
      for (size_t i = 0; i < batch.size(); ++i) {
        const auto &d = batch.descs_[i];
        if (!batch.RecordValid(i) || d.blob_name_.size() > wtc::kMaxBlobName) {
          if (task->first_rc_ == 0) task->first_rc_ = 1;
          continue;
        }
        wtc::Status st = wtc::Status::kNoSpace;
        while ((st = view_.TryPut(d.tag_id_.major_, d.tag_id_.minor_,
                                  d.blob_name_.c_str(), d.blob_name_.size(),
                                  d.offset_, d.size_,
                                  batch.base_ + d.payload_off_, d.flags_)) ==
               wtc::Status::kNoSpace) {
          CLIO_CO_AWAIT(clio::run::yield(kNoSpacePollUs));
        }
        if (st != wtc::Status::kOk) {
          if (task->first_rc_ == 0) task->first_rc_ = 1;
          continue;
        }
        ++task->num_ok_;
      }
      task->SetReturnCode(task->first_rc_ == 0 ? 0 : 1);
      CLIO_CO_RETURN;
    }
  }
  CLIO_CO_AWAIT(BarrierForward(Method::kMultiPutBlob,
                               task.template Cast<clio::run::Task>()));
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::DelBlob(
    clio::run::shared_ptr<clio::cte::core::DelBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    const std::string name = task->blob_name_.str();
    CLIO_CO_AWAIT(WaitBlobFlushed(task->tag_id_, name));
    if (view_.Ready()) {
      view_.DropBlob(task->tag_id_.major_, task->tag_id_.minor_, name.c_str(),
                     name.size());
    }
    auto f = Next()->AsyncDelBlob(task->tag_id_, name,
                                  clio::run::PoolQuery::Dynamic(),
                                  task->del_flags_);
    CLIO_CO_AWAIT(f);
    task->SetReturnCode(f->GetReturnCode());
    CLIO_CO_AWAIT(FanoutInvalidate(task->tag_id_, name));
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

// ===========================================================================
// Reads
// ===========================================================================

clio::run::TaskResume Runtime::RegisterSharer(const TagId &tag,
                                              const std::string &name) {
  CLIO_TASK_BODY_BEGIN
  {
    const clio::run::u32 home = HomeOf(tag, name);
    if (home == container_id_) {
      std::lock_guard<std::mutex> lk(sharers_mtx_);
      sharers_[SharerKey(tag.major_, tag.minor_, name)].nodes_.insert(
          container_id_);
    } else {
      auto *ipc = CLIO_IPC;
      BlobKey k{tag.major_, tag.minor_, name};
      auto reg = ipc->NewTask<InvalidateTask>(
          clio::run::CreateTaskId(), pool_id_,
          clio::run::PoolQuery::DirectId(home), kInvalidateRegister,
          container_id_, PackKeys({k}));
      auto fut = ipc->Send(reg);
      CLIO_CO_AWAIT(fut);
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::FillFromChain(const TagId &tag,
                                             const std::string &name,
                                             bool *filled) {
  CLIO_TASK_BODY_BEGIN
  *filled = false;
  {
    clio::run::u64 gen = 0;
    clio::run::u64 epoch = 0;
    if (!view_.SnapshotGens(tag.major_, tag.minor_, name.c_str(), name.size(),
                            &gen, &epoch)) {
      CLIO_CO_RETURN;
    }
    // Register BEFORE fetching: a write landing at the home after this point
    // invalidates us, which bumps the epoch and voids the fill below.
    CLIO_CO_AWAIT(RegisterSharer(tag, name));
    auto sz = Next()->AsyncGetBlobSize(tag, name);
    CLIO_CO_AWAIT(sz);
    const clio::run::u64 size = sz->size_;
    if (sz->GetReturnCode() != 0 || size == 0 || size > config_.max_fill_) {
      CLIO_CO_RETURN;
    }
    auto *ipc = CLIO_IPC;
    ctp::ipc::FullPtr<char> buf = ipc->AllocateBuffer(size);
    if (buf.IsNull()) {
      CLIO_CO_RETURN;
    }
    auto get = Next()->AsyncGetBlob(tag, name, 0, size, 0,
                                    ctp::ipc::ShmPtr<>(buf.shm_),
                                    clio::run::PoolQuery::Dynamic());
    CLIO_CO_AWAIT(get);
    if (get->GetReturnCode() == 0) {
      *filled = view_.Fill(tag.major_, tag.minor_, name.c_str(), name.size(),
                           buf.ptr_, size, gen, epoch);
    }
    ipc->FreeBuffer(buf);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::GetBlob(
    clio::run::shared_ptr<clio::cte::core::GetBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    const std::string name = task->GetBlobName();
    auto *ipc = CLIO_IPC;
    char *dst = task->blob_data_.IsNull()
                    ? nullptr
                    : ipc->ToFullPtr<char>(task->blob_data_.template Cast<char>()).ptr_;
    const bool eligible = view_.Ready() && task->flags_ == 0 &&
                          PlainContext(task->context_) &&
                          task->segments_.empty() && dst != nullptr &&
                          task->size_ > 0 && name.size() <= wtc::kMaxBlobName;
    const TagId tag = task->tag_id_;
    if (eligible) {
      if (view_.TryGet(tag.major_, tag.minor_, name.c_str(), name.size(),
                       task->offset_, task->size_, dst) == wtc::Status::kOk) {
        task->SetReturnCode(0);
        CLIO_CO_RETURN;
      }
      CLIO_CO_AWAIT(WaitBlobFlushed(tag, name));
      bool filled = false;
      CLIO_CO_AWAIT(FillFromChain(tag, name, &filled));
      if (filled &&
          view_.TryGet(tag.major_, tag.minor_, name.c_str(), name.size(),
                       task->offset_, task->size_, dst) == wtc::Status::kOk) {
        task->SetReturnCode(0);
        CLIO_CO_RETURN;
      }
    } else {
      CLIO_CO_AWAIT(WaitBlobFlushed(tag, name));
    }
    if (task->segments_.empty()) {
      auto f = Next()->AsyncGetBlob(tag, name, task->offset_, task->size_,
                                    task->flags_, task->blob_data_,
                                    clio::run::PoolQuery::Dynamic(),
                                    task->context_);
      CLIO_CO_AWAIT(f);
      task->context_ = f->context_;
      task->SetReturnCode(f->GetReturnCode());
    } else {
      CLIO_CO_AWAIT(ForwardToCore(Method::kGetBlob,
                                  task.template Cast<clio::run::Task>()));
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::GetBlobSize(
    clio::run::shared_ptr<clio::cte::core::GetBlobSizeTask> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    const std::string name = task->blob_name_.str();
    const TagId tag = task->tag_id_;
    clio::run::u64 size = 0;
    if (task->replica_ == 0 && view_.Ready() &&
        view_.TryGetSize(tag.major_, tag.minor_, name.c_str(), name.size(),
                         &size) == wtc::Status::kOk) {
      task->size_ = size;
      task->lost_bytes_ = 0;
      task->SetReturnCode(0);
      CLIO_CO_RETURN;
    }
    CLIO_CO_AWAIT(WaitBlobFlushed(tag, name));
    auto f = Next()->AsyncGetBlobSize(tag, name, clio::run::PoolQuery::Dynamic(),
                                      task->replica_);
    CLIO_CO_AWAIT(f);
    task->size_ = f->size_;
    task->lost_bytes_ = f->lost_bytes_;
    task->SetReturnCode(f->GetReturnCode());
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

}  // namespace clio::cte::shm_wtc
