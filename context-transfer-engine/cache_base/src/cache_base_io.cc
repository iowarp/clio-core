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
 * Policy cache base: the data path. Puts and gets against the volatile
 * core (always through vol_client_ at PoolQuery::Local(); the cores are
 * disconnected), fills, flushes, eviction and the volatile-full fallback.
 *
 * Invariant the read path relies on: a tracked copy is COMPLETE (created by
 * a put covering the whole blob, or by a whole-blob fill) and removed
 * whole. A busy entry is mid-fill/evict/drop; every path waits it out.
 */

#include <clio_cte/cache_base/cache_base.h>

#include <algorithm>

namespace clio::cte::cache_base {

namespace Method = clio::cte::core::Method;

namespace {
/** Chunk size for copies between the two cores (fill and flush). */
constexpr clio::run::u64 kCopyChunkBytes = 4ULL * 1024 * 1024;
/** Poll period while waiting out a busy copy (microseconds). */
constexpr double kBusyPollUs = 50.0;
/** WriteBackCreate outcomes. */
constexpr clio::run::u32 kCreateDone = 0;
constexpr clio::run::u32 kCreateWriteAround = 1;
constexpr clio::run::u32 kCreateRetry = 2;
}  // namespace

/** Last byte (exclusive) a put/get task addresses, over all its regions. */
template <typename TaskT>
static clio::run::u64 RegionEnd(TaskT &task) {
  clio::run::u64 end = 0;
  clio::cte::core::ForEachBlobRegion(
      task, [&end](const clio::cte::core::BlobRegion &r) {
        end = std::max(end, r.blob_off_ + r.size_);
        return true;
      });
  return end;
}

/** Whether a task addresses exactly one region starting at offset 0. */
template <typename TaskT>
static bool IsSingleFromZero(const TaskT &task) {
  return task.segments_.empty() && task.offset_ == 0;
}

// ===========================================================================
// Volatile / persistent primitives
// ===========================================================================

clio::run::TaskResume CacheBase::VolatilePutRegions(
    clio::run::shared_ptr<clio::cte::core::PutBlobTask> task,
    clio::run::u32 &rc) {
  CLIO_TASK_BODY_BEGIN
  {
    rc = 0;
    std::vector<clio::cte::core::BlobRegion> regions;
    clio::cte::core::ForEachBlobRegion(
        *task, [&regions](const clio::cte::core::BlobRegion &r) {
          regions.push_back(r);
          return true;
        });
    Context ctx = task->context_;
    ctx.replica_ = 0;
    ctx.replica_flags_ = 0;
    const std::string name = task->blob_name_.str();
    for (size_t i = 0; i < regions.size() && rc == 0; ++i) {
      auto put = vol_client_.AsyncPutBlob(
          task->tag_id_, name, regions[i].blob_off_, regions[i].size_,
          regions[i].data_, task->score_, ctx, task->flags_,
          clio::run::PoolQuery::Local());
      CLIO_CO_AWAIT(put);
      rc = put->GetReturnCode();
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume CacheBase::VolatileGetRegions(
    clio::run::shared_ptr<clio::cte::core::GetBlobTask> task,
    clio::run::u32 &rc) {
  CLIO_TASK_BODY_BEGIN
  {
    rc = 0;
    std::vector<clio::cte::core::BlobRegion> regions;
    clio::cte::core::ForEachBlobRegion(
        *task, [&regions](const clio::cte::core::BlobRegion &r) {
          regions.push_back(r);
          return true;
        });
    const std::string name = task->blob_name_.str();
    for (size_t i = 0; i < regions.size() && rc == 0; ++i) {
      auto get = vol_client_.AsyncGetBlob(
          task->tag_id_, name, regions[i].blob_off_, regions[i].size_,
          task->flags_, regions[i].data_, clio::run::PoolQuery::Local(),
          task->context_);
      CLIO_CO_AWAIT(get);
      rc = get->GetReturnCode();
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume CacheBase::PersistentSize(TagId tag, std::string name,
                                                clio::run::u64 &size,
                                                clio::run::u32 &rc) {
#ifdef CLIO_ENABLE_BOOST_COROUTINES
  clio::run::shared_ptr<clio::run::Task> cur_task = clio::run::GetCurrentTask();
#endif
  CLIO_TASK_BODY_BEGIN
  {
    auto sz = pers_client_.AsyncGetBlobSize(tag, name);
    CLIO_CO_AWAIT(sz);
    rc = sz->GetReturnCode();
    size = rc == 0 ? sz->size_ : 0;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume CacheBase::DeleteVolatileCopy(TagId tag,
                                                    std::string name) {
#ifdef CLIO_ENABLE_BOOST_COROUTINES
  clio::run::shared_ptr<clio::run::Task> cur_task = clio::run::GetCurrentTask();
#endif
  CLIO_TASK_BODY_BEGIN
  {
    // Eviction is an ordinary DelBlob on the volatile core; "not found" is
    // an expected answer (a refused create left nothing behind).
    auto del = vol_client_.AsyncDelBlob(tag, name,
                                        clio::run::PoolQuery::Local());
    CLIO_CO_AWAIT(del);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

// ===========================================================================
// PutBlob
// ===========================================================================

clio::run::TaskResume CacheBase::PutBlob(
    clio::run::shared_ptr<clio::cte::core::PutBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  if (task->context_.replica_ != 0) {
    // Explicit replica addressing belongs to the persistent chain.
    CLIO_CO_AWAIT(ForwardToCore(Method::kPutBlob,
                                task.template Cast<clio::run::Task>()));
    CLIO_CO_RETURN;
  }
  if (IsWriteBack() || !HasPersistent()) {
    CLIO_CO_AWAIT(WriteBackPut(task));
  } else {
    CLIO_CO_AWAIT(WriteThruPut(task));
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume CacheBase::WriteThruPut(
    clio::run::shared_ptr<clio::cte::core::PutBlobTask> task) {
  CLIO_TASK_BODY_BEGIN
  {
    const std::string name = task->blob_name_.str();
    const std::string key = Key(task->tag_id_, name);
    const clio::run::u64 end = RegionEnd(*task);
    const Context in_ctx = task->context_;
    // 1. The durable write. Its outcome is the put's outcome.
    CLIO_CO_AWAIT(ForwardToCore(Method::kPutBlob,
                                task.template Cast<clio::run::Task>()));
    const clio::run::u32 rc = task->GetReturnCode();
    const Context out_ctx = task->context_;
    if (rc != 0) {
      // A failed write may have changed part of the durable blob.
      clio::run::u32 drop_rc = 0;
      CLIO_CO_AWAIT(DropCopy(key, false, drop_rc));
      CLIO_CO_RETURN;
    }
    // 2. Keep the copy current: mirror into an existing copy, or create one
    //    when this put is the whole blob. A refused copy is just dropped.
    task->context_ = in_ctx;
    CLIO_CO_AWAIT(MirrorIntoCopy(task, key, end));
    if (params_.Fill() != FillMode::kNone && IsSingleFromZero(*task)) {
      clio::run::u64 psize = 0;
      clio::run::u32 src = 0;
      CLIO_CO_AWAIT(PersistentSize(task->tag_id_, name, psize, src));
      if (src == 0 && psize == task->size_ && Reserve(key, task->tag_id_, name)) {
        clio::run::u32 vrc = 0;
        CLIO_CO_AWAIT(VolatilePutRegions(task, vrc));
        if (IsNoSpaceRc(vrc)) {
          clio::run::u64 freed = 0;
          CLIO_CO_AWAIT(MakeRoom(task->size_, freed));
          CLIO_CO_AWAIT(VolatilePutRegions(task, vrc));
        }
        if (vrc != 0) {
          CLIO_CO_AWAIT(DeleteVolatileCopy(task->tag_id_, name));
        }
        ReleaseReserve(key, vrc == 0, psize, false);
      }
    }
    task->context_ = out_ctx;
    task->SetReturnCode(rc);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume CacheBase::WriteBackPut(
    clio::run::shared_ptr<clio::cte::core::PutBlobTask> task) {
  CLIO_TASK_BODY_BEGIN
  {
    const std::string name = task->blob_name_.str();
    const std::string key = Key(task->tag_id_, name);
    const clio::run::u64 end = RegionEnd(*task);
    const bool dirty = HasPersistent();
    while (true) {
      const PinState st = Pin(key);
      if (st == PinState::kBusy) {
        CLIO_CO_AWAIT(clio::run::yield(kBusyPollUs));
        continue;
      }
      if (st == PinState::kCached) {
        // The copy exists: the write lands there and is acked.
        clio::run::u32 rc = 0;
        CLIO_CO_AWAIT(VolatilePutRegions(task, rc));
        if (IsNoSpaceRc(rc)) {
          clio::run::u64 freed = 0;
          CLIO_CO_AWAIT(MakeRoom(task->size_, freed));
          CLIO_CO_AWAIT(VolatilePutRegions(task, rc));
        }
        if (rc == 0) NoteWrite(key, task->tag_id_, name, end, dirty);
        Unpin(key);
        if (IsNoSpaceRc(rc)) {
          CLIO_CO_AWAIT(WriteBackFallback(task, key));
        } else {
          task->SetReturnCode(rc);
        }
        CLIO_CO_RETURN;
      }
      clio::run::u32 outcome = kCreateDone;
      CLIO_CO_AWAIT(WriteBackCreate(task, key, end, outcome));
      if (outcome == kCreateRetry) continue;
      if (outcome == kCreateDone) CLIO_CO_RETURN;
      break;  // write around
    }
    // Write around: the put lands on persistent directly. A copy filled
    // meanwhile would miss it, so mirror it in.
    CLIO_CO_AWAIT(ForwardToCore(Method::kPutBlob,
                                task.template Cast<clio::run::Task>()));
    if (task->GetReturnCode() == 0) {
      const Context out_ctx = task->context_;
      CLIO_CO_AWAIT(MirrorIntoCopy(task, key, end));
      task->context_ = out_ctx;
      task->SetReturnCode(0);
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume CacheBase::WriteBackCreate(
    clio::run::shared_ptr<clio::cte::core::PutBlobTask> task,
    std::string key, clio::run::u64 end, clio::run::u32 &outcome) {
  CLIO_TASK_BODY_BEGIN
  {
    outcome = kCreateWriteAround;
    const std::string name = task->blob_name_.str();
    clio::run::u64 psize = 0;
    bool fetch = false;
    if (HasPersistent()) {
      // A partial write is absorbed too: when persistent holds bytes this
      // put does not overwrite, they are copied into the volatile copy first
      // (fetch-on-write), so the dirty copy is complete and a flush writes
      // the whole blob back. A blob persistent does not have (rc != 0)
      // starts empty; a gap before the write reads as zeros (the core
      // zero-fills it), exactly as on persistent.
      clio::run::u32 src = 0;
      CLIO_CO_AWAIT(PersistentSize(task->tag_id_, name, psize, src));
      const bool covers_all = IsSingleFromZero(*task) && task->size_ >= psize;
      fetch = src == 0 && psize > 0 && !covers_all;
    }
    if (!Reserve(key, task->tag_id_, name)) {
      outcome = kCreateRetry;  // created or being filled meanwhile
      CLIO_CO_RETURN;
    }
    clio::run::u32 rc = 0;
    bool fetch_failed = false;
    if (fetch) {
      CLIO_CO_AWAIT(CopyPersistentToVolatile(task->tag_id_, name, psize, rc));
      fetch_failed = rc != 0 && !IsNoSpaceRc(rc);
    }
    if (rc == 0) {
      CLIO_CO_AWAIT(VolatilePutRegions(task, rc));
      if (IsNoSpaceRc(rc)) {
        clio::run::u64 freed = 0;
        CLIO_CO_AWAIT(MakeRoom(task->size_, freed));
        CLIO_CO_AWAIT(VolatilePutRegions(task, rc));
      }
    }
    if (rc == 0) {
      ReleaseReserve(key, true, fetch ? std::max(end, psize) : end,
                     HasPersistent());
      task->SetReturnCode(0);
      outcome = kCreateDone;
      CLIO_CO_RETURN;
    }
    // A refused create leaves nothing usable behind.
    CLIO_CO_AWAIT(DeleteVolatileCopy(task->tag_id_, name));
    ReleaseReserve(key, false, 0, false);
    // Persistent is authoritative: a fetch that could not read it, or a
    // full volatile core (when allowed), writes this put around the cache.
    if (fetch_failed ||
        (IsNoSpaceRc(rc) && HasPersistent() && params_.fallback_to_persistent_)) {
      HLOG(kDebug, "write-back cache: writing {} around (rc {})", key, rc);
      outcome = kCreateWriteAround;
      CLIO_CO_RETURN;
    }
    task->SetReturnCode(IsNoSpaceRc(rc) ? clio::cte::core::kPutNoSpaceRc : rc);
    outcome = kCreateDone;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume CacheBase::WriteBackFallback(
    clio::run::shared_ptr<clio::cte::core::PutBlobTask> task,
    std::string key) {
  CLIO_TASK_BODY_BEGIN
  {
    // The copy exists but cannot grow, even after eviction. The copy is
    // unchanged (a refused put rolls back), so it can be written out whole.
    if (!HasPersistent() || !params_.fallback_to_persistent_) {
      task->SetReturnCode(clio::cte::core::kPutNoSpaceRc);
      CLIO_CO_RETURN;
    }
    clio::run::u32 rc = 0;
    CLIO_CO_AWAIT(DropCopy(key, /*flush_first=*/true, rc));
    if (rc != 0) {
      task->SetReturnCode(rc);
      CLIO_CO_RETURN;
    }
    HLOG(kDebug, "write-back cache: volatile full, {} falls back to persistent",
         key);
    CLIO_CO_AWAIT(ForwardToCore(Method::kPutBlob,
                                task.template Cast<clio::run::Task>()));
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume CacheBase::MirrorIntoCopy(
    clio::run::shared_ptr<clio::cte::core::PutBlobTask> task,
    std::string key, clio::run::u64 end) {
  CLIO_TASK_BODY_BEGIN
  {
    // A write that landed on persistent while a copy exists (or appears via
    // a fill) must reach the copy too, or the copy goes stale.
    while (true) {
      const PinState st = Pin(key);
      if (st == PinState::kAbsent) break;
      if (st == PinState::kBusy) {
        CLIO_CO_AWAIT(clio::run::yield(kBusyPollUs));
        continue;
      }
      clio::run::u32 rc = 0;
      CLIO_CO_AWAIT(VolatilePutRegions(task, rc));
      if (rc == 0) {
        NoteWrite(key, task->tag_id_, task->blob_name_.str(), end, false);
      }
      Unpin(key);
      if (rc != 0) {
        clio::run::u32 drop_rc = 0;
        CLIO_CO_AWAIT(DropCopy(key, false, drop_rc));
      }
      break;
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

// ===========================================================================
// GetBlob / GetBlobSize
// ===========================================================================

clio::run::TaskResume CacheBase::GetBlob(
    clio::run::shared_ptr<clio::cte::core::GetBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  if (task->context_.replica_ != 0) {
    CLIO_CO_AWAIT(ForwardToCore(Method::kGetBlob,
                                task.template Cast<clio::run::Task>()));
    CLIO_CO_RETURN;
  }
  {
    const std::string name = task->blob_name_.str();
    const std::string key = Key(task->tag_id_, name);
    while (true) {
      const PinState st = Pin(key);
      if (st == PinState::kBusy) {
        CLIO_CO_AWAIT(clio::run::yield(kBusyPollUs));
        continue;
      }
      if (st == PinState::kAbsent) break;
      clio::run::u32 rc = 0;
      CLIO_CO_AWAIT(VolatileGetRegions(task, rc));
      Unpin(key);
      if (rc == 0) {
        task->SetReturnCode(0);
        CLIO_CO_RETURN;
      }
      break;  // e.g. a read past the copy's end: let persistent answer
    }
    // Miss: the volatile core does not have it; persistent does.
    CLIO_CO_AWAIT(ForwardToCore(Method::kGetBlob,
                                task.template Cast<clio::run::Task>()));
    if (task->GetReturnCode() == 0 && ShouldFill(key)) {
      const Context out_ctx = task->context_;
      const bool whole = IsSingleFromZero(*task);
      CLIO_CO_AWAIT(Fill(task->tag_id_, name,
                         whole ? task->blob_data_ : ctp::ipc::ShmPtr<>(),
                         whole ? task->size_ : 0));
      task->context_ = out_ctx;
      task->SetReturnCode(0);
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume CacheBase::GetBlobSize(
    clio::run::shared_ptr<clio::cte::core::GetBlobSizeTask> &task) {
  CLIO_TASK_BODY_BEGIN
  if (task->replica_ != 0) {
    CLIO_CO_AWAIT(ForwardToCore(Method::kGetBlobSize,
                                task.template Cast<clio::run::Task>()));
    CLIO_CO_RETURN;
  }
  {
    const std::string name = task->blob_name_.str();
    const std::string key = Key(task->tag_id_, name);
    while (true) {
      const PinState st = Pin(key);
      if (st == PinState::kBusy) {
        CLIO_CO_AWAIT(clio::run::yield(kBusyPollUs));
        continue;
      }
      if (st == PinState::kAbsent) break;
      // A copy is complete, so its size is the blob's size (and the only
      // right answer for a dirty copy).
      auto sz = vol_client_.AsyncGetBlobSize(task->tag_id_, name,
                                             clio::run::PoolQuery::Local());
      CLIO_CO_AWAIT(sz);
      Unpin(key);
      if (sz->GetReturnCode() == 0) {
        task->size_ = sz->size_;
        task->SetReturnCode(0);
        CLIO_CO_RETURN;
      }
      break;
    }
    CLIO_CO_AWAIT(ForwardToCore(Method::kGetBlobSize,
                                task.template Cast<clio::run::Task>()));
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

// ===========================================================================
// Fill, flush, drop, evict
// ===========================================================================

clio::run::TaskResume CacheBase::CopyPersistentToVolatile(
    TagId tag, std::string name, clio::run::u64 total, clio::run::u32 &rc) {
#ifdef CLIO_ENABLE_BOOST_COROUTINES
  clio::run::shared_ptr<clio::run::Task> cur_task = clio::run::GetCurrentTask();
#endif
  CLIO_TASK_BODY_BEGIN
  {
    rc = 0;
    ctp::ipc::FullPtr<char> buf;
    if (total > 0) {
      buf = CLIO_IPC->AllocateBuffer(std::min(total, kCopyChunkBytes));
      if (buf.IsNull()) rc = 1;
    }
    for (clio::run::u64 off = 0; rc == 0 && off < total;) {
      const clio::run::u64 len = std::min(kCopyChunkBytes, total - off);
      ctp::ipc::ShmPtr<> src = buf.shm_.template Cast<void>();
      auto get = pers_client_.AsyncGetBlob(tag, name, off, len, 0, src);
      CLIO_CO_AWAIT(get);
      rc = get->GetReturnCode();
      if (rc != 0) break;
      for (int attempt = 0; attempt < 2; ++attempt) {
        auto put = vol_client_.AsyncPutBlob(tag, name, off, len, src, -1.0f,
                                            Context(), 0,
                                            clio::run::PoolQuery::Local());
        CLIO_CO_AWAIT(put);
        rc = put->GetReturnCode();
        if (!IsNoSpaceRc(rc) || attempt == 1) break;
        clio::run::u64 freed = 0;
        CLIO_CO_AWAIT(MakeRoom(total, freed));
      }
      off += len;
    }
    if (!buf.IsNull()) CLIO_IPC->FreeBuffer(buf);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume CacheBase::Fill(TagId tag, std::string name,
                                      ctp::ipc::ShmPtr<> whole_buf,
                                      clio::run::u64 whole_len) {
#ifdef CLIO_ENABLE_BOOST_COROUTINES
  clio::run::shared_ptr<clio::run::Task> cur_task = clio::run::GetCurrentTask();
#endif
  CLIO_TASK_BODY_BEGIN
  {
    const std::string key = Key(tag, name);
    if (!Reserve(key, tag, name)) CLIO_CO_RETURN;  // present or in progress
    clio::run::u64 total = 0;
    clio::run::u32 rc = 0;
    CLIO_CO_AWAIT(PersistentSize(tag, name, total, rc));
    if (rc != 0 || total == 0) {
      ReleaseReserve(key, false, 0, false);
      CLIO_CO_RETURN;
    }
    // Copy persistent -> volatile: straight from the caller's buffer when
    // the read was the whole blob, else chunk by chunk from persistent.
    if (!whole_buf.IsNull() && whole_len >= total) {
      for (int attempt = 0; attempt < 2; ++attempt) {
        auto put = vol_client_.AsyncPutBlob(tag, name, 0, total, whole_buf,
                                            -1.0f, Context(), 0,
                                            clio::run::PoolQuery::Local());
        CLIO_CO_AWAIT(put);
        rc = put->GetReturnCode();
        if (!IsNoSpaceRc(rc) || attempt == 1) break;
        clio::run::u64 freed = 0;
        CLIO_CO_AWAIT(MakeRoom(total, freed));
      }
    } else {
      CLIO_CO_AWAIT(CopyPersistentToVolatile(tag, name, total, rc));
    }
    if (rc != 0) {
      // A fill is optional: a refused one leaves the blob uncached.
      CLIO_CO_AWAIT(DeleteVolatileCopy(tag, name));
    }
    ReleaseReserve(key, rc == 0, total, false);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume CacheBase::FlushEntry(std::string key,
                                            clio::run::u32 &rc) {
#ifdef CLIO_ENABLE_BOOST_COROUTINES
  clio::run::shared_ptr<clio::run::Task> cur_task = clio::run::GetCurrentTask();
#endif
  CLIO_TASK_BODY_BEGIN
  {
    rc = 0;
    TagId tag;
    std::string name;
    clio::run::u64 size = 0;
    clio::run::u64 gen = 0;
    // One flusher per copy: two racing flushes could land an older read
    // after a newer one and then mark the copy clean.
    while (true) {
      {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = entries_.find(key);
        if (it == entries_.end() || !it->second.dirty_) CLIO_CO_RETURN;
        if (!it->second.flushing_) {
          it->second.flushing_ = true;
          it->second.pins_++;  // a delete waits for the flush to finish
          tag = it->second.tag_;
          name = it->second.name_;
          size = it->second.size_;
          gen = it->second.gen_;
          break;
        }
      }
      CLIO_CO_AWAIT(clio::run::yield(kBusyPollUs));
    }
    ctp::ipc::FullPtr<char> buf;
    if (size > 0) {
      buf = CLIO_IPC->AllocateBuffer(std::min(size, kCopyChunkBytes));
      if (buf.IsNull()) rc = 1;
    }
    for (clio::run::u64 off = 0; rc == 0 && off < size;) {
      const clio::run::u64 len = std::min(kCopyChunkBytes, size - off);
      ctp::ipc::ShmPtr<> p = buf.shm_.template Cast<void>();
      auto get = vol_client_.AsyncGetBlob(tag, name, off, len, 0, p,
                                          clio::run::PoolQuery::Local());
      CLIO_CO_AWAIT(get);
      rc = get->GetReturnCode();
      if (rc != 0) break;
      auto put = pers_client_.AsyncPutBlob(tag, name, off, len, p);
      CLIO_CO_AWAIT(put);
      rc = put->GetReturnCode();
      off += len;
    }
    if (!buf.IsNull()) CLIO_IPC->FreeBuffer(buf);
    {
      std::lock_guard<std::mutex> lock(mu_);
      auto it = entries_.find(key);
      if (it != entries_.end()) {
        it->second.flushing_ = false;
        if (it->second.pins_ > 0) it->second.pins_--;
        // A write during the flush bumped gen: those bytes stay dirty.
        if (rc == 0 && it->second.gen_ == gen) it->second.dirty_ = false;
      }
    }
    if (rc != 0) {
      HLOG(kWarning, "write-back cache: flushing {} failed rc={}", key, rc);
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume CacheBase::FlushKeys(std::vector<std::string> keys,
                                           clio::run::u32 &rc) {
#ifdef CLIO_ENABLE_BOOST_COROUTINES
  clio::run::shared_ptr<clio::run::Task> cur_task = clio::run::GetCurrentTask();
#endif
  CLIO_TASK_BODY_BEGIN
  {
    rc = 0;
    for (size_t i = 0; i < keys.size(); ++i) {
      clio::run::u32 one = 0;
      CLIO_CO_AWAIT(FlushEntry(keys[i], one));
      if (rc == 0) rc = one;  // keep going; report the first failure
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume CacheBase::DropCopy(std::string key, bool flush_first,
                                          clio::run::u32 &rc) {
#ifdef CLIO_ENABLE_BOOST_COROUTINES
  clio::run::shared_ptr<clio::run::Task> cur_task = clio::run::GetCurrentTask();
#endif
  CLIO_TASK_BODY_BEGIN
  {
    rc = 0;
    while (true) {
      {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = entries_.find(key);
        if (it == entries_.end()) CLIO_CO_RETURN;
        if (!it->second.busy_ && it->second.pins_ == 0) {
          it->second.busy_ = true;
          break;
        }
      }
      CLIO_CO_AWAIT(clio::run::yield(kBusyPollUs));
    }
    CLIO_CO_AWAIT(DropBusyCopy(key, flush_first, rc));
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume CacheBase::DropBusyCopy(std::string key,
                                              bool flush_first,
                                              clio::run::u32 &rc) {
#ifdef CLIO_ENABLE_BOOST_COROUTINES
  clio::run::shared_ptr<clio::run::Task> cur_task = clio::run::GetCurrentTask();
#endif
  CLIO_TASK_BODY_BEGIN
  {
    rc = 0;
    TagId tag;
    std::string name;
    bool dirty = false;
    {
      std::lock_guard<std::mutex> lock(mu_);
      auto it = entries_.find(key);
      if (it == entries_.end()) CLIO_CO_RETURN;
      tag = it->second.tag_;
      name = it->second.name_;
      dirty = it->second.dirty_;
    }
    if (flush_first && dirty) {
      // Never drop a dirty copy silently.
      CLIO_CO_AWAIT(FlushEntry(key, rc));
      if (rc != 0) {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = entries_.find(key);
        if (it != entries_.end()) it->second.busy_ = false;
        CLIO_CO_RETURN;
      }
    }
    CLIO_CO_AWAIT(DeleteVolatileCopy(tag, name));
    Erase(key);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume CacheBase::DropKeys(std::vector<std::string> keys,
                                          bool flush_first) {
#ifdef CLIO_ENABLE_BOOST_COROUTINES
  clio::run::shared_ptr<clio::run::Task> cur_task = clio::run::GetCurrentTask();
#endif
  CLIO_TASK_BODY_BEGIN
  {
    for (size_t i = 0; i < keys.size(); ++i) {
      clio::run::u32 rc = 0;
      CLIO_CO_AWAIT(DropCopy(keys[i], flush_first, rc));
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume CacheBase::MakeRoom(clio::run::u64 needed,
                                          clio::run::u64 &freed) {
#ifdef CLIO_ENABLE_BOOST_COROUTINES
  clio::run::shared_ptr<clio::run::Task> cur_task = clio::run::GetCurrentTask();
#endif
  CLIO_TASK_BODY_BEGIN
  {
    freed = 0;
    // Victims in policy order, snapshotted once; each is re-checked as it is
    // taken (MarkBusy) since the map moves on while we flush.
    std::vector<std::pair<clio::run::u64, std::string>> order;
    {
      std::lock_guard<std::mutex> lock(mu_);
      const bool lfu = params_.Evict() == EvictMode::kLfu;
      for (const auto &kv : entries_) {
        if (kv.second.busy_ || kv.second.pins_ > 0) continue;
        order.emplace_back(lfu ? kv.second.hits_ : kv.second.last_access_ns_,
                           kv.first);
      }
    }
    std::sort(order.begin(), order.end());
    size_t evicted = 0;
    for (size_t i = 0; i < order.size() && freed < needed; ++i) {
      const std::string &key = order[i].second;
      clio::run::u64 size = 0;
      {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = entries_.find(key);
        if (it == entries_.end()) continue;
        size = it->second.size_;
      }
      // Never wait for a victim: two writers each holding a pin while
      // making room would otherwise wait on each other forever.
      if (!MarkBusy(key)) continue;
      clio::run::u32 rc = 0;
      CLIO_CO_AWAIT(DropBusyCopy(key, /*flush_first=*/true, rc));
      if (rc != 0) continue;  // could not write a dirty victim out
      freed += size;
      ++evicted;
    }
    HLOG(kDebug, "policy cache: evicted {} copies ({} bytes) for {} bytes",
         evicted, freed, needed);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

}  // namespace clio::cte::cache_base
