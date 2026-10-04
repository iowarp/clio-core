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
 * Replication runtime: remote copies and failover. At a blob's owner every
 * change is mirrored to the next remote_copies containers by hash (other
 * nodes), where it is stored as a shadow copy. While the owner's node is
 * dead the core routes the blob to its first live successor, which serves
 * it from the shadow and records what it changed; the changes are handed
 * back when the owner returns (pulled by the owner at restart, pushed by the
 * HandoffSweep otherwise).
 */

#include <clio_cte/replication/replication_runtime.h>
#include <clio_cte/core/blob_placement.h>

#include <algorithm>

namespace clio::cte::replication {

namespace {
/** Chunk size for copying a shadow back to its owner. */
constexpr clio::run::u64 kHandoffChunk = 4ULL << 20;
/** Poll period while a hand-back pass or the hand-back gate waits (us). */
constexpr double kHandoffWaitUs = 500.0;
/** Conditional-put bits never apply to a mirrored copy. */
constexpr clio::run::u32 kConditionalBits =
    clio::cte::core::Context::kPutIfAbsent |
    clio::cte::core::Context::kPutIfVersion;
}  // namespace

// ===========================================================================
// Placement helpers
// ===========================================================================

clio::run::u32 Runtime::NumContainers() const {
  return clio::cte::core::PoolContainers(pool_id_);
}

clio::run::u32 Runtime::OwnerOf(const TagId &tag,
                                const std::string &name) const {
  const clio::run::u32 n = NumContainers();
  return n == 0 ? container_id_ : clio::cte::core::BlobHash(tag, name) % n;
}

bool Runtime::ContainerAlive(clio::run::u32 container) const {
  return clio::cte::core::ContainerNodeAlive(pool_id_, container);
}

clio::cte::core::Client *Runtime::Self() {
  if (!self_client_) {
    self_client_ = std::make_unique<clio::cte::core::Client>(pool_id_);
  }
  return self_client_.get();
}

clio::run::TaskResume Runtime::InvalidateCachedEverywhere(TagId tag,
                                                         std::string name) {
  CLIO_TASK_BODY_BEGIN
  const clio::run::u32 n = NumContainers();
  for (clio::run::u32 c = 0; c < n; ++c) {
    // Container ids are node ids; a dead node's cache died with it.
    if (c == container_id_ || !ContainerAlive(c)) continue;
    auto inval = GetCoreClient()->AsyncDelBlob(
        tag, name, clio::run::PoolQuery::Physical(c),
        clio::cte::core::kDelCacheCopyOnly);
    CLIO_CO_AWAIT(inval);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

void Runtime::NoteHandoff(clio::run::u32 owner, const TagId &tag,
                          const std::string &name, bool deleted) {
  std::lock_guard<std::mutex> g(handoff_mu_);
  const std::string key = std::to_string(tag.major_) + "." +
                          std::to_string(tag.minor_) + "." + name;
  handoff_[owner][key] = HandoffEntry{tag, name, deleted, ++handoff_seq_};
  LogHandoff(kHandoffNote, owner, handoff_[owner][key]);
}

// ===========================================================================
// Writes
// ===========================================================================

clio::run::TaskResume Runtime::MirrorRange(TagId tag, std::string name,
                                           clio::run::u64 off,
                                           clio::run::u64 size,
                                           ctp::ipc::ShmPtr<> data,
                                           float score) {
  CLIO_TASK_BODY_BEGIN
  const clio::run::u32 n = NumContainers();
  for (int i = 1; i <= config_.remote_copies_ && i < static_cast<int>(n); ++i) {
    const clio::run::u32 c = (container_id_ + i) % n;
    if (!ContainerAlive(c)) continue;  // degraded: one copy fewer for now
    Context ctx;
    ctx.op_flags_ |= Context::kShadowCopy;
    auto p = Self()->AsyncPutBlob(tag, name, off, size, data, score, ctx, 0u,
                                  clio::run::PoolQuery::DirectId(c));
    CLIO_CO_AWAIT(p);
    if (p->GetReturnCode() != 0) {
      HLOG(kWarning, "replication: remote copy of {}.{}/{} on container {} "
           "failed (rc {})", tag.major_, tag.minor_, name, c,
           p->GetReturnCode());
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::ReadRemoteCopy(
    clio::run::shared_ptr<clio::cte::core::GetBlobTask> &task, bool &served) {
  CLIO_TASK_BODY_BEGIN
  served = false;
  const clio::run::u32 n = NumContainers();
  const TagId tag = task->tag_id_;
  const std::string name = task->blob_name_.str();
  for (int i = 1; i <= config_.remote_copies_ && i < static_cast<int>(n) &&
                  !served; ++i) {
    const clio::run::u32 c = (container_id_ + i) % n;
    if (!ContainerAlive(c)) continue;
    const clio::run::PoolQuery at = clio::run::PoolQuery::DirectId(c);
    // One scalar read per region: the task's own range, or each segment of
    // a vectored read into that segment's buffer.
    std::vector<clio::cte::core::BlobSegment> regions;
    if (task->segments_.empty()) {
      regions.emplace_back(task->offset_, task->size_, task->blob_data_);
    } else {
      for (size_t k = 0; k < task->segments_.size(); ++k) {
        regions.push_back(task->segments_[k]);
      }
    }
    bool all = true;
    for (size_t k = 0; k < regions.size() && all; ++k) {
      auto g = GetCoreClient()->AsyncGetBlob(tag, name, regions[k].blob_off_,
                                             regions[k].size_, task->flags_,
                                             regions[k].data_, at);
      CLIO_CO_AWAIT(g);
      all = g->GetReturnCode() == 0;
    }
    if (all) {
      HLOG(kWarning, "replication: {}.{}/{} unreadable at its owner "
           "(device down); served from the remote copy on container {}",
           tag.major_, tag.minor_, name, c);
      served = true;
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

namespace {
/** Resends of a forwarded put whose owner died mid-flight. */
constexpr int kForwardRetries = 10;
/** Wait between those resends (us): the cluster needs a moment to mark the
 *  node dead, after which failover routes to its successor. */
constexpr double kForwardRetryUs = 1000000.0;
}  // namespace

clio::run::TaskResume Runtime::PutBlob(
    clio::run::shared_ptr<clio::cte::core::PutBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  if (task->context_.replica_ != 0) {
    CLIO_CO_AWAIT(PutBlobLocal(task));
    CLIO_CO_RETURN;
  }
  // A failover write is recorded for hand-back whatever remote_copies_ is
  // (#1130): with no remote copy (replication_factor 1, the default) the
  // stand-in holds the ONLY copy of what was written while the owner was
  // down, and the owner returns without it unless it is handed back.
  const TagId tag = task->tag_id_;
  const std::string name = task->blob_name_.str();
  const clio::run::u32 owner = OwnerOf(tag, name);
  if (owner != container_id_) {
    // A copy mirrored from the owner, or -- with no shadow mark -- a client
    // write routed here because the owner is down (failover).
    const bool failover =
        (task->context_.op_flags_ & Context::kShadowCopy) == 0;
    task->context_.op_flags_ |= Context::kShadowCopy;
    CLIO_CO_AWAIT(PutBlobLocal(task));
    if (failover && task->GetReturnCode() == 0) {
      NoteHandoff(owner, tag, name, false);
      CLIO_CO_AWAIT(InvalidateCachedEverywhere(tag, name));
    }
    CLIO_CO_RETURN;
  }
  // This container owns the blob: after a restart, the stand-in's copy of
  // it lands first.
  CLIO_CO_AWAIT(AwaitHandback(task->context_));
  CLIO_CO_AWAIT(PutBlobLocal(task));
  if (task->GetReturnCode() != 0) CLIO_CO_RETURN;
  // A hand-back push comes FROM the stand-in, which holds the newer copy:
  // mirroring it back would overwrite that copy with what the push read
  // before a still-running failover put finished (#1154).
  if ((task->context_.op_flags_ & Context::kHandoffPush) != 0) {
    CLIO_CO_RETURN;
  }
  std::vector<clio::cte::core::BlobRegion> regions;
  clio::cte::core::ForEachBlobRegion(*task,
      [&regions](const clio::cte::core::BlobRegion &r) {
        regions.push_back(r);
        return true;
      });
  for (const auto &r : regions) {
    CLIO_CO_AWAIT(MirrorRange(tag, name, r.blob_off_, r.size_, r.data_,
                              config_.replica_score_));
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::MultiPutBlob(
    clio::run::shared_ptr<clio::cte::core::MultiPutBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  CLIO_CO_AWAIT(AwaitHandback(task->context_));
  if (task->context_.replica_ != 0) {
    CLIO_CO_AWAIT(MultiPutBlobLocal(task));
    CLIO_CO_RETURN;
  }
  clio::cte::core::MultiPutBatchView batch;
  if (!clio::cte::core::MultiPutBatchView::Attach(*task, &batch)) {
    CLIO_CO_AWAIT(MultiPutBlobLocal(task));
    CLIO_CO_RETURN;
  }
  // A client's write-behind batch mixes blobs of many owners and arrives at
  // the submitting node. Records this container serves take the batched
  // local path; if any belongs elsewhere, every record goes through the
  // replication layer at its owner, so its replicas and remote copies are
  // written (the core alone would forward them to the owner's core).
  bool all_here = true;
  for (size_t d = 0; d < batch.size() && all_here; ++d) {
    if (!batch.RecordValid(d)) continue;
    const auto &desc = batch.descs_[d];
    const clio::run::u32 owner = OwnerOf(desc.tag_id_, desc.blob_name_);
    all_here = clio::cte::core::FailoverContainer(pool_id_, owner) ==
               container_id_;
  }
  if (!all_here) {
    task->num_ok_ = 0;
    task->first_rc_ = 0;
    for (size_t d = 0; d < batch.size(); ++d) {
      if (!batch.RecordValid(d)) continue;
      const auto &desc = batch.descs_[d];
      Context ctx = task->context_;
      auto p = Self()->AsyncPutBlob(desc.tag_id_, desc.blob_name_,
                                    desc.offset_, desc.size_,
                                    batch.RecordSlice(d), -1.0f, ctx, 0u,
                                    clio::run::PoolQuery::Dynamic());
      CLIO_CO_AWAIT(p);
      // The owner died with this put in flight: re-resolve it (failover now
      // names the successor) and write again -- a put is idempotent.
      for (int attempt = 0; attempt < kForwardRetries &&
                            clio::cte::core::IsNodeLostRc(p->GetReturnCode());
           ++attempt) {
        HLOG(kWarning, "replication: owner of {}.{}/{} was lost with the put "
             "in flight; resending (attempt {})", desc.tag_id_.major_,
             desc.tag_id_.minor_, desc.blob_name_, attempt + 1);
        CLIO_CO_AWAIT(clio::run::yield(kForwardRetryUs));
        p = Self()->AsyncPutBlob(desc.tag_id_, desc.blob_name_, desc.offset_,
                                 desc.size_, batch.RecordSlice(d), -1.0f, ctx,
                                 0u, clio::run::PoolQuery::Dynamic());
        CLIO_CO_AWAIT(p);
      }
      if (p->GetReturnCode() == 0) {
        task->num_ok_++;
      } else if (task->first_rc_ == 0) {
        task->first_rc_ = p->GetReturnCode();
      }
    }
    task->SetReturnCode(task->first_rc_);
    CLIO_CO_RETURN;
  }
  const bool owner_here = batch.size() == 0 ||
      OwnerOf(task->route_tag_id_, task->route_blob_.str()) == container_id_;
  if (!owner_here) {
    task->context_.op_flags_ |= Context::kShadowCopy;  // standing in
  }
  CLIO_CO_AWAIT(MultiPutBlobLocal(task));
  if (task->GetReturnCode() != 0) CLIO_CO_RETURN;
  for (size_t d = 0; d < batch.size(); ++d) {
    if (!batch.RecordValid(d)) continue;
    const auto &desc = batch.descs_[d];
    if (!owner_here) {
      // Standing in for a dead owner: always hand back (#1130).
      NoteHandoff(OwnerOf(desc.tag_id_, desc.blob_name_), desc.tag_id_,
                  desc.blob_name_, false);
      continue;
    }
    if (config_.remote_copies_ <= 0) continue;
    CLIO_CO_AWAIT(MirrorRange(desc.tag_id_, desc.blob_name_, desc.offset_,
                              desc.size_, batch.RecordSlice(d),
                              config_.replica_score_));
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::DelBlob(
    clio::run::shared_ptr<clio::cte::core::DelBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  const TagId tag = task->tag_id_;
  const std::string name = task->blob_name_.str();
  CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kDelBlob,
                              task.template Cast<clio::run::Task>()));
  // A cache-copy invalidation touches only this node's cache copy.
  if (task->del_flags_ & clio::cte::core::kDelCacheCopyOnly) CLIO_CO_RETURN;
  const clio::run::u32 owner = OwnerOf(tag, name);
  if (owner != container_id_) {
    // Standing in for a dead owner: it must learn of the delete (#1130:
    // whatever remote_copies_ is).
    if (clio::cte::core::FailoverContainer(pool_id_, owner) == container_id_) {
      NoteHandoff(owner, tag, name, true);
      CLIO_CO_AWAIT(InvalidateCachedEverywhere(tag, name));
    }
    CLIO_CO_RETURN;
  }
  if (config_.remote_copies_ <= 0) CLIO_CO_RETURN;
  const clio::run::u32 n = NumContainers();
  for (int i = 1; i <= config_.remote_copies_ && i < static_cast<int>(n); ++i) {
    const clio::run::u32 c = (container_id_ + i) % n;
    if (!ContainerAlive(c)) continue;
    auto d = Self()->AsyncDelBlob(tag, name, clio::run::PoolQuery::DirectId(c));
    CLIO_CO_AWAIT(d);  // "not found" is fine: the copy may never have existed
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::TruncateBlob(
    clio::run::shared_ptr<clio::cte::core::TruncateBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  const TagId tag = task->tag_id_;
  const std::string name = task->blob_name_.str();
  const clio::run::u64 new_size = task->new_size_;
  CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kTruncateBlob,
                              task.template Cast<clio::run::Task>()));
  if (task->GetReturnCode() != 0) CLIO_CO_RETURN;
  const clio::run::u32 owner = OwnerOf(tag, name);
  if (owner != container_id_) {
    if (clio::cte::core::FailoverContainer(pool_id_, owner) == container_id_) {
      NoteHandoff(owner, tag, name, false);  // #1130: always
      CLIO_CO_AWAIT(InvalidateCachedEverywhere(tag, name));
    }
    CLIO_CO_RETURN;
  }
  if (config_.remote_copies_ <= 0) CLIO_CO_RETURN;
  const clio::run::u32 n = NumContainers();
  for (int i = 1; i <= config_.remote_copies_ && i < static_cast<int>(n); ++i) {
    const clio::run::u32 c = (container_id_ + i) % n;
    if (!ContainerAlive(c)) continue;
    auto t = Self()->AsyncTruncateBlob(tag, name, new_size,
                                       clio::run::PoolQuery::DirectId(c));
    CLIO_CO_AWAIT(t);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

// ===========================================================================
// Handoff
// ===========================================================================

clio::run::TaskResume Runtime::PushOne(HandoffEntry e, clio::run::u32 owner,
                                       bool *ok) {
  CLIO_TASK_BODY_BEGIN
  *ok = false;
  const auto to_owner = clio::run::PoolQuery::DirectId(owner);
  // No blob lock here: the owner's mirror of this push writes back to this
  // container, and the read below may heal the primary under the same lock
  // -- holding it deadlocked the hand-back. A failover put landing while
  // this copy is taken re-notes the blob with a newer seq_, so PushHandoff
  // keeps it and the next pass sends the finished state (#1154).
  Context push_ctx;
  push_ctx.op_flags_ |= Context::kHandoffPush;
  if (e.deleted_) {
    auto d = Self()->AsyncDelBlob(e.tag_, e.name_, to_owner);
    CLIO_CO_AWAIT(d);
    *ok = true;  // gone at the owner either way
    CLIO_CO_RETURN;
  }
  // Read through THIS container's replication layer, not the raw core: a
  // restart here emptied the RAM-tier primary, and only this layer falls
  // back to the durable replica (the raw read handed back zeros).
  auto sz = Self()->AsyncGetBlobSize(e.tag_, e.name_,
                                     clio::run::PoolQuery::Local(), 0);
  CLIO_CO_AWAIT(sz);
  if (sz->GetReturnCode() != 0) {  // deleted again since: nothing to send
    *ok = true;
    CLIO_CO_RETURN;
  }
  const clio::run::u64 size = sz->size_;
  for (clio::run::u64 off = 0; off < size; off += kHandoffChunk) {
    const clio::run::u64 len = std::min(kHandoffChunk, size - off);
    auto buf = CLIO_IPC->AllocateBuffer(len);
    if (buf.IsNull()) CLIO_CO_RETURN;
    ctp::ipc::ShmPtr<> ptr = buf.shm_.template Cast<void>();
    auto g = Self()->AsyncGetBlob(e.tag_, e.name_, off, len, 0u, ptr,
                                  clio::run::PoolQuery::Local());
    CLIO_CO_AWAIT(g);
    bool good = g->GetReturnCode() == 0;
    if (good) {
      auto p = Self()->AsyncPutBlob(e.tag_, e.name_, off, len, ptr,
                                    config_.cache_score_, push_ctx, 0u,
                                    to_owner);
      CLIO_CO_AWAIT(p);
      good = p->GetReturnCode() == 0;
    }
    CLIO_IPC->FreeBuffer(buf);
    if (!good) CLIO_CO_RETURN;
  }
  {
    // A failover put still extending the blob when its size was taken: the
    // copy is a prefix, and truncating the owner to it would cut the page.
    // Leave the entry for the next pass, which sends the finished state.
    auto sz2 = Self()->AsyncGetBlobSize(e.tag_, e.name_,
                                        clio::run::PoolQuery::Local(), 0);
    CLIO_CO_AWAIT(sz2);
    if (sz2->GetReturnCode() != 0 || sz2->size_ != size) CLIO_CO_RETURN;
  }
  // The owner's stale copy may be longer. Straight to the owner's CORE: the
  // owner's replication layer would mirror the truncate to its remote copy
  // -- this very stand-in -- cutting the stand-in's newer copy short too
  // (#1154).
  auto t = GetCoreClient()->AsyncTruncateBlob(e.tag_, e.name_, size, to_owner);
  CLIO_CO_AWAIT(t);
  *ok = t->GetReturnCode() == 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::PushHandoff(clio::run::u32 owner,
                                           clio::run::u32 *pushed) {
  CLIO_TASK_BODY_BEGIN
  *pushed = 0;
  // One pass per owner at a time: the periodic sweep and the owner's pull
  // pushed the same blobs concurrently, each truncating the other's copy.
  for (;;) {
    {
      std::lock_guard<std::mutex> g(handoff_mu_);
      if (pushing_.insert(owner).second) break;
    }
    CLIO_CO_AWAIT(clio::run::yield(kHandoffWaitUs));
  }
  std::vector<std::pair<std::string, HandoffEntry>> work;
  {
    std::lock_guard<std::mutex> g(handoff_mu_);
    auto it = handoff_.find(owner);
    if (it != handoff_.end()) {
      for (auto &kv : it->second) work.push_back(kv);
    }
  }
  for (const auto &w : work) {
    bool ok = false;
    CLIO_CO_AWAIT(PushOne(w.second, owner, &ok));
    if (!ok) continue;  // retried by the next sweep
    ++*pushed;
    std::lock_guard<std::mutex> g(handoff_mu_);
    auto it = handoff_.find(owner);
    if (it == handoff_.end()) continue;
    auto e = it->second.find(w.first);
    // A newer failover change to the same blob stays for the next round.
    if (e != it->second.end() && e->second.seq_ == w.second.seq_) {
      LogHandoff(kHandoffDone, owner, w.second);
      it->second.erase(e);
    }
    if (it->second.empty()) handoff_.erase(it);
  }
  {
    std::lock_guard<std::mutex> g(handoff_mu_);
    pushing_.erase(owner);
    if (handoff_log_.BytesSinceCompact() > kHandoffCompactBytes) {
      CompactHandoffLogLocked();
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::AwaitHandback(const Context &ctx) {
#ifdef CLIO_ENABLE_BOOST_COROUTINES
  clio::run::shared_ptr<clio::run::Task> cur_task = clio::run::GetCurrentTask();
#endif
  CLIO_TASK_BODY_BEGIN
  if (!handed_back_.load(std::memory_order_acquire) &&
      (ctx.op_flags_ & Context::kHandoffPush) == 0) {
    const auto t0 = std::chrono::steady_clock::now();
    while (!handed_back_.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() - t0 <
               std::chrono::milliseconds(kHandbackWaitMs)) {
      CLIO_CO_AWAIT(clio::run::yield(kHandoffWaitUs));
    }
    if (!handed_back_.load(std::memory_order_acquire)) {
      HLOG(kError, "replication: hand-back still running after {} ms; "
           "serving a client request without it", kHandbackWaitMs);
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::HandoffPull(
    clio::run::shared_ptr<HandoffPullTask> &task) {
  CLIO_TASK_BODY_BEGIN
  clio::run::u32 pushed = 0;
  CLIO_CO_AWAIT(PushHandoff(task->owner_, &pushed));
  task->pushed_ = pushed;
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::HandoffSweep(
    clio::run::shared_ptr<HandoffSweepTask> &task) {
  CLIO_TASK_BODY_BEGIN
  std::vector<clio::run::u32> owners;
  {
    std::lock_guard<std::mutex> g(handoff_mu_);
    for (const auto &kv : handoff_) owners.push_back(kv.first);
  }
  for (clio::run::u32 owner : owners) {
    if (!ContainerAlive(owner)) continue;
    clio::run::u32 pushed = 0;
    CLIO_CO_AWAIT(PushHandoff(owner, &pushed));
    if (pushed != 0) {
      HLOG(kInfo, "replication: handed {} blob(s) back to container {}",
           pushed, owner);
    }
  }
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

}  // namespace clio::cte::replication
