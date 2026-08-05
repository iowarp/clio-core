/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */
#include <algorithm>
#include <string>
#include <vector>

#include <clio_cte/replication/replication_runtime.h>

namespace clio::cte::replication {

/**
 * Chunk size for primary → replica copies. Bounds the bounce buffer for
 * arbitrarily large blobs; each chunk is one GetBlob + one replica-targeted
 * PutBlob, and the CTE serializes each op under the blob's write token.
 */
static constexpr clio::run::u64 kReplicateChunkBytes = 4ULL * 1024 * 1024;

clio::run::TaskResume Runtime::Create(clio::run::shared_ptr<CreateTask> &task) {
  CLIO_TASK_BODY_BEGIN
  config_ = task->GetParams();
  interposer_next_pool_ = config_.next_pool_id_;  // base forwarding target
  if (!config_.next_pool_id_.IsNull()) {
    core_client_ =
        std::make_unique<clio::cte::core::Client>(config_.next_pool_id_);
  }
  // Async write-through (issue #886): spawn the periodic sweep that brings
  // replicas up to date with acked primaries. Fire-and-forget like the
  // core's periodic StatTargets/FlushMetadata drivers.
  if (config_.num_replicas_ > 0 && config_.replicate_period_ms_ > 0) {
    auto *ipc = CLIO_CPU_IPC;
    auto sweep = ipc->NewTask<ReplicateSweepTask>(
        clio::run::CreateTaskId(), pool_id_, clio::run::PoolQuery::Local());
    sweep->SetPeriod(static_cast<double>(config_.replicate_period_ms_),
                     clio::run::kMilli);
    sweep->SetFlags(TASK_PERIODIC);  // SetPeriod alone does not mark it
    ipc->Send(sweep);
    HLOG(kInfo, "replication: async write-through sweep every {} ms",
         config_.replicate_period_ms_);
  }
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Destroy(clio::run::shared_ptr<DestroyTask> &task) {
  CLIO_TASK_BODY_BEGIN
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

clio::cte::core::Client *Runtime::GetCoreClient() {
  if (!core_client_) {
    core_client_ = std::make_unique<clio::cte::core::Client>(CorePoolId());
  }
  return core_client_.get();
}

void Runtime::EnqueueReplication(const TagId &tag_id,
                                 const std::string &blob_name) {
  std::string key = std::to_string(tag_id.major_) + "." +
                    std::to_string(tag_id.minor_) + "." + blob_name;
  std::lock_guard<std::mutex> lk(pending_mtx_);
  pending_[std::move(key)] = std::make_pair(tag_id, blob_name);
}

clio::run::TaskResume Runtime::ReplicateSweep(
    clio::run::shared_ptr<ReplicateSweepTask> &task) {
#ifdef CLIO_ENABLE_BOOST_COROUTINES
  clio::run::shared_ptr<clio::run::Task> cur_task = clio::run::GetCurrentTask();
#endif
  CLIO_TASK_BODY_BEGIN
  task->blobs_swept_ = 0;
  {
    // Swap the dirty set out under the lock; replicate outside it. A put
    // racing the sweep re-inserts its key and is caught next period —
    // removal-before-replication is what makes that safe.
    std::unordered_map<std::string, std::pair<TagId, std::string>> batch;
    {
      std::lock_guard<std::mutex> lk(pending_mtx_);
      batch.swap(pending_);
    }
    for (auto it = batch.begin(); it != batch.end(); ++it) {
      const TagId &tag_id = it->second.first;
      const std::string &blob_name = it->second.second;
      Context rep_ctx;
      rep_ctx.replica_flags_ = clio::cte::core::REPLICA_FIXED |
                               clio::cte::core::REPLICA_PERSISTENT;
      rep_ctx.min_persistence_level_ = 1;
      bool failed = false;
      for (int r = 1; r <= config_.num_replicas_; ++r) {
        clio::run::u64 bytes = 0;
        clio::run::u32 rc = 0;
        CLIO_CO_AWAIT(ReplicateOne(tag_id, blob_name, r, rep_ctx, bytes, rc,
                              config_.replica_score_));
        if (rc == 11) {
          // Blob deleted between the put and this sweep — nothing to keep
          // durable; drop the entry.
          rc = 0;
        }
        if (rc != 0) {
          failed = true;
          break;
        }
      }
      if (failed) {
        // Leave it dirty for the next period; sweeping is best-effort and
        // periodic, so there is no retry loop to spin here.
        std::lock_guard<std::mutex> lk(pending_mtx_);
        pending_[it->first] = it->second;
      } else {
        task->blobs_swept_++;
      }
    }
  }
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::ReplicateOne(
    const TagId &tag_id, const std::string &blob_name,
    int replica_idx, const Context &context,
    clio::run::u64 &bytes_copied, clio::run::u32 &rc, float put_score) {
#ifdef CLIO_ENABLE_BOOST_COROUTINES
  clio::run::shared_ptr<clio::run::Task> cur_task = clio::run::GetCurrentTask();
#endif
  CLIO_TASK_BODY_BEGIN
  rc = 0;
  auto *cte = GetCoreClient();

  clio::run::u64 total = 0;
  {
    auto size_task = cte->AsyncGetBlobSize(tag_id, blob_name);
    CLIO_CO_AWAIT(size_task);
    if (size_task->GetReturnCode() != 0) {
      rc = 10 + size_task->GetReturnCode();
      CLIO_CO_RETURN;
    }
    total = size_task->size_;
  }
  if (total == 0) {
    // Empty primary: nothing to copy. Not an error — a FlushTag sweep may
    // legitimately see just-created blobs.
    CLIO_CO_RETURN;
  }

  for (clio::run::u64 off = 0; off < total; off += kReplicateChunkBytes) {
    clio::run::u64 len = std::min(kReplicateChunkBytes, total - off);
    auto buf = CLIO_IPC->AllocateBuffer(len);
    if (buf.IsNull()) {
      rc = 2;
      CLIO_CO_RETURN;
    }
    ctp::ipc::ShmPtr<> buf_ptr = buf.shm_.template Cast<void>();

    auto get_task = cte->AsyncGetBlob(tag_id, blob_name, off, len,
                                      /*flags=*/0, buf_ptr);
    CLIO_CO_AWAIT(get_task);
    if (get_task->GetReturnCode() != 0) {
      CLIO_IPC->FreeBuffer(buf);
      rc = 20 + get_task->GetReturnCode();
      CLIO_CO_RETURN;
    }

    // Aim the put at the replica, and stamp it with the STORED form's
    // transform flags (the get's OUT context reports them, replica-aware):
    // the copy is byte-for-byte of whatever the primary stores, so a
    // compressed primary yields a compressed — and therefore never
    // fast-pathable — replica (issue #886 per-replica transform flags).
    // Everything else in the caller's context (persistence level, target,
    // preallocation) passes through to place the replica's blocks.
    Context put_ctx = context;
    put_ctx.replica_ = replica_idx;
    put_ctx.transform_flags_ = get_task->context_.transform_flags_;
    auto put_task = cte->AsyncPutBlob(tag_id, blob_name, off, len, buf_ptr,
                                      put_score, put_ctx);
    CLIO_CO_AWAIT(put_task);
    CLIO_IPC->FreeBuffer(buf);
    if (put_task->GetReturnCode() != 0) {
      rc = 30 + put_task->GetReturnCode();
      CLIO_CO_RETURN;
    }
    bytes_copied += len;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::ReplicateBlob(
    clio::run::shared_ptr<ReplicateBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  task->bytes_copied_ = 0;
  if (task->replica_ <= 0 || task->blob_name_.size() == 0) {
    task->return_code_ = 1;
    CLIO_CO_RETURN;
  }
  {
    std::string blob_name = task->blob_name_.str();
    clio::run::u32 rc = 0;
    CLIO_CO_AWAIT(ReplicateOne(task->tag_id_, blob_name, task->replica_,
                          task->context_, task->bytes_copied_, rc));
    task->return_code_ = rc;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::FlushTag(clio::run::shared_ptr<FlushTagTask> &task) {
  CLIO_TASK_BODY_BEGIN
  task->blobs_replicated_ = 0;
  task->bytes_copied_ = 0;
  if (task->replica_ <= 0) {
    task->return_code_ = 1;
    CLIO_CO_RETURN;
  }
  {
    auto *cte = GetCoreClient();

    // Blob names are sharded across containers by HashBlobToContainer, so
    // the listing must be a Broadcast (GetContainedBlobsTask AggregateOut
    // merges the per-container name lists).
    std::vector<std::string> blob_names;
    {
      auto list_task = cte->AsyncGetContainedBlobs(
          task->tag_id_, clio::run::PoolQuery::Broadcast());
      CLIO_CO_AWAIT(list_task);
      if (list_task->GetReturnCode() != 0) {
        task->return_code_ = 40 + list_task->GetReturnCode();
        CLIO_CO_RETURN;
      }
      blob_names = list_task->blob_names_;
    }

    clio::run::u32 first_rc = 0;
    for (size_t i = 0; i < blob_names.size(); ++i) {
      if (task->min_score_ > 0.0f) {
        auto score_task = cte->AsyncGetBlobScore(task->tag_id_, blob_names[i]);
        CLIO_CO_AWAIT(score_task);
        if (score_task->GetReturnCode() != 0 ||
            score_task->score_ < task->min_score_) {
          continue;
        }
      }
      clio::run::u32 rc = 0;
      CLIO_CO_AWAIT(ReplicateOne(task->tag_id_, blob_names[i], task->replica_,
                            task->context_, task->bytes_copied_, rc));
      if (rc != 0) {
        // Keep sweeping — a durability flush should save what it can — but
        // report the first failure so the caller knows the sweep is partial.
        if (first_rc == 0) {
          first_rc = rc;
        }
        HLOG(kWarning, "FlushTag: failed to replicate blob '{}' (rc={})",
             blob_names[i], rc);
        continue;
      }
      task->blobs_replicated_++;
    }
    task->return_code_ = first_rc;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::PutBlob(
    clio::run::shared_ptr<clio::cte::core::PutBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  // Explicit replica addressing (replica-targeted put or kAllReplicas
  // write-through) passes through untouched — the caller is driving the
  // core mechanism directly and this module must not second-guess it.
  if (task->context_.replica_ != 0) {
    CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kPutBlob,
                           task.template Cast<clio::run::Task>()));
    CLIO_CO_RETURN;
  }
  // Primary FIRST, forwarded VERBATIM (score, context, vectored segments,
  // flags all intact): the DRAM cache copy must never serve stale bytes, so
  // it is updated before the durable copies. If a later replica write fails
  // the caller sees the error while reads already see new data — durability
  // lagging is reported; a stale cache never is.
  CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kPutBlob,
                         task.template Cast<clio::run::Task>()));
  if (task->GetReturnCode() != 0) {
    CLIO_CO_RETURN;
  }
  {
    auto *cte = GetCoreClient();
    std::string blob_name = task->blob_name_.str();

    // ASYNC write-through (replicate_period_ms_ > 0, the default): the put
    // acks NOW, after the primary; the blob is marked dirty and the
    // periodic sweep copies its then-current primary bytes into the
    // replica set. Rapid overwrites coalesce into one replica write.
    if (config_.replicate_period_ms_ > 0) {
      EnqueueReplication(task->tag_id_, blob_name);
      task->return_code_ = 0;
      CLIO_CO_RETURN;
    }

    // Synchronous write-through (replicate_period_ms_ == 0). Each copy is
    // pinned (REPLICA_FIXED) and volatile-banned (REPLICA_PERSISTENT); the
    // configured replica_score_ keeps the durable copies scored for the
    // slow tier they live on. Vectored puts replay their segments in list
    // order (same last-writer-wins rule as the core's primary path).
    for (int i = 1; i <= config_.num_replicas_; ++i) {
      Context rep_ctx = task->context_;
      rep_ctx.replica_ = i;
      rep_ctx.replica_flags_ =
          task->context_.replica_flags_ | clio::cte::core::REPLICA_FIXED |
          clio::cte::core::REPLICA_PERSISTENT;
      if (rep_ctx.min_persistence_level_ < 1) {
        rep_ctx.min_persistence_level_ = 1;
      }
      // Shared scalar-vs-vectored iteration (blob_batch.h): collect the
      // regions once, then one replica-targeted put per region in list
      // order (the core's last-writer-wins rule).
      std::vector<clio::cte::core::BlobRegion> regions;
      clio::cte::core::ForEachBlobRegion(*task,
          [&regions](const clio::cte::core::BlobRegion &r) {
            regions.push_back(r);
            return true;
          });
      for (size_t ri = 0; ri < regions.size(); ++ri) {
        auto put = cte->AsyncPutBlob(task->tag_id_, blob_name,
                                     regions[ri].blob_off_, regions[ri].size_,
                                     regions[ri].data_,
                                     config_.replica_score_, rep_ctx);
        CLIO_CO_AWAIT(put);
        if (put->GetReturnCode() != 0) {
          task->return_code_ = 30 + put->GetReturnCode();
          CLIO_CO_RETURN;
        }
      }
    }
    task->return_code_ = 0;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::RecachePrimary(const TagId &tag_id,
                                              const std::string &blob_name,
                                              int replica_idx,
                                              clio::run::u64 rep_size,
                                              clio::run::u64 &recached) {
#ifdef CLIO_ENABLE_BOOST_COROUTINES
  clio::run::shared_ptr<clio::run::Task> cur_task = clio::run::GetCurrentTask();
#endif
  CLIO_TASK_BODY_BEGIN
  auto *cte = GetCoreClient();
  for (clio::run::u64 off = 0; off < rep_size; off += kReplicateChunkBytes) {
    clio::run::u64 len = std::min(kReplicateChunkBytes, rep_size - off);
    auto buf = CLIO_IPC->AllocateBuffer(len);
    if (buf.IsNull()) {
      CLIO_CO_RETURN;  // best-effort: keep the valid prefix
    }
    ctp::ipc::ShmPtr<> buf_ptr = buf.shm_.template Cast<void>();

    Context get_ctx;
    get_ctx.replica_ = replica_idx;
    auto get_task = cte->AsyncGetBlob(tag_id, blob_name, off, len,
                                      /*flags=*/0, buf_ptr,
                                      clio::run::PoolQuery::Dynamic(),
                                      get_ctx);
    CLIO_CO_AWAIT(get_task);
    if (get_task->GetReturnCode() != 0) {
      CLIO_IPC->FreeBuffer(buf);
      CLIO_CO_RETURN;
    }

    Context put_ctx;
    put_ctx.replica_ = 0;
    put_ctx.min_persistence_level_ = 0;
    auto put_task = cte->AsyncPutBlob(tag_id, blob_name, off, len, buf_ptr,
                                      config_.cache_score_, put_ctx);
    CLIO_CO_AWAIT(put_task);
    CLIO_IPC->FreeBuffer(buf);
    if (put_task->GetReturnCode() != 0) {
      // No room in the fast tiers (or any tier): stop. The sequential-from-0
      // order means everything already copied is a valid prefix.
      CLIO_CO_RETURN;
    }
    recached += len;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::GetBlob(
    clio::run::shared_ptr<clio::cte::core::GetBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  // Explicit replica addressing passes through untouched.
  if (task->context_.replica_ != 0) {
    CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kGetBlob,
                           task.template Cast<clio::run::Task>()));
    CLIO_CO_RETURN;
  }
  {
    auto *cte = GetCoreClient();
    std::string blob_name = task->blob_name_.str();
    // The range a copy must COVER to serve this read: the scalar range, or
    // the union of a vectored read's segments (the same union the core's
    // torn-layout guard uses).
    clio::run::u64 req_lo = 0, end = 0;
    clio::cte::core::BlobRequestRange(*task, &req_lo, &end);
    (void)req_lo;
    clio::run::u64 served_total = 0;  // full size of whatever copy served

    // 1. Primary (the LOCAL core container — the task was routed here by
    //    the same hash the core pool uses, so this is the blob's owner):
    //    cache hit iff it COVERS the requested range. A dropped primary
    //    reads as size 0; an interrupted re-cache leaves a prefix that
    //    still hits for ranges inside it. Forwarding the ORIGINAL task
    //    keeps vectored segments, flags and OUT-context reporting intact.
    clio::run::u64 primary_size = 0;
    {
      auto size_task = cte->AsyncGetBlobSize(task->tag_id_, blob_name,
                                             clio::run::PoolQuery::Local());
      CLIO_CO_AWAIT(size_task);
      if (size_task->GetReturnCode() == 0) {
        primary_size = size_task->size_;
      }
    }
    if (primary_size >= end) {
      CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kGetBlob,
                             task.template Cast<clio::run::Task>()));
      CLIO_CO_RETURN;
    }

    // 2. Miss: serve from the first persistent replica that covers the
    //    range (the post-drop state), then restore the DRAM fast path by
    //    copying the WHOLE replica back into the primary (best-effort).
    bool served = false;
    for (int r = 1; r <= config_.num_replicas_ && !served; ++r) {
      clio::run::u64 rep_size = 0;
      {
        auto size_task = cte->AsyncGetBlobSize(
            task->tag_id_, blob_name, clio::run::PoolQuery::Local(), r);
        CLIO_CO_AWAIT(size_task);
        if (size_task->GetReturnCode() != 0) {
          continue;
        }
        rep_size = size_task->size_;
      }
      if (rep_size < end) {
        continue;
      }
      task->context_.replica_ = r;
      CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kGetBlob,
                             task.template Cast<clio::run::Task>()));
      task->context_.replica_ = 0;
      if (task->GetReturnCode() != 0) {
        continue;
      }
      clio::run::u64 recached = 0;
      CLIO_CO_AWAIT(RecachePrimary(task->tag_id_, blob_name, r, rep_size,
                              recached));
      served = true;
      served_total = rep_size;
    }
    if (!served) {
      // Nothing covers the range: forward to the core so the caller gets
      // the pool's NATIVE semantics for absent blobs and short ranges —
      // interposition must not invent its own error space.
      CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kGetBlob,
                             task.template Cast<clio::run::Task>()));
      CLIO_CO_RETURN;
    }

    // 3. Populate this node's local cache (issue #886 distributed
    //    coherence). Re-probe first: after the re-cache above the local
    //    primary usually covers, and copying onto ourselves would be a
    //    pointless duplicate write. Only a genuinely remote reader falls
    //    through to CacheLocalCopy + registration.
    {
      auto lsize = cte->AsyncGetBlobSize(task->tag_id_, blob_name,
                                         clio::run::PoolQuery::Local());
      CLIO_CO_AWAIT(lsize);
      if (!(lsize->GetReturnCode() == 0 && lsize->size_ >= served_total)) {
        bool cached = false;
        CLIO_CO_AWAIT(CacheLocalCopy(task->tag_id_, blob_name, served_total,
                                cached));
        if (cached) {
          auto reg = cte->AsyncRegisterReplicaContainer(
              task->tag_id_, blob_name, CLIO_IPC->GetNodeId());
          CLIO_CO_AWAIT(reg);
        }
      }
    }
    task->return_code_ = 0;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::MultiPutBlob(
    clio::run::shared_ptr<clio::cte::core::MultiPutBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  // Explicit replica addressing passes through untouched (same gate as the
  // scalar PutBlob): e.g. the cache layer's kCacheReplica-aimed batches are
  // internal replica writes, not primary puts to replicate.
  if (task->context_.replica_ != 0) {
    CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kMultiPutBlob,
                           task.template Cast<clio::run::Task>()));
    CLIO_CO_RETURN;
  }
  // Primary batch on the core, verbatim (zero-copy slices, one completion).
  CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kMultiPutBlob,
                         task.template Cast<clio::run::Task>()));
  if (task->GetReturnCode() != 0) {
    CLIO_CO_RETURN;
  }
  {
    auto *cte = GetCoreClient();
    // Shared batch decode (blob_batch.h — the same view the core executes);
    // the slices stay zero-copy: the batch buffer outlives this task, and
    // each replica put completes before the next is issued.
    clio::cte::core::MultiPutBatchView batch;
    if (clio::cte::core::MultiPutBatchView::Attach(*task, &batch) &&
        config_.replicate_period_ms_ > 0) {
      // ASYNC write-through: mark every record dirty for the periodic sweep
      // and ack the batch now (see PutBlob).
      for (size_t d = 0; d < batch.size(); ++d) {
        EnqueueReplication(batch.descs_[d].tag_id_, batch.descs_[d].blob_name_);
      }
    } else if (batch.base_ != nullptr) {
      for (size_t d = 0; d < batch.size(); ++d) {
        const auto &desc = batch.descs_[d];
        if (!batch.RecordValid(d)) {
          continue;  // malformed entry — already counted by the core batch
        }
        ctp::ipc::ShmPtr<> slice = batch.RecordSlice(d);
        for (int r = 1; r <= config_.num_replicas_; ++r) {
          // Same context derivation as the scalar sync path: the batch
          // context passes through (persistence, preallocation, transform)
          // with only the replica addressing overridden.
          Context rep_ctx = task->context_;
          rep_ctx.replica_ = r;
          rep_ctx.replica_flags_ = task->context_.replica_flags_ |
                                   clio::cte::core::REPLICA_FIXED |
                                   clio::cte::core::REPLICA_PERSISTENT;
          if (rep_ctx.min_persistence_level_ < 1) {
            rep_ctx.min_persistence_level_ = 1;
          }
          auto put = cte->AsyncPutBlob(desc.tag_id_, desc.blob_name_,
                                       desc.offset_, desc.size_, slice,
                                       config_.replica_score_, rep_ctx);
          CLIO_CO_AWAIT(put);
          if (put->GetReturnCode() != 0 && task->first_rc_ == 0) {
            task->first_rc_ = 30 + put->GetReturnCode();
            task->SetReturnCode(task->first_rc_);
          }
        }
      }
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::GetBlobSize(
    clio::run::shared_ptr<clio::cte::core::GetBlobSizeTask> &task) {
  CLIO_TASK_BODY_BEGIN
  // Explicit replica probes pass through untouched.
  if (task->replica_ != 0) {
    CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kGetBlobSize,
                           task.template Cast<clio::run::Task>()));
    CLIO_CO_RETURN;
  }
  CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kGetBlobSize,
                         task.template Cast<clio::run::Task>()));
  // A dropped primary reads as size 0 while its replicas still hold the
  // bytes. Size-then-read callers key "absent" off this, so report the
  // LOGICAL size: the best replica's. (A blob that never existed fails the
  // forward above with the core's native rc and never reaches here.)
  if (task->GetReturnCode() == 0 && task->size_ == 0) {
    auto *cte = GetCoreClient();
    std::string blob_name = task->blob_name_.str();
    for (int r = 1; r <= config_.num_replicas_; ++r) {
      auto size_task = cte->AsyncGetBlobSize(
          task->tag_id_, blob_name, clio::run::PoolQuery::Local(), r);
      CLIO_CO_AWAIT(size_task);
      if (size_task->GetReturnCode() == 0 && size_task->size_ > task->size_) {
        task->size_ = size_task->size_;
      }
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::CacheLocalCopy(const TagId &tag_id,
                                              const std::string &blob_name,
                                              clio::run::u64 total,
                                              bool &cached) {
#ifdef CLIO_ENABLE_BOOST_COROUTINES
  clio::run::shared_ptr<clio::run::Task> cur_task = clio::run::GetCurrentTask();
#endif
  CLIO_TASK_BODY_BEGIN
  cached = false;
  auto *cte = GetCoreClient();
  for (clio::run::u64 off = 0; off < total; off += kReplicateChunkBytes) {
    clio::run::u64 len = std::min(kReplicateChunkBytes, total - off);
    auto buf = CLIO_IPC->AllocateBuffer(len);
    if (buf.IsNull()) {
      CLIO_CO_RETURN;
    }
    ctp::ipc::ShmPtr<> buf_ptr = buf.shm_.template Cast<void>();

    auto get_task = cte->AsyncGetBlob(tag_id, blob_name, off, len,
                                      /*flags=*/0, buf_ptr);
    CLIO_CO_AWAIT(get_task);
    if (get_task->GetReturnCode() != 0) {
      CLIO_IPC->FreeBuffer(buf);
      CLIO_CO_RETURN;
    }

    Context put_ctx;
    put_ctx.replica_ = 0;
    put_ctx.min_persistence_level_ = 0;
    auto put_task = cte->AsyncPutBlob(tag_id, blob_name, off, len, buf_ptr,
                                      config_.cache_score_, put_ctx,
                                      /*flags=*/0,
                                      clio::run::PoolQuery::Local());
    CLIO_CO_AWAIT(put_task);
    CLIO_IPC->FreeBuffer(buf);
    if (put_task->GetReturnCode() != 0) {
      CLIO_CO_RETURN;  // abandon; partial local copies are never served
    }
  }
  cached = true;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

}  // namespace clio::cte::replication

// Define ChiMod entry points (alloc/new/name/destroy) so the runtime's module
// manager can dlopen and instantiate this chimod.
CLIO_TASK_CC(clio::cte::replication::Runtime)
