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
/** Poll period while a SyncTag barrier waits for a running sweep. */
static constexpr double kSweepWaitUs = 200.0;

clio::run::TaskResume Runtime::Create(clio::run::shared_ptr<CreateTask> &task) {
  CLIO_TASK_BODY_BEGIN
  // The task is NOT a CreateTask, whatever the parameter says. The generated
  // dispatch reinterprets whatever create task the runtime is holding into
  // this ChiMod's instantiation, and what it is actually holding depends on
  // who asked for the pool -- compose builds ComposeTask<PoolConfig>, a direct
  // caller builds its own. All of them ARE a CreatePoolFields, so reads go
  // through that; the config type is named at the call instead of being baked
  // into the object's type. Touching the task through `task` itself would be
  // undefined behaviour, which is what UBSan reports here.
  auto &fields = task.template Cast<clio::run::admin::CreatePoolFields>();
  config_ = fields->GetParamsAs<ReplicationConfig>();
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
  // Failover hand-back runs whenever there is another container to stand
  // in (#1130), not only with remote copies: with replication_factor 1 the
  // stand-in holds the only copy of what was written during an outage.
  if (NumContainers() > 1) {
    OpenHandoffLog();
    auto *ipc = CLIO_CPU_IPC;
    auto sweep = ipc->NewTask<HandoffSweepTask>(
        clio::run::CreateTaskId(), pool_id_, clio::run::PoolQuery::Local());
    sweep->SetPeriod(500.0, clio::run::kMilli);
    sweep->SetFlags(TASK_PERIODIC);
    ipc->Send(sweep);
    if (is_restart_) {
      // Pull what the successors changed while this node was down before
      // serving: otherwise it would answer with its stale copies.
      // Ask every other container: whichever stood in (the first live
      // successor at the time, which the remote-copy count does not bound).
      const clio::run::u32 n = NumContainers();
      for (clio::run::u32 i = 1; i < n; ++i) {
        const clio::run::u32 c = (container_id_ + i) % n;
        if (!ContainerAlive(c)) continue;  // a dead one cannot answer
        auto pull = ipc->NewTask<HandoffPullTask>(
            clio::run::CreateTaskId(), fields->new_pool_id_,
            clio::run::PoolQuery::DirectId(c), container_id_);
        auto f = ipc->Send(pull);
        CLIO_CO_AWAIT(f);
        HLOG(kInfo, "replication: container {} handed back {} blob(s) "
             "(rc {})", c, f->pushed_, f->GetReturnCode());
      }
    }
  }
  handed_back_.store(true, std::memory_order_release);
  fields->return_code_ = 0;
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
  // "replicate_period_ms": lets a data-path client (the FUSE adapter) learn,
  // once at mount time, whether this pool's write-through is synchronous (0
  // -- PutBlob already blocks on the replica write, so fsync needs no extra
  // barrier) or asynchronous (>0 -- fsync must force a FlushTag sweep of the
  // file's tag before the durability contract holds). See fuse_cte.cc's
  // NeedsReplicationFlushBarrier().
  if (task->query_ == "replicate_period_ms") {
    task->results_[container_id_] = std::to_string(config_.replicate_period_ms_);
  }
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
  const clio::run::u64 sweep_id = sweeps_started_.fetch_add(1) + 1;
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
      bool ok = false;
      CLIO_CO_AWAIT(ReplicateAllCopies(it->second.first, it->second.second,
                                       ok));
      if (!ok) {
        // Leave it dirty for the next period; sweeping is best-effort and
        // periodic, so there is no retry loop to spin here.
        std::lock_guard<std::mutex> lk(pending_mtx_);
        pending_[it->first] = it->second;
      } else {
        task->blobs_swept_++;
      }
    }
  }
  sweeps_done_.store(sweep_id);
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::ReplicateAllCopies(const TagId &tag_id,
                                                  const std::string &blob_name,
                                                  bool &ok) {
#ifdef CLIO_ENABLE_BOOST_COROUTINES
  clio::run::shared_ptr<clio::run::Task> cur_task = clio::run::GetCurrentTask();
#endif
  CLIO_TASK_BODY_BEGIN
  ok = true;
  Context rep_ctx;
  rep_ctx.replica_flags_ = clio::cte::core::REPLICA_FIXED |
                           clio::cte::core::REPLICA_PERSISTENT;
  rep_ctx.min_persistence_level_ = 1;
  for (int r = 1; r <= config_.num_replicas_; ++r) {
    clio::run::u64 bytes = 0;
    clio::run::u32 rc = 0;
    CLIO_CO_AWAIT(ReplicateOne(tag_id, blob_name, r, rep_ctx, bytes, rc,
                               config_.replica_score_));
    // 11: the blob was deleted after the put; nothing to keep durable.
    if (rc != 0 && rc != 11) {
      ok = false;
      break;
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::FlushTagReplicas(const TagId &tag_id,
                                                bool &current) {
#ifdef CLIO_ENABLE_BOOST_COROUTINES
  clio::run::shared_ptr<clio::run::Task> cur_task = clio::run::GetCurrentTask();
#endif
  CLIO_TASK_BODY_BEGIN
  current = true;
  {
    // A running sweep swapped its batch out of pending_ and may still be
    // copying this tag's blobs: wait for it, not for later sweeps.
    const clio::run::u64 running = sweeps_started_.load();
    while (sweeps_done_.load() < running) {
      CLIO_CO_AWAIT(clio::run::yield(kSweepWaitUs));
    }
  }
  std::vector<std::pair<std::string, std::pair<TagId, std::string>>> mine;
  {
    std::lock_guard<std::mutex> lk(pending_mtx_);
    for (auto it = pending_.begin(); it != pending_.end();) {
      if (it->second.first == tag_id) {
        mine.emplace_back(it->first, it->second);
        it = pending_.erase(it);
      } else {
        ++it;
      }
    }
  }
  for (size_t i = 0; i < mine.size(); ++i) {
    bool ok = false;
    CLIO_CO_AWAIT(ReplicateAllCopies(mine[i].second.first,
                                     mine[i].second.second, ok));
    if (!ok) {
      current = false;
      std::lock_guard<std::mutex> lk(pending_mtx_);
      pending_[mine[i].first] = mine[i].second;  // the sweep retries it
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::SyncTag(
    clio::run::shared_ptr<clio::cte::core::SyncTagTask> &task) {
  CLIO_TASK_BODY_BEGIN
  if (config_.num_replicas_ > 0) {
    bool current = false;
    CLIO_CO_AWAIT(FlushTagReplicas(task->tag_id_, current));
    if (current) task->sync_flags_ |= clio::cte::core::kSyncReplicasCurrent;
  }
  CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kSyncTag,
                              task.template Cast<clio::run::Task>()));
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
    auto size_task = cte->AsyncGetBlobSize(tag_id, blob_name,
                                          clio::run::PoolQuery::Local());
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
                                      /*flags=*/0, buf_ptr,
                                      clio::run::PoolQuery::Local());
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
                                      put_score, put_ctx, 0u,
                                      clio::run::PoolQuery::Local());
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

clio::run::TaskResume Runtime::PutBlobLocal(
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
  // One writer per blob through refill, primary and replicas (LockBlobs).
  struct Unlocker {
    Runtime *rt;
    std::vector<std::string> keys;
    ~Unlocker() { rt->UnlockBlobs(keys); }
  };
  std::vector<std::string> put_keys{
      BlobKey(task->tag_id_, task->blob_name_.str())};
  CLIO_CO_AWAIT(LockBlobs(put_keys));
  Unlocker put_unlock{this, put_keys};
  {
    // A write past the primary's end must not re-grow a primary that lost
    // its bytes (see RefillPrimaryBeforeWrite).
    clio::run::u64 lowest = ~0ULL;
    clio::cte::core::ForEachBlobRegion(*task,
        [&lowest](const clio::cte::core::BlobRegion &r) {
          lowest = std::min(lowest, r.blob_off_);
          return true;
        });
    if (lowest != ~0ULL && lowest > 0) {
      bool refilled = true;
      CLIO_CO_AWAIT(RefillPrimaryBeforeWrite(
          task->tag_id_, task->blob_name_.str(), lowest, refilled));
      if (!refilled) {
        // The replica's bytes below the write could not be put back in the
        // primary: writing now would shadow them with zeros. Out of space.
        task->return_code_ = 10 + clio::cte::core::kPutNoSpaceRc;
        CLIO_CO_RETURN;
      }
    }
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
        // Local: this container holds the blob (its owner, or the shadow
        // copy it keeps for a successor role).
        auto put = cte->AsyncPutBlob(task->tag_id_, blob_name,
                                     regions[ri].blob_off_, regions[ri].size_,
                                     regions[ri].data_,
                                     config_.replica_score_, rep_ctx, 0u,
                                     clio::run::PoolQuery::Local());
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

clio::run::TaskResume Runtime::RefillPrimaryBeforeWrite(
    const TagId &tag_id, const std::string &blob_name,
    clio::run::u64 write_off, bool &ok) {
#ifdef CLIO_ENABLE_BOOST_COROUTINES
  clio::run::shared_ptr<clio::run::Task> cur_task = clio::run::GetCurrentTask();
#endif
  CLIO_TASK_BODY_BEGIN
  ok = true;
  auto *cte = GetCoreClient();
  auto prim = cte->AsyncGetBlobSize(tag_id, blob_name,
                                    clio::run::PoolQuery::Local(), 0);
  CLIO_CO_AWAIT(prim);
  const clio::run::u64 prim_size =
      prim->GetReturnCode() == 0 ? prim->size_ : 0;
  if (prim_size >= write_off) {
    CLIO_CO_RETURN;  // the common case: the write extends a whole primary
  }
  for (int r = 1; r <= config_.num_replicas_; ++r) {
    auto rs = cte->AsyncGetBlobSize(tag_id, blob_name,
                                    clio::run::PoolQuery::Local(), r);
    CLIO_CO_AWAIT(rs);
    if (rs->GetReturnCode() != 0 || rs->size_ <= prim_size) continue;
    clio::run::u64 recached = 0;
    CLIO_CO_AWAIT(RecachePrimary(tag_id, blob_name, r, rs->size_, recached));
    if (recached < std::min(rs->size_, write_off)) {
      HLOG(kWarning, "replication: refilling primary {}.{}/{} from replica "
           "{} stopped at {} of {} bytes; the write below offset {} is "
           "refused (ENOSPC)", tag_id.major_, tag_id.minor_, blob_name, r,
           recached, rs->size_, write_off);
      ok = false;
    }
    break;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::LockBlobs(std::vector<std::string> keys) {
  CLIO_TASK_BODY_BEGIN
  std::sort(keys.begin(), keys.end());
  keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
  {
    const auto t0 = std::chrono::steady_clock::now();
    double next_report_s = 10.0;
    for (;;) {
      std::string busy_key;
      double held_s = 0;
      {
        std::lock_guard<std::mutex> g(blob_busy_mu_);
        const auto now = std::chrono::steady_clock::now();
        for (const auto &k : keys) {
          auto it = blob_busy_.find(k);
          if (it != blob_busy_.end()) {
            busy_key = k;
            held_s = std::chrono::duration<double>(now - it->second).count();
            break;
          }
        }
        if (busy_key.empty()) {
          for (const auto &k : keys) blob_busy_[k] = now;
          CLIO_CO_RETURN;
        }
      }
      const double waited = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - t0).count();
      if (waited >= next_report_s) {
        // Held this long is a stuck holder (e.g. a put awaiting a remote
        // copy that never answers), not contention.
        HLOG(kError, "replication [HANGWATCH-BLOB] waited {} ms for blob {} "
             "(held for {} ms)", waited * 1000.0, busy_key, held_s * 1000.0);
        next_report_s *= 2;
      }
      CLIO_CO_AWAIT(clio::run::yield(20.0));
    }
  }
  CLIO_TASK_BODY_END
}

void Runtime::UnlockBlobs(const std::vector<std::string> &keys) {
  std::lock_guard<std::mutex> g(blob_busy_mu_);
  for (const auto &k : keys) blob_busy_.erase(k);
}

clio::run::TaskResume Runtime::RecachePrimary(
    const TagId &tag_id, const std::string &blob_name, int replica_idx,
    clio::run::u64 rep_size, clio::run::u64 &recached,
    const clio::run::PoolQuery &from) {
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
                                      /*flags=*/0, buf_ptr, from, get_ctx);
    CLIO_CO_AWAIT(get_task);
    if (get_task->GetReturnCode() != 0) {
      CLIO_IPC->FreeBuffer(buf);
      CLIO_CO_RETURN;
    }

    Context put_ctx;
    put_ctx.replica_ = 0;
    put_ctx.min_persistence_level_ = 0;
    auto put_task = cte->AsyncPutBlob(tag_id, blob_name, off, len, buf_ptr,
                                      config_.cache_score_, put_ctx, 0u,
                                      clio::run::PoolQuery::Local());
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

clio::run::TaskResume Runtime::HealPrimaryFromRemote(
    const TagId &tag_id, const std::string &blob_name,
    clio::run::u32 remote_c, clio::run::u64 end) {
#ifdef CLIO_ENABLE_BOOST_COROUTINES
  clio::run::shared_ptr<clio::run::Task> cur_task = clio::run::GetCurrentTask();
#endif
  CLIO_TASK_BODY_BEGIN
  auto *cte = GetCoreClient();
  const clio::run::PoolQuery at = clio::run::PoolQuery::DirectId(remote_c);
  clio::run::u64 remote_size = 0;
  {
    auto st = cte->AsyncGetBlobSize(tag_id, blob_name, at, 0);
    CLIO_CO_AWAIT(st);
    if (st->GetReturnCode() != 0 || st->size_ < end) CLIO_CO_RETURN;
    remote_size = st->size_;
  }
  // Under the blob's write token: the re-cache re-reads the remote copy and
  // must not land over a write that refills the primary meanwhile.
  std::vector<std::string> keys{BlobKey(tag_id, blob_name)};
  CLIO_CO_AWAIT(LockBlobs(keys));
  {
    auto ps = cte->AsyncGetBlobSize(tag_id, blob_name,
                                    clio::run::PoolQuery::Local(), 0);
    CLIO_CO_AWAIT(ps);
    if (!(ps->GetReturnCode() == 0 && ps->size_ >= end)) {
      clio::run::u64 recached = 0;
      CLIO_CO_AWAIT(RecachePrimary(tag_id, blob_name, 0, remote_size,
                                   recached, at));
      HLOG(kWarning, "replication: {}.{}/{} primary re-cached from the remote "
           "copy on container {}: {} of {} byte(s)",
           tag_id.major_, tag_id.minor_, blob_name, remote_c, recached,
           remote_size);
    }
  }
  UnlockBlobs(keys);
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
    bool primary_unreadable = false;
    if (primary_size >= end) {
      CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kGetBlob,
                             task.template Cast<clio::run::Task>()));
      if (task->GetReturnCode() != clio::cte::core::kGetBlobIoErrorRc ||
          (config_.num_replicas_ <= 0 && config_.remote_copies_ <= 0)) {
        CLIO_CO_RETURN;
      }
      // The primary's blocks did not come back (their device or node is
      // down): serve the read from a replica instead, and do not re-cache
      // into a primary whose blocks are unreachable.
      primary_unreadable = true;
      task->SetReturnCode(0);
    }

    // 2. Miss: serve from the first persistent replica that covers the
    //    range (the post-drop state), then restore the DRAM fast path by
    //    copying the WHOLE replica back into the primary (best-effort).
    bool served = false;
    static const bool heal_trace = getenv("CLIO_HEAL_TRACE") != nullptr;
    if (heal_trace) {
      fprintf(stderr, "[heal] blob=%s primary_size=%llu end=%llu replicas=%d\n",
              blob_name.c_str(), (unsigned long long)primary_size,
              (unsigned long long)end, config_.num_replicas_);
    }
    for (int r = 1; r <= config_.num_replicas_ && !served; ++r) {
      clio::run::u64 rep_size = 0;
      {
        auto size_task = cte->AsyncGetBlobSize(
            task->tag_id_, blob_name, clio::run::PoolQuery::Local(), r);
        CLIO_CO_AWAIT(size_task);
        if (heal_trace) {
          fprintf(stderr, "[heal]   replica %d rc=%d size=%llu\n", r,
                  (int)size_task->GetReturnCode(),
                  (unsigned long long)size_task->size_);
        }
        if (size_task->GetReturnCode() != 0) {
          continue;
        }
        rep_size = size_task->size_;
      }
      // COVERAGE IS A STORED-BYTES QUESTION, AND `end` IS LOGICAL.
      //
      // `rep_size` is the replica's STORED size; `end` is the end of the
      // range the CALLER asked for, in the blob's logical bytes. For an
      // untransformed copy those are the same currency and comparing them
      // is right: a replica holding a prefix must not serve past it.
      //
      // For a TRANSFORMED copy they are not. A 64KB blob stored compressed
      // in 177 bytes made every replica look "too small" (177 < 65536), so
      // no replica could ever heal a compressed blob -- the read fell
      // through to a core whose primary had just been dropped and returned
      // an error. That is the #886 stack case: drop the primary, and the
      // durable replica that exists to survive exactly that could not be
      // used.
      //
      // A transformed replica holds the blob's WHOLE stored image, so it
      // can serve any logical range -- the codec above expands it. We only
      // learn the transform state from the get itself (context_ reports it
      // OUT), so a short replica is ATTEMPTED and then accepted only if it
      // really was transformed. Untransformed short replicas are rejected
      // exactly as before, without an extra round trip.
      const bool covers_logical = rep_size >= end;
      task->context_.replica_ = r;
      task->context_.transform_flags_ = 0;
      CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kGetBlob,
                             task.template Cast<clio::run::Task>()));
      const clio::run::u32 rep_transform = task->context_.transform_flags_;
      task->context_.replica_ = 0;
      if (task->GetReturnCode() != 0) {
        continue;
      }
      if (!covers_logical && rep_transform == 0) {
        if (heal_trace) {
          fprintf(stderr,
                  "[heal]   replica %d TOO SMALL and untransformed "
                  "(%llu < %llu)\n",
                  r, (unsigned long long)rep_size, (unsigned long long)end);
        }
        continue;
      }
      if (heal_trace) {
        fprintf(stderr, "[heal]   replica %d SERVED (stored %llu, transform "
                "0x%x)\n", r, (unsigned long long)rep_size, rep_transform);
      }
      clio::run::u64 recached = 0;
      if (!primary_unreadable) {
        // Under the blob's write token: the re-cache re-reads the replica
        // and must not copy it over a write that lands meanwhile.
        std::vector<std::string> heal_keys{BlobKey(task->tag_id_, blob_name)};
        CLIO_CO_AWAIT(LockBlobs(heal_keys));
        auto ps = cte->AsyncGetBlobSize(task->tag_id_, blob_name,
                                        clio::run::PoolQuery::Local(), 0);
        CLIO_CO_AWAIT(ps);
        // A writer refilled (or rewrote) the primary meanwhile: keep it.
        if (!(ps->GetReturnCode() == 0 && ps->size_ >= end)) {
          CLIO_CO_AWAIT(RecachePrimary(task->tag_id_, blob_name, r, rep_size,
                                       recached));
        }
        UnlockBlobs(heal_keys);
      }
      served = true;
      served_total = rep_size;
    }
    if (!served && config_.remote_copies_ > 0 &&
        (primary_unreadable || primary_size < end)) {
      // Another node holds a copy, in two cases. Every local copy sits on a
      // dead device (#1114): serve from the remote, no re-cache -- the
      // primary's blocks are unreachable. Or the primary is SHORT of the
      // range on a healthy device (#1161): after a restart the WAL replay
      // drops the blocks that lived in the RAM tier and keeps the blob at
      // what is left, so a fsynced range whose primary died with the node
      // read back as a HOLE through the native short-range semantics below,
      // while the remote copy -- written through before the fsync was acked
      // -- still held it. A legitimately short blob is short on the remote
      // too, and the read then falls through to those semantics as before.
      bool remote = false;
      clio::run::u32 remote_c = 0;
      CLIO_CO_AWAIT(ReadRemoteCopy(task, remote, &remote_c));
      if (remote) {
        if (!primary_unreadable) {
          CLIO_CO_AWAIT(HealPrimaryFromRemote(task->tag_id_, blob_name,
                                              remote_c, end));
        }
        task->SetReturnCode(0);
        CLIO_CO_RETURN;
      }
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

clio::run::TaskResume Runtime::MultiPutBlobLocal(
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
  // One writer per blob (LockBlobs): every blob of the batch, at once.
  struct Unlocker {
    Runtime *rt;
    std::vector<std::string> keys;
    ~Unlocker() { rt->UnlockBlobs(keys); }
  };
  std::vector<std::string> batch_keys;
  {
    clio::cte::core::MultiPutBatchView pre;
    if (clio::cte::core::MultiPutBatchView::Attach(*task, &pre)) {
      for (size_t d = 0; d < pre.size(); ++d) {
        if (!pre.RecordValid(d)) continue;
        batch_keys.push_back(
            BlobKey(pre.descs_[d].tag_id_, pre.descs_[d].blob_name_));
      }
    }
  }
  CLIO_CO_AWAIT(LockBlobs(batch_keys));
  Unlocker batch_unlock{this, batch_keys};
  {
    // Same guard as the scalar put: no record may re-grow a primary that
    // lost its bytes (see RefillPrimaryBeforeWrite).
    clio::cte::core::MultiPutBatchView pre;
    if (clio::cte::core::MultiPutBatchView::Attach(*task, &pre)) {
      for (size_t d = 0; d < pre.size(); ++d) {
        if (!pre.RecordValid(d) || pre.descs_[d].offset_ == 0) continue;
        bool refilled = true;
        CLIO_CO_AWAIT(RefillPrimaryBeforeWrite(pre.descs_[d].tag_id_,
                                               pre.descs_[d].blob_name_,
                                               pre.descs_[d].offset_,
                                               refilled));
        if (!refilled) {
          task->first_rc_ = 10 + clio::cte::core::kPutNoSpaceRc;
          task->SetReturnCode(task->first_rc_);
          CLIO_CO_RETURN;
        }
      }
    }
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
                                       config_.replica_score_, rep_ctx, 0u,
                                       clio::run::PoolQuery::Local());
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
