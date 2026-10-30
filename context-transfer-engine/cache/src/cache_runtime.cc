/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include <clio_cte/cache/cache_runtime.h>
#include <clio_cte/core/blob_placement.h>

namespace clio::cte::cache {

/** Chunk size for chain -> cache re-population copies. */
static constexpr clio::run::u64 kCacheChunkBytes = 4ULL * 1024 * 1024;

clio::run::TaskResume Runtime::Create(clio::run::shared_ptr<CreateTask> &task) {
  CLIO_TASK_BODY_BEGIN
  config_ = task->GetParams();
  interposer_next_pool_ = config_.next_pool_id_;  // base forwarding target
  if (!config_.next_pool_id_.IsNull()) {
    next_client_ =
        std::make_unique<clio::cte::core::Client>(config_.next_pool_id_);
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

clio::cte::core::Client *Runtime::GetNextClient() {
  if (!next_client_) {
    next_client_ = std::make_unique<clio::cte::core::Client>(CorePoolId());
  }
  return next_client_.get();
}

clio::cte::core::Client *Runtime::GetLocalClient() {
  if (!local_client_) {
    local_client_ =
        std::make_unique<clio::cte::core::Client>(clio::cte::core::kCtePoolId);
  }
  return local_client_.get();
}

clio::run::PoolQuery Runtime::ScheduleTask(
    const clio::run::shared_ptr<clio::run::Task> &task) {
  // Reads AND writes route SUBMITTER-LOCAL (issue #886 locality): the
  // local handler serves/updates the node-local raw copy and reaches the
  // blob's owner through the chain client — so the hot path (a rank
  // touching its own data) is all PoolQuery::Local, and only the
  // authoritative hop crosses nodes. Explicit-replica ops keep owner
  // routing — they address a concrete copy that lives with the blob.
  switch (task->method_) {
    case clio::cte::core::Method::kGetBlob: {
      auto typed = task.template Cast<clio::cte::core::GetBlobTask>();
      if (typed->context_.replica_ == 0) {
        return clio::run::PoolQuery::Local();
      }
      break;
    }
    case clio::cte::core::Method::kGetBlobSize: {
      auto typed = task.template Cast<clio::cte::core::GetBlobSizeTask>();
      if (typed->replica_ == 0) {
        return clio::run::PoolQuery::Local();
      }
      break;
    }
    case clio::cte::core::Method::kPutBlob: {
      auto typed = task.template Cast<clio::cte::core::PutBlobTask>();
      if (typed->context_.replica_ == 0) {
        return clio::run::PoolQuery::Local();
      }
      break;
    }
    case clio::cte::core::Method::kGetOrCreateTag:
    case clio::cte::core::Method::kDelTag:
      // Answered (lookup hit) or driven (owner write + broadcast) from here.
      return clio::run::PoolQuery::Local();
    case clio::cte::core::Method::kPodMultiGetBlob:
      // The pager's batch is served where it was issued: cache copies are
      // per node, and the core path below forwards remote records itself.
      return clio::run::PoolQuery::Local();
    default:
      break;
  }
  return CoreInterposer::ScheduleTask(task);
}

// ===========================================================================
// Tag-scoped caching (the GPU pager path)
//
// A tag ENROLLED with this module gets node-local cache copies of its
// remote-owned pages on every node that fetches them through
// PodMultiGetBlob. Ownership is untouched -- the hashed owner keeps the one
// authoritative copy -- so coherence is the core's owner-driven protocol:
// populating a copy registers this node with the page's owner
// (RegisterReplicaContainer), and any primary write to the page deletes
// every registered copy before it completes. Tag-level control on top:
// InvalidateTag drops every cached copy of a tag on every node at once (a
// dataset was rewritten out of band, an epoch ended); ForgetTag also stops
// caching it.
// ===========================================================================

bool Runtime::TagEnrolled(const TagId &tag_id, bool auto_enroll) {
  std::lock_guard<std::mutex> g(tags_mu_);
  if (tags_.find(TagKey(tag_id)) != tags_.end()) return true;
  if (!auto_enroll) return false;
  tags_.emplace(TagKey(tag_id), TagCache());
  return true;
}

clio::run::TaskResume Runtime::EnrollTag(
    clio::run::shared_ptr<TagCacheTask> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    std::lock_guard<std::mutex> g(tags_mu_);
    tags_.emplace(TagKey(task->tag_id_), TagCache());
  }
  task->dropped_ = 0;
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::InvalidateTag(
    clio::run::shared_ptr<TagCacheTask> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    // Snapshot and clear first: a fetch that repopulates during the deletes
    // below records its page afresh and stays (it read the current owner).
    std::vector<std::string> names;
    {
      std::lock_guard<std::mutex> g(tags_mu_);
      auto it = tags_.find(TagKey(task->tag_id_));
      if (it != tags_.end()) {
        names.assign(it->second.cached.begin(), it->second.cached.end());
        it->second.cached.clear();
      }
    }
    auto *local = GetLocalClient();
    clio::run::u64 dropped = 0;
    for (const std::string &name : names) {
      auto del = local->AsyncDelBlob(task->tag_id_, name,
                                     clio::run::PoolQuery::Local(),
                                     clio::cte::core::kDelCacheCopyOnly);
      CLIO_CO_AWAIT(del);
      if (del->GetReturnCode() == 0) ++dropped;
    }
    task->dropped_ = dropped;
    task->return_code_ = 0;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::ForgetTag(
    clio::run::shared_ptr<TagCacheTask> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    std::vector<std::string> names;
    {
      std::lock_guard<std::mutex> g(tags_mu_);
      auto it = tags_.find(TagKey(task->tag_id_));
      if (it != tags_.end()) {
        names.assign(it->second.cached.begin(), it->second.cached.end());
        tags_.erase(it);
      }
    }
    auto *local = GetLocalClient();
    clio::run::u64 dropped = 0;
    for (const std::string &name : names) {
      auto del = local->AsyncDelBlob(task->tag_id_, name,
                                     clio::run::PoolQuery::Local(),
                                     clio::cte::core::kDelCacheCopyOnly);
      CLIO_CO_AWAIT(del);
      if (del->GetReturnCode() == 0) ++dropped;
    }
    task->dropped_ = dropped;
    task->return_code_ = 0;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::PodMultiGetBlob(
    clio::run::shared_ptr<clio::cte::core::PodMultiGetBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    using clio::cte::core::kPodMultiMax;
    using clio::cte::core::PodBlobReq;
    const clio::run::u32 n = std::min<clio::run::u32>(task->count_, kPodMultiMax);
    if (n == 0 || CLIO_IPC->GetNumHosts() <= 1 ||
        !TagEnrolled(task->tag_id_, config_.auto_enroll_)) {
      CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kPodMultiGetBlob,
                                  task.template Cast<clio::run::Task>()));
      CLIO_CO_RETURN;
    }
    task->NormalizeBlobName();  // decimal names from here on
    const TagId tag = task->tag_id_;
    auto *local = GetLocalClient();
    auto *next = GetNextClient();

    // 1. Remote-owned pages: try this node's cache copy (all issued, then
    //    awaited). A cache copy is coherent by invalidation, so the
    //    generational stamp-wait does not apply to it.
    Context rctx = task->context_;
    rctx.op_flags_ &= ~Context::kGenerational;
    rctx.generation_ = 0;
    rctx.create_on_get_ = false;
    rctx.replica_ = clio::cte::core::kCacheReplica;
    bool remote[kPodMultiMax] = {};
    bool hit[kPodMultiMax] = {};
    clio::run::Future<clio::cte::core::GetBlobTask> gets[kPodMultiMax];
    for (clio::run::u32 i = 0; i < n; ++i) {
      const std::string name = task->GetBlobName(i);
      if (IsBlobOwnerLocal(tag, name)) continue;
      remote[i] = true;
      const PodBlobReq &r = task->reqs_[i];
      gets[i] = local->AsyncGetBlob(tag, name, r.offset_, r.size_,
                                    task->flags_, r.data_,
                                    clio::run::PoolQuery::Local(), rctx);
    }
    clio::run::u32 nhit = 0;
    for (clio::run::u32 i = 0; i < n; ++i) {
      if (!remote[i]) continue;
      CLIO_CO_AWAIT(gets[i]);
      if (gets[i]->GetReturnCode() == 0) {
        hit[i] = true;
        task->reqs_[i].rc_ = 0;
        ++nhit;
      }
    }
    if (nhit == n) {
      task->num_ok_ = n;
      std::lock_guard<std::mutex> g(tags_mu_);
      tags_[TagKey(tag)].hits += nhit;
      CLIO_CO_RETURN;
    }

    // 2. Everything else in ONE core batch (owner-local pages and cache
    //    misses), compacted to the front and restored after.
    PodBlobReq saved[kPodMultiMax];
    clio::run::u32 map[kPodMultiMax];
    clio::run::u32 m = 0;
    for (clio::run::u32 i = 0; i < n; ++i) saved[i] = task->reqs_[i];
    for (clio::run::u32 i = 0; i < n; ++i) {
      if (!hit[i]) {
        map[m] = i;
        task->reqs_[m] = saved[i];
        ++m;
      }
    }
    task->count_ = m;
    task->num_ok_ = 0;
    CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kPodMultiGetBlob,
                                task.template Cast<clio::run::Task>()));
    const clio::run::u32 core_ok = task->num_ok_;
    clio::run::u32 rcs[kPodMultiMax];
    for (clio::run::u32 j = 0; j < m; ++j) rcs[j] = task->reqs_[j].rc_;
    for (clio::run::u32 i = 0; i < n; ++i) task->reqs_[i] = saved[i];
    for (clio::run::u32 j = 0; j < m; ++j) task->reqs_[map[j]].rc_ = rcs[j];
    task->count_ = n;
    task->num_ok_ = nhit + core_ok;

    // 3. Populate this node's cache with the remote pages just fetched, then
    //    register with their owners so a primary write invalidates them. A
    //    copy whose registration fails is dropped (nobody would invalidate it).
    Context pctx;
    AimAtCacheReplica(&pctx);
    bool pop[kPodMultiMax] = {};
    clio::run::Future<clio::cte::core::PutBlobTask> puts[kPodMultiMax];
    for (clio::run::u32 j = 0; j < m; ++j) {
      const clio::run::u32 i = map[j];
      const PodBlobReq &r = task->reqs_[i];
      if (!remote[i] || r.rc_ != 0 || r.offset_ != 0 || r.size_ == 0) continue;
      pop[i] = true;
      puts[i] = local->AsyncPutBlob(tag, task->GetBlobName(i), 0, r.size_,
                                    r.data_, /*score=*/-1.0f, pctx,
                                    /*flags=*/0, clio::run::PoolQuery::Local());
    }
    clio::run::Future<clio::cte::core::RegisterReplicaContainerTask>
        regs[kPodMultiMax];
    for (clio::run::u32 i = 0; i < n; ++i) {
      if (!pop[i]) continue;
      CLIO_CO_AWAIT(puts[i]);
      if (puts[i]->GetReturnCode() != 0) {
        pop[i] = false;
        continue;
      }
      regs[i] = next->AsyncRegisterReplicaContainer(
          tag, task->GetBlobName(i), CLIO_IPC->GetNodeId(),
          clio::run::PoolQuery::Dynamic(), /*expected_version=*/0);
    }
    clio::run::u64 populated = 0;
    std::vector<std::string> added;
    for (clio::run::u32 i = 0; i < n; ++i) {
      if (!pop[i]) continue;
      CLIO_CO_AWAIT(regs[i]);
      const std::string name = task->GetBlobName(i);
      if (regs[i]->GetReturnCode() != 0) {
        auto del = local->AsyncDelBlob(tag, name, clio::run::PoolQuery::Local(),
                                       clio::cte::core::kDelCacheCopyOnly);
        CLIO_CO_AWAIT(del);
        continue;
      }
      added.push_back(name);
      ++populated;
    }
    {
      std::lock_guard<std::mutex> g(tags_mu_);
      TagCache &tc = tags_[TagKey(tag)];
      tc.hits += nhit;
      tc.misses += m;
      tc.populated += populated;
      for (auto &a : added) tc.cached.insert(std::move(a));
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

// ===========================================================================
// Tag METADATA cache -- write-through, strongly consistent on completion.
//
// Every node's cache container keeps name -> TagId for the tags it has seen,
// plus each tag's fault handler installed into its LOCAL core container. A
// lookup (GetOrCreateTag with no write arguments) is answered here with no
// round trip. A tag WRITE -- creation, fault-handler registration, delete,
// rename -- goes to the tag's owner through the core first, and only after
// the owner committed it is the change broadcast to every node's cache; the
// write returns once every node acknowledged. Guarantee: after a tag write
// returns, every lookup on every node sees it. While it is in flight, nodes
// may briefly disagree (the write has not completed). Concurrent writes to
// the SAME tag are serialized at its owner; their broadcasts are not
// ordered against each other (last broadcast wins) -- tags change rarely.
//
// Not cached: a tag's byte size (it changes on every page write). A lookup
// served from the cache reports tag_size_ = 0; ask GetTagSize for the size.
// ===========================================================================

bool Runtime::TagMetaLookup(const std::string &name, TagId *id) {
  std::lock_guard<std::mutex> g(tmeta_mu_);
  auto it = tmeta_by_name_.find(name);
  if (it == tmeta_by_name_.end()) return false;
  *id = it->second;
  return true;
}

void Runtime::TagMetaInsert(const std::string &name, const TagId &id) {
  if (name.empty() || id.IsNull()) return;
  std::lock_guard<std::mutex> g(tmeta_mu_);
  tmeta_by_name_[name] = id;
  tmeta_names_[TagKey(id)].insert(name);
}

void Runtime::TagMetaDrop(const TagId &id, const std::string &name) {
  std::lock_guard<std::mutex> g(tmeta_mu_);
  if (!id.IsNull()) {
    auto it = tmeta_names_.find(TagKey(id));
    if (it != tmeta_names_.end()) {
      for (const auto &n : it->second) tmeta_by_name_.erase(n);
      tmeta_names_.erase(it);
    }
  }
  if (!name.empty()) {
    auto f = tmeta_by_name_.find(name);
    if (f != tmeta_by_name_.end()) {
      auto ns = tmeta_names_.find(TagKey(f->second));
      if (ns != tmeta_names_.end()) ns->second.erase(name);
      tmeta_by_name_.erase(f);
    }
  }
}

clio::run::Future<TagCacheUpdateTask> Runtime::SendTagMeta(
    clio::run::u32 method, const std::string &name, const TagId &id,
    const clio::run::PoolId &fault_pool_id, const std::string &fault_pool_name,
    const std::string &fault_params) {
  auto *ipc = CLIO_CPU_IPC;
  auto t = ipc->NewTask<TagCacheUpdateTask>(
      clio::run::CreateTaskId(), pool_id_, clio::run::PoolQuery::Broadcast(),
      method, name, id, fault_pool_id, fault_pool_name, fault_params);
  return ipc->Send(t);
}

clio::run::TaskResume Runtime::GetOrCreateTag(
    clio::run::shared_ptr<
        clio::cte::core::GetOrCreateTagTask<clio::cte::core::CreateParams>>
        &task) {
  CLIO_TASK_BODY_BEGIN
  {
    const std::string name = task->tag_name_.str();
    const clio::run::PoolId fpool = task->fault_pool_id_;
    const std::string fname = task->fault_pool_name_.str();
    const std::string fparams = task->fault_params_.str();
    // A write: registers a fault handler, or pins a caller-chosen id.
    const bool is_write = !fpool.IsNull() || !task->tag_id_.IsNull();
    TagId hit = TagId::GetNull();
    if (!is_write && !name.empty() && TagMetaLookup(name, &hit)) {
      task->tag_id_ = hit;
      task->created_ = 0;
      task->tag_size_ = 0;  // size is not cached (see above)
      task->SetReturnCode(0);
      CLIO_CO_RETURN;
    }
    // Miss or write: the owner decides (normal core routing).
    auto f = GetNextClient()->AsyncGetOrCreateTag(
        name, fpool, fname, fparams, task->tag_id_,
        clio::run::PoolQuery::Dynamic());
    CLIO_CO_AWAIT(f);
    const clio::run::u32 rc = f->GetReturnCode();
    task->tag_id_ = f->tag_id_;
    task->created_ = f->created_;
    task->tag_size_ = f->tag_size_;
    if (rc == 0 && !task->tag_id_.IsNull()) {
      if (is_write || task->created_ != 0) {
        // Write-through: every node installs it before this returns.
        auto b = SendTagMeta(Method::kTagCachePut, name, task->tag_id_, fpool,
                             fname, fparams);
        CLIO_CO_AWAIT(b);
        if (b->GetReturnCode() != 0) {
          HLOG(kWarning, "tag cache: put of '{}' rc={} on some node", name,
               b->GetReturnCode());
        }
      } else {
        TagMetaInsert(name, task->tag_id_);  // read miss: fill this node
      }
    }
    task->SetReturnCode(rc);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::TagCachePut(
    clio::run::shared_ptr<TagCacheUpdateTask> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    const std::string name = task->tag_name_.str();
    TagMetaInsert(name, task->tag_id_);
    clio::run::u32 rc = 0;
    if (!task->fault_pool_id_.IsNull()) {
      // The fault handler must be visible to THIS node's core container: a
      // page miss is resolved by whichever container the page lands on.
      auto f = GetLocalClient()->AsyncGetOrCreateTag(
          name, task->fault_pool_id_, task->fault_pool_name_.str(),
          task->fault_params_.str(), task->tag_id_,
          clio::run::PoolQuery::Local());
      CLIO_CO_AWAIT(f);
      rc = f->GetReturnCode();
    }
    task->SetReturnCode(rc);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::TagCacheDrop(
    clio::run::shared_ptr<TagCacheUpdateTask> &task) {
  CLIO_TASK_BODY_BEGIN
  TagMetaDrop(task->tag_id_, task->tag_name_.str());
  task->SetReturnCode(0);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::DelTag(
    clio::run::shared_ptr<clio::cte::core::DelTagTask> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    const std::string name = task->tag_name_.str();
    const TagId in_id = task->tag_id_;
    clio::run::u32 rc = 0;
    TagId gone = in_id;
    if (name.empty()) {
      auto f = GetNextClient()->AsyncDelTag(in_id, clio::run::PoolQuery::Dynamic());
      CLIO_CO_AWAIT(f);
      rc = f->GetReturnCode();
      if (!f->tag_id_.IsNull()) gone = f->tag_id_;
    } else {
      auto f = GetNextClient()->AsyncDelTag(name, clio::run::PoolQuery::Dynamic(),
                                            task->posix_unlink_ != 0);
      CLIO_CO_AWAIT(f);
      rc = f->GetReturnCode();
      if (!f->tag_id_.IsNull()) gone = f->tag_id_;
    }
    task->tag_id_ = gone;
    // Drop everywhere before returning -- also on failure: a stale entry is
    // never worth keeping, and the next lookup re-reads the owner.
    auto b = SendTagMeta(Method::kTagCacheDrop, name, gone,
                         clio::run::PoolId::GetNull(), "", "");
    CLIO_CO_AWAIT(b);
    task->SetReturnCode(rc);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::RenameTag(
    clio::run::shared_ptr<clio::cte::core::RenameTagTask> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    // Rename is a core BROADCAST: this runs on every node's cache container,
    // so each renames in its own core and drops its own stale entries; the
    // broadcast completes only when every node did both.
    const std::string old_name = task->old_name_.str();
    CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kRenameTag,
                                task.template Cast<clio::run::Task>()));
    TagMetaDrop(task->tag_id_, old_name);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

bool Runtime::IsBlobOwnerLocal(const TagId &tag_id,
                               const std::string &blob_name) {
  // Same hash as the core's HashBlobToContainer (owner routing), resolved
  // against THIS pool's container map — every pool in the chain shares the
  // container layout (replication's primary-hit path relies on the same
  // property).
  auto *pool_manager = CLIO_POOL_MANAGER;
  const clio::run::PoolInfo *info = pool_manager->GetPoolInfo(pool_id_);
  if (info == nullptr || info->num_containers_ == 0) {
    return false;
  }
  std::hash<std::string> string_hasher;
  std::hash<clio::run::u32> u32_hasher;
  clio::run::u32 h = u32_hasher(tag_id.major_);
  h ^= u32_hasher(tag_id.minor_) + 0x9e3779b9 + (h << 6) + (h >> 2);
  h ^= static_cast<clio::run::u32>(string_hasher(blob_name)) + 0x9e3779b9 +
       (h << 6) + (h >> 2);
  clio::run::ContainerId cid = h % info->num_containers_;
  if (pool_manager->HasContainer(pool_id_, cid)) {
    return true;
  }
  if (pool_manager->GetContainerNodeId(pool_id_, cid) ==
      CLIO_IPC->GetNodeId()) {
    return true;
  }
  return IsStandingIn(tag_id, blob_name);
}

bool Runtime::IsStandingIn(const TagId &tag_id, const std::string &blob_name) {
  // Standing in for a dead owner: this node's core holds the blob's
  // authoritative copy (the replication chimod's remote copy, served by
  // failover) in the SAME BlobInfo as any cache copy it took as a reader
  // earlier. That cache slot was kept current by the owner's invalidations,
  // which stopped when the owner died -- serving it read a stale page
  // (zeros) while every other node, reading through this one, got the
  // right bytes. Callers behave as the owner: no cache copy, read the
  // primary.
  auto *pool_manager = CLIO_POOL_MANAGER;
  const clio::run::PoolInfo *info = pool_manager->GetPoolInfo(pool_id_);
  if (info == nullptr || info->num_containers_ <= 1) return false;
  const clio::run::u32 cid =
      clio::cte::core::BlobHash(tag_id, blob_name) % info->num_containers_;
  const clio::run::u32 serving =
      clio::cte::core::FailoverContainer(pool_id_, cid);
  return serving != cid && pool_manager->GetContainerNodeId(pool_id_, serving) ==
                               CLIO_IPC->GetNodeId();
}

/** Mutate a put context to aim at THE cache replica: raw bytes, the
 *  configured score floor, cache-slot addressing. */
void Runtime::AimAtCacheReplica(Context *ctx) const {
  ctx->replica_ = clio::cte::core::kCacheReplica;
  ctx->replica_flags_ |= clio::cte::core::REPLICA_CACHE;
  ctx->replica_min_score_ = config_.min_score_;
  ctx->transform_flags_ = 0;
  ctx->min_persistence_level_ = 0;
}

clio::run::TaskResume Runtime::PutBlob(
    clio::run::shared_ptr<clio::cte::core::PutBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  // Explicit replica addressing passes through untouched (owner-routed).
  if (task->context_.replica_ != 0) {
    CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kPutBlob,
                           task.template Cast<clio::run::Task>()));
    CLIO_CO_RETURN;
  }
  {
    // WRITER-LOCAL asynchronous write-through (issue #886 locality). This
    // handler runs on the SUBMITTER's node. The node-local raw copy is
    // written FIRST, then the authoritative put goes to the blob's owner
    // chain carrying origin_node_ — the owner invalidates every OTHER
    // registered copy and registers ours atomically under the write token,
    // so the fresh local copy is coherent by construction. Ordering is the
    // safety: the local write strictly precedes the registration, so any
    // LATER foreign put's invalidation always catches it.
    //
    // The local copy is CREATED only when this put demonstrably covers the
    // whole blob (single region at offset 0 AND the authoritative blob does
    // not exist yet) — a partial write must never mint a prefix that
    // pretends to be complete. An EXISTING local copy mirrors every put
    // in place, staying byte- and size-identical to the authoritative blob
    // for all writes issued through this node.
    auto *local = GetLocalClient();
    auto *next = GetNextClient();
    const std::string blob_name = task->blob_name_.str();
    const Context orig_ctx = task->context_;

    // The blob's OWNER node keeps no cache copy (issue #894): the primary
    // is already node-local (and zero-IPC readable), and an owner-side
    // copy sits outside the register/invalidate protocol — origin
    // registration skips self, so a later foreign write would never
    // invalidate it and this node would serve its own stale bytes forever
    // (the CI-caught fragmented winner split).
    const bool owner_local = IsBlobOwnerLocal(task->tag_id_, blob_name);
    if (owner_local) {
      // Owner-local: no copy to keep, so the put IS the authoritative put.
      // Run the next pool's handler inline, as the other interposers do,
      // instead of submitting a new task: the extra task cost a scheduler
      // route, a worker hop and a completion event per put, and large puts
      // all queue on the I/O workers -- 1 MiB puts ran ~4x slower through
      // this layer than straight into the layer below it.
      task->context_ = orig_ctx;
      CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kPutBlob,
                                  task.template Cast<clio::run::Task>()));
      task->context_.replica_ = orig_ctx.replica_;
      task->context_.replica_flags_ = orig_ctx.replica_flags_;
      task->context_.origin_node_ = Context::kNoOriginNode;
      CLIO_CO_RETURN;
    }

    // 1. ONE fused local task: apply the put to the local raw copy iff it
    //    exists (UPDATE_ONLY; kReplicaAbsentRc = it doesn't). No probes.
    bool wrote_local = false;
    bool speculative = false;
    if (!owner_local) {
      task->context_ = orig_ctx;
      AimAtCacheReplica(&task->context_);
      task->context_.replica_flags_ |= clio::cte::core::REPLICA_UPDATE_ONLY;
      CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kPutBlob,
                             task.template Cast<clio::run::Task>()));
      wrote_local = (task->GetReturnCode() == 0);
    }
    if (!owner_local && !wrote_local &&
        task->GetReturnCode() == clio::cte::core::kReplicaAbsentRc &&
        task->segments_.empty() && task->offset_ == 0) {
      // 1b. No copy yet and the put MIGHT cover the whole blob: create one
      //     SPECULATIVELY. The owner verifies (VERIFY_COMPLETE rides the
      //     authoritative put) and clears origin_node_ in the OUT context
      //     if the blob pre-existed beyond this put — we then drop it.
      task->context_ = orig_ctx;
      AimAtCacheReplica(&task->context_);
      CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kPutBlob,
                             task.template Cast<clio::run::Task>()));
      wrote_local = (task->GetReturnCode() == 0);
      speculative = true;
    }
    HLOG(kDebug,
         "[COH] cache put blob={} node={} wrote_local={} speculative={}",
         blob_name, CLIO_IPC->GetNodeId(), wrote_local, speculative);

    // 2. Authoritative put(s) to the owner chain (Dynamic hash-routes
    //    there). Vectored puts replay their regions in list order (the
    //    core's last-writer-wins rule, same as the replication sync path).
    {
      Context auth_ctx = orig_ctx;
      if (wrote_local) {
        auth_ctx.origin_node_ = CLIO_IPC->GetNodeId();
        if (speculative) {
          auth_ctx.replica_flags_ |=
              clio::cte::core::REPLICA_VERIFY_COMPLETE;
        }
      }
      std::vector<clio::cte::core::BlobRegion> regions;
      clio::cte::core::ForEachBlobRegion(*task,
          [&regions](const clio::cte::core::BlobRegion &r) {
            regions.push_back(r);
            return true;
          });
      clio::run::u32 rc = 0;
      Context out_ctx = auth_ctx;
      for (size_t i = 0; i < regions.size() && rc == 0; ++i) {
        auto put = next->AsyncPutBlob(task->tag_id_, blob_name,
                                      regions[i].blob_off_, regions[i].size_,
                                      regions[i].data_, task->score_,
                                      auth_ctx, task->flags_);
        CLIO_CO_AWAIT(put);
        rc = put->GetReturnCode();
        out_ctx = put->context_;
      }
      if (wrote_local &&
          (rc != 0 ||
           out_ctx.origin_node_ == Context::kNoOriginNode)) {
        // Failed put, or the owner REFUSED the speculative registration
        // (the blob pre-existed beyond this put, so our copy is a prefix):
        // the local copy must not survive.
        auto del = local->AsyncDelBlob(task->tag_id_, blob_name,
                                       clio::run::PoolQuery::Local(),
                                       clio::cte::core::kDelCacheCopyOnly);
        CLIO_CO_AWAIT(del);
      }
      out_ctx.replica_ = orig_ctx.replica_;
      out_ctx.replica_flags_ = orig_ctx.replica_flags_;
      out_ctx.origin_node_ = Context::kNoOriginNode;
      task->context_ = out_ctx;
      task->SetReturnCode(rc);
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::MultiPutBlob(
    clio::run::shared_ptr<clio::cte::core::MultiPutBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  // Batches get the same asynchronous write-through as scalar puts: the
  // authoritative batch lands below first, then ONE re-aimed batch writes
  // every record's raw cache copy (scalar-equivalent semantics; batching is
  // never passed on).
  if (task->context_.replica_ != 0) {
    CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kMultiPutBlob,
                           task.template Cast<clio::run::Task>()));
    CLIO_CO_RETURN;
  }
  {
    const Context orig_ctx = task->context_;
    CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kMultiPutBlob,
                           task.template Cast<clio::run::Task>()));
    if (task->GetReturnCode() != 0) {
      CLIO_CO_RETURN;
    }
    const clio::run::u32 auth_ok = task->num_ok_;
    const int auth_rc = task->first_rc_;

    // The mirror batch is SINGLE-NODE only (issue #894): batched cache
    // copies carry no origin registration, so on a cluster nothing would
    // ever invalidate them and any foreign write leaves them stale — and
    // records owned by this node are the owner blind spot outright.
    // Multinode read locality still comes from the read path's populate,
    // which registers (version-checked). Single-node has no foreign
    // writers, so the mirror is coherent by construction there.
    if (CLIO_IPC->GetNumHosts() <= 1) {
      task->context_ = orig_ctx;
      AimAtCacheReplica(&task->context_);
      task->num_ok_ = 0;
      task->first_rc_ = 0;
      CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kMultiPutBlob,
                             task.template Cast<clio::run::Task>()));
      if (task->GetReturnCode() != 0 || task->first_rc_ != 0) {
        HLOG(kWarning, "cache: batch cache-replica write failed rc={}/{} "
             "(batch ok)", task->GetReturnCode(), task->first_rc_);
      }
    }
    // Report the AUTHORITATIVE batch's outcome.
    task->context_ = orig_ctx;
    task->num_ok_ = auth_ok;
    task->first_rc_ = auth_rc;
    task->SetReturnCode(auth_rc == 0 ? 0 : auth_rc);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::GetBlob(
    clio::run::shared_ptr<clio::cte::core::GetBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  // Explicit replica addressing passes through untouched (these tasks were
  // routed to the blob's OWNER container by ScheduleTask).
  if (task->context_.replica_ != 0) {
    CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kGetBlob,
                           task.template Cast<clio::run::Task>()));
    CLIO_CO_RETURN;
  }
  {
    // This handler runs READER-LOCAL (ScheduleTask). The node-local raw
    // copy lives in the LOCAL core container's cache-replica slot — the
    // same slot the write path fills — so writer and remote reader serve
    // through one code path, and the node's SHM mirror record makes the
    // copy zero-IPC readable. INVARIANT: a present local copy is COMPLETE
    // and CURRENT (creation covers the whole blob; every put through this
    // node mirrors into it; invalidation and populate-failure delete it
    // whole), so the read is OPPORTUNISTIC — no size probe, just try it.
    const std::string blob_name = task->blob_name_.str();
    // The owner node has no cache copy (issue #894, see PutBlob) — its
    // authoritative primary is node-local already; skip straight to the
    // chain and never populate here.
    const bool owner_local = IsBlobOwnerLocal(task->tag_id_, blob_name);
    if (owner_local) {
      // A cache copy left from when this node was only a reader (it is now
      // standing in) must not outlive the stand-in period: drop it.
      if (IsStandingIn(task->tag_id_, blob_name)) {
        auto del = GetLocalClient()->AsyncDelBlob(
            task->tag_id_, blob_name, clio::run::PoolQuery::Local(),
            clio::cte::core::kDelCacheCopyOnly);
        CLIO_CO_AWAIT(del);  // "not found" (no copy) is the usual answer
      }
      // The primary is node-local: read it through the next pool inline (no
      // new task, see PutBlob) and never populate a copy.
      CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kGetBlob,
                                  task.template Cast<clio::run::Task>()));
      CLIO_CO_RETURN;
    }
    {
      task->context_.replica_ = clio::cte::core::kCacheReplica;
      CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kGetBlob,
                             task.template Cast<clio::run::Task>()));
      task->context_.replica_ = 0;
      if (task->GetReturnCode() == 0) {
        CLIO_CO_RETURN;
      }
    }

    // Miss: fetch from the blob's authoritative owner chain (Dynamic
    // hash-routes to the owner; the layers there decompress/heal). Each
    // region reads straight into the caller's buffer.
    auto *next = GetNextClient();
    std::vector<clio::cte::core::BlobRegion> regions;
    clio::cte::core::ForEachBlobRegion(*task,
        [&regions](const clio::cte::core::BlobRegion &r) {
          regions.push_back(r);
          return true;
        });
    {
      clio::run::u32 rc = 0;
      clio::run::u32 out_tflags = 0;
      clio::run::u64 fetch_version = 0;
      for (size_t i = 0; i < regions.size() && rc == 0; ++i) {
        auto get = next->AsyncGetBlob(
            task->tag_id_, blob_name, regions[i].blob_off_, regions[i].size_,
            task->flags_, regions[i].data_, clio::run::PoolQuery::Dynamic(),
            task->context_);
        CLIO_CO_AWAIT(get);
        rc = get->GetReturnCode();
        out_tflags = get->context_.transform_flags_;
        if (i == 0) {
          fetch_version = get->context_.version_;
        }
      }
      task->context_.transform_flags_ = out_tflags;
      task->context_.version_ = fetch_version;
      task->SetReturnCode(rc);
      if (rc != 0) {
        CLIO_CO_RETURN;
      }
    }

    // Best-effort local population + coherence registration, so the next
    // read (and the zero-IPC fast path) is node-local. When this read
    // fetched the WHOLE blob (the file-per-process page pattern), the
    // caller's buffer seeds the copy directly — no re-fetch.
    if (owner_local) {
      CLIO_CO_RETURN;  // the primary IS this node's local copy
    }
    {
      clio::run::u64 total = 0;
      {
        auto sz = next->AsyncGetBlobSize(task->tag_id_, blob_name);
        CLIO_CO_AWAIT(sz);
        if (sz->GetReturnCode() != 0 || sz->size_ == 0) {
          CLIO_CO_RETURN;  // raced away — nothing to populate
        }
        total = sz->size_;
      }
      if (regions.size() == 1 && regions[0].blob_off_ == 0 &&
          regions[0].size_ >= total &&
          task->context_.transform_flags_ == 0) {
        auto *local = GetLocalClient();
        Context put_ctx;
        AimAtCacheReplica(&put_ctx);
        auto put = local->AsyncPutBlob(task->tag_id_, blob_name, 0, total,
                                       regions[0].data_, /*score=*/-1.0f,
                                       put_ctx, /*flags=*/0,
                                       clio::run::PoolQuery::Local());
        CLIO_CO_AWAIT(put);
        if (put->GetReturnCode() != 0) {
          // A put that failed part-way (a device error, a full tier) leaves
          // a copy with chunks that were never written -- whatever their
          // reused extents held before. The present => COMPLETE invariant
          // the opportunistic read above relies on forbids keeping it: such
          // a copy was served as another file's old bytes with rc 0 (#1131).
          auto del = local->AsyncDelBlob(task->tag_id_, blob_name,
                                         clio::run::PoolQuery::Local(),
                                         clio::cte::core::kDelCacheCopyOnly);
          CLIO_CO_AWAIT(del);
        } else {
          // VERSION-CHECKED registration (issue #894): if the owner's
          // content moved on since our fetch, the registration is REJECTED
          // and the just-written copy is stale — delete it, the next read
          // refetches. Without the check, a put completing between fetch
          // and registration leaves this copy permanently stale (its
          // invalidation snapshot ran before we registered).
          auto reg = next->AsyncRegisterReplicaContainer(
              task->tag_id_, blob_name, CLIO_IPC->GetNodeId(),
              clio::run::PoolQuery::Dynamic(), task->context_.version_);
          CLIO_CO_AWAIT(reg);
          HLOG(kInfo,
               "[COH] cache populate blob={} node={} ver={} reg rc={}",
               blob_name, CLIO_IPC->GetNodeId(), task->context_.version_,
               reg->GetReturnCode());
          if (reg->GetReturnCode() != 0) {
            auto del = local->AsyncDelBlob(task->tag_id_, blob_name,
                                           clio::run::PoolQuery::Local(),
                                           clio::cte::core::kDelCacheCopyOnly);
            CLIO_CO_AWAIT(del);
          }
        }
      } else {
        CLIO_CO_AWAIT(PopulateLocal(task->tag_id_, blob_name));
      }
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::PopulateLocal(const TagId &tag_id,
                                             const std::string &blob_name) {
#ifdef CLIO_ENABLE_BOOST_COROUTINES
  clio::run::shared_ptr<clio::run::Task> cur_task = clio::run::GetCurrentTask();
#endif
  CLIO_TASK_BODY_BEGIN
  auto *next = GetNextClient();
  auto *local = GetLocalClient();
  clio::run::u64 total = 0;
  {
    // Logical size from the owner chain (the compressor reports original
    // sizes; Dynamic hash-routes to the owner).
    auto sz = next->AsyncGetBlobSize(tag_id, blob_name);
    CLIO_CO_AWAIT(sz);
    if (sz->GetReturnCode() != 0 || sz->size_ == 0) {
      CLIO_CO_RETURN;
    }
    total = sz->size_;
  }
  clio::run::u64 fetch_version = 0;
  {
    bool failed = false;
    for (clio::run::u64 off = 0; off < total && !failed;
         off += kCacheChunkBytes) {
      clio::run::u64 len = std::min(kCacheChunkBytes, total - off);
      auto buf = CLIO_IPC->AllocateBuffer(len);
      if (buf.IsNull()) {
        failed = true;
        break;
      }
      ctp::ipc::ShmPtr<> buf_ptr = buf.shm_.template Cast<void>();
      auto get = next->AsyncGetBlob(tag_id, blob_name, off, len, /*flags=*/0,
                                    buf_ptr);
      CLIO_CO_AWAIT(get);
      if (get->GetReturnCode() != 0) {
        CLIO_IPC->FreeBuffer(buf);
        failed = true;
        break;
      }
      if (off == 0) {
        fetch_version = get->context_.version_;
      }
      // Into THIS node's core container cache-replica slot (Local query
      // bypasses owner routing): raw bytes, mirror record published for the
      // node's zero-IPC fast path.
      Context put_ctx;
      AimAtCacheReplica(&put_ctx);
      auto put = local->AsyncPutBlob(tag_id, blob_name, off, len, buf_ptr,
                                     /*score=*/-1.0f, put_ctx, /*flags=*/0,
                                     clio::run::PoolQuery::Local());
      CLIO_CO_AWAIT(put);
      CLIO_IPC->FreeBuffer(buf);
      if (put->GetReturnCode() != 0) {
        failed = true;
      }
    }
    if (failed) {
      // The present ⇒ COMPLETE invariant (the opportunistic read and the
      // local size answer both rely on it) forbids keeping a prefix: a
      // partial population is deleted whole.
      auto del = local->AsyncDelBlob(tag_id, blob_name,
                                     clio::run::PoolQuery::Local(),
                                     clio::cte::core::kDelCacheCopyOnly);
      CLIO_CO_AWAIT(del);
      CLIO_CO_RETURN;
    }
  }
  {
    // Coherence registration at the blob's OWNER (Dynamic hash-routes
    // there): its next primary write invalidates this node's copy BEFORE
    // acking, so a covering local copy is never stale. VERSION-CHECKED
    // (issue #894): if the owner's content moved on since the first chunk
    // was fetched, the registration is REJECTED and the copy deleted —
    // otherwise a put landing between fetch and registration leaves this
    // copy permanently stale (its invalidation ran before we registered).
    auto reg = next->AsyncRegisterReplicaContainer(
        tag_id, blob_name, CLIO_IPC->GetNodeId(),
        clio::run::PoolQuery::Dynamic(), fetch_version);
    CLIO_CO_AWAIT(reg);
    HLOG(kInfo,
         "[COH] cache populate-chunked blob={} node={} ver={} reg rc={}",
         blob_name, CLIO_IPC->GetNodeId(), fetch_version,
         reg->GetReturnCode());
    if (reg->GetReturnCode() != 0) {
      auto del = local->AsyncDelBlob(tag_id, blob_name,
                                     clio::run::PoolQuery::Local(),
                                     clio::cte::core::kDelCacheCopyOnly);
      CLIO_CO_AWAIT(del);
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::GetBlobSize(
    clio::run::shared_ptr<clio::cte::core::GetBlobSizeTask> &task) {
  CLIO_TASK_BODY_BEGIN
  if (task->replica_ != 0) {
    CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kGetBlobSize,
                           task.template Cast<clio::run::Task>()));
    CLIO_CO_RETURN;
  }
  {
    // Node-local raw copy first (this handler runs READER-LOCAL): its raw
    // size IS the logical size (no transform), and invalidation keeps a
    // present copy current. NOTE: the copy is populated whole-blob and
    // reclaimed/invalidated whole-blob, so a present size is the full
    // logical size.
    if (IsBlobOwnerLocal(task->tag_id_, task->blob_name_.str())) {
      // Owner-local: ask the next pool inline (no new task, see PutBlob).
      CLIO_CO_AWAIT(ForwardToCore(clio::cte::core::Method::kGetBlobSize,
                                  task.template Cast<clio::run::Task>()));
      CLIO_CO_RETURN;
    }
    {
      auto *local = GetLocalClient();
      auto sz = local->AsyncGetBlobSize(task->tag_id_,
                                        task->blob_name_.str(),
                                        clio::run::PoolQuery::Local(),
                                        clio::cte::core::kCacheReplica);
      CLIO_CO_AWAIT(sz);
      if (sz->GetReturnCode() == 0 && sz->size_ > 0) {
        task->size_ = sz->size_;
        task->return_code_ = 0;
        CLIO_CO_RETURN;
      }
    }
  }
  {
    // Miss: ask the owner chain (Dynamic hash-routes there).
    auto sz = GetNextClient()->AsyncGetBlobSize(task->tag_id_,
                                                task->blob_name_.str());
    CLIO_CO_AWAIT(sz);
    task->size_ = sz->size_;
    task->SetReturnCode(sz->GetReturnCode());
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

}  // namespace clio::cte::cache

// ChiMod entry points (alloc/new/name/destroy) for the module manager.
CLIO_TASK_CC(clio::cte::cache::Runtime)
