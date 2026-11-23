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
 * Transport between filesystem containers (ShardOp), inodes, the deferred
 * data purge, the CTE tag-name publisher and the small per-container system
 * records (id reservations, orphans). The inode table lives in
 * fs_inode_table.cc.
 *
 * Multi-step operations (mkdir, rmdir, rename, link) keep consistency with
 * per-entry states instead of locks held across network round trips: a
 * PENDING entry is reserved but invisible, a LEAVING entry is still visible
 * but cannot be mutated by anyone else. Operations that must wait on such an
 * entry yield and retry; two-party operations (rename, link) fail EBUSY
 * instead and back off, which is what keeps two crossing renames from
 * deadlocking or building a directory cycle.
 */
#include <cerrno>
#include <fcntl.h>
#include <algorithm>
#include <chrono>
#include <atomic>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <clio_ctp/util/config_parse.h>
#include <clio_runtime/pool_manager.h>
#include <clio_cte/core/blob_placement.h>
#include <clio_cte/filesystem/filesystem_runtime.h>
#include <clio_runtime/evlat.h>

namespace clio::cte::filesystem {

namespace ev = clio::run::evlat;

namespace {

/** Wall-clock nanoseconds (the timestamps stat reports). */
inline clio::run::u64 NowNs() { return clio::cte::core::GetWallTimeNs(); }

// ---- ShardOp request / response codec ----

/** Encode a request. */
void EncReq(const Runtime::FsReq &r, std::string *out) {
  FsEnc e(out);
  e.Str(r.dir_);
  e.Str(r.leaf_);
  e.Str(r.str_);
  e.Str(r.str2_);
  e.U64(r.id_);
  e.U64(r.a_);
  e.U64(r.b_);
  e.U32(r.type_);
  e.U32(r.flags_);
  e.U32(r.mode_);
  e.U32(r.uid_);
  e.U32(r.gid_);
  e.U64(r.dir_id_);
  e.U32(r.block_);
  e.U32(r.from_);
  e.U64(r.have_);
  e.U64(r.reg_);
}

/** Decode a request. @return false on a malformed buffer */
bool DecReq(const std::string &s, Runtime::FsReq *r) {
  FsDec d(s.data(), s.size());
  return d.Str(&r->dir_) && d.Str(&r->leaf_) && d.Str(&r->str_) &&
         d.Str(&r->str2_) && d.U64(&r->id_) && d.U64(&r->a_) &&
         d.U64(&r->b_) && d.U32(&r->type_) && d.U32(&r->flags_) &&
         d.U32(&r->mode_) && d.U32(&r->uid_) && d.U32(&r->gid_) &&
         d.U64(&r->dir_id_) && d.U32(&r->block_) && d.U32(&r->from_) &&
         d.U64(&r->have_) && d.U64(&r->reg_);
}

/** Encode a response. */
void EncResp(const Runtime::FsResp &r, std::string *out) {
  FsEnc e(out);
  e.U32(r.rc_);
  e.U64(r.id_);
  e.U32(r.type_);
  e.U64(r.old_id_);
  e.U32(r.old_type_);
  e.U32(r.created_);
  e.U64(r.handle_);
  Runtime::EncodeAttr(e, r.attr_);
  e.Str(r.str_);
  e.U64(r.reg_);
  e.Str(r.deltas_);
}

/** Decode a response. @return false on a malformed buffer */
bool DecResp(const std::string &s, Runtime::FsResp *r) {
  FsDec d(s.data(), s.size());
  return d.U32(&r->rc_) && d.U64(&r->id_) && d.U32(&r->type_) &&
         d.U64(&r->old_id_) && d.U32(&r->old_type_) && d.U32(&r->created_) &&
         d.U64(&r->handle_) && Runtime::DecodeAttr(d, &r->attr_) && d.Str(&r->str_) &&
         d.U64(&r->reg_) && d.Str(&r->deltas_);
}

}  // namespace

void Runtime::EncodeAttr(FsEnc &e, const FsAttr &a) {
  e.U64(a.id_);
  e.U32(a.type_);
  e.U64(a.size_);
  e.U32(a.nlink_);
  e.U32(a.mode_);
  e.U32(a.uid_);
  e.U32(a.gid_);
  e.U64(a.atime_);
  e.U64(a.mtime_);
  e.U64(a.ctime_);
  e.U64(a.parent_);
  e.Str(a.parent_key_);
  e.U32(a.volume_flags_);
}

bool Runtime::DecodeAttr(FsDec &d, FsAttr *a) {
  return d.U64(&a->id_) && d.U32(&a->type_) && d.U64(&a->size_) &&
         d.U32(&a->nlink_) && d.U32(&a->mode_) && d.U32(&a->uid_) &&
         d.U32(&a->gid_) && d.U64(&a->atime_) && d.U64(&a->mtime_) &&
         d.U64(&a->ctime_) && d.U64(&a->parent_) && d.Str(&a->parent_key_) &&
         d.U32(&a->volume_flags_);
}

// ===========================================================================
// Ownership and transport
// ===========================================================================

clio::run::u32 Runtime::NumContainers() {
  if (num_containers_ != 0) return num_containers_;
  auto *pm = CLIO_POOL_MANAGER;
  const clio::run::PoolInfo *pi =
      pm != nullptr ? pm->GetPoolInfo(pool_id_) : nullptr;
  if (pi != nullptr && pi->num_containers_ != 0) {
    num_containers_ = pi->num_containers_;
    return num_containers_;
  }
  // Pool not registered yet (still inside Create): the pool has one
  // container per node, so the host count is the same number.
  auto *ipc = CLIO_IPC;
  const size_t n = ipc != nullptr ? ipc->GetNumHosts() : 1;
  return n == 0 ? 1u : static_cast<clio::run::u32>(n);
}

clio::run::u32 Runtime::InodeHome(clio::run::u64 packed) {
  return FsIdHasHome(packed)
             ? FsIdHome(packed)
             : static_cast<clio::run::u32>(FsMix64(packed)) % NumContainers();
}

clio::run::u32 Runtime::InodeOwner(clio::run::u64 packed) {
  const clio::run::u32 home = InodeHome(packed);
  // While the home's node is dead its successor serves the inode, loaded
  // from its CTE record (which replication keeps reachable) -- the same
  // rule the CTE uses for the dead node's blobs.
  return clio::cte::core::FailoverContainer(pool_id_, home);
}

clio::run::TaskResume Runtime::CallShard(clio::run::u32 target,
                                         clio::run::u32 op, const FsReq &req,
                                         FsResp &resp, bool *node_lost) {
  CLIO_TASK_BODY_BEGIN
  if (node_lost != nullptr) *node_lost = false;
  if (target == container_id_) {
    CLIO_CO_AWAIT(ExecShardOp(op, req, resp));
    CLIO_CO_RETURN;
  }
  std::string enc;
  EncReq(req, &enc);
  const auto shard_t0 = std::chrono::steady_clock::now();
  auto t = self_.AsyncShardOp(
      op, enc,
      clio::run::PoolQuery::DirectId(
          static_cast<clio::run::ContainerId>(target)));
  CLIO_CO_AWAIT(t);
  resp = FsResp();
  if (t->GetReturnCode() != 0 || !DecResp(t->resp_.str(), &resp)) {
    // The owner is unreachable (dead node, network timeout): the state it
    // owns is unavailable, which POSIX can only express as an I/O error.
    if (node_lost != nullptr &&
        clio::cte::core::IsNodeLostRc(
            static_cast<clio::run::u32>(t->GetReturnCode()))) {
      *node_lost = true;
    }
    resp = FsResp();
    resp.rc_ = EIO;
  }
  {
    // A shard call that took seconds, and where it went (#1169: a stat
    // took 274 s with two nodes down). Rate-limited.
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - shard_t0)
                          .count();
    if (ms > 2000.0) {
      static std::atomic<clio::run::u64> logged{0};
      const clio::run::u64 k = logged.fetch_add(1, std::memory_order_relaxed);
      if (k < 16 || k % 256 == 0) {
        auto *pm = CLIO_POOL_MANAGER;
        HLOG(kWarning, "filesystem: shard op {} to container {} (node {}, "
             "alive {}) took {} ms; task rc {}, resp rc {} ({} such calls)",
             op, target, pm->GetContainerNodeId(pool_id_, target),
             clio::cte::core::ContainerNodeAlive(pool_id_, target), ms,
             t->GetReturnCode(), resp.rc_, k + 1);
      }
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::ShardOp(clio::run::shared_ptr<ShardOpTask> &task) {
  CLIO_TASK_BODY_BEGIN
  FsReq req;
  FsResp resp;
  if (task->op_ == kShardNsDrain) {
    CLIO_CO_AWAIT(NsFastTick());
  } else if (task->op_ == kShardPurgeDrain) {
    CLIO_CO_AWAIT(SizeDrain());
    CLIO_CO_AWAIT(ExpireAppendRanges());
    CLIO_CO_AWAIT(PurgeDrain());
  } else if (!DecReq(task->req_.str(), &req)) {
    resp.rc_ = EINVAL;
  } else {
    CLIO_CO_AWAIT(ExecShardOp(task->op_, req, resp));
  }
  std::string enc;
  EncResp(resp, &enc);
  task->resp_ = clio::run::priv::string(CTP_MALLOC, enc);
  task->return_code_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::Future<ShardOpTask> Runtime::SendShard(clio::run::u32 target,
                                                  clio::run::u32 op,
                                                  const FsReq &req) {
  std::string enc;
  EncReq(req, &enc);
  return self_.AsyncShardOp(
      op, enc,
      clio::run::PoolQuery::DirectId(
          static_cast<clio::run::ContainerId>(target)));
}

bool Runtime::ReadShardResp(clio::run::Future<ShardOpTask> &f, FsResp *resp) {
  *resp = FsResp();
  if (f->GetReturnCode() != 0 || !DecResp(f->resp_.str(), resp)) {
    *resp = FsResp();
    resp->rc_ = EIO;
    return false;
  }
  return true;
}

clio::run::TaskResume Runtime::ExecShardOp(clio::run::u32 op, const FsReq &req,
                                           FsResp &resp) {
  CLIO_TASK_BODY_BEGIN
  int rc = 0;
  // Records dirtied from here on are this op's: only their failure fails it.
  const clio::run::u64 since = DirtySeq();
  switch (op) {
    case kTableInsert:
    case kTableRemove:
    case kTableRespell:
      CLIO_CO_AWAIT(RowMutation(op, req, resp));
      rc = static_cast<int>(resp.rc_);
      break;
    case kTableCreate:
      CLIO_CO_AWAIT(CreateTable(req, resp));
      rc = static_cast<int>(resp.rc_);
      break;
    case kTableSeal:
      CLIO_CO_AWAIT(SealBlock(req, resp));
      rc = static_cast<int>(resp.rc_);
      break;
    case kTableFetch:
      CLIO_CO_AWAIT(ServeBlockFetch(req, resp));
      rc = static_cast<int>(resp.rc_);
      break;
    case kTablePush: rc = ApplyTablePush(req); break;
    case kTableStandInFetch: rc = ServeStandInFetch(req, resp); break;
    case kTableLoadBlocks:
      CLIO_CO_AWAIT(LoadBlockGroup(req, resp));
      rc = static_cast<int>(resp.rc_);
      break;
    case kTableInstall:
      CLIO_CO_AWAIT(InstallBlock(req, resp));
      rc = static_cast<int>(resp.rc_);
      break;
    case kTableDrop:
      CLIO_CO_AWAIT(DropBlock(req, resp));
      rc = static_cast<int>(resp.rc_);
      break;
    case kShardInodePush: rc = ApplyInodePush(req); break;
    case kShardPurgeLocal:
      CLIO_CO_AWAIT(PurgeLocal(req));
      break;
    case kShardInodeCreate:
      CLIO_CO_AWAIT(InodeCreateOpen(req, resp, rc));
      break;
    case kShardInodeDiscard:
      CLIO_CO_AWAIT(InodeDiscard(req));
      break;
    default:
      if (op >= kShardInodeStat && op <= kShardInodeXattr) {
        CLIO_CO_AWAIT(ExecInodeOp(op, req, resp, rc));
      } else {
        rc = EINVAL;
      }
      break;
  }
  // Inode changes are durable (in CTE) and pushed before the caller hears.
  // Not for cache traffic (pushes, fetches): those change no inode, and a
  // push handler that waited for this node's own inode stores -- which push
  // back to the sender, whose handler waits the same way -- deadlocked two
  // nodes storing inodes cached on each other.
  if (op != kTablePush && op != kShardInodePush &&
      op != kTableFetch && op != kTableStandInFetch &&
      op != kTableLoadBlocks) {
    int ferr = 0;
    const unsigned long long ev_f0 = ev::Now();
    CLIO_CO_AWAIT(FlushInodes(&ferr, since));
    ev::Span(ev::kFsOpFlush, ev_f0, ev::Now());
    // The change could not be made durable (a full store): say so rather
    // than acknowledge it.
    if (rc == 0 && ferr != 0) rc = ferr;
  }
  resp.rc_ = static_cast<clio::run::u32>(rc);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::ExecInodeOp(clio::run::u32 op, const FsReq &req,
                                           FsResp &resp, int &rc) {
  CLIO_TASK_BODY_BEGIN
  rc = 0;
  int load_err = 0;
  CLIO_CO_AWAIT(EnsureInode(req.id_, &load_err));  // lazily loaded after a restart
  if (load_err != 0 && FindInode(req.id_) == nullptr) {
    // The record is unreadable (its owner down with no copy here, #1166):
    // nothing below may answer "no such inode" for it.
    rc = load_err;
    CLIO_CO_RETURN;
  }
  switch (op) {
    case kShardInodeStat: {
      auto fi = FindInode(req.id_);
      if (fi == nullptr) { rc = ENOENT; break; }
      std::lock_guard<std::mutex> g(meta_mu_);
      InodeAttrLocked(*fi, &resp.attr_);
      resp.str_ = fi->symlink_;
      // A stat from another container caches the result: register it, so
      // every later change reaches it before being acknowledged.
      const clio::run::u32 who = static_cast<clio::run::u32>(req.a_);
      if (req.b_ != 0 && who != container_id_) {
        fi->holders_[who] = reg_seq_++;
        fi->holder_lease_ms_[who] =
            SteadyMs() + kCacheLeaseMs + kCacheLeaseSlackMs;
      }
      break;
    }
    case kShardInodeOpen:
      if ((req.flags_ & O_TRUNC) != 0 && FindInode(req.id_) != nullptr) {
        FsReq tr;
        tr.id_ = req.id_;
        tr.a_ = 0;
        FsResp ignored;
        CLIO_CO_AWAIT(InodeTruncate(tr, ignored));
      }
      rc = InodeOpenLocal(req, resp);
      break;
    case kShardInodeNlink: rc = InodeNlink(req, resp); break;
    case kShardInodeSetAttr: rc = InodeSetAttr(req, resp); break;
    case kShardInodeTruncate:
      CLIO_CO_AWAIT(InodeTruncate(req, resp));
      rc = static_cast<int>(resp.rc_);
      break;
    case kShardInodeXattr:
      CLIO_CO_AWAIT(InodeXattr(req, resp));
      rc = static_cast<int>(resp.rc_);
      break;
    default:
      rc = EINVAL;
      break;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

// ===========================================================================
// Inodes
// ===========================================================================

clio::cte::core::TagId Runtime::MintIdFor(clio::run::u32 home) {
  std::lock_guard<std::mutex> g(meta_mu_);
  // Only ids covered by the durable reservation are handed out, so a
  // restart can never mint one that is still in use (EnsureIdReserve keeps
  // the reservation ahead; null asks the caller to wait for it).
  if (next_minor_ >= minted_hi_) return clio::cte::core::TagId::GetNull();
  const clio::run::u32 major =
      kFsIdFlag | kFsIdMinterFlag |
      ((container_id_ & kFsIdMinterMask) << kFsIdMinterShift) |
      (home & kFsHomeMask);
  return clio::cte::core::TagId(major, next_minor_++);
}

clio::run::TaskResume Runtime::MintIdReady(clio::run::u32 home,
                                           clio::cte::core::TagId &id,
                                           int &rc) {
  CLIO_TASK_BODY_BEGIN
  rc = 0;
  for (;;) {
    id = MintIdFor(home);
    if (!id.IsNull()) break;
    bool ok = false;
    CLIO_CO_AWAIT(EnsureIdReserve(1024, ok));
    if (!ok) {
      rc = EIO;  // never mint past a reservation that is not recorded
      break;
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

std::shared_ptr<Runtime::InodeState> Runtime::FindInode(clio::run::u64 packed) {
  std::lock_guard<std::mutex> g(meta_mu_);
  auto it = inodes_.find(packed);
  return it == inodes_.end() ? nullptr : it->second;
}

void Runtime::InodeAttrLocked(const InodeState &fi, FsAttr *attr) {
  const bool dir = fi.type_ == kFsTypeDir;
  attr->id_ = FsPack(fi.tag_id_);
  attr->type_ = fi.type_;
  attr->size_ = dir ? 0 : FileSize(fi);
  // A directory has one name, but "." and its own row: POSIX counts 2 (and
  // no subdirectory links, which tools treat as "unknown", as on btrfs).
  attr->nlink_ = dir ? 2u : fi.nlink_;
  attr->mode_ = fi.mode_;
  attr->uid_ = fi.uid_;
  attr->gid_ = fi.gid_;
  attr->atime_ = fi.atime_;
  attr->mtime_ = fi.mtime_;
  attr->ctime_ = fi.ctime_;
  attr->parent_ = fi.parent_;
  attr->parent_key_ = fi.parent_key_;
  attr->volume_flags_ = fi.volume_flags_;
}

int Runtime::InodeOpenLocal(const FsReq &req, FsResp &resp) {
  auto fi = FindInode(req.id_);
  if (fi == nullptr) return ENOENT;
  const clio::run::u64 h =
      FsMakeHandle(container_id_, next_handle_.fetch_add(1));
  std::lock_guard<std::mutex> g(meta_mu_);
  fi->open_count_++;
  handles_[h] = fi;
  resp.handle_ = h;
  InodeAttrLocked(*fi, &resp.attr_);
  if (!req.dir_.empty()) {
    fi->mirror_path_ = req.dir_;  // the name it was opened by (mirror key)
    MirrorFile(fi->mirror_path_, *fi);
  }
  return 0;
}

void Runtime::DropInodeLocked(const std::shared_ptr<InodeState> &fi,
                              bool size_trusted) {
  if (fi->open_count_ > 0) {
    // POSIX: an unlinked file lives until its last descriptor closes.
    fi->orphan_ = true;
    LogInode(*fi);
    return;
  }
  const clio::run::u64 packed = FsPack(fi->tag_id_);
  if (fi->orphan_) {
    std::lock_guard<std::mutex> g(purge_mu_);
    orphans_dirty_ = true;  // it leaves the orphan record
  }
  inodes_.erase(packed);
  // Its record still reads nlink 1 until the purge deletes it: an open
  // racing the rename that dropped it must not load it back (#1150).
  dying_.insert(packed);
  PurgeItem item;
  item.id_ = fi->tag_id_;
  item.data_ = true;  // pages and the inode record (symlinks have one too)
  item.xattr_ = fi->has_xattr_;
  RecordPurgeExtent(*fi, size_trusted, &item);
  QueuePurge(item);
}

void Runtime::RecordPurgeExtent(const InodeState &fi, bool size_trusted,
                                PurgeItem *item) {
  // The size is the stream's, read from the co-located stream: that exists
  // only where this container is the inode's home. A stand-in (failover)
  // holds no stream, and reading one would report a false zero.
  if (!size_trusted || fi.type_ != kFsTypeFile || stream_hold_ == nullptr) {
    return;  // extent unknown
  }
  item->size_ = FileSize(fi);
  item->extent_known_ = true;
}

int Runtime::InodeNlink(const FsReq &req, FsResp &resp) {
  const auto delta = static_cast<clio::run::i64>(req.a_);
  std::lock_guard<std::mutex> g(meta_mu_);
  auto it = inodes_.find(req.id_);
  if (it == inodes_.end()) return ENOENT;
  std::shared_ptr<InodeState> fi = it->second;
  const clio::run::i64 n = static_cast<clio::run::i64>(fi->nlink_) + delta;
  fi->nlink_ = n > 0 ? static_cast<clio::run::u32>(n) : 0u;
  fi->ctime_ = NowNs();
  // Keep the name the inode publishes itself under current: a size push
  // after a rename must not re-publish (resurrect) the old name.
  if (!req.str_.empty() && fi->type_ != kFsTypeDir) {
    if (delta > 0) {
      fi->mirror_path_ = req.str_;
    } else if (fi->mirror_path_ == req.str_) {
      fi->mirror_path_.clear();
    }
  }
  InodeAttrLocked(*fi, &resp.attr_);
  if (fi->nlink_ == 0) {
    DropInodeLocked(fi, /*size_trusted=*/true);
  } else {
    LogInode(*fi);
  }
  return 0;
}

int Runtime::InodeSetAttr(const FsReq &req, FsResp &resp) {
  const clio::run::u32 f = req.flags_;
  std::lock_guard<std::mutex> g(meta_mu_);
  auto it = inodes_.find(req.id_);
  if (it == inodes_.end()) return ENOENT;
  InodeState &fi = *it->second;
  const clio::run::u64 now = NowNs();
  const bool dir = fi.type_ == kFsTypeDir;
  if ((f & kAccessTouch) && dir) {
    // Directory reads (listings) do not track atime.
    InodeAttrLocked(fi, &resp.attr_);
    return 0;
  }
  if (f & kAccessTouch) {
    // relatime (the Linux default): a read moves atime only when it is not
    // newer than the last change or is a day old. A read is not a change,
    // so ctime stays.
    constexpr clio::run::u64 kDayNs = 86400ull * 1000000000ull;
    if ((f & kAccessStrict) || fi.atime_ <= fi.mtime_ ||
        fi.atime_ <= fi.ctime_ || now - fi.atime_ >= kDayNs) {
      fi.atime_ = now;
      LogInode(fi);
      if (!fi.mirror_path_.empty()) MirrorFile(fi.mirror_path_, fi);
    }
    InodeAttrLocked(fi, &resp.attr_);
    return 0;
  }
  if (f & kSetAtimeNow) fi.atime_ = now;
  else if (f & kSetAtime) fi.atime_ = req.a_;
  if (f & kSetMtimeNow) fi.mtime_ = now;
  else if (f & kSetMtime) fi.mtime_ = req.b_;
  if (f & (kSetMtimeNow | kSetMtime)) fi.mtime_set_ns_ = now;
  if (f & kSetUid) fi.uid_ = req.uid_;
  if (f & kSetGid) fi.gid_ = req.gid_;
  if (f & kSetMode) fi.mode_ = req.mode_ & 07777u;
  if ((f & kSetParent) && dir) {
    fi.parent_ = req.b_;
    fi.parent_key_ = req.str_;
  }
  fi.ctime_ = now;
  LogInode(fi);
  if (!fi.mirror_path_.empty()) MirrorFile(fi.mirror_path_, fi);
  InodeAttrLocked(fi, &resp.attr_);
  return 0;
}

clio::run::TaskResume Runtime::InodeTruncate(const FsReq &req, FsResp &resp) {
  CLIO_TASK_BODY_BEGIN
  resp.rc_ = 0;
  std::shared_ptr<InodeState> fi = FindInode(req.id_);
  if (fi == nullptr) {
    resp.rc_ = ENOENT;
    CLIO_CO_RETURN;
  }
  const clio::cte::core::TagId tag = fi->tag_id_;
  const clio::run::u64 new_size = req.a_;
  clio::run::u64 old_size = 0;
  {
    clio::run::u32 rc = 0;
    CLIO_CO_AWAIT(FileSizeOp(tag, clio::cte::stream::StreamSizeOp::kSet,
                             new_size, &old_size, nullptr, &rc));
    if (rc != 0) {
      resp.rc_ = EIO;
      CLIO_CO_RETURN;
    }
  }
  old_size = std::max<clio::run::u64>(old_size, req.b_);
  {
    std::lock_guard<std::mutex> g(meta_mu_);
    const clio::run::u64 now = NowNs();
    fi->mtime_ = fi->ctime_ = now;
    LogInode(*fi);
    if (!fi->mirror_path_.empty()) MirrorFile(fi->mirror_path_, *fi);
    InodeAttrLocked(*fi, &resp.attr_);
  }
  // The pages are the truth, not the size: a client pushes a file's size
  // lazily (at close), so the recorded size can trail bytes already stored.
  // Cut whatever the boundary page holds past the new end, and let the page
  // walk below go past the recorded end while pages exist.
  const clio::run::u64 boundary_page = new_size / kFsPageSize;
  const clio::run::u64 boundary_off = new_size % kFsPageSize;
  {
    auto bs = cte_.AsyncGetBlobSize(tag, std::to_string(boundary_page));
    CLIO_CO_AWAIT(bs);
    if (bs->GetReturnCode() == 0 && bs->size_ > boundary_off) {
      auto tb = cte_.AsyncTruncateBlob(tag, std::to_string(boundary_page),
                                       boundary_off,
                                       clio::run::PoolQuery::Dynamic());
      CLIO_CO_AWAIT(tb);
      old_size = std::max(old_size, boundary_page * kFsPageSize + bs->size_);
    }
  }
  if (new_size >= old_size) CLIO_CO_RETURN;
  // Shrink: physically ZERO the boundary page's surviving tail (a later
  // write past EOF re-extends over it; the old bytes must not reappear
  // inside what is now a hole), then drop whole pages beyond it.
  const clio::run::u64 page_tail_end =
      std::min(kFsPageSize, old_size - boundary_page * kFsPageSize);
  constexpr clio::run::u64 kZChunk = 64 * 1024;
  auto *ipc = CLIO_IPC;
  for (clio::run::u64 zoff = boundary_off; zoff < page_tail_end;) {
    const clio::run::u64 zlen = std::min(kZChunk, page_tail_end - zoff);
    ctp::ipc::FullPtr<char> zbuf = ipc->AllocateBuffer(zlen);
    if (zbuf.IsNull()) {
      resp.rc_ = EIO;
      CLIO_CO_RETURN;
    }
    std::memset(zbuf.ptr_, 0, zlen);
    auto zp = cte_.AsyncPutBlob(tag, std::to_string(boundary_page), zoff, zlen,
                                zbuf.shm_.template Cast<void>(),
                                page_score_.ScoreOf(boundary_page),
                                clio::cte::core::Context(), 0u,
                                clio::run::PoolQuery::Dynamic());
    CLIO_CO_AWAIT(zp);
    ipc->FreeBuffer(zbuf);
    if (zp->GetReturnCode() != 0) {
      HLOG(kWarning, "filesystem: zeroing truncated tail of {} page {} "
           "[{}, +{}) failed rc={}", req.id_, boundary_page, zoff, zlen,
           zp->GetReturnCode());
      resp.rc_ = EIO;
      CLIO_CO_RETURN;
    }
    zoff += zlen;
  }
  // Pages up to the recorded end, then on while they exist (the recorded
  // end may trail stored pages; see above). Bounded probe past the end.
  constexpr clio::run::u64 kMaxProbePages = 64;
  const clio::run::u64 last_page = (old_size - 1) / kFsPageSize;
  for (clio::run::u64 pg = boundary_page + 1;
       pg <= last_page + kMaxProbePages; ++pg) {
    auto d = cte_.AsyncDelBlob(tag, std::to_string(pg),
                               clio::run::PoolQuery::Dynamic());
    CLIO_CO_AWAIT(d);
    if (pg > last_page && d->GetReturnCode() != 0) break;  // past the data
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

void Runtime::QueuePurge(const PurgeItem &item) {
  if (!item.data_ && !item.xattr_) return;
  std::lock_guard<std::mutex> g(purge_mu_);
  purge_pending_.push_back(item);
}

clio::run::TaskResume Runtime::SizeDrain() {
  CLIO_TASK_BODY_BEGIN
  ShmFsCacheRoot *root = shm_fs_cache_.root();
  if (root == nullptr || !root->sizes_.Valid()) CLIO_CO_RETURN;
  std::vector<ShmSizeTable::Entry> pending;
  root->sizes_.Snapshot(&pending);
  // One size advance per file, all in flight together; each goes to its
  // inode's home (AdvanceSize forwards).
  constexpr size_t kSizeBatch = 256;
  for (size_t i = 0; i < pending.size(); i += kSizeBatch) {
    const size_t end = std::min(pending.size(), i + kSizeBatch);
    std::vector<clio::run::Future<AdvanceSizeTask>> futs;
    futs.reserve(end - i);
    for (size_t j = i; j < end; ++j) {
      futs.push_back(self_.AsyncAdvanceSize(pending[j].key, pending[j].end,
                                            pending[j].wtime));
    }
    for (size_t j = i; j < end; ++j) {
      auto &f = futs[j - i];
      CLIO_CO_AWAIT(f);
      const clio::run::u32 rc = f->GetReturnCode();
      // ENOENT for a file the shared-memory namespace still holds is its
      // create not applied yet: keep the size for the next period. For a
      // removed file the size has nowhere to go.
      const bool not_yet = rc == ENOENT && ns_view_.Ready() &&
                           ns_view_.HasInode(pending[j].key);
      if (rc == 0 || (rc == ENOENT && !not_yet)) {
        root->sizes_.Retire(pending[j].key, pending[j].end);
      } else if (!not_yet) {
        HLOG(kWarning, "filesystem: size drain of inode {} to {} failed "
             "(rc={}); retrying next period", pending[j].key, pending[j].end,
             rc);
      }
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

// ===========================================================================
// Appends: IdentifyRange / CommitRange at the owner, the merge at the origin
// (SHM_CACHE_DESIGN.md "Appends")
// ===========================================================================

namespace {
/** A reserved range whose origin never commits becomes a hole after this. */
constexpr clio::run::u64 kAppendCommitTimeoutMs = 30000;
/** Bytes of a staged generation moved per read/write slice. */
constexpr clio::run::u64 kAppendSliceBytes = 16ull << 20;
/** Poll interval while appends still copy into a sealed generation (us). */
constexpr double kAppendInflightPollUs = 50.0;

/** @return the staging blob of generation `gen` of node `node` */
std::string AppendBlobName(clio::run::u32 node, clio::run::u32 gen) {
  return "app." + std::to_string(node) + "." + std::to_string(gen);
}
}  // namespace

bool Runtime::AdvanceAppendFrontierLocked(AppendRanges &r,
                                          clio::run::u64 now_ms,
                                          clio::run::u64 *frontier) {
  bool advanced = false;
  while (!r.ranges_.empty()) {
    auto it = r.ranges_.begin();
    const bool expired = now_ms - it->second.reserved_ms_ >=
                         kAppendCommitTimeoutMs;
    if (!it->second.committed_ && !expired) break;
    if (!it->second.committed_) {
      HLOG(kWarning, "filesystem: append range [{}, +{}) was never "
           "committed; publishing it as a hole", it->first,
           it->second.len_);
    }
    *frontier = it->first + it->second.len_;
    r.ranges_.erase(it);
    advanced = true;
  }
  return advanced;
}

clio::run::TaskResume Runtime::PublishAppendSize(clio::run::u64 tag_packed,
                                                 clio::run::u64 size,
                                                 clio::run::u64 mtime_ns) {
  CLIO_TASK_BODY_BEGIN
  {
    // This container is the file's home: the ordinary size advance there
    // (stream size, inode record, mirror).
    auto f = self_.AsyncAdvanceSizeAt(
        clio::run::PoolQuery::DirectId(
            static_cast<clio::run::ContainerId>(container_id_)),
        tag_packed, size, 0u, mtime_ns);
    CLIO_CO_AWAIT(f);
    if (f->GetReturnCode() != 0) {
      HLOG(kWarning, "filesystem: publishing appended size {} of inode {} "
           "failed (rc={})", size, tag_packed, f->GetReturnCode());
    }
    if (ns_view_.Ready()) ns_view_.RaiseSize(tag_packed, size, mtime_ns);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::IdentifyRange(
    clio::run::shared_ptr<IdentifyRangeTask> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    const clio::run::u64 tag = task->tag_packed_;
    std::string members = task->members_.str();
    if (members.empty()) {
      PutRangeRec(&members, task->node_, task->gen_, task->len_);
    }
    // The file's size, when no reservation of it is open: reservations
    // start at the end every node can already see.
    bool need_size = false;
    {
      std::lock_guard<std::mutex> g(append_mu_);
      need_size = append_ranges_[tag].ranges_.empty();
    }
    clio::run::u64 cur = 0;
    if (need_size) {
      clio::run::u32 rc = 0;
      CLIO_CO_AWAIT(FileSizeOp(FsUnpack(tag),
                               clio::cte::stream::StreamSizeOp::kGet, 0,
                               nullptr, &cur, &rc));
      if (rc != 0) {
        task->return_code_ = EIO;
        CLIO_CO_RETURN;
      }
    }
    std::string offsets;
    {
      std::lock_guard<std::mutex> g(append_mu_);
      AppendRanges &r = append_ranges_[tag];
      if (need_size && r.ranges_.empty()) r.end_ = cur;
      const clio::run::u64 now_ms = SteadyMs();
      task->base_ = r.end_;
      clio::run::u32 node = 0;
      clio::run::u32 gen = 0;
      clio::run::u64 len = 0;
      for (size_t i = 0; GetRangeRec(members, i, &node, &gen, &len); ++i) {
        if (len == 0) continue;
        AppendRange &ar = r.ranges_[r.end_];
        ar.len_ = len;
        ar.committed_ = false;
        ar.reserved_ms_ = now_ms;
        PutRangeRec(&offsets, node, gen, r.end_);
        r.end_ += len;
      }
    }
    task->offsets_ = clio::run::priv::string(CTP_MALLOC, offsets);
    task->return_code_ = 0;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::CommitRange(
    clio::run::shared_ptr<CommitRangeTask> &task) {
  CLIO_TASK_BODY_BEGIN
  {
    clio::run::u64 frontier = 0;
    bool advanced = false;
    {
      std::lock_guard<std::mutex> g(append_mu_);
      auto it = append_ranges_.find(task->tag_packed_);
      if (it != append_ranges_.end()) {
        auto rit = it->second.ranges_.find(task->offset_);
        if (rit != it->second.ranges_.end()) rit->second.committed_ = true;
        advanced =
            AdvanceAppendFrontierLocked(it->second, SteadyMs(), &frontier);
      }
    }
    if (advanced) {
      CLIO_CO_AWAIT(
          PublishAppendSize(task->tag_packed_, frontier, task->mtime_ns_));
    }
    task->return_code_ = 0;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::ExpireAppendRanges() {
  CLIO_TASK_BODY_BEGIN
  {
    std::vector<std::pair<clio::run::u64, clio::run::u64>> publish;
    {
      std::lock_guard<std::mutex> g(append_mu_);
      const clio::run::u64 now_ms = SteadyMs();
      for (auto it = append_ranges_.begin(); it != append_ranges_.end();) {
        clio::run::u64 frontier = 0;
        if (AdvanceAppendFrontierLocked(it->second, now_ms, &frontier)) {
          publish.emplace_back(it->first, frontier);
        }
        it = it->second.ranges_.empty() ? append_ranges_.erase(it)
                                        : std::next(it);
      }
    }
    for (size_t i = 0; i < publish.size(); ++i) {
      CLIO_CO_AWAIT(PublishAppendSize(publish[i].first, publish[i].second,
                                      0));
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::MergeAppends(clio::run::u64 id) {
  CLIO_TASK_BODY_BEGIN
  {
    clio::run::u32 gen = 0;
    if (!ns_view_.AppendSeal(id, &gen)) CLIO_CO_RETURN;
    // Appends that took their offset before the seal finish copying.
    clio::run::u64 len = 0;
    while (ns_view_.AppendInflight(id, gen, &len)) {
      CLIO_CO_AWAIT(clio::run::yield(kAppendInflightPollUs));
    }
    const clio::cte::core::TagId tag = FsUnpack(id);
    const clio::run::u32 node = ns_view_.NodeId();
    const std::string blob = AppendBlobName(node, gen);
    clio::run::u64 off = 0;
    bool have = false;
    if (len != 0) {
      auto f = self_.AsyncIdentifyRange(id, node, gen, len);
      CLIO_CO_AWAIT(f);
      const std::string offsets = f->offsets_.str();
      clio::run::u32 n = 0;
      clio::run::u32 g = 0;
      clio::run::u64 o = 0;
      for (size_t i = 0; GetRangeRec(offsets, i, &n, &g, &o); ++i) {
        if (n == node && g == gen) {
          off = o;
          have = true;
          break;
        }
      }
      if (f->GetReturnCode() != 0 || !have) {
        HLOG(kError, "filesystem: no range for {} appended bytes of inode "
             "{} (rc={}); they are dropped", len, id, f->GetReturnCode());
      }
    }
    if (have) {
      // The bytes move once: from this node's staging blob into the file's
      // pages at the reserved offset.
      std::vector<char> buf(
          static_cast<size_t>(std::min(len, kAppendSliceBytes)));
      clio::run::u64 done = 0;
      while (done < len) {
        const clio::run::u64 n = std::min(len - done, kAppendSliceBytes);
        auto g = cte_.AsyncGetBlob(tag, blob, done, n, 0u, buf.data(),
                                   clio::run::PoolQuery::Dynamic());
        CLIO_CO_AWAIT(g);
        if (g->GetReturnCode() != 0) {
          HLOG(kError, "filesystem: reading staged appends {} of inode {} "
               "failed (rc={})", blob, id, g->GetReturnCode());
          break;
        }
        clio::run::u64 put = 0;
        while (put < n) {
          const clio::run::u64 cur = off + done + put;
          const clio::run::u64 page_off = cur % kFsPageSize;
          const clio::run::u64 k = std::min(kFsPageSize - page_off, n - put);
          auto p = cte_.AsyncPutBlob(tag, PageName(cur), page_off, k,
                                     buf.data() + put, -1.0f,
                                     clio::cte::core::Context(), 0u,
                                     clio::run::PoolQuery::Dynamic());
          CLIO_CO_AWAIT(p);
          put += k;
        }
        done += n;
      }
      auto c = self_.AsyncCommitRange(
          id, off, len,
          static_cast<clio::run::u64>(clio::cte::core::GetWallTimeNs()));
      CLIO_CO_AWAIT(c);
    }
    if (len != 0) {
      auto d = cte_.AsyncDelBlob(tag, blob);
      CLIO_CO_AWAIT(d);
    }
    ns_view_.AppendMerged(id, gen);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::RefillNsIds() {
  CLIO_TASK_BODY_BEGIN
  // Top the block up while half of it is left, so clients never wait.
  constexpr clio::run::u32 kNsIdBlock = 8192;
  if (!ns_view_.Ready() || ns_view_.IdsLeft() >= kNsIdBlock / 2) {
    CLIO_CO_RETURN;
  }
  bool ok = false;
  CLIO_CO_AWAIT(EnsureIdReserve(kNsIdBlock, ok));
  if (!ok) CLIO_CO_RETURN;  // retried next period
  {
    clio::run::u32 major = 0;
    clio::run::u64 first = 0;
    bool have = false;
    {
      std::lock_guard<std::mutex> g(meta_mu_);
      if (next_minor_ + kNsIdBlock <= minted_hi_) {
        first = next_minor_;
        next_minor_ += kNsIdBlock;
        // Home 0: the namespace segment exists on single-node volumes only.
        major = kFsIdFlag | kFsIdMinterFlag |
                ((container_id_ & kFsIdMinterMask) << kFsIdMinterShift);
        have = true;
      }
    }
    if (have) ns_view_.SetIdBlock(major, first, kNsIdBlock);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::NsApplyOne(const ns::OpRec &rec,
                                          const std::string &path,
                                          const std::string &path2,
                                          clio::run::u32 &rc) {
  CLIO_TASK_BODY_BEGIN
  rc = 0;
  if (rec.type_ == ns::kOpMkdir) {
    auto f = self_.AsyncMkdir(path);
    CLIO_CO_AWAIT(f);
    rc = f->GetReturnCode();
    if (rc == 0 && rec.mode_ != 0755u) {
      auto c = self_.AsyncChmod(path, rec.mode_);
      CLIO_CO_AWAIT(c);
      rc = c->GetReturnCode();
    }
  } else if (rec.type_ == ns::kOpUnlink) {
    auto f = self_.AsyncUnlink(path);
    CLIO_CO_AWAIT(f);
    rc = f->GetReturnCode();
  } else if (rec.type_ == ns::kOpRmdir) {
    auto f = self_.AsyncRmdir(path);
    CLIO_CO_AWAIT(f);
    rc = f->GetReturnCode();
  } else if (rec.type_ == ns::kOpRename) {
    auto f = self_.AsyncRename(path, path2);
    CLIO_CO_AWAIT(f);
    rc = f->GetReturnCode();
  } else if (rec.type_ == ns::kOpLink) {
    auto f = self_.AsyncLink(path, path2);
    CLIO_CO_AWAIT(f);
    rc = f->GetReturnCode();
  } else if (rec.type_ == ns::kOpSymlink) {
    auto f = self_.AsyncSymlink(path2, path);
    CLIO_CO_AWAIT(f);
    rc = f->GetReturnCode();
  } else if (rec.type_ == ns::kOpChmod) {
    auto f = self_.AsyncChmod(path, rec.mode_);
    CLIO_CO_AWAIT(f);
    rc = f->GetReturnCode();
  } else if (rec.type_ == ns::kOpChown) {
    auto f = self_.AsyncChown(path, rec.uid_, rec.gid_);
    CLIO_CO_AWAIT(f);
    rc = f->GetReturnCode();
  } else if (rec.type_ == ns::kOpUtimens) {
    const clio::run::u32 flags =
        (rec.t1_ != 0 ? 1u : 0u) | (rec.t2_ != 0 ? 2u : 0u);
    auto f = self_.AsyncUtimens(path, rec.t1_, rec.t2_, flags);
    CLIO_CO_AWAIT(f);
    rc = f->GetReturnCode();
  } else if (rec.type_ == ns::kOpSetxattr) {
    const size_t nul = path2.find('\0');
    const std::string name = path2.substr(0, nul);
    const std::string value =
        nul == std::string::npos ? std::string() : path2.substr(nul + 1);
    auto f = self_.AsyncSetxattr(path, name, value, rec.flags_);
    CLIO_CO_AWAIT(f);
    rc = f->GetReturnCode();
  } else if (rec.type_ == ns::kOpRemovexattr) {
    auto f = self_.AsyncRemovexattr(path, path2);
    CLIO_CO_AWAIT(f);
    rc = f->GetReturnCode();
  } else if (rec.type_ == ns::kOpTruncate) {
    auto f = self_.AsyncTruncate(path, rec.size_, rec.id_);
    CLIO_CO_AWAIT(f);
    rc = f->GetReturnCode();
  } else {
    rc = EINVAL;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::NsDrain() {
  CLIO_TASK_BODY_BEGIN
  if (!ns_view_.Ready()) CLIO_CO_RETURN;
  CLIO_CO_AWAIT(RefillNsIds());
  {
    constexpr size_t kCreateBatch = 256;
    ns::OpRec rec;
    std::string path;
    std::string path2;
    while (ns_view_.PeekOp(&rec, &path, &path2)) {
      if (rec.type_ == ns::kOpCreate) {
        // Consecutive creates go as one batch; the client minted each id.
        std::vector<MultiCreateEnt> ents;
        clio::run::u64 last_seq = 0;
        bool more = true;
        while (more && ents.size() < kCreateBatch &&
               rec.type_ == ns::kOpCreate) {
          MultiCreateEnt e;
          e.path_ = path;
          e.tag_packed_ = rec.id_;
          e.mode_ = rec.mode_;
          e.ctime_ns_ = rec.t1_;
          ents.push_back(std::move(e));
          last_seq = rec.seq_;
          ns_view_.PopOp();
          more = ns_view_.PeekOp(&rec, &path, &path2);
        }
        auto f = self_.AsyncMultiCreate(EncodeMultiCreate(ents));
        CLIO_CO_AWAIT(f);
        if (f->first_rc_ != 0 || f->GetReturnCode() != 0) {
          HLOG(kWarning, "filesystem: namespace drain: {} of {} creates "
               "failed (first rc={})", ents.size() - f->num_ok_,
               ents.size(), f->first_rc_);
        }
        // Landed: only now may waiters (fsync, xattrs) see them as applied.
        ns_view_.MarkApplied(last_seq);
        if (!more) break;
        continue;
      }
      if (rec.type_ == ns::kOpAppend) {
        // Everything staged for the file up to now merges in one pass.
        CLIO_CO_AWAIT(MergeAppends(rec.id_));
        ns_view_.PopOp();
        ns_view_.MarkApplied(rec.seq_);
        continue;
      }
      clio::run::u32 rc = 0;
      CLIO_CO_AWAIT(NsApplyOne(rec, path, path2, rc));
      if (rc != 0) {
        HLOG(kWarning, "filesystem: namespace drain: op {} on '{}' failed "
             "(rc={})", rec.type_, path, rc);
      }
      ns_view_.PopOp();
      ns_view_.MarkApplied(rec.seq_);
    }
  }
  CLIO_CO_AWAIT(RefillNsIds());
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::NsFastTick() {
  CLIO_TASK_BODY_BEGIN
  if (!ns_view_.Ready()) CLIO_CO_RETURN;
  CLIO_CO_AWAIT(RefillNsIds());
  // Namespace first: a size advance needs its file's create applied.
  if (!ns_view_.LogEmpty()) CLIO_CO_AWAIT(NsDrain());
  if (ns_view_.Urgent()) CLIO_CO_AWAIT(SizeDrain());
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::PurgeDrain() {
  CLIO_TASK_BODY_BEGIN
  // Anything a failed flush re-queued.
  CLIO_CO_AWAIT(FlushNames());
  CLIO_CO_AWAIT(FlushInodes());  // anything a handler left dirty
  // Bound what a power loss can take from the directory journal.
  SyncDirJournalIfDue();
  std::vector<PurgeItem> work;
  {
    std::lock_guard<std::mutex> g(purge_mu_);
    work.swap(purge_pending_);
  }
  // Pages are hash-spread over the containers. Each container gets ONE
  // request per batch of dead ids, listing only the ids whose pages it may
  // hold (see PurgeOwners), and deletes its own pages locally.
  constexpr size_t kPurgeBatch = 512;
  for (size_t i = 0; i < work.size(); i += kPurgeBatch) {
    const size_t end = std::min(work.size(), i + kPurgeBatch);
    for (size_t j = i; j < end; ++j) {
      if (!work[j].data_) continue;
      // Forget the stream first: a deferred append still in flight is then
      // discarded instead of re-creating pages after the purge below.
      clio::run::u32 rc = 0;
      CLIO_CO_AWAIT(FileSizeOp(work[j].id_,
                               clio::cte::stream::StreamSizeOp::kDrop, 0,
                               nullptr, nullptr, &rc));
    }
    std::vector<FsReq> requests = BuildPurgeRequests(work, i, end);
    CLIO_CO_AWAIT(SendPurgeRequests(requests));
    for (size_t j = i; j < end; ++j) {
      // The inode record: a key on this home's CTE container, which the
      // per-node page purge above does not reach when no page lives here.
      auto r = cte_.AsyncTagDelXattr(work[j].id_, std::string(), 0,
                                     clio::run::PoolQuery::Local());
      CLIO_CO_AWAIT(r);
    }
    {
      // The records are gone: EnsureInode finds nothing to load now.
      std::lock_guard<std::mutex> g(meta_mu_);
      for (size_t j = i; j < end; ++j) dying_.erase(FsPack(work[j].id_));
    }
    for (size_t j = i; j < end; ++j) {
      if (!work[j].xattr_ || xattr_tag_id_.IsNull()) continue;
      auto x = cte_.AsyncDelBlob(xattr_tag_id_,
                                 std::to_string(FsPack(work[j].id_)),
                                 clio::run::PoolQuery::Dynamic());
      CLIO_CO_AWAIT(x);
    }
  }
  CLIO_CO_AWAIT(StoreOrphans());
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::PurgeLocal(const FsReq &req) {
  CLIO_TASK_BODY_BEGIN
  FsDec dec(req.str_.data(), req.str_.size());
  clio::run::u64 packed = 0;
  std::vector<clio::cte::core::TagId> tags;
  while (dec.U64(&packed)) tags.push_back(FsUnpack(packed));
  if (!tags.empty()) {
    // Local: this container's share (accounting and page blobs) of each
    // file, in one call; a file with no pages here costs nothing.
    auto d = cte_.AsyncMultiDelTag(tags, clio::run::PoolQuery::Local());
    CLIO_CO_AWAIT(d);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

std::vector<clio::run::u32> Runtime::PurgeOwners(const PurgeItem &item) {
  const clio::run::u32 n = NumContainers();
  // Page placement is the CTE core's (DirectHash of BlobHash, failover to a
  // successor), so the owner count is the core pool's, not this pool's.
  const clio::run::u32 core_n =
      clio::cte::core::PoolContainers(clio::cte::core::kCtePoolId);
  if (!item.extent_known_ || core_n != n || core_n == 0) {
    std::vector<clio::run::u32> all(n);
    for (clio::run::u32 c = 0; c < n; ++c) all[c] = c;
    return all;
  }
  const clio::run::u64 pages = (item.size_ + kFsPageSize - 1) / kFsPageSize;
  if (pages > n) {
    std::vector<clio::run::u32> all(n);
    for (clio::run::u32 c = 0; c < n; ++c) all[c] = c;
    return all;
  }
  std::vector<bool> hit(n, false);
  for (clio::run::u64 pg = 0; pg < pages; ++pg) {
    const clio::run::u32 owner =
        clio::cte::core::BlobHash(item.id_, PageName(pg * kFsPageSize)) %
        core_n;
    hit[clio::cte::core::FailoverContainer(clio::cte::core::kCtePoolId,
                                           owner)] = true;
  }
  std::vector<clio::run::u32> owners;
  for (clio::run::u32 c = 0; c < n; ++c) {
    if (hit[c]) owners.push_back(c);
  }
  return owners;
}

std::vector<Runtime::FsReq> Runtime::BuildPurgeRequests(
    const std::vector<PurgeItem> &work, size_t begin, size_t end) {
  std::vector<FsReq> requests(NumContainers());
  for (size_t j = begin; j < end; ++j) {
    if (!work[j].data_) continue;
    const clio::run::u64 packed = FsPack(work[j].id_);
    // This container always purges too (no network): it is the inode's
    // home and may hold the tag's local share even when no page hashes here.
    bool home_listed = false;
    for (clio::run::u32 c : PurgeOwners(work[j])) {
      FsEnc(&requests[c].str_).U64(packed);
      home_listed = home_listed || c == container_id_;
    }
    if (!home_listed && container_id_ < requests.size()) {
      FsEnc(&requests[container_id_].str_).U64(packed);
    }
  }
  return requests;
}

clio::run::TaskResume Runtime::SendPurgeRequests(const std::vector<FsReq> &reqs) {
  CLIO_TASK_BODY_BEGIN
  // Every remote request is in flight before any reply is awaited, so the
  // containers purge in parallel. A purge that fails or times out leaves its
  // pages in place (a leak), as before: the reply is not needed.
  std::vector<clio::run::Future<ShardOpTask>> inflight;
  for (clio::run::u32 c = 0; c < reqs.size(); ++c) {
    if (reqs[c].str_.empty()) continue;
    if (c == container_id_) {
      FsResp ignored;
      CLIO_CO_AWAIT(ExecShardOp(kShardPurgeLocal, reqs[c], ignored));
      continue;
    }
    inflight.push_back(SendShard(c, kShardPurgeLocal, reqs[c]));
  }
  for (auto &f : inflight) {
    CLIO_CO_AWAIT(f);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

// ===========================================================================
// CTE tag-name publisher
// ===========================================================================

void Runtime::PublishName(clio::cte::core::TagNameOp op,
                          const clio::cte::core::TagId &id,
                          const std::string &name, const std::string &name2) {
  std::lock_guard<std::mutex> g(tn_mu_);
  clio::cte::core::EncodeTagNameOp(&tn_batch_, op, id, 0, name, name2);
}

clio::run::TaskResume Runtime::FlushNames() {
  CLIO_TASK_BODY_BEGIN
  std::string batch;
  {
    std::lock_guard<std::mutex> g(tn_mu_);
    batch.swap(tn_batch_);
  }
  if (!batch.empty()) {
    // One batched core call; the local CTE sends each step to the container
    // holding that name or tag. A call that fails outright is retried by the
    // next flush (#1182); every step is idempotent, so re-sending is harmless.
    auto u = cte_.AsyncUpdateTagNames(batch, clio::run::PoolQuery::Local());
    CLIO_CO_AWAIT(u);
    // Re-send what did not land: the whole batch when the call failed, else
    // each op a down container could not apply (kTnUnreached). The drain
    // flushes again every tick, so a name lands once its holder is back.
    std::string again;
    if (u->GetReturnCode() != 0) {
      again = batch;
    } else {
      std::vector<clio::cte::core::TagNameOpRec> ops, res;
      const std::string rb = u->results_.str();
      clio::cte::core::DecodeTagNameOps(batch.data(), batch.size(), &ops);
      clio::cte::core::DecodeTagNameOps(rb.data(), rb.size(), &res);
      for (size_t i = 0; i < ops.size(); ++i) {
        const bool unreached =
            i >= res.size() || static_cast<clio::run::u32>(res[i].op_) ==
                                   clio::cte::core::kTnUnreached;
        if (!unreached) continue;
        const auto &r = ops[i];
        clio::cte::core::EncodeTagNameOp(&again, r.op_, r.tag_, r.flags_,
                                         r.name_, r.name2_);
      }
    }
    if (again.empty()) {
      tn_flush_failures_ = 0;
    } else {
      ++tn_flush_failures_;
      if (tn_flush_failures_ <= 3 ||
          (tn_flush_failures_ & (tn_flush_failures_ - 1)) == 0) {
        HLOG(kWarning, "filesystem: {} byte(s) of tag-name ops did not land "
             "(rc {}); re-queued, {} attempt(s) so far", again.size(),
             u->GetReturnCode(), tn_flush_failures_);
      }
      std::lock_guard<std::mutex> g(tn_mu_);
      tn_batch_.insert(0, again);  // ahead of what was published since
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

// ===========================================================================
// System records: id reservations and orphans (CTE blobs in sys_tag_id_)
// ===========================================================================

namespace {
/** Xattr key (in the system tag) holding container `c`'s id reservation. */
std::string IdsKey(clio::run::u32 c) { return "~ids." + std::to_string(c); }
/** Blob that held the id reservation before it moved to IdsKey (read only,
 *  so a reservation stored by an earlier build is still honoured). */
std::string IdsBlob(clio::run::u32 c) { return "ids." + std::to_string(c); }
/** Blob holding container `c`'s orphan list. */
std::string OrphansBlob(clio::run::u32 c) {
  return "orphans." + std::to_string(c);
}
/** Context for a clio-fs system record (non-volatile tier when possible). */
clio::cte::core::Context SysCtx(bool volatile_only) {
  clio::cte::core::Context ctx;
  ctx.op_flags_ |= clio::cte::core::Context::kMetaBlob;
  ctx.min_persistence_level_ = volatile_only ? 0 : 1;
  return ctx;
}
}  // namespace

void Runtime::LogInode(const InodeState &fi) {
  // The inode's record lives in CTE (FlushInodes stores and pushes it before
  // the change is acknowledged). Orphans are also listed in this container's
  // orphan record, so a restart can destroy them.
  MarkInodeDirtyLocked(fi);
  if (fi.orphan_) {
    std::lock_guard<std::mutex> g(purge_mu_);
    orphans_dirty_ = true;
  }
}

clio::run::TaskResume Runtime::EnsureIdReserve(clio::run::u32 margin,
                                               bool &ok) {
  CLIO_TASK_BODY_BEGIN
  ok = true;
  for (;;) {
    clio::run::u32 want = 0;
    bool go = false;
    {
      std::lock_guard<std::mutex> g(meta_mu_);
      if (minted_hi_ >= next_minor_ + margin) break;
      if (!reserving_) {
        reserving_ = go = true;
        want = next_minor_ + margin + 4096;
      }
    }
    if (!go) {
      CLIO_CO_AWAIT(clio::run::yield(20.0));
      continue;
    }
    // The reservation is a key of this container's system tag, on its own
    // CTE container: an in-memory update made durable by the CTE WAL, like
    // the inode records. It used to be a blob that had to land on a
    // non-volatile tier; with none configured every store failed, the
    // counter restarted at 1 after each restart, and new directories took
    // the ids of live ones -- each showing the other's contents.
    if (sys_tag_id_.IsNull()) {
      ok = false;
    } else {
      std::string rec;
      FsEnc(&rec).U32(want);
      auto p = cte_.AsyncTagSetXattr(sys_tag_id_, IdsKey(container_id_), rec,
                                     0, clio::run::PoolQuery::Local());
      CLIO_CO_AWAIT(p);
      ok = p->GetReturnCode() == 0;
      if (!ok) {
        HLOG(kError, "filesystem: storing the id reservation failed (rc {}); "
             "no inode is created until it is stored, since an unrecorded "
             "reservation lets ids repeat after a restart",
             p->GetReturnCode());
      }
    }
    {
      std::lock_guard<std::mutex> g(meta_mu_);
      reserving_ = false;
      if (ok) minted_hi_ = std::max(minted_hi_, want);
    }
    break;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::StoreOrphans() {
  CLIO_TASK_BODY_BEGIN
  {
    std::lock_guard<std::mutex> g(purge_mu_);
    if (!orphans_dirty_ || sys_tag_id_.IsNull()) CLIO_CO_RETURN;
    orphans_dirty_ = false;
  }
  std::string rec;
  {
    FsEnc e(&rec);
    std::lock_guard<std::mutex> g(meta_mu_);
    std::vector<clio::run::u64> ids;
    for (const auto &kv : inodes_) {
      if (kv.second->orphan_) ids.push_back(kv.first);
    }
    e.U32(static_cast<clio::run::u32>(ids.size()));
    for (clio::run::u64 id : ids) e.U64(id);
  }
  auto p = cte_.AsyncPutBlob(sys_tag_id_, OrphansBlob(container_id_), 0,
                             rec.size(), rec.data(), -1.0f,
                             SysCtx(inode_volatile_only_), 0u,
                             clio::run::PoolQuery::Dynamic());
  CLIO_CO_AWAIT(p);
  if (p->GetReturnCode() != 0) {
    std::lock_guard<std::mutex> g(purge_mu_);
    orphans_dirty_ = true;  // retried by the next drain tick
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::LoadSysRecords() {
  CLIO_TASK_BODY_BEGIN
  if (sys_tag_id_.IsNull()) CLIO_CO_RETURN;
  std::string rec;
  {
    auto x = cte_.AsyncTagGetXattr(sys_tag_id_, IdsKey(container_id_),
                                   clio::run::PoolQuery::Local());
    CLIO_CO_AWAIT(x);
    clio::run::u32 hi = 0;
    if (x->GetReturnCode() == 0 && x->found_) {
      const std::string v = x->value_.str();
      FsDec d(v.data(), v.size());
      if (d.U32(&hi)) {
        // Everything reserved before the restart may be in use: start past it.
        std::lock_guard<std::mutex> lk(meta_mu_);
        next_minor_ = std::max<clio::run::u32>(hi, 1);
        minted_hi_ = next_minor_;
      }
    }
  }
  {
    // A reservation an earlier build stored as a blob.
    auto sz = cte_.AsyncGetBlobSize(sys_tag_id_, IdsBlob(container_id_));
    CLIO_CO_AWAIT(sz);
    if (sz->GetReturnCode() == 0 && sz->size_ >= 4) {
      rec.assign(sz->size_, '\0');
      auto g = cte_.AsyncGetBlob(sys_tag_id_, IdsBlob(container_id_), 0,
                                 rec.size(), 0u, rec.data());
      CLIO_CO_AWAIT(g);
      clio::run::u32 hi = 0;
      FsDec d(rec.data(), rec.size());
      if (g->GetReturnCode() == 0 && d.U32(&hi)) {
        std::lock_guard<std::mutex> lk(meta_mu_);
        next_minor_ = std::max<clio::run::u32>(next_minor_,
                                               std::max<clio::run::u32>(hi, 1));
        minted_hi_ = next_minor_;
      }
    }
  }
  std::vector<clio::run::u64> orphans;
  {
    auto sz = cte_.AsyncGetBlobSize(sys_tag_id_, OrphansBlob(container_id_));
    CLIO_CO_AWAIT(sz);
    if (sz->GetReturnCode() == 0 && sz->size_ >= 4) {
      rec.assign(sz->size_, '\0');
      auto g = cte_.AsyncGetBlob(sys_tag_id_, OrphansBlob(container_id_), 0,
                                 rec.size(), 0u, rec.data());
      CLIO_CO_AWAIT(g);
      FsDec d(rec.data(), rec.size());
      clio::run::u32 n = 0;
      clio::run::u64 id = 0;
      if (g->GetReturnCode() == 0 && d.U32(&n)) {
        for (clio::run::u32 i = 0; i < n && d.U64(&id); ++i) {
          orphans.push_back(id);
        }
      }
    }
  }
  // Handles do not survive a restart: an inode unlinked while open can go.
  for (clio::run::u64 id : orphans) {
    CLIO_CO_AWAIT(EnsureInode(id));
    std::lock_guard<std::mutex> g(meta_mu_);
    auto it = inodes_.find(id);
    if (it == inodes_.end()) continue;
    std::shared_ptr<InodeState> fi = it->second;
    fi->open_count_ = 0;
    fi->orphan_ = false;
    // Recovered after a restart: its stream may not be restored yet, so its
    // size is not trusted and its purge reaches every container.
    DropInodeLocked(fi, /*size_trusted=*/false);
  }
  if (!orphans.empty()) {
    {
      std::lock_guard<std::mutex> g(purge_mu_);
      orphans_dirty_ = true;
    }
    CLIO_CO_AWAIT(StoreOrphans());
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

}  // namespace clio::cte::filesystem
