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
 * records (id reservations, orphans). Directories live in fs_dirs.cc.
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
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <clio_ctp/util/config_parse.h>
#include <clio_runtime/pool_manager.h>
#include <clio_cte/core/blob_placement.h>
#include <clio_cte/filesystem/filesystem_runtime.h>

namespace clio::cte::filesystem {

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
}

/** Decode a request. @return false on a malformed buffer */
bool DecReq(const std::string &s, Runtime::FsReq *r) {
  FsDec d(s.data(), s.size());
  return d.Str(&r->dir_) && d.Str(&r->leaf_) && d.Str(&r->str_) &&
         d.Str(&r->str2_) && d.U64(&r->id_) && d.U64(&r->a_) &&
         d.U64(&r->b_) && d.U32(&r->type_) && d.U32(&r->flags_) &&
         d.U32(&r->mode_) && d.U32(&r->uid_) && d.U32(&r->gid_) &&
         d.U64(&r->dir_id_) && d.U32(&r->block_);
}

/** Encode attributes. */
void EncAttr(FsEnc &e, const Runtime::FsAttr &a) {
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
}

/** Decode attributes. */
bool DecAttr(FsDec &d, Runtime::FsAttr *a) {
  return d.U64(&a->id_) && d.U32(&a->type_) && d.U64(&a->size_) &&
         d.U32(&a->nlink_) && d.U32(&a->mode_) && d.U32(&a->uid_) &&
         d.U32(&a->gid_) && d.U64(&a->atime_) && d.U64(&a->mtime_) &&
         d.U64(&a->ctime_);
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
  EncAttr(e, r.attr_);
  e.Str(r.str_);
}

/** Decode a response. @return false on a malformed buffer */
bool DecResp(const std::string &s, Runtime::FsResp *r) {
  FsDec d(s.data(), s.size());
  return d.U32(&r->rc_) && d.U64(&r->id_) && d.U32(&r->type_) &&
         d.U64(&r->old_id_) && d.U32(&r->old_type_) && d.U32(&r->created_) &&
         d.U64(&r->handle_) && DecAttr(d, &r->attr_) && d.Str(&r->str_);
}

/** SetAttr flag bits (FsReq::flags_ of kShardInodeSetAttr / kShardDirAttr). */
enum : clio::run::u32 {
  kSetAtime = 1u, kSetMtime = 2u, kSetAtimeNow = 4u, kSetMtimeNow = 8u,
  kSetUid = 16u, kSetGid = 32u, kSetMode = 64u, kAttrRepair = 128u,
  kSetCtimeOnly = 256u,
  // A read: advance atime by the relatime rule, never ctime (see
  // InodeSetAttr). Set from the UtimensTask flag kUtimensAccess.
  kAccessTouch = 512u,
  kAccessStrict = 1024u,  // with kAccessTouch: strictatime, always move
};
}  // namespace

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

clio::run::u32 Runtime::InodeOwner(clio::run::u64 packed) {
  const clio::run::u32 home =
      FsIdHasHome(packed)
          ? FsIdHome(packed)
          : static_cast<clio::run::u32>(FsMix64(packed)) % NumContainers();
  // While the home's node is dead its successor serves the inode, loaded
  // from its CTE record (which replication keeps reachable) -- the same
  // rule the CTE uses for the dead node's blobs.
  return clio::cte::core::FailoverContainer(pool_id_, home);
}

clio::run::TaskResume Runtime::CallShard(clio::run::u32 target,
                                         clio::run::u32 op, const FsReq &req,
                                         FsResp &resp) {
  CLIO_TASK_BODY_BEGIN
  if (target == container_id_) {
    CLIO_CO_AWAIT(ExecShardOp(op, req, resp));
    CLIO_CO_RETURN;
  }
  std::string enc;
  EncReq(req, &enc);
  auto t = self_.AsyncShardOp(
      op, enc,
      clio::run::PoolQuery::DirectId(
          static_cast<clio::run::ContainerId>(target)));
  CLIO_CO_AWAIT(t);
  resp = FsResp();
  if (t->GetReturnCode() != 0 || !DecResp(t->resp_.str(), &resp)) {
    // The owner is unreachable (dead node, network timeout): the state it
    // owns is unavailable, which POSIX can only express as an I/O error.
    resp = FsResp();
    resp.rc_ = EIO;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::ShardOp(clio::run::shared_ptr<ShardOpTask> &task) {
  CLIO_TASK_BODY_BEGIN
  FsReq req;
  FsResp resp;
  if (task->op_ == kShardPurgeDrain) {
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
  switch (op) {
    case kShardInsert:
    case kShardRemove:
      CLIO_CO_AWAIT(EntryMutation(op, req, resp));
      rc = static_cast<int>(resp.rc_);
      break;
    case kShardDirCreate:
      CLIO_CO_AWAIT(DirCreate(req, resp));
      rc = static_cast<int>(resp.rc_);
      break;
    case kShardBlockSeal:
      CLIO_CO_AWAIT(SealBlock(req, resp));
      rc = static_cast<int>(resp.rc_);
      break;
    case kShardDirAttr:
      CLIO_CO_AWAIT(DirAttrOp(req, resp));
      rc = static_cast<int>(resp.rc_);
      break;
    case kShardBlockFetch:
      CLIO_CO_AWAIT(ServeBlockFetch(req, resp));
      rc = static_cast<int>(resp.rc_);
      break;
    case kShardBlockPush: rc = ApplyBlockPush(req); break;
    case kShardBlockInstall:
      CLIO_CO_AWAIT(InstallBlock(req, resp));
      rc = static_cast<int>(resp.rc_);
      break;
    case kShardBlockDrop:
      CLIO_CO_AWAIT(DropBlock(req, resp));
      rc = static_cast<int>(resp.rc_);
      break;
    case kShardInodePush: rc = ApplyInodePush(req); break;
    case kShardPurgeLocal:
      CLIO_CO_AWAIT(PurgeLocal(req));
      break;
    case kShardRepublish: {
      // A restarted node asks for this container's names (it reset its own).
      std::string batch;
      CLIO_CO_AWAIT(EncodeHomeNames(&batch));
      if (!batch.empty()) {
        auto u = cte_.AsyncUpdateTagNames(
            batch, clio::run::PoolQuery::DirectId(
                       static_cast<clio::run::ContainerId>(req.a_)));
        CLIO_CO_AWAIT(u);
        rc = static_cast<int>(u->GetReturnCode());
      }
      break;
    }
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
  if (op != kShardBlockPush && op != kShardInodePush &&
      op != kShardBlockFetch) {
    CLIO_CO_AWAIT(FlushInodes());
  }
  resp.rc_ = static_cast<clio::run::u32>(rc);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::ExecInodeOp(clio::run::u32 op, const FsReq &req,
                                           FsResp &resp, int &rc) {
  CLIO_TASK_BODY_BEGIN
  rc = 0;
  CLIO_CO_AWAIT(EnsureInode(req.id_));  // lazily loaded after a restart
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
      if (req.b_ != 0 && who != container_id_) fi->holders_[who] = reg_seq_++;
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

clio::cte::core::TagId Runtime::MintId() {
  std::lock_guard<std::mutex> g(meta_mu_);
  // Only ids covered by the durable reservation are handed out, so a
  // restart can never mint one that is still in use (EnsureIdReserve keeps
  // the reservation ahead; null asks the caller to wait for it).
  if (next_minor_ >= minted_hi_) return clio::cte::core::TagId::GetNull();
  return clio::cte::core::TagId(kFsIdFlag | (container_id_ & kFsHomeMask),
                                next_minor_++);
}

std::shared_ptr<Runtime::FileInfo> Runtime::NewInode(
    const clio::cte::core::TagId &id, clio::run::u32 type, clio::run::u32 mode,
    const std::string &symlink, const std::string &path) {
  auto fi = std::make_shared<FileInfo>();
  fi->tag_id_ = id;
  fi->path_ = path;
  fi->type_ = type;
  fi->mode_ = mode == 0xFFFFFFFFu ? mode : (mode & 07777u);
  fi->symlink_ = symlink;
  const clio::run::u64 now = NowNs();
  fi->atime_ = fi->mtime_ = fi->ctime_ = now;
  std::lock_guard<std::mutex> g(meta_mu_);
  by_tag_[FsPack(id)] = fi;
  LogInode(*fi);
  return fi;
}

std::shared_ptr<Runtime::FileInfo> Runtime::FindInode(clio::run::u64 packed) {
  std::lock_guard<std::mutex> g(meta_mu_);
  auto it = by_tag_.find(packed);
  return it == by_tag_.end() ? nullptr : it->second;
}

void Runtime::InodeAttrLocked(const FileInfo &fi, FsAttr *attr) {
  attr->id_ = FsPack(fi.tag_id_);
  attr->type_ = fi.type_;
  attr->size_ = FileSize(fi);
  attr->nlink_ = fi.nlink_;
  attr->mode_ = fi.mode_;
  attr->uid_ = fi.uid_;
  attr->gid_ = fi.gid_;
  attr->atime_ = fi.atime_;
  attr->mtime_ = fi.mtime_;
  attr->ctime_ = fi.ctime_;
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
    fi->path_ = req.dir_;  // the name it was opened by (mirror key)
    MirrorFile(fi->path_, *fi);
  }
  return 0;
}

void Runtime::DropInodeLocked(const std::shared_ptr<FileInfo> &fi) {
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
  by_tag_.erase(packed);
  PurgeItem item;
  item.id_ = fi->tag_id_;
  item.data_ = true;  // pages and the inode record (symlinks have one too)
  item.xattr_ = fi->has_xattr_;
  QueuePurge(item);
}

int Runtime::InodeNlink(const FsReq &req, FsResp &resp) {
  const auto delta = static_cast<clio::run::i64>(req.a_);
  std::lock_guard<std::mutex> g(meta_mu_);
  auto it = by_tag_.find(req.id_);
  if (it == by_tag_.end()) return ENOENT;
  std::shared_ptr<FileInfo> fi = it->second;
  const clio::run::i64 n = static_cast<clio::run::i64>(fi->nlink_) + delta;
  fi->nlink_ = n > 0 ? static_cast<clio::run::u32>(n) : 0u;
  fi->ctime_ = NowNs();
  // Keep the name the inode publishes itself under current: a size push
  // after a rename must not re-publish (resurrect) the old name.
  if (!req.str_.empty()) {
    if (delta > 0) {
      fi->path_ = req.str_;
    } else if (fi->path_ == req.str_) {
      fi->path_.clear();
    }
  }
  InodeAttrLocked(*fi, &resp.attr_);
  if (fi->nlink_ == 0) {
    DropInodeLocked(fi);
  } else {
    LogInode(*fi);
  }
  return 0;
}

int Runtime::InodeSetAttr(const FsReq &req, FsResp &resp) {
  const clio::run::u32 f = req.flags_;
  std::lock_guard<std::mutex> g(meta_mu_);
  auto it = by_tag_.find(req.id_);
  if (it == by_tag_.end()) return ENOENT;
  FileInfo &fi = *it->second;
  const clio::run::u64 now = NowNs();
  if (f & kAccessTouch) {
    // relatime (the Linux default): a read moves atime only when it is not
    // newer than the last change or is a day old. A read is not a change,
    // so ctime stays.
    constexpr clio::run::u64 kDayNs = 86400ull * 1000000000ull;
    if ((f & kAccessStrict) || fi.atime_ <= fi.mtime_ ||
        fi.atime_ <= fi.ctime_ || now - fi.atime_ >= kDayNs) {
      fi.atime_ = now;
      LogInode(fi);
      if (!fi.path_.empty()) MirrorFile(fi.path_, fi);
    }
    InodeAttrLocked(fi, &resp.attr_);
    return 0;
  }
  if (f & kSetAtimeNow) fi.atime_ = now;
  else if (f & kSetAtime) fi.atime_ = req.a_;
  if (f & kSetMtimeNow) fi.mtime_ = now;
  else if (f & kSetMtime) fi.mtime_ = req.b_;
  if (f & kSetUid) fi.uid_ = req.uid_;
  if (f & kSetGid) fi.gid_ = req.gid_;
  if (f & kSetMode) fi.mode_ = req.mode_ & 07777u;
  fi.ctime_ = now;
  LogInode(fi);
  if (!fi.path_.empty()) MirrorFile(fi.path_, fi);
  InodeAttrLocked(fi, &resp.attr_);
  return 0;
}

clio::run::TaskResume Runtime::InodeTruncate(const FsReq &req, FsResp &resp) {
  CLIO_TASK_BODY_BEGIN
  resp.rc_ = 0;
  std::shared_ptr<FileInfo> fi = FindInode(req.id_);
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
    if (!fi->path_.empty()) MirrorFile(fi->path_, *fi);
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
                                zbuf.shm_.template Cast<void>(), -1.0f,
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

clio::run::TaskResume Runtime::PurgeDrain() {
  CLIO_TASK_BODY_BEGIN
  // Names first: a create-then-unlink must reach each node as add, remove,
  // then the page purge.
  CLIO_CO_AWAIT(FlushNames());
  CLIO_CO_AWAIT(FlushInodes());  // anything a handler left dirty
  if (catchup_pending_) CLIO_CO_AWAIT(CatchUpNames());
  std::vector<PurgeItem> work;
  {
    std::lock_guard<std::mutex> g(purge_mu_);
    work.swap(purge_pending_);
  }
  // Pages are hash-spread over every node. Ship each node ONE request per
  // batch of dead ids (it deletes its own pages locally) instead of one
  // cluster-wide broadcast per file: 10k unlinks were 10k broadcasts that
  // saturated the network worker and stalled every other remote request.
  constexpr size_t kPurgeBatch = 512;
  for (size_t i = 0; i < work.size(); i += kPurgeBatch) {
    FsReq batch;
    FsEnc enc(&batch.str_);
    const size_t end = std::min(work.size(), i + kPurgeBatch);
    for (size_t j = i; j < end; ++j) {
      if (!work[j].data_) continue;
      enc.U64(FsPack(work[j].id_));
      // Forget the stream first: a deferred append still in flight is then
      // discarded instead of re-creating pages after the purge below.
      clio::run::u32 rc = 0;
      CLIO_CO_AWAIT(FileSizeOp(work[j].id_,
                               clio::cte::stream::StreamSizeOp::kDrop, 0,
                               nullptr, nullptr, &rc));
    }
    if (!batch.str_.empty()) {
      const clio::run::u32 n = NumContainers();
      for (clio::run::u32 c = 0; c < n; ++c) {
        FsResp ignored;  // an unreachable node keeps its pages (a leak)
        CLIO_CO_AWAIT(CallShard(c, kShardPurgeLocal, batch, ignored));
      }
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
  while (dec.U64(&packed)) {
    // Local: this container's accounting share and page blobs of the file
    // (a miss -- no pages here -- returns at once).
    auto d = cte_.AsyncDelTag(FsUnpack(packed), clio::run::PoolQuery::Local());
    CLIO_CO_AWAIT(d);
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
  clio::cte::core::EncodeTagNameOp(&tn_batch_, op, id, NowNs(), name, name2);
}

clio::run::TaskResume Runtime::FlushNames() {
  CLIO_TASK_BODY_BEGIN
  std::string batch;
  {
    std::lock_guard<std::mutex> g(tn_mu_);
    batch.swap(tn_batch_);
  }
  if (!batch.empty()) {
    // One broadcast per drain tick carries every change since the last one.
    // A node that is down misses it and resyncs when it restarts.
    auto u = cte_.AsyncUpdateTagNames(batch,
                                      clio::run::PoolQuery::Broadcast(0.0f));
    CLIO_CO_AWAIT(u);
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::CatchUpNames() {
  CLIO_TASK_BODY_BEGIN
  // Every peer re-sends the names it owns to this node. Peers still coming
  // up are retried on later ticks (about 1 s apart, up to ~2 minutes).
  if (++catchup_attempts_ % 50 != 1) CLIO_CO_RETURN;
  std::vector<clio::run::u32> peers(catchup_missing_.begin(),
                                    catchup_missing_.end());
  for (clio::run::u32 c : peers) {
    FsReq r;
    r.a_ = container_id_;
    FsResp resp;
    CLIO_CO_AWAIT(CallShard(c, kShardRepublish, r, resp));
    if (resp.rc_ == 0) catchup_missing_.erase(c);
  }
  if (catchup_missing_.empty() || catchup_attempts_ > 50 * 120) {
    if (!catchup_missing_.empty()) {
      HLOG(kWarning, "filesystem: {} peers never re-sent their names; CTE "
           "search on this node misses their part of the namespace",
           catchup_missing_.size());
    }
    catchup_pending_ = false;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

// ===========================================================================
// System records: id reservations and orphans (CTE blobs in sys_tag_id_)
// ===========================================================================

namespace {
/** Blob holding container `c`'s id reservation. */
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

void Runtime::LogInode(const FileInfo &fi) {
  // The inode's record lives in CTE (FlushInodes stores and pushes it before
  // the change is acknowledged). Orphans are also listed in this container's
  // orphan record, so a restart can destroy them.
  MarkInodeDirtyLocked(fi);
  if (fi.orphan_) {
    std::lock_guard<std::mutex> g(purge_mu_);
    orphans_dirty_ = true;
  }
}

clio::run::TaskResume Runtime::EnsureIdReserve(clio::run::u32 margin) {
  CLIO_TASK_BODY_BEGIN
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
    bool ok = true;
    if (!sys_tag_id_.IsNull()) {
      std::string rec;
      FsEnc(&rec).U32(want);
      auto p = cte_.AsyncPutBlob(sys_tag_id_, IdsBlob(container_id_), 0,
                                 rec.size(), rec.data(), -1.0f,
                                 SysCtx(inode_volatile_only_), 0u,
                                 clio::run::PoolQuery::Dynamic());
      CLIO_CO_AWAIT(p);
      ok = p->GetReturnCode() == 0;
      if (!ok) {
        HLOG(kError, "filesystem: storing the id reservation failed (rc {}); "
             "ids minted now could repeat after a restart",
             p->GetReturnCode());
      }
    }
    {
      std::lock_guard<std::mutex> g(meta_mu_);
      reserving_ = false;
      // Unavailability must not stop creates: advance anyway (logged above).
      minted_hi_ = std::max(minted_hi_, want);
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
    for (const auto &kv : by_tag_) {
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
        // Everything reserved before the restart may be in use: start past it.
        std::lock_guard<std::mutex> lk(meta_mu_);
        next_minor_ = std::max<clio::run::u32>(hi, 1);
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
    auto it = by_tag_.find(id);
    if (it == by_tag_.end()) continue;
    std::shared_ptr<FileInfo> fi = it->second;
    fi->open_count_ = 0;
    fi->orphan_ = false;
    DropInodeLocked(fi);
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
