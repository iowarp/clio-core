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
 * The hash-sharded namespace: the state each filesystem container owns
 * (directories whose path hashes to it, inodes it minted), the ShardOp
 * primitives other containers invoke on that state, and its persistence.
 *
 * Every primitive is a short critical section with no suspension inside, so
 * the multi-step operations built from them (mkdir, rmdir, rename, link)
 * keep consistency with per-entry states instead of locks held across
 * network round trips: a PENDING entry is reserved but invisible, a LEAVING
 * entry is still visible but cannot be mutated by anyone else. Operations
 * that must wait on such an entry yield and retry; two-party operations
 * (rename, link) fail EBUSY instead and back off, which is what keeps two
 * crossing renames from deadlocking or building a directory cycle.
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
}

/** Decode a request. @return false on a malformed buffer */
bool DecReq(const std::string &s, Runtime::FsReq *r) {
  FsDec d(s.data(), s.size());
  return d.Str(&r->dir_) && d.Str(&r->leaf_) && d.Str(&r->str_) &&
         d.Str(&r->str2_) && d.U64(&r->id_) && d.U64(&r->a_) &&
         d.U64(&r->b_) && d.U32(&r->type_) && d.U32(&r->flags_) &&
         d.U32(&r->mode_) && d.U32(&r->uid_) && d.U32(&r->gid_);
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

clio::run::u32 Runtime::DirOwner(const std::string &dir) {
  return FsDirContainer(dir, NumContainers());
}

clio::run::u32 Runtime::InodeOwner(clio::run::u64 packed) {
  if (FsIdHasHome(packed)) return FsIdHome(packed);
  return static_cast<clio::run::u32>(FsMix64(packed)) % NumContainers();
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

clio::run::TaskResume Runtime::ExecShardOp(clio::run::u32 op, const FsReq &req,
                                           FsResp &resp) {
  CLIO_TASK_BODY_BEGIN
  int rc = 0;
  switch (op) {
    case kShardLookup: {
      std::lock_guard<std::mutex> g(ns_mu_);
      Dentry e;
      rc = LookupLocked(req.dir_, req.leaf_, &e);
      resp.id_ = FsPack(e.id_);
      resp.type_ = e.type_;
      break;
    }
    case kShardInsert:
      while ((rc = InsertEntry(req, resp)) == kFsRetry) {
        CLIO_CO_AWAIT(clio::run::yield(50));
      }
      break;
    case kShardRemove:
      while ((rc = RemoveEntry(req, resp)) == kFsRetry) {
        CLIO_CO_AWAIT(clio::run::yield(50));
      }
      break;
    case kShardDirCreate: rc = DirCreate(req); break;
    case kShardDirRetire: rc = DirRetire(req); break;
    case kShardDirAttr: rc = DirAttrOp(req, resp); break;
    case kShardDirExport: rc = DirExport(req, resp); break;
    case kShardDirImport: rc = DirImport(req); break;
    case kShardDirDrop: rc = DirDropOrUnmark(req, true); break;
    case kShardDirUnmark: rc = DirDropOrUnmark(req, false); break;
    case kShardInodeStat: {
      auto fi = FindInode(req.id_);
      if (fi == nullptr) { rc = ENOENT; break; }
      std::lock_guard<std::mutex> g(meta_mu_);
      InodeAttrLocked(*fi, &resp.attr_);
      resp.str_ = fi->symlink_;
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
  resp.rc_ = static_cast<clio::run::u32>(rc);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

// ===========================================================================
// Directory state
// ===========================================================================

void Runtime::TouchDirLocked(DirState &ds) {
  const clio::run::u64 now = NowNs();
  ds.mtime_ = now;
  ds.ctime_ = now;
}

void Runtime::DirAttrLocked(const DirState &ds, FsAttr *attr) {
  attr->id_ = FsPack(ds.id_);
  attr->type_ = kFsTypeDir;
  attr->size_ = 0;
  attr->nlink_ = 2;
  attr->mode_ = ds.mode_;
  attr->uid_ = ds.uid_;
  attr->gid_ = ds.gid_;
  attr->atime_ = ds.atime_;
  attr->mtime_ = ds.mtime_;
  attr->ctime_ = ds.ctime_;
}

namespace {
/**
 * The directory state for `dir` (ns_mu_ held), creating the root on first
 * use: "/" exists by definition, so its owner materializes it lazily instead
 * of depending on a bootstrap step that could run before the pool is sized.
 */
template <typename MapT, typename StateT>
StateT *FindDir(MapT &dirs, const std::string &dir,
                const std::function<void(const std::string &, StateT &)> &log) {
  auto it = dirs.find(dir);
  if (it != dirs.end()) return it->second.get();
  if (dir != "/") return nullptr;
  auto st = std::make_shared<StateT>();
  st->id_ = FsRootId();
  const clio::run::u64 now = NowNs();
  st->atime_ = st->mtime_ = st->ctime_ = now;
  dirs[dir] = st;
  log(dir, *st);
  return st.get();
}
}  // namespace

int Runtime::LookupLocked(const std::string &dir, const std::string &leaf,
                          Dentry *out) {
  DirState *ds = FindDir<decltype(dirs_), DirState>(
      dirs_, dir, [this](const std::string &p, DirState &s) { LogDirPut(p, s); });
  if (ds == nullptr || ds->retiring_) return ENOENT;
  auto it = ds->ents_.find(leaf);
  if (it == ds->ents_.end() || it->second.state_ == kEntPending) return ENOENT;
  *out = it->second;
  return 0;
}

int Runtime::InsertEntry(const FsReq &req, FsResp &resp) {
  const clio::run::u32 f = req.flags_;
  std::lock_guard<std::mutex> g(ns_mu_);
  DirState *ds = FindDir<decltype(dirs_), DirState>(
      dirs_, req.dir_, [this](const std::string &p, DirState &s) { LogDirPut(p, s); });
  if (ds == nullptr || ds->retiring_) return ENOENT;
  if (ds->moving_) return (f & kInsFailBusy) ? EBUSY : kFsRetry;
  auto it = ds->ents_.find(req.leaf_);
  if ((f & kInsCommit) != 0) {
    if (it == ds->ents_.end() || it->second.state_ != kEntPending ||
        FsPack(it->second.id_) != req.id_) {
      return ENOENT;
    }
    it->second.state_ = kEntLive;
    TouchDirLocked(*ds);
    LogEnt(true, req.dir_, req.leaf_, it->second, *ds);
    return 0;
  }
  if (it != ds->ents_.end()) {
    Dentry &e = it->second;
    if (e.state_ != kEntLive) return (f & kInsFailBusy) ? EBUSY : kFsRetry;
    resp.id_ = FsPack(e.id_);
    resp.type_ = e.type_;
    if ((f & kInsExcl) != 0) return EEXIST;
    if ((f & (kInsReplace | kInsReplaceDir)) == 0) return 0;  // open existing
    if (FsPack(e.id_) == req.id_) {  // rename onto another link of itself
      resp.old_id_ = req.id_;
      return 0;
    }
    if (e.type_ == kFsTypeDir && req.type_ != kFsTypeDir) return EISDIR;
    if (e.type_ != kFsTypeDir && req.type_ == kFsTypeDir) return ENOTDIR;
    if (e.type_ == kFsTypeDir && (f & kInsReplaceDir) == 0) return ENOTEMPTY;
    resp.old_id_ = FsPack(e.id_);
    resp.old_type_ = e.type_;
    e.id_ = FsUnpack(req.id_);
    e.type_ = req.type_;
    TouchDirLocked(*ds);
    LogEnt(true, req.dir_, req.leaf_, e, *ds);
    resp.id_ = req.id_;
    resp.type_ = req.type_;
    return 0;
  }
  Dentry e;
  e.type_ = req.type_;
  e.state_ = (f & kInsPending) ? kEntPending : kEntLive;
  if ((f & kInsNewInode) != 0) {
    clio::run::u64 want = req.id_;
    if (want != 0 && (InodeOwner(want) != container_id_ ||
                      FindInode(want) != nullptr)) {
      return EEXIST;  // a client-minted id that is not ours or is taken
    }
    e.id_ = want != 0 ? FsUnpack(want) : MintId();
    NewInode(e.id_, req.type_, req.mode_, req.str_,
             FsJoin(req.dir_, req.leaf_));
  } else {
    e.id_ = req.id_ != 0 ? FsUnpack(req.id_) : MintId();
  }
  ds->ents_[req.leaf_] = e;
  if (e.state_ == kEntLive) {
    TouchDirLocked(*ds);
    LogEnt(true, req.dir_, req.leaf_, e, *ds);
  }
  resp.id_ = FsPack(e.id_);
  resp.type_ = e.type_;
  resp.created_ = 1;
  return 0;
}

int Runtime::RemoveEntry(const FsReq &req, FsResp &resp) {
  const clio::run::u32 f = req.flags_;
  std::lock_guard<std::mutex> g(ns_mu_);
  auto dit = dirs_.find(req.dir_);
  if (dit == dirs_.end()) return ENOENT;
  DirState &ds = *dit->second;
  if (ds.moving_ && (f & kRmRestore) == 0) {
    return (f & kRmFailBusy) ? EBUSY : kFsRetry;
  }
  auto it = ds.ents_.find(req.leaf_);
  if (it == ds.ents_.end()) return ENOENT;
  Dentry &e = it->second;
  const bool mine = req.id_ != 0 && FsPack(e.id_) == req.id_;
  if ((f & kRmRestore) != 0) {
    if (mine && e.state_ == kEntLeaving) e.state_ = kEntLive;
    return 0;
  }
  if (req.id_ != 0 && !mine) return ENOENT;  // replaced underneath us
  if (e.state_ == kEntPending && mine) {    // abort a reservation
    ds.ents_.erase(it);
    return 0;
  }
  // A leaving entry may only be finished by the operation that marked it
  // (which names the id); everyone else waits for the outcome.
  if (e.state_ != kEntLive && !(e.state_ == kEntLeaving && mine &&
                                (f & kRmMarkLeaving) == 0)) {
    return (f & kRmFailBusy) ? EBUSY : kFsRetry;
  }
  if ((f & kRmNonDir) != 0 && e.type_ == kFsTypeDir) return EISDIR;
  if ((f & kRmDirOnly) != 0 && e.type_ != kFsTypeDir) return ENOTDIR;
  resp.old_id_ = FsPack(e.id_);
  resp.old_type_ = e.type_;
  if ((f & kRmMarkLeaving) != 0) {
    e.state_ = kEntLeaving;
    return 0;
  }
  Dentry gone = e;
  ds.ents_.erase(it);
  TouchDirLocked(ds);
  LogEnt(false, req.dir_, req.leaf_, gone, ds);
  return 0;
}

int Runtime::DirCreate(const FsReq &req) {
  std::lock_guard<std::mutex> g(ns_mu_);
  auto &slot = dirs_[req.dir_];
  if (slot != nullptr && !slot->ents_.empty()) {
    // A state with entries but no parent entry: left by a crash between the
    // two halves of an earlier rename or rmdir. The caller just proved the
    // name absent, so the new directory replaces it.
    HLOG(kWarning, "filesystem: mkdir {} replaces an orphaned listing ({} "
         "entries)", req.dir_, slot->ents_.size());
  }
  slot = std::make_shared<DirState>();
  slot->id_ = FsUnpack(req.id_);
  slot->mode_ = req.mode_;
  slot->uid_ = req.uid_;
  slot->gid_ = req.gid_;
  const clio::run::u64 now = NowNs();
  slot->atime_ = slot->mtime_ = slot->ctime_ = now;
  LogDirPut(req.dir_, *slot);
  return 0;
}

int Runtime::DirRetire(const FsReq &req) {
  std::lock_guard<std::mutex> g(ns_mu_);
  auto it = dirs_.find(req.dir_);
  if (it == dirs_.end()) return 0;  // already gone (crash between halves)
  DirState &ds = *it->second;
  if (ds.moving_) return EBUSY;
  if (!ds.ents_.empty()) return ENOTEMPTY;
  dirs_.erase(it);
  std::string p;
  FsEnc(&p).Str(req.dir_);
  log_.Append(FsLogRec::kDirDel, p);
  return 0;
}

int Runtime::DirAttrOp(const FsReq &req, FsResp &resp) {
  const clio::run::u32 f = req.flags_;
  std::lock_guard<std::mutex> g(ns_mu_);
  DirState *ds = FindDir<decltype(dirs_), DirState>(
      dirs_, req.dir_, [this](const std::string &p, DirState &s) { LogDirPut(p, s); });
  if (ds == nullptr) {
    if ((f & kAttrRepair) == 0 || req.id_ == 0) return ENOENT;
    // The parent's live entry says this directory exists but its listing is
    // gone (crash between rmdir's two halves): recreate it empty.
    auto st = std::make_shared<DirState>();
    st->id_ = FsUnpack(req.id_);
    const clio::run::u64 now = NowNs();
    st->atime_ = st->mtime_ = st->ctime_ = now;
    dirs_[req.dir_] = st;
    LogDirPut(req.dir_, *st);
    ds = st.get();
  }
  const clio::run::u64 now = NowNs();
  bool changed = false;
  if (f & kSetAtimeNow) { ds->atime_ = now; changed = true; }
  else if (f & kSetAtime) { ds->atime_ = req.a_; changed = true; }
  if (f & kSetMtimeNow) { ds->mtime_ = now; changed = true; }
  else if (f & kSetMtime) { ds->mtime_ = req.b_; changed = true; }
  if (f & kSetUid) { ds->uid_ = req.uid_; changed = true; }
  if (f & kSetGid) { ds->gid_ = req.gid_; changed = true; }
  if (f & kSetMode) { ds->mode_ = req.mode_ & 07777u; changed = true; }
  if (changed || (f & kSetCtimeOnly)) {
    ds->ctime_ = now;
    LogDirPut(req.dir_, *ds);
  }
  DirAttrLocked(*ds, &resp.attr_);
  return 0;
}

int Runtime::DirExport(const FsReq &req, FsResp &resp) {
  std::lock_guard<std::mutex> g(ns_mu_);
  auto it = dirs_.find(req.dir_);
  if (it == dirs_.end()) return ENOENT;
  DirState &ds = *it->second;
  if (ds.moving_ || ds.retiring_) return EBUSY;
  for (const auto &kv : ds.ents_) {
    if (kv.second.state_ != kEntLive) return EBUSY;  // an op is mid-flight
  }
  ds.moving_ = true;
  std::string blob;
  FsEnc e(&blob);
  e.U64(FsPack(ds.id_));
  e.U32(ds.mode_);
  e.U32(ds.uid_);
  e.U32(ds.gid_);
  e.U64(ds.atime_);
  e.U64(ds.mtime_);
  e.U64(ds.ctime_);
  e.U32(static_cast<clio::run::u32>(ds.ents_.size()));
  for (const auto &kv : ds.ents_) {
    e.Str(kv.first);
    e.U64(FsPack(kv.second.id_));
    e.U32(kv.second.type_);
  }
  resp.str_ = std::move(blob);
  DirAttrLocked(ds, &resp.attr_);
  return 0;
}

int Runtime::DirImport(const FsReq &req) {
  auto st = std::make_shared<DirState>();
  FsDec d(req.str_.data(), req.str_.size());
  clio::run::u64 id = 0;
  clio::run::u32 n = 0;
  if (!d.U64(&id) || !d.U32(&st->mode_) || !d.U32(&st->uid_) ||
      !d.U32(&st->gid_) || !d.U64(&st->atime_) || !d.U64(&st->mtime_) ||
      !d.U64(&st->ctime_) || !d.U32(&n)) {
    return EINVAL;
  }
  st->id_ = FsUnpack(id);
  for (clio::run::u32 i = 0; i < n; ++i) {
    std::string leaf;
    clio::run::u64 cid = 0;
    Dentry e;
    if (!d.Str(&leaf) || !d.U64(&cid) || !d.U32(&e.type_)) return EINVAL;
    e.id_ = FsUnpack(cid);
    st->ents_[leaf] = e;
  }
  std::lock_guard<std::mutex> g(ns_mu_);
  dirs_[req.dir_] = st;
  LogDirPut(req.dir_, *st);
  for (const auto &kv : st->ents_) {
    LogEnt(true, req.dir_, kv.first, kv.second, *st);
  }
  return 0;
}

int Runtime::DirDropOrUnmark(const FsReq &req, bool drop) {
  std::lock_guard<std::mutex> g(ns_mu_);
  auto it = dirs_.find(req.dir_);
  if (it == dirs_.end()) return 0;
  if (req.id_ != 0 && FsPack(it->second->id_) != req.id_) return 0;
  if (!drop) {
    it->second->moving_ = false;
    return 0;
  }
  dirs_.erase(it);
  std::string p;
  FsEnc(&p).Str(req.dir_);
  log_.Append(FsLogRec::kDirDel, p);
  return 0;
}

// ===========================================================================
// Inodes
// ===========================================================================

clio::cte::core::TagId Runtime::MintId() {
  std::lock_guard<std::mutex> g(meta_mu_);
  if (next_minor_ >= minted_hi_) {
    // Reserve ids in blocks and log the reservation BEFORE using any of
    // them, so a restart can never hand out an id that is still in use.
    minted_hi_ = next_minor_ + 4096;
    std::string p;
    FsEnc(&p).U32(minted_hi_);
    log_.Append(FsLogRec::kNextId, p);
  }
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
  fi->size_.store(type == kFsTypeSymlink ? symlink.size() : 0);
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
  attr->size_ = fi.size_.load();
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
  by_tag_.erase(packed);
  std::string p;
  FsEnc(&p).U64(packed);
  log_.Append(FsLogRec::kInodeDel, p);
  PurgeItem item;
  item.id_ = fi->tag_id_;
  item.data_ = fi->type_ == kFsTypeFile;
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
  clio::run::u64 old_size = std::max<clio::run::u64>(fi->size_.load(), req.b_);
  fi->size_.store(new_size);
  {
    std::lock_guard<std::mutex> g(meta_mu_);
    const clio::run::u64 now = NowNs();
    fi->mtime_ = fi->ctime_ = now;
    LogInode(*fi);
    if (!fi->path_.empty()) MirrorFile(fi->path_, *fi);
    InodeAttrLocked(*fi, &resp.attr_);
  }
  if (new_size >= old_size) CLIO_CO_RETURN;
  // Shrink: trim the boundary page and physically ZERO its surviving tail
  // (a later write past EOF re-extends over it; the old bytes must not
  // reappear inside what is now a hole), then drop whole pages beyond it.
  const clio::run::u64 boundary_page = new_size / kFsPageSize;
  const clio::run::u64 boundary_off = new_size % kFsPageSize;
  {
    auto tb = cte_.AsyncTruncateBlob(tag, std::to_string(boundary_page),
                                     boundary_off,
                                     clio::run::PoolQuery::Dynamic());
    CLIO_CO_AWAIT(tb);
  }
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
  const clio::run::u64 last_page = (old_size - 1) / kFsPageSize;
  for (clio::run::u64 pg = boundary_page + 1; pg <= last_page; ++pg) {
    auto d = cte_.AsyncDelBlob(tag, std::to_string(pg),
                               clio::run::PoolQuery::Dynamic());
    CLIO_CO_AWAIT(d);
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
  std::vector<PurgeItem> work;
  {
    std::lock_guard<std::mutex> g(purge_mu_);
    work.swap(purge_pending_);
  }
  for (const PurgeItem &item : work) {
    if (item.data_) {
      // Pages are hash-spread across every node, so the purge asks all of
      // them (skipping nodes already known dead). Off the unlink path.
      auto d = cte_.AsyncDelTag(item.id_, clio::run::PoolQuery::Broadcast(0.0f));
      CLIO_CO_AWAIT(d);
    }
    if (item.xattr_ && !xattr_tag_id_.IsNull()) {
      auto x = cte_.AsyncDelBlob(xattr_tag_id_, std::to_string(FsPack(item.id_)),
                                 clio::run::PoolQuery::Dynamic());
      CLIO_CO_AWAIT(x);
    }
  }
  // Keep the log proportional to the live state.
  constexpr clio::run::u64 kCompactBytes = 256ull << 20;
  if (log_.IsOpen() && log_.BytesSinceCompact() > kCompactBytes) {
    CompactLog();
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

// ===========================================================================
// Persistence
// ===========================================================================

std::string Runtime::EncDirPut(const std::string &path, const DirState &ds) {
  std::string p;
  FsEnc e(&p);
  e.Str(path);
  e.U64(FsPack(ds.id_));
  e.U32(ds.mode_);
  e.U32(ds.uid_);
  e.U32(ds.gid_);
  e.U64(ds.atime_);
  e.U64(ds.mtime_);
  e.U64(ds.ctime_);
  return p;
}

std::string Runtime::EncInode(const FileInfo &fi) {
  std::string p;
  FsEnc e(&p);
  e.U64(FsPack(fi.tag_id_));
  e.U32(fi.type_);
  e.U64(fi.size_.load());
  e.U32(fi.nlink_);
  e.U32(fi.mode_);
  e.U32(fi.uid_);
  e.U32(fi.gid_);
  e.U64(fi.atime_);
  e.U64(fi.mtime_);
  e.U64(fi.ctime_);
  e.U32((fi.orphan_ ? 1u : 0u) | (fi.has_xattr_ ? 2u : 0u));
  e.Str(fi.symlink_);
  e.Str(fi.path_);
  return p;
}

void Runtime::LogDirPut(const std::string &path, const DirState &ds) {
  log_.Append(FsLogRec::kDirPut, EncDirPut(path, ds));
}

void Runtime::LogEnt(bool put, const std::string &dir, const std::string &leaf,
                     const Dentry &e, const DirState &ds) {
  std::string p;
  FsEnc enc(&p);
  enc.Str(dir);
  enc.Str(leaf);
  if (put) {
    enc.U64(FsPack(e.id_));
    enc.U32(e.type_);
  }
  enc.U64(ds.mtime_);
  enc.U64(ds.ctime_);
  log_.Append(put ? FsLogRec::kEntPut : FsLogRec::kEntDel, p);
}

void Runtime::LogInode(const FileInfo &fi) {
  log_.Append(FsLogRec::kInodePut, EncInode(fi));
}

void Runtime::ApplyLogRecord(FsLogRec type, const std::string &payload) {
  FsDec d(payload.data(), payload.size());
  std::string dir, leaf;
  clio::run::u64 id = 0, mt = 0, ct = 0;
  switch (type) {
    case FsLogRec::kDirPut: {
      auto st = std::make_shared<DirState>();
      if (!d.Str(&dir) || !d.U64(&id) || !d.U32(&st->mode_) ||
          !d.U32(&st->uid_) || !d.U32(&st->gid_) || !d.U64(&st->atime_) ||
          !d.U64(&st->mtime_) || !d.U64(&st->ctime_)) {
        return;
      }
      st->id_ = FsUnpack(id);
      auto &slot = dirs_[dir];
      if (slot != nullptr && slot->id_ == st->id_) {
        st->ents_ = std::move(slot->ents_);  // an attribute update
      }
      slot = st;
      return;
    }
    case FsLogRec::kDirDel:
      if (d.Str(&dir)) dirs_.erase(dir);
      return;
    case FsLogRec::kEntPut:
    case FsLogRec::kEntDel: {
      Dentry e;
      const bool put = type == FsLogRec::kEntPut;
      if (!d.Str(&dir) || !d.Str(&leaf)) return;
      if (put && (!d.U64(&id) || !d.U32(&e.type_))) return;
      if (!d.U64(&mt) || !d.U64(&ct)) return;
      auto &slot = dirs_[dir];
      if (slot == nullptr) slot = std::make_shared<DirState>();
      if (put) {
        e.id_ = FsUnpack(id);
        slot->ents_[leaf] = e;
      } else {
        slot->ents_.erase(leaf);
      }
      slot->mtime_ = mt;
      slot->ctime_ = ct;
      return;
    }
    case FsLogRec::kInodePut: {
      auto fi = std::make_shared<FileInfo>();
      clio::run::u64 size = 0;
      clio::run::u32 flags = 0;
      if (!d.U64(&id) || !d.U32(&fi->type_) || !d.U64(&size) ||
          !d.U32(&fi->nlink_) || !d.U32(&fi->mode_) || !d.U32(&fi->uid_) ||
          !d.U32(&fi->gid_) || !d.U64(&fi->atime_) || !d.U64(&fi->mtime_) ||
          !d.U64(&fi->ctime_) || !d.U32(&flags) || !d.Str(&fi->symlink_) ||
          !d.Str(&fi->path_)) {
        return;
      }
      fi->tag_id_ = FsUnpack(id);
      fi->size_.store(size);
      fi->orphan_ = (flags & 1u) != 0;
      fi->has_xattr_ = (flags & 2u) != 0;
      by_tag_[id] = fi;
      return;
    }
    case FsLogRec::kInodeDel:
      if (d.U64(&id)) by_tag_.erase(id);
      return;
    case FsLogRec::kNextId: {
      clio::run::u32 hi = 0;
      if (d.U32(&hi)) minted_hi_ = std::max(minted_hi_, hi);
      return;
    }
    default:
      return;
  }
}

void Runtime::RecoverShard(const std::string &log_path, bool replay) {
  log_path_ = log_path;
  if (log_path.empty()) {
    HLOG(kWarning, "filesystem: no metadata_log_path -- this node's slice of "
         "the namespace is volatile");
    return;
  }
  if (!log_.Open(log_path)) {
    HLOG(kError, "filesystem: cannot open metadata log {}: {}", log_path,
         std::strerror(errno));
    return;
  }
  if (!replay) {
    // Fresh start: whatever an earlier run left here describes data that is
    // not being recovered. Start from an empty namespace.
    CompactLog();
    HLOG(kInfo, "filesystem: fresh namespace (metadata log {})", log_path);
    return;
  }
  size_t n = 0;
  {
    std::lock_guard<std::mutex> g1(ns_mu_);
    std::lock_guard<std::mutex> g2(meta_mu_);
    n = log_.Replay([this](FsLogRec t, const std::string &p) {
      ApplyLogRecord(t, p);
    });
    // Everything reserved before the crash may be in use: start past it.
    next_minor_ = std::max<clio::run::u32>(minted_hi_, 1);
    minted_hi_ = next_minor_;
    // Handles do not survive a restart, so an orphan (unlinked while open)
    // can be destroyed now.
    std::vector<std::shared_ptr<FileInfo>> orphans;
    for (auto &kv : by_tag_) {
      if (kv.second->orphan_ || kv.second->nlink_ == 0) {
        orphans.push_back(kv.second);
      }
    }
    for (auto &fi : orphans) {
      fi->open_count_ = 0;
      DropInodeLocked(fi);
    }
  }
  CompactLog();
  HLOG(kInfo, "filesystem: recovered {} log records from {} ({} dirs, {} "
       "inodes)", n, log_path, dirs_.size(), by_tag_.size());
}

void Runtime::CompactLog() {
  if (!log_.IsOpen()) return;
  std::lock_guard<std::mutex> g1(ns_mu_);
  std::lock_guard<std::mutex> g2(meta_mu_);
  std::vector<std::pair<FsLogRec, std::string>> recs;
  {
    std::string p;
    FsEnc(&p).U32(std::max(minted_hi_, next_minor_));
    recs.emplace_back(FsLogRec::kNextId, std::move(p));
  }
  for (const auto &kv : dirs_) {
    recs.emplace_back(FsLogRec::kDirPut, EncDirPut(kv.first, *kv.second));
    for (const auto &ent : kv.second->ents_) {
      if (ent.second.state_ == kEntPending) continue;
      std::string p;
      FsEnc e(&p);
      e.Str(kv.first);
      e.Str(ent.first);
      e.U64(FsPack(ent.second.id_));
      e.U32(ent.second.type_);
      e.U64(kv.second->mtime_);
      e.U64(kv.second->ctime_);
      recs.emplace_back(FsLogRec::kEntPut, std::move(p));
    }
  }
  for (const auto &kv : by_tag_) {
    recs.emplace_back(FsLogRec::kInodePut, EncInode(*kv.second));
  }
  if (!log_.Rewrite(recs)) {
    HLOG(kError, "filesystem: compacting metadata log {} failed", log_path_);
  }
}

}  // namespace clio::cte::filesystem
