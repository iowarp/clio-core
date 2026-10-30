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
 * Filesystem runtime: inode records in CTE. Each inode's persistent
 * attributes are one key (kInodeKey) of the xattr section of the file's own
 * tag, held by the CTE container on the inode's home node: the home writes
 * it before it acknowledges a change and reads it back after a restart.
 * Other containers get attributes from the home (stat, pushes), never from
 * the record.
 */

#include <algorithm>
#include <clio_cte/core/blob_placement.h>
#include <clio_cte/filesystem/filesystem_runtime.h>

namespace clio::cte::filesystem {

namespace {
/** Xattr key of an inode's record in the file's tag section. */
constexpr const char *kInodeKey = "~i";
/** Format tag of inode records written before directories had records
 *  ("CFI1"): recognized only to say why such a volume is refused. */
constexpr clio::run::u32 kOldInodeRecMagic = 0x31494643u;
/** Wait between checks while another task stores the same inode (us). */
constexpr double kStoreBusyPollUs = 20.0;
}  // namespace

clio::run::PoolQuery Runtime::InodeRecQuery(clio::run::u64 packed) {
  const clio::run::u32 owner = InodeOwner(packed);
  return owner == container_id_
             ? clio::run::PoolQuery::Local()
             : clio::run::PoolQuery::DirectId(
                   static_cast<clio::run::ContainerId>(owner));
}

clio::run::TaskResume Runtime::CreateInodeRecord(
    Inode ino, const std::string &mirror_path, int &rc) {
  CLIO_TASK_BODY_BEGIN
  rc = 0;
  auto fi = std::make_shared<InodeState>();
  static_cast<Inode &>(*fi) = ino;
  fi->mirror_path_ = ino.type_ == kFsTypeDir ? std::string() : mirror_path;
  if (fi->mode_ != 0xFFFFFFFFu) fi->mode_ &= 07777u;
  fi->nlink_ = 1;
  const clio::run::u64 now = clio::cte::core::GetWallTimeNs();
  fi->atime_ = fi->mtime_ = fi->ctime_ = now;
  const std::string rec =
      EncodeInodeRecord(*fi, 0, container_id_, /*fresh=*/true);
  const clio::run::u64 packed = FsPack(fi->tag_id_);
  CLIO_CO_AWAIT(StoreInodeRec(packed, rec, rc));
  if (rc != 0) {
    rc = clio::cte::core::PutRcIsNoSpace(static_cast<clio::run::u32>(rc))
             ? ENOSPC
             : EIO;
    CLIO_CO_RETURN;
  }
  if (InodeOwner(packed) == container_id_) {
    // Homed here: resident at once. The record is stored, nothing is dirty.
    std::lock_guard<std::mutex> g(meta_mu_);
    if (inodes_.count(packed) == 0) inodes_[packed] = fi;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::DiscardInodeRecord(
    const clio::cte::core::TagId &id) {
  CLIO_TASK_BODY_BEGIN
  const clio::run::u64 packed = FsPack(id);
  {
    std::lock_guard<std::mutex> g(meta_mu_);
    inodes_.erase(packed);
    inode_dirty_.erase(packed);
    inode_dirty_seq_.erase(packed);
  }
  auto d = cte_.AsyncTagDelXattr(id, kInodeKey, 0, InodeRecQuery(packed));
  CLIO_CO_AWAIT(d);  // best effort: a leftover record names nothing
  if (NumContainers() > 1) {
    // And its mirror on the home's successor (StoreInodeRec).
    const clio::run::u32 succ = (InodeHome(packed) + 1) % NumContainers();
    if (succ != InodeOwner(packed)) {
      auto dm = cte_.AsyncTagDelXattr(
          id, kInodeKey, 0,
          clio::run::PoolQuery::DirectId(
              static_cast<clio::run::ContainerId>(succ)));
      CLIO_CO_AWAIT(dm);
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

void Runtime::MarkInodeDirtyLocked(const InodeState &fi) {
  const clio::run::u64 packed = FsPack(fi.tag_id_);
  inode_dirty_.insert(packed);
  inode_dirty_seq_[packed] = ++dirty_seq_;
}

bool Runtime::StoreBackingOff(clio::run::u64 packed, clio::run::u64 since) {
  auto fit = inode_store_failed_ms_.find(packed);
  if (fit == inode_store_failed_ms_.end()) return false;
  auto sit = inode_dirty_seq_.find(packed);
  if (sit != inode_dirty_seq_.end() && sit->second > since) return false;
  return SteadyMs() < fit->second + kInodeStoreRetryMs;
}

bool Runtime::DirtiedAfterLocked(clio::run::u64 packed, clio::run::u64 since) {
  auto sit = inode_dirty_seq_.find(packed);
  return sit != inode_dirty_seq_.end() && sit->second > since;
}

bool Runtime::IsDying(clio::run::u64 packed) {
  std::lock_guard<std::mutex> g(meta_mu_);
  return dying_.count(packed) != 0;
}

clio::run::u64 Runtime::DirtySeq() {
  std::lock_guard<std::mutex> g(meta_mu_);
  return dirty_seq_;
}

namespace {
/** Encode the attributes an inode push carries (attrs + symlink target). */
std::string EncInodePush(const Runtime::FsAttr &a, const std::string &link) {
  std::string out;
  FsEnc e(&out);
  Runtime::EncodeAttr(e, a);
  e.Str(link);
  return out;
}

/** Decode what EncInodePush wrote. @return false if malformed */
bool DecInodePush(const std::string &s, Runtime::FsAttr *a, std::string *link) {
  FsDec d(s.data(), s.size());
  return Runtime::DecodeAttr(d, a) && d.Str(link);
}

/** One inode record to store and push. */
struct InodeWork {
  clio::run::u64 packed = 0;
  std::string rec;                      ///< the CTE record
  std::string push;                     ///< EncInodePush of the same state
  std::map<clio::run::u32, clio::run::u64> holders;  ///< caching it (+ reg)
  std::map<clio::run::u32, clio::run::u64> leases;   ///< their lease ends
  bool resync = false;  ///< holders unknown: pushed to every container
  clio::run::u64 seq = 0;  ///< dirty sequence number of the stored change
};
}  // namespace

/** CLIO_TRACE_INODE=1: log every inode record stored and loaded (size,
 *  writer, container), to find where an fsynced file's size is lost. */
static bool TraceInode() {
  static const bool on = [] {
    const char *e = std::getenv("CLIO_TRACE_INODE");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}

static void TraceInodeRec(const char *what, clio::run::u64 packed,
                          const std::string &rec, clio::run::u32 at) {
  if (!TraceInode()) return;
  Inode ino;
  clio::run::u64 size = 0;
  clio::run::u32 writer = 0;
  const bool ok = DecodeInodeRecord(rec, &ino, &size, &writer, nullptr);
  HLOG(kWarning, "[INODE] {} {} on container {}: size {} writer {} "
       "mtime {} ctime {}{}", what, packed, at, size, writer, ino.mtime_,
       ino.ctime_, ok ? "" : " (undecodable)");
}

clio::run::TaskResume Runtime::StoreInodeRec(clio::run::u64 packed,
                                             const std::string &rec,
                                             int &rc) {
  CLIO_TASK_BODY_BEGIN
  TraceInodeRec("store", packed, rec, container_id_);
  {
    bool ready = false;
    CLIO_CO_AWAIT(AwaitCteReady(ready));
    if (!ready) {
      rc = EIO;
      CLIO_CO_RETURN;
    }
  }
  // The owner's own CTE container holds the record: one in-memory update and
  // a WAL append there, no blob placement or device write. Only the owner
  // reads it (EnsureInode, Local), so a record stored by another container --
  // a create's name inserter is usually not the inode's home -- is sent
  // there; stored Local it was invisible to the home (a dead home's records
  // are on its own node, #1263).
  auto p = cte_.AsyncTagSetXattr(FsUnpack(packed), kInodeKey, rec, 0,
                                 InodeRecQuery(packed));
  CLIO_CO_AWAIT(p);
  rc = static_cast<int>(p->GetReturnCode());
  if (rc == 0 && NumContainers() > 1) {
    // Mirror the record on the home's first successor: that is the node
    // failover hands the inode to while the home is down, and it can only
    // serve (stat, truncate, fsync an open file) what it has a record of.
    // Tag xattrs are not blobs, so the replication chimod never copies them.
    // Skipped when the successor is where it was just stored (it is the
    // stand-in now). Best effort: a missed mirror costs availability during
    // the next outage, never correctness (EnsureInode takes the newer).
    const clio::run::u32 n = NumContainers();
    const clio::run::u32 succ = (InodeHome(packed) + 1) % n;
    if (succ != InodeOwner(packed) &&
        clio::cte::core::ContainerNodeAlive(pool_id_, succ)) {
      auto m = cte_.AsyncTagSetXattr(
          FsUnpack(packed), kInodeKey, rec, 0,
          clio::run::PoolQuery::DirectId(
              static_cast<clio::run::ContainerId>(succ)));
      CLIO_CO_AWAIT(m);
    }
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::FlushInodes(int *err, clio::run::u64 since) {
  CLIO_TASK_BODY_BEGIN
  if (err != nullptr) *err = 0;
  std::unordered_set<clio::run::u64> failed;  // tried this call; stay dirty
  // Store what is dirty now, then push it to every container caching the
  // inode. An inode another task is storing stays dirty and is picked up
  // after it finishes: that store may predate this task's change, which must
  // be durable and pushed before this task replies.
  for (;;) {
    std::vector<InodeWork> work;
    size_t busy = 0;
    {
      std::lock_guard<std::mutex> g(meta_mu_);
      for (auto it = inode_dirty_.begin(); it != inode_dirty_.end();) {
        const clio::run::u64 packed = *it;
        if (failed.count(packed) != 0) {
          ++it;  // a later flush retries it
          continue;
        }
        if (since != 0 && !DirtiedAfterLocked(packed, since)) {
          // Another operation's change: that operation stores it (or the
          // periodic drain does). Storing or waiting on it here made every
          // Close on this container queue behind one slow store (#1149).
          ++it;
          continue;
        }
        if (inode_storing_.count(packed) != 0) {
          ++busy;
          ++it;
          continue;
        }
        if (StoreBackingOff(packed, since)) {
          ++it;  // failed moments ago and not this caller's change
          continue;
        }
        auto fit = inodes_.find(packed);
        if (fit != inodes_.end()) {
          const InodeState &fi = *fit->second;
          inode_storing_.insert(packed);
          InodeWork w;
          w.packed = packed;
          auto sit = inode_dirty_seq_.find(packed);
          w.seq = sit == inode_dirty_seq_.end() ? 0 : sit->second;
          w.rec = EncodeInodeRecord(fi, FileSize(fi), container_id_, false);
          FsAttr a;
          InodeAttrLocked(fi, &a);
          w.push = EncInodePush(a, fi.symlink_);
          w.holders = fi.holders_;
          w.leases = fi.holder_lease_ms_;
          if (fi.holders_unknown_) {
            fit->second->holders_unknown_ = false;
            w.resync = true;
            // Caches from an earlier incarnation expire by this horizon.
            const clio::run::u64 horizon =
                started_ms_ + kCacheLeaseMs + kCacheLeaseSlackMs;
            for (clio::run::u32 c = 0; c < NumContainers(); ++c) {
              if (c == container_id_) continue;
              w.holders.emplace(c, 0);
              w.leases[c] = std::max(w.leases[c], horizon);
            }
          }
          work.push_back(std::move(w));
        }
        if (fit == inodes_.end()) {
          if (TraceInode()) {
            // Suspected loss path: a dirty inode no longer in memory has its
            // unstored change discarded while the flush reports success.
            HLOG(kWarning, "[INODE] DROP dirty {} on container {}: not in "
                 "memory, its pending record is discarded", packed,
                 container_id_);
          }
          inode_dirty_seq_.erase(packed);
          inode_store_failed_ms_.erase(packed);
        }
        it = inode_dirty_.erase(it);  // dropped inodes need no record
      }
    }
    for (const InodeWork &w : work) {
      int src = 0;
      const clio::run::u64 t_store = SteadyMs();
      CLIO_CO_AWAIT(StoreInodeRec(w.packed, w.rec, src));
      const clio::run::u64 t_push = SteadyMs();
      std::vector<clio::run::u32> gone;
      if (!w.holders.empty()) {
        FsReq r;
        r.id_ = w.packed;
        r.str_ = w.push;
        std::vector<clio::run::u32> hs;
        std::vector<clio::run::u64> ls;
        for (const auto &hv : w.holders) {
          hs.push_back(hv.first);
          auto lt = w.leases.find(hv.first);
          ls.push_back(lt == w.leases.end() ? 0 : lt->second);
        }
        CLIO_CO_AWAIT(PushToHolders(kShardInodePush, r, hs, ls, &gone));
      }
      {
        const clio::run::u64 t_done = SteadyMs();
        if (t_done - t_store >= kSlowCloseMs) {
          HLOG(kWarning, "filesystem: inode {} record took {} ms to store "
               "(rc {}) and {} ms to push to {} holder(s)", w.packed,
               t_push - t_store, src, t_done - t_push, w.holders.size());
        }
      }
      std::lock_guard<std::mutex> g(meta_mu_);
      inode_storing_.erase(w.packed);
      auto fit = inodes_.find(w.packed);
      if (fit != inodes_.end()) {
        if (w.resync) {
          // Everyone who took the push caches it: register them, with the
          // lease the resync assumed (see CommitBlock).
          for (const auto &hv : w.holders) {
            if (std::find(gone.begin(), gone.end(), hv.first) != gone.end()) {
              continue;
            }
            if (fit->second->holders_.count(hv.first) == 0) {
              fit->second->holders_[hv.first] = reg_seq_++;
            }
            auto lt = w.leases.find(hv.first);
            if (lt != w.leases.end()) {
              clio::run::u64 &lease = fit->second->holder_lease_ms_[hv.first];
              lease = std::max(lease, lt->second);
            }
          }
        }
        DropGoneHolders(gone, w.holders, &fit->second->holders_,
                        &fit->second->holder_lease_ms_);
      }
      if (src == 0) {
        inode_store_failed_ms_.erase(w.packed);
        auto sit = inode_dirty_seq_.find(w.packed);
        if (inode_dirty_.count(w.packed) == 0 &&
            sit != inode_dirty_seq_.end() && sit->second == w.seq) {
          inode_dirty_seq_.erase(sit);
        }
      } else {
        HLOG(kWarning, "filesystem: storing inode {} failed (rc {}); will "
             "retry", w.packed, src);
        if (fit != inodes_.end()) inode_dirty_.insert(w.packed);
        inode_store_failed_ms_[w.packed] = SteadyMs();
        failed.insert(w.packed);
        if (err != nullptr && *err == 0 && w.seq > since) {
          *err = clio::cte::core::PutRcIsNoSpace(
                     static_cast<clio::run::u32>(src))
                     ? ENOSPC
                     : EIO;
        }
      }
    }
    if (busy == 0 && work.empty()) break;
    if (busy == 0) continue;  // anything dirtied meanwhile goes too
    CLIO_CO_AWAIT(clio::run::yield(kStoreBusyPollUs));
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::StatInode(clio::run::u64 packed,
                                         FsResp &resp) {
  CLIO_TASK_BODY_BEGIN
  resp = FsResp();
  const clio::run::u32 home = InodeOwner(packed);
  if (home == container_id_) {
    int load_err = 0;
    CLIO_CO_AWAIT(EnsureInode(packed, &load_err));
    std::shared_ptr<InodeState> fi = FindInode(packed);
    if (fi == nullptr) {
      // A record that could not be read is not a missing inode (#1166).
      resp.rc_ = load_err != 0 ? static_cast<clio::run::u32>(load_err) : ENOENT;
      CLIO_CO_RETURN;
    }
    std::lock_guard<std::mutex> g(meta_mu_);
    InodeAttrLocked(*fi, &resp.attr_);
    resp.str_ = fi->symlink_;
    CLIO_CO_RETURN;
  }
  bool fetch = false, cached = false;
  {
    std::lock_guard<std::mutex> g(icache_mu_);
    auto it = icache_.find(packed);
    // Past its lease the copy may have missed a push (a partition): ask.
    if (it != icache_.end() && it->second.home_ == home &&
        SteadyMs() < it->second.lease_until_ms_) {
      resp.attr_ = it->second.attr_;
      resp.str_ = it->second.symlink_;
      cached = true;
    } else if (iloading_.count(packed) == 0) {
      InodeCacheEnt none;
      none.home_ = ~0u;
      iloading_[packed] = none;
      fetch = true;
    }
  }
  if (cached) {
    // The size is not part of the pushed attributes: writes and truncates
    // change it at the file's stream, not through the inode. Ask the stream
    // (on the same home) so a stat is never behind an acknowledged write.
    if (resp.attr_.type_ == kFsTypeFile) {
      auto f = stream_.AsyncSizeOp(FsUnpack(packed), home,
                                   clio::cte::stream::StreamSizeOp::kGet);
      CLIO_CO_AWAIT(f);
      if (f->GetReturnCode() == 0) resp.attr_.size_ = f->new_size_;
    }
    CLIO_CO_RETURN;
  }
  FsReq r;
  r.id_ = packed;
  r.a_ = container_id_;
  r.b_ = fetch ? 1u : 0u;  // register for pushes only when we will cache it
  const clio::run::u64 asked_ms = SteadyMs();
  CLIO_CO_AWAIT(CallShard(home, kShardInodeStat, r, resp));
  if (!fetch) CLIO_CO_RETURN;
  std::lock_guard<std::mutex> g(icache_mu_);
  auto lit = iloading_.find(packed);
  if (resp.rc_ == 0) {
    InodeCacheEnt ent;
    ent.attr_ = resp.attr_;
    ent.symlink_ = resp.str_;
    // A push that raced the fetch is newer than what the fetch returned.
    if (lit != iloading_.end() && lit->second.home_ != ~0u) ent = lit->second;
    ent.home_ = home;
    ent.lease_until_ms_ = asked_ms + kCacheLeaseMs;
    icache_[packed] = ent;
  }
  if (lit != iloading_.end()) iloading_.erase(lit);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

void Runtime::RefreshCachedInode(clio::run::u64 packed, const FsAttr &attr) {
  std::lock_guard<std::mutex> g(icache_mu_);
  auto it = icache_.find(packed);
  if (it != icache_.end() && attr.ctime_ >= it->second.attr_.ctime_) {
    // Keep the type-independent fields the reply carries; the size is
    // always asked of the stream (see StatInode).
    it->second.attr_ = attr;
  }
  auto lit = iloading_.find(packed);
  if (lit != iloading_.end() &&
      (lit->second.home_ == ~0u || attr.ctime_ >= lit->second.attr_.ctime_)) {
    lit->second.attr_ = attr;
    lit->second.home_ = 0;  // "newer than the fetch": see StatInode
  }
}

int Runtime::ApplyInodePush(const FsReq &req) {
  InodeCacheEnt ent;
  if (!DecInodePush(req.str_, &ent.attr_, &ent.symlink_)) return EINVAL;
  std::lock_guard<std::mutex> g(icache_mu_);
  auto lit = iloading_.find(req.id_);
  if (lit != iloading_.end()) {
    if (lit->second.home_ != ~0u &&
        ent.attr_.ctime_ < lit->second.attr_.ctime_) {
      return 0;  // older than what already arrived
    }
    ent.home_ = 0;  // any value but ~0u: "a push arrived"
    lit->second = ent;
    return 0;
  }
  auto it = icache_.find(req.id_);
  if (it == icache_.end()) return ENOENT;
  // Pushes are not ordered with each other or with this node's own setattr
  // replies (RefreshCachedInode): every change bumps ctime on the home, so
  // an older push must not roll a newer copy back -- a chmod read back as
  // the mode before it (#1127).
  if (ent.attr_.ctime_ < it->second.attr_.ctime_) return 0;
  ent.home_ = it->second.home_;
  ent.lease_until_ms_ = it->second.lease_until_ms_;  // a push is no renewal
  it->second = ent;
  return 0;
}

bool Runtime::RecordNewer(const std::string &a, const std::string &b) {
  Inode ia, ib;
  clio::run::u64 sa = 0, sb = 0;
  clio::run::u32 wa = 0, wb = 0;
  if (!DecodeInodeRecord(a, &ia, &sa, &wa, nullptr)) return false;
  if (!DecodeInodeRecord(b, &ib, &sb, &wb, nullptr)) return true;
  const clio::run::u64 ta = std::max(ia.ctime_, ia.mtime_);
  const clio::run::u64 tb = std::max(ib.ctime_, ib.mtime_);
  return ta > tb;
}

clio::run::TaskResume Runtime::FetchStandInRecord(clio::run::u64 packed,
                                                  std::string *rec,
                                                  clio::run::u32 *from) {
  CLIO_TASK_BODY_BEGIN
  rec->clear();
  const clio::run::u32 n = NumContainers();
  // The stand-in is the first live container after the home (the CTE's
  // FailoverContainer rule). One failure at a time is the guarantee, so the
  // first live successor is the one that served it.
  for (clio::run::u32 i = 1; i < n; ++i) {
    const clio::run::u32 c = (container_id_ + i) % n;
    if (!clio::cte::core::ContainerNodeAlive(pool_id_, c)) continue;
    auto g = cte_.AsyncTagGetXattr(
        FsUnpack(packed), kInodeKey,
        clio::run::PoolQuery::DirectId(static_cast<clio::run::ContainerId>(c)));
    CLIO_CO_AWAIT(g);
    if (g->GetReturnCode() == 0 && g->found_) {
      *rec = g->value_.str();
      *from = c;
    }
    break;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::EnsureInode(clio::run::u64 packed, int *err) {
  CLIO_TASK_BODY_BEGIN
  if (err != nullptr) *err = 0;
  if (packed == 0 || InodeOwner(packed) != container_id_ ||
      FindInode(packed) != nullptr || IsDying(packed)) {
    CLIO_CO_RETURN;
  }
  {
    bool ready = false;
    CLIO_CO_AWAIT(AwaitCteReady(ready));
    if (!ready) CLIO_CO_RETURN;
  }
  const clio::cte::core::TagId tag = FsUnpack(packed);
  auto g = cte_.AsyncTagGetXattr(tag, kInodeKey,
                                 clio::run::PoolQuery::Local());
  CLIO_CO_AWAIT(g);
  if (g->GetReturnCode() != 0) {
    if (err != nullptr) *err = EIO;
    CLIO_CO_RETURN;
  }
  std::string rec = g->found_ ? g->value_.str() : std::string();
  bool found = g->found_;
  if (InodeHome(packed) == container_id_ && NumContainers() > 1) {
    // While this node was down (or cut off) its successor served the inode
    // and stored every record it wrote -- creates included -- in ITS core:
    // tag xattrs are not blobs, so replication never hands them back. Left
    // there, a file created during the outage vanishes once this home is
    // back, and a change made there is undone. Take the stand-in's record
    // when it is newer and keep it here.
    std::string theirs;
    clio::run::u32 from = 0;
    CLIO_CO_AWAIT(FetchStandInRecord(packed, &theirs, &from));
    if (!theirs.empty() && (!found || RecordNewer(theirs, rec))) {
      int src = 0;
      CLIO_CO_AWAIT(StoreInodeRec(packed, theirs, src));
      if (src != 0) {
        // Not kept here: serve nothing rather than the older record.
        if (err != nullptr) *err = EIO;
        CLIO_CO_RETURN;
      }
      HLOG(kWarning, "filesystem: inode {} took the record container {} "
           "stored while it served this home", packed, from);
      rec = theirs;
      found = true;
    }
    // The successor's copy stays: it is the mirror the next outage needs.
  }
  if (!found) {
    if (packed == FsPack(FsRootId()) && InodeHome(packed) == container_id_) {
      // "/" exists by definition: its home creates its record on first use,
      // and with it the volume's flags.
      int mrc = 0;
      CLIO_CO_AWAIT(MaterializeRoot(mrc));
      if (mrc != 0 && err != nullptr) *err = mrc;
      CLIO_CO_RETURN;
    }
    // Serving a dead home's inode: its record is in that node's CTE, not
    // here (#1263). Not "no such file", which would let a create replace it.
    if (NumContainers() > 1 && InodeHome(packed) != container_id_ &&
        err != nullptr) {
      *err = EIO;
    }
    CLIO_CO_RETURN;
  }
  TraceInodeRec("load", packed, rec, container_id_);
  auto fi = std::make_shared<InodeState>();
  clio::run::u64 size = 0;
  clio::run::u32 writer = kNoRecWriter;
  bool fresh = false;
  if (!DecodeInodeRecord(rec, fi.get(), &size, &writer, &fresh)) {
    FsDec md(rec.data(), rec.size());
    clio::run::u32 magic = 0;
    if (md.U32(&magic) && magic == kOldInodeRecMagic) {
      HLOG(kError, "filesystem: inode {} was written by an older clio-fs "
           "(record format CFI1); this volume must be recreated", packed);
    } else {
      HLOG(kError, "filesystem: inode record {} unreadable", packed);
    }
    if (err != nullptr) *err = EIO;
    CLIO_CO_RETURN;
  }
  fi->tag_id_ = tag;
  // Containers caching its attrs registered with an earlier incarnation of
  // this home: the next change goes to every container. A record written at
  // create (fresh) was never cached anywhere and has no stream to reconcile.
  fi->holders_unknown_ = NumContainers() > 1 && !fresh;
  const bool served_elsewhere =
      writer != kNoRecWriter && writer != container_id_;
  if (fi->type_ == kFsTypeFile && !fresh) {
    // Another container stored the latest record: it served the inode while
    // this one was away (a failover), so this container's stream state is
    // older than the record -- a truncate there must not be undone by the
    // larger size this stream still remembers. The record is never behind an
    // acknowledged size (AdvanceSize and truncate store it before replying),
    // so it is the size. Otherwise the record is a floor: this container's
    // stream may hold appends merged after the record was written (kMax never
    // lowers a size the stream already has).
    const auto op = served_elsewhere ? clio::cte::stream::StreamSizeOp::kSet
                                     : clio::cte::stream::StreamSizeOp::kMax;
    if (stream_hold_) {
      // The stream is co-located (this container is the inode's home, so it
      // homes the stream too). Reconciling in-process also releases the
      // stream if it was restored and held since the restart.
      auto &stream = static_cast<clio::cte::stream::Runtime &>(*stream_hold_);
      stream.ReconcileRestored(tag, op, size);
    } else if (size != 0 || served_elsewhere) {
      clio::run::u32 src = 0;
      CLIO_CO_AWAIT(FileSizeOp(tag, op, size, nullptr, nullptr, &src));
    }
  }
  std::lock_guard<std::mutex> lk(meta_mu_);
  // Dropped while the record was being read: it stays gone (#1150).
  if (dying_.count(packed) != 0) CLIO_CO_RETURN;
  inodes_.emplace(packed, fi);  // a racing loader may have won: keep it
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::MaterializeRoot(int &rc) {
  CLIO_TASK_BODY_BEGIN
  Inode root;
  root.tag_id_ = FsRootId();
  root.type_ = kFsTypeDir;
  root.volume_flags_ = case_insensitive_ ? kVolumeCaseInsensitive : 0u;
  root.fold_version_ = case_insensitive_ ? kFsFoldTableVersion : 0u;
  CLIO_CO_AWAIT(CreateInodeRecord(root, std::string(), rc));
  if (rc == 0) {
    HLOG(kInfo, "filesystem: new volume, names compared case-{}",
         case_insensitive_ ? "insensitively" : "sensitively");
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::CheckVolumeFlags(bool &ok) {
  CLIO_TASK_BODY_BEGIN
  ok = true;
  const clio::run::u64 root = FsPack(FsRootId());
  if (InodeHome(root) != container_id_) CLIO_CO_RETURN;
  int err = 0;
  CLIO_CO_AWAIT(EnsureInode(root, &err));
  std::shared_ptr<InodeState> fi = FindInode(root);
  if (fi == nullptr) {
    // Unreadable (it says why): a volume that cannot be checked is not
    // served, since its names may be keyed differently.
    ok = false;
    CLIO_CO_RETURN;
  }
  bool ci = false;
  clio::run::u32 fold = 0;
  {
    std::lock_guard<std::mutex> g(meta_mu_);
    ci = (fi->volume_flags_ & kVolumeCaseInsensitive) != 0;
    fold = fi->fold_version_;
  }
  if (ci != case_insensitive_) {
    HLOG(kError, "filesystem: this volume was created case-{} but the "
         "configuration says case_insensitive: {}; a volume keeps the "
         "setting it was created with (recreate it to change it)",
         ci ? "insensitive" : "sensitive",
         case_insensitive_ ? "true" : "false");
    ok = false;
  } else if (ci && fold != kFsFoldTableVersion) {
    HLOG(kError, "filesystem: this volume folds names with Unicode table "
         "{} but this build has {}; its names would be keyed "
         "differently", fold, kFsFoldTableVersion);
    ok = false;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

}  // namespace clio::cte::filesystem
