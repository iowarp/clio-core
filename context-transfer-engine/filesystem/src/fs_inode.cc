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
 * attributes are one small blob (kInodeBlob) in the file's own tag, written
 * by the inode's home before it acknowledges a change and read by any
 * container (through the cache chain) to stat the file.
 */

#include <algorithm>
#include <clio_cte/core/blob_placement.h>
#include <clio_cte/filesystem/filesystem_runtime.h>

namespace clio::cte::filesystem {

namespace {
/** Blob holding an inode's record, in the file's tag (never a page name). */
constexpr const char *kInodeBlob = "~i";
/** Record format tag. */
constexpr clio::run::u32 kInodeRecMagic = 0x31494643u;  // "CFI1"
/** Wait between checks while another task stores the same inode (us). */
constexpr double kStoreBusyPollUs = 20.0;
}  // namespace

std::string Runtime::EncInodeRec(const FileInfo &fi, clio::run::u64 size,
                                 clio::run::u32 writer) {
  std::string p;
  FsEnc e(&p);
  e.U32(kInodeRecMagic);
  e.U32(fi.type_);
  e.U64(size);
  e.U32(fi.nlink_);
  e.U32(fi.mode_);
  e.U32(fi.uid_);
  e.U32(fi.gid_);
  e.U64(fi.atime_);
  e.U64(fi.mtime_);
  e.U64(fi.ctime_);
  e.U32((fi.orphan_ ? 1u : 0u) | (fi.has_xattr_ ? 2u : 0u));
  e.Str(fi.symlink_);
  // Appended last so older decoders (and older records) still line up.
  e.U32(writer);
  return p;
}

bool Runtime::DecInodeRec(const std::string &rec, FileInfo *fi,
                          clio::run::u64 *size, clio::run::u32 *writer) {
  FsDec d(rec.data(), rec.size());
  clio::run::u32 magic = 0, flags = 0;
  if (!d.U32(&magic) || magic != kInodeRecMagic || !d.U32(&fi->type_) ||
      !d.U64(size) || !d.U32(&fi->nlink_) || !d.U32(&fi->mode_) ||
      !d.U32(&fi->uid_) || !d.U32(&fi->gid_) || !d.U64(&fi->atime_) ||
      !d.U64(&fi->mtime_) || !d.U64(&fi->ctime_) || !d.U32(&flags) ||
      !d.Str(&fi->symlink_)) {
    return false;
  }
  fi->orphan_ = (flags & 1u) != 0;
  fi->has_xattr_ = (flags & 2u) != 0;
  if (!d.U32(writer)) *writer = kNoRecWriter;
  return true;
}

void Runtime::MarkInodeDirtyLocked(const FileInfo &fi) {
  inode_dirty_.insert(FsPack(fi.tag_id_));
}

namespace {
/** Encode the attributes an inode push carries (attrs + symlink target). */
std::string EncInodePush(const Runtime::FsAttr &a, const std::string &link) {
  std::string out;
  FsEnc e(&out);
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
  e.Str(link);
  return out;
}

/** Decode what EncInodePush wrote. @return false if malformed */
bool DecInodePush(const std::string &s, Runtime::FsAttr *a, std::string *link) {
  FsDec d(s.data(), s.size());
  return d.U64(&a->id_) && d.U32(&a->type_) && d.U64(&a->size_) &&
         d.U32(&a->nlink_) && d.U32(&a->mode_) && d.U32(&a->uid_) &&
         d.U32(&a->gid_) && d.U64(&a->atime_) && d.U64(&a->mtime_) &&
         d.U64(&a->ctime_) && d.Str(link);
}

/** One inode record to store and push. */
struct InodeWork {
  clio::run::u64 packed = 0;
  std::string rec;                      ///< the CTE record
  std::string push;                     ///< EncInodePush of the same state
  std::map<clio::run::u32, clio::run::u64> holders;  ///< caching it (+ reg)
  std::map<clio::run::u32, clio::run::u64> leases;   ///< their lease ends
  bool resync = false;  ///< holders unknown: pushed to every container
};
}  // namespace

clio::run::TaskResume Runtime::StoreInodeRec(clio::run::u64 packed,
                                             const std::string &rec,
                                             int &rc) {
  CLIO_TASK_BODY_BEGIN
  // kMetaBlob: storing the record is not a change to the file (no ctime
  // bump); a non-volatile tier keeps it across a restart.
  clio::cte::core::Context meta_ctx;
  meta_ctx.op_flags_ |= clio::cte::core::Context::kMetaBlob;
  meta_ctx.min_persistence_level_ = inode_volatile_only_ ? 0 : 1;
  auto p = cte_.AsyncPutBlob(FsUnpack(packed), kInodeBlob, 0,
                             rec.size(), rec.data(), -1.0f, meta_ctx,
                             0u, clio::run::PoolQuery::Dynamic());
  CLIO_CO_AWAIT(p);
  if (p->GetReturnCode() != 0 && !inode_volatile_only_) {
    meta_ctx.min_persistence_level_ = 0;
    p = cte_.AsyncPutBlob(FsUnpack(packed), kInodeBlob, 0, rec.size(),
                          rec.data(), -1.0f, meta_ctx, 0u,
                          clio::run::PoolQuery::Dynamic());
    CLIO_CO_AWAIT(p);
    if (p->GetReturnCode() == 0) {
      HLOG(kWarning, "filesystem: no non-volatile CTE tier accepts inode "
           "records; they stay in RAM and do not survive a restart");
      inode_volatile_only_ = true;
    }
  }
  rc = static_cast<int>(p->GetReturnCode());
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::FlushInodes() {
  CLIO_TASK_BODY_BEGIN
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
        if (inode_storing_.count(packed) != 0) {
          ++busy;
          ++it;
          continue;
        }
        auto fit = by_tag_.find(packed);
        if (fit != by_tag_.end()) {
          const FileInfo &fi = *fit->second;
          inode_storing_.insert(packed);
          InodeWork w;
          w.packed = packed;
          w.rec = EncInodeRec(fi, FileSize(fi), container_id_);
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
        it = inode_dirty_.erase(it);  // dropped inodes need no record
      }
    }
    for (const InodeWork &w : work) {
      int src = 0;
      CLIO_CO_AWAIT(StoreInodeRec(w.packed, w.rec, src));
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
      std::lock_guard<std::mutex> g(meta_mu_);
      inode_storing_.erase(w.packed);
      auto fit = by_tag_.find(w.packed);
      if (fit != by_tag_.end()) {
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
      if (src != 0) {
        HLOG(kWarning, "filesystem: storing inode {} failed (rc {}); will "
             "retry", w.packed, src);
        if (fit != by_tag_.end()) inode_dirty_.insert(w.packed);
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
    CLIO_CO_AWAIT(EnsureInode(packed));
    std::shared_ptr<FileInfo> fi = FindInode(packed);
    if (fi == nullptr) {
      resp.rc_ = ENOENT;
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

int Runtime::ApplyInodePush(const FsReq &req) {
  InodeCacheEnt ent;
  if (!DecInodePush(req.str_, &ent.attr_, &ent.symlink_)) return EINVAL;
  std::lock_guard<std::mutex> g(icache_mu_);
  auto lit = iloading_.find(req.id_);
  if (lit != iloading_.end()) {
    ent.home_ = 0;  // any value but ~0u: "a push arrived"
    lit->second = ent;
    return 0;
  }
  auto it = icache_.find(req.id_);
  if (it == icache_.end()) return ENOENT;
  ent.home_ = it->second.home_;
  ent.lease_until_ms_ = it->second.lease_until_ms_;  // a push is no renewal
  it->second = ent;
  return 0;
}

clio::run::TaskResume Runtime::EnsureInode(clio::run::u64 packed) {
  CLIO_TASK_BODY_BEGIN
  if (packed == 0 || InodeOwner(packed) != container_id_ ||
      FindInode(packed) != nullptr) {
    CLIO_CO_RETURN;
  }
  const clio::cte::core::TagId tag = FsUnpack(packed);
  auto sz = cte_.AsyncGetBlobSize(tag, kInodeBlob);
  CLIO_CO_AWAIT(sz);
  if (sz->GetReturnCode() != 0 || sz->size_ == 0) CLIO_CO_RETURN;
  std::string rec(sz->size_, '\0');
  auto g = cte_.AsyncGetBlob(tag, kInodeBlob, 0, rec.size(), 0u, rec.data());
  CLIO_CO_AWAIT(g);
  auto fi = std::make_shared<FileInfo>();
  clio::run::u64 size = 0;
  clio::run::u32 writer = kNoRecWriter;
  if (g->GetReturnCode() != 0 ||
      !DecInodeRec(rec, fi.get(), &size, &writer)) {
    HLOG(kError, "filesystem: inode record {} unreadable", packed);
    CLIO_CO_RETURN;
  }
  fi->tag_id_ = tag;
  // Containers caching its attrs registered with an earlier incarnation of
  // this home: the next change goes to every container.
  fi->holders_unknown_ = NumContainers() > 1;
  const bool served_elsewhere =
      writer != kNoRecWriter && writer != container_id_;
  if (fi->type_ != kFsTypeSymlink) {
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
  by_tag_.emplace(packed, fi);  // a racing loader may have won: keep it
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

}  // namespace clio::cte::filesystem
