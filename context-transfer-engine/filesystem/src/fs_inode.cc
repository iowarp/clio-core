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

std::string Runtime::EncInodeRec(const FileInfo &fi, clio::run::u64 size) {
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
  return p;
}

bool Runtime::DecInodeRec(const std::string &rec, FileInfo *fi,
                          clio::run::u64 *size) {
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
  return true;
}

void Runtime::MarkInodeDirtyLocked(const FileInfo &fi) {
  inode_dirty_.insert(FsPack(fi.tag_id_));
}

clio::run::TaskResume Runtime::FlushInodes() {
  CLIO_TASK_BODY_BEGIN
  // Store what is dirty now. An inode another task is storing stays dirty
  // and is picked up after it finishes: that store may predate this task's
  // change, which must be durable before this task replies.
  for (;;) {
    std::vector<std::pair<clio::run::u64, std::string>> work;
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
          inode_storing_.insert(packed);
          work.emplace_back(packed,
                            EncInodeRec(*fit->second, FileSize(*fit->second)));
        }
        it = inode_dirty_.erase(it);  // dropped inodes need no record
      }
    }
    for (const auto &w : work) {
      auto p = cte_.AsyncPutBlob(FsUnpack(w.first), kInodeBlob, 0,
                                 w.second.size(), w.second.data(), -1.0f,
                                 clio::cte::core::Context(), 0u,
                                 clio::run::PoolQuery::Dynamic());
      CLIO_CO_AWAIT(p);
      std::lock_guard<std::mutex> g(meta_mu_);
      inode_storing_.erase(w.first);
      if (p->GetReturnCode() != 0) {
        HLOG(kWarning, "filesystem: storing inode {:x} failed (rc {}); will "
             "retry", w.first, p->GetReturnCode());
        if (by_tag_.count(w.first) != 0) inode_dirty_.insert(w.first);
      }
    }
    if (busy == 0 && work.empty()) break;
    if (busy == 0) continue;  // anything dirtied meanwhile goes too
    CLIO_CO_AWAIT(clio::run::yield(kStoreBusyPollUs));
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::ReadInodeRecord(clio::run::u64 packed,
                                               FsResp &resp) {
  CLIO_TASK_BODY_BEGIN
  const clio::cte::core::TagId tag = FsUnpack(packed);
  resp.rc_ = ENOENT;
  auto sz = cte_.AsyncGetBlobSize(tag, kInodeBlob);
  CLIO_CO_AWAIT(sz);
  if (sz->GetReturnCode() != 0 || sz->size_ == 0) CLIO_CO_RETURN;
  std::string rec(sz->size_, '\0');
  auto g = cte_.AsyncGetBlob(tag, kInodeBlob, 0, rec.size(), 0u, rec.data());
  CLIO_CO_AWAIT(g);
  if (g->GetReturnCode() != 0) {
    resp.rc_ = EIO;
    CLIO_CO_RETURN;
  }
  FileInfo fi;
  clio::run::u64 size = 0;
  if (!DecInodeRec(rec, &fi, &size)) {
    resp.rc_ = EIO;
    CLIO_CO_RETURN;
  }
  resp.attr_.id_ = packed;
  resp.attr_.type_ = fi.type_;
  resp.attr_.size_ = fi.type_ == kFsTypeSymlink ? fi.symlink_.size() : size;
  resp.attr_.nlink_ = fi.nlink_;
  resp.attr_.mode_ = fi.mode_;
  resp.attr_.uid_ = fi.uid_;
  resp.attr_.gid_ = fi.gid_;
  resp.attr_.atime_ = fi.atime_;
  resp.attr_.mtime_ = fi.mtime_;
  resp.attr_.ctime_ = fi.ctime_;
  resp.str_ = fi.symlink_;
  resp.rc_ = 0;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
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
  if (g->GetReturnCode() != 0 || !DecInodeRec(rec, fi.get(), &size)) {
    HLOG(kError, "filesystem: inode record {:x} unreadable", packed);
    CLIO_CO_RETURN;
  }
  fi->tag_id_ = tag;
  std::lock_guard<std::mutex> lk(meta_mu_);
  by_tag_.emplace(packed, fi);  // a racing loader may have won: keep it
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

}  // namespace clio::cte::filesystem
