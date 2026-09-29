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
 * Placement helpers of the clio-fs namespace.
 *
 * There is no metadata server. Directories are stored as directory blocks
 * (CTE blobs, see fs_dir_block.h) whose home is the container owning the
 * blob; inodes are homed on the container that minted their id (encoded in
 * the id); file pages follow the CTE core's per-page hash. Path operations
 * go to the local container, which resolves paths through its cache of
 * directory blocks.
 *
 * This header is shared by clients (who route tasks with it) and the
 * runtime, so the two can never disagree.
 */
#ifndef CLIO_CTE_FILESYSTEM_FS_SHARD_H_
#define CLIO_CTE_FILESYSTEM_FS_SHARD_H_

#include <cstring>
#include <string>

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_tasks.h>

namespace clio::cte::filesystem {

/** Entry types stored in a directory entry and an inode. */
GLOBAL_CROSS_CONST clio::run::u32 kFsTypeFile = 1;
GLOBAL_CROSS_CONST clio::run::u32 kFsTypeDir = 2;
GLOBAL_CROSS_CONST clio::run::u32 kFsTypeSymlink = 3;

/** Id minted by a filesystem container: major = kFsIdFlag | home. */
GLOBAL_CROSS_CONST clio::run::u32 kFsIdFlag = 0x40000000u;
/** Id minted by a client (sieve create): major = flag | node<<16 | home. */
GLOBAL_CROSS_CONST clio::run::u32 kFsClientIdFlag = 0x80000000u;
/** Low bits of an fs/client-minted major that hold the home container. */
GLOBAL_CROSS_CONST clio::run::u32 kFsHomeMask = 0xFFFFu;

/**
 * splitmix64 finalizer: spreads sequential or low-entropy keys uniformly.
 * @param x key
 * @return mixed 64-bit value
 */
inline clio::run::u64 FsMix64(clio::run::u64 x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

/**
 * Stable hash of a path string (FNV-1a, then mixed). Identical in every
 * process and build, unlike std::hash.
 * @param p absolute, normalized path
 * @return 32-bit shard hash
 */
inline clio::run::u32 FsPathHash(const std::string &p) {
  clio::run::u64 h = 0xCBF29CE484222325ULL;
  for (unsigned char c : p) {
    h ^= c;
    h *= 0x100000001B3ULL;
  }
  h = FsMix64(h);
  return static_cast<clio::run::u32>(h ^ (h >> 32));
}

/**
 * Normalize a path: drop one trailing '/' (keeping the bare root "/").
 * @param p path
 * @return normalized path
 */
inline std::string FsNormPath(std::string p) {
  while (p.size() > 1 && p.back() == '/') p.pop_back();
  if (p.empty()) p = "/";
  return p;
}

/**
 * Parent directory of an absolute path ("/a/b" -> "/a", "/a" -> "/",
 * "/" -> "/").
 * @param path absolute path
 * @return parent directory path
 */
inline std::string FsParentDir(const std::string &path) {
  std::string s = FsNormPath(path);
  size_t slash = s.find_last_of('/');
  if (slash == std::string::npos || slash == 0) return "/";
  return s.substr(0, slash);
}

/**
 * Last component of an absolute path ("/a/b" -> "b", "/" -> "").
 * @param path absolute path
 * @return leaf name
 */
inline std::string FsLeaf(const std::string &path) {
  std::string s = FsNormPath(path);
  if (s == "/") return std::string();
  size_t slash = s.find_last_of('/');
  return slash == std::string::npos ? s : s.substr(slash + 1);
}

/**
 * Join a directory and a leaf ("/" + "a" -> "/a", "/a" + "b" -> "/a/b").
 * @param dir directory path
 * @param leaf entry name
 * @return child path
 */
inline std::string FsJoin(const std::string &dir, const std::string &leaf) {
  return dir == "/" ? "/" + leaf : dir + "/" + leaf;
}

/**
 * Query for a path-based operation: the local container. It resolves the
 * path through its own cache of directory blocks (fs_dir_block.h) and sends
 * only mutations to the homes of the blocks involved.
 * @return Local query
 */
inline clio::run::PoolQuery FsPathQuery() {
  return clio::run::PoolQuery::Local();
}

/**
 * Query reaching the container that owns directory `dir` (its listing, its
 * attributes and every entry in it).
 * @param dir normalized directory path
 * @return DirectHash query
 */
inline clio::run::PoolQuery FsDirQuery(const std::string &dir) {
  return clio::run::PoolQuery::DirectHash(FsPathHash(dir));
}

/**
 * Query reaching the owner of the ENTRY for `path` (its parent directory).
 * @param path absolute path
 * @return DirectHash query
 */
inline clio::run::PoolQuery FsEntryQuery(const std::string &path) {
  return FsDirQuery(FsParentDir(path));
}

/**
 * Container owning directory `dir` among `n` containers (must match the
 * runtime's DirectHash resolution: hash % num_containers).
 * @param dir normalized directory path
 * @param n number of containers
 * @return container id
 */
inline clio::run::u32 FsDirContainer(const std::string &dir, clio::run::u32 n) {
  return n == 0 ? 0 : FsPathHash(dir) % n;
}

/** Pack a TagId as (major << 32) | minor. */
inline clio::run::u64 FsPack(const clio::cte::core::TagId &t) {
  return (static_cast<clio::run::u64>(t.major_) << 32) |
         static_cast<clio::run::u64>(t.minor_);
}

/** Unpack (major << 32) | minor into a TagId. */
inline clio::cte::core::TagId FsUnpack(clio::run::u64 p) {
  return clio::cte::core::TagId(static_cast<clio::run::u32>(p >> 32),
                                static_cast<clio::run::u32>(p & 0xFFFFFFFFULL));
}

/**
 * The root directory's id (its inode number). Its major is outside every
 * range a node, a filesystem container or a client ever mints.
 */
inline clio::cte::core::TagId FsRootId() {
  return clio::cte::core::TagId(0x3FFFFFFFu, 1u);
}

/**
 * True when `packed` names an inode with an encoded home container.
 * @param packed packed TagId
 * @return whether FsIdHome applies
 */
inline bool FsIdHasHome(clio::run::u64 packed) {
  const clio::run::u32 major = static_cast<clio::run::u32>(packed >> 32);
  return (major & (kFsIdFlag | kFsClientIdFlag)) != 0;
}

/**
 * Home container of an inode id (see FsIdHasHome).
 * @param packed packed TagId
 * @return the container that owns the inode
 */
inline clio::run::u32 FsIdHome(clio::run::u64 packed) {
  return static_cast<clio::run::u32>(packed >> 32) & kFsHomeMask;
}

/**
 * Query reaching the container that owns inode `packed`.
 * @param packed packed TagId
 * @return DirectId to the encoded home, or DirectHash of the id
 */
inline clio::run::PoolQuery FsInodeQuery(clio::run::u64 packed) {
  if (FsIdHasHome(packed)) {
    return clio::run::PoolQuery::DirectId(
        static_cast<clio::run::ContainerId>(FsIdHome(packed)));
  }
  return clio::run::PoolQuery::DirectHash(
      static_cast<clio::run::u32>(FsMix64(packed)));
}

/**
 * Build an open-file handle. The inode's home container rides in the top 16
 * bits so every handle operation routes straight to the inode owner.
 * @param home container holding the handle table entry
 * @param counter per-container sequence number (nonzero)
 * @return handle; never 0
 */
inline clio::run::u64 FsMakeHandle(clio::run::u32 home, clio::run::u64 counter) {
  return (static_cast<clio::run::u64>((home & 0xFFFFu) + 1) << 48) |
         (counter & 0xFFFFFFFFFFFFULL);
}

/**
 * Query reaching the container that holds handle `h`.
 * @param h handle from Open
 * @return DirectId to the handle's home (Local for a malformed handle)
 */
inline clio::run::PoolQuery FsHandleQuery(clio::run::u64 h) {
  const clio::run::u64 top = h >> 48;
  if (top == 0) return clio::run::PoolQuery::Local();
  return clio::run::PoolQuery::DirectId(
      static_cast<clio::run::ContainerId>(top - 1));
}

// ---------------------------------------------------------------------------
// Tiny binary codec for internal shard requests and the metadata log.
// ---------------------------------------------------------------------------

/** Appends little-endian scalars and length-prefixed strings. */
class FsEnc {
 public:
  explicit FsEnc(std::string *out) : out_(out) {}
  /** Append a u8. */
  void U8(clio::run::u32 v) { out_->push_back(static_cast<char>(v & 0xFF)); }
  /** Append a u32. */
  void U32(clio::run::u32 v) { out_->append(reinterpret_cast<const char *>(&v), 4); }
  /** Append a u64. */
  void U64(clio::run::u64 v) { out_->append(reinterpret_cast<const char *>(&v), 8); }
  /** Append a length-prefixed string. */
  void Str(const std::string &s) {
    U32(static_cast<clio::run::u32>(s.size()));
    out_->append(s);
  }

 private:
  std::string *out_;
};

/** Reads what FsEnc wrote; every getter fails (returns false) on underflow. */
class FsDec {
 public:
  FsDec(const char *data, size_t len) : data_(data), len_(len) {}
  /** Read a u8. */
  bool U8(clio::run::u32 *v) {
    if (off_ + 1 > len_) return false;
    *v = static_cast<unsigned char>(data_[off_++]);
    return true;
  }
  /** Read a u32. */
  bool U32(clio::run::u32 *v) { return Raw(v, 4); }
  /** Read a u64. */
  bool U64(clio::run::u64 *v) { return Raw(v, 8); }
  /** Read a length-prefixed string. */
  bool Str(std::string *s) {
    clio::run::u32 n = 0;
    if (!U32(&n) || off_ + n > len_) return false;
    s->assign(data_ + off_, n);
    off_ += n;
    return true;
  }
  /** True when every byte has been consumed. */
  bool Done() const { return off_ == len_; }

 private:
  bool Raw(void *v, size_t n) {
    if (off_ + n > len_) return false;
    std::memcpy(v, data_ + off_, n);
    off_ += n;
    return true;
  }
  const char *data_;
  size_t len_;
  size_t off_ = 0;
};

}  // namespace clio::cte::filesystem

#endif  // CLIO_CTE_FILESYSTEM_FS_SHARD_H_
