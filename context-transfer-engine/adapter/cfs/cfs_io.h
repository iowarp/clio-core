/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 *
 * Shared, byte-oriented I/O core for the CTE interceptor adapters (POSIX,
 * STDIO, MPI-IO). This replaces the old adapter/filesystem (clio_cte_fs_base)
 * layer: instead of paging + dual-writing to a backend file, every operation
 * delegates to the context-filesystem chimod via clio::cte::filesystem::Client
 * (CLIO_CFS_CLIENT), which owns the path->tag mapping, page-blob I/O and
 * per-file logical-size metadata.
 *
 * Interception is opt-in by the "clio::" path marker (a path component that
 * begins with "clio::"); the marker is stripped before the path is used as a
 * CTE tag name. CTE-issued descriptors start at kCfsFdBase (8192) so they do
 * not collide with kernel fds.
 */
#ifndef CLIO_CTE_ADAPTER_CFS_CFS_IO_H_
#define CLIO_CTE_ADAPTER_CFS_CFS_IO_H_

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

#include "clio_runtime/clio_runtime.h"
#include "clio_cte/core/content_transfer_engine.h"
#include "clio_cte/filesystem/filesystem_client.h"
#include "clio_ctp/util/logging.h"
#include "clio_ctp/util/singleton.h"

namespace clio::cae {

// ---- clio:: path marker (opt-in interception) ----------------------------

static constexpr char kClioPrefix[] = "clio::";
static constexpr size_t kClioPrefixLen = sizeof(kClioPrefix) - 1;  // 6

/** Byte offset where "clio::" appears as a path-component prefix, or npos. */
inline size_t FindClioPrefix(const std::string &path) {
  if (path.size() >= kClioPrefixLen &&
      path.compare(0, kClioPrefixLen, kClioPrefix) == 0) {
    return 0;
  }
  for (size_t cur = 0; cur < path.size(); ++cur) {
    if (path[cur] != '/') continue;
    if (cur + 1 + kClioPrefixLen <= path.size() &&
        path.compare(cur + 1, kClioPrefixLen, kClioPrefix) == 0) {
      return cur + 1;
    }
  }
  return std::string::npos;
}

inline bool HasClioPrefix(const std::string &path) {
  return FindClioPrefix(path) != std::string::npos;
}

/** Remove the first "clio::" marker, yielding the bare backend path. */
inline std::string StripClioPrefix(const std::string &path) {
  size_t pos = FindClioPrefix(path);
  if (pos == std::string::npos) {
    return path;
  }
  return path.substr(0, pos) + path.substr(pos + kClioPrefixLen);
}

// ---- adapter constants -----------------------------------------------------

/** CTE-issued descriptors start here so they never collide with kernel fds. */
static constexpr int kCfsFdBase = 8192;
/** Preferred block size reported by stat (matches the chimod page size). */
static constexpr size_t kCfsBlkSize = 1024 * 1024;
/** Synthetic device id — same for every clio:: file. */
static constexpr dev_t kClioStDev = static_cast<dev_t>(0xC110);

/**
 * Process-wide, thread-safe table of open clio:: descriptors, each delegating
 * to the context-filesystem chimod. Byte-oriented (offset/length); the
 * higher-level adapters (POSIX fd, STDIO FILE*, MPI-IO MPI_File) map their
 * handles onto these descriptors.
 */
class CfsIo {
 public:
  struct OpenFile {
    chi::u64 handle = 0;   ///< chimod handle
    std::string path;      ///< bare (stripped) path == CTE tag name
    chi::u64 off = 0;      ///< current seek offset
    int flags = 0;         ///< open() flags
  };

  CfsIo() = default;

  /** A path is intercepted iff it carries the clio:: marker. */
  static bool IsPathTracked(const std::string &path) {
    return HasClioPrefix(path);
  }

  /** Whether fd was issued by us (and is still open). */
  bool IsFdTracked(int fd) {
    if (fd < kCfsFdBase) {
      return false;
    }
    std::lock_guard<std::mutex> g(mu_);
    return fds_.find(fd) != fds_.end();
  }

  /**
   * open(2): resolve/create the tag, allocate a descriptor. Honors O_CREAT
   * (a plain open of a missing file fails with ENOENT), O_TRUNC (resets the
   * logical size) and O_APPEND (seek starts at EOF). Returns a CTE fd
   * (>= kCfsFdBase) or -1 with errno set.
   */
  int Open(const std::string &raw_path, int flags, int mode);

  /** read(2): from the current offset; advances it. */
  ssize_t Read(int fd, void *buf, size_t count);
  /** write(2): at the current offset (or EOF if O_APPEND); advances it. */
  ssize_t Write(int fd, const void *buf, size_t count);
  /** pread(2): explicit offset, current offset untouched. */
  ssize_t Pread(int fd, void *buf, size_t count, off_t offset);
  /** pwrite(2): explicit offset, current offset untouched. */
  ssize_t Pwrite(int fd, const void *buf, size_t count, off_t offset);
  /** lseek(2): SEEK_SET/CUR/END. */
  off_t Seek(int fd, off_t offset, int whence);
  /** Current offset of fd (ftell), or -1 if untracked. */
  off_t Tell(int fd);
  /** Logical size of the fd's file, or -1 if untracked. */
  off_t SizeFd(int fd);
  /** close(2): release the chimod handle + descriptor. */
  int Close(int fd);
  /** fsync(2): writes are synchronous, so a no-op success. */
  int Sync(int fd);
  /** ftruncate(2): set logical size of the fd's file. */
  int FtruncateFd(int fd, off_t length);
  /** truncate(2) / by path. */
  int TruncatePath(const std::string &raw_path, off_t length);
  /** unlink(2)/remove(3) by path. */
  int RemovePath(const std::string &raw_path);

  /** fstat(2): fill *buf from the fd's chimod metadata. */
  template <typename StatT>
  int StatFd(int fd, StatT *buf) {
    OpenFile of;
    {
      std::lock_guard<std::mutex> g(mu_);
      auto it = fds_.find(fd);
      if (it == fds_.end()) {
        errno = EBADF;
        return -1;
      }
      of = it->second;
    }
    chi::u64 size = 0;
    if (!QuerySize(of.path, &size)) {
      errno = EBADF;
      return -1;
    }
    FillStat(buf, of.path, size);
    return 0;
  }

  /** stat(2): fill *buf for a clio:: path; ENOENT if it does not exist. */
  template <typename StatT>
  int StatPath(const std::string &raw_path, StatT *buf) {
    std::string path = StripClioPrefix(raw_path);
    chi::u64 size = 0;
    bool exists = false;
    if (!QueryGetattr(path, &exists, &size) || !exists) {
      std::memset(buf, 0, sizeof(StatT));
      errno = ENOENT;
      return -1;
    }
    FillStat(buf, path, size);
    return 0;
  }

 private:
  std::mutex mu_;
  std::unordered_map<int, OpenFile> fds_;
  int next_fd_ = kCfsFdBase;
  bool client_ready_ = false;

  /** Lazily create/bind the filesystem pool on first tracked use. */
  bool EnsureClient();

  /** Logical size of a tag (0 if absent). */
  bool QuerySize(const std::string &path, chi::u64 *size);
  /** Existence + logical size of a tag. */
  bool QueryGetattr(const std::string &path, bool *exists, chi::u64 *size);

  /** Core read/write against a chimod handle (no offset bookkeeping). */
  ssize_t DoRead(chi::u64 handle, chi::u64 off, void *buf, size_t count);
  ssize_t DoWrite(chi::u64 handle, chi::u64 off, const void *buf, size_t count);

  /** Stable 64-bit synthetic inode from the bare path. */
  static uint64_t SyntheticInode(const std::string &path) {
    size_t h = std::hash<std::string>{}(path);
    return static_cast<uint64_t>(h) ^ (static_cast<uint64_t>(h) << 32);
  }

  template <typename StatT>
  static void FillStat(StatT *buf, const std::string &path, chi::u64 size) {
    std::memset(buf, 0, sizeof(StatT));
    buf->st_dev = kClioStDev;
    buf->st_ino = SyntheticInode(path);
    buf->st_mode = S_IFREG | 0644;
    buf->st_nlink = 1;
    buf->st_uid = CTP_SYSTEM_INFO->uid_;
    buf->st_gid = CTP_SYSTEM_INFO->gid_;
    buf->st_size = static_cast<off_t>(size);
    buf->st_blksize = static_cast<blksize_t>(kCfsBlkSize);
    // POSIX st_blocks counts 512-byte units; round up so non-empty != 0.
    buf->st_blocks = static_cast<blkcnt_t>((size + 511) / 512);
  }
};

}  // namespace clio::cae

namespace clio::cae {
CTP_DEFINE_GLOBAL_PTR_VAR_H(CfsIo, g_cfs_io);
}

/** Access the process-wide CfsIo singleton. */
#define CLIO_CTE_CFS \
  (CTP_GET_GLOBAL_PTR_VAR(clio::cae::CfsIo, clio::cae::g_cfs_io))
#define CLIO_CTE_CFS_T clio::cae::CfsIo *

#endif  // CLIO_CTE_ADAPTER_CFS_CFS_IO_H_
