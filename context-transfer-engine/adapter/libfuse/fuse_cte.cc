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

#include "fuse_cte.h"

#include <algorithm>
#include <climits>
#include <cstring>
#include <string>
#include <vector>
#include <cerrno>                 // errno
#include <cstdio>                 // snprintf, fprintf
#include <fcntl.h>                // O_CREAT, O_RDWR
#ifdef __linux__
#include <linux/falloc.h>         // FALLOC_FL_* (fallocate mode flags)
#endif

#ifndef _WIN32
#include <sys/statvfs.h>          // struct statvfs (statfs op)
#include <unistd.h>               // read, getuid, getgid
#endif  // _WIN32
// The mount / process-entry glue (fuse_main, the apptainer --fusemount custom-io
// path, and the fuse_lowlevel/dlsym/mount/uio machinery it needs) now lives in
// fuse_cte_main.cc. This file holds only the operation callbacks.

#include "clio_runtime/clio_runtime.h"
#include "clio_cte/core/content_transfer_engine.h"
#include "clio_cte/core/core_client.h"  // CLIO_CTE_CLIENT + GetCapacity
#include "clio_cte/filesystem/filesystem_client.h"

// Bridge the POSIX type spellings the callbacks use to the concrete types
// each platform's FUSE layer expects in `struct fuse_operations`. On Linux
// (libfuse) the high-level API uses the POSIX types directly; on Windows the
// shim maps them to WinFsp's fuse_* types and supplies getuid/getgid/S_IF*.
#ifdef _WIN32
#include "fuse_win_compat.h"
#else
using cte_stat_t = struct stat;
using cte_off_t = off_t;
using cte_mode_t = mode_t;
using cte_timespec_t = struct timespec;
using cte_statvfs_t = struct statvfs;
#endif

using namespace clio::cae::fuse;

// ============================================================================
// Helpers
// ============================================================================
//
// The libfuse adapter is now a thin shim over the filesystem chimod (issue
// #552): every FUSE callback resolves to a path/handle operation on the
// process-wide filesystem client (CLIO_CFS_CLIENT), which owns the path->tag
// mapping, page-blob I/O, and per-file logical-size metadata. Reads/writes
// are synchronous from FUSE's perspective (the client Waits on each op), so
// there is no per-fd pending-write queue here anymore.

namespace {
/** Per-open-file state: the chimod handle + the path it was opened on. */
struct CfsHandle {
  clio::run::u64 fh = 0;
  std::string path;
};

CfsHandle *GetHandle(struct fuse_file_info *fi) {
  return reinterpret_cast<CfsHandle *>(fi->fh);
}
}  // namespace

// ============================================================================
// FUSE lifecycle
// ============================================================================

static void *cte_fuse_init(struct fuse_conn_info *conn,
                           struct fuse_config *cfg) {
  (void)conn;
  // Trust the inode numbers we report (st_ino in getattr, d_ino in readdir),
  // both derived from the tag id, instead of letting FUSE auto-generate them.
  // This makes stat and readdir agree on d_ino/st_ino (generic/637), and gives
  // hard-link aliases (which share a TagId) the same inode.
  cfg->use_ino = 1;
  // Keep the kernel page cache for file data (direct_io OFF). mmap on a FUSE
  // file is served generically by the page cache — there is no .mmap callback
  // in the high-level API; the kernel faults mapped pages through cte_fuse_read
  // and flushes dirty pages through cte_fuse_write. direct_io bypasses the page
  // cache, so the kernel returns ENODEV ("No such device") for any mmap (issue
  // #597). Writes stay write-through (no FUSE_CAP_WRITEBACK_CACHE), so each
  // write() still reaches the chimod synchronously and the exact logical size
  // is preserved; only mmap dirty pages flush lazily, which is inherent to mmap.
  cfg->direct_io = 0;
  // Disable the kernel attribute/entry caches. Metadata (size, and especially
  // st_nlink for hard links) can change without this FUSE process being the one
  // that triggered the change, and there is no upcall to invalidate the cache.
  // Without this, e.g. `ln a b; stat a` returns a's stale cached nlink. Every
  // getattr/lookup goes to the chimod, which is the source of truth.
  cfg->attr_timeout = 0;
  cfg->entry_timeout = 0;
  cfg->negative_timeout = 0;

  bool success = clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
  if (!success) {
    fprintf(stderr, "ERROR: CLIO_INIT failed\n");
    return nullptr;
  }
  // Create-or-bind the filesystem chimod pool (which also brings up the CTE
  // core pool it sits over). Every FUSE op below routes through CLIO_CFS_CLIENT.
  if (!clio::cte::filesystem::CLIO_CFS_CLIENT_INIT()) {
    fprintf(stderr, "ERROR: filesystem client init failed\n");
    return nullptr;
  }
  // Bind the CTE core client to the same clio_cte_core pool (idempotent) so
  // statfs can query real capacity via GetCapacity.
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    fprintf(stderr, "WARNING: CTE core client init failed; statfs capacity=0\n");
  }
  return nullptr;
}

static void cte_fuse_destroy(void *private_data) {
  (void)private_data;
  clio::run::CLIO_RUNTIME_FINALIZE();
}

// ============================================================================
// Metadata
// ============================================================================

// Decode a tag timestamp (stored as the two's-complement bits of an i64
// nanoseconds-since-epoch value in a u64 field) into a POSIX timespec. Using
// signed floor division makes pre-epoch (negative) times round-trip correctly
// — an unsigned divide turns e.g. Jan 1 1960 into a huge positive year
// (generic/258). For normal post-epoch times the value and result are
// identical to the old unsigned path (remainder is non-negative). tv_nsec is
// always normalized into [0, 1e9).
// NsecT is templated so this binds to the platform's timespec tv_nsec type:
// `long` on Linux/libfuse, `int64_t` on Windows/WinFsp (struct fuse_timespec).
template <typename NsecT>
static inline void NsBitsToTimespec(clio::run::u64 bits, time_t &sec,
                                    NsecT &nsec) {
  int64_t ns = static_cast<int64_t>(bits);
  int64_t s = ns / 1000000000LL;
  int64_t rem = ns % 1000000000LL;
  if (rem < 0) {
    s -= 1;
    rem += 1000000000LL;
  }
  sec = static_cast<time_t>(s);
  nsec = static_cast<NsecT>(rem);
}

static int cte_fuse_getattr_stat(const char *path, cte_stat_t *stbuf,
                                 struct fuse_file_info *fi) {
  (void)fi;
  memset(stbuf, 0, sizeof(*stbuf));

  std::string p(path);

  // Root is always a directory.
  if (p == "/") {
    stbuf->st_mode = S_IFDIR | 0755;
    stbuf->st_nlink = 2;
    stbuf->st_ino = 1;  // fixed root inode
    stbuf->st_uid = getuid();
    stbuf->st_gid = getgid();
    return 0;
  }

  // Delegate to the filesystem chimod: it owns exists/is-dir/logical-size.
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncGetattr(p);
  t.Wait();
  if (t->GetReturnCode() != 0 || t->exists_ == 0) {
    return -ENOENT;
  }
  // Owner: a prior chown recorded an override (uid_/gid_ != 0xFFFFFFFF);
  // otherwise report the mounting user's uid/gid (files carry no stored owner).
  stbuf->st_uid =
      (t->uid_ != 0xFFFFFFFFu) ? static_cast<uid_t>(t->uid_) : getuid();
  stbuf->st_gid =
      (t->gid_ != 0xFFFFFFFFu) ? static_cast<gid_t>(t->gid_) : getgid();
  stbuf->st_ino = static_cast<ino_t>(t->ino_);  // stable inode = packed TagId
  // A chmod/create recorded permission bits (t->mode_ != 0xFFFFFFFF) win over
  // the synthesized defaults; keep only the low 12 bits (perms + setuid/gid/sticky).
  const bool have_mode = (t->mode_ != 0xFFFFFFFFu);
  const unsigned int perm = t->mode_ & 07777u;
  if (t->is_dir_) {
    stbuf->st_mode = S_IFDIR | (have_mode ? perm : 0755u);
    stbuf->st_nlink = 2;
  } else if (t->is_symlink_) {
    stbuf->st_mode = S_IFLNK | 0777;
    stbuf->st_nlink = 1;
    stbuf->st_size = static_cast<cte_off_t>(t->size_);  // target length
  } else {
    stbuf->st_mode = S_IFREG | (have_mode ? perm : 0644u);
    // POSIX link count = canonical name (1) + tag-level hard-link aliases.
    // Ask the CTE core how many extra names are bound to this tag.
    nlink_t nlink = 1;
    auto *cte = CLIO_CTE_CLIENT;
    if (cte != nullptr) {
      auto na = cte->AsyncGetNumAliases(p);
      na.Wait();
      if (na->return_code_ == 0 && na->found_) {
        nlink = static_cast<nlink_t>(na->num_aliases_) + 1;
      }
    }
    stbuf->st_nlink = nlink;
    // cte_off_t is off_t on Linux; the WinFsp shim maps it for Windows.
    stbuf->st_size = static_cast<cte_off_t>(t->size_);
  }
  // Report the 512-byte block count backing the file so stat(2) st_blocks is
  // non-zero for files that hold data (generic/615 asserts a buffered/direct
  // write shows allocated blocks). Derived from the logical size; directories
  // report 0. st_blksize advertises a sensible I/O unit for tools.
  stbuf->st_blksize = static_cast<decltype(stbuf->st_blksize)>(4096);
  stbuf->st_blocks = static_cast<decltype(stbuf->st_blocks)>(
      (static_cast<uint64_t>(stbuf->st_size) + 511) / 512);
  // Timestamps come from the tag as ns since the epoch (0 means the chimod had
  // no value, so leave that field at the epoch): ctime = last metadata change
  // (last_changed_), mtime = last content change (last_modified_), atime = last
  // access (last_read_). All three are surfaced from the same GetTagSize query.
  if (t->ctime_ != 0) {
    NsBitsToTimespec(t->ctime_, stbuf->st_ctim.tv_sec, stbuf->st_ctim.tv_nsec);
  }
  // Fall back to ctime when mtime is unknown, so a valid file never reports
  // mtime at the epoch while it has a real ctime (merged from #680).
  clio::run::u64 mtime_ns = (t->mtime_ != 0) ? t->mtime_ : t->ctime_;
  if (mtime_ns != 0) {
    NsBitsToTimespec(mtime_ns, stbuf->st_mtim.tv_sec, stbuf->st_mtim.tv_nsec);
  }
  if (t->atime_ != 0) {
    NsBitsToTimespec(t->atime_, stbuf->st_atim.tv_sec, stbuf->st_atim.tv_nsec);
  }
  return 0;
}

#ifdef __APPLE__
// macFUSE's high-level getattr callback fills a fuse_darwin_attr, not a
// struct stat. Compute into a struct stat (== cte_stat_t here) and translate.
static void CopyStatToDarwinAttr(const struct stat &st,
                                 struct fuse_darwin_attr *attr) {
  memset(attr, 0, sizeof(*attr));
  attr->ino = st.st_ino;
  attr->mode = st.st_mode;
  attr->nlink = st.st_nlink;
  attr->uid = st.st_uid;
  attr->gid = st.st_gid;
  attr->rdev = st.st_rdev;
  attr->size = st.st_size;
  attr->blocks = st.st_blocks;
  attr->blksize = st.st_blksize;
  attr->flags = st.st_flags;
  attr->atimespec = st.st_atimespec;
  attr->mtimespec = st.st_mtimespec;
  attr->ctimespec = st.st_ctimespec;
  attr->btimespec = st.st_birthtimespec;
}

static int cte_fuse_getattr(const char *path, struct fuse_darwin_attr *attr,
                            struct fuse_file_info *fi) {
  struct stat stbuf;
  int rc = cte_fuse_getattr_stat(path, &stbuf, fi);
  if (rc != 0) return rc;
  CopyStatToDarwinAttr(stbuf, attr);
  return 0;
}
#else
// Linux (struct stat) and Windows (WinFsp stat via cte_stat_t).
static int cte_fuse_getattr(const char *path, cte_stat_t *stbuf,
                            struct fuse_file_info *fi) {
  return cte_fuse_getattr_stat(path, stbuf, fi);
}
#endif

static int cte_fuse_utimens(const char *path, const cte_timespec_t tv[2],
                            struct fuse_file_info *fi) {
  (void)fi;
  // Translate the POSIX (atime, mtime) timespec pair into the chimod's flag
  // encoding: bit0/bit1 = explicit atime/mtime, bit2/bit3 = UTIME_NOW (resolved
  // server-side so it shares the tag clock). UTIME_OMIT leaves a field alone.
  clio::run::u32 flags = 0;
  clio::run::u64 atime_ns = 0, mtime_ns = 0;
#if defined(UTIME_NOW) && defined(UTIME_OMIT)
  if (tv != nullptr) {
    if (tv[0].tv_nsec == UTIME_NOW) {
      flags |= 0x4u;
    } else if (tv[0].tv_nsec != UTIME_OMIT) {
      flags |= 0x1u;
      // Signed arithmetic so pre-epoch times don't wrap (generic/258); stored
      // as the two's-complement bits, decoded symmetrically in NsBitsToTimespec.
      atime_ns = static_cast<clio::run::u64>(
          static_cast<int64_t>(tv[0].tv_sec) * 1000000000LL +
          static_cast<int64_t>(tv[0].tv_nsec));
    }
    if (tv[1].tv_nsec == UTIME_NOW) {
      flags |= 0x8u;
    } else if (tv[1].tv_nsec != UTIME_OMIT) {
      flags |= 0x2u;
      mtime_ns = static_cast<clio::run::u64>(
          static_cast<int64_t>(tv[1].tv_sec) * 1000000000LL +
          static_cast<int64_t>(tv[1].tv_nsec));
    }
  } else {
    // NULL tv means "set both to now".
    flags |= 0x4u | 0x8u;
  }
#else
  // Platform without UTIME_NOW/OMIT: treat as set-both-to-now.
  (void)tv;
  flags |= 0x4u | 0x8u;
#endif
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncUtimens(std::string(path), atime_ns, mtime_ns, flags);
  t.Wait();
  int rc = static_cast<int>(t->GetReturnCode());
  return rc == 0 ? 0 : -rc;
}

// chmod records the permission bits as a per-file override in the chimod
// (surfaced by getattr), mirroring chown. Storing the mode lets +x stick, so a
// binary copied onto the fs can be executed (generic/452) and chmod-then-proceed
// callers (e.g. mount's mtab updater in generic/089) still succeed. AsyncChmod
// resolves the path and returns ENOENT for a missing file, so no separate
// existence probe is needed.
static int cte_fuse_chmod(const char *path, cte_mode_t mode,
                          struct fuse_file_info *fi) {
  (void)fi;
  auto *cfs = CLIO_CFS_CLIENT;
  std::string p(path);
  // Probe existence first: the chmod path resolves via GetOrCreateTag, so a
  // missing target would otherwise be silently created. chmod(2) must ENOENT.
  auto g = cfs->AsyncGetattr(p);
  g.Wait();
  if (g->GetReturnCode() != 0 || g->exists_ == 0) return -ENOENT;
  auto t = cfs->AsyncChmod(p, static_cast<clio::run::u32>(mode) & 07777u);
  t.Wait();
  return t->GetReturnCode() == 0 ? 0 : -EIO;
}

// chown records a per-file owner uid/gid override in the chimod, surfaced by
// getattr (files carry no stored POSIX owner otherwise). A uid/gid of
// (uid_t)-1 == 0xFFFFFFFF means "leave that field unchanged" (POSIX), which is
// exactly the chimod's "unchanged" sentinel, so no translation is needed.
static int cte_fuse_chown(const char *path, uid_t uid, gid_t gid,
                          struct fuse_file_info *fi) {
  (void)fi;
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncChown(std::string(path),
                           static_cast<clio::run::u32>(uid),
                           static_cast<clio::run::u32>(gid));
  t.Wait();
  return t->GetReturnCode() == 0 ? 0 : -EIO;
}

// ============================================================================
// Directory operations
// ============================================================================

#ifdef __APPLE__
using ClioFuseFillDirT = fuse_darwin_fill_dir_t;
#else
using ClioFuseFillDirT = fuse_fill_dir_t;
#endif

static int cte_fuse_readdir(const char *path, void *buf,
                            ClioFuseFillDirT filler, cte_off_t offset,
                            struct fuse_file_info *fi,
                            enum fuse_readdir_flags flags) {
  (void)offset;
  (void)fi;
  (void)flags;

  std::string p(path);

  filler(buf, ".", nullptr, 0, static_cast<fuse_fill_dir_flags>(0));
  filler(buf, "..", nullptr, 0, static_cast<fuse_fill_dir_flags>(0));

  // Delegate listing to the filesystem chimod. It returns the full tag paths
  // of the directory's children; strip the directory prefix to get basenames.
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncReaddir(p);
  t.Wait();
  if (t->GetReturnCode() != 0) {
    return 0;
  }
  size_t prefix_len = p.size();
  if (!p.empty() && p.back() != '/') prefix_len++;
  // entries_ and inos_ are index-aligned (the chimod builds them together).
  for (size_t i = 0; i < t->entries_.size(); ++i) {
    std::string full = t->entries_[i].str();
    std::string name = full.size() > prefix_len ? full.substr(prefix_len) : full;
    // A child sentinel directory may come back as "<dir>/<name>/"; drop the
    // trailing slash so it shows as a plain directory entry.
    if (!name.empty() && name.back() == '/') name.pop_back();
    if (name.empty()) continue;
    // Supply only d_ino (the child's tag-derived inode) so getdents agrees with
    // a subsequent stat (generic/637). Leave st_mode = 0 (DT_UNKNOWN): the entry
    // type is not reliably known here, so the kernel issues a getattr to resolve
    // it — setting a wrong d_type would mislead `rm -rf`/`find`.
    cte_stat_t st;
    memset(&st, 0, sizeof(st));
    st.st_ino = i < t->inos_.size() ? static_cast<ino_t>(t->inos_[i]) : 0;
    filler(buf, name.c_str(), &st, 0, static_cast<fuse_fill_dir_flags>(0));
  }
  return 0;
}

static int cte_fuse_mkdir(const char *path, cte_mode_t mode) {
  (void)mode;
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncMkdir(std::string(path));
  t.Wait();
  int rc = static_cast<int>(t->GetReturnCode());  // errno-style (0/EEXIST/EIO)
  return rc == 0 ? 0 : -rc;
}

static int cte_fuse_rmdir(const char *path) {
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncRmdir(std::string(path));
  t.Wait();
  int rc = static_cast<int>(t->GetReturnCode());  // 0/ENOTEMPTY/ENOENT/EIO
  return rc == 0 ? 0 : -rc;
}

// ============================================================================
// File lifecycle
// ============================================================================

// O_DIRECT needs no special handling: we deliberately do NOT set the per-file
// direct_io flag. Our read/write handlers already work at arbitrary offsets and
// sizes, so an O_DIRECT open flows through the exact same buffered page-cache
// path as a regular open — it is never rejected. (Enabling per-file direct_io
// would make the kernel return ENODEV for any mmap of an O_DIRECT fd, breaking
// programs that both O_DIRECT and mmap the same file, e.g. fsx -Z; see #597.)

// Honor O_TRUNC: the chimod open resolves the tag but does not truncate, so an
// open/creat of an existing file with O_TRUNC would keep its old page-blobs
// (leaving stale data an app expects to be gone — e.g. reads of a re-created
// file's holes). Clear it to zero length here, which frees those blobs.
static inline void MaybeTruncateOnOpen(clio::cte::filesystem::Client *cfs,
                                       const std::string &p, int flags) {
  if (flags & O_TRUNC) {
    auto tr = cfs->AsyncTruncate(p, 0);
    tr.Wait();
  }
}

static int cte_fuse_create(const char *path, cte_mode_t mode,
                           struct fuse_file_info *fi) {
  std::string p(path);
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncOpen(p, O_CREAT | O_RDWR, static_cast<clio::run::u32>(mode));
  t.Wait();
  if (t->GetReturnCode() != 0) return -EIO;

  auto *handle = new CfsHandle();
  handle->fh = t->handle_;
  handle->path = p;
  fi->fh = reinterpret_cast<uint64_t>(handle);
  MaybeTruncateOnOpen(cfs, p, fi->flags);
  return 0;
}

static int cte_fuse_open(const char *path, struct fuse_file_info *fi) {
  std::string p(path);
  auto *cfs = CLIO_CFS_CLIENT;
  // The chimod honors O_CREAT: a plain open of a missing file returns
  // handle==0 so we can surface ENOENT.
  auto t = cfs->AsyncOpen(p, static_cast<clio::run::u32>(fi->flags), 0644);
  t.Wait();
  if (t->GetReturnCode() != 0) return -EIO;
  if (t->handle_ == 0) return -ENOENT;

  auto *handle = new CfsHandle();
  handle->fh = t->handle_;
  handle->path = p;
  fi->fh = reinterpret_cast<uint64_t>(handle);
  MaybeTruncateOnOpen(cfs, p, fi->flags);
  return 0;
}

// Writes go straight through to the chimod (each AsyncWrite is awaited), so
// there is nothing to drain on flush/fsync — they are durability no-ops.
static int cte_fuse_flush(const char *path, struct fuse_file_info *fi) {
  (void)path;
  (void)fi;
  return 0;
}

static int cte_fuse_fsync(const char *path, int /*datasync*/,
                          struct fuse_file_info *fi) {
  (void)path;
  (void)fi;
  return 0;
}

static int cte_fuse_release(const char *path, struct fuse_file_info *fi) {
  (void)path;
  auto *handle = GetHandle(fi);
  if (!handle) return 0;
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncClose(handle->fh);
  t.Wait();
  int rc = (t->GetReturnCode() == 0) ? 0 : -EIO;
  delete handle;
  fi->fh = 0;
  return rc;
}

// ============================================================================
// Read / Write — delegated to the chimod's page-based I/O
// ============================================================================

static int cte_fuse_read(const char *path, char *buf, size_t size,
                         cte_off_t offset, struct fuse_file_info *fi) {
  (void)path;
  auto *handle = GetHandle(fi);
  if (!handle) return -EBADF;

  if (size > static_cast<size_t>(INT_MAX))
    size = static_cast<size_t>(INT_MAX);
  if (size == 0) return 0;

  auto *ipc = CLIO_IPC;
  ctp::ipc::FullPtr<char> shm_buf = ipc->AllocateBuffer(size);
  if (shm_buf.IsNull()) return -ENOMEM;
  ctp::ipc::ShmPtr<> shm_ptr(shm_buf.shm_);

  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncRead(handle->fh, static_cast<clio::run::u64>(offset), size,
                          shm_ptr);
  t.Wait();
  int rc;
  if (t->GetReturnCode() == 0) {
    size_t got = static_cast<size_t>(t->bytes_read_);
    if (got > 0) memcpy(buf, shm_buf.ptr_, got);
    rc = static_cast<int>(got);
  } else {
    rc = -EIO;
  }
  ipc->FreeBuffer(shm_buf);
  return rc;
}

static int cte_fuse_write(const char *path, const char *buf, size_t size,
                          cte_off_t offset, struct fuse_file_info *fi) {
  (void)path;
  auto *handle = GetHandle(fi);
  if (!handle) return -EBADF;

  if (size > static_cast<size_t>(INT_MAX))
    size = static_cast<size_t>(INT_MAX);
  if (size == 0) return 0;

  auto *ipc = CLIO_IPC;
  ctp::ipc::FullPtr<char> shm_buf = ipc->AllocateBuffer(size);
  if (shm_buf.IsNull()) return -ENOMEM;
  memcpy(shm_buf.ptr_, buf, size);
  ctp::ipc::ShmPtr<> shm_ptr(shm_buf.shm_);

  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncWrite(handle->fh, static_cast<clio::run::u64>(offset), size,
                           shm_ptr);
  t.Wait();
  int rc;
  if (t->GetReturnCode() == 0) {
    rc = static_cast<int>(t->bytes_written_);
  } else {
    rc = -EIO;
  }
  ipc->FreeBuffer(shm_buf);
  return rc;
}

// ============================================================================
// Unlink / Truncate
// ============================================================================

static int cte_fuse_unlink(const char *path) {
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncUnlink(std::string(path));
  t.Wait();
  int rc = static_cast<int>(t->GetReturnCode());  // 0/EISDIR/EIO
  return rc == 0 ? 0 : -rc;
}

static int cte_fuse_truncate(const char *path, cte_off_t size,
                             struct fuse_file_info *fi) {
  (void)fi;
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncTruncate(std::string(path), static_cast<clio::run::u64>(size));
  t.Wait();
  return t->GetReturnCode() == 0 ? 0 : -EIO;
}

#ifdef __linux__
// Write `len` zero bytes at `off` through an open handle, chunked so a large
// range doesn't need one giant SHM buffer. Used by ZERO_RANGE. Returns 0 or a
// negative errno.
static int cte_fuse_write_zeros(CfsHandle *handle, cte_off_t off,
                                cte_off_t len) {
  auto *ipc = CLIO_IPC;
  auto *cfs = CLIO_CFS_CLIENT;
  constexpr cte_off_t kChunk = 1 << 20;  // 1 MiB
  const cte_off_t buf_sz = (len < kChunk) ? len : kChunk;
  ctp::ipc::FullPtr<char> zbuf = ipc->AllocateBuffer(buf_sz);
  if (zbuf.IsNull()) return -ENOMEM;
  memset(zbuf.ptr_, 0, static_cast<size_t>(buf_sz));
  int result = 0;
  for (cte_off_t done = 0; done < len;) {
    const cte_off_t n = ((len - done) < buf_sz) ? (len - done) : buf_sz;
    auto t = cfs->AsyncWrite(handle->fh, static_cast<clio::run::u64>(off + done),
                             static_cast<clio::run::u64>(n),
                             ctp::ipc::ShmPtr<>(zbuf.shm_));
    t.Wait();
    if (t->GetReturnCode() != 0) { result = -EIO; break; }
    done += n;
  }
  ipc->FreeBuffer(zbuf);
  return result;
}

// fallocate — page-blobs are created lazily on write and holes read back as
// zeros, so we never reserve storage ahead of time. Supported modes:
//   * mode==0            : grow EOF to offset+length (never shrinks).
//   * FALLOC_FL_KEEP_SIZE: no-op success (nothing to reserve).
//   * FALLOC_FL_ZERO_RANGE: make [offset,offset+length) read as zeros by
//     writing zeros through the open handle (with KEEP_SIZE, restore EOF after
//     if the write extended it). This is a correct ZERO_RANGE for our
//     hole-reads-as-zero model and unblocks the fzero xfstests.
// Layout-shifting modes (punch hole, collapse/insert range) would need a
// chimod-level block-dealloc/shift op and still return EOPNOTSUPP.
static int cte_fuse_fallocate(const char *path, int mode, cte_off_t offset,
                              cte_off_t length, struct fuse_file_info *fi) {
  const int kSupportedModes = FALLOC_FL_KEEP_SIZE | FALLOC_FL_ZERO_RANGE;
  if (mode & ~kSupportedModes) {
    return -EOPNOTSUPP;  // punch/collapse/insert: layout-changing
  }
  if (offset < 0 || length <= 0) {
    return -EINVAL;
  }

  if (mode & FALLOC_FL_ZERO_RANGE) {
    auto *handle = GetHandle(fi);
    if (!handle) return -EBADF;
    // Original size, so KEEP_SIZE can restore EOF if the zero-write grows it.
    cte_stat_t st;
    int rc = cte_fuse_getattr_stat(path, &st, fi);
    if (rc != 0) return rc;
    rc = cte_fuse_write_zeros(handle, offset, length);
    if (rc != 0) return rc;
    if ((mode & FALLOC_FL_KEEP_SIZE) && (offset + length) > st.st_size) {
      auto *cfs = CLIO_CFS_CLIENT;
      auto t = cfs->AsyncTruncate(std::string(path),
                                  static_cast<clio::run::u64>(st.st_size));
      t.Wait();
      if (t->GetReturnCode() != 0) return -EIO;
    }
    return 0;
  }

  if (mode & FALLOC_FL_KEEP_SIZE) {
    return 0;  // no size change requested, and there is nothing to reserve
  }

  // mode == 0: extend EOF to offset+length if the file is currently shorter.
  cte_stat_t st;
  int rc = cte_fuse_getattr_stat(path, &st, fi);
  if (rc != 0) return rc;
  cte_off_t need = offset + length;
  if (need <= st.st_size) return 0;  // already large enough; never shrink

  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncTruncate(std::string(path),
                              static_cast<clio::run::u64>(need));
  t.Wait();
  return t->GetReturnCode() == 0 ? 0 : -EIO;
}
#endif  // __linux__

static int cte_fuse_link(const char *from, const char *to) {
  // Hard link `to` -> existing file `from`. The chimod binds both names to the
  // same CTE tag (a tag-level alias), so they share all data.
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncLink(std::string(from), std::string(to));
  t.Wait();
  int rc = static_cast<int>(t->GetReturnCode());
  return rc == 0 ? 0 : -rc;  // chimod returns errno-style codes
}

static int cte_fuse_symlink(const char *target, const char *path) {
  // Create a symlink at `path` pointing at `target`. The chimod stores the
  // target string in a reserved marker blob under `path`'s tag.
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncSymlink(std::string(target), std::string(path));
  t.Wait();
  int rc = static_cast<int>(t->GetReturnCode());  // 0/EEXIST/EIO
  return rc == 0 ? 0 : -rc;  // chimod returns errno-style codes
}

static int cte_fuse_readlink(const char *path, char *buf, size_t size) {
  // Read the symlink target into `buf` (NUL-terminated). FUSE readlink returns
  // 0 on success (not the length).
  if (size == 0) {
    return -EINVAL;
  }
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncReadlink(std::string(path));
  t.Wait();
  int rc = static_cast<int>(t->GetReturnCode());  // 0/ENOENT/EINVAL
  if (rc != 0) {
    return -rc;
  }
  std::string target = t->target_.str();
  size_t n = std::min(target.size(), size - 1);
  std::memcpy(buf, target.data(), n);
  buf[n] = '\0';
  return 0;
}

static int cte_fuse_setxattr(const char *path, const char *name,
                             const char *value, size_t size, int flags) {
  // Set xattr `name` on `path`. `value` is raw bytes (may contain NULs), so
  // preserve its length rather than treating it as a C string. `flags` carries
  // XATTR_CREATE(1) / XATTR_REPLACE(2), matching the runtime's bit checks.
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncSetxattr(std::string(path), std::string(name),
                              std::string(value, size),
                              static_cast<unsigned int>(flags));
  t.Wait();
  int rc = static_cast<int>(t->GetReturnCode());
  return rc == 0 ? 0 : -rc;  // EEXIST/ENODATA/ENOENT/EIO -> negative errno
}

static int cte_fuse_getxattr(const char *path, const char *name, char *value,
                             size_t size) {
  // Read xattr `name` of `path`. Return the value length (POSIX getxattr);
  // size==0 is a length query. Missing attribute -> -ENODATA.
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncGetxattr(std::string(path), std::string(name));
  t.Wait();
  int rc = static_cast<int>(t->GetReturnCode());
  if (rc != 0) {
    return -rc;  // ENOENT (file absent)
  }
  if (t->found_ == 0) {
    return -ENODATA;  // attribute not present
  }
  std::string val = t->value_.str();
  size_t len = val.size();
  if (size == 0) {
    return static_cast<int>(len);  // length query
  }
  if (size < len) {
    return -ERANGE;
  }
  std::memcpy(value, val.data(), len);
  return static_cast<int>(len);
}

static int cte_fuse_listxattr(const char *path, char *list, size_t size) {
  // Return the NUL-separated, NUL-terminated list of xattr names. size==0 is a
  // length query.
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncListxattr(std::string(path));
  t.Wait();
  int rc = static_cast<int>(t->GetReturnCode());
  if (rc != 0) {
    return -rc;  // ENOENT
  }
  std::string names = t->names_.str();
  size_t len = names.size();
  if (size == 0) {
    return static_cast<int>(len);  // length query
  }
  if (size < len) {
    return -ERANGE;
  }
  std::memcpy(list, names.data(), len);
  return static_cast<int>(len);
}

static int cte_fuse_removexattr(const char *path, const char *name) {
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncRemovexattr(std::string(path), std::string(name));
  t.Wait();
  int rc = static_cast<int>(t->GetReturnCode());
  return rc == 0 ? 0 : -rc;  // ENODATA/ENOENT/EIO -> negative errno
}

#ifndef RENAME_NOREPLACE
#define RENAME_NOREPLACE (1 << 0)  // from <linux/fs.h>; guarded to avoid header clash
#endif

static int cte_fuse_rename(const char *from, const char *to,
                           unsigned int flags) {
  auto *cfs = CLIO_CFS_CLIENT;
  // RENAME_NOREPLACE: the rename must fail with EEXIST if `to` already exists.
  // Probe for the destination then fall through to a plain rename. This is the
  // standard high-level-FUSE approach (a tiny TOCTOU window vs a truly atomic
  // check, acceptable for a single-namespace rename). RENAME_EXCHANGE and
  // RENAME_WHITEOUT need chimod-level atomic swap / whiteout support and stay
  // EINVAL so callers fall back cleanly.
  if (flags & RENAME_NOREPLACE) {
    auto g = cfs->AsyncGetattr(std::string(to));
    g.Wait();
    if (g->GetReturnCode() == 0 && g->exists_ != 0) return -EEXIST;
    flags &= ~static_cast<unsigned int>(RENAME_NOREPLACE);
  }
  if (flags != 0) {
    return -EINVAL;  // RENAME_EXCHANGE / RENAME_WHITEOUT unsupported
  }
  auto t = cfs->AsyncRename(std::string(from), std::string(to));
  t.Wait();
  int rc = static_cast<int>(t->GetReturnCode());
  return rc == 0 ? 0 : -rc;  // chimod returns errno-style codes (ENOENT/EIO)
}

// ============================================================================
// Main
// ============================================================================

// Report filesystem statistics. Total and remaining capacity are the real
// cluster-wide values, obtained from the CTE: GetCapacity sums the registered
// targets' total and remaining capacity per node, and a Broadcast aggregates
// that across the cluster (the task's AggregateOut sums per-node results).
// Reporting a non-zero capacity also matters operationally: a 0-block fs is
// hidden by `df` (which lists no path), which breaks tools that probe free
// space and xfstests' mount detection.
static int cte_fuse_statfs(const char *path, cte_statvfs_t *stbuf) {
  (void)path;
  std::memset(stbuf, 0, sizeof(*stbuf));
  constexpr fsblkcnt_t kBlockSize = 4096;

  clio::run::u64 total_bytes = 0;
  clio::run::u64 remaining_bytes = 0;
  auto *cte = CLIO_CTE_CLIENT;
  if (cte != nullptr) {
    // Broadcast: total + remaining capacity across the whole cluster.
    auto t = cte->AsyncGetCapacity(clio::run::PoolQuery::Broadcast());
    t.Wait();
    if (t->return_code_ == 0) {
      total_bytes = t->total_capacity_;
      remaining_bytes = t->remaining_capacity_;
    }
  }
  fsblkcnt_t total_blocks = total_bytes / kBlockSize;
  fsblkcnt_t free_blocks = remaining_bytes / kBlockSize;

  stbuf->f_bsize = kBlockSize;
  stbuf->f_frsize = kBlockSize;
  stbuf->f_blocks = total_blocks;
  // Report real remaining space as both free and available (no reservation
  // distinction), so df shows used = total - remaining.
  stbuf->f_bfree = free_blocks;
  stbuf->f_bavail = free_blocks;
  stbuf->f_files = static_cast<fsfilcnt_t>(1) << 20;
  stbuf->f_ffree = static_cast<fsfilcnt_t>(1) << 20;
  stbuf->f_favail = static_cast<fsfilcnt_t>(1) << 20;
  stbuf->f_namemax = 255;
  return 0;
}

// External linkage (declared in fuse_cte.h): the mount/process-entry glue in
// fuse_cte_main.cc points fuse_main()/fuse_new() at this table. The callbacks
// it references keep internal linkage — their addresses are captured here.
const struct fuse_operations cte_fuse_ops = {
    .getattr = cte_fuse_getattr,
    .readlink = cte_fuse_readlink,
    .mkdir = cte_fuse_mkdir,
    .unlink = cte_fuse_unlink,
    .rmdir = cte_fuse_rmdir,
    .symlink = cte_fuse_symlink,
    .rename = cte_fuse_rename,
    .link = cte_fuse_link,
    .chmod = cte_fuse_chmod,
    .chown = cte_fuse_chown,
    .truncate = cte_fuse_truncate,
    .open = cte_fuse_open,
    .read = cte_fuse_read,
    .write = cte_fuse_write,
    .statfs = cte_fuse_statfs,
    .flush = cte_fuse_flush,
    .release = cte_fuse_release,
    .fsync = cte_fuse_fsync,
    .setxattr = cte_fuse_setxattr,
    .getxattr = cte_fuse_getxattr,
    .listxattr = cte_fuse_listxattr,
    .removexattr = cte_fuse_removexattr,
    .readdir = cte_fuse_readdir,
    .init = cte_fuse_init,
    .destroy = cte_fuse_destroy,
    .create = cte_fuse_create,
    .utimens = cte_fuse_utimens,
#ifdef __linux__
    .fallocate = cte_fuse_fallocate,
#endif
};
