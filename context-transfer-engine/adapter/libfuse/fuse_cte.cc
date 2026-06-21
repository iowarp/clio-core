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

#include <climits>
#include <cstring>
#include <string>
#include <vector>

#include <fuse3/fuse_lowlevel.h>  // fuse_session_custom_io, struct fuse_custom_io
#include <dlfcn.h>                // dlsym (resolve fuse_session_custom_io at runtime
                                  // so we can link against system libfuse 3.10.5
                                  // which lacks the symbol; only used in the
                                  // apptainer --fusemount fd-injection path)
#include <sys/uio.h>              // struct iovec, writev
#include <sys/mount.h>            // mount syscall
#include <sys/statvfs.h>          // struct statvfs (statfs op)
#include <unistd.h>               // read, getuid, getgid
#include <cerrno>                 // errno
#include <cstdio>                 // snprintf, fprintf
#include <fcntl.h>                // O_CREAT, O_RDWR

#include "clio_runtime/clio_runtime.h"
#include "clio_cte/core/content_transfer_engine.h"
#include "clio_cte/core/core_client.h"  // CLIO_CTE_CLIENT + GetCapacity
#include "clio_cte/filesystem/filesystem_client.h"

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
  chi::u64 fh = 0;
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

  bool success = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
  if (!success) {
    fprintf(stderr, "ERROR: CHIMAERA_INIT failed\n");
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
  chi::CHIMAERA_FINALIZE();
}

// ============================================================================
// Metadata
// ============================================================================

static int cte_fuse_getattr(const char *path, struct stat *stbuf,
                            struct fuse_file_info *fi) {
  (void)fi;
  memset(stbuf, 0, sizeof(struct stat));

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
  stbuf->st_uid = getuid();
  stbuf->st_gid = getgid();
  stbuf->st_ino = static_cast<ino_t>(t->ino_);  // stable inode = packed TagId
  if (t->is_dir_) {
    stbuf->st_mode = S_IFDIR | 0755;
    stbuf->st_nlink = 2;
  } else {
    stbuf->st_mode = S_IFREG | 0644;
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
    stbuf->st_size = static_cast<off_t>(t->size_);
  }
  return 0;
}

static int cte_fuse_utimens(const char *path, const struct timespec tv[2],
                            struct fuse_file_info *fi) {
  (void)path;
  (void)tv;
  (void)fi;
  // CTE timestamps are managed internally; accept silently
  return 0;
}

// ============================================================================
// Directory operations
// ============================================================================

static int cte_fuse_readdir(const char *path, void *buf,
                            fuse_fill_dir_t filler, off_t offset,
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
    struct stat st;
    memset(&st, 0, sizeof(st));
    st.st_ino = i < t->inos_.size() ? static_cast<ino_t>(t->inos_[i]) : 0;
    filler(buf, name.c_str(), &st, 0, static_cast<fuse_fill_dir_flags>(0));
  }
  return 0;
}

static int cte_fuse_mkdir(const char *path, mode_t mode) {
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

static int cte_fuse_create(const char *path, mode_t mode,
                           struct fuse_file_info *fi) {
  std::string p(path);
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncOpen(p, O_CREAT | O_RDWR, static_cast<chi::u32>(mode));
  t.Wait();
  if (t->GetReturnCode() != 0) return -EIO;

  auto *handle = new CfsHandle();
  handle->fh = t->handle_;
  handle->path = p;
  fi->fh = reinterpret_cast<uint64_t>(handle);
  return 0;
}

static int cte_fuse_open(const char *path, struct fuse_file_info *fi) {
  std::string p(path);
  auto *cfs = CLIO_CFS_CLIENT;
  // The chimod honors O_CREAT: a plain open of a missing file returns
  // handle==0 so we can surface ENOENT.
  auto t = cfs->AsyncOpen(p, static_cast<chi::u32>(fi->flags), 0644);
  t.Wait();
  if (t->GetReturnCode() != 0) return -EIO;
  if (t->handle_ == 0) return -ENOENT;

  auto *handle = new CfsHandle();
  handle->fh = t->handle_;
  handle->path = p;
  fi->fh = reinterpret_cast<uint64_t>(handle);
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
                         off_t offset, struct fuse_file_info *fi) {
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
  auto t = cfs->AsyncRead(handle->fh, static_cast<chi::u64>(offset), size,
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
                          off_t offset, struct fuse_file_info *fi) {
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
  auto t = cfs->AsyncWrite(handle->fh, static_cast<chi::u64>(offset), size,
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

static int cte_fuse_truncate(const char *path, off_t size,
                             struct fuse_file_info *fi) {
  (void)fi;
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncTruncate(std::string(path), static_cast<chi::u64>(size));
  t.Wait();
  return t->GetReturnCode() == 0 ? 0 : -EIO;
}

static int cte_fuse_link(const char *from, const char *to) {
  // Hard link `to` -> existing file `from`. The chimod binds both names to the
  // same CTE tag (a tag-level alias), so they share all data.
  auto *cfs = CLIO_CFS_CLIENT;
  auto t = cfs->AsyncLink(std::string(from), std::string(to));
  t.Wait();
  int rc = static_cast<int>(t->GetReturnCode());
  return rc == 0 ? 0 : -rc;  // chimod returns errno-style codes
}

static int cte_fuse_rename(const char *from, const char *to,
                           unsigned int flags) {
  // RENAME_NOREPLACE / RENAME_EXCHANGE aren't supported; POSIX replace is.
  if (flags != 0) {
    return -EINVAL;
  }
  auto *cfs = CLIO_CFS_CLIENT;
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
static int cte_fuse_statfs(const char *path, struct statvfs *stbuf) {
  (void)path;
  std::memset(stbuf, 0, sizeof(*stbuf));
  constexpr fsblkcnt_t kBlockSize = 4096;

  chi::u64 total_bytes = 0;
  chi::u64 remaining_bytes = 0;
  auto *cte = CLIO_CTE_CLIENT;
  if (cte != nullptr) {
    // Broadcast: total + remaining capacity across the whole cluster.
    auto t = cte->AsyncGetCapacity(chi::PoolQuery::Broadcast());
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

static const struct fuse_operations cte_fuse_ops = {
    .getattr = cte_fuse_getattr,
    .mkdir = cte_fuse_mkdir,
    .unlink = cte_fuse_unlink,
    .rmdir = cte_fuse_rmdir,
    .rename = cte_fuse_rename,
    .link = cte_fuse_link,
    .truncate = cte_fuse_truncate,
    .open = cte_fuse_open,
    .read = cte_fuse_read,
    .write = cte_fuse_write,
    .statfs = cte_fuse_statfs,
    .flush = cte_fuse_flush,
    .release = cte_fuse_release,
    .fsync = cte_fuse_fsync,
    .readdir = cte_fuse_readdir,
    .init = cte_fuse_init,
    .destroy = cte_fuse_destroy,
    .create = cte_fuse_create,
    .utimens = cte_fuse_utimens,
};

// custom_io callbacks: when apptainer hands us an already-mounted
// /dev/fuse fd, we need to drive the FUSE protocol on that fd directly
// instead of having libfuse mount its own. Plain read/writev syscalls
// suffice; splice is optional.
static ssize_t cte_custom_writev(int fd, struct iovec *iov, int count,
                                 void * /*userdata*/) {
  return writev(fd, iov, count);
}
static ssize_t cte_custom_read(int fd, void *buf, size_t buf_len,
                               void * /*userdata*/) {
  return read(fd, buf, buf_len);
}

int main(int argc, char *argv[]) {
  // Apptainer's --fusemount opens /dev/fuse on the host, performs the
  // kernel mount, and passes the fd to the FUSE binary as the last
  // argv ("/dev/fd/<N>"). libfuse 3's high-level argv parser doesn't
  // recognize this token (it's a libfuse2-era convention) and aborts
  // with "fuse: invalid argument '/dev/fd/N'". apptainer also strips
  // the mountpoint from the argv since it has already mounted, so
  // there's no fuse_main path that works.
  //
  // libfuse 3.14 added fuse_session_custom_io() which lets us drive
  // the protocol on an existing fd. When we detect /dev/fd/<N> as the
  // last argv, take the custom-io path; otherwise fall back to the
  // normal fuse_main mount-and-serve flow (host bare-metal use).
  int prefd = -1;
  int new_argc = argc;
  if (argc >= 2 && std::strncmp(argv[argc - 1], "/dev/fd/", 8) == 0) {
    prefd = std::atoi(argv[argc - 1] + 8);
    if (prefd < 0) {
      prefd = -1;
    } else {
      new_argc = argc - 1;  // drop /dev/fd/N from argv
    }
  }

  if (prefd == -1) {
    return fuse_main(argc, argv, &cte_fuse_ops, nullptr);
  }

  // Apptainer 1.2.5 (unprivileged, no starter-suid) doesn't actually
  // call mount(2) when --fusemount kicks in for a user-supplied
  // binary — it only opens /dev/fuse and hands us the fd. We have to
  // bind that fd to a mountpoint ourselves (this requires CAP_SYS_ADMIN
  // in the current user_ns, which we have via apptainer's userns
  // mapping). The mountpoint isn't communicated to us through argv or
  // env by apptainer, so the caller MUST set CLIO_CTE_FUSE_MOUNTPOINT
  // before exec'ing the FUSE binary via --fusemount.
  const char *mountpoint = std::getenv("CLIO_CTE_FUSE_MOUNTPOINT");
  if (mountpoint == nullptr) {
    std::fprintf(stderr,
                 "clio_cte_fuse: got pre-opened fd %d but "
                 "CLIO_CTE_FUSE_MOUNTPOINT env var is not set\n", prefd);
    return 1;
  }
  char mount_opts[256];
  std::snprintf(mount_opts, sizeof(mount_opts),
                "fd=%d,rootmode=040000,user_id=%u,group_id=%u",
                prefd, (unsigned)getuid(), (unsigned)getgid());
  if (mount("nodev", mountpoint, "fuse", MS_NODEV | MS_NOSUID,
            mount_opts) != 0) {
    std::fprintf(stderr, "clio_cte_fuse: mount(\"%s\", fuse) failed: %s\n",
                 mountpoint, std::strerror(errno));
    return 1;
  }
  std::fprintf(stderr, "clio_cte_fuse: mounted FUSE at %s with fd=%d\n",
               mountpoint, prefd);

  struct fuse_args args = FUSE_ARGS_INIT(new_argc, argv);
  struct fuse *fuse =
      fuse_new(&args, &cte_fuse_ops, sizeof(cte_fuse_ops), nullptr);
  if (!fuse) {
    fuse_opt_free_args(&args);
    return 1;
  }

  struct fuse_session *se = fuse_get_session(fuse);
  static const struct fuse_custom_io custom_io = {
      .writev = cte_custom_writev,
      .read = cte_custom_read,
      .splice_receive = nullptr,
      .splice_send = nullptr,
  };

  // Resolve fuse_session_custom_io at runtime via dlsym so this binary
  // links against system libfuse 3.10.5 (which has setuid
  // /usr/bin/fusermount3) without needing the 3.14+ symbol present at
  // link time. The symbol IS present at runtime when the caller uses
  // a newer libfuse runtime (e.g. apptainer --fusemount).
  using FuseSessionCustomIoFn = int (*)(struct fuse_session *,
                                        const struct fuse_custom_io *, int);
  auto fuse_session_custom_io_dyn = reinterpret_cast<FuseSessionCustomIoFn>(
      dlsym(RTLD_DEFAULT, "fuse_session_custom_io"));
  if (!fuse_session_custom_io_dyn) {
    std::fprintf(stderr,
                 "clio_cte_fuse: fuse_session_custom_io not available in "
                 "the loaded libfuse (need 3.14+); --fusemount mode "
                 "requires a newer libfuse runtime.\n");
    fuse_destroy(fuse);
    fuse_opt_free_args(&args);
    return 1;
  }
  if (fuse_session_custom_io_dyn(se, &custom_io, prefd) != 0) {
    fuse_destroy(fuse);
    fuse_opt_free_args(&args);
    return 1;
  }

  // Multi-threaded FUSE loop. Single-threaded `fuse_loop()` serializes
  // every fs op through one handler -- each call blocks on
  // AsyncPutBlob.Wait() before the next can run -- so 24 concurrent
  // ranks per node effectively run sequentially and a workload that
  // should finish in seconds hangs past the test timeout. fuse_loop_mt
  // spawns worker threads that handle ops in parallel; AsyncPutBlobs
  // from different ranks now overlap.
  //
  // The libfuse 3.16 headers we compile against rewrite `fuse_loop_mt`
  // to `fuse_loop_mt_32` (via macro), but the runtime libfuse 3.10.5
  // (system) only exports `fuse_loop_mt@@FUSE_3.2` (the unversioned
  // default), not `fuse_loop_mt_32`. Resolve the unversioned symbol
  // via dlsym -- same pattern this file already uses for
  // fuse_session_custom_io -- so the binary works against any
  // libfuse runtime >= 3.2.
  using FuseLoopMtFn = int (*)(struct fuse *, struct fuse_loop_config *);
  auto fuse_loop_mt_dyn = reinterpret_cast<FuseLoopMtFn>(
      dlsym(RTLD_DEFAULT, "fuse_loop_mt"));

  int ret;
  if (fuse_loop_mt_dyn != nullptr) {
    struct fuse_loop_config loop_cfg = {0};
    loop_cfg.clone_fd = 0;            // share /dev/fuse fd across workers
    loop_cfg.max_idle_threads = 32;   // headroom for 24 ranks/node
    ret = fuse_loop_mt_dyn(fuse, &loop_cfg);
  } else {
    std::fprintf(stderr,
                 "clio_cte_fuse: fuse_loop_mt not in runtime libfuse, "
                 "falling back to single-threaded loop.\n");
    ret = fuse_loop(fuse);
  }
  fuse_destroy(fuse);
  fuse_opt_free_args(&args);
  return ret;
}
