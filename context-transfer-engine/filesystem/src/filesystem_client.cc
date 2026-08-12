/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */
#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/filesystem/filesystem_client.h>

#include <mutex>
#include <string>
#include <vector>

namespace clio::cte::filesystem {

// Process-wide filesystem client singleton (defined inside the namespace so it
// is clio::cte::filesystem::g_fs_client, matching the CLIO_CFS_CLIENT macro).
CLIO_CTE_FS_DEFINE_GLOBAL_PTR_VAR_CC(clio::cte::filesystem::Client, g_fs_client);

/**
 * Create-or-bind the default filesystem pool over the default CTE core pool
 * and publish the process-wide client. Mirrors
 * clio::cte::core::CLIO_CTE_CLIENT_INIT / ContentTransferEngine::ClientInit:
 * GetOrCreatePool is idempotent, so this both creates the pool on first call
 * and binds to the existing one if a launcher already composed it.
 */
bool CLIO_CFS_CLIENT_INIT(const std::string &config_path,
                          const clio::run::PoolQuery &pool_query) {
  static bool s_initialized = false;
  if (s_initialized) {
    return true;
  }
  (void)config_path;  // configuration now flows through clio compose

  // The filesystem chimod sits over the default CTE core pool, so make sure
  // that exists first (also brings up the runtime client / IPC).
  if (!clio::cte::core::CLIO_CTE_CLIENT_INIT()) {
    HLOG(kError, "CFS ClientInit: failed to initialize the CTE core pool");
    return false;
  }

  // CLIO_CFS_CLIENT lazily allocates the global Client on first access.
  auto *fs_client = CLIO_CFS_CLIENT;
  if (fs_client == nullptr) {
    return false;
  }
  fs_client->Init(kCfsPoolId);

  FilesystemConfig params;
  params.next_pool_id_ = clio::cte::core::kCtePoolId;
  auto create_task =
      fs_client->AsyncCreate(pool_query, kCfsPoolName, kCfsPoolId, params);
  create_task.Wait();
  if (create_task->GetReturnCode() != 0) {
    HLOG(kError, "CFS ClientInit: failed to create filesystem pool '{}' (rc={})",
         kCfsPoolName, create_task->GetReturnCode());
    return false;
  }
  fs_client->pool_id_ = create_task->new_pool_id_;

  // issue #817: attach the filesystem attribute cache. Must run AFTER pool_id_
  // is set -- the directory is keyed by pool. Failure is not an error: it just
  // means every path lookup keeps going through the runtime.
  if (!fs_client->AttachShmCache()) {
    HLOG(kDebug,
         "CFS ClientInit: SHM attribute cache unavailable; using RPC");
  }

  s_initialized = true;
  return true;
}

#if !defined(_WIN32)
// Descriptor layer -- POSIX only, see filesystem_client.h.

/**
 * Bind the filesystem pool on first tracked use.
 *
 * Separate from the constructor so a process that never opens a clio:: path
 * never creates a pool -- every interceptor links this client, including ones
 * loaded into programs that do no clio I/O at all.
 */
bool Client::EnsureInit() {
  static bool ready = false;
  if (ready) {
    return true;
  }
  ready = CLIO_CFS_CLIENT_INIT();
  if (!ready) {
    HLOG(kError, "clio-fs: failed to initialize the filesystem client");
  }
  return ready;
}

int Client::OpenFd(const std::string &raw_path, int flags, int mode) {
  if (!EnsureInit()) {
    errno = EIO;
    return -1;
  }
  std::string path = StripClioPrefix(raw_path);
  auto t = AsyncOpen(path, static_cast<clio::run::u32>(flags),
                          static_cast<clio::run::u32>(mode));
  t.Wait();
  if (t->GetReturnCode() != 0) {
    errno = EIO;
    return -1;
  }
  if (t->handle_ == 0) {
    // Plain open of a missing file (chimod honors O_CREAT).
    errno = ENOENT;
    return -1;
  }
  clio::run::u64 size = t->size_;
  // O_TRUNC: drop the logical size to zero.
  if (flags & O_TRUNC) {
    auto tr = AsyncTruncate(path, 0);
    tr.Wait();
    size = 0;
  }
  std::lock_guard<std::mutex> g(fd_mu_);
  int fd = next_fd_++;
  OpenFile of;
  of.handle = t->handle_;
  of.path = path;
  of.flags = flags;
  of.off = (flags & O_APPEND) ? size : 0;
  fds_[fd] = of;
  return fd;
}

ssize_t Client::ReadFd(int fd, void *buf, size_t count) {
  OpenFile of;
  if (!LookupFd(fd, &of)) {
    return -1;
  }
  ssize_t n = Read(of.handle, of.path, of.off, buf, count);
  if (n > 0) {
    AdvanceFd(fd, static_cast<clio::run::u64>(n));
  }
  return n;
}

ssize_t Client::WriteFd(int fd, const void *buf, size_t count) {
  OpenFile of;
  if (!LookupFd(fd, &of)) {
    return -1;
  }
  ssize_t n = Write(of.handle, of.path, of.off, buf, count,
                                     IsSyncFd(of.flags));
  if (n > 0) {
    AdvanceFd(fd, static_cast<clio::run::u64>(n));
  }
  return n;
}

ssize_t Client::PreadFd(int fd, void *buf, size_t count, off_t offset) {
  OpenFile of;
  if (!LookupFd(fd, &of)) {
    return -1;
  }
  return Read(of.handle, of.path,
                               static_cast<clio::run::u64>(offset), buf, count);
}

ssize_t Client::PwriteFd(int fd, const void *buf, size_t count, off_t offset) {
  OpenFile of;
  if (!LookupFd(fd, &of)) {
    return -1;
  }
  return Write(of.handle, of.path,
                                static_cast<clio::run::u64>(offset), buf, count,
                                IsSyncFd(of.flags));
}

off_t Client::SeekFd(int fd, off_t offset, int whence) {
  OpenFile of;
  if (!LookupFd(fd, &of)) {
    return -1;
  }
  clio::run::u64 base = 0;
  switch (whence) {
    case SEEK_SET:
      base = 0;
      break;
    case SEEK_CUR:
      base = of.off;
      break;
    case SEEK_END: {
      // GetSize drains this file's deferred writes first, so EOF includes
      // them (issue #817).
      clio::run::u64 size = 0;
      if (!GetSize(of.path, &size)) {
        errno = EIO;
        return -1;
      }
      base = size;
      break;
    }
    default:
      errno = EINVAL;
      return -1;
  }
  off_t newoff = static_cast<off_t>(base) + offset;
  if (newoff < 0) {
    errno = EINVAL;
    return -1;
  }
  std::lock_guard<std::mutex> g(fd_mu_);
  auto it = fds_.find(fd);
  if (it == fds_.end()) {
    errno = EBADF;
    return -1;
  }
  it->second.off = static_cast<clio::run::u64>(newoff);
  return newoff;
}

off_t Client::TellFd(int fd) {
  OpenFile of;
  if (!LookupFd(fd, &of)) {
    return -1;
  }
  return static_cast<off_t>(of.off);
}

off_t Client::SizeFd(int fd) {
  OpenFile of;
  if (!LookupFd(fd, &of)) {
    return -1;
  }
  clio::run::u64 size = 0;
  if (!GetSize(of.path, &size)) {
    return -1;
  }
  return static_cast<off_t>(size);
}

int Client::SyncFd(int fd) {
  OpenFile of;
  if (!LookupFd(fd, &of)) {
    return -1;
  }
  // Wait for every deferred write on this file and report a latched failure
  // exactly once — fsync and close are the only two places a deferred write's
  // failure can reach the application.
  return Flush(of.path);
}

int Client::FtruncateFd(int fd, off_t length) {
  OpenFile of;
  if (!LookupFd(fd, &of)) {
    return -1;
  }
  return TruncatePath(std::string(kClioPrefix) + of.path, length);
}

int Client::TruncatePath(const std::string &raw_path, off_t length) {
  if (!EnsureInit()) {
    errno = EIO;
    return -1;
  }
  std::string path = StripClioPrefix(raw_path);
  // Order matters: a deferred write that landed AFTER the truncate would undo
  // it, so drain before resizing (issue #817).
  Flush(path);
  auto t = AsyncTruncate(path, static_cast<clio::run::u64>(length));
  t.Wait();
  if (t->GetReturnCode() != 0) {
    errno = EIO;
    return -1;
  }
  return 0;
}

int Client::RemovePath(const std::string &raw_path) {
  if (!EnsureInit()) {
    errno = EIO;
    return -1;
  }
  std::string path = StripClioPrefix(raw_path);
  // Drain first: a write still deferred against a deleted path would
  // resurrect the tag.
  Flush(path);
  auto t = AsyncUnlink(path);
  t.Wait();
  if (t->GetReturnCode() != 0) {
    errno = EIO;
    return -1;
  }
  return 0;
}

int Client::RenamePath(const std::string &raw_src, const std::string &raw_dst) {
  if (!EnsureInit()) {
    errno = EIO;
    return -1;
  }
  std::string src = StripClioPrefix(raw_src);
  std::string dst = StripClioPrefix(raw_dst);
  // Both sides drain: a deferred write to either path must land before the
  // namespace moves, or it would be applied to a name that no longer means
  // the same file.
  Flush(src);
  Flush(dst);
  auto t = AsyncRename(src, dst);
  t.Wait();
  if (t->GetReturnCode() != 0) {
    errno = EIO;
    return -1;
  }
  return 0;
}

int Client::ReaddirPath(const std::string &raw_path, std::vector<std::string> *out) {
  if (!EnsureInit()) {
    errno = EIO;
    return -1;
  }
  std::string path = StripClioPrefix(raw_path);
  auto t = AsyncReaddir(path);
  t.Wait();
  if (t->GetReturnCode() != 0) {
    errno = ENOENT;
    return -1;
  }
  out->clear();
  out->reserve(t->entries_.size());
  for (const auto &e : t->entries_) {
    out->emplace_back(e.str());
  }
  return 0;
}

int Client::CloseFd(int fd) {
  clio::run::u64 handle;
  std::string path;
  {
    std::lock_guard<std::mutex> g(fd_mu_);
    auto it = fds_.find(fd);
    if (it == fds_.end()) {
      errno = EBADF;
      return -1;
    }
    handle = it->second.handle;
    path = it->second.path;
    fds_.erase(it);
  }
  // Drain BEFORE releasing the chimod handle: a deferred write names that
  // handle, and the runtime answers EBADF once it is gone (issue #817).
  // close(2) is also the last chance to report a latched write failure, which
  // is why its return code is not simply the Close task's.
  int werr = Flush(path);
  auto t = AsyncClose(handle);
  t.Wait();
  if (werr != 0) {
    errno = EIO;
    return -1;
  }
  return (t->GetReturnCode() == 0) ? 0 : -1;
}

#endif  // !_WIN32

}  // namespace clio::cte::filesystem
