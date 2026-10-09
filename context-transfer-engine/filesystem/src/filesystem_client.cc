/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */
#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/filesystem/filesystem_client.h>
#include <clio_cte/stream/stream_client.h>

#include <atomic>
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

void Client::MakeParents(const std::string &path) {
  // Every ancestor, outermost first; EEXIST (or a race with another creator)
  // is fine -- the create that follows reports any real failure.
  size_t pos = 1;
  while ((pos = path.find('/', pos)) != std::string::npos) {
    auto m = AsyncMkdir(path.substr(0, pos));
    m.Wait();
    ++pos;
  }
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
  if ((flags & O_CREAT) != 0 && t->GetReturnCode() == ENOENT) {
    // A `clio::` path mirrors a host path whose directories exist on the host
    // but not (yet) in clio-fs: create the missing parents, then retry once.
    // (FUSE stays strictly POSIX: a create under a missing parent is ENOENT.)
    MakeParents(path);
    t = AsyncOpen(path, static_cast<clio::run::u32>(flags),
                  static_cast<clio::run::u32>(mode));
    t.Wait();
  }
  const clio::run::u32 rc = t->GetReturnCode();
  if (rc != 0) {
    errno = rc < 4096 ? static_cast<int>(rc) : EIO;
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

FsSsize Client::ReadFd(int fd, void *buf, size_t count) {
  OpenFile of;
  if (!LookupFd(fd, &of)) {
    return -1;
  }
  FsSsize n = Read(of.handle, of.path, of.off, buf, count);
  if (n > 0) {
    AdvanceFd(fd, static_cast<clio::run::u64>(n));
  }
  return n;
}

FsSsize Client::WriteFd(int fd, const void *buf, size_t count) {
  OpenFile of;
  if (!LookupFd(fd, &of)) {
    return -1;
  }
  FsSsize n = Write(of.handle, of.path, of.off, buf, count,
                                     IsSyncFd(of.flags));
  if (n > 0) {
    AdvanceFd(fd, static_cast<clio::run::u64>(n));
  }
  return n;
}

FsSsize Client::PreadFd(int fd, void *buf, size_t count, FsOff offset) {
  OpenFile of;
  if (!LookupFd(fd, &of)) {
    return -1;
  }
  return Read(of.handle, of.path,
                               static_cast<clio::run::u64>(offset), buf, count);
}

FsSsize Client::PwriteFd(int fd, const void *buf, size_t count, FsOff offset) {
  OpenFile of;
  if (!LookupFd(fd, &of)) {
    return -1;
  }
  return Write(of.handle, of.path,
                                static_cast<clio::run::u64>(offset), buf, count,
                                IsSyncFd(of.flags));
}

FsOff Client::SeekFd(int fd, FsOff offset, int whence) {
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
  FsOff newoff = static_cast<FsOff>(base) + offset;
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

FsOff Client::TellFd(int fd) {
  OpenFile of;
  if (!LookupFd(fd, &of)) {
    return -1;
  }
  return static_cast<FsOff>(of.off);
}

FsOff Client::SizeFd(int fd) {
  OpenFile of;
  if (!LookupFd(fd, &of)) {
    return -1;
  }
  clio::run::u64 size = 0;
  if (!GetSize(of.path, &size)) {
    return -1;
  }
  return static_cast<FsOff>(size);
}

int Client::SyncFd(int fd) {
  OpenFile of;
  if (!LookupFd(fd, &of)) {
    return -1;
  }
  // Wait for every deferred write on this file and report a latched failure
  // exactly once — fsync and close are the only two places a deferred write's
  // failure can reach the application.
  if (Flush(of.path) != 0) {
    return -1;
  }
  // Then put the bytes on a persistent device. Draining alone left them on
  // whatever tier took them (RAM) until the periodic flush, while the same
  // file fsynced through the FUSE mount was durable (#1188).
  int rc = SyncDurable(TagOfPath(of.path, /*want_dir=*/false),
                       FsParentDir(of.path));
  if (rc != 0) {
    errno = -rc;
    return -1;
  }
  return 0;
}

namespace {
/** CTE performance.fsync_mode from the first SyncTag reply: -1 not yet
 *  known, 0 "durable", 1 "deferred". */
std::atomic<int> g_fsync_deferred{-1};
}  // namespace

bool Client::FsyncDeferred() {
  return g_fsync_deferred.load(std::memory_order_relaxed) == 1;
}

int Client::SyncTagDurable(const clio::cte::core::TagId &tag, bool *lost_node,
                           clio::run::u64 *liveness_change_ns) {
  if (lost_node != nullptr) *lost_node = false;
  if (liveness_change_ns != nullptr) *liveness_change_ns = 0;
  if (FsyncDeferred()) return 0;
  auto *cte_c = CLIO_CTE_CLIENT;
  if (cte_c == nullptr || tag.IsNull()) return 0;
  auto fut = cte_c->AsyncSyncTag(tag);
  fut.Wait();
  const bool deferred = fut->deferred_ != 0;
  g_fsync_deferred.store(deferred ? 1 : 0, std::memory_order_relaxed);
  if (deferred) return 0;
  if (fut->containers_ == 0) {
    // A module in front of the core dropped the sync: nothing was made
    // durable, so fsync must not claim it was.
    static std::once_flag warned;
    std::call_once(warned, [&] {
      HLOG(kError, "fsync reached no CTE core container through pool {}.{}; "
           "fsync fails with EIO", cte_c->pool_id_.major_,
           cte_c->pool_id_.minor_);
    });
    return -EIO;
  }
  const clio::run::u32 rc = fut->GetReturnCode();
  if (rc == clio::cte::core::kSyncNoSpaceRc) return -ENOSPC;
  if (rc != 0 && !clio::cte::core::IsNodeLostRc(rc)) {
    HLOG(kError, "fsync of tag {}.{} failed (rc {}); reporting EIO",
         tag.major_, tag.minor_,
         static_cast<long long>(static_cast<clio::run::i32>(rc)));
    return -EIO;
  }
  // rc != 0 here is the lost-node code: the live containers synced.
  if (lost_node != nullptr) *lost_node = rc != 0;
  if (liveness_change_ns != nullptr) {
    *liveness_change_ns = fut->liveness_change_ns_;
  }
  return 0;
}

int Client::SyncFileSize(const clio::cte::core::TagId &tag) {
  if (tag.IsNull()) return 0;
  const clio::run::u64 packed = FsPack(tag);
  if (!FsIdHasHome(packed)) return 0;
  clio::cte::stream::Client stream;
  auto f = stream.AsyncSizeOp(tag, FsIdHome(packed),
                              clio::cte::stream::StreamSizeOp::kSync);
  f.Wait();
  const clio::run::u32 rc = f->GetReturnCode();
  if (rc == 0) return 0;
  if (clio::cte::core::IsNodeLostRc(rc)) {
    // The home died: a size it logged is in its log (replayed when it
    // restarts); sizes set since then are logged by its successor.
    HLOG(kWarning, "fsync of the size of {}.{}: its home is down",
         tag.major_, tag.minor_);
    return 0;
  }
  HLOG(kError, "fsync of the size of {}.{} failed (rc {})", tag.major_,
       tag.minor_, rc);
  return -EIO;
}

clio::cte::core::TagId Client::TagOfPath(const std::string &path,
                                         bool want_dir) {
  auto t = AsyncGetattr(path);
  t.Wait();
  if (t->GetReturnCode() != 0 || t->exists_ == 0 ||
      (want_dir && t->is_dir_ == 0)) {
    return clio::cte::core::TagId::GetNull();
  }
  return FsUnpack(t->ino_);
}

int Client::SyncDurable(const clio::cte::core::TagId &tag,
                        const std::string &dir) {
  int rc = SyncTagDurable(tag);
  if (rc != 0) return rc;
  if (FsyncDeferred()) return 0;
  rc = SyncFileSize(tag);
  if (rc != 0) return rc;
  return SyncTagDurable(TagOfPath(dir, /*want_dir=*/true));
}

int Client::FtruncateFd(int fd, FsOff length) {
  OpenFile of;
  if (!LookupFd(fd, &of)) {
    return -1;
  }
  return TruncatePath(std::string(kClioPrefix) + of.path, length);
}

int Client::TruncatePath(const std::string &raw_path, FsOff length) {
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
    // Pass the server's errno through: a listing that failed (EAGAIN, EIO)
    // is not a missing directory (#1029).
    const clio::run::u32 rc = t->GetReturnCode();
    errno = rc < 4096 ? static_cast<int>(rc) : EIO;  // runtime codes -> EIO
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


}  // namespace clio::cte::filesystem
