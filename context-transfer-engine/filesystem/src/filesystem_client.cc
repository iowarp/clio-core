/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */
#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/filesystem/filesystem_client.h>
#include <clio_cte/stream/stream_client.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace clio::cte::filesystem {

// Process-wide filesystem client singleton (defined inside the namespace so it
// is clio::cte::filesystem::g_fs_client, matching the CLIO_CFS_CLIENT macro).
CLIO_CTE_FS_DEFINE_GLOBAL_PTR_VAR_CC(clio::cte::filesystem::Client, g_fs_client);

/**
 * Find the filesystem pool this process should use, by name or id.
 * @param fs_client client to bind (pool_id_ is set on success)
 * @param pool_query routing of the lookup
 * @return true when the pool exists
 */
static bool ResolveFsPool(Client *fs_client,
                          const clio::run::PoolQuery &pool_query) {
  std::string target = kCfsPoolName;
  if (const char *env = std::getenv("CLIO_CFS_POOL")) {
    if (*env != '\0') target = env;
  }
  unsigned major = 0, minor = 0;
  char tail = 0;
  if (std::sscanf(target.c_str(), "%u.%u%c", &major, &minor, &tail) == 2) {
    fs_client->Init(clio::run::PoolId(major, minor));
    return true;
  }
  // A get-or-create with no pool id is a lookup by name: it returns the
  // existing pool's id and creates nothing (the admin waits for compose to
  // create a pool its config names, so a client racing startup still binds).
  auto lookup = fs_client->AsyncCreate(pool_query, target,
                                       clio::run::PoolId::GetNull(),
                                       FilesystemConfig());
  lookup.Wait();
  if (lookup->GetReturnCode() != 0 || lookup->new_pool_id_.IsNull()) {
    HLOG(kError,
         "CFS ClientInit: no filesystem pool named '{}' (rc={}); compose a "
         "clio_cte_filesystem pool with that pool_name, or set CLIO_CFS_POOL",
         target, lookup->GetReturnCode());
    return false;
  }
  fs_client->Init(lookup->new_pool_id_);
  return true;
}

/**
 * Bind the process-wide filesystem client to the composed filesystem pool
 * and to the pools that pool is configured with. Creates no pool: the
 * deployment's compose owns every pool and its id.
 */
bool CLIO_CFS_CLIENT_INIT(const std::string &config_path,
                          const clio::run::PoolQuery &pool_query) {
  static bool s_initialized = false;
  if (s_initialized) {
    return true;
  }
  (void)config_path;  // configuration now flows through clio compose
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false)) {
    HLOG(kError, "CFS ClientInit: runtime client init failed");
    return false;
  }

  // CLIO_CFS_CLIENT lazily allocates the global Client on first access.
  auto *fs_client = CLIO_CFS_CLIENT;
  if (fs_client == nullptr || !ResolveFsPool(fs_client, pool_query)) {
    return false;
  }
  if (!fs_client->BindPools()) {
    HLOG(kError,
         "CFS ClientInit: filesystem pool {}.{} did not report its data and "
         "stream pools", fs_client->pool_id_.major_,
         fs_client->pool_id_.minor_);
    return false;
  }

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

// ---------------------------------------------------------------------------
// Shared-memory data path
// ---------------------------------------------------------------------------

namespace {
/** Retries of a page read whose owner was just declared dead. */
constexpr int kReadNodeLostRetries = 20;
/** Pause between those retries (ms). */
constexpr int kReadNodeLostRetryMs = 500;
/** Poll interval while waiting for the background size drain (us). */
constexpr int kSizeDrainPollUs = 200;

/** @return the packed (major << 32 | minor) form of a tag */
clio::run::u64 PackTag(const clio::cte::core::TagId &tag) {
  return (static_cast<clio::run::u64>(tag.major_) << 32) |
         static_cast<clio::run::u64>(tag.minor_);
}
}  // namespace

FsSsize Client::WritePages(const clio::cte::core::TagId &tag,
                           clio::run::u64 off, const void *buf, size_t count,
                           bool *size_recorded) {
  *size_recorded = true;
  if (count == 0) return 0;
  if (!HasShmCache()) AttachShmCache();
  clio::cte::core::PageScorePolicy policy;
  bool load_policy = false;
  {
    std::lock_guard<std::mutex> g(fd_mu_);
    load_policy = !page_score_loaded_;
    policy = page_score_;
  }
  if (load_policy) {
    clio::cte::core::PageScorePolicy p;
    if (GetPageScorePolicy(&p)) policy = p;
    std::lock_guard<std::mutex> g(fd_mu_);
    page_score_ = policy;
    page_score_loaded_ = true;
  }
  auto *cte = Cte();
  const char *src = static_cast<const char *>(buf);
  size_t done = 0;
  while (done < count) {
    const clio::run::u64 cur = off + done;
    const clio::run::u64 page_off = cur % kFsPageSize;
    const clio::run::u64 n =
        std::min<clio::run::u64>(kFsPageSize - page_off, count - done);
    const clio::run::u32 rc =
        cte->PutBlob(tag, PageName(cur), page_off, n, src + done,
                     policy.ScoreOf(cur / kFsPageSize));
    if (rc != 0) {
      // Bytes already stored by earlier pages are kept: the file grows to
      // them, and write(2) reports what was stored.
      if (done > 0) break;
      if (clio::cte::core::PutRcIsNoSpace(rc)) return -ENOSPC;
      return rc == ENOMEM ? -ENOMEM : -EIO;
    }
    done += n;
  }
  const clio::run::u64 end = off + done;
  const clio::run::u64 wtime =
      static_cast<clio::run::u64>(clio::cte::core::GetWallTimeNs());
  *size_recorded = shm_fs_root_ != nullptr &&
                   shm_fs_root_->sizes_.Raise(PackTag(tag), end, wtime);
  return static_cast<FsSsize>(done);
}

int Client::ReadPage(const clio::cte::core::TagId &tag, clio::run::u64 cur,
                     clio::run::u64 n, char *dst) {
  auto *cte = Cte();
  const std::string page = PageName(cur);
  const clio::run::u64 page_off = cur % kFsPageSize;
  if (cte->TryGetCached(tag, page, page_off, n, dst)) return 0;
  int grc = 0;
  for (int attempt = 0;; ++attempt) {
    auto g = cte->AsyncGetBlob(tag, page, page_off, n, 0, dst);
    if (g.get() == nullptr) return ENOMEM;
    for (int slice = 1; !g.Wait(30.0f); ++slice) {
      HLOG(kWarning, "clio-fs: read of tag {}.{} page {} (off {} len {}) "
           "still outstanding after {} s", tag.major_, tag.minor_, page,
           page_off, n, slice * 30);
    }
    grc = static_cast<int>(g->GetReturnCode());
    // Owner-down from a stand-in (#1280) can be as transient as node-lost:
    // the stand-in's liveness view lags the router's.
    const bool transient =
        clio::cte::core::IsNodeLostRc(static_cast<clio::run::u32>(grc)) ||
        static_cast<clio::run::u32>(grc) == clio::cte::core::kBlobOwnerDownRc;
    if (!transient || attempt >= kReadNodeLostRetries) break;
    std::this_thread::sleep_for(
        std::chrono::milliseconds(kReadNodeLostRetryMs));
  }
  return grc;
}

int Client::FileSizeOf(const clio::cte::core::TagId &tag,
                       const std::string &path, clio::run::u64 *size) {
  if (!HasShmCache()) AttachShmCache();
  ShmFileRecord rec;
  if (TryGetFileRecordShm(path, &rec) && rec.Exists() && !rec.IsDir() &&
      rec.tag_id_.major_ == tag.major_ && rec.tag_id_.minor_ == tag.minor_) {
    *size = rec.size_;  // includes the pending size
    return 0;
  }
  // The pending size is read BEFORE the home's: an entry retired between
  // the two had already landed at the home, so the StatSize reply covers it.
  const clio::run::u64 pending = PendingSize(tag);
  auto t = AsyncStatSize(path);
  t.Wait();
  if (t->GetReturnCode() != 0) return -EIO;
  if (t->exists_ == 0) return -ENOENT;
  *size = std::max({t->size_, pending, PendingSize(tag)});
  return 0;
}

FsSsize Client::ReadPages(const clio::cte::core::TagId &tag,
                          const std::string &path, clio::run::u64 off,
                          void *buf, size_t count) {
  if (count == 0) return 0;
  clio::run::u64 fsize = 0;
  const int src = FileSizeOf(tag, path, &fsize);
  if (src != 0) return src;
  if (off >= fsize) return 0;
  const clio::run::u64 want = std::min<clio::run::u64>(count, fsize - off);
  char *dst = static_cast<char *>(buf);
  clio::run::u64 done = 0;
  while (done < want) {
    const clio::run::u64 cur = off + done;
    const clio::run::u64 n = std::min<clio::run::u64>(
        kFsPageSize - cur % kFsPageSize, want - done);
    const int grc = ReadPage(tag, cur, n, dst + done);
    if (grc == 1) {
      std::memset(dst + done, 0, n);  // a hole reads as zeros
    } else if (grc != 0) {
      HLOG(kError, "clio-fs: read of tag {}.{} page {} failed rc {}",
           tag.major_, tag.minor_, PageName(cur), grc);
      return -EIO;
    }
    done += n;
  }
  return static_cast<FsSsize>(want);
}

clio::run::u64 Client::PendingSize(const clio::cte::core::TagId &tag) {
  if (shm_fs_root_ == nullptr && !AttachShmCache()) return 0;
  return shm_fs_root_->sizes_.Get(PackTag(tag));
}

clio::run::u64 Client::PendingSizeOfPath(const std::string &path) {
  if (shm_fs_root_ == nullptr && !AttachShmCache()) return 0;
  const std::string key = shm_fs_root_->Key(path);
  ShmFileRecord rec;
  if (!shm_fs_root_->path_to_file_.TryGetBytes(key.data(), key.size(), &rec,
                                               16, nullptr) ||
      rec.tag_id_.IsNull() || (rec.flags_ & kShmFileIsDir)) {
    return 0;
  }
  return shm_fs_root_->sizes_.Get(
      (static_cast<clio::run::u64>(rec.tag_id_.major_) << 32) |
      static_cast<clio::run::u64>(rec.tag_id_.minor_));
}

int Client::WaitSizeDrained(const clio::cte::core::TagId &tag) {
  if (tag.IsNull() || PendingSize(tag) == 0) return 0;
  const auto t0 = std::chrono::steady_clock::now();
  for (;;) {
    const clio::run::u64 end = PendingSize(tag);
    if (end == 0) return 0;
    const double waited_ms = std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - t0)
                                 .count();
    if (waited_ms >= kSizeDrainWaitMs) {
      // The drain did not come: publish it here (a max, so a drain landing
      // the same size later is harmless).
      auto t = AsyncAdvanceSize(PackTag(tag), end);
      t.Wait();
      const clio::run::u32 rc = t->GetReturnCode();
      if (rc != 0 && rc != ENOENT) return rc < 4096 ? static_cast<int>(rc) : EIO;
      shm_fs_root_->sizes_.Retire(PackTag(tag), end);
      return 0;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(kSizeDrainPollUs));
  }
}

/**
 * Whether a write through this descriptor takes the shared-memory page path.
 * O_APPEND needs the home to place the bytes and O_SYNC asked for a durable
 * write, so both keep the filesystem's Write task; so does a file whose tag
 * the open did not report.
 * @param of the open file
 */
static bool UsesPagePath(const Client::OpenFile &of) {
  return !of.tag.IsNull() && (of.flags & O_APPEND) == 0 && !Client::IsSyncFd(of.flags);
}

/**
 * Write through the open file's pages (UsesPagePath) or the Write task.
 * @param cfs the client
 * @param of the open file
 * @param off file offset
 * @param buf source
 * @param count bytes
 * @return bytes written, or -1 with errno set
 */
static FsSsize WriteOpenFile(Client *cfs, const Client::OpenFile &of,
                             clio::run::u64 off, const void *buf,
                             size_t count) {
  if (!UsesPagePath(of)) {
    return cfs->Write(of.handle, of.path, off, buf, count,
                      Client::IsSyncFd(of.flags));
  }
  bool recorded = true;
  const FsSsize n = cfs->WritePages(of.tag, off, buf, count, &recorded);
  if (n < 0) {
    errno = static_cast<int>(-n);
    return -1;
  }
  if (!recorded && n > 0) {
    // No size table to take it: publish the new size now.
    auto t = cfs->AsyncAdvanceSize(
        (static_cast<clio::run::u64>(of.tag.major_) << 32) | of.tag.minor_,
        off + static_cast<clio::run::u64>(n));
    t.Wait();
    if (t->GetReturnCode() != 0) {
      errno = EIO;
      return -1;
    }
  }
  return n;
}

/**
 * Read through the open file's pages when its tag is known, else the Read
 * task.
 * @param cfs the client
 * @param of the open file
 * @param off file offset
 * @param buf destination
 * @param count bytes
 * @return bytes read, or -1 with errno set
 */
static FsSsize ReadOpenFile(Client *cfs, const Client::OpenFile &of,
                            clio::run::u64 off, void *buf, size_t count) {
  if (of.tag.IsNull()) return cfs->Read(of.handle, of.path, off, buf, count);
  const FsSsize n = cfs->ReadPages(of.tag, of.path, off, buf, count);
  if (n < 0) {
    errno = static_cast<int>(-n);
    return -1;
  }
  return n;
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
    if (t->tag_packed_ != 0) WaitSizeDrained(FsUnpack(t->tag_packed_));
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
  if (t->tag_packed_ != 0) of.tag = FsUnpack(t->tag_packed_);
  fds_[fd] = of;
  return fd;
}

FsSsize Client::ReadFd(int fd, void *buf, size_t count) {
  OpenFile of;
  if (!LookupFd(fd, &of)) {
    return -1;
  }
  FsSsize n = ReadOpenFile(this, of, of.off, buf, count);
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
  FsSsize n = WriteOpenFile(this, of, of.off, buf, count);
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
  return ReadOpenFile(this, of, static_cast<clio::run::u64>(offset), buf,
                      count);
}

FsSsize Client::PwriteFd(int fd, const void *buf, size_t count, FsOff offset) {
  OpenFile of;
  if (!LookupFd(fd, &of)) {
    return -1;
  }
  return WriteOpenFile(this, of, static_cast<clio::run::u64>(offset), buf,
                       count);
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
      if (!of.tag.IsNull()) {
        const int src = FileSizeOf(of.tag, of.path, &size);
        if (src != 0) {
          errno = -src;
          return -1;
        }
      } else if (!GetSize(of.path, &size)) {
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
  if (!of.tag.IsNull()) {
    const int src = FileSizeOf(of.tag, of.path, &size);
    if (src != 0) {
      errno = -src;
      return -1;
    }
  } else if (!GetSize(of.path, &size)) {
    return -1;
  }
  return static_cast<FsOff>(size);
}

int Client::SyncFd(int fd) {
  OpenFile of;
  if (!LookupFd(fd, &of)) {
    return -1;
  }
  // The size page writes raised reaches the home first; then every
  // write(2) (in the shm cache or the chain) is put on a persistent device.
  // Otherwise they stay on whatever tier took them (RAM) until the periodic
  // flush, while the same file fsynced through the FUSE mount was durable
  // (#1188).
  const int drc = WaitSizeDrained(of.tag);
  if (drc != 0) {
    errno = drc;
    return -1;
  }
  int rc = SyncDurable(of.tag.IsNull()
                           ? TagOfPath(of.path, /*want_dir=*/false)
                           : of.tag,
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
  // Through the filesystem's own chain: a cache above the cores flushes
  // its dirty copies of the tag before the cores sync.
  auto *cte_c = CLIO_CFS_CLIENT->Cte();
  if (cte_c->pool_id_.IsNull() || tag.IsNull()) return 0;
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
  auto f = CLIO_CFS_CLIENT->Stream()->AsyncSizeOp(
      tag, FsIdHome(packed), clio::cte::stream::StreamSizeOp::kSync);
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
  rc = SyncTagDurable(TagOfPath(dir, /*want_dir=*/true));
  if (rc != 0) return rc;
  return SyncMetaDurable();
}

int Client::SyncMetaDurable() {
  if (FsyncDeferred()) return 0;
  auto t = AsyncSyncMeta();
  t.Wait();
  const clio::run::u32 rc = t->GetReturnCode();
  if (rc != 0 && !clio::cte::core::IsNodeLostRc(rc)) {
    HLOG(kError, "fsync of the filesystem namespace failed (rc {})", rc);
    return -EIO;
  }
  return 0;
}

int Client::FtruncateFd(int fd, FsOff length) {
  OpenFile of;
  if (!LookupFd(fd, &of)) {
    return -1;
  }
  // The size page writes raised lands before the truncate, which then
  // applies after them.
  const int drc = WaitSizeDrained(of.tag);
  if (drc != 0) {
    errno = drc;
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
  // A size page writes raised and the drain has not landed would otherwise
  // be applied AFTER this truncate and resurrect the cut length.
  {
    ShmFileRecord rec;
    if (TryGetFileRecordShm(path, &rec) && !rec.tag_id_.IsNull()) {
      const int drc = WaitSizeDrained(
          clio::cte::core::TagId(rec.tag_id_.major_, rec.tag_id_.minor_));
      if (drc != 0) {
        errno = drc;
        return -1;
      }
    }
  }
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
  {
    std::lock_guard<std::mutex> g(fd_mu_);
    auto it = fds_.find(fd);
    if (it == fds_.end()) {
      errno = EBADF;
      return -1;
    }
    handle = it->second.handle;
    fds_.erase(it);
  }
  // Writes are synchronous, so nothing is pending against the handle.
  auto t = AsyncClose(handle);
  t.Wait();
  return (t->GetReturnCode() == 0) ? 0 : -1;
}


}  // namespace clio::cte::filesystem
