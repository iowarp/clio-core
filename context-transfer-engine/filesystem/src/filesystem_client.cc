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
#include <initializer_list>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
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
    if (FsMkdir(path.substr(0, pos), 0755) == kNsUnavailable) {
      auto m = AsyncMkdir(path.substr(0, pos));
      m.Wait();
    }
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
constexpr int kSizeDrainPollUs = 50;

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
  if (done > 0) UnsyncedNote(tag);
  const clio::run::u64 wtime =
      static_cast<clio::run::u64>(clio::cte::core::GetWallTimeNs());
  *size_recorded = shm_fs_root_ != nullptr &&
                   shm_fs_root_->sizes_.Raise(PackTag(tag), end, wtime);
  // The namespace's own inode holds the file's size and mtime locally; the
  // size table carries them to the home.
  if (ns::View *v = Ns()) v->RaiseSize(PackTag(tag), end, wtime);
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
  {
    ns::Attr a;
    const int nrc = FsStatId(FsPack(tag), &a);
    if (nrc == 0) {
      *size = a.size;  // includes the pending size
      return 0;
    }
  }
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
  // Ask the module for its fast size drain while this waits.
  ns::View *seg = NsSegment();
  if (seg != nullptr) seg->Urge();
  struct Unurge {
    ns::View *v;
    ~Unurge() {
      if (v != nullptr) v->Unurge();
    }
  } unurge{seg};
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
  if (of.handle == 0) return true;  // namespace-held: no module handle
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
  if ((of.flags & O_APPEND) != 0 && !of.tag.IsNull()) {
    FsSsize n = 0;
    const int arc = cfs->FsAppend(of.tag, of.path, buf, count, &n);
    if (arc != Client::kNsUnavailable) {
      if (arc != 0) {
        errno = arc;
        return -1;
      }
      return n;
    }
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
  if (of.handle == 0 && Client::IsSyncFd(of.flags) && n > 0) {
    // O_SYNC/O_DSYNC: durable before write(2) returns.
    const int drc = cfs->WaitSizeDrained(of.tag);
    if (drc != 0 || cfs->FsWaitApplied() != 0 ||
        cfs->SyncDurable(of.tag, FsParentDir(of.path)) != 0) {
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

// ---------------------------------------------------------------------------
// Namespace: the shared-memory namespace (fs_ns_shm.h)
// ---------------------------------------------------------------------------

namespace {
/** Fill rounds before a namespace call gives up on a missing name. */
constexpr int kNsFillRounds = 8;
/** Longest wait for a block of inode ids (ms). */
constexpr double kNsIdWaitMs = 10000.0;
/** "Leave alone" owner / the mounting user, as the module stores it. */
constexpr clio::run::u32 kNoOwner = 0xFFFFFFFFu;

/** @return wall-clock ns */
clio::run::u64 NowNs() {
  return static_cast<clio::run::u64>(clio::cte::core::GetWallTimeNs());
}

/**
 * Run a namespace operation, filling missing names and waiting for ids.
 * @param cfs the client
 * @param paths the paths the operation resolves (filled on a miss)
 * @param dir_fill a directory whose listing a miss may need (rmdir, a
 *        rename over a directory, readdir); empty for none
 * @param op the operation (0, errno, ns::kMiss, or EAGAIN for ids)
 */
template <typename Op>
int NsRetry(Client *cfs, std::initializer_list<const std::string *> paths,
            const std::string &dir_fill, Op op) {
  const auto t0 = std::chrono::steady_clock::now();
  for (int round = 0;; ) {
    const int rc = op();
    if (rc == EAGAIN) {
      // The module tops the id block up every period; wait for it.
      const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
      if (ms >= kNsIdWaitMs) {
        HLOG(kError, "filesystem: no inode id block after {:.0f} ms (the "
             "module did not refill it); reporting EIO for '{}'", ms,
             paths.size() != 0 ? **paths.begin() : std::string());
        return EIO;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(500));
      continue;
    }
    if (rc != ns::kMiss) return rc;
    if (++round > kNsFillRounds) {
      HLOG(kError, "filesystem: namespace still missing '{}' after {} fill "
           "rounds; reporting EIO",
           paths.size() != 0 ? **paths.begin() : std::string(), kNsFillRounds);
      return EIO;
    }
    for (const std::string *p : paths) {
      const int frc = cfs->NsFill(*p);
      if (frc != 0) return frc;
    }
    if (!dir_fill.empty() && round > 1) {
      const int frc = cfs->NsFillDir(dir_fill);
      if (frc != 0 && frc != ENOENT && frc != ENOTDIR) return frc;
    }
  }
}
}  // namespace

ns::View *Client::NsSegment() {
  if (ns_view_ == nullptr) AttachNs();
  return (ns_view_ != nullptr && ns_view_->Ready()) ? ns_view_.get()
                                                     : nullptr;
}

ns::View *Client::Ns() {
  ns::View *v = NsSegment();
  return (v != nullptr && v->NamespaceOn()) ? v : nullptr;
}

void Client::AttachNs() {
  if (ns_view_ != nullptr) return;
  auto *ipc = CLIO_CPU_IPC;
  auto *alloc = ipc != nullptr ? ipc->GetMetadataAllocator() : nullptr;
  auto *dir = ipc != nullptr ? ipc->GetMetadataDirectory() : nullptr;
  if (alloc == nullptr || dir == nullptr) return;  // TCP client, remote
  const clio::run::u64 off = dir->FindRoot(ns::LocatorKey(pool_id_.ToU64()));
  if (off == 0) return;  // not formatted (yet): retried on the next call
  const auto *l = reinterpret_cast<const ns::Locator *>(
      reinterpret_cast<const char *>(alloc) + off);
  if (l->magic_ != ns::Locator::kMagic ||
      l->ready_.load(std::memory_order_acquire) != 1) {
    return;
  }
  auto v = std::make_shared<ns::View>();
  if (v->Attach(ns::SegmentName(l->pid_, pool_id_.ToU64()))) ns_view_ = v;
}

namespace {
/** @return true when CLIO_CFS_APPEND_MERGE=1 forces the merge path */
bool AppendMergeForced() {
  static const bool forced = [] {
    const char *e = clio::run::env::GetCompat("CFS_APPEND_MERGE");
    return e != nullptr && *e == '1';
  }();
  return forced;
}
/** Bytes of one append moved per page put into the staging blob. */
constexpr size_t kAppendPutBytes = 16u << 20;
}  // namespace

int Client::FsAppend(const clio::cte::core::TagId &tag,
                     const std::string &path, const void *buf, size_t count,
                     FsSsize *written) {
  ns::View *v = NsSegment();
  if (v == nullptr) return kNsUnavailable;
  *written = 0;
  if (count == 0) return 0;
  const clio::run::u64 id = PackTag(tag);
  if (v->NamespaceOn() && !AppendMergeForced()) {
    // One node: the file's end is exact here. The append lock makes the
    // size read and the page writes one step for every appender.
    if (v->AppendLock(id) != 0) return ENOSPC;
    clio::run::u64 end = 0;
    int rc = -FileSizeOf(tag, path, &end);
    if (rc == 0) {
      bool recorded = true;
      const FsSsize n = WritePages(tag, end, buf, count, &recorded);
      if (n < 0) {
        rc = static_cast<int>(-n);
      } else {
        *written = n;
      }
    }
    v->AppendUnlock(id);
    return rc;
  }
  // Merge path: stage into this node's append blob (shared-memory puts),
  // then log it; the module merges it at the file's end in the background.
  clio::run::u32 gen = 0;
  clio::run::u64 off = 0;
  if (v->AppendStage(id, count, &gen, &off) != 0) return ENOSPC;
  const std::string blob =
      "app." + std::to_string(v->NodeId()) + "." + std::to_string(gen);
  const char *src = static_cast<const char *>(buf);
  int rc = 0;
  for (size_t done = 0; done < count;) {
    const size_t n = std::min(count - done, kAppendPutBytes);
    const clio::run::u32 prc =
        Cte()->PutBlob(tag, blob, off + done, n, src + done);
    if (prc != 0) {
      rc = clio::cte::core::PutRcIsNoSpace(prc) ? ENOSPC : EIO;
      break;
    }
    done += n;
  }
  // Always: the generation's in-flight count must drop for it to merge.
  v->AppendStaged(path, id, gen);
  if (rc == 0) UnsyncedNote(tag);
  if (rc == 0) *written = static_cast<FsSsize>(count);
  return rc;
}

int Client::WaitAppendsMerged(const clio::cte::core::TagId &tag) {
  ns::View *v = NsSegment();
  if (v == nullptr || tag.IsNull()) return 0;
  const clio::run::u64 id = PackTag(tag);
  if (v->AppendIdle(id)) return 0;
  // Its kOpAppend records are in the log: waiting for the log to apply
  // through the last push covers them.
  if (!v->WaitApplied(v->PushedSeq(), kNsDrainWaitMs)) return EIO;
  return 0;
}

int Client::NsFill(const std::string &path) {
  ns::View *v = Ns();
  if (v == nullptr) return kNsUnavailable;
  const std::vector<std::string> comps = ns::SplitPath(path);
  std::string parent = "/";
  for (size_t i = 0; i < comps.size(); ++i) {
    const std::string cur =
        (parent == "/" ? std::string("/") : parent + "/") + comps[i];
    ns::Attr have;
    const int lrc = v->Lookup(cur, &have);
    if (lrc == 0) {
      parent = cur;
      continue;
    }
    if (lrc == ENOENT || lrc == ENOTDIR) return 0;  // answered locally
    // Not held (or held without its inode): ask the module once.
    auto t = AsyncGetattr(cur);
    t.Wait();
    if (t->GetReturnCode() != 0) return EIO;
    if (t->exists_ == 0) {
      const int frc = v->FillAbsent(parent, comps[i]);
      return frc == 0 ? 0 : frc;
    }
    ns::Attr a;
    a.id = t->ino_;
    a.type = t->is_dir_ ? kFsTypeDir
                        : (t->is_symlink_ ? kFsTypeSymlink : kFsTypeFile);
    a.mode = t->mode_ != kNoOwner ? (t->mode_ & 07777u)
                                  : (t->is_dir_ ? 0755u : 0644u);
    a.uid = t->uid_;
    a.gid = t->gid_;
    a.nlink = t->is_dir_ ? 2u : t->nlink_;
    a.size = t->size_;
    a.atime = t->atime_;
    a.mtime = t->mtime_;
    a.ctime = t->ctime_;
    std::string target;
    if (t->is_symlink_) {
      auto r = AsyncReadlink(cur);
      r.Wait();
      if (r->GetReturnCode() == 0) target = r->target_.str();
    }
    const int frc = v->FillEntry(parent, comps[i], a, target);
    if (frc != 0) return frc;
    parent = cur;
  }
  return 0;
}

int Client::NsFillDir(const std::string &path) {
  ns::View *v = Ns();
  if (v == nullptr) return kNsUnavailable;
  const int prc = NsFill(path);
  if (prc != 0) return prc;
  auto t = AsyncReaddir(path);
  t.Wait();
  if (t->GetReturnCode() != 0) return EIO;
  const size_t n = t->entries_.size();
  for (size_t i = 0; i < n; ++i) {
    const clio::run::u64 id = i < t->inos_.size() ? t->inos_[i] : 0;
    const clio::run::u32 type = i < t->types_.size() ? t->types_[i] : 0;
    if (id == 0) continue;
    const int frc = v->FillName(path, t->entries_[i].str(), id,
                                type != 0 ? type : kFsTypeFile);
    if (frc != 0) return frc;
  }
  return v->MarkComplete(path);
}

int Client::FsStat(const std::string &path, ns::Attr *a) {
  ns::View *v = Ns();
  if (v == nullptr) return kNsUnavailable;
  const int rc = NsRetry(this, {&path}, std::string(),
                         [&] { return v->Lookup(path, a); });
  if (rc == 0 && a->type == kFsTypeFile) {
    a->size = std::max(a->size, PendingSize(FsUnpack(a->id)));
  }
  return rc;
}

int Client::FsStatId(clio::run::u64 id, ns::Attr *a) {
  ns::View *v = Ns();
  if (v == nullptr) return kNsUnavailable;
  if (!v->StatId(id, a)) return ENOENT;
  if (a->type == kFsTypeFile) {
    a->size = std::max(a->size, PendingSize(FsUnpack(id)));
  }
  return 0;
}

int Client::FsOpen(const std::string &path, int flags, clio::run::u32 mode,
                   ns::Attr *a, bool *created) {
  ns::View *v = Ns();
  if (v == nullptr) return kNsUnavailable;
  *created = false;
  int rc = 0;
  if (flags & O_CREAT) {
    const bool excl = (flags & O_EXCL) != 0;
    rc = NsRetry(this, {&path}, std::string(), [&] {
      return v->Create(path, mode, kNoOwner, kNoOwner, excl, NowNs(), a,
                       created);
    });
  } else {
    rc = NsRetry(this, {&path}, std::string(),
                 [&] { return v->Lookup(path, a); });
  }
  if (rc != 0) return rc;
  if (a->type == kFsTypeDir && (flags & (O_WRONLY | O_RDWR)) != 0) return EISDIR;
  if ((flags & O_TRUNC) && !*created && a->type == kFsTypeFile) {
    rc = FsTruncate(path, 0);
    if (rc != 0) return rc;
    a->size = 0;
  }
  if (a->type == kFsTypeFile) {
    a->size = std::max(a->size, PendingSize(FsUnpack(a->id)));
  }
  return 0;
}

int Client::FsMkdir(const std::string &path, clio::run::u32 mode) {
  ns::View *v = Ns();
  if (v == nullptr) return kNsUnavailable;
  return NsRetry(this, {&path}, std::string(), [&] {
    return v->Mkdir(path, mode, kNoOwner, kNoOwner, NowNs());
  });
}

int Client::FsRmdir(const std::string &path) {
  ns::View *v = Ns();
  if (v == nullptr) return kNsUnavailable;
  return NsRetry(this, {&path}, path,
                 [&] { return v->Rmdir(path, NowNs()); });
}

int Client::FsUnlink(const std::string &path) {
  ns::View *v = Ns();
  if (v == nullptr) return kNsUnavailable;
  {
    // Staged appends merge before the file goes.
    ns::Attr a;
    if (v->Lookup(path, &a) == 0 && a.type == kFsTypeFile) {
      const int wrc = WaitAppendsMerged(FsUnpack(a.id));
      if (wrc != 0) return wrc;
    }
  }
  return NsRetry(this, {&path}, std::string(),
                 [&] { return v->Unlink(path, NowNs()); });
}

int Client::FsRename(const std::string &src, const std::string &dst,
                     bool noreplace) {
  ns::View *v = Ns();
  if (v == nullptr) return kNsUnavailable;
  return NsRetry(this, {&src, &dst}, dst, [&] {
    return v->Rename(src, dst, noreplace, NowNs());
  });
}

int Client::FsLink(const std::string &src, const std::string &dst) {
  ns::View *v = Ns();
  if (v == nullptr) return kNsUnavailable;
  return NsRetry(this, {&src, &dst}, std::string(),
                 [&] { return v->Link(src, dst, NowNs()); });
}

int Client::FsSymlink(const std::string &target, const std::string &path) {
  ns::View *v = Ns();
  if (v == nullptr) return kNsUnavailable;
  return NsRetry(this, {&path}, std::string(), [&] {
    return v->Symlink(target, path, kNoOwner, kNoOwner, NowNs());
  });
}

int Client::FsReadlink(const std::string &path, std::string *target) {
  ns::View *v = Ns();
  if (v == nullptr) return kNsUnavailable;
  return NsRetry(this, {&path}, std::string(),
                 [&] { return v->Readlink(path, target); });
}

int Client::FsChmod(const std::string &path, clio::run::u32 mode) {
  ns::View *v = Ns();
  if (v == nullptr) return kNsUnavailable;
  return NsRetry(this, {&path}, std::string(),
                 [&] { return v->Chmod(path, mode, NowNs()); });
}

int Client::FsChown(const std::string &path, clio::run::u32 uid,
                    clio::run::u32 gid) {
  ns::View *v = Ns();
  if (v == nullptr) return kNsUnavailable;
  return NsRetry(this, {&path}, std::string(),
                 [&] { return v->Chown(path, uid, gid, NowNs()); });
}

int Client::FsUtimens(const std::string &path, clio::run::u64 atime_ns,
                      clio::run::u64 mtime_ns) {
  ns::View *v = Ns();
  if (v == nullptr) return kNsUnavailable;
  return NsRetry(this, {&path}, std::string(), [&] {
    return v->Utimens(path, atime_ns, mtime_ns, NowNs());
  });
}

int Client::FsTruncate(const std::string &path, clio::run::u64 size) {
  ns::View *v = Ns();
  if (v == nullptr) return kNsUnavailable;
  ns::Attr a;
  int rc = FsStat(path, &a);
  if (rc != 0) return rc;
  if (a.type == kFsTypeDir) return EISDIR;
  rc = WaitAppendsMerged(FsUnpack(a.id));  // the cut applies after them
  if (rc != 0) return rc;
  // The size earlier page writes raised lands first, so the cut applies
  // after them (a wait on the drain).
  rc = WaitSizeDrained(FsUnpack(a.id));
  if (rc != 0) return rc;
  clio::run::u64 id = 0;
  clio::run::u64 old_size = 0;
  clio::run::u64 seq = 0;
  rc = NsRetry(this, {&path}, std::string(), [&] {
    return v->Truncate(path, size, NowNs(), &id, &old_size, &seq);
  });
  if (rc != 0) return rc;
  // Every truncate waits for the module to apply it (a wait on the drain,
  // not a task). A shrink must cut the cached pages before a later write or
  // regrow can expose them; a grow zero-fills [old, new) in the pages, so
  // applied after later writes it would erase them (xfstests fsx: 075, 263).
  (void)old_size;
  if (!v->WaitApplied(seq, kNsDrainWaitMs)) return EIO;
  return 0;
}

int Client::FsFallocate(const clio::cte::core::TagId &tag,
                        const std::string &path, clio::run::u32 mode,
                        clio::run::u64 off, clio::run::u64 len) {
  if (Ns() == nullptr) return kNsUnavailable;
  if ((mode & ~(kFallocKeepSize | kFallocZeroRange)) != 0) return EOPNOTSUPP;
  if (len == 0) return EINVAL;
  clio::run::u64 size = 0;
  int rc = -FileSizeOf(tag, path, &size);
  if (rc != 0) return rc;
  if (mode & kFallocZeroRange) {
    // Holes read as zeros, but bytes already written must be cleared.
    static const std::vector<char> zeros(kFsPageSize, 0);
    for (clio::run::u64 done = 0; done < len;) {
      const size_t n = static_cast<size_t>(
          std::min<clio::run::u64>(len - done, kFsPageSize));
      bool recorded = true;
      const FsSsize w = WritePages(tag, off + done, zeros.data(), n, &recorded);
      if (w <= 0) return w < 0 ? static_cast<int>(-w) : EIO;
      if (!recorded) {
        auto t = AsyncAdvanceSize(PackTag(tag), off + done + w);
        t.Wait();
      }
      done += static_cast<clio::run::u64>(w);
    }
    if ((mode & kFallocKeepSize) && off + len > size) {
      return FsTruncate(path, size);  // the zero writes may not grow it
    }
    return 0;
  }
  // The space is promised: a request the store cannot hold fails now.
  {
    auto t = Cte()->AsyncGetCapacity(clio::run::PoolQuery::Broadcast());
    t.Wait();
    if (t->GetReturnCode() == 0 && len > t->remaining_capacity_) {
      return ENOSPC;
    }
  }
  if (mode & kFallocKeepSize) return 0;
  return off + len > size ? FsTruncate(path, off + len) : 0;
}

int Client::FsReaddir(const std::string &path, std::vector<ns::DirEnt> *out) {
  ns::View *v = Ns();
  if (v == nullptr) return kNsUnavailable;
  return NsRetry(this, {&path}, path, [&] {
    out->clear();
    return v->Readdir(path, out);
  });
}

int Client::NsFillXattrs(const std::string &path) {
  ns::View *v = Ns();
  if (v == nullptr) return kNsUnavailable;
  // The module answers for the file as this node last told it: its
  // changes land first.
  if (FsWaitApplied() != 0) return EIO;
  auto l = AsyncListxattr(path);
  l.Wait();
  if (l->GetReturnCode() != 0) {
    const clio::run::u32 rc = l->GetReturnCode();
    return rc < 4096 ? static_cast<int>(rc) : EIO;
  }
  const std::string names = l->names_.str();
  std::vector<std::pair<std::string, std::string>> attrs;
  for (size_t i = 0; i < names.size();) {
    const size_t end = names.find('\0', i);
    const std::string name =
        names.substr(i, end == std::string::npos ? std::string::npos : end - i);
    i = end == std::string::npos ? names.size() : end + 1;
    if (name.empty()) continue;
    auto g = AsyncGetxattr(path, name);
    g.Wait();
    if (g->GetReturnCode() != 0 || g->found_ == 0) continue;
    attrs.emplace_back(name, g->value_.str());
  }
  return v->FillXattrs(path, attrs);
}

namespace {
/**
 * Run an xattr operation, filling the path and then the file's xattrs on
 * a miss.
 * @param cfs the client
 * @param path the file
 * @param op the operation (0, errno, ns::kMiss, or ns::kTooBig)
 * @return its result; ns::kTooBig when the module holds the xattrs
 */
template <typename Op>
int XattrRetry(Client *cfs, const std::string &path, Op op) {
  for (int round = 0; round < 4; ++round) {
    const int rc = op();
    if (rc != ns::kMiss) return rc;
    int frc = cfs->NsFill(path);
    if (frc == 0) frc = cfs->NsFillXattrs(path);
    if (frc == ns::kTooBig) return ns::kTooBig;
    if (frc != 0) return frc;
  }
  return EIO;
}
}  // namespace

int Client::FsSetxattr(const std::string &path, const std::string &name,
                       const std::string &value, clio::run::u32 flags) {
  ns::View *v = Ns();
  if (v == nullptr) return kNsUnavailable;
  const int rc = XattrRetry(this, path, [&] {
    return v->Setxattr(path, name, value, flags, NowNs());
  });
  if (rc != ns::kTooBig) return rc;
  // Held by the module: set it there once this node's changes landed, and
  // let the next access refill the file's xattrs from it.
  if (FsWaitApplied() != 0) return EIO;
  auto t = AsyncSetxattr(path, name, value, flags);
  t.Wait();
  v->ForgetXattrs(path);
  const clio::run::u32 trc = t->GetReturnCode();
  return trc < 4096 ? static_cast<int>(trc) : EIO;
}

int Client::FsGetxattr(const std::string &path, const std::string &name,
                       std::string *value) {
  ns::View *v = Ns();
  if (v == nullptr) return kNsUnavailable;
  const int rc = XattrRetry(this, path,
                            [&] { return v->Getxattr(path, name, value); });
  if (rc != ns::kTooBig) return rc;
  if (FsWaitApplied() != 0) return EIO;
  auto t = AsyncGetxattr(path, name);
  t.Wait();
  if (t->GetReturnCode() != 0) {
    const clio::run::u32 trc = t->GetReturnCode();
    return trc < 4096 ? static_cast<int>(trc) : EIO;
  }
  if (t->found_ == 0) return ENODATA;
  *value = t->value_.str();
  return 0;
}

int Client::FsListxattr(const std::string &path, std::string *names) {
  ns::View *v = Ns();
  if (v == nullptr) return kNsUnavailable;
  const int rc =
      XattrRetry(this, path, [&] { return v->Listxattr(path, names); });
  if (rc != ns::kTooBig) return rc;
  if (FsWaitApplied() != 0) return EIO;
  auto t = AsyncListxattr(path);
  t.Wait();
  if (t->GetReturnCode() != 0) {
    const clio::run::u32 trc = t->GetReturnCode();
    return trc < 4096 ? static_cast<int>(trc) : EIO;
  }
  *names = t->names_.str();
  return 0;
}

int Client::FsRemovexattr(const std::string &path, const std::string &name) {
  ns::View *v = Ns();
  if (v == nullptr) return kNsUnavailable;
  const int rc = XattrRetry(this, path, [&] {
    return v->Removexattr(path, name, NowNs());
  });
  if (rc != ns::kTooBig) return rc;
  if (FsWaitApplied() != 0) return EIO;
  auto t = AsyncRemovexattr(path, name);
  t.Wait();
  const clio::run::u32 trc = t->GetReturnCode();
  return trc < 4096 ? static_cast<int>(trc) : EIO;
}

int Client::FsWaitApplied() {
  ns::View *v = NsSegment();  // appends too, on a multi-node volume
  if (v == nullptr) return 0;
  return v->WaitApplied(v->PushedSeq(), kNsDrainWaitMs) ? 0 : EIO;
}

int Client::OpenFd(const std::string &raw_path, int flags, int mode) {
  if (!EnsureInit()) {
    errno = EIO;
    return -1;
  }
  std::string path = StripClioPrefix(raw_path);
  {
    ns::Attr a;
    bool created = false;
    int nrc = FsOpen(path, flags, static_cast<clio::run::u32>(mode), &a,
                     &created);
    if (nrc == ENOENT && (flags & O_CREAT) != 0) {
      MakeParents(path);  // a clio:: path mirrors a host path (see below)
      nrc = FsOpen(path, flags, static_cast<clio::run::u32>(mode), &a,
                   &created);
    }
    if (nrc != kNsUnavailable) {
      if (nrc != 0) {
        errno = nrc;
        return -1;
      }
      std::lock_guard<std::mutex> g(fd_mu_);
      const int fd = next_fd_++;
      OpenFile of;
      of.handle = 0;  // no module handle: the namespace holds the file
      of.path = path;
      of.flags = flags;
      of.off = (flags & O_APPEND) ? a.size : 0;
      of.tag = FsUnpack(a.id);
      fds_[fd] = of;
      return fd;
    }
  }
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
  int drc = WaitSizeDrained(of.tag);
  // The namespace changes made so far (this file's create among them)
  // reach their homes before the file is made durable.
  if (drc == 0) drc = FsWaitApplied();
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

// Unsynced-write windows (issue #1133). With replication_factor 1 the bytes
// a node holds between two fsyncs (RAM tier, write-behind) die with it, and
// failover makes the next SyncTag succeed on the survivors. So each file
// written since its last fsync remembers when (wall clock) that window
// opened; fsync fails with EIO when a peer that has since died was last
// heard from after the window opened (SyncTagTask::liveness_change_ns_) --
// it may have accepted, and lost, some of those bytes. Conservative: the
// node may have held none of them. Reported once, like Linux's errseq.
namespace {
std::mutex &UnsyncedMu() {
  static std::mutex mu;
  return mu;
}
std::unordered_map<clio::run::u64, clio::run::u64> &Unsynced() {
  static std::unordered_map<clio::run::u64, clio::run::u64> m;
  return m;
}

/**
 * Close a file's unsynced window (taken BEFORE the sync, so a write racing
 * it opens a new one).
 * @param tag the file's tag
 * @param opened_ns receives the wall-clock ns the window opened at
 * @return true if the file had a window
 */
bool UnsyncedTake(const clio::cte::core::TagId &tag,
                  clio::run::u64 *opened_ns) {
  std::lock_guard<std::mutex> g(UnsyncedMu());
  auto it = Unsynced().find(PackTag(tag));
  if (it == Unsynced().end()) return false;
  *opened_ns = it->second;
  Unsynced().erase(it);
  return true;
}
}  // namespace

void Client::UnsyncedNote(const clio::cte::core::TagId &tag) {
  if (tag.IsNull()) return;
  std::lock_guard<std::mutex> g(UnsyncedMu());
  Unsynced().emplace(PackTag(tag), NowNs());  // keeps an open window
}

int Client::SyncTagWindow(const clio::cte::core::TagId &tag) {
  if (FsyncDeferred() || tag.IsNull()) return 0;
  clio::run::u64 opened_ns = 0;
  const bool unsynced = UnsyncedTake(tag, &opened_ns);
  bool lost_node = false;
  clio::run::u64 liveness_ns = 0;
  const int rc = SyncTagDurable(tag, &lost_node, &liveness_ns);
  if (rc != 0) return rc;
  // The live containers synced. A container whose node is down answers with
  // the lost-node code; once the cluster declares the node dead the sync
  // routes around it and succeeds. Either way, bytes written since the last
  // fsync that the node held are gone (#1133): fail if the window saw one.
  const bool moved = liveness_ns >= opened_ns;
  if (unsynced && (lost_node || moved)) {
    HLOG(kError, "clio-fs: fsync of tag {}.{}: a node {} while the file had "
         "unsynced writes; they may be lost, reporting EIO", tag.major_,
         tag.minor_, lost_node ? "is down" : "died");
    return -EIO;
  }
  if (lost_node) {
    HLOG(kWarning, "clio-fs: fsync of tag {}.{}: a container's node is down; "
         "the file had no unsynced writes", tag.major_, tag.minor_);
  }
  return 0;
}

int Client::SyncDurable(const clio::cte::core::TagId &tag,
                        const std::string &dir) {
  int rc = SyncTagWindow(tag);
  if (rc != 0) return rc;
  if (FsyncDeferred()) return 0;
  rc = SyncFileSize(tag);
  if (rc != 0) return rc;
  rc = SyncTagWindow(TagOfPath(dir, /*want_dir=*/true));
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
  {
    const int nrc = FsTruncate(path, static_cast<clio::run::u64>(length));
    if (nrc != kNsUnavailable) {
      if (nrc != 0) {
        errno = nrc;
        return -1;
      }
      return 0;
    }
  }
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
  {
    const int nrc = FsUnlink(path);
    if (nrc != kNsUnavailable) {
      if (nrc != 0) {
        errno = nrc;
        return -1;
      }
      return 0;
    }
  }
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
  {
    const int nrc = FsRename(src, dst, /*noreplace=*/false);
    if (nrc != kNsUnavailable) {
      if (nrc != 0) {
        errno = nrc;
        return -1;
      }
      return 0;
    }
  }
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
  {
    std::vector<ns::DirEnt> ents;
    const int nrc = FsReaddir(path, &ents);
    if (nrc != kNsUnavailable) {
      if (nrc != 0) {
        errno = nrc;
        return -1;
      }
      out->clear();
      out->reserve(ents.size());
      for (auto &e : ents) out->emplace_back(std::move(e.name));
      return 0;
    }
  }
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
  // A namespace-held file has no module handle: its size travels through
  // the size table, so close(2) has nothing to send.
  if (handle == 0) return 0;
  auto t = AsyncClose(handle);
  t.Wait();
  return (t->GetReturnCode() == 0) ? 0 : -1;
}


}  // namespace clio::cte::filesystem
