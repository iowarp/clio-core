/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */
#ifndef CLIO_CTE_FILESYSTEM_FILESYSTEM_CLIENT_H_
#define CLIO_CTE_FILESYSTEM_FILESYSTEM_CLIENT_H_

#include <clio_cte/core/core_client.h>
#include <atomic>
#include "clio_cte/filesystem/api.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
// The descriptor layer below is POSIX-SHAPED but not POSIX-only: nothing
// in it makes a system call. posix_compat.h supplies the vocabulary it is
// written in (FsSsize/FsOff/FsSsize) on every platform.
#include "clio_cte/filesystem/api.h"
#include "clio_cte/filesystem/posix_compat.h"
#ifndef _WIN32
#include <unistd.h>
#endif
#include <cerrno>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <clio_cte/filesystem/filesystem_tasks.h>
#include <clio_cte/stream/stream_client.h>
#include <clio_cte/filesystem/fs_shard.h>
#include <clio_cte/filesystem/shm_fs_cache.h>

namespace clio::cte::filesystem {

// ---- clio:: path marker (opt-in interception) -----------------------------
//
// Interception is opt-in by a "clio::" path component; the marker is stripped
// before the path is used as a CTE tag name. This lived in the POSIX adapter,
// which meant every other interceptor (STDIO, MPI-IO, libfuse, the HDF5 VFD)
// had to link that adapter to ask "is this path mine?" -- a question about the
// filesystem, not about POSIX.

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

/** CTE-issued descriptors start here so they never collide with kernel fds. */
static constexpr int kCfsFdBase = 8192;
/** Preferred block size reported by stat (matches the chimod page size). */
static constexpr size_t kCfsBlkSize = 1024 * 1024;
/** Synthetic device id -- same for every clio:: file. */
static constexpr dev_t kClioStDev = static_cast<dev_t>(0xC110);

/**
 * Filesystem client — the single API every interceptor (POSIX, STDIO,
 * libfuse, HDF5 VFD, MPI-IO) calls. Inherits the full CTE core client (so all
 * blob/tag/target operations remain available) and adds POSIX-shaped
 * filesystem operations routed through the filesystem chimod, which owns the
 * path->tag mapping, page-blob I/O, and per-file logical-size metadata.
 */
class Client : public clio::cte::core::Client {
 public:
  Client() = default;
  explicit Client(const clio::run::PoolId &fs_pool_id) { Init(fs_pool_id); }

#if CTP_IS_HOST
  // Copyable by hand, because the descriptor table below carries a std::mutex
  // and the compiler-generated copies would be deleted. The base client is
  // copyable by contract, and callers do copy it, so losing that here would
  // be a silent API break.
  //
  // The lock is on the SOURCE: we are reading its table. Our own mutex is
  // freshly constructed -- a lock is not state worth copying, and copying one
  // would be undefined anyway.
  Client(const Client &other)
      : clio::cte::core::Client(other),
        data_cte_(other.data_cte_),
        stream_(other.stream_) {
    std::lock_guard<std::mutex> g(other.fd_mu_);
    shm_fs_root_ = other.shm_fs_root_;
    fds_ = other.fds_;
    next_fd_ = other.next_fd_;
  }

  Client &operator=(const Client &other) {
    if (this == &other) {
      return *this;
    }
    clio::cte::core::Client::operator=(other);
    data_cte_ = other.data_cte_;
    stream_ = other.stream_;
    // Both tables are touched, so both locks are taken -- ordered by address
    // so two threads assigning A=B and B=A cannot deadlock.
    std::mutex *first = &fd_mu_;
    std::mutex *second = &other.fd_mu_;
    if (first > second) {
      std::swap(first, second);
    }
    std::lock_guard<std::mutex> g1(*first);
    std::lock_guard<std::mutex> g2(*second);
    shm_fs_root_ = other.shm_fs_root_;
    fds_ = other.fds_;
    next_fd_ = other.next_fd_;
    return *this;
  }
#endif

#if CTP_IS_HOST
  // =========================================================================
  // issue #817: shared-memory attribute cache (client side).
  //
  // Same contract as the CTE core's cache: strictly an OPTIMIZATION, every
  // accessor reports failure rather than guessing, and `false` means "ask the
  // runtime", never "the file does not exist".
  // =========================================================================

  /** Attach this filesystem pool's cache root via the segment directory. */
  bool AttachShmCache() {
    shm_fs_root_ = nullptr;
    auto *ipc = CLIO_CPU_IPC;
    if (ipc == nullptr) {
      return false;
    }
    auto *alloc = ipc->GetMetadataAllocator();
    auto *dir = ipc->GetMetadataDirectory();
    if (alloc == nullptr || dir == nullptr) {
      // No metadata segment (e.g. TCP client, remote node).
      HLOG(kDebug, "[#817] AttachShmCache: no metadata segment (alloc={}, dir={})",
           static_cast<const void *>(alloc), static_cast<const void *>(dir));
      return false;
    }
    clio::run::u64 root_off = dir->FindRoot(pool_id_.ToU64());
    if (root_off == 0) {
      // Either this pool is not caching, or its chimod has not registered its
      // root YET -- a client that raced pool creation must be able to attach
      // later, which is why callers retry rather than latching a failure.
      HLOG(kDebug, "[#817] AttachShmCache: no root for fs pool {} ({} entries)",
           pool_id_.ToString(), dir->num_entries_);
      return false;
    }
    auto *root = reinterpret_cast<ShmFsCacheRoot *>(
        reinterpret_cast<char *>(alloc) + root_off);
    // Refuse anything unrecognized: the cache is derived state, so declining
    // is always safe, while guessing at a layout is not.
    if (root->ready_ != 1 ||
        root->version_ != ShmFsCacheRoot::kLayoutVersion) {
      return false;
    }
    shm_fs_root_ = root;
    return true;
  }

  bool HasShmCache() const { return shm_fs_root_ != nullptr; }

  /**
   * Zero-IPC path lookup.
   *
   * @return true if a consistent record was read. false means "not cached /
   *         could not read consistently" -- fall back to the RPC path.
   */
  /** True once the runtime's mirror dropped ANY record for capacity: a miss
   *  then proves nothing and authoritative negatives must not fire. */
  bool MirrorSaturated() const {
    return shm_fs_root_ != nullptr &&
           shm_fs_root_->overflow_.load(std::memory_order_acquire) != 0;
  }

  /**
   * @param absent if non-null, set true only when a false return PROVES the
   *        path has no record (or only a stale-generation one); a torn or
   *        contended read leaves it false. Authoritative negatives must use
   *        it: a record being rewritten (a size push right after close) used
   *        to read as missing, and a COMPLETE parent turned that into ENOENT
   *        for a file that existed ("unable to stat just-written file").
   */
  bool TryGetFileRecordShm(const std::string &path, ShmFileRecord *out,
                           bool *absent = nullptr) const {
    if (absent != nullptr) *absent = false;
    if (shm_fs_root_ == nullptr || out == nullptr) {
      return false;
    }
    const std::string key = shm_fs_root_->Key(path);
    if (!shm_fs_root_->path_to_file_.TryGetBytes(key.data(), key.size(),
                                                 out, 16, absent)) {
      return false;
    }
    // A FILE record from an older namespace generation may sit under a stale
    // path key (its directory was renamed): treat as absent. Directory
    // records stay for IsDir purposes (a moved-away dir name SHOULD read
    // negative, and dir self-stats never serve from the mirror) — but their
    // COMPLETE claim dies with the generation: after the bump every file
    // record reads absent, so a still-complete parent would turn that into
    // authoritative ENOENT for REAL files across the whole tree (fsstress's
    // one dir rename made rm -r skip everything it then could not rmdir).
    if (out->nsgen_ !=
        shm_fs_root_->nsgen_.load(std::memory_order_relaxed)) {
      if (!(out->flags_ & kShmFileIsDir)) {
        if (absent != nullptr) *absent = true;  // stale name: no record
        return false;
      }
      out->flags_ &= ~kShmDirComplete;
    }
    return true;
  }
#endif  // CTP_IS_HOST

#if CTP_IS_HOST
  /**
   * Ask the local filesystem container for one of the pools it uses
   * (Monitor query; the filesystem's config is the only source of pool ids).
   * @param key "next_pool" (the data chain its pages go through) or
   *        "stream_pool" (where file sizes live)
   * @param out receives the pool id on success
   * @return true when the container answered with a valid pool id
   */
  bool QueryPoolId(const std::string &key, clio::run::PoolId *out) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<MonitorTask>(clio::run::CreateTaskId(), pool_id_,
                                          clio::run::PoolQuery::Local(), key);
    auto fut = ipc->Send(task);
    fut.Wait();
    if (fut->GetReturnCode() != 0 || fut->results_.empty()) {
      return false;
    }
    const std::string v = fut->results_.begin()->second;
    const size_t dot = v.find('.');
    if (dot == std::string::npos) {
      return false;
    }
    *out = clio::run::PoolId(
        static_cast<clio::run::u32>(std::strtoul(v.c_str(), nullptr, 10)),
        static_cast<clio::run::u32>(
            std::strtoul(v.c_str() + dot + 1, nullptr, 10)));
    return !out->IsNull();
  }

  /**
   * The pool this filesystem stores page blobs through (the top of its
   * interposition chain).
   * @param out receives the pool id on success
   * @return true when the filesystem answered
   */
  bool GetDataPoolId(clio::run::PoolId *out) {
    return QueryPoolId("next_pool", out);
  }

  /**
   * Bind the data-path and stream clients to the pools this filesystem
   * pool is configured with. Every CTE call a clio-fs client makes (page
   * I/O, fsync, capacity) must take the same chain the filesystem does, or
   * it bypasses the caches and replication above the cores.
   * @return true when both pools were learned
   */
  bool BindPools() {
    clio::run::PoolId data_pool;
    clio::run::PoolId stream_pool;
    if (!QueryPoolId("next_pool", &data_pool) ||
        !QueryPoolId("stream_pool", &stream_pool)) {
      return false;
    }
    data_cte_.Init(data_pool);
    // Connect the data client to the shm cache module when the data pool
    // fronts it (the keys come from the CTE Create; a lookup, no create).
    // Otherwise it stays on the task path.
    data_cte_.ConnectCache();
    stream_.Init(stream_pool);
    return true;
  }

  /** CTE client bound to this filesystem's data chain (see BindPools). */
  clio::cte::core::Client *Cte() { return &data_cte_; }

  /** Stream client bound to this filesystem's stream pool. */
  clio::cte::stream::Client *Stream() { return &stream_; }

  /**
   * Ask the local filesystem container for its file-data score rule (YAML
   * `data_score` / `data_score_head_pages`). A client that puts page blobs
   * itself applies it the way the container's own Write does:
   * `policy.ScoreOf(offset / kFsPageSize)` is the score of that page's put.
   * @param out receives the rule on success (untouched otherwise)
   * @return true when the home answered with a well-formed rule
   */
  bool GetPageScorePolicy(clio::cte::core::PageScorePolicy *out) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<MonitorTask>(clio::run::CreateTaskId(), pool_id_,
                                          clio::run::PoolQuery::Local(),
                                          std::string("data_score"));
    auto fut = ipc->Send(task);
    fut.Wait();
    if (fut->GetReturnCode() != 0 || fut->results_.empty()) {
      return false;
    }
    return DecodePageScore(fut->results_.begin()->second, out);
  }

  /** Create/initialize the filesystem container over a CTE core pool. */  /** Create/initialize the filesystem container over a CTE core pool. */
  clio::run::Future<CreateTask> AsyncCreate(const clio::run::PoolQuery &pool_query,
                                      const std::string &pool_name,
                                      const clio::run::PoolId &custom_pool_id,
                                      const FilesystemConfig &params) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<CreateTask>(
        clio::run::CreateTaskId(), clio::run::kAdminPoolId, pool_query,
        FilesystemConfig::chimod_lib_name, pool_name, custom_pool_id, this,
        params);
    return ipc->Send(task);
  }

  clio::run::Future<OpenTask> AsyncOpen(const std::string &path, clio::run::u32 flags,
                                  clio::run::u32 mode = 0644) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<OpenTask>(clio::run::CreateTaskId(), pool_id_,
                                       FsPathQuery(), path, flags,
                                       mode);
    return ipc->Send(task);
  }

  /**
   * fsync the namespace log on every container (see SyncMetaTask).
   * @return future; rc != 0 if any container failed to sync
   */
  clio::run::Future<SyncMetaTask> AsyncSyncMeta() {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<SyncMetaTask>(clio::run::CreateTaskId(), pool_id_,
                                           clio::run::PoolQuery::Broadcast());
    return ipc->Send(task);
  }

  /**
   * Tag-keyed size advance (see AdvanceSizeTask).
   * @param tag_packed the file's packed TagId
   * @param size the new size (never lowers it)
   * @param mtime_ns wall time (ns) of the last write the size covers, the
   *        file's new mtime; 0 = the time the push lands
   */
  clio::run::Future<AdvanceSizeTask> AsyncAdvanceSize(
      clio::run::u64 tag_packed, clio::run::u64 size,
      clio::run::u64 mtime_ns = 0) {
    // Local: the local container forwards to the inode's live home (the
    // encoded one, or its successor while that node is down).
    return AsyncAdvanceSizeAt(clio::run::PoolQuery::Local(), tag_packed, size,
                              0u, mtime_ns);
  }

  /**
   * Size advance / append reservation sent to an explicit container (the
   * runtime's forwarding step; clients use AsyncAdvanceSize /
   * AsyncReserveAppend).
   * @param query where to run it
   * @param tag_packed the file's packed TagId
   * @param size new size (raise) or length (reserve)
   * @param reserve nonzero for an append reservation
   * @param mtime_ns raise mode: the file's new mtime (0 = when it lands)
   */
  clio::run::Future<AdvanceSizeTask> AsyncAdvanceSizeAt(
      const clio::run::PoolQuery &query, clio::run::u64 tag_packed,
      clio::run::u64 size, clio::run::u32 reserve,
      clio::run::u64 mtime_ns = 0) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<AdvanceSizeTask>(clio::run::CreateTaskId(),
                                              pool_id_, query, tag_packed,
                                              size, reserve, mtime_ns);
    return ipc->Send(task);
  }

  /**
   * Reserve `len` bytes at the end of a file, cluster-wide: the home grows
   * the logical size atomically and reports the old end in old_size_ (the
   * offset an O_APPEND write must use).
   * @param tag_packed the file's packed TagId
   * @param len        bytes being appended
   */
  clio::run::Future<AdvanceSizeTask> AsyncReserveAppend(
      clio::run::u64 tag_packed, clio::run::u64 len) {
    return AsyncAdvanceSizeAt(clio::run::PoolQuery::Local(), tag_packed, len,
                              1u);
  }

  /** Batched sieve-flushed creation (see MultiCreateTask). */
  clio::run::Future<MultiCreateTask> AsyncMultiCreate(
      const std::string &packed) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<MultiCreateTask>(clio::run::CreateTaskId(),
                                              pool_id_,
                                              clio::run::PoolQuery::Local(),
                                              packed);
    return ipc->Send(task);
  }

  clio::run::Future<CloseTask> AsyncClose(clio::run::u64 handle,
                                          clio::run::u64 advance_size = 0) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<CloseTask>(clio::run::CreateTaskId(), pool_id_,
                                        FsHandleQuery(handle), handle,
                                        advance_size);
    return ipc->Send(task);
  }

  /** Fire-and-forget close: the task carries TASK_FIRE_AND_FORGET, so no
   *  response is produced and the returned future is complete immediately.
   *  close(2) is not a durability point — data durability lives in the
   *  awaited writes / fsync — and Close's only output is server-side handle
   *  bookkeeping, so metadata-heavy workloads (one close per file) need not
   *  pay a full round trip for it. */
  void AsyncCloseDetached(clio::run::u64 handle,
                          clio::run::u64 advance_size = 0) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<CloseTask>(clio::run::CreateTaskId(), pool_id_,
                                        FsHandleQuery(handle), handle,
                                        advance_size);
    // Flag must be set before Send (same ordering rule as the PutBlob
    // staging flags).
    task.get()->task_flags_.SetBits(TASK_FIRE_AND_FORGET);
    ipc->Send(task);
  }

  clio::run::Future<ReadTask> AsyncRead(clio::run::u64 handle, clio::run::u64 offset,
                                  clio::run::u64 size, ctp::ipc::ShmPtr<> data) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<ReadTask>(clio::run::CreateTaskId(), pool_id_,
                                       FsHandleQuery(handle), handle, offset,
                                       size, data);
    return ipc->Send(task);
  }

  clio::run::Future<WriteTask> AsyncWrite(clio::run::u64 handle, clio::run::u64 offset,
                                    clio::run::u64 size, ctp::ipc::ShmPtr<> data) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<WriteTask>(clio::run::CreateTaskId(), pool_id_,
                                        FsHandleQuery(handle), handle, offset,
                                        size, data);
    return ipc->Send(task);
  }

  clio::run::Future<GetattrTask> AsyncGetattr(const std::string &path) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<GetattrTask>(clio::run::CreateTaskId(), pool_id_,
                                          FsPathQuery(), path);
    return ipc->Send(task);
  }

  clio::run::Future<TruncateTask> AsyncTruncate(const std::string &path,
                                          clio::run::u64 new_size,
                                          clio::run::u64 tag_packed = 0,
                                          clio::run::u64 old_extent = 0) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<TruncateTask>(clio::run::CreateTaskId(), pool_id_,
                                           FsPathQuery(), path,
                                           new_size, tag_packed, old_extent);
    return ipc->Send(task);
  }

  clio::run::Future<UnlinkTask> AsyncUnlink(const std::string &path) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<UnlinkTask>(clio::run::CreateTaskId(), pool_id_,
                                         FsPathQuery(), path);
    return ipc->Send(task);
  }

  clio::run::Future<UtimensTask> AsyncUtimens(const std::string &path,
                                              clio::run::u64 atime_ns,
                                              clio::run::u64 mtime_ns,
                                              clio::run::u32 flags) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<UtimensTask>(clio::run::CreateTaskId(), pool_id_,
                                          FsPathQuery(), path,
                                          atime_ns, mtime_ns, flags);
    return ipc->Send(task);
  }

  /**
   * Create every missing ancestor directory of `path` (mkdir -p of its
   * parent). Existing directories are left alone.
   * @param path a normalized clio-fs path
   */
  void MakeParents(const std::string &path);

  /** Fire-and-forget utimens (TASK_FIRE_AND_FORGET; see AsyncCloseDetached).
   *  Timestamp stamping is pure metadata with no caller-visible output — the
   *  kernel's writeback SETATTR issues one per dirtied file, and paying a
   *  round trip for each put a wait on every file of a checkout. */
  void AsyncUtimensDetached(const std::string &path, clio::run::u64 atime_ns,
                            clio::run::u64 mtime_ns, clio::run::u32 flags) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<UtimensTask>(clio::run::CreateTaskId(), pool_id_,
                                          FsPathQuery(), path,
                                          atime_ns, mtime_ns, flags);
    task.get()->task_flags_.SetBits(TASK_FIRE_AND_FORGET);
    ipc->Send(task);
  }

  clio::run::Future<ChownTask> AsyncChown(const std::string &path,
                                          clio::run::u32 uid,
                                          clio::run::u32 gid,
                                          clio::run::u32 mode = 0xFFFFFFFFu) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<ChownTask>(clio::run::CreateTaskId(), pool_id_,
                                        FsPathQuery(), path, uid,
                                        gid, mode);
    return ipc->Send(task);
  }

  // chmod reuses the ChownTask (per-file mode override) with uid/gid left
  // unchanged, avoiding a separate RPC method for a single stored field.
  clio::run::Future<ChownTask> AsyncChmod(const std::string &path,
                                          clio::run::u32 mode) {
    return AsyncChown(path, 0xFFFFFFFFu, 0xFFFFFFFFu, mode);
  }

  clio::run::Future<MkdirTask> AsyncMkdir(const std::string &path) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<MkdirTask>(clio::run::CreateTaskId(), pool_id_,
                                        FsPathQuery(), path);
    return ipc->Send(task);
  }

  clio::run::Future<RmdirTask> AsyncRmdir(const std::string &path) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<RmdirTask>(clio::run::CreateTaskId(), pool_id_,
                                        FsPathQuery(), path);
    return ipc->Send(task);
  }

  clio::run::Future<RenameTask> AsyncRename(const std::string &src,
                                      const std::string &dst) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<RenameTask>(clio::run::CreateTaskId(), pool_id_,
                                         FsPathQuery(), src, dst);
    return ipc->Send(task);
  }

  clio::run::Future<LinkTask> AsyncLink(const std::string &target,
                                  const std::string &link) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<LinkTask>(clio::run::CreateTaskId(), pool_id_,
                                       FsPathQuery(), target, link);
    return ipc->Send(task);
  }

  clio::run::Future<SymlinkTask> AsyncSymlink(const std::string &target,
                                              const std::string &path) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<SymlinkTask>(clio::run::CreateTaskId(), pool_id_,
                                          FsPathQuery(), target,
                                          path);
    return ipc->Send(task);
  }

  clio::run::Future<ReadlinkTask> AsyncReadlink(const std::string &path) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<ReadlinkTask>(clio::run::CreateTaskId(), pool_id_,
                                           FsPathQuery(), path);
    return ipc->Send(task);
  }

  clio::run::Future<SetxattrTask> AsyncSetxattr(const std::string &path,
                                                const std::string &name,
                                                const std::string &value,
                                                clio::run::u32 flags) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<SetxattrTask>(clio::run::CreateTaskId(), pool_id_,
                                           FsPathQuery(), path,
                                           name, value, flags);
    return ipc->Send(task);
  }

  clio::run::Future<GetxattrTask> AsyncGetxattr(const std::string &path,
                                                const std::string &name) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<GetxattrTask>(clio::run::CreateTaskId(), pool_id_,
                                           FsPathQuery(), path,
                                           name);
    return ipc->Send(task);
  }

  clio::run::Future<ListxattrTask> AsyncListxattr(const std::string &path) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<ListxattrTask>(clio::run::CreateTaskId(), pool_id_,
                                            FsPathQuery(), path);
    return ipc->Send(task);
  }

  clio::run::Future<RemovexattrTask> AsyncRemovexattr(const std::string &path,
                                                      const std::string &name) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<RemovexattrTask>(clio::run::CreateTaskId(),
                                              pool_id_,
                                              FsPathQuery(),
                                              path, name);
    return ipc->Send(task);
  }

  /**
   * Internal: run one ShardOp on the container `pool_query` names (the
   * owner of the directory or inode the request addresses).
   * @param op FsShardOp code
   * @param req encoded request
   * @param pool_query owner container
   * @param period_us nonzero = periodic task with this period
   */
  clio::run::Future<ShardOpTask> AsyncShardOp(
      clio::run::u32 op, const std::string &req,
      const clio::run::PoolQuery &pool_query, double period_us = 0.0) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<ShardOpTask>(clio::run::CreateTaskId(), pool_id_,
                                          pool_query, op, req);
    if (period_us > 0.0) {
      task->SetPeriod(period_us, clio::run::kMicro);
      task->SetFlags(TASK_PERIODIC);
    }
    return ipc->Send(task);
  }

  clio::run::Future<ReaddirTask> AsyncReaddir(const std::string &path) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<ReaddirTask>(clio::run::CreateTaskId(), pool_id_,
                                          FsPathQuery(), path);
    return ipc->Send(task);
  }

  /**
   * List one chunk of a directory in offset order (Method::kReaddirRange).
   * @param path directory path
   * @param start_off first offset wanted: 0 for the start, else a previous
   *        chunk's next_off_ or any offset to seek to
   * @param max_entries entries after which the chunk ends (it may run a
   *        little over: chunks are never cut inside a table block)
   * @return future of the chunk; next_off_ is kDirOffEnd at the end
   */
  clio::run::Future<ReaddirRangeTask> AsyncReaddirRange(
      const std::string &path, clio::run::u64 start_off,
      clio::run::u32 max_entries = 8192) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<ReaddirRangeTask>(
        clio::run::CreateTaskId(), pool_id_, FsPathQuery(), path, start_off,
        max_entries);
    return ipc->Send(task);
  }

  clio::run::Future<StatSizeTask> AsyncStatSize(const std::string &path) {
    auto *ipc = CLIO_CPU_IPC;
    auto task = ipc->NewTask<StatSizeTask>(clio::run::CreateTaskId(), pool_id_,
                                           FsPathQuery(), path);
    return ipc->Send(task);
  }

  // =========================================================================
  // Byte-oriented filesystem I/O with POSIX semantics (issues #817, #862).
  //
  // POSIX-SHAPED, but not POSIX-only: nothing here makes a system call. The
  // layer is a descriptor table and a set of seek offsets over the task API,
  // so what tied it to POSIX was its vocabulary -- ssize_t, off_t, O_SYNC --
  // rather than its behaviour. That vocabulary comes from posix_compat.h now
  // (FsSsize/FsOff), and the layer builds on Windows too. Offsets are FsOff
  // and deliberately not off_t: MSVC's off_t is 32 bits and would cap every
  // offset here at 2 GiB.
  //
  // THIS is the interface an interceptor calls. An adapter (POSIX, STDIO,
  // MPI-IO, libfuse) should own nothing but its handle table and its seek
  // offsets: no pending-write queue, no staging allocator, no drain
  // bookkeeping, no fast-path of its own. Every one of those used to live in
  // the POSIX adapter, which meant the next adapter either duplicated them or
  // silently did without.
  //
  // Writes are synchronous: write(2) returns once the filesystem chimod has
  // the bytes, and a failure is reported by that write(2) itself.
  // =========================================================================

  /**
   * write(2) at an explicit offset. Copies `buf` into a runtime buffer (the
   * caller may reuse its buffer the instant write(2) returns), sends the
   * Write task and waits for it.
   *
   * @param handle chimod handle of the open file
   * @param path bare path of the file (unused; kept for the descriptor API)
   * @param off file offset
   * @param buf source bytes
   * @param count bytes to write
   * @param sync O_SYNC/O_DSYNC (writes already complete before returning)
   * @return bytes written, or -1 with errno set (ENOSPC for a full store,
   *         ENOMEM when no buffer could be allocated, else EIO).
   */
  FsSsize Write(clio::run::u64 handle, const std::string &path,
                clio::run::u64 off, const void *buf, size_t count,
                bool sync = false) {
    (void)path;
    (void)sync;
    if (count == 0) {
      return 0;
    }
    auto *ipc = CLIO_IPC;
    ctp::ipc::FullPtr<char> staging = ipc->AllocateBuffer(count);
    if (staging.IsNull()) {
      errno = ENOMEM;
      return -1;
    }
    std::memcpy(staging.ptr_, buf, count);
    auto fut = AsyncWrite(handle, off, count,
                          ctp::ipc::ShmPtr<>(staging.shm_));
    fut.Wait();
    FsSsize ret;
    if (fut->GetReturnCode() == 0) {
      ret = static_cast<FsSsize>(fut->bytes_written_);
    } else {
      // The chimod reports a full store as ENOSPC; all else is EIO.
      errno = fut->GetReturnCode() == ENOSPC ? ENOSPC : EIO;
      ret = -1;
    }
    ipc->FreeBuffer(staging);
    return ret;
  }

  /**
   * read(2) at an explicit offset. Committed bytes come from shared memory
   * when the mirrors can serve them (no IPC); anything else is one task to
   * the chimod.
   *
   * @param handle chimod handle of the open file
   * @param path bare path of the file
   * @param off file offset
   * @param buf destination
   * @param count bytes to read
   * @return bytes read (possibly 0 at EOF), or -1 with errno set.
   */
  FsSsize Read(clio::run::u64 handle, const std::string &path,
               clio::run::u64 off, void *buf, size_t count) {
    if (count == 0) {
      return 0;
    }
    FsSsize fast = TryReadShm(path, off, buf, count);
    if (fast >= 0) {
      return fast;
    }
    auto *ipc = CLIO_IPC;
    ctp::ipc::FullPtr<char> shm = ipc->AllocateBuffer(count);
    if (shm.IsNull()) {
      errno = ENOMEM;
      return -1;
    }
    auto t = AsyncRead(handle, off, count, ctp::ipc::ShmPtr<>(shm.shm_));
    t.Wait();
    FsSsize ret;
    if (t->GetReturnCode() == 0) {
      size_t got = static_cast<size_t>(t->bytes_read_);
      if (got > 0) {
        std::memcpy(buf, shm.ptr_, got);
      }
      ret = static_cast<FsSsize>(got);
    } else {
      errno = EIO;
      ret = -1;
    }
    ipc->FreeBuffer(shm);
    return ret;
  }

  /**
   * stat(2)-shaped query: existence and logical size.
   *
   * @param path bare path
   * @param exists receives whether the file is there
   * @param size receives the logical size
   * @return false when the query itself failed (ask again); *exists says
   *         whether the file is there.
   */
  bool GetAttr(const std::string &path, bool *exists, clio::run::u64 *size) {
    // Serve from the filesystem's shared-memory mirror when it has the path:
    // stat(2) is the most frequent metadata call a filesystem sees, and the
    // record the runtime publishes carries both answers. Writes are
    // synchronous, so the published size already includes every write(2)
    // that has returned.
    if (HasShmCache() || AttachShmCache()) {
      ShmFileRecord rec;
      if (TryGetFileRecordShm(path, &rec)) {
        *exists = rec.Exists();
        *size = rec.Exists() ? rec.size_ : 0;
        ShmStatHits().fetch_add(1, std::memory_order_relaxed);
        return true;
      }
    }
    ShmStatMisses().fetch_add(1, std::memory_order_relaxed);
    auto t = AsyncGetattr(path);
    t.Wait();
    if (t->GetReturnCode() != 0) {
      return false;
    }
    *exists = (t->exists_ != 0);
    *size = t->size_;
    return true;
  }

  /**
   * Logical size of `path` (0 when absent).
   * @param path bare path
   * @param size receives the size
   * @return false when the query itself failed
   */
  bool GetSize(const std::string &path, clio::run::u64 *size) {
    bool exists = false;
    if (!GetAttr(path, &exists, size)) {
      return false;
    }
    if (!exists) {
      *size = 0;
    }
    return true;
  }

  // =========================================================================
  // POSIX descriptor layer.
  //
  // Formerly adapter/cfs/CfsIo. It is here because a descriptor table, an
  // O_APPEND seek rule and a synthesized struct stat are FILESYSTEM state, not
  // POSIX-interceptor state: the STDIO, MPI-IO and HDF5-VFD interceptors all
  // need the same table, and while it lived in the POSIX adapter they had to
  // link that adapter to get at it. An interceptor should translate its own
  // API into these calls and hold nothing.
  //
  // The table is process-wide and static rather than a member because
  // clio::cte::core::Client is copyable by contract; a mutex member would
  // silently break that.
  // =========================================================================

  struct OpenFile {
    clio::run::u64 handle = 0;  ///< chimod handle
    std::string path;           ///< bare (stripped) path == CTE tag name
    clio::run::u64 off = 0;     ///< current seek offset
    int flags = 0;              ///< open() flags
  };

  /** A path is intercepted iff it carries the clio:: marker. */
  static bool IsPathTracked(const std::string &path) {
    return HasClioPrefix(path);
  }

  /** O_SYNC/O_DSYNC keep the blocking behaviour: the caller asked for
   *  durability over latency, and honouring that is the point of the flag. */
  static bool IsSyncFd(int flags) {
#ifdef O_DSYNC
    if (flags & O_DSYNC) return true;
#endif
#ifdef O_SYNC
    return (flags & O_SYNC) != 0;
#else
    /* MSVC defines neither, so no caller there can have requested it. */
    (void)flags;
    return false;
#endif
  }

  /** Whether fd was issued by us (and is still open). */
  bool IsFdTracked(int fd) {
    if (fd < kCfsFdBase) {
      return false;
    }
    std::lock_guard<std::mutex> g(fd_mu_);
    return fds_.find(fd) != fds_.end();
  }

  /** The chimod handle behind an fd, or 0 if untracked. */
  clio::run::u64 HandleOf(int fd) {
    std::lock_guard<std::mutex> g(fd_mu_);
    auto it = fds_.find(fd);
    return it == fds_.end() ? 0 : it->second.handle;
  }

  /** Copy out a descriptor's state, or set EBADF and return false. */
  bool LookupFd(int fd, OpenFile *out) {
    std::lock_guard<std::mutex> g(fd_mu_);
    auto it = fds_.find(fd);
    if (it == fds_.end()) {
      errno = EBADF;
      return false;
    }
    *out = it->second;
    return true;
  }

  /** Advance a descriptor's seek offset after a successful read/write. */
  void AdvanceFd(int fd, clio::run::u64 n) {
    std::lock_guard<std::mutex> g(fd_mu_);
    auto it = fds_.find(fd);
    if (it != fds_.end()) {
      it->second.off += n;
    }
  }

  /** Bind the filesystem pool on first tracked use. */
  static bool EnsureInit();

  /**
   * open(2) for a `clio::`-prefixed path (the interception adapters). With
   * O_CREAT, missing parent directories are created first: the path mirrors
   * a host path whose directories exist on the host.
   * @param raw_path path, optionally with the `clio::` prefix
   * @param flags open flags
   * @param mode creation mode
   * @return a descriptor, or -1 with errno set
   */
  int OpenFd(const std::string &raw_path, int flags, int mode);
  FsSsize ReadFd(int fd, void *buf, size_t count);
  FsSsize WriteFd(int fd, const void *buf, size_t count);
  FsSsize PreadFd(int fd, void *buf, size_t count, FsOff offset);
  FsSsize PwriteFd(int fd, const void *buf, size_t count, FsOff offset);
  FsOff SeekFd(int fd, FsOff offset, int whence);
  FsOff TellFd(int fd);
  FsOff SizeFd(int fd);
  int CloseFd(int fd);
  /** fsync(2): make the file durable (SyncDurable). Writes are synchronous,
   *  so there is nothing to drain first. */
  int SyncFd(int fd);

  // ---- fsync(2) durability (shared by the interceptors and FUSE) ----

  /**
   * Whether CTE performance.fsync_mode is "deferred", as learned from the
   * first SyncTag reply: fsync then skips the device sync and leaves
   * durability to the periodic flushes.
   * @return true once a reply said "deferred"
   */
  static bool FsyncDeferred();

  /**
   * Make one CTE tag durable: SyncTag, broadcast to every core container.
   * @param tag the tag (null is a no-op)
   * @param lost_node out (optional): a container's node was down
   * @param liveness_change_ns out (optional): wall-clock ns of the last
   *        liveness change the sync saw (SyncTagTask::liveness_change_ns_)
   * @return 0, -ENOSPC when no persistent tier had room, or -EIO
   */
  static int SyncTagDurable(const clio::cte::core::TagId &tag,
                            bool *lost_node = nullptr,
                            clio::run::u64 *liveness_change_ns = nullptr);

  /**
   * fsync(2)'s size step: fsync the size log at the file's stream home.
   * @param tag the file's tag (null, or an id without a home: nothing to do)
   * @return 0 or -EIO
   */
  static int SyncFileSize(const clio::cte::core::TagId &tag);

  /**
   * The CTE tag of a file or directory (its id is its inode number).
   * @param path the path
   * @param want_dir true: only a directory resolves
   * @return the tag, or null when the path does not resolve
   */
  clio::cte::core::TagId TagOfPath(const std::string &path, bool want_dir);

  /**
   * fsync(2)'s durability step: the file's blobs (pages and inode record),
   * its size, then the directory blocks naming it, so the bytes, the size
   * and the name all survive power loss.
   * @param tag the file's tag (null: only the directory)
   * @param dir the directory whose entries must be durable
   * @return 0, -ENOSPC or -EIO
   */
  int SyncDurable(const clio::cte::core::TagId &tag, const std::string &dir);

  /**
   * fsync every filesystem container's directory journal and system
   * records (SyncMeta, broadcast): the namespace half of fsync/fsyncdir.
   * A container on a lost node does not fail it (its journal is its own).
   * @return 0 or -EIO
   */
  int SyncMetaDurable();
  int FtruncateFd(int fd, FsOff length);
  int TruncatePath(const std::string &raw_path, FsOff length);
  int RemovePath(const std::string &raw_path);
  int RenamePath(const std::string &raw_src, const std::string &raw_dst);
  int ReaddirPath(const std::string &raw_path, std::vector<std::string> *out);

  /** Stable 64-bit synthetic inode from the bare path. */
  static uint64_t SyntheticInode(const std::string &path) {
    size_t h = std::hash<std::string>{}(path);
    return static_cast<uint64_t>(h) ^ (static_cast<uint64_t>(h) << 32);
  }

  template <typename StatT>
  static void FillStat(StatT *buf, const std::string &path,
                       clio::run::u64 size) {
    std::memset(buf, 0, sizeof(StatT));
    buf->st_dev = kClioStDev;
    buf->st_ino = SyntheticInode(path);
    buf->st_mode = S_IFREG | 0644;
    buf->st_nlink = 1;
    buf->st_uid = CTP_SYSTEM_INFO->uid_;
    buf->st_gid = CTP_SYSTEM_INFO->gid_;
    // Cast to whatever the caller's own struct declares rather than to fixed
    // POSIX types: this is a template instantiated with their stat struct.
    buf->st_size = static_cast<decltype(buf->st_size)>(size);
    // st_blksize/st_blocks are POSIX extensions that MSVC's struct stat simply
    // does not have. Detected on the caller's struct rather than on _WIN32,
    // because this is a template: a caller passing a struct that DOES declare
    // them still gets them filled, whatever the platform.
    if constexpr (requires { buf->st_blksize; }) {
      buf->st_blksize = static_cast<decltype(buf->st_blksize)>(kCfsBlkSize);
    }
    if constexpr (requires { buf->st_blocks; }) {
      // POSIX st_blocks counts 512-byte units; round up so non-empty != 0.
      buf->st_blocks =
          static_cast<decltype(buf->st_blocks)>((size + 511) / 512);
    }
  }

  /** fstat(2): fill *buf from the fd's chimod metadata. */
  template <typename StatT>
  int StatFd(int fd, StatT *buf) {
    OpenFile of;
    if (!LookupFd(fd, &of)) {
      return -1;
    }
    clio::run::u64 size = 0;
    if (!GetSize(of.path, &size)) {
      errno = EBADF;
      return -1;
    }
    FillStat(buf, of.path, size);
    return 0;
  }

  /** stat(2): fill *buf for a clio:: path; ENOENT if it does not exist. */
  template <typename StatT>
  int StatPath(const std::string &raw_path, StatT *buf) {
    // A stat can be the first clio:: call a process makes (IOR stats its
    // target before opening it), so it must bring the client up itself.
    if (!EnsureInit()) {
      std::memset(buf, 0, sizeof(StatT));
      errno = EIO;
      return -1;
    }
    std::string path = StripClioPrefix(raw_path);
    clio::run::u64 size = 0;
    bool exists = false;
    if (!GetAttr(path, &exists, &size) || !exists) {
      std::memset(buf, 0, sizeof(StatT));
      errno = ENOENT;
      return -1;
    }
    FillStat(buf, path, size);
    return 0;
  }

  /** Kill switch for the zero-IPC read fast path (diagnostics: forces every
   *  read through the authoritative RPC so mirror-record corruption can be
   *  discriminated from store corruption). */
  static bool ShmReadFastPathEnabled() {
    static const bool v = [] {
      const char *e = std::getenv("CLIO_CFS_SHM_READ");
      return e == nullptr || *e != '0';
    }();
    return v;
  }

  /**
   * Read committed bytes from shared memory, refusing rather than inventing
   * anything the shm cache does not hold (the HDF5 VFD's cache-over-file
   * semantics). Same as TryReadShm: a page the cache does not hold is never
   * zero-filled, because from this side "not cached" and "a hole" look alike.
   * @param path bare path
   * @param off file offset
   * @param buf destination
   * @param count bytes to read
   * @return bytes read, or -1 meaning "not fast-pathable, use the
   *         authoritative source"
   */
  FsSsize TryReadShmResident(const std::string &path, clio::run::u64 off,
                             void *buf, size_t count) {
    return TryReadShm(path, off, buf, count);
  }

  /**
   * Zero-IPC read of committed bytes: resolves the path to its tag and
   * logical size in this filesystem pool's shared-memory mirror, then copies
   * each page out of the data client's shm cache segment
   * (core::Client::TryGetCached).
   *
   * FsSsize, not ssize_t: MSVC has no ssize_t, and this layer's whole
   * vocabulary was ported to FsSsize/FsOff so it builds on Windows.
   *
   * @param path bare path
   * @param off file offset
   * @param buf destination
   * @param count bytes to read
   * @return bytes read (possibly 0 at EOF), or -1 meaning "not
   *         fast-pathable" -- NOT an error: mirror absent, path not mirrored,
   *         pending appends, no shm cache connected, or a page the cache does
   *         not hold. The caller then uses the task path, which is always
   *         correct.
   */
  FsSsize TryReadShm(const std::string &path, clio::run::u64 off, void *buf,
                     size_t count) {
    if (!ShmReadFastPathEnabled()) {
      return -1;
    }
    // Attach lazily, and RETRY when not yet attached: a client can come up
    // before its pool has been composed, and a one-shot attach would leave
    // that process on the task path forever.
    if (!HasShmCache() && !AttachShmCache()) {
      ShmReadMisses().fetch_add(1, std::memory_order_relaxed);
      return -1;
    }
    auto *cte = Cte();
    ShmFileRecord rec;
    if (cte->pool_id_.IsNull() || !cte->HasCache() ||
        !TryGetFileRecordShm(path, &rec) || !rec.IsFastPathable()) {
      ShmReadMisses().fetch_add(1, std::memory_order_relaxed);
      return -1;
    }
    // Clamp to the LOGICAL size: the number the chimod owns and the one POSIX
    // read() semantics are defined against. The runtime publishes a new size
    // at the end of the Write handler, before the write() that grew the file
    // returns, so a properly ordered reader never sees it too small.
    if (off >= rec.size_) {
      return 0;  // at or past EOF
    }
    const clio::run::u64 want = std::min<clio::run::u64>(count, rec.size_ - off);
    char *dst = static_cast<char *>(buf);
    clio::run::u64 done = 0;
    while (done < want) {
      const clio::run::u64 cur = off + done;
      const clio::run::u64 page_off = cur % kFsPageSize;
      const clio::run::u64 to_read =
          std::min<clio::run::u64>(kFsPageSize - page_off, want - done);
      if (!cte->TryGetCached(rec.tag_id_, PageName(cur), page_off, to_read,
                             dst + done)) {
        ShmReadMisses().fetch_add(1, std::memory_order_relaxed);
        return -1;
      }
      done += to_read;
    }
    ShmReadHits().fetch_add(1, std::memory_order_relaxed);
    return static_cast<FsSsize>(done);
  }

  // Zero-IPC read accounting. A silently-disabled fast path and a working one
  // differ only in latency, which is exactly the failure mode that is easiest
  // to misdiagnose as "the runtime is slow" -- so make the rate observable.
  static std::atomic<clio::run::u64> &ShmReadHits() {
    static std::atomic<clio::run::u64> v{0};
    return v;
  }
  static std::atomic<clio::run::u64> &ShmReadMisses() {
    static std::atomic<clio::run::u64> v{0};
    return v;
  }
  static std::atomic<clio::run::u64> &ShmStatHits() {
    static std::atomic<clio::run::u64> v{0};
    return v;
  }
  static std::atomic<clio::run::u64> &ShmStatMisses() {
    static std::atomic<clio::run::u64> v{0};
    return v;
  }

#endif  // CTP_IS_HOST

#if CTP_IS_HOST
 private:
  // Resolved address of the filesystem cache root in THIS process, or nullptr
  // when caching is unavailable. Not owned: the runtime owns the segment and
  // may drop the cache at any time, which is why every read is validated.
  ShmFsCacheRoot *shm_fs_root_ = nullptr;

  // Bound by BindPools from the filesystem pool's own config.
  clio::cte::core::Client data_cte_;
  clio::cte::stream::Client stream_;

  // ---- POSIX descriptor table ----
  // Guarded by fd_mu_. Held only around map lookups and insertions -- never
  // across a task Wait, so a slow chimod call cannot block an unrelated
  // open(2) or close(2).
  mutable std::mutex fd_mu_;
  std::unordered_map<int, OpenFile> fds_;
  int next_fd_ = kCfsFdBase;
#endif
};

// Process-wide filesystem client singleton, exported through THIS module's
// macro (clio_cte/filesystem/api.h) rather than the core's. Declared inside
// the namespace so it resolves as clio::cte::filesystem::g_fs_client
// (matching CLIO_CFS_CLIENT below and the definition in
// filesystem_client.cc), exactly like core's g_cte_client.
CLIO_CTE_FS_DEFINE_GLOBAL_PTR_VAR_H(clio::cte::filesystem::Client, g_fs_client);

/**
 * Initialize the filesystem client singleton: find the composed filesystem
 * pool by name (CLIO_CFS_POOL, default kCfsPoolName; a "major.minor" value
 * names the id directly) and bind the pools it is configured with. Creates
 * nothing and assumes no pool id.
 */
bool CLIO_CFS_CLIENT_INIT(const std::string &config_path = "",
                          const clio::run::PoolQuery &pool_query =
                              clio::run::PoolQuery::Dynamic());

}  // namespace clio::cte::filesystem

#define CLIO_CFS_CLIENT                                  \
  (&(*CTP_GET_GLOBAL_PTR_VAR(clio::cte::filesystem::Client, \
                             clio::cte::filesystem::g_fs_client)))

#endif  // CLIO_CTE_FILESYSTEM_FILESYSTEM_CLIENT_H_
