/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */
#ifndef CLIO_CTE_FILESYSTEM_SHM_FS_CACHE_H_
#define CLIO_CTE_FILESYSTEM_SHM_FS_CACHE_H_

#include <clio_cte/core/shm_metadata_cache.h>

#include <string>

namespace clio::cte::filesystem {

using clio::cte::core::ShmCacheAlloc;
using clio::cte::core::ShmCacheString;

/**
 * Flags on a cached filesystem record.
 *
 * The fast-path predicate is deliberately a whitelist of "nothing unusual is
 * going on": every flag below except kShmFileExists REFUSES the fast path. A
 * state this cache does not model yet must therefore be added as a refusing
 * flag, never left to be silently mistaken for a plain file.
 */
enum ShmFileFlags : clio::run::u32 {
  /** The record describes a live path. A record without this is a tombstone. */
  kShmFileExists = 1u << 0,
  /** Directory, not a regular file. Never payload-readable. */
  kShmFileIsDir = 1u << 1,
  /**
   * The file has appends staged but not yet merged into its tail (see
   * Runtime::Append). Until the AppendSequence pipeline drains them the bytes
   * live under the staging tag, not under this file's page blobs, and the
   * tracked size is explicitly best-effort -- so neither size nor pages may be
   * trusted for a direct read.
   */
  kShmFilePendingAppend = 1u << 2,
  /** Catch-all refusal for states the fast path does not model. */
  kShmFileNoFastPath = 1u << 3,
};

/** Sentinel for "no stored override" in the mode/uid/gid fields. */
static constexpr clio::run::u32 kShmFileNoOverride = 0xFFFFFFFFu;

/**
 * Cached per-path filesystem metadata (issue #817).
 *
 * WHY THIS EXISTS SEPARATELY FROM ShmTagRecord: the filesystem chimod owns a
 * LOGICAL size (FileInfo::size_) that is not the tag's physical size. They
 * diverge after ftruncate-grow, sparse writes and staged appends, and POSIX
 * read semantics -- short read at EOF, zeros inside a hole -- depend on the
 * logical one. A client that clamped against the physical size would return
 * the wrong bytes, so the fs number has to be published in its own right.
 *
 * POD ONLY, for the same reason as ShmBlobRecord: the map's reader copies the
 * whole record out between two reads of the slot generation, so an
 * allocator-owned member here would defeat the seqlock. The PATH is the map
 * key and is therefore not duplicated in the record.
 */
struct ShmFileRecord {
  clio::run::UniqueId tag_id_;  /**< CTE tag holding this file's page blobs */
  clio::run::u64 size_;         /**< LOGICAL size (not the tag's byte total) */
  clio::run::u64 ino_;          /**< inode as getattr reports it */
  /**
   * utimens OVERRIDES, not timestamps (0 = unset, defer to the tag).
   *
   * Only the overrides live here because only they are the chimod's to own --
   * the natural atime/mtime/ctime belong to the CTE tag and are already
   * mirrored in ShmTagRecord. Duplicating them here would invent a second,
   * drifting copy of a number this module does not maintain.
   */
  clio::run::u64 ov_atime_ns_;
  clio::run::u64 ov_mtime_ns_;
  clio::run::u64 ov_ctime_ns_;
  clio::run::u32 mode_;   /**< permission bits, or kShmFileNoOverride */
  clio::run::u32 uid_;    /**< owner, or kShmFileNoOverride */
  clio::run::u32 gid_;    /**< group, or kShmFileNoOverride */
  clio::run::u32 flags_;  /**< ShmFileFlags */
  clio::run::u32 reserved_;

  ShmFileRecord()
      : tag_id_(clio::run::UniqueId::GetNull()),
        size_(0),
        ino_(0),
        ov_atime_ns_(0),
        ov_mtime_ns_(0),
        ov_ctime_ns_(0),
        mode_(kShmFileNoOverride),
        uid_(kShmFileNoOverride),
        gid_(kShmFileNoOverride),
        flags_(0),
        reserved_(0) {}

  bool Exists() const { return (flags_ & kShmFileExists) != 0; }
  bool IsDir() const { return (flags_ & kShmFileIsDir) != 0; }

  /** True if a client may resolve this file's pages itself. */
  bool IsFastPathable() const {
    return (flags_ & kShmFileExists) != 0 &&
           (flags_ & (kShmFileIsDir | kShmFilePendingAppend |
                      kShmFileNoFastPath)) == 0 &&
           !tag_id_.IsNull();
  }
};

using ShmFsFileMap =
    ctp::ipc::unordered_map<ShmCacheString, ShmFileRecord, ShmCacheAlloc>;

/**
 * Root of the filesystem chimod's shared-memory cache (issue #817).
 *
 * Same ownership contract as ShmMetadataCacheRoot: written only by the
 * runtime, attached read-mostly by clients, never authoritative, droppable at
 * any moment. Registered in the process-wide MetadataDirectory under the
 * FILESYSTEM pool id, so it coexists with the CTE core pool's cache root
 * rather than competing for one slot.
 */
struct ShmFsCacheRoot {
  static constexpr clio::run::u32 kLayoutVersion = 1;

  clio::run::u32 version_;
  clio::run::u32 ready_;  /**< 0 until fully constructed; clients must check */
  ShmFsFileMap path_to_file_;

  ShmFsCacheRoot() : version_(0), ready_(0) {}
};

/**
 * Runtime-side owner of the filesystem cache.
 *
 * Entirely best-effort, exactly like ShmMetadataCache: every method is a no-op
 * when disabled and every failure is swallowed, because a filesystem operation
 * must never fail on account of a cache mirror.
 */
class ShmFsCache {
 public:
  ShmFsCache() = default;

  /**
   * Construct the cache in the runtime's metadata segment.
   *
   * @param capacity permanent slot count -- the map never rehashes, so a full
   *        table degrades to the RPC path rather than growing.
   * @param owner_pool the filesystem pool, used as the directory key.
   * @return true if the cache is live; false simply means caching is off.
   */
  bool Create(size_t capacity, const clio::run::PoolId &owner_pool) {
#if !CTP_IS_HOST
    (void)capacity;
    (void)owner_pool;
    return false;
#else
    auto *ipc = CLIO_IPC;
    if (ipc == nullptr) {
      return false;
    }
    alloc_ = ipc->GetMetadataAllocator();
    if (alloc_ == nullptr) {
      return false;  // no metadata segment -> caching off, not an error
    }
    try {
      auto fp = alloc_->template Allocate<ShmFsCacheRoot>(sizeof(ShmFsCacheRoot));
      if (fp.IsNull()) {
        alloc_ = nullptr;
        return false;
      }
      root_ = fp.ptr_;
      new (root_) ShmFsCacheRoot();
      new (&root_->path_to_file_) ShmFsFileMap(alloc_, capacity);
      if (!root_->path_to_file_.valid()) {
        root_ = nullptr;
        alloc_ = nullptr;
        return false;
      }
      root_->version_ = ShmFsCacheRoot::kLayoutVersion;
      // ready_ before publication, for the same ordering reason as the core
      // cache: a client that finds the root must find finished maps behind it.
      root_->ready_ = 1;
      if (auto *dir = ipc->GetMetadataDirectory()) {
        dir->RegisterRoot(owner_pool.ToU64(),
                          static_cast<clio::run::u64>(RootOffset()));
      }
      return true;
    } catch (...) {
      root_ = nullptr;
      alloc_ = nullptr;
      return false;
    }
#endif  // CTP_IS_HOST
  }

  bool IsEnabled() const { return root_ != nullptr && alloc_ != nullptr; }

  size_t RootOffset() const {
    if (!IsEnabled()) {
      return 0;
    }
    return static_cast<size_t>(reinterpret_cast<const char *>(root_) -
                               reinterpret_cast<const char *>(alloc_));
  }

  /** Mirror one path's attributes. */
  void PutFile(const std::string &path, const ShmFileRecord &rec) {
    if (!IsEnabled()) {
      return;
    }
    try {
      ShmFsFileMap::BytesProbe p{path.data(), path.size()};
      root_->path_to_file_.InsertOrAssign(
          p, ShmCacheString::HashBytes(path.data(), path.size()), rec);
      // A false return means the table is full: the path is simply not cached
      // and clients keep using RPC for it.
    } catch (...) {
    }
  }

  /**
   * Drop a path.
   *
   * Erase rather than tombstone: a missing record already means "ask the
   * runtime", which is the correct answer for a deleted path, and a tombstone
   * would consume a slot in a table that never grows.
   */
  void ErasePath(const std::string &path) {
    if (!IsEnabled()) {
      return;
    }
    try {
      ShmFsFileMap::BytesProbe p{path.data(), path.size()};
      root_->path_to_file_.Erase(
          p, ShmCacheString::HashBytes(path.data(), path.size()));
    } catch (...) {
    }
  }

  /** Read back a mirrored record (runtime-side helper for read-modify-write
   *  of a single field, e.g. a size bump that must not clobber mode/times). */
  bool TryGetFile(const std::string &path, ShmFileRecord *out) const {
    if (!IsEnabled() || out == nullptr) {
      return false;
    }
    return root_->path_to_file_.TryGetBytes(path.data(), path.size(), out);
  }

  ShmFsCacheRoot *root() { return root_; }

 private:
  ShmCacheAlloc *alloc_ = nullptr;
  ShmFsCacheRoot *root_ = nullptr;
};

}  // namespace clio::cte::filesystem

#endif  // CLIO_CTE_FILESYSTEM_SHM_FS_CACHE_H_
