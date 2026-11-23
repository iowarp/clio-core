/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */
#ifndef CLIO_CTE_FILESYSTEM_SHM_FS_CACHE_H_
#define CLIO_CTE_FILESYSTEM_SHM_FS_CACHE_H_

#include <clio_cte/core/shm_metadata_cache.h>
#include <clio_cte/filesystem/fs_canonical.h>

#include <atomic>
#include <cstddef>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

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
   * The file has appends staged but not yet merged into its tail: neither
   * size nor pages may be trusted for a direct read. Reserved: deferred
   * appends now go through the stream pool, whose writers flush before they
   * read, so the runtime no longer sets it.
   */
  kShmFilePendingAppend = 1u << 2,
  /** Catch-all refusal for states the fast path does not model. */
  kShmFileNoFastPath = 1u << 3,
  /**
   * Directory whose ENTIRE child set is reflected in the mirror: every child
   * mutation since its mkdir published a record (create/rename) or a
   * tombstone (unlink), so a MIRROR MISS under it is authoritative ENOENT —
   * the negative-lookup fast path (a task round trip per miss was the
   * dominant metadata cost of checkout workloads). Set at Mkdir (an empty
   * new dir is trivially complete) and at the root's creation; ops that add
   * names the mirror does not model (symlink, hard-link alias) demote the
   * parent to kShmFileNoFastPath. Implicit dirs (never Mkdir'd) simply have
   * no record, which reads as incomplete.
   */
  kShmDirComplete = 1u << 4,
};

/** Sentinel for "no stored override" in the mode/uid/gid fields. */
static constexpr clio::run::u32 kShmFileNoOverride = 0xFFFFFFFFu;

/**
 * Cached per-path filesystem metadata (issue #817).
 *
 * WHY THIS EXISTS: the filesystem chimod owns a LOGICAL size (FileInfo::size_) that is not the tag's physical size. They
 * diverge after ftruncate-grow, sparse writes and staged appends, and POSIX
 * read semantics -- short read at EOF, zeros inside a hole -- depend on the
 * logical one. A client that clamped against the physical size would return
 * the wrong bytes, so the fs number has to be published in its own right.
 *
 * POD ONLY: the map's reader copies the
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
   * the natural atime/mtime/ctime belong to the CTE tag. Duplicating them
   * here would invent a second, drifting copy of a number this module does
   * not maintain.
   */
  clio::run::u64 ov_atime_ns_;
  clio::run::u64 ov_mtime_ns_;
  clio::run::u64 ov_ctime_ns_;
  clio::run::u32 mode_;   /**< permission bits, or kShmFileNoOverride */
  clio::run::u32 uid_;    /**< owner, or kShmFileNoOverride */
  clio::run::u32 gid_;    /**< group, or kShmFileNoOverride */
  clio::run::u32 flags_;  /**< ShmFileFlags */
  /** ShmFsCacheRoot::nsgen_ at publish time. A non-directory record from an
   *  older generation is treated as ABSENT by readers: a directory rename
   *  moved some subtree, and path-keyed records under it kept stale keys. */
  clio::run::u32 nsgen_;

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
        nsgen_(0) {}

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
 * Pending logical sizes, written by CLIENTS.
 *
 * A page write that extends a file raises the file's entry here (an atomic
 * max under a stripe lock, no task); readers and stat take
 * max(record size, pending size); the filesystem module drains the entries
 * to each inode's home in the background (Runtime::SizeDrain) and removes an
 * entry once the home holds its size. Keyed by the packed tag id.
 */
struct ShmSizeSlot {
  clio::run::u64 key_;    ///< packed tag id; 0 = empty, kTomb = deleted
  clio::run::u64 end_;    ///< highest byte written + 1
  clio::run::u64 wtime_;  ///< wall ns of the last write (the new mtime)
  clio::run::u64 pad_;
};

/** One lock stripe of the size table. */
struct ShmSizeStripe {
  std::atomic<clio::run::u32> lock_;   ///< 0 free, 1 held
  std::atomic<clio::run::u32> dirty_;  ///< entries present (drain hint)
};

class ShmSizeTable {
 public:
  static constexpr clio::run::u32 kStripes = 64;
  static constexpr clio::run::u64 kTomb = ~0ull;

  /** A drained entry: the size the home must reach. */
  struct Entry {
    clio::run::u64 key;
    clio::run::u64 end;
    clio::run::u64 wtime;
  };

  ShmSizeTable() : retire_seq_(0), slots_per_stripe_(0), slots_rel_(0) {
    for (auto &s : stripes_) {
      s.lock_.store(0);
      s.dirty_.store(0);
    }
  }

  /**
   * Bind the slot array (runtime, once, before ready_).
   * @param slots the zeroed slot array, in the same segment as this table
   * @param slots_per_stripe slots in each stripe
   */
  void Init(ShmSizeSlot *slots, clio::run::u32 slots_per_stripe) {
    slots_per_stripe_ = slots_per_stripe;
    slots_rel_ = reinterpret_cast<char *>(slots) -
                 reinterpret_cast<char *>(this);
  }

  /** @return true once Init bound a slot array */
  bool Valid() const { return slots_per_stripe_ != 0; }

  /**
   * Count of retirements so far. A reader that found no pending size must
   * re-read the home's size if this moved during its lookup: the entry may
   * have been retired after it read the (older) home size.
   * @return the current retirement count
   */
  clio::run::u64 RetireSeq() const {
    return retire_seq_.load(std::memory_order_acquire);
  }

  /**
   * Raise a file's pending size.
   * @param key packed tag id (nonzero)
   * @param end highest byte written + 1
   * @param wtime wall ns of the write
   * @return false when the key's stripe is full (caller publishes itself)
   */
  bool Raise(clio::run::u64 key, clio::run::u64 end, clio::run::u64 wtime) {
    if (!Valid() || key == 0 || key == kTomb) return false;
    const clio::run::u32 s = StripeOf(key);
    Lock(s);
    ShmSizeSlot *slot = FindLocked(s, key, /*insert=*/true);
    if (slot != nullptr) {
      if (end > slot->end_) slot->end_ = end;
      if (wtime > slot->wtime_) slot->wtime_ = wtime;
    }
    Unlock(s);
    return slot != nullptr;
  }

  /**
   * @param key packed tag id
   * @return the file's pending size, 0 when none
   */
  clio::run::u64 Get(clio::run::u64 key) {
    if (!Valid() || key == 0 || key == kTomb) return 0;
    const clio::run::u32 s = StripeOf(key);
    if (stripes_[s].dirty_.load(std::memory_order_acquire) == 0) return 0;
    Lock(s);
    ShmSizeSlot *slot = FindLocked(s, key, /*insert=*/false);
    const clio::run::u64 end = slot != nullptr ? slot->end_ : 0;
    Unlock(s);
    return end;
  }

  /**
   * Copy every pending entry out (the drain's view).
   * @param out receives the entries
   */
  void Snapshot(std::vector<Entry> *out) {
    if (!Valid()) return;
    for (clio::run::u32 s = 0; s < kStripes; ++s) {
      if (stripes_[s].dirty_.load(std::memory_order_acquire) == 0) continue;
      Lock(s);
      ShmSizeSlot *base = Slots() + s * slots_per_stripe_;
      for (clio::run::u32 i = 0; i < slots_per_stripe_; ++i) {
        if (base[i].key_ != 0 && base[i].key_ != kTomb) {
          out->push_back(Entry{base[i].key_, base[i].end_, base[i].wtime_});
        }
      }
      Unlock(s);
    }
  }

  /**
   * Remove an entry the home now covers -- only if no write raised it since
   * the drain read it (a raise stays for the next drain).
   * @param key packed tag id
   * @param end the size the home reached
   */
  void Retire(clio::run::u64 key, clio::run::u64 end) {
    if (!Valid()) return;
    const clio::run::u32 s = StripeOf(key);
    Lock(s);
    ShmSizeSlot *slot = FindLocked(s, key, /*insert=*/false);
    if (slot != nullptr && slot->end_ <= end) {
      slot->key_ = kTomb;
      slot->end_ = 0;
      slot->wtime_ = 0;
      stripes_[s].dirty_.fetch_sub(1, std::memory_order_acq_rel);
      retire_seq_.fetch_add(1, std::memory_order_acq_rel);
    }
    Unlock(s);
  }

 private:
  static clio::run::u32 StripeOf(clio::run::u64 key) {
    return static_cast<clio::run::u32>((key * 0x9E3779B97F4A7C15ull) >> 58);
  }

  ShmSizeSlot *Slots() {
    return reinterpret_cast<ShmSizeSlot *>(reinterpret_cast<char *>(this) +
                                           slots_rel_);
  }

  void Lock(clio::run::u32 s) {
    auto &l = stripes_[s].lock_;
    for (;;) {
      clio::run::u32 expect = 0;
      if (l.compare_exchange_weak(expect, 1, std::memory_order_acquire)) {
        return;
      }
      std::this_thread::yield();
    }
  }

  void Unlock(clio::run::u32 s) {
    stripes_[s].lock_.store(0, std::memory_order_release);
  }

  /**
   * Find (or insert) a key inside its stripe; the stripe lock is held.
   * @param s stripe
   * @param key packed tag id
   * @param insert insert when absent
   * @return the slot, or nullptr (absent, or stripe full on insert)
   */
  ShmSizeSlot *FindLocked(clio::run::u32 s, clio::run::u64 key, bool insert) {
    ShmSizeSlot *base = Slots() + s * slots_per_stripe_;
    const clio::run::u32 start = static_cast<clio::run::u32>(
        (key ^ (key >> 29)) % slots_per_stripe_);
    ShmSizeSlot *tomb = nullptr;
    for (clio::run::u32 i = 0; i < slots_per_stripe_; ++i) {
      ShmSizeSlot *slot = base + (start + i) % slots_per_stripe_;
      if (slot->key_ == key) return slot;
      if (slot->key_ == kTomb) {
        if (tomb == nullptr) tomb = slot;
        continue;
      }
      if (slot->key_ == 0) {
        if (!insert) return nullptr;
        if (tomb == nullptr) tomb = slot;
        break;
      }
    }
    if (!insert || tomb == nullptr) return nullptr;
    tomb->key_ = key;
    tomb->end_ = 0;
    tomb->wtime_ = 0;
    stripes_[s].dirty_.fetch_add(1, std::memory_order_acq_rel);
    return tomb;
  }

  ShmSizeStripe stripes_[kStripes];
  std::atomic<clio::run::u64> retire_seq_;  ///< bumped by every Retire
  clio::run::u32 slots_per_stripe_;
  std::ptrdiff_t slots_rel_;  ///< slot array, relative to this table
};

/**
 * Root of the filesystem chimod's shared-memory cache (issue #817).
 *
 * Written only by the runtime, attached read-mostly by clients, never
 * authoritative, droppable at any moment. Registered in the process-wide
 * MetadataDirectory under the FILESYSTEM pool id, so it coexists with other
 * modules' roots rather than competing for one slot.
 */
struct ShmFsCacheRoot {
  static constexpr clio::run::u32 kLayoutVersion = 7;

  clio::run::u32 version_;
  clio::run::u32 ready_;  /**< 0 until fully constructed; clients must check */
  /**
   * Nonzero once ANY PutFile failed to land (table full). From that moment a
   * mirror MISS proves nothing — a real file may simply not be cached — so
   * the COMPLETE-dir authoritative-negative fast path must stand down.
   * Sticky by design: capacity does not come back.
   */
  std::atomic<clio::run::u32> overflow_;
  /** Namespace generation: bumped on every DIRECTORY rename (see
   *  ShmFileRecord::nsgen_). */
  std::atomic<clio::run::u32> nsgen_;
  /** Nonzero on a case-insensitive volume: records are keyed by
   *  FsCanonicalPath, which every reader and writer applies (Key). */
  clio::run::u32 case_insensitive_;
  ShmFsFileMap path_to_file_;
  /** Pending sizes raised by clients' page writes (see ShmSizeTable). */
  ShmSizeTable sizes_;

  ShmFsCacheRoot()
      : version_(0), ready_(0), overflow_(0), nsgen_(0),
        case_insensitive_(0) {}

  /**
   * The key a path's record is stored under.
   * @param path the path as spelled
   * @return its canonical form on this volume
   */
  std::string Key(const std::string &path) const {
    return FsCanonicalPath(path, case_insensitive_ != 0);
  }
};

/**
 * Runtime-side owner of the filesystem cache.
 *
 * Entirely best-effort: every method is a no-op
 * when disabled and every failure is swallowed, because a filesystem operation
 * must never fail on account of a cache mirror.
 */
class ShmFsCache {
 public:
  /** Pending-size slots (32 B each): files written and not yet drained. */
  static constexpr size_t kSizeSlots = 64 * 1024;

  ShmFsCache() = default;

  /**
   * Construct the cache in the runtime's metadata segment.
   *
   * @param capacity permanent slot count -- the map never rehashes, so a full
   *        table degrades to the RPC path rather than growing.
   * @param owner_pool the filesystem pool, used as the directory key.
   * @param case_insensitive the volume compares names case-insensitively
   *        (records are then keyed by canonical path)
   * @return true if the cache is live; false simply means caching is off.
   */
  bool Create(size_t capacity, const clio::run::PoolId &owner_pool,
              bool case_insensitive) {
#if !CTP_IS_HOST
    (void)capacity;
    (void)owner_pool;
    (void)case_insensitive;
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
      root_->case_insensitive_ = case_insensitive ? 1u : 0u;
      new (&root_->path_to_file_) ShmFsFileMap(alloc_, capacity);
      if (!root_->path_to_file_.valid()) {
        root_ = nullptr;
        alloc_ = nullptr;
        return false;
      }
      // The size table: kSizeSlots slots over the stripes. Without it page
      // writes publish sizes themselves (a task), so failing it is not fatal.
      const size_t size_bytes = sizeof(ShmSizeSlot) * kSizeSlots;
      auto sp = alloc_->template Allocate<char>(size_bytes);
      if (!sp.IsNull()) {
        std::memset(sp.ptr_, 0, size_bytes);
        root_->sizes_.Init(reinterpret_cast<ShmSizeSlot *>(sp.ptr_),
                           kSizeSlots / ShmSizeTable::kStripes);
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
      const std::string key = root_->Key(path);
      ShmFsFileMap::BytesProbe p{key.data(), key.size()};
      ShmFileRecord stamped = rec;
      stamped.nsgen_ = root_->nsgen_.load(std::memory_order_relaxed);
      bool ok = root_->path_to_file_.InsertOrAssign(
          p, ShmCacheString::HashBytes(key.data(), key.size()), stamped);
      // A false return means the table is full: the path is simply not
      // cached and clients keep using RPC for it — and authoritative
      // negatives must stand down forever (see overflow_).
      if (!ok) {
        root_->overflow_.store(1, std::memory_order_release);
      }
    } catch (...) {
      if (root_ != nullptr) {
        root_->overflow_.store(1, std::memory_order_release);
      }
    }
  }

  /**
   * Drop a path.
   *
   * Erase rather than tombstone: a missing record already means "ask the
   * runtime", which is the correct answer for a deleted path, and a tombstone
   * would consume a slot in a table that never grows.
   */
  /** Invalidate every non-directory record published before now (directory
   *  rename: subtree records keep stale path keys — see ShmFileRecord). */
  void BumpNsGen() {
    if (IsEnabled()) {
      root_->nsgen_.fetch_add(1, std::memory_order_relaxed);
    }
  }

  void ErasePath(const std::string &path) {
    if (!IsEnabled()) {
      return;
    }
    try {
      const std::string key = root_->Key(path);
      ShmFsFileMap::BytesProbe p{key.data(), key.size()};
      root_->path_to_file_.Erase(
          p, ShmCacheString::HashBytes(key.data(), key.size()));
    } catch (...) {
    }
  }

  /** Read back a mirrored record (runtime-side helper for read-modify-write
   *  of a single field, e.g. a size bump that must not clobber mode/times). */
  bool TryGetFile(const std::string &path, ShmFileRecord *out) const {
    if (!IsEnabled() || out == nullptr) {
      return false;
    }
    const std::string key = root_->Key(path);
    return root_->path_to_file_.TryGetBytes(key.data(), key.size(), out);
  }

  ShmFsCacheRoot *root() { return root_; }

 private:
  ShmCacheAlloc *alloc_ = nullptr;
  ShmFsCacheRoot *root_ = nullptr;
};

}  // namespace clio::cte::filesystem

#endif  // CLIO_CTE_FILESYSTEM_SHM_FS_CACHE_H_
