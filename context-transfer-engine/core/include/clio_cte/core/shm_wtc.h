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

/**
 * @file shm_wtc.h
 * The shared-memory write-through cache (shm_writethru_cache chimod) segment:
 * layout and the operations both sides run on it.
 *
 * One named segment per cache pool per node, formatted as a BuddyAllocator.
 * The chimod creates it; client processes map it and run Put/Get/GetSize
 * directly, with no task. Every write copies into a contiguous per-blob
 * buffer and pushes the exact byte range onto ONE mpsc ring (the dirty log);
 * the chimod's FlushLog task is that ring's only consumer and aggregates the
 * ranges into MultiPutBlob tasks down the chain.
 *
 * Concurrency:
 *  - The blob table is split into kStripes independent open-addressed
 *    regions, each guarded by one ctp::Mutex; a key probes only inside its
 *    stripe, so a stripe lock covers every slot the key can occupy.
 *  - Lock hold times cover a hash probe and buffer bookkeeping, never a
 *    payload memcpy. A memcpy holds a PIN on the entry instead; a pinned
 *    buffer is never freed, moved or evicted.
 *  - Stripe locks are only ever taken blocking one at a time; eviction from
 *    other stripes uses TryLock, so there is no lock-order cycle.
 *
 * Everything in the segment is reached by offset from the allocator's
 * backend data, so the segment may map at different addresses per process.
 * See context-transfer-engine/shm_writethru_cache/DESIGN.md.
 */

#ifndef CLIO_CTE_CORE_SHM_WTC_H_
#define CLIO_CTE_CORE_SHM_WTC_H_

#include <atomic>
#include <chrono>
#include <cstring>
#include <string>
#include <thread>

#include <clio_ctp/data_structures/ipc/ring_buffer.h>
#include <clio_ctp/memory/backend/posix_shm_mmap.h>
#include <clio_ctp/thread/lock/mutex.h>

#include "clio_runtime/clio_runtime.h"
#include "clio_runtime/types.h"

namespace clio::cte::core::wtc {

using u32 = clio::run::u32;
using u64 = clio::run::u64;
using WtcAlloc = CLIO_TASK_ALLOC_T;

/** Pool id of the default-compose shm write-through cache. */
inline const clio::run::PoolId kDefaultPoolId(565, 0);

/** Longest blob name the segment can hold; longer names take the task path. */
static constexpr u32 kMaxBlobName = 200;
/** Independent lock regions of the blob table. */
static constexpr u32 kStripes = 64;
/** Probe budget inside one stripe before an insert gives up. */
static constexpr u32 kMaxProbe = 64;
/** Smallest buffer allocated for a blob. */
static constexpr u64 kMinBuffer = 4096;

/** Slot states. */
enum SlotState : u32 { kSlotEmpty = 0, kSlotLive = 1, kSlotTomb = 2 };

/** Blob entry flags. */
enum BlobFlags : u32 {
  /** buf_off_ holds a buffer of cap_ bytes. */
  kResident = 1u << 0,
  /** The buffer holds the WHOLE blob (size_ bytes) and may serve reads. */
  kComplete = 1u << 1,
};

/** Tag entry flags. */
enum TagFlags : u32 {
  /** GetOrCreateTag reported that it created this tag: blobs written under it
   *  through this cache have no older bytes anywhere. */
  kTagCreated = 1u << 0,
};

/** Result of a client-side operation. */
enum class Status : u32 {
  kOk = 0,
  kMiss = 1,      /**< not servable here; use the task path */
  kNoSpace = 2,   /**< no buffer or slot could be made; wait for the flusher */
  kTooLong = 3,   /**< blob name too long for the segment */
};

/**
 * One cached blob. Lives in the segment; POD apart from atomics, zeroed at
 * format time. Fields without atomics are guarded by the stripe lock.
 */
struct BlobEntry {
  u32 state_;                      /**< SlotState */
  u32 flags_;                      /**< BlobFlags */
  u32 tag_major_;                  /**< TagId */
  u32 tag_minor_;
  u32 name_len_;
  u32 put_flags_;                  /**< put flags every write agreed on */
  u64 hash_;                       /**< full key hash (probe filter) */
  u64 buf_off_;                    /**< buffer offset (kResident) */
  u64 cap_;                        /**< buffer capacity */
  u64 size_;                       /**< logical size known to this cache */
  std::atomic<u32> pins_;          /**< in-flight memcpys (no free/move) */
  std::atomic<u32> pending_;       /**< logged ranges not yet flushed */
  std::atomic<u64> write_gen_;     /**< bumped by every logged write */
  std::atomic<u64> inval_epoch_;   /**< bumped by every invalidation */
  std::atomic<u64> hits_;          /**< frecency: access count */
  std::atomic<u64> last_access_ns_;/**< frecency: last access */
  char name_[kMaxBlobName];
};

/** One tag this cache has seen created. */
struct TagEntry {
  u32 state_;
  u32 flags_;
  u32 major_;
  u32 minor_;
  std::atomic<u32> sticky_rc_;     /**< first failed flush of this tag */
  u32 pad_;
};

/** One logged write: slot + exact byte range inside that blob. */
struct DirtyRange {
  u32 slot_;
  u32 pad_;
  u64 off_;
  u64 size_;
};

using DirtyLog = ctp::ipc::mpsc_ring_buffer<DirtyRange, WtcAlloc>;

/** Counters exported for tests and benchmarks. */
struct Stats {
  std::atomic<u64> puts_;
  std::atomic<u64> put_bytes_;
  std::atomic<u64> get_hits_;
  std::atomic<u64> get_misses_;
  std::atomic<u64> evictions_;
  std::atomic<u64> flush_batches_;
  std::atomic<u64> flush_ranges_;
  std::atomic<u64> flush_bytes_;
  std::atomic<u64> fills_;
  std::atomic<u64> invalidations_;
  std::atomic<u64> waits_;
};

/** Segment root: the first allocation in the segment. */
struct Root {
  static constexpr u64 kMagic = 0x3143544357534C43ull;  // "CLSWCTC1"
  static constexpr u32 kVersion = 1;

  u64 magic_;
  u32 version_;
  std::atomic<u32> ready_;
  u32 node_id_;
  u32 slots_per_stripe_;
  u32 max_tags_;
  u32 evict_sample_;
  u64 blobs_off_;
  u64 tags_off_;
  u64 log_off_;
  u64 max_fill_;
  u64 half_life_ns_;
  std::atomic<u64> pushed_;        /**< ranges pushed onto the log */
  std::atomic<u64> flushed_;       /**< ranges the flusher completed */
  std::atomic<u64> evict_cursor_;
  std::atomic<u32> global_rc_;     /**< sticky rc for tags not in TagTable */
  u32 pad_;
  Stats stats_;
  ctp::Mutex tag_lock_;
  ctp::Mutex stripes_[kStripes];
};

/**
 * Published in the runtime's metadata segment and registered in its
 * MetadataDirectory under the cache pool's id: how a client finds the
 * segment. The segment name is derived, never published.
 */
struct Locator {
  static constexpr u64 kMagic = 0x31434F4C43545743ull;  // "CWTCLOC1"
  u64 magic_;
  std::atomic<u32> ready_;
  u32 runtime_pid_;
  u64 root_off_;
  u64 seg_size_;
};

/** Segment name for a cache pool of a given runtime. */
inline std::string SegmentName(u32 runtime_pid, const clio::run::PoolId &pool) {
  return "clio_wtc_" + std::to_string(runtime_pid) + "_" +
         std::to_string(pool.major_) + "_" + std::to_string(pool.minor_);
}

/** Monotonic nanoseconds. */
inline u64 NowNs() {
  return static_cast<u64>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

/** FNV-1a over the blob key. */
inline u64 KeyHash(u32 major, u32 minor, const char *name, size_t len) {
  u64 h = 1469598103934665603ull;
  auto mix = [&h](const void *p, size_t n) {
    const unsigned char *b = static_cast<const unsigned char *>(p);
    for (size_t i = 0; i < n; ++i) {
      h ^= b[i];
      h *= 1099511628211ull;
    }
  };
  mix(&major, sizeof(major));
  mix(&minor, sizeof(minor));
  mix(name, len);
  return h;
}

/** Limits chosen by the chimod config at format time. */
struct Limits {
  u64 seg_size_ = 256ull << 20;
  u32 max_blobs_ = 65536;
  u32 max_tags_ = 4096;
  u32 log_depth_ = 65536;
  u32 evict_sample_ = 16;
  u64 max_fill_ = 16ull << 20;
  u64 half_life_ns_ = 10ull * 1000 * 1000 * 1000;
};

/**
 * A process's view of one cache segment. Runtime and client both use it; the
 * runtime additionally owns the backend (Format), clients attach (Attach).
 * Not copyable: holds the mapping.
 */
class View {
 public:
  View() = default;
  View(const View &) = delete;
  View &operator=(const View &) = delete;
  ~View() { Detach(); }

  /**
   * Create and format the segment (runtime side).
   * @param name segment name (SegmentName)
   * @param backend_id allocator/backend id stamped into the segment
   * @param lim sizes and policy knobs
   * @param node_id this node's id
   * @return true when the segment is ready for clients
   */
  bool Format(const std::string &name, const ctp::ipc::MemoryBackendId &backend_id,
              const Limits &lim, u32 node_id);

  /**
   * Map an existing segment (client side).
   * @param name segment name
   * @param root_off offset of the Root from the backend data
   * @return true when attached to a ready segment of a known layout
   */
  bool Attach(const std::string &name, u64 root_off);

  /** Unmap; the runtime also unlinks the segment when it owns it. */
  void Detach();

  /** True when attached/formatted and ready. */
  bool Ready() const {
    return root_ != nullptr && root_->ready_.load(std::memory_order_acquire) == 1;
  }

  /** Offset of the root (published in the Locator). */
  u64 RootOff() const { return root_off_; }
  /** Segment size. */
  u64 SegSize() const { return seg_size_; }
  /** Root accessor (counters, seqs). */
  Root *GetRoot() const { return root_; }
  /** The dirty log. */
  DirtyLog *Log() const { return log_; }

  // ---- Data path (client or runtime) ---------------------------------------

  /**
   * Copy a write into the blob's buffer and log its range.
   * @param tag_major,tag_minor the TagId
   * @param name,name_len blob name
   * @param off,size range inside the blob
   * @param src bytes to write
   * @param put_flags put flags (only kCtePutDroppable-style per-blob flags;
   *        a write whose flags differ from the blob's returns kMiss)
   * @return kOk, kMiss (use the task path), kTooLong, or kNoSpace (nothing
   *         written; caller waits)
   */
  Status TryPut(u32 tag_major, u32 tag_minor, const char *name, size_t name_len,
                u64 off, u64 size, const char *src, u32 put_flags = 0);

  /**
   * Serve a read from a complete buffer.
   * @param dst destination
   * @return kOk (dst filled) or kMiss
   */
  Status TryGet(u32 tag_major, u32 tag_minor, const char *name, size_t name_len,
                u64 off, u64 size, char *dst);

  /**
   * Blob size when this cache knows it completely.
   * @param out_size set on kOk
   * @return kOk or kMiss
   */
  Status TryGetSize(u32 tag_major, u32 tag_minor, const char *name,
                    size_t name_len, u64 *out_size);

  /** Pending (unflushed) ranges of one blob; 0 if unknown. */
  u32 BlobPending(u32 tag_major, u32 tag_minor, const char *name,
                  size_t name_len);

  /** True when every range pushed up to `target` has been flushed. */
  bool FlushedThrough(u64 target) const {
    return root_->flushed_.load(std::memory_order_acquire) >= target;
  }
  /** Current pushed count (barrier target). */
  u64 PushedNow() const {
    return root_->pushed_.load(std::memory_order_acquire);
  }

  // ---- Tag table -------------------------------------------------------------

  /** Record that the core created this tag (kTagCreated). */
  void NoteTagCreated(u32 major, u32 minor);
  /** True when this cache saw the tag created. */
  bool TagCreated(u32 major, u32 minor);
  /** Record a failed flush for a tag (first error sticks). */
  void SetTagError(u32 major, u32 minor, u32 rc);
  /** Return and clear a tag's sticky error (0 = none). */
  u32 TakeTagError(u32 major, u32 minor);
  /** Forget a tag (DelTag). */
  void DropTag(u32 major, u32 minor);

  // ---- Runtime-side maintenance ---------------------------------------------

  /**
   * Snapshot an entry's generations before a fill (inserting a metadata-only
   * entry if absent).
   * @return false when the name is too long or the table is full
   */
  bool SnapshotGens(u32 tag_major, u32 tag_minor, const char *name,
                    size_t name_len, u64 *write_gen, u64 *inval_epoch);

  /**
   * Install bytes fetched from the chain as the whole blob, unless a local
   * write or an invalidation raced the fetch.
   * @return true when installed
   */
  bool Fill(u32 tag_major, u32 tag_minor, const char *name, size_t name_len,
            const char *data, u64 size, u64 write_gen, u64 inval_epoch);

  /** Drop a blob's cached bytes and completeness (coherence invalidation). */
  void Invalidate(u32 tag_major, u32 tag_minor, const char *name,
                  size_t name_len);

  /** Remove a blob entirely; caller made sure nothing is pending. */
  void DropBlob(u32 tag_major, u32 tag_minor, const char *name,
                size_t name_len);

  /** Remove every blob of a tag that has nothing pending. */
  void DropTagBlobs(u32 major, u32 minor);

  /**
   * Flusher read of a logged range: copy the CURRENT bytes of [off, off+size)
   * of the blob in `slot`, and report its key.
   * @param dst destination (size bytes)
   * @param size in/out: clamped to the bytes the buffer holds
   * @return false if the slot no longer holds a live resident entry
   */
  bool ReadRange(u32 slot, u64 off, u64 *size, char *dst, u32 *tag_major,
                 u32 *tag_minor, std::string *name, u32 *put_flags);

  /** Flusher completion: `n` ranges of `slot` landed. */
  void CompleteRanges(u32 slot, u32 n);

  /** Total slots in the blob table. */
  u32 NumSlots() const { return root_->slots_per_stripe_ * kStripes; }

 private:
  /** Entry pointer by global slot index. */
  BlobEntry *Slot(u32 slot) const { return blobs_ + slot; }
  /** Stripe of a key hash. */
  static u32 StripeOf(u64 h) { return static_cast<u32>(h % kStripes); }
  /** Find a live key inside its stripe (stripe lock held). -1 if absent. */
  long FindLocked(u64 h, u32 major, u32 minor, const char *name, size_t len);
  /** Find or insert (stripe lock held). -1 when the stripe is full. */
  long FindOrInsertLocked(u64 h, u32 major, u32 minor, const char *name,
                          size_t len, bool *inserted);
  /** Reclaim a metadata-only entry in a full stripe (stripe lock held). */
  bool ReclaimSlotLocked(u32 stripe);
  /** Allocate a buffer, evicting clean buffers if needed. 0 on failure. */
  u64 AllocBuffer(u64 bytes, u32 held_stripe);
  /** Free a buffer. */
  void FreeBuffer(u64 off);
  /** Evict the lowest-frecency clean buffer from a sample. */
  bool EvictOne(u32 held_stripe);
  /** Release an entry's buffer (lock held, unpinned, nothing pending). */
  void ReleaseBufferLocked(BlobEntry *e);
  /** Grow (or create) an entry's buffer to `need` bytes (lock held). */
  bool EnsureCapacityLocked(BlobEntry *e, u64 need, u32 stripe);
  /** Frecency of an entry. */
  double Frecency(const BlobEntry *e, u64 now) const;
  /** Pointer into the segment. */
  char *Ptr(u64 off) const { return base_ + off; }
  /** Find a tag entry (tag lock held). */
  TagEntry *FindTagLocked(u32 major, u32 minor, bool insert);
  /** Wait until an entry has no pins (lock held; unpins need no lock). */
  static void WaitUnpinned(BlobEntry *e);

  ctp::ipc::PosixShmMmap backend_;
  bool owner_ = false;
  bool mapped_ = false;
  WtcAlloc *alloc_ = nullptr;
  char *base_ = nullptr;
  Root *root_ = nullptr;
  BlobEntry *blobs_ = nullptr;
  TagEntry *tags_ = nullptr;
  DirtyLog *log_ = nullptr;
  u64 root_off_ = 0;
  u64 seg_size_ = 0;
};

}  // namespace clio::cte::core::wtc

#include "clio_cte/core/shm_wtc_impl.h"

#endif  // CLIO_CTE_CORE_SHM_WTC_H_
