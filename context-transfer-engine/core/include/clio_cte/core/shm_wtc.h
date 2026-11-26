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
 * The shared-memory write-through cache (core_shm_cache_writethru chimod) segment:
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
 * See context-transfer-engine/core_shm_cache_writethru/DESIGN.md.
 */

#ifndef CLIO_CTE_CORE_SHM_WTC_H_
#define CLIO_CTE_CORE_SHM_WTC_H_

#include <atomic>
#include <chrono>
#include <cstring>
#include <functional>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <clio_ctp/data_structures/ipc/ring_buffer.h>
#include <clio_ctp/memory/backend/posix_shm_mmap.h>
#include <clio_ctp/thread/lock/mutex.h>
#include <clio_ctp/thread/lock/timed_mutex.h>

#include "clio_runtime/clio_runtime.h"
#include "clio_runtime/types.h"

namespace clio::cte::core::wtc {

using u32 = clio::run::u32;
using u64 = clio::run::u64;
using WtcAlloc = CLIO_TASK_ALLOC_T;

/** Pool id of the default-compose shm write-through cache. */
inline const clio::run::PoolId kDefaultPoolId(568, 0);

/** Put flags the segment can carry: per-blob properties only (equals
 *  kCtePutDroppable). Replace/append depend on the chain's state, so a put
 *  with any other flag is refused (kMiss) and takes the task path. */
static constexpr u32 kPerBlobPutFlags = 0x2u;

/** Longest blob name the segment can hold; longer names take the task path. */
static constexpr u32 kMaxBlobName = 200;
/** Independent lock regions of the blob table. */
static constexpr u32 kStripes = 64;
/** Probe budget inside one stripe before an insert gives up. */
static constexpr u32 kMaxProbe = 64;
/** Reader attempts while a writer is mid-copy before falling back to the
 *  task path. */
static constexpr u32 kTornRetries = 256;
/** Most blob-table slots one eviction attempt examines. */
static constexpr u32 kEvictScanMax = 4096;
/** Eviction rounds one buffer allocation makes before the caller writes
 *  around the cache. */
static constexpr u32 kAllocRounds = 8;
/** After kMergeFailsForBackoff merge-and-retry allocations fail in a row,
 *  skip further ones (each sorts every free page under the allocator lock)
 *  for this long. */
static constexpr u64 kCoalesceBackoffNs = 2ull * 1000 * 1000;
static constexpr u32 kMergeFailsForBackoff = 4;
/** Window evictions (each followed by one merge) per buffer allocation. */
static constexpr u32 kMergesPerAlloc = 2;
/** Bytes a window eviction frees (at most 1/8 of the segment, at least two
 *  requests with their page headers): one merge then serves many
 *  allocations of the request's size, since each merge sorts every free
 *  page. */
static constexpr u64 kWindowBytes = 8ull << 20;
/** Default thread slots (journal + dirty queue each) in a segment. */
static constexpr u32 kDefaultThreads = 256;
/** Fewest entries in one thread's dirty queue (segment size permitting). */
static constexpr u32 kMinQueueDepth = 1024;
/** LockSegment: a runtime waiter checks the holder's liveness every this
 *  many sleeps (~10 ms). */
static constexpr u32 kRepairCheckSpins = 200;
/** Queue depth and thread slots a small segment may shrink to. */
static constexpr u32 kFloorQueueDepth = 64;
static constexpr u32 kFloorThreads = 16;
/** Smallest buffer allocated for a blob. */
static constexpr u64 kMinBuffer = 4096;
/** Extent mode: bytes per granule of a leased RAM-bdev extent. */
static constexpr u64 kGranule = 4096;
/** Root::data_mode_: where blob bytes live (0 = metadata only). */
static constexpr u32 kDataBdev = 1;  /**< granules of a leased RAM bdev */
static constexpr u32 kDataSelf = 2;  /**< granules carved from this segment */
/** Heap bytes left to the segment allocator after the granule region. */
static constexpr u64 kSelfSlack = 4ull << 20;
/** Extent mode: runs of granules one entry can hold (no overflow list). */
static constexpr u32 kInlineRuns = 8;

/**
 * Extent mode: a run of consecutive granules, contiguous in the RAM bdev
 * (a run never crosses the boundary between two leased blocks).
 */
struct Run {
  u32 g_;  /**< first granule */
  u32 n_;  /**< granules */
};

/**
 * Extent mode: an entry's buffer as runs of RAM-bdev granules (cap_ = their
 * bytes; buf_off_ unused). One per blob slot, in a side table allocated only
 * in extent mode, so the default layout is unchanged. Entry's stripe lock.
 */
struct RunSet {
  u32 n_;
  u32 pad_;
  Run r_[kInlineRuns];
};

/** Slot states. */
enum SlotState : u32 { kSlotEmpty = 0, kSlotLive = 1, kSlotTomb = 2 };

/** Blob entry flags. */
enum BlobFlags : u32 {
  /** buf_off_ holds a buffer of cap_ bytes. */
  kResident = 1u << 0,
  /** The buffer holds the WHOLE blob (size_ bytes) and may serve reads. */
  kComplete = 1u << 1,
  /** size_ is the chain's size of the blob (metadata only; no bytes). */
  kSizeKnown = 1u << 2,
};

/** Tag entry flags. */
enum TagFlags : u32 {
  /** GetOrCreateTag reported that it created this tag: blobs written under it
   *  through this cache have no older bytes anywhere. */
  kTagCreated = 1u << 0,
  /** name_ holds the tag's name as the chain reports it. */
  kTagNameValid = 1u << 1,
  /** size_/times hold the tag's cluster-wide attributes. */
  kTagAttrValid = 1u << 2,
};

/** Longest tag name cached (longer names are always fetched). */
static constexpr u32 kMaxTagName = 255;

/**
 * Reserved blob name addressing a tag's METADATA in the coherence protocol:
 * (tag id, kTagMetaName) has a home like any blob key, which tracks the
 * nodes caching the tag's metadata and invalidates them when it changes.
 * Starts with a control byte, so no real blob name collides with it.
 */
static constexpr char kTagMetaName[] = "\x01tagmeta";
static constexpr size_t kTagMetaNameLen = sizeof(kTagMetaName) - 1;

/**
 * Method id of the cache module's TagAttrs verb: a core::GetTagSizeTask
 * sent to the local cache container, answered from the segment or fetched
 * (the cluster total) and cached. Must equal shm_wtc::Method::kTagAttrs.
 */
static constexpr u32 kTagAttrsMethod = 103;

/** Result of a client-side operation. */
enum class Status : u32 {
  kOk = 0,
  kMiss = 1,      /**< not servable here; use the task path */
  kNoSpace = 2,   /**< no buffer or slot could be made; wait for the flusher */
  kTooLong = 3,   /**< blob name too long for the segment */
  /** The entry is pinned (a flush in flight or a copy in progress) and the
   *  write needs its buffer to grow or move. Nothing was modified; retry
   *  WITHOUT holding any stripe lock, then fall back to the task path. A
   *  writer never waits for a pin under a stripe lock: a flush pin is
   *  released only after the home acks, and the home's ack waits for an
   *  Invalidate on this node that needs that very stripe lock. */
  kBusy = 4,
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
  std::atomic<u32> pins_;          /**< thread copies out (no free/move) */
  std::atomic<u32> rt_pins_;       /**< runtime pins (flusher reads, held
                                        until a home acks): counted apart so
                                        crash repair can zero pins_ */
  u32 entry_pad_;
  /** Writes retired: their range landed (flusher) or was rolled back (crash
   *  repair). wbegin_ - retired_ is the writes not yet landed ("pending"):
   *  each write raises wbegin_ exactly once, in ONE atomic step, so a write
   *  is pending exactly when its owner got that far. */
  std::atomic<u64> retired_;
  std::atomic<u64> write_gen_;     /**< bumped by every logged write */
  std::atomic<u64> inval_epoch_;   /**< bumped by every invalidation */
  std::atomic<u64> hits_;          /**< frecency: access count */
  std::atomic<u64> last_access_ns_;/**< frecency: last access */
  /** Seqlock over the buffer bytes: writers bump wbegin_ under the stripe
   *  lock before their memcpy and wend_ after it. A reader copies only when
   *  they are equal and re-checks wbegin_ afterwards (torn-read guard). */
  std::atomic<u64> wbegin_;
  std::atomic<u64> wend_;
  /** Bytes [valid_lo_, valid_hi_) of the buffer are the blob's current
   *  bytes even when the copy is not kComplete: written through this cache
   *  since the last invalidation. A read inside it is a hit. Stripe lock. */
  u64 valid_lo_;
  u64 valid_hi_;
  char name_[kMaxBlobName];
};

/** One tag: creation knowledge, sticky flush error and cached metadata. */
struct TagEntry {
  u32 state_;
  u32 flags_;                      /**< TagFlags */
  u32 major_;
  u32 minor_;
  std::atomic<u32> sticky_rc_;     /**< first failed flush of this tag */
  u32 name_len_;                   /**< kTagNameValid */
  u64 name_hash_;                  /**< KeyHash of the name (NameSlot key) */
  u64 size_;                       /**< kTagAttrValid: total bytes */
  u64 ctime_;                      /**< kTagAttrValid: change time (ns) */
  u64 mtime_;                      /**< kTagAttrValid: modify time (ns) */
  u64 atime_;                      /**< kTagAttrValid: access time at fill */
  u64 attr_write_seq_;             /**< local_write_seq_ when filled */
  char name_[kMaxTagName + 1];
};

/** Cluster-wide tag attributes (what GetTagSize returns). */
struct TagAttrs {
  u64 size_ = 0;
  u64 ctime_ = 0;
  u64 mtime_ = 0;
  u64 atime_ = 0;
};

/** Name -> tag id index (GetOrCreateTag hits). */
struct NameSlot {
  u32 state_;
  u32 major_;
  u32 minor_;
  u32 pad_;
  u64 hash_;
};

/** One logged write: slot + exact byte range inside that blob. */
struct DirtyRange {
  u32 slot_;
  u32 gen_;   /**< low 32 bits of the entry's write_gen_ after this write */
  u64 off_;
  u64 size_;
};

/** ThreadSlot::state_. */
enum ThreadSlotState : u32 {
  kThreadFree = 0,      /**< unowned, queue drained */
  kThreadClaiming = 1,  /**< a thread is stamping its identity */
  kThreadOwned = 2,     /**< a live thread writes through it */
  kThreadRetiring = 3,  /**< owner gone: freed once its queue drains */
};

/** ThreadSlot::op_: what the owner is doing to shared state. */
enum JournalOp : u32 {
  kJIdle = 0,     /**< nothing */
  kJStripe = 1,   /**< locking or holding stripe jstripe_ (changing jslot_) */
  kJCopyIn = 2,   /**< lock released; copying a write into jslot_, whose
                       range publishes at tail_ index jtail_ */
  kJCopyOut = 3,  /**< lock released; copying out of jslot_ (pinned) */
  kJTag = 4,      /**< locking or holding the tag lock (read only) */
};

/** ThreadSlot::jkind_: the operation a journal entry belongs to. */
enum JournalKind : u32 {
  kJKPut = 1,   /**< TryPut: changes jslot_ */
  kJKRead = 2,  /**< TryGet / TryGetSize: pins jslot_ at most */
};

/** No slot / stripe recorded in a journal field. */
static constexpr u32 kJNone = 0xFFFFFFFFu;

/**
 * One client or runtime thread's place in the segment: its identity, its
 * journal and its dirty queue.
 *
 * The journal tells the runtime, after the owner died, what the owner was
 * doing (see DESIGN.md, crash recovery). Each field is stored after the
 * state it describes exists, so the journal is never ahead of the truth:
 * jwbegin_ before jslot_, jslot_ before any change to that entry, op_
 * kJCopyIn only once the entry's counters are raised.
 *
 * The queue A single-producer ring: the owner writes a record, then
 * publishes it with ONE store of tail_, so a thread that dies leaves no
 * reserved-but-unfilled record; the runtime drains it (head_).
 */
struct ThreadSlot {
  std::atomic<u32> state_;  /**< ThreadSlotState */
  u32 pid_;                 /**< owner process */
  u64 start_;               /**< owner process start time (pid reuse) */
  u64 tid_;                 /**< owner thread (gettid) */
  u64 ring_off_;            /**< DirtyRange[Root::qdepth_] */
  char pad0_[24];           /**< owner's fields start a new 64 B line
                                 (padding, not alignas: the segment
                                 allocator does not align to 64) */
  std::atomic<u32> op_;     /**< JournalOp */
  std::atomic<u32> jkind_;   /**< JournalKind */
  std::atomic<u32> jstripe_; /**< stripe the op locks or holds (kJNone) */
  std::atomic<u32> jslot_;   /**< entry the op changes or pins (kJNone) */
  std::atomic<u32> jvictim_; /**< entry an eviction is releasing (kJNone) */
  u32 jpad_;
  std::atomic<u64> jwbegin_; /**< jslot_'s wbegin_ before the op */
  std::atomic<u64> jtail_;   /**< tail_ index the op's range publishes at */
  std::atomic<u64> tail_;    /**< ranges published (owner) */
  char pad1_[16];
  std::atomic<u64> head_;   /**< ranges drained (runtime) */
  char pad2_[56];
};

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
  std::atomic<u64> flush_skipped_;  /**< ranges retired without a resend: a
                                         later flush already carried them */
  std::atomic<u64> tag_hits_;       /**< tag lookups served from the segment */
  std::atomic<u64> md_fills_;       /**< metadata fetched and cached */
  std::atomic<u64> miss_absent_;    /**< get miss: no entry for the blob */
  std::atomic<u64> miss_nonres_;    /**< get miss: entry holds no bytes */
  std::atomic<u64> miss_range_;     /**< get miss: bytes not known current */
  std::atomic<u64> miss_busy_;      /**< get miss: writer mid-copy / torn */
  std::atomic<u64> put_full_log_;   /**< put refused: dirty log near full */
  std::atomic<u64> put_full_slot_;  /**< put refused: no blob entry free */
  std::atomic<u64> put_full_buf_;   /**< put refused: no buffer space */
  std::atomic<u64> put_miss_;       /**< put refused: md_only / put flags */
  std::atomic<u64> put_busy_;       /**< put refused: entry pinned, needed to
                                         grow its buffer (kBusy) */
  std::atomic<u64> repairs_;        /**< crash repairs of dead clients */
};

/** Segment root: the first allocation in the segment. */
struct Root {
  static constexpr u64 kMagic = 0x3143544357534C43ull;  // "CLSWCTC1"
  static constexpr u32 kVersion = 13;

  u64 magic_;
  u32 version_;
  std::atomic<u32> ready_;
  u64 format_id_;    /**< unique per Format: a thread's cached slot of an
                          earlier segment mapped at the same address is
                          never taken for one in this segment */
  u32 node_id_;
  u32 slots_per_stripe_;
  u32 max_tags_;
  u32 evict_sample_;
  u64 blobs_off_;
  u64 tags_off_;
  u64 names_off_;    /**< NameSlot[max_tags_ * 2] */
  u64 threads_off_;  /**< ThreadSlot[threads_] */
  u32 threads_;      /**< thread slots */
  u32 qdepth_;       /**< entries in each thread's dirty queue */
  std::atomic<u32> threads_hw_;  /**< slots ever claimed (drain bound) */
  /** Ranges published, counted by each producer AFTER its tail store: a
   *  hint that lets the runtime skip scanning every queue when nothing is
   *  new (it lags by at most the ranges of a producer that died between the
   *  two stores; a forced drain still finds those). */
  char published_pad_[64];  /**< own cache line (no alignas: the segment
                                 allocator does not align to 64) */
  std::atomic<u64> published_;
  char published_pad2_[56];
  u32 threads_pad_;
  u64 owners_off_;   /**< u32 per kMinBuffer granule: slot+1 of the buffer
                          whose data starts there (0 = none) */
  u64 owner_count_;  /**< granules covered by owners_off_ */
  u64 max_fill_;
  u64 half_life_ns_;
  std::atomic<u64> flushed_;       /**< ranges the flusher completed, in the
                                        order it drained them */
  std::atomic<u64> evict_cursor_;
  std::atomic<u64> coalesce_after_ns_;  /**< no merge-and-retry before this */
  std::atomic<u32> merge_fails_;        /**< consecutive failed merges */
  /** Bumped by every tag-metadata drop: a fill fetched before the bump
   *  might carry a name the drop invalidated, so it is not installed. */
  std::atomic<u64> tag_inval_seq_;
  /** Bumped by every write absorbed or forwarded on this node: cached tag
   *  attributes filled before it may miss that write (a coarse, lock-free
   *  read-your-writes guard; remote writes invalidate through the home). */
  std::atomic<u64> local_write_seq_;
  std::atomic<u32> global_rc_;     /**< sticky rc for tags not in TagTable */
  /** Containers (one per node) whose node is down, as a bitmap of container
   *  ids < 64; written by the runtime on every liveness change. A blob whose
   *  home is down is neither cached nor served here: remote writes
   *  invalidate copies THROUGH the home, so while it is down a copy here
   *  could go stale unseen (reads of old bytes during and after a failure). */
  std::atomic<u64> dead_homes_;
  std::atomic<u32> num_homes_;     /**< containers homes are hashed over */
  u32 homes_pad_;
  u32 md_only_;                    /**< metadata only: no blob bytes */
  /** Granule mode (kDataBdev / kDataSelf): blob bytes live in granules of extents
   *  leased from a node-local RAM bdev, not in this segment. */
  u32 data_mode_;
  u32 data_pid_;                   /**< runtime pid owning the RAM bdev */
  u32 data_pool_major_;            /**< the RAM bdev's pool */
  u32 data_pool_minor_;
  u64 granules_;                   /**< granules leased */
  u64 gran_off_off_;               /**< u64[granules_]: bdev offset of each */
  u64 runsets_off_;                /**< RunSet[slots]: each entry's runs */
  u64 gran_owner_off_;             /**< atomic u32[granules_]: slot + 1 */
  std::atomic<u64> evict_gcursor_; /**< next window to evict (granules) */
  u64 free_bits_off_;              /**< u64 words: bit set = granule free */
  u64 brk_bits_off_;               /**< bit set = granule starts a new block */
  u64 data_cursor_;                /**< next-fit start (data_lock_) */
  std::atomic<u64> free_granules_;
  ctp::TimedMutex data_lock_;      /**< guards the bitmaps and cursor */
  Stats stats_;
  ctp::TimedMutex tag_lock_;
  ctp::TimedMutex stripes_[kStripes];
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
  u64 entry_pool_;  ///< PoolId::ToU64 clients bind to (0 = the cache pool)
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
  u32 log_depth_ = 65536;  ///< dirty-queue entries over all thread slots
  u32 threads_ = kDefaultThreads;  ///< thread slots
  u32 evict_sample_ = 16;
  u64 max_fill_ = 16ull << 20;
  u64 half_life_ns_ = 10ull * 1000 * 1000 * 1000;
  bool md_only_ = false;  ///< cache metadata only (see WtcConfig::md_only_)
  /** Extent mode: (bdev offset, bytes) of each extent leased from the RAM
   *  bdev below; empty = bytes live in this segment (buddy allocator). */
  std::vector<std::pair<u64, u64>> extents_;
  u32 data_pid_ = 0;
  u32 data_pool_major_ = 0;
  u32 data_pool_minor_ = 0;
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
  /** True when the segment caches metadata only (no blob bytes). */
  bool MdOnly() const { return root_ != nullptr && root_->md_only_ != 0; }
  /** Root accessor (counters, seqs). */
  Root *GetRoot() const { return root_; }
  // ---- Dirty queues (the runtime drains them) -------------------------------

  /**
   * Move up to `max` published ranges from the thread queues into `out`, in
   * a fair rotation over the slots, and free retiring slots whose queue is
   * empty. Runtime only, one caller at a time.
   * @param out ranges appended here
   * @param max most ranges to take
   * @param force scan every queue even when the published count shows
   *        nothing new (periodic: catches a producer that died between its
   *        publish and its count)
   * @return ranges taken
   */
  size_t Drain(std::vector<DirtyRange> *out, size_t max, bool force = true);
  /** True when no thread queue holds an undrained range (by the published
   *  count; a dead producer's uncounted range is found by a forced Drain). */
  bool QueuesEmpty() const {
    return root_->published_.load(std::memory_order_acquire) <= drained_;
  }
  /**
   * Repair what threads of dead processes left half-done, then retire their
   * thread slots (runtime only; see RepairDead). Their queues still drain;
   * each slot is freed once empty.
   * @return slots retired
   */
  u32 ReapDeadSlots();

  // ---- Crash repair (runtime only; DESIGN.md "Crash recovery") -------------

  /**
   * Repair the segment after client processes died: rebuild the granule
   * bitmap if a dead thread held the data lock, roll back each dead write
   * (retire it, stop serving the entry's cached bytes), drop dead readers'
   * pins, release every lock a dead thread held, and retire the dead
   * threads' slots. Waits for live threads' in-progress copies on the
   * entries involved; never breaks a live holder's lock.
   * @param suspect_pid a lock holder found dead by a waiter (0 = none)
   * @return thread slots retired (0 also when another thread is repairing)
   */
  u32 RepairDead(u32 suspect_pid = 0);

  /**
   * Diagnostics for a quiescent segment: every granule is free exactly when
   * no live entry's runs hold it, owners match, and the free count agrees.
   * @return true when consistent
   */
  bool VerifyGranules() const;

  /**
   * Acquire a segment lock, waiting as long as it takes. Clients never
   * break a lock; a runtime thread that finds the holder dead repairs
   * (RepairDead), which releases it.
   * @param m the lock
   * @return the acquisition id (pass it to Unlock)
   */
  u64 LockSegment(ctp::TimedMutex &m);

  // ---- Data path (client or runtime) ---------------------------------------

  /**
   * Copy a write into the blob's buffer and log its range.
   * @param tag_major,tag_minor the TagId
   * @param name,name_len blob name
   * @param off,size range inside the blob
   * @param src bytes to write
   * @param put_flags put flags (only kCtePutDroppable-style per-blob flags;
   *        a write whose flags differ from the blob's returns kMiss)
   * @param may_block true: a full queue waits for the flusher to drain it.
   *        false (runtime workers, which must never spin: the flusher needs
   *        a worker): return kNoSpace instead when the calling thread's
   *        queue is full.
   * @return kOk, kMiss (use the task path), kTooLong, kNoSpace (nothing
   *         written; caller waits) or kBusy (nothing written: the entry is
   *         pinned and its buffer must grow; caller backs off without
   *         holding any lock and retries, then takes the task path)
   */
  Status TryPut(u32 tag_major, u32 tag_minor, const char *name, size_t name_len,
                u64 off, u64 size, const char *src, u32 put_flags = 0,
                bool may_block = true);

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

  /** True when the flusher has completed at least `target` ranges. */
  bool FlushedThrough(u64 target) const {
    return root_->flushed_.load(std::memory_order_acquire) >= target;
  }
  /**
   * Ranges published so far by every thread. A COUNT: FlushedThrough of it
   * means as many ranges landed, not these ones (queues drain in rotation),
   * which is enough to wait for room. An exact barrier is the runtime's
   * (drain, then wait for the drained count).
   */
  u64 PushedNow() const {
    return root_->published_.load(std::memory_order_acquire);
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

  // ---- Tag metadata ---------------------------------------------------------

  /** Current tag-metadata invalidation sequence (snapshot before a fetch). */
  u64 TagInvalSeq() const {
    return root_->tag_inval_seq_.load(std::memory_order_acquire);
  }
  /**
   * Install a tag's name fetched from the chain, unless a tag-metadata drop
   * happened after `seq` was snapshotted.
   * @return true when installed
   */
  bool FillTagName(u32 major, u32 minor, const char *name, size_t len,
                   u64 seq);
  /** Tag id cached for a name (GetOrCreateTag hit). */
  bool LookupTagByName(const char *name, size_t len, u32 *major, u32 *minor);
  /** Cached name of a tag. */
  bool LookupTagName(u32 major, u32 minor, std::string *name);
  /** Drop a tag's cached metadata (rename, name update, DelTag). */
  void DropTagMeta(u32 major, u32 minor);
  /** Note a local write to any tag (voids cached tag attributes). */
  void NoteLocalWrite() {
    root_->local_write_seq_.fetch_add(1, std::memory_order_acq_rel);
  }
  /** Snapshot of the local write sequence (before fetching attributes). */
  u64 LocalWriteSeq() const {
    return root_->local_write_seq_.load(std::memory_order_acquire);
  }
  /**
   * Install a tag's attributes fetched from the chain, unless a tag drop
   * (`seq`) or a local write (`write_seq`) happened since the snapshots.
   */
  bool FillTagAttrs(u32 major, u32 minor, const TagAttrs &a, u64 seq,
                    u64 write_seq);
  /** Cached attributes of a tag (no local write since they were filled). */
  bool LookupTagAttrs(u32 major, u32 minor, TagAttrs *a);

  /**
   * Record a blob size fetched from the chain (metadata only), unless the
   * entry was written or invalidated after its gens were snapshotted.
   */
  bool FillSize(u32 tag_major, u32 tag_minor, const char *name,
                size_t name_len, u64 size, u64 write_gen, u64 inval_epoch);

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
   * Zero-copy flusher read of a logged range: the address of the CURRENT
   * bytes of [off, off+size) of the blob in `slot`, pinned (the buffer is
   * neither freed nor moved) until UnpinRange. Writes may still change the
   * bytes while pinned; a later flush of the newer range lands after this
   * one. Segment mode only.
   * @param size in/out: clamped to the bytes the buffer holds
   * @param copied_gen out: as ReadRange's
   * @return the address, or nullptr in extent mode or when the slot holds
   *         no live resident entry (then nothing is pinned)
   */
  const char *PinRange(u32 slot, u64 off, u64 *size, u64 *copied_gen);

  /** Release a PinRange pin of `slot`. */
  void UnpinRange(u32 slot);

  /**
   * Flusher read of a logged range: copy the CURRENT bytes of [off, off+size)
   * of the blob in `slot`, and report its key.
   * @param dst destination (size bytes)
   * @param size in/out: clamped to the bytes the buffer holds
   * @param copied_gen out (optional): the entry's write_gen_ if no write was
   *        in progress when the copy started (every write up to it is in
   *        the copy), else 0
   * @return false if the slot no longer holds a live resident entry
   */
  bool ReadRange(u32 slot, u64 off, u64 *size, char *dst, u32 *tag_major,
                 u32 *tag_minor, std::string *name, u32 *put_flags,
                 u64 *copied_gen = nullptr);

  /**
   * Full write generation of a logged range: its 32-bit gen_ widened
   * against the entry's current write_gen_ (valid while the range is
   * pending, which keeps the slot's entry from being replaced).
   */
  u64 RangeGen(const DirtyRange &r) const;
  /**
   * The tag of the blob in `slot` (valid while the slot has pending ranges,
   * which keeps its entry from being replaced).
   * @param slot blob slot
   * @param major OUT tag major
   * @param minor OUT tag minor
   * @return false if the slot holds no live entry
   */
  bool SlotTag(u32 slot, u32 *major, u32 *minor) const;

  /** Flusher completion: `n` ranges of `slot` landed. */
  void CompleteRanges(u32 slot, u32 n);

  /** Ranges of `slot` logged and not yet completed. */
  u32 Pending(u32 slot) const {
    return slot < NumSlots() ? PendingOf(Slot(slot)) : 0;
  }
  /** Writes of an entry not yet landed (see BlobEntry::retired_). */
  static u32 PendingOf(const BlobEntry *e) {
    const u64 done = e->retired_.load(std::memory_order_acquire);
    return static_cast<u32>(e->wbegin_.load(std::memory_order_acquire) - done);
  }
  /** True when any thread or the runtime pins an entry's buffer. */
  static bool Pinned(const BlobEntry *e) {
    return e->pins_.load(std::memory_order_acquire) != 0 ||
           e->rt_pins_.load(std::memory_order_acquire) != 0;
  }

  /** Total slots in the blob table. */
  u32 NumSlots() const { return root_->slots_per_stripe_ * kStripes; }

  /**
   * Whether a blob's home container is marked down (see Root::dead_homes_).
   * @param tag_major tag major
   * @param tag_minor tag minor
   * @param name blob name
   * @param name_len its length
   * @return true when the blob must bypass this cache
   */
  bool HomeDown(u32 tag_major, u32 tag_minor, const char *name,
                size_t name_len) const;

  /**
   * Publish which homes are down (runtime only), then drop every cached
   * copy homed on a node that just went down: no invalidation for it can
   * arrive while its home is gone. Pending bytes still flush.
   * @param num_homes containers homes hash over
   * @param dead_mask bit i set = container i is down
   */
  void SetDeadHomes(u32 num_homes, u64 dead_mask);

  /**
   * Stop serving every cached copy and every cached tag attribute (runtime
   * only): after peers held this node dead, invalidations for it went
   * nowhere and writes went through stand-in homes, so any copy here may be
   * stale. Pending bytes still flush; copies refill (and re-register) on use.
   */
  void InvalidateAll();

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
  /**
   * Allocate a buffer for slot `owner`, evicting clean buffers if needed.
   * @param bytes       buffer size
   * @param held_stripe stripe the caller already holds (no relock)
   * @param owner       slot the buffer is for (recorded in the owner map)
   * @return the buffer's offset, or 0 (the caller writes around the cache)
   */
  u64 AllocBuffer(u64 bytes, u32 held_stripe, u32 owner);
  /** Free a buffer (clears its owner-map granule first). */
  void FreeBuffer(u64 off);
  /**
   * Pick an eviction candidate from a bounded sample of clean entries: the
   * lowest-frecency one whose capacity covers want_bytes, else the
   * lowest-frecency overall. Unlocked hint; EvictSlot re-verifies.
   * @param want_bytes size of the allocation the eviction makes room for
   * @return the slot, or -1 if the sample held nothing clean
   */
  long PickVictim(u64 want_bytes);
  /**
   * Evict one slot's buffer if it is still clean, resident and unpinned.
   * @param held_stripe stripe the caller already holds (no relock)
   * @param slot        the slot
   * @param granule     owner-map granule its buffer must start in, or
   *                    kAnyGranule
   * @return true only if a buffer was actually freed
   */
  bool EvictSlot(u32 held_stripe, u32 slot, u64 granule);
  /**
   * Evict every clean buffer that starts in [start, start + span), shifted
   * back to stay inside the heap: the free pages left behind are
   * contiguous, so one merge yields pages larger than any buffer evicted
   * even when the cache is full of small buffers.
   * @param held_stripe stripe the caller already holds (no relock)
   * @param start       data offset of the window's first buffer
   * @param span        window length in bytes (see WindowSpan)
   * @return true if anything was evicted
   */
  bool EvictWindow(u32 held_stripe, u64 start, u64 span);
  /** Window length a window eviction for a `bytes` allocation uses. */
  u64 WindowSpan(u64 bytes) const;
  /** EvictSlot: accept a buffer starting in any granule. */
  static constexpr u64 kAnyGranule = ~0ull;
  /** Release an entry's buffer (lock held, unpinned, nothing pending). */
  void ReleaseBufferLocked(BlobEntry *e);
  /**
   * Grow (or create) an entry's buffer to `need` bytes (stripe lock held).
   * Never waits for a pin: growing a pinned entry would free or move bytes
   * a flush or copy still reads, so a pinned entry is refused untouched.
   * @param e      the entry (its stripe lock is held)
   * @param need   bytes the buffer must hold
   * @param stripe the stripe whose lock the caller holds
   * @return kOk (buffer holds `need` bytes), kBusy (pinned; nothing
   *         changed) or kNoSpace (no buffer could be made)
   */
  Status EnsureCapacityLocked(BlobEntry *e, u64 need, u32 stripe);
  /** Thread slots. */
  ThreadSlot *Slots() const {
    return reinterpret_cast<ThreadSlot *>(Ptr(root_->threads_off_));
  }
  /** A slot's ring. */
  DirtyRange *Ring(const ThreadSlot *t) const {
    return reinterpret_cast<DirtyRange *>(Ptr(t->ring_off_));
  }
  /**
   * The calling thread's slot, claimed on first use (cached per thread and
   * per segment; re-claimed after a fork).
   * @return the slot, or nullptr when every slot is taken
   */
  ThreadSlot *MySlot();
  /** Claim a free slot for the calling thread; nullptr if none. */
  ThreadSlot *ClaimSlot();
  /**
   * Publish one range in the calling thread's queue.
   * @param t the caller's slot (room already checked)
   */
  void Publish(ThreadSlot *t, const DirtyRange &r);
  /** True when `m` is held by a pid in `dead`; *id = its acquisition. */
  bool HolderDead(const ctp::TimedMutex &m, const std::set<u32> &dead,
                  u64 *id) const;
  /** Pids of dead slot owners (and suspect_pid if dead), and their slots. */
  void CollectDead(u32 suspect_pid, std::set<u32> *dead,
                   std::vector<ThreadSlot *> *dslots);
  /** True when a dead slot was copying into or out of an entry of stripe. */
  bool HasCopyRecord(u32 stripe, const std::vector<ThreadSlot *> &dslots) const;
  /** Rebuild the bitmap and owner map if a dead thread held data_lock_. */
  void RepairDataLock(const std::set<u32> &dead);
  /** Mark an entry's runs used (data_lock_ held), trimming runs past cap_
   *  and dropping a freed slot's runs. @return granules claimed */
  u64 ClaimRunsLocked(BlobEntry *e);
  /**
   * Repair the entries dead threads touched in one stripe.
   * @param stripe the stripe
   * @param stolen acquisition id of the dead holder (0: lock it normally)
   */
  void RepairStripe(u32 stripe, u64 stolen, const std::set<u32> &dead,
                    const std::vector<ThreadSlot *> &dslots);
  /** Wait until no live thread copies into or out of any `touched` entry. */
  void WaitLiveCopies(const std::set<u32> &touched, const std::set<u32> &dead);
  /** Undo one dead thread's write on an entry (stripe lock held). */
  void RollBack(BlobEntry *e, const ThreadSlot &t);
  /** Make an entry's residency agree with its runs (stripe lock held). */
  void SanitizeLocked(BlobEntry *e);
  /** Record this mapping in the process-wide set of mapped segments. */
  void NoteMapped();
  /** Frecency of an entry. */
  double Frecency(const BlobEntry *e, u64 now) const;
  /** Pointer into the segment. */
  char *Ptr(u64 off) const { return base_ + off; }
  /** Find a tag entry (tag lock held). */
  TagEntry *FindTagLocked(u32 major, u32 minor, bool insert);
  /** Name index slot for (hash, name), verified against the tag entry. */
  NameSlot *FindNameLocked(u64 h, const char *name, size_t len, bool insert);
  /** Remove a tag's name from the name index (tag lock held). */
  void UnindexNameLocked(TagEntry *t);
  /**
   * Release an entry's buffer if nothing pins it (stripe lock held). Never
   * waits: a pinned buffer stays allocated, and the caller has already made
   * it unservable (flags and valid range cleared); eviction reclaims it
   * once the pin drops.
   * @param e the entry (its stripe lock is held; no pending ranges)
   * @return true if the buffer was released
   */
  bool ReleaseIfUnpinnedLocked(BlobEntry *e);
  /**
   * Drop a deleted blob's entry (stripe lock held, no pending ranges):
   * release its buffer and tombstone the slot, or, if a reader still pins
   * it, leave it live but empty for eviction. Never waits for a pin.
   * @param e the entry (its stripe lock is held)
   */
  void DropUnpendingLocked(BlobEntry *e);

  // ---- extent mode (blob bytes in leased RAM-bdev granules) ----
  /** True when blob bytes live in leased RAM-bdev extents. */
  bool Extents() const { return root_ != nullptr && root_->data_mode_ != 0; }
  /**
   * Carve the granule tables and index the granules (runtime, Format):
   * leased RAM-bdev extents, or the rest of this segment when none.
   * @param lim limits; extents_ empty selects the segment's own bytes
   * @return true when the granules are ready
   */
  bool FormatGranules(const Limits &lim);
  /** Map the extent tables and the RAM bdev's memory (both sides). */
  bool AttachData();
  /** CLIO_PREFAULT in a client: map every leased extent's pages into this
   *  process now ("0" = all of them, else a size budget), so first touches
   *  inside a measured run do not each take a fault. Contents unchanged. */
  void PrefaultData();
  /** Address of a granule's first byte in the RAM bdev. */
  char *GranPtr(u32 g) const { return data_base_ + gran_off_[g]; }
  /** Grow an entry by `add` granules (data_lock_ held); all or nothing. */
  bool AllocRunsLocked(BlobEntry *e, u64 add);
  /** Grow an entry by `add` granules, evicting victims until it fits. */
  bool AllocRunsEvicting(BlobEntry *e, u64 add, u32 stripe);
  /** First free granule at or after `from` (wrapping), or granules_. */
  u64 NextFreeLocked(u64 from) const;
  /** Evict the clean owners of one contiguous window of `span` granules,
   *  then point the next-fit cursor at it (one victim at a time cannot
   *  fix fragmentation: small victims free scattered granules). */
  void EvictGranWindow(u32 held_stripe, u64 span);
  /** Give an entry's runs back (entry's stripe lock held). */
  void FreeRuns(BlobEntry *e);
  /** Copy bytes into / out of an entry's buffer at a byte offset (the
   *  entry is pinned or its stripe lock held, so its runs cannot change). */
  void CopyIn(BlobEntry *e, u64 off, const char *src, u64 n);
  void CopyOut(const BlobEntry *e, u64 off, char *dst, u64 n) const;
  /**
   * Address of [off, off + n) of an entry's buffer when it lies inside one
   * run (granule mode), so it can be read in place.
   * @return the address, or nullptr if the range spans runs
   */
  const char *RunPtr(const BlobEntry *e, u64 off, u64 n) const;
  /** Zero [from, to) of an entry's buffer. */
  void ZeroRange(BlobEntry *e, u64 from, u64 to);

  ctp::ipc::PosixShmMmap data_backend_;  /**< the RAM bdev's segment */
  bool data_mapped_ = false;
  char *data_base_ = nullptr;   /**< byte 0 of the RAM bdev's device space */
  u64 *gran_off_ = nullptr;
  u64 *free_bits_ = nullptr;
  u64 *brk_bits_ = nullptr;
  RunSet *runsets_ = nullptr;
  std::atomic<u32> *gran_owner_ = nullptr;
  /** An entry's runs (extent mode only). */
  RunSet *RS(const BlobEntry *e) const { return &runsets_[e - blobs_]; }

  ctp::ipc::PosixShmMmap backend_;
  bool owner_ = false;
  bool mapped_ = false;
  WtcAlloc *alloc_ = nullptr;
  char *base_ = nullptr;
  Root *root_ = nullptr;
  BlobEntry *blobs_ = nullptr;
  TagEntry *tags_ = nullptr;
  NameSlot *names_ = nullptr;
  std::atomic<u32> *owners_ = nullptr;
  u64 root_off_ = 0;
  u64 seg_size_ = 0;
  u32 drain_cursor_ = 0;  /**< Drain's rotating start slot (runtime) */
  u64 drained_ = 0;       /**< ranges Drain has taken (runtime) */
  std::mutex repair_mu_;  /**< one RepairDead at a time (runtime) */
};

}  // namespace clio::cte::core::wtc

#include "clio_cte/core/shm_wtc_impl.h"

#endif  // CLIO_CTE_CORE_SHM_WTC_H_
