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

#ifndef SAFE_BDEV_RUNTIME_H_
#define SAFE_BDEV_RUNTIME_H_

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/comutex.h>
#include <clio_runtime/bdev/bdev_client.h>
#include <clio_runtime/bdev/transports/block_allocator.h>  // bdev::Block, LiveBlock
#include <clio_runtime/bdev/bdev_alloc_log.h>  // bdev::AllocatorLog (reused WAL)

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>  // std::FILE
#include <cstdlib>  // std::getenv
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <string>
#include <thread>
#include <vector>

#include <clio_ctp/io/io_error.h>  // ctp::IoError, ctp::IsFatalDevice

#include "ec/ec_array.h"  // ec::EcRole, ec::EcState, ec::ReedSolomon
#include "safe_bdev_client.h"
#include "safe_bdev_journal.h"  // StripeJournal (#1137)
#include "safe_bdev_superblock.h"  // MemberSuperblock, kMemberSuperblockMagic
#include "safe_bdev_tasks.h"

/**
 * Runtime container for safe_bdev ChiMod (DYNAMIC VIEW-GROUP model).
 *
 * safe_bdev presents the bdev task interface (AllocateBlocks/FreeBlocks/Write/
 * Read/GetStats) over a growable array of member bdevs, split into two logical
 * groups on a dedicated-parity Reed-Solomon model:
 *
 *   - data_members_   are DATA members (the VIEW group). Blocks are allocated
 *                     PER-CHUNK ROUND-ROBIN over the non-full members, each
 *                     filling to its OWN capacity -> total data capacity =
 *                     Sum(per-drive capacity), no stranding. GROWS via
 *                     AddBdev(as_data).
 *   - parity_members_ are the m dedicated PARITY members (appended via
 *                     AddBdev(as_parity)).
 *
 * ADDRESSING (banding, no map). Each data member owns fixed-size kChunkLen
 * SLOTS 0,1,2,...; a chunk's logical byte offset ENCODES its physical location:
 *   chunk_index = off / kChunkLen
 *   member  d   = chunk_index / kSlotsPerMember
 *   slot    s   = chunk_index % kSlotsPerMember
 *   within      = off % kChunkLen
 * The chunk's physical byte offset on member d is kSuperblockSize + s*kChunkLen.
 * Round-robin picks d for each new chunk; the member's own slot allocator picks
 * s; the returned offset bands (d,s). Decode is pure div/mod -- nothing extra is
 * persisted for addressing (only each member's slot allocator, via the WAL).
 *
 * STRIPES & PARITY (offset-aligned, fixed width -- #1126). A STRIPE s is the
 * chunk at physical slot s on EVERY data member, allocated or not; parity for
 * stripe s lives at the SAME slot s on each parity member:
 *   parity_j[s] = sum_d c(j, d) * data_d[s]      (GF(2^8), Cauchy RS)
 * over all N data members, where c(j, d) does not depend on N (Cauchy rows
 * x_j = 255 - j). An unallocated or freed chunk is a column holding whatever
 * bytes it holds (zeros if never written), so:
 *  - every write updates parity from the change alone (old ^ new, over the
 *    written range) -- fresh chunks included; frees never touch parity;
 *  - a down member's chunk decodes from any N of the N + m columns;
 *  - adding a data member is a new (zero) column: existing parity stays valid.
 * Parity is written before a write is acked (synchronous, #1121), under a
 * per-stripe lock, with a durable stripe intent logged first so a crash
 * between data and parity is re-encoded at restart. Requires
 * N + max_failures <= 256.
 */

namespace clio::run::safe_bdev {

/**
 * Runtime container for safe_bdev operations.
 */
class Runtime : public clio::run::Container {
 public:
  // Required typedef for CLIO_TASK_CC macro and autogen dispatcher.
  using CreateParams = clio::run::safe_bdev::CreateParams;

  Runtime()
      : max_failures_(1),
        parity_level_(0),
        rr_cursor_(0),
        reattached_members_(0) {}
  ~Runtime() override { StopIntentSync(); }

  /** EC chunk / slot length in bytes -- the fixed unit a member bdev is split
   *  into for erasure coding (the "1 MB" of the design; a configurable knob,
   *  kept at 64 KiB here so small test members still hold many slots). */
  static constexpr clio::run::u64 kChunkLen = 65536;

  /**
   * Reserved superblock area at the front of every member bdev (absolute
   * offset 0). The member's usable region begins at offset kSuperblockSize; a
   * member's chunk for SLOT s is at absolute offset kSuperblockSize +
   * s*kChunkLen.
   */
  static constexpr clio::run::u64 kSuperblockSize = 65536;

  /**
   * Banding stride: the number of kChunkLen slots each data member owns in the
   * logical address space. A chunk on member d at slot s bands to logical offset
   * (d*kSlotsPerMember + s)*kChunkLen, so off/kChunkLen decodes to (d,s) by
   * div/mod. Sized so a member can hold a very large capacity while the top
   * member's offset stays within u64 for any realistic member count
   * (2^32 slots * 64 KiB = 256 TiB per member).
   */
  static constexpr clio::run::u64 kSlotsPerMember = (1ull << 32);

  /** Background parity-builder poll period (microseconds): 50 ms. */
  static constexpr double kBuildParityPeriodUs = 50000.0;
  // Re-check period while waiting for a stripe another task holds (#1121).
  static constexpr double kStripeLockPollUs = 10.0;

  /** Allocator-WAL flush/compact poll period (microseconds): 50 ms. */
  static constexpr double kFlushAllocLogPeriodUs = 50000.0;

  /** Compaction policy: rewrite the WAL once on-disk records exceed
   *  max(kMinCompactRecords, live * kCompactGrowthFactor). Mirrors bdev. */
  static constexpr clio::run::u64 kMinCompactRecords = 1024;
  static constexpr clio::run::u64 kCompactGrowthFactor = 4;

  /**
   * Get live task statistics for this task instance.
   */
  clio::run::TaskStat GetTaskStats(const clio::run::Task *task) const override;

  //==========================================================================
  // Method handlers
  //==========================================================================

  /** Create the container (Method::kCreate). */
  clio::run::TaskResume Create(clio::run::shared_ptr<CreateTask> &task);

  /** Allocate multiple blocks (Method::kAllocateBlocks). */
  clio::run::TaskResume AllocateBlocks(clio::run::shared_ptr<AllocateBlocksTask> &task);

  /** Free data blocks (Method::kFreeBlocks). */
  clio::run::TaskResume FreeBlocks(clio::run::shared_ptr<FreeBlocksTask> &task);

  /** Write data (Method::kWrite). */
  clio::run::TaskResume Write(clio::run::shared_ptr<WriteTask> &task);

  /** Read data (Method::kRead). */
  clio::run::TaskResume Read(clio::run::shared_ptr<ReadTask> &task);

  /** Get performance statistics (Method::kGetStats). */
  clio::run::TaskResume GetStats(clio::run::shared_ptr<GetStatsTask> &task);

  /** Add a member bdev (Method::kAddBdev). */
  clio::run::TaskResume AddBdev(clio::run::shared_ptr<AddBdevTask> &task);

  /** Remove a member bdev (Method::kRemoveBdev). */
  clio::run::TaskResume RemoveBdev(clio::run::shared_ptr<RemoveBdevTask> &task);

  /** Recover a failed member bdev (Method::kRecoverBdev). */
  clio::run::TaskResume RecoverBdev(clio::run::shared_ptr<RecoverBdevTask> &task);

  /** Build/raise parity for dirty slots (Method::kBuildParity). */
  clio::run::TaskResume BuildParity(clio::run::shared_ptr<BuildParityTask> &task);

  /** Flush/compact the persistent allocator log (Method::kFlushAllocLog). */
  clio::run::TaskResume FlushAllocLog(clio::run::shared_ptr<FlushAllocLogTask> &task);

  /**
   * Make everything written so far durable AND parity-protected
   * (Method::kSync; bdev's SyncTask layout, so the CTE's plain bdev client
   * can fsync an array). Encodes every dirty stripe, syncs every active
   * member, then fsyncs the allocation/intent log.
   * @param task rc 0 on success; 1 if a member sync failed or a dirty stripe
   *        whose members are all active could not be encoded
   */
  clio::run::TaskResume Sync(clio::run::shared_ptr<SyncTask> &task);

  /** Monitor container state (Method::kMonitor). */
  clio::run::TaskResume Monitor(clio::run::shared_ptr<MonitorTask> &task);

  /** Destroy the container (Method::kDestroy). */
  clio::run::TaskResume Destroy(clio::run::shared_ptr<DestroyTask> &task);

  //==========================================================================
  // Required virtual methods from clio::run::Container
  //==========================================================================

  void Init(const clio::run::PoolId &pool_id, const std::string &pool_name,
            clio::run::u32 container_id = 0) override;

  clio::run::TaskResume Run(clio::run::u32 method,
                      clio::run::shared_ptr<clio::run::Task> task_ptr) override;

  clio::run::u64 GetWorkRemaining() const override;

  /**
   * Register the safe-bdev dashboard page's endpoints (issue #990):
   * the pool index, the add/remove member actions, and the create form. The
   * page itself ships in this module's viz/ directory. Defined in
   * safe_bdev_viz.cc.
   */
  void RegisterViz(clio::run::viz::VizServer &viz,
                   const std::string &mod_name) override;

  void SaveTask(clio::run::u32 method, clio::run::SaveTaskArchive &archive,
                clio::run::shared_ptr<clio::run::Task> &task_ptr) override;

  void LoadTask(clio::run::u32 method, clio::run::LoadTaskArchive &archive,
                clio::run::shared_ptr<clio::run::Task> &task_ptr) override;

  clio::run::shared_ptr<clio::run::Task> AllocLoadTask(
      clio::run::u32 method, clio::run::LoadTaskArchive &archive) override;

  void LocalLoadTask(clio::run::u32 method, clio::run::DefaultLoadArchive &archive,
                     clio::run::shared_ptr<clio::run::Task> &task_ptr) override;

  clio::run::shared_ptr<clio::run::Task> LocalAllocLoadTask(
      clio::run::u32 method, clio::run::DefaultLoadArchive &archive) override;

  void LocalSaveTask(clio::run::u32 method, clio::run::DefaultSaveArchive &archive,
                     clio::run::shared_ptr<clio::run::Task> &task_ptr) override;

  clio::run::shared_ptr<clio::run::Task> NewCopyTask(
      clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> &orig_task_ptr,
      bool deep) override;

  clio::run::shared_ptr<clio::run::Task> NewTask(clio::run::u32 method) override;

  void AggregateOut(clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> &orig_task,
                 const clio::run::shared_ptr<clio::run::Task> &replica_task) override;

 private:
  // Per-member runtime bookkeeping. role_ (DATA vs PARITY) and index_ are FIXED
  // for the member's lifetime -- no rotation. DATA members live in
  // data_members_ (index_ == data column d == position in the vector); PARITY
  // members live in parity_members_ (index_ == parity row j == position). A
  // member's chunk for SLOT s lives at absolute offset kSuperblockSize +
  // s*kChunkLen.
  /**
   * A member's ec::EcState, readable and writable from concurrent task
   * fibers. A data-plane task that sees a member's I/O fail faults it
   * (kActive -> kFaulty) while other workers' tasks are reading the state,
   * so the field is atomic; the copy operations let MemberSlot stay a value
   * type in the member vectors.
   */
  class AtomicEcState {
   public:
    AtomicEcState() : v_(ec::EcState::kActive) {}
    AtomicEcState(ec::EcState s) : v_(s) {}  // NOLINT: implicit by design
    AtomicEcState(const AtomicEcState &o) : v_(o.Load()) {}
    AtomicEcState &operator=(const AtomicEcState &o) {
      v_.store(o.Load(), std::memory_order_release);
      return *this;
    }
    AtomicEcState &operator=(ec::EcState s) {
      v_.store(s, std::memory_order_release);
      return *this;
    }
    operator ec::EcState() const { return Load(); }  // NOLINT
    /** @return the current state. */
    ec::EcState Load() const { return v_.load(std::memory_order_acquire); }
    /**
     * Atomically move from `from` to `to`.
     * @param from the state the caller expects
     * @param to the new state
     * @return true if this call made the transition
     */
    bool Transition(ec::EcState from, ec::EcState to) {
      return v_.compare_exchange_strong(from, to, std::memory_order_acq_rel);
    }

   private:
    std::atomic<ec::EcState> v_;
  };

  struct MemberSlot {
    clio::run::PoolId pool_id_;
    std::string pool_name_;
    clio::run::u32 node_id_ = 0;
    ec::EcRole role_ = ec::EcRole::kData;
    AtomicEcState state_ = ec::EcState::kActive;
    int index_ = -1;  // data column d (DATA) or parity row j (PARITY)
    // True while this member's chunks are being rebuilt onto pool_id_ (a
    // recovery target seated by RecoverBdev). Persisted in the member manifest
    // so an interrupted recovery is RESUMED on restart: the member stays
    // non-active (excluded from reads -> degraded path) until the rebuild
    // completes. RebuildMember is idempotent, so re-running after a crash is
    // safe.
    bool recovering_ = false;
    /** PARITY members: chunk slots the device holds (a data member's
     *  slot s needs parity slot s on every parity member). */
    clio::run::u64 cap_slots_ = ~0ULL;
  };

  // Per-DATA-member slot allocator (parallel to data_members_, indexed by data
  // column d). Each data member fills its OWN kChunkLen slots independently:
  // high_water_ is the next never-used slot; free_ holds freed slots for reuse;
  // live_ is the currently-live slot set (== this member's contribution to the
  // stripes it participates in). Reconstructable purely from the live set on
  // restart (WAL replay): high_water = max(live)+1, free = [0,high_water)\live.
  struct MemberAlloc {
    clio::run::u64 high_water_ = 0;      // next never-used slot
    clio::run::u64 cap_slots_ = 0;       // usable kChunkLen slots on this member
    std::vector<clio::run::u64> free_;   // freed slots, reusable (LIFO)
    std::set<clio::run::u64> live_;      // currently-live slots
    /** Live slots whose block is shorter than kChunkLen: slot -> bytes in
     *  use. The rest of such a chunk is never addressed, so a degraded
     *  write retry need not preserve it (see RetryStripeDegraded). */
    std::unordered_map<clio::run::u64, clio::run::u64> part_len_;

    /** @return bytes of slot `s` that hold addressable data. */
    clio::run::u64 LiveLen(clio::run::u64 s) const {
      auto it = part_len_.find(s);
      return it == part_len_.end() ? kChunkLen : it->second;
    }

    // Slots not yet handed out (bump headroom + reusable frees).
    clio::run::u64 RemainingSlots() const {
      const clio::run::u64 bump_left =
          (cap_slots_ > high_water_) ? (cap_slots_ - high_water_) : 0;
      return bump_left + free_.size();
    }
    bool Full() const { return RemainingSlots() == 0; }
    // Take the next slot (reuse a freed one first, else bump). Caller must have
    // checked !Full(). Records it live.
    clio::run::u64 Take() {
      clio::run::u64 s;
      if (!free_.empty()) {
        s = free_.back();
        free_.pop_back();
      } else {
        s = high_water_++;
      }
      live_.insert(s);
      return s;
    }
    // Release a live slot back to the free list.
    void Release(clio::run::u64 s) {
      if (live_.erase(s) != 0) {
        free_.push_back(s);
      }
      part_len_.erase(s);
    }
  };

  // Client for making calls back to this ChiMod.
  Client client_;

  // EC / membership state. DATA and PARITY members are kept in SEPARATE vectors
  // so a data drive can be appended (growing data_members_) without disturbing
  // parity column indices. Each member vector is index-aligned with its client
  // vector; data_alloc_ is index-aligned with data_members_.
  std::vector<MemberSlot> data_members_;    // index_ == data column d
  std::vector<MemberSlot> parity_members_;  // index_ == parity row j
  std::vector<clio::run::bdev::Client> data_clients_;
  std::vector<clio::run::bdev::Client> parity_clients_;
  std::vector<MemberAlloc> data_alloc_;     // per-data-member slot allocator
  clio::run::u32 rr_cursor_;                // round-robin data-member cursor

  // Reed-Solomon codec cache, keyed by stripe width k_s (data-shard count). Each
  // codec is RS(k_s, max_failures_); a stripe of width k_s uses parity shards
  // 0..parity_level_-1 of it. Few distinct widths, so this stays tiny.
  std::map<int, std::unique_ptr<ec::ReedSolomon>> rs_cache_;

  // Persistent allocator-state log (WAL). REUSED from the bdev module. All slot
  // allocations are logged under a SINGLE group id (kAllocGroup) as BANDED
  // logical offsets; on recovery, live(kAllocGroup) yields every live banded
  // offset, each decoded back to (member d, slot s) to rebuild data_alloc_.
  // Empty path => disabled (every API is a no-op).
  static constexpr clio::run::u32 kAllocGroup = 0;
  clio::run::bdev::AllocatorLog alloc_log_;

  //==========================================================================
  // Crash Consistency Intent Logging (issue #1121). Intent records track
  // when a slot's data write begins (dirty) and when parity write completes
  // (clean). On restart, slots marked dirty are rebuilt before degraded reads.
  //==========================================================================

  // Stripe intent log (#1121): stripes whose parity may be stale are logged
  // as LIVE entries of this alloc-log group (LogAlloc = "dirty", LogFree =
  // "clean", offset = slot). A write logs its stripes dirty and fsyncs the
  // log BEFORE any member write, and logs them clean once their parity is
  // written. After a crash, the group's live entries are exactly the stripes
  // whose data may have landed without parity; Create rebuilds them.
  static constexpr clio::run::u32 kIntentGroup = 1;
  // Array high water (#1126): one group-open record, rewritten whenever an
  // allocation first reaches a new slot (first_row = slots ever used). The
  // live set alone forgets freed slots above the highest live one, whose
  // bytes still count in their stripes' parity.
  static constexpr clio::run::u32 kHighWaterGroup = 2;
  // Parity format (#1126): k = kParityFormat once the array's parity is
  // fixed-width. An array without it re-encodes every slot at Create.
  static constexpr clio::run::u32 kFormatGroup = 3;
  static constexpr clio::run::u32 kParityFormat = 2;
  // Member faults (#1145): one live record per DATA member that is down,
  // key = data column, size = the newest intent key when it went down.
  // A write whose intent is newer saw the member down, so it journals
  // before landing anything; at restart such an intent with no journal
  // record of its own never landed, and its stripe's parity is intact.
  static constexpr clio::run::u32 kFaultGroup = 4;
  /** Intent key at each down data member's fault (guarded by intent_mu_). */
  std::map<clio::run::u32, clio::run::u64> fault_key_;
  /** Record that data member `d` went down (no-op if already recorded). */
  void LogMemberFault(size_t d);
  /** Drop data member `d`'s fault record (it is active again). */
  void LogMemberRecovered(size_t d);
  /** Whether live intent `key` with no journal record of its own is a
   *  write that never landed: every down data member went down before it
   *  (so it would have journaled first). Needs intent_mu_ NOT held. */
  bool IntentNeverLanded(clio::run::u64 key);
  /** Slots ever allocated on any member (guarded by alloc_mu_). */
  clio::run::u64 array_high_water_ = 0;

  /** One logged intent: its unique log key and the stripe it names. */
  struct IntentKey {
    clio::run::u64 key;
    clio::run::u64 slot;
  };

  /**
   * Log a write's stripes dirty (before their data is written) and hand the
   * records to the kernel; await AwaitIntentDurable(returned seq) before any
   * member write.
   * @param slots the stripes
   * @param keys receives one key per stripe, for LogCleanIntents
   * @return the intent's sequence number (0 when nothing was logged)
   */
  clio::run::u64 LogDirtyIntents(const std::set<clio::run::u64> &slots,
                                 std::vector<IntentKey> &keys);

  /**
   * Settle a write's intents for the stripes whose parity it brought current.
   * @param keys the write's intents (LogDirtyIntents)
   * @param clean stripes now encoded
   */
  void LogCleanIntents(const std::vector<IntentKey> &keys,
                       const std::set<clio::run::u64> &clean);


  /** @return the newest intent key; stale intents up to it are covered by
   *  an encode that starts now (pass it to LogStripesEncoded). */
  clio::run::u64 IntentWatermark();

  /**
   * Settle the stale intents of `slots` that an encode covered.
   * @param slots stripes just encoded under their lock
   * @param watermark IntentWatermark() taken once the stripe was held
   */
  void LogStripesEncoded(const std::set<clio::run::u64> &slots,
                         clio::run::u64 watermark);

  // Stale intents per stripe (frees, failed writes), settled by a later
  // encode; in-flight writes' intents are not here. Guarded by intent_mu_.
  std::map<clio::run::u64, std::set<clio::run::u64>> intent_stale_;

  /**
   * Wait until intent `seq` is fsynced, sharing one fsync among concurrent
   * writers (group commit).
   * @param seq a LogStripeIntent sequence number (0 = nothing to wait for)
   */
  clio::run::TaskResume AwaitIntentDurable(clio::run::u64 seq);

  /** Start the intent-log sync thread (alloc log enabled). */
  void StartIntentSync();
  /** Stop and join it, after a final sync of what was appended. */
  void StopIntentSync();
  /** Sync thread body: fsync the log whenever intents await durability. */
  void IntentSyncMain();

  // Group commit state for the intent log (#1121).
  std::mutex intent_mu_;                         // orders appends with seq
  clio::run::u64 intent_key_gen_ = 0;            // unique intent keys
  std::atomic<clio::run::u64> intent_seq_{0};      // last appended
  std::atomic<clio::run::u64> intent_durable_{0};  // last fsynced
  std::thread intent_thread_;
  std::mutex intent_cv_mu_;
  std::condition_variable intent_cv_;
  std::atomic<bool> intent_stop_{false};
  /** CreateParams::intent_sync_: fsync intents before data (power loss). */
  bool intent_sync_ = true;
  /** Down columns' chunks saved before degraded writes (#1137); lives next
   *  to the alloc log (`<alloc_log>.journal`), disabled without one. */
  StripeJournal journal_;
  // Re-check period while the sync thread's fsync covers our intent.
  static constexpr double kIntentSyncPollUs = 20.0;

  /**
   * Restart: mark every stripe the intent log still holds dirty, so its
   * parity is rebuilt before it can be trusted for reconstruction.
   * @return the number of stripes recovered as dirty
   */
  size_t ReplayStripeIntents();

  // Durable member manifest path (== alloc_log_path + ".members"; empty when
  // the alloc log is disabled). Records the CURRENT full membership (data +
  // parity, including runtime AddBdev drives and recovery replacements) plus
  // each member's state and recovery flag, so Create() restores the real array
  // on restart and resumes any interrupted recovery. It is a small WRITE-AHEAD
  // LOG (mirrors the bdev AllocatorLog): each membership change APPENDS a
  // snapshot of the current members (fopen("ab")+fwrite, no rename), and a
  // periodic pass COMPACTS to one record per current member (temp file +
  // std::filesystem::rename, atomic replace on every platform). Replay keeps the
  // last record per (role,index) slot. member_log_records_ is the on-disk record
  // count that drives compaction; member_log_mu_ serialises the file I/O.
  std::string members_manifest_path_;
  mutable std::mutex member_log_mu_;
  mutable clio::run::u64 member_log_records_ = 0;
  clio::run::u32 max_failures_;           // Fault-tolerance target (M == m_max)
  /** Most data (and, separately, parity) members an array can hold. The
   *  member vectors reserve this at Create, so AddBdev's append never
   *  reallocates them under a concurrent data-plane task holding a
   *  reference or index into them (the membership paths are not locked
   *  against I/O). */
  static constexpr size_t kMaxMembers = 256;
  /** Return code of an operation sent to a passive (non-home) container of
   *  a distributed array. */
  static constexpr clio::run::u32 kNotHomeRc = 40;
  bool distributed_ = false;  // CreateParams::distributed_
  bool is_home_ = true;       // distributed: this node runs the array
  clio::run::u32 parity_level_;           // Parity members added so far (m)
  clio::run::u32 reattached_members_;     // Members recognized as ours at Create

  //==========================================================================
  // Recovery observability. RecoverBdev rebuilds a failed member's chunk for
  // EVERY slot it participates in; each such slot is one "recovery operation".
  // These atomics let Monitor() -- and the dashboard safe-bdev
  // dashboard -- report live rebuild progress. Reset at the start of every
  // RecoverBdev. Atomic because Monitor() may run on a different worker fiber
  // than the in-progress rebuild.
  //==========================================================================
  std::atomic<clio::run::u64> recovery_ops_total_{0};      // slots to rebuild
  std::atomic<clio::run::u64> recovery_ops_completed_{0};  // slots rebuilt so far
  std::atomic<clio::run::u64> recovery_ops_in_flight_{0};  // slot mid-write now
  std::atomic<clio::run::u32> recovery_active_{0};         // 1 while a rebuild runs
  std::atomic<clio::run::u32> recovering_is_parity_{0};    // member role rebuilt
  std::atomic<int> recovering_index_{-1};                  // member index rebuilt

  // Async-parity bookkeeping. Write writes data chunks immediately and records
  // each touched SLOT as dirty (parity not yet current); BuildParity drains the
  // dirty set and (re)computes parity for each stripe under its width's code.
  // written_slots_ tracks every slot that currently holds data so a parity-level
  // increase (AddBdev as_parity) can re-dirty exactly those. Guarded by slot_mu_
  // (never held across a co_await). A slot is safe to reconstruct only when NOT
  // dirty -- degraded reads / recovery refuse a dirty (unprotected) slot.
  std::set<clio::run::u64> dirty_slots_;
  /** Bumped by every dirty mark of a slot (slot_mu_). BuildParity clears a
   *  slot only if its generation is unchanged since it read the stripe: a
   *  write that re-dirtied it mid-build (the slot already in the set, so the
   *  insert was a no-op) must not be erased along with the stale parity. */
  std::unordered_map<clio::run::u64, clio::run::u64> slot_gen_;
  // Every slot below any member's high water: parity covers the physical
  // bytes of every data column there, allocated or not (#1126), so a freed
  // chunk stays part of its stripe and recovery must rebuild it too.
  std::set<clio::run::u64> written_slots_;
  mutable std::mutex slot_mu_;

  // Slot-allocator lock. A container's methods are NOT serialized: client
  // threads hash to different lanes (map_by_pid_tid), so several workers can
  // be inside this container at once. data_alloc_ (per-member std::set + free
  // list), rr_cursor_ and alloc_log_ are mutated by AllocateBlocks/FreeBlocks
  // and read by GetStats, so every access takes this lock. Without it,
  // concurrent Take()/Release() corrupt the heap (observed as
  // "malloc(): unaligned tcache chunk detected" under >= 4 client threads,
  // preceded by "block ... not live" warnings from lost set updates).
  //
  // Lock order is alloc_mu_ -> slot_mu_ (FreeBlocks marks slots dirty while
  // holding it); never take alloc_mu_ while holding slot_mu_. Helpers that
  // read data_alloc_ (StripeMembers, ForgetSlotIfEmpty) require the caller to
  // hold alloc_mu_ and must not re-acquire it.
  //
  // The membership-change paths (AddBdev / RemoveBdev / RecoverBdev)
  // push_back/pop_back on the member vectors across co_awaits without this
  // lock; the vectors reserve kMaxMembers at Create, so those appends never
  // reallocate under a concurrent data-plane task.
  mutable std::mutex alloc_mu_;

  // Stripes held by an in-flight Write/encode (#1121), guarded by slot_mu_.
  // A writer takes ALL of its stripes at once or none (TryLockStripes), so
  // writers cannot deadlock; holders keep them across co_awaits from before
  // their data writes until their parity is written.
  std::set<clio::run::u64> busy_stripes_;

  /** Note a slot holds data, without marking its parity stale (#1121: the
   *  caller encodes it under the stripe lock right away). */
  void NoteSlotWritten(clio::run::u64 s) {
    std::lock_guard<std::mutex> g(slot_mu_);
    written_slots_.insert(s);
  }
  /** Mark a slot as holding data and needing (re)parity. */
  void MarkSlotDirty(clio::run::u64 s) {
    std::lock_guard<std::mutex> g(slot_mu_);
    written_slots_.insert(s);
    dirty_slots_.insert(s);
    ++slot_gen_[s];
  }
  /**
   * The RS data columns of every stripe: all data members, in member order
   * (fixed-width parity, #1126). Column d's coefficients do not depend on
   * how many columns exist, so an unallocated or freed chunk is just a
   * column holding whatever bytes it holds.
   * @return 0 .. data_members_.size()-1
   */
  /**
   * Restart: parity covers every slot below the array high water (from the
   * alloc log; at least every member's live high water). An array whose
   * parity predates the fixed-width format gets every slot marked dirty, so
   * BuildParity re-encodes it and nothing is reconstructed from old parity.
   * Caller has exclusive access (Create).
   */
  void RestoreParityCoverage();

  std::vector<int> CodeColumns() const {
    std::vector<int> cols(data_members_.size());
    for (size_t d = 0; d < cols.size(); ++d) cols[d] = static_cast<int>(d);
    return cols;
  }
  /** True if the slot's parity is not yet current (unprotected). */
  bool IsSlotDirty(clio::run::u64 s) const {
    std::lock_guard<std::mutex> g(slot_mu_);
    return dirty_slots_.count(s) != 0;
  }

  //==========================================================================
  // Banding address helpers (pure integer decode/encode; see class comment).
  //==========================================================================

  /** Logical byte offset of the chunk base for (member d, slot s). */
  static clio::run::u64 BandOffset(clio::run::u32 d, clio::run::u64 s) {
    return (static_cast<clio::run::u64>(d) * kSlotsPerMember + s) * kChunkLen;
  }
  /** Decode a logical byte offset into (member d, slot s, within-chunk). */
  static void Unband(clio::run::u64 off, clio::run::u32 &d, clio::run::u64 &s,
                     clio::run::u64 &within) {
    const clio::run::u64 chunk = off / kChunkLen;
    d = static_cast<clio::run::u32>(chunk / kSlotsPerMember);
    s = chunk % kSlotsPerMember;
    within = off % kChunkLen;
  }
  /** Absolute member-pool offset of slot `s`'s chunk start. */
  static clio::run::u64 SlotPhysOffset(clio::run::u64 s) {
    return kSuperblockSize + s * kChunkLen;
  }

  /** The data-member indices (sorted ascending) with slot `s` live -- the
   *  membership (and RS data-shard order) of stripe `s`. */
  std::vector<int> StripeMembers(clio::run::u64 s) const {
    std::vector<int> mem;
    for (size_t d = 0; d < data_alloc_.size(); ++d) {
      if (data_alloc_[d].live_.count(s) != 0) {
        mem.push_back(static_cast<int>(d));
      }
    }
    return mem;  // ascending by construction
  }

  /** @return slots of the smallest parity member (~0 with none): no data
   *  member may allocate a slot past it -- its parity would not fit. */
  clio::run::u64 MinParityCap() const {
    clio::run::u64 m = ~0ULL;
    for (const auto &p : parity_members_) m = std::min(m, p.cap_slots_);
    return m;
  }
  /** @return the highest slot any data member has used (+1). */
  clio::run::u64 MaxDataHighWater() const {
    clio::run::u64 m = 0;
    for (const auto &a : data_alloc_) m = std::max(m, a.high_water_);
    return m;
  }
  /** Clamp every data member's capacity to what the parity members can
   *  cover (never below what it already used). Caller holds alloc_mu_ or
   *  has exclusive access (Create). */
  void ClampDataCaps() {
    const clio::run::u64 lim = MinParityCap();
    for (auto &a : data_alloc_) {
      a.cap_slots_ = std::max(a.high_water_, std::min(a.cap_slots_, lim));
    }
  }
  /**
   * Chunk slots a member device can hold, from its bdev stats.
   * @param client the member's bdev client
   * @param q route to the member
   * @param slots receives the slot count
   * @param ok receives false if the device did not answer
   */
  clio::run::TaskResume QueryMemberSlots(clio::run::bdev::Client client,
                                         clio::run::PoolQuery q,
                                         clio::run::u64 &slots, bool &ok);

  /**
   * The array's usable capacity and what is left of it: active data members
   * contribute their (parity-clamped) slots, a down member only its live
   * chunks -- still stored, served degraded -- and no free space.
   * @param total receives usable bytes
   * @param remaining receives free usable bytes
   */
  void ArrayCapacity(clio::run::u64 *total, clio::run::u64 *remaining);

  /** StripeHasDownMember for callers that do not hold alloc_mu_. */
  bool StripeHasDownMemberLocked(clio::run::u64 s) const {
    std::lock_guard<std::mutex> g(alloc_mu_);
    return StripeHasDownMember(s);
  }
  /** @return true if any data column of stripe `s` is down: every stripe
   *  spans all data members (fixed-width parity). */
  bool StripeHasDownMember(clio::run::u64 s) const {
    (void)s;
    for (int d : CodeColumns()) {
      if (data_members_[static_cast<size_t>(d)].state_ !=
          ec::EcState::kActive) {
        return true;
      }
    }
    return false;
  }

  /** RS codec for a stripe of width `k` (RS(k, max_failures_)); cached. */
  ec::ReedSolomon *GetCodec(int k) {
    auto it = rs_cache_.find(k);
    if (it == rs_cache_.end()) {
      it = rs_cache_
               .emplace(k, std::make_unique<ec::ReedSolomon>(
                              k, static_cast<int>(max_failures_)))
               .first;
    }
    return it->second.get();
  }

  /** Query for this array's own pool (periodic self-tasks). */
  clio::run::PoolQuery SelfQuery() const { return clio::run::PoolQuery::Local(); }
  /**
   * Route to one member's container: its node's in a distributed array,
   * this node's otherwise.
   * @param m the member
   */
  clio::run::PoolQuery MemberQuery(const MemberSlot &m) const {
    return distributed_ ? clio::run::PoolQuery::Physical(m.node_id_)
                        : clio::run::PoolQuery::Local();
  }
  /** @return route to data member `d`. */
  clio::run::PoolQuery DataQuery(size_t d) const {
    return MemberQuery(data_members_[d]);
  }
  /** @return route to parity member `j`. */
  clio::run::PoolQuery ParityQuery(size_t j) const {
    return MemberQuery(parity_members_[j]);
  }
  /**
   * Distributed array: mark every active member whose node has died faulty
   * (the node-level "unplugged disk"); degraded reads and writes then route
   * around it until RecoverBdev. No-op for a node-local array.
   */
  void FaultMembersOnDeadNodes();
  /** @return true if this container runs the array (always, unless the
   *  array is distributed and this is not its home node). */
  bool IsHome() const { return !distributed_ || is_home_; }

  /**
   * Take a member out of service at runtime: kActive -> kFaulty, logged once
   * at kError and persisted in the member manifest (exactly as
   * RemoveBdev(was_faulty) records it), so a restart keeps the member down.
   * Reads then reconstruct its chunks and writes keep its bytes in the
   * parity. Safe from concurrent task fibers: only the call that makes the
   * transition logs and persists. Must not be called holding alloc_mu_ or
   * slot_mu_ (it takes member_log_mu_ and does file I/O).
   * @param is_parity true for parity_members_, false for data_members_
   * @param idx position in that vector
   * @param why short reason for the log line
   * @return true if this call faulted the member
   */
  bool FaultMember(bool is_parity, size_t idx, const char *why);

  /**
   * Automatic down-detection after a member I/O: a request the DEVICE failed
   * (ctp::IsFatalDevice: a media error or a vanished device) faults the
   * member, so a dying disk leaves the array instead of surfacing I/O errors
   * to the caller. Any other failure -- one the device did not attribute to
   * itself (kOk: a rejected range, a short count, no buffer), a full device
   * (kNoSpace) or a transient error -- fails only that request: faulting on
   * those took every member of a healthy array down under tier pressure.
   * @param is_parity true for a parity member
   * @param idx its position in the member vector
   * @param failed whether the request failed
   * @param io_error the request's ctp::IoError category
   */
  void FaultOnIoError(bool is_parity, size_t idx, bool failed,
                      clio::run::u32 io_error) {
    if (!failed) return;
    const auto e = static_cast<ctp::IoError>(io_error);
    if (ctp::IsFatalDevice(e)) {
      FaultMember(is_parity, idx, ctp::IoErrorName(e));
      return;
    }
    HLOG(kWarning, "safe_bdev: a request to {} member {} failed ({}); not a "
         "device fault, the member stays in the array",
         is_parity ? "parity" : "data", idx, ctp::IoErrorName(e));
  }

  /** @return true if data member `d` is serving I/O. */
  bool DataActive(size_t d) const {
    return data_members_[d].state_.Load() == ec::EcState::kActive;
  }

  //==========================================================================
  // Synchronous Parity (issue #1121): per-stripe locking and intent logging
  //==========================================================================

  /**
   * Take every stripe in `slots` if none is held by another task.
   * @param slots the stripes
   * @return true if all were taken (the caller must UnlockStripes them)
   */
  bool TryLockStripes(const std::set<clio::run::u64> &slots);

  /**
   * Test-only fault injection, like the file bdev's `.fail` marker: while
   * $CLIO_SAFE_BDEV_FAULT_SKIP_PARITY is set, writes land their data but not
   * their parity and the background builder stays idle -- the state a crash
   * between the two leaves on disk.
   * @return true while the fault is set
   */
  static bool FaultSkipParity() {
    const char *e = std::getenv("CLIO_SAFE_BDEV_FAULT_SKIP_PARITY");
    return e != nullptr && e[0] == '1';
  }

  /**
   * Take every stripe in `slots`, yielding the worker until none is held.
   * @param slots the stripes
   */
  clio::run::TaskResume LockStripes(const std::set<clio::run::u64> &slots);

  /** Release stripes taken by TryLockStripes/LockStripes. */
  void UnlockStripes(const std::set<clio::run::u64> &slots);

  /**
   * Encode stripe `s`'s parity from its current data chunks and write every
   * active parity shard; on success record the member set it was encoded
   * over (encoded_) and clear the slot's dirty mark unless a write
   * re-dirtied it meanwhile. The caller must hold the stripe.
   * @param s the stripe (slot)
   * @param ok false if a data member is down or any read/write failed
   */
  clio::run::TaskResume EncodeStripe(clio::run::u64 s, bool &ok);



  /** @return the number of members (data + parity) not serving I/O. */
  clio::run::u32 CountDownMembers() const {
    clio::run::u32 n = 0;
    for (const auto &m : data_members_) {
      if (m.state_.Load() != ec::EcState::kActive) ++n;
    }
    for (const auto &m : parity_members_) {
      if (m.state_.Load() != ec::EcState::kActive) ++n;
    }
    return n;
  }

  //==========================================================================
  // EC / I/O helpers (defined in safe_bdev_runtime.cc; run inside task fibers).
  //==========================================================================

  /** Block list addressing [offset, offset+len) on a member pool. */
  clio::run::priv::vector<clio::run::bdev::Block> MemberBlocks(clio::run::u64 offset,
                                                         clio::run::u64 len) const;

  /**
   * AsyncWrite `len` bytes from host buffer `src` to DATA member `d` at absolute
   * member-pool offset `offset`. Auto-faults the member on a fatal io_error.
   */
  clio::run::TaskResume WriteDataSegment(size_t d, clio::run::u64 offset,
                                   const uint8_t *src, clio::run::u64 len,
                                   bool &ok);

  /**
   * AsyncRead `len` bytes at absolute member-pool offset `offset` from DATA
   * member `d` into host buffer `dst`. Auto-faults the member on a fatal
   * io_error.
   */
  clio::run::TaskResume ReadDataSegment(size_t d, clio::run::u64 offset, uint8_t *dst,
                                  clio::run::u64 len, bool &ok);

  /**
   * Reconstruct ALL k_s data chunks of stripe `s`: the members its parity
   * encodes (encoded_) are decoded from active survivors under that set's
   * code, not reading the data members in `exclude` (their on-disk chunks
   * are not usable); a member the parity has not seen yet (allocated since)
   * holds no data the parity could give back, so its chunk is read as it is
   * on disk (zeros if it is down). `stripe` is StripeMembers(s) (the sorted
   * data-member indices in this stripe); `out` receives k_s buffers of
   * kChunkLen bytes indexed by stripe POSITION. Returns false on too few
   * survivors.
   */
  clio::run::TaskResume ReconstructStripe(clio::run::u64 s,
                                          const std::vector<int> &stripe,
                                          const std::vector<int> &exclude,
                                          std::vector<std::vector<uint8_t>> &out,
                                          bool &ok);
  /**
   * Read up to k = code.size() usable shards of slot `s`: data members of
   * the encoded set `code` that are active and not in `exclude`, then
   * active parity members.
   * @param s slot
   * @param code the data members the slot's parity encodes
   * @param exclude data members not to read
   * @param idx receives each shard's RS index (code position, or k + row)
   * @param bufs receives the shards
   */
  clio::run::TaskResume GatherSurvivors(clio::run::u64 s,
                                        const std::vector<int> &code,
                                        const std::vector<int> &exclude,
                                        std::vector<int> &idx,
                                        std::vector<std::vector<uint8_t>> &bufs);

  /**
   * Serialize this array's identity for a member into a zero-padded
   * kSuperblockSize buffer and AsyncWrite it to that member at absolute offset
   * 0. `is_parity` selects parity_members_/parity_clients_ vs data; `idx` is
   * the position in that vector. Returns false on I/O failure.
   */
  clio::run::TaskResume WriteSuperblock(bool is_parity, size_t idx, bool &ok);

  /**
   * AsyncRead kSuperblockSize bytes at absolute offset 0 from a member, parse
   * the superblock into `sb`, and set `present` = (magic matches && checksum
   * valid). A blank member reads back zeros => present == false. `is_parity`
   * selects the member vector; `idx` is its position.
   */
  clio::run::TaskResume ReadSuperblock(bool is_parity, size_t idx,
                                 MemberSuperblock &sb,
                                 bool &present, bool &ok);

  //==========================================================================
  // Durable member manifest (membership + recovery state persistence).
  //==========================================================================

  /** One persisted member record. Mirrors a MemberSlot's durable fields. */
  struct MemberManifestEntry {
    clio::run::u32 role_ = 0;        // ec::EcRole (0=data, 1=parity)
    clio::run::u32 index_ = 0;       // data column / parity row
    clio::run::u32 pool_major_ = 0;  // member bdev pool id
    clio::run::u32 pool_minor_ = 0;
    clio::run::u32 node_id_ = 0;
    clio::run::u32 state_ = 0;        // ec::EcState (0=active,1=faulty,2=removed)
    clio::run::u32 recovering_ = 0;  // 1 if mid-recovery onto (pool_major/minor)
    std::string pool_name_;
  };

  /** APPEND a snapshot of the current data/parity members to the member-log
   *  WAL (cheap; no rewrite/rename). No-op when logging is disabled. */
  void PersistMemberManifest();

  /** COMPACT the member-log WAL to one record per current member (temp file +
   *  atomic rename). Called after Create restores membership and periodically
   *  once the on-disk record count grows past a threshold. */
  void CompactMemberManifest();

  /** True if the member log has grown enough to warrant compaction. */
  bool MemberManifestNeedsCompaction() const;

  /** Replay the member-log WAL into `out`, keeping the last record per
   *  (role,index) slot. Returns false if absent/empty. */
  bool LoadMemberManifest(std::vector<MemberManifestEntry> &out) const;

  /** Write one member record to an open file handle (append or compact). */
  void WriteMemberRecord(std::FILE *f, const MemberSlot &m,
                         clio::run::u32 role) const;

  /**
   * A stripe a write touches while one of its data members is down, held
   * across the write: reconstructed BEFORE any member write lands (its
   * parity is still consistent then), overlaid with every byte the write
   * puts in the stripe, and re-encoded into every live parity shard after.
   */
  struct DegradedStripe {
    std::vector<int> members;                  // StripeMembers(s)
    std::vector<std::vector<uint8_t>> chunks;  // by stripe position
  };
  /** One piece of a Write that falls inside a single chunk. */
  struct WritePiece {
    clio::run::u64 slot;     // stripe (slot) it lands in
    clio::run::u32 member;   // data member d
    clio::run::u64 within;   // offset inside the chunk
    clio::run::u64 len;      // bytes
    clio::run::u64 buf_off;  // offset in the task's data buffer
  };
  /** The member writes one Write dispatched (index-aligned vectors). */
  struct MemberWrites {
    std::vector<clio::run::Future<WriteTask>> futs;
    std::vector<size_t> members;                // data member of each future
    std::vector<ctp::ipc::FullPtr<char>> bufs;  // staging buffers to free
    std::set<clio::run::u64> touched;           // slots whose data changes
    clio::run::u64 bytes = 0;                   // bytes the write covers
  };
  /**
   * @param task a Write
   * @return false (logged) if a block decodes to a member that does not exist
   */
  bool BlocksInRange(const WriteTask &task) const;
  /**
   * Stage and send one AsyncWrite per block to its member, skipping blocks
   * on down members (their bytes live in the parity).
   * @param task the write
   * @param data its buffer
   * @param mw receives the dispatched writes
   * @return false if a staging buffer could not be allocated
   */
  bool DispatchMemberWrites(const WriteTask &task, const char *data,
                            MemberWrites &mw);
  /**
   * Whether stripe `s` can take a delta parity update: healthy, not dirty,
   * and its parity encodes exactly its current members.
   * @param s the stripe
   */
  bool DeltaEligible(clio::run::u64 s);

  /**
   * Stripes a write rewrites whole (#1126): every data column's full chunk,
   * all of them up. Their parity follows from the write's own bytes, with
   * no reads.
   * @param pieces the write's pieces
   * @return the full stripes
   */
  std::set<clio::run::u64> FullStripes(const std::vector<WritePiece> &pieces);

  /**
   * Encode a stripe the write covers whole from the write's bytes and write
   * every live parity shard concurrently.
   * @param s the stripe (in FullStripes)
   * @param pieces the write's pieces
   * @param data the write's bytes
   * @param ok receives true when every live shard was written
   */
  clio::run::TaskResume EncodeFullStripe(clio::run::u64 s,
                                         const std::vector<WritePiece> &pieces,
                                         const char *data, bool &ok);

  /**
   * Before a write lands, read the bytes each of its pieces replaces, for
   * the stripes that can take a delta update (#1126).
   * @param pieces the write's pieces
   * @param skip stripes not to read (rewritten whole: FullStripes)
   * @param slots receives the delta-eligible stripes
   * @param old_bytes receives, per piece, the bytes it replaces (empty for
   *        pieces of other stripes)
   */
  clio::run::TaskResume ReadReplacedBytes(
      const std::vector<WritePiece> &pieces,
      const std::set<clio::run::u64> &skip, std::set<clio::run::u64> &slots,
      std::vector<std::vector<uint8_t>> &old_bytes);

  /**
   * Bring stripe `s`'s parity up to date from the change alone: for each
   * piece, parity_j[range] += c(j, pos) * (old ^ new). Reads and writes only
   * the written ranges of each parity shard instead of the whole stripe. The
   * caller holds the stripe; on failure it must fully re-encode.
   * @param s the stripe
   * @param pieces the write's pieces
   * @param data the write's bytes
   * @param old_bytes per piece, the bytes it replaced (ReadReplacedBytes)
   * @param ok false if the delta could not be applied
   */
  clio::run::TaskResume DeltaEncodeStripe(
      clio::run::u64 s, const std::vector<WritePiece> &pieces,
      const char *data, const std::vector<std::vector<uint8_t>> &old_bytes,
      bool &ok);

  /**
   * parity_j[offset, offset+len) += coeff * delta (GF(2^8)), in place.
   * @param j parity row
   * @param offset member offset
   * @param delta the data change (old ^ new)
   * @param len bytes
   * @param coeff the code coefficient for the changed data column
   * @param ok false on an I/O failure
   */
  clio::run::TaskResume ParityRangeAdd(size_t j, clio::run::u64 offset,
                                       const uint8_t *delta,
                                       clio::run::u64 len, uint8_t coeff,
                                       bool &ok);
  /**
   * The body of a Write once its stripes are held and logged dirty: land the
   * data (degraded stripes reconstructed and re-encoded), then encode the
   * parity of every healthy stripe it changed.
   * @param task the write
   * @param pieces the write split per stripe/member (SplitWrite)
   * @param data the write's bytes
   * @param ok false if the write must be failed
   * @param clean stripes whose parity is current afterwards (log them clean)
   * @param intents the write's stripe intents (journal keys for degraded
   *        stripes; empty when the intent log is disabled)
   */
  clio::run::TaskResume WriteStripes(clio::run::shared_ptr<WriteTask> &task,
                                     const std::vector<WritePiece> &pieces,
                                     const char *data, bool &ok,
                                     std::set<clio::run::u64> &clean,
                                     const std::vector<IntentKey> &intents);
  /**
   * Journal the down data columns of each degraded stripe (#1137) before any
   * byte of the write lands, so a crash mid-write cannot strand them.
   * @param degraded the write's degraded stripes, reconstructed
   * @param intents the write's stripe intents (keys the records)
   * @return false if a record could not be written (the write must fail)
   */
  bool JournalDownColumns(const std::map<clio::run::u64, DegradedStripe> &degraded,
                          const std::vector<IntentKey> &intents);
  /**
   * Await a Write's member writes and free their staging buffers. A member
   * whose write failed is faulted unless the error was transient.
   * @param mw the dispatched writes
   * @param any_failed set true if a member failed and was faulted
   * @param ok set false if a member failed transiently (still active)
   */
  clio::run::TaskResume AwaitMemberWrites(MemberWrites &mw, bool &any_failed,
                                          bool &ok);
  /**
   * Split a Write's blocks into per-chunk pieces.
   * @param task the write
   * @return one WritePiece per (block, chunk) intersection, in buffer order
   */
  std::vector<WritePiece> SplitWrite(const WriteTask &task) const;
  /**
   * Copy every byte the write puts in stripe `s` into its chunks.
   * @param s slot
   * @param pieces the write's pieces
   * @param data the write's buffer
   * @param st the stripe to overlay
   */
  static void OverlayPieces(clio::run::u64 s,
                            const std::vector<WritePiece> &pieces,
                            const char *data, DegradedStripe &st);
  /**
   * Whether a down member of stripe `s` needs a byte (inside its live
   * length, not rewritten by this write) at a position this write already
   * changed on a surviving member: the parity can then not produce it.
   * @param s slot
   * @param st the stripe (members)
   * @param need_len live bytes of each stripe position's chunk
   * @param pieces the write's pieces
   * @return true if so (logged)
   */
  bool DownBytesClobbered(clio::run::u64 s, const DegradedStripe &st,
                          const std::vector<clio::run::u64> &need_len,
                          const std::vector<WritePiece> &pieces) const;
  /**
   * A member write of this request failed and the member was faulted after
   * its siblings in the same stripe had already landed: redo stripe `s`
   * degraded. The old stripe is decoded from the members this write left
   * untouched and the parity (current: the slot was clean), overlaid with
   * this write's bytes, and the parity re-encoded over it. With too many
   * erasures for that, falls back to decoding from the survivors as they
   * are, and fails (never guesses) when a byte a down member still needs
   * was overwritten on a survivor by this same write.
   * @param s slot
   * @param pieces the write's pieces
   * @param data the write's buffer
   * @param ok receives success
   */
  clio::run::TaskResume RetryStripeDegraded(
      clio::run::u64 s, const std::vector<WritePiece> &pieces,
      const char *data, bool &ok,
      const std::vector<std::vector<uint8_t>> *old_bytes = nullptr);
  /**
   * Decode stripe `s` as it was BEFORE this write (#1139): the survivors'
   * chunks with every landed piece swapped back to the bytes it replaced,
   * which is what the (not yet updated) parity encodes. Works whenever the
   * down members alone fit in the parity, however many members the write
   * already changed.
   * @param s slot
   * @param stripe stripe columns (CodeColumns())
   * @param pieces this write's pieces
   * @param old_bytes the bytes each piece replaced (ReadReplacedBytes)
   * @param out decoded chunks, one per stripe position
   * @param ok true on success
   */
  clio::run::TaskResume ReconstructStripeAsBefore(
      clio::run::u64 s, const std::vector<int> &stripe,
      const std::vector<WritePiece> &pieces,
      const std::vector<std::vector<uint8_t>> &old_bytes,
      std::vector<std::vector<uint8_t>> &out, bool &ok);
  /**
   * Reconstruct stripe `s` for a degraded write. Refuses (ok=false) when its
   * parity is stale: the down member's bytes would then exist nowhere.
   * @param s slot
   * @param out receives the stripe
   * @param ok receives success
   */
  clio::run::TaskResume LoadDegradedStripe(clio::run::u64 s,
                                           DegradedStripe &out, bool &ok);
  /**
   * Re-encode a degraded stripe's parity and write every live shard.
   * @param s slot
   * @param st the overlaid stripe
   * @param ok receives true if at least one parity shard was written
   */
  clio::run::TaskResume StoreDegradedParity(clio::run::u64 s,
                                            const DegradedStripe &st,
                                            bool &ok);
  /** @return data members of stripe `st` that are down. */
  clio::run::u32 DownInStripe(const DegradedStripe &st) const {
    clio::run::u32 n = 0;
    for (int d : st.members) {
      if (!DataActive(static_cast<size_t>(d))) ++n;
    }
    return n;
  }
  /**
   * Read one block of a Read request from down member `d`'s stripes by
   * reconstruction, chunk by chunk.
   * @param off logical offset of the block
   * @param len its length
   * @param dst destination
   * @param ok receives success
   */
  clio::run::TaskResume ReadBlockDegraded(clio::run::u64 off,
                                          clio::run::u64 len, char *dst,
                                          bool &ok);

  /** One data column to seat at Create: where it is and what state the
   *  manifest left it in. */
  struct DataSeatSpec {
    MemberBdevDesc desc;
    clio::run::u32 state = 0;  // ec::EcState as persisted (0 = active)
    bool recovering = false;
  };
  /**
   * The data columns to seat: the configured members, overridden and
   * extended by the manifest. The manifest is the truth for membership: a
   * column recovered onto another disk, a faulty column, and every data
   * member added at runtime (which the config never listed) all come from
   * it, so a restart brings back the array that was running.
   * @param params Create parameters (configured members)
   * @param manifest replayed member manifest (may be empty)
   * @return one spec per data column, in column order
   */
  std::vector<DataSeatSpec> BuildDataMemberPlan(
      const CreateParams &params,
      const std::vector<MemberManifestEntry> &manifest) const;
  /**
   * Seat data column `col`: size its allocator from the member, then
   * stamp (fresh) or re-attach (ours) its superblock; refuse a foreign one.
   * A column the manifest recorded as faulty/removed is seated without
   * touching its device (it may be gone) and takes no new allocations.
   * @param spec the column
   * @param col its index
   * @param rc receives 0, or the Create return code (1 I/O, 2 foreign)
   */
  clio::run::TaskResume SeatDataMember(DataSeatSpec spec, int col,
                                       clio::run::u32 &rc);
  /**
   * Seat the parity columns at Create: from the manifest when it records
   * any (it is the truth: recoveries, faulty members), else from the
   * configured members marked `parity: true`.
   * @param params Create parameters
   * @param manifest_parity manifest parity entries, sorted by row
   * @param rc receives 0, or the Create return code (1 I/O, 2 foreign,
   *           3 more parity members than max_failures)
   */
  clio::run::TaskResume SeatParityMembers(
      const CreateParams &params,
      const std::vector<MemberManifestEntry> &manifest_parity,
      clio::run::u32 &rc);
  /**
   * Seat one configured (`parity: true`) parity column: stamp a fresh
   * device, re-attach our own, refuse a foreign one.
   * @param desc the member
   * @param rc receives 0, 1 (I/O) or 2 (foreign)
   * @param fresh receives true if its superblock was just written (its
   *              parity is not built yet)
   */
  clio::run::TaskResume SeatConfigParity(MemberBdevDesc desc,
                                         clio::run::u32 &rc, bool &fresh);

  /** Reconstruct + write EVERY slot below the array high water onto the
   *  member's (already-seated) client -- fixed-width parity covers every
   *  column's physical bytes, allocated or not (#1146). Slots that writes
   *  touch meanwhile are rebuilt again, and the last pass runs with writes
   *  gated: on success with `completed`, the write gate is left CLOSED so
   *  the caller can mark the member active before any write sees it
   *  non-active; the caller then calls OpenWriteGate(). Idempotent: safe to
   *  re-run after an interrupted recovery. On return `ok` reports I/O
   *  success and `completed` is false only when the
   *  CLIO_SAFE_BDEV_RECOVER_MAX_ROWS test hook stopped the rebuild early
   *  (recovery left in-progress; the gate is open). */
  clio::run::TaskResume RebuildMember(bool is_data, int idx, bool &ok,
                                      bool &completed);

  /**
   * Rebuild one slot of a recovering member under its stripe lock.
   * @param is_data data (true) or parity (false) member
   * @param idx the member's column / row
   * @param s the slot
   * @param ok receives false on an I/O or reconstruction failure
   */
  clio::run::TaskResume RebuildSlot(bool is_data, int idx, clio::run::u64 s,
                                    bool &ok);

  /** Rebuild every slot recorded in rebuild_redo_ (and clear them).
   * @param is_data data (true) or parity (false) member
   * @param idx the member's column / row
   * @param ok receives false on a failure */
  clio::run::TaskResume RebuildRedo(bool is_data, int idx, bool &ok);

  /**
   * Record stripes a write or encode is about to change while a member is
   * being rebuilt: the rebuild redoes them, since the recovering member does
   * not take writes (#1146).
   * @param slots the stripes
   */
  void NoteRebuildWrites(const std::set<clio::run::u64> &slots) {
    std::lock_guard<std::mutex> g(rebuild_mu_);
    if (rebuild_tracking_) rebuild_redo_.insert(slots.begin(), slots.end());
  }

  /** Let writes proceed again (after a rebuilt member was marked active). */
  void OpenWriteGate() {
    write_gate_.store(false, std::memory_order_seq_cst);
  }

  /** Stop tracking writes for a rebuild and open the gate (failure paths). */
  void EndRebuildTracking() {
    {
      std::lock_guard<std::mutex> g(rebuild_mu_);
      rebuild_tracking_ = false;
      rebuild_redo_.clear();
    }
    OpenWriteGate();
  }

  /** Writes wait here while a rebuild's final pass runs; then count
   *  themselves in flight (seq_cst pairs with the gate). */
  clio::run::TaskResume EnterWrite();

  /** A write's member and parity I/O is done. */
  void LeaveWrite() {
    writes_in_flight_.fetch_sub(1, std::memory_order_seq_cst);
  }

  // Rebuild-under-load state (#1146).
  std::mutex rebuild_mu_;
  bool rebuild_tracking_ = false;              // guarded by rebuild_mu_
  std::set<clio::run::u64> rebuild_redo_;      // guarded by rebuild_mu_
  std::atomic<bool> write_gate_{false};
  std::atomic<clio::run::u64> writes_in_flight_{0};
  /** Re-check period while waiting on the write gate / in-flight writes. */
  static constexpr double kWriteGatePollUs = 200.0;
  /** Redo passes before the final, gated one. */
  static constexpr int kRebuildRedoPasses = 4;

  /**
   * Make sure a node-local member's bdev pool exists before it is seated.
   * A member added or replaced at runtime (AddBdev / RecoverBdev) is not in
   * the server config's compose, which re-creates the array at a restart
   * BEFORE durable runtime pools come back -- the array then found no pool
   * for its replacement disk and every start failed (#1146). Its backing
   * file is re-attached here as a file bdev (its existing size) instead.
   * @param pool_id member pool id
   * @param pool_name member pool name (its backing file)
   */
  clio::run::TaskResume EnsureMemberPool(clio::run::PoolId pool_id,
                                         std::string pool_name);

  /** After membership is restored, resume any member left in the recovering
   *  state (crash mid-RecoverBdev). Persists the manifest as members come back
   *  online. */
  clio::run::TaskResume ResumeRecoveries();
};

}  // namespace clio::run::safe_bdev

#endif  // SAFE_BDEV_RUNTIME_H_
