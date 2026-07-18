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

#include <cstdio>  // std::FILE
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include <clio_ctp/io/io_error.h>  // ctp::IoError, ctp::IsFatalDevice

#include "ec/ec_array.h"  // ec::EcRole, ec::EcState, ec::ReedSolomon
#include "safe_bdev_client.h"
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
 * STRIPES & PARITY (offset-aligned, variable width). A STRIPE s = the live data
 * chunks at physical slot s across the data members deep enough to have one;
 * parity for stripe s lives at the SAME slot s on each parity member:
 *   parity[s] = RS(k_s, m) over { data_d[s] : member d has a LIVE chunk at s }
 * k_s (stripe width) varies by slot (StripeMembers(s) -> the sorted data-member
 * indices with slot s live; a member's RS data-shard index is its POSITION in
 * that sorted list). Codecs are cached by width in rs_cache_. Membership is
 * DERIVED from the per-member live sets, not stored. A single data chunk
 * (k_s==1) with m==1 is a MIRROR -- narrow writes are protected immediately and
 * widen to RS(k,m) as drives join. Parity is built off the write path (deferred
 * to a periodic BuildParity over a dirty-SLOT set); degraded reads reconstruct a
 * down member's chunk on demand from the stripe survivors + parity. Adding a
 * data drive moves NO data: new writes round-robin onto it, dirtying the slots
 * they widen so BuildParity re-derives that stripe's parity.
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
  ~Runtime() override = default;

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
  struct MemberSlot {
    clio::run::PoolId pool_id_;
    std::string pool_name_;
    clio::run::u32 node_id_ = 0;
    ec::EcRole role_ = ec::EcRole::kData;
    ec::EcState state_ = ec::EcState::kActive;
    int index_ = -1;  // data column d (DATA) or parity row j (PARITY)
    // True while this member's chunks are being rebuilt onto pool_id_ (a
    // recovery target seated by RecoverBdev). Persisted in the member manifest
    // so an interrupted recovery is RESUMED on restart: the member stays
    // non-active (excluded from reads -> degraded path) until the rebuild
    // completes. RebuildMember is idempotent, so re-running after a crash is
    // safe.
    bool recovering_ = false;
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
  clio::run::u32 parity_level_;           // Parity members added so far (m)
  clio::run::u32 reattached_members_;     // Members recognized as ours at Create

  //==========================================================================
  // Recovery observability. RecoverBdev rebuilds a failed member's chunk for
  // EVERY slot it participates in; each such slot is one "recovery operation".
  // These atomics let Monitor() -- and the context-visualizer safe-bdev
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
  std::set<clio::run::u64> written_slots_;
  mutable std::mutex slot_mu_;

  /** Mark a slot as holding data and needing (re)parity. */
  void MarkSlotDirty(clio::run::u64 s) {
    std::lock_guard<std::mutex> g(slot_mu_);
    written_slots_.insert(s);
    dirty_slots_.insert(s);
  }
  /** Note a slot no longer holds data (last live chunk freed): drop it from the
   *  written set once no data member has it live. Caller ensures liveness check
   *  already reflects the free. */
  void ForgetSlotIfEmpty(clio::run::u64 s) {
    for (const auto &a : data_alloc_) {
      if (a.live_.count(s) != 0) {
        return;  // still live somewhere -> keep tracking
      }
    }
    std::lock_guard<std::mutex> g(slot_mu_);
    written_slots_.erase(s);
    dirty_slots_.erase(s);
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

  /** Per-member pool query (members are independent local bdev pools). */
  clio::run::PoolQuery MemberQuery() const { return clio::run::PoolQuery::Local(); }

  /**
   * Automatic down-detection for a DATA member. Inspect a member-bdev future's
   * io_error_ after a chunk I/O completes: a fatal device error (DeviceFault /
   * Disconnected) ejects data member `d` (state_ = kFaulty) so the degraded-read
   * / reconstruct path takes over. A TRANSIENT error never faults the member.
   */
  void MaybeFaultData(size_t d, clio::run::u32 io_error) {
    const auto e = static_cast<ctp::IoError>(io_error);
    if (ctp::IsFatalDevice(e) && d < data_members_.size() &&
        data_members_[d].state_ == ec::EcState::kActive) {
      data_members_[d].state_ = ec::EcState::kFaulty;
      HLOG(kWarning,
           "safe_bdev: data member {} auto-faulted on fatal device error '{}' "
           "(io_error={})",
           d, ctp::IoErrorName(e), io_error);
    }
  }

  /** Automatic down-detection for a PARITY member (parity row j). */
  void MaybeFaultParity(size_t j, clio::run::u32 io_error) {
    const auto e = static_cast<ctp::IoError>(io_error);
    if (ctp::IsFatalDevice(e) && j < parity_members_.size() &&
        parity_members_[j].state_ == ec::EcState::kActive) {
      parity_members_[j].state_ = ec::EcState::kFaulty;
      HLOG(kWarning,
           "safe_bdev: parity member {} auto-faulted on fatal device error "
           "'{}' (io_error={})",
           j, ctp::IoErrorName(e), io_error);
    }
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
   * Reconstruct ALL k_s data chunks of stripe `s` from active survivors
   * (DecodeData under the width-k_s code), optionally excluding data member
   * `exclude_member`. `stripe` is StripeMembers(s) (the sorted data-member
   * indices in this stripe); `out` receives k_s buffers of kChunkLen bytes
   * indexed by stripe POSITION (shard index). Returns false on too few
   * survivors.
   */
  clio::run::TaskResume ReconstructStripe(clio::run::u64 s,
                                    const std::vector<int> &stripe,
                                    int exclude_member,
                                    std::vector<std::vector<uint8_t>> &out,
                                    bool &ok);

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

  /** Reconstruct + write EVERY slot this member participates in onto its
   *  (already-seated) client. Idempotent: safe to re-run after an interrupted
   *  recovery. On return `ok` reports I/O success and `completed` is false only
   *  when the CLIO_SAFE_BDEV_RECOVER_MAX_ROWS test hook stopped the rebuild
   *  early (recovery left in-progress). */
  clio::run::TaskResume RebuildMember(bool is_data, int idx, bool &ok,
                                      bool &completed);

  /** After membership is restored, resume any member left in the recovering
   *  state (crash mid-RecoverBdev). Persists the manifest as members come back
   *  online. */
  clio::run::TaskResume ResumeRecoveries();
};

}  // namespace clio::run::safe_bdev

#endif  // SAFE_BDEV_RUNTIME_H_
