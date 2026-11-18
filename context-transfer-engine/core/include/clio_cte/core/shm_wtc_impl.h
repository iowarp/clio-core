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
 * @file shm_wtc_impl.h
 * Inline definitions for shm_wtc.h. Included only from there.
 */

#ifndef CLIO_CTE_CORE_SHM_WTC_IMPL_H_
#define CLIO_CTE_CORE_SHM_WTC_IMPL_H_

#include <algorithm>
#include <bit>
#include <cmath>

#include <clio_ctp/introspect/system_info.h>
#include <clio_ctp/util/config_parse.h>
#include <clio_runtime/bdev/transports/mem_bdev_transport.h>

namespace clio::cte::core::wtc {

/** RAII stripe/tag lock. */
class Guard {
 public:
  explicit Guard(ctp::Mutex &m) : m_(m) { m_.Lock(0); }
  ~Guard() { m_.Unlock(); }
  Guard(const Guard &) = delete;
  Guard &operator=(const Guard &) = delete;

 private:
  ctp::Mutex &m_;
};

/** Round up to a multiple of kMinBuffer. */
inline u64 RoundBuffer(u64 n) {
  return ((n + kMinBuffer - 1) / kMinBuffer) * kMinBuffer;
}

inline bool View::Format(const std::string &name,
                         const ctp::ipc::MemoryBackendId &backend_id,
                         const Limits &lim, u32 node_id) {
  seg_size_ = lim.seg_size_;
  if (!backend_.shm_init(backend_id, seg_size_, name)) {
    return false;
  }
  // Map every page of the segment now (MADV_POPULATE_WRITE: maps, never
  // alters contents): otherwise each first write to a 4 KiB page faults in
  // the writing client, ~256 faults per MiB absorbed.
  ctp::SystemInfo::BulkFault(backend_.data_, backend_.data_capacity_);
  owner_ = true;
  mapped_ = true;
  alloc_ = backend_.MakeAlloc<WtcAlloc>();
  if (alloc_ == nullptr) {
    return false;
  }
  base_ = alloc_->GetBackendData();
  auto root_o = alloc_->AllocateOffset(sizeof(Root));
  if (root_o.IsNull()) {
    return false;
  }
  root_off_ = root_o.load();
  root_ = reinterpret_cast<Root *>(Ptr(root_off_));
  std::memset(static_cast<void *>(root_), 0, sizeof(Root));
  root_->tag_lock_.Init();
  for (u32 i = 0; i < kStripes; ++i) {
    root_->stripes_[i].Init();
  }
  u32 sps = lim.max_blobs_ / kStripes;
  root_->slots_per_stripe_ = sps == 0 ? 1 : sps;
  root_->max_tags_ = lim.max_tags_ == 0 ? 1 : lim.max_tags_;
  root_->evict_sample_ = lim.evict_sample_ == 0 ? 1 : lim.evict_sample_;
  root_->max_fill_ = lim.max_fill_;
  root_->half_life_ns_ = lim.half_life_ns_ == 0 ? 1 : lim.half_life_ns_;
  root_->node_id_ = node_id;
  root_->md_only_ = lim.md_only_ ? 1u : 0u;

  size_t blob_bytes = sizeof(BlobEntry) * NumSlots();
  auto b = alloc_->AllocateOffset(blob_bytes);
  size_t tag_bytes = sizeof(TagEntry) * root_->max_tags_;
  auto t = alloc_->AllocateOffset(tag_bytes);
  size_t name_bytes = sizeof(NameSlot) * root_->max_tags_ * 2;
  auto nm = alloc_->AllocateOffset(name_bytes);
  auto l = alloc_->AllocateOffset(sizeof(DirtyLog));
  root_->owner_count_ = seg_size_ / kMinBuffer + 1;
  size_t owner_bytes = sizeof(std::atomic<u32>) * root_->owner_count_;
  auto ow = alloc_->AllocateOffset(owner_bytes);
  if (b.IsNull() || t.IsNull() || l.IsNull() || ow.IsNull() || nm.IsNull()) {
    return false;
  }
  root_->names_off_ = nm.load();
  names_ = reinterpret_cast<NameSlot *>(Ptr(root_->names_off_));
  std::memset(static_cast<void *>(names_), 0, name_bytes);
  root_->owners_off_ = ow.load();
  owners_ = reinterpret_cast<std::atomic<u32> *>(Ptr(root_->owners_off_));
  std::memset(static_cast<void *>(owners_), 0, owner_bytes);
  root_->blobs_off_ = b.load();
  root_->tags_off_ = t.load();
  root_->log_off_ = l.load();
  blobs_ = reinterpret_cast<BlobEntry *>(Ptr(root_->blobs_off_));
  tags_ = reinterpret_cast<TagEntry *>(Ptr(root_->tags_off_));
  std::memset(static_cast<void *>(blobs_), 0, blob_bytes);
  std::memset(static_cast<void *>(tags_), 0, tag_bytes);
  log_ = new (Ptr(root_->log_off_)) DirtyLog(alloc_, lim.log_depth_);
  if (!lim.extents_.empty() && !lim.md_only_) {
    // Extent mode: index every leased granule, in lease order, and mark
    // where one leased block ends and a non-adjacent one begins (a run may
    // never cross that point, so a run's bytes are contiguous).
    u64 gcount = 0;
    for (const auto &x : lim.extents_) gcount += x.second / kGranule;
    if (gcount == 0 || gcount > 0xFFFFFFFFull) return false;
    const u64 words = (gcount + 63) / 64;
    auto go = alloc_->AllocateOffset(sizeof(u64) * gcount);
    auto fb = alloc_->AllocateOffset(sizeof(u64) * words);
    auto bb = alloc_->AllocateOffset(sizeof(u64) * words);
    auto rs = alloc_->AllocateOffset(sizeof(RunSet) * NumSlots());
    auto ow = alloc_->AllocateOffset(sizeof(std::atomic<u32>) * gcount);
    if (go.IsNull() || fb.IsNull() || bb.IsNull() || rs.IsNull() ||
        ow.IsNull()) {
      return false;
    }
    root_->gran_owner_off_ = ow.load();
    gran_owner_ = reinterpret_cast<std::atomic<u32> *>(Ptr(root_->gran_owner_off_));
    std::memset(static_cast<void *>(gran_owner_), 0,
                sizeof(std::atomic<u32>) * gcount);
    root_->runsets_off_ = rs.load();
    runsets_ = reinterpret_cast<RunSet *>(Ptr(root_->runsets_off_));
    std::memset(static_cast<void *>(runsets_), 0, sizeof(RunSet) * NumSlots());
    root_->gran_off_off_ = go.load();
    root_->free_bits_off_ = fb.load();
    root_->brk_bits_off_ = bb.load();
    gran_off_ = reinterpret_cast<u64 *>(Ptr(root_->gran_off_off_));
    free_bits_ = reinterpret_cast<u64 *>(Ptr(root_->free_bits_off_));
    brk_bits_ = reinterpret_cast<u64 *>(Ptr(root_->brk_bits_off_));
    std::memset(free_bits_, 0, sizeof(u64) * words);
    std::memset(brk_bits_, 0, sizeof(u64) * words);
    u64 g = 0;
    for (const auto &x : lim.extents_) {
      for (u64 k = 0; k < x.second / kGranule; ++k, ++g) {
        gran_off_[g] = x.first + k * kGranule;
        free_bits_[g / 64] |= 1ull << (g % 64);
        if (g == 0 || gran_off_[g] != gran_off_[g - 1] + kGranule) {
          brk_bits_[g / 64] |= 1ull << (g % 64);
        }
      }
    }
    root_->granules_ = gcount;
    root_->free_granules_.store(gcount);
    root_->data_cursor_ = 0;
    root_->data_lock_.Init();
    root_->data_pid_ = lim.data_pid_;
    root_->data_pool_major_ = lim.data_pool_major_;
    root_->data_pool_minor_ = lim.data_pool_minor_;
    root_->data_mode_ = 1;
    if (!AttachData()) return false;
  }
  root_->magic_ = Root::kMagic;
  root_->version_ = Root::kVersion;
  root_->ready_.store(1, std::memory_order_release);
  return true;
}

inline bool View::Attach(const std::string &name, u64 root_off) {
  if (!backend_.shm_attach(name)) {
    return false;
  }
  mapped_ = true;
  alloc_ = backend_.AttachAlloc<WtcAlloc>();
  if (alloc_ == nullptr) {
    Detach();
    return false;
  }
  base_ = alloc_->GetBackendData();
  root_off_ = root_off;
  root_ = reinterpret_cast<Root *>(Ptr(root_off));
  if (root_->magic_ != Root::kMagic || root_->version_ != Root::kVersion ||
      root_->ready_.load(std::memory_order_acquire) != 1) {
    Detach();
    return false;
  }
  blobs_ = reinterpret_cast<BlobEntry *>(Ptr(root_->blobs_off_));
  tags_ = reinterpret_cast<TagEntry *>(Ptr(root_->tags_off_));
  log_ = reinterpret_cast<DirtyLog *>(Ptr(root_->log_off_));
  owners_ = reinterpret_cast<std::atomic<u32> *>(Ptr(root_->owners_off_));
  names_ = reinterpret_cast<NameSlot *>(Ptr(root_->names_off_));
  if (root_->data_mode_ != 0) {
    gran_off_ = reinterpret_cast<u64 *>(Ptr(root_->gran_off_off_));
    free_bits_ = reinterpret_cast<u64 *>(Ptr(root_->free_bits_off_));
    brk_bits_ = reinterpret_cast<u64 *>(Ptr(root_->brk_bits_off_));
    runsets_ = reinterpret_cast<RunSet *>(Ptr(root_->runsets_off_));
    gran_owner_ =
        reinterpret_cast<std::atomic<u32> *>(Ptr(root_->gran_owner_off_));
    if (!AttachData()) {
      Detach();  // no bytes reachable: do not use this cache at all
      return false;
    }
    PrefaultData();
  }
  return true;
}

inline void View::PrefaultData() {
  const char *env = std::getenv("CLIO_PREFAULT");
  if (env == nullptr || *env == '\0') {
    return;
  }
  u64 budget = std::string(env) == "0"
                   ? ~0ull
                   : static_cast<u64>(ctp::ConfigParse::ParseSize(env));
  // One BulkFault (MADV_POPULATE_WRITE: maps, never alters contents) per
  // stretch of granules that is contiguous in the device.
  const u64 n = root_->granules_;
  u64 g = 0;
  while (g < n && budget > 0) {
    u64 e = g + 1;
    while (e < n && !((brk_bits_[e / 64] >> (e % 64)) & 1ull)) ++e;
    const u64 bytes = std::min((e - g) * kGranule, budget);
    ctp::SystemInfo::BulkFault(GranPtr(static_cast<u32>(g)),
                               static_cast<size_t>(bytes));
    budget -= bytes;
    g = e;
  }
}

inline bool View::AttachData() {
  if (data_mapped_) return true;
  const clio::run::PoolId pool(root_->data_pool_major_, root_->data_pool_minor_);
  const std::string name = clio::run::bdev::MemBdevTransport::ShmSegmentName(
      root_->data_pid_, pool);
  if (!data_backend_.shm_attach(name)) return false;
  data_mapped_ = true;
  char *raw = reinterpret_cast<char *>(data_backend_.data_);
  auto *hdr =
      reinterpret_cast<clio::run::bdev::MemBdevTransport::ShmRamHeader *>(raw);
  if (raw == nullptr ||
      hdr->version_ != clio::run::bdev::MemBdevTransport::ShmRamHeader::kVersion ||
      hdr->ready_ != 1) {
    data_backend_.shm_detach();
    data_mapped_ = false;
    return false;
  }
  data_base_ = raw + hdr->data_off_;
  return true;
}

inline void View::Detach() {
  if (!mapped_) {
    return;
  }
  if (owner_) {
    if (root_ != nullptr) {
      root_->ready_.store(0, std::memory_order_release);
    }
    backend_.shm_destroy();
  } else {
    backend_.shm_detach();
  }
  if (data_mapped_) {
    data_backend_.shm_detach();  // the RAM bdev owns it: never destroyed here
    data_mapped_ = false;
  }
  data_base_ = nullptr;
  gran_off_ = nullptr;
  free_bits_ = nullptr;
  brk_bits_ = nullptr;
  runsets_ = nullptr;
  gran_owner_ = nullptr;
  mapped_ = false;
  owner_ = false;
  alloc_ = nullptr;
  base_ = nullptr;
  root_ = nullptr;
  blobs_ = nullptr;
  tags_ = nullptr;
  log_ = nullptr;
  owners_ = nullptr;
  names_ = nullptr;
}

inline bool View::ReleaseIfUnpinnedLocked(BlobEntry *e) {
  // Pins are only ever added under the entry's stripe lock (TryPut, TryGet,
  // ReadRange, PinRange), which the caller holds: zero here stays zero.
  if (e->pins_.load(std::memory_order_acquire) != 0) {
    return false;
  }
  ReleaseBufferLocked(e);
  return true;
}

inline void View::DropUnpendingLocked(BlobEntry *e) {
  if (ReleaseIfUnpinnedLocked(e)) {
    e->state_ = kSlotTomb;
    return;
  }
  // A reader still copies out of the buffer: the slot cannot be reused yet.
  // Keep it live but empty (nothing served, no size known); eviction
  // reclaims it once the pin drops.
  e->flags_ &= ~(kComplete | kSizeKnown);
  e->size_ = 0;
}

inline double View::Frecency(const BlobEntry *e, u64 now) const {
  u64 last = e->last_access_ns_.load(std::memory_order_relaxed);
  double age = now > last ? static_cast<double>(now - last) : 0.0;
  double hits = static_cast<double>(e->hits_.load(std::memory_order_relaxed));
  return hits * std::exp2(-age / static_cast<double>(root_->half_life_ns_));
}

inline long View::FindLocked(u64 h, u32 major, u32 minor, const char *name,
                             size_t len) {
  u32 sps = root_->slots_per_stripe_;
  u32 stripe = StripeOf(h);
  u32 start = static_cast<u32>((h / kStripes) % sps);
  u32 probes = sps < kMaxProbe ? sps : kMaxProbe;
  for (u32 i = 0; i < probes; ++i) {
    u32 slot = stripe * sps + (start + i) % sps;
    BlobEntry *e = Slot(slot);
    if (e->state_ == kSlotEmpty) {
      return -1;
    }
    if (e->state_ == kSlotLive && e->hash_ == h && e->tag_major_ == major &&
        e->tag_minor_ == minor && e->name_len_ == len &&
        std::memcmp(e->name_, name, len) == 0) {
      return static_cast<long>(slot);
    }
  }
  return -1;
}

inline long View::FindOrInsertLocked(u64 h, u32 major, u32 minor,
                                     const char *name, size_t len,
                                     bool *inserted) {
  *inserted = false;
  long found = FindLocked(h, major, minor, name, len);
  if (found >= 0) {
    return found;
  }
  u32 sps = root_->slots_per_stripe_;
  u32 stripe = StripeOf(h);
  u32 start = static_cast<u32>((h / kStripes) % sps);
  u32 probes = sps < kMaxProbe ? sps : kMaxProbe;
  for (u32 i = 0; i < probes; ++i) {
    u32 slot = stripe * sps + (start + i) % sps;
    BlobEntry *e = Slot(slot);
    if (e->state_ == kSlotLive) {
      continue;
    }
    e->state_ = kSlotLive;
    e->flags_ = 0;
    e->tag_major_ = major;
    e->tag_minor_ = minor;
    e->name_len_ = static_cast<u32>(len);
    e->put_flags_ = 0;
    e->hash_ = h;
    e->buf_off_ = 0;
    e->cap_ = 0;
    if (Extents()) RS(e)->n_ = 0;
    e->size_ = 0;
    e->valid_lo_ = e->valid_hi_ = 0;
    std::memcpy(e->name_, name, len);
    // Generations start from the clock so a dropped-and-reinserted key can
    // never match a fill snapshot taken against the old entry (ABA).
    u64 now = NowNs();
    e->write_gen_.store(now, std::memory_order_relaxed);
    e->inval_epoch_.store(now, std::memory_order_relaxed);
    e->hits_.store(0, std::memory_order_relaxed);
    e->last_access_ns_.store(now, std::memory_order_relaxed);
    *inserted = true;
    return static_cast<long>(slot);
  }
  return -1;
}

inline void View::ReleaseBufferLocked(BlobEntry *e) {
  if (Extents()) {
    FreeRuns(e);
  } else if (e->flags_ & kResident) {
    FreeBuffer(e->buf_off_);
  }
  e->flags_ &= ~(kResident | kComplete);
  e->buf_off_ = 0;
  e->cap_ = 0;
  e->valid_lo_ = e->valid_hi_ = 0;
}

inline bool View::ReclaimSlotLocked(u32 stripe) {
  u32 sps = root_->slots_per_stripe_;
  u64 now = NowNs();
  BlobEntry *best = nullptr;
  double best_f = 0.0;
  for (u32 i = 0; i < sps; ++i) {
    BlobEntry *e = Slot(stripe * sps + i);
    if (e->state_ != kSlotLive || e->pending_.load() != 0 ||
        e->pins_.load() != 0) {
      continue;
    }
    double f = Frecency(e, now);
    if (best == nullptr || f < best_f) {
      best = e;
      best_f = f;
    }
  }
  if (best == nullptr) {
    return false;
  }
  ReleaseBufferLocked(best);
  best->state_ = kSlotTomb;
  return true;
}

inline void View::FreeBuffer(u64 off) {
  if (off == 0) {
    return;
  }
  // Clear before freeing: once freed, another allocation may start in the
  // same granule and record itself, which this store must not erase.
  u64 g = off / kMinBuffer;
  if (g < root_->owner_count_) {
    owners_[g].store(0, std::memory_order_release);
  }
  alloc_->FreeOffset(ctp::ipc::OffsetPtr<>(off));
}

inline long View::PickVictim(u64 want_bytes) {
  u32 n = NumSlots();
  u32 want = root_->evict_sample_;
  u64 now = NowNs();
  const u32 scan = n < kEvictScanMax ? n : kEvictScanMax;
  // Advance by the whole window, so a stretch with nothing clean is not
  // re-walked by the next attempts.
  u64 cursor = root_->evict_cursor_.fetch_add(scan) % n;
  long best = -1;
  double best_f = 0.0;
  long fit = -1;  // best candidate whose buffer alone can hold want_bytes
  double fit_f = 0.0;
  u32 seen = 0;
  // Approximate frecency: sample clean resident entries from a rotating
  // cursor (reads here are unlocked hints, re-verified under the lock).
  // Bounded: with few clean entries an unbounded walk visits every slot of
  // a large table on each allocation attempt (holding a stripe lock).
  // When the sample holds no buffer that alone fits the request, keep
  // walking (same bound) for one: evicting it frees room at once, where a
  // small victim needs a window eviction and a merge, which can fail.
  for (u32 k = 0; k < scan && (seen < want || fit < 0); ++k) {
    u32 slot = static_cast<u32>((cursor + k) % n);
    BlobEntry *e = Slot(slot);
    if (e->state_ != kSlotLive || !(e->flags_ & kResident) ||
        e->pending_.load() != 0 || e->pins_.load() != 0) {
      continue;
    }
    double f = Frecency(e, now);
    if (seen < want) {
      ++seen;
      if (best < 0 || f < best_f) {
        best = static_cast<long>(slot);
        best_f = f;
      }
    }
    if (e->cap_ >= want_bytes && (fit < 0 || f < fit_f)) {
      fit = static_cast<long>(slot);
      fit_f = f;
    }
  }
  return fit >= 0 ? fit : best;
}

inline bool View::EvictSlot(u32 held_stripe, u32 slot, u64 granule) {
  u32 stripe = slot / root_->slots_per_stripe_;
  ctp::Mutex &m = root_->stripes_[stripe];
  bool own = stripe == held_stripe;
  if (!own && !m.TryLock(0)) {
    return false;  // contended: no progress
  }
  BlobEntry *e = Slot(slot);
  bool evicted = false;
  if (e->state_ == kSlotLive && (e->flags_ & kResident) &&
      e->pending_.load() == 0 && e->pins_.load() == 0 &&
      (granule == kAnyGranule || e->buf_off_ / kMinBuffer == granule)) {
    ReleaseBufferLocked(e);
    root_->stats_.evictions_.fetch_add(1, std::memory_order_relaxed);
    evicted = true;
  }
  if (!own) {
    m.Unlock();
  }
  return evicted;
}

inline u64 View::WindowSpan(u64 bytes) const {
  // Much wider than the request so one merge (a sort of every free page)
  // serves many later allocations: a phase of new large blobs then costs
  // one merge per kWindowBytes, not one per blob. Bounded so it evicts a
  // small share of the cache.
  const u64 seg = (root_->owner_count_ - 1) * kMinBuffer;
  u64 span = kWindowBytes;
  if (span > seg / 8) {
    span = seg / 8;
  }
  // Two requests' pages, each with its allocator header and a partial
  // granule at the window's edge: exactly 2 * bytes holds only one.
  const u64 two = 2 * (bytes + kMinBuffer);
  return span < two ? two : span;
}

inline bool View::EvictWindow(u32 held_stripe, u64 start, u64 span) {
  // Keep the window inside the heap: one running off its end can never
  // yield a page of the requested size.
  const u64 seg_end = (root_->owner_count_ - 1) * kMinBuffer;
  if (start + span > seg_end) {
    start = seg_end > span ? seg_end - span : 0;
  }
  u64 first = start / kMinBuffer;
  u64 last = (start + span) / kMinBuffer;
  if (last >= root_->owner_count_) {
    last = root_->owner_count_ - 1;
  }
  // Best effort: a dirty, pinned or contended buffer splits the window but
  // the runs on either side of it may still be long enough.
  bool evicted = false;
  for (u64 g = first; g <= last; ++g) {
    u32 o = owners_[g].load(std::memory_order_acquire);
    if (o != 0 && EvictSlot(held_stripe, o - 1, g)) {
      evicted = true;
    }
  }
  return evicted;
}

inline u64 View::AllocBuffer(u64 bytes, u32 held_stripe, u32 owner) {
  // Every attempt runs under the caller's stripe lock, so each is cheap
  // except the merge-and-retry, which sorts every free page under the
  // allocator lock: it runs only after a window eviction, at most
  // kMergesPerAlloc times per call, and backs off globally after several
  // consecutive failures.
  u64 off = 0;
  u32 merges = 0;
  for (u32 round = 0; round < kAllocRounds && off == 0; ++round) {
    auto o = alloc_->AllocateOffsetNoCoalesce(bytes);
    if (!o.IsNull()) {
      off = o.load();
      break;
    }
    long v = PickVictim(bytes);
    if (v < 0) {
      break;
    }
    BlobEntry *e = Slot(static_cast<u32>(v));
    if (e->cap_ >= bytes) {
      // Its page alone fits: the next attempt finds it on the free list.
      EvictSlot(held_stripe, static_cast<u32>(v), kAnyGranule);
      continue;
    }
    if (merges >= kMergesPerAlloc ||
        NowNs() < root_->coalesce_after_ns_.load()) {
      break;  // small victims only, and no merge allowed: write around
    }
    // Small victims never make room one by one (the allocator does not
    // merge on free): free a contiguous span around this one, then merge.
    if (!EvictWindow(held_stripe, e->buf_off_, WindowSpan(bytes))) {
      continue;
    }
    ++merges;
    o = alloc_->AllocateOffset(bytes);
    if (!o.IsNull()) {
      off = o.load();
      root_->merge_fails_.store(0);
    } else if (root_->merge_fails_.fetch_add(1) + 1 >= kMergeFailsForBackoff) {
      root_->merge_fails_.store(0);
      root_->coalesce_after_ns_.store(NowNs() + kCoalesceBackoffNs);
    }
  }
  if (off != 0 && off / kMinBuffer < root_->owner_count_) {
    owners_[off / kMinBuffer].store(owner + 1, std::memory_order_release);
  }
  return off;
}

inline Status View::EnsureCapacityLocked(BlobEntry *e, u64 need,
                                         u32 stripe) {
  if ((e->flags_ & kResident) && e->cap_ >= need) {
    return Status::kOk;
  }
  // Never wait for the pin here: the flusher's zero-copy pin is released
  // only after the blob's home acks, and the home acks only after an
  // Invalidate on this node takes this stripe lock (a distributed
  // deadlock). Refuse untouched; the caller retries without the lock.
  if (e->pins_.load(std::memory_order_acquire) != 0) {
    return Status::kBusy;
  }
  if (Extents()) {
    // Grow by appending runs: the bytes already held never move.
    if (!(e->flags_ & kResident)) {
      RS(e)->n_ = 0;
      e->cap_ = 0;
    }
    const u64 have = e->cap_;
    u64 want = need - have;
    if (have != 0 && want < have / 2) want = have / 2;  // amortize appends
    const u64 add = (want + kGranule - 1) / kGranule;
    e->pins_.fetch_add(1, std::memory_order_acq_rel);
    const bool ok = AllocRunsEvicting(e, add, stripe);
    e->pins_.fetch_sub(1, std::memory_order_acq_rel);
    if (!ok) {
      return Status::kNoSpace;
    }
    // Holes read as zeros, exactly what the chain returns for unwritten bytes.
    ZeroRange(e, have, e->cap_);
    e->flags_ |= kResident;
    return Status::kOk;
  }
  u64 cap = RoundBuffer(need < kMinBuffer ? kMinBuffer : need);
  if (e->flags_ & kResident) {
    u64 grown = RoundBuffer(e->cap_ + e->cap_ / 2);
    cap = grown > cap ? grown : cap;
  }
  // Pin ourselves so eviction (which may run in this stripe) skips us.
  e->pins_.fetch_add(1, std::memory_order_acq_rel);
  u64 off = AllocBuffer(cap, stripe, static_cast<u32>(e - blobs_));
  e->pins_.fetch_sub(1, std::memory_order_acq_rel);
  if (off == 0) {
    return Status::kNoSpace;
  }
  u64 keep = 0;
  if (e->flags_ & kResident) {
    keep = e->size_ < e->cap_ ? e->size_ : e->cap_;
    std::memcpy(Ptr(off), Ptr(e->buf_off_), keep);
    FreeBuffer(e->buf_off_);
  }
  // Holes read as zeros, exactly what the chain returns for unwritten bytes.
  std::memset(Ptr(off) + keep, 0, cap - keep);
  e->buf_off_ = off;
  e->cap_ = cap;
  e->flags_ |= kResident;
  return Status::kOk;
}

inline u64 View::NextFreeLocked(u64 from) const {
  const u64 n = root_->granules_;
  u64 step = 0;
  while (step < n) {
    const u64 g = (from + step) % n;
    const u64 bits = free_bits_[g / 64] >> (g % 64);
    if (bits == 0) {
      step += 64 - g % 64;  // rest of this word is used
      continue;
    }
    const u64 hit = g + static_cast<u64>(std::countr_zero(bits));  // MSVC too
    if (hit < n) return hit;
    step += n - g;  // past the end: wrap to granule 0
  }
  return n;
}

inline bool View::AllocRunsLocked(BlobEntry *e, u64 add) {
  const u64 n = root_->granules_;
  auto is_free = [this](u64 g) {
    return (free_bits_[g / 64] >> (g % 64)) & 1ull;
  };
  auto is_brk = [this](u64 g) { return (brk_bits_[g / 64] >> (g % 64)) & 1ull; };
  const u32 owner = static_cast<u32>(e - blobs_) + 1;
  auto mark = [this, owner](u64 g, u64 len, bool free) {
    for (u64 k = 0; k < len; ++k) {
      const u64 x = g + k;
      if (free) {
        free_bits_[x / 64] |= 1ull << (x % 64);
        gran_owner_[x].store(0, std::memory_order_relaxed);
      } else {
        free_bits_[x / 64] &= ~(1ull << (x % 64));
        gran_owner_[x].store(owner, std::memory_order_relaxed);
      }
    }
  };
  if (add == 0) return true;
  if (add > root_->free_granules_.load()) return false;
  // Saved so a failed call leaves the entry exactly as it was.
  const u32 runs0 = RS(e)->n_;
  const Run tail0 = runs0 > 0 ? RS(e)->r_[runs0 - 1] : Run{0, 0};
  std::pair<u64, u64> taken[kInlineRuns + 1];
  u32 ntaken = 0;
  u64 need = add;
  u64 cursor = root_->data_cursor_;
  u64 scanned = 0;
  while (need > 0 && scanned < n && ntaken <= kInlineRuns) {
    const u64 g = NextFreeLocked(cursor);
    if (g >= n) break;
    scanned += (g >= cursor ? g - cursor : n - cursor + g) + 1;
    u64 len = 1;
    while (len < need && g + len < n && is_free(g + len) && !is_brk(g + len)) {
      ++len;
    }
    // Extend the last run when this span continues it in the bdev.
    Run *tail = RS(e)->n_ > 0 ? &RS(e)->r_[RS(e)->n_ - 1] : nullptr;
    if (tail != nullptr && tail->g_ + tail->n_ == g && !is_brk(g)) {
      tail->n_ += static_cast<u32>(len);
    } else if (RS(e)->n_ < kInlineRuns) {
      RS(e)->r_[RS(e)->n_++] = Run{static_cast<u32>(g), static_cast<u32>(len)};
    } else {
      break;  // out of run slots
    }
    mark(g, len, false);
    taken[ntaken++] = {g, len};
    need -= len;
    cursor = (g + len) % n;
  }
  if (need > 0) {
    for (u32 i = 0; i < ntaken; ++i) mark(taken[i].first, taken[i].second, true);
    RS(e)->n_ = runs0;
    if (runs0 > 0) RS(e)->r_[runs0 - 1] = tail0;
    return false;
  }
  root_->data_cursor_ = cursor;
  root_->free_granules_.fetch_sub(add);
  e->cap_ += add * kGranule;
  return true;
}

inline bool View::AllocRunsEvicting(BlobEntry *e, u64 add, u32 stripe) {
  const u64 want = add * kGranule;
  for (u32 round = 0; round < 4 * kAllocRounds; ++round) {
    {
      Guard g(root_->data_lock_);
      if (AllocRunsLocked(e, add)) return true;
    }
    // The entry itself is pinned, so it is never its own victim.
    long v = PickVictim(want);
    if (v < 0) return false;  // nothing clean: the put writes around
    if (Slot(static_cast<u32>(v))->cap_ >= want) {
      // Its runs alone hold the request (at most kInlineRuns of them).
      EvictSlot(stripe, static_cast<u32>(v), kAnyGranule);
      continue;
    }
    EvictGranWindow(stripe, add);
  }
  return false;
}

inline void View::EvictGranWindow(u32 held_stripe, u64 span) {
  const u64 n = root_->granules_;
  if (span > n) span = n;
  u64 start = root_->evict_gcursor_.fetch_add(span) % n;
  if (start + span > n) start = n - span;
  u32 last = 0;  // owners are slot + 1; 0 = free
  for (u64 k = 0; k < span; ++k) {
    const u64 g = start + k;
    if (k > 0 && ((brk_bits_[g / 64] >> (g % 64)) & 1ull)) break;  // new block
    const u32 o = gran_owner_[g].load(std::memory_order_relaxed);
    if (o == 0 || o == last) continue;
    last = o;
    EvictSlot(held_stripe, o - 1, kAnyGranule);  // re-verified under its lock
  }
  Guard g(root_->data_lock_);
  root_->data_cursor_ = start;  // next fit starts at the freed window
}

inline void View::FreeRuns(BlobEntry *e) {
  if (RS(e)->n_ != 0) {
    Guard g(root_->data_lock_);
    u64 freed = 0;
    for (u32 r = 0; r < RS(e)->n_; ++r) {
      for (u64 k = 0; k < RS(e)->r_[r].n_; ++k) {
        const u64 gr = RS(e)->r_[r].g_ + k;
        free_bits_[gr / 64] |= 1ull << (gr % 64);
        gran_owner_[gr].store(0, std::memory_order_relaxed);
      }
      freed += RS(e)->r_[r].n_;
    }
    root_->free_granules_.fetch_add(freed);
  }
  RS(e)->n_ = 0;
  e->cap_ = 0;
}

inline void View::CopyIn(BlobEntry *e, u64 off, const char *src, u64 n) {
  if (!Extents()) {
    std::memcpy(Ptr(e->buf_off_) + off, src, n);
    return;
  }
  u64 pos = 0;
  for (u32 r = 0; r < RS(e)->n_ && n > 0; ++r) {
    const u64 rb = static_cast<u64>(RS(e)->r_[r].n_) * kGranule;
    if (off < pos + rb) {
      const u64 intra = off - pos;
      const u64 chunk = std::min(rb - intra, n);
      std::memcpy(GranPtr(RS(e)->r_[r].g_) + intra, src, chunk);
      src += chunk;
      off += chunk;
      n -= chunk;
    }
    pos += rb;
  }
}

inline void View::CopyOut(const BlobEntry *e, u64 off, char *dst, u64 n) const {
  if (!Extents()) {
    std::memcpy(dst, base_ + e->buf_off_ + off, n);
    return;
  }
  u64 pos = 0;
  for (u32 r = 0; r < RS(e)->n_ && n > 0; ++r) {
    const u64 rb = static_cast<u64>(RS(e)->r_[r].n_) * kGranule;
    if (off < pos + rb) {
      const u64 intra = off - pos;
      const u64 chunk = std::min(rb - intra, n);
      std::memcpy(dst, GranPtr(RS(e)->r_[r].g_) + intra, chunk);
      dst += chunk;
      off += chunk;
      n -= chunk;
    }
    pos += rb;
  }
}

inline void View::ZeroRange(BlobEntry *e, u64 from, u64 to) {
  if (to <= from) return;
  if (!Extents()) {
    std::memset(Ptr(e->buf_off_) + from, 0, to - from);
    return;
  }
  u64 pos = 0;
  for (u32 r = 0; r < RS(e)->n_ && from < to; ++r) {
    const u64 rb = static_cast<u64>(RS(e)->r_[r].n_) * kGranule;
    if (from < pos + rb) {
      const u64 intra = from - pos;
      const u64 chunk = std::min(rb - intra, to - from);
      std::memset(GranPtr(RS(e)->r_[r].g_) + intra, 0, chunk);
      from += chunk;
    }
    pos += rb;
  }
}

inline Status View::TryPut(u32 tag_major, u32 tag_minor, const char *name,
                           size_t name_len, u64 off, u64 size,
                           const char *src, u32 put_flags, bool may_block) {
  if (name_len > kMaxBlobName) {
    return Status::kTooLong;
  }
  if (MdOnly() || (put_flags & ~kPerBlobPutFlags) != 0) {
    root_->stats_.put_miss_.fetch_add(1, std::memory_order_relaxed);
    return Status::kMiss;  // not expressible here: take the task path
  }
  if (!may_block) {
    const u64 cap = log_->Capacity();
    const u64 room = kLogHeadroom < cap / 4 ? kLogHeadroom : cap / 4;
    if (log_->Size() + room >= cap) {
      root_->stats_.put_full_log_.fetch_add(1, std::memory_order_relaxed);
      return Status::kNoSpace;  // nothing touched; caller yields and retries
    }
  }
  u64 h = KeyHash(tag_major, tag_minor, name, name_len);
  u32 stripe = StripeOf(h);
  u32 slot = 0;
  BlobEntry *dst_e = nullptr;
  u64 gen = 0;
  {
    Guard g(root_->stripes_[stripe]);
    bool inserted = false;
    long s = FindOrInsertLocked(h, tag_major, tag_minor, name, name_len,
                                &inserted);
    if (s < 0 && ReclaimSlotLocked(stripe)) {
      s = FindOrInsertLocked(h, tag_major, tag_minor, name, name_len,
                             &inserted);
    }
    if (s < 0) {
      root_->stats_.put_full_slot_.fetch_add(1, std::memory_order_relaxed);
      return Status::kNoSpace;
    }
    slot = static_cast<u32>(s);
    BlobEntry *e = Slot(slot);
    if (inserted) {
      e->put_flags_ = put_flags;
    } else if (e->put_flags_ != put_flags) {
      if (e->pending_.load() != 0 || (e->flags_ & kResident)) {
        return Status::kMiss;  // let the chain judge the conflicting flags
      }
      e->put_flags_ = put_flags;  // metadata-only entry: adopt
    }
    u64 need = off + size;
    const Status cap_st = EnsureCapacityLocked(e, need, stripe);
    if (cap_st == Status::kBusy) {
      root_->stats_.put_busy_.fetch_add(1, std::memory_order_relaxed);
      return Status::kBusy;  // pinned: nothing touched, retry unlocked
    }
    if (cap_st != Status::kOk) {
      root_->stats_.put_full_buf_.fetch_add(1, std::memory_order_relaxed);
      return Status::kNoSpace;
    }
    // Lock order is always stripe -> tag; nothing takes them the other way.
    if (inserted && TagCreated(tag_major, tag_minor)) {
      e->flags_ |= kComplete;  // brand-new blob of a brand-new tag
    }
    if (need > e->size_) {
      e->size_ = need;
    }
    // This write's bytes are now the blob's: grow the valid range when they
    // touch it, else (disjoint) track the newest write.
    if (e->valid_hi_ > e->valid_lo_ && off <= e->valid_hi_ &&
        need >= e->valid_lo_) {
      e->valid_lo_ = std::min(e->valid_lo_, off);
      e->valid_hi_ = std::max(e->valid_hi_, need);
    } else {
      e->valid_lo_ = off;
      e->valid_hi_ = need;
    }
    e->pins_.fetch_add(1, std::memory_order_acq_rel);
    e->pending_.fetch_add(1, std::memory_order_acq_rel);
    gen = e->write_gen_.fetch_add(1, std::memory_order_acq_rel) + 1;
    e->wbegin_.fetch_add(1, std::memory_order_acq_rel);
    e->hits_.fetch_add(1, std::memory_order_relaxed);
    e->last_access_ns_.store(NowNs(), std::memory_order_relaxed);
    dst_e = e;
  }
  CopyIn(dst_e, off, src, size);
  Slot(slot)->wend_.fetch_add(1, std::memory_order_release);
  // Unpin BEFORE the (possibly blocking) push: a full log waits on the
  // flusher, which must be free to take this stripe and read the entry.
  Slot(slot)->pins_.fetch_sub(1, std::memory_order_acq_rel);
  DirtyRange r{slot, static_cast<u32>(gen), off, size};
  // Count the range BEFORE it enters the log. A barrier's target
  // (PushedNow) then covers every range already in the log or about to be:
  // a writer counted after its push could leave pushed_ short of an earlier
  // writer's slot, and a later SyncTag would stop before its own range.
  root_->pushed_.fetch_add(1, std::memory_order_acq_rel);
  NoteLocalWrite();
  log_->Push(r);
  root_->stats_.puts_.fetch_add(1, std::memory_order_relaxed);
  root_->stats_.put_bytes_.fetch_add(size, std::memory_order_relaxed);
  return Status::kOk;
}

inline Status View::TryGet(u32 tag_major, u32 tag_minor, const char *name,
                           size_t name_len, u64 off, u64 size, char *dst) {
  if (MdOnly()) {
    return Status::kMiss;  // no bytes cached in metadata-only mode
  }
  if (name_len > kMaxBlobName || dst == nullptr) {
    return Status::kMiss;
  }
  u64 h = KeyHash(tag_major, tag_minor, name, name_len);
  u32 stripe = StripeOf(h);
  std::atomic<u64> *why = &root_->stats_.miss_busy_;
  for (u32 attempt = 0; attempt < kTornRetries; ++attempt) {
    const BlobEntry *src = nullptr;
    BlobEntry *e = nullptr;
    u64 seq = 0;
    bool busy = false;
    {
      Guard g(root_->stripes_[stripe]);
      long s = FindLocked(h, tag_major, tag_minor, name, name_len);
      why = &root_->stats_.miss_absent_;
      if (s >= 0) {
        e = Slot(static_cast<u32>(s));
        why = !(e->flags_ & kResident) ? &root_->stats_.miss_nonres_
                                       : &root_->stats_.miss_range_;
        const bool whole = (e->flags_ & kComplete) != 0 && off + size <= e->size_;
        const bool written = off >= e->valid_lo_ && off + size <= e->valid_hi_;
        if ((e->flags_ & kResident) != 0 && (whole || written)) {
          // wbegin_ only moves under this lock, so seq is stable here.
          seq = e->wbegin_.load(std::memory_order_acquire);
          busy = e->wend_.load(std::memory_order_acquire) != seq;
          if (!busy) {
            e->pins_.fetch_add(1, std::memory_order_acq_rel);
            e->hits_.fetch_add(1, std::memory_order_relaxed);
            e->last_access_ns_.store(NowNs(), std::memory_order_relaxed);
            src = e;
          }
        }
      }
    }
    if (busy) {
      why = &root_->stats_.miss_busy_;
      ctp::CpuRelax();
      continue;  // a writer is mid-copy into this buffer
    }
    if (src == nullptr) {
      break;
    }
    CopyOut(src, off, dst, size);
    std::atomic_thread_fence(std::memory_order_acquire);
    const bool torn = e->wbegin_.load(std::memory_order_relaxed) != seq;
    e->pins_.fetch_sub(1, std::memory_order_acq_rel);
    if (!torn) {
      root_->stats_.get_hits_.fetch_add(1, std::memory_order_relaxed);
      return Status::kOk;
    }
  }
  root_->stats_.get_misses_.fetch_add(1, std::memory_order_relaxed);
  why->fetch_add(1, std::memory_order_relaxed);
  return Status::kMiss;
}

inline Status View::TryGetSize(u32 tag_major, u32 tag_minor, const char *name,
                               size_t name_len, u64 *out_size) {
  if (name_len > kMaxBlobName) {
    return Status::kMiss;
  }
  u64 h = KeyHash(tag_major, tag_minor, name, name_len);
  Guard g(root_->stripes_[StripeOf(h)]);
  long s = FindLocked(h, tag_major, tag_minor, name, name_len);
  if (s < 0) {
    return Status::kMiss;
  }
  BlobEntry *e = Slot(static_cast<u32>(s));
  if (!(e->flags_ & (kComplete | kSizeKnown))) {
    return Status::kMiss;
  }
  *out_size = e->size_;
  return Status::kOk;
}

inline u32 View::BlobPending(u32 tag_major, u32 tag_minor, const char *name,
                             size_t name_len) {
  if (name_len > kMaxBlobName) {
    return 0;
  }
  u64 h = KeyHash(tag_major, tag_minor, name, name_len);
  Guard g(root_->stripes_[StripeOf(h)]);
  long s = FindLocked(h, tag_major, tag_minor, name, name_len);
  return s < 0 ? 0 : Slot(static_cast<u32>(s))->pending_.load();
}

inline TagEntry *View::FindTagLocked(u32 major, u32 minor, bool insert) {
  u32 n = root_->max_tags_;
  u64 h = KeyHash(major, minor, "", 0);
  TagEntry *free_slot = nullptr;
  for (u32 i = 0; i < n && i < kMaxProbe; ++i) {
    TagEntry *t = tags_ + (h + i) % n;
    if (t->state_ == kSlotLive && t->major_ == major && t->minor_ == minor) {
      return t;
    }
    if (t->state_ != kSlotLive && free_slot == nullptr) {
      free_slot = t;
    }
    if (t->state_ == kSlotEmpty) {
      break;
    }
  }
  if (!insert || free_slot == nullptr) {
    return nullptr;
  }
  free_slot->state_ = kSlotLive;
  free_slot->flags_ = 0;
  free_slot->major_ = major;
  free_slot->minor_ = minor;
  free_slot->sticky_rc_.store(0, std::memory_order_relaxed);
  free_slot->name_len_ = 0;
  free_slot->name_hash_ = 0;
  free_slot->attr_write_seq_ = 0;
  return free_slot;
}

inline void View::NoteTagCreated(u32 major, u32 minor) {
  Guard g(root_->tag_lock_);
  TagEntry *t = FindTagLocked(major, minor, true);
  if (t != nullptr) {
    t->flags_ |= kTagCreated;
  }
}

inline bool View::TagCreated(u32 major, u32 minor) {
  Guard g(root_->tag_lock_);
  TagEntry *t = FindTagLocked(major, minor, false);
  return t != nullptr && (t->flags_ & kTagCreated);
}

inline void View::SetTagError(u32 major, u32 minor, u32 rc) {
  Guard g(root_->tag_lock_);
  TagEntry *t = FindTagLocked(major, minor, true);
  std::atomic<u32> &slot = t != nullptr ? t->sticky_rc_ : root_->global_rc_;
  u32 expected = 0;
  slot.compare_exchange_strong(expected, rc);
}

inline u32 View::TakeTagError(u32 major, u32 minor) {
  Guard g(root_->tag_lock_);
  TagEntry *t = FindTagLocked(major, minor, false);
  u32 rc = t != nullptr ? t->sticky_rc_.exchange(0) : 0;
  if (rc == 0) {
    rc = root_->global_rc_.exchange(0);
  }
  return rc;
}

inline void View::DropTag(u32 major, u32 minor) {
  if (root_ == nullptr) return;  // no segment: nothing cached
  Guard g(root_->tag_lock_);
  root_->tag_inval_seq_.fetch_add(1, std::memory_order_acq_rel);
  TagEntry *t = FindTagLocked(major, minor, false);
  if (t != nullptr) {
    UnindexNameLocked(t);
    t->state_ = kSlotTomb;
    t->flags_ = 0;
  }
}

inline NameSlot *View::FindNameLocked(u64 h, const char *name, size_t len,
                                      bool insert) {
  const u32 n = root_->max_tags_ * 2;
  NameSlot *free_slot = nullptr;
  for (u32 i = 0; i < n && i < kMaxProbe; ++i) {
    NameSlot *ns = names_ + (h + i) % n;
    if (ns->state_ == kSlotLive && ns->hash_ == h) {
      // Verify against the tag's stored name (hashes can collide).
      TagEntry *t = FindTagLocked(ns->major_, ns->minor_, false);
      if (t != nullptr && (t->flags_ & kTagNameValid) && t->name_len_ == len &&
          std::memcmp(t->name_, name, len) == 0) {
        return ns;
      }
    }
    if (ns->state_ != kSlotLive && free_slot == nullptr) {
      free_slot = ns;
    }
    if (ns->state_ == kSlotEmpty) {
      break;
    }
  }
  if (!insert || free_slot == nullptr) {
    return nullptr;
  }
  free_slot->state_ = kSlotLive;
  free_slot->hash_ = h;
  return free_slot;
}

inline void View::UnindexNameLocked(TagEntry *t) {
  if (!(t->flags_ & kTagNameValid)) {
    return;
  }
  const u32 n = root_->max_tags_ * 2;
  for (u32 i = 0; i < n && i < kMaxProbe; ++i) {
    NameSlot *ns = names_ + (t->name_hash_ + i) % n;
    if (ns->state_ == kSlotLive && ns->major_ == t->major_ &&
        ns->minor_ == t->minor_) {
      ns->state_ = kSlotTomb;
      break;
    }
    if (ns->state_ == kSlotEmpty) {
      break;
    }
  }
  t->flags_ &= ~kTagNameValid;
  t->name_len_ = 0;
}

inline bool View::FillTagName(u32 major, u32 minor, const char *name,
                              size_t len, u64 seq) {
  if (len == 0 || len > kMaxTagName) {
    return false;
  }
  Guard g(root_->tag_lock_);
  if (root_->tag_inval_seq_.load(std::memory_order_acquire) != seq) {
    return false;  // a drop raced the fetch: the name may be stale
  }
  const u64 h = KeyHash(0, 0, name, len);
  // A different tag cached under this name lost it (renamed away).
  NameSlot *old = FindNameLocked(h, name, len, false);
  if (old != nullptr && (old->major_ != major || old->minor_ != minor)) {
    TagEntry *ot = FindTagLocked(old->major_, old->minor_, false);
    if (ot != nullptr) UnindexNameLocked(ot);
  }
  TagEntry *t = FindTagLocked(major, minor, true);
  if (t == nullptr) {
    return false;
  }
  UnindexNameLocked(t);
  std::memcpy(t->name_, name, len);
  t->name_[len] = '\0';
  t->name_len_ = static_cast<u32>(len);
  t->name_hash_ = h;
  t->flags_ |= kTagNameValid;
  NameSlot *ns = FindNameLocked(h, name, len, true);
  if (ns == nullptr) {
    UnindexNameLocked(t);  // index full: keep the table consistent
    return false;
  }
  ns->major_ = major;
  ns->minor_ = minor;
  return true;
}

inline bool View::LookupTagByName(const char *name, size_t len, u32 *major,
                                  u32 *minor) {
  if (len == 0 || len > kMaxTagName) {
    return false;
  }
  Guard g(root_->tag_lock_);
  NameSlot *ns = FindNameLocked(KeyHash(0, 0, name, len), name, len, false);
  if (ns == nullptr) {
    return false;
  }
  *major = ns->major_;
  *minor = ns->minor_;
  root_->stats_.tag_hits_.fetch_add(1, std::memory_order_relaxed);
  return true;
}

inline bool View::LookupTagName(u32 major, u32 minor, std::string *name) {
  Guard g(root_->tag_lock_);
  TagEntry *t = FindTagLocked(major, minor, false);
  if (t == nullptr || !(t->flags_ & kTagNameValid)) {
    return false;
  }
  name->assign(t->name_, t->name_len_);
  root_->stats_.tag_hits_.fetch_add(1, std::memory_order_relaxed);
  return true;
}

inline void View::DropTagMeta(u32 major, u32 minor) {
  if (root_ == nullptr) return;  // no segment: nothing cached
  Guard g(root_->tag_lock_);
  root_->tag_inval_seq_.fetch_add(1, std::memory_order_acq_rel);
  TagEntry *t = FindTagLocked(major, minor, false);
  if (t != nullptr) {
    UnindexNameLocked(t);
    t->flags_ &= ~kTagAttrValid;
  }
}

inline bool View::FillTagAttrs(u32 major, u32 minor, const TagAttrs &a,
                               u64 seq, u64 write_seq) {
  Guard g(root_->tag_lock_);
  if (root_->tag_inval_seq_.load(std::memory_order_acquire) != seq ||
      root_->local_write_seq_.load(std::memory_order_acquire) != write_seq) {
    return false;  // a drop or a local write raced the fetch
  }
  TagEntry *t = FindTagLocked(major, minor, true);
  if (t == nullptr) {
    return false;
  }
  t->size_ = a.size_;
  t->ctime_ = a.ctime_;
  t->mtime_ = a.mtime_;
  t->atime_ = a.atime_;
  t->attr_write_seq_ = write_seq;
  t->flags_ |= kTagAttrValid;
  root_->stats_.md_fills_.fetch_add(1, std::memory_order_relaxed);
  return true;
}

inline bool View::LookupTagAttrs(u32 major, u32 minor, TagAttrs *a) {
  Guard g(root_->tag_lock_);
  TagEntry *t = FindTagLocked(major, minor, false);
  if (t == nullptr || !(t->flags_ & kTagAttrValid)) {
    return false;
  }
  if (t->attr_write_seq_ !=
      root_->local_write_seq_.load(std::memory_order_acquire)) {
    t->flags_ &= ~kTagAttrValid;  // a local write since the fill
    return false;
  }
  a->size_ = t->size_;
  a->ctime_ = t->ctime_;
  a->mtime_ = t->mtime_;
  a->atime_ = t->atime_;
  root_->stats_.tag_hits_.fetch_add(1, std::memory_order_relaxed);
  return true;
}

inline bool View::SnapshotGens(u32 tag_major, u32 tag_minor, const char *name,
                               size_t name_len, u64 *write_gen,
                               u64 *inval_epoch) {
  if (name_len > kMaxBlobName) {
    return false;
  }
  u64 h = KeyHash(tag_major, tag_minor, name, name_len);
  u32 stripe = StripeOf(h);
  Guard g(root_->stripes_[stripe]);
  bool inserted = false;
  long s = FindOrInsertLocked(h, tag_major, tag_minor, name, name_len,
                              &inserted);
  if (s < 0 && ReclaimSlotLocked(stripe)) {
    s = FindOrInsertLocked(h, tag_major, tag_minor, name, name_len, &inserted);
  }
  if (s < 0) {
    return false;
  }
  BlobEntry *e = Slot(static_cast<u32>(s));
  *write_gen = e->write_gen_.load();
  *inval_epoch = e->inval_epoch_.load();
  return true;
}

inline bool View::Fill(u32 tag_major, u32 tag_minor, const char *name,
                       size_t name_len, const char *data, u64 size,
                       u64 write_gen, u64 inval_epoch) {
  if (name_len > kMaxBlobName || size > root_->max_fill_) {
    return false;
  }
  u64 h = KeyHash(tag_major, tag_minor, name, name_len);
  u32 stripe = StripeOf(h);
  Guard g(root_->stripes_[stripe]);
  long s = FindLocked(h, tag_major, tag_minor, name, name_len);
  if (s < 0) {
    return false;
  }
  BlobEntry *e = Slot(static_cast<u32>(s));
  if (e->write_gen_.load() != write_gen ||
      e->inval_epoch_.load() != inval_epoch || e->pending_.load() != 0) {
    return false;  // a local write or an invalidation raced the fetch
  }
  if (e->pins_.load(std::memory_order_acquire) != 0) {
    return false;  // a reader is mid-copy: skip this optional fill, never
                   // wait for a pin under the stripe lock
  }
  e->flags_ &= ~kComplete;
  e->valid_lo_ = e->valid_hi_ = 0;
  e->size_ = 0;
  if (EnsureCapacityLocked(e, size == 0 ? 1 : size, stripe) != Status::kOk) {
    return false;
  }
  CopyIn(e, 0, data, size);
  ZeroRange(e, size, e->cap_);
  e->size_ = size;
  e->flags_ |= kComplete;
  e->valid_lo_ = 0;
  e->valid_hi_ = size;
  e->hits_.fetch_add(1, std::memory_order_relaxed);
  e->last_access_ns_.store(NowNs(), std::memory_order_relaxed);
  root_->stats_.fills_.fetch_add(1, std::memory_order_relaxed);
  return true;
}

inline bool View::FillSize(u32 tag_major, u32 tag_minor, const char *name,
                           size_t name_len, u64 size, u64 write_gen,
                           u64 inval_epoch) {
  if (name_len > kMaxBlobName) {
    return false;
  }
  u64 h = KeyHash(tag_major, tag_minor, name, name_len);
  Guard g(root_->stripes_[StripeOf(h)]);
  long s = FindLocked(h, tag_major, tag_minor, name, name_len);
  if (s < 0) {
    return false;
  }
  BlobEntry *e = Slot(static_cast<u32>(s));
  if (e->write_gen_.load() != write_gen ||
      e->inval_epoch_.load() != inval_epoch || e->pending_.load() != 0) {
    return false;  // a local write or an invalidation raced the fetch
  }
  if (!(e->flags_ & kComplete)) {
    e->size_ = size;
  }
  e->flags_ |= kSizeKnown;
  root_->stats_.md_fills_.fetch_add(1, std::memory_order_relaxed);
  return true;
}

inline void View::Invalidate(u32 tag_major, u32 tag_minor, const char *name,
                             size_t name_len) {
  if (root_ == nullptr) return;  // no segment: nothing cached
  if (name_len == kTagMetaNameLen &&
      std::memcmp(name, kTagMetaName, kTagMetaNameLen) == 0) {
    DropTagMeta(tag_major, tag_minor);
    return;
  }
  if (name_len > kMaxBlobName) {
    return;
  }
  u64 h = KeyHash(tag_major, tag_minor, name, name_len);
  Guard g(root_->stripes_[StripeOf(h)]);
  long s = FindLocked(h, tag_major, tag_minor, name, name_len);
  if (s < 0) {
    return;
  }
  BlobEntry *e = Slot(static_cast<u32>(s));
  e->flags_ &= ~(kComplete | kSizeKnown);
  e->valid_lo_ = e->valid_hi_ = 0;  // pending bytes still flush, never serve
  e->inval_epoch_.fetch_add(1, std::memory_order_acq_rel);
  if (e->pending_.load() == 0) {
    // A pinned buffer is left to eviction: it is unservable already, and
    // waiting for the pin under this lock is what deadlocks.
    ReleaseIfUnpinnedLocked(e);
  }
  root_->stats_.invalidations_.fetch_add(1, std::memory_order_relaxed);
}

inline void View::DropBlob(u32 tag_major, u32 tag_minor, const char *name,
                           size_t name_len) {
  if (root_ == nullptr) return;  // no segment: nothing cached
  if (name_len > kMaxBlobName) {
    return;
  }
  u64 h = KeyHash(tag_major, tag_minor, name, name_len);
  Guard g(root_->stripes_[StripeOf(h)]);
  long s = FindLocked(h, tag_major, tag_minor, name, name_len);
  if (s < 0) {
    return;
  }
  BlobEntry *e = Slot(static_cast<u32>(s));
  e->inval_epoch_.fetch_add(1, std::memory_order_acq_rel);
  e->flags_ &= ~kComplete;
  e->valid_lo_ = e->valid_hi_ = 0;
  if (e->pending_.load() != 0) {
    return;  // a racing write logged after the caller's barrier keeps it
  }
  DropUnpendingLocked(e);
}

inline void View::DropTagBlobs(u32 major, u32 minor) {
  if (root_ == nullptr) return;  // no segment: nothing cached
  u32 sps = root_->slots_per_stripe_;
  for (u32 stripe = 0; stripe < kStripes; ++stripe) {
    Guard g(root_->stripes_[stripe]);
    for (u32 i = 0; i < sps; ++i) {
      BlobEntry *e = Slot(stripe * sps + i);
      if (e->state_ != kSlotLive || e->tag_major_ != major ||
          e->tag_minor_ != minor) {
        continue;
      }
      e->inval_epoch_.fetch_add(1, std::memory_order_acq_rel);
      e->flags_ &= ~kComplete;
      e->valid_lo_ = e->valid_hi_ = 0;
      if (e->pending_.load() != 0) {
        continue;
      }
      DropUnpendingLocked(e);
    }
  }
}

inline bool View::ReadRange(u32 slot, u64 off, u64 *size, char *dst,
                            u32 *tag_major, u32 *tag_minor,
                            std::string *name, u32 *put_flags,
                            u64 *copied_gen) {
  if (slot >= NumSlots()) {
    return false;
  }
  u32 stripe = slot / root_->slots_per_stripe_;
  BlobEntry *e = Slot(slot);
  {
    Guard g(root_->stripes_[stripe]);
    if (e->state_ != kSlotLive || !(e->flags_ & kResident)) {
      return false;
    }
    u64 end = off + *size;
    if (end > e->cap_) {
      end = e->cap_;
    }
    *size = end > off ? end - off : 0;
    *tag_major = e->tag_major_;
    *tag_minor = e->tag_minor_;
    *put_flags = e->put_flags_;
    name->assign(e->name_, e->name_len_);
    if (copied_gen != nullptr) {
      // Generations are assigned under this lock, before the data copy;
      // wend_ catching up with wbegin_ means every assigned one finished.
      bool quiet = e->wbegin_.load(std::memory_order_acquire) ==
                   e->wend_.load(std::memory_order_acquire);
      *copied_gen = quiet ? e->write_gen_.load(std::memory_order_acquire) : 0;
    }
    e->pins_.fetch_add(1, std::memory_order_acq_rel);
  }
  if (*size > 0) {
    CopyOut(e, off, dst, *size);
  }
  e->pins_.fetch_sub(1, std::memory_order_acq_rel);
  return true;
}

inline const char *View::PinRange(u32 slot, u64 off, u64 *size,
                                  u64 *copied_gen) {
  if (Extents() || slot >= NumSlots()) {
    return nullptr;
  }
  u32 stripe = slot / root_->slots_per_stripe_;
  BlobEntry *e = Slot(slot);
  Guard g(root_->stripes_[stripe]);
  if (e->state_ != kSlotLive || !(e->flags_ & kResident)) {
    return nullptr;
  }
  u64 end = off + *size;
  if (end > e->cap_) {
    end = e->cap_;
  }
  *size = end > off ? end - off : 0;
  bool quiet = e->wbegin_.load(std::memory_order_acquire) ==
               e->wend_.load(std::memory_order_acquire);
  *copied_gen = quiet ? e->write_gen_.load(std::memory_order_acquire) : 0;
  e->pins_.fetch_add(1, std::memory_order_acq_rel);
  return base_ + e->buf_off_ + off;
}

inline void View::UnpinRange(u32 slot) {
  Slot(slot)->pins_.fetch_sub(1, std::memory_order_acq_rel);
}

inline u64 View::RangeGen(const DirtyRange &r) const {
  if (r.slot_ >= NumSlots()) {
    return 0;
  }
  u64 cur = Slot(r.slot_)->write_gen_.load(std::memory_order_acquire);
  // The range's write happened at or before `cur`, within 2^32 writes.
  return cur - static_cast<u32>(static_cast<u32>(cur) - r.gen_);
}

inline void View::CompleteRanges(u32 slot, u32 n) {
  if (slot < NumSlots()) {
    Slot(slot)->pending_.fetch_sub(n, std::memory_order_acq_rel);
  }
}

}  // namespace clio::cte::core::wtc

#endif  // CLIO_CTE_CORE_SHM_WTC_IMPL_H_
