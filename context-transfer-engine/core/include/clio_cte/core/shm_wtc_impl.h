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

#include <cmath>

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

  size_t blob_bytes = sizeof(BlobEntry) * NumSlots();
  auto b = alloc_->AllocateOffset(blob_bytes);
  size_t tag_bytes = sizeof(TagEntry) * root_->max_tags_;
  auto t = alloc_->AllocateOffset(tag_bytes);
  auto l = alloc_->AllocateOffset(sizeof(DirtyLog));
  if (b.IsNull() || t.IsNull() || l.IsNull()) {
    return false;
  }
  root_->blobs_off_ = b.load();
  root_->tags_off_ = t.load();
  root_->log_off_ = l.load();
  blobs_ = reinterpret_cast<BlobEntry *>(Ptr(root_->blobs_off_));
  tags_ = reinterpret_cast<TagEntry *>(Ptr(root_->tags_off_));
  std::memset(static_cast<void *>(blobs_), 0, blob_bytes);
  std::memset(static_cast<void *>(tags_), 0, tag_bytes);
  log_ = new (Ptr(root_->log_off_)) DirtyLog(alloc_, lim.log_depth_);
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
  mapped_ = false;
  owner_ = false;
  alloc_ = nullptr;
  base_ = nullptr;
  root_ = nullptr;
  blobs_ = nullptr;
  tags_ = nullptr;
  log_ = nullptr;
}

inline void View::WaitUnpinned(BlobEntry *e) {
  u32 spins = 0;
  while (e->pins_.load(std::memory_order_acquire) != 0) {
    if (++spins < 64) {
      ctp::CpuRelax();
    } else {
      std::this_thread::yield();
    }
  }
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
    e->size_ = 0;
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
  if (e->flags_ & kResident) {
    FreeBuffer(e->buf_off_);
  }
  e->flags_ &= ~(kResident | kComplete);
  e->buf_off_ = 0;
  e->cap_ = 0;
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
  if (off != 0) {
    alloc_->FreeOffset(ctp::ipc::OffsetPtr<>(off));
  }
}

inline bool View::EvictOne(u32 held_stripe) {
  u32 n = NumSlots();
  u32 sps = root_->slots_per_stripe_;
  u32 want = root_->evict_sample_;
  u64 now = NowNs();
  u64 cursor = root_->evict_cursor_.fetch_add(want * 4) % n;
  long best = -1;
  double best_f = 0.0;
  u32 seen = 0;
  // Approximate frecency: sample clean resident entries from a rotating
  // cursor (reads here are unlocked hints, re-verified under the lock).
  for (u32 k = 0; k < n && seen < want; ++k) {
    u32 slot = static_cast<u32>((cursor + k) % n);
    BlobEntry *e = Slot(slot);
    if (e->state_ != kSlotLive || !(e->flags_ & kResident) ||
        e->pending_.load() != 0 || e->pins_.load() != 0) {
      continue;
    }
    ++seen;
    double f = Frecency(e, now);
    if (best < 0 || f < best_f) {
      best = static_cast<long>(slot);
      best_f = f;
    }
  }
  if (best < 0) {
    return false;
  }
  u32 stripe = static_cast<u32>(best) / sps;
  ctp::Mutex &m = root_->stripes_[stripe];
  bool own = stripe == held_stripe;
  if (!own && !m.TryLock(0)) {
    return true;  // contended: report progress so the caller re-samples
  }
  BlobEntry *e = Slot(static_cast<u32>(best));
  bool evicted = false;
  if (e->state_ == kSlotLive && (e->flags_ & kResident) &&
      e->pending_.load() == 0 && e->pins_.load() == 0) {
    ReleaseBufferLocked(e);
    root_->stats_.evictions_.fetch_add(1, std::memory_order_relaxed);
    evicted = true;
  }
  if (!own) {
    m.Unlock();
  }
  (void)evicted;
  return true;
}

inline u64 View::AllocBuffer(u64 bytes, u32 held_stripe) {
  for (u32 attempt = 0; attempt < 256; ++attempt) {
    auto o = alloc_->AllocateOffset(bytes);
    if (!o.IsNull()) {
      return o.load();
    }
    if (!EvictOne(held_stripe)) {
      return 0;
    }
  }
  return 0;
}

inline bool View::EnsureCapacityLocked(BlobEntry *e, u64 need, u32 stripe) {
  if ((e->flags_ & kResident) && e->cap_ >= need) {
    return true;
  }
  WaitUnpinned(e);
  u64 cap = RoundBuffer(need < kMinBuffer ? kMinBuffer : need);
  if (e->flags_ & kResident) {
    u64 grown = RoundBuffer(e->cap_ + e->cap_ / 2);
    cap = grown > cap ? grown : cap;
  }
  // Pin ourselves so eviction (which may run in this stripe) skips us.
  e->pins_.fetch_add(1, std::memory_order_acq_rel);
  u64 off = AllocBuffer(cap, stripe);
  e->pins_.fetch_sub(1, std::memory_order_acq_rel);
  if (off == 0) {
    return false;
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
  return true;
}

inline Status View::TryPut(u32 tag_major, u32 tag_minor, const char *name,
                           size_t name_len, u64 off, u64 size,
                           const char *src, u32 put_flags) {
  if (name_len > kMaxBlobName) {
    return Status::kTooLong;
  }
  u64 h = KeyHash(tag_major, tag_minor, name, name_len);
  u32 stripe = StripeOf(h);
  u32 slot = 0;
  char *dst = nullptr;
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
    if (!EnsureCapacityLocked(e, need, stripe)) {
      return Status::kNoSpace;
    }
    // Lock order is always stripe -> tag; nothing takes them the other way.
    if (inserted && TagCreated(tag_major, tag_minor)) {
      e->flags_ |= kComplete;  // brand-new blob of a brand-new tag
    }
    if (need > e->size_) {
      e->size_ = need;
    }
    e->pins_.fetch_add(1, std::memory_order_acq_rel);
    e->pending_.fetch_add(1, std::memory_order_acq_rel);
    e->write_gen_.fetch_add(1, std::memory_order_acq_rel);
    e->hits_.fetch_add(1, std::memory_order_relaxed);
    e->last_access_ns_.store(NowNs(), std::memory_order_relaxed);
    dst = Ptr(e->buf_off_) + off;
  }
  std::memcpy(dst, src, size);
  // Unpin BEFORE the (possibly blocking) push: a full log waits on the
  // flusher, which must be free to take this stripe and read the entry.
  Slot(slot)->pins_.fetch_sub(1, std::memory_order_acq_rel);
  DirtyRange r{slot, 0, off, size};
  log_->Push(r);
  root_->pushed_.fetch_add(1, std::memory_order_acq_rel);
  root_->stats_.puts_.fetch_add(1, std::memory_order_relaxed);
  root_->stats_.put_bytes_.fetch_add(size, std::memory_order_relaxed);
  return Status::kOk;
}

inline Status View::TryGet(u32 tag_major, u32 tag_minor, const char *name,
                           size_t name_len, u64 off, u64 size, char *dst) {
  if (name_len > kMaxBlobName || dst == nullptr) {
    return Status::kMiss;
  }
  u64 h = KeyHash(tag_major, tag_minor, name, name_len);
  u32 stripe = StripeOf(h);
  const char *src = nullptr;
  BlobEntry *e = nullptr;
  {
    Guard g(root_->stripes_[stripe]);
    long s = FindLocked(h, tag_major, tag_minor, name, name_len);
    if (s >= 0) {
      e = Slot(static_cast<u32>(s));
      if ((e->flags_ & (kResident | kComplete)) == (kResident | kComplete) &&
          off + size <= e->size_) {
        e->pins_.fetch_add(1, std::memory_order_acq_rel);
        e->hits_.fetch_add(1, std::memory_order_relaxed);
        e->last_access_ns_.store(NowNs(), std::memory_order_relaxed);
        src = Ptr(e->buf_off_) + off;
      }
    }
  }
  if (src == nullptr) {
    root_->stats_.get_misses_.fetch_add(1, std::memory_order_relaxed);
    return Status::kMiss;
  }
  std::memcpy(dst, src, size);
  e->pins_.fetch_sub(1, std::memory_order_acq_rel);
  root_->stats_.get_hits_.fetch_add(1, std::memory_order_relaxed);
  return Status::kOk;
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
  if (!(e->flags_ & kComplete)) {
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
  Guard g(root_->tag_lock_);
  TagEntry *t = FindTagLocked(major, minor, false);
  if (t != nullptr) {
    t->state_ = kSlotTomb;
    t->flags_ = 0;
  }
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
  WaitUnpinned(e);
  e->flags_ &= ~kComplete;
  e->size_ = 0;
  if (!EnsureCapacityLocked(e, size == 0 ? 1 : size, stripe)) {
    return false;
  }
  std::memcpy(Ptr(e->buf_off_), data, size);
  std::memset(Ptr(e->buf_off_) + size, 0, e->cap_ - size);
  e->size_ = size;
  e->flags_ |= kComplete;
  e->hits_.fetch_add(1, std::memory_order_relaxed);
  e->last_access_ns_.store(NowNs(), std::memory_order_relaxed);
  root_->stats_.fills_.fetch_add(1, std::memory_order_relaxed);
  return true;
}

inline void View::Invalidate(u32 tag_major, u32 tag_minor, const char *name,
                             size_t name_len) {
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
  e->flags_ &= ~kComplete;
  e->inval_epoch_.fetch_add(1, std::memory_order_acq_rel);
  if (e->pending_.load() == 0) {
    WaitUnpinned(e);
    ReleaseBufferLocked(e);
  }
  root_->stats_.invalidations_.fetch_add(1, std::memory_order_relaxed);
}

inline void View::DropBlob(u32 tag_major, u32 tag_minor, const char *name,
                           size_t name_len) {
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
  if (e->pending_.load() != 0) {
    return;  // a racing write logged after the caller's barrier keeps it
  }
  WaitUnpinned(e);
  ReleaseBufferLocked(e);
  e->state_ = kSlotTomb;
}

inline void View::DropTagBlobs(u32 major, u32 minor) {
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
      if (e->pending_.load() != 0) {
        continue;
      }
      WaitUnpinned(e);
      ReleaseBufferLocked(e);
      e->state_ = kSlotTomb;
    }
  }
}

inline bool View::ReadRange(u32 slot, u64 off, u64 *size, char *dst,
                            u32 *tag_major, u32 *tag_minor,
                            std::string *name, u32 *put_flags) {
  if (slot >= NumSlots()) {
    return false;
  }
  u32 stripe = slot / root_->slots_per_stripe_;
  BlobEntry *e = Slot(slot);
  const char *src = nullptr;
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
    e->pins_.fetch_add(1, std::memory_order_acq_rel);
    src = Ptr(e->buf_off_) + off;
  }
  if (*size > 0) {
    std::memcpy(dst, src, *size);
  }
  e->pins_.fetch_sub(1, std::memory_order_acq_rel);
  return true;
}

inline void View::CompleteRanges(u32 slot, u32 n) {
  if (slot < NumSlots()) {
    Slot(slot)->pending_.fetch_sub(n, std::memory_order_acq_rel);
  }
}

}  // namespace clio::cte::core::wtc

#endif  // CLIO_CTE_CORE_SHM_WTC_IMPL_H_
