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

#ifndef CTP_DATA_STRUCTURES_IPC_UNORDERED_MAP_H_
#define CTP_DATA_STRUCTURES_IPC_UNORDERED_MAP_H_

#include "clio_ctp/data_structures/ipc/shm_container.h"
#include "clio_ctp/data_structures/ipc/string.h"
#include "clio_ctp/types/atomic.h"

#include <new>
#include <type_traits>

namespace ctp::ipc {

/** Detects a key type that provides its own stable hash()/EqualsBytes(). */
namespace detail {
template <typename T, typename = void>
struct HasShmHash : std::false_type {};

template <typename T>
struct HasShmHash<T, std::void_t<decltype(std::declval<const T &>().hash())>>
    : std::true_type {};
}  // namespace detail

/**
 * Shared-memory hash map (issue #783).
 *
 * SINGLE WRITER, MANY CROSS-PROCESS READERS. The runtime owns the table and is
 * the only mutator; clients read it concurrently and may be SIGKILLed at any
 * instant, including mid-read.
 *
 * DESIGN CHOICES, and why:
 *
 * 1. OPEN ADDRESSING, not chaining. A lock-free reader can probe a flat array
 *    safely; chasing chain pointers while the writer recycles nodes would need
 *    epoch protection on every node.
 *
 * 2. FIXED CAPACITY -- the table NEVER rehashes. Growth is the one operation
 *    that cannot be made safe for untracked cross-process readers: a rehash
 *    must free the old table, and there is no way to know when the last reader
 *    has left it without exactly the epoch machinery this design rejected.
 *    Instead the map is sized up front and a full table is reported as an
 *    insert failure, which the cache already has to handle -- "cache full ->
 *    fall back to RPC" is a required path regardless. Capacity is rounded up
 *    to a power of two so the modulo is a mask.
 *
 * 3. PER-SLOT SEQLOCK. Readers store NOTHING: read the slot generation, copy
 *    key+value out, re-read the generation, and retry if it moved or was odd.
 *    A reader that dies leaves no state to reclaim -- which is precisely why
 *    the default read path is a seqlock rather than a Lease.
 *
 * 4. TOMBSTONES on erase, because a probe sequence must not be broken by a
 *    hole. Tombstones are reusable by a later insert of the same key.
 *
 * 5. READERS NEVER ALLOCATE. Lookup by raw bytes (TryGetBytes) exists so a
 *    client can probe a string-keyed table without constructing a key inside
 *    the runtime-owned segment.
 *
 * SHM-SAFE: no virtuals, offsets only.
 *
 * @tparam KeyT   key type; either provides hash()/EqualsBytes() (e.g.
 *                ipc::string) or is a trivially comparable POD.
 * @tparam ValueT value type; must be trivially copyable (it is memcpy'd out
 *                under the seqlock).
 * @tparam AllocT allocator type.
 */
template <typename KeyT, typename ValueT, typename AllocT>
class unordered_map : public ShmContainer<AllocT> {
 public:
  using allocator_type = AllocT;

  enum SlotState : min_u32 { kEmpty = 0, kOccupied = 1, kTombstone = 2 };

  /**
   * Maximum load factor (7/8). NEW keys are refused past this point even
   * though slots remain.
   *
   * This is not tidiness, it is the difference between a plateau and a cliff.
   * A probe run only terminates at an EMPTY slot, so a table with none makes
   * every MISS scan the whole table. Measured on a 65536-slot table:
   *
   *     load  50%  ->  miss   0.06 us
   *     load  90%  ->  miss   0.18 us
   *     load  99%  ->  miss  11.8  us
   *     load 100%  ->  miss 108.9  us
   *
   * At 100% a miss costs MORE than the ~90 us RPC it exists to avoid -- and
   * the caller then pays that RPC anyway, so the cache turns net-negative.
   * Reserving 1/8 of the slots keeps the expected miss probe length ~8 and
   * bounds it in practice.
   */
  static constexpr size_t kMaxLoadNum = 7;
  static constexpr size_t kMaxLoadDen = 8;

  /**
   * Hard cap on probe length for a single lookup.
   *
   * Purely a pathological-case backstop: it must NOT be the binding
   * constraint, the load factor must be. Linear probing forms clusters far
   * longer than the average probe length, so a tight cap silently rejects
   * legitimate inserts long before the table is full -- measured at 64, a
   * 65536-slot table stopped accepting new keys at ~51% occupancy instead of
   * 87.5%, cutting cache coverage for 1M blobs from 26% to 18%.
   *
   * 1024 leaves the load factor in charge while still bounding a worst-case
   * miss to ~1 us, versus the ~109 us unbounded scan it replaces.
   */
  static constexpr size_t kMaxProbe = 1024;

  struct Slot {
    ipc::atomic<min_u32> state_;
    /** Seqlock counter. Odd => a write is in progress on this slot. */
    ipc::atomic<min_u64> gen_;
    KeyT key_;
    ValueT value_;
  };

 private:
  size_t capacity_;  /**< Power of two; never changes after construction */
  size_t size_;      /**< Occupied slots. Writer-only; advisory for readers */
  size_t max_fill_;  /**< Refuse NEW keys past this; see kMaxLoadNum */
  OffsetPtr<Slot> slots_;

 public:
  CTP_CROSS_FUN
  unordered_map()
      : ShmContainer<AllocT>(),
        capacity_(0),
        size_(0),
        max_fill_(0),
        slots_(OffsetPtr<Slot>::GetNull()) {}

  /**
   * @param alloc    allocator (must own the same segment as this object)
   * @param capacity desired entry count; rounded up to a power of two. Size
   *                 generously: the table cannot grow later.
   */
  CTP_INLINE_CROSS_FUN
  unordered_map(AllocT *alloc, size_t capacity)
      : ShmContainer<AllocT>(alloc),
        capacity_(0),
        size_(0),
        max_fill_(0),
        slots_(OffsetPtr<Slot>::GetNull()) {
    Allocate(capacity);
  }

  unordered_map(const unordered_map &) = delete;
  unordered_map &operator=(const unordered_map &) = delete;

  CTP_INLINE_CROSS_FUN
  ~unordered_map() {
    AllocT *alloc = this->GetAllocator();
    if (alloc != nullptr && !slots_.IsNull()) {
      alloc->FreeOffsetNoNullCheck(OffsetPtr<>(slots_.load()));
    }
    slots_ = OffsetPtr<Slot>::GetNull();
    capacity_ = 0;
    size_ = 0;
  }

  CTP_INLINE_CROSS_FUN
  size_t size() const { return size_; }

  CTP_INLINE_CROSS_FUN
  size_t capacity() const { return capacity_; }

  CTP_INLINE_CROSS_FUN
  bool valid() const { return !slots_.IsNull() && capacity_ > 0; }

  // -------------------------------------------------------------------------
  // Writer side (runtime only)
  // -------------------------------------------------------------------------

  /**
   * Insert or overwrite. Returns false if the table is full (never grows).
   *
   * The value is published under the slot's seqlock: bump generation to odd,
   * write, bump back to even. A concurrent reader observing the odd value
   * retries rather than reading a half-written entry.
   */
  template <typename KeyInit>
  CTP_INLINE_CROSS_FUN bool InsertOrAssign(const KeyInit &key_probe,
                                           size_t key_hash,
                                           const ValueT &value) {
    if (!valid()) {
      return false;
    }
    Slot *slots = Slots();
    size_t mask = capacity_ - 1;
    size_t idx = key_hash & mask;
    size_t first_tombstone = capacity_;  // sentinel: none seen
    // Overwrites of an existing key are always allowed; only NEW keys are
    // subject to the load cap, so a saturated cache keeps serving (and
    // updating) what it already holds instead of going stale.
    const bool accept_new = (size_ < max_fill_);
    const size_t probe_limit =
        capacity_ < kMaxProbe ? capacity_ : kMaxProbe;

    for (size_t probe = 0; probe < probe_limit; ++probe) {
      Slot &s = slots[idx];
      min_u32 st = s.state_.load();
      if (st == kOccupied) {
        if (KeyMatches(s.key_, key_probe)) {
          BeginWrite(s);
          s.value_ = value;
          EndWrite(s);
          return true;  // key already correct; only the value changed
        }
      } else {
        if (st == kTombstone && first_tombstone == capacity_) {
          first_tombstone = idx;
        } else if (st == kEmpty) {
          if (!accept_new) {
            return false;  // at load cap: not cached, caller uses RPC
          }
          // Prefer an earlier tombstone to keep probe sequences short.
          size_t target = (first_tombstone != capacity_) ? first_tombstone : idx;
          Slot &t = slots[target];
          BeginWrite(t);
          if (!AssignKey(t.key_, key_probe)) {
            // Key storage could not be allocated. Abort the insert cleanly and
            // leave the slot unusable-but-consistent rather than publishing an
            // entry whose key is empty -- that would alias every other lookup
            // that happens to probe here.
            t.state_.store(kEmpty);
            EndWrite(t);
            return false;
          }
          t.value_ = value;
          t.state_.store(kOccupied);
          EndWrite(t);
          ++size_;
          return true;
        }
      }
      idx = (idx + 1) & mask;
    }
    return false;  // at load cap, or probe limit hit
  }

  /**
   * Writer-side lookup. Returns a pointer INTO the table, valid only while the
   * writer holds whatever external synchronization it uses. Readers in other
   * processes must use TryGet/TryGetBytes instead -- a raw pointer gives them
   * no way to detect a concurrent overwrite.
   */
  template <typename KeyInit>
  CTP_INLINE_CROSS_FUN Slot *FindSlot(const KeyInit &key_probe,
                                      size_t key_hash) {
    if (!valid()) {
      return nullptr;
    }
    Slot *slots = Slots();
    size_t mask = capacity_ - 1;
    size_t idx = key_hash & mask;
    const size_t probe_limit = capacity_ < kMaxProbe ? capacity_ : kMaxProbe;
    for (size_t probe = 0; probe < probe_limit; ++probe) {
      Slot &s = slots[idx];
      min_u32 st = s.state_.load();
      if (st == kEmpty) {
        return nullptr;  // probe sequence ends
      }
      if (st == kOccupied && KeyMatches(s.key_, key_probe)) {
        return &s;
      }
      idx = (idx + 1) & mask;
    }
    return nullptr;
  }

  /** Erase by key. Leaves a tombstone so probe sequences stay intact. */
  template <typename KeyInit>
  CTP_INLINE_CROSS_FUN bool Erase(const KeyInit &key_probe, size_t key_hash) {
    Slot *s = FindSlot(key_probe, key_hash);
    if (s == nullptr) {
      return false;
    }
    BeginWrite(*s);
    s->state_.store(kTombstone);
    EndWrite(*s);
    if (size_ > 0) {
      --size_;
    }
    return true;
  }

  // -------------------------------------------------------------------------
  // Reader side (safe cross-process; performs NO stores)
  // -------------------------------------------------------------------------

  /**
   * Copy a value out under the slot's seqlock.
   *
   * @param max_retries bounds the retry loop so a reader cannot spin forever
   *        against a pathologically hot slot (or a writer that died mid-update
   *        leaving the generation odd -- see the TimedMutex reclamation path).
   *        Exhausting retries returns false, and the caller falls back to RPC.
   * @return true if a consistent value was copied out.
   */
  template <typename KeyInit>
  CTP_INLINE_CROSS_FUN bool TryGet(const KeyInit &key_probe, size_t key_hash,
                                   ValueT *out,
                                   min_u32 max_retries = 16) const {
    if (!valid() || out == nullptr) {
      return false;
    }
    const Slot *slots = Slots();
    size_t mask = capacity_ - 1;

    for (min_u32 attempt = 0; attempt < max_retries; ++attempt) {
      size_t idx = key_hash & mask;
      bool saw_torn = false;
      const size_t probe_limit =
          capacity_ < kMaxProbe ? capacity_ : kMaxProbe;
      for (size_t probe = 0; probe < probe_limit; ++probe) {
        const Slot &s = slots[idx];
        min_u64 g0 = s.gen_.load();
        if ((g0 & 1) != 0) {
          saw_torn = true;  // write in progress; restart the whole probe
          break;
        }
        min_u32 st = s.state_.load();
        if (st == kEmpty) {
          // End of probe sequence. Only trust it if nothing changed under us.
          if (s.gen_.load() == g0) {
            return false;
          }
          saw_torn = true;
          break;
        }
        if (st == kOccupied && KeyMatches(s.key_, key_probe)) {
          ValueT tmp = s.value_;
          // Re-validate: if the generation moved, the bytes we just copied may
          // be a mix of the old and new value.
          if (s.gen_.load() == g0) {
            *out = tmp;
            return true;
          }
          saw_torn = true;
          break;
        }
        idx = (idx + 1) & mask;
      }
      if (!saw_torn) {
        return false;
      }
    }
    return false;  // gave up; caller falls back
  }

  /**
   * Lookup by raw bytes for string-keyed tables -- no key object is
   * constructed, so a client never allocates inside the segment it is only
   * supposed to read.
   */
  CTP_INLINE_CROSS_FUN
  bool TryGetBytes(const char *key, size_t len, ValueT *out,
                   min_u32 max_retries = 16) const {
    return TryGet(BytesProbe{key, len}, KeyT::HashBytes(key, len), out,
                  max_retries);
  }

  /** Hash a key the same way the table does. */
  template <typename KeyInit>
  CTP_INLINE_CROSS_FUN static size_t Hash(const KeyInit &k) {
    if constexpr (detail::HasShmHash<KeyInit>::value) {
      return k.hash();
    } else {
      return HashPod(&k, sizeof(KeyInit));
    }
  }

  CTP_INLINE_CROSS_FUN
  static size_t HashPod(const void *p, size_t n) {
    const unsigned char *b = static_cast<const unsigned char *>(p);
    size_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; ++i) {
      h ^= static_cast<size_t>(b[i]);
      h *= 1099511628211ULL;
    }
    return h;
  }

  /** Raw-bytes probe wrapper (see TryGetBytes). */
  struct BytesProbe {
    const char *p;
    size_t n;
  };

 private:
  CTP_INLINE_CROSS_FUN
  Slot *Slots() {
    return FullPtr<Slot>(this->GetAllocator(), slots_).ptr_;
  }

  CTP_INLINE_CROSS_FUN
  const Slot *Slots() const {
    return FullPtr<Slot>(this->GetAllocator(), slots_).ptr_;
  }

  /** Publish-begin: make the generation odd so readers retry. */
  CTP_INLINE_CROSS_FUN
  static void BeginWrite(Slot &s) { s.gen_.fetch_add(1); }

  /** Publish-end: back to even. */
  CTP_INLINE_CROSS_FUN
  static void EndWrite(Slot &s) { s.gen_.fetch_add(1); }

  template <typename KeyInit>
  CTP_INLINE_CROSS_FUN static bool KeyMatches(const KeyT &stored,
                                              const KeyInit &probe) {
    if constexpr (std::is_same_v<KeyInit, BytesProbe>) {
      return stored.EqualsBytes(probe.p, probe.n);
    } else {
      return stored == probe;
    }
  }

  /**
   * Copy key material into a slot.
   *
   * Slot keys are constructed once (with the allocator) when the table is
   * allocated and then reused, so an insert ASSIGNS rather than constructs --
   * a construct here would leak the previous key's buffer and, for
   * ipc::string, would need an allocator the probe does not carry.
   */
  template <typename KeyInit>
  CTP_INLINE_CROSS_FUN static bool AssignKey(KeyT &stored,
                                             const KeyInit &probe) {
    if constexpr (std::is_same_v<KeyInit, BytesProbe>) {
      return stored.assign(probe.p, probe.n);
    } else if constexpr (detail::HasShmHash<KeyT>::value) {
      return stored.assign(probe.data(), probe.size());
    } else {
      stored = probe;
      return true;
    }
  }

  CTP_INLINE_CROSS_FUN
  void Allocate(size_t capacity) {
    AllocT *alloc = this->GetAllocator();
    if (alloc == nullptr || capacity == 0) {
      return;
    }
    size_t cap = 1;
    while (cap < capacity) {
      cap <<= 1;
    }
    OffsetPtr<> off = OffsetPtr<>::GetNull();
#if !CTP_IS_DEVICE_PASS
    // Exhaustion must be recoverable, not an escaping exception -- see
    // ipc::string::reserve for the same reasoning.
    try {
      off = alloc->AllocateOffset(cap * sizeof(Slot));
    } catch (...) {
      return;
    }
#else
    off = alloc->AllocateOffset(cap * sizeof(Slot));
#endif
    if (off.IsNull()) {
      return;
    }
    slots_ = OffsetPtr<Slot>(off.load());
    capacity_ = cap;
    max_fill_ = (cap * kMaxLoadNum) / kMaxLoadDen;
    if (max_fill_ == 0) {
      max_fill_ = cap > 1 ? cap - 1 : 1;  // always leave one empty slot
    }
    Slot *slots = Slots();
    for (size_t i = 0; i < cap; ++i) {
      // The allocation is raw memory, so every slot member must be constructed
      // explicitly. Keys are constructed WITH the allocator here (once) and
      // assigned on each insert -- an ipc::string key cannot allocate its
      // buffer without one, and a probe does not carry an allocator.
      new (&slots[i].state_) ipc::atomic<min_u32>(kEmpty);
      new (&slots[i].gen_) ipc::atomic<min_u64>(0);
      if constexpr (detail::HasShmHash<KeyT>::value) {
        new (&slots[i].key_) KeyT(alloc);
      } else {
        new (&slots[i].key_) KeyT();
      }
      new (&slots[i].value_) ValueT();
    }
  }
};

}  // namespace ctp::ipc

#endif  // CTP_DATA_STRUCTURES_IPC_UNORDERED_MAP_H_
