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

#ifndef CTP_DATA_STRUCTURES_PRIV_UNORDERED_MAP_LHASH_H_
#define CTP_DATA_STRUCTURES_PRIV_UNORDERED_MAP_LHASH_H_

// `unordered_map_lhash` is the open-addressing / linear-probing hash table
// that used to be `unordered_map_ll`. Single contiguous slot array,
// stripe-locked across slots, GPU-friendly because every key lookup is a
// predictable linear scan. CPU code that wants better concurrent
// insert/erase performance should prefer the chaining `unordered_map_ll`
// in the sibling header.

#include "clio_ctp/data_structures/priv/vector.h"
#include "clio_ctp/types/hash.h"
#include "clio_ctp/memory/allocator/malloc_allocator.h"
#include "clio_ctp/thread/lock/mutex.h"

namespace ctp::priv {

#ifndef CTP_PRIV_INSERT_RESULT_DEFINED_
#define CTP_PRIV_INSERT_RESULT_DEFINED_
/** Result of insert / insert_or_assign operations. Same definition is
 *  guarded in `unordered_map_ll.h`; whichever header is included first
 *  installs it, the other one sees the guard and skips. */
template <typename T>
struct InsertResult {
  bool inserted;
  T *value;
};
#endif  // CTP_PRIV_INSERT_RESULT_DEFINED_

/**
 * GPU-compatible unordered map using open addressing with linear probing.
 *
 * Backed by a single priv::vector of slots. Each slot is either empty,
 * occupied, or a tombstone. Uses fine-grained stripe locking: the slot
 * space is divided into num_locks stripes, each protected by its own
 * Mutex. Operations on different stripes proceed in parallel.
 *
 * @tparam Key      Key type (must support copy/move and operator==)
 * @tparam T        Mapped value type
 * @tparam AllocT   Allocator type (e.g., BuddyAllocator, PartitionedAllocator)
 * @tparam Hash     Hash functor (defaults to ctp::hash<Key>)
 * @tparam KeyEqual Equality functor (defaults to ctp::equal_to<Key>)
 */
template <typename Key, typename T,
          typename AllocT = ctp::ipc::MallocAllocator,
          typename Hash = ctp::hash<Key>,
          typename KeyEqual = ctp::equal_to<Key>>
class unordered_map_lhash {
 public:
  using key_type = Key;
  using mapped_type = T;
  using size_type = std::size_t;
  using hasher = Hash;
  using key_equal = KeyEqual;

 private:
  static constexpr uint32_t kEmpty = 0;
  static constexpr uint32_t kOccupied = 1;
  static constexpr uint32_t kTombstone = 2;
  static constexpr size_type kDefaultNumLocks = 64;

  struct Slot {
    uint32_t state_;
    Key key_;
    T value_;

    CTP_CROSS_FUN Slot() : state_(kEmpty), key_(), value_() {}
    CTP_CROSS_FUN Slot(const Slot &o)
        : state_(o.state_), key_(o.key_), value_(o.value_) {}
    CTP_CROSS_FUN Slot &operator=(const Slot &o) {
      if (this != &o) {
        state_ = o.state_;
        key_ = o.key_;
        value_ = o.value_;
      }
      return *this;
    }
  };

  vector<Slot, AllocT> slots_;
  ctp::ipc::atomic<size_type> size_;
  AllocT *alloc_;
  Hash hash_fn_;
  KeyEqual key_eq_;
  vector<ctp::Mutex, AllocT> locks_;
  size_type num_locks_;

  /** Get the stripe index for a hash value */
  CTP_INLINE_CROSS_FUN
  size_type stripe_of(size_type h) const {
    return h % num_locks_;
  }

  /** Lock a stripe */
  CTP_INLINE_CROSS_FUN
  void lock_stripe(size_type stripe) {
    locks_[stripe].Lock(0);
  }

  /** Unlock a stripe */
  CTP_INLINE_CROSS_FUN
  void unlock_stripe(size_type stripe) {
    locks_[stripe].Unlock();
  }

  /** Lock all stripes (for rehash/clear/for_each) */
  CTP_INLINE_CROSS_FUN
  void lock_all() {
    for (size_type i = 0; i < num_locks_; ++i) {
      locks_[i].Lock(0);
    }
  }

  /** Unlock all stripes */
  CTP_INLINE_CROSS_FUN
  void unlock_all() {
    for (size_type i = 0; i < num_locks_; ++i) {
      locks_[i].Unlock();
    }
  }

  /** Find the slot index for a key (returns capacity if not found).
   *  Caller must hold the appropriate stripe lock. */
  CTP_INLINE_CROSS_FUN
  size_type find_slot(const Key &key) const {
    size_type cap = slots_.size();
    if (cap == 0) return cap;
    size_type h = hash_fn_(key) % cap;
    for (size_type i = 0; i < cap; ++i) {
      size_type idx = (h + i) % cap;
      if (slots_[idx].state_ == kEmpty) return cap;
      if (slots_[idx].state_ == kOccupied && key_eq_(slots_[idx].key_, key)) {
        return idx;
      }
    }
    return cap;
  }

  /** Find the first available slot (empty or tombstone) for insertion.
   *  Caller must hold the appropriate stripe lock. */
  CTP_INLINE_CROSS_FUN
  void find_insert_slot(const Key &key, size_type &out_idx,
                        bool &out_existing) const {
    size_type cap = slots_.size();
    size_type h = hash_fn_(key) % cap;
    size_type first_avail = cap;
    out_existing = false;
    out_idx = cap;
    for (size_type i = 0; i < cap; ++i) {
      size_type idx = (h + i) % cap;
      if (slots_[idx].state_ == kOccupied && key_eq_(slots_[idx].key_, key)) {
        out_idx = idx;
        out_existing = true;
        return;
      }
      if (slots_[idx].state_ != kOccupied && first_avail == cap) {
        first_avail = idx;
      }
      if (slots_[idx].state_ == kEmpty) {
        break;
      }
    }
    out_idx = first_avail;
  }

  /** Initialize stripe locks */
  CTP_CROSS_FUN
  void init_locks(size_type num_locks) {
    num_locks_ = num_locks;
    locks_.resize(num_locks_);
    for (size_type i = 0; i < num_locks_; ++i) {
      locks_[i].Init();
    }
  }

 public:
  /**
   * Constructor (host-only, uses global MallocAllocator)
   * @param capacity Initial number of slots (hash table size)
   * @param num_locks Number of stripe locks (default: 64)
   */
#if CTP_IS_HOST
  explicit unordered_map_lhash(size_type capacity = 16,
                            size_type num_locks = kDefaultNumLocks)
      : slots_(CTP_MALLOC), size_(0), alloc_(CTP_MALLOC),
        hash_fn_(), key_eq_(), locks_(CTP_MALLOC), num_locks_(0) {
    slots_.resize(capacity);
    init_locks(num_locks < capacity ? num_locks : capacity);
  }
#endif

  /**
   * Constructor with explicit allocator
   * @param alloc Allocator for the backing vector
   * @param capacity Initial number of slots (hash table size)
   * @param num_locks Number of stripe locks (default: 64)
   */
  CTP_CROSS_FUN
  explicit unordered_map_lhash(AllocT *alloc, size_type capacity = 16,
                            size_type num_locks = kDefaultNumLocks)
      : slots_(alloc), size_(0), alloc_(alloc),
        hash_fn_(), key_eq_(), locks_(alloc), num_locks_(0) {
    slots_.resize(capacity);
    init_locks(num_locks < capacity ? num_locks : capacity);
  }

  CTP_CROSS_FUN ~unordered_map_lhash() = default;

  /** Rehash the map to a new capacity, re-inserting all occupied entries.
   *  Acquires all stripe locks. Returns false if allocation fails. */
  CTP_CROSS_FUN
  bool rehash(size_type new_cap) {
    lock_all();
    bool result = rehash_no_lock(new_cap);
    unlock_all();
    return result;
  }

 private:
  /** Rehash without locking (caller must hold all locks) */
  CTP_CROSS_FUN
  bool rehash_no_lock(size_type new_cap) {
    // Try to allocate new slots first (before destroying old)
    vector<Slot, AllocT> new_slots(alloc_);
    if (!new_slots.resize(new_cap)) {
      return false;  // Allocation failed; keep existing map intact
    }

    // Move old slots out; slots_ becomes empty after move
    vector<Slot, AllocT> old_slots(static_cast<vector<Slot, AllocT>&&>(slots_));
    size_type old_size = old_slots.size();
    // Install the new (empty) slots
    slots_ = static_cast<vector<Slot, AllocT>&&>(new_slots);
    size_ = 0;
    for (size_type i = 0; i < old_size; ++i) {
      if (old_slots[i].state_ == kOccupied) {
        insert_no_rehash(old_slots[i].key_, old_slots[i].value_);
      }
    }
    return true;
  }

  /** Check load factor and rehash if needed (>75% full).
   *  Caller must hold all locks. */
  CTP_INLINE_CROSS_FUN
  void maybe_rehash_locked() {
    size_type cur_size = size_.load();
    if (cur_size * 4 > slots_.size() * 3) {
      rehash_no_lock(slots_.size() * 2);
    }
  }

  /** Insert without rehash check (used internally by rehash) */
  CTP_CROSS_FUN
  InsertResult<T> insert_no_rehash(const Key &key, const T &value) {
    size_type idx;
    bool existing;
    find_insert_slot(key, idx, existing);
    if (idx >= slots_.size()) {
      return {false, nullptr};
    }
    if (existing) {
      return {false, &slots_[idx].value_};
    }
    slots_[idx].state_ = kOccupied;
    slots_[idx].key_ = key;
    slots_[idx].value_ = value;
    size_.fetch_add(1);
    return {true, &slots_[idx].value_};
  }

 public:
  /** Insert or update a key-value pair (thread-safe) */
  CTP_CROSS_FUN
  InsertResult<T> insert_or_assign(const Key &key, const T &value) {
    size_type h = hash_fn_(key);
    size_type stripe = stripe_of(h);
    lock_stripe(stripe);
    InsertResult<T> result = insert_or_assign_no_lock(key, value);
    unlock_stripe(stripe);
    return result;
  }

  /** Insert a key-value pair, only if key doesn't exist (thread-safe) */
  CTP_CROSS_FUN
  InsertResult<T> insert(const Key &key, const T &value) {
    size_type h = hash_fn_(key);
    size_type stripe = stripe_of(h);
    lock_stripe(stripe);
    InsertResult<T> result = insert_no_lock(key, value);
    unlock_stripe(stripe);
    return result;
  }

  /** Access element, creates with default value if absent (thread-safe) */
  CTP_CROSS_FUN
  T &operator[](const Key &key) {
    size_type h = hash_fn_(key);
    size_type stripe = stripe_of(h);
    lock_stripe(stripe);
    T &ref = subscript_no_lock(key);
    unlock_stripe(stripe);
    return ref;
  }

  /** Lock the stripe for a key. Must call unlock_key() when done. */
  CTP_CROSS_FUN
  void lock_key(const Key &key) {
    size_type h = hash_fn_(key);
    size_type s = stripe_of(h);
    lock_stripe(s);
  }

  /** Unlock the stripe for a key. */
  CTP_CROSS_FUN
  void unlock_key(const Key &key) {
    size_type h = hash_fn_(key);
    size_type s = stripe_of(h);
    unlock_stripe(s);
  }

  /** Find an element (thread-safe, returns pointer valid while map lives) */
  CTP_CROSS_FUN
  T *find(const Key &key) {
    size_type h = hash_fn_(key);
    size_type stripe = stripe_of(h);
    lock_stripe(stripe);
    size_type idx = find_slot(key);
    T *result = (idx < slots_.size()) ? &slots_[idx].value_ : nullptr;
    unlock_stripe(stripe);
    return result;
  }

  /** Find while holding the stripe lock. Caller must have called lock_key(). */
  CTP_CROSS_FUN
  T *find_locked(const Key &key) {
    size_type idx = find_slot(key);
    return (idx < slots_.size()) ? &slots_[idx].value_ : nullptr;
  }

  /** Insert while holding the stripe lock. Caller must have called lock_key(). */
  CTP_CROSS_FUN
  InsertResult<T> insert_locked(const Key &key, const T &value) {
    return insert_no_lock(key, value);
  }

  /** Erase while holding the stripe lock. Caller must have called lock_key(). */
  CTP_CROSS_FUN
  size_type erase_locked(const Key &key) {
    size_type idx = find_slot(key);
    if (idx < slots_.size()) {
      slots_[idx].state_ = kTombstone;
      slots_[idx].key_ = Key();
      slots_[idx].value_ = T();
      size_.fetch_sub(1);
      return 1;
    }
    return 0;
  }

  /** Find an element (const, thread-safe) */
  CTP_CROSS_FUN
  const T *find(const Key &key) const {
    size_type h = hash_fn_(key);
    size_type stripe = stripe_of(h);
    const_cast<unordered_map_lhash*>(this)->lock_stripe(stripe);
    size_type idx = find_slot(key);
    const T *result = (idx < slots_.size()) ? &slots_[idx].value_ : nullptr;
    const_cast<unordered_map_lhash*>(this)->unlock_stripe(stripe);
    return result;
  }

  /** Check if key exists (thread-safe) */
  CTP_CROSS_FUN
  bool contains(const Key &key) {
    return find(key) != nullptr;
  }

  /** Count occurrences, 0 or 1 (thread-safe) */
  CTP_CROSS_FUN
  size_type count(const Key &key) {
    return contains(key) ? 1 : 0;
  }

  /** Erase element by key (thread-safe) */
  CTP_CROSS_FUN
  size_type erase(const Key &key) {
    size_type h = hash_fn_(key);
    size_type stripe = stripe_of(h);
    lock_stripe(stripe);
    size_type idx = find_slot(key);
    size_type result = 0;
    if (idx < slots_.size()) {
      slots_[idx].state_ = kTombstone;
      slots_[idx].key_ = Key();
      slots_[idx].value_ = T();
      size_.fetch_sub(1);
      result = 1;
    }
    unlock_stripe(stripe);
    return result;
  }

  /** Clear all elements (thread-safe, acquires all locks) */
  CTP_CROSS_FUN
  void clear() {
    lock_all();
    for (size_type i = 0; i < slots_.size(); ++i) {
      if (slots_[i].state_ == kOccupied) {
        slots_[i].key_ = Key();
        slots_[i].value_ = T();
      }
      slots_[i].state_ = kEmpty;
    }
    size_ = 0;
    unlock_all();
  }

  /** Total number of elements */
  CTP_INLINE_CROSS_FUN
  size_type size() const { return size_.load(); }

  /** Check if empty */
  CTP_INLINE_CROSS_FUN
  bool empty() const { return size() == 0; }

  /** Number of slots */
  CTP_INLINE_CROSS_FUN
  size_type bucket_count() const { return slots_.size(); }

  /** Apply function to each occupied entry (thread-safe, acquires all locks) */
  template <typename Func>
  CTP_CROSS_FUN void for_each(Func fn) {
    lock_all();
    for (size_type i = 0; i < slots_.size(); ++i) {
      if (slots_[i].state_ == kOccupied) {
        fn(slots_[i].key_, slots_[i].value_);
      }
    }
    unlock_all();
  }

  /** Apply function to each occupied entry (const, thread-safe) */
  template <typename Func>
  CTP_CROSS_FUN void for_each(Func fn) const {
    const_cast<unordered_map_lhash*>(this)->lock_all();
    for (size_type i = 0; i < slots_.size(); ++i) {
      if (slots_[i].state_ == kOccupied) {
        fn(slots_[i].key_, slots_[i].value_);
      }
    }
    const_cast<unordered_map_lhash*>(this)->unlock_all();
  }

 private:
  /** Insert or assign without locking (caller holds stripe lock) */
  CTP_CROSS_FUN
  InsertResult<T> insert_or_assign_no_lock(const Key &key, const T &value) {
    size_type idx;
    bool existing;
    find_insert_slot(key, idx, existing);
    if (existing) {
      slots_[idx].value_ = value;
      return {false, &slots_[idx].value_};
    }
    if (idx >= slots_.size()) {
      // Need rehash — must acquire all locks
      // Release our stripe lock first to avoid deadlock
      size_type h = hash_fn_(key);
      size_type stripe = stripe_of(h);
      unlock_stripe(stripe);
      lock_all();
      if (!rehash_no_lock(slots_.size() * 2)) {
        unlock_all();
        lock_stripe(stripe);
        return {false, nullptr};
      }
      // Re-do insert under all locks
      InsertResult<T> result = insert_no_rehash(key, value);
      unlock_all();
      lock_stripe(stripe);
      return result;
    }
    slots_[idx].state_ = kOccupied;
    slots_[idx].key_ = key;
    slots_[idx].value_ = value;
    size_.fetch_add(1);
    // Check if rehash needed
    size_type cur_size = size_.load();
    if (cur_size * 4 > slots_.size() * 3) {
      size_type h = hash_fn_(key);
      size_type stripe = stripe_of(h);
      unlock_stripe(stripe);
      lock_all();
      maybe_rehash_locked();
      unlock_all();
      lock_stripe(stripe);
    }
    return {true, &slots_[idx].value_};
  }

  /** Insert without locking (caller holds stripe lock) */
  CTP_CROSS_FUN
  InsertResult<T> insert_no_lock(const Key &key, const T &value) {
    size_type idx;
    bool existing;
    find_insert_slot(key, idx, existing);
    if (existing) {
      return {false, &slots_[idx].value_};
    }
    if (idx >= slots_.size()) {
      size_type h = hash_fn_(key);
      size_type stripe = stripe_of(h);
      unlock_stripe(stripe);
      lock_all();
      if (!rehash_no_lock(slots_.size() * 2)) {
        unlock_all();
        lock_stripe(stripe);
        return {false, nullptr};
      }
      InsertResult<T> result = insert_no_rehash(key, value);
      unlock_all();
      lock_stripe(stripe);
      return result;
    }
    slots_[idx].state_ = kOccupied;
    slots_[idx].key_ = key;
    slots_[idx].value_ = value;
    size_.fetch_add(1);
    size_type cur_size = size_.load();
    if (cur_size * 4 > slots_.size() * 3) {
      size_type h = hash_fn_(key);
      size_type stripe = stripe_of(h);
      unlock_stripe(stripe);
      lock_all();
      maybe_rehash_locked();
      unlock_all();
      lock_stripe(stripe);
    }
    return {true, &slots_[idx].value_};
  }

  /** operator[] without locking (caller holds stripe lock) */
  CTP_CROSS_FUN
  T &subscript_no_lock(const Key &key) {
    size_type idx;
    bool existing;
    find_insert_slot(key, idx, existing);
    if (existing) {
      return slots_[idx].value_;
    }
    if (idx >= slots_.size()) {
      size_type h = hash_fn_(key);
      size_type stripe = stripe_of(h);
      unlock_stripe(stripe);
      lock_all();
      rehash_no_lock(slots_.size() * 2);
      // Re-find after rehash
      find_insert_slot(key, idx, existing);
      if (existing) {
        unlock_all();
        lock_stripe(stripe);
        return slots_[idx].value_;
      }
      slots_[idx].state_ = kOccupied;
      slots_[idx].key_ = key;
      slots_[idx].value_ = T();
      size_.fetch_add(1);
      maybe_rehash_locked();
      T &ref = slots_[idx].value_;
      unlock_all();
      lock_stripe(stripe);
      return ref;
    }
    slots_[idx].state_ = kOccupied;
    slots_[idx].key_ = key;
    slots_[idx].value_ = T();
    size_.fetch_add(1);
    size_type cur_size = size_.load();
    if (cur_size * 4 > slots_.size() * 3) {
      size_type h = hash_fn_(key);
      size_type stripe = stripe_of(h);
      unlock_stripe(stripe);
      lock_all();
      maybe_rehash_locked();
      unlock_all();
      lock_stripe(stripe);
    }
    return slots_[idx].value_;
  }
};

}  // namespace ctp::priv

#endif  // CTP_DATA_STRUCTURES_PRIV_UNORDERED_MAP_LHASH_H_
