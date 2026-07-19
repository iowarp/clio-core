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

#ifndef CTP_DATA_STRUCTURES_PRIV_UNORDERED_MAP_LL_H_
#define CTP_DATA_STRUCTURES_PRIV_UNORDERED_MAP_LL_H_

// `unordered_map_ll` -- linked-list (chaining) unordered map.
//
// Bucket array of singly-linked lists. Each bucket owns its own RwLock
// when `EnableLocking` is true (the default), so concurrent inserts /
// erases / lookups against *different* buckets never contend, and
// concurrent lookups against the *same* bucket share a reader. Set
// `EnableLocking = false` for a single-threaded variant with zero lock
// overhead.
//
// THREAD-SAFETY CONTRACT (important):
//   * Single-key operations (insert, insert_or_assign, operator[],
//     find, contains, count, erase) ARE self-locking and safe to call
//     concurrently from many threads when EnableLocking is true —
//     INCLUDING while the table grows. They are safe to interleave with
//     automatic rehash (growth) at any time.
//   * Safe growth is provided by a map-wide RwLock (`global_lock_`): every
//     single-key op holds it in READ mode (shared) for the duration of its
//     bucket access, and rehash takes it in WRITE mode (exclusive). The
//     write lock drains all in-flight ops before reallocating the bucket
//     and per-bucket-lock arrays, and blocks new ops until growth finishes —
//     so the old use-after-free (a rehash racing the lock-array realloc) can
//     no longer happen. Automatic rehash fires inside maybe_rehash() once
//     size reaches ext_percent_ * bucket_count() after an insert; explicit
//     rehash() / clear() / for_each() also take the write lock.
//   * Bucket-local concurrency is preserved: while the global lock is held
//     shared, ops on DIFFERENT buckets still run in parallel via the
//     per-bucket RwLocks; only growth serializes everything briefly.
//   * Cost: each op takes one shared global read-lock in addition to its
//     bucket lock. A callback passed to for_each MUST NOT re-enter the same
//     map (the exclusive lock would deadlock).
//   * Sizing still matters for PERFORMANCE (not safety): construct with
//     bucket_count() near the expected key count to avoid growth churn.
//
// Counterpart to `unordered_map_lhash` (linear-probing open-addressing,
// previously named `unordered_map_ll`). Chaining trades the predictable
// cache behavior that made linear probing nice on GPU for an unbounded
// load factor and finer-grained locking that scales better on the CPU
// under high contention.

#include "clio_ctp/data_structures/priv/vector.h"
#include "clio_ctp/types/hash.h"
#include "clio_ctp/memory/allocator/malloc_allocator.h"
#include "clio_ctp/thread/lock/rwlock.h"

namespace ctp::priv {

/** Lock mode for unordered_map_ll::for_each(). kExclusive = map-wide write lock
 *  (quiescent map; callback may mutate values). kShared = map-wide read lock +
 *  per-bucket read lock (runs concurrently with readers/single-key ops;
 *  READ-ONLY callbacks only). See for_each() for the full contract. */
enum class ForEachLock { kExclusive, kShared };

#ifndef CTP_PRIV_INSERT_RESULT_DEFINED_
#define CTP_PRIV_INSERT_RESULT_DEFINED_
/** Result of insert / insert_or_assign operations. Guard-shared with
 *  `unordered_map_lhash.h` so callers can include either header (or
 *  both, in any order) without a redefinition. */
template <typename T>
struct InsertResult {
  bool inserted;
  T *value;
};
#endif  // CTP_PRIV_INSERT_RESULT_DEFINED_

/**
 * Chaining unordered map with optional per-bucket fine-grained RwLock.
 *
 * @tparam Key            Key type (copy/move + operator==)
 * @tparam T              Mapped value type
 * @tparam AllocT         Allocator (defaults to MallocAllocator)
 * @tparam Hash           Hash functor (defaults to ctp::hash<Key>)
 * @tparam KeyEqual       Equality functor (defaults to ctp::equal_to<Key>)
 * @tparam EnableLocking  Compile-time toggle for per-bucket RwLock
 *                        (defaults to true). When false the lock array is
 *                        elided and every operation is unsynchronized --
 *                        only use that mode when the map is owned by one
 *                        thread, or when the caller guarantees external
 *                        synchronization.
 */
template <typename Key, typename T,
          typename AllocT = ctp::ipc::MallocAllocator,
          typename Hash = ctp::hash<Key>,
          typename KeyEqual = ctp::equal_to<Key>,
          bool EnableLocking = true>
class unordered_map_ll {
 public:
  using key_type = Key;
  using mapped_type = T;
  using size_type = std::size_t;
  using hasher = Hash;
  using key_equal = KeyEqual;
  static constexpr bool kEnableLocking = EnableLocking;

 private:
  struct Node {
    Key key_;
    T value_;
    Node *next_;
    CTP_CROSS_FUN Node() : key_(), value_(), next_(nullptr) {}
    CTP_CROSS_FUN Node(const Key &k, const T &v, Node *n)
        : key_(k), value_(v), next_(n) {}
  };

  vector<Node *, AllocT> buckets_;  // bucket heads (nullptr = empty)
  vector<ctp::RwLock, AllocT> locks_;  // per-bucket; size 0 when !EnableLocking
  // Map-wide lock making in-place rehash safe under concurrency. Every
  // single-key op holds it in READ mode (shared) for the duration of the
  // bucket access, so the bucket/lock arrays can't be reallocated underneath
  // it; rehash/clear/for_each take it in WRITE mode (exclusive), which drains
  // all in-flight ops first. See the THREAD-SAFETY CONTRACT above.
  ctp::RwLock global_lock_;
  // Reader/writer starvation note: ctp::RwLock is reader-preferring, so a
  // sustained insert/find stream would keep readers_ > 0 and starve the rehash
  // WriteLock (a Windows/icx livelock/timeout in the growth stress test). Both
  // global_read_lock() and read_lock_bucket() therefore request RwLock's opt-in
  // writer_priority mode, which defers new readers while a writer is waiting —
  // so no separate hand-rolled writer-preference counter is needed here.
  ctp::ipc::atomic<size_type> size_;
  AllocT *alloc_;
  Hash hash_fn_;
  KeyEqual key_eq_;
  // Growth policy: rehash once size reaches ext_percent_ * bucket_count, to a
  // new bucket_count of ext_mult_ * bucket_count.
  double ext_percent_;
  size_type ext_mult_;

  /** Rehash threshold (>= 1) for the current bucket count. */
  CTP_INLINE_CROSS_FUN size_type rehash_threshold(size_type cap) const {
    size_type t = static_cast<size_type>(ext_percent_ * static_cast<double>(cap));
    return t < 1 ? 1 : t;
  }

  /** Map a key to a bucket index. */
  CTP_INLINE_CROSS_FUN
  size_type bucket_of(const Key &key) const {
    return hash_fn_(key) % buckets_.size();
  }

  /** Read-lock a bucket (no-op when locking is disabled). Uses RwLock's opt-in
   *  writer_priority mode so a sustained find() stream can't starve insert/erase
   *  on the bucket (see the global_lock_ starvation note above). */
  CTP_INLINE_CROSS_FUN
  void read_lock_bucket(size_type b) {
    if constexpr (EnableLocking)
      locks_[b].ReadLock(0, /*writer_priority=*/true);
  }
  CTP_INLINE_CROSS_FUN
  void read_unlock_bucket(size_type b) {
    if constexpr (EnableLocking) locks_[b].ReadUnlock();
  }
  /** Write-lock a bucket (no-op when locking is disabled). */
  CTP_INLINE_CROSS_FUN
  void write_lock_bucket(size_type b) {
    if constexpr (EnableLocking) locks_[b].WriteLock(0);
  }
  CTP_INLINE_CROSS_FUN
  void write_unlock_bucket(size_type b) {
    if constexpr (EnableLocking) locks_[b].WriteUnlock();
  }

  /** Map-wide read lock — shared; held during every single-key operation so a
   *  concurrent rehash (which needs the write lock) cannot reallocate the
   *  bucket/lock arrays mid-operation. Uses writer_priority so a steady op
   *  stream can't starve the rehash writer. No-op when locking is disabled. */
  CTP_INLINE_CROSS_FUN void global_read_lock() {
    if constexpr (EnableLocking)
      global_lock_.ReadLock(0, /*writer_priority=*/true);
  }
  CTP_INLINE_CROSS_FUN void global_read_unlock() {
    if constexpr (EnableLocking) global_lock_.ReadUnlock();
  }
  /** Map-wide write lock — exclusive; taken only to grow/clear/iterate. Drains
   *  all in-flight single-key ops (their read locks) before proceeding; new
   *  readers back off via RwLock's writer_priority mode (writer preference). */
  CTP_INLINE_CROSS_FUN void global_write_lock() {
    if constexpr (EnableLocking) global_lock_.WriteLock(0);
  }
  CTP_INLINE_CROSS_FUN void global_write_unlock() {
    if constexpr (EnableLocking) global_lock_.WriteUnlock();
  }

  /** Acquire every bucket's write lock. Used for rehash / clear / for_each. */
  CTP_INLINE_CROSS_FUN
  void write_lock_all() {
    if constexpr (EnableLocking) {
      for (size_type i = 0; i < buckets_.size(); ++i) locks_[i].WriteLock(0);
    }
  }
  CTP_INLINE_CROSS_FUN
  void write_unlock_all() {
    if constexpr (EnableLocking) {
      for (size_type i = 0; i < buckets_.size(); ++i) locks_[i].WriteUnlock();
    }
  }

  /** Walk a bucket chain. Caller holds whatever lock is appropriate. */
  CTP_INLINE_CROSS_FUN
  Node *find_in_bucket(size_type b, const Key &key) const {
    Node *cur = buckets_[b];
    while (cur != nullptr) {
      if (key_eq_(cur->key_, key)) return cur;
      cur = cur->next_;
    }
    return nullptr;
  }

  /** Allocate a node via the allocator. */
  CTP_INLINE_CROSS_FUN
  Node *new_node(const Key &k, const T &v, Node *next) {
    auto fp = alloc_->template NewObj<Node>(k, v, next);
    return fp.ptr_;
  }

  /** Free a node via the allocator. */
  CTP_INLINE_CROSS_FUN
  void del_node(Node *n) {
    ctp::ipc::FullPtr<Node> fp(alloc_, n);
    alloc_->template DelObj<Node>(fp);
  }

  /** Set up an empty bucket array. */
  CTP_CROSS_FUN
  void init_buckets(size_type num_buckets) {
    if (num_buckets == 0) num_buckets = 1;
    buckets_.resize(num_buckets);
    for (size_type i = 0; i < num_buckets; ++i) buckets_[i] = nullptr;
    if constexpr (EnableLocking) {
      locks_.resize(num_buckets);
      for (size_type i = 0; i < num_buckets; ++i) locks_[i].Init();
      // The map-wide lock persists across rehash (it guards the rehash), so it
      // is initialized once here and NEVER re-Init()'d by rehash_no_lock().
      global_lock_.Init();
    }
  }

  /** Rehash without taking locks (caller holds every bucket lock). */
  CTP_CROSS_FUN
  void rehash_no_lock(size_type new_bucket_count) {
    if (new_bucket_count == 0) new_bucket_count = 1;
    // Snapshot the current chains and the old bucket head array. We
    // re-thread the existing Node* values into the new buckets so we
    // don't have to re-alloc on rehash.
    size_type old_cap = buckets_.size();
    vector<Node *, AllocT> old_buckets(static_cast<vector<Node *, AllocT>&&>(buckets_));
    buckets_.resize(new_bucket_count);
    for (size_type i = 0; i < new_bucket_count; ++i) buckets_[i] = nullptr;
    if constexpr (EnableLocking) {
      // Re-init lock array to the new bucket count. Existing locks
      // are released; we'll re-acquire after the caller exits the
      // critical section.
      locks_.resize(new_bucket_count);
      for (size_type i = 0; i < new_bucket_count; ++i) locks_[i].Init();
    }
    for (size_type i = 0; i < old_cap; ++i) {
      Node *cur = old_buckets[i];
      while (cur != nullptr) {
        Node *next = cur->next_;
        size_type b = hash_fn_(cur->key_) % new_bucket_count;
        cur->next_ = buckets_[b];
        buckets_[b] = cur;
        cur = next;
      }
    }
  }

  /** Grow the table when size reaches ext_percent_ * bucket_count. Caller must
   *  NOT hold any per-bucket lock OR the global read lock (this takes the
   *  global WRITE lock).
   *
   *  Thread-safe: the global write lock drains every in-flight single-key op
   *  (each holds the global read lock for its duration) before the bucket and
   *  lock arrays are reallocated, and blocks new ops until the rehash
   *  completes. The cheap pre-check is racy but harmless — it is re-evaluated
   *  under the exclusive lock. */
  CTP_INLINE_CROSS_FUN
  void maybe_rehash() {
    size_type cur_size = size_.load();
    size_type cap = buckets_.size();
    if (cur_size < rehash_threshold(cap)) return;
    global_write_lock();
    cur_size = size_.load();
    cap = buckets_.size();
    if (cur_size >= rehash_threshold(cap)) {
      rehash_no_lock(cap * ext_mult_);
    }
    global_write_unlock();
  }

 public:
#if CTP_IS_HOST
  /** Host-side constructor using the global MallocAllocator.
   *  @param ext_percent grow once size reaches this fraction of bucket_count
   *  @param ext_mult    multiply bucket_count by this on each rehash (>= 2) */
  explicit unordered_map_ll(size_type num_buckets = 16,
                            double ext_percent = 0.6, size_type ext_mult = 2)
      : buckets_(CTP_MALLOC), locks_(CTP_MALLOC), size_(0),
        alloc_(CTP_MALLOC), hash_fn_(), key_eq_(),
        ext_percent_(ext_percent), ext_mult_(ext_mult < 2 ? 2 : ext_mult) {
    init_buckets(num_buckets);
  }
  /** Three-arg constructor kept for source compatibility with the
   *  prior open-addressing map -- num_locks is ignored (locking is now
   *  one-per-bucket and toggled by the EnableLocking template arg). */
  CTP_CROSS_FUN
  unordered_map_ll(size_type num_buckets, size_type /*num_locks_ignored*/)
      : buckets_(CTP_MALLOC), locks_(CTP_MALLOC), size_(0),
        alloc_(CTP_MALLOC), hash_fn_(), key_eq_(),
        ext_percent_(0.6), ext_mult_(2) {
    init_buckets(num_buckets);
  }
#endif

  /** Constructor with an explicit allocator. */
  CTP_CROSS_FUN
  explicit unordered_map_ll(AllocT *alloc, size_type num_buckets = 16,
                            double ext_percent = 0.6, size_type ext_mult = 2)
      : buckets_(alloc), locks_(alloc), size_(0),
        alloc_(alloc), hash_fn_(), key_eq_(),
        ext_percent_(ext_percent), ext_mult_(ext_mult < 2 ? 2 : ext_mult) {
    init_buckets(num_buckets);
  }

  /** Source-compatible four-arg constructor; second size_type is ignored. */
  CTP_CROSS_FUN
  unordered_map_ll(AllocT *alloc, size_type num_buckets,
                   size_type /*num_locks_ignored*/)
      : buckets_(alloc), locks_(alloc), size_(0),
        alloc_(alloc), hash_fn_(), key_eq_(),
        ext_percent_(0.6), ext_mult_(2) {
    init_buckets(num_buckets);
  }

  CTP_CROSS_FUN ~unordered_map_ll() {
    // Free every node. We don't bother locking -- destruction is
    // expected to be single-threaded.
    for (size_type i = 0; i < buckets_.size(); ++i) {
      Node *cur = buckets_[i];
      while (cur != nullptr) {
        Node *next = cur->next_;
        del_node(cur);
        cur = next;
      }
      buckets_[i] = nullptr;
    }
  }

  // -- size / capacity -------------------------------------------------

  CTP_INLINE_CROSS_FUN size_type size() const { return size_.load(); }
  CTP_INLINE_CROSS_FUN bool empty() const { return size() == 0; }
  CTP_INLINE_CROSS_FUN size_type bucket_count() const {
    return buckets_.size();
  }

  /** Force a rehash to a new bucket count.
   *
   *  NOT thread-safe (by design): see the THREAD-SAFETY CONTRACT at the
   *  top of this file. The caller must ensure no other thread is
   *  touching the map for the duration of this call -- the bucket and
   *  lock arrays are both reallocated, so concurrent inserts/finds/
   *  erases would race and use freed memory. */
  CTP_CROSS_FUN
  bool rehash(size_type new_bucket_count) {
    global_write_lock();
    rehash_no_lock(new_bucket_count);
    global_write_unlock();
    return true;
  }

  // -- locked variants (caller holds the bucket write lock) -----------

  /** Write-lock the bucket that owns this key. */
  CTP_CROSS_FUN
  void lock_key(const Key &key) {
    write_lock_bucket(bucket_of(key));
  }
  CTP_CROSS_FUN
  void unlock_key(const Key &key) {
    write_unlock_bucket(bucket_of(key));
  }

  /** Find while holding the bucket's write lock. */
  CTP_CROSS_FUN
  T *find_locked(const Key &key) {
    Node *n = find_in_bucket(bucket_of(key), key);
    return n != nullptr ? &n->value_ : nullptr;
  }

  /** Insert while holding the bucket's write lock; returns existing
   *  entry untouched if the key is already present. */
  CTP_CROSS_FUN
  InsertResult<T> insert_locked(const Key &key, const T &value) {
    size_type b = bucket_of(key);
    Node *n = find_in_bucket(b, key);
    if (n != nullptr) return {false, &n->value_};
    Node *fresh = new_node(key, value, buckets_[b]);
    if (fresh == nullptr) return {false, nullptr};
    buckets_[b] = fresh;
    size_.fetch_add(1);
    return {true, &fresh->value_};
  }

  /** Erase while holding the bucket's write lock. */
  CTP_CROSS_FUN
  size_type erase_locked(const Key &key) {
    size_type b = bucket_of(key);
    Node **prev = &buckets_[b];
    Node *cur = *prev;
    while (cur != nullptr) {
      if (key_eq_(cur->key_, key)) {
        *prev = cur->next_;
        del_node(cur);
        size_.fetch_sub(1);
        return 1;
      }
      prev = &cur->next_;
      cur = cur->next_;
    }
    return 0;
  }

  // -- self-locking primary API ---------------------------------------

  /** Insert if absent; thread-safe. */
  CTP_CROSS_FUN
  InsertResult<T> insert(const Key &key, const T &value) {
    global_read_lock();
    size_type b = bucket_of(key);
    write_lock_bucket(b);
    InsertResult<T> r = insert_locked(key, value);
    write_unlock_bucket(b);
    global_read_unlock();
    if (r.inserted) maybe_rehash();
    return r;
  }

  /** Insert if absent, else overwrite. */
  CTP_CROSS_FUN
  InsertResult<T> insert_or_assign(const Key &key, const T &value) {
    global_read_lock();
    size_type b = bucket_of(key);
    write_lock_bucket(b);
    Node *n = find_in_bucket(b, key);
    InsertResult<T> r;
    if (n != nullptr) {
      n->value_ = value;
      r = {false, &n->value_};
    } else {
      Node *fresh = new_node(key, value, buckets_[b]);
      if (fresh == nullptr) {
        r = {false, nullptr};
      } else {
        buckets_[b] = fresh;
        size_.fetch_add(1);
        r = {true, &fresh->value_};
      }
    }
    write_unlock_bucket(b);
    global_read_unlock();
    if (r.inserted) maybe_rehash();
    return r;
  }

  /** operator[] -- creates a default-valued entry if absent. */
  CTP_CROSS_FUN
  T &operator[](const Key &key) {
    global_read_lock();
    size_type b = bucket_of(key);
    write_lock_bucket(b);
    Node *n = find_in_bucket(b, key);
    bool inserted = false;
    if (n == nullptr) {
      Node *fresh = new_node(key, T(), buckets_[b]);
      buckets_[b] = fresh;
      size_.fetch_add(1);
      n = fresh;
      inserted = true;
    }
    // Safe to return the ref after unlocking: rehash only re-threads existing
    // Node objects (never reallocates them), so &n->value_ stays valid.
    T &ref = n->value_;
    write_unlock_bucket(b);
    global_read_unlock();
    if (inserted) maybe_rehash();
    return ref;
  }

  /** Lookup (mutable) returning a pointer or nullptr. WARNING: the returned
   *  pointer is only valid while no other thread erases `key`; for concurrent
   *  find-then-use, prefer get() with a shared_ptr value type. */
  CTP_CROSS_FUN
  T *find(const Key &key) {
    global_read_lock();
    size_type b = bucket_of(key);
    read_lock_bucket(b);
    Node *n = find_in_bucket(b, key);
    T *res = n != nullptr ? &n->value_ : nullptr;
    read_unlock_bucket(b);
    global_read_unlock();
    return res;
  }

  /** Return a COPY of the value bound to `key`, taken WHILE the read lock is
   *  held (or a default-constructed T if absent). When T is a shared_ptr<V>,
   *  the copy is an owning handle that stays valid even if another thread
   *  erases the entry concurrently -- erase just drops the map's reference, and
   *  the pointee lives until the last handle is gone. This is the safe way to
   *  find-then-use a value without an external lock. */
  CTP_CROSS_FUN
  T get(const Key &key) const {
    auto *self = const_cast<unordered_map_ll *>(this);
    self->global_read_lock();
    size_type b = bucket_of(key);
    self->read_lock_bucket(b);
    Node *n = find_in_bucket(b, key);
    T res = (n != nullptr) ? n->value_ : T{};
    self->read_unlock_bucket(b);
    self->global_read_unlock();
    return res;
  }

  /** Lookup (const). */
  CTP_CROSS_FUN
  const T *find(const Key &key) const {
    auto *self = const_cast<unordered_map_ll *>(this);
    self->global_read_lock();
    size_type b = bucket_of(key);
    self->read_lock_bucket(b);
    Node *n = find_in_bucket(b, key);
    const T *res = n != nullptr ? &n->value_ : nullptr;
    self->read_unlock_bucket(b);
    self->global_read_unlock();
    return res;
  }

  CTP_CROSS_FUN bool contains(const Key &key) { return find(key) != nullptr; }
  CTP_CROSS_FUN size_type count(const Key &key) { return contains(key) ? 1 : 0; }

  /** Erase by key. */
  CTP_CROSS_FUN
  size_type erase(const Key &key) {
    // Must hold the global read lock for the whole bucket access, exactly like
    // insert/find/operator[]: it pins the bucket/lock arrays so a concurrent
    // rehash (which takes the global write lock and momentarily empties
    // buckets_ during its move+resize) cannot run underneath us. Without it,
    // bucket_of() can read buckets_.size()==0 mid-rehash and divide by zero
    // (observed as a "Numerical" crash in the concurrent insert/erase/grow
    // stress test on stricter toolchains).
    global_read_lock();
    size_type b = bucket_of(key);
    write_lock_bucket(b);
    size_type r = erase_locked(key);
    write_unlock_bucket(b);
    global_read_unlock();
    return r;
  }

  /** Drop all entries. */
  CTP_CROSS_FUN
  void clear() {
    global_write_lock();
    for (size_type i = 0; i < buckets_.size(); ++i) {
      Node *cur = buckets_[i];
      while (cur != nullptr) {
        Node *next = cur->next_;
        del_node(cur);
        cur = next;
      }
      buckets_[i] = nullptr;
    }
    size_.store(0);
    global_write_unlock();
  }

  /** Lock mode for for_each().
   *  - kExclusive (default): takes the map-wide WRITE lock, so the callback sees
   *    a fully quiescent map -- no concurrent single-key op or rehash runs, and
   *    the callback may safely MUTATE values. Use this when the scan writes, or
   *    needs an atomic snapshot. Cost: it serializes against every reader, so a
   *    hot scan under a steady op stream is heavily contended.
   *  - kShared: takes the map-wide READ lock plus a per-bucket READ lock, so the
   *    scan runs CONCURRENTLY with other readers and single-key ops (a
   *    concurrent erase can't free the node being walked -- it needs the bucket
   *    WRITE lock -- and growth can't reallocate buckets_ -- it needs the map
   *    WRITE lock). Use this ONLY for READ-ONLY callbacks that never mutate a
   *    value or re-enter this map. It avoids the exclusive-lock contention that
   *    wedged DelTag's full-map blob scan under concurrent renames (issue #680).
   */

  /** Apply `fn(key, value)` to every entry under `mode` (see ForEachLock).
   *  The callback MUST NOT re-enter this same map. */
  template <typename Func>
  CTP_CROSS_FUN void for_each(Func fn,
                              ForEachLock mode = ForEachLock::kExclusive) {
    if (mode == ForEachLock::kShared) {
      // Fine-grained shared scan: hold the map-wide lock in READ (shared) mode
      // so a concurrent rehash (which needs it in WRITE mode) can't reallocate
      // buckets_/locks_ mid-scan, and take each bucket's read lock so a
      // concurrent single-key write can't mutate the chain being walked.
      // Crucially we request writer_priority=false here: unlike the single-key
      // read path, this scan must NOT defer to a waiting writer -- that map-wide
      // deferral is what wedged DelTag under concurrent renames (issue #680).
      // Because single-key ops DO use writer_priority, a waiting rehash still
      // drains and runs once this (bounded) scan finishes -- no writer
      // starvation. Callback must be READ-ONLY (no value mutation, no re-entry).
      if constexpr (EnableLocking) global_lock_.ReadLock(0, false);
      const size_type nb = buckets_.size();
      for (size_type i = 0; i < nb; ++i) {
        read_lock_bucket(i);
        for (Node *cur = buckets_[i]; cur != nullptr; cur = cur->next_) {
          fn(cur->key_, cur->value_);
        }
        read_unlock_bucket(i);
      }
      if constexpr (EnableLocking) global_lock_.ReadUnlock();
    } else {
      global_write_lock();
      for (size_type i = 0; i < buckets_.size(); ++i) {
        for (Node *cur = buckets_[i]; cur != nullptr; cur = cur->next_) {
          fn(cur->key_, cur->value_);
        }
      }
      global_write_unlock();
    }
  }

  template <typename Func>
  CTP_CROSS_FUN void for_each(Func fn,
                              ForEachLock mode = ForEachLock::kExclusive) const {
    const_cast<unordered_map_ll *>(this)->for_each(std::move(fn), mode);
  }
};

}  // namespace ctp::priv

#endif  // CTP_DATA_STRUCTURES_PRIV_UNORDERED_MAP_LL_H_
