// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).

//! Self-locking hash maps — port of `clio_ctp/data_structures/priv/unordered_map_ll.h`
//! (chaining, per-bucket `RwLock`) and `.../unordered_map_lhash.h` (open
//! addressing, linear probing, stripe locks).
//!
//! `UnorderedMapLl` is the map the CTE runtime uses everywhere
//! (`unordered_map_ll<std::string, std::shared_ptr<BlobInfo>>` and friends in
//! `core_runtime.h`; `clio::run::unordered_map` in `types.h`). Its thread-safety
//! contract is reproduced verbatim below because the runtime depends on it.
//!
//! # Thread-safety contract (ported from the C++ header)
//!
//! * Single-key ops (`insert`, `insert_or_assign`, `get_or_insert_default`,
//!   `find`, `get`, `contains`, `count`, `erase`) are **self-locking** and safe
//!   to call concurrently from many threads, *including while the table grows*.
//! * Safe growth comes from a map-wide `RwLock`: every single-key op holds it
//!   **shared** for the duration of its bucket access; rehash takes it
//!   **exclusive**, which drains all in-flight ops before the bucket array is
//!   rebuilt. Automatic rehash fires in `maybe_rehash()` once `size` reaches
//!   `ext_percent * bucket_count()` after an insert; `rehash()`, `clear()` and
//!   `for_each(_, Exclusive)` also take the exclusive lock.
//! * Bucket-local concurrency is preserved: under the shared map-wide lock, ops
//!   on **different** buckets run in parallel via the per-bucket `RwLock`s; only
//!   growth serializes everything briefly.
//! * A callback passed to `for_each` / `for_each_mut` / `with_key_locked` MUST
//!   NOT re-enter the same map (the locks are not reentrant — see divergence 11).
//! * Sizing still matters for PERFORMANCE (not safety): construct with
//!   `bucket_count()` near the expected key count to avoid growth churn.
//!
//! # C++ → Rust name mapping
//!
//! | C++ (`ctp::priv`) | Rust (`ctp_ds::unordered_map`) |
//! |---|---|
//! | `unordered_map_ll<Key, T, AllocT, Hash, KeyEqual, EnableLocking>` | `UnorderedMapLl<K, V>` |
//! | `unordered_map_lhash<Key, T, AllocT, Hash, KeyEqual>` | `UnorderedMapLhash<K, V>` |
//! | `InsertResult<T> { inserted, T *value }` | `InsertResult<V> { inserted, value: Option<V> }` |
//! | `ForEachLock::kExclusive` / `kShared` | `ForEachLock::Exclusive` / `Shared` |
//! | `unordered_map_ll(num_buckets = 16, ext_percent = 0.6, ext_mult = 2)` | `UnorderedMapLl::{new, with_buckets, with_growth}` |
//! | `unordered_map_ll(num_buckets, num_locks_ignored)` | `UnorderedMapLl::with_num_locks` |
//! | `unordered_map_lhash(capacity = 16, num_locks = 64)` | `UnorderedMapLhash::{new, with_capacity, with_num_locks}` |
//! | `size()` / `empty()` / `bucket_count()` | `size()` / `is_empty()` / `bucket_count()` |
//! | `rehash(n)` | `rehash(n)` |
//! | `insert(k, v)` | `insert(k, v)` |
//! | `insert_or_assign(k, v)` | `insert_or_assign(k, v)` |
//! | `operator[](k)` | `get_or_insert_default(k)` |
//! | `find(k) -> T*` / `find(k) const -> const T*` | `find(&k) -> Option<V>` (clone) or `with_value(&k, f)` |
//! | `get(k) -> T` (copy under lock) | `get(&k) -> V` |
//! | `contains(k)` / `count(k)` | `contains(&k)` / `count(&k)` |
//! | `erase(k)` | `erase(&k)` |
//! | `clear()` | `clear()` |
//! | `for_each(fn, mode)` | `for_each(f, mode)` (read-only) + `for_each_mut(f)` (exclusive) |
//! | `lock_key` + `find_locked` / `insert_locked` / `erase_locked` + `unlock_key` | `with_key_locked(&k, f)` → `LlLocked::{find, find_mut, insert, erase}` |
//! | lhash `lock_key` + `*_locked` + `unlock_key` | `UnorderedMapLhash::lock()` → `LhashLocked` |
//! | `Node { key_, value_, next_ }` | `Node { key, value, next }`, chain = `Option<Box<Node>>` |
//! | `buckets_` (`vector<Node*>`) + `locks_` (`vector<RwLock>`) | fused `Vec<RwLock<Chain<K, V>>>` |
//! | `global_lock_` (`ctp::RwLock`) | `RwLock<LlTable<K, V>>` (the table *is* the guarded datum) |
//! | `size_` (`ctp::ipc::atomic<size_t>`) | `AtomicUsize` (SeqCst, matching the C++ default) |
//! | `rehash_threshold` / `bucket_of` / `find_in_bucket` / `rehash_no_lock` / `maybe_rehash` | same names (private) |
//! | `Slot { state_, key_, value_ }` + `kEmpty` / `kOccupied` / `kTombstone` | `enum Slot { Empty, Occupied(K, V), Tombstone }` |
//! | `find_slot` / `find_insert_slot` / `insert_no_rehash` / `maybe_rehash_locked` | same names (private) |
//! | lhash `locks_` / `num_locks_` / `stripe_of` / `lock_all` | single `Mutex<LhashTable>`; `num_locks()` reported only (divergence 4) |
//! | `ctp::hash<Key>` | `ctp_types::hash::CtpHash` |
//! | `ctp::equal_to<Key>` | `PartialEq` |
//! | `AllocT` (`ctp::ipc::MallocAllocator`) | the global Rust allocator (divergence 2) |
//!
//! # Semantic divergences (explicit)
//!
//! 1. **Pointers → clones/closures.** C++ `find`/`operator[]`/`InsertResult::value`
//!    hand out `T*` that outlive the lock; the header itself warns the pointer is
//!    only valid "while no other thread erases `key`". Rust cannot express that,
//!    so lookups return `Option<V>`/`V` **cloned while the lock is held** (exactly
//!    what C++ `get()` does and what the header recommends), and reference access
//!    is scoped: `with_value`, `with_value_mut`, `with_key_locked`, `for_each_mut`.
//!    Consequently `insert`/`insert_or_assign`/`get_or_insert_default` require
//!    `V: Clone` and pay one clone per call that C++ does not. This also *fixes*
//!    a latent C++ bug: `unordered_map_lhash::insert_no_lock` returns
//!    `&slots_[idx].value_` after a rehash may have reallocated `slots_`
//!    (dangling); the Rust port returns a clone taken before the rehash.
//! 2. **`AllocT` template parameter is not ported.** Nodes/slots come from the
//!    global Rust allocator, equivalent to the C++ default `MallocAllocator`
//!    (`CTP_MALLOC`) that every ported call site uses. A shared-memory-resident
//!    variant would need `ShmPtr`-based nodes plus an in-segment lock per
//!    MEMORY_DESIGN.md; this file does not attempt it. Allocation failure aborts
//!    in Rust rather than returning `{false, nullptr}`, so `InsertResult::value`
//!    is only `None` on the lhash table-full path (see divergence 8).
//! 3. **`EnableLocking = false` is not ported.** Eliding the locks needs
//!    `UnsafeCell` interior mutability; this module is 100% safe Rust, so the
//!    locks are always taken. Observable single-threaded semantics are identical
//!    — only the C++ zero-lock-overhead *performance* knob is missing.
//! 4. **lhash stripe locking → one table lock.** C++ locks only the key's stripe
//!    and then linear-probes the *whole* slot array, so two threads on different
//!    stripes can touch the same slot — a data race (UB) the C++ tolerates.
//!    Linear probing cannot be made sound with stripe locks, so the port guards
//!    the slot array with a single `Mutex` (C++ `lock_all()` behaviour for every
//!    op). `num_locks` is accepted and reported by `num_locks()` for API parity
//!    but partitions nothing. Ops are therefore *more* atomic than the C++: the
//!    "unlock stripe → `lock_all` → rehash → relock stripe" window in
//!    `insert_no_lock`/`subscript_no_lock`, during which other threads observe a
//!    half-finished insert, does not exist here. `UnorderedMapLl` keeps its full
//!    two-level (map-wide + per-bucket) locking.
//! 5. **`writer_priority` has no Rust counterpart.** C++ passes
//!    `ReadLock(0, writer_priority = true)` on single-key paths (to stop a read
//!    stream starving the rehash writer) and `false` in `for_each(kShared)` (the
//!    issue #680 DelTag wedge). `std::sync::RwLock` exposes no such knob; it does
//!    not indefinitely starve writers on the supported platforms, and the
//!    Shared/Exclusive `for_each` distinction survives as lock *strength*. The
//!    lock order (map-wide before bucket) is unchanged, so `for_each(Shared)`
//!    still cannot deadlock against a waiting rehash.
//! 6. **`ctp::RwLock`/`ctp::Mutex` are not used.** The ported `ctp_thread`
//!    primitives are raw, SHM-capable spin locks with no guarded datum; wrapping
//!    data in them requires `UnsafeCell`. This module uses `std::sync::{RwLock,
//!    Mutex}`, which are parking (not spinning) — better under a `for_each`
//!    callback — and make the "no bucket access without the map-wide lock"
//!    invariant a compile-time property rather than a hand-audited one.
//! 7. **Lock poisoning is ignored** (`PoisonError::into_inner`). C++ has no
//!    poisoning; a panic in a user callback must not brick the map.
//! 8. **Degenerate sizes are clamped, not UB.** C++ `init_buckets` clamps
//!    `num_buckets == 0` to 1, but `unordered_map_lhash(capacity = 0)` leaves
//!    `slots_` empty and `num_locks_ == 0`, so every keyed op divides by zero
//!    (UB). This port applies the ll clamp to lhash too: capacity, bucket count
//!    and any `rehash(0)` become 1. `ext_mult < 2 → 2` is ported as-is.
//! 9. **Float → int conversion saturates.** `rehash_threshold` is
//!    `(ext_percent * cap as f64) as usize`, which in Rust saturates and maps NaN
//!    to 0 (then clamped to 1 by the C++ `t < 1 ? 1 : t`); the C++ `static_cast`
//!    is UB for negative/NaN/overflowing values. A negative or NaN `ext_percent`
//!    therefore yields threshold 1 (grow on every insert) instead of UB.
//! 10. **Shrinking `rehash` still silently drops entries** (C++ parity, kept
//!     deliberately): `unordered_map_lhash::rehash_no_lock` zeroes `size_` and
//!     re-inserts, so entries that do not fit the smaller table are discarded.
//!     `UnorderedMapLl::rehash` re-threads chains and never loses entries, also
//!     matching C++.
//! 11. **`bucket_count()` locks.** C++ reads `buckets_.size()` unsynchronized
//!     (racy but non-blocking); Rust takes the shared lock, so `bucket_count()`
//!     must not be called from inside a `for_each` callback or a `with_key_locked`
//!     /`lock()` guard — it would deadlock. `size()` stays lock-free (atomic).
//! 12. **`lock_key`/`unlock_key` → scoped guards.** The C++ pair is unbalanced by
//!     construction (`unlock_key(other_key)` unlocks the wrong bucket) and
//!     `*_locked` on a key outside the locked bucket silently races. `LlLocked` is
//!     bound to one bucket and asserts the key belongs to it; `LhashLocked` guards
//!     the whole table so any key is valid. Neither rehashes, matching
//!     `insert_locked`.
//! 13. **`for_each` has no default argument**; pass `ForEachLock::Exclusive`
//!     explicitly to get the C++ default. The C++ `kExclusive` overload passes a
//!     mutable `value_` to the callback: that is `for_each_mut` here, while
//!     `for_each` is read-only in both modes (`kShared` is read-only in C++ too).
//! 14. **`Drop` is implemented** for `UnorderedMapLl` (mirroring the C++
//!     destructor's unsynchronized node-freeing loop) purely to tear chains down
//!     *iteratively*: Rust's derived recursive `Box` drop would overflow the stack
//!     on a long chain. This is a heap-resident type, so the MEMORY_DESIGN.md
//!     "no `Drop`" rule (which governs `repr(C)` shared-memory types) does not
//!     apply.
//! 15. **Hash/equality are traits, not functor type parameters.** `Hash`/
//!     `KeyEqual` collapse into `K: CtpHash + PartialEq`; a custom hash means a
//!     newtype key. Bucket index is `key.ctp_hash() % bucket_count` exactly as
//!     C++ `hash_fn_(key) % buckets_.size()`.

use ctp_types::hash::CtpHash;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{
    Mutex, MutexGuard, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard,
};

/// C++ `unordered_map_ll` default `num_buckets`.
const DEFAULT_NUM_BUCKETS: usize = 16;
/// C++ `unordered_map_ll` default `ext_percent`.
const DEFAULT_EXT_PERCENT: f64 = 0.6;
/// C++ `unordered_map_ll` default `ext_mult`.
const DEFAULT_EXT_MULT: usize = 2;
/// C++ `ext_mult_(ext_mult < 2 ? 2 : ext_mult)`.
const MIN_EXT_MULT: usize = 2;
/// C++ `unordered_map_lhash::kDefaultNumLocks`.
const DEFAULT_NUM_LOCKS: usize = 64;
/// C++ `unordered_map_lhash` default `capacity`.
const DEFAULT_CAPACITY: usize = 16;

/// Lock mode for [`UnorderedMapLl::for_each`] (C++ `ctp::priv::ForEachLock`).
///
/// * [`Exclusive`](ForEachLock::Exclusive) — map-wide **write** lock: the
///   callback sees a fully quiescent map (no concurrent single-key op, no
///   rehash). This is the C++ default and the mode [`UnorderedMapLl::for_each_mut`]
///   always uses. Cost: it serializes against every reader.
/// * [`Shared`](ForEachLock::Shared) — map-wide **read** lock plus a per-bucket
///   **read** lock, so the scan runs concurrently with other readers and
///   single-key ops (a concurrent erase cannot free the node being walked — it
///   needs the bucket write lock — and growth cannot rebuild the bucket array —
///   it needs the map-wide write lock). Read-only callbacks only. This is the
///   mode that unwedged DelTag's full-map blob scan under concurrent renames
///   (issue #680).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ForEachLock {
    /// Map-wide write lock (C++ `kExclusive`).
    Exclusive,
    /// Map-wide read lock + per-bucket read lock (C++ `kShared`).
    Shared,
}

impl Default for ForEachLock {
    /// C++ `for_each(Func fn, ForEachLock mode = ForEachLock::kExclusive)`.
    fn default() -> Self {
        Self::Exclusive
    }
}

/// Result of `insert` / `insert_or_assign` (C++ `ctp::priv::InsertResult<T>`).
///
/// `value` is a **clone taken while the lock was held** of the value now bound
/// to the key, replacing the C++ `T *value` (see divergence 1). It is `None`
/// only where the C++ returns `nullptr`: an `UnorderedMapLhash` table with no
/// free slot (divergence 2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InsertResult<V> {
    /// True when a new entry was created (C++ `inserted`).
    pub inserted: bool,
    /// Clone of the stored value (C++ `value`, a pointer into the map).
    pub value: Option<V>,
}

// ---------------------------------------------------------------------------
// unordered_map_ll — chaining, per-bucket RwLock
// ---------------------------------------------------------------------------

/// C++ `unordered_map_ll::Node`.
struct Node<K, V> {
    key: K,
    value: V,
    next: Chain<K, V>,
}

/// One bucket's chain — C++ `Node *` bucket head (`nullptr` = empty).
type Chain<K, V> = Option<Box<Node<K, V>>>;

/// Tear a chain down iteratively (see divergence 14).
fn drop_chain<K, V>(mut cur: Chain<K, V>) {
    while let Some(mut node) = cur {
        cur = node.next.take();
    }
}

/// C++ `find_in_bucket`. Caller holds whatever lock is appropriate.
fn find_in_chain<'a, K: PartialEq, V>(chain: &'a Chain<K, V>, key: &K) -> Option<&'a Node<K, V>> {
    let mut cur = chain.as_deref();
    while let Some(node) = cur {
        if node.key == *key {
            return Some(node);
        }
        cur = node.next.as_deref();
    }
    None
}

/// Mutable `find_in_bucket`.
fn find_in_chain_mut<'a, K: PartialEq, V>(
    chain: &'a mut Chain<K, V>,
    key: &K,
) -> Option<&'a mut Node<K, V>> {
    let mut cur = chain.as_deref_mut();
    while let Some(node) = cur {
        if node.key == *key {
            return Some(node);
        }
        cur = node.next.as_deref_mut();
    }
    None
}

/// C++ `erase_locked`'s chain unlink. Returns 1/0 like the C++.
fn erase_from_chain<K: PartialEq, V>(chain: &mut Chain<K, V>, key: &K) -> usize {
    let mut cur = chain;
    loop {
        match cur {
            None => return 0,
            Some(node) if node.key == *key => {
                // `next` is taken first, so the removed node drops alone.
                let mut node = cur.take().expect("matched arm implies Some");
                *cur = node.next.take();
                return 1;
            }
            Some(_) => {
                cur = &mut cur.as_mut().expect("matched arm implies Some").next;
            }
        }
    }
}

fn read_chain<K, V>(lock: &RwLock<Chain<K, V>>) -> RwLockReadGuard<'_, Chain<K, V>> {
    lock.read().unwrap_or_else(PoisonError::into_inner)
}

fn write_chain<K, V>(lock: &RwLock<Chain<K, V>>) -> RwLockWriteGuard<'_, Chain<K, V>> {
    lock.write().unwrap_or_else(PoisonError::into_inner)
}

/// C++ `buckets_` + `locks_`, fused: the bucket head *is* the guarded datum.
struct LlTable<K, V> {
    buckets: Vec<RwLock<Chain<K, V>>>,
}

/// C++ `init_buckets`: `if (num_buckets == 0) num_buckets = 1;`
fn make_buckets<K, V>(num_buckets: usize) -> Vec<RwLock<Chain<K, V>>> {
    let n = if num_buckets == 0 { 1 } else { num_buckets };
    (0..n).map(|_| RwLock::new(None)).collect()
}

impl<K: CtpHash, V> LlTable<K, V> {
    /// C++ `bucket_of`: `hash_fn_(key) % buckets_.size()`.
    fn bucket_of(&self, key: &K) -> usize {
        (key.ctp_hash() % self.buckets.len() as u64) as usize
    }
}

/// Chaining unordered map with a map-wide `RwLock` plus one `RwLock` per bucket
/// — port of `ctp::priv::unordered_map_ll`.
///
/// See the [module docs](self) for the thread-safety contract and divergences.
///
/// ```
/// use ctp_ds::unordered_map::{ForEachLock, UnorderedMapLl};
///
/// let map = UnorderedMapLl::<u64, u64>::with_buckets(4);
/// assert!(map.insert(7, 42).inserted);
/// assert!(!map.insert(7, 99).inserted); // present: left untouched
/// assert_eq!(map.find(&7), Some(42));
/// assert_eq!(map.insert_or_assign(7, 99).value, Some(99));
/// assert_eq!(map.get(&404), 0); // absent -> V::default()
///
/// let mut seen = 0;
/// map.for_each(|_k, v| seen += *v, ForEachLock::Shared);
/// assert_eq!(seen, 99);
/// assert_eq!(map.erase(&7), 1);
/// assert!(map.is_empty());
/// ```
pub struct UnorderedMapLl<K, V> {
    /// C++ `global_lock_` guarding `buckets_`/`locks_`.
    table: RwLock<LlTable<K, V>>,
    /// C++ `size_`.
    size: AtomicUsize,
    /// C++ `ext_percent_`.
    ext_percent: f64,
    /// C++ `ext_mult_` (already clamped to >= 2).
    ext_mult: usize,
}

impl<K, V> UnorderedMapLl<K, V> {
    /// C++ `unordered_map_ll(num_buckets = 16, ext_percent = 0.6, ext_mult = 2)`.
    pub fn new() -> Self {
        Self::with_growth(DEFAULT_NUM_BUCKETS, DEFAULT_EXT_PERCENT, DEFAULT_EXT_MULT)
    }

    /// C++ `unordered_map_ll(num_buckets)`.
    pub fn with_buckets(num_buckets: usize) -> Self {
        Self::with_growth(num_buckets, DEFAULT_EXT_PERCENT, DEFAULT_EXT_MULT)
    }

    /// C++ `unordered_map_ll(num_buckets, ext_percent, ext_mult)`.
    ///
    /// * `ext_percent` — grow once `size` reaches this fraction of `bucket_count`.
    /// * `ext_mult` — multiply `bucket_count` by this on each rehash; values
    ///   below 2 are clamped to 2, exactly as the C++ ctor does.
    ///
    /// `num_buckets == 0` becomes 1 (C++ `init_buckets`).
    pub fn with_growth(num_buckets: usize, ext_percent: f64, ext_mult: usize) -> Self {
        Self {
            table: RwLock::new(LlTable {
                buckets: make_buckets(num_buckets),
            }),
            size: AtomicUsize::new(0),
            ext_percent,
            ext_mult: if ext_mult < MIN_EXT_MULT {
                MIN_EXT_MULT
            } else {
                ext_mult
            },
        }
    }

    /// C++ `unordered_map_ll(num_buckets, num_locks_ignored)` — the
    /// source-compatible ctor kept for the prior open-addressing map. Locking is
    /// one-per-bucket, so `num_locks` is ignored (as in C++).
    pub fn with_num_locks(num_buckets: usize, _num_locks_ignored: usize) -> Self {
        Self::with_buckets(num_buckets)
    }

    fn read_table(&self) -> RwLockReadGuard<'_, LlTable<K, V>> {
        self.table.read().unwrap_or_else(PoisonError::into_inner)
    }

    fn write_table(&self) -> RwLockWriteGuard<'_, LlTable<K, V>> {
        self.table.write().unwrap_or_else(PoisonError::into_inner)
    }

    /// C++ `size()` — lock-free.
    pub fn size(&self) -> usize {
        self.size.load(Ordering::SeqCst)
    }

    /// C++ `empty()`.
    pub fn is_empty(&self) -> bool {
        self.size() == 0
    }

    /// C++ `bucket_count()`. Takes the map-wide read lock (divergence 11).
    pub fn bucket_count(&self) -> usize {
        self.read_table().buckets.len()
    }

    /// C++ `rehash_threshold(cap)`: `max(1, (size_type)(ext_percent_ * cap))`.
    fn rehash_threshold(&self, cap: usize) -> usize {
        // Rust's float->int `as` saturates and maps NaN to 0 (divergence 9).
        let t = (self.ext_percent * cap as f64) as usize;
        if t < 1 {
            1
        } else {
            t
        }
    }
}

impl<K, V> Default for UnorderedMapLl<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> Drop for UnorderedMapLl<K, V> {
    /// Mirrors the C++ destructor (free every node, unsynchronized — destruction
    /// is single-threaded by contract). Iterative by necessity (divergence 14).
    fn drop(&mut self) {
        let table = self.table.get_mut().unwrap_or_else(PoisonError::into_inner);
        for bucket in table.buckets.iter_mut() {
            let chain = bucket.get_mut().unwrap_or_else(PoisonError::into_inner);
            drop_chain(chain.take());
        }
    }
}

impl<K: CtpHash + PartialEq, V> UnorderedMapLl<K, V> {
    /// C++ `rehash_no_lock` — caller holds the map-wide write lock.
    ///
    /// Existing `Node`s are re-threaded into the new buckets, never reallocated
    /// (the `Box` moves; the pointee does not), exactly like the C++.
    fn rehash_no_lock(table: &mut LlTable<K, V>, new_bucket_count: usize) {
        let n = if new_bucket_count == 0 {
            1
        } else {
            new_bucket_count
        };
        let old = std::mem::replace(&mut table.buckets, make_buckets(n));
        for bucket in old {
            let mut cur = bucket.into_inner().unwrap_or_else(PoisonError::into_inner);
            while let Some(mut node) = cur {
                cur = node.next.take();
                let b = (node.key.ctp_hash() % n as u64) as usize;
                let head = table.buckets[b]
                    .get_mut()
                    .unwrap_or_else(PoisonError::into_inner);
                node.next = head.take();
                *head = Some(node);
            }
        }
    }

    /// C++ `rehash(new_bucket_count)` — takes the map-wide write lock, which
    /// drains every in-flight op first. Always `true` (divergence 2).
    pub fn rehash(&self, new_bucket_count: usize) -> bool {
        let mut table = self.write_table();
        Self::rehash_no_lock(&mut table, new_bucket_count);
        true
    }

    /// C++ `maybe_rehash`. Caller must hold NO lock.
    fn maybe_rehash(&self) {
        // Cheap pre-check (racy in C++, shared-locked here) — re-evaluated under
        // the exclusive lock below.
        {
            let table = self.read_table();
            if self.size() < self.rehash_threshold(table.buckets.len()) {
                return;
            }
        }
        let mut table = self.write_table();
        let cap = table.buckets.len();
        if self.size() >= self.rehash_threshold(cap) {
            Self::rehash_no_lock(&mut table, cap.saturating_mul(self.ext_mult));
        }
    }

    /// C++ `insert(key, value)` — insert if absent; thread-safe.
    pub fn insert(&self, key: K, value: V) -> InsertResult<V>
    where
        V: Clone,
    {
        let result = {
            let table = self.read_table();
            let b = table.bucket_of(&key);
            let mut chain = write_chain(&table.buckets[b]);
            self.insert_into_chain(&mut chain, key, value)
        };
        if result.inserted {
            self.maybe_rehash();
        }
        result
    }

    /// C++ `insert_locked`'s body (no rehash; caller holds the bucket lock).
    fn insert_into_chain(&self, chain: &mut Chain<K, V>, key: K, value: V) -> InsertResult<V>
    where
        V: Clone,
    {
        if let Some(node) = find_in_chain(chain, &key) {
            return InsertResult {
                inserted: false,
                value: Some(node.value.clone()),
            };
        }
        let stored = value.clone();
        let next = chain.take();
        *chain = Some(Box::new(Node { key, value, next }));
        self.size.fetch_add(1, Ordering::SeqCst);
        InsertResult {
            inserted: true,
            value: Some(stored),
        }
    }

    /// C++ `insert_or_assign(key, value)` — insert if absent, else overwrite.
    pub fn insert_or_assign(&self, key: K, value: V) -> InsertResult<V>
    where
        V: Clone,
    {
        let result = {
            let table = self.read_table();
            let b = table.bucket_of(&key);
            let mut chain = write_chain(&table.buckets[b]);
            if let Some(node) = find_in_chain_mut(&mut chain, &key) {
                node.value = value;
                InsertResult {
                    inserted: false,
                    value: Some(node.value.clone()),
                }
            } else {
                let stored = value.clone();
                let next = chain.take();
                *chain = Some(Box::new(Node { key, value, next }));
                self.size.fetch_add(1, Ordering::SeqCst);
                InsertResult {
                    inserted: true,
                    value: Some(stored),
                }
            }
        };
        if result.inserted {
            self.maybe_rehash();
        }
        result
    }

    /// C++ `operator[](key)` — creates a default-valued entry if absent and
    /// returns the bound value (by clone; see divergence 1). Use
    /// [`with_value_mut`](Self::with_value_mut) or
    /// [`with_key_locked`](Self::with_key_locked) to mutate in place.
    pub fn get_or_insert_default(&self, key: K) -> V
    where
        V: Clone + Default,
    {
        let (value, inserted) = {
            let table = self.read_table();
            let b = table.bucket_of(&key);
            let mut chain = write_chain(&table.buckets[b]);
            if let Some(node) = find_in_chain(&chain, &key) {
                (node.value.clone(), false)
            } else {
                let next = chain.take();
                *chain = Some(Box::new(Node {
                    key,
                    value: V::default(),
                    next,
                }));
                self.size.fetch_add(1, Ordering::SeqCst);
                let stored = chain
                    .as_ref()
                    .expect("just inserted")
                    .value
                    .clone();
                (stored, true)
            }
        };
        if inserted {
            self.maybe_rehash();
        }
        value
    }

    /// C++ `find(key)` — returns a clone of the bound value taken while the read
    /// lock is held, or `None` (divergence 1).
    ///
    /// When `V` is an `Arc<T>`, the clone is an owning handle that stays valid
    /// even if another thread erases the entry concurrently — erase just drops
    /// the map's reference. This is the safe way to find-then-use a value, and
    /// mirrors what the C++ header recommends `get()` for.
    pub fn find(&self, key: &K) -> Option<V>
    where
        V: Clone,
    {
        self.with_value(key, |v| v.clone())
    }

    /// C++ `get(key)` — a COPY of the bound value taken while the read lock is
    /// held, or `T{}` if absent.
    pub fn get(&self, key: &K) -> V
    where
        V: Clone + Default,
    {
        self.find(key).unwrap_or_default()
    }

    /// Scoped read access — the Rust rendering of C++ `const T *find(key)`
    /// (divergence 1). `f` runs under the map-wide + bucket read locks and must
    /// not re-enter the map.
    pub fn with_value<R>(&self, key: &K, f: impl FnOnce(&V) -> R) -> Option<R> {
        let table = self.read_table();
        let b = table.bucket_of(key);
        let chain = read_chain(&table.buckets[b]);
        find_in_chain(&chain, key).map(|node| f(&node.value))
    }

    /// Scoped mutable access — the Rust rendering of C++ `T *find(key)`
    /// (divergence 1). `f` runs under the map-wide read lock and the bucket
    /// **write** lock, and must not re-enter the map.
    pub fn with_value_mut<R>(&self, key: &K, f: impl FnOnce(&mut V) -> R) -> Option<R> {
        let table = self.read_table();
        let b = table.bucket_of(key);
        let mut chain = write_chain(&table.buckets[b]);
        find_in_chain_mut(&mut chain, key).map(|node| f(&mut node.value))
    }

    /// C++ `contains(key)`.
    pub fn contains(&self, key: &K) -> bool {
        self.with_value(key, |_| ()).is_some()
    }

    /// C++ `count(key)` — 0 or 1.
    pub fn count(&self, key: &K) -> usize {
        if self.contains(key) {
            1
        } else {
            0
        }
    }

    /// C++ `erase(key)` — returns 1 if an entry was removed, else 0.
    ///
    /// Holds the map-wide read lock for the whole bucket access exactly like
    /// `insert`/`find`: it pins the bucket array so a concurrent rehash cannot
    /// run underneath (the C++ fix for the mid-rehash divide-by-zero).
    pub fn erase(&self, key: &K) -> usize {
        let table = self.read_table();
        let b = table.bucket_of(key);
        let mut chain = write_chain(&table.buckets[b]);
        let removed = erase_from_chain(&mut chain, key);
        if removed == 1 {
            self.size.fetch_sub(1, Ordering::SeqCst);
        }
        removed
    }

    /// C++ `clear()` — drops all entries under the map-wide write lock.
    pub fn clear(&self) {
        let mut table = self.write_table();
        for bucket in table.buckets.iter_mut() {
            let chain = bucket.get_mut().unwrap_or_else(PoisonError::into_inner);
            drop_chain(chain.take());
        }
        self.size.store(0, Ordering::SeqCst);
    }

    /// C++ `for_each(fn, mode)`, read-only. See [`ForEachLock`]. The callback
    /// MUST NOT re-enter this map.
    pub fn for_each<F: FnMut(&K, &V)>(&self, mut f: F, mode: ForEachLock) {
        match mode {
            ForEachLock::Shared => {
                let table = self.read_table();
                for bucket in &table.buckets {
                    let chain = read_chain(bucket);
                    let mut cur = chain.as_deref();
                    while let Some(node) = cur {
                        f(&node.key, &node.value);
                        cur = node.next.as_deref();
                    }
                }
            }
            ForEachLock::Exclusive => {
                let mut table = self.write_table();
                for bucket in table.buckets.iter_mut() {
                    // Map-wide write lock held: no bucket lock needed, as in C++.
                    let chain = bucket.get_mut().unwrap_or_else(PoisonError::into_inner);
                    let mut cur = chain.as_deref();
                    while let Some(node) = cur {
                        f(&node.key, &node.value);
                        cur = node.next.as_deref();
                    }
                }
            }
        }
    }

    /// C++ `for_each(fn, kExclusive)` with a value-mutating callback: takes the
    /// map-wide write lock, so the callback sees a quiescent map. The callback
    /// MUST NOT re-enter this map.
    pub fn for_each_mut<F: FnMut(&K, &mut V)>(&self, mut f: F) {
        let mut table = self.write_table();
        for bucket in table.buckets.iter_mut() {
            let chain = bucket.get_mut().unwrap_or_else(PoisonError::into_inner);
            let mut cur = chain.as_deref_mut();
            while let Some(node) = cur {
                f(&node.key, &mut node.value);
                cur = node.next.as_deref_mut();
            }
        }
    }

    /// C++ `lock_key(key)` … `find_locked`/`insert_locked`/`erase_locked` …
    /// `unlock_key(key)`, as one scoped guard (divergence 12).
    ///
    /// Holds the map-wide read lock and `key`'s bucket **write** lock for the
    /// duration of `f`. No rehash happens (matching `insert_locked`), so a heavy
    /// insert loop through this API should be followed by an explicit
    /// [`rehash`](Self::rehash). `f` must not re-enter the map.
    pub fn with_key_locked<R>(&self, key: &K, f: impl FnOnce(&mut LlLocked<'_, K, V>) -> R) -> R {
        let table = self.read_table();
        let bucket_count = table.buckets.len();
        let bucket = table.bucket_of(key);
        let mut chain = write_chain(&table.buckets[bucket]);
        let mut locked = LlLocked {
            chain: &mut chain,
            size: &self.size,
            bucket,
            bucket_count,
        };
        f(&mut locked)
    }
}

/// The locked bucket handed to [`UnorderedMapLl::with_key_locked`] — the C++
/// `*_locked` family (divergence 12).
///
/// Every method asserts its key belongs to the locked bucket; the C++ silently
/// touches another bucket's chain without its lock instead.
pub struct LlLocked<'a, K, V> {
    chain: &'a mut Chain<K, V>,
    size: &'a AtomicUsize,
    bucket: usize,
    bucket_count: usize,
}

impl<K: CtpHash + PartialEq, V> LlLocked<'_, K, V> {
    fn assert_key(&self, key: &K) {
        let b = (key.ctp_hash() % self.bucket_count as u64) as usize;
        assert_eq!(
            b, self.bucket,
            "LlLocked: key belongs to bucket {b}, but bucket {} is locked",
            self.bucket
        );
    }

    /// C++ `find_locked(key)`.
    pub fn find(&self, key: &K) -> Option<&V> {
        self.assert_key(key);
        find_in_chain(self.chain, key).map(|node| &node.value)
    }

    /// Mutable `find_locked` — C++ hands back a `T*` from the same call.
    pub fn find_mut(&mut self, key: &K) -> Option<&mut V> {
        self.assert_key(key);
        find_in_chain_mut(self.chain, key).map(|node| &mut node.value)
    }

    /// C++ `insert_locked(key, value)` — existing entries are left untouched.
    pub fn insert(&mut self, key: K, value: V) -> InsertResult<V>
    where
        V: Clone,
    {
        self.assert_key(&key);
        if let Some(node) = find_in_chain(self.chain, &key) {
            return InsertResult {
                inserted: false,
                value: Some(node.value.clone()),
            };
        }
        let stored = value.clone();
        let next = self.chain.take();
        *self.chain = Some(Box::new(Node { key, value, next }));
        self.size.fetch_add(1, Ordering::SeqCst);
        InsertResult {
            inserted: true,
            value: Some(stored),
        }
    }

    /// C++ `erase_locked(key)`.
    pub fn erase(&mut self, key: &K) -> usize {
        self.assert_key(key);
        let removed = erase_from_chain(self.chain, key);
        if removed == 1 {
            self.size.fetch_sub(1, Ordering::SeqCst);
        }
        removed
    }

    /// C++ `size()` (the map-wide atomic; readable while locked).
    pub fn size(&self) -> usize {
        self.size.load(Ordering::SeqCst)
    }
}

// ---------------------------------------------------------------------------
// unordered_map_lhash — open addressing, linear probing
// ---------------------------------------------------------------------------

/// C++ `Slot { state_, key_, value_ }` with `kEmpty`/`kOccupied`/`kTombstone`.
///
/// The C++ tombstone assigns `Key()`/`T()` over the dead entry; dropping them
/// here is equivalent (and frees).
enum Slot<K, V> {
    Empty,
    Occupied(K, V),
    Tombstone,
}

/// Outcome of placing an entry — lets `rehash_no_lock` reuse the insert path
/// without requiring `V: Clone` (C++ `insert_no_rehash` returns `InsertResult`).
enum Placement {
    Existing(usize),
    Placed(usize),
    /// C++ `idx >= slots_.size()` → `{false, nullptr}`.
    Full,
}

/// C++ `slots_`.
struct LhashTable<K, V> {
    slots: Vec<Slot<K, V>>,
}

fn make_slots<K, V>(capacity: usize) -> Vec<Slot<K, V>> {
    let n = if capacity == 0 { 1 } else { capacity };
    (0..n).map(|_| Slot::Empty).collect()
}

impl<K: CtpHash + PartialEq, V> LhashTable<K, V> {
    /// C++ `find_slot(key)` — returns `slots_.size()` when not found.
    fn find_slot(&self, key: &K) -> usize {
        let cap = self.slots.len();
        if cap == 0 {
            return cap;
        }
        let h = (key.ctp_hash() % cap as u64) as usize;
        for i in 0..cap {
            let idx = (h + i) % cap;
            match &self.slots[idx] {
                Slot::Empty => return cap,
                Slot::Occupied(k, _) if *k == *key => return idx,
                _ => {}
            }
        }
        cap
    }

    /// C++ `find_insert_slot(key, out_idx, out_existing)`.
    fn find_insert_slot(&self, key: &K) -> (usize, bool) {
        let cap = self.slots.len();
        let h = (key.ctp_hash() % cap as u64) as usize;
        let mut first_avail = cap;
        for i in 0..cap {
            let idx = (h + i) % cap;
            match &self.slots[idx] {
                Slot::Occupied(k, _) if *k == *key => return (idx, true),
                Slot::Occupied(_, _) => {}
                Slot::Tombstone => {
                    if first_avail == cap {
                        first_avail = idx;
                    }
                }
                Slot::Empty => {
                    if first_avail == cap {
                        first_avail = idx;
                    }
                    break;
                }
            }
        }
        (first_avail, false)
    }

    fn value_at(&self, idx: usize) -> &V {
        match &self.slots[idx] {
            Slot::Occupied(_, v) => v,
            _ => panic!("slot {idx} is not occupied"),
        }
    }

    fn value_at_mut(&mut self, idx: usize) -> &mut V {
        match &mut self.slots[idx] {
            Slot::Occupied(_, v) => v,
            _ => panic!("slot {idx} is not occupied"),
        }
    }

    /// Claim a slot for `key` (C++ `insert_no_rehash` without the result).
    fn place_entry(&mut self, key: K, value: V, size: &AtomicUsize) -> Placement {
        let (idx, existing) = self.find_insert_slot(&key);
        if existing {
            return Placement::Existing(idx);
        }
        if idx >= self.slots.len() {
            return Placement::Full;
        }
        self.slots[idx] = Slot::Occupied(key, value);
        size.fetch_add(1, Ordering::SeqCst);
        Placement::Placed(idx)
    }

    /// C++ `insert_no_rehash(key, value)`.
    fn insert_no_rehash(&mut self, key: K, value: V, size: &AtomicUsize) -> InsertResult<V>
    where
        V: Clone,
    {
        match self.place_entry(key, value, size) {
            Placement::Full => InsertResult {
                inserted: false,
                value: None,
            },
            Placement::Existing(idx) => InsertResult {
                inserted: false,
                value: Some(self.value_at(idx).clone()),
            },
            Placement::Placed(idx) => InsertResult {
                inserted: true,
                value: Some(self.value_at(idx).clone()),
            },
        }
    }

    /// C++ `rehash_no_lock(new_cap)` — always `true` here (divergence 2).
    ///
    /// Entries that do not fit a shrinking table are silently dropped, exactly
    /// as in C++ (divergence 10).
    fn rehash_no_lock(&mut self, new_cap: usize, size: &AtomicUsize) -> bool {
        let cap = if new_cap == 0 { 1 } else { new_cap };
        let old = std::mem::replace(&mut self.slots, make_slots(cap));
        size.store(0, Ordering::SeqCst);
        for slot in old {
            if let Slot::Occupied(k, v) = slot {
                let _ = self.place_entry(k, v, size);
            }
        }
        true
    }

    /// C++ `maybe_rehash_locked()` — grow past 75% load.
    fn maybe_rehash_locked(&mut self, size: &AtomicUsize) {
        let cur_size = size.load(Ordering::SeqCst);
        let cap = self.slots.len();
        if cur_size.saturating_mul(4) > cap.saturating_mul(3) {
            self.rehash_no_lock(cap.saturating_mul(2), size);
        }
    }
}

/// Open-addressing (linear-probing) unordered map — port of
/// `ctp::priv::unordered_map_lhash`, the GPU-friendly sibling of
/// [`UnorderedMapLl`].
///
/// CPU code that wants better concurrent insert/erase performance should prefer
/// the chaining [`UnorderedMapLl`]; this map's stripe locking collapses to one
/// table lock in Rust (divergence 4).
///
/// ```
/// use ctp_ds::unordered_map::UnorderedMapLhash;
///
/// let map = UnorderedMapLhash::<u64, u64>::with_capacity(16);
/// assert!(map.insert(1, 10).inserted);
/// assert!(map.insert(17, 170).inserted); // probes past the collision at slot 1
/// assert_eq!(map.erase(&1), 1);          // leaves a tombstone at slot 1
/// assert_eq!(map.find(&17), Some(170));  // still reachable through it
/// assert_eq!(map.size(), 1);
/// ```
pub struct UnorderedMapLhash<K, V> {
    table: Mutex<LhashTable<K, V>>,
    /// C++ `size_`.
    size: AtomicUsize,
    /// C++ `num_locks_` — reported only (divergence 4).
    num_locks: usize,
}

impl<K, V> UnorderedMapLhash<K, V> {
    /// C++ `unordered_map_lhash(capacity = 16, num_locks = 64)`.
    pub fn new() -> Self {
        Self::with_num_locks(DEFAULT_CAPACITY, DEFAULT_NUM_LOCKS)
    }

    /// C++ `unordered_map_lhash(capacity)`.
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_num_locks(capacity, DEFAULT_NUM_LOCKS)
    }

    /// C++ `unordered_map_lhash(capacity, num_locks)`.
    ///
    /// `capacity == 0` becomes 1 (divergence 8). `num_locks` is stored as
    /// `min(num_locks, capacity)` (C++ `init_locks(num_locks < capacity ?
    /// num_locks : capacity)`) but partitions nothing (divergence 4).
    pub fn with_num_locks(capacity: usize, num_locks: usize) -> Self {
        let slots = make_slots(capacity);
        let cap = slots.len();
        Self {
            table: Mutex::new(LhashTable { slots }),
            size: AtomicUsize::new(0),
            num_locks: if num_locks < cap { num_locks } else { cap },
        }
    }

    fn lock_table(&self) -> MutexGuard<'_, LhashTable<K, V>> {
        self.table.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// C++ `size()` — lock-free.
    pub fn size(&self) -> usize {
        self.size.load(Ordering::SeqCst)
    }

    /// C++ `empty()`.
    pub fn is_empty(&self) -> bool {
        self.size() == 0
    }

    /// C++ `bucket_count()` — number of slots. Takes the table lock
    /// (divergence 11).
    pub fn bucket_count(&self) -> usize {
        self.lock_table().slots.len()
    }

    /// C++ `num_locks_`. Reported for parity; it partitions nothing
    /// (divergence 4).
    pub fn num_locks(&self) -> usize {
        self.num_locks
    }
}

impl<K, V> Default for UnorderedMapLhash<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: CtpHash + PartialEq, V> UnorderedMapLhash<K, V> {
    /// C++ `rehash(new_cap)` — re-inserts every occupied entry. Always `true`
    /// (divergence 2); `new_cap == 0` becomes 1 (divergence 8); a shrinking
    /// rehash drops what does not fit (divergence 10).
    pub fn rehash(&self, new_cap: usize) -> bool {
        let mut table = self.lock_table();
        table.rehash_no_lock(new_cap, &self.size)
    }

    /// C++ `insert(key, value)` — insert only if the key is absent.
    pub fn insert(&self, key: K, value: V) -> InsertResult<V>
    where
        V: Clone,
    {
        let mut table = self.lock_table();
        Self::insert_in(&mut table, key, value, &self.size)
    }

    /// C++ `insert_no_lock`.
    fn insert_in(
        table: &mut LhashTable<K, V>,
        key: K,
        value: V,
        size: &AtomicUsize,
    ) -> InsertResult<V>
    where
        V: Clone,
    {
        let (idx, existing) = table.find_insert_slot(&key);
        if existing {
            return InsertResult {
                inserted: false,
                value: Some(table.value_at(idx).clone()),
            };
        }
        if idx >= table.slots.len() {
            // Table full: grow, then retry (C++ drops to `lock_all()` here).
            let grown = table.slots.len().saturating_mul(2);
            table.rehash_no_lock(grown, size);
            return table.insert_no_rehash(key, value, size);
        }
        let stored = value.clone();
        table.slots[idx] = Slot::Occupied(key, value);
        size.fetch_add(1, Ordering::SeqCst);
        // C++ re-checks the load factor and calls maybe_rehash_locked(), which
        // re-checks the identical condition; calling it directly is equivalent.
        table.maybe_rehash_locked(size);
        InsertResult {
            inserted: true,
            value: Some(stored),
        }
    }

    /// C++ `insert_or_assign(key, value)`.
    pub fn insert_or_assign(&self, key: K, value: V) -> InsertResult<V>
    where
        V: Clone,
    {
        let mut table = self.lock_table();
        Self::insert_or_assign_in(&mut table, key, value, &self.size)
    }

    /// C++ `insert_or_assign_no_lock`.
    fn insert_or_assign_in(
        table: &mut LhashTable<K, V>,
        key: K,
        value: V,
        size: &AtomicUsize,
    ) -> InsertResult<V>
    where
        V: Clone,
    {
        let (idx, existing) = table.find_insert_slot(&key);
        if existing {
            let slot = table.value_at_mut(idx);
            *slot = value;
            return InsertResult {
                inserted: false,
                value: Some(slot.clone()),
            };
        }
        if idx >= table.slots.len() {
            let grown = table.slots.len().saturating_mul(2);
            table.rehash_no_lock(grown, size);
            return table.insert_no_rehash(key, value, size);
        }
        let stored = value.clone();
        table.slots[idx] = Slot::Occupied(key, value);
        size.fetch_add(1, Ordering::SeqCst);
        table.maybe_rehash_locked(size);
        InsertResult {
            inserted: true,
            value: Some(stored),
        }
    }

    /// C++ `operator[](key)` (`subscript_no_lock`) — creates a default-valued
    /// entry if absent; returns the bound value by clone (divergence 1).
    pub fn get_or_insert_default(&self, key: K) -> V
    where
        V: Clone + Default,
    {
        let mut table = self.lock_table();
        let size = &self.size;
        let (mut idx, existing) = table.find_insert_slot(&key);
        if existing {
            return table.value_at(idx).clone();
        }
        if idx >= table.slots.len() {
            let grown = table.slots.len().saturating_mul(2);
            table.rehash_no_lock(grown, size);
            let (idx2, existing2) = table.find_insert_slot(&key);
            if existing2 {
                return table.value_at(idx2).clone();
            }
            // C++ omits the bounds re-check here (an out-of-bounds write if the
            // grown table were still full); after doubling it never is.
            idx = idx2;
        }
        table.slots[idx] = Slot::Occupied(key, V::default());
        size.fetch_add(1, Ordering::SeqCst);
        // Cloned before the rehash may move the slot — the C++ returns a
        // reference that the rehash can dangle (divergence 1).
        let stored = table.value_at(idx).clone();
        table.maybe_rehash_locked(size);
        stored
    }

    /// C++ `find(key)` — clone of the bound value taken under the lock
    /// (divergence 1).
    pub fn find(&self, key: &K) -> Option<V>
    where
        V: Clone,
    {
        self.with_value(key, |v| v.clone())
    }

    /// Scoped read access — the Rust rendering of C++ `find(key) -> T*`.
    pub fn with_value<R>(&self, key: &K, f: impl FnOnce(&V) -> R) -> Option<R> {
        let table = self.lock_table();
        let idx = table.find_slot(key);
        if idx < table.slots.len() {
            Some(f(table.value_at(idx)))
        } else {
            None
        }
    }

    /// Scoped mutable access — the Rust rendering of C++ `find(key) -> T*` used
    /// for mutation.
    pub fn with_value_mut<R>(&self, key: &K, f: impl FnOnce(&mut V) -> R) -> Option<R> {
        let mut table = self.lock_table();
        let idx = table.find_slot(key);
        if idx < table.slots.len() {
            Some(f(table.value_at_mut(idx)))
        } else {
            None
        }
    }

    /// C++ `contains(key)`.
    pub fn contains(&self, key: &K) -> bool {
        self.with_value(key, |_| ()).is_some()
    }

    /// C++ `count(key)` — 0 or 1.
    pub fn count(&self, key: &K) -> usize {
        if self.contains(key) {
            1
        } else {
            0
        }
    }

    /// C++ `erase(key)` — leaves a tombstone so probe chains stay intact.
    pub fn erase(&self, key: &K) -> usize {
        let mut table = self.lock_table();
        Self::erase_in(&mut table, key, &self.size)
    }

    fn erase_in(table: &mut LhashTable<K, V>, key: &K, size: &AtomicUsize) -> usize {
        let idx = table.find_slot(key);
        if idx < table.slots.len() {
            table.slots[idx] = Slot::Tombstone;
            size.fetch_sub(1, Ordering::SeqCst);
            1
        } else {
            0
        }
    }

    /// C++ `clear()` — every slot returns to `kEmpty` (tombstones included).
    pub fn clear(&self) {
        let mut table = self.lock_table();
        for slot in table.slots.iter_mut() {
            *slot = Slot::Empty;
        }
        self.size.store(0, Ordering::SeqCst);
    }

    /// C++ `for_each(fn)` — every occupied entry, under the table lock. The
    /// callback MUST NOT re-enter this map.
    pub fn for_each<F: FnMut(&K, &V)>(&self, mut f: F) {
        let table = self.lock_table();
        for slot in &table.slots {
            if let Slot::Occupied(k, v) = slot {
                f(k, v);
            }
        }
    }

    /// C++ `for_each(fn)` with a value-mutating callback.
    pub fn for_each_mut<F: FnMut(&K, &mut V)>(&self, mut f: F) {
        let mut table = self.lock_table();
        for slot in table.slots.iter_mut() {
            if let Slot::Occupied(k, v) = slot {
                f(k, v);
            }
        }
    }

    /// C++ `lock_key(key)` … `*_locked` … `unlock_key(key)`, as one scoped guard
    /// (divergence 12). Because the port has a single table lock, the guard
    /// covers every key, and `lock_all()`-style scans are just this guard.
    pub fn lock(&self) -> LhashLocked<'_, K, V> {
        LhashLocked {
            table: self.lock_table(),
            size: &self.size,
        }
    }
}

/// The locked table returned by [`UnorderedMapLhash::lock`] — the C++
/// `find_locked` / `insert_locked` / `erase_locked` family (divergence 12).
pub struct LhashLocked<'a, K, V> {
    table: MutexGuard<'a, LhashTable<K, V>>,
    size: &'a AtomicUsize,
}

impl<K: CtpHash + PartialEq, V> LhashLocked<'_, K, V> {
    /// C++ `find_locked(key)`.
    pub fn find(&self, key: &K) -> Option<&V> {
        let idx = self.table.find_slot(key);
        if idx < self.table.slots.len() {
            Some(self.table.value_at(idx))
        } else {
            None
        }
    }

    /// Mutable `find_locked` — C++ hands back a `T*` from the same call.
    pub fn find_mut(&mut self, key: &K) -> Option<&mut V> {
        let idx = self.table.find_slot(key);
        if idx < self.table.slots.len() {
            Some(self.table.value_at_mut(idx))
        } else {
            None
        }
    }

    /// C++ `insert_locked(key, value)`.
    pub fn insert(&mut self, key: K, value: V) -> InsertResult<V>
    where
        V: Clone,
    {
        UnorderedMapLhash::insert_in(&mut self.table, key, value, self.size)
    }

    /// C++ `insert_or_assign` under the caller's lock.
    pub fn insert_or_assign(&mut self, key: K, value: V) -> InsertResult<V>
    where
        V: Clone,
    {
        UnorderedMapLhash::insert_or_assign_in(&mut self.table, key, value, self.size)
    }

    /// C++ `erase_locked(key)`.
    pub fn erase(&mut self, key: &K) -> usize {
        UnorderedMapLhash::erase_in(&mut self.table, key, self.size)
    }

    /// C++ `bucket_count()` — readable while locked (unlike
    /// [`UnorderedMapLhash::bucket_count`], divergence 11).
    pub fn bucket_count(&self) -> usize {
        self.table.slots.len()
    }

    /// C++ `size()`.
    pub fn size(&self) -> usize {
        self.size.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicU64};
    use std::sync::Arc;

    // -----------------------------------------------------------------------
    // unordered_map_ll — construction / edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn ll_empty_map_has_cpp_defaults() {
        let map = UnorderedMapLl::<u64, u64>::new();
        assert_eq!(map.size(), 0);
        assert!(map.is_empty());
        assert_eq!(map.bucket_count(), DEFAULT_NUM_BUCKETS); // C++ num_buckets = 16
        assert_eq!(map.find(&7), None);
        assert_eq!(map.get(&7), 0); // C++ get() -> T{} when absent
        assert!(!map.contains(&7));
        assert_eq!(map.count(&7), 0);
        assert_eq!(map.erase(&7), 0);
        assert_eq!(map.with_value(&7, |v| *v), None);
        assert_eq!(map.with_value_mut(&7, |v| *v), None);
        // for_each over an empty map visits nothing in either mode.
        let mut seen = 0;
        map.for_each(|_, _| seen += 1, ForEachLock::Exclusive);
        map.for_each(|_, _| seen += 1, ForEachLock::Shared);
        map.for_each_mut(|_, _| seen += 1);
        assert_eq!(seen, 0);
        map.clear(); // clearing an empty map is a no-op
        assert!(map.is_empty());
    }

    #[test]
    fn ll_zero_buckets_clamps_to_one() {
        // C++ init_buckets: `if (num_buckets == 0) num_buckets = 1;`
        let map = UnorderedMapLl::<u64, u64>::with_growth(0, 1e9, 2);
        assert_eq!(map.bucket_count(), 1);
        assert!(map.insert(5, 50).inserted);
        assert_eq!(map.find(&5), Some(50));
    }

    #[test]
    fn ll_num_locks_ctor_arg_is_ignored() {
        // C++ unordered_map_ll(num_buckets, num_locks_ignored).
        let map = UnorderedMapLl::<u64, u64>::with_num_locks(8, 999);
        assert_eq!(map.bucket_count(), 8);
        assert!(map.insert(1, 1).inserted);
    }

    #[test]
    fn ll_default_is_new() {
        let map = UnorderedMapLl::<u64, u64>::default();
        assert_eq!(map.bucket_count(), DEFAULT_NUM_BUCKETS);
    }

    // -----------------------------------------------------------------------
    // unordered_map_ll — single-key semantics
    // -----------------------------------------------------------------------

    #[test]
    fn ll_insert_leaves_existing_entry_untouched() {
        let map = UnorderedMapLl::<u64, u64>::with_buckets(16);
        let r = map.insert(7, 1);
        assert!(r.inserted);
        assert_eq!(r.value, Some(1));
        // C++ insert(): returns the *existing* entry, does not overwrite.
        let r = map.insert(7, 2);
        assert!(!r.inserted);
        assert_eq!(r.value, Some(1));
        assert_eq!(map.find(&7), Some(1));
        assert_eq!(map.size(), 1);
    }

    #[test]
    fn ll_insert_or_assign_overwrites() {
        let map = UnorderedMapLl::<u64, u64>::with_buckets(16);
        let r = map.insert_or_assign(7, 1);
        assert!(r.inserted);
        assert_eq!(r.value, Some(1));
        let r = map.insert_or_assign(7, 2);
        assert!(!r.inserted); // C++: overwrite is not an insert
        assert_eq!(r.value, Some(2));
        assert_eq!(map.find(&7), Some(2));
        assert_eq!(map.size(), 1); // no duplicate entry
    }

    #[test]
    fn ll_get_or_insert_default_creates_entry() {
        let map = UnorderedMapLl::<u64, u64>::with_buckets(16);
        assert_eq!(map.get_or_insert_default(7), 0); // C++ operator[]: T()
        assert_eq!(map.size(), 1);
        assert!(map.contains(&7));
        map.with_value_mut(&7, |v| *v = 9);
        assert_eq!(map.get_or_insert_default(7), 9); // existing entry kept
        assert_eq!(map.size(), 1);
    }

    #[test]
    fn ll_get_vs_find_on_absent_key() {
        let map = UnorderedMapLl::<u64, u64>::with_buckets(4);
        map.insert(1, 11);
        assert_eq!(map.get(&1), 11);
        assert_eq!(map.get(&2), 0); // C++ get(): default-constructed T
        assert_eq!(map.find(&2), None); // C++ find(): nullptr
    }

    #[test]
    fn ll_contains_and_count() {
        let map = UnorderedMapLl::<u64, u64>::with_buckets(4);
        map.insert(1, 11);
        assert!(map.contains(&1));
        assert_eq!(map.count(&1), 1); // C++ count(): 0 or 1 only
        assert!(!map.contains(&2));
        assert_eq!(map.count(&2), 0);
    }

    #[test]
    fn ll_erase_at_every_chain_position() {
        // 4 buckets + a threshold so high nothing rehashes => keys 0/4/8/12 all
        // land in bucket 0, exercising head/middle/tail unlink.
        let map = UnorderedMapLl::<u64, u64>::with_growth(4, 1e9, 2);
        for k in [0u64, 4, 8, 12] {
            assert!(map.insert(k, k * 10).inserted);
        }
        assert_eq!(map.bucket_count(), 4); // never grew
        assert_eq!(map.size(), 4);

        assert_eq!(map.erase(&8), 1); // middle (chain is 12 -> 8 -> 4 -> 0)
        assert_eq!(map.size(), 3);
        assert_eq!(map.erase(&8), 0); // already gone
        assert_eq!(map.erase(&12), 1); // head
        assert_eq!(map.erase(&0), 1); // tail
        assert_eq!(map.size(), 1);
        assert_eq!(map.find(&4), Some(40)); // survivor intact
        assert_eq!(map.erase(&4), 1);
        assert!(map.is_empty());
        assert_eq!(map.erase(&4), 0); // erase from an empty chain
    }

    #[test]
    fn ll_erase_missing_key_from_populated_bucket() {
        let map = UnorderedMapLl::<u64, u64>::with_growth(4, 1e9, 2);
        map.insert(0, 1);
        map.insert(4, 2); // same bucket
        assert_eq!(map.erase(&8), 0); // walks the whole chain, finds nothing
        assert_eq!(map.size(), 2);
    }

    #[test]
    fn ll_clear_drops_all_entries() {
        let map = UnorderedMapLl::<u64, u64>::with_buckets(4);
        for k in 0..50u64 {
            map.insert(k, k);
        }
        let buckets_before = map.bucket_count();
        map.clear();
        assert_eq!(map.size(), 0);
        assert!(map.is_empty());
        assert_eq!(map.find(&10), None);
        // C++ clear() does not shrink the bucket array.
        assert_eq!(map.bucket_count(), buckets_before);
        map.insert(1, 1); // still usable
        assert_eq!(map.size(), 1);
    }

    // -----------------------------------------------------------------------
    // unordered_map_ll — growth policy
    // -----------------------------------------------------------------------

    #[test]
    fn ll_rehash_threshold_matches_cpp_truncation() {
        // C++: threshold = (size_type)(0.6 * 16) = 9 (truncating), grow at >= 9.
        let map = UnorderedMapLl::<u64, u64>::with_growth(16, 0.6, 2);
        for k in 1..=8u64 {
            map.insert(k, k);
        }
        assert_eq!(map.bucket_count(), 16, "must not grow below the threshold");
        map.insert(9, 9);
        assert_eq!(map.bucket_count(), 32, "size 9 >= threshold 9 => x2");
        assert_eq!(map.size(), 9);
        for k in 1..=9u64 {
            assert_eq!(map.find(&k), Some(k), "entry survived growth");
        }
    }

    #[test]
    fn ll_ext_mult_below_two_is_clamped() {
        // C++ ctor: ext_mult_(ext_mult < 2 ? 2 : ext_mult). Without the clamp a
        // multiplier of 0/1 could never grow the table.
        for mult in [0usize, 1] {
            let map = UnorderedMapLl::<u64, u64>::with_growth(4, 0.5, mult);
            map.insert(1, 1); // size 1 < threshold 2
            assert_eq!(map.bucket_count(), 4);
            map.insert(2, 2); // size 2 >= threshold 2 => grow
            assert_eq!(map.bucket_count(), 8, "ext_mult {mult} must clamp to 2");
        }
    }

    #[test]
    fn ll_large_ext_mult_is_honored() {
        let map = UnorderedMapLl::<u64, u64>::with_growth(4, 0.5, 8);
        map.insert(1, 1);
        map.insert(2, 2); // threshold 2 reached
        assert_eq!(map.bucket_count(), 32); // 4 * 8
    }

    #[test]
    fn ll_zero_ext_percent_grows_on_first_insert() {
        // threshold = max(1, (size_type)(0.0 * 16)) = 1.
        let map = UnorderedMapLl::<u64, u64>::with_growth(16, 0.0, 2);
        map.insert(1, 1);
        assert_eq!(map.bucket_count(), 32);
    }

    #[test]
    fn ll_negative_or_nan_ext_percent_saturates_to_threshold_one() {
        // Divergence 9: C++ static_cast<size_t>(-16.0) is UB; Rust `as`
        // saturates to 0, which the `t < 1 ? 1 : t` clamp lifts to 1.
        for pct in [-1.0f64, f64::NAN, f64::NEG_INFINITY] {
            let map = UnorderedMapLl::<u64, u64>::with_growth(16, pct, 2);
            map.insert(1, 1);
            assert_eq!(map.bucket_count(), 32, "ext_percent {pct} => threshold 1");
            assert_eq!(map.find(&1), Some(1));
        }
    }

    #[test]
    fn ll_huge_ext_percent_never_grows() {
        // threshold saturates at usize::MAX rather than overflowing.
        let map = UnorderedMapLl::<u64, u64>::with_growth(4, f64::INFINITY, 2);
        for k in 0..100u64 {
            map.insert(k, k);
        }
        assert_eq!(map.bucket_count(), 4);
        assert_eq!(map.size(), 100);
        assert_eq!(map.find(&99), Some(99));
    }

    #[test]
    fn ll_grows_and_keeps_all_entries() {
        // Mirrors the C++ "umap_ll: single-thread grows and keeps all entries".
        const N: u64 = 4000;
        let map = UnorderedMapLl::<u64, u64>::with_buckets(4);
        for i in 0..N {
            assert!(map.insert(i, i * 3 + 7).inserted);
        }
        assert_eq!(map.size(), N as usize);
        assert!(map.bucket_count() > 4, "it actually grew");
        for i in 0..N {
            assert_eq!(map.find(&i), Some(i * 3 + 7));
        }
        for i in (0..N).step_by(2) {
            assert_eq!(map.erase(&i), 1);
        }
        assert_eq!(map.size(), (N / 2) as usize);
        for i in (1..N).step_by(2) {
            assert_eq!(map.find(&i), Some(i * 3 + 7));
        }
        for i in (0..N).step_by(2) {
            assert_eq!(map.find(&i), None);
        }
    }

    #[test]
    fn ll_explicit_rehash_preserves_entries() {
        let map = UnorderedMapLl::<u64, u64>::with_growth(16, 1e9, 2);
        for k in 0..64u64 {
            map.insert(k, k * 2);
        }
        assert!(map.rehash(128));
        assert_eq!(map.bucket_count(), 128);
        assert_eq!(map.size(), 64);
        // Shrinking a chaining map never loses entries (unlike lhash).
        assert!(map.rehash(2));
        assert_eq!(map.bucket_count(), 2);
        assert_eq!(map.size(), 64);
        for k in 0..64u64 {
            assert_eq!(map.find(&k), Some(k * 2));
        }
    }

    #[test]
    fn ll_rehash_to_zero_clamps_to_one_bucket() {
        let map = UnorderedMapLl::<u64, u64>::with_growth(8, 1e9, 2);
        map.insert(1, 1);
        map.insert(2, 2);
        assert!(map.rehash(0)); // C++ rehash_no_lock: 0 -> 1
        assert_eq!(map.bucket_count(), 1);
        assert_eq!(map.size(), 2);
        assert_eq!(map.find(&1), Some(1));
        assert_eq!(map.find(&2), Some(2));
    }

    // -----------------------------------------------------------------------
    // unordered_map_ll — for_each / scoped access / locked guard
    // -----------------------------------------------------------------------

    #[test]
    fn ll_for_each_visits_every_entry_in_both_modes() {
        let map = UnorderedMapLl::<u64, u64>::with_buckets(4);
        for k in 0..40u64 {
            map.insert(k, k * 10);
        }
        for mode in [ForEachLock::Exclusive, ForEachLock::Shared] {
            let mut keys: Vec<u64> = Vec::new();
            let mut sum = 0u64;
            map.for_each(
                |k, v| {
                    keys.push(*k);
                    sum += *v;
                },
                mode,
            );
            keys.sort_unstable();
            assert_eq!(keys, (0..40).collect::<Vec<u64>>(), "mode {mode:?}");
            assert_eq!(sum, (0..40u64).map(|k| k * 10).sum::<u64>());
        }
        assert_eq!(ForEachLock::default(), ForEachLock::Exclusive);
    }

    #[test]
    fn ll_for_each_mut_mutates_values() {
        let map = UnorderedMapLl::<u64, u64>::with_buckets(4);
        for k in 0..20u64 {
            map.insert(k, k);
        }
        map.for_each_mut(|_, v| *v += 100);
        for k in 0..20u64 {
            assert_eq!(map.find(&k), Some(k + 100));
        }
    }

    #[test]
    fn ll_with_value_and_with_value_mut() {
        let map = UnorderedMapLl::<String, Vec<u64>>::with_buckets(4);
        map.insert("k".into(), vec![1, 2]);
        // Scoped read: no clone of the (non-Copy) value needed.
        assert_eq!(map.with_value(&"k".to_string(), |v| v.len()), Some(2));
        map.with_value_mut(&"k".to_string(), |v| v.push(3));
        assert_eq!(map.find(&"k".to_string()), Some(vec![1, 2, 3]));
        assert_eq!(map.with_value(&"nope".to_string(), |v| v.len()), None);
    }

    #[test]
    fn ll_with_key_locked_find_insert_erase() {
        let map = UnorderedMapLl::<u64, u64>::with_buckets(8);
        // C++: lock_key -> insert_locked -> unlock_key.
        let r = map.with_key_locked(&3, |l| l.insert(3, 30));
        assert!(r.inserted);
        assert_eq!(map.size(), 1);
        map.with_key_locked(&3, |l| {
            assert_eq!(l.find(&3), Some(&30));
            assert_eq!(l.size(), 1);
            // insert_locked leaves an existing entry untouched
            let r = l.insert(3, 99);
            assert!(!r.inserted);
            assert_eq!(r.value, Some(30));
            *l.find_mut(&3).unwrap() = 31;
            assert_eq!(l.find(&3), Some(&31));
        });
        assert_eq!(map.find(&3), Some(31));
        assert_eq!(map.with_key_locked(&3, |l| l.erase(&3)), 1);
        assert_eq!(map.size(), 0);
        assert_eq!(map.with_key_locked(&3, |l| l.erase(&3)), 0);
        assert_eq!(map.with_key_locked(&3, |l| l.find(&3).copied()), None);
    }

    #[test]
    #[should_panic(expected = "bucket 0 is locked")]
    fn ll_locked_guard_rejects_a_foreign_key() {
        // Divergence 12: C++ would silently touch bucket 1's chain while only
        // bucket 0 is locked.
        let map = UnorderedMapLl::<u64, u64>::with_buckets(4);
        map.with_key_locked(&0, |l| {
            let _ = l.find(&1);
        });
    }

    #[test]
    fn ll_arc_value_survives_concurrent_erase() {
        // The C++ get() contract: with T = shared_ptr<V> the returned copy is an
        // owning handle that stays valid even if another thread erases the entry.
        let map = UnorderedMapLl::<String, Arc<u64>>::with_buckets(4);
        map.insert("blob".into(), Arc::new(42));
        let handle = map.find(&"blob".to_string()).expect("present");
        assert_eq!(map.erase(&"blob".to_string()), 1);
        assert_eq!(*handle, 42); // pointee outlives the map entry
        assert_eq!(Arc::strong_count(&handle), 1);
        assert_eq!(map.find(&"blob".to_string()), None);
    }

    #[test]
    fn ll_string_keys_hash_via_ctp_hash() {
        // Mirrors CTE's unordered_map_ll<std::string, shared_ptr<BlobInfo>>.
        let map = UnorderedMapLl::<String, u64>::with_buckets(4);
        for i in 0..100u64 {
            assert!(map.insert(format!("/blob/{i}"), i).inserted);
        }
        assert_eq!(map.size(), 100);
        for i in 0..100u64 {
            assert_eq!(map.find(&format!("/blob/{i}")), Some(i));
        }
        assert_eq!(map.find(&"/blob/nope".to_string()), None);
        assert_eq!(map.erase(&"/blob/50".to_string()), 1);
        assert_eq!(map.find(&"/blob/50".to_string()), None);
    }

    #[test]
    fn ll_long_single_bucket_chain_tears_down_iteratively() {
        // Divergence 14: a recursive Box drop would risk a stack overflow. Every
        // key lands in bucket 0 and the table never grows.
        const N: u64 = 4000;
        let map = UnorderedMapLl::<u64, u64>::with_growth(1, 1e9, 2);
        for k in 0..N {
            map.insert(k, k);
        }
        assert_eq!(map.bucket_count(), 1);
        assert_eq!(map.size(), N as usize);
        assert_eq!(map.find(&(N - 1)), Some(N - 1));
        map.clear(); // iterative drop via clear()
        assert!(map.is_empty());
        for k in 0..N {
            map.insert(k, k);
        }
        drop(map); // iterative drop via Drop
    }

    // -----------------------------------------------------------------------
    // unordered_map_ll — concurrency
    // -----------------------------------------------------------------------

    #[test]
    fn ll_concurrent_inserts_survive_concurrent_growth() {
        // Mirrors "umap_ll: concurrent inserts survive concurrent growth": start
        // tiny so growth races the inserts.
        const THREADS: u64 = 8;
        const PER: u64 = 500;
        let map = UnorderedMapLl::<u64, u64>::with_buckets(4);
        std::thread::scope(|s| {
            for t in 0..THREADS {
                let map = &map;
                s.spawn(move || {
                    for i in 0..PER {
                        let k = t * PER + i;
                        assert!(map.insert(k, k ^ 0xA5A5_A5A5).inserted);
                    }
                });
            }
        });
        assert_eq!(map.size(), (THREADS * PER) as usize);
        assert!(map.bucket_count() > 4);
        for k in 0..THREADS * PER {
            assert_eq!(map.find(&k), Some(k ^ 0xA5A5_A5A5));
        }
    }

    #[test]
    fn ll_concurrent_find_during_expansion_is_consistent() {
        // Mirrors "umap_ll: concurrent find during expansion is consistent".
        const KEYS: u64 = 2000;
        let value_of = |k: u64| k.wrapping_mul(2_654_435_761);
        const SEED: u64 = 8;
        let map = Arc::new(UnorderedMapLl::<u64, u64>::with_buckets(4));
        let bad = Arc::new(AtomicU64::new(0));
        let reads = Arc::new(AtomicU64::new(0));
        let done = Arc::new(AtomicBool::new(false));

        // Seeded before the readers start, so every reader's guaranteed first
        // pass observes live entries no matter how the threads interleave —
        // otherwise `reads > 0` is a race on the writer being slower.
        for k in 0..SEED {
            map.insert(k, value_of(k));
        }

        std::thread::scope(|s| {
            // Readers race the writer's growth.
            for _ in 0..4 {
                let (map, bad, reads, done) = (
                    Arc::clone(&map),
                    Arc::clone(&bad),
                    Arc::clone(&reads),
                    Arc::clone(&done),
                );
                s.spawn(move || {
                    // do-while: always complete at least one full pass.
                    loop {
                        for k in 0..KEYS {
                            if let Some(v) = map.find(&k) {
                                reads.fetch_add(1, Ordering::Relaxed);
                                if v != value_of(k) {
                                    bad.fetch_add(1, Ordering::Relaxed);
                                }
                            }
                        }
                        if done.load(Ordering::SeqCst) {
                            break;
                        }
                    }
                });
            }
            for k in SEED..KEYS {
                map.insert(k, value_of(k));
            }
            done.store(true, Ordering::SeqCst);
        });

        assert_eq!(bad.load(Ordering::SeqCst), 0, "never observed a corrupt value");
        assert!(reads.load(Ordering::SeqCst) > 0, "readers found live entries");
        assert_eq!(map.size(), KEYS as usize);
        for k in 0..KEYS {
            assert_eq!(map.find(&k), Some(value_of(k)));
        }
    }

    #[test]
    fn ll_concurrent_insert_erase_find_churn_with_growth() {
        // Mirrors "umap_ll: concurrent insert/erase/find churn with growth":
        // each thread owns a disjoint key range, so results are deterministic.
        const THREADS: u64 = 8;
        const PER: u64 = 400;
        let map = UnorderedMapLl::<u64, u64>::with_buckets(4);
        std::thread::scope(|s| {
            for t in 0..THREADS {
                let map = &map;
                s.spawn(move || {
                    let base = t * PER;
                    for i in 0..PER {
                        let k = base + i;
                        assert!(map.insert(k, base + i).inserted);
                        assert_eq!(map.find(&k), Some(base + i));
                        assert_eq!(map.count(&k), 1);
                        if i % 2 == 0 {
                            assert_eq!(map.erase(&k), 1);
                        }
                        // A key no thread ever inserts stays absent throughout.
                        assert_eq!(map.find(&u64::MAX), None);
                    }
                });
            }
        });
        assert_eq!(map.size(), (THREADS * PER / 2) as usize);
        for t in 0..THREADS {
            let base = t * PER;
            for i in 0..PER {
                if i % 2 == 0 {
                    assert_eq!(map.find(&(base + i)), None);
                } else {
                    assert_eq!(map.find(&(base + i)), Some(base + i));
                }
            }
        }
    }

    #[test]
    fn ll_concurrent_insert_or_assign_on_one_key_is_atomic() {
        // Same bucket, same key: every writer must land a whole value and the
        // map must hold exactly one entry.
        const THREADS: u64 = 8;
        let map = UnorderedMapLl::<u64, u64>::with_buckets(4);
        std::thread::scope(|s| {
            for t in 0..THREADS {
                let map = &map;
                s.spawn(move || {
                    for _ in 0..500 {
                        let r = map.insert_or_assign(1, t);
                        assert!(r.value.is_some());
                        let observed = map.get(&1);
                        assert!(observed < THREADS, "torn value {observed}");
                    }
                });
            }
        });
        assert_eq!(map.size(), 1);
        assert!(map.get(&1) < THREADS);
    }

    #[test]
    fn ll_shared_for_each_runs_alongside_writers() {
        // Issue #680 shape: a full-map shared scan concurrent with single-key
        // writes must keep making progress (and must not deadlock against the
        // rehash writer).
        const KEYS: u64 = 400;
        let map = Arc::new(UnorderedMapLl::<u64, u64>::with_buckets(4));
        let scans = Arc::new(AtomicU64::new(0));
        let writer_ops = Arc::new(AtomicU64::new(0));
        let done = Arc::new(AtomicBool::new(false));

        std::thread::scope(|s| {
            for _ in 0..2 {
                let (map, scans, done) = (Arc::clone(&map), Arc::clone(&scans), Arc::clone(&done));
                s.spawn(move || {
                    // do-while: at least one scan always completes, so the
                    // progress assertions below can't race the main thread.
                    loop {
                        let mut seen = 0u64;
                        map.for_each(|_, _| seen += 1, ForEachLock::Shared);
                        scans.fetch_add(1, Ordering::Relaxed);
                        if done.load(Ordering::SeqCst) {
                            break;
                        }
                    }
                });
            }
            for _ in 0..4 {
                let (map, writer_ops, done) = (
                    Arc::clone(&map),
                    Arc::clone(&writer_ops),
                    Arc::clone(&done),
                );
                s.spawn(move || {
                    loop {
                        for k in 0..KEYS {
                            map.insert_or_assign(k, k);
                            map.erase(&k);
                            writer_ops.fetch_add(1, Ordering::Relaxed);
                        }
                        if done.load(Ordering::SeqCst) {
                            break;
                        }
                    }
                });
            }
            // Grow the table underneath both, then stop everyone.
            for k in 0..KEYS {
                map.insert(k, k);
            }
            for _ in 0..20 {
                map.rehash(64);
                map.rehash(8);
            }
            done.store(true, Ordering::SeqCst);
        });

        assert!(scans.load(Ordering::SeqCst) > 0, "shared scans made progress");
        assert!(writer_ops.load(Ordering::SeqCst) > 0, "writers made progress");
    }

    #[test]
    fn ll_exclusive_for_each_vs_concurrent_ops() {
        // Mirrors the "for_each(write) vs concurrent insert/erase/find" repro:
        // the exclusive scan must see a quiescent map and everyone must finish.
        const KEYS: u64 = 200;
        let map = Arc::new(UnorderedMapLl::<u64, u64>::with_buckets(8));
        for k in 0..KEYS {
            map.insert(k, k);
        }
        let scans = Arc::new(AtomicU64::new(0));
        let done = Arc::new(AtomicBool::new(false));

        std::thread::scope(|s| {
            {
                let (map, scans, done) = (Arc::clone(&map), Arc::clone(&scans), Arc::clone(&done));
                s.spawn(move || {
                    while !done.load(Ordering::SeqCst) {
                        // Values are invariant (v == k) for keys that exist, so a
                        // quiescent snapshot can be asserted from inside.
                        map.for_each(|k, v| assert_eq!(*k, *v), ForEachLock::Exclusive);
                        map.for_each_mut(|k, v| *v = *k);
                        scans.fetch_add(1, Ordering::Relaxed);
                    }
                });
            }
            for t in 0..4u64 {
                let (map, done) = (Arc::clone(&map), Arc::clone(&done));
                s.spawn(move || {
                    while !done.load(Ordering::SeqCst) {
                        let k = KEYS + t;
                        map.insert_or_assign(k, k);
                        let _ = map.find(&k);
                        map.erase(&k);
                    }
                });
            }
            while scans.load(Ordering::SeqCst) < 50 {
                std::hint::spin_loop();
            }
            done.store(true, Ordering::SeqCst);
        });
        assert!(scans.load(Ordering::SeqCst) >= 50);
        assert_eq!(map.size(), KEYS as usize);
    }

    #[test]
    fn ll_concurrent_with_key_locked_guards() {
        // The locked guard must be safe to interleave with self-locking ops on
        // other buckets.
        const THREADS: u64 = 8;
        const PER: u64 = 200;
        let map = UnorderedMapLl::<u64, u64>::with_buckets(64);
        std::thread::scope(|s| {
            for t in 0..THREADS {
                let map = &map;
                s.spawn(move || {
                    for i in 0..PER {
                        let k = t * PER + i;
                        map.with_key_locked(&k, |l| {
                            assert!(l.insert(k, k).inserted);
                            *l.find_mut(&k).unwrap() += 1;
                        });
                        assert_eq!(map.find(&k), Some(k + 1));
                    }
                });
            }
        });
        assert_eq!(map.size(), (THREADS * PER) as usize);
    }

    // -----------------------------------------------------------------------
    // unordered_map_lhash — construction / edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn lhash_empty_map_has_cpp_defaults() {
        let map = UnorderedMapLhash::<u64, u64>::new();
        assert_eq!(map.size(), 0);
        assert!(map.is_empty());
        assert_eq!(map.bucket_count(), DEFAULT_CAPACITY); // C++ capacity = 16
        // C++ init_locks(num_locks < capacity ? num_locks : capacity).
        assert_eq!(map.num_locks(), 16);
        assert_eq!(map.find(&7), None);
        assert!(!map.contains(&7));
        assert_eq!(map.count(&7), 0);
        assert_eq!(map.erase(&7), 0);
        let mut seen = 0;
        map.for_each(|_, _| seen += 1);
        map.for_each_mut(|_, _| seen += 1);
        assert_eq!(seen, 0);
        map.clear();
        assert!(map.is_empty());
    }

    #[test]
    fn lhash_zero_capacity_clamps_to_one_slot() {
        // Divergence 8: C++ leaves slots_ empty and num_locks_ == 0, so every
        // keyed op divides by zero (UB). We clamp, as the sibling ll map does.
        let map = UnorderedMapLhash::<u64, u64>::with_capacity(0);
        assert_eq!(map.bucket_count(), 1);
        assert_eq!(map.num_locks(), 1);
        assert_eq!(map.find(&1), None); // would be UB in C++
        assert_eq!(map.erase(&1), 0);
        assert!(map.insert(1, 10).inserted);
        assert_eq!(map.find(&1), Some(10));
    }

    #[test]
    fn lhash_num_locks_is_min_of_num_locks_and_capacity() {
        assert_eq!(UnorderedMapLhash::<u64, u64>::with_num_locks(16, 4).num_locks(), 4);
        assert_eq!(UnorderedMapLhash::<u64, u64>::with_num_locks(4, 64).num_locks(), 4);
        assert_eq!(UnorderedMapLhash::<u64, u64>::with_capacity(128).num_locks(), 64);
        assert_eq!(UnorderedMapLhash::<u64, u64>::default().bucket_count(), 16);
    }

    // -----------------------------------------------------------------------
    // unordered_map_lhash — probing / tombstones
    // -----------------------------------------------------------------------

    #[test]
    fn lhash_linear_probe_resolves_collisions() {
        // Identity hash: 1 and 17 both map to slot 1 in a 16-slot table.
        let map = UnorderedMapLhash::<u64, u64>::with_capacity(16);
        assert!(map.insert(1, 10).inserted);
        assert!(map.insert(17, 170).inserted); // probes to slot 2
        assert_eq!(map.find(&1), Some(10));
        assert_eq!(map.find(&17), Some(170));
        assert_eq!(map.size(), 2);
        assert_eq!(map.bucket_count(), 16); // 2/16 is under the 75% trigger
    }

    #[test]
    fn lhash_tombstone_keeps_probe_chain_intact_and_is_reused() {
        let map = UnorderedMapLhash::<u64, u64>::with_capacity(16);
        map.insert(1, 10); // slot 1
        map.insert(17, 170); // slot 2 (probed past 1)
        assert_eq!(map.erase(&1), 1); // slot 1 -> tombstone
        assert_eq!(map.size(), 1);
        // C++ find_slot walks *through* tombstones (only kEmpty stops it).
        assert_eq!(map.find(&17), Some(170));
        assert_eq!(map.find(&1), None);
        // C++ find_insert_slot reuses the first available (tombstone) slot.
        assert!(map.insert(33, 330).inserted); // 33 % 16 == 1 -> reuses slot 1
        assert_eq!(map.size(), 2);
        assert_eq!(map.find(&33), Some(330));
        assert_eq!(map.find(&17), Some(170)); // still reachable behind it
        assert_eq!(map.erase(&17), 1);
        assert_eq!(map.erase(&33), 1);
        assert!(map.is_empty());
    }

    #[test]
    fn lhash_erase_missing_key_is_zero() {
        let map = UnorderedMapLhash::<u64, u64>::with_capacity(16);
        map.insert(1, 10);
        assert_eq!(map.erase(&2), 0); // hits an empty slot, stops
        assert_eq!(map.erase(&17), 0); // probes past 1, then empty
        assert_eq!(map.size(), 1);
    }

    #[test]
    fn lhash_lookup_in_a_full_table_terminates() {
        // No empty slot to stop the probe: find_slot must bail after `cap` steps
        // rather than spin forever.
        let map = UnorderedMapLhash::<u64, u64>::with_capacity(8);
        map.insert(0, 0);
        map.insert(1, 1);
        map.rehash(2); // cap 2, both slots occupied
        assert_eq!(map.bucket_count(), 2);
        assert_eq!(map.size(), 2);
        assert_eq!(map.find(&99), None);
        assert!(!map.contains(&99));
    }

    // -----------------------------------------------------------------------
    // unordered_map_lhash — single-key semantics
    // -----------------------------------------------------------------------

    #[test]
    fn lhash_insert_leaves_existing_entry_untouched() {
        let map = UnorderedMapLhash::<u64, u64>::with_capacity(16);
        let r = map.insert(7, 1);
        assert!(r.inserted);
        assert_eq!(r.value, Some(1));
        let r = map.insert(7, 2);
        assert!(!r.inserted);
        assert_eq!(r.value, Some(1)); // C++ returns the existing value
        assert_eq!(map.find(&7), Some(1));
        assert_eq!(map.size(), 1);
    }

    #[test]
    fn lhash_insert_or_assign_overwrites() {
        let map = UnorderedMapLhash::<u64, u64>::with_capacity(16);
        assert!(map.insert_or_assign(7, 1).inserted);
        let r = map.insert_or_assign(7, 2);
        assert!(!r.inserted);
        assert_eq!(r.value, Some(2));
        assert_eq!(map.find(&7), Some(2));
        assert_eq!(map.size(), 1);
    }

    #[test]
    fn lhash_get_or_insert_default_creates_entry() {
        let map = UnorderedMapLhash::<u64, u64>::with_capacity(16);
        assert_eq!(map.get_or_insert_default(7), 0); // C++ operator[]: T()
        assert_eq!(map.size(), 1);
        map.with_value_mut(&7, |v| *v = 5);
        assert_eq!(map.get_or_insert_default(7), 5); // existing entry kept
        assert_eq!(map.size(), 1);
    }

    #[test]
    fn lhash_with_value_and_with_value_mut() {
        let map = UnorderedMapLhash::<String, Vec<u64>>::with_capacity(8);
        map.insert("k".into(), vec![1]);
        assert_eq!(map.with_value(&"k".to_string(), |v| v.len()), Some(1));
        map.with_value_mut(&"k".to_string(), |v| v.push(2));
        assert_eq!(map.find(&"k".to_string()), Some(vec![1, 2]));
        assert_eq!(map.with_value(&"nope".to_string(), |v| v.len()), None);
    }

    #[test]
    fn lhash_clear_resets_tombstones() {
        let map = UnorderedMapLhash::<u64, u64>::with_capacity(16);
        map.insert(1, 10);
        map.insert(17, 170);
        map.erase(&1); // tombstone
        map.clear();
        assert!(map.is_empty());
        assert_eq!(map.find(&17), None);
        assert_eq!(map.bucket_count(), 16); // C++ clear() does not resize
        assert!(map.insert(17, 171).inserted); // table is usable again
        assert_eq!(map.find(&17), Some(171));
    }

    // -----------------------------------------------------------------------
    // unordered_map_lhash — growth
    // -----------------------------------------------------------------------

    #[test]
    fn lhash_grows_past_seventy_five_percent_load() {
        // C++ maybe_rehash_locked: cur_size * 4 > slots_.size() * 3.
        let map = UnorderedMapLhash::<u64, u64>::with_capacity(16);
        for k in 0..12u64 {
            map.insert(k, k);
        }
        assert_eq!(map.bucket_count(), 16, "12*4 == 48 is not > 48");
        map.insert(12, 12);
        assert_eq!(map.bucket_count(), 32, "13*4 == 52 > 48 => x2");
        assert_eq!(map.size(), 13);
        for k in 0..13u64 {
            assert_eq!(map.find(&k), Some(k));
        }
    }

    #[test]
    fn lhash_grows_and_keeps_all_entries() {
        const N: u64 = 2000;
        let map = UnorderedMapLhash::<u64, u64>::with_capacity(4);
        for i in 0..N {
            assert!(map.insert(i, i * 3 + 7).inserted);
        }
        assert_eq!(map.size(), N as usize);
        assert!(map.bucket_count() > 4);
        for i in 0..N {
            assert_eq!(map.find(&i), Some(i * 3 + 7));
        }
        for i in (0..N).step_by(2) {
            assert_eq!(map.erase(&i), 1);
        }
        assert_eq!(map.size(), (N / 2) as usize);
        for i in (1..N).step_by(2) {
            assert_eq!(map.find(&i), Some(i * 3 + 7));
        }
    }

    #[test]
    fn lhash_insert_into_a_full_table_grows_then_retries() {
        // Exercises the `idx >= slots_.size()` branch of C++ insert_no_lock.
        let map = UnorderedMapLhash::<u64, u64>::with_capacity(8);
        map.insert(0, 0);
        map.insert(1, 1);
        map.rehash(2); // cap 2, 100% full, no free slot for any new key
        assert_eq!(map.bucket_count(), 2);
        let r = map.insert(9, 90);
        assert!(r.inserted);
        assert_eq!(r.value, Some(90));
        assert_eq!(map.bucket_count(), 4, "table full => grow, then insert");
        assert_eq!(map.size(), 3);
        for (k, v) in [(0u64, 0u64), (1, 1), (9, 90)] {
            assert_eq!(map.find(&k), Some(v));
        }
    }

    #[test]
    fn lhash_capacity_one_grows_immediately() {
        let map = UnorderedMapLhash::<u64, u64>::with_capacity(1);
        assert_eq!(map.bucket_count(), 1);
        assert!(map.insert(0, 0).inserted); // 1*4 > 1*3 => grow to 2
        assert_eq!(map.bucket_count(), 2);
        assert!(map.insert(1, 1).inserted); // 2*4 > 2*3 => grow to 4
        assert_eq!(map.bucket_count(), 4);
        assert_eq!(map.size(), 2);
        assert_eq!(map.find(&0), Some(0));
        assert_eq!(map.find(&1), Some(1));
    }

    #[test]
    fn lhash_rehash_preserves_entries_and_clears_tombstones() {
        let map = UnorderedMapLhash::<u64, u64>::with_capacity(16);
        for k in 0..8u64 {
            map.insert(k, k * 2);
        }
        map.erase(&3); // tombstone
        assert!(map.rehash(64));
        assert_eq!(map.bucket_count(), 64);
        assert_eq!(map.size(), 7);
        assert_eq!(map.find(&3), None);
        for k in (0..8u64).filter(|k| *k != 3) {
            assert_eq!(map.find(&k), Some(k * 2));
        }
        assert!(map.rehash(0)); // divergence 8: 0 -> 1 slot
        assert_eq!(map.bucket_count(), 1);
    }

    #[test]
    fn lhash_shrinking_rehash_silently_drops_entries() {
        // Divergence 10 (deliberate C++ parity): rehash_no_lock zeroes size_ and
        // re-inserts; whatever does not fit is discarded.
        let map = UnorderedMapLhash::<u64, u64>::with_capacity(16);
        for k in 0..5u64 {
            map.insert(k, k);
        }
        assert_eq!(map.size(), 5);
        assert!(map.rehash(2));
        assert_eq!(map.bucket_count(), 2);
        assert_eq!(map.size(), 2, "3 of 5 entries were dropped");
        let mut survivors = 0;
        map.for_each(|_, _| survivors += 1);
        assert_eq!(survivors, 2, "size_ agrees with the surviving slots");
    }

    // -----------------------------------------------------------------------
    // unordered_map_lhash — for_each / locked guard / concurrency
    // -----------------------------------------------------------------------

    #[test]
    fn lhash_for_each_visits_only_occupied_slots() {
        let map = UnorderedMapLhash::<u64, u64>::with_capacity(64);
        for k in 0..20u64 {
            map.insert(k, k * 10);
        }
        map.erase(&5); // tombstone must not be visited
        let mut keys: Vec<u64> = Vec::new();
        map.for_each(|k, v| {
            keys.push(*k);
            assert_eq!(*v, *k * 10);
        });
        keys.sort_unstable();
        assert_eq!(keys, (0..20).filter(|k| *k != 5).collect::<Vec<u64>>());

        map.for_each_mut(|_, v| *v += 1);
        assert_eq!(map.find(&7), Some(71));
    }

    #[test]
    fn lhash_lock_guard_find_insert_erase() {
        // C++: lock_key -> find_locked/insert_locked/erase_locked -> unlock_key.
        let map = UnorderedMapLhash::<u64, u64>::with_capacity(16);
        {
            let mut locked = map.lock();
            assert!(locked.insert(3, 30).inserted);
            assert_eq!(locked.find(&3), Some(&30));
            let r = locked.insert(3, 99); // existing: untouched
            assert!(!r.inserted);
            assert_eq!(r.value, Some(30));
            assert_eq!(locked.insert_or_assign(3, 31).value, Some(31));
            *locked.find_mut(&3).unwrap() = 32;
            assert_eq!(locked.size(), 1);
            assert_eq!(locked.bucket_count(), 16);
            // The guard covers every key, not just one stripe (divergence 12).
            assert!(locked.insert(4, 40).inserted);
            assert_eq!(locked.erase(&4), 1);
            assert_eq!(locked.erase(&4), 0);
            assert_eq!(locked.find(&4), None);
        }
        assert_eq!(map.find(&3), Some(32));
        assert_eq!(map.size(), 1);
    }

    #[test]
    fn lhash_concurrent_insert_find_erase_with_growth() {
        const THREADS: u64 = 8;
        const PER: u64 = 300;
        let map = UnorderedMapLhash::<u64, u64>::with_capacity(4);
        std::thread::scope(|s| {
            for t in 0..THREADS {
                let map = &map;
                s.spawn(move || {
                    let base = t * PER;
                    for i in 0..PER {
                        let k = base + i;
                        assert!(map.insert(k, k ^ 0xA5A5_A5A5).inserted);
                        assert_eq!(map.find(&k), Some(k ^ 0xA5A5_A5A5));
                        assert_eq!(map.count(&k), 1);
                        if i % 2 == 0 {
                            assert_eq!(map.erase(&k), 1);
                        }
                        assert_eq!(map.find(&u64::MAX), None);
                    }
                });
            }
        });
        assert_eq!(map.size(), (THREADS * PER / 2) as usize);
        assert!(map.bucket_count() > 4);
        for t in 0..THREADS {
            for i in 0..PER {
                let k = t * PER + i;
                if i % 2 == 0 {
                    assert_eq!(map.find(&k), None);
                } else {
                    assert_eq!(map.find(&k), Some(k ^ 0xA5A5_A5A5));
                }
            }
        }
    }

    #[test]
    fn lhash_concurrent_readers_and_writers_stay_consistent() {
        const KEYS: u64 = 500;
        let value_of = |k: u64| k.wrapping_mul(2_654_435_761);
        const SEED: u64 = 8;
        let map = Arc::new(UnorderedMapLhash::<u64, u64>::with_capacity(4));
        let bad = Arc::new(AtomicU64::new(0));
        let reads = Arc::new(AtomicU64::new(0));
        let done = Arc::new(AtomicBool::new(false));

        // Seeded before the readers start (see the ll twin): without this,
        // `reads > 0` races the writer finishing first.
        for k in 0..SEED {
            map.insert(k, value_of(k));
        }

        std::thread::scope(|s| {
            for _ in 0..4 {
                let (map, bad, reads, done) = (
                    Arc::clone(&map),
                    Arc::clone(&bad),
                    Arc::clone(&reads),
                    Arc::clone(&done),
                );
                s.spawn(move || {
                    // do-while: always complete at least one full pass.
                    loop {
                        for k in 0..KEYS {
                            if let Some(v) = map.find(&k) {
                                reads.fetch_add(1, Ordering::Relaxed);
                                if v != value_of(k) {
                                    bad.fetch_add(1, Ordering::Relaxed);
                                }
                            }
                        }
                        map.for_each(|k, v| {
                            if *v != value_of(*k) {
                                bad.fetch_add(1, Ordering::Relaxed);
                            }
                        });
                        if done.load(Ordering::SeqCst) {
                            break;
                        }
                    }
                });
            }
            for k in SEED..KEYS {
                map.insert(k, value_of(k));
            }
            done.store(true, Ordering::SeqCst);
        });

        assert_eq!(bad.load(Ordering::SeqCst), 0, "never observed a corrupt value");
        assert!(reads.load(Ordering::SeqCst) > 0);
        assert_eq!(map.size(), KEYS as usize);
        for k in 0..KEYS {
            assert_eq!(map.find(&k), Some(value_of(k)));
        }
    }

    #[test]
    fn lhash_concurrent_insert_or_assign_on_one_key_is_atomic() {
        const THREADS: u64 = 8;
        let map = UnorderedMapLhash::<u64, u64>::with_capacity(8);
        std::thread::scope(|s| {
            for t in 0..THREADS {
                let map = &map;
                s.spawn(move || {
                    for _ in 0..500 {
                        assert!(map.insert_or_assign(1, t).value.is_some());
                        let observed = map.find(&1).expect("present");
                        assert!(observed < THREADS, "torn value {observed}");
                    }
                });
            }
        });
        assert_eq!(map.size(), 1);
    }
}
