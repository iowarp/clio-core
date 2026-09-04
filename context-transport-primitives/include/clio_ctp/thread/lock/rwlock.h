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

#ifndef CTP_THREAD_RWLOCK_H_
#define CTP_THREAD_RWLOCK_H_

#include "clio_ctp/constants/macros.h"
#include "clio_ctp/introspect/system_info.h"
#include "clio_ctp/thread/lock.h"
#include "clio_ctp/thread/lock/mutex.h"
#include "clio_ctp/thread/thread_model_manager.h"
#include "clio_ctp/types/atomic.h"
#include "clio_ctp/types/numbers.h"

namespace ctp {

/** Retained for source compatibility. The lock no longer has a mode: every
 *  acquisition is exclusive. Nothing outside this header reads these. */
class RwLockMode {
 public:
  typedef int Type;
  CLS_CONST Type kNone = 0;
  CLS_CONST Type kWrite = 1;
  CLS_CONST Type kRead = 2;
};

/** {pid:32, tid:32} of the calling OS thread, cached in TLS.
 *
 *  Cached because SystemInfo::GetTid() is a raw syscall(SYS_gettid) (~50-100ns)
 *  — far too expensive to pay on every lock acquisition. Mirrors
 *  clio::run::GetCoLockThreadId(). Includes the pid so the id is unique across
 *  the processes sharing a segment; a tid alone collides trivially.
 *
 *  Never 0 for a live thread (pid is never 0), so 0 is usable as "no identity".
 *  Not fork()-safe: a child would inherit a stale cache. Nothing in the runtime
 *  forks after init, same assumption GetCoLockThreadId already makes.
 *
 *  NOTE: must be CTP_INLINE_CROSS_FUN (== CTP_CROSS_FUN inline), NOT
 *  CTP_INLINE — macros.h defines CTP_INLINE to NOTHING unless CTP_DEBUG is
 *  set, so a free function marked CTP_INLINE in a header multiply-defines at
 *  link time in every release build. Body is #if-guarded rather than the
 *  declaration, matching TimedMutex::StampOwner(). */
CTP_INLINE_CROSS_FUN ctp::big_uint GetRwLockSelfId() {
#if !CTP_IS_DEVICE_PASS
  static thread_local const ctp::big_uint kSelf =
      (static_cast<ctp::big_uint>(
           static_cast<ctp::reg_uint>(ctp::SystemInfo::GetPid()))
       << 32) |
      static_cast<ctp::big_uint>(
          static_cast<ctp::reg_uint>(ctp::SystemInfo::GetTid()));
  return kSelf;
#else
  // No stable per-thread identity on device; 0 disables the recursion path.
  return 0;
#endif
}

/**
 * RwLock — TEMPORARILY an exclusive, recursive lock over ctp::Mutex.
 *
 * WHY: the previous hand-rolled reader/writer algorithm did not provide
 * reader/writer exclusion (issue #927). A reader could be admitted while a
 * writer was inside, measured at ~1 in 6,500 read acquisitions at 8 readers
 * and ~1 in 120 at a single reader/single writer. The defect was structural:
 * admission was decided by reading one variable (`readers_`/`writers_`) and
 * then CASing another (`mode_`), so a stale counter read could flip the phase
 * out from under a writer that had already been granted the lock.
 *
 * Rather than ship a subtly-wrong lock, readers and writers now BOTH take the
 * same exclusive mutex. Mutual exclusion is then true by construction: there is
 * one holder, so there is nothing to get wrong. A proper parallel-reader
 * algorithm replaces this shortly — see the design on #927.
 *
 * WHAT THIS COSTS: readers no longer run in parallel. Every ReadLock is a full
 * exclusive acquisition. On read-heavy paths (CTE `target_lock_`, the client
 * put sieve's `sieve_rw_`, `unordered_map_ll`'s per-bucket locks) this is a
 * real throughput loss, accepted deliberately as the price of correctness until
 * the replacement lands.
 *
 * RECURSIVE, on purpose. Making readers exclusive would otherwise turn two
 * previously-working patterns into DETERMINISTIC self-deadlocks:
 *   - read -> read nesting on one thread (the old lock was reader-preferring
 *     precisely so this worked; clio::run::CoRwLock only short-circuits
 *     write->read, so a read->read nest reaches this lock twice), and
 *   - read -> write upgrade on one thread.
 * Tracking {pid,tid} + depth lets the same thread re-enter. Note that an
 * "upgrade" is not a lie here: the hold was already exclusive, so a nested
 * WriteLock genuinely has the exclusivity it thinks it has.
 *
 * The `holder_` fast path is sound for the reason CoRwLock's is NOT trivially
 * sound (see corwlock.h): `holder_` is written ONLY by the thread that holds
 * the mutex, is stamped after the acquisition, and is cleared BEFORE the
 * release. A non-holder therefore reads either 0 or the live holder's id — it
 * can never observe a stale copy of its OWN id from an earlier hold and let
 * itself in.
 *
 * NO DEAD-HOLDER RECOVERY, deliberately. ctp::Mutex is a fair ticket lock: if a
 * holder dies, head_ never advances and every waiter blocks forever. That is
 * acceptable because no RwLock instance currently lives in a segment shared
 * with an untrusted process -- every one of them (unordered_map_ll's
 * global_lock_/locks_ under MallocAllocator, CTE's sieve_rw_) is process-local,
 * and ctp::ipc::RwLock has no users. Cross-process shm data structures that DO
 * need a reclaimable reader/writer lock should use TimedRwLock instead (built
 * on ctp::TimedMutex, which can break a lock left by a PROVABLY dead owner).
 *
 * Choosing ctp::Mutex over ctp::TimedMutex here is not just about the unused
 * machinery: Mutex is a FAIR FIFO ticket lock, where TimedMutex is explicitly
 * an unfair CAS lock (fairness is what it trades away to be reclaimable).
 * Measured over TimedMutex, readers lost badly to writers -- 13:1 at one
 * reader and one writer. Mutex also costs 24 bytes against TimedMutex's 40,
 * and its Backoff() already does the yield-then-sleep escalation this wrapper
 * would otherwise have to hand-roll.
 *
 * SHM-SAFE in layout: fixed-width atomics only, no pointers, no vtable, no
 * heap — valid at any base address in any process. (Layout-safe is not the
 * same as safe to share with a process that may die holding it; see above.)
 * GPU: the recursion fast path is disabled on the device pass
 * (GetRwLockSelfId() returns 0), so device callers must not nest;
 * unordered_map_ll, the only device user, does not.
 */
struct RwLock {
  /** The actual exclusion. Sole source of truth for who holds the lock. */
  Mutex mutex_;
  /** {pid,tid} of the current holder; 0 when free. Holder-written only. */
  ipc::atomic<ctp::big_uint> holder_;
  /** Recursion depth of the current holder (>= 1 while held). */
  ipc::atomic<ctp::reg_uint> depth_;
  /** 1 when the OUTERMOST acquisition was a WriteLock. Preserves the original
   *  IsWriteLocked() meaning now that readers are exclusive too. */
  ipc::atomic<ctp::reg_uint> write_mode_;

  /** Default constructor */
  CTP_CROSS_FUN
  RwLock() : holder_(0), depth_(0), write_mode_(0) {}

  /** Explicit initializer (placement-new into shared memory) */
  CTP_CROSS_FUN
  void Init() {
    mutex_.Init();
    holder_.store(0);
    depth_.store(0);
    write_mode_.store(0);
  }

  /** Copy constructor: the copy is a FRESH, UNHELD lock. Copying a live lock
   *  would clone its ownership, which is never what callers want, but
   *  containers (priv::vector<RwLock>::operator=) need some copy constructor to
   *  be callable. Matches the previous RwLock and ctp::Mutex/TimedMutex. */
  CTP_CROSS_FUN
  RwLock(const RwLock & /*other*/) : holder_(0), depth_(0), write_mode_(0) {}

  /** Copy assignment (fresh + unheld, same rationale as the copy ctor). */
  CTP_CROSS_FUN
  RwLock &operator=(const RwLock & /*other*/) {
    Init();
    return *this;
  }

  /** Move constructor. Carries ownership across faithfully (the previous
   *  RwLock's move did the same) so relocating a HELD lock keeps working. */
  CTP_CROSS_FUN
  RwLock(RwLock &&other) noexcept
      : holder_(other.holder_.load()),
        depth_(other.depth_.load()),
        write_mode_(other.write_mode_.load()) {
    MoveMutexFrom(other);
  }

  /** Move assignment operator */
  CTP_CROSS_FUN
  RwLock &operator=(RwLock &&other) noexcept {
    if (this != &other) {
      MoveMutexFrom(other);
      holder_.store(other.holder_.load());
      depth_.store(other.depth_.load());
      write_mode_.store(other.write_mode_.load());
    }
    return *this;
  }

  /** Acquire read lock.
   *
   *  EXCLUSIVE for now (see the class comment) — readers do not run in
   *  parallel. Both parameters are accepted and ignored: `owner` was always
   *  unused, and `writer_priority` is meaningless when every acquisition is
   *  already exclusive and FIFO-ordered by ctp::Mutex's ticket. */
  CTP_CROSS_FUN
  void ReadLock(uint32_t /*owner*/, bool /*writer_priority*/ = false) {
    Acquire(false);
  }

  /** Release read lock */
  CTP_CROSS_FUN
  void ReadUnlock() { Release(); }

  /** Acquire write lock */
  CTP_CROSS_FUN
  void WriteLock(uint32_t /*owner*/) { Acquire(true); }

  /** Release write lock */
  CTP_CROSS_FUN
  void WriteUnlock() { Release(); }

  /** @return true while the lock is held in WRITE mode. Advisory — the answer
   *  may be stale the instant it is returned. Deliberately NOT "held at all":
   *  readers are exclusive now, and reporting a reader as a writer would change
   *  the meaning callers were written against. */
  CTP_CROSS_FUN
  bool IsWriteLocked() const { return write_mode_.load() != 0; }

 private:
  /** Copy Mutex's ticket state field-by-field. ctp::Mutex has no move ctor and
   *  its copy ctor deliberately leaves the copy unheld, so a faithful
   *  relocation of a held lock has to go through the fields. */
  CTP_INLINE_CROSS_FUN
  void MoveMutexFrom(RwLock &other) {
    mutex_.lock_.store(other.mutex_.lock_.load());
    mutex_.head_.store(other.mutex_.head_.load());
    mutex_.try_lock_.store(other.mutex_.try_lock_.load());
  }

  /** Acquire exclusively, or bump the depth if this thread already holds it. */
  CTP_INLINE_CROSS_FUN
  void Acquire(bool write) {
    const ctp::big_uint me = GetRwLockSelfId();
    // `me != 0` matters: on the device pass GetRwLockSelfId() is 0, and a free
    // lock also has holder_ == 0 — without this guard every device acquisition
    // would take the recursion path and skip the mutex entirely.
    if (me != 0 && holder_.load() == me) {
      depth_.fetch_add(1);
      return;
    }
    // Straight to the fair ticket queue. A barge-in fast path was tried here
    // and REMOVED: it only fires when the queue is empty (lock_ == head_),
    // which under real contention never happens, and it measured 2.7x SLOWER
    // at 8 readers/4 writers (134k vs 370k ops/4s) while skewing the
    // reader/writer split to 2.2:1. It was compensating for ctp::Mutex's
    // sub-slack Backoff() sleep, which is now fixed at the source.
    mutex_.Lock(0);
    write_mode_.store(write ? 1 : 0);
    depth_.store(1);
    // Publish identity LAST: once holder_ is visible the rest must already be.
    holder_.store(me);
  }

  /** Release one level; unlock the mutex when the outermost level exits. */
  CTP_INLINE_CROSS_FUN
  void Release() {
    if (depth_.load() > 1) {
      depth_.fetch_sub(1);
      return;
    }
    // Retract identity BEFORE unlocking: the moment the mutex is released
    // another thread may acquire, and it must not observe us as the holder.
    holder_.store(0);
    depth_.store(0);
    write_mode_.store(0);
    mutex_.Unlock();
  }
};

/** Acquire the read lock in a scope */
struct ScopedRwReadLock {
  RwLock &lock_;
  bool is_locked_;

  /** Acquire the read lock */
  CTP_CROSS_FUN
  explicit ScopedRwReadLock(RwLock &lock, uint32_t owner)
      : lock_(lock), is_locked_(false) {
    Lock(owner);
  }

  /** Release the read lock */
  CTP_CROSS_FUN
  ~ScopedRwReadLock() { Unlock(); }

  /** Explicitly acquire read lock */
  CTP_CROSS_FUN
  void Lock(uint32_t owner) {
    if (!is_locked_) {
      lock_.ReadLock(owner);
      is_locked_ = true;
    }
  }

  /** Explicitly release read lock */
  CTP_CROSS_FUN
  void Unlock() {
    if (is_locked_) {
      lock_.ReadUnlock();
      is_locked_ = false;
    }
  }
};

/** Acquire scoped write lock */
struct ScopedRwWriteLock {
  RwLock &lock_;
  bool is_locked_;

  /** Acquire the write lock */
  CTP_CROSS_FUN
  explicit ScopedRwWriteLock(RwLock &lock, uint32_t owner)
      : lock_(lock), is_locked_(false) {
    Lock(owner);
  }

  /** Release the write lock */
  CTP_CROSS_FUN
  ~ScopedRwWriteLock() { Unlock(); }

  /** Explicity acquire the write lock */
  CTP_CROSS_FUN
  void Lock(uint32_t owner) {
    if (!is_locked_) {
      lock_.WriteLock(owner);
      is_locked_ = true;
    }
  }

  /** Explicitly release the write lock */
  CTP_CROSS_FUN
  void Unlock() {
    if (is_locked_) {
      lock_.WriteUnlock();
      is_locked_ = false;
    }
  }
};

}  // namespace ctp

namespace ctp::ipc {

using ctp::RwLock;
using ctp::ScopedRwReadLock;
using ctp::ScopedRwWriteLock;

}  // namespace ctp::ipc

#endif  // CTP_THREAD_RWLOCK_H_
