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

#ifndef CTP_THREAD_LOCK_TIMED_MUTEX_H_
#define CTP_THREAD_LOCK_TIMED_MUTEX_H_

#include "clio_ctp/thread/thread_model_manager.h"
#include "clio_ctp/types/atomic.h"
#include "clio_ctp/types/numbers.h"
#include "clio_ctp/introspect/system_info.h"

#if !CTP_IS_DEVICE_PASS
#include <chrono>
#include <cstdlib>
#include <thread>
#endif

// POSIX liveness (kill(pid,0)) is available on Linux and macOS/BSD alike.
// Only the /proc start-time refinement is Linux-specific.
#if !defined(_WIN32) && !CTP_IS_DEVICE_PASS
#define CTP_TIMED_MUTEX_POSIX_LIVENESS 1
#include <cerrno>
#include <csignal>
#include <cstdio>
#endif

namespace ctp {

/**
 * TimedMutex -- a cross-process mutex whose holder can be RECLAIMED if it dies
 * (issue #783).
 *
 * WHY NOT `ctp::Mutex`: `Mutex` is a fair TICKET lock -- `Lock()` takes
 * `lock_.fetch_add(1)` and spins until `head_` reaches its ticket. If the
 * holder dies, `head_` is never advanced and every subsequent waiter blocks
 * forever; and force-advancing `head_` releases several waiters at once,
 * destroying mutual exclusion. A ticket lock fundamentally cannot be reclaimed.
 * `TimedMutex` is therefore an unfair CAS lock, which trades FIFO fairness for
 * the ability to break a lock left behind by a dead process.
 *
 * INTENDED USE: guarding shared-memory objects that are read by client
 * processes the owner cannot trust to stay alive. A client may be SIGKILLed at
 * any instant, including while holding this lock.
 *
 * NOT A GENERAL-PURPOSE MUTEX. Prefer `ctp::Mutex` for anything that lives
 * inside one process: it is fair, cheaper, and has no reclamation machinery.
 *
 * SHM-SAFE: no virtual functions, no pointers, no heap. Every field is a
 * fixed-width atomic, so the object is valid at any base address in any
 * process. (Any type stored in shared memory must satisfy this -- a vtable
 * pointer would be a process-local address.)
 *
 * RECLAMATION ALGORITHM. Each acquisition stamps `state_` with a unique,
 * never-reused id drawn from `ticket_`. A waiter:
 *   1. samples `state_` (call it `observed`, non-zero => held),
 *   2. waits, re-checking; if `state_` ever changes the holder made progress,
 *      so the wait restarts,
 *   3. once `observed` has been continuously unchanged for `timeout_ms`, checks
 *      whether the owner process is still alive,
 *   4. steals ONLY if the owner is gone, via `CAS(observed -> 0)`.
 *
 * Two properties fall out of this:
 *   - Elapsed time is measured on the WAITER's own clock against an unchanged
 *     acquisition id, so no shared timestamp and no cross-process clock
 *     agreement is required.
 *   - The CAS makes the steal ABA-safe: if the holder released and someone else
 *     acquired in the meantime, `state_` no longer equals `observed` and the
 *     steal fails harmlessly.
 *
 * A live-but-slow holder is NEVER stolen from, only a provably dead one. That
 * matters: stealing from a live reader would let a writer mutate data out from
 * under it, which is precisely the silent corruption this exists to prevent.
 * A timeout alone cannot distinguish "dead" from "descheduled / page-faulting",
 * so the liveness check is mandatory, not advisory.
 */
struct TimedMutex {
  /** 0 = unlocked; otherwise the unique id of the current acquisition. */
  ipc::atomic<min_u64> state_;
  /** Source of unique acquisition ids. Bumped per ATTEMPT; only uniqueness
   *  matters, not density. */
  ipc::atomic<min_u64> ticket_;
  /** OS pid of the current holder (0 if none). Written after winning the CAS. */
  ipc::atomic<min_u32> owner_pid_;
  /** Holder's process start time, to defeat PID REUSE: a recycled pid would
   *  otherwise look alive and make a dead holder unreclaimable forever.
   *  Linux: field 22 of /proc/<pid>/stat. 0 = unknown. */
  ipc::atomic<min_u64> owner_start_;
  /** Count of locks broken from dead owners. Diagnostics only -- a non-zero
   *  value means clients are dying while holding leases. */
  ipc::atomic<min_u64> steal_count_;

  /** Default reclamation threshold. A lock continuously held this long by a
   *  process that has since died is eligible to be broken. */
  static constexpr min_u64 kDefaultTimeoutMs = 500;

  CTP_INLINE_CROSS_FUN
  TimedMutex()
      : state_(0), ticket_(0), owner_pid_(0), owner_start_(0), steal_count_(0) {}

  /** Copy ctor leaves the copy UNLOCKED. Copying a locked mutex would
   *  otherwise duplicate ownership. Mirrors ctp::Mutex. */
  CTP_INLINE_CROSS_FUN
  TimedMutex(const TimedMutex &other) { Init(); }

  CTP_INLINE_CROSS_FUN
  TimedMutex &operator=(const TimedMutex &other) {
    if (this != &other) {
      Init();
    }
    return *this;
  }

  /** Explicit initialization (placement-new into shared memory). */
  CTP_INLINE_CROSS_FUN
  void Init() {
    state_ = 0;
    ticket_ = 0;
    owner_pid_ = 0;
    owner_start_ = 0;
    steal_count_ = 0;
  }

  /** True if currently held by anyone. Advisory: the answer may be stale the
   *  instant it is returned. */
  CTP_INLINE_CROSS_FUN
  bool IsLocked() const { return state_.load() != 0; }

  /**
   * Try to acquire without waiting.
   * @return the acquisition id on success (pass it to Unlock), or 0 on failure.
   */
  CTP_INLINE_CROSS_FUN
  min_u64 TryLock() {
    min_u64 my = ticket_.fetch_add(1) + 1;  // never 0
    min_u64 expected = 0;
    if (!state_.compare_exchange_strong(expected, my)) {
      return 0;
    }
    // We own it: publish identity for liveness checks by future waiters.
    StampOwner();
    return my;
  }

  /**
   * Acquire, blocking the calling THREAD until success.
   *
   * Callers on a runtime worker should be aware this occupies the worker for
   * the duration; the issue #781 work-orchestrator stall detection is what
   * makes that survivable. Prefer TryLock() plus a caller-level retry where
   * skipping the work is acceptable (e.g. background reorganization).
   *
   * @param timeout_ms how long an UNCHANGED acquisition must persist before a
   *        dead owner's lock is broken.
   * @return the acquisition id (pass it to Unlock).
   */
  CTP_INLINE_CROSS_FUN
  min_u64 Lock(min_u64 timeout_ms = kDefaultTimeoutMs) {
    for (;;) {
      min_u64 my = TryLock();
      if (my != 0) {
        return my;
      }
      WaitAndMaybeReclaim(timeout_ms);
    }
  }

  /**
   * Release.
   *
   * @param id the value returned by TryLock/Lock. The release is a CAS, so if
   *        the lock was already broken (owner presumed dead) and re-acquired by
   *        someone else, this is a NO-OP rather than a silent unlock of the new
   *        owner's acquisition. Passing 0 force-unlocks and is for teardown only.
   * @return true if this call actually released the lock.
   */
  CTP_INLINE_CROSS_FUN
  bool Unlock(min_u64 id) {
    if (id == 0) {
      owner_pid_ = 0;
      owner_start_ = 0;
      state_ = 0;
      return true;
    }
    min_u64 expected = id;
    // Clear identity BEFORE releasing: after state_ hits 0 another process may
    // acquire immediately, and it must not observe our pid as the owner.
    owner_pid_ = 0;
    owner_start_ = 0;
    if (state_.compare_exchange_strong(expected, 0)) {
      return true;
    }
    return false;  // already stolen; the new owner has re-stamped identity
  }

  /** Number of locks broken from dead owners (diagnostics). */
  CTP_INLINE_CROSS_FUN
  min_u64 GetStealCount() const { return steal_count_.load(); }

 private:
  /** Publish this process's identity as the holder. */
  CTP_INLINE_CROSS_FUN
  void StampOwner() {
#if !CTP_IS_DEVICE_PASS
    owner_pid_ = static_cast<min_u32>(ctp::SystemInfo::GetPid());
    owner_start_ = GetProcessStartTime(static_cast<min_u32>(
        ctp::SystemInfo::GetPid()));
#endif
  }

  /**
   * Wait for the lock to change hands; if it does not within `timeout_ms` AND
   * the owner is dead, break it.
   */
  CTP_INLINE_CROSS_FUN
  void WaitAndMaybeReclaim(min_u64 timeout_ms) {
#if !CTP_IS_DEVICE_PASS
    min_u64 observed = state_.load();
    if (observed == 0) {
      return;  // became free; caller retries immediately
    }
    auto start = std::chrono::steady_clock::now();
    u32 spin = 0;
    for (;;) {
      CTP_THREAD_MODEL->Yield();
      min_u64 now_state = state_.load();
      if (now_state != observed) {
        return;  // holder made progress -- restart the wait from scratch
      }
      auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::steady_clock::now() - start)
                         .count();
      if (static_cast<min_u64>(elapsed) >= timeout_ms) {
        TryReclaim(observed);
        return;  // caller retries; if the steal worked the lock is now free
      }
      // Back off so a descheduled holder can get the core. Purely cooperative
      // for user-level-thread models, where sleeping would block sibling ULTs
      // (possibly the holder itself).
      if (++spin > 64) {
        ThreadType ty = CTP_THREAD_MODEL->GetType();
        if (ty == ThreadType::kPthread || ty == ThreadType::kStdThread) {
          std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
      }
    }
#else
    (void)timeout_ms;
#endif
  }

  /** Break `observed` if and only if its owner process is gone. */
  CTP_INLINE_CROSS_FUN
  void TryReclaim(min_u64 observed) {
#if !CTP_IS_DEVICE_PASS
    min_u32 pid = owner_pid_.load();
    min_u64 start = owner_start_.load();
    if (pid == 0) {
      return;  // identity not yet published; too early to judge
    }
    if (IsProcessAlive(pid, start)) {
      return;  // live but slow -- stealing here would corrupt its read
    }
    min_u64 expected = observed;
    if (state_.compare_exchange_strong(expected, 0)) {
      owner_pid_ = 0;
      owner_start_ = 0;
      steal_count_.fetch_add(1);
    }
    // CAS failure is fine: the lock changed hands on its own.
#else
    (void)observed;
#endif
  }

  /**
   * Read a process's start time (Linux: /proc/<pid>/stat field 22, in clock
   * ticks since boot). Returns 0 when unavailable, which callers treat as
   * "unknown" rather than "mismatch".
   */
  CTP_INLINE_CROSS_FUN
  static min_u64 GetProcessStartTime(min_u32 pid) {
#if defined(__linux__) && !CTP_IS_DEVICE_PASS
    char path[64];
    snprintf(path, sizeof(path), "/proc/%u/stat", static_cast<unsigned>(pid));
    FILE *f = fopen(path, "r");
    if (f == nullptr) {
      return 0;
    }
    // Field 2 (comm) may contain spaces and parentheses, so scan from the
    // LAST ')' rather than tokenizing from the start.
    char buf[4096];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    if (n == 0) {
      return 0;
    }
    buf[n] = '\0';
    char *rparen = nullptr;
    for (char *p = buf; *p != '\0'; ++p) {
      if (*p == ')') {
        rparen = p;
      }
    }
    if (rparen == nullptr) {
      return 0;
    }
    // After ") " comes field 3; starttime is field 22, i.e. the 20th token.
    char *p = rparen + 1;
    int field = 2;
    unsigned long long starttime = 0;
    while (*p != '\0') {
      while (*p == ' ') {
        ++p;
      }
      if (*p == '\0') {
        break;
      }
      ++field;
      if (field == 22) {
        starttime = strtoull(p, nullptr, 10);
        break;
      }
      while (*p != ' ' && *p != '\0') {
        ++p;
      }
    }
    return static_cast<min_u64>(starttime);
#else
    (void)pid;
    return 0;
#endif
  }

  /**
   * Is `pid` alive AND the same process instance that took the lock?
   *
   * Deliberately CONSERVATIVE: anything uncertain returns true (= do not
   * steal). A false "dead" frees a lock a live process is still using, which
   * corrupts data; a false "alive" only delays reclamation until the next
   * timeout. The asymmetry is not close, so ambiguity always resolves to
   * "alive".
   */
  CTP_INLINE_CROSS_FUN
  static bool IsProcessAlive(min_u32 pid, min_u64 start_time) {
#if defined(CTP_TIMED_MUTEX_POSIX_LIVENESS)
    // kill(pid, 0) is POSIX and works on macOS/BSD too, not just Linux. An
    // earlier version gated this on __linux__ and returned "always alive"
    // everywhere else, which meant a lease abandoned by a killed process was
    // NEVER reclaimed off-Linux -- the reclamation torture test hung until the
    // ctest timeout on macOS.
    if (kill(static_cast<pid_t>(pid), 0) != 0) {
      // ESRCH => gone. EPERM => it exists but belongs to another user, which
      // still means it is alive, so only treat "no such process" as dead.
      if (errno == ESRCH) {
        return false;
      }
      return true;
    }
    // The pid exists, but it may be a REUSED pid belonging to a different
    // process. Where a start time is available (Linux /proc), a mismatch
    // proves the original holder is gone. Platforms without it accept a small
    // pid-reuse window rather than never reclaiming at all.
    if (start_time != 0) {
      min_u64 cur = GetProcessStartTime(pid);
      if (cur != 0 && cur != start_time) {
        return false;
      }
    }
    return true;
#else
    // No liveness check available (Windows, device passes): never steal. The
    // lock still works, it just cannot be reclaimed from a dead holder here.
    (void)pid;
    (void)start_time;
    return true;
#endif
  }
};

/**
 * RAII guard for TimedMutex. Holds the acquisition id so the release is the
 * ABA-safe CAS form rather than a blind unlock.
 */
struct ScopedTimedLock {
  TimedMutex *mutex_;
  min_u64 id_;

  CTP_INLINE_CROSS_FUN
  explicit ScopedTimedLock(TimedMutex &m,
                           min_u64 timeout_ms = TimedMutex::kDefaultTimeoutMs)
      : mutex_(&m), id_(m.Lock(timeout_ms)) {}

  /** Non-blocking form: check `Owns()` before touching the guarded data. */
  CTP_INLINE_CROSS_FUN
  ScopedTimedLock(TimedMutex &m, bool try_only) : mutex_(&m), id_(0) {
    if (try_only) {
      id_ = m.TryLock();
    } else {
      id_ = m.Lock();
    }
  }

  CTP_INLINE_CROSS_FUN
  ~ScopedTimedLock() {
    if (mutex_ != nullptr && id_ != 0) {
      mutex_->Unlock(id_);
    }
  }

  ScopedTimedLock(const ScopedTimedLock &) = delete;
  ScopedTimedLock &operator=(const ScopedTimedLock &) = delete;

  /** True if this guard actually holds the lock. */
  CTP_INLINE_CROSS_FUN
  bool Owns() const { return id_ != 0; }
};

}  // namespace ctp

#endif  // CTP_THREAD_LOCK_TIMED_MUTEX_H_
