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

#ifndef CTP_THREAD_MUTEX_H_
#define CTP_THREAD_MUTEX_H_

#include "clio_ctp/thread/thread_model_manager.h"
#include "clio_ctp/types/atomic.h"
#include "clio_ctp/types/numbers.h"
#if !CTP_IS_DEVICE_PASS
#include <chrono>
#include <thread>
#endif

namespace ctp {

struct Mutex {
  ipc::atomic<ctp::min_u64> lock_;
  ipc::atomic<ctp::min_u64> head_;
  ipc::atomic<ctp::min_u32> try_lock_;
  /** Default constructor */
  CTP_INLINE_CROSS_FUN
  Mutex() : lock_(0), head_(0), try_lock_(0) {}

  /** Copy constructor */
  CTP_INLINE_CROSS_FUN
  Mutex(const Mutex &other) {}

  /** Explicit initialization */
  CTP_INLINE_CROSS_FUN
  void Init() {
    lock_ = 0;
    head_ = 0;
  }

  /** Acquire lock */
  CTP_INLINE_CROSS_FUN
  void Lock(u32 owner) {
    min_u64 tkt = lock_.fetch_add(1);
    u32 spin_count = 0;
    do {
      // Use load_device() for cross-SM L2 visibility on GPU.
      // Unlock() advances head_ via fetch_add (L2 atomic), but
      // a volatile load() on a different SM reads stale L1 data.
      min_u64 cur = head_.load_device();
      if (tkt == cur) {
        return;
      }
#if CTP_IS_GPU
      ++spin_count;
      if (spin_count == 5000000) {
        printf("[MUTEX] STUCK: tkt=%llu head=%llu this=%p\n",
               (unsigned long long)tkt,
               (unsigned long long)head_.load_device(),
               (void*)this);
        spin_count = 0;
      }
#endif
      // Yielding to a host thread model only makes sense on the CPU. On any
      // device pass (CUDA, ROCm, SYCL) we busy-spin instead — the singleton
      // chain reaches a non-const static which DPC++ rejects in kernels,
      // and there's no host scheduler to yield to anyway.
#if !CTP_IS_DEVICE_PASS
      Backoff(++spin_count, tkt - cur);
#endif
    } while (true);
  }

  /**
   * Adaptive contention backoff for the host (CPU) ticket-wait path.
   *
   * This is a FAIR ticket lock: a waiter holding ticket `tkt` cannot acquire
   * until `ahead = tkt - head` preceding holders each call Unlock(). The
   * Pthread thread model's Yield() is a bare PAUSE (x86) / `yield` (arm)
   * hint that does NOT deschedule the CPU. So a far-back waiter that only
   * busy-spins keeps a core 100% busy while contributing nothing, and under
   * heavy cross-process contention on macOS — whose scheduler will not
   * preempt a PAUSE-spinning thread to run the descheduled lock holder — it
   * starves the holder and the whole lock livelocks for seconds (observed as
   * the ctp_mp_allocator_multiprocess hang, issue #483; Linux happens to make
   * progress via timer preemption).
   *
   * Strategy: a short cooperative-yield fast path preserves low handoff
   * latency for the common lightly-contended case; once contention is
   * sustained we additionally deschedule the OS thread so the holder gets the
   * core. Sleeping is only safe for OS-thread models — for user-level-thread
   * models (Argobots) sleeping the execution stream would block sibling ULTs
   * (possibly the holder), so we stay purely cooperative there.
   *
   * Uncontended acquire never reaches here (Lock returns on the first
   * iteration), so this adds zero overhead to the fast path.
   */
  CTP_INLINE_CROSS_FUN
  void Backoff(u32 spin_count, min_u64 ahead) {
#if !CTP_IS_DEVICE_PASS
    // Cooperative yield first: cheap PAUSE / std::yield / ULT yield.
    CTP_THREAD_MODEL->Yield();
    if (spin_count < 64) {
      return;
    }
    // Sustained contention: for OS-thread models, sleep briefly so a
    // descheduled holder can run. Sleep grows with queue position (we
    // provably cannot acquire until `ahead` Unlock()s happen) but is capped
    // so worst-case handoff latency stays bounded.
    ThreadType ty = CTP_THREAD_MODEL->GetType();
    if (ty == ThreadType::kPthread || ty == ThreadType::kStdThread) {
      min_u64 us = ahead < 100 ? ahead : 100;
      if (us == 0) {
        us = 1;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(us));
    }
#else
    (void)spin_count;
    (void)ahead;
#endif
  }

  /** Try to acquire the lock */
  CTP_INLINE_CROSS_FUN
  bool TryLock(u32 owner) {
    if (try_lock_.fetch_add(1) > 0 || lock_.load_device() > head_.load_device()) {
      try_lock_.fetch_sub(1);
      return false;
    }
    Lock(owner);
    return true;
  }

  /** Unlock */
  CTP_INLINE_CROSS_FUN
  void Unlock() {
    head_.fetch_add(1);
  }
};

struct ScopedMutex {
  Mutex &lock_;
  bool is_locked_;

  /** Acquire the mutex */
  CTP_INLINE_CROSS_FUN explicit ScopedMutex(Mutex &lock, u32 owner)
      : lock_(lock), is_locked_(false) {
    Lock(owner);
  }

  /** Release the mutex */
  CTP_INLINE_CROSS_FUN
  ~ScopedMutex() { Unlock(); }

  /** Explicitly acquire the mutex */
  CTP_INLINE_CROSS_FUN
  void Lock(u32 owner) {
    if (!is_locked_) {
      lock_.Lock(owner);
      is_locked_ = true;
    }
  }

  /** Explicitly try to lock the mutex */
  CTP_INLINE_CROSS_FUN
  bool TryLock(u32 owner) {
    if (!is_locked_) {
      is_locked_ = lock_.TryLock(owner);
    }
    return is_locked_;
  }

  /** Explicitly unlock the mutex */
  CTP_INLINE_CROSS_FUN
  void Unlock() {
    if (is_locked_) {
      lock_.Unlock();
      is_locked_ = false;
    }
  }
};

}  // namespace ctp

namespace ctp::ipc {

using ctp::Mutex;
using ctp::ScopedMutex;

}  // namespace ctp::ipc

#undef Mutex
#undef ScopedMutex

#endif  // CTP_THREAD_MUTEX_H_
