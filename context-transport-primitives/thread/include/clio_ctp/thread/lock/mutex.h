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
#if defined(_MSC_VER)
// Umbrella intrinsics header: _mm_pause on x64/x86, __yield on ARM64. The
// ARM64 one is NOT in <immintrin.h>, and windows-11-arm is in the CI matrix.
#include <intrin.h>
#endif
#endif

namespace ctp {

/** CPU spin hint: relax the pipeline for a few tens of nanoseconds WITHOUT
 *  entering the kernel.
 *
 *  Deliberately not CTP_THREAD_MODEL->Yield(). That is only a pause under the
 *  Pthread model on x86/arm; under StdThread -- the default on Windows, which
 *  has no pthreads -- it is std::this_thread::yield(), i.e. a SwitchToThread
 *  SYSCALL of roughly 0.5-1us. Backoff pays rung 1 on every iteration of the
 *  ticket wait loop, so routing it through the thread model made the Windows
 *  spin ~30-50x more expensive than the same rung on Linux and turned
 *  ctp_priv_umap_ll's ~100k lock acquisitions into 143M spin iterations. */
CTP_INLINE_CROSS_FUN void CpuRelax() {
#if !CTP_IS_DEVICE_PASS
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
  _mm_pause();
#elif defined(_MSC_VER) && defined(_M_ARM64)
  __yield();
#elif defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#elif defined(__aarch64__) || defined(__arm__)
  __asm__ __volatile__("yield" ::: "memory");
#else
  // No pause hint on this ISA; a bare compiler barrier keeps the loop honest.
  __asm__ __volatile__("" ::: "memory");
#endif
#endif
}

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
    BackoffState st;
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
      Backoff(++spin_count, tkt - cur, st);
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
  /** PAUSE spins before escalating to a descheduling yield. */
  static constexpr u32 kSpinRung = 64;
  /** How long a waiter yield-spins before it is treated as genuinely stuck
   *  and parks.
   *
   *  A DURATION, not an iteration count, because the same count means wildly
   *  different amounts of waiting per platform: sched_yield() under contention
   *  on Linux costs ~1us, while SwitchToThread() on Windows returns in ~100ns
   *  when no other thread is ready. The previous fixed 1024 iterations
   *  therefore spanned ~1ms on Linux but only ~100us on Windows -- short
   *  enough that routine contention reached rung 3, and once one waiter in a
   *  FIFO ticket queue parks, every waiter behind it waits at least a park and
   *  parks too. That cascade is what pushed ctp_priv_umap_ll from 31s to a
   *  hang on windows-2025: ~1M parks in a run that needs a few thousand. */
  static constexpr min_u64 kYieldWindowUs = 2000;
  /** Yields between clock samples in rung 2. Must be a power of two. */
  static constexpr u32 kClockEvery = 64;
  /** Upper bound on the rung-3 park, in microseconds. */
  static constexpr min_u64 kParkUs = 100;

  /** Per-wait escalation state. Stack-local to Lock(), never shared, so it
   *  costs nothing in the uncontended case and needs no thread-local. */
  struct BackoffState {
#if !CTP_IS_DEVICE_PASS
    bool yielding = false;
    std::chrono::steady_clock::time_point yield_start{};
#endif
  };

  CTP_INLINE_CROSS_FUN
  void Backoff(u32 spin_count, min_u64 ahead, BackoffState &st) {
#if !CTP_IS_DEVICE_PASS
    // Rungs 2 and 3 are for OS-thread models only. Under a user-level-thread
    // model (Argobots) the model's own Yield() is ABT_thread_yield, which
    // deschedules the ULT and lets the holder run; sleeping the execution
    // stream there would block sibling ULTs, possibly the holder itself.
    const ThreadType ty = CTP_THREAD_MODEL->GetType();
    const bool os_thread =
        ty == ThreadType::kPthread || ty == ThreadType::kStdThread;
    // Rung 1 -- cheap CPU pause. An OS-thread model must NOT go through
    // CTP_THREAD_MODEL->Yield() here: under StdThread that is a SwitchToThread
    // syscall (see CpuRelax). A ULT model must, or the holder -- a sibling ULT
    // on this execution stream -- never gets scheduled at all.
    if (os_thread) {
      CpuRelax();
    } else {
      CTP_THREAD_MODEL->Yield();
    }
    if (spin_count < kSpinRung) {
      return;
    }
    if (!os_thread) {
      return;
    }
    // Rung 2 -- deschedule WITHOUT arming a timer. Rung 1 is a bare CPU pause
    // under both OS-thread models, so this is the first step that actually
    // hands the core to a runnable holder: std::this_thread::yield() is
    // sched_yield() on POSIX and SwitchToThread() on Windows. One syscall per
    // iteration is affordable here, where a wait has already proven itself
    // long; it was not at rung 1 -- see CpuRelax.
    //
    if (!st.yielding) {
      st.yielding = true;
      st.yield_start = std::chrono::steady_clock::now();
      std::this_thread::yield();
      return;
    }
    // Sample the clock every kClockEvery yields rather than every one. At the
    // ~100ns-1us cost of a yield that bounds the sampling error to a few tens
    // of microseconds against a millisecond-scale window, and keeps
    // steady_clock::now() off all but 1/kClockEvery of the wait iterations.
    if ((spin_count & (kClockEvery - 1)) != 0) {
      std::this_thread::yield();
      return;
    }
    if (std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - st.yield_start)
            .count() < static_cast<long long>(kYieldWindowUs)) {
      std::this_thread::yield();
      return;
    }
    // Rung 3 -- genuinely stuck (holder descheduled and not coming back
    // soon). Park, but at or above the hrtimer slack.
    //
    // The previous code slept min(ahead, 100)us here, which for the common
    // small `ahead` is a SUB-SLACK sleep: Linux rounds any sleep below the
    // hrtimer slack (~50us) up to it, so a "2us" backoff parked the successor
    // for ~50us, and FIFO order means the lock idles until precisely that
    // thread wakes. Same trap already documented in pthread.h's Yield().
    // Measured on a bare ticket lock, this rung structure against the old
    // one: 11-136x more throughput across 2..16 threads at both 20-iteration
    // and 500-iteration critical sections, with per-thread fairness unchanged
    // at 1.00. Pure spinning (no rungs 2/3) is faster still below core count
    // but collapses above it -- 420 ops vs 97k at 16 threads -- and destroys
    // fairness (up to 152x spread), which is the issue #483 livelock this
    // escalation exists to prevent.
    // Park scaled by queue position, as before. Routed through the thread
    // model rather than std::this_thread::sleep_for: on Windows the latter
    // rounds any sub-millisecond request up to the ~15.6ms timer tick, so a
    // "100us" park actually idled the whole FIFO queue for ~15ms.
    // SleepForUs uses a high-resolution waitable timer there (system_info.cc)
    // and nanosleep/usleep on POSIX, where behaviour is unchanged.
    min_u64 us = ahead < kParkUs ? ahead : kParkUs;
    if (us == 0) {
      us = 1;
    }
    CTP_THREAD_MODEL->SleepForUs(static_cast<size_t>(us));
#else
    (void)spin_count;
    (void)ahead;
    (void)st;
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
