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

#ifndef CTP_THREAD_CVRWLOCK_H_
#define CTP_THREAD_CVRWLOCK_H_

// `CvRwLock` -- a writer-preferring reader/writer lock that blocks (parks) on a
// condition variable instead of spinning, as a sleep-friendly alternative to
// the spin-based ctp::RwLock.
//
// Readers are PARALLEL: the common no-writer read path is lock-free (a single
// atomic increment on `readers_` guarded by an optimistic re-check of the
// writer-intent counter) and never touches the mutex, so concurrent readers do
// not serialize on it. The mutex + condition variable are used only on the slow
// path -- when a reader must park behind a writer, or when a writer must wait
// for readers to drain. (The earlier version took the mutex on every
// ReadLock/ReadUnlock, which serialized readers and defeated the point of a
// shared lock.)
//
// Writer preference: a writer bumps `write_intent_` on arrival, which makes new
// readers back off, so a steady reader stream cannot starve writers.
//
// HOST ONLY: std::atomic is fine on device, but std::mutex / condition_variable
// are not, so the class is compiled only on the host pass (mirrors mutex.h).

#include "clio_ctp/constants/macros.h"

#if !CTP_IS_DEVICE_PASS
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>

namespace ctp {

/**
 * Writer-preferring reader/writer lock with a lock-free parallel reader path
 * and condition-variable blocking on the slow path.
 *
 * Method names mirror ctp::RwLock (ReadLock/ReadUnlock/WriteLock/WriteUnlock)
 * so it can be swapped in for comparison.
 */
class CvRwLock {
 public:
  CvRwLock() = default;

  /** Non-copyable / non-movable (owns a std::mutex + condition_variable). */
  CvRwLock(const CvRwLock &) = delete;
  CvRwLock &operator=(const CvRwLock &) = delete;

  /** No-op initializer, for API parity with ctp::RwLock. */
  void Init() {}

  // -- reader side ----------------------------------------------------

  /** Acquire the lock in shared (read) mode. Lock-free when no writer is
   *  waiting/active; otherwise parks on the condition variable. */
  void ReadLock() {
    for (;;) {
      if (write_intent_.load() == 0) {
        // Optimistically register as a reader, then re-check for a writer that
        // may have arrived in between. If none, we hold the lock -- no mutex.
        readers_.fetch_add(1);
        if (write_intent_.load() == 0) {
          return;
        }
        // A writer appeared; back out. If we were the last reader, wake it.
        if (readers_.fetch_sub(1) == 1) {
          std::lock_guard<std::mutex> lk(mtx_);
          cv_.notify_all();
        }
      }
      // Slow path: park until no writer is waiting/active, then retry.
      std::unique_lock<std::mutex> lk(mtx_);
      cv_.wait(lk, [this]() { return write_intent_.load() == 0; });
    }
  }

  /** Release a shared (read) hold. Lock-free unless we are the last reader and
   *  a writer is waiting, in which case we wake it. */
  void ReadUnlock() {
    if (readers_.fetch_sub(1) == 1 && write_intent_.load() > 0) {
      std::lock_guard<std::mutex> lk(mtx_);
      cv_.notify_all();
    }
  }

  // -- writer side ----------------------------------------------------

  /** Acquire the lock in exclusive (write) mode. Announces intent so new
   *  readers back off, then waits until no writer holds it and readers drain. */
  void WriteLock() {
    write_intent_.fetch_add(1);  // block new readers (writer preference)
    std::unique_lock<std::mutex> lk(mtx_);
    cv_.wait(lk, [this]() {
      return !writer_active_ && readers_.load() == 0;
    });
    writer_active_ = true;
  }

  /** Release an exclusive (write) hold; wake the next writer and/or readers. */
  void WriteUnlock() {
    std::lock_guard<std::mutex> lk(mtx_);
    writer_active_ = false;
    write_intent_.fetch_sub(1);  // re-admit readers if no other writer waits
    cv_.notify_all();
  }

 private:
  // Active readers. Modified lock-free on the reader fast path; read under the
  // mutex by a waiting writer's predicate.
  std::atomic<std::int64_t> readers_{0};
  // Writers waiting OR active. Readers back off (do not enter) while > 0.
  std::atomic<std::int64_t> write_intent_{0};
  std::mutex mtx_;
  std::condition_variable cv_;
  bool writer_active_ = false;  // guarded by mtx_
};

}  // namespace ctp

#endif  // !CTP_IS_DEVICE_PASS

#endif  // CTP_THREAD_CVRWLOCK_H_
