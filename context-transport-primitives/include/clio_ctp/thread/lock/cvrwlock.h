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

// `CvRwLock` -- a fair reader/writer lock built on std::mutex +
// std::condition_variable, as an alternative to the spin-based ctp::RwLock.
//
// It is writer-fair: an arriving writer announces itself (waiting_writers_)
// which blocks NEW readers from entering, so a steady reader stream cannot
// starve writers. Threads block on the condition variable instead of spinning,
// so it trades the spin lock's low uncontended latency for zero CPU burn while
// waiting. This header exists primarily so the lock-latency benchmark can
// compare the two approaches head-to-head.
//
// HOST ONLY: std::mutex / std::condition_variable are not device-safe, so the
// class is compiled only on the host pass (mirrors mutex.h's <thread> guard).

#include "clio_ctp/constants/macros.h"

#if !CTP_IS_DEVICE_PASS
#include <condition_variable>
#include <mutex>

namespace ctp {

/**
 * Fair reader/writer lock using a condition variable.
 *
 * Method names mirror ctp::RwLock (ReadLock/ReadUnlock/WriteLock/WriteUnlock)
 * so it can be swapped in for comparison. Unlike ctp::RwLock, acquisition
 * blocks on a condition variable rather than spinning, and writers are given
 * priority over newly-arriving readers.
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

  /** Acquire the lock in shared (read) mode. Blocks while a writer holds the
   *  lock or is waiting, so writers are not starved by a reader stream. */
  void ReadLock() {
    std::unique_lock<std::mutex> lock(mtx_);
    cv_.wait(lock, [this]() {
      return !writer_active_ && waiting_writers_ == 0;
    });
    ++active_readers_;
  }

  /** Release a shared (read) hold. Wakes waiters once the last reader leaves. */
  void ReadUnlock() {
    std::unique_lock<std::mutex> lock(mtx_);
    --active_readers_;
    if (active_readers_ == 0) {
      cv_.notify_all();
    }
  }

  // -- writer side ----------------------------------------------------

  /** Acquire the lock in exclusive (write) mode. Announces its arrival so new
   *  readers back off, then waits until no reader or writer holds the lock. */
  void WriteLock() {
    std::unique_lock<std::mutex> lock(mtx_);
    ++waiting_writers_;  // announce arrival to stop new readers
    cv_.wait(lock, [this]() {
      return !writer_active_ && active_readers_ == 0;
    });
    --waiting_writers_;
    writer_active_ = true;
  }

  /** Release an exclusive (write) hold and wake all waiters. Waiting writers
   *  win the next round via the reader wait predicate (writer preference). */
  void WriteUnlock() {
    std::unique_lock<std::mutex> lock(mtx_);
    writer_active_ = false;
    cv_.notify_all();
  }

 private:
  std::mutex mtx_;
  std::condition_variable cv_;
  int active_readers_ = 0;
  bool writer_active_ = false;
  int waiting_writers_ = 0;
};

}  // namespace ctp

#endif  // !CTP_IS_DEVICE_PASS

#endif  // CTP_THREAD_CVRWLOCK_H_
