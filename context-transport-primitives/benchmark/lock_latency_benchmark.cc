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

/**
 * Lock Latency Benchmark
 *
 * Measures the raw single-threaded (uncontended) lock+unlock latency of the
 * context-transport-primitives locking implementations. For each lock type it
 * repeats a lock+unlock pair in a tight loop for a fixed duration and reports
 * the average time per pair.
 *
 * Cases:
 *   - ctp::Mutex        Lock + Unlock
 *   - ctp::RwLock       ReadLock + ReadUnlock
 *   - ctp::RwLock       WriteLock + WriteUnlock
 *   - ctp::CvRwLock     ReadLock + ReadUnlock   (condition-variable based)
 *   - ctp::CvRwLock     WriteLock + WriteUnlock (condition-variable based)
 *
 * Usage:
 *   lock_latency_benchmark [seconds_per_case]
 *   (default: 2 seconds per case)
 */

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

#include "clio_ctp/thread/lock/cvrwlock.h"
#include "clio_ctp/thread/lock/mutex.h"
#include "clio_ctp/thread/lock/rwlock.h"

namespace {

using Clock = std::chrono::steady_clock;

/**
 * Run `lock()`/`unlock()` back-to-back for `seconds`, counting completed pairs.
 * The loop is batched so the clock is only read once per batch (amortizing the
 * timer cost out of the measured latency). Prints a formatted result row.
 */
template <typename LockFn, typename UnlockFn>
void RunCase(const std::string &name, double seconds, LockFn lock,
             UnlockFn unlock) {
  constexpr uint64_t kBatch = 4096;
  const auto deadline =
      Clock::now() + std::chrono::duration_cast<Clock::duration>(
                         std::chrono::duration<double>(seconds));

  uint64_t ops = 0;
  const auto start = Clock::now();
  while (Clock::now() < deadline) {
    for (uint64_t i = 0; i < kBatch; ++i) {
      lock();
      unlock();
    }
    ops += kBatch;
  }
  const auto elapsed = Clock::now() - start;

  const double elapsed_ns =
      std::chrono::duration<double, std::nano>(elapsed).count();
  const double ns_per_op = ops ? elapsed_ns / static_cast<double>(ops) : 0.0;
  const double mops_per_sec = ns_per_op > 0.0 ? 1000.0 / ns_per_op : 0.0;

  std::cout << std::left << std::setw(34) << name << std::right
            << std::setw(15) << ops << std::setw(14) << std::fixed
            << std::setprecision(2) << ns_per_op << std::setw(14)
            << std::setprecision(2) << mops_per_sec << "\n";
}

}  // namespace

int main(int argc, char **argv) {
  double seconds = 2.0;
  if (argc > 1) {
    seconds = std::atof(argv[1]);
    if (seconds <= 0.0) seconds = 2.0;
  }

  std::cout << "Lock latency benchmark (single-threaded, uncontended)\n";
  std::cout << "Duration per case: " << seconds << " s\n\n";
  std::cout << std::left << std::setw(34) << "case" << std::right
            << std::setw(15) << "ops" << std::setw(14) << "ns/op"
            << std::setw(14) << "M ops/s" << "\n";
  std::cout << std::string(77, '-') << "\n";

  {
    ctp::Mutex mtx;
    RunCase(
        "ctp::Mutex Lock+Unlock", seconds, [&]() { mtx.Lock(0); },
        [&]() { mtx.Unlock(); });
  }
  {
    ctp::RwLock rw;
    RunCase(
        "ctp::RwLock ReadLock+Unlock", seconds, [&]() { rw.ReadLock(0); },
        [&]() { rw.ReadUnlock(); });
    RunCase(
        "ctp::RwLock WriteLock+Unlock", seconds, [&]() { rw.WriteLock(0); },
        [&]() { rw.WriteUnlock(); });
  }
  {
    ctp::CvRwLock cv;
    RunCase(
        "ctp::CvRwLock ReadLock+Unlock", seconds, [&]() { cv.ReadLock(); },
        [&]() { cv.ReadUnlock(); });
    RunCase(
        "ctp::CvRwLock WriteLock+Unlock", seconds, [&]() { cv.WriteLock(); },
        [&]() { cv.WriteUnlock(); });
  }

  std::cout << "\nNote: measures uncontended lock+unlock pair cost; it does not\n"
            << "capture fairness or behavior under multi-threaded contention.\n";
  return 0;
}
