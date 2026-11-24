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
 * Regression tests for ctp::GetGlobalPtrVar (issue #929).
 *
 * The accessor used to be an unsynchronized check-then-set:
 *
 *   if (instance == nullptr) { instance = new T(); }
 *
 * Two threads could both observe null, each construct a T, and the last
 * writer would win the variable -- leaving the other instance orphaned and,
 * worse, leaving different callers holding different instances.
 *
 * That is what produced the "two PoolManager addresses in one process"
 * signature behind issues #923, #928 and #929: PoolManager::ServerInit()
 * initialized one instance while every later CLIO_POOL_MANAGER lookup
 * returned a different, permanently-blank one, so the route-retry loop spun
 * its whole 30s deadline and failed the task with (u32)-1.
 *
 * These tests race many threads on a cold pointer and assert the two
 * properties that failure violated: everyone agrees on one pointer, and
 * exactly one instance stays alive.
 */

#include <atomic>
#include <thread>
#include <vector>

#include "basic_test.h"
#include "clio_ctp/util/singleton.h"

namespace {

/** Counts how many instances exist so a leaked loser is detectable. */
std::atomic<int> g_live_count{0};
std::atomic<int> g_ctor_count{0};

struct CountedSingleton {
  // A payload wide enough that a torn publish would be observable: a thread
  // that saw the pointer before the constructor's writes would read zeros.
  static constexpr int kSentinel = 0x5EED;
  int sentinel_;

  CountedSingleton() : sentinel_(kSentinel) {
    g_live_count.fetch_add(1, std::memory_order_relaxed);
    g_ctor_count.fetch_add(1, std::memory_order_relaxed);
  }

  ~CountedSingleton() { g_live_count.fetch_sub(1, std::memory_order_relaxed); }
};

}  // namespace

TEST_CASE("GetGlobalPtrVarPublishesExactlyOneInstance") {
  // Many rounds: the race window is narrow, so a single round proves little.
  // On the pre-fix code this fails within the first few rounds on a
  // multi-core box.
  // 50 rounds is already overwhelming evidence: measured standalone, the
  // pre-fix accessor splits threads across instances in ~95% of contended
  // rounds, so a real regression fails within the first few. 200 rounds x 8
  // threads meant 1,600 thread creations, which timed out the whole test on
  // the Windows Debug runner.
  const int kRounds = 50;
  const int kThreads = 8;

  for (int round = 0; round < kRounds; ++round) {
    g_live_count.store(0, std::memory_order_relaxed);
    g_ctor_count.store(0, std::memory_order_relaxed);

    // A fresh cold pointer per round -- this is the state a process is in
    // before the first CLIO_POOL_MANAGER / CLIO_IPC access.
    std::atomic<CountedSingleton *> instance{nullptr};

    std::vector<CountedSingleton *> observed(kThreads, nullptr);
    std::atomic<int> ready{0};
    std::vector<std::thread> threads;
    threads.reserve(kThreads);

    for (int t = 0; t < kThreads; ++t) {
      threads.emplace_back([&, t]() {
        // Barrier so the threads reach the accessor together; without it they
        // serialize and the race never opens.
        //
        // yield() is not optional here. kThreads oversubscribes a 2-vCPU CI
        // runner, and a pure spin means the threads that arrived first burn
        // both cores while the ones that still have to increment `ready`
        // cannot get scheduled to do so -- progress depends on the OS
        // preempting a spinner. That is what timed out ctp_singleton on
        // windows-2022 Debug; yielding hands the core to the thread the
        // barrier is actually waiting for.
        ready.fetch_add(1, std::memory_order_acq_rel);
        while (ready.load(std::memory_order_acquire) < kThreads) {
          std::this_thread::yield();
        }
        observed[t] = ctp::GetGlobalPtrVar<CountedSingleton>(instance);
      });
    }
    for (auto &th : threads) {
      th.join();
    }

    CountedSingleton *winner = instance.load(std::memory_order_acquire);
    REQUIRE(winner != nullptr);

    // (1) Every thread must have been handed the instance that is actually
    //     published. This is the property whose failure stranded ServerInit
    //     on an orphan.
    for (int t = 0; t < kThreads; ++t) {
      REQUIRE(observed[t] == winner);
    }

    // (2) Exactly one instance may remain alive. Losing a publish race is
    //     allowed to construct speculatively, but the loser must be
    //     destroyed rather than orphaned.
    REQUIRE(g_live_count.load(std::memory_order_relaxed) == 1);

    // (3) The published object must be fully constructed -- the acquire on
    //     the pointer is what orders the constructor's writes ahead of
    //     another thread's first dereference.
    REQUIRE(winner->sentinel_ == CountedSingleton::kSentinel);

    delete winner;
    REQUIRE(g_live_count.load(std::memory_order_relaxed) == 0);
  }
}

TEST_CASE("GetGlobalPtrVarIsStableOnceWarm") {
  std::atomic<CountedSingleton *> instance{nullptr};
  g_live_count.store(0, std::memory_order_relaxed);

  CountedSingleton *first = ctp::GetGlobalPtrVar<CountedSingleton>(instance);
  REQUIRE(first != nullptr);

  // A warm pointer must never construct again, no matter how many callers
  // or threads ask for it.
  for (int i = 0; i < 1000; ++i) {
    REQUIRE(ctp::GetGlobalPtrVar<CountedSingleton>(instance) == first);
  }
  REQUIRE(g_live_count.load(std::memory_order_relaxed) == 1);

  delete first;
}
