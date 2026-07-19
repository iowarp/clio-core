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

// Minimal reproducer for the #680 concurrency crash seen through fsx
// (generic/075,112,127 crash the FUSE daemon). fsx is single-threaded but the
// FUSE daemon is fuse_loop_mt, and mmap async writeback issues concurrent
// cte_fuse_write on multiple FUSE threads; single-threaded FUSE eliminates the
// crash, so the trigger is concurrent client-thread I/O against the SHARED
// MultiProcessAllocator (both the client threads and the runtime workers run in
// the embedded-runtime process and draw buffers from the same allocator).
//
// This test cuts out FUSE + fsx entirely: an embedded runtime (CLIO_INIT
// kClient,true, same mode as clio_cte_fuse) plus N threads that repeatedly
// AllocateBuffer(random size) -> PutBlob -> GetBlob -> FreeBuffer, hammering the
// shared allocator directly.
//
// Env knobs for isolating the failure:
//   STRESS_THREADS (8)   number of concurrent client threads
//   STRESS_ITERS   (4000) iterations per thread
//   STRESS_IO      (1)   1 = run PutBlob/GetBlob/DelBlob; 0 = alloc/free only
//   STRESS_MEMSET  (1)   1 = fill buffers; 0 = leave untouched
//   STRESS_MINKB/STRESS_MAXKB (0/128) bound allocation sizes (KiB)
//
// Two distinct bugs surface here. (1) a now-fixed buddy free-list bug in
// RepopulateSmallArena (raw page->size_ carrying kFreeMask). (2) the dominant,
// still-open failure: a shared_ptr<Task>/Future completion race between the
// client-submit path and the worker that processes the task (TSan flags
// concurrent shared_ptr<Task>::Release on the same object). Run under TSan
// (setarch $(uname -m) -R) to observe (2).

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/bdev/bdev_client.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "simple_test.h"

namespace {

const char *kTargetName = "concurrent_stress_target";
constexpr clio::run::u64 kTargetSize = 2ULL * 1024 * 1024 * 1024;  // 2 GiB
constexpr clio::run::u64 kMaxBlob = 128 * 1024;                    // 128 KiB

int FromEnv(const char *name, int dflt) {
  if (const char *e = std::getenv(name)) {
    int n = std::atoi(e);
    if (n > 0) return n;
  }
  return dflt;
}

class Fixture {
 public:
  bool initialized_ = false;
  Fixture() {
    bool ok = clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
    REQUIRE(ok);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    ok = clio::cte::core::CLIO_CTE_CLIENT_INIT();
    REQUIRE(ok);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    auto *cte = CLIO_CTE_CLIENT;
    clio::run::PoolId bdev_pool_id(901, 0);
    clio::run::bdev::Client bdev_client(bdev_pool_id);
    auto create = bdev_client.AsyncCreate(
        clio::run::PoolQuery::Dynamic(), kTargetName, bdev_pool_id,
        clio::run::bdev::BdevType::kRam, kTargetSize);
    create.Wait();
    auto reg = cte->AsyncRegisterTarget(kTargetName,
                                        clio::run::bdev::BdevType::kRam,
                                        kTargetSize, clio::run::PoolQuery::Local(),
                                        bdev_pool_id);
    reg.Wait();
    REQUIRE(reg->GetReturnCode() == 0);
    initialized_ = true;
  }
};

Fixture *g_fixture = nullptr;

}  // namespace

TEST_CASE("ConcurrentIoStress - many threads Put/Get/Free hammer the shared "
          "allocator (#680 repro)",
          "[cte][concurrent][stress]") {
  REQUIRE(g_fixture != nullptr);
  REQUIRE(g_fixture->initialized_);

  const int kThreads = FromEnv("STRESS_THREADS", 8);
  const int kIters = FromEnv("STRESS_ITERS", 4000);
  // STRESS_IO=0 bisects the crash: skip all Put/Get/Del so only the
  // AllocateBuffer/memset/FreeBuffer allocator churn runs. If it still
  // crashes with IO off, the bug is the allocator concurrency itself; if
  // it only crashes with IO on, the bug is the bdev/ToFullPtr I/O path.
  const bool kDoIo = FromEnv("STRESS_IO", 1) != 0;
  // STRESS_MEMSET=0 skips filling the buffer. If the crash needs memset, the
  // allocator handed back fewer usable bytes than requested (OOB write); if it
  // crashes without memset too, the corruption is pure alloc/free bookkeeping.
  const bool kDoMemset = FromEnv("STRESS_MEMSET", 1) != 0;

  // STRESS_POOL=N: single-threaded pool mode. Keep up to N buffers outstanding
  // and free them in a NON-LIFO order, reproducing the many-outstanding-buffers
  // allocation pattern that concurrency produces — but single-threaded and
  // DETERMINISTIC, to get a debuggable repro of the buddy corruption (#680).
  const int kPool = FromEnv("STRESS_POOL", 0);
  if (kPool > 0) {
    auto *ipc = CLIO_IPC;
    std::vector<ctp::ipc::FullPtr<char>> slots(kPool);
    for (int i = 0; i < kPool; ++i) slots[i] = ctp::ipc::FullPtr<char>::GetNull();
    std::vector<clio::run::u64> szs(kPool, 0);
    std::mt19937 rng(20260703u);
    std::uniform_int_distribution<clio::run::u64> sizedist(1, kMaxBlob);
    std::uniform_int_distribution<int> slotdist(0, kPool - 1);
    const long kTotal = static_cast<long>(kIters) * kThreads;  // same op budget
    for (long i = 0; i < kTotal; ++i) {
      int s = slotdist(rng);
      if (slots[s].IsNull()) {
        clio::run::u64 sz = sizedist(rng);
        slots[s] = ipc->AllocateBuffer(sz);
        if (slots[s].IsNull()) { REQUIRE(false); }
        szs[s] = sz;
        if (kDoMemset) std::memset(slots[s].ptr_, static_cast<int>(i), sz);
      } else {
        ipc->FreeBuffer(slots[s]);
        slots[s] = ctp::ipc::FullPtr<char>::GetNull();
      }
    }
    for (int s = 0; s < kPool; ++s)
      if (!slots[s].IsNull()) ipc->FreeBuffer(slots[s]);
    REQUIRE(true);
    return;
  }

  std::atomic<bool> failed{false};
  std::atomic<long> total_ops{0};

  auto worker = [&](int tid) {
    auto *ipc = CLIO_IPC;
    clio::cte::core::Tag tag("stress_tag_" + std::to_string(tid));
    clio::cte::core::TagId tag_id = tag.GetTagId();
    std::mt19937 rng(1234 + tid);
    // STRESS_MINKB/STRESS_MAXKB bound the allocation sizes (KiB) to bisect the
    // small (<16KB arena) vs large (>16KB) buddy paths.
    clio::run::u64 lo = static_cast<clio::run::u64>(FromEnv("STRESS_MINKB", 0)) * 1024;
    clio::run::u64 hi = static_cast<clio::run::u64>(FromEnv("STRESS_MAXKB", 128)) * 1024;
    if (lo < 1) lo = 1;
    if (hi > kMaxBlob) hi = kMaxBlob;
    if (hi < lo) hi = lo;
    std::uniform_int_distribution<clio::run::u64> sizedist(lo, hi);

    for (int i = 0; i < kIters && !failed.load(std::memory_order_relaxed); ++i) {
      const clio::run::u64 sz = sizedist(rng);
      const std::string blob = "b" + std::to_string(i % 16);

      // Allocate + fill + PutBlob (write path).
      ctp::ipc::FullPtr<char> wb = ipc->AllocateBuffer(sz);
      if (wb.IsNull()) { failed.store(true); return; }
      if (kDoMemset) std::memset(wb.ptr_, static_cast<int>(tid + i), sz);
      if (kDoIo) {
        auto put = tag.AsyncPutBlob(blob, wb.shm_.template Cast<void>(), sz, 0,
                                    1.0f);
        put.Wait();
      }
      ipc->FreeBuffer(wb);

      // GetBlob (read path) into a freshly allocated buffer.
      ctp::ipc::FullPtr<char> rb = ipc->AllocateBuffer(sz);
      if (rb.IsNull()) { failed.store(true); return; }
      if (kDoMemset) std::memset(rb.ptr_, static_cast<int>(tid - i), sz);
      if (kDoIo) {
        auto got = CLIO_CTE_CLIENT->AsyncGetBlob(
            tag_id, blob, 0, sz, 0u, rb.shm_.template Cast<void>(),
            clio::run::PoolQuery::Local());
        got.Wait();
      }
      ipc->FreeBuffer(rb);

      // Occasionally delete a blob to churn metadata + block frees.
      if (kDoIo && (i & 31) == 0) {
        auto del = CLIO_CTE_CLIENT->AsyncDelBlob(tag_id, blob);
        del.Wait();
      }
      total_ops.fetch_add(1, std::memory_order_relaxed);
    }
  };

  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; ++t) threads.emplace_back(worker, t);
  for (auto &t : threads) t.join();

  // Reaching here without a crash / allocator corruption is the pass condition.
  REQUIRE_FALSE(failed.load());
  REQUIRE(total_ops.load() > 0);
}

int main(int argc, char **argv) {
  g_fixture = new Fixture();
  std::string filter = (argc > 1) ? argv[1] : "";
  int rc = SimpleTest::run_all_tests(filter);
  delete g_fixture;
  g_fixture = nullptr;
  return rc;
}
