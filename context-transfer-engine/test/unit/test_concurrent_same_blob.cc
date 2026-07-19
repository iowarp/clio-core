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

// Regression test for the #680 same-blob write race (surfaced by xfstests
// generic/074 as fsx O_DIRECT content mismatches).
//
// Unlike test_concurrent_io_stress (each thread uses its OWN tag/blob, so it
// only stresses the shared allocator), here EVERY thread writes ONE shared
// blob at a distinct offset region. Concurrent PutBlob tasks for the same blob
// hash to the same container and interleave on ONE worker at each co_await in
// the write path (ExtendBlob, sparse-hole zero-fill, ModifyExistingData),
// racing on the blob's blocks_ vector (its block layout) and the size
// read-modify-write. Without serialization the shared blocks_ vector is mutated
// (push_back / realloc) by one task's ExtendBlob while another iterates it,
// corrupting the layout so region data lands in the wrong physical blocks — the
// read-back below then mismatches (or the daemon crashes).
//
// The fix is a per-blob async write token (BlobInfo::TryLockWrite/UnlockWrite)
// held across the whole read-modify-write in PutBlobImpl/TruncateBlob/DelBlob;
// on contention a task busy-polls via `co_await yield()`, which cannot deadlock
// the single worker (a thread-blocking lock would) and cannot lose a wakeup
// (it re-checks every worker iteration). With the token, each region reads back
// exactly its owning thread's byte pattern.
//
// Env knobs:
//   SAME_BLOB_THREADS (8)     concurrent writer threads (one region each)
//   SAME_BLOB_ITERS   (1500)  re-writes per thread
//   SAME_BLOB_CHUNK   (65536) bytes per region

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/bdev/bdev_client.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>

#include <algorithm>
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

const char *kTargetName = "same_blob_race_target";
constexpr clio::run::u64 kTargetSize = 2ULL * 1024 * 1024 * 1024;  // 2 GiB

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

TEST_CASE("ConcurrentSameBlob - many threads write disjoint regions of ONE "
          "blob; each region must survive intact (#680/generic074 repro)",
          "[cte][concurrent][same_blob]") {
  REQUIRE(g_fixture != nullptr);
  REQUIRE(g_fixture->initialized_);

  const int kThreads = FromEnv("SAME_BLOB_THREADS", 8);
  const int kIters = FromEnv("SAME_BLOB_ITERS", 1500);
  const clio::run::u64 kMinSz =
      static_cast<clio::run::u64>(FromEnv("SAME_BLOB_MINKB", 4)) * 1024;
  const clio::run::u64 kMaxSz =
      static_cast<clio::run::u64>(FromEnv("SAME_BLOB_MAXKB", 128)) * 1024;

  clio::cte::core::Tag tag("same_blob_race_tag");
  clio::cte::core::TagId tag_id = tag.GetTagId();
  const std::string blob = "shared";

  std::atomic<bool> failed{false};
  // Shared write frontier: each write carves a non-overlapping, EXTENDING region
  // [off, off+sz) out of the blob by fetch_add. Because regions never overlap
  // and always sit at the growing tail, every PutBlob extends the blob — so many
  // concurrent ExtendBlob calls mutate the SAME blocks_ vector at once, which is
  // exactly the #680/generic074 race. Content is a pure function of absolute
  // position (byte at position p == p & 0xff), so verification does not depend
  // on which thread wrote which region.
  std::atomic<clio::run::u64> frontier{0};

  auto writer = [&](int tid) {
    auto *ipc = CLIO_IPC;
    std::mt19937 rng(0xC0FFEE ^ (tid * 2654435761u));
    std::uniform_int_distribution<clio::run::u64> szdist(kMinSz, kMaxSz);
    for (int i = 0; i < kIters && !failed.load(std::memory_order_relaxed); ++i) {
      clio::run::u64 sz = szdist(rng);
      clio::run::u64 off = frontier.fetch_add(sz, std::memory_order_relaxed);
      ctp::ipc::FullPtr<char> wb = ipc->AllocateBuffer(sz);
      if (wb.IsNull()) { failed.store(true); return; }
      for (clio::run::u64 b = 0; b < sz; ++b)
        wb.ptr_[b] = static_cast<char>((off + b) & 0xff);
      auto put = tag.AsyncPutBlob(blob, wb.shm_.template Cast<void>(), sz, off,
                                  1.0f);
      put.Wait();
      if (put->GetReturnCode() != 0) { failed.store(true); ipc->FreeBuffer(wb); return; }
      ipc->FreeBuffer(wb);
    }
  };

  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; ++t) threads.emplace_back(writer, t);
  for (auto &t : threads) t.join();
  REQUIRE_FALSE(failed.load());

  // Every byte in [0, frontier) was written exactly once (contiguous carve), so
  // it must read back as (position & 0xff). A corrupted block layout shows up
  // here as bytes that belong to a different region (wrong low byte) or zero.
  const clio::run::u64 total = frontier.load();
  REQUIRE(total > 0);
  auto *ipc = CLIO_IPC;
  const clio::run::u64 kReadWin = kMaxSz;
  clio::run::u64 total_mismatches = 0;
  for (clio::run::u64 base = 0; base < total; base += kReadWin) {
    clio::run::u64 win = std::min(kReadWin, total - base);
    ctp::ipc::FullPtr<char> rb = ipc->AllocateBuffer(win);
    REQUIRE_FALSE(rb.IsNull());
    std::memset(rb.ptr_, 0xAA, win);
    auto got = CLIO_CTE_CLIENT->AsyncGetBlob(
        tag_id, blob, base, win, 0u, rb.shm_.template Cast<void>(),
        clio::run::PoolQuery::Local());
    got.Wait();
    REQUIRE(got->GetReturnCode() == 0);
    for (clio::run::u64 b = 0; b < win; ++b) {
      unsigned char v = static_cast<unsigned char>(rb.ptr_[b]);
      unsigned char expect = static_cast<unsigned char>((base + b) & 0xff);
      if (v != expect) {
        if (total_mismatches == 0) {
          HLOG(kError,
               "[same_blob] FIRST mismatch at pos={} expected=0x{:02x} "
               "got=0x{:02x} (total_bytes={})",
               base + b, static_cast<int>(expect), static_cast<int>(v), total);
        }
        ++total_mismatches;
      }
    }
    ipc->FreeBuffer(rb);
  }
  if (total_mismatches != 0) {
    HLOG(kError, "[same_blob] {} mismatched bytes out of {}", total_mismatches,
         total);
  }
  REQUIRE(total_mismatches == 0);
}

int main(int argc, char **argv) {
  g_fixture = new Fixture();
  std::string filter = (argc > 1) ? argv[1] : "";
  int rc = SimpleTest::run_all_tests(filter);
  delete g_fixture;
  g_fixture = nullptr;
  return rc;
}
