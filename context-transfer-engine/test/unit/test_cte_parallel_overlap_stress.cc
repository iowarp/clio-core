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

// Parallel blob-metadata overlap stress for the CTE core.
//
// Many threads hammer PutBlob / GetBlob / DelBlob / GetBlobInfo at random over a
// DELIBERATELY TINY key space -- only 4 tags with 4 blob names each (16 blobs
// total) -- so every operation contends with concurrent operations on the SAME
// (tag, blob). This maximizes overlap on the shared metadata (tag blob maps,
// blob-block lists, the shared allocator) to shake out races, use-after-free,
// and deadlocks. clio exposes no literal "GetBlobId": a blob's identity IS
// (tag_id, blob_name); AsyncGetBlobInfo is the metadata resolver for a named
// blob, so it stands in for GetBlobId here.
//
// Success metric (per the request): NO crashes, NO hangs, NO memory leaks.
//   - crashes  -> the process aborts / segfaults and the test fails to finish.
//   - hangs    -> every op is a bounded Async+Wait and all threads join; a hang
//                 trips the ctest per-test TIMEOUT.
//   - leaks    -> every buffer this test allocates is freed on every path; run
//                 under ASan/MSan (CI/run_sanitizers.sh) to enforce.
// Individual ops are NOT required to succeed: a GetBlob/GetBlobInfo/DelBlob may
// legitimately race a concurrent DelBlob and return non-zero. We only assert the
// system stays alive and responsive.
//
// Env knobs: STRESS_THREADS (8), STRESS_ITERS (4000).

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

const char *kTargetName = "parallel_overlap_target";
constexpr clio::run::u64 kTargetSize = 2ULL * 1024 * 1024 * 1024;  // 2 GiB
constexpr int kNumTags = 4;
constexpr int kBlobsPerTag = 4;
constexpr clio::run::u64 kMaxBlob = 64 * 1024;  // 64 KiB

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
  std::vector<clio::cte::core::TagId> tag_ids_;
  Fixture() {
    bool ok = clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
    REQUIRE(ok);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    ok = clio::cte::core::CLIO_CTE_CLIENT_INIT();
    REQUIRE(ok);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    auto *cte = CLIO_CTE_CLIENT;
    clio::run::PoolId bdev_pool_id(902, 0);
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

    // Pre-create the 4 shared tags so every thread resolves the SAME tag ids.
    for (int t = 0; t < kNumTags; ++t) {
      clio::cte::core::Tag tag("overlap_tag_" + std::to_string(t));
      tag_ids_.push_back(tag.GetTagId());
    }
    initialized_ = true;
  }
};

Fixture *g_fixture = nullptr;

}  // namespace

TEST_CASE("CteParallelOverlapStress - Put/Get/Del/GetInfo race on 4 tags x 4 "
          "blobs with heavy overlap (no crash/hang/leak)",
          "[cte][concurrent][stress]") {
  REQUIRE(g_fixture != nullptr);
  REQUIRE(g_fixture->initialized_);

  const int kThreads = FromEnv("STRESS_THREADS", 8);
  const int kIters = FromEnv("STRESS_ITERS", 4000);
  const auto &tag_ids = g_fixture->tag_ids_;

  std::atomic<bool> failed{false};
  std::atomic<long> total_ops{0};

  auto worker = [&](int tid) {
    auto *ipc = CLIO_IPC;
    std::mt19937 rng(0xC0FFEE ^ (tid * 2654435761u));
    std::uniform_int_distribution<int> tagdist(0, kNumTags - 1);
    std::uniform_int_distribution<int> blobdist(0, kBlobsPerTag - 1);
    std::uniform_int_distribution<int> opdist(0, 3);
    std::uniform_int_distribution<clio::run::u64> sizedist(1, kMaxBlob);

    for (int i = 0; i < kIters && !failed.load(std::memory_order_relaxed); ++i) {
      const clio::cte::core::TagId tag_id = tag_ids[tagdist(rng)];
      const std::string blob = "b" + std::to_string(blobdist(rng));
      const int op = opdist(rng);

      switch (op) {
        case 0: {  // PutBlob (write) — establishes/overwrites (tag, blob)
          const clio::run::u64 sz = sizedist(rng);
          ctp::ipc::FullPtr<char> wb = ipc->AllocateBuffer(sz);
          if (wb.IsNull()) { failed.store(true); return; }
          std::memset(wb.ptr_, static_cast<int>(tid + i), sz);
          clio::cte::core::Tag tag(tag_id);
          auto put = tag.AsyncPutBlob(blob, wb.shm_.template Cast<void>(), sz, 0,
                                      1.0f);
          put.Wait();
          ipc->FreeBuffer(wb);
          break;
        }
        case 1: {  // GetBlob (read) — may race a concurrent DelBlob (ok)
          const clio::run::u64 sz = sizedist(rng);
          ctp::ipc::FullPtr<char> rb = ipc->AllocateBuffer(sz);
          if (rb.IsNull()) { failed.store(true); return; }
          std::memset(rb.ptr_, 0, sz);
          auto got = CLIO_CTE_CLIENT->AsyncGetBlob(
              tag_id, blob, 0, sz, 0u, rb.shm_.template Cast<void>(),
              clio::run::PoolQuery::Local());
          got.Wait();
          ipc->FreeBuffer(rb);
          break;
        }
        case 2: {  // DelBlob (delete)
          auto del = CLIO_CTE_CLIENT->AsyncDelBlob(tag_id, blob);
          del.Wait();
          break;
        }
        case 3: {  // GetBlobInfo — the blob-id/metadata resolver
          auto info = CLIO_CTE_CLIENT->AsyncGetBlobInfo(
              tag_id, blob, clio::run::PoolQuery::Local());
          info.Wait();
          break;
        }
        default:
          break;
      }
      total_ops.fetch_add(1, std::memory_order_relaxed);
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) threads.emplace_back(worker, t);
  for (auto &t : threads) t.join();

  // Reaching here without a crash / deadlock is the pass condition.
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
