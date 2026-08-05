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
 * Concurrent data-plane regression test for the safe_bdev ChiMod (issue #909).
 *
 * A container's methods are NOT serialized. Client threads submit independently
 * and the scheduler places each task on the least-loaded worker, so several
 * workers can execute inside the SAME safe_bdev container at once. This test
 * drives Allocate -> Write -> Free from many threads and asserts that every
 * slot handed out is returned.
 *
 * Without synchronization on data_alloc_/rr_cursor_, concurrent Take() and
 * Release() lose std::set updates: FreeBlocks reports "block ... not live" and
 * SKIPS the release (still returning rc=0), so slots leak and the post-run
 * capacity comes back short. The same race corrupts the heap outright -- the
 * unfixed module aborts with "malloc(): unaligned tcache chunk" inside
 * safe_bdev::Runtime::Write. The capacity comparison is the assertion; a crash
 * fails the test just as loudly.
 *
 * This lives in its own binary because reproducing the race needs more workers
 * than the default 4 (with 4, only ~2 are general-purpose and every task lands
 * on the same one). The ctest entry sets CLIO_NUM_THREADS accordingly.
 */

#ifndef _WIN32
#include <unistd.h>
#endif

#include <chrono>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "simple_test.h"

using namespace std::chrono_literals;

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/pool_query.h>
#include <clio_runtime/singletons.h>
#include <clio_runtime/types.h>

#include <clio_runtime/bdev/bdev_client.h>
#include <clio_runtime/bdev/bdev_tasks.h>

#include <clio_runtime/safe_bdev/safe_bdev_client.h>
#include <clio_runtime/safe_bdev/safe_bdev_tasks.h>

#include <clio_runtime/admin/admin_client.h>
#include <clio_runtime/admin/admin_tasks.h>

namespace {

bool g_initialized = false;

/** Per-member RAM size; 4 MiB holds many 64 KiB slots. */
constexpr clio::run::u64 kMemberRamSize = 4 * 1024 * 1024;
/** Mirrors Runtime::kChunkLen. */
constexpr clio::run::u64 kChunkLen = 65536;

/** Initialize Clio once for this test binary (runtime co-located). */
void EnsureInit() {
  if (!g_initialized) {
    bool success = clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
    if (success) {
      g_initialized = true;
      SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;
      std::this_thread::sleep_for(500ms);
    }
  }
}

/** Create a RAM-backed member bdev pool. */
bool CreateRamMember(clio::run::bdev::Client &client,
                     const std::string &pool_name,
                     const clio::run::PoolId &pool_id) {
  auto create_task =
      client.AsyncCreate(clio::run::PoolQuery::Dynamic(), pool_name, pool_id,
                         clio::run::bdev::BdevType::kRam, kMemberRamSize);
  create_task.Wait();
  client.pool_id_ = create_task->new_pool_id_;
  client.return_code_ = create_task->return_code_;
  return create_task->GetReturnCode() == 0;
}

/** A repeating, position-dependent byte pattern. */
std::vector<ctp::u8> MakePattern(size_t size, ctp::u8 seed) {
  std::vector<ctp::u8> v(size);
  for (size_t i = 0; i < size; ++i) {
    v[i] = static_cast<ctp::u8>((seed + i * 7 + (i >> 8)) & 0xFF);
  }
  return v;
}

}  // namespace

TEST_CASE("safe_bdev_concurrent_data_plane",
          "[safe_bdev][concurrency][regression]") {
  EnsureInit();
  REQUIRE(g_initialized);
  std::this_thread::sleep_for(100ms);

  const int pidsalt = static_cast<int>(getpid() & 0xFFF);
  auto member_name = [&](int idx) {
    return "cc_member_" + std::to_string(getpid()) + "_" + std::to_string(idx);
  };

  const int k = 4;
  std::vector<clio::run::PoolId> data_ids;
  for (int c = 0; c < k; ++c) {
    clio::run::PoolId id(static_cast<clio::run::u32>(9500 + pidsalt + c), 0);
    clio::run::bdev::Client client(id);
    REQUIRE(CreateRamMember(client, member_name(c), id));
    data_ids.push_back(client.pool_id_);
  }

  clio::run::PoolId safe_id(static_cast<clio::run::u32>(9550 + pidsalt), 0);
  clio::run::safe_bdev::Client safe(safe_id);
  std::vector<clio::run::safe_bdev::MemberBdevDesc> members;
  for (int c = 0; c < k; ++c) {
    members.emplace_back(member_name(c), /*node_id=*/0, data_ids[c]);
  }
  auto create_task = safe.AsyncCreate(clio::run::PoolQuery::Dynamic(),
                                      "safe_bdev_cc_pool", safe_id,
                                      /*max_failures=*/1, members);
  create_task.Wait();
  safe.pool_id_ = create_task->new_pool_id_;
  REQUIRE(create_task->GetReturnCode() == 0);

  // Baseline capacity: every slot this test allocates must be back in the
  // per-member free lists by the end.
  auto stats0 = safe.AsyncGetStats();
  stats0.Wait();
  REQUIRE(stats0->GetReturnCode() == 0);
  const clio::run::u64 remaining_before = stats0->remaining_size_;
  REQUIRE(remaining_before > 0);

  // 16 threads keeps enough requests in flight that the scheduler spreads them
  // over distinct workers; 4 MiB members leave ample slots for the working set.
  constexpr size_t kThreads = 16;
  constexpr size_t kItersPerThread = 300;
  constexpr clio::run::u64 kIoLen = kChunkLen;

  // REQUIRE throws, and an exception escaping a std::thread calls terminate --
  // so worker threads only record; every assertion runs on the main thread.
  std::vector<size_t> alloc_failures(kThreads, 0);
  std::vector<size_t> write_failures(kThreads, 0);
  std::vector<size_t> free_failures(kThreads, 0);
  std::vector<size_t> completed(kThreads, 0);

  auto worker = [&](size_t tid) {
    const std::vector<ctp::u8> pattern =
        MakePattern(static_cast<size_t>(kIoLen), static_cast<ctp::u8>(tid));
    for (size_t i = 0; i < kItersPerThread; ++i) {
      auto alloc =
          safe.AsyncAllocateBlocks(clio::run::PoolQuery::Dynamic(), kIoLen);
      alloc.Wait();
      if (alloc->GetReturnCode() != 0 || alloc->blocks_.size() == 0) {
        ++alloc_failures[tid];
        continue;
      }
      clio::run::priv::vector<clio::run::bdev::Block> blocks(CTP_MALLOC);
      for (size_t b = 0; b < alloc->blocks_.size(); ++b) {
        blocks.push_back(alloc->blocks_[b]);
      }

      auto wbuf = CLIO_IPC->AllocateBuffer(kIoLen);
      if (!wbuf.IsNull()) {
        memcpy(wbuf.ptr_, pattern.data(), static_cast<size_t>(kIoLen));
        auto wt = safe.AsyncWrite(clio::run::PoolQuery::Dynamic(), blocks,
                                  wbuf.shm_.template Cast<void>(), kIoLen);
        wt.Wait();
        if (wt->GetReturnCode() != 0) {
          ++write_failures[tid];
        }
        CLIO_IPC->FreeBuffer(wbuf);
      } else {
        ++write_failures[tid];
      }

      auto fr = safe.AsyncFreeBlocks(clio::run::PoolQuery::Dynamic(), blocks);
      fr.Wait();
      if (fr->GetReturnCode() != 0) {
        ++free_failures[tid];
      }
      ++completed[tid];
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (size_t t = 0; t < kThreads; ++t) {
    threads.emplace_back(worker, t);
  }
  for (auto &t : threads) {
    t.join();
  }

  size_t total_alloc_failures = 0, total_write_failures = 0;
  size_t total_free_failures = 0, total_completed = 0;
  for (size_t t = 0; t < kThreads; ++t) {
    total_alloc_failures += alloc_failures[t];
    total_write_failures += write_failures[t];
    total_free_failures += free_failures[t];
    total_completed += completed[t];
  }

  REQUIRE(total_alloc_failures == 0);
  REQUIRE(total_write_failures == 0);
  REQUIRE(total_free_failures == 0);
  REQUIRE(total_completed == kThreads * kItersPerThread);

  // The invariant: a lost Release() leaks its slot, so remaining capacity comes
  // back short of the baseline.
  auto stats1 = safe.AsyncGetStats();
  stats1.Wait();
  REQUIRE(stats1->GetReturnCode() == 0);
  const clio::run::u64 remaining_after = stats1->remaining_size_;
  HLOG(kInfo,
       "safe_bdev concurrency: {} threads x {} iters, remaining {} -> {} "
       "({} slots leaked)",
       kThreads, kItersPerThread, remaining_before, remaining_after,
       (remaining_before - remaining_after) / kChunkLen);
  REQUIRE(remaining_after == remaining_before);

  clio::run::admin::Client admin(clio::run::kAdminPoolId);
  auto d =
      admin.AsyncDestroyPool(clio::run::PoolQuery::Dynamic(), safe.pool_id_);
  d.Wait();
}

SIMPLE_TEST_MAIN()
