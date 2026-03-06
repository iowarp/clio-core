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
 * Regression test: Future::~Future() must free the heap-allocated task object.
 *
 * Before the fix, ~Future() freed FutureShm but never called DelTask(), leaking
 * the heap-allocated task object on every AsyncXxx().Wait() call.
 *
 * Strategy
 * --------
 * Measure glibc heap bytes in-use (mallinfo.uordblks) before and after running
 * many AsyncCustom().Wait() calls.  Each call submits a task, waits for it,
 * and lets the Future go out of scope.  If ~Future() does not call DelTask(),
 * every task object is retained on the heap and uordblks grows linearly.
 * With the fix, the tasks are freed and uordblks stays bounded.
 *
 * We validate both directions:
 *   1. Total heap growth across N tasks stays below a tight per-task ceiling.
 *   2. Heap usage after two equal-sized batches is the same (no monotonic
 *      growth), measured after malloc_trim(0) forces freed pages back.
 *
 * Reproducer described in:
 *   CTE_MEMORY_LEAK_REPORT.md – "Memory Leak in Future::~Future()"
 */

#include "../simple_test.h"
#include "chimaera/chimaera.h"
#include "chimaera/ipc_manager.h"
#include "chimaera/MOD_NAME/MOD_NAME_client.h"
#include "chimaera/MOD_NAME/MOD_NAME_tasks.h"

#include <malloc.h>   // mallinfo / malloc_trim
#include <cstring>
#include <thread>
#include <chrono>

using namespace std::chrono_literals;

// Distinct pool ID — must not collide with other test files.
constexpr chi::PoolId kCleanupTestPoolId = chi::PoolId(777, 0);

// Number of tasks per batch.
static constexpr int kBatchSize = 500;

// Maximum tolerated heap growth per completed task (bytes).
// The pre-fix leak was the size of a full Task object + internal allocations
// (thousands of bytes).  We allow 256 bytes to accommodate transient allocator
// bookkeeping that isn't strictly per-task.
static constexpr long kMaxBytesPerTask = 256;

static bool g_initialized = false;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Returns mallinfo.uordblks — heap bytes currently in active use. */
static long heap_in_use() {
  struct mallinfo mi = mallinfo();
  return (long)(unsigned int)mi.uordblks;
}

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

class FutureCleanupFixture {
public:
  FutureCleanupFixture() {
    if (!g_initialized) {
      bool ok = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
      if (ok) {
        g_initialized = true;
        SimpleTest::g_test_finalize = chi::CHIMAERA_FINALIZE;
        std::this_thread::sleep_for(500ms);
      }
    }
  }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST_CASE("Future destructor frees task — heap does not grow across batches",
          "[future][memory][regression]") {
  FutureCleanupFixture fixture;
  REQUIRE(g_initialized);

  chimaera::MOD_NAME::Client client(kCleanupTestPoolId);
  chi::PoolQuery pool_query = chi::PoolQuery::Dynamic();

  auto create_task =
      client.AsyncCreate(pool_query, "future_cleanup_test", kCleanupTestPoolId);
  create_task.Wait();
  client.pool_id_ = create_task->new_pool_id_;
  REQUIRE(create_task->return_code_ == 0);

  // Warm-up: let allocator reach steady state so transient bookkeeping doesn't
  // pollute the measurement.
  for (int i = 0; i < 50; ++i) {
    auto t = client.AsyncCustom(pool_query, "warmup", 0);
    t.Wait();
  }
  malloc_trim(0);  // return freed pages so measurement is clean

  SECTION("Heap-in-use growth per task is below threshold") {
    long before = heap_in_use();
    INFO("Heap in-use before batch: " << before << " bytes");

    for (int i = 0; i < kBatchSize; ++i) {
      // Future goes out of scope here — ~Future() must call DelTask().
      auto task = client.AsyncCustom(pool_query, "data", i);
      task.Wait();
      REQUIRE(task->return_code_ == 0);
    }

    malloc_trim(0);
    long after = heap_in_use();
    INFO("Heap in-use after batch:  " << after << " bytes");

    long growth = after - before;
    long per_task = growth / kBatchSize;
    INFO("Total heap growth: " << growth << " bytes  (" << per_task
                               << " bytes/task)");
    REQUIRE(per_task < kMaxBytesPerTask);
  }

  SECTION("Heap-in-use is stable across two equal batches (no monotonic leak)") {
    // Run batch A, snapshot, run batch B, snapshot.  If tasks are leaked, the
    // heap grows by (BatchSize * sizeof(task)) between snapshots.  If freed,
    // the heap returns to roughly the same level.
    for (int i = 0; i < kBatchSize; ++i) {
      auto task = client.AsyncCustom(pool_query, "batchA", i);
      task.Wait();
    }
    malloc_trim(0);
    long after_a = heap_in_use();
    INFO("Heap after batch A: " << after_a << " bytes");

    for (int i = 0; i < kBatchSize; ++i) {
      auto task = client.AsyncCustom(pool_query, "batchB", i);
      task.Wait();
    }
    malloc_trim(0);
    long after_b = heap_in_use();
    INFO("Heap after batch B: " << after_b << " bytes");

    // The delta between the two snapshots should not be proportional to batch
    // size; allow only a small absolute slack for bookkeeping drift.
    long delta = after_b - after_a;
    long per_task = delta / kBatchSize;
    INFO("Heap delta A→B: " << delta << " bytes  (" << per_task
                            << " bytes/task)");
    REQUIRE(per_task < kMaxBytesPerTask);
  }
}

TEST_CASE(
    "Future destructor: no crash on destruction of consumed Future",
    "[future][memory][regression]") {
  FutureCleanupFixture fixture;
  REQUIRE(g_initialized);

  chimaera::MOD_NAME::Client client(kCleanupTestPoolId);
  chi::PoolQuery pool_query = chi::PoolQuery::Dynamic();

  auto create_task =
      client.AsyncCreate(pool_query, "future_null_test", kCleanupTestPoolId);
  create_task.Wait();
  client.pool_id_ = create_task->new_pool_id_;
  REQUIRE(create_task->return_code_ == 0);

  SECTION("Destroying many consumed Futures does not crash (no double-free)") {
    // If ~Future() called DelTask() without the null-check guard, a double-free
    // would crash here (ASAN would report heap-use-after-free).
    REQUIRE_NOTHROW({
      for (int i = 0; i < 200; ++i) {
        auto task = client.AsyncCustom(pool_query, "nofree", i);
        task.Wait();
      }
    });
  }
}

SIMPLE_TEST_MAIN()
