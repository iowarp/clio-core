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
 * Test task output streaming functionality
 * Uses only client APIs to test large output streaming (>4KB)
 */

#include "../simple_test.h"
#include "clio_runtime/clio_runtime.h"
#include "clio_runtime/task.h"
#include "clio_runtime/ipc_manager.h"
#include "clio_runtime/MOD_NAME/MOD_NAME_client.h"
#include "clio_runtime/MOD_NAME/MOD_NAME_tasks.h"

#include <vector>
#include <chrono>
#include <thread>

using namespace clio::run;
using namespace std::chrono_literals;

// Test pool ID for MOD_NAME
constexpr clio::run::PoolId kTestModNamePoolId = clio::run::PoolId(200, 0);

// Global flag to track runtime initialization
static bool g_initialized = false;

/**
 * Fixture for streaming tests
 */
class StreamingTestFixture {
public:
  StreamingTestFixture() {
    if (!g_initialized) {
      bool success = clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
      if (success) {
        g_initialized = true;
        SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;
        std::this_thread::sleep_for(500ms);
      }
    }
  }
};

TEST_CASE("Task Streaming - Small Output", "[streaming][small]") {
  StreamingTestFixture fixture;
  REQUIRE(g_initialized);

  // Initialize MOD_NAME client
  clio::run::MOD_NAME::Client client(kTestModNamePoolId);

  // Create the MOD_NAME container
  clio::run::PoolQuery pool_query = clio::run::PoolQuery::Dynamic();
  std::string pool_name = "streaming_test_small";
  auto create_task = client.AsyncCreate(pool_query, pool_name, kTestModNamePoolId);
  create_task.Wait();
  client.pool_id_ = create_task->new_pool_id_;
  REQUIRE(create_task->return_code_ == 0);

  // Submit CustomTask with small data (< 4KB)
  std::string input_data = "small test data for streaming";
  auto task = client.AsyncCustom(pool_query, input_data, 42);
  task.Wait();

  // Verify task completed successfully
  REQUIRE(task->return_code_ == 0);
  REQUIRE(task->data_.size() > 0);
  INFO("Small output test completed with " << task->data_.size() << " bytes");
}

TEST_CASE("Task Streaming - Large Output (1MB)", "[streaming][large]") {
  StreamingTestFixture fixture;
  REQUIRE(g_initialized);

  // Initialize MOD_NAME client
  clio::run::MOD_NAME::Client client(kTestModNamePoolId);

  // Create the MOD_NAME container
  clio::run::PoolQuery pool_query = clio::run::PoolQuery::Dynamic();
  std::string pool_name = "streaming_test_large";
  auto create_task = client.AsyncCreate(pool_query, pool_name, kTestModNamePoolId);
  create_task.Wait();
  client.pool_id_ = create_task->new_pool_id_;
  REQUIRE(create_task->return_code_ == 0);

  // Submit TestLargeOutput task - this generates 1MB output
  auto task = client.AsyncTestLargeOutput(pool_query);

  // Wait for task completion (this tests streaming if output > 4KB)
  task.Wait();

  // Verify the large output was received correctly
  REQUIRE(task->data_.size() == 1024 * 1024);  // 1MB

  // Verify the pattern: data[i] = i % 256
  bool pattern_valid = true;
  size_t error_count = 0;
  constexpr size_t MAX_ERRORS_TO_SHOW = 5;

  for (size_t i = 0; i < task->data_.size(); ++i) {
    if (task->data_[i] != static_cast<uint8_t>(i % 256)) {
      pattern_valid = false;
      if (error_count < MAX_ERRORS_TO_SHOW) {
        INFO("Pattern mismatch at index " << i
             << ": expected " << static_cast<int>(i % 256)
             << ", got " << static_cast<int>(task->data_[i]));
      }
      error_count++;
    }
  }

  if (!pattern_valid) {
    INFO("Total pattern mismatches: " << error_count);
  }

  REQUIRE(pattern_valid);

  INFO("Large output test completed: received and verified 1MB output");
  INFO("Streaming mechanism tested successfully for output > 4KB copy space");
}

TEST_CASE("Task completion signals", "[streaming][completion]") {
  StreamingTestFixture fixture;
  REQUIRE(g_initialized);

  // Initialize MOD_NAME client
  clio::run::MOD_NAME::Client client(kTestModNamePoolId);

  // Create the MOD_NAME container
  clio::run::PoolQuery pool_query = clio::run::PoolQuery::Dynamic();
  std::string pool_name = "streaming_test_bitfield";
  auto create_task = client.AsyncCreate(pool_query, pool_name, kTestModNamePoolId);
  create_task.Wait();
  client.pool_id_ = create_task->new_pool_id_;
  REQUIRE(create_task->return_code_ == 0);

  // Submit a simple task
  auto task = client.AsyncCustom(pool_query, "bitfield test", 1);

  // Wait returns true once the task completes. CPU/host completion now lives on
  // Task::is_complete_ (managed by the Future), replacing FutureShm::flags_
  // FUTURE_COMPLETE.
  INFO("Testing completion via Task::is_complete_");
  REQUIRE(task.Wait());

  // Exercise the Task completion / new-data signal accessors directly.
  INFO("Testing Task is_complete_ / is_new_data_ accessors");
  auto t = ctp::make_shared<clio::run::Task>(CTP_MALLOC);
  t->UnsetComplete();
  t->UnsetNewData();
  REQUIRE_FALSE(t->IsComplete());
  t->SetComplete();
  REQUIRE(t->IsComplete());

  REQUIRE_FALSE(t->IsNewData());
  t->SetNewData();
  REQUIRE(t->IsNewData());
  REQUIRE(t->IsComplete());  // new-data is independent of completion

  t->UnsetNewData();
  REQUIRE_FALSE(t->IsNewData());
  REQUIRE(t->IsComplete());  // completion still set

  t->UnsetComplete();
  REQUIRE_FALSE(t->IsComplete());

  INFO("Task completion-signal accessors verified successfully");
}

// Regression guard for runtime-internal allocation leaks (e.g. #560: the
// server-side FutureShm that leaked once per cross-process RPC). Most valuable
// under the force-net variant, which routes every RPC through the ZMQ
// cpu2cpu path that allocates the server-side FutureShm. Only asserts when
// built with -DCLIO_CORE_ENABLE_LEAK_CHECK=ON (CTP_ALLOC_TRACK_SIZE); otherwise
// GetRuntimeHeapAllocatedBytes() returns 0 and this is a cheap no-op.
TEST_CASE("Runtime Heap Leak Check", "[streaming][leak]") {
  StreamingTestFixture fixture;
  REQUIRE(g_initialized);

  clio::run::MOD_NAME::Client client(kTestModNamePoolId);
  clio::run::PoolQuery pool_query = clio::run::PoolQuery::Dynamic();
  std::string pool_name = "streaming_test_leak";
  auto create_task =
      client.AsyncCreate(pool_query, pool_name, kTestModNamePoolId);
  create_task.Wait();
  client.pool_id_ = create_task->new_pool_id_;
  REQUIRE(create_task->return_code_ == 0);

#ifdef CTP_ALLOC_TRACK_SIZE
  auto *ipc = CLIO_IPC;
  REQUIRE(ipc != nullptr);

  // Submit one Custom RPC and block until the response is received.
  auto run_rpc = [&](int i) {
    auto task = client.AsyncCustom(pool_query, "leak probe", i);
    task.Wait();
    REQUIRE(task->return_code_ == 0);
  };

  // The server frees its per-request FutureShm *after* sending the response
  // (SendOut), so the free can lag the client's Wait(). Poll until the
  // runtime private-heap usage stops moving before snapshotting.
  auto stabilized_heap = [&]() -> size_t {
    size_t prev = ipc->GetRuntimeHeapAllocatedBytes();
    for (int i = 0; i < 150; ++i) {  // up to ~3s
      std::this_thread::sleep_for(20ms);
      size_t cur = ipc->GetRuntimeHeapAllocatedBytes();
      if (cur == prev) return cur;
      prev = cur;
    }
    return prev;
  };

  // Warm up: the first RPCs lazily allocate caches/pools that legitimately
  // persist, so they must not count against the measured window.
  constexpr int kWarmup = 50;
  for (int i = 0; i < kWarmup; ++i) run_rpc(i);
  const size_t baseline = stabilized_heap();

  // Measured window: a per-RPC leak makes the post-drain heap grow ~linearly
  // with the RPC count.
  constexpr int kMeasured = 400;
  for (int i = 0; i < kMeasured; ++i) run_rpc(i);
  const size_t after = stabilized_heap();

  const size_t delta = (after > baseline) ? (after - baseline) : 0;
  const double per_rpc = static_cast<double>(delta) / kMeasured;
  INFO("Runtime heap: baseline=" << baseline << " after=" << after
       << " delta=" << delta << " B (" << per_rpc << " B/RPC over "
       << kMeasured << " RPCs)");

  // With #560 fixed, steady-state per-RPC growth is ~0. The leaked FutureShm is
  // well over 100 B/RPC, so this tolerance cleanly separates pass from
  // regression while absorbing minor incidental allocations.
  constexpr double kMaxBytesPerRpc = 16.0;
  REQUIRE(per_rpc <= kMaxBytesPerRpc);
#else
  INFO("Leak check disabled (build with -DCLIO_CORE_ENABLE_LEAK_CHECK=ON)");
  REQUIRE(true);
#endif
}

// ===========================================================================
// Positive controls for the leak detector itself: deliberately leak (skip the
// free), confirm GetRuntimeHeapAllocatedBytes() *observes* the leak, then free
// so the case ends balanced (also confirming the detector drops back toward
// baseline on free). These prove a real un-freed allocation would be caught.
// Only assert under CLIO_CORE_ENABLE_LEAK_CHECK (CTP_ALLOC_TRACK_SIZE); a no-op
// otherwise. Run via cr_streaming_force_net (the whole binary).
// ===========================================================================

TEST_CASE("Leak Detector - AllocateBuffer leak is detected",
          "[streaming][leak-detector]") {
  StreamingTestFixture fixture;
  REQUIRE(g_initialized);
#ifdef CTP_ALLOC_TRACK_SIZE
  auto *ipc = CLIO_IPC;
  REQUIRE(ipc != nullptr);
  constexpr size_t kLeakBytes = 1u << 20;       // 1 MiB — dwarfs idle churn
  constexpr size_t kSlack = 64u << 10;          // tolerate background activity

  const size_t before = ipc->GetRuntimeHeapAllocatedBytes();

  // Leak: allocate from the runtime private heap (CTP_MALLOC) and DON'T free.
  auto buf = ipc->AllocateBuffer(kLeakBytes);
  REQUIRE(!buf.IsNull());
  const size_t leaked = ipc->GetRuntimeHeapAllocatedBytes();
  INFO("AllocateBuffer leak: detector observed +" << (leaked - before)
       << " B (leaked " << kLeakBytes << ")");
  REQUIRE(leaked >= before + kLeakBytes);        // detector found the leak

  // Free it: the detector must drop back toward baseline.
  ipc->FreeBuffer(buf);
  const size_t after_free = ipc->GetRuntimeHeapAllocatedBytes();
  REQUIRE(after_free < leaked);
  REQUIRE(after_free <= before + kSlack);
#else
  INFO("Leak tracking off; build -DCLIO_CORE_ENABLE_LEAK_CHECK=ON to exercise");
  REQUIRE(true);
#endif
}

TEST_CASE("Leak Detector - NewTask leak is detected",
          "[streaming][leak-detector]") {
  StreamingTestFixture fixture;
  REQUIRE(g_initialized);
#ifdef CTP_ALLOC_TRACK_SIZE
  auto *ipc = CLIO_IPC;
  REQUIRE(ipc != nullptr);
  using TaskT = clio::run::MOD_NAME::CustomTask;
  constexpr int kNumTasks = 256;                 // dominate idle churn
  constexpr size_t kSlack = 64u << 10;
  const size_t expected = kNumTasks * sizeof(TaskT);

  const size_t before = ipc->GetRuntimeHeapAllocatedBytes();

  // Leak: allocate tasks via NewTask (global new — invisible to CTP_MALLOC, so
  // this exercises the NewTask/DelTask accounting) and DON'T DelTask them.
  std::vector<clio::run::shared_ptr<TaskT>> tasks;
  tasks.reserve(kNumTasks);
  for (int i = 0; i < kNumTasks; ++i) {
    tasks.push_back(ipc->NewTask<TaskT>());
    REQUIRE(!tasks.back().IsNull());
  }
  const size_t leaked = ipc->GetRuntimeHeapAllocatedBytes();
  INFO("NewTask leak: detector observed +" << (leaked - before) << " B over "
       << kNumTasks << " tasks (sizeof=" << sizeof(TaskT) << ")");
  REQUIRE(leaked >= before + expected);          // detector found the leak

  // Free them: the detector must drop back toward baseline.
  for (auto &t : tasks) t.reset();
  const size_t after_free = ipc->GetRuntimeHeapAllocatedBytes();
  REQUIRE(after_free < leaked);
  REQUIRE(after_free <= before + kSlack);
#else
  INFO("Leak tracking off; build -DCLIO_CORE_ENABLE_LEAK_CHECK=ON to exercise");
  REQUIRE(true);
#endif
}

SIMPLE_TEST_MAIN()
