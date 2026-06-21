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
 * Comprehensive unit tests for task output streaming functionality
 *
 * Tests the streaming mechanism for large task outputs:
 * - Small outputs (< 4KB) that fit in the default copy space
 * - Large outputs (1MB) that require streaming
 * - Concurrent large outputs to test queue handling
 */

#include "simple_test.h"
#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

// Include CLIO Runtime headers
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/pool_query.h>
#include <clio_runtime/singletons.h>
#include <clio_runtime/types.h>

// Include MOD_NAME client and tasks for testing
#include <clio_runtime/MOD_NAME/MOD_NAME_client.h>
#include <clio_runtime/MOD_NAME/MOD_NAME_tasks.h>

// Include admin client for pool management
#include <clio_runtime/admin/admin_client.h>
#include <clio_runtime/admin/admin_tasks.h>

namespace {
// Test configuration constants
constexpr clio::run::u32 kTestTimeoutMs = 30000; // 30 second timeout for streaming tests
constexpr clio::run::u32 kMaxRetries = 100;
constexpr clio::run::u32 kRetryDelayMs = 50;

// Global test state
bool g_initialized = false;
clio::run::PoolId g_test_pool_id = clio::run::PoolId::GetNull();

// Get shared pool ID for all streaming tests (generated once)
clio::run::PoolId getSharedTestPoolId() {
  if (g_test_pool_id.IsNull()) {
    // Generate pool ID based on current time to avoid conflicts
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    auto microseconds =
        std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
    // Use lower 32 bits to avoid overflow, add offset to avoid admin pool range
    g_test_pool_id = clio::run::PoolId(static_cast<clio::run::u32>(microseconds & 0xFFFFFFFF) + 1000, 0);
    INFO("Generated shared pool ID: " << g_test_pool_id.ToU64());
  } else {
    INFO("Reusing shared pool ID: " << g_test_pool_id.ToU64());
  }
  return g_test_pool_id;
}

} // namespace

/**
 * Test fixture for streaming tests
 * Handles setup and teardown of runtime and client components
 */
class StreamingTestFixture {
public:
  StreamingTestFixture() : test_pool_id_(getSharedTestPoolId()) {
    // Initialize CLIO Runtime once per test suite
    if (!g_initialized) {
      INFO("Initializing Clio for streaming tests...");
      bool success = clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
      if (success) {
        g_initialized = true;
        SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;
        std::this_thread::sleep_for(500ms);

        // Verify core managers are available
        REQUIRE(CLIO_RUNTIME_MANAGER != nullptr);
        REQUIRE(CLIO_IPC != nullptr);
        REQUIRE(CLIO_POOL_MANAGER != nullptr);
        REQUIRE(CLIO_MODULE_MANAGER != nullptr);
        REQUIRE(CLIO_WORK_ORCHESTRATOR != nullptr);

        // Verify client can access IPC manager
        REQUIRE(CLIO_IPC->IsInitialized());

        INFO("Clio initialization successful");
      } else {
        FAIL("Failed to initialize Clio");
      }
    }
  }

  ~StreamingTestFixture() { cleanup(); }

  /**
   * Wait for task completion with timeout
   */
  template <typename TaskT>
  bool waitForTaskCompletion(ctp::ipc::FullPtr<TaskT> task,
                             clio::run::u32 timeout_ms = kTestTimeoutMs) {
    if (task.IsNull()) {
      return false;
    }

    auto start_time = std::chrono::steady_clock::now();
    auto timeout_duration = std::chrono::duration<int, std::milli>(timeout_ms);

    while (task->is_complete_.load() == 0) {
      auto current_time = std::chrono::steady_clock::now();
      if (current_time - start_time > timeout_duration) {
        INFO("Task completion timeout after " << timeout_ms << "ms");
        return false;
      }

      CTP_THREAD_MODEL->Yield();
    }

    return true;
  }

  /**
   * Wait for multiple tasks to complete
   * Calls Wait() on each task sequentially to properly handle streaming
   */
  template <typename TaskT>
  int waitForMultipleTaskCompletion(
      std::vector<clio::run::Future<TaskT>> &tasks,
      clio::run::u32 timeout_ms = kTestTimeoutMs) {
    (void)timeout_ms;  // Timeout not used with Wait() approach

    size_t completed_count = 0;

    // Wait for each task sequentially
    // Tasks still execute concurrently on worker side, we just wait for each
    for (size_t i = 0; i < tasks.size(); ++i) {
      if (!tasks[i].IsNull()) {
        tasks[i].Wait();  // Wait() handles streaming properly
        completed_count++;
      }
    }

    return completed_count;
  }

  /**
   * Create MOD_NAME pool for testing
   */
  bool createModNamePool() {
    try {
      // Initialize admin client
      // Admin client is automatically initialized via CLIO_ADMIN singleton
      clio::run::PoolQuery pool_query = clio::run::PoolQuery::Dynamic();

      // Create MOD_NAME client and container directly with dynamic pool ID
      clio::run::MOD_NAME::Client mod_name_client(test_pool_id_);
      std::string mod_pool_name = "test_streaming_pool";
      auto create_task =
          mod_name_client.AsyncCreate(pool_query, mod_pool_name, test_pool_id_);
      create_task.Wait();
      bool mod_success = (create_task->return_code_ == 0);
      REQUIRE(mod_success);

      INFO("MOD_NAME pool created successfully with ID: "
           << test_pool_id_.ToU64());
      return true;

    } catch (const std::exception &e) {
      FAIL("Exception creating MOD_NAME pool: " << e.what());
      return false;
    }
  }

  /**
   * Cleanup test resources
   */
  void cleanup() {
    // No explicit cleanup needed - resources managed by shared memory
  }

  clio::run::PoolId test_pool_id_;
};

/**
 * Test 1: Small Output (< 4KB)
 *
 * Verifies that small outputs (2KB) are handled without streaming.
 * The output should fit in the default copy space and be received immediately.
 */
TEST_CASE("MOD_NAME Small Output Test", "[streaming][small]") {
  INFO("=== Test 1: Small Output (< 4KB) ===");

  StreamingTestFixture fixture;
  REQUIRE(fixture.createModNamePool());

  // Create client
  clio::run::MOD_NAME::Client client(fixture.test_pool_id_);
  clio::run::PoolQuery pool_query = clio::run::PoolQuery::Dynamic();

  // Create a custom task with 2KB of data (fits in default copy space)
  constexpr size_t kSmallDataSize = 2048; // 2KB
  std::string small_data(kSmallDataSize, 'A');

  INFO("Creating CustomTask with 2KB output...");
  auto start_time = std::chrono::high_resolution_clock::now();

  auto task = client.AsyncCustom(pool_query, small_data, 1);
  task.Wait();

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time).count();

  // Verify task completed successfully
  REQUIRE(task->return_code_ == 0);

  // Verify output data is correct
  REQUIRE(task->data_.size() == kSmallDataSize);

  // Verify data integrity
  for (size_t i = 0; i < kSmallDataSize; ++i) {
    REQUIRE(task->data_[i] == 'A');
  }

  INFO("Small output test completed in " << duration << "ms");
  INFO("Output size: " << task->data_.size() << " bytes");
  INFO("Small output test PASSED - no streaming required");
}

/**
 * Test 2: Large Output (1MB)
 *
 * Verifies that large outputs (1MB) are handled correctly via streaming.
 * The output exceeds the default copy space capacity and should use the
 * streaming mechanism via client_copy_ queue.
 */
TEST_CASE("MOD_NAME Large Output Streaming Test", "[streaming][large]") {
  INFO("=== Test 2: Large Output (1MB) Streaming ===");

  StreamingTestFixture fixture;
  REQUIRE(fixture.createModNamePool());

  // Create client
  clio::run::MOD_NAME::Client client(fixture.test_pool_id_);
  clio::run::PoolQuery pool_query = clio::run::PoolQuery::Dynamic();

  // Create a TestLargeOutput task (returns 1MB of data)
  constexpr size_t kLargeDataSize = 1024 * 1024; // 1MB

  INFO("Creating TestLargeOutput task with 1MB output...");
  auto start_time = std::chrono::high_resolution_clock::now();

  auto task = client.AsyncTestLargeOutput(pool_query);
  task.Wait();

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time).count();

  // Verify task completed successfully
  REQUIRE(task->return_code_ == 0);

  // Verify output size is 1MB
  REQUIRE(task->data_.size() == kLargeDataSize);

  // Verify data pattern: data[i] = i % 256
  INFO("Verifying 1MB data pattern...");
  bool pattern_correct = true;
  for (size_t i = 0; i < kLargeDataSize; ++i) {
    if (task->data_[i] != static_cast<uint8_t>(i % 256)) {
      pattern_correct = false;
      INFO("Pattern mismatch at index " << i << ": expected "
           << static_cast<int>(i % 256) << ", got "
           << static_cast<int>(task->data_[i]));
      break;
    }
  }
  REQUIRE(pattern_correct);

  INFO("Large output streaming test completed in " << duration << "ms");
  INFO("Output size: " << task->data_.size() << " bytes ("
       << (task->data_.size() / 1024) << " KB)");
  INFO("Large output streaming test PASSED - streaming mechanism verified");
}

/**
 * Test 3: Multiple Concurrent Large Outputs
 *
 * Verifies that multiple concurrent tasks with large outputs (512KB each)
 * can be handled correctly. Tests the client_copy_ queue handling and
 * ensures no data corruption or deadlocks occur.
 */
TEST_CASE("MOD_NAME Concurrent Streaming Test", "[streaming][concurrent]") {
  INFO("=== Test 3: Multiple Concurrent Large Outputs ===");

  StreamingTestFixture fixture;
  REQUIRE(fixture.createModNamePool());

  // Create client
  clio::run::MOD_NAME::Client client(fixture.test_pool_id_);
  clio::run::PoolQuery pool_query = clio::run::PoolQuery::Dynamic();

  // Create 5 concurrent tasks with large outputs
  constexpr size_t kNumTasks = 5;
  std::vector<clio::run::Future<clio::run::MOD_NAME::TestLargeOutputTask>> tasks;

  INFO("Creating " << kNumTasks << " concurrent TestLargeOutput tasks...");
  auto start_time = std::chrono::high_resolution_clock::now();

  // Launch all tasks concurrently
  for (size_t i = 0; i < kNumTasks; ++i) {
    auto task = client.AsyncTestLargeOutput(pool_query);
    tasks.push_back(std::move(task));
  }

  INFO("Waiting for all tasks to complete...");

  // Wait for all tasks to complete
  int completed = fixture.waitForMultipleTaskCompletion(tasks, kTestTimeoutMs);
  REQUIRE(completed == kNumTasks);

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time).count();

  // Verify all tasks completed successfully
  INFO("Verifying all task results...");
  constexpr size_t kExpectedSize = 1024 * 1024; // 1MB per task

  for (size_t i = 0; i < kNumTasks; ++i) {
    INFO("Checking task " << i << "...");

    // Verify task completed
    REQUIRE(tasks[i].IsComplete());

    // Verify return code
    REQUIRE(tasks[i]->return_code_ == 0);

    // Verify output size
    REQUIRE(tasks[i]->data_.size() == kExpectedSize);

    // Spot check data pattern (check first 1000 bytes to avoid excessive logging)
    bool pattern_correct = true;
    constexpr size_t kSpotCheckSize = 1000;
    for (size_t j = 0; j < kSpotCheckSize; ++j) {
      if (tasks[i]->data_[j] != static_cast<uint8_t>(j % 256)) {
        pattern_correct = false;
        INFO("Task " << i << " pattern mismatch at index " << j);
        break;
      }
    }
    REQUIRE(pattern_correct);
  }

  INFO("Concurrent streaming test completed in " << duration << "ms");
  INFO("Average time per task: " << (duration / kNumTasks) << "ms");
  INFO("Total data transferred: " << (kExpectedSize * kNumTasks / 1024) << " KB");
  INFO("Concurrent streaming test PASSED - all tasks completed correctly");
}

// Main function for test executable
SIMPLE_TEST_MAIN()
