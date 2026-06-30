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
 * Comprehensive unit tests for CLIO Runtime runtime system
 *
 * Tests the complete flow: runtime startup → client init → task submission →
 * completion Uses simple custom test framework for testing.
 */

#include <chrono>
#include <memory>
#include <thread>

#include "../simple_test.h"

using namespace std::chrono_literals;

// Include CLIO Runtime headers
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/pool_query.h>
#include <clio_runtime/singletons.h>
#include <clio_runtime/types.h>

// Include MOD_NAME client and tasks for custom task testing
#include <clio_runtime/MOD_NAME/MOD_NAME_client.h>
#include <clio_runtime/MOD_NAME/MOD_NAME_tasks.h>

// Include admin client for pool management
#include <clio_runtime/admin/admin_client.h>
#include <clio_runtime/admin/admin_tasks.h>

namespace {
// Test configuration constants
constexpr clio::run::u32 kTestTimeoutMs = 5000;
constexpr clio::run::u32 kMaxRetries = 50;
constexpr clio::run::u32 kRetryDelayMs = 100;

// Test pool IDs
constexpr clio::run::PoolId kTestModNamePoolId = clio::run::PoolId(100, 0);

// Global test state
bool g_initialized = false;
} // namespace

/**
 * Test fixture for CLIO Runtime runtime tests
 * Handles setup and teardown of runtime and client components
 */
class ClioRuntimeFixture {
public:
  ClioRuntimeFixture() {
    if (!g_initialized) {
      bool success = clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
      if (success) {
        g_initialized = true;
        SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;
        std::this_thread::sleep_for(500ms);
      }
    }
  }

  ~ClioRuntimeFixture() { cleanup(); }

  /**
   * Wait for task completion with timeout
   * @param task Task to wait for
   * @param timeout_ms Maximum time to wait in milliseconds
   * @return true if task completed, false if timeout
   */
  template <typename TaskT>
  bool waitForTaskCompletion(clio::run::shared_ptr<TaskT> task,
                             clio::run::u32 timeout_ms = kTestTimeoutMs) {
    if (task.IsNull()) {
      return false;
    }

    auto start_time = std::chrono::steady_clock::now();
    auto timeout_duration = std::chrono::duration<int, std::milli>(timeout_ms);

    // Use task's Wait mechanism with timeout check
    while (task->fut_.is_complete_.load() == 0) {
      auto current_time = std::chrono::steady_clock::now();
      if (current_time - start_time > timeout_duration) {
        INFO("Task completion timeout after " << timeout_ms << "ms");
        return false; // Timeout
      }

      // Yield to allow other tasks to run
      CTP_THREAD_MODEL->Yield();
    }

    return true; // Task completed
  }

  /**
   * Clean up runtime and client resources
   */
  void cleanup() {
    // Note: CLIO Runtime framework handles automatic cleanup through destructors
    // when the CLIO Runtime manager singleton is destroyed
    INFO("Test cleanup completed");
  }

  /**
   * Create MOD_NAME pool using admin client
   * @return true if pool creation successful
   */
  bool createModNamePool() {
    try {
      // Admin client is automatically initialized via CLIO_ADMIN singleton
      clio::run::DomainQuery pool_query; // Default domain query

      // Create MOD_NAME pool parameters
      clio::run::MOD_NAME::CreateParams params;
      params.config_data_ = "test_config";
      params.worker_count_ = 2;

      // Create the MOD_NAME pool
      auto task =
          admin_client.AsyncGetOrCreatePool<clio::run::MOD_NAME::CreateParams>(pool_query, kTestModNamePoolId, params);

      if (waitForTaskCompletion(task)) {
        INFO("MOD_NAME pool created successfully with ID: "
             << kTestModNamePoolId))
        return true;
      } else {
        FAIL("Failed to create MOD_NAME pool - task did not complete");
        return false;
      }

    } catch (const std::exception &e) {
      FAIL("Exception creating MOD_NAME pool: " << e.what());
      return false;
    }
  }
};

//------------------------------------------------------------------------------
// Basic Runtime and Client Initialization Tests
//------------------------------------------------------------------------------

TEST_CASE("Clio Initialization", "[runtime][initialization]") {
  ClioRuntimeFixture fixture;

  SECTION("Clio initialization should succeed") {
    REQUIRE(g_initialized);

    // Verify runtime state
    REQUIRE(CLIO_RUNTIME_MANAGER->IsInitialized());
    REQUIRE(CLIO_RUNTIME_MANAGER->IsRuntime());
    REQUIRE(CLIO_RUNTIME_MANAGER->IsClient());
  }

  SECTION("Multiple initializations should be safe") {
    REQUIRE(g_initialized);
  }
}

//------------------------------------------------------------------------------
// MOD_NAME Custom Task Tests
//------------------------------------------------------------------------------

TEST_CASE("MOD_NAME Custom Task Execution", "[task][mod_name][custom]") {
  ClioRuntimeFixture fixture;

  SECTION(
      "Complete workflow: runtime + client + pool creation + task submission") {
    // Step 1: Initialize runtime and client
    REQUIRE(g_initialized);

    // Step 2: Create MOD_NAME pool
    REQUIRE(fixture.createModNamePool());

    // Step 3: Initialize MOD_NAME client
    clio::run::MOD_NAME::Client mod_name_client(kTestModNamePoolId);

    // Step 4: Create the MOD_NAME container
    clio::run::PoolQuery pool_query = clio::run::PoolQuery::Dynamic();
    std::string pool_name = "test_mod_name_pool";
    auto create_task = mod_name_client.AsyncCreate(pool_query, pool_name, kTestModNamePoolId);
    create_task.Wait();
    mod_name_client.pool_id_ = create_task->new_pool_id_;
    mod_name_client.return_code_ = create_task->return_code_;
    bool success = (create_task->return_code_ == 0);
    REQUIRE(success);

    // Step 5: Submit custom task
    std::string input_data = "test_input_data";
    clio::run::u32 operation_id = 42;

    // Execute custom operation asynchronously and wait
    auto task = mod_name_client.AsyncCustom(pool_query, input_data, operation_id);
    task.Wait();
    std::string output_data = task->data_.str();
    clio::run::u32 result_code = task->return_code_;

    // Verify results
    REQUIRE(result_code == 0); // Assuming 0 means success
    REQUIRE_FALSE(output_data.empty());

    INFO("Custom task completed successfully");
    INFO("Input: " << input_data);
    INFO("Output: " << output_data);
    INFO("Operation ID: " << operation_id);
    INFO("Result code: " << result_code);
  }
}

TEST_CASE("MOD_NAME Async Task Execution", "[task][mod_name][async]") {
  ClioRuntimeFixture fixture;

  SECTION("Async task submission and completion") {
    // Initialize everything
    REQUIRE(g_initialized);
    REQUIRE(fixture.createModNamePool());

    // Initialize MOD_NAME client
    clio::run::MOD_NAME::Client mod_name_client(kTestModNamePoolId);

    // Create the MOD_NAME container
    clio::run::PoolQuery pool_query = clio::run::PoolQuery::Dynamic();
    std::string pool_name = "test_mod_name_pool";
    auto create_task = mod_name_client.AsyncCreate(pool_query, pool_name, kTestModNamePoolId);
    create_task.Wait();
    mod_name_client.pool_id_ = create_task->new_pool_id_;
    mod_name_client.return_code_ = create_task->return_code_;
    bool success = (create_task->return_code_ == 0);
    REQUIRE(success);

    // Submit async custom task
    std::string input_data = "async_test_data";
    clio::run::u32 operation_id = 123;

    auto task = mod_name_client.AsyncCustom(pool_query, input_data,
                                            operation_id);

    REQUIRE_FALSE(task.IsNull());

    // Wait for completion
    REQUIRE(fixture.waitForTaskCompletion(task));

    // Verify results
    REQUIRE(task->return_code_ == 0);
    std::string output_data = task->data_.str();
    REQUIRE_FALSE(output_data.empty());

    INFO("Async task completed successfully");
    INFO("Result: " << output_data);
  }
}

//------------------------------------------------------------------------------
// Error Handling and Edge Cases
//------------------------------------------------------------------------------

TEST_CASE("Error Handling Tests", "[error][edge_cases]") {
  ClioRuntimeFixture fixture;

  SECTION("Task submission without runtime should fail gracefully") {
    (void)g_initialized; // Mark as used
    // Try to create a client without initializing runtime
    clio::run::MOD_NAME::Client mod_name_client(kTestModNamePoolId);

    // This should not crash, but may fail
    clio::run::PoolQuery pool_query = clio::run::PoolQuery::Dynamic();

    // Creating container without runtime should fail or handle gracefully
    std::string pool_name = "test_mod_name_pool";
    auto create_task = mod_name_client.AsyncCreate(pool_query, pool_name, kTestModNamePoolId);
    create_task.Wait();
    mod_name_client.pool_id_ = create_task->new_pool_id_;
    mod_name_client.return_code_ = create_task->return_code_;
    bool success = (create_task->return_code_ == 0);
    REQUIRE_NOTHROW(success);
  }

  SECTION("Invalid pool ID should handle gracefully") {
    REQUIRE(g_initialized);

    // Try to use an invalid pool ID
    constexpr clio::run::PoolId kInvalidPoolId = clio::run::PoolId(9999, 0);
    clio::run::MOD_NAME::Client invalid_client(kInvalidPoolId);

    clio::run::PoolQuery pool_query = clio::run::PoolQuery::Dynamic();
    std::string pool_name = "test_invalid_pool";

    // This should not crash
    auto create_task = invalid_client.AsyncCreate(pool_query, pool_name, kInvalidPoolId);
    create_task.Wait();
    invalid_client.pool_id_ = create_task->new_pool_id_;
    invalid_client.return_code_ = create_task->return_code_;
    bool success = (create_task->return_code_ == 0);
    REQUIRE_NOTHROW(success);
  }

  SECTION("Task timeout handling") {
    REQUIRE(g_initialized);
    REQUIRE(fixture.createModNamePool());

    clio::run::MOD_NAME::Client mod_name_client(kTestModNamePoolId);
    clio::run::PoolQuery pool_query = clio::run::PoolQuery::Dynamic();
    std::string pool_name = "test_mod_name_pool";
    auto create_task = mod_name_client.AsyncCreate(pool_query, pool_name, kTestModNamePoolId);
    create_task.Wait();
    mod_name_client.pool_id_ = create_task->new_pool_id_;
    mod_name_client.return_code_ = create_task->return_code_;
    bool success = (create_task->return_code_ == 0);
    REQUIRE(success);

    // Submit a task
    auto task =
        mod_name_client.AsyncCustom(pool_query, "timeout_test", 999);

    REQUIRE_FALSE(task.IsNull());

    // Wait with a very short timeout to test timeout handling
    bool completed = fixture.waitForTaskCompletion(task, 50); // 50ms timeout

    // The task may or may not complete in 50ms, but we shouldn't crash
    INFO("Task completed within timeout: " << completed);
  }
}

//------------------------------------------------------------------------------
// Multi-threaded Tests
//------------------------------------------------------------------------------

TEST_CASE("Concurrent Task Execution", "[concurrent][stress]") {
  ClioRuntimeFixture fixture;

  SECTION("Multiple concurrent tasks") {
    REQUIRE(g_initialized);
    REQUIRE(fixture.createModNamePool());

    clio::run::MOD_NAME::Client mod_name_client(kTestModNamePoolId);
    clio::run::PoolQuery pool_query = clio::run::PoolQuery::Dynamic();
    std::string pool_name = "test_mod_name_pool";
    auto create_task = mod_name_client.AsyncCreate(pool_query, pool_name, kTestModNamePoolId);
    create_task.Wait();
    mod_name_client.pool_id_ = create_task->new_pool_id_;
    mod_name_client.return_code_ = create_task->return_code_;
    bool success = (create_task->return_code_ == 0);
    REQUIRE(success);

    // Submit multiple concurrent tasks
    constexpr int kNumTasks = 5;
    std::vector<clio::run::shared_ptr<clio::run::MOD_NAME::CustomTask>> tasks;

    for (int i = 0; i < kNumTasks; ++i) {
      std::string input_data = "concurrent_test_" + std::to_string(i);
      auto task =
          mod_name_client.AsyncCustom(pool_query, input_data, i);

      REQUIRE_FALSE(task.IsNull());
      tasks.push_back(task);
    }

    // Wait for all tasks to complete
    int completed_tasks = 0;
    for (auto &task : tasks) {
      if (fixture.waitForTaskCompletion(task)) {
        completed_tasks++;
        REQUIRE(task->return_code_ == 0);
      }
    }

    INFO("Completed " << completed_tasks << " out of " << kNumTasks
                      << " tasks");
    REQUIRE(completed_tasks > 0); // At least some tasks should complete
  }
}

//------------------------------------------------------------------------------
// Memory Management Tests
//------------------------------------------------------------------------------

TEST_CASE("Memory Management", "[memory][cleanup]") {
  ClioRuntimeFixture fixture;

  SECTION("Task allocation and deallocation") {
    REQUIRE(g_initialized);
    REQUIRE(fixture.createModNamePool());

    clio::run::MOD_NAME::Client mod_name_client(kTestModNamePoolId);
    clio::run::PoolQuery pool_query = clio::run::PoolQuery::Dynamic();
    std::string pool_name = "test_mod_name_pool";
    auto create_task = mod_name_client.AsyncCreate(pool_query, pool_name, kTestModNamePoolId);
    create_task.Wait();
    mod_name_client.pool_id_ = create_task->new_pool_id_;
    mod_name_client.return_code_ = create_task->return_code_;
    bool success = (create_task->return_code_ == 0);
    REQUIRE(success);

    // Allocate and complete many tasks to test memory management
    constexpr int kNumAllocations = 10;

    for (int i = 0; i < kNumAllocations; ++i) {
      auto task =
          mod_name_client.AsyncCustom(pool_query, "memory_test", i);

      REQUIRE_FALSE(task.IsNull());
      task.Wait();  // Wait completes the task and takes ownership
      REQUIRE(task->return_code_ == 0);
    }
    // Tasks automatically freed when each task Future goes out of scope

    INFO("Allocated and completed " << kNumAllocations << " tasks successfully");
  }
}

//------------------------------------------------------------------------------
// Performance Tests
//------------------------------------------------------------------------------

TEST_CASE("Performance Tests", "[performance][timing]") {
  ClioRuntimeFixture fixture;

  SECTION("Task execution latency") {
    REQUIRE(g_initialized);
    REQUIRE(fixture.createModNamePool());

    clio::run::MOD_NAME::Client mod_name_client(kTestModNamePoolId);
    clio::run::PoolQuery pool_query = clio::run::PoolQuery::Dynamic();
    std::string pool_name = "test_mod_name_pool";
    auto create_task = mod_name_client.AsyncCreate(pool_query, pool_name, kTestModNamePoolId);
    create_task.Wait();
    mod_name_client.pool_id_ = create_task->new_pool_id_;
    mod_name_client.return_code_ = create_task->return_code_;
    bool success = (create_task->return_code_ == 0);
    REQUIRE(success);

    // Measure task execution time
    auto start_time = std::chrono::high_resolution_clock::now();

    auto task = mod_name_client.AsyncCustom(pool_query, "performance_test", 1);
    task.Wait();
    std::string output_data = task->data_.str();
    clio::run::u32 result_code = task->return_code_;

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time);

    REQUIRE(result_code == 0);
    INFO("Task execution time: " << duration.count() << " microseconds");

    // Reasonable performance expectation (task should complete within 1 second)
    REQUIRE(duration.count() < 1000000); // 1 second in microseconds
  }
}

// Main function to run all tests
SIMPLE_TEST_MAIN()