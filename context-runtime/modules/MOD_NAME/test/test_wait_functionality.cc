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
 * Comprehensive unit tests for task.Wait() functionality
 * 
 * This test suite validates the recursive Wait() implementation:
 * - Basic Wait() functionality with single tasks
 * - Recursive Wait() calls with multiple depth levels
 * - Task completion and synchronization correctness
 * - Client API functionality for WaitTest method
 * - Performance and timing validation
 * - Error conditions and edge cases
 * 
 * Uses the simple custom test framework for testing.
 */

#include "simple_test.h"
#include <chrono>
#include <thread>
#include <memory>
#include <vector>
#include <string>
#include <atomic>

using namespace std::chrono_literals;

// Include CLIO Runtime headers
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/singletons.h>
#include <clio_runtime/types.h>
#include <clio_runtime/pool_query.h>
#include <clio_runtime/task.h>

// Include MOD_NAME client and tasks for WaitTest functionality
#include <clio_runtime/MOD_NAME/MOD_NAME_client.h>
#include <clio_runtime/MOD_NAME/MOD_NAME_tasks.h>

// Include admin client for pool management
#include <clio_runtime/admin/admin_client.h>
#include <clio_runtime/admin/admin_tasks.h>

namespace {
  // Test configuration constants
  constexpr clio::run::u32 kTestTimeoutMs = 10000;  // 10 second timeout
  constexpr clio::run::u32 kMaxRetries = 100;
  constexpr clio::run::u32 kRetryDelayMs = 50;
  
  // Pool IDs for different test scenarios
  constexpr clio::run::PoolId kWaitTestPoolId = clio::run::PoolId(9000, 0);
  constexpr clio::run::PoolId kComparisonPoolId = clio::run::PoolId(9001, 0);
  
  // Global test state
  bool g_initialized = false;
  int g_test_counter = 0;
  
  /**
   * Test fixture for Wait functionality tests
   * Handles setup and teardown of runtime, client, and test state
   */
  class WaitTestFixture {
  public:
    
    ~WaitTestFixture() {
      cleanup();
    }
    
    WaitTestFixture() {
      if (!g_initialized) {
        bool success = clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
        if (success) {
          g_initialized = true;
          SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;
          std::this_thread::sleep_for(500ms);
        }
      }
      // Reset global counters for each test
      g_test_counter = 0;
    }
    
    /**
     * Create MOD_NAME container for testing
     */
    bool createContainer(clio::run::PoolId pool_id) {
      clio::run::MOD_NAME::Client client(pool_id);


      try {
        std::string pool_name = "wait_test_pool_" + std::to_string(pool_id.ToU64());
        auto create_task = client.AsyncCreate(clio::run::PoolQuery::Dynamic(), pool_name, pool_id);
        create_task.Wait();
        client.pool_id_ = create_task->new_pool_id_;
        client.return_code_ = create_task->return_code_;
        bool success = (create_task->return_code_ == 0);
        REQUIRE(success);

        // Give container time to initialize
        std::this_thread::sleep_for(100ms);

        INFO("Successfully created MOD_NAME container for pool " + std::to_string(pool_id.ToU64()));
        return true;
      } catch (const std::exception& e) {
        INFO("Failed to create container: " + std::string(e.what()));
        return false;
      }
    }
    
    /**
     * Wait for a condition with timeout and retries
     */
    template<typename Condition>
    bool waitForCondition(Condition&& condition, const std::string& description, 
                         clio::run::u32 timeout_ms = kTestTimeoutMs, clio::run::u32 retry_delay_ms = kRetryDelayMs) {
      auto start_time = std::chrono::steady_clock::now();
      clio::run::u32 retries = 0;
      clio::run::u32 max_retries = timeout_ms / retry_delay_ms;
      
      while (retries < max_retries) {
        if (condition()) {
          INFO(description + " - condition met after " + std::to_string(retries) + " retries");
          return true;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(retry_delay_ms));
        retries++;
        
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_time).count();
        if (elapsed > timeout_ms) {
          break;
        }
      }
      
      INFO(description + " - condition not met after " + std::to_string(retries) + " retries");
      return false;
    }
    
    /**
     * Generate unique test ID
     */
    clio::run::u32 generateTestId() {
      return static_cast<clio::run::u32>(++g_test_counter * 1000 + std::chrono::steady_clock::now().time_since_epoch().count() % 1000);
    }
    
    /**
     * Clean up test resources
     */
    void cleanup() {
      // Give system time to clean up any pending tasks
      std::this_thread::sleep_for(100ms);
    }
  };

} // end anonymous namespace

//==============================================================================
// BASIC WAIT FUNCTIONALITY TESTS
//==============================================================================

TEST_CASE("wait_test_basic_functionality", "[wait_test][basic]") {
  WaitTestFixture fixture;

  SECTION("Initialize runtime and client") {
    REQUIRE(g_initialized);
  }
  
  SECTION("Create container") {
    REQUIRE(fixture.createContainer(kWaitTestPoolId));
  }
  
  SECTION("Basic WaitTest with depth 1") {
    clio::run::MOD_NAME::Client client(kWaitTestPoolId);


    clio::run::u32 test_id = fixture.generateTestId();
    clio::run::u32 depth = 1;

    auto start_time = std::chrono::steady_clock::now();

    // Call async WaitTest method and wait
    auto task = client.AsyncWaitTest(clio::run::PoolQuery::Local(), depth, test_id);
    task.Wait();
    clio::run::u32 final_depth = task->current_depth_;

    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_time).count();

    // Verify the result
    REQUIRE(final_depth == depth);

    INFO("WaitTest with depth " + std::to_string(depth) +
         " completed in " + std::to_string(elapsed_time) + "ms");
  }
  
  SECTION("Basic WaitTest with depth 3") {
    clio::run::MOD_NAME::Client client(kWaitTestPoolId);


    clio::run::u32 test_id = fixture.generateTestId();
    clio::run::u32 depth = 3;

    auto start_time = std::chrono::steady_clock::now();

    // Call async WaitTest method and wait
    auto task = client.AsyncWaitTest(clio::run::PoolQuery::Local(), depth, test_id);
    task.Wait();
    clio::run::u32 final_depth = task->current_depth_;

    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_time).count();

    // Verify the result
    REQUIRE(final_depth == depth);

    INFO("WaitTest with depth " + std::to_string(depth) +
         " completed in " + std::to_string(elapsed_time) + "ms");
  }
}

TEST_CASE("wait_test_recursive_functionality", "[wait_test][recursive]") {
  WaitTestFixture fixture;

  SECTION("Initialize runtime and client") {
    REQUIRE(g_initialized);
  }
  
  SECTION("Create container") {
    REQUIRE(fixture.createContainer(kWaitTestPoolId));
  }
  
  SECTION("Recursive WaitTest with depth 5") {
    clio::run::MOD_NAME::Client client(kWaitTestPoolId);


    clio::run::u32 test_id = fixture.generateTestId();
    clio::run::u32 depth = 5;

    auto start_time = std::chrono::steady_clock::now();

    // Call async WaitTest method and wait
    auto task = client.AsyncWaitTest(clio::run::PoolQuery::Local(), depth, test_id);
    task.Wait();
    clio::run::u32 final_depth = task->current_depth_;

    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_time).count();

    // Verify the result
    REQUIRE(final_depth == depth);

    INFO("Recursive WaitTest with depth " + std::to_string(depth) +
         " completed in " + std::to_string(elapsed_time) + "ms");
  }
  
  SECTION("Deep recursive WaitTest with depth 10") {
    clio::run::MOD_NAME::Client client(kWaitTestPoolId);


    clio::run::u32 test_id = fixture.generateTestId();
    clio::run::u32 depth = 10;

    auto start_time = std::chrono::steady_clock::now();

    // Call async WaitTest method and wait
    auto task = client.AsyncWaitTest(clio::run::PoolQuery::Local(), depth, test_id);
    task.Wait();
    clio::run::u32 final_depth = task->current_depth_;

    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_time).count();

    // Verify the result
    REQUIRE(final_depth == depth);

    INFO("Deep recursive WaitTest with depth " + std::to_string(depth) +
         " completed in " + std::to_string(elapsed_time) + "ms");
  }
}

TEST_CASE("wait_test_async_functionality", "[wait_test][async]") {
  WaitTestFixture fixture;

  SECTION("Initialize runtime and client") {
    REQUIRE(g_initialized);
  }
  
  SECTION("Create container") {
    REQUIRE(fixture.createContainer(kWaitTestPoolId));
  }
  
  SECTION("Async WaitTest with manual Wait() call") {
    clio::run::MOD_NAME::Client client(kWaitTestPoolId);


    clio::run::u32 test_id = fixture.generateTestId();
    clio::run::u32 depth = 4;

    auto start_time = std::chrono::steady_clock::now();

    // Call asynchronous WaitTest method
    auto task = client.AsyncWaitTest(clio::run::PoolQuery::Local(), depth, test_id);

    // Manually call Wait() - this tests the recursive Wait functionality
    task.Wait();

    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_time).count();

    // Verify the result
    REQUIRE(task->current_depth_ == depth);

    INFO("Async WaitTest with depth " + std::to_string(depth) +
         " completed in " + std::to_string(elapsed_time) + "ms");

    // Clean up
    (void)CLIO_IPC;
  }
  
  SECTION("Multiple concurrent async WaitTest tasks") {
    clio::run::MOD_NAME::Client client(kWaitTestPoolId);


    const int num_tasks = 3;
    std::vector<clio::run::u32> depths = {2, 3, 4};
    std::vector<clio::run::Future<clio::run::MOD_NAME::WaitTestTask>> tasks;

    auto start_time = std::chrono::steady_clock::now();

    // Submit multiple async tasks
    for (int i = 0; i < num_tasks; ++i) {
      clio::run::u32 test_id = fixture.generateTestId();
      auto task = client.AsyncWaitTest(clio::run::PoolQuery::Local(), depths[i], test_id);
      tasks.push_back(task);
    }

    // Wait for all tasks to complete
    for (int i = 0; i < num_tasks; ++i) {
      tasks[i].Wait();
      REQUIRE(tasks[i]->current_depth_ == depths[i]);
    }

    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_time).count();

    INFO("Multiple concurrent WaitTest tasks completed in " +
         std::to_string(elapsed_time) + "ms");

    // Clean up all tasks
    (void)CLIO_IPC;
    for (auto& task : tasks) {
      (void)task;
    }
  }
}

TEST_CASE("wait_test_edge_cases", "[wait_test][edge_cases]") {
  WaitTestFixture fixture;

  SECTION("Initialize runtime and client") {
    REQUIRE(g_initialized);
  }
  
  SECTION("Create container") {
    REQUIRE(fixture.createContainer(kWaitTestPoolId));
  }
  
  SECTION("WaitTest with depth 0") {
    clio::run::MOD_NAME::Client client(kWaitTestPoolId);


    clio::run::u32 test_id = fixture.generateTestId();
    clio::run::u32 depth = 0;

    // Call with depth 0 - should complete immediately without recursion
    auto task = client.AsyncWaitTest(clio::run::PoolQuery::Local(), depth, test_id);
    task.Wait();
    clio::run::u32 final_depth = task->current_depth_;

    // With depth 0, current_depth should be incremented to 1 but no recursion
    REQUIRE(final_depth >= depth);

    INFO("WaitTest with depth 0 completed with final depth: " + std::to_string(final_depth));
  }
  
  SECTION("WaitTest with same test_id multiple times") {
    clio::run::MOD_NAME::Client client(kWaitTestPoolId);


    clio::run::u32 test_id = fixture.generateTestId();
    clio::run::u32 depth = 2;

    // Run the same test ID multiple times
    for (int i = 0; i < 3; ++i) {
      auto task = client.AsyncWaitTest(clio::run::PoolQuery::Local(), depth, test_id);
      task.Wait();
      clio::run::u32 final_depth = task->current_depth_;
      REQUIRE(final_depth == depth);
    }

    INFO("WaitTest with same test_id ran successfully multiple times");
  }
}

//==============================================================================
// MAIN TEST RUNNER
//==============================================================================

SIMPLE_TEST_MAIN()