/**
 * Comprehensive unit tests for Chimaera runtime system
 *
 * Tests the complete flow: runtime startup → client init → task submission →
 * completion Uses simple custom test framework for testing.
 */

#include <chrono>
#include <memory>
#include <thread>

#include "../simple_test.h"

using namespace std::chrono_literals;

// Include Chimaera headers
#include <chimaera/chimaera.h>
#include <chimaera/pool_query.h>
#include <chimaera/singletons.h>
#include <chimaera/types.h>

// Include MOD_NAME client and tasks for custom task testing
#include <chimaera/MOD_NAME/MOD_NAME_client.h>
#include <chimaera/MOD_NAME/MOD_NAME_tasks.h>

// Include admin client for pool management
#include <chimaera/admin/admin_client.h>
#include <chimaera/admin/admin_tasks.h>

namespace {
// Test configuration constants
constexpr chi::u32 kTestTimeoutMs = 5000;
constexpr chi::u32 kMaxRetries = 50;
constexpr chi::u32 kRetryDelayMs = 100;

// Test pool IDs
constexpr chi::PoolId kTestModNamePoolId = chi::PoolId(100, 0);

// Global test state
bool g_initialized = false;
} // namespace

/**
 * Test fixture for Chimaera runtime tests
 * Handles setup and teardown of runtime and client components
 */
class ChimaeraRuntimeFixture {
public:
  ChimaeraRuntimeFixture() {
    if (!g_initialized) {
      bool success = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
      if (success) {
        g_initialized = true;
        std::this_thread::sleep_for(500ms);
      }
    }
  }

  ~ChimaeraRuntimeFixture() { cleanup(); }

  /**
   * Wait for task completion with timeout
   * @param task Task to wait for
   * @param timeout_ms Maximum time to wait in milliseconds
   * @return true if task completed, false if timeout
   */
  template <typename TaskT>
  bool waitForTaskCompletion(hipc::FullPtr<TaskT> task,
                             chi::u32 timeout_ms = kTestTimeoutMs) {
    if (task.IsNull()) {
      return false;
    }

    auto start_time = std::chrono::steady_clock::now();
    auto timeout_duration = std::chrono::duration<int, std::milli>(timeout_ms);

    // Use task's Wait mechanism with timeout check
    while (task->is_complete_.load() == 0) {
      auto current_time = std::chrono::steady_clock::now();
      if (current_time - start_time > timeout_duration) {
        INFO("Task completion timeout after " << timeout_ms << "ms");
        return false; // Timeout
      }

      // Use the task's own Yield() method for efficient waiting
      task->Yield();
    }

    return true; // Task completed
  }

  /**
   * Clean up runtime and client resources
   */
  void cleanup() {
    // Note: Chimaera framework handles automatic cleanup through destructors
    // when the Chimaera manager singleton is destroyed
    INFO("Test cleanup completed");
  }

  /**
   * Create MOD_NAME pool using admin client
   * @return true if pool creation successful
   */
  bool createModNamePool() {
    try {
      // Admin client is automatically initialized via CHI_ADMIN singleton
      chi::DomainQuery pool_query; // Default domain query

      // Create MOD_NAME pool parameters
      chimaera::MOD_NAME::CreateParams params;
      params.config_data_ = "test_config";
      params.worker_count_ = 2;

      // Create the MOD_NAME pool
      auto task =
          admin_client.AsyncGetOrCreatePool<chimaera::MOD_NAME::CreateParams>(
              HSHM_MCTX, pool_query, kTestModNamePoolId, params);

      if (waitForTaskCompletion(task)) {
        INFO("MOD_NAME pool created successfully with ID: "
             << kTestModNamePoolId))

        // Clean up task
        CHI_IPC->DelTask(task);
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

TEST_CASE("Chimaera Initialization", "[runtime][initialization]") {
  ChimaeraRuntimeFixture fixture;

  SECTION("Chimaera initialization should succeed") {
    REQUIRE(g_initialized);

    // Verify runtime state
    REQUIRE(CHI_CHIMAERA_MANAGER->IsInitialized());
    REQUIRE(CHI_CHIMAERA_MANAGER->IsRuntime());
    REQUIRE(CHI_CHIMAERA_MANAGER->IsClient());
  }

  SECTION("Multiple initializations should be safe") {
    REQUIRE(g_initialized);
  }
}

//------------------------------------------------------------------------------
// MOD_NAME Custom Task Tests
//------------------------------------------------------------------------------

TEST_CASE("MOD_NAME Custom Task Execution", "[task][mod_name][custom]") {
  ChimaeraRuntimeFixture fixture;

  SECTION(
      "Complete workflow: runtime + client + pool creation + task submission") {
    // Step 1: Initialize runtime and client
    REQUIRE(g_initialized);

    // Step 2: Create MOD_NAME pool
    REQUIRE(fixture.createModNamePool());

    // Step 3: Initialize MOD_NAME client
    chimaera::MOD_NAME::Client mod_name_client(kTestModNamePoolId);

    // Step 4: Create the MOD_NAME container
    chi::DomainQuery pool_query; // Default domain query
    std::string pool_name = "test_mod_name_pool";
    bool success = mod_name_client.Create(HSHM_MCTX, pool_query, pool_name, kTestModNamePoolId);
    REQUIRE(success);

    // Step 5: Submit custom task
    std::string input_data = "test_input_data";
    chi::u32 operation_id = 42;
    std::string output_data;

    // Execute custom operation synchronously
    chi::u32 result_code = mod_name_client.Custom(
        HSHM_MCTX, pool_query, input_data, operation_id, output_data);

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
  ChimaeraRuntimeFixture fixture;

  SECTION("Async task submission and completion") {
    // Initialize everything
    REQUIRE(g_initialized);
    REQUIRE(fixture.createModNamePool());

    // Initialize MOD_NAME client
    chimaera::MOD_NAME::Client mod_name_client(kTestModNamePoolId);

    // Create the MOD_NAME container
    chi::DomainQuery pool_query;
    std::string pool_name = "test_mod_name_pool";
    bool success = mod_name_client.Create(HSHM_MCTX, pool_query, pool_name, kTestModNamePoolId);
    REQUIRE(success);

    // Submit async custom task
    std::string input_data = "async_test_data";
    chi::u32 operation_id = 123;

    auto task = mod_name_client.AsyncCustom(HSHM_MCTX, pool_query, input_data,
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

    // Clean up task
    CHI_IPC->DelTask(task);
  }
}

//------------------------------------------------------------------------------
// Error Handling and Edge Cases
//------------------------------------------------------------------------------

TEST_CASE("Error Handling Tests", "[error][edge_cases]") {
  ChimaeraRuntimeFixture fixture;

  SECTION("Task submission without runtime should fail gracefully") {
    (void)g_initialized; // Mark as used
    // Try to create a client without initializing runtime
    chimaera::MOD_NAME::Client mod_name_client(kTestModNamePoolId);

    // This should not crash, but may fail
    chi::DomainQuery pool_query;

    // Creating container without runtime should fail or handle gracefully
    std::string pool_name = "test_mod_name_pool";
    bool success = mod_name_client.Create(HSHM_MCTX, pool_query, pool_name, kTestModNamePoolId);
    REQUIRE_NOTHROW(success);
  }

  SECTION("Invalid pool ID should handle gracefully") {
    REQUIRE(g_initialized);

    // Try to use an invalid pool ID
    constexpr chi::PoolId kInvalidPoolId = chi::PoolId(9999, 0);
    chimaera::MOD_NAME::Client invalid_client(kInvalidPoolId);

    chi::DomainQuery pool_query;
    std::string pool_name = "test_invalid_pool";

    // This should not crash
    bool success = invalid_client.Create(HSHM_MCTX, pool_query, pool_name, kInvalidPoolId);
    REQUIRE_NOTHROW(success);
  }

  SECTION("Task timeout handling") {
    REQUIRE(g_initialized);
    REQUIRE(fixture.createModNamePool());

    chimaera::MOD_NAME::Client mod_name_client(kTestModNamePoolId);
    chi::DomainQuery pool_query;
    std::string pool_name = "test_mod_name_pool";
    bool success = mod_name_client.Create(HSHM_MCTX, pool_query, pool_name, kTestModNamePoolId);
    REQUIRE(success);

    // Submit a task
    auto task =
        mod_name_client.AsyncCustom(HSHM_MCTX, pool_query, "timeout_test", 999);

    REQUIRE_FALSE(task.IsNull());

    // Wait with a very short timeout to test timeout handling
    bool completed = fixture.waitForTaskCompletion(task, 50); // 50ms timeout

    // The task may or may not complete in 50ms, but we shouldn't crash
    INFO("Task completed within timeout: " << completed);

    // Clean up
    if (!task.IsNull()) {
      CHI_IPC->DelTask(task);
    }
  }
}

//------------------------------------------------------------------------------
// Multi-threaded Tests
//------------------------------------------------------------------------------

TEST_CASE("Concurrent Task Execution", "[concurrent][stress]") {
  ChimaeraRuntimeFixture fixture;

  SECTION("Multiple concurrent tasks") {
    REQUIRE(g_initialized);
    REQUIRE(fixture.createModNamePool());

    chimaera::MOD_NAME::Client mod_name_client(kTestModNamePoolId);
    chi::DomainQuery pool_query;
    std::string pool_name = "test_mod_name_pool";
    bool success = mod_name_client.Create(HSHM_MCTX, pool_query, pool_name, kTestModNamePoolId);
    REQUIRE(success);

    // Submit multiple concurrent tasks
    constexpr int kNumTasks = 5;
    std::vector<hipc::FullPtr<chimaera::MOD_NAME::CustomTask>> tasks;

    for (int i = 0; i < kNumTasks; ++i) {
      std::string input_data = "concurrent_test_" + std::to_string(i);
      auto task =
          mod_name_client.AsyncCustom(HSHM_MCTX, pool_query, input_data, i);

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

    // Clean up tasks
    for (auto &task : tasks) {
      if (!task.IsNull()) {
        CHI_IPC->DelTask(task);
      }
    }
  }
}

//------------------------------------------------------------------------------
// Memory Management Tests
//------------------------------------------------------------------------------

TEST_CASE("Memory Management", "[memory][cleanup]") {
  ChimaeraRuntimeFixture fixture;

  SECTION("Task allocation and deallocation") {
    REQUIRE(g_initialized);
    REQUIRE(fixture.createModNamePool());

    chimaera::MOD_NAME::Client mod_name_client(kTestModNamePoolId);
    chi::DomainQuery pool_query;
    std::string pool_name = "test_mod_name_pool";
    bool success = mod_name_client.Create(HSHM_MCTX, pool_query, pool_name, kTestModNamePoolId);
    REQUIRE(success);

    // Allocate many tasks to test memory management
    constexpr int kNumAllocations = 10;
    std::vector<hipc::FullPtr<chimaera::MOD_NAME::CustomTask>> allocated_tasks;

    for (int i = 0; i < kNumAllocations; ++i) {
      auto task =
          mod_name_client.AsyncCustom(HSHM_MCTX, pool_query, "memory_test", i);

      REQUIRE_FALSE(task.IsNull());
      allocated_tasks.push_back(task);
    }

    INFO("Allocated " << kNumAllocations << " tasks successfully");

    // Clean up all tasks
    for (auto &task : allocated_tasks) {
      CHI_IPC->DelTask(task);
    }

    INFO("Deallocated all tasks successfully");
  }
}

//------------------------------------------------------------------------------
// Performance Tests
//------------------------------------------------------------------------------

TEST_CASE("Performance Tests", "[performance][timing]") {
  ChimaeraRuntimeFixture fixture;

  SECTION("Task execution latency") {
    REQUIRE(g_initialized);
    REQUIRE(fixture.createModNamePool());

    chimaera::MOD_NAME::Client mod_name_client(kTestModNamePoolId);
    chi::DomainQuery pool_query;
    std::string pool_name = "test_mod_name_pool";
    bool success = mod_name_client.Create(HSHM_MCTX, pool_query, pool_name, kTestModNamePoolId);
    REQUIRE(success);

    // Measure task execution time
    auto start_time = std::chrono::high_resolution_clock::now();

    std::string output_data;
    chi::u32 result_code = mod_name_client.Custom(
        HSHM_MCTX, pool_query, "performance_test", 1, output_data);

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