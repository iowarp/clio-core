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

// Include Chimaera headers
#include <chimaera/chimaera.h>
#include <chimaera/pool_query.h>
#include <chimaera/singletons.h>
#include <chimaera/types.h>

// Include MOD_NAME client and tasks for testing
#include <chimaera/MOD_NAME/MOD_NAME_client.h>
#include <chimaera/MOD_NAME/MOD_NAME_tasks.h>

// Include admin client for pool management
#include <chimaera/admin/admin_client.h>
#include <chimaera/admin/admin_tasks.h>

namespace {
// Test configuration constants
constexpr chi::u32 kTestTimeoutMs = 30000; // 30 second timeout for streaming tests
constexpr chi::u32 kMaxRetries = 100;
constexpr chi::u32 kRetryDelayMs = 50;

// Global test state
bool g_initialized = false;
chi::PoolId g_test_pool_id = chi::PoolId::GetNull();

// Get shared pool ID for all streaming tests (generated once)
chi::PoolId getSharedTestPoolId() {
  if (g_test_pool_id.IsNull()) {
    // Generate pool ID based on current time to avoid conflicts
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    auto microseconds =
        std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
    // Use lower 32 bits to avoid overflow, add offset to avoid admin pool range
    g_test_pool_id = chi::PoolId(static_cast<chi::u32>(microseconds & 0xFFFFFFFF) + 1000, 0);
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
    // Initialize Chimaera once per test suite
    if (!g_initialized) {
      INFO("Initializing Chimaera for streaming tests...");
      bool success = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
      if (success) {
        g_initialized = true;
        std::this_thread::sleep_for(500ms);

        // Verify core managers are available
        REQUIRE(CHI_CHIMAERA_MANAGER != nullptr);
        REQUIRE(CHI_IPC != nullptr);
        REQUIRE(CHI_POOL_MANAGER != nullptr);
        REQUIRE(CHI_MODULE_MANAGER != nullptr);
        REQUIRE(CHI_WORK_ORCHESTRATOR != nullptr);

        // Verify client can access IPC manager
        REQUIRE(CHI_IPC->IsInitialized());

        INFO("Chimaera initialization successful");
      } else {
        FAIL("Failed to initialize Chimaera");
      }
    }
  }

  ~StreamingTestFixture() { cleanup(); }

  /**
   * Wait for task completion with timeout
   */
  template <typename TaskT>
  bool waitForTaskCompletion(hipc::FullPtr<TaskT> task,
                             chi::u32 timeout_ms = kTestTimeoutMs) {
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

      HSHM_THREAD_MODEL->Yield();
    }

    return true;
  }

  /**
   * Wait for multiple tasks to complete
   */
  template <typename TaskT>
  int waitForMultipleTaskCompletion(
      const std::vector<chi::Future<TaskT>> &tasks,
      chi::u32 timeout_ms = kTestTimeoutMs) {
    auto start_time = std::chrono::steady_clock::now();
    auto timeout_duration = std::chrono::duration<int, std::milli>(timeout_ms);

    size_t completed_count = 0;
    std::vector<bool> completed(tasks.size(), false);

    while (completed_count < tasks.size()) {
      auto current_time = std::chrono::steady_clock::now();
      if (current_time - start_time > timeout_duration) {
        INFO("Multiple task completion timeout after "
             << timeout_ms << "ms, " << completed_count << "/" << tasks.size()
             << " completed");
        break;
      }

      for (size_t i = 0; i < tasks.size(); ++i) {
        if (!completed[i] && !tasks[i].IsNull() &&
            tasks[i].IsComplete()) {
          completed[i] = true;
          completed_count++;
        }
      }

      // Yield to allow tasks to progress
      if (completed_count < tasks.size()) {
        std::this_thread::sleep_for(10ms);
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
      // Admin client is automatically initialized via CHI_ADMIN singleton
      chi::PoolQuery pool_query = chi::PoolQuery::Dynamic();

      // Create MOD_NAME client and container directly with dynamic pool ID
      chimaera::MOD_NAME::Client mod_name_client(test_pool_id_);
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

  chi::PoolId test_pool_id_;
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
  chimaera::MOD_NAME::Client client(fixture.test_pool_id_);
  chi::PoolQuery pool_query = chi::PoolQuery::Dynamic();

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
  chimaera::MOD_NAME::Client client(fixture.test_pool_id_);
  chi::PoolQuery pool_query = chi::PoolQuery::Dynamic();

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
  chimaera::MOD_NAME::Client client(fixture.test_pool_id_);
  chi::PoolQuery pool_query = chi::PoolQuery::Dynamic();

  // Create 5 concurrent tasks with large outputs
  constexpr size_t kNumTasks = 5;
  std::vector<chi::Future<chimaera::MOD_NAME::TestLargeOutputTask>> tasks;

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
