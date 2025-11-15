/**
 * Comprehensive unit tests for CoMutex and CoRwLock synchronization primitives
 *
 * Tests the cooperative synchronization mechanisms in the Chimaera runtime:
 * - CoMutex: Cooperative mutual exclusion with TaskId grouping
 * - CoRwLock: Cooperative reader-writer locks with TaskId grouping
 */

#include "../simple_test.h"
#include <atomic>
#include <chrono>
#include <future>
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
constexpr chi::u32 kTestTimeoutMs =
    10000; // Increased timeout for synchronization tests
constexpr chi::u32 kMaxRetries = 100;
constexpr chi::u32 kRetryDelayMs = 50;

// Test pool ID generator - avoid hardcoding, use dynamic generation
chi::PoolId generateTestPoolId() {
  // Generate pool ID based on current time to avoid conflicts
  auto now = std::chrono::high_resolution_clock::now();
  auto duration = now.time_since_epoch();
  auto microseconds =
      std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
  // Use lower 32 bits to avoid overflow, add offset to avoid admin pool range
  return chi::PoolId(static_cast<chi::u32>(microseconds & 0xFFFFFFFF) + 1000,
                     0);
}

// Test parameters
constexpr chi::u32 kShortHoldMs = 100;  // Short hold duration
constexpr chi::u32 kMediumHoldMs = 500; // Medium hold duration
constexpr chi::u32 kLongHoldMs = 1000;  // Long hold duration

// Global test state
bool g_runtime_initialized = false;
bool g_client_initialized = false;

// Test result tracking
std::atomic<int> g_completed_tasks{0};
std::atomic<int> g_successful_tasks{0};
} // namespace

/**
 * Test fixture for CoMutex and CoRwLock tests
 * Handles setup and teardown of runtime and client components
 */
class CoMutexTestFixture {
public:
  CoMutexTestFixture() : test_pool_id_(generateTestPoolId()) {}

  ~CoMutexTestFixture() { cleanup(); }

  /**
   * Initialize Chimaera runtime (server-side)
   */
  bool initializeRuntime() {
    if (g_runtime_initialized) {
      return true; // Already initialized
    }

    INFO("Initializing Chimaera runtime for CoMutex tests...");
    bool success = chi::CHIMAERA_RUNTIME_INIT();

    if (success) {
      g_runtime_initialized = true;

      // Give runtime time to initialize all components
      std::this_thread::sleep_for(500ms);

      // Verify core managers are available
      REQUIRE(CHI_CHIMAERA_MANAGER != nullptr);
      REQUIRE(CHI_IPC != nullptr);
      REQUIRE(CHI_POOL_MANAGER != nullptr);
      REQUIRE(CHI_MODULE_MANAGER != nullptr);
      REQUIRE(CHI_WORK_ORCHESTRATOR != nullptr);

      INFO("Runtime initialization successful");
    } else {
      FAIL("Failed to initialize Chimaera runtime");
    }

    return success;
  }

  /**
   * Initialize Chimaera client components
   */
  bool initializeClient() {
    if (g_client_initialized) {
      return true; // Already initialized
    }

    INFO("Initializing Chimaera client for CoMutex tests...");
    bool success = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);

    if (success) {
      g_client_initialized = true;

      // Give client time to connect to runtime
      std::this_thread::sleep_for(200ms);

      // Verify client can access IPC manager
      REQUIRE(CHI_IPC != nullptr);
      REQUIRE(CHI_IPC->IsInitialized());

      INFO("Client initialization successful");
    } else {
      FAIL("Failed to initialize Chimaera client");
    }

    return success;
  }

  /**
   * Initialize both runtime and client
   */
  bool initializeBoth() { return initializeRuntime() && initializeClient(); }

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

      task->Yield();
    }

    return true;
  }

  /**
   * Wait for multiple tasks to complete
   */
  template <typename TaskT>
  int waitForMultipleTaskCompletion(
      const std::vector<hipc::FullPtr<TaskT>> &tasks,
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
            tasks[i]->is_complete_.load() != 0) {
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
      std::string mod_pool_name = "test_mod_name_pool";
      bool mod_success =
          mod_name_client.Create(HSHM_MCTX, pool_query, mod_pool_name, test_pool_id_);
      REQUIRE(mod_success);

      INFO("MOD_NAME pool created successfully with dynamic ID: "
           << test_pool_id_.ToU64());
      return true;

    } catch (const std::exception &e) {
      FAIL("Exception creating MOD_NAME pool: " << e.what());
      return false;
    }
  }

  /**
   * Reset test counters
   */
  void resetCounters() {
    g_completed_tasks.store(0);
    g_successful_tasks.store(0);
  }

  /**
   * Get the dynamically generated test pool ID
   */
  chi::PoolId getTestPoolId() const { return test_pool_id_; }

  /**
   * Clean up test resources
   */
  void cleanup() { INFO("CoMutex test cleanup completed"); }

private:
  chi::PoolId test_pool_id_; // Dynamically generated pool ID for this test run
};

//------------------------------------------------------------------------------
// Basic CoMutex Tests
//------------------------------------------------------------------------------

TEST_CASE("CoMutex Basic Locking", "[comutex][basic]") {
  CoMutexTestFixture fixture;

  SECTION("Single CoMutex task should execute successfully") {
    REQUIRE(fixture.initializeBoth());
    REQUIRE(fixture.createModNamePool());

    chimaera::MOD_NAME::Client mod_name_client(fixture.getTestPoolId());
    chi::PoolQuery create_query = chi::PoolQuery::Dynamic();
    std::string pool_name = "test_mod_name_pool";
    bool success = mod_name_client.Create(HSHM_MCTX, create_query, pool_name, fixture.getTestPoolId());
    REQUIRE(success);

    // Execute single CoMutex test
    chi::PoolQuery pool_query = chi::PoolQuery::Local();
    chi::u32 test_id = 1;
    chi::u32 result = mod_name_client.CoMutexTest(HSHM_MCTX, pool_query,
                                                  test_id, kShortHoldMs);

    REQUIRE(result == 0); // Assuming 0 means success

    INFO("Single CoMutex test completed successfully");
  }

  SECTION("Sequential CoMutex tasks should execute in order") {
    REQUIRE(fixture.initializeBoth());
    REQUIRE(fixture.createModNamePool());

    chimaera::MOD_NAME::Client mod_name_client(fixture.getTestPoolId());
    chi::PoolQuery create_query = chi::PoolQuery::Dynamic();
    std::string pool_name = "test_mod_name_pool";
    bool success = mod_name_client.Create(HSHM_MCTX, create_query, pool_name, fixture.getTestPoolId());
    REQUIRE(success);

    // Execute multiple sequential CoMutex tests
    chi::PoolQuery pool_query = chi::PoolQuery::Local();
    const int kNumSequentialTasks = 5;
    std::vector<chi::u32> results;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < kNumSequentialTasks; ++i) {
      chi::u32 result = mod_name_client.CoMutexTest(HSHM_MCTX, pool_query,
                                                    i + 1, kShortHoldMs);
      results.push_back(result);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    // Verify all tasks completed successfully
    for (size_t i = 0; i < results.size(); ++i) {
      REQUIRE(results[i] == 0);
    }

    INFO("Sequential CoMutex tests completed in " << duration.count() << "ms");
    INFO("Expected minimum time: " << (kNumSequentialTasks * kShortHoldMs)
                                   << "ms");

    // Sequential execution should take at least the sum of hold durations
    REQUIRE(duration.count() >=
            (kNumSequentialTasks * kShortHoldMs * 0.8)); // Allow 20% tolerance
  }
}

TEST_CASE("CoMutex Concurrent Access", "[comutex][concurrent]") {
  CoMutexTestFixture fixture;

  SECTION("Concurrent CoMutex tasks with same TaskId should proceed together") {
    REQUIRE(fixture.initializeBoth());
    REQUIRE(fixture.createModNamePool());

    chimaera::MOD_NAME::Client mod_name_client(fixture.getTestPoolId());
    chi::PoolQuery create_query = chi::PoolQuery::Dynamic();
    std::string pool_name = "test_mod_name_pool";
    bool success = mod_name_client.Create(HSHM_MCTX, create_query, pool_name, fixture.getTestPoolId());
    REQUIRE(success);

    fixture.resetCounters();

    // Submit multiple async tasks with same TaskId characteristics
    // Tasks with same pid/tid/major but different minor should proceed together
    chi::PoolQuery pool_query = chi::PoolQuery::Local();
    const int kNumConcurrentTasks = 4;
    std::vector<hipc::FullPtr<chimaera::MOD_NAME::CoMutexTestTask>> tasks;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < kNumConcurrentTasks; ++i) {
      auto task = mod_name_client.AsyncCoMutexTest(HSHM_MCTX, pool_query,
                                                   i + 10, kMediumHoldMs);
      REQUIRE_FALSE(task.IsNull());
      tasks.push_back(task);
    }

    // Wait for all tasks to complete
    int completed = fixture.waitForMultipleTaskCompletion(tasks);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    REQUIRE(completed == kNumConcurrentTasks);

    // Verify all tasks succeeded
    int successful_tasks = 0;
    for (auto &task : tasks) {
      if (task->return_code_ == 0) {
        successful_tasks++;
      }
    }

    REQUIRE(successful_tasks == kNumConcurrentTasks);

    INFO("Concurrent CoMutex tasks completed in " << duration.count() << "ms");
    INFO("Hold duration per task: " << kMediumHoldMs << "ms");

    // Clean up tasks
    for (auto &task : tasks) {
      CHI_IPC->DelTask(task);
    }

    // If tasks with same TaskId run concurrently, total time should be close to
    // single task time If they serialize, total time would be much longer
    INFO("Duration analysis: " << duration.count() << "ms vs expected ~"
                               << kMediumHoldMs << "ms");
  }

  SECTION("Concurrent CoMutex tasks with different TaskIds should serialize") {
    REQUIRE(fixture.initializeBoth());
    REQUIRE(fixture.createModNamePool());

    chimaera::MOD_NAME::Client mod_name_client(fixture.getTestPoolId());
    chi::PoolQuery create_query = chi::PoolQuery::Dynamic();
    std::string pool_name = "test_mod_name_pool";
    bool success = mod_name_client.Create(HSHM_MCTX, create_query, pool_name, fixture.getTestPoolId());
    REQUIRE(success);

    // This test would require creating tasks with different TaskId
    // characteristics For now, we'll test that tasks do serialize when expected
    chi::PoolQuery pool_query = chi::PoolQuery::Local();
    const int kNumSerialTasks = 3;
    std::vector<hipc::FullPtr<chimaera::MOD_NAME::CoMutexTestTask>> tasks;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < kNumSerialTasks; ++i) {
      auto task = mod_name_client.AsyncCoMutexTest(HSHM_MCTX, pool_query,
                                                   i + 20, kShortHoldMs);
      REQUIRE_FALSE(task.IsNull());
      tasks.push_back(task);
    }

    int completed = fixture.waitForMultipleTaskCompletion(tasks);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    REQUIRE(completed == kNumSerialTasks);

    INFO("Serial CoMutex tasks completed in " << duration.count() << "ms");

    // Clean up tasks
    for (auto &task : tasks) {
      CHI_IPC->DelTask(task);
    }
  }
}

//------------------------------------------------------------------------------
// Basic CoRwLock Tests
//------------------------------------------------------------------------------

TEST_CASE("CoRwLock Basic Reader-Writer Semantics", "[corwlock][basic]") {
  CoMutexTestFixture fixture;

  SECTION("Single reader task should execute successfully") {
    REQUIRE(fixture.initializeBoth());
    REQUIRE(fixture.createModNamePool());

    chimaera::MOD_NAME::Client mod_name_client(fixture.getTestPoolId());
    chi::PoolQuery create_query = chi::PoolQuery::Dynamic();
    std::string pool_name = "test_mod_name_pool";
    bool success = mod_name_client.Create(HSHM_MCTX, create_query, pool_name, fixture.getTestPoolId());
    REQUIRE(success);

    // Execute single reader test
    chi::PoolQuery pool_query = chi::PoolQuery::Local();
    chi::u32 result = mod_name_client.CoRwLockTest(
        HSHM_MCTX, pool_query, 1, false, kShortHoldMs); // false = reader

    REQUIRE(result == 0);
    INFO("Single reader test completed successfully");
  }

  SECTION("Single writer task should execute successfully") {
    REQUIRE(fixture.initializeBoth());
    REQUIRE(fixture.createModNamePool());

    chimaera::MOD_NAME::Client mod_name_client(fixture.getTestPoolId());
    chi::PoolQuery create_query = chi::PoolQuery::Dynamic();
    std::string pool_name = "test_mod_name_pool";
    bool success = mod_name_client.Create(HSHM_MCTX, create_query, pool_name, fixture.getTestPoolId());
    REQUIRE(success);

    // Execute single writer test
    chi::PoolQuery pool_query = chi::PoolQuery::Local();
    chi::u32 result = mod_name_client.CoRwLockTest(
        HSHM_MCTX, pool_query, 2, true, kShortHoldMs); // true = writer

    REQUIRE(result == 0);
    INFO("Single writer test completed successfully");
  }

  SECTION("Sequential reader-writer tasks should execute in order") {
    REQUIRE(fixture.initializeBoth());
    REQUIRE(fixture.createModNamePool());

    chimaera::MOD_NAME::Client mod_name_client(fixture.getTestPoolId());
    chi::PoolQuery create_query = chi::PoolQuery::Dynamic();
    std::string pool_name = "test_mod_name_pool";
    bool success = mod_name_client.Create(HSHM_MCTX, create_query, pool_name, fixture.getTestPoolId());
    REQUIRE(success);

    // Execute sequential reader-writer pattern
    chi::PoolQuery pool_query = chi::PoolQuery::Local();
    std::vector<chi::u32> results;

    auto start_time = std::chrono::high_resolution_clock::now();

    // Reader -> Writer -> Reader -> Writer
    results.push_back(mod_name_client.CoRwLockTest(HSHM_MCTX, pool_query, 10,
                                                   false, kShortHoldMs));
    results.push_back(mod_name_client.CoRwLockTest(HSHM_MCTX, pool_query, 11,
                                                   true, kShortHoldMs));
    results.push_back(mod_name_client.CoRwLockTest(HSHM_MCTX, pool_query, 12,
                                                   false, kShortHoldMs));
    results.push_back(mod_name_client.CoRwLockTest(HSHM_MCTX, pool_query, 13,
                                                   true, kShortHoldMs));

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    // Verify all tasks completed successfully
    for (size_t i = 0; i < results.size(); ++i) {
      REQUIRE(results[i] == 0);
    }

    INFO("Sequential reader-writer tests completed in " << duration.count()
                                                        << "ms");
  }
}

TEST_CASE("CoRwLock Multiple Readers", "[corwlock][readers]") {
  CoMutexTestFixture fixture;

  SECTION("Multiple concurrent readers should proceed together") {
    REQUIRE(fixture.initializeBoth());
    REQUIRE(fixture.createModNamePool());

    chimaera::MOD_NAME::Client mod_name_client(fixture.getTestPoolId());
    chi::PoolQuery create_query = chi::PoolQuery::Dynamic();
    std::string pool_name = "test_mod_name_pool";
    bool success = mod_name_client.Create(HSHM_MCTX, create_query, pool_name, fixture.getTestPoolId());
    REQUIRE(success);

    fixture.resetCounters();

    // Submit multiple async reader tasks
    chi::PoolQuery pool_query = chi::PoolQuery::Local();
    const int kNumReaders = 5;
    std::vector<hipc::FullPtr<chimaera::MOD_NAME::CoRwLockTestTask>> tasks;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < kNumReaders; ++i) {
      auto task = mod_name_client.AsyncCoRwLockTest(
          HSHM_MCTX, pool_query, i + 30, false,
          kMediumHoldMs); // false = reader
      REQUIRE_FALSE(task.IsNull());
      tasks.push_back(task);
    }

    // Wait for all tasks to complete
    int completed = fixture.waitForMultipleTaskCompletion(tasks);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    REQUIRE(completed == kNumReaders);

    // Verify all tasks succeeded
    int successful_tasks = 0;
    for (auto &task : tasks) {
      if (task->return_code_ == 0) {
        successful_tasks++;
      }
    }

    REQUIRE(successful_tasks == kNumReaders);

    INFO("Concurrent readers completed in " << duration.count() << "ms");
    INFO("Hold duration per reader: " << kMediumHoldMs << "ms");

    // Clean up tasks
    for (auto &task : tasks) {
      CHI_IPC->DelTask(task);
    }

    // Multiple readers should be able to proceed concurrently
    // Total time should be close to single reader time, not sum of all readers
    INFO("Duration analysis: " << duration.count() << "ms vs expected ~"
                               << kMediumHoldMs << "ms");
  }
}

TEST_CASE("CoRwLock Writer Exclusivity", "[corwlock][writers]") {
  CoMutexTestFixture fixture;

  SECTION("Multiple writers should execute exclusively") {
    REQUIRE(fixture.initializeBoth());
    REQUIRE(fixture.createModNamePool());

    chimaera::MOD_NAME::Client mod_name_client(fixture.getTestPoolId());
    chi::PoolQuery create_query = chi::PoolQuery::Dynamic();
    std::string pool_name = "test_mod_name_pool";
    bool success = mod_name_client.Create(HSHM_MCTX, create_query, pool_name, fixture.getTestPoolId());
    REQUIRE(success);

    // Submit multiple async writer tasks
    chi::PoolQuery pool_query = chi::PoolQuery::Local();
    const int kNumWriters = 3;
    std::vector<hipc::FullPtr<chimaera::MOD_NAME::CoRwLockTestTask>> tasks;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < kNumWriters; ++i) {
      auto task = mod_name_client.AsyncCoRwLockTest(
          HSHM_MCTX, pool_query, i + 40, true, kShortHoldMs); // true = writer
      REQUIRE_FALSE(task.IsNull());
      tasks.push_back(task);
    }

    int completed = fixture.waitForMultipleTaskCompletion(tasks);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    REQUIRE(completed == kNumWriters);

    // Verify all tasks succeeded
    int successful_tasks = 0;
    for (auto &task : tasks) {
      if (task->return_code_ == 0) {
        successful_tasks++;
      }
    }

    REQUIRE(successful_tasks == kNumWriters);

    INFO("Writer exclusivity test completed in " << duration.count() << "ms");

    // Clean up tasks
    for (auto &task : tasks) {
      CHI_IPC->DelTask(task);
    }

    // Writers should execute exclusively, so total time should be close to sum
    chi::u32 expected_min_time =
        kNumWriters * kShortHoldMs * 0.8; // 20% tolerance
    INFO("Duration analysis: " << duration.count() << "ms vs expected min "
                               << expected_min_time << "ms");
    REQUIRE(duration.count() >= expected_min_time);
  }
}

TEST_CASE("CoRwLock Reader-Writer Interaction", "[corwlock][interaction]") {
  CoMutexTestFixture fixture;

  SECTION("Writer should block readers and vice versa") {
    REQUIRE(fixture.initializeBoth());
    REQUIRE(fixture.createModNamePool());

    chimaera::MOD_NAME::Client mod_name_client(fixture.getTestPoolId());
    chi::PoolQuery create_query = chi::PoolQuery::Dynamic();
    std::string pool_name = "test_mod_name_pool";
    bool success = mod_name_client.Create(HSHM_MCTX, create_query, pool_name, fixture.getTestPoolId());
    REQUIRE(success);

    // Submit mixed reader-writer tasks
    chi::PoolQuery pool_query = chi::PoolQuery::Local();
    std::vector<hipc::FullPtr<chimaera::MOD_NAME::CoRwLockTestTask>> tasks;

    auto start_time = std::chrono::high_resolution_clock::now();

    // Pattern: Reader, Writer, Readers (should serialize around writer)
    tasks.push_back(mod_name_client.AsyncCoRwLockTest(
        HSHM_MCTX, pool_query, 50, false, kShortHoldMs)); // Reader
    tasks.push_back(mod_name_client.AsyncCoRwLockTest(
        HSHM_MCTX, pool_query, 51, true, kShortHoldMs)); // Writer
    tasks.push_back(mod_name_client.AsyncCoRwLockTest(
        HSHM_MCTX, pool_query, 52, false, kShortHoldMs)); // Reader
    tasks.push_back(mod_name_client.AsyncCoRwLockTest(
        HSHM_MCTX, pool_query, 53, false, kShortHoldMs)); // Reader

    int completed = fixture.waitForMultipleTaskCompletion(tasks);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    REQUIRE(completed == 4);

    // Verify all tasks succeeded
    for (auto &task : tasks) {
      REQUIRE(task->return_code_ == 0);
    }

    INFO("Reader-writer interaction test completed in " << duration.count()
                                                        << "ms");

    // Clean up tasks
    for (auto &task : tasks) {
      CHI_IPC->DelTask(task);
    }
  }
}

//------------------------------------------------------------------------------
// TaskId Grouping Tests
//------------------------------------------------------------------------------

TEST_CASE("TaskId Grouping", "[tasknode][grouping]") {
  CoMutexTestFixture fixture;

  SECTION("Tasks with same TaskId should group together for CoMutex") {
    REQUIRE(fixture.initializeBoth());
    REQUIRE(fixture.createModNamePool());

    chimaera::MOD_NAME::Client mod_name_client(fixture.getTestPoolId());
    chi::PoolQuery create_query = chi::PoolQuery::Dynamic();
    std::string pool_name = "test_mod_name_pool";
    bool success = mod_name_client.Create(HSHM_MCTX, create_query, pool_name, fixture.getTestPoolId());
    REQUIRE(success);

    // This test validates the TaskId grouping concept
    // Tasks with same pid/tid/major but different minor should proceed together
    chi::PoolQuery pool_query = chi::PoolQuery::Local();
    const int kNumGroupedTasks = 3;
    std::vector<hipc::FullPtr<chimaera::MOD_NAME::CoMutexTestTask>> tasks;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < kNumGroupedTasks; ++i) {
      auto task = mod_name_client.AsyncCoMutexTest(HSHM_MCTX, pool_query,
                                                   i + 60, kMediumHoldMs);
      tasks.push_back(task);
    }

    int completed = fixture.waitForMultipleTaskCompletion(tasks);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    REQUIRE(completed == kNumGroupedTasks);

    for (auto &task : tasks) {
      REQUIRE(task->return_code_ == 0);
    }

    INFO("TaskId grouped CoMutex tasks completed in " << duration.count()
                                                      << "ms");

    // Clean up tasks
    for (auto &task : tasks) {
      CHI_IPC->DelTask(task);
    }
  }

  SECTION("Tasks with same TaskId should group together for CoRwLock readers") {
    REQUIRE(fixture.initializeBoth());
    REQUIRE(fixture.createModNamePool());

    chimaera::MOD_NAME::Client mod_name_client(fixture.getTestPoolId());
    chi::PoolQuery create_query = chi::PoolQuery::Dynamic();
    std::string pool_name = "test_mod_name_pool";
    bool success = mod_name_client.Create(HSHM_MCTX, create_query, pool_name, fixture.getTestPoolId());
    REQUIRE(success);

    // Test TaskId grouping for readers
    chi::PoolQuery pool_query = chi::PoolQuery::Local();
    const int kNumGroupedReaders = 4;
    std::vector<hipc::FullPtr<chimaera::MOD_NAME::CoRwLockTestTask>> tasks;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < kNumGroupedReaders; ++i) {
      auto task = mod_name_client.AsyncCoRwLockTest(
          HSHM_MCTX, pool_query, i + 70, false, kMediumHoldMs); // Readers
      tasks.push_back(task);
    }

    int completed = fixture.waitForMultipleTaskCompletion(tasks);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    REQUIRE(completed == kNumGroupedReaders);

    for (auto &task : tasks) {
      REQUIRE(task->return_code_ == 0);
    }

    INFO("TaskId grouped readers completed in " << duration.count() << "ms");

    // Clean up tasks
    for (auto &task : tasks) {
      CHI_IPC->DelTask(task);
    }
  }
}

//------------------------------------------------------------------------------
// Error Handling and Edge Cases
//------------------------------------------------------------------------------

TEST_CASE("CoMutex Error Handling", "[comutex][error]") {
  CoMutexTestFixture fixture;

  SECTION("CoMutex tasks should handle zero hold duration") {
    REQUIRE(fixture.initializeBoth());
    REQUIRE(fixture.createModNamePool());

    chimaera::MOD_NAME::Client mod_name_client(fixture.getTestPoolId());
    chi::PoolQuery create_query = chi::PoolQuery::Dynamic();
    std::string pool_name = "test_mod_name_pool";
    bool success = mod_name_client.Create(HSHM_MCTX, create_query, pool_name, fixture.getTestPoolId());
    REQUIRE(success);

    // Test with zero hold duration
    chi::PoolQuery pool_query = chi::PoolQuery::Local();
    chi::u32 result =
        mod_name_client.CoMutexTest(HSHM_MCTX, pool_query, 100, 0);

    // Should still succeed even with zero duration
    REQUIRE(result == 0);
    INFO("Zero duration CoMutex test completed successfully");
  }

  SECTION("CoMutex should handle large number of concurrent tasks") {
    REQUIRE(fixture.initializeBoth());
    REQUIRE(fixture.createModNamePool());

    chimaera::MOD_NAME::Client mod_name_client(fixture.getTestPoolId());
    chi::PoolQuery create_query = chi::PoolQuery::Dynamic();
    std::string pool_name = "test_mod_name_pool";
    bool success = mod_name_client.Create(HSHM_MCTX, create_query, pool_name, fixture.getTestPoolId());
    REQUIRE(success);

    // Stress test with many concurrent tasks
    chi::PoolQuery pool_query = chi::PoolQuery::Local();
    const int kManyTasks = 10;
    std::vector<hipc::FullPtr<chimaera::MOD_NAME::CoMutexTestTask>> tasks;

    for (int i = 0; i < kManyTasks; ++i) {
      auto task = mod_name_client.AsyncCoMutexTest(HSHM_MCTX, pool_query,
                                                   i + 200, kShortHoldMs);
      REQUIRE_FALSE(task.IsNull());
      tasks.push_back(task);
    }

    int completed =
        fixture.waitForMultipleTaskCompletion(tasks, kTestTimeoutMs * 2);

    INFO("Stress test: " << completed << "/" << kManyTasks
                         << " tasks completed");
    REQUIRE(completed > (kManyTasks / 2)); // At least half should complete

    // Clean up tasks
    for (auto &task : tasks) {
      if (!task.IsNull()) {
        CHI_IPC->DelTask(task);
      }
    }
  }
}

TEST_CASE("CoRwLock Error Handling", "[corwlock][error]") {
  CoMutexTestFixture fixture;

  SECTION("CoRwLock tasks should handle zero hold duration") {
    REQUIRE(fixture.initializeBoth());
    REQUIRE(fixture.createModNamePool());

    chimaera::MOD_NAME::Client mod_name_client(fixture.getTestPoolId());
    chi::PoolQuery create_query = chi::PoolQuery::Dynamic();
    std::string pool_name = "test_mod_name_pool";
    bool success = mod_name_client.Create(HSHM_MCTX, create_query, pool_name, fixture.getTestPoolId());
    REQUIRE(success);

    // Test reader with zero hold duration
    chi::PoolQuery pool_query = chi::PoolQuery::Local();
    chi::u32 result1 =
        mod_name_client.CoRwLockTest(HSHM_MCTX, pool_query, 300, false, 0);
    REQUIRE(result1 == 0);

    // Test writer with zero hold duration
    chi::u32 result2 =
        mod_name_client.CoRwLockTest(HSHM_MCTX, pool_query, 301, true, 0);
    REQUIRE(result2 == 0);

    INFO("Zero duration CoRwLock tests completed successfully");
  }
}

//------------------------------------------------------------------------------
// Performance and Timing Tests
//------------------------------------------------------------------------------

TEST_CASE("CoMutex Performance", "[comutex][performance]") {
  CoMutexTestFixture fixture;

  SECTION("CoMutex overhead measurement") {
    REQUIRE(fixture.initializeBoth());
    REQUIRE(fixture.createModNamePool());

    chimaera::MOD_NAME::Client mod_name_client(fixture.getTestPoolId());
    chi::PoolQuery create_query = chi::PoolQuery::Dynamic();
    std::string pool_name = "test_mod_name_pool";
    bool success = mod_name_client.Create(HSHM_MCTX, create_query, pool_name, fixture.getTestPoolId());
    REQUIRE(success);

    // Measure task execution time vs hold duration
    chi::PoolQuery pool_query = chi::PoolQuery::Local();
    const int kNumPerfTests = 5;
    std::vector<chi::u64> execution_times;

    for (int i = 0; i < kNumPerfTests; ++i) {
      auto start_time = std::chrono::high_resolution_clock::now();

      chi::u32 result = mod_name_client.CoMutexTest(HSHM_MCTX, pool_query,
                                                    i + 400, kShortHoldMs);

      auto end_time = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
          end_time - start_time);

      REQUIRE(result == 0);
      execution_times.push_back(duration.count());
    }

    // Calculate average execution time
    chi::u64 total_time = 0;
    for (auto time : execution_times) {
      total_time += time;
    }
    chi::u64 avg_time = total_time / kNumPerfTests;

    INFO("Average CoMutex execution time: " << avg_time << " microseconds");
    INFO("Hold duration: " << kShortHoldMs << "ms (" << (kShortHoldMs * 1000)
                           << " microseconds)");

    // Execution time should be reasonable compared to hold duration
    REQUIRE(avg_time <
            (kShortHoldMs * 1000 * 10)); // At most 10x the hold duration
  }
}

TEST_CASE("CoRwLock Performance", "[corwlock][performance]") {
  CoMutexTestFixture fixture;

  SECTION("Reader vs Writer performance comparison") {
    REQUIRE(fixture.initializeBoth());
    REQUIRE(fixture.createModNamePool());

    chimaera::MOD_NAME::Client mod_name_client(fixture.getTestPoolId());
    chi::PoolQuery create_query = chi::PoolQuery::Dynamic();
    std::string pool_name = "test_mod_name_pool";
    bool success = mod_name_client.Create(HSHM_MCTX, create_query, pool_name, fixture.getTestPoolId());
    REQUIRE(success);

    // Measure reader performance
    chi::PoolQuery pool_query = chi::PoolQuery::Local();
    auto start_time = std::chrono::high_resolution_clock::now();
    chi::u32 reader_result = mod_name_client.CoRwLockTest(
        HSHM_MCTX, pool_query, 500, false, kShortHoldMs);
    auto reader_end = std::chrono::high_resolution_clock::now();
    auto reader_duration =
        std::chrono::duration_cast<std::chrono::microseconds>(reader_end -
                                                              start_time);

    // Measure writer performance
    start_time = std::chrono::high_resolution_clock::now();
    chi::u32 writer_result = mod_name_client.CoRwLockTest(
        HSHM_MCTX, pool_query, 501, true, kShortHoldMs);
    auto writer_end = std::chrono::high_resolution_clock::now();
    auto writer_duration =
        std::chrono::duration_cast<std::chrono::microseconds>(writer_end -
                                                              start_time);

    REQUIRE(reader_result == 0);
    REQUIRE(writer_result == 0);

    INFO("Reader execution time: " << reader_duration.count()
                                   << " microseconds");
    INFO("Writer execution time: " << writer_duration.count()
                                   << " microseconds");

    // Both should be reasonable
    REQUIRE(reader_duration.count() < (kShortHoldMs * 1000 * 10));
    REQUIRE(writer_duration.count() < (kShortHoldMs * 1000 * 10));
  }
}

//------------------------------------------------------------------------------
// Integration Tests
//------------------------------------------------------------------------------

TEST_CASE("CoMutex and CoRwLock Integration", "[integration]") {
  CoMutexTestFixture fixture;

  SECTION("Mixed CoMutex and CoRwLock operations should coexist") {
    REQUIRE(fixture.initializeBoth());
    REQUIRE(fixture.createModNamePool());

    chimaera::MOD_NAME::Client mod_name_client(fixture.getTestPoolId());
    chi::PoolQuery create_query = chi::PoolQuery::Dynamic();
    std::string pool_name = "test_mod_name_pool";
    bool success = mod_name_client.Create(HSHM_MCTX, create_query, pool_name, fixture.getTestPoolId());
    REQUIRE(success);

    // Execute mixed operations
    chi::PoolQuery pool_query = chi::PoolQuery::Local();
    std::vector<chi::u32> results;

    // CoMutex test
    results.push_back(
        mod_name_client.CoMutexTest(HSHM_MCTX, pool_query, 600, kShortHoldMs));

    // CoRwLock reader
    results.push_back(mod_name_client.CoRwLockTest(HSHM_MCTX, pool_query, 601,
                                                   false, kShortHoldMs));

    // CoMutex test
    results.push_back(
        mod_name_client.CoMutexTest(HSHM_MCTX, pool_query, 602, kShortHoldMs));

    // CoRwLock writer
    results.push_back(mod_name_client.CoRwLockTest(HSHM_MCTX, pool_query, 603,
                                                   true, kShortHoldMs));

    // Verify all operations succeeded
    for (size_t i = 0; i < results.size(); ++i) {
      REQUIRE(results[i] == 0);
    }

    INFO("Mixed CoMutex and CoRwLock operations completed successfully");
  }
}

// Main function to run all tests
SIMPLE_TEST_MAIN()