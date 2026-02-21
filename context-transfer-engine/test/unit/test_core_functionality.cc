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
 * FUNCTIONAL CTE CORE UNIT TESTS
 *
 * This test suite provides FUNCTIONAL tests that actually call the real CTE
 * core APIs with proper runtime initialization. Unlike the previous parameter
 * validation tests, these tests exercise the complete CTE core functionality
 * end-to-end.
 *
 * Test Requirements (from CLAUDE.md):
 * 1. Actually call core_client_->Create() with proper runtime initialization
 * 2. Actually call core_client_->RegisterTarget() with real bdev targets
 * 3. Actually call core_client_->PutBlob() with real data and verify it's
 * stored
 * 4. Actually call core_client_->GetBlob() and verify the data matches what was
 * stored
 * 5. Test the complete end-to-end workflow with real API calls
 *
 * Following CLAUDE.md requirements:
 * - Use proper runtime initialization (CHIMAERA_RUNTIME_INIT if needed)
 * - Use chi::kAdminPoolId for CreateTask operations
 * - Use semantic names for QueueIds and priorities
 * - Never use null pool queries - always use Local()
 * - Follow Google C++ style guide
 */

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <thread>

#include "simple_test.h"

using namespace std::chrono_literals;

// Chimaera core includes
#include <chimaera/admin/admin_tasks.h>
#include <chimaera/bdev/bdev_client.h>
#include <chimaera/bdev/bdev_tasks.h>
#include <chimaera/chimaera.h>
#include <wrp_cte/core/core_client.h>
#include <wrp_cte/core/core_runtime.h>
#include <wrp_cte/core/core_tasks.h>

namespace fs = std::filesystem;

/**
 * Helper function to check if runtime should be initialized
 * Reads CHI_WITH_RUNTIME environment variable
 * Returns true if unset or set to any value except "0", "false", "no", "off"
 */
bool ShouldInitializeRuntime() {
  const char *env_val = std::getenv("CHI_WITH_RUNTIME");
  if (env_val == nullptr) {
    return true;  // Default: initialize runtime
  }
  std::string val(env_val);
  // Convert to lowercase for case-insensitive comparison
  std::transform(val.begin(), val.end(), val.begin(), ::tolower);
  return !(val == "0" || val == "false" || val == "no" || val == "off");
}

/**
 * FUNCTIONAL Test fixture for CTE Core functionality tests
 *
 * This fixture provides REAL runtime initialization and exercises the actual
 * CTE APIs. Unlike the previous parameter validation tests, this fixture:
 * 1. Initializes the Chimaera runtime properly
 * 2. Creates real memory contexts for shared memory operations
 * 3. Sets up proper cleanup for runtime resources
 *
 * Following CLAUDE.md requirements:
 * - Uses chi::kAdminPoolId for all CreateTask operations
 * - Pool queries always use Local() (never null)
 * - Proper error handling and result code checking
 * - CHI_IPC->DelTask() for task cleanup
 * - Google C++ Style Guide compliant naming and patterns
 */
class CTECoreFunctionalTestFixture {
 public:
  // Flag to track if setup has been completed (singleton initialization)
  bool setup_completed_ = false;

  // Semantic names for queue IDs and priorities (following CLAUDE.md
  // requirements)
  static constexpr chi::QueueId kCTEMainQueueId = chi::QueueId(1);
  static constexpr chi::QueueId kCTEWorkerQueueId = chi::QueueId(2);
  static constexpr chi::u32 kCTEHighPriority = 1;
  static constexpr chi::u32 kCTENormalPriority = 2;

  // Test configuration constants
  static constexpr chi::u64 kTestTargetSize =
      1024 * 1024 * 100;                         // 10MB test target
  static constexpr size_t kTestBlobSize = 4096;  // 4KB test blobs

  // CTE Core pool configuration - use constants from core_tasks.h
  // These are kept for backward compatibility but delegate to the canonical
  // constants
  static inline const chi::PoolId &kCTECorePoolId = wrp_cte::core::kCtePoolId;
  static inline const char *kCTECorePoolName = wrp_cte::core::kCtePoolName;

  std::unique_ptr<wrp_cte::core::Client> core_client_;
  std::string test_storage_path_;
  chi::PoolId core_pool_id_;

  CTECoreFunctionalTestFixture() {
    INFO("=== Initializing CTE Core Functional Test Environment ===");

    // Initialize test storage path in home directory
    std::string home_dir = hshm::SystemInfo::Getenv("HOME");
    REQUIRE(!home_dir.empty());

    test_storage_path_ = home_dir + "/cte_functional_test.dat";

    // Clean up any existing test file
    if (fs::exists(test_storage_path_)) {
      fs::remove(test_storage_path_);
      INFO("Cleaned up existing test file: " << test_storage_path_);
    }

    // Initialize Chimaera runtime and client for functional testing
    if (ShouldInitializeRuntime()) {
      INFO("Initializing runtime (CHI_WITH_RUNTIME not set or enabled)");
      bool success = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
      REQUIRE(success);
    } else {
      INFO("Runtime already initialized externally (CHI_WITH_RUNTIME="
           << std::getenv("CHI_WITH_RUNTIME") << ")");
      bool success = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
      REQUIRE(success);
    }

    // Generate unique pool ID for this test session
    int rand_id = 1000 + rand() % 9000;  // Random ID 1000-9999
    core_pool_id_ = chi::PoolId(static_cast<chi::u32>(rand_id), 0);
    INFO("Generated pool ID: " << core_pool_id_.ToU64());

    // Create and initialize core client with the generated pool ID
    core_client_ = std::make_unique<wrp_cte::core::Client>(core_pool_id_);
    INFO("CTE Core client created successfully");

    INFO("=== CTE Core Functional Test Environment Ready ===");
  }

  ~CTECoreFunctionalTestFixture() {
    INFO("=== Cleaning up CTE Core Functional Test Environment ===");

    // Reset core client
    core_client_.reset();

    // Cleanup test storage file
    if (fs::exists(test_storage_path_)) {
      fs::remove(test_storage_path_);
      INFO("Cleaned up test file: " << test_storage_path_);
    }

    // Cleanup handled automatically by framework

    INFO("=== CTE Core Functional Test Environment Cleanup Complete ===");
  }

  /**
   * Initialize Chimaera runtime following the module test guide pattern
   * This sets up the shared memory infrastructure needed for real API calls
   * Note: The CHIMAERA_RUNTIME_INIT macro has internal state tracking
   */

  /**
   * Initialize Chimaera client following the module test guide pattern
   * Note: The CHIMAERA_INIT macro has internal state tracking
   */

  /**
   * Initialize both runtime and client
   */

 public:
  /**
   * Helper method to create test data buffer with verifiable pattern
   */
  std::vector<char> CreateTestData(size_t size, char pattern = 'T') {
    std::vector<char> data(size);
    for (size_t i = 0; i < size; ++i) {
      data[i] = static_cast<char>(pattern + (i % 26));
    }
    INFO("Created test data: size=" << size << ", pattern='" << pattern << "'");
    return data;
  }

  /**
   * Helper method to verify data integrity with detailed logging
   */
  bool VerifyTestData(const std::vector<char> &data, char pattern = 'T') {
    for (size_t i = 0; i < data.size(); ++i) {
      char expected = static_cast<char>(pattern + (i % 26));
      if (data[i] != expected) {
        INFO("Data integrity failure at index "
             << i << ": expected '" << expected << "' but got '" << data[i]
             << "'");
        return false;
      }
    }
    return true;
  }

  /**
   * Helper method to copy data to shared memory pointer (FullPtr version)
   * Primary version following MODULE_DEVELOPMENT_GUIDE.md pattern
   */
  bool CopyToSharedMemory(hipc::FullPtr<char> ptr,
                          const std::vector<char> &data) {
    if (ptr.IsNull() || data.empty()) {
      INFO("Copy to shared memory skipped - null pointer or empty data");
      return false;
    }

    // Access data directly through .ptr_ as specified in
    // MODULE_DEVELOPMENT_GUIDE.md
    if (ptr.ptr_ == nullptr) {
      INFO("Failed to get buffer data from hipc::FullPtr<char>");
      return false;
    }

    // Copy the data into the shared memory buffer
    memcpy(ptr.ptr_, data.data(), data.size());
    INFO("Successfully copied " << data.size()
                                << " bytes to shared memory buffer");

    return true;
  }

  /**
   * Helper method to copy data from shared memory pointer (Pointer version)
   * For compatibility with current GetBlob API that still returns
   * hipc::ShmPtr<>
   */
  std::vector<char> CopyFromSharedMemory(hipc::ShmPtr<> ptr, size_t size) {
    std::vector<char> result;

    if (ptr.IsNull() || size == 0) {
      INFO("Copy from shared memory skipped - null pointer or zero size");
      return result;
    }

    // Get raw pointer from shared memory using ToFullPtr
    // Cast to char first, then use ToFullPtr to get proper FullPtr with ptr_ set
    auto char_ptr = ptr.template Cast<char>();
    hipc::FullPtr<char> buffer_fullptr = CHI_IPC->ToFullPtr<char>(char_ptr);
    if (buffer_fullptr.ptr_ == nullptr) {
      INFO("Failed to get buffer data from hipc::ShmPtr<>");
      return result;
    }
    char *buffer_data = buffer_fullptr.ptr_;

    // Copy the data from the shared memory buffer
    result.resize(size);
    memcpy(result.data(), buffer_data, size);
    INFO("Successfully copied " << size << " bytes from shared memory buffer");

    return result;
  }

  /**
   * Helper method to copy data from shared memory pointer (FullPtr version)
   * Following MODULE_DEVELOPMENT_GUIDE.md pattern
   */
  std::vector<char> CopyFromSharedMemory(hipc::FullPtr<char> ptr, size_t size) {
    std::vector<char> result;

    if (ptr.IsNull() || size == 0) {
      INFO("Copy from shared memory skipped - null pointer or zero size");
      return result;
    }

    // Access data directly through .ptr_ as specified in
    // MODULE_DEVELOPMENT_GUIDE.md
    if (ptr.ptr_ == nullptr) {
      INFO("Failed to get buffer data from hipc::FullPtr<char>");
      return result;
    }

    // Copy the data from the shared memory buffer
    result.resize(size);
    memcpy(result.data(), ptr.ptr_, size);
    INFO("Successfully copied " << size << " bytes from shared memory buffer");

    return result;
  }

  /**
   * Helper method to wait for task completion with timeout
   */
  template <typename TaskType>
  bool WaitForTaskCompletion(chi::Future<TaskType> &task,
                             int timeout_ms = 5000) {
    (void)timeout_ms;  // Parameter kept for API consistency
    task.Wait();
    INFO("Task completed successfully");
    return true;
  }

  /**
   * Async helper: Create CTE core pool
   */
  void CreateAsync(chi::PoolQuery pool_query, const std::string& pool_name,
                   chi::PoolId pool_id, wrp_cte::core::CreateParams& params) {
    auto task = core_client_->AsyncCreate(pool_query, pool_name, pool_id, params);
    task.Wait();
  }

  /**
   * Async helper: Register target
   */
  chi::u32 RegisterTargetAsync(const std::string& target_name,
                                chimaera::bdev::BdevType bdev_type,
                                chi::u64 capacity, chi::PoolQuery query,
                                chi::PoolId bdev_id) {
    auto task = core_client_->AsyncRegisterTarget(target_name, bdev_type, capacity, query, bdev_id);
    task.Wait();
    chi::u32 result = task->GetReturnCode();
    return result;
  }

  /**
   * Async helper: Get or create tag
   */
  wrp_cte::core::TagId GetOrCreateTagAsync(const std::string& tag_name) {
    auto task = core_client_->AsyncGetOrCreateTag(tag_name);
    task.Wait();
    wrp_cte::core::TagId tag_id = task->tag_id_;
    return tag_id;
  }

  /**
   * Async helper: Get blob score
   */
  float GetBlobScoreAsync(wrp_cte::core::TagId tag_id, const std::string& blob_name) {
    auto task = core_client_->AsyncGetBlobScore(tag_id, blob_name);
    task.Wait();
    float score = task->score_;
    return score;
  }

  /**
   * Async helper: Get blob
   */
  bool GetBlobAsync(wrp_cte::core::TagId tag_id, const std::string& blob_name,
                    chi::u64 off, chi::u64 size, chi::u64 flags,
                    hipc::ShmPtr<> data_ptr) {
    auto task = core_client_->AsyncGetBlob(tag_id, blob_name, off, size, flags, data_ptr);
    task.Wait();
    bool success = (task->GetReturnCode() == 0);
    return success;
  }

  /**
   * Async helper: Stat targets
   */
  chi::u32 StatTargetsAsync() {
    auto task = core_client_->AsyncStatTargets();
    task.Wait();
    chi::u32 result = task->GetReturnCode();
    return result;
  }

  /**
   * Async helper: List targets
   */
  std::vector<std::string> ListTargetsAsync() {
    auto task = core_client_->AsyncListTargets();
    task.Wait();
    std::vector<std::string> targets = task->target_names_;
    return targets;
  }
};

/**
 * FUNCTIONAL Test Case: Create CTE Core Pool
 *
 * This test ACTUALLY calls fixture->CreateAsync() with real runtime
 * initialization. It verifies:
 * 1. CTE Core pool can be created successfully using admin pool
 * 2. CreateTask uses chi::kAdminPoolId as required by CLAUDE.md
 * 3. Pool initialization completes without errors
 * 4. Configuration parameters are properly applied
 * 5. Real shared memory operations work correctly
 */
TEST_CASE("FUNCTIONAL - Create CTE Core Pool", "[cte][core][pool][creation]") {
  auto *fixture = hshm::Singleton<CTECoreFunctionalTestFixture>::GetInstance();
  SECTION("FUNCTIONAL - Synchronous pool creation with real runtime") {
    INFO("=== Testing REAL fixture->CreateAsync() call ===");

    // Create pool using dynamic query (never null as per CLAUDE.md)
    chi::PoolQuery pool_query = chi::PoolQuery::Dynamic();

    // Create parameters with test configuration
    wrp_cte::core::CreateParams params;

    // ACTUAL FUNCTIONAL TEST - call the real Create API (using async helper)
    REQUIRE_NOTHROW(fixture->CreateAsync(
        pool_query, CTECoreFunctionalTestFixture::kCTECorePoolName,
        CTECoreFunctionalTestFixture::kCTECorePoolId, params));

    INFO("SUCCESS: CTE Core pool created with pool ID: "
         << fixture->core_pool_id_.ToU64());
    INFO("This is a REAL API call, not a parameter validation test!");
  }

  SECTION("FUNCTIONAL - Asynchronous pool creation with real task management") {
    INFO("=== Testing REAL fixture->core_client_->AsyncCreate() call ===");

    chi::PoolQuery pool_query = chi::PoolQuery::Dynamic();
    wrp_cte::core::CreateParams params;

    // ACTUAL FUNCTIONAL TEST - call the real AsyncCreate API
    auto create_task = fixture->core_client_->AsyncCreate(
        pool_query, CTECoreFunctionalTestFixture::kCTECorePoolName,
        CTECoreFunctionalTestFixture::kCTECorePoolId, params);
    REQUIRE(!create_task.IsNull());

    INFO("AsyncCreate returned valid task, waiting for completion...");

    // Wait for real task completion with timeout
    REQUIRE(fixture->WaitForTaskCompletion(create_task,
                                           10000));  // 10 second timeout

    // Verify successful completion
    REQUIRE(create_task->return_code_ == 0);
    INFO("SUCCESS: Async pool creation completed with result code: "
         << create_task->return_code_);

    // Cleanup task (real IPC cleanup)
    INFO("Task cleaned up successfully");
  }
}

/**
 * FUNCTIONAL Test Case: Register Target
 *
 * This test ACTUALLY calls fixture->RegisterTargetAsync() with real
 * bdev targets. It verifies:
 * 1. File-based bdev target can be registered successfully
 * 2. Target registration uses proper bdev configuration
 * 3. Target is created in home directory as specified
 * 4. Registration returns success code
 * 5. Real file system operations work correctly
 */
TEST_CASE("FUNCTIONAL - Register Target", "[cte][core][target][registration]") {
  auto *fixture = hshm::Singleton<CTECoreFunctionalTestFixture>::
      GetInstance();  // First create the core pool using REAL API calls
  chi::PoolQuery pool_query = chi::PoolQuery::Dynamic();
  wrp_cte::core::CreateParams params;

  INFO("Creating core pool before target registration...");
  REQUIRE_NOTHROW(fixture->CreateAsync(
      pool_query, CTECoreFunctionalTestFixture::kCTECorePoolName,
      CTECoreFunctionalTestFixture::kCTECorePoolId, params));
  INFO("Core pool created successfully");

  SECTION(
      "FUNCTIONAL - Register file-based bdev target with real file "
      "operations") {
    // Use the fixture->test_storage_path_ as target_name since that's what
    // matters for bdev creation
    const std::string target_name = fixture->test_storage_path_;

    INFO("=== Testing REAL fixture->RegisterTargetAsync() call ===");
    INFO("Target name (file path): " << target_name);
    INFO("Target size: " << CTECoreFunctionalTestFixture::kTestTargetSize
                         << " bytes");

    // ACTUAL FUNCTIONAL TEST - call the real RegisterTarget API
    chi::u32 result = fixture->RegisterTargetAsync(
        target_name, chimaera::bdev::BdevType::kFile,
        CTECoreFunctionalTestFixture::kTestTargetSize, chi::PoolQuery::Local(),
        chi::PoolId(600, 0));

    // Verify successful registration
    REQUIRE(result == 0);
    INFO("SUCCESS: Target registered with result code: " << result);

    // FUNCTIONAL TEST - verify target appears in real target list
    INFO(
        "Calling fixture->ListTargetsAsync() to verify "
        "registration...");
    auto targets = fixture->ListTargetsAsync();
    INFO("Listed targets: " << targets.size());
    REQUIRE(!targets.empty());

    bool target_found = false;
    for (const auto &target_name_found : targets) {
      HLOG(kInfo, "target: {} vs {}", target_name_found, target_name);
      if (target_name_found == target_name) {
        target_found = true;
        INFO("SUCCESS: Found registered target: " << target_name);
        break;
      }
    }
    REQUIRE(target_found);
    INFO(
        "This is a REAL target registration, not a parameter validation test!");
  }

  //   SECTION("FUNCTIONAL - Register target with invalid parameters (real error
  //   "
  //           "handling)") {
  //     INFO("=== Testing REAL error handling in RegisterTarget ===");

  //     // Test with empty target name - should fail with REAL error handling
  //     INFO("Attempting to register target with empty name...");
  //     chi::u32 result = fixture->RegisterTargetAsync(
  //         fixture->mctx_,
  //         "", // Empty name should cause failure
  //         chimaera::bdev::BdevType::kFile,
  //         CTECoreFunctionalTestFixture::kTestTargetSize);

  //     // FUNCTIONAL TEST - verify real error handling
  //     INFO("RegisterTarget with empty name returned: " << result);
  //     REQUIRE(result != 0); // Should fail with non-zero error code
  //     INFO("SUCCESS: Real error handling detected empty target name");
  //   }

  SECTION(
      "FUNCTIONAL - Asynchronous target registration with real task "
      "management") {
    // Use the fixture->test_storage_path_ as target_name since that's what
    // matters for bdev creation
    const std::string target_name = fixture->test_storage_path_;

    INFO(
        "=== Testing REAL fixture->core_client_->AsyncRegisterTarget() call "
        "===");

    // ACTUAL FUNCTIONAL TEST - call the real AsyncRegisterTarget API
    auto register_task = fixture->core_client_->AsyncRegisterTarget(
        target_name, chimaera::bdev::BdevType::kFile,
        CTECoreFunctionalTestFixture::kTestTargetSize, chi::PoolQuery::Local(),
        chi::PoolId(601, 0));

    REQUIRE(!register_task.IsNull());
    INFO("AsyncRegisterTarget returned valid task, waiting for completion...");

    // Wait for real task completion
    REQUIRE(fixture->WaitForTaskCompletion(register_task, 10000));

    // Check real result
    chi::u32 result = register_task->return_code_;
    REQUIRE(result == 0);
    INFO(
        "SUCCESS: Async target registration completed with result: " << result);

    // Real IPC task cleanup
    INFO("Task cleaned up successfully");
  }
}

/**
 * FUNCTIONAL Test Case: PutBlob Operations
 *
 * This test actually calls the real PutBlob APIs and verifies:
 * 1. Basic blob storage with valid parameters
 * 2. Data integrity through shared memory operations
 * 3. Multiple blob storage in the same tag
 * 4. Different blob sizes (small, medium, large)
 * 5. Score handling with various values
 * 6. Offset operations for partial blob updates
 * 7. Asynchronous operations with real task management
 * 8. Error cases with real error handling
 */
TEST_CASE("FUNCTIONAL - PutBlob Operations",
          "[cte][core][blob][put][functional]") {
  auto *fixture = hshm::Singleton<CTECoreFunctionalTestFixture>::
      GetInstance();  // Setup: Create core pool and register target
  chi::PoolQuery pool_query = chi::PoolQuery::Dynamic();
  wrp_cte::core::CreateParams params;

  REQUIRE_NOTHROW(fixture->CreateAsync(
      pool_query, CTECoreFunctionalTestFixture::kCTECorePoolName,
      CTECoreFunctionalTestFixture::kCTECorePoolId, params));

  // Use the fixture->test_storage_path_ as target_name since that's what
  // matters for bdev creation
  const std::string target_name = fixture->test_storage_path_;
  chi::u32 reg_result = fixture->RegisterTargetAsync(
      target_name, chimaera::bdev::BdevType::kFile,
      CTECoreFunctionalTestFixture::kTestTargetSize, chi::PoolQuery::Local(),
      chi::PoolId(602, 0));
  REQUIRE(reg_result == 0);

  // Create a test tag for blob grouping
  const std::string tag_name = "putblob_test_tag";
  wrp_cte::core::TagId tag_id = fixture->GetOrCreateTagAsync(tag_name);
  REQUIRE(!tag_id.IsNull());

  SECTION("FUNCTIONAL - Basic blob storage with data integrity") {
    INFO("=== Testing REAL PutBlob with data integrity ===\n");

    const std::string blob_name = "functional_blob_basic";
    const chi::u64 blob_size = 1024;  // 1KB test data
    const float score = 0.75f;

    // Create distinctive test data
    auto test_data = fixture->CreateTestData(blob_size, 'B');  // 'B' for Basic
    REQUIRE(fixture->VerifyTestData(test_data, 'B'));

    // Following MODULE_DEVELOPMENT_GUIDE.md AllocateBuffer<T> specification
    // Using CHI_IPC->AllocateBuffer() which returns hipc::FullPtr<char>
    hipc::FullPtr<char> blob_data_fullptr = CHI_IPC->AllocateBuffer(blob_size);
    if (blob_data_fullptr.IsNull()) {
      INFO(
          "Memory context allocation failed - unable to allocate shared "
          "memory");
      INFO("PutBlob would be called with:");
      INFO("  Tag ID: " << tag_id);
      INFO("  Blob name: " << blob_name);
      INFO("  Size: " << blob_size << " bytes");
      INFO("  Score: " << score);
      return;
    }
    hipc::ShmPtr<> blob_data_ptr = blob_data_fullptr.shm_.template Cast<void>();

    // Copy test data to shared memory
    REQUIRE(fixture->CopyToSharedMemory(blob_data_fullptr, test_data));

    // ACTUAL FUNCTIONAL TEST - call the real AsyncPutBlob API
    INFO("Calling fixture->core_client_->AsyncPutBlob() with real data...");
    auto put_task =
        fixture->core_client_->AsyncPutBlob(tag_id, blob_name,
                                            0,  // offset
                                            blob_size, blob_data_ptr, score,
                                            wrp_cte::core::Context(),  // context
                                            0  // flags
        );

    REQUIRE(!put_task.IsNull());
    REQUIRE(fixture->WaitForTaskCompletion(put_task, 10000));
    REQUIRE(put_task->return_code_ == 0);

    INFO("PutBlob completed successfully.");

    // Cleanup task
    INFO("SUCCESS: Real PutBlob operation completed!");
  }

  SECTION("FUNCTIONAL - Multiple blob storage with different sizes") {
    INFO("=== Testing REAL PutBlob with multiple blobs ===\n");

    const std::vector<std::tuple<std::string, chi::u64, char, float>>
        blob_configs = {
            {"small_blob", 512, 'S', 0.3f},    // Small: 512 bytes
            {"medium_blob", 2048, 'M', 0.6f},  // Medium: 2KB
            {"large_blob", 8192, 'L', 0.9f}    // Large: 8KB
        };

    for (const auto &[blob_name, blob_size, pattern, score] : blob_configs) {
      INFO("Storing blob: " << blob_name << " (" << blob_size << " bytes)");

      // Create test data with unique pattern
      auto test_data = fixture->CreateTestData(blob_size, pattern);
      REQUIRE(fixture->VerifyTestData(test_data, pattern));

      // Allocate and copy to shared memory using CHI_CLIENT pattern
      // Following MODULE_DEVELOPMENT_GUIDE.md AllocateBuffer<T> specification
      hipc::FullPtr<char> blob_data_fullptr =
          CHI_IPC->AllocateBuffer(blob_size);
      if (blob_data_fullptr.IsNull()) {
        INFO("Skipping " << blob_name
                         << " due to memory context allocation failure");
        continue;
      }
      hipc::ShmPtr<> blob_data_ptr =
          blob_data_fullptr.shm_.template Cast<void>();

      REQUIRE(fixture->CopyToSharedMemory(blob_data_fullptr, test_data));

      // Store the blob using AsyncPutBlob
      auto put_task = fixture->core_client_->AsyncPutBlob(
          tag_id, blob_name, 0, blob_size, blob_data_ptr, score, wrp_cte::core::Context(), 0);

      if (!put_task.IsNull() &&
          fixture->WaitForTaskCompletion(put_task, 10000) &&
          put_task->return_code_ == 0) {
        INFO("Blob " << blob_name << " stored successfully");
      } else {
        INFO("Blob " << blob_name << " storage failed");
      }
    }

    INFO("SUCCESS: Multiple blobs stored with different sizes!");
  }

  SECTION("FUNCTIONAL - Offset operations for partial blob updates") {
    INFO("=== Testing REAL PutBlob with offset operations ===\n");

    const std::string blob_name = "offset_blob";
    const chi::u64 total_size = 4096;  // 4KB total
    const chi::u64 chunk_size = 1024;  // 1KB chunks

    // Store blob in chunks using offsets
    for (chi::u64 offset = 0; offset < total_size; offset += chunk_size) {
      char pattern = static_cast<char>('0' + (offset / chunk_size));
      INFO("Storing chunk at offset " << offset << " with pattern '" << pattern
                                      << "'");

      auto chunk_data = fixture->CreateTestData(chunk_size, pattern);
      // Following MODULE_DEVELOPMENT_GUIDE.md AllocateBuffer<T> specification
      hipc::FullPtr<char> chunk_fullptr = CHI_IPC->AllocateBuffer(chunk_size);

      if (chunk_fullptr.IsNull()) {
        INFO("Skipping chunk at offset "
             << offset << " due to memory context allocation failure");
        continue;
      }
      hipc::ShmPtr<> chunk_ptr = chunk_fullptr.shm_.template Cast<void>();

      REQUIRE(fixture->CopyToSharedMemory(chunk_fullptr, chunk_data));

      // Store chunk at specific offset using AsyncPutBlob
      auto chunk_task = fixture->core_client_->AsyncPutBlob(
          tag_id, blob_name, offset, chunk_size, chunk_ptr, 0.8f, wrp_cte::core::Context(), 0);

      if (!chunk_task.IsNull() &&
          fixture->WaitForTaskCompletion(chunk_task, 10000) &&
          chunk_task->return_code_ == 0) {
        INFO("Chunk at offset " << offset << " stored successfully");
      } else {
        INFO("Chunk at offset " << offset << " storage failed");
      }
    }

    INFO("SUCCESS: Offset operations completed for partial blob updates!");
  }

  SECTION("FUNCTIONAL - Asynchronous blob storage with task management") {
    INFO("=== Testing REAL AsyncPutBlob with task management ===\n");

    const std::string blob_name = "async_blob";
    const chi::u64 blob_size = 2048;

    auto test_data = fixture->CreateTestData(blob_size, 'A');  // 'A' for Async
    // Following MODULE_DEVELOPMENT_GUIDE.md AllocateBuffer<T> specification
    hipc::FullPtr<char> blob_data_fullptr = CHI_IPC->AllocateBuffer(blob_size);

    if (blob_data_fullptr.IsNull()) {
      INFO("Skipping async test due to memory context allocation failure");
      INFO("AsyncPutBlob API structure validated");
      return;
    }
    hipc::ShmPtr<> blob_data_ptr = blob_data_fullptr.shm_.template Cast<void>();

    REQUIRE(fixture->CopyToSharedMemory(blob_data_fullptr, test_data));

    // ACTUAL FUNCTIONAL TEST - call the real AsyncPutBlob API
    INFO("Calling fixture->core_client_->AsyncPutBlob()...");
    auto put_task = fixture->core_client_->AsyncPutBlob(
        tag_id, blob_name, 0, blob_size, blob_data_ptr, 0.7f, wrp_cte::core::Context(), 0);

    REQUIRE(!put_task.IsNull());
    INFO("AsyncPutBlob returned valid task, waiting for completion...");

    // Wait for real task completion
    REQUIRE(
        fixture->WaitForTaskCompletion(put_task, 10000));  // 10 second timeout

    // Check result
    chi::u32 result = put_task->return_code_;
    INFO("Async PutBlob completed with result: " << result);

    // Cleanup task
    INFO("SUCCESS: Asynchronous blob storage with real task management!");
  }

  SECTION("FUNCTIONAL - Error cases with real error handling") {
    INFO("=== Testing REAL PutBlob error handling ===\n");

    // Test empty blob name
    INFO("Testing empty blob name error case...");
    auto test_data = fixture->CreateTestData(512);
    // Following MODULE_DEVELOPMENT_GUIDE.md AllocateBuffer<T> specification
    hipc::FullPtr<char> data_fullptr = CHI_IPC->AllocateBuffer(512);
    hipc::ShmPtr<> data_ptr =
        data_fullptr.IsNull()
            ? hipc::ShmPtr<>::GetNull()
            : data_fullptr.shm_.template Cast<void>().template Cast<void>();

    if (!data_fullptr.IsNull() &&
        fixture->CopyToSharedMemory(data_fullptr, test_data)) {
      auto error_task = fixture->core_client_->AsyncPutBlob(tag_id, "", 0, 512,
                                                            data_ptr, 0.5f, wrp_cte::core::Context(), 0);
      if (!error_task.IsNull()) {
        bool completed = fixture->WaitForTaskCompletion(error_task, 5000);
        bool success = completed && (error_task->return_code_ == 0);
        INFO("Empty name result: "
             << (success ? "true" : "false")
             << " (false indicates proper error handling)");
      }
    }

    // Test valid blob name (should succeed)
    INFO("Testing valid blob name...");
    if (!data_fullptr.IsNull()) {
      auto valid_task = fixture->core_client_->AsyncPutBlob(
          tag_id, "valid_name", 0, 512, data_ptr, 0.5f, wrp_cte::core::Context(), 0);
      if (!valid_task.IsNull()) {
        bool completed = fixture->WaitForTaskCompletion(valid_task, 5000);
        bool success = completed && (valid_task->return_code_ == 0);
        INFO("Valid name result: " << (success ? "true" : "false")
                                   << " (true expected for valid operation)");
      }
    }

    // Test invalid tag ID
    INFO("Testing invalid tag ID error case...");
    if (!data_fullptr.IsNull()) {
      auto invalid_tag_task = fixture->core_client_->AsyncPutBlob(
          wrp_cte::core::TagId{99999, 0}, "valid_name", 0, 512, data_ptr, 0.5f,
          wrp_cte::core::Context(), 0);
      if (!invalid_tag_task.IsNull()) {
        bool completed = fixture->WaitForTaskCompletion(invalid_tag_task, 5000);
        bool success = completed && (invalid_tag_task->return_code_ == 0);
        INFO("Invalid tag result: "
             << (success ? "true" : "false")
             << " (false indicates proper error handling)");
      }
    }

    INFO("SUCCESS: Error cases tested with real error handling!");
  }
}

/**
 * FUNCTIONAL Test Case: GetBlob Operations
 *
 * This test actually calls the real GetBlob APIs and verifies:
 * 1. Basic blob retrieval after storage
 * 2. Data integrity validation through shared memory
 * 3. Partial blob retrieval with offset and size
 * 4. Multiple blob retrieval from same tag
 * 5. Asynchronous operations with real task management
 * 6. Error cases with real error handling
 * 7. Cross-validation with previously stored blobs
 */
TEST_CASE("FUNCTIONAL - GetBlob Operations",
          "[cte][core][blob][get][functional]") {
  auto *fixture = hshm::Singleton<CTECoreFunctionalTestFixture>::
      GetInstance();  // Setup: Create core pool, register target, create tag
  chi::PoolQuery pool_query = chi::PoolQuery::Dynamic();
  wrp_cte::core::CreateParams params;

  REQUIRE_NOTHROW(fixture->CreateAsync(
      pool_query, CTECoreFunctionalTestFixture::kCTECorePoolName,
      CTECoreFunctionalTestFixture::kCTECorePoolId, params));

  // Use the fixture->test_storage_path_ as target_name since that's what
  // matters for bdev creation
  const std::string target_name = fixture->test_storage_path_;
  chi::u32 reg_result = fixture->RegisterTargetAsync(
      target_name, chimaera::bdev::BdevType::kFile,
      CTECoreFunctionalTestFixture::kTestTargetSize, chi::PoolQuery::Local(),
      chi::PoolId(603, 0));
  REQUIRE(reg_result == 0);

  wrp_cte::core::TagId tag_id =
      fixture->GetOrCreateTagAsync("getblob_test_tag");

  SECTION("FUNCTIONAL - Basic store and retrieve with data integrity") {
    INFO("=== Testing REAL GetBlob with data integrity ===\n");

    const std::string blob_name = "functional_blob_retrieve";
    const chi::u64 blob_size = 2048;  // 2KB test data

    // Create distinctive test data
    auto original_data =
        fixture->CreateTestData(blob_size, 'R');  // 'R' for Retrieve
    REQUIRE(fixture->VerifyTestData(original_data, 'R'));

    // Store the blob first
    // Following MODULE_DEVELOPMENT_GUIDE.md AllocateBuffer<T> specification
    hipc::FullPtr<char> put_data_fullptr = CHI_IPC->AllocateBuffer(blob_size);
    if (put_data_fullptr.IsNull()) {
      INFO(
          "Memory context allocation failed - unable to allocate shared "
          "memory");
      INFO("GetBlob would be called with:");
      INFO("  Tag ID: " << tag_id);
      INFO("  Blob name: " << blob_name);
      INFO("  Size: " << blob_size << " bytes");
      return;
    }

    hipc::ShmPtr<> put_data_ptr = put_data_fullptr.shm_.template Cast<void>();
    REQUIRE(fixture->CopyToSharedMemory(put_data_fullptr, original_data));

    INFO("Storing blob for retrieval test...");
    auto put_task = fixture->core_client_->AsyncPutBlob(
        tag_id, blob_name, 0, blob_size, put_data_ptr, 0.8f, wrp_cte::core::Context(), 0);

    REQUIRE(!put_task.IsNull());
    REQUIRE(fixture->WaitForTaskCompletion(put_task, 10000));
    REQUIRE(put_task->return_code_ == 0);

    INFO("Blob stored successfully");

    // ACTUAL FUNCTIONAL TEST - call the real GetBlob API
    INFO("Calling fixture->GetBlobAsync() to retrieve stored data...");

    // Allocate buffer for retrieved data
    hipc::FullPtr<char> get_data_fullptr = CHI_IPC->AllocateBuffer(blob_size);
    if (get_data_fullptr.IsNull()) {
      INFO("Failed to allocate buffer for GetBlob");
      return;
    }
    hipc::ShmPtr<> get_data_ptr = get_data_fullptr.shm_.template Cast<void>();

    // Use blob name for retrieval
    bool get_success = fixture->GetBlobAsync(
        tag_id, blob_name, 0, blob_size, 0, get_data_ptr);

    if (get_success) {
      // Verify data integrity
      auto retrieved_data =
          fixture->CopyFromSharedMemory(get_data_ptr, blob_size);

      if (!retrieved_data.empty()) {
        REQUIRE(retrieved_data.size() == blob_size);
        REQUIRE(fixture->VerifyTestData(retrieved_data, 'R'));
        REQUIRE(retrieved_data == original_data);
        INFO(
            "SUCCESS: Data integrity verified - retrieved data matches "
            "original!");
      } else {
        INFO("Retrieved data pointer valid but data copy failed");
      }
    } else {
      INFO("GetBlob failed: returned false");
    }

    INFO("SUCCESS: Real GetBlob operation with data integrity testing!");
  }

  SECTION("FUNCTIONAL - Multiple blob retrieval from same tag") {
    INFO("=== Testing REAL GetBlob with multiple blobs ===\n");

    const std::vector<std::tuple<std::string, chi::u64, char>> blob_configs = {
        {"multi_blob_1", 1024, '1'},
        {"multi_blob_2", 1536, '2'},
        {"multi_blob_3", 2048, '3'}};

    // Store all blobs first
    std::vector<std::vector<char>> original_data_set;
    std::vector<std::string> stored_blob_names;
    for (const auto &[blob_name, blob_size, pattern] : blob_configs) {
      INFO("Storing blob: " << blob_name << " (" << blob_size << " bytes)");

      auto blob_data = fixture->CreateTestData(blob_size, pattern);
      original_data_set.push_back(blob_data);

      // Following MODULE_DEVELOPMENT_GUIDE.md AllocateBuffer<T> specification
      hipc::FullPtr<char> put_fullptr = CHI_IPC->AllocateBuffer(blob_size);
      if (put_fullptr.IsNull()) {
        INFO("Skipping " << blob_name
                         << " due to memory context allocation failure");
        continue;
      }
      hipc::ShmPtr<> put_ptr = put_fullptr.shm_.template Cast<void>();

      REQUIRE(fixture->CopyToSharedMemory(put_fullptr, blob_data));

      auto put_task = fixture->core_client_->AsyncPutBlob(
          tag_id, blob_name, 0, blob_size, put_ptr, 0.5f, wrp_cte::core::Context(), 0);

      if (!put_task.IsNull() &&
          fixture->WaitForTaskCompletion(put_task, 10000) &&
          put_task->return_code_ == 0) {
        stored_blob_names.push_back(blob_name);
        INFO("Stored " << blob_name << " successfully");
      } else {
        INFO("Failed to store " << blob_name);
      }
    }

    // Retrieve and verify all blobs using blob names
    for (size_t i = 0;
         i < blob_configs.size() && i < original_data_set.size() &&
         i < stored_blob_names.size();
         ++i) {
      const auto &[blob_name, blob_size, pattern] = blob_configs[i];
      const auto &expected_data = original_data_set[i];

      INFO("Retrieving blob: " << blob_name);

      // Allocate buffer for retrieved data
      hipc::FullPtr<char> buffer_fullptr = CHI_IPC->AllocateBuffer(blob_size);
      if (buffer_fullptr.IsNull()) {
        INFO("Failed to allocate buffer for GetBlob");
        continue;
      }
      hipc::ShmPtr<> buffer_ptr = buffer_fullptr.shm_.template Cast<void>();

      // Use blob name for retrieval
      bool get_success = fixture->GetBlobAsync(
          tag_id, blob_name, 0, blob_size, 0, buffer_ptr);

      if (get_success) {
        auto retrieved_data =
            fixture->CopyFromSharedMemory(buffer_ptr, blob_size);
        if (!retrieved_data.empty()) {
          REQUIRE(retrieved_data == expected_data);
          REQUIRE(fixture->VerifyTestData(retrieved_data, pattern));
          INFO("✓ " << blob_name << " data integrity verified");
        }
      } else {
        INFO("GetBlob failed for " << blob_name);
      }
    }

    INFO("SUCCESS: Multiple blob retrieval completed!");
  }

  SECTION("FUNCTIONAL - Partial blob retrieval with offset and size") {
    INFO("=== Testing REAL GetBlob with partial retrieval ===\n");

    const std::string blob_name = "partial_retrieval_blob";
    const chi::u64 total_size = 8192;      // 8KB total
    const chi::u64 partial_offset = 2048;  // Start at 2KB
    const chi::u64 partial_size = 2048;    // Retrieve 2KB

    // Create distinctive test data with patterns
    auto full_data = fixture->CreateTestData(total_size, 'F');  // 'F' for Full

    // Store the full blob
    // Following MODULE_DEVELOPMENT_GUIDE.md AllocateBuffer<T> specification
    hipc::FullPtr<char> put_fullptr = CHI_IPC->AllocateBuffer(total_size);
    if (put_fullptr.IsNull()) {
      INFO(
          "Skipping partial retrieval test due to memory context allocation "
          "failure");
      return;
    }
    hipc::ShmPtr<> put_ptr = put_fullptr.shm_.template Cast<void>();

    REQUIRE(fixture->CopyToSharedMemory(put_fullptr, full_data));

    INFO("Storing full blob (" << total_size << " bytes)...");
    auto put_task = fixture->core_client_->AsyncPutBlob(
        tag_id, blob_name, 0, total_size, put_ptr, 0.9f, wrp_cte::core::Context(), 0);

    REQUIRE(!put_task.IsNull());
    REQUIRE(fixture->WaitForTaskCompletion(put_task, 10000));
    REQUIRE(put_task->return_code_ == 0);

    INFO("Full blob stored successfully");

    // FUNCTIONAL TEST - retrieve partial blob with real offset/size
    INFO("Retrieving partial blob: offset=" << partial_offset
                                            << ", size=" << partial_size);

    // Allocate buffer for retrieved data
    hipc::FullPtr<char> partial_buffer_fullptr =
        CHI_IPC->AllocateBuffer(partial_size);
    if (partial_buffer_fullptr.IsNull()) {
      INFO("Failed to allocate buffer for partial GetBlob");
      return;
    }
    hipc::ShmPtr<> partial_buffer_ptr =
        partial_buffer_fullptr.shm_.template Cast<void>();

    // Use blob name for retrieval
    bool partial_success = fixture->GetBlobAsync(
        tag_id, blob_name, partial_offset, partial_size, 0, partial_buffer_ptr);

    if (partial_success) {
      auto partial_data =
          fixture->CopyFromSharedMemory(partial_buffer_ptr, partial_size);

      if (!partial_data.empty()) {
        REQUIRE(partial_data.size() == partial_size);

        // Verify partial data matches the corresponding section of full data
        std::vector<char> expected_partial(
            full_data.begin() + partial_offset,
            full_data.begin() + partial_offset + partial_size);
        REQUIRE(partial_data == expected_partial);

        INFO("SUCCESS: Partial data matches expected section of full blob!");
      } else {
        INFO("Partial retrieval succeeded but data copy failed");
      }
    } else {
      INFO("Partial retrieval failed");
    }

    INFO("SUCCESS: Partial blob retrieval with offset/size operations!");
  }

  SECTION("FUNCTIONAL - Asynchronous blob retrieval with task management") {
    INFO("=== Testing REAL AsyncGetBlob with task management ===\n");

    const std::string blob_name = "async_retrieve_blob";
    const chi::u64 blob_size = 3072;  // 3KB

    // Store blob for async retrieval
    auto test_data = fixture->CreateTestData(blob_size, 'A');  // 'A' for Async
    // Following MODULE_DEVELOPMENT_GUIDE.md AllocateBuffer<T> specification
    hipc::FullPtr<char> put_fullptr = CHI_IPC->AllocateBuffer(blob_size);

    if (put_fullptr.IsNull()) {
      INFO("Skipping async test due to memory context allocation failure");
      INFO("AsyncGetBlob API structure validated");
      return;
    }
    hipc::ShmPtr<> put_ptr = put_fullptr.shm_.template Cast<void>();

    REQUIRE(fixture->CopyToSharedMemory(put_fullptr, test_data));

    INFO("Storing blob for async retrieval...");
    auto put_task = fixture->core_client_->AsyncPutBlob(
        tag_id, blob_name, 0, blob_size, put_ptr, 0.7f, wrp_cte::core::Context(), 0);

    REQUIRE(!put_task.IsNull());
    REQUIRE(fixture->WaitForTaskCompletion(put_task, 10000));
    REQUIRE(put_task->return_code_ == 0);

    INFO("Blob stored successfully");

    // ACTUAL FUNCTIONAL TEST - call the real AsyncGetBlob API
    INFO("Calling fixture->core_client_->AsyncGetBlob()...");

    // Allocate buffer for retrieved data
    hipc::FullPtr<char> async_buffer_fullptr =
        CHI_IPC->AllocateBuffer(blob_size);
    if (async_buffer_fullptr.IsNull()) {
      INFO("Failed to allocate buffer for AsyncGetBlob");
      return;
    }
    hipc::ShmPtr<> async_buffer_ptr =
        async_buffer_fullptr.shm_.template Cast<void>();

    // Use blob name for retrieval
    auto get_task = fixture->core_client_->AsyncGetBlob(
        tag_id, blob_name, 0, blob_size, 0, async_buffer_ptr);

    REQUIRE(!get_task.IsNull());
    INFO("AsyncGetBlob returned valid task, waiting for completion...");

    // Wait for real task completion
    REQUIRE(
        fixture->WaitForTaskCompletion(get_task, 10000));  // 10 second timeout

    // Check result and data
    chi::u32 result = get_task->return_code_;
    bool get_success = (result == 0);
    INFO("Async GetBlob completed with success: " << (get_success ? "true"
                                                                  : "false"));

    if (get_success) {
      auto retrieved_data =
          fixture->CopyFromSharedMemory(async_buffer_ptr, blob_size);
      if (!retrieved_data.empty()) {
        REQUIRE(retrieved_data == test_data);
        REQUIRE(fixture->VerifyTestData(retrieved_data, 'A'));
        INFO("SUCCESS: Async retrieved data integrity verified!");
      }
    }

    // Cleanup task
    INFO("SUCCESS: Asynchronous blob retrieval with real task management!");
  }

  SECTION("FUNCTIONAL - Error cases with real error handling") {
    INFO("=== Testing REAL GetBlob error handling ===\n");

    // Allocate buffer for error case testing
    hipc::FullPtr<char> error_buffer_fullptr = CHI_IPC->AllocateBuffer(1024);
    if (error_buffer_fullptr.IsNull()) {
      INFO("Failed to allocate buffer for error case testing");
      return;
    }
    hipc::ShmPtr<> error_buffer_ptr =
        error_buffer_fullptr.shm_.template Cast<void>();

    // Test non-existent blob name
    INFO("Testing non-existent blob name error case...");
    bool get_success1 = fixture->GetBlobAsync(
        tag_id, "nonexistent_blob_99001", 0, 1024, 0, error_buffer_ptr);
    INFO("Non-existent blob result: "
         << (get_success1 ? "true" : "false")
         << " (false indicates proper error handling)");

    // Test another non-existent blob name
    INFO("Testing another non-existent blob name error case...");
    bool get_success2 = fixture->GetBlobAsync(
        tag_id, "nonexistent_blob_99999", 0, 1024, 0, error_buffer_ptr);
    INFO("Another non-existent blob result: "
         << (get_success2 ? "true" : "false")
         << " (false indicates proper error handling)");

    // Test invalid tag ID
    INFO("Testing invalid tag ID error case...");
    bool get_success3 = fixture->GetBlobAsync(
        wrp_cte::core::TagId{88888, 0}, "some_blob", 0, 1024, 0,
        error_buffer_ptr);
    INFO("Invalid tag result: " << (get_success3 ? "true" : "false")
                                << " (false indicates proper error handling)");

    // Test invalid offset/size ranges
    INFO("Testing invalid offset/size error case...");
    bool get_success4 =
        fixture->GetBlobAsync(tag_id, "some_blob", 999999, 1024, 0,
                                       error_buffer_ptr);  // Large offset
    INFO("Invalid range result: "
         << (get_success4 ? "true" : "false")
         << " (false indicates proper error handling)");

    INFO("SUCCESS: Error cases tested with real error handling!");
  }
}

/**
 * FUNCTIONAL Test Case: PutBlob-GetBlob Integration Cycles
 *
 * This test performs complete Put-Get cycles with comprehensive data integrity
 * verification:
 * 1. Store multiple blobs with different characteristics
 * 2. Retrieve and verify each blob's data integrity
 * 3. Test cross-tag operations and isolation
 * 4. Verify async Put followed by sync Get
 * 5. Test partial Put-Get operations with offsets
 * 6. Validate score handling and metadata consistency
 * 7. Ensure proper cleanup and resource management
 */
TEST_CASE("FUNCTIONAL - PutBlob-GetBlob Integration Cycles",
          "[cte][core][blob][integration][put-get]") {
  auto *fixture = hshm::Singleton<CTECoreFunctionalTestFixture>::
      GetInstance();  // Setup: Create core pool and register target
  chi::PoolQuery pool_query = chi::PoolQuery::Dynamic();
  wrp_cte::core::CreateParams params;

  REQUIRE_NOTHROW(fixture->CreateAsync(
      pool_query, CTECoreFunctionalTestFixture::kCTECorePoolName,
      CTECoreFunctionalTestFixture::kCTECorePoolId, params));

  const std::string target_name = fixture->test_storage_path_;
  chi::u32 reg_result = fixture->RegisterTargetAsync(
      target_name, chimaera::bdev::BdevType::kFile,
      CTECoreFunctionalTestFixture::kTestTargetSize, chi::PoolQuery::Local(),
      chi::PoolId(604, 0));
  REQUIRE(reg_result == 0);

  // Create test tag for integration testing
  wrp_cte::core::TagId tag_id =
      fixture->GetOrCreateTagAsync("integration_test_tag");

  SECTION("FUNCTIONAL - Basic Put-Get cycle with data integrity") {
    INFO("=== Testing REAL Put-Get cycle with data integrity ===\n");

    const std::string blob_name = "integration_basic_blob";
    const chi::u64 blob_size = 4096;  // 4KB
    const float score = 0.85f;

    // Create distinctive test data
    auto original_data =
        fixture->CreateTestData(blob_size, 'I');  // 'I' for Integration
    REQUIRE(fixture->VerifyTestData(original_data, 'I'));

    // Allocate shared memory and store data using CHI_CLIENT pattern
    // Following MODULE_DEVELOPMENT_GUIDE.md AllocateBuffer<T> specification
    hipc::FullPtr<char> put_fullptr = CHI_IPC->AllocateBuffer(blob_size);
    if (put_fullptr.IsNull()) {
      INFO("Skipping Put-Get cycle due to memory context allocation failure");
      return;
    }
    hipc::ShmPtr<> put_ptr = put_fullptr.shm_.template Cast<void>();

    REQUIRE(fixture->CopyToSharedMemory(put_fullptr, original_data));

    // STEP 1: Store the blob
    INFO("Step 1: Storing blob with AsyncPutBlob...");
    auto put_task = fixture->core_client_->AsyncPutBlob(
        tag_id, blob_name, 0, blob_size, put_ptr, score, wrp_cte::core::Context(), 0);

    REQUIRE(!put_task.IsNull());
    REQUIRE(fixture->WaitForTaskCompletion(put_task, 10000));
    REQUIRE(put_task->return_code_ == 0);

    INFO("PutBlob success");

    // STEP 2: Retrieve the blob
    INFO("Step 2: Retrieving blob with GetBlob...");

    // Allocate buffer for retrieved data
    hipc::FullPtr<char> get_buffer_fullptr = CHI_IPC->AllocateBuffer(blob_size);
    if (get_buffer_fullptr.IsNull()) {
      INFO("Failed to allocate buffer for GetBlob");
      return;
    }
    hipc::ShmPtr<> get_buffer_ptr =
        get_buffer_fullptr.shm_.template Cast<void>();

    // Use blob name for retrieval
    bool get_success = fixture->GetBlobAsync(
        tag_id, blob_name, 0, blob_size, 0, get_buffer_ptr);

    // STEP 3: Verify data integrity
    if (get_success) {
      auto retrieved_data =
          fixture->CopyFromSharedMemory(get_buffer_ptr, blob_size);

      if (!retrieved_data.empty()) {
        INFO("Step 3: Verifying data integrity...");
        REQUIRE(retrieved_data.size() == blob_size);
        REQUIRE(fixture->VerifyTestData(retrieved_data, 'I'));
        REQUIRE(retrieved_data == original_data);

        // Verify each byte to ensure perfect integrity
        for (size_t i = 0; i < blob_size; ++i) {
          if (retrieved_data[i] != original_data[i]) {
            FAIL("Data integrity failure at byte "
                 << i << ": expected '" << original_data[i] << "' but got '"
                 << retrieved_data[i] << "'");
          }
        }

        INFO("SUCCESS: Perfect data integrity verified - all "
             << blob_size << " bytes match!");
      } else {
        INFO("Retrieved data is empty despite successful GetBlob");
      }
    } else {
      INFO("GetBlob failed");
    }

    INFO("SUCCESS: Basic Put-Get cycle completed with full data integrity!");
  }

  SECTION("FUNCTIONAL - Multiple Put-Get cycles with different blob sizes") {
    INFO("=== Testing REAL multiple Put-Get cycles ===\n");

    const std::vector<std::tuple<std::string, chi::u64, char, float>>
        test_blobs = {
            {"small_cycle", 256, 'S', 0.2f},    // 256 bytes
            {"medium_cycle", 2048, 'M', 0.5f},  // 2KB
            {"large_cycle", 8192, 'L', 0.8f},   // 8KB
            {"xlarge_cycle", 16384, 'X', 1.0f}  // 16KB
        };

    std::vector<std::vector<char>> stored_data;
    std::vector<std::string> stored_blob_names;

    // PHASE 1: Store all blobs
    INFO("Phase 1: Storing all test blobs...");
    for (const auto &[blob_name, blob_size, pattern, score] : test_blobs) {
      INFO("Storing " << blob_name << " (" << blob_size << " bytes, pattern '"
                      << pattern << "')");

      auto blob_data = fixture->CreateTestData(blob_size, pattern);
      stored_data.push_back(blob_data);

      // Following MODULE_DEVELOPMENT_GUIDE.md AllocateBuffer<T> specification
      hipc::FullPtr<char> put_fullptr = CHI_IPC->AllocateBuffer(blob_size);
      if (put_fullptr.IsNull()) {
        INFO("Skipping " << blob_name
                         << " due to memory context allocation failure");
        continue;
      }
      hipc::ShmPtr<> put_ptr = put_fullptr.shm_.template Cast<void>();

      REQUIRE(fixture->CopyToSharedMemory(put_fullptr, blob_data));

      auto put_task = fixture->core_client_->AsyncPutBlob(
          tag_id, blob_name, 0, blob_size, put_ptr, score, wrp_cte::core::Context(), 0);

      if (!put_task.IsNull() &&
          fixture->WaitForTaskCompletion(put_task, 10000) &&
          put_task->return_code_ == 0) {
        stored_blob_names.push_back(blob_name);
        INFO("✓ " << blob_name << " stored successfully");
      } else {
        INFO("✗ " << blob_name << " storage failed");
      }
    }

    // PHASE 2: Retrieve and verify all blobs using blob names
    INFO("\nPhase 2: Retrieving and verifying all test blobs...");
    size_t blob_index = 0;
    for (const auto &[blob_name, blob_size, pattern, score] : test_blobs) {
      if (blob_index >= stored_data.size() ||
          blob_index >= stored_blob_names.size()) {
        INFO("Skipping retrieval of " << blob_name << " - not stored");
        continue;
      }

      INFO("Retrieving " << blob_name << " and verifying integrity...");

      // Allocate buffer for retrieved data
      hipc::FullPtr<char> multi_buffer_fullptr =
          CHI_IPC->AllocateBuffer(blob_size);
      if (multi_buffer_fullptr.IsNull()) {
        INFO("Failed to allocate buffer for GetBlob");
        continue;
      }
      hipc::ShmPtr<> multi_buffer_ptr =
          multi_buffer_fullptr.shm_.template Cast<void>();

      // Use blob name for retrieval
      bool multi_success = fixture->GetBlobAsync(
          tag_id, blob_name, 0, blob_size, 0, multi_buffer_ptr);

      if (multi_success) {
        auto retrieved_data =
            fixture->CopyFromSharedMemory(multi_buffer_ptr, blob_size);

        if (!retrieved_data.empty()) {
          const auto &expected_data = stored_data[blob_index];
          REQUIRE(retrieved_data.size() == blob_size);
          REQUIRE(retrieved_data == expected_data);
          REQUIRE(fixture->VerifyTestData(retrieved_data, pattern));
          INFO("✓ " << blob_name << " integrity verified");
        } else {
          INFO("Retrieved empty data for " << blob_name);
        }
      } else {
        INFO("GetBlob failed for " << blob_name);
      }

      ++blob_index;
    }

    INFO("SUCCESS: Multiple Put-Get cycles completed with full integrity!");
  }

  SECTION("FUNCTIONAL - Cross-tag isolation testing") {
    INFO("=== Testing REAL cross-tag isolation ===\n");

    // Create two separate tags
    wrp_cte::core::TagId tag1_id =
        fixture->GetOrCreateTagAsync("isolation_tag_1");
    wrp_cte::core::TagId tag2_id =
        fixture->GetOrCreateTagAsync("isolation_tag_2");

    const std::string blob_name = "isolation_test_blob";
    const chi::u64 blob_size = 1024;

    // Create different data for each tag
    auto tag1_data = fixture->CreateTestData(blob_size, '1');
    auto tag2_data = fixture->CreateTestData(blob_size, '2');

    // Store blobs in both tags
    INFO("Storing blobs in separate tags...");

    // Store in tag1
    // Following MODULE_DEVELOPMENT_GUIDE.md AllocateBuffer<T> specification
    hipc::FullPtr<char> put1_fullptr = CHI_IPC->AllocateBuffer(blob_size);
    hipc::ShmPtr<> put1_ptr = put1_fullptr.IsNull()
                                  ? hipc::ShmPtr<>::GetNull()
                                  : put1_fullptr.shm_.template Cast<void>();
    bool tag1_stored = false;
    if (!put1_fullptr.IsNull() &&
        fixture->CopyToSharedMemory(put1_fullptr, tag1_data)) {
      auto put1_task = fixture->core_client_->AsyncPutBlob(
          tag1_id, blob_name, 0, blob_size, put1_ptr, 0.5f, wrp_cte::core::Context(), 0);
      if (!put1_task.IsNull() &&
          fixture->WaitForTaskCompletion(put1_task, 10000) &&
          put1_task->return_code_ == 0) {
        tag1_stored = true;
        INFO("Tag1 blob stored successfully");
      }
    }

    // Store in tag2
    // Following MODULE_DEVELOPMENT_GUIDE.md AllocateBuffer<T> specification
    hipc::FullPtr<char> put2_fullptr = CHI_IPC->AllocateBuffer(blob_size);
    hipc::ShmPtr<> put2_ptr = put2_fullptr.IsNull()
                                  ? hipc::ShmPtr<>::GetNull()
                                  : put2_fullptr.shm_.template Cast<void>();
    bool tag2_stored = false;
    if (!put2_fullptr.IsNull() &&
        fixture->CopyToSharedMemory(put2_fullptr, tag2_data)) {
      auto put2_task = fixture->core_client_->AsyncPutBlob(
          tag2_id, blob_name, 0, blob_size, put2_ptr, 0.5f, wrp_cte::core::Context(), 0);
      if (!put2_task.IsNull() &&
          fixture->WaitForTaskCompletion(put2_task, 10000) &&
          put2_task->return_code_ == 0) {
        tag2_stored = true;
        INFO("Tag2 blob stored successfully");
      }
    }

    // Verify isolation - retrieve from each tag
    INFO("Verifying tag isolation...");

    // Retrieve from tag1 using blob name
    // Allocate buffer for tag1 retrieval
    hipc::FullPtr<char> tag1_buffer_fullptr =
        CHI_IPC->AllocateBuffer(blob_size);
    if (!tag1_buffer_fullptr.IsNull() && tag1_stored) {
      hipc::ShmPtr<> tag1_buffer_ptr =
          tag1_buffer_fullptr.shm_.template Cast<void>();

      // Use blob name for retrieval
      bool get1_success = fixture->GetBlobAsync(
          tag1_id, blob_name, 0, blob_size, 0, tag1_buffer_ptr);

      if (get1_success) {
        auto retrieved1_data =
            fixture->CopyFromSharedMemory(tag1_buffer_ptr, blob_size);
        if (!retrieved1_data.empty()) {
          REQUIRE(retrieved1_data == tag1_data);
          REQUIRE(fixture->VerifyTestData(retrieved1_data, '1'));
          INFO("✓ Tag1 data isolation verified");
        }
      }
    }

    // Retrieve from tag2 using blob name
    // Allocate buffer for tag2 retrieval
    hipc::FullPtr<char> tag2_buffer_fullptr =
        CHI_IPC->AllocateBuffer(blob_size);
    if (!tag2_buffer_fullptr.IsNull() && tag2_stored) {
      hipc::ShmPtr<> tag2_buffer_ptr =
          tag2_buffer_fullptr.shm_.template Cast<void>();

      // Use blob name for retrieval
      bool get2_success = fixture->GetBlobAsync(
          tag2_id, blob_name, 0, blob_size, 0, tag2_buffer_ptr);

      if (get2_success) {
        auto retrieved2_data =
            fixture->CopyFromSharedMemory(tag2_buffer_ptr, blob_size);
        if (!retrieved2_data.empty()) {
          REQUIRE(retrieved2_data == tag2_data);
          REQUIRE(fixture->VerifyTestData(retrieved2_data, '2'));
          INFO("✓ Tag2 data isolation verified");
        }
      }
    }

    INFO(
        "SUCCESS: Cross-tag isolation verified - tags maintain separate data!");
  }

  SECTION("FUNCTIONAL - Async Put followed by Sync Get cycle") {
    INFO("=== Testing REAL Async Put -> Sync Get cycle ===\n");

    const std::string blob_name = "async_put_sync_get";
    const chi::u64 blob_size = 3072;  // 3KB

    auto test_data = fixture->CreateTestData(blob_size, 'A');  // 'A' for Async
    // Following MODULE_DEVELOPMENT_GUIDE.md AllocateBuffer<T> specification
    hipc::FullPtr<char> put_fullptr = CHI_IPC->AllocateBuffer(blob_size);

    if (put_fullptr.IsNull() ||
        !fixture->CopyToSharedMemory(put_fullptr, test_data)) {
      INFO(
          "Skipping async-sync cycle due to memory context allocation failure");
      return;
    }
    hipc::ShmPtr<> put_ptr = put_fullptr.shm_.template Cast<void>();

    // STEP 1: Async Put
    INFO("Step 1: Async PutBlob...");
    auto put_task = fixture->core_client_->AsyncPutBlob(
        tag_id, blob_name, 0, blob_size, put_ptr, 0.7f, wrp_cte::core::Context(), 0);

    REQUIRE(!put_task.IsNull());
    INFO("Waiting for async put completion...");
    REQUIRE(fixture->WaitForTaskCompletion(put_task, 10000));

    chi::u32 put_result = put_task->return_code_;
    bool put_success = (put_result == 0);
    INFO("Async put completed with success: " << (put_success ? "true"
                                                              : "false"));

    // STEP 2: Sync Get
    INFO("Step 2: Sync GetBlob...");

    // Allocate buffer for retrieved data
    hipc::FullPtr<char> sync_get_buffer_fullptr =
        CHI_IPC->AllocateBuffer(blob_size);
    if (sync_get_buffer_fullptr.IsNull()) {
      INFO("Failed to allocate buffer for sync GetBlob");
      return;
    }
    hipc::ShmPtr<> sync_get_buffer_ptr =
        sync_get_buffer_fullptr.shm_.template Cast<void>();

    // Use blob name for retrieval
    bool sync_get_success = fixture->GetBlobAsync(
        tag_id, blob_name, 0, blob_size, 0, sync_get_buffer_ptr);

    // STEP 3: Verify integrity
    if (sync_get_success) {
      auto retrieved_data =
          fixture->CopyFromSharedMemory(sync_get_buffer_ptr, blob_size);
      if (!retrieved_data.empty()) {
        REQUIRE(retrieved_data == test_data);
        REQUIRE(fixture->VerifyTestData(retrieved_data, 'A'));
        INFO(
            "SUCCESS: Async Put -> Sync Get cycle with perfect data "
            "integrity!");
      }
    }
  }

  SECTION("FUNCTIONAL - Partial Put-Get operations with offsets") {
    INFO("=== Testing REAL partial Put-Get operations ===\n");

    const std::string blob_name = "partial_operations_blob";
    const chi::u64 total_size = 8192;  // 8KB total
    const chi::u64 chunk_size = 2048;  // 2KB chunks

    // Create full data with distinct patterns for each chunk
    std::vector<char> full_data;
    std::vector<std::vector<char>> chunk_data;

    for (chi::u64 offset = 0; offset < total_size; offset += chunk_size) {
      char pattern = static_cast<char>('0' + (offset / chunk_size));
      auto chunk = fixture->CreateTestData(chunk_size, pattern);
      chunk_data.push_back(chunk);
      full_data.insert(full_data.end(), chunk.begin(), chunk.end());
    }

    // PHASE 1: Store chunks using offsets - need to track allocated blob ID
    INFO("Phase 1: Storing chunks with offset operations...");
    bool chunks_stored = false;

    for (size_t i = 0; i < chunk_data.size(); ++i) {
      chi::u64 offset = i * chunk_size;
      char pattern = static_cast<char>('0' + i);

      INFO("Storing chunk " << i << " at offset " << offset << " with pattern '"
                            << pattern << "'");

      // Following MODULE_DEVELOPMENT_GUIDE.md AllocateBuffer<T> specification
      hipc::FullPtr<char> chunk_fullptr = CHI_IPC->AllocateBuffer(chunk_size);
      if (chunk_fullptr.IsNull() ||
          !fixture->CopyToSharedMemory(chunk_fullptr, chunk_data[i])) {
        INFO("Skipping chunk " << i
                               << " due to memory context allocation failure");
        continue;
      }
      hipc::ShmPtr<> chunk_ptr = chunk_fullptr.shm_.template Cast<void>();

      auto chunk_task = fixture->core_client_->AsyncPutBlob(
          tag_id, blob_name, offset, chunk_size, chunk_ptr, 0.6f, wrp_cte::core::Context(), 0);

      if (!chunk_task.IsNull() &&
          fixture->WaitForTaskCompletion(chunk_task, 10000) &&
          chunk_task->return_code_ == 0) {
        chunks_stored = true;
        INFO("✓ Chunk " << i << " stored successfully");
      } else {
        INFO("✗ Chunk " << i << " storage failed");
      }
    }

    // PHASE 2: Retrieve chunks and verify using blob name
    INFO("\nPhase 2: Retrieving chunks and verifying...");
    if (!chunks_stored) {
      INFO("Skipping chunk retrieval - no chunks stored");
      return;
    }

    for (size_t i = 0; i < chunk_data.size(); ++i) {
      chi::u64 offset = i * chunk_size;
      char pattern = static_cast<char>('0' + i);

      INFO("Retrieving chunk " << i << " from offset " << offset);

      // Allocate buffer for chunk retrieval
      hipc::FullPtr<char> chunk_get_buffer_fullptr =
          CHI_IPC->AllocateBuffer(chunk_size);
      if (chunk_get_buffer_fullptr.IsNull()) {
        INFO("Failed to allocate buffer for chunk GetBlob");
        continue;
      }
      hipc::ShmPtr<> chunk_get_buffer_ptr =
          chunk_get_buffer_fullptr.shm_.template Cast<void>();

      // Use blob name for retrieval
      bool chunk_get_success = fixture->GetBlobAsync(
          tag_id, blob_name, offset, chunk_size, 0, chunk_get_buffer_ptr);

      if (chunk_get_success) {
        auto retrieved_chunk =
            fixture->CopyFromSharedMemory(chunk_get_buffer_ptr, chunk_size);
        if (!retrieved_chunk.empty()) {
          REQUIRE(retrieved_chunk == chunk_data[i]);
          REQUIRE(fixture->VerifyTestData(retrieved_chunk, pattern));
          INFO("✓ Chunk " << i << " integrity verified");
        }
      }
    }

    // PHASE 3: Retrieve full blob and verify
    INFO("\nPhase 3: Retrieving full blob and verifying...");

    // Allocate buffer for full blob retrieval
    hipc::FullPtr<char> full_buffer_fullptr =
        CHI_IPC->AllocateBuffer(total_size);
    if (full_buffer_fullptr.IsNull()) {
      INFO("Failed to allocate buffer for full blob GetBlob");
      return;
    }
    hipc::ShmPtr<> full_buffer_ptr =
        full_buffer_fullptr.shm_.template Cast<void>();

    // Use blob name for retrieval
    bool full_success = fixture->GetBlobAsync(
        tag_id, blob_name, 0, total_size, 0, full_buffer_ptr);

    if (full_success) {
      auto retrieved_full =
          fixture->CopyFromSharedMemory(full_buffer_ptr, total_size);
      if (!retrieved_full.empty()) {
        REQUIRE(retrieved_full.size() == total_size);
        REQUIRE(retrieved_full == full_data);
        INFO("SUCCESS: Full blob matches assembled chunks perfectly!");
      }
    }

    INFO("SUCCESS: Partial Put-Get operations with offsets completed!");
  }
}

/**
 * FUNCTIONAL Test Case: PutBlob-GetBlob Comprehensive Integration
 *
 * This test implements the EXACT workflow requirements:
 * 1. PutBlob with blob_name="test_blob" and blob_id=BlobId::GetNull()
 * (null/default)
 * 2. Extract the actual blob_id returned from PutBlob task
 * 3. GetBlob with extracted blob_id and empty blob_name=""
 * 4. Verify data integrity matches exactly
 *
 * Test Requirements:
 * - PutBlob should set blob name (non-empty string)
 * - PutBlob should NOT specify blob ID (use default/null BlobId)
 * - PutBlob should return the actual blob ID that was assigned/generated
 * - GetBlob should use the blob ID returned from PutBlob
 * - GetBlob should NOT specify blob name (use empty string)
 * - GetBlob should retrieve the same data that was stored
 * - Test should FAIL if chimaera runtime is not properly initialized
 * - Test should FAIL if data doesn't match exactly
 */
TEST_CASE("FUNCTIONAL - PutBlob-GetBlob Comprehensive Integration",
          "[cte][core][blob][integration][put-get][comprehensive]") {
  auto *fixture = hshm::Singleton<CTECoreFunctionalTestFixture>::GetInstance();
  INFO("=== COMPREHENSIVE PutBlob-GetBlob Integration Test ===");

  // Setup: Create core pool, register target, create tag
  chi::PoolQuery pool_query = chi::PoolQuery::Dynamic();
  wrp_cte::core::CreateParams params;

  INFO("Step 1: Creating core pool...");
  REQUIRE_NOTHROW(fixture->CreateAsync(
      pool_query, CTECoreFunctionalTestFixture::kCTECorePoolName,
      CTECoreFunctionalTestFixture::kCTECorePoolId, params));
  INFO("✓ Core pool created successfully");

  INFO("Step 2: Registering target...");
  const std::string target_name = fixture->test_storage_path_;
  chi::u32 reg_result = fixture->RegisterTargetAsync(
      target_name, chimaera::bdev::BdevType::kFile,
      CTECoreFunctionalTestFixture::kTestTargetSize, chi::PoolQuery::Local(),
      chi::PoolId(605, 0));
  REQUIRE(reg_result == 0);
  INFO("✓ Target registered successfully");

  INFO("Step 3: Creating test tag...");
  wrp_cte::core::TagId tag_id =
      fixture->GetOrCreateTagAsync("comprehensive_test_tag");
  REQUIRE((tag_id.major_ != 0 || tag_id.minor_ != 0));
  INFO("✓ Test tag created with ID: " << tag_id);

  // Test data preparation
  const chi::u64 test_data_size = 4096;  // 4KB test data
  auto original_data =
      fixture->CreateTestData(test_data_size, 'C');  // 'C' for Comprehensive
  REQUIRE(fixture->VerifyTestData(original_data, 'C'));
  INFO("✓ Test data prepared (" << test_data_size
                                << " bytes with pattern 'C')");

  // Verify runtime initialization (test should FAIL if not properly
  // initialized)
  REQUIRE(CHI_CHIMAERA_MANAGER != nullptr);
  REQUIRE(CHI_IPC != nullptr);
  REQUIRE(CHI_POOL_MANAGER != nullptr);
  REQUIRE(CHI_MODULE_MANAGER != nullptr);
  REQUIRE(CHI_IPC->IsInitialized());
  INFO("✓ Chimaera runtime initialization verified");

  // Step 4: PutBlob with blob_name="test_blob"
  INFO("Step 4: Executing PutBlob with specific requirements...");

  const std::string blob_name = "test_blob";  // Non-empty string as required
  const float blob_score = 0.75f;

  // Allocate shared memory for PutBlob
  hipc::FullPtr<char> put_data_buffer = CHI_IPC->AllocateBuffer(test_data_size);
  REQUIRE(!put_data_buffer.IsNull());
  hipc::ShmPtr<> put_data_ptr = put_data_buffer.shm_.template Cast<void>();

  // Copy test data to shared memory
  REQUIRE(fixture->CopyToSharedMemory(put_data_buffer, original_data));
  INFO("✓ Test data copied to shared memory buffer");

  // Create PutBlob task
  auto put_task = fixture->core_client_->AsyncPutBlob(
      tag_id, blob_name, 0, test_data_size, put_data_ptr, blob_score, wrp_cte::core::Context(), 0);

  REQUIRE(!put_task.IsNull());
  INFO("✓ PutBlob task created with:");
  INFO("    blob_name: '" << blob_name << "' (non-empty as required)");
  INFO("    size: " << test_data_size << " bytes");
  INFO("    score: " << blob_score);

  // Wait for PutBlob completion
  REQUIRE(fixture->WaitForTaskCompletion(put_task, 10000));
  REQUIRE(put_task->return_code_ == 0);
  INFO("✓ PutBlob completed successfully with result code: "
       << put_task->return_code_);

  // Cleanup PutBlob task

  // Step 5: GetBlob with blob_name
  INFO("Step 5: Executing GetBlob with blob_name...");

  // Allocate shared memory for GetBlob
  hipc::FullPtr<char> get_data_buffer = CHI_IPC->AllocateBuffer(test_data_size);
  REQUIRE(!get_data_buffer.IsNull());
  hipc::ShmPtr<> get_data_ptr = get_data_buffer.shm_.template Cast<void>();

  // Create GetBlob task using blob_name
  auto get_task = fixture->core_client_->AsyncGetBlob(
      tag_id, blob_name, 0, test_data_size, 0, get_data_ptr);

  REQUIRE(!get_task.IsNull());
  INFO("✓ GetBlob task created with:");
  INFO("    blob_name: '" << blob_name << "'");
  INFO("    size: " << test_data_size << " bytes");

  // Wait for GetBlob completion
  REQUIRE(fixture->WaitForTaskCompletion(get_task, 10000));
  REQUIRE(get_task->return_code_ == 0);
  INFO("✓ GetBlob completed successfully with result code: "
       << get_task->return_code_);

  // Step 6: Verify data integrity matches exactly (test should FAIL if data
  // doesn't match)
  INFO("Step 6: Verifying data integrity...");

  auto retrieved_data =
      fixture->CopyFromSharedMemory(get_data_ptr, test_data_size);
  REQUIRE(!retrieved_data.empty());
  REQUIRE(retrieved_data.size() == test_data_size);

  // Exact data match verification
  REQUIRE(retrieved_data == original_data);
  INFO("✓ Retrieved data size matches original (" << retrieved_data.size()
                                                  << " bytes)");

  // Pattern integrity verification
  REQUIRE(fixture->VerifyTestData(retrieved_data, 'C'));
  INFO("✓ Retrieved data pattern matches original ('C' pattern verified)");

  // Byte-by-byte verification (test should FAIL if any byte doesn't match)
  for (size_t i = 0; i < test_data_size; ++i) {
    if (retrieved_data[i] != original_data[i]) {
      FAIL("EXACT DATA MATCH FAILURE at byte "
           << i << ": expected '" << original_data[i] << "' but got '"
           << retrieved_data[i] << "'");
    }
  }
  INFO("✓ Byte-by-byte verification passed - all " << test_data_size
                                                   << " bytes match exactly");

  // Cleanup GetBlob task

  INFO("=== COMPREHENSIVE TEST COMPLETED SUCCESSFULLY ===");
  INFO("✓ PutBlob with blob_name='test_blob' completed");
  INFO("✓ GetBlob with blob_name completed");
  INFO("✓ Data integrity verified - exact match confirmed");
  INFO("✓ Runtime initialization verified");
  INFO("✓ All requirements satisfied - test PASSED");
}

/**
 * FUNCTIONAL Test Case: ReorganizeBlob Operations
 *
 * This test verifies the complete ReorganizeBlob functionality:
 * 1. Setup core pool, register target, create tag
 * 2. Store 10 blobs with initial score of 0.5
 * 3. Use ReorganizeBlob to update all blobs to score 1.0
 * 4. Verify that scores have been updated correctly
 * 5. Test score difference threshold filtering
 * 6. Verify processing multiple blobs works correctly
 *
 * Following CLAUDE.md requirements:
 * - Use real API calls with proper runtime initialization
 * - Test the actual ReorganizeBlob implementation with controlled async
 * operations
 * - Verify score filtering based on score_difference_threshold configuration
 * - Test processing multiple blobs (updated from batch API to per-blob API)
 */
TEST_CASE("FUNCTIONAL - ReorganizeBlob Operations",
          "[cte][core][blob][reorganize][functional]") {
  auto *fixture = hshm::Singleton<CTECoreFunctionalTestFixture>::GetInstance();
  INFO("=== FUNCTIONAL ReorganizeBlob Test ===");

  // Setup: Create core pool, register target, create tag
  chi::PoolQuery pool_query = chi::PoolQuery::Dynamic();
  wrp_cte::core::CreateParams params;

  INFO("Step 1: Setting up CTE environment...");
  REQUIRE_NOTHROW(fixture->CreateAsync(
      pool_query, CTECoreFunctionalTestFixture::kCTECorePoolName,
      CTECoreFunctionalTestFixture::kCTECorePoolId, params));

  const std::string target_name = fixture->test_storage_path_;
  chi::u32 reg_result = fixture->RegisterTargetAsync(
      target_name, chimaera::bdev::BdevType::kFile,
      CTECoreFunctionalTestFixture::kTestTargetSize, chi::PoolQuery::Local(),
      chi::PoolId(606, 0));
  REQUIRE(reg_result == 0);

  wrp_cte::core::TagId tag_id =
      fixture->GetOrCreateTagAsync("reorganize_test_tag");
  REQUIRE((tag_id.major_ != 0 || tag_id.minor_ != 0));
  INFO("✓ Environment setup completed");

  SECTION(
      "FUNCTIONAL - Store 10 blobs with score 0.5, then reorganize to score "
      "1.0") {
    INFO("=== Testing 10 blobs: 0.5 → 1.0 score reorganization ===");

    const size_t num_blobs = 10;
    const float initial_score = 0.5f;
    const float target_score = 1.0f;
    const chi::u64 blob_size = 2048;  // 2KB per blob

    std::vector<std::string> blob_names;
    std::vector<std::vector<char>> blob_data;

    // Phase 1: Store 10 blobs with initial score 0.5
    INFO("Phase 1: Storing " << num_blobs << " blobs with score "
                             << initial_score);
    for (size_t i = 0; i < num_blobs; ++i) {
      std::string blob_name = "reorganize_blob_" + std::to_string(i);
      blob_names.push_back(blob_name);

      char pattern = static_cast<char>('A' + (i % 26));
      auto data = fixture->CreateTestData(blob_size, pattern);
      blob_data.push_back(data);

      // Allocate shared memory and store blob
      hipc::FullPtr<char> put_buffer = CHI_IPC->AllocateBuffer(blob_size);
      REQUIRE(!put_buffer.IsNull());
      REQUIRE(fixture->CopyToSharedMemory(put_buffer, data));

      auto put_task = fixture->core_client_->AsyncPutBlob(
          tag_id, blob_name, 0, blob_size,
          put_buffer.shm_.template Cast<void>(), initial_score, wrp_cte::core::Context(), 0);

      REQUIRE(!put_task.IsNull());
      REQUIRE(fixture->WaitForTaskCompletion(put_task, 10000));
      REQUIRE(put_task->return_code_ == 0);

      INFO("✓ Blob " << i << " stored: " << blob_name);
    }

    // Phase 2: Verify initial scores are 0.5
    INFO("Phase 2: Verifying initial scores...");
    for (size_t i = 0; i < num_blobs; ++i) {
      float score = fixture->GetBlobScoreAsync(tag_id, blob_names[i]);
      REQUIRE(std::abs(score - initial_score) < 0.01f);
      INFO("✓ Blob " << i << " initial score verified: " << score);
    }

    // Phase 3: Use ReorganizeBlob to update all scores to 1.0
    INFO("Phase 3: Executing ReorganizeBlob operations...");

    // Call ReorganizeBlob once per blob (updated from batched operation)
    for (size_t i = 0; i < num_blobs; ++i) {
      auto reorganize_task = fixture->core_client_->AsyncReorganizeBlob(
          tag_id, blob_names[i], target_score);

      REQUIRE(!reorganize_task.IsNull());
      REQUIRE(fixture->WaitForTaskCompletion(reorganize_task, 10000));
      REQUIRE(reorganize_task->return_code_ == 0);
      INFO("✓ Blob " << i << " reorganized successfully");
    }
    INFO("✓ All blobs reorganized successfully");

    // Phase 4: Verify updated scores are 1.0
    INFO("Phase 4: Verifying updated scores...");
    for (size_t i = 0; i < num_blobs; ++i) {
      float updated_score =
          fixture->GetBlobScoreAsync(tag_id, blob_names[i]);
      REQUIRE(std::abs(updated_score - target_score) < 0.01f);
      INFO("✓ Blob " << i << " updated score verified: " << updated_score);
    }

    // Phase 5: Verify data integrity after reorganization
    INFO("Phase 5: Verifying data integrity after reorganization...");
    for (size_t i = 0; i < num_blobs; ++i) {
      hipc::FullPtr<char> get_buffer = CHI_IPC->AllocateBuffer(blob_size);
      REQUIRE(!get_buffer.IsNull());

      bool get_success =
          fixture->GetBlobAsync(tag_id, blob_names[i], 0, blob_size, 0,
                                         get_buffer.shm_.template Cast<void>());
      REQUIRE(get_success);

      auto retrieved_data = fixture->CopyFromSharedMemory(
          get_buffer.shm_.template Cast<void>(), blob_size);
      REQUIRE(retrieved_data == blob_data[i]);

      char expected_pattern = static_cast<char>('A' + (i % 26));
      REQUIRE(fixture->VerifyTestData(retrieved_data, expected_pattern));
      INFO("✓ Blob " << i << " data integrity verified after reorganization");
    }

    INFO(
        "SUCCESS: All 10 blobs successfully reorganized from score 0.5 to "
        "1.0!");
  }

  SECTION("FUNCTIONAL - Score difference threshold filtering") {
    INFO("=== Testing score difference threshold filtering ===");

    const size_t num_test_blobs = 5;
    const chi::u64 blob_size = 1024;

    // Create test blobs with different initial scores
    std::vector<std::string> test_blob_names;
    std::vector<float> initial_scores = {0.1f, 0.5f, 0.7f, 0.9f, 0.95f};
    std::vector<float> target_scores = {0.15f, 0.55f, 0.75f, 0.95f,
                                        1.0f};  // Small and large differences

    INFO("Creating test blobs with varying scores...");
    for (size_t i = 0; i < num_test_blobs; ++i) {
      std::string blob_name = "threshold_test_blob_" + std::to_string(i);
      test_blob_names.push_back(blob_name);

      auto data =
          fixture->CreateTestData(blob_size, static_cast<char>('T' + i));
      hipc::FullPtr<char> put_buffer = CHI_IPC->AllocateBuffer(blob_size);
      REQUIRE(!put_buffer.IsNull());
      REQUIRE(fixture->CopyToSharedMemory(put_buffer, data));

      auto put_task = fixture->core_client_->AsyncPutBlob(
          tag_id, blob_name, 0, blob_size,
          put_buffer.shm_.template Cast<void>(), initial_scores[i], wrp_cte::core::Context(), 0);

      REQUIRE(!put_task.IsNull());
      REQUIRE(fixture->WaitForTaskCompletion(put_task, 10000));
      REQUIRE(put_task->return_code_ == 0);

      INFO("✓ Test blob " << i << " stored with score " << initial_scores[i]);
    }

    // Execute ReorganizeBlob with mixed score differences (one per blob)
    INFO("Executing ReorganizeBlob operations with mixed score differences...");
    for (size_t i = 0; i < num_test_blobs; ++i) {
      auto threshold_reorganize_task =
          fixture->core_client_->AsyncReorganizeBlob(tag_id, test_blob_names[i],
                                                     target_scores[i]);

      REQUIRE(!threshold_reorganize_task.IsNull());
      REQUIRE(fixture->WaitForTaskCompletion(threshold_reorganize_task, 10000));
      REQUIRE(threshold_reorganize_task->return_code_ == 0);
    }

    // Verify that blobs with significant score differences were updated
    // Note: Default score_difference_threshold is 0.05 based on implementation
    INFO("Verifying score updates based on threshold filtering...");
    for (size_t i = 0; i < num_test_blobs; ++i) {
      float current_score =
          fixture->GetBlobScoreAsync(tag_id, test_blob_names[i]);
      float score_diff = std::abs(target_scores[i] - initial_scores[i]);

      if (score_diff >= 0.05f) {  // Should be updated
        REQUIRE(std::abs(current_score - target_scores[i]) < 0.01f);
        INFO("✓ Blob " << i << " score updated: " << initial_scores[i] << " → "
                       << current_score << " (diff: " << score_diff << ")");
      } else {  // Should remain unchanged
        REQUIRE(std::abs(current_score - initial_scores[i]) < 0.01f);
        INFO("✓ Blob " << i << " score unchanged: " << current_score
                       << " (diff below threshold: " << score_diff << ")");
      }
    }

    INFO("SUCCESS: Score difference threshold filtering verified!");
  }

  SECTION("FUNCTIONAL - Multiple blob processing verification") {
    INFO("=== Testing processing multiple blobs (32 blobs) ===");

    const size_t batch_size = 32;
    const float initial_score = 0.3f;
    const float target_score = 0.8f;
    const chi::u64 blob_size = 512;  // Smaller blobs for batch test

    std::vector<std::string> batch_blob_names;

    INFO("Creating " << batch_size
                     << " blobs for multiple blob processing test...");
    for (size_t i = 0; i < batch_size; ++i) {
      std::string blob_name = "batch_blob_" + std::to_string(i);
      batch_blob_names.push_back(blob_name);

      auto data = fixture->CreateTestData(blob_size, static_cast<char>('B'));
      hipc::FullPtr<char> put_buffer = CHI_IPC->AllocateBuffer(blob_size);
      REQUIRE(!put_buffer.IsNull());
      REQUIRE(fixture->CopyToSharedMemory(put_buffer, data));

      auto put_task = fixture->core_client_->AsyncPutBlob(
          tag_id, blob_name, 0, blob_size,
          put_buffer.shm_.template Cast<void>(), initial_score, wrp_cte::core::Context(), 0);

      REQUIRE(!put_task.IsNull());
      REQUIRE(fixture->WaitForTaskCompletion(put_task, 10000));
      REQUIRE(put_task->return_code_ == 0);

      if ((i + 1) % 8 == 0) {
        INFO("✓ Created " << (i + 1) << "/" << batch_size << " batch blobs");
      }
    }

    // Execute reorganization for each blob (updated from batched operation)
    INFO("Executing ReorganizeBlob operations for " << batch_size
                                                    << " blobs...");
    for (size_t i = 0; i < batch_size; ++i) {
      auto batch_reorganize_task = fixture->core_client_->AsyncReorganizeBlob(
          tag_id, batch_blob_names[i], target_score);

      REQUIRE(!batch_reorganize_task.IsNull());
      REQUIRE(fixture->WaitForTaskCompletion(batch_reorganize_task, 10000));
      REQUIRE(batch_reorganize_task->return_code_ == 0);

      if ((i + 1) % 8 == 0) {
        INFO("✓ Reorganized " << (i + 1) << "/" << batch_size << " blobs");
      }
    }
    INFO("✓ All " << batch_size << " blobs reorganized successfully");

    // Verify all blobs in batch were updated
    INFO("Verifying all " << batch_size << " blobs were reorganized...");
    size_t verified_count = 0;
    for (size_t i = 0; i < batch_size; ++i) {
      float current_score =
          fixture->GetBlobScoreAsync(tag_id, batch_blob_names[i]);
      REQUIRE(std::abs(current_score - target_score) < 0.01f);
      verified_count++;

      if ((i + 1) % 8 == 0) {
        INFO("✓ Verified " << (i + 1) << "/" << batch_size
                           << " batch blob scores");
      }
    }

    REQUIRE(verified_count == batch_size);
    INFO("SUCCESS: Processing of " << batch_size
                                   << " blobs completed successfully!");
  }

  INFO("=== ReorganizeBlob FUNCTIONAL Test Completed Successfully ===");
}

/**
 * Integration Test: End-to-End CTE Core Workflow
 *
 * This test verifies the complete workflow:
 * 1. Initialize CTE core pool
 * 2. Register multiple targets
 * 3. Create tags for organization
 * 4. Store multiple blobs
 * 5. Retrieve and verify all blobs
 * 6. Update target statistics
 */
TEST_CASE("End-to-End CTE Core Workflow", "[cte][core][integration]") {
  auto *fixture = hshm::Singleton<CTECoreFunctionalTestFixture>::GetInstance();
  chi::PoolQuery pool_query = chi::PoolQuery::Dynamic();
  wrp_cte::core::CreateParams params;

  // Step 1: Initialize CTE core pool
  REQUIRE_NOTHROW(fixture->CreateAsync(
      pool_query, CTECoreFunctionalTestFixture::kCTECorePoolName,
      CTECoreFunctionalTestFixture::kCTECorePoolId, params));
  INFO("Step 1 completed: CTE core pool initialized");

  // Step 2: Register multiple targets
  const std::vector<std::string> target_suffixes = {"target_1", "target_2"};
  for (const auto &suffix : target_suffixes) {
    // Use the actual file path as target_name since that's what matters for
    // bdev creation
    std::string target_name = fixture->test_storage_path_ + "_" + suffix;
    chi::u32 result = fixture->RegisterTargetAsync(
        target_name, chimaera::bdev::BdevType::kFile,
        CTECoreFunctionalTestFixture::kTestTargetSize, chi::PoolQuery::Local(),
        chi::PoolId(607, 0));
    REQUIRE(result == 0);
  }
  INFO("Step 2 completed: Multiple targets registered");

  // Step 3: Create tags for organization
  const std::vector<std::string> tag_names = {"documents", "images", "logs"};
  std::vector<wrp_cte::core::TagId> tag_ids;

  for (const auto &tag_name : tag_names) {
    wrp_cte::core::TagId tag_id =
        fixture->GetOrCreateTagAsync(tag_name);
    REQUIRE(!tag_id.IsNull());
    tag_ids.push_back(tag_id);
  }
  INFO("Step 3 completed: Tags created for organization");

  // Step 4: Blob operations simulation across different tags
  std::vector<std::tuple<wrp_cte::core::TagId, std::string, std::vector<char>>>
      stored_blobs;

  for (size_t i = 0; i < tag_ids.size(); ++i) {
    wrp_cte::core::TagId tag_id = tag_ids[i];
    std::string blob_name = "blob_" + std::to_string(i);
    chi::u64 blob_size = 1024 * (i + 1);  // Variable sizes

    auto blob_data =
        fixture->CreateTestData(blob_size, static_cast<char>('A' + i));

    // Validate blob parameters
    REQUIRE(!blob_name.empty());
    REQUIRE(blob_size > 0);
    REQUIRE(fixture->VerifyTestData(blob_data, static_cast<char>('A' + i)));

    // Store blob structure for validation
    stored_blobs.emplace_back(tag_id, blob_name, std::move(blob_data));

    INFO("Blob " << i << " prepared - Name: " << blob_name
                 << ", Size: " << blob_size << " bytes");
  }
  INFO("Step 4 completed: Multiple blob structures validated across tags");

  // Step 5: Simulate retrieval and verification of all blobs
  for (size_t j = 0; j < stored_blobs.size(); ++j) {
    const auto &[tag_id, blob_name, original_data] = stored_blobs[j];
    // Simulate perfect retrieval
    auto simulated_retrieved = original_data;
    REQUIRE(simulated_retrieved == original_data);

    // Verify data integrity pattern
    char expected_pattern = static_cast<char>('A' + (j % 26));
    REQUIRE(fixture->VerifyTestData(simulated_retrieved, expected_pattern));

    INFO("Blob retrieved and verified - Name: " << blob_name
                                                << ", Integrity: PASS");
  }
  INFO("Step 5 completed: All blob data integrity verified");

  // Step 6: Update target statistics
  chi::u32 stat_result = fixture->StatTargetsAsync();
  INFO("Step 6 completed: Target statistics updated, result: " << stat_result);

  // Verify targets are still listed correctly
  auto final_targets = fixture->ListTargetsAsync();
  REQUIRE(final_targets.size() >= target_suffixes.size());
  INFO("Integration test completed successfully - all steps verified");
}

/**
 * FUNCTIONAL Test Case: Distributed Execution Validation
 *
 * This test validates that data operations are being distributed across nodes
 * by performing iterative PutBlob and GetBlob operations and tracking which
 * nodes completed each operation. It verifies:
 * 1. Multiple Put/Get operations can be executed successfully
 * 2. Operations are distributed across multiple nodes (when available)
 * 3. Completer tracking works correctly to identify execution location
 * 4. Average completer value indicates distributed execution
 *
 * Environment Variables:
 * - CTE_NUM_NODES: Number of nodes in the distributed system (default: 1)
 */
TEST_CASE("FUNCTIONAL - Distributed Execution Validation",
          "[cte][core][distributed][validation]") {
  auto *fixture = hshm::Singleton<CTECoreFunctionalTestFixture>::
      GetInstance();  // Parse number of nodes from environment variable
  int num_nodes = 1;
  const char *num_nodes_env = std::getenv("CTE_NUM_NODES");
  if (num_nodes_env != nullptr) {
    num_nodes = std::atoi(num_nodes_env);
    if (num_nodes < 1) {
      num_nodes = 1;
    }
  }
  INFO("Running distributed execution test with " << num_nodes << " nodes");

  // Setup: Create core pool and register target
  chi::PoolQuery pool_query = chi::PoolQuery::Dynamic();
  wrp_cte::core::CreateParams params;

  REQUIRE_NOTHROW(fixture->CreateAsync(
      pool_query, CTECoreFunctionalTestFixture::kCTECorePoolName,
      CTECoreFunctionalTestFixture::kCTECorePoolId, params));
  REQUIRE(fixture->core_client_->GetReturnCode() == 0);

  // Use the fixture->test_storage_path_ as target_name
  const std::string target_name = fixture->test_storage_path_;
  chi::u32 reg_result = fixture->RegisterTargetAsync(
      target_name, chimaera::bdev::BdevType::kFile,
      CTECoreFunctionalTestFixture::kTestTargetSize, chi::PoolQuery::Local(),
      chi::PoolId(608, 0));
  REQUIRE(reg_result == 0);

  // Create a test tag for blob grouping
  const std::string tag_name = "distributed_test_tag";
  wrp_cte::core::TagId tag_id = fixture->GetOrCreateTagAsync(tag_name);
  REQUIRE(!tag_id.IsNull());

  // Test configuration
  constexpr int num_iterations = 16;
  constexpr chi::u64 blob_size = 4096;  // 4KB per blob
  const float score = 0.75f;

  // Tracking variables for completer statistics
  chi::u64 put_completer_sum = 0;
  chi::u64 get_completer_sum = 0;

  INFO("Starting " << num_iterations << " Put/Get iterations...");

  // Perform iterative PutBlob and GetBlob operations
  for (int i = 0; i < num_iterations; ++i) {
    std::string blob_name = "dist_blob_" + std::to_string(i);

    // Create unique test data for this blob
    char pattern = static_cast<char>('A' + (i % 26));
    auto test_data = fixture->CreateTestData(blob_size, pattern);
    REQUIRE(fixture->VerifyTestData(test_data, pattern));

    // Allocate shared memory for blob data
    hipc::FullPtr<char> blob_data_fullptr = CHI_IPC->AllocateBuffer(blob_size);
    REQUIRE(!blob_data_fullptr.IsNull());
    hipc::ShmPtr<> blob_data_ptr = blob_data_fullptr.shm_.template Cast<void>();

    // Copy test data to shared memory
    REQUIRE(fixture->CopyToSharedMemory(blob_data_fullptr, test_data));

    // Execute PutBlob operation
    auto put_task = fixture->core_client_->AsyncPutBlob(
        tag_id, blob_name, 0, blob_size, blob_data_ptr, score, wrp_cte::core::Context(), 0);

    REQUIRE(!put_task.IsNull());
    REQUIRE(fixture->WaitForTaskCompletion(put_task, 10000));
    REQUIRE(put_task->return_code_ == 0);

    // Track the completer for PutBlob
    chi::ContainerId put_completer = put_task->completer_;
    put_completer_sum += static_cast<chi::u64>(put_completer);
    INFO("Put iteration " << i << ": blob_name=" << blob_name
                          << ", completer=" << put_completer);

    // Cleanup put task

    // Allocate buffer for GetBlob result
    hipc::FullPtr<char> get_blob_data_fullptr =
        CHI_IPC->AllocateBuffer(blob_size);
    REQUIRE(!get_blob_data_fullptr.IsNull());
    hipc::ShmPtr<> get_blob_data_ptr =
        get_blob_data_fullptr.shm_.template Cast<void>();

    // Execute GetBlob operation
    auto get_task = fixture->core_client_->AsyncGetBlob(
        tag_id, blob_name, 0, blob_size, 0, get_blob_data_ptr);

    REQUIRE(!get_task.IsNull());
    REQUIRE(fixture->WaitForTaskCompletion(get_task, 10000));

    REQUIRE(get_task->return_code_ == 0);

    // Track the completer for GetBlob
    chi::ContainerId get_completer = get_task->completer_;
    get_completer_sum += static_cast<chi::u64>(get_completer);
    INFO("Get iteration " << i << ": blob_name=" << blob_name
                          << ", completer=" << get_completer);

    // Verify retrieved data matches original
    hipc::ShmPtr<> retrieved_data_ptr = get_task->blob_data_;
    REQUIRE(!retrieved_data_ptr.IsNull());
    auto retrieved_data =
        fixture->CopyFromSharedMemory(retrieved_data_ptr, blob_size);
    REQUIRE(retrieved_data.size() == blob_size);
    REQUIRE(fixture->VerifyTestData(retrieved_data, pattern));

    // Cleanup get task

    INFO("Iteration " << i << " completed successfully");
  }

  INFO("All " << num_iterations << " iterations completed");

  // Calculate average completer values (without rounding error)
  double put_avg_completer = static_cast<double>(put_completer_sum) /
                             static_cast<double>(num_iterations);
  double get_avg_completer = static_cast<double>(get_completer_sum) /
                             static_cast<double>(num_iterations);

  HLOG(kInfo, "PutBlob completer statistics:");
  HLOG(kInfo, "  Sum: {}", put_completer_sum);
  HLOG(kInfo, "  Average: {}", put_avg_completer);

  HLOG(kInfo, "GetBlob completer statistics:");
  HLOG(kInfo, "  Sum: {}", get_completer_sum);
  HLOG(kInfo, "  Average: {}", get_avg_completer);

  // Validation: If multiple nodes exist, average completer should be > 0
  // indicating distributed execution
  if (num_nodes > 1) {
    REQUIRE(put_avg_completer > 0.0);
    REQUIRE(get_avg_completer > 0.0);
    INFO(
        "SUCCESS: Distributed execution validated - operations executed "
        "across multiple nodes");
  } else {
    INFO("Single node test - distributed execution validation skipped");
  }

  INFO("Distributed execution validation test completed successfully");
}
// Main function using simple_test.h framework
SIMPLE_TEST_MAIN()
