/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Distributed under BSD 3-Clause license.                                   *
 * Copyright by The HDF Group.                                               *
 * Copyright by the Illinois Institute of Technology.                        *
 * All rights reserved.                                                      *
 *                                                                           *
 * This file is part of Hermes. The full Hermes copyright notice, including  *
 * terms governing use, modification, and redistribution, is contained in    *
 * the COPYING file, which can be found at the top directory. If you do not  *
 * have access to the file, you may request a copy from help@hdfgroup.org.   *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**
 * TAG OPERATION TESTS
 *
 * Tests for Tag API to improve coverage of tag.cc from 12.5% to 80%+
 * Focuses on exercising all public methods of the Tag class.
 */

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>

#include "simple_test.h"

#include <chimaera/admin/admin_tasks.h>
#include <chimaera/bdev/bdev_client.h>
#include <chimaera/bdev/bdev_tasks.h>
#include <chimaera/chimaera.h>
#include <wrp_cte/core/core_client.h>
#include <wrp_cte/core/core_runtime.h>
#include <wrp_cte/core/core_tasks.h>

namespace fs = std::filesystem;

/**
 * Test fixture for Tag operation tests
 */
class TagTestFixture {
 public:
  static constexpr chi::u64 kTestTargetSize = 1024 * 1024 * 10;  // 10MB
  static constexpr size_t kSmallDataSize = 1024;                  // 1KB
  static constexpr size_t kMediumDataSize = 16 * 1024;           // 16KB
  static constexpr size_t kLargeDataSize = 1024 * 1024;          // 1MB

  std::string test_storage_path_;
  bool initialized_ = false;
  static inline bool g_cte_initialized = false;

  TagTestFixture() {
    INFO("=== Initializing Tag Test Environment ===");

    // Initialize test storage path
    std::string home_dir = hshm::SystemInfo::Getenv("HOME");
    REQUIRE(!home_dir.empty());
    test_storage_path_ = home_dir + "/cte_tag_test.dat";

    // Clean up existing test file
    if (fs::exists(test_storage_path_)) {
      fs::remove(test_storage_path_);
    }

    // Initialize Chimaera and CTE client once
    if (!g_cte_initialized) {
      bool success = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
      REQUIRE(success);

      // Give runtime time to initialize
      std::this_thread::sleep_for(std::chrono::milliseconds(500));

      // Initialize CTE client subsystem (required for Tag API)
      success = wrp_cte::core::WRP_CTE_CLIENT_INIT();
      REQUIRE(success);

      // Get the global CTE client and initialize it with kCtePoolId
      auto *cte_client = WRP_CTE_CLIENT;
      REQUIRE(cte_client != nullptr);

      // Initialize the client's pool_id (required for Tag operations)
      INFO("Before Init: client pool_id=" << cte_client->pool_id_.ToU64());
      cte_client->Init(wrp_cte::core::kCtePoolId);
      INFO("After Init: client pool_id=" << cte_client->pool_id_.ToU64()
           << " (expected: " << wrp_cte::core::kCtePoolId.ToU64() << ")");

      // Create the CTE core pool explicitly
      wrp_cte::core::CreateParams params;
      auto create_task = cte_client->AsyncCreate(
          chi::PoolQuery::Dynamic(),
          wrp_cte::core::kCtePoolName,
          wrp_cte::core::kCtePoolId,
          params);
      create_task.Wait();
      REQUIRE(create_task->GetReturnCode() == 0);

      INFO("After AsyncCreate: client pool_id=" << cte_client->pool_id_.ToU64());
      INFO("CTE core pool created successfully");

      g_cte_initialized = true;
    }

    INFO("=== Tag Test Environment Ready ===");
  }

  ~TagTestFixture() {
    INFO("=== Cleaning up Tag Test Environment ===");
    if (fs::exists(test_storage_path_)) {
      fs::remove(test_storage_path_);
    }
  }

  /**
   * Setup CTE pool and register target
   */
  void SetupCTEWithTarget() {
    if (initialized_) {
      return;
    }

    auto *cte_client = WRP_CTE_CLIENT;
    REQUIRE(cte_client != nullptr);

    // Create test storage target using bdev client
    chi::PoolId bdev_pool_id(800, 0);
    chimaera::bdev::Client bdev_client(bdev_pool_id);
    auto create_task = bdev_client.AsyncCreate(
        chi::PoolQuery::Dynamic(), test_storage_path_,
        bdev_pool_id, chimaera::bdev::BdevType::kFile);
    create_task.Wait();

    // Wait for storage target creation
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Register the storage target with CTE
    auto reg_task = cte_client->AsyncRegisterTarget(
        test_storage_path_, chimaera::bdev::BdevType::kFile,
        kTestTargetSize, chi::PoolQuery::Local(), bdev_pool_id);
    reg_task.Wait();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    initialized_ = true;
    INFO("=== CTE Pool and Target Registered ===");
  }

  /**
   * Create test data with a pattern
   */
  std::vector<char> CreateTestData(size_t size, char pattern = 'T') {
    std::vector<char> data(size);
    for (size_t i = 0; i < size; ++i) {
      data[i] = static_cast<char>(pattern + (i % 26));
    }
    return data;
  }

  /**
   * Verify test data integrity
   */
  bool VerifyTestData(const std::vector<char> &data, char pattern = 'T') {
    for (size_t i = 0; i < data.size(); ++i) {
      char expected = static_cast<char>(pattern + (i % 26));
      if (data[i] != expected) {
        INFO("Data mismatch at index " << i);
        return false;
      }
    }
    return true;
  }
};

// ============================================================================
// Tag Construction Tests
// ============================================================================

TEST_CASE("Tag - Construction with Name", "[cte][tag][construction]") {
  TagTestFixture fixture;
  fixture.SetupCTEWithTarget();

  SECTION("Create tag with simple name") {
    wrp_cte::core::Tag tag("test_tag");
    INFO("Tag created successfully with name: test_tag");
  }

  SECTION("Create tag with empty name") {
    wrp_cte::core::Tag tag("");
    INFO("Tag created with empty name (allowed)");
  }

  SECTION("Create tag with long name") {
    std::string long_name(1024, 'X');
    wrp_cte::core::Tag tag(long_name);
    INFO("Tag created with long name (1024 chars)");
  }
}

TEST_CASE("Tag - Construction with TagId", "[cte][tag][construction]") {
  TagTestFixture fixture;
  fixture.SetupCTEWithTarget();

  wrp_cte::core::TagId tag_id(42, 0);
  wrp_cte::core::Tag tag(tag_id);
  INFO("Tag created with direct TagId");
}

TEST_CASE("Tag - Multiple Tags Same Name", "[cte][tag][construction]") {
  TagTestFixture fixture;
  fixture.SetupCTEWithTarget();

  wrp_cte::core::Tag tag1("shared_tag");
  wrp_cte::core::Tag tag2("shared_tag");
  INFO("Multiple tags with same name created");
}

// ============================================================================
// PutBlob Tests
// ============================================================================

TEST_CASE("Tag - PutBlob Basic Operation", "[cte][tag][putblob]") {
  TagTestFixture fixture;
  fixture.SetupCTEWithTarget();

  wrp_cte::core::Tag tag("putblob_basic");
  auto data = fixture.CreateTestData(fixture.kSmallDataSize, 'B');

  tag.PutBlob("test_blob", data.data(), data.size());

  // Verify blob was stored by checking size
  chi::u64 size = tag.GetBlobSize("test_blob");
  REQUIRE(size == fixture.kSmallDataSize);
}

TEST_CASE("Tag - PutBlob with Offset", "[cte][tag][putblob]") {
  TagTestFixture fixture;
  fixture.SetupCTEWithTarget();

  wrp_cte::core::Tag tag("putblob_offset");
  auto data = fixture.CreateTestData(fixture.kSmallDataSize, 'O');

  // Write data at offset 1024
  tag.PutBlob("offset_blob", data.data(), data.size(), 1024);

  INFO("PutBlob with offset completed");
}

TEST_CASE("Tag - PutBlob with Custom Score", "[cte][tag][putblob]") {
  TagTestFixture fixture;
  fixture.SetupCTEWithTarget();

  wrp_cte::core::Tag tag("putblob_score");
  auto data = fixture.CreateTestData(fixture.kSmallDataSize, 'S');

  // Put blob with custom score
  tag.PutBlob("scored_blob", data.data(), data.size(), 0, 5.0f);

  // Verify score was set
  float score = tag.GetBlobScore("scored_blob");
  REQUIRE(score == 5.0f);
}

TEST_CASE("Tag - PutBlob with Context", "[cte][tag][putblob]") {
  TagTestFixture fixture;
  fixture.SetupCTEWithTarget();

  wrp_cte::core::Tag tag("putblob_context");
  auto data = fixture.CreateTestData(fixture.kSmallDataSize, 'C');

  // Create compression context
  wrp_cte::core::Context ctx;
  ctx.dynamic_compress_ = 1;
  ctx.compress_lib_ = 2;
  ctx.max_performance_ = true;

  tag.PutBlob("context_blob", data.data(), data.size(), 0, 1.0f, ctx);

  INFO("PutBlob with context completed");
}

TEST_CASE("Tag - PutBlob SHM Version", "[cte][tag][putblob]") {
  TagTestFixture fixture;
  fixture.SetupCTEWithTarget();

  wrp_cte::core::Tag tag("putblob_shm");
  auto data = fixture.CreateTestData(fixture.kSmallDataSize, 'M');

  // Allocate shared memory
  auto *ipc = CHI_IPC;
  hipc::FullPtr<char> shm_fullptr = ipc->AllocateBuffer(data.size());
  REQUIRE(!shm_fullptr.IsNull());

  // Copy data to shared memory
  memcpy(shm_fullptr.ptr_, data.data(), data.size());

  // Convert to ShmPtr and call SHM version
  hipc::ShmPtr<> shm_ptr(shm_fullptr.shm_);
  tag.PutBlob("shm_blob", shm_ptr, data.size());

  // Free shared memory
  ipc->FreeBuffer(shm_fullptr);

  INFO("PutBlob SHM version completed");
}

TEST_CASE("Tag - PutBlob Multiple Blobs", "[cte][tag][putblob]") {
  TagTestFixture fixture;
  fixture.SetupCTEWithTarget();

  wrp_cte::core::Tag tag("putblob_multi");

  // Put multiple blobs
  for (int i = 0; i < 5; ++i) {
    auto data = fixture.CreateTestData(fixture.kSmallDataSize, 'A' + i);
    std::string blob_name = "blob_" + std::to_string(i);
    tag.PutBlob(blob_name, data.data(), data.size());
  }

  // Verify all blobs exist
  auto blobs = tag.GetContainedBlobs();
  REQUIRE(blobs.size() >= 5);
}

// ============================================================================
// GetBlob Tests
// ============================================================================

TEST_CASE("Tag - GetBlob Basic Operation", "[cte][tag][getblob]") {
  TagTestFixture fixture;
  fixture.SetupCTEWithTarget();

  wrp_cte::core::Tag tag("getblob_basic");
  auto original_data = fixture.CreateTestData(fixture.kSmallDataSize, 'G');

  // Put data
  tag.PutBlob("get_test_blob", original_data.data(), original_data.size());

  // Get data back
  std::vector<char> retrieved_data(fixture.kSmallDataSize);
  tag.GetBlob("get_test_blob", retrieved_data.data(), retrieved_data.size());

  // Verify data integrity
  REQUIRE(fixture.VerifyTestData(retrieved_data, 'G'));
}

TEST_CASE("Tag - GetBlob with Offset", "[cte][tag][getblob]") {
  TagTestFixture fixture;
  fixture.SetupCTEWithTarget();

  wrp_cte::core::Tag tag("getblob_offset");
  auto original_data = fixture.CreateTestData(fixture.kMediumDataSize, 'H');

  // Put full data
  tag.PutBlob("offset_get_blob", original_data.data(), original_data.size());

  // Get only partial data at offset
  std::vector<char> retrieved_data(fixture.kSmallDataSize);
  tag.GetBlob("offset_get_blob", retrieved_data.data(), retrieved_data.size(), 1024);

  INFO("GetBlob with offset completed");
}

TEST_CASE("Tag - GetBlob SHM Version", "[cte][tag][getblob]") {
  TagTestFixture fixture;
  fixture.SetupCTEWithTarget();

  wrp_cte::core::Tag tag("getblob_shm");
  auto original_data = fixture.CreateTestData(fixture.kSmallDataSize, 'I');

  // Put data
  tag.PutBlob("shm_get_blob", original_data.data(), original_data.size());

  // Allocate shared memory for retrieval
  auto *ipc = CHI_IPC;
  hipc::FullPtr<char> shm_fullptr = ipc->AllocateBuffer(original_data.size());
  REQUIRE(!shm_fullptr.IsNull());

  // Get blob using SHM version
  hipc::ShmPtr<> shm_ptr(shm_fullptr.shm_);
  tag.GetBlob("shm_get_blob", shm_ptr, original_data.size());

  // Free shared memory
  ipc->FreeBuffer(shm_fullptr);

  INFO("GetBlob SHM version completed");
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST_CASE("Tag - GetBlob Null Buffer", "[cte][tag][errors]") {
  TagTestFixture fixture;
  fixture.SetupCTEWithTarget();

  wrp_cte::core::Tag tag("error_null_buffer");
  auto data = fixture.CreateTestData(fixture.kSmallDataSize, 'E');
  tag.PutBlob("error_blob", data.data(), data.size());

  // Try to get blob with null buffer
  bool caught_exception = false;
  try {
    tag.GetBlob("error_blob", nullptr, fixture.kSmallDataSize);
  } catch (const std::invalid_argument &e) {
    caught_exception = true;
    INFO("Caught expected exception: " << e.what());
  }
  REQUIRE(caught_exception);
}

TEST_CASE("Tag - GetBlob Zero Size", "[cte][tag][errors]") {
  TagTestFixture fixture;
  fixture.SetupCTEWithTarget();

  wrp_cte::core::Tag tag("error_zero_size");
  std::vector<char> buffer(100);

  // Try to get blob with zero size
  bool caught_exception = false;
  try {
    tag.GetBlob("any_blob", buffer.data(), 0);
  } catch (const std::invalid_argument &e) {
    caught_exception = true;
    INFO("Caught expected exception: " << e.what());
  }
  REQUIRE(caught_exception);
}

TEST_CASE("Tag - GetBlob SHM Null Pointer", "[cte][tag][errors]") {
  TagTestFixture fixture;
  fixture.SetupCTEWithTarget();

  wrp_cte::core::Tag tag("error_shm_null");

  // Try to get blob with null ShmPtr
  hipc::ShmPtr<> null_ptr;
  bool caught_exception = false;
  try {
    tag.GetBlob("any_blob", null_ptr, fixture.kSmallDataSize);
  } catch (const std::invalid_argument &e) {
    caught_exception = true;
    INFO("Caught expected exception: " << e.what());
  }
  REQUIRE(caught_exception);
}

// ============================================================================
// Metadata Tests
// ============================================================================

TEST_CASE("Tag - GetBlobSize", "[cte][tag][metadata]") {
  TagTestFixture fixture;
  fixture.SetupCTEWithTarget();

  wrp_cte::core::Tag tag("metadata_size");
  auto data = fixture.CreateTestData(fixture.kMediumDataSize, 'Z');
  tag.PutBlob("size_blob", data.data(), data.size());

  chi::u64 size = tag.GetBlobSize("size_blob");
  REQUIRE(size == fixture.kMediumDataSize);
}

TEST_CASE("Tag - GetBlobScore", "[cte][tag][metadata]") {
  TagTestFixture fixture;
  fixture.SetupCTEWithTarget();

  wrp_cte::core::Tag tag("metadata_score");
  auto data = fixture.CreateTestData(fixture.kSmallDataSize, 'Y');
  float expected_score = 7.5f;
  tag.PutBlob("score_blob", data.data(), data.size(), 0, expected_score);

  float actual_score = tag.GetBlobScore("score_blob");
  REQUIRE(actual_score == expected_score);
}

TEST_CASE("Tag - GetContainedBlobs", "[cte][tag][metadata]") {
  TagTestFixture fixture;
  fixture.SetupCTEWithTarget();

  wrp_cte::core::Tag tag("metadata_contained");

  // Put multiple blobs with known names
  std::vector<std::string> blob_names = {"blob1", "blob2", "blob3"};
  for (const auto &name : blob_names) {
    auto data = fixture.CreateTestData(fixture.kSmallDataSize, 'X');
    tag.PutBlob(name, data.data(), data.size());
  }

  // Get contained blobs
  auto contained = tag.GetContainedBlobs();
  REQUIRE(contained.size() >= 3);

  INFO("GetContainedBlobs returned " << contained.size() << " blobs");
}

// ============================================================================
// ReorganizeBlob Tests
// ============================================================================

TEST_CASE("Tag - ReorganizeBlob Basic", "[cte][tag][reorganize]") {
  TagTestFixture fixture;
  fixture.SetupCTEWithTarget();

  wrp_cte::core::Tag tag("reorg_basic");
  auto data = fixture.CreateTestData(fixture.kSmallDataSize, 'R');

  // Put blob with initial score
  tag.PutBlob("reorg_blob", data.data(), data.size(), 0, 2.0f);

  // Reorganize with new score
  tag.ReorganizeBlob("reorg_blob", 8.0f);

  // Verify new score
  float new_score = tag.GetBlobScore("reorg_blob");
  REQUIRE(new_score == 8.0f);
}

TEST_CASE("Tag - ReorganizeBlob Multiple Times", "[cte][tag][reorganize]") {
  TagTestFixture fixture;
  fixture.SetupCTEWithTarget();

  wrp_cte::core::Tag tag("reorg_multi");
  auto data = fixture.CreateTestData(fixture.kSmallDataSize, 'Q');

  tag.PutBlob("multi_reorg", data.data(), data.size(), 0, 1.0f);

  // Reorganize multiple times
  for (float score = 2.0f; score <= 10.0f; score += 1.0f) {
    tag.ReorganizeBlob("multi_reorg", score);
  }

  // Verify final score
  float final_score = tag.GetBlobScore("multi_reorg");
  REQUIRE(final_score == 10.0f);
}

// ============================================================================
// AsyncPutBlob Tests
// ============================================================================

TEST_CASE("Tag - AsyncPutBlob", "[cte][tag][async]") {
  TagTestFixture fixture;
  fixture.SetupCTEWithTarget();

  wrp_cte::core::Tag tag("async_put");
  auto data = fixture.CreateTestData(fixture.kSmallDataSize, 'A');

  // Allocate shared memory (must remain alive until task completes)
  auto *ipc = CHI_IPC;
  hipc::FullPtr<char> shm_fullptr = ipc->AllocateBuffer(data.size());
  REQUIRE(!shm_fullptr.IsNull());
  memcpy(shm_fullptr.ptr_, data.data(), data.size());

  // Call async version
  hipc::ShmPtr<> shm_ptr(shm_fullptr.shm_);
  auto future = tag.AsyncPutBlob("async_blob", shm_ptr, data.size());

  // Wait for completion
  future.Wait();

  // Free shared memory
  ipc->FreeBuffer(shm_fullptr);

  // Verify blob was stored
  chi::u64 size = tag.GetBlobSize("async_blob");
  REQUIRE(size == fixture.kSmallDataSize);
}

// ============================================================================
// Large Data Tests
// ============================================================================

TEST_CASE("Tag - Large Data PutGet", "[cte][tag][large]") {
  TagTestFixture fixture;
  fixture.SetupCTEWithTarget();

  wrp_cte::core::Tag tag("large_data");
  auto large_data = fixture.CreateTestData(fixture.kLargeDataSize, 'L');

  // Put large blob
  tag.PutBlob("large_blob", large_data.data(), large_data.size());

  // Get it back
  std::vector<char> retrieved_data(fixture.kLargeDataSize);
  tag.GetBlob("large_blob", retrieved_data.data(), retrieved_data.size());

  // Verify integrity
  REQUIRE(fixture.VerifyTestData(retrieved_data, 'L'));

  INFO("Large data (1MB) put/get completed successfully");
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_CASE("Tag - Overwrite Blob", "[cte][tag][edge]") {
  TagTestFixture fixture;
  fixture.SetupCTEWithTarget();

  wrp_cte::core::Tag tag("overwrite");
  auto data1 = fixture.CreateTestData(fixture.kSmallDataSize, 'F');
  auto data2 = fixture.CreateTestData(fixture.kSmallDataSize, 'S');

  // Put initial data
  tag.PutBlob("overwrite_blob", data1.data(), data1.size());

  // Overwrite with new data
  tag.PutBlob("overwrite_blob", data2.data(), data2.size());

  // Verify second data is stored
  std::vector<char> retrieved(fixture.kSmallDataSize);
  tag.GetBlob("overwrite_blob", retrieved.data(), retrieved.size());
  REQUIRE(fixture.VerifyTestData(retrieved, 'S'));
}

SIMPLE_TEST_MAIN()
