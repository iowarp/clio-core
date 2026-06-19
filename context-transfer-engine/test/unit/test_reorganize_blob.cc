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
 * REORGANIZE BLOB TEST
 *
 * Tests score-based data movement between tiers:
 * - 16MB DRAM (fast tier, score 1.0)
 * - 64MB File (slow tier, score 0.2)
 *
 * Test cases:
 * 1. PutBlob(score=1.0) should place data in DRAM
 * 2. ReorganizeBlob(score=0.2) should move data to disk
 * 3. GetBlobInfo() verifies score and block placement changes
 */

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#include "simple_test.h"

namespace fs = std::filesystem;

static std::string chi_test_data_dir() {
  const char *d = chi::env::GetCompat("TEST_DATA_DIR");
  return (d && *d) ? d : ".";
}

// Test constants for two-tier storage
static constexpr chi::u64 kDramCapacity = 16 * 1024 * 1024;  // 16MB
static constexpr chi::u64 kDiskCapacity = 64 * 1024 * 1024;  // 64MB
static constexpr chi::u64 kBlobSize = 1 * 1024 * 1024;       // 1MB per blob

// Tier scores (higher score = faster tier in this config)
static constexpr float kDramScore = 1.0f;    // Fast tier - DRAM
static constexpr float kDiskScore = 0.2f;    // Slow tier - Disk

/**
 * Test fixture for reorganize blob tests
 */
class ReorganizeBlobTestFixture {
 public:
  std::string config_path_;
  std::string file_storage_path_;
  bool initialized_ = false;

  ReorganizeBlobTestFixture() {
    INFO("=== Initializing ReorganizeBlob Test ===");

    // Setup paths
    std::string home_dir = ctp::SystemInfo::GetHomeDir();
    REQUIRE(!home_dir.empty());
    config_path_ = chi_test_data_dir() + "/reorganize_blob_config.yaml";
    file_storage_path_ = chi_test_data_dir() + "/reorganize_blob_storage.bin";

    // Clean up existing files
    Cleanup();

    // Create config file
    CreateConfigFile();

    // Set environment variable for runtime config
    // CHI_SERVER_CONF is checked first, so set it to override any existing value
    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", config_path_.c_str(), 1);
    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", config_path_.c_str(), 1);

    // Initialize CLIO Runtime runtime
    bool success = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
    REQUIRE(success);

    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // Initialize CTE client
    success = clio::cte::core::CLIO_CTE_CLIENT_INIT();
    REQUIRE(success);

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    initialized_ = true;
    INFO("=== ReorganizeBlob Test Environment Ready ===");
  }

  ~ReorganizeBlobTestFixture() {
    INFO("=== Cleaning up ReorganizeBlob Test ===");
    Cleanup();
  }

  void Cleanup() {
    if (fs::exists(config_path_)) {
      fs::remove(config_path_);
    }
    if (fs::exists(file_storage_path_)) {
      fs::remove(file_storage_path_);
    }
  }

  /**
   * Create configuration file with 16MB DRAM and 64MB file storage
   * DRAM: score=1.0 (fast tier)
   * DISK: score=0.2 (slow tier)
   */
  void CreateConfigFile() {
    std::ofstream config_file(config_path_);
    REQUIRE(config_file.is_open());

    config_file << R"(
# ReorganizeBlob Test Configuration
# - 16MB DRAM (fast tier, score 1.0)
# - 64MB File (slow tier, score 0.2)

runtime:
  num_threads: 2
  queue_depth: 1024
  first_busy_wait: 10000
  max_sleep: 50000

compose:
  - mod_name: clio_cte_core
    pool_name: clio_cte
    pool_query: local
    pool_id: 512.0

    targets:
      neighborhood: 1
      default_target_timeout_ms: 30000
      poll_period_ms: 5000

    storage:
      # Fast tier: 16MB DRAM (score 1.0)
      - path: "ram::reorganize_dram"
        bdev_type: "ram"
        capacity_limit: "16MB"
        score: 1.0

      # Slow tier: 64MB File (score 0.2)
      - path: ")" << file_storage_path_ << R"("
        bdev_type: "file"
        capacity_limit: "64MB"
        score: 0.2

    dpe:
      dpe_type: "max_bw"
)";

    config_file.close();
    INFO("Created config file: " << config_path_);
    INFO("  DRAM: 16MB @ score 1.0");
    INFO("  Disk: 64MB @ score 0.2");
  }

  /**
   * Create test data pattern
   */
  std::vector<char> CreateTestData(size_t size, char pattern = 'R') {
    std::vector<char> data(size);
    std::memset(data.data(), pattern, size);
    return data;
  }

  /**
   * Verify test data pattern
   */
  bool VerifyTestData(const std::vector<char>& data, char expected_pattern = 'R') {
    for (size_t i = 0; i < data.size(); ++i) {
      if (data[i] != expected_pattern) {
        return false;
      }
    }
    return true;
  }
};

// Global fixture instance
static ReorganizeBlobTestFixture* g_fixture = nullptr;

/**
 * Test: Put blob with score=1.0 (should go to DRAM)
 */
TEST_CASE("ReorganizeBlob - PutBlob to DRAM", "[reorganize][put][dram]") {
  REQUIRE(g_fixture != nullptr);
  REQUIRE(g_fixture->initialized_);

  auto* cte_client = CLIO_CTE_CLIENT;
  REQUIRE(cte_client != nullptr);

  // Create a tag for our test blobs
  std::string tag_name = "reorganize_test_tag";
  clio::cte::core::Tag tag(tag_name);
  clio::cte::core::TagId tag_id = tag.GetTagId();

  INFO("Putting blob with score=1.0 (should go to DRAM)");

  // Allocate shared memory buffer
  auto shm_buffer = CLIO_IPC->AllocateBuffer(kBlobSize);
  REQUIRE(!shm_buffer.IsNull());
  ctp::ipc::ShmPtr<> shm_ptr = shm_buffer.shm_.template Cast<void>();

  // Fill buffer with pattern
  auto test_data = g_fixture->CreateTestData(kBlobSize, 'D');  // 'D' for DRAM
  std::memcpy(shm_buffer.ptr_, test_data.data(), kBlobSize);

  // Put blob with high score (1.0) - should go to DRAM
  std::string blob_name = "test_blob_dram";
  auto put_task = tag.AsyncPutBlob(blob_name, shm_ptr, kBlobSize, 0, kDramScore);
  put_task.Wait();

  REQUIRE(put_task->GetReturnCode() == 0);
  INFO("PutBlob succeeded with score=" << kDramScore);

  // Get blob score to verify placement
  INFO("Calling AsyncGetBlobScore...");
  auto score_task = cte_client->AsyncGetBlobScore(tag_id, blob_name);
  score_task.Wait();

  if (score_task->GetReturnCode() != 0) {
    INFO("GetBlobScore failed with return code: " << score_task->GetReturnCode());
    REQUIRE(score_task->GetReturnCode() == 0);
  }

  float blob_score = score_task->score_;
  INFO("GetBlobScore returned: " << blob_score);

  // Verify score is close to 1.0
  REQUIRE(std::abs(blob_score - kDramScore) < 0.01f);

  // Get blob size
  auto size_task = cte_client->AsyncGetBlobSize(tag_id, blob_name);
  size_task.Wait();
  REQUIRE(size_task->GetReturnCode() == 0);
  REQUIRE(size_task->size_ == kBlobSize);
  INFO("Blob size: " << size_task->size_);

  CLIO_IPC->FreeBuffer(shm_buffer);
  INFO("SUCCESS: Blob placed with score=1.0");
}

/**
 * Test: ReorganizeBlob to score=0.2 (should move to disk)
 */
TEST_CASE("ReorganizeBlob - Move to Disk", "[reorganize][move][disk][noleak]") {
  REQUIRE(g_fixture != nullptr);
  REQUIRE(g_fixture->initialized_);

  auto* cte_client = CLIO_CTE_CLIENT;
  REQUIRE(cte_client != nullptr);

  std::string tag_name = "reorganize_test_tag";
  clio::cte::core::Tag tag(tag_name);
  clio::cte::core::TagId tag_id = tag.GetTagId();
  std::string blob_name = "test_blob_dram";

  // Get blob score before reorganization
  auto score_before_task = cte_client->AsyncGetBlobScore(tag_id, blob_name);
  score_before_task.Wait();
  REQUIRE(score_before_task->GetReturnCode() == 0);

  float score_before = score_before_task->score_;
  INFO("Before ReorganizeBlob:");
  INFO("  Score: " << score_before);

  INFO("Calling ReorganizeBlob with score=" << kDiskScore);

  // Reorganize blob to score 0.2 (should move to disk)
  auto reorg_task = cte_client->AsyncReorganizeBlob(tag_id, blob_name, kDiskScore);
  reorg_task.Wait();

  REQUIRE(reorg_task->GetReturnCode() == 0);
  INFO("ReorganizeBlob completed successfully");

  // Get blob score after reorganization
  auto score_after_task = cte_client->AsyncGetBlobScore(tag_id, blob_name);
  score_after_task.Wait();
  REQUIRE(score_after_task->GetReturnCode() == 0);

  float score_after = score_after_task->score_;
  INFO("After ReorganizeBlob:");
  INFO("  Score: " << score_after);

  // Verify score changed to the new value
  REQUIRE(std::abs(score_after - kDiskScore) < 0.01f);
  INFO("SUCCESS: Score changed from " << score_before << " to " << score_after);
}

/**
 * Test: Verify data integrity after reorganization
 */
TEST_CASE("ReorganizeBlob - Verify Data Integrity", "[reorganize][integrity]") {
  REQUIRE(g_fixture != nullptr);
  REQUIRE(g_fixture->initialized_);

  std::string tag_name = "reorganize_test_tag";
  clio::cte::core::Tag tag(tag_name);
  std::string blob_name = "test_blob_dram";

  INFO("Verifying data integrity after reorganization");

  // Allocate buffer for reading
  auto read_buffer = CLIO_IPC->AllocateBuffer(kBlobSize);
  REQUIRE(!read_buffer.IsNull());

  // Read blob data
  tag.GetBlob(blob_name, read_buffer.shm_.template Cast<void>(), kBlobSize, 0);

  // Verify data pattern
  std::vector<char> read_data(kBlobSize);
  std::memcpy(read_data.data(), read_buffer.ptr_, kBlobSize);

  bool data_valid = g_fixture->VerifyTestData(read_data, 'D');  // 'D' pattern from put
  REQUIRE(data_valid);

  CLIO_IPC->FreeBuffer(read_buffer);
  INFO("SUCCESS: Data integrity verified after reorganization");
}

/**
 * Test: ReorganizeBlob back to DRAM (promote data)
 */
TEST_CASE("ReorganizeBlob - Promote to DRAM", "[reorganize][promote][dram][noleak]") {
  REQUIRE(g_fixture != nullptr);
  REQUIRE(g_fixture->initialized_);

  auto* cte_client = CLIO_CTE_CLIENT;
  REQUIRE(cte_client != nullptr);

  std::string tag_name = "reorganize_test_tag";
  clio::cte::core::Tag tag(tag_name);
  clio::cte::core::TagId tag_id = tag.GetTagId();
  std::string blob_name = "test_blob_dram";

  // Get blob score before promotion
  auto score_before_task = cte_client->AsyncGetBlobScore(tag_id, blob_name);
  score_before_task.Wait();
  REQUIRE(score_before_task->GetReturnCode() == 0);

  float score_before = score_before_task->score_;
  INFO("Before promotion - Score: " << score_before);

  INFO("Promoting blob back to DRAM with score=" << kDramScore);

  // Reorganize blob back to score 1.0 (promote to DRAM)
  auto reorg_task = cte_client->AsyncReorganizeBlob(tag_id, blob_name, kDramScore);
  reorg_task.Wait();

  REQUIRE(reorg_task->GetReturnCode() == 0);

  // Get blob score after promotion
  auto score_after_task = cte_client->AsyncGetBlobScore(tag_id, blob_name);
  score_after_task.Wait();
  REQUIRE(score_after_task->GetReturnCode() == 0);

  float score_after = score_after_task->score_;
  INFO("After promotion - Score: " << score_after);

  // Verify score changed back to DRAM score
  REQUIRE(std::abs(score_after - kDramScore) < 0.01f);
  INFO("SUCCESS: Score changed from " << score_before << " to " << score_after);

  // Verify data integrity after promotion
  auto read_buffer = CLIO_IPC->AllocateBuffer(kBlobSize);
  REQUIRE(!read_buffer.IsNull());

  tag.GetBlob(blob_name, read_buffer.shm_.template Cast<void>(), kBlobSize, 0);

  std::vector<char> read_data(kBlobSize);
  std::memcpy(read_data.data(), read_buffer.ptr_, kBlobSize);

  bool data_valid = g_fixture->VerifyTestData(read_data, 'D');
  REQUIRE(data_valid);

  CLIO_IPC->FreeBuffer(read_buffer);
  INFO("SUCCESS: Blob promoted back to DRAM with data integrity");
}

/**
 * Test: Cleanup all blobs and tags
 */
TEST_CASE("ReorganizeBlob - Cleanup", "[reorganize][cleanup]") {
  REQUIRE(g_fixture != nullptr);
  REQUIRE(g_fixture->initialized_);

  auto* cte_client = CLIO_CTE_CLIENT;
  REQUIRE(cte_client != nullptr);

  std::string tag_name = "reorganize_test_tag";
  clio::cte::core::Tag tag(tag_name);
  clio::cte::core::TagId tag_id = tag.GetTagId();

  INFO("Cleaning up test blobs and tags");

  // Delete the blob
  auto del_blob_task = cte_client->AsyncDelBlob(tag_id, "test_blob_dram");
  del_blob_task.Wait();

  // Delete the tag
  auto del_tag_task = cte_client->AsyncDelTag(tag_name);
  del_tag_task.Wait();

  INFO("Cleanup complete");
}

int main(int argc, char** argv) {
  // Create fixture (initializes runtime)
  g_fixture = new ReorganizeBlobTestFixture();

  // Run tests with optional filter from command line
  std::string filter = (argc > 1) ? argv[1] : "";
  int result = SimpleTest::run_all_tests(filter);

  // Cleanup
  delete g_fixture;
  g_fixture = nullptr;

  SIMPLE_TEST_PROCESS_EXIT(result);
  return result;  // unreachable on Windows
}
