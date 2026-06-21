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
 * TIERED STORAGE STRESS TEST
 *
 * Tests automatic tiering and reorganization with constrained storage:
 * - 64MB DRAM (fast tier, score 0.0)
 * - 256MB File (slow tier, score 1.0)
 * - Put 128MB of data (exceeds DRAM, must overflow to file)
 * - ReorganizeBlob all data to score 0.0 (tests capacity handling)
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
  const char *d = clio::run::env::GetCompat("TEST_DATA_DIR");
  return (d && *d) ? d : ".";
}

// Test constants
static constexpr clio::run::u64 kDramCapacity = 64 * 1024 * 1024;   // 64MB
static constexpr clio::run::u64 kFileCapacity = 256 * 1024 * 1024;  // 256MB
static constexpr clio::run::u64 kTotalDataSize = 128 * 1024 * 1024; // 128MB
static constexpr clio::run::u64 kBlobSize = 1 * 1024 * 1024;        // 1MB per blob
static constexpr int kNumBlobs = kTotalDataSize / kBlobSize;  // 128 blobs

static constexpr float kFastTierScore = 0.0f;  // DRAM
static constexpr float kSlowTierScore = 1.0f;  // File

/**
 * Test fixture for tiered storage stress tests
 */
class TieredStorageStressFixture {
 public:
  std::string config_path_;
  std::string file_storage_path_;
  bool initialized_ = false;

  TieredStorageStressFixture() {
    INFO("=== Initializing Tiered Storage Stress Test ===");

    // Setup paths
    std::string home_dir = ctp::SystemInfo::GetHomeDir();
    REQUIRE(!home_dir.empty());
    config_path_ = chi_test_data_dir() + "/tiered_stress_config.yaml";
    file_storage_path_ = chi_test_data_dir() + "/tiered_stress_storage.bin";

    // Clean up existing files
    Cleanup();

    // Create config file
    CreateConfigFile();

    // Set environment variable for runtime config
    // CLIO_SERVER_CONF is checked first, so set it to override any existing value
    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", config_path_.c_str(), 1);
    ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", config_path_.c_str(), 1);

    // Initialize CLIO Runtime runtime
    bool success = clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
    REQUIRE(success);

    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // Initialize CTE client
    success = clio::cte::core::CLIO_CTE_CLIENT_INIT();
    REQUIRE(success);

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    initialized_ = true;
    INFO("=== Tiered Storage Stress Test Environment Ready ===");
  }

  ~TieredStorageStressFixture() {
    INFO("=== Cleaning up Tiered Storage Stress Test ===");
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
   * Create configuration file with 64MB DRAM and 256MB file storage
   */
  void CreateConfigFile() {
    std::ofstream config_file(config_path_);
    REQUIRE(config_file.is_open());

    config_file << R"(
# Tiered Storage Stress Test Configuration
# - 64MB DRAM (fast tier, score 0.0)
# - 256MB File (slow tier, score 1.0)

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
      # Fast tier: 64MB DRAM
      - path: "ram::stress_dram"
        bdev_type: "ram"
        capacity_limit: "64MB"
        score: 0.0

      # Slow tier: 256MB File
      - path: ")" << file_storage_path_ << R"("
        bdev_type: "file"
        capacity_limit: "256MB"
        score: 1.0

    dpe:
      dpe_type: "max_bw"
)";

    config_file.close();
    INFO("Created config file: " << config_path_);
  }

  /**
   * Create test data pattern
   */
  std::vector<char> CreateTestData(size_t size, int blob_index) {
    std::vector<char> data(size);
    char pattern = static_cast<char>('A' + (blob_index % 26));
    std::memset(data.data(), pattern, size);
    return data;
  }

  /**
   * Verify test data pattern
   */
  bool VerifyTestData(const std::vector<char>& data, int blob_index) {
    char expected_pattern = static_cast<char>('A' + (blob_index % 26));
    for (size_t i = 0; i < data.size(); ++i) {
      if (data[i] != expected_pattern) {
        return false;
      }
    }
    return true;
  }
};

// Global fixture instance
static TieredStorageStressFixture* g_fixture = nullptr;

/**
 * Test: Put 128MB of data with only 64MB DRAM available
 * Expected: Data should automatically overflow to file storage
 */
TEST_CASE("TieredStorage - Put 128MB with 64MB DRAM", "[tiered][stress][put]") {
  REQUIRE(g_fixture != nullptr);
  REQUIRE(g_fixture->initialized_);

  auto* cte_client = CLIO_CTE_CLIENT;
  REQUIRE(cte_client != nullptr);

  // Create a tag for our test blobs
  std::string tag_name = "stress_test_tag";
  clio::cte::core::Tag tag(tag_name);

  INFO("Putting " << kNumBlobs << " blobs (" << (kTotalDataSize / (1024 * 1024))
                  << " MB total) with only 64MB DRAM available");

  // Allocate shared memory buffer (reuse for all blobs)
  auto shm_buffer = CLIO_IPC->AllocateBuffer(kBlobSize);
  REQUIRE(!shm_buffer.IsNull());
  ctp::ipc::ShmPtr<> shm_ptr = shm_buffer.shm_.template Cast<void>();

  int success_count = 0;
  int failure_count = 0;

  // Put all blobs - they should automatically tier to file when DRAM is full
  for (int i = 0; i < kNumBlobs; ++i) {
    std::string blob_name = "blob_" + std::to_string(i);

    // Fill buffer with pattern
    auto test_data = g_fixture->CreateTestData(kBlobSize, i);
    std::memcpy(shm_buffer.ptr_, test_data.data(), kBlobSize);

    // Put blob with high score (prefer slow tier to leave DRAM space)
    // Using score 0.5 - system should place where there's capacity
    auto put_task =
        tag.AsyncPutBlob(blob_name, shm_ptr, kBlobSize, 0, 0.5f);
    put_task.Wait();

    if (put_task->GetReturnCode() == 0) {
      success_count++;
    } else {
      failure_count++;
      INFO("Put failed for blob " << i << " with return code "
                                  << put_task->GetReturnCode());
    }
  }

  CLIO_IPC->FreeBuffer(shm_buffer);

  INFO("Put results: " << success_count << " succeeded, " << failure_count
                       << " failed");

  // All puts should succeed - data should overflow to file storage
  REQUIRE(failure_count == 0);
  REQUIRE(success_count == kNumBlobs);

  INFO("SUCCESS: All " << kNumBlobs << " blobs stored successfully");
}

/**
 * Test: ReorganizeBlob all data to score 0 (DRAM tier)
 * Expected: Should succeed even though DRAM can't hold all data
 */
TEST_CASE("TieredStorage - ReorganizeBlob to score 0",
          "[tiered][stress][reorganize]") {
  REQUIRE(g_fixture != nullptr);
  REQUIRE(g_fixture->initialized_);

  auto* cte_client = CLIO_CTE_CLIENT;
  REQUIRE(cte_client != nullptr);

  // Get the tag we created in the previous test
  std::string tag_name = "stress_test_tag";
  clio::cte::core::Tag tag(tag_name);
  clio::cte::core::TagId tag_id = tag.GetTagId();

  INFO("Reorganizing all " << kNumBlobs << " blobs to score 0.0 (DRAM tier)");
  INFO("Note: Only 64MB DRAM available for 128MB of data");

  int success_count = 0;
  int failure_count = 0;

  // Reorganize all blobs to score 0 (DRAM tier)
  for (int i = 0; i < kNumBlobs; ++i) {
    std::string blob_name = "blob_" + std::to_string(i);

    auto reorg_task =
        cte_client->AsyncReorganizeBlob(tag_id, blob_name, kFastTierScore);
    reorg_task.Wait();

    if (reorg_task->GetReturnCode() == 0) {
      success_count++;
    } else {
      failure_count++;
      INFO("Reorganize failed for blob " << i << " with return code "
                                         << reorg_task->GetReturnCode());
    }
  }

  INFO("Reorganize results: " << success_count << " succeeded, " << failure_count
                              << " failed");

  // All reorganizations should succeed
  // The system should handle the case where DRAM is full
  REQUIRE(failure_count == 0);
  REQUIRE(success_count == kNumBlobs);

  INFO("SUCCESS: All " << kNumBlobs << " blobs reorganized successfully");
}

/**
 * Test: Verify data integrity after reorganization
 */
TEST_CASE("TieredStorage - Verify data integrity",
          "[tiered][stress][integrity]") {
  REQUIRE(g_fixture != nullptr);
  REQUIRE(g_fixture->initialized_);

  // Get the tag
  std::string tag_name = "stress_test_tag";
  clio::cte::core::Tag tag(tag_name);

  INFO("Verifying data integrity for all " << kNumBlobs << " blobs");

  // Allocate buffer for reading
  auto read_buffer = CLIO_IPC->AllocateBuffer(kBlobSize);
  REQUIRE(!read_buffer.IsNull());

  int verified_count = 0;
  int corrupted_count = 0;

  for (int i = 0; i < kNumBlobs; ++i) {
    std::string blob_name = "blob_" + std::to_string(i);

    // Read blob
    tag.GetBlob(blob_name, read_buffer.shm_.template Cast<void>(), kBlobSize, 0);

    // Verify data pattern
    std::vector<char> read_data(kBlobSize);
    std::memcpy(read_data.data(), read_buffer.ptr_, kBlobSize);

    if (g_fixture->VerifyTestData(read_data, i)) {
      verified_count++;
    } else {
      corrupted_count++;
      INFO("Data corruption detected in blob " << i);
    }
  }

  CLIO_IPC->FreeBuffer(read_buffer);

  INFO("Integrity results: " << verified_count << " verified, " << corrupted_count
                             << " corrupted");

  REQUIRE(corrupted_count == 0);
  REQUIRE(verified_count == kNumBlobs);

  INFO("SUCCESS: All data verified successfully");
}

/**
 * Test: Cleanup all blobs
 */
TEST_CASE("TieredStorage - Cleanup", "[tiered][stress][cleanup]") {
  REQUIRE(g_fixture != nullptr);
  REQUIRE(g_fixture->initialized_);

  auto* cte_client = CLIO_CTE_CLIENT;
  REQUIRE(cte_client != nullptr);

  std::string tag_name = "stress_test_tag";
  clio::cte::core::Tag tag(tag_name);
  clio::cte::core::TagId tag_id = tag.GetTagId();

  INFO("Cleaning up all blobs");

  // Delete all blobs
  for (int i = 0; i < kNumBlobs; ++i) {
    std::string blob_name = "blob_" + std::to_string(i);
    auto del_task = cte_client->AsyncDelBlob(tag_id, blob_name);
    del_task.Wait();
  }

  // Delete the tag
  auto del_tag_task = cte_client->AsyncDelTag(tag_name);
  del_tag_task.Wait();

  INFO("Cleanup complete");
}

int main(int argc, char** argv) {
  // Create fixture (initializes runtime)
  g_fixture = new TieredStorageStressFixture();

  // Run tests with optional filter from command line
  std::string filter = (argc > 1) ? argv[1] : "";
  int result = SimpleTest::run_all_tests(filter);

  // Cleanup
  delete g_fixture;
  g_fixture = nullptr;

  // SIMPLE_TEST_PROCESS_EXIT is TerminateProcess on Windows to dodge the
  // libzmq teardown abort; a plain return elsewhere.
  SIMPLE_TEST_PROCESS_EXIT(result);
  return result;  // unreachable on Windows
}
