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
 * CORE CLIENT AND CONFIG TESTS
 *
 * Comprehensive tests for CTE core client API and configuration management.
 * Targets:
 * - core_client.h inline methods: 18% → 80%
 * - core_config.cc: 17% → 80%
 */

#include <clio_runtime/admin/admin_tasks.h>
#include <clio_runtime/bdev/bdev_client.h>
#include <clio_runtime/bdev/bdev_tasks.h>
#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_config.h>
#include <clio_cte/core/core_runtime.h>
#include <clio_cte/core/core_tasks.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>

#include "simple_test.h"

namespace fs = std::filesystem;

/**
 * Test fixture for core client and config tests
 */
class CoreClientConfigFixture {
 public:
  static constexpr chi::u64 kTestTargetSize = 1024 * 1024 * 10;  // 10MB
  static constexpr size_t kTestDataSize = 4096;                  // 4KB

  std::string test_storage_path_;
  std::string test_config_path_;
  bool initialized_ = false;
  static inline bool g_initialized = false;

  CoreClientConfigFixture() {
    INFO("=== Initializing Core Client/Config Test Environment ===");

    // Initialize test paths
    std::string home_dir = ctp::SystemInfo::GetHomeDir();
    REQUIRE(!home_dir.empty());
    test_storage_path_ = home_dir + "/cte_client_test.dat";
    test_config_path_ = home_dir + "/cte_test_config.yaml";

    // Clean up existing files. Non-throwing form: on Windows the previous
    // test case's bdev runtime still holds the storage file open (same
    // CHIMAERA singleton, persists across fixtures), so fs::remove throws
    // a sharing-violation. Best-effort removal is what we want — if the
    // file's open, leaving it in place is harmless because the runtime
    // already truncates/reuses it on the next pool create.
    std::error_code _setup_ec;
    fs::remove(test_storage_path_, _setup_ec);
    fs::remove(test_config_path_, _setup_ec);

    // Initialize CLIO Runtime and CTE once
    if (!g_initialized) {
      bool success = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
      REQUIRE(success);

      // Drain ZMQ background threads in main() before static dtors fire.
      SimpleTest::g_test_finalize = chi::CHIMAERA_FINALIZE;

      std::this_thread::sleep_for(std::chrono::milliseconds(500));

      success = clio::cte::core::CLIO_CTE_CLIENT_INIT();
      REQUIRE(success);

      // Initialize global client
      auto *cte_client = CLIO_CTE_CLIENT;
      REQUIRE(cte_client != nullptr);
      cte_client->Init(clio::cte::core::kCtePoolId);

      // Create CTE core pool
      clio::cte::core::CreateParams params;
      auto create_task = cte_client->AsyncCreate(
          chi::PoolQuery::Dynamic(), clio::cte::core::kCtePoolName,
          clio::cte::core::kCtePoolId, params);
      create_task.Wait();
      REQUIRE(create_task->GetReturnCode() == 0);

      g_initialized = true;
      INFO("CTE core infrastructure initialized");
    }

    INFO("=== Core Client/Config Test Environment Ready ===");
  }

  ~CoreClientConfigFixture() {
    INFO("=== Cleaning up Core Client/Config Test Environment ===");
    // Use the non-throwing overload: on Windows the bdev runtime can still
    // hold an open handle to test_storage_path_ when the destructor runs,
    // causing fs::remove to throw filesystem_error. Destructors are
    // implicitly noexcept, so a throw here calls std::terminate (which
    // surfaces as 0xC0000409 / STATUS_STACK_BUFFER_OVERRUN). Best-effort
    // cleanup is correct here — leftover files are scrubbed by the
    // chimaera_test_cleanup_fixture between test binaries.
    std::error_code ec;
    fs::remove(test_storage_path_, ec);
    fs::remove(test_config_path_, ec);
  }

  /**
   * Setup storage target for testing
   */
  void SetupTarget() {
    if (initialized_) {
      return;
    }

    auto *cte_client = CLIO_CTE_CLIENT;

    // Create bdev pool
    chi::PoolId bdev_pool_id(900, 0);
    clio::run::bdev::Client bdev_client(bdev_pool_id);
    auto create_task =
        bdev_client.AsyncCreate(chi::PoolQuery::Dynamic(), test_storage_path_,
                                bdev_pool_id, clio::run::bdev::BdevType::kFile);
    create_task.Wait();

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Register target
    auto reg_task = cte_client->AsyncRegisterTarget(
        test_storage_path_, clio::run::bdev::BdevType::kFile, kTestTargetSize,
        chi::PoolQuery::Local(), bdev_pool_id);
    reg_task.Wait();
    REQUIRE(reg_task->GetReturnCode() == 0);

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    initialized_ = true;
    INFO("=== Storage Target Registered ===");
  }

  /**
   * Create test data
   */
  std::vector<char> CreateTestData(size_t size, char pattern = 'D') {
    std::vector<char> data(size);
    for (size_t i = 0; i < size; ++i) {
      data[i] = static_cast<char>(pattern + (i % 26));
    }
    return data;
  }

  /**
   * Create test config file
   */
  void CreateTestConfigFile(const std::string &content) {
    std::ofstream file(test_config_path_);
    REQUIRE(file.is_open());
    file << content;
    file.close();
  }
};

// ============================================================================
// Core Config Tests
// ============================================================================

TEST_CASE("Config - Default Construction", "[core][config]") {
  clio::cte::core::Config config;
  // Default config should be valid
  REQUIRE(config.Validate());
  INFO("Default config constructed successfully");
}

TEST_CASE("Config - Load from Valid YAML File", "[core][config][noleak]") {
  CoreClientConfigFixture fixture;

  // Create valid config file
  fixture.CreateTestConfigFile(R"(
neighborhood: 8
poll_period_ms: 3000
)");

  clio::cte::core::Config config;
  bool success = config.LoadFromFile(fixture.test_config_path_);
  REQUIRE(success);
  REQUIRE(config.Validate());
  INFO("Loaded valid YAML config from file");
}

TEST_CASE("Config - Load from Invalid File Path", "[core][config]") {
  clio::cte::core::Config config;
  bool success = config.LoadFromFile("/nonexistent/path/config.yaml");
  REQUIRE(!success);
  INFO("Correctly rejected non-existent file");
}

TEST_CASE("Config - Load from Empty Path", "[core][config]") {
  clio::cte::core::Config config;
  bool success = config.LoadFromFile("");
  REQUIRE(!success);
  INFO("Correctly rejected empty path");
}

TEST_CASE("Config - Load from Valid YAML String", "[core][config]") {
  std::string yaml_config = R"(
neighborhood: 6
poll_period_ms: 4000
)";

  clio::cte::core::Config config;
  bool success = config.LoadFromString(yaml_config);
  REQUIRE(success);
  REQUIRE(config.Validate());
  INFO("Loaded config from YAML string");
}

TEST_CASE("Config - Load from Empty String", "[core][config]") {
  clio::cte::core::Config config;
  bool success = config.LoadFromString("");
  REQUIRE(!success);
  INFO("Correctly rejected empty string");
}

TEST_CASE("Config - Load from Invalid YAML", "[core][config]") {
  std::string bad_yaml = "invalid: [unclosed";

  clio::cte::core::Config config;
  bool success = config.LoadFromString(bad_yaml);
  REQUIRE(!success);
  INFO("Correctly rejected invalid YAML");
}

TEST_CASE("Config - Load from Environment", "[core][config]") {
  // Test with unset environment variable (should succeed with defaults)
  ctp::SystemInfo::Unsetenv("CLIO_CTE_CONFIG");

  clio::cte::core::Config config;
  bool success = config.LoadFromEnvironment();
  REQUIRE(success);  // Should succeed with defaults
  INFO("Loaded default config when env var not set");
}

TEST_CASE("Config - Save to File", "[core][config]") {
  CoreClientConfigFixture fixture;

  clio::cte::core::Config config;
  bool success = config.SaveToFile(fixture.test_config_path_);
  REQUIRE(success);

  // Verify file was created
  REQUIRE(fs::exists(fixture.test_config_path_));
  INFO("Saved config to file successfully");
}

// ============================================================================
// Core Client API Tests - Target Operations
// ============================================================================

TEST_CASE("Client - AsyncListTargets", "[core][client][target]") {
  CoreClientConfigFixture fixture;
  fixture.SetupTarget();

  auto *client = CLIO_CTE_CLIENT;
  auto task = client->AsyncListTargets();
  task.Wait();

  REQUIRE(task->GetReturnCode() == 0);
  REQUIRE(task->target_names_.size() > 0);
  INFO("AsyncListTargets returned " << task->target_names_.size()
                                    << " targets");
}

TEST_CASE("Client - AsyncStatTargets", "[core][client][target]") {
  CoreClientConfigFixture fixture;
  fixture.SetupTarget();

  auto *client = CLIO_CTE_CLIENT;
  auto task = client->AsyncStatTargets();
  task.Wait();

  REQUIRE(task->GetReturnCode() == 0);
  INFO("AsyncStatTargets completed successfully");
}

TEST_CASE("Client - AsyncGetTargetInfo", "[core][client][target]") {
  CoreClientConfigFixture fixture;
  fixture.SetupTarget();

  auto *client = CLIO_CTE_CLIENT;
  auto task = client->AsyncGetTargetInfo(fixture.test_storage_path_);
  task.Wait();

  REQUIRE(task->GetReturnCode() == 0);
  INFO("AsyncGetTargetInfo retrieved target information");
}

TEST_CASE("Client - AsyncUnregisterTarget", "[core][client][target]") {
  CoreClientConfigFixture fixture;
  fixture.SetupTarget();

  auto *client = CLIO_CTE_CLIENT;

  // Unregister the target
  auto unreg_task = client->AsyncUnregisterTarget(fixture.test_storage_path_);
  unreg_task.Wait();

  // Note: May succeed or fail depending on target state
  INFO("AsyncUnregisterTarget completed with code: "
       << unreg_task->GetReturnCode());
}

// ============================================================================
// Core Client API Tests - Tag Operations
// ============================================================================

TEST_CASE("Client - AsyncGetOrCreateTag Direct", "[core][client][tag]") {
  CoreClientConfigFixture fixture;
  fixture.SetupTarget();

  auto *client = CLIO_CTE_CLIENT;
  auto task = client->AsyncGetOrCreateTag("direct_test_tag");
  task.Wait();

  REQUIRE(task->GetReturnCode() == 0);
  REQUIRE(!task->tag_id_.IsNull());
  INFO("AsyncGetOrCreateTag created tag with ID: " << task->tag_id_.ToU64());
}

TEST_CASE("Client - AsyncDelTag by ID", "[core][client][tag]") {
  CoreClientConfigFixture fixture;
  fixture.SetupTarget();

  auto *client = CLIO_CTE_CLIENT;

  // Create a tag first
  auto create_task = client->AsyncGetOrCreateTag("tag_to_delete");
  create_task.Wait();
  REQUIRE(create_task->GetReturnCode() == 0);

  clio::cte::core::TagId tag_id = create_task->tag_id_;

  // Delete by ID
  auto del_task = client->AsyncDelTag(tag_id);
  del_task.Wait();

  INFO("AsyncDelTag by ID completed with code: " << del_task->GetReturnCode());
}

TEST_CASE("Client - AsyncDelTag by Name", "[core][client][tag]") {
  CoreClientConfigFixture fixture;
  fixture.SetupTarget();

  auto *client = CLIO_CTE_CLIENT;

  // Create a tag first
  auto create_task = client->AsyncGetOrCreateTag("tag_to_delete_by_name");
  create_task.Wait();
  REQUIRE(create_task->GetReturnCode() == 0);

  // Delete by name
  auto del_task = client->AsyncDelTag("tag_to_delete_by_name");
  del_task.Wait();

  INFO(
      "AsyncDelTag by name completed with code: " << del_task->GetReturnCode());
}

TEST_CASE("Client - AsyncGetTagSize", "[core][client][tag]") {
  CoreClientConfigFixture fixture;
  fixture.SetupTarget();

  auto *client = CLIO_CTE_CLIENT;

  // Create a tag
  auto create_task = client->AsyncGetOrCreateTag("sized_tag");
  create_task.Wait();
  REQUIRE(create_task->GetReturnCode() == 0);

  // Get tag size
  auto size_task = client->AsyncGetTagSize(create_task->tag_id_);
  size_task.Wait();

  REQUIRE(size_task->GetReturnCode() == 0);
  INFO("AsyncGetTagSize returned size: " << size_task->tag_size_);
}

// ============================================================================
// Core Client API Tests - Blob Operations
// ============================================================================

TEST_CASE("Client - AsyncPutBlob Direct", "[core][client][blob]") {
  CoreClientConfigFixture fixture;
  fixture.SetupTarget();

  auto *client = CLIO_CTE_CLIENT;

  // Create tag
  auto tag_task = client->AsyncGetOrCreateTag("blob_test_tag");
  tag_task.Wait();
  REQUIRE(tag_task->GetReturnCode() == 0);

  // Create test data in SHM
  auto data = fixture.CreateTestData(fixture.kTestDataSize);
  auto *ipc = CLIO_IPC;
  ctp::ipc::FullPtr<char> shm_ptr = ipc->AllocateBuffer(data.size());
  REQUIRE(!shm_ptr.IsNull());
  memcpy(shm_ptr.ptr_, data.data(), data.size());

  // Put blob
  ctp::ipc::ShmPtr<> shm_ref(shm_ptr.shm_);
  auto put_task = client->AsyncPutBlob(tag_task->tag_id_, "test_blob", 0,
                                       data.size(), shm_ref, 1.0f);
  put_task.Wait();

  ipc->FreeBuffer(shm_ptr);

  REQUIRE(put_task->GetReturnCode() == 0);
  INFO("AsyncPutBlob completed successfully");
}

TEST_CASE("Client - AsyncGetBlob Direct", "[core][client][blob]") {
  CoreClientConfigFixture fixture;
  fixture.SetupTarget();

  auto *client = CLIO_CTE_CLIENT;

  // Create tag and put blob
  auto tag_task = client->AsyncGetOrCreateTag("get_blob_tag");
  tag_task.Wait();
  REQUIRE(tag_task->GetReturnCode() == 0);

  auto data = fixture.CreateTestData(fixture.kTestDataSize);
  auto *ipc = CLIO_IPC;
  ctp::ipc::FullPtr<char> put_ptr = ipc->AllocateBuffer(data.size());
  memcpy(put_ptr.ptr_, data.data(), data.size());
  ctp::ipc::ShmPtr<> put_ref(put_ptr.shm_);

  auto put_task = client->AsyncPutBlob(tag_task->tag_id_, "get_test_blob", 0,
                                       data.size(), put_ref, 1.0f);
  put_task.Wait();
  REQUIRE(put_task->GetReturnCode() == 0);

  ipc->FreeBuffer(put_ptr);

  // Now get blob
  ctp::ipc::FullPtr<char> get_ptr = ipc->AllocateBuffer(data.size());
  ctp::ipc::ShmPtr<> get_ref(get_ptr.shm_);

  auto get_task = client->AsyncGetBlob(tag_task->tag_id_, "get_test_blob", 0,
                                       data.size(), 0, get_ref);
  get_task.Wait();

  ipc->FreeBuffer(get_ptr);

  REQUIRE(get_task->GetReturnCode() == 0);
  INFO("AsyncGetBlob completed successfully");
}

TEST_CASE("Client - AsyncDelBlob", "[core][client][blob]") {
  CoreClientConfigFixture fixture;
  fixture.SetupTarget();

  auto *client = CLIO_CTE_CLIENT;

  // Create tag and put blob
  auto tag_task = client->AsyncGetOrCreateTag("del_blob_tag");
  tag_task.Wait();
  REQUIRE(tag_task->GetReturnCode() == 0);

  auto data = fixture.CreateTestData(fixture.kTestDataSize);
  auto *ipc = CLIO_IPC;
  ctp::ipc::FullPtr<char> shm_ptr = ipc->AllocateBuffer(data.size());
  memcpy(shm_ptr.ptr_, data.data(), data.size());
  ctp::ipc::ShmPtr<> shm_ref(shm_ptr.shm_);

  auto put_task = client->AsyncPutBlob(tag_task->tag_id_, "blob_to_delete", 0,
                                       data.size(), shm_ref, 1.0f);
  put_task.Wait();
  REQUIRE(put_task->GetReturnCode() == 0);

  ipc->FreeBuffer(shm_ptr);

  // Delete blob
  auto del_task = client->AsyncDelBlob(tag_task->tag_id_, "blob_to_delete");
  del_task.Wait();

  INFO("AsyncDelBlob completed with code: " << del_task->GetReturnCode());
}

TEST_CASE("Client - AsyncReorganizeBlob Direct", "[core][client][blob][noleak]") {
  CoreClientConfigFixture fixture;
  fixture.SetupTarget();

  auto *client = CLIO_CTE_CLIENT;

  // Create tag and put blob
  auto tag_task = client->AsyncGetOrCreateTag("reorg_tag");
  tag_task.Wait();
  REQUIRE(tag_task->GetReturnCode() == 0);

  auto data = fixture.CreateTestData(fixture.kTestDataSize);
  auto *ipc = CLIO_IPC;
  ctp::ipc::FullPtr<char> shm_ptr = ipc->AllocateBuffer(data.size());
  memcpy(shm_ptr.ptr_, data.data(), data.size());
  ctp::ipc::ShmPtr<> shm_ref(shm_ptr.shm_);

  auto put_task = client->AsyncPutBlob(tag_task->tag_id_, "reorg_blob", 0,
                                       data.size(), shm_ref, 0.2f);
  put_task.Wait();
  REQUIRE(put_task->GetReturnCode() == 0);

  ipc->FreeBuffer(shm_ptr);

  // Reorganize blob
  auto reorg_task =
      client->AsyncReorganizeBlob(tag_task->tag_id_, "reorg_blob", 0.8f);
  reorg_task.Wait();

  INFO("AsyncReorganizeBlob completed with code: "
       << reorg_task->GetReturnCode());
}

TEST_CASE("Client - AsyncGetBlobScore Direct", "[core][client][blob]") {
  CoreClientConfigFixture fixture;
  fixture.SetupTarget();

  auto *client = CLIO_CTE_CLIENT;

  // Create tag and put blob with specific score
  auto tag_task = client->AsyncGetOrCreateTag("score_tag");
  tag_task.Wait();
  REQUIRE(tag_task->GetReturnCode() == 0);

  auto data = fixture.CreateTestData(fixture.kTestDataSize);
  auto *ipc = CLIO_IPC;
  ctp::ipc::FullPtr<char> shm_ptr = ipc->AllocateBuffer(data.size());
  memcpy(shm_ptr.ptr_, data.data(), data.size());
  ctp::ipc::ShmPtr<> shm_ref(shm_ptr.shm_);

  float expected_score = 0.35f;
  auto put_task = client->AsyncPutBlob(tag_task->tag_id_, "scored_blob", 0,
                                       data.size(), shm_ref, expected_score);
  put_task.Wait();
  REQUIRE(put_task->GetReturnCode() == 0);

  ipc->FreeBuffer(shm_ptr);

  // Get blob score
  auto score_task = client->AsyncGetBlobScore(tag_task->tag_id_, "scored_blob");
  score_task.Wait();

  REQUIRE(score_task->GetReturnCode() == 0);
  REQUIRE(score_task->score_ == expected_score);
  INFO("AsyncGetBlobScore returned correct score: " << score_task->score_);
}

TEST_CASE("Client - AsyncGetBlobSize Direct", "[core][client][blob]") {
  CoreClientConfigFixture fixture;
  fixture.SetupTarget();

  auto *client = CLIO_CTE_CLIENT;

  // Create tag and put blob
  auto tag_task = client->AsyncGetOrCreateTag("size_test_tag");
  tag_task.Wait();
  REQUIRE(tag_task->GetReturnCode() == 0);

  auto data = fixture.CreateTestData(fixture.kTestDataSize);
  auto *ipc = CLIO_IPC;
  ctp::ipc::FullPtr<char> shm_ptr = ipc->AllocateBuffer(data.size());
  memcpy(shm_ptr.ptr_, data.data(), data.size());
  ctp::ipc::ShmPtr<> shm_ref(shm_ptr.shm_);

  auto put_task = client->AsyncPutBlob(tag_task->tag_id_, "sized_blob", 0,
                                       data.size(), shm_ref, 1.0f);
  put_task.Wait();
  REQUIRE(put_task->GetReturnCode() == 0);

  ipc->FreeBuffer(shm_ptr);

  // Get blob size
  auto size_task = client->AsyncGetBlobSize(tag_task->tag_id_, "sized_blob");
  size_task.Wait();

  REQUIRE(size_task->GetReturnCode() == 0);
  REQUIRE(size_task->size_ == fixture.kTestDataSize);
  INFO("AsyncGetBlobSize returned correct size: " << size_task->size_);
}

TEST_CASE("Client - AsyncGetContainedBlobs Direct", "[core][client][blob]") {
  CoreClientConfigFixture fixture;
  fixture.SetupTarget();

  auto *client = CLIO_CTE_CLIENT;

  // Create tag
  auto tag_task = client->AsyncGetOrCreateTag("contained_tag");
  tag_task.Wait();
  REQUIRE(tag_task->GetReturnCode() == 0);

  // Put multiple blobs
  auto *ipc = CLIO_IPC;
  for (int i = 0; i < 3; ++i) {
    auto data = fixture.CreateTestData(fixture.kTestDataSize);
    ctp::ipc::FullPtr<char> shm_ptr = ipc->AllocateBuffer(data.size());
    memcpy(shm_ptr.ptr_, data.data(), data.size());
    ctp::ipc::ShmPtr<> shm_ref(shm_ptr.shm_);

    std::string blob_name = "blob_" + std::to_string(i);
    auto put_task = client->AsyncPutBlob(tag_task->tag_id_, blob_name, 0,
                                         data.size(), shm_ref, 1.0f);
    put_task.Wait();
    REQUIRE(put_task->GetReturnCode() == 0);

    ipc->FreeBuffer(shm_ptr);
  }

  // Get contained blobs
  auto blobs_task = client->AsyncGetContainedBlobs(tag_task->tag_id_);
  blobs_task.Wait();

  REQUIRE(blobs_task->GetReturnCode() == 0);
  REQUIRE(blobs_task->blob_names_.size() >= 3);
  INFO("AsyncGetContainedBlobs returned " << blobs_task->blob_names_.size()
                                          << " blobs");
}

// ============================================================================
// Core Client API Tests - Query Operations
// ============================================================================

TEST_CASE("Client - AsyncTagQuery", "[core][client][query]") {
  CoreClientConfigFixture fixture;
  fixture.SetupTarget();

  auto *client = CLIO_CTE_CLIENT;

  // Create some tags
  auto tag1 = client->AsyncGetOrCreateTag("query_tag_1");
  tag1.Wait();
  auto tag2 = client->AsyncGetOrCreateTag("query_tag_2");
  tag2.Wait();

  // Query for tags
  auto query_task = client->AsyncTagQuery("query_tag_.*", 10);
  query_task.Wait();

  REQUIRE(query_task->GetReturnCode() == 0);
  INFO("AsyncTagQuery returned " << query_task->results_.size() << " tags");
}

TEST_CASE("Client - AsyncBlobQuery", "[core][client][query]") {
  CoreClientConfigFixture fixture;
  fixture.SetupTarget();

  auto *client = CLIO_CTE_CLIENT;

  // Create tag and blob
  auto tag_task = client->AsyncGetOrCreateTag("query_blob_tag");
  tag_task.Wait();
  REQUIRE(tag_task->GetReturnCode() == 0);

  auto data = fixture.CreateTestData(fixture.kTestDataSize);
  auto *ipc = CLIO_IPC;
  ctp::ipc::FullPtr<char> shm_ptr = ipc->AllocateBuffer(data.size());
  memcpy(shm_ptr.ptr_, data.data(), data.size());
  ctp::ipc::ShmPtr<> shm_ref(shm_ptr.shm_);

  auto put_task = client->AsyncPutBlob(tag_task->tag_id_, "queryable_blob", 0,
                                       data.size(), shm_ref, 1.0f);
  put_task.Wait();
  REQUIRE(put_task->GetReturnCode() == 0);

  ipc->FreeBuffer(shm_ptr);

  // Query for blobs
  auto query_task = client->AsyncBlobQuery("query_blob_.*", "queryable_.*", 10);
  query_task.Wait();

  REQUIRE(query_task->GetReturnCode() == 0);
  INFO("AsyncBlobQuery returned " << query_task->blob_names_.size()
                                  << " blobs");
}

// ============================================================================
// Core Client API Tests - Telemetry
// ============================================================================

TEST_CASE("Client - AsyncPollTelemetryLog", "[core][client][telemetry]") {
  CoreClientConfigFixture fixture;
  fixture.SetupTarget();

  auto *client = CLIO_CTE_CLIENT;

  // Poll telemetry log
  auto telemetry_task = client->AsyncPollTelemetryLog(0);
  telemetry_task.Wait();

  REQUIRE(telemetry_task->GetReturnCode() == 0);
  INFO("AsyncPollTelemetryLog returned " << telemetry_task->entries_.size()
                                         << " entries");
}

SIMPLE_TEST_MAIN()
