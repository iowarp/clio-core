/**
 * test_core_runtime_coverage.cc
 *
 * Comprehensive tests for CTE Core Runtime execution paths
 * Focus: Improving coverage of core_runtime.cc methods
 */

#include "simple_test.h"
#include <wrp_cte/core/core_client.h>
#include <wrp_cte/core/core_tasks.h>
#include <chimaera/chimaera.h>
#include <chimaera/admin/admin_client.h>
#include <chimaera/bdev/bdev_client.h>
#include <chimaera/bdev/bdev_tasks.h>
#include <hermes_shm/util/logging.h>

using namespace wrp_cte::core;

/**
 * Test fixture for runtime coverage tests
 * Ensures proper CTE initialization following the established pattern
 */
class RuntimeCoverageFixture {
public:
  static inline bool g_initialized = false;
  static inline bool g_target_setup = false;
  std::string test_storage_path_;
  static constexpr size_t kTestDataSize = 1024;

  RuntimeCoverageFixture() {
    if (!g_initialized) {
      INFO("=== Initializing Runtime Coverage Test Environment ===");

      // Step 1: Initialize Chimaera runtime
      bool success = chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, true);
      if (!success) {
        throw std::runtime_error("CHIMAERA_INIT failed");
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(500));

      // Step 2: Initialize CTE client subsystem (CRITICAL!)
      success = wrp_cte::core::WRP_CTE_CLIENT_INIT();
      if (!success) {
        throw std::runtime_error("WRP_CTE_CLIENT_INIT failed");
      }

      // Step 3: Set pool ID on global client
      auto *cte_client = WRP_CTE_CLIENT;
      cte_client->Init(wrp_cte::core::kCtePoolId);

      // Step 4: Create CTE core pool
      wrp_cte::core::CreateParams params;
      auto create_task = cte_client->AsyncCreate(
          chi::PoolQuery::Dynamic(),
          wrp_cte::core::kCtePoolName,
          wrp_cte::core::kCtePoolId,
          params);
      create_task.Wait();

      if (create_task->GetReturnCode() != 0) {
        throw std::runtime_error("CTE pool creation failed");
      }

      INFO("CTE core infrastructure initialized");
      g_initialized = true;
    }

    test_storage_path_ = std::string(std::getenv("HOME")) + "/cte_runtime_test.dat";
  }

  void SetupTarget() {
    if (g_target_setup) {
      return;
    }

    auto *cte_client = WRP_CTE_CLIENT;

    // Create bdev pool for storage
    chi::PoolId bdev_pool_id(900, 0);
    size_t target_size = 10 * 1024 * 1024;  // 10 MB
    chimaera::bdev::Client bdev_client(bdev_pool_id);
    auto bdev_create = bdev_client.AsyncCreate(
        chi::PoolQuery::Dynamic(),
        test_storage_path_,
        bdev_pool_id,
        chimaera::bdev::BdevType::kFile,
        target_size);
    bdev_create.Wait();

    // Register target with CTE
    auto reg_task = cte_client->AsyncRegisterTarget(
        test_storage_path_,
        chimaera::bdev::BdevType::kFile,
        target_size,
        chi::PoolQuery::Local(),
        bdev_pool_id);
    reg_task.Wait();

    if (reg_task->GetReturnCode() != 0) {
      throw std::runtime_error("Target registration failed");
    }

    INFO("Storage target registered");
    g_target_setup = true;
  }

  std::vector<char> CreateTestData(size_t size) {
    std::vector<char> data(size);
    for (size_t i = 0; i < size; ++i) {
      data[i] = static_cast<char>('A' + (i % 26));
    }
    return data;
  }

  ~RuntimeCoverageFixture() {
    INFO("Runtime coverage test cleanup");
  }
};

// ============================================================================
// GetTargetInfo Runtime Tests - Comprehensive Coverage
// ============================================================================

TEST_CASE("Runtime - GetTargetInfo Success Path", "[runtime][target]") {
  RuntimeCoverageFixture fixture;
  fixture.SetupTarget();

  auto *client = WRP_CTE_CLIENT;

  // Test successful target info retrieval
  auto task = client->AsyncGetTargetInfo(fixture.test_storage_path_);
  task.Wait();

  REQUIRE(task->GetReturnCode() == 0);
  INFO("GetTargetInfo: target_score=" << task->target_score_
       << ", remaining_space=" << task->remaining_space_);
}

TEST_CASE("Runtime - GetTargetInfo NonExistent Target", "[runtime][target][error]") {
  RuntimeCoverageFixture fixture;
  fixture.SetupTarget();

  auto *client = WRP_CTE_CLIENT;

  // Test getting info for non-existent target (error path)
  auto task = client->AsyncGetTargetInfo("/nonexistent/target.dat");
  task.Wait();

  REQUIRE(task->GetReturnCode() != 0);
  INFO("GetTargetInfo correctly failed for non-existent target");
}

// ============================================================================
// UnregisterTarget Runtime Tests
// ============================================================================

TEST_CASE("Runtime - UnregisterTarget Success", "[runtime][target]") {
  RuntimeCoverageFixture fixture;
  fixture.SetupTarget();

  auto *client = WRP_CTE_CLIENT;

  // Register a temporary target
  std::string temp_target = std::string(std::getenv("HOME")) + "/temp_unregister_test.dat";

  chi::PoolId temp_bdev_pool_id(901, 0);
  size_t temp_target_size = 5 * 1024 * 1024;

  chimaera::bdev::Client temp_bdev_client(temp_bdev_pool_id);
  auto bdev_create = temp_bdev_client.AsyncCreate(
      chi::PoolQuery::Dynamic(),
      temp_target,
      temp_bdev_pool_id,
      chimaera::bdev::BdevType::kFile,
      temp_target_size);
  bdev_create.Wait();

  auto reg_task = client->AsyncRegisterTarget(
      temp_target,
      chimaera::bdev::BdevType::kFile,
      temp_target_size,
      chi::PoolQuery::Local(),
      temp_bdev_pool_id);
  reg_task.Wait();
  REQUIRE(reg_task->GetReturnCode() == 0);

  // Now unregister it
  auto unreg_task = client->AsyncUnregisterTarget(temp_target);
  unreg_task.Wait();

  REQUIRE(unreg_task->GetReturnCode() == 0);
  INFO("UnregisterTarget successful");
}

TEST_CASE("Runtime - UnregisterTarget NonExistent", "[runtime][target][error]") {
  RuntimeCoverageFixture fixture;
  fixture.SetupTarget();

  auto *client = WRP_CTE_CLIENT;

  // Try to unregister non-existent target (error path)
  auto unreg_task = client->AsyncUnregisterTarget("/nonexistent/target.dat");
  unreg_task.Wait();

  REQUIRE(unreg_task->GetReturnCode() != 0);
  INFO("UnregisterTarget correctly failed for non-existent target");
}

// ============================================================================
// DelBlob Runtime Tests - Comprehensive Coverage
// ============================================================================

TEST_CASE("Runtime - DelBlob Success Path", "[runtime][blob]") {
  RuntimeCoverageFixture fixture;
  fixture.SetupTarget();

  auto *client = WRP_CTE_CLIENT;

  // Create tag and put blob
  auto tag_task = client->AsyncGetOrCreateTag("delblob_test_tag");
  tag_task.Wait();
  REQUIRE(tag_task->GetReturnCode() == 0);

  auto data = fixture.CreateTestData(fixture.kTestDataSize);
  auto *ipc = CHI_IPC;
  hipc::FullPtr<char> shm_ptr = ipc->AllocateBuffer(data.size());
  memcpy(shm_ptr.ptr_, data.data(), data.size());
  hipc::ShmPtr<> shm_ref(shm_ptr.shm_);

  auto put_task = client->AsyncPutBlob(
      tag_task->tag_id_, "blob_for_deletion", 0, data.size(), shm_ref, 1.0f);
  put_task.Wait();
  REQUIRE(put_task->GetReturnCode() == 0);

  ipc->FreeBuffer(shm_ptr);

  // Delete the blob
  auto del_task = client->AsyncDelBlob(tag_task->tag_id_, "blob_for_deletion");
  del_task.Wait();

  REQUIRE(del_task->GetReturnCode() == 0);
  INFO("DelBlob successful");
}

TEST_CASE("Runtime - DelBlob NonExistent Blob", "[runtime][blob][error]") {
  RuntimeCoverageFixture fixture;
  fixture.SetupTarget();

  auto *client = WRP_CTE_CLIENT;

  // Create tag
  auto tag_task = client->AsyncGetOrCreateTag("delblob_error_tag");
  tag_task.Wait();
  REQUIRE(tag_task->GetReturnCode() == 0);

  // Try to delete non-existent blob (error path)
  auto del_task = client->AsyncDelBlob(tag_task->tag_id_, "nonexistent_blob");
  del_task.Wait();

  REQUIRE(del_task->GetReturnCode() != 0);
  INFO("DelBlob correctly failed for non-existent blob");
}

TEST_CASE("Runtime - DelBlob Empty Name", "[runtime][blob][error]") {
  RuntimeCoverageFixture fixture;
  fixture.SetupTarget();

  auto *client = WRP_CTE_CLIENT;

  // Create tag
  auto tag_task = client->AsyncGetOrCreateTag("delblob_empty_tag");
  tag_task.Wait();
  REQUIRE(tag_task->GetReturnCode() == 0);

  // Try to delete blob with empty name (error path)
  auto del_task = client->AsyncDelBlob(tag_task->tag_id_, "");
  del_task.Wait();

  REQUIRE(del_task->GetReturnCode() != 0);
  INFO("DelBlob correctly failed for empty blob name");
}

// ============================================================================
// GetBlobScore Runtime Tests
// ============================================================================

TEST_CASE("Runtime - GetBlobScore Success", "[runtime][blob]") {
  RuntimeCoverageFixture fixture;
  fixture.SetupTarget();

  auto *client = WRP_CTE_CLIENT;

  // Create tag and put blob with specific score
  auto tag_task = client->AsyncGetOrCreateTag("score_test_tag");
  tag_task.Wait();
  REQUIRE(tag_task->GetReturnCode() == 0);

  auto data = fixture.CreateTestData(fixture.kTestDataSize);
  auto *ipc = CHI_IPC;
  hipc::FullPtr<char> shm_ptr = ipc->AllocateBuffer(data.size());
  memcpy(shm_ptr.ptr_, data.data(), data.size());
  hipc::ShmPtr<> shm_ref(shm_ptr.shm_);

  float expected_score = 4.75f;
  auto put_task = client->AsyncPutBlob(
      tag_task->tag_id_, "scored_blob_test", 0, data.size(), shm_ref, expected_score);
  put_task.Wait();
  REQUIRE(put_task->GetReturnCode() == 0);

  ipc->FreeBuffer(shm_ptr);

  // Get blob score
  auto score_task = client->AsyncGetBlobScore(tag_task->tag_id_, "scored_blob_test");
  score_task.Wait();

  REQUIRE(score_task->GetReturnCode() == 0);
  REQUIRE(score_task->score_ == expected_score);
  INFO("GetBlobScore returned correct score: " << score_task->score_);
}

TEST_CASE("Runtime - GetBlobScore NonExistent Blob", "[runtime][blob][error]") {
  RuntimeCoverageFixture fixture;
  fixture.SetupTarget();

  auto *client = WRP_CTE_CLIENT;

  // Create tag
  auto tag_task = client->AsyncGetOrCreateTag("score_error_tag");
  tag_task.Wait();
  REQUIRE(tag_task->GetReturnCode() == 0);

  // Try to get score for non-existent blob (error path)
  auto score_task = client->AsyncGetBlobScore(tag_task->tag_id_, "nonexistent_blob");
  score_task.Wait();

  REQUIRE(score_task->GetReturnCode() != 0);
  INFO("GetBlobScore correctly failed for non-existent blob");
}

TEST_CASE("Runtime - GetBlobScore Empty Name", "[runtime][blob][error]") {
  RuntimeCoverageFixture fixture;
  fixture.SetupTarget();

  auto *client = WRP_CTE_CLIENT;

  // Create tag
  auto tag_task = client->AsyncGetOrCreateTag("score_empty_tag");
  tag_task.Wait();
  REQUIRE(tag_task->GetReturnCode() == 0);

  // Try to get score with empty name (error path)
  auto score_task = client->AsyncGetBlobScore(tag_task->tag_id_, "");
  score_task.Wait();

  REQUIRE(score_task->GetReturnCode() != 0);
  INFO("GetBlobScore correctly failed for empty blob name");
}

// ============================================================================
// GetBlobSize Runtime Tests
// ============================================================================

TEST_CASE("Runtime - GetBlobSize Success", "[runtime][blob]") {
  RuntimeCoverageFixture fixture;
  fixture.SetupTarget();

  auto *client = WRP_CTE_CLIENT;

  // Create tag and put blob
  auto tag_task = client->AsyncGetOrCreateTag("blobsize_tag");
  tag_task.Wait();
  REQUIRE(tag_task->GetReturnCode() == 0);

  size_t expected_size = 2048;
  auto data = fixture.CreateTestData(expected_size);
  auto *ipc = CHI_IPC;
  hipc::FullPtr<char> shm_ptr = ipc->AllocateBuffer(data.size());
  memcpy(shm_ptr.ptr_, data.data(), data.size());
  hipc::ShmPtr<> shm_ref(shm_ptr.shm_);

  auto put_task = client->AsyncPutBlob(
      tag_task->tag_id_, "sized_blob", 0, data.size(), shm_ref, 1.0f);
  put_task.Wait();
  REQUIRE(put_task->GetReturnCode() == 0);

  ipc->FreeBuffer(shm_ptr);

  // Get blob size
  auto size_task = client->AsyncGetBlobSize(tag_task->tag_id_, "sized_blob");
  size_task.Wait();

  REQUIRE(size_task->GetReturnCode() == 0);
  REQUIRE(size_task->size_ == expected_size);
  INFO("GetBlobSize returned correct size: " << size_task->size_);
}

TEST_CASE("Runtime - GetBlobSize NonExistent", "[runtime][blob][error]") {
  RuntimeCoverageFixture fixture;
  fixture.SetupTarget();

  auto *client = WRP_CTE_CLIENT;

  // Create tag
  auto tag_task = client->AsyncGetOrCreateTag("size_error_tag");
  tag_task.Wait();
  REQUIRE(tag_task->GetReturnCode() == 0);

  // Try to get size for non-existent blob
  auto size_task = client->AsyncGetBlobSize(tag_task->tag_id_, "nonexistent_blob");
  size_task.Wait();

  REQUIRE(size_task->GetReturnCode() != 0);
  INFO("GetBlobSize correctly failed for non-existent blob");
}

// ============================================================================
// GetTagSize Runtime Tests
// ============================================================================

TEST_CASE("Runtime - GetTagSize Success", "[runtime][tag]") {
  RuntimeCoverageFixture fixture;
  fixture.SetupTarget();

  auto *client = WRP_CTE_CLIENT;

  // Create tag and put multiple blobs
  auto tag_task = client->AsyncGetOrCreateTag("tagsize_test");
  tag_task.Wait();
  REQUIRE(tag_task->GetReturnCode() == 0);

  auto *ipc = CHI_IPC;

  // Put first blob
  size_t size1 = 1024;
  auto data1 = fixture.CreateTestData(size1);
  hipc::FullPtr<char> shm1 = ipc->AllocateBuffer(data1.size());
  memcpy(shm1.ptr_, data1.data(), data1.size());
  hipc::ShmPtr<> ref1(shm1.shm_);

  auto put1 = client->AsyncPutBlob(tag_task->tag_id_, "blob1", 0, size1, ref1, 1.0f);
  put1.Wait();
  REQUIRE(put1->GetReturnCode() == 0);
  ipc->FreeBuffer(shm1);

  // Put second blob
  size_t size2 = 512;
  auto data2 = fixture.CreateTestData(size2);
  hipc::FullPtr<char> shm2 = ipc->AllocateBuffer(data2.size());
  memcpy(shm2.ptr_, data2.data(), data2.size());
  hipc::ShmPtr<> ref2(shm2.shm_);

  auto put2 = client->AsyncPutBlob(tag_task->tag_id_, "blob2", 0, size2, ref2, 1.0f);
  put2.Wait();
  REQUIRE(put2->GetReturnCode() == 0);
  ipc->FreeBuffer(shm2);

  // Get tag size
  auto size_task = client->AsyncGetTagSize(tag_task->tag_id_);
  size_task.Wait();

  REQUIRE(size_task->GetReturnCode() == 0);
  INFO("GetTagSize returned: " << size_task->tag_size_);
}

// ============================================================================
// DelTag Runtime Tests
// ============================================================================

TEST_CASE("Runtime - DelTag by Name Success", "[runtime][tag]") {
  RuntimeCoverageFixture fixture;
  fixture.SetupTarget();

  auto *client = WRP_CTE_CLIENT;

  // Create tag
  std::string tag_name = "tag_to_delete";
  auto create_task = client->AsyncGetOrCreateTag(tag_name);
  create_task.Wait();
  REQUIRE(create_task->GetReturnCode() == 0);

  // Delete tag by name
  auto del_task = client->AsyncDelTag(tag_name);
  del_task.Wait();

  INFO("DelTag by name completed with code: " << del_task->GetReturnCode());
}

TEST_CASE("Runtime - DelTag by ID Success", "[runtime][tag]") {
  RuntimeCoverageFixture fixture;
  fixture.SetupTarget();

  auto *client = WRP_CTE_CLIENT;

  // Create tag
  auto create_task = client->AsyncGetOrCreateTag("tag_to_delete_by_id");
  create_task.Wait();
  REQUIRE(create_task->GetReturnCode() == 0);

  TagId tag_id = create_task->tag_id_;

  // Delete tag by ID
  auto del_task = client->AsyncDelTag(tag_id);
  del_task.Wait();

  INFO("DelTag by ID completed with code: " << del_task->GetReturnCode());
}

// ============================================================================
// GetContainedBlobs Runtime Tests
// ============================================================================

TEST_CASE("Runtime - GetContainedBlobs Success", "[runtime][blob]") {
  RuntimeCoverageFixture fixture;
  fixture.SetupTarget();

  auto *client = WRP_CTE_CLIENT;

  // Create tag and add multiple blobs
  auto tag_task = client->AsyncGetOrCreateTag("contained_blobs_tag");
  tag_task.Wait();
  REQUIRE(tag_task->GetReturnCode() == 0);

  auto *ipc = CHI_IPC;
  auto data = fixture.CreateTestData(fixture.kTestDataSize);

  // Add 3 blobs
  for (int i = 0; i < 3; ++i) {
    std::string blob_name = "blob_" + std::to_string(i);
    hipc::FullPtr<char> shm = ipc->AllocateBuffer(data.size());
    memcpy(shm.ptr_, data.data(), data.size());
    hipc::ShmPtr<> ref(shm.shm_);

    auto put = client->AsyncPutBlob(tag_task->tag_id_, blob_name, 0, data.size(), ref, 1.0f);
    put.Wait();
    REQUIRE(put->GetReturnCode() == 0);
    ipc->FreeBuffer(shm);
  }

  // Get contained blobs
  auto blobs_task = client->AsyncGetContainedBlobs(tag_task->tag_id_);
  blobs_task.Wait();

  REQUIRE(blobs_task->GetReturnCode() == 0);
  INFO("GetContainedBlobs returned " << blobs_task->blob_names_.size() << " blobs");
}

TEST_CASE("Runtime - GetContainedBlobs Empty Tag", "[runtime][blob]") {
  RuntimeCoverageFixture fixture;
  fixture.SetupTarget();

  auto *client = WRP_CTE_CLIENT;

  // Create empty tag
  auto tag_task = client->AsyncGetOrCreateTag("empty_tag");
  tag_task.Wait();
  REQUIRE(tag_task->GetReturnCode() == 0);

  // Get contained blobs (should be empty)
  auto blobs_task = client->AsyncGetContainedBlobs(tag_task->tag_id_);
  blobs_task.Wait();

  REQUIRE(blobs_task->GetReturnCode() == 0);
  INFO("GetContainedBlobs for empty tag returned "
       << blobs_task->blob_names_.size() << " blobs");
}

// ============================================================================
// ListTargets Runtime Tests
// ============================================================================

TEST_CASE("Runtime - ListTargets Success", "[runtime][target]") {
  RuntimeCoverageFixture fixture;
  fixture.SetupTarget();

  auto *client = WRP_CTE_CLIENT;

  auto list_task = client->AsyncListTargets();
  list_task.Wait();

  REQUIRE(list_task->GetReturnCode() == 0);
  REQUIRE(list_task->target_names_.size() > 0);
  INFO("ListTargets returned " << list_task->target_names_.size() << " targets");
}

// ============================================================================
// StatTargets Runtime Tests
// ============================================================================

TEST_CASE("Runtime - StatTargets Success", "[runtime][target]") {
  RuntimeCoverageFixture fixture;
  fixture.SetupTarget();

  auto *client = WRP_CTE_CLIENT;

  auto stat_task = client->AsyncStatTargets();
  stat_task.Wait();

  REQUIRE(stat_task->GetReturnCode() == 0);
  INFO("StatTargets completed successfully");
}

SIMPLE_TEST_MAIN()
